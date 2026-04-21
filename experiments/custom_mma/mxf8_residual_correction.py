"""Test per-row residual mean correction.

Theory: if we quantize fp32 scale to ue8m0 (pow-of-2), the residual r = fp32/ue8m0
varies per block. If we can approximate the per-output-row contribution of
`r_a_k * r_b_k` across k by a single per-row factor, we can apply it as a cheap
post-GEMM multiply and recover most of the precision.

Test variants:
  A) "scale_only_ue8m0" — baseline: scale quantized, no correction, no residual.
  B) "r_geom_row_correction" — correct output by sqrt(r_a_rowmean) * sqrt(r_b_rowmean).
  C) "r_row_correction_A_only" — correct by r_a_rowmean (activations only).
  D) "r_row_correction_B_only" — correct by r_b_colmean (weights only).
  E) "r_perblock_weighted_sum" — analytic per-block correction (expensive, oracle).

Also characterize:
  - For each tensor, stddev of log2(r) across K (how uniform is r along K?).
"""
import modal
app = modal.App("mxf8-residual-correction")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run(uuids: str = "e05c6c03,1a4c6ba1,5e8dc11c") -> str:
    import os, sys
    from pathlib import Path
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    def q_ceil_pow2(x):
        x = x.clamp_min(1e-30)
        exp = torch.ceil(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def residual_stats(scale_fp32: torch.Tensor, label: str) -> str:
        """Compute stats about residual uniformity along K."""
        scale_hw = q_ceil_pow2(scale_fp32)
        r = scale_fp32 / scale_hw
        log_r = torch.log2(r)
        # K axis is last for weights; for hidden scale may be transposed.
        # We want std along the K axis.
        K_axis = -1
        mean_log = log_r.mean(dim=K_axis, keepdim=True)
        std_log = log_r.std(dim=K_axis, keepdim=True)
        mean_r = r.mean(dim=K_axis)
        max_r = r.max().item()
        min_r = r.min().item()
        return (f"{label}: r range=[{min_r:.3f}, {max_r:.3f}] "
                f"std_log2_along_K={std_log.mean().item():.4f} "
                f"mean_r={mean_r.mean().item():.3f}")

    def run_one(u: str):
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            return [f"{u}: not found"]

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = int(inputs[0].shape[0])

        lines = []
        lines.append(f"=== {u[:8]} T={T} ===")
        # hidden: [T, K], scale [K/128, T] (transposed)
        hs_scale = inputs[3]
        # Transpose to [T, K/128] for easier K-axis stats.
        if hs_scale.dim() == 2 and hs_scale.shape[0] * 128 >= inputs[2].shape[1]:
            hs_scale_tk = hs_scale.transpose(0, 1).contiguous()
        else:
            hs_scale_tk = hs_scale
        lines.append("  " + residual_stats(hs_scale_tk, "hs_scale[T,K/128]"))
        lines.append("  " + residual_stats(inputs[5], "gemm1_weights_scale[E,N/128,K/128]"))
        lines.append("  " + residual_stats(inputs[7], "gemm2_weights_scale[E,N/128,K/128]"))

        # Reference output
        out_ref = K.custom_kernel(*inputs).float()

        # Variant A: quantize gemm2_weights_scale to ue8m0 only (scale-only).
        saved = inputs[7].clone()
        try:
            inputs[7].copy_(q_ceil_pow2(saved))
            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            matchA = (diff <= tol).float().mean().item()
            lines.append(f"  A) gemm2 scale-only ue8m0:         match={matchA*100:.2f}% "
                         f"maxabs={diff.max().item():.1f}")
        finally:
            inputs[7].copy_(saved)

        # Variant B: quantize gemm2 scales AND apply per-output-row r_mean correction.
        saved = inputs[7].clone()
        try:
            s_fp32 = saved  # [E, N/128, K/128]
            s_hw = q_ceil_pow2(s_fp32)
            r = s_fp32 / s_hw  # [E, N/128, K/128]
            r_mean = r.mean(dim=-1)  # [E, N/128]
            # Expand r_mean to [E, N]
            r_mean_full = r_mean.repeat_interleave(128, dim=-1)  # [E, N]
            inputs[7].copy_(s_hw)
            out_q = K.custom_kernel(*inputs).float()
            # Apply correction: out[m, n] *= r_mean[expert_of_token[m], n]
            # But out is [T, N2] where the expert mapping was lost in combine.
            # For correction to apply at the combine level, we'd need to apply
            # r_mean BEFORE combine, per-token-per-expert.
            # Approximation: use MEAN r_mean across experts (per output-col only).
            r_global_per_col = r_mean.mean(dim=0)  # [N/128]
            r_global_col_full = r_global_per_col.repeat_interleave(128)  # [N]
            out_corr = (out_q * r_global_col_full.unsqueeze(0).to(out_q.dtype)).float()
            diff = (out_corr - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            matchB = (diff <= tol).float().mean().item()
            lines.append(f"  B) gemm2 ue8m0 + global-col r-mean: match={matchB*100:.2f}% "
                         f"maxabs={diff.max().item():.1f}")
        finally:
            inputs[7].copy_(saved)

        # Variant C: exact oracle — apply per-expert-per-N r_mean (knows which expert each output came from).
        # This requires knowing the routing, which is topk_idx. The output out_bf16 is already
        # combined across experts weighted by assign_w. The per-expert contribution isn't separable
        # after combine. So this variant isn't directly applicable without recomputing the pipeline.
        # We can however test variant A with per-128-N-block multiplier, which is separable:
        # out[T, N] = combine(T' * gemm2_out[m, N]); gemm2_out[m, N] = GEMM(act_q[m], W_gemm2[e, N, K]).
        # The gemm2's per-row-n scale affects all tokens routed through that expert identically.
        # If we knew the expert-to-output-token mapping, we could apply r_mean[e, n] per output.
        # Since outputs combine multiple experts per token, we'd need to weight.
        # Skip oracle for now.

        # Variant D: compute expected error bound — assume residuals are lognormal with std_log
        # along K. Error bound ~ sum_k r_k / K ≈ 1 + 0.5*var. If var small, correction is easy.
        return lines

    all_lines = []
    for u in uuids.split(","):
        all_lines.extend(run_one(u.strip()))
    return "\n".join(all_lines)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,6230e838,1a4c6ba1,5e8dc11c"):
    print(run.remote(uuids=uuids))
