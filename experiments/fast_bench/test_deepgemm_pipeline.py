"""
End-to-end MoE pipeline probe using DeepGEMM as the grouped GEMM backend.

This is not intended as a contest-safe submission path. It measures how much
of DeepGEMM's isolated grouped-GEMM advantage survives once we keep the rest of
our pipeline structure (routing, dispatch, gather, SwiGLU, scatter) intact.
"""
import modal

app = modal.App("deepgemm-pipeline-probe")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
    )
    .run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM",
        "cd /opt/DeepGEMM && bash install.sh || echo 'install may have issues'",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


DEFAULT_UUIDS = "1a4c6ba1,5e8dc11c"


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def bench(uuids: str = DEFAULT_UUIDS, warmup: int = 2, iters: int = 8) -> str:
    import os
    import sys
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    import deep_gemm  # type: ignore
    import kernel as K  # type: ignore

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    def deepgemm_pipeline(*inputs):
        (
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
        ) = inputs

        ext = K._get_ext()  # noqa: SLF001
        T = int(routing_logits.shape[0])
        device = hidden_states.device
        ne = int(gemm1_weights.shape[0])
        N1 = int(gemm1_weights.shape[1])
        K1 = int(gemm1_weights.shape[2])
        N2 = int(gemm2_weights.shape[1])
        K2 = int(gemm2_weights.shape[2])
        ls = int(local_expert_offset)
        rsf = float(routed_scaling_factor)

        bufs = K._get_workspace(device, ne, T, N1, K1, N2, K2)  # noqa: SLF001
        topk_idx, assign_w = K._route(routing_logits, routing_bias, rsf, T, ls, ne)  # noqa: SLF001
        counts, sorted_tids, sorted_weights = K._dispatch_dynamic(topk_idx, assign_w, T, ls, ne, bufs)  # noqa: SLF001
        total_valid = int(sorted_tids.shape[0])

        hs_scale = hidden_states_scale
        if hs_scale.shape[0] != T:
            hs_scale = hs_scale.t().contiguous()

        packed_acts = bufs["packed_acts"][:total_valid]
        packed_act_scales = bufs["packed_act_scales"][:total_valid]
        ext.fused_gather_hidden_scales(
            hidden_states, hs_scale, sorted_tids, packed_acts, packed_act_scales
        )

        aligned_offsets = torch.empty(ne + 1, device=device, dtype=torch.int32)
        dg_acts_1 = torch.empty((total_valid + ne * alignment, K1), device=device, dtype=packed_acts.dtype)
        dg_scales_1 = torch.empty((total_valid + ne * alignment, packed_act_scales.shape[1]), device=device, dtype=packed_act_scales.dtype)
        dg_tids = torch.empty(total_valid + ne * alignment, device=device, dtype=sorted_tids.dtype)
        dg_weights = torch.empty(total_valid + ne * alignment, device=device, dtype=sorted_weights.dtype)
        grouped_layout = torch.empty(total_valid + ne * alignment, device=device, dtype=torch.int32)
        ext.repack_aligned_expert_layout(
            packed_acts,
            packed_act_scales,
            sorted_tids,
            sorted_weights,
            bufs["expert_offsets"],
            counts,
            int(alignment),
            aligned_offsets,
            dg_acts_1,
            dg_scales_1,
            dg_tids,
            dg_weights,
            grouped_layout,
        )
        total_aligned = int(aligned_offsets[ne].item())
        dg_acts_1 = dg_acts_1[:total_aligned]
        dg_scales_1 = dg_scales_1[:total_aligned]
        dg_tids = dg_tids[:total_aligned]
        dg_weights = dg_weights[:total_aligned]
        grouped_layout = grouped_layout[:total_aligned]

        gemm1_out = torch.empty((dg_acts_1.shape[0], N1), device=device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (dg_acts_1, dg_scales_1),
            (gemm1_weights, gemm1_weights_scale),
            gemm1_out,
            grouped_layout,
        )

        act_q = torch.empty((dg_acts_1.shape[0], N1 // 2), device=device, dtype=torch.float8_e4m3fn)
        row_scales = torch.empty(dg_acts_1.shape[0], device=device, dtype=torch.float32)
        act_scale_for_gemm2 = torch.empty((dg_acts_1.shape[0], K2 // 128), device=device, dtype=torch.float32)
        ext.swiglu_fp8_requant(gemm1_out, act_q, row_scales, act_scale_for_gemm2)

        gemm2_out = torch.empty((dg_acts_1.shape[0], N2), device=device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (act_q, act_scale_for_gemm2),
            (gemm2_weights, gemm2_weights_scale),
            gemm2_out,
            grouped_layout,
        )

        out = torch.empty((T, N2), device=device, dtype=torch.bfloat16)
        dg_token_counts = torch.empty(T, device=device, dtype=torch.int32)
        dg_token_offsets = torch.empty(T + 1, device=device, dtype=torch.int32)
        dg_token_perm = torch.empty(dg_tids.shape[0], device=device, dtype=torch.int32)
        ext.reduce_scatter(
            gemm2_out,
            dg_weights,
            dg_tids,
            out,
            dg_token_counts,
            dg_token_offsets,
            dg_token_perm,
            T,
        )
        return out

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE_PATH)
    definition = trace_set.definitions[DEF_NAME]

    selected_uuids = [u.strip() for u in uuids.split(",") if u.strip()]
    target_workloads = {}
    for wl in trace_set.workloads.get(DEF_NAME, []):
        wobj = getattr(wl, "workload", wl)
        uuid = getattr(wobj, "uuid", "")
        for sel in selected_uuids:
            if uuid.startswith(sel):
                target_workloads[sel] = wobj
                break

    out = []
    out.append(f"GPU: {torch.cuda.get_device_name()}")
    out.append(f"DeepGEMM contiguous alignment={alignment}")
    out.append(f"warmup={warmup}, iters={iters}")
    out.append("=" * 72)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for sel in selected_uuids:
        wobj = target_workloads[sel]
        loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()
        ) else {}
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        for _ in range(max(1, warmup)):
            result = deepgemm_pipeline(*inputs)
        torch.cuda.synchronize()

        start_ev.record()
        for _ in range(iters):
            result = deepgemm_pipeline(*inputs)
        end_ev.record()
        torch.cuda.synchronize()

        ms = start_ev.elapsed_time(end_ev) / iters
        nz = (result.abs().float().sum(dim=1) > 0).sum().item()
        out.append(f"{sel[:8]} T={T:>5}  {ms:7.3f} ms   nonzero={nz}/{result.shape[0]}")

    return "\n".join(out)


@app.local_entrypoint()
def main(uuids: str = DEFAULT_UUIDS, warmup: int = 2, iters: int = 8):
    print(bench.remote(uuids=uuids, warmup=warmup, iters=iters))
