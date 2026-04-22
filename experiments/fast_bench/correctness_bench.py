"""
Correctness check for the solution kernel against a trusted PyTorch reference
(FP8 dequant + BF16 matmul). Contest tolerance: atol=1.0, rtol=0.3,
required_matched_ratio=0.9.

Usage:  modal run experiments/fast_bench/correctness_bench.py
"""
import modal

app = modal.App("fused-moe-correctness")
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


DEFAULT_UUIDS = "e05c6c03,81955b1e,1a4c6ba1,58a34f27,5e8dc11c"


def _reference_impl(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    """Trusted PyTorch reference: FP8 dequant + BF16 matmul, full precision."""
    import torch
    import torch.nn.functional as F

    T = routing_logits.shape[0]
    E_GLOBAL = 256; N_GROUP = 8; GROUP_SIZE = 32; TOPK_GROUP = 4; TOP_K = 8
    E_LOCAL = int(gemm1_weights.shape[0])
    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])
    H = N1 // 2
    ls = int(local_expert_offset)
    rsf = float(routed_scaling_factor)

    # --- Routing ---
    s = torch.sigmoid(routing_logits.float())
    s_wb = s + routing_bias.float()
    s_wb_g = s_wb.view(T, N_GROUP, GROUP_SIZE)
    group_top2 = torch.topk(s_wb_g, k=2, dim=2).values
    group_scores = group_top2.sum(dim=2)
    valid_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1).indices
    group_mask = torch.zeros((T, N_GROUP), device=s.device, dtype=torch.bool)
    group_mask.scatter_(1, valid_groups, True)
    valid_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(T, E_GLOBAL)
    filtered = s_wb.masked_fill(~valid_mask, float("-inf"))
    _, topk_idx = torch.topk(filtered, k=TOP_K, dim=1)
    topk_s = s.gather(1, topk_idx.long())
    sum_s = topk_s.sum(dim=1, keepdim=True)
    assign_w = topk_s * (rsf / (sum_s + 1e-20))

    # Dequantize FP8 inputs
    def dq(x, sc):
        # x: fp8 [*, K], sc: fp32 [*, K/128] (per-K-block scales)
        # replicate scale across 128 K-lane block
        K_ = x.shape[-1]
        return x.float() * sc.unsqueeze(-1).expand(*sc.shape, 128)[..., :K_].reshape(*x.shape)

    hs_bf = dq(hidden_states, hidden_states_scale) if hidden_states_scale.shape[0] == T \
        else dq(hidden_states, hidden_states_scale.t())

    # Per-expert GEMMs
    out = torch.zeros(T, N2, device=hidden_states.device, dtype=torch.bfloat16)
    for e_local in range(E_LOCAL):
        e_global = ls + e_local
        mask = (topk_idx == e_global).any(dim=1)  # [T]
        if not mask.any():
            continue
        tok_ids = mask.nonzero(as_tuple=True)[0]
        a = hs_bf[tok_ids]  # [M, K1]

        # GEMM1 weights: dequant [N1, K1]
        w1 = gemm1_weights[e_local].float()  # fp8 -> float
        # weight scale shape is [N1/128, K1/128] per expert
        w1_sc = gemm1_weights_scale[e_local].float()
        # Per-block scaling: [N1/128, K1/128] -> broadcast to [N1, K1]
        w1_dq = w1 * w1_sc.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        gemm1_out = a @ w1_dq.T  # [M, N1]

        # SwiGLU
        g, u = gemm1_out.chunk(2, dim=-1)
        act = g * (u * torch.sigmoid(u))

        # Per-token quantize for GEMM2 inputs (fp8 with per-row scale)
        row_abs = act.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (row_abs / 448.0).clamp_min(1e-8)
        act_q = (act / scale).to(torch.float8_e4m3fn)

        # GEMM2 weights
        w2 = gemm2_weights[e_local].float()
        w2_sc = gemm2_weights_scale[e_local].float()
        w2_dq = w2 * w2_sc.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        gemm2_out = (act_q.float() * scale) @ w2_dq.T  # [M, N2]

        # Weighted scatter per-token
        for j, t in enumerate(tok_ids.tolist()):
            # Find k where topk_idx[t,k] == e_global
            k_match = (topk_idx[t] == e_global).nonzero(as_tuple=True)[0][0].item()
            w = assign_w[t, k_match].item()
            out[t] += (gemm2_out[j] * w).to(torch.bfloat16)

    return out


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/mnt": trace_volume})
def check(uuids: str = DEFAULT_UUIDS) -> str:
    import sys, os
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    import kernel as K
    _ = K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE_PATH)
    definition = trace_set.definitions[DEF_NAME]

    out = []
    out.append(f"Contest tolerance: atol=1, rtol=0.3, required_matched_ratio=0.9")
    out.append("=" * 72)
    for sel in uuids.split(","):
        sel = sel.strip()
        if not sel:
            continue
        wobj = None
        for wl in trace_set.workloads.get(DEF_NAME, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(sel):
                wobj = w
                break
        if wobj is None:
            out.append(f"{sel}: NOT FOUND")
            continue

        loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        # Only run small workloads through the Python reference (it's slow).
        if T > 128:
            out.append(f"{sel[:8]} T={T:>5}: skip (reference too slow for T>128)")
            continue

        our = K.custom_kernel(*inputs)
        ref = _reference_impl(*inputs).to(our.dtype)

        # Contest metric
        abs_diff = (our.float() - ref.float()).abs()
        tol = 1.0 + 0.3 * ref.abs().float()
        matched = (abs_diff <= tol).float().mean().item()
        max_err = abs_diff.max().item()
        rel_err = (abs_diff / (ref.abs().float() + 1e-6)).median().item()
        passed = "PASS" if matched >= 0.9 else "FAIL"
        out.append(
            f"{sel[:8]} T={T:>5}  matched={matched*100:5.1f}%  max_abs={max_err:6.3f}  "
            f"med_rel={rel_err:.3f}  [{passed}]"
        )

    return "\n".join(out)


@app.local_entrypoint()
def main(uuids: str = DEFAULT_UUIDS):
    print(check.remote(uuids=uuids))
