"""
Deep-dive probe on the failing workload 4822167c (seq_len=56).

On Modal B200, load the workload inputs, run:
  - our current kernel (the CUTLASS solution/python/kernel.py)
  - a trusted PyTorch fp32-dequant reference

Then report:
  - per-local-expert token counts (distribution)
  - hidden_state and weight-scale magnitude stats
  - where the match-ratio is failing (worst output rows/cols)
  - errors stratified by output-token's primary expert
  - also compare against our Triton submission (solution/triton/kernel.py)

This is diagnosis only, no changes to the active solution.
"""

from pathlib import Path

import modal

app = modal.App("debug-4822167c")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution_python",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/triton",
        remote_path="/root/solution_triton",
    )
)


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def probe() -> str:
    import os
    import sys
    import torch
    from pathlib import Path

    out = []
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    TRACE = "/mnt/mlsys26-contest"
    DEF = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE)
    definition = trace_set.definitions[DEF]

    wobj = None
    for wl in trace_set.workloads.get(DEF, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith("4822167c"):
            wobj = w
            break
    if wobj is None:
        return "workload 4822167c not found"

    loaded_st = load_safetensors(definition, wobj, Path(TRACE)) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()
    ) else {}
    inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)

    (
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
    ) = inputs

    T = int(routing_logits.shape[0])
    ne = int(gemm1_weights.shape[0])
    N1, K1 = int(gemm1_weights.shape[1]), int(gemm1_weights.shape[2])
    N2, K2 = int(gemm2_weights.shape[1]), int(gemm2_weights.shape[2])
    H = N1 // 2
    ls = int(local_expert_offset) if torch.is_tensor(local_expert_offset) else int(local_expert_offset)
    rsf = float(routed_scaling_factor) if torch.is_tensor(routed_scaling_factor) else float(routed_scaling_factor)

    out.append(f"Workload uuid={wobj.uuid}")
    out.append(f"T={T}, E_local={ne}, local_start={ls}, rsf={rsf}")
    out.append(f"hidden_states shape={hidden_states.shape} dtype={hidden_states.dtype}")
    out.append(f"hidden_states_scale shape={hidden_states_scale.shape} dtype={hidden_states_scale.dtype}")
    out.append(f"gemm1_weights shape={gemm1_weights.shape} dtype={gemm1_weights.dtype}")
    out.append(f"gemm2_weights shape={gemm2_weights.shape} dtype={gemm2_weights.dtype}")

    # ---- Routing computation (mirrors reference semantics) ----
    E_GLOBAL = 256
    N_GROUP = 8
    TOPK_GROUP = 4
    GROUP_SIZE = E_GLOBAL // N_GROUP
    TOP_K = 8

    s = torch.sigmoid(routing_logits.float())
    s_wb = s + routing_bias.float()
    s_wb_g = s_wb.view(T, N_GROUP, GROUP_SIZE)
    group_top2 = torch.topk(s_wb_g, k=2, dim=2).values
    group_scores = group_top2.sum(dim=2)
    valid_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1).indices
    group_mask = torch.zeros((T, N_GROUP), device="cuda", dtype=torch.bool)
    group_mask.scatter_(1, valid_groups, True)
    valid_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(T, E_GLOBAL)
    filtered = s_wb.masked_fill(~valid_mask, float("-inf"))
    _, topk_idx = torch.topk(filtered, k=TOP_K, dim=1)
    topk_s = s.gather(1, topk_idx.long())
    sum_s = topk_s.sum(dim=1, keepdim=True)
    assign_w = topk_s * (rsf / (sum_s + 1e-20))

    flat_idx = topk_idx.reshape(-1).int()
    local_mask = (flat_idx >= ls) & (flat_idx < ls + ne)
    local_expert = (flat_idx[local_mask] - ls).long()
    counts = torch.bincount(local_expert, minlength=ne).int().cpu().tolist()

    out.append("")
    out.append("Per-local-expert token count (non-zero entries only):")
    nz_counts = [(i, c) for i, c in enumerate(counts) if c > 0]
    for i, c in nz_counts:
        out.append(f"  expert[{i:2d}] count={c}")
    total_valid = int(local_mask.sum().item())
    out.append(f"  total valid (local) assignments = {total_valid}")
    out.append(f"  experts with count=0: {sum(1 for c in counts if c == 0)}/{ne}")
    out.append(f"  experts with count=1: {sum(1 for c in counts if c == 1)}")
    out.append(f"  experts with count>=2: {sum(1 for c in counts if c >= 2)}")
    out.append(f"  max count: {max(counts)}  min (nonzero): {min([c for c in counts if c>0]) if nz_counts else 0}")

    # ---- Input tensor magnitude stats ----
    out.append("")
    hs_f32 = hidden_states.float()
    hss_f32 = hidden_states_scale.float()
    out.append(f"hidden_states: min={hs_f32.min().item():.3e} max={hs_f32.max().item():.3e} mean_abs={hs_f32.abs().mean().item():.3e}")
    out.append(f"hidden_states_scale: min={hss_f32.min().item():.3e} max={hss_f32.max().item():.3e} mean={hss_f32.mean().item():.3e}")
    out.append(f"  hidden_states has NaN: {torch.isnan(hs_f32).any().item()}, Inf: {torch.isinf(hs_f32).any().item()}")
    out.append(f"  hidden_states_scale has NaN: {torch.isnan(hss_f32).any().item()}, Inf: {torch.isinf(hss_f32).any().item()}")
    g1w = gemm1_weights.float()
    g1ws = gemm1_weights_scale.float()
    out.append(f"gemm1_weights: min={g1w.min().item():.3e} max={g1w.max().item():.3e} mean_abs={g1w.abs().mean().item():.3e}")
    out.append(f"gemm1_weights_scale: min={g1ws.min().item():.3e} max={g1ws.max().item():.3e} mean={g1ws.mean().item():.3e}")
    g2w = gemm2_weights.float()
    g2ws = gemm2_weights_scale.float()
    out.append(f"gemm2_weights: min={g2w.min().item():.3e} max={g2w.max().item():.3e} mean_abs={g2w.abs().mean().item():.3e}")
    out.append(f"gemm2_weights_scale: min={g2ws.min().item():.3e} max={g2ws.max().item():.3e} mean={g2ws.mean().item():.3e}")

    # ---- Load both kernels ----
    sys.path.insert(0, "/root/solution_python")
    import importlib, importlib.util

    spec = importlib.util.spec_from_file_location("cutlass_kernel", "/root/solution_python/kernel.py")
    cutlass_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cutlass_mod)

    spec2 = importlib.util.spec_from_file_location("triton_kernel", "/root/solution_triton/kernel.py")
    triton_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(triton_mod)

    # ---- Run our CUTLASS kernel ----
    for _ in range(2):
        _ = cutlass_mod.run(*inputs)
    torch.cuda.synchronize()
    ours_cutlass = cutlass_mod.run(*inputs).clone()

    # ---- Run Triton submission kernel ----
    for _ in range(2):
        _ = triton_mod.run(*inputs)
    torch.cuda.synchronize()
    ours_triton = triton_mod.run(*inputs).clone()

    # ---- Trusted PyTorch fp32 reference ----
    def dq(x, sc):
        K_ = x.shape[-1]
        return x.float() * sc.unsqueeze(-1).expand(*sc.shape, 128)[..., :K_].reshape(*x.shape)

    hs_bf = dq(hidden_states, hidden_states_scale) if hidden_states_scale.shape[0] == T \
        else dq(hidden_states, hidden_states_scale.t())

    ref = torch.zeros(T, N2, device="cuda", dtype=torch.bfloat16)
    for e_local in range(ne):
        e_global = ls + e_local
        mask = (topk_idx == e_global).any(dim=1)
        if not mask.any():
            continue
        tok_ids = mask.nonzero(as_tuple=True)[0]
        a = hs_bf[tok_ids]
        w1 = gemm1_weights[e_local].float()
        w1_sc = gemm1_weights_scale[e_local].float()
        w1_dq = w1 * w1_sc.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        g1 = a @ w1_dq.T
        gg, uu = g1.chunk(2, dim=-1)
        act = gg * (uu * torch.sigmoid(uu))
        row_abs = act.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale_q = (row_abs / 448.0).clamp_min(1e-8)
        act_q = (act / scale_q).to(torch.float8_e4m3fn)
        w2 = gemm2_weights[e_local].float()
        w2_sc = gemm2_weights_scale[e_local].float()
        w2_dq = w2 * w2_sc.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        g2 = (act_q.float() * scale_q) @ w2_dq.T
        for j, t in enumerate(tok_ids.tolist()):
            k_match = (topk_idx[t] == e_global).nonzero(as_tuple=True)[0][0].item()
            w = assign_w[t, k_match].item()
            ref[t] += (g2[j] * w).to(torch.bfloat16)

    # ---- Compare both vs ref ----
    def compare(name, ours, ref):
        diff = (ours.float() - ref.float()).abs()
        tol = 1.0 + 0.3 * ref.abs().float()
        matched = (diff <= tol).float()
        match_ratio = matched.mean().item()
        rows_matched = matched.mean(dim=1)
        bad_rows = (rows_matched < 0.9).nonzero(as_tuple=True)[0].tolist()
        max_abs = diff.max().item()
        out.append(f"")
        out.append(f"[{name}] vs fp32-ref:  match_ratio={match_ratio*100:.2f}% (need >=90%)"
                   f"  max_abs={max_abs:.3e}")
        out.append(f"  rows with row-match < 90%: {len(bad_rows)}/{T}")
        if bad_rows:
            show = bad_rows[:10]
            out.append(f"  first bad rows: {show}")
            for t in show[:5]:
                primary = topk_idx[t].tolist()
                local_primary = [p - ls for p in primary if ls <= p < ls + ne]
                non_local = [p for p in primary if not (ls <= p < ls + ne)]
                counts_for_t = [counts[p] for p in local_primary]
                out.append(
                    f"    row {t}: topk_idx={primary}  local_primary={local_primary}"
                    f"  non_local={non_local}  cnts_for_local_primary={counts_for_t}"
                    f"  row_match={rows_matched[t].item()*100:.1f}%"
                    f"  row_max_abs={diff[t].max().item():.3e}"
                )
        return match_ratio, max_abs

    compare("CUTLASS", ours_cutlass, ref)
    compare("Triton", ours_triton, ref)

    # Compare CUTLASS vs Triton directly
    dd = (ours_cutlass.float() - ours_triton.float()).abs()
    tol_tt = 1.0 + 0.3 * ours_triton.abs().float()
    m = (dd <= tol_tt).float().mean().item()
    out.append("")
    out.append(f"[CUTLASS vs Triton submission]: match_ratio={m*100:.2f}% max_abs={dd.max().item():.3e}")
    row_m = (dd <= tol_tt).float().mean(dim=1)
    bad = (row_m < 0.9).nonzero(as_tuple=True)[0].tolist()
    out.append(f"  rows disagreeing >10%: {len(bad)}/{T}")
    if bad:
        for t in bad[:10]:
            primary = topk_idx[t].tolist()
            local_primary = [p - ls for p in primary if ls <= p < ls + ne]
            counts_for_t = [counts[p] for p in local_primary]
            out.append(
                f"    row {t}: topk_idx={primary}  local_primary={local_primary}"
                f"  cnts_for_local_primary={counts_for_t}"
                f"  row_match={row_m[t].item()*100:.1f}%"
            )

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
