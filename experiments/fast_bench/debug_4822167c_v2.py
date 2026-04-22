"""
Follow-up probe on 4822167c.

v1 showed CUTLASS matches the manual fp32 dequant+matmul reference at 98.12%
while the harness reports CUTLASS failing. The harness's reference must be
using a different numerical path. This probe:
  - inspects the definition.reference source
  - runs the reference the SAME way the harness does (via the registry)
  - compares CUTLASS, Triton, and manual-fp32 to the harness reference
"""

from pathlib import Path
import modal

app = modal.App("debug-4822167c-v2")

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
    import importlib.util
    import inspect
    import sys
    import torch
    from pathlib import Path

    out = []
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    TRACE = "/mnt/mlsys26-contest"
    DEF = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE)
    definition = trace_set.definitions[DEF]

    out.append("=== DEFINITION.reference ===")
    try:
        ref_obj = definition.reference
        out.append(f"type={type(ref_obj).__name__}")
        for attr in ("language", "entry_point", "author", "name"):
            v = getattr(ref_obj, attr, None)
            out.append(f"  {attr}={v}")
        if hasattr(ref_obj, "sources"):
            for src in ref_obj.sources:
                out.append(f"  --- source path={src.path} ---")
                out.append(src.content[:4000])
                if len(src.content) > 4000:
                    out.append(f"  ... truncated ({len(src.content)} chars total)")
    except Exception as e:
        out.append(f"reference introspection error: {e!r}")

    wobj = None
    for wl in trace_set.workloads.get(DEF, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith("4822167c"):
            wobj = w
            break
    if wobj is None:
        return "\n".join(out) + "\nworkload 4822167c not found"

    loaded_st = load_safetensors(definition, wobj, Path(TRACE)) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()
    ) else {}

    def fresh_inputs():
        return gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)

    inputs_ours = fresh_inputs()
    inputs_ref  = fresh_inputs()
    inputs_triton = fresh_inputs()

    out.append("")
    out.append("=== Reference runnable build ===")
    registry = BuilderRegistry.get_instance()

    # Try to build reference directly. BuilderRegistry.build takes (definition, solution).
    # Reference is a Solution-like object. Most installs also expose a helper.
    try:
        # Try explicit API
        from flashinfer_bench.bench.reference import build_reference as _build_ref  # type: ignore
        ref_runnable = _build_ref(definition)
    except Exception as e:
        out.append(f"build_reference helper not available: {e!r}")
        try:
            ref_runnable = registry.build(definition, definition.reference)
            out.append("used registry.build(definition, definition.reference)")
        except Exception as e2:
            out.append(f"registry.build failed: {e2!r}")
            ref_runnable = None

    def run_one(runnable, inputs, name, warmup=2):
        try:
            for _ in range(warmup):
                _ = runnable(*inputs)
            torch.cuda.synchronize()
            y = runnable(*inputs).clone()
            torch.cuda.synchronize()
            return y
        except Exception as e:
            out.append(f"[{name}] run error: {e!r}")
            return None

    harness_ref_out = None
    if ref_runnable is not None:
        harness_ref_out = run_one(ref_runnable, inputs_ref, "harness_reference", warmup=1)

    # ---- Build CUTLASS (our current solution) ----
    sys.path.insert(0, "/root/solution_python")
    spec = importlib.util.spec_from_file_location("cutlass_kernel", "/root/solution_python/kernel.py")
    cutlass_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cutlass_mod)
    ours_out = run_one(cutlass_mod.run, inputs_ours, "cutlass", warmup=2)

    spec2 = importlib.util.spec_from_file_location("triton_kernel", "/root/solution_triton/kernel.py")
    triton_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(triton_mod)
    triton_out = run_one(triton_mod.run, inputs_triton, "triton", warmup=2)

    # ---- Manual fp32 dequant+matmul reference (mirrors what I had in v1) ----
    (
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor,
    ) = inputs_triton  # reuse (dispatched tensors are not mutated by kernels)
    T = int(routing_logits.shape[0])
    ne = int(gemm1_weights.shape[0])
    N2, K2 = int(gemm2_weights.shape[1]), int(gemm2_weights.shape[2])
    ls = int(local_expert_offset) if torch.is_tensor(local_expert_offset) else int(local_expert_offset)
    rsf = float(routed_scaling_factor) if torch.is_tensor(routed_scaling_factor) else float(routed_scaling_factor)

    E_GLOBAL = 256; N_GROUP = 8; TOPK_GROUP = 4
    GROUP_SIZE = E_GLOBAL // N_GROUP; TOP_K = 8
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

    def dq(x, sc):
        K_ = x.shape[-1]
        return x.float() * sc.unsqueeze(-1).expand(*sc.shape, 128)[..., :K_].reshape(*x.shape)
    hs_bf = dq(hidden_states, hidden_states_scale) if hidden_states_scale.shape[0] == T \
        else dq(hidden_states, hidden_states_scale.t())

    manual_ref = torch.zeros(T, N2, device="cuda", dtype=torch.bfloat16)
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
            manual_ref[t] += (g2[j] * w).to(torch.bfloat16)

    def compare(name_a, a, name_b, b):
        if a is None or b is None:
            out.append(f"skip compare {name_a} vs {name_b}: None tensor")
            return
        if a.shape != b.shape:
            out.append(f"[{name_a} vs {name_b}] shape mismatch {a.shape} vs {b.shape}")
            return
        diff = (a.float() - b.float()).abs()
        tol = 1.0 + 0.3 * b.abs().float()
        matched = (diff <= tol).float()
        out.append(
            f"[{name_a}] vs [{name_b}]: match={matched.mean().item()*100:.2f}%  "
            f"max_abs={diff.max().item():.3e}  "
            f"median_diff={diff.median().item():.3e}"
        )

    out.append("")
    out.append("=== comparisons ===")
    if harness_ref_out is not None:
        compare("cutlass",     ours_out,   "harness_ref", harness_ref_out)
        compare("triton",      triton_out, "harness_ref", harness_ref_out)
        compare("manual_fp32", manual_ref, "harness_ref", harness_ref_out)
    compare("cutlass", ours_out,    "manual_fp32", manual_ref)
    compare("triton",  triton_out,  "manual_fp32", manual_ref)
    compare("cutlass", ours_out,    "triton",      triton_out)

    # Magnitude of outputs, so we know if any kernel is producing overflowed garbage
    for name, y in [("cutlass", ours_out), ("triton", triton_out),
                    ("manual_fp32", manual_ref), ("harness_ref", harness_ref_out)]:
        if y is None:
            continue
        yf = y.float()
        out.append(
            f"[{name}]  min={yf.min().item():.3e}  max={yf.max().item():.3e}  "
            f"mean_abs={yf.abs().mean().item():.3e}  has_nan={torch.isnan(yf).any().item()}  has_inf={torch.isinf(yf).any().item()}"
        )

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
