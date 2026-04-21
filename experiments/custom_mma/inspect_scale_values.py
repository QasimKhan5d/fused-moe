"""Inspect actual contest scale values to rule out encoding misinterpretation."""
import modal
app = modal.App("scale-inspect")
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


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/mnt": trace_volume})
def run() -> str:
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

    # Pick one workload
    wobj = None
    for wl in trace_set.workloads.get(def_name, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith("5e8dc11c"):
            wobj = w; break
    assert wobj is not None

    loaded_st = load_safetensors(
        definition, wobj, Path("/mnt/mlsys26-contest")
    ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
    inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))

    out = []
    out.append(f"Workload: 5e8dc11c, T={inputs[0].shape[0]}")
    out.append("")

    # Print first 20 values of each scale tensor.
    names = {
        3: "hs_scale",
        5: "gemm1_w_scale",
        7: "gemm2_w_scale",
    }
    for idx, name in names.items():
        x = inputs[idx]
        out.append(f"=== {name} (inputs[{idx}]) shape={tuple(x.shape)} dtype={x.dtype} ===")
        flat = x.flatten()
        first20 = flat[:20].cpu().tolist()
        out.append(f"  first 20 values: {['%.6g' % v for v in first20]}")
        # Sorted-by-abs sample
        abs_sorted = flat.abs().sort().values
        out.append(f"  |x| percentiles: p0={abs_sorted[0].item():.3e} "
                   f"p1={abs_sorted[int(abs_sorted.numel()*0.01)].item():.3e} "
                   f"p50={abs_sorted[int(abs_sorted.numel()*0.50)].item():.3e} "
                   f"p99={abs_sorted[int(abs_sorted.numel()*0.99)].item():.3e} "
                   f"p100={abs_sorted[-1].item():.3e}")
        # Check how many are +/- 1e-10 (near zero)
        tiny_count = (flat.abs() < 1e-10).sum().item()
        out.append(f"  near-zero (abs < 1e-10): {tiny_count} / {flat.numel()}")
        # Sign distribution
        pos = (flat > 0).sum().item()
        neg = (flat < 0).sum().item()
        zero = (flat == 0).sum().item()
        out.append(f"  sign: pos={pos} neg={neg} zero={zero}")
        out.append("")

    # Now check a round-trip using the reference implementation: does the kernel
    # produce the SAME output whether scales are used as-is (possibly negative) or
    # coerced positive? That would tell us if contest treats negative as |value|.
    out_ref = K.custom_kernel(*inputs).float()
    saved = [inputs[3].clone(), inputs[5].clone(), inputs[7].clone()]
    try:
        inputs[3].copy_(saved[0].abs())
        inputs[5].copy_(saved[1].abs())
        inputs[7].copy_(saved[2].abs())
        out_abs = K.custom_kernel(*inputs).float()
        diff = (out_abs - out_ref).abs()
        tol = 1.0 + 0.3 * out_ref.abs()
        match_abs = (diff <= tol).float().mean().item()
        out.append(f"Round-trip: using abs(scales) vs original: match={match_abs*100:.2f}% "
                   f"max_abs_diff={diff.max().item():.1f}")
    finally:
        inputs[3].copy_(saved[0])
        inputs[5].copy_(saved[1])
        inputs[7].copy_(saved[2])

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(run.remote())
