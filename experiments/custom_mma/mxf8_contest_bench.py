"""Full contest-workload sweep comparing MxF8 vs baseline (FP32-blockwise).

For each of the 19 workloads:
  1. Run baseline path (USE_MXF8=0) — captures baseline output + latency.
  2. Run MxF8 path (USE_MXF8=1) — captures MxF8 output + latency.
  3. Compute contest match ratio between MxF8 and baseline.
  4. Report speedup.

Usage:
    modal run experiments/custom_mma/mxf8_contest_bench.py
"""
import modal

app = modal.App("mxf8-contest-bench")
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

DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run(warmup: int = 3, iters: int = 20, mxf8: bool = True) -> str:
    import os, sys, time
    from pathlib import Path
    import torch

    os.environ["USE_MXF8"] = "1" if mxf8 else "0"
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    _ = K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions[DEF_NAME]

    all_wls = [getattr(wl, "workload", wl)
               for wl in trace_set.workloads.get(DEF_NAME, [])]
    all_wls.sort(key=lambda w: w.axes.get("seq_len", 0))

    out = []
    out.append(f"mxf8={mxf8} warmup={warmup} iters={iters}")
    out.append("uuid      T      lat_ms   nonzero       status")
    out.append("-" * 72)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    results_by_uuid = {}
    for w in all_wls:
        u = getattr(w, "uuid", "")[:8]
        loaded_st = load_safetensors(definition, w, Path("/mnt/mlsys26-contest")) if any(
            d.type == "safetensors" for d in getattr(w, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, w, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        try:
            for _ in range(max(1, warmup)):
                out_t = K.custom_kernel(*inputs)
            torch.cuda.synchronize()
        except Exception as exc:
            results_by_uuid[u] = {
                "T": T, "lat_ms": None,
                "out_cpu": None, "nonzero": 0,
                "error": f"WARMUP FAIL: {type(exc).__name__}: {str(exc)[:120]}",
            }
            out.append(f"{u:8} T={T:>5}  WARMUP FAIL: {type(exc).__name__}: {str(exc)[:120]}")
            continue

        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(iters):
            out_t = K.custom_kernel(*inputs)
        end_ev.record()
        torch.cuda.synchronize()
        lat_ms = start_ev.elapsed_time(end_ev) / iters

        nz = int((out_t.abs().float().sum(dim=1) > 0).sum().item())
        out_cpu = out_t.cpu().float()
        results_by_uuid[u] = {
            "T": T, "lat_ms": lat_ms, "out_cpu": out_cpu,
            "nonzero": nz, "error": None,
        }
        out.append(f"{u:8} T={T:>5}  {lat_ms:7.3f}  {nz:>5}/{out_t.shape[0]:<5}")

    # Serialize per-uuid results to disk so caller can load + compare.
    save_path = f"/tmp/mxf8_bench_{'on' if mxf8 else 'off'}.pt"
    torch.save(results_by_uuid, save_path)
    return "\n".join(out) + f"\nSaved: {save_path}"


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run_both() -> str:
    """Run both baseline and MxF8, compare, and report speedups + correctness."""
    import os, sys, time
    from pathlib import Path
    import torch

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions[DEF_NAME]

    all_wls = [getattr(wl, "workload", wl)
               for wl in trace_set.workloads.get(DEF_NAME, [])]
    all_wls.sort(key=lambda w: w.axes.get("seq_len", 0))

    # Preload all inputs once (big tensors but avoids reload).
    wl_inputs = []
    for w in all_wls:
        u = getattr(w, "uuid", "")[:8]
        loaded_st = load_safetensors(definition, w, Path("/mnt/mlsys26-contest")) if any(
            d.type == "safetensors" for d in getattr(w, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, w, device="cuda", safe_tensors=loaded_st)
        wl_inputs.append((u, inputs))

    def bench_one(kmod, inputs, warmup=3, iters=20):
        T = int(inputs[0].shape[0])
        try:
            for _ in range(max(1, warmup)):
                out_t = kmod.custom_kernel(*inputs)
            torch.cuda.synchronize()
        except Exception as exc:
            return None, None, f"FAIL: {type(exc).__name__}: {str(exc)[:200]}"
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            out_t = kmod.custom_kernel(*inputs)
        e.record()
        torch.cuda.synchronize()
        return out_t, s.elapsed_time(e) / iters, None

    # ==== Phase 1: baseline (USE_MXF8=0) ====
    os.environ["USE_MXF8"] = "0"
    os.environ["MXF8_MIN_T"] = "4096"
    for k in list(sys.modules):
        if k == "kernel" or k.startswith("kernel."):
            del sys.modules[k]
    import kernel as K
    _ = K._get_ext()

    baseline = {}
    for u, inputs in wl_inputs:
        out_t, lat, err = bench_one(K, inputs)
        if err:
            baseline[u] = (None, None, err, int(inputs[0].shape[0]))
        else:
            baseline[u] = (out_t.cpu().float(), lat, None, int(inputs[0].shape[0]))

    # ==== Phase 2: MxF8 (USE_MXF8=1) ====
    # Need to reload kernel module to flip USE_MXF8 (it's read at workspace init).
    # Clear workspace cache.
    K._workspace_cache.clear()
    os.environ["USE_MXF8"] = "1"

    mxf8_res = {}
    for u, inputs in wl_inputs:
        out_t, lat, err = bench_one(K, inputs)
        if err:
            mxf8_res[u] = (None, None, err)
        else:
            mxf8_res[u] = (out_t.cpu().float(), lat, None)

    # ==== Report ====
    lines = []
    lines.append("uuid      T       base_ms   mxf8_ms   speedup  match%  status")
    lines.append("-" * 76)
    speedups = []
    for u, inputs in wl_inputs:
        out_b, lat_b, err_b, T = baseline[u]
        out_m, lat_m, err_m = mxf8_res[u]
        if err_b:
            lines.append(f"{u:8} T={T:>5}  base FAIL: {err_b[:50]}")
            continue
        if err_m:
            lines.append(f"{u:8} T={T:>5}  {lat_b:7.3f}   --        --       --     MXF8 FAIL: {err_m[:50]}")
            continue

        abs_diff = (out_m - out_b).abs()
        tol = 1.0 + 0.3 * out_b.abs()
        matched = (abs_diff <= tol).float().mean().item()
        passed = "PASS" if matched >= 0.9 else "FAIL"
        speedup = lat_b / lat_m if lat_m > 0 else 0
        speedups.append(speedup)
        lines.append(
            f"{u:8} T={T:>5}  {lat_b:7.3f}  {lat_m:7.3f}  {speedup:5.2f}x   "
            f"{matched * 100:5.1f}  [{passed}]"
        )
    if speedups:
        import statistics as st
        lines.append("-" * 76)
        lines.append(f"geo_mean_speedup={st.geometric_mean(speedups):.3f}x   "
                     f"arith_mean={sum(speedups) / len(speedups):.3f}x   "
                     f"min={min(speedups):.3f}x   max={max(speedups):.3f}x")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(run_both.remote())
