"""Profile each stage of the MxF8 pipeline when running through custom_kernel.
Compare each stage against baseline for the same workload."""
import modal
app = modal.App("mxf8-pipeline-profile")
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


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/mnt": trace_volume})
def run(uuid: str = "5e8dc11c") -> str:
    import os, sys, time
    from pathlib import Path
    import torch

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]
    wobj = None
    for wl in trace_set.workloads.get(def_name, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith(uuid):
            wobj = w; break
    loaded_st = load_safetensors(definition, wobj, Path("/mnt/mlsys26-contest")) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
    inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
    T = int(inputs[0].shape[0])

    def bench_custom(mxf8: bool, trace: bool = False, iters: int = 30):
        for k in list(sys.modules):
            if k == "kernel" or k.startswith("kernel."):
                del sys.modules[k]
        os.environ["USE_MXF8"] = "1" if mxf8 else "0"
        if trace:
            os.environ["MXF8_TRACE"] = "1"
        else:
            os.environ.pop("MXF8_TRACE", None)
        import kernel as K
        _ = K._get_ext()
        K._workspace_cache.clear()
        # Warmup
        for _ in range(3):
            out = K.custom_kernel(*inputs)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            out = K.custom_kernel(*inputs)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t0 = bench_custom(mxf8=False, trace=False)
    t1 = bench_custom(mxf8=True, trace=True, iters=1)   # trace confirms path taken
    t2 = bench_custom(mxf8=True, trace=False)

    return f"UUID={uuid} T={T}  baseline={t0:.3f}ms  mxf8={t2:.3f}ms  speedup={t0 / t2:.3f}x"


@app.local_entrypoint()
def main(uuid: str = "5e8dc11c"):
    print(run.remote(uuid=uuid))
