"""
Profile each phase of the MoE pipeline on a single workload to find which
chunks dominate small-T latency. Uses CUDA events + torch.profiler.
"""
import modal

app = modal.App("fused-moe-profile")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/mnt": trace_volume})
def profile(uuid: str = "1a4c6ba1") -> str:
    import os
    import sys
    import torch
    from pathlib import Path

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    os.environ["DISABLE_CUDA_GRAPH"] = "1"  # profile individual kernels

    import kernel as K
    _ = K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE_PATH)
    definition = trace_set.definitions[DEF_NAME]

    wobj = None
    for wl in trace_set.workloads.get(DEF_NAME, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith(uuid):
            wobj = w
            break
    assert wobj is not None, f"uuid {uuid} not found"

    loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
    inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
    T = int(inputs[0].shape[0])

    # Warmup (JIT, allocator, everything)
    for _ in range(5):
        _ = K.custom_kernel(*inputs)
    torch.cuda.synchronize()

    # Record with torch.profiler
    torch_prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
    )
    torch_prof.start()
    for _ in range(10):
        _ = K.custom_kernel(*inputs)
    torch.cuda.synchronize()
    torch_prof.stop()

    # Summarize top GPU kernels
    out = []
    out.append(f"T={T}, uuid={uuid}, 10 iters")
    out.append("=" * 72)
    ka = torch_prof.key_averages()
    rows = sorted(ka, key=lambda r: r.device_time_total, reverse=True)[:25]
    out.append(f"{'kernel':<60s}  count   ms_avg  ms_tot")
    for r in rows:
        name = r.key[:58]
        # device_time is per-invocation, device_time_total across all.
        cnt = r.count
        if cnt == 0 or r.device_time == 0:
            continue
        ms_avg = r.device_time / 1000.0  # μs → ms (per call)
        ms_tot = r.device_time_total / 1000.0
        out.append(f"{name:<60s}  {cnt:>5d}  {ms_avg:7.3f}  {ms_tot:7.3f}")

    # Total pipeline time from CUDA events
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    N_ITER = 50
    start_ev.record()
    for _ in range(N_ITER):
        _ = K.custom_kernel(*inputs)
    end_ev.record()
    torch.cuda.synchronize()
    per_iter = start_ev.elapsed_time(end_ev) / N_ITER
    out.append("")
    out.append(f"Total per-call (no graph): {per_iter:.3f} ms")
    return "\n".join(out)


@app.local_entrypoint()
def main(uuid: str = "1a4c6ba1"):
    print(profile.remote(uuid=uuid))
