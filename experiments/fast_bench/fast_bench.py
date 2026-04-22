"""
Fast iteration harness for solution/python/kernel.py on real contest traces.

Loads all requested workloads ONCE, JIT-compiles the CUTLASS extension ONCE,
then benchmarks each workload in-process using CUDA events. Much faster than
the official flashinfer-bench canary (no subprocess spawn / re-JIT per call).

Usage:
    modal run experiments/fast_bench/fast_bench.py
    modal run experiments/fast_bench/fast_bench.py --uuids "e05c6c03,1a4c6ba1"
    modal run experiments/fast_bench/fast_bench.py --iters 20 --warmup 3
    modal run experiments/fast_bench/fast_bench.py --all  # run all 19 workloads
"""
import modal

app = modal.App("fused-moe-fast-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
        "cupti-python>=13",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


DEFAULT_UUIDS = "e05c6c03,81955b1e,1a4c6ba1,58a34f27,5e8dc11c"


@app.function(
    image=image,
    gpu="B200:1",
    timeout=1800,
    volumes={"/mnt": trace_volume},
)
def bench(uuids: str = DEFAULT_UUIDS, warmup: int = 3, iters: int = 20,
          trials: int = 1, run_all: bool = False) -> str:
    """Run all specified workloads through the in-process solution kernel."""
    import os
    import sys
    import time

    import torch
    from pathlib import Path

    t_start = time.time()

    def mark(msg):
        sys.stdout.write(f"[{time.time() - t_start:7.2f}s] {msg}\n")
        sys.stdout.flush()

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    # Pass GRAPH_T_MAX from environment if set on the local side.
    if os.environ.get("GRAPH_T_MAX"):
        pass  # already set

    mark("importing kernel")
    import kernel as K  # noqa: E402

    mark("calling _get_ext()")
    _ = K._get_ext()  # noqa: SLF001
    mark("_get_ext() done")

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

    mark(f"loading TraceSet from {TRACE_PATH}")
    trace_set = TraceSet.from_path(TRACE_PATH)
    mark("TraceSet loaded")
    definition = trace_set.definitions[DEF_NAME]

    if run_all:
        # use all workloads in the trace set, sorted by T
        all_wls = [getattr(wl, "workload", wl)
                   for wl in trace_set.workloads.get(DEF_NAME, [])]
        all_wls.sort(key=lambda w: w.axes.get("seq_len", 0))
        target_workloads = {getattr(w, "uuid", ""): w for w in all_wls}
        selected_uuids = list(target_workloads.keys())
    else:
        selected_uuids = [u.strip() for u in uuids.split(",") if u.strip()]
        target_workloads = {}
        for wl in trace_set.workloads.get(DEF_NAME, []):
            wobj = getattr(wl, "workload", wl)
            uuid = getattr(wobj, "uuid", "")
            for sel in selected_uuids:
                if uuid.startswith(sel):
                    target_workloads[sel] = wobj
                    break

    results = []
    results.append(f"GPU: {torch.cuda.get_device_name()}")
    results.append(f"warmup={warmup}, iters={iters}, trials={trials}")
    results.append("=" * 72)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for sel in selected_uuids:
        if sel not in target_workloads:
            results.append(f"{sel}: NOT FOUND in trace set")
            continue
        wobj = target_workloads[sel]

        mark(f"{sel}: loading safetensors")
        loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
            d.type == "safetensors"
            for d in getattr(wobj, "inputs", {}).values()
        ) else {}
        mark(f"{sel}: gen_inputs")
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        mark(f"{sel}: warmup T={T}")
        try:
            for _ in range(max(1, warmup)):
                out = K.custom_kernel(*inputs)
            torch.cuda.synchronize()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            import traceback
            results.append(
                f"{sel[:8]} T={T}: WARMUP FAIL ({exc!r})\n"
                f"{traceback.format_exc()[-600:]}"
            )
            continue

        mark(f"{sel}: timed run ({iters} iters × {trials} trials)")
        trial_times = []
        for _ in range(max(1, trials)):
            torch.cuda.synchronize()
            start_ev.record()
            for _ in range(iters):
                out = K.custom_kernel(*inputs)
            end_ev.record()
            torch.cuda.synchronize()
            trial_times.append(start_ev.elapsed_time(end_ev) / iters)
        trial_times.sort()
        median_ms = trial_times[len(trial_times) // 2]
        min_ms = trial_times[0]

        nz = (out.abs().float().sum(dim=1) > 0).sum().item()
        if trials > 1:
            results.append(
                f"{sel[:8]} T={T:>5}  median={median_ms:7.3f} min={min_ms:7.3f} ms   "
                f"nz={nz}/{out.shape[0]}"
            )
        else:
            results.append(
                f"{sel[:8]} T={T:>5}  {median_ms:7.3f} ms   "
                f"nonzero={nz}/{out.shape[0]}  dtype={out.dtype}"
            )

    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = DEFAULT_UUIDS, warmup: int = 3, iters: int = 20,
         trials: int = 1, all: bool = False):
    out = bench.remote(uuids=uuids, warmup=warmup, iters=iters,
                       trials=trials, run_all=all)
    print(out)
