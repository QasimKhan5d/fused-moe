"""Test CfgMidM (_64, _128, _128) for large-T workloads to see if better tile-M
utilization helps (partial last tile becomes 56/64=87.5% instead of 56/128=44%)."""
import modal
import os

app = modal.App("test-cfgmidm-large")
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


@app.function(image=image, gpu="B200:1", timeout=1200, volumes={"/mnt": trace_volume})
def run(uuids: str = "5e8dc11c,58a34f27") -> str:
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

    out = []
    for mode in ("A", "M"):
        os.environ["CUTLASS_CFG_MODE"] = mode
        # Re-import kernel to ensure fresh ext loaded once; modifying env won't
        # affect already-loaded ext module since we check env inside C++.
        import kernel as K
        K._get_ext()

        out.append(f"\n=== CUTLASS_CFG_MODE={mode} ===")
        for u in uuids.split(","):
            u = u.strip()
            wobj = None
            for wl in trace_set.workloads.get(def_name, []):
                w = getattr(wl, "workload", wl)
                if getattr(w, "uuid", "").startswith(u):
                    wobj = w; break
            if wobj is None:
                continue

            loaded_st = load_safetensors(
                definition, wobj, Path("/mnt/mlsys26-contest")
            ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
            inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
            T = inputs[0].shape[0]

            # Warmup
            for _ in range(3):
                _ = K.custom_kernel(*inputs)
            torch.cuda.synchronize()

            # Timed runs
            N = 10
            times = []
            for _ in range(N):
                t0 = time.perf_counter_ns()
                _ = K.custom_kernel(*inputs)
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                times.append((t1 - t0) / 1e3)
            times.sort()
            med = times[N//2]
            out.append(f"  {u[:8]} T={T}: median_us={med:.1f} "
                       f"min={times[0]:.1f} max={times[-1]:.1f}")

    return "\n".join(out)


@app.local_entrypoint()
def main(uuids: str = "5e8dc11c,58a34f27,1a4c6ba1"):
    print(run.remote(uuids=uuids))
