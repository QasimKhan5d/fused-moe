"""
Ground-truth HBM bandwidth and kernel-time measurement via NVIDIA Nsight Compute.
Runs our custom_kernel on a single T workload, profiles each kernel launch,
and reports:
  - dram__bytes.sum (total HBM bytes moved)
  - gpu__time_duration.sum (kernel duration on device)
  - achieved HBM GB/s per kernel
  - total HBM bytes and time summed across all kernels in the pipeline
  - (implied) overall SoL = measured_bytes / HBM_peak
"""
import modal

app = modal.App("moe-ncu-profile")
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


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def profile(uuid: str) -> str:
    import os
    import sys
    import subprocess
    import textwrap

    # Write a helper script that runs the kernel once (after warmup) so NCU
    # captures every kernel launch in the fused MoE pipeline.
    HELPER = textwrap.dedent(f"""
    import os, sys, torch
    from pathlib import Path
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions["moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]

    wobj = None
    for wl in trace_set.workloads.get("moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048", []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith("{uuid}"):
            wobj = w; break
    assert wobj is not None, "uuid not found"

    loaded_st = load_safetensors(definition, wobj, Path("/mnt/mlsys26-contest")) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {{}}).values()) else {{}}
    inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
    T = int(inputs[0].shape[0])
    print(f"running T={{T}} under NCU", flush=True)

    # Warmup outside timed region
    for _ in range(3):
        _ = K.custom_kernel(*inputs)
    torch.cuda.synchronize()

    # Single call to profile (keep it to one set of kernel launches for readability)
    torch.cuda.cudart().cudaProfilerStart()
    _ = K.custom_kernel(*inputs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    """)
    helper_path = "/tmp/_ncu_helper.py"
    with open(helper_path, "w") as f:
        f.write(HELPER)

    # Run NCU with HBM + time metrics via a built-in section. Using --section
    # MemoryWorkloadAnalysis collects dram__bytes + gpu__time reliably, and
    # --print-summary per-gpu outputs compact per-kernel rows.
    ncu_cmd = [
        "ncu",
        "--profile-from-start", "off",
        "--target-processes", "all",
        "--metrics", "dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,sm__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "--csv",
        "--print-units", "base",
        "python", helper_path,
    ]
    proc = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=1500)

    out = []
    out.append(f"NCU exit code: {proc.returncode}")
    out.append("=" * 100)
    if proc.stderr:
        # NCU prints a lot of noise to stderr; keep a tail
        out.append("--- NCU stderr (tail) ---")
        out.append(proc.stderr[-2000:])
    out.append("")
    out.append("--- NCU CSV (full) ---")
    out.append(proc.stdout)
    return "\n".join(out)


@app.local_entrypoint()
def main(uuid: str = "5e8dc11c"):
    # Accept comma-separated uuids and run in parallel (Modal .map).
    uuids = [u.strip() for u in uuid.split(",") if u.strip()]
    if len(uuids) == 1:
        print(profile.remote(uuid=uuids[0]))
    else:
        results = list(profile.map(uuids))
        for u, r in zip(uuids, results):
            print("\n" + "=" * 100)
            print(f"### NCU profile: {u}")
            print("=" * 100)
            print(r)
