"""NCU-based per-kernel breakdown on a given workload.

Runs NCU with a wider launch-skip to capture all major kernels in one
iteration of a 2-iter warm run.
"""
from pathlib import Path
import modal

app = modal.App("moe-ncu-breakdown")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .apt_install("wget", "gnupg")
    .run_commands(
        "apt-get update && apt-get install -y nsight-compute-2026.1.0 || "
        "(wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' > /etc/apt/sources.list.d/cuda.list && "
        "apt-get update && apt-get install -y nsight-compute-2026.1.0)"
    )
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
        "cupti-python>=13",
    )
)


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run_ncu(solution_json_str: str, uuid_prefix: str = "5e8dc11c", launch_skip: int = 20, launch_count: int = 30) -> str:
    import os
    import glob
    import shutil
    import subprocess
    import sys
    import tempfile
    import textwrap
    from pathlib import Path

    from flashinfer_bench import Solution, TraceSet

    solution = Solution.model_validate_json(solution_json_str)
    TRACE = "/mnt/mlsys26-contest"
    DEF = solution.definition
    trace_set = TraceSet.from_path(TRACE)
    definition = trace_set.definitions[DEF]

    wobj = None
    for wl in trace_set.workloads.get(DEF, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith(uuid_prefix):
            wobj = w
            break
    if wobj is None:
        return f"workload {uuid_prefix} not found"

    with tempfile.TemporaryDirectory(prefix="moe-ncu-") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "solution.json").write_text(solution.model_dump_json())
        (tmpdir / "definition.json").write_text(definition.model_dump_json())
        (tmpdir / "workload.json").write_text(wobj.model_dump_json())
        runner = tmpdir / "run.py"
        runner.write_text(textwrap.dedent(f"""
            import os
            os.environ["DISABLE_CUDA_GRAPH"] = "1"
            import torch
            from pathlib import Path
            from flashinfer_bench import Solution, Definition, Workload
            from flashinfer_bench.compile import BuilderRegistry
            from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

            root = Path(r"{tmpdir!s}")
            solution = Solution.model_validate_json((root/"solution.json").read_text())
            definition = Definition.model_validate_json((root/"definition.json").read_text())
            workload = Workload.model_validate_json((root/"workload.json").read_text())
            runnable = BuilderRegistry.get_instance().build(definition, solution)
            loaded = load_safetensors(definition, workload, Path(r"{TRACE}")) if any(
                d.type == "safetensors" for d in getattr(workload, "inputs", {{}}).values()) else {{}}
            inputs = gen_inputs(definition, workload, device="cuda", safe_tensors=loaded)
            for _ in range(2):
                with torch.no_grad(): _ = runnable(*inputs)
            torch.cuda.synchronize()
            for _ in range(1):
                with torch.no_grad(): _ = runnable(*inputs)
            torch.cuda.synchronize()
            runnable.cleanup()
        """).strip() + "\n")

        ncu = shutil.which("ncu") or (sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu")) or [None])[-1]
        if not ncu:
            return "ncu not found"
        ncu = str(ncu)

        lib_dirs = ["/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64",
                    "/usr/local/cuda/extras/CUPTI/lib64"]
        for p in sys.path:
            lib_dirs.extend(glob.glob(os.path.join(p, "nvidia", "*", "lib")))
            lib_dirs.extend(glob.glob(os.path.join(p, "torch", "lib")))
        lib_dirs = [d for d in lib_dirs if os.path.isdir(d)]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [env.get("LD_LIBRARY_PATH", "")])

        cmd = [
            ncu, "--csv",
            "--metrics",
            "gpu__time_duration.sum,"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
            "gpc__cycles_elapsed.avg.per_second,"
            "dram__bytes.sum",
            "--target-processes", "all",
            "--launch-skip", str(launch_skip), "--launch-count", str(launch_count),
            "python", str(runner),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
        out = [proc.stdout, "--- stderr (tail) ---", proc.stderr[-4000:]]
        return "\n".join(out)


@app.local_entrypoint()
def main(solution_json: str, uuid_prefix: str = "5e8dc11c", launch_skip: int = 20, launch_count: int = 30):
    js = Path(solution_json).expanduser().read_text()
    print(run_ncu.remote(js, uuid_prefix, launch_skip, launch_count))
