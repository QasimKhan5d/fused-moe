"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/mlsys26-contest/
"""

import math
import shutil
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
MOUNT_PATH = "/mnt"
TRACE_SET_PATH = "/mnt/mlsys26-contest"
OFFICIAL_BENCHMARK_IMAGE = "flashinfer/flashinfer-ci-cu132:latest"
FLASHINFER_BENCH_GIT_URL = "git+https://github.com/flashinfer-ai/flashinfer-bench.git"
NCU_OUTPUT_START = "=== NCU OUTPUT START ==="
NCU_OUTPUT_END = "=== NCU OUTPUT END ==="

def format_official_stack_overlay() -> str:
    return (
        f"base image {OFFICIAL_BENCHMARK_IMAGE} + "
        "flashinfer-bench@main (built from source) + cupti-python>=13"
    )

benchmark_image = (
    modal.Image.from_registry(
        OFFICIAL_BENCHMARK_IMAGE,
    )
    .entrypoint([])
    .apt_install("wget", "gnupg")
    .run_commands(
        "apt-get update && apt-get install -y nsight-compute-2026.1.0 || "
        "(wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg && "
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' > /etc/apt/sources.list.d/cuda.list && "
        "apt-get update && apt-get install -y nsight-compute-2026.1.0)"
    )
    .pip_install(
        f"flashinfer-bench @ {FLASHINFER_BENCH_GIT_URL}",
        "cupti-python>=13",
    )
)


def _find_ncu_binary() -> str:
    """Return the Nsight Compute binary path inside the Modal container."""
    ncu_path = shutil.which("ncu")
    if ncu_path:
        return ncu_path

    candidates = sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu"))
    if not candidates:
        raise FileNotFoundError("Nsight Compute binary not found in Modal image")
    return str(candidates[-1])


def _workload_size_hint(workload) -> float:
    """Best-effort numeric size for choosing representative workloads."""
    workload_obj = getattr(workload, "workload", workload)
    axes = getattr(workload_obj, "axes", None)
    if isinstance(axes, dict) and axes:
        first_val = next(iter(axes.values()))
        if isinstance(first_val, (int, float)):
            return float(first_val)

    for attr in ("seq_len", "seqlen", "num_tokens", "tokens", "length"):
        val = getattr(workload_obj, attr, None)
        if isinstance(val, (int, float)):
            return float(val)

    return float("nan")


def _select_workload(workloads: list, strategy: str) -> object:
    """Choose a representative workload for profiling."""
    if not workloads:
        raise ValueError("No workloads available for profiling")

    if len(workloads) < 3:
        return workloads[0]

    hinted = [(i, _workload_size_hint(w)) for i, w in enumerate(workloads)]
    valid = [(i, size) for i, size in hinted if math.isfinite(size)]
    if not valid:
        return workloads[len(workloads) // 2]

    valid.sort(key=lambda item: item[1])
    if strategy == "largest":
        return workloads[valid[-1][0]]
    return workloads[valid[len(valid) // 2][0]]


def _find_flashinfer_bench_command() -> list[str]:
    """Resolve the flashinfer-bench CLI entrypoint inside the container."""
    binary = shutil.which("flashinfer-bench")
    if binary:
        return [binary]
    return [sys.executable, "-m", "flashinfer_bench"]


def _official_moe_cli_args(dataset_path: Path, definition_name: str) -> list[str]:
    """Mirror the MoE command in EVALUATION.md as closely as possible."""
    return _find_flashinfer_bench_command() + [
        "run",
        "--local",
        str(dataset_path),
        "--definitions",
        definition_name,
        "--save-results",
        "--use-isolated-runner",
        "--log-level",
        "INFO",
        "--resume",
        "--timeout",
        "300",
        "--atol",
        "1",
        "--rtol",
        "0.3",
        "--required-matched-ratio",
        "0.9",
    ]


def _serialize_jsonl(models: list) -> str:
    return "".join(f"{model.model_dump_json(indent=None)}\n" for model in models)


def _results_from_trace_set(result_trace_set: TraceSet, definition_name: str) -> dict:
    traces = result_trace_set.traces.get(definition_name, [])
    results = {definition_name: {}}

    for trace in traces:
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "solution": trace.solution,
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        if trace.evaluation.log:
            entry["log"] = trace.evaluation.log
        results[definition_name][trace.workload.uuid] = entry

    return results


@app.cls(
    image=benchmark_image,
    gpu="B200:1",
    timeout=3600,
    volumes={MOUNT_PATH: trace_volume},
    scaledown_window=900,
)
class BenchmarkRunner:
    @modal.enter()
    def load_trace_set(self):
        self.trace_set = TraceSet.from_path(TRACE_SET_PATH)

    @modal.method()
    def run_benchmark(
        self,
        solution: Solution,
        profile: bool = False,
        workload_uuids: tuple[str, ...] = (),
    ) -> dict:
        """Run benchmark on Modal B200 and return results."""
        import subprocess
        import tempfile

        if solution.definition not in self.trace_set.definitions:
            raise ValueError(f"Definition '{solution.definition}' not found in trace set")

        definition = self.trace_set.definitions[solution.definition]
        workloads = self.trace_set.workloads.get(solution.definition, [])

        if workload_uuids:
            selected = set(workload_uuids)
            filtered_workloads = []
            for workload in workloads:
                workload_obj = getattr(workload, "workload", workload)
                uuid = getattr(workload_obj, "uuid", "")
                if uuid in selected or any(uuid.startswith(p) for p in selected):
                    filtered_workloads.append(workload)
            workloads = filtered_workloads

        if not workloads:
            raise ValueError(f"No workloads found for definition '{solution.definition}'")

        with tempfile.TemporaryDirectory(prefix="flashinfer-bench-cli-") as tmpdir:
            dataset_path = Path(tmpdir)
            op_type = definition.op_type
            definition_dir = dataset_path / "definitions" / op_type
            solution_dir = dataset_path / "solutions" / solution.author / op_type / definition.name
            workload_dir = dataset_path / "workloads" / op_type
            traces_dir = dataset_path / "traces" / op_type
            blob_link = dataset_path / "blob"

            definition_dir.mkdir(parents=True, exist_ok=True)
            solution_dir.mkdir(parents=True, exist_ok=True)
            workload_dir.mkdir(parents=True, exist_ok=True)
            traces_dir.mkdir(parents=True, exist_ok=True)

            source_blob = self.trace_set.root / "blob"
            if source_blob.exists():
                blob_link.symlink_to(source_blob, target_is_directory=True)

            (definition_dir / f"{definition.name}.json").write_text(
                definition.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (solution_dir / f"{solution.name}.json").write_text(
                solution.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (workload_dir / f"{definition.name}.jsonl").write_text(
                _serialize_jsonl(workloads),
                encoding="utf-8",
            )

            proc = subprocess.run(
                _official_moe_cli_args(dataset_path, definition.name),
                capture_output=True,
                text=True,
            )

            result_trace_set = TraceSet.from_path(str(dataset_path))
            results = _results_from_trace_set(result_trace_set, definition.name)

            if proc.returncode != 0 and not results[definition.name]:
                combined_output = []
                if proc.stdout.strip():
                    combined_output.append(f"STDOUT:\n{proc.stdout.strip()}")
                if proc.stderr.strip():
                    combined_output.append(f"STDERR:\n{proc.stderr.strip()}")
                raise RuntimeError(
                    "flashinfer-bench CLI failed before producing any traces.\n\n"
                    + "\n\n".join(combined_output)
                )

        if profile:
            profiling_text = _profile_solution(
                solution, definition, workloads, self.trace_set.root
            )
            if profiling_text:
                results["__profiling__"] = profiling_text

        return results

@app.cls(
    image=benchmark_image,
    gpu="B200:1",
    timeout=3600,
    volumes={MOUNT_PATH: trace_volume},
    scaledown_window=900,
)
class NcuRunner:
    @modal.enter()
    def load_trace_set(self):
        self.trace_set = TraceSet.from_path(TRACE_SET_PATH)

    @modal.method()
    def run_ncu(
        self,
        solution: Solution,
        workload_uuids: tuple[str, ...] = (),
    ) -> str:
        """Run Nsight Compute on a representative workload inside Modal."""
        if solution.definition not in self.trace_set.definitions:
            raise ValueError(f"Definition '{solution.definition}' not found in trace set")

        definition = self.trace_set.definitions[solution.definition]
        workloads = self.trace_set.workloads.get(solution.definition, [])

        if workload_uuids:
            selected = set(workload_uuids)
            workloads = [
                workload
                for workload in workloads
                if (lambda uuid: uuid in selected or any(uuid.startswith(p) for p in selected))(
                    getattr(getattr(workload, "workload", workload), "uuid", "")
                )
            ]

        if not workloads:
            raise ValueError(f"No workloads found for definition '{solution.definition}'")

        return _run_ncu_solution(solution, definition, workloads, self.trace_set.root)

def _profile_solution(
    solution: Solution,
    definition,
    workloads: list,
    trace_set_root,
) -> str:
    """Run torch.profiler on the solution kernel for a representative workload.

    Profiles the MEDIAN workload (by token count / variable axis size) to get
    a representative profile that's not dominated by edge-case behavior.

    Returns the profiling summary as a formatted string, or empty string on failure.
    """
    import io
    import torch
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    try:
        # Build the solution into a callable
        registry = BuilderRegistry.get_instance()
        runnable = registry.build(definition, solution)

        # Pick a representative workload near the median size.
        workload = _select_workload(workloads, strategy="median")
        workload_obj = getattr(workload, "workload", workload)

        # Load safetensors if needed, generate inputs
        loaded_safe_tensors = (
            load_safetensors(definition, workload_obj, trace_set_root)
            if any(
                d.type == "safetensors"
                for d in getattr(workload_obj, "inputs", {}).values()
            )
            else {}
        )
        inputs = gen_inputs(
            definition, workload_obj, device="cuda", safe_tensors=loaded_safe_tensors
        )

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = runnable(*inputs)
        torch.cuda.synchronize()

        # Profile
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            for _ in range(5):
                with torch.no_grad():
                    _ = runnable(*inputs)
            torch.cuda.synchronize()

        # Format output in the same format the KernelEvolve profiling pipeline expects
        buf = io.StringIO()
        buf.write("\n" + "-" * 80 + "\n")
        buf.write("PROFILING SUMMARY (Top Operations by CUDA Time)\n")
        buf.write("-" * 80 + "\n")

        # Get key averages sorted by CUDA time
        key_averages = prof.key_averages()
        key_averages_sorted = sorted(
            key_averages,
            key=lambda e: e.device_time_total,
            reverse=True,
        )

        # Header
        header_fmt = "{:<60}  {:>10}  {:>10}  {:>10}  {:>12}  {:>10}"
        buf.write(
            header_fmt.format(
                "Name", "Self CPU%", "CPU Total%", "Self CUDA%",
                "CUDA Total", "# Calls",
            )
            + "\n"
        )
        buf.write("-" * 120 + "\n")

        total_cuda = sum(
            e.device_time_total for e in key_averages if e.device_time_total > 0
        )
        total_cpu = sum(
            e.self_cpu_time_total for e in key_averages if e.self_cpu_time_total > 0
        )

        for item in key_averages_sorted[:20]:
            if item.device_time_total <= 0:
                continue
            name = item.key[:60]
            self_cpu_pct = (
                f"{item.self_cpu_time_total / total_cpu * 100:.1f}%"
                if total_cpu > 0 else "0.0%"
            )
            cpu_total_pct = (
                f"{item.cpu_time_total / total_cpu * 100:.1f}%"
                if total_cpu > 0 else "0.0%"
            )
            cuda_pct = (
                f"{item.device_time_total / total_cuda * 100:.1f}%"
                if total_cuda > 0 else "0.0%"
            )
            # Format CUDA time
            cuda_us = item.device_time_total
            if cuda_us >= 1_000_000:
                cuda_str = f"{cuda_us / 1_000_000:.3f}s"
            elif cuda_us >= 1000:
                cuda_str = f"{cuda_us / 1000:.3f}ms"
            else:
                cuda_str = f"{cuda_us:.1f}us"

            buf.write(
                header_fmt.format(
                    name, self_cpu_pct, cpu_total_pct, cuda_pct,
                    cuda_str, str(item.count),
                )
                + "\n"
            )

        # Summary lines
        buf.write("-" * 120 + "\n")
        if total_cpu > 0:
            cpu_ms = total_cpu / 1000
            buf.write(f"Self CPU time total: {cpu_ms:.3f}ms\n")
        if total_cuda > 0:
            cuda_ms = total_cuda / 1000
            buf.write(f"Self CUDA time total: {cuda_ms:.3f}ms\n")

        runnable.cleanup()
        return buf.getvalue()

    except Exception as e:
        # Keep output parseable by KernelEvolve's profiling extractor.
        err = str(e).replace("\n", " ").strip()
        return (
            "PROFILING SUMMARY (Top Operations by CUDA Time)\n"
            f"[Profiling failed: {err}]\n"
            "Self CPU time total: 0.000ms\n"
            "Self CUDA time total: 0.000ms"
        )


def _run_ncu_solution(
    solution: Solution,
    definition,
    workloads: list,
    trace_set_root,
) -> str:
    """Run NCU via subprocess using the Modal-supported workflow."""
    import glob
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    workload = _select_workload(workloads, strategy="largest")
    workload_obj = getattr(workload, "workload", workload)

    with tempfile.TemporaryDirectory(prefix="fused-moe-ncu-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        solution_path = tmpdir_path / "solution.json"
        definition_path = tmpdir_path / "definition.json"
        workload_path = tmpdir_path / "workload.json"
        runner_path = tmpdir_path / "run_ncu_target.py"

        solution_path.write_text(solution.model_dump_json())
        definition_path.write_text(definition.model_dump_json())
        workload_path.write_text(workload_obj.model_dump_json())

        runner_path.write_text(
            textwrap.dedent(
                f"""
                import json
                from pathlib import Path

                import torch
                from flashinfer_bench import Solution, Definition, Workload
                from flashinfer_bench.compile import BuilderRegistry
                from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

                tmpdir = Path({tmpdir!r})
                trace_set_root = Path({str(trace_set_root)!r})

                solution = Solution.model_validate_json((tmpdir / "solution.json").read_text())
                definition = Definition.model_validate_json((tmpdir / "definition.json").read_text())
                workload = Workload.model_validate_json((tmpdir / "workload.json").read_text())

                registry = BuilderRegistry.get_instance()
                runnable = registry.build(definition, solution)

                loaded_safe_tensors = (
                    load_safetensors(definition, workload, trace_set_root)
                    if any(d.type == "safetensors" for d in getattr(workload, "inputs", {{}}).values())
                    else {{}}
                )
                inputs = gen_inputs(
                    definition,
                    workload,
                    device="cuda",
                    safe_tensors=loaded_safe_tensors,
                )

                for _ in range(3):
                    with torch.no_grad():
                        _ = runnable(*inputs)
                torch.cuda.synchronize()

                with torch.no_grad():
                    _ = runnable(*inputs)
                torch.cuda.synchronize()

                runnable.cleanup()
                """
            ).strip()
            + "\n"
        )

        candidate_lib_dirs = []
        candidate_lib_patterns = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/extras/CUPTI/lib64",
        ]
        for pattern in sys.path:
            candidate_lib_patterns.extend(
                [
                    os.path.join(pattern, "nvidia", "*", "lib"),
                    os.path.join(pattern, "nvidia", "*", "bin"),
                    os.path.join(pattern, "torch", "lib"),
                ]
            )

        seen_dirs = set()
        for pattern in candidate_lib_patterns:
            for path in glob.glob(pattern):
                if os.path.isdir(path) and path not in seen_dirs:
                    seen_dirs.add(path)
                    candidate_lib_dirs.append(path)

        env = os.environ.copy()
        existing_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        ld_library_parts = candidate_lib_dirs[:]
        if existing_ld_library_path:
            ld_library_parts.append(existing_ld_library_path)
        env["LD_LIBRARY_PATH"] = ":".join(ld_library_parts)

        cmd = [
            _find_ncu_binary(),
            "--set",
            "full",
            "--target-processes",
            "all",
            "--launch-skip",
            "30",
            "--launch-count",
            "20",
            "python",
            str(runner_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
        except Exception as exc:
            return f"ERROR: NCU profiling failed: {exc}"

        parts = [
            f"Representative workload UUID: {getattr(workload_obj, 'uuid', 'unknown')}",
            f"NCU exit code: {proc.returncode}",
        ]
        if proc.stdout.strip():
            parts.append(proc.stdout.strip())
        if proc.stderr.strip():
            parts.append(f"STDERR:\n{proc.stderr.strip()}")

        if proc.returncode != 0:
            diagnostics = []
            diagnostics.append(
                "ENVIRONMENT:\n"
                f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}\n"
                f"PATH={os.environ.get('PATH', '')}"
            )
            diagnostics.append(
                "CANDIDATE_LIB_DIRS:\n" + "\n".join(candidate_lib_dirs)
            )

            for diag_cmd in (
                ["python", "--version"],
                ["nvidia-smi"],
                ["bash", "-lc", "ldconfig -p | rg 'libcuda|libcupti|libnvidia' || true"],
                ["bash", "-lc", "ls -l /usr/lib/x86_64-linux-gnu/libcuda* /usr/lib/x86_64-linux-gnu/libcupti* 2>/dev/null || true"],
                ["bash", "-lc", "ls -l /usr/local/cuda/lib64/libcupti* 2>/dev/null || true"],
                ["bash", "-lc", "env | rg 'CUDA|NVIDIA|LD_' || true"],
                ["bash", "-lc", "ls -1 /tmp/nsight-compute-*.log 2>/dev/null || true"],
                ["bash", "-lc", "for f in /tmp/nsight-compute-*.log; do [ -f \"$f\" ] && echo \"--- $f ---\" && sed -n '1,220p' \"$f\"; done"],
            ):
                try:
                    diag_proc = subprocess.run(
                        diag_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    rendered_cmd = " ".join(diag_cmd)
                    output = (diag_proc.stdout or "").strip()
                    err = (diag_proc.stderr or "").strip()
                    block = [f"$ {rendered_cmd}"]
                    if output:
                        block.append(output)
                    if err:
                        block.append(f"STDERR:\n{err}")
                    diagnostics.append("\n".join(block))
                except Exception as diag_exc:
                    diagnostics.append(
                        f"$ {' '.join(diag_cmd)}\n[diagnostic command failed: {diag_exc}]"
                    )

            parts.append("DIAGNOSTICS:\n\n" + "\n\n".join(diagnostics))
        return "\n\n".join(parts)


def print_results(results: dict, full_error_log: bool = False):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        if def_name.startswith("__"):  # Skip internal keys like __profiling__
            continue
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

        # Print detailed error summary for failed workloads
        failed = {k: v for k, v in traces.items() if v.get("status") != "PASSED"}
        if failed:
            print(f"\n  === FAILURE DETAILS ({len(failed)} workloads) ===")
            # Group by status + first line of log to avoid repetition
            error_groups = {}
            for uuid, result in failed.items():
                status = result.get("status", "UNKNOWN")
                log = result.get("log", "")
                # Build a concise error description
                abs_err = result.get("max_abs_error")
                rel_err = result.get("max_rel_error")
                err_parts = [f"status={status}"]
                if abs_err is not None:
                    err_parts.append(f"max_abs_err={abs_err:.2e}")
                if rel_err is not None:
                    err_parts.append(f"max_rel_err={rel_err:.2e}")
                summary = ", ".join(err_parts)

                key = f"{status}:{log[:200]}"
                if key not in error_groups:
                    error_groups[key] = {
                        "uuids": [], "summary": summary, "log": log,
                    }
                error_groups[key]["uuids"].append(uuid[:8])

            for group in error_groups.values():
                uuids_str = ", ".join(group["uuids"])
                print(f"  Workloads [{uuids_str}]: {group['summary']}")
                if group["log"]:
                    # Keep the default output compact unless explicitly asked for full logs.
                    log_preview = group["log"] if full_error_log else group["log"][:800]
                    for line in log_preview.split("\n"):
                        print(f"    {line}")
            print()


def _load_solution(solution_json: str = "") -> Solution:
    if solution_json:
        solution_path = Path(solution_json).expanduser().resolve()
        print(f"Loading solution from: {solution_path}")
    else:
        from scripts.pack_solution import pack_solution

        print("Packing solution from source files...")
        solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    print(f"Loaded: {solution.name} ({solution.definition})")
    return solution


@app.local_entrypoint()
def main(
    profile: bool = False,
    ncu: bool = False,
    ncu_only: bool = False,
    workload_uuids: str = "",
    full_error_log: bool = False,
    solution_json: str = "",
):
    """Pack solution and run benchmark on Modal.
    
    Args:
        profile: Run torch.profiler on a representative workload (adds ~5s).
        ncu: Run Nsight Compute on one representative large workload.
        ncu_only: Skip the benchmark pass and run only Nsight Compute.
        workload_uuids: Comma-separated workload UUIDs to benchmark.
        full_error_log: Print complete failure logs instead of truncating them.
        solution_json: Optional path to an existing solution JSON file. If empty,
            the script packs the current repo's sources before benchmarking.
    """
    solution = _load_solution(solution_json)

    print("\nRunning benchmark on Modal B200...")
    print(f"Using benchmark environment: {format_official_stack_overlay()}")
    print("Benchmark container stays warm between runs for reproducibility and lower startup overhead.")
    print("Evaluation strategy mirrors `EVALUATION.md` for MoE: isolated runner, resume,")
    print("saved traces, timeout=300, atol=1, rtol=0.3, required_matched_ratio=0.9.")
    if profile:
        print("  (profiling enabled)")
    if ncu:
        print("  (NCU enabled)")
    if ncu_only:
        print("  (NCU-only mode)")
    if full_error_log:
        print("  (full error log enabled)")
    selected_workloads = tuple(
        uuid.strip() for uuid in workload_uuids.split(",") if uuid.strip()
    )
    if selected_workloads:
        print(f"  (filtering to {len(selected_workloads)} workloads)")
    runner = BenchmarkRunner()
    ncu_runner = NcuRunner() if ncu else None

    if ncu_only and not ncu:
        print("NCU-only mode requires `--ncu`.")
        return

    if not ncu_only:
        results = runner.run_benchmark.remote(
            solution,
            profile=profile,
            workload_uuids=selected_workloads,
        )

        if not results:
            print("No results returned!")
            return

        print_results(results, full_error_log=full_error_log)

        # Print profiling data if present (will be captured by model.py)
        profiling_text = results.get("__profiling__")
        if profiling_text:
            print(profiling_text)
            print("\nTEST HARNESS COMPLETE")

    if ncu:
        ncu_output = ncu_runner.run_ncu.remote(
            solution,
            workload_uuids=selected_workloads,
        )
        print(f"\n{NCU_OUTPUT_START}")
        print(f"NCU environment: {format_official_stack_overlay()}")
        print(ncu_output)
        print(NCU_OUTPUT_END)
