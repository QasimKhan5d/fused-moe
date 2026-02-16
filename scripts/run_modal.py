"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
MOUNT_PATH = "/mnt"
TRACE_SET_PATH = "/mnt/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "build-essential", "cmake",
        "zlib1g-dev", "libxml2-dev",  # Required for LLVM/TLX build
    )
    .pip_install("flashinfer-bench", "torch", "numpy")
    # Install TLX-enabled Triton from Facebook's experimental repo
    .run_commands(
        "git clone https://github.com/facebookexperimental/triton.git /tmp/triton",
        "cd /tmp/triton && pip install -r python/requirements.txt",
        "cd /tmp/triton && pip install -e . --no-build-isolation",
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={MOUNT_PATH: trace_volume})
def run_benchmark(
    solution: Solution,
    config: BenchmarkConfig = None,
    profile: bool = False,
) -> dict:
    """Run benchmark on Modal B200 and return results.

    Args:
        solution: The solution to benchmark.
        config: Benchmark configuration.
        profile: If True, run torch.profiler on a representative workload
                 and include profiling data in the results.
    """
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
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
            # Capture evaluation log (stdout/stderr from the run — has tracebacks)
            if trace.evaluation.log:
                entry["log"] = trace.evaluation.log
            results[definition.name][trace.workload.uuid] = entry

    # Optional: run torch.profiler on a representative workload
    if profile:
        profiling_text = _profile_solution(
            solution, definition, workloads, trace_set.root
        )
        if profiling_text:
            results["__profiling__"] = profiling_text

    return results


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

        # Pick the median workload (by variable axis size) for representative profiling
        if len(workloads) >= 3:
            # Sort by the first variable axis value as proxy for workload size
            sorted_wl = sorted(
                workloads,
                key=lambda w: list(w.axes.values())[0] if w.axes else 0,
            )
            workload = sorted_wl[len(sorted_wl) // 2]
        else:
            workload = workloads[0]

        # Load safetensors if needed, generate inputs
        loaded_safe_tensors = (
            load_safetensors(definition, workload, trace_set_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )
        inputs = gen_inputs(
            definition, workload, device="cuda", safe_tensors=loaded_safe_tensors
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
        return f"[Profiling failed: {e}]"


def print_results(results: dict):
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
                    # Print first 800 chars of log (tracebacks, errors)
                    log_preview = group["log"][:800]
                    for line in log_preview.split("\n"):
                        print(f"    {line}")
            print()


@app.local_entrypoint()
def main(profile: bool = False):
    """Pack solution and run benchmark on Modal.
    
    Args:
        profile: Run torch.profiler on a representative workload (adds ~5s).
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    if profile:
        print("  (profiling enabled)")
    results = run_benchmark.remote(solution, profile=profile)

    if not results:
        print("No results returned!")
        return

    print_results(results)

    # Print profiling data if present (will be captured by model.py)
    profiling_text = results.get("__profiling__")
    if profiling_text:
        print(profiling_text)
        print("\nTEST HARNESS COMPLETE")
