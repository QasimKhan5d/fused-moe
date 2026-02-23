"""
FlashInfer-Bench Local Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks locally.
"""

import os
import statistics
import sys
from pathlib import Path

import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution import pack_solution

# Workaround for https://github.com/flashinfer-ai/flashinfer-bench/issues/195
# The library's do_bench() calls torch.cuda.synchronize() per-iteration inside
# the benchmark loop, inflating runtimes for fast kernels. Replace with CUPTI.
import flashinfer_bench.bench.timing as _timing


def _patched_time_runnable(fn, args, warmup, iters, device):
    lock = _timing._device_lock(device)
    with lock:
        with torch.cuda.device(device):
            times = bench_gpu_time_with_cupti(
                fn=fn,
                dry_run_iters=warmup,
                repeat_iters=iters,
                input_args=tuple(args),
                cold_l2_cache=True,
            )
            return statistics.median(times)


_timing.time_runnable = _patched_time_runnable


def get_trace_set_path() -> str:
    """Get trace set path from environment variable."""
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH environment variable not set. "
            "Please set it to the path of your flashinfer-trace dataset."
        )
    return path


def run_benchmark(solution: Solution, config: BenchmarkConfig = None) -> dict:
    """Run benchmark locally and return results."""
    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    trace_set_path = get_trace_set_path()
    print(f"Loading trace set from: {trace_set_path}")
    trace_set = TraceSet.from_path(trace_set_path)

    if solution.definition not in trace_set.definitions:
        available = list(trace_set.definitions.keys())
        raise ValueError(
            f"Definition '{solution.definition}' not found in trace set.\n"
            f"Available definitions: {available}"
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    print(f"Found {len(workloads)} workloads for {solution.definition}")

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
    print(f"Got {len(traces)} traces from benchmark")
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
            results[definition.name][trace.workload.uuid] = entry
        else:
            print(f"Warning: Trace for workload {trace.workload.uuid} has no evaluation")

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        if not traces:
            print("  No results found!")
            return
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


def main():
    """Pack solution and run benchmark."""
    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark...")
    try:
        results = run_benchmark(solution)

        if not results:
            print("No results returned!")
            return

        print_results(results)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Required for multiprocessing on non-fork platforms
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
