"""
Run an existing experiment kernel file through the official flashinfer-bench
CLI without modifying the active solution.

Usage:
  python -m modal run experiments/fast_bench/run_official_candidate.py \
    --kernel-path experiments/fp8_dotscaled/kernel_b200_grouped_tma_hugepersistent.py \
    --workload-uuids "1a4c6ba1,5e8dc11c"
"""

from pathlib import Path

import modal

app = modal.App("run-official-candidate")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

benchmark_image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
        "cupti-python>=13",
    )
)

DEFINITION_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


@app.function(
    image=benchmark_image,
    gpu="B200:1",
    timeout=1800,
    volumes={"/mnt": trace_volume},
)
def run_candidate(kernel_source: str, kernel_name: str, workload_uuids: str = ""):
    import subprocess
    import tempfile
    from pathlib import Path

    from flashinfer_bench import BuildSpec, Solution, SourceFile, TraceSet

    trace_set_path = "/mnt/mlsys26-contest"

    def serialize_jsonl(models: list) -> str:
        return "".join(f"{model.model_dump_json(indent=None)}\n" for model in models)

    def official_moe_cli_args(dataset_path: Path, definition_name: str) -> list[str]:
        return [
            "flashinfer-bench",
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

    def results_from_trace_set(result_trace_set: TraceSet, definition_name: str) -> dict:
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

    def print_results(results: dict):
        for def_name, traces in results.items():
            print(f"\n{def_name}:")
            for workload_uuid, result in traces.items():
                status = result.get("status")
                line = f"  {workload_uuid[:8]}: {status}"
                if result.get("latency_ms") is not None:
                    line += f" | {result['latency_ms']:.3f} ms"
                if result.get("speedup_factor") is not None:
                    line += f" | {result['speedup_factor']:.2f}x"
                if result.get("max_abs_error") is not None:
                    line += (
                        f" | abs_err={result['max_abs_error']:.2e}"
                        f", rel_err={result.get('max_rel_error', 0):.2e}"
                    )
                print(line)
                log = result.get("log", "")
                if log:
                    print("    LOG:")
                    for line in log.split("\n")[:40]:
                        print(f"    {line}")

    solution = Solution(
        name=f"candidate_{kernel_name}",
        definition=DEFINITION_NAME,
        author="Cursor",
        spec=BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="kernel.py::run",
            binding="torch",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path="kernel.py", content=kernel_source)],
        description=f"Candidate kernel {kernel_name}",
    )

    trace_set = TraceSet.from_path(trace_set_path)
    definition = trace_set.definitions[solution.definition]
    selected = {uuid.strip() for uuid in workload_uuids.split(",") if uuid.strip()}
    all_workloads = trace_set.workloads.get(solution.definition, [])
    if selected:
        workloads = [
            workload
            for workload in all_workloads
            if (
                lambda uuid: uuid in selected or any(uuid.startswith(prefix) for prefix in selected)
            )(getattr(getattr(workload, "workload", workload), "uuid", ""))
        ]
        if not workloads:
            raise ValueError(f"No workloads found for selection: {workload_uuids}")
    else:
        workloads = list(all_workloads)

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

        source_blob = trace_set.root / "blob"
        if source_blob.exists():
            blob_link.symlink_to(source_blob, target_is_directory=True)

        (definition_dir / f"{definition.name}.json").write_text(
            definition.model_dump_json(indent=2), encoding="utf-8"
        )
        (solution_dir / f"{solution.name}.json").write_text(
            solution.model_dump_json(indent=2), encoding="utf-8"
        )
        (workload_dir / f"{definition.name}.jsonl").write_text(
            serialize_jsonl(workloads), encoding="utf-8"
        )

        proc = subprocess.run(
            official_moe_cli_args(dataset_path, definition.name),
            capture_output=True,
            text=True,
        )

        print(f"=== Candidate {kernel_name} stdout ===")
        if proc.stdout.strip():
            print(proc.stdout)
        print(f"=== Candidate {kernel_name} stderr ===")
        if proc.stderr.strip():
            print(proc.stderr)

        result_trace_set = TraceSet.from_path(str(dataset_path))
        results = results_from_trace_set(result_trace_set, definition.name)
        print_results(results)
        if proc.returncode != 0:
            print(f"\nCLI exit code: {proc.returncode}")
            raise RuntimeError("flashinfer-bench CLI returned non-zero exit code")


@app.local_entrypoint()
def main(kernel_path: str, workload_uuids: str = ""):
    kernel_file = Path(kernel_path).expanduser()
    kernel_source = kernel_file.read_text(encoding="utf-8")
    run_candidate.remote(
        kernel_source=kernel_source,
        kernel_name=kernel_file.stem,
        workload_uuids=workload_uuids,
    )
