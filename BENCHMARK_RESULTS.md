# Fused MoE Benchmark Results

Date: 2026-03-25

## Setup

Benchmarks were run from `scripts/run_modal.py` against the dataset uploaded to the Modal volume `flashinfer-trace`, using `/mnt/mlsys26-contest` inside the container.

Pinned Modal stack:

- `flashinfer-bench @ 0cd5b6e1ed0b5416866d6b81a8295ac2f1e22982`
- `flashinfer-python==0.6.4`
- `torch==2.9.1`
- `numpy==2.4.2`
- `triton==3.5.1`
- `cupti-python>=13`

Evaluation settings were aligned with `EVALUATION.md` for MoE:

- `warmup_runs=10`
- `iterations=50`
- `num_trials=3`
- `use_isolated_runner=True`
- `timeout_seconds=300`
- `atol=1.0`
- `rtol=0.3`
- `required_matched_ratio=0.9`
- `dump_traces=True`
- `resume=True`

## Solution Under Test

Current local solution:

- Name: `fp8_mloop_fused_moe_v6_three_regime_64x64`
- Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

Reference solution:

- Name: `flashinfer_wrapper_9sdjf3`
- Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

## Benchmark Summary

### Current kernel, plain benchmark run

Command:

```bash
modal run scripts/run_modal.py
```

Outcome:

- `19/19` workloads passed
- Average latency: `0.910 ms`
- Average speedup vs reference latency: `32.96x`

Per-workload latencies from the completed run ranged from `0.195 ms` to `5.957 ms`.

### Current kernel, profiling-enabled benchmark run

Command:

```bash
modal run scripts/run_modal.py --profile --ncu
```

Outcome:

- `19/19` workloads passed
- Average latency: `0.978 ms`
- Average speedup vs reference latency: `30.26x`
- Latency range: `0.276 ms` to `5.980 ms`

The representative profiling workload selected by the runner for NCU was:

- `5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d`

### Reference baseline, official-style benchmark run

Command:

```bash
modal run scripts/run_modal.py --solution-json tmp/baseline_moe_solution.json
```

Outcome:

- The baseline did not complete a full result set under the evaluation-aligned isolated-runner configuration.
- Repeated timeout behavior was observed immediately on the first workloads seen in the rerun:
  - `b8f4f012-a32e-4356-b4e1-7665b3d598af`
  - `e05c6c03-5603-4a1c-b34c-dcce0ecaeea4`

This is consistent with the earlier baseline attempt, which also timed out on the first workload under isolated-runner execution.

## Profiling Results

### Torch profiler summary for current kernel

The profiling-enabled run completed and printed a summary block, but it did not contain useful CUDA operation rows. The emitted summary ended with:

```text
PROFILING SUMMARY (Top Operations by CUDA Time)
...
Self CPU time total: 0.467ms
```

That run also emitted CUPTI subscriber warnings:

```text
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
CUPTI initialization failed - CUDA profiler activities will be missing
```

Interpretation:

- The benchmark itself completed successfully.
- `torch.profiler` did not produce a meaningful CUDA breakdown in that run because CUPTI was already contested by profiler tooling in the same process.

## NCU Results

To avoid the `torch.profiler` and Nsight Compute conflict, `scripts/run_modal.py` was updated to support an `--ncu-only` path.

Commands:

```bash
modal run scripts/run_modal.py --ncu --ncu-only
modal run scripts/run_modal.py --solution-json tmp/baseline_moe_solution.json --ncu --ncu-only
```

Representative workload selected in both NCU-only runs:

- `5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d`

Outcome for both current kernel and reference baseline:

- `NCU exit code: 9`
- Nsight Compute connected to the process, but failed to initialize the profiler

Observed error:

```text
==ERROR== Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.
```

Interpretation:

- NCU invocation itself is wired up correctly and reaches the target process.
- On the current Modal environment used here, Nsight Compute cannot actually collect counters because the required profiling library/driver support is not available to the container in a compatible way.
- This failure mode is independent of the kernel under test; it happened for both the local Triton kernel and the reference baseline.

## Overall Takeaways

- The local Triton kernel benchmarked successfully across all `19` official MoE workloads under the `EVALUATION.md`-aligned software settings.
- Measured performance for the local kernel was strong in this environment, with roughly `31x` to `33x` average speedup depending on whether profiler overhead was present.
- The reference FlashInfer baseline did not complete under the same isolated-runner configuration because it timed out very early.
- Nsight Compute could be launched from Modal, but actual counter collection failed with `LibraryNotLoaded`, so there is no valid NCU hardware counter report to compare for either kernel from this environment.

## Recommended Next Step

If you want a true apples-to-apples NCU report for both kernels, the next thing to try is running the same `--ncu-only` flow on an environment where the GPU driver stack explicitly supports Nsight Compute counter collection for containers, or on the official contest image if that exposes the necessary profiling libraries.
