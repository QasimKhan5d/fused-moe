# Fused MoE Benchmark Results

Date: 2026-04-07

## Latest Fresh Run

This file tracks the current benchmark numbers for the hybrid submission kernel, using a fresh rerun of the current repo state rather than older archived results.

Benchmark command:

```bash
modal run scripts/run_modal.py --full-error-log
```

This run used the same MoE evaluation path described in `EVALUATION.md`:

- Base image: `flashinfer/flashinfer-ci-cu132:latest`
- Added on top: `flashinfer-bench@main` built from source
- Added on top: `cupti-python>=13`
- Dataset: Modal volume `flashinfer-trace`, mounted at `/mnt/mlsys26-contest`
- CLI flags: `--save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --atol 1 --rtol 0.3 --required-matched-ratio 0.9`

## Kernel Under Test

Current solution metadata from `config.toml`:

- Name: `fp8_mloop_fused_moe_v10_hybrid_g33_small_g14_medium_g34_large`
- Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Author: `KernelEvolve`
- Build language: `triton`
- Entry point: `run`
- Binding: `torch`

Hybrid structure:

- Originally generated via `KernelEvolve/results/fused_moe_v10`
- `gen_33` path for very small workloads (`T < 21`)
- `gen_14` path for medium workloads (`21 <= T < 800`)
- `gen_34` path for large workloads (`T >= 800`)

Reference baseline:

- `flashinfer_wrapper_9sdjf3`

## Benchmark Summary

Fresh Modal B200 result:

- `19/19` workloads passed
- Arithmetic mean latency: `0.562 ms`
- Arithmetic mean speedup vs reference: `49.62x`
- Latency range: `0.111 ms` to `3.132 ms`
- Speedup range: `14.80x` to `104.35x`

These figures are computed from the per-workload results printed by the fresh run above.

## Per-Workload Results

| seq_len | workload | latency_ms | speedup_x |
|---:|---|---:|---:|
| 1 | `e05c6c03...` | 0.111 | 104.35 |
| 7 | `b8f4f012...` | 0.185 | 69.96 |
| 14 | `8cba5890...` | 0.258 | 52.69 |
| 15 | `2e69caee...` | 0.153 | 79.36 |
| 16 | `a7c2bcfd...` | 0.271 | 50.67 |
| 32 | `6230e838...` | 0.318 | 51.30 |
| 52 | `f7d6ac7c...` | 0.293 | 49.74 |
| 53 | `fc378037...` | 0.320 | 50.28 |
| 54 | `76010cb4...` | 0.315 | 50.79 |
| 55 | `81955b1e...` | 0.405 | 40.00 |
| 56 | `4822167c...` | 0.422 | 39.53 |
| 57 | `74d7ff04...` | 0.330 | 49.66 |
| 58 | `e626d3e6...` | 0.325 | 51.41 |
| 59 | `eedc63b2...` | 0.302 | 49.61 |
| 62 | `5eadab1e...` | 0.302 | 50.83 |
| 80 | `8f1ff9f1...` | 0.436 | 39.92 |
| 901 | `1a4c6ba1...` | 0.687 | 30.53 |
| 11948 | `58a34f27...` | 2.121 | 17.32 |
| 14107 | `5e8dc11c...` | 3.132 | 14.80 |

## Takeaways

The current hybrid is materially stronger than the old single-kernel result previously documented here. The biggest gains are on the smallest workloads, where the hybrid now clears `100x` on `seq_len=1` and stays very strong through the low-token regime.

The long-sequence workloads remain the weakest part of the profile, but they are still passing and they are much faster than the stale numbers this file previously reported. As of this fresh rerun, the hybrid kernel is validating at roughly `49.6x` arithmetic mean speedup under the official-image Modal harness.
