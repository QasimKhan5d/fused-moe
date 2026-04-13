# Fused MoE Benchmark Results

Date: 2026-04-13

## Latest Fresh Run

This file tracks the current benchmark numbers for the submitted kernel, using fresh Modal B200 runs rather than older archived results.

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

- Name: `fp8_mloop_fused_moe_v12_splitk_bm256_safe`
- Definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Author: `KernelEvolve`
- Build language: `triton`
- Entry point: `run`
- Binding: `torch`
- Tagged commit: `submission-v12`

Architecture:

- Based on `gen_14_idea_59aef860_v0_r1` from KernelEvolve v10
- Per-tile BF16 weight dequant in the GEMM inner loop: loads FP8 weight tiles, multiplies by block scale to BF16, then `tl.dot` runs BF16 x BF16 (no predequantized weight buffers)
- Pre-dequantized BF16 hidden states via a separate Triton kernel per call
- Fused gate+up double-dot GEMM1 with SwiGLU epilogue
- Fused routing + top-k + expert counting in a single Triton kernel
- Split-K=4 for tiny workloads (T<16, `num_assignments < 128`) with warps=4, stages=3
- BM=256 for large workloads (`num_assignments >= 8192`) for improved weight L2 reuse
- CUDA graph caching for the full pipeline

Safety: all workspace buffers are persisted between runs but contents are recalculated every call. No cached derived results from a previous call are reused.

Reference baseline:

- `flashinfer_wrapper_9sdjf3`

## Per-Workload Results (Submission Validation Run)

| seq_len | workload | latency_ms | speedup_x |
|---:|---|---:|---:|
| 1 | `e05c6c03...` | 0.121 | 96.31 |
| 7 | `b8f4f012...` | 0.165 | 76.58 |
| 14 | `8cba5890...` | 0.237 | 57.04 |
| 15 | `2e69caee...` | 0.155 | 78.53 |
| 16 | `a7c2bcfd...` | 0.288 | 48.95 |
| 32 | `6230e838...` | 0.321 | 48.56 |
| 52 | `f7d6ac7c...` | 0.302 | 48.26 |
| 53 | `fc378037...` | 0.330 | 48.90 |
| 54 | `76010cb4...` | 0.325 | 48.82 |
| 55 | `81955b1e...` | 0.347 | 46.40 |
| 56 | `4822167c...` | 0.439 | 38.17 |
| 57 | `74d7ff04...` | 0.336 | 49.07 |
| 58 | `e626d3e6...` | 0.335 | 50.26 |
| 59 | `eedc63b2...` | 0.310 | 49.22 |
| 62 | `5eadab1e...` | 0.308 | 49.25 |
| 80 | `8f1ff9f1...` | 0.453 | 38.39 |
| 901 | `1a4c6ba1...` | 0.866 | 25.11 |
| 11948 | `58a34f27...` | 2.313 | 16.21 |
| 14107 | `5e8dc11c...` | 3.200 | 14.65 |

All `19/19` workloads PASSED.

## Change from Previous Submission

submission-v12 adds two optimizations on top of submission-v11's gen_14 baseline:

1. **Split-K=4 for tiny workloads (T<16)**: partitions the K-loop across 4 CTAs with a separate reduce+SwiGLU kernel. Gains +36x on T=1 (57x -> 96x), +15x on T=7 (57x -> 77x). Tuned to warps=4, stages=3 for the Split-K regime after 8 experiments.

2. **BM=256 for large workloads (T>=10000)**: doubles the M-tile size for the `num_assignments >= 8192` regime, improving weight L2 cache reuse. Gains +29% on T=11948 (12.5x -> 16.2x), +37% on T=14107 (10.7x -> 14.7x).

Both changes only affect their targeted workload regimes; medium workloads (T=32-80) are unchanged.

## Submission History

| Tag | Kernel | Mean Speedup | Notes |
|---|---|---|---|
| `submission-v10` | 3-regime hybrid (r33+r14+r34) | ~49.6x (single run) | Cached dequantized weights (unsafe) |
| `submission-v11` | gen_14 standalone | ~45.0x (3-run avg) | Online FP8 dequant, fully safe |
| `submission-v12` | gen_14 + Split-K + BM=256 | ~50.8x | Split-K for tiny, BM=256 for large, safe |
