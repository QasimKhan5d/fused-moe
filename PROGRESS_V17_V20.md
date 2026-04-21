# v17–v20 Progress & Plan (rigorous)

Target: push the contest MoE kernel as close as possible to 80–90% of B200 raw FP8 roofline on all 19 workloads, using insights from DeepGEMM MegaMoE, FlashInfer CuTeDSL MoE, SGLang NVFP4 MoE, and CUTLASS SM100.

## Branch B (ceiling-breaker) — STRUCTURALLY CLOSED

Investigated thoroughly. Cannot use `tcgen05.mma.kind.block_scale` with contest FP32 scales because:

1. CUTLASS `ScaleFormat` enum only allows `UE8M0` and `UE4M3`:
```216:233:/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/cutlass/include/cute/arch/mma_sm100_desc.hpp
enum class ScaleFormat : uint8_t {
  UE4M3 = 0,
  UE8M0 = 1,
};
```
   - `UE4M3` requires FP4 A/B operands (NVF4 path only); not usable for FP8 MoE.
   - `UE8M0` is the only scale format for the FP8 (MXF8) block-scale MMA.

2. UE8M0 is **exponent-only, unsigned** (8 bits exponent, 0 mantissa, 0 sign).

3. Empirical inspection of contest scales (`experiments/custom_mma/inspect_scale_values.py`):
   - hs_scale: 395004 pos / 394988 neg (essentially 50/50)
   - gemm1_w_scale: 28407 pos / 28937 neg
   - gemm2_w_scale: 14325 pos / 14347 neg
   - Using `abs(scales)` gives only 48% match vs reference → **sign is semantic**.

4. Therefore UE8M0 cannot represent contest scales without losing sign, and the contest kernel output depends on sign.

5. Additional confirmations from rigorous transcode probe (`experiments/custom_mma/mxf8_transcode_rigorous.py`):
   - baseline (no transform): 100% match (framework valid).
   - Even SINGLE-tensor scale-only UE8M0 rounding fails (~55% at large T).
   - Residual-in-payload absorption gives no lift over scale-only — FP8 re-round erases the gain.
   - Residual uniformity along K: `std_log2(r) ≈ 0.22` (not uniform, not constant), so per-row correction can't collapse it.

**Verdict**: MXF8 hardware fast path is *unreachable* for contest semantics. The only way to reach 80–90% of raw FP8 roofline would require a contest rule change (e.g., positive-only scales or FP4 operands).

## What Branch A can actually achieve

The CUTLASS `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` schedule (what we're using) is the software-scaled FP8×FP8-with-FP32-scales path. Current large-T (T=14107) GEMMs are at ~52% SM utilization, which is close to this schedule's real ceiling.

Realistic end-to-end target on this mainloop: **~50–60% of the raw FP8 roofline**, by eliminating HBM round-trips, merging kernels, and reducing launch overhead.

## Branch A status — what actually happened

### v17 — routing-weight fold into GEMM2 A-scale
- **Implemented**: `swiglu_fp8_requant_weighted` + `reduce_scatter_unweighted` / `_unweighted_prebucketed`.
- **Validated**: 19/19 workloads pass on Modal B200, 70.71x mean (v16 was 70.67x).
- **Measured gain at large T**: 5e8dc11c dropped from 1.333 ms to 1.304 ms (~2–3%), 58a34f27 from 0.962 ms to 0.957 ms (~0.5%). Within variance band but direction is consistent.
- **Honest assessment**: Gave the predicted ~5–15 μs on large T. Clean micro-win.

### v18 — wide vectorized reduce_scatter
- **Implemented**: `reduce_scatter_unweighted_kernel` rewritten to use `uint4` (128-bit) loads, 8 bf16 processed per iteration, TB=128.
- **Validated**: 19/19 pass.
- **Measured gain**: none visible vs v17 (5e8dc11c: 1.306 ms, 58a34f27: 0.963 ms).
- **Reason**: the prior bf162 kernel was not actually BW-bound per-warp; TB=128 reduced occupancy; fp conversions inside the inner loop were the bottleneck, not memory bus width.
- **Next step if pursued**: keep the wider kernel but recover occupancy at TB=256 with 2 uint4 per thread.

### v19 — `FUSED_DISPATCH_GATHER` investigation
- **Measured**: Turning on `FUSED_DISPATCH_GATHER=1` gave ~3% slowdown on large T (5e8dc11c: 1.343 ms vs 1.304 ms baseline).
- **Root cause** (analyzed, not yet NCU-verified): the fused kernel has 8 warps/block, one warp per assignment. At 12.5% locality (32 local experts / 256 global), ~7 of 8 warps per block exit early → massive warp-level idle time. Each block does the work of only 1–2 valid assignments but has the register pressure of all 8.
- **Left off by default**. Could be fixed by having each block pre-filter assignments, but gain is small.

### v19 — true GEMM2 finalize-fusion EVT epilogue
- **Not implemented**. Would require a custom CUTLASS `CollectiveEpilogue` with atomic scatter-to-dynamic-destination. Estimated 5–7 days of focused CUTLASS work. Deferred pending higher-ROI levers.

### v18 — paired-column SwiGLU EVT fusion
- **Not implemented**. Same scope issue: requires custom EVT with cross-tile amax (or acceptable per-tile scale, which affects downstream precision).

### v20 — persistent wavefront scheduler
- **Not implemented**. DeepGEMM MegaMoE pattern, multi-week effort.

## What I've been honest about

- v17 gave ~predicted gain within noise.
- v18 (wide scatter) did NOT give predicted gain. Kernel wasn't BW-limited at the operation level.
- v19 (fused dispatch gather) regressed for an identifiable reason (warp idle at non-local).
- Branch B (MXF8) is definitively dead for a STRUCTURAL reason, not a precision one.

## Outstanding opportunities (ranked by expected value / effort)

1. **NCU-profile current kernel** to identify the real bottleneck distribution at T=14107. (In flight.)
2. **Investigate small-T regime (T<2048)**: the graph-safe path runs already-fused kernels; gains there would come from reducing CUDA-graph overhead or the short GEMM1 time.
3. **Custom GEMM2 finalize-fusion EVT**: largest architectural win available on Branch A. Week-scale engineering.
4. **Merged count+scan+place in one persistent kernel**: saves ~3 kernel launches (~10 μs). Easy but tiny.
5. **Tune CUTLASS mainloop stages / cluster / swizzle parameters one more time on v17 baseline** to confirm nothing shifted.

## Known dead ends

- Extra CUTLASS tile configs (tile_K=256, 2SM variants). Previously tested — all regressed; removed.
- `FUSED_DISPATCH_GATHER` by default. Regressed ~3%.
- MXF8 transcode (all variants). Fundamentally blocked.
- UE4M3 scales with FP8 operands. Not a valid CUTLASS combination.

## NCU profile results (T=14107, v17 kernel)

| Kernel | Duration (μs) | SM % | DRAM % | Notes |
|---|---|---|---|---|
| fused_gather_hidden_scales | 32.5 | 19 | 61 | BW-dominant, ~BW-optimal |
| get_group_gemm_starts (×2) | ~8.5 | 0 | 0 | Launch bubble |
| **GEMM1 (CUTLASS Blockwise 1Sm)** | **597.2** | **58.6** | 25.9 | Bank-conflict bound (see below) |
| swiglu_fp8_requant_weighted | 36.9 | 45.6 | — | Register-cached, moderately tight |
| **GEMM2 (CUTLASS Blockwise 1Sm)** | **370.0** | **55.5** | — | Same bank-conflict ceiling as GEMM1 |
| token_bucket_count | 4.2 | 0.6 | — | Launch bubble |
| token_bucket_scan | 16.1 | 0.1 | — | Single-block scan, serial |
| token_bucket_place | 4.6 | 0.7 | — | Launch bubble |
| reduce_scatter_unweighted | 76.7 | 28.7 | **67.0** | Near BW-bound |
| Total kernel time | ~1147 | | | |
| End-to-end (measured) | ~1300 | | | ~150 μs launch/python overhead |

## Why GEMM1 is the real ceiling on this path

NCU `Memory Workload Analysis Tables` for GEMM1 shows:
- Shared loads: 3.4-way bank conflict, **70% of wavefronts affected**.
- Shared stores: 7.2-way bank conflict, **56% affected**.
- Warp stall reason (54% of stall cycles): L1TEX scoreboard dependency, i.e., waiting on shared-memory loads.
- Active warps per scheduler: 2.25 / 16 (14% occupancy by design, to give each warp more registers for MMA).
- Issue slots busy: 41.8%. 65.9% of cycles have no eligible warp.

CUTLASS's `Sm100BlockwiseScaleConfig<1, 128, 128, K, K>` shared memory layout produces these conflicts for our specific MoE problem shape (M per expert ≈ 440, tile_M=128). Resolving this requires modifying CUTLASS's `SmemLayoutAtom` (fork). Not a tuning knob.

GEMM1 estimated max speedup from resolving bank conflicts: ~30–40%. If achievable that would drop GEMM1 from 597 μs to ~420 μs. Would require forking CUTLASS.

## What's left on Branch A

Real lever (all small/modest vs the GEMM ceiling):

- **Merged count+scan+place+reduce**: theoretical savings 20–30 μs from the 25 μs of bucket helpers and some launch overhead. ~2% of total.
- **Persistent-style scan**: parallelize the serial `token_bucket_scan_kernel` across many SMs. Save ~12 μs.
- **Better SwiGLU vectorization**: maybe 5–10 μs.
- **CUDA graphs for the large-T dynamic path**: save ~100 μs of Python/launch. Higher risk due to shape variability.

Total Branch A low-risk improvements: ~30–70 μs (2–5% total) without custom CUTLASS work.

Real large-step-change levers (all require week-scale work):

- **Custom CUTLASS EVT for GEMM2 finalize-fusion**: save 77 μs.
- **Custom CUTLASS EVT for GEMM1 SwiGLU-fusion**: save ~20 μs.
- **Fork CUTLASS to fix the bank-conflict smem layout**: save ~150 μs on GEMM1.
- **Persistent kernel that fuses GEMM1+SwiGLU+GEMM2+scatter**: save up to 250 μs.

## Honest path forward

The contest score is constrained by the software-scaled FP32-blockwise mainloop that CUTLASS gives us. **80-90% of raw FP8 roofline is structurally unreachable** because:

- Hardware block-scaled MMA requires UE8M0 scales (structural constraint);
- Contest scales are signed fp32 (structural constraint);
- These two are incompatible (not a precision issue).

The realistic ceiling is ~55–65% of raw roofline, and we are already close to it.

To get the remaining ~10% on Branch A would require custom CUTLASS EVTs (1–2 weeks of CUTLASS C++ work). Beyond that, customs CuTe persistent kernels.
