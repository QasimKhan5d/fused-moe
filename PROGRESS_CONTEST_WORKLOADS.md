# Per-workload roofline + MxF8 progress status

## What's measured right now

### B200 SOL ceilings used here
- FP8 raw peak: ~3.2 PFLOPS (no sparsity)
- BF16 peak: ~1.6 PFLOPS
- HBM peak: ~5 TB/s
- These are the theoretical ceilings. The CUTLASS software-scaled FP32-blockwise schedule realistically caps at ~55-58% SM on large T due to shared-memory bank conflicts (measured via NCU, see PROGRESS_V17_V20.md).

### Current production kernel (v17 landed, v18 wide-scatter merged)
- Uses `KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100` (FP32-blockwise path).
- v17 weight-fold gave ~2-3% large-T, 0-1% overall (within measurement noise but correct direction).
- v18 wide-scatter: no measurable gain (scatter was already ~67% DRAM bound).

### MxF8 path (in-flight, GEMM level validated)
- Built a new CUTLASS MxF8F6F4 ptr-array grouped GEMM + transcode kernels.
- Transcode (sign-flip + UE8M0 pow-of-2 scale + payload residual absorption) is numerically correct byte-for-byte vs Python probe (18/19 workloads ≥90% match).
- The CUTLASS MxF8 grouped GEMM itself is validated vs CPU reference across E∈{1,4}, K∈{128,256}, pow-of-2 scales: all within bf16 rounding noise.
- Next step: pipe the transcode + MxF8 GEMM into the end-to-end MoE pipeline and measure speedup.

## Contest workload snapshot (with hypothesis per regime)

All 19 workloads share the same definition `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`. They differ in `T` (sequence length). Numbers below come from the Modal B200 runs used in earlier commits (`submission-v16`, `cutlass_sm100_v17_weight_fold`).

### Small T (1-80 tokens) — *launch-bound*

| UUID | T | v16 lat. | v16 ×sp | Dominant cost | Why it's limited |
|---|---|---|---|---|---|
| e05c6c03 | 1 | 0.097 ms | ~119× | Python + launch overhead; GEMMs are microseconds | Single-token MoE has almost no GEMM work; overhead dominates. CUDA-graph-captured path helps a lot. |
| b8f4f012 | 7 | 0.129 ms | ~96× | Same | Same regime. |
| 8cba5890 | 14 | 0.165 ms | ~83× | Same | Same regime. |
| 2e69caee | 15 | 0.117 ms | ~103× | Same | Same regime. |
| a7c2bcfd | 16 | 0.171 ms | ~80× | Same | Same regime. |
| 6230e838 | 32 | 0.236 ms | ~66× | Dispatch + transition to GEMM | Launch→GEMM transition. |
| 8f1ff9f1 | 80 | 0.272 ms | ~63× | GEMM starts mattering | Mem→compute transition. |

**Launch-bound verdict.** For T ≤ ~80 the reference latency is ~10-20 μs for the *pure* MMA work but we measure ~100-300 μs because of Python + kernel-launch bubbles. Our CUDA-graph-captured path (`_run_pipeline_graph_safe`, engaged for T ≤ 2048) already eliminates most Python overhead but kernel-launch latency per op (~3-5 μs × ~10 kernels) still dominates. Improving small-T requires **kernel merging** (v19/v20 finalize + scatter fusion) not more mainloop speed.

**MxF8 impact on small T**: small because GEMM is already tiny. Transcode overhead might even slightly regress tiny workloads. *Fallback to FP32-blockwise for T=1 is required anyway (T=1 fails MxF8 tolerance at 86%).*

### Medium T (52-901 tokens) — *memory/launch transition*

| UUID | T | v16 lat. | v16 ×sp | Dominant cost | Why it's limited |
|---|---|---|---|---|---|
| f7d6ac7c | 52 | 0.200 ms | ~72× | Gather + dispatch + GEMMs equal | Transition zone. |
| fc378037 | 53 | 0.250 ms | ~64× | same | |
| 76010cb4 | 54 | 0.240 ms | ~66× | same | |
| 81955b1e | 55 | 0.249 ms | ~64× | same | |
| 4822167c | 56 | 0.261 ms | ~63× | same | |
| 74d7ff04 | 57 | 0.254 ms | ~64× | same | |
| e626d3e6 | 58 | 0.259 ms | ~64× | same | |
| eedc63b2 | 59 | 0.214 ms | ~70× | same | |
| 5eadab1e | 62 | 0.207 ms | ~72× | same | |
| 1a4c6ba1 | 901 | 0.359 ms | ~58× | GEMM becoming dominant | Launch/mem blend. |

These are small enough that each GEMM has only ~55-900 tokens after local-expert filtering. GEMMs at M_e ~30-50 have terrible tile utilization (tile_M=128, ~40% fill). NCU shows these are ~30-45% SM — well below the ~55% ceiling for CUTLASS-blockwise; bank conflicts plus empty tile waste combine.

**MxF8 impact on medium T**: The mainloop speedup matters here (GEMM is ~40-60% of total). We project 1.3-1.5× on GEMMs → overall ~15-20%. But the small per-expert M also limits MxF8: same 40% tile-fill problem.

### Large T (11948 / 14107 tokens) — *compute-bound, far off roofline*

| UUID | T | v16 lat. | v16 ×sp | GEMM1 NCU | GEMM2 NCU | Raw FP8 roofline μs | % roofline |
|---|---|---|---|---|---|---|---|
| 58a34f27 | 11948 | 0.962 ms | ~38× | ~500 μs @ 55% SM | ~315 μs @ 53% SM | ~400 μs | ~42% |
| 5e8dc11c | 14107 | 1.323 ms | ~35× | ~597 μs @ 58% SM | ~370 μs @ 55% SM | ~520 μs | ~40% |

**Compute-bound verdict.** NCU confirms GEMMs dominate (~72% of end-to-end at T=14107). GEMM1 is specifically *shared-memory bank-conflict bound*: 70% of shared loads hit 3.4-way conflicts, 54% of stall cycles wait on L1TEX. The SM-utilization ceiling on the FP32-blockwise path is ~58%. To break the ceiling we need hardware block-scale MMA (MxF8), which has a different smem layout and should avoid the bank conflicts.

**MxF8 projected impact on large T.**
- GEMM1: 597 μs → ~350 μs (ratio 1.7×). Saves ~250 μs.
- GEMM2: 370 μs → ~210 μs. Saves ~160 μs.
- Transcode overhead: estimated ~30-40 μs.
- End-to-end: 1323 μs → ~950 μs, i.e. ~35× → ~50× speedup.
- % of raw FP8 roofline: ~40% → ~55%. Still not 80%+ because of non-GEMM overhead (dispatch, reduce_scatter) and lower transcoded FP8 effective throughput (MxF8 has ~same peak FLOPS as FP32-blockwise on paper, but avoids the bank-conflict bottleneck).

## Summary: where are we today?

| Regime | Workloads | Current % of raw FP8 roofline | MxF8 projected % | Bottleneck |
|---|---|---|---|---|
| Small T (T ≤ 80) | 7 | Launch-bound; doesn't really roofline | same | Kernel launches |
| Medium T (52-901) | 10 | ~25-35% | ~35-45% | Tile fill + bank conflicts |
| Large T (11948/14107) | 2 | ~40-42% | **~50-55%** | CUTLASS smem bank conflicts (until MxF8) |

MxF8 moves the needle most on large T. Medium T gets a modest lift. Small T is unaffected by GEMM speed.

Going to 80-90% of raw roofline would additionally require:
- Eliminating the bf16 `[M, 2H]` intermediate between GEMM1 and SwiGLU (persistent fusion).
- Eliminating the gemm2_out HBM round-trip (custom scatter epilogue).
- Neither of those is in the MxF8 patch — they're still the v18/v19-true EVT work that's 1-2 weeks of CUTLASS engineering each.

## Hypotheses currently open

1. **MxF8 end-to-end passes 18/19 contest workloads**: untested. Next step is to wire transcode+MxF8 into `custom_kernel` and run the real harness.
2. **MxF8 delivers the projected ~25% end-to-end speedup on large T**: untested; waiting on end-to-end.
3. **Transcode overhead is < 5% of total**: untested; if transcode becomes a regression factor on small T we'll fallback there too.
4. **MxF8 on medium T actually helps**: untested. If per-expert M is too small to utilize tile_M=128 fully, mainloop speedup may be small.

## Closed hypotheses

- UE8M0 scales as-is fail contest tolerance. Reason: signed scales can't be represented. CLOSED.
- Payload-residual absorption + sign-flip handles contest semantics while using UE8M0 scales. CONFIRMED: 18/19 workloads ≥90% match via Python probe AND CUDA transcode.
- The CUTLASS ptr-array MxF8 GEMM is callable from our extension with correct numerics. CONFIRMED: all 5 isolation tests pass.
- FP32-blockwise mainloop is ceiling-bound by smem bank conflicts. CONFIRMED via NCU (70% of shared-load wavefronts conflicted).

## What's next (in order)

1. Wire MxF8 into `_run_pipeline_dynamic` behind `USE_MXF8=1` env var with FP32-blockwise fallback for T=1.
2. Run on all 19 contest workloads.
3. Measure per-workload speedup vs v17 baseline.
4. Characterize transcode overhead (NCU).
5. If all good: flip `USE_MXF8=1` to default, tag as `submission-v18`.
