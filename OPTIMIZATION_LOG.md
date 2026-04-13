# Fused MoE Kernel — B200 Optimization Log

Target: NVIDIA B200 (SM100 / Blackwell), MLSys 2026 FlashInfer contest.
Harness: `flashinfer-bench` with `atol=1, rtol=0.3, required_matched_ratio=0.9`.
Environment: `flashinfer/flashinfer-ci-cu132:latest` on Modal (torch 2.12.0+cu132, triton 3.6.0, CUDA 13.2).

## Current Kernel Architecture

- **GEMM1**: FP8 activations + FP8 weights → dequantize both to BF16 (with scales applied) → BF16×BF16 dot → FP32 accumulator → SiLU in FP32 → store C_act as BF16
- **GEMM2**: BF16 C_act × FP8 weights (dequantized to BF16) → BF16×BF16 dot → FP32 accumulator → atomic_add to FP32 output → finalize kernel casts to BF16
- **Dispatch**: Triton-based count/scan/scatter for expert assignment, CUDA graph for routing

### Current Tile Config (well-tuned, see tuning section below)

| Parameter | GEMM1 | GEMM2 |
|---|---|---|
| BLOCK_M | 64 (large) / 16 (small) | 64 (large) / 16 (small) |
| BLOCK_N | 64 | 64 |
| BLOCK_K | 128 | 64 |
| num_warps | 4 | 4 (8 for huge regime) |
| num_stages | 3 | 3 |

### Current Performance (19/19 workloads pass)

| Workload regime | Avg latency | Avg speedup vs reference |
|---|---|---|
| Small (T < 901, 16 workloads) | 0.324 ms | ~47x |
| Large (T >= 901, 3 workloads) | 4.524 ms | ~10x |
| Overall (19 workloads) | 0.998 ms | ~43x |

## Proven Constraints

### FP8 MMA is not viable for GEMM1 with this kernel

The contest uses block-scaled FP8 (e4m3 values, FP32 per-block scales, group size 128). Three paths to FP8 tensor cores have been tried and all failed:

1. **Raw FP8 dot + post-scale** (`tl.dot(a_fp8, w_fp8) * sa * sw`): Stochastically fails the harness for large workloads (T >= 901). The post-multiply amplifies FP8 rounding errors through SwiGLU (`gate * SiLU(up)`), pushing ~10-11% of output elements past `atol=1, rtol=0.3`. With BLOCK_M=64, it fails. With BLOCK_M=16, it passes but at the same speed as BF16 MMA. Confirmed across multiple runs — the failure is at the statistical boundary of `required_matched_ratio=0.9`.

2. **Mixed BF16×FP8 dot** (`tl.dot(a_bf16, w_fp8)`): Triton compilation error — `Unsupported rhs dtype fp8e4nv`. Triton requires both operands to have the same type for `tl.dot`.

3. **`tl.dot_scaled` (MXFP8 native)**: The contest's FP8 format uses FP32 per-block scales with group size 128. `tl.dot_scaled` expects MXFP8 with e8m0 scales and group size 32. Re-quantizing scales to match gave incorrect numerical results. Additionally, `tl.dot_scaled` requires Triton ≥ 3.6 which conflicts with the contest image's torch/triton stack.

### The dequantize-then-dot approach is the correct design

The current approach dequantizes both operands to BF16 with scales applied before the dot:
```python
a_dq = (a.to(tl.float32) * sa).to(tl.bfloat16)
w_dq = (w.to(tl.float32) * sw).to(tl.bfloat16)
acc += tl.dot(a_dq, w_dq)
```
No post-dot scale multiplication. The accumulator sees properly scaled values. This passes all 19 workloads reliably with no precision branches.

### NCU profiling does not work on Modal B200

Nsight Compute (`ncu`) fails with `LibraryNotLoaded` in the Modal container environment, even with the official contest image. This means we cannot get low-level metrics (occupancy, memory throughput, compute utilization, stall reasons) to guide optimization. All tuning is blind.

## Optimization History

### Changes that improved performance (applied)

| Change | Impact | Mechanism |
|---|---|---|
| Dequantize-both GEMM1 (remove BF16 branch) | Fixed 3 failing workloads, ~15% faster on large vs branched Option 2 | Single code path, no post-dot scale multiply, compiler generates tighter code |
| C_act FP32 → BF16 | -5% small, -8% to -12% large | Halves memory traffic between GEMM1 and GEMM2 (the dominant intermediate tensor) |
| GEMM2 FP32 dot → BF16 dot | Included in C_act change | Uses BF16 tensor cores instead of TF32/FP32 CUDA cores |
| GEMM1 store: cast SiLU output to BF16 | Included in C_act change | Writes half the data to C_act |

### Changes that were rejected (tried and measured)

| Change | Result | Why it failed |
|---|---|---|
| GEMM2 BLOCK_K=64→128 | +2% to +9% worse | Register pressure: [128,64] BF16 weight tile = 16KB/warp. Spills to local memory. |
| GEMM1 BLOCK_N=64→128 | +22% to +28% worse | GEMM1 does TWO dots per K-iteration (gate + up). Doubling N doubles accumulator + weight register usage for both. |
| GEMM2 BLOCK_N=64→128 | Mixed: -2% on largest, +12% on medium | Inconsistent. Larger N helps largest workloads but hurts medium ones. Not a net win. |
| GEMM1 num_stages=3→4 | Flat / noise | 3 stages already saturates pipelining benefit. 4th stage uses SMEM without improving overlap. |
| GEMM1 warps=4→8 (all regimes) | -3% large, +6% small | 8 warps helps hide memory latency for large workloads but hurts small workloads (BLOCK_M=16 with 8 warps = too little work per warp, lower occupancy). |
| GEMM1 warps=8 (large regime only) | -0% to -3% large | Marginal. Saves ~0.6% on overall mean. Not worth the added complexity. |
| GEMM1 BF16 dequant (skip FP32 intermediate) | +15% to +18% worse | Same op count (2 casts + 1 mul both ways), but Triton generates worse code for the BF16 multiply chain — likely different register allocation or instruction scheduling. |
| Atomic elimination in GEMM2 (write-then-reduce) | +6% to +8% worse on large | Scattered reads in reduction kernel worse than L2-cached atomic writes. B200's L2 handles atomic_add efficiently. |
| `tl.dot_scaled` for GEMM2 (naive e8m0 conversion) | INCORRECT_NUMERICAL | FP32 block scales can't be losslessly converted to e8m0. Truncating mantissa introduces up to ~50% scale error per block. |
| `tl.dot_scaled` for GEMM2 (weight baking, separate quant kernel) | +18% worse (8.83 vs 7.48 ms) | Re-quantized weights to MX format (correct numerically, PASSED). Extra `quantize_fp8_kernel` for C_act adds a full memory pass. |
| `tl.dot_scaled` for GEMM2 (weight baking, fused quant in GEMM1) | +15% worse (8.62 vs 7.48 ms) | Fused FP8 quantization into GEMM1 output (no separate kernel). Still slower — `tl.dot_scaled` itself underperforms BF16 dot on Triton 3.6.0. |
| FP8 C_act + BF16 dot hybrid (fused quant, original weight dequant) | +4-7% worse across all | Stores C_act as FP8 (half bandwidth) but dequants to BF16 in GEMM2 for BF16 dot. The e8m0→FP32→BF16 dequant in GEMM2 costs more than the bandwidth savings. |
| Manual FP8 dot + power-of-2 post-scale (BLOCK_K=32) | +47% worse (11.0 vs 7.48 ms) | Uses standard `tl.dot(fp8, fp8)` with BLOCK_K=32 (=MX_GROUP) and post-multiplies by combined e8m0 scale (exact power-of-2). Loop overhead of 224 iterations (vs 112) + per-iteration scale loads/exp2/multiply dominates. |
| Manual FP8 dot, BLOCK_K=64, 2 sub-dots of 32 | +106% worse (15.4 vs 7.48 ms) | Inner loop over 2 MX groups prevents Triton from pipelining loads across K-iterations. 4 loads per outer iter vs 2. Worst of all FP8 attempts. |
| Warp specialization (`enable_warp_specialization`, `num_consumer_groups`) | NOT AVAILABLE as launch kwarg | Probed CUDAOptions in Triton 3.6.0 — not exposed as kernel launch parameter. |
| Warp specialization via `tl.range(warp_specialize=True)` on K-loops | +238% worse (25.3 vs 7.48 ms) | Correct API found in Triton's reference MoE kernel. But their kernel uses TMA tensor descriptors which cleanly separate into producer/consumer warps. Our manual pointer arithmetic loads can't be cleanly partitioned — the compiler generates a bad schedule that serializes instead of overlapping. Warp specialization requires TMA to be effective. |

### Key insight: GEMM1's double-dot structure constrains tile sizes

GEMM1 computes both gate and up projections in the same kernel:
```python
acc_gate += tl.dot(a_dq, w_gate_dq)   # [M, N] accumulator #1
acc_up += tl.dot(a_dq, w_up_dq)       # [M, N] accumulator #2
```
This means every tile-size increase has ~2x the register cost compared to a standard GEMM. The current BLOCK_N=64 is the sweet spot — going to 128 blows past the register budget.

## Known Performance Ceiling

The dominant bottleneck is **GEMM1 running at BF16 tensor core throughput instead of FP8**. On B200:
- FP8 MMA: ~2x throughput of BF16 MMA
- We run BF16 because FP8 + post-scale is numerically unstable (proven above)
- The dequant overhead (6 element-wise ops per K-iteration) adds to this gap

The secondary bottleneck is **`tl.atomic_add` in GEMM2** for multi-expert output reduction (TOP_K=8 experts per token).

### CUDA graph the full compute pipeline (applied)

Wrapped the entire compute pipeline (assign_w → dispatch → GEMM1 → GEMM2 → finalize) in a CUDA graph, captured per unique T. Routing was already graphed separately.

| Metric | Before | After | Improvement |
|---|---|---|---|
| Small workloads avg (T<901) | 0.419 ms | 0.324 ms | **-23%** |
| Large workloads avg (T>=901) | 4.499 ms | 4.524 ms | flat |
| Overall avg (19 workloads) | 1.103 ms | 0.998 ms | **-10%** |
| Mean speedup | ~32x | ~43x | **+34%** |

Eliminated ~0.1 ms of per-kernel launch overhead per invocation. Huge win for small workloads (20-36% faster) where launch overhead dominated. Large workloads unchanged (compute-dominated).

### Atomic elimination in GEMM2 (rejected)

Replaced `tl.atomic_add` with per-assignment BF16 output buffer + gather-sum reduction kernel. Required adding a reverse mapping (`assign_map`) during dispatch_scatter. Initial attempt caused GPU MMU faults due to non-local expert assignments leaving `assign_map` uninitialized; fixed with `assign_map.fill_(-1)` and conditional skip in reduction.

| Metric | Before (atomic_add) | After (reduce kernel) |
|---|---|---|
| Small workloads | flat | flat |
| Large workloads | 7.48 / 5.05 ms | 7.91 / 5.46 ms (+6-8% worse) |

Atomic contention was NOT the bottleneck on B200. The reduction kernel's scattered reads (8 non-contiguous rows via `assign_map`) were worse than the L2-cached atomic writes. B200's L2 handles atomic add efficiently.

## Post-Submission Summary (submission-v8)

Submitted as `submission-v8` on 2026-03-31. Validated on Modal B200 with the official `flashinfer/flashinfer-ci-cu132:latest` image (torch 2.12.0, triton 3.6.0, CUDA 13.2).

| Metric | v7 baseline (old) | v8 submitted |
|---|---|---|
| Pass rate | 17-19/19 (stochastic) | **19/19 (stable)** |
| Mean latency (all 19) | ~1.03 ms | **1.000 ms** |
| Mean speedup | ~27x | **42.13x** |
| Small workloads avg (T<901) | ~0.50 ms | **0.33 ms** |
| Large workloads avg (T>=901) | ~5.0 ms | **4.51 ms** |

Key improvements that stuck:
1. Branchless BF16 dequant-then-dot GEMM1 (fixed correctness, simplified code)
2. BF16 C_act intermediate (halved memory traffic)
3. BF16 GEMM2 dot (replaced TF32/FP32 with BF16 tensor cores)
4. Full-pipeline CUDA graph (eliminated inter-kernel launch overhead)

Key experiments that were rejected (8 FP8 variants, 6 tile-tuning variants, 2 warp-spec variants, 1 atomic-elimination variant). All documented above.

## Safety Fix (submission-v11)

The previous kernel (submission-v10) cached dequantized BF16 weight tensors across calls via `_dq_cache`. Contest rules require buffer contents to be recalculated every run. The `gen_14` architecture avoids this entirely — it loads FP8 weight tiles and dequantizes to BF16 per tile inside the GEMM inner loop on every call. No derived results are reused across invocations.

A/B testing (3 runs each, Modal B200) confirmed no meaningful performance difference between the safe gen_14 kernel and a hybrid that graph-captured the dequant work, so the simpler gen_14 standalone was chosen.

## Split-K for Small Workloads (submission-v12 candidate)

Added Split-K=4 on GEMM1 for the smallest workloads (T<16, `num_assignments < 128`). Each of the 4 CTAs computes a K-slice of both gate and up projections, writes FP32 partials to workspace, then a separate reduce+SwiGLU kernel sums partials and applies `SiLU(up) * gate`.

### Split-K experiment results (8 variants, 2 rounds run in parallel on Modal B200)

**Round 1 — Split-K range and factor:**

| Variant | Config | T=1 | T=7 | T=15 | T=32 | T=55 | Sample Mean |
|---|---|---|---|---|---|---|---|
| BASE | SK=4, T<16 only | 94.2 | 74.6 | 73.8 | 52.7 | 53.0 | 55.0 |
| EXP1 | SK=2, all T<128 | 81.0 | 78.7 | 76.5 | 42.8 | 40.7 | 50.9 |
| EXP2 | SK=4, all T<128 | 89.1 | 70.1 | 69.3 | 42.9 | 43.0 | 50.3 |
| EXP3 | SK=8, T<16 only | 90.7 | 68.0 | 72.0 | 51.1 | 51.5 | 52.8 |

Conclusions:
- Extending Split-K to medium workloads (T=32, T=55) causes 10-12x regression — the reduction kernel overhead dominates when there are enough M-tiles for parallelism.
- SK=8 gives each CTA only 4 K-iterations (2048/64/8), too little work — launch overhead dominates.
- SK=4 with tight cutoff (T<16 only) is optimal.

**Round 2 — Tile size tuning for the Split-K regime:**

| Variant | Config | T=1 | T=7 | T=15 | T=32 | T=55 | Sample Mean |
|---|---|---|---|---|---|---|---|
| BASE | BN=64, BK=64, w=2, stg=2 | 94.2 | 74.6 | 73.8 | 52.7 | 53.0 | 55.0 |
| EXP4 | BN=128, w=4 | 91.0 | 67.3 | 68.9 | 50.2 | 50.7 | 52.1 |
| EXP5 | BK=128 | 96.6 | 76.1 | 75.5 | 48.7 | 38.3 | 53.0 |
| **EXP6** | **w=4, stg=3** | **97.0** | **77.3** | **80.6** | 50.9 | 52.2 | **56.3** |

Conclusions:
- BN=128 hurts small workloads (double-dot register pressure, same constraint as noted in main tuning section).
- BK=128 helps T=1 slightly but causes severe regression on T=55 (register spill).
- **Winner: `num_warps=4, num_stages=3`** — more warps hide memory latency in the shorter K-loop, extra pipeline stage overlaps loads. Gains +2.8x on T=1, +6.9x on T=15, flat on medium/large.

### Final Split-K performance

| Metric | gen_14 baseline | + Split-K (v1) | + Split-K tuned (v2) |
|---|---|---|---|
| T=1 speedup | ~58x | 94x | **99x** |
| T=7 speedup | ~57x | 75x | **77x** |
| T=15 speedup | ~63x | 74x | **81x** |
| Medium (T=32-80) | ~50x | ~50x | ~49x |
| Large (T=901-14107) | ~16x | ~16x | ~16x |
| **Mean (19 workloads)** | **45.0x** | **48.8x** | **49.1x** |

Split-K is now exhausted. The remaining headroom is in the large workloads (T=901 at 24x, T=11948 at 12.5x, T=14107 at 10.7x).

## NCU Profiling (T=14107 on B200)

Successfully ran NCU `--set full` on Modal B200 after fixing `flashinfer_bench` import paths (`entities.*` moved to `data.*` in newer versions) and tuning `--launch-skip 30` to skip warmup kernels.

### GEMM1 (`gemm1_mloop_kernel`, grid=(16, 32, 1)x(256, 1, 1))

| Metric | Value |
|---|---|
| Duration | 1.64 ms per call |
| Compute (SM) Throughput | 34.8% |
| Memory Throughput | 36.4% |
| DRAM Throughput | 21.3% |
| HBM Bandwidth achieved | 1.64 TB/s |
| L2 Hit Rate | 45.3% |
| Highest pipeline | TC (Tensor Core) at 34.8% |
| Register spilling | **1.28M requests, 100% overhead** |
| Uncoalesced loads | 22.1/32 bytes per sector |
| NCU verdict | **Latency-bound** (neither compute nor memory near peak) |

Root cause: the double-dot gate+up structure (`acc_gate` + `acc_up`, two `[BM, BN]` FP32 accumulators) exhausts the register file. Spills go through L2, adding latency to every K-iteration.

### GEMM2 (`gemm2_mloop_kernel`, grid=(56, 32, 1)x(256, 1, 1))

| Metric | Value |
|---|---|
| Duration | 1.34 ms per call |
| Compute (SM) Throughput | 22.2% |
| Memory Throughput | 29.6% |
| DRAM Throughput | 9.1% |
| HBM Bandwidth achieved | 695 GB/s |
| L2 Hit Rate | 76.5% |
| Highest pipeline | TC (Tensor Core) at 22.2% |
| Register spilling | None |
| NCU verdict | **Latency-bound** |

GEMM2 has excellent L2 hit rate but only 22% compute utilization. The `atomic_add` output pattern and scattered token access via `sorted_token_ids` are likely causing stalls.

## BM=256 for Large Workloads (applied)

Increased BLOCK_M from 128 to 256 for the `num_assignments >= 8192` regime (T=11948, T=14107). Larger M-tiles keep expert weight tiles in L2 longer, reducing redundant HBM fetches.

| Workload | BM=128 | BM=256 | Delta |
|---|---|---|---|
| T=901 | 25.8x | 25.2x | -0.6x (not in this regime) |
| T=11948 | 12.5x | **16.2x** | **+3.7x (+29%)** |
| T=14107 | 10.7x | **14.7x** | **+4.0x (+37%)** |
| Overall mean | 49.1x | **50.8x** | **+1.7x (+3.4%)** |

BM=512 crashes (exceeds register/SMEM budget). BM=256 is the sweet spot.

## Persistent Kernel Rewrite (rejected)

Attempted to convert GEMM1 and GEMM2 from grid-per-expert M-loop kernels into persistent kernels using `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`. Two approaches tested:

### Phase 1A: Persistent fused gate+up (BM=64, BN=64)

Smaller tiles to avoid register spilling. Persistent scheduling to compensate via more tiles per CTA.

| Workload | Baseline | P1A Persistent | Delta |
|---|---|---|---|
| T=14107 | 14.4x | **4.73x** | **-67%** |
| T=901 | 25.8x | 24.8x | flat |
| Small/medium | unchanged | unchanged | — |

### Phase 1B: Persistent split gate/up (BM=128, BN=128)

Split GEMM1 into separate gate and up projection kernels (single accumulator each, no spilling) plus a SwiGLU fusion kernel. Each projection uses full BM=128, BN=128.

| Workload | Baseline | P1B Persistent | Delta |
|---|---|---|---|
| T=14107 | 14.4x | **6.78x** | **-53%** |
| T=901 | 25.8x | 25.4x | flat |
| T=55 | 49x | 51.8x | +5% |

### Why persistent scheduling failed

The MoE dispatch pattern is fundamentally incompatible with Triton's `tl.range(flatten=True)` pipelining:

1. **Variable token counts per expert**: Each expert gets a different number of tokens (`count = offsets[eid+1] - offsets[eid]`). This creates a dynamic inner M-loop `range(0, count, BLOCK_M)` inside the persistent outer loop.

2. **`continue` not supported**: Triton 3.6.0 does not support `continue` inside `tl.range`. The workaround (`if count > 0:` guard) creates divergent branches that prevent load pipelining across tile boundaries.

3. **Standard persistent matmul assumes uniform work**: The Triton reference persistent matmul and the PyTorch grouped GEMM blog both assume fixed (M, N, K) per tile. The MoE pattern has variable M per expert, which breaks the pipelining model.

4. **The real fix would require a flat tile list**: Pre-computing all valid (expert, m_tile, n_tile) combinations into a flat index buffer that the persistent kernel iterates over uniformly. This eliminates the dynamic M-loop but adds a host-side dispatch kernel to build the tile list — essentially a different kernel architecture.

Phases 2 (TMA) and 3 (warp specialization) were cancelled since they depend on a working persistent base.

## Split GEMM1 Gate/Up (rejected)

Hypothesis: NCU showed 1.28M register spill requests with 100% overhead in the fused GEMM1 kernel due to the double-dot gate+up accumulator structure (two `[BM, BN]` FP32 accumulators). Splitting into separate gate and up kernels (single accumulator each) should eliminate spilling entirely.

### Results

| Config | T=14107 | T=11948 | T=901 |
|---|---|---|---|
| Fused BM=256 (baseline) | **14.4x** | **16.2x** | 25.2x |
| Split BM=256 | 12.1x | — | 24.4x |
| Split BM=512 | 12.6x | 11.2x | 25.2x |

Split is **worse** on the exact workloads where spilling was the bottleneck.

### Why the split lost

1. **Activation loads are the expensive part, not the spills.** The fused kernel loads each `[BM, BK]` activation tile once and uses it for both gate and up projections. The split loads it twice — once per kernel. At BM=256, K=2048, BK=64, that's 32 activation tile loads per M-block, doubled to 64 in the split. The activation tensor is scattered via `sorted_token_ids`, so each load is an irregular gather that hits DRAM.

2. **Register spills go through L2, not DRAM.** NCU showed 71.8% of local loads hit L2. The spill penalty is L2 latency (~100 cycles) not DRAM latency (~400 cycles). The activation sharing in the fused kernel saves a full DRAM round-trip per K-iteration, which outweighs the L2 spill cost.

3. **Three kernel launches** (gate + up + SwiGLU) add dispatch overhead and prevent cross-kernel optimization.

The fused double-dot structure is architecturally correct for this workload. The NCU spilling metric is misleading: "100% overhead" means extra instructions relative to zero spills, not a 2x latency penalty. The L2 absorbs the spill traffic.

## Sorted Dispatch for GEMM2 Coalescing (rejected)

Hypothesis: GEMM2's 22% compute utilization despite 76.5% L2 hit rate suggests warp stalls from scattered `C_act` reads via `sorted_token_ids`. Sorting token IDs within each expert's dispatch slice should improve memory coalescing.

Implementation: Triton kernel builds sort keys (`expert_id * T_max + token_id`), PyTorch `argsort` + `index_select` reorders `sorted_token_ids` and `sorted_weights` to be ascending within each expert.

### Result

All workloads failed `INCORRECT_NUMERICAL` (abs_err > 300K) even without CUDA graph capture. The correctness failure persisted after fixing int32/int64 type mismatches in the sort key kernel. Root cause not fully diagnosed — likely a subtle interaction between the `index_select` gather and the routing buffer view semantics.

Abandoned: even if corrected, the expected gain is marginal since GEMM2's primary bottleneck is `atomic_add` contention on the output tensor (TOP_K=8 experts per token), not the `C_act` read coalescing.

## Current Best: 50.8x

The kernel has been optimized through:
- Online FP8 weight dequant (BF16 x BF16 tensor core path)
- Split-K=4 for tiny workloads (T<16) with warps=4, stages=3
- BM=256 for large workloads (T>=10000) for weight L2 reuse
- CUDA graph caching for the full pipeline
- Fused routing + top-k + expert counting in a single Triton kernel

This appears to be near the ceiling for the Triton M-loop kernel architecture on B200.

## What Has NOT Been Tried (Next Steps)

1. **Flat tile list + persistent kernel**: Pre-compute a 1D tile list of all valid (expert, m_tile, n_tile) triples in a dispatch kernel. The persistent kernel iterates this list with uniform work per tile, eliminating the dynamic M-loop. This is the only path to make persistent scheduling work for MoE with variable expert counts.

2. **Revisit `tl.dot_scaled` with proper e8m0 weight preprocessing**: Both GEMMs are latency-bound at ~35% compute. If latency hiding can be improved, FP8 native MMA (2x throughput of BF16) becomes worth the e8m0 scale format rework.

3. **CUTLASS/CUDA kernel**: Bypass Triton for full control over MMA instruction selection, register allocation, TMA, and warp scheduling. Removes all Triton-imposed constraints.
