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

## What Has NOT Been Tried (Next Steps)

The single most promising remaining path is a **TMA + warp specialization rewrite**, modelled on Triton's own reference MoE kernel (`_p_matmul.py`). Key elements:

1. **Persistent kernel with `tl.range(warp_specialize=True, flatten=True)`**: Eliminates the M-loop, absorbs all tiles into a flat persistent work-stealing loop. The compiler auto-partitions TMA loads (producer warps) from MMA compute (consumer warps).

2. **TMA tensor descriptors**: Host-side `TensorDescriptor` for weights (dense 3D `[1, BLOCK_K, BLOCK_N]`), activation gather via `desc.gather(indices, k_offset)` using 2D `[1, BLOCK_K]` descriptors, output scatter via `desc.scatter(out, indices, n_offset)`.

3. **TMA gather fuses the dispatch step**: Instead of a separate dispatch_scatter kernel writing sorted token IDs to global memory, TMA gather loads tokens directly from their original positions using the index array — eliminating one full memory pass.

4. **Warp specialization requires TMA**: Confirmed experimentally — `warp_specialize=True` without TMA caused a 3.4x regression because the compiler can't partition manual `tl.load` calls into clean producer warps.

5. **Revisit `tl.dot_scaled` with proper e8m0 weight preprocessing**: Once TMA is in place and warp spec hides load latency, the MMA units become the true bottleneck. At that point, FP8 native MMA (2x throughput of BF16) becomes worth the e8m0 scale format rework. Scales need to be stored as `torch.uint8` with e8m0 encoding baked in at weight-load time, not computed on the fly.

6. **CUTLASS/CUDA kernel**: Bypass Triton entirely for more control over MMA instruction selection, register allocation, and TMA. Highest effort but removes all Triton-imposed constraints.
