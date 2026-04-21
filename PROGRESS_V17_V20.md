# v17–v20 Progress & Plan

Target: push the contest MoE kernel as close as possible to 80–90% of B200 raw FP8 roofline on all 19 workloads, using insights from DeepGEMM MegaMoE, FlashInfer CuTeDSL MoE, SGLang NVFP4 MoE, and CUTLASS SM100 — without copying any of them directly (contest compliance).

## Hardware constraint recap

- `tcgen05.mma.kind.block_scale` on SM100 accepts only `UE8M0` (vec=32) for the FP8/MXF8 path, or `UE4M3` for NVFP4 (FP4 operands only).

```216:233:/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/cutlass/include/cute/arch/mma_sm100_desc.hpp
enum class ScaleFormat : uint8_t {
  UE4M3 = 0,
  UE8M0 = 1,
};

template <class T>
CUTE_HOST_DEVICE constexpr ScaleFormat to_ScaleFormat() {
  if constexpr (is_same_v<T, float_ue4m3_t>) { return ScaleFormat::UE4M3;  } else
  if constexpr (is_same_v<T, float_ue8m0_t>) { return ScaleFormat::UE8M0;  } else
  { static_assert(sizeof(T) == 0, "Unknown type for ScaleFormat"); }
}
```

- Contest data is FP8 payload + FP32 per-128-block scales, which forces us onto CUTLASS `KernelPtrArrayTmaWarpSpecializedBlockwise{1,2}SmSm100` (software-scaled path) whose mainloop ceiling is strictly below hardware block-scale MMA.
- Current large-T perf (submission-v16, T=14107): ~35x speedup, ~52% SM utilization on GEMMs. That is close to the real ceiling for this mainloop.

## Ceilings

- On the FP32-blockwise mainloop: **pipeline-fusion** (eliminate HBM round-trips, merge kernels) can plausibly push us to ~55–70% of raw FP8 roofline.
- To reach **≥80% of raw FP8 roofline** at large T: must unlock `tcgen05.mma.kind.block_scale`. The only contest-compliant way is a **tile-local MXF8 transcode** of `(fp8 payload, fp32 scale) -> (fp8 payload', UE8M0 scale)` done on-chip in a dedicated transform warp before MMA. Numerically unproven.

## Two-branch plan

### Branch A — Pipeline fusion on the current FP32-scale mainloop (mandatory, safe)

All contest-specific; not a copy of any external kernel.

- **v17 — routing-weight fold into A-scale for GEMM2**
  - Already integrated: `swiglu_fp8_requant_weighted` + `reduce_scatter_unweighted_prebucketed`.
  - Eliminates the per-element `w * v` multiply in reduce_scatter.
  - Expected: ~5–15 μs at T=14107.
  - Pattern: FlashInfer finalize-fusion’s `router_scale * acc_scaled`.

- **v18 — paired GEMM1 epilogue SwiGLU**
  - Pre-pack `gemm1_weights` as `[E, H, 2, K]` interleaved gate/up pairs.
  - Custom EVT: each tile (tile_N=128 = 64 `(up, gate)` pairs) computes `silu(gate)*up*alpha`, CTA-local amax, writes bf16 [M, H] plus per-tile amax to `[M, H/128]`.
  - Tiny second pass converts bf16 -> fp8 using per-row amax.
  - Eliminates bf16 [M, 2H] round-trip; halves intermediate footprint.
  - Expected: ~20–40 μs at T=14107.
  - Pattern: FlashInfer `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion.py` + DeepGEMM MegaMoE paired `SM100_TMEM_LOAD_16dp256b1x`.

- **v19 — GEMM2 finalize/scatter EVT epilogue**
  - Custom CUTLASS EVT: `tcgen05.ld -> alpha -> router_scale[m] -> bf16 cast -> vectorized atomicAdd bf16x8 into out[sorted_tids[m], n]`.
  - Removes `gemm2_out` HBM write, `reduce_scatter` read, and one kernel launch.
  - Expected: ~50–100 μs at T=14107.
  - Pattern: FlashInfer `blockscaled_contiguous_grouped_gemm_finalize_fusion.py` + TRT-LLM `sm90_visitor_scatter.hpp`.

- **v20 — persistent wavefront scheduler (GEMM1 + SwiGLU pool + GEMM2)**
  - Expert-wave scheduler: `BlockPhase::{Linear1, Linear2}`, wave-sized FP8 pool, single kernel.
  - Omits NVLink/sym-mem/dispatch-fusion layers of DeepGEMM MegaMoE (single-GPU).
  - Expected: another ~50–100 μs by eliminating a kernel boundary and shrinking working set.
  - Pattern: DeepGEMM MegaMoE scheduler + FlashInfer warp-specialization structure.

### Branch B — Ceiling breaker: tile-local MXF8 transcode (CLOSED – infeasible)

- Phase 1 probe implemented in `experiments/custom_mma/mxf8_transcode_probe.py`.
- Test: transcode all three `(fp8, fp32)` inputs to `(fp8', pow2 scale)` with residual absorbed into payload, keep the FP32-scale mainloop for apples-to-apples accuracy check.
- Result: match ratio 3–73% on 7 sampled workloads (both round and ceil variants). Contest requires ≥90%. Large_T workloads 5e8dc11c/58a34f27 both at ~45% / ~58%.
- Root cause: re-rounding `payload_fp8 * residual` back to FP8 E4M3 (3-bit mantissa) injects per-element quantization error ~2× on top of contest FP8 quantization. Over K=7168 accumulation this blows past `atol=1, rtol=0.3`.
- **Decision**: Branch B dropped. All remaining effort on Branch A.

Attempted but also dead:
- Round-to-pow2 UE8M0 scale (earlier `ue8m0_careful.py` experiment).
- Ceil-to-pow2 UE8M0 scale.
- Using `UE4M3` scale format on FP8/MXF8 path (CUTLASS only allows UE4M3 with FP4 operands via NVF4).

## Targets

| Milestone | Projected large-T (T=14107) latency | % of raw FP8 roofline |
|---|---|---|
| v16 (current) | ~1.33 ms | ~40% |
| + v17 | ~1.31 ms | ~41% |
| + v19 | ~1.21 ms | ~44% |
| + v18 | ~1.16 ms | ~46% |
| + v20 | ~1.07 ms | ~50% |
| + MXF8 transcode (FAILED tolerance) | n/a | n/a |

Branch B ruled out by numerical probe. Branch A alone reaches ~80–90% of the *contest-achievable* FP32-blockwise ceiling, which corresponds to ~50–55% of raw FP8 roofline on large-T workloads.

## Non-goals

- Do not copy DeepGEMM MegaMoE (multi-GPU, FP4, UE8M0).
- Do not copy FlashInfer CuTeDSL MoE (MXF8/NVF4 only).
- Do not copy SGLang NVFP4 grouped GEMM (FP4 weights).
- Do adopt their scheduler, warp specialization, paired SwiGLU epilogue, finalize/scatter epilogue, and persistent wavefront structure.

## Execution order

1. ✅ Validate v17 on all 19 workloads (Modal). All pass; absolute latencies stable vs v16.
2. ✅ Run MXF8 transcode Phase-1 probe. Failed tolerance → Branch B dropped.
3. **Next**: implement v19 finalize/scatter kernel (custom post-GEMM2 that keeps gemm2_out but fuses per-row weight + scatter more aggressively, OR replaces GEMM2 with custom-epilogue kernel).
4. Then: v18 paired SwiGLU epilogue.
5. Then: v20 persistent wavefront scheduler.
6. **Final**: submit best combined variant (tags `submission-v17` onward).

## Submission tagging rules

- No tag until the winning variant passes all 19 workloads on Modal.
- Tag monotonically (`submission-v17`, `submission-v18`, …); only the latest tag is evaluated by contest.

## Open questions

1. Does GEMM2-only MXF8 transcode pass contest tolerance on all 19 workloads?
2. Is FlashInfer-style weight pre-packing for v18 worth it vs. a runtime shuffle kernel?
3. Can v20’s single-persistent-kernel structure be built on top of CUTLASS directly, or do we need raw CuTe for the wavefront scheduler?
