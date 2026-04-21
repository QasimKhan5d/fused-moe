"""
Fused MoE kernel — CUTLASS SM100 FP8 blockwise-scaled grouped GEMMs.

Architecture (based on SGLang/DeepGEMM reference):
  - K-major SFA/SFB layout → no per-expert scale packing needed
  - Device-side `get_group_gemm_starts` kernel builds per-expert ptr/layout arrays
  - AB-swap path for small-M (MmaConfig1: <256,32,128> cluster<2,1,1>)
  - Standard path for large-M (MmaConfig2: <128,128,128> cluster<1,1,1>)
  - Pre-allocated workspace cached per (T, device) key

Pipeline:
  1. Route (PyTorch)
  2. Dispatch (argsort + bincount)
  3. Gather hidden+scales by sorted token order (no padding)
  4. GEMM1 via CUTLASS blockwise grouped
  5. SwiGLU + FP8 per-token requant
  6. GEMM2 via CUTLASS blockwise grouped
  7. Weighted scatter back to [T, N2]

FlashInfer entry: kernel.py::kernel
"""
import glob
import os
import sys
import tempfile
import torch
import os as _os_cfg_X
import os as _os_cfg_J
import os as _os_cfg_V
import os as _os_cfg_K

E_GLOBAL = 256
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = 32
TOP_K = 8

_ext = None

# Embedded C++ kernel source
_MOE_GEMM_CU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;

// ============================================================================
// Config 1: Small M (AB-swap, <256,32,128> cluster<2,1,1>, 2SmSm100)
//   Used when total_tokens <= 2048 (small-T).
//   Computes D.T = B @ A.T where A=weights (big M), B=activations (small N).
// ============================================================================
struct CfgSmallM {
  using MmaTileShape = Shape<_256, _32, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

// ============================================================================
// Config 2: Large M — 1SM, tile K=256 for fewer mainloop iterations per tile.
// With K=7168, K-iters drop from 56 to 28.
// ============================================================================
struct CfgLargeM {
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

// ============================================================================
// Config 2b: Mid M — 1SM with smaller M tile for experts whose local token
// counts are too small to fully utilize 128x128 tiles, but too large for the
// AB-swap small-M path.
// ============================================================================
struct CfgMidM {
  using MmaTileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

// ============================================================================
// Config 3: Very Large M — 2SM cooperative MMA for workloads where per-expert
// M >= 256 (total_tokens > ~8192). 2 CTAs cooperate on a 256-M tile; their
// 2 TMA engines pull weight data in parallel, improving HBM BW utilization
// when compute-bound. Cluster<2,1,1> multicasts B across the 2 CTAs.
// ============================================================================
struct CfgVeryLargeM {
  using MmaTileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
};

// ============================================================================
// v18 (ceiling-breaker): MxF8F6F4 hardware block-scale MMA config.
// Uses tcgen05.mma.kind.mxf8f6f4.block_scale with UE8M0 scales at sf_vec_size=32.
// Requires inputs to be transcoded: sign-flip + residual absorbed into payload
// (done upstream in a dedicated transcode kernel).
// ============================================================================
struct CfgMxF8Large {
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

struct CfgMxF8VeryLarge {
  using MmaTileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// ============================================================================
// Build the CUTLASS Gemm type from a config
// ============================================================================
using ElementAB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 16;  // 128bits / 8 = 16 fp8 elements
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 8;   // 128bits / 16 = 8 bf16 elements

// MxF8 operand types: wrap FP8 E4M3 with MxF8 tag so CUTLASS selects the
// hardware block-scale MMA path.
using ElementSF_mx = cutlass::float_ue8m0_t;  // 8-bit exponent-only scale type
using MmaTypePairA = decltype(cute::make_tuple(ElementAB{}, ElementSF_mx{}));
using MmaTypePairB = decltype(cute::make_tuple(ElementAB{}, ElementSF_mx{}));

// ============================================================================
// MxF8 GemmBuilder: uses hardware block-scale MMA schedule with UE8M0 scales
// at SFVecSize=32. Requires transcoded inputs (see mxf8_transcode_*).
// ============================================================================
template <typename Cfg, typename LayoutOut>
struct MxF8GemmBuilder {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      typename Cfg::MmaTileShape, typename Cfg::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void, LayoutOut*, AlignmentC,
      ElementC, LayoutOut*, AlignmentC,
      typename Cfg::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      MmaTypePairA, LayoutA*, AlignmentA,
      MmaTypePairB, LayoutB*, AlignmentB,
      ElementAccumulator,
      typename Cfg::MmaTileShape, typename Cfg::ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Cfg::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using InternalLayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using InternalLayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  static constexpr int SFVecSize = Gemm::GemmKernel::CollectiveMainloop::SFVecSize;
};

// MxF8 per-expert starts kernel: fills ptr arrays, strides, and SFA/SFB
// layouts for ptr-array MxF8 grouped GEMM.
template <typename Cfg, typename LayoutOut>
__global__ void get_group_gemm_starts_kernel_mxf8(
    int32_t const* __restrict__ expert_offsets,  // [E]
    int32_t const* __restrict__ sfa_offsets,     // [E+1] cumulative SFA byte offsets
    int32_t const* __restrict__ sfb_offsets,     // [E+1] cumulative SFB byte offsets
    ElementAB** a_ptrs,
    ElementAB** b_ptrs,
    ElementC** out_ptrs,
    ElementSF_mx** sfa_ptrs,
    ElementSF_mx** sfb_ptrs,
    ElementAB* a_base,
    ElementAB* b_base,
    ElementC* out_base,
    ElementSF_mx* sfa_base,
    ElementSF_mx* sfb_base,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* layout_sfa_base,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFB* layout_sfb_base,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::StrideA* stride_a_base,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::StrideB* stride_b_base,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::StrideC* stride_c_base,
    int32_t const* problem_sizes)                // [E, 3]
{
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;

  int eid = threadIdx.x;
  int m_e = problem_sizes[eid * 3];
  int n_e = problem_sizes[eid * 3 + 1];
  int k_e = problem_sizes[eid * 3 + 2];

  int64_t expert_offset = static_cast<int64_t>(expert_offsets[eid]);
  int64_t a_stride       = expert_offset * k_e;
  int64_t b_stride       = int64_t(eid) * int64_t(k_e) * int64_t(n_e);

  a_ptrs[eid]         = a_base + a_stride;
  b_ptrs[eid]         = b_base + b_stride;
  out_ptrs[eid]       = out_base + expert_offset * n_e;

  sfa_ptrs[eid] = sfa_base + sfa_offsets[eid];
  sfb_ptrs[eid] = sfb_base + sfb_offsets[eid];

  layout_sfa_base[eid] = SfConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_e, n_e, k_e, 1));
  layout_sfb_base[eid] = SfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m_e, n_e, k_e, 1));

  stride_a_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideA{}, cute::make_shape(m_e, k_e, 1));
  stride_b_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideB{}, cute::make_shape(n_e, k_e, 1));
  stride_c_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideC{}, cute::make_shape(m_e, n_e, 1));
}

template <typename Cfg, typename LayoutOut>
void launch_mxf8_group_gemm(
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& workspace,
    int num_experts,
    cudaStream_t stream)
{
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using Gemm = typename MxFB::Gemm;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementAB**>(a_ptrs.data_ptr()),
      static_cast<typename MxFB::StrideA*>(const_cast<void*>(stride_a.data_ptr())),
      static_cast<const ElementAB**>(b_ptrs.data_ptr()),
      static_cast<typename MxFB::StrideB*>(const_cast<void*>(stride_b.data_ptr())),
      static_cast<const ElementSF_mx**>(sfa_ptrs.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(const_cast<void*>(layout_sfa.data_ptr())),
      static_cast<const ElementSF_mx**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(const_cast<void*>(layout_sfb.data_ptr()))
  };

  typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<typename MxFB::StrideC*>(const_cast<void*>(stride_c.data_ptr())),
      static_cast<ElementC**>(out_ptrs.data_ptr()),
      static_cast<typename MxFB::StrideC*>(const_cast<void*>(stride_c.data_ptr()))
  };

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  scheduler.max_swizzle_size = 1;

  auto* ps = static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.data_ptr()));
  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, ps, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler
  };
  auto& fusion_args = args.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "MxF8 can_implement failed: ", int(status));
  status = gemm_op.initialize(args, const_cast<void*>(workspace.data_ptr()), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "MxF8 initialize failed: ", int(status));
  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "MxF8 run failed: ", int(status));
}

template <typename Cfg, typename LayoutOut>
struct GemmBuilder {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      typename Cfg::MmaTileShape, typename Cfg::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void, LayoutOut*, AlignmentC,
      ElementC, LayoutOut*, AlignmentC,
      typename Cfg::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementAB, cute::tuple<LayoutA*, typename Cfg::LayoutSFA*>, AlignmentA,
      ElementAB, cute::tuple<LayoutB*, typename Cfg::LayoutSFB*>, AlignmentB,
      ElementAccumulator,
      typename Cfg::MmaTileShape, typename Cfg::ClusterShape,
      // AutoCarveout picks ~3-5 stages based on SMEM. Leave default (our
      // experiments with explicit StageCount<6> and <8> showed no difference).
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename Cfg::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
};

// ============================================================================
// Device kernel: per-expert pointer + layout + stride array builder.
// Runs 1 block, E threads. Fills all arg arrays in one launch.
// ============================================================================
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig,
          typename StrideA, typename StrideB, typename StrideC,
          typename OutT>
__global__ void get_group_gemm_starts_kernel(
    int32_t const* __restrict__ expert_offsets,  // [E]
    ElementAB**   a_ptrs,
    ElementAB**   b_ptrs,
    OutT**        out_ptrs,
    float**       a_scales_ptrs,
    float**       b_scales_ptrs,
    ElementAB*    a_base,
    ElementAB*    b_base,
    OutT*         out_base,
    float*        a_scales_base,
    float*        b_scales_base,
    LayoutSFA*    layout_sfa_base,
    LayoutSFB*    layout_sfb_base,
    StrideA*      stride_a_base,
    StrideB*      stride_b_base,
    StrideC*      stride_c_base,
    int32_t const* problem_sizes,            // [E, 3]
    int32_t*       problem_sizes_transpose,  // [E, 3] output
    bool transpose)
{
  int eid = threadIdx.x;
  int m = problem_sizes[eid * 3];
  int n = problem_sizes[eid * 3 + 1];
  int k = problem_sizes[eid * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[eid * 3]     = n;
    problem_sizes_transpose[eid * 3 + 1] = m;
    problem_sizes_transpose[eid * 3 + 2] = k;
  }
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[eid]);
  int64_t a_stride, b_stride, a_scale_stride, b_scale_stride;
  if (!transpose) {
    a_stride       = expert_offset * k;
    b_stride       = int64_t(eid) * int64_t(k) * int64_t(n);
    a_scale_stride = expert_offset * k / 128;
    b_scale_stride = int64_t(eid) * int64_t(k) * int64_t(n) / 128 / 128;
  } else {
    a_stride       = int64_t(eid) * int64_t(k) * int64_t(n);
    b_stride       = expert_offset * k;
    a_scale_stride = int64_t(eid) * int64_t(k) * int64_t(n) / 128 / 128;
    b_scale_stride = expert_offset * k / 128;
  }
  a_ptrs[eid]         = a_base + a_stride;
  b_ptrs[eid]         = b_base + b_stride;
  out_ptrs[eid]       = out_base + expert_offset * n;
  a_scales_ptrs[eid]  = a_scales_base + a_scale_stride;
  b_scales_ptrs[eid]  = b_scales_base + b_scale_stride;

  int M_e = transpose ? n : m;
  int N_e = transpose ? m : n;

  if (!transpose) {
    layout_sfa_base[eid] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_sfb_base[eid] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    layout_sfa_base[eid] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    layout_sfb_base[eid] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }

  // Strides for per-expert (M_e, N_e, K) problem (L=1).
  stride_a_base[eid] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_e, k, 1));
  stride_b_base[eid] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_e, k, 1));
  stride_c_base[eid] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_e, N_e, 1));
}

// ============================================================================
// Launch CUTLASS grouped GEMM with pre-filled ptr/layout arrays
// ============================================================================
template <typename Cfg, typename LayoutOut>
void launch_group_gemm(
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& workspace,
    int num_experts,
    cudaStream_t stream)
{
  using GB = GemmBuilder<Cfg, LayoutOut>;
  using Gemm = typename GB::Gemm;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  typename Gemm::GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementAB**>(a_ptrs.data_ptr()),
      static_cast<typename GB::StrideA*>(const_cast<void*>(stride_a.data_ptr())),
      static_cast<const ElementAB**>(b_ptrs.data_ptr()),
      static_cast<typename GB::StrideB*>(const_cast<void*>(stride_b.data_ptr())),
      static_cast<const float**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename Cfg::LayoutSFA*>(const_cast<void*>(layout_sfa.data_ptr())),
      static_cast<const float**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename Cfg::LayoutSFB*>(const_cast<void*>(layout_sfb.data_ptr()))
  };

  typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<typename GB::StrideC*>(const_cast<void*>(stride_c.data_ptr())),
      static_cast<ElementC**>(out_ptrs.data_ptr()),
      static_cast<typename GB::StrideC*>(const_cast<void*>(stride_c.data_ptr()))
  };

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  // AlongM raster (empirically ~3-5% faster than AlongN on T>=11948 where
  // per-expert M dominates; flat on small/mid T). max_swizzle_size=1 stays
  // (2/4 regressed by 5-17% on large T).
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  scheduler.max_swizzle_size = 1;
  if (const char* env = std::getenv("CUTLASS_RASTER_ORDER")) {
    if (env[0] == 'M' || env[0] == 'm') {
      scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
    } else if (env[0] == 'H' || env[0] == 'h') {
      scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;
    }
  }
  if (const char* env = std::getenv("CUTLASS_MAX_SWIZZLE")) {
    int swizzle = std::atoi(env);
    if (swizzle > 1) {
      scheduler.max_swizzle_size = swizzle;
    }
  }

  auto* ps = static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.data_ptr()));
  typename Gemm::GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, ps, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler
  };
  auto& fusion_args = args.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed: ", int(status));
  status = gemm_op.initialize(args, const_cast<void*>(workspace.data_ptr()), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed: ", int(status));
  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed: ", int(status));
}

// ============================================================================
// Main MoE blockwise grouped GEMM entry point.
//
// a           : [total_tokens, K] fp8_e4m3 (dispatched activations)
// b           : [E, N, K] fp8_e4m3 (weights)
// scales_a    : [total_tokens, K/128] fp32 (kb-fast per token)
// scales_b    : [E, N/128, K/128] fp32 (kb-fast per (N/128)-block)
// expert_offsets: [E] int32 (cumsum of tokens-per-expert EXCLUSIVE — first element is 0)
//   ... wait, SGLang uses expert_offset[e] = cumulative prev tokens.
// problem_sizes: [E, 3] int32 (M=tokens_e, N, K per expert)
// Workspace tensors (all device, pre-allocated):
//   a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs : [E] int64 (pointer arrays)
//   stride_a, stride_b, stride_c : [E * 3] int64 (per-expert strides; same for all experts
//       since problem shape is variable-M, K and N are fixed → strides constant, precompute once)
//   layout_sfa, layout_sfb : [E, sizeof(Layout) / 4] int32
//   problem_sizes_transpose : [E, 3] int32 (scratch for AB-swap path)
//   workspace : [workspace_size] uint8 (CUTLASS GEMM workspace)
// ============================================================================
void moe_blockwise_grouped_mm_v2(
    torch::Tensor& output,           // [total_tokens, N] bf16
    torch::Tensor const& a,          // [total_tokens, K] fp8
    torch::Tensor const& b,          // [E, N, K] fp8
    torch::Tensor const& scales_a,   // [total_tokens, K/128] fp32 (kb-fast row-major)
    torch::Tensor const& scales_b,   // [E, N/128, K/128] fp32 (kb-fast row-major per expert)
    torch::Tensor const& expert_offsets,  // [E] int32 (cumulative; expert_offsets[e] = prev tokens)
    torch::Tensor const& problem_sizes,   // [E, 3] int32
    torch::Tensor& problem_sizes_transpose,  // [E, 3] int32 (scratch)
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor const& workspace)
{
  int total_tokens = a.size(0);
  int K = a.size(1);
  int E = b.size(0);
  int N = b.size(1);
  TORCH_CHECK(b.size(2) == K, "b K mismatch");
  TORCH_CHECK(output.size(0) == total_tokens && output.size(1) == N, "output shape mismatch");

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  char schedule_mode = 'A';
  if (const char* env = std::getenv("CUTLASS_CFG_MODE")) {
    schedule_mode = static_cast<char>(std::toupper(env[0]));
  }

  // Three tile configs based on total-M (dispatched tokens):
  //   <=2048         : AB-swap + CfgSmallM (tile<256,32,128>)
  //   2048 < M <=8192: standard CfgLargeM (tile<128,128,128>, 1SM)
  //   >8192          : 2SM cooperative CfgVeryLargeM (tile<256,128,128>)
  //
  // The 8192 threshold corresponds to per-expert M ≈ 256 (E_LOCAL=32), which
  // is the minimum M for tile_M=256 to be well-utilized.
  auto launch_large_m = [&](auto cfg_tag) {
    using Cfg = decltype(cfg_tag);
    using LayoutOut = cutlass::layout::RowMajor;
    using GB = GemmBuilder<Cfg, LayoutOut>;

    get_group_gemm_starts_kernel<
        typename Cfg::LayoutSFA, typename Cfg::LayoutSFB, typename Cfg::ScaleConfig,
        typename GB::StrideA, typename GB::StrideB, typename GB::StrideC, ElementC>
        <<<1, E, 0, stream>>>(
        static_cast<int32_t const*>(expert_offsets.data_ptr()),
        static_cast<ElementAB**>(a_ptrs.data_ptr()),
        static_cast<ElementAB**>(b_ptrs.data_ptr()),
        static_cast<ElementC**>(out_ptrs.data_ptr()),
        static_cast<float**>(a_scales_ptrs.data_ptr()),
        static_cast<float**>(b_scales_ptrs.data_ptr()),
        static_cast<ElementAB*>(const_cast<void*>(a.data_ptr())),
        static_cast<ElementAB*>(const_cast<void*>(b.data_ptr())),
        static_cast<ElementC*>(output.data_ptr()),
        static_cast<float*>(const_cast<void*>(scales_a.data_ptr())),
        static_cast<float*>(const_cast<void*>(scales_b.data_ptr())),
        reinterpret_cast<typename Cfg::LayoutSFA*>(layout_sfa.data_ptr()),
        reinterpret_cast<typename Cfg::LayoutSFB*>(layout_sfb.data_ptr()),
        reinterpret_cast<typename GB::StrideA*>(stride_a.data_ptr()),
        reinterpret_cast<typename GB::StrideB*>(stride_b.data_ptr()),
        reinterpret_cast<typename GB::StrideC*>(stride_c.data_ptr()),
        static_cast<int32_t const*>(problem_sizes.data_ptr()),
        static_cast<int32_t*>(problem_sizes_transpose.data_ptr()),
        /*transpose=*/false);

    launch_group_gemm<Cfg, LayoutOut>(
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        problem_sizes, workspace, E, stream);
  };

  // Note: CfgVeryLargeM (2SM) was evaluated but regressed by 1-5% on all
  // workloads — per-expert M (~400) is too small to benefit from tile_M=256.
  // GEMM isn't the bottleneck anyway; remaining wins are in non-GEMM kernels.
  if (schedule_mode == 'S' || (schedule_mode == 'A' && total_tokens <= 2048)) {
    using Cfg = CfgSmallM;
    using LayoutOut = cutlass::layout::ColumnMajor;
    using GB = GemmBuilder<Cfg, LayoutOut>;

    // AB-swap: pass weights where a_base expected, activations where b_base expected.
    get_group_gemm_starts_kernel<
        typename Cfg::LayoutSFA, typename Cfg::LayoutSFB, typename Cfg::ScaleConfig,
        typename GB::StrideA, typename GB::StrideB, typename GB::StrideC, ElementC>
        <<<1, E, 0, stream>>>(
        static_cast<int32_t const*>(expert_offsets.data_ptr()),
        static_cast<ElementAB**>(a_ptrs.data_ptr()),
        static_cast<ElementAB**>(b_ptrs.data_ptr()),
        static_cast<ElementC**>(out_ptrs.data_ptr()),
        static_cast<float**>(a_scales_ptrs.data_ptr()),
        static_cast<float**>(b_scales_ptrs.data_ptr()),
        static_cast<ElementAB*>(const_cast<void*>(b.data_ptr())),       // a-role = weights (swap)
        static_cast<ElementAB*>(const_cast<void*>(a.data_ptr())),       // b-role = activations
        static_cast<ElementC*>(output.data_ptr()),
        static_cast<float*>(const_cast<void*>(scales_b.data_ptr())),
        static_cast<float*>(const_cast<void*>(scales_a.data_ptr())),
        reinterpret_cast<typename Cfg::LayoutSFA*>(layout_sfa.data_ptr()),
        reinterpret_cast<typename Cfg::LayoutSFB*>(layout_sfb.data_ptr()),
        reinterpret_cast<typename GB::StrideA*>(stride_a.data_ptr()),
        reinterpret_cast<typename GB::StrideB*>(stride_b.data_ptr()),
        reinterpret_cast<typename GB::StrideC*>(stride_c.data_ptr()),
        static_cast<int32_t const*>(problem_sizes.data_ptr()),
        static_cast<int32_t*>(problem_sizes_transpose.data_ptr()),
        /*transpose=*/true);

    launch_group_gemm<Cfg, LayoutOut>(
        a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        problem_sizes_transpose, workspace, E, stream);
  } else if (schedule_mode == 'M') {
    launch_large_m(CfgMidM{});
  } else {
    launch_large_m(CfgLargeM{});
  }
}

// ============================================================================
// v18: MxF8 path — transcode kernels.
//
// Transcode: (payload_fp8, signed_fp32_scale) per K-128-block -> (payload'_fp8, |scale|_fp32).
//   payload'[i] = sign(scale[block(i)]) * round_fp8(payload[i] * r[block(i)])
//   where r = |scale| / ceil_pow2(|scale|), in (0.5, 1.0].
// |scale| is then passed to CUTLASS MxF8 which internally converts to UE8M0
// via `float_ue8m0_t(float)` (ceil-to-pow2).
// ============================================================================
__device__ __forceinline__ uint8_t ue8m0_ceil_from_abs_fp32(float x) {
  // Ceil to pow-of-2; returns the UE8M0 exponent byte (8-bit bias 0; value 2^(byte-127)).
  // Actually UE8M0 encodes exponent in [-127, 127] directly via the 8-bit field
  // where `byte` = ieee32_exponent + rounding_increment.
  // For ceil: if mantissa > 0 and not max exponent, increment exponent.
  if (!isfinite(x) || x <= 0.0f) return 0;
  uint32_t bits = __float_as_uint(x);
  uint8_t exp = (bits >> 23) & 0xff;  // IEEE-32 exponent (biased)
  uint32_t mant = bits & 0x7fffff;
  if (mant > 0 && exp != 0xFE) exp++;
  return exp;
}

__device__ __forceinline__ float ue8m0_byte_to_fp32(uint8_t b) {
  uint32_t f = (uint32_t)(b) << 23;
  return __uint_as_float(f);
}

// Transcode kernel, GENERIC layout for A-side (activations) or B-side (weights).
// Input layout: payload is [..., K] fp8 E4M3; scale is per-128 K-block signed fp32.
// - For activations: payload [M, K] row-major, scale [M, K/128] row-major (kb-fast).
// - For weights:      payload [E, N, K] row-major, scale [E, N/128, K/128] row-major.
// Output:
// - payload': overwrite payload in place (sign-flip + residual absorption).
// - scale_abs: fp32 buffer with |scale| values (same shape as input scale). CUTLASS
//              converts these to UE8M0 internally via `ElementSF(float)` which ceils.
//   Output size same as input scale.
__global__ void mxf8_transcode_kernel(
    __nv_fp8_e4m3*       __restrict__ payload,  // in/out, size N_pay = M * K
    const float*         __restrict__ scale_signed,  // in, size N_sc = M * K/128 (contiguous)
    float*               __restrict__ scale_ue8m0,  // out: ue8m0 as fp32 (pow-of-2)
    int                   M,                        // leading dim (rows)
    int                   K,                        // K
    int                   K_blocks,                 // = K/128
    int                   payload_row_stride,       // = K for contiguous row-major
    int                   scale_row_stride          // = K_blocks for contiguous
) {
  // Grid: (M * K_blocks) blocks; each block handles one 128-element sub-row.
  int m = blockIdx.y;
  int kb = blockIdx.x;
  if (m >= M || kb >= K_blocks) return;

  int tid = threadIdx.x;
  int scale_idx = m * scale_row_stride + kb;
  float s_signed = scale_signed[scale_idx];
  float s_abs = fabsf(s_signed);
  float sign = (s_signed < 0.0f) ? -1.0f : 1.0f;

  // Ceil to pow-of-2.
  uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(s_abs);
  float ue8m0_val = ue8m0_byte_to_fp32(ue8m0_byte);

  // Write ue8m0 (fp32 power-of-2 value) as the scale. This is the value that
  // MxF8 hardware uses and also the value that our FP32-blockwise kernel
  // uses as stand-in. `r = |scale|/ue8m0` is absorbed into the payload below.
  if (tid == 0) {
    scale_ue8m0[scale_idx] = ue8m0_val;
  }

  float r = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
  float sr = sign * r;

  // Transcode payload: 128 FP8 elements per warp block.
  __nv_fp8_e4m3* row = payload + m * payload_row_stride + kb * 128;
  // 128 threads each handle 1 element (if blockDim.x==128). If smaller, stride.
  int TB = blockDim.x;
  for (int j = tid; j < 128; j += TB) {
    float v = (float)row[j];
    float vn = v * sr;
    // Clamp to FP8 E4M3 range to avoid NaN on overflow.
    if (vn > 448.0f) vn = 448.0f;
    if (vn < -448.0f) vn = -448.0f;
    row[j] = __nv_fp8_e4m3(vn);
  }
}

// Host wrappers for transcoding activations and weights.
void mxf8_transcode_activations(
    torch::Tensor& payload,         // [M, K] fp8 (in/out)
    torch::Tensor const& scale,     // [M, K/128] fp32 signed (in)
    torch::Tensor& scale_ue8m0      // [M, K/128] fp32 (out), holds pow-of-2 values
) {
  TORCH_CHECK(payload.is_cuda() && payload.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32);
  int M = payload.size(0);
  int K = payload.size(1);
  int K_blocks = scale.size(1);
  TORCH_CHECK(K_blocks == K / 128, "scale K/128 mismatch");
  TORCH_CHECK(scale.size(0) == M, "scale M mismatch");

  auto stream = at::cuda::getCurrentCUDAStream(payload.get_device()).stream();
  dim3 grid(K_blocks, M);
  mxf8_transcode_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<__nv_fp8_e4m3*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      scale_ue8m0.data_ptr<float>(),
      M, K, K_blocks,
      payload.stride(0),   // payload row stride
      scale.stride(0));    // scale row stride
}

// Dedicated weight-transcode kernel: per-expert, per-output-row, per-k-block.
__global__ void mxf8_transcode_weight_kernel(
    __nv_fp8_e4m3*       __restrict__ payload,  // [E, N, K] fp8 (in/out)
    const float*         __restrict__ scale_signed,  // [E, N/128, K/128] fp32
    float*               __restrict__ scale_ue8m0,   // [E, N/128, K/128] fp32 (out, pow-of-2)
    int E, int N, int K, int N_blocks, int K_blocks
) {
  int kb = blockIdx.x;
  int n = blockIdx.y;
  int e = blockIdx.z;
  if (kb >= K_blocks || n >= N || e >= E) return;
  int nb = n / 128;
  int tid = threadIdx.x;

  int scale_idx = (e * N_blocks + nb) * K_blocks + kb;
  float s_signed = scale_signed[scale_idx];
  float s_abs = fabsf(s_signed);
  float sign = (s_signed < 0.0f) ? -1.0f : 1.0f;

  uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(s_abs);
  float ue8m0_val = ue8m0_byte_to_fp32(ue8m0_byte);

  // Only ONE thread in the FIRST n of each nb-row writes scale_ue8m0 (dedup).
  if (tid == 0 && (n % 128) == 0) {
    scale_ue8m0[scale_idx] = ue8m0_val;
  }

  float r = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
  float sr = sign * r;

  __nv_fp8_e4m3* row = payload + ((e * (int64_t)N + n) * (int64_t)K) + kb * 128;
  int TB = blockDim.x;
  for (int j = tid; j < 128; j += TB) {
    float v = (float)row[j];
    float vn = v * sr;
    if (vn > 448.0f) vn = 448.0f;
    if (vn < -448.0f) vn = -448.0f;
    row[j] = __nv_fp8_e4m3(vn);
  }
}

void mxf8_transcode_weights_impl(
    torch::Tensor& payload,         // [E, N, K] fp8 (in/out)
    torch::Tensor const& scale,     // [E, N/128, K/128] fp32 signed (in)
    torch::Tensor& scale_ue8m0      // [E, N/128, K/128] fp32 (out, pow-of-2)
) {
  TORCH_CHECK(payload.is_cuda() && payload.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32);
  int E = payload.size(0);
  int N = payload.size(1);
  int K = payload.size(2);
  int N_blocks = scale.size(1);
  int K_blocks = scale.size(2);

  auto stream = at::cuda::getCurrentCUDAStream(payload.get_device()).stream();
  dim3 grid(K_blocks, N, E);
  mxf8_transcode_weight_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<__nv_fp8_e4m3*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      scale_ue8m0.data_ptr<float>(),
      E, N, K, N_blocks, K_blocks);
}

// ============================================================================
// Helper: pack per-128-K-block fp32 pow-of-2 scales into the CUTLASS-expected
// MxF8 UE8M0 byte layout for SFA (per-token) / SFB (per-128-row block).
//
// CUTLASS layout (SFVecSize=32, Blk_MN=128, Blk_SF=4) is produced by
// `Sm1xxBlkScaledConfig<32>::tile_atom_to_shape_SFA({M, N, K, 1})`. We
// compute the tile's element offset for each (m, k_32block) coordinate from
// the runtime layout struct.
//
// To avoid needing the raw Layout-functor evaluation on device, we use a
// pre-computed cache: for each expert we store the LAYOUT struct, then the
// kernel evaluates the layout in-place (simple `layout(m, k, 0)` call).
// ============================================================================
template <typename Cfg, typename LayoutOut>
__global__ void pack_sfa_per_expert_kernel(
    const float* __restrict__ scale_fp32,      // [M, K/128] per-token pow2 fp32
    const int* __restrict__ expert_offsets,     // [E]
    const int* __restrict__ sfa_byte_offsets,   // [E+1]
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* layouts,  // [E]
    typename MxF8GemmBuilder<Cfg, LayoutOut>::Sm1xxBlkScaledConfig::SfAtom sfatom,
    int K,
    int K_blocks_128,  // K/128
    cutlass::float_ue8m0_t* sfa_out  // flat UE8M0 output, per-expert regions
) {
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;

  int e  = blockIdx.z;
  int m  = blockIdx.y;        // local m within expert
  int kb = blockIdx.x;        // K-block-128 index
  int tid = threadIdx.x;

  // m_count = counts[e] = expert_offsets[e+1] - expert_offsets[e]
  int expert_start = expert_offsets[e];
  int next_start   = expert_offsets[e + 1];
  int m_count = next_start - expert_start;
  if (m >= m_count || kb >= K_blocks_128) return;

  int global_m = expert_start + m;
  float fp32_val = scale_fp32[global_m * K_blocks_128 + kb];  // pow2 value
  uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(fp32_val);

  // Per-expert sub-layout.
  auto layout = layouts[e];
  cutlass::float_ue8m0_t* out = sfa_out + sfa_byte_offsets[e];

  // The 128-element K-block corresponds to 4 sub-blocks of 32 elements. Write
  // the same UE8M0 byte at each of those 4 sub-block positions, at row m.
  // Layout indexing: layout(m, k, 0) -> offset.
  for (int sub = tid; sub < 4; sub += blockDim.x) {
    int k = kb * 128 + sub * 32;
    int off = layout(m, k, 0);
    out[off] = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
  }
}

// Similar for SFB (per-128-row blocks of N; per-32 K-sub-blocks).
template <typename Cfg, typename LayoutOut>
__global__ void pack_sfb_per_expert_kernel(
    const float* __restrict__ scale_fp32,       // [E, N/128, K/128] per-row-block pow2 fp32
    const int* __restrict__ sfb_byte_offsets,    // [E+1]
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFB* layouts,  // [E]
    int E, int N, int K,
    int N_blocks_128,
    int K_blocks_128,
    cutlass::float_ue8m0_t* sfb_out
) {
  int e  = blockIdx.z;
  int n  = blockIdx.y;        // local n within expert
  int kb = blockIdx.x;
  int tid = threadIdx.x;

  if (e >= E || n >= N || kb >= K_blocks_128) return;
  int nb = n / 128;
  float fp32_val = scale_fp32[(e * N_blocks_128 + nb) * K_blocks_128 + kb];
  uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(fp32_val);

  auto layout = layouts[e];
  cutlass::float_ue8m0_t* out = sfb_out + sfb_byte_offsets[e];

  for (int sub = tid; sub < 4; sub += blockDim.x) {
    int k = kb * 128 + sub * 32;
    int off = layout(n, k, 0);
    out[off] = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
  }
}

// ============================================================================
// Main MxF8 MoE grouped GEMM entry. Same interface as moe_blockwise_grouped_mm_v2
// but expects `scales_a`, `scales_b` to be POW-OF-2 fp32 values (from the
// transcode kernel output). The payloads `a`, `b` must already be transcoded
// in place (sign flip + residual absorbed). Caller must pre-compute
// sfa_byte_offsets[E+1] and sfb_byte_offsets[E+1] based on problem_sizes[e]
// and the per-expert layout size.
// ============================================================================
void moe_mxf8_grouped_mm(
    torch::Tensor& output,           // [total_tokens, N] bf16
    torch::Tensor const& a,          // [total_tokens, K] fp8 (transcoded)
    torch::Tensor const& b,          // [E, N, K] fp8 (transcoded)
    torch::Tensor const& scales_a,   // [total_tokens, K/128] fp32 pow-of-2
    torch::Tensor const& scales_b,   // [E, N/128, K/128] fp32 pow-of-2
    torch::Tensor const& expert_offsets,   // [E+1] int32 (inclusive scan; [0] = 0, [E] = total)
    torch::Tensor const& problem_sizes,    // [E, 3] int32
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& sfa_buffer,       // UE8M0 packed scales for A (flat)
    torch::Tensor& sfb_buffer,       // UE8M0 packed scales for B (flat)
    torch::Tensor const& sfa_byte_offsets,  // [E+1] int32
    torch::Tensor const& sfb_byte_offsets,  // [E+1] int32
    torch::Tensor const& workspace)
{
  int total_tokens = a.size(0);
  int K = a.size(1);
  int E = b.size(0);
  int N = b.size(1);
  TORCH_CHECK(b.size(2) == K, "b K mismatch");
  TORCH_CHECK(output.size(0) == total_tokens && output.size(1) == N, "output shape mismatch");

  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  // 1) Pack fp32 pow2 scales into UE8M0 layout per expert.
  // SFA: per-token, SFB: per-128-row-block.
  dim3 sfa_grid(K / 128, total_tokens / E + 1, E);  // over-approximates; kernel guards with m_count
  // Cleaner: grid.y = max per-expert m across all experts; conservative pick total_tokens/E+1.
  // For correctness we iterate m up to m_count inside the kernel.
  // Use a single max(m_e) across experts: easier is to just launch total_tokens Y (upper bound of per-expert m).
  int max_m_per_expert = 0;
  {
    auto ps_cpu = problem_sizes.cpu();
    auto* p = ps_cpu.data_ptr<int32_t>();
    for (int e = 0; e < E; ++e) max_m_per_expert = std::max(max_m_per_expert, p[e * 3]);
  }
  sfa_grid = dim3(K / 128, max_m_per_expert, E);
  typename SfConfig::SfAtom sfatom{};
  pack_sfa_per_expert_kernel<Cfg, LayoutOut><<<sfa_grid, 4, 0, stream>>>(
      scales_a.data_ptr<float>(),
      expert_offsets.data_ptr<int>(),
      sfa_byte_offsets.data_ptr<int>(),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(layout_sfa.data_ptr()),
      sfatom,
      K, K / 128,
      reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr()));

  dim3 sfb_grid(K / 128, N, E);
  pack_sfb_per_expert_kernel<Cfg, LayoutOut><<<sfb_grid, 4, 0, stream>>>(
      scales_b.data_ptr<float>(),
      sfb_byte_offsets.data_ptr<int>(),
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(layout_sfb.data_ptr()),
      E, N, K, N / 128, K / 128,
      reinterpret_cast<cutlass::float_ue8m0_t*>(sfb_buffer.data_ptr()));

  // 2) Build ptr arrays, strides, per-expert layouts.
  get_group_gemm_starts_kernel_mxf8<Cfg, LayoutOut><<<1, E, 0, stream>>>(
      expert_offsets.data_ptr<int>(),
      sfa_byte_offsets.data_ptr<int>(),
      sfb_byte_offsets.data_ptr<int>(),
      static_cast<ElementAB**>(a_ptrs.data_ptr()),
      static_cast<ElementAB**>(b_ptrs.data_ptr()),
      static_cast<ElementC**>(out_ptrs.data_ptr()),
      static_cast<ElementSF_mx**>(sfa_ptrs.data_ptr()),
      static_cast<ElementSF_mx**>(sfb_ptrs.data_ptr()),
      static_cast<ElementAB*>(const_cast<void*>(a.data_ptr())),
      static_cast<ElementAB*>(const_cast<void*>(b.data_ptr())),
      static_cast<ElementC*>(output.data_ptr()),
      reinterpret_cast<ElementSF_mx*>(sfa_buffer.data_ptr()),
      reinterpret_cast<ElementSF_mx*>(sfb_buffer.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(layout_sfa.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(layout_sfb.data_ptr()),
      reinterpret_cast<typename MxFB::StrideA*>(const_cast<void*>(stride_a.data_ptr())),
      reinterpret_cast<typename MxFB::StrideB*>(const_cast<void*>(stride_b.data_ptr())),
      reinterpret_cast<typename MxFB::StrideC*>(const_cast<void*>(stride_c.data_ptr())),
      problem_sizes.data_ptr<int>());

  // 3) Launch CUTLASS MxF8 grouped GEMM.
  launch_mxf8_group_gemm<Cfg, LayoutOut>(
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb,
      problem_sizes, workspace, E, stream);
}

// Helper: compute per-expert SFA buffer size (bytes = UE8M0 count) from
// problem_sizes. Returns {cumulative_offsets[E+1], total_size_elems}.
std::tuple<std::vector<int32_t>, int64_t>
compute_mxf8_sfa_layout_offsets_host(torch::Tensor const& problem_sizes) {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;

  int E = problem_sizes.size(0);
  auto ps_cpu = problem_sizes.cpu();
  auto* p = ps_cpu.data_ptr<int32_t>();

  std::vector<int32_t> offsets(E + 1, 0);
  for (int e = 0; e < E; ++e) {
    int m_e = p[e * 3];
    int n_e = p[e * 3 + 1];
    int k_e = p[e * 3 + 2];
    auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(cute::make_shape(m_e, n_e, k_e, 1));
    int size_e = cute::size(cute::filter_zeros(layout_sfa));
    offsets[e + 1] = offsets[e] + size_e;
  }
  return {offsets, offsets.back()};
}

std::tuple<std::vector<int32_t>, int64_t>
compute_mxf8_sfb_layout_offsets_host(torch::Tensor const& problem_sizes) {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;

  int E = problem_sizes.size(0);
  auto ps_cpu = problem_sizes.cpu();
  auto* p = ps_cpu.data_ptr<int32_t>();

  std::vector<int32_t> offsets(E + 1, 0);
  for (int e = 0; e < E; ++e) {
    int m_e = p[e * 3];
    int n_e = p[e * 3 + 1];
    int k_e = p[e * 3 + 2];
    auto layout_sfb = SfConfig::tile_atom_to_shape_SFB(cute::make_shape(m_e, n_e, k_e, 1));
    int size_e = cute::size(cute::filter_zeros(layout_sfb));
    offsets[e + 1] = offsets[e] + size_e;
  }
  return {offsets, offsets.back()};
}

int64_t get_mxf8_sizes_stride() {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  return std::max({
      sizeof(typename MxFB::StrideA), sizeof(typename MxFB::StrideB), sizeof(typename MxFB::StrideC)
  });
}

int64_t get_mxf8_sizes_layout_sfa() {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  return sizeof(typename MxFB::InternalLayoutSFA);
}

int64_t get_mxf8_sizes_layout_sfb() {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  return sizeof(typename MxFB::InternalLayoutSFB);
}

// Debug helper: evaluate SFA layout at a few (m, k) points on the host and
// return the resulting offsets. Helps diagnose whether our packing kernel's
// `layout(m, k, 0)` indexing is consistent with what CUTLASS expects.
std::vector<int64_t> probe_mxf8_sfa_layout(int m, int n, int k) {
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;

  auto layout = SfConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  std::vector<int64_t> offsets;
  offsets.push_back(static_cast<int64_t>(cute::size(cute::filter_zeros(layout))));  // total size
  // Probe (0,0), (0,1), (0,31), (0,32), (0,63), (0,64), (1,0), (127,0), (127,31), (127,32).
  offsets.push_back(layout(0, 0, 0));
  offsets.push_back(layout(0, 1, 0));
  offsets.push_back(layout(0, 31, 0));
  offsets.push_back(layout(0, 32, 0));
  offsets.push_back(layout(0, 63, 0));
  offsets.push_back(layout(0, 64, 0));
  offsets.push_back(layout(1, 0, 0));
  offsets.push_back(layout(127, 0, 0));
  if (m >= 128) {
    offsets.push_back(layout(128, 0, 0));
  } else {
    offsets.push_back(-1);
  }
  return offsets;
}

// Expose sizeof() info for Python to size workspace tensors correctly.
// We use max across configs to be safe. Stride/Layout types are the same across
// configs since they only differ by template-static numeric params.
std::tuple<int64_t, int64_t, int64_t> get_sizes() {
  using GBL = GemmBuilder<CfgLargeM, cutlass::layout::RowMajor>;
  using GBM = GemmBuilder<CfgMidM, cutlass::layout::RowMajor>;
  using GBS = GemmBuilder<CfgSmallM, cutlass::layout::ColumnMajor>;
  int64_t stride_sz = std::max({
      sizeof(typename GBL::StrideA), sizeof(typename GBL::StrideB), sizeof(typename GBL::StrideC),
      sizeof(typename GBM::StrideA), sizeof(typename GBM::StrideB), sizeof(typename GBM::StrideC),
      sizeof(typename GBS::StrideA), sizeof(typename GBS::StrideB), sizeof(typename GBS::StrideC)});
  int64_t sfa_sz = std::max({sizeof(typename CfgLargeM::LayoutSFA),
                             sizeof(typename CfgMidM::LayoutSFA),
                             sizeof(typename CfgSmallM::LayoutSFA)});
  int64_t sfb_sz = std::max({sizeof(typename CfgLargeM::LayoutSFB),
                             sizeof(typename CfgMidM::LayoutSFB),
                             sizeof(typename CfgSmallM::LayoutSFB)});
  return {stride_sz, sfa_sz, sfb_sz};
}

// Query workspace size for a given (max_total_tokens, E, N, K). Conservative upper bound.
int64_t get_workspace_size(int max_total_tokens, int E, int N, int K, bool use_small_m) {
  (void)max_total_tokens; (void)E; (void)N; (void)K; (void)use_small_m;
  return 64 * 1024 * 1024;
}
'''

# -----------------------------------------------------------------------------
# Light-compile file: fused helper kernels (SwiGLU+requant, weighted scatter).
# Separated from CUTLASS-heavy file so that iterating on these kernels does NOT
# trigger a full CUTLASS re-compile (which takes ~3-5 min).
# Both files link together via PYBIND11_MODULE defined in this file; the CUTLASS
# functions are declared `extern` here.
# -----------------------------------------------------------------------------
_MOE_FUSED_CU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Forward declarations of CUTLASS-backed functions defined in moe_cutlass.cu.
// Signatures must match exactly.
void moe_blockwise_grouped_mm_v2(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scales_a,
    torch::Tensor const& scales_b,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor& problem_sizes_transpose,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor const& workspace);

std::tuple<int64_t, int64_t, int64_t> get_sizes();
int64_t get_workspace_size(int max_total_tokens, int E, int N, int K, bool use_small_m);

// v18 MxF8 transcode forward decls.
void mxf8_transcode_activations(torch::Tensor& payload, torch::Tensor const& scale, torch::Tensor& scale_abs);
void mxf8_transcode_weights_impl(torch::Tensor& payload, torch::Tensor const& scale, torch::Tensor& scale_abs);
void moe_mxf8_grouped_mm(
    torch::Tensor& output, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scales_a, torch::Tensor const& scales_b,
    torch::Tensor const& expert_offsets, torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b, torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb,
    torch::Tensor& sfa_buffer, torch::Tensor& sfb_buffer,
    torch::Tensor const& sfa_byte_offsets, torch::Tensor const& sfb_byte_offsets,
    torch::Tensor const& workspace);
std::tuple<std::vector<int32_t>, int64_t> compute_mxf8_sfa_layout_offsets_host(torch::Tensor const&);
std::tuple<std::vector<int32_t>, int64_t> compute_mxf8_sfb_layout_offsets_host(torch::Tensor const&);
int64_t get_mxf8_sizes_stride();
int64_t get_mxf8_sizes_layout_sfa();
int64_t get_mxf8_sizes_layout_sfb();
std::vector<int64_t> probe_mxf8_sfa_layout(int m, int n, int k);

// ============================================================================
// Fused SwiGLU + per-row FP8 requant kernel.
//
// In:   gemm1_out  [M, N1]       bf16  — GEMM1 output where first half = gate, second = up
// Out:  act_q      [M, N1/2]     fp8   — quantized act = gate * SiLU(up) / row_scale
//       row_scales [M]           fp32  — per-row quantization scale (max/|act| * (1/448))
//       broadcast_scales [M, N1/2/128] fp32 — SFA layout input for GEMM2 (broadcast row_scale
//                                              across all K-blocks, since activation has a
//                                              single per-row scale)
//
// One kernel replaces ~12 PyTorch ops: .float(), slice, sigmoid, mul, amax,
// clamp, div, cast-to-fp8, broadcast-and-contiguous for SFA.
//
// Launch: grid=(M,), block=256 threads per row. Each thread handles N1/2 / 256
//         elements. Block-wide reduction via shared mem gets row absmax.
// ============================================================================
template <int H_>
__global__ void swiglu_fp8_requant_kernel(
    const __nv_bfloat16* __restrict__ gemm1_out,   // [M, 2*H]  bf16
    __nv_fp8_e4m3*       __restrict__ act_q,        // [M, H]    fp8
    float*               __restrict__ row_scales,   // [M]       fp32
    float*               __restrict__ broadcast_scales,  // [M, H/128] fp32
    int M)
{
  constexpr int H = H_;
  constexpr int TB = 256;  // threads per block
  const int m = blockIdx.x;
  if (m >= M) return;
  const __nv_bfloat16* row_in = gemm1_out + m * (2 * H);
  __nv_fp8_e4m3*       row_q  = act_q + m * H;

  const int tid = threadIdx.x;

  __shared__ float s_absmax;
  if (tid == 0) s_absmax = 0.0f;
  __syncthreads();

  // First pass: compute act[i] = gate[i] * up[i] * sigmoid(up[i]) and local
  // absmax. Cache the ITERS act values in REGISTERS (each thread holds H/TB
  // floats, 8 floats for H=2048,TB=256 = 32B/thread) so pass 2 just scales +
  // casts without re-doing the expensive expf(-u). Saves ~30% kernel time.
  constexpr int ITERS = H / TB;
  float act_cache[ITERS];
  float thread_absmax = 0.0f;
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = tid + j * TB;
    float g = __bfloat162float(row_in[i]);
    float u = __bfloat162float(row_in[H + i]);
    float act = g * (u * (0.5f + 0.5f * __tanhf(u * 0.5f)));
    act_cache[j] = act;
    thread_absmax = fmaxf(thread_absmax, fabsf(act));
  }
  // Block reduce to find row max. Pattern: warp-reduce → store 8 partials in
  // smem → warp 0 reduces the 8 partials. All 32 lanes of warp 0 must
  // participate in the second __shfl_xor_sync (padded lanes read -INFINITY).
  constexpr int NW = TB / 32;  // number of warps = 8
  __shared__ float s_partial[NW];
  float v = thread_absmax;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  if ((tid & 31) == 0) s_partial[tid >> 5] = v;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < NW) ? s_partial[tid] : -INFINITY;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, offset));
    }
    if (tid == 0) s_absmax = w;
  }
  __syncthreads();

  float row_max = fmaxf(s_absmax, 1e-8f);
  float scale = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float inv_scale = 1.0f / scale;

  if (tid == 0) {
    row_scales[m] = scale;
  }

  // Broadcast the row scale across all K-blocks for SFA input to GEMM2.
  constexpr int KBLOCKS = H / 128;
  if (tid < KBLOCKS) {
    broadcast_scales[m * KBLOCKS + tid] = scale;
  }

  // Second pass: read cached act from registers, scale, cast to fp8.
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = tid + j * TB;
    float q = act_cache[j] * inv_scale;
    row_q[i] = __nv_fp8_e4m3(q);
  }
}

// ============================================================================
// v17: Fused SwiGLU+requant with routing weight absorption into the GEMM2 A
// scale. This is a clean architectural change: multiplying broadcast_scale by
// weights[m] means GEMM2's accumulator already contains the weighted value, so
// reduce_scatter no longer needs the `w * v` multiply (saves one multiply per
// element, ~5-15μs at large T). The row_scale (used nowhere downstream now
// that broadcast already encodes the weight) is still written for diagnostics.
// ============================================================================
template <int H_>
__global__ void swiglu_fp8_requant_weighted_kernel(
    const __nv_bfloat16* __restrict__ gemm1_out,       // [M, 2*H]  bf16
    const float*         __restrict__ sorted_weights,  // [M]       fp32
    __nv_fp8_e4m3*       __restrict__ act_q,           // [M, H]    fp8
    float*               __restrict__ row_scales,      // [M]       fp32 (for diagnostics)
    float*               __restrict__ broadcast_scales, // [M, H/128] fp32 scale*weight
    int M)
{
  constexpr int H = H_;
  constexpr int TB = 256;
  const int m = blockIdx.x;
  if (m >= M) return;
  const __nv_bfloat16* row_in = gemm1_out + m * (2 * H);
  __nv_fp8_e4m3*       row_q  = act_q + m * H;

  const int tid = threadIdx.x;

  __shared__ float s_absmax;
  if (tid == 0) s_absmax = 0.0f;
  __syncthreads();

  constexpr int ITERS = H / TB;
  float act_cache[ITERS];
  float thread_absmax = 0.0f;
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = tid + j * TB;
    float g = __bfloat162float(row_in[i]);
    float u = __bfloat162float(row_in[H + i]);
    float act = g * (u * (0.5f + 0.5f * __tanhf(u * 0.5f)));
    act_cache[j] = act;
    thread_absmax = fmaxf(thread_absmax, fabsf(act));
  }
  constexpr int NW = TB / 32;
  __shared__ float s_partial[NW];
  float v = thread_absmax;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  if ((tid & 31) == 0) s_partial[tid >> 5] = v;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < NW) ? s_partial[tid] : -INFINITY;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, offset));
    }
    if (tid == 0) s_absmax = w;
  }
  __syncthreads();

  float row_max = fmaxf(s_absmax, 1e-8f);
  float scale = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float inv_scale = 1.0f / scale;

  // Fold routing weight into the per-row A scale of GEMM2.
  float w_route = sorted_weights[m];
  float scale_weighted = scale * w_route;

  if (tid == 0) {
    row_scales[m] = scale;  // unweighted, diagnostic only
  }

  constexpr int KBLOCKS = H / 128;
  if (tid < KBLOCKS) {
    broadcast_scales[m * KBLOCKS + tid] = scale_weighted;
  }

  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = tid + j * TB;
    float q = act_cache[j] * inv_scale;
    row_q[i] = __nv_fp8_e4m3(q);
  }
}

void swiglu_fp8_requant_weighted(
    torch::Tensor const& gemm1_out,
    torch::Tensor const& sorted_weights,
    torch::Tensor&       act_q,
    torch::Tensor&       row_scales,
    torch::Tensor&       broadcast_scales)
{
  TORCH_CHECK(gemm1_out.is_cuda() && gemm1_out.dim() == 2);
  TORCH_CHECK(gemm1_out.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(sorted_weights.is_cuda() && sorted_weights.scalar_type() == torch::kFloat32);
  int M = gemm1_out.size(0);
  int N1 = gemm1_out.size(1);
  int H = N1 / 2;
  TORCH_CHECK(N1 % 2 == 0 && H % 128 == 0, "H must be /128-aligned");
  TORCH_CHECK(act_q.size(0) == M && act_q.size(1) == H);
  TORCH_CHECK(act_q.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(row_scales.size(0) == M);
  TORCH_CHECK(sorted_weights.size(0) == M);
  TORCH_CHECK(broadcast_scales.size(0) == M && broadcast_scales.size(1) == H / 128);

  auto stream = at::cuda::getCurrentCUDAStream(gemm1_out.get_device()).stream();

  #define LAUNCH_HW(HCONST)                                                    \
    do {                                                                        \
      swiglu_fp8_requant_weighted_kernel<HCONST>                               \
          <<<M, 256, 0, stream>>>(                                              \
              reinterpret_cast<const __nv_bfloat16*>(gemm1_out.data_ptr()),    \
              sorted_weights.data_ptr<float>(),                                \
              reinterpret_cast<__nv_fp8_e4m3*>(act_q.data_ptr()),              \
              row_scales.data_ptr<float>(),                                    \
              broadcast_scales.data_ptr<float>(), M);                          \
    } while (0)

  switch (H) {
    case 1024: LAUNCH_HW(1024); break;
    case 2048: LAUNCH_HW(2048); break;
    case 4096: LAUNCH_HW(4096); break;
    default:
      TORCH_CHECK(false, "Unsupported H=", H);
  }
  #undef LAUNCH_HW
}

// ============================================================================
// v17/v18: reduce_scatter variant that does NOT multiply by weights (they are
// already baked into the GEMM2 output by swiglu_fp8_requant_weighted).
// ============================================================================
__global__ void reduce_scatter_unweighted_kernel(
    const __nv_bfloat16* __restrict__ gemm2_out,
    const int*           __restrict__ token_offsets,
    const int*           __restrict__ token_perm,
    __nv_bfloat16*       __restrict__ out,
    int T, int N2)
{
  int t = blockIdx.x;
  if (t >= T) return;
  int beg = token_offsets[t];
  int end = token_offsets[t + 1];

  const int TB = blockDim.x;
  int lane = threadIdx.x;

  // v19-tight: 128-bit vectorized IO. Each uint4 carries 4 bf162 pairs = 8 bf16.
  // At N2=7168 we have 896 uint4 units per row, matching nicely with TB=128.
  const int n_vec = N2 / 8;  // number of uint4 along N

  const uint4* out_v_base = reinterpret_cast<const uint4*>(out + t * N2);
  uint4*       out_v      = const_cast<uint4*>(out_v_base);

  #pragma unroll 2
  for (int j = lane; j < n_vec; j += TB) {
    float2 a0 = make_float2(0.0f, 0.0f);
    float2 a1 = make_float2(0.0f, 0.0f);
    float2 a2 = make_float2(0.0f, 0.0f);
    float2 a3 = make_float2(0.0f, 0.0f);
    for (int k = beg; k < end; ++k) {
      int m = token_perm[k];
      const uint4 v = reinterpret_cast<const uint4*>(gemm2_out + m * N2)[j];
      __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
      __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
      __nv_bfloat162 p2 = *reinterpret_cast<const __nv_bfloat162*>(&v.z);
      __nv_bfloat162 p3 = *reinterpret_cast<const __nv_bfloat162*>(&v.w);
      a0.x += __bfloat162float(p0.x); a0.y += __bfloat162float(p0.y);
      a1.x += __bfloat162float(p1.x); a1.y += __bfloat162float(p1.y);
      a2.x += __bfloat162float(p2.x); a2.y += __bfloat162float(p2.y);
      a3.x += __bfloat162float(p3.x); a3.y += __bfloat162float(p3.y);
    }
    uint4 packed;
    __nv_bfloat162 r0 = __floats2bfloat162_rn(a0.x, a0.y);
    __nv_bfloat162 r1 = __floats2bfloat162_rn(a1.x, a1.y);
    __nv_bfloat162 r2 = __floats2bfloat162_rn(a2.x, a2.y);
    __nv_bfloat162 r3 = __floats2bfloat162_rn(a3.x, a3.y);
    packed.x = *reinterpret_cast<const uint32_t*>(&r0);
    packed.y = *reinterpret_cast<const uint32_t*>(&r1);
    packed.z = *reinterpret_cast<const uint32_t*>(&r2);
    packed.w = *reinterpret_cast<const uint32_t*>(&r3);
    out_v[j] = packed;
  }
}

void reduce_scatter_unweighted_prebucketed(
    torch::Tensor const& gemm2_out,
    torch::Tensor const& token_offsets,
    torch::Tensor const& token_perm,
    torch::Tensor&       out,
    int T)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 2 == 0);
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  reduce_scatter_unweighted_kernel<<<T, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      token_offsets.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2);
}

void swiglu_fp8_requant(
    torch::Tensor const& gemm1_out,     // [M, N1] bf16
    torch::Tensor&       act_q,          // [M, N1/2] fp8
    torch::Tensor&       row_scales,     // [M] fp32
    torch::Tensor&       broadcast_scales)  // [M, N1/2/128] fp32
{
  TORCH_CHECK(gemm1_out.is_cuda() && gemm1_out.dim() == 2);
  TORCH_CHECK(gemm1_out.scalar_type() == torch::kBFloat16);
  int M = gemm1_out.size(0);
  int N1 = gemm1_out.size(1);
  int H = N1 / 2;
  TORCH_CHECK(N1 % 2 == 0 && H % 128 == 0, "H must be /128-aligned");
  TORCH_CHECK(act_q.size(0) == M && act_q.size(1) == H);
  TORCH_CHECK(act_q.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(row_scales.size(0) == M);
  TORCH_CHECK(broadcast_scales.size(0) == M && broadcast_scales.size(1) == H / 128);

  auto stream = at::cuda::getCurrentCUDAStream(gemm1_out.get_device()).stream();

  // Dispatch on H (the intermediate hidden size). For DeepSeek-V3 MoE this is
  // 2048 always, but also handle 1024/4096 in case of other shapes.
  #define LAUNCH_H(HCONST)                                                      \
    do {                                                                        \
      swiglu_fp8_requant_kernel<HCONST>                                         \
          <<<M, 256, 0, stream>>>(                                              \
              reinterpret_cast<const __nv_bfloat16*>(gemm1_out.data_ptr()),      \
              reinterpret_cast<__nv_fp8_e4m3*>(act_q.data_ptr()),                 \
              row_scales.data_ptr<float>(),                                     \
              broadcast_scales.data_ptr<float>(), M);                            \
    } while (0)

  switch (H) {
    case 1024: LAUNCH_H(1024); break;
    case 2048: LAUNCH_H(2048); break;
    case 4096: LAUNCH_H(4096); break;
    default:
      TORCH_CHECK(false, "Unsupported H=", H);
  }
  #undef LAUNCH_H
}

// ============================================================================
// Fused weighted scatter: computes
//   out[tid[i]] += weights[i] * gemm2_out[i]  (bf16 accumulation via atomics)
//
// Vectorized with __nv_bfloat162 (2 bf16 at a time, 4-byte atomic) — native
// fast-path on Hopper+/Blackwell. Halves the atomic count and uses hardware
// atomic_add.bf16x2 instruction.
// ============================================================================
__global__ void weighted_scatter_kernel(
    const __nv_bfloat16* __restrict__ gemm2_out,    // [M, N2] bf16
    const float*         __restrict__ weights,       // [M]
    const int*           __restrict__ token_ids,     // [M]
    __nv_bfloat16*       __restrict__ out,           // [T, N2] bf16
    int M, int N2, int T)
{
  int m = blockIdx.x;
  if (m >= M) return;
  float w = weights[m];
  if (w == 0.0f) return;
  int tid = token_ids[m];
  if (tid < 0 || tid >= T) return;

  const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(gemm2_out + m * N2);
  __nv_bfloat162*       dst2 = reinterpret_cast<__nv_bfloat162*>(out + tid * N2);

  const int TB = 256;
  const int lane = threadIdx.x;
  const int n2_pairs = N2 / 2;  // bf16 pairs
  __nv_bfloat162 w2 = __float2bfloat162_rn(w);
  #pragma unroll 2
  for (int j = lane; j < n2_pairs; j += TB) {
    __nv_bfloat162 v = src2[j];
    v = __hmul2(v, w2);  // elementwise bf162 multiply
    atomicAdd(dst2 + j, v);  // native bf162 atomic on SM90+
  }
}

void weighted_scatter(
    torch::Tensor const& gemm2_out,
    torch::Tensor const& weights,
    torch::Tensor const& token_ids,
    torch::Tensor&       out,
    int T)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int M = gemm2_out.size(0);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 2 == 0, "N2 must be even for bf162 vectorized scatter");
  TORCH_CHECK(weights.size(0) == M);
  TORCH_CHECK(token_ids.size(0) == M);
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  weighted_scatter_kernel<<<M, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      weights.data_ptr<float>(),
      token_ids.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      M, N2, T);
}

// ============================================================================
// Fused routing kernel for DeepSeek-V3 MoE topk selection.
//
// Replaces ~10 PyTorch ops (sigmoid, add, view, topk×3, scatter, masked_fill,
// gather, sum, div, mul). One block per token, 256 threads = 1 per global
// expert.
//
// Algorithm:
//   1. Each thread i computes s_wb[i] = sigmoid(logits[i]) + bias[i].
//   2. Group-top2: within each warp (GROUP_SIZE=32 experts per group), find
//      the 2 largest s_wb values. Sum them → group_score.
//   3. Top-K_GROUP across 8 group_scores (held in warp 0 via smem).
//   4. Mark experts in non-top-K groups with s_wb = -inf.
//   5. Top-K (K=8) across all 256 experts, emit (expert_id, sigmoid_value).
//   6. Normalize: assign_w[k] = sigmoid[topk_idx[k]] * rsf / sum(sigmoid[topk]).
//
// Constants are hardcoded for DeepSeek-V3 MoE:
//   E_GLOBAL=256, N_GROUP=8, GROUP_SIZE=32, TOPK_GROUP=4, TOP_K=8.
// ============================================================================
__global__ void fused_route_topk_kernel(
    const __nv_bfloat16* __restrict__ routing_logits,  // [T, 256] bf16
    const __nv_bfloat16* __restrict__ routing_bias,    // [256] bf16
    int*                 __restrict__ topk_idx,         // [T, 8] int32 (global expert ids)
    float*               __restrict__ assign_w,         // [T, 8] float32
    int T, float rsf)
{
  constexpr int E_GLOBAL = 256;
  constexpr int N_GROUP = 8;
  constexpr int GROUP_SIZE = 32;  // = warp size
  constexpr int TOPK_GROUP = 4;
  constexpr int TOP_K_VAL = 8;

  int tok = blockIdx.x;
  if (tok >= T) return;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  // Step 1: load + sigmoid + bias
  float logit = __bfloat162float(routing_logits[tok * E_GLOBAL + tid]);
  float bias  = __bfloat162float(routing_bias[tid]);
  float s     = 0.5f + 0.5f * __tanhf(logit * 0.5f);
  float s_wb  = s + bias;

  // Step 2: within-warp (group) top-2 values. Find max, then max with it masked.
  // Warp-level reduction: every lane holds its s_wb; find global max across 32 lanes.
  float v1 = s_wb;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v1 = fmaxf(v1, __shfl_xor_sync(0xffffffff, v1, off));
  }
  // v1 is now the max in every lane. Mask out lanes that equal v1 (possibly
  // multiple if tied, handled by picking the first).
  float v2_in = (s_wb >= v1 - 1e-30f && s_wb <= v1 + 1e-30f) ? -INFINITY : s_wb;
  float v2 = v2_in;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v2 = fmaxf(v2, __shfl_xor_sync(0xffffffff, v2, off));
  }
  float group_score = v1 + v2;  // same in every lane

  // Step 3: collect 8 group_scores (one per warp) into shared mem, pick top-K_GROUP.
  __shared__ float smem_group[N_GROUP];
  __shared__ bool  smem_group_valid[N_GROUP];
  if (lane == 0) {
    smem_group[warp] = group_score;
    smem_group_valid[warp] = false;
  }
  __syncthreads();

  // Top-K_GROUP selection by N_GROUP rounds of find-max-and-mask. Done in warp 0.
  if (warp == 0) {
    float my = (lane < N_GROUP) ? smem_group[lane] : -INFINITY;
    bool selected = false;
    for (int k = 0; k < TOPK_GROUP; ++k) {
      // Find argmax over 8 values held in lanes 0..7.
      float val = my;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
      }
      if (!selected && my == val && lane < N_GROUP) {
        smem_group_valid[lane] = true;
        my = -INFINITY;
        selected = true;
      }
      // Re-propagate `my` so that the SAME winning lane is masked, not a tied one.
      // (Simple handling: each lane just checks its own state above; "val" is the
      // max still held by non-selected lanes.)
    }
  }
  __syncthreads();

  // Step 4: mask non-valid-group experts.
  bool my_group_valid = smem_group_valid[warp];
  float s_wb_filtered = my_group_valid ? s_wb : -INFINITY;

  // Step 5: Top-K over 256 filtered values via K rounds of find-max-and-mask.
  //   Store s_wb_filtered in smem; each round finds the argmax, records it, and
  //   sets that slot to -inf.
  __shared__ float smem_vals[E_GLOBAL];
  smem_vals[tid] = s_wb_filtered;
  __syncthreads();

  __shared__ int   out_idx[TOP_K_VAL];
  __shared__ float out_s_sigmoid[TOP_K_VAL];

  // Per-round block-wide max + index using reduction in smem (simple and correct).
  // We reuse smem_group as scratch for partial reductions across warps.
  for (int k = 0; k < TOP_K_VAL; ++k) {
    float my_val = smem_vals[tid];
    int   my_idx = tid;

    // Warp reduce: keep max + its index.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float other_val = __shfl_xor_sync(0xffffffff, my_val, off);
      int   other_idx = __shfl_xor_sync(0xffffffff, my_idx, off);
      if (other_val > my_val ||
          (other_val == my_val && other_idx < my_idx)) {
        my_val = other_val;
        my_idx = other_idx;
      }
    }
    // Lane 0 of each warp writes partial to smem.
    __shared__ float warp_val[N_GROUP];
    __shared__ int   warp_idx[N_GROUP];
    if (lane == 0) {
      warp_val[warp] = my_val;
      warp_idx[warp] = my_idx;
    }
    __syncthreads();
    // Warp 0 reduces the N_GROUP partials.
    if (warp == 0) {
      float v = (lane < N_GROUP) ? warp_val[lane] : -INFINITY;
      int   i = (lane < N_GROUP) ? warp_idx[lane] : 0;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, v, off);
        int   oi = __shfl_xor_sync(0xffffffff, i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
      }
      if (lane == 0) {
        out_idx[k] = i;
        // Read the pre-mask sigmoid value (not s_wb) for the normalizer.
        // We stored s_wb_filtered in smem_vals; we need s (sigmoid) at index i.
        // Re-read from logits + bias? Easier: every thread already knows its
        // own s. Use a small broadcast via smem.
        out_s_sigmoid[k] = 0.0f;  // filled below by thread i in next sync step
      }
    }
    __syncthreads();
    // Thread i writes its sigmoid to the output slot.
    if (tid == out_idx[k]) {
      out_s_sigmoid[k] = s;  // pre-bias sigmoid
    }
    // Mask this slot so next iteration picks the next-largest.
    __syncthreads();
    if (tid == out_idx[k]) {
      smem_vals[tid] = -INFINITY;
    }
    __syncthreads();
  }

  // Step 6: normalize. Thread 0 does this (8 values).
  if (tid == 0) {
    float sum_s = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) sum_s += out_s_sigmoid[k];
    float scale = rsf / (sum_s + 1e-20f);
    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) {
      topk_idx[tok * TOP_K_VAL + k] = out_idx[k];
      assign_w[tok * TOP_K_VAL + k] = out_s_sigmoid[k] * scale;
    }
  }
}

void fused_route_topk(
    torch::Tensor const& routing_logits,  // [T, 256] bf16
    torch::Tensor const& routing_bias,    // [256] bf16
    torch::Tensor&       topk_idx,         // [T, 8] int32
    torch::Tensor&       assign_w,         // [T, 8] float32
    float rsf)
{
  TORCH_CHECK(routing_logits.is_cuda() && routing_logits.dim() == 2);
  TORCH_CHECK(routing_logits.size(1) == 256, "expected E_GLOBAL=256");
  TORCH_CHECK(routing_logits.scalar_type() == torch::kBFloat16, "logits must be bf16");
  TORCH_CHECK(routing_bias.numel() == 256);
  int T = routing_logits.size(0);
  TORCH_CHECK(topk_idx.size(0) == T && topk_idx.size(1) == 8);
  TORCH_CHECK(topk_idx.scalar_type() == torch::kInt32);
  TORCH_CHECK(assign_w.size(0) == T && assign_w.size(1) == 8);
  TORCH_CHECK(assign_w.scalar_type() == torch::kFloat32);

  auto stream = at::cuda::getCurrentCUDAStream(routing_logits.get_device()).stream();
  fused_route_topk_kernel<<<T, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(routing_logits.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
      topk_idx.data_ptr<int>(),
      assign_w.data_ptr<float>(),
      T, rsf);
}

// ============================================================================
// Fused gather: pulls rows of `hidden_states` (FP8) AND `hidden_states_scale`
// (fp32) into pre-allocated packed buffers in a single kernel launch, replacing
// `hidden_states[sorted_tids.long()]` + `hs_scale_t[sorted_tids.long()]` which
// cost ~70μs/iter of PyTorch-index overhead.
//
// Launch: one block per destination row; threads cooperatively copy K1 fp8
// elements + K1/128 fp32 scales per row, using vectorized 16B loads/stores.
// ============================================================================
__global__ void fused_gather_hidden_scales_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden_states,    // [T, K1] fp8 (row-major)
    const float*         __restrict__ hs_scale,          // arbitrary strided layout
    int                   hs_scale_stride_t,              // stride to advance token
    int                   hs_scale_stride_b,              // stride to advance K-block
    const int*           __restrict__ sorted_tids,       // [M] int32
    __nv_fp8_e4m3*       __restrict__ packed_acts,       // [M, K1] fp8
    float*               __restrict__ packed_act_scales, // [M, K1/128] fp32
    int T, int K1, int K1_blocks)
{
  int m = blockIdx.x;
  int tid_src = sorted_tids[m];
  if (tid_src < 0 || tid_src >= T) tid_src = 0;

  const __nv_fp8_e4m3* src = hidden_states + tid_src * K1;
  __nv_fp8_e4m3*       dst = packed_acts    + m       * K1;
  const float*         ssrc_base = hs_scale + tid_src * hs_scale_stride_t;
  float*               sdst = packed_act_scales + m   * K1_blocks;

  int lane = threadIdx.x;
  int BW = blockDim.x;
  const uint4* src_v = reinterpret_cast<const uint4*>(src);
  uint4*       dst_v = reinterpret_cast<uint4*>(dst);
  int n_v = K1 / 16;
  #pragma unroll 2
  for (int i = lane; i < n_v; i += BW) {
    dst_v[i] = src_v[i];
  }
  // Scales: read from strided src (handles both [T, K/128] and [K/128, T] layouts).
  for (int i = lane; i < K1_blocks; i += BW) {
    sdst[i] = ssrc_base[i * hs_scale_stride_b];
  }
}

void fused_gather_hidden_scales(
    torch::Tensor const& hidden_states,
    torch::Tensor const& hs_scale,
    torch::Tensor const& sorted_tids,
    torch::Tensor&       packed_acts,
    torch::Tensor&       packed_act_scales)
{
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dim() == 2);
  TORCH_CHECK(hs_scale.is_cuda() && hs_scale.dim() == 2);
  TORCH_CHECK(sorted_tids.is_cuda() && sorted_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_acts.is_cuda() && packed_act_scales.is_cuda());

  int T = hidden_states.size(0);
  int K1 = hidden_states.size(1);
  int K1_blocks = packed_act_scales.size(1);
  int M = sorted_tids.size(0);
  TORCH_CHECK(K1 % 16 == 0, "K1 must be multiple of 16 for vectorized fp8 gather");
  TORCH_CHECK(packed_acts.size(0) == M && packed_acts.size(1) == K1);
  TORCH_CHECK(packed_act_scales.size(0) == M);

  // Figure out strides: which dim is T (length-T), which is K/128?
  int hs_size_0 = hs_scale.size(0);
  int hs_size_1 = hs_scale.size(1);
  int hs_stride_0 = hs_scale.stride(0);
  int hs_stride_1 = hs_scale.stride(1);

  int stride_t, stride_b;
  // Prefer matching the K-block dim first. This resolves the T == K1_blocks
  // tie-case (e.g. T=56, K1=7168 -> K1_blocks=56) correctly: contest data is
  // provided as [K/128, T] so we must interpret dim 0 as the K-block axis.
  if (hs_size_0 == K1_blocks && hs_size_1 == T) {
    // [K/128, T]: token stride = stride(1), block stride = stride(0).
    stride_t = hs_stride_1;
    stride_b = hs_stride_0;
  } else if (hs_size_0 == T && hs_size_1 == K1_blocks) {
    // [T, K/128]: token stride = stride(0), block stride = stride(1).
    stride_t = hs_stride_0;
    stride_b = hs_stride_1;
  } else {
    TORCH_CHECK(false,
                "hs_scale shape doesn't match either [T, K/128] or [K/128, T]");
  }

  auto stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();
  fused_gather_hidden_scales_kernel<<<M, 128, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      hs_scale.data_ptr<float>(),
      stride_t, stride_b,
      sorted_tids.data_ptr<int>(),
      reinterpret_cast<__nv_fp8_e4m3*>(packed_acts.data_ptr()),
      packed_act_scales.data_ptr<float>(),
      T, K1, K1_blocks);
}

// ============================================================================
// Repack expert-sorted rows into DeepGEMM-style contiguous grouped layout.
//
// Input rows are already grouped by expert with compact offsets `expert_offsets`
// and lengths `counts`. Output rows are padded so each expert segment length is
// rounded up to `alignment`, with metadata filled for grouped contiguous GEMM.
// ============================================================================
__global__ void build_aligned_offsets_kernel(
    const int* __restrict__ counts,   // [E]
    int*       __restrict__ offsets,  // [E+1]
    int E,
    int alignment)
{
  if (blockIdx.x != 0) return;
  int lane = threadIdx.x;
  int c = (lane < E) ? counts[lane] : 0;
  int aligned = c > 0 ? ((c + alignment - 1) / alignment) * alignment : 0;
  int v = aligned;
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int n = __shfl_up_sync(0xffffffff, v, off);
    if (lane >= off) v += n;
  }
  if (lane < E) offsets[lane] = v - aligned;
  if (lane == 31) offsets[E] = v;
}

__global__ void repack_aligned_expert_layout_kernel(
    const __nv_fp8_e4m3* __restrict__ in_acts,     // [M, K]
    const float*         __restrict__ in_scales,   // [M, KB]
    const int*           __restrict__ in_tids,     // [M]
    const float*         __restrict__ in_weights,  // [M]
    const int*           __restrict__ expert_offsets, // [E]
    const int*           __restrict__ counts,         // [E]
    const int*           __restrict__ aligned_offsets,// [E+1]
    __nv_fp8_e4m3*       __restrict__ out_acts,    // [M_aligned, K]
    float*               __restrict__ out_scales,  // [M_aligned, KB]
    int*                 __restrict__ out_tids,    // [M_aligned]
    float*               __restrict__ out_weights, // [M_aligned]
    int*                 __restrict__ grouped_layout, // [M_aligned]
    int E,
    int K,
    int KB)
{
  int e = blockIdx.x;
  if (e >= E) return;
  int tid = threadIdx.x;

  int src_off = expert_offsets[e];
  int count = counts[e];
  int dst_off = aligned_offsets[e];
  int aligned_count = aligned_offsets[e + 1] - dst_off;

  int n_vec = K / 16;
  const uint4* in_vec = reinterpret_cast<const uint4*>(in_acts);
  uint4* out_vec = reinterpret_cast<uint4*>(out_acts);

  for (int idx = tid; idx < count * n_vec; idx += blockDim.x) {
    int row = idx / n_vec;
    int vec = idx % n_vec;
    out_vec[(dst_off + row) * n_vec + vec] = in_vec[(src_off + row) * n_vec + vec];
  }
  for (int idx = tid; idx < count * KB; idx += blockDim.x) {
    int row = idx / KB;
    int b = idx % KB;
    out_scales[(dst_off + row) * KB + b] = in_scales[(src_off + row) * KB + b];
  }

  int pad = aligned_count - count;
  for (int idx = tid; idx < pad * n_vec; idx += blockDim.x) {
    int row = idx / n_vec;
    int vec = idx % n_vec;
    out_vec[(dst_off + count + row) * n_vec + vec] = make_uint4(0, 0, 0, 0);
  }
  for (int idx = tid; idx < pad * KB; idx += blockDim.x) {
    int row = idx / KB;
    int b = idx % KB;
    out_scales[(dst_off + count + row) * KB + b] = 1.0f;
  }

  if (tid == 0) {
    for (int row = 0; row < count; ++row) {
      out_tids[dst_off + row] = in_tids[src_off + row];
      out_weights[dst_off + row] = in_weights[src_off + row];
      grouped_layout[dst_off + row] = e;
    }
    for (int row = count; row < aligned_count; ++row) {
      out_tids[dst_off + row] = 0;
      out_weights[dst_off + row] = 0.0f;
      grouped_layout[dst_off + row] = -1;
    }
  }
}

void repack_aligned_expert_layout(
    torch::Tensor const& in_acts,         // [M, K] fp8
    torch::Tensor const& in_scales,       // [M, KB] fp32
    torch::Tensor const& in_tids,         // [M] int32
    torch::Tensor const& in_weights,      // [M] float32
    torch::Tensor const& expert_offsets,  // [E] int32
    torch::Tensor const& counts,          // [E] int32
    int alignment,
    torch::Tensor& aligned_offsets,       // [E+1] int32
    torch::Tensor& out_acts,              // [>=M_aligned, K] fp8
    torch::Tensor& out_scales,            // [>=M_aligned, KB] fp32
    torch::Tensor& out_tids,              // [>=M_aligned] int32
    torch::Tensor& out_weights,           // [>=M_aligned] float32
    torch::Tensor& grouped_layout)        // [>=M_aligned] int32
{
  TORCH_CHECK(in_acts.is_cuda() && in_acts.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(in_scales.is_cuda() && in_scales.scalar_type() == torch::kFloat32);
  TORCH_CHECK(in_tids.is_cuda() && in_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(in_weights.is_cuda() && in_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(counts.is_cuda() && counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(aligned_offsets.is_cuda() && aligned_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(out_acts.is_cuda() && out_acts.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(out_scales.is_cuda() && out_scales.scalar_type() == torch::kFloat32);
  TORCH_CHECK(out_tids.is_cuda() && out_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(out_weights.is_cuda() && out_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(grouped_layout.is_cuda() && grouped_layout.scalar_type() == torch::kInt32);

  int M = in_acts.size(0);
  int K = in_acts.size(1);
  int KB = in_scales.size(1);
  int E = counts.size(0);
  TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");
  TORCH_CHECK(in_scales.size(0) == M && in_tids.numel() == M && in_weights.numel() == M);
  TORCH_CHECK(expert_offsets.numel() == E);
  TORCH_CHECK(aligned_offsets.numel() >= E + 1);
  TORCH_CHECK(out_acts.size(0) >= M && out_acts.size(1) == K);
  TORCH_CHECK(out_scales.size(0) >= M && out_scales.size(1) == KB);
  TORCH_CHECK(out_tids.numel() >= M && out_weights.numel() >= M && grouped_layout.numel() >= M);

  auto stream = at::cuda::getCurrentCUDAStream(in_acts.get_device()).stream();
  build_aligned_offsets_kernel<<<1, 32, 0, stream>>>(
      counts.data_ptr<int>(),
      aligned_offsets.data_ptr<int>(),
      E,
      alignment);
  repack_aligned_expert_layout_kernel<<<E, 256, 0, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(in_acts.data_ptr()),
      in_scales.data_ptr<float>(),
      in_tids.data_ptr<int>(),
      in_weights.data_ptr<float>(),
      expert_offsets.data_ptr<int>(),
      counts.data_ptr<int>(),
      aligned_offsets.data_ptr<int>(),
      reinterpret_cast<__nv_fp8_e4m3*>(out_acts.data_ptr()),
      out_scales.data_ptr<float>(),
      out_tids.data_ptr<int>(),
      out_weights.data_ptr<float>(),
      grouped_layout.data_ptr<int>(),
      E,
      K,
      KB);
}

// ============================================================================
// Token-bucket reduce-scatter: for each output token t, SUM contributions of
// all valid assignments routed to it (weight * gemm2_out row), writing once
// (no atomic, no pre-zero needed). Eliminates the 100+μs pre-zero cost and
// the atomic contention of the previous weighted_scatter design.
//
// Inputs:
//   gemm2_out[M, N2] bf16    — expert-sorted GEMM2 output
//   weights[M] fp32          — per-assignment scaling
//   sorted_tids[M] int32     — target output token for each assignment
//   token_offsets[T+1] int32 — exclusive scan of per-token assignment counts
//   token_perm[M] int32      — assignment indices sorted by output token
// Output:
//   out[T, N2] bf16          — reduced + scaled output (overwritten)
//
// Grid: T blocks (one per output token). Each block reads its N_t assignments
// from perm[offsets[t]..offsets[t+1]], sums them in registers (fp32 accum),
// then writes the row once as bf16.
// ============================================================================
__global__ void reduce_scatter_kernel(
    const __nv_bfloat16* __restrict__ gemm2_out,     // [M, N2] bf16
    const float*         __restrict__ weights,        // [M]
    const int*           __restrict__ token_offsets,  // [T+1]
    const int*           __restrict__ token_perm,     // [M]
    __nv_bfloat16*       __restrict__ out,            // [T, N2] bf16
    int T, int N2)
{
  int t = blockIdx.x;
  if (t >= T) return;
  int beg = token_offsets[t];
  int end = token_offsets[t + 1];

  const int TB = blockDim.x;
  int lane = threadIdx.x;
  const int n2_pairs = N2 / 2;

  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + t * N2);

  // Loop over N2 in TB-stride chunks; each thread handles a set of bf16 pairs.
  #pragma unroll 2
  for (int j = lane; j < n2_pairs; j += TB) {
    float2 acc = make_float2(0.0f, 0.0f);
    for (int k = beg; k < end; ++k) {
      int m = token_perm[k];
      float w = weights[m];
      const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(
          gemm2_out + m * N2);
      __nv_bfloat162 v = src2[j];
      acc.x += __bfloat162float(v.x) * w;
      acc.y += __bfloat162float(v.y) * w;
    }
    out2[j] = __floats2bfloat162_rn(acc.x, acc.y);
  }
}

// Build the per-token bucket map from sorted_tids. Three tiny passes:
//   Pass 1: count per-token assignments (atomicAdd on token_counts[T])
//   Pass 2: exclusive scan on token_counts to get token_offsets[T+1]
//   Pass 3: place each valid index into token_perm via atomicAdd cursor
__global__ void token_bucket_count_kernel(
    const int* __restrict__ sorted_tids, int M, int T, int* token_counts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  int t = sorted_tids[i];
  if (t < 0 || t >= T) return;
  atomicAdd(token_counts + t, 1);
}

__global__ void token_bucket_scan_kernel(
    int* token_counts, int* token_offsets, int T)
{
  // Block-wide scan over T counts. T can be up to ~16k for contest workloads,
  // which fits comfortably in a single block's workload when we use block-
  // stride iteration. Use a straightforward 3-stage scan for simplicity:
  //   1. block-reduce per-thread partial sums
  //   2. warp-scan the reduction results
  //   3. broadcast + local scan
  // For TB=1024 and T up to 16k, we do ceil(T/1024) iterations.
  extern __shared__ int smem[];
  int* partials = smem;  // [num_warps]
  constexpr int TB = 1024;
  constexpr int NW = TB / 32;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int carry = 0;
  for (int base = 0; base < T; base += TB) {
    int idx = base + tid;
    int val = (idx < T) ? token_counts[idx] : 0;

    // Warp-scan (inclusive).
    int v = val;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int n = __shfl_up_sync(0xffffffff, v, off);
      if (lane >= off) v += n;
    }
    if (lane == 31) partials[warp] = v;
    __syncthreads();
    // Scan the per-warp totals in warp 0.
    if (warp == 0) {
      int p = (lane < NW) ? partials[lane] : 0;
      #pragma unroll
      for (int off = 1; off < NW; off <<= 1) {
        int n = __shfl_up_sync(0xffffffff, p, off);
        if (lane >= off) p += n;
      }
      if (lane < NW) partials[lane] = p;
    }
    __syncthreads();
    int warp_prefix = (warp > 0) ? partials[warp - 1] : 0;
    int incl = v + warp_prefix;
    int excl = incl - val;

    if (idx < T) {
      token_offsets[idx] = carry + excl;
    }
    // Update carry with block total (partials[NW-1] after warp-scan).
    int block_total = partials[NW - 1];
    carry += block_total;
    __syncthreads();
  }
  if (tid == 0) {
    token_offsets[T] = carry;
  }
}

__global__ void token_bucket_place_kernel(
    const int* __restrict__ sorted_tids,
    const int* __restrict__ token_offsets,
    int M, int T,
    int* cursors, int* token_perm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  int t = sorted_tids[i];
  if (t < 0 || t >= T) return;
  int pos = atomicAdd(cursors + t, 1);
  token_perm[token_offsets[t] + pos] = i;
}

// Host wrapper for the full reduce-scatter path. Works on pre-allocated
// scratch buffers supplied by Python (token_counts, token_offsets, token_perm,
// cursors, all sized for max(T_max, M_max)).
void reduce_scatter(
    torch::Tensor const& gemm2_out,     // [M, N2] bf16
    torch::Tensor const& weights,        // [M] fp32
    torch::Tensor const& sorted_tids,    // [M] int32
    torch::Tensor&       out,            // [T, N2] bf16
    torch::Tensor&       token_counts,   // [T] int32 (scratch, zero-init internally)
    torch::Tensor&       token_offsets,  // [T+1] int32 (scratch)
    torch::Tensor&       token_perm,     // [M] int32 (scratch)
    int T)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int M = gemm2_out.size(0);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 2 == 0);
  TORCH_CHECK(weights.size(0) == M);
  TORCH_CHECK(sorted_tids.size(0) == M);
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);
  TORCH_CHECK(token_counts.numel() >= T);
  TORCH_CHECK(token_offsets.numel() >= T + 1);
  TORCH_CHECK(token_perm.numel() >= M);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();

  // Zero the counts/cursors (use token_counts buffer as cursors later).
  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  // Pass 1: count per-token assignments.
  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  token_bucket_count_kernel<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), M, T, token_counts.data_ptr<int>());

  // Pass 2: scan counts → offsets.
  // Use TB=1024, smem for (TB/32)=32 warp partials.
  token_bucket_scan_kernel<<<1, 1024, (1024 / 32) * sizeof(int), stream>>>(
      token_counts.data_ptr<int>(), token_offsets.data_ptr<int>(), T);

  // Reuse token_counts as cursor buffer (re-zero it).
  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  // Pass 3: place each assignment into its sorted bucket.
  token_bucket_place_kernel<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), token_offsets.data_ptr<int>(),
      M, T, token_counts.data_ptr<int>(), token_perm.data_ptr<int>());

  // Pass 4: per-token reduce + single bf16 write (no atomics, no pre-zero).
  reduce_scatter_kernel<<<T, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      weights.data_ptr<float>(),
      token_offsets.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2);
}

// v17: full 4-pass reduce-scatter (count + scan + place + reduce) without
// per-element weight multiply. Used when routing weights are already baked
// into GEMM2's A-scale upstream.
void reduce_scatter_unweighted(
    torch::Tensor const& gemm2_out,
    torch::Tensor const& sorted_tids,
    torch::Tensor&       out,
    torch::Tensor&       token_counts,
    torch::Tensor&       token_offsets,
    torch::Tensor&       token_perm,
    int T)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int M = gemm2_out.size(0);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 2 == 0);
  TORCH_CHECK(sorted_tids.size(0) == M);
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);
  TORCH_CHECK(token_counts.numel() >= T);
  TORCH_CHECK(token_offsets.numel() >= T + 1);
  TORCH_CHECK(token_perm.numel() >= M);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  token_bucket_count_kernel<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), M, T, token_counts.data_ptr<int>());

  token_bucket_scan_kernel<<<1, 1024, (1024 / 32) * sizeof(int), stream>>>(
      token_counts.data_ptr<int>(), token_offsets.data_ptr<int>(), T);

  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  token_bucket_place_kernel<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), token_offsets.data_ptr<int>(),
      M, T, token_counts.data_ptr<int>(), token_perm.data_ptr<int>());

  reduce_scatter_unweighted_kernel<<<T, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      token_offsets.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2);
}

void reduce_scatter_prebucketed(
    torch::Tensor const& gemm2_out,     // [M, N2] bf16
    torch::Tensor const& weights,       // [M] fp32
    torch::Tensor const& token_offsets, // [T+1] int32
    torch::Tensor const& token_perm,    // [M] int32
    torch::Tensor&       out,           // [T, N2] bf16
    int T)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int M = gemm2_out.size(0);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 2 == 0);
  TORCH_CHECK(weights.size(0) == M);
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);
  TORCH_CHECK(token_offsets.numel() >= T + 1 && token_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(token_perm.numel() >= M && token_perm.scalar_type() == torch::kInt32);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  reduce_scatter_kernel<<<T, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      weights.data_ptr<float>(),
      token_offsets.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2);
}

void token_bucket_scan_and_place(
    torch::Tensor const& sorted_tids,     // [M] int32
    torch::Tensor const& token_counts,    // [T] int32 (already populated)
    torch::Tensor&       token_offsets,   // [T+1] int32
    torch::Tensor&       token_perm,      // [M] int32
    int T)
{
  TORCH_CHECK(sorted_tids.is_cuda() && sorted_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(token_counts.is_cuda() && token_counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(token_offsets.is_cuda() && token_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(token_perm.is_cuda() && token_perm.scalar_type() == torch::kInt32);
  int M = sorted_tids.size(0);
  TORCH_CHECK(token_counts.numel() >= T);
  TORCH_CHECK(token_offsets.numel() >= T + 1);
  TORCH_CHECK(token_perm.numel() >= M);

  auto stream = at::cuda::getCurrentCUDAStream(sorted_tids.get_device()).stream();
  token_bucket_scan_kernel<<<1, 1024, (1024 / 32) * sizeof(int), stream>>>(
      const_cast<int*>(token_counts.data_ptr<int>()), token_offsets.data_ptr<int>(), T);
  cudaMemsetAsync(const_cast<int*>(token_counts.data_ptr<int>()), 0, T * sizeof(int), stream);
  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  token_bucket_place_kernel<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), token_offsets.data_ptr<int>(),
      M, T, const_cast<int*>(token_counts.data_ptr<int>()), token_perm.data_ptr<int>());
}

// ============================================================================
// Fused dispatch kernel: given topk_idx + assign_w (shape [T, TOP_K]),
// computes per-local-expert counts and produces sorted (sorted_tids,
// sorted_weights) buckets without invoking PyTorch bincount/argsort/gather.
//
// Uses a 2-pass bucket-sort: (1) atomically count local assignments per expert,
// (2) compute exclusive prefix-sum → offsets, (3) each thread finds its slot
// within its expert's bucket via atomicAdd(write_cursor[e], 1).
//
// All arrays are sized to T*TOP_K max; invalid (non-local) entries get
// sorted_weights = 0 so they contribute nothing to the final scatter.
//
// Key trick: one kernel does ALL THREE passes by walking across the whole
// grid. We launch with enough blocks to cover T*TOP_K, use atomics on
// per-expert counts + cursors in shared memory first, then flush via atomicAdd
// to global counts. Second call of the same kernel (with a "place" mode) reads
// counts → offsets → writes sorted tensors.
//
// For implementation simplicity we split into two kernels below.
// ============================================================================

// Pass 1: count local assignments per expert.
__global__ void dispatch_count_kernel(
    const int*   __restrict__ topk_idx,   // [T*TOP_K] flattened int32 (global ids)
    int          NA,                       // = T * TOP_K
    int          local_start,
    int          num_experts,
    int*         counts                    // [num_experts] int32, zero-init
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NA) return;
  int g = topk_idx[i];
  int e = g - local_start;
  if (e >= 0 && e < num_experts) {
    atomicAdd(counts + e, 1);
  }
}

// Pass 2: exclusive prefix-sum on counts (single block, 32 threads).
// Writes offsets_out[0..E-1] = exclusive scan, offsets_out[E] = total.
// ALSO writes counts[e] into problem_sizes_1[e][0] and problem_sizes_2[e][0]
// (the M column of the [E, 3] problem-sizes table) — eliminates 2 aten ops
// in the Python hot path.
__global__ void exclusive_scan_kernel(
    int*  counts_in,       // [E]
    int*  offsets_out,     // [E+1]
    int*  problem_sizes_1, // [E, 3] int32 (M, N, K); or nullptr to skip
    int*  problem_sizes_2, // [E, 3] int32; or nullptr to skip
    int   E)
{
  if (blockIdx.x != 0) return;
  int lane = threadIdx.x;
  int self = (lane < E) ? counts_in[lane] : 0;
  int v = self;
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int n = __shfl_up_sync(0xffffffff, v, off);
    if (lane >= off) v += n;
  }
  int excl = v - self;
  if (lane < E) {
    offsets_out[lane] = excl;
    // Write M column of problem_sizes (row stride = 3, column 0).
    if (problem_sizes_1 != nullptr) problem_sizes_1[lane * 3] = self;
    if (problem_sizes_2 != nullptr) problem_sizes_2[lane * 3] = self;
  }
  if (lane == 31) {
    offsets_out[E] = v;
  }
}

// Pass 3: scatter each valid (t, k) assignment to its sorted slot.
__global__ void dispatch_place_kernel(
    const int*   __restrict__ topk_idx,    // [T*TOP_K] global ids
    const float* __restrict__ assign_w,     // [T*TOP_K] weights
    const int*   __restrict__ offsets,      // [E+1]
    int          NA,
    int          TOP_K,                     // usually 8
    int          local_start,
    int          num_experts,
    int*         cursors,                   // [E] int32, zero-init (working counter per expert)
    int*         sorted_tids,               // [NA] padded with -1
    float*       sorted_weights             // [NA] padded with 0
)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NA) return;
  int t = i / TOP_K;           // token id
  int g = topk_idx[i];
  int e = g - local_start;
  if (e >= 0 && e < num_experts) {
    int slot_in_expert = atomicAdd(cursors + e, 1);
    int slot = offsets[e] + slot_in_expert;
    sorted_tids[slot] = t;
    sorted_weights[slot] = assign_w[i];
  }
  // Invalid slots stay at their pre-zeroed defaults (0 weight, -1 tid).
}

// Pass 3b: dispatch placement + hidden-state gather in one kernel.
// Each warp handles one (token, topk-slot) assignment:
//   - lane 0 reserves the destination slot within its expert bucket
//   - the whole warp copies hidden-state FP8 bytes and K/128 scales into that slot
// This collapses a whole stage boundary in the large-T path.
__global__ void dispatch_place_and_gather_hidden_scales_kernel(
    const int*   __restrict__ topk_idx,      // [T*TOP_K] global ids
    const float* __restrict__ assign_w,      // [T*TOP_K] weights
    const int*   __restrict__ offsets,       // [E+1]
    int          NA,
    int          TOP_K,
    int          local_start,
    int          num_experts,
    int*         cursors,                    // [E] int32, zero-init
    int*         sorted_tids,                // [NA]
    float*       sorted_weights,             // [NA]
    int*         token_counts,               // [T] optional, may be nullptr
    const __nv_fp8_e4m3* __restrict__ hidden_states,    // [T, K1] fp8
    const float*         __restrict__ hs_scale,         // arbitrary strided layout
    int          hs_scale_stride_t,
    int          hs_scale_stride_b,
    __nv_fp8_e4m3* __restrict__ packed_acts,            // [NA, K1]
    float*       __restrict__ packed_act_scales,        // [NA, K1/128]
    int          T,
    int          K1,
    int          K1_blocks)
{
  constexpr int WARPS_PER_BLOCK = 8;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int i = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (i >= NA) return;

  int tok = i / TOP_K;
  int g = topk_idx[i];
  int e = g - local_start;
  if (e < 0 || e >= num_experts || tok < 0 || tok >= T) return;

  int slot = 0;
  if (lane == 0) {
    int slot_in_expert = atomicAdd(cursors + e, 1);
    slot = offsets[e] + slot_in_expert;
    sorted_tids[slot] = tok;
    sorted_weights[slot] = assign_w[i];
    if (token_counts != nullptr) {
      atomicAdd(token_counts + tok, 1);
    }
  }
  slot = __shfl_sync(0xffffffff, slot, 0);
  tok = __shfl_sync(0xffffffff, tok, 0);

  const __nv_fp8_e4m3* src = hidden_states + tok * K1;
  __nv_fp8_e4m3* dst = packed_acts + slot * K1;
  const float* ssrc_base = hs_scale + tok * hs_scale_stride_t;
  float* sdst = packed_act_scales + slot * K1_blocks;

  const uint4* src_v = reinterpret_cast<const uint4*>(src);
  uint4* dst_v = reinterpret_cast<uint4*>(dst);
  int n_v = K1 / 16;
  for (int j = lane; j < n_v; j += 32) {
    dst_v[j] = src_v[j];
  }
  for (int j = lane; j < K1_blocks; j += 32) {
    sdst[j] = ssrc_base[j * hs_scale_stride_b];
  }
}

void fused_dispatch_gather_hidden_scales(
    torch::Tensor const& topk_idx,        // [T, TOP_K] int32
    torch::Tensor const& assign_w,        // [T, TOP_K] float32
    torch::Tensor const& hidden_states,   // [T, K1] fp8
    torch::Tensor const& hs_scale,        // [T, K1/128] or [K1/128, T] fp32
    int                   local_start,
    int                   num_experts,
    torch::Tensor&       counts,          // [E] int32
    torch::Tensor&       sorted_tids,     // [T*TOP_K] int32
    torch::Tensor&       sorted_weights,  // [T*TOP_K] float32
    torch::Tensor&       offsets,         // [E+1] int32
    torch::Tensor&       problem_sizes_1, // [E, 3] int32
    torch::Tensor&       problem_sizes_2, // [E, 3] int32
    torch::Tensor&       token_counts,    // [T] int32 (optional prebucket counts)
    torch::Tensor&       packed_acts,     // [T*TOP_K, K1] fp8
    torch::Tensor&       packed_act_scales) // [T*TOP_K, K1/128] fp32
{
  TORCH_CHECK(topk_idx.is_cuda() && topk_idx.scalar_type() == torch::kInt32);
  TORCH_CHECK(assign_w.is_cuda() && assign_w.scalar_type() == torch::kFloat32);
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.dim() == 2);
  TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(hs_scale.is_cuda() && hs_scale.dim() == 2 && hs_scale.scalar_type() == torch::kFloat32);

  int T = topk_idx.size(0);
  int TOP_K = topk_idx.size(1);
  int NA = T * TOP_K;
  int K1 = hidden_states.size(1);
  int K1_blocks = packed_act_scales.size(1);
  TORCH_CHECK(K1 % 16 == 0, "K1 must be multiple of 16");
  TORCH_CHECK(num_experts <= 32, "only E<=32 supported by single-warp scan");
  TORCH_CHECK(hidden_states.size(0) == T, "hidden_states T mismatch");
  TORCH_CHECK(counts.size(0) == num_experts && counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(sorted_tids.size(0) == NA && sorted_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(sorted_weights.size(0) == NA && sorted_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.size(0) == num_experts + 1 && offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(token_counts.numel() >= T && token_counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(packed_acts.size(0) == NA && packed_acts.size(1) == K1);
  TORCH_CHECK(packed_act_scales.size(0) == NA && packed_act_scales.size(1) == K1_blocks);

  int hs_size_0 = hs_scale.size(0);
  int hs_size_1 = hs_scale.size(1);
  int hs_stride_0 = hs_scale.stride(0);
  int hs_stride_1 = hs_scale.stride(1);
  int stride_t, stride_b;
  // See fused_gather_hidden_scales for rationale: match K-block dim first so
  // the square-shape case T == K1_blocks is resolved toward contest layout.
  if (hs_size_0 == K1_blocks && hs_size_1 == T) {
    stride_t = hs_stride_1;
    stride_b = hs_stride_0;
  } else if (hs_size_0 == T && hs_size_1 == K1_blocks) {
    stride_t = hs_stride_0;
    stride_b = hs_stride_1;
  } else {
    TORCH_CHECK(false,
                "hs_scale shape doesn't match either [T, K/128] or [K/128, T]");
  }

  auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device()).stream();

  cudaMemsetAsync(counts.data_ptr(), 0, counts.numel() * sizeof(int), stream);
  cudaMemsetAsync(sorted_weights.data_ptr(), 0, sorted_weights.numel() * sizeof(float), stream);
  cudaMemsetAsync(sorted_tids.data_ptr(), 0xFF, sorted_tids.numel() * sizeof(int), stream);
  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  int threads = 256;
  int blocks = (NA + threads - 1) / threads;
  dispatch_count_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      NA, local_start, num_experts,
      static_cast<int*>(counts.data_ptr()));

  exclusive_scan_kernel<<<1, 32, 0, stream>>>(
      static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(offsets.data_ptr()),
      static_cast<int*>(problem_sizes_1.data_ptr()),
      static_cast<int*>(problem_sizes_2.data_ptr()),
      num_experts);

  cudaMemsetAsync(counts.data_ptr(), 0, counts.numel() * sizeof(int), stream);
  constexpr int WARPS_PER_BLOCK = 8;
  int gather_blocks = (NA + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  dispatch_place_and_gather_hidden_scales_kernel<<<gather_blocks, WARPS_PER_BLOCK * 32, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const float*>(assign_w.data_ptr()),
      static_cast<const int*>(offsets.data_ptr()),
      NA, TOP_K, local_start, num_experts,
      static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(sorted_tids.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()),
      static_cast<int*>(token_counts.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      hs_scale.data_ptr<float>(),
      stride_t, stride_b,
      reinterpret_cast<__nv_fp8_e4m3*>(packed_acts.data_ptr()),
      packed_act_scales.data_ptr<float>(),
      T, K1, K1_blocks);
}

// Host wrapper — allocates temporary cursor buffer and orchestrates the 3
// kernel launches on the captured stream.
void fused_dispatch(
    torch::Tensor const& topk_idx,     // [T, TOP_K] int32
    torch::Tensor const& assign_w,      // [T, TOP_K] float32
    int                   local_start,
    int                   num_experts,
    torch::Tensor&       counts,        // [E] int32
    torch::Tensor&       sorted_tids,   // [T*TOP_K] int32
    torch::Tensor&       sorted_weights,// [T*TOP_K] float32
    torch::Tensor&       offsets,       // [E+1] int32 (exclusive scan + total)
    torch::Tensor&       problem_sizes_1,  // [E, 3] int32 (M col written by scan)
    torch::Tensor&       problem_sizes_2)  // [E, 3] int32 (M col written by scan)
{
  TORCH_CHECK(topk_idx.is_cuda() && topk_idx.scalar_type() == torch::kInt32);
  TORCH_CHECK(assign_w.is_cuda() && assign_w.scalar_type() == torch::kFloat32);
  int T = topk_idx.size(0);
  int TOP_K = topk_idx.size(1);
  int NA = T * TOP_K;
  TORCH_CHECK(num_experts <= 32, "only E<=32 supported by single-warp scan");
  TORCH_CHECK(counts.size(0) == num_experts && counts.scalar_type() == torch::kInt32);
  TORCH_CHECK(sorted_tids.size(0) == NA && sorted_tids.scalar_type() == torch::kInt32);
  TORCH_CHECK(sorted_weights.size(0) == NA && sorted_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(offsets.size(0) == num_experts + 1 && offsets.scalar_type() == torch::kInt32);

  auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device()).stream();

  // Zero counts + sorted_weights; init sorted_tids to -1 (0xFF bytes) so
  // invalid slots are distinguishable from valid (≥0) tids downstream.
  cudaMemsetAsync(counts.data_ptr(), 0, counts.numel() * sizeof(int), stream);
  cudaMemsetAsync(sorted_weights.data_ptr(), 0, sorted_weights.numel() * sizeof(float), stream);
  cudaMemsetAsync(sorted_tids.data_ptr(), 0xFF, sorted_tids.numel() * sizeof(int), stream);

  // Pass 1.
  int threads = 256;
  int blocks = (NA + threads - 1) / threads;
  dispatch_count_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      NA, local_start, num_experts,
      static_cast<int*>(counts.data_ptr()));

  // Pass 2 — scan + simultaneously write M column of problem_sizes_{1,2}.
  exclusive_scan_kernel<<<1, 32, 0, stream>>>(
      static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(offsets.data_ptr()),
      static_cast<int*>(problem_sizes_1.data_ptr()),
      static_cast<int*>(problem_sizes_2.data_ptr()),
      num_experts);

  // Pass 3: use `counts` as the cursor buffer (re-zero it first).
  cudaMemsetAsync(counts.data_ptr(), 0, counts.numel() * sizeof(int), stream);
  dispatch_place_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const float*>(assign_w.data_ptr()),
      static_cast<const int*>(offsets.data_ptr()),
      NA, TOP_K, local_start, num_experts,
      static_cast<int*>(counts.data_ptr()),  // reused as cursors
      static_cast<int*>(sorted_tids.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()));

  // counts now holds final counts (since place-kernel atomicAdd them up
  // again) so no need to re-count.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_blockwise_grouped_mm_v2", &moe_blockwise_grouped_mm_v2);
  m.def("get_sizes", &get_sizes);
  m.def("get_workspace_size", &get_workspace_size);
  m.def("swiglu_fp8_requant", &swiglu_fp8_requant);
  m.def("swiglu_fp8_requant_weighted", &swiglu_fp8_requant_weighted);
  m.def("weighted_scatter", &weighted_scatter);
  m.def("reduce_scatter", &reduce_scatter);
  m.def("reduce_scatter_prebucketed", &reduce_scatter_prebucketed);
  m.def("reduce_scatter_unweighted_prebucketed", &reduce_scatter_unweighted_prebucketed);
  m.def("reduce_scatter_unweighted", &reduce_scatter_unweighted);
  m.def("token_bucket_scan_and_place", &token_bucket_scan_and_place);
  m.def("fused_route_topk", &fused_route_topk);
  m.def("fused_gather_hidden_scales", &fused_gather_hidden_scales);
  m.def("repack_aligned_expert_layout", &repack_aligned_expert_layout);
  m.def("fused_dispatch", &fused_dispatch);
  m.def("fused_dispatch_gather_hidden_scales", &fused_dispatch_gather_hidden_scales);
  m.def("mxf8_transcode_activations", &mxf8_transcode_activations);
  m.def("mxf8_transcode_weights_impl", &mxf8_transcode_weights_impl);
  m.def("moe_mxf8_grouped_mm", &moe_mxf8_grouped_mm);
  m.def("compute_mxf8_sfa_layout_offsets_host", &compute_mxf8_sfa_layout_offsets_host);
  m.def("compute_mxf8_sfb_layout_offsets_host", &compute_mxf8_sfb_layout_offsets_host);
  m.def("get_mxf8_sizes_stride", &get_mxf8_sizes_stride);
  m.def("get_mxf8_sizes_layout_sfa", &get_mxf8_sizes_layout_sfa);
  m.def("get_mxf8_sizes_layout_sfb", &get_mxf8_sizes_layout_sfb);
  m.def("probe_mxf8_sfa_layout", &probe_mxf8_sfa_layout);
}
'''


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    cutlass_includes = set()
    for f in glob.glob("/opt/conda/**/cutlass/cutlass.h", recursive=True):
        cutlass_includes.add(os.path.dirname(os.path.dirname(f)))
    for f in glob.glob("/opt/conda/**/cutlass/util/packed_stride.hpp", recursive=True):
        cutlass_includes.add(os.path.dirname(os.path.dirname(os.path.dirname(f))))
    if not cutlass_includes:
        raise RuntimeError("CUTLASS headers not found")

    # Prefer a persistent volume path for the JIT build dir so subsequent Modal
    # runs reuse the compiled extension (.so). Fall back to /tmp if /mnt is not
    # writable (e.g., when running outside the bench container).
    candidates = [
        "/mnt/build_cache/fused_moe_cutlass_v5",
        os.path.join(tempfile.gettempdir(), "fused_moe_cutlass_v5"),
    ]
    build_dir = None
    for cand in candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            with open(os.path.join(cand, ".probe"), "w", encoding="utf-8") as f:
                f.write("ok")
            build_dir = cand
            break
        except OSError:
            continue
    assert build_dir is not None, "No writable build dir found"

    def _write_if_changed(path, content):
        existing = ""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = f.read()
            except OSError:
                existing = ""
        if existing != content:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    cutlass_cu = os.path.join(build_dir, "moe_cutlass.cu")
    fused_cu   = os.path.join(build_dir, "moe_fused.cu")
    _write_if_changed(cutlass_cu, _MOE_GEMM_CU)
    _write_if_changed(fused_cu,   _MOE_FUSED_CU)

    # Two separate .cu files → ninja compiles them independently and only
    # rebuilds the one that changed. The CUTLASS-heavy file (stable) compiles
    # once; the fused-helpers file (iterated on) rebuilds in ~10s.
    from torch.utils.cpp_extension import load
    _ext = load(
        name="moe_gemm_v5",
        sources=[cutlass_cu, fused_cu],
        extra_include_paths=sorted(cutlass_includes),
        extra_cuda_cflags=[
            "-O3", "--std=c++17", "-arch=sm_100a",
            "--expt-relaxed-constexpr", "-DNDEBUG",
        ],
        build_directory=build_dir,
        verbose=False,
    )
    return _ext


# ------------------------------ pipeline --------------------------------

def _route_pytorch(routing_logits, routing_bias, rsf, T, local_start, num_experts):
    """PyTorch reference implementation — used by tests and as fallback."""
    s = torch.sigmoid(routing_logits.float())
    s_wb = s + routing_bias.float()
    s_wb_g = s_wb.view(T, N_GROUP, GROUP_SIZE)
    group_top2 = torch.topk(s_wb_g, k=2, dim=2).values
    group_scores = group_top2.sum(dim=2)
    valid_groups = torch.topk(group_scores, k=TOPK_GROUP, dim=1).indices
    group_mask = torch.zeros((T, N_GROUP), device=s.device, dtype=torch.bool)
    group_mask.scatter_(1, valid_groups, True)
    valid_mask = group_mask.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(T, E_GLOBAL)
    filtered = s_wb.masked_fill(~valid_mask, float("-inf"))
    _, topk_idx = torch.topk(filtered, k=TOP_K, dim=1)
    topk_s = s.gather(1, topk_idx.long())
    sum_s = topk_s.sum(dim=1, keepdim=True)
    assign_w = topk_s * (rsf / (sum_s + 1e-20))
    return topk_idx.int(), assign_w


_route = None  # set after _route_fused is defined


def _route_fused(routing_logits, routing_bias, rsf, T, local_start, num_experts):
    """Single-kernel fused routing (DeepSeek-V3 topk8+group4). 6-32x faster
    than the PyTorch chain. Requires bf16 logits/bias (contest format)."""
    ext = _get_ext()
    logits = routing_logits if routing_logits.dtype == torch.bfloat16 \
        else routing_logits.to(torch.bfloat16)
    bias = routing_bias if routing_bias.dtype == torch.bfloat16 \
        else routing_bias.to(torch.bfloat16)
    topk_idx = torch.empty(T, TOP_K, device=logits.device, dtype=torch.int32)
    assign_w = torch.empty(T, TOP_K, device=logits.device, dtype=torch.float32)
    ext.fused_route_topk(logits, bias, topk_idx, assign_w, float(rsf))
    return topk_idx, assign_w


# Switch _route to the fused implementation.
_route = _route_fused


def _dispatch_graph_safe_pytorch(topk_idx, assign_w, T, local_start, num_experts):
    """Pure PyTorch graph-safe dispatch — used by fallback / reference."""
    flat_idx = topk_idx.reshape(-1)
    flat_w = assign_w.reshape(-1)
    flat_tok = torch.arange(T, device=flat_idx.device, dtype=torch.int32)\
        .unsqueeze(1).expand(-1, TOP_K).reshape(-1)

    valid = (flat_idx >= local_start) & (flat_idx < local_start + num_experts)
    masked_idx = torch.where(valid, flat_idx - local_start, torch.full_like(flat_idx, num_experts))
    masked_w = torch.where(valid, flat_w, torch.zeros_like(flat_w))

    perm = masked_idx.argsort(stable=True)
    sorted_tids = flat_tok[perm]
    sorted_weights = masked_w[perm]

    counts_full = torch.zeros(num_experts + 1, device=flat_idx.device, dtype=torch.int32)
    ones = torch.ones_like(masked_idx, dtype=torch.int32)
    counts_full.scatter_add_(0, masked_idx.long(), ones)
    counts = counts_full[:num_experts].contiguous()
    return counts, sorted_tids, sorted_weights


def _dispatch_graph_safe(topk_idx, assign_w, T, local_start, num_experts, bufs):
    """Graph-capture-safe dispatch using the single CUDA fused_dispatch kernel.
    Replaces argsort + scatter_add + where chain (~40-60μs on big-T) with 3
    tiny kernels (~10μs total). Also populates problem_sizes_{1,2}[:, 0].
    """
    ext = _get_ext()
    counts = bufs["counts_buf"]
    offsets = bufs["offsets_buf"]
    sorted_tids = bufs["sorted_tids_buf"]
    sorted_weights = bufs["sorted_weights_buf"]
    ext.fused_dispatch(
        topk_idx.contiguous(), assign_w.contiguous(),
        int(local_start), int(num_experts),
        counts, sorted_tids, sorted_weights, offsets,
        bufs["problem_sizes_1"], bufs["problem_sizes_2"])
    return counts, sorted_tids, sorted_weights


def _dispatch_dynamic(topk_idx, assign_w, T, local_start, num_experts, bufs):
    """Large-T dispatch: single fused CUDA kernel (bincount + sort + gather)
    with compact output. Entries are written contiguously from slot 0, so
    [0:total_valid] extracts the valid ones for downstream compact paths.
    Also writes M-col of problem_sizes_{1,2} inside the scan kernel."""
    ext = _get_ext()
    counts = bufs["counts_buf"]
    offsets = bufs["offsets_buf"]
    sorted_tids = bufs["sorted_tids_buf"]
    sorted_weights = bufs["sorted_weights_buf"]
    ext.fused_dispatch(
        topk_idx.contiguous(), assign_w.contiguous(),
        int(local_start), int(num_experts),
        counts, sorted_tids, sorted_weights, offsets,
        bufs["problem_sizes_1"], bufs["problem_sizes_2"])
    total_valid = int(offsets[num_experts].item())
    return counts, sorted_tids[:total_valid], sorted_weights[:total_valid]


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _maybe_pad_expert_blocks(sorted_tids, sorted_weights, counts, bufs, pad_multiple):
    """Pad each expert's local M with zero-weight dummy rows.

    This is a large-T experiment for B200 FP8 alignment cliffs. We keep the
    dummy token id in-bounds (`0`) and force its combine weight to zero, so the
    padded rows do not affect the final output.
    """
    if pad_multiple <= 1:
        return sorted_tids, sorted_weights, bufs["expert_offsets"], int(sorted_tids.shape[0])

    counts_cpu = counts.cpu().tolist()
    padded_counts = [_round_up(c, pad_multiple) if c > 0 else 0 for c in counts_cpu]
    total_valid = int(sorted_tids.shape[0])
    total_padded = sum(padded_counts)
    if total_padded == total_valid:
        return sorted_tids, sorted_weights, bufs["expert_offsets"], total_valid
    if total_padded > bufs["sorted_tids_buf"].numel():
        return sorted_tids, sorted_weights, bufs["expert_offsets"], total_valid

    src_tids = sorted_tids.clone()
    src_weights = sorted_weights.clone()
    dst_tids = bufs["sorted_tids_buf"][:total_padded]
    dst_weights = bufs["sorted_weights_buf"][:total_padded]

    src_off = 0
    dst_off = 0
    offsets = []
    for count, padded in zip(counts_cpu, padded_counts):
        offsets.append(dst_off)
        if count > 0:
            dst_tids[dst_off : dst_off + count].copy_(src_tids[src_off : src_off + count])
            dst_weights[dst_off : dst_off + count].copy_(src_weights[src_off : src_off + count])
            if padded > count:
                dst_tids[dst_off + count : dst_off + padded].fill_(0)
                dst_weights[dst_off + count : dst_off + padded].zero_()
        src_off += count
        dst_off += padded

    padded_counts_t = torch.tensor(padded_counts, device=counts.device, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, device=counts.device, dtype=torch.int32)
    bufs["counts_buf"].copy_(padded_counts_t)
    bufs["expert_offsets"].copy_(offsets_t)
    bufs["offsets_buf"][len(offsets)] = total_padded
    bufs["problem_sizes_1"][:, 0].copy_(padded_counts_t)
    bufs["problem_sizes_2"][:, 0].copy_(padded_counts_t)
    return dst_tids, dst_weights, bufs["expert_offsets"], total_padded


def _maybe_build_bucket_problem_sizes(counts, bufs, N1, K1, N2, K2):
    """Build zero-masked grouped-GEMM problem lists for expert M buckets.

    This mirrors the SGLang pattern of launching multiple grouped GEMMs over the
    same expert layout but with non-target groups set to M=0.
    """
    if not os.environ.get("EXPERT_M_BUCKETING"):
        return None

    small_max = int(os.environ.get("EXPERT_M_SMALL_MAX", "128"))
    mid_max = int(os.environ.get("EXPERT_M_MID_MAX", "384"))
    counts_cpu = counts.cpu().tolist()
    ne = len(counts_cpu)
    bucket_modes = ["S", "M", "L"]
    ps1_cpu = [[[0, N1, K1] for _ in range(ne)] for _ in range(3)]
    ps2_cpu = [[[0, N2, K2] for _ in range(ne)] for _ in range(3)]
    active = [0, 0, 0]

    for e, m in enumerate(counts_cpu):
        if m <= 0:
            continue
        if m <= small_max:
            idx = 0
        elif m <= mid_max:
            idx = 1
        else:
            idx = 2
        ps1_cpu[idx][e][0] = m
        ps2_cpu[idx][e][0] = m
        active[idx] += 1

    launches = []
    device = counts.device
    for idx, mode in enumerate(bucket_modes):
        if active[idx] == 0:
            continue
        ps1 = bufs["problem_sizes_1_bucketed"][idx]
        ps2 = bufs["problem_sizes_2_bucketed"][idx]
        ps1.copy_(torch.tensor(ps1_cpu[idx], device=device, dtype=torch.int32))
        ps2.copy_(torch.tensor(ps2_cpu[idx], device=device, dtype=torch.int32))
        launches.append((mode, ps1, ps2, active[idx]))
    return launches


def _build_expert_segments(counts):
    """Materialize contiguous expert segments from the dispatch counts.

    This is intentionally host-side: we use it only for alternative execution
    models where each expert becomes its own GEMM launch.
    """
    counts_cpu = counts.cpu().tolist()
    segments = []
    start = 0
    for expert_idx, m in enumerate(counts_cpu):
        end = start + m
        if m > 0:
            segments.append((expert_idx, start, end))
        start = end
    return segments


def _run_segmented_expert_grouped_mm(
    output,
    activations,
    weights,
    scales_a,
    scales_b,
    segments,
    N,
    K,
    bufs,
    ext,
):
    """Launch one E=1 grouped GEMM per expert segment.

    This is a much more invasive alternative to the monolithic ragged grouped
    launch: each expert gets its own auto-selected CUTLASS regime based on its
    local M instead of inheriting a schedule from the batch-wide total_valid.
    """
    if not segments:
        return

    single_offsets = bufs["expert_offsets"][:1]
    single_problem_sizes = bufs["problem_sizes_1"][:1]
    single_problem_sizes_t = bufs["problem_sizes_transpose"][:1]
    stride_sz = bufs["stride_sz"]
    sfa_sz = bufs["sfa_sz"]
    sfb_sz = bufs["sfb_sz"]

    single_offsets.zero_()
    single_problem_sizes[0, 1] = N
    single_problem_sizes[0, 2] = K

    for expert_idx, start, end in segments:
        single_problem_sizes[0, 0] = end - start
        ext.moe_blockwise_grouped_mm_v2(
            output[start:end],
            activations[start:end],
            weights[expert_idx : expert_idx + 1],
            scales_a[start:end],
            scales_b[expert_idx : expert_idx + 1],
            single_offsets,
            single_problem_sizes,
            single_problem_sizes_t,
            bufs["a_ptrs"][:1],
            bufs["b_ptrs"][:1],
            bufs["out_ptrs"][:1],
            bufs["a_scales_ptrs"][:1],
            bufs["b_scales_ptrs"][:1],
            bufs["stride_a"][:stride_sz],
            bufs["stride_b"][:stride_sz],
            bufs["stride_c"][:stride_sz],
            bufs["layout_sfa"][:sfa_sz],
            bufs["layout_sfb"][:sfb_sz],
            bufs["workspace"],
        )


def _build_expert_chunks(counts, max_nonempty_experts):
    """Group the expert-sorted token stream into larger contiguous chunks."""
    counts_cpu = counts.cpu().tolist()
    chunks = []
    chunk_start_e = None
    chunk_start_tok = 0
    chunk_last_nonempty_e = None
    nonempty = 0
    tok = 0

    for expert_idx, m in enumerate(counts_cpu):
        next_tok = tok + m
        if m > 0:
            if chunk_start_e is None:
                chunk_start_e = expert_idx
                chunk_start_tok = tok
            chunk_last_nonempty_e = expert_idx
            nonempty += 1
            if nonempty >= max_nonempty_experts:
                chunks.append((chunk_start_e, chunk_last_nonempty_e + 1, chunk_start_tok, next_tok))
                chunk_start_e = None
                chunk_last_nonempty_e = None
                nonempty = 0
        tok = next_tok

    if chunk_start_e is not None and chunk_last_nonempty_e is not None:
        chunks.append((chunk_start_e, chunk_last_nonempty_e + 1, chunk_start_tok, tok))

    return chunks


def _run_streaming_chunked_pipeline(
    packed_acts,
    packed_act_scales,
    sorted_tids,
    sorted_weights,
    expert_offsets,
    counts,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    N1,
    K1,
    N2,
    K2,
    T,
    bufs,
    ext,
    chunk_experts,
):
    """Process the large-T path as chunked expert streams.

    Each chunk runs GEMM1 -> SwiGLU -> GEMM2 -> scatter before moving on,
    reusing the same scratch prefix and avoiding full-batch intermediate writes.
    """
    chunks = _build_expert_chunks(counts, chunk_experts)
    chunk_offsets_buf = bufs["chunk_offsets_buf"]

    bufs["out_bf16"].zero_()

    for e_begin, e_end, tok_begin, tok_end in chunks:
        chunk_tokens = tok_end - tok_begin
        local_ne = e_end - e_begin
        if chunk_tokens <= 0 or local_ne <= 0:
            continue

        chunk_offsets = chunk_offsets_buf[:local_ne]
        chunk_offsets.copy_(expert_offsets[e_begin:e_end])
        chunk_offsets.sub_(tok_begin)

        chunk_problem_sizes_1 = bufs["problem_sizes_1"][e_begin:e_end]
        chunk_problem_sizes_2 = bufs["problem_sizes_2"][e_begin:e_end]
        chunk_problem_sizes_t = bufs["problem_sizes_transpose"][:local_ne]

        chunk_packed_acts = packed_acts[tok_begin:tok_end]
        chunk_packed_act_scales = packed_act_scales[tok_begin:tok_end]
        chunk_gemm1_out = bufs["gemm1_out"][:chunk_tokens]

        ext.moe_blockwise_grouped_mm_v2(
            chunk_gemm1_out,
            chunk_packed_acts,
            gemm1_weights[e_begin:e_end],
            chunk_packed_act_scales,
            gemm1_weights_scale[e_begin:e_end],
            chunk_offsets,
            chunk_problem_sizes_1,
            chunk_problem_sizes_t,
            bufs["a_ptrs"][:local_ne],
            bufs["b_ptrs"][:local_ne],
            bufs["out_ptrs"][:local_ne],
            bufs["a_scales_ptrs"][:local_ne],
            bufs["b_scales_ptrs"][:local_ne],
            bufs["stride_a"][: local_ne * bufs["stride_sz"]],
            bufs["stride_b"][: local_ne * bufs["stride_sz"]],
            bufs["stride_c"][: local_ne * bufs["stride_sz"]],
            bufs["layout_sfa"][: local_ne * bufs["sfa_sz"]],
            bufs["layout_sfb"][: local_ne * bufs["sfb_sz"]],
            bufs["workspace"],
        )

        chunk_act_q = bufs["act_q"][:chunk_tokens]
        chunk_row_scales = bufs["row_scales"][:chunk_tokens]
        chunk_act_scale_for_gemm2 = bufs["act_scale_for_gemm2"][:chunk_tokens]
        ext.swiglu_fp8_requant(
            chunk_gemm1_out, chunk_act_q, chunk_row_scales, chunk_act_scale_for_gemm2
        )

        chunk_gemm2_out = bufs["gemm2_out"][:chunk_tokens]
        ext.moe_blockwise_grouped_mm_v2(
            chunk_gemm2_out,
            chunk_act_q,
            gemm2_weights[e_begin:e_end],
            chunk_act_scale_for_gemm2,
            gemm2_weights_scale[e_begin:e_end],
            chunk_offsets,
            chunk_problem_sizes_2,
            chunk_problem_sizes_t,
            bufs["a_ptrs"][:local_ne],
            bufs["b_ptrs"][:local_ne],
            bufs["out_ptrs"][:local_ne],
            bufs["a_scales_ptrs"][:local_ne],
            bufs["b_scales_ptrs"][:local_ne],
            bufs["stride_a"][: local_ne * bufs["stride_sz"]],
            bufs["stride_b"][: local_ne * bufs["stride_sz"]],
            bufs["stride_c"][: local_ne * bufs["stride_sz"]],
            bufs["layout_sfa"][: local_ne * bufs["sfa_sz"]],
            bufs["layout_sfb"][: local_ne * bufs["sfb_sz"]],
            bufs["workspace"],
        )

        ext.weighted_scatter(
            chunk_gemm2_out,
            sorted_weights[tok_begin:tok_end],
            sorted_tids[tok_begin:tok_end],
            bufs["out_bf16"],
            T,
        )

    return bufs["out_bf16"]


# Per-device workspace cache, keyed on (ne, T, dims...)
_workspace_cache = {}

# CUDA-graph cache, keyed on input tensor data_ptrs. Once warm, each call is
# just `g.replay()` + a handful of host ops. No Python overhead visible to
# CUPTI timing between kernel launches.
_graph_cache = {}


def _get_workspace(device, ne, T, N1, K1, N2, K2):
    """Pre-allocate all intermediates once per workload (same addresses forever)."""
    total_tokens = T * TOP_K
    key = (device, ne, T, N1, K1, N2, K2)
    if key not in _workspace_cache:
        ext = _get_ext()
        stride_sz, sfa_sz, sfb_sz = ext.get_sizes()
        bufs = dict(
            stride_sz=stride_sz,
            sfa_sz=sfa_sz,
            sfb_sz=sfb_sz,
            a_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            b_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            out_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            a_scales_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            b_scales_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            stride_a=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            stride_b=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            stride_c=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            layout_sfa=torch.empty(ne * sfa_sz, device=device, dtype=torch.uint8),
            layout_sfb=torch.empty(ne * sfb_sz, device=device, dtype=torch.uint8),
            problem_sizes_1=torch.empty(ne, 3, device=device, dtype=torch.int32),
            problem_sizes_2=torch.empty(ne, 3, device=device, dtype=torch.int32),
            problem_sizes_1_bucketed=torch.empty(3, ne, 3, device=device, dtype=torch.int32),
            problem_sizes_2_bucketed=torch.empty(3, ne, 3, device=device, dtype=torch.int32),
            problem_sizes_transpose=torch.empty(ne, 3, device=device, dtype=torch.int32),
            workspace=torch.empty(
                ext.get_workspace_size(total_tokens, ne, 0, 0, False),
                device=device, dtype=torch.uint8),
            packed_acts=torch.empty(total_tokens, K1, device=device, dtype=torch.float8_e4m3fn),
            packed_act_scales=torch.empty(total_tokens, K1 // 128, device=device, dtype=torch.float32),
            gemm1_out=torch.empty(total_tokens, N1, device=device, dtype=torch.bfloat16),
            act_q=torch.empty(total_tokens, N1 // 2, device=device, dtype=torch.float8_e4m3fn),
            row_scales=torch.empty(total_tokens, device=device, dtype=torch.float32),
            act_scale_for_gemm2=torch.empty(
                total_tokens, K2 // 128, device=device, dtype=torch.float32),
            gemm2_out=torch.empty(total_tokens, N2, device=device, dtype=torch.bfloat16),
            out_bf16=torch.empty(T, N2, device=device, dtype=torch.bfloat16),
            # Buffers for fused_dispatch (sorted_tids/weights are T*TOP_K long)
            counts_buf=torch.empty(ne, device=device, dtype=torch.int32),
            offsets_buf=torch.empty(ne + 1, device=device, dtype=torch.int32),
            sorted_tids_buf=torch.empty(total_tokens, device=device, dtype=torch.int32),
            sorted_weights_buf=torch.empty(total_tokens, device=device, dtype=torch.float32),
            # Buffers for reduce_scatter (per-output-token bucket map)
            token_counts_buf=torch.empty(T, device=device, dtype=torch.int32),
            token_offsets_buf=torch.empty(T + 1, device=device, dtype=torch.int32),
            token_perm_buf=torch.empty(total_tokens, device=device, dtype=torch.int32),
            chunk_offsets_buf=torch.empty(ne, device=device, dtype=torch.int32),
        )
        # expert_offsets is a view into offsets_buf[:ne] — fused_dispatch
        # writes the exclusive scan there directly, so no extra cumsum needed.
        bufs["expert_offsets"] = bufs["offsets_buf"][:ne]
        # problem_sizes_1/2 N,K columns are fixed — fill once here.
        bufs["problem_sizes_1"][:, 1] = N1
        bufs["problem_sizes_1"][:, 2] = K1
        bufs["problem_sizes_2"][:, 1] = N2
        bufs["problem_sizes_2"][:, 2] = K2
        _workspace_cache[key] = bufs
    return _workspace_cache[key]


def _run_pipeline_graph_safe(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks,
    bufs, ext,
):
    """Fixed-shape pipeline (total=T*TOP_K) usable inside a CUDA graph.
    All writes go into pre-allocated buffers in `bufs`. No .item() syncs."""
    topk_idx, assign_w = _route(routing_logits, routing_bias, rsf, T, ls, ne)
    counts, sorted_tids, sorted_weights = _dispatch_graph_safe(topk_idx, assign_w, T, ls, ne, bufs)
    # fused_dispatch writes the exclusive-scan directly into offsets_buf,
    # so bufs["expert_offsets"] (aliased to offsets_buf[:ne] in _get_workspace)
    # is already populated; no extra cumsum needed. Similarly problem_sizes
    # M-column is written directly by the fused_dispatch scan kernel.

    # Pass hs_scale through as-is. The C++ gather kernel detects [T, K/128] vs
    # [K/128, T] via strides, so we don't transpose here (doing so inside the
    # CUDA-graph capture context would invalidate replay).
    hs_scale = hidden_states_scale

    # Single fused kernel gathers both hidden_states and its per-K/128 scales.
    # Replaces two `aten::index` ops + two `copy_` ops.
    ext.fused_gather_hidden_scales(
        hidden_states, hs_scale, sorted_tids,
        bufs["packed_acts"], bufs["packed_act_scales"])

    ext.moe_blockwise_grouped_mm_v2(
        bufs["gemm1_out"],
        bufs["packed_acts"], gemm1_weights, bufs["packed_act_scales"], gemm1_weights_scale,
        bufs["expert_offsets"], bufs["problem_sizes_1"],
        bufs["problem_sizes_transpose"],
        bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
        bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
        bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
        bufs["layout_sfa"], bufs["layout_sfb"],
        bufs["workspace"],
    )

    # Fused SwiGLU + per-row FP8 requant in ONE kernel. Replaces ~12 PyTorch ops.
    ext.swiglu_fp8_requant(
        bufs["gemm1_out"], bufs["act_q"],
        bufs["row_scales"], bufs["act_scale_for_gemm2"])

    ext.moe_blockwise_grouped_mm_v2(
        bufs["gemm2_out"],
        bufs["act_q"], gemm2_weights, bufs["act_scale_for_gemm2"], gemm2_weights_scale,
        bufs["expert_offsets"], bufs["problem_sizes_2"],
        bufs["problem_sizes_transpose"],
        bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
        bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
        bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
        bufs["layout_sfa"], bufs["layout_sfb"],
        bufs["workspace"],
    )

    # For small-T graph path, the 4-kernel reduce_scatter pattern adds more
    # launch overhead than it saves — use atomic weighted_scatter instead.
    bufs["out_bf16"].zero_()
    ext.weighted_scatter(
        bufs["gemm2_out"], sorted_weights, sorted_tids, bufs["out_bf16"], T)


def _run_pipeline_dynamic(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf,
    bufs, ext, device,
):
    """Dynamic-shape pipeline used for large T (graph-capture not worth it).
    Only gathers/computes on num_local_valid rows (≈ T) vs T*TOP_K for graph
    path — key for efficiency since >80% of top-k assignments are non-local.
    All intermediate buffers come from the cached workspace (sized for T*TOP_K
    max) so there are no per-call tensor allocations on the hot path."""
    topk_idx, assign_w = _route(routing_logits, routing_bias, rsf, T, ls, ne)
    # Benchmarked: fused dispatch-gather is slightly slower at large T due to
    # higher register pressure in the merged kernel. Keep off by default.
    use_fused_dispatch_gather = bool(int(os.environ.get("FUSED_DISPATCH_GATHER", "0")))
    # Pass hs_scale through as-is: the C++ gather kernel detects [T, K/128] vs
    # [K/128, T] via strides (K-block-dim match prioritized), so no Python-side
    # transpose is needed. Keeping the transpose here would break CUDA-graph
    # capture in the sibling graph-safe path.
    hs_scale = hidden_states_scale

    if use_fused_dispatch_gather:
        ext.fused_dispatch_gather_hidden_scales(
            topk_idx.contiguous(),
            assign_w.contiguous(),
            hidden_states,
            hs_scale,
            int(ls),
            int(ne),
            bufs["counts_buf"],
            bufs["sorted_tids_buf"],
            bufs["sorted_weights_buf"],
            bufs["offsets_buf"],
            bufs["problem_sizes_1"],
            bufs["problem_sizes_2"],
            bufs["token_counts_buf"],
            bufs["packed_acts"],
            bufs["packed_act_scales"],
        )
        counts = bufs["counts_buf"]
        total_valid = int(bufs["offsets_buf"][ne].item())
        sorted_tids = bufs["sorted_tids_buf"][:total_valid]
        sorted_weights = bufs["sorted_weights_buf"][:total_valid]
    else:
        counts, sorted_tids, sorted_weights = _dispatch_dynamic(topk_idx, assign_w, T, ls, ne, bufs)
        total_valid = sorted_tids.shape[0]

    pad_multiple = int(os.environ.get("PAD_EXPERT_M_MULTIPLE", "1"))
    if pad_multiple > 1:
        sorted_tids, sorted_weights, expert_offsets, total_valid = _maybe_pad_expert_blocks(
            sorted_tids, sorted_weights, counts, bufs, pad_multiple
        )
        use_fused_dispatch_gather = False
    else:
        # expert_offsets for dynamic path: bufs["offsets_buf"][:ne] already holds
        # the exclusive scan (written by fused_dispatch). Just alias it.
        expert_offsets = bufs["offsets_buf"][:ne]

    stream_chunk_experts = int(os.environ.get("STREAM_CHUNK_EXPERTS", "0"))
    segmented_mode = os.environ.get("SEGMENTED_EXPERT_GEMM", "").strip().lower()
    segmented_gemm1 = segmented_mode in ("1", "gemm1", "both", "true")
    segmented_gemm2 = segmented_mode in ("2", "gemm2", "both", "true")
    expert_segments = (
        _build_expert_segments(counts) if (segmented_gemm1 or segmented_gemm2) else None
    )

    # problem_sizes_{1,2} M column is written by fused_dispatch scan kernel.
    bucket_launches = None
    if not (segmented_gemm1 or segmented_gemm2):
        bucket_launches = _maybe_build_bucket_problem_sizes(counts, bufs, N1, K1, N2, K2)

    # Use pre-allocated workspace buffers, narrowed to total_valid rows.
    packed_acts = bufs["packed_acts"][:total_valid]
    packed_act_scales = bufs["packed_act_scales"][:total_valid]
    if not use_fused_dispatch_gather:
        ext.fused_gather_hidden_scales(
            hidden_states, hs_scale, sorted_tids,
            packed_acts, packed_act_scales)

    if stream_chunk_experts > 0:
        return _run_streaming_chunked_pipeline(
            packed_acts,
            packed_act_scales,
            sorted_tids,
            sorted_weights,
            expert_offsets,
            counts,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            N1,
            K1,
            N2,
            K2,
            T,
            bufs,
            ext,
            stream_chunk_experts,
        )

    gemm1_out = bufs["gemm1_out"][:total_valid]
    prev_cfg_mode = os.environ.get("CUTLASS_CFG_MODE")
    try:
        if segmented_gemm1:
            _run_segmented_expert_grouped_mm(
                gemm1_out,
                packed_acts,
                gemm1_weights,
                packed_act_scales,
                gemm1_weights_scale,
                expert_segments,
                N1,
                K1,
                bufs,
                ext,
            )
        elif bucket_launches:
            for mode, ps1, _, _ in bucket_launches:
                os.environ["CUTLASS_CFG_MODE"] = mode
                ext.moe_blockwise_grouped_mm_v2(
                    gemm1_out,
                    packed_acts, gemm1_weights, packed_act_scales, gemm1_weights_scale,
                    expert_offsets, ps1,
                    bufs["problem_sizes_transpose"],
                    bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
                    bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
                    bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
                    bufs["layout_sfa"], bufs["layout_sfb"],
                    bufs["workspace"],
                )
        else:
            ext.moe_blockwise_grouped_mm_v2(
                gemm1_out,
                packed_acts, gemm1_weights, packed_act_scales, gemm1_weights_scale,
                expert_offsets, bufs["problem_sizes_1"],
                bufs["problem_sizes_transpose"],
                bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
                bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
                bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
                bufs["layout_sfa"], bufs["layout_sfb"],
                bufs["workspace"],
            )

        # Fused SwiGLU + per-row FP8 requant using pre-allocated buffers.
        act_q = bufs["act_q"][:total_valid]
        row_scales = bufs["row_scales"][:total_valid]
        act_scale_for_gemm2 = bufs["act_scale_for_gemm2"][:total_valid]
        # v17: fold routing weight into A scale of GEMM2 -> unweighted reduce_scatter.
        use_weighted_fold = bool(int(os.environ.get("V17_WEIGHTED_FOLD", "1")))
        if use_weighted_fold:
            ext.swiglu_fp8_requant_weighted(
                gemm1_out, sorted_weights, act_q, row_scales, act_scale_for_gemm2)
        else:
            ext.swiglu_fp8_requant(gemm1_out, act_q, row_scales, act_scale_for_gemm2)

        gemm2_out = bufs["gemm2_out"][:total_valid]
        if segmented_gemm2:
            _run_segmented_expert_grouped_mm(
                gemm2_out,
                act_q,
                gemm2_weights,
                act_scale_for_gemm2,
                gemm2_weights_scale,
                expert_segments,
                N2,
                K2,
                bufs,
                ext,
            )
        elif bucket_launches:
            for mode, _, ps2, _ in bucket_launches:
                os.environ["CUTLASS_CFG_MODE"] = mode
                ext.moe_blockwise_grouped_mm_v2(
                    gemm2_out,
                    act_q, gemm2_weights, act_scale_for_gemm2, gemm2_weights_scale,
                    expert_offsets, ps2,
                    bufs["problem_sizes_transpose"],
                    bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
                    bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
                    bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
                    bufs["layout_sfa"], bufs["layout_sfb"],
                    bufs["workspace"],
                )
        else:
            ext.moe_blockwise_grouped_mm_v2(
                gemm2_out,
                act_q, gemm2_weights, act_scale_for_gemm2, gemm2_weights_scale,
                expert_offsets, bufs["problem_sizes_2"],
                bufs["problem_sizes_transpose"],
                bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
                bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
                bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
                bufs["layout_sfa"], bufs["layout_sfb"],
                bufs["workspace"],
            )
    finally:
        if prev_cfg_mode is None:
            os.environ.pop("CUTLASS_CFG_MODE", None)
        else:
            os.environ["CUTLASS_CFG_MODE"] = prev_cfg_mode

    # Reduce-scatter: per-output-token summation, single bf16 write. Avoids
    # the 100μs out.zero_() and the atomic-contention on bf162 adds.
    # v17: when weighted fold is on, routing weights were baked into GEMM2's
    # A-scale inside swiglu_fp8_requant_weighted, so gemm2_out already encodes
    # w * v. reduce_scatter must NOT multiply by weights again.
    if use_fused_dispatch_gather:
        ext.token_bucket_scan_and_place(
            sorted_tids,
            bufs["token_counts_buf"],
            bufs["token_offsets_buf"],
            bufs["token_perm_buf"],
            T,
        )
        if use_weighted_fold:
            ext.reduce_scatter_unweighted_prebucketed(
                gemm2_out,
                bufs["token_offsets_buf"],
                bufs["token_perm_buf"],
                bufs["out_bf16"],
                T,
            )
        else:
            ext.reduce_scatter_prebucketed(
                gemm2_out,
                sorted_weights,
                bufs["token_offsets_buf"],
                bufs["token_perm_buf"],
                bufs["out_bf16"],
                T,
            )
    else:
        if use_weighted_fold:
            ext.reduce_scatter_unweighted(
                gemm2_out, sorted_tids, bufs["out_bf16"],
                bufs["token_counts_buf"], bufs["token_offsets_buf"], bufs["token_perm_buf"],
                T)
        else:
            ext.reduce_scatter(
                gemm2_out, sorted_weights, sorted_tids, bufs["out_bf16"],
                bufs["token_counts_buf"], bufs["token_offsets_buf"], bufs["token_perm_buf"],
                T)
    return bufs["out_bf16"]


@torch.no_grad()
def custom_kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    ext = _get_ext()

    T = int(routing_logits.shape[0])
    device = hidden_states.device
    ne = int(gemm1_weights.shape[0])
    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])
    H = N1 // 2
    ls = int(local_expert_offset)
    rsf = float(routed_scaling_factor)

    total_tokens = T * TOP_K
    k2_blocks = K2 // 128
    bufs = _get_workspace(device, ne, T, N1, K1, N2, K2)

    # Below threshold: Python overhead dominates → CUDA graph replay wins.
    # Above threshold: GEMM compute dominates, non-local filtering saves ~8x
    # data movement vs fixed-shape path.
    # Graph-safe path (fixed shape T*TOP_K) wins for small-medium T due to
    # graph-replay overhead elimination. Dynamic path (only num_local_valid
    # tokens, ~T) wins for large T where graph-safe's extra 8x gather+scatter
    # work on T*TOP_K dominates. Verified empirical crossover ~T=2048.
    use_graph = (T <= 2048) and not os.environ.get("DISABLE_CUDA_GRAPH")

    if not use_graph:
        return _run_pipeline_dynamic(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            T, ne, N1, K1, N2, K2, H, ls, rsf, bufs, ext, device)

    pkey = (
        T, device,
        int(routing_logits.data_ptr()), int(routing_bias.data_ptr()),
        int(hidden_states.data_ptr()), int(hidden_states_scale.data_ptr()),
        int(gemm1_weights.data_ptr()), int(gemm1_weights_scale.data_ptr()),
        int(gemm2_weights.data_ptr()), int(gemm2_weights_scale.data_ptr()),
    )

    if pkey not in _graph_cache:
        for _ in range(2):
            _run_pipeline_graph_safe(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks, bufs, ext)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_pipeline_graph_safe(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks, bufs, ext)
        _graph_cache[pkey] = g

    _graph_cache[pkey].replay()
    return bufs["out_bf16"]


kernel = custom_kernel


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    return custom_kernel(routing_logits, routing_bias, hidden_states,
                         hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                         gemm2_weights, gemm2_weights_scale,
                         local_expert_offset, routed_scaling_factor)
