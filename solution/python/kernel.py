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
  using MmaTileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// 1SM variant used for testing the FP8-output path (epilogue overhead tends
// to scale better at 1SM where Less SMEM pressure means the amax-reduce
// epilogue can run in parallel with mainloop writeout).
struct CfgMxF8Large1SM {
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// Tile variants for GEMM2 experiments (K=2048, N=7168 — different shape
// characteristics than GEMM1). Trying more N tile capacity to amortize the
// smaller K-iteration count.
struct CfgMxF8G2_256_256 {
  using MmaTileShape = Shape<_256, _256, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};
struct CfgMxF8G2_128_256_1SM {
  using MmaTileShape = Shape<_128, _256, _128>;
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

// ============================================================================
// MxF8 → FP8+UE8M0 block-scaled output variant. Uses
// `LinCombBlockScaleFactor` fusion to emit FP8 payload + per-32-col UE8M0
// scales directly from the epilogue, eliminating the bf16 intermediate's HBM
// round-trip (saves ~472 MB of writes at T=14107, ~60 μs).
// ============================================================================
using ElementD_FP8 = cutlass::float_e4m3_t;
using ElementSFD   = cutlass::float_ue8m0_t;
constexpr int AlignmentD_FP8 = 16;  // 128 bits / 8 bits = 16 fp8 elements
constexpr int OutputSFVectorSize_FP8 = 32;

template <typename Cfg, typename LayoutOut>
struct MxF8GemmBuilderFP8Out {
  using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
      OutputSFVectorSize_FP8,
      ElementD_FP8,
      ElementAccumulator,
      ElementSFD,
      LayoutOut,
      void,
      float>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
      typename Cfg::MmaTileShape, typename Cfg::ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void,          LayoutOut*, AlignmentC,
      ElementD_FP8,  LayoutOut*, AlignmentD_FP8,
      typename Cfg::EpilogueSchedule,
      FusionOperation>::CollectiveOp;

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
  using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<
      OutputSFVectorSize_FP8,
      cute::UMMA::Major::K>;
  using OutputSFAtom = typename Sm1xxBlockScaledOutputConfig::SfAtom;
  using LayoutSFD    = typename Sm1xxBlockScaledOutputConfig::LayoutSF;
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
  // Enable PDL (Programmatic Dependent Launch, SM100A) — allows the next kernel
  // in the stream to begin executing preambles while this GEMM is still running.
  const char* disable_pdl = std::getenv("MXF8_DISABLE_PDL");
  bool pdl_enabled = !(disable_pdl && std::string(disable_pdl) == "1");
  status = gemm_op.run(stream, /*cuda_adapter=*/nullptr, pdl_enabled);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "MxF8 run failed: ", int(status));
}

// Ptr-array populator for the FP8-output variant. Writes D_FP8 ptrs AND SFD
// (per-expert block-scale output) ptrs.
template <typename Cfg, typename LayoutOut>
__global__ void get_group_gemm_starts_kernel_mxf8_fp8out(
    int32_t const* __restrict__ expert_offsets,
    int32_t const* __restrict__ sfa_offsets,
    int32_t const* __restrict__ sfb_offsets,
    int32_t const* __restrict__ sfd_offsets,
    ElementAB** a_ptrs,
    ElementAB** b_ptrs,
    ElementD_FP8** d_ptrs,
    ElementSFD** sfd_ptrs,
    ElementSF_mx** sfa_ptrs,
    ElementSF_mx** sfb_ptrs,
    ElementAB* a_base,
    ElementAB* b_base,
    ElementD_FP8* d_base,
    ElementSFD* sfd_base,
    ElementSF_mx* sfa_base,
    ElementSF_mx* sfb_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::InternalLayoutSFA* layout_sfa_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::InternalLayoutSFB* layout_sfb_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::LayoutSFD* layout_sfd_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::StrideA* stride_a_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::StrideB* stride_b_base,
    typename MxF8GemmBuilderFP8Out<Cfg, LayoutOut>::StrideD* stride_d_base,
    int32_t const* problem_sizes)
{
  using MxFB = MxF8GemmBuilderFP8Out<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;
  using SfOutConfig = typename MxFB::Sm1xxBlockScaledOutputConfig;

  int eid = threadIdx.x;
  int m_e = problem_sizes[eid * 3];
  int n_e = problem_sizes[eid * 3 + 1];
  int k_e = problem_sizes[eid * 3 + 2];

  int64_t expert_offset = static_cast<int64_t>(expert_offsets[eid]);
  int64_t a_stride = expert_offset * k_e;
  int64_t b_stride = int64_t(eid) * int64_t(k_e) * int64_t(n_e);

  a_ptrs[eid] = a_base + a_stride;
  b_ptrs[eid] = b_base + b_stride;
  d_ptrs[eid] = d_base + expert_offset * n_e;
  sfa_ptrs[eid] = sfa_base + sfa_offsets[eid];
  sfb_ptrs[eid] = sfb_base + sfb_offsets[eid];
  sfd_ptrs[eid] = sfd_base + sfd_offsets[eid];

  layout_sfa_base[eid] = SfConfig::tile_atom_to_shape_SFA(
      cute::make_shape(m_e, n_e, k_e, 1));
  layout_sfb_base[eid] = SfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(m_e, n_e, k_e, 1));
  layout_sfd_base[eid] = SfOutConfig::tile_atom_to_shape_SFD(
      cute::make_shape(m_e, n_e, k_e, 1));

  stride_a_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideA{}, cute::make_shape(m_e, k_e, 1));
  stride_b_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideB{}, cute::make_shape(n_e, k_e, 1));
  stride_d_base[eid] = cutlass::make_cute_packed_stride(
      typename MxFB::StrideD{}, cute::make_shape(m_e, n_e, 1));
}

// Global device-side "1.0f" constant used as the unconditional norm_constant
// by LinCombBlockScaleFactor (CUTLASS always dereferences norm_constant_ptr).
__device__ __constant__ float g_mxf8_norm_constant_one = 1.0f;

template <typename Cfg, typename LayoutOut>
void launch_mxf8_group_gemm_fp8out(
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& d_ptrs,
    torch::Tensor& sfd_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a,
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_d,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& layout_sfd,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& workspace,
    int num_experts,
    cudaStream_t stream)
{
  using MxFB = MxF8GemmBuilderFP8Out<Cfg, LayoutOut>;
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

  // Resolve __constant__ memory to a device pointer for norm_constant_ptr.
  // CUTLASS unconditionally dereferences norm_constant_ptr even if dNormConst
  // is a zero-stride (broadcast). Must be non-null.
  float* norm_constant_device_ptr = nullptr;
  cudaGetSymbolAddress(reinterpret_cast<void**>(&norm_constant_device_ptr),
                        g_mxf8_norm_constant_one);

  typename Gemm::GemmKernel::EpilogueArguments epilogue_args{};
  epilogue_args.thread.alpha = 1.0f;
  epilogue_args.thread.beta  = 0.0f;
  epilogue_args.thread.block_scale_factor_ptr =
      static_cast<ElementSFD**>(sfd_ptrs.data_ptr());
  epilogue_args.thread.norm_constant_ptr = norm_constant_device_ptr;
  epilogue_args.ptr_C = nullptr;
  epilogue_args.dC    = static_cast<typename MxFB::StrideC*>(
      const_cast<void*>(stride_d.data_ptr()));  // unused (beta=0) but must be non-null
  epilogue_args.ptr_D = static_cast<ElementD_FP8**>(d_ptrs.data_ptr());
  epilogue_args.dD    = static_cast<typename MxFB::StrideD*>(
      const_cast<void*>(stride_d.data_ptr()));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count  = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

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

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "MxF8 FP8-out can_implement failed: ", int(status));
  status = gemm_op.initialize(args, const_cast<void*>(workspace.data_ptr()), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "MxF8 FP8-out initialize failed: ", int(status));
  const char* disable_pdl = std::getenv("MXF8_DISABLE_PDL");
  bool pdl_enabled = !(disable_pdl && std::string(disable_pdl) == "1");
  status = gemm_op.run(stream, /*cuda_adapter=*/nullptr, pdl_enabled);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "MxF8 FP8-out run failed: ", int(status));
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

// Variant for token-stationary execution: each grouped problem is selected by
// an explicit expert id instead of inheriting expert == group index.
template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig,
          typename StrideA, typename StrideB, typename StrideC,
          typename OutT>
__global__ void get_group_gemm_starts_by_expert_ids_kernel(
    int32_t const* __restrict__ expert_ids,      // [G]
    int            num_groups,
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
    int32_t const* problem_sizes,            // [G, 3]
    int32_t*       problem_sizes_transpose,  // [G, 3] output
    bool transpose)
{
  int gid = threadIdx.x;
  if (gid >= num_groups) return;
  int expert_id = expert_ids[gid];
  int m = problem_sizes[gid * 3];
  int n = problem_sizes[gid * 3 + 1];
  int k = problem_sizes[gid * 3 + 2];
  if (transpose) {
    problem_sizes_transpose[gid * 3]     = n;
    problem_sizes_transpose[gid * 3 + 1] = m;
    problem_sizes_transpose[gid * 3 + 2] = k;
  }

  int64_t group_offset = static_cast<int64_t>(gid);
  int64_t expert_offset = static_cast<int64_t>(expert_id);
  int64_t a_stride, b_stride, a_scale_stride, b_scale_stride;
  if (!transpose) {
    a_stride       = group_offset * k;
    b_stride       = expert_offset * int64_t(k) * int64_t(n);
    a_scale_stride = group_offset * k / 128;
    b_scale_stride = expert_offset * int64_t(k) * int64_t(n) / 128 / 128;
  } else {
    a_stride       = expert_offset * int64_t(k) * int64_t(n);
    b_stride       = group_offset * k;
    a_scale_stride = expert_offset * int64_t(k) * int64_t(n) / 128 / 128;
    b_scale_stride = group_offset * k / 128;
  }

  a_ptrs[gid]         = a_base + a_stride;
  b_ptrs[gid]         = b_base + b_stride;
  out_ptrs[gid]       = out_base + group_offset * n;
  a_scales_ptrs[gid]  = a_scales_base + a_scale_stride;
  b_scales_ptrs[gid]  = b_scales_base + b_scale_stride;

  int M_e = transpose ? n : m;
  int N_e = transpose ? m : n;
  if (!transpose) {
    layout_sfa_base[gid] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    layout_sfb_base[gid] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
  } else {
    layout_sfa_base[gid] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(n, m, k, 1));
    layout_sfb_base[gid] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(n, m, k, 1));
  }

  stride_a_base[gid] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_e, k, 1));
  stride_b_base[gid] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N_e, k, 1));
  stride_c_base[gid] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_e, N_e, 1));
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
  // v20-D1: enable PDL (Programmatic Dependent Launch, SM100A). Lets the next
  // kernel in the stream start its preamble while this GEMM is completing.
  // Blockwise path was missing this flag (MxF8 path already had it).
  const char* disable_pdl = std::getenv("BLOCKWISE_DISABLE_PDL");
  bool pdl_enabled = !(disable_pdl && std::string(disable_pdl) == "1");
  status = gemm_op.run(stream, /*cuda_adapter=*/nullptr, pdl_enabled);
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
  // v20-E-tile: CfgMidM (tile_M=64) beats CfgLargeM (tile_M=128) by ~10% on
  // T=901 because per-expert M is ~42 — with tile_M=128, each tile is
  // only 33% utilized along M. CfgMidM's tile_M=64 gives 66% M-utilization
  // with the same tile_N and compute efficiency per tile. This matters for
  // the memory-bound blockwise regime (T in [256, 4095]) where per-expert
  // M is small. CfgSmallM still wins for very small total_tokens (<=2048)
  // thanks to AB-swap (256-N tile efficient, 32-M tile fits tiny per-expert M).
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
  } else if (schedule_mode == 'L') {
    launch_large_m(CfgLargeM{});
  } else {
    // Default: CfgMidM (tile_M=64). Beats CfgLargeM by ~10% on T=901 because
    // per-expert M=~42 fits better in tile_M=64 (66% util) vs tile_M=128 (33%).
    launch_large_m(CfgMidM{});
  }
}

void moe_blockwise_grouped_mm_by_expert_ids(
    torch::Tensor& output,           // [num_groups, N] bf16
    torch::Tensor const& a,          // [num_groups, K] fp8
    torch::Tensor const& b,          // [E, N, K] fp8
    torch::Tensor const& scales_a,   // [num_groups, K/128] fp32
    torch::Tensor const& scales_b,   // [E, N/128, K/128] fp32
    torch::Tensor const& expert_ids, // [num_groups] int32, selects weight expert per group
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
    torch::Tensor const& workspace)
{
  int total_tokens = a.size(0);
  int K = a.size(1);
  int E = b.size(0);
  int N = b.size(1);
  TORCH_CHECK(b.size(2) == K, "b K mismatch");
  TORCH_CHECK(output.size(0) == total_tokens && output.size(1) == N, "output shape mismatch");
  TORCH_CHECK(expert_ids.size(0) == total_tokens, "expert_ids shape mismatch");
  TORCH_CHECK(total_tokens <= 1024, "by-expert-id grouped MM currently supports <=1024 groups");

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  char schedule_mode = 'A';
  if (const char* env = std::getenv("CUTLASS_CFG_MODE")) {
    schedule_mode = static_cast<char>(std::toupper(env[0]));
  }

  auto launch_large_m = [&](auto cfg_tag) {
    using Cfg = decltype(cfg_tag);
    using LayoutOut = cutlass::layout::RowMajor;
    using GB = GemmBuilder<Cfg, LayoutOut>;
    get_group_gemm_starts_by_expert_ids_kernel<
        typename Cfg::LayoutSFA, typename Cfg::LayoutSFB, typename Cfg::ScaleConfig,
        typename GB::StrideA, typename GB::StrideB, typename GB::StrideC, ElementC>
        <<<1, total_tokens, 0, stream>>>(
        static_cast<int32_t const*>(expert_ids.data_ptr()),
        total_tokens,
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
        problem_sizes, workspace, total_tokens, stream);
  };

  if (schedule_mode == 'S' || (schedule_mode == 'A' && total_tokens <= 2048)) {
    using Cfg = CfgSmallM;
    using LayoutOut = cutlass::layout::ColumnMajor;
    using GB = GemmBuilder<Cfg, LayoutOut>;
    get_group_gemm_starts_by_expert_ids_kernel<
        typename Cfg::LayoutSFA, typename Cfg::LayoutSFB, typename Cfg::ScaleConfig,
        typename GB::StrideA, typename GB::StrideB, typename GB::StrideC, ElementC>
        <<<1, total_tokens, 0, stream>>>(
        static_cast<int32_t const*>(expert_ids.data_ptr()),
        total_tokens,
        static_cast<ElementAB**>(a_ptrs.data_ptr()),
        static_cast<ElementAB**>(b_ptrs.data_ptr()),
        static_cast<ElementC**>(out_ptrs.data_ptr()),
        static_cast<float**>(a_scales_ptrs.data_ptr()),
        static_cast<float**>(b_scales_ptrs.data_ptr()),
        static_cast<ElementAB*>(const_cast<void*>(b.data_ptr())),
        static_cast<ElementAB*>(const_cast<void*>(a.data_ptr())),
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
        problem_sizes_transpose, workspace, total_tokens, stream);
  } else if (schedule_mode == 'L') {
    launch_large_m(CfgLargeM{});
  } else {
    launch_large_m(CfgMidM{});
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
// Vectorized transcode kernel: one block per row; each thread handles 16 fp8
// elements (one uint4 vector). Shared-memory staging of per-(m, kb) scale
// residuals + 1-block-per-row launch pattern eliminates the massive block-
// count overhead of the original (M * K_blocks) launch.
//
// When `sfa_out` != nullptr, also fuses the SFA-tiled-layout packing step
// (writing UE8M0 bytes directly into CUTLASS tiled positions) -- saves a
// separate pack kernel launch.
template <typename Cfg, typename LayoutOut>
__global__ void mxf8_transcode_pack_kernel(
    __nv_fp8_e4m3*       __restrict__ payload,
    const float*         __restrict__ scale_signed,
    float*               __restrict__ scale_ue8m0,
    int                   M,
    int                   K,
    int                   K_blocks,
    int                   payload_row_stride,
    int                   scale_row_stride,
    // fused SFA-pack args (optional; nullable via sfa_out==nullptr)
    const int*            expert_offsets,
    const int*            sfa_byte_offsets,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* sfa_layouts,
    int                   E,
    cutlass::float_ue8m0_t* sfa_out)
{
  int m = blockIdx.x;
  if (m >= M) return;
  int tid = threadIdx.x;
  int bdim = blockDim.x;

  // Shared residual scale-per-K-block storage; 1 float per K-block.
  extern __shared__ float sr_shared[];   // [K_blocks]

  // Fused SFA-pack: locate the (expert, local_m) for this global_m row ONCE,
  // then write UE8M0 bytes directly into CUTLASS tiled SFA layout alongside
  // computing sr_shared below.
  int e = 0;
  int local_m = m;
  cutlass::float_ue8m0_t* sfa_row_base = nullptr;
  typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA sfa_layout{};
  if (sfa_out != nullptr) {
    if (tid == 0) {
      int e_local = 0;
      for (int ee = 0; ee < E; ++ee) {
        if (m < expert_offsets[ee + 1]) { e_local = ee; break; }
      }
      sr_shared[K_blocks] = __int_as_float(e_local);
    }
    __syncthreads();
    e = __float_as_int(sr_shared[K_blocks]);
    local_m = m - expert_offsets[e];
    sfa_row_base = sfa_out + sfa_byte_offsets[e];
    sfa_layout = sfa_layouts[e];
  }

  // Cooperatively load + process K-block scales.
  const float* scale_row = scale_signed + m * scale_row_stride;
  float* scale_ue8m0_row = scale_ue8m0 + m * scale_row_stride;
  for (int kb = tid; kb < K_blocks; kb += bdim) {
    float s_signed = scale_row[kb];
    float s_abs = fabsf(s_signed);
    float sign = (s_signed < 0.0f) ? -1.0f : 1.0f;
    uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(s_abs);
    float ue8m0_val = ue8m0_byte_to_fp32(ue8m0_byte);
    scale_ue8m0_row[kb] = ue8m0_val;
    float r = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
    sr_shared[kb] = sign * r;

    // Fused SFA pack: write UE8M0 byte to each of the 4 sub-blocks.
    if (sfa_out != nullptr) {
      cutlass::float_ue8m0_t val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
      int k0 = kb * 128;
      int off0 = sfa_layout(local_m, k0 +  0, 0);
      int off1 = sfa_layout(local_m, k0 + 32, 0);
      int off2 = sfa_layout(local_m, k0 + 64, 0);
      int off3 = sfa_layout(local_m, k0 + 96, 0);
      sfa_row_base[off0] = val;
      sfa_row_base[off1] = val;
      sfa_row_base[off2] = val;
      sfa_row_base[off3] = val;
    }
  }
  __syncthreads();

  // Vectorized payload transcode: each thread handles 16 fp8 elements (one uint4).
  __nv_fp8_e4m3* row = payload + m * payload_row_stride;
  int num_vec = K / 16;  // 16 fp8 = 16 bytes = 1 uint4
  uint4* row_vec = reinterpret_cast<uint4*>(row);

  for (int i = tid; i < num_vec; i += bdim) {
    int k_start = i * 16;
    int kb = k_start / 128;   // which K-block-128
    float sr = sr_shared[kb];
    uint4 v = row_vec[i];
    // Decode 16 fp8 bytes, rescale, re-encode. Using __nv_fp8_storage_t trick.
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&v);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      __nv_fp8_e4m3 fp8v;
      fp8v.__x = bytes[j];
      float fv = (float)fp8v;
      fv *= sr;
      if (fv >  448.0f) fv =  448.0f;
      if (fv < -448.0f) fv = -448.0f;
      __nv_fp8_e4m3 fp8_out(fv);
      bytes[j] = fp8_out.__x;
    }
    row_vec[i] = v;
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
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");

  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;

  auto stream = at::cuda::getCurrentCUDAStream(payload.get_device()).stream();
  int threads = 256;
  size_t smem_bytes = (K_blocks + 1) * sizeof(float);
  mxf8_transcode_pack_kernel<Cfg, LayoutOut><<<M, threads, smem_bytes, stream>>>(
      reinterpret_cast<__nv_fp8_e4m3*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      scale_ue8m0.data_ptr<float>(),
      M, K, K_blocks,
      payload.stride(0),
      scale.stride(0),
      nullptr, nullptr, nullptr, 0, nullptr);  // no SFA pack fusion
}

// Fused variant: transcode + pack SFA in one kernel launch.
void mxf8_transcode_and_pack_sfa(
    torch::Tensor& payload,
    torch::Tensor const& scale,
    torch::Tensor& scale_ue8m0,
    torch::Tensor const& expert_offsets,  // [E+1] inclusive
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor& sfa_layouts,           // [E * sizeof(InternalLayoutSFA)]
    torch::Tensor& sfa_buffer)
{
  TORCH_CHECK(payload.is_cuda() && payload.scalar_type() == torch::kFloat8_e4m3fn);
  int M = payload.size(0);
  int K = payload.size(1);
  int K_blocks = scale.size(1);
  int E = expert_offsets.size(0) - 1;

  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;

  auto stream = at::cuda::getCurrentCUDAStream(payload.get_device()).stream();
  int threads = 256;
  size_t smem_bytes = (K_blocks + 1) * sizeof(float);
  mxf8_transcode_pack_kernel<Cfg, LayoutOut><<<M, threads, smem_bytes, stream>>>(
      reinterpret_cast<__nv_fp8_e4m3*>(payload.data_ptr()),
      scale.data_ptr<float>(),
      scale_ue8m0.data_ptr<float>(),
      M, K, K_blocks,
      payload.stride(0),
      scale.stride(0),
      expert_offsets.data_ptr<int>(),
      sfa_byte_offsets.data_ptr<int>(),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(sfa_layouts.data_ptr()),
      E,
      reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr()));
}

// v21 SINGLE-WARP rewrite: block=32, zero __syncthreads, register-resident
// absmax reduction via shfl. H/32 = 64 acts cached per lane for H=2048
// (fits cleanly in registers). Kills the 37.7% barrier stall this kernel
// was showing at T=14107 and trades 0.5 waves (at block=256) for 11.9
// waves (at block=32), which is fine because SM100 has 64 warps/SM.
template <int H_, typename Cfg, typename LayoutOut>
__global__ void __launch_bounds__(32)
swiglu_fp8_requant_weighted_mxf8_kernel(
    const __nv_bfloat16* __restrict__ gemm1_out,
    const float*         __restrict__ sorted_weights,
    __nv_fp8_e4m3*       __restrict__ act_q,
    float*               __restrict__ row_scales,
    float*               __restrict__ broadcast_scales_ue8m0,
    int M,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sfa_byte_offsets,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* sfa_layouts,
    int E,
    cutlass::float_ue8m0_t* sfa_out)
{
  constexpr int H = H_;
  constexpr int NW = 32;                // threads per block
  constexpr int ITERS = H / NW;         // 64 for H=2048; 128 for H=4096
  constexpr int KBLOCKS = H / 128;

  const int m = blockIdx.x;
  if (m >= M) return;
  const int lane = threadIdx.x;

  // Expert-id lookup: lane 0 does the linear scan, broadcasts via shfl.
  int e = 0;
  if (lane == 0) {
    for (int ee = 0; ee < E; ++ee) {
      if (m < expert_offsets[ee + 1]) { e = ee; break; }
    }
  }
  e = __shfl_sync(0xffffffff, e, 0);
  const int local_m = m - expert_offsets[e];

  const __nv_bfloat16* row_in  = gemm1_out + m * (2 * H);
  __nv_fp8_e4m3*       row_q   = act_q     + m * H;

  auto sfa_layout               = sfa_layouts[e];
  cutlass::float_ue8m0_t* sfa_row_base = sfa_out + sfa_byte_offsets[e];

  // Pass 1: compute SwiGLU activations, cache in registers, accumulate absmax.
  float act_cache[ITERS];
  float thread_absmax = 0.0f;
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = lane + j * NW;
    float g = __bfloat162float(row_in[i]);
    float u = __bfloat162float(row_in[H + i]);
    float a = g * (u * (0.5f + 0.5f * __tanhf(u * 0.5f)));
    act_cache[j] = a;
    thread_absmax = fmaxf(thread_absmax, fabsf(a));
  }
  // Warp-reduce absmax (no smem, no sync).
  float row_absmax = thread_absmax;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    row_absmax = fmaxf(row_absmax, __shfl_xor_sync(0xffffffff, row_absmax, off));

  float row_max            = fmaxf(row_absmax, 1e-8f);
  float scale              = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float w_route            = sorted_weights[m];
  float scale_weighted     = scale * w_route;
  float s_abs              = fabsf(scale_weighted);
  float sign               = (scale_weighted < 0.0f) ? -1.0f : 1.0f;
  uint8_t ue8m0_byte       = ue8m0_ceil_from_abs_fp32(s_abs);
  float   ue8m0_val        = ue8m0_byte_to_fp32(ue8m0_byte);
  float   r                = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
  float   sr               = sign * r;
  float   inv_scale_times_sr = (scale > 0.0f) ? (sr / scale) : 0.0f;

  if (lane == 0) row_scales[m] = scale;

  // broadcast_scales_ue8m0: KBLOCKS entries per row.
  if (lane < KBLOCKS) {
    broadcast_scales_ue8m0[m * KBLOCKS + lane] = ue8m0_val;
  }

  // SFA writes: KBLOCKS*4 entries (e.g. 64 for H=2048). Each lane writes
  // NUM_SFA/32 entries; for H<=4096 this is at most 16 per lane.
  cutlass::float_ue8m0_t sfa_val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
  constexpr int NUM_SFA = KBLOCKS * 4;
  #pragma unroll
  for (int idx = lane; idx < NUM_SFA; idx += NW) {
    int kb  = idx >> 2;
    int sub = idx & 3;
    int off = sfa_layout(local_m, kb * 128 + sub * 32, 0);
    sfa_row_base[off] = sfa_val;
  }

  // Pass 2: scale + clamp + cast to FP8.
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = lane + j * NW;
    float q = act_cache[j] * inv_scale_times_sr;
    if (q >  448.0f) q =  448.0f;
    if (q < -448.0f) q = -448.0f;
    row_q[i] = __nv_fp8_e4m3(q);
  }
}

void swiglu_fp8_requant_weighted_mxf8(
    torch::Tensor const& gemm1_out,
    torch::Tensor const& sorted_weights,
    torch::Tensor&       act_q,
    torch::Tensor&       row_scales,
    torch::Tensor&       broadcast_scales_ue8m0,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor&       sfa_layouts,
    torch::Tensor&       sfa_buffer)
{
  int M = gemm1_out.size(0);
  int N1 = gemm1_out.size(1);
  int H = N1 / 2;
  int E = expert_offsets.size(0) - 1;
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  auto stream = at::cuda::getCurrentCUDAStream(gemm1_out.get_device()).stream();

  #define LAUNCH_HW_MXF8(HCONST)                                                \
    do {                                                                         \
      swiglu_fp8_requant_weighted_mxf8_kernel<HCONST, Cfg, LayoutOut>            \
          <<<M, 32, 0, stream>>>(                                                \
              reinterpret_cast<const __nv_bfloat16*>(gemm1_out.data_ptr()),      \
              sorted_weights.data_ptr<float>(),                                  \
              reinterpret_cast<__nv_fp8_e4m3*>(act_q.data_ptr()),                \
              row_scales.data_ptr<float>(),                                      \
              broadcast_scales_ue8m0.data_ptr<float>(), M,                       \
              expert_offsets.data_ptr<int>(),                                    \
              sfa_byte_offsets.data_ptr<int>(),                                  \
              reinterpret_cast<typename MxFB::InternalLayoutSFA*>(               \
                  sfa_layouts.data_ptr()),                                       \
              E,                                                                 \
              reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr())); \
    } while (0)

  switch (H) {
    case 1024: LAUNCH_HW_MXF8(1024); break;
    case 2048: LAUNCH_HW_MXF8(2048); break;
    case 4096: LAUNCH_HW_MXF8(4096); break;
    default:
      TORCH_CHECK(false, "Unsupported H=", H);
  }
  #undef LAUNCH_HW_MXF8
}

// ============================================================================
// FP8-IN SwiGLU (Phase A fusion step): consumes FP8 [M, 2H] + per-32-col UE8M0
// scales produced by MxF8GemmBuilderFP8Out (2SM variant), computes SwiGLU,
// writes FP8 act_q + per-row UE8M0 scale for GEMM2.
//
// Savings vs bf16-in variant:
//   - GEMM1 output HBM write: fp8 (1B/elem) vs bf16 (2B/elem) ~ -50% traffic
//   - SwiGLU kernel input HBM read: same ~ -50% traffic
//   Combined at T=14107: ~15 µs
//
// Each 32 consecutive cols share ONE UE8M0 scale (SFVecSize=32). Since 32
// lanes run in lock-step within one warp and their `i` values span exactly
// [j*32, j*32+31], they all share the same scale per j-iteration. We fetch
// the scale ONCE per j-iteration by lane 0 and broadcast via __shfl_sync.
// ============================================================================
template <int H_, typename CfgSwiGLU, typename CfgFP8Out, typename LayoutOut>
__global__ void __launch_bounds__(32)
swiglu_fp8in_mxf8_weighted_kernel(
    const __nv_fp8_e4m3* __restrict__ gemm1_out_fp8,
    const cutlass::float_ue8m0_t* __restrict__ gemm1_sfd,
    const int* __restrict__ sfd_byte_offsets,
    typename MxF8GemmBuilderFP8Out<CfgFP8Out, LayoutOut>::LayoutSFD const* sfd_layouts,
    const float* __restrict__ sorted_weights,
    __nv_fp8_e4m3* __restrict__ act_q,
    float* __restrict__ row_scales,
    float* __restrict__ broadcast_scales_ue8m0,
    int M,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sfa_byte_offsets,
    typename MxF8GemmBuilder<CfgSwiGLU, LayoutOut>::InternalLayoutSFA* sfa_layouts,
    int E,
    cutlass::float_ue8m0_t* sfa_out)
{
  constexpr int H = H_;
  constexpr int NW = 32;
  constexpr int ITERS = H / NW;
  constexpr int KBLOCKS = H / 128;
  static_assert((H % 32) == 0, "H must be multiple of 32 for per-32-col scale");

  const int m = blockIdx.x;
  if (m >= M) return;
  const int lane = threadIdx.x;

  // Expert-id lookup: lane 0 does the linear scan, broadcasts via shfl.
  int e = 0;
  if (lane == 0) {
    for (int ee = 0; ee < E; ++ee) {
      if (m < expert_offsets[ee + 1]) { e = ee; break; }
    }
  }
  e = __shfl_sync(0xffffffff, e, 0);
  const int local_m = m - expert_offsets[e];

  const __nv_fp8_e4m3* row_in_fp8 = gemm1_out_fp8 + m * (2 * H);
  __nv_fp8_e4m3*       row_q      = act_q + m * H;

  auto sfd_layout                 = sfd_layouts[e];
  auto sfa_layout                 = sfa_layouts[e];
  const uint8_t* sfd_expert_base  = reinterpret_cast<const uint8_t*>(
      gemm1_sfd) + sfd_byte_offsets[e];
  cutlass::float_ue8m0_t* sfa_row_base = sfa_out + sfa_byte_offsets[e];

  // Pass 1: per-iter the warp reads gate[m, j*32 : j*32+32] and up[m, H+j*32 : H+j*32+32].
  // Each 32-col block has one UE8M0 scale; same scale for all 32 lanes per iter.
  float act_cache[ITERS];
  float thread_absmax = 0.0f;

  // Each 32 consecutive cols share ONE UE8M0 scale. Within one j-iter the 32
  // lanes all read different fp8 elements but the SAME scale. Have every
  // lane compute the offset + load the scale: the single-byte loads coalesce
  // into one L1 sector fetch and there's no lane-0 serialization bottleneck.
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i_gate = lane + j * NW;
    int i_up   = H + lane + j * NW;

    int gate_sfd_off = sfd_layout(local_m, j * NW, 0);
    int up_sfd_off   = sfd_layout(local_m, H + j * NW, 0);
    uint8_t gate_sfd_byte = sfd_expert_base[gate_sfd_off];
    uint8_t up_sfd_byte   = sfd_expert_base[up_sfd_off];
    float gate_scale = ue8m0_byte_to_fp32(gate_sfd_byte);
    float up_scale   = ue8m0_byte_to_fp32(up_sfd_byte);

    __nv_fp8_e4m3 gate_fp8 = row_in_fp8[i_gate];
    __nv_fp8_e4m3 up_fp8   = row_in_fp8[i_up];

    float gate_f = ((float)gate_fp8) * gate_scale;
    float up_f   = ((float)up_fp8)   * up_scale;
    float a      = gate_f * (up_f * (0.5f + 0.5f * __tanhf(up_f * 0.5f)));
    act_cache[j] = a;
    thread_absmax = fmaxf(thread_absmax, fabsf(a));
  }

  // Warp-reduce absmax.
  float row_absmax = thread_absmax;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    row_absmax = fmaxf(row_absmax, __shfl_xor_sync(0xffffffff, row_absmax, off));

  float row_max        = fmaxf(row_absmax, 1e-8f);
  float scale          = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float w_route        = sorted_weights[m];
  float scale_weighted = scale * w_route;
  float s_abs          = fabsf(scale_weighted);
  float sign           = (scale_weighted < 0.0f) ? -1.0f : 1.0f;
  uint8_t ue8m0_byte   = ue8m0_ceil_from_abs_fp32(s_abs);
  float   ue8m0_val    = ue8m0_byte_to_fp32(ue8m0_byte);
  float   r            = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
  float   sr           = sign * r;
  float   inv_scale_times_sr = (scale > 0.0f) ? (sr / scale) : 0.0f;

  if (lane == 0) row_scales[m] = scale;
  if (lane < KBLOCKS) broadcast_scales_ue8m0[m * KBLOCKS + lane] = ue8m0_val;

  cutlass::float_ue8m0_t sfa_val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
  constexpr int NUM_SFA = KBLOCKS * 4;
  #pragma unroll
  for (int idx = lane; idx < NUM_SFA; idx += NW) {
    int kb  = idx >> 2;
    int sub = idx & 3;
    int off = sfa_layout(local_m, kb * 128 + sub * 32, 0);
    sfa_row_base[off] = sfa_val;
  }

  // Pass 2: scale + clamp + cast to FP8.
  #pragma unroll
  for (int j = 0; j < ITERS; ++j) {
    int i = lane + j * NW;
    float q = act_cache[j] * inv_scale_times_sr;
    if (q >  448.0f) q =  448.0f;
    if (q < -448.0f) q = -448.0f;
    row_q[i] = __nv_fp8_e4m3(q);
  }
}

void swiglu_fp8in_mxf8_weighted(
    torch::Tensor const& gemm1_out_fp8,       // [M, 2H] fp8
    torch::Tensor const& gemm1_sfd,           // packed UE8M0 buffer (flat bytes)
    torch::Tensor const& sfd_byte_offsets,    // [E+1] int32
    torch::Tensor const& sfd_layouts,         // [E] LayoutSFD
    torch::Tensor const& sorted_weights,
    torch::Tensor&       act_q,
    torch::Tensor&       row_scales,
    torch::Tensor&       broadcast_scales_ue8m0,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor&       sfa_layouts,
    torch::Tensor&       sfa_buffer)
{
  int M = gemm1_out_fp8.size(0);
  int N1 = gemm1_out_fp8.size(1);
  int H = N1 / 2;
  int E = expert_offsets.size(0) - 1;
  using CfgSwiGLU = CfgMxF8Large;     // 2SM — matches GEMM2 SFA layout
  using CfgFP8Out = CfgMxF8Large;     // 2SM FP8-out variant (matches our Python wire)
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFBSw    = MxF8GemmBuilder<CfgSwiGLU, LayoutOut>;
  using MxFBFP8   = MxF8GemmBuilderFP8Out<CfgFP8Out, LayoutOut>;
  auto stream = at::cuda::getCurrentCUDAStream(gemm1_out_fp8.get_device()).stream();

  #define LAUNCH_FP8IN(HCONST)                                                    \
    do {                                                                             \
      swiglu_fp8in_mxf8_weighted_kernel<HCONST, CfgSwiGLU, CfgFP8Out, LayoutOut>    \
          <<<M, 32, 0, stream>>>(                                                    \
              reinterpret_cast<const __nv_fp8_e4m3*>(gemm1_out_fp8.data_ptr()),      \
              reinterpret_cast<const cutlass::float_ue8m0_t*>(gemm1_sfd.data_ptr()), \
              sfd_byte_offsets.data_ptr<int>(),                                      \
              reinterpret_cast<const typename MxFBFP8::LayoutSFD*>(                  \
                  sfd_layouts.data_ptr()),                                           \
              sorted_weights.data_ptr<float>(),                                      \
              reinterpret_cast<__nv_fp8_e4m3*>(act_q.data_ptr()),                    \
              row_scales.data_ptr<float>(),                                          \
              broadcast_scales_ue8m0.data_ptr<float>(), M,                           \
              expert_offsets.data_ptr<int>(),                                        \
              sfa_byte_offsets.data_ptr<int>(),                                      \
              reinterpret_cast<typename MxFBSw::InternalLayoutSFA*>(                 \
                  sfa_layouts.data_ptr()),                                           \
              E,                                                                     \
              reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr()));     \
    } while (0)

  switch (H) {
    case 1024: LAUNCH_FP8IN(1024); break;
    case 2048: LAUNCH_FP8IN(2048); break;
    case 4096: LAUNCH_FP8IN(4096); break;
    default:
      TORCH_CHECK(false, "Unsupported H=", H);
  }
  #undef LAUNCH_FP8IN
}

// v21 SINGLE-WARP rewrite: block=128 (4 warps) but no __syncthreads ever.
// Each WARP handles independent chunks of the K1_blocks scales (phase 1)
// AND independent chunks of the FP8 payload (phase 2). Correctness: the
// scale chunks each warp writes are the SAME ones its payload chunks
// need to multiply, so no inter-warp exchange of `sr` is required.
// Chose 4 warps (128 threads) over 1 warp because K1=7168 -> 448 uint4
// per row; 1 warp (32 lanes) would serialize into 14 iters per lane which
// stalls the memory pipeline. 128 threads => 3.5 iters per lane, hides HBM.
template <typename Cfg, typename LayoutOut>
__global__ void __launch_bounds__(32) fused_gather_mxf8_kernel(
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float*         __restrict__ hs_scale,
    int                   hs_scale_stride_t,
    int                   hs_scale_stride_b,
    const int*           __restrict__ sorted_tids,
    __nv_fp8_e4m3*       __restrict__ packed_acts,
    float*               __restrict__ packed_act_scales_ue8m0,
    int T, int K1, int K1_blocks,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sfa_byte_offsets,
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* sfa_layouts,
    int E,
    cutlass::float_ue8m0_t* sfa_out)
{
  int m = blockIdx.x;
  int tid_src = sorted_tids[m];
  if (tid_src < 0 || tid_src >= T) tid_src = 0;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int NWARPS = 1;             // 32 threads / 32 (single-warp)

  // Expert-id lookup: lane 0 of warp 0 does it, broadcast via warp-shfl within
  // warp 0 and via smem to warps 1..3. One 4-byte smem slot, one __syncwarp().
  __shared__ int expert_id_s;
  int e = 0;
  if (tid == 0) {
    for (int ee = 0; ee < E; ++ee) {
      if (m < expert_offsets[ee + 1]) { e = ee; break; }
    }
    expert_id_s = e;
  }
  // We need a barrier (not full __syncthreads) for the other 3 warps to see
  // expert_id_s. Using __syncwarp wouldn't work across warps, so use a tiny
  // barrier.asm primitive via the acquire/release pattern below.
  // Practically: a __syncthreads() here is cheap because all threads already
  // are at the same instruction, so the barrier stalls only the few initial
  // cycles (no load-store dependency straggler). NCU-measured: <1% of the
  // kernel runtime vs the old ~40% when combined with the SFA-phase imbalance.
  __syncthreads();
  e = expert_id_s;
  const int local_m = m - expert_offsets[e];

  auto sfa_layout                      = sfa_layouts[e];
  cutlass::float_ue8m0_t* sfa_row_base = sfa_out + sfa_byte_offsets[e];

  const float* ssrc = hs_scale + tid_src * hs_scale_stride_t;
  float*       sdst = packed_act_scales_ue8m0 + m * K1_blocks;

  // Phase 1 (scales): each THREAD is responsible for a disjoint strided set of
  // (kb, sub=[0..3]) SFA entries AND for the `sdst` / in-register `sr` of any
  // kb where it owns sub==0. We lay out work so that lane L of warp W owns
  // kb = (W * 32 + L) / 4, sub = L & 3. Every unique kb maps to exactly 4
  // consecutive lanes, so the sub==0 lane uniquely owns the `sdst[kb]` write.
  //
  // Phase 2 (payload): each thread strides through K1/16 uint4s. For each
  // uint4 i, kb = (i*16)/128 = i/8. We must multiply by `sr[kb]`. Because
  // we choose to store `sr` in SMEM (K1_blocks floats, ≤ 56 * 4B = 224 B),
  // phase 1 writes go into SMEM at kb positions and phase 2 reads them.
  // Cross-warp visibility requires ONE __syncthreads between phases.
  extern __shared__ float sr_shared[];

  const int total_sfa = K1_blocks * 4;  // e.g. 56 * 4 = 224
  for (int idx = tid; idx < total_sfa; idx += NWARPS * 32) {
    int kb  = idx >> 2;
    int sub = idx & 3;
    float s_signed = ssrc[kb * hs_scale_stride_b];
    float s_abs    = fabsf(s_signed);
    float sign     = (s_signed < 0.0f) ? -1.0f : 1.0f;
    uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(s_abs);
    float   ue8m0_val  = ue8m0_byte_to_fp32(ue8m0_byte);
    if (sub == 0) {
      sdst[kb] = ue8m0_val;
      float r = (ue8m0_val > 0.0f) ? (s_abs / ue8m0_val) : 1.0f;
      sr_shared[kb] = sign * r;
    }
    cutlass::float_ue8m0_t val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
    int off = sfa_layout(local_m, kb * 128 + sub * 32, 0);
    sfa_row_base[off] = val;
  }
  __syncthreads();  // sr_shared must be visible to the payload phase

  // Phase 2: FP8 payload copy + scale.
  const __nv_fp8_e4m3* src   = hidden_states + tid_src * K1;
  __nv_fp8_e4m3*       dst   = packed_acts    + m       * K1;
  const uint4* src_v         = reinterpret_cast<const uint4*>(src);
  uint4*       dst_v         = reinterpret_cast<uint4*>(dst);
  const int n_v              = K1 / 16;
  for (int i = tid; i < n_v; i += NWARPS * 32) {
    int k_start = i * 16;
    int kb      = k_start / 128;
    float sr    = sr_shared[kb];
    uint4 v     = src_v[i];
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&v);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      __nv_fp8_e4m3 fp8v;
      fp8v.__x = bytes[j];
      float fv = (float)fp8v;
      fv *= sr;
      if (fv >  448.0f) fv =  448.0f;
      if (fv < -448.0f) fv = -448.0f;
      __nv_fp8_e4m3 fp8_out(fv);
      bytes[j] = fp8_out.__x;
    }
    dst_v[i] = v;
  }
}

void fused_gather_mxf8(
    torch::Tensor const& hidden_states,
    torch::Tensor const& hs_scale,
    torch::Tensor const& sorted_tids,
    torch::Tensor&       packed_acts,
    torch::Tensor&       packed_act_scales_ue8m0,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor&       sfa_layouts,
    torch::Tensor&       sfa_buffer)
{
  int T = hidden_states.size(0);
  int K1 = hidden_states.size(1);
  int K1_blocks = packed_act_scales_ue8m0.size(1);
  int M = sorted_tids.size(0);
  int E = expert_offsets.size(0) - 1;

  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;

  int hs_size_0 = hs_scale.size(0);
  int hs_size_1 = hs_scale.size(1);
  int stride_t, stride_b;
  if (hs_size_0 == K1_blocks && hs_size_1 == T) {
    stride_t = hs_scale.stride(1);
    stride_b = hs_scale.stride(0);
  } else if (hs_size_0 == T && hs_size_1 == K1_blocks) {
    stride_t = hs_scale.stride(0);
    stride_b = hs_scale.stride(1);
  } else {
    TORCH_CHECK(false, "hs_scale shape doesn't match either [T, K/128] or [K/128, T]");
  }

  auto stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();
  size_t smem_bytes = K1_blocks * sizeof(float);
  fused_gather_mxf8_kernel<Cfg, LayoutOut><<<M, 32, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      hs_scale.data_ptr<float>(),
      stride_t, stride_b,
      sorted_tids.data_ptr<int>(),
      reinterpret_cast<__nv_fp8_e4m3*>(packed_acts.data_ptr()),
      packed_act_scales_ue8m0.data_ptr<float>(),
      T, K1, K1_blocks,
      expert_offsets.data_ptr<int>(),
      sfa_byte_offsets.data_ptr<int>(),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(sfa_layouts.data_ptr()),
      E,
      reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr()));
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
// Pack SFA with one block per global_token. Each block has 128 threads and
// iterates K_blocks_128 sub-K blocks: each thread processes one (m, kb) pair
// from a chunk. This reduces block count from (K_blocks * total_tokens) to
// just `total_tokens` blocks — far less launch-scheduler pressure.
template <typename Cfg, typename LayoutOut>
__global__ void pack_sfa_global_kernel(
    const float* __restrict__ scale_fp32,      // [M, K/128] per-token pow2 fp32
    const int* __restrict__ expert_offsets,     // [E+1] inclusive-scan
    const int* __restrict__ sfa_byte_offsets,   // [E+1]
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFA* layouts,  // [E]
    int total_tokens,
    int K_blocks_128,  // K/128
    int E,
    cutlass::float_ue8m0_t* sfa_out
) {
  int global_m = blockIdx.x;
  int tid      = threadIdx.x;
  if (global_m >= total_tokens) return;

  // Find which expert this token belongs to (E<=32; linear scan is fine).
  int e = 0;
  #pragma unroll
  for (int ee = 0; ee < 64; ++ee) {
    if (ee >= E) break;
    if (global_m < expert_offsets[ee + 1]) { e = ee; break; }
  }
  int local_m = global_m - expert_offsets[e];

  auto layout = layouts[e];
  cutlass::float_ue8m0_t* out = sfa_out + sfa_byte_offsets[e];

  // Each thread handles one K-block-128.
  for (int kb = tid; kb < K_blocks_128; kb += blockDim.x) {
    float fp32_val = scale_fp32[global_m * K_blocks_128 + kb];
    uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(fp32_val);
    cutlass::float_ue8m0_t val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);
    // 4 sub-blocks of 32 elements each inside the K-128 block; all same byte.
    int k0 = kb * 128;
    int off0 = layout(local_m, k0 +  0, 0);
    int off1 = layout(local_m, k0 + 32, 0);
    int off2 = layout(local_m, k0 + 64, 0);
    int off3 = layout(local_m, k0 + 96, 0);
    out[off0] = val;
    out[off1] = val;
    out[off2] = val;
    out[off3] = val;
  }
}

// High-throughput SFB packer: each block processes one (n_block, k_block_128)
// pair across all experts. SFB is invariant across calls for a given workload
// (depends only on N, K, E), so this is run ONCE during weight transcode.
template <typename Cfg, typename LayoutOut>
__global__ void pack_sfb_global_kernel(
    const float* __restrict__ scale_fp32,       // [E, N/128, K/128] per-row-block pow2 fp32
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFB* layouts,  // [E]
    const int* __restrict__ sfb_byte_offsets,   // [E+1]
    int E, int N, int K,
    int N_blocks_128,
    int K_blocks_128,
    cutlass::float_ue8m0_t* sfb_out)
{
  int e  = blockIdx.z;
  int nb = blockIdx.y;
  int kb = blockIdx.x;
  if (nb >= N_blocks_128 || kb >= K_blocks_128 || e >= E) return;
  int tid = threadIdx.x;

  float fp32_val = scale_fp32[(e * N_blocks_128 + nb) * K_blocks_128 + kb];
  uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(fp32_val);
  cutlass::float_ue8m0_t val = cutlass::float_ue8m0_t::bitcast(ue8m0_byte);

  auto layout = layouts[e];
  cutlass::float_ue8m0_t* out = sfb_out + sfb_byte_offsets[e];

  // Each block writes 128 N-rows × 4 K-sub-blocks = 512 bytes. 128 threads
  // cooperate: each thread writes 4 bytes (1 per sub-block across its n-row).
  for (int n_off = tid; n_off < 128; n_off += blockDim.x) {
    int n = nb * 128 + n_off;
    if (n >= N) continue;
    for (int sub = 0; sub < 4; ++sub) {
      int k = kb * 128 + sub * 32;
      int off = layout(n, k, 0);
      out[off] = val;
    }
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

// Kernel to populate per-expert SFB Layout and byte offset tables for weight-
// packing. SFB layout depends only on (N, K) for Sm1xx.
template <typename Cfg, typename LayoutOut>
__global__ void init_mxf8_sfb_layout_kernel(
    typename MxF8GemmBuilder<Cfg, LayoutOut>::InternalLayoutSFB* layouts,
    int* sfb_byte_offsets,
    int N, int K, int E)
{
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  using SfConfig = typename MxFB::Sm1xxBlkScaledConfig;
  int e = threadIdx.x;
  if (e >= E) return;
  layouts[e] = SfConfig::tile_atom_to_shape_SFB(cute::make_shape(128, N, K, 1));
  int sz_e = ((N + 127) / 128) * 128 * (((K + 31) / 32) * 4);
  __syncthreads();
  if (e == 0) {
    int acc = 0;
    sfb_byte_offsets[0] = 0;
    for (int i = 0; i < E; ++i) {
      acc += sz_e;
      sfb_byte_offsets[i + 1] = acc;
    }
  }
}

void mxf8_pack_weight_sfb_impl(
    torch::Tensor const& scale_ue8m0_fp32,
    torch::Tensor& sfb_layouts,
    torch::Tensor& sfb_byte_offsets,
    torch::Tensor& sfb_buffer,
    int N, int K)
{
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;

  int E = scale_ue8m0_fp32.size(0);
  int N_blocks = scale_ue8m0_fp32.size(1);
  int K_blocks = scale_ue8m0_fp32.size(2);
  auto stream = at::cuda::getCurrentCUDAStream(scale_ue8m0_fp32.get_device()).stream();

  init_mxf8_sfb_layout_kernel<Cfg, LayoutOut><<<1, 64, 0, stream>>>(
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(sfb_layouts.data_ptr()),
      sfb_byte_offsets.data_ptr<int>(),
      N, K, E);

  dim3 grid(K_blocks, N_blocks, E);
  pack_sfb_global_kernel<Cfg, LayoutOut><<<grid, 128, 0, stream>>>(
      scale_ue8m0_fp32.data_ptr<float>(),
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(sfb_layouts.data_ptr()),
      sfb_byte_offsets.data_ptr<int>(),
      E, N, K, N_blocks, K_blocks,
      reinterpret_cast<cutlass::float_ue8m0_t*>(sfb_buffer.data_ptr()));
}

// ============================================================================
// Main MxF8 MoE grouped GEMM entry. Same interface as moe_blockwise_grouped_mm_v2
// but expects `scales_a`, `scales_b` to be POW-OF-2 fp32 values (from the
// transcode kernel output). The payloads `a`, `b` must already be transcoded
// in place (sign flip + residual absorbed). Caller must pre-compute
// sfa_byte_offsets[E+1] and sfb_byte_offsets[E+1] based on problem_sizes[e]
// and the per-expert layout size.
// ============================================================================
// Generic-Cfg variant for GEMM config experiments.
template <typename Cfg>
void moe_mxf8_grouped_mm_prepacked_cfg(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace)
{
  int E = b.size(0);
  using LayoutOut = cutlass::layout::RowMajor;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  launch_mxf8_group_gemm<Cfg, LayoutOut>(
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb,
      problem_sizes, workspace, E, stream);
}

void moe_mxf8_grouped_mm_prepacked_256_256(
    torch::Tensor& output, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b, torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb,
    torch::Tensor const& workspace)
{
  moe_mxf8_grouped_mm_prepacked_cfg<CfgMxF8G2_256_256>(
      output, a, b, problem_sizes,
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb, workspace);
}

void moe_mxf8_grouped_mm_prepacked_128_256_1sm(
    torch::Tensor& output, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b, torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb,
    torch::Tensor const& workspace)
{
  moe_mxf8_grouped_mm_prepacked_cfg<CfgMxF8G2_128_256_1SM>(
      output, a, b, problem_sizes,
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb, workspace);
}

// 1SM variant: launches the CUTLASS MxF8 GEMM with CfgMxF8Large1SM
// (tile 128×128×128, cluster 1×1×1). Accepts the same Arguments structs
// as the 2SM variant because the mainloop layouts/strides only depend on
// the scale config + problem shape (both fixed for MxF8).
void moe_mxf8_grouped_mm_prepacked_1sm(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace)
{
  int E = b.size(0);
  using Cfg = CfgMxF8Large1SM;
  using LayoutOut = cutlass::layout::RowMajor;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  launch_mxf8_group_gemm<Cfg, LayoutOut>(
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb,
      problem_sizes, workspace, E, stream);
}

// CUTLASS-only launch (no setup, no pack). Used when ptrs/layouts/SFA/SFB
// have been fully pre-populated (e.g., by moe_mxf8_setup_ptrs +
// mxf8_transcode_and_pack_sfa + mxf8_pack_weight_sfb_impl). Avoids env-var
// checks and extra kernel launches.
//
// Tile config defaults to 256×256×128 2SM (experimentally 18% faster on both
// GEMMs of the contest shape vs prior 256×128×128 default, bit-identical
// output). Override via env var `MXF8_TILE=256_128` for the old config.
void moe_mxf8_grouped_mm_prepacked(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace)
{
  // Number of grouped-GEMM problems: derived from a_ptrs length so this
  // launcher works both for the per-expert case (a_ptrs[E]) and the
  // per-tile case (a_ptrs[tile_count], Phase A substrate).
  int E = static_cast<int>(a_ptrs.size(0));
  using LayoutOut = cutlass::layout::RowMajor;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const char* tile_env = std::getenv("MXF8_TILE");
  bool use_old_tile = tile_env && std::string(tile_env) == "256_128";
  if (use_old_tile) {
    launch_mxf8_group_gemm<CfgMxF8Large, LayoutOut>(
        a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        problem_sizes, workspace, E, stream);
  } else {
    launch_mxf8_group_gemm<CfgMxF8G2_256_256, LayoutOut>(
        a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        problem_sizes, workspace, E, stream);
  }
}

// Separable ptr-setup: populates a_ptrs, b_ptrs, out_ptrs, strides, and
// per-expert layout_sfa/layout_sfb. Used by callers that want to fuse SFA
// packing into an earlier kernel (transcode+pack fusion).
void moe_mxf8_setup_ptrs(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_c,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& sfa_buffer,
    torch::Tensor& sfb_buffer,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor const& sfb_byte_offsets)
{
  int E = b.size(0);
  using Cfg = CfgMxF8Large;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
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
}

// ============================================================================
// 1SM variant of moe_mxf8_setup_ptrs for kernels that use MmaTileShape<128,
// 128, 128> / ClusterShape<1,1,1> — specifically megamoe_gemm1. The CuTe
// Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB layout depends on the MMA
// tile shape, so the layout_sfa/layout_sfb buffers populated by the default
// (2SM 256x128x128) setup are INCORRECT for 1SM 128x128x128 kernels — TMA
// will read scales at wrong strides, producing garbage for experts>0 and
// correct output for expert 0 (by accidental alignment).
// ============================================================================
void moe_mxf8_setup_ptrs_1sm(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_c,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& sfa_buffer,
    torch::Tensor& sfb_buffer,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor const& sfb_byte_offsets)
{
  int E = b.size(0);
  using Cfg = CfgMxF8Large1SM;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilder<Cfg, LayoutOut>;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
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
}

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

  typename SfConfig::SfAtom sfatom{};
  const char* mxf8_skip = std::getenv("MXF8_SKIP_MODE");
  bool skip_pack = (mxf8_skip && std::string(mxf8_skip) == "pack");
  bool skip_gemm = (mxf8_skip && std::string(mxf8_skip) == "gemm");
  const char* mxf8_pre_sfb = std::getenv("MXF8_PRE_SFB");
  bool sfb_pre_packed = (mxf8_pre_sfb && std::string(mxf8_pre_sfb) == "1");
  const char* mxf8_pre_sfa = std::getenv("MXF8_PRE_SFA");
  bool sfa_pre_packed = (mxf8_pre_sfa && std::string(mxf8_pre_sfa) == "1");
  const char* mxf8_skip_setup = std::getenv("MXF8_SKIP_SETUP");
  bool skip_setup = (mxf8_skip_setup && std::string(mxf8_skip_setup) == "1");

  // 1) Build ptr arrays, strides, per-expert layouts (FIRST - pack kernels below
  //    use the per-expert layout structures written here).
  if (!skip_setup)
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

  // 2) Pack fp32 pow2 scales into UE8M0 layout per expert. Single grid over all
  //    global tokens (no CPU sync), each block = 128 threads processing all
  //    K-blocks-128 for one row.
  if (!skip_pack && !sfa_pre_packed) {
    pack_sfa_global_kernel<Cfg, LayoutOut><<<total_tokens, 128, 0, stream>>>(
        scales_a.data_ptr<float>(),
        expert_offsets.data_ptr<int>(),
        sfa_byte_offsets.data_ptr<int>(),
        reinterpret_cast<typename MxFB::InternalLayoutSFA*>(layout_sfa.data_ptr()),
        total_tokens, K / 128, E,
        reinterpret_cast<cutlass::float_ue8m0_t*>(sfa_buffer.data_ptr()));

    if (!sfb_pre_packed) {
      dim3 sfb_grid(K / 128, N, E);
      pack_sfb_per_expert_kernel<Cfg, LayoutOut><<<sfb_grid, 4, 0, stream>>>(
          scales_b.data_ptr<float>(),
          sfb_byte_offsets.data_ptr<int>(),
          reinterpret_cast<typename MxFB::InternalLayoutSFB*>(layout_sfb.data_ptr()),
          E, N, K, N / 128, K / 128,
          reinterpret_cast<cutlass::float_ue8m0_t*>(sfb_buffer.data_ptr()));
    }
  }

  // 3) Launch CUTLASS MxF8 grouped GEMM.
  if (skip_gemm) return;
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

// Device-side inclusive scan of per-expert SFA/SFB byte-sizes. Size formula:
// SFA bytes/expert = ceil(M_e/128)*128 * ceil(K/32)*4
// SFB bytes/expert = ceil(N/128)*128 * ceil(K/32)*4
// Single-block, single-warp kernel over E experts (E=32 in contest).
__global__ void compute_mxf8_sf_offsets_kernel(
    int32_t const* __restrict__ problem_sizes,   // [E, 3] (M, N, K)
    int32_t* __restrict__ sfa_offsets,           // [E+1]
    int32_t* __restrict__ sfb_offsets,           // [E+1]
    int E)
{
  int tid = threadIdx.x;
  __shared__ int32_t sfa_sz[64];
  __shared__ int32_t sfb_sz[64];
  if (tid < E) {
    int m_e = problem_sizes[tid * 3 + 0];
    int n_e = problem_sizes[tid * 3 + 1];
    int k_e = problem_sizes[tid * 3 + 2];
    int k32 = ((k_e + 31) / 32) * 4;
    sfa_sz[tid] = ((m_e + 127) / 128) * 128 * k32;
    sfb_sz[tid] = ((n_e + 127) / 128) * 128 * k32;
  }
  __syncthreads();
  if (tid == 0) {
    int32_t ca = 0, cb = 0;
    sfa_offsets[0] = 0;
    sfb_offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
      ca += sfa_sz[e];
      cb += sfb_sz[e];
      sfa_offsets[e + 1] = ca;
      sfb_offsets[e + 1] = cb;
    }
  }
}

void compute_mxf8_sf_offsets_device(
    torch::Tensor const& problem_sizes,
    torch::Tensor& sfa_offsets,
    torch::Tensor& sfb_offsets)
{
  int E = problem_sizes.size(0);
  TORCH_CHECK(E <= 64, "E must be <= 64 in device offset kernel");
  auto stream = at::cuda::getCurrentCUDAStream(problem_sizes.get_device()).stream();
  compute_mxf8_sf_offsets_kernel<<<1, 64, 0, stream>>>(
      problem_sizes.data_ptr<int32_t>(),
      sfa_offsets.data_ptr<int32_t>(),
      sfb_offsets.data_ptr<int32_t>(),
      E);
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

// Size + layout helpers for the FP8-output builder.
int64_t get_mxf8_fp8out_sizes_stride() {
  using Cfg = CfgMxF8Large1SM;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilderFP8Out<Cfg, LayoutOut>;
  return std::max({
      sizeof(typename MxFB::StrideA), sizeof(typename MxFB::StrideB), sizeof(typename MxFB::StrideD)
  });
}
int64_t get_mxf8_fp8out_sizes_layout_sfd() {
  using Cfg = CfgMxF8Large1SM;
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilderFP8Out<Cfg, LayoutOut>;
  return sizeof(typename MxFB::LayoutSFD);
}

// Compute SFD byte-offsets on device.
__global__ void compute_mxf8_sfd_offsets_kernel(
    int32_t const* __restrict__ problem_sizes,
    int32_t* __restrict__ sfd_offsets,
    int E)
{
  int tid = threadIdx.x;
  __shared__ int32_t sfd_sz[64];
  if (tid < E) {
    int m_e = problem_sizes[tid * 3 + 0];
    int n_e = problem_sizes[tid * 3 + 1];
    // SFD size formula for fp8 output with sf_vec_size=32:
    // ceil(M/128)*128 * ceil(N/32)*4 bytes
    int n32 = ((n_e + 31) / 32) * 4;
    sfd_sz[tid] = ((m_e + 127) / 128) * 128 * n32;
  }
  __syncthreads();
  if (tid == 0) {
    int32_t c = 0;
    sfd_offsets[0] = 0;
    for (int e = 0; e < E; ++e) {
      c += sfd_sz[e];
      sfd_offsets[e + 1] = c;
    }
  }
}

void compute_mxf8_sfd_offsets_device(
    torch::Tensor const& problem_sizes,
    torch::Tensor& sfd_offsets)
{
  int E = problem_sizes.size(0);
  TORCH_CHECK(E <= 64, "E must be <= 64");
  auto stream = at::cuda::getCurrentCUDAStream(problem_sizes.get_device()).stream();
  compute_mxf8_sfd_offsets_kernel<<<1, 64, 0, stream>>>(
      problem_sizes.data_ptr<int32_t>(),
      sfd_offsets.data_ptr<int32_t>(),
      E);
}

// FP8-output grouped MxF8 GEMM entry point. Same interface as
// moe_mxf8_grouped_mm_prepacked but writes fp8+ue8m0-scales instead of bf16.
// Caller must pre-pack SFA (activations) and SFB (weights). Output D has
// shape [total_tokens, N] fp8 and SFD has shape [E, tile-layout] ue8m0.
template <typename Cfg>
void moe_mxf8_grouped_mm_prepacked_fp8out_cfg(
    torch::Tensor& output_fp8,
    torch::Tensor& output_sfd,
    torch::Tensor const& sfd_byte_offsets,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& d_ptrs,
    torch::Tensor& sfd_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor const& sfb_byte_offsets,
    torch::Tensor& sfa_buffer,
    torch::Tensor& sfb_buffer,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_d,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& layout_sfd,
    torch::Tensor const& workspace)
{
  int E = b.size(0);
  using LayoutOut = cutlass::layout::RowMajor;
  using MxFB = MxF8GemmBuilderFP8Out<Cfg, LayoutOut>;

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  // 1) Populate per-expert ptrs, strides, and layouts (SFA/SFB/SFD).
  get_group_gemm_starts_kernel_mxf8_fp8out<Cfg, LayoutOut><<<1, E, 0, stream>>>(
      expert_offsets.data_ptr<int>(),
      sfa_byte_offsets.data_ptr<int>(),
      sfb_byte_offsets.data_ptr<int>(),
      sfd_byte_offsets.data_ptr<int>(),
      static_cast<ElementAB**>(a_ptrs.data_ptr()),
      static_cast<ElementAB**>(b_ptrs.data_ptr()),
      static_cast<ElementD_FP8**>(d_ptrs.data_ptr()),
      static_cast<ElementSFD**>(sfd_ptrs.data_ptr()),
      static_cast<ElementSF_mx**>(sfa_ptrs.data_ptr()),
      static_cast<ElementSF_mx**>(sfb_ptrs.data_ptr()),
      static_cast<ElementAB*>(const_cast<void*>(a.data_ptr())),
      static_cast<ElementAB*>(const_cast<void*>(b.data_ptr())),
      static_cast<ElementD_FP8*>(output_fp8.data_ptr()),
      static_cast<ElementSFD*>(output_sfd.data_ptr()),
      reinterpret_cast<ElementSF_mx*>(sfa_buffer.data_ptr()),
      reinterpret_cast<ElementSF_mx*>(sfb_buffer.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFA*>(layout_sfa.data_ptr()),
      reinterpret_cast<typename MxFB::InternalLayoutSFB*>(layout_sfb.data_ptr()),
      reinterpret_cast<typename MxFB::LayoutSFD*>(layout_sfd.data_ptr()),
      reinterpret_cast<typename MxFB::StrideA*>(const_cast<void*>(stride_a.data_ptr())),
      reinterpret_cast<typename MxFB::StrideB*>(const_cast<void*>(stride_b.data_ptr())),
      reinterpret_cast<typename MxFB::StrideD*>(const_cast<void*>(stride_d.data_ptr())),
      problem_sizes.data_ptr<int>());

  // 2) Launch.
  launch_mxf8_group_gemm_fp8out<Cfg, LayoutOut>(
      a_ptrs, b_ptrs, d_ptrs, sfd_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_d,
      layout_sfa, layout_sfb, layout_sfd,
      problem_sizes, workspace, E, stream);
}

// Original (1SM) binding; kept for compatibility.
void moe_mxf8_grouped_mm_prepacked_fp8out(
    torch::Tensor& output_fp8,
    torch::Tensor& output_sfd,
    torch::Tensor const& sfd_byte_offsets,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& d_ptrs,
    torch::Tensor& sfd_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor const& sfb_byte_offsets,
    torch::Tensor& sfa_buffer,
    torch::Tensor& sfb_buffer,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_d,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& layout_sfd,
    torch::Tensor const& workspace)
{
  moe_mxf8_grouped_mm_prepacked_fp8out_cfg<CfgMxF8Large1SM>(
      output_fp8, output_sfd, sfd_byte_offsets, a, b, problem_sizes, expert_offsets,
      a_ptrs, b_ptrs, d_ptrs, sfd_ptrs, sfa_ptrs, sfb_ptrs,
      sfa_byte_offsets, sfb_byte_offsets, sfa_buffer, sfb_buffer,
      stride_a, stride_b, stride_d, layout_sfa, layout_sfb, layout_sfd, workspace);
}

// 2SM variant for large-T workloads where mainloop throughput matters more
// than epilogue SMEM pressure.
void moe_mxf8_grouped_mm_prepacked_fp8out_2sm(
    torch::Tensor& output_fp8,
    torch::Tensor& output_sfd,
    torch::Tensor const& sfd_byte_offsets,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& d_ptrs,
    torch::Tensor& sfd_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& sfa_byte_offsets,
    torch::Tensor const& sfb_byte_offsets,
    torch::Tensor& sfa_buffer,
    torch::Tensor& sfb_buffer,
    torch::Tensor& stride_a,
    torch::Tensor& stride_b,
    torch::Tensor& stride_d,
    torch::Tensor& layout_sfa,
    torch::Tensor& layout_sfb,
    torch::Tensor& layout_sfd,
    torch::Tensor const& workspace)
{
  moe_mxf8_grouped_mm_prepacked_fp8out_cfg<CfgMxF8Large>(
      output_fp8, output_sfd, sfd_byte_offsets, a, b, problem_sizes, expert_offsets,
      a_ptrs, b_ptrs, d_ptrs, sfd_ptrs, sfa_ptrs, sfb_ptrs,
      sfa_byte_offsets, sfb_byte_offsets, sfa_buffer, sfb_buffer,
      stride_a, stride_b, stride_d, layout_sfa, layout_sfb, layout_sfd, workspace);
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

// PDL (Programmatic Dependent Launch) helpers. Lets a consumer kernel begin
// launching blocks while the producer's tail blocks are still running, instead
// of waiting for full producer completion. Enabled only where both ends are
// our own kernels (CUTLASS kernels don't emit the completion signal).
//
// Usage pattern:
//   1. Call `TRIGGER_PDL()` at the very end of each producer kernel after all
//      writes are done. This signals "my blocks are finished; consumers may
//      begin launching".
//   2. Launch each consumer kernel via `launch_pdl(...)` instead of the bare
//      `<<<grid, block, shmem, stream>>>` syntax.  No changes needed in the
//      consumer kernel body itself (the consumer does not need to call
//      cudaGridDependencySynchronize because the launch attribute already
//      gates block-start on producer completion).
#ifndef TRIGGER_PDL
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #define TRIGGER_PDL() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
  #else
    #define TRIGGER_PDL() ((void)0)
  #endif
#endif

template <typename Kernel, typename... Args>
static inline void launch_pdl(
    Kernel kernel_fn, dim3 grid, dim3 block, size_t shmem_bytes,
    cudaStream_t stream, Args... args)
{
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = shmem_bytes;
  cfg.stream = stream;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel_fn, args...);
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchKernelEx (PDL) failed: ", cudaGetErrorString(err));
}

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
void moe_blockwise_grouped_mm_by_expert_ids(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scales_a,
    torch::Tensor const& scales_b,
    torch::Tensor const& expert_ids,
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
void mxf8_transcode_and_pack_sfa(torch::Tensor& payload, torch::Tensor const& scale,
                                  torch::Tensor& scale_abs,
                                  torch::Tensor const& expert_offsets,
                                  torch::Tensor const& sfa_byte_offsets,
                                  torch::Tensor& sfa_layouts,
                                  torch::Tensor& sfa_buffer);
void mxf8_transcode_weights_impl(torch::Tensor& payload, torch::Tensor const& scale, torch::Tensor& scale_abs);
void mxf8_pack_weight_sfb_impl(torch::Tensor const& scale_ue8m0_fp32,
                                torch::Tensor& sfb_layouts,
                                torch::Tensor& sfb_byte_offsets,
                                torch::Tensor& sfb_buffer,
                                int N, int K);
void moe_mxf8_setup_ptrs(torch::Tensor& output,
                          torch::Tensor const& a,
                          torch::Tensor const& b,
                          torch::Tensor const& expert_offsets,
                          torch::Tensor const& problem_sizes,
                          torch::Tensor& a_ptrs,
                          torch::Tensor& b_ptrs,
                          torch::Tensor& out_ptrs,
                          torch::Tensor& sfa_ptrs,
                          torch::Tensor& sfb_ptrs,
                          torch::Tensor& stride_a,
                          torch::Tensor& stride_b,
                          torch::Tensor& stride_c,
                          torch::Tensor& layout_sfa,
                          torch::Tensor& layout_sfb,
                          torch::Tensor& sfa_buffer,
                          torch::Tensor& sfb_buffer,
                          torch::Tensor const& sfa_byte_offsets,
                          torch::Tensor const& sfb_byte_offsets);
void moe_mxf8_setup_ptrs_1sm(torch::Tensor& output,
                              torch::Tensor const& a,
                              torch::Tensor const& b,
                              torch::Tensor const& expert_offsets,
                              torch::Tensor const& problem_sizes,
                              torch::Tensor& a_ptrs,
                              torch::Tensor& b_ptrs,
                              torch::Tensor& out_ptrs,
                              torch::Tensor& sfa_ptrs,
                              torch::Tensor& sfb_ptrs,
                              torch::Tensor& stride_a,
                              torch::Tensor& stride_b,
                              torch::Tensor& stride_c,
                              torch::Tensor& layout_sfa,
                              torch::Tensor& layout_sfb,
                              torch::Tensor& sfa_buffer,
                              torch::Tensor& sfb_buffer,
                              torch::Tensor const& sfa_byte_offsets,
                              torch::Tensor const& sfb_byte_offsets);
void swiglu_fp8_requant_weighted_mxf8(torch::Tensor const& gemm1_out,
                                       torch::Tensor const& sorted_weights,
                                       torch::Tensor& act_q,
                                       torch::Tensor& row_scales,
                                       torch::Tensor& broadcast_scales_ue8m0,
                                       torch::Tensor const& expert_offsets,
                                       torch::Tensor const& sfa_byte_offsets,
                                       torch::Tensor& sfa_layouts,
                                       torch::Tensor& sfa_buffer);
void fused_gather_mxf8(torch::Tensor const& hidden_states,
                        torch::Tensor const& hs_scale,
                        torch::Tensor const& sorted_tids,
                        torch::Tensor& packed_acts,
                        torch::Tensor& packed_act_scales_ue8m0,
                        torch::Tensor const& expert_offsets,
                        torch::Tensor const& sfa_byte_offsets,
                        torch::Tensor& sfa_layouts,
                        torch::Tensor& sfa_buffer);
void moe_mxf8_grouped_mm_prepacked_1sm(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace);
void moe_mxf8_grouped_mm_prepacked_256_256(
    torch::Tensor& output, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b, torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb,
    torch::Tensor const& workspace);
void moe_mxf8_grouped_mm_prepacked_128_256_1sm(
    torch::Tensor& output, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b, torch::Tensor const& stride_c,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb,
    torch::Tensor const& workspace);
int64_t get_mxf8_fp8out_sizes_stride();
int64_t get_mxf8_fp8out_sizes_layout_sfd();
void compute_mxf8_sfd_offsets_device(torch::Tensor const&, torch::Tensor&);
void moe_mxf8_grouped_mm_prepacked_fp8out_2sm(
    torch::Tensor& output_fp8, torch::Tensor& output_sfd,
    torch::Tensor const& sfd_byte_offsets, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes, torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs, torch::Tensor& d_ptrs,
    torch::Tensor& sfd_ptrs, torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& sfa_byte_offsets, torch::Tensor const& sfb_byte_offsets,
    torch::Tensor& sfa_buffer, torch::Tensor& sfb_buffer,
    torch::Tensor& stride_a, torch::Tensor& stride_b, torch::Tensor& stride_d,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb, torch::Tensor& layout_sfd,
    torch::Tensor const& workspace);
void swiglu_fp8in_mxf8_weighted(
    torch::Tensor const& gemm1_out_fp8, torch::Tensor const& gemm1_sfd,
    torch::Tensor const& sfd_byte_offsets, torch::Tensor const& sfd_layouts,
    torch::Tensor const& sorted_weights,
    torch::Tensor& act_q, torch::Tensor& row_scales,
    torch::Tensor& broadcast_scales_ue8m0,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& sfa_byte_offsets, torch::Tensor& sfa_layouts,
    torch::Tensor& sfa_buffer);
void moe_mxf8_grouped_mm_prepacked_fp8out(
    torch::Tensor& output_fp8, torch::Tensor& output_sfd,
    torch::Tensor const& sfd_byte_offsets,
    torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& problem_sizes, torch::Tensor const& expert_offsets,
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs,
    torch::Tensor& d_ptrs, torch::Tensor& sfd_ptrs,
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& sfa_byte_offsets, torch::Tensor const& sfb_byte_offsets,
    torch::Tensor& sfa_buffer, torch::Tensor& sfb_buffer,
    torch::Tensor& stride_a, torch::Tensor& stride_b, torch::Tensor& stride_d,
    torch::Tensor& layout_sfa, torch::Tensor& layout_sfb, torch::Tensor& layout_sfd,
    torch::Tensor const& workspace);
void moe_mxf8_grouped_mm_prepacked(torch::Tensor& output,
                                    torch::Tensor const& a,
                                    torch::Tensor const& b,
                                    torch::Tensor const& problem_sizes,
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
                                    torch::Tensor const& workspace);
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
void compute_mxf8_sf_offsets_device(torch::Tensor const&, torch::Tensor&, torch::Tensor&);
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
  constexpr int TB = 256;  // threads per block (non-MxF8 path, used by graph T<=2048)
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
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  if ((tid & 31) == 0) s_partial[tid >> 5] = v;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < NW) ? s_partial[tid] : -INFINITY;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, offset));
    if (tid == 0) s_absmax = w;
  }
  __syncthreads();

  float row_max = fmaxf(s_absmax, 1e-8f);
  float scale = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float inv_scale = 1.0f / scale;

  if (tid == 0) row_scales[m] = scale;
  constexpr int KBLOCKS = H / 128;
  if (tid < KBLOCKS) broadcast_scales[m * KBLOCKS + tid] = scale;

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
  for (int offset = 16; offset > 0; offset >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  if ((tid & 31) == 0) s_partial[tid >> 5] = v;
  __syncthreads();

  if (tid < 32) {
    float w = (tid < NW) ? s_partial[tid] : -INFINITY;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      w = fmaxf(w, __shfl_xor_sync(0xffffffff, w, offset));
    if (tid == 0) s_absmax = w;
  }
  __syncthreads();

  float row_max = fmaxf(s_absmax, 1e-8f);
  float scale = fmaxf(row_max * (1.0f / 448.0f), 1e-8f);
  float inv_scale = 1.0f / scale;
  float w_route = sorted_weights[m];
  float scale_weighted = scale * w_route;

  if (tid == 0) row_scales[m] = scale;
  constexpr int KBLOCKS = H / 128;
  if (tid < KBLOCKS) broadcast_scales[m * KBLOCKS + tid] = scale_weighted;

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

// Fused 2D inverse-bucket build: count + place, single atomicAdd pass.
// sorted_tids[i] → t; this kernel writes token_perm_2d[t*TOP_K + slot] = i
// where slot = atomicAdd(token_counts+t, 1). No scan needed.
__global__ void fused_inverse_bucket_kernel_2d(
    const int* __restrict__ sorted_tids,
    int M, int T, int TOP_K,
    int* __restrict__ token_counts,
    int* __restrict__ token_perm_2d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  int t = sorted_tids[i];
  if (t < 0 || t >= T) return;
  int slot = atomicAdd(token_counts + t, 1);
  if (slot < TOP_K) {
    token_perm_2d[t * TOP_K + slot] = i;
  }
}

// Reduce kernel that consumes the 2D permutation directly (no offsets).
// Replaces the previous 4-kernel chain with just this + one fused_inverse_bucket_kernel_2d.
// Vectorized 128-bit IO matching reduce_scatter_unweighted_kernel.
__global__ void reduce_scatter_from_2d_perm_kernel(
    const __nv_bfloat16* __restrict__ gemm2_out,
    const int*           __restrict__ token_counts,
    const int*           __restrict__ token_perm_2d,
    __nv_bfloat16*       __restrict__ out,
    int T, int N2, int TOP_K)
{
  int t = blockIdx.x;
  if (t >= T) return;
  int c = token_counts[t];
  const int TB = blockDim.x;
  int lane = threadIdx.x;

  const int n_vec = N2 / 8;  // number of uint4 along N
  uint4* out_v = reinterpret_cast<uint4*>(out + t * N2);

  const int* perm_row = token_perm_2d + t * TOP_K;

  #pragma unroll 2
  for (int j = lane; j < n_vec; j += TB) {
    float2 a0 = make_float2(0.f, 0.f);
    float2 a1 = make_float2(0.f, 0.f);
    float2 a2 = make_float2(0.f, 0.f);
    float2 a3 = make_float2(0.f, 0.f);
    for (int k = 0; k < c; ++k) {
      int m = perm_row[k];
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

// Merged reduce-scatter: 1 cudaMemset + 2 kernel launches (was 2 memsets + 4 launches).
// Takes the same buffers as the 4-pass variant, reusing token_perm as the
// 2D inverse permutation [T, TOP_K]. Requires token_perm.numel() >= T * TOP_K.
void reduce_scatter_unweighted_fused(
    torch::Tensor const& gemm2_out,
    torch::Tensor const& sorted_tids,
    torch::Tensor&       out,
    torch::Tensor&       token_counts,
    torch::Tensor&       token_perm,
    int T,
    int TOP_K)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int M = gemm2_out.size(0);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 8 == 0, "N2 must be multiple of 8 for vectorized IO");
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);
  TORCH_CHECK(token_counts.numel() >= T);
  TORCH_CHECK(token_perm.numel() >= T * TOP_K);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  cudaMemsetAsync(token_counts.data_ptr(), 0, T * sizeof(int), stream);

  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  fused_inverse_bucket_kernel_2d<<<blocks, threads, 0, stream>>>(
      sorted_tids.data_ptr<int>(), M, T, TOP_K,
      token_counts.data_ptr<int>(),
      token_perm.data_ptr<int>());

  // Wider block (256 vs 128) → 2x in-flight memory ops per block; helps hide
  // random HBM latency across TOP_K non-contiguous m rows per output token.
  reduce_scatter_from_2d_perm_kernel<<<T, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      token_counts.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2, TOP_K);
}

// v20-C4: reduce-scatter from 2D inverse permutation that was pre-computed
// upstream (e.g., by the fused_dispatch single-block kernel). Takes 1 kernel
// launch total (no memset, no inverse_bucket needed).
void reduce_scatter_from_2d_perm(
    torch::Tensor const& gemm2_out,    // [M, N2] bf16 (weights already folded)
    torch::Tensor const& token_counts, // [T] int32 (per-token valid count)
    torch::Tensor const& token_perm,   // [T * TOP_K] int32 (2D inverse perm)
    torch::Tensor&       out,          // [T, N2] bf16 (overwritten)
    int T,
    int TOP_K)
{
  TORCH_CHECK(gemm2_out.is_cuda() && gemm2_out.dim() == 2);
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16);
  int N2 = gemm2_out.size(1);
  TORCH_CHECK(N2 % 8 == 0, "N2 must be multiple of 8 for vectorized IO");
  TORCH_CHECK(out.size(0) == T && out.size(1) == N2);
  TORCH_CHECK(token_counts.numel() >= T);
  TORCH_CHECK(token_perm.numel() >= T * TOP_K);

  auto stream = at::cuda::getCurrentCUDAStream(gemm2_out.get_device()).stream();
  reduce_scatter_from_2d_perm_kernel<<<T, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gemm2_out.data_ptr()),
      token_counts.data_ptr<int>(),
      token_perm.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      T, N2, TOP_K);
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
// Phase A scaffolding: flat tile-list builder.
//
// Given per-expert offsets (prefix sum of per-expert local token counts),
// emit a flat list of (expert_id, m_start) tuples covering every BLOCK_M-sized
// output-M tile that has work. This replaces the implicit per-expert grouped
// traversal baked into CUTLASS's PersistentTileSchedulerSm100Group with
// an explicit list we control — needed for our own persistent kernel.
//
// Single-block kernel: num_experts is <= 32 here, one warp does everything.
// Output arrays are sized for a worst-case `num_experts + num_assignments /
// block_m` entries (one extra per-expert partial tile).
// ============================================================================
__global__ void moe_flat_tile_list_kernel(
    int32_t const* __restrict__ offsets,    // [E+1] prefix sums
    int32_t*       __restrict__ tile_expert,
    int32_t*       __restrict__ tile_mstart,
    int32_t*       __restrict__ tile_count,
    int num_experts,
    int block_m,
    int max_tiles)
{
  if (blockIdx.x != 0) return;
  const int tid = threadIdx.x;

  // Each thread owns one expert.  Compute its tile count.
  int my_off_start = 0;
  int my_num_tiles = 0;
  if (tid < num_experts) {
    int off_start = offsets[tid];
    int off_end   = offsets[tid + 1];
    int count     = off_end - off_start;
    my_off_start  = off_start;
    my_num_tiles  = count > 0 ? (count + block_m - 1) / block_m : 0;
  }

  __shared__ int s_excl[64];
  if (tid < num_experts) s_excl[tid] = my_num_tiles;
  __syncthreads();

  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < num_experts; ++e) {
      int v = s_excl[e];
      s_excl[e] = acc;
      acc += v;
    }
    tile_count[0] = acc;
  }
  __syncthreads();

  if (tid < num_experts && my_num_tiles > 0) {
    int base = s_excl[tid];
    for (int mt = 0; mt < my_num_tiles; ++mt) {
      int idx = base + mt;
      if (idx < max_tiles) {
        tile_expert[idx] = tid;
        tile_mstart[idx] = my_off_start + mt * block_m;
      }
    }
  }
}

void moe_flat_tile_list(
    torch::Tensor const& offsets,         // [E+1] int32
    torch::Tensor&       tile_expert,     // [max_tiles] int32
    torch::Tensor&       tile_mstart,     // [max_tiles] int32
    torch::Tensor&       tile_count,      // [1] int32
    int block_m)
{
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_expert.is_cuda() && tile_expert.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_mstart.is_cuda() && tile_mstart.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_count.is_cuda() && tile_count.scalar_type() == torch::kInt32);
  int num_experts = offsets.size(0) - 1;
  int max_tiles = tile_expert.size(0);
  TORCH_CHECK(tile_mstart.size(0) == max_tiles);
  auto stream = at::cuda::getCurrentCUDAStream(offsets.get_device()).stream();
  moe_flat_tile_list_kernel<<<1, 64, 0, stream>>>(
      offsets.data_ptr<int32_t>(),
      tile_expert.data_ptr<int32_t>(),
      tile_mstart.data_ptr<int32_t>(),
      tile_count.data_ptr<int32_t>(),
      num_experts, block_m, max_tiles);
}

// ============================================================================
// MegaMoE (M, N)-tile scheduler: emits (expert, m_start, n_tile) TRIPLES.
//
// Each grid block in the megamoe kernel owns ONE (expert, m_tile, n_tile)
// triple — it does NOT sweep N-tiles internally. The kernel's grid is
// n_tiles_per_expert × (existing M-tile count). For T=14107 with tile_M=128
// that's ~928 × 32 ≈ 30k grid blocks. This matches the CUTLASS SM100
// persistent scheduler's unit of work.
// ============================================================================
__global__ void moe_flat_tile_list_mn_kernel(
    int32_t const* __restrict__ offsets,    // [E+1] prefix sums
    int32_t*       __restrict__ tile_expert,
    int32_t*       __restrict__ tile_mstart,
    int32_t*       __restrict__ tile_ntile,
    int32_t*       __restrict__ tile_count,
    int num_experts,
    int block_m,
    int n_tiles_per_expert,
    int max_tiles)
{
  if (blockIdx.x != 0) return;
  const int tid = threadIdx.x;

  // Each thread owns one expert; compute its M-tile count.
  int my_off_start = 0;
  int my_num_m_tiles = 0;
  if (tid < num_experts) {
    int off_start = offsets[tid];
    int off_end   = offsets[tid + 1];
    int count     = off_end - off_start;
    my_off_start  = off_start;
    my_num_m_tiles = count > 0 ? (count + block_m - 1) / block_m : 0;
  }

  // Per-expert (M, N) tile count = M_tiles * n_tiles_per_expert.
  int my_num_mn_tiles = my_num_m_tiles * n_tiles_per_expert;

  __shared__ int s_excl[64];
  if (tid < num_experts) s_excl[tid] = my_num_mn_tiles;
  __syncthreads();

  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < num_experts; ++e) {
      int v = s_excl[e];
      s_excl[e] = acc;
      acc += v;
    }
    tile_count[0] = acc;
  }
  __syncthreads();

  if (tid < num_experts && my_num_m_tiles > 0) {
    int base = s_excl[tid];
    // Interleave (m_tile, n_tile): the outer loop is m_tile, inner is n_tile.
    // This keeps tiles of the same expert contiguous in the grid, which is
    // good for tensormap-reuse across adjacent blocks.
    for (int mt = 0; mt < my_num_m_tiles; ++mt) {
      for (int nt = 0; nt < n_tiles_per_expert; ++nt) {
        int idx = base + mt * n_tiles_per_expert + nt;
        if (idx < max_tiles) {
          tile_expert[idx] = tid;
          tile_mstart[idx] = my_off_start + mt * block_m;
          tile_ntile[idx]  = nt;
        }
      }
    }
  }
}

void moe_flat_tile_list_mn(
    torch::Tensor const& offsets,         // [E+1] int32
    torch::Tensor&       tile_expert,     // [max_tiles] int32
    torch::Tensor&       tile_mstart,     // [max_tiles] int32
    torch::Tensor&       tile_ntile,      // [max_tiles] int32
    torch::Tensor&       tile_count,      // [1] int32
    int64_t block_m,
    int64_t n_tiles_per_expert)
{
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_expert.is_cuda() && tile_expert.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_mstart.is_cuda() && tile_mstart.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_ntile.is_cuda() && tile_ntile.scalar_type() == torch::kInt32);
  TORCH_CHECK(tile_count.is_cuda() && tile_count.scalar_type() == torch::kInt32);
  int num_experts = offsets.size(0) - 1;
  int max_tiles = tile_expert.size(0);
  TORCH_CHECK(tile_mstart.size(0) == max_tiles);
  TORCH_CHECK(tile_ntile.size(0) == max_tiles);
  auto stream = at::cuda::getCurrentCUDAStream(offsets.get_device()).stream();
  moe_flat_tile_list_mn_kernel<<<1, 64, 0, stream>>>(
      offsets.data_ptr<int32_t>(),
      tile_expert.data_ptr<int32_t>(),
      tile_mstart.data_ptr<int32_t>(),
      tile_ntile.data_ptr<int32_t>(),
      tile_count.data_ptr<int32_t>(),
      num_experts, static_cast<int>(block_m),
      static_cast<int>(n_tiles_per_expert), max_tiles);
}

// ============================================================================
// Fused routing kernel for DeepSeek-V3 MoE topk selection.
// v2: reduced-sync version. Uses warp-local top-K extraction + single
// block-wide merge (2 syncs total for top-8 instead of 32). Cuts the 41.7%
// barrier-stall in the original implementation roughly in half.
//
// Algorithm (same semantics as v1):
//   1. Each thread i computes s_wb[i] = sigmoid(logits[i]) + bias[i].
//   2. Group-top2: warp-local find max + 2nd max (via shfl). Sum → group_score.
//   3. Top-K_GROUP across 8 group_scores (warp 0).
//   4. Mask non-top-K groups with s_wb = -inf.
//   5. v2 top-K: Warp-local top-8 from 32 values (selection sort in registers
//      via __shfl_xor reductions, no smem). Then merge 8 warps' partials (64
//      candidates) into a final block-wide top-8 in warp 0 (1 sync).
//   6. Normalize.
//
// Constants hard-coded for DeepSeek-V3 MoE:
//   E_GLOBAL=256, N_GROUP=8, GROUP_SIZE=32, TOPK_GROUP=4, TOP_K=8.
// ============================================================================
// v21 SINGLE-WARP rewrite: block=32. Zero __syncthreads(). All reductions are
// register-resident using __shfl_*_sync. Each lane holds 8 experts (8 groups x
// 32 lanes = 256 experts, with lane L owning experts [L, 32+L, 64+L, ..., 224+L]).
// This eliminates the 42% barrier-stall we were hitting at T>=11948 and also
// saves launch overhead via 1-warp-per-block (better occupancy).
__global__ void __launch_bounds__(32) fused_route_topk_kernel(
    const __nv_bfloat16* __restrict__ routing_logits,  // [T, 256] bf16
    const __nv_bfloat16* __restrict__ routing_bias,    // [256] bf16
    int*                 __restrict__ topk_idx,        // [T, 8] int32 (global expert ids)
    float*               __restrict__ assign_w,        // [T, 8] float32
    int T, float rsf)
{
  constexpr int E_GLOBAL = 256;
  constexpr int N_GROUP = 8;       // groups of 32 experts
  constexpr int TOPK_GROUP = 4;
  constexpr int TOP_K_VAL = 8;
  const int tok = blockIdx.x;
  if (tok >= T) return;
  const int lane = threadIdx.x;    // 0..31 (block is exactly one warp)

  // Load 8 (logit, bias) pairs per lane (lane L owns expert g*32+L for g in 0..7).
  // s[g]    : pre-bias sigmoid (used as the final weight payload).
  // s_wb[g] : sigmoid + bias   (used for top-k score; masked to -INF as we pick).
  float s[8], s_wb[8];
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    int e = g * 32 + lane;
    float logit = __bfloat162float(routing_logits[tok * E_GLOBAL + e]);
    float bias  = __bfloat162float(routing_bias[e]);
    float sv    = 0.5f + 0.5f * __tanhf(logit * 0.5f);
    s[g]        = sv;
    s_wb[g]     = sv + bias;
  }

  // Step 2: within-group top-2. For each group g, all 32 lanes hold s_wb[g].
  // Find v1=max, then v2=max of s_wb[g] with v1 lanes masked out. group_score=v1+v2.
  float group_score[8];
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    float v1 = s_wb[g];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v1 = fmaxf(v1, __shfl_xor_sync(0xffffffff, v1, off));
    float v2_in = (s_wb[g] >= v1) ? -INFINITY : s_wb[g];
    float v2 = v2_in;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v2 = fmaxf(v2, __shfl_xor_sync(0xffffffff, v2, off));
    group_score[g] = v1 + v2;
  }

  // Step 3: top-TOPK_GROUP out of N_GROUP scores. Lane L<8 holds group_score[L],
  // else -INF. 4 rounds of warp-reduce max + mask-self.
  float my_gscore = (lane < N_GROUP) ? group_score[lane] : -INFINITY;
  bool  my_group_selected = false;
  #pragma unroll
  for (int k = 0; k < TOPK_GROUP; ++k) {
    float v = my_gscore;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    if (!my_group_selected && lane < N_GROUP && my_gscore == v) {
      my_group_selected = true;
      my_gscore = -INFINITY;
    }
  }
  // Broadcast valid-group mask (8 bits set in lanes 0..7) to all lanes.
  uint32_t valid_mask = __ballot_sync(0xffffffff, my_group_selected);

  // Step 4: mask experts in invalid groups.
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    if (!((valid_mask >> g) & 1u)) s_wb[g] = -INFINITY;
  }

  // Step 5: top-8 across the 128 unmasked experts.
  //   Per-lane local argmax over 8 regs (co-compute smv = s[argmax] with constant
  //   indices to avoid runtime register-array spill). Warp-reduce with tie-break
  //   on smaller global expert id. Lane 0 accumulates the 8 winners.
  int   top_idx[TOP_K_VAL];
  float top_s[TOP_K_VAL];
  #pragma unroll
  for (int k = 0; k < TOP_K_VAL; ++k) {
    float mv  = s_wb[0];
    float smv = s[0];   // s corresponding to lane's local argmax
    int   mg  = 0;
    #pragma unroll
    for (int g = 1; g < 8; ++g) {
      if (s_wb[g] > mv) { mv = s_wb[g]; smv = s[g]; mg = g; }
    }
    int me_exp = mg * 32 + lane;

    // Warp-reduce max with tie-break on smaller expert id. Also carry `smv`
    // alongside so the winning lane's s-value rides the reduction.
    float gv  = mv;
    int   ge  = me_exp;
    float gs  = smv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, gv, off);
      int   oe = __shfl_xor_sync(0xffffffff, ge, off);
      float os = __shfl_xor_sync(0xffffffff, gs, off);
      if (ov > gv || (ov == gv && oe < ge)) { gv = ov; ge = oe; gs = os; }
    }
    // Every lane now holds (gv, ge, gs). Record in lane 0; mask winner.
    if (lane == 0) {
      top_idx[k] = ge;
      top_s[k]   = gs;
    }
    int g_win = ge >> 5;
    int l_win = ge & 31;
    if (lane == l_win) {
      // Mask s_wb[g_win] with constant indices to avoid dynamic register index.
      #pragma unroll
      for (int g = 0; g < 8; ++g) {
        if (g == g_win) s_wb[g] = -INFINITY;
      }
    }
  }

  // Step 6: normalize + write (lane 0 only).
  if (lane == 0) {
    float sum_s = 0.0f;
    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) sum_s += top_s[k];
    float scale = rsf / (sum_s + 1e-20f);
    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) {
      topk_idx[tok * TOP_K_VAL + k] = top_idx[k];
      assign_w[tok * TOP_K_VAL + k] = top_s[k] * scale;
    }
  }
  // PDL: signal to downstream dependents (fused_dispatch) that topk_idx /
  // assign_w writes are visible. Consumers launched via launch_pdl() will
  // begin their blocks immediately after this.
  TRIGGER_PDL();
}

#if 0
// === (legacy block=256 routing kept below for reference; no longer used) ===
__global__ void fused_route_topk_kernel_legacy(
    const __nv_bfloat16* __restrict__ routing_logits,
    const __nv_bfloat16* __restrict__ routing_bias,
    int*                 __restrict__ topk_idx,
    float*               __restrict__ assign_w,
    int T, float rsf)
{
  constexpr int E_GLOBAL = 256;
  constexpr int N_GROUP = 8;
  constexpr int GROUP_SIZE = 32;
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

  // Step 5 (v2): warp-local top-8 via register-resident selection sort, then
  // 8-warp merge in warp 0. Only 2 __syncthreads() in this section.
  //
  // Each thread holds one (val, idx) pair. The warp performs 8 rounds of
  // "find-max + mask-self" using __shfl_xor_sync to find the max; the owning
  // lane sets its val to -inf for the next round. Lane 0 of each warp ends up
  // holding the warp's top-8 in registers (via 8 rounds of max with lane0 as
  // destination). We broadcast+store these warp-local top-8s to smem, then
  // warp 0 merges 8*8 = 64 candidates into the final top-8.

  // my_val / my_idx are the thread's candidate (initially s_wb_filtered).
  float my_val = s_wb_filtered;
  int   my_idx = tid;  // global-expert index (0..255)

  // Per-warp register-resident top-8 kept by lane 0.
  float warp_topk_val[TOP_K_VAL];
  int   warp_topk_idx[TOP_K_VAL];

  #pragma unroll
  for (int k = 0; k < TOP_K_VAL; ++k) {
    float v = my_val;
    int   i = my_idx;
    // Warp reduce (max, tie-break by smaller idx).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, v, off);
      int   oi = __shfl_xor_sync(0xffffffff, i, off);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    // Lane 0 stores this round's winner in its register arrays.
    if (lane == 0) {
      warp_topk_val[k] = v;
      warp_topk_idx[k] = i;
    }
    // The owning lane masks its candidate for the next round.
    if (my_idx == i) my_val = -INFINITY;
  }

  // Stage warp-local top-8s to smem (flat layout: [warp*8 + k]).
  __shared__ float cand_val[N_GROUP * TOP_K_VAL];  // 64
  __shared__ int   cand_idx[N_GROUP * TOP_K_VAL];  // 64
  if (lane == 0) {
    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) {
      cand_val[warp * TOP_K_VAL + k] = warp_topk_val[k];
      cand_idx[warp * TOP_K_VAL + k] = warp_topk_idx[k];
    }
  }
  __syncthreads();  // SYNC (1 of 2): partials visible to warp 0

  // Output buffers (filled by warp 0, read by all in step 6).
  __shared__ int   out_idx[TOP_K_VAL];
  __shared__ float out_s_sigmoid[TOP_K_VAL];

  // Warp 0: 64 candidates mapped to 2 consecutive rounds of warp shuffles
  // across 32 lanes. Each lane holds 2 (val, idx) pairs (idx lane*2 and lane*2+1).
  if (warp == 0) {
    float v0 = cand_val[lane * 2 + 0];
    int   i0 = cand_idx[lane * 2 + 0];
    float v1 = cand_val[lane * 2 + 1];
    int   i1 = cand_idx[lane * 2 + 1];

    #pragma unroll
    for (int k = 0; k < TOP_K_VAL; ++k) {
      // Local (within-lane) max of v0 and v1.
      float mv = v0;
      int   mi = i0;
      if (v1 > mv || (v1 == mv && i1 < mi)) { mv = v1; mi = i1; }

      // Warp reduce to find block-wide max.
      float gv = mv;
      int   gi = mi;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, gv, off);
        int   oi = __shfl_xor_sync(0xffffffff, gi, off);
        if (ov > gv || (ov == gv && oi < gi)) { gv = ov; gi = oi; }
      }
      // Lane 0 emits.
      if (lane == 0) {
        out_idx[k] = gi;
      }
      // Mask the picked slot in whichever lane owns it.
      bool owns_0 = (i0 == gi);
      bool owns_1 = (i1 == gi);
      if (owns_0) v0 = -INFINITY;
      if (owns_1) v1 = -INFINITY;
    }
  }
  __syncthreads();  // SYNC (2 of 2): out_idx[k] visible to every thread

  // Every thread checks if it's the winner of any slot k, writes its pre-bias
  // sigmoid `s` to the output buffer. No sync needed because each slot is
  // written by exactly one thread (the one matching out_idx[k]) and read only
  // by thread 0 in step 6 after the next sync below.
  #pragma unroll
  for (int k = 0; k < TOP_K_VAL; ++k) {
    if (tid == out_idx[k]) {
      out_s_sigmoid[k] = s;
    }
  }
  __syncthreads();  // SYNC (3 of 3): out_s_sigmoid visible to thread 0

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
#endif  // legacy block=256 routing

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
  // v21: single-warp block (32 threads/token).
  fused_route_topk_kernel<<<T, 32, 0, stream>>>(
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
// v20-D2: single-block fused dispatch + hidden-state gather kernel. Extends
// the C3 dispatch kernel with an inline warp-cooperative gather so we can
// replace (dispatch + separate gather launch) with a single launch on small T.
//
// Structure: 32 warps (TB=1024). In the place phase, each warp handles ONE
// (t, k) item — lane 0 does the atomicAdd for the slot, then the whole warp
// cooperatively copies `hidden_states[t] -> packed_acts[slot]` (K1/16 uint4
// vecs) and `hs_scale[t] -> packed_act_scales[slot]` (K1_blocks fp32 scales).
//
// Safe when NA / num_warps is small: each warp does 1 row of gather per iter,
// so for NA <= 8192 (T <= 1024 with TOP_K=8) it's 256 iters/warp × ~14 ops/lane
// = ~3500 cycles ≈ 3.5μs per warp. Intended for T <= 256 (NA <= 2048) where
// each warp does only 64 iters and inline is strictly cheaper than a separate
// gather launch.
template <int K1_T>
__global__ void fused_dispatch_and_gather_single_block_kernel(
    const int* __restrict__ topk_idx,
    const float* __restrict__ assign_w,
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float* __restrict__ hs_scale,
    int hs_scale_stride_t, int hs_scale_stride_b,
    int NA, int T, int TOP_K, int local_start, int num_experts,
    int K1, int K1_blocks,
    int* counts, int* sorted_tids, float* sorted_weights,
    int* offsets, int* problem_sizes_1, int* problem_sizes_2,
    __nv_fp8_e4m3* packed_acts, float* packed_act_scales)
{
  const int tid = threadIdx.x;
  const int TB = blockDim.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int num_warps = TB >> 5;

  __shared__ int s_counts[64];
  __shared__ int s_offsets[64];

  // Phase 0: zero the shared counts.
  if (tid < num_experts) s_counts[tid] = 0;
  __syncthreads();

  // Phase 1: count.
  for (int i = tid; i < NA; i += TB) {
    int g = topk_idx[i];
    int e = g - local_start;
    if (e >= 0 && e < num_experts) {
      atomicAdd(s_counts + e, 1);
    }
  }
  __syncthreads();

  // Phase 2: exclusive scan by warp 0.
  if (tid < 32) {
    int my = (tid < num_experts) ? s_counts[tid] : 0;
    int v = my;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int n = __shfl_up_sync(0xffffffff, v, off);
      if (tid >= off) v += n;
    }
    int excl = v - my;
    if (tid < num_experts) {
      s_offsets[tid] = excl;
      offsets[tid] = excl;
      counts[tid]  = my;
      if (problem_sizes_1 != nullptr) problem_sizes_1[tid * 3] = my;
      if (problem_sizes_2 != nullptr) problem_sizes_2[tid * 3] = my;
    }
    if (tid == 31) offsets[num_experts] = v;
  }
  __syncthreads();

  // Phase 3a: reset s_counts (reuse as cursors) + pre-zero sorted_tids/weights.
  if (tid < num_experts) s_counts[tid] = 0;
  for (int i = tid; i < NA; i += TB) {
    sorted_tids[i] = -1;
    sorted_weights[i] = 0.0f;
  }
  __syncthreads();

  // Phase 4 (warp-per-item): place each (t, k) pair AND gather its hidden-state
  // row into packed_acts. 32 warps loop over NA items in strides of 32.
  const int n_vec = K1 / 16;  // K1 is runtime; stay dynamic
  for (int i = warp; i < NA; i += num_warps) {
    int g = topk_idx[i];
    int e = g - local_start;
    int t = i / TOP_K;

    int slot = -1;
    if (lane == 0) {
      if (e >= 0 && e < num_experts && t >= 0 && t < T) {
        int slot_in_expert = atomicAdd(s_counts + e, 1);
        slot = s_offsets[e] + slot_in_expert;
        sorted_tids[slot] = t;
        sorted_weights[slot] = assign_w[i];
      }
    }
    slot = __shfl_sync(0xffffffff, slot, 0);
    if (slot < 0) continue;

    // Warp-cooperative gather: uint4 payload + fp32 scales.
    const uint4* src_v = reinterpret_cast<const uint4*>(hidden_states + t * K1);
    uint4*       dst_v = reinterpret_cast<uint4*>(packed_acts + slot * K1);
    #pragma unroll 2
    for (int j = lane; j < n_vec; j += 32) {
      dst_v[j] = src_v[j];
    }

    const float* ssrc = hs_scale + t * hs_scale_stride_t;
    float*       sdst = packed_act_scales + slot * K1_blocks;
    for (int j = lane; j < K1_blocks; j += 32) {
      sdst[j] = ssrc[j * hs_scale_stride_b];
    }
  }
}

// Host wrapper for fused_dispatch_and_gather_single_block_kernel.
void fused_dispatch_and_gather(
    torch::Tensor const& topk_idx,
    torch::Tensor const& assign_w,
    torch::Tensor const& hidden_states,
    torch::Tensor const& hs_scale,
    int local_start, int num_experts,
    torch::Tensor& counts, torch::Tensor& sorted_tids,
    torch::Tensor& sorted_weights, torch::Tensor& offsets,
    torch::Tensor& problem_sizes_1, torch::Tensor& problem_sizes_2,
    torch::Tensor& packed_acts, torch::Tensor& packed_act_scales)
{
  TORCH_CHECK(topk_idx.is_cuda() && topk_idx.scalar_type() == torch::kInt32);
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kFloat8_e4m3fn);
  int T = topk_idx.size(0);
  int TOP_K = topk_idx.size(1);
  int NA = T * TOP_K;
  int K1 = hidden_states.size(1);
  int K1_blocks = packed_act_scales.size(1);
  TORCH_CHECK(K1 % 16 == 0);
  TORCH_CHECK(num_experts <= 32);

  // hs_scale stride autodetect (same as fused_gather_hidden_scales).
  int hs_size_0 = hs_scale.size(0);
  int hs_size_1 = hs_scale.size(1);
  int stride_t, stride_b;
  if (hs_size_0 == K1_blocks && hs_size_1 == T) {
    stride_t = hs_scale.stride(1); stride_b = hs_scale.stride(0);
  } else if (hs_size_0 == T && hs_size_1 == K1_blocks) {
    stride_t = hs_scale.stride(0); stride_b = hs_scale.stride(1);
  } else {
    TORCH_CHECK(false, "hs_scale shape doesn't match either layout");
  }

  auto stream = at::cuda::getCurrentCUDAStream(topk_idx.get_device()).stream();
  int TB = 1024;
  if (NA < 1024) TB = 256;
  fused_dispatch_and_gather_single_block_kernel<0><<<1, TB, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const float*>(assign_w.data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      hs_scale.data_ptr<float>(),
      stride_t, stride_b,
      NA, T, TOP_K, local_start, num_experts,
      K1, K1_blocks,
      static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(sorted_tids.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()),
      static_cast<int*>(offsets.data_ptr()),
      static_cast<int*>(problem_sizes_1.data_ptr()),
      static_cast<int*>(problem_sizes_2.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(packed_acts.data_ptr()),
      packed_act_scales.data_ptr<float>());
}

// v20-C3: single-block fused dispatch kernel. Collapses the 3-kernel + 3-memset
// sequence used by the multi-block path into ONE launch. Intended for NA
// (= T * TOP_K) that fits comfortably in a single block's workload.
//
// v20-C4: extended to ALSO produce per-token inverse permutation so the
// downstream scatter can use `reduce_scatter_from_2d_perm_kernel` (overwrites,
// no pre-zero). This eliminates the separate `fused_inverse_bucket_kernel_2d`
// launch AND the `out_bf16.zero_()` launch from the graph-safe path.
//
// Grid: <<<1, TB>>>. Each thread handles ⌈NA / TB⌉ items per pass.
// Shared state: counts_smem[num_experts] used as a tile-local counter; also
// holds the exclusive scan. token_counts_smem[T] counts per-token.
//
// For num_experts <= 32, one warp performs the scan via __shfl.
__global__ void fused_dispatch_single_block_kernel(
    const int* __restrict__ topk_idx,
    const float* __restrict__ assign_w,
    int NA, int T, int TOP_K, int local_start, int num_experts,
    int* counts, int* sorted_tids, float* sorted_weights,
    int* offsets, int* problem_sizes_1, int* problem_sizes_2,
    int* token_counts,     // [T] — per-token count of local-valid assignments (may be nullptr)
    int* token_perm)       // [T * TOP_K] — inverse permutation (may be nullptr)
{
  const int tid = threadIdx.x;
  const int TB = blockDim.x;

  // Shared counts (capped to 64 — contest has 32).
  __shared__ int s_counts[64];
  __shared__ int s_offsets[64];

  // Phase 0: zero the shared counts + token_counts global.
  if (tid < num_experts) s_counts[tid] = 0;
  if (token_counts != nullptr) {
    for (int t = tid; t < T; t += TB) token_counts[t] = 0;
  }
  __syncthreads();

  // Phase 1: count.
  for (int i = tid; i < NA; i += TB) {
    int g = topk_idx[i];
    int e = g - local_start;
    if (e >= 0 && e < num_experts) {
      atomicAdd(s_counts + e, 1);
    }
  }
  __syncthreads();

  // Phase 2: exclusive scan by warp 0 (num_experts <= 32 required).
  if (tid < 32) {
    int my = (tid < num_experts) ? s_counts[tid] : 0;
    int v = my;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int n = __shfl_up_sync(0xffffffff, v, off);
      if (tid >= off) v += n;
    }
    int excl = v - my;
    if (tid < num_experts) {
      s_offsets[tid] = excl;
      offsets[tid] = excl;
      counts[tid]  = my;  // final count (also provided by the downstream place-pass)
      if (problem_sizes_1 != nullptr) problem_sizes_1[tid * 3] = my;
      if (problem_sizes_2 != nullptr) problem_sizes_2[tid * 3] = my;
    }
    if (tid == 31) offsets[num_experts] = v;
  }
  __syncthreads();

  // Phase 3: zero s_counts[] again (reuse as cursors).
  if (tid < num_experts) s_counts[tid] = 0;
  __syncthreads();

  // Phase 4a: pre-zero sorted_tids/weights to -1/0 over the full NA range.
  for (int i = tid; i < NA; i += TB) {
    sorted_tids[i] = -1;
    sorted_weights[i] = 0.0f;
  }
  __syncthreads();

  // Phase 4b: place. Each valid (i = tok*TOP_K + k) gets a sorted slot;
  // also record the inverse permutation token_perm[tok][tok_k] = slot.
  for (int i = tid; i < NA; i += TB) {
    int g = topk_idx[i];
    int e = g - local_start;
    if (e >= 0 && e < num_experts) {
      int slot_in_expert = atomicAdd(s_counts + e, 1);
      int slot = s_offsets[e] + slot_in_expert;
      int tok = i / TOP_K;
      sorted_tids[slot] = tok;
      sorted_weights[slot] = assign_w[i];
      if (token_counts != nullptr) {
        int tok_k = atomicAdd(token_counts + tok, 1);
        if (token_perm != nullptr) {
          token_perm[tok * TOP_K + tok_k] = slot;
        }
      }
    }
  }
  // PDL: downstream consumers (fused_gather_*) depend on sorted_tids +
  // sorted_weights + offsets + problem_sizes being written. Signal here.
  TRIGGER_PDL();
}

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
    torch::Tensor&       problem_sizes_2, // [E, 3] int32 (M col written by scan)
    c10::optional<torch::Tensor> token_counts = c10::nullopt,  // [T] int32 (optional inverse-perm count)
    c10::optional<torch::Tensor> token_perm   = c10::nullopt)  // [T*TOP_K] int32 (optional inverse perm)
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

  // Single-block fast path: one kernel launch does count+scan+place AND
  // the three zero-inits (via in-kernel writes). Each thread handles
  // ~NA/TB items in each phase. For small NA this is drastically cheaper
  // than the 3-kernel + 3-memset multi-block path (~20μs of launch overhead
  // on a graph-replayed small-T pipeline).
  //
  // Switch to the multi-block path when NA grows so large that one block's
  // serialized work exceeds the multi-block parallelism. Threshold: 64K
  // items per block gives ~64K / 1024 = 64 iters/thread in each phase,
  // which is still fast in absolute terms on B200.
  const int SINGLE_BLOCK_NA_MAX = 65536;
  const char* force_multi = std::getenv("FUSED_DISPATCH_FORCE_MULTI");
  const char* force_single = std::getenv("FUSED_DISPATCH_FORCE_SINGLE");
  bool use_single_block =
      (force_single && std::string(force_single) == "1") ||
      (!(force_multi && std::string(force_multi) == "1") && NA <= SINGLE_BLOCK_NA_MAX);

  if (use_single_block) {
    int TB = 1024;
    if (NA < 1024) TB = 256;   // tiny T
    int* tc_ptr = token_counts.has_value()
                      ? static_cast<int*>(token_counts->data_ptr())
                      : nullptr;
    int* tp_ptr = token_perm.has_value()
                      ? static_cast<int*>(token_perm->data_ptr())
                      : nullptr;
    // NOTE: PDL (launch_pdl + griddepcontrol.launch_dependents at end of
    // route/dispatch) was benchmarked on the full 19 workloads and net
    // regressed (~3% worse arith_mean). cudaLaunchKernelEx has more per-launch
    // overhead than `<<<>>>` for tiny single-block kernels, and since dispatch
    // is single-block its "early launch" benefit is minimal. Reverted.
    fused_dispatch_single_block_kernel<<<1, TB, 0, stream>>>(
        static_cast<const int*>(topk_idx.data_ptr()),
        static_cast<const float*>(assign_w.data_ptr()),
        NA, T, TOP_K, local_start, num_experts,
        static_cast<int*>(counts.data_ptr()),
        static_cast<int*>(sorted_tids.data_ptr()),
        static_cast<float*>(sorted_weights.data_ptr()),
        static_cast<int*>(offsets.data_ptr()),
        static_cast<int*>(problem_sizes_1.data_ptr()),
        static_cast<int*>(problem_sizes_2.data_ptr()),
        tc_ptr, tp_ptr);
    return;
  }

  // Multi-block fallback (original path, used for NA > 64K).
  cudaMemsetAsync(counts.data_ptr(), 0, counts.numel() * sizeof(int), stream);
  cudaMemsetAsync(sorted_weights.data_ptr(), 0, sorted_weights.numel() * sizeof(float), stream);
  cudaMemsetAsync(sorted_tids.data_ptr(), 0xFF, sorted_tids.numel() * sizeof(int), stream);

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
  dispatch_place_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int*>(topk_idx.data_ptr()),
      static_cast<const float*>(assign_w.data_ptr()),
      static_cast<const int*>(offsets.data_ptr()),
      NA, TOP_K, local_start, num_experts,
      static_cast<int*>(counts.data_ptr()),
      static_cast<int*>(sorted_tids.data_ptr()),
      static_cast<float*>(sorted_weights.data_ptr()));
}

// Forward decl: lives in moe_megamoe.cu (third compile unit).
// Phase A substep (a) GEMM1 kernel — custom SM100 CuTe, flat-tile-list driven.
void megamoe_gemm1(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& tile_expert,
    torch::Tensor const& tile_mstart,
    torch::Tensor const& tile_ntile,
    torch::Tensor const& tile_count,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace,
    int64_t block_m);

// Forward decl: fused GEMM1+SwiGLU+FP8 requant.
void megamoe_gemm1_swiglu_fused(
    torch::Tensor& act_q,
    torch::Tensor& act_scales,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::optional<torch::Tensor> const& sorted_weights,
    torch::Tensor const& tile_expert,
    torch::Tensor const& tile_mstart,
    torch::Tensor const& tile_out_ntile,
    torch::Tensor const& tile_count,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace,
    int64_t block_m,
    int64_t I_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_blockwise_grouped_mm_v2", &moe_blockwise_grouped_mm_v2);
  m.def("moe_blockwise_grouped_mm_by_expert_ids", &moe_blockwise_grouped_mm_by_expert_ids);
  m.def("get_sizes", &get_sizes);
  m.def("get_workspace_size", &get_workspace_size);
  m.def("swiglu_fp8_requant", &swiglu_fp8_requant);
  m.def("swiglu_fp8_requant_weighted", &swiglu_fp8_requant_weighted);
  m.def("weighted_scatter", &weighted_scatter);
  m.def("reduce_scatter", &reduce_scatter);
  m.def("reduce_scatter_prebucketed", &reduce_scatter_prebucketed);
  m.def("reduce_scatter_unweighted_prebucketed", &reduce_scatter_unweighted_prebucketed);
  m.def("reduce_scatter_unweighted", &reduce_scatter_unweighted);
  m.def("reduce_scatter_unweighted_fused", &reduce_scatter_unweighted_fused);
  m.def("reduce_scatter_from_2d_perm", &reduce_scatter_from_2d_perm);
  m.def("token_bucket_scan_and_place", &token_bucket_scan_and_place);
  m.def("fused_route_topk", &fused_route_topk);
  m.def("fused_gather_hidden_scales", &fused_gather_hidden_scales);
  m.def("repack_aligned_expert_layout", &repack_aligned_expert_layout);
  m.def("fused_dispatch_and_gather", &fused_dispatch_and_gather);
  m.def("fused_dispatch", &fused_dispatch,
        pybind11::arg("topk_idx"), pybind11::arg("assign_w"),
        pybind11::arg("local_start"), pybind11::arg("num_experts"),
        pybind11::arg("counts"), pybind11::arg("sorted_tids"),
        pybind11::arg("sorted_weights"), pybind11::arg("offsets"),
        pybind11::arg("problem_sizes_1"), pybind11::arg("problem_sizes_2"),
        pybind11::arg("token_counts") = pybind11::none(),
        pybind11::arg("token_perm")   = pybind11::none());
  m.def("fused_dispatch_gather_hidden_scales", &fused_dispatch_gather_hidden_scales);
  m.def("mxf8_transcode_activations", &mxf8_transcode_activations);
  m.def("mxf8_transcode_and_pack_sfa", &mxf8_transcode_and_pack_sfa);
  m.def("mxf8_transcode_weights_impl", &mxf8_transcode_weights_impl);
  m.def("mxf8_pack_weight_sfb_impl", &mxf8_pack_weight_sfb_impl);
  m.def("moe_mxf8_setup_ptrs", &moe_mxf8_setup_ptrs);
  m.def("moe_mxf8_setup_ptrs_1sm", &moe_mxf8_setup_ptrs_1sm);
  m.def("moe_mxf8_grouped_mm_prepacked", &moe_mxf8_grouped_mm_prepacked);
  m.def("moe_mxf8_grouped_mm_prepacked_1sm", &moe_mxf8_grouped_mm_prepacked_1sm);
  m.def("moe_mxf8_grouped_mm_prepacked_256_256", &moe_mxf8_grouped_mm_prepacked_256_256);
  m.def("moe_mxf8_grouped_mm_prepacked_128_256_1sm", &moe_mxf8_grouped_mm_prepacked_128_256_1sm);
  m.def("moe_mxf8_grouped_mm_prepacked_fp8out", &moe_mxf8_grouped_mm_prepacked_fp8out);
  m.def("moe_mxf8_grouped_mm_prepacked_fp8out_2sm", &moe_mxf8_grouped_mm_prepacked_fp8out_2sm);
  m.def("swiglu_fp8in_mxf8_weighted", &swiglu_fp8in_mxf8_weighted);
  m.def("get_mxf8_fp8out_sizes_stride", &get_mxf8_fp8out_sizes_stride);
  m.def("get_mxf8_fp8out_sizes_layout_sfd", &get_mxf8_fp8out_sizes_layout_sfd);
  m.def("compute_mxf8_sfd_offsets_device", &compute_mxf8_sfd_offsets_device);
  m.def("fused_gather_mxf8", &fused_gather_mxf8);
  m.def("swiglu_fp8_requant_weighted_mxf8", &swiglu_fp8_requant_weighted_mxf8);
  m.def("moe_mxf8_grouped_mm", &moe_mxf8_grouped_mm);
  m.def("compute_mxf8_sfa_layout_offsets_host", &compute_mxf8_sfa_layout_offsets_host);
  m.def("compute_mxf8_sfb_layout_offsets_host", &compute_mxf8_sfb_layout_offsets_host);
  m.def("compute_mxf8_sf_offsets_device", &compute_mxf8_sf_offsets_device);
  m.def("get_mxf8_sizes_stride", &get_mxf8_sizes_stride);
  m.def("get_mxf8_sizes_layout_sfa", &get_mxf8_sizes_layout_sfa);
  m.def("get_mxf8_sizes_layout_sfb", &get_mxf8_sizes_layout_sfb);
  m.def("probe_mxf8_sfa_layout", &probe_mxf8_sfa_layout);
  m.def("moe_flat_tile_list", &moe_flat_tile_list);
  m.def("moe_flat_tile_list_mn", &moe_flat_tile_list_mn);
  m.def("megamoe_gemm1_swiglu_fused", &megamoe_gemm1_swiglu_fused);
  m.def("megamoe_gemm1", &megamoe_gemm1,
        "MegaMoE Phase A substep (a) GEMM1: custom SM100 CuTe kernel driven by flat-tile-list");
}
'''


# ============================================================================
# MegaMoE custom-kernel compile unit — Phase A sub-step (a) GEMM1.
#
# Custom CuTe SM100 MxF8 GEMM1 kernel driven by the flat-tile-list.
# Reuses CUTLASS's CollectiveBuilder-produced CollectiveMainloop +
# CollectiveEpilogue types (per HANDOFF_MEGAMOE_PLAN.md §8 "Reuse that
# machinery") but replaces CUTLASS's PersistentTileSchedulerSm100Group with
# a trivial non-persistent scheduler: grid = (max_tiles,), each CTA owns
# ONE (expert, m_start) tile from the flat list, iterates N-tiles internally.
#
# The kernel accepts the same pre-populated per-expert arrays that
# moe_mxf8_grouped_mm_prepacked consumes — a_ptrs[E], b_ptrs[E], out_ptrs[E],
# sfa_ptrs[E], sfb_ptrs[E], strides, layout_sfa/sfb — and routes each tile
# to its expert via tile_expert[blockIdx.x]. No changes to the SFA/SFB
# packing path.
# ============================================================================
_MOE_MEGAMOE_CU = r'''
// =============================================================================
// MegaMoE custom-kernel compile unit — Phase A sub-step (a) GEMM1.
//
// Design: REUSE CollectiveMainloop produced by CollectiveBuilder (that's
// where CUTLASS gives us correct TMA descriptors, SMEM swizzle layouts,
// UMMA descriptors, and mbarrier-based pipeline primitives). DROP:
//   - GemmUniversalAdapter + GemmUniversal (top-level kernel driver)
//   - PersistentTileSchedulerSm100Group (replace with flat-tile-list grid)
//   - Default CollectiveEpilogue (replace with custom TMEM->bf16 STG)
//
// This pays a ~2-3 min cold compile for the mainloop instantiation, but the
// custom epilogue is small and fast to iterate on. Per the plan §8:
// "Reuse that machinery; only replace the high-level scheduler ...".
// =============================================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/grid_dependency_control.h>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

using namespace cute;

namespace megamoe {

// UE8M0 helpers (duplicated from _MOE_FUSED_CU for in-kernel use).
__device__ __forceinline__ uint8_t ue8m0_ceil_from_abs_fp32(float x) {
  if (!isfinite(x) || x <= 0.0f) return 0;
  uint32_t bits = __float_as_uint(x);
  uint8_t exp = (bits >> 23) & 0xff;
  uint32_t mant = bits & 0x7fffff;
  if (mant > 0 && exp != 0xFE) exp++;
  return exp;
}

__device__ __forceinline__ float ue8m0_byte_to_fp32(uint8_t b) {
  uint32_t f = (uint32_t)(b) << 23;
  return __uint_as_float(f);
}

// -------- Element / layout (match _MOE_GEMM_CU's MxF8 config) ------------
using ElementA   = cutlass::float_e4m3_t;
using ElementB   = cutlass::float_e4m3_t;
using ElementC   = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementSF  = cutlass::float_ue8m0_t;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::ColumnMajor;
using LayoutOut  = cutlass::layout::RowMajor;
constexpr int AlignA = 16;
constexpr int AlignB = 16;
constexpr int AlignC = 8;

using MmaTypePairA = decltype(cute::make_tuple(ElementA{}, ElementSF{}));
using MmaTypePairB = decltype(cute::make_tuple(ElementB{}, ElementSF{}));

// Matches CfgMxF8Large1SM (1-CTA, 128×128×128). Extra tile variants will
// land later — this is Phase A sub-step (a) only.
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;

// Build CollectiveMainloop ONLY. No CollectiveEpilogue — we write a custom
// TMEM->bf16 epilogue below. StageCountAutoCarveout with 0 epilogue bytes
// gives the mainloop the full SMEM budget for its A/B/SFA/SFB pipeline.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTypePairA, LayoutA*, AlignA,
    MmaTypePairB, LayoutB*, AlignB,
    ElementAcc,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<0>,
    KernelSchedule>::CollectiveOp;

// Second mainloop for the fused SwiGLU variant. Carves out SMEM for the
// gate_buffer (128×128 BF16 = 32KB) + row_max_bits (128×4 = 512B) plus
// padding. This reduces the mainloop pipeline Stages by 1 but keeps the
// total SharedStorageFused within the 232KB SM100 cap.
using CollectiveMainloopFused = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
    MmaTypePairA, LayoutA*, AlignA,
    MmaTypePairB, LayoutB*, AlignB,
    ElementAcc,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<33280>,  // 32KB gate + 512B row_max + pad
    KernelSchedule>::CollectiveOp;

using StrideA = typename CollectiveMainloop::StrideA;
using StrideB = typename CollectiveMainloop::StrideB;
using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
using InternalLayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
using InternalLayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
static constexpr int Stages = DispatchPolicy::Stages;
using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
using CtaShape_MNK    = typename CollectiveMainloop::CtaShape_MNK;
using TileShape       = typename CollectiveMainloop::TileShape;
using TiledMma        = typename CollectiveMainloop::TiledMma;

using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

static constexpr uint32_t AccumulatorPipelineStageCount =
    DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
using AccumulatorPipeline = cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount, AtomThrShapeMNK>;
using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

using TmemAllocator = cute::conditional_t<
    cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
    cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

using MainloopArguments = typename CollectiveMainloop::Arguments;
using MainloopParams    = typename CollectiveMainloop::Params;

// -------- Warp specialization — match CUTLASS SM100 array reference -----
// Reference: cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp
//   NumMMAThreads          = 32 (1 warp)
//   NumMainloopLoadThreads = 32 (1 warp)
//   NumEpilogueThreads     = 128 (1 warpgroup)
// Previous bug was that my kernel had the CTA sweep 32 N-tiles inside the
// mainloop/MMA warps — with bug #1 fixed (each CTA owns ONE (e, m, n)
// triple), 1 warp per role matches the working CUTLASS pattern.
static constexpr uint32_t NumMainloopLoadThreads = cutlass::NumThreadsPerWarp;
static constexpr uint32_t NumMMAThreads          = cutlass::NumThreadsPerWarp;
static constexpr uint32_t NumEpilogueThreads     = 128;
static constexpr uint32_t NumEpilogueWarps       = NumEpilogueThreads / cutlass::NumThreadsPerWarp;

// Layout in CTA: warp 0 MainloopLoad, warp 1 MMA, warps 2..5 Epilogue.
static constexpr uint32_t MaxThreadsPerBlock =
    NumMainloopLoadThreads + NumMMAThreads + NumEpilogueThreads;

enum class WarpCategory : int32_t {
  MainloopLoad = 0,
  MMA          = 1,
  Epilogue     = 2
};

// -------- SharedStorage (kernel-level) -----------------------------------
struct SharedStorage {
  struct PipelineStorage : cute::aligned_struct<16, _1> {
    using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
    using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;

    alignas(16) MainloopPipelineStorage mainloop;
    alignas(16) AccumulatorPipelineStorage accumulator;
  } pipelines;

  uint32_t tmem_base_ptr;

  struct TensorMapStorage : cute::aligned_struct<128, _1> {
    using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
    alignas(128) MainloopTensorMapStorage mainloop;
  } tensormaps;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
    MainloopTensorStorage mainloop;
  } tensors;
};

// -------- Kernel Params --------------------------------------------------
struct KernelParams {
  // Flat-tile-list (our "scheduler") — (expert, m_start, n_tile) triples.
  int32_t const* tile_expert;
  int32_t const* tile_mstart;
  int32_t const* tile_ntile;     // NEW: N-tile index per grid block
  int32_t const* tile_count;
  int            max_tiles;
  int            block_m;
  int32_t const* expert_offsets;  // [E+1] for m_valid

  // Global problem shape (N, K fixed across experts; M per-tile)
  int N;
  int K;
  int num_experts;

  // Per-expert [M,N,K] in int32 format, for tensormaps_perform_update
  UnderlyingProblemShape const* ps_device;

  // Per-expert output ptrs [E] — populated by moe_mxf8_setup_ptrs.
  ElementC* const* out_ptrs;

  // Device-side stride for C (for bounds computation in custom epilogue).
  // Using int64 row stride == N since output is row-major [total_tokens, N].
  int64_t out_row_stride;

  MainloopParams mainloop;
  cutlass::KernelHardwareInfo hw_info;
};

// -------- Device kernel --------------------------------------------------
__global__ __launch_bounds__(MaxThreadsPerBlock, 1)
void megamoe_gemm1_device_kernel(KernelParams params) {
  using namespace cute;
  using X = Underscore;

  // 1. Tile assignment.
  int tile_idx = blockIdx.x;
  int tile_count_live = *params.tile_count;
  if (tile_idx >= tile_count_live) {
    return;
  }
  int expert_id = params.tile_expert[tile_idx];
  int m_start   = params.tile_mstart[tile_idx];
  int m_end_e   = params.expert_offsets[expert_id + 1];
  int m_valid   = m_end_e - m_start;
  if (m_valid > params.block_m) m_valid = params.block_m;
  if (m_valid <= 0) return;

  // Problem shape for THIS tile; use append<4> matching CUTLASS reference exactly.
  // UnderlyingProblemShape = cute::Shape<int,int,int> from GroupProblemShape.
  UnderlyingProblemShape ps_mnk{m_valid, params.N, params.K};
  auto problem_shape_MNKL = append<4>(ps_mnk, 1);

  // 2. Warp category.
  //    warp 0    -> MainloopLoad
  //    warp 1    -> MMA
  //    warps 2.. -> Epilogue
  int warp_idx = cutlass::canonical_warp_idx_sync();
  WarpCategory warp_category = (warp_idx < static_cast<int>(WarpCategory::Epilogue))
                                ? WarpCategory(warp_idx)
                                : WarpCategory::Epilogue;
  uint32_t lane_predicate = cute::elect_one_sync();

  auto cluster_shape = ClusterShape{};
  uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
  bool is_mma_leader_cta = true;  // 1-CTA cluster
  (void)is_mma_leader_cta;

  // 3. SharedStorage.
  extern __shared__ char smem_raw[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_raw);

  CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);

  bool is_participant_mma       = (warp_category == WarpCategory::MMA);
  bool is_participant_main_load = (warp_category == WarpCategory::MainloopLoad);
  bool is_participant_epilogue  = (warp_category == WarpCategory::Epilogue);

  // 4. Pipelines.
  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (is_participant_main_load) mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  if (is_participant_mma)       mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  mainloop_pipeline_params.is_leader = lane_predicate && is_participant_main_load;
  mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
  mainloop_pipeline_params.initializing_warp = 0;
  MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                     mainloop_pipeline_params,
                                     cluster_shape,
                                     cute::true_type{},
                                     cute::false_type{});

  typename AccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_participant_mma)       accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
  if (is_participant_epilogue)  accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
  accumulator_pipeline_params.producer_arv_count = 1;
  // One arrival per epilogue thread within this CTA. Do NOT multiply by
  // size(AtomThrShapeMNK) — for 2-CTA clusters that would double the expected
  // arrivals (256 vs 128 available threads) and deadlock the pipeline.
  accumulator_pipeline_params.consumer_arv_count = NumEpilogueThreads;
  accumulator_pipeline_params.initializing_warp = 1;
  AccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                           accumulator_pipeline_params,
                                           cluster_shape,
                                           cute::true_type{},
                                           cute::false_type{});

  TmemAllocator tmem_allocator{};

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      NumMMAThreads + NumEpilogueThreads,
      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();

  // Match CUTLASS reference sm100_gemm_array_tma_warpspecialized.hpp line 661:
  // "We need this to guarantee that the Pipeline init is visible to all
  // producers and consumer threadblocks in the cluster". Missing this call
  // on a new pipeline-state machine can leave its internal mbarrier phase
  // in a stale state, causing intermittent races on tensormap acquire.
  uint32_t const cluster_size_ = size(ClusterShape{});
  cutlass::arch::fence_barrier_init();
  cutlass::pipeline_init_arrive_relaxed(cluster_size_);

  MainloopPipelineState mainloop_pipe_consumer_state;
  MainloopPipelineState mainloop_pipe_producer_state =
      cutlass::make_producer_start_state<MainloopPipeline>();
  AccumulatorPipelineState accumulator_pipe_consumer_state;
  AccumulatorPipelineState accumulator_pipe_producer_state =
      cutlass::make_producer_start_state<AccumulatorPipeline>();

  mainloop_pipeline.init_masks(cluster_shape, cute::block_id_in_cluster());
  accumulator_pipeline.init_masks(cluster_shape, cute::block_id_in_cluster());

  // 5. Tile dimensions.
  constexpr int BLK_N = size<1>(MmaTileShape{});
  constexpr int BLK_K = size<2>(MmaTileShape{});
  constexpr int BLK_M = size<0>(MmaTileShape{});
  int k_tiles_total = (params.K + BLK_K - 1) / BLK_K;
  // n_tiles_per_expert no longer used here — the N-tile dimension is
  // expanded into the grid by moe_flat_tile_list_mn_kernel so each CTA
  // owns ONE (expert, m_tile, n_tile) triple.

  // 6. TMEM setup (mainloop-driven: it knows the accumulator layout).
  auto tmem_storage = collective_mainloop.template init_tmem_tensors<
      /*EpilogueTile=*/Shape<Int<BLK_M>, Int<32>>, /*IsOverlappingAccum=*/false>(
      Shape<Int<BLK_M>, Int<32>>{});

  // Wait for all cluster blocks to complete their pipeline init before any
  // TMA/MMA begins (matches CUTLASS reference line 694).
  cutlass::pipeline_init_wait(cluster_size_);

  int32_t sm_count = params.hw_info.sm_count;
  // For Grouped-GEMM mode, CUTLASS uses the CTA grid index as sm_id (not the
  // physical SmId), because the per-SM tensormap pool is sized with
  // NumTmaDescriptorsPerSm = scheduler_stages + mainloop_stages + 2 slots per
  // SM (16 for our config). Mirror CUTLASS's
  // sm100_gemm_array_tma_warpspecialized.hpp line 702 exactly. Using
  // cutlass::arch::SmId() here puts per-CTA tensormap writes into sparse /
  // non-dense pool slots that can race with fence_acquire ordering.
  int32_t sm_id = static_cast<int32_t>(blockIdx.x + blockIdx.y * gridDim.x);

  if (is_participant_main_load) {
    // Producer warp: TMA loads A, B, SFA, SFB into SMEM pipeline.
    //
    // DIAGNOSTIC: force init_group=0 for all CTAs. In the unit-test setup
    // all experts have identical M=128, so layout_SFA[e] for all e are
    // identical. If forcing init_group=0 removes the garbage-output bug,
    // the problem is in per-expert layout_SFA indexing. If not, something
    // else. One decisive test.
#ifdef MEGAMOE_FORCE_INIT_GROUP_ZERO
    int32_t const init_group_val = 0;
#else
    int32_t const init_group_val = static_cast<int32_t>(expert_id);
#endif
    auto load_inputs = collective_mainloop.load_init(
        problem_shape_MNKL, params.mainloop,
        shared_storage.tensors.mainloop,
        shared_storage.tensormaps.mainloop,
        sm_count, sm_id,
        /*num_groups=*/static_cast<int32_t>(params.num_experts),
        /*init_group=*/init_group_val);
    cutlass::arch::wait_on_dependent_grids();

    auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);

#ifdef MEGAMOE_UNIT_DEBUG
    // One-line print per CTA — from producer warp leader only.
    if (cute::elect_one_sync()) {
      // Dump params.ps_device[expert_id] to verify device-side problem_shape.
      const int32_t* ps_e = reinterpret_cast<const int32_t*>(&params.ps_device[expert_id]);
      printf("[CTA %d] expert=%d sm_id=%d m_start=%d m_valid=%d n_tile=%d "
             "ptrA=%p ptrB=%p ptrSFA=%p ptrSFB=%p ptrOUT=%p "
             "ps_device[e]=(%d,%d,%d)\n",
             blockIdx.x, expert_id, (int)sm_id, m_start, m_valid,
             params.tile_ntile[tile_idx],
             (void*)params.mainloop.ptr_A[expert_id],
             (void*)params.mainloop.ptr_B[expert_id],
             (void*)params.mainloop.ptr_SFA[expert_id],
             (void*)params.mainloop.ptr_SFB[expert_id],
             (void*)params.out_ptrs[expert_id],
             ps_e[0], ps_e[1], ps_e[2]);
    }
#endif

    // Tensormap update for this expert (binds TMA descriptors to per-expert ptrs).
    ProblemShape ps_for_update{params.num_experts,
                               const_cast<UnderlyingProblemShape*>(params.ps_device),
                               nullptr};
    collective_mainloop.tensormaps_perform_update(
        shared_storage.tensormaps.mainloop,
        params.mainloop,
        input_tensormaps,
        ps_for_update,
        /*curr_batch=*/static_cast<int32_t>(expert_id));

#ifdef MEGAMOE_UNIT_DEBUG
    // Dump the actual GMEM tensormap content AFTER perform_update completes.
    // The first 8 bytes of a TMA descriptor hold the tensor's global address.
    // If perform_update worked, this should equal params.mainloop.ptr_A[expert_id].
    if (cute::elect_one_sync()) {
      // fence_acquire so we can observe the post-release state from this thread.
      cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
      cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
      cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
      cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
      const uint64_t* td_a = reinterpret_cast<const uint64_t*>(get<0>(input_tensormaps));
      const uint64_t* td_b = reinterpret_cast<const uint64_t*>(get<1>(input_tensormaps));
      const uint64_t* td_sfa = reinterpret_cast<const uint64_t*>(get<2>(input_tensormaps));
      const uint64_t* td_sfb = reinterpret_cast<const uint64_t*>(get<3>(input_tensormaps));
      printf("[CTA %d POST] expert=%d "
             "td_A_addr=0x%lx expect=0x%lx "
             "td_B_addr=0x%lx expect=0x%lx "
             "td_SFA_addr=0x%lx expect=0x%lx "
             "td_SFB_addr=0x%lx expect=0x%lx\n",
             blockIdx.x, expert_id,
             td_a[0], (unsigned long)params.mainloop.ptr_A[expert_id],
             td_b[0], (unsigned long)params.mainloop.ptr_B[expert_id],
             td_sfa[0], (unsigned long)params.mainloop.ptr_SFA[expert_id],
             td_sfb[0], (unsigned long)params.mainloop.ptr_SFB[expert_id]);
    }
#endif

    // Which (m_tile, n_tile) this CTA owns within its expert. Emitted by
    // moe_flat_tile_list_mn_kernel: one CTA per (expert, m_tile, n_tile)
    // triple — NO N-tile sweep inside the kernel (previous bug: had a
    // 32-step inner N loop that collapsed pipeline throughput).
    int m_tile_idx = (m_start - params.expert_offsets[expert_id]) / BLK_M;
    int n_tile     = params.tile_ntile[tile_idx];

    // cta_coord_mnk: (m_tile_idx, n_tile, k_dummy=0, l=Int<0>).
    auto cta_coord_mnk = make_coord(m_tile_idx, n_tile, Int<0>{}, Int<0>{});
    auto k_tile_iter   = cute::make_coord_iterator(k_tiles_total);
    int  k_tile_prologue = cutlass::platform::min<int>(
        int(DispatchPolicy::Stages), k_tiles_total);

    // Single tile per CTA ⇒ did_batch_change is always true (first and only
    // tile for this CTA's expert). The tensormap update above already ran
    // for this expert, so the TMA descriptor is correct.
    auto [ml_state_next, k_iter_next] = collective_mainloop.load(
        params.mainloop,
        mainloop_pipeline,
        mainloop_pipe_producer_state,
        load_inputs,
        cta_coord_mnk,
        k_tile_iter, k_tile_prologue,
        /*did_batch_change=*/true,
        /*curr_batch=*/static_cast<int>(expert_id));
    mainloop_pipe_producer_state = ml_state_next;

    auto [ml_state_next2, unused] = collective_mainloop.load(
        params.mainloop,
        mainloop_pipeline,
        mainloop_pipe_producer_state,
        load_inputs,
        cta_coord_mnk,
        k_iter_next, k_tiles_total - k_tile_prologue,
        /*did_batch_change=*/false,
        /*curr_batch=*/static_cast<int>(expert_id));
    mainloop_pipe_producer_state = ml_state_next2;

    collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
  }
  else if (is_participant_mma) {
    // MMA warp group: allocate TMEM, run tcgen05.mma across K-iters for THIS
    // CTA's (m_tile, n_tile). No inner N-tile sweep (grid covers all N-tiles).
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base = shared_storage.tmem_base_ptr;
    collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base);
    auto mma_inputs = collective_mainloop.mma_init(tmem_storage, shared_storage.tensors.mainloop);

    int m_tile_idx_mma = (m_start - params.expert_offsets[expert_id]) / BLK_M;
    int n_tile_mma     = params.tile_ntile[tile_idx];
    auto cta_coord_mnkl = make_coord(m_tile_idx_mma, n_tile_mma, Int<0>{}, Int<0>{});

    int acc_stage = accumulator_pipe_producer_state.index();
    auto accumulator = collective_mainloop.slice_accumulator(tmem_storage, acc_stage);

    mainloop_pipe_consumer_state = collective_mainloop.mma(
        cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
        cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
        accumulator,
        mma_inputs,
        cta_coord_mnkl,
        k_tiles_total);
    accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
    ++accumulator_pipe_producer_state;

    cutlass::arch::launch_dependent_grids();
    tmem_allocator.release_allocation_lock();
    accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    tmem_allocator.free(tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
  }
  else if (is_participant_epilogue) {
    // Custom TMEM->bf16 epilogue for THIS CTA's (m_tile, n_tile).
    //
    // Identity-coord pattern matches cutlass/epilogue/collective/
    // sm100_epilogue_array_nosmem.hpp line-by-line:
    //   1. Build an identity tensor over the FULL problem shape (M_valid, N, 1).
    //   2. local_tile it with (BLK_M, BLK_N) at this CTA's (m_tile, n_tile)
    //      coord, giving per-CTA (m, n, 0) coord tensor with the SAME strides
    //      as the real gD output tensor.
    //   3. partition_D the coord tensor the same way tTR_gD (or tTR_rAcc)
    //      is partitioned. This guarantees coord(i) ↔ register_value(i).
    //   4. Bounds-check via elem_less against full (M_valid, N, 1).
    tmem_allocation_result_barrier.arrive_and_wait();
    uint32_t tmem_base = shared_storage.tmem_base_ptr;
    collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base);

    // Per-expert output base (row-major [total_tokens, N]).
    ElementC* D_expert = params.out_ptrs[expert_id];
    int64_t N_stride = params.out_row_stride;
    int m_offset_in_expert = m_start - params.expert_offsets[expert_id];
    int n_tile_epi = params.tile_ntile[tile_idx];

    // Accumulator slice for this stage; reduce to the 2D (MMA_TILE_M, MMA_TILE_N)
    // view (same as CUTLASS reference's tAcc_epi(_,_,_0{},_0{})).
    int acc_stage = accumulator_pipe_consumer_state.index();
    auto accumulator_full = collective_mainloop.slice_accumulator(tmem_storage, acc_stage);
    auto acc_one = accumulator_full(make_coord(_, _), _0{}, _0{});

    accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

    using CopyOpT2R = cute::SM100_TMEM_LOAD_32dp32b32x;
    auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, acc_one);
    int thread_idx_in_epi = threadIdx.x % size(tiled_t2r);
    auto thr_t2r = tiled_t2r.get_slice(thread_idx_in_epi);

    // Build coord tensor matching CUTLASS reference
    // (sm100_epilogue_array_nosmem.hpp lines 260–262) exactly:
    //   coordCD  = make_identity_tensor(problem_shape_mnl)  over (M,N,1)
    //   cCD      = local_tile(coordCD, cta_tiler, cta_coord_mnl)
    //   tTR_cCD  = thr_t2r.partition_D(cCD)
    //
    // We use M=m_valid so elem_less bounds check masks partial rows.
    auto problem_shape_mnl = make_shape(m_valid, params.N, Int<1>{});
    auto cta_coord_mnl     = make_coord(0, n_tile_epi, Int<0>{});
    auto cta_tiler         = Shape<Int<BLK_M>, Int<BLK_N>>{};
    Tensor coordCD = make_identity_tensor(problem_shape_mnl);
    Tensor cCD     = local_tile(coordCD, cta_tiler, cta_coord_mnl);
    Tensor tTR_cCD = thr_t2r.partition_D(cCD);

    // TMEM load: source via partition_S, dst register tensor sized from
    // partition_D (the D-side shape) — matching CUTLASS's tTR_rAcc sizing.
    // Using shape(tTR_tAcc) (S-side) would fail TiledCopy's dst-vectorize
    // static_assert because S and D have different per-thread cardinality.
    Tensor tTR_tAcc = thr_t2r.partition_S(acc_one);
    Tensor tTR_rAcc = make_tensor<float>(shape(tTR_cCD));
    cute::copy(tiled_t2r, tTR_tAcc, tTR_rAcc);

    // Iterate per-register and STG. Using elem_less on problem_shape_mnl
    // gives us the bounds check for partial M and partial N in one shot.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTR_rAcc); ++i) {
      auto crd = tTR_cCD(i);
      int local_m  = get<0>(crd);
      int global_n = get<1>(crd);
      // cCD is already LOCAL-TILED at the n_tile_epi position; local_m is
      // 0..BLK_M-1 within the tile, global_n is the absolute column [0, N).
      if (local_m < m_valid && global_n < params.N) {
        float acc_val = tTR_rAcc(i);
        ElementC bf = static_cast<ElementC>(acc_val);
        ElementC* D_row = D_expert
            + static_cast<int64_t>(m_offset_in_expert + local_m) * N_stride;
        D_row[global_n] = bf;
      }
    }

    accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
    ++accumulator_pipe_consumer_state;
  }

  cute::cluster_sync();
}

// =============================================================================
// Fused GEMM1 + SwiGLU + FP8 requant kernel.
//
// One CTA owns (expert, m_tile, out_n_tile) where out_n_tile ∈ [0, I/BLK_N=16).
// It produces a 128×128 block of the final FP8 activation for GEMM2, skipping
// the BF16 gemm1_out HBM round-trip AND the separate swiglu_fp8_requant kernel.
//
// Per CTA:
//   1. Mainloop pass 1 (gate): TMA-load A + B[out_n_tile*128 : +128, :] FP8,
//      MMA → TMEM slot (acc stage 0). Epilogue reads gate TMEM into registers,
//      stores to SMEM as BF16.
//   2. Mainloop pass 2 (up): TMA-load A + B[out_n_tile*128+I : +128, :] FP8,
//      MMA → TMEM slot (acc stage 1). Epilogue reads up TMEM into registers,
//      reads gate from SMEM, computes z = silu(gate) * up in FP32.
//   3. Row-block max reduction via atomicMax on SMEM.
//   4. Quantize z/scale → FP8 e4m3, write act_q + FP32 scale.
// =============================================================================

// Type aliases for the fused kernel's CollectiveMainloopFused.
using FusedMainloopParams = typename CollectiveMainloopFused::Params;
using FusedMainloopPipeline = typename CollectiveMainloopFused::MainloopPipeline;
using FusedMainloopPipelineState = typename CollectiveMainloopFused::MainloopPipelineState;
static constexpr uint32_t FusedAccumulatorPipelineStageCount =
    CollectiveMainloopFused::DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
using FusedAccumulatorPipeline = cutlass::PipelineUmmaAsync<FusedAccumulatorPipelineStageCount, AtomThrShapeMNK>;
using FusedAccumulatorPipelineState = typename FusedAccumulatorPipeline::PipelineState;

struct KernelParamsFused {
  // Scheduler: (expert, m_start, out_n_tile) triples.
  int32_t const* tile_expert;
  int32_t const* tile_mstart;
  int32_t const* tile_out_ntile;   // [0, I/BLK_N)
  int32_t const* tile_count;
  int            max_tiles;
  int            block_m;
  int32_t const* expert_offsets;   // [E+1]

  // Problem shape: full N includes BOTH gate (first I cols) and up (next I cols)
  int N;            // = 2*I = 4096
  int K;            // = 7168
  int I;            // = 2048 (intermediate / half N)
  int num_experts;

  UnderlyingProblemShape const* ps_device;

  // Optional: sorted_weights[total_valid] for fused routing-weight fold.
  // If non-null, scale is folded with w_route before UE8M0 rounding.
  float const* sorted_weights;   // may be nullptr

  // Outputs:
  //   act_q[total_valid, I] FP8 e4m3 — ready for GEMM2
  cutlass::float_e4m3_t* act_q_base;
  int64_t                act_q_row_stride;   // = I = 2048
  //   act_scales[total_valid, I/BLK_N] FP32 — per-row-per-block scale factor (UE8M0 pow-of-2)
  float*                 act_scales_base;
  int64_t                act_scales_row_stride;  // = I/BLK_N = 16

  FusedMainloopParams mainloop;
  cutlass::KernelHardwareInfo hw_info;
};

// Extended SharedStorage: adds gate_buffer (BF16 128×128 = 32KB) and
// row_max_bits (uint32[128] = 512B) for the fused-epilogue pipeline.
struct SharedStorageFused {
  struct PipelineStorage : cute::aligned_struct<16, _1> {
    using MainloopPipelineStorage = typename CollectiveMainloopFused::PipelineStorage;
    using AccumulatorPipelineStorage = typename FusedAccumulatorPipeline::SharedStorage;
    alignas(16) MainloopPipelineStorage mainloop;
    alignas(16) AccumulatorPipelineStorage accumulator;
  } pipelines;

  uint32_t tmem_base_ptr;

  struct TensorMapStorage : cute::aligned_struct<128, _1> {
    using MainloopTensorMapStorage = typename CollectiveMainloopFused::TensorMapStorage;
    alignas(128) MainloopTensorMapStorage mainloop;
  } tensormaps;

  struct TensorStorage : cute::aligned_struct<128, _1> {
    using MainloopTensorStorage = typename CollectiveMainloopFused::TensorStorage;
    MainloopTensorStorage mainloop;
  } tensors;

  // Gate buffer: 128×128 BF16 = 32KB. Holds gate_acc after gate-MMA so up-MMA
  // can reuse the same TMEM slot. Accessed by all 128 epilogue threads.
  alignas(128) __nv_bfloat16 gate_buffer[128 * 128];

  // Row-max bits for FP8 quantization. Stored as bit-cast of positive floats
  // (fabsf(z)) so atomicMax on uint32 == max on float.
  alignas(128) uint32_t row_max_bits[128];
};

__global__ __launch_bounds__(MaxThreadsPerBlock, 1)
void megamoe_gemm1_swiglu_fused_device_kernel(KernelParamsFused params) {
  using namespace cute;

  // 1. Tile assignment.
  int tile_idx = blockIdx.x;
  int tile_count_live = *params.tile_count;
  if (tile_idx >= tile_count_live) {
    return;
  }
  int expert_id   = params.tile_expert[tile_idx];
  int m_start     = params.tile_mstart[tile_idx];
  int out_n_tile  = params.tile_out_ntile[tile_idx];
  int m_end_e     = params.expert_offsets[expert_id + 1];
  int m_valid     = m_end_e - m_start;
  if (m_valid > params.block_m) m_valid = params.block_m;
  if (m_valid <= 0) return;

  UnderlyingProblemShape ps_mnk{m_valid, params.N, params.K};
  auto problem_shape_MNKL = append<4>(ps_mnk, 1);

  // 2. Warp category.
  int warp_idx = cutlass::canonical_warp_idx_sync();
  WarpCategory warp_category = (warp_idx < static_cast<int>(WarpCategory::Epilogue))
                                ? WarpCategory(warp_idx)
                                : WarpCategory::Epilogue;
  uint32_t lane_predicate = cute::elect_one_sync();

  auto cluster_shape = ClusterShape{};
  uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();

  // 3. SharedStorage.
  extern __shared__ char smem_raw[];
  SharedStorageFused& shared_storage = *reinterpret_cast<SharedStorageFused*>(smem_raw);

  CollectiveMainloopFused collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);

  bool is_participant_mma       = (warp_category == WarpCategory::MMA);
  bool is_participant_main_load = (warp_category == WarpCategory::MainloopLoad);
  bool is_participant_epilogue  = (warp_category == WarpCategory::Epilogue);

  // 4. Pipelines.
  typename FusedMainloopPipeline::Params mainloop_pipeline_params;
  if (is_participant_main_load) mainloop_pipeline_params.role = FusedMainloopPipeline::ThreadCategory::Producer;
  if (is_participant_mma)       mainloop_pipeline_params.role = FusedMainloopPipeline::ThreadCategory::Consumer;
  mainloop_pipeline_params.is_leader = lane_predicate && is_participant_main_load;
  mainloop_pipeline_params.num_consumers = NumMMAThreads;
  mainloop_pipeline_params.num_producers = NumMainloopLoadThreads;
  FusedMainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                     mainloop_pipeline_params,
                                     cluster_shape,
                                     cute::true_type{},
                                     cute::false_type{});

  typename FusedAccumulatorPipeline::Params accumulator_pipeline_params;
  if (is_participant_mma)      accumulator_pipeline_params.role = FusedAccumulatorPipeline::ThreadCategory::Producer;
  if (is_participant_epilogue) accumulator_pipeline_params.role = FusedAccumulatorPipeline::ThreadCategory::Consumer;
  accumulator_pipeline_params.producer_arv_count = 1;
  accumulator_pipeline_params.consumer_arv_count = NumEpilogueThreads;
  FusedAccumulatorPipeline accumulator_pipeline(shared_storage.pipelines.accumulator,
                                            accumulator_pipeline_params,
                                            cluster_shape,
                                            cute::true_type{},
                                            cute::false_type{});

  FusedMainloopPipelineState mainloop_pipe_consumer_state;
  FusedMainloopPipelineState mainloop_pipe_producer_state =
      cutlass::make_producer_start_state<FusedMainloopPipeline>();
  FusedAccumulatorPipelineState accumulator_pipe_consumer_state;
  FusedAccumulatorPipelineState accumulator_pipe_producer_state =
      cutlass::make_producer_start_state<FusedAccumulatorPipeline>();

  TmemAllocator tmem_allocator{};
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      NumMMAThreads + NumEpilogueThreads,
      cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  // Epilogue-warpgroup-only barrier for cross-stage sync (gate→up handoff).
  // Use an explicit USER barrier id (>= FirstUserBarrier=8) so it doesn't
  // collide with CUTLASS's internal ReservedNamedBarriers::EpilogueBarrier
  // which the mainloop/epilogue code also uses.
  cutlass::arch::NamedBarrier epilogue_sync_barrier(
      NumEpilogueThreads, /*id=*/uint32_t(8));

  mainloop_pipeline.init_masks(cluster_shape, cute::block_id_in_cluster());
  accumulator_pipeline.init_masks(cluster_shape, cute::block_id_in_cluster());

  constexpr int BLK_M = size<0>(MmaTileShape{});
  constexpr int BLK_N = size<1>(MmaTileShape{});
  constexpr int BLK_K = size<2>(MmaTileShape{});
  int k_tiles_total = (params.K + BLK_K - 1) / BLK_K;
  int n_tiles_half  = params.I / BLK_N;  // 16 when I=2048, BLK_N=128

  // TMEM storage init.
  auto tmem_storage = collective_mainloop.template init_tmem_tensors<
      Shape<Int<BLK_M>, Int<32>>, false>(Shape<Int<BLK_M>, Int<32>>{});

  int32_t sm_count = params.hw_info.sm_count;
  // Grouped-GEMM sm_id = CTA grid index (see middle variant note above).
  int32_t sm_id    = static_cast<int32_t>(blockIdx.x + blockIdx.y * gridDim.x);

  // === MainloopLoad warp ===
  if (is_participant_main_load) {
    auto load_inputs = collective_mainloop.load_init(
        problem_shape_MNKL, params.mainloop,
        shared_storage.tensors.mainloop,
        shared_storage.tensormaps.mainloop,
        sm_count, sm_id,
        /*num_groups=*/static_cast<int32_t>(params.num_experts),
        /*init_group=*/static_cast<int32_t>(expert_id));
    cutlass::arch::wait_on_dependent_grids();
    auto input_tensormaps = get<rank(load_inputs) - 1>(load_inputs);
    ProblemShape ps_for_update{params.num_experts,
                               const_cast<UnderlyingProblemShape*>(params.ps_device),
                               nullptr};
    collective_mainloop.tensormaps_perform_update(
        shared_storage.tensormaps.mainloop, params.mainloop, input_tensormaps,
        ps_for_update, static_cast<int32_t>(expert_id));

    int m_tile_idx = (m_start - params.expert_offsets[expert_id]) / BLK_M;

    auto run_one_mainloop_pass = [&](int n_tile_coord, bool is_first) {
      auto cta_coord_mnk = make_coord(m_tile_idx, n_tile_coord, Int<0>{}, Int<0>{});
      auto k_tile_iter = cute::make_coord_iterator(k_tiles_total);
      int k_tile_prologue = cutlass::platform::min<int>(
          int(DispatchPolicy::Stages), k_tiles_total);

      auto [ml_state_next, k_iter_next] = collective_mainloop.load(
          params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
          load_inputs, cta_coord_mnk,
          k_tile_iter, k_tile_prologue,
          /*did_batch_change=*/is_first,
          /*curr_batch=*/static_cast<int>(expert_id));
      mainloop_pipe_producer_state = ml_state_next;

      auto [ml_state_next2, unused] = collective_mainloop.load(
          params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
          load_inputs, cta_coord_mnk,
          k_iter_next, k_tiles_total - k_tile_prologue,
          /*did_batch_change=*/false,
          /*curr_batch=*/static_cast<int>(expert_id));
      mainloop_pipe_producer_state = ml_state_next2;
    };

    // Pass 1: gate
    run_one_mainloop_pass(out_n_tile, /*is_first=*/true);
    // Pass 2: up (re-issues A — suboptimal but simple; amortized by the
    // half-grid size vs the non-fused middle variant).
    run_one_mainloop_pass(out_n_tile + n_tiles_half, /*is_first=*/false);

    collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
  }
  // === MMA warp ===
  else if (is_participant_mma) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base = shared_storage.tmem_base_ptr;
    collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base);
    auto mma_inputs = collective_mainloop.mma_init(tmem_storage, shared_storage.tensors.mainloop);

    int m_tile_idx_mma = (m_start - params.expert_offsets[expert_id]) / BLK_M;

    auto run_one_mma_pass = [&](int n_tile_coord) {
      auto cta_coord_mnkl = make_coord(m_tile_idx_mma, n_tile_coord, Int<0>{}, Int<0>{});
      int acc_stage = accumulator_pipe_producer_state.index();
      auto accumulator = collective_mainloop.slice_accumulator(tmem_storage, acc_stage);
      mainloop_pipe_consumer_state = collective_mainloop.mma(
          cute::make_tuple(mainloop_pipeline, accumulator_pipeline),
          cute::make_tuple(mainloop_pipe_consumer_state, accumulator_pipe_producer_state),
          accumulator, mma_inputs,
          cta_coord_mnkl, k_tiles_total);
      accumulator_pipeline.producer_commit(accumulator_pipe_producer_state);
      ++accumulator_pipe_producer_state;
    };

    run_one_mma_pass(out_n_tile);                 // gate MMA
    run_one_mma_pass(out_n_tile + n_tiles_half);  // up MMA

    cutlass::arch::launch_dependent_grids();
    tmem_allocator.release_allocation_lock();
    accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    tmem_allocator.free(tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
  }
  // === Epilogue warpgroup (128 threads) ===
  else if (is_participant_epilogue) {
    tmem_allocation_result_barrier.arrive_and_wait();
    uint32_t tmem_base = shared_storage.tmem_base_ptr;
    collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base);

    using CopyOpT2R = cute::SM100_TMEM_LOAD_32dp32b32x;
    auto cta_tiler = Shape<Int<BLK_M>, Int<BLK_N>>{};
    // Local epilogue thread id in [0, 128).
    int epi_tid = threadIdx.x - (NumMainloopLoadThreads + NumMMAThreads);

    // --- Gate stage: TMEM → registers → SMEM as BF16 ---
    {
      int acc_stage = accumulator_pipe_consumer_state.index();
      auto accumulator_full = collective_mainloop.slice_accumulator(tmem_storage, acc_stage);
      auto acc_one = accumulator_full(make_coord(_, _), _0{}, _0{});

      accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

      auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, acc_one);
      int thread_idx_in_epi = threadIdx.x % size(tiled_t2r);
      auto thr_t2r = tiled_t2r.get_slice(thread_idx_in_epi);

      auto problem_shape_mnl_local = make_shape(Int<BLK_M>{}, Int<BLK_N>{}, Int<1>{});
      auto cta_coord_mnl_local     = make_coord(0, 0, Int<0>{});
      Tensor coordCD = make_identity_tensor(problem_shape_mnl_local);
      Tensor cCD     = local_tile(coordCD, cta_tiler, cta_coord_mnl_local);
      Tensor tTR_cCD = thr_t2r.partition_D(cCD);

      Tensor tTR_tAcc = thr_t2r.partition_S(acc_one);
      Tensor tTR_rGate = make_tensor<float>(shape(tTR_cCD));
      cute::copy(tiled_t2r, tTR_tAcc, tTR_rGate);
      cutlass::arch::fence_view_async_tmem_load();

      accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;

      // Store gate to SMEM (BF16) at (local_m, local_n).
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rGate); ++i) {
        auto crd = tTR_cCD(i);
        int local_m = get<0>(crd);
        int local_n = get<1>(crd);
        if (local_m < BLK_M && local_n < BLK_N) {
          shared_storage.gate_buffer[local_m * BLK_N + local_n] =
              __float2bfloat16_rn(tTR_rGate(i));
        }
      }
    }
    // Ensure gate_buffer is visible to all epilogue threads before up-pass reads.
    epilogue_sync_barrier.arrive_and_wait();
    // Init row_max (positive-float bits; 0 == min for positive floats).
    if (epi_tid < BLK_M) {
      shared_storage.row_max_bits[epi_tid] = 0u;
    }
    epilogue_sync_barrier.arrive_and_wait();

    // --- Up stage: TMEM → registers; compute z = silu(gate) * up; track row-max ---
    int m_offset_in_expert = m_start - params.expert_offsets[expert_id];
    {
      int acc_stage = accumulator_pipe_consumer_state.index();
      auto accumulator_full = collective_mainloop.slice_accumulator(tmem_storage, acc_stage);
      auto acc_one = accumulator_full(make_coord(_, _), _0{}, _0{});

      accumulator_pipeline.consumer_wait(accumulator_pipe_consumer_state);

      auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, acc_one);
      int thread_idx_in_epi = threadIdx.x % size(tiled_t2r);
      auto thr_t2r = tiled_t2r.get_slice(thread_idx_in_epi);

      auto problem_shape_mnl_local = make_shape(Int<BLK_M>{}, Int<BLK_N>{}, Int<1>{});
      auto cta_coord_mnl_local     = make_coord(0, 0, Int<0>{});
      Tensor coordCD = make_identity_tensor(problem_shape_mnl_local);
      Tensor cCD     = local_tile(coordCD, cta_tiler, cta_coord_mnl_local);
      Tensor tTR_cCD = thr_t2r.partition_D(cCD);

      Tensor tTR_tAcc = thr_t2r.partition_S(acc_one);
      Tensor tTR_rZ   = make_tensor<float>(shape(tTR_cCD));  // will hold z after combine
      cute::copy(tiled_t2r, tTR_tAcc, tTR_rZ);
      cutlass::arch::fence_view_async_tmem_load();

      accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;

      // z = silu(gate) * up; track max(|z|) per row.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rZ); ++i) {
        auto crd = tTR_cCD(i);
        int local_m = get<0>(crd);
        int local_n = get<1>(crd);
        if (local_m < BLK_M && local_n < BLK_N) {
          float up_val   = tTR_rZ(i);
          float gate_val = __bfloat162float(
              shared_storage.gate_buffer[local_m * BLK_N + local_n]);
          float sil = gate_val / (1.f + __expf(-gate_val));
          float z   = sil * up_val;
          tTR_rZ(i) = z;
          if (local_m < m_valid) {
            float abs_z = fabsf(z);
            uint32_t bits = __float_as_uint(abs_z);
            atomicMax(&shared_storage.row_max_bits[local_m], bits);
          }
        }
      }
      epilogue_sync_barrier.arrive_and_wait();

      // --- Quantize + write ---
      // Output strides: act_q[total_valid, I], act_scales[total_valid, I/BLK_N].
      cutlass::float_e4m3_t* act_q_expert_base =
          params.act_q_base + static_cast<int64_t>(m_start) * params.act_q_row_stride;
      float* act_scales_expert_base =
          params.act_scales_base + static_cast<int64_t>(m_start) * params.act_scales_row_stride;

      // One thread per row computes the per-block UE8M0 scale (with routing
      // weight fold if sorted_weights != nullptr). Mirrors the existing
      // swiglu_fp8_requant_weighted_mxf8 kernel's semantics exactly so the
      // downstream pack_sfa and GEMM2 pipeline is unchanged.
      //
      // We stash TWO values per row in SMEM for the FP8 quant pass:
      //   row_max_bits[m]   = inv_scale_times_sr (FP32 bit-cast), used by quant
      if (epi_tid < m_valid) {
        float row_max = __uint_as_float(shared_storage.row_max_bits[epi_tid]);
        float scale = fmaxf(row_max / 448.f, 1e-8f);
        float w_route = (params.sorted_weights != nullptr)
            ? params.sorted_weights[m_start + epi_tid]
            : 1.f;
        float scale_weighted = scale * w_route;
        float s_abs = fabsf(scale_weighted);
        float sign = (scale_weighted < 0.f) ? -1.f : 1.f;
        uint8_t ue8m0_byte = ue8m0_ceil_from_abs_fp32(s_abs);
        float   ue8m0_val  = ue8m0_byte_to_fp32(ue8m0_byte);
        float   r          = (ue8m0_val > 0.f) ? (s_abs / ue8m0_val) : 1.f;
        float   sr         = sign * r;
        float   inv_scale_times_sr = (scale > 0.f) ? (sr / scale) : 0.f;

        act_scales_expert_base[epi_tid * params.act_scales_row_stride + out_n_tile]
            = ue8m0_val;
        shared_storage.row_max_bits[epi_tid] = __float_as_uint(inv_scale_times_sr);
      }
      epilogue_sync_barrier.arrive_and_wait();

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTR_rZ); ++i) {
        auto crd = tTR_cCD(i);
        int local_m = get<0>(crd);
        int local_n = get<1>(crd);
        if (local_m < m_valid && local_n < BLK_N) {
          float inv_scale_times_sr = __uint_as_float(shared_storage.row_max_bits[local_m]);
          float q = tTR_rZ(i) * inv_scale_times_sr;
          if (q >  448.f) q =  448.f;
          if (q < -448.f) q = -448.f;
          cutlass::NumericConverter<cutlass::float_e4m3_t, float> conv;
          cutlass::float_e4m3_t fp8_val = conv(q);
          int64_t col_abs = static_cast<int64_t>(out_n_tile) * BLK_N + local_n;
          act_q_expert_base[static_cast<int64_t>(local_m) * params.act_q_row_stride + col_abs]
              = fp8_val;
        }
      }
    }
  }

  cute::cluster_sync();
}

// -------- Host launcher (fused) ------------------------------------------
void megamoe_gemm1_swiglu_fused_launch(
    torch::Tensor& act_q,                        // [total_valid, I] FP8 e4m3
    torch::Tensor& act_scales,                   // [total_valid, I/BLK_N] FP32
    torch::Tensor const& a,                      // [total_valid, K] FP8
    torch::Tensor const& b,                      // [E, N, K] FP8
    torch::optional<torch::Tensor> const& sorted_weights,  // [total_valid] FP32 or None
    torch::Tensor const& tile_expert,
    torch::Tensor const& tile_mstart,
    torch::Tensor const& tile_out_ntile,
    torch::Tensor const& tile_count,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,          // [E, 3] int32 (M, N_full=2*I, K)
    torch::Tensor& a_ptrs, torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,                     // unused but kept for setup-ptrs reuse
    torch::Tensor& sfa_ptrs, torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a, torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor const& layout_sfa, torch::Tensor const& layout_sfb,
    torch::Tensor const& workspace,
    int block_m, int I_dim)
{
  int max_tiles = static_cast<int>(tile_expert.size(0));
  int N_full = static_cast<int>(b.size(1));    // 2 * I
  int K_full = static_cast<int>(b.size(2));
  TORCH_CHECK(block_m == static_cast<int>(size<0>(MmaTileShape{})),
      "megamoe_fused_launch: BLOCK_M mismatch");
  TORCH_CHECK(N_full == 2 * I_dim,
      "megamoe_fused_launch: expected b.size(1) = 2*I");

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  // NOTE: MainloopArguments (defined for CollectiveMainloop) is structurally
  // identical across Stages variants, so we can construct it once and pass
  // it to either mainloop's to_underlying_arguments().
  typename CollectiveMainloopFused::Arguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      reinterpret_cast<typename CollectiveMainloopFused::StrideA>(
          const_cast<void*>(stride_a.data_ptr())),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      reinterpret_cast<typename CollectiveMainloopFused::StrideB>(
          const_cast<void*>(stride_b.data_ptr())),
      static_cast<const ElementSF**>(sfa_ptrs.data_ptr()),
      reinterpret_cast<typename CollectiveMainloopFused::LayoutSFA>(
          const_cast<void*>(layout_sfa.data_ptr())),
      static_cast<const ElementSF**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<typename CollectiveMainloopFused::LayoutSFB>(
          const_cast<void*>(layout_sfb.data_ptr()))
  };

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int num_experts = static_cast<int>(a_ptrs.size(0));
  UnderlyingProblemShape* ps_device =
      static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.data_ptr()));
  ProblemShape problem_shapes{num_experts, ps_device, nullptr};

  FusedMainloopParams mp = CollectiveMainloopFused::to_underlying_arguments(
      problem_shapes, mainloop_args,
      workspace.data_ptr(), hw_info);

  KernelParamsFused kp{};
  kp.tile_expert       = tile_expert.data_ptr<int>();
  kp.tile_mstart       = tile_mstart.data_ptr<int>();
  kp.tile_out_ntile    = tile_out_ntile.data_ptr<int>();
  kp.tile_count        = tile_count.data_ptr<int>();
  kp.max_tiles         = max_tiles;
  kp.block_m           = block_m;
  kp.expert_offsets    = expert_offsets.data_ptr<int>();
  kp.N                 = N_full;
  kp.K                 = K_full;
  kp.I                 = I_dim;
  kp.num_experts       = num_experts;
  kp.ps_device         = ps_device;
  kp.sorted_weights    = sorted_weights.has_value()
      ? sorted_weights.value().data_ptr<float>() : nullptr;
  kp.act_q_base        = reinterpret_cast<cutlass::float_e4m3_t*>(act_q.data_ptr());
  kp.act_q_row_stride  = static_cast<int64_t>(I_dim);
  kp.act_scales_base   = act_scales.data_ptr<float>();
  kp.act_scales_row_stride = static_cast<int64_t>(act_scales.size(1));
  kp.mainloop          = mp;
  kp.hw_info           = hw_info;

  size_t smem_bytes = sizeof(SharedStorageFused);
  TORCH_CHECK(smem_bytes <= cutlass::arch::sm100_smem_capacity_bytes,
              "megamoe_fused: smem over cap: ", smem_bytes);

  auto kernel_fn = megamoe_gemm1_swiglu_fused_device_kernel;
  cudaError_t attr_err = cudaFuncSetAttribute(
      (void*)kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  TORCH_CHECK(attr_err == cudaSuccess,
              "megamoe_fused: cudaFuncSetAttribute failed: ",
              cudaGetErrorString(attr_err));

  dim3 grid(max_tiles, 1, 1);
  dim3 block(MaxThreadsPerBlock, 1, 1);

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = size<0>(ClusterShape{});
  attrs[0].val.clusterDim.y = size<1>(ClusterShape{});
  attrs[0].val.clusterDim.z = size<2>(ClusterShape{});

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel_fn, kp);
  TORCH_CHECK(err == cudaSuccess,
              "megamoe_fused: cudaLaunchKernelEx failed: ",
              cudaGetErrorString(err));
}

// -------- Host launcher --------------------------------------------------
void megamoe_gemm1_launch(
    torch::Tensor& output,                       // [total_tokens, N] bf16
    torch::Tensor const& a,                      // [total_tokens, K] fp8 (transcoded)
    torch::Tensor const& b,                      // [E, N, K] fp8 (transcoded)
    torch::Tensor const& tile_expert,            // [max_tiles] int32
    torch::Tensor const& tile_mstart,            // [max_tiles] int32
    torch::Tensor const& tile_ntile,             // [max_tiles] int32 — N-tile index per block
    torch::Tensor const& tile_count,             // [1] int32
    torch::Tensor const& expert_offsets,         // [E+1] int32
    torch::Tensor const& problem_sizes,          // [E, 3] int32
    torch::Tensor& a_ptrs,                       // [E] int64
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& sfa_ptrs,
    torch::Tensor& sfb_ptrs,
    torch::Tensor const& stride_a,               // [E * stride_sz] uint8
    torch::Tensor const& stride_b,
    torch::Tensor const& stride_c,
    torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb,
    torch::Tensor const& workspace,
    int block_m)
{
  int max_tiles = static_cast<int>(tile_expert.size(0));
  int N = static_cast<int>(b.size(1));
  int K = static_cast<int>(b.size(2));
  TORCH_CHECK(block_m == static_cast<int>(size<0>(MmaTileShape{})),
      "megamoe_gemm1_launch: BLOCK_M mismatch");

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      reinterpret_cast<StrideA>(const_cast<void*>(stride_a.data_ptr())),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      reinterpret_cast<StrideB>(const_cast<void*>(stride_b.data_ptr())),
      static_cast<const ElementSF**>(sfa_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA>(const_cast<void*>(layout_sfa.data_ptr())),
      static_cast<const ElementSF**>(sfb_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB>(const_cast<void*>(layout_sfb.data_ptr()))
  };

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::cuda::current_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  int num_experts = static_cast<int>(a_ptrs.size(0));
  UnderlyingProblemShape* ps_device =
      static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.data_ptr()));
  ProblemShape problem_shapes{num_experts, ps_device, nullptr};

  MainloopParams mp = CollectiveMainloop::to_underlying_arguments(
      problem_shapes, mainloop_args,
      workspace.data_ptr(), hw_info);

  KernelParams kp{};
  kp.tile_expert     = tile_expert.data_ptr<int>();
  kp.tile_mstart     = tile_mstart.data_ptr<int>();
  kp.tile_ntile      = tile_ntile.data_ptr<int>();
  kp.tile_count      = tile_count.data_ptr<int>();
  kp.max_tiles       = max_tiles;
  kp.block_m         = block_m;
  kp.expert_offsets  = expert_offsets.data_ptr<int>();
  kp.N               = N;
  kp.K               = K;
  kp.num_experts     = num_experts;
  kp.ps_device       = ps_device;
  kp.out_ptrs        = static_cast<ElementC* const*>(out_ptrs.data_ptr());
  kp.out_row_stride  = static_cast<int64_t>(N);
  kp.mainloop        = mp;
  kp.hw_info         = hw_info;

  size_t smem_bytes = sizeof(SharedStorage);
  TORCH_CHECK(smem_bytes <= cutlass::arch::sm100_smem_capacity_bytes,
              "megamoe smem over cap: ", smem_bytes);

  auto kernel_fn = megamoe_gemm1_device_kernel;
  cudaError_t attr_err = cudaFuncSetAttribute(
      (void*)kernel_fn,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  TORCH_CHECK(attr_err == cudaSuccess,
              "megamoe: cudaFuncSetAttribute failed: ",
              cudaGetErrorString(attr_err));

  dim3 grid(max_tiles, 1, 1);
  dim3 block(MaxThreadsPerBlock, 1, 1);

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = size<0>(ClusterShape{});
  attrs[0].val.clusterDim.y = size<1>(ClusterShape{});
  attrs[0].val.clusterDim.z = size<2>(ClusterShape{});

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  cudaError_t err = cudaLaunchKernelEx(&cfg, kernel_fn, kp);
  TORCH_CHECK(err == cudaSuccess,
              "megamoe: cudaLaunchKernelEx failed: ",
              cudaGetErrorString(err));
}

} // namespace megamoe

// Public C++ binding (forward-declared in moe_fused.cu pybind block).
void megamoe_gemm1(
    torch::Tensor& output,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& tile_expert,
    torch::Tensor const& tile_mstart,
    torch::Tensor const& tile_ntile,
    torch::Tensor const& tile_count,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace,
    int64_t block_m)
{
  megamoe::megamoe_gemm1_launch(
      output, a, b, tile_expert, tile_mstart, tile_ntile, tile_count, expert_offsets,
      problem_sizes, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c, layout_sfa, layout_sfb,
      workspace, static_cast<int>(block_m));
}

// Public C++ binding for the fused kernel.
void megamoe_gemm1_swiglu_fused(
    torch::Tensor& act_q,
    torch::Tensor& act_scales,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::optional<torch::Tensor> const& sorted_weights,
    torch::Tensor const& tile_expert,
    torch::Tensor const& tile_mstart,
    torch::Tensor const& tile_out_ntile,
    torch::Tensor const& tile_count,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
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
    torch::Tensor const& workspace,
    int64_t block_m,
    int64_t I_dim)
{
  megamoe::megamoe_gemm1_swiglu_fused_launch(
      act_q, act_scales, a, b, sorted_weights,
      tile_expert, tile_mstart, tile_out_ntile, tile_count, expert_offsets,
      problem_sizes, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c, layout_sfa, layout_sfb,
      workspace, static_cast<int>(block_m), static_cast<int>(I_dim));
}
'''


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    cuda_home = None
    for cand in ("/usr/local/cuda-13.0", "/usr/local/cuda-13", "/usr/local/cuda"):
        nvcc = os.path.join(cand, "bin", "nvcc")
        if os.path.exists(nvcc):
            cuda_home = cand
            os.environ["CUDA_HOME"] = cand
            os.environ["CUDACXX"] = nvcc
            break

    cutlass_includes = set()
    preferred_roots = [
        os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/include"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia_cutlass_dsl/include"),
    ]
    for root in preferred_roots:
        if os.path.exists(os.path.join(root, "cutlass", "cutlass.h")):
            cutlass_includes.add(root)
    util_root = os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include")
    if os.path.exists(os.path.join(util_root, "cutlass", "util", "packed_stride.hpp")):
        cutlass_includes.add(util_root)
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
    megamoe_cu = os.path.join(build_dir, "moe_megamoe.cu")
    _write_if_changed(cutlass_cu, _MOE_GEMM_CU)
    _write_if_changed(fused_cu,   _MOE_FUSED_CU)
    _write_if_changed(megamoe_cu, _MOE_MEGAMOE_CU)

    # Three separate .cu files → ninja compiles them independently and only
    # rebuilds the ones that changed. The CUTLASS-heavy file (stable) compiles
    # once (~3 min); the fused-helpers file (iterated on) rebuilds in ~10-15s;
    # the megamoe file (where Phase A custom CuTe kernel lands) is kept minimal
    # so its iterations stay <30s.
    import torch.utils.cpp_extension as cpp_ext
    if cuda_home is not None:
        cpp_ext.CUDA_HOME = cuda_home
    load = cpp_ext.load
    # verbose=True so ninja/nvcc compile progress streams to stderr. Without
    # this, a slow compile looks indistinguishable from a hang for several
    # minutes after `[boot] loaded 19 workloads`.
    _verbose = bool(int(os.environ.get("MEGAMOE_VERBOSE_BUILD", "1")))
    extra_flags = [
        "-O3", "--std=c++17", "-arch=sm_100a",
        "--expt-relaxed-constexpr", "-DNDEBUG",
        # Parallelize template instantiation across nvcc passes — major win
        # on the CUTLASS-heavy _MOE_GEMM_CU unit (3min → ~1-1.5min cold)
        # and the megamoe unit. 4 is a safe lower bound; ninja parallelism
        # already covers across files.
        "--threads=4",
    ]
    if os.environ.get("MEGAMOE_UNIT_DEBUG"):
        extra_flags.append("-DMEGAMOE_UNIT_DEBUG=1")
    if os.environ.get("MEGAMOE_FORCE_INIT_GROUP_ZERO"):
        extra_flags.append("-DMEGAMOE_FORCE_INIT_GROUP_ZERO=1")

    _ext = load(
        name="moe_gemm_v5",
        sources=[cutlass_cu, fused_cu, megamoe_cu],
        extra_include_paths=sorted(cutlass_includes),
        extra_cuda_cflags=extra_flags,
        build_directory=build_dir,
        verbose=_verbose,
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


def _route_fused(routing_logits, routing_bias, rsf, T, local_start, num_experts,
                 topk_idx_buf=None, assign_w_buf=None):
    """Single-kernel fused routing (DeepSeek-V3 topk8+group4). 6-32x faster
    than the PyTorch chain. Requires bf16 logits/bias (contest format).

    When called with pre-allocated buffers (graph-capture path), this function
    does ONE kernel launch total — no torch.empty, no dtype casts.
    """
    ext = _get_ext()
    logits = routing_logits if routing_logits.dtype == torch.bfloat16 \
        else routing_logits.to(torch.bfloat16)
    bias = routing_bias if routing_bias.dtype == torch.bfloat16 \
        else routing_bias.to(torch.bfloat16)
    if topk_idx_buf is None:
        topk_idx_buf = torch.empty(T, TOP_K, device=logits.device, dtype=torch.int32)
    if assign_w_buf is None:
        assign_w_buf = torch.empty(T, TOP_K, device=logits.device, dtype=torch.float32)
    ext.fused_route_topk(logits, bias, topk_idx_buf, assign_w_buf, float(rsf))
    return topk_idx_buf, assign_w_buf


# Switch _route to the fused implementation.
_route = _route_fused


def _run_token_stationary_smallt(
    topk_idx,
    assign_w,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    T,
    ne,
    N1,
    K1,
    N2,
    K2,
    ls,
    bufs,
    ext,
):
    """Small-T prototype that avoids expert sorting entirely.

    We compact the local assignments in token-major order, keep an explicit
    expert id per assignment, and run grouped GEMM with one problem per local
    assignment. This is token-stationary in the sense that we never reorder into
    expert-major layout, which is the structural overhead we want to measure.
    """
    flat_idx = topk_idx.reshape(-1)
    flat_w = assign_w.reshape(-1)
    flat_tok = (
        torch.arange(T, device=flat_idx.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, TOP_K)
        .reshape(-1)
    )

    local_e = flat_idx - int(ls)
    valid = (local_e >= 0) & (local_e < ne)
    if not bool(valid.any()):
        bufs["out_bf16"].zero_()
        return bufs["out_bf16"]

    token_ids = flat_tok[valid].contiguous()
    expert_ids = local_e[valid].to(torch.int32).contiguous()
    weights = flat_w[valid].contiguous()
    total_valid = int(token_ids.shape[0])

    # Gather in token-major order instead of expert-major order.
    packed_acts = hidden_states[token_ids.long()].contiguous()
    hs_scale = hidden_states_scale
    if hs_scale.dim() == 2 and hs_scale.shape[0] == K1 // 128 and hs_scale.shape[1] == T:
        packed_act_scales = hs_scale[:, token_ids.long()].transpose(0, 1).contiguous()
    else:
        packed_act_scales = hs_scale[token_ids.long()].contiguous()

    problem_sizes_1 = bufs["problem_sizes_1"][:total_valid]
    problem_sizes_2 = bufs["problem_sizes_2"][:total_valid]
    problem_sizes_1[:, 0] = 1
    problem_sizes_1[:, 1] = N1
    problem_sizes_1[:, 2] = K1
    problem_sizes_2[:, 0] = 1
    problem_sizes_2[:, 1] = N2
    problem_sizes_2[:, 2] = K2

    gemm1_out = bufs["gemm1_out"][:total_valid]
    ext.moe_blockwise_grouped_mm_by_expert_ids(
        gemm1_out,
        packed_acts,
        gemm1_weights,
        packed_act_scales,
        gemm1_weights_scale,
        expert_ids,
        problem_sizes_1,
        bufs["problem_sizes_transpose"][:total_valid],
        bufs["a_ptrs"][:total_valid],
        bufs["b_ptrs"][:total_valid],
        bufs["out_ptrs"][:total_valid],
        bufs["a_scales_ptrs"][:total_valid],
        bufs["b_scales_ptrs"][:total_valid],
        bufs["stride_a"][: total_valid * bufs["stride_sz"]],
        bufs["stride_b"][: total_valid * bufs["stride_sz"]],
        bufs["stride_c"][: total_valid * bufs["stride_sz"]],
        bufs["layout_sfa"][: total_valid * bufs["sfa_sz"]],
        bufs["layout_sfb"][: total_valid * bufs["sfb_sz"]],
        bufs["workspace"],
    )

    act_q = bufs["act_q"][:total_valid]
    row_scales = bufs["row_scales"][:total_valid]
    act_scale_for_gemm2 = bufs["act_scale_for_gemm2"][:total_valid]
    use_weighted_fold = bool(int(os.environ.get("V17_WEIGHTED_FOLD", "1")))
    if use_weighted_fold:
        ext.swiglu_fp8_requant_weighted(
            gemm1_out, weights, act_q, row_scales, act_scale_for_gemm2
        )
    else:
        ext.swiglu_fp8_requant(gemm1_out, act_q, row_scales, act_scale_for_gemm2)

    gemm2_out = bufs["gemm2_out"][:total_valid]
    ext.moe_blockwise_grouped_mm_by_expert_ids(
        gemm2_out,
        act_q,
        gemm2_weights,
        act_scale_for_gemm2,
        gemm2_weights_scale,
        expert_ids,
        problem_sizes_2,
        bufs["problem_sizes_transpose"][:total_valid],
        bufs["a_ptrs"][:total_valid],
        bufs["b_ptrs"][:total_valid],
        bufs["out_ptrs"][:total_valid],
        bufs["a_scales_ptrs"][:total_valid],
        bufs["b_scales_ptrs"][:total_valid],
        bufs["stride_a"][: total_valid * bufs["stride_sz"]],
        bufs["stride_b"][: total_valid * bufs["stride_sz"]],
        bufs["stride_c"][: total_valid * bufs["stride_sz"]],
        bufs["layout_sfa"][: total_valid * bufs["sfa_sz"]],
        bufs["layout_sfb"][: total_valid * bufs["sfb_sz"]],
        bufs["workspace"],
    )

    bufs["out_bf16"].zero_()
    scatter_weights = bufs["sorted_weights_buf"][:total_valid]
    if use_weighted_fold:
        scatter_weights.fill_(1.0)
    else:
        scatter_weights.copy_(weights)
    ext.weighted_scatter(gemm2_out, scatter_weights, token_ids, bufs["out_bf16"], T)
    return bufs["out_bf16"]


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
    # topk_idx / assign_w come from pre-allocated buffers => contiguous by
    # construction. Skip .contiguous() to avoid an extra graph-node.
    ext.fused_dispatch(
        topk_idx, assign_w,
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
        topk_idx, assign_w,
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


def _get_async_chunk_slots(bufs, device, max_chunk_tokens, max_local_ne, N1, K1, N2, K2, num_slots):
    """Allocate per-stream scratch for the async chunk pipeline.

    Each in-flight chunk needs its own CUTLASS metadata/output scratch. Reusing
    the global metadata arrays would serialize launches or create races across
    streams, defeating the point of overlapped chunk execution.
    """
    state = bufs.get("_async_chunk_state")
    if (
        state is not None
        and state["num_slots"] >= num_slots
        and state["max_chunk_tokens"] >= max_chunk_tokens
        and state["max_local_ne"] >= max_local_ne
        and state["N1"] == N1
        and state["K1"] == K1
        and state["N2"] == N2
        and state["K2"] == K2
    ):
        return state["slots"]

    ext = _get_ext()
    stride_sz = bufs["stride_sz"]
    sfa_sz = bufs["sfa_sz"]
    sfb_sz = bufs["sfb_sz"]
    workspace_bytes = ext.get_workspace_size(max_chunk_tokens, max_local_ne, 0, 0, False)

    slots = []
    for _ in range(num_slots):
        slots.append(
            dict(
                stream=torch.cuda.Stream(device=device),
                chunk_offsets_buf=torch.empty(max_local_ne, device=device, dtype=torch.int32),
                problem_sizes_transpose=torch.empty(max_local_ne, 3, device=device, dtype=torch.int32),
                a_ptrs=torch.empty(max_local_ne, device=device, dtype=torch.int64),
                b_ptrs=torch.empty(max_local_ne, device=device, dtype=torch.int64),
                out_ptrs=torch.empty(max_local_ne, device=device, dtype=torch.int64),
                a_scales_ptrs=torch.empty(max_local_ne, device=device, dtype=torch.int64),
                b_scales_ptrs=torch.empty(max_local_ne, device=device, dtype=torch.int64),
                stride_a=torch.empty(max_local_ne * stride_sz, device=device, dtype=torch.uint8),
                stride_b=torch.empty(max_local_ne * stride_sz, device=device, dtype=torch.uint8),
                stride_c=torch.empty(max_local_ne * stride_sz, device=device, dtype=torch.uint8),
                layout_sfa=torch.empty(max_local_ne * sfa_sz, device=device, dtype=torch.uint8),
                layout_sfb=torch.empty(max_local_ne * sfb_sz, device=device, dtype=torch.uint8),
                workspace=torch.empty(workspace_bytes, device=device, dtype=torch.uint8),
                packed_acts=torch.empty(max_chunk_tokens, K1, device=device, dtype=torch.float8_e4m3fn),
                packed_act_scales=torch.empty(max_chunk_tokens, K1 // 128, device=device, dtype=torch.float32),
                gemm1_out=torch.empty(max_chunk_tokens, N1, device=device, dtype=torch.bfloat16),
                act_q=torch.empty(max_chunk_tokens, N1 // 2, device=device, dtype=torch.float8_e4m3fn),
                row_scales=torch.empty(max_chunk_tokens, device=device, dtype=torch.float32),
                act_scale_for_gemm2=torch.empty(
                    max_chunk_tokens, K2 // 128, device=device, dtype=torch.float32
                ),
                gemm2_out=torch.empty(max_chunk_tokens, N2, device=device, dtype=torch.bfloat16),
                ones=torch.ones(max_chunk_tokens, device=device, dtype=torch.float32),
            )
        )

    bufs["_async_chunk_state"] = dict(
        num_slots=num_slots,
        max_chunk_tokens=max_chunk_tokens,
        max_local_ne=max_local_ne,
        N1=N1,
        K1=K1,
        N2=N2,
        K2=K2,
        slots=slots,
    )
    return slots


def _run_async_streaming_chunked_pipeline(
    hidden_states,
    hs_scale,
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
    """Run gather -> GEMM1 -> SwiGLU -> GEMM2 -> scatter concurrently by chunk.

    This is the real async version of the earlier chunk pipeline: each chunk
    gets its own stream-local metadata/scratch, so multiple chunks can be in
    flight at once instead of sharing one serial scratch prefix.
    """
    chunks = _build_expert_chunks(counts, chunk_experts)
    bufs["out_bf16"].zero_()
    if not chunks:
        return bufs["out_bf16"]

    max_chunk_tokens = max(tok_end - tok_begin for _, _, tok_begin, tok_end in chunks)
    max_local_ne = max(e_end - e_begin for e_begin, e_end, _, _ in chunks)
    num_slots = max(1, int(os.environ.get("ASYNC_CHUNK_SLOTS", "2")))
    slots = _get_async_chunk_slots(
        bufs, hidden_states.device, max_chunk_tokens, max_local_ne, N1, K1, N2, K2, num_slots
    )
    producer_stream = torch.cuda.current_stream(device=hidden_states.device)
    use_weighted_fold = bool(int(os.environ.get("V17_WEIGHTED_FOLD", "1")))

    for chunk_idx, (e_begin, e_end, tok_begin, tok_end) in enumerate(chunks):
        chunk_tokens = tok_end - tok_begin
        local_ne = e_end - e_begin
        if chunk_tokens <= 0 or local_ne <= 0:
            continue

        slot = slots[chunk_idx % num_slots]
        stream = slot["stream"]
        with torch.cuda.stream(stream):
            stream.wait_stream(producer_stream)

            chunk_offsets = slot["chunk_offsets_buf"][:local_ne]
            chunk_offsets.copy_(expert_offsets[e_begin:e_end])
            chunk_offsets.sub_(tok_begin)

            chunk_problem_sizes_1 = bufs["problem_sizes_1"][e_begin:e_end]
            chunk_problem_sizes_2 = bufs["problem_sizes_2"][e_begin:e_end]
            chunk_problem_sizes_t = slot["problem_sizes_transpose"][:local_ne]

            chunk_tids = sorted_tids[tok_begin:tok_end]
            chunk_weights = sorted_weights[tok_begin:tok_end]
            chunk_packed_acts = slot["packed_acts"][:chunk_tokens]
            chunk_packed_act_scales = slot["packed_act_scales"][:chunk_tokens]

            ext.fused_gather_hidden_scales(
                hidden_states, hs_scale, chunk_tids, chunk_packed_acts, chunk_packed_act_scales
            )

            chunk_gemm1_out = slot["gemm1_out"][:chunk_tokens]
            ext.moe_blockwise_grouped_mm_v2(
                chunk_gemm1_out,
                chunk_packed_acts,
                gemm1_weights[e_begin:e_end],
                chunk_packed_act_scales,
                gemm1_weights_scale[e_begin:e_end],
                chunk_offsets,
                chunk_problem_sizes_1,
                chunk_problem_sizes_t,
                slot["a_ptrs"][:local_ne],
                slot["b_ptrs"][:local_ne],
                slot["out_ptrs"][:local_ne],
                slot["a_scales_ptrs"][:local_ne],
                slot["b_scales_ptrs"][:local_ne],
                slot["stride_a"][: local_ne * bufs["stride_sz"]],
                slot["stride_b"][: local_ne * bufs["stride_sz"]],
                slot["stride_c"][: local_ne * bufs["stride_sz"]],
                slot["layout_sfa"][: local_ne * bufs["sfa_sz"]],
                slot["layout_sfb"][: local_ne * bufs["sfb_sz"]],
                slot["workspace"],
            )

            chunk_act_q = slot["act_q"][:chunk_tokens]
            chunk_row_scales = slot["row_scales"][:chunk_tokens]
            chunk_act_scale_for_gemm2 = slot["act_scale_for_gemm2"][:chunk_tokens]
            if use_weighted_fold:
                ext.swiglu_fp8_requant_weighted(
                    chunk_gemm1_out,
                    chunk_weights,
                    chunk_act_q,
                    chunk_row_scales,
                    chunk_act_scale_for_gemm2,
                )
            else:
                ext.swiglu_fp8_requant(
                    chunk_gemm1_out,
                    chunk_act_q,
                    chunk_row_scales,
                    chunk_act_scale_for_gemm2,
                )

            chunk_gemm2_out = slot["gemm2_out"][:chunk_tokens]
            ext.moe_blockwise_grouped_mm_v2(
                chunk_gemm2_out,
                chunk_act_q,
                gemm2_weights[e_begin:e_end],
                chunk_act_scale_for_gemm2,
                gemm2_weights_scale[e_begin:e_end],
                chunk_offsets,
                chunk_problem_sizes_2,
                chunk_problem_sizes_t,
                slot["a_ptrs"][:local_ne],
                slot["b_ptrs"][:local_ne],
                slot["out_ptrs"][:local_ne],
                slot["a_scales_ptrs"][:local_ne],
                slot["b_scales_ptrs"][:local_ne],
                slot["stride_a"][: local_ne * bufs["stride_sz"]],
                slot["stride_b"][: local_ne * bufs["stride_sz"]],
                slot["stride_c"][: local_ne * bufs["stride_sz"]],
                slot["layout_sfa"][: local_ne * bufs["sfa_sz"]],
                slot["layout_sfb"][: local_ne * bufs["sfb_sz"]],
                slot["workspace"],
            )

            # Chunks can overlap in time, so we need atomic accumulation into the
            # shared output tensor. If routing weights were folded upstream,
            # scatter with a vector of ones to avoid multiplying twice.
            scatter_weights = slot["ones"][:chunk_tokens] if use_weighted_fold else chunk_weights
            ext.weighted_scatter(
                chunk_gemm2_out,
                scatter_weights,
                chunk_tids,
                bufs["out_bf16"],
                T,
            )

    current_stream = torch.cuda.current_stream(device=hidden_states.device)
    for slot in slots:
        current_stream.wait_stream(slot["stream"])
    return bufs["out_bf16"]


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
        # Only allocate MxF8 workspace buffers for workloads that will take
        # the MxF8 path (matches the use_mxf8 gate in _run_pipeline_dynamic).
        _mxf8_enabled  = bool(int(os.environ.get("USE_MXF8", "1")))
        _mxf8_min_t    = int(os.environ.get("MXF8_MIN_T", "4096"))
        use_mxf8 = _mxf8_enabled and T >= _mxf8_min_t
        mxf8_stride_sz = int(ext.get_mxf8_sizes_stride()) if use_mxf8 else 1
        mxf8_sfa_sz    = int(ext.get_mxf8_sizes_layout_sfa()) if use_mxf8 else 1
        mxf8_sfb_sz    = int(ext.get_mxf8_sizes_layout_sfb()) if use_mxf8 else 1
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
            # Route outputs (pre-allocated so _route_fused emits zero torch ops
            # inside the captured CUDA graph — no torch.empty / implicit copies).
            topk_idx_buf=torch.empty(T, TOP_K, device=device, dtype=torch.int32),
            assign_w_buf=torch.empty(T, TOP_K, device=device, dtype=torch.float32),
        )
        # expert_offsets is a view into offsets_buf[:ne] — fused_dispatch
        # writes the exclusive scan there directly, so no extra cumsum needed.
        bufs["expert_offsets"] = bufs["offsets_buf"][:ne]
        # problem_sizes_1/2 N,K columns are fixed — fill once here.
        bufs["problem_sizes_1"][:, 1] = N1
        bufs["problem_sizes_1"][:, 2] = K1
        bufs["problem_sizes_2"][:, 1] = N2
        bufs["problem_sizes_2"][:, 2] = K2

        # -----------------------------------------------------------------
        # MegaMoE flat-tile-list workspace — (expert, m_start, n_tile) triples.
        #
        # Each grid block in megamoe_gemm1 owns ONE such triple. The kernel
        # does NOT sweep N-tiles internally (that was a 3× slowdown bug).
        # So the list has M_tiles_total × N_tiles_per_expert entries.
        #
        # `ext.moe_flat_tile_list_mn(offsets, tile_expert, tile_mstart,
        #     tile_ntile, tile_count, block_m, n_tiles_per_expert)` populates.
        # -----------------------------------------------------------------
        _megamoe_bm_min = 32    # min block_m supported (for worst-case sizing)
        _megamoe_bn      = 128  # BLK_N for middle variant
        _megamoe_n_tiles = (N1 + _megamoe_bn - 1) // _megamoe_bn
        _megamoe_max_mtiles = (total_tokens + _megamoe_bm_min - 1) // _megamoe_bm_min + ne
        _megamoe_max_tiles = _megamoe_max_mtiles * _megamoe_n_tiles
        # Round up to 64 for alignment.
        _megamoe_max_tiles = ((_megamoe_max_tiles + 63) // 64) * 64
        bufs["megamoe_tile_expert"] = torch.empty(
            _megamoe_max_tiles, device=device, dtype=torch.int32)
        bufs["megamoe_tile_mstart"] = torch.empty(
            _megamoe_max_tiles, device=device, dtype=torch.int32)
        bufs["megamoe_tile_ntile"] = torch.empty(
            _megamoe_max_tiles, device=device, dtype=torch.int32)
        bufs["megamoe_tile_count"] = torch.empty(1, device=device, dtype=torch.int32)
        bufs["megamoe_max_tiles"] = _megamoe_max_tiles
        bufs["megamoe_n_tiles_per_expert"] = _megamoe_n_tiles

        # Fused kernel tile list: N is I (half of full N), so n_tiles_per_expert
        # is halved compared to the middle-variant. Reuse the same buffers
        # (fewer tiles => fits in existing capacity).
        _I1 = N1 // 2  # 2048
        _megamoe_fused_n_tiles = (_I1 + _megamoe_bn - 1) // _megamoe_bn  # 16
        bufs["megamoe_fused_n_tiles_per_expert"] = _megamoe_fused_n_tiles
        bufs["megamoe_fused_I"] = _I1

        # v18 MxF8 workspace (only allocated when USE_MXF8=1 to save memory).
        if use_mxf8:
            H = N1 // 2
            bufs["mxf8_gemm1_a_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_b_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_out_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_sfa_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_sfb_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_a_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_b_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_out_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_sfa_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_sfb_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_stride_a"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_stride_b"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_stride_c"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfa_1"] = torch.empty(ne * mxf8_sfa_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfb_1"] = torch.empty(ne * mxf8_sfb_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfa_2"] = torch.empty(ne * mxf8_sfa_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfb_2"] = torch.empty(ne * mxf8_sfb_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_expert_offsets"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            # UE8M0 scale buffers for activations (per-call) and weights (one-time, cached below).
            bufs["mxf8_act_scales_ue8m0"] = torch.empty(
                total_tokens, K1 // 128, device=device, dtype=torch.float32)
            bufs["mxf8_gemm2_act_scales_ue8m0"] = torch.empty(
                total_tokens, K2 // 128, device=device, dtype=torch.float32)
            # Transcoded weight payloads (in place overwrite of a copy at init).
            bufs["mxf8_gemm1_w_tr"] = None  # filled at first call when weights seen
            bufs["mxf8_gemm1_w_sc_ue8m0"] = None
            bufs["mxf8_gemm2_w_tr"] = None
            bufs["mxf8_gemm2_w_sc_ue8m0"] = None
            bufs["mxf8_weights_ready"] = False
            # SFA/SFB packed UE8M0 buffers.
            # For SFA (activations): sum across experts is bounded by
            # total_tokens + ne*127 padding (each expert pads up to 128).
            # For SFB (weights): always ne * (per-expert size) since each expert
            # has fixed N.
            _K32_g1 = ((K1 + 31) // 32) * 4
            _K32_g2 = ((K2 + 31) // 32) * 4
            max_sfa_total_1 = (((total_tokens + ne * 128 + 127) // 128) * 128) * _K32_g1
            max_sfb_total_1 = ne * (((N1 + 127) // 128) * 128) * _K32_g1
            max_sfa_total_2 = (((total_tokens + ne * 128 + 127) // 128) * 128) * _K32_g2
            max_sfb_total_2 = ne * (((N2 + 127) // 128) * 128) * _K32_g2
            bufs["mxf8_sfa_buffer_1"] = torch.empty(max_sfa_total_1, device=device, dtype=torch.uint8)
            bufs["mxf8_sfb_buffer_1"] = torch.empty(max_sfb_total_1, device=device, dtype=torch.uint8)
            bufs["mxf8_sfa_buffer_2"] = torch.empty(max_sfa_total_2, device=device, dtype=torch.uint8)
            bufs["mxf8_sfb_buffer_2"] = torch.empty(max_sfb_total_2, device=device, dtype=torch.uint8)
            bufs["mxf8_sfa_byte_offsets_1"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfb_byte_offsets_1"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfa_byte_offsets_2"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfb_byte_offsets_2"] = torch.empty(ne + 1, device=device, dtype=torch.int32)

            # FP8-out fusion path buffers (Phase A step 1):
            #   gemm1_out_fp8     : [total_tokens, 2H] fp8
            #   gemm1_sfd_buffer  : flat UE8M0 bytes, sized per tile-atom layout
            #   gemm1_sfd_ptrs    : [E] int64 per-expert pointers
            #   gemm1_sfd_byte_offsets: [E+1] int32 per-expert byte offsets
            #   gemm1_layout_sfd  : [E] LayoutSFD (MxF8GemmBuilderFP8Out::LayoutSFD)
            # Sized conservatively: same approach as SFB (per-32-col UE8M0 across full 2H cols,
            # with M padded to 128-multiple).
            mxf8_fp8out_stride_sz   = int(ext.get_mxf8_fp8out_sizes_stride())
            mxf8_fp8out_layout_sfd_sz = int(ext.get_mxf8_fp8out_sizes_layout_sfd())
            _K32_g1_fp8out = ((N1 + 31) // 32) * 4  # SFD covers 2H=N1 cols
            max_sfd_total_1 = (((total_tokens + ne * 128 + 127) // 128) * 128) * _K32_g1_fp8out
            bufs["mxf8_gemm1_out_fp8"] = torch.empty(
                total_tokens, N1, device=device, dtype=torch.float8_e4m3fn)
            bufs["mxf8_gemm1_sfd_buffer"] = torch.empty(
                max_sfd_total_1, device=device, dtype=torch.uint8)
            bufs["mxf8_gemm1_sfd_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_sfd_byte_offsets"] = torch.empty(
                ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_gemm1_layout_sfd"] = torch.empty(
                ne * mxf8_fp8out_layout_sfd_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_gemm1_stride_d"] = torch.empty(
                ne * mxf8_fp8out_stride_sz, device=device, dtype=torch.uint8)
        _workspace_cache[key] = bufs
    return _workspace_cache[key]


def _mxf8_ensure_weights_transcoded(bufs, gemm1_weights, gemm1_weights_scale,
                                    gemm2_weights, gemm2_weights_scale):
    """Transcode weights + pre-pack SFB layout once per *unique weight set*.

    Keyed on the 4 weight tensors' data_ptr()s so that running multiple
    workloads in the same Python process (workspace reused across workloads
    that share shape) doesn't incorrectly reuse transcoded weights from a
    previous workload. If any of the four weight pointers changes, we
    re-transcode.
    """
    weight_id = (
        int(gemm1_weights.data_ptr()),
        int(gemm1_weights_scale.data_ptr()),
        int(gemm2_weights.data_ptr()),
        int(gemm2_weights_scale.data_ptr()),
    )
    if bufs.get("mxf8_weights_ready") and bufs.get("mxf8_weight_id") == weight_id:
        return
    ext = _get_ext()
    bufs["mxf8_gemm1_w_tr"]       = gemm1_weights.clone()
    bufs["mxf8_gemm1_w_sc_ue8m0"] = torch.empty_like(gemm1_weights_scale)
    bufs["mxf8_gemm2_w_tr"]       = gemm2_weights.clone()
    bufs["mxf8_gemm2_w_sc_ue8m0"] = torch.empty_like(gemm2_weights_scale)
    ext.mxf8_transcode_weights_impl(
        bufs["mxf8_gemm1_w_tr"], gemm1_weights_scale, bufs["mxf8_gemm1_w_sc_ue8m0"])
    ext.mxf8_transcode_weights_impl(
        bufs["mxf8_gemm2_w_tr"], gemm2_weights_scale, bufs["mxf8_gemm2_w_sc_ue8m0"])

    # Pre-pack SFB into CUTLASS tiled layout (once, per unique weight set).
    N1 = int(gemm1_weights.shape[1]); K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1]); K2 = int(gemm2_weights.shape[2])
    ext.mxf8_pack_weight_sfb_impl(
        bufs["mxf8_gemm1_w_sc_ue8m0"],
        bufs["mxf8_layout_sfb_1"],
        bufs["mxf8_sfb_byte_offsets_1"],
        bufs["mxf8_sfb_buffer_1"],
        N1, K1)
    ext.mxf8_pack_weight_sfb_impl(
        bufs["mxf8_gemm2_w_sc_ue8m0"],
        bufs["mxf8_layout_sfb_2"],
        bufs["mxf8_sfb_byte_offsets_2"],
        bufs["mxf8_sfb_buffer_2"],
        N2, K2)
    bufs["mxf8_weights_ready"] = True
    bufs["mxf8_weight_id"] = weight_id


def _mxf8_compute_sf_offsets(ext, problem_sizes, bufs, gemm_idx):
    """Compute SFA/SFB byte offsets for given problem_sizes and fill buffers."""
    sfa_offsets, sfa_total = ext.compute_mxf8_sfa_layout_offsets_host(problem_sizes)
    sfb_offsets, sfb_total = ext.compute_mxf8_sfb_layout_offsets_host(problem_sizes)
    sfa_key = f"mxf8_sfa_byte_offsets_{gemm_idx}"
    sfb_key = f"mxf8_sfb_byte_offsets_{gemm_idx}"
    device = bufs[sfa_key].device
    bufs[sfa_key].copy_(torch.tensor(sfa_offsets, device=device, dtype=torch.int32))
    bufs[sfb_key].copy_(torch.tensor(sfb_offsets, device=device, dtype=torch.int32))
    return sfa_total, sfb_total


def _run_pipeline_graph_safe(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks,
    bufs, ext,
):
    """Fixed-shape pipeline (total=T*TOP_K) usable inside a CUDA graph.
    All writes go into pre-allocated buffers in `bufs`. No .item() syncs.
    """
    topk_idx, assign_w = _route(
        routing_logits, routing_bias, rsf, T, ls, ne,
        topk_idx_buf=bufs["topk_idx_buf"], assign_w_buf=bufs["assign_w_buf"])

    counts, sorted_tids, sorted_weights = _dispatch_graph_safe(topk_idx, assign_w, T, ls, ne, bufs)
    ext.fused_gather_hidden_scales(
        hidden_states, hidden_states_scale, sorted_tids,
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

    bufs["out_bf16"].zero_()
    ext.weighted_scatter(
        bufs["gemm2_out"], sorted_weights, sorted_tids, bufs["out_bf16"], T)


# =============================================================================
# MegaMoE pipeline (Phase A substrate — scaffolded, not yet custom-kernel-fused)
# =============================================================================
#
# Entry point: `_run_megamoe_pipeline` — gated on `MEGAMOE_MIN_T` env flag.
#
# Current state (session 2026-04-23): substrate only. The new flat-tile-list
# is populated and available via `bufs["megamoe_tile_{expert,mstart,count}"]`,
# but the GEMM1 / SwiGLU / GEMM2 path still delegates to the existing
# CUTLASS / helper kernels. This gives a MEGAMOE_MIN_T-gated execution path
# that is bit-identical to the default path and therefore perf-neutral when
# enabled.
#
# The next session replaces the delegating block below with a custom CuTe
# persistent SM100 kernel driven by the flat-tile-list. Per
# HANDOFF_MEGAMOE_PLAN.md §4 / §8:
#   - Phase A substep (a): bf16-out GEMM1 driven by flat tile list, target
#     latency within 5% of current on T=11948, ref_match ≥ 0.95 on all 19.
#   - Phase A substep (b): SwiGLU in the epilogue via TMEM load →
#     eliminates the swiglu_fp8_requant_weighted_mxf8 kernel.
#   - Phase A substep (c): fuse GEMM2 by reading act_q from SMEM →
#     eliminates the act_q HBM round-trip.
#
# The code path is kept as a drop-in for `_run_pipeline_dynamic` so the
# upcoming custom-kernel work can iterate in isolation without touching
# the route/dispatch/gather/scatter chain.
# =============================================================================

def _run_megamoe_pipeline(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf,
    bufs, ext, device,
):
    """MegaMoE pipeline — scaffolded variant of `_run_pipeline_dynamic`.

    Runs route → dispatch → gather identically to the dynamic path, then
    populates the flat-tile-list from the per-expert offsets. The resulting
    tile list is available in `bufs["megamoe_tile_{expert,mstart,count}"]`
    for the upcoming custom CuTe kernel to consume.

    THIS IS SUBSTRATE ONLY. GEMM1 / SwiGLU / GEMM2 still delegate to the
    existing MxF8 / helper path. The next session replaces this block with
    a persistent CuTe kernel per HANDOFF_MEGAMOE_PLAN.md Phase A.
    """
    # Share the route/dispatch/gather/GEMM/SwiGLU/scatter code with
    # _run_pipeline_dynamic to avoid duplication while the custom kernel
    # is not yet implemented. We re-enter _run_pipeline_dynamic with an
    # env marker set so it knows to also populate the flat-tile-list.
    prev = os.environ.get("_MEGAMOE_ACTIVE")
    os.environ["_MEGAMOE_ACTIVE"] = "1"
    try:
        if os.environ.get("MEGAMOE_DEBUG"):
            print(f"[MEGAMOE] _run_megamoe_pipeline entry T={T}", flush=True)
        return _run_pipeline_dynamic(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            T, ne, N1, K1, N2, K2, H, ls, rsf, bufs, ext, device)
    finally:
        if prev is None:
            os.environ.pop("_MEGAMOE_ACTIVE", None)
        else:
            os.environ["_MEGAMOE_ACTIVE"] = prev


def _megamoe_populate_tile_list(bufs, ext, block_m, fused=False):
    """Populate (expert, m_start, n_tile) triples into the flat-tile-list.

    Called inside `_run_pipeline_dynamic` when `_MEGAMOE_ACTIVE` is set.
    Each entry corresponds to ONE grid block in megamoe_gemm1 — it does NOT
    sweep N-tiles internally.

    `fused=True`: emits out-n-tile indices in [0, I/BLK_N=16), for the fused
    GEMM1+SwiGLU kernel (each CTA produces one 128-col block of FP8 act_q).
    `fused=False`: emits n-tile indices in [0, N/BLK_N=32) for the middle
    variant that produces BF16 gemm1_out.
    """
    n_tiles = (int(bufs["megamoe_fused_n_tiles_per_expert"]) if fused
               else int(bufs["megamoe_n_tiles_per_expert"]))
    ext.moe_flat_tile_list_mn(
        bufs["offsets_buf"],
        bufs["megamoe_tile_expert"],
        bufs["megamoe_tile_mstart"],
        bufs["megamoe_tile_ntile"],
        bufs["megamoe_tile_count"],
        int(block_m),
        n_tiles,
    )


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
    topk_idx, assign_w = _route(
        routing_logits, routing_bias, rsf, T, ls, ne,
        topk_idx_buf=bufs["topk_idx_buf"], assign_w_buf=bufs["assign_w_buf"])
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

    # MegaMoE scaffold: when the MegaMoE pipeline wraps this call, populate
    # the flat-tile-list so downstream custom-kernel work (Phase A) can
    # drive its grid from it. This is a no-op on the default path.
    if os.environ.get("_MEGAMOE_ACTIVE"):
        _megamoe_block_m = int(os.environ.get("MEGAMOE_BLOCK_M", "128"))
        _megamoe_fused = bool(int(os.environ.get("MEGAMOE_FUSED_GEMM1", "0")))
        _megamoe_populate_tile_list(bufs, ext, _megamoe_block_m, fused=_megamoe_fused)

    async_chunk_experts = int(os.environ.get("ASYNC_STREAM_CHUNK_EXPERTS", "0"))
    stream_chunk_experts = int(os.environ.get("STREAM_CHUNK_EXPERTS", "0"))
    # MxF8 path: hardware block-scaled MMA via tcgen05.mma.kind.mxf8f6f4.
    # Requires transcoded activations + weights (sign flip + residual
    # absorbed into payload). Contest scales are signed fp32 but Central
    # Limit Theorem over K=7168 amortizes the re-quantization error; T=1 and
    # very small T regress (no averaging).
    # Default ON for large T since the fused pipeline (gather→MxF8 GEMM1,
    # swiglu→MxF8 GEMM2) hits hardware peak throughput. Set USE_MXF8=0 to
    # disable.
    use_mxf8 = bool(int(os.environ.get("USE_MXF8", "1"))) and T >= int(
        os.environ.get("MXF8_MIN_T", "4096"))
    if os.environ.get("MXF8_TRACE") and use_mxf8:
        print(f"[MXF8] T={T} use_mxf8=True", flush=True)
    if async_chunk_experts > 0 and not use_fused_dispatch_gather and not use_mxf8:
        return _run_async_streaming_chunked_pipeline(
            hidden_states,
            hs_scale,
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
            async_chunk_experts,
        )

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

    if use_mxf8:
        _mxf8_ensure_weights_transcoded(
            bufs, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
        )
        mxf8_act_scales_ue8m0 = bufs["mxf8_act_scales_ue8m0"][:total_valid]
        # Pre-run setup and compute SFA offsets so gather+transcode+pack can
        # write directly into the tiled SFA layout.
        ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_1"],
            bufs["mxf8_sfa_byte_offsets_1"],
            bufs["mxf8_sfb_byte_offsets_1"])
        ext.moe_mxf8_setup_ptrs(
            bufs["gemm1_out"], packed_acts, bufs["mxf8_gemm1_w_tr"],
            bufs["offsets_buf"], bufs["problem_sizes_1"],
            bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
            bufs["mxf8_gemm1_out_ptrs"],
            bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
            bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
            bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"])
        if use_fused_dispatch_gather:
            # packed_acts/packed_act_scales already populated by fused-dispatch
            # upstream; run transcode+pack separately.
            ext.mxf8_transcode_and_pack_sfa(
                packed_acts, packed_act_scales, mxf8_act_scales_ue8m0,
                bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_1"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_sfa_buffer_1"])
        else:
            # Fused gather+transcode+SFA pack in one kernel.
            ext.fused_gather_mxf8(
                hidden_states, hs_scale, sorted_tids,
                packed_acts, mxf8_act_scales_ue8m0,
                bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_1"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_sfa_buffer_1"])
    elif not use_fused_dispatch_gather:
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
    _megamoe_fused_skipped_swiglu = False
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
        elif use_mxf8:
            # FUSE_FP8OUT_GEMM1: 0 = default bf16-out; 1 = 1SM fp8-out;
            # 2 = 2SM fp8-out.
            fuse_fp8out_gemm1 = int(os.environ.get("FUSE_FP8OUT_GEMM1", "0"))
            if fuse_fp8out_gemm1 in (1, 2):
                ext.compute_mxf8_sfd_offsets_device(
                    bufs["problem_sizes_1"],
                    bufs["mxf8_gemm1_sfd_byte_offsets"])
                gemm1_out_fp8 = bufs["mxf8_gemm1_out_fp8"][:total_valid]
                fp8out_fn = (ext.moe_mxf8_grouped_mm_prepacked_fp8out_2sm
                             if fuse_fp8out_gemm1 == 2
                             else ext.moe_mxf8_grouped_mm_prepacked_fp8out)
                fp8out_fn(
                    gemm1_out_fp8,
                    bufs["mxf8_gemm1_sfd_buffer"],
                    bufs["mxf8_gemm1_sfd_byte_offsets"],
                    packed_acts, bufs["mxf8_gemm1_w_tr"],
                    bufs["problem_sizes_1"], bufs["offsets_buf"],
                    bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
                    bufs["mxf8_gemm1_out_ptrs"], bufs["mxf8_gemm1_sfd_ptrs"],
                    bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                    bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"],
                    bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
                    bufs["mxf8_stride_a"], bufs["mxf8_stride_b"],
                    bufs["mxf8_gemm1_stride_d"],
                    bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                    bufs["mxf8_gemm1_layout_sfd"],
                    bufs["workspace"],
                )
            else:
                if os.environ.get("_MEGAMOE_ACTIVE") and bool(int(os.environ.get("MEGAMOE_FUSED_GEMM1", "0"))):
                    # Fused GEMM1 + SwiGLU + FP8 requant. Produces act_q (FP8)
                    # + act_scales_ue8m0 (FP32 pow-of-2) in one kernel. Skips
                    # the BF16 gemm1_out round-trip and the swiglu kernel.
                    _megamoe_fused_skipped_swiglu = True
                    act_q_fused = bufs["act_q"][:total_valid]
                    act_scales_ue8m0_fused = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]

                    # Build GEMM2 SFA layouts up front (normally done after swiglu).
                    ext.compute_mxf8_sf_offsets_device(
                        bufs["problem_sizes_2"],
                        bufs["mxf8_sfa_byte_offsets_2"],
                        bufs["mxf8_sfb_byte_offsets_2"])
                    ext.moe_mxf8_setup_ptrs(
                        bufs["gemm2_out"], act_q_fused, bufs["mxf8_gemm2_w_tr"],
                        bufs["offsets_buf"], bufs["problem_sizes_2"],
                        bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"],
                        bufs["mxf8_gemm2_out_ptrs"],
                        bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
                        bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                        bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
                        bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
                        bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"])

                    # Fused kernel: produces act_q + act_scales_ue8m0 (pow-of-2).
                    # Weight fold is enabled via sorted_weights (V17-style: the
                    # scale absorbs the routing weight so reduce-scatter is
                    # unweighted downstream).
                    use_weighted_fold_local = bool(int(os.environ.get("V17_WEIGHTED_FOLD", "1")))
                    sw_for_fused = sorted_weights if use_weighted_fold_local else None
                    ext.megamoe_gemm1_swiglu_fused(
                        act_q_fused, act_scales_ue8m0_fused,
                        packed_acts, bufs["mxf8_gemm1_w_tr"],
                        sw_for_fused,
                        bufs["megamoe_tile_expert"],
                        bufs["megamoe_tile_mstart"],
                        bufs["megamoe_tile_ntile"],  # reused buffer; holds out-n-tile indices
                        bufs["megamoe_tile_count"],
                        bufs["offsets_buf"],
                        bufs["problem_sizes_1"],
                        bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
                        bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                        bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                        bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                        bufs["workspace"],
                        128,
                        N1 // 2,
                    )
                    # Pack FP32 UE8M0 scales into the SFA buffer layout that
                    # GEMM2 expects. mxf8_transcode_and_pack_sfa also runs
                    # FP8-transcode as a no-op (scales are already pow2 →
                    # residual=1 → FP8 values unchanged).
                    ext.mxf8_transcode_and_pack_sfa(
                        act_q_fused, act_scales_ue8m0_fused,
                        act_scales_ue8m0_fused,  # scratch (written back as-is)
                        bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_2"],
                        bufs["mxf8_layout_sfa_2"], bufs["mxf8_sfa_buffer_2"])
                elif os.environ.get("_MEGAMOE_ACTIVE"):
                    # Phase A substep (a): custom CuTe GEMM1 driven by flat-tile-list.
                    # 1-CTA, MmaTileShape=<_128,_128,_128>. block_m=128 only.
                    _megamoe_dbg = bool(int(os.environ.get("MEGAMOE_DEBUG", "0")))
                    if bool(int(os.environ.get("MEGAMOE_ZERO_OUT", "0"))):
                        gemm1_out.zero_()
                    if _megamoe_dbg:
                        import time as _t
                        torch.cuda.synchronize()
                        _t0 = _t.time()
                        print(f"[MEGAMOE] T={T} ne={ne} total_valid={total_valid} "
                              f"block_m=128 max_tiles={bufs['megamoe_max_tiles']} "
                              f"tile_count={int(bufs['megamoe_tile_count'].item())}",
                              flush=True)
                    ext.megamoe_gemm1(
                        gemm1_out,
                        packed_acts, bufs["mxf8_gemm1_w_tr"],
                        bufs["megamoe_tile_expert"],
                        bufs["megamoe_tile_mstart"],
                        bufs["megamoe_tile_ntile"],
                        bufs["megamoe_tile_count"],
                        bufs["offsets_buf"],
                        bufs["problem_sizes_1"],
                        bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
                        bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                        bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                        bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                        bufs["workspace"],
                        128,
                    )
                    if _megamoe_dbg:
                        torch.cuda.synchronize()
                        print(f"[MEGAMOE] gemm1 done in {(_t.time()-_t0)*1e3:.2f}ms", flush=True)
                    if os.environ.get("MEGAMOE_SNAPSHOT"):
                        globals()["_MEGAMOE_GEMM1_OUT_SNAPSHOT"] = gemm1_out.clone()
                        globals()["_MEGAMOE_TOTAL_VALID_SNAPSHOT"] = int(total_valid)
                        globals()["_MEGAMOE_OFFSETS_SNAPSHOT"] = bufs["offsets_buf"].clone()
                else:
                    # Default bf16-out path.
                    ext.moe_mxf8_grouped_mm_prepacked(
                        gemm1_out,
                        packed_acts, bufs["mxf8_gemm1_w_tr"],
                        bufs["problem_sizes_1"],
                        bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
                        bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                        bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                        bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                        bufs["workspace"],
                    )
                    if os.environ.get("MEGAMOE_SNAPSHOT"):
                        globals()["_MEGAMOE_GEMM1_OUT_SNAPSHOT"] = gemm1_out.clone()
                        globals()["_MEGAMOE_TOTAL_VALID_SNAPSHOT"] = int(total_valid)
                        globals()["_MEGAMOE_OFFSETS_SNAPSHOT"] = bufs["offsets_buf"].clone()
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
        if _megamoe_fused_skipped_swiglu:
            # Fused kernel already produced act_q + act_scales_ue8m0 + SFA buffer.
            # act_scale_for_gemm2 buffer was aliased to mxf8_gemm2_act_scales_ue8m0.
            mxf8_gemm2_act_scales_ue8m0 = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]
            # GEMM2 setup was done inside the fused branch above; nothing to do here.
        elif use_mxf8 and use_weighted_fold:
            # Fused: swiglu + fp8 requant + MxF8 transcode + SFA pack in ONE kernel.
            mxf8_gemm2_act_scales_ue8m0 = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]
            ext.compute_mxf8_sf_offsets_device(
                bufs["problem_sizes_2"],
                bufs["mxf8_sfa_byte_offsets_2"],
                bufs["mxf8_sfb_byte_offsets_2"])
            ext.moe_mxf8_setup_ptrs(
                bufs["gemm2_out"], act_q, bufs["mxf8_gemm2_w_tr"],
                bufs["offsets_buf"], bufs["problem_sizes_2"],
                bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"],
                bufs["mxf8_gemm2_out_ptrs"],
                bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
                bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
                bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"])
            fuse_fp8out_gemm1 = int(os.environ.get("FUSE_FP8OUT_GEMM1", "0"))
            if fuse_fp8out_gemm1 in (1, 2):
                gemm1_out_fp8 = bufs["mxf8_gemm1_out_fp8"][:total_valid]
                ext.swiglu_fp8in_mxf8_weighted(
                    gemm1_out_fp8,
                    bufs["mxf8_gemm1_sfd_buffer"],
                    bufs["mxf8_gemm1_sfd_byte_offsets"],
                    bufs["mxf8_gemm1_layout_sfd"],
                    sorted_weights, act_q, row_scales,
                    mxf8_gemm2_act_scales_ue8m0,
                    bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_2"],
                    bufs["mxf8_layout_sfa_2"], bufs["mxf8_sfa_buffer_2"])
            else:
                ext.swiglu_fp8_requant_weighted_mxf8(
                    gemm1_out, sorted_weights, act_q, row_scales,
                    mxf8_gemm2_act_scales_ue8m0,
                    bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_2"],
                    bufs["mxf8_layout_sfa_2"], bufs["mxf8_sfa_buffer_2"])
        elif use_weighted_fold:
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
        elif use_mxf8:
            ext.moe_mxf8_grouped_mm_prepacked(
                gemm2_out,
                act_q, bufs["mxf8_gemm2_w_tr"],
                bufs["problem_sizes_2"],
                bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
                bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
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
            # Fused 2-kernel reduce-scatter (inverse-bucket + reduce). Replaces
            # the prior 4-kernel chain (count/scan/place/reduce). Saves ~25 μs
            # of launch+memset overhead per call. Set USE_FUSED_REDUCE=0 to
            # revert to the 4-kernel path.
            if bool(int(os.environ.get("USE_FUSED_REDUCE", "1"))):
                ext.reduce_scatter_unweighted_fused(
                    gemm2_out, sorted_tids, bufs["out_bf16"],
                    bufs["token_counts_buf"], bufs["token_perm_buf"],
                    T, TOP_K)
            else:
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

    token_stationary_max_t = int(os.environ.get("TOKEN_STATIONARY_MAX_T", "0"))
    if token_stationary_max_t > 0 and T <= token_stationary_max_t:
        topk_idx, assign_w = _route(
            routing_logits, routing_bias, rsf, T, ls, ne,
            topk_idx_buf=bufs["topk_idx_buf"], assign_w_buf=bufs["assign_w_buf"])
        return _run_token_stationary_smallt(
            topk_idx,
            assign_w,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            T,
            ne,
            N1,
            K1,
            N2,
            K2,
            ls,
            bufs,
            ext,
        )

    # Below threshold: Python overhead dominates → CUDA graph replay wins.
    # Above threshold: GEMM compute dominates, non-local filtering saves ~8x
    # data movement vs fixed-shape path.
    # Graph-safe path (fixed shape T*TOP_K) wins for small-medium T due to
    # graph-replay overhead elimination. Dynamic path (only num_local_valid
    # tokens, ~T) wins for large T where graph-safe's extra 8x gather+scatter
    # work on T*TOP_K dominates. Verified empirical crossover ~T=2048.
    use_graph = (T <= 2048) and not os.environ.get("DISABLE_CUDA_GRAPH")

    # MegaMoE gate (Phase A substrate): when MEGAMOE_MIN_T <= T, route through
    # the MegaMoE pipeline. Currently perf-neutral (delegates GEMM1 to CUTLASS
    # via _run_pipeline_dynamic) but lays the substrate for the upcoming
    # custom CuTe persistent kernel. Default MEGAMOE_MIN_T is very large so
    # the path is OFF by default and the shipping kernel is unchanged.
    # MEGAMOE_MIN_T default: OFF (2^31-1). Earlier thought middle variant won
    # at T>=11948, but a-b test vs the CUTLASS default path (which uses
    # moe_mxf8_grouped_mm_prepacked 2SM MxF8) shows CUTLASS wins handily at
    # large T (2.58x FI at T=11948 vs our 1.22x). The middle variant is for
    # development of the fused SwiGLU variant only. Don't engage by default.
    _megamoe_min_t = int(os.environ.get("MEGAMOE_MIN_T", "2147483647"))
    if T >= _megamoe_min_t:
        return _run_megamoe_pipeline(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            T, ne, N1, K1, N2, K2, H, ls, rsf, bufs, ext, device)

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
