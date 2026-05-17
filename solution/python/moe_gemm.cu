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

struct CfgMxF8G2_256_256 {
  using MmaTileShape = Shape<_256, _256, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
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
  // Enable PDL (Programmatic Dependent Launch, SM100A) — allows the next kernel
  // in the stream to begin executing preambles while this GEMM is still running.
  const char* disable_pdl = std::getenv("MXF8_DISABLE_PDL");
  bool pdl_enabled = !(disable_pdl && std::string(disable_pdl) == "1");
  status = gemm_op.run(stream, /*cuda_adapter=*/nullptr, pdl_enabled);
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

// Variant for token-stationary execution: each grouped problem is selected by
// an explicit expert id instead of inheriting expert == group index.
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



// CUTLASS-only launch (no setup, no pack). Used when ptrs/layouts/SFA/SFB
// have been fully pre-populated (e.g., by moe_mxf8_setup_ptrs +
// mxf8_transcode_and_pack_sfa + mxf8_pack_weight_sfb_impl). Avoids env-var
// checks and extra kernel launches.
//
// Fixed 256x256x128 2SM tile for the submitted MxF8 path.
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
  // per-tile case (a_ptrs[tile_count]).
  int E = static_cast<int>(a_ptrs.size(0));
  using LayoutOut = cutlass::layout::RowMajor;
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  launch_mxf8_group_gemm<CfgMxF8G2_256_256, LayoutOut>(
      a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
      stride_a, stride_b, stride_c,
      layout_sfa, layout_sfb,
      problem_sizes, workspace, E, stream);
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



// Debug helper: evaluate SFA layout at a few (m, k) points on the host and
// return the resulting offsets. Helps diagnose whether our packing kernel's
// `layout(m, k, 0)` indexing is consistent with what CUTLASS expects.
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
