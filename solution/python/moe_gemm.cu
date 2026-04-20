/*
 * FP8 blockscaled rcgrouped GEMM for NVIDIA B200 (SM100).
 *
 * This is the validated CUTLASS contract we tested on B200:
 *   - blockscaled rcgrouped grouped GEMM
 *   - pointer-array activations per expert
 *   - contiguous per-expert weight bank
 *   - contest-style [M/128, K/128] / [T/128, K/128] scales adapted into
 *     CUTLASS's internal K/32 ue8m0 blockscale layout
 *
 * Core kernel logic is our own; CUTLASS headers are used as building blocks.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>

#include <cute/tensor.hpp>

using namespace cute;

namespace {

using ProblemShape = cutlass::gemm::MoEProblemShape<Shape<int, int, int>>;
using ElementInput = cutlass::float_e4m3_t;
using ElementA = cutlass::mx_float8_t<ElementInput>;
using ElementB = cutlass::mx_float8_t<ElementInput>;
using ElementC = cutlass::bfloat16_t;
using ElementD = ElementC;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;

struct MMA1SMConfig {
  using MmaTileShape = Shape<_128, _256, _128>;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    typename MMA1SMConfig::MmaTileShape,
    ClusterShape,
    Shape<_128, _64>,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC*,
    AlignmentC,
    ElementD,
    LayoutC*,
    AlignmentD,
    typename MMA1SMConfig::EpilogueSchedule
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB*,
    AlignmentB,
    ElementAccumulator,
    typename MMA1SMConfig::MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    typename MMA1SMConfig::KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = typename Gemm::GemmKernel::ElementSF;

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::stringstream ss;
    ss << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

torch::Tensor make_device_pointer_array(std::vector<int64_t> const& ptrs, torch::Device device) {
  auto out = torch::empty(
      {static_cast<int64_t>(ptrs.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(device));
  check_cuda(
      cudaMemcpy(out.data_ptr<int64_t>(), ptrs.data(), sizeof(int64_t) * ptrs.size(), cudaMemcpyHostToDevice),
      "cudaMemcpy(pointer array)");
  return out;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> moe_blockscaled_rcgrouped_gemm(
    torch::Tensor weights,                    // [E, M, K] fp8
    torch::Tensor weight_block_scales,        // [E, M/128, K/128] float32 CPU or CUDA
    torch::Tensor activations_ptrs,           // [E] int64 device pointers to [tokens_e, K] fp8
    torch::Tensor activation_block_scales,    // [E, T_max, K/128] float32 CPU or CUDA (per-token)
    torch::Tensor tokens_per_expert,          // [E] int32 CUDA
    torch::Tensor tokens_per_expert_cpu       // [E] int32 CPU
) {
  TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
  TORCH_CHECK(activations_ptrs.is_cuda(), "activations_ptrs must be CUDA");
  TORCH_CHECK(tokens_per_expert.is_cuda(), "tokens_per_expert must be CUDA");
  TORCH_CHECK(tokens_per_expert_cpu.device().is_cpu(), "tokens_per_expert_cpu must be CPU");
  if (weight_block_scales.is_cuda()) {
    weight_block_scales = weight_block_scales.to(torch::kCPU);
  }
  if (activation_block_scales.is_cuda()) {
    activation_block_scales = activation_block_scales.to(torch::kCPU);
  }
  TORCH_CHECK(weights.dim() == 3, "weights must be [E, M, K]");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat8_e4m3fn, "weights must be float8_e4m3fn");
  TORCH_CHECK(weight_block_scales.dim() == 3, "weight_block_scales must be [E, M/128, K/128]");
  TORCH_CHECK(weight_block_scales.scalar_type() == torch::kFloat32, "weight_block_scales must be float32");
  TORCH_CHECK(activations_ptrs.scalar_type() == torch::kInt64, "activations_ptrs must be int64");
  TORCH_CHECK(activation_block_scales.dim() == 3, "activation_block_scales must be [E, T_max, K/128]");
  TORCH_CHECK(activation_block_scales.scalar_type() == torch::kFloat32, "activation_block_scales must be float32");

  int64_t E = weights.size(0);
  int64_t M = weights.size(1);
  int64_t K = weights.size(2);
  int64_t K128 = K / 128;
  int64_t M128 = M / 128;
  TORCH_CHECK(M % 128 == 0, "M must be divisible by 128");
  TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");
  TORCH_CHECK(activations_ptrs.size(0) == E, "ptr count mismatch");
  TORCH_CHECK(tokens_per_expert.size(0) == E, "device counts mismatch");
  TORCH_CHECK(tokens_per_expert_cpu.size(0) == E, "cpu counts mismatch");
  TORCH_CHECK(weight_block_scales.size(0) == E, "weight_block_scales E mismatch");
  TORCH_CHECK(weight_block_scales.size(1) == M128, "weight_block_scales M mismatch");
  TORCH_CHECK(weight_block_scales.size(2) == K128, "weight_block_scales K mismatch");

  auto counts_cpu = tokens_per_expert_cpu.contiguous();
  auto* counts_ptr = counts_cpu.data_ptr<int32_t>();
  auto weight_scales_cpu = weight_block_scales.contiguous();
  auto act_scales_cpu = activation_block_scales.contiguous();

  int64_t max_tokens = 0;
  std::vector<int64_t> out_offsets_host(E + 1, 0);
  std::vector<int64_t> sfb_offsets_host(E + 1, 0);
  for (int64_t e = 0; e < E; ++e) {
    int64_t t = counts_ptr[e];
    TORCH_CHECK(t >= 0, "negative tokens_per_expert");
    max_tokens = std::max(max_tokens, t);
    out_offsets_host[e + 1] = out_offsets_host[e] + M * t;
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(static_cast<int32_t>(M),
                         static_cast<int32_t>(t),
                         static_cast<int32_t>(K),
                         1));
    sfb_offsets_host[e + 1] = sfb_offsets_host[e] + size(filter_zeros(layout_SFB));
  }
  TORCH_CHECK(
      activation_block_scales.size(0) == E,
      "activation_block_scales E mismatch");
  TORCH_CHECK(
      activation_block_scales.size(1) >= max_tokens,
      "activation_block_scales T_max mismatch");
  TORCH_CHECK(
      activation_block_scales.size(2) == K128,
      "activation_block_scales K mismatch");

  if (out_offsets_host.back() == 0) {
    auto empty = torch::empty(
        {0},
        torch::TensorOptions().dtype(torch::kBFloat16).device(weights.device()));
    auto offsets = torch::zeros(
        {E + 1},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return {empty, offsets};
  }

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int32_t>(M),
                       static_cast<int32_t>(max_tokens),
                       static_cast<int32_t>(K),
                       static_cast<int32_t>(E)));
  int64_t sfa_elems = size(filter_zeros(layout_SFA));

  auto ptr_opts = torch::TensorOptions().dtype(torch::kInt64).device(weights.device());
  auto out_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(weights.device());
  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(weights.device());

  auto D_flat = torch::zeros({out_offsets_host.back()}, out_opts);
  auto C_flat = torch::zeros({out_offsets_host.back()}, out_opts);
  auto SFA = torch::empty({sfa_elems}, byte_opts);
  auto SFB_flat = torch::empty({sfb_offsets_host.back()}, byte_opts);

  {
    std::vector<ElementSF> sfa_host(sfa_elems);
    auto tensor_sfa = make_tensor(sfa_host.data(), layout_SFA);
    auto* weight_scale_ptr = weight_scales_cpu.data_ptr<float>();
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t m = 0; m < M; ++m) {
        int64_t mb = m / 128;
        for (int64_t kb = 0; kb < K128; ++kb) {
          float scale = weight_scale_ptr[(e * M128 + mb) * K128 + kb];
          ElementSF sf(scale);
          for (int sub = 0; sub < 4; ++sub) {
            int64_t k = kb * 128 + sub * 32;
            tensor_sfa(m, k, e) = sf;
          }
        }
      }
    }
    check_cuda(
        cudaMemcpy(SFA.data_ptr(), sfa_host.data(), sfa_elems * sizeof(ElementSF), cudaMemcpyHostToDevice),
        "cudaMemcpy(SFA)");

    auto* act_scale_ptr = act_scales_cpu.data_ptr<float>();
    int64_t T_max = activation_block_scales.size(1);
    std::vector<ElementSF> sfb_host(sfb_offsets_host.back());
    for (int64_t e = 0; e < E; ++e) {
      int64_t t = counts_ptr[e];
      auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
          cute::make_shape(static_cast<int32_t>(M),
                           static_cast<int32_t>(t),
                           static_cast<int32_t>(K),
                           1));
      auto tensor_sfb = make_tensor(sfb_host.data() + sfb_offsets_host[e], layout_SFB);
      for (int64_t n = 0; n < t; ++n) {
        for (int64_t kb = 0; kb < K128; ++kb) {
          float scale = act_scale_ptr[(e * T_max + n) * K128 + kb];
          ElementSF sf(scale);
          for (int sub = 0; sub < 4; ++sub) {
            int64_t k = kb * 128 + sub * 32;
            tensor_sfb(n, k, 0) = sf;
          }
        }
      }
    }
    check_cuda(
        cudaMemcpy(SFB_flat.data_ptr(), sfb_host.data(), sfb_offsets_host.back() * sizeof(ElementSF), cudaMemcpyHostToDevice),
        "cudaMemcpy(SFB)");
  }

  auto* c_base = reinterpret_cast<ElementC*>(C_flat.data_ptr());
  auto* d_base = reinterpret_cast<ElementD*>(D_flat.data_ptr());
  auto* sfb_base = reinterpret_cast<ElementSF*>(SFB_flat.data_ptr());

  std::vector<int64_t> c_ptrs_host(E);
  std::vector<int64_t> d_ptrs_host(E);
  std::vector<int64_t> sfb_ptrs_host(E);
  for (int64_t e = 0; e < E; ++e) {
    c_ptrs_host[e] = reinterpret_cast<int64_t>(c_base + out_offsets_host[e]);
    d_ptrs_host[e] = reinterpret_cast<int64_t>(d_base + out_offsets_host[e]);
    sfb_ptrs_host[e] = reinterpret_cast<int64_t>(sfb_base + sfb_offsets_host[e]);
  }

  auto C_ptrs = make_device_pointer_array(c_ptrs_host, weights.device());
  auto D_ptrs = make_device_pointer_array(d_ptrs_host, weights.device());
  auto SFB_ptrs = make_device_pointer_array(sfb_ptrs_host, weights.device());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = weights.get_device();
  hw_info.sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  hw_info.cluster_shape = dim3(2, 1, 1);
  hw_info.cluster_shape_fallback = dim3(2, 1, 1);

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args{};
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.dBeta = {_0{}, _0{}, 0};

  typename Gemm::GemmKernel::TileSchedulerArguments scheduler{};
  scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {static_cast<int32_t>(M),
       static_cast<int32_t>(max_tokens),
       static_cast<int32_t>(K),
       static_cast<int32_t>(E),
       tokens_per_expert.data_ptr<int32_t>()},
      {reinterpret_cast<typename Gemm::ElementA*>(weights.data_ptr()),
       reinterpret_cast<typename Gemm::ElementB const**>(activations_ptrs.data_ptr<int64_t>()),
       reinterpret_cast<ElementSF const*>(SFA.data_ptr()),
       reinterpret_cast<ElementSF const**>(SFB_ptrs.data_ptr<int64_t>())},
      {fusion_args,
       reinterpret_cast<ElementC const**>(C_ptrs.data_ptr<int64_t>()),
       nullptr,
       reinterpret_cast<ElementD**>(D_ptrs.data_ptr<int64_t>()),
       nullptr},
      hw_info,
      scheduler};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "can_implement failed: ", int(status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(weights.device()));

  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "initialize failed: ", int(status));

  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "run failed: ", int(status));
  check_cuda(cudaGetLastError(), "kernel launch");

  auto offsets = torch::empty(
      {E + 1},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  std::memcpy(offsets.data_ptr<int64_t>(), out_offsets_host.data(), sizeof(int64_t) * (E + 1));
  return {D_flat, offsets};
}

std::string test_compilation() {
  std::stringstream ss;
  ss << "MoE blockscaled rcgrouped kernel compiled successfully\n";
  ss << "sizeof(Gemm::ElementA)=" << sizeof(typename Gemm::ElementA) << "\n";
  ss << "sizeof(Gemm::ElementB)=" << sizeof(typename Gemm::ElementB) << "\n";
  ss << "sizeof(ElementSF)=" << sizeof(ElementSF) << "\n";
  ss << "AlignmentA=" << AlignmentA << " AlignmentB=" << AlignmentB << "\n";
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_blockscaled_rcgrouped_gemm", &moe_blockscaled_rcgrouped_gemm, "MoE blockscaled rcgrouped FP8 GEMM via CUTLASS SM100");
  m.def("test_compilation", &test_compilation, "Test compilation");
}
