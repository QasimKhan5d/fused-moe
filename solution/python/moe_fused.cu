#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef TRIGGER_PDL
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #define TRIGGER_PDL() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
  #else
    #define TRIGGER_PDL() ((void)0)
  #endif
#endif

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
std::tuple<std::vector<int32_t>, int64_t> compute_mxf8_sfa_layout_offsets_host(torch::Tensor const&);
std::tuple<std::vector<int32_t>, int64_t> compute_mxf8_sfb_layout_offsets_host(torch::Tensor const&);
void compute_mxf8_sf_offsets_device(torch::Tensor const&, torch::Tensor&, torch::Tensor&);
int64_t get_mxf8_sizes_stride();
int64_t get_mxf8_sizes_layout_sfa();
int64_t get_mxf8_sizes_layout_sfb();

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
  // PDL completion signal. It is harmless for normal stream launches and lets
  // this kernel remain usable with dependent-launch experiments.
  TRIGGER_PDL();
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
    // cudaLaunchKernelEx-based dependent launch was benchmarked on the full
    // workload set and regressed; the standard launch path is faster here.
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_blockwise_grouped_mm_v2", &moe_blockwise_grouped_mm_v2);
  m.def("get_sizes", &get_sizes);
  m.def("get_workspace_size", &get_workspace_size);
  m.def("swiglu_fp8_requant", &swiglu_fp8_requant);
  m.def("swiglu_fp8_requant_weighted", &swiglu_fp8_requant_weighted);
  m.def("weighted_scatter", &weighted_scatter);
  m.def("reduce_scatter_unweighted_fused", &reduce_scatter_unweighted_fused);
  m.def("fused_route_topk", &fused_route_topk);
  m.def("fused_gather_hidden_scales", &fused_gather_hidden_scales);
  m.def("fused_dispatch", &fused_dispatch,
        pybind11::arg("topk_idx"), pybind11::arg("assign_w"),
        pybind11::arg("local_start"), pybind11::arg("num_experts"),
        pybind11::arg("counts"), pybind11::arg("sorted_tids"),
        pybind11::arg("sorted_weights"), pybind11::arg("offsets"),
        pybind11::arg("problem_sizes_1"), pybind11::arg("problem_sizes_2"),
        pybind11::arg("token_counts") = pybind11::none(),
        pybind11::arg("token_perm")   = pybind11::none());
  m.def("mxf8_transcode_activations", &mxf8_transcode_activations);
  m.def("mxf8_transcode_and_pack_sfa", &mxf8_transcode_and_pack_sfa);
  m.def("mxf8_transcode_weights_impl", &mxf8_transcode_weights_impl);
  m.def("mxf8_pack_weight_sfb_impl", &mxf8_pack_weight_sfb_impl);
  m.def("moe_mxf8_setup_ptrs", &moe_mxf8_setup_ptrs);
  m.def("moe_mxf8_grouped_mm_prepacked", &moe_mxf8_grouped_mm_prepacked);
  m.def("fused_gather_mxf8", &fused_gather_mxf8);
  m.def("swiglu_fp8_requant_weighted_mxf8", &swiglu_fp8_requant_weighted_mxf8);
  m.def("compute_mxf8_sfa_layout_offsets_host", &compute_mxf8_sfa_layout_offsets_host);
  m.def("compute_mxf8_sfb_layout_offsets_host", &compute_mxf8_sfb_layout_offsets_host);
  m.def("compute_mxf8_sf_offsets_device", &compute_mxf8_sf_offsets_device);
  m.def("get_mxf8_sizes_stride", &get_mxf8_sizes_stride);
  m.def("get_mxf8_sizes_layout_sfa", &get_mxf8_sizes_layout_sfa);
  m.def("get_mxf8_sizes_layout_sfb", &get_mxf8_sizes_layout_sfb);
}
