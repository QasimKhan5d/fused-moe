"""Head-to-head: MxF8 GEMM vs FP32-blockwise GEMM on same transcoded inputs.

This is the definitive MxF8 plumbing test. If the two CUTLASS paths produce the
same numerical output when fed the same (transcoded fp8, pow-of-2 fp32 scale)
inputs, then the MxF8 hardware block-scale MMA is working correctly in our
kernel.

Setup:
  - Synthetic small-scale grouped GEMM: E=4, M_per_expert=128, N=256, K=256.
  - Random fp32 SIGNED scales, random fp8 payload.
  - Transcode both inputs using our CUDA kernels.
  - Run FP32-blockwise (our current moe_blockwise_grouped_mm_v2).
  - Run MxF8 (new moe_mxf8_grouped_mm).
  - Compare.
"""
import modal
app = modal.App("mxf8-vs-blockwise")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={"/mnt": trace_volume})
def run() -> str:
    import os, sys
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    ext = K._get_ext()

    device = torch.device("cuda")
    torch.manual_seed(42)

    # Grouped problem: E=4 experts, each with M=128 rows.
    E = 4
    M_per = 128
    total_m = E * M_per
    N = 256
    K = 256
    K_blocks = K // 128

    # Synthesize inputs.
    a_orig_fp8 = torch.randn(total_m, K, device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    b_orig_fp8 = torch.randn(E, N, K, device=device).clamp(-3, 3).to(torch.float8_e4m3fn)
    # Signed fp32 scales in range [-5, 5] matching contest distribution.
    scales_a_fp32 = torch.randn(total_m, K_blocks, device=device) * 2.0
    scales_b_fp32 = torch.randn(E, N // 128, K_blocks, device=device) * 2.0

    expert_offsets = torch.tensor([i * M_per for i in range(E + 1)], device=device, dtype=torch.int32)
    problem_sizes = torch.tensor([[M_per, N, K]] * E, device=device, dtype=torch.int32).contiguous()

    # ============ A) FP32-blockwise reference ============
    # Workspace sizes.
    stride_sz_fp32, sfa_sz_fp32, sfb_sz_fp32 = ext.get_sizes()
    a_ptrs_fp32   = torch.empty(E, device=device, dtype=torch.int64)
    b_ptrs_fp32   = torch.empty(E, device=device, dtype=torch.int64)
    out_ptrs_fp32 = torch.empty(E, device=device, dtype=torch.int64)
    sfa_ptrs_fp32 = torch.empty(E, device=device, dtype=torch.int64)
    sfb_ptrs_fp32 = torch.empty(E, device=device, dtype=torch.int64)
    stride_a_fp32 = torch.empty(E * stride_sz_fp32, device=device, dtype=torch.uint8)
    stride_b_fp32 = torch.empty(E * stride_sz_fp32, device=device, dtype=torch.uint8)
    stride_c_fp32 = torch.empty(E * stride_sz_fp32, device=device, dtype=torch.uint8)
    layout_sfa_fp32 = torch.empty(E * sfa_sz_fp32, device=device, dtype=torch.uint8)
    layout_sfb_fp32 = torch.empty(E * sfb_sz_fp32, device=device, dtype=torch.uint8)
    problem_sizes_t = torch.empty(E, 3, device=device, dtype=torch.int32)
    workspace_fp32 = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)

    out_fp32 = torch.zeros(total_m, N, device=device, dtype=torch.bfloat16)

    ext.moe_blockwise_grouped_mm_v2(
        out_fp32,
        a_orig_fp8.clone(),
        b_orig_fp8.clone(),
        scales_a_fp32.clone(),
        scales_b_fp32.clone(),
        expert_offsets[:E].contiguous(),  # [E] (not E+1)
        problem_sizes,
        problem_sizes_t,
        a_ptrs_fp32, b_ptrs_fp32, out_ptrs_fp32, sfa_ptrs_fp32, sfb_ptrs_fp32,
        stride_a_fp32, stride_b_fp32, stride_c_fp32,
        layout_sfa_fp32, layout_sfb_fp32,
        workspace_fp32,
    )
    torch.cuda.synchronize()

    # ============ B) Transcode inputs ============
    a_transcoded = a_orig_fp8.clone()
    b_transcoded = b_orig_fp8.clone()
    scales_a_ue8m0 = torch.empty_like(scales_a_fp32)
    scales_b_ue8m0 = torch.empty_like(scales_b_fp32)
    ext.mxf8_transcode_activations(a_transcoded, scales_a_fp32, scales_a_ue8m0)
    ext.mxf8_transcode_weights_impl(b_transcoded, scales_b_fp32, scales_b_ue8m0)
    torch.cuda.synchronize()

    # ============ C) FP32-blockwise with transcoded inputs (stand-in for MxF8) ============
    out_fp32_tr = torch.zeros(total_m, N, device=device, dtype=torch.bfloat16)
    ext.moe_blockwise_grouped_mm_v2(
        out_fp32_tr,
        a_transcoded.clone(),
        b_transcoded.clone(),
        scales_a_ue8m0.clone(),
        scales_b_ue8m0.clone(),
        expert_offsets[:E].contiguous(),
        problem_sizes,
        problem_sizes_t,
        a_ptrs_fp32, b_ptrs_fp32, out_ptrs_fp32, sfa_ptrs_fp32, sfb_ptrs_fp32,
        stride_a_fp32, stride_b_fp32, stride_c_fp32,
        layout_sfa_fp32, layout_sfb_fp32,
        workspace_fp32,
    )
    torch.cuda.synchronize()

    # ============ D) MxF8 GEMM with transcoded inputs ============
    stride_sz_mx = ext.get_mxf8_sizes_stride()
    sfa_sz_mx    = ext.get_mxf8_sizes_layout_sfa()
    sfb_sz_mx    = ext.get_mxf8_sizes_layout_sfb()

    a_ptrs_mx   = torch.empty(E, device=device, dtype=torch.int64)
    b_ptrs_mx   = torch.empty(E, device=device, dtype=torch.int64)
    out_ptrs_mx = torch.empty(E, device=device, dtype=torch.int64)
    sfa_ptrs_mx = torch.empty(E, device=device, dtype=torch.int64)
    sfb_ptrs_mx = torch.empty(E, device=device, dtype=torch.int64)
    stride_a_mx = torch.empty(E * stride_sz_mx, device=device, dtype=torch.uint8)
    stride_b_mx = torch.empty(E * stride_sz_mx, device=device, dtype=torch.uint8)
    stride_c_mx = torch.empty(E * stride_sz_mx, device=device, dtype=torch.uint8)
    layout_sfa_mx = torch.empty(E * sfa_sz_mx, device=device, dtype=torch.uint8)
    layout_sfb_mx = torch.empty(E * sfb_sz_mx, device=device, dtype=torch.uint8)

    sfa_offsets, sfa_total = ext.compute_mxf8_sfa_layout_offsets_host(problem_sizes)
    sfb_offsets, sfb_total = ext.compute_mxf8_sfb_layout_offsets_host(problem_sizes)
    sfa_byte_offsets = torch.tensor(sfa_offsets, device=device, dtype=torch.int32)
    sfb_byte_offsets = torch.tensor(sfb_offsets, device=device, dtype=torch.int32)
    sfa_buffer = torch.empty(sfa_total, device=device, dtype=torch.uint8)
    sfb_buffer = torch.empty(sfb_total, device=device, dtype=torch.uint8)
    workspace_mx = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)

    out_mx = torch.zeros(total_m, N, device=device, dtype=torch.bfloat16)

    # moe_mxf8_grouped_mm expects expert_offsets of size E+1 (inclusive scan).
    ext.moe_mxf8_grouped_mm(
        out_mx,
        a_transcoded,
        b_transcoded,
        scales_a_ue8m0,
        scales_b_ue8m0,
        expert_offsets,  # [E+1]
        problem_sizes,
        a_ptrs_mx, b_ptrs_mx, out_ptrs_mx, sfa_ptrs_mx, sfb_ptrs_mx,
        stride_a_mx, stride_b_mx, stride_c_mx,
        layout_sfa_mx, layout_sfb_mx,
        sfa_buffer, sfb_buffer,
        sfa_byte_offsets, sfb_byte_offsets,
        workspace_mx,
    )
    torch.cuda.synchronize()

    lines = []
    lines.append(f"E={E} M={M_per}x{E} N={N} K={K}")
    lines.append(f"Reference (pristine fp32-blockwise):       norm={out_fp32.float().norm().item():.3f}")
    lines.append(f"Transcoded fp32-blockwise (Python probe):  norm={out_fp32_tr.float().norm().item():.3f}")
    lines.append(f"Transcoded MxF8 (hardware block-scale):    norm={out_mx.float().norm().item():.3f}")

    # Compare transcoded fp32-blockwise vs transcoded MxF8: SHOULD BE ~IDENTICAL,
    # since both apply the same numerical model (fp8_payload * ue8m0_scale summed
    # over K). Any divergence indicates a CUTLASS MxF8 bug.
    diff_mx_vs_tr = (out_mx.float() - out_fp32_tr.float()).abs()
    max_abs = diff_mx_vs_tr.max().item()
    rel_err = (diff_mx_vs_tr / (out_fp32_tr.float().abs() + 1e-6)).median().item()
    lines.append(f"  MxF8 vs transcoded-fp32-blockwise: max_abs_diff={max_abs:.3f} median_rel_err={rel_err:.5f}")

    # And compare MxF8 to pristine reference (should be the transcoded ~92% match).
    diff_mx_vs_ref = (out_mx.float() - out_fp32.float()).abs()
    tol = 1.0 + 0.3 * out_fp32.float().abs()
    match = (diff_mx_vs_ref <= tol).float().mean().item()
    lines.append(f"  MxF8 vs pristine-fp32-blockwise: match={match*100:.2f}% max_abs_diff={diff_mx_vs_ref.max().item():.3f}")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(run.remote())
