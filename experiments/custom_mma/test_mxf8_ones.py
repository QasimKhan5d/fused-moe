"""Sanity test: MxF8 GEMM with all scales = 1.0 should produce same output as
pure FP8 GEMM (without scales). This isolates the scale-packing correctness
from the rest of the MxF8 machinery.
"""
import modal
app = modal.App("mxf8-ones-test")
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
    torch.manual_seed(7)

    E = 1
    M = 128
    N = 128
    K = 128

    # FP8 payloads in [-1, 1] for predictable GEMM output magnitude.
    a = (torch.randn(M, K, device=device) * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn(E, N, K, device=device) * 0.5).to(torch.float8_e4m3fn)
    scales_a = torch.ones(M, K // 128, device=device, dtype=torch.float32)
    scales_b = torch.ones(E, N // 128, K // 128, device=device, dtype=torch.float32)

    expert_offsets = torch.tensor([0, M], device=device, dtype=torch.int32)
    problem_sizes  = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)

    # Reference: plain FP8 GEMM via torch (A @ B.T with f32 dequant via identity scales).
    # For a = [M, K], b_e = [N, K]: out = a @ b_e.T  (each scale = 1 so no effect)
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)  # [E, N, K]
    out_ref = a_f32 @ b_f32[0].T  # [M, N]

    # MxF8 GEMM path.
    stride_sz = ext.get_mxf8_sizes_stride()
    sfa_sz    = ext.get_mxf8_sizes_layout_sfa()
    sfb_sz    = ext.get_mxf8_sizes_layout_sfb()

    a_ptrs   = torch.empty(E, device=device, dtype=torch.int64)
    b_ptrs   = torch.empty(E, device=device, dtype=torch.int64)
    out_ptrs = torch.empty(E, device=device, dtype=torch.int64)
    sfa_ptrs = torch.empty(E, device=device, dtype=torch.int64)
    sfb_ptrs = torch.empty(E, device=device, dtype=torch.int64)
    stride_a = torch.empty(E * stride_sz, device=device, dtype=torch.uint8)
    stride_b = torch.empty(E * stride_sz, device=device, dtype=torch.uint8)
    stride_c = torch.empty(E * stride_sz, device=device, dtype=torch.uint8)
    layout_sfa = torch.empty(E * sfa_sz, device=device, dtype=torch.uint8)
    layout_sfb = torch.empty(E * sfb_sz, device=device, dtype=torch.uint8)

    sfa_offsets, sfa_total = ext.compute_mxf8_sfa_layout_offsets_host(problem_sizes)
    sfb_offsets, sfb_total = ext.compute_mxf8_sfb_layout_offsets_host(problem_sizes)
    sfa_byte_offsets = torch.tensor(sfa_offsets, device=device, dtype=torch.int32)
    sfb_byte_offsets = torch.tensor(sfb_offsets, device=device, dtype=torch.int32)
    sfa_buffer = torch.empty(sfa_total, device=device, dtype=torch.uint8)
    sfb_buffer = torch.empty(sfb_total, device=device, dtype=torch.uint8)
    workspace = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    out_mx = torch.zeros(M, N, device=device, dtype=torch.bfloat16)

    ext.moe_mxf8_grouped_mm(
        out_mx, a, b, scales_a, scales_b,
        expert_offsets, problem_sizes,
        a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        sfa_buffer, sfb_buffer,
        sfa_byte_offsets, sfb_byte_offsets,
        workspace)
    torch.cuda.synchronize()

    # Check a few bytes of sfa_buffer — they should all be UE8M0 byte 127 (= 2^0 = 1.0).
    sfa_bytes_cpu = sfa_buffer.cpu()
    ones_count = (sfa_bytes_cpu == 127).sum().item()
    nonzero_count = (sfa_bytes_cpu != 0).sum().item()

    sfb_bytes_cpu = sfb_buffer.cpu()
    ones_count_b = (sfb_bytes_cpu == 127).sum().item()
    nonzero_count_b = (sfb_bytes_cpu != 0).sum().item()

    diff = (out_mx.float() - out_ref).abs()
    max_abs = diff.max().item()
    rel_err = (diff / (out_ref.abs() + 1e-6)).median().item()

    return (
        f"Test: all scales = 1.0, M={M} N={N} K={K}\n"
        f"sfa_buffer size={sfa_total}, bytes with value 127 (= UE8M0 for 1.0): {ones_count}, nonzero total: {nonzero_count}\n"
        f"sfb_buffer size={sfb_total}, bytes with value 127: {ones_count_b}, nonzero total: {nonzero_count_b}\n"
        f"out_mx: norm={out_mx.float().norm().item():.3f}, range=[{out_mx.float().min().item():.3f}, {out_mx.float().max().item():.3f}]\n"
        f"out_ref: norm={out_ref.norm().item():.3f}, range=[{out_ref.min().item():.3f}, {out_ref.max().item():.3f}]\n"
        f"MxF8 vs ref: max_abs_diff={max_abs:.3f}, median_rel_err={rel_err:.6f}\n"
    )


@app.local_entrypoint()
def main():
    print(run.remote())
