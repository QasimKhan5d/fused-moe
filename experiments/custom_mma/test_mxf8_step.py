"""Step through MxF8 correctness:
  Step 1: E=1, scales = ones → OK (confirmed).
  Step 2: E=1, random pow-of-2 scales.
  Step 3: E=4, random pow-of-2 scales.
Compare MxF8 output to a CPU reference computed from the same fp8 payload +
pow-of-2 scales. We use the CPU reference (not fp32-blockwise CUTLASS) so we
can isolate MxF8 kernel bugs from fp32-blockwise kernel semantics differences.
"""
import modal
app = modal.App("mxf8-step")
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


def run_mxf8(ext, E, m_per_expert, N, K, a, b, scales_a, scales_b):
    import torch
    device = a.device
    total_m = sum(m_per_expert)
    expert_offsets = torch.tensor([0] + [sum(m_per_expert[: i + 1]) for i in range(E)],
                                  device=device, dtype=torch.int32)
    problem_sizes = torch.tensor([[m_per_expert[e], N, K] for e in range(E)],
                                 device=device, dtype=torch.int32).contiguous()

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
    out = torch.zeros(total_m, N, device=device, dtype=torch.bfloat16)

    ext.moe_mxf8_grouped_mm(
        out, a, b, scales_a, scales_b,
        expert_offsets, problem_sizes,
        a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
        stride_a, stride_b, stride_c,
        layout_sfa, layout_sfb,
        sfa_buffer, sfb_buffer,
        sfa_byte_offsets, sfb_byte_offsets,
        workspace)
    torch.cuda.synchronize()
    return out, sfa_buffer, sfb_buffer


def cpu_ref(a, b, scales_a, scales_b, m_per_expert, N, K):
    """CPU reference: apply per-128-K-block scales to fp8 payload then gemm."""
    import torch
    E = len(m_per_expert)
    total_m = sum(m_per_expert)
    out = torch.zeros(total_m, N, dtype=torch.float32)
    m_off = 0
    K_blocks = K // 128
    for e in range(E):
        me = m_per_expert[e]
        a_e = a[m_off : m_off + me].to(torch.float32)   # [me, K]
        b_e = b[e].to(torch.float32)                    # [N, K]
        sa_e = scales_a[m_off : m_off + me]             # [me, K/128]
        sb_e = scales_b[e]                              # [N/128, K/128]
        # decode
        a_dec = a_e.clone()
        for kb in range(K_blocks):
            k_s = kb * 128
            k_e = (kb + 1) * 128
            a_dec[:, k_s:k_e] *= sa_e[:, kb:kb + 1]
        b_dec = b_e.clone()
        N_blocks = N // 128
        for nb in range(N_blocks):
            n_s = nb * 128
            n_e = (nb + 1) * 128
            for kb in range(K_blocks):
                k_s = kb * 128
                k_e2 = (kb + 1) * 128
                b_dec[n_s:n_e, k_s:k_e2] *= sb_e[nb, kb]
        out[m_off : m_off + me] = a_dec @ b_dec.T
        m_off += me
    return out


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run() -> str:
    import os, sys
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    ext = K._get_ext()

    device = torch.device("cuda")
    torch.manual_seed(123)

    lines = []

    # Step 1: E=1, ones scales. Already known to pass.
    # Step 2: E=1, random pow-of-2 positive scales.
    for step_name, E, M_per, N, K in [
        ("Step1 E=1 ones K=128",  1, [128], 128, 128),
        ("Step2 E=1 pow2 K=128",  1, [128], 128, 128),
        ("Step3 E=1 pow2 K=256",  1, [128], 128, 256),
        ("Step4 E=4 pow2 K=256",  4, [128, 128, 128, 128], 128, 256),
        ("Step5 E=1 pow2 K=128 pos-only-scales",  1, [128], 128, 128),
    ]:
        total_m = sum(M_per)
        a = (torch.randn(total_m, K, device=device) * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(E, N, K, device=device) * 0.5).to(torch.float8_e4m3fn)
        if "ones" in step_name:
            scales_a = torch.ones(total_m, K // 128, device=device, dtype=torch.float32)
            scales_b = torch.ones(E, N // 128, K // 128, device=device, dtype=torch.float32)
        else:
            # Random positive pow-of-2 in [2^-3, 2^3].
            scales_a = torch.pow(2.0, torch.randint(-3, 4, (total_m, K // 128), device=device).float())
            scales_b = torch.pow(2.0, torch.randint(-3, 4, (E, N // 128, K // 128), device=device).float())

        out_mx, sfa_buf, sfb_buf = run_mxf8(ext, E, M_per, N, K, a, b, scales_a, scales_b)
        out_ref = cpu_ref(a.cpu(), b.cpu(), scales_a.cpu(), scales_b.cpu(), M_per, N, K)
        diff = (out_mx.float().cpu() - out_ref).abs()
        max_abs = diff.max().item()
        rel_err = (diff / (out_ref.abs() + 1e-6)).median().item()
        lines.append(
            f"{step_name}: norm_mx={out_mx.float().norm().item():.2f}  "
            f"norm_ref={out_ref.norm().item():.2f}  "
            f"max_abs={max_abs:.3f}  median_rel={rel_err:.5f}"
        )

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(run.remote())
