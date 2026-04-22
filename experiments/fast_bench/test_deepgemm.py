"""
Benchmark DeepGEMM's hand-tuned SM100 m_grouped FP8 GEMM against our CUTLASS
implementation on the exact contest GEMM shapes.

If DeepGEMM meaningfully beats CUTLASS, we can swap it in as the GEMM backend.
"""
import modal

app = modal.App("deepgemm-vs-cutlass")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

# Build an image with DeepGEMM installed.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .apt_install("git", "ninja-build")
    .pip_install(
        "flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git",
    )
    .run_commands(
        # Clone and install DeepGEMM
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/DeepGEMM",
        "cd /opt/DeepGEMM && bash install.sh || echo 'install may have issues'",
    )
    .add_local_dir(
        "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan/fused-moe/solution/python",
        remote_path="/root/solution",
    )
)


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def test() -> str:
    import os, sys, torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    out = []

    # 1. Check DeepGEMM availability
    try:
        import deep_gemm
        out.append(f"DeepGEMM imported OK, version: {getattr(deep_gemm, '__version__', 'unknown')}")
        dg_fns = [n for n in dir(deep_gemm) if 'grouped' in n]
        out.append(f"Grouped fns: {dg_fns[:8]}")
    except Exception as e:
        out.append(f"DeepGEMM import FAILED: {e!r}")
        import traceback
        out.append(traceback.format_exc()[-600:])
        return "\n".join(out)

    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    out.append(f"Using DeepGEMM contiguous alignment={alignment}")

    def align(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def per_token_cast_to_fp8(x: torch.Tensor, gran_k: int = 128):
        m, n = x.shape
        padded_n = align(n, gran_k)
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, padded_n // gran_k, gran_k)
        x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
        sf = x_amax / 448.0
        x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
        return x_fp8, sf

    def grouped_block_cast_to_fp8(x: torch.Tensor, gran_k: int = 128):
        # DeepGEMM grouped FP8 tests use per-block scales for B in shape
        # [G, ceil_div(N,128), ceil_div(K,128)] on the K-major path.
        g, m, n = x.shape
        x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        x_sf = torch.empty((g, (m + gran_k - 1) // gran_k, (n + gran_k - 1) // gran_k),
                           device=x.device, dtype=torch.float32)
        for i in range(g):
            x_i = x[i]
            mp = align(x_i.shape[0], gran_k)
            np = align(x_i.shape[1], gran_k)
            x_pad = torch.zeros((mp, np), dtype=x_i.dtype, device=x.device)
            x_pad[: x_i.shape[0], : x_i.shape[1]] = x_i
            x_view = x_pad.view(mp // gran_k, gran_k, np // gran_k, gran_k)
            sf = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4) / 448.0
            x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn).view(mp, np)
            x_fp8[i] = x_scaled[: x_i.shape[0], : x_i.shape[1]]
            x_sf[i] = sf.view(mp // gran_k, np // gran_k)
        return x_fp8, x_sf

    def make_contiguous_grouped_inputs(num_groups: int, expected_m_per_group: int, n: int, k: int):
        actual_ms = [expected_m_per_group] * num_groups
        aligned_ms = [align(m, alignment) for m in actual_ms]
        m_total = sum(aligned_ms)

        a_bf16 = torch.randn((m_total, k), device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
        grouped_layout = torch.empty(m_total, device="cuda", dtype=torch.int32)

        start = 0
        for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
            actual_end = start + actual_m
            aligned_end = start + aligned_m
            grouped_layout[start:actual_end] = i
            grouped_layout[actual_end:aligned_end] = -1
            a_bf16[actual_end:aligned_end] = 0
            start = aligned_end

        a_fp8, a_sf = per_token_cast_to_fp8(a_bf16, 128)
        b_fp8, b_sf = grouped_block_cast_to_fp8(b_bf16, 128)
        return m_total, (a_fp8, a_sf), (b_fp8, b_sf), grouped_layout

    # 2. Basic smoke test: 1 expert, small shape
    try:
        G, N, K = 4, 4096, 7168
        M, a, b, grouped_layout = make_contiguous_grouped_inputs(G, 64, N, K)
        d = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a, b, d, grouped_layout)
        torch.cuda.synchronize()
        out.append(f"Smoke test 1 PASSED: m={M}, n={N}, k={K}, g={G}, out.shape={d.shape}, "
                   f"nonzero_rows={(d.abs().float().sum(dim=1) > 0).sum().item()}/{M}")
    except Exception as e:
        import traceback
        out.append(f"Smoke test 1 FAILED: {e!r}")
        out.append(traceback.format_exc()[-800:])

    # 3. Bench on contest-sized shape: T=14107 equivalent
    try:
        G = 32
        expected_m_per_group = 14107 // G
        N1, K1 = 4096, 7168
        N2, K2 = 7168, 2048

        # GEMM1 inputs
        M, a1, b1, grouped_layout = make_contiguous_grouped_inputs(G, expected_m_per_group, N1, K1)
        d1 = torch.empty(M, N1, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a1, b1, d1, grouped_layout)
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        N_ITER = 30
        start_ev.record()
        for _ in range(N_ITER):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a1, b1, d1, grouped_layout)
        end_ev.record()
        torch.cuda.synchronize()
        dg_ms = start_ev.elapsed_time(end_ev) / N_ITER
        out.append(f"DeepGEMM GEMM1 on M={M}, N={N1}, K={K1}, G={G}: {dg_ms:.3f}ms")

        # Same shape with GEMM2
        M2, a2, b2, grouped_layout2 = make_contiguous_grouped_inputs(G, expected_m_per_group, N2, K2)
        d2 = torch.empty(M, N2, device="cuda", dtype=torch.bfloat16)
        assert M2 == M

        for _ in range(3):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a2, b2, d2, grouped_layout2)
        torch.cuda.synchronize()

        start_ev.record()
        for _ in range(N_ITER):
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a2, b2, d2, grouped_layout2)
        end_ev.record()
        torch.cuda.synchronize()
        dg_ms2 = start_ev.elapsed_time(end_ev) / N_ITER
        out.append(f"DeepGEMM GEMM2 on M={M}, N={N2}, K={K2}, G={G}: {dg_ms2:.3f}ms")

        # Compute HBM BW (rough)
        bytes_1 = G * N1 * K1 + M * K1 + M * K1 // 128 * 4 + M * N1 * 2 + G * (N1 // 128) * (K1 // 128) * 4
        bw1 = bytes_1 / (dg_ms * 1e-3) / 1e9
        bytes_2 = G * N2 * K2 + M * K2 + M * K2 // 128 * 4 + M * N2 * 2 + G * (N2 // 128) * (K2 // 128) * 4
        bw2 = bytes_2 / (dg_ms2 * 1e-3) / 1e9
        out.append(f"HBM BW: GEMM1={bw1:.0f} GB/s, GEMM2={bw2:.0f} GB/s, combined={dg_ms+dg_ms2:.3f}ms")

        # Compare to our CUTLASS. Ours was ~460μs per GEMM at M=14107 ≈ 920μs combined.
        out.append("vs CUTLASS (our): ~0.460ms per GEMM, ~0.920ms combined")
        out.append(f"Speedup: {0.920 / (dg_ms + dg_ms2):.2f}x")

    except Exception as e:
        import traceback
        out.append(f"GEMM1/GEMM2 bench FAILED: {e!r}")
        out.append(traceback.format_exc()[-1200:])

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(test.remote())
