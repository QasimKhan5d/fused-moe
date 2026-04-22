"""
Compute speed-of-light (SoL) per workload and compare against our kernel.

The fused MoE op is memory-bound on B200 (compute roofline 4.5 PFLOP/s FP8,
HBM roofline 3.0-3.5 TB/s). Hence SoL is dominated by bytes-moved / HBM-BW.

HBM traffic breakdown for one call (T tokens, E_LOCAL=32 local experts):
  Inputs (read):
    hidden_states         = T * K1 * 1B         (fp8)
    hidden_scale          = T * K1/128 * 4B     (fp32)
    routing_logits        = T * E_GLOBAL * 2B   (bf16)
    routing_bias          = E_GLOBAL * 2B
    gemm1_weights         = E_LOCAL * N1 * K1 * 1B   (fp8, fetched when any token hits)
    gemm1_weights_scale   = E_LOCAL * (N1/128)*(K1/128) * 4B
    gemm2_weights         = E_LOCAL * N2 * K2 * 1B
    gemm2_weights_scale   = E_LOCAL * (N2/128)*(K2/128) * 4B
  Intermediates (write + read in current pipeline):
    gemm1_out             = M_local * N1 * 2B   (bf16)       — eliminable via epilogue fusion
    act_q                 = M_local * H * 1B                 — written once, read once
    gemm2_out             = M_local * N2 * 2B   (bf16)       — eliminable via epilogue fusion
  Outputs (write):
    out                   = T * N2 * 2B

Where M_local is the number of valid (token, expert) pairs that fall on the
local expert bank, expected to equal T (since TOP_K=8, local_ratio=32/256=1/8).
"""
import modal

app = modal.App("moe-roofline")
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

# Shape constants (DeepSeek-V3 MoE).
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N1 = 4096
K1 = 7168
N2 = 7168
K2 = 2048
H = N1 // 2  # 2048 — SwiGLU output size

# B200 SXM5 effective HBM BW observed on well-tuned kernels (NVIDIA spec: 8 TB/s,
# DeepGEMM benchmarks on H200 get ~2.5-3.0; B200 HBM3e is ~4.0-4.5 observed).
HBM_BW_GBS = 4000  # 4.0 TB/s — conservative achievable number
FP8_COMPUTE_PFLOPS = 4.5  # B200 FP8 dense tensor ops (non-sparse)


def traffic_bytes(T: int, M_local: int, fused_epilogues: bool = False) -> dict:
    """Bytes of HBM traffic per call, with optional epilogue fusion savings."""
    # Input reads
    input_bytes = (
        T * K1 * 1           # hidden_states fp8
        + T * (K1 // 128) * 4  # hs_scale fp32
        + T * E_GLOBAL * 2   # routing_logits bf16
        + E_GLOBAL * 2       # routing_bias bf16
        + E_LOCAL * N1 * K1  # gemm1_weights fp8
        + E_LOCAL * (N1 // 128) * (K1 // 128) * 4  # gemm1_scale fp32
        + E_LOCAL * N2 * K2  # gemm2_weights fp8
        + E_LOCAL * (N2 // 128) * (K2 // 128) * 4  # gemm2_scale fp32
    )
    # Intermediates (write + read in the un-fused pipeline)
    g1_out = M_local * N1 * 2  # bf16 write + read = 2x if un-fused
    act_q = M_local * H * 1  # fp8 write + read = 2x
    g2_out = M_local * N2 * 2  # bf16 write + read = 2x if un-fused

    if fused_epilogues:
        # Best case: GEMM1 epilogue -> SwiGLU+FP8 quant straight to act_q (no g1_out)
        # GEMM2 epilogue -> weighted scatter (no g2_out materialization)
        intermediate = act_q * 2  # write + read
    else:
        intermediate = (g1_out + act_q + g2_out) * 2

    # Output writes
    output_bytes = T * N2 * 2  # final out bf16

    total = input_bytes + intermediate + output_bytes
    return {
        "input_B": input_bytes,
        "intermediate_B": intermediate,
        "output_B": output_bytes,
        "total_B": total,
        "total_MB": total / (1 << 20),
    }


def flops(M_local: int) -> float:
    """FLOPs per call (non-routing, dominated by GEMMs)."""
    g1 = 2 * M_local * N1 * K1
    g2 = 2 * M_local * N2 * K2
    return g1 + g2


def sol_ms(T: int, M_local: int, fused: bool = False) -> dict:
    """SoL time in ms: max of (memory_time, compute_time)."""
    t = traffic_bytes(T, M_local, fused)
    mem_ms = t["total_B"] / (HBM_BW_GBS * 1e9) * 1e3
    comp_ms = flops(M_local) / (FP8_COMPUTE_PFLOPS * 1e15) * 1e3
    return {
        "mem_ms": mem_ms,
        "compute_ms": comp_ms,
        "sol_ms": max(mem_ms, comp_ms),
        "bound_by": "mem" if mem_ms > comp_ms else "compute",
    }


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/mnt": trace_volume})
def analyze() -> str:
    import sys, os, time, torch
    from pathlib import Path

    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    import kernel as K
    _ = K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE_PATH)
    definition = trace_set.definitions[DEF_NAME]

    all_wls = [getattr(wl, "workload", wl)
               for wl in trace_set.workloads.get(DEF_NAME, [])]
    all_wls.sort(key=lambda w: w.axes.get("seq_len", 0))

    lines = []
    lines.append(f"HBM BW: {HBM_BW_GBS} GB/s  |  FP8 peak: {FP8_COMPUTE_PFLOPS} PFLOP/s")
    lines.append("=" * 102)
    lines.append(
        f"{'uuid':<9} {'T':>6} {'M_loc':>6} {'MB/call':>8} {'SoL_unf':>8} "
        f"{'SoL_fus':>8} {'measured':>9} {'pct_unf':>8} {'pct_fus':>8}"
    )
    lines.append("-" * 102)

    total_sol = 0.0
    total_sol_fused = 0.0
    total_measured = 0.0

    for wobj in all_wls:
        uuid = getattr(wobj, "uuid", "")
        loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        # Estimate M_local: route once and count local assignments.
        routing_logits, routing_bias = inputs[0], inputs[1]
        local_start = int(inputs[8])
        topk_idx, _ = K._route(routing_logits, routing_bias, 1.0, T, local_start, E_LOCAL)
        local_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_LOCAL)
        M_local = int(local_mask.sum().item())

        t_unf = sol_ms(T, M_local, fused=False)
        t_fus = sol_ms(T, M_local, fused=True)
        traf = traffic_bytes(T, M_local)

        # Warmup + measure
        for _ in range(3):
            _ = K.custom_kernel(*inputs)
        torch.cuda.synchronize()
        se, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        N = 30
        se.record()
        for _ in range(N):
            _ = K.custom_kernel(*inputs)
        ee.record()
        torch.cuda.synchronize()
        meas_ms = se.elapsed_time(ee) / N

        pct_unf = t_unf["sol_ms"] / meas_ms * 100
        pct_fus = t_fus["sol_ms"] / meas_ms * 100

        total_sol += t_unf["sol_ms"]
        total_sol_fused += t_fus["sol_ms"]
        total_measured += meas_ms

        lines.append(
            f"{uuid[:8]:<9} {T:>6d} {M_local:>6d} {traf['total_MB']:>8.1f} "
            f"{t_unf['sol_ms']:>8.3f} {t_fus['sol_ms']:>8.3f} {meas_ms:>9.3f} "
            f"{pct_unf:>7.1f}% {pct_fus:>7.1f}%"
        )

    lines.append("-" * 102)
    lines.append(
        f"{'TOTAL':<9} {'':>6} {'':>6} {'':>8} "
        f"{total_sol:>8.3f} {total_sol_fused:>8.3f} {total_measured:>9.3f} "
        f"{total_sol/total_measured*100:>7.1f}% {total_sol_fused/total_measured*100:>7.1f}%"
    )
    lines.append("")
    lines.append("SoL_unf = current pipeline (3 intermediates)")
    lines.append("SoL_fus = fully fused epilogues (SwiGLU in GEMM1 epi, scatter in GEMM2 epi)")
    lines.append("pct_X = SoL_X / measured — higher is better, 100% = memory-bound SoL")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print(analyze.remote())
