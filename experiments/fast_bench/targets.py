"""
Computes the 80%-of-speed-of-light target latency per workload, using NCU-
validated HBM traffic estimates where available.

Model (per workload):
  bytes_moved = weights(n_active) + intermediates(M_local) + output(T) + fixed
  sol_hbm_us = bytes_moved / (HBM_PEAK * 0.8) / 1e-6
  sol_compute_us = flops / (FP8_PEAK * 0.8) / 1e-6
  sol_practical_us = max(sol_hbm_us, sol_compute_us)

For SMALL T, kernel launch FLOOR starts to dominate: we have ~12-15 kernel
launches each with ~3-5μs minimum overhead. So the realistic target adds
a launch floor estimate:
  launch_floor_us = n_launches * min_launch_us

Achievable lower bound ≈ max(sol_practical_us, launch_floor_us)

Calibration: NCU on T=14107 measured 2488 MB vs our model 2530 MB (+1.7%),
so the model is accurate for large T. For small T we currently over-estimate
weight traffic (assumes all n_active experts' full weights are fetched); real
HBM is often 60-80% of model when L2 soaks up repeat reads.
"""

# Constants
E_LOCAL = 32
N1, K1 = 4096, 7168
N2, K2 = 7168, 2048
H = N1 // 2
E_GLOBAL = 256
TOP_K = 8

HBM_PEAK_GBS = 5000     # B200 HBM3e best-case achievable
TARGET_FRAC = 0.80      # 80% of peak is the "near SoL" bar
FP8_PEAK_PFLOPS = 4.5

N_KERNELS_IN_PIPELINE = 13  # route + dispatch(3) + gather + gemm1 + swiglu + gemm2 + scatter(4) + ptrs(2)
LAUNCH_FLOOR_US = 3.5   # per-kernel min overhead on B200


def bytes_per_expert() -> int:
    return (
        N1 * K1                               # gemm1_weights fp8
        + (N1 // 128) * (K1 // 128) * 4        # gemm1 scales
        + N2 * K2                              # gemm2_weights fp8
        + (N2 // 128) * (K2 // 128) * 4        # gemm2 scales
    )


def bytes_per_mlocal_row() -> int:
    """Intermediate HBM I/O per M_local row through the pipeline (unfused)."""
    return (
        K1 * 1             # packed_acts write+read (gather + GEMM1 A): count twice
        + (K1 // 128) * 4   # packed_act_scales write+read
        + N1 * 2            # gemm1_out write+read (bf16)
        + H * 1             # act_q write+read (fp8)
        + (H // 128) * 4    # scales
        + K2 * 1            # act_q read in GEMM2 (L2 hit usually, count half)
        + (K2 // 128) * 4
        + N2 * 2            # gemm2_out write+read
    ) * 1  # we do NOT double-count since most are single HBM trips with L2 hits between


def estimate_bytes(T: int, M_local: int, n_active: int) -> float:
    """Returns estimated HBM bytes moved per call, in MB."""
    # Input reads (routing_logits + bias)
    routing_in = T * E_GLOBAL * 2 + E_GLOBAL * 2
    # Hidden states are read only at sorted_tids indices = M_local rows
    hidden_in = M_local * K1 * 1 + M_local * (K1 // 128) * 4
    # Weights of active experts
    weights = n_active * bytes_per_expert()
    # Intermediate I/O (gemm1_out, swiglu, gemm2_out)
    intermediate = (
        M_local * N1 * 2 * 2    # gemm1_out write + swiglu read
        + M_local * H * 1 * 2    # act_q write + GEMM2 read (L2 partly)
        + M_local * (H // 128) * 4 * 2
        + M_local * N2 * 2 * 2  # gemm2_out write + scatter read
    )
    # Output write
    out_write = T * N2 * 2
    # Dispatch bookkeeping (tiny)
    dispatch = E_LOCAL * 16 + T * TOP_K * 16

    total = routing_in + hidden_in + weights + intermediate + out_write + dispatch
    return total / (1 << 20)  # MB


def flops(M_local: int) -> int:
    return 2 * M_local * (N1 * K1 + N2 * K2)


def launch_floor_us() -> float:
    return N_KERNELS_IN_PIPELINE * LAUNCH_FLOOR_US


def coupon_collector(M_local: int, E: int = E_LOCAL) -> int:
    if M_local <= 0:
        return 0
    return max(1, int(round(E * (1.0 - ((E - 1) / E) ** M_local))))


def row(T: int, M_local: int, measured_us: float, ncu_bytes_mb: float | None = None):
    n_active = coupon_collector(M_local)
    # Model-based
    model_mb = estimate_bytes(T, M_local, n_active)
    fl = flops(M_local)

    # If we have NCU, use it; else use model
    actual_mb = ncu_bytes_mb if ncu_bytes_mb is not None else model_mb

    # 80%-SoL HBM time = bytes / (0.8 * peak_hbm)
    sol_hbm_us = actual_mb / (HBM_PEAK_GBS * TARGET_FRAC * 1e3 / 1e6)  # MB / (GB/s) -> ms, *1000 for us
    # Actually: bytes / (bw_bps) = seconds. bw in GB/s = 1e9 B/s
    sol_hbm_us = (actual_mb * (1 << 20)) / (HBM_PEAK_GBS * TARGET_FRAC * 1e9) * 1e6

    # 80% compute SoL
    sol_compute_us = fl / (FP8_PEAK_PFLOPS * TARGET_FRAC * 1e15) * 1e6

    # Practical target = max(hbm, compute, launch_floor)
    floor = launch_floor_us()
    target_us = max(sol_hbm_us, sol_compute_us, floor)

    # Achieved HBM BW on measured data
    achieved_gbs = (actual_mb * (1 << 20)) / (measured_us * 1e-6) / 1e9
    # Current % of 80% SoL target
    pct_of_target = target_us / measured_us * 100

    src = "NCU" if ncu_bytes_mb is not None else "model"
    return {
        "T": T, "M_local": M_local, "n_active": n_active,
        "bytes_MB": actual_mb, "source": src,
        "sol_hbm_us": sol_hbm_us, "sol_compute_us": sol_compute_us,
        "launch_floor_us": floor, "target_us": target_us,
        "measured_us": measured_us, "achieved_gbs": achieved_gbs,
        "pct_of_target": pct_of_target,
        "gap_us": measured_us - target_us,
    }


# Sample workloads — M_local from our earlier measurements, measured_us from 5-trial median
SAMPLES = [
    # (T, M_local, measured_us, ncu_bytes_mb_or_None)
    (1,     3,      80.0,   None),     # NCU not yet run
    (80,    109,    262.0,  None),     # NCU not yet run
    (901,   1331,   399.0,  None),     # NCU not yet run
    (11948, 10416,  1047.0, None),     # NCU not yet run
    (14107, 16099,  1480.0, 2488.17),  # NCU measured
]


def main():
    print(f"B200 HBM3e peak: {HBM_PEAK_GBS} GB/s.  Target = {int(TARGET_FRAC*100)}% = {int(HBM_PEAK_GBS*TARGET_FRAC)} GB/s")
    print(f"FP8 peak: {FP8_PEAK_PFLOPS} PFLOPS.  Target compute = {FP8_PEAK_PFLOPS*TARGET_FRAC} PFLOPS")
    print(f"Launch floor: {N_KERNELS_IN_PIPELINE} kernels × {LAUNCH_FLOOR_US}μs = {launch_floor_us():.0f}μs")
    print()
    hdr = (f"{'T':>6} {'M_loc':>6} {'n_a':>4} {'bytes_MB':>9} {'src':>6} "
           f"{'SoL_hbm':>8} {'SoL_cmp':>8} {'floor':>6} "
           f"{'target':>8} {'meas':>8} {'pct':>6} {'gap':>6} {'GB/s':>6}")
    print(hdr)
    print("-" * len(hdr))
    for T, M, meas, ncu in SAMPLES:
        r = row(T, M, meas, ncu)
        print(f"{r['T']:>6d} {r['M_local']:>6d} {r['n_active']:>4d} "
              f"{r['bytes_MB']:>9.1f} {r['source']:>6} "
              f"{r['sol_hbm_us']:>8.1f} {r['sol_compute_us']:>8.1f} "
              f"{r['launch_floor_us']:>6.1f} "
              f"{r['target_us']:>8.1f} {r['measured_us']:>8.1f} "
              f"{r['pct_of_target']:>5.1f}% {r['gap_us']:>6.0f} "
              f"{r['achieved_gbs']:>6.0f}")
    print("-" * len(hdr))
    print()
    print("target = your 80%-SoL latency target to beat (max of HBM-bound, compute-bound, launch-floor)")
    print("pct    = target / measured * 100 (100% = at 80%-SoL; higher = closer)")
    print("gap    = measured - target (microseconds to shave)")
    print("GB/s   = current achieved HBM BW (compare to 4000 target = 80% peak)")


if __name__ == "__main__":
    main()
