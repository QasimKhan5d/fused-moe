"""
Offline SoL calc using already-measured kernel times, with the bugs from v1
fixed:

  FIX 1: Only count ACTIVE-expert weight traffic (not all 32).
         CUTLASS grouped-GEMM only fetches weights for experts with
         count[e] > 0. For small T (e.g. T=1 with M_local=3), only ~3
         experts are active and 3 * 44MB of weights are loaded, not 32 * 44MB.

  FIX 2: Use realistic B200 HBM ceilings instead of an unspecified "4 TB/s".
         Nominal HBM3e = 8 TB/s.
         Peak achievable on copy/memset: ~5.0 TB/s.
         Practical ceiling for compute+memory kernels (DeepGEMM publishes
         ~4.0-4.3 TB/s on H200, B200 is newer and higher): ~4.2 TB/s.

  FIX 3: Count the ACTUAL pipeline traffic including intermediates:
         routing, dispatch buffers, gather copy, gemm1_out, swiglu writes,
         gemm2_out, scatter output. (v1 only counted the 3 major inputs and
         the 3 intermediates once, missing dispatch overhead.)

  FIX 4: Separately report memory-bound SoL (bytes / peak_bw) and compute-
         bound SoL (FLOPs / peak_flops). Pick max as practical SoL.

Uses measured latencies from the prior 5-trial fast_bench run; no Modal call
needed. If you want live numbers, run this as a Modal function with
hook to K._route for real n_active counts.
"""

# Model constants
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N1 = 4096
K1 = 7168
N2 = 7168
K2 = 2048
H = N1 // 2

HBM_PEAK_GBS = 5000        # ~5 TB/s, best-case (copy/memset) achievable on B200 HBM3e
HBM_PRACTICAL_GBS = 4200   # 80-85% of peak; DeepGEMM-style hand-tuned kernels
FP8_PEAK_PFLOPS = 4.5      # B200 FP8 dense tensor ops (non-sparse)
FP8_PRACTICAL_PFLOPS = 3.5  # ~78% of peak


def expected_n_active(M_local: int, E: int = E_LOCAL) -> int:
    """Coupon-collector-style expected # of unique experts hit by M_local
    uniform local picks (a reasonable upper bound; actual routing is biased
    toward certain groups but all-32 is reached rapidly for M_local >= ~80)."""
    if M_local <= 0:
        return 0
    # E[unique] = E * (1 - ((E-1)/E)**M_local)
    return max(1, int(round(E * (1.0 - ((E - 1) / E) ** M_local))))


def traffic_bytes(T: int, M_local: int, n_active: int,
                  fused_epilogues: bool = False) -> dict:
    """HBM bytes moved by our current pipeline."""
    # 1. Routing (fused_route_topk)
    routing = (
        T * E_GLOBAL * 2 + E_GLOBAL * 2    # read logits + bias (bf16)
        + T * TOP_K * 4 + T * TOP_K * 4   # write topk_idx + assign_w
    )

    # 2. Dispatch (count + scan + place)
    dispatch = (
        T * TOP_K * 4 * 2                  # re-read topk_idx+weights
        + E_LOCAL * 4 * 4                   # counts+offsets+cursors+zeros
        + T * TOP_K * 4 * 2                # sorted_tids+sorted_weights write
        + T * 4 * 3                         # token_bucket bookkeeping (for reduce_scatter)
        + T * TOP_K * 4                    # token_perm
    )

    # 3. Gather hidden_states + scales at sorted indices
    gather = (
        M_local * K1 * 1                        # packed_acts write
        + M_local * (K1 // 128) * 4              # packed_act_scales write
        + M_local * K1 * 1                      # packed_acts read into GEMM (L2 likely hits)
        + M_local * (K1 // 128) * 4              # scales read
    )

    # 4. GEMM1 — weight fetch is n_active experts, NOT E_LOCAL
    gemm1 = (
        n_active * N1 * K1 * 1                        # gemm1_weights fp8
        + n_active * (N1 // 128) * (K1 // 128) * 4    # scales fp32
        + (M_local * N1 * 2 if not fused_epilogues else 0)  # gemm1_out write
    )

    # 5. SwiGLU (only if unfused — reads gemm1_out, writes act_q+scales)
    swiglu = (
        (M_local * N1 * 2 if not fused_epilogues else 0)  # read gemm1_out
        + M_local * H * 1                                   # act_q write
        + M_local * 4                                       # row scale
        + M_local * (H // 128) * 4                          # broadcast scale
    )

    # 6. GEMM2 — weight fetch is n_active experts
    gemm2 = (
        M_local * K2 * 1                                   # act_q read
        + M_local * (K2 // 128) * 4                         # scales read
        + n_active * N2 * K2 * 1                           # gemm2_weights
        + n_active * (N2 // 128) * (K2 // 128) * 4         # scales
        + (M_local * N2 * 2 if not fused_epilogues else 0) # gemm2_out write
    )

    # 7. Scatter / reduce_scatter
    scatter = (
        (M_local * N2 * 2 if not fused_epilogues else 0)   # read gemm2_out
        + M_local * 4                                        # weights
        + M_local * 4                                        # tids
        + T * N2 * 2                                         # out write
    )

    total = routing + dispatch + gather + gemm1 + swiglu + gemm2 + scatter
    return {
        "routing_B": routing,
        "dispatch_B": dispatch,
        "gather_B": gather,
        "gemm1_B": gemm1,
        "swiglu_B": swiglu,
        "gemm2_B": gemm2,
        "scatter_B": scatter,
        "total_B": total,
        "weights_B": n_active * (N1 * K1 + N2 * K2)
                     + n_active * ((N1 // 128) * (K1 // 128) + (N2 // 128) * (K2 // 128)) * 4,
    }


def flops(M_local: int) -> int:
    return 2 * M_local * (N1 * K1 + N2 * K2)


def fmt_row(uuid: str, T: int, measured_ms: float, M_local: int, n_active: int):
    t_unf = traffic_bytes(T, M_local, n_active, fused_epilogues=False)
    t_fus = traffic_bytes(T, M_local, n_active, fused_epilogues=True)
    fl = flops(M_local)

    # Memory SoL
    mem_ms_peak = t_unf["total_B"] / (HBM_PEAK_GBS * 1e9) * 1e3
    mem_ms_prac = t_unf["total_B"] / (HBM_PRACTICAL_GBS * 1e9) * 1e3
    mem_ms_fus = t_fus["total_B"] / (HBM_PRACTICAL_GBS * 1e9) * 1e3
    # Compute SoL
    comp_ms = fl / (FP8_PRACTICAL_PFLOPS * 1e15) * 1e3
    # Practical SoL = max of mem and compute paths
    sol_prac = max(mem_ms_prac, comp_ms)
    sol_fus_prac = max(mem_ms_fus, comp_ms)

    # Achieved HBM BW (assume ALL measured time is HBM time — underestimates for
    # compute-bound workloads but is a useful upper-bound of BW achieved)
    achieved_hbm = t_unf["total_B"] / (measured_ms * 1e-3) / 1e9

    # % of peak HBM SoL (theoretical minimum)
    pct_peak = mem_ms_peak / measured_ms * 100
    # % of practical SoL (80% peak) — our realistic target
    pct_prac = sol_prac / measured_ms * 100
    # % of fused-practical SoL
    pct_fus = sol_fus_prac / measured_ms * 100

    return (
        f"{uuid[:8]:<9} {T:>6d} {M_local:>6d} {n_active:>4d} "
        f"{t_unf['total_B']/(1<<20):>6.0f} {t_unf['weights_B']/(1<<20):>7.0f} "
        f"{mem_ms_peak:>7.3f} {mem_ms_prac:>7.3f} {comp_ms:>7.3f} "
        f"{measured_ms:>6.3f} {pct_peak:>5.1f}% {pct_prac:>5.1f}% "
        f"{achieved_hbm:>5.0f}"
    )


# 5-trial median times from our most recent bench (offline data)
# (uuid_prefix, T, measured_ms, M_local) — M_local for large-T extrapolated from
# earlier roofline_v1 output where M_local was measured by K._route.
ROWS = [
    # (uuid, T, ms, M_local)
    ("e05c6c03", 1, 0.080, 3),
    ("b8f4f012", 7, 0.114, 7),
    ("8cba5890", 14, 0.148, 15),
    ("2e69caee", 15, 0.114, 10),
    ("a7c2bcfd", 16, 0.159, 17),
    ("6230e838", 32, 0.219, 36),
    ("f7d6ac7c", 52, 0.188, 26),
    ("fc378037", 53, 0.239, 53),
    ("76010cb4", 54, 0.228, 43),
    ("81955b1e", 55, 0.239, 55),
    ("4822167c", 56, 0.245, 72),
    ("74d7ff04", 57, 0.240, 63),
    ("e626d3e6", 58, 0.247, 77),
    ("eedc63b2", 59, 0.204, 41),
    ("5eadab1e", 62, 0.196, 59),
    ("8f1ff9f1", 80, 0.262, 109),
    ("1a4c6ba1", 901, 0.385, 1331),
    ("58a34f27", 11948, 1.047, 10416),
    ("5e8dc11c", 14107, 1.480, 16099),
]


def main():
    print(f"B200 HBM3e: peak={HBM_PEAK_GBS/1000:.1f} TB/s (copy/memset)")
    print(f"           practical={HBM_PRACTICAL_GBS/1000:.1f} TB/s (80% of peak, DeepGEMM-tuned)")
    print(f"           FP8 compute practical={FP8_PRACTICAL_PFLOPS} PFLOPS")
    print()
    print("ACTIVE-expert weight traffic model: n_active = expected # of unique")
    print("local experts hit by M_local uniform picks (coupon-collector upper bound).")
    print()
    print("%peak = SoL_mem@peak / meas — theoretical 'absolute speed of light'")
    print("%prac = max(SoL_mem@4.2TB/s, SoL_compute@3.5PFLOPS) / meas — realistic target")
    print()

    header = (
        f"{'uuid':<9} {'T':>6} {'M_loc':>6} {'n_a':>4} {'MB':>6} {'wtMB':>7} "
        f"{'mem_pk':>7} {'mem_pr':>7} {'comp':>7} "
        f"{'meas':>6} {'%peak':>6} {'%prac':>6} {'HBMGB/s':>5}"
    )
    print(header)
    print("-" * len(header))

    total_meas = 0.0
    total_sol_prac = 0.0
    total_sol_fus = 0.0
    for uuid, T, ms, M_local in ROWS:
        n_active = expected_n_active(M_local)
        print(fmt_row(uuid, T, ms, M_local, n_active))
        total_meas += ms
        t_unf = traffic_bytes(T, M_local, n_active, False)
        t_fus = traffic_bytes(T, M_local, n_active, True)
        fl = flops(M_local)
        sol_prac = max(
            t_unf["total_B"] / (HBM_PRACTICAL_GBS * 1e9) * 1e3,
            fl / (FP8_PRACTICAL_PFLOPS * 1e15) * 1e3,
        )
        sol_fus = max(
            t_fus["total_B"] / (HBM_PRACTICAL_GBS * 1e9) * 1e3,
            fl / (FP8_PRACTICAL_PFLOPS * 1e15) * 1e3,
        )
        total_sol_prac += sol_prac
        total_sol_fus += sol_fus

    print("-" * len(header))
    print()
    print(f"Totals (sum of 19 workloads):")
    print(f"  measured     = {total_meas:.3f} ms")
    print(f"  SoL unfused  = {total_sol_prac:.3f} ms  → {total_sol_prac/total_meas*100:.1f}% of measured")
    print(f"  SoL fused    = {total_sol_fus:.3f} ms  → {total_sol_fus/total_meas*100:.1f}% of measured")
    print()
    print("Caveats:")
    print("  - n_active is ESTIMATED via coupon-collector; may over/under-count by 1-2 experts.")
    print("  - Weight counts assume full-expert fetch: for small M_local, CUTLASS may only")
    print("    fetch the specific tiles of an active expert that are touched by M<tile_M rows,")
    print("    which can be ~half the total weight. → our SoL may be overestimated by ~20-30%.")
    print("  - NCU profile (running in background) will give ground-truth dram__bytes.")


if __name__ == "__main__":
    main()
