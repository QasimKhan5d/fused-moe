"""Fast-iteration client for bench_server.py.

Flow:
  1. Read local solution/python/kernel.py
  2. Send to the warm Modal bench server
  3. Server writes the file, reloads, benches, returns results
  4. Print speedup/match table

Usage:
  # Capture the baseline once (baseline = USE_MXF8 off).
  python experiments/fast_bench/bench_client.py --save-baseline

  # Fast dev iteration (quick bench, compare against cached baseline).
  python experiments/fast_bench/bench_client.py --quick

  # Full precision bench.
  python experiments/fast_bench/bench_client.py

  # Subset of workloads.
  python experiments/fast_bench/bench_client.py --uuids 5e8dc11c,58a34f27

  # Change env vars sent to the server.
  python experiments/fast_bench/bench_client.py --env USE_MXF8=1 MXF8_MIN_T=4096
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "fused-moe-bench-server"
CLS_NAME = "BenchServer"

KERNEL_PATH = Path(
    "/home/bits/go/src/github.com/DataDog/experimental/users/qasim.khan"
    "/fused-moe/solution/python/kernel.py"
)


def parse_env_list(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--env arg {it!r} must be KEY=VALUE")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def format_results(payload: dict, label: str, show_baseline: bool) -> str:
    results = payload["results"]
    compare = payload.get("compare") or {}

    have_baseline = bool(compare) and show_baseline
    lines = []
    lines.append(f"server reload={payload.get('reload_s', 0):.1f}s  "
                 f"bench={payload.get('bench_s', 0):.1f}s  label={label}")
    if have_baseline:
        lines.append(f"{'uuid':8} {'T':>6}  {'base_ms':>8} {'var_ms':>8} "
                     f"{'x':>5}  {'match%':>6}  {'max_abs':>10}  status")
    else:
        lines.append(f"{'uuid':8} {'T':>6}  {'lat_ms':>8}  status")
    lines.append("-" * (80 if have_baseline else 48))

    speedups = []
    sorted_u = sorted(results.keys(),
                      key=lambda u: results[u].get("T", 0) or 0)
    for u in sorted_u:
        r = results[u]
        if r.get("error"):
            lines.append(f"{u:8} T={r.get('T', '?'):>4}  "
                         f"ERROR: {r['error'][:60]}")
            continue
        T = r["T"]
        lat = r["lat_ms"]
        if have_baseline and u in compare:
            c = compare[u]
            b_lat = c["b_lat_ms"]
            matched = c["matched"]
            max_abs = c["max_abs"]
            passed = "PASS" if matched >= 0.9 else ("FAIL" if matched >= 0 else "SHAPE")
            sp = b_lat / lat if lat > 0 else 0
            speedups.append(sp)
            lines.append(
                f"{u:8} T={T:>4}  {b_lat:>8.3f} {lat:>8.3f} {sp:>5.2f} "
                f" {matched*100:>5.1f}  {max_abs:>10.2f}  [{passed}]")
        else:
            lines.append(f"{u:8} T={T:>4}  {lat:>8.3f}  ok")

    if speedups:
        import statistics as st
        lines.append("-" * 80)
        lines.append(
            f"geo_mean_speedup={st.geometric_mean(speedups):.3f}x "
            f"arith={sum(speedups)/len(speedups):.3f}x "
            f"min={min(speedups):.3f}x max={max(speedups):.3f}x")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="warmup=1, iters=5 for sub-second bench loop")
    ap.add_argument("--uuids", type=str, default=None,
                    help="Comma-separated uuid prefixes to bench (default: all 19)")
    ap.add_argument("--env", nargs="*", default=[],
                    help="KEY=VALUE env vars passed to server")
    ap.add_argument("--save-baseline", action="store_true",
                    help="Capture current kernel output+latency as baseline "
                    "(implies USE_MXF8=0 unless you pass --env).")
    ap.add_argument("--label", type=str, default="variant")
    ap.add_argument("--kernel-path", type=str, default=str(KERNEL_PATH))
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()

    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)
    iters  = args.iters  if args.iters  is not None else (5 if args.quick else 20)

    env = parse_env_list(args.env)
    if args.save_baseline and "USE_MXF8" not in env:
        env["USE_MXF8"] = "0"

    kernel_src = Path(args.kernel_path).read_text()
    t0 = time.time()
    cls = modal.Cls.from_name(APP_NAME, CLS_NAME)
    server = cls()
    payload = server.bench.remote(
        kernel_src=kernel_src,
        uuids=args.uuids,
        warmup=warmup,
        iters=iters,
        env=env,
        save_baseline=args.save_baseline,
        compare_to_baseline=not args.save_baseline,
        label=args.label,
    )
    total = time.time() - t0

    print(format_results(payload, args.label, show_baseline=not args.save_baseline))
    print(f"\nclient total: {total:.1f}s")


if __name__ == "__main__":
    main()
