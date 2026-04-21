"""
Force a specific CUTLASS_CFG_MODE on large workloads by monkeypatching kernel.py
just before modal packs the solution. Runs `modal run scripts/run_modal.py
--workload-uuids=<large>` for each mode and logs results.

Usage:
    python experiments/v17_v18_v19/bench_cfg_mode.py A K V J X
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KERNEL_PY = REPO / "solution" / "python" / "kernel.py"
MARKER = "# __CFG_MODE_FORCE__"

LARGE_UUIDS = "5e8dc11c,58a34f27,1a4c6ba1"  # T=14107, 11948, 9601


def set_cfg_mode(mode: str):
    """Inject `os.environ['CUTLASS_CFG_MODE'] = mode` near top of custom_kernel."""
    src = KERNEL_PY.read_text()
    # Remove any prior force
    src = re.sub(rf"^.*{re.escape(MARKER)}.*\n", "", src, flags=re.MULTILINE)
    if mode == "A":
        # baseline: no force
        KERNEL_PY.write_text(src)
        return
    # Insert one-time env setting at module top (after imports)
    inject = (
        f"import os as _os_cfg_{mode}\n"
        f"_os_cfg_{mode}.environ['CUTLASS_CFG_MODE'] = {mode!r}  {MARKER}\n"
    )
    # Place after the initial torch import line.
    src = src.replace("import torch\n", f"import torch\n{inject}", 1)
    KERNEL_PY.write_text(src)


def main():
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["A", "K", "V", "J", "X"]
    results = {}
    for mode in modes:
        print(f"\n=== Running CFG_MODE={mode} ===", flush=True)
        set_cfg_mode(mode)
        t0 = time.time()
        proc = subprocess.run(
            ["modal", "run", "scripts/run_modal.py", "--workload-uuids", LARGE_UUIDS],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.time() - t0
        log = proc.stdout + proc.stderr
        out_path = REPO / f"experiments/v17_v18_v19/bench_cfg_{mode}.log"
        out_path.write_text(log)
        speedups = []
        for line in log.splitlines():
            m = re.search(r"Workload (\w+)\.\.\.:.*\|\s+([\d.]+)x speedup", line)
            if m:
                speedups.append((m.group(1), float(m.group(2))))
        results[mode] = (elapsed, speedups)
        print(f"  elapsed={elapsed:.1f}s, speedups={speedups}")

    # Cleanup: revert to baseline
    set_cfg_mode("A")

    print("\n=== Summary ===")
    for mode, (t, sps) in results.items():
        avg = sum(x for _, x in sps) / max(len(sps), 1)
        print(f"  MODE={mode}: avg={avg:.2f}x, details={sps}")


if __name__ == "__main__":
    main()
