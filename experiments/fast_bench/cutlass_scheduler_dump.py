"""Dump key CUTLASS files to understand the grouped/persistent tile scheduler."""
from pathlib import Path
import modal

app = modal.App("cutlass-sched-dump")
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
)


@app.function(image=image, gpu="B200:1", timeout=300)
def dump() -> str:
    import os, subprocess
    base = "/opt/conda/envs/py312/lib/python3.12/site-packages/flashinfer/data/cutlass/include"
    if not os.path.isdir(base):
        return f"no base at {base}"
    out = []
    paths = [
        "cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp",
        "cutlass/gemm/kernel/sm100_tile_scheduler.hpp",
        "cutlass/gemm/kernel/tile_scheduler_params.h",
    ]
    for p in paths:
        fp = os.path.join(base, p)
        if os.path.isfile(fp):
            out.append(f"\n===== {p} =====")
            with open(fp) as f:
                content = f.read()
            # just print first 200 lines
            lines = content.splitlines()
            out.append("\n".join(lines[:200]))
            out.append(f"... [truncated; {len(lines)} total lines]")
        else:
            out.append(f"missing: {p}")
    # find KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100 definition + impl
    r = subprocess.run(["grep", "-rln", "KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100"],
                       cwd=base, capture_output=True, text=True)
    out.append("\n## files referencing Blockwise1SmSm100:\n" + r.stdout)
    # find blockwise kernel impl
    r2 = subprocess.run(["grep", "-rlE", "class GemmUniversal[A-Za-z0-9_]*Blockwise|PtrArrayBlockwise"],
                        cwd=base, capture_output=True, text=True)
    out.append("\n## files with PtrArrayBlockwise impl:\n" + r2.stdout)
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(dump.remote())
