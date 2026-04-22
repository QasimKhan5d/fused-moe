"""List CUTLASS SM100 kernel schedules available in the contest container."""
from pathlib import Path
import modal

app = modal.App("cutlass-schedule-probe")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
)


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/mnt": trace_volume})
def probe() -> str:
    import os, subprocess, glob
    out = []
    # 1. Find CUTLASS install location
    bases = [
        "/opt/conda/envs/py312/lib/python3.12/site-packages/flashinfer/data/cutlass",
        "/opt/conda/envs/py312/lib/python3.12/site-packages/nvidia/cutlass",
        "/opt/conda/envs/py312/lib/python3.12/site-packages/cutlass",
        "/opt/conda/envs/py312/lib/python3.12/site-packages/flashinfer_bench/third_party/cutlass",
    ]
    found_include = None
    for b in bases:
        if os.path.isdir(b):
            out.append(f"[exists] {b}")
            for sub in ("include", "cutlass/include"):
                if os.path.isdir(os.path.join(b, sub)):
                    found_include = os.path.join(b, sub)
                    break
    if not found_include:
        # wide search
        r = subprocess.run(["find", "/opt", "-name", "dispatch_policy.hpp", "-path", "*cutlass*"],
                           capture_output=True, text=True, timeout=60)
        out.append("find dispatch_policy.hpp: " + r.stdout[:2000])
        # pick first
        for line in r.stdout.splitlines():
            if "cutlass" in line:
                found_include = os.path.dirname(os.path.dirname(line))
                break
    out.append(f"using include: {found_include}")
    if not found_include:
        return "\n".join(out)

    # 2. Grep all KernelPtrArray* names in dispatch_policy.hpp
    dp = os.path.join(found_include, "cutlass/gemm/dispatch_policy.hpp")
    if os.path.isfile(dp):
        r = subprocess.run(["grep", "-n", "-E", "(struct|using)\\s+KernelPtrArray.*Sm100", dp],
                           capture_output=True, text=True)
        out.append("\n## KernelPtrArray*Sm100 schedules in dispatch_policy.hpp:\n" + r.stdout[:5000])
    # 3. Broad scan across all .hpp for Sm100 blockwise schedules
    r = subprocess.run(
        ["grep", "-rn", "-E", "KernelPtrArray[A-Za-z]+Blockwise[A-Za-z0-9_]*Sm100|KernelScheduleSm100|Sm100Blockwise"],
        cwd=found_include, capture_output=True, text=True, timeout=60,
    )
    out.append("\n## Sm100 blockwise schedules (wide scan):\n" + r.stdout[:6000])
    # 4. Look for persistent/pingpong/cooperative in Sm100 contexts
    r2 = subprocess.run(
        ["grep", "-rlE", "(Pingpong|Cooperative|Persistent)[A-Za-z0-9_]*Sm100|Sm100[A-Za-z0-9_]*(Pingpong|Cooperative|Persistent)"],
        cwd=found_include, capture_output=True, text=True, timeout=60,
    )
    out.append("\n## Files mentioning Sm100 + (Pingpong|Cooperative|Persistent):\n" + r2.stdout[:3000])
    # 5. Actual schedule class names we can use
    r3 = subprocess.run(
        ["grep", "-rh", "-E", "^struct Kernel[A-Za-z0-9_]*Sm100[^A-Za-z0-9_]"],
        cwd=found_include, capture_output=True, text=True, timeout=60,
    )
    lines = sorted(set(r3.stdout.splitlines()))
    out.append("\n## Unique 'struct Kernel...Sm100' definitions:\n" + "\n".join(lines[:120]))
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
