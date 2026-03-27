"""
Minimal Nsight Compute smoke test on Modal B200.

Usage:
    modal run scripts/test_ncu_modal.py
"""

import modal

app = modal.App("test-ncu-modal")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
    .apt_install("wget", "gnupg")
    .run_commands(
        "wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /' > /etc/apt/sources.list.d/cuda.list",
        "apt-get update && apt-get install -y nsight-compute-2026.1.0",
    )
)

KERNEL = r"""
import torch
A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
for _ in range(3):
    torch.mm(A, B)
torch.cuda.synchronize()
torch.mm(A, B)
torch.cuda.synchronize()
"""


@app.function(image=image, gpu="B200:1", timeout=300)
def run():
    import glob
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ncu-smoke-") as tmpdir:
        kernel_path = os.path.join(tmpdir, "k.py")
        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(KERNEL)

        ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
        env = os.environ.copy()
        site_packages = next(
            p for p in sys.path if p.endswith("site-packages") and os.path.isdir(p)
        )
        lib_dirs = []
        for pattern in (
            os.path.join(site_packages, "nvidia", "*", "lib"),
            os.path.join(site_packages, "torch", "lib"),
            "/usr/lib/x86_64-linux-gnu",
        ):
            lib_dirs.extend(sorted(glob.glob(pattern)))
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

        r = subprocess.run(
            [
                ncu,
                "--set",
                "basic",
                "--target-processes",
                "all",
                "--kernel-name",
                "regex:.*gemm.*",
                "--launch-count",
                "1",
                "python",
                kernel_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

        print(f"exit: {r.returncode}")
        print(f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
        if r.stdout:
            print("STDOUT:")
            print(r.stdout[-5000:])
        if r.stderr:
            print("STDERR:")
            print(r.stderr[-2000:])
        subprocess.run(
            [
                "bash",
                "-lc",
                "for f in /tmp/nsight-compute-*.log; do [ -f \"$f\" ] && echo \"--- $f ---\" && sed -n '1,200p' \"$f\"; done",
            ],
            text=True,
            timeout=30,
            env=env,
            check=False,
        )


@app.local_entrypoint()
def main():
    run.remote()
