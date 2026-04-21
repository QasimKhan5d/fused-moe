"""Probe the CUTLASS Sm1xxBlkScaledConfig<32> SFA layout to understand indexing."""
import modal
app = modal.App("probe-sfa-layout")
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


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/mnt": trace_volume})
def run() -> str:
    import sys, os
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    ext = K._get_ext()

    lines = []
    for (m, n, k) in [(128, 128, 128), (128, 256, 256), (256, 128, 128), (440, 4096, 7168)]:
        off = ext.probe_mxf8_sfa_layout(m, n, k)
        lines.append(
            f"SFA layout (M={m}, N={n}, K={k}):\n"
            f"  total_size={off[0]}\n"
            f"  (0,0)={off[1]}  (0,1)={off[2]}  (0,31)={off[3]}  (0,32)={off[4]}\n"
            f"  (0,63)={off[5]}  (0,64)={off[6]}  (1,0)={off[7]}  (127,0)={off[8]}  (128,0)={off[9]}"
        )
    return "\n\n".join(lines)


@app.local_entrypoint()
def main():
    print(run.remote())
