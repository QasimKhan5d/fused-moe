"""Save reference impl source to a file in /mnt to read offline."""
import modal
app = modal.App("save-ref")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
)


@app.function(image=image, timeout=120, volumes={"/mnt": trace_volume})
def save() -> str:
    from flashinfer_bench import TraceSet
    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions[
        "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]
    ref = definition.reference
    with open("/mnt/ref_impl.py", "w") as f:
        f.write(str(ref))
    return f"written {len(ref)} chars to /mnt/ref_impl.py"


@app.function(image=image, timeout=120, volumes={"/mnt": trace_volume})
def read() -> str:
    return open("/mnt/ref_impl.py").read()


@app.local_entrypoint()
def main():
    print(save.remote())
    print("\n=== REFERENCE SOURCE ===\n")
    print(read.remote())
