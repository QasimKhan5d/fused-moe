"""Dump the full reference implementation source."""
import modal
app = modal.App("dump-ref")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
)


@app.function(image=image, timeout=120, volumes={"/mnt": trace_volume})
def dump() -> str:
    from flashinfer_bench import TraceSet
    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions[
        "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]
    ref = definition.reference
    # Try a few attrs
    for attr in ["source", "code", "func", "impl", "body"]:
        v = getattr(ref, attr, None)
        if v:
            return f"--- .{attr} ---\n{v}"
    return f"type(ref)={type(ref)}, dir(ref)={dir(ref)}, str(ref)=\n{ref}"


@app.local_entrypoint()
def main():
    print(dump.remote())
