"""
Inspect the contest's reference implementation for the MoE FP8 definition.
We need to know whether the reference uses fp32 scales directly or applies
some quantization (e.g., ue8m0). This determines whether ue8m0 quantization
in our kernel is compatible with the reference output.
"""
import modal

app = modal.App("inspect-ref")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .entrypoint([])
    .pip_install("flashinfer-bench @ git+https://github.com/flashinfer-ai/flashinfer-bench.git")
)


@app.function(image=image, gpu="B200:1", timeout=300, volumes={"/mnt": trace_volume})
def inspect() -> str:
    from flashinfer_bench import TraceSet
    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    out = []
    out.append(f"Definition: {def_name}")
    out.append(f"type: {type(definition).__name__}")
    out.append("")

    # Print the reference impl
    # Definitions usually have a `reference` field with source code
    for attr in dir(definition):
        if attr.startswith("_"): continue
        try:
            val = getattr(definition, attr)
        except Exception:
            continue
        if callable(val): continue
        s = str(val)
        if len(s) > 200:
            s = s[:200] + f" ... [{len(s)} total chars]"
        out.append(f"  .{attr} = {s}")

    # Inputs / outputs metadata
    out.append("")
    out.append("=== Reference source ===")
    try:
        ref = definition.reference
        if hasattr(ref, "source"):
            out.append(ref.source)
        else:
            out.append(str(ref))
    except AttributeError:
        pass

    # Also print the full dict/object
    out.append("")
    out.append("=== dict ===")
    try:
        d = definition.model_dump() if hasattr(definition, "model_dump") else vars(definition)
        for k, v in d.items():
            s = str(v)
            out.append(f"{k}:")
            # Print full if small, else truncate
            if len(s) < 3000:
                out.append(f"  {s}")
            else:
                out.append(f"  {s[:3000]} ... [truncated, {len(s)} chars]")
    except Exception as e:
        out.append(f"dict dump failed: {e!r}")

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(inspect.remote())
