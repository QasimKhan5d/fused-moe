"""Test CUDA transcode kernel produces the same result as the Python probe.

Step 1 of MxF8 integration: validate the CUDA transcode kernel is correct by
feeding its output back into our existing FP32-blockwise kernel and confirming
it matches the Python V3 probe's 91.8% at T=14107.

If CUDA transcode == Python V3: we've validated the GPU kernel.
If they differ: bug in the CUDA transcode kernel.
"""
import modal
app = modal.App("mxf8-cuda-transcode-test")
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


@app.function(image=image, gpu="B200:1", timeout=1800, volumes={"/mnt": trace_volume})
def run(uuids: str) -> str:
    import os, sys
    from pathlib import Path
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    ext = K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    # Ensure the new symbols are exposed.
    assert hasattr(ext, "mxf8_transcode_activations"), "mxf8_transcode_activations missing"
    assert hasattr(ext, "mxf8_transcode_weights_impl"), "mxf8_transcode_weights_impl missing"

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    def run_one(u: str):
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            return f"{u}: not found"

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = inputs[0].shape[0]

        # Reference: pristine fp32 scales, unmodified kernel.
        out_ref = K.custom_kernel(*inputs).float()

        # Clone originals since transcode is IN-PLACE.
        orig_hs     = inputs[2].clone()
        orig_hs_sc  = inputs[3].clone()
        orig_g1     = inputs[4].clone()
        orig_g1_sc  = inputs[5].clone()
        orig_g2     = inputs[6].clone()
        orig_g2_sc  = inputs[7].clone()

        try:
            # Allocate |scale| output buffers.
            hs_sc_abs = torch.empty_like(orig_hs_sc)
            g1_sc_abs = torch.empty_like(orig_g1_sc)
            g2_sc_abs = torch.empty_like(orig_g2_sc)

            # hs_scale shape: contest stores as [K/128, T]. Transcode activation
            # helper expects [M, K/128]. Transpose first.
            hs_sc_for_transcode = orig_hs_sc
            hs_scale_is_transposed = False
            if orig_hs_sc.dim() == 2 and orig_hs_sc.shape[0] * 128 == orig_hs.shape[1] and orig_hs_sc.shape[1] == orig_hs.shape[0]:
                hs_sc_for_transcode = orig_hs_sc.transpose(0, 1).contiguous()
                hs_scale_is_transposed = True
                hs_sc_abs = torch.empty_like(hs_sc_for_transcode)

            # Transcode activations: payload = hidden_states, scale = hs_scale (T-major).
            ext.mxf8_transcode_activations(orig_hs, hs_sc_for_transcode, hs_sc_abs)
            # Transcode weights (E, N, K) format.
            ext.mxf8_transcode_weights_impl(orig_g1, orig_g1_sc, g1_sc_abs)
            ext.mxf8_transcode_weights_impl(orig_g2, orig_g2_sc, g2_sc_abs)

            # Swap in transcoded (payload', |scale|).
            inputs[2].copy_(orig_hs)  # transcoded in place
            if hs_scale_is_transposed:
                inputs[3].copy_(hs_sc_abs.transpose(0, 1).contiguous())
            else:
                inputs[3].copy_(hs_sc_abs)
            inputs[4].copy_(orig_g1)
            inputs[5].copy_(g1_sc_abs)
            inputs[6].copy_(orig_g2)
            inputs[7].copy_(g2_sc_abs)

            # Run through existing FP32-blockwise kernel.
            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match = (diff <= tol).float().mean().item()
            max_abs = diff.max().item()
            mean_rel = (diff / (out_ref.abs() + 1e-6)).median().item()
            return f"{u[:8]} T={T}  match={match*100:5.2f}%  max_abs={max_abs:.2f}  mean_rel={mean_rel:.4f}"
        finally:
            # Restore inputs for any subsequent test.
            # (orig_* were CLONED from inputs, so we need to re-clone them from
            # the transcoded payloads' originals. But we already transcoded
            # in-place on orig_hs/orig_g1/orig_g2 clones, so `inputs[2]` etc
            # currently hold transcoded values. Need to re-gen inputs to restore.
            pass

    results = []
    for u in uuids.split(","):
        results.append(run_one(u.strip()))
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "5e8dc11c,58a34f27,1a4c6ba1,6230e838"):
    print(run.remote(uuids=uuids))
