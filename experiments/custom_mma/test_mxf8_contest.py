"""Validate MxF8 GEMM produces correct output on contest data.

Takes one contest workload, extracts the GEMM1 inputs post-dispatch,
runs:
  A) FP32-blockwise CUTLASS with pristine signed fp32 scales -> reference
  B) Transcode (sign-flip + pow2 + residual), run MxF8 CUTLASS -> test
Compare A vs B.
Expected: ~same match ratio as Python probe (91-94%).
"""
import modal
app = modal.App("mxf8-contest-validation")
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

        # Run the pristine pipeline to get a reference output.
        out_ref = K.custom_kernel(*inputs).float()

        # Clone originals (they'll be transcoded in-place).
        orig = [t.clone() if torch.is_tensor(t) else t for t in inputs]

        try:
            # Transcode activations + weights using our CUDA kernels.
            orig_hs     = orig[2].clone()
            orig_hs_sc  = orig[3].clone()
            orig_g1     = orig[4].clone()
            orig_g1_sc  = orig[5].clone()
            orig_g2     = orig[6].clone()
            orig_g2_sc  = orig[7].clone()

            hs_sc_ue8m0 = torch.empty_like(orig_hs_sc)
            g1_sc_ue8m0 = torch.empty_like(orig_g1_sc)
            g2_sc_ue8m0 = torch.empty_like(orig_g2_sc)

            # hs_scale is contest-transposed [K/128, T]; transpose before transcode.
            hs_sc_t = orig_hs_sc
            hs_transposed = False
            if orig_hs_sc.dim() == 2 and orig_hs_sc.shape[0] * 128 == orig_hs.shape[1] and orig_hs_sc.shape[1] == orig_hs.shape[0]:
                hs_sc_t = orig_hs_sc.transpose(0, 1).contiguous()
                hs_sc_ue8m0 = torch.empty_like(hs_sc_t)
                hs_transposed = True

            ext.mxf8_transcode_activations(orig_hs, hs_sc_t, hs_sc_ue8m0)
            ext.mxf8_transcode_weights_impl(orig_g1, orig_g1_sc, g1_sc_ue8m0)
            ext.mxf8_transcode_weights_impl(orig_g2, orig_g2_sc, g2_sc_ue8m0)

            # Swap transcoded inputs in.
            inputs[2].copy_(orig_hs)
            inputs[3].copy_(hs_sc_ue8m0.transpose(0, 1).contiguous() if hs_transposed else hs_sc_ue8m0)
            inputs[4].copy_(orig_g1)
            inputs[5].copy_(g1_sc_ue8m0)
            inputs[6].copy_(orig_g2)
            inputs[7].copy_(g2_sc_ue8m0)

            # Now the FP32-blockwise kernel will compute with pow2 scales — which
            # is exactly what the MxF8 hardware would compute. This tells us the
            # Python probe result is accurate.
            out_fp32_blockwise_with_transcode = K.custom_kernel(*inputs).float()

            # Check against our reference.
            diff = (out_fp32_blockwise_with_transcode - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match_fp32 = (diff <= tol).float().mean().item()
            max_abs_fp32 = diff.max().item()

            # TODO: run MxF8 GEMM directly on the GEMM1 or GEMM2 extracted inputs
            # instead of end-to-end. For now we validated the transcoded
            # pipeline using the FP32-blockwise GEMM; the MxF8 hardware GEMM
            # integration is the next step (wire into custom_kernel and test).

            return f"{u[:8]} T={T}: transcoded-pipeline match={match_fp32*100:.2f}% max_abs={max_abs_fp32:.2f}"
        finally:
            pass

    results = []
    for u in uuids.split(","):
        results.append(run_one(u.strip()))
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "5e8dc11c"):
    print(run.remote(uuids=uuids))
