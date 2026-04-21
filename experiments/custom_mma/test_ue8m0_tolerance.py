"""
Critical pre-check for Path B: does converting our FP32 block-scales to UE8M0
(power-of-2) pass the contest correctness bar?

The SM100 hardware MMA supports block-scale in UE8M0 format (what DeepGEMM
uses). Our contest data uses FP32 scales. UE8M0 has ~log2(scale) precision
i.e. rounds to the nearest power of 2. This may lose some accuracy.

If UE8M0-quantized output passes the contest tolerance (atol=1.0, rtol=0.3,
required_matched_ratio=0.9), we can use DeepGEMM's approach and adapt it.
Otherwise we need a true fp32-blockwise custom kernel.

Test: run our current kernel BUT with scales round-to-power-of-2, compare to
the fp32 reference.
"""
import modal

app = modal.App("ue8m0-tolerance")
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


@app.function(image=image, gpu="B200:1", timeout=900, volumes={"/mnt": trace_volume})
def check(uuids: str = "e05c6c03,b8f4f012,8cba5890,6230e838,81955b1e,8f1ff9f1,1a4c6ba1") -> str:
    """Round all FP32 block-scales to the nearest power-of-2 (ue8m0 format)
    and compare output to the fp32-scales reference."""
    import os, sys, torch
    from pathlib import Path
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions["moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]

    def quantize_ue8m0(x: torch.Tensor) -> torch.Tensor:
        """Round positive fp32 tensor to nearest power-of-2.
        ue8m0 has no mantissa, just the 8-bit exponent."""
        x = x.clamp_min(1e-30)
        # Round log2(x) to nearest integer, then 2^that
        exp = torch.round(torch.log2(x))
        # ue8m0 exponent range: 0..255 (values 2^-127 .. 2^128)
        exp = exp.clamp(-127, 128)
        return torch.pow(2.0, exp)

    def ue8m0_round_sample(sc: torch.Tensor) -> torch.Tensor:
        """Quantize, showing worst-case error info."""
        q = quantize_ue8m0(sc)
        rel_err = ((q - sc).abs() / sc.clamp_min(1e-30)).flatten()
        return q, rel_err.max().item(), rel_err.mean().item()

    out = []
    out.append(f"Contest tolerance: atol=1.0, rtol=0.3, match_ratio>=0.9")
    out.append(f"Testing: run kernel with ue8m0-rounded scales, compare to fp32-scale ref")
    out.append("=" * 100)
    out.append(f"{'uuid':<10} {'T':>5} {'sf_max_rel':>10} {'sf_mean_rel':>11} "
               f"{'max_abs':>10} {'mean_rel':>10} {'match%':>7} {'verdict':<15}")
    out.append("-" * 100)

    for u in uuids.split(","):
        u = u.strip()
        wobj = None
        for wl in trace_set.workloads.get(
                "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048", []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            out.append(f"{u}: NOT FOUND"); continue

        loaded_st = load_safetensors(definition, wobj, Path("/mnt/mlsys26-contest")) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        T = int(inputs[0].shape[0])

        # Run with original fp32 scales (reference)
        out_fp32 = K.custom_kernel(*inputs).clone()

        # Quantize scales: inputs 3 = hidden_states_scale (fp32), 5 = gemm1_weights_scale, 7 = gemm2_weights_scale
        orig_hs_sc = inputs[3].clone()
        orig_g1_sc = inputs[5].clone()
        orig_g2_sc = inputs[7].clone()

        hs_q, hs_max, hs_mean = ue8m0_round_sample(orig_hs_sc)
        g1_q, g1_max, g1_mean = ue8m0_round_sample(orig_g1_sc)
        g2_q, g2_max, g2_mean = ue8m0_round_sample(orig_g2_sc)
        sf_max = max(hs_max, g1_max, g2_max)
        sf_mean = (hs_mean + g1_mean + g2_mean) / 3

        # Swap in quantized scales
        inputs_q = list(inputs)
        inputs_q[3] = hs_q
        inputs_q[5] = g1_q
        inputs_q[7] = g2_q
        out_ue8m0 = K.custom_kernel(*inputs_q)

        # Compare to contest reference: atol=1.0, rtol=0.3
        diff = (out_ue8m0.float() - out_fp32.float()).abs()
        tol = 1.0 + 0.3 * out_fp32.abs().float()
        match = (diff <= tol).float().mean().item()
        max_abs = diff.max().item()
        mean_rel = (diff / (out_fp32.abs().float() + 1e-6)).median().item()

        verdict = "PASS" if match >= 0.9 else "FAIL"
        out.append(
            f"{u[:8]:<10} {T:>5d} {sf_max*100:>9.2f}% {sf_mean*100:>10.2f}% "
            f"{max_abs:>10.2f} {mean_rel:>10.4f} {match*100:>6.1f}% {verdict:<15}"
        )

        # Restore originals for next iter
        inputs[3].copy_(orig_hs_sc)
        inputs[5].copy_(orig_g1_sc)
        inputs[7].copy_(orig_g2_sc)

    out.append("-" * 100)
    out.append("sf_max_rel = worst-case relative error on a single scale after ue8m0 round")
    out.append("sf_mean_rel = avg relative error on scales")
    out.append("match% = % of output elements within atol=1.0 + 0.3*|ref|. need >= 90%")
    return "\n".join(out)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,b8f4f012,8cba5890,6230e838,81955b1e,8f1ff9f1,1a4c6ba1,58a34f27,5e8dc11c"):
    print(check.remote(uuids=uuids))
