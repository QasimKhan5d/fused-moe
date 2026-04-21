"""
More careful test of UE8M0 tolerance. Previously I used round-to-nearest which
biases errors. Try variants:
  1. Round-to-nearest ue8m0 (baseline, what I tried before)
  2. Ceil ue8m0 (DeepGEMM convention — scale always >= fp32 scale, data never clips)
  3. Floor ue8m0
  4. Round-half-up
  5. 'Nearest-representable' via bit manipulation (correct ue8m0 semantics)
  6. Hybrid: apply ue8m0 to scales, then multiply output by fp32_sa / ue8m0_sa (compensation)
  7. Use finer block sizes (split 128-block scales into 4 identical 32-block MXFP8)

Goal: find any scheme that gets >= 90% match_ratio on contest workloads.
"""
import modal
app = modal.App("ue8m0-careful")
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


@app.function(image=image, gpu="B200:1", timeout=1200, volumes={"/mnt": trace_volume})
def run(uuids: str = "e05c6c03,1a4c6ba1,5e8dc11c") -> str:
    import os, sys, torch, math
    from pathlib import Path
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    # Quantization variants
    def q_round(x):
        """Round to nearest power-of-2."""
        x = x.clamp_min(1e-30)
        exp = torch.round(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def q_ceil(x):
        """Always round up (DeepGEMM convention)."""
        x = x.clamp_min(1e-30)
        exp = torch.ceil(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def q_floor(x):
        """Always round down."""
        x = x.clamp_min(1e-30)
        exp = torch.floor(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def q_hybrid_compensated(x):
        """Use ue8m0 for MMA but store residual fp32 factor.
        Returns (ue8m0_scale, residual_factor) where
            original_scale ≈ ue8m0_scale * residual_factor
            residual_factor ∈ [0.707, 1.414] (half-octave)
        """
        x = x.clamp_min(1e-30)
        exp = torch.round(torch.log2(x)).clamp(-127, 127)
        q = torch.pow(2.0, exp)
        residual = x / q  # in [0.707, 1.414] for round-to-nearest
        return q, residual

    def compare(inputs, kind: str):
        """Run kernel with quantized scales, return (match%, max_abs, mean_rel)."""
        # Reference: fp32 scales (our current kernel)
        out_ref = K.custom_kernel(*inputs).float()

        # Save originals
        orig = [inputs[3].clone(), inputs[5].clone(), inputs[7].clone()]

        try:
            if kind == "round":
                inputs[3].copy_(q_round(orig[0]))
                inputs[5].copy_(q_round(orig[1]))
                inputs[7].copy_(q_round(orig[2]))
            elif kind == "ceil":
                inputs[3].copy_(q_ceil(orig[0]))
                inputs[5].copy_(q_ceil(orig[1]))
                inputs[7].copy_(q_ceil(orig[2]))
            elif kind == "floor":
                inputs[3].copy_(q_floor(orig[0]))
                inputs[5].copy_(q_floor(orig[1]))
                inputs[7].copy_(q_floor(orig[2]))
            elif kind == "hybrid":
                # Scale by (ue8m0 * residual) == original exactly, so just
                # baseline identity. Keep as sanity check: kernel uses fp32.
                inputs[3].copy_(orig[0])
                inputs[5].copy_(orig[1])
                inputs[7].copy_(orig[2])

            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match = (diff <= tol).float().mean().item()
            max_abs = diff.max().item()
            mean_rel = (diff / (out_ref.abs() + 1e-6)).median().item()
            return match, max_abs, mean_rel
        finally:
            inputs[3].copy_(orig[0])
            inputs[5].copy_(orig[1])
            inputs[7].copy_(orig[2])

    # Also quickly inspect scale distribution
    def scale_stats(x: torch.Tensor):
        """Distribution of how 'power-of-2 friendly' the scales are."""
        x = x.flatten()
        l = torch.log2(x.clamp_min(1e-30))
        # Distance to nearest integer in log2 space
        frac = (l - torch.round(l)).abs()
        return frac.mean().item(), frac.max().item()

    results = []
    results.append("Variant analysis: how lossy is each ue8m0 quantization vs the fp32 reference?")
    results.append(f"{'uuid':<10} {'T':>5} {'variant':<12} {'match%':>7} {'max_abs':>10} {'mean_rel':>10} "
                   f"{'sf_l2frac_avg':>13}")
    results.append("-" * 80)

    for u in uuids.split(","):
        u = u.strip()
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            results.append(f"{u}: not found"); continue

        loaded_st = load_safetensors(definition, wobj, Path("/mnt/mlsys26-contest")) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = int(inputs[0].shape[0])

        # Scale stats
        avg_frac_hs, _ = scale_stats(inputs[3])
        avg_frac_g1, _ = scale_stats(inputs[5])
        avg_frac_g2, _ = scale_stats(inputs[7])
        avg = (avg_frac_hs + avg_frac_g1 + avg_frac_g2) / 3

        for kind in ("round", "ceil", "floor"):
            m, mx, mr = compare(inputs, kind)
            ok = "PASS" if m >= 0.9 else "FAIL"
            results.append(
                f"{u[:8]:<10} {T:>5} {kind:<12} {m*100:>6.1f}% {mx:>10.2f} {mr:>10.4f} "
                f"{avg:>12.3f}  [{ok}]"
            )

    results.append("")
    results.append("sf_l2frac_avg = average |log2(scale) - round(log2(scale))|")
    results.append("  0.0 = all scales are exact powers of 2 (ue8m0 lossless)")
    results.append("  0.5 = worst case (scales halfway between powers of 2)")
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,1a4c6ba1,5e8dc11c"):
    print(run.remote(uuids=uuids))
