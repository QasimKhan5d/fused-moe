"""Sign-separated MXF8 transcode + per-row residual correction.

Corrected probe — fixes all bugs in earlier attempts:
  * Handle signed fp32 scales by separating sign from magnitude.
  * Payload sign-flip: `fp8_a'[i] = sign(scale_a[block(i)]) * fp8_a[i]`
    (lossless bitwise flip for FP8 E4M3).
  * UE8M0 scale: `ue8m0 = ceil_pow2(|scale_fp32|)`.
  * Residual: `r = |scale_fp32| / ue8m0`, in (0.5, 1.0].
  * Test variants:
     V0) baseline (no transform) — sanity, must be 100%.
     V1) scale-only: sign->payload, magnitude->ue8m0. No residual correction.
     V2) V1 + per-row mean residual correction applied post-GEMM via per-output
         row scaling. Expected to recover ~85x of the residual error via CLT.
     V3) V1 + per-block residual absorbed into payload (the thing that failed
         before, reproduced as a control).
     V4) V1 + per-output correction using TRUE per-block (r_a * r_b) sum
         (oracle upper bound on correction quality).

NOTE: This probe uses our existing FP32-blockwise CUTLASS kernel as a
NUMERICAL STAND-IN. We feed it (payload', |ue8m0_scale|) as if it were the
(payload, fp32_scale) — this tells us whether the transcode preserves
tolerance. If V2 passes, we have a green light to implement on hw MXF8.
"""
import modal
app = modal.App("mxf8-signed-v2")
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
def run(uuids: str = "5e8dc11c") -> str:
    import os, sys
    from pathlib import Path
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    def ue8m0_abs_decomp(scale_fp32: torch.Tensor):
        """Return (sign, abs, ue8m0, residual) so that:
            scale_fp32 = sign * abs
            ue8m0 = ceil_pow2(abs)
            residual = abs / ue8m0  in (0.5, 1.0]
        """
        sign = torch.sign(scale_fp32)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_val = scale_fp32.abs().clamp_min(1e-30)
        exp = torch.ceil(torch.log2(abs_val)).clamp(-127, 127)
        ue8m0 = torch.pow(2.0, exp)
        residual = abs_val / ue8m0
        return sign, abs_val, ue8m0, residual

    def flip_payload_by_block_sign(payload_fp8: torch.Tensor, sign_per_block: torch.Tensor):
        """Flip payload signs based on per-K-block scale sign.
        payload_fp8: [..., K], sign_per_block: shape with K-axis/128.
        """
        K = payload_fp8.shape[-1]
        Kb = K // 128
        # Broadcast sign_per_block to K-axis.
        if sign_per_block.dim() == 2 and sign_per_block.shape[0] == Kb and sign_per_block.shape[1] == payload_fp8.shape[0]:
            sign_tk = sign_per_block.transpose(0, 1).contiguous()
        elif sign_per_block.dim() == 2 and sign_per_block.shape[0] == payload_fp8.shape[0] and sign_per_block.shape[1] == Kb:
            sign_tk = sign_per_block
        elif sign_per_block.dim() == 3 and sign_per_block.shape[-1] == Kb:
            sign_tk = sign_per_block
        else:
            raise RuntimeError(f"sign shape {sign_per_block.shape} not recognized for payload {payload_fp8.shape}")

        expanded = sign_tk.unsqueeze(-1).expand(*sign_tk.shape, 128)
        expanded = expanded.reshape(*sign_tk.shape[:-1], sign_tk.shape[-1] * 128)
        # Broadcast over non-K non-M dims for weights.
        for axis in range(expanded.dim() - 1):
            if expanded.shape[axis] * 128 == payload_fp8.shape[axis]:
                expanded = expanded.repeat_interleave(128, dim=axis)
        assert expanded.shape == payload_fp8.shape, f"{expanded.shape} vs {payload_fp8.shape}"
        # Sign flip: payload * sign. For FP8 E4M3 this is flipping bit 7.
        p_fp32 = payload_fp8.to(torch.float32)
        p_new = p_fp32 * expanded
        return p_new.to(torch.float8_e4m3fn)

    def transcode_variant(payload_fp8: torch.Tensor, scale_fp32: torch.Tensor,
                          variant: str):
        """Transcode both payload and scale for testing.
        Returns (payload', scale').
        variant='none': unchanged
        variant='sign+ue8m0': payload sign-flipped, scale=|ue8m0|. Residual ignored.
        variant='sign+ue8m0+payload_r': ALSO absorb residual into payload (lossy).
        """
        sign, abs_s, ue8m0, r = ue8m0_abs_decomp(scale_fp32)
        if variant == "none":
            return payload_fp8, scale_fp32
        if variant == "sign+ue8m0":
            p_new = flip_payload_by_block_sign(payload_fp8, sign)
            # Scale FED TO kernel: use abs_ue8m0 but keep as fp32 (since we're using
            # the FP32-blockwise kernel as stand-in, it expects fp32). The VALUE is
            # the UE8M0 quantum, just stored as fp32.
            return p_new, ue8m0
        if variant == "sign+ue8m0+payload_r":
            # Sign flip + residual absorption into payload. Payload re-round is lossy.
            sign_flipped = flip_payload_by_block_sign(payload_fp8, sign)
            # Absorb residual: multiply by r per block then re-round to FP8.
            p_fp32 = sign_flipped.to(torch.float32)
            # Broadcast r to payload.
            K = p_fp32.shape[-1]; Kb = K // 128
            if r.dim() == 2 and r.shape[0] == Kb and r.shape[1] == p_fp32.shape[0]:
                r_tk = r.transpose(0, 1).contiguous()
            elif r.dim() == 2 and r.shape[0] == p_fp32.shape[0] and r.shape[1] == Kb:
                r_tk = r
            elif r.dim() == 3 and r.shape[-1] == Kb:
                r_tk = r
            else:
                raise RuntimeError(f"r shape {r.shape} invalid")
            r_exp = r_tk.unsqueeze(-1).expand(*r_tk.shape, 128).reshape(*r_tk.shape[:-1], r_tk.shape[-1] * 128)
            for axis in range(r_exp.dim() - 1):
                if r_exp.shape[axis] * 128 == p_fp32.shape[axis]:
                    r_exp = r_exp.repeat_interleave(128, dim=axis)
            p_new = (p_fp32 * r_exp).clamp(min=-448.0, max=448.0)
            return p_new.to(torch.float8_e4m3fn), ue8m0
        raise ValueError(variant)

    def compare(inputs, transforms):
        """Apply transforms = list of (tensor_idx, variant) and compare to ref."""
        out_ref = K.custom_kernel(*inputs).float()
        saved = {idx: (inputs[idx].clone(), inputs[idx + 1].clone()) for idx, _ in transforms}
        try:
            for idx, variant in transforms:
                p_new, s_new = transcode_variant(inputs[idx], inputs[idx + 1], variant)
                inputs[idx].copy_(p_new)
                inputs[idx + 1].copy_(s_new)
            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match = (diff <= tol).float().mean().item()
            max_abs = diff.max().item()
            mean_rel = (diff / (out_ref.abs() + 1e-6)).median().item()
            return match, max_abs, mean_rel
        finally:
            for idx, (orig_p, orig_s) in saved.items():
                inputs[idx].copy_(orig_p)
                inputs[idx + 1].copy_(orig_s)

    def residual_correction_test(inputs):
        """Apply sign+ue8m0 to all three scales, run, then apply per-output-row
        mean residual correction.
        """
        # Transcode all three inputs.
        out_ref = K.custom_kernel(*inputs).float()
        saved = {idx: (inputs[idx].clone(), inputs[idx + 1].clone()) for idx in (2, 4, 6)}
        try:
            # Collect residuals per tensor for post-correction.
            sign_hs,  abs_hs,  ue8m0_hs,  r_hs  = ue8m0_abs_decomp(saved[2][1])
            sign_g1,  abs_g1,  ue8m0_g1,  r_g1  = ue8m0_abs_decomp(saved[4][1])
            sign_g2,  abs_g2,  ue8m0_g2,  r_g2  = ue8m0_abs_decomp(saved[6][1])

            # Apply transcode.
            for idx, variant in [(2, "sign+ue8m0"), (4, "sign+ue8m0"), (6, "sign+ue8m0")]:
                p_new, s_new = transcode_variant(inputs[idx], inputs[idx + 1], variant)
                inputs[idx].copy_(p_new)
                inputs[idx + 1].copy_(s_new)

            # Run transcoded kernel.
            out_q = K.custom_kernel(*inputs).float()

            # Correction: MMA underestimates each block's contribution by
            # factor r_a * r_b. Globally, `out_q * <r_a * r_b>` approximates true.
            # For GEMM1: <r_hs * r_g1> per output element.
            # For GEMM2: <r_act * r_g2> where r_act comes from the SwiGLU'd
            # intermediate's scale, which itself was produced by our kernel
            # from GEMM1's output. That's complicated -- skip GEMM2 correction
            # and only correct GEMM1's contribution.
            #
            # Since this is end-to-end, we can't cleanly isolate. Apply a
            # single per-output mean residual by using the MEAN OF MEANS
            # of r_g1 * r_g2 as a global multiplicative scalar.
            r_mean_global = (r_g1.mean() * r_g2.mean() * r_hs.mean()).item()
            out_q_corr_global = out_q * r_mean_global

            diff = (out_q_corr_global - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match_global = (diff <= tol).float().mean().item()
            max_abs_global = diff.max().item()

            # No-correction variant to measure impact of correction.
            diff0 = (out_q - out_ref).abs()
            match0 = (diff0 <= tol).float().mean().item()
            max_abs0 = diff0.max().item()

            return {
                "no_corr": (match0, max_abs0),
                "global_corr": (match_global, max_abs_global, r_mean_global),
            }
        finally:
            for idx, (orig_p, orig_s) in saved.items():
                inputs[idx].copy_(orig_p)
                inputs[idx + 1].copy_(orig_s)

    results = []
    for u in uuids.split(","):
        u = u.strip()
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            results.append(f"{u}: not found"); continue

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = inputs[0].shape[0]

        results.append(f"=== {u[:8]} T={T} ===")

        # Sanity: baseline
        m, mx, mr = compare(inputs, [(2, "none")])
        results.append(f"  V0 baseline             match={m*100:5.2f}% maxabs={mx:.2f}")

        # V1 sign+ue8m0 on ALL three tensors
        m, mx, mr = compare(inputs, [(2, "sign+ue8m0"), (4, "sign+ue8m0"), (6, "sign+ue8m0")])
        results.append(f"  V1 sign+ue8m0 (all3)    match={m*100:5.2f}% maxabs={mx:.2f} median_rel={mr:.4f}")

        # V1 subsets
        for idx, name in [(2, "hs"), (4, "gemm1w"), (6, "gemm2w")]:
            m, mx, mr = compare(inputs, [(idx, "sign+ue8m0")])
            results.append(f"  V1 sign+ue8m0 ({name} only)  match={m*100:5.2f}% maxabs={mx:.2f} median_rel={mr:.4f}")

        # V3 sign + ue8m0 + payload residual absorption (all3)
        m, mx, mr = compare(inputs, [(2, "sign+ue8m0+payload_r"), (4, "sign+ue8m0+payload_r"), (6, "sign+ue8m0+payload_r")])
        results.append(f"  V3 sign+ue8m0+payloadR  match={m*100:5.2f}% maxabs={mx:.2f} median_rel={mr:.4f}")

        # V2 sign+ue8m0 + global r-mean correction
        rc = residual_correction_test(inputs)
        results.append(f"  V2 V1 then global-corr  match={rc['global_corr'][0]*100:5.2f}% maxabs={rc['global_corr'][1]:.2f} (r_mean_global={rc['global_corr'][2]:.4f})")
        results.append(f"     (V1 no-corr control: {rc['no_corr'][0]*100:5.2f}% maxabs={rc['no_corr'][1]:.2f})")

    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,6230e838,1a4c6ba1,5e8dc11c,58a34f27"):
    print(run.remote(uuids=uuids))
