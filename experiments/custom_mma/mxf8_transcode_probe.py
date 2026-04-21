"""MXF8 transcode probe for contest FP32 per-128 scales.

Question: can we re-encode (fp8 payload, fp32 per-128 block scale) into
(fp8 payload', ue8m0 scale) such that the MXF8 hardware block-scale MMA
produces outputs within contest tolerance (atol=1, rtol=0.3, matched>=0.9)?

Hardware path assumed: SM100 `tcgen05.mma.kind.mxf8f6f4.block_scale` with
UE8M0 scales at sf_vec_size=32. That means our 128-wide FP32 block must be
split into 4 contiguous 32-element sub-blocks. The scale for each sub-block
is identical (= UE8M0(original fp32 scale)), but the payload is re-encoded
per sub-block to partially recover precision lost to the UE8M0 round.

Transcode recipe per 128-block:
    s_fp32   : original FP32 block scale
    s_hw     : UE8M0 approximation of s_fp32 (pow-of-2)
    r        : residual = s_fp32 / s_hw           (in [0.5, 1.0] for ceil, [0.7, 1.4] for round)
    x_fp8    : contest FP8 payload
    x_real   : x_fp8 * s_fp32                     (contest decoded value)
    x_new_f  : x_real / s_hw = x_fp8 * r          (desired FP8 input to MXF8 MMA)
    x_new_fp8 : round_fp8(x_new_f)                 (actual MXF8 operand)

Then MXF8 MMA computes:
    x_new_fp8 * s_hw  ~=  round_fp8(x_fp8 * r) * s_hw  ~=  x_real    (if round error small)

Error sources:
  1. FP8 re-rounding: when r > 1, some values overflow 448 and clip.
  2. Scale quantization: UE8M0 collapses mantissa to 0 bits; r in [0.5,1.0]
     for ceil means x_new_f <= x_fp8.
  3. Cross-block accumulation error is independent and partially averages.

Test plan:
  - For each of 3 representative contest workloads (small/mid/large T),
    transcode activations + weights, run our current CUTLASS Blockwise kernel
    using the new (fp8', UE8M0-as-fp32) as INPUTS, compare to the FP32-scale
    reference on the same kernel. This measures NUMERICAL loss independently
    of hardware speed (we are still on the FP32 mainloop because we cannot
    trivially switch CUTLASS schedules at runtime).
  - If this passes tolerance, we have green-lit MXF8: the only remaining
    question is whether CUTLASS's Mxf8f6f4 schedule gives the expected 1.5-2x
    mainloop speedup.

  - PHASE 2 would actually swap the mainloop. Out of scope for this probe.
"""
import math
import modal
app = modal.App("mxf8-transcode-probe")
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
def run(uuids: str = "e05c6c03,1a4c6ba1,5e8dc11c", variant: str = "round,ceil") -> str:
    import os
    import sys
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

    def q_round_pow2(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-30)
        exp = torch.round(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def q_ceil_pow2(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-30)
        exp = torch.ceil(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def reencode_payload(
        payload_fp8: torch.Tensor,  # float8_e4m3fn, K axis LAST
        scale_fp32: torch.Tensor,   # fp32, per-128 along the K axis
        mode: str,
    ):
        """Re-encode `(payload_fp8, scale_fp32)` into `(payload'_fp8, scale_hw)`
        such that `payload'_fp8 * scale_hw  ~=  payload_fp8 * scale_fp32`.

        Shapes supported:
          - hidden_states:    payload=[T, K],        scale=[K/128, T]   (contest transposed)
          - hidden_states:    payload=[T, K],        scale=[T, K/128]
          - gemm1/2_weights:  payload=[E, N, K],     scale=[E, N/128, K/128]
          - gemm1/2_weights:  payload=[E, N, K],     scale=[E, N, K/128]  (per-row act scale)
        """
        q_fn = q_round_pow2 if mode == "round" else q_ceil_pow2
        scale_hw = q_fn(scale_fp32)
        residual = scale_fp32 / scale_hw  # in [0.5,1.0] for ceil, [0.707,1.414] for round

        p_fp32 = payload_fp8.to(torch.float32)
        K = p_fp32.shape[-1]
        Kb = K // 128

        # Align residual's K axis (last) with payload's K axis, expanding 128x.
        # Permute residual so its LAST dim is the blocked-K dim (= Kb).
        if residual.dim() == 2 and residual.shape[0] == Kb and residual.shape[1] == p_fp32.shape[0]:
            # hidden_states_scale = [K/128, T] -> transpose to [T, K/128]
            residual_tk = residual.transpose(0, 1).contiguous()
            scale_hw_tk = scale_hw.transpose(0, 1).contiguous()
        elif residual.dim() == 2 and residual.shape[0] == p_fp32.shape[0] and residual.shape[1] == Kb:
            residual_tk = residual
            scale_hw_tk = scale_hw
        elif residual.dim() == 3 and residual.shape[-1] == Kb:
            # weight scale: [E, N or N/128, K/128]
            residual_tk = residual
            scale_hw_tk = scale_hw
        else:
            raise RuntimeError(f"unsupported scale shape {tuple(residual.shape)} for payload {tuple(p_fp32.shape)} (Kb={Kb})")

        # Broadcast residual along the per-element K axis.
        expanded = residual_tk.unsqueeze(-1).expand(*residual_tk.shape, 128)
        expanded = expanded.reshape(*residual_tk.shape[:-1], residual_tk.shape[-1] * 128)

        # If scale is per-BLOCK-of-N (weight case), also broadcast across N.
        # For weights: payload=[E, N, K], residual=[E, N/128, K/128] -> need [E, N, K].
        if expanded.dim() == p_fp32.dim() and expanded.shape != p_fp32.shape:
            # Broadcast each non-K dim similarly if it's /128.
            for axis in range(expanded.dim() - 1):
                if expanded.shape[axis] * 128 == p_fp32.shape[axis]:
                    expanded = expanded.repeat_interleave(128, dim=axis)
                elif expanded.shape[axis] == p_fp32.shape[axis]:
                    pass
                else:
                    raise RuntimeError(
                        f"axis {axis}: expanded {expanded.shape[axis]} vs payload {p_fp32.shape[axis]}"
                    )
        if expanded.shape != p_fp32.shape:
            raise RuntimeError(
                f"expanded residual {tuple(expanded.shape)} != payload {tuple(p_fp32.shape)}"
            )

        p_new = (p_fp32 * expanded).clamp(min=-448.0, max=448.0)
        p_new_fp8 = p_new.to(torch.float8_e4m3fn)
        # Return scale_hw in the ORIGINAL layout so downstream code is happy.
        return p_new_fp8, scale_hw

    def run_one(u: str, mode: str):
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w
                break
        if wobj is None:
            return f"{u}: NOT FOUND"

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}

        # inputs order (from flashinfer-bench gen_inputs for this definition):
        # 0: routing_logits   1: routing_bias
        # 2: hidden_states (fp8)      3: hidden_states_scale (fp32 per-128)
        # 4: gemm1_weights (fp8)      5: gemm1_weights_scale (fp32 per-128)
        # 6: gemm2_weights (fp8)      7: gemm2_weights_scale (fp32 per-128)
        # 8: local_expert_offset      9: routed_scaling_factor
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = int(inputs[0].shape[0])

        # Reference run with original FP32 scales.
        out_ref = K.custom_kernel(*inputs).float()

        # Save originals.
        orig_hs     = inputs[2].clone()
        orig_hs_sc  = inputs[3].clone()
        orig_g1     = inputs[4].clone()
        orig_g1_sc  = inputs[5].clone()
        orig_g2     = inputs[6].clone()
        orig_g2_sc  = inputs[7].clone()

        try:
            new_hs,    new_hs_sc = reencode_payload(orig_hs,    orig_hs_sc, mode)
            new_g1,    new_g1_sc = reencode_payload(orig_g1,    orig_g1_sc, mode)
            new_g2,    new_g2_sc = reencode_payload(orig_g2,    orig_g2_sc, mode)
            inputs[2].copy_(new_hs);    inputs[3].copy_(new_hs_sc)
            inputs[4].copy_(new_g1);    inputs[5].copy_(new_g1_sc)
            inputs[6].copy_(new_g2);    inputs[7].copy_(new_g2_sc)

            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            match = (diff <= tol).float().mean().item()
            max_abs = diff.max().item()
            mean_rel = (diff / (out_ref.abs() + 1e-6)).median().item()
            status = "PASS" if match >= 0.9 else "FAIL"
            return (f"{u[:8]} T={T:>5} mode={mode:<5} "
                    f"match={match*100:>5.2f}% max_abs={max_abs:>9.3f} "
                    f"mean_rel={mean_rel:>6.4f} [{status}]")
        finally:
            inputs[2].copy_(orig_hs);     inputs[3].copy_(orig_hs_sc)
            inputs[4].copy_(orig_g1);     inputs[5].copy_(orig_g1_sc)
            inputs[6].copy_(orig_g2);     inputs[7].copy_(orig_g2_sc)

    results = []
    results.append("MXF8 transcode probe: re-encode (fp8 payload, fp32 scale) -> (fp8', pow2 scale)")
    results.append("via ceil/round UE8M0, using residual in activation payload.")
    results.append("-" * 90)
    for u in uuids.split(","):
        u = u.strip()
        for mode in variant.split(","):
            mode = mode.strip()
            try:
                results.append(run_one(u, mode))
            except Exception as e:
                results.append(f"{u[:8]} mode={mode}: ERROR {type(e).__name__}: {e}")
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "b8f4f012,e05c6c03,6230e838,1a4c6ba1,a7c2bcfd,5e8dc11c,58a34f27",
         variant: str = "round,ceil"):
    print(run.remote(uuids=uuids, variant=variant))
