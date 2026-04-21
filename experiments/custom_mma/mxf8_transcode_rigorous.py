"""Rigorous MXF8 transcode diagnostic.

Previous probe tried transcoding ALL THREE tensors (hidden, gemm1_w, gemm2_w)
at once with naive round/ceil. Match was 3-73%. Need to:

1. Isolate which tensor is the dominant error source (hidden? gemm1_w? gemm2_w?).
2. Test per-32 reblocking (replicate per-128 scale 4x into per-32 slots).
3. Test "keep residual as post-GEMM correction" variants:
      a) Apply residual to output row as a multiplicative correction (only valid
         if residual is ROW-uniform across the entire K axis, which it isn't).
      b) Absorb residual ONLY into payload of one side (activation OR weight),
         not both. The side NOT transcoded keeps its original FP8 payload and
         its original FP32 scale.
4. Test round-to-nearest-FP8 with saturating clamp vs truncate.
5. For any single-tensor transcode that passes, confirm BOTH sides need it.

Output: a table that for each (workload, tensor-subset, mode) shows match%,
max_abs_err, mean_rel_err, and whether any variant crosses 90% tolerance.
"""
import modal
app = modal.App("mxf8-transcode-rigorous")
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
def run(uuids: str = "e05c6c03,1a4c6ba1,5e8dc11c") -> str:
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

    def q_round_pow2(x):
        x = x.clamp_min(1e-30)
        exp = torch.round(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def q_ceil_pow2(x):
        x = x.clamp_min(1e-30)
        exp = torch.ceil(torch.log2(x)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def broadcast_to_payload(residual, payload, Kb):
        """Broadcast residual scale to match payload's per-element shape."""
        # Figure out residual layout vs payload.
        K = payload.shape[-1]
        if residual.dim() == 2 and residual.shape[0] == Kb and residual.shape[1] == payload.shape[0]:
            # [Kb, T] -> [T, Kb]
            residual_tk = residual.transpose(0, 1).contiguous()
        elif residual.dim() == 2 and residual.shape[0] == payload.shape[0] and residual.shape[1] == Kb:
            residual_tk = residual
        elif residual.dim() == 3 and residual.shape[-1] == Kb:
            residual_tk = residual
        else:
            raise RuntimeError(f"residual shape {residual.shape} not recognized for payload {payload.shape} (Kb={Kb})")

        expanded = residual_tk.unsqueeze(-1).expand(*residual_tk.shape, 128)
        expanded = expanded.reshape(*residual_tk.shape[:-1], residual_tk.shape[-1] * 128)
        if expanded.dim() < payload.dim():
            # weight layout: residual [E, N/128, K/128] -> expand N/128 -> N via repeat_interleave
            pass
        if expanded.shape != payload.shape:
            for axis in range(expanded.dim() - 1):
                if expanded.shape[axis] * 128 == payload.shape[axis]:
                    expanded = expanded.repeat_interleave(128, dim=axis)
        assert expanded.shape == payload.shape, f"expanded {expanded.shape} vs payload {payload.shape}"
        return expanded

    def transcode_payload_only(payload_fp8, scale_fp32, mode, residual_target="payload"):
        """Return (payload_new_fp8, scale_hw) where decoded value is preserved approximately.

        residual_target:
          "payload": absorb residual into payload (our original probe)
          "scale":   don't absorb, just round scale (pure UE8M0, no correction)
          "none":    leave everything as-is (reference)
        """
        q_fn = q_round_pow2 if mode == "round" else q_ceil_pow2
        scale_hw = q_fn(scale_fp32)

        if residual_target == "none":
            return payload_fp8, scale_fp32

        if residual_target == "scale":
            # Pure scale rounding: data unchanged, scale quantized.
            return payload_fp8, scale_hw

        # "payload": absorb residual into payload.
        residual = scale_fp32 / scale_hw  # ratio
        p_fp32 = payload_fp8.to(torch.float32)
        expanded = broadcast_to_payload(residual, p_fp32, p_fp32.shape[-1] // 128)
        p_new = (p_fp32 * expanded).clamp(min=-448.0, max=448.0)
        return p_new.to(torch.float8_e4m3fn), scale_hw

    def compare(inputs, transforms):
        """transforms: dict {tensor_idx: (mode, residual_target)} applied to each tensor.
        Returns (match_ratio, max_abs, mean_rel)."""
        out_ref = K.custom_kernel(*inputs).float()
        saved = {idx: (inputs[idx].clone(), inputs[idx + 1].clone())
                 for idx in transforms.keys()}
        try:
            for idx, (mode, res_target) in transforms.items():
                p_new, s_new = transcode_payload_only(
                    inputs[idx], inputs[idx + 1], mode, res_target)
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

    # Variants:
    # Tensor indices: 2=hidden, 4=gemm1_w, 6=gemm2_w.
    # Name -> transforms dict.
    variants = {
        "baseline":              {},
        "hs_payload_ceil":       {2: ("ceil", "payload")},
        "hs_scale_only_ceil":    {2: ("ceil", "scale")},
        "gemm1_payload_ceil":    {4: ("ceil", "payload")},
        "gemm1_scale_only_ceil": {4: ("ceil", "scale")},
        "gemm2_payload_ceil":    {6: ("ceil", "payload")},
        "gemm2_scale_only_ceil": {6: ("ceil", "scale")},
        "all_payload_ceil":      {2: ("ceil", "payload"), 4: ("ceil", "payload"), 6: ("ceil", "payload")},
        "all_scale_only_ceil":   {2: ("ceil", "scale"),   4: ("ceil", "scale"),   6: ("ceil", "scale")},
        "hs_payload_round":      {2: ("round", "payload")},
        "hs_scale_only_round":   {2: ("round", "scale")},
    }

    results = []
    results.append(f"{'uuid':<10} {'T':>5} {'variant':<24} {'match%':>7} {'max_abs':>10} {'mean_rel':>8}")
    results.append("-" * 90)
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
        T = int(inputs[0].shape[0])

        for name, transforms in variants.items():
            try:
                m, mx, mr = compare(inputs, transforms)
                ok = "PASS" if m >= 0.9 else "fail"
                results.append(
                    f"{u[:8]:<10} {T:>5} {name:<24} {m*100:>6.2f}% {mx:>10.2f} {mr:>7.4f} [{ok}]")
            except Exception as e:
                results.append(f"{u[:8]:<10} {T:>5} {name:<24} ERROR {type(e).__name__}: {str(e)[:40]}")
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,6230e838,1a4c6ba1,5e8dc11c,58a34f27"):
    print(run.remote(uuids=uuids))
