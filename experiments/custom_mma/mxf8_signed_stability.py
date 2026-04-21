"""H1 + H2: confirm V3 result is stable; study T=1 failure mode.

Run V3 three times (fresh inputs each time) to see if match ratio is stable.
For T=1, test several alternative transcode variants:
  - A) Default V3: sign-flip + ue8m0 + absorb residual into payload (ceil rounding of ue8m0 scale).
  - B) Same but use ROUND-to-nearest pow2 for ue8m0 (residual r in [0.707, 1.414]).
  - C) Same but use TRUNCATE (floor) pow2 (residual in [1, 2)).
  - D) Variant with larger-precision residual (absorb residual + a small per-row fp32 scalar correction).
"""
import modal
app = modal.App("mxf8-signed-stability")
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
    K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    def ue8m0_ceil(abs_val):
        abs_val = abs_val.clamp_min(1e-30)
        exp = torch.ceil(torch.log2(abs_val)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def ue8m0_round(abs_val):
        abs_val = abs_val.clamp_min(1e-30)
        exp = torch.round(torch.log2(abs_val)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def ue8m0_floor(abs_val):
        abs_val = abs_val.clamp_min(1e-30)
        exp = torch.floor(torch.log2(abs_val)).clamp(-127, 127)
        return torch.pow(2.0, exp)

    def transcode(payload_fp8, scale_fp32, mode):
        """sign_flip + ue8m0 + payload residual. mode ∈ {'ceil','round','floor'}."""
        sign = torch.sign(scale_fp32)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        abs_val = scale_fp32.abs()
        if mode == "ceil":
            ue8m0 = ue8m0_ceil(abs_val)
        elif mode == "round":
            ue8m0 = ue8m0_round(abs_val)
        else:
            ue8m0 = ue8m0_floor(abs_val)
        r = abs_val.clamp_min(1e-30) / ue8m0

        p_fp32 = payload_fp8.to(torch.float32)
        K_ = p_fp32.shape[-1]
        Kb = K_ // 128

        # Align scale shape with payload.
        if sign.dim() == 2 and sign.shape[0] == Kb and sign.shape[1] == p_fp32.shape[0]:
            sign_tk = sign.transpose(0, 1).contiguous()
            r_tk = r.transpose(0, 1).contiguous()
        elif sign.dim() == 2 and sign.shape[0] == p_fp32.shape[0] and sign.shape[1] == Kb:
            sign_tk = sign
            r_tk = r
        elif sign.dim() == 3 and sign.shape[-1] == Kb:
            sign_tk = sign
            r_tk = r
        else:
            raise RuntimeError(f"scale shape {sign.shape} invalid for payload {p_fp32.shape}")

        def expand128(x):
            exp = x.unsqueeze(-1).expand(*x.shape, 128)
            exp = exp.reshape(*x.shape[:-1], x.shape[-1] * 128)
            for axis in range(exp.dim() - 1):
                if exp.shape[axis] * 128 == p_fp32.shape[axis]:
                    exp = exp.repeat_interleave(128, dim=axis)
            assert exp.shape == p_fp32.shape, (exp.shape, p_fp32.shape)
            return exp

        sign_full = expand128(sign_tk)
        r_full = expand128(r_tk)
        p_new = (p_fp32 * sign_full * r_full).clamp(min=-448.0, max=448.0)
        return p_new.to(torch.float8_e4m3fn), ue8m0

    def compare(inputs, mode):
        out_ref = K.custom_kernel(*inputs).float()
        saved = {idx: (inputs[idx].clone(), inputs[idx + 1].clone()) for idx in (2, 4, 6)}
        try:
            for idx in (2, 4, 6):
                p_new, s_new = transcode(inputs[idx], inputs[idx + 1], mode)
                inputs[idx].copy_(p_new)
                inputs[idx + 1].copy_(s_new)
            out_q = K.custom_kernel(*inputs).float()
            diff = (out_q - out_ref).abs()
            tol = 1.0 + 0.3 * out_ref.abs()
            return (diff <= tol).float().mean().item(), diff.max().item()
        finally:
            for idx, (op, os_) in saved.items():
                inputs[idx].copy_(op)
                inputs[idx + 1].copy_(os_)

    lines = []
    for u in uuids.split(","):
        u = u.strip()
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            lines.append(f"{u}: not found"); continue

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))
        T = inputs[0].shape[0]

        lines.append(f"=== {u[:8]} T={T} ===")
        for mode in ("ceil", "round", "floor"):
            runs = []
            for trial in range(3):
                m, mx = compare(inputs, mode)
                runs.append((m, mx))
            m_min = min(r[0] for r in runs)
            m_max = max(r[0] for r in runs)
            m_mean = sum(r[0] for r in runs) / 3
            status = "PASS" if m_min >= 0.90 else "FAIL"
            lines.append(f"  {mode:<6} match: min={m_min*100:.2f}% mean={m_mean*100:.2f}% max={m_max*100:.2f}% [{status}]")
    return "\n".join(lines)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,b8f4f012,5e8dc11c,58a34f27,1a4c6ba1,6230e838"):
    print(run.remote(uuids=uuids))
