"""Characterize contest FP32 scale distribution.

Key questions:
  1. Are any scales negative? (Should all be |fp32|/448 magnitudes.)
  2. How much does the UE8M0 ceil residual r = s_fp32 / s_hw vary along K
     for a given (m, n) / (n, e)? If std is small, a per-row correction
     could recover precision. If std is ~0.5 (uniform on log2), no single
     correction works.
  3. What's the dynamic range of scales across blocks?
"""
import modal
app = modal.App("scale-distribution")
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
def run(uuids: str = "5e8dc11c") -> str:
    import os, sys
    from pathlib import Path
    import torch
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    def_name = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    definition = trace_set.definitions[def_name]

    def describe(x: torch.Tensor, label: str):
        lines = []
        lines.append(f"{label}: shape={tuple(x.shape)} dtype={x.dtype}")
        lines.append(f"  range=[{x.min().item():.3e}, {x.max().item():.3e}]")
        lines.append(f"  nonpositive count: {(x <= 0).sum().item()} / {x.numel()}")
        x_pos = x.clamp_min(1e-30)
        log_x = torch.log2(x_pos)
        lines.append(f"  log2(x): mean={log_x.mean().item():.2f} std={log_x.std().item():.2f}")
        # residual = x / ceil_pow2(x)
        exp = torch.ceil(log_x).clamp(-127, 127)
        x_hw = torch.pow(2.0, exp)
        r = x_pos / x_hw  # in (0, 1]
        log_r = torch.log2(r.clamp_min(1e-30))
        lines.append(f"  ceil residual r: range=[{r.min().item():.3f}, {r.max().item():.3f}] "
                     f"mean={r.mean().item():.3f} std={r.std().item():.3f}")
        lines.append(f"  log2(r): mean={log_r.mean().item():.3f} std={log_r.std().item():.3f}")
        return lines

    def uniformity_along_K(scale: torch.Tensor, K_axis: int, label: str):
        """Compute std of log2(r) along K axis (uniformity of residual)."""
        lines = []
        x_pos = scale.clamp_min(1e-30)
        log_x = torch.log2(x_pos)
        exp = torch.ceil(log_x).clamp(-127, 127)
        x_hw = torch.pow(2.0, exp)
        r = x_pos / x_hw
        log_r = torch.log2(r.clamp_min(1e-30))
        std_along_K = log_r.std(dim=K_axis)
        lines.append(
            f"{label}: std_log2(r) along K: mean={std_along_K.mean().item():.3f} "
            f"max={std_along_K.max().item():.3f} "
            f"(0.0=perfectly uniform per row/col, ~0.29 = uniform on [0,1])"
        )
        # Also show per-row min and max r (directly)
        r_min_along_K = r.min(dim=K_axis).values
        r_max_along_K = r.max(dim=K_axis).values
        r_range = r_max_along_K / r_min_along_K
        lines.append(
            f"  per-row r_max/r_min: mean={r_range.mean().item():.2f} "
            f"p99={r_range.quantile(0.99).item():.2f} "
            f"max={r_range.max().item():.2f}"
        )
        return lines

    out = []
    for u in uuids.split(","):
        u = u.strip()
        wobj = None
        for wl in trace_set.workloads.get(def_name, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(u):
                wobj = w; break
        if wobj is None:
            out.append(f"{u}: not found"); continue

        loaded_st = load_safetensors(
            definition, wobj, Path("/mnt/mlsys26-contest")
        ) if any(d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = list(gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st))

        out.append(f"=== {u[:8]} T={inputs[0].shape[0]} ===")
        # 2=hidden, 4=gemm1_w, 6=gemm2_w; 3,5,7 = their scales
        out.extend(describe(inputs[3], "hs_scale"))
        out.extend(describe(inputs[5], "gemm1_w_scale"))
        out.extend(describe(inputs[7], "gemm2_w_scale"))

        # Uniformity: hs_scale shape [K/128, T] -> K axis is 0 (since transposed).
        hs = inputs[3]
        if hs.shape[0] * 128 == inputs[2].shape[1]:
            out.extend(uniformity_along_K(hs, K_axis=0, label="hs (K-major=dim0)"))
        else:
            out.extend(uniformity_along_K(hs, K_axis=-1, label="hs"))
        # weights: [E, N/128, K/128] -> K axis -1
        out.extend(uniformity_along_K(inputs[5], K_axis=-1, label="gemm1_w"))
        out.extend(uniformity_along_K(inputs[7], K_axis=-1, label="gemm2_w"))
        out.append("")

    return "\n".join(out)


@app.local_entrypoint()
def main(uuids: str = "e05c6c03,6230e838,1a4c6ba1,5e8dc11c"):
    print(run.remote(uuids=uuids))
