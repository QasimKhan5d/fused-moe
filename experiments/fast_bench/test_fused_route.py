"""Correctness + perf test for the fused_route_topk CUDA kernel."""
import modal

app = modal.App("test-fused-route")
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
def test() -> str:
    import os, sys, torch, time
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")

    import kernel as K
    ext = K._get_ext()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from pathlib import Path

    TRACE_PATH = "/mnt/mlsys26-contest"
    DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
    trace_set = TraceSet.from_path(TRACE_PATH)
    definition = trace_set.definitions[DEF_NAME]

    out_lines = []

    for sel in ["e05c6c03", "81955b1e", "1a4c6ba1", "58a34f27", "5e8dc11c"]:
        wobj = None
        for wl in trace_set.workloads.get(DEF_NAME, []):
            w = getattr(wl, "workload", wl)
            if getattr(w, "uuid", "").startswith(sel):
                wobj = w; break
        if wobj is None:
            out_lines.append(f"{sel}: NOT FOUND")
            continue

        loaded_st = load_safetensors(definition, wobj, Path(TRACE_PATH)) if any(
            d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
        inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
        (routing_logits, routing_bias, _, _, _, _, _, _, _, routed_scaling_factor) = inputs
        T = int(routing_logits.shape[0])
        rsf = float(routed_scaling_factor)

        # Reference: pure PyTorch implementation
        ref_topk, ref_w = K._route(routing_logits, routing_bias, rsf, T, 0, 32)

        # Our fused kernel: takes bf16 logits/bias directly
        logits_bf16 = routing_logits.to(torch.bfloat16) if routing_logits.dtype != torch.bfloat16 else routing_logits
        bias_bf16 = routing_bias.to(torch.bfloat16) if routing_bias.dtype != torch.bfloat16 else routing_bias
        my_topk = torch.empty(T, 8, device="cuda", dtype=torch.int32)
        my_w = torch.empty(T, 8, device="cuda", dtype=torch.float32)
        ext.fused_route_topk(logits_bf16, bias_bf16, my_topk, my_w, rsf)
        torch.cuda.synchronize()

        # Correctness: our topk might pick same experts but in different order
        # (ties). Compare sorted sets per-token.
        ref_sorted = torch.sort(ref_topk.to(torch.int32), dim=1).values
        my_sorted = torch.sort(my_topk, dim=1).values
        sets_match = torch.all(ref_sorted == my_sorted).item()
        n_mismatch_rows = (ref_sorted != my_sorted).any(dim=1).sum().item()

        # Weight sum should be close to rsf (since we normalize). Compare per-token sum.
        ref_sum = ref_w.sum(dim=1)
        my_sum = my_w.sum(dim=1)
        max_abs_diff_sum = (ref_sum - my_sum).abs().max().item()

        # Perf: time both
        N = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            K._route(routing_logits, routing_bias, rsf, T, 0, 32)
        torch.cuda.synchronize()
        ref_ms = (time.time() - t0) * 1000 / N

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            ext.fused_route_topk(logits_bf16, bias_bf16, my_topk, my_w, rsf)
        torch.cuda.synchronize()
        fused_ms = (time.time() - t0) * 1000 / N

        out_lines.append(
            f"{sel[:8]} T={T:>5} sets_match={sets_match} mismatch_rows={n_mismatch_rows:>4}/{T} "
            f"sum_diff={max_abs_diff_sum:.2e} | pytorch={ref_ms:.3f}ms  fused={fused_ms:.3f}ms  "
            f"speedup={ref_ms/fused_ms:.2f}x"
        )

    return "\n".join(out_lines)


@app.local_entrypoint()
def main():
    print(test.remote())
