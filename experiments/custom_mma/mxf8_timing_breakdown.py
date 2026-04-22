"""Profile which step in the MxF8 pipeline dominates latency.

Breaks down the total MxF8 GEMM1 path into phases:
  1. mxf8_transcode_activations
  2. compute_mxf8_sf_offsets_device
  3. moe_mxf8_grouped_mm (which itself includes: get_ptrs + pack_sfa + pack_sfb + CUTLASS GEMM)

We time each with CUDA events and compare against plain FP32-blockwise.
"""
import modal

app = modal.App("mxf8-timing")
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
def run(uuid: str = "5e8dc11c") -> str:
    import os, sys
    from pathlib import Path
    import torch

    os.environ["USE_MXF8"] = "1"
    sys.path.insert(0, "/root/solution")
    os.chdir("/root/solution")
    import kernel as K
    ext = K._get_ext()
    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
    definition = trace_set.definitions["moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"]
    wobj = None
    for wl in trace_set.workloads.get(definition.name, []):
        w = getattr(wl, "workload", wl)
        if getattr(w, "uuid", "").startswith(uuid):
            wobj = w; break
    loaded_st = load_safetensors(definition, wobj, Path("/mnt/mlsys26-contest")) if any(
        d.type == "safetensors" for d in getattr(wobj, "inputs", {}).values()) else {}
    inputs = gen_inputs(definition, wobj, device="cuda", safe_tensors=loaded_st)
    T = int(inputs[0].shape[0])

    # Warmup + first call to populate workspace + transcode weights.
    for _ in range(3):
        _ = K.custom_kernel(*inputs)
    torch.cuda.synchronize()

    # Dig into pipeline state to re-run just sub-steps.
    bufs = K._get_workspace(inputs[2].device, int(inputs[4].shape[0]), T,
                             int(inputs[4].shape[1]), int(inputs[4].shape[2]),
                             int(inputs[6].shape[1]), int(inputs[6].shape[2]))
    # Manually replay routing/dispatch/gather to set buffers up.
    routing_logits, routing_bias, hidden_states, hs_scale = inputs[0], inputs[1], inputs[2], inputs[3]
    gemm1_w, gemm1_ws = inputs[4], inputs[5]
    gemm2_w, gemm2_ws = inputs[6], inputs[7]
    ls = int(inputs[8]); rsf = float(inputs[9])
    ne = int(gemm1_w.shape[0]); N1 = int(gemm1_w.shape[1]); K1 = int(gemm1_w.shape[2])

    topk_idx, assign_w = K._route(routing_logits, routing_bias, rsf, T, ls, ne)
    counts, sorted_tids, sorted_weights = K._dispatch_dynamic(
        topk_idx, assign_w, T, ls, ne, bufs)
    total_valid = sorted_tids.shape[0]

    packed_acts = bufs["packed_acts"][:total_valid]
    packed_act_scales = bufs["packed_act_scales"][:total_valid]
    ext.fused_gather_hidden_scales(
        hidden_states, hs_scale, sorted_tids,
        packed_acts, packed_act_scales)

    mxf8_act_scales_ue8m0 = bufs["mxf8_act_scales_ue8m0"][:total_valid]
    gemm1_out = bufs["gemm1_out"][:total_valid]

    def time_block(fn, name, iters=20):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        return f"{name:40s} {ms*1000:8.1f} us"

    lines = [f"UUID={uuid} T={T}  total_valid={total_valid}"]

    lines.append(time_block(
        lambda: ext.mxf8_transcode_activations(packed_acts, packed_act_scales, mxf8_act_scales_ue8m0),
        "mxf8_transcode_activations"))
    lines.append(time_block(
        lambda: ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_1"],
            bufs["mxf8_sfa_byte_offsets_1"],
            bufs["mxf8_sfb_byte_offsets_1"]),
        "compute_mxf8_sf_offsets_device"))

    # Set up GEMM2 buffers: fake act_q/scale/gemm2_out to isolate GEMM2.
    act_q = bufs["act_q"][:total_valid]
    act_scale_for_gemm2 = bufs["act_scale_for_gemm2"][:total_valid]
    gemm2_out = bufs["gemm2_out"][:total_valid]
    # Ensure weights transcoded for GEMM2.
    mxf8_gemm2_act_scales_ue8m0 = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]

    def run_mxf8_gemm1():
        ext.moe_mxf8_grouped_mm(
            gemm1_out,
            packed_acts, bufs["mxf8_gemm1_w_tr"],
            mxf8_act_scales_ue8m0, bufs["mxf8_gemm1_w_sc_ue8m0"],
            bufs["offsets_buf"], bufs["problem_sizes_1"],
            bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
            bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
            bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
            bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"],
            bufs["workspace"],
        )
    lines.append(time_block(run_mxf8_gemm1, "MxF8 GEMM1 (full)"))

    def run_mxf8_gemm2():
        ext.moe_mxf8_grouped_mm(
            gemm2_out,
            act_q, bufs["mxf8_gemm2_w_tr"],
            mxf8_gemm2_act_scales_ue8m0, bufs["mxf8_gemm2_w_sc_ue8m0"],
            bufs["offsets_buf"], bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
            bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"],
            bufs["workspace"],
        )
    lines.append(time_block(run_mxf8_gemm2, "MxF8 GEMM2 (full)"))

    def run_blockwise_gemm1():
        ext.moe_blockwise_grouped_mm_v2(
            gemm1_out,
            packed_acts, gemm1_w, packed_act_scales, gemm1_ws,
            bufs["expert_offsets"], bufs["problem_sizes_1"],
            bufs["problem_sizes_transpose"],
            bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
            bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
            bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
            bufs["layout_sfa"], bufs["layout_sfb"],
            bufs["workspace"],
        )
    lines.append(time_block(run_blockwise_gemm1, "blockwise GEMM1 (baseline)"))

    def run_blockwise_gemm2():
        ext.moe_blockwise_grouped_mm_v2(
            gemm2_out,
            act_q, gemm2_w, act_scale_for_gemm2, gemm2_ws,
            bufs["expert_offsets"], bufs["problem_sizes_2"],
            bufs["problem_sizes_transpose"],
            bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
            bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
            bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
            bufs["layout_sfa"], bufs["layout_sfb"],
            bufs["workspace"],
        )
    lines.append(time_block(run_blockwise_gemm2, "blockwise GEMM2 (baseline)"))

    return "\n".join(lines)


@app.local_entrypoint()
def main(uuid: str = "5e8dc11c"):
    print(run.remote(uuid=uuid))
