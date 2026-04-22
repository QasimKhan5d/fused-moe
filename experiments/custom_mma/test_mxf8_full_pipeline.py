"""End-to-end contest-workload test with MxF8 GEMM in place of FP32-blockwise.

Runs the full MoE pipeline (route/dispatch/gather/GEMM1/swiglu/GEMM2/scatter)
but swaps the two CUTLASS grouped GEMMs to use our new MxF8 path, with
transcoded activations and pre-transcoded weights. Compares output against
the stock FP32-blockwise pipeline (via K.custom_kernel).
"""
import modal
app = modal.App("mxf8-full-pipeline")
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

    TOP_K = 8

    def mxf8_moe_forward(
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_w_tr, gemm1_w_sc, gemm2_w_tr, gemm2_w_sc,
        local_expert_offset, routed_scaling_factor,
    ):
        """Manual MoE pipeline using MxF8 GEMMs for both GEMM1 and GEMM2."""
        T = int(routing_logits.shape[0])
        device = hidden_states.device
        ne = int(gemm1_w_tr.shape[0])
        N1 = int(gemm1_w_tr.shape[1])
        K1 = int(gemm1_w_tr.shape[2])
        N2 = int(gemm2_w_tr.shape[1])
        K2 = int(gemm2_w_tr.shape[2])
        H = N1 // 2
        ls = int(local_expert_offset)
        rsf = float(routed_scaling_factor)
        total_tokens = T * TOP_K

        # 1) Routing
        topk_idx = torch.empty(T, TOP_K, device=device, dtype=torch.int32)
        assign_w = torch.empty(T, TOP_K, device=device, dtype=torch.float32)
        ext.fused_route_topk(
            routing_logits if routing_logits.dtype == torch.bfloat16 else routing_logits.to(torch.bfloat16),
            routing_bias if routing_bias.dtype == torch.bfloat16 else routing_bias.to(torch.bfloat16),
            topk_idx, assign_w, rsf,
        )

        # 2) Dispatch (3-pass count/scan/place)
        counts = torch.empty(ne, device=device, dtype=torch.int32)
        offsets = torch.empty(ne + 1, device=device, dtype=torch.int32)
        sorted_tids = torch.empty(total_tokens, device=device, dtype=torch.int32)
        sorted_weights = torch.empty(total_tokens, device=device, dtype=torch.float32)
        problem_sizes_1 = torch.empty(ne, 3, device=device, dtype=torch.int32)
        problem_sizes_2 = torch.empty(ne, 3, device=device, dtype=torch.int32)
        problem_sizes_1[:, 1] = N1; problem_sizes_1[:, 2] = K1
        problem_sizes_2[:, 1] = N2; problem_sizes_2[:, 2] = K2
        ext.fused_dispatch(
            topk_idx.contiguous(), assign_w.contiguous(),
            ls, ne, counts, sorted_tids, sorted_weights, offsets,
            problem_sizes_1, problem_sizes_2,
        )
        total_valid = int(offsets[ne].item())
        sorted_tids = sorted_tids[:total_valid]
        sorted_weights = sorted_weights[:total_valid]
        expert_offsets = offsets[:ne]

        # 3) Gather hidden_states + their scales (signed fp32).
        packed_acts = torch.empty(total_valid, K1, device=device, dtype=torch.float8_e4m3fn)
        packed_act_scales = torch.empty(total_valid, K1 // 128, device=device, dtype=torch.float32)
        ext.fused_gather_hidden_scales(
            hidden_states, hidden_states_scale, sorted_tids,
            packed_acts, packed_act_scales,
        )

        # 4) Transcode gathered activations (sign-flip + residual absorb).
        packed_act_scales_ue8m0 = torch.empty_like(packed_act_scales)
        ext.mxf8_transcode_activations(packed_acts, packed_act_scales, packed_act_scales_ue8m0)

        # 5) MxF8 GEMM1: output -> gemm1_out (bf16)
        gemm1_out = torch.zeros(total_valid, N1, device=device, dtype=torch.bfloat16)

        # Workspace for MxF8 grouped GEMM.
        stride_sz = ext.get_mxf8_sizes_stride()
        sfa_sz    = ext.get_mxf8_sizes_layout_sfa()
        sfb_sz    = ext.get_mxf8_sizes_layout_sfb()
        a_ptrs   = torch.empty(ne, device=device, dtype=torch.int64)
        b_ptrs   = torch.empty(ne, device=device, dtype=torch.int64)
        out_ptrs = torch.empty(ne, device=device, dtype=torch.int64)
        sfa_ptrs = torch.empty(ne, device=device, dtype=torch.int64)
        sfb_ptrs = torch.empty(ne, device=device, dtype=torch.int64)
        stride_a = torch.empty(ne * stride_sz, device=device, dtype=torch.uint8)
        stride_b = torch.empty(ne * stride_sz, device=device, dtype=torch.uint8)
        stride_c = torch.empty(ne * stride_sz, device=device, dtype=torch.uint8)
        layout_sfa = torch.empty(ne * sfa_sz, device=device, dtype=torch.uint8)
        layout_sfb = torch.empty(ne * sfb_sz, device=device, dtype=torch.uint8)

        # expert_offsets for MxF8 needs to be size E+1 (inclusive). Extract from offsets.
        exp_off_e1 = offsets.contiguous()  # size E+1 already

        sfa_offsets_1, sfa_tot_1 = ext.compute_mxf8_sfa_layout_offsets_host(problem_sizes_1)
        sfb_offsets_1, sfb_tot_1 = ext.compute_mxf8_sfb_layout_offsets_host(problem_sizes_1)
        sfa_byte_off_1 = torch.tensor(sfa_offsets_1, device=device, dtype=torch.int32)
        sfb_byte_off_1 = torch.tensor(sfb_offsets_1, device=device, dtype=torch.int32)
        sfa_buf_1 = torch.empty(sfa_tot_1, device=device, dtype=torch.uint8)
        sfb_buf_1 = torch.empty(sfb_tot_1, device=device, dtype=torch.uint8)
        workspace = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)

        ext.moe_mxf8_grouped_mm(
            gemm1_out, packed_acts, gemm1_w_tr,
            packed_act_scales_ue8m0, gemm1_w_sc,
            exp_off_e1, problem_sizes_1,
            a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
            stride_a, stride_b, stride_c,
            layout_sfa, layout_sfb,
            sfa_buf_1, sfb_buf_1,
            sfa_byte_off_1, sfb_byte_off_1,
            workspace,
        )

        # 6) SwiGLU + per-row FP8 requant (weighted fold).
        act_q = torch.empty(total_valid, H, device=device, dtype=torch.float8_e4m3fn)
        row_scales = torch.empty(total_valid, device=device, dtype=torch.float32)
        act_scale_for_gemm2 = torch.empty(total_valid, K2 // 128, device=device, dtype=torch.float32)
        ext.swiglu_fp8_requant_weighted(
            gemm1_out, sorted_weights, act_q, row_scales, act_scale_for_gemm2)

        # 7) Transcode act_q + act_scale_for_gemm2 (positive scales, no sign).
        act_scale_ue8m0 = torch.empty_like(act_scale_for_gemm2)
        ext.mxf8_transcode_activations(act_q, act_scale_for_gemm2, act_scale_ue8m0)

        # 8) MxF8 GEMM2.
        gemm2_out = torch.zeros(total_valid, N2, device=device, dtype=torch.bfloat16)
        sfa_offsets_2, sfa_tot_2 = ext.compute_mxf8_sfa_layout_offsets_host(problem_sizes_2)
        sfb_offsets_2, sfb_tot_2 = ext.compute_mxf8_sfb_layout_offsets_host(problem_sizes_2)
        sfa_byte_off_2 = torch.tensor(sfa_offsets_2, device=device, dtype=torch.int32)
        sfb_byte_off_2 = torch.tensor(sfb_offsets_2, device=device, dtype=torch.int32)
        sfa_buf_2 = torch.empty(sfa_tot_2, device=device, dtype=torch.uint8)
        sfb_buf_2 = torch.empty(sfb_tot_2, device=device, dtype=torch.uint8)

        ext.moe_mxf8_grouped_mm(
            gemm2_out, act_q, gemm2_w_tr,
            act_scale_ue8m0, gemm2_w_sc,
            exp_off_e1, problem_sizes_2,
            a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
            stride_a, stride_b, stride_c,
            layout_sfa, layout_sfb,
            sfa_buf_2, sfb_buf_2,
            sfa_byte_off_2, sfb_byte_off_2,
            workspace,
        )

        # 9) Reduce-scatter (unweighted since weight is baked into gemm2_out via v17).
        token_counts = torch.zeros(T, device=device, dtype=torch.int32)
        token_offsets = torch.empty(T + 1, device=device, dtype=torch.int32)
        token_perm = torch.empty(total_valid, device=device, dtype=torch.int32)
        out_bf16 = torch.zeros(T, N2, device=device, dtype=torch.bfloat16)
        ext.reduce_scatter_unweighted(
            gemm2_out, sorted_tids, out_bf16,
            token_counts, token_offsets, token_perm, T)
        return out_bf16

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

        # Reference: stock kernel.
        out_ref = K.custom_kernel(*inputs).float()

        # Pre-transcode weights (one-time per workload).
        g1_w_tr = inputs[4].clone()
        g1_w_sc_ue8m0 = torch.empty_like(inputs[5])
        ext.mxf8_transcode_weights_impl(g1_w_tr, inputs[5], g1_w_sc_ue8m0)

        g2_w_tr = inputs[6].clone()
        g2_w_sc_ue8m0 = torch.empty_like(inputs[7])
        ext.mxf8_transcode_weights_impl(g2_w_tr, inputs[7], g2_w_sc_ue8m0)

        # Run MxF8 pipeline.
        try:
            out_mx = mxf8_moe_forward(
                inputs[0], inputs[1], inputs[2], inputs[3],
                g1_w_tr, g1_w_sc_ue8m0, g2_w_tr, g2_w_sc_ue8m0,
                inputs[8], inputs[9],
            ).float()
        except Exception as e:
            return f"{u[:8]} T={T}: MxF8 pipeline FAILED: {type(e).__name__}: {str(e)[:200]}"

        diff = (out_mx - out_ref).abs()
        tol = 1.0 + 0.3 * out_ref.abs()
        match = (diff <= tol).float().mean().item()
        max_abs = diff.max().item()
        mean_rel = (diff / (out_ref.abs() + 1e-6)).median().item()
        status = "PASS" if match >= 0.9 else "FAIL"
        return f"{u[:8]} T={T}  match={match*100:.2f}%  max_abs={max_abs:.2f}  median_rel={mean_rel:.4f}  [{status}]"

    results = []
    for u in uuids.split(","):
        results.append(run_one(u.strip()))
    return "\n".join(results)


@app.local_entrypoint()
def main(uuids: str = "5e8dc11c,58a34f27,1a4c6ba1,6230e838"):
    print(run.remote(uuids=uuids))
