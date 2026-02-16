"""
Fused MoE Triton Kernel - Evolved by KernelEvolve.

FlashInfer entry: kernel.py::kernel
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Constants (DeepSeek-V3/R1 geometry)
H = 7168
I = 2048
E_LOCAL = 32
E_GLOBAL = 256
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = E_GLOBAL // N_GROUP

# =========================
# Triton Kernels
# =========================

@triton.jit
def grouped_gemm1_swiglu_kernel(
    A_ptr, A_scale_ptr, token_sorted_ptr, expert_offsets_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    H, K,
    stride_am, stride_ak, stride_asm, stride_ask,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    A_ptr = tl.multiple_of(A_ptr, 16)
    A_scale_ptr = tl.multiple_of(A_scale_ptr, 16)
    W_ptr = tl.multiple_of(W_ptr, 16)
    W_scale_ptr = tl.multiple_of(W_scale_ptr, 16)
    Out_ptr = tl.multiple_of(Out_ptr, 16)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    tok_start = tl.load(expert_offsets_ptr + expert_id)
    tok_end = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens = tok_end - tok_start
    if num_tokens <= 0:
        return

    num_pid_m = tl.cdiv(num_tokens, BLOCK_M)
    if pid_m >= num_pid_m:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    m_mask = offs_m < num_tokens
    n_mask = offs_n < H

    row_ids = tok_start + offs_m
    token_idx = tl.load(token_sorted_ptr + row_ids, mask=m_mask, other=0)

    a_scale_base = A_scale_ptr + token_idx[:, None] * stride_asm

    ws_ptr_gate = W_scale_ptr + expert_id * stride_wse + (offs_n[None, :] // 128) * stride_wsn
    ws_ptr_up   = W_scale_ptr + expert_id * stride_wse + ((offs_n[None, :] + H) // 128) * stride_wsn

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a_ptr_k = A_ptr + token_idx[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptr_k, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptr_gate = W_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk
        b_gate = tl.load(w_ptr_gate, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        w_ptr_up = W_ptr + expert_id * stride_we + (offs_n[None, :] + H) * stride_wn + (k + offs_k[:, None]) * stride_wk
        b_up = tl.load(w_ptr_up, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        p_gate = tl.dot(a, b_gate)
        p_up   = tl.dot(a, b_up)

        k_blk = k // 128
        sa = tl.load(a_scale_base + k_blk * stride_ask, mask=m_mask[:, None], other=1.0)
        sw_gate = tl.load(ws_ptr_gate + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)
        sw_up   = tl.load(ws_ptr_up   + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)

        acc_gate += p_gate * sa * sw_gate
        acc_up   += p_up   * sa * sw_up

    out = acc_gate * (acc_up * tl.sigmoid(acc_up))

    out_ptr_base = Out_ptr + row_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr_base, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def grouped_gemm2_atomic_kernel(
    C_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    token_sorted_ptr, expert_offsets_ptr,
    Weight_ptr,
    N, K,
    stride_cm, stride_ck,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    C_ptr = tl.multiple_of(C_ptr, 16)
    W_ptr = tl.multiple_of(W_ptr, 16)
    W_scale_ptr = tl.multiple_of(W_scale_ptr, 16)
    Out_ptr = tl.multiple_of(Out_ptr, 16)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    tok_start = tl.load(expert_offsets_ptr + expert_id)
    tok_end = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens = tok_end - tok_start
    if num_tokens <= 0:
        return

    num_pid_m = tl.cdiv(num_tokens, BLOCK_M)
    if pid_m >= num_pid_m:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    m_mask = offs_m < num_tokens
    n_mask = offs_n < N

    row_ids = tok_start + offs_m

    token_idx = tl.load(token_sorted_ptr + row_ids, mask=m_mask, other=0)
    gating_w = tl.load(Weight_ptr + row_ids, mask=m_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_scale_base = W_scale_ptr + expert_id * stride_wse + (offs_n[None, :] // 128) * stride_wsn

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        c = tl.load(
            C_ptr + row_ids[:, None] * stride_cm + (k + offs_k[None, :]) * stride_ck,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        w_ptr_k = W_ptr + expert_id * stride_we + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk
        w_fp8 = tl.load(w_ptr_k, mask=n_mask[None, :] & k_mask[:, None], other=0.0)

        k_blk = k // 128
        sw = tl.load(w_scale_base + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)

        w_dequant = w_fp8.to(tl.float32) * sw
        acc += tl.dot(c, w_dequant)

    acc = acc * gating_w[:, None]

    out_ptrs = Out_ptr + token_idx[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# =========================
# Routing Utilities
# =========================

@torch.no_grad()
def route_topk(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    T = routing_logits.shape[0]
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    s_wb_grouped = s_with_bias.view(T, N_GROUP, GROUP_SIZE)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, GROUP_SIZE).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor
    return weights, topk_idx


@torch.no_grad()
def build_local_dispatch(
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_start: int,
):
    T, K = topk_idx.shape
    device = topk_idx.device

    token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, K)
    assign_w = torch.gather(weights, 1, topk_idx)

    local_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_LOCAL)
    if not local_mask.any():
        return None, None, None

    local_tokens = token_ids[local_mask]
    local_experts = (topk_idx[local_mask] - local_start).to(torch.int64)
    local_weights = assign_w[local_mask]

    sort_keys = local_experts * T + local_tokens
    order = torch.argsort(sort_keys)
    token_sorted = local_tokens[order].contiguous().to(torch.int32)
    expert_sorted = local_experts[order]
    weight_sorted = local_weights[order].contiguous()

    counts = torch.bincount(expert_sorted, minlength=E_LOCAL)
    offsets = torch.zeros(E_LOCAL + 1, device=device, dtype=torch.int32)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    return token_sorted, weight_sorted, offsets


# =========================
# Main Entry
# =========================

@torch.no_grad()
def custom_kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor = None,
) -> torch.Tensor:
    torch.set_float32_matmul_precision('high')

    T = routing_logits.shape[0]
    device = hidden_states.device

    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])

    weights, topk_idx = route_topk(routing_logits, routing_bias, routed_scaling_factor)
    A_scale = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()

    if output is None:
        out_fp32 = torch.zeros((T, N2), dtype=torch.float32, device=device)
        out_target = None
    else:
        out_target = output
        if output.dtype != torch.float32:
            out_fp32 = torch.zeros((T, N2), dtype=torch.float32, device=device)
        else:
            out_fp32 = output
            out_fp32.zero_()

    local_start = int(local_expert_offset)
    token_sorted, weight_sorted, offsets = build_local_dispatch(topk_idx, weights, local_start)
    if token_sorted is None:
        if out_target is not None and out_target is not out_fp32:
            out_target.zero_()
            return out_target
        return out_fp32.to(torch.bfloat16)

    num_assignments = token_sorted.shape[0]

    BLOCK_M_1 = 64
    BLOCK_N_1 = 128
    BLOCK_K_1 = 128

    BLOCK_M_2 = 64
    BLOCK_N_2 = 256
    BLOCK_K_2 = 64

    H_gemm1 = N1 // 2

    C_act = torch.empty((num_assignments, H_gemm1), device=device, dtype=torch.float32)

    grid_1 = (triton.cdiv(T, BLOCK_M_1), triton.cdiv(H_gemm1, BLOCK_N_1), E_LOCAL)
    grouped_gemm1_swiglu_kernel[grid_1](
        hidden_states, A_scale, token_sorted, offsets,
        gemm1_weights, gemm1_weights_scale,
        C_act,
        H_gemm1, K1,
        hidden_states.stride(0), hidden_states.stride(1), A_scale.stride(0), A_scale.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        C_act.stride(0), C_act.stride(1),
        BLOCK_M=BLOCK_M_1, BLOCK_N=BLOCK_N_1, BLOCK_K=BLOCK_K_1,
        num_warps=8, num_stages=4
    )

    grid_2 = (triton.cdiv(T, BLOCK_M_2), triton.cdiv(N2, BLOCK_N_2), E_LOCAL)
    grouped_gemm2_atomic_kernel[grid_2](
        C_act,
        gemm2_weights, gemm2_weights_scale,
        out_fp32,
        token_sorted, offsets,
        weight_sorted,
        N2, K2,
        C_act.stride(0), C_act.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        out_fp32.stride(0), out_fp32.stride(1),
        BLOCK_M=BLOCK_M_2, BLOCK_N=BLOCK_N_2, BLOCK_K=BLOCK_K_2,
        num_warps=8, num_stages=4
    )

    if out_target is not None:
        if out_target is not out_fp32:
            out_target.copy_(out_fp32.to(out_target.dtype))
        return out_target
    return out_fp32.to(torch.bfloat16)

# FlashInfer entry point
kernel = custom_kernel
