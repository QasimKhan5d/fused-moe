"""Fused MoE - M-loop weight reuse (KSearch-inspired)."""
import torch
import triton
import triton.language as tl

H = 7168
I = 2048
E_LOCAL = 32
E_GLOBAL = 256
BLOCK = 128
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = E_GLOBAL // N_GROUP

# EVOLVE-BLOCK-START
# <NAME>mloop_weight_reuse_v1</NAME>
# EDITABLE-IMPORTS

import torch
import triton
import triton.language as tl
from torch.nn import functional as F


def evolve_precision_overrides():
    torch.set_float32_matmul_precision('high')


@triton.jit
def dispatch_count_kernel(
    TopkIdx_ptr,
    Counts_ptr,
    num_assignments,
    local_start, num_local_experts,
    stride_idx,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_assignments

    expert_global = tl.load(TopkIdx_ptr + offs * stride_idx, mask=mask, other=-1).to(tl.int32)
    is_local = (expert_global >= local_start) & (expert_global < (local_start + num_local_experts))
    expert_local = expert_global - local_start
    tl.atomic_add(Counts_ptr + expert_local, 1, mask=mask & is_local)


@triton.jit
def dispatch_scatter_kernel(
    TopkIdx_ptr,
    Weights_ptr,
    Offsets_ptr,
    CurrentCnts_ptr,
    SortedTokenIds_ptr,
    SortedWeights_ptr,
    num_assignments, K,
    local_start, num_local_experts,
    stride_idx, stride_w,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_assignments

    expert_global = tl.load(TopkIdx_ptr + offs * stride_idx, mask=mask, other=-1).to(tl.int32)
    weight = tl.load(Weights_ptr + offs * stride_w, mask=mask, other=0.0)

    is_local = (expert_global >= local_start) & (expert_global < (local_start + num_local_experts))
    expert_local = expert_global - local_start

    base_offset = tl.load(Offsets_ptr + expert_local, mask=mask & is_local, other=0)
    local_cnt = tl.atomic_add(CurrentCnts_ptr + expert_local, 1, mask=mask & is_local)
    dest_idx = base_offset + local_cnt

    token_id = offs // K
    tl.store(SortedTokenIds_ptr + dest_idx, token_id, mask=mask & is_local)
    tl.store(SortedWeights_ptr + dest_idx, weight, mask=mask & is_local)


@triton.jit
def gemm1_mloop_kernel(
    A_ptr, A_scale_ptr, Idx_ptr, Offsets_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    H, K,
    stride_am, stride_ak, stride_asm, stride_ask,
    stride_we, stride_wn, stride_wk, stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    eid = tl.program_id(1)

    off_start = tl.load(Offsets_ptr + eid)
    off_end = tl.load(Offsets_ptr + eid + 1)
    count = off_end - off_start
    if count == 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < H

    w_ptr_base = W_ptr + eid * stride_we
    ws_ptr_base = W_scale_ptr + eid * stride_wse

    for m_base in range(0, count, BLOCK_M):
        offs_m = m_base + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count

        token_offset = off_start + offs_m
        idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)

        a_ptr_base = A_ptr + idx[:, None] * stride_am
        a_scale_base = A_scale_ptr + idx[:, None] * stride_asm

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K

            a = tl.load(
                a_ptr_base + (k + offs_k[None, :]) * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            k_blk = k // 128
            sa = tl.load(
                a_scale_base + k_blk * stride_ask,
                mask=m_mask[:, None],
                other=1.0,
            ).to(tl.float32)

            w_gate = tl.load(
                w_ptr_base + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
                eviction_policy='evict_last',
            )
            w_up = tl.load(
                w_ptr_base + (offs_n[None, :] + H) * stride_wn + (k + offs_k[:, None]) * stride_wk,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
                eviction_policy='evict_last',
            )

            sw_gate = tl.load(
                ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk,
                mask=n_mask[None, :],
                other=1.0,
            ).to(tl.float32)
            sw_up = tl.load(
                ws_ptr_base + ((offs_n[None, :] + H) // 128) * stride_wsn + k_blk * stride_wsk,
                mask=n_mask[None, :],
                other=1.0,
            ).to(tl.float32)

            p_gate = tl.dot(a, w_gate, allow_tf32=True)
            p_up = tl.dot(a, w_up, allow_tf32=True)

            acc_gate += p_gate * sa * sw_gate
            acc_up += p_up * sa * sw_up

        out = acc_gate * (acc_up * tl.sigmoid(acc_up))
        out_ptr_base = Out_ptr + token_offset[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptr_base, out, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def gemm2_mloop_kernel(
    C_ptr, Offsets_ptr, Weight_ptr, Idx_ptr,
    W_ptr, W_scale_ptr,
    Out_ptr,
    N, K,
    stride_cm, stride_ck,
    stride_we, stride_wn, stride_wk, stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    eid = tl.program_id(1)

    off_start = tl.load(Offsets_ptr + eid)
    off_end = tl.load(Offsets_ptr + eid + 1)
    count = off_end - off_start
    if count == 0:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N

    w_ptr_base = W_ptr + eid * stride_we
    ws_ptr_base = W_scale_ptr + eid * stride_wse

    for m_base in range(0, count, BLOCK_M):
        offs_m = m_base + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count

        token_offset = off_start + offs_m
        gating_w = tl.load(Weight_ptr + token_offset, mask=m_mask, other=0.0)

        c_ptr_base = C_ptr + token_offset[:, None] * stride_cm

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K

            c = tl.load(
                c_ptr_base + (k + offs_k[None, :]) * stride_ck,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w_tile = tl.load(
                w_ptr_base + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk,
                mask=n_mask[None, :] & k_mask[:, None],
                other=0.0,
                eviction_policy='evict_last',
            )

            k_blk = k // 128
            sw = tl.load(
                ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk,
                mask=n_mask[None, :],
                other=1.0,
            ).to(tl.float32)

            w_dequant = w_tile.to(tl.float32) * sw
            acc += tl.dot(c, w_dequant, allow_tf32=True)

        acc = acc * gating_w[:, None]
        orig_idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)
        out_ptrs = Out_ptr + orig_idx[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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

    group_idx_exp = group_idx.unsqueeze(-1).expand(T, TOPK_GROUP, GROUP_SIZE)
    selected = torch.gather(s_wb_grouped, 1, group_idx_exp)
    selected_flat = selected.reshape(T, TOPK_GROUP * GROUP_SIZE)

    _, topk_local = torch.topk(selected_flat, k=TOP_K, dim=1, largest=True, sorted=False)

    group_sel = torch.div(topk_local, GROUP_SIZE, rounding_mode='floor')
    in_group = topk_local - group_sel * GROUP_SIZE

    topk_group = torch.gather(group_idx, 1, group_sel)
    topk_idx = topk_group * GROUP_SIZE + in_group

    topk_s = torch.gather(s, 1, topk_idx)
    return topk_s, topk_idx


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
) -> torch.Tensor:
    T = routing_logits.shape[0]
    device = hidden_states.device

    num_experts = int(gemm1_weights.shape[0])
    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])
    H = N1 // 2

    topk_s, topk_idx = route_topk(routing_logits, routing_bias, routed_scaling_factor)
    TOP_K_val = topk_idx.shape[1]

    weights_sum = topk_s.sum(dim=1, keepdim=True) + 1e-20
    assign_w = (topk_s / weights_sum) * routed_scaling_factor

    num_local_experts = num_experts
    local_start = int(local_expert_offset)

    counts = torch.zeros(num_local_experts, device=device, dtype=torch.int32)
    num_assignments = T * TOP_K_val

    BLOCK_DISPATCH = 1024
    grid_dispatch = (triton.cdiv(num_assignments, BLOCK_DISPATCH),)

    topk_idx = topk_idx.contiguous()
    assign_w = assign_w.contiguous()

    dispatch_count_kernel[grid_dispatch](
        topk_idx, counts,
        num_assignments, local_start, num_local_experts,
        1,
        BLOCK=BLOCK_DISPATCH
    )

    offsets = torch.zeros(num_local_experts + 1, device=device, dtype=torch.int32)
    torch.cumsum(counts, 0, out=offsets[1:])

    sorted_token_ids = torch.empty(num_assignments, device=device, dtype=torch.int32)
    sorted_weights = torch.empty(num_assignments, device=device, dtype=torch.float32)
    current_cnts = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

    dispatch_scatter_kernel[grid_dispatch](
        topk_idx, assign_w,
        offsets, current_cnts,
        sorted_token_ids, sorted_weights,
        num_assignments, TOP_K_val,
        local_start, num_local_experts,
        1, 1,
        BLOCK=BLOCK_DISPATCH
    )

    BLOCK_M1 = 32
    BLOCK_N1 = 64
    BLOCK_K1 = 128
    BLOCK_M2 = 32
    BLOCK_N2 = 128
    BLOCK_K2 = 128

    C_act = torch.empty((num_assignments, H), device=device, dtype=torch.float32)

    grid_1 = (triton.cdiv(H, BLOCK_N1), num_experts)

    gemm1_mloop_kernel[grid_1](
        hidden_states, hidden_states_scale, sorted_token_ids, offsets,
        gemm1_weights, gemm1_weights_scale,
        C_act,
        H, K1,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(1), hidden_states_scale.stride(0),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        C_act.stride(0), C_act.stride(1),
        BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
        num_warps=4, num_stages=3,
    )

    out_fp32 = torch.zeros((T, N2), dtype=torch.float32, device=device)

    grid_2 = (triton.cdiv(N2, BLOCK_N2), num_experts)

    gemm2_mloop_kernel[grid_2](
        C_act, offsets, sorted_weights, sorted_token_ids,
        gemm2_weights, gemm2_weights_scale,
        out_fp32,
        N2, K2,
        C_act.stride(0), C_act.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        out_fp32.stride(0), out_fp32.stride(1),
        BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
        num_warps=4, num_stages=3,
    )

    return out_fp32.to(torch.bfloat16)
# EVOLVE-BLOCK-END

kernel = custom_kernel
run = custom_kernel
