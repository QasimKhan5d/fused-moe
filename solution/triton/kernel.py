"""
Fused MoE Triton Kernel - Evolved by KernelEvolve.

FlashInfer entry: kernel.py::kernel
"""
import torch
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

# <NAME>revert_evict_first_restore_baseline_fix_imports</NAME>
import torch
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

@triton.jit
def dispatch_scatter_kernel(
    TopkIdx_ptr, Weights_ptr, Offsets_ptr, CurrentCnts_ptr,
    SortedTokenIds_ptr, SortedWeights_ptr,
    num_assignments, K, local_start, num_local_experts,
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
def _scan_add(a, b):
    return a + b

@triton.jit
def scan_and_zero_kernel(Counts_ptr, Offsets_ptr, num_items, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < num_items
    c = tl.load(Counts_ptr + offs, mask=mask, other=0)
    acc = tl.associative_scan(c, 0, _scan_add)
    tl.store(Offsets_ptr + 1 + offs, acc, mask=mask)
    tl.store(Offsets_ptr + offs, 0, mask=(offs == 0))
    tl.store(Counts_ptr + offs, 0, mask=mask)

@triton.jit
def predequant_hidden_kernel(
    A_ptr, A_scale_ptr, A_dq_ptr, Out_ptr, T, K, N2,
    stride_am, stride_ak, stride_asm, stride_ask,
    stride_dm, stride_dk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < T
    k_mask = offs_k < K
    mask = m_mask[:, None] & k_mask[None, :]
    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask, other=0.0)
    k_blk = offs_k // 128
    s = tl.load(A_scale_ptr + offs_m[:, None] * stride_asm + k_blk[None, :] * stride_ask, mask=mask, other=1.0)
    a_dq = (a.to(tl.float32) * s.to(tl.float32)).to(tl.bfloat16)
    tl.store(A_dq_ptr + offs_m[:, None] * stride_dm + offs_k[None, :] * stride_dk, a_dq, mask=mask)

    n2_mask = m_mask[:, None] & (offs_k[None, :] < N2)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on, 0.0, mask=n2_mask)

@triton.jit
def gemm1_mloop_kernel(
    A_dq_ptr, Idx_ptr, Offsets_ptr,
    W_ptr, W_scale_ptr, Out_ptr,
    Ws_gate_ptr, Ws_up_ptr,
    H, K,
    stride_adm, stride_adk,
    stride_we, stride_wn, stride_wk, stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    stride_wsg_sk, stride_wsg_m, stride_wsg_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    eid = tl.program_id(1)
    pid_k = tl.program_id(2)
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
    k_start = pid_k * BLOCK_K
    k_step = SPLIT_K * BLOCK_K
    for m_base in range(0, count, BLOCK_M):
        offs_m = m_base + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count
        token_offset = off_start + offs_m
        idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)
        a_ptr_base = A_dq_ptr + idx[:, None] * stride_adm
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(k_start, K, k_step):
            k_cur = k + offs_k
            a = tl.load(
                a_ptr_base + k_cur[None, :] * stride_adk,
                mask=m_mask[:, None],
                other=0.0,
            )

            w_gate_blk = tl.make_block_ptr(
                base=w_ptr_base,
                shape=(K, H),
                strides=(stride_wk, stride_wn),
                offsets=(k, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            w_up_blk = tl.make_block_ptr(
                base=w_ptr_base + H * stride_wn,
                shape=(K, H),
                strides=(stride_wk, stride_wn),
                offsets=(k, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
            w_gate = tl.load(w_gate_blk, boundary_check=(0, 1), padding_option="zero")
            w_up = tl.load(w_up_blk, boundary_check=(0, 1), padding_option="zero")

            k_blk = k // 128
            sw_gate = tl.load(ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)
            sw_up = tl.load(ws_ptr_base + ((offs_n[None, :] + H) // 128) * stride_wsn + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)
            w_gate_dq = (w_gate.to(tl.float32) * sw_gate.to(tl.float32)).to(tl.bfloat16)
            w_up_dq = (w_up.to(tl.float32) * sw_up.to(tl.float32)).to(tl.bfloat16)
            acc_gate += tl.dot(a, w_gate_dq, allow_tf32=True)
            acc_up += tl.dot(a, w_up_dq, allow_tf32=True)
        if SPLIT_K == 1:
            acc_up = acc_up * tl.sigmoid(acc_up)
            out = (acc_gate * acc_up).to(tl.bfloat16)
            out_ptr_base = Out_ptr + token_offset[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptr_base, out, mask=m_mask[:, None] & n_mask[None, :])
        else:
            gate_ptr = Ws_gate_ptr + pid_k * stride_wsg_sk + token_offset[:, None] * stride_wsg_m + offs_n[None, :] * stride_wsg_n
            up_ptr = Ws_up_ptr + pid_k * stride_wsg_sk + token_offset[:, None] * stride_wsg_m + offs_n[None, :] * stride_wsg_n
            tl.store(gate_ptr, acc_gate, mask=m_mask[:, None] & n_mask[None, :])
            tl.store(up_ptr, acc_up, mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def gemm1_reduce_swiglu_kernel(
    Ws_gate_ptr, Ws_up_ptr, Out_ptr,
    num_assignments, H,
    stride_wsg_sk, stride_wsg_m, stride_wsg_n,
    stride_om, stride_on,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_assignments
    n_mask = offs_n < H
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for sk in range(SPLIT_K):
        gate = tl.load(Ws_gate_ptr + sk * stride_wsg_sk + offs_m[:, None] * stride_wsg_m + offs_n[None, :] * stride_wsg_n, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        up = tl.load(Ws_up_ptr + sk * stride_wsg_sk + offs_m[:, None] * stride_wsg_m + offs_n[None, :] * stride_wsg_n, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
        acc_gate += gate
        acc_up += up
    acc_up = acc_up * tl.sigmoid(acc_up)
    out = (acc_gate * acc_up).to(tl.bfloat16)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def gemm2_mloop_kernel(
    C_ptr, Offsets_ptr, Weight_ptr, Idx_ptr,
    W_ptr, W_scale_ptr, Out_ptr, N, K,
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
            k_cur = k + offs_k
            k_mask = k_cur < K
            c = tl.load(c_ptr_base + k_cur[None, :] * stride_ck, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_tile = tl.load(w_ptr_base + offs_n[None, :] * stride_wn + k_cur[:, None] * stride_wk, mask=n_mask[None, :] & k_mask[:, None], other=0.0, eviction_policy='evict_last')
            k_blk = k // 128
            sw = tl.load(ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk, mask=n_mask[None, :], other=1.0)
            w_dq = (w_tile.to(tl.float32) * sw.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(c, w_dq, allow_tf32=True)
        acc = acc * gating_w[:, None]
        orig_idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)
        out_ptrs = Out_ptr + orig_idx[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def fused_route_topk_kernel(
    Logits_ptr, Bias_ptr, TopkWeights_ptr, TopkIdx_ptr, Counts_ptr,
    stride_l_t, stride_l_e, stride_w_t, stride_w_k, stride_i_t, stride_i_k,
    T, routed_scaling_factor, local_start, num_local_experts,
    E_GLOBAL: tl.constexpr, N_GROUP: tl.constexpr, GROUP_SIZE: tl.constexpr,
    TOPK_GROUP: tl.constexpr, TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= T:
        return
    offs = tl.arange(0, E_GLOBAL)
    logits = tl.load(Logits_ptr + pid * stride_l_t + offs * stride_l_e)
    bias = tl.load(Bias_ptr + offs)
    s = tl.sigmoid(logits)
    s_wb = s + bias
    s_wb_2d = tl.reshape(s_wb, (N_GROUP, GROUP_SIZE))
    max_idx1 = tl.argmax(s_wb_2d, axis=1)
    offs_group = tl.arange(0, GROUP_SIZE)
    mask1 = offs_group[None, :] == max_idx1[:, None]
    s_wb_2d_masked = tl.where(mask1, -float('inf'), s_wb_2d)
    max_val1 = tl.max(s_wb_2d, axis=1)
    max_val2 = tl.max(s_wb_2d_masked, axis=1)
    group_scores = max_val1 + max_val2
    offs_ng = tl.arange(0, N_GROUP)
    valid_groups_mask = tl.zeros((N_GROUP,), dtype=tl.int32)
    for _ in range(TOPK_GROUP):
        g_max_idx = tl.argmax(group_scores, axis=0)
        valid_groups_mask = tl.where(offs_ng == g_max_idx, 1, valid_groups_mask)
        group_scores = tl.where(offs_ng == g_max_idx, -float('inf'), group_scores)
    valid_groups_mask_2d = tl.broadcast_to(valid_groups_mask[:, None], (N_GROUP, GROUP_SIZE))
    valid_elements_mask = tl.reshape(valid_groups_mask_2d, (E_GLOBAL,))
    s_wb_filtered = tl.where(valid_elements_mask == 1, s_wb, -float('inf'))
    sum_s = 0.0
    for k in range(TOP_K):
        max_idx = tl.argmax(s_wb_filtered, axis=0)
        s_val = tl.sum(tl.where(offs == max_idx, s, 0.0), axis=0)
        expert_i32 = max_idx.to(tl.int32)
        tl.store(TopkWeights_ptr + pid * stride_w_t + k * stride_w_k, s_val)
        tl.store(TopkIdx_ptr + pid * stride_i_t + k * stride_i_k, expert_i32)
        expert_local = expert_i32 - local_start
        is_local = (expert_local >= 0) & (expert_local < num_local_experts)
        tl.atomic_add(Counts_ptr + expert_local, 1, mask=is_local)
        sum_s += s_val
        s_wb_filtered = tl.where(offs == max_idx, -float('inf'), s_wb_filtered)
    offs_k = tl.arange(0, TOP_K)
    topk_s_loaded = tl.load(TopkWeights_ptr + pid * stride_w_t + offs_k * stride_w_k)
    assign_w = topk_s_loaded * (routed_scaling_factor / (sum_s + 1e-20))
    tl.store(TopkWeights_ptr + pid * stride_w_t + offs_k * stride_w_k, assign_w)


_workspace_cache = {}
_routing_buffer_cache = {}
_pipeline_cache = {}

def _exclusive_cumsum_zero_triton(counts, offsets, num_items):
    block = min(1024, triton.next_power_of_2(max(1, int(num_items))))
    scan_and_zero_kernel[(1,)](counts, offsets, num_items, BLOCK=block, num_warps=4, num_stages=3)

def _get_routing_buffers(device, T, top_k_val, num_assignments, num_local_experts):
    state = _routing_buffer_cache.get(device)
    if state is None:
        t_cap, a_cap, e_cap = max(1, T), max(1, num_assignments), max(1, num_local_experts)
    else:
        t_cap, a_cap, e_cap = state["t_cap"], state["a_cap"], state["e_cap"]
    need_grow = state is None or T > t_cap or num_assignments > a_cap or num_local_experts > e_cap
    if need_grow:
        if state is not None:
            t_cap, a_cap, e_cap = max(T, t_cap * 2), max(num_assignments, a_cap * 2), max(num_local_experts, e_cap * 2)
        _routing_buffer_cache[device] = {
            "t_cap": t_cap, "a_cap": a_cap, "e_cap": e_cap,
            "assign_w": torch.empty((t_cap, TOP_K), device=device, dtype=torch.float32),
            "topk_idx": torch.empty((t_cap, TOP_K), device=device, dtype=torch.int32),
            "counts": torch.zeros(e_cap, device=device, dtype=torch.int32),
            "offsets": torch.empty(e_cap + 1, device=device, dtype=torch.int32),
            "sorted_token_ids": torch.empty(a_cap, device=device, dtype=torch.int32),
            "sorted_weights": torch.empty(a_cap, device=device, dtype=torch.float32),
        }
    state = _routing_buffer_cache[device]
    return (
        state["assign_w"][:T, :top_k_val],
        state["topk_idx"][:T, :top_k_val],
        state["counts"][:num_local_experts],
        state["offsets"][:num_local_experts + 1],
        state["sorted_token_ids"][:num_assignments],
        state["sorted_weights"][:num_assignments],
    )

def _run_compute_pipeline(
    routing_logits, routing_bias, assign_w, topk_idx,
    hidden_states, hidden_states_scale, hidden_states_dq,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_start, routed_scaling_factor,
    T, num_experts, K1, N2, K2, H, num_assignments, TOP_K_val,
    counts, offsets, sorted_token_ids, sorted_weights, C_act, out_bf16,
    ws_gate, ws_up,
    is_medium_large_regime, is_huge_regime, num_local_experts,
):
    counts.zero_()
    fused_route_topk_kernel[(T,)](
        routing_logits, routing_bias, assign_w, topk_idx, counts,
        routing_logits.stride(0), routing_logits.stride(1),
        assign_w.stride(0), assign_w.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        T, routed_scaling_factor, local_start, num_local_experts,
        E_GLOBAL=E_GLOBAL, N_GROUP=N_GROUP, GROUP_SIZE=GROUP_SIZE,
        TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K_val, num_warps=4, num_stages=1
    )
    _exclusive_cumsum_zero_triton(counts, offsets, num_local_experts)

    if num_assignments < 1024:
        BDS, sw_, ss_ = 64, 2, 1
    elif num_assignments < 4096:
        BDS, sw_, ss_ = 128, 4, 1
    else:
        BDS, sw_, ss_ = 1024, 2, 2

    dispatch_scatter_kernel[(triton.cdiv(num_assignments, BDS),)](
        topk_idx, assign_w, offsets, counts, sorted_token_ids, sorted_weights,
        num_assignments, TOP_K_val, local_start, num_local_experts, 1, 1,
        BLOCK=BDS, num_warps=sw_, num_stages=ss_
    )

    PRE_BM, PRE_BK = 32, 128
    max_k = max(K1, N2)
    predequant_hidden_kernel[(triton.cdiv(T, PRE_BM), triton.cdiv(max_k, PRE_BK))](
        hidden_states, hidden_states_scale, hidden_states_dq, out_bf16, T, K1, N2,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(1), hidden_states_scale.stride(0),
        hidden_states_dq.stride(0), hidden_states_dq.stride(1),
        out_bf16.stride(0), out_bf16.stride(1),
        BLOCK_M=PRE_BM, BLOCK_K=PRE_BK, num_warps=4, num_stages=2
    )

    if num_assignments >= 8192:
        BM1, BM2, gs1, gs2 = 256, 256, 3, 3
        BN1, BK1, BK2, BN2 = 128, 64, 64, 128
        gw1, gw2 = 8, 8
        SPLIT_K = 1
    elif num_assignments >= 4096:
        BM1, BM2, gs1, gs2 = 64, 64, 3, 3
        BN1, BK1, BK2, BN2 = 128, 64, 64, 128
        gw1, gw2 = 8, 8
        SPLIT_K = 1
    elif num_assignments >= 1024:
        BM1, BM2, gs1, gs2 = 32, 32, 3, 3
        BN1, BK1, BK2, BN2 = 64, 64, 64, 64
        gw1, gw2 = 4, 4
        SPLIT_K = 1
    elif num_assignments >= 128:
        BM1, BM2, gs1, gs2 = 16, 16, 2, 2
        BN1, BK1, BK2, BN2 = 64, 64, 64, 64
        gw1, gw2 = 2, 2
        SPLIT_K = 1
    else:
        BM1, BM2, gs1, gs2 = 16, 16, 3, 2
        BN1, BK1, BK2, BN2 = 64, 64, 64, 64
        gw1, gw2 = 4, 2
        SPLIT_K = 4

    gemm1_mloop_kernel[(triton.cdiv(H, BN1), num_experts, SPLIT_K)](
        hidden_states_dq, sorted_token_ids, offsets,
        gemm1_weights, gemm1_weights_scale, C_act,
        ws_gate, ws_up,
        H, K1,
        hidden_states_dq.stride(0), hidden_states_dq.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        C_act.stride(0), C_act.stride(1),
        ws_gate.stride(0), ws_gate.stride(1), ws_gate.stride(2),
        BLOCK_M=BM1, BLOCK_N=BN1, BLOCK_K=BK1,
        SPLIT_K=SPLIT_K, num_warps=gw1, num_stages=gs1
    )

    if SPLIT_K > 1 and num_assignments > 0:
        gemm1_reduce_swiglu_kernel[(triton.cdiv(num_assignments, 64), triton.cdiv(H, 64))](
            ws_gate, ws_up, C_act,
            num_assignments, H,
            ws_gate.stride(0), ws_gate.stride(1), ws_gate.stride(2),
            C_act.stride(0), C_act.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

    gemm2_mloop_kernel[(triton.cdiv(N2, BN2), num_experts)](
        C_act, offsets, sorted_weights, sorted_token_ids,
        gemm2_weights, gemm2_weights_scale, out_bf16, N2, K2,
        C_act.stride(0), C_act.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        out_bf16.stride(0), out_bf16.stride(1),
        BLOCK_M=BM2, BLOCK_N=BN2, BLOCK_K=BK2, num_warps=gw2, num_stages=gs2
    )

@torch.no_grad()
def custom_kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    is_medium_large_regime = T >= 901 and T < 10000
    is_huge_regime = T >= 10000
    num_experts = int(gemm1_weights.shape[0])
    N1, K1 = int(gemm1_weights.shape[1]), int(gemm1_weights.shape[2])
    N2, K2 = int(gemm2_weights.shape[1]), int(gemm2_weights.shape[2])
    H = N1 // 2
    TOP_K_val = TOP_K
    num_assignments = T * TOP_K_val
    num_local_experts = num_experts
    local_start = int(local_expert_offset)

    assign_w, topk_idx, counts, offsets, sorted_token_ids, sorted_weights = _get_routing_buffers(
        device, T, TOP_K_val, num_assignments, num_local_experts
    )

    MAX_SPLIT_K = 4
    need_split = (num_assignments < 4096)
    ws_key = (device, T, num_assignments, K1, H, N2)
    if ws_key not in _workspace_cache:
        _workspace_cache[ws_key] = (
            torch.empty((T, K1), device=device, dtype=torch.bfloat16),
            torch.empty((num_assignments, H), device=device, dtype=torch.bfloat16),
            torch.empty((T, N2), device=device, dtype=torch.bfloat16),
            torch.empty((MAX_SPLIT_K, num_assignments, H), device=device, dtype=torch.float32) if need_split else torch.empty((1, 1, 1), device=device, dtype=torch.float32),
            torch.empty((MAX_SPLIT_K, num_assignments, H), device=device, dtype=torch.float32) if need_split else torch.empty((1, 1, 1), device=device, dtype=torch.float32),
        )
    hidden_states_dq, C_act, out_bf16, ws_gate, ws_up = _workspace_cache[ws_key]

    pkey = (
        T, device, id(gemm1_weights), id(gemm1_weights_scale), id(gemm2_weights), id(gemm2_weights_scale),
        int(routing_logits.data_ptr()), int(routing_bias.data_ptr()),
        int(hidden_states.data_ptr()), int(hidden_states_scale.data_ptr()),
    )
    if pkey not in _pipeline_cache:
        args = (
            routing_logits, routing_bias, assign_w, topk_idx,
            hidden_states, hidden_states_scale, hidden_states_dq,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_start, routed_scaling_factor,
            T, num_experts, K1, N2, K2, H, num_assignments, TOP_K_val,
            counts, offsets, sorted_token_ids, sorted_weights, C_act, out_bf16,
            ws_gate, ws_up,
            is_medium_large_regime, is_huge_regime, num_local_experts,
        )
        _run_compute_pipeline(*args)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_compute_pipeline(*args)
        _pipeline_cache[pkey] = (g,)

    (g,) = _pipeline_cache[pkey]
    g.replay()
    return out_bf16

# FlashInfer entry point
kernel = custom_kernel

def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    return custom_kernel(routing_logits, routing_bias, hidden_states,
                         hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                         gemm2_weights, gemm2_weights_scale,
                         local_expert_offset, routed_scaling_factor)
