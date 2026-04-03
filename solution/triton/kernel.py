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
def _scan_add(a, b):
    return a + b


@triton.jit
def scan_and_zero_kernel(
    Counts_ptr,
    Offsets_ptr,
    num_items,
    BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < num_items

    c = tl.load(Counts_ptr + offs, mask=mask, other=0)
    acc = tl.associative_scan(c, 0, _scan_add)

    tl.store(Offsets_ptr + 1 + offs, acc, mask=mask)
    tl.store(Offsets_ptr + offs, 0, mask=(offs == 0))
    tl.store(Counts_ptr + offs, 0, mask=mask)


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

    w_ptr_base = W_ptr + eid * stride_we
    ws_ptr_base = W_scale_ptr + eid * stride_wse

    for m_base in range(0, count, BLOCK_M):
        offs_m = m_base + tl.arange(0, BLOCK_M)
        m_mask = offs_m < count

        token_offset = off_start + offs_m
        idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)

        a_ptr_iter = A_ptr + idx[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_scale_base = A_scale_ptr + idx[:, None] * stride_asm
        
        w_gate_ptr_iter = w_ptr_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_up_ptr_iter = w_ptr_base + (offs_n[None, :] + H) * stride_wn + offs_k[:, None] * stride_wk

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Unpredicated inner loop via pointer scalar increments
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr_iter)

            k_blk = k // 128
            sa = tl.load(a_scale_base + k_blk * stride_ask)

            # Action: gemm1_weight_evict_first
            w_gate = tl.load(w_gate_ptr_iter, eviction_policy='evict_first')
            w_up = tl.load(w_up_ptr_iter, eviction_policy='evict_first')

            sw_gate = tl.load(ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk)
            sw_up = tl.load(ws_ptr_base + ((offs_n[None, :] + H) // 128) * stride_wsn + k_blk * stride_wsk)

            a_dq = (a.to(tl.float32) * sa).to(tl.bfloat16)
            w_gate_dq = (w_gate.to(tl.float32) * sw_gate).to(tl.bfloat16)
            w_up_dq = (w_up.to(tl.float32) * sw_up).to(tl.bfloat16)
            acc_gate += tl.dot(a_dq, w_gate_dq)
            acc_up += tl.dot(a_dq, w_up_dq)

            a_ptr_iter += BLOCK_K * stride_ak
            w_gate_ptr_iter += BLOCK_K * stride_wk
            w_up_ptr_iter += BLOCK_K * stride_wk

        # Downcast accumulators before gating multiplication
        acc_up_bf16 = (acc_up * tl.sigmoid(acc_up)).to(tl.bfloat16)
        acc_gate_bf16 = acc_gate.to(tl.bfloat16)
        out = acc_gate_bf16 * acc_up_bf16

        out_block_ptr = tl.make_block_ptr(
            base=Out_ptr + off_start * stride_om,
            shape=(count, H),
            strides=(stride_om, stride_on),
            offsets=(m_base, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(out_block_ptr, out, boundary_check=(0, 1))


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

        c_ptr_iter = C_ptr + token_offset[:, None] * stride_cm + offs_k[None, :] * stride_ck
        w_ptr_iter = w_ptr_base + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Unpredicated inner loop via pointer scalar increments
        for k in range(0, K, BLOCK_K):
            c = tl.load(c_ptr_iter)

            # Evict first to prevent the massive K=14336 weight tensors from polluting L2 cache
            w_tile = tl.load(w_ptr_iter, eviction_policy='evict_first')

            k_blk = k // 128
            sw = tl.load(ws_ptr_base + (offs_n[None, :] // 128) * stride_wsn + k_blk * stride_wsk)

            w_dq = (w_tile.to(tl.float32) * sw).to(tl.bfloat16)
            acc += tl.dot(c, w_dq)

            c_ptr_iter += BLOCK_K * stride_ck
            w_ptr_iter += BLOCK_K * stride_wk

        # Downcast accumulators and gating weights before elementwise multiply to reduce ALU cost
        gating_w_bf16 = gating_w[:, None].to(tl.bfloat16)
        acc_bf16 = acc.to(tl.bfloat16) * gating_w_bf16
        orig_idx = tl.load(Idx_ptr + token_offset, mask=m_mask, other=0)
        out_ptrs = Out_ptr + orig_idx[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.atomic_add(out_ptrs, acc_bf16, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def fused_route_topk_kernel(
    Logits_ptr, Bias_ptr,
    TopkWeights_ptr, TopkIdx_ptr,
    Counts_ptr,
    stride_l_t, stride_l_e,
    stride_w_t, stride_w_k,
    stride_i_t, stride_i_k,
    T, routed_scaling_factor,
    local_start, num_local_experts,
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


def _exclusive_cumsum_zero_triton(counts: torch.Tensor, offsets: torch.Tensor, num_items: int):
    block = min(1024, triton.next_power_of_2(max(1, int(num_items))))
    scan_and_zero_kernel[(1,)](
        counts, offsets,
        num_items,
        BLOCK=block,
        num_warps=4, num_stages=3,
    )


def _get_routing_buffers(device, T, top_k_val, num_assignments, num_local_experts):
    state = _routing_buffer_cache.get(device)
    if state is None:
        t_cap = max(1, T)
        a_cap = max(1, num_assignments)
        e_cap = max(1, num_local_experts)
    else:
        t_cap = state["t_cap"]
        a_cap = state["a_cap"]
        e_cap = state["e_cap"]

    need_grow = (
        state is None
        or T > t_cap
        or num_assignments > a_cap
        or num_local_experts > e_cap
    )
    if need_grow:
        if state is not None:
            t_cap = max(T, t_cap * 2)
            a_cap = max(num_assignments, a_cap * 2)
            e_cap = max(num_local_experts, e_cap * 2)
        _routing_buffer_cache[device] = {
            "t_cap": t_cap,
            "a_cap": a_cap,
            "e_cap": e_cap,
            "assign_w": torch.empty((t_cap, TOP_K), device=device, dtype=torch.float32),
            "topk_idx": torch.empty((t_cap, TOP_K), device=device, dtype=torch.int32),
            "counts": torch.zeros(e_cap, device=device, dtype=torch.int32),
            "offsets": torch.empty(e_cap + 1, device=device, dtype=torch.int32),
            "sorted_token_ids": torch.empty(a_cap, device=device, dtype=torch.int32),
            "sorted_weights": torch.empty(a_cap, device=device, dtype=torch.float32),
        }
        state = _routing_buffer_cache[device]
    else:
        state = _routing_buffer_cache[device]

    assign_w = state["assign_w"][:T, :top_k_val]
    topk_idx = state["topk_idx"][:T, :top_k_val]
    counts = state["counts"][:num_local_experts]
    offsets = state["offsets"][:num_local_experts + 1]
    sorted_token_ids = state["sorted_token_ids"][:num_assignments]
    sorted_weights = state["sorted_weights"][:num_assignments]
    return assign_w, topk_idx, counts, offsets, sorted_token_ids, sorted_weights


def _run_compute_pipeline(
    routing_logits, routing_bias, assign_w, topk_idx,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_start, routed_scaling_factor,
    T, num_experts, K1, N2, K2, H, num_assignments, TOP_K_val,
    counts, offsets, sorted_token_ids, sorted_weights, C_act, out_bf16,
    is_medium_large_regime, is_huge_regime, num_local_experts,
):
    counts.zero_()
    fused_route_topk_kernel[(T,)](
        routing_logits, routing_bias,
        assign_w, topk_idx, counts,
        routing_logits.stride(0), routing_logits.stride(1),
        assign_w.stride(0), assign_w.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        T, routed_scaling_factor, local_start, num_local_experts,
        E_GLOBAL=E_GLOBAL, N_GROUP=N_GROUP, GROUP_SIZE=GROUP_SIZE,
        TOPK_GROUP=TOPK_GROUP, TOP_K=TOP_K_val,
        num_warps=4, num_stages=1
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
        BLOCK=BDS, num_warps=sw_, num_stages=ss_,
    )

    if num_assignments >= 8192:
        BM1, BM2, gs1, gs2 = 128, 128, 4, 4
        BN1, BK1, BK2, BN2 = 128, 64, 64, 128
        gw1, gw2 = 8, 8
    elif num_assignments >= 4096:
        BM1, BM2, gs1, gs2 = 64, 64, 4, 4
        BN1, BK1, BK2, BN2 = 128, 64, 64, 128
        gw1, gw2 = 8, 8
    elif num_assignments >= 1024:
        BM1, BM2, gs1, gs2 = 32, 32, 4, 4
        BN1, BK1, BK2, BN2 = 64, 64, 64, 64
        gw1, gw2 = 4, 4
    else:
        BM1, BM2, gs1, gs2 = 16, 16, 3, 3
        BN1, BK1, BK2, BN2 = 64, 64, 64, 64
        gw1, gw2 = 2, 2

    gemm1_num_warps = gw1
    if BM1 <= 32:
        gemm1_num_warps = gw1

    gemm1_mloop_kernel[(triton.cdiv(H, BN1), num_experts)](
        hidden_states, hidden_states_scale, sorted_token_ids, offsets,
        gemm1_weights, gemm1_weights_scale, C_act, H, K1,
        hidden_states.stride(0), hidden_states.stride(1),
        hidden_states_scale.stride(1), hidden_states_scale.stride(0),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        C_act.stride(0), C_act.stride(1),
        BLOCK_M=BM1, BLOCK_N=BN1, BLOCK_K=BK1, num_warps=gemm1_num_warps, num_stages=gs1,
    )
    out_bf16.zero_()
    gemm2_mloop_kernel[(triton.cdiv(N2, BN2), num_experts)](
        C_act, offsets, sorted_weights, sorted_token_ids,
        gemm2_weights, gemm2_weights_scale, out_bf16, N2, K2,
        C_act.stride(0), C_act.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        out_bf16.stride(0), out_bf16.stride(1),
        BLOCK_M=BM2, BLOCK_N=BN2, BLOCK_K=BK2, num_warps=gw2, num_stages=gs2,
    )


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
    is_medium_large_regime = T >= 901 and T < 10000
    is_huge_regime = T >= 10000

    num_experts = int(gemm1_weights.shape[0])
    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])
    H = N1 // 2

    TOP_K_val = TOP_K
    num_assignments = T * TOP_K_val
    num_local_experts = num_experts
    local_start = int(local_expert_offset)

    assign_w, topk_idx, counts, offsets, sorted_token_ids, sorted_weights = _get_routing_buffers(
        device, T, TOP_K_val, num_assignments, num_local_experts
    )

    ws_key = (device, T, num_assignments, H, N2)
    if ws_key not in _workspace_cache:
        _workspace_cache[ws_key] = (
            torch.empty((num_assignments + 128, H), device=device, dtype=torch.bfloat16), # Padded to allow unmasked load/stores
            torch.empty((T, N2), device=device, dtype=torch.bfloat16),
        )
    C_act, out_bf16 = _workspace_cache[ws_key]

    pkey = (T, device, id(gemm1_weights), id(gemm1_weights_scale), id(gemm2_weights), id(gemm2_weights_scale))
    if pkey not in _pipeline_cache:
        static_logits = torch.empty_like(routing_logits)
        static_bias = torch.empty_like(routing_bias)
        static_hs = torch.empty_like(hidden_states)
        static_hs_scale = torch.empty_like(hidden_states_scale)

        static_logits.copy_(routing_logits)
        static_bias.copy_(routing_bias)
        static_hs.copy_(hidden_states)
        static_hs_scale.copy_(hidden_states_scale)

        args = (
            static_logits, static_bias, assign_w, topk_idx,
            static_hs, static_hs_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_start, routed_scaling_factor,
            T, num_experts, K1, N2, K2, H, num_assignments, TOP_K_val,
            counts, offsets, sorted_token_ids, sorted_weights, C_act, out_bf16,
            is_medium_large_regime, is_huge_regime, num_local_experts,
        )
        _run_compute_pipeline(*args)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_compute_pipeline(*args)
        _pipeline_cache[pkey] = (g, static_logits, static_bias, static_hs, static_hs_scale)

    g, static_logits, static_bias, static_hs, static_hs_scale = _pipeline_cache[pkey]
    static_logits.copy_(routing_logits)
    static_bias.copy_(routing_bias)
    static_hs.copy_(hidden_states)
    static_hs_scale.copy_(hidden_states_scale)
    g.replay()
    # Action: optimize_inner_loops_and_pipeline
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
