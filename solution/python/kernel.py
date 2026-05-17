"""DeepSeek-V3 fused MoE kernel for the FlashInfer contest.

The implementation uses a Python entry point that JIT-compiles CUDA/CUTLASS
helpers and runs a fused routing, dispatch, grouped-GEMM, SwiGLU, and
reduce-scatter pipeline. Small and medium workloads use a CUDA-graph-safe
fixed-shape path; large workloads use a compact dynamic path with an SM100 MxF8
hardware block-scaled GEMM sub-path.
"""
import glob
import os
import tempfile
import torch

E_GLOBAL = 256
N_GROUP = 8
TOPK_GROUP = 4
GROUP_SIZE = 32
TOP_K = 8

_ext = None
_CUDA_SOURCE_DIR = os.path.dirname(__file__)
_MOE_GEMM_CU_PATH = os.path.join(_CUDA_SOURCE_DIR, "moe_gemm.cu")
_MOE_FUSED_CU_PATH = os.path.join(_CUDA_SOURCE_DIR, "moe_fused.cu")


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    cuda_home = None
    for cand in ("/usr/local/cuda-13.0", "/usr/local/cuda-13", "/usr/local/cuda"):
        nvcc = os.path.join(cand, "bin", "nvcc")
        if os.path.exists(nvcc):
            cuda_home = cand
            os.environ["CUDA_HOME"] = cand
            os.environ["CUDACXX"] = nvcc
            break

    cutlass_includes = set()
    preferred_roots = [
        os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/include"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia_cutlass_dsl/include"),
    ]
    for root in preferred_roots:
        if os.path.exists(os.path.join(root, "cutlass", "cutlass.h")):
            cutlass_includes.add(root)
    util_root = os.path.expanduser("~/.local/lib/python3.12/site-packages/flashinfer/data/cutlass/tools/util/include")
    if os.path.exists(os.path.join(util_root, "cutlass", "util", "packed_stride.hpp")):
        cutlass_includes.add(util_root)
    for f in glob.glob("/opt/conda/**/cutlass/cutlass.h", recursive=True):
        cutlass_includes.add(os.path.dirname(os.path.dirname(f)))
    for f in glob.glob("/opt/conda/**/cutlass/util/packed_stride.hpp", recursive=True):
        cutlass_includes.add(os.path.dirname(os.path.dirname(os.path.dirname(f))))
    if not cutlass_includes:
        raise RuntimeError("CUTLASS headers not found")

    # Prefer a persistent volume path for the JIT build dir so subsequent Modal
    # runs reuse the compiled extension (.so). Fall back to /tmp if /mnt is not
    # writable (e.g., when running outside the bench container).
    candidates = [
        "/mnt/build_cache/fused_moe_cutlass_v6_multifile",
        os.path.join(tempfile.gettempdir(), "fused_moe_cutlass_v6_multifile"),
    ]
    build_dir = None
    for cand in candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            with open(os.path.join(cand, ".probe"), "w", encoding="utf-8") as f:
                f.write("ok")
            build_dir = cand
            break
        except OSError:
            continue
    assert build_dir is not None, "No writable build dir found"

    if not os.path.exists(_MOE_GEMM_CU_PATH) or not os.path.exists(_MOE_FUSED_CU_PATH):
        raise RuntimeError("Packaged CUDA source files are missing")

    def _copy_if_changed(src, dst):
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        existing = None
        if os.path.exists(dst):
            try:
                with open(dst, "r", encoding="utf-8") as f:
                    existing = f.read()
            except OSError:
                existing = None
        if existing != content:
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)

    # Compile stable build-cache paths, not the per-extraction source paths.
    # This keeps the repo/package readable while preserving the warm-cache
    # behavior of the original embedded-string implementation.
    cutlass_cu = os.path.join(build_dir, "moe_cutlass.cu")
    fused_cu = os.path.join(build_dir, "moe_fused.cu")
    _copy_if_changed(_MOE_GEMM_CU_PATH, cutlass_cu)
    _copy_if_changed(_MOE_FUSED_CU_PATH, fused_cu)

    # Separate translation units let benchmark runs reuse the stable
    # CUTLASS-heavy object file while rebuilding only small helper changes.
    import torch.utils.cpp_extension as cpp_ext
    if cuda_home is not None:
        cpp_ext.CUDA_HOME = cuda_home
    load = cpp_ext.load
    # Stream compiler output so long template instantiations are visible.
    _verbose = bool(int(os.environ.get("VERBOSE_BUILD", "1")))
    extra_flags = [
        "-O3", "--std=c++17", "-arch=sm_100a",
        "--expt-relaxed-constexpr", "-DNDEBUG",
        # Parallelize template instantiation across nvcc passes — major win
        # on the CUTLASS-heavy GEMM unit (3min → ~1-1.5min cold)
        # across files.
        "--threads=4",
    ]
    _ext = load(
        name="moe_gemm_v6_multifile",
        sources=[cutlass_cu, fused_cu],
        extra_include_paths=sorted(cutlass_includes),
        extra_cuda_cflags=extra_flags,
        build_directory=build_dir,
        verbose=_verbose,
    )
    return _ext


# ------------------------------ pipeline --------------------------------



_route = None  # set after _route_fused is defined


def _route_fused(routing_logits, routing_bias, rsf, T, local_start, num_experts,
                 topk_idx_buf=None, assign_w_buf=None):
    """Single-kernel fused routing (DeepSeek-V3 topk8+group4). 6-32x faster
    than the PyTorch chain. Requires bf16 logits/bias (contest format).

    When called with pre-allocated buffers (graph-capture path), this function
    does ONE kernel launch total — no torch.empty, no dtype casts.
    """
    ext = _get_ext()
    logits = routing_logits if routing_logits.dtype == torch.bfloat16 \
        else routing_logits.to(torch.bfloat16)
    bias = routing_bias if routing_bias.dtype == torch.bfloat16 \
        else routing_bias.to(torch.bfloat16)
    if topk_idx_buf is None:
        topk_idx_buf = torch.empty(T, TOP_K, device=logits.device, dtype=torch.int32)
    if assign_w_buf is None:
        assign_w_buf = torch.empty(T, TOP_K, device=logits.device, dtype=torch.float32)
    ext.fused_route_topk(logits, bias, topk_idx_buf, assign_w_buf, float(rsf))
    return topk_idx_buf, assign_w_buf


_route = _route_fused


def _dispatch_graph_safe(topk_idx, assign_w, T, local_start, num_experts, bufs):
    """Graph-capture-safe dispatch using the single CUDA fused_dispatch kernel.
    Replaces argsort + scatter_add + where chain (~40-60μs on big-T) with 3
    tiny kernels (~10μs total). Also populates problem_sizes_{1,2}[:, 0].
    """
    ext = _get_ext()
    counts = bufs["counts_buf"]
    offsets = bufs["offsets_buf"]
    sorted_tids = bufs["sorted_tids_buf"]
    sorted_weights = bufs["sorted_weights_buf"]
    # topk_idx / assign_w come from pre-allocated buffers => contiguous by
    # construction. Skip .contiguous() to avoid an extra graph-node.
    ext.fused_dispatch(
        topk_idx, assign_w,
        int(local_start), int(num_experts),
        counts, sorted_tids, sorted_weights, offsets,
        bufs["problem_sizes_1"], bufs["problem_sizes_2"])
    return counts, sorted_tids, sorted_weights


def _dispatch_dynamic(topk_idx, assign_w, T, local_start, num_experts, bufs):
    """Large-T dispatch: single fused CUDA kernel (bincount + sort + gather)
    with compact output. Entries are written contiguously from slot 0, so
    [0:total_valid] extracts the valid ones for downstream compact paths.
    Also writes M-col of problem_sizes_{1,2} inside the scan kernel."""
    ext = _get_ext()
    counts = bufs["counts_buf"]
    offsets = bufs["offsets_buf"]
    sorted_tids = bufs["sorted_tids_buf"]
    sorted_weights = bufs["sorted_weights_buf"]
    ext.fused_dispatch(
        topk_idx, assign_w,
        int(local_start), int(num_experts),
        counts, sorted_tids, sorted_weights, offsets,
        bufs["problem_sizes_1"], bufs["problem_sizes_2"])
    total_valid = int(offsets[num_experts].item())
    return counts, sorted_tids[:total_valid], sorted_weights[:total_valid]


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


_workspace_cache = {}
_graph_cache = {}


def _get_workspace(device, ne, T, N1, K1, N2, K2):
    """Pre-allocate all intermediates once per workload (same addresses forever)."""
    total_tokens = T * TOP_K
    key = (device, ne, T, N1, K1, N2, K2)
    if key not in _workspace_cache:
        ext = _get_ext()
        stride_sz, sfa_sz, sfb_sz = ext.get_sizes()
        # Only allocate MxF8 workspace buffers for workloads that will take
        # the MxF8 path (matches the use_mxf8 gate in _run_pipeline_dynamic).
        _mxf8_enabled  = bool(int(os.environ.get("USE_MXF8", "1")))
        _mxf8_min_t    = int(os.environ.get("MXF8_MIN_T", "4096"))
        use_mxf8 = _mxf8_enabled and T >= _mxf8_min_t
        mxf8_stride_sz = int(ext.get_mxf8_sizes_stride()) if use_mxf8 else 1
        mxf8_sfa_sz    = int(ext.get_mxf8_sizes_layout_sfa()) if use_mxf8 else 1
        mxf8_sfb_sz    = int(ext.get_mxf8_sizes_layout_sfb()) if use_mxf8 else 1
        bufs = dict(
            stride_sz=stride_sz,
            sfa_sz=sfa_sz,
            sfb_sz=sfb_sz,
            a_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            b_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            out_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            a_scales_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            b_scales_ptrs=torch.empty(ne, device=device, dtype=torch.int64),
            stride_a=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            stride_b=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            stride_c=torch.empty(ne * stride_sz, device=device, dtype=torch.uint8),
            layout_sfa=torch.empty(ne * sfa_sz, device=device, dtype=torch.uint8),
            layout_sfb=torch.empty(ne * sfb_sz, device=device, dtype=torch.uint8),
            problem_sizes_1=torch.empty(ne, 3, device=device, dtype=torch.int32),
            problem_sizes_2=torch.empty(ne, 3, device=device, dtype=torch.int32),
            problem_sizes_transpose=torch.empty(ne, 3, device=device, dtype=torch.int32),
            workspace=torch.empty(
                ext.get_workspace_size(total_tokens, ne, 0, 0, False),
                device=device, dtype=torch.uint8),
            packed_acts=torch.empty(total_tokens, K1, device=device, dtype=torch.float8_e4m3fn),
            packed_act_scales=torch.empty(total_tokens, K1 // 128, device=device, dtype=torch.float32),
            gemm1_out=torch.empty(total_tokens, N1, device=device, dtype=torch.bfloat16),
            act_q=torch.empty(total_tokens, N1 // 2, device=device, dtype=torch.float8_e4m3fn),
            row_scales=torch.empty(total_tokens, device=device, dtype=torch.float32),
            act_scale_for_gemm2=torch.empty(
                total_tokens, K2 // 128, device=device, dtype=torch.float32),
            gemm2_out=torch.empty(total_tokens, N2, device=device, dtype=torch.bfloat16),
            out_bf16=torch.empty(T, N2, device=device, dtype=torch.bfloat16),
            # Buffers for fused_dispatch (sorted_tids/weights are T*TOP_K long)
            counts_buf=torch.empty(ne, device=device, dtype=torch.int32),
            offsets_buf=torch.empty(ne + 1, device=device, dtype=torch.int32),
            sorted_tids_buf=torch.empty(total_tokens, device=device, dtype=torch.int32),
            sorted_weights_buf=torch.empty(total_tokens, device=device, dtype=torch.float32),
            # Buffers for reduce_scatter (per-output-token bucket map)
            token_counts_buf=torch.empty(T, device=device, dtype=torch.int32),
            token_offsets_buf=torch.empty(T + 1, device=device, dtype=torch.int32),
            token_perm_buf=torch.empty(total_tokens, device=device, dtype=torch.int32),
            chunk_offsets_buf=torch.empty(ne, device=device, dtype=torch.int32),
            # Route outputs (pre-allocated so _route_fused emits zero torch ops
            # inside the captured CUDA graph — no torch.empty / implicit copies).
            topk_idx_buf=torch.empty(T, TOP_K, device=device, dtype=torch.int32),
            assign_w_buf=torch.empty(T, TOP_K, device=device, dtype=torch.float32),
        )
        # expert_offsets is a view into offsets_buf[:ne] — fused_dispatch
        # writes the exclusive scan there directly, so no extra cumsum needed.
        bufs["expert_offsets"] = bufs["offsets_buf"][:ne]
        # problem_sizes_1/2 N,K columns are fixed — fill once here.
        bufs["problem_sizes_1"][:, 1] = N1
        bufs["problem_sizes_1"][:, 2] = K1
        bufs["problem_sizes_2"][:, 1] = N2
        bufs["problem_sizes_2"][:, 2] = K2

        # MxF8 workspace is allocated only for workloads that use the MxF8 path.
        if use_mxf8:
            H = N1 // 2
            bufs["mxf8_gemm1_a_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_b_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_out_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_sfa_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm1_sfb_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_a_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_b_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_out_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_sfa_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_gemm2_sfb_ptrs"] = torch.empty(ne, device=device, dtype=torch.int64)
            bufs["mxf8_stride_a"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_stride_b"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_stride_c"] = torch.empty(ne * mxf8_stride_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfa_1"] = torch.empty(ne * mxf8_sfa_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfb_1"] = torch.empty(ne * mxf8_sfb_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfa_2"] = torch.empty(ne * mxf8_sfa_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_layout_sfb_2"] = torch.empty(ne * mxf8_sfb_sz, device=device, dtype=torch.uint8)
            bufs["mxf8_expert_offsets"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            # UE8M0 scale buffers for activations (per-call) and weights (one-time, cached below).
            bufs["mxf8_act_scales_ue8m0"] = torch.empty(
                total_tokens, K1 // 128, device=device, dtype=torch.float32)
            bufs["mxf8_gemm2_act_scales_ue8m0"] = torch.empty(
                total_tokens, K2 // 128, device=device, dtype=torch.float32)
            # Transcoded weight payloads (in place overwrite of a copy at init).
            bufs["mxf8_gemm1_w_tr"] = None  # filled at first call when weights seen
            bufs["mxf8_gemm1_w_sc_ue8m0"] = None
            bufs["mxf8_gemm2_w_tr"] = None
            bufs["mxf8_gemm2_w_sc_ue8m0"] = None
            bufs["mxf8_weights_ready"] = False
            # SFA/SFB packed UE8M0 buffers.
            # For SFA (activations): sum across experts is bounded by
            # total_tokens + ne*127 padding (each expert pads up to 128).
            # For SFB (weights): always ne * (per-expert size) since each expert
            # has fixed N.
            _K32_g1 = ((K1 + 31) // 32) * 4
            _K32_g2 = ((K2 + 31) // 32) * 4
            max_sfa_total_1 = (((total_tokens + ne * 128 + 127) // 128) * 128) * _K32_g1
            max_sfb_total_1 = ne * (((N1 + 127) // 128) * 128) * _K32_g1
            max_sfa_total_2 = (((total_tokens + ne * 128 + 127) // 128) * 128) * _K32_g2
            max_sfb_total_2 = ne * (((N2 + 127) // 128) * 128) * _K32_g2
            bufs["mxf8_sfa_buffer_1"] = torch.empty(max_sfa_total_1, device=device, dtype=torch.uint8)
            bufs["mxf8_sfb_buffer_1"] = torch.empty(max_sfb_total_1, device=device, dtype=torch.uint8)
            bufs["mxf8_sfa_buffer_2"] = torch.empty(max_sfa_total_2, device=device, dtype=torch.uint8)
            bufs["mxf8_sfb_buffer_2"] = torch.empty(max_sfb_total_2, device=device, dtype=torch.uint8)
            bufs["mxf8_sfa_byte_offsets_1"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfb_byte_offsets_1"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfa_byte_offsets_2"] = torch.empty(ne + 1, device=device, dtype=torch.int32)
            bufs["mxf8_sfb_byte_offsets_2"] = torch.empty(ne + 1, device=device, dtype=torch.int32)

        _workspace_cache[key] = bufs
    return _workspace_cache[key]


def _mxf8_ensure_weights_transcoded(bufs, gemm1_weights, gemm1_weights_scale,
                                    gemm2_weights, gemm2_weights_scale):
    """Transcode weights + pre-pack SFB layout once per *unique weight set*.

    Keyed on the 4 weight tensors' data_ptr()s so that running multiple
    workloads in the same Python process (workspace reused across workloads
    that share shape) doesn't incorrectly reuse transcoded weights from a
    previous workload. If any of the four weight pointers changes, we
    re-transcode.
    """
    weight_id = (
        int(gemm1_weights.data_ptr()),
        int(gemm1_weights_scale.data_ptr()),
        int(gemm2_weights.data_ptr()),
        int(gemm2_weights_scale.data_ptr()),
    )
    if bufs.get("mxf8_weights_ready") and bufs.get("mxf8_weight_id") == weight_id:
        return
    ext = _get_ext()
    bufs["mxf8_gemm1_w_tr"]       = gemm1_weights.clone()
    bufs["mxf8_gemm1_w_sc_ue8m0"] = torch.empty_like(gemm1_weights_scale)
    bufs["mxf8_gemm2_w_tr"]       = gemm2_weights.clone()
    bufs["mxf8_gemm2_w_sc_ue8m0"] = torch.empty_like(gemm2_weights_scale)
    ext.mxf8_transcode_weights_impl(
        bufs["mxf8_gemm1_w_tr"], gemm1_weights_scale, bufs["mxf8_gemm1_w_sc_ue8m0"])
    ext.mxf8_transcode_weights_impl(
        bufs["mxf8_gemm2_w_tr"], gemm2_weights_scale, bufs["mxf8_gemm2_w_sc_ue8m0"])

    # Pre-pack SFB into CUTLASS tiled layout (once, per unique weight set).
    N1 = int(gemm1_weights.shape[1]); K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1]); K2 = int(gemm2_weights.shape[2])
    ext.mxf8_pack_weight_sfb_impl(
        bufs["mxf8_gemm1_w_sc_ue8m0"],
        bufs["mxf8_layout_sfb_1"],
        bufs["mxf8_sfb_byte_offsets_1"],
        bufs["mxf8_sfb_buffer_1"],
        N1, K1)
    ext.mxf8_pack_weight_sfb_impl(
        bufs["mxf8_gemm2_w_sc_ue8m0"],
        bufs["mxf8_layout_sfb_2"],
        bufs["mxf8_sfb_byte_offsets_2"],
        bufs["mxf8_sfb_buffer_2"],
        N2, K2)
    bufs["mxf8_weights_ready"] = True
    bufs["mxf8_weight_id"] = weight_id




def _run_pipeline_graph_safe(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks,
    bufs, ext,
):
    """Fixed-shape pipeline (total=T*TOP_K) usable inside a CUDA graph.
    All writes go into pre-allocated buffers in `bufs`. No .item() syncs.
    """
    topk_idx, assign_w = _route(
        routing_logits, routing_bias, rsf, T, ls, ne,
        topk_idx_buf=bufs["topk_idx_buf"], assign_w_buf=bufs["assign_w_buf"])

    counts, sorted_tids, sorted_weights = _dispatch_graph_safe(topk_idx, assign_w, T, ls, ne, bufs)
    ext.fused_gather_hidden_scales(
        hidden_states, hidden_states_scale, sorted_tids,
        bufs["packed_acts"], bufs["packed_act_scales"])

    ext.moe_blockwise_grouped_mm_v2(
        bufs["gemm1_out"],
        bufs["packed_acts"], gemm1_weights, bufs["packed_act_scales"], gemm1_weights_scale,
        bufs["expert_offsets"], bufs["problem_sizes_1"],
        bufs["problem_sizes_transpose"],
        bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
        bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
        bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
        bufs["layout_sfa"], bufs["layout_sfb"],
        bufs["workspace"],
    )

    ext.swiglu_fp8_requant(
        bufs["gemm1_out"], bufs["act_q"],
        bufs["row_scales"], bufs["act_scale_for_gemm2"])

    ext.moe_blockwise_grouped_mm_v2(
        bufs["gemm2_out"],
        bufs["act_q"], gemm2_weights, bufs["act_scale_for_gemm2"], gemm2_weights_scale,
        bufs["expert_offsets"], bufs["problem_sizes_2"],
        bufs["problem_sizes_transpose"],
        bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
        bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
        bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
        bufs["layout_sfa"], bufs["layout_sfb"],
        bufs["workspace"],
    )

    bufs["out_bf16"].zero_()
    ext.weighted_scatter(
        bufs["gemm2_out"], sorted_weights, sorted_tids, bufs["out_bf16"], T)


def _run_pipeline_dynamic(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    T, ne, N1, K1, N2, K2, H, ls, rsf,
    bufs, ext, device,
):
    """Dynamic-shape pipeline for large T.

    This path compacts to local expert rows before the MLP, avoiding the fixed
    T*TOP_K work used by the graph-safe path. It keeps only the submitted
    default launch sequence: dispatch, gather/transcode, GEMM1, fused SwiGLU +
    requant, GEMM2, and fused unweighted reduce-scatter.
    """
    topk_idx, assign_w = _route(
        routing_logits, routing_bias, rsf, T, ls, ne,
        topk_idx_buf=bufs["topk_idx_buf"], assign_w_buf=bufs["assign_w_buf"])
    hs_scale = hidden_states_scale

    counts, sorted_tids, sorted_weights = _dispatch_dynamic(topk_idx, assign_w, T, ls, ne, bufs)
    total_valid = sorted_tids.shape[0]
    expert_offsets = bufs["offsets_buf"][:ne]

    use_mxf8 = bool(int(os.environ.get("USE_MXF8", "1"))) and T >= int(
        os.environ.get("MXF8_MIN_T", "4096"))

    packed_acts = bufs["packed_acts"][:total_valid]
    packed_act_scales = bufs["packed_act_scales"][:total_valid]

    if use_mxf8:
        _mxf8_ensure_weights_transcoded(
            bufs, gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
        )
        mxf8_act_scales_ue8m0 = bufs["mxf8_act_scales_ue8m0"][:total_valid]
        ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_1"],
            bufs["mxf8_sfa_byte_offsets_1"],
            bufs["mxf8_sfb_byte_offsets_1"])
        ext.moe_mxf8_setup_ptrs(
            bufs["gemm1_out"], packed_acts, bufs["mxf8_gemm1_w_tr"],
            bufs["offsets_buf"], bufs["problem_sizes_1"],
            bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
            bufs["mxf8_gemm1_out_ptrs"],
            bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
            bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
            bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"])
        ext.fused_gather_mxf8(
            hidden_states, hs_scale, sorted_tids,
            packed_acts, mxf8_act_scales_ue8m0,
            bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_1"],
            bufs["mxf8_layout_sfa_1"], bufs["mxf8_sfa_buffer_1"])
    else:
        ext.fused_gather_hidden_scales(
            hidden_states, hs_scale, sorted_tids,
            packed_acts, packed_act_scales)

    gemm1_out = bufs["gemm1_out"][:total_valid]
    if use_mxf8:
        ext.moe_mxf8_grouped_mm_prepacked(
            gemm1_out,
            packed_acts, bufs["mxf8_gemm1_w_tr"],
            bufs["problem_sizes_1"],
            bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
            bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
            bufs["workspace"],
        )
    else:
        ext.moe_blockwise_grouped_mm_v2(
            gemm1_out,
            packed_acts, gemm1_weights, packed_act_scales, gemm1_weights_scale,
            expert_offsets, bufs["problem_sizes_1"],
            bufs["problem_sizes_transpose"],
            bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
            bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
            bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
            bufs["layout_sfa"], bufs["layout_sfb"],
            bufs["workspace"],
        )

    act_q = bufs["act_q"][:total_valid]
    row_scales = bufs["row_scales"][:total_valid]
    act_scale_for_gemm2 = bufs["act_scale_for_gemm2"][:total_valid]
    if use_mxf8:
        mxf8_gemm2_act_scales_ue8m0 = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]
        ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_2"],
            bufs["mxf8_sfa_byte_offsets_2"],
            bufs["mxf8_sfb_byte_offsets_2"])
        ext.moe_mxf8_setup_ptrs(
            bufs["gemm2_out"], act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["offsets_buf"], bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"],
            bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
            bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"])
        ext.swiglu_fp8_requant_weighted_mxf8(
            gemm1_out, sorted_weights, act_q, row_scales,
            mxf8_gemm2_act_scales_ue8m0,
            bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_2"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_sfa_buffer_2"])
    else:
        ext.swiglu_fp8_requant_weighted(
            gemm1_out, sorted_weights, act_q, row_scales, act_scale_for_gemm2)

    gemm2_out = bufs["gemm2_out"][:total_valid]
    if use_mxf8:
        ext.moe_mxf8_grouped_mm_prepacked(
            gemm2_out,
            act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["workspace"],
        )
    else:
        ext.moe_blockwise_grouped_mm_v2(
            gemm2_out,
            act_q, gemm2_weights, act_scale_for_gemm2, gemm2_weights_scale,
            expert_offsets, bufs["problem_sizes_2"],
            bufs["problem_sizes_transpose"],
            bufs["a_ptrs"], bufs["b_ptrs"], bufs["out_ptrs"],
            bufs["a_scales_ptrs"], bufs["b_scales_ptrs"],
            bufs["stride_a"], bufs["stride_b"], bufs["stride_c"],
            bufs["layout_sfa"], bufs["layout_sfb"],
            bufs["workspace"],
        )

    ext.reduce_scatter_unweighted_fused(
        gemm2_out, sorted_tids, bufs["out_bf16"],
        bufs["token_counts_buf"], bufs["token_perm_buf"],
        T, TOP_K)
    return bufs["out_bf16"]


@torch.no_grad()
def custom_kernel(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor,
):
    ext = _get_ext()

    T = int(routing_logits.shape[0])
    device = hidden_states.device
    ne = int(gemm1_weights.shape[0])
    N1 = int(gemm1_weights.shape[1])
    K1 = int(gemm1_weights.shape[2])
    N2 = int(gemm2_weights.shape[1])
    K2 = int(gemm2_weights.shape[2])
    H = N1 // 2
    ls = int(local_expert_offset)
    rsf = float(routed_scaling_factor)

    total_tokens = T * TOP_K
    k2_blocks = K2 // 128
    bufs = _get_workspace(device, ne, T, N1, K1, N2, K2)

    # Below threshold: Python overhead dominates → CUDA graph replay wins.
    # Above threshold: GEMM compute dominates, non-local filtering saves ~8x
    # data movement vs fixed-shape path.
    # Graph-safe path (fixed shape T*TOP_K) wins for small-medium T due to
    # graph-replay overhead elimination. Dynamic path (only num_local_valid
    # tokens, ~T) wins for large T where graph-safe's extra 8x gather+scatter
    # work on T*TOP_K dominates. Verified empirical crossover ~T=2048.
    use_graph = (T <= 2048) and not os.environ.get("DISABLE_CUDA_GRAPH")

    if not use_graph:
        return _run_pipeline_dynamic(
            routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            T, ne, N1, K1, N2, K2, H, ls, rsf, bufs, ext, device)

    pkey = (
        T, device,
        int(routing_logits.data_ptr()), int(routing_bias.data_ptr()),
        int(hidden_states.data_ptr()), int(hidden_states_scale.data_ptr()),
        int(gemm1_weights.data_ptr()), int(gemm1_weights_scale.data_ptr()),
        int(gemm2_weights.data_ptr()), int(gemm2_weights_scale.data_ptr()),
    )

    if pkey not in _graph_cache:
        for _ in range(2):
            _run_pipeline_graph_safe(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks, bufs, ext)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _run_pipeline_graph_safe(
                routing_logits, routing_bias, hidden_states, hidden_states_scale,
                gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                T, ne, N1, K1, N2, K2, H, ls, rsf, total_tokens, k2_blocks, bufs, ext)
        _graph_cache[pkey] = g

    _graph_cache[pkey].replay()
    return bufs["out_bf16"]


kernel = custom_kernel


def run(routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
        local_expert_offset, routed_scaling_factor):
    return custom_kernel(routing_logits, routing_bias, hidden_states,
                         hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                         gemm2_weights, gemm2_weights_scale,
                         local_expert_offset, routed_scaling_factor)
