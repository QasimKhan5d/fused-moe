"""Persistent Modal bench server — keeps one warm B200 container that:
  - Loads all 19 workloads into GPU memory once at startup (~30s)
  - Builds the kernel extension once at startup (~20s-3min)
  - Exposes hot methods for rapid iteration: bench(kernel_src, mode, ...) -> dict

Client iteration cost drops from ~90s/iter to ~10-15s/iter.

Usage:
  Terminal 1 (once per session):
      modal serve experiments/fast_bench/bench_server.py

  Terminal 2 (any number of times, fast):
      python experiments/fast_bench/bench_client.py
      python experiments/fast_bench/bench_client.py --quick
      python experiments/fast_bench/bench_client.py --uuids 5e8dc11c,58a34f27
"""
from __future__ import annotations

import modal

app = modal.App("fused-moe-bench-server")
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

DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
BASELINE_CACHE_PATH = "/mnt/bench_cache/baseline.pt"
SOLUTION_PY_PATH = "/root/solution/kernel.py"


@app.cls(
    image=image,
    gpu="B200:1",
    volumes={"/mnt": trace_volume},
    container_idle_timeout=3600,
    timeout=3600,
    max_containers=1,
    min_containers=1,   # keep at least one warm
    buffer_containers=0,
)
class BenchServer:
    """One long-running container. Loads inputs + builds once; benches many."""

    @modal.enter()
    def on_start(self):
        """Boot-time setup: import torch, load all workloads, prebuild extension."""
        import os, sys, time
        from pathlib import Path
        import torch

        # Deterministic inputs across container boots: seed torch before
        # gen_inputs. This ensures baseline and variant runs across different
        # container invocations see the same random inputs for each workload.
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        sys.path.insert(0, "/root/solution")
        os.chdir("/root/solution")

        t0 = time.time()
        from flashinfer_bench import TraceSet
        from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

        trace_set = TraceSet.from_path("/mnt/mlsys26-contest")
        definition = trace_set.definitions[DEF_NAME]
        all_wls = [getattr(wl, "workload", wl)
                   for wl in trace_set.workloads.get(DEF_NAME, [])]
        all_wls.sort(key=lambda w: w.axes.get("seq_len", 0))

        self.inputs_by_uuid = {}  # uuid_prefix8 -> (T, inputs_tuple)
        for w in all_wls:
            u = getattr(w, "uuid", "")[:8]
            # Reset seed before each gen_inputs so per-uuid inputs are
            # deterministic regardless of ordering.
            torch.manual_seed(hash(u) & 0x7fff_ffff)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(hash(u) & 0x7fff_ffff)
            loaded_st = load_safetensors(
                definition, w, Path("/mnt/mlsys26-contest")) if any(
                d.type == "safetensors" for d in getattr(w, "inputs", {}).values()) else {}
            inputs = gen_inputs(definition, w, device="cuda", safe_tensors=loaded_st)
            T = int(inputs[0].shape[0])
            self.inputs_by_uuid[u] = (T, inputs)
        print(f"[boot] loaded {len(self.inputs_by_uuid)} workloads in {time.time()-t0:.1f}s",
              flush=True)

        # Prebuild extension with the bootstrapped kernel.py from add_local_dir.
        t0 = time.time()
        import kernel as K
        _ = K._get_ext()
        self._K_module_name = "kernel"
        print(f"[boot] initial extension build {time.time()-t0:.1f}s", flush=True)

        os.makedirs("/mnt/bench_cache", exist_ok=True)

    def _reload_kernel(self):
        """Flush module cache and re-import kernel. Torch JIT will detect source
        changes in the .cu files written by _get_ext() and rebuild only what
        changed (~15-20s for fused.cu only, ~2-3min for gemm.cu)."""
        import sys
        for k in list(sys.modules):
            if k == "kernel" or k.startswith("kernel."):
                del sys.modules[k]
        import kernel as K
        return K

    def _run_bench(self, K, uuids, warmup, iters, env):
        import os, time, traceback
        import torch

        # Apply env overrides.
        for k, v in env.items():
            os.environ[str(k)] = str(v)

        # Ensure extension is built with new source.
        _ = K._get_ext()
        # Clear per-shape workspace cache so buffers resize based on USE_MXF8 flag.
        if hasattr(K, "_workspace_cache"):
            K._workspace_cache.clear()

        if uuids:
            uuid_list = [u.strip()[:8] for u in (
                uuids.split(",") if isinstance(uuids, str) else uuids)]
        else:
            uuid_list = list(self.inputs_by_uuid.keys())

        results = {}
        for u in uuid_list:
            if u not in self.inputs_by_uuid:
                results[u] = {"error": "NOT_FOUND"}
                continue
            T, inputs = self.inputs_by_uuid[u]
            try:
                for _ in range(max(1, warmup)):
                    out_t = K.custom_kernel(*inputs)
                torch.cuda.synchronize()
            except Exception as exc:
                results[u] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}",
                              "T": T, "traceback": traceback.format_exc()[-600:]}
                continue
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            for _ in range(iters):
                out_t = K.custom_kernel(*inputs)
            e.record()
            torch.cuda.synchronize()
            lat_ms = s.elapsed_time(e) / iters
            # Keep output on GPU for baseline save / compare; avoids expensive
            # CPU round-trip + Modal gRPC bandwidth cap (was ~60MB/wl × 19).
            results[u] = {"T": T, "lat_ms": lat_ms,
                          "out_gpu": out_t.detach(), "error": None}
        return results

    @modal.method()
    def bench(self,
              kernel_src: str | None = None,
              uuids: str | None = None,
              warmup: int = 3,
              iters: int = 20,
              env: dict | None = None,
              save_baseline: bool = False,
              compare_to_baseline: bool = True,
              label: str = "variant") -> dict:
        """Run a benchmark pass.

        If kernel_src is given, it is written to /root/solution/kernel.py and
        the module is re-imported, triggering a torch JIT rebuild of any changed
        .cu files.
        """
        import os, time
        import torch

        env = env or {}

        t0 = time.time()
        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        t_reload = time.time() - t0

        t0 = time.time()
        results = self._run_bench(K, uuids, warmup, iters, env)
        t_bench = time.time() - t0

        if save_baseline:
            # Pickle the GPU tensor .cpu() copies ONCE to the volume. This is
            # a one-time cost; subsequent variants don't pay it.
            import pickle
            to_save = {}
            for u, r in results.items():
                if r.get("error") is None:
                    to_save[u] = {"T": r["T"], "lat_ms": r["lat_ms"],
                                   "out_cpu": r["out_gpu"].to("cpu", dtype=torch.float32).numpy()}
            with open(BASELINE_CACHE_PATH, "wb") as f:
                pickle.dump(to_save, f)

        # Compare-on-server: load baseline outputs (from volume, not RPC),
        # move each to GPU once, compare against variant output GPU-side, free.
        compare = {}
        if compare_to_baseline and os.path.exists(BASELINE_CACHE_PATH):
            import pickle
            with open(BASELINE_CACHE_PATH, "rb") as f:
                baseline_cache = pickle.load(f)
            for u, r in results.items():
                if r.get("error") or u not in baseline_cache:
                    continue
                b = baseline_cache[u]
                out_b = torch.from_numpy(b["out_cpu"]).to("cuda",
                                                           dtype=torch.float32)
                out_v = r["out_gpu"].to(torch.float32)
                if out_b.shape != out_v.shape:
                    compare[u] = {"b_lat_ms": b["lat_ms"], "matched": -1.0,
                                   "max_abs": -1.0}
                    continue
                abs_diff = (out_b - out_v).abs()
                tol = 1.0 + 0.3 * out_b.abs()
                matched = float((abs_diff <= tol).float().mean().item())
                max_abs = float(abs_diff.max().item())
                compare[u] = {"b_lat_ms": b["lat_ms"], "matched": matched,
                               "max_abs": max_abs}

        # Return ONLY scalar stats — no big tensors across RPC.
        return {
            "results": {u: {"T": r.get("T"), "lat_ms": r.get("lat_ms"),
                             "error": r.get("error")}
                        for u, r in results.items()},
            "compare": compare,
            "reload_s": t_reload, "bench_s": t_bench, "label": label,
        }

    @modal.method()
    def warm(self) -> str:
        return f"warm, {len(self.inputs_by_uuid)} workloads ready"

    @modal.method()
    def time_gemm_configs(self, kernel_src: str | None = None,
                           uuid: str = "5e8dc11c", iters: int = 100,
                           gemm: int = 1) -> dict:
        """Time a GEMM with multiple tile configs. Sweeps:
          - 256x128x128 2SM  (current default)
          - 128x128x128 1SM
          - 256x256x128 2SM
          - 128x256x128 1SM
        """
        import os, torch
        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        ext = K._get_ext()
        os.environ["USE_MXF8"] = "1"; os.environ["MXF8_MIN_T"] = "4096"
        if hasattr(K, "_workspace_cache"):
            K._workspace_cache.clear()
        T, inputs = self.inputs_by_uuid[uuid]
        device = inputs[0].device
        ne = int(inputs[4].shape[0])
        N1 = int(inputs[4].shape[1]); K1 = int(inputs[4].shape[2])
        N2 = int(inputs[6].shape[1]); K2 = int(inputs[6].shape[2])
        for _ in range(3):
            _ = K.custom_kernel(*inputs)
        torch.cuda.synchronize()

        bufs = K._get_workspace(device, ne, T, N1, K1, N2, K2)
        total_valid = int(bufs["offsets_buf"][ne].item())
        if gemm == 1:
            a = bufs["packed_acts"][:total_valid]; w = bufs["mxf8_gemm1_w_tr"]
            out = bufs["gemm1_out"][:total_valid]; ps = bufs["problem_sizes_1"]
            a_ptrs = bufs["mxf8_gemm1_a_ptrs"]; b_ptrs = bufs["mxf8_gemm1_b_ptrs"]
            out_ptrs = bufs["mxf8_gemm1_out_ptrs"]
            sfa_ptrs = bufs["mxf8_gemm1_sfa_ptrs"]; sfb_ptrs = bufs["mxf8_gemm1_sfb_ptrs"]
            layout_sfa = bufs["mxf8_layout_sfa_1"]; layout_sfb = bufs["mxf8_layout_sfb_1"]
            sfa_off = bufs["mxf8_sfa_byte_offsets_1"]; sfb_off = bufs["mxf8_sfb_byte_offsets_1"]
            sfa_buf = bufs["mxf8_sfa_buffer_1"]; sfb_buf = bufs["mxf8_sfb_buffer_1"]
            N = N1; K = K1
        else:
            a = bufs["act_q"][:total_valid]; w = bufs["mxf8_gemm2_w_tr"]
            out = bufs["gemm2_out"][:total_valid]; ps = bufs["problem_sizes_2"]
            a_ptrs = bufs["mxf8_gemm2_a_ptrs"]; b_ptrs = bufs["mxf8_gemm2_b_ptrs"]
            out_ptrs = bufs["mxf8_gemm2_out_ptrs"]
            sfa_ptrs = bufs["mxf8_gemm2_sfa_ptrs"]; sfb_ptrs = bufs["mxf8_gemm2_sfb_ptrs"]
            layout_sfa = bufs["mxf8_layout_sfa_2"]; layout_sfb = bufs["mxf8_layout_sfb_2"]
            sfa_off = bufs["mxf8_sfa_byte_offsets_2"]; sfb_off = bufs["mxf8_sfb_byte_offsets_2"]
            sfa_buf = bufs["mxf8_sfa_buffer_2"]; sfb_buf = bufs["mxf8_sfb_buffer_2"]
            N = N2; K = K2

        ext.compute_mxf8_sf_offsets_device(ps, sfa_off, sfb_off)
        ext.moe_mxf8_setup_ptrs(
            out, a, w, bufs["offsets_buf"], ps,
            a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            layout_sfa, layout_sfb, sfa_buf, sfb_buf, sfa_off, sfb_off)

        def run(fn):
            fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / iters

        results = {}
        outputs = {}
        fns = {
            "256_128_2sm_current": lambda: ext.moe_mxf8_grouped_mm_prepacked(
                out, a, w, ps, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                layout_sfa, layout_sfb, bufs["workspace"]),
            "128_128_1sm": lambda: ext.moe_mxf8_grouped_mm_prepacked_1sm(
                out, a, w, ps, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                layout_sfa, layout_sfb, bufs["workspace"]),
            "256_256_2sm": lambda: ext.moe_mxf8_grouped_mm_prepacked_256_256(
                out, a, w, ps, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                layout_sfa, layout_sfb, bufs["workspace"]),
            "128_256_1sm": lambda: ext.moe_mxf8_grouped_mm_prepacked_128_256_1sm(
                out, a, w, ps, a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                layout_sfa, layout_sfb, bufs["workspace"]),
        }
        # Reference: run current (256x128) and save output.
        fns["256_128_2sm_current"]()
        torch.cuda.synchronize()
        ref = out.clone()
        for name, fn in fns.items():
            out.zero_()
            try:
                fn()
                torch.cuda.synchronize()
                v = out.clone()
                if v.shape == ref.shape:
                    diff = (v.to(torch.float32) - ref.to(torch.float32)).abs()
                    tol = 1.0 + 0.3 * ref.to(torch.float32).abs()
                    matched = float((diff <= tol).float().mean().item())
                else:
                    matched = -1.0
                results[name] = run(fn)
                outputs[name] = {"match%": matched * 100,
                                  "max_abs": float((v.to(torch.float32) - ref.to(torch.float32)).abs().max().item())}
            except Exception as exc:
                results[name] = f"ERR: {type(exc).__name__}: {str(exc)[:80]}"
                outputs[name] = {"err": str(exc)[:120]}
        gflops = 2 * total_valid * N * K / 1e9
        return {"gemm": gemm, "gflops": gflops,
                 "total_valid": total_valid, "N": N, "K": K,
                 "us": results, "correctness": outputs}

    @modal.method()
    def time_gemm2_configs(self, kernel_src: str | None = None,
                            uuid: str = "5e8dc11c", iters: int = 100) -> dict:
        """Time GEMM2 with multiple CUTLASS tile configs to find the best.
        GEMM2 shape: M_e ≈ 500, N=7168, K=2048 per expert.
        Currently at 62% of peak (337 μs for 473 GFLOPs), so 1SM variants
        or smaller tiles might win.
        """
        import os, torch
        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        ext = K._get_ext()
        os.environ["USE_MXF8"] = "1"
        os.environ["MXF8_MIN_T"] = "4096"
        if hasattr(K, "_workspace_cache"):
            K._workspace_cache.clear()

        T, inputs = self.inputs_by_uuid[uuid]
        device = inputs[0].device
        ne = int(inputs[4].shape[0])
        N1 = int(inputs[4].shape[1]); K1 = int(inputs[4].shape[2])
        N2 = int(inputs[6].shape[1]); K2 = int(inputs[6].shape[2])

        for _ in range(3):
            _ = K.custom_kernel(*inputs)
        torch.cuda.synchronize()

        bufs = K._get_workspace(device, ne, T, N1, K1, N2, K2)
        total_valid = int(bufs["offsets_buf"][ne].item())
        act_q = bufs["act_q"][:total_valid]
        gemm2_out = bufs["gemm2_out"][:total_valid]

        # Pre-populate GEMM2 setup with CfgMxF8Large (2SM).
        ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_2"],
            bufs["mxf8_sfa_byte_offsets_2"],
            bufs["mxf8_sfb_byte_offsets_2"])
        ext.moe_mxf8_setup_ptrs(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["offsets_buf"], bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"],
            bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
            bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"])

        def t(fn):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / iters

        us_2sm = t(lambda: ext.moe_mxf8_grouped_mm_prepacked(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["workspace"],
        ))
        us_1sm = t(lambda: ext.moe_mxf8_grouped_mm_prepacked_1sm(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["workspace"],
        ))
        us_256_256 = t(lambda: ext.moe_mxf8_grouped_mm_prepacked_256_256(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["workspace"],
        ))
        us_128_256_1sm = t(lambda: ext.moe_mxf8_grouped_mm_prepacked_128_256_1sm(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["workspace"],
        ))
        gflops = 2 * total_valid * N2 * K2 / 1e9
        peak_fp8 = 2250.0  # B200 dense FP8 peak TFLOPS
        return {
            "us_256_128_2sm (current)": us_2sm,
            "us_128_128_1sm": us_1sm,
            "us_256_256_2sm": us_256_256,
            "us_128_256_1sm": us_128_256_1sm,
            "gflops": gflops,
        }

    @modal.method()
    def per_stage_timing(self, kernel_src: str | None = None,
                          uuid: str = "5e8dc11c", iters: int = 100) -> dict:
        """Measure EACH stage of the MxF8 pipeline on real inputs.

        Instrument the dynamic pipeline manually (replicating _run_pipeline_dynamic
        step by step with CUDA events between) to understand where the 1.2 ms
        goes.
        """
        import os, torch
        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        ext = K._get_ext()
        os.environ["USE_MXF8"] = "1"
        os.environ["MXF8_MIN_T"] = "4096"
        if hasattr(K, "_workspace_cache"):
            K._workspace_cache.clear()

        T, inputs = self.inputs_by_uuid[uuid]
        device = inputs[0].device

        # Warmup.
        for _ in range(3):
            _ = K.custom_kernel(*inputs)
        torch.cuda.synchronize()

        # Pull canonical shapes.
        ne = int(inputs[4].shape[0])
        N1 = int(inputs[4].shape[1]); K1 = int(inputs[4].shape[2])
        N2 = int(inputs[6].shape[1]); K2 = int(inputs[6].shape[2])
        H  = N1 // 2
        ls = int(inputs[8]); rsf = float(inputs[9])

        bufs = K._get_workspace(device, ne, T, N1, K1, N2, K2)
        K._mxf8_ensure_weights_transcoded(
            bufs, inputs[4], inputs[5], inputs[6], inputs[7])

        stages = []
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)

        def time_fn(name, fn, iters=iters):
            torch.cuda.synchronize()
            ev_s.record()
            for _ in range(iters):
                fn()
            ev_e.record()
            torch.cuda.synchronize()
            stages.append((name, ev_s.elapsed_time(ev_e) * 1000.0 / iters))

        # ---- Set up a full pipeline state via one forward ----
        routing_logits, routing_bias, hidden_states, hs_scale = inputs[0], inputs[1], inputs[2], inputs[3]
        gemm1_weights, gemm1_weights_scale = inputs[4], inputs[5]
        gemm2_weights, gemm2_weights_scale = inputs[6], inputs[7]

        topk_idx, assign_w = K._route(routing_logits, routing_bias, rsf, T, ls, ne)
        counts, sorted_tids, sorted_weights = K._dispatch_dynamic(
            topk_idx, assign_w, T, ls, ne, bufs)
        total_valid = sorted_tids.shape[0]

        packed_acts = bufs["packed_acts"][:total_valid]
        packed_act_scales = bufs["packed_act_scales"][:total_valid]
        mxf8_act_scales_ue8m0 = bufs["mxf8_act_scales_ue8m0"][:total_valid]
        gemm1_out = bufs["gemm1_out"][:total_valid]

        # Per-stage timing of each kernel.
        time_fn("route",
            lambda: K._route(routing_logits, routing_bias, rsf, T, ls, ne))

        time_fn("dispatch",
            lambda: K._dispatch_dynamic(topk_idx, assign_w, T, ls, ne, bufs))

        time_fn("compute_offsets",
            lambda: ext.compute_mxf8_sf_offsets_device(
                bufs["problem_sizes_1"],
                bufs["mxf8_sfa_byte_offsets_1"],
                bufs["mxf8_sfb_byte_offsets_1"]))

        time_fn("setup_ptrs",
            lambda: ext.moe_mxf8_setup_ptrs(
                gemm1_out, packed_acts, bufs["mxf8_gemm1_w_tr"],
                bufs["offsets_buf"], bufs["problem_sizes_1"],
                bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
                bufs["mxf8_gemm1_out_ptrs"],
                bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
                bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"]))

        time_fn("fused_gather_mxf8",
            lambda: ext.fused_gather_mxf8(
                hidden_states, hs_scale, sorted_tids,
                packed_acts, mxf8_act_scales_ue8m0,
                bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_1"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_sfa_buffer_1"]))

        time_fn("gemm1_mxf8",
            lambda: ext.moe_mxf8_grouped_mm_prepacked(
                gemm1_out, packed_acts, bufs["mxf8_gemm1_w_tr"],
                bufs["problem_sizes_1"],
                bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"], bufs["mxf8_gemm1_out_ptrs"],
                bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                bufs["workspace"]))

        act_q = bufs["act_q"][:total_valid]
        row_scales = bufs["row_scales"][:total_valid]
        mxf8_gemm2_act_scales_ue8m0 = bufs["mxf8_gemm2_act_scales_ue8m0"][:total_valid]
        gemm2_out = bufs["gemm2_out"][:total_valid]

        # Pre-populate GEMM2 state before timing.
        ext.compute_mxf8_sf_offsets_device(
            bufs["problem_sizes_2"],
            bufs["mxf8_sfa_byte_offsets_2"],
            bufs["mxf8_sfb_byte_offsets_2"])
        ext.moe_mxf8_setup_ptrs(
            gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
            bufs["offsets_buf"], bufs["problem_sizes_2"],
            bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"],
            bufs["mxf8_gemm2_out_ptrs"],
            bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
            bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
            bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
            bufs["mxf8_sfa_buffer_2"], bufs["mxf8_sfb_buffer_2"],
            bufs["mxf8_sfa_byte_offsets_2"], bufs["mxf8_sfb_byte_offsets_2"])

        time_fn("swiglu_mxf8_fused",
            lambda: ext.swiglu_fp8_requant_weighted_mxf8(
                gemm1_out, sorted_weights, act_q, row_scales,
                mxf8_gemm2_act_scales_ue8m0,
                bufs["offsets_buf"], bufs["mxf8_sfa_byte_offsets_2"],
                bufs["mxf8_layout_sfa_2"], bufs["mxf8_sfa_buffer_2"]))

        time_fn("gemm2_mxf8",
            lambda: ext.moe_mxf8_grouped_mm_prepacked(
                gemm2_out, act_q, bufs["mxf8_gemm2_w_tr"],
                bufs["problem_sizes_2"],
                bufs["mxf8_gemm2_a_ptrs"], bufs["mxf8_gemm2_b_ptrs"], bufs["mxf8_gemm2_out_ptrs"],
                bufs["mxf8_gemm2_sfa_ptrs"], bufs["mxf8_gemm2_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_2"], bufs["mxf8_layout_sfb_2"],
                bufs["workspace"]))

        time_fn("reduce_scatter_4kern",
            lambda: ext.reduce_scatter_unweighted(
                gemm2_out, sorted_tids, bufs["out_bf16"],
                bufs["token_counts_buf"], bufs["token_offsets_buf"],
                bufs["token_perm_buf"], T))
        time_fn("reduce_scatter_fused2",
            lambda: ext.reduce_scatter_unweighted_fused(
                gemm2_out, sorted_tids, bufs["out_bf16"],
                bufs["token_counts_buf"], bufs["token_perm_buf"], T, 8))

        total = sum(s for _, s in stages)
        return {
            "uuid": uuid, "T": T, "total_valid": total_valid,
            "stages_us": dict(stages),
            "sum_us": total,
        }

    @modal.method()
    def time_fp8out_vs_bf16out(self, kernel_src: str | None = None,
                                iters: int = 100) -> dict:
        """Time GEMM1 alone with bf16 output vs fp8+SFD output, using
        REALISTIC T=14107 problem sizes (post-expert filtering).
        """
        import torch, time
        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        ext = K._get_ext()

        # Use REAL T=14107 workload shape (after dispatch).
        uuid = "5e8dc11c"
        T, inputs = self.inputs_by_uuid[uuid]
        device = inputs[0].device
        N1 = int(inputs[4].shape[1])  # 4096
        K1 = int(inputs[4].shape[2])  # 7168
        ne = int(inputs[4].shape[0])  # 32

        # Warm up the full pipeline to populate workspace.
        import os
        os.environ["USE_MXF8"] = "1"
        os.environ["MXF8_MIN_T"] = "4096"
        if hasattr(K, "_workspace_cache"):
            K._workspace_cache.clear()
        for _ in range(3):
            _ = K.custom_kernel(*inputs)
        torch.cuda.synchronize()

        # Pull workspace and grab the mxf8-ready GEMM1 inputs.
        bufs = K._get_workspace(device, ne, T, N1, K1,
                                 int(inputs[6].shape[1]),
                                 int(inputs[6].shape[2]))
        total_valid = int(bufs["offsets_buf"][ne].item())
        packed_acts = bufs["packed_acts"][:total_valid]
        gemm1_out_bf16 = bufs["gemm1_out"][:total_valid]

        def t(fn, iters=iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0 / iters  # μs

        # Existing bf16-out GEMM1.
        def run_bf16():
            ext.moe_mxf8_grouped_mm_prepacked(
                gemm1_out_bf16,
                packed_acts, bufs["mxf8_gemm1_w_tr"],
                bufs["problem_sizes_1"],
                bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
                bufs["mxf8_gemm1_out_ptrs"],
                bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], bufs["mxf8_stride_c"],
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"],
                bufs["workspace"],
            )

        bf16_us = t(run_bf16)

        # New fp8+SFD output GEMM1.
        # Allocate the new buffers.
        stride_fp8_sz = int(ext.get_mxf8_fp8out_sizes_stride())
        sfd_sz        = int(ext.get_mxf8_fp8out_sizes_layout_sfd())

        gemm1_out_fp8 = torch.empty_like(packed_acts[:, :N1]).resize_(total_valid, N1).to(torch.float8_e4m3fn) \
            if False else torch.empty(total_valid, N1, device=device, dtype=torch.float8_e4m3fn)
        sfd_offsets = torch.zeros(ne + 1, device=device, dtype=torch.int32)
        ext.compute_mxf8_sfd_offsets_device(bufs["problem_sizes_1"], sfd_offsets)
        sfd_total = int(sfd_offsets[-1].item())
        sfd_buf = torch.empty(sfd_total, device=device, dtype=torch.uint8)
        d_ptrs = torch.empty(ne, device=device, dtype=torch.int64)
        sfd_ptrs = torch.empty(ne, device=device, dtype=torch.int64)
        stride_d_buf = torch.empty(ne * stride_fp8_sz, device=device, dtype=torch.uint8)
        layout_sfd = torch.empty(ne * sfd_sz, device=device, dtype=torch.uint8)

        def run_fp8():
            ext.moe_mxf8_grouped_mm_prepacked_fp8out(
                gemm1_out_fp8, sfd_buf, sfd_offsets,
                packed_acts, bufs["mxf8_gemm1_w_tr"],
                bufs["problem_sizes_1"], bufs["offsets_buf"],
                bufs["mxf8_gemm1_a_ptrs"], bufs["mxf8_gemm1_b_ptrs"],
                d_ptrs, sfd_ptrs,
                bufs["mxf8_gemm1_sfa_ptrs"], bufs["mxf8_gemm1_sfb_ptrs"],
                bufs["mxf8_sfa_byte_offsets_1"], bufs["mxf8_sfb_byte_offsets_1"],
                bufs["mxf8_sfa_buffer_1"], bufs["mxf8_sfb_buffer_1"],
                bufs["mxf8_stride_a"], bufs["mxf8_stride_b"], stride_d_buf,
                bufs["mxf8_layout_sfa_1"], bufs["mxf8_layout_sfb_1"], layout_sfd,
                bufs["workspace"],
            )

        # One warmup to hit JIT + CUTLASS cache.
        run_fp8()
        torch.cuda.synchronize()
        fp8_us = t(run_fp8)

        return {
            "uuid": uuid, "T": T, "total_valid": total_valid,
            "bf16_out_gemm1_us": bf16_us,
            "fp8_out_gemm1_us": fp8_us,
            "delta_us": bf16_us - fp8_us,
            "delta_pct": (bf16_us - fp8_us) / bf16_us * 100.0,
            "sfd_total_bytes": sfd_total,
        }

    @modal.method()
    def test_fp8_out_gemm(self, kernel_src: str | None = None, debug: bool = False) -> dict:
        import os
        if debug:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        """Isolation test for the new FP8-output MxF8 GEMM.

        Runs a small grouped GEMM (E=4 experts, M_e=128 each, N=4096, K=7168),
        both with FP8-out and with the existing bf16-out kernel, dequantizes
        the FP8-out result via the emitted UE8M0 scales, and compares to the
        bf16-out result. Pass criterion: ≥ 99% match on elements (values match
        within bf16 rounding + fp8 block-scale quantization).
        """
        import os, sys, time
        import torch

        if kernel_src is not None:
            with open(SOLUTION_PY_PATH, "w", encoding="utf-8") as f:
                f.write(kernel_src)
        K = self._reload_kernel()
        ext = K._get_ext()

        device = "cuda"
        E, M_e, N, K_ = 4, 128, 4096, 7168
        total_m = E * M_e
        # Deterministic synthetic inputs.
        torch.manual_seed(42)
        # FP8 activations in [-1, 1] payload range.
        a_bf16 = (torch.randn(total_m, K_, device=device) * 0.1).clamp(-1, 1)
        a_fp8  = a_bf16.to(torch.float8_e4m3fn)
        b_bf16 = (torch.randn(E, N, K_, device=device) * 0.1).clamp(-1, 1)
        b_fp8  = b_bf16.to(torch.float8_e4m3fn)

        # UE8M0 scales = 1.0 (power-of-2 0 byte 127) for simplicity.
        sfa_fp32 = torch.ones(total_m, K_ // 128, device=device)
        sfb_fp32 = torch.ones(E, N // 128, K_ // 128, device=device)

        expert_offsets = torch.tensor(
            [i * M_e for i in range(E + 1)], device=device, dtype=torch.int32)
        problem_sizes = torch.tensor(
            [[M_e, N, K_] for _ in range(E)], device=device, dtype=torch.int32)

        # Buffer sizes.
        stride_sz = int(ext.get_mxf8_sizes_stride())
        sfa_sz    = int(ext.get_mxf8_sizes_layout_sfa())
        sfb_sz    = int(ext.get_mxf8_sizes_layout_sfb())
        stride_fp8_sz = int(ext.get_mxf8_fp8out_sizes_stride())
        sfd_sz        = int(ext.get_mxf8_fp8out_sizes_layout_sfd())

        def make_ptrs():
            return torch.empty(E, device=device, dtype=torch.int64)
        def make_stride(sz):
            return torch.empty(E * sz, device=device, dtype=torch.uint8)
        def make_layout(sz):
            return torch.empty(E * sz, device=device, dtype=torch.uint8)

        workspace = torch.empty(128 << 20, device=device, dtype=torch.uint8)

        # ---- bf16 reference path ----
        out_bf16 = torch.zeros(total_m, N, device=device, dtype=torch.bfloat16)
        sfa_offsets_bf16 = torch.zeros(E + 1, device=device, dtype=torch.int32)
        sfb_offsets_bf16 = torch.zeros(E + 1, device=device, dtype=torch.int32)
        ext.compute_mxf8_sf_offsets_device(problem_sizes, sfa_offsets_bf16, sfb_offsets_bf16)
        # Allocate SFA/SFB flat buffers sized by offsets[E].
        sfa_total_bf16 = int(sfa_offsets_bf16[-1].item())
        sfb_total_bf16 = int(sfb_offsets_bf16[-1].item())
        sfa_buf_bf16 = torch.empty(sfa_total_bf16, device=device, dtype=torch.uint8)
        sfb_buf_bf16 = torch.empty(sfb_total_bf16, device=device, dtype=torch.uint8)

        a_ptrs = make_ptrs(); b_ptrs = make_ptrs(); out_ptrs = make_ptrs()
        sfa_ptrs = make_ptrs(); sfb_ptrs = make_ptrs()
        stride_a = make_stride(stride_sz); stride_b = make_stride(stride_sz); stride_c = make_stride(stride_sz)
        layout_sfa = make_layout(sfa_sz); layout_sfb = make_layout(sfb_sz)

        ext.moe_mxf8_setup_ptrs(
            out_bf16, a_fp8, b_fp8,
            expert_offsets, problem_sizes,
            a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
            stride_a, stride_b, stride_c, layout_sfa, layout_sfb,
            sfa_buf_bf16, sfb_buf_bf16,
            sfa_offsets_bf16, sfb_offsets_bf16)
        # Manually pack SFA/SFB (all ones → write ue8m0 byte 127 into every slot).
        # For the bf16 test we need valid SFA/SFB data; use mxf8_transcode_and_pack_sfa
        # on a-payload with sfa_fp32=1.0.
        a_fp8_cp = a_fp8.clone()
        sfa_u8m0_fp32 = torch.empty_like(sfa_fp32)
        ext.mxf8_transcode_and_pack_sfa(
            a_fp8_cp, sfa_fp32, sfa_u8m0_fp32,
            expert_offsets, sfa_offsets_bf16,
            layout_sfa, sfa_buf_bf16)
        # Pack SFB once.
        b_fp8_cp = b_fp8.clone()
        sfb_u8m0_fp32 = torch.empty_like(sfb_fp32)
        ext.mxf8_transcode_weights_impl(b_fp8_cp, sfb_fp32, sfb_u8m0_fp32)
        ext.mxf8_pack_weight_sfb_impl(
            sfb_u8m0_fp32, layout_sfb, sfb_offsets_bf16, sfb_buf_bf16, N, K_)

        ext.moe_mxf8_grouped_mm_prepacked(
            out_bf16, a_fp8_cp, b_fp8_cp,
            problem_sizes,
            a_ptrs, b_ptrs, out_ptrs, sfa_ptrs, sfb_ptrs,
            stride_a, stride_b, stride_c,
            layout_sfa, layout_sfb,
            workspace)
        torch.cuda.synchronize()

        # ---- FP8-out path ----
        out_fp8 = torch.zeros(total_m, N, device=device, dtype=torch.float8_e4m3fn)
        sfd_offsets = torch.zeros(E + 1, device=device, dtype=torch.int32)
        ext.compute_mxf8_sfd_offsets_device(problem_sizes, sfd_offsets)
        sfd_total = int(sfd_offsets[-1].item())
        sfd_buf = torch.zeros(sfd_total, device=device, dtype=torch.uint8)
        d_ptrs = make_ptrs(); sfd_ptrs = make_ptrs()
        stride_d = make_stride(stride_fp8_sz)
        layout_sfd = make_layout(sfd_sz)

        ext.moe_mxf8_grouped_mm_prepacked_fp8out(
            out_fp8, sfd_buf, sfd_offsets,
            a_fp8_cp, b_fp8_cp,
            problem_sizes, expert_offsets,
            a_ptrs, b_ptrs, d_ptrs, sfd_ptrs, sfa_ptrs, sfb_ptrs,
            sfa_offsets_bf16, sfb_offsets_bf16,
            sfa_buf_bf16, sfb_buf_bf16,
            stride_a, stride_b, stride_d,
            layout_sfa, layout_sfb, layout_sfd,
            workspace)
        torch.cuda.synchronize()

        # Decode FP8 output via UE8M0 scales and compare to bf16 reference.
        # SFD layout is tiled: ceil(M/128)*128 rows × ceil(N/32)*4 sub-blocks.
        # For each element (m, n), its block-scale byte lives at
        # sfd[((m/128)*128 + (m%128)) * (N/32*4) + ((n/32) * 4 + (n%32)/8)]? —
        # we can just round-trip through CUTLASS's layout, but for validation
        # it's simpler to use the IDENTITY property that scales should rescale
        # the fp8 payload back to (approximately) the bf16 accumulated value.
        #
        # Simpler verification: the total norms / dot products should match.
        out_bf16_f32 = out_bf16.to(torch.float32)
        bf16_norm = float(out_bf16_f32.norm().item())

        # Extract SFD bytes. For each row m, loop over N-blocks of 32 and
        # find the corresponding UE8M0 byte in the tiled layout.
        #   tile_atom is Blk_MN=128, Blk_SF=4 (major K for RowMajor D).
        #   Layout:  (M/128, N/128) outer, then within a tile: (m_inner=128) × (kb_inner=4).
        # So for block (mb, nb_32): nb_128 = nb_32 / 4; kb_inner = nb_32 % 4;
        #   m_inner = m % 128; mb_outer = m / 128.
        # Offset formula (per expert): e_base + mb_outer * Blk_M_per_tile +
        #   nb_128 * (128 * 4) + m_inner * 4 + kb_inner.
        # CUTLASS uses different stride arrangement per expert; below is the
        # row-major M-outer iteration pattern for one expert:
        import numpy as np
        sfd_np = sfd_buf.cpu().numpy()
        sfd_offsets_np = sfd_offsets.cpu().numpy()
        out_fp8_np = out_fp8.to(torch.float32).cpu().numpy()
        out_bf16_np = out_bf16_f32.cpu().numpy()

        def decode_row_block(e_idx, m_global, n):
            # Decode one element by locating its UE8M0 byte and multiplying.
            m_in_expert = m_global - e_idx * M_e
            nb = n // 32
            mb_outer = m_in_expert // 128
            m_inner = m_in_expert % 128
            nb_outer = nb // 4
            kb_inner = nb % 4
            # Layout: within one (mb_outer, nb_outer) tile: 128 × 4 bytes
            # flattened row-major as m_inner stride=4, kb_inner stride=1.
            # Between tiles: outer shape is (M/128) × (N/128), stride (N/128)*128*4 and 128*4.
            N_outer_blocks = (N + 127) // 128
            M_outer_blocks = (M_e + 127) // 128
            tile_sz = 128 * 4
            e_base = sfd_offsets_np[e_idx]
            off = (e_base +
                   mb_outer * N_outer_blocks * tile_sz +
                   nb_outer * tile_sz +
                   m_inner * 4 +
                   kb_inner)
            return int(sfd_np[off])

        # Sample a few points and decode.
        samples = []
        for e in range(E):
            for off in [0, 32, 1024, 2048, 3000]:
                m = e * M_e + 5
                n = off
                ue8m0_byte = decode_row_block(e, m, n)
                scale = 2.0 ** (ue8m0_byte - 127)
                decoded = float(out_fp8_np[m, n]) * scale
                ref = float(out_bf16_np[m, n])
                samples.append({"e": e, "m": m, "n": n,
                                 "fp8_raw": float(out_fp8_np[m, n]),
                                 "ue8m0_byte": ue8m0_byte,
                                 "scale": scale,
                                 "decoded": decoded,
                                 "ref_bf16": ref,
                                 "abs_err": abs(decoded - ref)})

        return {
            "bf16_max_abs": float(out_bf16_f32.abs().max().item()),
            "fp8_max_abs_raw": float(out_fp8_np.max()),
            "bf16_norm": bf16_norm,
            "sfd_total_bytes": sfd_total,
            "samples": samples,
        }


@app.local_entrypoint()
def main():
    """Sanity check: ping the server."""
    s = BenchServer()
    print(s.warm.remote())
