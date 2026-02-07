# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

flash-moe enables large MoE (Mixture-of-Experts) models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD, rather than keeping all experts in Metal GPU memory. Target: run a 46GB model on a 32GB Mac at ≥3 tok/s.

The approach draws from Apple's "LLM in a Flash" and more recent research (MoEpic, FloE, FlashMoE, ActiveFlow). The target model is Qwen3-Coder-Next-4bit (48 MoE layers, 512 experts per layer, top-10 routing). See the full updated proposal in Obsidian: `flash-moe/updated project document.md`.

## Two-Repo Setup

This project spans two repositories:

- **`PATH_REMOVED (this repo) — Benchmarks, analysis tools, and the integration script `generate_lazy.py`. Uses uv with Python 3.13.
- **`PATH_REMOVED (local fork, `lazy-experts` branch) — The actual lazy expert loading implementation lives in `mlx_lm/lazy_experts.py`. Has its own `.venv`.

The core innovation is in mlx-lm: `LazyQuantizedSwitchLinear` replaces `QuantizedSwitchLinear` modules in MoE layers. Instead of holding weight arrays, it stores shard paths and loads only the needed experts (e.g., 10 out of 512) from safetensors on each forward pass.

## Commands

```bash
# Install flash-moe dependencies
uv sync

# Run the lazy expert integration test (MUST use mlx-lm's venv)
PATH_REMOVED generate_lazy.py ["prompt"] [max_tokens]

# Run benchmarks (most require sudo for fs_usage/purge)
sudo uv run python benchmarks/phase1_bench.py <model.gguf>
sudo uv run python benchmarks/cache_verify.py <model.gguf>
sudo uv run python benchmarks/madvise_bench.py <model.gguf>

# Run analysis
uv run python analysis/viability_calc.py --interactive
uv run python analysis/analyze_expert_log.py <expert_log.csv>
```

For mlx-lm changes, work from `PATH_REMOVED on the `lazy-experts` branch.

## Architecture

### How Lazy Expert Loading Works

1. Model loads with `lazy=True` — creates lazy tensor refs, not materialized arrays
2. `enable_lazy_experts(model, model_path)` walks `model.layers[i].mlp.switch_mlp` and replaces each `QuantizedSwitchLinear` (gate_proj, up_proj, down_proj) with `LazyQuantizedSwitchLinear` — expects 48 layers × 3 = 144 replacements
3. `mx.eval(model.parameters())` materializes only non-expert weights (~1.4 GB) because the lazy modules have zero `mx.array` attributes
4. During generation, each `LazyQuantizedSwitchLinear.__call__`:
   - Calls `mx.load(shard_path)` to get a fresh lazy ref (free — just reads headers)
   - Extracts only the unique expert IDs selected by the router
   - Remaps global indices (0–511) to local indices (0–N) for `gather_qmm()`
   - Full shard tensors are freed after evaluation

### Key Design Decision: Per-Call mx.load()

Early design held persistent `_LazyRef` objects, but fancy indexing (`tensor[expert_ids]`) materializes the **entire** source tensor (~258 MB per projection). The fix: load a fresh lazy ref each forward pass (free), index into it (materializes only the slice), and let the transient full tensor be freed. This keeps peak memory to ~2.4 GB during generation.

### Model Structure (Qwen3-Coder-Next)

```
Model → Qwen3NextModel → layers[0..47] → Qwen3NextDecoderLayer
  ├── self_attn / linear_attn  (alternating full/linear attention)
  ├── mlp: Qwen3NextSparseMoeBlock (on sparse layers)
  │   ├── gate: nn.Linear → softmax → top-10 selection
  │   ├── switch_mlp: SwitchGLU
  │   │   ├── gate_proj: QuantizedSwitchLinear  ← replaced by lazy
  │   │   ├── up_proj: QuantizedSwitchLinear    ← replaced by lazy
  │   │   └── down_proj: QuantizedSwitchLinear  ← replaced by lazy
  │   ├── shared_expert: Qwen3NextMLP (always active, not offloaded)
  │   └── shared_expert_gate: nn.Linear
  └── mlp: Qwen3NextMLP (on dense layers)
```

### This Repo's Code

- **`generate_lazy.py`** — Integration test: loads model, enables lazy experts, generates text, reports memory
- **`src/`** — Phase 1 building blocks (GGUF parsing, Metal buffer management, I/O strategies) — these were used for benchmarking; the actual inference path uses mlx-lm's safetensors loading
- **`benchmarks/`** — Phase 1 benchmarks: I/O throughput, cache verification, madvise prefetch, pipeline prototyping
- **`analysis/`** — Expert reuse analysis, cache simulation, viability calculation

## Current Status

**Phase 3 complete. Predictive zero-eval mode: 19.2 tok/s at 256 experts/layer (23 GB).**

| Phase | Speed | Memory | Status |
|-------|-------|--------|--------|
| 1: Lazy loading | 0.15 tok/s | 2.4 GB | Done |
| 2: LCP cache | 0.17 tok/s | 23 GB | Done (bottlenecked by tensor assembly) |
| 2.5: Predictive cache | **19.2 tok/s** | 23 GB | Done (zero-eval forward pass) |
| 3: Delta warmup | 19.2 tok/s | 23 GB | Done (scatter-based rebuild) |
| 3.5: Fast delta warmup | 19.2 tok/s | 23 GB | Done (42-55s cross-domain, ~2x faster) |

### Delta Warmup Performance (cross-domain switch)

`fast_delta_warmup()` in lazy_experts.py — best: **42s**, typical: 45-55s (vs 70-94s full re-warmup).

| Component | Time | Notes |
|-----------|------|-------|
| Discovery (predictive) | 0.9-1.5s | Through zero-eval predictive cache |
| Shard I/O (9 shards) | ~15s | Physical I/O floor, already optimal |
| Scatter eval (batched) | 25-37s | Metal buffer throughput limit (~72 GB traffic) |

**Router-only discovery rejected**: 40% overlap (router_only_forward), 74% overlap (speculative_router_probe) but 7-47s runtime — slower than predictive discovery at 1.3s.

**Cross-domain switches affect all 48 layers** (15-53 missing experts/layer). Skip-layers optimization only helps for same-domain switches.

**Next work:**
- Lower capacity experiments (128 experts → half Metal traffic → faster scatter eval)
- Same-domain delta warmup benchmarks (expect skip-layers to help there)
- Expert split (Phase 4) — cache top fraction permanently, load rest on demand
- Speculative prefetch — run next layer's router on current hidden state

## Measurement Discipline

- Before reporting benchmark results, explicitly state what was verified vs assumed
- Use `sudo purge` before EVERY cold I/O trial; verify with `fs_usage`
- Micro-benchmark projections must enumerate ALL pipeline costs before extrapolating
- Always run full end-to-end tests, not just isolated operations
- Validate measurements against proxy signals (high fallback rate should produce garbled output; if it reports 0% but output is garbled, the measurement is wrong)
- When comparing approaches, note run order — first run pays cold disk cache penalty

## MLX Internals to Know

- `nn.Module` extends `dict` — parameters stored in dict, not Python attributes
- `__setattr__`: if value is `mx.array|dict|list|tuple` → stored in dict (visible to `parameters()`); else → `super().__setattr__()` (hidden)
- This is why `LazyQuantizedSwitchLinear` stores only Python str/int — `parameters()` sees nothing, so `mx.eval(model.parameters())` skips expert weights
- `gather_qmm()` is the key MLX op for quantized expert dispatch
- **Buffer donation:** `x = dict.pop(key); x[idx] = val; dict[key] = x` enables zero-copy scatter (20x faster than list decompose + mx.stack)
- **Never cache `mx.load()` dicts** — holds lazy refs that pin full tensors in memory → OOM
- **Never `mx.eval()` inside forward pass** — breaks async_eval pipeline (2.2x slowdown)
- **Eval per-layer during rebuilds** — deferring eval across 48 layers causes OOM (old+new coexist)
- **Batched eval (EVAL_BATCH=8-10)** — middle ground between per-layer (safe but slow, 48 sync points) and deferred (OOM). Cuts rebuild time ~2.5x by amortizing Metal command buffer overhead.
- **Micro-benchmarks lie under memory pressure** — isolated scatter: 2.5ms/layer. Full model at 30 GB: 0.5-0.8s/layer (200x gap). Always validate with end-to-end tests at realistic memory load.
