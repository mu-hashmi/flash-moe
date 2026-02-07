# CLAUDE.md

## What This Project Is

flash-moe enables large MoE models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD. Runs a 46GB model on a 32GB Mac at 6-13 tok/s using 17-19 GB.

Target model: Qwen3-Coder-Next-4bit (48 MoE layers, 512 experts/layer, top-10 routing).

## Two-Repo Setup

- **`/Users/muhash/flash-moe/`** (this repo) — Benchmarks, analysis, `generate_lazy.py`. Uses uv, Python 3.13.
- **`/Users/muhash/mlx-lm/`** (local fork, `lazy-experts` branch) — Core implementation in `mlx_lm/lazy_experts.py`. Has its own `.venv`.

## Commands

```bash
# Integration test (MUST use mlx-lm's venv)
/Users/muhash/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens] [capacity] [mode]
# Modes: predictive (default), async-delta, delta-warmup, sync-predictive, cached, lazy
# Example: .../python generate_lazy.py "Write hello world" 100 208 async-delta
```

For mlx-lm changes, work from `/Users/muhash/mlx-lm/` on the `lazy-experts` branch.

## Architecture

### Model Structure (Qwen3-Coder-Next)

```
layers[0..47] → Qwen3NextDecoderLayer
  ├── self_attn / linear_attn
  ├── mlp: Qwen3NextSparseMoeBlock
  │   ├── gate: nn.Linear → top-10 selection
  │   ├── switch_mlp: SwitchGLU
  │   │   ├── gate_proj: QuantizedSwitchLinear  ← replaced by lazy
  │   │   ├── up_proj: QuantizedSwitchLinear    ← replaced by lazy
  │   │   └── down_proj: QuantizedSwitchLinear  ← replaced by lazy
  │   ├── shared_expert: Qwen3NextMLP (always active)
  │   └── shared_expert_gate: nn.Linear
  └── mlp: Qwen3NextMLP (dense layers)
```

### How It Works

1. `enable_lazy_experts(model, model_path, capacity, predictive=True)` replaces `QuantizedSwitchLinear` modules (144 total)
2. `mx.eval(model.parameters())` loads only non-expert weights (~1.4 GB)
3. Warmup: generate 10 tokens to discover expert routing
4. `upgrade_to_predictive(model, model_path, capacity)` pre-loads top experts into GPU tensors
5. Generation uses zero-eval forward pass with pre-stacked expert lookup tables

Per-call `mx.load()` is intentional — fancy indexing materializes full source tensors, so fresh lazy refs each call prevents OOM.

## Current Status

**Phase 4 complete + validated.** Full results in `final_validation.md` (Obsidian vault).

**Recommended configs for 32 GB Mac:**
- **208 capacity + cache_limit(0)**: 19.1 GB, 8.1 tok/s — optimal
- **192 capacity**: 17.7 GB, 6.1 tok/s — safe default
- `select_capacity(base_gb, sys_gb)` in lazy_experts.py auto-selects

**Async delta warmup:** 2.5 tok/s during swaps → 12.9 tok/s after. Swaps complete at token 24. KV cache does NOT poison past swap completion.

**Known limitations:**
- Cold start: ~89s (75s warmup gen + 12s upgrade)
- Long generation degrades at ~300 tokens at 192 cap (filler expert limitation)
- Quality cliff: capacity <192 produces garbled output
- Metal pressure cliff: capacity >208 triggers 3x eval degradation (cache_limit(0) pushes to 240)

## Next Work

- **Dynamic cache refresh** during generation (fix 300-token degradation — highest priority)
- Expert split (Phase 5) — pin top-frequency experts permanently
- `_find_switch_mlp()` helper for multi-model support (25+ models use SwitchGLU)
- "Hold and discard" mode — buffer during swaps, stream after convergence

## MLX Pitfalls (project-specific)

- **Never cache `mx.load()` dicts** — lazy refs pin full tensors → OOM
- **Never `mx.eval()` inside forward pass** — breaks async_eval (2.2x slowdown)
- **Eval per-layer during rebuilds** — deferred eval across 48 layers causes OOM
- **Buffer donation pattern:** `x = dict.pop(key); x[idx] = val; dict[key] = x` — 20x faster scatter
- **Micro-benchmarks lie under memory pressure** — 200x gap vs full model. Always validate end-to-end.
- **`mx.metal.set_cache_limit(0)`** before heavy ops reclaims several GB headroom
- **MLX NOT thread-safe** for GPU eval — cooperative single-thread only
