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
/Users/muhash/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens] [capacity] [mode] [refresh_interval]
# Modes: predictive (default), async-delta, async-delta-coherent, async-delta-hybrid,
#         delta-warmup, sync-predictive, cached, lazy
# Example: .../python generate_lazy.py "Write hello world" 100 208 predictive 50

# Cache-persistent generation (skips 75s warmup on repeat runs)
/Users/muhash/mlx-lm/.venv/bin/python generate_persistent.py <cache.json> ["prompt"] [max_tokens] [capacity]

# Universal expert profiling (22 prompts, ~33 min)
/Users/muhash/mlx-lm/.venv/bin/python benchmarks/profile_experts.py [capacity] [threshold] [output.json]

# Pinning benchmark (4 configs × 1000 tokens)
/Users/muhash/mlx-lm/.venv/bin/python benchmarks/bench_pinning.py [profile.json] [capacity] [max_tokens]
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

**Pre-production sweep complete.** Results in `pre_prod_sweep.md` (Obsidian vault).

**Recommended config for 32 GB Mac:** 208 capacity + pinning + cache_limit(0). 19.1 GB, 8.7 tok/s, coherent through 1000 tokens.

**What's implemented:**
- `upgrade_to_predictive_with_pinning()` — pins universal experts, fixes 300-token degradation (rep 0.20→0.03)
- `generate_persistent.py` — cache persistence, 60s warm start (was 155s cold)
- `async-delta-coherent` mode — buffers stale tokens, streams from first coherent token
- `_find_switch_mlp()` + `_detect_num_experts()` — supports Qwen + Mixtral + GLM model families
- `dynamic_cache_update()` wired into generation loop with configurable refresh interval
- `compute_adaptive_allocations()` — MoEpic greedy per-layer budget
- `dynamic_cache_update_ml()` — ML eviction scorer (untrained)

**Known limitations:**
- Cold start: 60s with cache persistence (12s upgrade), 155s without
- Quality cliff: capacity <192 produces garbled output
- Metal pressure cliff: capacity >208 triggers 3x eval degradation (cache_limit(0) pushes to 240)
- Mild sentence-level repetition persists at 1000+ tokens (model capacity limitation, not flash-moe)

## Next Work

- Test on GLM-4.7-Flash-4bit and Mixtral-8x7B-Instruct-4bit (validate generalization)
- Per-layer adaptive budget (profiling done, compute allocations)
- Cache persistence + pinning integration (save universal profile path in cache state JSON)
- Train ML eviction models (low priority)

## MLX Pitfalls (project-specific)

- **Never cache `mx.load()` dicts** — lazy refs pin full tensors → OOM
- **Never `mx.eval()` inside forward pass** — breaks async_eval (2.2x slowdown)
- **Eval per-layer during rebuilds** — deferred eval across 48 layers causes OOM
- **Buffer donation pattern:** `x = dict.pop(key); x[idx] = val; dict[key] = x` — 20x faster scatter
- **Micro-benchmarks lie under memory pressure** — 200x gap vs full model. Always validate end-to-end.
- **`mx.metal.set_cache_limit(0)`** before heavy ops reclaims several GB headroom
- **`mx.metal.get_cache_limit()` does NOT exist** — restore with `device_info()["memory_size"] // 4`
- **MLX NOT thread-safe** for GPU eval — cooperative single-thread only
