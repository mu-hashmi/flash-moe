# CLAUDE.md

## What This Project Is

flash-moe enables large MoE models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD. Runs a 46GB model on a 32GB Mac at 6-13 tok/s using 17-19 GB.

Target model: Qwen3-Coder-Next-4bit (48 MoE layers, 512 experts/layer, top-10 routing).

## Two-Repo Setup

- **`PATH_REMOVED (this repo) — Benchmarks, analysis, `generate_lazy.py`. Uses uv, Python 3.13.
- **`PATH_REMOVED (local fork, `lazy-experts` branch) — Core implementation in `mlx_lm/lazy_experts/` sub-package. Has its own `.venv`.

## Commands

```bash
# Integration test (MUST use mlx-lm's venv)
PATH_REMOVED generate_lazy.py ["prompt"] [max_tokens] [capacity] [mode] [refresh_interval]
# Modes: predictive (default), async-delta, async-delta-coherent, async-delta-hybrid,
#         delta-warmup, sync-predictive, cached, lazy
# Example: .../python generate_lazy.py "Write hello world" 100 208 predictive 50

# Cache-persistent generation (skips 75s warmup on repeat runs)
PATH_REMOVED generate_persistent.py <cache.json> ["prompt"] [max_tokens] [capacity]

# Universal expert profiling (22 prompts, ~33 min per model)
PATH_REMOVED benchmarks/profile_experts.py --model qwen   # or mixtral, glm, or HF name
PATH_REMOVED benchmarks/profile_experts.py --model mixtral --output mixtral_experts.json

# Streaming generation
PATH_REMOVED generate_streaming.py --model qwen --prompt "Write a Flask server" --tokens 200

# Multi-turn session benchmark (memory growth + quality over N turns)
PATH_REMOVED benchmarks/bench_multiturn.py --model qwen --turns 20 --tokens 200

# Pinning benchmark (4 configs × 1000 tokens)
PATH_REMOVED benchmarks/bench_pinning.py [profile.json] [capacity] [max_tokens]

# Warmup optimization benchmark (5 configs)
PATH_REMOVED benchmarks/bench_warmup.py [profile_path] [capacity]
```

For mlx-lm changes, work from `PATH_REMOVED on the `lazy-experts` branch.

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
3. Expert discovery (one of):
   - **Prepacked warm start** (~4s): `load_prepacked_weights()` from saved safetensors
   - **Profile-based cold start** (~11s): `upgrade_from_profile()` using universal expert profile
   - **Router-only discovery** (~1s discovery + ~15s upgrade): `router_only_discovery()` then `upgrade_to_predictive()`
   - Legacy: full 10-token generation (~76s discovery + ~15s upgrade)
4. Generation uses zero-eval forward pass with pre-stacked expert lookup tables

Per-call `mx.load()` in Phase 2 is intentional — fancy indexing materializes full source tensors, so fresh lazy refs each call prevents OOM.

## Current Status

**Warmup optimization complete.** Results in `warmup_optim.md` (Obsidian vault).

**Recommended config for 32 GB Mac:** 208 capacity + pinning + wired_limit + cache=256MB. 19.0 GB, ~23 tok/s (200-tok burst). **6s warm start, 13s cold start.**

**3 models validated:** Qwen3-Coder-Next-4bit, Mixtral-8x7B-Instruct-4bit, GLM-4.7-Flash-4bit.

**What's implemented:**
- `flash_generate()` — one-call API with auto capacity, cache persistence, pinning, wired residency, prepacked weights, memory guards
- `router_only_discovery()` — 76x faster cold-start discovery (76s → 1s) via batched eval
- `save_prepacked_weights()` / `load_prepacked_weights()` — skip upgrade_to_predictive on warm start (14.8s → 3.9s)
- `upgrade_from_profile()` — profile-based cold start, skip discovery entirely
- `upgrade_to_predictive_with_pinning()` — pins universal experts, fixes 300-token degradation
- `_find_switch_mlp()` + `_detect_num_experts()` — supports Qwen + Mixtral + GLM model families
- `_load_proj_experts()` — format-agnostic expert loading (stacked + per-expert + cross-shard)
- `dynamic_cache_update()` wired into generation loop with configurable refresh interval
- `compute_adaptive_allocations()` — MoEpic greedy per-layer budget

**Known limitations:**
- Quality cliff: capacity <192 produces garbled output
- Metal pressure cliff: capacity >208 triggers 3x eval degradation (set_wired_limit does NOT extend ceiling)
- Mild sentence-level repetition persists at 1000+ tokens (model capacity limitation, not flash-moe)

## Next Work

- Per-layer adaptive budget (profiling done, compute allocations implemented)
- Profile pinning on Mixtral/GLM (run profile_experts.py on non-Qwen models)
- Streaming flash_generate variant for chat applications
- Upstream to mlx-lm (lazy-experts branch needs PR or plugin architecture)
- Test DeepSeek, Hunyuan, PhiMoE (_find_switch_mlp() lists support but untested)

## MLX Pitfalls (project-specific)

- **Never cache `mx.load()` dicts** — lazy refs pin full tensors → OOM
- **Never `mx.eval()` inside forward pass** — breaks async_eval (2.2x slowdown)
- **Eval per-layer during rebuilds** — deferred eval across 48 layers causes OOM
- **Buffer donation pattern:** `x = dict.pop(key); x[idx] = val; dict[key] = x` — 20x faster scatter
- **Micro-benchmarks lie under memory pressure** — 200x gap vs full model. Always validate end-to-end.
- **`mx.set_cache_limit(0)`** before heavy ops reclaims several GB headroom
- **`mx.get_cache_limit()` does NOT exist** — restore with `device_info()["memory_size"] // 4`
- **`mx.set_wired_limit(bytes)`** — pins Metal buffers in physical RAM via MTLResidencySet (macOS 15+, no-ops on older). Call after model+cache are loaded to prevent OS paging during generation.
- **MLX NOT thread-safe** for GPU eval — cooperative single-thread only
- **`sudo sysctl iogpu.wired_limit_mb=28672`** — raises GPU memory cap from ~75% to ~88% of RAM. Doesn't persist across reboots. Risk of OS instability.
