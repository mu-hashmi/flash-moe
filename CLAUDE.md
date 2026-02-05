# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

flash-moe enables large MoE (Mixture-of-Experts) models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD, rather than keeping all experts in Metal GPU memory. Target: run a 46GB model on a 32GB Mac at ≥3 tok/s.

The approach is based on Apple's "LLM in a Flash" paper. The target model is Qwen3-Coder-Next-4bit (48 MoE layers, 512 experts per layer, top-10 routing).

## Two-Repo Setup

This project spans two repositories:

- **`/Users/muhash/flash-moe/`** (this repo) — Benchmarks, analysis tools, and the integration script `generate_lazy.py`. Uses uv with Python 3.13.
- **`/Users/muhash/mlx-lm/`** (local fork, `lazy-experts` branch) — The actual lazy expert loading implementation lives in `mlx_lm/lazy_experts.py`. Has its own `.venv`.

The core innovation is in mlx-lm: `LazyQuantizedSwitchLinear` replaces `QuantizedSwitchLinear` modules in MoE layers. Instead of holding weight arrays, it stores shard paths and loads only the needed experts (e.g., 10 out of 512) from safetensors on each forward pass.

## Commands

```bash
# Install flash-moe dependencies
uv sync

# Run the lazy expert integration test (MUST use mlx-lm's venv)
/Users/muhash/mlx-lm/.venv/bin/python generate_lazy.py ["prompt"] [max_tokens]

# Run benchmarks (most require sudo for fs_usage/purge)
sudo uv run python benchmarks/phase1_bench.py <model.gguf>
sudo uv run python benchmarks/cache_verify.py <model.gguf>
sudo uv run python benchmarks/madvise_bench.py <model.gguf>

# Run analysis
uv run python analysis/viability_calc.py --interactive
uv run python analysis/analyze_expert_log.py <expert_log.csv>
```

For mlx-lm changes, work from `/Users/muhash/mlx-lm/` on the `lazy-experts` branch.

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

## Current Status and Known Limitations

**Phase 1 complete** — lazy loading works, memory reduced from ~40 GB to ~2.4 GB peak. But generation speed is ~0.1 tok/s due to:
- 48 `mx.eval` sync points per token (one per MoE layer)
- 144 `mx.load` header parses per token
- Full expert tensor disk reads (no caching of recently-used experts)

**Next phases:** Expert caching (Phase 2), async prefetch (Phase 3), LLM-in-a-Flash optimizations (Phase 4).

## MLX Internals to Know

- `nn.Module` extends `dict` — parameters stored in dict, not Python attributes
- `__setattr__`: if value is `mx.array|dict|list|tuple` → stored in dict (visible to `parameters()`); else → `super().__setattr__()` (hidden)
- This is why `LazyQuantizedSwitchLinear` stores only Python str/int — `parameters()` sees nothing, so `mx.eval(model.parameters())` skips expert weights
- `gather_qmm()` is the key MLX op for quantized expert dispatch
