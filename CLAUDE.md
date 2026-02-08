# CLAUDE.md

## What This Project Is

flash-moe enables large MoE models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD. Runs a 46GB model on a 32GB Mac at 6-23 tok/s using 19 GB.

## Project Structure

```
flash_moe/                  # The package (pip-installable)
  __init__.py               # Exports: flash_generate, flash_stream_generate, FlashSession
  lazy_experts/             # Core implementation
    core.py                 # enable/upgrade/reset, cache stats, dynamic refresh
    modules.py              # ExpertCache, Lazy/Cached/Predictive module classes
    loading.py              # Weight loading, shard maps, capacity selection
    discovery.py            # Router-only discovery, speculative probes
    warmup.py               # Delta warmup, incremental warmup
    persistence.py          # Cache state save/load, prepacked weights, profiles
    generate.py             # flash_generate, flash_stream_generate, FlashSession

  cli.py                    # CLI entry point: flash-moe serve
  server.py                 # OpenAI + Anthropic API server (Starlette + uvicorn)

benchmarks/                 # Profiling and benchmark scripts
profiles/                   # Pre-computed expert profiles (auto-detected by model name)
universal_experts.json      # Pre-computed Qwen expert profile (saves 33 min)
```

Depends on stock `mlx-lm >= 0.30.0` — no fork needed. Uses `mlx-lm` for model loading (`mlx_lm.load`), generation (`mlx_lm.generate`, `mlx_lm.stream_generate`), and the `QuantizedSwitchLinear` base class.

## Dev Setup

This is a uv project. Python 3.13.

```bash
uv sync
uv run python -c "from flash_moe import flash_generate; print('ok')"
```

## Running Benchmarks

```bash
# Expert profiling (22 prompts, ~33 min for Qwen, ~7 min for Mixtral/GLM)
uv run python benchmarks/profile_experts.py --model qwen
uv run python benchmarks/profile_experts.py --model mixtral --output benchmarks/mixtral_experts.json

# Multi-turn session benchmark
uv run python benchmarks/bench_multiturn.py --model qwen --turns 20 --tokens 200

# Pinning benchmark (4 configs × N tokens)
uv run python benchmarks/bench_pinning.py --model glm --profile benchmarks/glm_experts.json

# Adaptive per-layer budget
uv run python benchmarks/bench_adaptive.py --model qwen --profile universal_experts.json

# Context growth (accumulating conversation history)
uv run python benchmarks/bench_context_growth.py --model qwen --turns 10
```

## Architecture

### How It Works

1. `enable_lazy_experts(model, model_path, capacity)` replaces `QuantizedSwitchLinear` modules with lazy-loading versions
2. `mx.eval(model.parameters())` loads only non-expert weights (~1.4 GB)
3. Expert warmup via router-only discovery (~1s) or pre-computed profile (0s)
4. `upgrade_to_predictive()` loads top experts into GPU-resident stacked tensors
5. Generation uses zero-eval forward pass with pre-stacked expert lookup tables

### Supported Models

Any MLX model using `SwitchGLU` with either module path:
- `layer.mlp.switch_mlp` (Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE)
- `layer.block_sparse_moe.switch_mlp` (Mixtral, PhiMoE, MiniMax, GraniteMoE)

### Key Constraints

- **Quality cliff at capacity < 192** — garbled output regardless of warmup (Qwen)
- **Metal pressure cliff at capacity > 208** on 32 GB Macs — 3x eval degradation
- **Never `mx.eval()` inside forward pass** — breaks async_eval pipeline (2.2x slowdown)
- **Never cache `mx.load()` dicts** — lazy refs pin full tensors → OOM
- **Eval per-layer during rebuilds** — deferred eval across 48 layers causes OOM
- **MLX is NOT thread-safe** for GPU eval — cooperative single-thread only
- Per-call `mx.load()` in warmup is intentional — fancy indexing materializes full source tensors, so fresh lazy refs each call prevents OOM

## API Server (`flash-moe serve`)

```bash
flash-moe serve mlx-community/Qwen3-Coder-Next-4bit [--port 8080] [--host 127.0.0.1] [--capacity N] [--profile PATH] [--max-tokens N] [--max-input-tokens N]
```

Endpoints: `/v1/chat/completions` (OpenAI), `/v1/messages` (Anthropic), `/v1/models`.

Sampling parameters (`temperature`, `top_p`, `top_k`) are passed through from the request. Tool calls are parsed and converted between Anthropic and OpenAI formats. Truncated tool calls (hit token cap mid-generation) are salvage-parsed.

### Server limits

- `--max-tokens` (default 4096) — max output tokens per request, caps KV cache growth
- `--max-input-tokens` (default 16384) — rejects requests over this
- Non-streaming responses capped at 512 output tokens to avoid blocking

## Code Style

- No unnecessary comments. Comments explain WHY, not WHAT.
- No defensive checks for impossible conditions.
- No silent fallbacks. If something fails unexpectedly, crash loudly.
- No backwards-compatibility shims for deleted code.
