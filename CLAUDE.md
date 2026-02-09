# CLAUDE.md

## What This Project Is

mlx-moe enables large MoE models to run on memory-constrained Macs by loading only router-selected experts on demand from SSD. Runs a 46GB model on a 32GB Mac at 6-23 tok/s using 19 GB.

## Project Structure

```
mlx_moe/                    # The package (pip-installable)
  __init__.py               # Exports: generate, stream_generate, Session
  lazy_experts/             # Core implementation
    core.py                 # enable/upgrade/reset, cache stats, skip-fallback
    modules.py              # ExpertCache, Lazy/Cached/Predictive module classes
    loading.py              # Weight loading, shard maps, capacity selection
    discovery.py            # Router-only discovery, speculative probes
    warmup.py               # Delta warmup, incremental warmup
    persistence.py          # Cache state save/load, prepacked weights, profiles
    generate.py             # generate, stream_generate, Session, _startup

  cli.py                    # CLI entry point: mlx-moe serve
  server.py                 # OpenAI + Anthropic API server (Starlette + uvicorn)

tests/                      # Automated test suite (uv run pytest)
  test_unit_core.py         # ExpertCache, PredictiveExpertCache, select_capacity, module detection
  test_unit_persistence.py  # save/load roundtrips, SafetensorsMap
  test_integration.py       # Synthetic 8-expert model pipeline, server endpoints

benchmarks/                 # Performance benchmarks and smoke tests (run manually)
profiles/                   # Pre-computed expert profiles (auto-detected by model name)
```

Depends on stock `mlx-lm >= 0.30.0` — no fork needed. Uses `mlx-lm` for model loading (`mlx_lm.load`), generation (`mlx_lm.generate`, `mlx_lm.stream_generate`), and the `QuantizedSwitchLinear` base class.

## Dev Setup

This is a uv project. Python 3.13.

```bash
uv sync
uv run pytest                # 100 tests, ~0.5s
uv run python -c "from mlx_moe import generate; print('ok')"
```

## Testing

```bash
uv run pytest                # unit + integration tests (no model download needed)
uv run pytest -v             # verbose output

# Smoke tests against real models (manual, takes minutes)
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit --capacity 208 --tokens 50
```

## Architecture

### How It Works

1. `enable_lazy_experts(model, model_path, capacity)` replaces `QuantizedSwitchLinear` modules with lazy-loading versions
2. `mx.eval(model.parameters())` loads only non-expert weights (~1.4 GB)
3. Expert warmup via router-only discovery (~1s) or pre-computed profile (0s)
4. `upgrade_to_predictive()` loads top experts into GPU-resident stacked tensors
5. Generation uses zero-eval forward pass with pre-stacked expert lookup tables

### Auto Capacity Selection

`select_capacity()` uses Metal's `max_recommended_working_set_size` (hardware-reported, not a percentage of system RAM) to pick the right number of experts. Targets 71% of the recommended limit, leaving headroom for KV cache growth. Expert slot sizes include weight + scales + biases (all quantization metadata). After upgrade, `mx.get_peak_memory()` is checked against `gc_limit` (0.95 × recommended) and warns if exceeded.

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

## API Server (`mlx-moe serve`)

```bash
mlx-moe serve mlx-community/Qwen3-Coder-Next-4bit [--port 8080] [--host 127.0.0.1] [--capacity N] [--profile PATH] [--max-tokens N] [--max-input-tokens N] [--kv-bits N]
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
