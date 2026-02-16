# Agent Instructions

Instructions for coding agents working on this codebase.

## What This Project Is

MLX-MoE runs large MoE models on memory-constrained Macs by loading only routed experts from SSD into Metal memory. The primary target is `mlx-community/Qwen3-Coder-Next-4bit` (512 experts/layer, 48 MoE layers).

## Project Structure

```text
mlx_moe/                      # Package
  __init__.py                 # Exports: generate, stream_generate, Session
  cli.py                      # CLI entrypoint: mlx-moe serve
  server.py                   # OpenAI + Anthropic API server
  lazy_experts/
    core.py                   # enable/upgrade/reset, cache stats, dynamic updates
    modules.py                # Lazy/Cached/Predictive switch modules + caches
    loading.py                # Capacity selection, shard maps, selective loading
    discovery.py              # Router-only discovery
    warmup.py                 # Delta warmup and warmup helpers
    persistence.py            # Cache state, profiles, prepacked weights
    generate.py               # generate, stream_generate, Session, _startup

benchmarks/
  test_model.py               # Quick local smoke run (throughput + fallback)
  validate_quality.py         # Quality/warmup/memory/adaptive experiments
  profile_experts.py          # Build expert profiles (diverse/coding/tool-chat/mixed)
  benchmark_mlx_server.py     # mlx-moe-only server benchmark with streamed output
  benchmark_backends.py       # llama.cpp vs mlx-moe comparison (optional)
  sweep_profile_pinning.py    # Mix/pin sweep using real tool-chat traffic
  tool_chat_scenarios.py      # Tool schemas + agentic prompt scenarios

tests/
  test_unit_core.py
  test_unit_persistence.py
  test_integration.py         # Synthetic MoE + server endpoint tests

profiles/                     # Checked-in profile JSONs
logs/                         # Local benchmark/sweep outputs (gitignored)
docs/README.md                # High-level architecture overview
```

Depends on stock `mlx-lm >= 0.30.0` (no fork).

## Dev Setup

This is a uv project (`pyproject.toml`, Python 3.13).

```bash
uv sync
uv run python -c "from mlx_moe import generate; print('ok')"
uv run pytest
```

## Testing and Benchmarks

Core tests:

```bash
uv run pytest
```

Quick model smoke checks:

```bash
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit --capacity 208 --tokens 50
```

Validation suite (`validate_quality.py` uses positional experiment name):

```bash
uv run python benchmarks/validate_quality.py quality
uv run python benchmarks/validate_quality.py warmup
uv run python benchmarks/validate_quality.py memory
uv run python benchmarks/validate_quality.py memory-predictive
uv run python benchmarks/validate_quality.py delta-warmup
uv run python benchmarks/validate_quality.py adaptive
uv run python benchmarks/validate_quality.py expert-size
```

Profile generation:

```bash
uv run python benchmarks/profile_experts.py --model mlx-community/Qwen3-Coder-Next-4bit --prompts mixed --coding-weight 70 --toolchat-weight 30
```

mlx-moe-only server benchmark with live output:

```bash
uv run python benchmarks/benchmark_mlx_server.py \
  --model mlx-community/Qwen3-Coder-Next-4bit \
  --profile profiles/qwen3-coder-next-4bit.json \
  --capacity 208 --pin-top-k 32 --tools-mode field
```

This writes timestamped artifacts under:

```text
logs/model/<model_slug>/<profile_slug>/<datetime>/
  benchmark.json
  benchmark.md
  server.log
```

Pinning sweep (long-running):

```bash
uv run python benchmarks/sweep_profile_pinning.py
```

Cross-backend comparison (`benchmark_backends.py`) is optional and requires working llama.cpp.

## Architecture

### Startup Pipeline

`_startup()` in `mlx_moe/lazy_experts/generate.py` performs:

1. Load model lazy (`mlx_lm.load(..., lazy=True)`), detect MoE structure.
2. Auto-select capacity if omitted (`select_capacity` against Metal recommended working set).
3. Replace SwitchLinear modules with predictive-capable lazy modules.
4. Materialize non-expert weights (`mx.eval(model.parameters())`).
5. Build shard maps + `SafetensorsMap`.
6. Startup path (in order):
   - Prepacked weights (`*.weights.safetensors`) if present.
   - Cache-state upgrade (`*.json`) if present.
   - Else one of:
     - `warmup=full`: LCP warmup + predictive upgrade
     - profile-based upgrade (`--profile`)
     - router-only discovery + upgrade
7. Optional hybrid refinement (`warmup=hybrid`, only on fresh startup path).
8. Save cache state / prepacked snapshot if newly built.
9. Enable skip-fallback mode.
10. Apply `mx.set_wired_limit(...)` after expert loading completes.

### Module Replacement Chain

`QuantizedSwitchLinear` -> `LazyQuantizedSwitchLinear` -> `CachedQuantizedSwitchLinear` -> `PredictiveCachedSwitchLinear`

Final dispatch is zero-eval via pre-stacked tensors and lookup remap (`gather_qmm` path).

### Dynamic Cache Updates

`dynamic_cache_update()` runs between tokens and is invoked in server `_stream()` with adaptive interval/budget policy. Telemetry includes:

- `prefill`
- `ttft`
- `decode tok/s`
- `dcu_calls`
- `swaps`
- `fallback_rate`

### Wired Memory

`mx.set_wired_limit()` is applied after startup to pin the working set in a Metal residency set. Keep this ordering; moving/removing it regresses throughput.

### Key Constraints

- Capacity too low on Qwen3-Coder degrades quality sharply.
- Capacity too high on 32 GB machines hits Metal pressure cliffs.
- Do not call `mx.eval()` in predictive forward paths.
- MLX GPU eval is not thread-safe; server serialization is intentional.

## API Server (`mlx-moe serve`)

```bash
mlx-moe serve MODEL \
  [--host 127.0.0.1] [--port 8080] \
  [--capacity N] [--profile PATH] [--pin-top-k N] \
  [--max-tokens N] [--max-input-tokens N] \
  [--kv-bits N] [--kv-cache-slots N] \
  [--warmup hybrid|full|none] \
  [--shutdown-timeout N]
```

Endpoints:

- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /v1/models`

Sampling:

- Request fields `temperature`, `top_p`, `top_k` are passed through.
- Default for Qwen3-Coder family is `temp=0.2`, `top_p=0.95`, `top_k=40`.

Profile resolution:

- If `--profile` is omitted, server auto-detects from `profiles/`:
  - `<model-slug>-toolchat.json`
  - `<model-slug>.json`

KV cache behavior:

- Keyed LRU cache (`--kv-cache-slots`, default 1).
- Reuse is based on longest common token prefix for each cache key.
- Cache is invalidated before generation and restored only on successful completion.

Hybrid startup refinement:

- With `--warmup hybrid`, server load runs two short coding prompts and dynamic updates before `Ready.`.

Limits:

- `--max-input-tokens` rejects oversized requests.
- `--max-tokens` caps request output.
- Anthropic non-streaming responses are additionally capped to 512 output tokens.

## Code Style

- Comments explain why, not what.
- No defensive checks for impossible states.
- No silent fallbacks that hide failures.
- Remove dead code paths cleanly; no compatibility shims for deleted behavior.
