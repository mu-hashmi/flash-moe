# Contributing

## Setup

```bash
git clone https://github.com/mu-hashmi/mlx-moe.git
cd mlx-moe
uv sync
uv run pytest
```

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

## Running Tests

```bash
uv run pytest                # unit + integration (~0.5s, no model download)
uv run pytest -v             # verbose

# Smoke tests against real models (manual, downloads model on first run)
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit
uv run python benchmarks/test_model.py mlx-community/Qwen3-30B-A3B-4bit --capacity 128 --tokens 50
```

## Adding Model Support

mlx-moe auto-detects any MLX model using `QuantizedSwitchLinear` inside a `SwitchGLU` block. Two module path conventions are supported:

- `layer.mlp.switch_mlp` -- Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE
- `layer.block_sparse_moe.switch_mlp` -- Mixtral, PhiMoE, MiniMax, GraniteMoE

If a new model uses a different path, add detection in `_find_switch_mlp()` and `_find_moe_block()` in `mlx_moe/lazy_experts/loading.py`.

Two weight storage formats are handled:
- **Stacked** (Qwen): `{prefix}.switch_mlp.gate_proj.weight` is `(E, ...)`
- **Per-expert** (Mixtral, GLM): `{prefix}.experts.{i}.w1.weight` etc.

The shard map builder (`_build_shard_map`) adds synthetic stacked-format keys for per-expert models, so the rest of the code works uniformly.

## Generating Expert Profiles

```bash
uv run python benchmarks/profile_experts.py --model mlx-community/MODEL-NAME
```

Place the output JSON in `profiles/` with a name matching the model (auto-detected at runtime).

## Code Style

- No unnecessary comments. Comments explain WHY, not WHAT.
- No defensive checks for conditions that cannot happen. Trust the code.
- No silent fallbacks. If something fails unexpectedly, crash loudly.
- No backwards-compatibility shims for deleted code.
- No `try/except` around code that cannot raise that exception.

## MLX-Specific Rules

- Never call `mx.eval()` inside the forward pass -- breaks `async_eval` pipelining.
- Never cache `mx.load()` return values -- lazy refs pin full tensors, causing OOM.
- Call `mx.eval()` per-layer during tensor rebuilds -- deferring across 48 layers OOMs.
- Expert slot sizes must include weight + scales + biases, not just weight.
