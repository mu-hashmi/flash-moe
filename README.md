# flash-moe

Run large Mixture-of-Experts models on memory-constrained Macs by loading only router-selected experts on demand from SSD. A 46 GB Qwen3-Coder-Next model runs on a 32 GB Mac at 6-23 tok/s using 17-19 GB, with coherent output through 1000+ tokens via universal expert pinning.

Built on [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm). Requires a local fork of mlx-lm with the `lazy-experts` branch.

## Hardware Requirements

| System RAM | Capacity | Est. Memory | tok/s | Notes |
|-----------|----------|-------------|-------|-------|
| 32 GB | 192 | 17.7 GB | 6-13 | Safe default, most headroom |
| 32 GB | 208 | 19.1 GB | 8-23 | Best config with `cache_limit(0)` |
| 48 GB | 320 | ~28 GB | 15-25 | Estimated |
| 64 GB | 432 | ~37 GB | 20-30 | Estimated |
| 128 GB | 512 (full) | ~44 GB | 30-40 | Full expert coverage, no fallback |

"Capacity" = number of experts cached per MoE layer (out of 512 for Qwen3-Coder-Next).

## Supported Models

| Model | Experts | Top-K | Tested Config | Memory | tok/s |
|-------|---------|-------|---------------|--------|-------|
| [Qwen3-Coder-Next-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit) | 512/layer, 48 MoE layers | 10 | cap 208 + pinning | 19.1 GB | 8.7 |
| [Mixtral-8x7B-Instruct-v0.1-4bit](https://huggingface.co/mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit) | 8/layer, 32 MoE layers | 2 | cap 6 | 19.9 GB | 4.5 |
| [GLM-4.7-Flash-4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) | 64/layer, 46 MoE layers | 4 | cap 48 | 13.0 GB | 5.7 |

Any MLX model using `SwitchGLU` with either module path is supported:
- `layer.mlp.switch_mlp` (Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE)
- `layer.block_sparse_moe.switch_mlp` (Mixtral, PhiMoE, MiniMax, GraniteMoE)

Both stacked safetensors (Qwen) and per-expert safetensors (Mixtral, GLM) formats are handled automatically.

## Quick Start

```bash
# 1. Clone and set up mlx-lm fork
git clone https://github.com/<your-fork>/mlx-lm.git
cd mlx-lm && git checkout lazy-experts
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Clone flash-moe
git clone https://github.com/<your-fork>/flash-moe.git

# 3. Generate (auto-selects capacity for your hardware)
cd flash-moe
/path/to/mlx-lm/.venv/bin/python -c "
from mlx_lm.lazy_experts import flash_generate
print(flash_generate('mlx-community/Qwen3-Coder-Next-4bit',
                      'Write a Python hello world program',
                      max_tokens=200))
"
```

For more control, use `generate_lazy.py`:

```bash
# Format: generate_lazy.py [prompt] [max_tokens] [capacity] [mode]
/path/to/mlx-lm/.venv/bin/python generate_lazy.py \
    "Write a Flask web server" 200 208 predictive
```

## How It Works

```
                    +-----------------+
                    |  Load model     |  mx.load(lazy=True)
                    |  (non-expert    |  ~1.4 GB for Qwen3-Coder
                    |   weights only) |
                    +--------+--------+
                             |
                    +--------v--------+
                    |  enable_lazy_   |  Replace QuantizedSwitchLinear
                    |  experts()      |  with cached loaders
                    +--------+--------+
                             |
                    +--------v--------+
                    |  Warmup         |  Generate 10 tokens to
                    |  (Phase 2)      |  discover expert routing
                    +--------+--------+
                             |
                    +--------v--------+
                    |  upgrade_to_    |  Harvest LCP caches into
                    |  predictive()   |  zero-eval GPU tensors
                    +--------+--------+
                             |
                    +--------v--------+
                    |  Generate       |  Zero mx.eval() forward pass
                    |  (Phase 3)      |  using pre-stacked lookups
                    +-----------------+
```

### Architecture (Qwen3-Coder-Next)

```
layers[0..47] -> DecoderLayer
  +-- self_attn / linear_attn
  +-- mlp: SparseMoeBlock
  |   +-- gate: nn.Linear -> top-10 selection
  |   +-- switch_mlp: SwitchGLU
  |   |   +-- gate_proj: QuantizedSwitchLinear  <-- replaced
  |   |   +-- up_proj:   QuantizedSwitchLinear  <-- replaced
  |   |   +-- down_proj: QuantizedSwitchLinear  <-- replaced
  |   +-- shared_expert: MLP (always active)
  |   +-- shared_expert_gate: nn.Linear
  +-- mlp: MLP (dense layers)
```

Each MoE layer has 512 experts but only routes to 10 per token. flash-moe replaces the 3 expert projection modules per layer (144 total) with lazy-loading versions that fetch only the needed experts from SSD.

## Key Features

**Universal Expert Pinning** -- Profiles routing across 22 diverse prompts to identify universally-activated experts. Pins these in the cache so filler slots produce reasonable output instead of garbage. Reduces repetition score from 0.20 to 0.03 at 1000 tokens with zero performance cost.

**Cache Persistence** -- Saves expert routing state to JSON after warmup. Subsequent launches skip the 75s discovery phase, cutting cold start from 155s to 60s (2.6x faster).

**Predictive Cache (Phase 3)** -- Pre-loads top experts into GPU-resident tensors with a lookup table for zero-eval dispatch. No `mx.eval()` in the forward pass, preserving MLX's async eval pipeline.

**Dynamic Cache Refresh** -- Swaps cold experts for newly-requested ones between tokens. Useful for long-running multi-turn conversations where routing patterns shift.

**Hold-and-Discard** -- During async delta warmup (domain switches), buffers stale tokens and streams from the first coherent token. No garbage visible to the user.

## Performance

### Qwen3-Coder-Next-4bit on 32 GB Mac

| Config | Memory | tok/s | Rep@1000 | Notes |
|--------|--------|-------|----------|-------|
| Baseline (cap 208) | 19.1 GB | 8.8 | 0.20 | Degrades at ~250 tokens |
| + Pinning | 19.1 GB | 8.7 | 0.03 | Coherent through 1000+ tokens |
| + Cache persistence | 19.1 GB | 8.7 | 0.03 | 60s warm start (was 155s) |
| Cap 192 (safe) | 17.7 GB | 6.1 | -- | More memory headroom |

### Startup Times

| Scenario | Time |
|----------|------|
| Cold start (first run) | 155s |
| Warm start (cached state) | 60s |
| Cross-domain delta warmup | ~10s |

## Project Structure

```
flash-moe/                          # This repo
  generate_lazy.py                  # Main generation script
  generate_persistent.py            # Cache-persistent generation
  test_generalization.py            # Multi-model smoke tests
  benchmarks/
    profile_experts.py              # Universal expert profiling
    bench_pinning.py                # Pinning benchmark

mlx-lm/                            # Local fork (lazy-experts branch)
  mlx_lm/lazy_experts.py           # Core implementation (~2800 lines)
```

## API Reference

### One-Call API

```python
from mlx_lm.lazy_experts import flash_generate

text = flash_generate(
    "mlx-community/Qwen3-Coder-Next-4bit",
    "Your prompt here",
    max_tokens=200,
    cache_dir="~/.cache/flash-moe",     # optional: persist cache state
    profile_path="expert_profile.json",  # optional: enable pinning
)
```

### Step-by-Step API

```python
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from mlx_lm.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats
)

model, tokenizer = mlx_lm.load("mlx-community/Qwen3-Coder-Next-4bit", lazy=True)
model_path = hf_repo_to_path("mlx-community/Qwen3-Coder-Next-4bit")

enable_lazy_experts(model, model_path, cache_capacity_per_layer=208, predictive=True)
mx.eval(model.parameters())

# Warmup: discover expert routing
mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=10, verbose=False)

# Upgrade to zero-eval predictive cache
upgrade_to_predictive(model, model_path, 208)

# Generate
output = mlx_lm.generate(model, tokenizer,
                          prompt="Write a Flask web server",
                          max_tokens=200, verbose=False)
print(output)
print(get_fallback_stats(model))
```

## Known Limitations

- **Cold start**: 60s minimum with cache persistence (12s tensor upgrade from disk)
- **Quality cliff below capacity 192**: Garbled output regardless of warmup (Qwen3-Coder-Next)
- **Metal pressure cliff above capacity 208**: 3x eval degradation on 32 GB Macs. Use `cache_limit(0)` to push to ~240
- **Mild repetition at 1000+ tokens**: Sentence-level topic looping (model limitation, not flash-moe)
- **Chat template required**: Some models (GLM-4.7-Flash) need the chat template for coherent output
- **Not thread-safe**: MLX GPU eval is single-threaded. Use cooperative patterns, not background threads

## License

MIT
