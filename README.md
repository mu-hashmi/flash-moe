# flash-moe

Run large Mixture-of-Experts models on memory-constrained Macs by loading only router-selected experts on demand from SSD. A 46 GB Qwen3-Coder-Next model runs on a 32 GB Mac at 6-23 tok/s using 19 GB, with coherent output through 1000+ tokens.

## Quick Start

```bash
git clone https://github.com/mu-hashmi/flash-moe.git
cd flash-moe
uv sync
```

```python
uv run python -c "
from flash_moe import flash_generate
print(flash_generate('mlx-community/Qwen3-Coder-Next-4bit',
                      'Write a Python hello world program',
                      max_tokens=200))
"
```

Streaming:

```python
uv run python -c "
from flash_moe import flash_stream_generate
for response in flash_stream_generate('mlx-community/Qwen3-Coder-Next-4bit',
                                       'Write a Flask server', max_tokens=200):
    print(response.text, end='', flush=True)
"
```

Multi-turn sessions:

```python
uv run python -c "
from flash_moe import FlashSession
session = FlashSession('mlx-community/Qwen3-Coder-Next-4bit',
                       cache_dir='~/.cache/flash-moe')
for response in session.stream('Write a linked list in Python'):
    print(response.text, end='', flush=True)
print()
print(session.generate('Now add type hints'))
"
```

First launch downloads the model (~24 GB) and warms up experts (~13s). Subsequent launches: ~6s to first token.

## Supported Models

| Model | Experts | Top-K | MoE Layers | Memory | tok/s |
|-------|--------:|------:|-----------:|-------:|------:|
| [Qwen3-Coder-Next-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit) | 512 | 10 | 48 | 19.1 GB | 8-23 |
| [Mixtral-8x7B-Instruct-v0.1-4bit](https://huggingface.co/mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit) | 8 | 2 | 32 | 19.9 GB | 4.5 |
| [GLM-4.7-Flash-4bit](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) | 64 | 4 | 46 | 13.0 GB | 5.7 |

Any MLX model using `SwitchGLU` is supported — covers Qwen, Mixtral, GLM, DeepSeek, Hunyuan, PhiMoE, Jamba, OLMoE, MiniMax, and GraniteMoE families. Both stacked and per-expert safetensors formats are handled automatically.

## Hardware Requirements

| System RAM | Capacity | Memory | tok/s | Notes |
|-----------:|---------:|-------:|------:|-------|
| 32 GB | 208 | 19 GB | 8-23 | Recommended |
| 48 GB | 320 | ~28 GB | 15-25 | Estimated |
| 64 GB | 432 | ~37 GB | 20-30 | Estimated |
| 128 GB | 512 | ~44 GB | 30-40 | Full expert coverage |

"Capacity" = experts cached per MoE layer. Auto-selected based on available RAM.

## How It Works

Each MoE layer has hundreds of experts but only routes to a few per token (e.g. 10 out of 512 for Qwen). flash-moe replaces the expert weight modules with lazy-loading versions that:

1. **Load the model without expert weights** (~1.4 GB instead of 46 GB)
2. **Discover which experts matter** via router-only forward pass (~1s)
3. **Load only those experts** into GPU-resident stacked tensors (~15s)
4. **Generate with zero-eval dispatch** — no `mx.eval()` in the forward pass, preserving MLX's async pipeline

### Expert Profiles

Pre-computed profiles in `profiles/` identify universally-activated experts per model. These are determined by the model's router weights (same for everyone) and serve two purposes:

- **Pinning** — universal experts stay in cache permanently, preventing quality degradation at 300+ tokens
- **Fast cold start** — skip the discovery step entirely (13s instead of 17s)

Profiles are shipped for Qwen, Mixtral, and GLM. Generate profiles for other models with:

```bash
uv run python benchmarks/profile_experts.py --model mlx-community/Some-MoE-Model-4bit
```

## Performance

### Qwen3-Coder-Next-4bit on 32 GB Mac

| Config | tok/s | Repetition @ 1000 tokens |
|--------|------:|-------------------------:|
| Baseline | 8.8 | 0.20 (degrades at ~250) |
| + Pinning | 8.7 | 0.03 |
| + Wired limit | 10.6 | 0.03 |
| + All optimizations (burst) | ~23 | 0.03 |

### Startup Times

| Scenario | Time |
|----------|-----:|
| Warm start (prepacked weights exist) | ~6s |
| Cold start (with profile) | ~13s |
| Cold start (no profile, router discovery) | ~17s |
| Domain switch (delta warmup) | ~2-3s |

### Multi-Turn Stability

Tested over 20 turns with varied prompts:
- Memory growth: **+0.00 GB** (19.04 GB flat)
- Speed: 5-20 tok/s (improves as caches warm)
- No KV cache leak, no degradation wall

## API

### `flash_generate(model_name, prompt, **kwargs) -> str`

One-call generation. Auto-selects capacity, loads cached state, applies pinning.

```python
text = flash_generate(
    "mlx-community/Qwen3-Coder-Next-4bit",
    "Your prompt",
    max_tokens=200,
    cache_dir="~/.cache/flash-moe",                  # persist state between runs
    profile_path="profiles/qwen3-coder-next.json",    # enable pinning
)
```

### `flash_stream_generate(model_name, prompt, **kwargs) -> Generator`

Same as `flash_generate` but yields `GenerationResponse` objects token-by-token.

### `FlashSession(model_name, **kwargs)`

Reusable session for multi-turn generation. Loads the model once, runs delta warmup automatically when prompts change domain.

```python
session = FlashSession("mlx-community/Qwen3-Coder-Next-4bit",
                       cache_dir="~/.cache/flash-moe")
session.stream(prompt, max_tokens=200)   # yields GenerationResponse
session.generate(prompt, max_tokens=200) # returns str
session.memory_gb                        # current GPU memory
session.close()                          # release model
```

### Step-by-Step API

For full control over the loading pipeline:

```python
import mlx.core as mx
import mlx_lm
from mlx_lm.utils import hf_repo_to_path
from flash_moe.lazy_experts import (
    enable_lazy_experts, upgrade_to_predictive, get_fallback_stats
)

model, tokenizer = mlx_lm.load("mlx-community/Qwen3-Coder-Next-4bit", lazy=True)
model_path = hf_repo_to_path("mlx-community/Qwen3-Coder-Next-4bit")

enable_lazy_experts(model, model_path, cache_capacity_per_layer=208, predictive=True)
mx.eval(model.parameters())

mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=10, verbose=False)
upgrade_to_predictive(model, model_path, 208)

output = mlx_lm.generate(model, tokenizer, prompt="Write a Flask server",
                          max_tokens=200, verbose=False)
```

## Known Limitations

- **Quality cliff below capacity 192** — garbled output regardless of warmup (Qwen)
- **Metal pressure cliff above capacity 208** on 32 GB — 3x eval degradation
- **Mild repetition at 1000+ tokens** — model limitation at 40% expert coverage
- **Chat template required** for some models (GLM-4.7-Flash)
- **Not thread-safe** — MLX GPU eval is single-threaded

## License

MIT
