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
| + Wired limit | 19.0 GB | 10.6 | 0.03 | +21% from Metal residency |
| + cache=256MB warmup | 19.0 GB | 12.1 | 0.03 | +14% from warmup cache floor |
| Full optimized (burst) | 19.0 GB | ~23 | 0.03 | 200-token burst with all opts |
| Cap 192 (safe) | 17.7 GB | 6.1 | -- | More memory headroom |

### Startup Times

| Scenario | Time | Method |
|----------|------|--------|
| Warm start (prepacked weights) | ~6s | `load_prepacked_weights()` |
| Cold start (with profile) | ~13s | `upgrade_from_profile()` |
| Cold start (router-only discovery) | ~17s | `router_only_discovery()` + `upgrade_to_predictive()` |
| Cold start (legacy full warmup) | ~94s | 10-token generation + `upgrade_to_predictive()` |
| Cross-domain delta warmup | ~2-3s | `fast_delta_warmup()` |

## Project Structure

```
flash-moe/                          # This repo
  generate_lazy.py                  # Main generation script
  generate_persistent.py            # Cache-persistent generation
  generate_streaming.py             # Streaming generation wrapper
  test_generalization.py            # Multi-model smoke tests
  benchmarks/
    profile_experts.py              # Universal expert profiling (multi-model)
    bench_pinning.py                # Pinning benchmark
    bench_multiturn.py              # Multi-turn session memory benchmark
    bench_warmup.py                 # Warmup optimization benchmark

mlx-lm/                            # Local fork (lazy-experts branch)
  mlx_lm/lazy_experts/             # Core implementation (sub-package)
    __init__.py                    # Re-exports all public API
    core.py                        # enable/upgrade/reset, cache management
    modules.py                     # Expert cache + lazy/predictive modules
    loading.py                     # Weight loading, shard maps, capacity
    discovery.py                   # Router-only discovery, speculative probe
    warmup.py                      # Delta warmup, incremental warmup
    persistence.py                 # Cache state save/load, prepacked weights
    generate.py                    # flash_generate one-call API
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

## Memory Budget

flash-moe's working set at capacity 208 is ~19 GB on a 32 GB Mac. This leaves ~13 GB for macOS, apps, and Metal overhead. If you're running memory-heavy applications alongside (browsers with many tabs, Docker, IDEs), consider:

- **Reducing capacity to 192** (17.7 GB) for more headroom
- **Closing unnecessary apps** before long generation sessions
- The `flash_generate()` memory guard automatically reduces capacity if projected memory exceeds 85% of device RAM

## Advanced: GPU Wired Memory Limit (sysctl)

macOS caps GPU-wired memory at ~75% of system RAM (`recommendedMaxWorkingSetSize`). On a 32 GB Mac this is ~24 GB. The `set_wired_limit()` call in flash-moe respects this cap.

For power users who want to push beyond 75%, macOS exposes a sysctl knob:

```bash
# Raise GPU memory cap to ~88% of 32 GB RAM (WARNING: see risks below)
sudo sysctl iogpu.wired_limit_mb=28672

# Check current value
sysctl iogpu.wired_limit_mb

# Reset to default (or reboot — the setting doesn't persist)
sudo sysctl -d iogpu.wired_limit_mb  # removes override
```

**Risks and limitations:**

- **OS instability**: Starving macOS of physical RAM can cause system-wide stalls, UI freezes, or kernel panics under memory pressure. The 75% default exists for a reason.
- **Does not persist across reboots**: Must be re-applied after every restart.
- **Diminishing returns**: Our testing shows capacity >208 on 32 GB hits a Metal pressure cliff (3x eval degradation) regardless of the wired limit. The bottleneck shifts to Metal's internal scheduling, not RAM availability.
- **No undo mid-session**: Once buffers are wired at the higher limit, releasing them requires stopping the process.

**Recommendation**: Only use this if you have 48+ GB RAM and want to run higher capacities. On 32 GB, stick with capacity 208 — the wired limit override provides no measurable benefit.

## Streaming Generation

For token-by-token streaming (chat UIs, coding agents):

```python
from generate_streaming import flash_stream_generate

for response in flash_stream_generate(
    "mlx-community/Qwen3-Coder-Next-4bit",
    "Write a Flask server",
    max_tokens=200,
    cache_dir="~/.cache/flash-moe",
):
    print(response.text, end="", flush=True)
```

Startup (model load + expert warmup) is blocking. Tokens stream after startup completes. Time to first token: ~6s warm / ~13s cold.

## Multi-Turn Sessions

For multi-turn use (coding agents, chatbots), reuse the model across turns. The `bench_multiturn.py` benchmark tests memory stability and generation quality over 20+ turns.

```bash
/path/to/mlx-lm/.venv/bin/python benchmarks/bench_multiturn.py --model qwen --turns 20
```

Monitors: active memory growth, generation speed drift, repetition score, and fallback rate across turns. Reports warnings if memory grows unboundedly or speed degrades.

## Known Limitations

- **Quality cliff below capacity 192**: Garbled output regardless of warmup (Qwen3-Coder-Next)
- **Metal pressure cliff above capacity 208**: 3x eval degradation on 32 GB Macs. `set_wired_limit` does not extend this ceiling.
- **Mild repetition at 1000+ tokens**: Sentence-level topic looping (model capacity limitation at 40% expert coverage, not flash-moe)
- **Chat template required**: Some models (GLM-4.7-Flash) need the chat template for coherent output
- **Not thread-safe**: MLX GPU eval is single-threaded. Use cooperative patterns, not background threads
- **MLX version sensitivity**: Uses `mx.metal.*` (soon `mx.*`) APIs. Future MLX releases may require updates to import paths.

## License

MIT
