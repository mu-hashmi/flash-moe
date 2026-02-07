# Additional Model Compatibility Analysis

Analysis of three models for flash-moe lazy expert loading compatibility.

## 1. Devstral (Mistral)

### Architecture

Devstral 2 (released December 2025) is a **dense transformer**, NOT a Mixture-of-Experts model. Mistral deliberately chose a dense architecture for their coding models:

- **Devstral 2**: 123B dense parameters
- **Devstral Small 2**: 24B dense parameters

No experts, no routing, no SwitchGLU. The model treats its entire parameter set as a single coherent knowledge base.

Note: Mistral Large 3 IS an MoE model (41B active / 675B total), but that is a separate model from Devstral.

### mlx-lm Support

No `devstral` model file in `/Users/muhash/mlx-lm/mlx_lm/models/`. Devstral models likely map to `llama` or `mistral3` via `MODEL_REMAPPING` since they're dense Mistral variants.

### Compatibility Verdict: INCOMPATIBLE

Devstral is a dense model. flash-moe's lazy expert loading is designed for MoE architectures. There is nothing to offload.

---

## 2. GLM-4 / GLM-4.5 / GLM-4.7 (Zhipu AI / Z.ai)

### Architecture

The GLM-4 series includes multiple MoE models:

| Model | Total Params | Active Params | Experts | Top-K | Layers | hidden_size | moe_intermediate_size |
|-------|-------------|---------------|---------|-------|--------|-------------|----------------------|
| GLM-4.5 | 355B | 32B | 160 | 8 | 92 | 5120 | 1536 |
| GLM-4.6 | 355B | 32B | 160 | 8 | 92 | 5120 | ~1536 |
| GLM-4.7 | 355B | 32B | 160 | 8 | 92 | 5120 | 1536 |
| GLM-4.7-Flash | 30B | 3B | 64 | 4 | 47 | 2048 | 1536 |

Key config for GLM-4.7 (the flagship):
- `model_type`: `glm4_moe`
- `n_routed_experts`: 160
- `num_experts_per_tok`: 8
- `n_shared_experts`: 1
- `first_k_dense_replace`: 3 (first 3 layers are dense)
- `routed_scaling_factor`: 2.5
- `topk_method`: `noaux_tc`
- `scoring_func`: `sigmoid`

### mlx-lm Support

Two model files exist:
- `/Users/muhash/mlx-lm/mlx_lm/models/glm4_moe.py` -- for GLM-4.5/4.6/4.7 (model_type: `glm4_moe`)
- `/Users/muhash/mlx-lm/mlx_lm/models/glm4_moe_lite.py` -- for GLM-4.7-Flash (model_type: `glm4_moe_lite`)

Both files import `SwitchGLU` from `switch_layers` and use the **exact same MoE pattern** as flash-moe's target.

**Module path**: `layer.mlp.switch_mlp` (verified in source code)

From `glm4_moe.py` line 197:
```python
class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)
```

Decoder layers: `layer.mlp = MoE(config)` for MoE layers (lines 232-239).

Router: `MoEGate` -- uses `sigmoid` scoring + `e_score_correction_bias` + group expert selection, same as DeepSeek V3.

### HuggingFace Quantized Models

Available on mlx-community:
- `mlx-community/GLM-4.5-4bit`
- `mlx-community/GLM-4.6-4bit`
- `mlx-community/GLM-4.7-4bit`
- `mlx-community/GLM-4.7-Flash-4bit`

### Per-Expert Size Estimate

**GLM-4.7 (355B, 160 experts)**:
- Each expert has 3 projections (gate/up/down) of shape related to hidden_size=5120 and moe_intermediate_size=1536
- gate_proj: [5120, 1536] = 7.86M elements
- up_proj: [5120, 1536] = 7.86M elements
- down_proj: [1536, 5120] = 7.86M elements
- Total per expert: 23.6M elements = ~11.8 MB at fp16, ~5.9 MB at 4-bit
- Total expert weight: 160 experts x 89 MoE layers x 5.9 MB = ~84 GB at 4-bit

**GLM-4.7-Flash (30B, 64 experts)**:
- gate/up/down: [2048, 1536] / [1536, 2048]
- Per expert: 3 x 2048 x 1536 = 9.44M elements = ~4.7 MB at fp16, ~2.4 MB at 4-bit
- Total expert weight: 64 experts x 46 MoE layers x 2.4 MB = ~7.1 GB at 4-bit

### Weight Key Prefix

The `glm4_moe.py` model uses standard structure: `model.layers.{i}.mlp.switch_mlp.{name}`. The sanitize method stacks individual expert weights into the switch_mlp format:
```python
weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
```

This matches exactly what `lazy_experts.py` expects with `key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"`.

### Compatibility Verdict: DROP-IN COMPATIBLE

GLM-4 MoE models (all variants) use the identical `layer.mlp.switch_mlp` pattern with `SwitchGLU` and the same weight key layout as Qwen3-Next. The `enable_lazy_experts()` function would work without any code changes.

The GLM-4.7-Flash (30B) is particularly interesting as a test target -- at ~7.1 GB expert weight at 4-bit, it's much more manageable than the 355B models.

**Changes needed**: None for the basic lazy loading path. The router uses `MoEGate` (sigmoid + e_score_correction_bias) instead of `nn.Linear`, so `model.layers[i].mlp.gate` is a `MoEGate` instance rather than a raw `nn.Linear`. The `PredictiveExpertCache` code in lazy_experts.py that accesses the router would need to handle this (calling `gate(x)` returns `(inds, scores)` directly rather than needing to do softmax + topk externally). However, this only affects the predictive cache warmup path, not the basic lazy loading.

---

## 3. Kimi-K2.5 (Moonshot AI)

### Architecture

Kimi K2.5 (released January 2026) is a 1T-parameter MoE model:

| Parameter | Value |
|-----------|-------|
| model_type | kimi_k25 |
| Total params | ~1.04T |
| Active params | 32B |
| hidden_size | 7168 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 384 |
| num_experts_per_tok | 8 |
| n_shared_experts | 1 |
| num_hidden_layers | 61 |
| first_k_dense_replace | 1 |
| moe_layer_freq | 1 |
| topk_method | noaux_tc |

K2.5 adds vision capabilities and subagent spawning over K2.

### mlx-lm Support

Model file: `/Users/muhash/mlx-lm/mlx_lm/models/kimi_k25.py`

This file is a **thin wrapper around DeepSeek V3**:
```python
from .deepseek_v3 import DeepseekV3Model
from .deepseek_v3 import Model as DeepseekV3LM
from .deepseek_v3 import ModelArgs as TextConfig
```

The actual MoE logic lives in `deepseek_v3.py`, which uses `SwitchGLU` at line 258:
```python
class DeepseekV3MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
```

The `kimi_k2` model type (Kimi K2 without .5) is remapped to `deepseek_v3` in `MODEL_REMAPPING`.

There's also `kimi_linear.py` for the Kimi-Linear variant which has its own MoE class (`KimiSparseMoE`) but also uses `SwitchGLU` via `self.switch_mlp`.

### HuggingFace Quantized Models

Available:
- `mlx-community/Kimi-K2-Instruct-4bit` (K2, not K2.5)
- `mlx-community/Kimi-K2-Thinking-4bit`
- `mlx-community/Kimi-K2.5` (converted with mlx-lm 0.30.5)
- `inferencerlabs/Kimi-K2.5-MLX-4.2bit`
- `inferencerlabs/Kimi-K2.5-MLX-3.6bit`

### Per-Expert Size Estimate

- gate_proj: [7168, 2048] = 14.7M elements
- up_proj: [7168, 2048] = 14.7M elements
- down_proj: [2048, 7168] = 14.7M elements
- Total per expert: 44.1M elements = ~22 MB at fp16, ~11 MB at 4-bit
- Total expert weight: 384 experts x 60 MoE layers x 11 MB = ~253 GB at 4-bit

This is a very large model. Even at 4-bit, the expert weights alone are ~253 GB.

### Module Path and Weight Key Issue

The module path through Kimi K2.5's model structure:
```
Model → language_model → model (DeepseekV3Model) → layers[i] → mlp → switch_mlp
```

The `Model.layers` property returns `self.language_model.model.pipeline_layers`, so `model.layers[i].mlp.switch_mlp` works at the Python level.

**However**, the safetensors weight keys are prefixed with `language_model.model.layers.{i}.mlp.switch_mlp.{name}` (because the nesting goes through `language_model.model`), while `lazy_experts.py` hardcodes `key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"`.

This is the same kind of prefix mismatch. The `sanitize()` method in `kimi_k25.py` handles this during initial loading, but `lazy_experts.py` constructs the key prefix independently when loading expert shards from safetensors files.

### Compatibility Verdict: MINOR CHANGES NEEDED

The MoE architecture is identical to DeepSeek V3 (which uses the same `layer.mlp.switch_mlp` SwitchGLU pattern). Two issues:

1. **Weight key prefix**: `lazy_experts.py` hardcodes `"model.layers.{i}..."` but Kimi K2.5 safetensors use `"language_model.model.layers.{i}..."`. Fix: make the key prefix configurable or detect it from the model's weight keys.

2. **Router access**: Same as GLM-4, the router is a `MoEGate` (not `nn.Linear`), so predictive cache warmup code that directly accesses `model.layers[i].mlp.gate` would need to know that `gate(x)` returns `(inds, scores)` rather than raw logits.

3. **Scale**: At ~253 GB expert weights (4-bit), this model requires significant storage. With 384 experts and top-8 routing, the expert coverage ratio is 8/384 = 2.1%, much sparser than Qwen3-Next's 10/512 = 2.0%. The cache capacity math is similar.

---

## Summary Table

| Model | MoE? | Experts | Top-K | mlx-lm? | SwitchGLU? | Module Path | 4-bit Expert Size | Quantized on HF? | Compatibility |
|-------|------|---------|-------|---------|------------|-------------|-------------------|-------------------|---------------|
| Devstral 2 | No (dense) | N/A | N/A | No (likely llama) | N/A | N/A | N/A | N/A | **Incompatible** |
| GLM-4.7 | Yes | 160 | 8 | Yes (`glm4_moe.py`) | Yes | `layer.mlp.switch_mlp` | ~5.9 MB/expert | Yes (mlx-community) | **Drop-in** |
| GLM-4.7-Flash | Yes | 64 | 4 | Yes (`glm4_moe_lite.py`) | Yes | `layer.mlp.switch_mlp` | ~2.4 MB/expert | Yes (mlx-community) | **Drop-in** |
| Kimi-K2.5 | Yes | 384 | 8 | Yes (`kimi_k25.py` -> `deepseek_v3.py`) | Yes | `layer.mlp.switch_mlp` | ~11 MB/expert | Yes (mlx-community) | **Minor changes** |

### Recommended Test Targets

1. **GLM-4.7-Flash** (30B, 64 experts) -- drop-in compatible, small enough to test easily. Only ~7.1 GB expert weight at 4-bit. The entire 4-bit model likely fits in 32 GB with room for lazy loading experiments.

2. **GLM-4.7** (355B, 160 experts) -- drop-in compatible, large enough to benefit significantly from lazy loading. At ~84 GB expert weight (4-bit), this is a clear win for flash-moe on memory-constrained Macs.

3. **Kimi-K2.5** (1T, 384 experts) -- needs minor key prefix fix. Shares DeepSeek V3 architecture, so any fix here also enables DeepSeek V3 support. The model is very large (253 GB expert weight at 4-bit).

### Required Code Changes for Full Coverage

1. **Weight key prefix** (for Kimi-K2.5 and other wrapper models): Make `enable_lazy_experts()` detect or accept the safetensors key prefix instead of hardcoding `"model.layers.{i}..."`.

2. **Router interface** (for GLM-4, Kimi-K2.5, DeepSeek V3 families): The `PredictiveExpertCache` warmup code assumes the router is an `nn.Linear` (`model.layers[i].mlp.gate`). For these models, `gate` is a `MoEGate` class that takes raw hidden states and returns `(inds, scores)` directly (includes sigmoid + bias correction + group selection). The warmup code should support both interfaces.
