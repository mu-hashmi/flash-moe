# Mixtral Model Compatibility Gap Analysis

## Executive Summary

Mixtral 8x7B is **highly compatible** with flash-moe's lazy expert loading approach. The architectural differences from Qwen3-Coder-Next are minor and well-contained. Both models use the same `SwitchGLU` container with the same 3-projection layout (`gate_proj`, `up_proj`, `down_proj`), dispatched via `gather_qmm`. The primary work to support Mixtral is addressing the different module path (`block_sparse_moe` vs `mlp`) and the hardcoded `num_experts=512` assumption. No fundamental redesign is required.

## 1. Mixtral Architecture in mlx-lm

Source: `/Users/muhash/mlx-lm/mlx_lm/models/mixtral.py`

### MoE Structure

| Property | Mixtral 8x7B | Qwen3-Coder-Next-4bit |
|----------|-------------|----------------------|
| Expert count (`num_local_experts`) | 8 | 512 |
| Experts per token (`num_experts_per_tok`) | 2 | 10 |
| Hidden size | 4096 | 2048 |
| Intermediate (per-expert FFN) size | 14336 | 512 |
| MoE layers | 32 (all layers) | 48 (sparse; every `decoder_sparse_step` layer) |
| Shared expert | No | Yes (`shared_expert` + `shared_expert_gate`) |
| Expert container | `SwitchGLU` | `SwitchGLU` |
| Router | `nn.Linear(4096, 8, bias=False)` | `nn.Linear(2048, 512, bias=False)` |
| Routing activation | softmax after top-k selection | softmax before top-k selection |
| MoE block class | `MixtralSparseMoeBlock` | `Qwen3NextSparseMoeBlock` |
| MoE block attr name | `layer.block_sparse_moe` | `layer.mlp` |

### Key Observation: Identical Expert Dispatch

Both models use the exact same expert execution path:

```
MoE block → gate(x) → top-k indices → SwitchGLU(x, indices)
                                         ├── up_proj(x, indices)   → QuantizedSwitchLinear → gather_qmm
                                         ├── gate_proj(x, indices) → QuantizedSwitchLinear → gather_qmm
                                         └── down_proj(activation, indices) → QuantizedSwitchLinear → gather_qmm
```

The `SwitchGLU` class (in `switch_layers.py`) is shared across both models. After quantization, both models have `QuantizedSwitchLinear` modules as their leaf expert layers with the same 3D weight tensors: `(num_experts, output_dims, input_dims)`.

### Mixtral Weight Format

The `sanitize()` method in `mixtral.py` (lines 195-211) converts per-expert weights to stacked format:

```python
# Original HuggingFace layout (per-expert):
#   model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight  (gate_proj)
#   model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight  (down_proj)
#   model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight  (up_proj)

# After sanitize → stacked for SwitchGLU:
#   model.layers.{l}.block_sparse_moe.switch_mlp.gate_proj.weight
#   model.layers.{l}.block_sparse_moe.switch_mlp.down_proj.weight
#   model.layers.{l}.block_sparse_moe.switch_mlp.up_proj.weight
```

Note the naming: `w1 → gate_proj`, `w2 → down_proj`, `w3 → up_proj`. This is the same SwitchGLU projection naming as Qwen3-Next.

### Module Path Difference

This is the main structural difference:

```
# Qwen3-Coder-Next:
model.layers[i].mlp.switch_mlp.{gate_proj,up_proj,down_proj}
#             ^^^^
# Mixtral:
model.layers[i].block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}
#             ^^^^^^^^^^^^^^^^^^
```

In Qwen3-Next, the MoE block IS `layer.mlp` (an instance of `Qwen3NextSparseMoeBlock`). In Mixtral, the MoE block is `layer.block_sparse_moe` (an instance of `MixtralSparseMoeBlock`), while `layer.mlp` does not exist.

### Per-Expert Size Estimate

Mixtral's experts are much larger than Qwen3-Next's:

- Qwen3-Next: `moe_intermediate_size=512`, hidden=2048 → ~1.69 MB/expert
- Mixtral: `intermediate_size=14336`, hidden=4096 → at 4-bit quantization, approximately:
  - gate_proj: (14336 x 4096) / 2 bytes = ~29.4 MB
  - up_proj: same = ~29.4 MB
  - down_proj: (4096 x 14336) / 2 bytes = ~29.4 MB
  - Total per expert: ~88 MB (plus scales/biases overhead)
  - All 8 experts: ~704 MB
  - 4-bit quantized total model: ~24 GB (8x7B params at 4 bits)

With only 8 experts, you could plausibly hold all experts for most layers in memory. The interesting use case is when memory is constrained below the full model size.

### Available Quantized Models on HuggingFace

4-bit MLX-format Mixtral models exist and are confirmed working with mlx-lm:

- `mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx`
- `mlx-community/Mixtral-8x7B-v0.1-hf-4bit-mlx`
- `mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit`

These use the standard mlx-lm quantization format (safetensors shards with `model.safetensors.index.json`), which is exactly what flash-moe's `_build_shard_map()` expects.

## 2. What Is Model-Specific in lazy_experts.py

### Model-Specific Code (Must Change for Mixtral)

**A. Module path traversal** — hardcoded in every function that walks MoE layers:

```python
# Current (lines 532-535, repeated ~30 times):
if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "switch_mlp"):
    continue
switch = layer.mlp.switch_mlp
```

For Mixtral, this should be:
```python
if not hasattr(layer, "block_sparse_moe") or not hasattr(layer.block_sparse_moe, "switch_mlp"):
    continue
switch = layer.block_sparse_moe.switch_mlp
```

**B. Weight key prefix** — hardcoded safetensor key paths:

```python
# Current (line 540):
key_prefix = f"model.layers.{i}.mlp.switch_mlp.{name}"

# Mixtral needs:
key_prefix = f"model.layers.{i}.block_sparse_moe.switch_mlp.{name}"
```

**C. `num_experts=512` assumption** — hardcoded in `PredictiveExpertCache` and `upgrade_to_predictive`:

```python
# PredictiveExpertCache.__init__ (line 269):
def __init__(self, capacity: int, num_experts: int = 512):

# upgrade_to_predictive (line 659):
num_experts = 512
```

Mixtral has `num_local_experts=8`. The default of 512 would waste memory on a 512-element lookup table when only 8 entries are needed. Not a correctness bug (the lookup table would still work), but wasteful and confusing.

**D. `router_only_forward` and `speculative_router_*`** — import and isinstance-check `Qwen3NextSparseMoeBlock`:

```python
# Line 1794:
from .models.qwen3_next import Qwen3NextSparseMoeBlock
```

These functions would need to be parameterized or duplicated for `MixtralSparseMoeBlock`.

**E. Shared expert handling** — `router_only_forward` falls back to `self.shared_expert(x)`:

```python
# Line 1821:
shared_y = self.shared_expert(x)
shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
return shared_y
```

Mixtral has no shared expert. The skip-MoE fallback would need to return zeros or skip the MoE output entirely.

### Model-Agnostic Code (Works as-is for Mixtral)

**A. `SwitchGLU` / `QuantizedSwitchLinear` replacement** — the core mechanism:

- `LazyQuantizedSwitchLinear.__call__` — loads from shard, slices experts by ID, remaps, calls `gather_qmm`. Fully generic.
- `CachedQuantizedSwitchLinear.__call__` — same, with LCP cache. Fully generic.
- `PredictiveCachedSwitchLinear.__call__` — GPU lookup + pre-stacked weights + `gather_qmm`. Fully generic.
- All three are drop-in replacements for `QuantizedSwitchLinear` and depend only on the `gather_qmm` interface.

**B. Expert cache classes** — `ExpertCache`, `PredictiveExpertCache`, `IncrementalDeltaWarmup`:

- LCP eviction policy, frequency/recency tracking, scatter-based tensor updates — all operate on slot indices and expert IDs. No model-specific logic.

**C. Shard loading** — `_build_shard_map()` reads `model.safetensors.index.json`. Works for any model using the standard mlx-lm safetensors layout.

**D. `gather_qmm` dispatch** — the core MLX op that does quantized expert matmul. Model-agnostic by design.

**E. Delta warmup logic** — `delta_warmup`, `fast_delta_warmup`, `IncrementalDeltaWarmup` — all operate on the abstract cache/module interface. The only model-specific parts are the module path traversal patterns listed above.

## 3. Required Changes for Mixtral Support

### Minimal Changes (Estimated Effort: Small)

1. **Extract a `find_switch_mlp(layer)` helper** that returns `(switch_mlp_module, key_prefix_base)` or `None`. This replaces the repeated `hasattr(layer, "mlp") or hasattr(layer.mlp, "switch_mlp")` pattern. Implementation:

   ```python
   def _find_switch_mlp(layer, layer_idx):
       """Find the SwitchGLU module and weight key prefix for an MoE layer."""
       # Qwen3-Next, Qwen2-MoE, DeepSeek, etc.
       if hasattr(layer, "mlp") and hasattr(layer.mlp, "switch_mlp"):
           return layer.mlp.switch_mlp, f"model.layers.{layer_idx}.mlp.switch_mlp"
       # Mixtral, PhiMoE, MiniMax, GraniteMoE
       if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "switch_mlp"):
           return layer.block_sparse_moe.switch_mlp, f"model.layers.{layer_idx}.block_sparse_moe.switch_mlp"
       return None
   ```

   This single helper covers both Mixtral-family and Qwen-family models. It also covers PhiMoE, MiniMax, and GraniteMoE which use the same `block_sparse_moe` path.

2. **Parameterize `num_experts`** — read from the model's config or detect from the `QuantizedSwitchLinear.num_experts` property (available at line 73 of `switch_layers.py`):

   ```python
   num_experts = orig.num_experts  # already a property on QuantizedSwitchLinear
   ```

3. **Make `router_only_forward` model-agnostic** — instead of importing a specific MoE block class, detect MoE blocks by checking for `switch_mlp` attribute. For the fallback output, check `hasattr(block, "shared_expert")` and use zeros if absent.

### No Changes Needed

- The `SwitchGLU` → `QuantizedSwitchLinear` → `gather_qmm` pipeline is identical.
- The 3-projection structure (`gate_proj`, `up_proj`, `down_proj`) is identical.
- The safetensors weight format (after `sanitize()`) is identical.
- The cache eviction, scatter update, and lookup table mechanisms are fully generic.

## 4. Practical Considerations for Mixtral

### Is Lazy Expert Loading Useful for Mixtral?

Mixtral 8x7B has only 8 experts per layer (vs Qwen3-Next's 512). At 4-bit quantization, total expert weight per layer is ~704 MB. The full model at 4-bit is ~24 GB.

**32 GB Mac scenario:**
- Non-expert weights (attention, embeddings, norms): ~3 GB
- Expert weights (32 layers x 8 experts x ~88 MB): ~22 GB
- Total: ~25 GB — fits, but tight with KV cache headroom

Lazy loading could help by caching only top-2 experts per layer (~176 MB/layer x 32 = ~5.6 GB), bringing total to ~8.6 GB. But with only 8 experts and top-2 routing, the hit rate would be low (8 experts, 2 active = 25% chance of reuse) unless expert selection is highly correlated across tokens (which it often is).

The more compelling Mixtral use case is **16 GB Macs** or running alongside other applications on 32 GB. With cache capacity of 4 experts/layer (50% of 8), memory drops from ~25 GB to ~14 GB.

### Capacity Selection

With only 8 experts, the capacity knob is much coarser:

| Capacity | Memory (est.) | Coverage | Notes |
|----------|--------------|----------|-------|
| 2 | ~8.6 GB | 25% | High fallback, likely poor quality |
| 4 | ~14.2 GB | 50% | Moderate quality, main target for 16 GB Macs |
| 6 | ~19.8 GB | 75% | Good quality, fits 32 GB with headroom |
| 8 | ~25.4 GB | 100% | Full model, no offloading needed |

### Performance Implications

- **Per-expert size is 52x larger** than Qwen3-Next (88 MB vs 1.69 MB). Shard loads will take proportionally longer.
- **Fewer experts means simpler routing** — the lookup table is trivial (8 entries) and cache management has almost no overhead.
- **Delta warmup is fast** — with only 8 experts per layer, the maximum number of misses per layer is 6 (if caching 2). Total swaps across 32 layers: up to 192. Compare to Qwen3-Next's thousands of swaps across 48 layers.
- **The Metal memory pressure cliff still applies** — if the cached subset puts you at 20+ GB on a 32 GB Mac, scatter eval will be slow.

## 5. Broader Generalization

The SwitchGLU pattern is used by a large number of MoE models in mlx-lm (found 25+ models). They fall into two families based on module path:

**`layer.mlp.switch_mlp` family:**
- Qwen3-Coder-Next, Qwen3-MoE, Qwen2-MoE
- DeepSeek v2/v3/v3.2
- Hunyuan, Jamba, OLMoE, Ernie, LFM2, Kimi, Step3.5, BailingMoE, DOTS1, Klear
- LongcatFlash, GLM4-MoE, ExaoneMoE, MIMO

**`layer.block_sparse_moe.switch_mlp` family:**
- Mixtral, PhiMoE, MiniMax, GraniteMoE, GraniteMoEHybrid

**Other (different attribute name):**
- AFMoE, GPT-OSS, LLaMA 4 — use `self.experts = SwitchGLU(...)` instead of `self.switch_mlp`

The `_find_switch_mlp` helper described above covers the first two families. Adding support for the third family (`.experts` attribute) would require one more `hasattr` check.

One outlier: `Phixtral` and `Nemotron-H` use `SwitchMLP` (2 projections: `fc1`, `fc2`) instead of `SwitchGLU` (3 projections: `gate_proj`, `up_proj`, `down_proj`). These would need a different replacement strategy since the projection count differs.

## 6. Summary

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Expert dispatch (`gather_qmm`) | Identical | Same `QuantizedSwitchLinear` |
| Expert container (`SwitchGLU`) | Identical | Same 3 projections, same call order |
| Weight format (safetensors) | Identical | Same shard layout after `sanitize()` |
| Module path | **Different** | `block_sparse_moe` vs `mlp` |
| Weight key prefix | **Different** | Follows module path |
| Expert count | **Different** | 8 vs 512 (hardcoded in several places) |
| Shared expert | **Absent in Mixtral** | Affects `router_only_forward` fallback |
| Available 4-bit models | Yes | `mlx-community/Mixtral-8x7B-*-4bit-mlx` |

**Conclusion:** The core lazy expert mechanism (replace `QuantizedSwitchLinear`, load on demand from safetensors, cache with LCP eviction, scatter-update between tokens) generalizes to Mixtral with minimal changes. The required work is extracting a module-path helper function, parameterizing expert count, and making the router-only discovery functions model-agnostic. No changes to the caching, eviction, scatter, or `gather_qmm` dispatch code are needed.
