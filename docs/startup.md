# Startup Architecture

[Back to Architecture Overview](architecture.md)

This page covers `_startup()` and cache materialization flow in `mlx_moe/lazy_experts/generate.py`.

## Startup Goals

- Load lazily and choose a safe capacity.
- Build predictive expert state from the fastest valid source.
- Finalize runtime safety knobs (skip-fallback + wired limit).

## Startup Decision Tree

```mermaid
flowchart TD
    A["mlx_lm.load(..., lazy=True)"] --> B["Detect MoE + estimate slot size (weight/scales/biases)"]
    B --> C{"capacity provided?"}
    C -- no --> D["enable_lazy_experts(cap=0) + mx.eval(params) + select_capacity(71% recommended)"]
    C -- yes --> E["Use provided capacity"]
    D --> F["enable_lazy_experts(cap=C, predictive=True) + mx.eval(params)"]
    E --> F
    F --> G["Build shard_map + SafetensorsMap"]
    G --> H{"prepacked weights exist?"}

    H -- yes --> I["load_prepacked_weights(...)"]
    I --> J{"saved prompt != current prompt?"}
    J -- yes --> K["fast_delta_warmup(...)"]
    J -- no --> X["saved path used"]
    K --> X

    H -- no --> L{"cache-state JSON exists?"}
    L -- yes --> M["upgrade_from_saved_state(...)"]
    M --> N{"saved prompt != current prompt?"}
    N -- yes --> K
    N -- no --> X

    L -- no --> O{"fresh startup mode"}
    O -- warmup=full --> P["short generate + upgrade_to_predictive(...)"]
    O -- profile --> Q["upgrade_from_profile(...)"]
    O -- otherwise --> R["router_only_discovery(...) + upgrade_to_predictive(...)"]
    P --> S{"warmup=hybrid?"}
    Q --> S
    R --> S
    S -- yes --> T["short stream_generate + dynamic_cache_update(...)"]
    S -- no --> U["skip hybrid refinement"]
    T --> V["save cache-state / prepacked when newly built"]
    U --> V

    X --> W["enable_skip_fallback(...)"]
    V --> W
    W --> Y["mx.set_wired_limit(min(active, 0.75 * device_memory))"]
```

Key points:

- Persistence precedence is `prepacked -> cache-state -> fresh build`.
- `enable_lazy_experts(..., predictive=True)` installs phase-2 cached modules first; predictive modules are installed by upgrade.
- Fresh path is `full`, `profile`, or `router-only discovery` depending on flags.
- Hybrid refinement runs only on fresh path inside `_startup()`.
- Wiring is always the final startup action when supported.

## Module Replacement

```mermaid
flowchart LR
    A["QuantizedSwitchLinear"] --> B["LazyQuantizedSwitchLinear"]
    B --> C["CachedQuantizedSwitchLinear (ExpertCache)"]
    C --> D["PredictiveCachedSwitchLinear (PredictiveExpertCache)"]
```

```mermaid
flowchart TD
    L["MoE Layer"] --> GP["gate_proj"]
    L --> UP["up_proj"]
    L --> DP["down_proj"]
    GP --> EC["Shared per-layer ExpertCache"]
    UP --> EC
    DP --> EC
```

### Supported MoE Paths

- `layer.mlp.switch_mlp`
- `layer.block_sparse_moe.switch_mlp`

### Why the Chain Exists

- `LazyQuantizedSwitchLinear` defers expert tensor loading to on-demand shard access.
- `CachedQuantizedSwitchLinear` introduces per-layer LCP caches shared by `gate_proj`, `up_proj`, and `down_proj`.
- `PredictiveCachedSwitchLinear` removes forward-path sync points by dispatching directly from pre-stacked tensors and a GPU lookup table.

LCP priority in phase-2 caches is:

`priority = frequency * 0.25^(recency / 128)`

## Discovery and Upgrade

```mermaid
flowchart LR
    A["router_only_discovery(...)"] --> B["router + shared expert only"]
    B --> C["collect inds tensors lazily"]
    C --> D["single mx.eval(*all_tensors)"]
    D --> E["seed phase-2 cache state"]
```

```mermaid
flowchart TD
    A["Pass 1: Harvest"] --> A1["top-C by LCP priority + filler"]
    A1 --> B["Pass 2: Batched shard loading"]
    B --> B1["group by shard, load once, eval per-layer batch"]
    B1 --> C["Pass 3: Assemble + install"]
    C --> C1["stack weights/scales/biases"]
    C1 --> C2["build lookup + hit_mask"]
    C2 --> D["replace with predictive modules"]
```

### Notes

- `router_only_discovery(...)` collects router indices lazily and bulk-evals once (`mx.eval(*all_tensors)`), then seeds phase-2 cache frequency/recency.
- `upgrade_to_predictive(...)` runs harvest, batched load, and assemble/install passes.
- Uncached IDs map to slot `0`; skip-fallback masking prevents wrong-expert contamination.
- Profile startup in `_startup()` uses `upgrade_from_profile(...)` then predictive upgrade.

Pinning sources:

- `pin_top_k` for exact top-K.
- `pin_threshold` for activation-fraction threshold.

Pinned experts are excluded from eviction during dynamic updates.

## Startup Artifacts

```mermaid
flowchart LR
    A["Cache state JSON (.json)"] <--> B["_startup()"]
    C["Prepacked weights (.weights.safetensors + .meta.json)"] <--> B
```

Artifact notes:

- Cache-state JSON restores routing metadata for phase-2 reconstruction.
- Prepacked weights restore phase-3 stacked tensors directly for fastest warm starts.
