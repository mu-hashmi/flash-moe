# Runtime Architecture

[Back to Architecture Overview](architecture.md)

This page covers predictive forward dispatch, skip-fallback behavior, and dynamic cache updates.

## Predictive Forward Path

```mermaid
sequenceDiagram
    participant Router as MoE Router
    participant PSL as PredictiveCachedSwitchLinear
    participant PEC as PredictiveExpertCache
    participant QMM as mx.gather_qmm

    Router->>PSL: expert indices
    PSL->>PEC: append indices buffer (up_proj only)
    PSL->>PEC: remap(indices) using lookup
    PEC-->>PSL: local slot indices
    PSL->>QMM: gather_qmm(x, stacked tensors, local slots)
    QMM-->>PSL: projected output
```

Key invariant: predictive module `__call__` stays lazy and does not call `mx.eval()`.

## Skip-Fallback Path

```mermaid
flowchart LR
    A["Router indices + scores"] --> B["hit_mask[indices]"]
    B --> C["zero scores for misses"]
    C --> D["renormalize remaining scores"]
    D --> E["mix switch_mlp outputs"]
    E --> F["add shared expert contribution"]
```

`enable_skip_fallback(...)` patches MoE block call paths so uncached experts do not route to wrong-expert slot outputs.

## Dynamic Cache Update

```mermaid
flowchart TD
    A["Between tokens"] --> B["dynamic_cache_update(...)"]
    B --> C["skip in-flight buffer tail"]
    C --> D["collect requested expert IDs"]
    D --> E["update frequency/recency + fallback stats"]
    E --> F{"misses and swap budget available?"}
    F -- no --> G["stats only"]
    F -- yes --> H["choose evictions (LCP, avoid pinned/currently requested)"]
    H --> I["load replacement experts"]
    I --> J["scatter update stacked weights/scales/biases"]
    J --> K["rebuild lookup + hit_mask"]
```

Swap control:

- Global per-call limit: `max_layer_updates`.
- Per-layer cap inside cache update path.

## Runtime Loop

```mermaid
flowchart LR
    A["Token t forward"] --> B["capture router indices"]
    B --> C["emit token t"]
    C --> D["dynamic_cache_update between tokens"]
    D --> E["Token t+1 forward"]
```
