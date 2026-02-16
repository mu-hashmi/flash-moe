# Memory and Capacity

[Back to Architecture Overview](README.md)

This page covers capacity math, working-set targeting, and runtime memory behavior.

## Capacity Selection

`select_capacity(...)` targets 71% of Metal's recommended working-set size:

```text
target_gb = 0.71 * max_recommended_working_set_size_gb
slot_gb = num_moe_layers * expert_slot_mb / 1024
capacity = floor((target_gb - base_memory_gb) / slot_gb)
```

`expert_slot_mb` includes `weight + scales + biases`.

## Memory Control Flow

```mermaid
flowchart TD
    A["base model active memory"] --> B["capacity selection"]
    B --> C["predictive cache materialization"]
    C --> D["peak memory check"]
    D --> E["warning if peak > 95% of recommended working set"]
    E --> F["enable wired limit after startup"]
```

## Runtime Memory Buckets

```mermaid
flowchart LR
    A["Non-expert model weights"] --> T["Total active memory"]
    B["Predictive expert cache tensors"] --> T
    C["Lookup and hit-mask metadata"] --> T
    D["KV cache (grows with context)"] --> T
```

Example at capacity around `208` on a `32 GB` machine:

- non-expert base weights
- predictive expert cache (`capacity x moe_layers`)
- lookup/hit-mask metadata
- KV cache growth during generation

## Wired Memory

```mermaid
flowchart LR
    A["startup complete"] --> B["active = mx.get_active_memory()"]
    B --> C["limit = 0.75 * device memory"]
    C --> D["mx.set_wired_limit(min(active, limit))"]
```

Wiring is intentionally applied after expert loading/refinement.

## Loading Strategy Notes

- Fresh `mx.load()` calls in hot paths avoid retaining lazy refs that can keep large source tensors materialized.
- `SafetensorsMap` enables mmap-based expert slicing to avoid loading full stacked tensors when only selected experts are needed.
