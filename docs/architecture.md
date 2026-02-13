# Architecture

## The Problem

Mixture-of-Experts (MoE) models have hundreds of experts per layer but route each token to only a few (e.g., 10 out of 512 for Qwen3-Coder). Loading all experts into GPU memory OOMs on constrained hardware -- a 46 GB model cannot fit in 32 GB of unified memory. But at any given token, only ~2% of expert weights are actually used.

mlx-moe exploits this sparsity by keeping only the most-needed experts in Metal memory and loading the rest on demand from SSD.

## Pipeline Overview

```
Load Model (lazy)  →  Router Discovery  →  Expert Selection  →  Predictive Upgrade  →  Generation
     ~1.4 GB              ~1s              top-N per layer         ~5-15s            zero-eval dispatch
```

## Stage 1: Lazy Loading

`enable_lazy_experts(model, model_path, capacity, predictive=True)` walks the model tree, finds every `QuantizedSwitchLinear` module (the expert projection layers inside SwitchGLU blocks), and replaces them with lazy-loading versions. The full replacement chain across stages is: `QuantizedSwitchLinear` (stock mlx-lm) → `LazyQuantizedSwitchLinear` (lazy disk loading) → `CachedQuantizedSwitchLinear` (LRU cache) → `PredictiveCachedSwitchLinear` (pre-stacked, zero-eval dispatch). Two module path conventions are detected:

- `layer.mlp.switch_mlp` -- Qwen, DeepSeek, GLM, Hunyuan, Jamba, OLMoE
- `layer.block_sparse_moe.switch_mlp` -- Mixtral, PhiMoE, MiniMax, GraniteMoE

After replacement, `mx.eval(model.parameters())` materializes only non-expert weights (~1.4 GB for a 46 GB model). Expert weights remain as lazy references in safetensors files on disk.

Each MoE layer's SwitchGLU contains 3 projection modules (gate_proj, up_proj, down_proj) — each is a separate `QuantizedSwitchLinear` holding all 512 experts. The SwitchGLU also has a **shared expert** (a standard MLP, always active on every token, never offloaded). The shared expert is critical for quality at low capacity — it provides a baseline MLP contribution even when routed experts are missing.

Each `CachedQuantizedSwitchLinear` shares a per-layer `ExpertCache` (LCP eviction: priority = frequency * 0.25^(recency/128)). During warmup, cache misses batch-load experts from the safetensors shard, eval once, then insert into the cache.

## Stage 2: Router Discovery

Before loading experts, we need to know *which* experts matter. `router_only_discovery()` monkey-patches MoE blocks to skip the expensive `switch_mlp` computation and run only the gate (router) + shared expert. Hidden states drift without MoE output, but routers still produce plausible expert selections.

Key optimization: instead of `mx.eval()` per layer per token (480 sync points for a 48-layer model generating 10 tokens), all router indices are collected as lazy tensors and bulk-evaluated in a single `mx.eval(*all_tensors)` call. This batched approach reduced discovery from 76s to ~1s.

The discovered expert IDs and their frequency counts are injected into the Phase 2 `ExpertCache` instances, seeding the LCP priority scores for the next stage.

## Stage 3: Predictive Expert Cache

`upgrade_to_predictive()` converts Phase 2 caches into GPU-resident `PredictiveExpertCache` objects. This is a three-pass process:

**Pass 1 -- Harvest.** For each MoE layer, rank discovered experts by LCP priority, take the top-C (capacity), and collect any weights already cached. Experts not in cache are marked for disk loading. If discovered experts don't fill capacity, filler experts pad the remaining slots.

**Pass 2 -- Batched shard loading.** Disk loads are grouped by safetensors shard file. Each shard is loaded once (via `mx.load` or `SafetensorsMap` mmap), and all needed experts across all layers are extracted. `mx.eval()` is called per-layer within each shard batch to control memory.

**Pass 3 -- Assemble and install.** Per-layer expert weights are `mx.stack`'d into stacked tensors `(C, ...)`. A lookup table (`mx.array` of shape `(num_experts,)`) maps global expert IDs (0-511) to cache slots (0 to C-1). Uncached experts map to slot 0 (fallback). `PredictiveCachedSwitchLinear` modules replace the Phase 2 modules.

When a universal expert profile is available, `upgrade_to_predictive_with_pinning()` places universally-active experts (activated >50% of the time across diverse prompts) in the first N slots and marks them non-evictable. This prevents degradation at 300+ tokens where dynamic updates would otherwise evict important experts.

**Prepacked weights** (`save_prepacked_weights` / `load_prepacked_weights`) serialize the stacked tensors to a single safetensors file. Warm start loads these directly, skipping the entire harvest/load/stack pipeline (14.8s -> 3.9s).

## Stage 4: Generation

The forward pass through `PredictiveCachedSwitchLinear` is fully lazy -- no `mx.eval()` calls:

1. Router indices arrive as an `mx.array`
2. `cache.remap(indices)` maps global expert IDs to local cache slots via the pre-built lookup table (pure array indexing, no eval)
3. `mx.gather_qmm()` dispatches the quantized matmul using pre-loaded stacked weight tensors

This keeps the entire forward pass in MLX's lazy evaluation graph, evaluated only at the output token. Adding `mx.eval()` inside the forward pass breaks the async_eval pipeline and causes a 2.2x slowdown.

**Skip-fallback** (`enable_skip_fallback`) monkey-patches MoE blocks to zero out router scores for uncached experts instead of letting them fall through to slot 0 (wrong expert). The remaining cached experts' scores are renormalized. This eliminates wrong-expert contamination -- the residual connection passes through unchanged input for the missing expert's contribution.

**Dynamic cache updates** (`dynamic_cache_update`) run between tokens: router indices captured during the forward pass are drained from a buffer, miss statistics are updated, and cold experts can be swapped for hot ones. Swaps load new expert weights from disk and scatter-update the stacked tensors in place.

## Key Design Decisions

**Auto capacity from Metal limits.** `select_capacity()` reads `mx.device_info()["max_recommended_working_set_size"]` (hardware-reported per-process GPU memory limit, not total system RAM) and targets 71% of it. The performance cliff on Apple Silicon starts at ~75%; 71% leaves headroom for KV cache growth. After upgrade, `mx.get_peak_memory()` is checked against 95% of recommended and warns if exceeded.

**Expert slot = weight + scales + biases.** Quantized experts have metadata tensors (scales, biases) alongside the weight tensor. At `group_size=64`, these add ~12.5% to the per-expert size. Omitting them from the slot size calculation causes capacity overshoot and OOM.

**Per-call `mx.load()` during warmup.** Fancy indexing on a lazy-loaded tensor materializes the *entire source tensor* permanently in Metal memory. Fresh `mx.load()` calls per batch give the GC a chance to free the source after slicing.

**Wired memory via `mx.set_wired_limit()`.** Pins the working set in a Metal `MTLResidencySet`, preventing macOS from evicting Metal buffers to SSD when total memory exceeds `recommendedMaxWorkingSetSize` (~75% of RAM). Wiring alone provides +21% tok/s; combined with `cache_limit(256MB)` during warmup (retains buffer cache for intermediate reuse), the total improvement is +47% (15.9 → 23.4 tok/s at capacity 208).

**mmap expert loading.** `SafetensorsMap` parses safetensors headers at init and mmaps each shard. `get_expert_slices()` computes byte offsets for each expert's row and reads only those bytes, avoiding materializing the full stacked tensor. 3.1x faster than `mx.load` for top-2 loading (crossover at 5-6 experts).

## Server: Prompt Caching (KV Cache Reuse)

The API server (`server.py`) caches the KV state (`_cached_tokens` + `_cached_kv`) from the previous request. On each new request, `_stream()` tokenizes the prompt and finds the longest common prefix (token-by-token comparison) with the cached sequence. It trims the KV cache to the prefix length and only prefills the new suffix tokens. For multi-turn agentic conversations where successive requests share a long prefix (system prompt + tools + full conversation history), this reduces TTFT from processing the full prompt to processing only the new user/tool message.

The cached sequence includes the model's generated tokens — so the prefix match extends through the model's previous response, not just the previous prompt. The cache is invalidated before generation starts and restored on successful completion; a client disconnect mid-stream loses the cache.

## Server: Hybrid Warmup

On the first request, the server generates 10 tokens from a short coding prompt. This serves two purposes:

1. **Expert coverage confirmation**: `dynamic_cache_update` runs between tokens and swaps in any experts the profile missed for the prompt's domain.
2. **One-time cost front-loading**: the first forward pass pays for Metal shader compilation and faulting mmap'd prepacked weights into physical memory. Without warmup, these costs shift to the user's first real request.

## Memory Layout

At capacity 208 on a 32 GB Mac (Qwen3-Coder-Next-4bit, 512 experts, top-10, 48 MoE layers):

```
Non-expert weights     ~1.4 GB
Expert cache (208×48)  ~17.0 GB    ← stacked tensors in Metal
Lookup tables          ~0.1 GB
KV cache (growing)     ~0.5 GB+
─────────────────────────────────
Total                  ~19.0 GB    (59% of 32 GB)
```
