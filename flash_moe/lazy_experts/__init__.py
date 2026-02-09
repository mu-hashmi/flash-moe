# Copyright © 2023-2025 Apple Inc.

from .loading import (
    _find_switch_mlp,
    _find_moe_block,
    _detect_num_experts,
    _build_shard_map,
    _load_proj_experts,
    _load_experts,
    _mmap_load_proj_experts,
    SafetensorsMap,
    _PROJ_TO_EXPERT_NAMES,
    select_capacity,
    _with_cache_limit_zero,
    compute_adaptive_allocations,
)

from .modules import (
    ExpertCache,
    LazyQuantizedSwitchLinear,
    CachedQuantizedSwitchLinear,
    PredictiveExpertCache,
    PredictiveCachedSwitchLinear,
    SyncPredictiveCachedSwitchLinear,
)

from .core import (
    enable_lazy_experts,
    enable_skip_fallback,
    reset_to_cached,
    upgrade_to_predictive,
    upgrade_to_predictive_with_pinning,
    dynamic_cache_update,
    dynamic_cache_update_ml,
    get_fallback_stats,
    measure_fallback,
    get_cache_stats,
    adaptive_capacity_upgrade,
)

from .warmup import (
    delta_warmup,
    fast_delta_warmup,
    IncrementalDeltaWarmup,
    incremental_delta_warmup,
)

from .discovery import (
    router_only_forward,
    router_only_discovery,
    speculative_router_probe,
    speculative_router_cross_layer,
)

from .persistence import (
    save_cache_state,
    load_cache_state,
    save_prepacked_weights,
    load_prepacked_weights,
    upgrade_from_saved_state,
    load_universal_profile,
    upgrade_from_profile,
)

from .generate import flash_generate, flash_stream_generate, FlashSession
