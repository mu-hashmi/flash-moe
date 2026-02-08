# Copyright © 2023-2025 Apple Inc.

import os
import time
from pathlib import Path

import mlx.core as mx

from .loading import (
    _find_switch_mlp,
    _detect_num_experts,
    select_capacity,
    _with_cache_limit_zero,
)
from .core import enable_lazy_experts, upgrade_to_predictive
from .warmup import fast_delta_warmup
from .discovery import router_only_discovery
from .persistence import (
    save_cache_state,
    load_cache_state,
    save_prepacked_weights,
    load_prepacked_weights,
    upgrade_from_saved_state,
    load_universal_profile,
    upgrade_from_profile,
)


_WARMUP_CACHE = 256 * 1024 * 1024


def _flash_startup(model_name, prompt, cache_dir=None, profile_path=None,
                   prepacked=True):
    """Shared startup for flash_generate and flash_stream_generate.

    Returns (model, tokenizer, model_path) with all warmup/upgrade/wiring done.
    """
    import mlx_lm as _mlx_lm
    from mlx_lm.utils import hf_repo_to_path

    t_total_start = time.perf_counter()

    model_path = hf_repo_to_path(model_name)
    t0 = time.perf_counter()
    model, tokenizer = _mlx_lm.load(model_name, lazy=True)

    num_moe_layers = 0
    num_experts = 512
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            num_moe_layers += 1
            num_experts = _detect_num_experts(switch)

    device_gb = mx.device_info()["memory_size"] / 1e9
    capacity = select_capacity(1.4, device_gb, num_moe_layers=num_moe_layers)

    enable_lazy_experts(model, model_path,
                        cache_capacity_per_layer=capacity,
                        predictive=True)
    mx.eval(model.parameters())

    # Memory guard
    active_gb = mx.get_active_memory() / 1e9
    expert_slot_mb = 1.69
    projected_gb = active_gb + capacity * num_moe_layers * expert_slot_mb / 1024
    limit_gb = 0.85 * device_gb
    if projected_gb > limit_gb:
        max_cap = int((limit_gb - active_gb) * 1024 / (num_moe_layers * expert_slot_mb))
        capacity = (max_cap // 8) * 8
        capacity = max(capacity, 0)
        print(f"  [memory guard: reducing capacity to {capacity}]")
        enable_lazy_experts(model, model_path,
                            cache_capacity_per_layer=capacity,
                            predictive=True)
        mx.eval(model.parameters())

    t_load = time.perf_counter() - t0
    print(f"  Model load: {t_load:.1f}s ({mx.get_active_memory() / 1e9:.1f} GB)")

    cache_path = None
    prepacked_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        safe_name = model_name.replace("/", "--")
        cache_path = os.path.join(cache_dir, f"{safe_name}.json")
        prepacked_path = cache_path.replace(".json", ".weights.safetensors")

    used_saved_state = False

    if prepacked and prepacked_path and os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        with _with_cache_limit_zero(_WARMUP_CACHE):
            load_prepacked_weights(model, prepacked_path, model_path=model_path)
        print(f"  Prepacked load: {time.perf_counter() - t0:.1f}s")

        if cache_path and os.path.exists(cache_path):
            cache_state = load_cache_state(cache_path)
            saved_prompt = cache_state.get("metadata", {}).get("prompt")
            if saved_prompt and saved_prompt != prompt:
                t0 = time.perf_counter()
                with _with_cache_limit_zero(_WARMUP_CACHE):
                    fast_delta_warmup(model, tokenizer, model_path, prompt,
                                      discovery_tokens=10)
                print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    elif cache_path and os.path.exists(cache_path):
        t0 = time.perf_counter()
        cache_state = load_cache_state(cache_path)
        with _with_cache_limit_zero(_WARMUP_CACHE):
            upgrade_from_saved_state(model, model_path, cache_state, capacity)
        print(f"  Cache state upgrade: {time.perf_counter() - t0:.1f}s")

        saved_prompt = cache_state.get("metadata", {}).get("prompt")
        if saved_prompt and saved_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                fast_delta_warmup(model, tokenizer, model_path, prompt,
                                  discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    else:
        if profile_path is not None:
            t0 = time.perf_counter()
            profile = load_universal_profile(profile_path)
            with _with_cache_limit_zero(_WARMUP_CACHE):
                upgrade_from_profile(model, model_path, capacity, profile)
            print(f"  Profile-based upgrade: {time.perf_counter() - t0:.1f}s")
        else:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                router_only_discovery(model, tokenizer, prompt, max_tokens=10)
                upgrade_to_predictive(model, model_path, capacity)
            print(f"  Router-only discovery + upgrade: {time.perf_counter() - t0:.1f}s")

    if cache_path and not used_saved_state:
        save_cache_state(model, cache_path,
                         metadata={"prompt": prompt, "capacity": capacity})

    if prepacked and prepacked_path and not os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        save_prepacked_weights(model, prepacked_path)
        print(f"  Save prepacked: {time.perf_counter() - t0:.1f}s")

    t_total = time.perf_counter() - t_total_start
    print(f"  Total startup: {t_total:.1f}s")

    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        wired = min(active, limit)
        mx.set_wired_limit(wired)
        print(f"  Wired {wired / 1e9:.1f} GB in residency set")

    return model, tokenizer, model_path


def flash_generate(model_name: str, prompt: str, max_tokens: int = 200,
                   cache_dir: str | None = None,
                   profile_path: str | None = None,
                   prepacked: bool = True) -> str:
    """One-call generation with all optimizations.

    Auto-detects RAM, selects capacity, loads cached state if available,
    applies pinning if profile exists, uses cache_limit(0) during warmup.

    Args:
        model_name: HuggingFace model name (e.g. "mlx-community/Qwen3-Coder-Next-4bit").
        prompt: Text prompt for generation.
        max_tokens: Maximum tokens to generate.
        cache_dir: Directory for cache state persistence. None disables caching.
        profile_path: Path to universal expert profile JSON for pinning.
        prepacked: Save/load prepacked weight files for fastest warm start.

    Returns:
        Generated text string.
    """
    import mlx_lm as _mlx_lm

    model, tokenizer, _ = _flash_startup(
        model_name, prompt, cache_dir=cache_dir,
        profile_path=profile_path, prepacked=prepacked)

    return _mlx_lm.generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, verbose=False)


def flash_stream_generate(model_name: str, prompt: str, max_tokens: int = 200,
                          cache_dir: str | None = None,
                          profile_path: str | None = None,
                          prepacked: bool = True):
    """Streaming variant of flash_generate.

    Startup is blocking. After startup, yields GenerationResponse objects
    from mlx_lm.stream_generate.

    Args:
        Same as flash_generate.

    Yields:
        mlx_lm.generate.GenerationResponse with token-by-token output.
    """
    import mlx_lm as _mlx_lm

    model, tokenizer, _ = _flash_startup(
        model_name, prompt, cache_dir=cache_dir,
        profile_path=profile_path, prepacked=prepacked)

    yield from _mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                        max_tokens=max_tokens)


class FlashSession:
    """Reusable session for multi-turn generation.

    Loads the model once on first use. Subsequent calls reuse the loaded model
    and run delta warmup if the prompt domain changed.

    Usage:
        session = FlashSession("mlx-community/Qwen3-Coder-Next-4bit",
                               cache_dir="~/.cache/flash-moe")
        for resp in session.stream("Write a Flask server"):
            print(resp.text, end="")
        text = session.generate("Now add tests")
        session.close()
    """

    def __init__(self, model_name: str, cache_dir: str | None = None,
                 profile_path: str | None = None, prepacked: bool = True):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._profile_path = profile_path
        self._prepacked = prepacked
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._last_prompt = None

    def _ensure_loaded(self, prompt: str):
        if self._model is None:
            self._model, self._tokenizer, self._model_path = _flash_startup(
                self._model_name, prompt, cache_dir=self._cache_dir,
                profile_path=self._profile_path, prepacked=self._prepacked)
            self._last_prompt = prompt
        elif self._last_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                fast_delta_warmup(self._model, self._tokenizer,
                                  self._model_path, prompt, discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
            self._last_prompt = prompt

    def stream(self, prompt: str, max_tokens: int = 200):
        """Stream tokens for a single turn."""
        import mlx_lm as _mlx_lm

        self._ensure_loaded(prompt)
        yield from _mlx_lm.stream_generate(self._model, self._tokenizer,
                                            prompt=prompt, max_tokens=max_tokens)

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Non-streaming generation for a single turn."""
        import mlx_lm as _mlx_lm

        self._ensure_loaded(prompt)
        return _mlx_lm.generate(self._model, self._tokenizer,
                                 prompt=prompt, max_tokens=max_tokens, verbose=False)

    @property
    def memory_gb(self) -> float:
        return mx.get_active_memory() / 1e9

    def close(self):
        """Release model and clear GPU memory."""
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._last_prompt = None
        mx.clear_cache()
