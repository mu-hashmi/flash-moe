import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_moe.lazy_experts.modules import (
    PredictiveExpertCache,
    PredictiveCachedSwitchLinear,
    CachedQuantizedSwitchLinear,
    ExpertCache,
)
from mlx_moe.lazy_experts.persistence import (
    save_cache_state,
    load_cache_state,
    save_prepacked_weights,
    load_prepacked_weights,
    load_universal_profile,
    upgrade_from_profile,
)
from mlx_moe.lazy_experts.loading import SafetensorsMap


def _make_predictive_cache(capacity=4, num_experts=8, cached_ids=None):
    if cached_ids is None:
        cached_ids = list(range(capacity))
    cache = PredictiveExpertCache(capacity, num_experts)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        cache.weights[proj] = mx.zeros((capacity, 32, 16))
        cache.scales[proj] = mx.zeros((capacity, 32, 1))
        cache.biases[proj] = None
    cache.build_lookup(cached_ids)
    cache.frequency = {eid: (eid + 1) * 3 for eid in cached_ids}
    cache.last_active = {eid: eid + 1 for eid in cached_ids}
    cache.step = 10
    cache.pinned_set = {cached_ids[0]}
    return cache


def _make_model_with_predictive_cache(n_layers=2, capacity=4, num_experts=8):
    """Build a mock model whose layers have the structure _find_switch_mlp expects."""
    layers = []
    caches = []
    for _ in range(n_layers):
        cache = _make_predictive_cache(capacity, num_experts)
        caches.append(cache)
        gate = PredictiveCachedSwitchLinear(group_size=64, bits=4, mode="default", proj_name="gate_proj", cache=cache)
        up = PredictiveCachedSwitchLinear(group_size=64, bits=4, mode="default", proj_name="up_proj", cache=cache)
        down = PredictiveCachedSwitchLinear(group_size=64, bits=4, mode="default", proj_name="down_proj", cache=cache)
        switch = SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down)
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        layers.append(layer)
    model = SimpleNamespace(layers=layers)
    return model, caches


def _make_model_with_phase2_cache(n_layers=2, capacity=4, num_experts=8):
    """Build a mock model with CachedQuantizedSwitchLinear (Phase 2) modules."""
    layers = []
    for _ in range(n_layers):
        ec = ExpertCache(capacity)
        ec.all_seen = set(range(num_experts))
        for eid in range(num_experts):
            ec.entries[eid] = {}
            ec.frequency[eid] = eid + 1
            ec.last_active[eid] = eid
        ec.step = 5

        projs = {}
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mod = CachedQuantizedSwitchLinear(
                shard_path="dummy.safetensors",
                key_prefix="model.layers.0.mlp.switch_mlp." + proj_name,
                group_size=64, bits=4, mode="default",
                proj_name=proj_name, cache=ec,
            )
            mod.num_experts = num_experts
            projs[proj_name] = mod

        switch = SimpleNamespace(**projs)
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        layers.append(layer)

    return SimpleNamespace(layers=layers)


# --- save_cache_state / load_cache_state roundtrip ---

class TestCacheStateRoundtrip:
    def test_roundtrip_predictive(self, tmp_path):
        model, caches = _make_model_with_predictive_cache(n_layers=2, capacity=4)
        path = tmp_path / "cache.json"
        save_cache_state(model, path, metadata={"prompt": "hello"})

        state = load_cache_state(path)
        assert state["version"] == 1
        assert state["capacity"] == 4
        assert state["metadata"] == {"prompt": "hello"}
        assert "0" in state["layers"]
        assert "1" in state["layers"]

        for layer_key in ("0", "1"):
            layer_state = state["layers"][layer_key]
            assert layer_state["cached_ids"] == [0, 1, 2, 3]
            assert layer_state["step"] == 10
            assert "0" in layer_state["frequency"]
            assert int(layer_state["frequency"]["0"]) == 3

    def test_roundtrip_phase2(self, tmp_path):
        model = _make_model_with_phase2_cache(n_layers=1, capacity=4, num_experts=8)
        path = tmp_path / "cache_p2.json"
        save_cache_state(model, path)

        state = load_cache_state(path)
        assert state["version"] == 1
        layer_state = state["layers"]["0"]
        assert layer_state["step"] == 5
        assert set(layer_state["all_seen"]) == set(range(8))

    def test_load_rejects_bad_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": 99}))
        with pytest.raises(ValueError, match="Unsupported cache state version"):
            load_cache_state(path)


# --- save_prepacked_weights / load_prepacked_weights roundtrip ---

class TestPrepackedWeightsRoundtrip:
    def test_roundtrip(self, tmp_path):
        model, caches = _make_model_with_predictive_cache(n_layers=2, capacity=4, num_experts=8)

        for cache in caches:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                cache.weights[proj] = mx.random.normal((4, 32, 16))
                cache.scales[proj] = mx.random.normal((4, 32, 1))
                cache.biases[proj] = None
            mx.eval(cache.weights["gate_proj"], cache.weights["up_proj"], cache.weights["down_proj"],
                    cache.scales["gate_proj"], cache.scales["up_proj"], cache.scales["down_proj"])

        weights_path = tmp_path / "prepacked.safetensors"
        save_prepacked_weights(model, weights_path)

        assert weights_path.exists()
        meta_path = tmp_path / "prepacked.safetensors.meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["version"] == 1
        assert "0" in meta["layers"]
        assert "1" in meta["layers"]
        assert meta["layers"]["0"]["cached_ids"] == [0, 1, 2, 3]
        assert meta["layers"]["0"]["num_experts"] == 8

        # Load back into a Phase 2 model
        model_p2 = _make_model_with_phase2_cache(n_layers=2, capacity=4, num_experts=8)
        upgraded = load_prepacked_weights(model_p2, weights_path)
        assert upgraded == 6  # 3 projections × 2 layers

        # Verify weight values match
        for li in range(2):
            switch = model_p2.layers[li].mlp.switch_mlp
            loaded_cache = switch.gate_proj._cache
            original_cache = caches[li]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert mx.array_equal(loaded_cache.weights[proj], original_cache.weights[proj])
                assert mx.array_equal(loaded_cache.scales[proj], original_cache.scales[proj])

    def test_meta_has_pinned_set(self, tmp_path):
        model, caches = _make_model_with_predictive_cache(n_layers=1, capacity=4)
        caches[0].pinned_set = {0, 2}

        weights_path = tmp_path / "packed.safetensors"
        save_prepacked_weights(model, weights_path)

        with open(str(weights_path) + ".meta.json") as f:
            meta = json.load(f)
        assert sorted(meta["layers"]["0"]["pinned_set"]) == [0, 2]


# --- load_universal_profile ---

class TestLoadUniversalProfile:
    def test_load(self, tmp_path):
        profile = {
            "num_prompts": 22,
            "layers": {
                "0": {"activation_counts": {"10": 20, "42": 15, "100": 5}},
                "5": {"activation_counts": {"3": 22}},
            },
        }
        path = tmp_path / "profile.json"
        path.write_text(json.dumps(profile))

        loaded = load_universal_profile(path)
        assert loaded["num_prompts"] == 22
        assert "0" in loaded["layers"]
        assert loaded["layers"]["0"]["activation_counts"]["10"] == 20
        assert loaded["layers"]["5"]["activation_counts"]["3"] == 22


class TestUpgradeFromProfile:
    def _patch_fake_upgrade(self, monkeypatch, num_experts=8):
        def fake_upgrade(model, model_path, capacity):
            upgraded = 0
            for layer in model.layers:
                switch = layer.mlp.switch_mlp
                gate = getattr(switch, "gate_proj", None)
                if not isinstance(gate, CachedQuantizedSwitchLinear):
                    continue
                phase2_cache = gate._cache
                sorted_ids = sorted(
                    phase2_cache.entries.keys(),
                    key=lambda eid: (-phase2_cache.frequency.get(eid, 0), eid),
                )
                cached_ids = sorted_ids[:capacity]
                pred_cache = PredictiveExpertCache(capacity, num_experts)
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    pred_cache.weights[proj_name] = mx.zeros((capacity, 1, 1))
                    pred_cache.scales[proj_name] = mx.zeros((capacity, 1, 1))
                    pred_cache.biases[proj_name] = None
                pred_cache.build_lookup(cached_ids)
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    phase2 = getattr(switch, proj_name)
                    setattr(
                        switch,
                        proj_name,
                        PredictiveCachedSwitchLinear(
                            group_size=phase2.group_size,
                            bits=phase2.bits,
                            mode=phase2.mode,
                            proj_name=proj_name,
                            cache=pred_cache,
                        ),
                    )
                    upgraded += 1
            return upgraded

        monkeypatch.setattr(
            "mlx_moe.lazy_experts.persistence.upgrade_to_predictive",
            fake_upgrade,
        )

    def test_pin_top_k_zero_disables_pinning(self, monkeypatch):
        self._patch_fake_upgrade(monkeypatch)
        model = _make_model_with_phase2_cache(n_layers=1, capacity=4, num_experts=8)
        profile = {
            "num_prompts": 10,
            "layers": {
                "0": {
                    "activation_counts": {
                        "0": 10,
                        "1": 9,
                        "2": 8,
                        "3": 7,
                        "4": 6,
                    }
                }
            },
        }

        upgraded = upgrade_from_profile(
            model, model_path=".", capacity=4, profile=profile, pin_top_k=0
        )
        assert upgraded == 3
        cache = model.layers[0].mlp.switch_mlp.gate_proj._cache
        assert cache.pinned_set == set()

    def test_pin_top_k_selects_most_frequent(self, monkeypatch):
        self._patch_fake_upgrade(monkeypatch)
        model = _make_model_with_phase2_cache(n_layers=1, capacity=4, num_experts=8)
        profile = {
            "num_prompts": 10,
            "layers": {
                "0": {
                    "activation_counts": {
                        "4": 5,
                        "1": 9,
                        "0": 10,
                        "3": 6,
                        "2": 7,
                    }
                }
            },
        }

        upgraded = upgrade_from_profile(
            model, model_path=".", capacity=4, profile=profile, pin_top_k=2
        )
        assert upgraded == 3
        cache = model.layers[0].mlp.switch_mlp.gate_proj._cache
        assert cache.pinned_set == {0, 1}

    def test_pin_top_k_negative_rejected(self, monkeypatch):
        self._patch_fake_upgrade(monkeypatch)
        model = _make_model_with_phase2_cache(n_layers=1, capacity=4, num_experts=8)
        profile = {"num_prompts": 10, "layers": {"0": {"activation_counts": {"0": 10}}}}
        with pytest.raises(ValueError, match="pin_top_k must be >= 0"):
            upgrade_from_profile(
                model, model_path=".", capacity=4, profile=profile, pin_top_k=-1
            )


# --- SafetensorsMap ---

class TestSafetensorsMap:
    def _create_test_safetensors(self, path, tensors):
        mx.save_safetensors(str(path), tensors)

    def test_contains_and_get_tensor(self, tmp_path):
        t1 = mx.array([[1.0, 2.0], [3.0, 4.0]])
        t2 = mx.array([10, 20, 30])
        path = tmp_path / "test.safetensors"
        self._create_test_safetensors(path, {"weight": t1, "bias": t2})

        st = SafetensorsMap([str(path)])
        assert "weight" in st
        assert "bias" in st
        assert "nonexistent" not in st

        w = st.get_tensor("weight")
        assert mx.array_equal(w, t1)

        b = st.get_tensor("bias")
        assert mx.array_equal(b, t2)

        st.close()

    def test_get_expert_slices(self, tmp_path):
        stacked = mx.arange(24).reshape(4, 2, 3).astype(mx.float32)
        path = tmp_path / "experts.safetensors"
        self._create_test_safetensors(path, {"experts": stacked})

        st = SafetensorsMap([str(path)])
        sliced = st.get_expert_slices("experts", [1, 3])
        expected = mx.stack([stacked[1], stacked[3]])
        assert mx.array_equal(sliced, expected)
        st.close()

    def test_multiple_shards(self, tmp_path):
        p1 = tmp_path / "shard1.safetensors"
        p2 = tmp_path / "shard2.safetensors"
        self._create_test_safetensors(p1, {"a": mx.array([1.0])})
        self._create_test_safetensors(p2, {"b": mx.array([2.0])})

        st = SafetensorsMap([str(p1), str(p2)])
        assert "a" in st
        assert "b" in st
        assert mx.array_equal(st.get_tensor("a"), mx.array([1.0]))
        assert mx.array_equal(st.get_tensor("b"), mx.array([2.0]))
        st.close()

    def test_close_clears_state(self, tmp_path):
        path = tmp_path / "close_test.safetensors"
        self._create_test_safetensors(path, {"x": mx.array([0.0])})

        st = SafetensorsMap([str(path)])
        assert "x" in st
        st.close()
        assert "x" not in st

    def test_double_close(self, tmp_path):
        path = tmp_path / "double.safetensors"
        self._create_test_safetensors(path, {"y": mx.array([1.0])})
        st = SafetensorsMap([str(path)])
        st.close()
        st.close()
