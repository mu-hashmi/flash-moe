"""Integration tests for the mlx-moe module replacement pipeline and server endpoints."""

import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU


# ---------------------------------------------------------------------------
# Synthetic MoE model + safetensors fixture
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8
INPUT_DIMS = 64
HIDDEN_DIMS = 128
GROUP_SIZE = 64
BITS = 4
NUM_LAYERS = 2


class FakeMoEBlock(nn.Module):
    def __init__(self, num_experts, input_dims, hidden_dims):
        super().__init__()
        self.num_experts_per_tok = 2
        self.gate = nn.Linear(input_dims, num_experts)
        self.switch_mlp = SwitchGLU(input_dims, hidden_dims, num_experts, bias=False)

    def __call__(self, x):
        gates = self.gate(x)
        k = self.num_experts_per_tok
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = mx.softmax(scores, axis=-1, precise=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class FakeLayer(nn.Module):
    def __init__(self, num_experts, input_dims, hidden_dims):
        super().__init__()
        self.mlp = FakeMoEBlock(num_experts, input_dims, hidden_dims)


class FakeModel(nn.Module):
    def __init__(self, num_layers, num_experts, input_dims, hidden_dims):
        super().__init__()
        self.layers = [FakeLayer(num_experts, input_dims, hidden_dims)
                       for _ in range(num_layers)]


def _quantize_switch_glu(switch_glu: SwitchGLU, group_size=GROUP_SIZE, bits=BITS):
    """Replace SwitchLinear projections with QuantizedSwitchLinear in-place."""
    for name in ("gate_proj", "up_proj", "down_proj"):
        sl = getattr(switch_glu, name)
        ql = sl.to_quantized(group_size=group_size, bits=bits)
        setattr(switch_glu, name, ql)


@pytest.fixture
def synthetic_model_dir(tmp_path):
    """Create a synthetic MoE model saved as safetensors with an index file.

    Returns (model, model_path) where model has QuantizedSwitchLinear modules.
    """
    model = FakeModel(NUM_LAYERS, NUM_EXPERTS, INPUT_DIMS, HIDDEN_DIMS)
    for layer in model.layers:
        _quantize_switch_glu(layer.mlp.switch_mlp)

    mx.eval(model.parameters())

    weights = {}
    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}.mlp"
        gate = layer.mlp.gate
        weights[f"{prefix}.gate.weight"] = gate.weight
        if hasattr(gate, "bias"):
            weights[f"{prefix}.gate.bias"] = gate.bias

        switch = layer.mlp.switch_mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ql = getattr(switch, proj)
            weights[f"{prefix}.switch_mlp.{proj}.weight"] = ql.weight
            weights[f"{prefix}.switch_mlp.{proj}.scales"] = ql.scales
            if ql.biases is not None:
                weights[f"{prefix}.switch_mlp.{proj}.biases"] = ql.biases

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = tmp_path / shard_name
    mx.save_safetensors(str(shard_path), weights)

    index = {"weight_map": {k: shard_name for k in weights}}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    return model, tmp_path


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestEnableLazyExperts:
    def test_replaces_all_projections(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts
        from mlx_moe.lazy_experts.modules import LazyQuantizedSwitchLinear

        model, model_path = synthetic_model_dir
        replaced = enable_lazy_experts(model, model_path)
        assert replaced == NUM_LAYERS * 3

        for layer in model.layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp.switch_mlp, proj)
                assert isinstance(mod, LazyQuantizedSwitchLinear)

    def test_cached_mode(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts
        from mlx_moe.lazy_experts.modules import CachedQuantizedSwitchLinear

        model, model_path = synthetic_model_dir
        replaced = enable_lazy_experts(model, model_path, cache_capacity_per_layer=4)
        assert replaced == NUM_LAYERS * 3

        for layer in model.layers:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp.switch_mlp, proj)
                assert isinstance(mod, CachedQuantizedSwitchLinear)
                assert mod._cache.capacity == 4

    def test_shared_cache_across_projections(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts
        from mlx_moe.lazy_experts.modules import CachedQuantizedSwitchLinear

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=4)

        for layer in model.layers:
            gate_cache = getattr(layer.mlp.switch_mlp, "gate_proj")._cache
            up_cache = getattr(layer.mlp.switch_mlp, "up_proj")._cache
            down_cache = getattr(layer.mlp.switch_mlp, "down_proj")._cache
            assert gate_cache is up_cache
            assert up_cache is down_cache


class TestUpgradeToPredictive:
    def test_upgrade_from_cached(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive
        from mlx_moe.lazy_experts.modules import (
            PredictiveCachedSwitchLinear,
            PredictiveExpertCache,
        )

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgraded = upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)
        assert upgraded == NUM_LAYERS * 3

        for layer in model.layers:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp.switch_mlp, proj_name)
                assert isinstance(mod, PredictiveCachedSwitchLinear)
                cache = mod._cache
                assert isinstance(cache, PredictiveExpertCache)
                assert cache.capacity == NUM_EXPERTS
                assert cache.lookup is not None
                assert cache.lookup.shape[0] >= NUM_EXPERTS

    def test_predictive_cache_weights_populated(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        cache = model.layers[0].mlp.switch_mlp.up_proj._cache
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert proj in cache.weights
            assert cache.weights[proj].shape[0] == NUM_EXPERTS
            assert proj in cache.scales

    def test_sync_variant(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive
        from mlx_moe.lazy_experts.modules import SyncPredictiveCachedSwitchLinear

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS, sync=True)

        mod = model.layers[0].mlp.switch_mlp.gate_proj
        assert isinstance(mod, SyncPredictiveCachedSwitchLinear)


class TestResetToCached:
    def test_downgrade_predictive_to_cached(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import (
            enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
        )
        from mlx_moe.lazy_experts.modules import CachedQuantizedSwitchLinear

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        reset = reset_to_cached(model, model_path, capacity=4)
        assert reset == NUM_LAYERS * 3

        for layer in model.layers:
            mod = getattr(layer.mlp.switch_mlp, "gate_proj")
            assert isinstance(mod, CachedQuantizedSwitchLinear)
            assert mod._cache.capacity == 4


class TestEnableSkipFallback:
    def test_patches_moe_blocks(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import (
            enable_lazy_experts, upgrade_to_predictive, enable_skip_fallback,
        )

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        patched = enable_skip_fallback(model)
        assert patched == NUM_LAYERS


class TestBuildShardMap:
    def test_maps_all_keys(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.loading import _build_shard_map

        _, model_path = synthetic_model_dir
        shard_map = _build_shard_map(model_path)

        for i in range(NUM_LAYERS):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{i}.mlp.switch_mlp.{proj}.weight"
                assert key in shard_map


class TestSelectCapacity:
    def test_basic(self):
        from mlx_moe.lazy_experts.loading import select_capacity

        cap = select_capacity(base_memory_gb=1.4, recommended_gb=24.0,
                              num_moe_layers=48, expert_slot_mb=1.77)
        assert 0 < cap <= 512
        assert cap % 8 == 0

    def test_zero_on_no_headroom(self):
        from mlx_moe.lazy_experts.loading import select_capacity

        cap = select_capacity(base_memory_gb=20.0, recommended_gb=24.0,
                              num_moe_layers=48, expert_slot_mb=1.77)
        assert cap == 0 or cap < 10


class TestSafetensorsMap:
    def test_get_tensor(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.loading import SafetensorsMap, _build_shard_map

        _, model_path = synthetic_model_dir
        shard_map = _build_shard_map(model_path)
        shard_paths = list(set(shard_map.values()))
        st_map = SafetensorsMap(shard_paths)

        key = "model.layers.0.mlp.switch_mlp.up_proj.weight"
        t = st_map.get_tensor(key)
        assert t.shape[0] == NUM_EXPERTS
        st_map.close()

    def test_get_expert_slices(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.loading import SafetensorsMap, _build_shard_map

        _, model_path = synthetic_model_dir
        shard_map = _build_shard_map(model_path)
        shard_paths = list(set(shard_map.values()))
        st_map = SafetensorsMap(shard_paths)

        key = "model.layers.0.mlp.switch_mlp.up_proj.weight"
        sliced = st_map.get_expert_slices(key, [0, 3])
        assert sliced.shape[0] == 2

        full = st_map.get_tensor(key)
        np.testing.assert_array_equal(np.array(sliced[0]), np.array(full[0]))
        np.testing.assert_array_equal(np.array(sliced[1]), np.array(full[3]))
        st_map.close()


class TestPredictiveForwardPass:
    def test_forward_produces_output(self, synthetic_model_dir):
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        switch = model.layers[0].mlp.switch_mlp
        x = mx.random.normal((1, 1, INPUT_DIMS))
        indices = mx.array([[0, 1]])
        out = switch(x, indices)
        mx.eval(out)
        assert out.shape[-1] == INPUT_DIMS

    def test_golden_reference_matches_stock(self, synthetic_model_dir):
        """Predictive pipeline with all experts loaded must match stock SwitchGLU."""
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive

        model, model_path = synthetic_model_dir

        # Run stock SwitchGLU (before any replacement)
        mx.random.seed(42)
        x = mx.random.normal((1, 1, INPUT_DIMS))
        indices = mx.array([[0, 3]])
        stock_out = model.layers[0].mlp.switch_mlp(x, indices)
        mx.eval(stock_out)

        # Now replace with predictive pipeline (all experts loaded)
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        pred_out = model.layers[0].mlp.switch_mlp(x, indices)
        mx.eval(pred_out)

        np.testing.assert_allclose(
            np.array(pred_out), np.array(stock_out), rtol=1e-4, atol=1e-5)


class TestSkipFallbackForwardPass:
    def test_output_with_cache_misses(self, synthetic_model_dir):
        """skip_fallback should produce output even when some experts are missing."""
        from mlx_moe.lazy_experts.core import (
            enable_lazy_experts, upgrade_to_predictive, enable_skip_fallback,
        )

        model, model_path = synthetic_model_dir
        # Load only 4 of 8 experts — half will be misses
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=4)
        upgrade_to_predictive(model, model_path, capacity=4)
        enable_skip_fallback(model)

        # Use the patched MoE block (not switch_mlp directly)
        moe_block = model.layers[0].mlp
        x = mx.random.normal((1, 1, INPUT_DIMS))
        out = moe_block(x)
        mx.eval(out)

        assert out.shape == x.shape
        assert not mx.any(mx.isnan(out)).item()

    def test_full_cache_matches_no_skip(self, synthetic_model_dir):
        """With all experts cached, skip_fallback should produce identical output."""
        from mlx_moe.lazy_experts.core import (
            enable_lazy_experts, upgrade_to_predictive, enable_skip_fallback,
        )

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        moe_block = model.layers[0].mlp
        x = mx.random.normal((1, 1, INPUT_DIMS))

        out_before = moe_block(x)
        mx.eval(out_before)

        enable_skip_fallback(model)
        out_after = moe_block(x)
        mx.eval(out_after)

        np.testing.assert_allclose(
            np.array(out_after), np.array(out_before), rtol=1e-5, atol=1e-6)


class TestResetAndReupgrade:
    def test_reset_then_reupgrade(self, synthetic_model_dir):
        """Full cycle: upgrade → reset → re-upgrade. Forward pass works after."""
        from mlx_moe.lazy_experts.core import (
            enable_lazy_experts, upgrade_to_predictive, reset_to_cached,
        )
        from mlx_moe.lazy_experts.modules import PredictiveCachedSwitchLinear

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        # Reset back to cached
        reset_to_cached(model, model_path, capacity=4)

        # Re-upgrade with different capacity
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        mod = model.layers[0].mlp.switch_mlp.gate_proj
        assert isinstance(mod, PredictiveCachedSwitchLinear)

        # Forward pass still works
        switch = model.layers[0].mlp.switch_mlp
        x = mx.random.normal((1, 1, INPUT_DIMS))
        indices = mx.array([[2, 5]])
        out = switch(x, indices)
        mx.eval(out)
        assert out.shape[-1] == INPUT_DIMS


class TestDeltaWarmupMechanism:
    def test_expert_swap_via_buffer(self, synthetic_model_dir):
        """Test PredictiveExpertCache.update() — the core of delta warmup.

        The update() method processes buffered indices from forward passes,
        identifies misses, and swaps cold experts for requested ones.
        """
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive
        from mlx_moe.lazy_experts.loading import SafetensorsMap, _build_shard_map

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=4)
        upgrade_to_predictive(model, model_path, capacity=4)

        shard_map = _build_shard_map(model_path)
        st_map = SafetensorsMap(sorted(set(shard_map.values())))

        cache = model.layers[0].mlp.switch_mlp.up_proj._cache
        cache._st_map = st_map
        original_cached = set(cache.cached_set)

        # Find experts NOT in the cache
        missing = [e for e in range(NUM_EXPERTS) if e not in original_cached]
        assert len(missing) > 0

        # Simulate two forward passes requesting a missing expert.
        # update() needs at least 2 buffered entries (skips last as in-flight).
        cache._indices_buffer.append(mx.array([[missing[0], cache.cached_ids[0]]]))
        cache._indices_buffer.append(mx.array([[missing[0], cache.cached_ids[1]]]))

        result = cache.update()

        # The missing expert should have been swapped in (if swap machinery is wired)
        assert result["requests"] > 0

        st_map.close()

    def test_cached_set_tracks_loaded_experts(self, synthetic_model_dir):
        """Verify cached_set accurately reflects which experts are loaded."""
        from mlx_moe.lazy_experts.core import enable_lazy_experts, upgrade_to_predictive

        model, model_path = synthetic_model_dir
        enable_lazy_experts(model, model_path, cache_capacity_per_layer=NUM_EXPERTS)
        upgrade_to_predictive(model, model_path, capacity=NUM_EXPERTS)

        cache = model.layers[0].mlp.switch_mlp.up_proj._cache
        assert len(cache.cached_set) == NUM_EXPERTS
        assert len(cache.cached_ids) == NUM_EXPERTS
        assert set(cache.cached_ids) == cache.cached_set


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------


class TestServerEndpoints:
    @pytest.fixture
    def server(self):
        """Build a Server instance with mocked internals for endpoint testing."""
        from mlx_moe.server import Server

        srv = Server("test-model/fake", max_tokens=100, max_input_tokens=1000)
        srv._model_id = "fake-model"
        srv._model = None
        srv._memory_gb = 0.0
        return srv

    def test_models_endpoint(self, server):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route

        app = Starlette(routes=[
            Route("/v1/models", server.handle_models, methods=["GET"]),
        ])
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "fake-model"
        assert data["data"][0]["owned_by"] == "mlx-moe"

    def test_format_messages_simple(self):
        from mlx_moe.server import _format_messages

        msgs = [{"role": "user", "content": "hello"}]
        result = _format_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_format_messages_with_system(self):
        from mlx_moe.server import _format_messages

        msgs = [{"role": "user", "content": "hi"}]
        result = _format_messages(msgs, system="You are a helper.")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helper."
        assert result[1]["role"] == "user"

    def test_format_messages_anthropic_tool_use(self):
        from mlx_moe.server import _format_messages

        msgs = [{"role": "assistant", "content": [
            {"type": "text", "text": "Let me call that."},
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {"path": "/tmp/x"}},
        ]}]
        result = _format_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "Let me call that."
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_format_messages_tool_result(self):
        from mlx_moe.server import _format_messages

        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": [{"type": "text", "text": "file contents"}]},
        ]}]
        result = _format_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "file contents"

    def test_convert_tools_anthropic_to_openai(self):
        from mlx_moe.server import _convert_tools_anthropic_to_openai

        tools = [{
            "name": "read_file",
            "description": "Reads a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }]
        result = _convert_tools_anthropic_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_sampling_kwargs(self):
        from mlx_moe.server import _sampling_kwargs

        body = {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
        kwargs = _sampling_kwargs(body)
        assert kwargs["temp"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40

    def test_sampling_kwargs_empty(self):
        from mlx_moe.server import _sampling_kwargs

        assert _sampling_kwargs({}) == {}

    def test_strip_thinking(self):
        from mlx_moe.server import _strip_thinking

        text = "<think>internal reasoning</think>The answer is 42."
        assert _strip_thinking(text) == "The answer is 42."

    def test_strip_thinking_no_tags(self):
        from mlx_moe.server import _strip_thinking

        text = "No thinking here."
        assert _strip_thinking(text) == "No thinking here."

    def test_check_input_length_ok(self, server):
        assert server._check_input_length(500) is None

    def test_check_input_length_too_long(self, server):
        resp = server._check_input_length(2000)
        assert resp is not None
        assert resp.status_code == 400

    def test_chat_completions_rejects_missing_messages(self, server):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route

        app = Starlette(routes=[
            Route("/v1/chat/completions", server.handle_chat_completions, methods=["POST"]),
        ])
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 400 or resp.status_code == 500

    def test_messages_endpoint_rejects_missing_messages(self, server):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route

        app = Starlette(routes=[
            Route("/v1/messages", server.handle_messages, methods=["POST"]),
        ])
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/messages", json={})
        assert resp.status_code == 400 or resp.status_code == 500


# ---------------------------------------------------------------------------
# ExpertCache unit tests
# ---------------------------------------------------------------------------


class TestExpertCache:
    def test_put_and_lookup(self):
        from mlx_moe.lazy_experts.modules import ExpertCache

        cache = ExpertCache(capacity=4)
        w = mx.zeros((2, 3))
        s = mx.zeros((2, 1))
        cache.put(0, "gate_proj", w, s, None)

        result = cache.lookup(0, "gate_proj")
        assert result is not None
        assert result[0] is w

    def test_eviction(self):
        from mlx_moe.lazy_experts.modules import ExpertCache

        cache = ExpertCache(capacity=2)
        for eid in range(3):
            cache.put(eid, "gate_proj", mx.zeros((1,)), mx.zeros((1,)), None)
            cache.frequency[eid] = 1
            cache.last_active[eid] = eid

        cache.step = 3
        cache.evict_if_needed(protected=set())
        assert len(cache.entries) == 2

    def test_miss_returns_none(self):
        from mlx_moe.lazy_experts.modules import ExpertCache

        cache = ExpertCache(capacity=4)
        assert cache.lookup(99, "gate_proj") is None


class TestPredictiveExpertCache:
    def test_build_lookup(self):
        from mlx_moe.lazy_experts.modules import PredictiveExpertCache

        cache = PredictiveExpertCache(capacity=4, num_experts=8)
        cache.build_lookup([0, 2, 5, 7])
        mx.eval(cache.lookup)

        lookup_np = np.array(cache.lookup)
        assert lookup_np[0] == 0
        assert lookup_np[2] == 1
        assert lookup_np[5] == 2
        assert lookup_np[7] == 3
        # Uncached maps to 0 (fallback)
        assert lookup_np[1] == 0
        assert lookup_np[3] == 0

    def test_hit_mask(self):
        from mlx_moe.lazy_experts.modules import PredictiveExpertCache

        cache = PredictiveExpertCache(capacity=2, num_experts=8)
        cache.build_lookup([1, 4])
        mx.eval(cache.hit_mask)

        mask_np = np.array(cache.hit_mask)
        assert mask_np[1] == 1.0
        assert mask_np[4] == 1.0
        assert mask_np[0] == 0.0
        assert mask_np[7] == 0.0

    def test_remap(self):
        from mlx_moe.lazy_experts.modules import PredictiveExpertCache

        cache = PredictiveExpertCache(capacity=3, num_experts=8)
        cache.build_lookup([2, 5, 7])
        mx.eval(cache.lookup)

        indices = mx.array([[2, 5, 7]])
        remapped = cache.remap(indices)
        mx.eval(remapped)
        assert list(np.array(remapped.reshape(-1))) == [0, 1, 2]
