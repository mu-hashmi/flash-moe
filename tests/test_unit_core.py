import numpy as np
import pytest
from types import SimpleNamespace

import mlx.core as mx

from mlx_moe.lazy_experts.modules import ExpertCache, PredictiveExpertCache
from mlx_moe.lazy_experts.loading import (
    _find_switch_mlp,
    _find_moe_block,
    _detect_num_experts,
    select_capacity,
    compute_adaptive_allocations,
)


# ---------------------------------------------------------------------------
# ExpertCache
# ---------------------------------------------------------------------------

class TestExpertCache:

    def test_put_and_lookup(self):
        cache = ExpertCache(capacity=4)
        cache.put(0, "gate_proj", "w0", "s0", "b0")
        assert cache.lookup(0, "gate_proj") == ("w0", "s0", "b0")
        assert cache.lookup(0, "up_proj") is None
        assert cache.lookup(99, "gate_proj") is None

    def test_capacity_limit(self):
        cache = ExpertCache(capacity=2)
        for i in range(4):
            cache.put(i, "gate_proj", f"w{i}", f"s{i}", None)
            cache.frequency[i] = 1
            cache.last_active[i] = 1
        cache.evict_if_needed(protected=set())
        assert len(cache.entries) == 2

    def test_lcp_eviction_order(self):
        cache = ExpertCache(capacity=2)
        cache.step = 10
        # Expert 0: high frequency, recently active -> high priority
        cache.put(0, "gate_proj", "w0", "s0", None)
        cache.frequency[0] = 100
        cache.last_active[0] = 10
        # Expert 1: low frequency, stale -> low priority (evicted first)
        cache.put(1, "gate_proj", "w1", "s1", None)
        cache.frequency[1] = 1
        cache.last_active[1] = 1
        # Expert 2: medium
        cache.put(2, "gate_proj", "w2", "s2", None)
        cache.frequency[2] = 5
        cache.last_active[2] = 8

        cache.evict_if_needed(protected=set())
        assert 0 in cache.entries
        assert 1 not in cache.entries

    def test_evict_respects_protected(self):
        cache = ExpertCache(capacity=1)
        for i in range(3):
            cache.put(i, "gate_proj", f"w{i}", f"s{i}", None)
            cache.frequency[i] = 1
            cache.last_active[i] = 1

        cache.evict_if_needed(protected={0, 1, 2})
        # Can't evict anything, all protected
        assert len(cache.entries) == 3

    def test_evict_skips_protected_evicts_others(self):
        cache = ExpertCache(capacity=1)
        cache.step = 5
        cache.put(0, "gate_proj", "w0", "s0", None)
        cache.frequency[0] = 100
        cache.last_active[0] = 5
        cache.put(1, "gate_proj", "w1", "s1", None)
        cache.frequency[1] = 1
        cache.last_active[1] = 1
        cache.put(2, "gate_proj", "w2", "s2", None)
        cache.frequency[2] = 1
        cache.last_active[2] = 1

        cache.evict_if_needed(protected={1})
        assert len(cache.entries) == 1
        # Expert 1 is protected, so it survives even though low priority
        assert 1 in cache.entries

    def test_projection_called_step_counting(self):
        cache = ExpertCache(capacity=8)
        ids = np.array([0, 1])
        # 3 projections = 1 step
        cache.projection_called(ids)
        cache.projection_called(ids)
        cache.projection_called(ids)
        assert cache.step == 1
        assert cache.frequency[0] == 1
        assert cache.frequency[1] == 1
        # Another 3 projections = step 2
        cache.projection_called(ids)
        cache.projection_called(ids)
        cache.projection_called(ids)
        assert cache.step == 2
        assert cache.frequency[0] == 2

    def test_hits_misses_tracking(self):
        cache = ExpertCache(capacity=4)
        cache.hits = 0
        cache.misses = 0
        cache.put(0, "gate_proj", "w0", "s0", None)
        assert cache.lookup(0, "gate_proj") is not None
        assert cache.lookup(0, "up_proj") is None

    def test_all_seen_tracking(self):
        cache = ExpertCache(capacity=4)
        cache.projection_called(np.array([0, 3, 7]))
        cache.projection_called(np.array([0, 3, 7]))
        cache.projection_called(np.array([0, 3, 7]))
        cache.projection_called(np.array([1, 2]))
        assert cache.all_seen == {0, 1, 2, 3, 7}


# ---------------------------------------------------------------------------
# PredictiveExpertCache
# ---------------------------------------------------------------------------

class TestPredictiveExpertCache:

    def test_slot_allocation(self):
        pc = PredictiveExpertCache(capacity=4, num_experts=16)
        assert pc.capacity == 4
        assert pc.num_experts == 16
        assert pc.cached_ids == []
        assert pc.cached_set == set()

    def test_build_lookup_and_remap(self):
        pc = PredictiveExpertCache(capacity=4, num_experts=16)
        pc.build_lookup([2, 5, 9, 11])

        assert pc.cached_ids == [2, 5, 9, 11]
        assert pc.cached_set == {2, 5, 9, 11}

        lookup_np = np.array(pc.lookup)
        # Expert 2 -> slot 0, expert 5 -> slot 1, etc.
        assert lookup_np[2] == 0
        assert lookup_np[5] == 1
        assert lookup_np[9] == 2
        assert lookup_np[11] == 3
        # Uncached experts map to slot 0 (fallback)
        assert lookup_np[0] == 0
        assert lookup_np[15] == 0

    def test_hit_mask(self):
        pc = PredictiveExpertCache(capacity=3, num_experts=8)
        pc.build_lookup([1, 4, 6])

        mask_np = np.array(pc.hit_mask)
        assert mask_np[1] == 1.0
        assert mask_np[4] == 1.0
        assert mask_np[6] == 1.0
        assert mask_np[0] == 0.0
        assert mask_np[2] == 0.0
        assert mask_np[7] == 0.0

    def test_remap_mx_array(self):
        pc = PredictiveExpertCache(capacity=3, num_experts=8)
        pc.build_lookup([1, 4, 6])

        indices = mx.array([1, 4, 6, 0])
        remapped = pc.remap(indices)
        mx.eval(remapped)
        r = np.array(remapped)
        assert r[0] == 0  # expert 1 -> slot 0
        assert r[1] == 1  # expert 4 -> slot 1
        assert r[2] == 2  # expert 6 -> slot 2
        assert r[3] == 0  # expert 0 -> fallback slot 0

    def test_rebuild_lookup_after_swap(self):
        pc = PredictiveExpertCache(capacity=2, num_experts=8)
        pc.build_lookup([0, 3])

        # Simulate a swap: replace expert 3 with expert 7
        pc.cached_ids[1] = 7
        pc.cached_set.discard(3)
        pc.cached_set.add(7)
        pc.rebuild_lookup()

        lookup_np = np.array(pc.lookup)
        assert lookup_np[0] == 0
        assert lookup_np[7] == 1
        assert lookup_np[3] == 0  # no longer cached

        mask_np = np.array(pc.hit_mask)
        assert mask_np[7] == 1.0
        assert mask_np[3] == 0.0

    def test_expert_to_slot_mapping_consistency(self):
        pc = PredictiveExpertCache(capacity=4, num_experts=32)
        ids = [3, 10, 20, 31]
        pc.build_lookup(ids)
        for slot, eid in enumerate(ids):
            assert np.array(pc.lookup)[eid] == slot

    def test_frequency_and_last_active_from_build(self):
        pc = PredictiveExpertCache(capacity=3, num_experts=8)
        pc.build_lookup([0, 2, 5])
        for eid in [0, 2, 5]:
            assert pc.frequency[eid] == 1
            assert pc.last_active[eid] == 0

    def test_pinned_set(self):
        pc = PredictiveExpertCache(capacity=4, num_experts=16)
        pc.pinned_set = {0, 1}
        assert 0 in pc.pinned_set
        assert 2 not in pc.pinned_set

    def test_indices_buffer(self):
        pc = PredictiveExpertCache(capacity=4, num_experts=8)
        pc.build_lookup([0, 1, 2, 3])
        pc._indices_buffer.append(mx.array([0, 1]))
        pc._indices_buffer.append(mx.array([2, 3]))
        assert len(pc._indices_buffer) == 2


# ---------------------------------------------------------------------------
# select_capacity
# ---------------------------------------------------------------------------

class TestSelectCapacity:

    def test_basic(self):
        cap = select_capacity(1.4, 24.0, num_moe_layers=48, expert_slot_mb=1.77)
        assert cap > 0
        assert cap <= 512

    def test_rounds_to_multiple_of_8(self):
        cap = select_capacity(1.4, 24.0, num_moe_layers=48, expert_slot_mb=1.77)
        if cap >= 16:
            assert cap % 8 == 0

    def test_never_negative(self):
        cap = select_capacity(100.0, 10.0, num_moe_layers=48, expert_slot_mb=1.77)
        assert cap >= 0

    def test_never_exceeds_512(self):
        cap = select_capacity(0.0, 1000.0, num_moe_layers=1, expert_slot_mb=0.001)
        assert cap <= 512

    def test_zero_slot_size(self):
        cap = select_capacity(1.4, 24.0, num_moe_layers=48, expert_slot_mb=0.0)
        assert cap == 0

    def test_large_base_exceeds_target(self):
        cap = select_capacity(50.0, 24.0, num_moe_layers=48, expert_slot_mb=1.77)
        assert cap == 0

    @pytest.mark.parametrize("recommended,expected_range", [
        (24.0, (150, 210)),
        (48.0, (350, 420)),
        (8.0, (10, 80)),
    ])
    def test_scales_with_recommended(self, recommended, expected_range):
        cap = select_capacity(1.4, recommended, num_moe_layers=48, expert_slot_mb=1.77)
        lo, hi = expected_range
        assert lo <= cap <= hi, f"cap={cap} not in [{lo}, {hi}] for recommended={recommended}"

    def test_small_capacity_no_rounding(self):
        # Contrive a case where capacity < 16
        cap = select_capacity(5.0, 8.0, num_moe_layers=48, expert_slot_mb=1.77)
        assert cap < 16 or cap % 8 == 0


# ---------------------------------------------------------------------------
# _find_switch_mlp / _find_moe_block
# ---------------------------------------------------------------------------

class TestFindSwitchMlp:

    def test_qwen_path(self):
        switch = SimpleNamespace()
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        found, prefix = _find_switch_mlp(layer, layer_idx=3)
        assert found is switch
        assert prefix == "model.layers.3.mlp.switch_mlp"

    def test_mixtral_path(self):
        switch = SimpleNamespace()
        bsm = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(block_sparse_moe=bsm)
        found, prefix = _find_switch_mlp(layer, layer_idx=0)
        assert found is switch
        assert prefix == "model.layers.0.block_sparse_moe.switch_mlp"

    def test_no_moe(self):
        layer = SimpleNamespace(mlp=SimpleNamespace())
        found, prefix = _find_switch_mlp(layer, layer_idx=5)
        assert found is None
        assert prefix is None

    def test_no_layer_idx(self):
        switch = SimpleNamespace()
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        found, prefix = _find_switch_mlp(layer)
        assert found is switch
        assert prefix == "mlp.switch_mlp"

    def test_qwen_preferred_over_mixtral(self):
        switch_q = SimpleNamespace()
        switch_m = SimpleNamespace()
        mlp = SimpleNamespace(switch_mlp=switch_q)
        bsm = SimpleNamespace(switch_mlp=switch_m)
        layer = SimpleNamespace(mlp=mlp, block_sparse_moe=bsm)
        found, _ = _find_switch_mlp(layer, layer_idx=0)
        assert found is switch_q

    def test_resolves_language_model_root_with_shard_map(self):
        switch = SimpleNamespace()
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        shard_map = {
            "language_model.model.layers.2.mlp.switch_mlp.gate_proj.weight": "x.safetensors"
        }
        found, prefix = _find_switch_mlp(layer, layer_idx=2, shard_map=shard_map)
        assert found is switch
        assert prefix == "language_model.model.layers.2.mlp.switch_mlp"


class TestFindMoeBlock:

    def test_qwen_returns_mlp(self):
        switch = SimpleNamespace()
        mlp = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(mlp=mlp)
        assert _find_moe_block(layer) is mlp

    def test_mixtral_returns_bsm(self):
        switch = SimpleNamespace()
        bsm = SimpleNamespace(switch_mlp=switch)
        layer = SimpleNamespace(block_sparse_moe=bsm)
        assert _find_moe_block(layer) is bsm

    def test_none_for_non_moe(self):
        layer = SimpleNamespace(mlp=SimpleNamespace())
        assert _find_moe_block(layer) is None


# ---------------------------------------------------------------------------
# _detect_num_experts
# ---------------------------------------------------------------------------

class TestDetectNumExperts:

    def test_from_gate_proj(self):
        gate = SimpleNamespace(num_experts=64)
        switch = SimpleNamespace(gate_proj=gate)
        assert _detect_num_experts(switch) == 64

    def test_from_up_proj(self):
        up = SimpleNamespace(num_experts=128)
        switch = SimpleNamespace(up_proj=up)
        assert _detect_num_experts(switch) == 128

    def test_from_down_proj(self):
        down = SimpleNamespace(num_experts=8)
        switch = SimpleNamespace(down_proj=down)
        assert _detect_num_experts(switch) == 8

    def test_prefers_gate_over_up(self):
        gate = SimpleNamespace(num_experts=64)
        up = SimpleNamespace(num_experts=999)
        switch = SimpleNamespace(gate_proj=gate, up_proj=up)
        assert _detect_num_experts(switch) == 64

    def test_raises_without_projection_metadata(self):
        switch = SimpleNamespace()
        with pytest.raises(ValueError, match="Could not detect num_experts"):
            _detect_num_experts(switch)

    def test_raises_when_proj_missing_num_experts(self):
        gate = SimpleNamespace()  # has no num_experts
        switch = SimpleNamespace(gate_proj=gate)
        with pytest.raises(ValueError, match="Could not detect num_experts"):
            _detect_num_experts(switch)


# ---------------------------------------------------------------------------
# compute_adaptive_allocations
# ---------------------------------------------------------------------------

class TestComputeAdaptiveAllocations:

    def _make_profile(self, expert_counts):
        """Create layer_profiles from a list of (layer_idx, [(eid, count), ...])."""
        return {
            li: {
                "working_set": ws,
                "entropy": 0.0,
                "unique_count": len(ws),
            }
            for li, ws in expert_counts
        }

    def test_uniform_layers(self):
        profiles = self._make_profile([
            (0, [(i, 10) for i in range(64)]),
            (1, [(i, 10) for i in range(64)]),
        ])
        result = compute_adaptive_allocations(profiles, total_budget=64, min_per_layer=8)
        assert sum(result["allocations"].values()) == 64
        assert result["allocations"][0] == 32
        assert result["allocations"][1] == 32

    def test_concentrated_vs_uniform(self):
        # Layer 0: concentrated (top 4 experts have 90% of traffic)
        ws_concentrated = [(i, 100) for i in range(4)] + [(i, 1) for i in range(4, 64)]
        # Layer 1: uniform
        ws_uniform = [(i, 10) for i in range(64)]
        profiles = self._make_profile([
            (0, ws_concentrated),
            (1, ws_uniform),
        ])
        result = compute_adaptive_allocations(profiles, total_budget=40, min_per_layer=4)
        allocs = result["allocations"]
        assert sum(allocs.values()) == 40
        # Concentrated layer needs fewer slots, uniform gets more
        assert allocs[0] < allocs[1]

    def test_min_per_layer_respected(self):
        profiles = self._make_profile([
            (0, [(i, 10) for i in range(8)]),
            (1, [(i, 10) for i in range(8)]),
        ])
        result = compute_adaptive_allocations(profiles, total_budget=64, min_per_layer=16)
        for alloc in result["allocations"].values():
            assert alloc >= 16

    def test_miss_rates_zero_when_fully_covered(self):
        profiles = self._make_profile([
            (0, [(i, 10) for i in range(8)]),
        ])
        result = compute_adaptive_allocations(profiles, total_budget=8, min_per_layer=4)
        assert result["miss_rates"][0] == 0.0

    def test_miss_rates_nonzero_when_underfunded(self):
        ws = [(i, 10) for i in range(64)]
        profiles = self._make_profile([(0, ws)])
        result = compute_adaptive_allocations(profiles, total_budget=4, min_per_layer=4)
        assert result["miss_rates"][0] > 0.0

    def test_empty_working_set(self):
        profiles = self._make_profile([(0, [])])
        result = compute_adaptive_allocations(profiles, total_budget=8, min_per_layer=4)
        assert result["miss_rates"][0] == 0.0

    def test_iterations_returned(self):
        profiles = self._make_profile([
            (0, [(i, 10) for i in range(64)]),
            (1, [(i, 10) for i in range(64)]),
        ])
        result = compute_adaptive_allocations(profiles, total_budget=64, min_per_layer=8)
        assert "iterations" in result
        assert isinstance(result["iterations"], int)

    def test_budget_preserved_across_many_layers(self):
        profiles = self._make_profile([
            (i, [(e, 10) for e in range(32)])
            for i in range(10)
        ])
        budget = 200
        result = compute_adaptive_allocations(profiles, total_budget=budget, min_per_layer=8)
        assert sum(result["allocations"].values()) == budget
