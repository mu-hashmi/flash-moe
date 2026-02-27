import gc
import json
import math
import random
import statistics
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models import cache as mlx_cache
from mlx.utils import tree_reduce

from benchmarks import profile_experts as pe
from mlx_moe.lazy_experts.core import (
    dynamic_cache_update,
    get_fallback_stats,
)
from mlx_moe.lazy_experts.generate import _load_model_and_tokenizer, _startup
from mlx_moe.lazy_experts.loading import _detect_num_experts, _find_switch_mlp
from mlx_moe.lazy_experts import enable_lazy_experts, upgrade_to_predictive


RESULTS_DIR = Path("benchmarks/results")
TOKENS_TASK_1 = 200
TOKENS_TASK_3 = 200
TOKENS_TASK_4 = 500

QUALITY_PROMPTS = [
    {
        "id": "code",
        "text": (
            "Write a Python implementation of an interval tree that supports insert, "
            "delete, and overlap queries. Include complexity analysis and one short usage example."
        ),
    },
    {
        "id": "reasoning",
        "text": (
            "A vault has 5 locks and 8 keys. Exactly 5 keys open one lock each, and 3 are decoys. "
            "You can test one key on one lock per minute. Give a strategy that minimizes expected time "
            "to open all locks and explain why."
        ),
    },
    {
        "id": "multilingual",
        "text": (
            "Explain in Spanish and Chinese how dynamic programming differs from greedy algorithms, "
            "and provide one example problem for each approach."
        ),
    },
    {
        "id": "factual",
        "text": (
            "Give a concise timeline of major milestones in the development of modern transformer models "
            "from 2017 onward, with one sentence per milestone."
        ),
    },
    {
        "id": "creative",
        "text": (
            "Write a short science-fiction scene where an engineer debugs a city-scale AI during a solar storm."
        ),
    },
]

LAYER_FALLBACK_PROMPTS = [
    "Design a lock-free queue in C++ and explain memory ordering choices.",
    "Explain the difference between soundness and completeness in formal logic with examples.",
    "Summarize the key tradeoffs between vector databases and keyword search for code retrieval.",
]

PINNING_PROMPT_35B = (
    "Implement a production-ready Python rate limiter middleware for FastAPI with Redis backing, "
    "sliding-window counters, and structured error responses. Include unit tests."
)

NEW_DOMAIN_PROMPTS = [
    "Draft a legal memo outline for a software licensing dispute involving API terms and derivative works.",
    "Explain differential diagnosis workflow for chest pain in emergency medicine with a triage-first structure.",
    "Prove that every finite group of prime order is cyclic, with a compact formal proof.",
    "Write a noir fiction opening scene set in a data center during a city blackout.",
    "Create a lesson plan to teach introductory macroeconomics using inflation and unemployment case studies.",
    "Design an experiment in cognitive psychology to test working-memory limits under multitasking.",
    "Summarize major obligations in GDPR for handling user deletion requests in SaaS products.",
    "Solve a constrained optimization problem with Lagrange multipliers and explain each step.",
    "Write a concise pathology report template for a biopsy sample with standardized sections.",
    "Propose a climate-risk stress-test framework for a regional bank loan portfolio.",
    "Compose a historical analysis of competing causes of the 1848 European revolutions.",
    "Explain a practical incident-response playbook for ransomware in a hospital network.",
]

MODEL_RUNS = [
    {
        "id": "qwen3_coder_next",
        "model": "mlx-community/Qwen3-Coder-Next-4bit",
        "capacity": 208,
        "profile": "profiles/qwen3-coder-next-4bit.json",
        "native_scoring": True,
    },
    {
        "id": "qwen35_35b_a3b",
        "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "capacity": 248,
        "profile": "profiles/qwen3.5-35b-a3b-4bit.json",
        "native_scoring": True,
    },
    {
        "id": "qwen35_122b_a10b",
        "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "capacity": 68,
        "profile": "profiles/qwen3.5-122b-a10b-4bit.json",
        "native_scoring": False,
    },
]


def apply_chat_template(tokenizer, text: str) -> str:
    if not getattr(tokenizer, "has_chat_template", False):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


def close_model(model) -> None:
    st_map = getattr(model, "_st_map", None)
    if st_map is not None:
        st_map.close()
    del model
    mx.clear_cache()
    gc.collect()


def detect_num_experts(model) -> int:
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is not None:
            return _detect_num_experts(switch)
    raise RuntimeError("No MoE layers found in model")


def inspect_moe_geometry(model) -> dict:
    moe_layers = 0
    num_experts = 0
    expert_slot_mb = 0.0

    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        moe_layers += 1
        if num_experts == 0:
            num_experts = _detect_num_experts(switch)
        if expert_slot_mb == 0.0 and num_experts > 0:
            proj = getattr(switch, "gate_proj", None) or getattr(switch, "up_proj", None)
            if proj is not None and hasattr(proj, "weight"):
                per_expert = proj.weight.nbytes // num_experts
                for attr in ("scales", "biases"):
                    tensor = getattr(proj, attr, None)
                    if tensor is not None:
                        per_expert += tensor.nbytes // num_experts
                expert_slot_mb = (per_expert * 3) / 1e6

    return {
        "moe_layers": moe_layers,
        "num_experts": num_experts,
        "expert_slot_mb": expert_slot_mb,
    }


def estimate_startup_memory(model_name: str, capacity: int) -> dict:
    model, tokenizer, model_path = _load_model_and_tokenizer(model_name, lazy=True)
    geometry = inspect_moe_geometry(model)
    effective_capacity = min(capacity, geometry["num_experts"]) if geometry["num_experts"] > 0 else capacity

    enable_lazy_experts(
        model,
        model_path,
        cache_capacity_per_layer=0,
        predictive=True,
    )
    mx.eval(model.parameters())
    base_gb = mx.get_active_memory() / 1e9
    expert_gb = (
        geometry["moe_layers"] * effective_capacity * geometry["expert_slot_mb"] / 1e3
    )
    predicted_gb = base_gb + expert_gb
    budget_gb = mx.device_info()["memory_size"] / 1e9 * 0.95

    close_model(model)
    del tokenizer

    return {
        "base_gb": base_gb,
        "expert_gb": expert_gb,
        "predicted_gb": predicted_gb,
        "budget_gb": budget_gb,
        "effective_capacity": effective_capacity,
        "moe_layers": geometry["moe_layers"],
        "num_experts": geometry["num_experts"],
    }


def load_native_model(model_name: str):
    model, tokenizer, _ = _load_model_and_tokenizer(model_name, lazy=True)
    return model, tokenizer


def estimate_model_bytes(model) -> int:
    return tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc,
        model,
        0,
    )


def reset_fallback_counters(model) -> None:
    for i, layer in enumerate(model.layers):
        switch, _ = _find_switch_mlp(layer, i)
        if switch is None:
            continue
        proj = getattr(switch, "up_proj", None)
        cache = getattr(proj, "_cache", None)
        if cache is None:
            continue
        cache.total_requests = 0
        cache.total_fallbacks = 0
        cache._indices_buffer.clear()


def drain_dynamic_updates(model, passes: int = 2) -> None:
    for _ in range(passes):
        dynamic_cache_update(model, max_layer_updates=48)


def generate_token_trace(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    dynamic_updates: bool,
) -> dict:
    pieces = []
    token_ids = []
    final = None
    t0 = time.perf_counter()
    for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        pieces.append(response.text)
        token_ids.append(int(response.token))
        if dynamic_updates:
            dynamic_cache_update(model, max_layer_updates=48)
        final = response
    elapsed = time.perf_counter() - t0
    return {
        "tokens": token_ids,
        "text": "".join(pieces),
        "elapsed_s": elapsed,
        "generation_tps": float(final.generation_tps if final else 0.0),
        "finish_reason": final.finish_reason if final else None,
    }


def mean_logprob_for_continuation(
    model,
    prompt_tokens: list[int],
    continuation_tokens: list[int],
) -> float:
    if not continuation_tokens:
        return float("nan")

    prompt = mx.array(prompt_tokens, dtype=mx.uint32)
    prompt_cache = mlx_cache.make_prompt_cache(model)

    while prompt.size > 1:
        n_to_process = min(1024, int(prompt.size) - 1)
        model(prompt[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        prompt = prompt[n_to_process:]
        mx.clear_cache()

    logits = model(prompt[None], cache=prompt_cache)
    log_probs = nn.log_softmax(logits[:, -1, :].astype(mx.float32), axis=-1)
    scores = [float(log_probs[0, continuation_tokens[0]].item())]

    for i, (prev_token, target_token) in enumerate(
        zip(continuation_tokens[:-1], continuation_tokens[1:]),
        start=1,
    ):
        logits = model(mx.array([[prev_token]], dtype=mx.uint32), cache=prompt_cache)
        log_probs = nn.log_softmax(logits[:, -1, :].astype(mx.float32), axis=-1)
        scores.append(float(log_probs[0, target_token].item()))
        if i % 64 == 0:
            mx.clear_cache()

    return statistics.fmean(scores)


def repetition_score(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(grams))
    return 1.0 - (unique / len(grams))


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def run_perplexity_baseline() -> tuple[dict, dict]:
    print("\n[1/4] Running perplexity baseline comparison")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokens_per_prompt": TOKENS_TASK_1,
        "prompts": [{"id": p["id"], "text": p["text"]} for p in QUALITY_PROMPTS],
        "models": [],
    }
    summary = {"pass": True, "notes": []}

    for cfg in MODEL_RUNS:
        print(f"\n  Model: {cfg['model']} (capacity={cfg['capacity']})")
        preflight = estimate_startup_memory(cfg["model"], cfg["capacity"])
        model_out = {
            "id": cfg["id"],
            "model": cfg["model"],
            "capacity": cfg["capacity"],
            "native_scoring": cfg["native_scoring"],
            "preflight": preflight,
            "prompt_results": [],
        }

        if preflight["predicted_gb"] > preflight["budget_gb"]:
            reason = (
                f"Skipped: estimated startup memory {preflight['predicted_gb']:.1f} GB "
                f"exceeds budget {preflight['budget_gb']:.1f} GB"
            )
            model_out["skipped_reason"] = reason
            summary["pass"] = False
            summary["notes"].append(f"{cfg['id']} {reason}")
            result["models"].append(model_out)
            continue

        startup_prompt = QUALITY_PROMPTS[0]["text"]
        prod_model, prod_tokenizer, _ = _startup(
            cfg["model"],
            startup_prompt,
            profile_path=cfg["profile"],
            capacity=cfg["capacity"],
            warmup="hybrid",
            pin_top_k=None,
        )

        num_experts = detect_num_experts(prod_model)
        model_out["num_experts"] = num_experts
        model_out["pinning"] = "profile_universal"

        for prompt in QUALITY_PROMPTS:
            formatted = apply_chat_template(prod_tokenizer, prompt["text"])
            trace = generate_token_trace(
                prod_model,
                prod_tokenizer,
                formatted,
                TOKENS_TASK_1,
                dynamic_updates=True,
            )
            drain_dynamic_updates(prod_model, passes=3)
            prompt_tokens = encode_prompt(prod_tokenizer, formatted)
            mlx_mean_lp = mean_logprob_for_continuation(prod_model, prompt_tokens, trace["tokens"])
            mlx_ppl = math.exp(-mlx_mean_lp)

            model_out["prompt_results"].append(
                {
                    "prompt_id": prompt["id"],
                    "prompt": prompt["text"],
                    "mlx_output_tokens": len(trace["tokens"]),
                    "mlx_decode_tps": trace["generation_tps"],
                    "mlx_elapsed_s": trace["elapsed_s"],
                    "mlx_finish_reason": trace["finish_reason"],
                    "mlx_output_preview": trace["text"][:220],
                    "mlx_mean_logprob": mlx_mean_lp,
                    "mlx_perplexity": mlx_ppl,
                    "_formatted_prompt": formatted,
                    "_mlx_tokens": trace["tokens"],
                }
            )

        if cfg["native_scoring"]:
            close_model(prod_model)
            del prod_tokenizer

            full_model, full_tokenizer = load_native_model(cfg["model"])
            native_bytes = estimate_model_bytes(full_model)
            native_gb = native_bytes / 1e9
            memory_bytes = int(mx.device_info()["memory_size"])
            memory_gb = memory_bytes / 1e9
            model_out["native_model_bytes_gb"] = native_gb

            if native_bytes > int(memory_bytes * 0.9):
                reason = (
                    f"Native scoring skipped: estimated native model size {native_gb:.1f} GB "
                    f"exceeds host memory budget {memory_gb:.1f} GB on this 32GB machine"
                )
                model_out["native_scoring_skipped_reason"] = reason
                summary["pass"] = False
                summary["notes"].append(f"{cfg['id']} {reason}")

                for row in model_out["prompt_results"]:
                    row.pop("_formatted_prompt")
                    row.pop("_mlx_tokens")
                model_out["model_avg_mlx_perplexity"] = statistics.fmean(
                    row["mlx_perplexity"] for row in model_out["prompt_results"]
                )

                close_model(full_model)
                del full_tokenizer
                result["models"].append(model_out)
                continue

            retention_values = []
            native_ppl_values = []
            mlx_ppl_values = []
            for row in model_out["prompt_results"]:
                formatted = row["_formatted_prompt"]
                prompt_tokens = encode_prompt(full_tokenizer, formatted)
                mlx_tokens = row["_mlx_tokens"]

                native_trace = generate_token_trace(
                    full_model,
                    full_tokenizer,
                    formatted,
                    TOKENS_TASK_1,
                    dynamic_updates=False,
                )
                native_tokens = native_trace["tokens"]

                native_mean_lp_on_mlx = mean_logprob_for_continuation(
                    full_model, prompt_tokens, mlx_tokens
                )
                native_mean_lp_on_native = mean_logprob_for_continuation(
                    full_model, prompt_tokens, native_tokens
                )

                retention = math.exp(native_mean_lp_on_mlx - native_mean_lp_on_native) * 100.0
                retention_values.append(retention)
                native_ppl_values.append(math.exp(-native_mean_lp_on_native))
                mlx_ppl_values.append(row["mlx_perplexity"])

                row.update(
                    {
                        "native_output_tokens": len(native_tokens),
                        "native_output_preview": native_trace["text"][:220],
                        "native_decode_tps": native_trace["generation_tps"],
                        "native_elapsed_s": native_trace["elapsed_s"],
                        "native_finish_reason": native_trace["finish_reason"],
                        "native_mean_logprob_on_mlx_output": native_mean_lp_on_mlx,
                        "native_mean_logprob_on_native_output": native_mean_lp_on_native,
                        "quality_retention_pct": retention,
                    }
                )

                row.pop("_formatted_prompt")
                row.pop("_mlx_tokens")

            model_out["model_avg_quality_retention_pct"] = statistics.fmean(retention_values)
            model_out["model_avg_native_perplexity"] = statistics.fmean(native_ppl_values)
            model_out["model_avg_mlx_perplexity"] = statistics.fmean(mlx_ppl_values)

            if model_out["model_avg_quality_retention_pct"] < 90.0:
                summary["pass"] = False
                summary["notes"].append(
                    f"{cfg['id']} quality retention below expectation: "
                    f"{model_out['model_avg_quality_retention_pct']:.2f}%"
                )

            close_model(full_model)
            del full_tokenizer
        else:
            model_out["native_scoring_skipped_reason"] = (
                "Full-capacity native scoring skipped on 32GB hardware for 122B model"
            )
            model_out["model_avg_mlx_perplexity"] = statistics.fmean(
                row["mlx_perplexity"] for row in model_out["prompt_results"]
            )
            for row in model_out["prompt_results"]:
                row.pop("_formatted_prompt")
                row.pop("_mlx_tokens")

            close_model(prod_model)
            del prod_tokenizer

        result["models"].append(model_out)

    return result, summary


def run_profile_stability() -> tuple[dict, dict]:
    print("\n[2/4] Running profile stability test")

    model_name = "mlx-community/Qwen3-Coder-Next-4bit"
    capacity = 208
    threshold = 0.5

    baseline_items = pe.resolve_prompt_items(
        "mixed", coding_weight=70, toolchat_weight=30, num_prompts=24, seed=0
    )
    rng = random.Random(7)
    subset_indices = sorted(rng.sample(range(len(baseline_items)), 12))
    subset_items = [deepcopy(baseline_items[idx]) for idx in subset_indices]
    new_domain_items = NEW_DOMAIN_PROMPTS

    runs_spec = [
        ("mixed_24", baseline_items),
        ("mixed_subset_12", subset_items),
        ("new_domains_12", new_domain_items),
    ]

    runs_output = {}
    layer_universal_sets = {}
    all_layers = set()

    for run_id, prompt_items in runs_spec:
        print(f"  Profiling run: {run_id} ({len(prompt_items)} prompts)")
        model, tokenizer, model_path = _load_model_and_tokenizer(model_name, lazy=True)
        use_chat_template = bool(getattr(tokenizer, "has_chat_template", False))

        enable_lazy_experts(
            model,
            model_path,
            cache_capacity_per_layer=capacity,
            predictive=True,
        )
        mx.eval(model.parameters())

        _, bootstrap_prompt = pe.render_prompt_item(tokenizer, prompt_items[0], use_chat_template)
        pe.router_only_discovery(model, tokenizer, bootstrap_prompt, max_tokens=pe.WARMUP_TOKENS)
        upgrade_to_predictive(model, model_path, capacity)

        activation_counts = defaultdict(lambda: defaultdict(int))

        for item in prompt_items:
            _, prompt = pe.render_prompt_item(tokenizer, item, use_chat_template)
            layer_experts = pe.collect_expert_activations(
                model,
                tokenizer,
                model_path,
                prompt,
                capacity,
                discovery_mode="router-only",
            )
            for layer_idx, experts in layer_experts.items():
                for eid in experts:
                    activation_counts[layer_idx][eid] += 1

        universal_by_layer = {}
        min_count = int(threshold * len(prompt_items))

        for i, layer in enumerate(model.layers):
            switch, _ = _find_switch_mlp(layer, i)
            if switch is None:
                continue
            all_layers.add(i)
            counts = activation_counts.get(i, {})
            universal = sorted(eid for eid, cnt in counts.items() if cnt >= min_count)
            universal_by_layer[i] = universal

        runs_output[run_id] = {
            "num_prompts": len(prompt_items),
            "threshold": threshold,
            "min_count": min_count,
            "universal_by_layer": {str(k): v for k, v in sorted(universal_by_layer.items())},
        }
        layer_universal_sets[run_id] = {k: set(v) for k, v in universal_by_layer.items()}

        close_model(model)
        del tokenizer

    ordered_layers = sorted(all_layers)

    pairs = [
        ("mixed_24", "mixed_subset_12"),
        ("mixed_24", "new_domains_12"),
        ("mixed_subset_12", "new_domains_12"),
    ]
    pairwise = {}

    for a, b in pairs:
        scores = []
        for layer in ordered_layers:
            set_a = layer_universal_sets[a].get(layer, set())
            set_b = layer_universal_sets[b].get(layer, set())
            scores.append(jaccard(set_a, set_b))
        pairwise[f"{a}__vs__{b}"] = {
            "avg_jaccard": statistics.fmean(scores),
            "per_layer_jaccard": {
                str(layer): score for layer, score in zip(ordered_layers, scores)
            },
        }

    subset_vs_full = pairwise["mixed_24__vs__mixed_subset_12"]["avg_jaccard"]
    new_vs_full = pairwise["mixed_24__vs__new_domains_12"]["avg_jaccard"]

    summary = {"pass": subset_vs_full > 0.8, "notes": []}
    if subset_vs_full <= 0.8:
        summary["notes"].append(
            f"mixed_subset_12 vs mixed_24 avg Jaccard {subset_vs_full:.3f} <= 0.8"
        )
    if new_vs_full + 0.1 < subset_vs_full:
        summary["notes"].append(
            "New-domain prompts are materially less stable than same-domain subset; profile appears domain-sensitive"
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "capacity": capacity,
        "threshold": threshold,
        "run_specs": {
            "mixed_24": {
                "preset": "mixed",
                "coding_weight": 70,
                "toolchat_weight": 30,
                "num_prompts": 24,
                "seed": 0,
            },
            "mixed_subset_12": {
                "subset_indices": subset_indices,
                "source": "mixed_24",
            },
            "new_domains_12": {
                "prompts": NEW_DOMAIN_PROMPTS,
            },
        },
        "runs": runs_output,
        "pairwise": pairwise,
        "interpretation": {
            "sample_size_adequate": subset_vs_full > 0.8,
            "domain_sensitive_signal": new_vs_full + 0.1 < subset_vs_full,
        },
    }

    return result, summary


def run_layer_fallback_distribution() -> tuple[dict, dict]:
    print("\n[3/4] Running layer-position-weighted fallback test")
    model_name = "mlx-community/Qwen3-Coder-Next-4bit"
    capacity = 208
    profile = "profiles/qwen3-coder-next-4bit.json"

    model, tokenizer, _ = _startup(
        model_name,
        LAYER_FALLBACK_PROMPTS[0],
        profile_path=profile,
        capacity=capacity,
        warmup="hybrid",
        pin_top_k=None,
    )

    aggregated = defaultdict(lambda: {"requests": 0, "fallbacks": 0})
    prompt_rows = []

    for prompt_text in LAYER_FALLBACK_PROMPTS:
        prompt = apply_chat_template(tokenizer, prompt_text)
        reset_fallback_counters(model)

        trace = generate_token_trace(
            model,
            tokenizer,
            prompt,
            TOKENS_TASK_3,
            dynamic_updates=True,
        )
        drain_dynamic_updates(model, passes=3)

        stats = get_fallback_stats(model)
        per_layer = []
        for row in stats["layers"]:
            layer_idx = int(row["layer"])
            aggregated[layer_idx]["requests"] += int(row["requests"])
            aggregated[layer_idx]["fallbacks"] += int(row["fallbacks"])
            per_layer.append(
                {
                    "layer": layer_idx,
                    "requests": int(row["requests"]),
                    "fallbacks": int(row["fallbacks"]),
                    "fallback_rate": float(row["fallback_rate"]),
                }
            )
        prompt_rows.append(
            {
                "prompt": prompt_text,
                "generated_tokens": len(trace["tokens"]),
                "decode_tps": trace["generation_tps"],
                "per_layer": sorted(per_layer, key=lambda x: x["layer"]),
            }
        )

    per_layer_agg = []
    for layer_idx in sorted(aggregated):
        req = aggregated[layer_idx]["requests"]
        fb = aggregated[layer_idx]["fallbacks"]
        per_layer_agg.append(
            {
                "layer": layer_idx,
                "requests": req,
                "fallbacks": fb,
                "fallback_rate": (fb / req) if req > 0 else 0.0,
            }
        )

    buckets = {
        "early_0_15": range(0, 16),
        "middle_16_31": range(16, 32),
        "late_32_47": range(32, 48),
    }

    bucket_rows = {}
    for name, layer_range in buckets.items():
        rows = [row for row in per_layer_agg if row["layer"] in layer_range]
        total_req = sum(row["requests"] for row in rows)
        total_fb = sum(row["fallbacks"] for row in rows)
        avg_layer_rate = statistics.fmean(row["fallback_rate"] for row in rows) if rows else 0.0
        bucket_rows[name] = {
            "layers": [row["layer"] for row in rows],
            "avg_layer_fallback_rate": avg_layer_rate,
            "request_weighted_fallback_rate": (total_fb / total_req) if total_req > 0 else 0.0,
            "requests": total_req,
            "fallbacks": total_fb,
        }

    rates = [bucket_rows[name]["avg_layer_fallback_rate"] for name in buckets]
    skew = max(rates) - min(rates)

    summary = {"pass": skew < 0.05, "notes": []}
    if skew >= 0.05:
        summary["notes"].append(
            f"Bucket fallback skew is high (max-min={skew:.3f})"
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "capacity": capacity,
        "tokens_per_prompt": TOKENS_TASK_3,
        "prompts": LAYER_FALLBACK_PROMPTS,
        "per_prompt": prompt_rows,
        "per_layer_aggregated": per_layer_agg,
        "bucket_summary": bucket_rows,
        "max_min_bucket_skew": skew,
    }

    close_model(model)
    del tokenizer

    return result, summary


def run_pinning_sanity_35b() -> tuple[dict, dict]:
    print("\n[4/4] Running 35B pinning sanity check")
    model_name = "mlx-community/Qwen3.5-35B-A3B-4bit"
    capacity = 248
    profile = "profiles/qwen3.5-35b-a3b-4bit.json"
    preflight = estimate_startup_memory(model_name, capacity)

    if preflight["predicted_gb"] > preflight["budget_gb"]:
        reason = (
            f"Skipped: estimated startup memory {preflight['predicted_gb']:.1f} GB "
            f"exceeds budget {preflight['budget_gb']:.1f} GB"
        )
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "capacity": capacity,
            "preflight": preflight,
            "skipped_reason": reason,
            "results": [],
        }
        summary = {"pass": False, "notes": [reason]}
        return result, summary

    configs = [
        {"name": "no_pinning", "pin_top_k": 0},
        {"name": "pin_top_k_32", "pin_top_k": 32},
        {"name": "pin_top_k_64", "pin_top_k": 64},
    ]

    rows = []

    for cfg in configs:
        print(f"  Config: {cfg['name']}")
        model, tokenizer, _ = _startup(
            model_name,
            PINNING_PROMPT_35B,
            profile_path=profile,
            capacity=capacity,
            warmup="hybrid",
            pin_top_k=cfg["pin_top_k"],
        )

        prompt = apply_chat_template(tokenizer, PINNING_PROMPT_35B)
        reset_fallback_counters(model)

        trace = generate_token_trace(
            model,
            tokenizer,
            prompt,
            TOKENS_TASK_4,
            dynamic_updates=True,
        )
        drain_dynamic_updates(model, passes=3)

        fallback = get_fallback_stats(model)
        tokens = trace["tokens"]

        rows.append(
            {
                "config": cfg["name"],
                "pin_top_k": cfg["pin_top_k"],
                "generated_tokens": len(tokens),
                "decode_tps": trace["generation_tps"],
                "fallback_rate": float(fallback["fallback_rate"]),
                "repetition_score_300": repetition_score(tokens[:300]),
                "repetition_score_500": repetition_score(tokens[:500]),
                "output_preview": trace["text"][:220],
            }
        )

        close_model(model)
        del tokenizer

    decode_values = [row["decode_tps"] for row in rows]
    fallback_values = [row["fallback_rate"] for row in rows]
    rep_values = [row["repetition_score_500"] for row in rows]

    decode_spread = (max(decode_values) - min(decode_values)) / max(decode_values)
    fallback_spread = max(fallback_values) - min(fallback_values)
    repetition_spread = max(rep_values) - min(rep_values)

    summary = {
        "pass": decode_spread < 0.10 and fallback_spread < 0.01 and repetition_spread < 0.05,
        "notes": [],
    }
    if decode_spread >= 0.10:
        summary["notes"].append(f"Decode spread across pinning configs is {decode_spread:.3f}")
    if fallback_spread >= 0.01:
        summary["notes"].append(f"Fallback spread across pinning configs is {fallback_spread:.3f}")
    if repetition_spread >= 0.05:
        summary["notes"].append(f"Repetition spread across pinning configs is {repetition_spread:.3f}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "capacity": capacity,
        "prompt": PINNING_PROMPT_35B,
        "tokens": TOKENS_TASK_4,
        "results": rows,
        "spreads": {
            "decode_tps_relative_spread": decode_spread,
            "fallback_rate_abs_spread": fallback_spread,
            "repetition_500_abs_spread": repetition_spread,
        },
    }

    return result, summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    perplexity_result, perplexity_summary = run_perplexity_baseline()
    write_json(RESULTS_DIR / "perplexity_baseline.json", perplexity_result)

    profile_result, profile_summary = run_profile_stability()
    write_json(RESULTS_DIR / "profile_stability.json", profile_result)

    layer_result, layer_summary = run_layer_fallback_distribution()
    write_json(RESULTS_DIR / "layer_fallback_distribution.json", layer_result)

    pinning_result, pinning_summary = run_pinning_sanity_35b()
    write_json(RESULTS_DIR / "pinning_35b.json", pinning_result)

    elapsed = time.perf_counter() - t0

    print("\n=== Validation Sweep Summary ===")
    print(
        f"Perplexity baseline comparison: {'PASS' if perplexity_summary['pass'] else 'FAIL'}"
    )
    for note in perplexity_summary["notes"]:
        print(f"  - {note}")

    print(
        f"Profile stability test: {'PASS' if profile_summary['pass'] else 'FAIL'}"
    )
    for note in profile_summary["notes"]:
        print(f"  - {note}")

    print(
        f"Layer-position fallback distribution: {'PASS' if layer_summary['pass'] else 'FAIL'}"
    )
    for note in layer_summary["notes"]:
        print(f"  - {note}")

    print(
        f"Pinning sanity check (35B): {'PASS' if pinning_summary['pass'] else 'FAIL'}"
    )
    for note in pinning_summary["notes"]:
        print(f"  - {note}")

    surprises = []
    for note in (
        perplexity_summary["notes"]
        + profile_summary["notes"]
        + layer_summary["notes"]
        + pinning_summary["notes"]
    ):
        surprises.append(note)

    if surprises:
        print("Surprises:")
        for item in surprises:
            print(f"  - {item}")
    else:
        print("Surprises: none")

    print(f"Total elapsed: {elapsed/60.0:.1f} min")


if __name__ == "__main__":
    main()
