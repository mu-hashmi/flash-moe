import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models import cache as mlx_cache

from mlx_moe.lazy_experts.core import dynamic_cache_update
from mlx_moe.lazy_experts.generate import _startup

TOKENS = 200
PHASE1_PATH = Path("benchmarks/results/perplexity_phase1_tokens.json")
OUTPUT_PATH = Path("benchmarks/results/perplexity_baseline.json")

PROMPTS = [
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

MODELS = [
    {
        "id": "qwen3_coder_next",
        "model": "mlx-community/Qwen3-Coder-Next-4bit",
        "capacity": 208,
        "profile": "profiles/qwen3-coder-next-4bit.json",
        "native_phase2": True,
    },
    {
        "id": "qwen35_35b_a3b",
        "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "capacity": 248,
        "profile": "profiles/qwen3.5-35b-a3b-4bit.json",
        "native_phase2": True,
    },
    {
        "id": "qwen35_122b_a10b",
        "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "capacity": 68,
        "profile": "profiles/qwen3.5-122b-a10b-4bit.json",
        "native_phase2": False,
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


def run_phase1_for_model(config: dict) -> dict:
    model, tokenizer, _ = _startup(
        config["model"],
        PROMPTS[0]["text"],
        profile_path=config["profile"],
        capacity=config["capacity"],
        warmup="hybrid",
        pin_top_k=None,
    )

    rows = []
    for prompt in PROMPTS:
        formatted = apply_chat_template(tokenizer, prompt["text"])
        prompt_token_ids = encode_prompt(tokenizer, formatted)

        token_ids = []
        pieces = []
        final = None
        t0 = time.perf_counter()
        for response in mlx_lm.stream_generate(model, tokenizer, prompt=formatted, max_tokens=TOKENS):
            token_ids.append(int(response.token))
            pieces.append(response.text)
            dynamic_cache_update(model, max_layer_updates=48)
            final = response
        elapsed = time.perf_counter() - t0
        for _ in range(3):
            dynamic_cache_update(model, max_layer_updates=48)

        mlx_mean_lp = mean_logprob_for_continuation(model, prompt_token_ids, token_ids)

        rows.append(
            {
                "prompt_id": prompt["id"],
                "prompt": prompt["text"],
                "formatted_prompt": formatted,
                "prompt_token_ids": prompt_token_ids,
                "generated_token_ids": token_ids,
                "generated_tokens": len(token_ids),
                "mlx_decode_tps": float(final.generation_tps if final else 0.0),
                "mlx_elapsed_s": elapsed,
                "mlx_output_preview": "".join(pieces)[:220],
                "mlx_mean_logprob": mlx_mean_lp,
                "mlx_perplexity": math.exp(-mlx_mean_lp),
            }
        )

    close_model(model)
    del tokenizer

    return {
        "status": "ok",
        "id": config["id"],
        "model": config["model"],
        "capacity": config["capacity"],
        "pinning": "profile_universal",
        "prompts": rows,
        "model_avg_mlx_perplexity": statistics.fmean(r["mlx_perplexity"] for r in rows),
    }


def load_native_model(model_name: str):
    try:
        return mlx_lm.load(model_name, lazy=False), False
    except ValueError as exc:
        msg = str(exc)
        if "vision_tower" not in msg:
            raise
        from mlx_moe.lazy_experts.generate import _load_model_and_tokenizer

        model, tokenizer, _ = _load_model_and_tokenizer(model_name, lazy=False)
        return (model, tokenizer), True


def run_phase2_for_model(model_row: dict) -> dict:
    (model, tokenizer), used_fallback = load_native_model(model_row["model"])

    scored = []
    for row in model_row["prompts"]:
        native_mean_lp = mean_logprob_for_continuation(
            model,
            row["prompt_token_ids"],
            row["generated_token_ids"],
        )
        scored.append(
            {
                "prompt_id": row["prompt_id"],
                "native_mean_logprob_on_mlx_tokens": native_mean_lp,
                "native_perplexity_on_mlx_tokens": math.exp(-native_mean_lp),
                "quality_retention_vs_mlx_pct": math.exp(native_mean_lp - row["mlx_mean_logprob"]) * 100.0,
            }
        )

    close_model(model)
    del tokenizer

    return {
        "status": "ok",
        "id": model_row["id"],
        "native_load_fallback_strict_false": used_fallback,
        "prompt_scores": scored,
        "model_avg_native_perplexity_on_mlx_tokens": statistics.fmean(
            r["native_perplexity_on_mlx_tokens"] for r in scored
        ),
        "model_avg_quality_retention_vs_mlx_pct": statistics.fmean(
            r["quality_retention_vs_mlx_pct"] for r in scored
        ),
    }


def run_child_phase1(model_id: str, output: Path) -> int:
    config = next(c for c in MODELS if c["id"] == model_id)
    payload = run_phase1_for_model(config)
    output.write_text(json.dumps(payload), encoding="utf-8")
    return 0


def run_child_phase2(input_path: Path, output: Path) -> int:
    model_row = json.loads(input_path.read_text(encoding="utf-8"))
    payload = run_phase2_for_model(model_row)
    output.write_text(json.dumps(payload), encoding="utf-8")
    return 0


def run_parent() -> int:
    phase1_models = []

    for config in MODELS:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-phase1",
            config["id"],
            "--output",
            str(tmp_path),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            phase1_models.append(
                {
                    "status": "failed",
                    "id": config["id"],
                    "model": config["model"],
                    "capacity": config["capacity"],
                    "phase1_error": "timeout",
                }
            )
            tmp_path.unlink(missing_ok=True)
            continue

        if proc.returncode != 0 or not tmp_path.exists():
            phase1_models.append(
                {
                    "status": "failed",
                    "id": config["id"],
                    "model": config["model"],
                    "capacity": config["capacity"],
                    "phase1_error": f"child_exit_{proc.returncode}",
                }
            )
            tmp_path.unlink(missing_ok=True)
            continue

        phase1_models.append(json.loads(tmp_path.read_text(encoding="utf-8")))
        tmp_path.unlink(missing_ok=True)

    phase1_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokens_per_prompt": TOKENS,
        "models": phase1_models,
    }
    PHASE1_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE1_PATH.write_text(json.dumps(phase1_payload, indent=2), encoding="utf-8")

    final_models = []
    for model_row in phase1_models:
        out_row = {
            "id": model_row["id"],
            "model": model_row["model"],
            "capacity": model_row["capacity"],
        }

        if model_row.get("status") != "ok":
            out_row["status"] = "failed"
            out_row["reason"] = f"phase1 failed: {model_row.get('phase1_error', 'unknown')}"
            final_models.append(out_row)
            continue

        out_row["status"] = "ok"
        out_row["prompts"] = model_row["prompts"]
        out_row["model_avg_mlx_perplexity"] = model_row["model_avg_mlx_perplexity"]

        cfg = next(c for c in MODELS if c["id"] == model_row["id"])
        if not cfg["native_phase2"]:
            out_row["native_scoring_skipped_reason"] = "122B native model requires >32GB hardware"
            final_models.append(out_row)
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as in_tmp:
            in_path = Path(in_tmp.name)
            in_path.write_text(json.dumps(model_row), encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as out_tmp:
            out_path = Path(out_tmp.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-phase2",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
        ]

        timeout_s = 600 if model_row["id"] == "qwen3_coder_next" else 1200

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = None

        in_path.unlink(missing_ok=True)

        if timed_out:
            out_row["native_scoring_skipped_reason"] = "native scoring exceeded 10 minutes; requires >32GB hardware"
            out_path.unlink(missing_ok=True)
            final_models.append(out_row)
            continue

        if proc is None or proc.returncode != 0 or not out_path.exists():
            out_row["native_scoring_skipped_reason"] = "native scoring OOM or failed; requires >32GB hardware"
            out_path.unlink(missing_ok=True)
            final_models.append(out_row)
            continue

        phase2_row = json.loads(out_path.read_text(encoding="utf-8"))
        out_path.unlink(missing_ok=True)

        scores = {s["prompt_id"]: s for s in phase2_row["prompt_scores"]}
        for prompt_row in out_row["prompts"]:
            prompt_row.update(scores[prompt_row["prompt_id"]])

        out_row["model_avg_native_perplexity_on_mlx_tokens"] = phase2_row[
            "model_avg_native_perplexity_on_mlx_tokens"
        ]
        out_row["model_avg_quality_retention_vs_mlx_pct"] = phase2_row[
            "model_avg_quality_retention_vs_mlx_pct"
        ]
        out_row["native_load_fallback_strict_false"] = phase2_row[
            "native_load_fallback_strict_false"
        ]

        final_models.append(out_row)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "two_phase_sequential",
        "tokens_per_prompt": TOKENS,
        "prompts": PROMPTS,
        "phase1_tokens_path": str(PHASE1_PATH),
        "models": final_models,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Perplexity baseline (two-phase) complete")
    for row in final_models:
        if "native_scoring_skipped_reason" in row:
            print(f"- {row['id']}: native skipped ({row['native_scoring_skipped_reason']})")
        else:
            print(
                f"- {row['id']}: mlx ppl={row['model_avg_mlx_perplexity']:.4f}, "
                f"native ppl on mlx={row.get('model_avg_native_perplexity_on_mlx_tokens', float('nan')):.4f}, "
                f"retention={row.get('model_avg_quality_retention_vs_mlx_pct', float('nan')):.2f}%"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-phase perplexity baseline runner")
    parser.add_argument("--child-phase1", type=str, default=None)
    parser.add_argument("--child-phase2", action="store_true")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.child_phase1 is not None:
        if not args.output:
            raise ValueError("--output is required with --child-phase1")
        return run_child_phase1(args.child_phase1, Path(args.output))

    if args.child_phase2:
        if not args.input or not args.output:
            raise ValueError("--input and --output are required with --child-phase2")
        return run_child_phase2(Path(args.input), Path(args.output))

    return run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
