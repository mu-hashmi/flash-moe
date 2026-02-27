import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlx_lm

from mlx_moe.lazy_experts.core import dynamic_cache_update, get_fallback_stats
from mlx_moe.lazy_experts.generate import _startup

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"
PROFILE_PATH = "profiles/qwen3.5-35b-a3b-4bit.json"
CAPACITY = 248
TOKENS = 500
PROMPT = (
    "Implement a production-ready Python rate limiter middleware for FastAPI with Redis backing, "
    "sliding-window counters, and structured error responses. Include unit tests."
)
CONFIGS = [0, 32, 64]
OUTPUT_PATH = Path("benchmarks/results/pinning_35b.json")


def apply_chat_template(tokenizer, text: str) -> str:
    if not getattr(tokenizer, "has_chat_template", False):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def close_model(model) -> None:
    st_map = getattr(model, "_st_map", None)
    if st_map is not None:
        st_map.close()


def reset_fallback_counters(model) -> None:
    for layer in model.layers:
        switch = getattr(layer, "mlp", None)
        if switch is None or not hasattr(switch, "switch_mlp"):
            continue
        proj = getattr(switch.switch_mlp, "up_proj", None)
        cache = getattr(proj, "_cache", None)
        if cache is None:
            continue
        cache.total_requests = 0
        cache.total_fallbacks = 0
        cache._indices_buffer.clear()


def repetition_score(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def run_single(pin_top_k: int) -> dict:
    model, tokenizer, _ = _startup(
        MODEL_NAME,
        PROMPT,
        profile_path=PROFILE_PATH,
        capacity=CAPACITY,
        warmup="hybrid",
        pin_top_k=pin_top_k,
    )
    prompt = apply_chat_template(tokenizer, PROMPT)
    reset_fallback_counters(model)

    tokens = []
    pieces = []
    final = None
    for response in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=TOKENS):
        tokens.append(int(response.token))
        pieces.append(response.text)
        dynamic_cache_update(model, max_layer_updates=48)
        final = response

    for _ in range(3):
        dynamic_cache_update(model, max_layer_updates=48)

    fallback = get_fallback_stats(model)
    row = {
        "status": "ok",
        "pin_top_k": pin_top_k,
        "generated_tokens": len(tokens),
        "decode_tps": float(final.generation_tps if final else 0.0),
        "fallback_rate": float(fallback["fallback_rate"]),
        "repetition_score_300": repetition_score(tokens[:300]),
        "repetition_score_500": repetition_score(tokens[:500]),
        "output_preview": "".join(pieces)[:220],
    }
    close_model(model)
    return row


def run_parent() -> dict:
    rows = []
    base_cmd = [sys.executable, str(Path(__file__).resolve()), "--child"]

    for pin_top_k in CONFIGS:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        cmd = base_cmd + ["--pin-top-k", str(pin_top_k), "--output", str(tmp_path)]
        proc = subprocess.run(cmd, text=True, capture_output=True)

        if proc.returncode == 0 and tmp_path.exists():
            rows.append(json.loads(tmp_path.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "status": "failed",
                    "pin_top_k": pin_top_k,
                    "error": f"child_exit_{proc.returncode}",
                }
            )

        tmp_path.unlink(missing_ok=True)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if len(ok_rows) >= 2:
        decode_vals = [row["decode_tps"] for row in ok_rows]
        fallback_vals = [row["fallback_rate"] for row in ok_rows]
        rep_vals = [row["repetition_score_500"] for row in ok_rows]
        spreads = {
            "decode_tps_relative_spread": (max(decode_vals) - min(decode_vals)) / max(decode_vals),
            "fallback_rate_abs_spread": max(fallback_vals) - min(fallback_vals),
            "repetition_500_abs_spread": max(rep_vals) - min(rep_vals),
        }
    else:
        spreads = {
            "decode_tps_relative_spread": None,
            "fallback_rate_abs_spread": None,
            "repetition_500_abs_spread": None,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "capacity": CAPACITY,
        "prompt": PROMPT,
        "tokens": TOKENS,
        "results": rows,
        "spreads": spreads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--pin-top-k", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.child:
        if not args.output:
            raise ValueError("--output is required in --child mode")
        row = run_single(args.pin_top_k)
        Path(args.output).write_text(json.dumps(row), encoding="utf-8")
        return

    result = run_parent()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
