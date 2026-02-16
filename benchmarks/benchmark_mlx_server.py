#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from tool_chat_scenarios import OPENAI_AGENT_TOOLS

REPO_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_LINE = (
    "repo_map: mlx_moe/server.py handles OpenAI+Anthropic adapters, keyed KV cache LRU, "
    "dynamic_cache_update policy, telemetry printouts, and tool-call salvage. tests/test_integration.py "
    "validates cache keys, KV eviction, and fallback policy. benchmarks/profile_experts.py generates "
    "universal expert activation profiles."
)

COLD_PROMPTS = [
    "Audit mlx_moe/server.py for tokenization duplication and propose a minimal patch.",
    "Write targeted tests for KV cache reuse edge cases in a 3-turn coding chat.",
    "Explain why fallback_rate spikes on first request and how profile warmup mitigates it.",
    "Propose a low-risk refactor plan for dynamic cache update scheduling.",
    "Design a benchmark checklist for latency, throughput, and output quality.",
]

REUSE_TURN_1 = (
    "Audit the server request path for tokenization duplication and propose a minimal patch plus tests."
)
REUSE_TURN_2 = (
    "Now focus on fallback telemetry: explain why first-turn fallback can spike and how to reduce it "
    "with profile-driven startup."
)
REUSE_TURN_3 = (
    "Given the previous plan, produce a concise implementation checklist and a risk-oriented test matrix."
)

REUSE_ASSISTANT_STUB_1 = (
    "I will audit the request path for tokenization duplication and propose a minimal patch with tests."
)
REUSE_ASSISTANT_STUB_2 = (
    "Fallback can spike on the first turn because profile coverage is incomplete before dynamic cache converges."
)

MODES = ("cold", "reuse")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    w = rank - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def _slugify_model(model: str) -> str:
    return model.split("/")[-1].lower()


def _slugify_profile(profile_path: str | None) -> str:
    if profile_path is None:
        return "auto"
    return Path(profile_path).stem.lower()


def _build_system_prompt(context_repeat: int, tools_mode: str) -> str:
    repeated = "\n".join([SYSTEM_LINE] * context_repeat)
    if tools_mode != "embedded":
        return repeated
    tools_json = json.dumps(OPENAI_AGENT_TOOLS, ensure_ascii=False, separators=(",", ":"))
    return f"{repeated}\n\nTOOLS_JSON:\n{tools_json}"


def _build_reuse_messages(system_prompt: str, turn: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": REUSE_TURN_1},
    ]
    if turn >= 2:
        messages.extend(
            [
                {"role": "assistant", "content": REUSE_ASSISTANT_STUB_1},
                {"role": "user", "content": REUSE_TURN_2},
            ]
        )
    if turn >= 3:
        messages.extend(
            [
                {"role": "assistant", "content": REUSE_ASSISTANT_STUB_2},
                {"role": "user", "content": REUSE_TURN_3},
            ]
        )
    return messages


def _build_cold_messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _render_prompt_preview(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            lines.append(f"- system: {len(content)} chars")
            continue
        clean = " ".join(str(content).split())
        lines.append(f"- {role}: {clean}")
    return "\n".join(lines)


def _extract_visible_token_event(obj: dict[str, Any]) -> bool:
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            return True
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def _extract_delta_content(obj: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    tool_chunks: list[dict[str, Any]] = []
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return "", []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tool_chunks.append(tc)
    return "".join(text_parts), tool_chunks


def _stream_chat(
    *,
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    print_output: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    first_token_t: float | None = None
    usage: dict[str, Any] | None = None
    full_text: list[str] = []
    tool_chunks: list[dict[str, Any]] = []
    saw_tool_chunks = False

    with client.stream("POST", url, json=payload, timeout=None) as resp:
        if resp.status_code != 200:
            body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status_code}: {body[:1200]}")

        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if first_token_t is None and _extract_visible_token_event(obj):
                first_token_t = time.perf_counter()
            text_piece, tc = _extract_delta_content(obj)
            if text_piece:
                full_text.append(text_piece)
                if print_output:
                    print(text_piece, end="", flush=True)
            if tc:
                tool_chunks.extend(tc)
                if print_output and not saw_tool_chunks:
                    saw_tool_chunks = True
                    print("\n[tool-calls stream]", flush=True)
                if print_output:
                    print(json.dumps(tc, ensure_ascii=False), flush=True)
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]

    if print_output:
        print()

    t1 = time.perf_counter()
    total_latency_ms = (t1 - t0) * 1000.0
    ttft_ms = (first_token_t - t0) * 1000.0 if first_token_t is not None else None

    return {
        "ttft_ms_client": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "usage": usage,
        "output_text": "".join(full_text),
        "tool_chunks": tool_chunks,
    }


def _parse_mlx_segment(segment: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    openai_matches = list(
        re.finditer(r"\[openai\].*?input_tokens=(\d+).*?cache_key=([^\s]+)", segment)
    )
    if openai_matches:
        m = openai_matches[-1]
        out["prompt_tokens"] = int(m.group(1))
        out["cache_key"] = m.group(2)

    tel_matches = list(
        re.finditer(
            r"\[telemetry\]\s+cache_key=([^\s]+)\s+"
            r"prefill=([0-9.]+)ms\s+ttft=([0-9.]+)ms\s+"
            r"decode=([0-9.]+)\s+tok/s\s+tokens=(\d+)\s+"
            r"dcu_calls=(\d+)\s+swaps=(\d+)\s+fallback_rate=([0-9.]+)%",
            segment,
        )
    )
    if tel_matches:
        m = tel_matches[-1]
        out.update(
            {
                "cache_key_telemetry": m.group(1),
                "prefill_ms": float(m.group(2)),
                "ttft_ms_server": float(m.group(3)),
                "decode_tps_server": float(m.group(4)),
                "completion_tokens": int(m.group(5)),
                "dcu_calls": int(m.group(6)),
                "swaps": int(m.group(7)),
                "fallback_rate_pct": float(m.group(8)),
            }
        )

    reuse_matches = list(
        re.finditer(r"\[kv cache: reusing\s+(\d+),\s+processing\s+(\d+)\s+new tokens\]", segment)
    )
    if reuse_matches:
        m = reuse_matches[-1]
        out["kv_reuse_tokens"] = int(m.group(1))
        out["kv_new_tokens"] = int(m.group(2))

    return out


@dataclass
class ServerSpec:
    cmd: list[str]
    cwd: Path
    ready_url: str
    log_path: Path


class ServerProcess:
    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.proc: subprocess.Popen[str] | None = None
        self.log_fh: Any | None = None

    def start(self, timeout_s: int = 7200) -> None:
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_fh = open(self.spec.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.spec.cmd,
            cwd=str(self.spec.cwd),
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"[{_now()}] Starting mlx-moe server: {' '.join(self.spec.cmd)}")
        deadline = time.time() + timeout_s
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"mlx-moe server exited with code {self.proc.returncode}. See {self.spec.log_path}"
                    )
                try:
                    r = client.get(self.spec.ready_url)
                    if r.status_code == 200:
                        print(f"[{_now()}] Server ready: {self.spec.ready_url}")
                        return
                except Exception:
                    pass
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for server readiness: {self.spec.ready_url}")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        print(f"[{_now()}] Stopped mlx-moe server (code={self.proc.returncode})")
        if self.log_fh is not None:
            self.log_fh.close()
        self.proc = None
        self.log_fh = None

    def log_size(self) -> int:
        if not self.spec.log_path.exists():
            return 0
        return self.spec.log_path.stat().st_size

    def read_from_offset(self, offset: int) -> str:
        if not self.spec.log_path.exists():
            return ""
        with open(self.spec.log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return f.read()


def _build_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        "uv",
        "run",
        "mlx-moe",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--warmup",
        args.warmup,
    ]
    if args.capacity is not None:
        cmd.extend(["--capacity", str(args.capacity)])
    if args.profile is not None:
        cmd.extend(["--profile", args.profile])
    if args.pin_top_k is not None:
        cmd.extend(["--pin-top-k", str(args.pin_top_k)])
    if args.max_output_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_output_tokens)])
    if args.max_input_tokens is not None:
        cmd.extend(["--max-input-tokens", str(args.max_input_tokens)])
    if args.kv_bits is not None:
        cmd.extend(["--kv-bits", str(args.kv_bits)])
    return cmd


def _build_payload(
    *,
    mode: str,
    trial: int,
    turn: int,
    model_id: str,
    system_prompt: str,
    tools_mode: str,
    max_tokens: int,
    cache_key: str,
    slot_id: int,
) -> dict[str, Any]:
    if mode == "cold":
        user_prompt = COLD_PROMPTS[(trial - 1) % len(COLD_PROMPTS)]
        messages = _build_cold_messages(system_prompt, user_prompt)
    else:
        messages = _build_reuse_messages(system_prompt, turn)

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": cache_key,
        "id_slot": slot_id,
        "cache_prompt": (mode == "reuse"),
    }
    if tools_mode == "field":
        payload["tools"] = OPENAI_AGENT_TOOLS
    return payload


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode and r.get("error") is None]
        ttft = [r["ttft_ms_client"] for r in subset if isinstance(r.get("ttft_ms_client"), (int, float))]
        dec_client = [r["decode_tps_client"] for r in subset if isinstance(r.get("decode_tps_client"), (int, float))]
        dec_server = [r["decode_tps_server"] for r in subset if isinstance(r.get("decode_tps_server"), (int, float))]
        latency = [r["total_latency_ms"] for r in subset if isinstance(r.get("total_latency_ms"), (int, float))]
        fallback = [r["fallback_rate_pct"] for r in subset if isinstance(r.get("fallback_rate_pct"), (int, float))]
        out[mode] = {
            "n_requests": len(subset),
            "ttft_ms": {
                "mean": statistics.fmean(ttft) if ttft else None,
                "p50": _percentile(ttft, 50),
                "p95": _percentile(ttft, 95),
            },
            "decode_tps_client": {
                "mean": statistics.fmean(dec_client) if dec_client else None,
                "p50": _percentile(dec_client, 50),
                "p95": _percentile(dec_client, 95),
            },
            "decode_tps_server": {
                "mean": statistics.fmean(dec_server) if dec_server else None,
                "p50": _percentile(dec_server, 50),
                "p95": _percentile(dec_server, 95),
            },
            "total_latency_ms": {
                "mean": statistics.fmean(latency) if latency else None,
                "p50": _percentile(latency, 50),
                "p95": _percentile(latency, 95),
            },
            "fallback_rate_pct": {
                "mean": statistics.fmean(fallback) if fallback else None,
                "p50": _percentile(fallback, 50),
                "p95": _percentile(fallback, 95),
            },
        }
    return out


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Mode | N req | TTFT ms (mean/p50/p95) | Decode tok/s client (mean/p50/p95) | Decode tok/s server (mean/p50/p95) | Total latency ms (mean/p50/p95) | Fallback % (mean/p50/p95) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        s = summary[mode]
        lines.append(
            "| "
            f"{mode} | {s['n_requests']} | "
            f"{_fmt(s['ttft_ms']['mean'],1)}/{_fmt(s['ttft_ms']['p50'],1)}/{_fmt(s['ttft_ms']['p95'],1)} | "
            f"{_fmt(s['decode_tps_client']['mean'],2)}/{_fmt(s['decode_tps_client']['p50'],2)}/{_fmt(s['decode_tps_client']['p95'],2)} | "
            f"{_fmt(s['decode_tps_server']['mean'],2)}/{_fmt(s['decode_tps_server']['p50'],2)}/{_fmt(s['decode_tps_server']['p95'],2)} | "
            f"{_fmt(s['total_latency_ms']['mean'],1)}/{_fmt(s['total_latency_ms']['p50'],1)}/{_fmt(s['total_latency_ms']['p95'],1)} | "
            f"{_fmt(s['fallback_rate_pct']['mean'],2)}/{_fmt(s['fallback_rate_pct']['p50'],2)}/{_fmt(s['fallback_rate_pct']['p95'],2)} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark mlx-moe server with streamed output and telemetry capture")
    p.add_argument("--model", default="mlx-community/Qwen3-Coder-Next-4bit")
    p.add_argument("--profile", default=None)
    p.add_argument("--capacity", type=int, default=208)
    p.add_argument("--pin-top-k", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--warmup", choices=["none", "hybrid", "full"], default="hybrid")
    p.add_argument("--kv-bits", type=int, default=None)
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--max-input-tokens", type=int, default=None)
    p.add_argument("--request-max-tokens", type=int, default=120)
    p.add_argument("--cold-trials", type=int, default=5)
    p.add_argument("--reuse-trials", type=int, default=5)
    p.add_argument("--context-repeat", type=int, default=200)
    p.add_argument(
        "--tools-mode",
        choices=["field", "embedded", "none"],
        default="field",
        help="field=OpenAI tools field, embedded=tools JSON in system prompt, none=no tools",
    )
    p.add_argument("--print-output", action="store_true", default=True)
    p.add_argument("--no-print-output", action="store_false", dest="print_output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = _slugify_model(args.model)
    profile_slug = _slugify_profile(args.profile)
    run_dir = REPO_ROOT / "logs" / "model" / model_slug / profile_slug / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    out_json = run_dir / "benchmark.json"
    out_md = run_dir / "benchmark.md"
    server_log = run_dir / "server.log"

    server_cmd = _build_server_cmd(args)
    server = ServerProcess(
        ServerSpec(
            cmd=server_cmd,
            cwd=REPO_ROOT,
            ready_url=f"http://{args.host}:{args.port}/v1/models",
            log_path=server_log,
        )
    )

    system_prompt = _build_system_prompt(args.context_repeat, args.tools_mode)
    model_id = args.model.split("/")[-1]
    rows: list[dict[str, Any]] = []

    try:
        server.start()
        time.sleep(1)

        with httpx.Client(timeout=None) as client:
            for mode in MODES:
                trials = args.cold_trials if mode == "cold" else args.reuse_trials
                for trial in range(1, trials + 1):
                    turns = [1] if mode == "cold" else [1, 2, 3]
                    slot_id = (100 + trial) if mode == "cold" else (trial - 1)
                    cache_key = f"{mode}-trial-{trial}"

                    for turn in turns:
                        payload = _build_payload(
                            mode=mode,
                            trial=trial,
                            turn=turn,
                            model_id=model_id,
                            system_prompt=system_prompt,
                            tools_mode=args.tools_mode,
                            max_tokens=args.request_max_tokens,
                            cache_key=cache_key,
                            slot_id=slot_id,
                        )

                        label = f"mode={mode} trial={trial} turn={turn}"
                        print()
                        print("=" * 88)
                        print(f"[{_now()}] Request start: {label}")
                        print(_render_prompt_preview(payload["messages"]))
                        print("Model output:")

                        offset = server.log_size()
                        error_text: str | None = None
                        stream: dict[str, Any] | None = None
                        try:
                            stream = _stream_chat(
                                client=client,
                                url=f"http://{args.host}:{args.port}/v1/chat/completions",
                                payload=payload,
                                print_output=args.print_output,
                            )
                        except Exception as exc:
                            error_text = str(exc)
                            print(f"\n[{_now()}] Request error: {error_text}")

                        time.sleep(0.25)
                        segment = server.read_from_offset(offset)
                        mlx = _parse_mlx_segment(segment)

                        row: dict[str, Any] = {
                            "mode": mode,
                            "trial": trial,
                            "turn": turn,
                            "cache_key": cache_key,
                            "slot_id": slot_id,
                            "cache_prompt": (mode == "reuse"),
                            "tools_mode": args.tools_mode,
                            "messages": payload["messages"],
                            "error": error_text,
                            "mlx": mlx,
                        }

                        if stream is not None:
                            row.update(stream)
                            usage = stream.get("usage")
                            if isinstance(usage, dict):
                                if isinstance(usage.get("prompt_tokens"), int):
                                    row["prompt_tokens_usage"] = usage["prompt_tokens"]
                                if isinstance(usage.get("completion_tokens"), int):
                                    row["completion_tokens_usage"] = usage["completion_tokens"]

                        if isinstance(mlx.get("prompt_tokens"), int):
                            row["prompt_tokens_server"] = mlx["prompt_tokens"]
                        if isinstance(mlx.get("completion_tokens"), int):
                            row["completion_tokens_server"] = mlx["completion_tokens"]
                            row["completion_tokens"] = mlx["completion_tokens"]
                        elif isinstance(row.get("completion_tokens_usage"), int):
                            row["completion_tokens"] = row["completion_tokens_usage"]

                        ttft = row.get("ttft_ms_client")
                        total = row.get("total_latency_ms")
                        comp = row.get("completion_tokens")
                        if (
                            isinstance(ttft, (int, float))
                            and isinstance(total, (int, float))
                            and isinstance(comp, int)
                            and total > ttft
                        ):
                            decode_s = (total - ttft) / 1000.0
                            row["decode_tps_client"] = comp / decode_s if decode_s > 0 else None
                        else:
                            row["decode_tps_client"] = None

                        row["decode_tps_server"] = mlx.get("decode_tps_server")
                        row["fallback_rate_pct"] = mlx.get("fallback_rate_pct")

                        rows.append(row)

                        print(
                            f"[{_now()}] Request done: {label} "
                            f"status={'ok' if error_text is None else 'error'} "
                            f"prompt={row.get('prompt_tokens_server', row.get('prompt_tokens_usage'))} "
                            f"comp={row.get('completion_tokens')} "
                            f"ttft={_fmt(row.get('ttft_ms_client'), 1)}ms "
                            f"decode_client={_fmt(row.get('decode_tps_client'), 2)} tok/s "
                            f"decode_server={_fmt(row.get('decode_tps_server'), 2)} tok/s "
                            f"fallback={_fmt(row.get('fallback_rate_pct'), 2)}%"
                        )
                        time.sleep(0.5)
    finally:
        server.stop()

    summary = _summarize(rows)
    report = {
        "timestamp": ts,
        "model": args.model,
        "profile": args.profile,
        "profile_slug": profile_slug,
        "settings": {
            "capacity": args.capacity,
            "pin_top_k": args.pin_top_k,
            "warmup": args.warmup,
            "kv_bits": args.kv_bits,
            "request_max_tokens": args.request_max_tokens,
            "cold_trials": args.cold_trials,
            "reuse_trials": args.reuse_trials,
            "context_repeat": args.context_repeat,
            "tools_mode": args.tools_mode,
            "sampling": {"temperature": 0.2, "top_p": 0.95, "top_k": 40},
        },
        "paths": {
            "run_dir": str(run_dir),
            "server_log": str(server_log),
            "json": str(out_json),
            "markdown": str(out_md),
        },
        "rows": rows,
        "summary": summary,
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = _render_markdown(summary)
    out_md.write_text(md + "\n", encoding="utf-8")

    print()
    print("=" * 88)
    print("SUMMARY")
    print(md)
    print()
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote Markdown: {out_md}")
    print(f"Server log: {server_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
