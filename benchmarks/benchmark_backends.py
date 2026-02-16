#!/usr/bin/env python3
from __future__ import annotations
"""Repeatable backend benchmark for Qwen3-Coder-Next on a single Mac.

Methodology implemented by this script:

1. Runs two OpenAI-compatible servers sequentially (never in parallel):
   - llama.cpp (`/v1/chat/completions`)
   - mlx-moe (`/v1/chat/completions`)

2. Uses one identical request shape for both servers:
   - `temperature=0.2`, `top_p=0.95`, `top_k=40`, `max_tokens=100`
   - `stream=true`, `stream_options.include_usage=true`
   - tool schema imported from `benchmarks/tool_chat_scenarios.py`
   - long system prompt: fixed repo-map line repeated 200 times
   - by default, tool JSON is embedded in system text for both servers and
     `tools` is omitted for tokenizer fairness (`--use-tools-field` toggles this)

3. Runs two request modes:
   - `cold`: 5 independent requests (single turn each, no prefix reuse intent)
   - `reuse`: 5 trials of a 3-turn conversation sharing the same session key/slot
     so prefix KV can be reused across turns

4. Measures per request:
   - `prompt_tokens`, `completion_tokens` (from usage/telemetry)
   - client TTFT (time to first streamed visible token/tool event)
   - client decode tok/s (`completion_tokens / (total - TTFT)`)
   - total latency
   - process peak memory via `footprint --sample ... --noCategories`
   - mlx-specific telemetry from logs: `prefill`, `ttft`, `decode`,
     `dcu_calls`, `swaps`, `fallback_rate`

5. Computes summary stats:
   - mean / p50 / p95 for TTFT, decode tok/s, latency, peak memory
   - prompt-token fairness check between backends for matched requests
   - reuse-minus-cold deltas

Outputs:
- `logs/backends/backend_compare_<timestamp>.json`
- `logs/backends/backend_compare_<timestamp>.md`
- backend server logs alongside the result files

Run examples:
- Default:
  `uv run python benchmarks/benchmark_backends.py`
- Use an already-downloaded GGUF path for llama.cpp:
  `uv run python benchmarks/benchmark_backends.py --llama-cmd '/path/to/llama-server -m /path/to/model.gguf --host 127.0.0.1 --port {port}'`

Notes:
- This script is intentionally long-running (model startup + 40 total requests).
- If llama.cpp OOMs on first prefill, results are still emitted with failed rows,
  and mlx-moe continues so partial data is preserved.
"""

import argparse
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "logs" / "backends"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_LINE = (
    "repo_map: mlx_moe/server.py handles OpenAI+Anthropic adapters, keyed KV cache LRU, "
    "dynamic_cache_update policy, telemetry printouts, and tool-call salvage. tests/test_integration.py "
    "validates cache keys, KV eviction, and fallback policy. benchmarks/profile_experts.py generates "
    "universal expert activation profiles."
)

TURN_1 = (
    "Audit the server request path for tokenization duplication and propose a minimal patch plus tests."
)
TURN_2 = (
    "Now focus on fallback telemetry: explain why first-turn fallback can spike and how to reduce it "
    "with profile-driven startup."
)
TURN_3 = (
    "Given the previous plan, produce a concise implementation checklist and a risk-oriented test matrix."
)

ASSISTANT_STUB_1 = (
    "I will audit the request path for tokenization duplication and propose a minimal patch with tests."
)
ASSISTANT_STUB_2 = (
    "Fallback can spike on the first turn because profile coverage is incomplete before dynamic cache converges."
)

OPENAI_URL = "/v1/chat/completions"
MODES = ("cold", "reuse")
TRIALS = 5


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


def _bytes_to_gib(b: int | None) -> float | None:
    if b is None:
        return None
    return b / (1024 ** 3)


def _read_tools() -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.tool_chat_scenarios import OPENAI_AGENT_TOOLS

    return OPENAI_AGENT_TOOLS


def build_system_prompt(tools: list[dict[str, Any]], use_tools_field: bool) -> str:
    repeated = "\n".join([SYSTEM_LINE] * 200)
    if use_tools_field:
        return repeated
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return f"{repeated}\n\nTOOLS_JSON:\n{tools_json}"


def build_messages(system_prompt: str, turn: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": TURN_1},
    ]
    if turn >= 2:
        messages.extend(
            [
                {"role": "assistant", "content": ASSISTANT_STUB_1},
                {"role": "user", "content": TURN_2},
            ]
        )
    if turn >= 3:
        messages.extend(
            [
                {"role": "assistant", "content": ASSISTANT_STUB_2},
                {"role": "user", "content": TURN_3},
            ]
        )
    return messages


def build_payload(
    *,
    turn: int,
    tools: list[dict[str, Any]],
    system_prompt: str,
    use_tools_field: bool,
    session_key: str,
    slot_id: int,
    cache_prompt: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "qwen3-coder-next-4bit",
        "messages": build_messages(system_prompt, turn),
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 100,
        "stream": True,
        "stream_options": {"include_usage": True},
        "user": session_key,
        "id_slot": slot_id,
        "cache_prompt": cache_prompt,
    }
    if use_tools_field:
        payload["tools"] = tools
    return payload


@dataclass
class ServerSpec:
    name: str
    cmd: list[str]
    cwd: Path
    base_url: str
    ready_url: str
    log_path: Path


class ServerProcess:
    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        self.proc: subprocess.Popen[str] | None = None
        self.log_fh: Any | None = None
        self._monitored_pid: int | None = None

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
        print(f"[{_now()}] Starting {self.spec.name}: {' '.join(self.spec.cmd)}")
        deadline = time.time() + timeout_s
        with httpx.Client(timeout=5.0) as client:
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.spec.name} exited during startup with code {self.proc.returncode}. "
                        f"See {self.spec.log_path}"
                    )
                try:
                    r = client.get(self.spec.ready_url)
                    if r.status_code == 200:
                        self._monitored_pid = self._resolve_monitored_pid()
                        print(f"[{_now()}] {self.spec.name} ready: {self.spec.ready_url}")
                        print(
                            f"[{_now()}] {self.spec.name} monitored pid for footprint: "
                            f"{self._monitored_pid}"
                        )
                        return
                except Exception:
                    pass
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for {self.spec.name} readiness: {self.spec.ready_url}")

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
        print(f"[{_now()}] Stopped {self.spec.name} (code={self.proc.returncode})")
        if self.log_fh is not None:
            self.log_fh.close()
        self.proc = None
        self.log_fh = None
        self._monitored_pid = None

    @property
    def pid(self) -> int:
        if self.proc is None:
            raise RuntimeError("Server not running")
        return self.proc.pid

    @property
    def monitored_pid(self) -> int:
        if self._monitored_pid is None:
            self._monitored_pid = self._resolve_monitored_pid()
        return self._monitored_pid

    @staticmethod
    def _descendant_pids(root_pid: int) -> list[int]:
        try:
            out = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
        except Exception:
            return []

        children: dict[int, list[int]] = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            children.setdefault(ppid, []).append(pid)

        descendants: list[int] = []
        stack = [root_pid]
        while stack:
            cur = stack.pop()
            for child in children.get(cur, []):
                descendants.append(child)
                stack.append(child)
        return descendants

    def _resolve_monitored_pid(self) -> int:
        root_pid = self.pid
        candidates = [root_pid, *self._descendant_pids(root_pid)]
        uniq_candidates = sorted(set(candidates))
        if not uniq_candidates:
            return root_pid

        try:
            out = subprocess.check_output(
                [
                    "ps",
                    "-o",
                    "pid=",
                    "-o",
                    "rss=",
                    "-o",
                    "command=",
                    "-p",
                    ",".join(str(pid) for pid in uniq_candidates),
                ],
                text=True,
            )
        except Exception:
            return root_pid

        best_pid = root_pid
        best_rss = -1
        for line in out.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                rss = int(parts[1])
            except ValueError:
                continue
            if rss > best_rss:
                best_rss = rss
                best_pid = pid

        return best_pid

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


class FootprintSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [
                "footprint",
                "--pid",
                str(self.pid),
                "--sample",
                "0.5",
                "--sample-duration",
                "7200",
                "--format",
                "bytes",
                "--noCategories",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop_and_peak_bytes(self) -> tuple[int | None, str]:
        if self.proc is None:
            return None, ""
        out = ""
        try:
            self.proc.send_signal(signal.SIGINT)
            out, _ = self.proc.communicate(timeout=30)
        except Exception:
            self.proc.kill()
            out, _ = self.proc.communicate(timeout=10)

        peaks: list[int] = []
        for pat in (
            r"phys_footprint:\s+([\d,]+)\s+B",
            r"phys_footprint_peak:\s+([\d,]+)\s+B",
            r"Footprint:\s+([\d,]+)\s+B",
        ):
            for m in re.finditer(pat, out):
                peaks.append(int(m.group(1).replace(",", "")))

        return (max(peaks) if peaks else None), out


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


def stream_chat(
    *,
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    first_token_t: float | None = None
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None

    with client.stream("POST", url, json=payload, timeout=None) as resp:
        status = resp.status_code
        if status != 200:
            body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status} from {url}: {body[:1200]}")

        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            if first_token_t is None and _extract_visible_token_event(obj):
                first_token_t = now
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]

    t1 = time.perf_counter()
    total_latency_ms = (t1 - t0) * 1000.0
    ttft_ms = (first_token_t - t0) * 1000.0 if first_token_t is not None else None

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "usage": usage,
        "timings": timings,
    }


def parse_mlx_segment(segment: str) -> dict[str, Any]:
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


def erase_llama_slot(client: httpx.Client, base_url: str, slot_id: int) -> None:
    url = f"{base_url}/slots/{slot_id}?action=erase"
    try:
        r = client.post(url, timeout=30)
        if r.status_code not in (200, 404):
            print(f"[{_now()}] WARN slot erase {slot_id}: HTTP {r.status_code} body={r.text[:200]}")
    except Exception as exc:
        print(f"[{_now()}] WARN slot erase {slot_id} failed: {exc}")


def run_backend(
    *,
    backend: str,
    server: ServerProcess,
    tools: list[dict[str, Any]],
    use_tools_field: bool,
) -> list[dict[str, Any]]:
    system_prompt = build_system_prompt(tools, use_tools_field)
    all_rows: list[dict[str, Any]] = []

    server.start()
    time.sleep(1)

    with httpx.Client(timeout=None) as client:
        for mode in MODES:
            for trial in range(1, TRIALS + 1):
                if mode == "reuse":
                    turns = [1, 2, 3]
                    slot_id = trial - 1
                    cache_prompt = True
                else:
                    turns = [1]
                    slot_id = 100 + trial
                    cache_prompt = False

                if backend == "llama":
                    erase_llama_slot(client, server.spec.base_url, slot_id)

                session_key = f"{mode}-trial-{trial}"

                for turn in turns:
                    payload = build_payload(
                        turn=turn,
                        tools=tools,
                        system_prompt=system_prompt,
                        use_tools_field=use_tools_field,
                        session_key=session_key,
                        slot_id=slot_id,
                        cache_prompt=cache_prompt,
                    )

                    req_label = f"{backend} mode={mode} trial={trial} turn={turn}"
                    print(f"[{_now()}] Request start: {req_label}")

                    log_offset = server.log_size()
                    monitor_pid = server.monitored_pid
                    fps = FootprintSampler(monitor_pid)
                    fps.start()

                    error_text: str | None = None
                    stream: dict[str, Any] | None = None
                    try:
                        stream = stream_chat(
                            client=client,
                            url=f"{server.spec.base_url}{OPENAI_URL}",
                            payload=payload,
                        )
                    except Exception as exc:
                        error_text = str(exc)

                    peak_b, fp_raw = fps.stop_and_peak_bytes()

                    # Give logger a brief window to flush telemetry lines.
                    time.sleep(0.25)
                    segment = server.read_from_offset(log_offset)

                    row: dict[str, Any] = {
                        "backend": backend,
                        "mode": mode,
                        "trial": trial,
                        "turn": turn,
                        "session_key": session_key,
                        "slot_id": slot_id,
                        "cache_prompt": cache_prompt,
                        "use_tools_field": use_tools_field,
                        "monitored_pid": monitor_pid,
                        "peak_memory_bytes": peak_b,
                        "peak_memory_gib": _bytes_to_gib(peak_b),
                        "error": error_text,
                    }

                    if stream is not None:
                        row.update(
                            {
                                "ttft_ms_client": stream["ttft_ms"],
                                "total_latency_ms": stream["total_latency_ms"],
                                "usage": stream["usage"],
                                "timings": stream["timings"],
                            }
                        )

                    if backend == "mlx":
                        mlx = parse_mlx_segment(segment)
                        row.update({"mlx": mlx})
                        if "prompt_tokens" in mlx:
                            row["prompt_tokens"] = mlx["prompt_tokens"]
                        if "completion_tokens" in mlx:
                            row["completion_tokens"] = mlx["completion_tokens"]
                    else:
                        usage = row.get("usage")
                        if isinstance(usage, dict):
                            if isinstance(usage.get("prompt_tokens"), int):
                                row["prompt_tokens"] = usage["prompt_tokens"]
                            if isinstance(usage.get("completion_tokens"), int):
                                row["completion_tokens"] = usage["completion_tokens"]

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

                    all_rows.append(row)

                    status = "ok" if error_text is None else f"error={error_text[:120]}"
                    print(
                        f"[{_now()}] Request done: {req_label} status={status} "
                        f"prompt={row.get('prompt_tokens')} comp={row.get('completion_tokens')} "
                        f"ttft={_fmt(row.get('ttft_ms_client'), 1)}ms "
                        f"decode={_fmt(row.get('decode_tps_client'), 2)} tok/s "
                        f"lat={_fmt(row.get('total_latency_ms'), 1)}ms "
                        f"peak={_fmt(row.get('peak_memory_gib'), 2)} GiB"
                    )

                    time.sleep(0.5)

    server.stop()
    return all_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for backend in ("llama", "mlx"):
        out[backend] = {}
        for mode in MODES:
            subset = [r for r in rows if r["backend"] == backend and r["mode"] == mode and not r.get("error")]
            ttft_vals = [r["ttft_ms_client"] for r in subset if isinstance(r.get("ttft_ms_client"), (int, float))]
            dec_vals = [r["decode_tps_client"] for r in subset if isinstance(r.get("decode_tps_client"), (int, float))]
            lat_vals = [r["total_latency_ms"] for r in subset if isinstance(r.get("total_latency_ms"), (int, float))]
            mem_vals = [r["peak_memory_gib"] for r in subset if isinstance(r.get("peak_memory_gib"), (int, float))]

            out[backend][mode] = {
                "n_requests": len(subset),
                "ttft_ms": {
                    "mean": statistics.fmean(ttft_vals) if ttft_vals else None,
                    "p50": _percentile(ttft_vals, 50),
                    "p95": _percentile(ttft_vals, 95),
                },
                "decode_tps": {
                    "mean": statistics.fmean(dec_vals) if dec_vals else None,
                    "p50": _percentile(dec_vals, 50),
                    "p95": _percentile(dec_vals, 95),
                },
                "total_latency_ms": {
                    "mean": statistics.fmean(lat_vals) if lat_vals else None,
                    "p50": _percentile(lat_vals, 50),
                    "p95": _percentile(lat_vals, 95),
                },
                "peak_memory_gib": {
                    "mean": statistics.fmean(mem_vals) if mem_vals else None,
                    "p50": _percentile(mem_vals, 50),
                    "p95": _percentile(mem_vals, 95),
                },
            }

    # Fairness check: prompt token drift for matched (mode, trial, turn)
    ll = {
        (r["mode"], r["trial"], r["turn"]): r
        for r in rows
        if r["backend"] == "llama" and isinstance(r.get("prompt_tokens"), int)
    }
    mm = {
        (r["mode"], r["trial"], r["turn"]): r
        for r in rows
        if r["backend"] == "mlx" and isinstance(r.get("prompt_tokens"), int)
    }

    fairness: list[dict[str, Any]] = []
    for key, lr in ll.items():
        mr = mm.get(key)
        if mr is None:
            continue
        a = lr["prompt_tokens"]
        b = mr["prompt_tokens"]
        avg = (a + b) / 2.0
        diff_pct = (abs(a - b) / avg * 100.0) if avg > 0 else 0.0
        fairness.append(
            {
                "mode": key[0],
                "trial": key[1],
                "turn": key[2],
                "llama_prompt_tokens": a,
                "mlx_prompt_tokens": b,
                "diff_pct": diff_pct,
                "within_1pct": diff_pct <= 1.0,
            }
        )

    out["fairness"] = fairness
    if fairness:
        out["fairness_summary"] = {
            "n": len(fairness),
            "mean_diff_pct": statistics.fmean(f["diff_pct"] for f in fairness),
            "p95_diff_pct": _percentile([f["diff_pct"] for f in fairness], 95),
            "all_within_1pct": all(f["within_1pct"] for f in fairness),
        }
    else:
        out["fairness_summary"] = {
            "n": 0,
            "mean_diff_pct": None,
            "p95_diff_pct": None,
            "all_within_1pct": False,
        }

    # Delta cold -> reuse for each backend (means)
    deltas: dict[str, Any] = {}
    for backend in ("llama", "mlx"):
        c = out[backend]["cold"]
        r = out[backend]["reuse"]
        deltas[backend] = {
            "ttft_ms": None,
            "decode_tps": None,
            "total_latency_ms": None,
        }
        for field in ("ttft_ms", "decode_tps", "total_latency_ms"):
            c_mean = c[field]["mean"]
            r_mean = r[field]["mean"]
            if isinstance(c_mean, (int, float)) and isinstance(r_mean, (int, float)):
                deltas[backend][field] = r_mean - c_mean

    out["deltas_reuse_minus_cold"] = deltas
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("| Backend | Mode | N req | TTFT ms (mean/p50/p95) | Decode tok/s (mean/p50/p95) | Total latency ms (mean/p50/p95) | Peak memory GiB (mean/p50/p95) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for backend in ("llama", "mlx"):
        for mode in MODES:
            row = summary[backend][mode]
            lines.append(
                "| "
                f"{backend} | {mode} | {row['n_requests']} | "
                f"{_fmt(row['ttft_ms']['mean'],1)}/{_fmt(row['ttft_ms']['p50'],1)}/{_fmt(row['ttft_ms']['p95'],1)} | "
                f"{_fmt(row['decode_tps']['mean'],2)}/{_fmt(row['decode_tps']['p50'],2)}/{_fmt(row['decode_tps']['p95'],2)} | "
                f"{_fmt(row['total_latency_ms']['mean'],1)}/{_fmt(row['total_latency_ms']['p50'],1)}/{_fmt(row['total_latency_ms']['p95'],1)} | "
                f"{_fmt(row['peak_memory_gib']['mean'],2)}/{_fmt(row['peak_memory_gib']['p50'],2)}/{_fmt(row['peak_memory_gib']['p95'],2)} |"
            )

    lines.append("")
    lines.append("### Reuse - Cold deltas (mean)")
    lines.append("| Backend | TTFT ms | Decode tok/s | Total latency ms |")
    lines.append("|---|---:|---:|---:|")
    for backend in ("llama", "mlx"):
        d = summary["deltas_reuse_minus_cold"][backend]
        lines.append(
            "| "
            f"{backend} | {_fmt(d['ttft_ms'],1)} | {_fmt(d['decode_tps'],2)} | {_fmt(d['total_latency_ms'],1)} |"
        )

    fs = summary.get("fairness_summary", {})
    lines.append("")
    lines.append("### Prompt token fairness")
    lines.append(
        f"Matched requests: {fs.get('n', 0)}, "
        f"mean diff: {_fmt(fs.get('mean_diff_pct'), 3)}%, "
        f"p95 diff: {_fmt(fs.get('p95_diff_pct'), 3)}%, "
        f"all within 1%: {fs.get('all_within_1pct')}"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp vs mlx-moe sequentially")
    parser.add_argument("--llama-port", type=int, default=8081)
    parser.add_argument("--mlx-port", type=int, default=8080)
    parser.add_argument(
        "--llama-cmd",
        type=str,
        default="/Users/muhash/llama.cpp/build/bin/llama-server --hf-repo unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M --host 127.0.0.1 --port {port}",
        help="llama-server command template; {port} will be replaced",
    )
    parser.add_argument(
        "--mlx-cmd",
        type=str,
        default="uv run mlx-moe serve mlx-community/Qwen3-Coder-Next-4bit --host 127.0.0.1 --port {port}",
        help="mlx-moe serve command template; {port} will be replaced",
    )
    parser.add_argument(
        "--use-tools-field",
        action="store_true",
        help="send OpenAI tools in request payload (default uses embedded tool JSON for tokenizer fairness)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tools = _read_tools()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = RESULTS_DIR / f"backend_compare_{ts}.json"
    out_md = RESULTS_DIR / f"backend_compare_{ts}.md"

    use_tools_field = bool(args.use_tools_field)
    print(f"[{_now()}] tools mode: {'tools field' if use_tools_field else 'embedded tools JSON'}")

    llama_cmd = shlex.split(args.llama_cmd.format(port=args.llama_port))
    mlx_cmd = shlex.split(args.mlx_cmd.format(port=args.mlx_port))

    llama = ServerProcess(
        ServerSpec(
            name="llama.cpp",
            cmd=llama_cmd,
            cwd=Path("/Users/muhash/llama.cpp/build/bin"),
            base_url=f"http://127.0.0.1:{args.llama_port}",
            ready_url=f"http://127.0.0.1:{args.llama_port}/v1/models",
            log_path=RESULTS_DIR / f"llama_server_{ts}.log",
        )
    )

    mlx = ServerProcess(
        ServerSpec(
            name="mlx-moe",
            cmd=mlx_cmd,
            cwd=REPO_ROOT,
            base_url=f"http://127.0.0.1:{args.mlx_port}",
            ready_url=f"http://127.0.0.1:{args.mlx_port}/v1/models",
            log_path=RESULTS_DIR / f"mlx_server_{ts}.log",
        )
    )

    rows: list[dict[str, Any]] = []

    try:
        rows.extend(
            run_backend(
                backend="llama",
                server=llama,
                tools=tools,
                use_tools_field=use_tools_field,
            )
        )

        # Grace period for Metal/OS memory reclamation between backends.
        time.sleep(15)

        rows.extend(
            run_backend(
                backend="mlx",
                server=mlx,
                tools=tools,
                use_tools_field=use_tools_field,
            )
        )
    finally:
        llama.stop()
        mlx.stop()

    summary = summarize(rows)
    report = {
        "timestamp": ts,
        "settings": {
            "use_tools_field": use_tools_field,
            "trials_per_mode": TRIALS,
            "modes": list(MODES),
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_tokens": 100,
        },
        "rows": rows,
        "summary": summary,
    }

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_markdown(summary)
    out_md.write_text(md + "\n", encoding="utf-8")

    print(f"[{_now()}] Wrote JSON: {out_json}")
    print(f"[{_now()}] Wrote Markdown: {out_md}")
    print()
    print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
