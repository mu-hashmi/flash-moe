"""Sweep profile mixes and pin counts for Qwen3-Coder-Next-4bit.

Builds mixed coding/tool-chat profiles, evaluates pin_top_k candidates with
real server traffic, and selects the best configuration under a strict quality
gate.
"""

import argparse
import json
import random
import re
import signal
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib import error, request

from tool_chat_scenarios import OPENAI_AGENT_TOOLS, get_profile_scenarios


DEFAULT_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"
DEFAULT_PIN_COUNTS = [0, 16, 32, 48, 64]
DEFAULT_MIXES = [(70, 30), (50, 50), (30, 70)]
DEFAULT_REPEATS = 2
DEFAULT_NUM_PROFILE_PROMPTS = 24
DEFAULT_NUM_EVAL_REQUESTS = 12
DEFAULT_PORT_BASE = 8081
DEFAULT_MAX_TOKENS = 256
DEFAULT_REQUEST_TIMEOUT_S = 180
DEFAULT_STARTUP_TIMEOUT_S = 240
DEFAULT_OUTPUT_DIR = "logs"
DEFAULT_PROFILES_DIR = "profiles"

TELEMETRY_RE = re.compile(
    r"\[telemetry\].*prefill=([0-9.]+)ms ttft=([0-9.]+)ms "
    r"decode=([0-9.]+) tok/s tokens=(\d+) dcu_calls=(\d+) "
    r"swaps=(\d+) fallback_rate=([0-9.]+)%"
)

STRICT_GATE = {
    "tool_call_schema_success_min": 0.98,
    "loop_failure_rate_max": 0.01,
    "unrecoverable_drops_max": 0,
}

CODING_EVAL_PROMPTS = [
    "Use tools to inspect src/main.py and summarize exactly what it does.",
    "Run tests for cache-key handling and patch failures with a minimal diff.",
    "Find dynamic cache update policy in mlx_moe/server.py and propose a focused fix.",
    "Search for tool parser logic and fix any truncated argument edge case.",
    "Inspect benchmarks/profile_experts.py and add a mixed prompt preset.",
    "Open mlx_moe/lazy_experts/persistence.py and explain pinning behavior.",
    "Patch CLI to add one new flag and add tests for it.",
    "Read failing integration output and propose the smallest safe patch.",
]


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._handle = path.open("w", encoding="utf-8")

    def write_line(self, line: str) -> None:
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def _print_line(line: str, run_logger: RunLogger | None = None) -> None:
    print(line)
    if run_logger is not None:
        run_logger.write_line(line)


def _parse_mix_list(spec: str) -> list[tuple[int, int]]:
    mixes = []
    for item in spec.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid mix entry '{part}', expected C:T")
        coding_str, tool_str = part.split(":", 1)
        coding = int(coding_str)
        tool = int(tool_str)
        if coding < 0 or tool < 0:
            raise ValueError(f"Invalid mix entry '{part}', weights must be >= 0")
        if coding + tool == 0:
            raise ValueError(f"Invalid mix entry '{part}', total weight must be > 0")
        mixes.append((coding, tool))
    if not mixes:
        raise ValueError("No valid mixes provided")
    return mixes


def _parse_int_list(spec: str) -> list[int]:
    values = []
    for item in spec.split(","):
        part = item.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("No integer values provided")
    return values


def _model_cache_paths(model_name: str) -> list[Path]:
    safe_name = model_name.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "mlx-moe"
    base = cache_dir / f"{safe_name}.json"
    return [
        base,
        Path(str(base).replace(".json", ".weights.safetensors")),
        Path(str(base).replace(".json", ".weights.safetensors.meta.json")),
    ]


def _clear_model_cache(model_name: str) -> None:
    for path in _model_cache_paths(model_name):
        path.unlink(missing_ok=True)


def _sample_items(pool: list, n: int, rng: random.Random) -> list:
    if n <= 0:
        return []
    sampled = []
    while len(sampled) < n:
        batch = list(pool)
        rng.shuffle(batch)
        take = min(n - len(sampled), len(batch))
        sampled.extend(batch[:take])
    return [deepcopy(item) for item in sampled]


def _scenario_to_request(scenario: dict, max_tokens: int) -> dict:
    messages = []
    if scenario.get("system"):
        messages.append({"role": "system", "content": scenario["system"]})
    messages.extend(deepcopy(scenario["messages"]))
    return {
        "messages": messages,
        "tools": deepcopy(scenario.get("tools", OPENAI_AGENT_TOOLS)),
        "stream": True,
        "max_tokens": max_tokens,
    }


def _coding_prompt_to_request(prompt: str, max_tokens: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a coding assistant. Use tool calls when file access is needed."},
            {"role": "user", "content": prompt},
        ],
        "tools": deepcopy(OPENAI_AGENT_TOOLS),
        "stream": True,
        "max_tokens": max_tokens,
    }


def _build_eval_requests(
    coding_weight: int,
    tool_weight: int,
    n_requests: int,
    seed: int,
    max_tokens: int,
) -> list[dict]:
    total_weight = coding_weight + tool_weight
    coding_n = int(round(n_requests * coding_weight / total_weight))
    tool_n = n_requests - coding_n
    rng = random.Random(seed)

    tool_scenarios = [s for s in get_profile_scenarios() if s.get("api") == "openai"]
    coding_items = _sample_items(CODING_EVAL_PROMPTS, coding_n, rng)
    tool_items = _sample_items(tool_scenarios, tool_n, rng)

    requests_data = [_coding_prompt_to_request(p, max_tokens=max_tokens) for p in coding_items]
    requests_data.extend(_scenario_to_request(s, max_tokens=max_tokens) for s in tool_items)
    rng.shuffle(requests_data)
    return requests_data


def _read_lines(
    proc: subprocess.Popen,
    sink: list[str],
    run_logger: RunLogger | None = None,
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        sink.append(stripped)
        if run_logger is not None:
            run_logger.write_line(stripped)
        print(line, end="")


def _start_server(
    model_name: str,
    profile_path: Path,
    pin_top_k: int,
    capacity: int,
    port: int,
    warmup: str,
    startup_timeout_s: int,
    run_logger: RunLogger | None = None,
) -> tuple[subprocess.Popen, list[str], threading.Thread]:
    cmd = [
        sys.executable,
        "-c",
        "from mlx_moe.cli import main; main()",
        "serve",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--capacity",
        str(capacity),
        "--profile",
        str(profile_path),
        "--pin-top-k",
        str(pin_top_k),
        "--warmup",
        warmup,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    reader = threading.Thread(
        target=_read_lines,
        args=(proc, lines, run_logger),
        daemon=True,
    )
    reader.start()

    deadline = time.time() + startup_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        if any("  Ready." in line for line in lines):
            return proc, lines, reader
        time.sleep(0.1)

    proc.kill()
    raise TimeoutError("Server startup timed out before readiness")


def _stop_server(proc: subprocess.Popen, reader: threading.Thread) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    reader.join(timeout=2)


def _parse_stream_response(raw_bytes_iter) -> dict:
    text = ""
    finish_reason = None
    calls_by_index: dict[int, dict] = {}
    for raw in raw_bytes_iter:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        choice = chunk["choices"][0]
        finish = choice.get("finish_reason")
        if finish is not None:
            finish_reason = finish
        delta = choice.get("delta", {})
        text += delta.get("content", "")
        for tc in delta.get("tool_calls", []):
            idx = int(tc.get("index", 0))
            entry = calls_by_index.setdefault(idx, {"name": None, "arguments_parts": []})
            func = tc.get("function", {})
            if "name" in func:
                entry["name"] = func["name"]
            if "arguments" in func:
                entry["arguments_parts"].append(func["arguments"])

    tool_calls = []
    for idx in sorted(calls_by_index):
        entry = calls_by_index[idx]
        tool_calls.append({
            "name": entry["name"],
            "arguments": "".join(entry["arguments_parts"]),
        })

    return {"text": text, "tool_calls": tool_calls, "finish_reason": finish_reason}


def _is_type_match(value, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True


def _validate_object_schema(data: dict, schema: dict) -> bool:
    if schema.get("type") and schema.get("type") != "object":
        return False
    props = schema.get("properties", {})
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)

    for key in required:
        if key not in data:
            return False

    if additional is False:
        for key in data:
            if key not in props:
                return False

    for key, value in data.items():
        if key not in props:
            continue
        prop_schema = props[key]
        schema_type = prop_schema.get("type")
        if schema_type and not _is_type_match(value, schema_type):
            return False
        if schema_type == "integer":
            min_val = prop_schema.get("minimum")
            max_val = prop_schema.get("maximum")
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
    return True


def _validate_tool_call_schema(tool_call: dict, tools_by_name: dict) -> bool:
    name = tool_call.get("name")
    if not name or name not in tools_by_name:
        return False

    args_str = tool_call.get("arguments", "")
    try:
        parsed = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False

    schema = tools_by_name[name].get("function", {}).get("parameters", {})
    return _validate_object_schema(parsed, schema)


def _looks_looped(text: str) -> bool:
    if not text:
        return False
    markers = [
        "No changes to apply: oldString and newString are identical.",
        "I made a typo in the file name",
    ]
    for marker in markers:
        if text.count(marker) >= 3:
            return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    tail = lines[-1]
    if not tail:
        return False
    return sum(1 for ln in lines if ln == tail) >= 3


def _run_stream_request(
    port: int,
    body: dict,
    timeout_s: int,
) -> dict:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout_s) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        return _parse_stream_response(resp)


def _parse_telemetry(lines: list[str]) -> dict:
    decode = []
    fallback = []
    ttft = []
    unrecoverable = 0
    for line in lines:
        if "[tool call unrecoverable, dropping]" in line:
            unrecoverable += 1
        m = TELEMETRY_RE.search(line)
        if not m:
            continue
        ttft.append(float(m.group(2)))
        decode.append(float(m.group(3)))
        fallback.append(float(m.group(7)))
    return {
        "decode_tps_mean": sum(decode) / len(decode) if decode else 0.0,
        "fallback_rate_pct_mean": sum(fallback) / len(fallback) if fallback else 0.0,
        "ttft_ms_mean": sum(ttft) / len(ttft) if ttft else 0.0,
        "unrecoverable_drops": unrecoverable,
    }


def _run_candidate(
    model_name: str,
    profile_path: Path,
    pin_top_k: int,
    capacity: int,
    warmup: str,
    port: int,
    requests_data: list[dict],
    request_timeout_s: int,
    startup_timeout_s: int,
    run_label: str,
    run_logger: RunLogger | None = None,
) -> dict:
    _clear_model_cache(model_name)
    proc, lines, reader = _start_server(
        model_name=model_name,
        profile_path=profile_path,
        pin_top_k=pin_top_k,
        capacity=capacity,
        port=port,
        warmup=warmup,
        startup_timeout_s=startup_timeout_s,
        run_logger=run_logger,
    )
    valid_tool_calls = 0
    total_tool_calls = 0
    loop_failures = 0
    request_failures = 0
    total_requests = len(requests_data)

    tools_by_name = {t["function"]["name"]: t for t in OPENAI_AGENT_TOOLS}
    t0 = time.perf_counter()
    try:
        for i, body in enumerate(requests_data):
            payload = deepcopy(body)
            payload["cache_key"] = f"{run_label}-req-{i}"
            try:
                resp = _run_stream_request(port=port, body=payload, timeout_s=request_timeout_s)
                if _looks_looped(resp["text"]):
                    loop_failures += 1
                for tc in resp["tool_calls"]:
                    total_tool_calls += 1
                    if _validate_tool_call_schema(tc, tools_by_name):
                        valid_tool_calls += 1
            except (RuntimeError, TimeoutError, error.HTTPError, error.URLError):
                request_failures += 1
    finally:
        _stop_server(proc, reader)

    telemetry = _parse_telemetry(lines)
    elapsed = time.perf_counter() - t0
    schema_success = (
        valid_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0
    )
    loop_rate = loop_failures / total_requests if total_requests > 0 else 0.0
    request_failure_rate = request_failures / total_requests if total_requests > 0 else 0.0

    return {
        "pin_top_k": pin_top_k,
        "total_requests": total_requests,
        "elapsed_s": elapsed,
        "tool_calls_total": total_tool_calls,
        "tool_calls_valid": valid_tool_calls,
        "tool_call_schema_success": schema_success,
        "loop_failures": loop_failures,
        "loop_failure_rate": loop_rate,
        "request_failures": request_failures,
        "request_failure_rate": request_failure_rate,
        **telemetry,
    }


def _profile_filename(model_name: str, coding: int, toolchat: int) -> str:
    stem = model_name.split("/")[-1].lower()
    return f"{stem}-mix-{coding}-{toolchat}.json"


def _generate_profile(
    model_name: str,
    capacity: int,
    threshold: float,
    coding: int,
    toolchat: int,
    num_prompts: int,
    seed: int,
    output_path: Path,
    run_logger: RunLogger | None = None,
) -> None:
    cmd = [
        sys.executable,
        "benchmarks/profile_experts.py",
        "--model",
        model_name,
        "--capacity",
        str(capacity),
        "--threshold",
        str(threshold),
        "--prompts",
        "mixed",
        "--coding-weight",
        str(coding),
        "--toolchat-weight",
        str(toolchat),
        "--num-prompts",
        str(num_prompts),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    _print_line(f"\n[profile] generating {output_path.name} ...", run_logger)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        if run_logger is not None:
            run_logger.write_line(stripped)
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _aggregate_runs(runs: list[dict]) -> dict:
    total_tool_calls = sum(r["tool_calls_total"] for r in runs)
    total_valid_calls = sum(r["tool_calls_valid"] for r in runs)
    total_requests = sum(r["total_requests"] for r in runs)
    total_loops = sum(r["loop_failures"] for r in runs)
    total_req_fail = sum(r["request_failures"] for r in runs)
    total_unrecoverable = sum(r["unrecoverable_drops"] for r in runs)

    decode_vals = [r["decode_tps_mean"] for r in runs]
    fallback_vals = [r["fallback_rate_pct_mean"] for r in runs]
    ttft_vals = [r["ttft_ms_mean"] for r in runs]

    schema_success = total_valid_calls / total_tool_calls if total_tool_calls > 0 else 0.0
    loop_rate = total_loops / total_requests if total_requests > 0 else 0.0
    req_fail_rate = total_req_fail / total_requests if total_requests > 0 else 0.0

    return {
        "runs": runs,
        "tool_calls_total": total_tool_calls,
        "tool_calls_valid": total_valid_calls,
        "tool_call_schema_success": schema_success,
        "loop_failures": total_loops,
        "loop_failure_rate": loop_rate,
        "request_failures": total_req_fail,
        "request_failure_rate": req_fail_rate,
        "unrecoverable_drops": total_unrecoverable,
        "decode_tps_mean": sum(decode_vals) / len(decode_vals) if decode_vals else 0.0,
        "fallback_rate_pct_mean": sum(fallback_vals) / len(fallback_vals) if fallback_vals else 0.0,
        "ttft_ms_mean": sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0.0,
    }


def _passes_gate(metrics: dict) -> bool:
    return (
        metrics["tool_call_schema_success"] >= STRICT_GATE["tool_call_schema_success_min"]
        and metrics["loop_failure_rate"] <= STRICT_GATE["loop_failure_rate_max"]
        and metrics["unrecoverable_drops"] <= STRICT_GATE["unrecoverable_drops_max"]
    )


def _write_summary_md(output_path: Path, candidates: list[dict], winner: dict | None) -> None:
    lines = []
    lines.append("| mix | pin_k | pass | schema_success | loop_rate | unrecoverable | decode tok/s | fallback % | ttft ms |")
    lines.append("|---|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for c in candidates:
        m = c["metrics"]
        lines.append(
            f"| {c['mix_label']} | {c['pin_top_k']} | {'Y' if c['pass_gate'] else 'N'} | "
            f"{m['tool_call_schema_success']:.3f} | {m['loop_failure_rate']:.3f} | "
            f"{m['unrecoverable_drops']} | {m['decode_tps_mean']:.2f} | "
            f"{m['fallback_rate_pct_mean']:.2f} | {m['ttft_ms_mean']:.1f} |"
        )
    lines.append("")
    if winner is None:
        lines.append("No candidate passed strict quality gate.")
    else:
        lines.append(
            f"Winner: mix={winner['mix_label']} pin_top_k={winner['pin_top_k']} "
            f"decode={winner['metrics']['decode_tps_mean']:.2f} tok/s"
        )
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep profile mixes and pin_top_k values.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--capacity", type=int, default=208)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mixes", default="70:30,50:50,30:70")
    parser.add_argument("--pin-counts", default="0,16,32,48,64")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--profile-prompts", type=int, default=DEFAULT_NUM_PROFILE_PROMPTS)
    parser.add_argument("--eval-requests", type=int, default=DEFAULT_NUM_EVAL_REQUESTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", choices=["hybrid", "full", "none"], default="hybrid")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--request-timeout-s", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout-s", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--port-base", type=int, default=DEFAULT_PORT_BASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR)
    args = parser.parse_args()

    mixes = _parse_mix_list(args.mixes)
    pin_counts = _parse_int_list(args.pin_counts)
    output_dir = Path(args.output_dir)
    profiles_dir = Path(args.profiles_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"pin_sweep_{ts}.log"
    run_logger = RunLogger(log_path)

    try:
        profiles: dict[tuple[int, int], Path] = {}
        for idx, (coding, toolchat) in enumerate(mixes):
            profile_path = profiles_dir / _profile_filename(args.model, coding, toolchat)
            _generate_profile(
                model_name=args.model,
                capacity=args.capacity,
                threshold=args.threshold,
                coding=coding,
                toolchat=toolchat,
                num_prompts=args.profile_prompts,
                seed=args.seed + idx,
                output_path=profile_path,
                run_logger=run_logger,
            )
            profiles[(coding, toolchat)] = profile_path

        results = {
            "timestamp": ts,
            "model": args.model,
            "capacity": args.capacity,
            "threshold": args.threshold,
            "mixes": mixes,
            "pin_counts": pin_counts,
            "repeats": args.repeats,
            "profile_prompts": args.profile_prompts,
            "eval_requests": args.eval_requests,
            "gate": STRICT_GATE,
            "profiles_dir": str(profiles_dir),
            "log_path": str(log_path),
            "candidates": [],
        }

        port = args.port_base
        for mix_idx, (coding, toolchat) in enumerate(mixes):
            profile_path = profiles[(coding, toolchat)]
            mix_label = f"{coding}/{toolchat}"
            requests_data = _build_eval_requests(
                coding_weight=coding,
                tool_weight=toolchat,
                n_requests=args.eval_requests,
                seed=args.seed + 1000 + mix_idx,
                max_tokens=args.max_tokens,
            )
            for pin_top_k in pin_counts:
                run_metrics = []
                for rep in range(args.repeats):
                    run_label = f"mix-{coding}-{toolchat}-k-{pin_top_k}-r-{rep}"
                    _print_line(
                        f"\n[sweep] mix={mix_label} pin_top_k={pin_top_k} repeat={rep + 1}/{args.repeats} "
                        f"port={port}",
                        run_logger,
                    )
                    metrics = _run_candidate(
                        model_name=args.model,
                        profile_path=profile_path,
                        pin_top_k=pin_top_k,
                        capacity=args.capacity,
                        warmup=args.warmup,
                        port=port,
                        requests_data=requests_data,
                        request_timeout_s=args.request_timeout_s,
                        startup_timeout_s=args.startup_timeout_s,
                        run_label=run_label,
                        run_logger=run_logger,
                    )
                    run_metrics.append(metrics)
                    port += 1

                aggregated = _aggregate_runs(run_metrics)
                candidate = {
                    "mix": [coding, toolchat],
                    "mix_label": mix_label,
                    "profile_path": str(profile_path),
                    "pin_top_k": pin_top_k,
                    "metrics": aggregated,
                }
                candidate["pass_gate"] = _passes_gate(aggregated)
                results["candidates"].append(candidate)

        passing = [c for c in results["candidates"] if c["pass_gate"]]
        winner = None
        if passing:
            winner = max(
                passing,
                key=lambda c: (
                    c["metrics"]["decode_tps_mean"],
                    -c["metrics"]["fallback_rate_pct_mean"],
                    -c["metrics"]["ttft_ms_mean"],
                ),
            )
            results["winner"] = winner
        else:
            results["winner"] = None

        json_path = output_dir / f"pin_sweep_{ts}.json"
        md_path = output_dir / f"pin_sweep_summary_{ts}.md"
        json_path.write_text(json.dumps(results, indent=2))
        _write_summary_md(md_path, results["candidates"], winner)
        _print_line(f"\nResults JSON: {json_path}", run_logger)
        _print_line(f"Summary MD:   {md_path}", run_logger)
        _print_line(f"Run log:      {log_path}", run_logger)
        if winner is None:
            _print_line("No candidate passed strict quality gate.", run_logger)
        else:
            _print_line(
                "Winner: "
                f"mix={winner['mix_label']} pin_top_k={winner['pin_top_k']} "
                f"decode={winner['metrics']['decode_tps_mean']:.2f} tok/s "
                f"fallback={winner['metrics']['fallback_rate_pct_mean']:.2f}%",
                run_logger,
            )
    finally:
        run_logger.close()


if __name__ == "__main__":
    main()
