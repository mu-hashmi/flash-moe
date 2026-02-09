import asyncio
import json
import os
import re
import signal
import time
import uuid
from pathlib import Path

MAX_TOKENS_CAP = 4096
MAX_INPUT_TOKENS = 16384

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .lazy_experts.generate import Session

import sys
if not sys.stdout.isatty():
    import functools
    print = functools.partial(print, flush=True)


def _find_profile(model_name: str) -> str | None:
    profiles_dir = Path(__file__).parent.parent / "profiles"
    if not profiles_dir.is_dir():
        return None
    slug = model_name.split("/")[-1].lower()
    for suffix in ("-4bit", "-8bit", "-bf16", "-fp16"):
        slug = slug.removesuffix(suffix)
    for f in profiles_dir.iterdir():
        if f.suffix == ".json" and f.stem == slug:
            return str(f)
    return None


def _format_messages(messages: list, system: str | None = None) -> list:
    """Normalize Anthropic/OpenAI message formats to HF chat template format."""
    formatted = []
    if system:
        formatted.append({"role": "system", "content": system})
    for msg in messages:
        content = msg.get("content", "")
        role = msg["role"]
        if isinstance(content, list):
            parts = []
            tool_uses = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_uses.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": block.get("input", {}),
                        },
                    })
                elif block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b["text"] for b in tool_content if b.get("type") == "text"
                        )
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": tool_content,
                    })
                    continue
            if parts or tool_uses:
                entry = {"role": role, "content": "\n".join(parts) if parts else ""}
                if tool_uses:
                    entry["tool_calls"] = tool_uses
                formatted.append(entry)
        elif role == "tool":
            formatted.append(msg)
        else:
            entry = {"role": role, "content": content or ""}
            if "tool_calls" in msg:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc)
                    if "function" in tc_copy:
                        func = dict(tc_copy["function"])
                        if isinstance(func.get("arguments"), str):
                            func["arguments"] = json.loads(func["arguments"])
                        tc_copy["function"] = func
                    tool_calls.append(tc_copy)
                entry["tool_calls"] = tool_calls
            formatted.append(entry)
    return formatted


def _convert_tools_anthropic_to_openai(tools: list) -> list:
    """Convert Anthropic tool format to OpenAI format for chat templates."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        },
    } for t in tools]


MODEL_SAMPLING_DEFAULTS = {
    "qwen3-coder": {"temp": 1.0, "top_p": 0.95, "top_k": 40},
}


def _sampling_defaults(model_name: str) -> dict:
    slug = model_name.split("/")[-1].lower()
    for prefix, defaults in MODEL_SAMPLING_DEFAULTS.items():
        if prefix in slug:
            return dict(defaults)
    return {}


def _sampling_kwargs(body: dict, model_name: str) -> dict:
    defaults = _sampling_defaults(model_name)
    if "temperature" in body:
        defaults["temp"] = float(body["temperature"])
    if "top_p" in body:
        defaults["top_p"] = float(body["top_p"])
    if "top_k" in body:
        defaults["top_k"] = int(body["top_k"])
    return defaults


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text)


def _make_msg_id():
    return f"msg_{uuid.uuid4().hex[:24]}"


def _make_chatcmpl_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class Server:
    def __init__(self, model_name: str, capacity: int | None = None,
                 profile_path: str | None = None,
                 max_tokens: int = MAX_TOKENS_CAP,
                 max_input_tokens: int = MAX_INPUT_TOKENS,
                 kv_bits: int | None = None):
        self._model_name = model_name
        self._capacity = capacity
        self._profile_path = profile_path or _find_profile(model_name)
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self._model_id = model_name.split("/")[-1]
        self._max_tokens = max_tokens
        self._max_input_tokens = max_input_tokens
        self._kv_bits = kv_bits

    def load(self):
        session = Session(
            self._model_name,
            cache_dir=str(Path.home() / ".cache" / "mlx-moe"),
            profile_path=self._profile_path,
            capacity=self._capacity,
        )
        session._ensure_loaded("Hello")
        self._model = session._model
        self._tokenizer = session._tokenizer
        self._memory_gb = session.memory_gb

    def _stream(self, prompt: str, max_tokens: int, **sampling):
        import mlx_lm as _mlx_lm
        from mlx_lm.sample_utils import make_sampler
        kv_kwargs = {}
        if self._kv_bits is not None:
            kv_kwargs = dict(kv_bits=self._kv_bits, kv_group_size=64, quantized_kv_start=0)
        yield from _mlx_lm.stream_generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=make_sampler(**sampling), **kv_kwargs,
        )

    def _tokenize_messages(self, messages: list, tools: list | None = None) -> str:
        tokenizer = self._tokenizer
        if tokenizer.has_chat_template:
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if tools and tokenizer.has_tool_calling:
                kwargs["tools"] = tools
            if tokenizer.has_thinking:
                kwargs["enable_thinking"] = False
            return tokenizer.apply_chat_template(messages, **kwargs)
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg.get('content', '')}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _check_input_length(self, input_tokens: int):
        if input_tokens > self._max_input_tokens:
            return JSONResponse({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Prompt too long: {input_tokens} tokens > {self._max_input_tokens} limit",
                },
            }, status_code=400)
        return None

    async def handle_models(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": self._model_id,
                "object": "model",
                "owned_by": "mlx-moe",
            }],
        })

    async def handle_chat_completions(self, request: Request):
        body = await request.json()
        messages = body["messages"]
        raw = body.get("max_tokens")
        if raw is None:
            raw = body.get("max_completion_tokens")
        max_tokens = min(raw if raw is not None else self._max_tokens, self._max_tokens)
        stream = body.get("stream", False)
        tools = body.get("tools")
        sampling = _sampling_kwargs(body, self._model_name)

        formatted = _format_messages(messages)
        prompt = self._tokenize_messages(formatted, tools=tools)
        input_tokens = len(self._tokenizer.encode(prompt))

        if err := self._check_input_length(input_tokens):
            return err

        print(f"  [openai] max_tokens={max_tokens} stream={stream} input_tokens={input_tokens}")

        if stream:
            return StreamingResponse(
                self._stream_openai(prompt, max_tokens, tools, **sampling),
                media_type="text/event-stream",
            )

        async with self._lock:
            text, gen_tokens, tps = self._generate_sync(prompt, max_tokens, **sampling)

        return JSONResponse({
            "id": _make_chatcmpl_id(),
            "object": "chat.completion",
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": gen_tokens,
                "total_tokens": input_tokens + gen_tokens,
            },
        })

    async def handle_messages(self, request: Request):
        body = await request.json()
        messages = body["messages"]
        system = body.get("system")
        if isinstance(system, list):
            system = "\n".join(
                b["text"] for b in system if b.get("type") == "text"
            )
        max_tokens = min(body.get("max_tokens", self._max_tokens), self._max_tokens)
        stream = body.get("stream", False)
        sampling = _sampling_kwargs(body, self._model_name)

        tools_anthropic = body.get("tools")
        tools_openai = _convert_tools_anthropic_to_openai(tools_anthropic) if tools_anthropic else None

        formatted = _format_messages(messages, system=system)
        prompt = self._tokenize_messages(formatted, tools=tools_openai)
        input_tokens = len(self._tokenizer.encode(prompt))

        if err := self._check_input_length(input_tokens):
            return err

        if not stream:
            max_tokens = min(max_tokens, 512)

        print(f"  [anthropic] max_tokens={max_tokens} stream={stream} tools={len(tools_anthropic) if tools_anthropic else 0} msgs={len(messages)} input_tokens={input_tokens}")

        if stream:
            return StreamingResponse(
                self._stream_anthropic(prompt, max_tokens, tools_openai, input_tokens, **sampling),
                media_type="text/event-stream",
            )

        async with self._lock:
            text, gen_tokens, tps = self._generate_sync(prompt, max_tokens, **sampling)

        return JSONResponse({
            "id": _make_msg_id(),
            "type": "message",
            "role": "assistant",
            "model": self._model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": gen_tokens,
            },
        })

    def _generate_sync(self, prompt: str, max_tokens: int, **sampling) -> tuple[str, int, float]:
        text = ""
        gen_tokens = 0
        tps = 0.0
        for resp in self._stream(prompt, max_tokens=max_tokens, **sampling):
            text += resp.text
            gen_tokens = resp.generation_tokens
            tps = resp.generation_tps
        text = _strip_thinking(text)
        if gen_tokens > 1:
            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        return text, gen_tokens, tps

    async def _stream_openai(self, prompt: str, max_tokens: int,
                             tools: list | None, **sampling):
        chat_id = _make_chatcmpl_id()
        created = int(time.time())

        chunk = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": self._model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        gen_tokens = 0
        tps = 0.0
        finish = "stop"

        tokenizer = self._tokenizer
        has_tools = tools and tokenizer.has_tool_calling
        in_tool_call = False
        tool_text = ""
        tool_idx = 0
        in_thinking = False

        try:
            async with self._lock:
                for resp in self._stream(prompt, max_tokens=max_tokens, **sampling):
                    gen_tokens = resp.generation_tokens
                    tps = resp.generation_tps

                    if tokenizer.has_thinking:
                        if resp.text == tokenizer.think_start:
                            in_thinking = True
                            continue
                        if in_thinking:
                            if resp.text == tokenizer.think_end:
                                in_thinking = False
                            continue

                    if has_tools and resp.text == tokenizer.tool_call_start:
                        in_tool_call = True
                        continue
                    if in_tool_call:
                        if resp.text == tokenizer.tool_call_end:
                            in_tool_call = False
                            parsed = tokenizer.tool_parser(tool_text, tools)
                            if not isinstance(parsed, list):
                                parsed = [parsed]
                            for tc in parsed:
                                tc_id = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:12]}"
                                args_str = json.dumps(tc["arguments"], ensure_ascii=False)
                                chunk = {
                                    "id": chat_id, "object": "chat.completion.chunk",
                                    "created": created, "model": self._model_id,
                                    "choices": [{"index": 0, "delta": {
                                        "tool_calls": [{"index": tool_idx, "id": tc_id, "type": "function",
                                                        "function": {"name": tc["name"], "arguments": ""}}],
                                    }, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                chunk = {
                                    "id": chat_id, "object": "chat.completion.chunk",
                                    "created": created, "model": self._model_id,
                                    "choices": [{"index": 0, "delta": {
                                        "tool_calls": [{"index": tool_idx, "function": {"arguments": args_str}}],
                                    }, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                tool_idx += 1
                            tool_text = ""
                            finish = "tool_calls"
                            continue
                        tool_text += resp.text
                        continue

                    chunk = {
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": self._model_id,
                        "choices": [{"index": 0, "delta": {"content": resp.text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0)

                    if resp.finish_reason == "length":
                        finish = "length"

            chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": self._model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        except (OSError, asyncio.CancelledError):
            print(f"  [client disconnected after {gen_tokens} tokens]")

    async def _stream_anthropic(self, prompt: str, max_tokens: int,
                                tools: list | None, input_tokens: int, **sampling):
        msg_id = _make_msg_id()

        event = {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": self._model_id, "stop_reason": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(event)}\n\n"

        gen_tokens = 0
        tps = 0.0
        stop_reason = "end_turn"
        content_idx = 0

        tokenizer = self._tokenizer
        has_tools = tools and tokenizer.has_tool_calling
        in_tool_call = False
        tool_text = ""
        text_block_open = False
        in_thinking = False

        try:
            async with self._lock:
                for resp in self._stream(prompt, max_tokens=max_tokens, **sampling):
                    gen_tokens = resp.generation_tokens
                    tps = resp.generation_tps

                    if tokenizer.has_thinking:
                        if resp.text == tokenizer.think_start:
                            in_thinking = True
                            continue
                        if in_thinking:
                            if resp.text == tokenizer.think_end:
                                in_thinking = False
                            continue

                    if has_tools and resp.text == tokenizer.tool_call_start:
                        if text_block_open:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx - 1})}\n\n"
                            text_block_open = False
                        in_tool_call = True
                        continue
                    if in_tool_call:
                        if resp.text == tokenizer.tool_call_end:
                            in_tool_call = False
                            parsed = tokenizer.tool_parser(tool_text, tools)
                            if not isinstance(parsed, list):
                                parsed = [parsed]
                            for tc in parsed:
                                tc_id = tc.pop("id", None) or f"toolu_{uuid.uuid4().hex[:24]}"
                                event = {
                                    "type": "content_block_start", "index": content_idx,
                                    "content_block": {"type": "tool_use", "id": tc_id, "name": tc["name"], "input": {}},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                                args_json = json.dumps(tc["arguments"], ensure_ascii=False)
                                event = {
                                    "type": "content_block_delta", "index": content_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args_json},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                                content_idx += 1
                            tool_text = ""
                            stop_reason = "tool_use"
                            continue
                        tool_text += resp.text
                        continue

                    if not text_block_open:
                        event = {
                            "type": "content_block_start", "index": content_idx,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                        text_block_open = True
                        content_idx += 1

                    event = {
                        "type": "content_block_delta", "index": content_idx - 1,
                        "delta": {"type": "text_delta", "text": resp.text},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0)

                    if resp.finish_reason == "length":
                        stop_reason = "max_tokens"

            # Salvage truncated tool calls — if the model hit the token cap
            # mid-tool-call, try to parse what we have before dropping it.
            if in_tool_call and tool_text:
                print(f"  [WARNING: tool call truncated at {gen_tokens} tokens, {len(tool_text)} chars buffered]")
                try:
                    parsed = tokenizer.tool_parser(tool_text, tools)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    for tc in parsed:
                        tc_id = tc.pop("id", None) or f"toolu_{uuid.uuid4().hex[:24]}"
                        event = {
                            "type": "content_block_start", "index": content_idx,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": tc["name"], "input": {}},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                        args_json = json.dumps(tc["arguments"], ensure_ascii=False)
                        event = {
                            "type": "content_block_delta", "index": content_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args_json},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                        content_idx += 1
                    stop_reason = "tool_use"
                except Exception:
                    print(f"  [tool call unrecoverable, dropping]")

            if text_block_open:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx - 1})}\n\n"

            event = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": gen_tokens},
            }
            yield f"event: message_delta\ndata: {json.dumps(event)}\n\n"

            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        except (OSError, asyncio.CancelledError):
            print(f"  [client disconnected after {gen_tokens} tokens]")


def run_server(model_name: str, host: str = "127.0.0.1", port: int = 8080,
               capacity: int | None = None, profile_path: str | None = None,
               max_tokens: int = MAX_TOKENS_CAP,
               max_input_tokens: int = MAX_INPUT_TOKENS,
               kv_bits: int | None = None):
    import uvicorn
    from contextlib import asynccontextmanager

    server = Server(model_name, capacity=capacity, profile_path=profile_path,
                    max_tokens=max_tokens, max_input_tokens=max_input_tokens,
                    kv_bits=kv_bits)

    print(f"mlx-moe serve")
    print(f"  Model:    {model_name}")
    print(f"  Profile:  {server._profile_path or 'none'}")
    print(f"  Endpoint: http://{host}:{port}")
    defaults = _sampling_defaults(model_name)
    print(f"  Limits:   {max_input_tokens} input, {max_tokens} output")
    if defaults:
        print(f"  Sampling: {defaults}")
    print()
    print("Loading model...")
    server.load()
    print(f"  Memory: {server._memory_gb:.1f} GB")
    print(f"  Ready.")
    print()

    @asynccontextmanager
    async def lifespan(app):
        signal.signal(signal.SIGINT, lambda *_: os._exit(0))
        yield

    app = Starlette(
        routes=[
            Route("/v1/models", server.handle_models, methods=["GET"]),
            Route("/v1/chat/completions", server.handle_chat_completions, methods=["POST"]),
            Route("/v1/messages", server.handle_messages, methods=["POST"]),
        ],
        lifespan=lifespan,
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")
