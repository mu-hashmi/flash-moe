"""Tool-chat scenarios that mirror server-side message and tool formatting."""

from __future__ import annotations

from copy import deepcopy
import json

from mlx_moe.server import _convert_tools_anthropic_to_openai, _format_messages


OPENAI_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the current repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for a regex pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories from a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Execute a test command and return stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "integer", "minimum": 1},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified patch to files in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_pull_request",
            "description": "Create a pull request with summary and risk notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "branch": {"type": "string"},
                },
                "required": ["title", "body", "branch"],
                "additionalProperties": False,
            },
        },
    },
]

ANTHROPIC_AGENT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the current repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_code",
        "description": "Search the repository for a regex pattern and return matching lines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories from a path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_tests",
        "description": "Execute a test command and return stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {"type": "integer", "minimum": 1},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_patch",
        "description": "Apply a unified patch to files in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    },
    {
        "name": "open_pull_request",
        "description": "Create a pull request with summary and risk notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "branch": {"type": "string"},
            },
            "required": ["title", "body", "branch"],
            "additionalProperties": False,
        },
    },
]

PROFILE_SCENARIOS = [
    {
        "name": "openai_repo_triage",
        "api": "openai",
        "system": (
            "You are an expert coding agent working in a local repository. "
            "Read files, reason about tradeoffs, and produce minimal-risk patches."
        ),
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Investigate why startup refinement reports high fallback_rate in serve mode. "
                    "Focus on mlx_moe/server.py and suggest a patch with tests."
                ),
            }
        ],
    },
    {
        "name": "openai_patch_after_search",
        "api": "openai",
        "system": "You are a coding assistant that must use tools before editing.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {"role": "user", "content": "Find where dynamic cache updates are scheduled and optimize policy tuning."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_code", "arguments": {"pattern": "_dynamic_update_policy", "path": "mlx_moe"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "mlx_moe/server.py:307:def _dynamic_update_policy(...)\nmlx_moe/server.py:414: interval, budget = self._dynamic_update_policy(...)",
            },
            {"role": "assistant", "content": "I found the scheduling policy and call-site in stream generation."},
            {"role": "user", "content": "Patch it to be less aggressive after fallback drops below 8% and add tests."},
        ],
    },
    {
        "name": "openai_tool_call_history",
        "api": "openai",
        "system": "You are helping with a production incident in an ML inference server.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {"role": "user", "content": "Read the telemetry logs and explain TTFT regressions after request 1."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": {"path": "logs/serve.log"}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": (
                    "[telemetry] prefill=55526ms ttft=55580ms decode=4.8 tok/s dcu_calls=82 fallback_rate=33.46%\n"
                    "[telemetry] prefill=447ms ttft=507ms decode=6.2 tok/s dcu_calls=69 fallback_rate=21.46%"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Request 1 is full prefill, while request 2 reuses cached KV prefix and only prefills a suffix. "
                    "Fallback is still high, suggesting profile mismatch."
                ),
            },
            {"role": "user", "content": "Propose a profile-generation approach tailored for tool-chat traffic."},
        ],
    },
    {
        "name": "openai_bugfix_with_tests",
        "api": "openai",
        "system": "Be concise and only modify files required for the fix.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Fix OpenAI truncated tool-call salvage so partially emitted JSON still returns structured arguments. "
                    "Add integration tests."
                ),
            }
        ],
    },
    {
        "name": "openai_large_tool_schema_context",
        "api": "openai",
        "system": (
            "Act as a senior code reviewer. Follow this rubric: correctness, latency, memory pressure, "
            "regression risk, and test adequacy."
        ),
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Review the proposed keyed KV-cache LRU design. Include edge cases for session_id, metadata, "
                    "and user-derived cache keys. Provide concrete test vectors."
                ),
            }
        ],
    },
    {
        "name": "anthropic_tool_use_roundtrip",
        "api": "anthropic",
        "system": "You are a repository-aware assistant. Use tools for file access.",
        "tools": ANTHROPIC_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Find all references to _stream_openai and summarize tool-call handling."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_code",
                        "input": {"pattern": "_stream_openai", "path": "mlx_moe"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "mlx_moe/server.py:633:async def _stream_openai(..."}],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I found the OpenAI streaming implementation and parser path."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Now patch truncated tool-call recovery to fail open."}],
            },
        ],
    },
    {
        "name": "anthropic_multi_tool_sequence",
        "api": "anthropic",
        "system": "You are auditing an ML serving stack.",
        "tools": ANTHROPIC_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "List benchmark scripts and identify profile-generation entrypoints."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "list_files",
                        "input": {"path": "benchmarks", "recursive": False},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": [
                            {"type": "text", "text": "profile_experts.py\ntest_model.py\nvalidate_quality.py"}
                        ],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_3",
                        "name": "read_file",
                        "input": {"path": "benchmarks/profile_experts.py", "start_line": 1, "end_line": 120},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_3",
                        "content": [{"type": "text", "text": "PROMPTS_DIVERSE = [...] PROMPTS_CODING = [...]"}],
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Add a new prompt preset that mirrors server-side tool formatting."}],
            },
        ],
    },
    {
        "name": "anthropic_test_failure_debug",
        "api": "anthropic",
        "system": "Use tools, then provide a minimal patch and test plan.",
        "tools": ANTHROPIC_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Pytest fails in TestServerCache. Diagnose and fix."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_4",
                        "name": "run_tests",
                        "input": {"command": "uv run pytest tests/test_integration.py -k cache", "timeout_s": 120},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_4",
                        "content": [{"type": "text", "text": "AssertionError: expected cache_key k1 but got default"}],
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Patch the key-resolution order and add targeted tests."}],
            },
        ],
    },
    {
        "name": "openai_architecture_reasoning",
        "api": "openai",
        "system": "Prioritize practical latency wins over theoretical purity.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain why MLX MoE optimizations should avoid mx.eval in forward pass and why Python-side "
                    "tensor assembly dominates cost on Apple Silicon."
                ),
            }
        ],
    },
    {
        "name": "openai_agentic_refactor_plan",
        "api": "openai",
        "system": "Act as a principal engineer reviewing a refactor proposal.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Refactor server request handling to tokenize once, enforce max input length once, "
                    "and pass prompt_tokens through generation. Include migration risks."
                ),
            }
        ],
    },
    {
        "name": "anthropic_cross_api_tooling",
        "api": "anthropic",
        "system": "Maintain compatibility between OpenAI and Anthropic wire formats.",
        "tools": ANTHROPIC_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Ensure tool calls are losslessly converted between APIs."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_5",
                        "name": "read_file",
                        "input": {"path": "mlx_moe/server.py", "start_line": 1, "end_line": 140},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_5",
                        "content": [{"type": "text", "text": "def _format_messages ... def _convert_tools_anthropic_to_openai ..."}],
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Propose tests that validate conversion semantics for nested arguments."}],
            },
        ],
    },
    {
        "name": "openai_long_context_debugging",
        "api": "openai",
        "system": "You are diagnosing performance regressions in a long-context coding session.",
        "tools": OPENAI_AGENT_TOOLS,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Given a 16k-token conversation history and repeated tool schemas, explain why KV prefix reuse "
                    "can cut prefill latency from 55s to under 1s on subsequent requests."
                ),
            }
        ],
    },
]


def get_profile_scenarios() -> list[dict]:
    return deepcopy(PROFILE_SCENARIOS)


def render_scenario_prompt(tokenizer, scenario: dict) -> str:
    api = scenario["api"]
    messages = scenario["messages"]
    system = scenario.get("system")
    tools = scenario.get("tools")

    if api == "anthropic":
        tools_openai = _convert_tools_anthropic_to_openai(tools) if tools else None
    elif api == "openai":
        tools_openai = tools
    else:
        raise ValueError(f"Unsupported scenario api: {api}")

    formatted = _format_messages(messages, system=system)

    if tokenizer.has_chat_template:
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if tools_openai and tokenizer.has_tool_calling:
            kwargs["tools"] = tools_openai
        if tokenizer.has_thinking:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(formatted, **kwargs)

    parts = []
    if system:
        parts.append(f"system: {system}")
    for msg in formatted:
        if msg["role"] == "tool":
            parts.append(f"tool[{msg.get('tool_call_id', '')}]: {msg.get('content', '')}")
            continue
        content = msg.get("content", "")
        if "tool_calls" in msg:
            content = f"{content}\nTOOL_CALLS: {json.dumps(msg['tool_calls'], ensure_ascii=False)}"
        parts.append(f"{msg['role']}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)
