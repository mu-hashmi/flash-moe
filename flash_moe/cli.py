import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="flash-moe")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start OpenAI/Anthropic-compatible API server")
    serve.add_argument("model", help="HuggingFace model name (e.g. mlx-community/Qwen3-Coder-Next-4bit)")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--capacity", type=int, default=None)
    serve.add_argument("--profile", default=None, help="Path to expert profile JSON")

    args = parser.parse_args()

    if args.command == "serve":
        from .server import run_server
        run_server(
            model_name=args.model,
            host=args.host,
            port=args.port,
            capacity=args.capacity,
            profile_path=args.profile,
        )
    else:
        parser.print_help()
        sys.exit(1)
