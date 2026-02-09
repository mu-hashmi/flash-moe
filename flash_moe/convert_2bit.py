"""Requantize an MLX MoE model with mixed precision: 2-bit experts + 4-bit base.

Works from any existing quantized model (4-bit, 8-bit) by chaining
dequantize → requantize through MLX lazy evaluation. No fp16 intermediate
files — peak RAM is ~one shard eval at a time (~5 GB).

Usage:
    uv run python -m flash_moe.convert_2bit \
        mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit \
        --output ~/models/Mixtral-8x22B-Instruct-v0.1-2bit \
        --expert-bits 2 --base-bits 4

    # Uniform 2-bit (all layers):
    uv run python -m flash_moe.convert_2bit \
        mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit \
        --output ~/models/Mixtral-8x22B-Instruct-v0.1-2bit \
        --expert-bits 2 --base-bits 2

    # Dry run (shows what would change, no disk writes):
    uv run python -m flash_moe.convert_2bit \
        mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit \
        --dry-run
"""

import argparse
import sys
from pathlib import Path

import mlx.nn as nn

from mlx_lm.utils import (
    dequantize_model,
    load,
    quantize_model,
    save,
)


def _is_expert_path(path: str) -> bool:
    return "switch_mlp" in path


def _build_predicate(expert_bits, base_bits, group_size):
    def predicate(path: str, module: nn.Module):
        bits = expert_bits if _is_expert_path(path) else base_bits
        return {"group_size": group_size, "bits": bits, "mode": "affine"}

    return predicate


def dry_run(model, expert_bits, base_bits, group_size):
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    expert_params = 0
    base_params = 0

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        is_quantized = isinstance(
            module, (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)
        )

        if not is_quantized:
            continue

        # Weight is packed — recover original param count
        n_packed = 1
        for d in module.weight.shape:
            n_packed *= d
        n_params = n_packed * 32 // module.bits

        if _is_expert_path(name):
            expert_params += n_params
        else:
            base_params += n_params

    total = expert_params + base_params
    expert_size = expert_params * expert_bits / 8
    base_size = base_params * base_bits / 8
    total_size = expert_size + base_size

    print(f"Expert params: {expert_params / 1e9:.1f}B ({expert_params * 100 / total:.1f}%)")
    print(f"Base params:   {base_params / 1e9:.1f}B ({base_params * 100 / total:.1f}%)")
    print()
    print(f"Expert bits: {expert_bits}, base bits: {base_bits}, group size: {group_size}")
    print()
    print(f"Estimated output size:")
    print(f"  Experts: {expert_size / 1e9:.1f} GB at {expert_bits}-bit")
    print(f"  Base:    {base_size / 1e9:.1f} GB at {base_bits}-bit")
    print(f"  Total:   {total_size / 1e9:.1f} GB")

    # Show per-layer breakdown for first MoE layer
    print()
    print("Sample module assignments (first 3 layers):")
    count = 0
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        is_expert = _is_expert_path(name)
        bits = expert_bits if is_expert else base_bits
        tag = "EXPERT" if is_expert else "BASE"
        if "layers.0." in name or "layers.1." in name or "layers.2." in name:
            print(f"  {name}: {bits}-bit [{tag}]")
            count += 1
        if count > 20:
            print("  ...")
            break


def main():
    parser = argparse.ArgumentParser(description="Requantize MLX MoE model with mixed precision")
    parser.add_argument("model", help="HuggingFace repo or local path to source model")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--expert-bits", type=int, default=2)
    parser.add_argument("--base-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without converting")
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("--output is required unless using --dry-run")

    if not args.dry_run:
        output = Path(args.output).expanduser()
        if output.exists():
            print(f"Error: output path {output} already exists", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Loading {args.model} (lazy)")
    model, tokenizer, config = load(args.model, return_config=True, lazy=True)

    if args.dry_run:
        dry_run(model, args.expert_bits, args.base_bits, args.group_size)
        return

    print("[INFO] Dequantizing")
    model = dequantize_model(model)
    config.pop("quantization", None)
    config.pop("quantization_config", None)

    predicate = _build_predicate(args.expert_bits, args.base_bits, args.group_size)

    print(f"[INFO] Requantizing (experts={args.expert_bits}-bit, base={args.base_bits}-bit)")
    model, config = quantize_model(
        model, config, args.group_size, args.expert_bits, quant_predicate=predicate
    )

    print(f"[INFO] Saving to {output}")
    save(output, args.model, model, tokenizer, config)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
