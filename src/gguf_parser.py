"""GGUF parser for extracting expert tensor offsets."""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: GGMLQuantizationType
    offset: int  # byte offset in file
    size: int    # size in bytes
    n_elements: int


@dataclass
class ExpertInfo:
    layer: int
    expert_id: int
    tensor_type: str  # 'down', 'gate', 'up'
    offset: int       # byte offset in file
    size: int         # size in bytes


def get_quant_block_size(dtype: GGMLQuantizationType) -> tuple[int, int]:
    """Return (block_size, bytes_per_block) for quantization type."""
    quant_info = {
        GGMLQuantizationType.F32: (1, 4),
        GGMLQuantizationType.F16: (1, 2),
        GGMLQuantizationType.BF16: (1, 2),
        GGMLQuantizationType.F64: (1, 8),
        GGMLQuantizationType.Q4_0: (32, 18),
        GGMLQuantizationType.Q4_1: (32, 20),
        GGMLQuantizationType.Q5_0: (32, 22),
        GGMLQuantizationType.Q5_1: (32, 24),
        GGMLQuantizationType.Q8_0: (32, 34),
        GGMLQuantizationType.Q8_1: (32, 36),
        GGMLQuantizationType.Q2_K: (256, 84),
        GGMLQuantizationType.Q3_K: (256, 110),
        GGMLQuantizationType.Q4_K: (256, 144),
        GGMLQuantizationType.Q5_K: (256, 176),
        GGMLQuantizationType.Q6_K: (256, 210),
        GGMLQuantizationType.Q8_K: (256, 292),
        GGMLQuantizationType.IQ1_S: (256, 50),
        GGMLQuantizationType.IQ1_M: (256, 56),
        GGMLQuantizationType.IQ2_XXS: (256, 66),
        GGMLQuantizationType.IQ2_XS: (256, 74),
        GGMLQuantizationType.IQ2_S: (256, 82),
        GGMLQuantizationType.IQ3_XXS: (256, 98),
        GGMLQuantizationType.IQ3_S: (256, 110),
        GGMLQuantizationType.IQ4_NL: (32, 18),
        GGMLQuantizationType.IQ4_XS: (256, 138),
        GGMLQuantizationType.TQ1_0: (256, 54),
        GGMLQuantizationType.TQ2_0: (256, 66),
        GGMLQuantizationType.I8: (1, 1),
        GGMLQuantizationType.I16: (1, 2),
        GGMLQuantizationType.I32: (1, 4),
        GGMLQuantizationType.I64: (1, 8),
    }
    if dtype not in quant_info:
        raise ValueError(f"Unknown quantization type: {dtype}")
    return quant_info[dtype]


def calc_tensor_size(shape: tuple[int, ...], dtype: GGMLQuantizationType) -> int:
    """Calculate tensor size in bytes."""
    block_size, bytes_per_block = get_quant_block_size(dtype)
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    n_blocks = (n_elements + block_size - 1) // block_size
    return n_blocks * bytes_per_block


class GGUFModel:
    """Parsed GGUF model with expert offset calculation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.reader = GGUFReader(str(self.path))
        self.tensors: dict[str, TensorInfo] = {}
        self._parse_tensors()
        self._extract_model_info()

    def _parse_tensors(self):
        """Parse all tensor metadata."""
        for tensor in self.reader.tensors:
            shape = tuple(tensor.shape)
            dtype = tensor.tensor_type
            offset = tensor.data_offset
            n_elements = int(np.prod(shape))
            size = calc_tensor_size(shape, dtype)

            self.tensors[tensor.name] = TensorInfo(
                name=tensor.name,
                shape=shape,
                dtype=dtype,
                offset=offset,
                size=size,
                n_elements=n_elements,
            )

    def _extract_model_info(self):
        """Extract model architecture info."""
        self.n_layers = 0
        self.n_experts = 0
        self.expert_tensors: dict[int, dict[str, TensorInfo]] = {}

        for name, info in self.tensors.items():
            if 'ffn_down_exps' in name or 'ffn_gate_exps' in name or 'ffn_up_exps' in name:
                parts = name.split('.')
                layer_idx = int(parts[1])
                self.n_layers = max(self.n_layers, layer_idx + 1)

                if len(info.shape) >= 3:
                    self.n_experts = max(self.n_experts, info.shape[-1])

                if layer_idx not in self.expert_tensors:
                    self.expert_tensors[layer_idx] = {}

                if 'down' in name:
                    self.expert_tensors[layer_idx]['down'] = info
                elif 'gate' in name:
                    self.expert_tensors[layer_idx]['gate'] = info
                elif 'up' in name:
                    self.expert_tensors[layer_idx]['up'] = info

    def get_expert_offset(self, layer: int, expert_id: int, tensor_type: str) -> ExpertInfo:
        """
        Calculate byte offset for a single expert within merged tensor.

        Merged tensors have shape [dim1, dim2, n_experts].
        Expert i's data is at offset: base_offset + i * expert_stride
        """
        if layer not in self.expert_tensors:
            raise ValueError(f"Layer {layer} not found")
        if tensor_type not in self.expert_tensors[layer]:
            raise ValueError(f"Tensor type {tensor_type} not found in layer {layer}")

        tensor = self.expert_tensors[layer][tensor_type]

        if expert_id >= tensor.shape[-1]:
            raise ValueError(f"Expert {expert_id} out of range (max {tensor.shape[-1] - 1})")

        # Calculate expert size
        # Shape is [dim1, dim2, n_experts], expert slice is [dim1, dim2, 1]
        expert_shape = tensor.shape[:-1]  # Remove n_experts dimension
        expert_elements = int(np.prod(expert_shape))

        block_size, bytes_per_block = get_quant_block_size(tensor.dtype)
        expert_blocks = (expert_elements + block_size - 1) // block_size
        expert_size = expert_blocks * bytes_per_block

        # Expert offset within tensor
        expert_offset = expert_id * expert_size
        absolute_offset = tensor.offset + expert_offset

        return ExpertInfo(
            layer=layer,
            expert_id=expert_id,
            tensor_type=tensor_type,
            offset=absolute_offset,
            size=expert_size,
        )

    def get_layer_experts(self, layer: int, expert_ids: list[int]) -> list[ExpertInfo]:
        """Get all expert infos for a layer (down, gate, up for each expert)."""
        experts = []
        for eid in expert_ids:
            for ttype in ['down', 'gate', 'up']:
                if ttype in self.expert_tensors.get(layer, {}):
                    experts.append(self.get_expert_offset(layer, eid, ttype))
        return experts

    def summary(self) -> str:
        """Print model summary."""
        lines = [
            f"Model: {self.path.name}",
            f"Layers: {self.n_layers}",
            f"Experts per layer: {self.n_experts}",
            "",
            "Expert tensor sizes (layer 0):",
        ]

        if 0 in self.expert_tensors:
            for ttype, tensor in self.expert_tensors[0].items():
                expert_info = self.get_expert_offset(0, 0, ttype)
                lines.append(
                    f"  {ttype}: {tensor.shape} {tensor.dtype.name} "
                    f"(total: {tensor.size/1e6:.1f} MB, per-expert: {expert_info.size/1e6:.3f} MB)"
                )

        total_expert_size = sum(
            t.size for layer in self.expert_tensors.values() for t in layer.values()
        )
        lines.append(f"\nTotal expert weights: {total_expert_size/1e9:.2f} GB")

        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gguf_parser.py <model.gguf>")
        sys.exit(1)

    model = GGUFModel(sys.argv[1])
    print(model.summary())

    # Test expert offset calculation
    print("\nSample expert offsets (layer 0):")
    for eid in [0, 1, 255, 511]:
        if eid < model.n_experts:
            info = model.get_expert_offset(0, eid, 'down')
            print(f"  Expert {eid}: offset={info.offset:,}, size={info.size:,}")
