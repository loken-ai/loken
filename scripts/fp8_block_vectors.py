#!/usr/bin/env python3
"""Block-scaled fp8 weights and what torch makes of them, for the readers' tests.

Each tensor is stored as a checkpoint that quantises its weights to fp8 e4m3 stores it: the
bytes as `<name>.weight` and one f32 scale per 128 x 128 tile as `<name>.weight_scale_inv`, the
value being the byte's number times the scale. `<name>.expected` holds the product torch
computes, f32, to compare value by value. A last row or column of tiles is partial in each
tensor: 576 rows are 4.5 tiles, the other tensor is partial both ways.

Usage: fp8_block_vectors.py OUT_DIR
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

BLOCK = 128
SHAPES = {"rows": (576, 64), "both": (129, 130)}


def tensors(name, shape, seed):
    g = torch.Generator().manual_seed(seed)
    rows, cols = shape
    w = torch.randn(rows, cols, generator=g)
    tiles = ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK)
    # A scale that is not a power of two, so that reading it as one is seen.
    scale = torch.exp(torch.randn(tiles, generator=g) * 0.5)
    expand = scale.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    q = (w / expand).to(torch.float8_e4m3fn)
    expected = q.to(torch.float32) * expand
    return {
        f"{name}.weight": q.contiguous(),
        f"{name}.weight_scale_inv": scale.to(torch.float32).contiguous(),
        f"{name}.expected": expected.contiguous(),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    state = {}
    for i, (name, shape) in enumerate(SHAPES.items()):
        state.update(tensors(name, shape, 1000 + i))
    save_file(state, str(out / "blocks.safetensors"), metadata={"format": "pt"})
    print(f"{sum(t.numel() * t.element_size() for t in state.values())} bytes")


if __name__ == "__main__":
    main()
