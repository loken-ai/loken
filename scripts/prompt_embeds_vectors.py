#!/usr/bin/env python3
"""Prompt embeddings files and the values they hold, for the `prompt_embeds` readers' tests.

Writes into OUT_DIR:
- `torch_<dtype>_<case>.pt`: `torch.save` of one tensor into a buffer, as a vLLM client sends
  it, for float32, float16 and bfloat16, in two and three dimensions (a batch of one), contiguous
  and not (transposed, sliced at an offset, strided columns); one more saved to a path, whose
  archive is named after the file;
- `loken_<dtype>.bin`: the same values in loken's format, written by `encode_loken` below, the
  reference encoder `docs/API.md` describes;
- `manifest.json`: per file, the model width it is read with and the values row after row, as
  torch converts them to float32.

Usage: prompt_embeds_vectors.py OUT_DIR
"""

import io
import json
import struct
import sys
import zlib
from pathlib import Path

import torch

MAGIC = b"\x00\x00\xc0\xffLKEM"
VERSION = 1
DTYPE_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
DTYPE_NAMES = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}


def encode_loken(t: torch.Tensor) -> bytes:
    """`t`, (rows, width), in loken's format: header, values row after row, CRC-32 of both."""
    assert sys.byteorder == "little"
    rows, width = t.shape
    t = t.contiguous()
    raw = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()
    header = MAGIC + struct.pack("<HBBIII", VERSION, DTYPE_CODES[t.dtype], 0, rows, width, 0)
    body = header + raw
    return body + struct.pack("<I", zlib.crc32(body))


def saved(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def cases(dtype: torch.dtype):
    """(name, tensor) pairs, each a prompt of 3 rows of width 4."""
    g = torch.Generator().manual_seed(0)
    base = torch.randn(5, 8, generator=g).to(dtype)
    yield "2d", base[:3, :4].clone()
    yield "3d", base[:3, :4].clone().unsqueeze(0)
    yield "transposed", base[:4, :3].clone().t()
    yield "offset", base.clone()[1:4, 4:8]
    yield "strided", base[:3].clone()[:, ::2]
    yield "transposed_3d", base[:4, :3].clone().t().unsqueeze(0)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def record(file: str, t: torch.Tensor) -> None:
        flat = t.squeeze(0) if t.dim() == 3 else t
        manifest.append(
            {
                "file": file,
                "width": flat.shape[1],
                "values": flat.float().reshape(-1).tolist(),
            }
        )

    for dtype, name in DTYPE_NAMES.items():
        for case, t in cases(dtype):
            assert t.is_contiguous() == (case in ("2d", "3d")), case
            file = f"torch_{name}_{case}.pt"
            (out / file).write_bytes(saved(t))
            record(file, t)
        t = dict(cases(dtype))["2d"]
        file = f"loken_{name}.bin"
        (out / file).write_bytes(encode_loken(t))
        record(file, t)

    t = dict(cases(torch.float32))["2d"]
    file = "torch_f32_to_path.pt"
    torch.save(t, out / file)
    record(file, t)

    (out / "manifest.json").write_text(json.dumps({"cases": manifest}, indent=1) + "\n")
    print(f"{len(manifest)} files in {out}")


if __name__ == "__main__":
    main()
