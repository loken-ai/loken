#!/usr/bin/env python3
"""Render the graphs the docs embed from their `.dot` sources.

Run from the repository root: `python3 scripts/figures.py`. It needs graphviz (`dot`); a graph
whose source is unchanged is rewritten with the same bytes, and a missing `dot` leaves the
rendered files as they are.

Every figure has a source in the tree, so a diff shows what changed rather than that some bytes did.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"


def render_dot(name):
    src = IMG / f"{name}.dot"
    out = IMG / f"{name}.svg"
    try:
        subprocess.run(
            ["dot", "-Tsvg", "-o", str(out), str(src)], check=True, capture_output=True
        )
    except FileNotFoundError:
        print(f"graphviz absent, {out.name} left as it is", file=sys.stderr)
        return False
    return True


def main():
    for name in ("cluster-topology", "marlin-pipeline"):
        if (IMG / f"{name}.dot").exists():
            render_dot(name)


if __name__ == "__main__":
    main()
