#!/usr/bin/env python3
"""Remove public functions whose name appears nowhere in the crate but their own definition.

This is one crate with no workspace and no library consumer, so `src/` is the whole world: a
`pub fn` that nothing in it names cannot be reached. `pub` is exactly what stops the compiler
from saying so, which is why a scan has to.

Deliberately conservative. A function is left alone when it is:
  - a test or bench (its name is never referenced by design),
  - `extern`/`#[no_mangle]` (the caller is outside Rust),
  - named in any comment anywhere (the count includes prose, so a mention protects it).

The scan is re-run after each pass: removing a function can leave its callee unreferenced in
turn, and one pass would stop at the first layer.
"""

import collections
import pathlib
import re
import sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DEF = re.compile(r"^(\s*)pub(?:\([a-z():& ]+\))? fn (\w+)")
KEEP = {"new", "forward", "default", "from", "fmt", "clone", "drop", "next", "len",
        "is_empty", "from_str", "into", "main"}


def scan(root: pathlib.Path):
    counts, defs = collections.Counter(), collections.defaultdict(list)
    for p in sorted(root.rglob("*.rs")):
        lines = p.read_text().split("\n")
        counts.update(IDENT.findall("\n".join(lines)))
        for i, ln in enumerate(lines):
            m = DEF.match(ln)
            if not m:
                continue
            attrs = "".join(lines[max(0, i - 5):i])
            if any(k in attrs for k in ("#[test]", "#[bench]", "no_mangle", "extern")):
                continue
            defs[m.group(2)].append((p, i, m.group(1)))
    return [(p, i, ind, n) for n, sites in defs.items()
            if n not in KEEP and counts[n] == len(sites)
            for (p, i, ind) in sites]


def extent(lines, i, indent):
    """The whole item: its doc block and attributes above, its body to the matching brace."""
    top = i
    while top > 0 and (lines[top - 1].strip().startswith("///")
                       or lines[top - 1].strip().startswith("#[")):
        top -= 1
    depth, j, seen = 0, i, False
    while j < len(lines):
        for ch in lines[j]:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
        if seen and depth == 0:
            break
        j += 1
    else:
        return None                      # unbalanced: refuse rather than guess
    if lines[j].rstrip() != f"{indent}}}":
        return None                      # not a plain block end - leave it to a human
    while j + 1 < len(lines) and lines[j + 1].strip() == "":
        j += 1
    return top, j


def main() -> int:
    root = pathlib.Path("src")
    total, rounds = 0, 0
    while True:
        found = scan(root)
        if not found:
            break
        rounds += 1
        by_file = collections.defaultdict(list)
        for p, i, ind, n in found:
            by_file[p].append((i, ind, n))
        removed = 0
        for p, items in by_file.items():
            lines = p.read_text().split("\n")
            for i, ind, n in sorted(items, reverse=True):   # bottom-up keeps indices valid
                span = extent(lines, i, ind)
                if span is None:
                    print(f"  refuse {p}:{i + 1} {n} (limites incertaines)")
                    continue
                del lines[span[0]:span[1] + 1]
                removed += 1
            p.write_text("\n".join(lines))
        print(f"passe {rounds}: {removed} retirees")
        total += removed
        if removed == 0:
            break
    print(f"total {total} fonctions publiques inatteignables, en {rounds} passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
