#!/usr/bin/env python3
"""Turn `foo.rs`, with inline `mod bar { ... }` blocks, into `foo/mod.rs` + `foo/bar.rs`.

A file that needs a banner comment to announce "here starts something else" is a directory.
The largest files here are already divided that way - by inline modules and by section
banners - so the seams exist; they are just not load-bearing.

Extracting an inline module is the one rearrangement that changes NO paths. Inside
`foo/bar.rs`, `super::` still names `foo`, exactly as it did inside `mod bar { ... }`, and
`crate::...::foo::bar` still resolves. So this can be applied to a ten-thousand-line file
without touching a single caller, which is the difference between a split that is worth doing
and one that is a rewrite in disguise.

    python3 scripts/split_module.py src/tensor/native/quant_cpu.rs            # what it would do
    python3 scripts/split_module.py src/tensor/native/quant_cpu.rs --apply
    python3 scripts/split_module.py src/tensor/native/quant_cpu.rs --apply --min-lines 200

Modules below `--min-lines` stay where they are: a twelve-line helper module in its own file
is not clearer, it is just further away.
"""

import argparse
import pathlib
import re
import subprocess
import sys

# A top-level module block. Indented ones belong to something else and are left alone.
BLOCK = re.compile(r"^((?:pub(?:\([a-z:]+\))? )?mod ([a-z_0-9]+)) \{$", re.M)


def block_end(text: str, open_brace: int) -> int:
    """Index just past the line `}` in column zero that closes a top-level block.

    This code is rustfmt-formatted, so a top-level item's closing brace is the only `}` that
    ever starts a line at column zero - which makes it a far more reliable end marker than
    counting braces. Counting is what the first version did, and it mis-cut `cuda.rs` in the
    middle of a function: some token inside a test body drove the depth to zero early, the
    module was written out truncated, and the remainder stayed behind. `cargo fmt` caught it,
    but only because the result happened not to parse; a cut that balanced by accident would
    have moved code silently.
    """
    nl = text.find("\n", open_brace)
    if nl != -1:
        close = text.find("\n}\n", nl)
        if close != -1:
            # Just past the `}` itself, not past its newline: the caller reads the body as
            # text[open+1:end-1], so a byte too far keeps the closing brace inside it and
            # every extracted file ends up with one delimiter too many.
            return close + 2
    return _counted_block_end(text, open_brace)


def _counted_block_end(text: str, open_brace: int) -> int:
    """Brace counting, kept for a block that is not formatted the usual way."""
    i, depth, n = open_brace, 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i = text.find("\n", i)
            if i == -1:
                return n
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == "r" and i + 1 < n and text[i + 1] in '#"':
            j = i + 1
            hashes = 0
            while j < n and text[j] == "#":
                hashes += 1
                j += 1
            if j < n and text[j] == '"':
                close = '"' + "#" * hashes
                k = text.find(close, j + 1)
                i = n if k == -1 else k + len(close)
                continue
        if c in "\"'":
            quote, j = c, i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote:
                    break
                if quote == "'" and text[j] == "\n":
                    break  # a lifetime, not a char literal
                j += 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def preceding_attributes(text: str, start: int) -> tuple[str, int]:
    """The `#[...]` lines and doc comments directly above a declaration travel with it."""
    lines = text[:start].split("\n")
    keep = 0
    for ln in reversed(lines[:-1]):
        s = ln.strip()
        if s.startswith("#[") or s.startswith("///") or s.startswith("//"):
            keep += 1
        else:
            break
    if keep == 0:
        return "", start
    head = "\n".join(lines[len(lines) - 1 - keep:-1])
    return head, start - len(head) - 1


def dedent(body: str) -> str:
    out = []
    for ln in body.split("\n"):
        out.append(ln[4:] if ln.startswith("    ") else ln)
    return "\n".join(out).strip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-lines", type=int, default=120)
    args = ap.parse_args()

    src = pathlib.Path(args.file)
    if not src.is_file() or src.suffix != ".rs":
        print(f"{src}: not a Rust file", file=sys.stderr)
        return 2
    text = src.read_text()

    found, refused = [], []
    for m in BLOCK.finditer(text):
        open_brace = text.index("{", m.start())
        end = block_end(text, open_brace)
        try:
            counted = _counted_block_end(text, open_brace)
        except ValueError:
            counted = end
        if abs(counted - end) > 2:
            # The two rules disagree, which means something inside the block looks like a
            # closing brace to one of them - `cuda.rs` embeds CUDA source whose `}` sits in
            # column zero inside a string. Refusing is the only safe answer: a wrong cut here
            # moves code silently, and only luck decides whether the result still parses.
            refused.append((m.group(2), abs(counted - end)))
            continue
        body = text[open_brace + 1:end - 1]
        if body.count("\n") < args.min_lines:
            continue
        attrs, decl_start = preceding_attributes(text, m.start())
        found.append((decl_start, end, m.group(1), m.group(2), attrs, body))

    for name, gap in refused:
        print(f"{src}: REFUSING mod {name} - the two end rules disagree by {gap} bytes, so "
              f"something inside it reads as a closing brace to one of them")
    if not found:
        print(f"{src}: nothing to extract")
        return 0

    total = text.count("\n")
    print(f"{src}  ({total} lines)")
    for _, _, decl, name, attrs, body in found:
        print(f"  {name:<24} {body.count(chr(10)):>6} lines"
              + (f"   [{attrs.strip().splitlines()[0][:40]}]" if attrs.strip().startswith("#[") else ""))
    kept = total - sum(b.count("\n") for *_, b in found)
    print(f"  {'mod.rs':<24} {kept:>6} lines remain")
    if not args.apply:
        print("  (dry run - pass --apply)")
        return 0

    # A `mod.rs` already IS its directory's root; `foo.rs` becomes `foo/mod.rs`.
    is_mod = src.name == "mod.rs"
    out_dir = src.parent if is_mod else src.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    for start, end, decl, name, attrs, body in reversed(found):
        (out_dir / f"{name}.rs").write_text(dedent(body))
        replacement = (attrs + "\n" if attrs else "") + f"{decl};"
        text = text[:start] + replacement + text[end:]

    (out_dir / "mod.rs").write_text(text)
    if not is_mod:
        subprocess.run(["git", "rm", "-q", "--cached", str(src)], check=False)
        src.unlink()
    subprocess.run(["git", "add", str(out_dir)], check=False)
    print(f"  -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
