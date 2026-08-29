#!/usr/bin/env python3
"""Split one enormous `impl Type { ... }` into several files, each with its own `impl Type`.

The other two splitters stop at the same wall: `llm_engine` is ten thousand lines of which
seven thousand four hundred are a single inherent impl, and neither an inline `mod` nor a
top-level item range can cut into it. Rust does allow inherent impls to be spread across
modules of the same crate, so the block CAN be divided - the only question is doing it without
changing what resolves.

Two things need care, and both are handled here:

  * A method with no visibility keyword is visible in the module holding its `impl` and in that
    module's descendants - so once a method moves into a child, the parent can no longer call
    it. Every moved method that had no keyword is promoted to `pub(super)`, which restores
    exactly the reach it had.
  * The methods left behind still call the moved ones and vice versa; `use super::*;` at the
    top of each new file covers the free functions and types, and `pub(super)` covers the
    methods, so no call site changes.

    python3 scripts/split_impl.py src/inference/engine/llm_engine/mod.rs LlmEngine \\
        generate:@2100 prefill:@3400 --apply

Anchors are line numbers INSIDE the impl; each section runs to the line before the next, and
the last runs to the end of the impl body.
"""

import argparse
import pathlib
import re
import subprocess
import sys

# A method inside an impl: four spaces, no visibility keyword.
PRIVATE_METHOD = re.compile(
    r"^    (?!pub\b)((?:pub\([a-z]+\) )?(?:unsafe |async |const |extern )*)(fn )", re.M
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("type_name")
    ap.add_argument("sections", nargs="+", metavar="name:@line")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.file)
    lines = src.read_text().split("\n")

    # Find the impl block: its header, and the `}` in column zero that closes it.
    header = re.compile(rf"^impl(?:<[^>]*>)? {re.escape(args.type_name)}\b.*\{{$")
    start = next((i for i, ln in enumerate(lines, 1) if header.match(ln)), None)
    if start is None:
        print(f"{src}: no `impl {args.type_name}` at top level", file=sys.stderr)
        return 2
    end = next((i for i, ln in enumerate(lines[start:], start + 1) if ln == "}"), None)
    if end is None:
        print(f"{src}: the impl block does not close", file=sys.stderr)
        return 2
    print(f"{src}: impl {args.type_name} spans {start}..{end} ({end - start} lines)")

    def method_start(anchor: int) -> int:
        j = anchor
        while j > start + 1 and lines[j - 2].lstrip().startswith(("///", "//", "#[")):
            j -= 1
        return j

    cuts = []
    for spec in args.sections:
        name, _, rng = spec.partition(":")
        if not rng.startswith("@"):
            print(f"{spec}: expected name:@line", file=sys.stderr)
            return 2
        cuts.append((name, method_start(int(rng[1:]))))
    cuts.sort(key=lambda c: c[1])
    if cuts[0][1] <= start:
        print("a section starts before the impl does", file=sys.stderr)
        return 2

    ranges = []
    for i, (name, first) in enumerate(cuts):
        last = cuts[i + 1][1] - 1 if i + 1 < len(cuts) else end - 1
        ranges.append((name, first, last))

    total_promoted = 0
    bodies = {}
    for name, first, last in ranges:
        body = "\n".join(lines[first - 1:last])
        body, n = PRIVATE_METHOD.subn(r"    pub(super) \1\2", body)
        total_promoted += n
        bodies[name] = body
        print(f"  {name:<14} {last - first + 1:>6} lines   {n} methods promoted to pub(super)")
    kept = len(lines) - sum(b - a + 1 for _, a, b in ranges)
    print(f"  {'mod.rs':<14} {kept:>6} lines remain")
    if not args.apply:
        print("  (dry run - pass --apply)")
        return 0

    out_dir = src.parent if src.name == "mod.rs" else src.with_suffix("")
    out_dir.mkdir(exist_ok=True)
    for name, body in bodies.items():
        (out_dir / f"{name}.rs").write_text(
            f"//! Part of `impl {args.type_name}`, split out of the parent module.\n"
            "//!\n"
            "//! Rust lets one inherent impl live in several modules of a crate, so this is the\n"
            "//! same impl - only the file changed. Methods that were private are `pub(super)`\n"
            "//! here, which is the reach they had when they sat beside their callers.\n\n"
            "use super::*;\n\n"
            f"impl {args.type_name} {{\n{body.strip(chr(10))}\n}}\n"
        )

    keep = []
    for i, ln in enumerate(lines, 1):
        owner = next((n for n, a, b in ranges if a <= i <= b), None)
        if owner is None:
            keep.append(ln)
        elif i == next(a for n, a, _ in ranges if n == owner):
            keep.append(f"__IMPLSECTION__{owner}")
    text = "\n".join(keep)
    # The declarations go after the impl closes, not inside it.
    for name, _, _ in ranges:
        text = text.replace(f"__IMPLSECTION__{name}\n", "")
        text = text.replace(f"__IMPLSECTION__{name}", "")
    text = text.rstrip() + "\n\n" + "".join(f"mod {n};\n" for n, _, _ in ranges)
    (out_dir / "mod.rs" if src.name != "mod.rs" else src).write_text(text)
    if src.name != "mod.rs":
        subprocess.run(["git", "rm", "-q", "--cached", str(src)], check=False)
        src.unlink()
    subprocess.run(["git", "add", str(out_dir)], check=False)
    print(f"  -> {out_dir}/  ({total_promoted} methods promoted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
