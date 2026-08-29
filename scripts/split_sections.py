#!/usr/bin/env python3
"""Split a file at declared line ranges into `foo/<name>.rs`, keeping every path working.

`split_module.py` handles the easy half - a file already divided by inline `mod` blocks. What
it cannot touch is the other half: ten thousand lines of top-level items with nothing but a
banner comment between subjects. This moves those ranges into submodules.

Unlike the inline case this is NOT path-neutral, so two things are done to keep every caller
working:

  * the parent re-exports each section with `pub use <name>::*;`, so `...::foo::thing` still
    resolves wherever `thing` ended up;
  * each section opens with `use super::*;`, so items that referred to their neighbours by bare
    name still find them.

One thing it cannot fix and you should expect: an item DEFINED in the file shadowed anything
of the same name arriving through a `use super::*`. Once it moves into a section and comes back
through `pub use <section>::*`, it is a glob competing with a glob, and Rust calls that
ambiguous rather than picking one. The fix is to name the one you mean at the call site -
`super::models::model_capabilities(..)` - which is clearer than the shadowing was.

What it cannot fix by itself is visibility: an item that was private was visible to the whole
file and is now visible only inside its section. Every moved item that had no visibility
keyword is therefore promoted to `pub(super)` - which restores exactly the reach it had, and
no more. The compiler is the check on whatever this misses.

    python3 scripts/split_sections.py src/tensor/native/quant_cpu/mod.rs \\
        scale:29-594 block:595-2905 dispatch:2906-3255 --apply

Ranges are 1-based and inclusive, given against the CURRENT file, and must not overlap.
"""

import argparse
import pathlib
import re
import subprocess
import sys

# A top-level item with no visibility keyword: private today, `pub(super)` after the move.
PRIVATE = re.compile(
    r"^(?!\s)(?!pub\b)(?!use\b)(?!mod\b)(?!#)(?!//)(?!/\*)"
    r"((?:unsafe |extern |async |const |default )*)"
    r"(fn |struct |enum |trait |type |static |union |const )",
    re.M,
)


# A struct field with no visibility keyword. Same problem as an item: it was readable across
# the whole file and is now readable only inside its section.
FIELD = re.compile(r"^(    )(?!pub\b)(?!#)(?!//)([a-z_][a-z_0-9]*): ", re.M)


def promote_fields(body: str) -> tuple[str, int]:
    """Widen the fields of structs this range defines, and only those.

    Bounded on purpose: the indentation-only rule would also hit a struct LITERAL or a match
    arm, so each `struct X {` body is located first and the rewrite applied inside it.
    """
    out, pos, n = [], 0, 0
    for m in re.finditer(r"^(?:pub(?:\([a-z]+\))? )?struct [A-Za-z_][A-Za-z_0-9]*(?:<[^>]*>)? \{$",
                         body, re.M):
        end = body.find("\n}\n", m.end())
        if end == -1:
            continue
        out.append(body[pos:m.end()])
        inner, k = FIELD.subn(r"\1pub(super) \2: ", body[m.end():end])
        out.append(inner)
        n += k
        pos = end
    out.append(body[pos:])
    return "".join(out), n


# A method inside an `impl` block the range carries, and a tuple struct's fields. Both were
# reachable across the file and are not across the split - the same rule as a top-level item,
# in the two places the top-level rule cannot see.
METHOD = re.compile(r"^    (?!pub\b)((?:unsafe |async |const |extern )*)(fn )", re.M)
TUPLE = re.compile(
    r"^((?:pub(?:\([a-z]+\))? )?struct [A-Za-z_][A-Za-z_0-9]*(?:<[^>]*>)?)\(([^)]*)\);", re.M)


# Blocks where a visibility qualifier is ILLEGAL, not merely unnecessary: a trait declaration
# and a trait IMPLEMENTATION both take their visibility from the trait. Missing the second one
# cost a build: the first version excluded `trait X {` only, and every `impl GgmlType for
# BlockQ4_0` method was rewritten into something that does not compile.
TRAIT = re.compile(
    r"^((?:pub(?:\([a-z]+\))? )?(?:unsafe )?trait [^\n{]*\{"
    r"|(?:unsafe )?impl(?:<[^>]*>)? [^\n{]* for [^\n{]*\{)", re.M)


def promote_methods(body: str) -> tuple[str, int]:
    """Widen methods in `impl` blocks - but never in a `trait` body.

    A trait method takes its visibility from the trait; writing one on it is an error, not a
    no-op. The first version of this rule turned every `GgmlType` method into
    `pub(super) fn` and the file stopped compiling for a reason unrelated to the split.
    """
    spans = []
    for m in TRAIT.finditer(body):
        # `unsafe impl Send for X {}` closes on its own line. Without this the search for the
        # next `\n}\n` runs past it and swallows the following block, so a scan reports
        # illegal qualifiers in code that has none.
        if body[m.start():m.end()].rstrip().endswith("{}"):
            continue
        line_end = body.find("\n", m.end())
        if line_end != -1 and body[m.end():line_end].strip() == "}":
            continue
        end = body.find("\n}\n", m.end())
        if end != -1:
            spans.append((m.end(), end))

    def in_trait(i):
        return any(a <= i < b for a, b in spans)

    out, pos, n = [], 0, 0
    for m in METHOD.finditer(body):
        if in_trait(m.start()):
            continue
        out.append(body[pos:m.start()])
        out.append(f"    pub(super) {m.group(1)}{m.group(2)}")
        pos = m.end()
        n += 1
    out.append(body[pos:])
    return "".join(out), n


def promote_tuple_fields(body: str) -> tuple[str, int]:
    n = 0

    def rep(m):
        nonlocal n
        fields = [f.strip() for f in m.group(2).split(",") if f.strip()]
        if not fields:
            return m.group(0)
        n += sum(1 for f in fields if not f.startswith("pub"))
        fields = [f if f.startswith("pub") else f"pub(super) {f}" for f in fields]
        return f"{m.group(1)}({', '.join(fields)});"

    return TUPLE.sub(rep, body), n


def promote(body: str) -> tuple[str, int]:
    n = 0

    def rep(m):
        nonlocal n
        n += 1
        return f"pub(super) {m.group(1)}{m.group(2)}"

    return PRIVATE.sub(rep, body), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("sections", nargs="+", metavar="name:first-last")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.file)
    lines = src.read_text().split("\n")

    def doc_start(anchor: int) -> int:
        """First line of the doc/attribute block above `anchor`, 1-based.

        Worth having rather than eyeballing: reading the number off a grep gives the item's own
        line, and starting a section there leaves its doc comment behind. Backing up by hand is
        where the off-by-one lives - a section that starts one line early swallows the previous
        item's closing brace, and the file stops parsing for a reason that looks nothing like
        the cause.
        """
        j = anchor
        while j > 1 and lines[j - 2].lstrip().startswith(("///", "//", "#[")):
            j -= 1
        return j

    cuts = []
    anchors = []
    for spec in args.sections:
        name, _, rng = spec.partition(":")
        if rng.startswith("@"):
            anchors.append((name, doc_start(int(rng[1:]))))
            continue
        a, _, b = rng.partition("-")
        cuts.append((name, int(a), int(b)))
    if anchors:
        # Anchor form: each section runs to the line before the next one starts.
        anchors.sort(key=lambda c: c[1])
        for i, (name, start) in enumerate(anchors):
            end = anchors[i + 1][1] - 1 if i + 1 < len(anchors) else len(lines)
            cuts.append((name, start, end))
    cuts.sort(key=lambda c: c[1])
    for (n1, _, e1), (n2, s2, _) in zip(cuts, cuts[1:]):
        if s2 <= e1:
            print(f"{n1} and {n2} overlap", file=sys.stderr)
            return 2

    out_dir = src.parent if src.name == "mod.rs" else src.with_suffix("")
    print(f"{src}  ({len(lines)} lines)")
    bodies = {}
    for name, first, last in cuts:
        body = "\n".join(lines[first - 1:last])
        body, promoted = promote(body)
        body, fields = promote_fields(body)
        body, methods = promote_methods(body)
        body, tuples = promote_tuple_fields(body)
        promoted += fields + methods + tuples
        bodies[name] = body
        print(f"  {name:<16} {last - first + 1:>6} lines   {promoted} items promoted to pub(super)")
    kept = len(lines) - sum(b - a + 1 for _, a, b in cuts)
    print(f"  {'mod.rs':<16} {kept:>6} lines remain")
    if not args.apply:
        print("  (dry run - pass --apply)")
        return 0

    # A `mod X;` inside a moved range would look for `<section>/X.rs` instead of `X.rs`, so it
    # stays with the parent. Found the hard way: three `mod tests;` declarations travelled into
    # sections and their files became undeclared, which is invisible until something needs them.
    out_dir.mkdir(exist_ok=True)
    lifted_uses = []
    # A `use` declaration is a private binding in the module that writes it, so a child cannot
    # see the parent's - and once one MOVES into a child, the parent loses it. Imports belong
    # to mod.rs whichever range they happened to fall in.
    imports = re.compile(r"^use [^\n]*;\n", re.M)
    for name in list(bodies):
        found = imports.findall(bodies[name])
        if found:
            bodies[name] = imports.sub("", bodies[name])
            lifted_uses.extend(found)
    stray = re.compile(
        # `#[cfg(all(test, feature = "cuda"))]` nests parentheses, so a lazy `[^)]*`
        # stops at the first one and leaves the attribute orphaned behind the
        # declaration it belongs to.
        r"^((?:#\[[^\n]*\]\n)*)(?:pub(?:\([a-z]+\))? )?mod ([a-z_0-9]+);\n", re.M)
    lifted = []
    for name in list(bodies):
        kept = []
        pos = 0
        for m in stray.finditer(bodies[name]):
            if not (out_dir / f"{m.group(2)}.rs").exists() and not (out_dir / m.group(2)).is_dir():
                continue
            kept.append(bodies[name][pos:m.start()])
            lifted.append(m.group(0))
            pos = m.end()
        if lifted and pos:
            kept.append(bodies[name][pos:])
            bodies[name] = "".join(kept)
    for name, body in bodies.items():
        (out_dir / f"{name}.rs").write_text(
            "//! Split out of the parent module; see its header for what this file is part of.\n"
            "//!\n"
            "//! `use super::*` keeps the names its items referred to before the split in reach.\n\n"
            "use super::*;\n\n" + body.strip("\n") + "\n"
        )

    keep, decls = [], []
    for i, ln in enumerate(lines, 1):
        owner = next((n for n, a, b in cuts if a <= i <= b), None)
        if owner is None:
            keep.append(ln)
        elif i == next(a for n, a, _ in cuts if n == owner):
            decls.append(owner)
            keep.append(f"__SECTION__{owner}")

    text = "\n".join(keep)
    for name in decls:
        text = text.replace(f"__SECTION__{name}", f"mod {name};\npub use {name}::*;")
    if lifted_uses:
        text = "".join(dict.fromkeys(lifted_uses)) + text
    # An inner doc comment is only legal at the very top of a file, and the declarations this
    # inserts go where the range began - which can be above the module's own header.
    if "//!" in text:
        lines = text.split("\n")
        head = [l for l in lines if l.startswith("//!")]
        if head and not lines[0].startswith("//!"):
            text = "\n".join(head + [""] + [l for l in lines if not l.startswith("//!")])
    if lifted:
        text = text.rstrip() + "\n\n" + "".join(lifted)
    (out_dir / "mod.rs" if src.name != "mod.rs" else src).write_text(text)
    if src.name != "mod.rs":
        subprocess.run(["git", "rm", "-q", "--cached", str(src)], check=False)
        src.unlink()
    subprocess.run(["git", "add", str(out_dir)], check=False)
    print(f"  -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
