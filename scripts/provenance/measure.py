#!/usr/bin/env python3
"""Measure how much of each source file is still somebody else's, against the real checkouts.

Provenance is measured or it is guessed. Splitting a file into fifteen and rewriting two
functions in each FEELS like rewriting it, and a claim made from that feeling has been off by
sixty points against a snippet scanner. This is the instrument that can contradict it.

Method: build one index of every line in the reference corpus, then for each of our files
report the share of its own lines that appear anywhere in that index.

The corpus is a directory holding read-only checkouts of the projects listed in `CORPUS`. Say
where yours are with `LOKEN_PROVENANCE_REFS=/path/to/checkouts`; there is no default, because a
path that is true on one machine does not belong in a file everyone gets.

Two figures, because they answer different questions and only the second is interesting:

  all    every line, declarations included
  body   with declarations, signatures and attributes removed

A file's struct fields ARE the format it decodes and its trait signatures are the API; both
match upstream by necessity and neither can be rewritten. In a small file they dominate - q8_0
is 38 useful lines of which 16 are declaration - so the `all` figure says mostly how much
boilerplate a file has. `body` is what is left to actually rewrite.

Both are FLOORS. Neither sees a line that was reformatted, and both count a line the format
dictates the same as a free choice. Cross-check against a snippet scanner, which matches
contiguous regions rather than exact lines and will disagree in both directions: it reported
64% for a file this measured at 0%, because winnowed fingerprints match a shape that no single
line preserves. Read the code before concluding.

    python3 scripts/provenance/measure.py            # the ranked table
    python3 scripts/provenance/measure.py --min 40   # only what is still substantially theirs
"""

import argparse
import collections
import pathlib
import re
import os
import sys

CORPUS_NAMES = ["candle", "llama.cpp", "vllm", "attention.rs", "vllm.rs"]
CORPUS = CORPUS_NAMES
_refs = os.environ.get("LOKEN_PROVENANCE_REFS")
if not _refs:
    sys.exit(
        "set LOKEN_PROVENANCE_REFS to the directory holding read-only checkouts of "
        + ", ".join(CORPUS_NAMES)
    )
REFS = pathlib.Path(_refs)

SUFFIXES = {".rs", ".cu", ".cuh", ".h", ".hpp", ".c", ".cpp"}
MIN_LEN = 10

# A line that declares rather than computes. These match upstream because the format and the
# trait say so, and separating them is what makes the remaining figure mean something.
DECLARATION = (
    "#[", "pub struct", "struct ", "pub enum", "enum ", "impl ", "pub(crate) ", "pub fn ",
    "fn ", "const ", "pub const ", "type ", "use ", "using ", "typedef ", "mod ",
    "pub mod ", "}", "};", ")]",
    # C++ spellings of the same thing. `const` was already here; `constexpr` names a constant
    # the same way, and a tile shape fixed by an instruction is not an algorithm either way.
    "constexpr ", "static constexpr ", "inline constexpr ", "template ", "template<",
)
COMMENT = ("//", "/*", "*", "#", "///", "//!")


#: `name: Type,` - a struct field, an enum payload, a function parameter. It declares rather
#: than computes, exactly like the keywords above, and it matches upstream for the same reason:
#: there is one way to say that an attention holds four projections, and every file that holds
#: four projections says it. Told apart from a struct LITERAL's `name: expression,` by what is
#: on the right - a type has no call, no field access and no arithmetic in it.
#: `name: Type,`, public or not - visibility does not change that it declares.
FIELD = re.compile(r"^(?:pub(?:\([\w:]+\))? )?\w+: (?P<ty>[^=]+),$")
#: What only an expression contains: a call, a field access, a question mark, arithmetic, a
#: macro, a string. A type has none of them.
EXPRESSION = re.compile(r"""[.?%!"]|\w\(|\s[-+/*]\s|\bas\b|=>""")
PRIMITIVE = re.compile(r"usize|isize|[ui](?:8|16|32|64|128)|f(?:16|32|64)|bool|char|str")

#: `uint8_t qs[QK_K/2];` - the C spelling of the same thing: a type, a name, an optional array
#: length, and nothing else. A block layout is the format's own statement, so every file that
#: reads GGUF blocks declares them identically; counting those as body made a header of pure
#: format definitions read as a copied algorithm. Anything with an initialiser, a call or an
#: operator outside the brackets is a statement and stays counted.
C_FIELD = re.compile(
    r"^(?:(?:const|static|unsigned|signed|volatile|struct|union|extern)\s+)*"
    r"(?!return\b|break\b|continue\b|goto\b|delete\b|throw\b)"
    r"[A-Za-z_][\w:]*\s*[*&]?\s*"
    r"[A-Za-z_]\w*(?:\s*\[[^\]]*\])*\s*;$"
)


def is_declaration(line: str) -> bool:
    """A line that declares rather than computes.

    Beyond the keywords above, `name: Type,` - a struct field, an enum payload, a function
    parameter. It matches upstream for the reason the keywords do: there is one way to say that
    an attention holds four projections, and every file that holds four is going to say it that
    way. A file written here from nothing still scored ten such lines in a row.

    Told apart from a struct LITERAL's `name: value,` by the right-hand side, and told apart
    conservatively: anything that could be an expression is counted rather than excused.
    """
    if line.startswith(DECLARATION):
        return True
    if C_FIELD.match(line):
        return True
    field = FIELD.match(line)
    if not field:
        return False
    ty = field.group("ty").strip()
    if not ty or EXPRESSION.search(ty):
        return False
    if ty[0].isdigit() or ty[0] == "-" or ty in ("true", "false"):
        return False
    # A bare lowercase name is a variable being moved into a field, not a type - except for
    # the primitives, which are the one family of lowercase type names.
    if re.fullmatch(r"[a-z_]\w*", ty) and not PRIMITIVE.fullmatch(ty):
        return False
    return True


def useful(path: pathlib.Path):
    """The lines worth comparing: no blanks, no comments, nothing short enough to be generic."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    out = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < MIN_LEN or line.startswith(COMMENT):
            continue
        out.append(line)
    return out


#: A line on its own says nothing. `let (b, c, h, w) = q.shape().dims4()?;` appears in every
#: file that reads a four-dimensional tensor, and `.get(name)` in every file that reads a map:
#: at ten characters the corpus contains most of what Rust or CUDA looks like. What means
#: something is a RUN - this many consecutive lines all present upstream, in order.
RUN = 3


def in_runs(lines, index):
    """Body lines sitting in a run of at least RUN consecutive matched lines.

    `lines` must be the file's comparable lines IN ORDER, declarations included: dropping them
    first would join passages that are not adjacent and invent runs. A declaration inside a run
    keeps it going - a copied passage does not stop being one because a field sits in it - but
    only the body lines are counted, so the figure stays comparable to the body denominator.
    """
    total = run = 0
    for line in list(lines) + [None]:
        if line is not None and line in index:
            run += 0 if is_declaration(line) else 1
            continue
        if run >= RUN:
            total += run
        run = 0
    return total


def build_index():
    """line -> the upstream file it came from (first one wins; ties do not matter here)."""
    index = {}
    for name in CORPUS:
        root = REFS / name
        if not root.is_dir():
            print(f"note: {root} absent - its projects cannot be detected", file=sys.stderr)
            continue
        for path in root.rglob("*"):
            if path.suffix not in SUFFIXES or ".git" in path.parts:
                continue
            label = f"{name}/{path.relative_to(root)}"
            for line in useful(path):
                index.setdefault(line, label)
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.0, help="only report at or above this BODY %")
    ap.add_argument("--tsv", metavar="PATH", help="write the notice_gate manifest there")
    ap.add_argument("--markdown", metavar="PATH", help="write the table there instead")
    ap.add_argument("roots", nargs="*", default=["src", "cuda"])
    args = ap.parse_args()

    index = build_index()
    if not index:
        print("no reference corpus - nothing can be measured", file=sys.stderr)
        return 2
    print(f"corpus: {len(index)} distinct lines\n", file=sys.stderr)

    rows = []
    for root in args.roots:
        base = pathlib.Path(root)
        # `rglob` on a FILE yields nothing, and a root named as a single file would then be
        # reported as "no match" - the answer a rewritten file gives too.
        if base.is_file():
            candidates = [base]
        elif base.is_dir():
            candidates = sorted(base.rglob("*"))
        else:
            print(f"note: {base} does not exist - it cannot be measured", file=sys.stderr)
            continue
        measurable = [p for p in candidates if p.suffix in SUFFIXES]
        if not measurable:
            print(f"note: {base} holds nothing this can read", file=sys.stderr)
        for path in measurable:
            mine = useful(path)
            if len(mine) < 30:  # too small for the share to mean anything
                if base.is_file():
                    print(f"note: {path} has {len(mine)} comparable lines, too few to judge",
                          file=sys.stderr)
                continue
            hits = [index[l] for l in mine if l in index]
            if not hits:
                continue
            body = [l for l in mine if not is_declaration(l)]
            run_hits = in_runs(mine, index)
            pct = 100.0 * len(hits) / len(mine)
            body_pct = 100.0 * run_hits / max(len(body), 1)
            if body_pct >= args.min:
                top, n = collections.Counter(hits).most_common(1)[0]
                rows.append((body_pct, pct, run_hits, len(body), str(path), top, n))

    rows.sort(reverse=True)
    if args.tsv:
        measured = {r[4] for r in rows}
        lines = [f"{p}\t{b:.1f}%\t{h}\t{t}" for b, _, h, _, p, t, _ in rows]
        # Every CUDA translation unit, on either side of the nvcc/NVRTC split, holds a row even
        # when it matches nothing: a file that LEFT the table must be distinguishable from one
        # that was never measured.
        for root in ("cuda", "src"):
            for unit in sorted(pathlib.Path(root).rglob("*")):
                if unit.suffix in {".cu", ".cuh", ".h", ".hpp"} and str(unit) not in measured:
                    lines.append(f"{unit}\t0.0%\t0\t-")
        lines.sort()
        pathlib.Path(args.tsv).write_text("\n".join(lines) + "\n")
        print(f"{args.tsv}: {len(lines)} rows")
        return 0
    if args.markdown:
        out = [
            "# Provenance",
            "",
            "Generated by `scripts/provenance/measure.py`; do not hand-edit. `body` is the share of a",
            "file's own lines that sit in a RUN of three or more consecutive lines found verbatim in",
            "the reference checkouts; `all` is the older figure, every matching",
            "line counted alone.",
            "",
            "The run is what the first column exists for. A single line proves nothing - at ten",
            "characters the corpus holds most of what Rust or CUDA looks like, and `.get(name)`",
            "matching is not a provenance. Three in a row, in order, is a passage.",
            "",
            "Both figures are FLOORS: neither sees a line that was reformatted, and both count a",
            "line the format dictates the same as a line that was a free choice - a q4_K",
            "dequantiser has little room to differ from any other. Read them as \"at least this",
            "much is still theirs\", then read the code.",
            "",
            "`NOTICE.md` is the licensing record and takes precedence; this file is the",
            "measurement it should agree with, and the work list for replacing what it counts.",
            "",
            "| body | all | body lines | file | closest upstream |",
            "|-----:|----:|-----------:|------|------------------|",
        ]
        for body_pct, pct, hit, tot, path, top, n in rows:
            out.append(f"| {body_pct:.1f}% | {pct:.1f}% | {hit}/{tot} | `{path}` | `{top}` |")
        out.append("")
        out.append(f"{len(rows)} files at or above {args.min:.0f}%.")
        pathlib.Path(args.markdown).write_text("\n".join(out) + "\n")
        print(f"{args.markdown}: {len(rows)} rows")
        return 0
    print("body      all   body lines")
    for body_pct, pct, hit, tot, path, top, n in rows:
        print(f"{body_pct:5.1f}%  {pct:5.1f}%  {hit:4}/{tot:<4}  {path:50}  {top}")
    print(f"\n{len(rows)} files with a match"
          f"{f' at or above {args.min:.0f}%' if args.min else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
