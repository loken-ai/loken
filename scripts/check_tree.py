#!/usr/bin/env python3
"""Refuse a module tree that a build would reject, in seconds instead of twenty minutes.

A release build of this repository takes about twenty minutes, and roughly fifteen of those
are `ptxas`/`cicc` rebuilding the CUDA kernels. Spending one to be told that a file was moved
and its `mod` declaration was not updated is a bad trade, and the four checks below are the
ones that were actually needed while `src/inference` was reorganised - each of them caught a
real defect that would otherwise have cost a full build to discover:

  includes      `include_str!("cuda/fused.cu")` breaks the moment its file changes depth, and
                fifteen of them did.
  declarations  a `mod x;` whose file no longer exists, honouring `#[path = "..."]`.
  orphans       an `#[cfg(...)]` left behind by a removed declaration attaches itself to the NEXT
                one. A stray `#[cfg(feature = "video")]` once stacked on top of
                `#[cfg(feature = "cuda")]` and made the MoE kernels vanish from any build
                without video - silently, because the code still compiled.
  undeclared    a `.rs` beside a `mod.rs` that nothing declares is dead weight nobody notices.

Run it before every build:  python3 scripts/check_tree.py
Exit status is 0 when the tree is sound, 1 otherwise, so it also works as a gate.

One thing it deliberately does NOT do is stand in for building the test target. `cargo check
--lib` and `cargo test --no-run` compile different sets of files - `#[cfg(test)]` modules, and
in this repository several thousand lines of them - so a tree that passes both this script and
`--lib` can still have a broken test build. That happened: fifteen commits went by on `--lib`
alone while `use super::crate::...` sat in three test modules, invisible because the library
never sees them. Build the test target too, and not only at the end.
"""

import pathlib
import re
import sys

SRC = pathlib.Path("src")
DECL = re.compile(r"^\s*(?:pub(?:\([a-z]+\))? )?mod ([a-z_0-9]+);")
PATH_ATTR = re.compile(r'#\[path = "([^"]+)"\]')
INCLUDE = re.compile(r'include_(?:str|bytes)!\("([^"]+)"\)')


def broken_includes():
    """A relative include is resolved from the including file, so moving it breaks the path."""
    for f in sorted(SRC.rglob("*.rs")):
        for m in INCLUDE.finditer(f.read_text()):
            if not (f.parent / m.group(1)).exists():
                yield f"{f}: include_str!(\"{m.group(1)}\") has no file"


def _declarations(mod: pathlib.Path):
    """Each `mod x;` with the attribute lines immediately above it."""
    lines = mod.read_text().splitlines()
    attrs: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#["):
            attrs.append(s)
            continue
        g = DECL.match(ln)
        if g:
            yield g.group(1), list(attrs)
            attrs = []
        elif not s.startswith("//"):
            # Anything else ends the run; a blank line or an item is not a declaration's
            # preamble, and treating it as one is how orphans go unnoticed.
            attrs = []


def missing_targets():
    for mod in sorted(SRC.rglob("mod.rs")):
        for name, attrs in _declarations(mod):
            override = next((PATH_ATTR.search(a) for a in attrs if PATH_ATTR.search(a)), None)
            if override:
                if not (mod.parent / override.group(1)).exists():
                    yield f"{mod}: mod {name} points at {override.group(1)}, which is absent"
            elif not (mod.parent / f"{name}.rs").exists() and not (mod.parent / name / "mod.rs").exists():
                yield f"{mod}: mod {name} has neither {name}.rs nor {name}/mod.rs"


def stacked_cfgs():
    """Two `cfg` attributes on one declaration mean it needs BOTH features.

    That is occasionally deliberate, so this reports rather than condemns - but it is what an
    orphaned attribute looks like, and the difference is worth a human glance.
    """
    for mod in sorted(SRC.rglob("mod.rs")):
        for name, attrs in _declarations(mod):
            cfgs = [a for a in attrs if a.startswith("#[cfg(")]
            if len(cfgs) > 1:
                yield f"{mod}: mod {name} requires all of {' '.join(cfgs)}"


def undeclared_files():
    for mod in sorted(SRC.rglob("mod.rs")):
        body = mod.read_text()
        declared = set(re.findall(r"mod ([a-z_0-9]+);", body)) | set(PATH_ATTR.findall(body))
        for f in sorted(mod.parent.glob("*.rs")):
            if f.name != "mod.rs" and f.stem not in declared and f.name not in declared:
                yield f"{f}: beside a mod.rs that does not declare it"


def unknown_module_paths():
    """`crate::inference::foo` where no `foo` module exists.

    Worth its own check because the obvious way to update these paths - substitute
    `inference::<name>` - cannot see `inference::{a, b}`, and a grouped import is exactly where
    a stale path hides. Both forms are resolved here against the directory tree.
    """
    root = SRC / "inference"
    known = {p.stem for p in root.glob("*.rs") if p.name != "mod.rs"}
    known |= {d.name for d in root.iterdir() if d.is_dir() and (d / "mod.rs").exists()}
    # `#[path]` gives a module a name its file does not carry.
    known |= set(re.findall(r"^\s*(?:pub(?:\([a-z]+\))? )?mod ([a-z_0-9]+);",
                            (root / "mod.rs").read_text(), re.M))
    known |= set(re.findall(r"pub use [a-z_0-9]+ as ([a-z_0-9]+);", (root / "mod.rs").read_text()))

    single = re.compile(r"\binference::([a-z_0-9]+)\b")
    grouped = re.compile(r"\binference::\{([^}]*)\}")
    for f in sorted(SRC.rglob("*.rs")):
        text = f.read_text()
        heads = {m.group(1) for m in single.finditer(text)}
        for m in grouped.finditer(text):
            heads |= {p.strip().split("::")[0].split(" as ")[0]
                      for p in m.group(1).split(",") if p.strip()}
        for h in sorted(heads - known):
            yield f"{f}: crate::inference::{h} names no module"


CHECKS = [
    ("relative includes", broken_includes, True),
    ("module paths", unknown_module_paths, True),
    ("module declarations", missing_targets, True),
    ("undeclared files", undeclared_files, True),
    ("stacked cfg attributes", stacked_cfgs, False),
]


def main() -> int:
    if not SRC.is_dir():
        print("run me from the repository root", file=sys.stderr)
        return 2
    failed = False
    for title, check, fatal in CHECKS:
        problems = list(check())
        if not problems:
            print(f"  ok    {title}")
            continue
        print(f"  {'FAIL' if fatal else 'note'}  {title}")
        for p in problems:
            print(f"          {p}")
        failed |= fatal
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
