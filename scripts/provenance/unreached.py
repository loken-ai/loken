#!/usr/bin/env python3
"""Report every device function that no translation unit including it can reach.

A definition elsewhere is not a use. "Does this name appear anywhere else in the tree" answers
yes for a function nothing calls, as soon as another header declares an OVERLOAD of the same
name - so the question is asked of one translation unit's include closure, with definition
heads subtracted from the count.

A unit here is a `.cu` file plus everything it includes, transitively, from within the
repository. A function is reported only when every unit that compiles it leaves it unreached:
a helper used by one format's instance and not another's is alive.

The C preprocessor is not run, so a name reached only from inside a `#if` this build never
takes still counts as reached. That direction is deliberate - the report is a list of things
that are certainly dead, not a list of everything that might be.

    python3 scripts/provenance/unreached.py           # cuda/ and src/, every unit
    python3 scripts/provenance/unreached.py cuda      # one tree
"""

import pathlib
import re
import sys

SOURCE = {".cu", ".cuh", ".h", ".hpp"}
INCLUDE_STR = re.compile(r'include_str!\("([^"]+)"\)')


def assembled_units() -> list:
    """The NVRTC modules, read from the loaders that concatenate them.

    A module is built at run time by pasting several `.cu` files into one source, so those
    files are ONE translation unit though each is a `.cu` on disk. The Rust module that names
    them with `include_str!` is the authority on which they are - hardcoding the list here
    would leave a file behind the day the loader changes.
    """
    groups = []
    for rs in sorted(pathlib.Path("src").rglob("*.rs")):
        members = []
        for rel in INCLUDE_STR.findall(rs.read_text(errors="ignore")):
            target = (rs.parent / rel).resolve()
            try:
                target = target.relative_to(pathlib.Path.cwd())
            except ValueError:
                continue
            if target.exists() and target.suffix in SOURCE:
                members.append(target)
        if len(members) > 1:
            groups.append(sorted(set(members)))
    return groups
INCLUDE = re.compile(r'(?m)^\s*#\s*include\s+"([^"]+)"')
DEFINITION = re.compile(
    r"(?m)^(?:template\s*<[^\n]*>\s*\n?)?\s*(?:static\s+)?(?:__device__|__global__|__host__)"
    r"[^\n(]*?\b(\w+)\s*\("
)
# A name the preprocessor builds cannot be searched for; leave those stems alone.
PASTED = re.compile(r"##\s*(\w+)|(\w+)\s*##")


def closure(entry: pathlib.Path, cache: dict) -> set:
    """Every repository file `entry` pulls in, transitively."""
    if entry in cache:
        return cache[entry]
    cache[entry] = seen = {entry}
    try:
        text = entry.read_text(errors="ignore")
    except OSError:
        return seen
    for rel in INCLUDE.findall(text):
        target = (entry.parent / rel).resolve()
        try:
            target = target.relative_to(pathlib.Path.cwd())
        except ValueError:
            continue  # a system header, or something outside the tree
        if target.exists() and target.suffix in SOURCE:
            seen |= closure(target, cache)
    return seen


def uses(name: str, text: str) -> int:
    """Occurrences of `name` that are not the head of a definition of `name`."""
    n = len(re.findall(r"\b" + re.escape(name) + r"\b", text))
    return n - sum(1 for m in DEFINITION.finditer(text) if m.group(1) == name)


def host_requested() -> set:
    """Kernel names the host asks the driver for, by string.

    An entry point is often assembled by the preprocessor - `dequantize_block_##T##_f32` - so
    the stem appears nowhere a search can find it, while the host asks for the finished name.
    Any definition whose name is a prefix of one of these is reached.
    """
    names = set()
    for rs in pathlib.Path("src").rglob("*.rs"):
        text = rs.read_text(errors="ignore")
        names |= set(re.findall(r'"([a-z_][a-z0-9_]{6,})"', text))
        # A name the host assembles - `format!("dequantize_block_{dtype}_f32")` - spells only
        # its fixed parts, so each of those counts as a stem in its own right.
        for lit in re.findall(r'"([a-z_][a-z0-9_{}]*)"', text):
            names |= {part for part in re.split(r"\{[^}]*\}", lit) if len(part) > 6}
    return names


def main() -> int:
    roots = sys.argv[1:] or ["cuda", "src"]
    ASSEMBLED = assembled_units()
    assembled = {f for group in ASSEMBLED for f in group}
    units = [p for root in roots for p in sorted(pathlib.Path(root).rglob("*.cu"))
             if p not in assembled]
    units += [g[0] for g in ASSEMBLED if g]
    if not units:
        print("no translation unit found - wrong directory?", file=sys.stderr)
        return 2

    cache: dict = {}
    asked = host_requested()
    ASSEMBLED = assembled_units()
    # file -> the names it defines; and name -> whether any unit reaches it
    reached: dict = {}
    defined: dict = {}

    for unit in units:
        group = next((g for g in ASSEMBLED if g and g[0] == unit), [unit])
        files = set()
        for member in group:
            files |= closure(member, cache)
        texts = {f: f.read_text(errors="ignore") for f in files}
        pasted = set()
        for text in texts.values():
            for a, b in PASTED.findall(text):
                pasted.add(a or b)
        for f, text in texts.items():
            for m in DEFINITION.finditer(text):
                name = m.group(1)
                if name in {"if", "while", "for", "switch", "return", "operator"}:
                    continue
                # `dequantize_block_##T` reaches `dequantize_block_q5_0` while spelling only
                # the stem, so a stem that PREFIXES the name is as good as the name. An
                # exact-match test here reports live templates as dead.
                if any(name.startswith(stem) or stem.startswith(name) for stem in pasted):
                    continue
                head = text[m.start():m.end()]
                if "__global__" in head and "static" not in head:
                    continue   # an entry point, reached from the host by name
                defined.setdefault((f, name), 0)
                if any(uses(name, t) > 0 for t in texts.values()):
                    reached[(f, name)] = True

    dead = sorted(k for k in defined if k not in reached
                  and not any(n.startswith(k[1]) or k[1].startswith(n) for n in asked))
    for f, name in dead:
        print(f"{f}: {name}")
    print(f"\n{len(dead)} definitions no unit reaches, out of {len(defined)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
