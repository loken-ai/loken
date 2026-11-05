#!/usr/bin/env python3
"""Refuse a model that does not go through the generic hetero route.

The rule: the generic hetero route (HeteroPlan and the OOM ladder in src/inference/place/) is
the only loading and placement path. A backend, a family or a streaming client outside it
duplicates placement, batching and expert streaming.

  python3 scripts/check_hetero.py                    the whole tree (CI, pre-commit)
  python3 scripts/check_hetero.py --message FILE     the staged diff against a commit message
                                                     (commit-msg)

A commit whose subject has the scope `hetero` is a migration step and may add lines to the paths
that bypass the route. Any other commit may only shrink them, or add tests.

scripts/hetero_route.txt classifies every backend and every model family. A new one must be
listed there, which is a decision, and a bypass is listed as debt that only shrinks.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "scripts" / "hetero_route.txt"


def read_list():
    kinds = {"backend": set(), "debt-backend": set(), "family": set(), "debt-family": set(), "debt-path": []}
    for raw in LIST.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        kind, name = line.split(None, 1)
        if kind == "debt-path":
            kinds[kind].append(name)
        else:
            kinds[kind].add(name)
    return kinds


def rust_files():
    return [p for p in (ROOT / "src").rglob("*.rs")]


def check_tree(kinds):
    problems = []
    known = kinds["backend"] | kinds["debt-backend"]
    seen = set()
    stream_clients = []
    for path in rust_files():
        text = path.read_text(errors="replace")
        rel = path.relative_to(ROOT).as_posix()
        for name in re.findall(r"^\s*(?:unsafe\s+)?impl(?:<[^>]*>)?\s+ModelBackend\s+for\s+(\w+)", text, re.M):
            seen.add(name)
            if name not in known:
                problems.append(f"{rel}: `impl ModelBackend for {name}` is not classified in scripts/hetero_route.txt: "
                                "a backend must load and place through HeteroPlan and the ladder")
        if re.search(r"^\s*impl(?:<[^>]*>)?\s+StreamedModel\s+for\s", text, re.M):
            if not any(rel.startswith(p) for p in kinds["debt-path"]):
                problems.append(f"{rel}: a new `impl StreamedModel`: experts stream through the generic tiers "
                                "(src/inference/generic_transformer/expert_tier*.rs), not through offload/")
    for name in sorted(kinds["debt-backend"] - seen):
        problems.append(f"scripts/hetero_route.txt: debt-backend {name} no longer exists, remove the line")
    for name in sorted(kinds["backend"] - seen):
        problems.append(f"scripts/hetero_route.txt: backend {name} no longer exists, remove the line")
    dirs = {p.name for p in (ROOT / "src/inference/model").iterdir() if p.is_dir()}
    listed = kinds["family"] | kinds["debt-family"]
    for name in sorted(dirs - listed):
        problems.append(f"src/inference/model/{name}: a model family not classified in scripts/hetero_route.txt: "
                        "it must load through HeteroPlan, or place a whole block through ladder::load_whole")
    for name in sorted(listed - dirs):
        problems.append(f"scripts/hetero_route.txt: family {name} no longer exists, remove the line")
    for p in kinds["debt-path"]:
        if not (ROOT / p).exists():
            problems.append(f"scripts/hetero_route.txt: debt-path {p} no longer exists, remove the line")
    return problems


def is_test_file(rel):
    name = Path(rel).name
    return (rel.startswith("tests/") or "test" in name or name.startswith("oracle")
            or "/tests/" in rel)


def migration(message_file):
    subject = Path(message_file).read_text(errors="replace").splitlines()[0] if message_file else ""
    return re.match(r"^\w+\(hetero\)!?:", subject) is not None


def check_staged(kinds):
    problems = []
    out = subprocess.run(["git", "diff", "--cached", "--numstat", "--diff-filter=ACMR"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    for line in out.splitlines():
        added, removed, rel = line.split("\t", 2)
        if added == "-":
            continue
        if any(rel.startswith(p) for p in kinds["debt-path"]) and not is_test_file(rel):
            if int(added) > int(removed):
                problems.append(f"{rel}: +{added} -{removed}: this path bypasses the hetero route and may only shrink "
                                "(tests may be added). Move the capability onto the generic route, "
                                "or give the commit the scope `hetero` if it is a migration step.")
    return problems


def main():
    kinds = read_list()
    problems = []
    if "--message" in sys.argv:
        if not migration(sys.argv[sys.argv.index("--message") + 1]):
            problems += check_staged(kinds)
    else:
        problems += check_tree(kinds)
    if problems:
        print("check_hetero: a model path outside the hetero route", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
