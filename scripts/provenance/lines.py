"""Print the body lines of a file that the corpus also contains, with their line numbers.

`measure.py` says how many lines a file shares with the reference corpus; this says *which*,
so a rewrite can be aimed at them instead of at the whole file. Same rules as the measure:
a line counts only inside a run of at least `RUN` consecutive matches - a lone `}` proves
nothing - and declarations keep a run going without being counted themselves.
"""
import sys, re, pathlib, importlib.util

spec = importlib.util.spec_from_file_location("m", "scripts/provenance/measure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
index = m.build_index()


def comparable(path):
    """(line number, normalised text) for every line the measure would compare."""
    for n, raw in enumerate(path.read_text(errors="ignore").split("\n"), 1):
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) >= m.MIN_LEN and not line.startswith(m.COMMENT):
            yield n, line


for arg in sys.argv[1:]:
    path = pathlib.Path(arg)
    run, hits = [], []
    for entry in list(comparable(path)) + [None]:
        if entry is not None and entry[1] in index:
            run.append(entry)
            continue
        if sum(1 for _, t in run if not m.is_declaration(t)) >= m.RUN:
            hits.extend(run)
        run = []
    body = [(n, t) for n, t in hits if not m.is_declaration(t)]
    print(f"-- {path}  {len(body)} lignes de corps, {len(hits)} avec les declarations")
    for n, text in hits:
        print(f"{n:5d} {' ' if m.is_declaration(text) else '*'} {text}")
