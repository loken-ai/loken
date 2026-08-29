"""Attribute a file's matched body lines to the top-level item they sit in.

Boundaries come from column-zero signatures, not from brace depth: the conditional
compilation in these headers leaves braces unbalanced by design, so a depth counter walks off
after the first `#if`/`#else` pair and charges the rest of the file to one function.
"""
import sys, re, pathlib, importlib.util
spec = importlib.util.spec_from_file_location("m", "scripts/provenance/measure.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
index = m.build_index()
raw = pathlib.Path(sys.argv[1]).read_text().split("\n")

SIG = re.compile(r"^(?:static|template|extern|struct|union|class|namespace|__global__|__device__|"
                 r"[A-Za-z_]\w*\s+[A-Za-z_])")
starts = [i for i, r in enumerate(raw) if SIG.match(r)]
# a template clause and the declaration under it are one item
starts = [i for j, i in enumerate(starts)
          if not (j and raw[starts[j-1]].startswith("template") and i - starts[j-1] <= 3)]
bounds = list(zip([0] + starts, starts + [len(raw)]))

rows = []
for a, b in bounds:
    body = [re.sub(r"\s+", " ", r).strip() for r in raw[a:b]]
    body = [l for l in body if len(l) >= m.MIN_LEN and not l.startswith(m.COMMENT)
            and not m.is_declaration(l)]
    if not body:
        continue
    hit = sum(1 for l in body if l in index)
    if hit:
        name = re.findall(r"\b(\w+)\s*[({]", raw[a] + " " + (raw[a+1] if a+1 < len(raw) else ""))
        rows.append((hit, len(body), a + 1, b - a, name[-1] if name else raw[a][:40]))
for hit, tot, first, span, name in sorted(rows, reverse=True):
    print(f"{hit:5}/{tot:<5} {100*hit/tot:5.1f}%  L{first:<6} {span:5} lignes  {name}")
print(f"\ntotal apparie: {sum(r[0] for r in rows)}")
