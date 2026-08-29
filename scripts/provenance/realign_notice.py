"""Copy the manifest's figures into NOTICE.md's table, and report what must be added by hand.

The gate holds NOTICE to the measurement; this is the half of that contract a script can do.
Rows the measurement puts under the floor leave; rows above it that are missing are named,
because a new entry needs an upstream and a licence that only a person can attest.
"""
import pathlib, re, sys

root = pathlib.Path(__file__).resolve().parents[2]
man = {}
for line in (root / "scripts/provenance/manifest.tsv").read_text().splitlines():
    f, pct, n, _ = line.split("\t")
    man[f] = (float(pct.rstrip("%")), int(n))

p = root / "NOTICE.md"
t = p.read_text()


def restate(m):
    f = m.group(1)
    if f not in man:
        return m.group(0)
    pct, n = man[f]
    return f"| `{f}` | {round(pct)}% | {n} |"


t = re.sub(r"^\| `([^`]+)` \| \d+% \| \d+ \|", restate, t, flags=re.M)

kept, dropped = [], []
for line in t.split("\n"):
    m = re.match(r"^\| `([^`]+)` \|", line)
    if m and m.group(1) in man and man[m.group(1)][0] < 10:
        dropped.append(m.group(1))
        continue
    kept.append(line)
t = "\n".join(kept)
p.write_text(t)

listed = {m.group(1) for m in re.finditer(r"^\| `([^`]+)` \| \d+%", t, flags=re.M)}
missing = [(f, *man[f]) for f in man if man[f][0] >= 30 and f not in listed]
for f in dropped:
    print(f"left the table: {f}")
for f, pct, n in sorted(missing, key=lambda r: -r[2]):
    print(f"NEEDS A ROW: {n:>4} lines, {pct:.0f}%  {f}", file=sys.stderr)
sys.exit(1 if missing else 0)
