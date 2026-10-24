#!/usr/bin/env python3
"""Write the benchmark pages from the store, one per hardware configuration.

    report.py                         # regenerate docs/benchmarks/ and the index block
    report.py file results/*.json     # file records into the store, then regenerate
    report.py learn docs/BENCHMARKS.md  # record which model sits in which family section

The store (bench/store in this tree, filled by lokprof/scripts/store.py) holds every run
once. This groups the runs by `hardware_id` and writes one page per group with the report's
usual table. Nothing in a page is typed: the
hardware block comes from the records, the dates from their timestamps, the builds from what
the servers said.

A model's family is the `general.architecture` of its file. A model whose file is no longer
on disk keeps the family it was last seen under (`architectures.tsv` in the store).
"""
import json, pathlib, re, sys, tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# The record schema and the hardware identity belong to lokprof, the tool that writes the records.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "lokprof" / "scripts"))
import bench_row as B
import classes as C
import hardware as H
import store as S

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "bench" / "store"
DOCS = ROOT / "docs"
OUT = DOCS / "benchmarks"
BEGIN, END = "<!-- benchmarks:index -->", "<!-- /benchmarks:index -->"


def models_dir(path):
    """The server's model store, where a model's file names its family."""
    return tomllib.loads(path.read_text()).get("ollama_models_dir")


def runs(store):
    """(record, source, run id) for every run in the store, oldest first."""
    out = []
    for p in sorted((store / "runs").glob("*.json")):
        doc = json.loads(p.read_text())
        out.append((doc, doc.get("converted_from") or p.name, p.stem))
    out.sort(key=lambda t: (t[0].get("timestamp") or "", t[2]))
    return out


def hardware_id(doc):
    m = doc.get("machine")
    if not m:
        return None
    return m.get("hardware_id") or H.hardware_id(m)


def slug(machine, taken):
    cards = sorted((g.get("name") or "card").replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
                   for g in machine.get("gpus", []))
    name = "-".join(re.sub(r"[^a-z0-9]+", "", c.lower()) for c in cards) or "cpu-only"
    cpu = re.sub(r"[^a-z0-9]+", "", (machine.get("cpu") or "").lower().replace("intel(r)", "")
                 .replace("core(tm)", "").replace("amd", "").split("cpu")[0].split("@")[0])
    name = f"{name}-{cpu}" if cpu else name
    while name in taken:
        name += "x"
    return name


def hardware_block(machine, docs, backend):
    lines = ["| | |", "|---|---|"]
    for i, g in enumerate(machine.get("gpus", [])):
        cap = f", {g['power_cap_w']} W" if g.get("power_cap_w") else ""
        mem = f" - {g['vram_mib'] / 1024:.0f} GB" if g.get("vram_mib") else ""
        lines.append(f"| GPU {i} | {g.get('name') or '?'}{mem}{cap} |")
    if not machine.get("gpus"):
        lines.append("| GPU | none recorded |")
    cpu, threads = machine.get("cpu") or "?", machine.get("threads")
    lines.append(f"| CPU | {cpu}" + (f" - {threads} threads" if threads else "") + " |")
    if machine.get("ram_gib"):
        lines.append(f"| RAM | {machine['ram_gib']} GB |")
    seen = lambda k: sorted({(d["machine"] or {}).get(k) for d, _, _ in docs
                             if (d.get("machine") or {}).get(k)})
    lines.append(f"| Backend | {backend} |")
    if seen("driver"):
        lines.append(f"| Driver | {', '.join(seen('driver'))} |")
    if seen("kernel"):
        lines.append(f"| Kernel | {', '.join(seen('kernel'))} |")
    note = ""
    if any((d.get("machine") or {}).get("attested") for d, _, _ in docs):
        note = ("\nRuns taken before the runner recorded its machine carry the hardware an "
                "operator attested for them; their driver and kernel are unknown.\n")
    return "\n".join(lines) + "\n" + note


def read_arch_cache(store):
    p = store / "architectures.tsv"
    if not p.is_file():
        return {}
    return dict(l.split("\t", 1) for l in p.read_text().splitlines() if "\t" in l)


def write_arch_cache(store, cache):
    (store / "architectures.tsv").write_text(
        "".join(f"{m}\t{a}\n" for m, a in sorted(cache.items())))


def learn(store, md):
    """Record the family section each model sits in, from an existing page."""
    cache = read_arch_cache(store)
    family = None
    for line in pathlib.Path(md).read_text().splitlines():
        if line.startswith("### "):
            family = line[4:].strip()
        elif family and family != "unfiled" and line.startswith("| ") and not line.startswith("| Model"):
            cache[line.split("|")[1].strip()] = family
    write_arch_cache(store, cache)
    print(f"{len(cache)} model(s) known in {store / 'architectures.tsv'}")


def families(models, cache, models_dir):
    """model -> family, resolving the file where it exists and the cache where it does not."""
    out = {}
    for m in models:
        arch = None
        if models_dir:
            try:
                arch = B.architecture_of(m, models_dir)
            except Exception:
                arch = None
        if arch:
            cache[m] = arch
        out[m] = arch or cache.get(m)
    return out


def span(docs):
    days = sorted((d.get("timestamp") or "")[:10] for d, _, _ in docs if d.get("timestamp"))
    return (days[0], days[-1]) if days else ("?", "?")


def backend_note(backend):
    """The line a page carries when no reference engine runs its backend."""
    if backend == "opencl":
        return ["No reference engine runs a language model through OpenCL on a GPU (ollama and "
                "vLLM do not), so this page holds loken alone.", ""]
    return []


def page(title, machine, group, merged, rows_, fam, replays, archived, backend):
    first, last = span(group)
    builds = sorted({e["version"] for by in merged.values() for k, e in by.items()
                     if k.startswith("loken") and e.get("version")})
    out = [f"# Benchmarks - {title}", "",
           f"{len(group)} run(s), {first} to {last}. loken builds measured: "
           f"{', '.join(builds) or 'unstated'}. A row's `Date` and `Run` name the run behind it."
           + (f" {replays} replayable record(s) of this hardware are in the store and not tabled." if replays else "")
           + (f" {archived} archived run(s) (superseded builds or another protocol) are held and not tabled." if archived else ""),
           "", "## Hardware", "", hardware_block(machine, group, backend), "## Results", "",
           "Method and column definitions: [BENCHMARKS.md](../BENCHMARKS.md). Each rate reads `cold / warm`: "
           "the first request of a cell, then the median of the later ones ([why](../BENCHMARKS.md#cold-and-warm)).", ""]
    out += backend_note(backend)
    by_family = {}
    for r in rows_:
        by_family.setdefault(fam.get(r[0].strip()) or "unfiled", []).append(r)
    for name in sorted(by_family, key=lambda n: (n == "unfiled", n)):
        out += [f"### {name}", ""]
        if (DOCS / "img" / f"family-{name}.svg").is_file():
            out += [f"![{name}](../img/family-{name}.svg)", ""]
        elif name == "unfiled":
            out += ["Measured, and not yet placed in a family section.", ""]
        out += [B.table(by_family[name]), ""]
    return "\n".join(out).rstrip("\n") + "\n"


def decode_of(by, kind):
    """The decode rate and answer length of the plain loken entry, or of the best peer."""
    if kind == "loken":
        picks = [v for k, v in by.items() if k.startswith("loken") and " + " not in k]
    else:
        picks = [v for k, v in by.items() if k.split("@")[0] == "ollama"]
    picks = [v for v in picks if v.get("dec_w") and v.get("ntok") and v.get("coh") is not False]
    return max(picks, key=lambda v: v["dec_w"]) if picks else None


def main(argv):
    cfg = pathlib.Path(argv[argv.index("--config") + 1]) if "--config" in argv else ROOT / "config.toml"
    store = STORE
    models = models_dir(cfg)
    argv = [a for a in argv if a != "--config" and a != str(cfg)]
    if argv[:1] == ["learn"]:
        return learn(store, argv[1])
    if argv[:1] == ["file"]:
        paths = [pathlib.Path(a) for a in argv[1:]]
        S.add(paths, store)
        S.index(store)
    groups, nameless = {}, 0
    for doc, source, rid in runs(store):
        hid = hardware_id(doc)
        if hid:
            # One page per hardware and per backend loken answered on: the same card through
            # two backends is two measurements of ours, not one.
            groups.setdefault((hid, C.loken_backend(doc)), []).append((doc, source, rid))
        else:
            nameless += 1
    cache = read_arch_cache(store)
    OUT.mkdir(parents=True, exist_ok=True)
    taken, written, index_rows = set(), [], []
    for (hid, backend), group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        machine = next(d["machine"] for d, _, _ in reversed(group) if d.get("machine"))
        # A replayable record is one engine on one prompt, kept to be played again on another
        # binary; it has no peer to compare against and is not a row of these tables.
        replays = sum(1 for d, _, _ in group if d.get("replay"))
        group = [t for t in group if not t[0].get("replay")]
        # A record taken from a named archive (superseded builds, a contended protocol, one
        # card) stays in the store and is not a row of the current tables.
        archived = sum(1 for d, _, _ in group if d.get("archive"))
        group = [t for t in group if not t[0].get("archive")]
        merged = B.load_docs(group)
        rows_ = B.cells(merged)
        fam = families({r[0].strip() for r in rows_}, cache, models)
        name = slug(machine, taken) + (f"-{backend}" if backend != "unstated" else "")
        taken.add(name)
        (OUT / f"{name}.md").write_text(page(H.label(machine), machine, group,
                                              merged, rows_, fam, replays, archived, backend))
        written.append(f"{name}.md")
        first, last = span(group)
        index_rows.append([f"[{H.label(machine)}](benchmarks/{name}.md)", backend, hid,
                           f"{first} to {last}", str(len(group)), str(len(rows_))])
    write_arch_cache(store, cache)

    idx = [BEGIN, "", "| Hardware | Backend | Id | Runs from | Runs | Rows |", "|---|---|---|---|---|---|"]
    idx += ["| " + " | ".join(r) + " |" for r in index_rows]
    if nameless:
        idx += ["", f"{nameless} run(s) in the store carry no machine and appear on no page."]
    idx += ["", END]
    page_md = DOCS / "BENCHMARKS.md"
    text = page_md.read_text()
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: "\n".join(idx), text, flags=re.S)
    else:
        sys.exit(f"{page_md} has no {BEGIN} block to fill")
    page_md.write_text(text)
    for stale in OUT.glob("*.md"):
        if stale.name not in written:
            stale.unlink()
            print(f"removed {stale.name}: no hardware in the store behind it", file=sys.stderr)
    print(f"wrote {len(written)} page(s) for {len(groups)} hardware and backend group(s)")
    return 0


def _table(head, rows_):
    w = [max(len(head[i]), *(len(r[i]) for r in rows_)) for i in range(len(head))]
    line = lambda c: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(c)) + " |"
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows_])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
