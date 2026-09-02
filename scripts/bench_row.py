"""Turn bench JSONs into the report's table, aligned, with nothing retyped."""
import re
import json, pathlib, sys

import subprocess
def _ollama_version():
    """Ask the binary rather than carry a literal: a table that names the wrong version
    of the engine it measured is worse than one that names none."""
    try:
        out = subprocess.run(["/usr/local/bin/ollama", "--version"],
                             capture_output=True, text=True, timeout=10)
        m = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", out.stdout + out.stderr)
        return "Ollama " + m.group(1) if m else "Ollama"
    except Exception:
        return "Ollama"

VERS = {"ollama": _ollama_version(), "loken": "loken 0.1.0", "vllm": "vLLM 0.22.0"}

# Results measured before the project was renamed carry the old target name. They are the same
# engine and the same campaign, so they are folded in rather than dropped - reading only the
# current name would empty most of the table without saying anything was missing.
RETIRED_TARGETS = {"llmuse": "loken", "llmserver": "loken"}
COLS = ["Model", "Ctx", "Prompt", "Mode", "Device", "Engine", "Prefill tok/s", "Decode tok/s", "Tokens",
        "E2E ms", "J/req", "J/token", "Δ decode", "Δ energy", "Date"]

def measured_prefill(r, st):
    """Prompt tokens divided by the measured time to first token, or nothing.

    Blank rather than falling back to the engine's own figure: a column holding two
    definitions is the defect this replaces, not a lesser version of it.
    """
    # The MEDIAN, where every other figure here takes the mean: the first iteration of a
    # cell is a cold load, and its time to first token is seconds where the warm ones are
    # milliseconds. Averaged in, it put ollama's prefill at 4 tok/s on a cell where it does
    # 642. Decode tolerates the mean because it does not carry that outlier.
    v = st.get("TTFT")
    ttft = v.get("p50") if isinstance(v, dict) else v
    if not ttft:
        return None
    toks = next((i.get("prompt_tokens") for i in (r.get("iterations") or [])
                 if i.get("prompt_tokens")), None)
    if not toks or ttft <= 0:
        return None
    return toks / (ttft / 1000.0)

def stat(st, name):
    """The harness's own aggregate, so the report never defines a second one."""
    v = st.get(name)
    if v is None: return None
    return v.get("mean", v.get("value")) if isinstance(v, dict) else v

def energy_stat(st, prefix):
    """The energy aggregate, and the domains it counted.

    The label carries its domains - "Energy J/tok [gpu]" - because a total over the cards
    alone and one that also counted the host are not the same measurement. Read by prefix,
    so a machine that gains a readable RAPL counter does not silently blank the column; and
    refused when several keys match rather than picking one of them.
    """
    hits = [k for k in st if k.startswith(prefix)]
    if len(hits) != 1:
        return None, None
    dom = hits[0][len(prefix):].strip().strip("[]") or None
    return stat(st, hits[0]), dom

def rows(path, mode, device):
    d = json.load(open(path)); out = {}
    # A run that was stopped before it measured anything writes the file with no results.
    # Saying which one is empty beats a traceback that names only the line that tripped.
    if not d.get("results"):
        print(f"skipping {path}: no results in it", file=sys.stderr)
        return {}
    for r in d["results"]:
        st = r["stats"]
        jtok, jdom = energy_stat(st, "Energy J/tok")
        ntok = stat(st, "Tokens generated")
        # The prompt is part of the key: two cells that differ only by it are two
        # measurements, and merging them silently keeps whichever was read last.
        # The device belongs in the key for the same reason the mode and the prompt do:
        # a CPU cell and a GPU cell of one model are two measurements, and a key that
        # cannot tell them apart keeps whichever was read last.
        target = r["target"].lower()
        target = RETIRED_TARGETS.get(target, target)
        out.setdefault((r["model"], r["num_ctx"], r.get("prompt", "short"), mode, device), {})[target] = {
            # Prompt tokens over the time the bench itself measured to the first one.
            #
            # NOT each engine's own prompt_tok_s, which is that engine's timer minus whatever
            # it does not count, and the engines do not leave out the same things: on a
            # 15-token prompt ollama reported 652 tok/s where its own time to first token
            # implies 35, and this engine 49 373 where its own implies 728. Published side by
            # side those reversed the verdict on three cells of four. One definition, measured
            # here, or the column is not a comparison.
            "prefill": measured_prefill(r, st),
            "decode":  stat(st, "Completion tok/s"),
            "e2e":     stat(st, "E2E latency"),
            # What the whole request cost. The per-token rate divides this by a token
            # count that is fixed per cell, so the two say the same thing at one size and
            # only the absolute one stays comparable when the size changes.
            "jreq":    (jtok * ntok) if (jtok and ntok) else None,
            "j":       jtok,
            "jdom":    jdom,
            "ntok":    ntok,
            # Whether the answer was an answer. A degenerate cell produces a perfectly
            # ordinary-looking rate - gemma4:31b emitted "--- --- ---" for its full 128
            # tokens and was tabled as -70% against ollama's prose - so the verdict has
            # to travel with the numbers, or the table states a speed for nothing.
            "coh":     r.get("coherence_pass"),
        }
    return out

def fmt(v, n=1):
    return " - " if v is None else f"{v:,.{n}f}".replace(",", " ")

def cells(merged):
    out = []
    for key, by in sorted(merged.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])):
        model, ctx, prompt, mode, device = key
        # Neither comparison survives the two engines answering at different lengths. A
        # model that stops on its own end token stops at a different point per engine, so
        # the energies then differ by the length of the answer rather than by efficiency -
        # and a rate averaged over a handful of steps carries the fixed cost of a request
        # instead of the decode: the same engine on the same model measured 398 tok/s over
        # 30 tokens and 417 over 128. Both deltas are therefore left blank unless the two
        # sides produced the same amount.
        mine = by.get("loken", {}).get("ntok")
        same = lambda v: mine and v["ntok"] and abs(v["ntok"] - mine) <= 0.02 * mine
        # An engine that answered with a repeating pattern is not a baseline and not a
        # result. Excluding it from the comparison matters in both directions: as the
        # reference it would flatter us, and as our own cell it would publish a delta
        # against work nobody did.
        ok = lambda v: v.get("coh") is not False
        others = [v["decode"] for k, v in by.items()
                  if k in ("ollama", "vllm") and v["decode"] and same(v) and ok(v)]
        # Comparable only when both sides counted the same domains: a total over the cards
        # alone against one that also counted the host is not a delta.
        mydom = by.get("loken", {}).get("jdom")
        othere = [v["jreq"] for k, v in by.items()
                  if k in ("ollama", "vllm") and v["jreq"] and same(v) and ok(v)
                  and v.get("jdom") == mydom]
        order = sorted(e for e in by if e.startswith("ollama@"))
        for eng in order + ["ollama", "vllm", "loken"]:
            if eng not in by: continue
            v = by[eng]; dd = de = ""
            if eng == "loken" and ok(v):
                if others and v["decode"]:
                    p = (v["decode"] / max(others) - 1) * 100
                    dd = f"**{p:+.1f}%**" if p > 0 else f"{p:+.1f}%"
                # Positive means loken spends less for the same request.
                if othere and v["jreq"]:
                    p = (min(othere) / v["jreq"] - 1) * 100
                    de = f"**{p:+.1f}%**" if p > 0 else f"{p:+.1f}%"
            b = (lambda x: f"**{x}**") if eng == "loken" else (lambda x: x)
            if not any(v[k] is not None for k in ("prefill", "decode", "ntok", "e2e", "jreq")):
                # Nothing was measured, so there is no row. An all-dash line reads as a cell
                # that was tried and yielded something unprintable, which is not what it means
                # and not something a reader can act on - absence already shows the coverage
                # gap. This is also what keeps locally requantized tags, which exist on one
                # machine and were never carried to a measurement, out of a published table.
                continue
            if not ok(v):
                # The rates are withheld rather than shown with a caveat: a number in a
                # performance column is read as performance, whatever sits beside it.
                out.append([model, str(ctx), prompt, mode, device, b(VERS[eng]), " - ", " - ",
                            b(fmt(v["ntok"], 0)), " - ", " - ", " - ", "incoherent", "", v.get("day", "")])
                continue
            if v["ntok"] is not None and v["ntok"] <= 1:
                # One token is not a rate. It divides by whatever interval the clock rounded
                # to, so the harness emits its sentinel - and the table published a million
                # tokens per second against a competitor, three times over, in cells where
                # our own was empty. A cell that produced no answer says so. What the request
                # itself cost still stands, because that part really was measured.
                out.append([model, str(ctx), prompt, mode, device, b(VERS[eng]), b(fmt(v["prefill"])),
                            " - ", b(fmt(v["ntok"], 0)), b(fmt(v["e2e"], 0)), b(fmt(v["jreq"], 0)),
                            " - ", "no answer", "", v.get("day", "")])
                continue
            out.append([model, str(ctx), prompt, mode, device, b(VERS[eng]), b(fmt(v["prefill"])),
                        b(fmt(v["decode"])), b(fmt(v["ntok"], 0)), b(fmt(v["e2e"], 0)),
                        b(fmt(v["jreq"], 0)), b(fmt(v["j"], 3)), dd, de, v.get("day", "")])
    return out

def table(rows_):
    w = [max(len(COLS[i]), *(len(r[i]) for r in rows_)) if rows_ else len(COLS[i])
         for i in range(len(COLS))]
    line = lambda c: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(c)) + " |"
    sep  = "|" + "|".join("-" * (w[i] + 2) for i in range(len(COLS))) + "|"
    return "\n".join([line(COLS), sep] + [line(r) for r in rows_])

def load_all(results_dir):
    merged = {}
    # When two files describe the same cell and the same engine, the newer measurement
    # wins. They are read in name order, so without this a standalone `*_stream_vllm.json`
    # from an older reconnaissance run overwrites the vLLM entry of the three-engine
    # `*_stream.json` measured minutes earlier - silently, and by four days. Caught the
    # first time a cell carried all three engines: the console said 85.1 tok/s and the
    # table printed 69.2.
    seen_at = {}
    for p in sorted(pathlib.Path(results_dir).glob("*.json")):
        mode = "non-stream" if "non-stream" in p.name else "stream"
        device = "CPU" if p.name.endswith("_cpu.json") else "GPU"
        # A cell measured against an older reference keeps its place beside the newer one:
        # the version column exists so both can be read, and dropping the earlier rows
        # would throw away the only evidence of what the upgrade changed.
        ver = json.load(open(p)).get("ollama_version")
        day = __import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        for k, v in rows(p, mode, device).items():
            if ver and "ollama" in v:
                tag = "ollama@" + ver
                v[tag] = v.pop("ollama")
                VERS[tag] = "Ollama " + ver
            for e in v.values():
                e["day"] = day
            slot = merged.setdefault(k, {})
            for engine, entry in v.items():
                mt = p.stat().st_mtime
                if seen_at.get((k, engine), -1) > mt:
                    continue
                seen_at[(k, engine)] = mt
                slot[engine] = entry
    return merged

if __name__ == "__main__":
    md, res = sys.argv[1], sys.argv[2]
    # A table is only readable if every row came from one build of the engine. Say so
    # out loud when they disagree, rather than letting the rows look comparable.
    builds = {}
    for f in pathlib.Path(res).glob("*.json"):
        if f.name.endswith("_vllm.json"):
            continue  # measures vLLM; the loken build is not part of it
        b = json.load(open(f)).get("engine_build")
        builds.setdefault(b, []).append(f.name)
    if len(builds) > 1:
        print("WARNING - the table mixes %d builds of the engine:" % len(builds), file=sys.stderr)
        for b, names in sorted(builds.items(), key=lambda x: (x[0] is None, x[0])):
            print("  build %s : %d cells (%s...)" % (b, len(names), names[0]), file=sys.stderr)
    # Rows go back into the sections they came from.
    #
    # This used to keep everything before the first "| Model" and replace the rest with one
    # flat table, which erased the per-architecture split and left nineteen of the twenty
    # figures pointing at nothing - once per measured cell, silently, for a whole campaign.
    #
    # Membership is read from the document rather than from a table kept here: which model
    # belongs to which family is already written, once, in the file being rewritten.
    p = pathlib.Path(md)
    doc = p.read_text()
    all_rows = cells(load_all(res))

    lines = doc.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("### ")]
    if not starts:
        # No sections: the flat form, kept working rather than turned into an error.
        head = doc[:doc.index("| Model")]
        p.write_text(head + table(all_rows) + "\n")
        print(table(all_rows))
        raise SystemExit(0)

    bounds = starts + [len(lines)]
    preamble = "\n".join(lines[:starts[0]])
    placed, out = set(), [preamble]
    # There is one unfiled section, never a new one per unclassified model. Its blocks are
    # collected here and re-emitted once at the end: a section whose only membership rule is
    # "no family claimed it" cannot claim a model the next run measures, so appending grew a
    # section per cell.
    for a, b in zip(starts, bounds[1:]):
        block = lines[a:b]
        if block[0].strip() == "### unfiled":
            continue
        # Everything down to the header row is the section's own: its title, its prose, its
        # figure. Only the rows are regenerated.
        keep = []
        models = []
        for ln in block:
            if ln.startswith("| Model") or ln.startswith("|---"):
                continue
            if ln.startswith("| "):
                models.append(ln.split("|")[1].strip())
                continue
            keep.append(ln)
        while keep and not keep[-1].strip():
            keep.pop()
        mine = [r for r in all_rows if r[0].strip() in set(models)]
        placed.update(r[0].strip() for r in mine)
        out.append("\n".join(keep) + "\n\n" + table(mine))

    orphans = [r for r in all_rows if r[0].strip() not in placed]
    if orphans:
        # Named and shown rather than dropped: a model measured into no section would
        # otherwise vanish from the report while its result file sat in results/.
        names = sorted({r[0].strip() for r in orphans})
        print("WARNING - %d model(s) belong to no section, filed at the end: %s"
              % (len(names), ", ".join(names)), file=sys.stderr)
        out.append("### unfiled\n\nMeasured, and not yet placed in a family section.\n\n"
                   + table(orphans))

    p.write_text("\n\n".join(out) + "\n")
    print(table(all_rows))
