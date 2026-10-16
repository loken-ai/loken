"""Turn bench JSONs into the report's table, aligned, with nothing retyped."""
import re
import datetime
import json, os, pathlib, struct, sys


def _measured_at(doc, p):
    """When a cell was measured, as (epoch, "YYYY-MM-DD HH:MM"), from the file's OWN `timestamp` -
    not the file's mtime, which any later touch (a version stamp, a copy, a resave) rewrites and
    which then reads as if every historical bench were run today. Falls back to mtime only for a
    file that recorded no timestamp of its own."""
    ts = doc.get("timestamp")
    if ts:
        try:
            # ISO 8601 with offset; trim sub-microsecond digits so fromisoformat accepts it.
            dt = datetime.datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", ts))
            return dt.timestamp(), dt.astimezone().strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    try:
        m = pathlib.Path(p).stat().st_mtime
    except OSError:
        return 0.0, ""
    return m, datetime.datetime.fromtimestamp(m).strftime("%Y-%m-%d %H:%M")

# Engine version labels come from the result FILES, never from whatever happens to be installed
# when the report is generated: a report built after an engine upgrade must not stamp the new
# version onto rows measured with the old one. Each file records `ollama_version`/`vllm_version`
# for the engines that measured it (load_all folds them in as `ollama@<ver>`/`vllm@<ver>`); a file
# that recorded none is labelled without a number rather than with a guess. That is why there is no
# call here that asks the installed binary its version - that call was the bug.
VERS = {"ollama": "Ollama", "vllm": "vLLM", "loken": "loken"}

_DEV_VERSIONS = {}
def _dev_version_at(epoch):
    """The version a loken row that recorded none was built as: the Cargo.toml version of the
    last commit before it was measured, as the pre-release it was (no release tag carries it).
    None when the repository cannot answer."""
    import subprocess
    day = datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")
    if day not in _DEV_VERSIONS:
        try:
            root = pathlib.Path(__file__).resolve().parent.parent
            rev = subprocess.run(["git", "-C", str(root), "rev-list", "-1", "--before=" + day, "HEAD"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            toml = subprocess.run(["git", "-C", str(root), "show", rev + ":Cargo.toml"],
                                  capture_output=True, text=True, timeout=10).stdout if rev else ""
            m = re.search(r'^version\s*=\s*"([^"]+)"', toml, re.M)
            _DEV_VERSIONS[day] = m.group(1) + "-dev" if m else None
        except Exception:
            _DEV_VERSIONS[day] = None
    return _DEV_VERSIONS[day]

def engine_label(target, version=None):
    """The Engine column for a target. A variant of this engine is measured under the label
    "LOKEN+<drafter>" - the same binary with a drafter named in its configuration - and is
    shown as "loken <version> + <drafter>" so a row can hold both without the two being confused,
    and so the scoreboard can tell them apart by the " + ". `version` is what the server said
    when the row was measured; a row that recorded none is labelled without a number."""
    base = VERS["loken"] + (" " + version if version else "")
    if target == "loken":
        return base
    if target in VERS:
        return VERS[target]
    if target.startswith("loken+"):
        return base + " + " + target[len("loken+"):]
    return target

# Results measured before the project was renamed carry the old target name. They are the same
# engine and the same campaign, so they are folded in rather than dropped - reading only the
# current name would empty most of the table without saying anything was missing.
RETIRED_TARGETS = {"llmuse": "loken", "llmserver": "loken"}
COLS = ["Model", "Ctx", "Prompt", "Mode", "Device", "Engine", "Prefill tok/s", "Decode tok/s", "Tokens",
        "E2E ms", "J/req", "J/token", "Δ decode", "Δ energy", "Date", "Run"]

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

# Which family a model belongs to is a property of the model file - `general.architecture`
# in its GGUF - not of the rows a previous run happened to leave behind. Reading it from the
# document works only while the document already has rows: emptying it to re-measure left
# every model unfiled, twenty family sections without a single row, and their twenty figures
# pointing at nothing - the defect those sections were built to remove.

_ARCH_CACHE = {}


def _gguf_architecture(blob):
    """`general.architecture` from a GGUF header, reading only what it takes to find it."""
    with open(blob, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        struct.unpack("<I", f.read(4))[0]                    # version
        struct.unpack("<Q", f.read(8))[0]                    # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]
        rd = lambda n: f.read(n)
        def s():
            return rd(struct.unpack("<Q", rd(8))[0]).decode("utf-8", "replace")
        SCALAR = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        def skip(t):
            if t == 8:
                s()
            elif t == 9:
                et = struct.unpack("<I", rd(4))[0]
                n = struct.unpack("<Q", rd(8))[0]
                for _ in range(n):
                    skip(et)
            else:
                rd(SCALAR[t])
        for _ in range(n_kv):
            key = s()
            t = struct.unpack("<I", rd(4))[0]
            if key == "general.architecture" and t == 8:
                return s()
            skip(t)
    return None

def architecture_of(tag, store):
    """The architecture an ollama tag declares, or None when it cannot be resolved."""
    name, _, ver = tag.partition(":")
    ver = ver or "latest"
    ns = "library"
    if "/" in name:
        ns, name = name.split("/", 1)
    man = pathlib.Path(store) / "manifests" / "registry.ollama.ai" / ns / name / ver
    if not man.is_file():
        return None
    try:
        layers = json.load(open(man)).get("layers", [])
    except Exception:
        return None
    for l in layers:
        if l.get("mediaType", "").endswith(".image.model"):
            digest = l["digest"].replace(":", "-")
            blob = pathlib.Path(store) / "blobs" / digest
            if blob.is_file():
                try:
                    return _gguf_architecture(blob)
                except Exception:
                    return None
    return None

def arch_of(tag):
    """Cached, and silent about what it cannot resolve: a locally requantised tag has no
    manifest, and the membership read from the document still covers it."""
    if tag not in _ARCH_CACHE:
        store = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
        try:
            _ARCH_CACHE[tag] = architecture_of(tag, store)
        except Exception:
            _ARCH_CACHE[tag] = None
    return _ARCH_CACHE[tag]

def looks_looping(text):
    """A loop that emits no whitespace, which the harness's gate cannot see.

    Every check in assay's `looks_degenerate` splits on whitespace first, so a 128-token
    answer that is one enormous word - "arousingerser_er_er_er...", "misterer/misterer/" -
    has fewer than twelve units and is declared fine. Both engines produced exactly that on
    gemma4:26b and both passed.

    Same reasoning as the harness's phrase check, one level down: distinct character windows
    over total lands near 1.0 for prose and collapses toward k/n for anything cycling with
    period k, whatever k is. Judged here as well as there so that every cell in the report -
    including those measured before the rule existed - is judged by one rule.
    """
    t = "".join(text.split())
    if len(t) < 40:
        return False
    # A long cycle: distinct windows over total collapses toward k/n whatever the period k is.
    w = [t[i:i + 8] for i in range(len(t) - 7)]
    if len(set(w)) / len(w) < 0.5:
        return True
    # A sane opening followed by a looping tail keeps that share high, because the head
    # supplies the distinct windows - the same escape the harness found at word level. What
    # gives it away is one short pattern repeating back to back over much of the answer.
    for k in range(2, 17):
        i = 0
        while i + k <= len(t):
            reps = 1
            while t[i + reps * k: i + (reps + 1) * k] == t[i:i + k]:
                reps += 1
            if reps >= 3 and reps * k * 10 >= len(t) * 3:
                return True
            i += 1
    # NEAR-repetition, which neither check above can see because nothing repeats exactly:
    # "Once up to once, Once up once, Once up once. Once said that once upon a day." Every
    # window differs slightly and no run is byte-identical, but the vocabulary has collapsed.
    # Measured on this campaign's own previews: prose runs 0.75 to 1.00 distinct words, the
    # answers that are visibly circling run 0.33 to 0.45. The line is drawn between them.
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) >= 16 and len(set(words)) / len(words) < 0.55:
        return True
    return False


def stat(st, name):
    """The harness's own aggregate, so the report never defines a second one."""
    v = st.get(name)
    if v is None: return None
    return v.get("mean", v.get("value")) if isinstance(v, dict) else v

def stat_warm(st, name):
    """The median, for a figure whose first iteration is a cold load.

    The same rule `measured_prefill` applies, for the same reason, against an assumption
    stated there that does not hold: decode was taken as the mean because it was thought
    not to carry the cold outlier. On a large mixture it does. qwen3next measured
    3.8, 15.2, 15.2 tok/s - stddev 6.6 on a mean of 11.4 - and the table published 11.4
    for a model that decodes at 15.2, a quarter of its rate spent on a warm-up the other
    engine does not pay in the same cell.

    Every cell whose iterations are flat is unchanged, which is all of them but the
    mixtures too large to hold: on those the mean measures the repack, not the engine.
    """
    v = st.get(name)
    if v is None: return None
    if not isinstance(v, dict): return v
    return v.get("p50", v.get("mean", v.get("value")))

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
    # The median, for the reason `stat_warm` gives: the first iteration pays for loading
    # the weights, and that energy lands in the same per-token average as the decode it
    # is not part of. It inflates BOTH engines and not by the same factor - on qwen3:0.6b
    # ollama's mean is 2.4x its median and this engine's 1.24x - so the delta it produced
    # was not a smaller version of the truth but a different number: +383.6% published
    # where the steady state is +150%. A claim that flatters us by construction is the one
    # to remove first.
    return stat_warm(st, hits[0]), dom

def rows(d, mode, device):
    path = d.get("converted_from") or d.get("timestamp")
    out = {}
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
            "decode":  stat_warm(st, "Completion tok/s"),
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
            "preview": r.get("first_response_preview") or "",
            # An engine that could not take the model at all, or one that took it and kept no
            # iteration: either way no rate, and the result file carries the reason.
            "impossible": bool(r.get("load_error")),
            "failed": not (r.get("iterations") or []) and not r.get("load_error"),
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
        ok = lambda v: v.get("coh") is not False and not looks_looping(v.get("preview", ""))
        others = [v["decode"] for k, v in by.items()
                  if k.split("@")[0] in ("ollama", "vllm") and v["decode"] and same(v) and ok(v)]
        # Comparable only when both sides counted the same domains: a total over the cards
        # alone against one that also counted the host is not a delta.
        mydom = by.get("loken", {}).get("jdom")
        othere = [v["jreq"] for k, v in by.items()
                  if k.split("@")[0] in ("ollama", "vllm") and v["jreq"] and same(v) and ok(v)
                  and v.get("jdom") == mydom]
        ollama_cols = sorted(e for e in by if e.startswith("ollama@"))
        vllm_cols = sorted(e for e in by if e.startswith("vllm@"))
        variants = sorted(e for e in by if e.startswith("loken+"))
        for eng in ollama_cols + ["ollama"] + vllm_cols + ["vllm", "loken"] + variants:
            if eng not in by: continue
            v = by[eng]; dd = de = ""
            if eng.startswith("loken") and ok(v):
                if others and v["decode"]:
                    p = (v["decode"] / max(others) - 1) * 100
                    dd = f"**{p:+.1f}%**" if p > 0 else f"{p:+.1f}%"
                # Positive means loken spends less for the same request.
                if othere and v["jreq"]:
                    p = (min(othere) / v["jreq"] - 1) * 100
                    de = f"**{p:+.1f}%**" if p > 0 else f"{p:+.1f}%"
            # Bold a loken cell to set our rows apart, but never an empty one: `**` around a
            # spaced dash (`** - **`) is not valid emphasis and shows the asterisks verbatim.
            b = lambda x: f"**{x}**" if eng.startswith("loken") and str(x).strip() not in ("-", "") else str(x)
            if v.get("impossible") or v.get("failed"):
                # The engine was given the model and could not run it, or ran it and kept no
                # iteration: the row says which, since a comparison that leaves a cell out reads
                # as a cell that was not tried.
                mark = "impossible" if v.get("impossible") else "failed"
                out.append([model, str(ctx), prompt, mode, device, b(engine_label(eng, v.get("version"))), " - ", " - ",
                            " - ", " - ", " - ", " - ", mark, "", v.get("day", ""), v.get("run", "")])
                continue
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
                out.append([model, str(ctx), prompt, mode, device, b(engine_label(eng, v.get("version"))), " - ", " - ",
                            b(fmt(v["ntok"], 0)), " - ", " - ", " - ", "incoherent", "", v.get("day", ""), v.get("run", "")])
                continue
            if v["ntok"] is not None and v["ntok"] <= 1:
                # One token is not a rate. It divides by whatever interval the clock rounded
                # to, so the harness emits its sentinel - and the table published a million
                # tokens per second against a competitor, three times over, in cells where
                # our own was empty. A cell that produced no answer says so. What the request
                # itself cost still stands, because that part really was measured.
                out.append([model, str(ctx), prompt, mode, device, b(engine_label(eng, v.get("version"))), b(fmt(v["prefill"])),
                            " - ", b(fmt(v["ntok"], 0)), b(fmt(v["e2e"], 0)), b(fmt(v["jreq"], 0)),
                            " - ", "no answer", "", v.get("day", ""), v.get("run", "")])
                continue
            out.append([model, str(ctx), prompt, mode, device, b(engine_label(eng, v.get("version"))), b(fmt(v["prefill"])),
                        b(fmt(v["decode"])), b(fmt(v["ntok"], 0)), b(fmt(v["e2e"], 0)),
                        b(fmt(v["jreq"], 0)), b(fmt(v["j"], 3)), dd, de, v.get("day", ""), v.get("run", "")])
    return out

def table(rows_):
    w = [max(len(COLS[i]), *(len(r[i]) for r in rows_)) if rows_ else len(COLS[i])
         for i in range(len(COLS))]
    line = lambda c: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(c)) + " |"
    sep  = "|" + "|".join("-" * (w[i] + 2) for i in range(len(COLS))) + "|"
    return "\n".join([line(COLS), sep] + [line(r) for r in rows_])

def mode_device(doc):
    """(mode, device) of a record, from what it states: a run that asked for no streaming is
    non-stream, one that gave the engine no card (`num_gpu` 0) is CPU. Both defaults are the
    campaign's own."""
    cfg = doc.get("config") or {}
    mode = "non-stream" if cfg.get("streaming") is False else "stream"
    return mode, "CPU" if cfg.get("num_gpu") == 0 else "GPU"


def load_all(results_dir):
    """The cells of every result file in a directory."""
    docs = [(json.load(open(p)), p, "") for p in sorted(pathlib.Path(results_dir).glob("*.json"))]
    return load_docs(docs)


def load_docs(docs):
    """The cells of records: (record, source, run id) triples, in any order.

    When two records describe the same cell and the same engine, the newer measurement wins,
    ranked by when it was MEASURED (the record's timestamp), not by the file's mtime - a
    version stamp touches mtime without re-measuring. Without this a standalone vLLM
    reconnaissance run overwrote the vLLM entry of the three-engine run measured minutes
    earlier, silently and by four days: the console said 85.1 tok/s and the table printed 69.2.
    """
    merged = {}
    seen_at = {}
    for doc, source, run in docs:
        mode, device = mode_device(doc)
        # A cell measured against an older reference keeps its place beside the newer one:
        # the version column exists so both can be read, and dropping the earlier rows
        # would throw away the only evidence of what the upgrade changed.
        # The version each engine was measured with, read from the record. An engine whose row
        # is here but whose version the record did not state keeps its bare key ("ollama"/"vllm")
        # and is labelled without a number - never with the version installed now.
        said = lambda eng: ((doc.get("subjects") or {}).get(eng) or {}).get("says") or {}
        engine_vers = (("ollama", said("ollama").get("version") or doc.get("ollama_version"), "Ollama"),
                       ("vllm", said("vllm").get("version") or doc.get("vllm_version"), "vLLM"))
        # loken keeps its key (the deltas compare against it); its version rides on the entry.
        meas_epoch, day = _measured_at(doc, source)
        loken_ver = said("loken").get("version") or _dev_version_at(meas_epoch)
        for k, v in rows(doc, mode, device).items():
            for eng, ver, label in engine_vers:
                if ver and eng in v:
                    tag = eng + "@" + ver
                    v[tag] = v.pop(eng)
                    VERS[tag] = label + " " + ver
            for e in v.values():
                e["day"] = day
                e["run"] = run[:8]
            for eng, e in v.items():
                if eng.startswith("loken") and loken_ver:
                    e["version"] = loken_ver
            slot = merged.setdefault(k, {})
            for engine, entry in v.items():
                if seen_at.get((k, engine), -1) > meas_epoch:
                    continue
                seen_at[(k, engine)] = meas_epoch
                slot[engine] = entry
    return merged
