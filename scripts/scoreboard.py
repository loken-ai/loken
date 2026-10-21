"""Answer the campaign's actual question: how many cells win on BOTH axes.

The table states a delta per cell and reading 200 rows to count them is how a regression
hides. The goal was never an average - it is every cell, on throughput AND on energy - so
the count that matters is the one that separates "faster but hungrier" from "won".

Deltas come from bench_row itself rather than a second computation here. That is what keeps
this honest: bench_row blanks a delta when the two engines produced different token counts,
and withholds one entirely when the answer was degenerate. An independent parser would have
to re-derive both rules and would eventually disagree - I published a -69.7% from an ad-hoc
script that was comparing 4 tokens against 128 while the table, correctly, showed nothing.
"""
import sys
import json
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_row as B


def vintage(results_dir):
    """How many result files came from the binary that is on disk right now.

    A campaign runs for days and the engine changes under it. Mixing vintages is not a
    detail: the same 'long' cells read 87% won under the old protocol and 59% under the
    corrected one, and I published the 87% as a trend twice before checking. The table's
    header already warns when builds are mixed; the score has to say it too, because the
    score is what gets quoted.
    """
    binary = pathlib.Path(results_dir).parent / "target/release/lokend"
    if not binary.exists():
        return None
    stamp = int(binary.stat().st_mtime)
    fresh = total = 0
    for p in pathlib.Path(results_dir).glob("*.json"):
        try:
            if not p.stat().st_size:
                continue
            total += 1
            fresh += json.load(open(p)).get("engine_build") == stamp
        except Exception:
            pass
    return fresh, total


TEMPS = (("cold", 0, "first request of a cell, no prompt cache hit"),
         ("warm", 1, "median of the later iterations, cache served where the engine caches"))


def tally(detail, t):
    """Count the cells of one temperature (0 cold, 1 warm) that win, and why the others do not.

    Who was degenerate matters. A cell we cannot publish because OUR answer collapsed is a
    defect; one we cannot publish because the PEER's did is the gate refusing us a free
    win - qwen3:0.6b on the raw code-review prompt, where ollama emitted nothing but code
    fences and we answered. A cell where BOTH answers collapsed accuses neither engine: it is
    the model, or the prompt. A cell whose loken row ran a single iteration has cold only.
    """
    # A variant of this engine - the same binary plus a drafter - shares the cell key with the
    # plain row and is a documented extra, not the engine being scored.
    ours = {i["key"] for _, i in detail if i["plain"] and i["mark"] == "incoherent"}
    peers = {i["key"] for r, i in detail if "loken" not in r[5] and i["mark"] == "incoherent"}
    both = ours & peers
    c = dict(won=0, lost_energy=0, lost_decode=0, ours=0, peer=0, both=0, uneven=0, single=0, losses=[])
    for r, i in detail:
        if not i["plain"]:
            continue
        d, e = i["dec"][t], i["energy"]
        if d is None or e is None:
            if t and i["dec"][0] is not None and i["dec"][1] is None and e is not None:
                c["single"] += 1
            elif i["key"] in both:
                c["both"] += 1
            elif i["key"] in ours:
                c["ours"] += 1
            elif i["key"] in peers:
                c["peer"] += 1
            else:
                c["uneven"] += 1
            continue
        where = f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]}"
        if d > 0 and e > 0:
            c["won"] += 1
        elif d > 0:
            c["lost_energy"] += 1
            c["losses"].append((e, where, d, e))
        else:
            c["lost_decode"] += 1
            c["losses"].append((d if e > 0 else min(d, e), where, d, e))
    c["total"] = c["won"] + c["lost_energy"] + c["lost_decode"]
    return c


def main(results_dir, brief=False):
    detail = B.cells_detail(B.load_all(results_dir))
    counts = [(name, what, tally(detail, t)) for name, t, what in TEMPS]
    v = vintage(results_dir)
    age = "" if not v or v[0] == v[1] else f"   [{v[0]}/{v[1]} from the current binary]"
    complete = all(c["total"] and c["won"] == c["total"] for _, _, c in counts)
    if brief:
        # Two lines after every cell, because a campaign that only scores itself at the end
        # cannot tell a correction from a regression while there is still time to act.
        for name, _, c in counts:
            print(f"  goal {name}: {c['won']}/{c['total']} cells win on BOTH axes"
                  f"   ({c['lost_decode']} throughput, {c['lost_energy']} energy"
                  f" | blank: {c['ours']} ours, {c['peer']} peer, {c['both']} both,"
                  f" {c['uneven']} uneven tokens, {c['single']} cold only){age}")
        return 0 if complete else 1
    if v and v[0] != v[1]:
        print(f"  WARNING: {v[1]-v[0]} of {v[1]} files come from an earlier binary.")
        print(f"  A trend read across a mixture of protocols is not a trend.")
    for name, what, c in counts:
        total = c["total"]
        print(f"\n  {name.upper()}: {what}")
        print(f"  {c['won']}/{total} cells win on BOTH axes" + (f"   ({c['won']*100//total}%)" if total else ""))
        print(f"  lost on energy alone    {c['lost_energy']}")
        print(f"  lost on throughput      {c['lost_decode']}")
        print(f"  blank, our answer       {c['ours']}   (degenerate: the cell is a defect)")
        print(f"  blank, peer answer      {c['peer']}   (degenerate: not a free win)")
        print(f"  blank, both answers     {c['both']}   (degenerate on both: the model or the prompt)")
        print(f"  blank, uneven tokens    {c['uneven']}")
        if name == "warm":
            print(f"  cold only               {c['single']}   (a single iteration: no warm figure)")
        if c["losses"]:
            print("\n  the 12 largest gaps:")
            for _, where, d, e in sorted(c["losses"])[:12]:
                print(f"    {where:<52} throughput {d:+7.1f}%   energy {e:+7.1f}%")
    return 0 if complete else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--brief"]
    sys.exit(main(args[0] if args else "results", brief="--brief" in sys.argv))
