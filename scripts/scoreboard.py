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


def pct(s):
    """A delta cell as a number. Blank means the cell was not comparable."""
    s = (s or "").replace("*", "").replace("%", "").strip()
    if not s or s == "incoherent":
        return None
    try:
        return float(s)
    except ValueError:
        return None


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


def main(results_dir, brief=False):
    rows = B.cells(B.load_all(results_dir))
    # Who was degenerate matters. A cell we cannot publish because OUR answer collapsed is a
    # defect; one we cannot publish because the PEER's did is the gate refusing us a free
    # win - qwen3:0.6b on the raw code-review prompt, where ollama emitted nothing but code
    # fences and we answered. Counting both as "incomparable" hides the difference, and I
    # reported "six degenerate models" when five were ours and the sixth was ollama's.
    ours = {tuple(r[:5]) for r in rows if "loken" in r[5] and r[12] == "incoherent"}
    peers = {tuple(r[:5]) for r in rows if "loken" not in r[5] and r[12] == "incoherent"}
    # A cell where BOTH answers collapsed accuses neither engine: it is the model, or the
    # prompt it was given. Counting it as ours read as a defect of ours - gemma4:26b, where
    # ollama looped on "er_er_er" and we looped on "misterer/".
    both = ours & peers
    won = tied = lost_energy = lost_decode = 0
    bad_ours = bad_peer = bad_both = uneven = 0
    losses = []
    for r in rows:
        if "loken" not in r[5]:
            continue
        d, e = pct(r[12]), pct(r[13])
        if d is None or e is None:
            key = tuple(r[:5])
            if key in both:
                bad_both += 1
            elif key in ours:
                bad_ours += 1
            elif key in peers:
                bad_peer += 1
            else:
                uneven += 1
            continue
        where = f"{r[0]} {r[1]} {r[2]} {r[3]} {r[4]}"
        if d > 0 and e > 0:
            won += 1
        elif d <= 0 and e <= 0:
            lost_decode += 1
            losses.append((min(d, e), where, d, e))
        elif d <= 0:
            lost_decode += 1
            losses.append((d, where, d, e))
        else:
            lost_energy += 1
            losses.append((e, where, d, e))
    total = won + lost_energy + lost_decode
    if brief:
        # One line, after every cell, because a campaign that only scores itself at the end
        # cannot tell a correction from a regression while there is still time to act.
        v = vintage(results_dir)
        age = "" if not v or v[0] == v[1] else f"   [{v[0]}/{v[1]} from the current binary]"
        print(f"  goal: {won}/{total} cells win on BOTH axes"
              f"   ({lost_decode} throughput, {lost_energy} energy"
              f" | blank: {bad_ours} ours, {bad_peer} peer, {bad_both} both,"
              f" {uneven} uneven tokens){age}")
        return 0 if total and won == total else 1
    v = vintage(results_dir)
    if v and v[0] != v[1]:
        print(f"  WARNING: {v[1]-v[0]} of {v[1]} files come from an earlier binary.")
        print(f"  A trend read across a mixture of protocols is not a trend.")
    print(f"  comparable cells        {total}")
    print(f"  won on BOTH axes        {won}" + (f"   ({won*100//total}%)" if total else ""))
    print(f"  lost on energy alone    {lost_energy}")
    print(f"  lost on throughput      {lost_decode}")
    print(f"  blank, our answer       {bad_ours}   (degenerate: the cell is a defect)")
    print(f"  blank, peer answer      {bad_peer}   (degenerate: not a free win)")
    print(f"  blank, both answers     {bad_both}   (degenerate on both: the model or the prompt)")
    print(f"  blank, uneven tokens    {uneven}")
    if losses:
        print("\n  les 12 ecarts les plus grands:")
        for _, where, d, e in sorted(losses)[:12]:
            print(f"    {where:<52} throughput {d:+7.1f}%   energy {e:+7.1f}%")
    return 0 if total and won == total else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--brief"]
    sys.exit(main(args[0] if args else "results", brief="--brief" in sys.argv))
