#!/usr/bin/env python3
"""Regenerate the figures the docs embed, and the one table that must match them.

Run from the repository root: `python3 scripts/figures.py`.

Every figure has a source in the tree - a `.dot` for the graphs, this file for the chart - so a
diff shows what changed rather than that some bytes did. The graphs need graphviz; the chart
needs nothing but the standard library.

The comparison chart and the table in `docs/STATUS.md` are produced from the SAME parse of
`docs/BENCHMARKS.md`, because they disagreed once: the status page quoted a parity for
granite3-moe that appears nowhere in the measurements, and reasoned from it.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"

# The run every figure describes: streamed, short prompt, and the LATEST campaign day that
# measured both engines on it. Derived rather than written down: the date used to be a
# constant here, and the figures kept describing a run from a fortnight before the table
# they sat next to. RUN_DATE=YYYY-MM-DD in the environment pins another day; RUN_CARDS names
# the card policy in the chart header, both cards being what a campaign runs with.
import os
PROMPT = "short"
MODE = "stream"
RUN_CARDS = os.environ.get("RUN_CARDS", "both cards")


def latest_run_date():
    """The most recent date on which a short/stream row exists for both engines."""
    clean = lambda s: re.sub(r"[*\s  ]", "", s)
    seen = {}
    for line in (ROOT / "docs" / "BENCHMARKS.md").read_text().splitlines():
        if not line.startswith("| ") or line.startswith("|--"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 15 or c[0] == "Model" or c[2] != PROMPT or c[3] != MODE:
            continue
        seen.setdefault(c[14][:10], set()).add(clean(c[5]))
    days = [d for d, engs in seen.items() if "Ollama0.32.6" in engs and "loken0.1.0" in engs]
    return max(days) if days else ""


RUN_DATE = os.environ.get("RUN_DATE") or latest_run_date()


def bench_rows():
    """(model, engine) -> decode tok/s, for the run above."""
    clean = lambda s: re.sub(r"[*\s  ]", "", s)
    out = {}
    for line in (ROOT / "docs" / "BENCHMARKS.md").read_text().splitlines():
        if not line.startswith("| ") or line.startswith("|--"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 15 or c[0] == "Model":
            continue
        if c[14][:10] != RUN_DATE or c[2] != PROMPT or c[3] != MODE:
            continue
        try:
            decode = float(clean(c[7]))
        except ValueError:
            continue
        out.setdefault(c[0], {})[clean(c[5])] = decode
    return {
        m: (e["Ollama0.32.6"], e["loken0.1.0"])
        for m, e in out.items()
        if "Ollama0.32.6" in e and "loken0.1.0" in e
    }


def comparison_table(rows):
    """The markdown STATUS.md carries, sorted by how this engine fares."""
    lines = ["| model | ollama | loken | |", "|---|---:|---:|---|"]
    for model, (ollama, loken) in sorted(
        rows.items(), key=lambda kv: -(kv[1][1] / kv[1][0])
    ):
        delta = (loken / ollama - 1) * 100
        # Bold whichever engine won, and mark a loss so it cannot be skimmed past.
        o = f"**{ollama:.1f}**" if ollama > loken else f"{ollama:.1f}"
        l = f"**{loken:.1f}**" if loken > ollama else f"{loken:.1f}"
        mark = f"{delta:+.0f}%" if abs(delta) >= 1 else "-"
        if delta <= -1:
            mark = f"**{mark}**"
        lines.append(f"| {model} | {o} | {l} | {mark} |")
    return "\n".join(lines)


def bar_chart(rows, path, header=None):
    """Paired bars, one model per row, widest scale set by the fastest engine.

    Hand-written rather than plotted by a library: the repository adds a dependency only for a
    reason, and this is a hundred lines of arithmetic.
    """
    data = sorted(rows.items(), key=lambda kv: -(kv[1][1] / kv[1][0]))
    top = max(max(v) for v in rows.values())
    row_h, bar_h, gap, pad = 34, 11, 3, 16
    # Both margins follow the text they hold. Fixed, they clipped
    # `nemotron-3-nano:latest` to `emotron-3-nano:latest` on the left and `674 (+12%)` to
    # `674 (+12%` on the right. 7.4px per character is what the sans stack below takes at
    # 12.5px, plus the gap to the axis and a margin so nothing starts at zero.
    label_w = int(max(len(m) for m in rows) * 6.5) + 22
    chart_w = 420
    value_w = (
        max(len(f"{l:.0f}  ({(l / o - 1) * 100:+.0f}%)") for o, l in rows.values()) * 7 + 12
    )
    header = header or f"DECODE TOK/S, {RUN_CARDS.upper()}, {RUN_DATE}"
    width = max(label_w + chart_w + value_w, label_w + int(len(header) * 7.0) + 20)
    height = pad * 2 + 34 + row_h * len(data) + 30

    def bar(x, y, w, h, cls):
        return f'<rect class="{cls}" x="{x}" y="{y}" width="{max(w, 1):.1f}" height="{h}" rx="2"/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="DejaVu Sans, Liberation Sans, Helvetica, Arial, sans-serif">',
        # Mid-tones legible on a white page and on a dark one, and redefined when the viewer
        # says which it is. A figure that only reads on one background is half a figure.
        "<style>"
        ".t{fill:#57606a;font-size:10.5px}"
        ".m{fill:#24292f;font-size:11px}"
        ".h{fill:#57606a;font-size:11px;letter-spacing:.06em}"
        ".o{fill:#8c959f}"
        ".l{fill:#2da44e}"
        ".ax{stroke:#d0d7de;stroke-width:1}"
        "@media (prefers-color-scheme:dark){"
        ".t,.h{fill:#8b949e}.m{fill:#e6edf3}.o{fill:#6e7681}.l{fill:#3fb950}"
        ".ax{stroke:#30363d}}"
        "</style>",
        # The date is on the figure because these numbers are old and untrusted, and a chart
        # travels away from the paragraph that says so.
        f'<text class="h" x="{label_w}" y="{pad + 6}">{header}</text>',
    ]
    y0 = pad + 22
    svg.append(f'<line class="ax" x1="{label_w}" y1="{y0}" x2="{label_w}" y2="{height - pad}"/>')
    for i, (model, (ollama, loken)) in enumerate(data):
        y = y0 + i * row_h + 6
        svg.append(f'<text class="m" x="{label_w - 10}" y="{y + 12}" text-anchor="end">{model}</text>')
        svg.append(bar(label_w, y, ollama / top * chart_w, bar_h, "o"))
        svg.append(bar(label_w, y + bar_h + gap, loken / top * chart_w, bar_h, "l"))
        svg.append(
            f'<text class="t" x="{label_w + ollama / top * chart_w + 6}" y="{y + 10}">'
            f"{ollama:.0f}</text>"
        )
        delta = (loken / ollama - 1) * 100
        svg.append(
            f'<text class="t" x="{label_w + loken / top * chart_w + 6}" y="{y + bar_h + gap + 12}">'
            f"{loken:.0f}  ({delta:+.0f}%)</text>"
        )
    svg.append(
        f'<text class="h" x="{label_w}" y="{height - 14}">'
        "GREY OLLAMA 0.32.6   GREEN LOKEN 0.1.0</text>"
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def section_cells():
    """Every comparable cell, grouped by the section it sits under.

    A cell is (model, ctx, prompt, mode, device) on a date, and it is comparable when both
    engines produced a decode rate for it. Returns per section the comparable cells and how
    many cells the section holds in total, because a figure that silently drops most of its
    data is worse than no figure - the first attempt charted 5 sections out of 20 and showed
    one slice of each.
    """
    clean = lambda s: re.sub(r"[*\s  ]", "", s)
    section, cells, total = None, {}, {}
    for line in (ROOT / "docs" / "BENCHMARKS.md").read_text().splitlines():
        if line.startswith("### "):
            section = line[4:].strip()
            cells.setdefault(section, {})
            total.setdefault(section, set())
            continue
        if section is None or not line.startswith("| ") or line.startswith("| Model"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 15:
            continue
        key = (c[0], c[1], c[2], c[3], c[4], c[14][:10])
        total[section].add(key)
        try:
            decode = float(clean(c[7]))
        except ValueError:
            continue
        cells[section].setdefault(key, {})[clean(c[5])] = decode
    out = {}
    for sec, found in cells.items():
        pairs = {
            f"{m} {ctx} {prompt} {'S' if mode == 'stream' else 'NS'} {dev}": (
                e["Ollama0.32.6"],
                e["loken0.1.0"],
            )
            for (m, ctx, prompt, mode, dev, _), e in found.items()
            if "Ollama0.32.6" in e and "loken0.1.0" in e
        }
        if pairs:
            out[sec] = (pairs, len(total[sec]))
    return out


def section_charts():
    """One figure per section, one bar pair per comparable cell.

    Anything no longer produced is removed first: a generated file left behind after the rule
    that made it changed keeps being embedded under a caption that stopped describing it.
    """
    for stale in IMG.glob("family-*.svg"):
        stale.unlink()
    made = []
    for sec, (pairs, total) in sorted(section_cells().items()):
        # Ten cells, by RELATIVE gap - an absolute one would rank a 70B model's 1-against-2
        # tok/s below a 1B model's noise. Each model's widest gap is taken first, then the
        # widest remaining: ranking on the gap alone gave six bars of one checkpoint at
        # +167, +167, +167, +166, +164, +162 percent, which is one fact drawn six times while
        # the other models in the section went unshown.
        gap = lambda kv: -abs(kv[1][1] / kv[1][0] - 1)
        ranked = sorted(pairs.items(), key=gap)
        widest_per_model, spare = {}, []
        for label, values in ranked:
            model = label.split(" ", 1)[0]
            if model in widest_per_model:
                spare.append((label, values))
            else:
                widest_per_model[model] = (label, values)
        shown = dict(list(widest_per_model.values())[:10])
        for label, values in spare[: max(0, 10 - len(shown))]:
            shown[label] = values
        shown = dict(sorted(shown.items(), key=gap))
        header = f"{sec.upper()} - DECODE TOK/S, {len(shown)}"
        if len(shown) < len(pairs):
            header += f" WIDEST GAPS OF {len(pairs)} COMPARABLE"
        else:
            header += f" COMPARABLE"
        header += f" CELLS, {total} MEASURED"
        bar_chart(shown, IMG / f"family-{sec}.svg", header=header)
        made.append(sec)
    return made


def render_dot(name):
    src = IMG / f"{name}.dot"
    out = IMG / f"{name}.svg"
    try:
        subprocess.run(
            ["dot", "-Tsvg", "-o", str(out), str(src)], check=True, capture_output=True
        )
    except FileNotFoundError:
        print(f"graphviz absent, {out.name} left as it is", file=sys.stderr)
        return False
    return True


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rows = bench_rows()
    if not rows:
        sys.exit(f"no {RUN_DATE} {PROMPT}/{MODE} rows in docs/BENCHMARKS.md")
    bar_chart(rows, IMG / "decode-vs-ollama.svg")
    for name in ("cluster-topology", "marlin-pipeline"):
        if (IMG / f"{name}.dot").exists():
            render_dot(name)
    print(f"section charts: {len(section_charts())}")

    # The table lives between markers so the page carries one copy of these numbers, not a
    # transcription of them.
    status = ROOT / "docs" / "STATUS.md"
    text = status.read_text()
    start, end = "<!-- table:decode -->", "<!-- /table:decode -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        status.write_text(f"{head}{start}\n{comparison_table(rows)}\n{end}{tail}")
        print(f"docs/STATUS.md: {len(rows)} rows regenerated")
    else:
        print(comparison_table(rows))


if __name__ == "__main__":
    main()
