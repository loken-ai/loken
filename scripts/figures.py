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

# The run every figure describes: one card for both engines, streamed, short prompt.
RUN_DATE = "2026-08-27"
PROMPT = "short"
MODE = "stream"


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


def bar_chart(rows, path):
    """Paired bars, one model per row, widest scale set by the fastest engine.

    Hand-written rather than plotted by a library: the repository adds a dependency only for a
    reason, and this is a hundred lines of arithmetic.
    """
    data = sorted(rows.items(), key=lambda kv: -(kv[1][1] / kv[1][0]))
    top = max(max(v) for v in rows.values())
    row_h, bar_h, gap, label_w, pad = 46, 15, 4, 150, 16
    chart_w = 420
    width = label_w + chart_w + 70
    height = pad * 2 + 34 + row_h * len(data) + 30

    def bar(x, y, w, h, cls):
        return f'<rect class="{cls}" x="{x}" y="{y}" width="{max(w, 1):.1f}" height="{h}" rx="2"/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">',
        # Mid-tones legible on a white page and on a dark one, and redefined when the viewer
        # says which it is. A figure that only reads on one background is half a figure.
        "<style>"
        ".t{fill:#57606a;font-size:12px}"
        ".m{fill:#24292f;font-size:12.5px}"
        ".h{fill:#57606a;font-size:11px;letter-spacing:.06em}"
        ".o{fill:#8c959f}"
        ".l{fill:#2da44e}"
        ".ax{stroke:#d0d7de;stroke-width:1}"
        "@media (prefers-color-scheme:dark){"
        ".t,.h{fill:#8b949e}.m{fill:#e6edf3}.o{fill:#6e7681}.l{fill:#3fb950}"
        ".ax{stroke:#30363d}}"
        "</style>",
        # The date is on the figure because these numbers are superseded, and a chart
        # travels away from the paragraph that says so.
        f'<text class="h" x="{label_w}" y="{pad + 6}">DECODE TOK/S, ONE CARD, '
        f'{RUN_DATE} - SUPERSEDED</text>',
    ]
    y0 = pad + 22
    svg.append(f'<line class="ax" x1="{label_w}" y1="{y0}" x2="{label_w}" y2="{height - pad}"/>')
    for i, (model, (ollama, loken)) in enumerate(data):
        y = y0 + i * row_h + 6
        svg.append(f'<text class="m" x="{label_w - 10}" y="{y + 15}" text-anchor="end">{model}</text>')
        svg.append(bar(label_w, y, ollama / top * chart_w, bar_h, "o"))
        svg.append(bar(label_w, y + bar_h + gap, loken / top * chart_w, bar_h, "l"))
        svg.append(
            f'<text class="t" x="{label_w + ollama / top * chart_w + 6}" y="{y + 12}">'
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
