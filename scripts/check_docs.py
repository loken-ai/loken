#!/usr/bin/env python3
"""Refuse a documentation tree with a link that leads nowhere.

The markdown pages under `docs/` link to each other, to the guides, to the images and to
files in the source tree, and several of those links carry an anchor into a heading of the
target page. Nothing checked them: the CI's documentation step is `cargo doc`, which sees
rustdoc links and no markdown. A page renamed, a heading reworded or an image regenerated
under another name broke a link that stayed broken until a reader followed it.

Three checks, each on every markdown page of the repository root and of `docs/`:

  target     a relative link names a file or directory that exists.
  anchor     a `#fragment` on a markdown target names a heading of that page, in the slug
             GitHub gives it (lower case, punctuation dropped, spaces to hyphens, a numeric
             suffix for a repeated heading).
  image      an image reference resolves like a link.

Absolute URLs are not followed: a network check would make the gate depend on the network.
Links inside fenced code blocks are not read.

Run it before committing a page:  python3 scripts/check_docs.py
Exit status is 0 when every link resolves, 1 otherwise, one line per defect.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").rglob("*.md"))
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:")


def slug(text):
    """The anchor GitHub derives from a heading's text."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(page):
    """Every anchor a page offers, with the suffixes GitHub adds to repeated headings."""
    seen = {}
    out = set()
    for line in prose_lines(page):
        m = HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def prose_lines(page):
    """The lines of a page outside its fenced code blocks."""
    fence = None
    for line in page.read_text(encoding="utf-8").splitlines():
        m = FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            continue
        if fence is None:
            yield line


def check(page, anchor_cache):
    defects = []
    for lineno, line in enumerate(prose_lines(page), 1):
        for target in LINK.findall(line):
            if SCHEME.match(target):
                continue
            path, _, fragment = target.partition("#")
            resolved = page if path == "" else (page.parent / path).resolve()
            if not resolved.exists():
                defects.append(f"{page.relative_to(ROOT)}:{lineno}: {target}: no such file")
                continue
            if fragment and resolved.suffix == ".md":
                if resolved not in anchor_cache:
                    anchor_cache[resolved] = anchors(resolved)
                if fragment not in anchor_cache[resolved]:
                    defects.append(
                        f"{page.relative_to(ROOT)}:{lineno}: {target}: no heading for #{fragment}"
                    )
    return defects


def main():
    anchor_cache = {}
    defects = []
    for page in PAGES:
        defects.extend(check(page, anchor_cache))
    for line in defects:
        print(line)
    if defects:
        print(f"{len(defects)} broken link(s) in {len(PAGES)} pages", file=sys.stderr)
        return 1
    print(f"{len(PAGES)} pages, every link resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
