#!/usr/bin/env python3
"""Transcribe a clip set through the chat audio input of a running lokend and score it.

Each clip goes to `/v1/chat/completions` as one user turn: an `input_audio` part and a text
part, "Transcribe this audio" unless `--prompt` says otherwise, greedy. The transcript is
scored against the clip set's ground truth by word error rate, both sides normalised the same
way (lowercase, punctuation stripped). The real-time factor is the request's wall time over
the clip's duration.

    python3 scripts/asr_chat.py --model <name> --clips <directory of wav clips and transcripts.txt>
                                [--port 11435] [--prompt TEXT] [--out results.json]

The clip directory holds `<id>.wav` files and `transcripts.txt`, one `<id> <text>` per line.
"""
import argparse
import base64
import json
import pathlib
import re
import sys
import time
import urllib.request
import wave

PROMPT = "Transcribe this audio"
MAX_TOKENS = 256


def normalise(text):
    """Lowercase words with every character but letters and digits removed."""
    text = text.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", " ", text).split()


def edits(ref, hyp):
    """Word-level Levenshtein distance between two word lists."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
            prev, row[j] = row[j], cur
    return row[len(hyp)]


def score(rows):
    """Corpus WER, mean errors per clip and real-time factor of `rows`, each a dict with
    `reference`, `transcript`, `seconds` and `elapsed`."""
    errors = sum(edits(normalise(r["reference"]), normalise(r["transcript"])) for r in rows)
    words = sum(len(normalise(r["reference"])) for r in rows)
    audio = sum(r["seconds"] for r in rows)
    return {
        "wer": errors / max(words, 1),
        "errors": errors,
        "words": words,
        "errors_per_clip": errors / max(len(rows), 1),
        "rtf": sum(r["elapsed"] for r in rows) / max(audio, 1e-9),
    }


def clips(directory):
    """`(id, wav path, reference text)` of a clip set, in transcript order."""
    directory = pathlib.Path(directory)
    for line in (directory / "transcripts.txt").read_text().splitlines():
        if line.strip():
            cid, text = line.split(" ", 1)
            yield cid, directory / f"{cid}.wav", text


def duration(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def transcribe(model, path, port, prompt=PROMPT):
    data = base64.b64encode(path.read_bytes()).decode()
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--limit", type=int, help="score only the first N clips")
    ap.add_argument("--out")
    args = ap.parse_args()
    # One request first loads the model, so the timed requests measure transcription only.
    first = next(clips(args.clips))
    transcribe(args.model, first[1], args.port, args.prompt)
    rows = []
    for cid, path, reference in list(clips(args.clips))[: args.limit]:
        start = time.monotonic()
        text = transcribe(args.model, path, args.port, args.prompt)
        elapsed = time.monotonic() - start
        row = {
            "id": cid,
            "reference": reference,
            "transcript": text,
            "seconds": duration(path),
            "elapsed": elapsed,
        }
        rows.append(row)
        e = edits(normalise(reference), normalise(text))
        print(f"{cid}  {e:2d} errors  {elapsed:6.1f} s  {text}", file=sys.stderr)
    result = {"model": args.model, "prompt": args.prompt, "rows": rows, **score(rows)}
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
