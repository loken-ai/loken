"""Same prompt, same seed, same tokens - run it N times and refuse to average.

Greedy decoding is a function, so two runs of one prompt must agree token for token. On a
model whose layers span several cards that stops being free: the ordering between the cards
is what makes it true, and an ordering defect does not announce itself. It produces fluent
text from a slightly different state, which reads as correct and passes a coherence check.

So the comparison is the WHOLE completion, byte for byte, against the first run - never a
similarity, never a prefix. A single differing character fails the gate and prints where.
"""
import sys
import json
import urllib.request

PORT = 11435  # loken; ollama is 11434


def once(model, prompt, n_predict, port):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"temperature": 0, "seed": 42, "num_predict": n_predict},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def first_difference(a, b):
    """Where they diverge, with enough context to recognise the position."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, repr(a[max(0, i - 40):i + 40]), repr(b[max(0, i - 40):i + 40])
    return min(len(a), len(b)), repr(a[-80:]), repr(b[-80:])


def main(model, runs=3, n_predict=128, port=PORT):
    prompt = "Explain, step by step, how a binary search finds a value in a sorted array."
    ref = None
    for k in range(runs):
        out = once(model, prompt, n_predict, port)
        text = out.get("response", "")
        count = out.get("eval_count")
        print(f"  run {k + 1}: {count} jetons, {len(text)} octets")
        if ref is None:
            ref, ref_count = text, count
            continue
        if text != ref or count != ref_count:
            pos, left, right = first_difference(ref, text)
            print(f"\n  ECHEC: la execution {k + 1} diverge de la premiere a l'octet {pos}")
            print(f"    run 1   {left}")
            print(f"    run {k + 1}   {right}")
            print(f"    jetons: {ref_count} puis {count}")
            return 1
    if not ref:
        print("  ECHEC: reponse vide - rien a comparer, la sonde ne peut pas conclure")
        return 1
    print(f"  OK: {runs} executions identiques au jeton pres ({ref_count} jetons)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: determinism.py <model> [runs] [n_predict] [port]")
        sys.exit(2)
    sys.exit(main(sys.argv[1],
                  int(sys.argv[2]) if len(sys.argv) > 2 else 3,
                  int(sys.argv[3]) if len(sys.argv) > 3 else 128,
                  int(sys.argv[4]) if len(sys.argv) > 4 else PORT))
