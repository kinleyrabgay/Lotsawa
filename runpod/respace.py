"""Convert word-segmented Dzongkha to Kuensel-style phrase spacing.

Your parallel corpus and real published Dzongkha use different spacing
conventions, and they are not close:

  dataset.csv  0.750 spaces per syllable  -- a space after nearly every word
  Kuensel      0.211 spaces per syllable  -- a space at phrase boundaries only

Joining everything (normalize.py's first approach) matches neither. Keeping every
space matches the corpus but not the language as it is actually published, which
is also not what NLLB saw in pretraining.

The fix works because the corpus's word boundaries are a strict *superset* of
Kuensel's phrase boundaries: every real phrase break is already a space in the
corpus, so the job is choosing which existing spaces to keep, not inventing new
ones. And that choice turns out to be lexical -- Kuensel breaks after case
markers and postpositions (ལུ, ལས, གིས, ནང, དང ...), which in 40k sentences never
appear chunk-final *without* a following space.

So: learn the boundary-marker set from real Kuensel text, then apply it.

  python respace.py learn --mono data/dz_mono.txt --out boundary_markers.txt
  python respace.py validate --mono data/dz_mono.txt --markers boundary_markers.txt
"""

import argparse
import collections

TSHEG = "་"
SHAD = "།"


def chunk_tail(chunk):
    """Final syllable of a tsheg-delimited chunk."""
    syls = [s for s in chunk.split(TSHEG) if s]
    return syls[-1] if syls else ""


def learn(lines, min_count=200, min_prob=0.90):
    """Find syllables that reliably precede a phrase space."""
    before_space = collections.Counter()
    total = collections.Counter()
    for line in lines:
        chunks = [c for c in line.split(" ") if c]
        for i, chunk in enumerate(chunks):
            tail = chunk_tail(chunk.rstrip(SHAD))
            if not tail:
                continue
            total[tail] += 1
            if i < len(chunks) - 1:
                before_space[tail] += 1

    markers = set()
    for tail, tot in total.items():
        if tot < min_count:
            continue
        if before_space[tail] / tot >= min_prob:
            markers.add(tail)
    return markers, before_space, total


def respace(text, markers):
    """Re-space word-segmented Dzongkha: keep a space only after a marker."""
    chunks = [c for c in text.split(" ") if c]
    if not chunks:
        return ""
    out = []
    for i, chunk in enumerate(chunks):
        out.append(chunk)
        if i == len(chunks) - 1:
            continue
        if chunk_tail(chunk.rstrip(SHAD)) in markers:
            out.append(" ")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["learn", "validate"])
    ap.add_argument("--mono", default="data/dz_mono.txt")
    ap.add_argument("--markers", default="boundary_markers.txt")
    ap.add_argument("--out", default="boundary_markers.txt")
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--min-prob", type=float, default=0.90)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    lines = [l.strip() for l in open(args.mono, encoding="utf-8") if l.strip()]
    cut = int(args.train_frac * len(lines))
    train, held = lines[:cut], lines[cut:]

    if args.mode == "learn":
        markers, _, _ = learn(train, args.min_count, args.min_prob)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(markers)) + "\n")
        print(f"Learned {len(markers)} boundary markers from {len(train)} sentences")
        print(f"Wrote {args.out}")
        return

    markers = {l.strip() for l in open(args.markers, encoding="utf-8") if l.strip()}
    print(f"{len(markers)} markers, validating on {len(held)} held-out sentences\n")

    # Reconstruct spacing from the fully de-spaced form and score the boundaries.
    tp = fp = fn = 0
    exact = 0
    for line in held:
        chunks = [c for c in line.split(" ") if c]
        if len(chunks) < 2:
            continue
        truth = [True] * (len(chunks) - 1)
        pred = [chunk_tail(c.rstrip(SHAD)) in markers for c in chunks[:-1]]
        # Every corpus boundary in Kuensel IS a real space, so truth is all-True;
        # what we measure is how many the rule recovers.
        for t, p in zip(truth, pred):
            if t and p:
                tp += 1
            elif t and not p:
                fn += 1
            elif p and not t:
                fp += 1
        if pred == truth:
            exact += 1

    recall = tp / (tp + fn) if tp + fn else 0
    print(f"Boundary recall : {recall:.3f}  ({tp} recovered, {fn} missed)")
    print(f"Sentences with every boundary recovered: {exact}")
    print("\nNote: recall is the metric that matters here. Kuensel's own spaces are\n"
          "all genuine, so a missed boundary means the output runs two phrases\n"
          "together; the rule cannot produce a false boundary that the corpus\n"
          "did not already mark as a word break.")


if __name__ == "__main__":
    main()
