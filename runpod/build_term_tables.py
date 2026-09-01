"""Extract candidate bilingual term tables from the corpus, with confidence.

Nothing here is invented. Every Dzongkha form is one that already appears in
dataset.csv, scored by how reliably it co-occurs with the English word, so a
reviewer can see the evidence rather than trusting a guess.

Method: for each English word, take every pair whose English side contains it,
count the space-delimited Dzongkha chunks in those pairs, and score each chunk by

    coverage = (carriers containing the chunk) / (carriers)
    lift     = coverage / (chunk's overall frequency in the corpus)

High coverage and high lift together mean the chunk is the term. Grammatical
particles score high coverage but low lift, so lift is what separates them.

Confidence is coverage, banded:
    >=0.80 high     -- almost certainly right, spot-check only
    >=0.50 medium   -- probably right, confirm
    <0.50  low      -- ambiguous, a reviewer must decide (or the corpus lacks it)

    python build_term_tables.py --csv ../dataset.csv --out terms/
"""

import argparse
import collections
import csv
import os
import re

CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear",
               "guava", "papaya", "lemon", "pineapple", "watermelon", "plum",
               "apricot", "walnut", "strawberry", "coconut", "date", "fig"],
    "vegetables": ["potato", "onion", "chilli", "cabbage", "carrot", "spinach",
                   "tomato", "garlic", "ginger", "radish", "pumpkin", "beans",
                   "peas", "mushroom", "turnip", "cauliflower", "cucumber"],
    "animals": ["cow", "dog", "cat", "horse", "yak", "goat", "sheep", "pig",
                "chicken", "tiger", "bear", "monkey", "snake", "bird", "fish",
                "elephant", "rabbit", "mouse", "deer", "wolf", "donkey", "duck",
                "crow", "eagle", "butterfly", "bee", "ant", "spider"],
    "names": ["tom", "mary", "john", "alice", "david", "mike", "tony", "ken",
              "jane", "bob"],
    "kinship": ["mother", "father", "brother", "sister", "uncle", "aunt", "son",
                "daughter", "grandmother", "grandfather", "wife", "husband",
                "child", "friend", "neighbour", "cousin"],
    "places": ["thimphu", "paro", "punakha", "bumthang", "trongsa", "wangdue",
               "trashigang", "mongar", "samtse", "india", "nepal", "bhutan",
               "china", "japan", "america", "england"],
    "colours": ["red", "blue", "green", "yellow", "white", "black", "brown",
                "grey", "pink", "purple"],
    "body": ["head", "hand", "eye", "ear", "nose", "mouth", "foot", "leg",
             "arm", "hair", "tooth", "heart", "stomach", "back", "finger"],
    "time": ["today", "tomorrow", "yesterday", "morning", "evening", "night",
             "week", "month", "year", "hour", "minute", "day"],
    "food": ["rice", "bread", "milk", "tea", "water", "meat", "egg", "salt",
             "sugar", "oil", "cheese", "butter", "soup", "chilli"],
    "occupation": ["teacher", "doctor", "farmer", "student", "driver", "nurse",
                   "monk", "police", "soldier", "shopkeeper", "engineer"],
    "places_local": ["school", "hospital", "market", "shop", "office", "temple",
                     "house", "road", "bridge", "river", "mountain", "forest"],
}

BAND = [(0.80, "high"), (0.50, "medium"), (0.0, "low")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../dataset.csv")
    ap.add_argument("--out", default="terms")
    ap.add_argument("--min-carriers", type=int, default=4)
    ap.add_argument("--top", type=int, default=3, help="candidates to report")
    args = ap.parse_args()

    csv.field_size_limit(10 ** 9)
    with open(args.csv, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        rows = [(a.strip(), b.strip()) for a, b in reader]
    print(f"Corpus: {len(rows)} pairs")

    overall = collections.Counter()
    for dz, _ in rows:
        overall.update(set(dz.split()))
    n = len(rows)

    os.makedirs(args.out, exist_ok=True)
    summary = collections.Counter()

    for category, words in CATEGORIES.items():
        out_rows = []
        for word in words:
            pat = re.compile(rf"\b{word}s?\b", re.I)
            carriers = [dz for dz, en in rows if pat.search(en)]
            if len(carriers) < args.min_carriers:
                out_rows.append({
                    "en": word, "dz_preferred": "", "confidence": "MISSING",
                    "coverage": "", "carriers": len(carriers),
                    "dz_candidates": "", "review": "YES - not in corpus",
                })
                summary["missing"] += 1
                continue

            local = collections.Counter()
            for dz in carriers:
                local.update(set(dz.split()))

            scored = []
            for chunk, c in local.items():
                cov = c / len(carriers)
                if cov < 0.15:
                    continue
                lift = cov / max(overall[chunk] / n, 1e-9)
                scored.append((lift, cov, chunk))
            scored.sort(reverse=True)
            if not scored:
                out_rows.append({
                    "en": word, "dz_preferred": "", "confidence": "MISSING",
                    "coverage": "", "carriers": len(carriers),
                    "dz_candidates": "", "review": "YES - no clear candidate",
                })
                summary["missing"] += 1
                continue

            lift, cov, best = scored[0]
            band = next(b for t, b in BAND if cov >= t)
            summary[band] += 1
            out_rows.append({
                "en": word,
                "dz_preferred": best,
                "confidence": band,
                "coverage": f"{cov:.2f}",
                "carriers": len(carriers),
                "dz_candidates": " | ".join(
                    f"{c}({cv:.2f})" for _, cv, c in scored[:args.top]),
                "review": "" if band == "high" else "YES",
            })

        path = os.path.join(args.out, f"{category}.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "en", "dz_preferred", "confidence", "coverage", "carriers",
                "dz_candidates", "review", "dz_variants", "notes"])
            w.writeheader()
            for r in out_rows:
                r.setdefault("dz_variants", "")
                r.setdefault("notes", "")
                w.writerow(r)
        hi = sum(1 for r in out_rows if r["confidence"] == "high")
        print(f"  {category:15s} {len(out_rows):3d} terms, {hi:3d} high-confidence"
              f"  -> {path}")

    print(f"\nTotals: {summary['high']} high, {summary['medium']} medium, "
          f"{summary['low']} low, {summary['missing']} missing from corpus")
    print("\nHigh-confidence rows still deserve a spot-check: the corpus stores "
          "some terms as\ntransliterations of the English word rather than the "
          "native Dzongkha term.\nEverything else needs a reviewer or the DDC "
          "dictionary.")


if __name__ == "__main__":
    main()
