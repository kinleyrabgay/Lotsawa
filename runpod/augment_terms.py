"""Teach the model new or corrected vocabulary, without inventing Dzongkha grammar.

The problem with training on bare word pairs ("apple" / ཀུ་ཤུ་) is that the model
learns to translate isolated words and gets no signal about how the term behaves
in a sentence -- which case marker follows it, how it pluralises, where it sits.
Worse, a few thousand word-pair rows against 500k sentence rows either vanish in
the mix or damage sentence fluency.

So we substitute instead. The corpus already contains real human sentences for
many concrete nouns: 183 for "apple", 931 for "dog", 1,230 for "father". Swap the
noun slot in those sentences and you get grammatical carriers for a new term at no
linguistic cost -- the same trick that fixed numerals, applied to vocabulary.

Two modes:

  discover   Scan the corpus and draft a term table: for each English word, the
             Dzongkha renderings actually used and how often. A native speaker
             then marks the preferred form. This is also how you find errors --
             it is what revealed that "apple" appears as three different
             transliterations of the English word rather than a native term.

  build      Read the reviewed table and emit training pairs: every carrier with
             the term normalised to the preferred form, plus cross-substituted
             carriers for every other term in the same category.

    python augment_terms.py discover --csv ../dataset.csv --out terms_draft.tsv
    # native speaker edits terms_draft.tsv -> terms.tsv
    python augment_terms.py build --csv ../dataset.csv --terms terms.tsv \
        --phrases phrases.tsv --out /workspace/data/dz_en_terms
"""

import argparse
import collections
import csv
import random
import re
import sys

sys.path.insert(0, ".")
from normalize import normalize_pair  # noqa: E402

DZ = "dzo_Tibt"
EN = "eng_Latn"
TSHEG = "་"

# English words whose Dzongkha rendering is worth auditing. Extend freely -- this
# list only seeds `discover`.
SEED_WORDS = {
    "fruit": ["apple", "banana", "mango", "orange", "grape", "peach", "pear",
              "guava", "papaya", "lemon", "pineapple", "watermelon"],
    "vegetable": ["potato", "onion", "chilli", "cabbage", "carrot", "spinach",
                  "tomato", "garlic", "ginger", "radish", "pumpkin"],
    "animal": ["cow", "dog", "cat", "horse", "yak", "goat", "sheep", "pig",
               "chicken", "tiger", "bear", "monkey", "snake", "bird", "fish"],
    "kinship": ["mother", "father", "brother", "sister", "uncle", "aunt", "son",
                "daughter", "grandmother", "grandfather", "wife", "husband"],
    "place": ["thimphu", "paro", "punakha", "bumthang", "trongsa", "wangdue",
              "trashigang", "india", "nepal", "bhutan"],
    "colour": ["red", "blue", "green", "yellow", "white", "black", "orange"],
}


def load_corpus(path):
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        return [(a.strip(), b.strip()) for a, b in reader]


def dz_chunks(dz_text):
    """Space-delimited chunks of the raw (word-segmented) Dzongkha side."""
    return [c for c in dz_text.split() if c]


def discover(rows, min_count):
    """For each seed word, report the Dzongkha chunks that co-occur with it.

    Crude but effective: a chunk appearing in most sentences containing the
    English word, and rarely otherwise, is almost certainly its translation.
    """
    # Baseline chunk frequency, to discount common grammatical chunks.
    total = collections.Counter()
    for dz, _ in rows:
        total.update(set(dz_chunks(dz)))
    n_rows = len(rows)

    out = []
    for category, words in SEED_WORDS.items():
        for word in words:
            pat = re.compile(rf"\b{word}s?\b", re.I)
            hits = [(dz, en) for dz, en in rows if pat.search(en)]
            if len(hits) < min_count:
                continue
            local = collections.Counter()
            for dz, _ in hits:
                local.update(set(dz_chunks(dz)))
            # Score by lift: frequent here, rare overall.
            scored = []
            for chunk, c in local.items():
                if c < max(3, 0.15 * len(hits)):
                    continue
                lift = (c / len(hits)) / max(total[chunk] / n_rows, 1e-9)
                scored.append((lift, c, chunk))
            scored.sort(reverse=True)
            cands = [f"{ch}({c})" for _, c, ch in scored[:4]]
            out.append({
                "category": category,
                "en": word,
                "carriers": len(hits),
                "dz_preferred": scored[0][2] if scored else "",
                "dz_candidates": " | ".join(cands),
            })
    return out


def swap_english(sentence, old, new):
    """Replace an English noun, keeping plural -s and fixing a/an."""
    def repl(m):
        return new + ("s" if m.group(0).lower().endswith("s") else "")

    s = re.sub(rf"\b{re.escape(old)}s?\b", repl, sentence, flags=re.I)
    # "an mango" -> "a mango", "a apple" -> "an apple"
    s = re.sub(r"\ban (?=[^aeiouAEIOU\W])", "a ", s)
    s = re.sub(r"\ba (?=[aeiouAEIOU])", "an ", s)
    return s


def build(rows, terms, phrases, args):
    rng = random.Random(args.seed)
    by_cat = collections.defaultdict(list)
    for t in terms:
        by_cat[t["category"]].append(t)

    pairs = []
    stats = collections.Counter()

    for term in terms:
        forms = [term["dz"]] + [v for v in term.get("variants", []) if v]
        pat = re.compile(rf"\b{re.escape(term['en'])}s?\b", re.I)
        carriers = []
        for dz, en in rows:
            if not pat.search(en):
                continue
            found = next((f for f in forms if f and f in dz), None)
            if found:
                carriers.append((dz, en, found))
        if not carriers:
            stats["terms_without_carrier"] += 1
            if args.verbose:
                print(f"  no carrier found: {term['en']}")
            continue
        stats["terms_with_carrier"] += 1
        rng.shuffle(carriers)
        carriers = carriers[:args.max_carriers]

        # 1. Normalise the term to its preferred form in its own carriers.
        for dz, en, found in carriers:
            fixed = dz.replace(found, term["dz"]) if found != term["dz"] else dz
            pairs.append((fixed, en))
            stats["corrected" if found != term["dz"] else "kept"] += 1

        # 2. Lend those carriers to every other term in the category.
        for other in by_cat[term["category"]]:
            if other["en"] == term["en"]:
                continue
            for dz, en, found in carriers[:args.cross_per_term]:
                new_dz = dz.replace(found, other["dz"])
                new_en = swap_english(en, term["en"], other["en"])
                if new_dz == dz or new_en == en:
                    continue
                pairs.append((new_dz, new_en))
                stats["cross_substituted"] += 1

    # 3. Phrases and proverbs are non-compositional -- no substitution, just
    #    include them, repeated so they are not lost in a 500k-row mix.
    for dz, en in phrases:
        for _ in range(args.phrase_repeat):
            pairs.append((dz, en))
            stats["phrase_rows"] += 1

    return pairs, stats


def read_tsv(path, required):
    if not path:
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) < required:
                continue
            out.append(cells)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["discover", "build"])
    ap.add_argument("--csv", default="../dataset.csv")
    ap.add_argument("--terms", default="terms.tsv")
    ap.add_argument("--phrases", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=5,
                    help="discover: minimum carrier sentences to report a word.")
    ap.add_argument("--max-carriers", type=int, default=40,
                    help="build: carriers used per term.")
    ap.add_argument("--cross-per-term", type=int, default=8,
                    help="build: carriers lent to each sibling term.")
    ap.add_argument("--phrase-repeat", type=int, default=8,
                    help="build: copies of each proverb or fixed phrase.")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip orthography restoration (only for inspection).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = load_corpus(args.csv)
    print(f"Corpus: {len(rows)} pairs")

    if args.mode == "discover":
        found = discover(rows, args.min_count)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("# Review this table, then save it as terms.tsv.\n")
            fh.write("# dz_preferred is a GUESS from co-occurrence. A native "
                     "speaker must confirm it,\n# and should replace English "
                     "transliterations with the proper Dzongkha term.\n")
            fh.write("# Put any wrong forms already in the corpus into "
                     "dz_variants, pipe-separated,\n# so build can find and "
                     "correct them.\n")
            fh.write("#\n# category\ten\tdz_preferred\tdz_variants\n")
            for r in found:
                fh.write(f"{r['category']}\t{r['en']}\t{r['dz_preferred']}\t"
                         f"\t# {r['carriers']} carriers | candidates: "
                         f"{r['dz_candidates']}\n")
        print(f"Drafted {len(found)} terms -> {args.out}")
        print("Next: have a native speaker confirm dz_preferred and fill "
              "dz_variants, then run `build`.")
        return

    terms = []
    for cells in read_tsv(args.terms, 3):
        terms.append({
            "category": cells[0].strip(),
            "en": cells[1].strip(),
            "dz": cells[2].strip(),
            "variants": cells[3].split("|") if len(cells) > 3 else [],
        })
    phrases = [(c[1].strip(), c[0].strip()) for c in read_tsv(args.phrases, 2)]
    print(f"Terms: {len(terms)}   Phrases: {len(phrases)}")

    pairs, stats = build(rows, terms, phrases, args)
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())))

    from datasets import Dataset
    flat = {"src": [], "tgt": [], "src_lang": [], "tgt_lang": []}
    for dz, en in pairs:
        if not args.no_normalize:
            dz, en = normalize_pair(dz, en)
        for s, t, sl, tl in ((dz, en, DZ, EN), (en, dz, EN, DZ)):
            flat["src"].append(s)
            flat["tgt"].append(t)
            flat["src_lang"].append(sl)
            flat["tgt_lang"].append(tl)

    ds = Dataset.from_dict(flat)
    ds.save_to_disk(args.out)
    print(f"\nWrote {len(ds)} rows (both directions) -> {args.out}")
    print("\nSample:")
    for i in range(min(6, len(ds))):
        if ds[i]["src_lang"] == EN:
            print(f"  {ds[i]['src']}\n   -> {ds[i]['tgt']}")


if __name__ == "__main__":
    main()
