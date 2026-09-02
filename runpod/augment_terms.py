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
import os
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
    # Personal names are the richest category in this corpus: 4,745 carriers for
    # "tom" alone, 1,661 for "mary", each already transliterated on the Dzongkha
    # side. Substituting Bhutanese names into those sentences is the cheapest
    # possible route to name coverage.
    "person": ["tom", "mary", "john", "alice", "david", "mike"],
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
            pairs.append((fixed, en, False))
            stats["corrected" if found != term["dz"] else "kept"] += 1

        # 2. Lend those carriers to every other term in the category.
        for other in (by_cat[term["category"]] if args.cross_per_term else []):
            if other["en"] == term["en"]:
                continue
            for dz, en, found in carriers[:args.cross_per_term]:
                new_dz = dz.replace(found, other["dz"])
                new_en = swap_english(en, term["en"], other["en"])
                if new_dz == dz or new_en == en:
                    continue
                pairs.append((new_dz, new_en, False))
                stats["cross_substituted"] += 1

        # 3. Bare term pairs, so a single word typed on its own is in
        #    distribution. People use a translator as a dictionary -- "apple",
        #    "Tashi", "mango" -- and a model trained only on sentences treats a
        #    lone word as out-of-distribution input and may answer with a whole
        #    sentence. Carriers teach usage; bare pairs teach lookup. Both are
        #    needed, which is why these are emitted alongside, not instead.
        for _ in range(args.bare_repeat):
            # bare=True: a lookup is not a sentence, so it must not pick up a
            # shad or a full stop from normalize_pair.
            pairs.append((term["dz"], term["en"], True))
            stats["bare_rows"] += 1
            # Capitalised too, since that is how people type a name.
            if term["en"][:1].islower():
                pairs.append((term["dz"], term["en"].capitalize(), True))
                stats["bare_rows"] += 1

    # 4. Phrases and proverbs are non-compositional -- no substitution, just
    #    include them, repeated so they are not lost in a 500k-row mix.
    for dz, en in phrases:
        for _ in range(args.phrase_repeat):
            pairs.append((dz, en, False))
            stats["phrase_rows"] += 1

    return pairs, stats


TIBETAN = re.compile(r"[\u0f00-\u0fff]")
LATIN = re.compile(r"[A-Za-z]")


def read_term_dir(path):
    """Load per-category CSVs. Category comes from the filename."""
    import glob
    terms = []
    for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
        category = os.path.splitext(os.path.basename(f))[0]
        with open(f, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                en = (row.get("en") or "").strip()
                dz = (row.get("dz_preferred") or "").strip()
                if not en:
                    continue
                terms.append({
                    "category": category,
                    "en": en,
                    "dz": dz,
                    "variants": [v.strip() for v in
                                 (row.get("dz_variants") or "").split("|")
                                 if v.strip()],
                    "confidence": (row.get("confidence") or "").strip().lower(),
                    "_file": os.path.basename(f),
                })
    return terms


def check_terms(terms):
    """Lint a reviewed term table. Returns (usable, problems)."""
    problems = []
    usable = []
    seen_en = {}
    seen_dz = {}

    for t in terms:
        where = f"{t['_file']}:{t['en']}"
        if not t["dz"]:
            problems.append(("skip", where, "dz_preferred is empty"))
            continue
        if LATIN.search(t["dz"]):
            problems.append(("error", where,
                             f"dz_preferred contains Latin letters: {t['dz']!r}"))
            continue
        if not TIBETAN.search(t["dz"]):
            problems.append(("error", where,
                             f"dz_preferred has no Tibetan script: {t['dz']!r}"))
            continue
        if not t["dz"].endswith(TSHEG):
            problems.append(("warn", where,
                             f"dz_preferred does not end in a tsheg: {t['dz']!r}"))
        if " " in t["en"]:
            problems.append(("warn", where,
                             "multi-word English term -- carriers rarely match"))
        key = (t["category"], t["en"].lower())
        if key in seen_en:
            problems.append(("warn", where, "duplicate English term in category"))
        seen_en[key] = True
        if t["dz"] in seen_dz and seen_dz[t["dz"]] != t["en"]:
            problems.append(("warn", where,
                             f"same Dzongkha form as '{seen_dz[t['dz']]}' -- "
                             "copy-paste error?"))
        seen_dz.setdefault(t["dz"], t["en"])
        usable.append(t)

    return usable, problems


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
    ap.add_argument("mode", choices=["discover", "build", "check"])
    ap.add_argument("--csv", default="../dataset.csv")
    ap.add_argument("--terms", default="",
                    help="Single TSV: category, en, dz, variants.")
    ap.add_argument("--terms-dir", default="terms",
                    help="Directory of per-category CSVs as written by "
                         "build_term_tables.py. The filename is the category. "
                         "Reads en, dz_preferred and dz_variants; every other "
                         "column is ignored, so diagnostics can stay in place.")
    ap.add_argument("--phrases", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=5,
                    help="discover: minimum carrier sentences to report a word.")
    ap.add_argument("--max-carriers", type=int, default=40,
                    help="build: carriers used per term.")
    ap.add_argument("--cross-per-term", type=int, default=0,
                    help="build: carriers lent to each sibling term. DEFAULT 0 "
                         "(off), because category members are not freely "
                         "interchangeable -- lending 'milk a cow' to 'dog' "
                         "yields 'milk a dog' and overwrites whatever the "
                         "matched Dzongkha chunk actually was. Enable only for "
                         "categories whose members are genuinely swappable, and "
                         "read the printed sample before training.")
    ap.add_argument("--phrase-repeat", type=int, default=8,
                    help="build: copies of each proverb or fixed phrase.")
    ap.add_argument("--bare-repeat", type=int, default=6,
                    help="build: copies of each term as a bare word pair, so "
                         "single-word queries work. 0 to disable.")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip orthography restoration (only for inspection).")
    ap.add_argument("--min-confidence", default="high",
                    choices=["dict", "high", "medium", "low"],
                    help="build: lowest confidence band to train on. Default "
                         "'high'; 'dict' trains only on dictionary-sourced "
                         "rows. Medium and low rows are extraction guesses -- "
                         "'cow' resolved to the verb 'to milk' because milking "
                         "sentences dominate its carriers -- and a wrong bare "
                         "pair teaches the wrong word outright.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rows = load_corpus(args.csv)
    print(f"Corpus: {len(rows)} pairs")

    if args.mode == "check":
        raw = (read_term_dir(args.terms_dir) if not args.terms
               else [{"category": c[0].strip(), "en": c[1].strip(),
                      "dz": c[2].strip(),
                      "variants": c[3].split("|") if len(c) > 3 else [],
                      "confidence": "", "_file": os.path.basename(args.terms)}
                     for c in read_tsv(args.terms, 3)])
        usable, problems = check_terms(raw)
        by_kind = collections.Counter(k for k, _, _ in problems)
        print(f"\n{len(raw)} rows, {len(usable)} usable for training")
        print(f"  unfilled (no dz_preferred): {by_kind['skip']}")
        print(f"  errors:                     {by_kind['error']}")
        print(f"  warnings:                   {by_kind['warn']}")
        for kind, where, msg in problems:
            if kind in ("error", "warn"):
                print(f"  {kind.upper():5s} {where}: {msg}")
        # Carrier availability decides how much data each term can produce.
        print("\nCarrier counts for usable terms (0 means no training data):")
        counts = []
        for t in usable:
            forms = [t["dz"]] + t["variants"]
            pat = re.compile(rf"\b{re.escape(t['en'])}s?\b", re.I)
            n_car = sum(1 for dz, en in rows
                        if pat.search(en) and any(f in dz for f in forms if f))
            counts.append((n_car, t["category"], t["en"]))
        counts.sort()
        zero = [c for c in counts if c[0] == 0]
        print(f"  {len(zero)} of {len(counts)} usable terms have NO carrier")
        for n_car, cat, en in zero[:10]:
            print(f"    no carrier: {cat}/{en}")
        if len(counts) > len(zero):
            top = counts[-5:]
            print("  richest:", ", ".join(f"{en}({n})" for n, _, en in reversed(top)))
        print(f"\n{'READY' if not by_kind['error'] else 'NOT READY'} -- "
              f"{len(usable)} terms would train.")
        return

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

    if args.terms:
        raw = [{"category": c[0].strip(), "en": c[1].strip(), "dz": c[2].strip(),
                "variants": c[3].split("|") if len(c) > 3 else [],
                "confidence": "", "_file": os.path.basename(args.terms)}
               for c in read_tsv(args.terms, 3)]
    else:
        raw = read_term_dir(args.terms_dir)

    # "dict" is a form taken from the DDC dictionary, which outranks any
    # confidence band inferred from corpus co-occurrence.
    order = {"dict": 4, "high": 3, "medium": 2, "low": 1}
    floor = order[args.min_confidence]
    before = len(raw)
    raw = [t for t in raw if order.get(t["confidence"], 3) >= floor]
    if before != len(raw):
        print(f"Confidence gate '{args.min_confidence}': dropped "
              f"{before - len(raw)} of {before} rows below it")

    terms, problems = check_terms(raw)
    errors = [p for p in problems if p[0] == "error"]
    skips = [p for p in problems if p[0] == "skip"]
    warns = [p for p in problems if p[0] == "warn"]
    print(f"Term table: {len(raw)} rows -> {len(terms)} usable "
          f"({len(skips)} unfilled, {len(errors)} errors, {len(warns)} warnings)")
    for kind, where, msg in errors[:10]:
        print(f"  ERROR {where}: {msg}")
    if errors:
        raise SystemExit("Fix the errors above, or drop those rows, then re-run.")
    for kind, where, msg in warns[:8]:
        print(f"  warn  {where}: {msg}")

    phrases = [(c[1].strip(), c[0].strip()) for c in read_tsv(args.phrases, 2)]
    print(f"Phrases: {len(phrases)}")

    pairs, stats = build(rows, terms, phrases, args)
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())))

    from datasets import Dataset
    flat = {"src": [], "tgt": [], "src_lang": [], "tgt_lang": []}
    for dz, en, bare in pairs:
        if bare:
            # Keep the term exactly as written: tsheg intact, no shad, no stop.
            dz, en = dz.strip(), en.strip()
        elif not args.no_normalize:
            dz, en = normalize_pair(dz, en)
        for s, t, sl, tl in ((dz, en, DZ, EN), (en, dz, EN, DZ)):
            flat["src"].append(s)
            flat["tgt"].append(t)
            flat["src_lang"].append(sl)
            flat["tgt_lang"].append(tl)

    ds = Dataset.from_dict(flat)
    ds.save_to_disk(args.out)
    print(f"\nWrote {len(ds)} rows (both directions) -> {args.out}")

    if stats["cross_substituted"]:
        print(f"\nWARNING  {stats['cross_substituted']} cross-substituted rows. "
              f"Read a sample before training --\n"
              f"         substitution assumes category members are "
              f"interchangeable, which fails\n"
              f"         for verb-bound carriers such as 'milk a cow':")
        shown = 0
        for r in ds:
            if r["src_lang"] == EN and len(r["src"].split()) > 5:
                print(f"           {r['src']}\n            -> {r['tgt']}")
                shown += 1
                if shown >= 4:
                    break

    print("\nBare lookup pairs:")
    shown = 0
    for r in ds:
        if r["src_lang"] == EN and len(r["src"].split()) == 1:
            print(f"  {r['src']:14s} -> {r['tgt']}")
            shown += 1
            if shown >= 8:
                break


if __name__ == "__main__":
    main()
