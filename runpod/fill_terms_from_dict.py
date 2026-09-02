"""Fill the per-category term tables from the DDC dictionary extraction.

build_term_tables.py guesses a Dzongkha form from corpus co-occurrence, which
leaves two kinds of hole: rows marked MISSING because the word never appears in
dataset.csv, and rows marked low because several chunks scored alike. The
dictionary answers both, so this script fills them from dict/dz_en.tsv rather
than from a reviewer's memory.

It does not overwrite a form the corpus already backs at medium or high
confidence. Those forms are attested in real sentences, which is what the
carrier substitution in augment_terms.py needs; when the dictionary disagrees
with one, the row is flagged for review instead of silently rewritten.

    python extract_dictionary.py          # produces dict/
    python fill_terms_from_dict.py --dry-run
    python fill_terms_from_dict.py
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

# The part of speech a category's terms should have. Everything here is a noun
# except colour words, which the book files as adjectives.
CATEGORY_POS = collections.defaultdict(lambda: ("noun",), {
    "colours": ("adj", "noun"),
})

# The A-Z body has no English proper nouns, so a hit on one of these categories
# is a coincidence -- "mongar" landing on a common noun, not on the dzongkhag.
# Their only real source is the appendix, which romanises Indic personal names
# and world capitals but leaves Bhutanese place names in Dzongkha script alone.
PROPER_NOUN = {"names", "places", "mountains"}
APPENDIX = [("dict/names_indic.tsv", "en", "dz"),
            ("dict/countries.tsv", "en_country", "dz_country"),
            ("dict/countries.tsv", "en_capital", "dz_capital")]

FIELDS = ["en", "dz_preferred", "confidence", "coverage", "carriers",
          "dz_candidates", "review", "dz_variants", "notes"]


def load_entries(path):
    """English gloss -> list of candidate readings, keeping enough of the entry
    to judge which reading is the central one."""
    index = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            n_glosses = sum(len(s["english"]) for s in e["senses"])
            for sense in e["senses"]:
                for pos, gloss in enumerate(sense["english"]):
                    index[gloss.lower().strip()].append({
                        "dz": e["headword"],
                        "pos": e["pos"],
                        "page": e["page"],
                        "sense": sense["n"],
                        "note": sense["note"],
                        "gloss_pos": pos,
                        "n_glosses": n_glosses,
                        "honorific": bool(e["relations"].get("is_honorific")),
                    })
    return index


def variants(term):
    """Spellings to try, best first."""
    t = term.lower().strip()
    yield t
    if t.endswith("ies"):
        yield t[:-3] + "y"
    if t.endswith("es"):
        yield t[:-2]
    if t.endswith("s"):
        yield t[:-1]
    else:
        yield t + "s"
    if " colour" in t:
        yield t.replace(" colour", "")
    if "-" in t:
        yield t.replace("-", " ")
    if " " in t:
        yield t.replace(" ", "-")


def corpus_form(dz):
    """Write a dictionary headword the way the corpus writes a word.

    A headword is printed as a citation form ending in a shad (སློབ་དཔོན།); the
    same word inside a sentence -- which is what the carrier substitution in
    augment_terms.py splices in -- ends in a tsheg (སློབ་དཔོན་).
    """
    return re.sub(r"[\u0f0b\u0f0c\u0f0d\u0f0e\s]+$", "", dz.strip()) + "\u0f0b"


def norm(dz):
    """Compare Dzongkha forms without punctuation noise.

    Corpus chunks are space-delimited and end in a tsheg; dictionary headwords
    end in a shad. The same word therefore never matches as a raw string.
    """
    dz = dz.replace("\u0f0c", "\u0f0b")
    return re.sub(r"[\u0f0b\u0f0d\u0f0e\s]+$", "", dz.strip())


def score(cand, wanted_pos, corpus_forms):
    """Rank the readings the book gives for one English word.

    The book lists every Dzongkha word an English gloss can translate, so
    "bull" returns both the animal and the stock-market sense, and "dog"
    returns the animal, the verb "to dog", and a machine part. Four things
    separate the everyday word from the marginal one: the part of speech the
    category expects, how early the gloss sits in the entry, how short the
    headword is (basic vocabulary is one or two syllables), and -- decisive
    where it applies -- whether the corpus already uses that form.
    """
    s = 0.0
    if norm(cand["dz"]) in corpus_forms:
        s += 8.0
    if cand["pos"] in wanted_pos:
        s += 6.0
    if cand["sense"] == 1:
        s += 2.0
    s += max(0.0, 2.0 - 0.5 * cand["gloss_pos"])
    s -= min(3.0, 0.15 * (cand["n_glosses"] - 1))
    s -= min(3.0, 0.35 * (cand["dz"].count("\u0f0b") - 1))
    if cand["honorific"]:
        s -= 1.5
    return s


def lookup(index, term, wanted_pos, corpus_forms, exact_only=False):
    spellings = [term.lower().strip()] if exact_only else list(variants(term))
    for spelling in spellings:
        hits = index.get(spelling)
        if not hits:
            continue
        ranked = sorted(hits, key=lambda c: -score(c, wanted_pos, corpus_forms))
        seen, out = set(), []
        for c in ranked:
            if norm(c["dz"]) in seen:
                continue
            seen.add(norm(c["dz"]))
            out.append(c)
        return spelling, out
    return None, []


def load_appendix(index):
    """The appendix tables are English->Dzongkha already; fold them in so the
    proper-noun categories have something to match against."""
    for path, en_col, dz_col in APPENDIX:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                en, dz = (row.get(en_col) or "").strip(), (row.get(dz_col) or "").strip()
                if en and dz:
                    index[en.lower()].append({
                        "dz": dz, "pos": "noun", "page": 0, "sense": 1, "note": "",
                        "gloss_pos": 0, "n_glosses": 1, "honorific": False,
                        "source": os.path.basename(path),
                    })


def corpus_forms_of(row):
    """Every Dzongkha form the corpus scan already proposed for this row."""
    forms = {norm((row.get("dz_preferred") or "").strip())}
    for chunk in (row.get("dz_candidates") or "").split("|"):
        chunk = re.sub(r"\([\d.]+\)\s*$", "", chunk.strip())
        if chunk.strip():
            forms.add(norm(chunk.strip()))
    return {f for f in forms if f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms-dir", default="terms")
    ap.add_argument("--entries", default="dict/entries.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index = load_entries(args.entries)
    load_appendix(index)
    totals = collections.Counter()

    for path in sorted(glob.glob(os.path.join(args.terms_dir, "*.csv"))):
        category = os.path.splitext(os.path.basename(path))[0]
        wanted_pos = CATEGORY_POS[category]
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        filled = confirmed = disagreed = unfound = 0
        for row in rows:
            en = (row.get("en") or "").strip()
            if not en:
                continue
            current = (row.get("dz_preferred") or "").strip()
            confidence = (row.get("confidence") or "").strip().lower()
            spelling, hits = lookup(index, en, wanted_pos, corpus_forms_of(row),
                                    exact_only=category in PROPER_NOUN)
            totals["rows"] += 1

            if not hits:
                unfound += 1
                totals["unfound"] += 1
                if not current:
                    row["notes"] = join_note(row, "not in DDC dictionary")
                continue

            best = hits[0]
            agrees = norm(best["dz"]) in corpus_forms_of(row)
            alts = [c["dz"] for c in hits[1:4]]
            provenance = (f"DDC {best['source']}" if best.get("source")
                          else f"DDC p.{best['page']} {best['pos']}")
            if spelling != en.lower().strip():
                provenance += f" (as '{spelling}')"
            if best["note"]:
                provenance += f" [{best['note']}]"

            if norm(best["dz"]) == norm(current):
                confirmed += 1
                totals["confirmed"] += 1
                if confidence in ("low", "missing", ""):
                    row["confidence"] = "dict"
                    row["review"] = ""
                row["notes"] = join_note(row, provenance + "; agrees with corpus")
            elif confidence in ("high", "medium"):
                # the corpus form is attested in real sentences, so keep it and
                # let a reviewer settle the difference
                disagreed += 1
                totals["disagreed"] += 1
                row["review"] = "YES - dictionary gives " + best["dz"]
                row["notes"] = join_note(row, provenance + f"; corpus kept ({confidence})")
            else:
                filled += 1
                totals["filled"] += 1
                if current:
                    provenance += f"; replaced corpus guess {current}"
                row["dz_preferred"] = corpus_form(best["dz"])
                row["confidence"] = "dict"
                # Two independent sources agreeing is as good as review gets;
                # a dictionary form the corpus never proposed still wants eyes.
                row["review"] = "" if agrees else "check - corpus proposed something else"
                row["notes"] = join_note(row, provenance)

            if alts:
                existing = [v.strip() for v in (row.get("dz_variants") or "").split("|") if v.strip()]
                merged = list(dict.fromkeys(existing + [corpus_form(a) for a in alts]))
                row["dz_variants"] = " | ".join(
                    m for m in merged if norm(m) != norm(row["dz_preferred"]))

        print(f"{category:15s} filled {filled:3d}  confirmed {confirmed:3d}  "
              f"disagreed {disagreed:3d}  not in dictionary {unfound:3d}")

        if not args.dry_run:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    print(f"\n{totals['rows']} rows: {totals['filled']} filled, "
          f"{totals['confirmed']} confirmed, {totals['disagreed']} disagreed, "
          f"{totals['unfound']} not in the dictionary")
    if args.dry_run:
        print("dry run -- nothing written")


def join_note(row, note):
    old = (row.get("notes") or "").strip()
    # drop whatever a previous run left, so notes do not pile up
    old = "; ".join(part for part in old.split("; ")
                    if not part.startswith(("DDC ", "not in DDC")))
    return f"{old}; {note}" if old else note


if __name__ == "__main__":
    main()
