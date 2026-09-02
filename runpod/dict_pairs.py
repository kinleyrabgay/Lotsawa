"""Turn the extracted dictionary into training pairs for the v3 fine-tune.

The corpus this project trains on is 232k short sentences with a narrow
vocabulary: no digits, almost no honorifics, no proper nouns, and nothing
outside everyday register. The dictionary covers exactly those gaps -- 35,549
headwords, the four tense stems of 4,500 verbs, the honorific counterpart of 507
words, and the appendix tables of countries, capitals and number words.

What it does not give is sentences. Training on bare word pairs has a known
failure mode: a model fed enough isolated words starts answering a sentence with
a word. Two things keep that in check here.

The first is quantity. Lookup rows are capped -- by glosses per headword, and
then by a total ceiling that keeps the most central readings and drops the
marginal senses first -- so they stay a minority of the mix and the sentence
data still decides what a sentence looks like. The default ceiling puts them at
roughly a tenth of the 439k sentence rows.
Carrier substitution -- teaching a term inside real sentences -- is a separate
job that augment_terms.py already does, and the two are meant to run together.

The second is direction. Dzongkha->English lookup is safe: a headword has one
meaning-set, so the target is well defined, and this is the direction that
absorbs the honorific vocabulary and the tense stems -- teaching the model to
*read* forms the corpus never shows it. English->Dzongkha is not safe by
default: "dog" has five Dzongkha renderings in the book, and training all five
against the same source teaches the model that the target is arbitrary. So that
direction gets one form per English word, the most central reading, and skips
honorific headwords entirely -- picking a register the user did not ask for is
worse than not knowing it.

    python extract_dictionary.py
    python dict_pairs.py --dict dict --out data/dz_dict
    python train.py --model kinleyrabgay/lotsawa-600m-dz-en-v2 \
        --data data/dz_en_bidi --extra-data data/dz_dict,data/dz_terms
"""

import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import Dataset, load_from_disk

DZ = "dzo_Tibt"
EN = "eng_Latn"
TSHEG = "་"
TIBETAN = re.compile(r"[ༀ-࿿]")
LATIN = re.compile(r"[A-Za-z]")


def corpus_form(dz):
    """Citation form (སློབ་དཔོན།) -> the form a sentence uses (སློབ་དཔོན་)."""
    return re.sub(r"[་-༎\s]+$", "", dz.strip()) + TSHEG


def usable_gloss(gloss, max_words):
    gloss = gloss.strip()
    if not gloss or not LATIN.search(gloss) or TIBETAN.search(gloss):
        return ""
    if gloss.startswith("(") or len(gloss.split()) > max_words:
        return ""
    return gloss


def usable_headword(dz):
    dz = dz.strip()
    if not dz or not TIBETAN.search(dz) or LATIN.search(dz):
        return ""
    if " " in dz or dz.count(TSHEG) > 6:
        return ""      # a phrase or a whole clause, not a lookup
    return dz


def training_priority(entry, sense, gloss_pos):
    """Which readings are worth spending training rows on.

    This is not the same question as which form to pick for a given English
    word. Here a one-syllable headword is a liability: in Dzongkha a lone
    syllable is more often a bound root or a verb stem than a word anyone types,
    and the book glosses it anyway, so ranking by shortness fills the budget
    with things like \u0f5a\u0f0b -> "wall". Prefer the first sense of a specific entry
    whose headword is a plausible standalone word.
    """
    n_glosses = sum(len(s["english"]) for s in entry["senses"])
    syllables = entry["headword"].count(TSHEG)
    score = 4.0 if sense["n"] == 1 else 0.0
    score += max(0.0, 2.0 - 0.5 * gloss_pos)
    score -= min(3.0, 0.15 * (n_glosses - 1))
    score += 0.5 if 2 <= syllables <= 4 else -1.5
    if entry["relations"].get("is_honorific"):
        score -= 1.0
    return score


def centrality(entry, sense, gloss_pos):
    """How central a reading is to its headword.

    The book lists every word an English gloss can translate, marginal senses
    included. The everyday word is the one in the first sense, early in a short
    entry, spelled in few syllables, and not marked honorific.
    """
    n_glosses = sum(len(s["english"]) for s in entry["senses"])
    score = 4.0 if sense["n"] == 1 else 0.0
    score += max(0.0, 2.0 - 0.5 * gloss_pos)
    score -= min(3.0, 0.15 * (n_glosses - 1))
    score -= min(3.0, 0.35 * (entry["headword"].count(TSHEG) - 1))
    if entry["relations"].get("is_honorific"):
        score -= 1.5
    return score


def read_entries(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def lookup_pairs(entries, args):
    """dz -> en for every headword; en -> dz for the best form of each word."""
    pairs = []
    best_for_en = {}
    stats = collections.Counter()

    for entry in entries:
        head = usable_headword(entry["headword"])
        if not head:
            stats["headword_skipped"] += 1
            continue
        dz = corpus_form(head)
        honorific = bool(entry["relations"].get("is_honorific"))
        kept = 0
        for sense in entry["senses"]:
            for pos, raw in enumerate(sense["english"]):
                gloss = usable_gloss(raw, args.max_gloss_words)
                if not gloss:
                    continue
                if kept < args.glosses_per_headword:
                    pairs.append((training_priority(entry, sense, pos),
                                  (dz, gloss, DZ, EN)))
                    kept += 1
                    stats["dz_en"] += 1
                if honorific:
                    continue      # never teach English -> honorific by default
                key = gloss.lower()
                score = centrality(entry, sense, pos)
                if key not in best_for_en or score > best_for_en[key][0]:
                    best_for_en[key] = (score, training_priority(entry, sense, pos),
                                        gloss, dz)

    for _score, priority, gloss, dz in best_for_en.values():
        pairs.append((priority, (gloss, dz, EN, DZ)))
        stats["en_dz"] += 1
    return pairs, stats


def tense_pairs(entries, args):
    """Every tense stem of a verb, pointing at what the verb means.

    The corpus shows one spelling of a verb; the book gives four. Teaching the
    other three in the reading direction costs nothing and lets the model
    recognise a tense it has never seen written.
    """
    pairs, stats = [], collections.Counter()
    for entry in entries:
        if entry["pos"] != "verb" or not entry["tenses"]:
            continue
        glosses = [g for s in entry["senses"] for g in s["english"]][:args.glosses_per_headword]
        glosses = [usable_gloss(g, args.max_gloss_words) for g in glosses]
        glosses = [g for g in glosses if g]
        if not glosses:
            continue
        forms = {usable_headword(f) for f in entry["tenses"].values()}
        forms.discard(usable_headword(entry["headword"]))
        for form in filter(None, forms):
            for gloss in glosses:
                pairs.append((corpus_form(form), gloss, DZ, EN))
                stats["tense_forms"] += 1
    return pairs, stats


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def appendix_pairs(dict_dir, args):
    """Countries, capitals, Indic names and number words.

    These are the corpus's flat spots -- it contains no proper nouns and not one
    digit -- and there are only a few hundred of them, so they are repeated to
    survive a half-million-row mix.
    """
    both = []
    stats = collections.Counter()
    for row in read_tsv(os.path.join(dict_dir, "countries.tsv")):
        both.append((row["dz_country"], row["en_country"], "country"))
        both.append((row["dz_capital"], row["en_capital"], "capital"))
    for row in read_tsv(os.path.join(dict_dir, "names_indic.tsv")):
        both.append((row["dz"], row["en"], "name"))
    for row in read_tsv(os.path.join(dict_dir, "numbers.tsv")):
        both.append((row["dz"], row["en"], "number"))

    pairs = []
    for dz, en, kind in both:
        dz, en = dz.strip(), en.strip()
        if not dz or not en or not TIBETAN.search(dz) or not LATIN.search(en):
            continue
        dz = corpus_form(dz)
        for _ in range(args.appendix_repeat):
            pairs.append((dz, en, DZ, EN))
            pairs.append((en, dz, EN, DZ))
            stats[kind] += 2
    return pairs, stats


def drop_leaks(pairs, data_dir):
    """Nothing that appears in validation or test may be trained on.

    A dictionary headword is unlikely to be a held-out sentence, but "unlikely"
    is not a reason to skip the check -- a leaked eval row makes every number
    that follows meaningless.
    """
    if not data_dir or not os.path.exists(data_dir):
        print(f"WARNING  {data_dir or '--data'} not found -- eval leakage NOT "
              f"checked. Run this where the prepared dataset lives.")
        return pairs, 0
    held = set()
    splits = load_from_disk(data_dir)
    for name in ("validation", "test"):
        if name in splits:
            held.update(s.strip() for s in splits[name]["src"])
            held.update(t.strip() for t in splits[name]["tgt"])
    kept = [p for p in pairs if p[0].strip() not in held and p[1].strip() not in held]
    return kept, len(pairs) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="dict", help="Directory extract_dictionary.py wrote")
    ap.add_argument("--out", default="/workspace/data/dz_dict")
    ap.add_argument("--data", default="data/dz_en_bidi",
                    help="Prepared dataset, read only to check for eval leakage")
    ap.add_argument("--max-lookup-rows", type=int, default=55000,
                    help="Ceiling on dictionary lookup rows. Bare word pairs "
                         "teach vocabulary but, past roughly a tenth of the "
                         "sentence data, also teach the model to answer a "
                         "sentence with a word.")
    ap.add_argument("--glosses-per-headword", type=int, default=2,
                    help="Cap on English glosses trained per Dzongkha headword. "
                         "The book lists up to 30 for a broad word; the first "
                         "few are the ones anyone means.")
    ap.add_argument("--max-gloss-words", type=int, default=4,
                    help="Longer glosses are definitions, not translations.")
    ap.add_argument("--appendix-repeat", type=int, default=3)
    ap.add_argument("--no-tenses", action="store_true")
    args = ap.parse_args()

    entries = list(read_entries(os.path.join(args.dict, "entries.jsonl")))
    print(f"Read {len(entries)} dictionary entries")

    scored, stats = lookup_pairs(entries, args)
    scored.sort(key=lambda p: -p[0])
    if len(scored) > args.max_lookup_rows:
        print(f"Capping lookup rows at {args.max_lookup_rows} of {len(scored)}: "
              f"the marginal senses go first")
        scored = scored[:args.max_lookup_rows]
    pairs = [p for _, p in scored]
    if not args.no_tenses:
        more, s = tense_pairs(entries, args)
        pairs += more
        stats.update(s)
    more, s = appendix_pairs(args.dict, args)
    pairs += more
    stats.update(s)

    before = len(pairs)
    pairs = list(dict.fromkeys(pairs))
    print(f"\n{before} rows, {before - len(pairs)} exact duplicates removed")

    pairs, leaked = drop_leaks(pairs, args.data)
    if leaked:
        print(f"Dropped {leaked} rows that appear in validation or test")

    for key in sorted(stats):
        print(f"  {key:22s} {stats[key]:7d}")
    directions = collections.Counter(p[2] for p in pairs)
    print(f"\n  dz->en {directions[DZ]}, en->dz {directions[EN]}, total {len(pairs)}")

    ds = Dataset.from_dict({
        "src": [p[0] for p in pairs],
        "tgt": [p[1] for p in pairs],
        "src_lang": [p[2] for p in pairs],
        "tgt_lang": [p[3] for p in pairs],
    })
    ds.save_to_disk(args.out)
    print(f"Wrote {len(ds)} rows -> {args.out}")


if __name__ == "__main__":
    main()
