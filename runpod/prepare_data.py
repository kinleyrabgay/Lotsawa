"""Build a clean, bidirectional Dzongkha<->English dataset for NLLB fine-tuning.

Loads kinleyrabgay/dz_to_en, removes duplicates and train/eval leakage, then
writes a DatasetDict to disk where every row is a flat (src, tgt, src_lang,
tgt_lang) record. Each original pair appears twice: once dz->en, once en->dz.
Training one model on both directions gives the low-resource side extra signal.
"""

import argparse
import csv

from datasets import Dataset, DatasetDict, load_dataset

from normalize import normalize_pair

DZ = "dzo_Tibt"
EN = "eng_Latn"


def flatten(split, both_ways, normalize=True):
    """Turn {'translation': {'dz':..., 'en':...}} rows into flat directional rows.

    With normalize=True (the default) each pair is passed through
    normalize.normalize_pair first, which restores the shad on the Dzongkha side
    and the casing/commas/terminal punctuation on the English side. Without it
    the model learns to emit the corpus's stripped orthography.
    """
    src, tgt, src_lang, tgt_lang = [], [], [], []
    for row in split["translation"]:
        dz, en = row["dz"].strip(), row["en"].strip()
        if not dz or not en:
            continue
        if normalize:
            dz, en = normalize_pair(dz, en)
        src.append(dz)
        tgt.append(en)
        src_lang.append(DZ)
        tgt_lang.append(EN)
        if both_ways:
            src.append(en)
            tgt.append(dz)
            src_lang.append(EN)
            tgt_lang.append(DZ)
    return {"src": src, "tgt": tgt, "src_lang": src_lang, "tgt_lang": tgt_lang}


# The Hub splits are contiguous slices of dataset.csv in file order -- verified
# by matching row counts (232,489 = 225,565 + 3,436 + 3,488).
CSV_SPLITS = (("train", 0, 225565), ("validation", 225565, 229001),
              ("test", 229001, 232489))


def load_csv_splits(path):
    """Rebuild the Hub's DatasetDict from the local CSV."""
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header: dz,en
        rows = [{"dz": a, "en": b} for a, b in reader]
    if len(rows) != CSV_SPLITS[-1][2]:
        raise SystemExit(
            f"{path} has {len(rows)} rows; expected {CSV_SPLITS[-1][2]}. "
            "Split boundaries would not match the Hub dataset -- pass --dataset "
            "instead, or update CSV_SPLITS."
        )
    return DatasetDict({
        name: Dataset.from_dict({"translation": rows[lo:hi]})
        for name, lo, hi in CSV_SPLITS
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kinleyrabgay/dz_to_en")
    ap.add_argument("--csv", default="",
                    help="Read from a local two-column dz,en CSV instead of the "
                         "Hub, using the same split sizes. Useful when the Hub "
                         "dataset is private and no token is configured.")
    ap.add_argument("--out", default="/workspace/data/dz_en_bidi")
    ap.add_argument(
        "--single-direction",
        action="store_true",
        help="Only dz->en. Default builds both directions.",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Skip orthography restoration. Only for an A/B against the "
             "stripped corpus -- the resulting model emits spaced, shad-less "
             "Dzongkha and lowercase unpunctuated English.",
    )
    args = ap.parse_args()

    if args.csv:
        raw = load_csv_splits(args.csv)
    else:
        raw = load_dataset(args.dataset)
    print("Loaded:", {k: len(v) for k, v in raw.items()})

    # Dedupe train, then drop any train pair that also appears in val/test.
    # 53 exact test pairs and 127 shared source sides leak in the original splits;
    # leaving them in inflates BLEU.
    eval_pairs = set()
    eval_srcs = set()
    for name in ("validation", "test"):
        for row in raw[name]["translation"]:
            dz, en = row["dz"].strip(), row["en"].strip()
            eval_pairs.add((dz, en))
            eval_srcs.add(dz)
            eval_srcs.add(en)

    seen = set()
    kept = []
    dropped_dupe = dropped_leak = 0
    for row in raw["train"]["translation"]:
        dz, en = row["dz"].strip(), row["en"].strip()
        if not dz or not en:
            continue
        key = (dz, en)
        if key in seen:
            dropped_dupe += 1
            continue
        if key in eval_pairs or dz in eval_srcs or en in eval_srcs:
            dropped_leak += 1
            continue
        seen.add(key)
        kept.append({"dz": dz, "en": en})

    print(f"Train: kept {len(kept)}, dropped {dropped_dupe} dupes, {dropped_leak} leaked")

    both = not args.single_direction
    norm = not args.raw
    print(f"Orthography restoration: {'ON' if norm else 'OFF (--raw)'}")
    out = DatasetDict(
        {
            "train": Dataset.from_dict(flatten({"translation": kept}, both, norm)),
            # Eval splits stay single-direction per row but keep both directions
            # so we can report dz->en and en->dz separately.
            "validation": Dataset.from_dict(
                flatten(raw["validation"], both, norm)
            ),
            "test": Dataset.from_dict(flatten(raw["test"], both, norm)),
        }
    )
    print("Final:", {k: len(v) for k, v in out.items()})
    print("Example:", out["train"][0])
    print("Example:", out["train"][1])

    out.save_to_disk(args.out)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
