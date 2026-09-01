"""Turn monolingual Dzongkha into synthetic training pairs for en->dz.

This is how data/dz_mono.txt enters training. The direction is not arbitrary:
back-translation works because the REAL human text ends up on the TARGET side.
Monolingual Dzongkha therefore improves en->dz, where the model learns to
produce genuine Dzongkha from machine-made English. Doing the reverse -- putting
machine-made Dzongkha on the target side -- would teach the model to imitate its
own errors. For dz->en you need monolingual English, which is abundant.

Order of operations, because this step has a prerequisite:

  1. Train a dz->en model first (Stage 0). That is the "backward" model here.
  2. Run this script to translate data/dz_mono.txt into English.
  3. Retrain on the real corpus plus these synthetic pairs.

Why it is worth the round trip: 47.3% of the sentences in dz_mono.txt carry
Tibetan numerals, and the parallel corpus contains none at all. Those sentences
are also 48 syllables at the median against your corpus's 9, so they carry the
long-sentence and news-register signal that nothing else in your data provides.

  python backtranslate.py --model /workspace/ckpt --mono data/dz_mono.txt \
      --out /workspace/data/dz_en_bt
"""

import argparse
import random
import re

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DZ = "dzo_Tibt"
EN = "eng_Latn"

TIBETAN_DIGITS = "༠༡༢༣༤༥༦༧༨༩"
ARABIC_DIGITS = "0123456789"
TIB_TO_ARABIC = str.maketrans(TIBETAN_DIGITS, ARABIC_DIGITS)
ARABIC_TO_TIB = str.maketrans(ARABIC_DIGITS, TIBETAN_DIGITS)

# Prepended to the synthetic English source only. The model learns that tagged
# input is machine-made and should be trusted less than untagged input, which
# measurably helps over mixing synthetic and real data indistinguishably. Real
# input at inference is never tagged -- that is the point.
BT_TAG = "<bt> "


def digits_in(text):
    """Digit sequences in a string, with Tibetan numerals folded to Arabic."""
    return set(re.findall(r"\d+", text.translate(TIB_TO_ARABIC)))


def digit_conflict(dz, en):
    """True only when the English contradicts the Dzongkha numbers.

    Naively requiring digits_in(dz) == digits_in(en) is wrong. Measured on real
    Kuensel sentences, the dz->en model spells small numbers as English words
    ("Ten army officers") while keeping large ones as digits ("350 meters"). A
    spelled-out number is not an error -- and the pair "Ten ..." / "...༡༠..." is
    precisely the word-to-digit mapping en->dz needs to learn, since published
    Dzongkha writes numerals as Tibetan digits.

    So reject only a real contradiction: the English states digits that differ
    from the Dzongkha's. English with no digits at all is kept.
    """
    en_d = digits_in(en)
    if not en_d:
        return False
    return en_d != digits_in(dz)


def substitute_digits(dz, en, rng):
    """Swap every number in an aligned pair for a different one, consistently.

    Only valid when both sides carry the same digits. Turns one numeral example
    into many, which is how a corpus with 0.00% digit coverage learns arithmetic
    surface forms without inventing Dzongkha grammar.
    """
    numbers = sorted(digits_in(dz), key=len, reverse=True)
    if not numbers:
        return None
    new_dz, new_en = dz, en
    for i, num in enumerate(numbers):
        # Same digit count keeps the result plausible (a year stays a year).
        lo = 10 ** (len(num) - 1) if len(num) > 1 else 0
        hi = 10 ** len(num) - 1
        repl = str(rng.randint(lo, hi)).zfill(len(num))
        if repl == num:
            return None
        hole = f"\x00{i}\x00"
        new_en = new_en.replace(num, hole)
        new_dz = new_dz.replace(num.translate(ARABIC_TO_TIB), hole)
        new_en = new_en.replace(hole, repl)
        new_dz = new_dz.replace(hole, repl.translate(ARABIC_TO_TIB))
    if new_dz == dz or new_en == en:
        return None
    return new_dz, new_en


def has_repetition(text, n=4, limit=3):
    """True if any n-gram repeats more than `limit` times -- a decoding failure."""
    words = text.split()
    if len(words) < n * 2:
        return False
    grams = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] > limit:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="A trained dz->en checkpoint.")
    ap.add_argument("--mono", default="data/dz_mono.txt")
    ap.add_argument("--out", default="/workspace/data/dz_en_bt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=192,
                    help="These sentences are long -- 48 syllables at the median. "
                         "The 64 used for the short parallel corpus would truncate "
                         "most of them.")
    ap.add_argument("--limit", type=int, default=0, help="Cap sentences, for a smoke test.")
    ap.add_argument("--beam", action="store_true",
                    help="Beam search instead of sampling. Sampling is the default "
                         "because it yields more varied, more useful synthetic "
                         "source text; beam output is cleaner but less informative.")
    ap.add_argument("--no-tag", action="store_true", help="Omit the <bt> tag.")
    ap.add_argument("--no-digit-filter", action="store_true",
                    help="Keep pairs whose digits do not round-trip. Use when the "
                         "digit-mismatch drop rate is high because the model spells "
                         "numbers as Dzongkha words instead of digits.")
    ap.add_argument("--digit-aug", type=int, default=4,
                    help="Extra copies of each digit-preserving pair, with the "
                         "numbers substituted consistently on both sides. 0 to "
                         "disable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ratio", type=float, default=0.25)
    ap.add_argument("--max-ratio", type=float, default=3.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=DZ, tgt_lang=EN)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()

    lines = [l.strip() for l in open(args.mono, encoding="utf-8") if l.strip()]
    if args.limit:
        lines = lines[:args.limit]
    print(f"Back-translating {len(lines)} Dzongkha sentences on {device}")

    gen_kwargs = dict(
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(EN),
        max_length=args.max_length,
    )
    if args.beam:
        gen_kwargs.update(num_beams=4)
    else:
        gen_kwargs.update(do_sample=True, top_k=50, temperature=0.9, num_beams=1)

    pairs = []
    dropped = {"empty": 0, "ratio": 0, "repeat": 0, "digits": 0}
    digit_samples = []
    augmented = 0
    rng = random.Random(args.seed)

    for start in range(0, len(lines), args.batch):
        batch = lines[start:start + args.batch]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        hyps = tokenizer.batch_decode(out, skip_special_tokens=True)

        for dz, en in zip(batch, hyps):
            en = en.strip()
            if not en:
                dropped["empty"] += 1
                continue
            ratio = len(en.split()) / max(len(dz.split()), 1)
            if not (args.min_ratio <= ratio <= args.max_ratio):
                dropped["ratio"] += 1
                continue
            if has_repetition(en):
                dropped["repeat"] += 1
                continue
            # The whole point of this data is numerals, so a pair that lost or
            # invented one is worse than no pair at all.
            #
            # Caveat worth watching: the model often renders numbers as Dzongkha
            # *words* rather than digits, so a source carrying ༣༡ can translate to
            # "thirty-one" and be rejected here despite being a good pair. Check
            # the reported drop rate on a --limit run; if it is high, the filter is
            # discarding the very sentences this data exists to provide, and
            # --no-digit-filter is the escape hatch.
            if not args.no_digit_filter and digit_conflict(dz, en):
                dropped["digits"] += 1
                if len(digit_samples) < 5:
                    digit_samples.append((dz[:70], en[:70]))
                continue
            def emit(dz_text, en_text):
                pairs.append({
                    "src": en_text if args.no_tag else BT_TAG + en_text,
                    "tgt": dz_text,
                    "src_lang": EN,
                    "tgt_lang": DZ,
                })

            emit(dz, en)

            # Where the numbers round-trip as digits on both sides, multiply the
            # example. This is the cheapest route to numeral coverage: the corpus
            # has none, and the substitution is arithmetic, not grammar.
            if args.digit_aug and digits_in(dz) and digits_in(dz) == digits_in(en):
                for _ in range(args.digit_aug):
                    variant = substitute_digits(dz, en, rng)
                    if variant:
                        emit(*variant)
                        augmented += 1

        done = start + len(batch)
        if done % (args.batch * 20) == 0:
            print(f"  {done}/{len(lines)}  kept {len(pairs)}")

    print(f"\nKept {len(pairs)} synthetic pairs "
          f"({augmented} from digit augmentation)")
    print(f"Dropped -- empty {dropped['empty']}, length ratio {dropped['ratio']}, "
          f"repetition {dropped['repeat']}, digit mismatch {dropped['digits']}")

    considered = len(pairs) + sum(dropped.values())
    digit_rate = 100 * dropped["digits"] / max(considered, 1)
    if digit_rate > 25:
        print(f"\nWARNING  {digit_rate:.0f}% of pairs were dropped on digit "
              f"mismatch.\n"
              f"         The model likely spells numbers as Dzongkha words rather "
              f"than digits,\n"
              f"         which means this filter is discarding the numeral-bearing "
              f"sentences you\n"
              f"         most want. Inspect these, then consider "
              f"--no-digit-filter:")
        for dz, en in digit_samples:
            print(f"           dz: {dz}")
            print(f"           en: {en}")

    Dataset.from_list(pairs).save_to_disk(args.out)
    print(f"Wrote {args.out}")
    print("\nNext: retrain with both sets, and raise --max-length for the longer data:")
    print("  python train.py --data /workspace/data/dz_en_bidi \\")
    print(f"      --extra-data {args.out} --max-length 128 --out /workspace/ckpt-bt")


if __name__ == "__main__":
    main()
