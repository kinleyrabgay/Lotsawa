"""Score any Dzongkha<->English model, in both directions, on a fixed test set.

This is the Stage 0 gate. NLLB-200 already covers dzo_Tibt, so Meta shipped a
Dzongkha translator before this project started -- that model, not zero, is the
number a fine-tune has to beat. Run this on the base model first, then on your
checkpoint, and compare:

  python evaluate.py --model facebook/nllb-200-distilled-600M --data data/dz_en_bidi
  python evaluate.py --model /workspace/ckpt --data data/dz_en_bidi

Or against FLORES-200, the standard benchmark for this language pair, which makes
your numbers comparable to published work:

  python evaluate.py --model /workspace/ckpt --flores

chrF++ is the metric to trust. BLEU's default tokenizer is unreliable on Tibetan
script, so it is reported for comparability only -- do not gate on it for en->dz.
"""

import argparse
import json

import sacrebleu
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DZ = "dzo_Tibt"
EN = "eng_Latn"


# FLORES-200 devtest is 1,012 human-translated sentences and the standard
# benchmark for this pair. facebook/flores is GATED -- it needs a one-click access
# request on the Hub -- so try the ungated sources first. Never train on any of
# these.
#
#   flores_plus  : one config per language, joined on `id`. The maintained
#                  successor, published by the Open Language Data Initiative.
#   paired       : a single `dzo_Tibt-eng_Latn` config with sentence_* columns.
FLORES_SOURCES = [
    ("openlanguagedata/flores_plus", "per_language"),
    ("Muennighoff/flores200", "paired"),
    ("facebook/flores", "paired"),
]


def _flores_per_language(repo, split):
    dz = load_dataset(repo, DZ, split=split)
    en = load_dataset(repo, EN, split=split)
    by_id = {row["id"]: row["text"] for row in en}
    pairs = [(row["text"], by_id[row["id"]]) for row in dz if row["id"] in by_id]
    if not pairs:
        raise ValueError(f"{repo}: no sentences aligned on `id`")
    return pairs


def _flores_paired(repo, split):
    ds = load_dataset(repo, f"{DZ}-{EN}", split=split, trust_remote_code=True)
    return list(zip(ds[f"sentence_{DZ}"], ds[f"sentence_{EN}"]))


def load_flores(split, forced=""):
    """Load FLORES dz-en from whichever source is reachable."""
    sources = ([(forced, "per_language"), (forced, "paired")] if forced
               else FLORES_SOURCES)
    errors = []
    for repo, shape in sources:
        try:
            loader = _flores_per_language if shape == "per_language" else _flores_paired
            pairs = loader(repo, split)
            print(f"FLORES source: {repo} ({shape}, {len(pairs)} sentences)")
            return pairs
        except Exception as exc:
            errors.append(f"  {repo} [{shape}]: {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:140]}")
    raise SystemExit(
        "Could not load FLORES from any source:\n" + "\n".join(errors) +
        "\n\nfacebook/flores is gated -- request access at\n"
        "  https://huggingface.co/datasets/facebook/flores\n"
        "then re-run. Or score without FLORES: drop --flores and use\n"
        "--data with your own test split, which is already a valid comparison."
    )


def load_pairs(args):
    """Return {(src_lang, tgt_lang): [(src, ref), ...]} for both directions."""
    out = {}
    if args.flores:
        pairs = load_flores(args.flores_split, args.flores_dataset)
        out[(DZ, EN)] = pairs
        out[(EN, DZ)] = [(en, dz) for dz, en in pairs]
    else:
        ds = load_from_disk(args.data)[args.split]
        for src_lang, tgt_lang in ((DZ, EN), (EN, DZ)):
            rows = [(r["src"], r["tgt"]) for r in ds
                    if r["src_lang"] == src_lang and r["tgt_lang"] == tgt_lang]
            if rows:
                out[(src_lang, tgt_lang)] = rows
    if args.limit:
        out = {k: v[:args.limit] for k, v in out.items()}
    return out


def translate(model, tokenizer, texts, src_lang, tgt_lang, args, device):
    tokenizer.src_lang = src_lang
    bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    hyps = []
    for i in range(0, len(texts), args.batch):
        batch = texts[i:i + args.batch]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_length).to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=bos,
                                 num_beams=args.beams, max_length=args.max_length)
        hyps.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        if args.verbose and i == 0:
            for s, h in list(zip(batch, hyps))[:3]:
                print(f"    src: {s[:90]}")
                print(f"    hyp: {h[:90]}")
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint path.")
    ap.add_argument("--data", default="data/dz_en_bidi")
    ap.add_argument("--split", default="test")
    ap.add_argument("--flores", action="store_true", help="Score on FLORES-200 instead.")
    ap.add_argument("--flores-split", default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--flores-dataset", default="",
                    help="Force one FLORES source instead of trying each in turn.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="Cap sentences per direction.")
    ap.add_argument("--out", default="", help="Write results to this JSON file.")
    ap.add_argument("--verbose", action="store_true", help="Print a few translations.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=DZ, tgt_lang=EN)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    print(f"Model: {args.model}\nDevice: {device}")

    pairs = load_pairs(args)
    results = {"model": args.model,
               "data": "flores200-" + args.flores_split if args.flores else f"{args.data}:{args.split}"}

    for (src_lang, tgt_lang), rows in pairs.items():
        srcs = [s for s, _ in rows]
        refs = [r for _, r in rows]
        print(f"\n{src_lang} -> {tgt_lang}  ({len(srcs)} sentences)")
        hyps = translate(model, tokenizer, srcs, src_lang, tgt_lang, args, device)

        tibetan = tgt_lang == DZ
        # chrF++ is character-based and needs nothing downloaded, so it always
        # works. BLEU's flores200 tokenizer fetches a SentencePiece model on
        # first use, which can fail behind a proxy or a broken cert store -- do
        # not let that throw away a multi-hour generation run.
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
        tok = "flores200" if tibetan else "13a"
        try:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)
        except Exception as exc:
            print(f"  ! BLEU tokenizer '{tok}' unavailable ({type(exc).__name__}); "
                  f"falling back to 13a. chrF++ is unaffected.")
            tok = "13a"
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)

        key = f"{src_lang[:3]}2{tgt_lang[:3]}"
        results[key] = {"chrf2": round(chrf.score, 2), "bleu": round(bleu.score, 2),
                        "bleu_tokenizer": tok, "n": len(srcs)}
        star = "  <- trust this one" if tibetan else ""
        print(f"  chrF++ {chrf.score:6.2f}{star}")
        print(f"  BLEU   {bleu.score:6.2f}" + ("  (unreliable on Tibetan script)" if tibetan else ""))

    print("\n" + json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
