# Lotsawa

**Dzongkha ↔ English neural machine translation.**

ལོ་ཙཱ་བ (*lotsāwa*) is the Dzongkha and Tibetan word for *translator* — the title
borne by the great translators who rendered Sanskrit into Tibetan. They worked in
pairs: a lotsāwa who knew the target language and a paṇḍita who knew the source,
neither sufficient alone. A fitting name for a bidirectional model.

---

## What this is

A fine-tune of `facebook/nllb-200-distilled-600M` on 232,489 human-translated
Dzongkha–English pairs, trained in both directions as a single model, aimed at
daily production use rather than a benchmark number.

| | |
|---|---|
| Language codes | `dzo_Tibt` ↔ `eng_Latn` |
| Training rows | 439,120 (219,560 pairs × 2 directions) |
| Base model | NLLB-200 distilled 600M |
| Primary metric | chrF++ (BLEU is unreliable on Tibetan script) |
| Benchmark | FLORES-200 `dzo_Tibt` devtest |

## Layout

```
dataset.csv            232,489 human dz–en pairs (not in this repo — see Data)
runpod/
  RUNPOD.md            step-by-step training runbook — start here
  normalize.py         orthography restoration: shad, casing, phrase spacing
  respace.py           learns phrase-boundary markers from real Dzongkha
  boundary_markers.txt 125 learned case markers and postpositions
  prepare_data.py      dedupe, de-leak, normalize, build both directions
  fetch_monolingual.py pulls clean monolingual Dzongkha from FineTranslations
  backtranslate.py     monolingual dz → synthetic en→dz training pairs
  train.py             the fine-tune
  evaluate.py          chrF++/BLEU both directions, base model or checkpoint
  translate.py         inference
  data/
    dz_mono.txt        60,750 real Kuensel sentences (not in this repo — see Data)
    dz_en_bidi/        the prepared bidirectional dataset (derived, gitignored)
```

## Data

The corpora are deliberately not committed. Both are reproducible:

| File | How to get it |
|---|---|
| `dataset.csv` | The parallel corpus. Private — also on the Hub as `kinleyrabgay/dz_to_en`. `prepare_data.py --dataset kinleyrabgay/dz_to_en` pulls it with an `HF_TOKEN`. |
| `runpod/data/dz_mono.txt` | `python runpod/fetch_monolingual.py --out runpod/data/dz_mono.txt` — rebuilds it from FineTranslations in about a minute. |
| `runpod/data/dz_en_bidi/` | `python runpod/prepare_data.py --csv ../dataset.csv --out data/dz_en_bidi` |

`dz_mono.txt` is excluded because it is Kuensel-derived: the FineTranslations
extraction is ODC-BY, but the source articles remain their publishers' property.
Rebuild it locally rather than redistributing it.

## Quick start

```bash
cd runpod
python prepare_data.py --csv ../dataset.csv --out data/dz_en_bidi
python evaluate.py --model facebook/nllb-200-distilled-600M --flores   # the baseline to beat
python train.py --data data/dz_en_bidi --out ckpt
```

Full sequence, pod setup and troubleshooting: **[runpod/RUNPOD.md](runpod/RUNPOD.md)**

## What makes this corpus hard

Measured, not assumed — see the audit for the full picture:

- **0.00%** of the 232,489 pairs contain a digit. No dates, prices, or quantities.
- Longest sentence is **16 words**; 95% are under 10.
- Honorific register is near-absent (`ཕེབས` at 0.02%), so formal Dzongkha is untrained.
- The corpus shipped with its orthography stripped: no shad on the Dzongkha side,
  no capital letters or terminal punctuation on the English side.
- Its spacing convention (0.750 spaces per syllable) does not match published
  Dzongkha (0.212).

`normalize.py` and `respace.py` repair the last two. The first three need data,
not code — that is what `data/dz_mono.txt` and the staged plan are for.

## Provenance

Every training run writes `run_manifest.json` and a `preprocessing/` directory
into its checkpoint: the exact arguments, row counts, environment, and hashed
copies of the preprocessing code. A model trained here expects respaced,
shad-restored input — the checkpoint carries the contract that describes it.

## Attribution

- Base model: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0)
- Monolingual Dzongkha: [FineTranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations) (ODC-BY); source articles remain their publishers' property
- Benchmark: FLORES-200

Note that NLLB-200's CC-BY-NC licence is **non-commercial**. Confirm the licensing
path before shipping a paid product on top of it.
