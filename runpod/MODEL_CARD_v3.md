---
license: cc-by-nc-4.0
language:
  - dz
  - en
base_model: facebook/nllb-200-distilled-600M
pipeline_tag: translation
tags:
  - dzongkha
  - bhutan
  - nllb
  - lotsawa
---

# Lotsawa 600M dz↔en — v3

Bidirectional Dzongkha ↔ English translation. A continued fine-tune of
[`kinleyrabgay/lotsawa-600m-dz-en-v2`](https://huggingface.co/kinleyrabgay/lotsawa-600m-dz-en-v2),
which is itself a fine-tune of NLLB-200 distilled 600M.

| | |
|---|---|
| Language codes | `dzo_Tibt` ↔ `eng_Latn` |
| Base of the series | `facebook/nllb-200-distilled-600M` |
| Initialised from | `kinleyrabgay/lotsawa-600m-dz-en-v2` |
| Metric | chrF++ (BLEU is unreliable on Tibetan script) |

## What v3 adds

v2 was trained on 232k human-translated sentence pairs plus back-translated
monolingual Dzongkha. That corpus has measured gaps: no digits at all, a longest
sentence of 16 words, almost no honorific register, and no proper nouns. v3 adds
what a dictionary knows and a sentence corpus does not.

The source is the *Dzongkha–English Pocket Dictionary*, 2nd edition (Dzongkha
Development Commission, 2013): 35,549 headwords, the tense paradigms of ~4,500
verbs, 507 plain/honorific pairs, and appendix tables of countries, capitals and
number words. Three kinds of row come out of it:

- **Lookup pairs**, capped at ~55k and ranked so marginal senses are dropped
  first. Dzongkha→English carries the whole vocabulary including honorific
  headwords and the tense stems the corpus never shows. English→Dzongkha gets
  one form per word — the most central reading — and no honorifics, because
  picking a register the user did not ask for is worse than not knowing one.
- **Carrier sentences**: dictionary-confirmed terms substituted into real
  corpus sentences, so a term is learned in context rather than as a word pair.
- **Appendix pairs** for countries, capitals and numbers, repeated so a few
  hundred rows survive a half-million-row mix.

Dictionary-derived rows are roughly 13% of the training mix. The rest is the v2
data, replayed: fine-tuning on word pairs alone teaches a model to answer a
sentence with a word.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "kinleyrabgay/lotsawa-600m-dz-en-v3"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="dzo_Tibt")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

batch = tok("ང་ཡིག་ཚང་ནང་ མེད།", return_tensors="pt")
out = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"))
print(tok.batch_decode(out, skip_special_tokens=True))
```

**Input contract.** This model expects respaced, shad-restored Dzongkha and
ordinary punctuated English — the orthography `normalize.py` and `respace.py`
produce. The checkpoint ships `preprocessing/` with the exact code and
`run_manifest.json` with the exact arguments; feed it corpus-style stripped text
and quality will drop.

## Limitations

- The parallel corpus contains no digits and few long sentences. Numerals reach
  the model only through back-translation and the dictionary's number words;
  check any output containing a date or an amount.
- Honorific Dzongkha is understood better than it is produced. The model is
  taught to read honorific forms, not to choose them — there is no register
  control token.
- FLORES-200 measures ordinary prose and will understate a vocabulary gain.

## Licence and attribution

- Base model: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M),
  **CC-BY-NC 4.0 — non-commercial**. That licence flows through to this model.
  Confirm the licensing path before shipping a paid product on it.
- Monolingual Dzongkha: [FineTranslations](https://huggingface.co/datasets/HuggingFaceFW/finetranslations)
  (ODC-BY); source articles remain their publishers' property.
- Lexicon: *Dzongkha–English Pocket Dictionary*, 2nd ed., Dzongkha Development
  Commission, 2013, ISBN 978-99936-15-21-7. © DDC, all rights reserved. The
  extracted tables are used as training data and are not redistributed.
- Benchmark: FLORES-200.
