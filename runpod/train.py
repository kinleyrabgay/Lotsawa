"""Fine-tune NLLB-200 on Dzongkha<->English.

Fixes the issues in the original notebook:
  * real FLORES-200 language codes (dzo_Tibt / eng_Latn) via tokenizer.src_lang,
    not Marian-style ">>xx<<" text prefixes
  * labels padded with -100 so padding does not contribute to the loss
  * dynamic padding via the collator instead of padding every row to max_length
  * forced_bos_token_id set per direction, so generation targets the right language
  * bf16 by default (fp16 overflows on NLLB), full corpus, resumable checkpoints

Run:
  python train.py --data /workspace/data/dz_en_bidi --out /workspace/ckpt
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import sacrebleu
import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DZ = "dzo_Tibt"
EN = "eng_Latn"


def tokenize_by_direction(dataset, tokenizer, max_length):
    """Tokenize with the correct language codes.

    tokenizer.src_lang / tgt_lang are global state, so we cannot vary them
    row-by-row inside a batched map. Instead we group rows by direction, set the
    codes once per group, and concatenate. The target-language token that the
    tokenizer prepends to the labels is what tells the model which direction to
    decode, so a single model handles both.
    """
    parts = []
    for src_lang, tgt_lang in ((DZ, EN), (EN, DZ)):
        subset = dataset.filter(
            lambda ex, s=src_lang: ex["src_lang"] == s, num_proc=8
        )
        if len(subset) == 0:
            continue
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        def encode(batch):
            return tokenizer(
                batch["src"],
                text_target=batch["tgt"],
                max_length=max_length,
                truncation=True,
            )

        parts.append(
            subset.map(
                encode,
                batched=True,
                num_proc=8,
                remove_columns=subset.column_names,
                desc=f"tokenize {src_lang}->{tgt_lang}",
            )
        )
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def build_metrics(tokenizer, target_lang):
    """BLEU + chrF++ for one decoding direction.

    chrF++ matters here: default BLEU tokenization is unreliable on Tibetan
    script, so it is the number to trust for en->dz.
    """
    tibetan = target_lang == DZ

    def compute(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        hyp = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        ref = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        bleu = sacrebleu.corpus_bleu(
            hyp, [ref], tokenize="flores200" if tibetan else "13a"
        )
        chrf = sacrebleu.corpus_chrf(hyp, [ref], word_order=2)
        return {"bleu": bleu.score, "chrf2": chrf.score}

    return compute


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def save_provenance(out_dir, args, row_counts):
    """Write everything needed to explain and reproduce this checkpoint.

    A checkpoint alone is not a deliverable. The Dzongkha side of the training
    data was respaced against boundary_markers.txt and had its shad restored
    under a specific SHAD_AFTER_GA setting, so a model trained on it expects
    input preprocessed the same way. Losing those files means losing the ability
    to feed the model correctly -- so they ship inside the checkpoint directory,
    not beside it.
    """
    os.makedirs(out_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    prep_dir = os.path.join(out_dir, "preprocessing")
    os.makedirs(prep_dir, exist_ok=True)
    contract = {}
    for name in ("normalize.py", "respace.py", "boundary_markers.txt"):
        src = os.path.join(here, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(prep_dir, name))
            contract[name] = sha256(src)

    shad_after_ga = None
    marker_count = None
    try:
        import normalize as _norm
        shad_after_ga = _norm.SHAD_AFTER_GA
        marker_count = len(_norm.BOUNDARY_MARKERS)
    except Exception:
        pass

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_sha = None

    import torch as _torch
    import transformers as _tf

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.model,
        "git_sha": git_sha,
        "args": vars(args),
        "row_counts": row_counts,
        "preprocessing": {
            "shad_after_ga": shad_after_ga,
            "boundary_marker_count": marker_count,
            "file_hashes": contract,
        },
        "env": {
            "torch": _torch.__version__,
            "transformers": _tf.__version__,
            "gpu": _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None,
        },
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    with open(os.path.join(out_dir, "PREPROCESSING.md"), "w") as fh:
        fh.write(
            "# Preprocessing contract\n\n"
            "This model was trained on Dzongkha text that was **respaced and "
            "shad-restored**, not on raw corpus text. Input at inference must be "
            "preprocessed the same way or quality degrades.\n\n"
            f"- `SHAD_AFTER_GA = {shad_after_ga}`\n"
            f"- {marker_count} phrase-boundary markers in "
            "`preprocessing/boundary_markers.txt`\n"
            f"- Trained with `--max-length {args.max_length}`; longer input is "
            "truncated\n\n"
            "Use `preprocessing/normalize.py` on word-segmented corpus text. Text "
            "already in conventional Dzongkha orthography (shad present, phrase "
            "spacing) passes through as-is.\n\n"
            "See `run_manifest.json` for the full training configuration.\n"
        )
    print(f"Provenance written to {out_dir}: run_manifest.json, PREPROCESSING.md, "
          f"preprocessing/ ({len(contract)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--data", default="/workspace/data/dz_en_bidi")
    ap.add_argument("--extra-data", default="",
                    help="Additional flat dataset to concatenate into training, "
                         "e.g. the synthetic pairs from backtranslate.py. Eval "
                         "always stays on the real held-out data.")
    ap.add_argument("--out", default="/workspace/ckpt")
    ap.add_argument("--hub-id", default="")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--eval-subset", type=int, default=1000)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="Only for pre-Ampere GPUs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="Rotating checkpoints kept on disk, plus the best one, "
                         "which is exempt from rotation. Each 600M checkpoint is "
                         "~7.4GB (2.46GB weights + 4.92GB Adam states), so the "
                         "default holds ~22GB -- sized for a 50GB network volume "
                         "that also caches the base model. Raise it only if you "
                         "provisioned more disk.")
    args = ap.parse_args()

    bf16 = not args.fp16 and torch.cuda.is_bf16_supported()
    print(f"GPU: {torch.cuda.get_device_name(0)}  bf16={bf16}  fp16={args.fp16}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=DZ, tgt_lang=EN)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    ds = load_from_disk(args.data)
    print("Data:", {k: len(v) for k, v in ds.items()})

    train_parts = [ds["train"]]
    if args.extra_data:
        extra = load_from_disk(args.extra_data)
        if hasattr(extra, "keys"):
            extra = extra["train"]
        print(f"Extra training data: {len(extra)} rows from {args.extra_data}")
        train_parts.append(extra)
    train_raw = (concatenate_datasets(train_parts)
                 if len(train_parts) > 1 else train_parts[0])
    print(f"Training rows: {len(train_raw)}")

    train_ds = tokenize_by_direction(train_raw, tokenizer, args.max_length).shuffle(seed=42)

    # Written before training starts, so even an interrupted run leaves a record
    # of what it was doing.
    save_provenance(args.out, args,
                    dict({k: len(v) for k, v in ds.items()},
                         train_total=len(train_raw)))

    # In-training eval is dz->en only, on a subset: generation is slow and we
    # just need a checkpoint-selection signal. Full per-direction scores come
    # from the final test pass below.
    val_dz_en = ds["validation"].filter(lambda ex: ex["src_lang"] == DZ)
    if args.eval_subset and len(val_dz_en) > args.eval_subset:
        val_dz_en = val_dz_en.select(range(args.eval_subset))
    tokenizer.src_lang, tokenizer.tgt_lang = DZ, EN
    eval_ds = tokenize_by_direction(val_dz_en, tokenizer, args.max_length)

    eng_bos = tokenizer.convert_tokens_to_ids(EN)
    dzo_bos = tokenizer.convert_tokens_to_ids(DZ)
    model.generation_config.forced_bos_token_id = eng_bos
    model.generation_config.max_length = args.max_length

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=100,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        bf16=bf16,
        fp16=args.fp16,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        push_to_hub=bool(args.hub_id),
        hub_model_id=args.hub_id or None,
        hub_strategy="every_save" if args.hub_id else "end",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=build_metrics(tokenizer, EN),
    )

    result = trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    # Final scores on the full test split, one direction at a time with the
    # right forced BOS token.
    final = {}
    for src_lang, tgt_lang, bos in ((DZ, EN, eng_bos), (EN, DZ, dzo_bos)):
        subset = ds["test"].filter(lambda ex, s=src_lang: ex["src_lang"] == s)
        if len(subset) == 0:
            continue
        tokenizer.src_lang, tokenizer.tgt_lang = src_lang, tgt_lang
        tokenized = tokenize_by_direction(subset, tokenizer, args.max_length)
        trainer.compute_metrics = build_metrics(tokenizer, tgt_lang)
        model.generation_config.forced_bos_token_id = bos
        scores = trainer.evaluate(
            eval_dataset=tokenized,
            metric_key_prefix=f"test_{src_lang[:3]}2{tgt_lang[:3]}",
            num_beams=4,
            max_length=args.max_length,
        )
        final.update(scores)
        print(f"{src_lang}->{tgt_lang}: {scores}")

    trainer.log_metrics("test", final)
    trainer.save_metrics("test", final)
    if args.hub_id:
        trainer.push_to_hub(commit_message="Final bidirectional dz<->en model")


if __name__ == "__main__":
    main()
