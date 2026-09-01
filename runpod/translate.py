"""Inference for the fine-tuned bidirectional Dzongkha<->English model."""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DZ = "dzo_Tibt"
EN = "eng_Latn"


class Translator:
    def __init__(self, path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device).eval()

    @torch.inference_mode()
    def translate(self, texts, src_lang, tgt_lang, num_beams=4, max_length=64):
        if isinstance(texts, str):
            texts = [texts]
        self.tokenizer.src_lang = src_lang
        batch = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)
        out = self.model.generate(
            **batch,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            num_beams=num_beams,
            max_length=max_length,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/ckpt")
    ap.add_argument("--direction", choices=["dz2en", "en2dz"], default="dz2en")
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()

    src, tgt = (DZ, EN) if args.direction == "dz2en" else (EN, DZ)
    t = Translator(args.model)
    for line in t.translate(args.text, src, tgt):
        print(line)


if __name__ == "__main__":
    main()
