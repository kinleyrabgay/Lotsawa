"""Translate a fixed battery of test sentences, both directions.

A quick qualitative read on a checkpoint, to sit alongside evaluate.py's scores.
The numeral cases at the end are the important ones: the parallel corpus contains
zero digits, so before back-translation the model has never seen a number and
will usually drop or invent them. That failure is expected here -- it is what
Stage 1 exists to fix, and this is how you watch it get fixed.

    python demo.py --model /workspace/ckpt
    python demo.py --model /workspace/ckpt --cpu      # leave the GPU to training
"""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DZ = "dzo_Tibt"
EN = "eng_Latn"

DZ_TO_EN = [
    "ང་ཡིག་ཚང་ནང་ མེད།",
    "ཁྱོད་ཀྱི་སྨན་ཚོང་ཁང་གི་མིང་ག་ཅི་སྨོ།",
    "ཁོ་གིས་རྒྱ་གར་ནང་ངོ་རྒོལ་བཀག་ཡི།",
    "རྒྱལ་ཁབ་གཉིས་ཀྱི་བར་ན་ཕྱི་འབྲེལ་སྲིད་སྐྱོང་གི་མཐུན་འབྲེལ་མེད།",
]

EN_TO_DZ = [
    "I am not in office.",
    "The two countries do not have diplomatic relations.",
    "What is the name of your pharmacy?",
    "A true gentleman never betrays his friends.",
]

# The corpus has 0.00% digit coverage, so these probe a genuine blind spot.
NUMERALS = [
    "Submit the form by 15 January 2027 and pay Nu. 4,500.",
    "The meeting starts at 9:30 and lasts 45 minutes.",
    "Only 12 of the 340 households have received compensation.",
]

# Formal register: honorifics are under 1.6% of the corpus.
FORMAL = [
    "The Minister will arrive tomorrow and address the Assembly.",
    "Please be seated; His Excellency will speak shortly.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/ckpt")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU. Use this while a training run owns the GPU.")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    print(f"model  {args.model}\ndevice {device}\n")

    def translate(texts, src_lang, tgt_lang):
        tok.src_lang = src_lang
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=args.max_length).to(device)
        with torch.inference_mode():
            out = model.generate(
                **batch,
                forced_bos_token_id=tok.convert_tokens_to_ids(tgt_lang),
                num_beams=args.beams,
                max_length=args.max_length,
            )
        return tok.batch_decode(out, skip_special_tokens=True)

    for title, texts, src, tgt in (
        ("dz -> en", DZ_TO_EN, DZ, EN),
        ("en -> dz", EN_TO_DZ, EN, DZ),
        ("en -> dz  NUMERALS (corpus has 0.00% digits)", NUMERALS, EN, DZ),
        ("en -> dz  FORMAL REGISTER (honorifics <1.6%)", FORMAL, EN, DZ),
    ):
        print(f"=== {title} ===")
        for src_text, hyp in zip(texts, translate(texts, src, tgt)):
            print(f"  {src_text}")
            print(f"   -> {hyp}\n")


if __name__ == "__main__":
    main()
