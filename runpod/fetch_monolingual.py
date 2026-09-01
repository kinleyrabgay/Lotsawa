"""Pull clean monolingual Dzongkha out of HuggingFaceFW/finetranslations.

Why this file exists: your parallel corpus has no numbers, nothing over 16 words,
and almost no formal register. Monolingual Dzongkha news prose fixes all three
via back-translation -- and FineTranslations already extracted it from Common
Crawl under ODC-BY, so there is nothing to scrape.

The catch is that its dzo_Tibt split is heavily contaminated. In an 800-row
sample: ~25% is genuine Bhutanese Dzongkha (Kuensel), ~33% is Tibetan from VOA
Tibetan, RFA, tibettimes and Chinese Tibetan portals, and ~11% is LibreOffice UI
localization. Dzongkha and Tibetan share a script, so the language classifier
conflates them -- 43.8% of rows score below 0.75 confidence.

So we filter on provenance rather than trusting the language label, because the
domain is a far better signal than the classifier. Do NOT use the dataset's
`translated_text` field as an English reference: it is machine translation, not
human, and using it as a target would just distill another MT system.

Output is one Dzongkha sentence per line, ready to back-translate.

  python fetch_monolingual.py --out /workspace/data/dz_mono.txt
"""

import argparse
import re
import unicodedata

from datasets import load_dataset

# Bhutanese Dzongkha publishers. Add to this list only after eyeballing a sample
# from the domain -- a Tibetan site slipping in here silently poisons training.
DZONGKHA_DOMAINS = {
    "dzkuensel.com",
    "www.dzkuensel.com",
    "dzkuensel.bt",
    "www.dzkuensel.bt",
    "kuenseldzongkhaonline.blogspot.com",
    "bbs.bt",
    "www.bbs.bt",
    "dzongkha.gov.bt",
    "www.dzongkha.gov.bt",
    "nab.gov.bt",
    "www.nab.gov.bt",
    "molhr.gov.bt",
    "gnhc.gov.bt",
}

# Tibetan and non-Dzongkha sources observed in the split. Excluded explicitly so
# that a --any-domain run still drops the known-bad ones.
BLOCKED_DOMAINS = {
    "voatibetan.com", "www.voatibetan.com", "voanews.com", "www.voanews.com",
    "rfa.org", "www.rfa.org", "tibettimes.net", "www.tibettimes.net",
    "ti.tibet3.com", "tibet3.com", "ti.zangdiyg.com", "zangdiyg.com",
    "baike.yongzin.com", "yongzin.com", "rgbm123.com", "qhtibetan.com",
    "ti.kbcmw.com", "kbcmw.com", "help.libreoffice.org",
    "audio-video.shanti.virginia.edu", "shanti.virginia.edu",
}

TSHEG = "་"  # ་
SHAD = "།"  # །
TIBETAN_BLOCK = re.compile(r"[ༀ-࿿]")

# Head marks (yig mgo) open a Tibetan paragraph and carry no sentence content;
# the pipes are leftover markup from the crawl. Both appear at the start of a
# meaningful share of extracted sentences (7.3% and 1.1% respectively).
LEADING_JUNK = re.compile(r"^[\s|༉༄༅༆༇࿐࿑\-–—•·:;,\.]+")
TRAILING_JUNK = re.compile(r"[\s|]+$")


def host_of(url):
    if not url or "://" not in url:
        return ""
    return url.split("/")[2].lower()


def tibetan_ratio(text):
    """Share of non-space characters that sit in the Tibetan Unicode block."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if TIBETAN_BLOCK.match(c)) / len(chars)


def clean(sent):
    """Strip crawl markup and Tibetan head marks from a sentence's edges."""
    sent = sent.replace("|", " ")
    sent = LEADING_JUNK.sub("", sent)
    sent = TRAILING_JUNK.sub("", sent)
    return re.sub(r"\s+", " ", sent).strip()


def split_sentences(text):
    """Split Dzongkha on the shad, which terminates a sentence or clause."""
    text = unicodedata.normalize("NFC", text)
    parts = re.split(r"[།༎]+", text)
    out = []
    for p in parts:
        p = clean(p)
        if p:
            out.append(p + SHAD)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/data/dz_mono.txt")
    ap.add_argument("--min-lang-score", type=float, default=0.85)
    ap.add_argument("--min-tibetan-ratio", type=float, default=0.80,
                    help="Reject text that is mostly not Tibetan script.")
    ap.add_argument("--min-syllables", type=int, default=4,
                    help="Minimum tsheg count per sentence.")
    ap.add_argument("--max-syllables", type=int, default=70,
                    help="Upper bound on tsheg count. News prose runs long -- the "
                         "median extracted sentence carries ~52 syllables, so "
                         "raise train.py's --max-length when training on this.")
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--any-domain", action="store_true",
                    help="Keep every domain except the blocklist. Raises volume "
                         "and lowers purity -- sample the output before training.")
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceFW/finetranslations", "dzo_Tibt", split="train")
    print(f"Loaded {len(ds)} documents")

    stats = {"domain": 0, "lang_score": 0, "script": 0, "kept_docs": 0}
    sentences = []
    seen = set()

    for row in ds:
        host = host_of(row.get("url"))
        if host in BLOCKED_DOMAINS:
            stats["domain"] += 1
            continue
        if not args.any_domain and host not in DZONGKHA_DOMAINS:
            stats["domain"] += 1
            continue
        if (row.get("og_language_score") or 0) < args.min_lang_score:
            stats["lang_score"] += 1
            continue

        text = row.get("og_full_text") or ""
        if tibetan_ratio(text) < args.min_tibetan_ratio:
            stats["script"] += 1
            continue

        stats["kept_docs"] += 1
        for sent in split_sentences(text):
            if len(sent) > args.max_chars:
                continue
            syllables = sent.count(TSHEG)
            if syllables < args.min_syllables or syllables > args.max_syllables:
                continue
            if sent in seen:
                continue
            seen.add(sent)
            sentences.append(sent)

    print(f"Rejected -- domain: {stats['domain']}, lang score: {stats['lang_score']}, "
          f"script: {stats['script']}")
    print(f"Kept {stats['kept_docs']} documents -> {len(sentences)} unique sentences")

    # The headline number for our purposes: the parallel corpus contains no
    # numerals at all, so how many of these carry one decides how much of the
    # numeral gap back-translation can close on its own.
    with_digits = sum(1 for s in sentences if re.search(r"[༠-༩0-9]", s))
    syl = sorted(s.count(TSHEG) for s in sentences)
    print(f"  carrying a numeral: {with_digits} ({100 * with_digits / max(len(sentences), 1):.1f}%)")
    if syl:
        print(f"  syllables per sentence: median {syl[len(syl) // 2]}, "
              f"p95 {syl[int(0.95 * len(syl))]}")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sentences) + "\n")
    print(f"Wrote {args.out}")

    print("\nSample:")
    for s in sentences[:5]:
        print("  " + s[:110])
    print("\nAttribution: HuggingFaceFW/finetranslations (ODC-BY). Source articles "
          "remain the property of their publishers -- confirm reuse terms before "
          "shipping a commercial product trained on them.")


if __name__ == "__main__":
    main()
