"""Restore orthography to the Dzongkha-English corpus.

The corpus at kinleyrabgay/dz_to_en has been word-segmented and depunctuated on
both sides: 0% of the 232,489 Dzongkha sentences contain a shad, and 0% of the
English sentences contain a capital letter or a sentence-final period. A model
trained on it as-is emits spaced, shad-less Dzongkha and lowercase unpunctuated
English -- unusable output regardless of BLEU.

Each side lost something different, and each is recoverable:

  Dzongkha: words are space-separated and 99.73% of non-final tokens end in a
    tsheg. The 0.27% that do NOT end in a tsheg are exactly the clause
    boundaries, i.e. the positions a shad was deleted from. So: join on tsheg,
    insert a shad wherever a tsheg is absent, terminate with a shad.

  English: a double space marks a deleted comma. Casing follows from
    sentence position plus a gazetteer. Sentence-final punctuation is chosen
    using the Dzongkha question particles, which predict an interrogative with
    ~91% precision at 5.9% leakage.

Everything here is heuristic restoration of information that was destroyed
upstream. It is not a substitute for a correctly-punctuated source corpus, and
the Dzongkha output should be spot-checked by a native reader before you trust
it -- see SHAD_AFTER_GA below for the one rule that is a real editorial choice.
"""

import os
import re

TSHEG = "་"  # ་ syllable delimiter
SHAD = "།"  # ། sentence/clause terminator
NGA = "ང"  # ང

# Dzongkha sentence-final question particles, with measured precision at
# predicting an interrogative English reference:
#   སྨོ 92.9%   ག 91.6%   ཨིན་ན 91.0%   ན 89.9%
# གོ measured 24.0% and is deliberately excluded.
QUESTION_PARTICLES = {"ག", "སྨོ", "ན", "ཨིན་ན"}

# Traditional Tibetan orthography omits the shad after a syllable ending in ཀ or
# ག, the descending stroke standing in for it. Modern Dzongkha keeps it, and that
# is what this project uses -- confirmed by the project owner, 2026-09-01. So
# ནུག, འདུག and the question particle ག all take a shad (འདུག།, དོ་ག།), which
# affects ~16% of the corpus since those are among the most common final
# syllables.
#
# This is baked into every dataset built by prepare_data.py. Changing it means
# rebuilding the data and redoing the baseline evaluation.
SHAD_AFTER_GA = True
GA_KA = {"ག", "ཀ"}  # ག ཀ

# Kuensel spaces at phrase boundaries (0.212 spaces per syllable), while this
# corpus spaces at word boundaries (0.750). Deleting every space matches neither,
# so we keep the subset that falls after a case marker or postposition -- the set
# respace.py learned from real Kuensel text, where it recovers 91.2% of genuine
# boundaries on held-out data. Absent the file we fall back to joining
# everything, which is the previous behaviour.
_MARKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "boundary_markers.txt")
try:
    with open(_MARKER_FILE, encoding="utf-8") as _fh:
        BOUNDARY_MARKERS = {ln.strip() for ln in _fh if ln.strip()}
except OSError:
    BOUNDARY_MARKERS = set()


def _tail_syllable(chunk):
    syls = [s for s in chunk.split(TSHEG) if s]
    return syls[-1] if syls else ""

# Proper nouns are unrecoverable in general -- lowercasing is lossy and no rule
# brings them back. This gazetteer covers the high-frequency cases only; expect
# residual lowercase proper nouns in the output and extend the list as you find
# them in your own data.
GAZETTEER = {
    "i": "I",
    "bhutan": "Bhutan", "bhutanese": "Bhutanese", "thimphu": "Thimphu",
    "paro": "Paro", "punakha": "Punakha", "wangdue": "Wangdue",
    "trongsa": "Trongsa", "bumthang": "Bumthang", "trashigang": "Trashigang",
    "samdrup": "Samdrup", "jongkhar": "Jongkhar", "gelephu": "Gelephu",
    "phuentsholing": "Phuentsholing", "haa": "Haa", "dagana": "Dagana",
    "tsirang": "Tsirang", "zhemgang": "Zhemgang", "lhuentse": "Lhuentse",
    "mongar": "Mongar", "pemagatshel": "Pemagatshel", "samtse": "Samtse",
    "chukha": "Chukha", "gasa": "Gasa", "druk": "Druk", "dzongkha": "Dzongkha",
    "india": "India", "indian": "Indian", "nepal": "Nepal", "nepali": "Nepali",
    "china": "China", "chinese": "Chinese", "tibet": "Tibet", "tibetan": "Tibetan",
    "japan": "Japan", "japanese": "Japanese", "america": "America",
    "american": "American", "england": "England", "english": "English",
    "britain": "Britain", "british": "British", "france": "France",
    "french": "French", "germany": "Germany", "german": "German",
    "korea": "Korea", "korean": "Korean", "thailand": "Thailand",
    "bangladesh": "Bangladesh", "buddha": "Buddha", "buddhism": "Buddhism",
    "buddhist": "Buddhist", "monday": "Monday", "tuesday": "Tuesday",
    "wednesday": "Wednesday", "thursday": "Thursday", "friday": "Friday",
    "saturday": "Saturday", "sunday": "Sunday", "january": "January",
    "february": "February", "march": "March", "april": "April", "may": "May",
    "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November",
    "december": "December",
}


def dz_normalize(text, shad_after_ga=SHAD_AFTER_GA, markers=None):
    """Turn space-segmented, shad-less Dzongkha into conventional orthography.

    'ང་ ཡིག་ཚང་ ནང་ མེད' -> 'ང་ཡིག་ཚང་ནང་ མེད།'
    'ཉོ་ ཆོག དེ་འབདཝ་ད་ ... མེན' -> 'ཉོ་ཆོག། དེ་འབདཝ་ད་...མེན།'
    """
    if markers is None:
        markers = BOUNDARY_MARKERS
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return ""

    out = []
    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1
        if tok.endswith(TSHEG):
            out.append(tok)
            # Keep this word break only where Kuensel would: after a case
            # marker or postposition.
            if not last and _tail_syllable(tok) in markers:
                out.append(" ")
            continue

        # No tsheg => a shad was deleted here. Decide whether to write it back.
        base = tok
        if base.endswith(NGA):
            # The one case where the tsheg is kept before the shad.
            out.append(base + TSHEG + SHAD)
        elif base and base[-1] in GA_KA and not shad_after_ga:
            out.append(base)
        else:
            out.append(base + SHAD)
        if not last:
            out.append(" ")

    result = "".join(out)
    # A tsheg-final last token still needs its terminator.
    if not result.endswith(SHAD):
        if result.endswith(TSHEG):
            result = result[:-1]
        if result.endswith(NGA):
            result += TSHEG
        result += SHAD
    return re.sub(r"\s+", " ", result).strip()


def en_normalize(text, is_question=False):
    """Restore commas, casing and terminal punctuation to the English side.

    'to be honest  i really do not know' -> 'To be honest, I really do not know.'
    """
    s = text.strip()
    if not s:
        return ""

    # A double space is a deleted comma.
    s = re.sub(r"\s{2,}", ", ", s)

    words = s.split(" ")
    words = [GAZETTEER.get(w, w) for w in words]
    s = " ".join(words)

    # Sentence-initial capital, including after a restored terminator.
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)

    if not s.endswith((".", "!", "?")):
        s += "?" if is_question else "."
    return s


def is_dz_question(dz_text):
    """True if the raw (pre-normalization) Dzongkha ends in a question particle."""
    tokens = [t for t in dz_text.strip().split() if t]
    return bool(tokens) and tokens[-1] in QUESTION_PARTICLES


def normalize_pair(dz_raw, en_raw, shad_after_ga=SHAD_AFTER_GA, markers=None):
    """Normalize one aligned pair. Question detection reads the Dzongkha side."""
    q = is_dz_question(dz_raw)
    return (dz_normalize(dz_raw, shad_after_ga, markers),
            en_normalize(en_raw, is_question=q))


if __name__ == "__main__":
    import csv
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    csv.field_size_limit(10**9)
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader)
        for i, (dz, en) in enumerate(reader):
            if i >= 15:
                break
            ndz, nen = normalize_pair(dz, en)
            print(f"dz  in : {dz.strip()}")
            print(f"dz  out: {ndz}")
            print(f"en  in : {en.strip()}")
            print(f"en  out: {nen}")
            print()
