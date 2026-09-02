"""Extract the DDC Dzongkha-English Pocket Dictionary (2nd ed., 2013) into
machine-readable tables.

Unlike build_term_tables.py, nothing here is inferred from corpus statistics.
Every pair below is what a lexicographer published, so it is gold data: usable
as a term table, as a translation memory, and as a source of the grammatical
facts the corpus cannot supply (verb tense stems, honorific register, plain
vs. honorific alternation, short forms, antonyms).

The PDF has a real text layer, two columns per page, and a set of private-use
glyphs that mark the relation between a headword and the Dzongkha form that
follows it. Page x of the book documents them; the mapping is MARKERS below.

Sections of the book:
    pp.  13-1107   main A-Z body, Dzongkha headword -> English glosses
    p.  1110       counting numbers
    pp. 1114-1117  Dzongkha personal names
    pp. 1118-1119  Indic names, English -> Dzongkha
    pp. 1120-1123  Bhutanese place names
    pp. 1124-1135  countries and capitals, English -> Dzongkha
    pp. 1140-1179  verb form tables (future / present / past / imperative)

    python extract_dictionary.py --pdf PP-XII-Dzongkha-English-Dictionary.pdf --out dict/
"""

import argparse
import collections
import json
import os
import re

import pymupdf

import pdf_glyphs

# --- page ranges, 0-based PDF indices (the printed page number is 12 lower) ---
BODY = (13, 1108)
NUMBERS = (1110, 1111)
NAMES_DZ = (1114, 1118)
NAMES_INDIC = (1118, 1120)
PLACES_BT = (1120, 1124)
COUNTRIES = (1124, 1136)
VERB_TABLES = (1140, 1180)

# --- layout ---
COLUMN_SPLIT = 290.0   # x of the gutter
HEADER_Y = 62.0        # running head above this
FOOTER_Y = 780.0       # page number below this

TIB = "\u0f00-\u0fff"
TIB_RUN = "[" + TIB + r"\s]"

# Private-use glyphs the typesetter used as relation markers, per the
# "Guide to dictionary entries" on printed pages x-xi.
MARKERS = {
    "\uf044": "synonym",        # མིང་གི་རྣམ་གྲངས།
    "\uf041": "opposite",       # འགལ་མིང།
    "\uf062": "honorific",      # ཞེ་ས།  -- the honorific of this headword
    "\uf068": "variant",        # ཚིག་གི་འབྲི་ལུགས་གཞན། other written form
    "\uf069": "short_form",     # ཚིག་གི་མིང་སྡུད།
    "\uf063": "is_honorific",   # this headword is itself an honorific form
}
MARKER_CLASS = "".join(MARKERS)

POS = r"(?:noun|verb|adj\.|adv\.|num\.|conj\.|excl\.|prep\.|pron\.)"
POS_RE = re.compile(POS + r"(?![A-Za-z])")
TENSE_RE = re.compile(r"\b(fut|prs|pst|imp)\.\s*(" + TIB_RUN + r"+?)(?=\s*(?:fut\.|prs\.|pst\.|imp\.|[A-Za-z0-9]|$))")
SENSE_RE = re.compile(r"(?:^|\s)([1-9])\s+")
TENSE_KEY = {"fut": "future", "prs": "present", "pst": "past", "imp": "imperative"}


def page_lines(page, repairs):
    """Blocks of one page in reading order: left column top-to-bottom, then right.

    The text comes back through pdf_glyphs, which puts back the characters the
    PDF's ToUnicode table drops -- without that step the Dzongkha is quietly
    wrong rather than obviously broken.
    """
    blocks = []
    for x0, y0, text in pdf_glyphs.repaired_blocks(page, repairs):
        if y0 < HEADER_Y or y0 > FOOTER_Y:
            continue
        if len(text.strip()) <= 2:
            continue          # single-letter thumb tab in the outer margin
        blocks.append((0 if x0 < COLUMN_SPLIT else 1, y0, text))
    blocks.sort(key=lambda b: (b[0], b[1]))
    return blocks


def page_text(page, repairs):
    flat = " ".join(b[2].replace("\n", " ") for b in page_lines(page, repairs))
    flat = re.sub(r"\s+", " ", flat)
    # A line break inside a Dzongkha word leaves a space after the tsheg, and a
    # break inside an English compound leaves one after the hyphen. Both are
    # artefacts of justification, not of the text.
    flat = flat.replace("་ ", "་").replace(" ་", "་")
    flat = re.sub(r"(?<=[A-Za-z])- (?=[a-z])", "-", flat)
    return flat.strip()


def clean_dz(s):
    s = re.sub(r"\s+", " ", s.strip())
    # the same justification artefact page_text repairs, on the appendix path
    return s.replace("\u0f0b ", "\u0f0b").replace(" \u0f0b", "\u0f0b").strip()


def split_dz_forms(s):
    """A marker introduces one or more Dzongkha forms, separated by spaces."""
    return [f for f in (clean_dz(p) for p in s.split(" ")) if f]


def split_glosses(s):
    """Split an English definition on commas that are not inside brackets."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [g.strip(" .;") for g in out if g.strip(" .;")]


def parse_definition(text):
    """The part of an entry after the part-of-speech tag.

    Returns (relations, senses). A sense is {"n", "english", "note"} where note
    is the Dzongkha disambiguation the book prints in brackets.
    """
    relations = collections.defaultdict(list)
    text = text.strip()

    # Leading marker + Dzongkha form groups, e.g.  <syn> ཀ་སད། ཀ་མད།  <hon> ཞབས།
    while text and text[0] in MARKER_CLASS:
        kind = MARKERS[text[0]]
        text = text[1:].lstrip(". ")
        if kind == "is_honorific":
            relations["is_honorific"] = [True]
            continue
        m = re.match(TIB_RUN + r"+", text)
        if not m:
            continue
        forms, rest = m.group(), text[m.end():]
        # a trailing marker belongs to the next group, not to this one
        forms = forms.rstrip()
        relations[kind].extend(split_dz_forms(forms))
        text = rest.lstrip()

    parts = SENSE_RE.split(" " + text)
    senses = []
    if len(parts) == 1:
        chunks = [(None, parts[0])]
    else:
        chunks = [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    for num, chunk in chunks:
        chunk = chunk.strip()
        note = ""
        m = re.search(r"\((" + TIB_RUN + r"+)\)\s*$", chunk)
        if m:
            note = clean_dz(m.group(1))
            chunk = chunk[: m.start()].strip()
        # any remaining Dzongkha inside the gloss is a bracketed aside
        english = [g for g in split_glosses(chunk) if re.search(r"[A-Za-z]", g)]
        if english:
            senses.append({"n": int(num) if num else 1, "english": english, "note": note})
    return dict(relations), senses


def parse_body(doc, repairs):
    entries = []
    for pno in range(*BODY):
        text = page_text(doc[pno], repairs)
        hits = list(POS_RE.finditer(text))
        spans = []
        for m in hits:
            start = m.start()
            # walk back over the Dzongkha headword to the end of the previous entry
            k = start
            while k > 0 and re.match("[" + TIB + r"\s]", text[k - 1]):
                k -= 1
            run = text[k:start]
            # The run can also hold the tail of the previous entry (its Dzongkha
            # synonyms). A headword is a single unbroken chunk, so keep the last.
            head_off = run.rstrip().rfind(" ")
            if head_off != -1:
                k += head_off + 1
            spans.append((k, start, m.end(), m.group()))
        for i, (k, hstart, hend, pos) in enumerate(spans):
            head = clean_dz(text[k:hstart])
            head = re.sub(r"^\u0f3c[^\u0f3d]*\u0f3d\s*", "", head)
            if not head:
                continue
            end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
            rest = text[hend:end]
            tenses = {}
            if pos == "verb":
                m = re.match(r"\s*\(([^)]*)\)", rest)
                if m:
                    for t in re.findall(r"(fut|prs|pst|imp)\.", m.group(1)):
                        tenses[TENSE_KEY[t]] = head
                    rest = rest[m.end():]
                # explicit stems for the tenses whose spelling differs
                while True:
                    m = TENSE_RE.match(rest.lstrip())
                    if not m:
                        break
                    rest = rest.lstrip()
                    tenses[TENSE_KEY[m.group(1)]] = clean_dz(m.group(2))
                    rest = rest[m.end():]
            relations, senses = parse_definition(rest)
            entries.append({
                "page": pno - 12,
                "headword": head,
                "pos": pos.rstrip("."),
                "tenses": tenses,
                "relations": relations,
                "senses": senses,
            })
    return entries


def parse_verb_tables(doc, repairs):
    """The appendix prints four columns: future, present, past, imperative."""
    rows = []
    for pno in range(*VERB_TABLES):
        for _col, _y, text in page_lines(doc[pno], repairs):
            if "Future" in text or "Imperative" in text:
                continue
            cells = [clean_dz(c) for c in text.split("\n")]
            cells = [c for c in cells if c]
            if len(cells) < 2 or not re.match("[" + TIB + "]", cells[0]):
                continue
            cells = (cells + ["", "", "", ""])[:4]
            rows.append(cells)
    return rows


def parse_pipe_pages(doc, span, repairs, en_first=False):
    """Appendix pages that print  English | Dzongkha  one pair per line."""
    pairs = []
    for pno in range(*span):
        for _col, _y, text in page_lines(doc[pno], repairs):
            for line in text.split("\n"):
                if "|" not in line:
                    continue
                left, right = line.split("|", 1)
                left, right = clean_dz(left), clean_dz(right)
                if not left or not right:
                    continue
                if en_first and re.search("[" + TIB + "]", left):
                    continue      # the page's own heading, printed Dzongkha first
                pairs.append((left, right))
    return pairs


def parse_countries(doc, repairs):
    """`Bulgaria | བྷཱལ་གེ་རི་ཡ། > Sofia | སོ་ཕི་ཡ།`"""
    out = []
    for pno in range(*COUNTRIES):
        for _col, _y, text in page_lines(doc[pno], repairs):
            for line in text.split("\n"):
                if line.count("|") != 2 or ">" not in line:
                    continue
                country, rest = line.split("|", 1)
                dz_country, capital = rest.split(">", 1)
                capital, dz_capital = capital.split("|", 1)
                out.append((country.strip(), clean_dz(dz_country),
                            capital.strip(), clean_dz(dz_capital)))
    return out


def parse_dz_list(doc, span, repairs):
    out = []
    for pno in range(*span):
        for _col, _y, text in page_lines(doc[pno], repairs):
            for line in text.split("\n"):
                line = clean_dz(line)
                if line and re.fullmatch("[" + TIB + r"\s]+", line):
                    out.append(line)
    return out


def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(c).replace("\t", " ") for c in r) + "\n")
    print(f"  {os.path.basename(path):24s} {len(rows):6d} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="PP-XII-Dzongkha-English-Dictionary.pdf")
    ap.add_argument("--out", default="dict")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    doc = pymupdf.open(args.pdf)
    out = lambda name: os.path.join(args.out, name)

    repairs = pdf_glyphs.load_repairs(doc)
    pdf_glyphs.write_report(repairs, out("glyph_repairs.tsv"))
    print(f"  {'glyph_repairs.tsv':24s} {len(repairs):6d} glyphs the PDF mis-maps")
    entries = parse_body(doc, repairs)
    with open(out("entries.jsonl"), "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  {'entries.jsonl':24s} {len(entries):6d} entries")

    # dz -> en, one row per gloss, keeping the sense note so a reviewer can see
    # which meaning a gloss belongs to
    dz_en, en_dz = [], collections.defaultdict(list)
    for e in entries:
        hon = "yes" if e["relations"].get("is_honorific") else ""
        for s in e["senses"]:
            for g in s["english"]:
                dz_en.append((e["headword"], g, e["pos"], s["n"], s["note"], hon, e["page"]))
                en_dz[g.lower()].append(e["headword"])
    write_tsv(out("dz_en.tsv"),
              ["dz", "en", "pos", "sense", "dz_note", "honorific", "page"], dz_en)

    write_tsv(out("en_dz.tsv"), ["en", "n_dz", "dz_forms"],
              [(en, len(set(dz)), " | ".join(dict.fromkeys(dz)))
               for en, dz in sorted(en_dz.items())])

    # relation tables: the grammar and register facts
    for kind, fname, cols in [
        ("honorific", "honorific.tsv", ["plain", "honorific"]),
        ("synonym", "synonyms.tsv", ["dz", "synonym"]),
        ("opposite", "antonyms.tsv", ["dz", "opposite"]),
        ("variant", "variants.tsv", ["dz", "other_spelling"]),
        ("short_form", "short_forms.tsv", ["dz", "short_form"]),
    ]:
        rows = [(e["headword"], f) for e in entries for f in e["relations"].get(kind, [])]
        write_tsv(out(fname), cols, rows)

    inline = [(e["headword"], e["tenses"].get("future", ""), e["tenses"].get("present", ""),
               e["tenses"].get("past", ""), e["tenses"].get("imperative", ""), "entry")
              for e in entries if e["tenses"]]
    table = [(r[0], r[0], r[1], r[2], r[3], "appendix") for r in parse_verb_tables(doc, repairs)]
    write_tsv(out("verb_forms.tsv"),
              ["lemma", "future", "present", "past", "imperative", "source"],
              inline + table)

    write_tsv(out("countries.tsv"), ["en_country", "dz_country", "en_capital", "dz_capital"],
              [c for c in parse_countries(doc, repairs) if c[0] != "Country"])
    write_tsv(out("numbers.tsv"), ["dz", "en"],
              [(dz, en) for dz, en in parse_pipe_pages(doc, NUMBERS, repairs)
               if not re.search(r"Counting numbers|measures", en)])
    write_tsv(out("names_indic.tsv"), ["en", "dz"], parse_pipe_pages(doc, NAMES_INDIC, repairs, en_first=True))
    write_tsv(out("names_dz.tsv"), ["dz"], [(n,) for n in parse_dz_list(doc, NAMES_DZ, repairs)])
    write_tsv(out("places_bt.tsv"), ["dz"], [(p,) for p in parse_dz_list(doc, PLACES_BT, repairs)])


if __name__ == "__main__":
    main()
