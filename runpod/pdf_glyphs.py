"""Repair the Dzongkha the dictionary PDF hands out.

The book is typeset in DDC Uchen, which draws a stacked Dzongkha syllable as a
single ligature glyph. For most of those glyphs the PDF's ToUnicode table maps
the ligature back to every character it stands for. For 27 of them it does not:
it names only the first component and silently drops the rest, so the extracted
text reads སོབ where the page shows སློབ, བ་བ where it shows བྱ་བ, and སོད for
three different words (སྤྱོད, སྤྲོད, སྡོད). Nothing in the extracted string marks
these as damaged -- they are ordinary, wrong Dzongkha.

The font carries what is needed to undo this. Its GSUB ligature table says which
glyphs each ligature was built from, and the ligatures that *are* mapped
correctly reveal what each component glyph stands for. Read the components off
the good ligatures, then apply them to the truncated ones:

    glyph 1347 = ligature(uni0F66, glyph00462)   from GSUB
    glyph 1347 emits "ས"                          from ToUnicode -- one short
    glyph00462 = ླ                                from other ligatures that map it
    => glyph 1347 is སླ

Because the repair is keyed on the glyph, not on the text, it fixes the
ambiguous cases too: the three words that all extract as སོད use three different
glyphs and come back as three different words.

The map is written to dict/glyph_repairs.tsv so it can be checked by hand.
"""

import collections
import io
import json
import os
import re

import pymupdf
from fontTools.ttLib import TTFont


def _ligatures(doc, xref):
    """glyph id -> the glyph ids it was composed from, per the font's GSUB."""
    try:
        font = TTFont(io.BytesIO(doc.extract_font(xref)[3]), lazy=False)
    except Exception:
        return {}
    if "GSUB" not in font:
        return {}
    gid = {name: i for i, name in enumerate(font.getGlyphOrder())}
    out = {}
    for lookup in font["GSUB"].table.LookupList.Lookup:
        for sub in lookup.SubTable:
            if getattr(sub, "LookupType", lookup.LookupType) != 4:
                continue
            for first, ligset in getattr(sub, "ligatures", {}).items():
                for lig in ligset:
                    out.setdefault(gid[lig.LigGlyph],
                                   [gid[first]] + [gid[c] for c in lig.Component])
    return out


def _emitted(doc):
    """font -> glyph id -> the text ToUnicode produces for it.

    PyMuPDF reports a glyph's first character with its glyph id and the rest of
    the characters that glyph maps to with an id of -1, so a run of -1s belongs
    to the glyph before it.
    """
    seen = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for page in doc:
        for span in page.get_texttrace():
            font, cur = span["font"], None
            for ch in span["chars"]:
                if ch[1] >= 0:
                    if cur:
                        seen[font][cur[0]][cur[1]] += 1
                    cur = [ch[1], chr(ch[0])]
                elif cur:
                    cur[1] += chr(ch[0])
            if cur:
                seen[font][cur[0]][cur[1]] += 1
    return seen


def build_repairs(doc):
    """(font name, glyph id) -> the text the glyph should have produced."""
    by_name = {}
    for xref in range(1, doc.xref_length()):
        try:
            info = doc.extract_font(xref)
        except Exception:
            continue
        if info and info[0] and info[3]:
            by_name.setdefault(info[0].split("+")[-1], xref)

    repairs = {}
    for font, glyphs in _emitted(doc).items():
        ligs = _ligatures(doc, by_name[font]) if font in by_name else {}
        if not ligs:
            continue
        text = {g: c.most_common(1)[0][0] for g, c in glyphs.items()}

        # A ligature whose ToUnicode is complete tells us what each of its
        # components stands for.
        parts = collections.defaultdict(collections.Counter)
        for g, s in text.items():
            if g in ligs and len(s) == len(ligs[g]):
                for component, ch in zip(ligs[g], s):
                    parts[component][ch] += 1
        component = {g: c.most_common(1)[0][0] for g, c in parts.items()}

        def resolve(g, depth=0):
            if depth > 6:
                return None
            if g in component:
                return component[g]
            if g in ligs:
                got = [resolve(c, depth + 1) for c in ligs[g]]
                if all(got):
                    return "".join(got)
            if g in text and g not in ligs:
                return text[g]
            return None

        for g, s in text.items():
            if g not in ligs or len(s) == len(ligs[g]):
                continue
            full = resolve(g)
            if full and len(full) > len(s):
                repairs[(font, g)] = full
    return repairs


def load_repairs(doc, cache="dict/glyph_repairs.json"):
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return {(f, int(g)): s for f, g, s in json.load(fh)}
    repairs = build_repairs(doc)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump([[f, g, s] for (f, g), s in sorted(repairs.items())], fh,
                  ensure_ascii=False, indent=1)
    return repairs


def repaired_blocks(page, repairs):
    """Text blocks in the shape get_text("blocks") returns them, but with the
    dropped characters put back: (x0, y0, text) with lines separated by \\n."""
    fix = {}
    for span in page.get_texttrace():
        for ch in span["chars"]:
            full = repairs.get((span["font"], ch[1])) if ch[1] >= 0 else None
            if full:
                fix[(round(ch[2][0], 1), round(ch[2][1], 1), ch[0])] = full

    out = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type", 0) != 0:
            continue
        text = []
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    key = (round(ch["origin"][0], 1), round(ch["origin"][1], 1), ord(ch["c"]))
                    text.append(fix.get(key, ch["c"]))
            text.append("\n")
        out.append((block["bbox"][0], block["bbox"][1], "".join(text)))
    return out


def write_report(repairs, path="dict/glyph_repairs.tsv"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("font\tglyph\textracted_as\tshould_be\n")
        for (font, gid), full in sorted(repairs.items()):
            fh.write(f"{font}\t{gid}\t{full[0]}\t{full}\n")
    return len(repairs)
