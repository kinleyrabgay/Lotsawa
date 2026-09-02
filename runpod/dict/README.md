# DDC Dzongkha–English Pocket Dictionary, extracted

Source: *Dzongkha–English Pocket Dictionary*, 2nd edition, Dzongkha Development
Commission, Thimphu, 2013 (ISBN 978-99936-15-21-7), 1180 pages with a real text
layer. Regenerate everything here with:

    python extract_dictionary.py --pdf PP-XII-Dzongkha-English-Dictionary.pdf --out dict/

This is gold data, not corpus statistics. `build_term_tables.py` guesses a
Dzongkha form from how often it co-occurs with an English word; these tables are
what a lexicographer published. The two are complementary: the corpus tells you
which form is *used*, the dictionary tells you which form is *right*.

## Tables

| file | rows | contents |
|---|---|---|
| `glyph_repairs.tsv` | 27 | glyphs whose ToUnicode the PDF gets wrong, and what they should be |
| `entries.jsonl` | 35,549 | one JSON object per headword: part of speech, tense stems, relations, numbered senses |
| `dz_en.tsv` | 66,321 | one row per gloss: `dz, en, pos, sense, dz_note, honorific, page` |
| `en_dz.tsv` | 26,143 | the inverse index: every Dzongkha form the book gives for an English word |
| `verb_forms.tsv` | 4,511 | future / present / past / imperative, from the entries and from the appendix tables |
| `honorific.tsv` | 507 | plain form → its honorific (ཞེ་ས།) |
| `synonyms.tsv` | 854 | headword → synonym (མིང་གི་རྣམ་གྲངས།) |
| `antonyms.tsv` | 394 | headword → opposite (འགལ་མིང།) |
| `variants.tsv` | 21 | headword → other accepted spelling |
| `short_forms.tsv` | 36 | headword → contracted form (མིང་སྡུད།) |
| `countries.tsv` | 184 | country and capital, English ↔ Dzongkha |
| `numbers.tsv` | 15 | decimal counting words |
| `names_indic.tsv` | 67 | Indic personal names, English → Dzongkha |
| `names_dz.tsv` | 230 | Dzongkha personal names |
| `places_bt.tsv` | 210 | Bhutanese place names |

## The PDF drops characters, and the extractor puts them back

DDC Uchen draws a stacked Dzongkha syllable as one ligature glyph. For 27 of
those glyphs the PDF's ToUnicode table names only the first character the glyph
stands for and silently drops the rest, so a naive extraction reads སོབ where
the page shows སློབ, བ་བ for བྱ་བ, and the same སོད for three different words
(སྤྱོད, སྤྲོད, སྡོད). Nothing marks the damage: the output is ordinary, wrong
Dzongkha, and it hits common vocabulary hardest.

`pdf_glyphs.py` repairs it from the font itself. The GSUB table says which
glyphs each ligature was built from, and the ligatures the PDF *does* map
correctly reveal what each component glyph stands for; apply those components to
the truncated ligatures and the dropped characters come back. Because the fix is
keyed on the glyph rather than on the text, the three words that all extract as
སོད come back as three different words. `glyph_repairs.tsv` lists every
substitution so it can be checked against the page.

The effect: headwords carrying a visible extraction artefact fall from 6.7% to
4 of 35,549, and the corpus and the dictionary independently agree on 262 term
rows instead of 226.

## How the book encodes grammar

The typesetter used private-use glyphs to mark what the Dzongkha form after a
headword *is*. Printed pages x–xi document them; the parser maps them to the
relation names above. Two of them are register, not meaning: `honorific` gives
the ཞེ་ས། form of a plain word (རྐང་པ། → ཞབས།), and a headword may itself be
flagged honorific, which lands in the `honorific` column of `dz_en.tsv`.

Verbs carry their tense paradigm inline. `verb (fut., prs., pst., imp.)` means
one spelling serves all four; `verb (fut., pst.) prs. སྐྲུན། imp. སྐྲུན།` means
the listed tenses differ. Where the book prints a bare stem for one tense
(`བཏང་།` for a compound whose lemma is `ཀྲིག་ཀྲི་གཏང་བ།`), the stem is stored as
printed — it is the changing element, not the whole verb.

A sense's `dz_note` is the Dzongkha disambiguation the book prints in brackets,
e.g. ཀེར་ཐིག sense 1 *horizontal line* (ཕར་ཚུར།) vs. sense 2 *vertical line*
(ཡར་མར།).

## Caveats

- English glosses are 2013 lexicography, occasionally idiosyncratic. The
  weekday entries follow the Bhutanese convention (འབྲུག་གཟའ་མིག་དམར། is glossed
  *Monday*), which is the book's, not an extraction error.
- An English word with several Dzongkha rows in `en_dz.tsv` is usually English
  polysemy, not Dzongkha variation — *cow* returns both the animal and the verb
  *to cow*. Filter on `pos` before using it as a term table.
- The abbreviation/ligature appendix (printed pp. 1124–1127) is not extracted:
  the contracted forms are stacked ligatures that do not survive as text.
- The PDF itself is not committed; keep it beside the script or pass `--pdf`.

## Filling the term tables

`fill_terms_from_dict.py` writes these findings back into `runpod/terms/*.csv`:

    python extract_dictionary.py
    python fill_terms_from_dict.py --dry-run   # see what it would change
    python fill_terms_from_dict.py

Of 686 term rows: **256 confirmed** (the dictionary independently gives the form
the corpus scan already found), **218 filled** (rows that were MISSING or low
confidence), **59 disagreements** left for a reviewer with the corpus form kept,
and **153 not in the book** — almost all of them Bhutanese proper nouns, which
the A-Z body does not carry. Rows with a Dzongkha form go from 479 to 616.

Filled rows get `confidence=dict`, which `augment_terms.py` ranks above `high`:
a lexicographer's form outranks one inferred from co-occurrence. Rows where the
corpus already had medium or high confidence are never overwritten — that form
is attested in real sentences, which is what carrier substitution needs — but
the row is flagged with the dictionary's alternative. Headwords are rewritten
from citation form (སློབ་དཔོན།) to the sentence-internal form (སློབ་དཔོན་) the
carriers expect.
