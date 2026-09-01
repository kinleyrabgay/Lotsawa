"""Load FLORES-200 Dzongkha-English from whichever mirror is reachable.

FLORES-200 devtest is 1,012 human-translated sentences and the standard benchmark
for this pair. Both `facebook/flores` and `openlanguagedata/flores_plus` are GATED
on the Hub -- they need a one-click access request -- so try each source in turn
and use the first that loads.

Never train on any of these. dev is for checkpoint selection, devtest for scoring.
"""

from datasets import load_dataset

DZ = "dzo_Tibt"
EN = "eng_Latn"

#   per_language : one config per language, joined on `id` (flores_plus shape)
#   paired       : a single `dzo_Tibt-eng_Latn` config with sentence_* columns
SOURCES = [
    ("Muennighoff/flores200", "paired"),
    ("openlanguagedata/flores_plus", "per_language"),
    ("facebook/flores", "paired"),
]


def _per_language(repo, split):
    dz = load_dataset(repo, DZ, split=split)
    en = load_dataset(repo, EN, split=split)
    by_id = {row["id"]: row["text"] for row in en}
    pairs = [(row["text"], by_id[row["id"]]) for row in dz if row["id"] in by_id]
    if not pairs:
        raise ValueError(f"{repo}: no sentences aligned on `id`")
    return pairs


def _paired(repo, split):
    ds = load_dataset(repo, f"{DZ}-{EN}", split=split, trust_remote_code=True)
    return list(zip(ds[f"sentence_{DZ}"], ds[f"sentence_{EN}"]))


def load(split="devtest", forced=""):
    """Return [(dzongkha, english), ...] for the requested split."""
    sources = ([(forced, "paired"), (forced, "per_language")] if forced
               else SOURCES)
    errors = []
    for repo, shape in sources:
        try:
            loader = _per_language if shape == "per_language" else _paired
            pairs = loader(repo, split)
            print(f"FLORES source: {repo} ({shape}, {len(pairs)} sentences)")
            return pairs
        except Exception as exc:
            errors.append(f"  {repo} [{shape}]: {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:140]}")
    raise SystemExit(
        "Could not load FLORES from any source:\n" + "\n".join(errors) +
        "\n\nSeveral mirrors are gated. Request access at one of:\n"
        "  https://huggingface.co/datasets/facebook/flores\n"
        "  https://huggingface.co/datasets/openlanguagedata/flores_plus\n"
        "Approval is usually immediate."
    )
