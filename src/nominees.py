"""One identity per nominee across the 32 FIFA contests.

python -I nhb/src/nominees.py  ->  nhb/data/processed/fifa_nominee_names.csv (year, award, raw, person)

Ballots name nominees surname first up to 2019 and first name first from 2020, with changing accents
and transliterations. Two spellings are the same person when their accent-folded tokens agree as
sets; ALIASES covers the remaining true variants (nicknames, married names, typos). Each alias was
checked against the nominee lists; similar-looking names of different people (the Chawinga sisters,
Thiago Silva and Thiago Alcântara) are deliberately not merged.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PLACEHOLDERS = {"", "INVALID", "NOT_COUNTED", "NA", "NO_VOTE"}

# raw spelling -> spelling whose key it should share
ALIASES = {
    "Alves Daniel": "Dani Alves",
    "Mueller Thomas": "Thomas Müller",
    "Goetze Mario": "Mario Götze",
    "Jenni Hermoso": "Jennifer Hermoso",
    "Maroszán Dzsenifer": "Dzsenifer Marozsán",
    "Vinícius Junior": "Vinícius Jr",
    "Lindsey Heaps": "Lindsey Horan",          # married name from 2025
    "Goessling Lena": "Lena Gößling",
}
# display name for persons who appear only in surname-first years
DISPLAY = {}


def deaccent(s):
    s = s.replace("ß", "ss").replace("ı", "i")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def key(name):
    name = ALIASES.get(name, name)
    toks = re.split(r"[\s\-]+", re.sub(r"[’'`.]", "", deaccent(name).lower()))
    return " ".join(sorted(t for t in toks if t))


def main():
    b = pd.read_csv(PROC / "fifa_ballots_raw.csv", keep_default_na=False)
    long = b.melt(id_vars=["year", "award"], value_vars=["pick1", "pick2", "pick3"], value_name="raw")
    names = (long[~long.raw.isin(PLACEHOLDERS)].groupby(["year", "award", "raw"]).size()
             .rename("picks").reset_index())
    names["key"] = names.raw.map(key)
    # display: the latest first-name-first spelling (2020 onwards), else the latest spelling
    latest = names.sort_values("year").groupby("key").agg(last=("raw", "last"), last_year=("year", "last"))
    late = names[names.year >= 2020].sort_values("year").groupby("key").raw.last()
    latest["display"] = late.reindex(latest.index).fillna(latest["last"])
    latest["display"] = [DISPLAY.get(k, d) for k, d in latest.display.items()]
    names["person"] = names.key.map(latest.display)
    names = names.drop(columns="key")
    names.to_csv(PROC / "fifa_nominee_names.csv", index=False)

    spellings = names.groupby("person").raw.unique()
    multi = spellings[spellings.map(len) > 1]
    print(f"{names.raw.nunique()} spellings -> {names.person.nunique()} persons; "
          f"{len(multi)} persons with several spellings:", file=sys.stderr)
    for p, s in multi.items():
        print(f"  {p}: {' | '.join(s)}", file=sys.stderr)
    # near-collisions worth a look: persons sharing a surname-like token
    tok = {}
    for p in spellings.index:
        for t in key(p).split():
            tok.setdefault(t, []).append(p)
    shared = {t: ps for t, ps in tok.items() if len(ps) > 1 and len(t) > 3}
    print("persons sharing a name token (check these are different people):", file=sys.stderr)
    for t, ps in sorted(shared.items()):
        print(f"  {t}: {' | '.join(ps)}", file=sys.stderr)


if __name__ == "__main__":
    main()
