"""Canonical country names, FIFA codes and confederations for voters, nominees and clubs.

python -I nhb/src/countries.py  ->  nhb/data/processed/countries.csv, country_aliases.csv

Canonical names and confederations come from the confederation sections of Wikipedia's "List of
men's national association football teams" (data/raw/wikipedia/national_teams.json); FIFA codes
from "List of FIFA country codes" (fifa_codes.json). Ballot country names (FIFA's own, varying by
year), Wikipedia nationality names and club flag names are matched to them after folding accents
and punctuation; ALIASES lists the remaining variants by hand. Every raw name must resolve.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia"
PROC = ROOT / "data" / "processed"
CONFEDS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")

ALIASES = {
    # FIFA ballot spellings, including typos and a name cut off by the PDF layout
    "Bosnia Herzegovina": "Bosnia and Herzegovina", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Burma": "Myanmar", "Cape Verde Islands": "Cabo Verde", "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia", "Equitorial Guinea": "Equatorial Guinea", "FYR Macedonia": "North Macedonia",
    "Gambia": "The Gambia", "Hong Kong": "Hong Kong, China", "Iran": "IR Iran", "Kyrgyzstan": "Kyrgyz Republic",
    "Lybia": "Libya", "Maleysia": "Malaysia", "North Korea": "Korea DPR", "Papua New Guines": "Papua New Guinea",
    "Sao Tome e Principe": "São Tomé and Príncipe", "São Tomé e Príncipe": "São Tomé and Príncipe",
    "South Korea": "Korea Republic", "St Kitts & Nevis": "Saint Kitts and Nevis",
    "St Vincent and the": "Saint Vincent and the Grenadines", "St. Vincent and the": "Saint Vincent and the Grenadines",
    "Swaziland": "Eswatini", "Tanzanie": "Tanzania", "Trinidad & Tobago": "Trinidad and Tobago", "Turkey": "Türkiye",
    "USA": "United States",
    "Ireland": "Republic of Ireland",   # listed separately from Northern Ireland on the same ballots
    # Wikipedia nationality and club-flag names
    "Ivory Coast": "Côte d'Ivoire", "China": "China PR", "Brunei": "Brunei Darussalam", "DR Congo": "Congo DR",
    "U.S. Virgin Islands": "US Virgin Islands", "United States Virgin Islands": "US Virgin Islands",
    "The Bahamas": "Bahamas", "French Polynesia": "Tahiti", "Taiwan": "Chinese Taipei",
    "Democratic Republic of the Congo": "Congo DR", "Republic of the Congo": "Congo", "Republic of Kosova": "Kosovo",
    # coaching-history flags: current names of the same associations
    "Zaire": "Congo DR", "West Germany": "Germany",
}


def fold(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\bst\.?\s", "saint ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def soup(name):
    return BeautifulSoup(json.loads((RAW / name).read_text())["parse"]["text"], "html.parser")


def members():
    out = []
    for t in soup("national_teams.json").find_all("table"):
        h = t.find_previous(["h2", "h3"])
        conf = h.get_text(" ", strip=True).split(" ")[0] if h else ""
        if conf not in CONFEDS or t.get("class"):
            continue
        for a in t.find_all("a"):
            if a.find_parent("sup") is None and a.get_text(strip=True):
                out.append({"country": a.get_text(strip=True), "confederation": conf})
    return pd.DataFrame(out).drop_duplicates("country")


def codes():
    out = []
    s = soup("fifa_codes.json")
    for t in s.find_all("table", class_="wikitable"):
        h = t.find_previous(["h2", "h3"])
        if not h or h.get_text(" ", strip=True) != "FIFA member codes":
            continue
        for tr in t.find_all("tr")[1:]:
            c = [x.get_text(" ", strip=True) for x in tr.find_all(["td", "th"])]
            if len(c) >= 2:
                out.append({"name": c[0], "code": re.sub(r"\s*\[.*", "", c[1])})
    return pd.DataFrame(out)


def raw_names():
    b = pd.read_csv(PROC / "fifa_ballots_raw.csv", keep_default_na=False)
    m = pd.read_csv(PROC / "fifa_nominee_meta.csv", keep_default_na=False)
    clubs = {c for x in m.club_countries for c in x.split(" | ") if c}
    return {"ballot": set(b.country), "nation": set(m.nation), "club": clubs}


def resolver(canon):
    by_fold = {fold(c): c for c in canon}

    def resolve(raw):
        raw = ALIASES.get(raw, raw)
        return by_fold.get(fold(raw))
    return resolve


def main():
    mem = members()
    cod = codes()
    resolve = resolver(mem.country)
    cod["country"] = cod.name.map(resolve)
    bad = cod[cod.country.isna()]
    mem = mem.merge(cod.dropna(subset=["country"])[["country", "code"]], on="country", how="left")
    rows, missing = [], []
    for kind, names in raw_names().items():
        for n in sorted(names):
            c = resolve(n)
            (rows if c else missing).append({"kind": kind, "raw": n, "country": c})
    print(f"{len(mem)} members ({mem.code.isna().sum()} without code); code rows unmatched: "
          f"{list(bad.name)}", file=sys.stderr)
    print(f"unresolved raw names: {len(missing)}", file=sys.stderr)
    for r in missing:
        print(f"  {r['kind']}: {r['raw']}", file=sys.stderr)
    mem.to_csv(PROC / "countries.csv", index=False)
    pd.DataFrame(rows).to_csv(PROC / "country_aliases.csv", index=False)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
