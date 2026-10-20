"""Nominee nationality and clubs per contest, from the Wikipedia award pages.

python -I nhb/src/nominee_meta.py  ->  nhb/data/processed/fifa_nominee_meta.csv

Source: data/raw/wikipedia/awards_{year}.json (MediaWiki parse API, saved 2026-10-17). Each nominee
row gives the national team and the club(s) during the eligibility period, each club with the flag
of its country; a player who moved clubs has several. The 2025 FIFA page lists only the top three,
so the other 2025 nominees come from the 2025 Ballon d'Or page (ballondor_2025.json), whose voting
period (1 August 2024 to 2 August 2025) nearly matches FIFA's (11 August 2024 to 2 August 2025);
the `source` column records which page each row came from. Rows are matched to the ballot
identities of nominees.py by the same accent- and order-free key; unmatched names are reported.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nominees import key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia"
PROC = ROOT / "data" / "processed"
SECTIONS = {"MP": re.compile(r"^(FIFA Ballon d'Or|The Best FIFA Men's Player)$"),
            "WP": re.compile(r"^(FIFA Women's World Player of the Year|The Best FIFA Women's Player)$")}
SUPPLEMENT = {(2025, "MP"): ("ballondor_2025.json", re.compile(r"^Ballon d'Or$")),
              (2025, "WP"): ("ballondor_2025.json", re.compile(r"^Ballon d'Or Féminin$"))}
# 2025 nominees on neither page: clubs and national team from each player's infobox
# (data/raw/wikipedia/player_*.json), restricted to the 2024-25 voting period
MANUAL = [
    (2025, "WP", "Nathalie Björn", "Sweden", "Everton | Chelsea", "England | England", "player_Nathalie_Bj_C3_B6rn.json"),
    (2025, "WP", "Kadidiatou Diani", "France", "Lyon", "France", "player_Kadidiatou_Diani.json"),
    (2025, "WP", "Patri Guijarro", "Spain", "Barcelona", "Spain", "player_Patricia_Guijarro.json"),
    (2025, "WP", "Lauren James", "England", "Chelsea", "England", "player_Lauren_James.json"),
]
# Wikipedia name -> ballot spelling, where the two differ beyond accents and word order
WIKI_ALIAS = {
    "Thomas Müller": "Thomas Müller", "Mario Götze": "Goetze Mario", "Lena Goeßling": "Goessling Lena",
    "Ji So-yun": "Ji So-yun", "Vinícius Júnior": "Vinícius Jr", "Lindsey Horan": "Lindsey Horan",
    "Samantha Kerr": "Sam Kerr", "Thiago": "Thiago Alcântara",
}


def heading(t):
    h = t.find_previous(["h2", "h3"])
    return h.get_text(" ", strip=True) if h else ""


def cols(t):
    hdr = [th.get_text(" ", strip=True).lower() for th in t.find("tr").find_all(["th", "td"])]
    find = lambda *ws: next((i for i, h in enumerate(hdr) if any(w in h for w in ws)), None)
    return find("player"), find("national team", "nationality"), find("club"), find("percent", "points")


def clubs(cell):
    out = []
    for flag in cell.find_all("span", class_="flagicon"):
        img = flag.find("img")
        link = flag.find_next("a", href=re.compile(r"^/wiki/"), class_=lambda c: True)
        while link is not None and link.find_parent("span", class_="flagicon") is not None:
            link = link.find_next("a")
        if link is None or link.find_parent("td") is not cell:
            continue
        out.append((link.get_text(strip=True), img.get("alt", "") if img else ""))
    if not out:   # club without a flag
        out = [(a.get_text(strip=True), "") for a in cell.find_all("a")]
    return out


def player_link(cell):
    links = [a for a in cell.find_all("a")
             if a.find_parent("span", class_="flagicon") is None and a.find_parent("sup") is None]
    return links[-1] if links else None


def contest_rows(year, award, soup, section, source):
    rows = []
    for t in soup.find_all("table", class_="wikitable"):
        if not section.match(heading(t)):
            continue
        ip, inat, iclub, ipct = cols(t)
        if ip is None or inat is None or iclub is None:
            continue
        width = len(t.find("tr").find_all(["th", "td"]))
        for tr in t.find_all("tr")[1:]:
            cs = tr.find_all(["td", "th"])
            # a rank cell spanning several rows is absent from the later rows of the span
            shift = {width: 0, width - 1: -1}.get(len(cs))
            if shift is None or ip + shift < 0:
                continue
            a = player_link(cs[ip + shift])
            if a is None:
                continue
            cl = clubs(cs[iclub + shift])
            score = cs[ipct + shift].get_text(" ", strip=True) if ipct is not None and ipct + shift < len(cs) else ""
            rows.append({"year": year, "award": award, "wiki_name": a.get_text(strip=True), "wiki_score": score,
                         "wiki_title": a.get("title", ""), "nation": cs[inat + shift].get_text(" ", strip=True),
                         "clubs": " | ".join(c for c, _ in cl), "club_countries": " | ".join(k for _, k in cl),
                         "source": source})
    return rows


def main():
    load = lambda path: BeautifulSoup(json.loads(path.read_text())["parse"]["text"], "html.parser")
    rows = []
    for path in sorted(RAW.glob("awards_*.json")):
        year = int(path.stem.split("_")[1])
        soup = load(path)
        for award, section in SECTIONS.items():
            rows += contest_rows(year, award, soup, section, path.name)
    for (year, award), (fname, section) in SUPPLEMENT.items():
        rows += contest_rows(year, award, load(RAW / fname), section, fname)
    rows += [dict(zip(["year", "award", "wiki_name", "nation", "clubs", "club_countries", "source"], r),
                  wiki_title=r[2], wiki_score="") for r in MANUAL]
    meta = pd.DataFrame(rows)
    meta["key"] = meta.wiki_name.map(lambda n: key(WIKI_ALIAS.get(n, n)))
    meta = meta.drop_duplicates(["year", "award", "key"])   # FIFA page first, supplement only fills gaps

    names = pd.read_csv(PROC / "fifa_nominee_names.csv", keep_default_na=False)
    names["key"] = names.raw.map(key)
    per = names.groupby(["year", "award", "key"]).agg(person=("person", "first"), picks=("picks", "sum")).reset_index()
    m = per.merge(meta, on=["year", "award", "key"], how="outer", indicator=True)
    miss_wiki = m[m._merge == "left_only"]
    miss_ballot = m[m._merge == "right_only"]
    print(f"wikipedia rows {len(meta)}; ballot nominees {len(per)}; matched {(m._merge == 'both').sum()}",
          file=sys.stderr)
    print("on ballots, not found on Wikipedia:\n" + miss_wiki[["year", "award", "person", "picks"]].to_string(),
          file=sys.stderr)
    print("on Wikipedia, never picked on a ballot:\n" + miss_ballot[["year", "award", "wiki_name"]].to_string(),
          file=sys.stderr)
    out = m[m._merge != "right_only"].drop(columns=["_merge"])
    out.to_csv(PROC / "fifa_nominee_meta.csv", index=False)


if __name__ == "__main__":
    main()
