"""Nominee identities and metadata for the Ballon d'Or contests with per-juror ballots.

python -I nhb/src/bdo_nominees.py  ->  nhb/data/processed/bdo_nominee_names.csv, bdo_nominee_meta.csv

Each contest (men 2019, 2022-2025; women 2023-2025) has a closed list of 30 nominees, taken from the
ranking table of the Wikipedia "{year} Ballon d'Or" page (parse_bdo.wiki_tables). Every spelling seen in
the ballots (L'Équipe short names such as "Vinicius Jr.", "D. Doué", "Te. Chawinga"; Wikipedia
per-country column headers; JASA full names; L'Équipe ranking names) is mapped to one nominee of the
same contest:
1. same nominees.key() (accent- and order-free tokens, with nominees.ALIASES);
2. otherwise every token of the spelling matches a different token of exactly one nominee's name:
   equal after folding accents, or a prefix when the spelling abbreviates it with a full stop
   ("D." for Désiré, "Te." for Temwa); "Jr" counts as "Junior";
3. otherwise MANUAL below. A spelling that fits several nominees or none is reported, never guessed.
The person name is the nominee's Wikipedia name, unified across years by key (latest spelling).
`fifa_person` gives the identity used in fifa_nominee_names.csv when the same key occurs there
(nominee_meta.WIKI_ALIAS bridges the remaining spelling differences), to link the two datasets.

Metadata mirror nominee_meta.py: nation from the Nationality column (2019: the flag in the player
cell), clubs from the club cell, each with the country of its flag. Wikipedia lists one club per player
from 2022 and explains moves in footnotes. A move the footnote dates "during the season" adds the
earlier club (and a loan "from A to B during the season" adds both), so every club of the eligibility
period is listed; a move "after the end of season" adds nothing. Such footnote clubs carry no flag:
their country comes from the same club's flag elsewhere on the saved Wikipedia pages, else CLUB_COUNTRY.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from countries import resolver  # noqa: E402
from nominee_meta import WIKI_ALIAS, clubs, player_link  # noqa: E402
from nominees import deaccent, key  # noqa: E402
from parse_bdo import ranking_rows, wiki_soup, wiki_source, wiki_tables  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAWW = ROOT / "data" / "raw" / "wikipedia"
RAWB = ROOT / "data" / "raw" / "ballondor"
CONTESTS = [(2019, "MP"), (2022, "MP"), (2023, "MP"), (2024, "MP"), (2025, "MP"),
            (2023, "WP"), (2024, "WP"), (2025, "WP")]
EQUIV = {"jr": "junior"}
LETTERS = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "Ae", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L"})
# (year, award, spelling) -> Wikipedia name, for spellings the rules cannot place; each checked against
# the contest's nominee list (the only nominee with that surname)
MANUAL = {
    (2019, "MP", "Alisson Becker"): "Alisson",               # JASA full name; Wikipedia uses the mononym
    (2022, "MP", "Son Heug-min"): "Son Heung-min",           # typo in L'Équipe's 2022 ranking box
    (2023, "WP", "Alba Redon do"): "Alba Redondo",           # typo (split word) in L'Équipe's 2023 ranking box
    (2025, "WP", "Leonhardsen Maanum"): "Frida Maanum",      # full surname Leonhardsen Maanum
}
# club -> country for footnote clubs that have no flag anywhere on the saved Wikipedia pages
CLUB_COUNTRY = {"Fiorentina": "Italy"}
# footnote with no date, resolved from L'Équipe's 2023 ranking "Debinha (Brésil, Courage de Caroline du
# Nord puis Current de Kansas City)": both clubs in the eligibility period
UNDATED_MOVES = {(2023, "WP", "Debinha"): "North Carolina Courage"}


def tokens(s):
    """Folded name tokens; a token written with a final full stop is an abbreviation (prefix)."""
    s = deaccent(s.translate(LETTERS)).lower().replace("’", "'")
    out = []
    for t in re.split(r"[\s\-]+", s):
        abbr = t.endswith(".")
        t = re.sub(r"[^a-z0-9]", "", t)
        if t:
            out.append((EQUIV.get(t, t), abbr))
    return out


def fits(raw, name):
    """Every raw token matches a distinct token of `name` (equal, or prefix if abbreviated)."""
    cand = [t for t, _ in tokens(name)]

    def go(i, used):
        if i == len(raw):
            return True
        t, abbr = raw[i]
        for j, c in enumerate(cand):
            if j not in used and (c == t or (abbr and c.startswith(t))):
                if go(i + 1, used | {j}):
                    return True
        return False
    return bool(raw) and go(0, frozenset())


def match(raw, names):
    """(wiki name, rule) or (None, reason) for a spelling against one contest's nominee names."""
    fkey = lambda x: key(x.translate(LETTERS))
    if (hit := [n for n in names if fkey(n) == fkey(raw)]) and len(hit) == 1:
        return hit[0], "key"
    hit = [n for n in names if fits(tokens(raw), n)]
    if len(hit) == 1:
        return hit[0], "tokens"
    return None, ("ambiguous: " + " / ".join(hit)) if hit else "no nominee fits"


def flag_country(cell):
    img = cell.find("img")
    return img.get("alt", "") if img else ""


def footnotes(cell, soup):
    out = []
    for sup in cell.find_all("sup", class_="reference"):
        a = sup.find("a")
        tgt = soup.find(id=a["href"][1:]) if a and a.get("href", "").startswith("#") else None
        if tgt is not None:
            out.append(re.sub(r"^\^\s*", "", tgt.get_text(" ", strip=True)))
    return out


def moves(note):
    """Clubs added by a footnote: (earlier clubs, later clubs) of a move within the season."""
    if m := re.search(r"on loan from (.+?) to (.+?) during the season", note):
        return [m.group(1)], [m.group(2)]
    if m := re.search(r"signed by (.+?) (?:on loan )?from (.+?) during the season", note):
        return [m.group(2)], [m.group(1)]
    return [], []


def club_index():
    """Club name -> set of flag names, from every flagged club cell on the saved Wikipedia pages."""
    idx = {}
    paths = sorted(RAWW.glob("awards_*.json")) + [RAWW / "ballondor_2025.json"] + sorted(RAWB.glob("*_wiki/wikipedia.json"))
    for p in paths:
        s = BeautifulSoup(json.loads(p.read_text())["parse"]["text"], "html.parser")
        for t in s.find_all("table", class_="wikitable"):
            for td in t.find_all("td"):
                if td.find("span", class_="flagicon") is None:
                    continue
                for club, flag in clubs(td):
                    if flag:
                        idx.setdefault(club, set()).add(flag)
    return idx


def nominees(year, award, soup, source, resolve, cidx, problems):
    t, _ = wiki_tables(soup, award)
    out = []
    for rank, cells, _ in ranking_rows(t):
        a = player_link(cells["Player"])
        name = a.get_text(strip=True)
        nat_raw = (cells["Nationality"].get_text(" ", strip=True) if "Nationality" in cells
                   else flag_country(cells["Player"]))
        ccell = next(v for h, v in cells.items() if h.startswith("Club"))
        cl = [(c, k) for c, k in clubs(ccell)]
        notes = footnotes(ccell, soup)
        before, after = [], []
        for n in notes:
            b, f = moves(n)
            before += b
            after += f
        if (year, award, name) in UNDATED_MOVES:
            before.append(UNDATED_MOVES[(year, award, name)])
        listed = [c for c, _ in cl]
        for c in before + after:
            if c in listed:
                continue
            flags = cidx.get(c, set())
            k = next(iter(flags)) if len(flags) == 1 else CLUB_COUNTRY.get(c, "")
            if not k:
                problems.append(f"{year} {award} {name}: no country for footnote club {c} (flags seen: {flags})")
            entry = (c, k)
            cl = [entry] + cl if c in before else cl + [entry]
            listed.append(c)
        nat = resolve(nat_raw) or ""
        if not nat:
            problems.append(f"{year} {award} {name}: unresolved nationality {nat_raw!r}")
        cc = []
        for c, k in cl:
            r = resolve(k) if k else ""
            if k and not r:
                problems.append(f"{year} {award} {name}: unresolved club country {k!r} ({c})")
            cc.append(r or "")
        pts = re.search(r"\d+", cells["Points"].get_text())
        out.append({"year": year, "award": award, "wiki_name": name, "nation": nat,
                    "clubs": ";".join(c for c, _ in cl), "club_countries": ";".join(cc), "source": source,
                    "nation_raw": nat_raw, "wiki_rank": rank, "wiki_points": int(pts.group()) if pts else None,
                    "club_notes": " || ".join(notes)})
    return out


def spellings():
    """(year, award, raw) -> (picks on the ballots, where seen)."""
    seen = {}

    def add(y, a, raw, where, n=0):
        if raw:
            p, w = seen.get((y, a, raw), (0, set()))
            seen[(y, a, raw)] = (p + n, w | {where})
    for f, where in (("bdo_ballots_raw.csv", "ballots"), ("bdo_ballots_jasa.csv", "ballots"),
                     ("bdo_wiki_ballots.csv", "wiki_country_table")):
        b = pd.read_csv(PROC / f, keep_default_na=False)
        for c in [c for c in b.columns if c.startswith("pick")]:
            for (y, a, raw), n in b.groupby(["year", "award", c]).size().items():
                add(y, a, raw, where, n if where == "ballots" else 0)
    off = pd.read_csv(PROC / "bdo_official_points.csv", keep_default_na=False)
    for r in off.itertuples(index=False):
        add(r.year, r.award, r.name, "official_" + r.source.split(":")[0])
    return seen


def main():
    resolve = resolver(pd.read_csv(PROC / "countries.csv", keep_default_na=False).country)
    cidx = club_index()
    problems, meta = [], []
    for year, award in CONTESTS:
        meta += nominees(year, award, wiki_soup(year), wiki_source(year), resolve, cidx, problems)
    meta = pd.DataFrame(meta)

    # person: Wikipedia name, unified across years by key (latest spelling); link to the FIFA identity
    meta["key"] = meta.wiki_name.map(lambda n: key(WIKI_ALIAS.get(n, n)))
    latest = meta.sort_values("year").groupby("key").wiki_name.last()
    meta["person"] = meta.key.map(latest)
    fifa = pd.read_csv(PROC / "fifa_nominee_names.csv", keep_default_na=False)
    fifa_by_key = dict(zip(fifa.raw.map(key), fifa.person))
    meta["fifa_person"] = meta.key.map(lambda k: fifa_by_key.get(k, ""))

    rows, unresolved = [], []
    names_of = meta.groupby(["year", "award"]).wiki_name.apply(list).to_dict()
    person_of = dict(zip(zip(meta.year, meta.award, meta.wiki_name), meta.person))
    fifa_of = dict(zip(zip(meta.year, meta.award, meta.wiki_name), meta.fifa_person))
    for (y, a, raw), (picks, where) in sorted(spellings().items()):
        if (y, a, raw) in MANUAL:
            wiki, rule = MANUAL[(y, a, raw)], "manual"
        else:
            wiki, rule = match(raw, names_of[(y, a)])
        if wiki is None:
            unresolved.append((y, a, raw, picks, ";".join(sorted(where)), rule))
            wiki, rule = "", "unresolved: " + rule
        rows.append({"year": y, "award": a, "raw": raw, "picks": picks, "seen_in": ";".join(sorted(where)),
                     "wiki_name": wiki, "person": person_of.get((y, a, wiki), ""),
                     "fifa_person": fifa_of.get((y, a, wiki), ""), "rule": rule})
    names = pd.DataFrame(rows)
    names.to_csv(PROC / "bdo_nominee_names.csv", index=False)

    picks = names[names.picks > 0].groupby(["year", "award", "wiki_name"]).picks.sum()
    meta["picks"] = [int(picks.get((y, a, n), 0)) for y, a, n in zip(meta.year, meta.award, meta.wiki_name)]
    cols = ["year", "award", "person", "nation", "clubs", "club_countries", "source", "wiki_name", "fifa_person",
            "nation_raw", "wiki_rank", "wiki_points", "picks", "club_notes"]
    meta[cols].to_csv(PROC / "bdo_nominee_meta.csv", index=False)

    print(f"{len(meta)} nominee rows ({meta.person.nunique()} persons, "
          f"{(meta.fifa_person != '').sum()} rows linked to a FIFA identity); {len(names)} spellings: "
          f"{names.rule.str.split(':').str[0].value_counts().to_dict()}", file=sys.stderr)
    multi = meta.groupby("person").wiki_name.unique()
    for p, s in multi[multi.map(len) > 1].items():
        print(f"  several Wikipedia spellings: {p}: {' | '.join(s)}", file=sys.stderr)
    for u in unresolved:
        print(f"  UNRESOLVED {u}", file=sys.stderr)
    for p in problems:
        print(f"  META {p}", file=sys.stderr)
    tok = {}
    for p in meta.person.unique():
        for t in key(p).split():
            tok.setdefault(t, set()).add(p)
    shared = {t: ps for t, ps in tok.items() if len(ps) > 1 and len(t) > 3}
    print("persons sharing a name token (check these are different people):", file=sys.stderr)
    for t, ps in sorted(shared.items()):
        print(f"  {t}: {' | '.join(sorted(ps))}", file=sys.stderr)
    return 1 if unresolved or problems else 0


if __name__ == "__main__":
    sys.exit(main())
