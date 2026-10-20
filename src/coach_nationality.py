"""Nationality of the national-team coaches who voted in the men's awards.

python -I nhb/src/coach_nationality.py  ->  nhb/data/processed/manager_snapshots.csv, fifa_coach_nationality.csv

Source: quarterly revisions of Wikipedia's "List of current national association football team
managers" (fetch_managers.py). Each row names a team, its manager and the manager's nationality
(the flag beside the name). The list covered only UEFA teams until early 2024, and its 2016-2018
revisions show no flags, so nationality is treated as an attribute of the person: a manager's
flag in any snapshot (keyed by the manager's Wikipedia page) fills the snapshots without one.
A coach ballot of year Y is matched by name to the managers listed for the voter's team, first in
the snapshots from 1 February Y to 1 February Y+1 ("window"), else in any snapshot ("any_date");
name plus team identifies the person. Names match when every token of the shorter one appears in
the other after folding accents (ballots up to 2019 put the surname first). Ballots with no listed
manager of that name stay unmatched. Women's teams are not covered by this list.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from countries import ALIASES, fold as cfold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia" / "managers"
PROC = ROOT / "data" / "processed"


def tokens(name):
    s = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    return {t for t in re.split(r"[^a-z]+", s) if len(t) > 1}


def same_person(a, b):
    ta, tb = tokens(a), tokens(b)
    short, long_ = sorted((ta, tb), key=len)
    return bool(short) and short <= long_


def snapshot_rows(path):
    d = json.loads(path.read_text())
    soup = BeautifulSoup(d["html"], "html.parser")
    out = []
    for t in soup.find_all("table", class_="wikitable"):
        trs = t.find_all("tr")
        hdr = [c.get_text(" ", strip=True).lower() for c in trs[0].find_all(["th", "td"])]
        it = next((i for i, h in enumerate(hdr) if h.startswith(("country", "team", "nation"))), None)
        im = next((i for i, h in enumerate(hdr) if "manager" in h or "coach" in h), None)
        inat = next((i for i, h in enumerate(hdr) if "nationality" in h), None)
        if it is None or im is None:
            continue
        for tr in trs[1:]:
            cs = tr.find_all(["td", "th"])
            if len(cs) <= max(it, im):
                continue
            team = cs[it].get_text(" ", strip=True)
            mcell = cs[im]
            links = [a for a in mcell.find_all("a") if a.find_parent("span", class_="flagicon") is None
                     and a.find_parent("sup") is None]
            manager = links[0].get_text(strip=True) if links else mcell.get_text(" ", strip=True)
            page = links[0].get("title", "") if links else ""
            flags = [i.get("alt", "") for i in mcell.find_all("img") if i.get("alt")]
            if inat is not None and inat < len(cs):
                flags = [i.get("alt", "") for i in cs[inat].find_all("img") if i.get("alt")] or \
                    [cs[inat].get_text(" ", strip=True)]
            out.append({"snapshot": d["date"], "revid": d["revid"], "rev_time": d["timestamp"], "team": team,
                        "manager": manager, "page": page, "nationality": flags[0] if flags else "",
                        "nationality2": flags[1] if len(flags) > 1 else ""})
    return out


def main():
    snaps = pd.DataFrame([r for p in sorted(RAW.glob("rev_*.json")) for r in snapshot_rows(p)])
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False).country
    by_fold = {cfold(c): c for c in countries}
    def resolve(n):
        n = re.sub(r"\s*\(.*?\)|\[.*?\]", "", n).strip()
        return by_fold.get(cfold(ALIASES.get(n, n)), "") if n else ""
    snaps["team_country"] = snaps.team.map(resolve)
    snaps["nat_country"] = snaps.nationality.map(resolve)
    # nationality is a property of the person: fill snapshots without a flag from the same page
    known = snaps[(snaps.page != "") & (snaps.nat_country != "")].groupby("page").nat_country \
        .agg(lambda x: x.mode().iloc[0])
    fill = (snaps.nat_country == "") & snaps.page.isin(known.index)
    snaps.loc[fill, "nat_country"] = snaps.loc[fill, "page"].map(known)
    snaps["nat_filled"] = fill
    snaps.to_csv(PROC / "manager_snapshots.csv", index=False)
    cover = snaps.groupby("snapshot").agg(teams=("team_country", lambda s: (s != "").sum()),
                                          with_nat=("nat_country", lambda s: (s != "").sum()))
    print("snapshot coverage (teams resolved, nationality resolved):\n" + cover.to_string(), file=sys.stderr)
    unres = snaps[(snaps.team_country == "")].team.unique()
    print(f"unresolved team names: {list(unres)[:30]}", file=sys.stderr)
    unres = snaps[(snaps.nat_country == "") & (snaps.nationality != "")].nationality.unique()
    print(f"unresolved nationality names: {list(unres)[:30]}", file=sys.stderr)

    b = pd.read_csv(PROC / "fifa_ballots.csv", keep_default_na=False)
    coaches = b[(b.role == "coach") & (b.award == "MP")][["ballot", "year", "voter_country", "voter"]]
    rows = []
    team_snaps = dict(tuple(snaps.groupby("team_country")))
    for r in coaches.itertuples(index=False):
        lo, hi = f"{r.year}-02-01", f"{r.year + 1}-02-01"
        team = team_snaps.get(r.voter_country, snaps.iloc[:0])
        window = team[(team.snapshot >= lo) & (team.snapshot <= hi)]
        how, hit = "", team.iloc[:0]
        for label, cand in (("window", window), ("any_date", team)):
            hit = cand[np.array([same_person(r.voter, m) for m in cand.manager], dtype=bool)]
            if len(hit):
                how = label
                break
        nats = sorted(set(hit.nat_country) - {""})
        rows.append({"ballot": r.ballot, "year": r.year, "voter_country": r.voter_country, "voter": r.voter,
                     "listed": len(window) > 0, "match": how, "matched_name": hit.manager.iloc[0] if len(hit) else "",
                     "page": hit.page.iloc[0] if len(hit) else "",
                     "coach_nationality": nats[0] if len(nats) == 1 else "", "conflict": len(nats) > 1})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "fifa_coach_nationality.csv", index=False)
    s = out.groupby("year").agg(ballots=("ballot", "size"), team_listed=("listed", "sum"),
                                name_matched=("match", lambda x: (x != "").sum()),
                                any_date=("match", lambda x: (x == "any_date").sum()),
                                matched=("coach_nationality", lambda x: (x != "").sum()),
                                conflicts=("conflict", "sum"))
    s["foreign"] = out[out.coach_nationality != ""].assign(f=lambda d: d.coach_nationality != d.voter_country) \
        .groupby("year").f.sum()
    print("men's coach ballots matched to a listed manager with nationality:\n" + s.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
