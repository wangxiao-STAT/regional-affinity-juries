"""Coach nationality from the national teams' own Wikipedia pages (men's and women's teams).

python -I nhb/src/coach_history.py  ->  nhb/data/processed/coach_history_entries.csv,
                                         fifa_coach_nationality_all.csv

Source: data/raw/wikipedia/teams/ (fetch_team_pages.py). Under every heading that mentions coaches
or managers (coaching history, managerial history, coaching staff) each table row or list item
naming a person with a flag gives (person, nationality). A coach ballot is matched by name to the
people listed for the voter's team (men's page for the men's award, women's page for the women's
award), with the token rule of coach_nationality.py; name plus team identifies the person, so no
dates are needed. For the men's ballots a match from the dated manager-list snapshots
(fifa_coach_nationality.csv) takes precedence. A person listed with flags of two different
countries, or two different listed people matching one ballot name, leaves the ballot unmatched.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coach_nationality import same_person  # noqa: E402
from nominees import key  # noqa: E402
from countries import ALIASES, fold as cfold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia" / "teams"
PROC = ROOT / "data" / "processed"
HEAD = re.compile(r"coach|manag", re.I)


def person_flag(node):
    """(name, page, flag) of the first person link in a cell or list item that carries a flag."""
    flag = node.find("span", class_="flagicon")
    if flag is None:
        return None
    img = flag.find("img")
    links = [a for a in node.find_all("a", href=re.compile(r"^/wiki/"))
             if a.find_parent("span", class_="flagicon") is None and a.find_parent("sup") is None]
    if not links or img is None:
        return None
    return links[0].get_text(strip=True), links[0].get("title", ""), img.get("alt", "")


def entries(path):
    d = json.loads(path.read_text())
    if d.get("missing"):
        return []
    soup = BeautifulSoup(d["html"], "html.parser")
    out = []
    for node in soup.find_all(["table", "ul"]):
        h = node.find_previous(["h2", "h3", "h4"])
        if h is None or not HEAD.search(h.get_text(" ", strip=True)):
            continue
        if node.find_parent(["table", "ul"]) is not None:
            continue
        items = []
        if node.name == "table":
            for tr in node.find_all("tr"):
                for cell in tr.find_all(["td", "th"]):
                    pf = person_flag(cell)
                    if pf:
                        items.append(pf)
                        break
        else:
            items = [pf for li in node.find_all("li", recursive=False) if (pf := person_flag(li))]
        for name, page, flag in items:
            out.append({"side": d["side"], "team": d["country"], "page_title": d["title"], "section":
                        h.get_text(" ", strip=True)[:60], "person": name, "person_page": page, "flag": flag})
    return out


def main():
    ent = pd.DataFrame([e for p in sorted(RAW.glob("*.json")) for e in entries(p)])
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False).country
    by_fold = {cfold(c): c for c in countries}

    def resolve(n):
        n = re.sub(r"\s*\(.*?\)|\[.*?\]", "", n).strip()
        return by_fold.get(cfold(ALIASES.get(n, n)), "")
    ent["nationality"] = ent.flag.map(resolve)
    ent.to_csv(PROC / "coach_history_entries.csv", index=False)
    print(f"{len(ent)} entries from {ent.page_title.nunique()} pages; unresolved flags: "
          f"{sorted(set(ent.flag[ent.nationality == '']))[:25]}", file=sys.stderr)

    b = pd.read_csv(PROC / "fifa_ballots.csv", keep_default_na=False)
    snap = pd.read_csv(PROC / "fifa_coach_nationality.csv", keep_default_na=False).set_index("ballot")
    by_team = {k: g for k, g in ent.groupby(["side", "team"])}
    rows = []
    for r in b[b.role == "coach"].itertuples(index=False):
        side = "men" if r.award == "MP" else "women"
        nat, how, who, page = "", "", "", ""
        if side == "men" and r.ballot in snap.index and snap.loc[r.ballot, "coach_nationality"] != "":
            nat, how, who = snap.loc[r.ballot, "coach_nationality"], "manager_list", snap.loc[r.ballot, "matched_name"]
            page = snap.loc[r.ballot, "page"]
        else:
            g = by_team.get((side, r.voter_country))
            if g is not None:
                hit = g[[same_person(r.voter, p) for p in g.person]]
                people = set(hit.person_page.replace("", pd.NA).dropna()) or set(hit.person)
                nats = set(hit.nationality) - {""}
                if len(people) == 1 and len(nats) == 1:
                    nat, how, who = nats.pop(), "team_page", hit.person.iloc[0]
                    page = hit.person_page.iloc[0]
                elif len(hit):
                    how = "ambiguous"
        # a stable identity across ballots: the coach's Wikipedia page, else the order-free name key
        coach_id = page or (key(who) if who else key(r.voter))
        rows.append({"ballot": r.ballot, "year": r.year, "award": r.award, "voter_country": r.voter_country,
                     "voter": r.voter, "coach_nationality": nat, "match": how, "matched_name": who,
                     "coach_id": coach_id})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "fifa_coach_nationality_all.csv", index=False)
    known = out[out.coach_nationality != ""]
    s = out.assign(known=out.coach_nationality != "",
                   foreign=(out.coach_nationality != "") & (out.coach_nationality != out.voter_country)) \
        .groupby(["award", "year"]).agg(ballots=("ballot", "size"), known=("known", "sum"), foreign=("foreign", "sum"))
    print(s.to_string(), file=sys.stderr)
    print(f"known {len(known)} of {len(out)}; foreign {int((known.coach_nationality != known.voter_country).sum())}; "
          f"by source {known.match.value_counts().to_dict()}; ambiguous {(out.match == 'ambiguous').sum()}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
