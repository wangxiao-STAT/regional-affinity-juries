"""Eurovision jury rankings in the analysis format of fifa_long.csv.

python -I nhb/src/build_eurovision.py  ->  nhb/data/processed/esc_long.csv, esc_regions.csv, esc_borders.csv

One row per juror and song in each grand final (the juror's own country's song is not in the juror's
choice set: jurors may not rank it). Similarity indicators:
  same_region   the juror's country and the song's country are in the same UN M49 subregion (Northern,
                Western, Southern and Eastern Europe, Western Asia, Australia and New Zealand; checked
                against unstats.un.org/unsd/methodology/m49 on 2026-10-20), the counterpart of FIFA's
                confederations;
  same_bloc     the two countries are in the same voting bloc of the five Gatherer (2006) found in the
                contests of 2001-2005 (BLOCS); 0 for a country in no bloc;
  voter_bloc,   the groups used to compose simulated juries (GROUP): the country's bloc, or for a country
  nominee_bloc  in no bloc the other such countries of its UN subregion ("<subregion> (other)"); these
                residual groups carry no bonus term;
  same_fine     the two countries are in the same group of a finer grouping chosen after the main analysis
                (FINE; a sensitivity check);
  same_border   the two countries' home territories share a land border (Wikipedia, "List of countries and
                territories by number of land borders", revision in data/raw/eurovision/borders; borders
                through overseas territories excluded, see OVERSEAS), a narrower tie.
Jurors are citizens of the country they vote for, so there is no compatriot term: the rules ban it.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "eurovision"

M49 = {
    "Northern Europe": ["Denmark", "Estonia", "Finland", "Iceland", "Ireland", "Latvia", "Lithuania", "Norway",
                        "Sweden", "United Kingdom"],
    "Western Europe": ["Austria", "Belgium", "France", "Germany", "Liechtenstein", "Luxembourg", "Monaco",
                       "Netherlands", "Switzerland"],
    "Southern Europe": ["Albania", "Andorra", "Bosnia and Herzegovina", "Croatia", "Greece", "Italy", "Malta",
                        "Montenegro", "North Macedonia", "Portugal", "San Marino", "Serbia", "Slovenia", "Spain"],
    "Eastern Europe": ["Belarus", "Bulgaria", "Czechia", "Hungary", "Moldova", "Poland", "Romania", "Russia",
                       "Slovakia", "Ukraine"],
    "Western Asia": ["Armenia", "Azerbaijan", "Cyprus", "Georgia", "Israel", "Turkey"],
    "Australia and New Zealand": ["Australia"],
}
REGION = {c: r for r, cs in M49.items() for c in cs}
# Voting blocs identified from the contests of 2001-2005 (Gatherer 2006, JASSS 9(2):1, tables 7 and 10
# and text): the "Viking Empire", the "Balkan Bloc" (Serbia and Montenegro then one country), the
# "Warsaw Pact", the "Pyrenean Axis" and the Netherlands-Belgium partnership. They predate the data
# (2016-2026), so the grouping is not fitted to the votes analysed. The bonus indicator same_bloc uses
# these blocs only. For composing simulated juries every country needs a group: a country in none of
# them is grouped with the other such countries of its UN subregion ("<subregion> (other)").
BLOCS = {
    "Viking Empire": ["Sweden", "Denmark", "Norway", "Iceland", "Estonia", "Finland", "Latvia", "Lithuania"],
    "Balkan Bloc": ["Serbia", "Montenegro", "Croatia", "North Macedonia", "Slovenia", "Bosnia and Herzegovina",
                    "Greece", "Cyprus", "Romania", "Albania", "Turkey"],   # Turkey: absent since 2012
    "Warsaw Pact": ["Russia", "Ukraine", "Poland"],
    "Pyrenean Axis": ["Andorra", "Spain"],
    "Netherlands-Belgium": ["Netherlands", "Belgium"],
}
BLOC = {c: b for b, cs in BLOCS.items() for c in cs}
GROUP = {c: BLOC.get(c, f"{r} (other)") for c, r in REGION.items()}
# A finer grouping by shared history and geography, chosen after the main analysis (a sensitivity check;
# it was proposed in review): Nordic and Baltic states apart, the British Isles, former Yugoslavia, the
# Visegrad countries, the former Soviet republics outside the Baltic, and Bulgaria with Romania.
FINE = {
    "Nordic": ["Denmark", "Finland", "Iceland", "Norway", "Sweden"],
    "Baltic": ["Estonia", "Latvia", "Lithuania"],
    "British Isles": ["Ireland", "United Kingdom"],
    "Western": ["Austria", "Belgium", "France", "Germany", "Liechtenstein", "Luxembourg", "Monaco",
                "Netherlands", "Switzerland"],
    "Former Yugoslavia": ["Bosnia and Herzegovina", "Croatia", "Montenegro", "North Macedonia", "Serbia",
                          "Slovenia"],
    "Mediterranean": ["Albania", "Andorra", "Cyprus", "Greece", "Italy", "Malta", "Portugal", "San Marino",
                      "Spain"],
    "Visegrad": ["Czechia", "Hungary", "Poland", "Slovakia"],
    "Former USSR": ["Armenia", "Azerbaijan", "Belarus", "Georgia", "Moldova", "Russia", "Ukraine"],
    "Bulgaria-Romania": ["Bulgaria", "Romania"],
    "Israel": ["Israel"], "Turkey": ["Turkey"], "Australia": ["Australia"],
}
FINE_GROUP = {c: g for g, cs in FINE.items() for c in cs}
# borders through overseas territories, not between the two countries' home territories
OVERSEAS = {("Cyprus", "United Kingdom"),     # the UK's Sovereign Base Areas on Cyprus
            ("Spain", "United Kingdom")}      # Gibraltar
BORDER_ALIASES = {"Czech Republic": "Czechia", "Metropolitan France": "France", "France, Metropolitan": "France",
                  "Kingdom of the Netherlands": "Netherlands", "Republic of Ireland": "Ireland",
                  "State of Palestine": "Palestine", "Türkiye": "Turkey"}


def borders():
    t = json.loads((RAW / "borders" / "land_borders.json").read_text())["parse"]["wikitext"]
    pairs = set()
    for row in re.split(r"\n\|-[^\n]*\n", t):
        cells = [c for c in row.split("\n|") if c.strip()]
        if len(cells) < 2:
            continue
        head = re.search(r"\{\{(?:flag|flagicon|flagcountry)\|([^|}]+)", cells[0])
        if not head:
            continue
        a = BORDER_ALIASES.get(head.group(1).strip(), head.group(1).strip())
        for b in re.findall(r"\{\{flag\|([^|}]+)", cells[-1]):
            b = BORDER_ALIASES.get(b.strip(), b.strip())
            if a in REGION and b in REGION and a != b and tuple(sorted((a, b))) not in OVERSEAS:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def main():
    ranks = pd.read_csv(PROC / "esc_juror_ranks.csv")
    nat = pd.read_csv(PROC / "esc_national.csv", keep_default_na=False)
    unknown = sorted((set(ranks.voter) | set(ranks.song_country)) - (set(REGION) & set(FINE_GROUP)))
    if unknown:
        raise SystemExit(f"countries without a region: {unknown}")
    bset = borders()
    pd.DataFrame([{"country": c, "region": r, "bloc": GROUP[c], "fine": FINE_GROUP.get(c, "")}
                  for c, r in REGION.items()]).to_csv(
        PROC / "esc_regions.csv", index=False)
    pd.DataFrame(sorted(bset), columns=["a", "b"]).to_csv(PROC / "esc_borders.csv", index=False)
    # songs in each final: every song a national table lists (own song included there, unranked)
    songs = nat.groupby("year").song_country.apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    ballot = 0
    for (year, voter, juror), g in ranks.groupby(["year", "voter", "juror"], sort=True):
        rk = dict(zip(g.song_country, g["rank"]))
        for s in songs[year]:
            if s == voter:
                continue                      # own song: not in the juror's choice set
            rows.append({"ballot": ballot, "year": year, "contest": f"ESC{year}", "role": "esc_jury",
                         "voter_country": voter, "voter_region": REGION[voter], "juror": juror,
                         "person": s, "nominee_country": s, "nominee_region": REGION[s],
                         "rank": rk.get(s, 0),
                         "voter_bloc": GROUP[voter], "nominee_bloc": GROUP[s],
                         "same_region": int(REGION[voter] == REGION[s]),
                         "same_bloc": int(voter in BLOC and BLOC[voter] == BLOC.get(s)),
                         "voter_fine": FINE_GROUP[voter], "nominee_fine": FINE_GROUP[s],
                         "same_fine": int(FINE_GROUP[voter] == FINE_GROUP[s]),
                         "same_border": int(tuple(sorted((voter, s))) in bset)})
        ballot += 1
    long = pd.DataFrame(rows)
    long.to_csv(PROC / "esc_long.csv", index=False)
    print(f"esc_long: {long.ballot.nunique()} juror ballots, {long.contest.nunique()} finals, "
          f"same_region {long.same_region.mean():.3f}, same_bloc {long.same_bloc.mean():.3f}, "
          f"same_fine {long.same_fine.mean():.3f}, "
          f"same_border {long.same_border.mean():.3f}, "
          f"border pairs {len(bset)}", file=sys.stderr)


if __name__ == "__main__":
    main()
