"""Assemble the analysis tables for the FIFA ballots.

python -I nhb/src/build_fifa.py  ->  nhb/data/processed/fifa_ballots.csv (one row per ballot)
                                     nhb/data/processed/fifa_long.csv (one row per ballot x nominee)

Inputs: fifa_ballots_raw.csv (parse_fifa.py), fifa_nominee_names.csv (nominees.py),
fifa_nominee_meta.csv (nominee_meta.py), countries.csv and country_aliases.csv (countries.py).

Choice set of a ballot: the contest's nominees that received at least one pick (a nominee never
picked has an infinitely negative fixed effect and drops out of every choice probability), minus
the voter when a captain is a nominee (captains may not vote for themselves), minus the voter's
compatriots on 2010 captain and coach ballots (FIFA's 2010 rules voided votes for players of the
voter's own national team: captains and coaches picked none of their 55 compatriot nominees that
year, against 40-60 per cent later, and the voided cells of e.g. the Dutch captain and coach are
consistent with that rule; journalists were allowed and did pick compatriots). Placeholder picks
(invalid, not counted, no vote, blank) are dropped and the valid picks keep their order; such
ballots carry `incomplete = True`. FIFA's test record (flag test_entry) is excluded.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nominees import key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PLACEHOLDERS = {"", "INVALID", "NOT_COUNTED", "NA", "NO_VOTE"}


def main():
    b = pd.read_csv(PROC / "fifa_ballots_raw.csv", keep_default_na=False)
    names = pd.read_csv(PROC / "fifa_nominee_names.csv", keep_default_na=False)
    meta = pd.read_csv(PROC / "fifa_nominee_meta.csv", keep_default_na=False)
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False)
    alias = pd.read_csv(PROC / "country_aliases.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    canon = {(k, r): c for k, r, c in alias[["kind", "raw", "country"]].itertuples(index=False)}

    b = b[~b.flag.str.contains("test_entry")].reset_index(drop=True)
    b.insert(0, "ballot", range(len(b)))
    b["contest"] = b.year.astype(str) + "_" + b.award
    b["voter_country"] = [canon[("ballot", c)] for c in b.country]
    b["voter_confed"] = b.voter_country.map(confed)
    person = {(y, a, r): p for y, a, r, p in names[["year", "award", "raw", "person"]].itertuples(index=False)}
    for k in (1, 2, 3):
        b[f"pick{k}"] = [person.get((y, a, r), r) for y, a, r in b[["year", "award", f"pick{k}"]].itertuples(index=False)]
    b["incomplete"] = b[["pick1", "pick2", "pick3"]].isin(PLACEHOLDERS).any(axis=1)

    meta["nominee_country"] = [canon[("nation", n)] for n in meta.nation]
    meta["nominee_confed"] = meta.nominee_country.map(confed)
    meta["club_country_set"] = [";".join(sorted({canon[("club", c)] for c in x.split(" | ") if c}))
                                for x in meta.club_countries]
    noms = meta[["year", "award", "person", "nominee_country", "nominee_confed", "club_country_set", "clubs"]]
    assert not noms.duplicated(["year", "award", "person"]).any()

    long = b[["ballot", "year", "award", "contest", "role", "voter_country", "voter_confed", "voter",
              "pick1", "pick2", "pick3"]].merge(noms, on=["year", "award"])
    vk, pk = long.voter.map(key), long.person.map(key)
    named = pd.Series([set(p.split()) <= set(v.split()) for v, p in zip(vk, pk)], index=long.index)
    long["self"] = (long.role == "captain") & (long.voter_country == long.nominee_country) & named
    print("captains removed from their own choice set:\n"
          + long[long.self][["contest", "voter_country", "voter", "person"]].to_string(index=False), file=sys.stderr)
    barred = (long.year == 2010) & long.role.isin(["captain", "coach"]) & (long.voter_country == long.nominee_country)
    print(f"compatriots removed from 2010 captain and coach choice sets: {int(barred.sum())} rows, "
          f"{int((barred & (long[['pick1', 'pick2', 'pick3']].eq(long.person, axis=0).any(axis=1))).sum())} of them picked",
          file=sys.stderr)
    long = long[~(long.self | barred)].drop(columns="self")
    long["rank"] = 0
    for k in (3, 2, 1):
        long.loc[long.person == long[f"pick{k}"], "rank"] = k
    # renumber valid picks 1..m when a placeholder sits between them
    valid = long[long["rank"] > 0].groupby("ballot")["rank"].rank(method="first").astype(int)
    long.loc[valid.index, "rank"] = valid
    long["same_nation"] = (long.voter_country == long.nominee_country).astype(int)
    long["same_confed"] = (long.voter_confed == long.nominee_confed).astype(int)
    long["same_league"] = [int(v in s.split(";")) for v, s in zip(long.voter_country, long.club_country_set)]
    long = long.drop(columns=["pick1", "pick2", "pick3"])

    picked = long[long["rank"] > 0].groupby("ballot").size()
    b["n_valid"] = b.ballot.map(picked).fillna(0).astype(int)
    lost = b[b.n_valid != 3 - b[["pick1", "pick2", "pick3"]].isin(PLACEHOLDERS).sum(axis=1)]
    if len(lost):
        print("picks lost in the merge:\n" + lost.to_string(), file=sys.stderr)
        return 1
    b.to_csv(PROC / "fifa_ballots.csv", index=False)
    long.to_csv(PROC / "fifa_long.csv", index=False)
    print(f"{len(b)} ballots ({int(b.incomplete.sum())} incomplete, {int((b.n_valid == 0).sum())} empty), "
          f"{len(long)} ballot-nominee rows, {long.contest.nunique()} contests", file=sys.stderr)
    by = long.groupby("role")[["same_nation", "same_confed", "same_league"]].mean()
    print("share of ballot-nominee pairs with each affinity:\n" + by.round(3).to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
