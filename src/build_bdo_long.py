"""Ballon d'Or journalist ballots in the analysis format of fifa_long.csv.

python -I nhb/src/build_bdo_long.py  ->  nhb/data/processed/bdo_long.csv

One row per juror x nominee (the 30 nominees of each contest), with rank 1..k for the juror's
picks and 0 otherwise. Contests are labelled "BdO{year}_{award}", the role is "bdo" (journalists
of the France Football jury) and ballot ids start at 1,000,000 so that they never collide with
the FIFA ballots. Affinities are defined as for the FIFA ballots (same nation, same confederation,
voter's country among the nominee's club countries).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OFFSET = 1_000_000


def main():
    b = pd.read_csv(PROC / "bdo_ballots.csv", keep_default_na=False)
    meta = pd.read_csv(PROC / "bdo_nominee_meta.csv", keep_default_na=False)
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    b = b.reset_index(drop=True)
    b["ballot"] = OFFSET + b.index
    b["contest"] = "BdO" + b.year.astype(str) + "_" + b.award
    b["role"] = "bdo"
    b["voter_country"] = b.country
    b["voter_confed"] = b.country.map(confed)
    b["voter"] = b.juror
    picks = [f"pick{i}" for i in range(1, 11)]
    noms = meta.rename(columns={"nation": "nominee_country", "club_countries": "club_country_set"})
    noms["nominee_confed"] = noms.nominee_country.map(confed)
    long = b[["ballot", "year", "award", "contest", "role", "voter_country", "voter_confed", "voter"] + picks] \
        .merge(noms[["year", "award", "person", "nominee_country", "nominee_confed", "club_country_set", "clubs"]],
               on=["year", "award"])
    long["rank"] = 0
    for i, c in enumerate(picks, 1):
        long.loc[long.person == long[c], "rank"] = i
    lost = b.set_index("ballot")[picks].apply(lambda r: (r != "").sum(), axis=1) - \
        long[long["rank"] > 0].groupby("ballot").size().reindex(b.ballot).fillna(0)
    if (lost != 0).any():
        print(f"picks not among the nominees: {int(lost.abs().sum())}", file=sys.stderr)
        return 1
    long["same_nation"] = (long.voter_country == long.nominee_country).astype(int)
    long["same_confed"] = (long.voter_confed == long.nominee_confed).astype(int)
    long["same_league"] = [int(v in s.split(";")) for v, s in zip(long.voter_country, long.club_country_set)]
    long = long.drop(columns=picks)
    assert long.voter_confed.notna().all() and long.nominee_confed.notna().all()
    long.to_csv(PROC / "bdo_long.csv", index=False)
    print(f"{b.ballot.nunique()} ballots, {len(long)} rows, {long.contest.nunique()} contests; affinity shares:\n"
          + long[["same_nation", "same_confed", "same_league"]].mean().round(3).to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
