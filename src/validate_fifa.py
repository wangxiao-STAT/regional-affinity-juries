"""Check the parsed FIFA ballots against what the PDFs themselves report.

python -I nhb/src/validate_fifa.py  ->  prints the checks; nhb/data/processed/fifa_validation.csv

1. Official per-group points of the top three (printed for 2023 and 2024) against 5/3/1 points
   recomputed from the parsed ballots.
2. 2010-2015 overall vote percentages (Wikipedia nominee tables, all nominees) against the shares
   recomputed with the three juror groups weighted equally.
3. One ballot per (contest, role, country): duplicates are listed.
4. No voter picks the same player twice; empty and placeholder picks are counted.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
POINTS = {"pick1": 5, "pick2": 3, "pick3": 1}
PLACEHOLDERS = {"", "INVALID", "NOT_COUNTED", "NA", "NO_VOTE"}


def points(b):
    long = b.melt(id_vars=["year", "award", "role", "country"], value_vars=list(POINTS), var_name="slot",
                  value_name="name")
    long = long[~long.name.isin(PLACEHOLDERS)]
    long["pts"] = long.slot.map(POINTS)
    return long.groupby(["year", "award", "role", "name"]).pts.sum()


def main():
    b = pd.read_csv(PROC / "fifa_ballots_raw.csv", keep_default_na=False)
    off = pd.read_csv(PROC / "fifa_official_top3.csv", keep_default_na=False)
    pts = points(b)
    rows = []
    for _, o in off.iterrows():
        for role in ("coach", "captain", "media"):
            got = pts.get((o.year, o.award, role, o["name"]), 0)
            rows.append({"year": o.year, "award": o.award, "rank": o["rank"], "name": o["name"], "role": role,
                         "official": o[role], "parsed": int(got), "diff": int(got) - o[role]})
    v = pd.DataFrame(rows)
    v.to_csv(PROC / "fifa_validation.csv", index=False)
    print(v.to_string(index=False))
    print(f"official top-3 cells matched exactly: {(v['diff'] == 0).sum()} of {len(v)}")

    meta = pd.read_csv(PROC / "fifa_nominee_meta.csv", keep_default_na=False)
    names = pd.read_csv(PROC / "fifa_nominee_names.csv", keep_default_na=False)
    person = names.set_index(["year", "award", "raw"]).person
    by_person = pts.rename(lambda n: n, level=3).reset_index()
    by_person["person"] = [person.get((y, a, n), n) for y, a, n in by_person[["year", "award", "name"]].itertuples(index=False)]
    g = by_person.groupby(["year", "award", "role", "person"]).pts.sum()
    share = 100 * (g / g.groupby(level=[0, 1, 2]).transform("sum")).groupby(level=[0, 1, 3]).sum() / 3
    mm = meta[meta.year <= 2015].copy()
    mm["wiki_pct"] = pd.to_numeric(mm.wiki_score.str.replace("%", "").str.strip(), errors="coerce")
    mm = mm.dropna(subset=["wiki_pct"])
    mm["parsed"] = [share.get((y, a, p), 0.0) for y, a, p in mm[["year", "award", "person"]].itertuples(index=False)]
    mm["diff"] = mm.parsed - mm.wiki_pct
    mm[["year", "award", "person", "wiki_pct", "parsed", "diff"]].to_csv(PROC / "fifa_validation_shares.csv", index=False)
    print("\n2010-15 vote shares, max |parsed - published| (percentage points) by contest:")
    print(mm.groupby(["year", "award"])["diff"].apply(lambda d: d.abs().max()).round(2).unstack().to_string())

    dup = b[b.duplicated(["year", "award", "role", "country"], keep=False)]
    print(f"\nduplicate (contest, role, country): {len(dup)} rows")
    if len(dup):
        print(dup[["year", "award", "role", "country", "voter", "line"]].to_string(index=False))

    picks = b[list(POINTS)]
    real = picks.where(~picks.isin(PLACEHOLDERS))
    repeat = real.apply(lambda r: r.dropna().duplicated().any(), axis=1)
    print(f"\nballots repeating a player: {int(repeat.sum())}")
    if repeat.any():
        print(b[repeat].to_string(index=False))
    for code in sorted(PLACEHOLDERS):
        print(f"cells '{code}': {int((picks == code).sum().sum())}")
    return 0 if (v["diff"] == 0).all() else 1


if __name__ == "__main__":
    sys.exit(main())
