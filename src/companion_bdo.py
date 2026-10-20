"""Convert the companion study's Ballon d'Or ballots (men 2019 and 2024) to the schema of parse_bdo.py.

python -I nhb/src/companion_bdo.py  ->  nhb/data/processed/bdo_ballots_companion.csv, bdo_companion_issues.csv

Input (read only, never modified): the companion study's model_input/{2019,2024}/, next to nhb/
- rankings_long.csv: voter (the juror's country; French names in 2019, English in 2024), rank, player
  (a short key such as "VanDijk"), points (6-4-3-2-1 in 2019; 15-12-...-1 in 2024);
- voter_source.csv: voter, continent, source (2019: "real (France Football 04.12.2019)", the print
  issue; 2024: "France Football 2024 (99 FIFA-ranked voters)");
- model_players.csv: player key -> full_name (30 nominees per year).
Neither year names the juror or the outlet, so those columns stay blank. Each pick is written as the
companion-study full name of the player key; bdo_nominees.py maps it to a person like any other spelling. Checks:
ranks 1..k once per voter, printed points equal to the scheme's points for the rank, the same voters
in both files, every player key in model_players.csv. Failures go to the issues file, never fixed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from countries_fr import resolve_fr  # noqa: E402
from parse_bdo import COLUMNS, K_OF_YEAR, KMAX, SCHEME, canon_resolver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPANION = ROOT.parent / "model_input"
PROC = ROOT / "data" / "processed"
YEARS = (2019, 2024)


def convert(year, resolve):
    d = COMPANION / str(year)
    long = pd.read_csv(d / "rankings_long.csv", keep_default_na=False)
    voters = pd.read_csv(d / "voter_source.csv", keep_default_na=False)
    players = pd.read_csv(d / "model_players.csv", keep_default_na=False)
    k = K_OF_YEAR[year]
    full = dict(zip(players.player, players.full_name))
    src_of = dict(zip(voters.voter, voters.source))
    rows, issues = [], []

    def issue(voter, reason):
        issues.append({"year": year, "award": "MP", "voter": voter, "reason": reason})

    for v in sorted(set(voters.voter) - set(long.voter)):
        issue(v, "in voter_source.csv but casts no ballot")
    for voter, g in long.groupby("voter", sort=False):
        g = g.sort_values("rank")
        row = {"year": year, "award": "MP", "country_raw": voter,
               "country": resolve_fr(voter) or resolve(voter) or "", "juror": "", "outlet": "", "k": k,
               "source": f"companion:model_input/{year}/rankings_long.csv ({src_of.get(voter, 'no voter_source row')})",
               "flag": ""}
        for i in range(1, KMAX + 1):
            row[f"pick{i}"] = ""
        if list(g["rank"]) != list(range(1, k + 1)):
            issue(voter, f"ranks {list(g['rank'])} are not 1..{k}")
            row["flag"] = "incomplete_ranks"
        for r in g.itertuples(index=False):
            if r.player not in full:
                issue(voter, f"player key {r.player} not in model_players.csv")
                continue
            if 1 <= r.rank <= k:
                if int(r.points) != SCHEME[k][r.rank - 1]:
                    issue(voter, f"rank {r.rank} carries {r.points} points, scheme gives {SCHEME[k][r.rank - 1]}")
                row[f"pick{r.rank}"] = full[r.player]
        if voter not in src_of:
            issue(voter, "no row in voter_source.csv")
        if not row["country"]:
            issue(voter, "unresolved country")
        rows.append(row)
    return rows, issues


def main():
    resolve = canon_resolver()
    rows, issues = [], []
    for year in YEARS:
        r, i = convert(year, resolve)
        rows += r
        issues += i
        print(f"{year}: {len(r)} voters, {len(i)} issues", file=sys.stderr)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(PROC / "bdo_ballots_companion.csv", index=False)
    pd.DataFrame(issues, columns=["year", "award", "voter", "reason"]).to_csv(PROC / "bdo_companion_issues.csv",
                                                                               index=False)
    for i in issues:
        print(f"  {i}", file=sys.stderr)


if __name__ == "__main__":
    main()
