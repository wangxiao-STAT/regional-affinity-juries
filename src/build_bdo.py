"""One row per Ballon d'Or juror, all contests, with picks mapped to persons.

python -I nhb/src/build_bdo.py  ->  nhb/data/processed/bdo_ballots.csv

Stacks bdo_ballots_jasa.csv (men 2019 and 2024, JASA data) and bdo_ballots_raw.csv (men 2022, 2023 and
2025; women 2023-2025) and replaces each printed pick by its person from bdo_nominee_names.csv.
Columns: year, award (MP/WP), country_raw (as printed), country (canonical), juror, outlet (blank where
the source names none: men 2019, 2023, 2024), k (5 up to 2023, 10 from 2024), pick1..pick10 (blank
beyond k), source, flag (parser notes; see data/README.md). A pick without a person stops the build.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_bdo import COLUMNS, KMAX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PICKS = [f"pick{i}" for i in range(1, KMAX + 1)]


def main():
    read = lambda f: pd.read_csv(PROC / f, keep_default_na=False, dtype={c: str for c in PICKS})
    b = pd.concat([read("bdo_ballots_jasa.csv"), read("bdo_ballots_raw.csv")], ignore_index=True)
    names = pd.read_csv(PROC / "bdo_nominee_names.csv", keep_default_na=False)
    person = {(y, a, r): p for y, a, r, p in zip(names.year, names.award, names.raw, names.person)}
    missing = set()
    for c in PICKS:
        new = []
        for y, a, r in zip(b.year, b.award, b[c]):
            p = person.get((y, a, r), "") if r else ""
            if r and not p:
                missing.add((y, a, r))
            new.append(p)
        b[c] = new
    if missing:
        print(f"picks without a person: {sorted(missing)}", file=sys.stderr)
        return 1
    b = b[COLUMNS].sort_values(["year", "award", "country"], kind="stable")
    b.to_csv(PROC / "bdo_ballots.csv", index=False)
    print(b.groupby(["year", "award"]).agg(jurors=("country", "size"), k=("k", "first"),
                                            named=("juror", lambda s: (s != "").sum())).to_string(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
