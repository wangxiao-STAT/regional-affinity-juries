"""Which rule turns the jurors' rankings into each national jury's ranking? Checked against the published
national jury ranks and points.

python -I nhb/src/validate_eurovision.py  ->  nhb/data/processed/esc_rule_fit.csv, esc_validation.csv

2016-2017 (EBU: the jurors' rankings are averaged): national rank = order of the sum of ranks.
From 2018 (EBU: "exponential weight model"; the score values are not published): each juror rank r
scores base^(r-1), the scores are summed, and the order of the sums is the national rank. One base for
all years from 2018: the value on a grid (step 0.0001 between 0.80 and 0.86) that reproduces the most
published national rankings, pooled over the years; esc_rule_fit.csv reports every candidate. A national jury whose ranks are not all published (juries replaced
by the EBU, missing tables) is listed in esc_validation.csv and kept out of the juror-level data.
Points: the national top ten receive 12, 10, 8, 7, 6, 5, 4, 3, 2, 1 (ineligible songs skipped).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
POINTS = [12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
BASES = np.round(np.concatenate([np.arange(0.50, 0.80, 0.005), np.arange(0.80, 0.86, 0.0001),
                                  np.arange(0.86, 0.995, 0.005)]), 4)
LINEAR_YEARS = (2016, 2017)
# songs ranked by the juries but excluded from the points: the Netherlands, disqualified in 2024 after
# the jury show; the songs ranked below it moved up one place in each national ranking
INELIGIBLE = {2024: {"Netherlands"}}


def national_order(ranks, base):
    """ranks: (jurors, songs) array of ranks 1..n; returns the songs' national ranks (1 = best)."""
    score = -ranks.sum(axis=0) if base is None else (base ** (ranks - 1.0)).sum(axis=0)
    # ties: the better best single-juror rank, then the running order (stable sort keeps the input order)
    best = ranks.min(axis=0)
    order = np.lexsort((best, -score))
    out = np.empty(len(order), int)
    out[order] = np.arange(1, len(order) + 1)
    return out


def juries():
    r = pd.read_csv(PROC / "esc_juror_ranks.csv")
    nat = pd.read_csv(PROC / "esc_national.csv", keep_default_na=False)
    nat["jury_rank"] = pd.to_numeric(nat.jury_rank, errors="coerce")
    nat["jury_points"] = pd.to_numeric(nat.jury_points, errors="coerce").fillna(0)
    out = []
    for (y, v), g in r.groupby(["year", "voter"]):
        w = g.pivot_table(index="juror", columns="song_country", values="rank")
        n = nat[(nat.year == y) & (nat.voter == v) & nat.jury_rank.notna()].set_index("song_country")
        complete = w.notna().all().all() and set(w.columns) == set(n.index) and len(w.columns) > 0
        out.append((y, v, w, n, complete))
    return out


def main():
    js = juries()
    fits = []
    for base in [None] + list(BASES):
        for y in sorted({j[0] for j in js}):
            ok = tot = 0
            for yy, v, w, n, complete in js:
                if yy != y or not complete:
                    continue
                pred = national_order(w.to_numpy(float), base)
                tot += 1
                ok += int((pred == n.loc[w.columns, "jury_rank"].to_numpy()).all())
            fits.append({"year": y, "rule": "sum of ranks" if base is None else f"exponential {base}",
                         "base": base, "juries": tot, "reproduced": ok})
    fit = pd.DataFrame(fits)
    fit.to_csv(PROC / "esc_rule_fit.csv", index=False)
    # one base for every year from 2018: the value reproducing the most national rankings, pooled
    exp_years = fit[fit.base.notna() & ~fit.year.isin(LINEAR_YEARS)]
    pooled = exp_years.groupby("base").reproduced.sum()
    b_star = float(pooled.idxmax())
    best = {y: (None if y in LINEAR_YEARS else b_star) for y in fit.year.unique()}
    for y in sorted(best):
        g = fit[(fit.year == y) & (fit.base.isna() if best[y] is None else (fit.base == b_star))].iloc[0]
        print(f"{y}: {g.rule}: {g.reproduced} of {g.juries} national rankings reproduced", file=sys.stderr)
    val = []
    for y, v, w, n, complete in js:
        row = {"year": y, "voter": v, "jurors": w.shape[0], "songs_ranked": w.shape[1], "complete": complete}
        if complete:
            base = None if pd.isna(best[y]) else best[y]
            pred = national_order(w.to_numpy(float), base)
            out = np.isin(w.columns, list(INELIGIBLE.get(y, ())))
            eff = pred - np.array([(out & (pred < p)).sum() for p in pred])   # places after removing them
            pts = np.array([0 if o else (POINTS[k - 1] if k <= len(POINTS) else 0) for k, o in zip(eff, out)])
            row.update({"ranks_match": bool((pred == n.loc[w.columns, "jury_rank"].to_numpy()).all()),
                        "points_match": bool((pts == n.loc[w.columns, "jury_points"].to_numpy()).all()),
                        "base": base})
        val.append(row)
    pd.DataFrame(val).to_csv(PROC / "esc_validation.csv", index=False)


if __name__ == "__main__":
    main()
