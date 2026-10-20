"""Cross-jury simulations: the same nominee field judged by different juries.

python -I nhb/src/cross_jury.py [--model C1_role] [--reps 400] [--draws 40]
    ->  nhb/results/design/cross_jury.csv

Nominee appeal (alpha, per contest) and the affinity coefficients of the four juror types come
from the combined fit (fit_combined.py). For each contest the field is judged by:
  actual        the contest's own jury
  fifa_jury     (Ballon d'Or contests) the FIFA jury of the same year and award: captains,
                coaches and journalists of every member, top three, FIFA's equal-weight rule
  bdo_jury      (FIFA contests) the Ballon d'Or jury of the same year and award, its ballot depth
                and points
  global_press  one journalist per country that sent a FIFA media ballot that year, with the
                ballot depth and points of the contest's own award (Ballon d'Or journalists'
                coefficients for Ballon d'Or contests, FIFA media coefficients otherwise)
Each replication draws Gumbel noise once and ranks with and without the affinity terms; the
share of replications with different winners is the probability that similarity bias decides the
award under that jury. Closeness is varied as in design_sim.py (runner-up's appeal set to the
leader's minus d; "obs" keeps the fitted appeals). Juries with fewer than 20 voters are skipped.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_combined  # noqa: E402
from fit_pooled import hybrid_draws  # noqa: E402
from counterfactuals import BDO_TOP5, BDO_TOP10, POINTS  # noqa: E402
from fifa_rules import jury_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
GAPS = ["obs", 0.0, 0.1, 0.2, 0.4]
AFF = ("same_nation", "same_confed", "same_league")


def affinity(voters, noms, confed):
    """(B, J, 3) affinity indicators for arbitrary voters and nominees."""
    vc = voters.voter_country.to_numpy()[:, None]
    vf = np.array([confed.get(v, "") for v in voters.voter_country])[:, None]
    nat = noms.nominee_country.to_numpy()[None, :]
    nconf = noms.nominee_confed.to_numpy()[None, :]
    clubs = [set(s.split(";")) for s in noms.club_country_set]
    lg = np.array([[v in c for c in clubs] for v in voters.voter_country])
    return np.stack([vc == nat, vf == nconf, lg], axis=2).astype(float)


def level(role, country, column, confed):
    """The voter's level of the variable that splits a coefficient (fit_pooled.prepare columns)."""
    c = confed.get(country, "")
    return {"role": role, "voter_confed": c, "role_confed": f"{role}:{c}",
            "org_confed": f"{'bdo' if role == 'bdo' else 'fifa'}:{c}"}[column]


def bonus(voters, noms, confed, beta_by, split):
    """Affinity utility of each voter for each nominee, with the coefficients of the voter's level of
    each term's split (role, or confederation for the confederation-specific regional bonus); a level
    absent from the fit (no such jurors) gets no bonus."""
    A = affinity(voters, noms, confed)
    col = lambda a: split.get(a) if isinstance(split, dict) else split
    B = np.array([[beta_by.get((a, level(r, v, col(a), confed)), 0.0) for a in AFF]
                  for r, v in zip(voters.role, voters.voter_country)])
    return np.einsum("bjk,bk->bj", A, B)


def winners(U, role, points, rule):
    S, Bn, J = U.shape
    k = len(points)
    top = np.argpartition(-U, k - 1, axis=2)[:, :, :k]
    order = np.argsort(-np.take_along_axis(U, top, axis=2), axis=2)
    top = np.take_along_axis(top, order, axis=2)
    pts = np.zeros((S, Bn, J))
    np.put_along_axis(pts, top, np.broadcast_to(points, (S, Bn, k)), axis=2)
    if rule == "points":
        return pts.sum(axis=1).argmax(axis=1)
    if rule == "scoring":
        cap = role == "captain"
        return jury_score(pts, role, "scoring", (pts[:, cap, :] == points[0]).sum(axis=1)).argmax(axis=1)
    share = np.zeros((S, J))
    groups = list(np.unique(role))
    for r in groups:
        g = pts[:, role == r, :].sum(axis=1)
        share += g / g.sum(axis=1, keepdims=True) / len(groups)
    return share.argmax(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C1_role")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--contests", nargs="*")
    ap.add_argument("--res", default=str(RES / "combined"))
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()
    out = RES / "design"
    out.mkdir(parents=True, exist_ok=True)
    long, X, names, fe_sets = fit_combined.build(a.model)
    assert all(re.fullmatch(r"same_\w+\[[^\]]+\]", n) for n in names), names
    split = fit_combined.CSPECS[a.model][0]
    theta = pd.read_csv(Path(a.res) / f"theta_{a.model}.csv", keep_default_na=False)
    cov = np.load(Path(a.res) / f"cov_{a.model}.npz")
    est = theta[theta.kind != "ref"]
    n_alpha = int((est.kind == "alpha").sum())
    assert list(est.param[n_alpha:]) == names
    alpha_col = pd.Series(np.arange(n_alpha), index=est.param[:n_alpha].to_numpy())
    alpha_col = pd.concat([alpha_col, pd.Series(n_alpha, index=theta.param[theta.kind == "ref"].to_numpy())])
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    rng = np.random.default_rng(a.seed)
    mle = est.value.to_numpy()
    draws = hybrid_draws(mle, cov["V"], cov["V_cluster"], n_alpha, a.draws, rng)
    per = max(1, a.reps // a.draws)
    voters_all = long.drop_duplicates("ballot")[["ballot", "contest", "year", "award", "role", "voter_country"]]

    rows = []
    for contest in a.contests or sorted(long.contest.unique()):
        d = long[long.contest == contest]
        year, award, is_bdo = int(d.year.iloc[0]), d.award.iloc[0], contest.startswith("BdO")
        noms = d.drop_duplicates("person").set_index("person")[["nominee_country", "nominee_confed", "club_country_set"]]
        fe_lab = (contest + "#" + noms.index.to_series()).to_numpy()
        a_idx = pd.Series(fe_lab).map(alpha_col).fillna(n_alpha + 1).astype(int).to_numpy()
        own_points = (BDO_TOP5 if year <= 2023 else BDO_TOP10) if is_bdo else POINTS
        own = voters_all[voters_all.contest == contest]
        fifa = voters_all[voters_all.contest == f"{year}_{award}"]
        bdo = voters_all[voters_all.contest == f"BdO{year}_{award}"]
        press_role = "bdo" if is_bdo else "media"
        press = fifa[fifa.role == "media"].assign(role=press_role)
        fifa_rule = "scoring" if year >= 2023 else "fifa"      # FIFA's rule of that year (fifa_rules.py)
        juries = {"actual": (own, own_points, "points" if is_bdo else fifa_rule)}
        if is_bdo and len(fifa):
            juries["fifa_jury"] = (fifa, POINTS, fifa_rule)
        if not is_bdo and len(bdo):
            juries["bdo_jury"] = (bdo, BDO_TOP5 if year <= 2023 else BDO_TOP10, "points")
        if len(press):
            juries["global_press"] = (press, own_points, "points")
        for gap in GAPS:
            for jname, (jv, pts, rule) in juries.items():
                if len(jv) < 20:
                    continue
                flips = []
                for th in draws:
                    alpha = np.append(th[:n_alpha], [0.0, -np.inf])[a_idx]
                    if gap != "obs":
                        o = np.argsort(-alpha)
                        alpha = alpha.copy()
                        alpha[o[1]] = alpha[o[0]] - gap
                    beta_by = {(n.split("[")[0], n.split("[")[1][:-1]): v for n, v in zip(names, th[n_alpha:])}
                    bon = bonus(jv, noms.reset_index(), confed, beta_by, split)
                    G = rng.gumbel(size=(per, len(jv), len(noms)))
                    base = np.where(np.isfinite(alpha), alpha, -np.inf)[None, None, :]
                    wf = winners(base + G, jv.role.to_numpy(), pts, rule)
                    wb = winners(base + bon[None] + G, jv.role.to_numpy(), pts, rule)
                    flips.append(wf != wb)
                f = np.concatenate(flips)
                rows.append({"contest": contest, "jury": jname, "gap": gap, "voters": len(jv), "p_flip": f.mean(),
                             "reps": len(f)})
        print(contest, flush=True, file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"cross_jury_{a.model}.csv", index=False)
    print(df[df.gap.astype(str) == "obs"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
