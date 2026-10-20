"""Jury-design simulations: when does similarity bias change the winner?

python -I nhb/src/design_sim.py [--model M1_role] [--reps 400] [--draws 40]
    ->  nhb/results/design/design_contests_<model>.csv, design_summary_<model>.csv

For each FIFA contest, nominee appeal (alpha) and the affinity coefficients come from a pooled fit
(fit_pooled.py). A jury is a list of voters (role, country); every voter ranks three nominees by
utility alpha_j + x_ij'beta_role + Gumbel noise. The same noise is used with the affinity terms
set to zero (a bias-free jury), so the share of replications whose winners differ is the
probability that similarity bias decides the award under that design. Designs:
  fifa          the contest's actual voters, FIFA's rule of that year (fifa_rules.py: group shares
                to 2022, art. 11 scoring points from 2023), fans not included
  fifa_nocompat the same, but no voter may rank a compatriot (the bias-free jury keeps them)
  media         the contest's journalists only (one group: total points order nominees as the rule does)
  size_n        n voters drawn at random from the actual voters in each replication, FIFA's rule of the year
  uefa_s        100 voters with a share s from UEFA countries, drawn from the actual voters, rule of the year
  equal_confed  the same number of voters from each of the six confederations (EQUAL_K, or fewer when a
                confederation has fewer voters, not below EQUAL_K_MIN), rule of the year
  contenders_equal  the contest's voters with the larger of the two leading nominees' confederations
                subsampled to the size of the smaller; only parameter draws whose leading nominees come
                from different confederations
Each row also records the contenders' bloc difference (bloc_gap), the modal pair of the contenders'
confederations and the share of draws in which they differ (share_cross); --rule share / scoring_zero
repeat FIFA's 2023+ contests under the 2010-2022 rule or with zero-point nominees scoring nothing.
Closeness: the runner-up's appeal is moved to alpha_winner - d for a grid of d, which sets how
close the bias-free race is; d = "obs" keeps the fitted appeals. Parameter uncertainty enters
through hybrid draws (fit_pooled.hybrid_draws); each draw gets reps / draws replications.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pooled import SPECS, design, hybrid_draws, prepare, psd_sqrt  # noqa: E402
from fifa_rules import jury_score, rule_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
POINTS = np.array([5.0, 3.0, 1.0])
ROLES = ("captain", "coach", "media")
GAPS = ["obs", 0.0, 0.1, 0.2, 0.4, 0.8]
SIZES = [25, 50, 100, 200]
UEFA_SHARES = [0.1, 0.3, 0.5, 0.7, 0.9]
UEFA_JURY = 100
EQUAL_K = 16           # jurors per confederation in the equal-confederation design (6 x 16 = 96)
EQUAL_K_MIN = 8        # skipped when a confederation has fewer jurors in the contest
CONFEDS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")


def points_by_role(U, role, k=3, with_pts=False):
    """U: (S, B, J) utilities; returns (S, J) points per role-weighted share and per total
    (and the (S, B, J) points by ballot with with_pts)."""
    S, B, J = U.shape
    top = np.argpartition(-U, k - 1, axis=2)[:, :, :k]
    order = np.argsort(-np.take_along_axis(U, top, axis=2), axis=2)
    top = np.take_along_axis(top, order, axis=2)
    pts = np.zeros((S, B, J))
    np.put_along_axis(pts, top, np.broadcast_to(POINTS[:k], (S, B, k)), axis=2)
    total = pts.sum(axis=1)
    share = np.zeros((S, J))
    groups = [r for r in ROLES if (role == r).any()]
    for r in groups:
        g = pts[:, role == r, :].sum(axis=1)
        share += g / g.sum(axis=1, keepdims=True) / len(groups)
    return (share, total, pts) if with_pts else (share, total)


def flips(U_bias, U_free, role, rule):
    """rule: "fifa" (shares, FIFA 2010-2022), "scoring" (FIFA from 2023, art. 11-12) or "points"."""
    sb, tb, pb = points_by_role(U_bias, role, with_pts=True)
    sf, tf, pf = points_by_role(U_free, role, with_pts=True)
    if rule in ("scoring", "scoring_zero"):
        cap, z = role == "captain", rule == "scoring_zero"
        a = jury_score(pb, role, "scoring", (pb[:, cap, :] == POINTS[0]).sum(axis=1), zero_none=z)
        b = jury_score(pf, role, "scoring", (pf[:, cap, :] == POINTS[0]).sum(axis=1), zero_none=z)
    else:
        a, b = (sb, sf) if rule == "fifa" else (tb, tf)
    return a.argmax(axis=1) != b.argmax(axis=1), a, b


def contest_inputs(long, X, names, fe_col, contest):
    d = long[long.contest == contest]
    ballots = np.sort(d.ballot.unique())
    noms = np.sort(d.person.unique())
    bi = np.searchsorted(ballots, d.ballot.to_numpy())
    ji = np.searchsorted(noms, d.person.to_numpy())
    idx = d.index.to_numpy()
    voters = d.groupby("ballot")[["role", "voter_confed", "voter_country"]].first().reindex(ballots)
    return dict(B=len(ballots), J=len(noms), bi=bi, ji=ji, idx=idx, noms=noms, role=voters.role.to_numpy(),
                confed=voters.voter_confed.to_numpy())


def utilities(ci, X, fe_col, th, n_alpha, gap, avail_mask=None):
    alpha = np.append(th[:n_alpha], [0.0, -np.inf])[fe_col[ci["idx"]]]
    a = np.full(ci["J"], np.nan)
    a[ci["ji"]] = alpha                       # alpha does not vary across voters in these models
    order = np.argsort(-a)
    if gap != "obs":
        a[order[1]] = a[order[0]] - gap
    beta = th[n_alpha:]
    aff = np.zeros((ci["B"], ci["J"]))
    aff[ci["bi"], ci["ji"]] = X[ci["idx"]] @ beta
    avail = np.zeros((ci["B"], ci["J"]), bool)
    avail[ci["bi"], ci["ji"]] = True
    if avail_mask is not None:
        avail &= avail_mask
    base = np.where(avail, a[None, :], -np.inf)
    return base, np.where(avail, base + aff, -np.inf), order[:2]


def cross_contests(df, gap="0.0"):
    """Contests whose two leading nominees come from different confederations in at least half of
    the parameter draws (the contenders' design runs only on such draws, so its replications count
    them), read from a design_contests_*.csv table."""
    x = df[df.gap.astype(str) == str(gap)]
    full = x[x.design == "fifa"].set_index("contest").reps
    ce = x[x.design == "contenders_equal"].set_index("contest").reps
    return sorted(c for c in ce.index if ce[c] >= 0.5 * full[c])


def run_design(ci, base, biased, rng, reps, rule, pick=None, compat_block=None, contenders=None):
    """Replications with common Gumbel noise; `pick(rng)` selects voters for each replication.
    With `contenders` (the two leading nominees' confederations) also returns, per replication, the
    share of jurors from the leader's confederation minus the share from the runner-up's."""
    out, gaps = [], []
    for _ in range(reps):
        sel = pick(rng) if pick is not None else np.arange(ci["B"])
        if contenders is not None:
            cf = ci["confed"][sel]
            gaps.append(float(np.mean(cf == contenders[0]) - np.mean(cf == contenders[1])))
        G = rng.gumbel(size=(1, len(sel), ci["J"]))
        ub = biased[sel][None] + G
        if compat_block is not None:
            ub = np.where(compat_block[sel][None], -np.inf, ub)
        uf = base[sel][None] + G
        f, _, _ = flips(ub, uf, ci["role"][sel], rule)
        out.append(f[0])
    return (np.array(out), gaps) if contenders is not None else np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1_role")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--contests", nargs="*")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--rule", choices=["own", "share", "scoring_zero"], default="own",
                    help="FIFA 2023+ contests: own (art. 11 scoring), share (2010-2022 rule) or scoring with "
                         "zero-point nominees scoring nothing; outputs get a _<rule> suffix")
    ap.add_argument("--tag", default=None, help="output suffix instead of _<rule> (e.g. a matched subset run)")
    a = ap.parse_args()
    out = RES / "design"
    out.mkdir(parents=True, exist_ok=True)
    split, fe_by, aff = SPECS[a.model]
    assert fe_by == (), "design simulations need alpha common to all voters of a contest"
    long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
    X, names, fe, _ = design(long, split, fe_by, aff)
    theta = pd.read_csv(RES / "pooled" / f"theta_{a.model}.csv", keep_default_na=False)
    cov = np.load(RES / "pooled" / f"cov_{a.model}.npz")
    est = theta[theta.kind != "ref"]
    n_alpha = int((est.kind == "alpha").sum())
    col = pd.Series(np.arange(n_alpha), index=est.param[:n_alpha].to_numpy())
    col = pd.concat([col, pd.Series(n_alpha, index=theta.param[theta.kind == "ref"].to_numpy())])
    fe_col = fe.map(col).fillna(n_alpha + 1).astype(int).to_numpy()
    rng = np.random.default_rng(a.seed)
    mle = est.value.to_numpy()
    draws = hybrid_draws(mle, cov["V"], cov["V_cluster"], n_alpha, a.draws, rng)
    per = max(1, a.reps // a.draws)
    same_nation = long.same_nation.to_numpy().astype(bool)

    rows = []
    for contest in a.contests or sorted(long.contest.unique()):
        ci = contest_inputs(long, X, names, fe_col, contest)
        nom_confed = (long[long.contest == contest].drop_duplicates("person").set_index("person")
                      .nominee_confed.reindex(ci["noms"]).to_numpy())
        compat = np.zeros((ci["B"], ci["J"]), bool)
        compat[ci["bi"], ci["ji"]] = same_nation[ci["idx"]]
        media = np.flatnonzero(ci["role"] == "media")
        uefa, other = np.flatnonzero(ci["confed"] == "UEFA"), np.flatnonzero(ci["confed"] != "UEFA")
        by_confed = {c: np.flatnonzero(ci["confed"] == c) for c in CONFEDS}
        k_eq = min(EQUAL_K, min(len(v) for v in by_confed.values()))
        # FIFA's rule of the year for every jury drawn from FIFA's three groups (fifa_rules.py)
        rule = "scoring" if rule_for(contest) == "scoring" else "fifa"
        if rule == "scoring" and a.rule != "own":
            rule = "fifa" if a.rule == "share" else "scoring_zero"

        def designs_for(top2):
            c1, c2 = nom_confed[top2[0]], nom_confed[top2[1]]
            ds = [("fifa", None, rule, None, None), ("fifa_nocompat", None, rule, None, compat),
                  ("media", None, "points", (lambda r, m=media: m), None)]
            ds += [(f"size_{n}", n, rule, (lambda r, n=n: r.choice(ci["B"], n, replace=False)), None)
                   for n in SIZES if n < ci["B"]]
            for sh in UEFA_SHARES:
                nu = int(round(UEFA_JURY * sh))
                if nu <= len(uefa) and UEFA_JURY - nu <= len(other):
                    ds.append((f"uefa_{sh}", sh, rule,
                               (lambda r, nu=nu: np.concatenate([r.choice(uefa, nu, replace=False),
                                                                 r.choice(other, UEFA_JURY - nu, replace=False)])),
                               None))
            if k_eq >= EQUAL_K_MIN:     # the same number of jurors from each confederation
                ds.append(("equal_confed", k_eq, rule,
                           (lambda r: np.concatenate([r.choice(v, k_eq, replace=False) for v in by_confed.values()])),
                           None))
            if c1 != c2:                # the contenders' confederations equally represented
                b1, b2 = by_confed[c1], by_confed[c2]
                k = min(len(b1), len(b2))
                rest = np.setdiff1d(np.arange(ci["B"]), np.concatenate([b1, b2]))
                ds.append(("contenders_equal", k, rule,
                           (lambda r, b1=b1, b2=b2, k=k, rest=rest:
                            np.concatenate([r.choice(b1, k, replace=False), r.choice(b2, k, replace=False), rest])),
                           None))
            return ds, (c1, c2)

        for gap in GAPS:
            res, bloc, meta, pairs = {}, {}, {}, {}
            for th in draws:
                base, biased, top2 = utilities(ci, X, fe_col, th, n_alpha, gap)
                ds, (c1, c2) = designs_for(top2)
                for name, param, rl, pick, block in ds:
                    f, g = run_design(ci, base, biased, rng, per, rl, pick, block, (c1, c2))
                    res.setdefault(name, []).append(f)
                    bloc.setdefault(name, []).extend(g)
                    meta[name] = (param, rl)
                    pairs.setdefault(name, []).append((c1, c2))     # the contenders can differ by draw
            for name, fl in res.items():
                f = np.concatenate(fl)
                param, rl = meta[name]
                pr = pd.Series([f"{x}|{y}" for x, y in pairs[name]])
                c1, c2 = pr.mode().iloc[0].split("|")
                rows.append({"contest": contest, "design": name, "param": param, "rule": rl, "gap": gap,
                             "p_flip": f.mean(), "reps": len(f), "leader_confed": c1, "runner_up_confed": c2,
                             "share_cross": float(np.mean([x != y for x, y in pairs[name]])),
                             "bloc_gap": float(np.mean(bloc[name]))})
        print(contest, flush=True, file=sys.stderr)
    df = pd.DataFrame(rows)
    tag = f"_{a.tag}" if a.tag else ("" if a.rule == "own" else f"_{a.rule}")
    df.to_csv(out / f"design_contests_{a.model}{tag}.csv", index=False)
    summ = df.groupby(["design", "gap"], dropna=False).p_flip.agg(["mean", "median", "max"]).reset_index()
    summ.to_csv(out / f"design_summary_{a.model}{tag}.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
