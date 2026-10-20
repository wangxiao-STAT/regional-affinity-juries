"""Paired counterfactual juries: each observed ballot with the affinity bonuses removed.

python -I nhb/src/counterfactuals.py [--model M1_role] [--draws 200]
    ->  nhb/results/pooled/cf_{model}_contests.csv, cf_{model}_nominees.csv

Under the Plackett-Luce model a ballot is the order in which independent exponential clocks
T_j ~ Exp(lambda_j), lambda_j = exp(u_j), ring. Given the observed top m, the clock times can be
drawn exactly: the k-th ring comes Exp(sum of the remaining rates) after the (k-1)-th, and every
unranked nominee rings Exp(lambda_j) after the m-th (memorylessness). Writing T_j = X_j / lambda_j,
the same X_j with the counterfactual rates lambda'_j gives the voter's counterfactual ranking, so
factual and counterfactual ballots share their randomness. Parameters are drawn from the normal
approximation (fit_pooled.hybrid_draws: similarity coefficients with the voter-country clustered
covariance, nominee effects model-based given them); p_flip_mle repeats the clock draws at the
maximum-likelihood estimates, the probability from the ballots' randomness alone. FIFA standings follow FIFA's 2010-2015
rule (validated against the published percentages): within each juror group a nominee's share of
5/3/1 points, averaged over coaches, captains and media. From 2016 fans add a fourth quarter that is
not published per voter, so these are jury standings. From 2023 FIFA decides the winner by art. 11
scoring points (fifa_rules.py); winners, flips and margins follow each year's rule (margins in per
cent of all points or scoring points awarded).
Ballon d'Or standings are total points,
6-4-3-2-1 for the top five (2019-2023) and 15-12-10-8-7-5-4-3-2-1 for the top ten (2024-2025).
Models from fit_pooled.py (FIFA only) and fit_combined.py (FIFA and Ballon d'Or) are both accepted.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pooled import SPECS, design, hybrid_draws, prepare, psd_sqrt  # noqa: E402
import fit_combined  # noqa: E402
from fifa_rules import jury_score, rule_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "pooled"
POINTS = np.array([5.0, 3.0, 1.0])
BDO_TOP5 = np.array([6.0, 4.0, 3.0, 2.0, 1.0])
BDO_TOP10 = np.array([15.0, 12.0, 10.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0])
Q_LO, Q_HI = 0.05, 0.95          # interval of the counterfactual margins and shifts
SCENARIOS = {"all": ["same_nation", "same_confed", "same_league"], "nation": ["same_nation"],
             "confed": ["same_confed"], "league": ["same_league"]}


def conditional_clocks(loglam, ranks, rng):
    """log X, with X = lambda * T, for clocks consistent with the observed ranking (rows: ballots).

    loglam: (B, J) with -inf for nominees outside the choice set; ranks: (B, J) ints, 0 = unranked.
    Rates are rescaled per ballot (which leaves X unchanged) and X is kept in logs: for an unranked
    nominee X = lambda * t_m + E with E ~ Exp(1), which stays exact when lambda underflows.
    """
    B, J = loglam.shape
    avail = np.isfinite(loglam)
    s = np.where(avail, loglam - np.where(avail, loglam, -np.inf).max(axis=1, keepdims=True), -np.inf)
    lam = np.exp(s)
    m = ranks.max(axis=1)
    t = np.zeros(B)
    logX = np.full((B, J), np.inf)
    remaining = np.where(avail, lam, 0.0)
    for k in range(1, ranks.max() + 1):
        on = m >= k
        rate = remaining.sum(axis=1)
        t = np.where(on, t + rng.exponential(size=B) / np.where(rate > 0, rate, 1.0), t)
        hit = (ranks == k) & on[:, None]
        logX[hit] = (s + np.log(np.repeat(t, J).reshape(B, J)))[hit]
        remaining[hit] = 0.0
    rest = avail & (ranks == 0)
    with np.errstate(divide="ignore"):                      # log(0) = -inf for ballots with no pick
        lt = np.log(np.repeat(t, J).reshape(B, J))
        logX[rest] = np.logaddexp(s + lt, np.log(rng.exponential(size=(B, J))))[rest]
    return logX, lam


def rerank(logX, loglam_cf, m):
    """Top-m order of the clocks X / lambda' (ranks 1..m, 0 otherwise), computed in logs."""
    avail = np.isfinite(loglam_cf)
    s = loglam_cf - np.where(avail, loglam_cf, -np.inf).max(axis=1, keepdims=True)
    logT = np.where(avail & np.isfinite(logX), logX - np.where(avail, s, 0.0), np.inf)
    order = np.argsort(logT, axis=1)
    ranks = np.zeros_like(order)
    rows = np.arange(len(logT))[:, None]
    ranks[rows, order] = np.arange(1, logT.shape[1] + 1)
    return np.where(ranks <= m[:, None], ranks, 0)


def scheme(contest):
    if not contest.startswith("BdO"):
        return POINTS
    return BDO_TOP5 if int(contest[3:7]) <= 2023 else BDO_TOP10


def standings(ranks, role, points=POINTS):
    """Equal-weight average over the juror groups present of each nominee's share of points
    (a single group, as in the Ballon d'Or, gives the share of total points)."""
    pts = np.where(ranks > 0, points[np.clip(ranks - 1, 0, len(points) - 1)], 0.0)
    groups = [r for r in np.unique(role) if pts[role == r].sum() > 0]
    share = np.zeros(ranks.shape[1])
    for r in groups:
        g = pts[role == r].sum(axis=0)
        share += g / g.sum() / len(groups)
    return 100 * share


def rule_percent(ranks, role, points, rule):
    """Each nominee's standing in per cent of everything awarded under the award's rule, the scale of
    the margins: group shares (FIFA to 2022), shares of total points (Ballon d'Or), or shares of all
    scoring points awarded (FIFA from 2023, whose leader can trail in shares of points)."""
    if rule != "scoring":
        return standings(ranks, role, points)
    pts = np.where(ranks > 0, points[np.clip(ranks - 1, 0, len(points) - 1)], 0.0)
    s = jury_score(pts, role, "scoring")
    return 100 * s / s.sum()


def rule_scores(ranks, role, points, rule):
    """Scores whose argmax is the winner under the award's own rule (fifa_rules.py)."""
    pts = np.where(ranks > 0, points[np.clip(ranks - 1, 0, len(points) - 1)], 0.0)
    firsts = (ranks[role == "captain"] == 1).sum(axis=0) if rule == "scoring" else None
    return jury_score(pts, role, rule, firsts)


def contest_arrays(long, contest):
    d = long[long.contest == contest]
    ballots = np.sort(d.ballot.unique())
    noms = np.sort(d.person.unique())
    bi = np.searchsorted(ballots, d.ballot.to_numpy())
    ji = np.searchsorted(noms, d.person.to_numpy())
    return d, ballots, noms, bi, ji


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1_role")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--cov", choices=["hybrid", "cluster"], default="hybrid",
                    help="hybrid: coefficients clustered, nominee effects model-based given them")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--contests", nargs="*", help="restrict to these contests (default: all)")
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--rule", choices=["own", "share"], default="own",
                    help="share: FIFA's 2010-2022 group-share rule for every FIFA contest (output *_share_*)")
    a = ap.parse_args()
    tag = "" if a.rule == "own" else f"_{a.rule}"
    res_dir = Path(a.res)
    if a.model in SPECS:
        split, fe_by, aff = SPECS[a.model]
        long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
        X, names, fe, _ = design(long, split, fe_by, aff)
    else:
        long, X, names, fe_sets = fit_combined.build(a.model)
        fe = pd.Series(fe_sets[0][0])
    theta = pd.read_csv(res_dir / f"theta_{a.model}.csv", keep_default_na=False)
    cov = np.load(res_dir / f"cov_{a.model}.npz")
    V, V_model = cov["V_cluster"], cov["V"]
    est = theta[theta.kind != "ref"]
    n_alpha = int((est.kind == "alpha").sum())
    assert list(est.param[n_alpha:]) == names
    # fixed-effect cell -> column of the parameter vector; reference cells point to an appended 0,
    # cells dropped for having no pick to an appended -inf
    col = pd.Series(np.arange(n_alpha), index=est.param[:n_alpha].to_numpy())
    col = pd.concat([col, pd.Series(n_alpha, index=theta.param[theta.kind == "ref"].to_numpy())])
    fe_col = fe.map(col).fillna(n_alpha + 1).astype(int).to_numpy()
    rng = np.random.default_rng(a.seed)
    mle = est.value.to_numpy()
    if a.cov == "hybrid":
        draws = [mle] + hybrid_draws(mle, V_model, V, n_alpha, a.draws, rng)
    else:
        L = psd_sqrt(V)
        draws = [mle] + [mle + L @ rng.standard_normal(len(V)) for _ in range(a.draws)]
    # the same number of draws at the MLE: the flip probability from the ballots' randomness alone
    mle_draws = [mle] * a.draws
    aff_cols = {s: np.array([any(n.startswith(f) for f in fs) for n in names]) for s, fs in SCENARIOS.items()}

    rows_c, rows_n = [], []
    for contest in a.contests or sorted(long.contest.unique()):
        d, ballots, noms, bi, ji = contest_arrays(long, contest)
        B, J = len(ballots), len(noms)
        idx = d.index.to_numpy()
        ranks = np.zeros((B, J), int)
        ranks[bi, ji] = d["rank"].to_numpy()
        role = d.groupby("ballot").role.first().reindex(ballots).to_numpy()
        m = ranks.max(axis=1)
        pts = scheme(contest)
        rule = rule_for(contest)
        if a.rule == "share" and rule == "scoring":
            rule = "share"
        fact = rule_percent(ranks, role, pts, rule)         # the scale of the margins
        order = np.argsort(-rule_scores(ranks, role, pts, rule))   # winner by the award's own rule
        w, ru = order[0], order[1]
        res = {s: [] for s in SCENARIOS}
        won = {s: [] for s in SCENARIOS}
        won_mle = {s: [] for s in SCENARIOS}
        for kind, th in [("param", t) for t in draws[1:]] + [("mle", t) for t in mle_draws]:
            alpha = np.append(th[:n_alpha], [0.0, -np.inf])[fe_col[idx]]
            beta = th[n_alpha:]
            u = np.full((B, J), -np.inf)
            u[bi, ji] = alpha + X[idx] @ beta
            Xc, _ = conditional_clocks(u, ranks, rng)
            for s, cols in aff_cols.items():
                ucf = np.full((B, J), -np.inf)
                ucf[bi, ji] = alpha + X[idx] @ np.where(cols, 0.0, beta)
                cf_ranks = rerank(Xc, ucf, m)
                if kind == "mle":
                    won_mle[s].append(rule_scores(cf_ranks, role, pts, rule).argmax())
                    continue
                res[s].append(rule_percent(cf_ranks, role, pts, rule))
                won[s].append(rule_scores(cf_ranks, role, pts, rule).argmax())
        for s in SCENARIOS:
            S = np.array(res[s])
            win = np.array(won[s])
            rows_c.append({"model": a.model, "draws": a.draws, "contest": contest, "rule": rule, "scenario": s,
                           "winner": noms[w],
                           "runner_up": noms[ru],
                           "margin": fact[w] - fact[ru], "p_flip": float((win != w).mean()),
                           "p_flip_mle": float((np.array(won_mle[s]) != w).mean()), "cov": a.cov,
                           "cf_margin_mean": float((S[:, w] - S[:, ru]).mean()),
                           "cf_margin_q05": float(np.quantile(S[:, w] - S[:, ru], Q_LO)),
                           "cf_margin_q95": float(np.quantile(S[:, w] - S[:, ru], Q_HI)),
                           "cf_winner_mode": noms[np.bincount(win, minlength=J).argmax()]})
            for j in range(J):
                rows_n.append({"model": a.model, "contest": contest, "scenario": s, "person": noms[j],
                               "share": fact[j], "delta_mean": float((S[:, j] - fact[j]).mean()),
                               "delta_q05": float(np.quantile(S[:, j] - fact[j], Q_LO)),
                               "delta_q95": float(np.quantile(S[:, j] - fact[j], Q_HI))})
        print(f"{contest}: {noms[w]} over {noms[ru]} by {fact[w] - fact[ru]:.2f}; "
              f"P(flip | all removed) = {rows_c[-4]['p_flip']:.3f}", file=sys.stderr, flush=True)
    pd.DataFrame(rows_c).to_csv(res_dir / f"cf_{a.model}{tag}_contests.csv", index=False)
    pd.DataFrame(rows_n).to_csv(res_dir / f"cf_{a.model}{tag}_nominees.csv", index=False)


if __name__ == "__main__":
    main()
