"""How the counterfactual flip probabilities behave when the ranking errors are not Gumbel.

python -I nhb/src/misspec.py --noise gumbel|normal [--reps 25 --seed 1 --clock 50]
    ->  nhb/results/misspec/flips_<noise>_<seed>.csv, betas_<noise>_<seed>.csv

Ballots are simulated from the fitted FIFA model M1_role: its contest-by-nominee effects and
similarity bonuses, with the observed jurors, choice sets and ballot lengths. The errors are
independent and either standard Gumbel (the Plackett-Luce model, which our estimator assumes) or
normal with the same variance, pi^2/6 (a Thurstone ranking model, which it does not). In a simulated
data set the true counterfactual is known: the same errors with the bonuses removed give each
juror's ballot without them, and the leader under each year's rule shows whether removing the
bonuses changes it (true_flip). The model is then refitted to the simulated ballots and the flip
probability estimated as in counterfactuals.py, at the estimates, from `clock` conditional clock
draws per contest (p_hat). Over data sets, a calibrated estimator gives mean p_hat equal to the
frequency of true flips, contest by contest and summed over contests.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactuals import SCENARIOS, conditional_clocks, contest_arrays, rerank, rule_scores, scheme  # noqa: E402
from fifa_rules import rule_for  # noqa: E402
from fit_pooled import SPECS, Model, design, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
MODEL = "M1_role"
CLOCK = 50            # conditional clock draws per contest and data set (--clock)


def alpha_rows(fe, param, kind, value):
    """Per-row contest-by-nominee effect: estimated cells, 0 for reference cells, -inf for dropped ones."""
    a = pd.Series(value[kind == "alpha"], index=param[kind == "alpha"])
    a = pd.concat([a, pd.Series(0.0, index=param[kind == "ref"])])
    return fe.map(a).fillna(-np.inf).to_numpy()


def top_m(U, m):
    """Ranks 1..m of the m largest utilities in each row (0 otherwise); -inf entries are never ranked."""
    order = np.argsort(-U, axis=1)
    ranks = np.zeros(U.shape, int)
    rows = np.arange(len(U))
    for k in range(int(m.max())):
        on = m > k
        ranks[rows[on], order[on, k]] = k + 1
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", choices=["gumbel", "normal"], required=True)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clock", type=int, default=CLOCK, help="conditional clock draws per contest")
    a = ap.parse_args()
    out = RES / "misspec"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([20261022, a.seed, int(a.noise == "normal")])
    split, fe_by, aff = SPECS[MODEL]
    long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
    X, names, fe, _ = design(long, split, fe_by, aff)
    theta = pd.read_csv(RES / "pooled" / f"theta_{MODEL}.csv", keep_default_na=False)
    assert list(theta.param[theta.kind == "beta"]) == names
    beta = theta.value[theta.kind == "beta"].to_numpy()
    alpha = alpha_rows(fe, theta.param.to_numpy(), theta.kind.to_numpy(), theta.value.to_numpy())
    cols = {s: np.array([any(n.startswith(f) for f in fs) for n in names]) for s, fs in SCENARIOS.items()}
    contests = []
    for c in sorted(long.contest.unique()):
        d, ballots, noms, bi, ji = contest_arrays(long, c)
        B, J = len(ballots), len(noms)
        idx = d.index.to_numpy()
        obs = np.zeros((B, J), int)
        obs[bi, ji] = d["rank"].to_numpy()
        A = np.full((B, J), -np.inf)
        A[bi, ji] = alpha[idx]
        bon = {s: np.zeros((B, J)) for s in ["none"] + list(SCENARIOS)}
        bon["none"][bi, ji] = X[idx] @ beta
        for s in SCENARIOS:
            bon[s][bi, ji] = X[idx] @ np.where(cols[s], 0.0, beta)
        contests.append({"contest": c, "idx": idx, "bi": bi, "ji": ji, "B": B, "J": J, "A": A, "bon": bon,
                         "m": obs.max(axis=1), "role": d.groupby("ballot").role.first().reindex(ballots).to_numpy(),
                         "pts": scheme(c), "rule": rule_for(c)})
    scale = np.pi / np.sqrt(6.0)          # sd of the standard Gumbel
    rows, brows = [], []
    for r in range(a.reps):
        sim_rank = np.zeros(len(long), int)
        truth = {}
        for ci in contests:
            shape = (ci["B"], ci["J"])
            eps = rng.gumbel(size=shape) if a.noise == "gumbel" else rng.normal(scale=scale, size=shape)
            ranks = top_m(ci["A"] + ci["bon"]["none"] + eps, ci["m"])
            lead = rule_scores(ranks, ci["role"], ci["pts"], ci["rule"]).argmax()
            truth[ci["contest"]] = (ranks, lead, {s: int(rule_scores(top_m(ci["A"] + ci["bon"][s] + eps, ci["m"]),
                                                                     ci["role"], ci["pts"], ci["rule"]).argmax() != lead)
                                                  for s in SCENARIOS})
            sim_rank[ci["idx"]] = ranks[ci["bi"], ci["ji"]]
        fit = Model(long.assign(rank=sim_rank), split, fe_by, aff).fit()
        kind = np.array(["alpha"] * fit.n_alpha + ["ref"] * len(fit.ref_names))
        a_hat = alpha_rows(fe, np.array(fit.fe_names + fit.ref_names), kind,
                           np.concatenate([fit.theta[:fit.n_alpha], np.zeros(len(fit.ref_names))]))
        b_hat = fit.theta[fit.n_alpha:]
        assert fit.names == names
        brows += [{"rep": r, "noise": a.noise, "term": n, "beta_true": bt, "beta_hat": bh}
                  for n, bt, bh in zip(names, beta, b_hat)]
        for ci in contests:
            ranks, lead, tf = truth[ci["contest"]]
            idx, bi, ji = ci["idx"], ci["bi"], ci["ji"]
            u = np.full((ci["B"], ci["J"]), -np.inf)
            u[bi, ji] = a_hat[idx] + X[idx] @ b_hat
            ucf = {}
            for s in SCENARIOS:
                ucf[s] = np.full((ci["B"], ci["J"]), -np.inf)
                ucf[s][bi, ji] = a_hat[idx] + X[idx] @ np.where(cols[s], 0.0, b_hat)
            hits = {s: 0 for s in SCENARIOS}
            for _ in range(a.clock):
                logX, _ = conditional_clocks(u, ranks, rng)
                for s in SCENARIOS:
                    cf = rerank(logX, ucf[s], ci["m"])
                    hits[s] += int(rule_scores(cf, ci["role"], ci["pts"], ci["rule"]).argmax() != lead)
            rows += [{"rep": r, "noise": a.noise, "contest": ci["contest"], "scenario": s, "true_flip": tf[s],
                      "p_hat": hits[s] / a.clock} for s in SCENARIOS]
        print(f"{a.noise} rep {r}: true flips {sum(t[2]['all'] for t in truth.values())}, "
              f"estimated {sum(x['p_hat'] for x in rows if x['rep'] == r and x['scenario'] == 'all'):.2f}",
              file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(out / f"flips_{a.noise}_{a.seed}.csv", index=False)
    pd.DataFrame(brows).to_csv(out / f"betas_{a.noise}_{a.seed}.csv", index=False)


if __name__ == "__main__":
    main()
