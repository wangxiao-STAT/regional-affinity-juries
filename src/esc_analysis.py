"""Eurovision juries: similarity bonuses, counterfactual jury standings and jury-design simulations.

python -I nhb/src/esc_analysis.py fit            ->  nhb/results/eurovision/estimates.csv, fits.json,
                                                    theta_<model>.csv, cov_<model>.npz
python -I nhb/src/esc_analysis.py counterfactual [--draws 1000]   ->  cf_<model>_contests.csv, standing_check_<model>.csv
python -I nhb/src/esc_analysis.py design [--draws 40 --reps 800]  ->  design_<model>_contests.csv
python -I nhb/src/esc_analysis.py bases [--draws 200]  ->  bases_<model>.csv (other national aggregation rules)

Data: esc_long.csv (build_eurovision.py), restricted to national juries whose five (seven in 2026)
rankings are complete (validate_eurovision.py). Each juror ranks every song in the final except the
own country's, so the ranking model uses the full ranking.
Models (contest-by-song effects in all), under three groupings of the countries (build_eurovision.py):
voting blocs (B: the five blocs of 2001-2005 in Gatherer 2006; a country in no bloc has no group bonus,
and for the designs it is grouped with the other such countries of its UN subregion), UN M49
subregions (E) and a finer grouping chosen after the main analysis (C):
  B1, E1, C1              bonus for a song from the juror's group and for a song from a bordering country
  B2, E2, C2              the group bonus specific to the juror's group
  B1_top10, E1_top10, ... B1, E1, C1 with every ranking cut to its first ten places (the places that score)
  B1_region           B1 plus the subregion bonus (fit only: does geography add to the blocs?)
Designs: actual juries; neighbours barred (in the biased and the bias-free jury); the same number of
countries from each group (EQUAL_K); the contenders' groups equalised; and, to create imbalance,
half or all of one contender's group removed (tilt_lead_*: the runner-up's group removed, so the
leader's group is over-represented; tilt_run_*: the leader's group removed).
The top-ten models' counterfactuals keep the full rankings (every rank enters the national score) and
remove the bonuses estimated from the top ten.
Jury standing: each national jury's ranking (sum of ranks to 2017; exponential weights from 2018, base
from validate_eurovision.py; exact ties broken at random) gives 12, 10, 8, ..., 1 points to its top
ten; the standing is the sum over countries. National juries without complete published rankings keep
their published points. The televote is not modelled, so the outcome is the jury standing, as for
FIFA's awards from 2016.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactuals import conditional_clocks, rerank  # noqa: E402
from fit_pooled import EXTREME_ALPHA, ExplodedLogit, hybrid_draws  # noqa: E402
from validate_eurovision import INELIGIBLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "eurovision"
POINTS = np.array([12, 10, 8, 7, 6, 5, 4, 3, 2, 1], float)
GROUPING = {"B": ("same_bloc", "voter_bloc", "nominee_bloc"), "E": ("same_region", "voter_region", "nominee_region"),
            "C": ("same_fine", "voter_fine", "nominee_fine")}
GROUPING_NAME = {"B": "blocs", "E": "M49", "C": "finer"}
SCENARIOS = {"all": ["same_group", "same_border"], "group": ["same_group"], "border": ["same_border"]}
MODELS = ("B1", "B2", "B1_top10", "E1", "E2", "E1_top10", "C1", "C2", "C1_top10", "B1_region")
BASES = [None, 0.80, 0.85, 0.90]   # other national aggregation rules for the years from 2018 ("bases")
GAPS = ["obs", 0.0, 0.1, 0.2, 0.4, 0.8]
EQUAL_K = 4          # countries per group in the equal-group design (groups with fewer are skipped)
CONTENDER_K_MIN = 3  # the contenders' design needs at least this many countries in each contender's group


def load():
    long = pd.read_csv(PROC / "esc_long.csv")
    val = pd.read_csv(PROC / "esc_validation.csv")
    ok = set(map(tuple, val[val.complete.astype(bool)][["year", "voter"]].to_numpy()))
    keep = [(y, v) in ok for y, v in zip(long.year, long.voter_country)]
    return long[keep].reset_index(drop=True), val


def grouped(long, model):
    """long with same_group, voter_group and nominee_group for the model's grouping."""
    same, vg, ng = GROUPING[model[0]]
    return long.assign(same_group=long[same], voter_group=long[vg], nominee_group=long[ng])


def rule_base(val):
    """year -> None (sum of ranks) or the exponential base reproducing the published ranks."""
    b = val[val.complete.astype(bool)].groupby("year").base.first()
    return {int(y): (None if pd.isna(v) else float(v)) for y, v in b.items()}


def design_matrix(long, model):
    """Columns named same_group (the model's grouping), same_group[<group>] and same_border."""
    d = grouped(long, model)
    if model.endswith("2"):
        cols, names = [], []
        for r in sorted(d.voter_group.unique()):
            m = (d.voter_group == r).to_numpy()
            cols.append(d.same_group.to_numpy() * m)
            names.append(f"same_group[{r}]")
        cols.append(d.same_border.to_numpy())
        names.append("same_border")
        X = np.column_stack(cols).astype(float)
        keep = X.any(axis=0)
        return X[:, keep], [n for n, k in zip(names, keep) if k]
    terms = ["same_group", "same_border"] + (["same_region"] if model == "B1_region" else [])
    return d[terms].to_numpy(float), terms


def fit(long, model):
    d = long.copy()
    if model.endswith("_top10"):
        d["rank"] = np.where(d["rank"] <= 10, d["rank"], 0)
    X, names = design_matrix(d, model)
    fe = [((d.contest + "#" + d.person).to_numpy(), d.contest.to_numpy())]
    t = time.time()
    m = ExplodedLogit(d, X, names, fe).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    est = pd.DataFrame({"model": model, "grouping": GROUPING_NAME[model[0]], "term": names,
                        "beta": b, "se": np.sqrt(np.diag(V)[m.n_alpha:]),
                        "se_cluster": np.sqrt(np.diag(Vc)[m.n_alpha:])})
    est["z_cluster"] = est.beta / est.se_cluster
    info = {"model": model, "loglik": m.ll, "n_alpha": m.n_alpha, "n_beta": m.n_beta,
            "ballots": int(d.ballot.nunique()), "contests": int(d.contest.nunique()),
            "voter_countries": int(d.voter_country.nunique()), "converged": bool(m.opt.success),
            "grad_max": float(np.abs(m.opt.jac).max()), "hessian_cond": m.cond,
            "extreme_alpha_cells": int((np.abs(m.theta[:m.n_alpha]) > EXTREME_ALPHA).sum()),
            "seconds": round(time.time() - t, 1)}
    theta = pd.DataFrame({"param": m.fe_names + names + m.ref_names,
                          "kind": ["alpha"] * m.n_alpha + ["beta"] * m.n_beta + ["ref"] * len(m.ref_names),
                          "value": np.concatenate([m.theta, np.zeros(len(m.ref_names))])})
    print(f"{model}: ballots={info['ballots']} ll={m.ll:.1f} conv={m.opt.success} cond={m.cond:.1e}",
          file=sys.stderr, flush=True)
    return est, info, theta, V, Vc


def national_points(ranks, own, base, rng):
    """ranks: (..., jurors, songs) juror ranks (own song: any value); own: (songs,) bool.
    Returns (..., songs) national jury points (12, 10, 8, ..., 1 to the top ten)."""
    r = ranks.astype(float)
    score = -r.sum(axis=-2) if base is None else (base ** (r - 1.0)).sum(axis=-2)
    score = score + 1e-9 * rng.random(score.shape)          # exact ties: at random
    score = np.where(own, -np.inf, score)
    order = np.argsort(-score, axis=-1)
    pts = np.zeros(score.shape)
    np.put_along_axis(pts, order[..., :len(POINTS)], np.broadcast_to(POINTS, order[..., :len(POINTS)].shape), axis=-1)
    return np.where(own, 0.0, pts)


class Contest:
    """Arrays of one final: songs, juror ballots grouped by voting country, similarity indicators."""

    def __init__(self, long, contest, fe_col, X):
        d = long[long.contest == contest]
        self.year = int(contest[3:])
        self.songs = np.array(sorted(set(d.person)))
        self.ballots = np.sort(d.ballot.unique())
        bi = np.searchsorted(self.ballots, d.ballot.to_numpy())
        ji = np.searchsorted(self.songs, d.person.to_numpy())
        B, J = len(self.ballots), len(self.songs)
        self.idx, self.bi, self.ji = d.index.to_numpy(), bi, ji
        self.ranks = np.zeros((B, J), int)
        self.ranks[bi, ji] = d["rank"].to_numpy()
        self.avail = np.zeros((B, J), bool)
        self.avail[bi, ji] = True
        self.voter = d.groupby("ballot").voter_country.first().reindex(self.ballots).to_numpy()
        self.voter_group = d.groupby("ballot").voter_group.first().reindex(self.ballots).to_numpy()
        reg = d.drop_duplicates("person").set_index("person").nominee_group
        self.song_group = reg.reindex(self.songs).to_numpy()
        self.countries = np.unique(self.voter)
        self.Xb = np.zeros((B, J, X.shape[1]))
        self.Xb[bi, ji] = X[d.index.to_numpy()]
        self.fe_col = fe_col[d.index.to_numpy()]
        self.border = np.zeros((B, J), bool)
        self.border[bi, ji] = d.same_border.to_numpy().astype(bool)

    def utilities(self, theta, n_alpha, beta_mask=None):
        alpha = np.append(theta[:n_alpha], [0.0, -np.inf])[self.fe_col]
        a = np.full(len(self.songs), np.nan)
        a[self.ji] = alpha
        beta = theta[n_alpha:] if beta_mask is None else np.where(beta_mask, 0.0, theta[n_alpha:])
        u = np.where(self.avail, a[None, :] + self.Xb @ beta, -np.inf)
        return a, u

    def standing(self, ranks, base, fixed, rng):
        """Jury standing (points per song) from juror ranks: national aggregation, sum over countries."""
        tot = fixed.copy()
        out = np.isin(self.songs, list(INELIGIBLE.get(self.year, ())))
        for c in self.countries:
            rows = self.voter == c
            own = (self.songs == c) | out
            tot += national_points(ranks[rows], own, base, rng)
        return tot


def setup(model):
    long, val = load()
    X, names = design_matrix(long, model)
    long = grouped(long, model)
    theta = pd.read_csv(OUT / f"theta_{model}.csv")
    cov = np.load(OUT / f"cov_{model}.npz")
    est = theta[theta.kind != "ref"]
    n_alpha = int((est.kind == "alpha").sum())
    col = pd.Series(np.arange(n_alpha), index=est.param[:n_alpha].to_numpy())
    col = pd.concat([col, pd.Series(n_alpha, index=theta.param[theta.kind == "ref"].to_numpy())])
    fe = (long.contest + "#" + long.person)
    fe_col = fe.map(col).fillna(n_alpha + 1).astype(int).to_numpy()
    return long, val, X, names, est.value.to_numpy(), cov, n_alpha, fe_col


def official_totals(year, songs):
    t = pd.read_csv(PROC / "esc_jury_totals.csv")
    return t[t.year == year].set_index("song_country").jury_score.reindex(songs).fillna(0).to_numpy(float)


def fixed_points(val, contest_year, songs):
    """Points of the national juries without complete published rankings (juries replaced by the EBU or
    by the televote, missing tables): the official jury score minus the published points of the
    complete juries."""
    nat = pd.read_csv(PROC / "esc_national.csv", keep_default_na=False)
    done = set(val[(val.year == contest_year) & val.complete.astype(bool)].voter)
    n = nat[(nat.year == contest_year) & nat.voter.isin(done)]
    n = n.assign(p=pd.to_numeric(n.jury_points, errors="coerce").fillna(0))
    published = n.groupby("song_country").p.sum().reindex(songs).fillna(0).to_numpy()
    return official_totals(contest_year, songs) - published


def counterfactual(model, n_draws, seed):
    long, val, X, names, mle, cov, n_alpha, fe_col = setup(model)
    base = rule_base(val)
    rng = np.random.default_rng(seed)
    draws = hybrid_draws(mle, cov["V"], cov["V_cluster"], n_alpha, n_draws, rng)
    rows, check = [], []
    for contest in sorted(long.contest.unique()):
        C = Contest(long, contest, fe_col, X)
        fixed = fixed_points(val, C.year, C.songs)
        obs = C.standing(C.ranks, base[C.year], fixed, rng)
        pub = official_totals(C.year, C.songs)
        check.append({"contest": contest, "max_abs_diff_points": float(np.abs(obs - pub).max()),
                      "official_leader": C.songs[np.argmax(pub)], "recomputed_leader": C.songs[np.argmax(obs)],
                      "fixed_points": float(fixed.sum())})
        order = np.argsort(-obs)
        w, ru = order[0], order[1]
        won = {s: [] for s in SCENARIOS}
        won_mle = {s: [] for s in SCENARIOS}
        m = C.ranks.max(axis=1)
        for kind, th in [("draw", t) for t in draws] + [("mle", mle)] * 200:
            a, u = C.utilities(th, n_alpha)
            logX, _ = conditional_clocks(u, C.ranks, rng)
            for s, terms in SCENARIOS.items():
                mask = np.array([n.split("[")[0] in terms for n in names])
                _, ucf = C.utilities(th, n_alpha, mask)
                cf = rerank(logX, ucf, m)
                st = C.standing(cf, base[C.year], fixed, rng)
                (won if kind == "draw" else won_mle)[s].append(int(np.argmax(st)))
        po = np.argsort(-pub)
        for s in SCENARIOS:
            rows.append({"model": model, "contest": contest, "scenario": s, "leader": C.songs[w],
                         "runner_up": C.songs[ru], "margin_points": float(obs[w] - obs[ru]),
                         "official_leader": C.songs[po[0]], "official_runner_up": C.songs[po[1]],
                         "official_margin_points": float(pub[po[0]] - pub[po[1]]),
                         "margin_pct": float(100 * (obs[w] - obs[ru]) / obs.sum()),
                         "p_flip": float(np.mean(np.array(won[s]) != w)),
                         "p_flip_mle": float(np.mean(np.array(won_mle[s]) != w)), "draws": n_draws})
        print(contest, flush=True, file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"cf_{model}_contests.csv", index=False)
    pd.DataFrame(check).to_csv(OUT / f"standing_check_{model}.csv", index=False)


def simulate(C, u_bias, u_free, sel, rng, base, reps, ban=None):
    """P(winner differs) over reps juries made of the countries in sel(rng) (all their jurors)."""
    flips, gaps = [], []
    for _ in range(reps):
        countries = sel(rng)
        rows = np.isin(C.voter, countries)
        G = rng.gumbel(size=(rows.sum(), len(C.songs)))
        ub = u_bias[rows] + G
        uf = u_free[rows] + G
        if ban is not None:                   # the bar applies to the biased and the bias-free jury alike
            ub = np.where(ban[rows], -np.inf, ub)
            uf = np.where(ban[rows], -np.inf, uf)
        rb = np.argsort(np.argsort(-ub, axis=1), axis=1) + 1
        rf = np.argsort(np.argsort(-uf, axis=1), axis=1) + 1
        tb = np.zeros(len(C.songs))
        tf = np.zeros(len(C.songs))
        vr = C.voter[rows]
        out = np.isin(C.songs, list(INELIGIBLE.get(C.year, ())))
        for c in np.unique(vr):
            own = (C.songs == c) | out
            tb += national_points(rb[vr == c], own, base, rng)
            tf += national_points(rf[vr == c], own, base, rng)
        flips.append(np.argmax(tb) != np.argmax(tf))
        gaps.append(countries)
    return np.array(flips), gaps


def design(model, n_draws, reps, seed):
    long, val, X, names, mle, cov, n_alpha, fe_col = setup(model)
    base = rule_base(val)
    rng = np.random.default_rng(seed)
    draws = hybrid_draws(mle, cov["V"], cov["V_cluster"], n_alpha, n_draws, rng)
    per = max(1, reps // n_draws)
    rows = []
    for contest in sorted(long.contest.unique()):
        C = Contest(long, contest, fe_col, X)
        creg = {c: C.voter_group[C.voter == c][0] for c in C.countries}
        by_reg = {}
        for c, r in creg.items():
            by_reg.setdefault(r, []).append(c)
        for gap in GAPS:
            res, bloc, meta, pairs = {}, {}, {}, {}
            for th in draws:
                a, u = C.utilities(th, n_alpha)
                order = np.argsort(-np.nan_to_num(a, nan=-np.inf))
                th2 = th.copy()
                if gap != "obs":
                    a = a.copy()
                    a[order[1]] = a[order[0]] - gap
                u_bias = np.where(C.avail, a[None, :] + C.Xb @ th[n_alpha:], -np.inf)
                u_free = np.where(C.avail, a[None, :], -np.inf)
                r1, r2 = C.song_group[order[0]], C.song_group[order[1]]
                big = [r for r, cs in by_reg.items() if len(cs) >= EQUAL_K and r in by_reg]
                designs = {"actual": lambda g: C.countries,
                           "border_ban": lambda g: C.countries}
                if len(big) >= 2 and all(len(by_reg[r]) >= EQUAL_K for r in big):
                    designs["equal_groups"] = lambda g, big=big: np.concatenate(
                        [g.choice(by_reg[r], EQUAL_K, replace=False) for r in big])
                # imbalance: drop half or all of one contender's group (the other group stays)
                if r1 != r2:
                    for drop, tag in ((r2, "lead"), (r1, "run")):
                        if drop not in by_reg:
                            continue
                        keep = [c for c in C.countries if creg[c] != drop]
                        half = len(by_reg[drop]) // 2
                        designs[f"tilt_{tag}_half"] = lambda g, keep=keep, drop=drop, half=half: np.concatenate(
                            [keep, g.choice(by_reg[drop], half, replace=False)])
                        designs[f"tilt_{tag}_full"] = lambda g, keep=keep: np.array(keep)
                if r1 != r2 and r1 in by_reg and r2 in by_reg and \
                        min(len(by_reg[r1]), len(by_reg[r2])) >= CONTENDER_K_MIN:
                    k = min(len(by_reg[r1]), len(by_reg[r2]))
                    rest = [c for c in C.countries if creg[c] not in (r1, r2)]
                    designs["contenders_equal"] = lambda g, k=k, r1=r1, r2=r2, rest=rest: np.concatenate(
                        [g.choice(by_reg[r1], k, replace=False), g.choice(by_reg[r2], k, replace=False), rest])
                for name, sel in designs.items():
                    f, sets = simulate(C, u_bias, u_free, sel, rng, base[C.year], per,
                                       ban=C.border if name == "border_ban" else None)
                    res.setdefault(name, []).append(f)
                    for s in sets:
                        sh = np.array([creg[c] for c in s])
                        bloc.setdefault(name, []).append(float(np.mean(sh == r1) - np.mean(sh == r2)))
                    pairs.setdefault(name, []).append((r1, r2))
            for name, fl in res.items():
                f = np.concatenate(fl)
                pr = pd.Series([f"{x}|{y}" for x, y in pairs[name]])
                l1, l2 = pr.mode().iloc[0].split("|")
                rows.append({"contest": contest, "design": name, "gap": gap, "p_flip": f.mean(), "reps": len(f),
                             "leader_group": l1, "runner_up_group": l2,
                             "share_cross": float(np.mean([x != y for x, y in pairs[name]])),
                             "bloc_gap": float(np.mean(bloc[name]))})
        print(contest, flush=True, file=sys.stderr)
    pd.DataFrame(rows).to_csv(OUT / f"design_{model}_contests.csv", index=False)


def bases(model, n_draws, seed):
    """The bonuses' effect on the jury winners of 2018-2026 under other national aggregation rules (sum of
    ranks; exponential weights with other bases): leader, margin and P(winner differs) as in counterfactual."""
    long, val, X, names, mle, cov, n_alpha, fe_col = setup(model)
    rng = np.random.default_rng(seed)
    draws = hybrid_draws(mle, cov["V"], cov["V_cluster"], n_alpha, n_draws, rng)
    fitted = rule_base(val)
    rows = []
    for contest in sorted(long.contest.unique()):
        C = Contest(long, contest, fe_col, X)
        if fitted[C.year] is None:
            continue
        fixed = fixed_points(val, C.year, C.songs)
        m = C.ranks.max(axis=1)
        for base in BASES + [fitted[C.year]]:
            obs = C.standing(C.ranks, base, fixed, rng)
            order = np.argsort(-obs)
            w = order[0]
            won = {"draw": [], "mle": []}
            for kind, th in [("draw", t) for t in draws] + [("mle", mle)] * 200:
                _, u = C.utilities(th, n_alpha)
                logX, _ = conditional_clocks(u, C.ranks, rng)
                _, ucf = C.utilities(th, n_alpha, np.ones(len(names), bool))
                won[kind].append(int(np.argmax(C.standing(rerank(logX, ucf, m), base, fixed, rng))))
            rows.append({"model": model, "contest": contest, "rule": "sum of ranks" if base is None else f"{base:.4g}",
                         "fitted": base == fitted[C.year], "leader": C.songs[w], "runner_up": C.songs[order[1]],
                         "margin_points": float(obs[w] - obs[order[1]]),
                         "p_flip": float(np.mean(np.array(won["draw"]) != w)),
                         "p_flip_mle": float(np.mean(np.array(won["mle"]) != w)), "draws": n_draws})
        print(contest, flush=True, file=sys.stderr)
    pd.DataFrame(rows).to_csv(OUT / f"bases_{model}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "counterfactual", "design", "bases"])
    ap.add_argument("--model", default="B1")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "fit":
        long, _ = load()
        ests, infos = [], []
        for model in MODELS:
            est, info, theta, V, Vc = fit(long, model)
            ests.append(est)
            infos.append(info)
            theta.to_csv(OUT / f"theta_{model}.csv", index=False)
            np.savez_compressed(OUT / f"cov_{model}.npz", V=V, V_cluster=Vc)
        pd.concat(ests).to_csv(OUT / "estimates.csv", index=False)
        (OUT / "fits.json").write_text(json.dumps(infos, indent=1))
        print(pd.concat(ests).round(3).to_string(index=False))
    elif a.step == "counterfactual":
        counterfactual(a.model, a.draws, a.seed)
    elif a.step == "bases":
        bases(a.model, a.draws, a.seed)
    else:
        design(a.model, a.draws, a.reps, a.seed)


if __name__ == "__main__":
    main()
