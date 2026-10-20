"""Robustness of the pooled estimates: placebo permutations, strategic downgrading, multiverse.

python -I nhb/src/robustness.py [--perms 100]  ->  nhb/results/robustness/{placebo,strategic,multiverse}.csv

Placebo. Voter countries are shuffled among the ballots of the same contest and juror role
("all"), or of the same contest, role and voter confederation ("within_confed"), and the three
affinities are recomputed from the shuffled countries; the role-specific model (M1) is refitted.
The first shuffle breaks every voter-nominee link, so all coefficients should vanish; the second
keeps each voter's region, so same_confed should survive and same_nation should vanish.
Strategic downgrading. A nominee is a "rival" for a voter when both the nominee and a compatriot
of the voter are among the contest's top three (official standings recomputed from the ballots)
and the nominee is not that compatriot. A negative rival coefficient means voters mark down the
main competitors of their own country's contender.
Multiverse. The compatriot and confederation coefficients by role across fixed-effect structures
(contest, contest x role, contest x voter confederation), samples (all, complete ballots, men,
women, 2010-15, 2016-25, and men 2011-14, the sample of Coupe, Gergaud and Noury 2018) and outcome
(top three ranked, first choice only). The men 2011-14 first-choice fit is the estimand closest to
theirs and is used to reconcile the magnitudes.
Placebo with role-specific nominee effects (placebo_rolefe) and leave-one-association-out influence on
the women's bonuses and closest women's contests (loao_women) answer the first-round review.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pooled import ExplodedLogit, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "robustness"
ROLES = ("captain", "coach", "media")


def features(d, terms):
    cols, names = [], []
    for a in terms:
        for r in ROLES:
            m = d.role.eq(r).to_numpy()
            if m.any():
                cols.append(d[a].to_numpy().astype(float) * m)
                names.append(f"{a}[{r}]")
    X = np.column_stack(cols)
    keep = X.any(axis=0)
    return X[:, keep], [n for n, k in zip(names, keep) if k]


def fit(d, terms, fe_by=()):
    d = d.reset_index(drop=True)
    X, names = features(d, terms)
    group = d.contest.astype(str)
    for c in fe_by:
        group = group + "|" + d[c].astype(str)
    m = ExplodedLogit(d, X, names, [((group + "#" + d.person).to_numpy(), group.to_numpy())]).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    return pd.DataFrame({"term": names, "beta": b, "se_cluster": np.sqrt(np.diag(Vc)[m.n_alpha:])}), m


def recompute_affinity(d, confed):
    d = d.copy()
    d["voter_confed"] = d.voter_country.map(confed)
    d["same_nation"] = (d.voter_country == d.nominee_country).astype(int)
    d["same_confed"] = (d.voter_confed == d.nominee_confed).astype(int)
    d["same_league"] = [int(v in s.split(";")) for v, s in zip(d.voter_country, d.club_country_set)]
    return d


def shuffle_countries(d, rng, within):
    keys = ["contest", "role"] + (["voter_confed"] if within else [])
    b = d.drop_duplicates("ballot")[["ballot", "voter_country"] + keys].copy()
    b["voter_country"] = b.groupby(keys).voter_country.transform(lambda s: rng.permutation(s.to_numpy()))
    return d.drop(columns="voter_country").merge(b[["ballot", "voter_country"]], on="ballot")


def contenders(d):
    pts = d[d["rank"] > 0].assign(p=lambda x: x["rank"].map({1: 5, 2: 3, 3: 1}))
    g = pts.groupby(["contest", "role", "person"]).p.sum()
    share = (g / g.groupby(level=[0, 1]).transform("sum")).groupby(level=[0, 2]).mean()
    top = share.groupby(level=0, group_keys=False).nlargest(3).reset_index()[["contest", "person"]]
    return set(map(tuple, top.to_numpy()))


def loao_women(long, rng, contests=("2018_WP", "2014_WP"), clocks=300):
    """Leave one association out of the women's award: role-by-award bonuses (M2) and, at the refitted
    estimates, the probability (from the ballots' randomness) that removing all bonuses changes the
    jury standing of the closest women's contests."""
    from counterfactuals import conditional_clocks, rerank, rule_scores, scheme
    from fifa_rules import rule_for
    from fit_pooled import SPECS, Model, design
    split, fe_by, aff = SPECS["M2_role_award"]
    rows = []
    women = sorted(long[long.award == "WP"].voter_country.unique())
    for k, country in enumerate([None] + women):
        d = long[~((long.award == "WP") & (long.voter_country == country))].reset_index(drop=True)
        m = Model(d, split, fe_by, aff).fit()
        V, Vc = m.covariances()
        b = dict(zip(m.names, m.theta[m.n_alpha:]))
        row = {"left_out": country or "none"}
        row.update({f"b_{n}": v for n, v in b.items() if n.endswith(":WP]")})
        X, names, fe, _ = design(d, split, fe_by, aff)
        alpha = pd.Series(np.append(m.theta[:m.n_alpha], np.zeros(len(m.ref_names))),
                          index=m.fe_names + m.ref_names)
        for contest in contests:
            c = d[d.contest == contest]
            ballots, noms = np.sort(c.ballot.unique()), np.sort(c.person.unique())
            bi, ji = np.searchsorted(ballots, c.ballot.to_numpy()), np.searchsorted(noms, c.person.to_numpy())
            idx = c.index.to_numpy()
            ranks = np.zeros((len(ballots), len(noms)), int)
            ranks[bi, ji] = c["rank"].to_numpy()
            role = c.groupby("ballot").role.first().reindex(ballots).to_numpy()
            pts, rule = scheme(contest), rule_for(contest)
            w = rule_scores(ranks, role, pts, rule).argmax()
            a_ = fe.iloc[idx].map(alpha).fillna(-np.inf).to_numpy()
            u = np.full(ranks.shape, -np.inf)
            u[bi, ji] = a_ + X[idx] @ m.theta[m.n_alpha:]
            ucf = np.full(ranks.shape, -np.inf)
            ucf[bi, ji] = a_
            flips = []
            for _ in range(clocks):
                Xc, _ = conditional_clocks(u, ranks, rng)
                flips.append(rule_scores(rerank(Xc, ucf, ranks.max(axis=1)), role, pts, rule).argmax() != w)
            row[f"p_flip_mle_{contest}"] = float(np.mean(flips))
        rows.append(row)
        print(f"loao {k}/{len(women)} {country}", file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "loao_women.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=["placebo", "strategic", "multiverse"],
                    help="also: placebo_rolefe (placebo with contest x role x nominee effects), loao_women")
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    terms = ["same_nation", "same_confed", "same_league"]
    rng = np.random.default_rng(a.seed)

    if "placebo" in a.only:
        rows = []
        for within in (False, True):
            for p in range(a.perms):
                t = time.time()
                d = recompute_affinity(shuffle_countries(long, rng, within), confed)
                e, _ = fit(d, terms)
                e["perm"], e["shuffle"] = p, "within_confed" if within else "all"
                rows.append(e)
                print(f"placebo within={within} {p}: {time.time() - t:.1f} s", file=sys.stderr, flush=True)
        pd.concat(rows).to_csv(OUT / "placebo.csv", index=False)

    if "placebo_rolefe" in a.only:
        # nominee effects specific to each profession absorb role-level tastes, which the shared
        # effects leave in the shuffled regional coefficients of single professions
        rows = []
        for within in (False, True):
            for p in range(a.perms):
                d = recompute_affinity(shuffle_countries(long, rng, within), confed)
                e, _ = fit(d, terms, fe_by=("role",))
                e["perm"], e["shuffle"] = p, "within_confed" if within else "all"
                rows.append(e)
                print(f"placebo_rolefe within={within} {p}", file=sys.stderr, flush=True)
        pd.concat(rows).to_csv(OUT / "placebo_rolefe.csv", index=False)

    if "loao_women" in a.only:
        loao_women(long, rng)

    if "strategic" in a.only:
        top = contenders(long)
        d = long.copy()
        d["contender"] = [(c, p) in top for c, p in zip(d.contest, d.person)]
        comp = d[d.contender & d.same_nation.eq(1)].groupby("ballot").size()
        d["has_compatriot_contender"] = d.ballot.isin(comp.index)
        d["rival"] = (d.contender & d.has_compatriot_contender & d.same_nation.eq(0)).astype(int)
        e, m = fit(d, terms + ["rival"])
        e["n_rival_rows"] = [int(d.loc[d.role.eq(t.split("[")[1][:-1]), "rival"].sum()) if t.startswith("rival")
                             else np.nan for t in e.term]
        e.to_csv(OUT / "strategic.csv", index=False)
        print(e.round(3).to_string(index=False), file=sys.stderr)

    if "multiverse" in a.only:
        samples = {"all": long, "complete": long[long.groupby("ballot")["rank"].transform("max") == 3],
                   "men": long[long.award == "MP"], "women": long[long.award == "WP"],
                   "2010-15": long[long.year <= 2015], "2016-25": long[long.year >= 2016],
                   "men2011-14": long[(long.award == "MP") & long.year.between(2011, 2014)]}
        rows = []
        for sname, s in samples.items():
            for outcome in ("top3", "first"):
                ss = s if outcome == "top3" else s.assign(rank=np.where(s["rank"] == 1, 1, 0))
                for fe_by in ((), ("role",), ("voter_confed",)):
                    tt = terms if "voter_confed" not in fe_by else ["same_nation", "same_league"]
                    e, m = fit(ss, tt, fe_by)
                    e["sample"], e["outcome"], e["fe"] = sname, outcome, "+".join(("contest",) + fe_by)
                    e["converged"] = bool(m.opt.success)
                    rows.append(e)
                    print(f"multiverse {sname} {outcome} {fe_by}", file=sys.stderr, flush=True)
        pd.concat(rows).to_csv(OUT / "multiverse.csv", index=False)


if __name__ == "__main__":
    main()
