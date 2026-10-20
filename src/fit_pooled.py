"""First pooled estimates: top-3 exploded logit with contest-by-nominee fixed effects.

python -I nhb/src/fit_pooled.py [--only M1_role ...]  ->  nhb/results/pooled/estimates.csv, fits.json,
                                                         theta_{model}.csv, cov_{model}.npz

Utility of nominee j for voter i in contest c: u_ij = alpha_{c,j,g(i)} + sum_k beta_k x_ijk, with
affinities x = same nation, same confederation, same league country, and g(i) the voter's
fixed-effect group: none, juror role ("_rfe"), or the voter's confederation ("_confFE", which
absorbs each nominee's regional appeal, so same_confed is not identified there and is left out).
same_confed includes compatriots, so beta_confed is the bonus for a different nation of the same
confederation and beta_nation the further bonus for a compatriot. A ballot's ranked picks are the
successive Plackett-Luce choices from the nominees not yet chosen. A fixed-effect cell with no
pick has alpha = -inf at the maximum, so its rows are dropped (exactly); in each group the
most-picked nominee is the reference (alpha = 0). Standard errors: inverse observed information,
and a sandwich clustered by voter country (the same seat across years and roles).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "pooled"
AFF = ["same_nation", "same_confed", "same_league"]
AFF_WITHIN = ["same_nation", "same_league"]
# label: (affinity split, fixed-effect group columns, affinity terms)
SPECS = {
    "M0_pooled": (None, (), AFF),
    "M1_role": ("role", (), AFF),
    "M2_role_award": ("role_award", (), AFF),
    "M3_role_era": ("role_era", (), AFF),
    "M1_role_rfe": ("role", ("role",), AFF),
    "M2_role_award_rfe": ("role_award", ("role",), AFF),
    "M4_role_confFE": ("role", ("voter_confed",), AFF_WITHIN),
    "M4_role_award_confFE": ("role_award", ("voter_confed",), AFF_WITHIN),
    # regional bonus specific to the juror's confederation (and to role x confederation)
    "M5_role_confhet": ({"same_nation": "role", "same_confed": "voter_confed", "same_league": "role"}, (), AFF),
    "M6_role_confhet_role": ({"same_nation": "role", "same_confed": "role_confed", "same_league": "role"}, (), AFF),
}
MIN_GROUP = 10   # in contest-by-juror-confederation effects, smaller groups are pooled within the contest
EXTREME_ALPHA = 30   # |nominee effect| above this signals a separated cell (reported in fits.json)


def prepare(long):
    long = long.copy()
    long["era"] = np.where(long.year <= 2015, "2010-15", "2016-25")
    long["role_award"] = long.role + ":" + long.award
    long["role_era"] = long.role + ":" + long.era
    long["role_confed"] = long.role + ":" + long.voter_confed
    long["org_confed"] = np.where(long.role.eq("bdo"), "bdo", "fifa") + ":" + long.voter_confed
    return long


def design(long, split, fe_by=(), aff=AFF):
    """Affinity columns interacted with `split` (a column for all terms, or a dict term -> column);
    fixed-effect cell and group labels. Terms that are zero for every row are dropped. In
    contest-by-juror-confederation effects, a confederation
    with fewer than MIN_GROUP ballots in a contest is pooled with the contest's other small groups,
    because a handful of jurors can rank a nominee in every possible way and separate the cell."""
    cols, names = [], []
    for a in aff:
        col = split.get(a) if isinstance(split, dict) else split
        levels = [None] if col is None else sorted(long[col].unique())
        for lv in levels:
            m = np.ones(len(long), bool) if lv is None else (long[col] == lv).to_numpy()
            cols.append(long[a].to_numpy() * m)
            names.append(a if lv is None else f"{a}[{lv}]")
    X = np.column_stack(cols).astype(float)
    keep = X.any(axis=0)       # e.g. the regional bonus of Oceanian jurors: no nominee is from OFC
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    group = long.contest.astype(str)
    for c in fe_by:
        lab = long[c].astype(str)
        if c == "voter_confed":
            # Ballon d'Or juries are small outside Europe: their jurors are grouped as European or not,
            # because a few jurors from one region can all rank the same nominee first (separation)
            bdo = long.role.eq("bdo").to_numpy()
            lab = lab.where(~bdo, np.where(long.voter_confed.eq("UEFA"), "UEFA", "nonUEFA"))
            n = long.assign(_g=lab).groupby(["contest", "_g"]).ballot.transform("nunique")
            lab = lab.where(n >= MIN_GROUP, "small")
        group = group + "|" + lab
    fe = group + "#" + long.person
    return X, names, fe, group


def stages(long):
    """Replicate ballot-nominee rows into the risk sets of the successive picks."""
    nvalid = long.groupby("ballot")["rank"].transform("max").to_numpy()
    rank = long["rank"].to_numpy()
    depth = int(rank.max())                       # 3 for FIFA ballots, up to 10 for the Ballon d'Or
    rows, grp, chosen = [], [], []
    for s in range(1, depth + 1):
        keep = np.flatnonzero((nvalid >= s) & ((rank == 0) | (rank >= s)))
        rows.append(keep)
        grp.append(long.ballot.to_numpy()[keep].astype(np.int64) * depth + (s - 1))
        chosen.append(rank[keep] == s)
    rows, grp, chosen = map(np.concatenate, (rows, grp, chosen))
    order = np.lexsort((rows, grp))
    rows, grp, chosen = rows[order], grp[order], chosen[order]
    _, g = np.unique(grp, return_inverse=True)
    return rows, g, chosen


def hybrid_draws(mle, V_model, V_cluster, n_alpha, n, rng):
    """Parameter draws: the similarity coefficients from the clustered covariance (whose rank is at
    most the number of clusters, too low for the hundreds of nominee effects), and the nominee
    effects from their model-based distribution conditional on those coefficients."""
    a, b = slice(0, n_alpha), slice(n_alpha, None)
    Lb = psd_sqrt(V_cluster[b, b])
    K = V_model[a, b] @ np.linalg.inv(V_model[b, b])
    La = psd_sqrt(V_model[a, a] - K @ V_model[a, b].T)
    out = []
    for _ in range(n):
        db = Lb @ rng.standard_normal(Lb.shape[1])
        da = K @ db + La @ rng.standard_normal(La.shape[1])
        out.append(mle + np.concatenate([da, db]))
    return out


def psd_sqrt(V):
    """A square root of a covariance that may be singular (a clustered sandwich has rank at most the
    number of clusters): eigenvalues below zero from rounding are set to zero."""
    w, U = np.linalg.eigh((V + V.T) / 2)
    return U * np.sqrt(np.clip(w, 0, None))


class ExplodedLogit:
    """Top-k exploded logit with several fixed-effect sets.

    fe_sets: list of (labels, groups) arrays aligned with `long`; a row with label "" has no
    effect in that set. Cells without a pick are dropped (alpha = -inf at the maximum); in each
    group the most-picked cell is the reference (alpha = 0).
    """

    def __init__(self, long, X, names, fe_sets):
        self.names = list(names)
        picked = long["rank"].to_numpy() > 0
        keep = np.ones(len(long), bool)
        for labels, _ in fe_sets:
            labels = np.asarray(labels)
            cell = pd.Series(picked).groupby(labels).transform("sum").to_numpy()
            keep &= (labels == "") | (cell > 0)
        self.n_dropped = int((~keep).sum())
        long, X = long[keep].reset_index(drop=True), X[keep]
        fe_sets = [(np.asarray(lab)[keep], np.asarray(grp)[keep]) for lab, grp in fe_sets]
        self.rows, self.g, self.chosen = stages(long)
        picked = long["rank"].to_numpy() > 0
        blocks, self.fe_names, self.ref_names = [], [], []
        n = len(self.rows)
        for labels, groups in fe_sets:
            has = labels != ""
            codes, uniq = pd.factorize(labels[has])
            picks = pd.Series(picked[has]).groupby(codes).sum()
            ugroup = pd.Series(groups[has]).groupby(codes).first().to_numpy()
            ref = picks.groupby(ugroup).idxmax().to_numpy()
            free = np.setdiff1d(np.arange(len(uniq)), ref)
            col = np.full(len(uniq), -1)
            col[free] = np.arange(len(free))
            full = np.full(len(labels), -1)
            full[has] = col[codes]
            a = full[self.rows]
            blocks.append(sp.csr_matrix((np.ones((a >= 0).sum()), (np.flatnonzero(a >= 0), a[a >= 0])),
                                        shape=(n, len(free))))
            self.fe_names += list(uniq[free])
            self.ref_names += list(uniq[ref])
        self.n_alpha, self.n_beta = len(self.fe_names), X.shape[1]
        self.Z = sp.hstack(blocks + [sp.csr_matrix(X[self.rows])]).tocsr()
        self.G = sp.csr_matrix((np.ones(n), (self.g, np.arange(n))))
        self.n_groups = self.G.shape[0]
        self.cluster = long.voter_country.to_numpy()[self.rows]

    def probs(self, theta):
        u = self.Z @ theta
        m = np.full(self.n_groups, -np.inf)
        np.maximum.at(m, self.g, u)
        e = np.exp(u - m[self.g])
        den = np.bincount(self.g, e, minlength=self.n_groups)
        return u, e / den[self.g], np.log(den) + m

    def nll(self, theta):
        u, p, lse = self.probs(theta)
        ll = u[self.chosen].sum() - lse.sum()
        grad = self.Z.T @ (self.chosen - p)
        return -ll, -np.asarray(grad).ravel()

    def fit(self):
        theta0 = np.zeros(self.n_alpha + self.n_beta)
        opts = {"maxiter": 50000, "maxfun": 100000, "ftol": 1e-15, "gtol": 1e-7}
        r = minimize(self.nll, theta0, jac=True, method="L-BFGS-B", options=opts)
        self.theta, self.ll, self.opt = r.x, -r.fun, r
        return self

    def covariances(self):
        _, p, _ = self.probs(self.theta)
        P = sp.diags(p)
        M = self.G @ P @ self.Z
        H = (self.Z.T @ P @ self.Z - M.T @ M).toarray()
        Hinv = np.linalg.inv(H)
        resid = sp.diags(self.chosen - p) @ self.Z          # per-row score contributions
        cl_codes, _ = pd.factorize(self.cluster)
        C = sp.csr_matrix((np.ones(len(cl_codes)), (cl_codes, np.arange(len(cl_codes)))))
        S = (C @ resid).toarray()
        k = C.shape[0]
        self.cond = float(np.linalg.cond(H))
        return Hinv, Hinv @ (S.T @ S) @ Hinv * k / (k - 1)


class Model(ExplodedLogit):
    """The pooled specifications: affinities split by `split`, one fixed-effect set."""

    def __init__(self, long, split=None, fe_by=(), aff=AFF):
        X, names, fe, group = design(long, split, fe_by, aff)
        super().__init__(long, X, names, [(fe.to_numpy(), group.to_numpy())])


def run(long, label, out=OUT):
    split, fe_by, aff = SPECS[label]
    t = time.time()
    m = Model(long, split, fe_by, aff).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    se = np.sqrt(np.diag(V)[m.n_alpha:])
    sec = np.sqrt(np.diag(Vc)[m.n_alpha:])
    est = pd.DataFrame({"model": label, "term": m.names, "beta": b, "se": se, "se_cluster": sec,
                        "odds_ratio": np.exp(b), "z_cluster": b / sec})
    info = {"model": label, "loglik": m.ll, "n_alpha": m.n_alpha, "n_beta": m.n_beta,
            "n_choices": int(m.chosen.sum()), "rows_dropped_zero_pick_cells": m.n_dropped,
            "converged": bool(m.opt.success), "message": str(m.opt.message),
            "extreme_alpha_cells": int((np.abs(m.theta[:m.n_alpha]) > EXTREME_ALPHA).sum()),
            "grad_max": float(np.abs(m.opt.jac).max()), "hessian_cond": m.cond, "seconds": round(time.time() - t, 1)}
    print(f"{label}: ll={m.ll:.2f} conv={m.opt.success} |grad|max={info['grad_max']:.2e} "
          f"dropped={m.n_dropped} cond={m.cond:.1e} ({info['seconds']} s)", file=sys.stderr)
    pd.DataFrame({"param": m.fe_names + m.ref_names + m.names,
                  "kind": ["alpha"] * m.n_alpha + ["ref"] * len(m.ref_names) + ["beta"] * m.n_beta,
                  "value": np.concatenate([m.theta[:m.n_alpha], np.zeros(len(m.ref_names)), b])}) \
        .to_csv(out / f"theta_{label}.csv", index=False)
    np.savez_compressed(out / f"cov_{label}.npz", V=V, V_cluster=Vc)
    return est, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SPECS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
    ests, infos = [], []
    for label in a.only:
        e, i = run(long, label)
        ests.append(e)
        infos.append(i)
    est = pd.concat(ests)
    est.to_csv(OUT / "estimates.csv", index=False)
    (OUT / "fits.json").write_text(json.dumps(infos, indent=1))
    pd.set_option("display.width", 200)
    print(est.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
