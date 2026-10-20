"""FIFA and Ballon d'Or ballots in one model: four juror types, one likelihood.

python -I nhb/src/fit_combined.py [--only C1_role ...]  ->  nhb/results/combined/estimates.csv, fits.json,
                                                           theta_{model}.csv, cov_{model}.npz

The FIFA ballots (captains, coaches, media; top 3) and the France Football Ballon d'Or ballots
(journalists, "bdo"; top 5 or top 10) enter the same exploded logit, each contest with its own
nominee effects, so the affinity coefficients of the four juror types are estimated on one scale.
Specifications:
  C1_role          affinities by juror type, contest-by-nominee effects, all 40 contests
  C1_role_award    affinities by juror type and award (men, women)
  C1_same_years    C1 on the years with both juries (2019, 2022-2025), to compare FIFA and Ballon
                   d'Or journalists over the same period and nominee pools
  C4_role_confFE   contest-by-nominee-by-voter-confederation effects (same_confed not identified)
  C5_bdo_depth3    C1 with every Ballon d'Or ballot cut to its first three picks, so that a deeper
                   ballot cannot explain a difference between the juries
  C2_confhet       C1 with the regional bonus specific to the juror's confederation, separately for
                   FIFA's jurors and the Ballon d'Or's
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pooled import AFF, AFF_WITHIN, EXTREME_ALPHA, ExplodedLogit, design, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "combined"
SAME_YEARS = {2019, 2022, 2023, 2024, 2025}
# label: (affinity split, fixed-effect group columns, affinity terms, sample)
CSPECS = {
    "C1_role": ("role", (), AFF, "all"),
    "C1_role_award": ("role_award", (), AFF, "all"),
    "C1_same_years": ("role", (), AFF, "same_years"),
    "C4_role_confFE": ("role", ("voter_confed",), AFF_WITHIN, "all"),
    "C5_bdo_depth3": ("role", (), AFF, "depth3"),
    "C2_confhet": ({"same_nation": "role", "same_confed": "org_confed", "same_league": "role"}, (), AFF, "all"),
}


def load():
    fifa = pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False)
    bdo = pd.read_csv(PROC / "bdo_long.csv", keep_default_na=False)
    cols = [c for c in fifa.columns if c in bdo.columns]
    return prepare(pd.concat([fifa[cols], bdo[cols]], ignore_index=True))


def build(label, long=None):
    split, fe_by, aff, sample = CSPECS[label]
    d = load() if long is None else long
    if sample == "same_years":
        d = d[d.year.isin(SAME_YEARS)]
    elif sample == "depth3":
        d = d.assign(rank=np.where(d.role.eq("bdo") & (d["rank"] > 3), 0, d["rank"]))
    d = d.reset_index(drop=True)
    X, names, fe, group = design(d, split, fe_by, aff)
    return d, X, names, [(fe.to_numpy(), group.to_numpy())]


def run(label, long, out=OUT):
    d, X, names, fe_sets = build(label, long)
    m = ExplodedLogit(d, X, names, fe_sets).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    est = pd.DataFrame({"model": label, "term": names, "beta": b, "se": np.sqrt(np.diag(V)[m.n_alpha:]),
                        "se_cluster": np.sqrt(np.diag(Vc)[m.n_alpha:])})
    est["odds_ratio"], est["z_cluster"] = np.exp(est.beta), est.beta / est.se_cluster
    info = {"model": label, "loglik": m.ll, "n_alpha": m.n_alpha, "n_beta": m.n_beta,
            "ballots": int(d.ballot.nunique()), "contests": int(d.contest.nunique()),
            "n_choices": int(m.chosen.sum()), "rows_dropped_zero_pick_cells": m.n_dropped,
            "converged": bool(m.opt.success), "grad_max": float(np.abs(m.opt.jac).max()),
            "extreme_alpha_cells": int((np.abs(m.theta[:m.n_alpha]) > EXTREME_ALPHA).sum()),
            "hessian_cond": m.cond}
    print(f"{label}: ballots={info['ballots']} ll={m.ll:.1f} conv={m.opt.success} "
          f"|grad|max={info['grad_max']:.1e} cond={m.cond:.1e}", file=sys.stderr, flush=True)
    pd.DataFrame({"param": m.fe_names + m.ref_names + m.names,
                  "kind": ["alpha"] * m.n_alpha + ["ref"] * len(m.ref_names) + ["beta"] * m.n_beta,
                  "value": np.concatenate([m.theta[:m.n_alpha], np.zeros(len(m.ref_names)), b])}) \
        .to_csv(out / f"theta_{label}.csv", index=False)
    np.savez_compressed(out / f"cov_{label}.npz", V=V, V_cluster=Vc)
    return est, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(CSPECS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = load()
    ests, infos = [], []
    for label in a.only:
        e, i = run(label, long)
        ests.append(e)
        infos.append(i)
    est = pd.concat(ests)
    est.to_csv(OUT / "estimates.csv", index=False)
    (OUT / "fits.json").write_text(json.dumps(infos, indent=1))
    pd.set_option("display.width", 200)
    print(est.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
