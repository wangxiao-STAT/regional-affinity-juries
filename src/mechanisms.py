"""Mechanism tests: foreign national-team coaches and nominee transfers.

python -I nhb/src/mechanisms.py [--only A1_coaches ...]  ->  nhb/results/mechanisms/estimates.csv,
                                                            contrasts.csv, fits.json

A. Foreign coaches (coach ballots whose coach nationality is known: coach_history.py, men's and
   women's awards; A1m and A1w by award; A1x both awards in one fit with every term by award, so
   that a contrast can be compared across awards). Each affinity is split three ways: with the team's country for domestic
   coaches (identity and environment together), with the team's country for foreign coaches
   (environment or representation only), and with the coach's own nationality for foreign coaches
   (identity only). A1 uses the coach ballots alone with contest-by-nominee effects; A2 adds all
   captain and media ballots of the men's awards, sharing the contest-by-nominee effects across
   roles so that their picks help pin down each nominee's baseline appeal. (With role-specific
   effects the coach ballots would form a separate likelihood and A2 would reproduce A1 exactly.)
   A3 (within coach). Coaches who voted while working in two or more confederations get a fixed
   effect per coach and nominee confederation, which absorbs their lasting regional tastes
   (including any taste for their own region of origin); their workplace-region bonus is then
   identified only by the same coach changing region. Other coaches' ballots stay in, with their
   own terms, to pin down the nominee effects.
B. Transfers. For a voter country that hosts the nominee's club in some contest, each contest is
   "future" (the move is still to come), "current" (the voter's country is among the nominee's
   club countries) or "past" (only an earlier contest's club country). B1 adds a fixed effect per
   nominee and host voter country, which absorbs any lasting bilateral taste; the three states then
   sum to one within the pair, so "future" is the baseline and the league effect is identified only
   by nominees whose clubs changed country (current: joining; past: after leaving). B2 drops the
   pair effects for nominee-by-voter-confederation effects and keeps all three states: a positive
   "future" there means selection (players moving to leagues whose countries already favoured them)
   rather than exposure. Contests are those in which the person was nominated; club years without
   a nomination are unknown.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_pooled import EXTREME_ALPHA, ExplodedLogit, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "mechanisms"
ROLES = ("captain", "coach", "media")
MIN_STINT = 2   # ballots per confederation for a coach to count as a mover
CONTRASTS = [("nation_own_foreign", "nation_team_foreign"), ("confed_own_foreign", "confed_team_foreign"),
             ("league_own_foreign", "league_team_foreign"), ("nation_team_domestic", "nation_team_foreign"),
             ("confed_team_domestic", "confed_team_foreign"),
             ("confed_team[mover]", "confed_team_foreign[other]")]
# A1x: does a contrast differ between the awards? (women's minus men's, from one joint fit)
AWARD_DIFFS = [("nation_team_domestic", "nation_team_foreign"), ("nation_own_foreign", "nation_team_foreign"),
               ("confed_own_foreign", "confed_team_foreign")]


def by_award(d, X, names):
    """Every term interacted with the award, so that one fit (one clustered covariance) gives both."""
    cols, out = [], []
    for a in ("MP", "WP"):
        m = d.award.eq(a).to_numpy()
        cols.append(X * m[:, None])
        out += [f"{n}[{a}]" for n in names]
    return np.hstack(cols), out


def coach_terms(d, confed):
    nat = d.coach_nationality
    dom = (nat == d.voter_country).to_numpy()
    own_nation = (nat == d.nominee_country).to_numpy()
    own_confed = (nat.map(confed) == d.nominee_confed).to_numpy()
    own_league = np.array([n in s.split(";") for n, s in zip(nat, d.club_country_set)])
    t = {"nation_team_domestic": d.same_nation * dom, "nation_team_foreign": d.same_nation * ~dom,
         "nation_own_foreign": own_nation * ~dom,
         "confed_team_domestic": d.same_confed * dom, "confed_team_foreign": d.same_confed * ~dom,
         "confed_own_foreign": own_confed * ~dom,
         "league_team_domestic": d.same_league * dom, "league_team_foreign": d.same_league * ~dom,
         "league_own_foreign": own_league * ~dom}
    return {k: np.asarray(v, float) for k, v in t.items()}


def load_coaches():
    path = PROC / "fifa_coach_nationality_all.csv"
    if not path.exists():
        path = PROC / "fifa_coach_nationality.csv"
    return pd.read_csv(path, keep_default_na=False)


def data_coaches(long, coach, confed, with_others, awards=("MP", "WP")):
    men = long[long.award.isin(awards)]
    known = coach[coach.coach_nationality != ""].set_index("ballot").coach_nationality
    is_coach = men.role == "coach"
    d = men[(~is_coach & with_others) | (is_coach & men.ballot.isin(known.index))].copy()
    d["coach_nationality"] = d.ballot.map(known).fillna("")
    cols, names = [], []
    c = d.role.eq("coach").to_numpy()
    for k, v in coach_terms(d, confed).items():
        cols.append(v * c)
        names.append(k)
    if with_others:
        for r in ("captain", "media"):
            m = d.role.eq(r).to_numpy()
            for a in ("same_nation", "same_confed", "same_league"):
                cols.append(d[a].to_numpy() * m)
                names.append(f"{a}[{r}]")
    X = np.column_stack(cols)
    group = d.contest
    fe = [((group + "#" + d.person).to_numpy(), group.to_numpy())]
    return d.reset_index(drop=True), X, names, fe


def data_movers(long, coach, confed):
    known = coach[coach.coach_nationality != ""].set_index("ballot")
    d = long[(long.role == "coach") & long.ballot.isin(known.index)].copy()
    d["coach_nationality"] = d.ballot.map(known.coach_nationality)
    d["coach_id"] = d.ballot.map(known.coach_id)
    # movers: coaches with at least MIN_STINT ballots in each of two or more confederations, so that
    # each stint carries information about the workplace bonus (single-ballot stints made the
    # coach-by-region effects nearly unidentified in the first version)
    stints = d.drop_duplicates("ballot").groupby(["coach_id", "voter_confed"]).size()
    regions = (stints >= MIN_STINT).groupby(level=0).sum()
    movers = set(regions[regions >= 2].index)
    mv = d.coach_id.isin(movers).to_numpy()
    t = coach_terms(d, confed)
    cols, names = [], []
    # movers: one workplace term per affinity, whether the stint is at home or abroad, so that the
    # within-coach contrast uses every stint; own-nationality terms as for other foreign coaches
    for a, short in (("same_nation", "nation"), ("same_confed", "confed"), ("same_league", "league")):
        cols.append(d[a].to_numpy().astype(float) * mv)
        names.append(f"{short}_team[mover]")
    for k in ("nation_own_foreign", "league_own_foreign"):   # confed_own is absorbed by the pair effects
        cols.append(t[k] * mv)
        names.append(f"{k}[mover]")
    for k, v in t.items():
        cols.append(v * ~mv)
        names.append(f"{k}[other]")
    X = np.column_stack(cols)
    keep = X.any(axis=0)
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    primary = ((d.contest + "#" + d.person).to_numpy(), d.contest.to_numpy())
    pair = np.where(mv, "coach#" + d.coach_id + "#" + d.nominee_confed, "")
    pair_group = np.where(mv, d.coach_id, "")
    info = {"mover_coaches": len(movers), "mover_ballots": int(d[mv].ballot.nunique()),
            "other_ballots": int(d[~mv].ballot.nunique())}
    return d.reset_index(drop=True), X, names, [primary, (pair, pair_group)], info


def league_timeline(long):
    """Per (award, person, contest year): current, earlier-only and later-only club countries."""
    t = long.drop_duplicates(["award", "person", "year"])[["award", "person", "year", "club_country_set"]]
    t = t.assign(clubs=t.club_country_set.str.split(";").map(lambda x: {c for c in x if c}))
    out = {}
    for (award, person), g in t.sort_values("year").groupby(["award", "person"]):
        years, sets = g.year.to_list(), g.clubs.to_list()
        for i, y in enumerate(years):
            past = set().union(*sets[:i]) - sets[i]
            future = set().union(*sets[i + 1:]) - sets[i] - past
            out[(award, person, y)] = (sets[i], past, future)
    return out


def data_transfers(long, within=True):
    d = long.copy().reset_index(drop=True)
    tl = league_timeline(d)
    key = list(zip(d.award, d.person, d.year))
    cur = np.array([v in tl[k][0] for v, k in zip(d.voter_country, key)], float)
    past = np.array([v in tl[k][1] for v, k in zip(d.voter_country, key)], float)
    fut = np.array([v in tl[k][2] for v, k in zip(d.voter_country, key)], float)
    hosts = set().union(*[s for s, _, _ in tl.values()])
    cols, names = [], []
    for r in ROLES:
        m = d.role.eq(r).to_numpy()
        cols.append(cur * m)
        names.append(f"league_current[{r}]")
    cols.append(past)
    names.append("league_past")
    if not within:
        cols.append(fut)
        names.append("league_future")
    for r in ROLES:
        m = d.role.eq(r).to_numpy()
        for a in ("same_nation",) + (("same_confed",) if within else ()):
            cols.append(d[a].to_numpy() * m)
            names.append(f"{a}[{r}]")
    X = np.column_stack(cols)
    in_host = d.voter_country.isin(hosts).to_numpy()
    pair = np.where(in_host, "pair#" + d.award + "#" + d.person + "#" + d.voter_country, "")
    # identifying variation: host-country pairs whose state changes across contests
    state = cur + 2 * past
    v = pd.DataFrame({"pair": pair, "state": state})[pair != ""].groupby("pair").state.nunique()
    info = {"host_countries": len(hosts), "pairs": int(len(v)), "pairs_with_change": int((v > 1).sum())}
    if within:
        primary = ((d.contest + "#" + d.person).to_numpy(), d.contest.to_numpy())
        pair_group = np.where(in_host, d.award + "#" + d.voter_country, "")
        return d, X, names, [primary, (pair, pair_group)], info
    group = d.contest + "|" + d.voter_confed
    return d, X, names, [((group + "#" + d.person).to_numpy(), group.to_numpy())], info


def fit(label, d, X, names, fe, extra=None):
    t = time.time()
    keep = X.any(axis=0)
    if not keep.all():
        print(f"{label}: dropping all-zero terms {[n for n, k in zip(names, keep) if not k]}", file=sys.stderr)
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    m = ExplodedLogit(d, X, names, fe).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    Vb, Vcb = V[m.n_alpha:, m.n_alpha:], Vc[m.n_alpha:, m.n_alpha:]
    nonzero = [int((X[:, k] > 0).sum()) for k in range(X.shape[1])]
    est = pd.DataFrame({"model": label, "term": names, "beta": b, "se": np.sqrt(np.diag(Vb)),
                        "se_cluster": np.sqrt(np.diag(Vcb)), "rows_nonzero": nonzero})
    est["z_cluster"] = est.beta / est.se_cluster
    con = []

    def combo(name, w):           # linear combination w of the coefficients, clustered s.e.
        v = np.zeros(len(names))
        for t, x in w.items():
            v[names.index(t)] = x
        diff, se = v @ b, np.sqrt(v @ Vcb @ v)
        con.append({"model": label, "contrast": name, "diff": diff, "se_cluster": se, "z": diff / se})
    for a, c in CONTRASTS:
        for sfx in ("", "[MP]", "[WP]"):
            if a + sfx in names and c + sfx in names:
                combo(f"{a + sfx} - {c + sfx}", {a + sfx: 1, c + sfx: -1})
    for a, c in AWARD_DIFFS:
        terms = [f"{a}[WP]", f"{c}[WP]", f"{a}[MP]", f"{c}[MP]"]
        if all(t in names for t in terms):
            combo(f"({a} - {c})[WP] - [MP]", dict(zip(terms, (1, -1, -1, 1))))
    info = {"model": label, "loglik": m.ll, "n_alpha": m.n_alpha, "n_beta": m.n_beta,
            "ballots": int(d.ballot.nunique()), "n_choices": int(m.chosen.sum()),
            "rows_dropped_zero_pick_cells": m.n_dropped, "converged": bool(m.opt.success),
            "grad_max": float(np.abs(m.opt.jac).max()), "hessian_cond": m.cond,
            "extreme_alpha_cells": int((np.abs(m.theta[:m.n_alpha]) > EXTREME_ALPHA).sum()),
            "max_abs_alpha": float(np.abs(m.theta[:m.n_alpha]).max()),
            "seconds": round(time.time() - t, 1), **(extra or {})}
    print(f"{label}: ballots={info['ballots']} ll={m.ll:.2f} conv={m.opt.success} "
          f"|grad|max={info['grad_max']:.1e} cond={m.cond:.1e} ({info['seconds']} s)", file=sys.stderr)
    return est, pd.DataFrame(con), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["A1_coaches", "A1m_coaches", "A1w_coaches", "A1x_coaches", "A2_all",
                                                   "A3_movers", "B1_transfers_within", "B2_transfers_between"])
    ap.add_argument("--contests", nargs="*", help="restrict to these contests (smoke tests)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    long = prepare(pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False))
    if a.contests:
        long = long[long.contest.isin(a.contests)]
    coach = load_coaches()
    countries = pd.read_csv(PROC / "countries.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    ests, cons, infos = [], [], []
    for label in a.only:
        if label in ("A1_coaches", "A1m_coaches", "A1w_coaches"):
            awards = {"A1_coaches": ("MP", "WP"), "A1m_coaches": ("MP",), "A1w_coaches": ("WP",)}[label]
            e, c, i = fit(label, *data_coaches(long, coach, confed, with_others=False, awards=awards))
        elif label == "A1x_coaches":
            d, X, names, fe = data_coaches(long, coach, confed, with_others=False)
            X, names = by_award(d, X, names)
            e, c, i = fit(label, d, X, names, fe)
        elif label == "A2_all":
            e, c, i = fit(label, *data_coaches(long, coach, confed, with_others=True))
        elif label == "A3_movers":
            d, X, names, fe, extra = data_movers(long, coach, confed)
            e, c, i = fit(label, d, X, names, fe, extra)
        elif label in ("B1_transfers_within", "B2_transfers_between"):
            d, X, names, fe, extra = data_transfers(long, within=label.startswith("B1"))
            e, c, i = fit(label, d, X, names, fe, extra)
        else:
            raise ValueError(label)
        ests.append(e)
        cons.append(c)
        infos.append(i)
    est = pd.concat(ests)
    est.to_csv(out / "estimates.csv", index=False)
    pd.concat(cons).to_csv(out / "contrasts.csv", index=False)
    (out / "fits.json").write_text(json.dumps(infos, indent=1))
    pd.set_option("display.width", 200)
    print(est.round(3).to_string(index=False))
    print(pd.concat(cons).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
