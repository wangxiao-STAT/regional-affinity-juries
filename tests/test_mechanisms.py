"""Recovery checks for the transfer design and the foreign-coach term split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fit_pooled import ExplodedLogit  # noqa: E402
from mechanisms import by_award, coach_terms, data_movers, data_transfers, league_timeline  # noqa: E402


def simulate_transfers(seed=0, beta_league=0.8, n_years=8, n_voters=360):
    """Persons persist across contests and some change league; voters from host countries hold a
    lasting taste gamma[p, v] that the pair fixed effects must absorb."""
    rng = np.random.default_rng(seed)
    hosts = ["H0", "H1", "H2", "H3"]
    voters = hosts * 6 + [f"O{i}" for i in range(40)]
    persons = [f"P{j}" for j in range(12)]
    gamma = {(p, v): rng.normal(0, 1.0) for p in persons for v in hosts}
    league = {p: rng.choice(hosts) for p in persons}
    rows, ballot = [], 0
    for y in range(n_years):
        for p in persons[:6]:                       # half the persons may move each year
            if rng.random() < 0.35:
                league[p] = rng.choice(hosts)
        alpha = {p: rng.normal(0, 1.0) for p in persons}
        for i in range(n_voters):
            v = voters[i % len(voters)]
            role = ("captain", "coach", "media")[i % 3]
            u = np.array([alpha[p] + gamma.get((p, v), 0.0) + beta_league * (league[p] == v) for p in persons])
            order = np.argsort(-(u + rng.gumbel(size=len(persons))))[:3]
            rank = np.zeros(len(persons), int)
            rank[order] = [1, 2, 3]
            for j, p in enumerate(persons):
                rows.append({"ballot": ballot, "year": 2010 + y, "award": "MP", "contest": f"{2010 + y}_MP",
                             "role": role, "voter_country": v, "person": p, "rank": rank[j],
                             "same_nation": 0, "same_confed": float(rng.random() < 0.2), "voter_confed": v[0],
                             "same_league": float(league[p] == v), "club_country_set": league[p]})
            ballot += 1
    return pd.DataFrame(rows)


def fit_design(d, X, names, fe):
    keep = X.any(axis=0)
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    m = ExplodedLogit(d, X, names, fe).fit()
    V, _ = m.covariances()
    return m, dict(zip(names, m.theta[m.n_alpha:])), dict(zip(names, np.sqrt(np.diag(V)[m.n_alpha:])))


def test_within_pair_design_recovers_league_effect_despite_lasting_tastes():
    d, X, names, fe, info = data_transfers(simulate_transfers(), within=True)
    m, b, se = fit_design(d, X, names, fe)
    assert info["pairs_with_change"] > 0 and m.opt.success and m.cond < 1e10
    pooled = np.mean([b[f"league_current[{r}]"] for r in ("captain", "coach", "media")])
    assert abs(pooled - 0.8) < 0.25, b
    assert abs(b["league_past"]) < 4 * se["league_past"], (b["league_past"], se["league_past"])


def test_between_design_runs_with_placebo_term():
    d, X, names, fe, info = data_transfers(simulate_transfers(seed=1), within=False)
    m, b, se = fit_design(d, X, names, fe)
    assert m.opt.success and "league_future" in b


def test_timeline_splits_current_past_future():
    long = pd.DataFrame({"award": "MP", "person": "P", "year": [2010, 2012, 2014],
                         "club_country_set": ["Spain", "Spain;France", "England"]})
    tl = league_timeline(long)
    assert tl[("MP", "P", 2010)] == ({"Spain"}, set(), {"France", "England"})
    assert tl[("MP", "P", 2012)] == ({"Spain", "France"}, set(), {"England"})
    assert tl[("MP", "P", 2014)] == ({"England"}, {"Spain", "France"}, set())


def test_coach_terms_split_identity_from_team():
    d = pd.DataFrame({"coach_nationality": ["Germany", "Germany", "Zimbabwe", "Zimbabwe"],
                      "voter_country": ["Zimbabwe", "Zimbabwe", "Zimbabwe", "Zimbabwe"],
                      "nominee_country": ["Germany", "Zimbabwe", "Zimbabwe", "Egypt"],
                      "nominee_confed": ["UEFA", "CAF", "CAF", "CAF"],
                      "same_nation": [0, 1, 1, 0], "same_confed": [0, 1, 1, 1], "same_league": [0, 0, 0, 0],
                      "club_country_set": ["Germany", "England", "Spain", "England"]})
    t = coach_terms(d, {"Germany": "UEFA", "Zimbabwe": "CAF"})
    assert list(t["nation_own_foreign"]) == [1, 0, 0, 0]
    assert list(t["nation_team_foreign"]) == [0, 1, 0, 0]
    assert list(t["nation_team_domestic"]) == [0, 0, 1, 0]
    assert list(t["confed_own_foreign"]) == [1, 0, 0, 0]
    assert list(t["confed_team_foreign"]) == [0, 1, 0, 0]
    assert list(t["league_own_foreign"]) == [1, 0, 0, 0]


def test_movers_get_coach_region_effects_and_drop_absorbed_terms():
    rows = []
    # coach K works in CAF (2018) then AFC (2022); coach L stays in UEFA
    for b, (cid, vc, vconf, year) in enumerate([("K", "Zambia", "CAF", 2017), ("K", "Zambia", "CAF", 2018),
                                                ("K", "Iran", "AFC", 2021), ("K", "Iran", "AFC", 2022),
                                                ("L", "Wales", "UEFA", 2018), ("L", "Wales", "UEFA", 2022)]):
        for j, (p, nc, ncf) in enumerate([("P1", "Egypt", "CAF"), ("P2", "Japan", "AFC"), ("P3", "France", "UEFA")]):
            rows.append({"ballot": b, "contest": f"{year}_MP", "role": "coach", "voter_country": vc,
                         "voter_confed": vconf, "person": p, "nominee_country": nc, "nominee_confed": ncf,
                         "rank": j + 1, "same_nation": 0, "same_confed": int(vconf == ncf), "same_league": 0,
                         "club_country_set": "England"})
    long = pd.DataFrame(rows)
    coach = pd.DataFrame({"ballot": [0, 1, 2, 3, 4, 5],
                          "coach_nationality": ["France"] * 4 + ["Wales"] * 2,
                          "coach_id": ["K"] * 4 + ["L"] * 2})
    d, X, names, fe, info = data_movers(long, coach, {"France": "UEFA", "Wales": "UEFA"})
    assert info["mover_coaches"] == 1 and info["mover_ballots"] == 4
    assert not any(n.startswith("confed_own") and n.endswith("[mover]") for n in names)
    assert "confed_team[mover]" in names
    pair = fe[1][0]
    assert all(p.startswith("coach#K#") for p in pair[d.coach_id.eq("K").to_numpy()])
    assert (pair[d.coach_id.eq("L").to_numpy()] == "").all()


def test_by_award_interacts_every_term():
    d = pd.DataFrame({"award": ["MP", "WP", "MP"]})
    X = np.array([[1.0, 0], [1, 1], [0, 1]])
    Z, names = by_award(d, X, ["a", "b"])
    assert names == ["a[MP]", "b[MP]", "a[WP]", "b[WP]"]
    assert np.array_equal(Z, [[1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]])
