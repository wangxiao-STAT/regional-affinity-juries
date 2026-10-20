"""Exactness checks for the paired counterfactual sampler."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from counterfactuals import conditional_clocks, rerank, rule_percent, standings  # noqa: E402


def random_ballots(rng, B=300, J=8):
    u = rng.normal(0, 1.5, (B, J))
    u[rng.random((B, J)) < 0.05] = -np.inf             # some nominees outside the choice set
    u[:, 0] = np.where(np.isfinite(u).sum(axis=1) < 4, 0.0, u[:, 0])
    g = np.where(np.isfinite(u), u + rng.gumbel(size=(B, J)), -np.inf)
    order = np.argsort(-g, axis=1)
    ranks = np.zeros((B, J), int)
    m = rng.choice([1, 2, 3], size=B, p=[0.02, 0.03, 0.95])
    for b in range(B):
        ranks[b, order[b, :m[b]]] = np.arange(1, m[b] + 1)
    return u, ranks, m


def test_unchanged_rates_reproduce_the_observed_ballots():
    rng = np.random.default_rng(3)
    u, ranks, m = random_ballots(rng)
    logX, _ = conditional_clocks(u, ranks, rng)
    assert np.array_equal(rerank(logX, u, m), ranks)


def test_clock_marginals_match_plackett_luce():
    # unconditional check: drawing a ranking, then clocks given it, gives X ~ Exp(1) for every nominee
    rng = np.random.default_rng(4)
    u = np.tile(np.array([[1.0, 0.0, -0.5, 0.3, -1.0]]), (40000, 1))
    g = u + rng.gumbel(size=u.shape)
    order = np.argsort(-g, axis=1)
    ranks = np.zeros(u.shape, int)
    rows = np.arange(len(u))[:, None]
    ranks[rows, order[:, :3]] = np.arange(1, 4)
    logX, lam = conditional_clocks(u, ranks, rng)
    X = np.exp(logX)
    assert np.allclose(X.mean(axis=0), 1.0, atol=0.03)
    assert np.allclose((X > 1.0).mean(axis=0), np.exp(-1.0), atol=0.01)


def test_standings_equal_group_weights():
    ranks = np.array([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2]])
    role = np.array(["captain", "coach", "media"])
    s = standings(ranks, role)
    assert np.isclose(s.sum(), 100)
    assert np.isclose(s[0], 100 * (5 + 0 + 1) / 9 / 3)


def test_extremely_unattractive_nominees_do_not_break_the_clocks():
    rng = np.random.default_rng(6)
    u, ranks, m = random_ballots(rng)
    u[:, 5] = np.where(np.isfinite(u[:, 5]), -900.0, u[:, 5])      # exp underflows to zero
    ranks[:, 5] = 0
    ranks = np.where(ranks > 0, np.argsort(np.argsort(-np.where(ranks > 0, -ranks, -99), axis=1), axis=1) + 1, 0)
    logX, _ = conditional_clocks(u, ranks, rng)
    assert np.isfinite(logX[np.isfinite(u)]).all()
    assert np.array_equal(rerank(logX, u, ranks.max(axis=1)), ranks)


def test_rule_percent_matches_each_rule():
    ranks = np.array([[1, 2, 3, 0], [2, 1, 0, 3], [1, 3, 2, 0], [3, 1, 2, 0]])
    role = np.array(["captain", "captain", "coach", "coach"])
    pts = np.array([5.0, 3, 1])
    share = rule_percent(ranks, role, pts, "share")
    assert np.allclose(share, standings(ranks, role, pts))
    sc = rule_percent(ranks, role, pts, "scoring")
    assert np.isclose(sc.sum(), 100)
    # captains: 8, 8, 1, 1 points (0 and 1 tie for first: 13 each; 2 and 3 tie for third: 9 each);
    # coaches: 6, 6, 6, 0 (three-way tie for first: 13 each; nominee 3 fourth: 8).
    # Scoring points 26, 26, 22, 17 of 91
    assert np.allclose(sc, 100 * np.array([26, 26, 22, 17]) / 91)
