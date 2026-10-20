"""Recovery and gradient checks for the pooled exploded-logit estimator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fit_pooled import Model  # noqa: E402


def simulate(seed=0, n_contests=6, n_noms=10, n_voters=400, beta=(1.2, 0.4, 0.6), k=3):
    rng = np.random.default_rng(seed)
    countries = [f"C{i}" for i in range(40)]
    confed = {c: f"F{i % 4}" for i, c in enumerate(countries)}
    rows, ballot = [], 0
    for c in range(n_contests):
        alpha = rng.normal(0, 1.5, n_noms)
        nat = rng.choice(countries[:12], n_noms)
        league = rng.choice(countries[:6], n_noms)
        for v in range(n_voters):
            vc = countries[v % len(countries)]
            role = ["captain", "coach", "media"][v % 3]
            x = np.column_stack([nat == vc, [confed[n] == confed[vc] for n in nat], league == vc]).astype(float)
            u = alpha + x @ np.array(beta)
            g = u + rng.gumbel(size=n_noms)
            order = np.argsort(-g)[:k]
            rank = np.zeros(n_noms, int)
            rank[order] = np.arange(1, k + 1)
            for j in range(n_noms):
                rows.append({"ballot": ballot, "year": 2010 + c, "award": "MP", "contest": f"{2010 + c}_MP",
                             "role": role, "voter_country": vc, "person": f"N{c}_{j}", "rank": rank[j],
                             "same_nation": x[j, 0], "same_confed": x[j, 1], "same_league": x[j, 2]})
            ballot += 1
    return pd.DataFrame(rows)


def test_gradient_matches_finite_differences():
    long = simulate(n_contests=2, n_voters=60)
    m = Model(long)
    theta = np.random.default_rng(1).normal(0, 0.3, m.n_alpha + m.n_beta)
    f, g = m.nll(theta)
    eps = 1e-6
    for k in [0, 3, m.n_alpha, m.n_alpha + 2]:
        d = np.zeros_like(theta)
        d[k] = eps
        num = (m.nll(theta + d)[0] - m.nll(theta - d)[0]) / (2 * eps)
        assert abs(num - g[k]) < 1e-4 * max(1, abs(num))


def test_recovers_affinity_coefficients():
    beta = (1.2, 0.4, 0.6)
    m = Model(simulate(beta=beta)).fit()
    V, Vc = m.covariances()
    b = m.theta[m.n_alpha:]
    se = np.sqrt(np.diag(V)[m.n_alpha:])
    assert m.opt.success
    assert np.all(np.abs(b - np.array(beta)) < 4 * se), (b, se)
    assert np.all(np.isfinite(np.diag(Vc)))


def test_role_split_and_role_fixed_effects_run():
    long = simulate(n_contests=2, n_voters=150)
    m = Model(long, split="role", fe_by=("role",)).fit()
    assert m.n_beta == 9 and m.opt.success


def test_cells_without_picks_are_dropped_and_information_is_invertible():
    long = simulate(n_contests=2, n_voters=150)
    long["voter_confed"] = long.voter_country.str[1:].astype(int).mod(4).astype(str)
    # a nominee nobody in one confederation picked: its cell must be dropped, not estimated
    cell = (long.person == "N0_9") & (long.voter_confed == "0")
    long.loc[cell & (long["rank"] > 0), "rank"] = 0
    pos = long["rank"] > 0
    long.loc[pos, "rank"] = long[pos].groupby("ballot")["rank"].rank(method="first").astype(int)
    m = Model(long, split="role", fe_by=("voter_confed",), aff=["same_nation", "same_league"]).fit()
    V, Vc = m.covariances()
    assert m.n_dropped >= int(cell.sum())
    assert "2010_MP|0#N0_9" not in set(m.fe_names) | set(m.ref_names)
    assert np.isfinite(np.diag(V)).all() and m.cond < 1e12


def test_recovers_affinity_coefficients_from_five_deep_ballots():
    beta = (1.2, 0.4, 0.6)
    m = Model(simulate(beta=beta, k=5, n_voters=250, seed=3)).fit()
    V, _ = m.covariances()
    b = m.theta[m.n_alpha:]
    se = np.sqrt(np.diag(V)[m.n_alpha:])
    assert m.opt.success and np.all(np.abs(b - np.array(beta)) < 4 * se), (b, se)


def test_psd_sqrt_handles_rank_deficient_sandwich():
    from fit_pooled import psd_sqrt
    rng = np.random.default_rng(5)
    S = rng.normal(size=(30, 200))                  # 30 clusters, 200 parameters: rank 30
    V = S.T @ S / 30
    L = psd_sqrt(V)
    assert np.allclose(L @ L.T, V, atol=1e-8)
    assert np.isfinite(L).all()


def test_hybrid_draws_have_the_intended_covariance():
    from fit_pooled import hybrid_draws
    rng = np.random.default_rng(5)
    A = rng.normal(size=(5, 5))
    V_model = A @ A.T + 5 * np.eye(5)                 # 3 nominee effects, 2 coefficients
    B = rng.normal(size=(2, 2))
    V_cluster = np.zeros((5, 5))
    V_cluster[3:, 3:] = B @ B.T + np.eye(2)
    d = np.array(hybrid_draws(np.zeros(5), V_model, V_cluster, 3, 40000, rng))
    C = np.cov(d.T)
    assert np.allclose(C[3:, 3:], V_cluster[3:, 3:], atol=0.08 * np.abs(V_cluster[3:, 3:]).max())
    K = V_model[:3, 3:] @ np.linalg.inv(V_model[3:, 3:])
    resid = d[:, :3] - d[:, 3:] @ K.T                 # nominee effects given the coefficients
    S = V_model[:3, :3] - K @ V_model[:3, 3:].T
    assert np.allclose(np.cov(resid.T), S, atol=0.08 * np.abs(S).max())
