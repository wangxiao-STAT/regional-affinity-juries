"""Checks for the jury-design simulator."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from design_sim import flips, points_by_role, run_design  # noqa: E402


def test_points_and_group_shares():
    # two voters per role, three nominees, utilities give a fixed order per voter
    U = np.array([[[3, 2, 1], [3, 1, 2], [1, 3, 2], [1, 3, 2], [2, 1, 3], [1, 2, 3]]], float)
    role = np.array(["captain", "captain", "coach", "coach", "media", "media"])
    share, total = points_by_role(U, role)
    assert np.allclose(total[0], [5 + 5 + 1 + 1 + 3 + 1, 3 + 1 + 5 + 5 + 1 + 3, 1 + 3 + 3 + 3 + 5 + 5])
    assert np.isclose(share.sum(), 1.0)
    assert np.isclose(share[0, 0], (10 / 18 + 2 / 18 + 4 / 18) / 3)


def test_no_bias_never_flips():
    rng = np.random.default_rng(0)
    B, J = 120, 8
    base = np.tile(rng.normal(0, 1, J), (B, 1))
    ci = {"B": B, "J": J, "role": np.array(["captain", "coach", "media"] * (B // 3))}
    for rule in ("fifa", "points"):
        f = run_design(ci, base, base.copy(), rng, 50, rule)
        assert not f.any()
        f = run_design(ci, base, base.copy(), rng, 50, rule, pick=lambda r: r.choice(B, 30, replace=False))
        assert not f.any()


def test_strong_bias_flips_close_race():
    rng = np.random.default_rng(1)
    B, J = 90, 4
    base = np.tile(np.array([0.05, 0.0, -2.0, -2.0]), (B, 1))
    biased = base.copy()
    biased[:, 1] += 1.0                                  # every voter favours nominee 1
    ci = {"B": B, "J": J, "role": np.array(["captain", "coach", "media"] * (B // 3))}
    f = run_design(ci, base, biased, rng, 100, "points")
    assert f.mean() > 0.3
