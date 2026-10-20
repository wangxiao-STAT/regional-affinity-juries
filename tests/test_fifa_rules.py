"""FIFA's aggregation rules (fifa_rules.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fifa_rules import _scoring, jury_score, rule_for  # noqa: E402


def test_rule_by_contest():
    assert rule_for("2022_MP") == "share" and rule_for("2023_WP") == "scoring" and rule_for("BdO2024_MP") == "points"


def test_scoring_by_rank_and_competition_ties():
    assert np.allclose(_scoring(np.array([50.0, 30, 10, 0])), [13, 11, 9, 8])
    # a tie for second gives both nominees second place; the next nominee is fourth
    assert np.allclose(_scoring(np.array([50.0, 30, 30, 5])), [13, 11, 11, 8])
    # works on stacked draws
    assert np.allclose(_scoring(np.array([[50.0, 30, 30, 5], [1, 2, 3, 4]])), [[13, 11, 11, 8], [8, 9, 11, 13]])
    # nominees below 11th receive nothing
    s = _scoring(np.arange(13, 0, -1).astype(float))
    assert s[10] == 1 and s[11] == 0 and s[12] == 0


def test_jury_score_rules_and_tie_break():
    pts = np.array([[5, 3, 1], [5, 1, 3], [3, 5, 1], [1, 5, 3]], float)   # two captains, two coaches
    role = np.array(["captain", "captain", "coach", "coach"])
    assert np.allclose(jury_score(pts, role, "points"), [14, 14, 8])
    share = jury_score(pts, role, "share")
    assert np.isclose(share.sum(), 100) and np.isclose(share[0], share[1])
    tied = jury_score(pts, role, "scoring")
    assert np.isclose(tied[0], tied[1])                    # 13 + 11 each
    firsts = (pts[role == "captain"] == 5).sum(axis=0)
    assert jury_score(pts, role, "scoring", firsts).argmax() == 0   # art. 12: more captains' first places


def test_zero_point_nominees_can_score_nothing():
    pts = np.array([50.0, 30, 0, 0])
    assert np.allclose(_scoring(pts), [13, 11, 9, 9])               # tied at third
    assert np.allclose(_scoring(pts, zero_none=True), [13, 11, 0, 0])
