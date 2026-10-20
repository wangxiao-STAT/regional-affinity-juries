"""Checks for the cross-jury simulator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cross_jury import affinity, bonus, winners  # noqa: E402


def test_affinity_and_role_specific_bonus():
    voters = pd.DataFrame({"voter_country": ["Spain", "Brazil"], "role": ["bdo", "coach"]})
    noms = pd.DataFrame({"nominee_country": ["Spain", "Argentina"], "nominee_confed": ["UEFA", "CONMEBOL"],
                         "club_country_set": ["Spain", "Spain;France"]})
    confed = {"Spain": "UEFA", "Brazil": "CONMEBOL"}
    A = affinity(voters, noms, confed)
    assert A[0, 0].tolist() == [1, 1, 1] and A[0, 1].tolist() == [0, 0, 1] and A[1, 1].tolist() == [0, 1, 0]
    beta = {(a, r): v for r, v in (("bdo", 1.0), ("coach", 2.0)) for a in ("same_nation", "same_confed", "same_league")}
    b = bonus(voters, noms, confed, beta, "role")
    assert np.allclose(b, [[3.0, 1.0], [0.0, 2.0]])


def test_winners_rules():
    U = np.array([[[3, 2, 1], [3, 1, 2], [1, 3, 2]]], float)
    assert winners(U, np.array(["a", "a", "a"]), np.array([5.0, 3.0, 1.0]), "points")[0] == 0
    # groups weighted equally: shares a = (10, 6, 2)/18, b = (1, 5, 3)/9, so nominee 1 averages 0.444 > 0.333
    U2 = np.array([[[3, 2, 1], [3, 2, 1], [1, 3, 2]]], float)
    assert winners(U2, np.array(["a", "a", "b"]), np.array([5.0, 3.0, 1.0]), "fifa")[0] == 1
    assert winners(U2, np.array(["a", "a", "a"]), np.array([5.0, 3.0, 1.0]), "points")[0] == 0


def test_bonus_with_confederation_specific_regional_coefficient():
    voters = pd.DataFrame({"voter_country": ["Spain", "Brazil"], "role": ["bdo", "coach"]})
    noms = pd.DataFrame({"nominee_country": ["France"], "nominee_confed": ["UEFA"], "club_country_set": ["France"]})
    confed = {"Spain": "UEFA", "Brazil": "CONMEBOL", "France": "UEFA"}
    beta = {("same_confed", "bdo:UEFA"): 0.7, ("same_confed", "fifa:CONMEBOL"): 0.3,
            ("same_nation", "bdo"): 1.0, ("same_nation", "coach"): 2.0, ("same_league", "bdo"): 0.1,
            ("same_league", "coach"): 0.2}
    split = {"same_nation": "role", "same_confed": "org_confed", "same_league": "role"}
    assert np.allclose(bonus(voters, noms, confed, beta, split), [[0.7], [0.0]])
