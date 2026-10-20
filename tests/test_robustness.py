"""Checks for the placebo shuffles and the rival indicator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robustness import contenders, recompute_affinity, shuffle_countries  # noqa: E402


def toy():
    rows = []
    for b, (vc, conf, role) in enumerate([("A", "X", "coach"), ("B", "X", "coach"), ("C", "Y", "coach"),
                                          ("A", "X", "media"), ("C", "Y", "media"), ("D", "Y", "media")]):
        for j, (p, nc, ncf, rank) in enumerate([("P1", "A", "X", 1), ("P2", "C", "Y", 2), ("P3", "D", "Y", 3),
                                                 ("P4", "B", "X", 0)]):
            rows.append({"ballot": b, "contest": "2019_MP", "role": role, "voter_country": vc, "voter_confed": conf,
                         "person": p, "nominee_country": nc, "nominee_confed": ncf, "rank": (rank + b) % 4,
                         "club_country_set": "A"})
    return pd.DataFrame(rows)


def test_shuffles_keep_the_multiset_of_countries():
    d = toy()
    rng = np.random.default_rng(0)
    for within in (False, True):
        s = shuffle_countries(d, rng, within)
        for keys in (["role"], ["role", "voter_confed"]) if within else (["role"],):
            a = d.drop_duplicates("ballot").groupby(keys).voter_country.apply(sorted)
            b = s.drop_duplicates("ballot").groupby(keys).voter_country.apply(sorted)
            assert a.equals(b)


def test_recomputed_affinities():
    d = recompute_affinity(toy(), {"A": "X", "B": "X", "C": "Y", "D": "Y"})
    r = d[(d.ballot == 0) & (d.person == "P1")].iloc[0]
    assert r.same_nation == 1 and r.same_confed == 1 and r.same_league == 1
    r = d[(d.ballot == 0) & (d.person == "P2")].iloc[0]
    assert r.same_nation == 0 and r.same_confed == 0


def test_contenders_are_three_per_contest():
    assert len(contenders(toy())) == 3
