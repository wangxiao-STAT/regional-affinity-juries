"""How each award turns ballots into a winner.

FIFA 2010-2022: each jury group's share of 5/3/1 points, averaged over the groups (validated against
the published 2010-2015 percentages; FIFA describes the groups as equally weighted).
FIFA 2023 onward (Rules of Allocation, art. 11 and 12, printed in the 2023 and 2024 result files):
within each group nominees are ranked by 5/3/1 points and receive 13, 11, 9, 8, 7, 6, 5, 4, 3, 2 or 1
scoring points by rank (none below 11th); the scores are summed over the groups, and a tie for first
goes to the nominee with more first places from the captains (the nominee's own group of voters for
the player awards). The rules do not say how ties within a group are scored; nominees tied on points
receive the scoring points of the best position they share (competition ranking), the only one of the
conventions tried (shared average, best, worst) that leaves every implied fan score in the 2023 and
2024 published tables a distinct scoring value.
Ballon d'Or: total points.
Fans form a fourth group from 2016 and are not published by voter, so FIFA standings here are the
juries' (three groups).
"""
import numpy as np

SCORING = np.array([13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1], float)


def rule_for(contest):
    if contest.startswith("BdO"):
        return "points"
    return "scoring" if int(contest[:4]) >= 2023 else "share"


def _scoring(points, zero_none=False):
    """points: (..., J) group points; scoring points by competition rank (ties take the best position).
    Nominees without points share the next place unless zero_none, which gives them nothing (the
    published 2023-2024 totals cannot tell the two readings of art. 11 apart)."""
    J = points.shape[-1]
    table = np.zeros(J)
    table[:min(J, len(SCORING))] = SCORING[:J]
    higher = (points[..., None, :] > points[..., :, None]).sum(-1)     # nominees strictly ahead
    out = table[higher]
    return np.where(points > 0, out, 0.0) if zero_none else out


def jury_score(pts, role, rule, captain_firsts=None, zero_none=False):
    """pts: (..., B, J) points by ballot; role: (B,). Returns (..., J) scores whose argmax is the winner.

    share: average over groups of each nominee's share of the group's points (in per cent);
    scoring: sum over groups of art. 11 scoring points, plus a tie-break term (art. 12) that is
    smaller than any scoring difference; points: total points.
    """
    if rule == "points":
        return pts.sum(axis=-2)
    groups = [g for g in np.unique(role) if pts[..., role == g, :].sum() > 0]
    out = np.zeros(pts.shape[:-2] + pts.shape[-1:])
    for g in groups:
        gp = pts[..., role == g, :].sum(axis=-2)
        if rule == "share":
            tot = gp.sum(axis=-1, keepdims=True)
            out += 100 * gp / np.where(tot > 0, tot, 1) / len(groups)
        else:
            out += _scoring(gp, zero_none)
    if rule == "scoring" and captain_firsts is not None:
        out += 1e-6 * captain_firsts
    return out


def captain_first_places(ranks_or_pts, role, top_value):
    """Number of first places each nominee receives from captains (art. 12 tie-break)."""
    cap = role == "captain"
    return (ranks_or_pts[..., cap, :] == top_value).sum(axis=-2)
