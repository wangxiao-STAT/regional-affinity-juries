"""Which reading of FIFA's 2023+ scoring rule (art. 11) reproduces the published totals?

python -I nhb/src/check_fifa_scoring.py [raw_fifa_dir]  ->  nhb/data/processed/fifa_scoring_conventions.csv

The 2023 and 2024 result files print every nominee's total scoring points (four groups, fans
included). Subtracting the recomputed jury scores (captains, coaches, media) leaves the fans' score,
which must be one of the scoring values (13, 11, ..., 1, or 0 below 11th) and, fans' vote totals
being large, distinct across the nominees who scored. Two readings are tested: tied nominees take the
best, average or worst of the places they share, and nominees without points in a group either share
the next place or score nothing. Needs pdftotext and the raw PDFs (default data/raw/fifa).
"""
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SCORING = np.array([13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1], float)
POINTS = {1: 5, 2: 3, 3: 1}
CONTESTS = ("2023_MP", "2023_WP", "2024_MP", "2024_WP")


def fold(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def scoring(p, ties, zero):
    out = np.zeros(len(p))
    for j in range(len(p)):
        hi, n = (p > p[j]).sum(), (p == p[j]).sum()
        vals = np.array([SCORING[k] if k < len(SCORING) else 0.0 for k in range(hi, hi + n)])
        out[j] = {"average": vals.mean(), "best": vals[0], "worst": vals[-1]}[ties]
        if zero and p[j] == 0:
            out[j] = 0.0
    return out


def official_totals(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "r.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        text = txt.read_text()
    block = text.split("SCORING POINTS")[1].split("Voting Party")[0]
    return [(m.group(1).strip(), int(m.group(2)))
            for m in re.finditer(r"^\s+\d+\s+(.+?)\s{2,}\S.*?\s(\d+)\*?\s*$", block, re.M)]


def main():
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "fifa"
    long = pd.read_csv(PROC / "fifa_long.csv", keep_default_na=False)
    rows = []
    for c in CONTESTS:
        d = long[long.contest == c].copy()
        d["pts"] = pd.to_numeric(d["rank"], errors="coerce").map(POINTS).fillna(0)
        noms = sorted(d.person.unique())

        def match(name):
            f = fold(name)
            for n in noms:     # printed names are fuller or reordered ("Rodrigo Hernández Cascante (Rodri)")
                if fold(n) == f or fold(n) in f or f in fold(n) or fold(n).split()[-1] == f.split()[-1]:
                    return n
            return None
        off = official_totals(raw / f"{c}.pdf")
        offd = {match(n): v for n, v in off}
        offd.pop(None, None)
        everyone = sorted(set(noms) | set(offd))
        for ties in ("best", "average", "worst"):
            for zero in (False, True):
                jury = np.zeros(len(everyone))
                for g in ("captain", "coach", "media"):
                    gp = d[d.role == g].groupby("person").pts.sum().reindex(everyone).fillna(0).to_numpy()
                    jury += scoring(gp, ties, zero)
                fan = np.array([offd.get(n, np.nan) for n in everyone]) - jury
                fan = fan[~np.isnan(fan)]
                scored = [x for x in fan if x > 0]
                ok = all(x in set(SCORING) | {0.0} for x in fan) and len(scored) == len(set(scored))
                rows.append({"contest": c, "ties": ties, "zero_points_score_nothing": zero, "consistent": ok,
                             "nominees_printed": len(off), "nominees_matched": len(offd)})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "fifa_scoring_conventions.csv", index=False)
    print(out.groupby(["ties", "zero_points_score_nothing"]).consistent.all().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
