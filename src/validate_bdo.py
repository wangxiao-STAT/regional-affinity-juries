"""Check the Ballon d'Or per-juror ballots against the published results.

python -I nhb/src/validate_bdo.py  ->  prints the checks; nhb/data/processed/bdo_validation.csv,
                                       bdo_validation_summary.csv, bdo_crosscheck.csv

1. Points: each nominee's total recomputed from bdo_ballots.csv (6-4-3-2-1 up to 2023, 15-12-10-8-7-5-
   4-3-2-1 from 2024) against the official totals of the Wikipedia ranking table and, where the page
   prints one, L'Équipe's own ranking (men 2022, women 2023 and 2024). Every nonzero difference is listed.
2. Jurors: parsed count against the count the source states, and against the count implied by the
   official points (sum of totals / points per full ballot).
3. Statements in the L'Équipe articles that the ballots must reproduce (first-place counts, ballots
   naming a player, the one juror without the winner).
4. Integrity: no juror names a player twice, no empty slot within 1..k, one ballot per country.
5. Cross-source agreement, juror by juror: men 2024 (companion-study data) and men 2025 (L'Équipe) against
   Wikipedia's per-country tables (bdo_wiki_ballots.csv).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_bdo import KMAX, SCHEME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PICKS = [f"pick{i}" for i in range(1, KMAX + 1)]
STATED_JURORS = {
    (2019, "MP"): (176, "France Football print issue of 4 Dec 2019, as recorded in the companion study's data dictionary"),
    (2022, "MP"): (93, "L'Équipe 2022: Benzema 'premier du classement des jurés 89 fois sur 93'"),
    (2023, "MP"): (None, "no stated count (L'Équipe page paywalled)"),
    (2024, "MP"): (99, "companion-study voter_source.csv: 'France Football 2024 (99 FIFA-ranked voters)'"),
    (2025, "MP"): (100, "L'Équipe 2025: 'l'unanimité parmi les 100 jurés'"),
    (2023, "WP"): (47, "L'Équipe 2023: Bonmatí 'citée à la première place 42 fois sur 47'"),
    (2024, "WP"): (49, "L'Équipe 2024: 'l'intégralité des votes des 49 jurés'"),
    (2025, "WP"): (None, "no stated count (50 countries invited)"),
}


def at(person, slot):
    return lambda b: int((b[f"pick{slot}"] == person).sum())


def naming(person):
    return lambda b: int((b[PICKS] == person).any(axis=1).sum())


def without(person):
    return lambda b: ";".join(sorted(b.loc[~(b[PICKS] == person).any(axis=1), "country"]))


CLAIMS = [   # (year, award, what, function of the contest's ballots, stated value, quote)
    (2022, "MP", "Benzema first places", at("Karim Benzema", 1), 89, "89 fois sur 93"),
    (2022, "MP", "Benzema second places", at("Karim Benzema", 2), 3, "seulement trois fois à la deuxième place"),
    (2022, "MP", "Benzema third places", at("Karim Benzema", 3), 1, "et une fois à la troisième"),
    (2022, "MP", "De Bruyne first places", at("Kevin De Bruyne", 1), 1, "Le votant slovène ... lui accorde la première place"),
    (2023, "WP", "Bonmatí first places", at("Aitana Bonmatí", 1), 42, "citée à la première place 42 fois sur 47"),
    (2023, "WP", "ballots without Bonmatí", without("Aitana Bonmatí"), "Chile",
     "parmi les tops 5 de tous les votants, à l'exception du Chili"),
    (2025, "MP", "ballots naming Dembélé", naming("Ousmane Dembélé"), 100, "Il est cité dans tous les votes"),
    (2025, "MP", "Dembélé first places", at("Ousmane Dembélé", 1), 73, "dont 73 fois à la première place"),
    (2025, "WP", "ballots naming Bonmatí", naming("Aitana Bonmatí"), 49, "citée 49 fois"),
    (2025, "WP", "ballots naming Caldentey", naming("Mariona Caldentey"), 44, "contre 44 pour ... Mariona Caldentey"),
]


# Differences traced to their cause; any other nonzero difference fails the run.
_SWAP_2019 = ("The companion study's 2019 ballots (France Football print issue) give Ronaldo one point fewer and Sterling one "
              "more than the official totals (476 and 30); the companion study logs the same difference "
              "(KNOWN_POINT_DISCREPANCY). Some ballot must have Sterling one point higher and Ronaldo one lower "
              "than printed. Four ballots fit: Australia, Norway and the Philippines (Sterling directly above "
              "Ronaldo) and Pakistan (Sterling fifth, Ronaldo absent). The print page is needed to tell which; "
              "the data are left as in the companion study's source.")
EXPLAINED = {
    (2019, "MP", "wikipedia", "Cristiano Ronaldo"): (-1, _SWAP_2019),
    (2019, "MP", "wikipedia", "Raheem Sterling"): (1, _SWAP_2019),
    (2023, "WP", "wikipedia", "Debinha"): (-1, "Wikipedia error: its ranking table gives Debinha 1 point, L'Équipe's "
                                               "ranking gives 0 (28th, below the 1-point players) and the ballots "
                                               "agree; Wikipedia's total (753) exceeds 47 full ballots (752) by one."),
}


def points(b):
    long = b.melt(id_vars=["year", "award", "country", "k"], value_vars=PICKS, var_name="slot", value_name="person")
    long = long[long.person != ""]
    long["pts"] = [SCHEME[k][int(s[4:]) - 1] for k, s in zip(long.k, long.slot)]
    return long.groupby(["year", "award", "person"]).pts.sum()


def main():
    b = pd.read_csv(PROC / "bdo_ballots.csv", keep_default_na=False, dtype={c: str for c in PICKS})
    off = pd.read_csv(PROC / "bdo_official_points.csv", keep_default_na=False)
    names = pd.read_csv(PROC / "bdo_nominee_names.csv", keep_default_na=False)
    person = {(y, a, r): p for y, a, r, p in zip(names.year, names.award, names.raw, names.person)}
    off["person"] = [person.get((y, a, n), "") for y, a, n in zip(off.year, off.award, off["name"])]
    off["kind"] = off.source.str.split(":").str[0]
    pts = points(b)
    status = 0

    # 1. points
    rows = []
    for r in off.itertuples(index=False):
        got = int(pts.get((r.year, r.award, r.person), 0))
        rows.append({"year": r.year, "award": r.award, "source": r.kind, "official_rank": r.rank,
                     "person": r.person, "official": r.points, "parsed": got, "diff": got - r.points})
    v = pd.DataFrame(rows)
    v["cause"] = [EXPLAINED[(y, a, k, p)][1] if EXPLAINED.get((y, a, k, p), (None,))[0] == d else ""
                  for y, a, k, p, d in zip(v.year, v.award, v.source, v.person, v["diff"])]
    # nominees with parsed points but missing from an official list
    for (y, a, kind), g in v.groupby(["year", "award", "source"]):
        listed = set(g.person)
        for (yy, aa, p), x in pts.items():
            if (yy, aa) == (y, a) and p not in listed:
                v.loc[len(v)] = {"year": y, "award": a, "source": kind, "official_rank": None, "person": p,
                                 "official": None, "parsed": int(x), "diff": None, "cause": ""}
    v.to_csv(PROC / "bdo_validation.csv", index=False)

    # 2. juror counts and summary
    summary = []
    for (y, a), g in b.groupby(["year", "award"]):
        k = int(g.k.iloc[0])
        stated, note = STATED_JURORS[(y, a)]
        row = {"year": y, "award": a, "k": k, "jurors_parsed": len(g), "jurors_stated": stated, "stated_source": note,
               "named_jurors": int((g.juror != "").sum())}
        for kind in ("wikipedia", "lequipe"):
            o = v[(v.year == y) & (v.award == a) & (v.source == kind)]
            if o.empty:
                continue
            tot = o.official.dropna().sum()
            row[f"{kind}_implied_jurors"] = round(tot / sum(SCHEME[k]), 3)
            row[f"{kind}_nominees"] = int(o.official.notna().sum())
            row[f"{kind}_max_abs_diff"] = int(o["diff"].abs().max()) if o["diff"].notna().all() else None
            row[f"{kind}_n_diff"] = int((o["diff"].fillna(1) != 0).sum())
        summary.append(row)
    s = pd.DataFrame(summary)
    for c in s.columns:
        if c.endswith(("_stated", "_nominees", "_max_abs_diff", "_n_diff")):
            s[c] = s[c].astype("Int64")
    s.to_csv(PROC / "bdo_validation_summary.csv", index=False)
    print("Per contest (k, jurors parsed / stated, implied by official points, max |points diff|):")
    print(s.drop(columns=["stated_source"]).to_string(index=False))
    bad = v[(v["diff"] != 0) | v["diff"].isna()]
    print(f"\nNominee totals differing from an official list: {len(bad)} "
          f"({int((bad.cause != '').sum())} with a traced cause)")
    for r in bad.itertuples(index=False):
        print(f"  {r.year} {r.award} {r.source}: {r.person} official {r.official}, parsed {r.parsed}, diff {r.diff}"
              f"\n    cause: {r.cause or 'UNEXPLAINED'}")
    status |= int((bad.cause == "").any())

    # 3. statements in the articles
    print("\nStatements in the L'Équipe articles:")
    for y, a, what, f, stated, quote in CLAIMS:
        got = f(b[(b.year == y) & (b.award == a)])
        ok = got == stated
        status |= 0 if ok else 1
        print(f"  {y} {a} {what}: parsed {got}, stated {stated} [{'ok' if ok else 'MISMATCH'}] ('{quote}')")

    # 4. integrity
    picks = b[PICKS]
    filled = picks.ne("")
    repeat = picks.apply(lambda r: r[r != ""].duplicated().any(), axis=1)
    gaps = [i for i, (k, f) in enumerate(zip(b.k, filled.values)) if not f[:k].all() or f[k:].any()]
    dup_country = b[b.duplicated(["year", "award", "country"], keep=False)]
    print(f"\nballots naming a player twice: {int(repeat.sum())}; ballots with an empty slot within 1..k or a "
          f"pick beyond k: {len(gaps)}; contests with two ballots from one country: {len(dup_country)} rows")
    for frame in (b[repeat], b.iloc[gaps], dup_country):
        if len(frame):
            print(frame.to_string(index=False))
            status = 1

    # 5. cross-source agreement
    w = pd.read_csv(PROC / "bdo_wiki_ballots.csv", keep_default_na=False, dtype={c: str for c in PICKS})
    for c in PICKS:
        w[c] = [person.get((y, a, r), "") if r else "" for y, a, r in zip(w.year, w.award, w[c])]
    out = []
    for y in (2024, 2025):
        ours = b[(b.year == y) & (b.award == "MP")].set_index("country")
        theirs = w[w.year == y].set_index("country")
        for c in sorted(set(ours.index) | set(theirs.index)):
            if c not in theirs.index or c not in ours.index:
                out.append({"year": y, "country": c, "status": "only in " + ("ours" if c in ours.index else "wikipedia"),
                            "slots": "", "ours": "", "wikipedia": ""})
                continue
            o, t = ours.loc[c, PICKS], theirs.loc[c, PICKS]
            diff = [i + 1 for i in range(KMAX) if o.iloc[i] != t.iloc[i]]
            out.append({"year": y, "country": c, "status": "identical" if not diff else "differs",
                        "slots": ";".join(map(str, diff)),
                        "ours": " > ".join(o.iloc[i - 1] for i in diff), "wikipedia": " > ".join(t.iloc[i - 1] for i in diff)})
    x = pd.DataFrame(out)
    x.to_csv(PROC / "bdo_crosscheck.csv", index=False)
    print("\nJuror-by-juror agreement with Wikipedia's per-country tables (men):")
    print(x.groupby(["year", "status"]).size().to_string())
    nd = x[x.status != "identical"]
    if len(nd):
        print(nd.to_string(index=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
