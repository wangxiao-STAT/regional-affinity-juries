"""Supplementary tables of the NHB manuscript: nhb/manuscript/generated/sup_*.tex.

python -I nhb/src/make_tables.py

sup_sources      ballots per contest with the source of each file
sup_coaches      coach ballots with known coach nationality, by award and year
sup_bdo          Ballon d'Or validation: jurors parsed and stated, largest point difference
sup_estimates    similarity bonuses of the combined models, clustered and model-based standard errors
sup_fits         size, log-likelihood, condition number and extreme nominee effects of every fit
sup_contests     every contest: rule, leader, margin and the probability that the bonuses decided it
A table whose inputs are missing is written as a \\Pending marker.
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "manuscript" / "generated"
RES = ROOT / "results"
AWARD = {"MP": "Men", "WP": "Women"}


def esc(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def tabular(header, rows, spec):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(esc(c) if not str(c).startswith("\\") else str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write(name, fn):
    try:
        text = fn()
    except (FileNotFoundError, KeyError, AttributeError) as e:
        text = f"\\Pending{{{name}: {type(e).__name__}}}\n"
    (OUT / f"{name}.tex").write_text(text)


def sup_sources():
    b = pd.read_csv(PROC / "fifa_ballots.csv", keep_default_na=False)
    man = pd.read_csv(RAW / "fifa" / "manifest.csv", keep_default_na=False)
    n = b.groupby(["year", "award", "role"]).size().unstack(fill_value=0)
    rows = []
    for r in man.itertuples(index=False):
        host = (urlparse(r.url.split("id_/")[-1]).netloc or urlparse(r.url).netloc).split(":")[0]
        via = "Wayback copy of " if "web.archive.org" in r.url else ""
        c = n.loc[(r.year, r.award)]
        rows.append([r.year, AWARD[r.award], c.get("coach", 0), c.get("captain", 0), c.get("media", 0),
                     f"{via}{host}"])
    bb = pd.read_csv(PROC / "bdo_ballots.csv", keep_default_na=False)
    for (y, a), g in bb.groupby(["year", "award"]):
        src = g.source.iloc[0].split(":")[0]
        label = {"jasa": "companion study (France Football)", "lequipe": "L'Équipe (Wayback copy)",
                 "wikipedia": "Wikipedia per-country table"}.get(src, src)
        rows.append([y, f"{AWARD[a]} (Ballon d'Or)", "", "", len(g), label])
    return tabular(["Year", "Award", "Coaches", "Captains", "Journalists", "Source"], rows, "llrrrl")


def sup_coaches():
    c = pd.read_csv(PROC / "fifa_coach_nationality_all.csv", keep_default_na=False)
    c["known"] = c.coach_nationality != ""
    c["foreign"] = c.known & (c.coach_nationality != c.voter_country)
    g = c.groupby(["year", "award"]).agg(n=("ballot", "size"), known=("known", "sum"), foreign=("foreign", "sum"))
    rows = []
    for y in sorted(c.year.unique()):
        row = [y]
        for a in ("MP", "WP"):
            r = g.loc[(y, a)]
            row += [r.n, r.known, r.foreign]
        rows.append(row)
    return tabular(["Year", "Men: ballots", "known", "foreign", "Women: ballots", "known", "foreign"], rows,
                   "lrrrrrr")


def sup_bdo():
    v = pd.read_csv(PROC / "bdo_validation_summary.csv", keep_default_na=False)
    rows = []
    for r in v.itertuples(index=False):
        diff = r.wikipedia_max_abs_diff if str(r.wikipedia_max_abs_diff) != "" else r.lequipe_max_abs_diff
        rows.append([r.year, AWARD[r.award], r.k, r.jurors_parsed, r.jurors_stated or "not stated",
                     int(float(diff)) if str(diff) != "" else ""])
    return tabular(["Year", "Award", "Ranks", "Jurors parsed", "Jurors stated", "Largest point difference"],
                   rows, "llrrrr")


TERM = {"same_nation": "Compatriot", "same_confed": "Same confederation", "same_league": "Club in juror's country"}
LEVEL = {"captain": "captains", "coach": "coaches", "media": "FIFA journalists", "bdo": "Ballon d'Or journalists"}
MODELS = [("combined", "C1_role", "Common appeal per nominee and contest"),
          ("combined", "C4_role_confFE", "Separate appeal for each juror region"),
          ("combined", "C2_confhet", "Regional bonus by juror confederation")]


def label(term):
    t, lv = term.split("[")
    lv = lv[:-1]
    if ":" in lv:                                  # org:confederation
        org, conf = lv.split(":")
        lv = f"{'Ballon d' + chr(39) + 'Or' if org == 'bdo' else 'FIFA'} jurors from {conf}"
    else:
        lv = LEVEL.get(lv, lv)
    return f"{TERM[t]}, {lv}"


def sup_estimates():
    rows = []
    for folder, model, desc in MODELS:
        e = pd.read_csv(RES / folder / "estimates.csv").query("model == @model")
        rows.append([f"\\multicolumn{{4}}{{l}}{{\\textit{{{esc(desc)}}}}}"])
        for _, r in e.iterrows():
            rows.append([label(r.term), f"{r.beta:.2f}", f"{r.se_cluster:.2f}", f"{r.se:.2f}"])
    return tabular(["Bonus", "Log-odds", "s.e. (clustered)", "s.e. (model)"], rows, "lrrr")


FIT_DESC = {
    "M0_pooled": "FIFA: one bonus per similarity",
    "M1_role": "FIFA: bonuses by jury group",
    "M2_role_award": "FIFA: by jury group and award",
    "M3_role_era": "FIFA: by jury group and period",
    "M1_role_rfe": "FIFA: by jury group, appeal by jury group",
    "M2_role_award_rfe": "FIFA: by group and award, appeal by group",
    "M4_role_confFE": "FIFA: appeal by juror confederation",
    "M4_role_award_confFE": "FIFA: by award, appeal by juror confederation",
    "M5_role_confhet": "FIFA: regional bonus by juror confederation",
    "M6_role_confhet_role": "FIFA: regional bonus by group and confederation",
    "C1_role": "Both awards: bonuses by jury group",
    "C1_role_award": "Both awards: by jury group and award",
    "C1_same_years": "Both awards: years with both juries",
    "C2_confhet": "Both awards: regional bonus by juror region",
    "C4_role_confFE": "Both awards: appeal by juror region",
    "C5_bdo_depth3": "Both awards: Ballon d'Or ballots cut to three",
    "A1_coaches": "Coaches: nationality and coached country",
    "A1m_coaches": "Men's award coaches",
    "A1w_coaches": "Women's award coaches",
    "A1x_coaches": "Coaches: terms by award, one fit",
    "A2_all": "All groups, coach terms, shared appeal",
    "A3_movers": "Coaches who changed confederation",
    "B1_transfers_within": "Club moves within nominee-country pairs",
    "B2_transfers_between": "Club country: current, past, future",
}


def sup_fits():
    rows = []
    for folder in ("pooled", "combined", "mechanisms"):
        for f in json.loads((RES / folder / "fits.json").read_text()):
            rows.append([FIT_DESC.get(f["model"], f["model"]), f"\\texttt{{{esc(f['model'])}}}",
                         f"{f['n_alpha'] + f['n_beta']:,}", f"{f['loglik']:,.1f}", f"{f['hessian_cond']:.1e}",
                         str(f.get("extreme_alpha_cells", "--"))])
    return tabular(["Model", "Code", "Parameters", "Log-likelihood", "Condition number", "Extreme effects"],
                   rows, "llrrrr")


RULE = {"share": "Group shares", "scoring": "Scoring points", "points": "Total points"}


def contest_name(c):
    if c.startswith("BdO"):
        return f"Ballon d'Or {c[3:7]} {'men' if c.endswith('MP') else 'women'}"
    return f"FIFA {c[:4]} {'men' if c.endswith('MP') else 'women'}"


def sup_contests():
    base = pd.read_csv(RES / "combined" / "cf_C1_role_contests.csv").query("scenario == 'all'").set_index("contest")
    het = pd.read_csv(RES / "combined" / "cf_C2_confhet_contests.csv").query("scenario == 'all'").set_index("contest")
    reg = pd.read_csv(RES / "combined" / "cf_C1_role_contests.csv").query("scenario == 'confed'").set_index("contest")
    share = pd.read_csv(RES / "combined" / "cf_C1_role_share_contests.csv").query("scenario == 'all'").set_index("contest")
    order = sorted(base.index, key=lambda c: (c.startswith("BdO"), c[3:7] if c.startswith("BdO") else c[:4], c[-2:]))
    rows = []
    for c in order:
        r = base.loc[c]
        rows.append([contest_name(c), RULE[r.rule], r.winner, f"{r.margin:.2f}", f"{r.p_flip:.2f}",
                     f"{r.p_flip_mle:.2f}", f"{reg.loc[c].p_flip:.2f}", f"{het.loc[c].p_flip:.2f}",
                     f"{share.loc[c].p_flip:.2f}" if c in share.index else "--"])
    return tabular(["Contest", "Rule", "Leader", "Margin", "P(all)", "P(all, estimates)", "P(regional)",
                    "P(all, by region)", "P(all, share rule)"], rows, "lllrrrrrr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("sup_sources", sup_sources), ("sup_coaches", sup_coaches), ("sup_bdo", sup_bdo),
                     ("sup_estimates", sup_estimates), ("sup_fits", sup_fits), ("sup_contests", sup_contests)):
        write(name, fn)
    print("tables: sup_sources, sup_coaches, sup_bdo, sup_estimates, sup_fits, sup_contests", file=sys.stderr)


if __name__ == "__main__":
    main()
