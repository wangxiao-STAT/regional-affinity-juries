"""Supplementary tables of the NHB manuscript: nhb/manuscript/generated/sup_*.tex.

python -I nhb/src/make_tables.py

sup_sources      ballots per contest with the source of each file
sup_coaches      coach ballots with known coach nationality, by award and year
sup_bdo          Ballon d'Or validation: jurors parsed and stated, largest point difference
sup_estimates    similarity bonuses of the combined models, clustered and model-based standard errors
sup_fits         size, log-likelihood, condition number and extreme nominee effects of every fit
sup_contests     every contest: rule, leader, margin and the probability that the bonuses decided it
sup_contrasts    the coach contrasts quoted as z in the text, with clustered s.e. and two-sided P values
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


def display_names():
    """Wikipedia display name of each nominee, as in the text (FIFA files print many surname-first)."""
    out = {}
    for f in ("fifa_nominee_meta.csv", "bdo_nominee_meta.csv"):
        m = pd.read_csv(PROC / f, keep_default_na=False)
        for r in m.itertuples(index=False):
            if r.wiki_name:
                out[(int(r.year), r.award, r.person)] = r.wiki_name
    return out


def sup_contests():
    base = pd.read_csv(RES / "combined" / "cf_C1_role_contests.csv").query("scenario == 'all'").set_index("contest")
    het = pd.read_csv(RES / "combined" / "cf_C2_confhet_contests.csv").query("scenario == 'all'").set_index("contest")
    reg = pd.read_csv(RES / "combined" / "cf_C1_role_contests.csv").query("scenario == 'confed'").set_index("contest")
    share = pd.read_csv(RES / "combined" / "cf_C1_role_share_contests.csv").query("scenario == 'all'").set_index("contest")
    order = sorted(base.index, key=lambda c: (c.startswith("BdO"), c[3:7] if c.startswith("BdO") else c[:4], c[-2:]))
    names = display_names()
    rows = []
    for c in order:
        r = base.loc[c]
        year, award = (int(c[3:7]), c[-2:]) if c.startswith("BdO") else (int(c[:4]), c[-2:])
        rows.append([contest_name(c), RULE[r.rule], names.get((year, award, r.winner), r.winner), f"{r.margin:.2f}",
                     f"{r.p_flip:.2f}",
                     f"{r.p_flip_mle:.2f}", f"{reg.loc[c].p_flip:.2f}", f"{het.loc[c].p_flip:.2f}",
                     f"{share.loc[c].p_flip:.2f}" if c in share.index else "--"])
    return tabular(["Contest", "Rule", "Leader", "Margin", "P(all)", "P(all, estimates)", "P(regional)",
                    "P(all, by region)", "P(all, share rule)"], rows, "lllrrrrrr")


CONTRASTS = [  # (model, contrast as in contrasts.csv, description)
    ("A1_coaches", "nation_own_foreign - nation_team_foreign", "Foreign coaches: own nationality minus coached country"),
    ("A1_coaches", "confed_own_foreign - confed_team_foreign", "Foreign coaches: confederation of origin minus of work"),
    ("A1_coaches", "nation_team_domestic - nation_team_foreign", "Compatriot bonus: domestic minus foreign coaches"),
    ("A1x_coaches", "nation_own_foreign[MP] - nation_team_foreign[MP]", "Own nationality minus coached country, men's award"),
    ("A1x_coaches", "nation_own_foreign[WP] - nation_team_foreign[WP]", "Own nationality minus coached country, women's award"),
    ("A1x_coaches", "confed_own_foreign[MP] - confed_team_foreign[MP]", "Origin minus work confederation, men's award"),
    ("A1x_coaches", "confed_own_foreign[WP] - confed_team_foreign[WP]", "Origin minus work confederation, women's award"),
    ("A1x_coaches", "(nation_team_domestic - nation_team_foreign)[WP] - [MP]", "Domestic minus foreign, women's minus men's award"),
    ("A1x_coaches", "(nation_own_foreign - nation_team_foreign)[WP] - [MP]", "Own minus coached country, women's minus men's"),
    ("A1x_coaches", "(confed_own_foreign - confed_team_foreign)[WP] - [MP]", "Origin minus work region, women's minus men's"),
    ("A3_movers", "confed_team[mover] - confed_team_foreign[other]", "Workplace region: within movers minus between coaches"),
]


def sup_contrasts():
    from scipy.stats import norm
    c = pd.read_csv(RES / "mechanisms" / "contrasts.csv").set_index(["model", "contrast"])
    rows = []
    for model, name, desc in CONTRASTS:
        r = c.loc[(model, name)]
        p = 2 * norm.sf(abs(r.z))
        rows.append([desc, f"{r['diff']:.2f}", f"{r.se_cluster:.2f}", f"{r.z:.1f}",
                     f"{p:.4f}" if p >= 1e-4 else "<0.0001"])
    return tabular(["Contrast", "Difference", "s.e. (clustered)", "z", "P (two-sided)"], rows, "lrrrr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("sup_sources", sup_sources), ("sup_coaches", sup_coaches), ("sup_bdo", sup_bdo),
                     ("sup_estimates", sup_estimates), ("sup_fits", sup_fits), ("sup_contests", sup_contests),
                     ("sup_contrasts", sup_contrasts)):
        write(name, fn)
    print("tables: sup_sources, sup_coaches, sup_bdo, sup_estimates, sup_fits, sup_contests, sup_contrasts",
          file=sys.stderr)


if __name__ == "__main__":
    main()
