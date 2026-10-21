"""Every number quoted in the NHB manuscript, as LaTeX macros: nhb/manuscript/generated/numbers.tex.

python -I nhb/src/make_numbers.py

Each macro is computed from nhb/data/processed or nhb/results. A value whose source is missing is
written as \\Pending{name}, which prints visibly and fails check_manuscript.py in a final build.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
OUT = ROOT / "manuscript" / "generated"


def fmt(x, d=2):
    if not np.isfinite(x):          # e.g. the mean over an empty selection: reported as pending
        raise ValueError(f"not finite: {x}")
    s = f"{x:,.{d}f}"
    return s.replace("-", "\\ensuremath{-}") if s.startswith("-") else s


def fmt_int(x):
    if not np.isfinite(x):
        raise ValueError(f"not finite: {x}")
    s = f"{int(round(x)):,}"
    return s.replace(",", "{,}").replace("-", "\\ensuremath{-}")


class Registry:
    def __init__(self):
        self.items = []

    def add(self, name, fn, note=""):
        self.items.append((name, fn, note))

    def write(self, path):
        lines, pending = [], 0
        for name, fn, note in self.items:
            try:
                v = fn()
            except (FileNotFoundError, KeyError, IndexError, ValueError, StopIteration, AttributeError) as e:
                v, pending = f"\\Pending{{{name}}}", pending + 1
                note = f"{note} [missing: {type(e).__name__}]"
            lines.append(f"\\newcommand{{\\{name}}}{{{v}}}  % {note}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"numbers.tex: {len(lines)} macros, {pending} pending", file=sys.stderr)
        return pending


def csv(name, base=PROC):
    return pd.read_csv(base / name, keep_default_na=False)


def data_numbers(reg):
    b = lambda: csv("fifa_ballots.csv")
    long = lambda: csv("fifa_long.csv")
    reg.add("NFifaBallots", lambda: fmt_int(len(b())), "FIFA ballots (test record excluded)")
    reg.add("NFifaContests", lambda: fmt_int(b().contest.nunique()), "FIFA contests")
    reg.add("NFifaRows", lambda: fmt_int(len(long())), "FIFA ballot x nominee rows")
    reg.add("NFifaVoterCountries", lambda: fmt_int(b().voter_country.nunique()), "voter countries, FIFA")
    reg.add("NFifaNominees", lambda: fmt_int(long().person.nunique()), "distinct FIFA nominees")
    for role, mac in (("captain", "Captain"), ("coach", "Coach"), ("media", "Media")):
        reg.add(f"NFifa{mac}Ballots", lambda r=role: fmt_int((b().role == r).sum()), f"FIFA {role} ballots")
    reg.add("NFifaMenBallots", lambda: fmt_int((b().award == "MP").sum()), "FIFA men's award ballots")
    reg.add("NFifaWomenBallots", lambda: fmt_int((b().award == "WP").sum()), "FIFA women's award ballots")
    reg.add("NFifaIncomplete", lambda: fmt_int(b().incomplete.astype(str).eq("True").sum()), "ballots with a placeholder")
    val = lambda: csv("fifa_validation.csv")
    reg.add("NOfficialCells", lambda: fmt_int(len(val())), "official top-3 group points checked")
    reg.add("NOfficialCellsExact", lambda: fmt_int((val()["diff"] == 0).sum()), "of which exact")
    sh = lambda: csv("fifa_validation_shares.csv").assign(a=lambda d: d["diff"].abs())
    reg.add("MaxShareDiffLate", lambda: fmt(sh()[sh().year >= 2013].a.max(), 2), "max |share diff| 2013-15, pp")
    reg.add("MaxShareDiffEarly", lambda: fmt(sh()[sh().year <= 2012].a.max(), 2), "max |share diff| 2010-12, pp")
    reg.add("NShareNominees", lambda: fmt_int(len(sh())), "nominees with published 2010-15 shares")
    bb = lambda: csv("bdo_ballots.csv")
    reg.add("NBdoBallots", lambda: fmt_int(len(bb())), "Ballon d'Or juror ballots")
    reg.add("NBdoContests", lambda: fmt_int(bb().groupby(["year", "award"]).ngroups), "Ballon d'Or contests")
    reg.add("NBdoJurorsMin", lambda: fmt_int(bb().groupby(["year", "award"]).size().min()), "smallest BdO jury")
    reg.add("NBdoJurorsMax", lambda: fmt_int(bb().groupby(["year", "award"]).size().max()), "largest BdO jury")
    reg.add("NBdoCountries", lambda: fmt_int(bb().country.nunique()), "BdO juror countries")
    reg.add("NAllBallots", lambda: fmt_int(len(b()) + len(bb())), "all ballots")
    reg.add("NAllContests", lambda: fmt_int(b().contest.nunique() + bb().groupby(["year", "award"]).ngroups),
            "all contests")


def results_numbers(reg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import design_sim
    long = lambda: csv("fifa_long.csv")

    def later_share():
        d = long()
        d = d[(d.same_nation == 1) & d.role.isin(["captain", "coach"]) & d.year.between(2011, 2013)]
        return fmt(100 * (d["rank"] > 0).mean(), 0) + "\\%"
    reg.add("ShareCompatPickedLater", later_share, "captains' and coaches' compatriot nominees picked, 2011-13")
    coach = lambda: csv("fifa_coach_nationality_all.csv")
    reg.add("NCoachBallots", lambda: fmt_int(len(coach())), "coach ballots")
    reg.add("NCoachKnown", lambda: fmt_int((coach().coach_nationality != "").sum()), "with coach nationality")
    reg.add("NCoachForeign", lambda: fmt_int(((coach().coach_nationality != "")
                                             & (coach().coach_nationality != coach().voter_country)).sum()),
            "foreign coaches' ballots")
    import make_figures
    reg.add("LabelThreshold", lambda: fmt(make_figures.LABEL_P, 1), "make_figures.LABEL_P")
    reg.add("PlaceboQLo", lambda: fmt(100 * make_figures.PLACEBO_Q[0], 1), "make_figures.PLACEBO_Q, per cent")
    reg.add("PlaceboQHi", lambda: fmt(100 * make_figures.PLACEBO_Q[1], 1), "make_figures.PLACEBO_Q, per cent")
    reg.add("SymlogLin", lambda: fmt(make_figures.SYMLOG_LIN, 1), "make_figures.SYMLOG_LIN")
    reg.add("DesignSizeMin", lambda: fmt_int(min(design_sim.SIZES)), "design_sim.SIZES")
    reg.add("DesignSizeMax", lambda: fmt_int(max(design_sim.SIZES)), "design_sim.SIZES")
    reg.add("DesignUefaN", lambda: fmt_int(design_sim.UEFA_JURY), "design_sim.UEFA_JURY")
    reg.add("DesignUefaMin", lambda: fmt_int(100 * min(design_sim.UEFA_SHARES)), "design_sim.UEFA_SHARES")
    reg.add("DesignUefaMax", lambda: fmt_int(100 * max(design_sim.UEFA_SHARES)), "design_sim.UEFA_SHARES")
    plac = lambda: csv("placebo.csv", RES / "robustness")
    reg.add("NPlaceboPerms", lambda: fmt_int(plac().groupby("shuffle").perm.nunique().min()), "placebo shuffles")
    cf = lambda: csv("cf_C1_role_contests.csv", RES / "combined")
    reg.add("NCfDraws", lambda: fmt_int(cf().draws.iloc[0]), "counterfactual parameter draws")
    dz = lambda: csv("design_contests_M1_role.csv", RES / "design")
    reg.add("NDesignReps", lambda: fmt_int(dz().query("design == 'fifa'").reps.min()), "replications per design cell")
    import fit_pooled
    reg.add("MinGroup", lambda: fmt_int(fit_pooled.MIN_GROUP), "fit_pooled.MIN_GROUP, ballots")
    reg.add("ExtremeAlpha", lambda: fmt_int(fit_pooled.EXTREME_ALPHA), "fit_pooled.EXTREME_ALPHA")
    import fifa_rules
    sc = [str(int(v)) for v in fifa_rules.SCORING]
    reg.add("ScoringPoints", lambda: ", ".join(sc[:-1]) + " and " + sc[-1], "fifa_rules.SCORING (art. 11)")
    import mechanisms
    reg.add("MinStint", lambda: fmt_int(mechanisms.MIN_STINT), "mechanisms.MIN_STINT, ballots per confederation")
    reg.add("MinStintWord", lambda: word(mechanisms.MIN_STINT), "mechanisms.MIN_STINT in words")


ROLE_MAC = {"captain": "Captain", "coach": "Coach", "media": "Media", "bdo": "Bdo"}
TERM_MAC = {"same_nation": "Nation", "same_confed": "Confed", "same_league": "League"}


def estimate_macros(reg, path, model, prefix, terms):
    """beta, 95% CI (clustered) and odds ratio for each term of a fitted model."""
    def get(term):
        e = pd.read_csv(path)
        r = e[(e.model == model) & (e.term == term)].iloc[0]
        return r.beta, r.beta - 1.96 * r.se_cluster, r.beta + 1.96 * r.se_cluster, r.se_cluster
    for term, mac in terms:
        src = f"{path.relative_to(ROOT)}:{model}:{term}"
        reg.add(f"{prefix}{mac}", lambda t=term: fmt(get(t)[0], 2), src)
        reg.add(f"{prefix}{mac}Lo", lambda t=term: fmt(get(t)[1], 2), src + " 95% CI")
        reg.add(f"{prefix}{mac}Hi", lambda t=term: fmt(get(t)[2], 2), src + " 95% CI")
        reg.add(f"{prefix}{mac}OR", lambda t=term: fmt(np.exp(get(t)[0]), 1), src + " odds ratio")


def bonus_numbers(reg):
    pooled = RES / "pooled" / "estimates.csv"
    comb = RES / "combined" / "estimates.csv"
    fifa_terms = [(f"{t}[{r}]", f"{TERM_MAC[t]}{ROLE_MAC[r]}") for t in TERM_MAC for r in ("captain", "coach", "media")]
    estimate_macros(reg, pooled, "M1_role", "B", fifa_terms)
    estimate_macros(reg, pooled, "M4_role_confFE", "BReg",
                    [(f"same_nation[{r}]", f"Nation{ROLE_MAC[r]}") for r in ("captain", "coach", "media")])
    estimate_macros(reg, pooled, "M2_role_award", "BAward",
                    [(f"same_nation[{r}:{a}]", f"Nation{ROLE_MAC[r]}{'Men' if a == 'MP' else 'Women'}")
                     for r in ("captain", "coach", "media") for a in ("MP", "WP")])
    estimate_macros(reg, comb, "C1_role", "B", [(f"{t}[bdo]", f"{TERM_MAC[t]}Bdo") for t in TERM_MAC])
    estimate_macros(reg, comb, "C4_role_confFE", "BReg", [("same_nation[bdo]", "NationBdo")])

    def or_range(terms, model, path=pooled):
        e = pd.read_csv(path)
        v = np.exp(e[(e.model == model) & e.term.isin(terms)].beta)
        return v.min(), v.max()
    nat = [f"same_nation[{r}]" for r in ("captain", "coach", "media")]
    reg.add("ORNationMin", lambda: fmt(or_range(nat, "M1_role")[0], 1), "smallest compatriot odds ratio, FIFA roles")
    reg.add("ORNationMax", lambda: fmt(or_range(nat, "M1_role")[1], 1), "largest compatriot odds ratio, FIFA roles")
    conf = [f"same_confed[{r}]" for r in ("captain", "coach", "media")]
    reg.add("ORConfedMin", lambda: fmt(or_range(conf, "M1_role")[0], 1), "smallest confederation odds ratio")
    reg.add("ORConfedMax", lambda: fmt(or_range(conf, "M1_role")[1], 1), "largest confederation odds ratio")


def mechanism_numbers(reg):
    path = RES / "mechanisms" / "estimates.csv"
    terms = [(f"{p}_{s}", f"{p.capitalize()}{s.title().replace('_', '')}")
             for p in ("nation", "confed", "league") for s in ("team_domestic", "team_foreign", "own_foreign")]
    estimate_macros(reg, path, "A1_coaches", "Coach", terms)
    cpath = RES / "mechanisms" / "contrasts.csv"

    def contrast(name, col):
        c = pd.read_csv(cpath)
        return c[(c.model == "A1_coaches") & (c.contrast == name)].iloc[0][col]
    for name, mac in (("nation_own_foreign - nation_team_foreign", "NationOwnVsTeam"),
                      ("confed_own_foreign - confed_team_foreign", "ConfedOwnVsTeam"),
                      ("nation_team_domestic - nation_team_foreign", "NationDomVsFor")):
        reg.add(f"Contrast{mac}", lambda n=name: fmt(contrast(n, "diff"), 2), f"{name}")
        reg.add(f"Contrast{mac}Z", lambda n=name: fmt(contrast(n, "z"), 1), f"{name}, z")
    # by award, from the joint fit (A1x): the coach terms of each award and their difference
    ax = lambda: pd.read_csv(path).query("model == 'A1x_coaches'").set_index("term")
    for aw, amac in (("MP", "Men"), ("WP", "Women")):
        for term, tmac in (("nation_team_domestic", "NationTeamDomestic"), ("nation_team_foreign", "NationTeamForeign"),
                           ("nation_own_foreign", "NationOwnForeign"), ("confed_team_foreign", "ConfedTeamForeign"),
                           ("confed_own_foreign", "ConfedOwnForeign")):
            reg.add(f"Coach{tmac}{amac}", lambda t=f"{term}[{aw}]": fmt(ax().loc[t].beta, 2), f"A1x {term}[{aw}]")
            reg.add(f"Coach{tmac}{amac}SE", lambda t=f"{term}[{aw}]": fmt(ax().loc[t].se_cluster, 2),
                    f"A1x {term}[{aw}], clustered s.e.")
        reg.add(f"NForeignTeamPairs{amac}", lambda t=f"nation_team_foreign[{aw}]": fmt_int(ax().loc[t].rows_nonzero),
                f"foreign-coach x nominee-from-coached-country rows, {aw}")

    def acon(name, col):
        c = pd.read_csv(cpath)
        return c[(c.model == "A1x_coaches") & (c.contrast == name)].iloc[0][col]
    for name, mac in (("(nation_team_domestic - nation_team_foreign)[WP] - [MP]", "AwardDiffDomFor"),
                      ("(nation_own_foreign - nation_team_foreign)[WP] - [MP]", "AwardDiffOwnTeam"),
                      ("(confed_own_foreign - confed_team_foreign)[WP] - [MP]", "AwardDiffConfedOwnTeam"),
                      ("nation_own_foreign[MP] - nation_team_foreign[MP]", "MenOwnVsTeam"),
                      ("nation_own_foreign[WP] - nation_team_foreign[WP]", "WomenOwnVsTeam"),
                      ("confed_own_foreign[MP] - confed_team_foreign[MP]", "MenConfedOwnVsTeam"),
                      ("confed_own_foreign[WP] - confed_team_foreign[WP]", "WomenConfedOwnVsTeam")):
        reg.add(f"Contrast{mac}", lambda n=name: fmt(acon(n, "diff"), 2), f"A1x {name}")
        reg.add(f"Contrast{mac}Z", lambda n=name: fmt(acon(n, "z"), 1), f"A1x {name}, z")
    a1 = lambda: pd.read_csv(path).query("model == 'A1_coaches'").set_index("term")
    reg.add("NForeignTeamPairs", lambda: fmt_int(a1().loc["nation_team_foreign"].rows_nonzero),
            "foreign-coach x nominee-from-coached-country rows")
    mfits = lambda: json.loads((RES / "mechanisms" / "fits.json").read_text())
    reg.add("NTransferPairsChange", lambda: fmt_int(next(f for f in mfits() if f["model"].startswith("B1"))
                                                    ["pairs_with_change"]), "nominee x host-country pairs that change state")
    mv = lambda: pd.read_csv(path).query("model == 'A3_movers'").set_index("term")
    fits = lambda: json.loads((RES / "mechanisms" / "fits.json").read_text())
    for term, mac in (("confed_team[mover]", "MoverConfed"), ("nation_team[mover]", "MoverNationTeam"),
                      ("nation_own_foreign[mover]", "MoverNationOwn")):
        reg.add(mac, lambda t=term: fmt(mv().loc[t].beta, 2), f"A3 {term}")
        reg.add(f"{mac}SE", lambda t=term: fmt(mv().loc[t].se_cluster, 2), f"A3 {term}, clustered SE")
    mcon = lambda: pd.read_csv(cpath).query("model == 'A3_movers'").set_index("contrast")
    reg.add("MoverVsOtherConfedZ", lambda: fmt(mcon().loc["confed_team[mover] - confed_team_foreign[other]"].z, 1),
            "within-coach minus between-coach workplace-region bonus, z")
    reg.add("NMoverBallots", lambda: fmt_int(next(f for f in fits() if f["model"] == "A3_movers")["mover_ballots"]),
            "ballots of coaches who moved between confederations")
    fits = lambda: json.loads((RES / "mechanisms" / "fits.json").read_text())
    reg.add("NMoverCoaches", lambda: fmt_int(next(f for f in fits() if f["model"] == "A3_movers")["mover_coaches"]),
            "coaches who voted from two or more confederations")


def outcome_numbers(reg):
    def cf(model="M1_role", base=RES / "pooled"):
        c = pd.read_csv(base / f"cf_{model}_contests.csv")
        return c[c.scenario == "all"]
    reg.add("NFifaFlipHalf", lambda: fmt_int((cf().p_flip > 0.5).sum()), "FIFA contests with P(flip) > 0.5")
    reg.add("NFifaFlipTenth", lambda: fmt_int((cf().p_flip > 0.1).sum()), "FIFA contests with P(flip) > 0.1")
    reg.add("ExpFifaFlips", lambda: fmt(cf().p_flip.sum(), 1), "expected number of FIFA winners changed")
    reg.add("MaxFifaFlip", lambda: fmt(cf().p_flip.max(), 2), "largest FIFA flip probability")
    def top():
        c = cf()
        return c.loc[c.p_flip.idxmax()]
    award = {"MP": "men's", "WP": "women's"}
    reg.add("MaxFifaFlipContest", lambda: f"the {top().contest[:4]} {award[top().contest[5:]]} award",
            "contest with the largest P(flip)")
    def display(person, contest):
        meta = csv("fifa_nominee_meta.csv")
        r = meta[(meta.year == int(contest[:4])) & (meta.award == contest[5:]) & (meta.person == person)]
        return r.wiki_name.iloc[0] if len(r) and r.wiki_name.iloc[0] else person
    reg.add("MaxFifaFlipCfWinner", lambda: display(top().cf_winner_mode, top().contest),
            "most frequent counterfactual winner there (Wikipedia spelling)")
    reg.add("NFifaFlipConfFE", lambda: fmt_int((cf("M4_role_confFE").p_flip > 0.5).sum()),
            "FIFA contests with P(flip) > 0.5 under nominee-by-region effects")
    comb = lambda: cf("C1_role", RES / "combined")
    for c, mac in (("BdO2019_MP", "BdoNineteen"), ("BdO2024_MP", "BdoTwentyFour")):
        reg.add(f"Flip{mac}", lambda c=c: fmt(comb().set_index("contest").loc[c].p_flip, 2), f"P(flip) {c}")
        reg.add(f"Margin{mac}", lambda c=c: fmt(comb().set_index("contest").loc[c].margin, 2), f"margin {c}, points")
    reg.add("NBdoFlipHalf", lambda: fmt_int((comb()[comb().contest.str.startswith("BdO")].p_flip > 0.5).sum()),
            "Ballon d'Or contests with P(flip) > 0.5")
    # all 40 contests, combined model, by which bonuses are removed
    def comb_s(scen):
        c = pd.read_csv(RES / "combined" / "cf_C1_role_contests.csv")
        return c[c.scenario == scen]
    reg.add("NAllFlipHalf", lambda: fmt_int((comb_s("all").p_flip > 0.5).sum()), "all contests, P(flip) > 0.5")
    reg.add("NAllFlipTenth", lambda: fmt_int((comb_s("all").p_flip > 0.1).sum()), "all contests, P(flip) > 0.1")
    reg.add("NAllFlipHalfWord", lambda: word((comb_s("all").p_flip > 0.5).sum()), "all contests, P(flip) > 0.5, words")
    reg.add("NAllFlipTenthWord", lambda: word((comb_s("all").p_flip > 0.1).sum()), "all contests, P(flip) > 0.1, words")
    for scen, mac in (("all", "All"), ("nation", "Nation"), ("confed", "Confed"), ("league", "League")):
        reg.add(f"ExpFlips{mac}", lambda sc=scen: fmt(comb_s(sc).p_flip.sum(), 1),
                f"expected number of changed winners, {scen} bonuses removed")
    one = lambda c, col, sc="all": comb_s(sc).set_index("contest").loc[c][col]
    for c, mac in (("2018_WP", "WomenEighteen"), ("BdO2019_MP", "BdoNineteen"), ("BdO2024_MP", "BdoTwentyFour")):
        reg.add(f"FlipC{mac}", lambda c=c: fmt(one(c, "p_flip"), 2), f"P(flip) {c}")
        reg.add(f"MarginC{mac}", lambda c=c: fmt(one(c, "margin"), 2), f"observed margin {c}")
        reg.add(f"CfMargin{mac}", lambda c=c: fmt(one(c, "cf_margin_mean"), 2), f"counterfactual margin {c}")
        reg.add(f"CfMargin{mac}Lo", lambda c=c: fmt(one(c, "cf_margin_q05"), 2), f"counterfactual margin {c}, 5%")
        reg.add(f"CfMargin{mac}Hi", lambda c=c: fmt(one(c, "cf_margin_q95"), 2), f"counterfactual margin {c}, 95%")
    def cf_level():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import counterfactuals
        return fmt_int(100 * (counterfactuals.Q_HI - counterfactuals.Q_LO))
    reg.add("CfIntervalLevel", cf_level, "counterfactuals.Q_LO/Q_HI, per cent")
    reg.add("FlipCWomenFourteen", lambda: fmt(one("2014_WP", "p_flip"), 2), "P(flip) 2014_WP, C1")
    reg.add("FlipConfedWomenEighteen", lambda: fmt(one("2018_WP", "p_flip", "confed"), 2), "2018_WP, regional bonus only")
    def disp(person, contest):
        m = csv("fifa_nominee_meta.csv")
        r = m[(m.year == int(contest[:4])) & (m.award == contest[5:]) & (m.person == person)]
        return r.wiki_name.iloc[0] if len(r) and r.wiki_name.iloc[0] else person
    k = "2024_MP"
    reg.add("JuryLeaderMenTwentyFour", lambda: disp(one(k, "winner"), k), "jury leader, 2024 men's award (scoring rule)")
    reg.add("JuryRunnerUpMenTwentyFour", lambda: disp(one(k, "runner_up"), k), "jury runner-up, 2024 men's award")
    reg.add("MarginCMenTwentyFour", lambda: fmt(one(k, "margin"), 2), "observed margin 2024_MP, % of scoring points")
    reg.add("FlipCMenTwentyFour", lambda: fmt(one(k, "p_flip"), 2), "P(flip) 2024_MP, C1")
    reg.add("FlipCMenTwentyFourHet", lambda: fmt(pd.read_csv(RES / "combined" / "cf_C2_confhet_contests.csv")
                                                  .query("scenario == 'all'").set_index("contest").loc[k].p_flip, 2),
            "P(flip) 2024_MP, C2")

    def group_margin(role):
        d = csv("fifa_long.csv")
        d = d[(d.contest == k) & (d.role == role)]
        pts = pd.to_numeric(d["rank"]).map({1: 5, 2: 3, 3: 1}).fillna(0)
        tot = pts.groupby(d.person).sum()
        return tot[one(k, "winner")] - tot[one(k, "runner_up")]
    reg.add("MediaMarginMenTwentyFour", lambda: fmt_int(group_margin("media")),
            "2024_MP: jury leader's lead in journalists' points over the runner-up")
    reg.add("CoachMarginMenTwentyFour", lambda: fmt_int(group_margin("coach")),
            "2024_MP: jury leader's lead in coaches' points over the runner-up")
    reg.add("CaptainMarginMenTwentyFour", lambda: fmt_int(-group_margin("captain")),
            "2024_MP: runner-up's lead in captains' points over the jury leader")

    # FIFA 2023-2025 under the 2010-2022 group-share rule (R2-1): the same counterfactuals for those six
    # contests, substituted into the all-contest totals
    def with_share(model):
        own = pd.read_csv(RES / "combined" / f"cf_{model}_contests.csv").query("scenario == 'all'").set_index("contest")
        sh = pd.read_csv(RES / "combined" / f"cf_{model}_share_contests.csv").query("scenario == 'all'").set_index("contest")
        out = own.copy()
        out.loc[sh.index, "p_flip"] = sh.p_flip
        return out, sh
    reg.add("ExpFlipsAllShare", lambda: fmt(with_share("C1_role")[0].p_flip.sum(), 1),
            "expected changed leaders, FIFA 2023-2025 under the share rule, C1")
    reg.add("ExpFlipsAllShareHet", lambda: fmt(with_share("C2_confhet")[0].p_flip.sum(), 1),
            "expected changed leaders, FIFA 2023-2025 under the share rule, C2")
    reg.add("NAllFlipHalfShareWord", lambda: word((with_share("C1_role")[0].p_flip > 0.5).sum()),
            "contests with P(flip) > 0.5, FIFA 2023-2025 under the share rule, C1")
    reg.add("NAllFlipHalfShareHetWord", lambda: word((with_share("C2_confhet")[0].p_flip > 0.5).sum()),
            "contests with P(flip) > 0.5, FIFA 2023-2025 under the share rule, C2")
    reg.add("ShareLeaderMenTwentyFour", lambda: disp(with_share("C1_role")[1].loc[k].winner, k),
            "2024_MP leader under the share rule")
    reg.add("ShareMarginMenTwentyFour", lambda: fmt(with_share("C1_role")[1].loc[k].margin, 2),
            "2024_MP margin under the share rule, percentage points of the jury vote")
    reg.add("FlipShareMenTwentyFour", lambda: fmt(with_share("C1_role")[1].loc[k].p_flip, 2),
            "2024_MP P(flip) under the share rule, C1")
    reg.add("MaxFlipShareScoringYears", lambda: fmt(with_share("C1_role")[1].p_flip.max(), 2),
            "largest P(flip) among FIFA 2023-2025 under the share rule, C1")
    # the same with the regional bonus specific to the juror's confederation (and organisation)
    het = lambda sc="all": pd.read_csv(RES / "combined" / "cf_C2_confhet_contests.csv").query("scenario == @sc")
    reg.add("NAllFlipHalfHet", lambda: fmt_int((het().p_flip > 0.5).sum()), "all contests, P(flip) > 0.5, C2")
    reg.add("NAllFlipTenthHet", lambda: fmt_int((het().p_flip > 0.1).sum()), "all contests, P(flip) > 0.1, C2")
    for scen, mac in (("all", "All"), ("nation", "Nation"), ("confed", "Confed"), ("league", "League")):
        reg.add(f"ExpFlips{mac}Het", lambda sc=scen: fmt(het(sc).p_flip.sum(), 1),
                f"expected number of changed winners, {scen} bonuses removed, C2")
    for c, mac in (("2018_WP", "WomenEighteen"), ("BdO2019_MP", "BdoNineteen"), ("BdO2024_MP", "BdoTwentyFour"),
                   ("2014_WP", "WomenFourteen")):
        reg.add(f"FlipC{mac}Het", lambda c=c: fmt(het().set_index("contest").loc[c].p_flip, 2), f"P(flip) {c}, C2")
    # parameter uncertainty: the same probabilities with the parameters fixed at their estimates
    reg.add("ExpFlipsAllMle", lambda: fmt(comb_s("all").p_flip_mle.sum(), 1), "expected changed winners at the MLE, C1")
    reg.add("FlipCWomenEighteenMle", lambda: fmt(one("2018_WP", "p_flip_mle"), 2), "P(flip) 2018_WP at the MLE, C1")
    reg.add("FlipCBdoTwentyFourMle", lambda: fmt(one("BdO2024_MP", "p_flip_mle"), 2), "P(flip) BdO2024 at the MLE, C1")

    # regional bonus by juror confederation (C2_confhet; FIFA's jurors)
    def het_est():
        e = pd.read_csv(RES / "combined" / "estimates.csv").query("model == 'C2_confhet'")
        e = e[e.term.str.startswith("same_confed[fifa:")].copy()
        e["confed"] = e.term.str.extract(r"fifa:(\w+)\]")[0]
        return e.set_index("confed")
    for conf in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "UEFA"):
        mac = "Het" + conf.capitalize()
        reg.add(mac, lambda c=conf: fmt(het_est().loc[c].beta, 2), f"regional bonus, FIFA jurors from {conf}")
        reg.add(mac + "Lo", lambda c=conf: fmt(het_est().loc[c].beta - 1.96 * het_est().loc[c].se_cluster, 2), "95% CI")
        reg.add(mac + "Hi", lambda c=conf: fmt(het_est().loc[c].beta + 1.96 * het_est().loc[c].se_cluster, 2), "95% CI")
    reg.add("HetBdoUefa", lambda: fmt(pd.read_csv(RES / "combined" / "estimates.csv").query(
        "model == 'C2_confhet' and term == 'same_confed[bdo:UEFA]'").beta.iloc[0], 2), "regional bonus, BdO jurors from UEFA")
    reg.add("MaxFlipConfFEComb", lambda: fmt(pd.read_csv(RES / "combined" / "cf_C4_role_confFE_contests.csv")
                                             .query("scenario == 'all'").p_flip.max(), 2),
            "largest P(flip) under nominee-by-region appeals (compatriot and league bonuses removed)")
    reg.add("AwardWinnerMenTwentyFour", lambda: disp(award_winners()["2024_MP"], "2024_MP"), "award winner, 2024 men's")
    reg.add("NFlipConfFEComb", lambda: fmt_int((pd.read_csv(RES / "combined" / "cf_C4_role_confFE_contests.csv")
                                               .query("scenario == 'all'").p_flip > 0.1).sum()),
            "contests with P(flip) > 0.1 under nominee-by-region appeals (compatriot and league bonuses removed)")


def robustness_numbers(reg):
    rob = RES / "robustness"
    plac = lambda: pd.read_csv(rob / "placebo.csv").assign(t=lambda d: d.term.str.split("[").str[0])

    def pl(shuffle, term, stat):
        p = plac()
        return getattr(p[(p["shuffle"] == shuffle) & (p.t == term)].beta, stat)()
    for sh, smac in (("all", "All"), ("within_confed", "Within")):
        for term, tmac in (("same_nation", "Nation"), ("same_confed", "Confed")):
            reg.add(f"Placebo{smac}{tmac}Mean", lambda s=sh, t=term: fmt(pl(s, t, "mean"), 2), f"placebo {sh} {term} mean")
            reg.add(f"Placebo{smac}{tmac}SD", lambda s=sh, t=term: fmt(pl(s, t, "std"), 2), f"placebo {sh} {term} sd")
    def role_max(file, shuffle, term):
        """Largest absolute mean of a profession's coefficient over the shuffles."""
        p = pd.read_csv(rob / file)
        p = p[(p["shuffle"] == shuffle) & p.term.str.startswith(term + "[")]
        return p.groupby("term").beta.mean().abs().max()
    for file, fmac in (("placebo.csv", ""), ("placebo_rolefe.csv", "RoleFe")):
        for term, tmac in (("same_nation", "Nation"), ("same_confed", "Confed")):
            reg.add(f"Placebo{fmac}All{tmac}MaxRole", lambda f=file, t=term: fmt(role_max(f, "all", t), 2),
                    f"{file}: largest |mean| of a profession's {term} coefficient, shuffle within contest and role")
    # leave one association out of the women's award (M2, role-by-award bonuses)
    lo_ = lambda: pd.read_csv(rob / "loao_women.csv")

    def loao_dev(prefix):
        l = lo_()
        base, rest = l[l.left_out == "none"].iloc[0], l[l.left_out != "none"]
        cols = [c for c in l.columns if c.startswith(prefix)]
        return max((rest[c] - base[c]).abs().max() for c in cols)
    reg.add("LoaoConfedMaxDev", lambda: fmt(loao_dev("b_same_confed"), 2),
            "largest change of a women's regional bonus when one association is left out")
    reg.add("LoaoNationMaxDev", lambda: fmt(loao_dev("b_same_nation"), 2),
            "largest change of a women's compatriot bonus when one association is left out")
    reg.add("NLoao", lambda: fmt_int((lo_().left_out != "none").sum()), "associations left out in turn")
    for c, mac in (("2018_WP", "Eighteen"), ("2014_WP", "Fourteen")):
        col = f"p_flip_mle_{c}"
        reg.add(f"LoaoFlip{mac}Lo", lambda col=col: fmt(lo_().query("left_out != 'none'")[col].min(), 2), f"{col} min")
        reg.add(f"LoaoFlip{mac}Hi", lambda col=col: fmt(lo_().query("left_out != 'none'")[col].max(), 2), f"{col} max")
        reg.add(f"LoaoFlip{mac}Base", lambda col=col: fmt(lo_().query("left_out == 'none'")[col].iloc[0], 2),
                f"{col} with every association (M2, parameters at the estimates)")
    reg.add("NPlaceboPermsRoleFe", lambda: fmt_int(pd.read_csv(rob / "placebo_rolefe.csv").groupby("shuffle").perm.nunique().min()),
            "placebo shuffles with profession-specific nominee effects")
    mv = lambda: pd.read_csv(rob / "multiverse.csv").assign(lo=lambda d: d.beta - 1.96 * d.se_cluster)

    def spec(prefix, stat):
        d = mv()[mv().term.str.startswith(prefix)]
        return {"n": d.groupby(["sample", "outcome", "fe"]).ngroups, "pos": int((d.lo > 0).sum()),
                "cells": len(d), "min": d.beta.min(), "max": d.beta.max()}[stat]
    reg.add("NSpecs", lambda: fmt_int(spec("same_nation", "n")), "multiverse specifications")
    reg.add("NSpecNationCells", lambda: fmt_int(spec("same_nation", "cells")), "compatriot coefficients (role x spec)")
    reg.add("NSpecNationPos", lambda: fmt_int(spec("same_nation", "pos")), "with 95% interval above zero")
    reg.add("SpecNationMin", lambda: fmt(spec("same_nation", "min"), 2), "smallest compatriot coefficient")
    reg.add("SpecNationMax", lambda: fmt(spec("same_nation", "max"), 2), "largest compatriot coefficient")
    reg.add("NSpecConfedCells", lambda: fmt_int(spec("same_confed", "cells")), "confederation coefficients")
    reg.add("NSpecConfedPos", lambda: fmt_int(spec("same_confed", "pos")), "with 95% interval above zero")

    def coupe(role):
        d = mv()
        r = d[(d["sample"] == "men2011-14") & (d.outcome == "first") & (d.fe == "contest")
              & (d.term == f"same_nation[{role}]")].iloc[0]
        return np.exp(r.beta)
    for role, mac in (("media", "Media"), ("coach", "Coach"), ("captain", "Captain")):
        reg.add(f"CoupeOR{mac}", lambda r=role: fmt(coupe(r), 0 if coupe(r) >= 20 else 1),
                f"first-choice compatriot odds ratio, men 2011-14, contest effects ({role})")
    st = lambda: pd.read_csv(rob / "strategic.csv").set_index("term")
    for role, mac in (("captain", "Captain"), ("coach", "Coach"), ("media", "Media")):
        reg.add(f"Rival{mac}", lambda r=role: fmt(st().loc[f"rival[{r}]"].beta, 2), f"rival coefficient, {role}")
        reg.add(f"Rival{mac}SE", lambda r=role: fmt(st().loc[f"rival[{r}]"].se_cluster, 2), f"rival s.e., {role}")


BLOC_CUTS = (0.05, 0.25)    # |share of leader's region - share of runner-up's region|: balanced below, unbalanced above


def design_numbers(reg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import design_sim
    gap = design_sim.GAPS[-2]                      # the moderate gap reported in the text
    for model, suffix in (("M1_role", ""), ("M5_role_confhet", "Het")):
        d = lambda m=model: pd.read_csv(RES / "design" / f"design_contests_{m}.csv")

        def flip(design, gap, d=d, cross=False):
            x = d()
            x = x[x.gap.astype(str) == str(gap)]
            if cross or design == "contenders_equal":   # races between two confederations
                x = x[x.contest.isin(design_sim.cross_contests(d()))]
            return x[x.design == design].p_flip.mean()
        for design, mac in (("fifa", "Fifa"), ("fifa_nocompat", "NoCompat"), ("media", "Media"),
                            ("uefa_0.1", "UefaLow"), ("uefa_0.9", "UefaHigh"), ("uefa_0.5", "UefaMid"),
                            ("size_25", "SizeSmall"), ("size_200", "SizeLarge"), ("equal_confed", "EqualConfed"),
                            ("contenders_equal", "ContEqual")):
            for g_, gmac in (("0.0", "Tie"), (str(gap), "Gap")):
                reg.add(f"Design{mac}{gmac}{suffix}", lambda a=design, g=g_, f=flip: fmt(f(a, g), 2),
                        f"mean P(flip), {model}, {design}, gap {g_}")
        for g_, gmac in (("0.0", "Tie"), (str(gap), "Gap")):
            reg.add(f"DesignFifaCross{gmac}{suffix}", lambda g=g_, f=flip: fmt(f("fifa", g, cross=True), 2),
                    f"mean P(flip), {model}, FIFA jury, contests with contenders from two confederations, gap {g_}")
        for design, mac in (("equal_confed", "EqualConfed"), ("uefa_0.1", "UefaLow"), ("uefa_0.9", "UefaHigh")):
            reg.add(f"Design{mac}CrossTie{suffix}", lambda a=design, f=flip: fmt(f(a, "0.0", cross=True), 2),
                    f"mean P(flip), {model}, {design}, contests with contenders from two confederations, tie")
        reg.add(f"NDesignCross{suffix}", lambda d=d: fmt_int(len(design_sim.cross_contests(d()))),
                f"FIFA contests whose leading nominees come from two confederations ({model})")

        def bloc(which, d=d):
            """Mean P(flip) at a tie over all simulated juries (every design but the compatriot ban, which
            keeps the FIFA jury), by the asymmetry of the two contenders' regional blocs."""
            x = d()
            x = x[(x.gap.astype(str) == "0.0") & (x.design != "fifa_nocompat")]
            same = ~x.contest.isin(design_sim.cross_contests(d()))
            a = x.bloc_gap.abs()
            sel = {"same": same, "low": ~same & (a < BLOC_CUTS[0]), "high": ~same & (a >= BLOC_CUTS[1]),
                   "mid": ~same & (a >= BLOC_CUTS[0]) & (a < BLOC_CUTS[1])}[which]
            return x[sel].p_flip.mean()
        for which, mac in (("same", "Same"), ("low", "Low"), ("mid", "Mid"), ("high", "High")):
            reg.add(f"DesignBloc{mac}{suffix}", lambda w=which, f=bloc: fmt(f(w), 2),
                    f"mean P(flip) at a tie, {model}, contenders' blocs: {which}")
    def cross_tie():
        x = pd.read_csv(RES / "design" / "design_contests_M1_role.csv")
        x = x[x.gap.astype(str) == "0.0"]
        return x[x.contest.isin(design_sim.cross_contests(x))].assign(a=lambda t: t.bloc_gap.abs())
    for design, mac in (("fifa", "Fifa"), ("uefa_0.1", "UefaLow"), ("uefa_0.9", "UefaHigh")):
        reg.add(f"DesignBlocAbs{mac}", lambda dz=design: fmt_int(100 * cross_tie().query("design == @dz").a.mean()),
                f"mean |contenders' bloc difference|, {design}, races between two confederations, percentage points")

    def bloc_corr():
        x = pd.read_csv(RES / "design" / "design_contests_M1_role.csv")
        x = x[(x.gap.astype(str) == "0.0") & (x.design != "fifa_nocompat")]
        x = x[x.contest.isin(design_sim.cross_contests(x))]
        return np.corrcoef(x.bloc_gap.abs(), x.p_flip)[0, 1]
    # contests near the half-of-draws threshold for a race between two confederations (R3-2)
    def knife():
        x = pd.read_csv(RES / "design" / "design_contests_M1_role.csv")
        x = x[x.gap.astype(str) == "0.0"]
        ce = x[x.design == "fifa"].set_index("contest").share_cross     # the FIFA design runs on every draw
        base = set(design_sim.cross_contests(x))
        near = sorted(c for c in ce.index if 0.4 <= ce[c] < 0.5)
        diffs = []
        for design in ("fifa", "equal_confed", "contenders_equal", "uefa_0.1", "uefa_0.9"):
            m0 = x[(x.design == design) & x.contest.isin(base)].p_flip.mean()
            m1 = x[(x.design == design) & x.contest.isin(base | set(near))].p_flip.mean()
            diffs.append(abs(m1 - m0))
        return near, ce, max(diffs)
    aw = {"MP": "men's", "WP": "women's"}
    reg.add("CrossNearContest", lambda: " and ".join(f"{c[:4]} {aw[c[5:]]}" for c in knife()[0]) or "none",
            "contests with 40-50% of draws having contenders from two confederations, M1")
    reg.add("CrossNearShare", lambda: ", ".join(fmt_int(100 * knife()[1][c]) + "\\%" for c in knife()[0]),
            "their share of such draws")
    reg.add("CrossNearMaxDiff", lambda: fmt(knife()[2], 3), "largest change of a cross-race design mean if they count")
    reg.add("DesignBlocCorr", lambda: fmt(bloc_corr(), 2), "correlation of |bloc difference| and P(flip) at a tie, M1")

    def bloc_corr_fifa():
        x = pd.read_csv(RES / "design" / "design_contests_M1_role.csv")
        x = x[x.gap.astype(str) == "0.0"]
        x = x[x.contest.isin(design_sim.cross_contests(x)) & (x.design == "fifa")]
        return np.corrcoef(x.bloc_gap.abs(), x.p_flip)[0, 1]
    reg.add("DesignBlocCorrFifa", lambda: fmt(bloc_corr_fifa(), 2),
            "correlation across contests within FIFA's own jury (races between two confederations), M1")
    # rule sensitivity of the design results for FIFA 2023-2025 (same random numbers: tags own6 / share /
    # scoring_zero), largest change of a design's mean P(flip) over the six contests, any gap
    def rule_diff(variant, model="M1_role"):
        a = pd.read_csv(RES / "design" / f"design_contests_{model}_own6.csv")
        b = pd.read_csv(RES / "design" / f"design_contests_{model}_{variant}.csv")
        ma = a.groupby(["design", a.gap.astype(str)]).p_flip.mean()
        mb = b.groupby(["design", b.gap.astype(str)]).p_flip.mean()
        return (ma - mb).abs().max()
    for variant, mac in (("share", "Share"), ("scoring_zero", "Zero")):
        reg.add(f"DesignRuleDiff{mac}", lambda v=variant: fmt(rule_diff(v), 2),
                f"largest change of a design mean (FIFA 2023-2025, M1) under the {variant} variant")
        reg.add(f"DesignRuleDiff{mac}Het", lambda v=variant: fmt(rule_diff(v, "M5_role_confhet"), 2),
                f"largest change of a design mean (FIFA 2023-2025, M5) under the {variant} variant")
    reg.add("BlocCutLow", lambda: fmt_int(100 * BLOC_CUTS[0]), "make_numbers.BLOC_CUTS, percentage points")
    reg.add("BlocCutHigh", lambda: fmt_int(100 * BLOC_CUTS[1]), "make_numbers.BLOC_CUTS, percentage points")
    reg.add("DesignEqualK", lambda: fmt_int(design_sim.EQUAL_K), "design_sim.EQUAL_K, jurors per confederation")
    reg.add("DesignEqualKMin", lambda: fmt_int(design_sim.EQUAL_K_MIN), "design_sim.EQUAL_K_MIN")
    reg.add("DesignGap", lambda: fmt(gap, 1), "design_sim.GAPS[-2], the moderate gap reported (log-odds)")
    d = lambda: pd.read_csv(RES / "design" / "design_contests_M1_role.csv")
    reg.add("DesignNoCompatMaxDiff", lambda: fmt((d()[d().design == "fifa_nocompat"].set_index(["contest", "gap"]).p_flip
                                                 - d()[d().design == "fifa"].set_index(["contest", "gap"]).p_flip)
                                                .abs().max(), 2), "largest |P(flip) change| from barring compatriots")
    for org, mac, f in (("FIFA", "Fifa", "fifa_long.csv"), ("BdO", "Bdo", "bdo_long.csv")):
        reg.add(f"PairShareNation{mac}", lambda f=f: fmt(100 * csv(f).same_nation.mean(), 1) + "\\%",
                f"share of juror-nominee pairs sharing nationality, {org}")
        reg.add(f"PairShareConfed{mac}", lambda f=f: fmt_int(100 * csv(f).same_confed.mean()) + "\\%",
                f"share of juror-nominee pairs sharing a confederation, {org}")
    cj = lambda: pd.read_csv(RES / "design" / "cross_jury_C1_role.csv")

    def cross(contest, jury):
        x = cj()
        return x[(x.contest == contest) & (x.jury == jury) & (x.gap.astype(str) == "obs")].p_flip.iloc[0]
    for contest, cmac in (("BdO2019_MP", "Nineteen"), ("BdO2024_MP", "TwentyFour")):
        for jury, jmac in (("actual", "Own"), ("fifa_jury", "Fifa"), ("global_press", "Global")):
            reg.add(f"Cross{jmac}{cmac}", lambda c=contest, j=jury: fmt(cross(c, j), 2),
                    f"cross-jury P(flip), {contest}, {jury}")


CARDINAL = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
            9: "nine", 10: "ten"}


def word(n):
    """Numbers below ten as words in running text (Nature style); larger ones as digits."""
    return CARDINAL[int(n)] if int(n) in CARDINAL else fmt_int(n)


ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
           8: "eighth", 9: "ninth", 10: "tenth"}


def award_winners():
    """Official winners of FIFA's awards from 2016 (fans included): FIFA's top-three table where it is
    printed (2023, 2024: rank 1, which settles the 2023 men's tie), otherwise the highest score in the
    nominee tables of FIFA's own award pages (wiki_score of rows taken from awards_{year}.json), and
    for contests with neither, data/processed/fifa_award_winners_manual.csv (winner and source)."""
    meta = csv("fifa_nominee_meta.csv")
    meta = meta[meta.source.str.startswith("awards_")].assign(
        s=lambda d: pd.to_numeric(d.wiki_score.str.replace("%", "").str.strip(), errors="coerce"))
    top3 = csv("fifa_official_top3.csv")
    names = csv("fifa_nominee_names.csv")
    person = {(y, a, k): p for y, a, k, p in names[["year", "award", "raw", "person"]].itertuples(index=False)}
    out = {}
    for (y, a), g in meta[meta.year >= 2016].groupby(["year", "award"]):
        t = top3[(top3.year == y) & (top3.award == a) & (top3["rank"] == 1)]
        if len(t):
            nm = t.name.iloc[0]
            out[f"{y}_{a}"] = g.person[g.wiki_name.eq(nm) | g.person.eq(nm)].iloc[0] if (
                g.wiki_name.eq(nm) | g.person.eq(nm)).any() else person.get((y, a, nm), nm)
        elif g.s.notna().any():
            out[f"{y}_{a}"] = g.loc[g.s.idxmax(), "person"]
    # winners missing from both sources, recorded by hand with their source (2025 women's award)
    man = PROC / "fifa_award_winners_manual.csv"
    if man.exists():
        for y, a, person in pd.read_csv(man, keep_default_na=False)[["year", "award", "person"]].itertuples(index=False):
            out.setdefault(f"{y}_{a}", person)
    return out


def jury_numbers(reg):
    cf = lambda: csv("cf_C1_role_contests.csv", RES / "combined").query("scenario == 'all'").set_index("contest")
    nom = lambda: csv("cf_C1_role_nominees.csv", RES / "combined").query("scenario == 'all'")
    meta = lambda: csv("fifa_nominee_meta.csv")

    def display(person, contest):
        m = meta()
        r = m[(m.year == int(contest[:4])) & (m.award == contest[5:]) & (m.person == person)]
        return r.wiki_name.iloc[0] if len(r) and r.wiki_name.iloc[0] else person

    def coincide():
        w, c = award_winners(), cf()
        return sum(c.loc[k, "winner"] == v for k, v in w.items()), len(w)
    reg.add("NJuryLeaderWon", lambda: fmt_int(coincide()[0]), "contests from 2016 whose jury leader won the award")
    reg.add("NJuryOnlyContests", lambda: fmt_int(coincide()[1]), "FIFA contests from 2016 with a known award winner")

    def exceptions():
        w, c = award_winners(), cf()
        aw = {"MP": "men's", "WP": "women's"}
        ks = sorted(k for k, v in w.items() if c.loc[k, "winner"] != v)
        names = [f"{k[:4]} {aw[k[5:]]}" for k in ks]
        if not names:
            raise ValueError("no exceptions")
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    reg.add("JuryLeaderExceptions", exceptions, "contests from 2016 whose jury leader did not win the award")
    k = "2018_WP"
    reg.add("JuryLeaderWomenEighteen", lambda: display(cf().loc[k, "winner"], k), "jury leader, 2018 women's award")
    reg.add("AwardWinnerWomenEighteen", lambda: display(award_winners()[k], k), "award winner, 2018 women's award")
    reg.add("CfLeaderWomenEighteen", lambda: display(cf().loc[k, "cf_winner_mode"], k),
            "most frequent counterfactual jury leader, 2018 women's award")

    def jury_rank():
        n = nom()
        n = n[n.contest == k].sort_values("share", ascending=False).reset_index(drop=True)
        return ORDINAL[int(n.index[n.person == award_winners()[k]][0]) + 1]
    reg.add("JuryRankAwardWinnerWomenEighteen", jury_rank, "award winner's place in the 2018 women's jury standing")
    est = lambda: csv("estimates.csv", RES / "mechanisms").query("model == 'A3_movers'").set_index("term")

    def hi_share():
        e = est()
        mv = e.loc["confed_team[mover]"]
        return fmt_int(100 * (mv.beta + 1.96 * mv.se_cluster) / e.loc["confed_team_foreign[other]"].beta) + "\\%"
    reg.add("MoverConfedHiShare", hi_share, "upper 95% limit of the within-coach workplace bonus / between-coach bonus")


def esc_numbers(reg):
    """Eurovision juries (second domain): data, bonuses under three groupings (B literature blocs, the main
    one; E UN subregions; C a finer grouping chosen after the main analysis), counterfactual jury
    standings and designs. Means over finals in the designs are weighted by replications."""
    E = RES / "eurovision"
    long = lambda: csv("esc_long.csv")
    val = lambda: csv("esc_validation.csv")
    reg.add("EscBallots", lambda: fmt_int(long().ballot.nunique()), "Eurovision juror rankings")
    reg.add("EscFinals", lambda: fmt_int(long().contest.nunique()), "Eurovision finals")
    reg.add("EscCountries", lambda: fmt_int(long().voter_country.nunique()), "Eurovision jury countries")
    reg.add("EscJuries", lambda: fmt_int(len(val())), "national juries with complete rankings")
    reg.add("EscFirstYear", lambda: str(int(long().year.min())), "first final")
    reg.add("EscLastYear", lambda: str(int(long().year.max())), "last final")
    size = lambda k: (pd.to_numeric(val().jurors) == k).sum()
    reg.add("NEscJuriesFourWord", lambda: word(size(4)), "juries with four ranked jurors (words)")
    reg.add("NEscJuriesSixWord", lambda: word(size(6)), "juries with six ranked jurors, 2026 (words)")
    tot = lambda: pd.read_csv(PROC / "esc_jury_totals.csv")

    def incomplete():
        n = tot().groupby("year").jury_score.sum() / sum(int(v) for v in __import__("esc_analysis").POINTS)
        return int(round((n - val().groupby(pd.to_numeric(val().year)).size()).sum()))
    reg.add("NEscIncomplete", lambda: fmt_int(incomplete()), "national juries that voted without complete published rankings")
    for col, mac in (("same_region", "Region"), ("same_bloc", "Bloc"), ("same_fine", "Fine")):
        reg.add(f"EscPairShare{mac}", lambda c=col: fmt_int(100 * long()[c].mean()) + "\\%", f"juror-song pairs, {col}")
    reg.add("EscPairShareBorder", lambda: fmt(100 * long().same_border.mean(), 1) + "\\%", "pairs sharing a land border")
    import build_eurovision
    import esc_analysis
    reg.add("NEscBlocsWord", lambda: word(len(build_eurovision.BLOCS)), "literature blocs (Gatherer 2006), words")
    reg.add("EscEqualK", lambda: word(esc_analysis.EQUAL_K), "esc_analysis.EQUAL_K, countries per group (words)")
    pts = [str(int(v)) for v in esc_analysis.POINTS]
    reg.add("EscPoints", lambda: ", ".join(pts[:-1]) + " and " + pts[-1], "esc_analysis.POINTS")
    bl = [fmt(b, 2) for b in esc_analysis.BASES if b is not None]
    reg.add("EscBasesList", lambda: ", ".join(bl[:-1]) + " and " + bl[-1], "esc_analysis.BASES, other exponential bases")
    reg.add("EscBase", lambda: fmt(pd.to_numeric(val().base, errors="coerce").dropna().iloc[0], 4),
            "exponential base of the national jury rule")
    late = lambda: val()[pd.to_numeric(val().year) >= 2018]
    reg.add("EscRulePoints", lambda: fmt_int((late().points_match.astype(str) == "True").sum()),
            "juries from 2018 whose points are reproduced")
    reg.add("EscRuleJuries", lambda: fmt_int(len(late())), "juries from 2018")
    # bonuses
    est = lambda: pd.read_csv(E / "estimates.csv").set_index(["model", "term"])

    def bonus(name, model, term, note):
        r = lambda: est().loc[(model, term)]
        reg.add(name, lambda: fmt(r().beta, 2), f"{note} ({model}, {term})")
        reg.add(f"{name}Lo", lambda: fmt(r().beta - 1.96 * r().se_cluster, 2), "95% CI")
        reg.add(f"{name}Hi", lambda: fmt(r().beta + 1.96 * r().se_cluster, 2), "95% CI")
    for g, gm in (("B", "Bloc"), ("E", "Region"), ("C", "Fine")):
        bonus(f"Esc{gm}", f"{g}1", "same_group", "common group bonus")
        bonus(f"Esc{gm}Top", f"{g}1_top10", "same_group", "common group bonus, top ten")
        bonus(f"Esc{gm}Border", f"{g}1", "same_border", "neighbour bonus")
        bonus(f"Esc{gm}BorderTop", f"{g}1_top10", "same_border", "neighbour bonus, top ten")
    for grp, mac in (("Balkan Bloc", "Balkan"), ("Viking Empire", "Viking"), ("Warsaw Pact", "Warsaw"),
                     ("Netherlands-Belgium", "Benelux")):
        bonus(f"EscBloc{mac}", "B2", f"same_group[{grp}]", "bloc-specific bonus")
    for grp, mac in (("Former Yugoslavia", "Yugo"), ("Nordic", "Nordic"), ("Baltic", "Baltic"),
                     ("British Isles", "Brit")):
        bonus(f"EscFine{mac}", "C2", f"same_group[{grp}]", "finer-group bonus")
    bonus("EscRegionGivenBloc", "B1_region", "same_region", "subregion bonus added to the bloc bonus")
    fits = lambda: {f["model"]: f for f in json.loads((E / "fits.json").read_text())}
    reg.add("EscLLBlocRegion", lambda: fmt(fits()["B1"]["loglik"] - fits()["E1"]["loglik"], 0),
            "log-likelihood, blocs minus subregions (same parameters)")
    reg.add("EscLLFineRegion", lambda: fmt(fits()["C1"]["loglik"] - fits()["E1"]["loglik"], 0),
            "log-likelihood, finer grouping minus subregions (same parameters)")
    fifa = lambda: pd.read_csv(RES / "combined" / "estimates.csv").query(
        "model == 'C1_role' and term.str.startswith('same_confed[') and term != 'same_confed[bdo]'", engine="python")
    reg.add("FifaConfedMin", lambda: fmt(fifa().beta.min(), 2), "smallest FIFA confederation bonus by profession (C1_role)")
    reg.add("FifaConfedMax", lambda: fmt(fifa().beta.max(), 2), "largest FIFA confederation bonus by profession (C1_role)")
    # counterfactual jury standings
    MODELS = ("B1", "B2", "B1_top10", "E1", "E2", "E1_top10", "C1", "C2", "C1_top10")
    cf = lambda m="B1", sc="all": pd.read_csv(E / f"cf_{m}_contests.csv").query("scenario == @sc")
    reg.add("NEscSpecsWord", lambda: word(len(MODELS)), "Eurovision specifications (words)")
    for m, mm in (("B1", ""), ("B2", "Het"), ("B1_top10", "Top")):
        for sc, mac in (("all", "All"), ("group", "Bloc"), ("border", "Border")):
            reg.add(f"ExpFlipsEsc{mac}{mm}", lambda m=m, sc=sc: fmt(cf(m, sc).p_flip.sum(), 2),
                    f"Eurovision expected changed jury leaders, {m}, {sc}")
    flips = lambda: pd.Series({m: cf(m).p_flip.sum() for m in MODELS})
    reg.add("ExpFlipsEscMin", lambda: fmt(flips().min(), 2), "smallest expected number over the nine specifications")
    reg.add("ExpFlipsEscMax", lambda: fmt(flips().max(), 2), "largest expected number over the nine specifications")
    reg.add("ExpFlipsEscMaxPer", lambda: fmt(flips().max() / len(cf()), 3), "the same per final")
    fifa_per = lambda f: pd.read_csv(RES / "combined" / f).query("scenario == 'all'").p_flip.mean()
    reg.add("ExpFlipsFifaPer", lambda: fmt(fifa_per("cf_C1_role_contests.csv"), 3), "football: expected changed winners per contest, C1")
    reg.add("ExpFlipsFifaPerHet", lambda: fmt(fifa_per("cf_C2_confhet_contests.csv"), 3), "the same, C2")
    top = lambda m="B1": cf(m).set_index("contest").p_flip.idxmax()
    one = lambda col, m="B1": cf(m).set_index("contest").loc[top(m), col]
    reg.add("NEscFlipTenthWord", lambda: word((cf().p_flip > 0.1).sum()), "finals with P(flip) > 0.1, B1 (words)")
    reg.add("MaxEscFlipYear", lambda: top()[3:], "final with the largest P(flip), B1")
    reg.add("MaxEscFlip", lambda: fmt(cf().p_flip.max(), 2), "its P(flip), B1")
    reg.add("MaxEscFlipHet", lambda: fmt(cf("B2").p_flip.max(), 2), "its P(flip), B2")
    reg.add("MaxEscFlipTop", lambda: fmt(cf("B1_top10").p_flip.max(), 2), "its P(flip), B1_top10")
    pmax = lambda: pd.Series({m: cf(m).set_index("contest").p_flip[top()] for m in MODELS})
    reg.add("NEscSpecsHalfWord", lambda: word((pmax() > 0.5).sum()), "specifications with P(flip) of that final above one half (words)")
    reg.add("MaxEscFlipMinModel", lambda: {"E1": "the United Nations' subregions with a common bonus"}.get(pmax().idxmin(), pmax().idxmin()),
            "specification with the smallest P(flip) of that final")
    reg.add("MaxEscFlipMin", lambda: fmt(pmax().min(), 2), "P(flip) of that final, smallest over nine specifications")
    reg.add("MaxEscFlipMax", lambda: fmt(pmax().max(), 2), "P(flip) of that final, largest over nine specifications")
    others = lambda: max(cf(m).set_index("contest").drop(top()).p_flip.max() for m in MODELS)
    reg.add("EscOtherFlipCeil", lambda: fmt(np.ceil(100 * others()) / 100, 2),
            "largest P(flip) of any other final, nine specifications, rounded up")
    reg.add("MaxEscFlipMargin", lambda: fmt_int(one("official_margin_points")), "official jury margin (points) of that final")
    reg.add("MaxEscFlipLeader", lambda: one("official_leader"), "its official jury leader")
    reg.add("MaxEscFlipRunner", lambda: one("official_runner_up"), "its official jury runner-up")

    def overall_place():
        t = tot()[tot().year == int(top()[3:])]
        lead = t.set_index("song_country").total.rank(ascending=False, method="min")[one("official_leader")]
        return ORDINAL[int(lead)]
    reg.add("MaxEscFlipOverall", lambda: overall_place(), "overall place (jury plus televote) of that jury leader")
    bases = lambda: (lambda b: b[b.contest == top()])(pd.read_csv(E / "bases_B1.csv"))
    alt = lambda: bases()[bases().fitted.astype(str) == "False"]
    reg.add("EscBaseFlipMin", lambda: fmt(alt()[alt().rule != "sum of ranks"].p_flip.min(), 2),
            "P(flip) of that final, other exponential bases, B1")
    reg.add("EscBaseFlipMax", lambda: fmt(alt()[alt().rule != "sum of ranks"].p_flip.max(), 2), "the same, largest")
    reg.add("EscBaseLo", lambda: fmt(float(alt()[alt().rule != "sum of ranks"].rule.astype(float).min()), 2), "smallest other base")
    reg.add("EscBaseHi", lambda: fmt(float(alt()[alt().rule != "sum of ranks"].rule.astype(float).max()), 2), "largest other base")
    reg.add("EscBaseFlipSum", lambda: fmt(alt()[alt().rule == "sum of ranks"].p_flip.iloc[0], 2),
            "P(flip) of that final with summed ranks, B1")
    chk = lambda: pd.read_csv(E / "standing_check_B1.csv")
    reg.add("NEscLeaderReproduced", lambda: fmt_int((chk().official_leader == chk().recomputed_leader).sum()),
            "finals whose recomputed jury leader is the official one")
    # designs
    d = lambda m="B1": pd.read_csv(E / f"design_{m}_contests.csv")

    def cross(m="B1"):
        """Finals whose two leading songs came from different groups in at least half the draws."""
        x = d(m)
        x = x[(x.gap.astype(str) == "0.0") & (x.design == "actual")]
        return sorted(x[x.share_cross >= 0.5].contest)

    def wmean(x, col="p_flip"):
        if x.reps.sum() == 0:
            raise ValueError("empty selection")
        return float(np.average(x[col], weights=x.reps))

    def dmean(design, gap="0.0", m="B1", only_cross=False):
        x = d(m)
        x = x[(x.gap.astype(str) == gap) & x.design.str.fullmatch(design)]
        if only_cross:
            x = x[x.contest.isin(cross(m))]
        return wmean(x)
    for design, mac in (("actual", "Actual"), ("border_ban", "BorderBan"), ("equal_groups", "EqualGroups")):
        for m, mm in (("B1", ""), ("B2", "Het")):
            reg.add(f"DesignEsc{mac}Tie{mm}", lambda a=design, m=m: fmt(dmean(a, "0.0", m), 2), f"{m}, {design}, tie, all finals")
    for design, mac in (("actual", "Actual"), ("equal_groups", "EqualGroups"), ("contenders_equal", "ContEqual"),
                        ("tilt_(lead|run)_half", "TiltHalf"), ("tilt_(lead|run)_full", "TiltFull")):
        for m, mm in (("B1", ""), ("B2", "Het"), ("E1", "Region"), ("C1", "Fine")):
            reg.add(f"DesignEsc{mac}CrossTie{mm}", lambda a=design, m=m: fmt(dmean(a, "0.0", m, True), 2),
                    f"{m}, {design}, tie, finals with contenders from two groups")
    reg.add("NDesignEscCrossWord", lambda: word(len(cross())), "finals with contenders from two blocs (words)")
    reg.add("EscContKMinWord", lambda: word(esc_analysis.CONTENDER_K_MIN), "esc_analysis.CONTENDER_K_MIN (words)")

    def one_bloc():
        """Cross finals (B1) by how many of the modal contenders' groups are literature blocs; for those
        with exactly one, the change in P(flip) at a tie when that contender's own bloc or the rival's
        group is removed (tilt_run_full removes the leader's group, tilt_lead_full the runner-up's)."""
        x = d()
        x = x[(x.gap.astype(str) == "0.0") & x.contest.isin(cross())]
        a = x[x.design == "actual"].set_index("contest")
        nb = {c: int(r.leader_group in build_eurovision.BLOCS) + int(r.runner_up_group in build_eurovision.BLOCS)
              for c, r in a.iterrows()}
        own, rival = [], []
        for c in [c for c, k in nb.items() if k == 1]:
            p = x[x.contest == c].set_index("design").p_flip
            lead_in = a.loc[c].leader_group in build_eurovision.BLOCS
            own.append(p["tilt_run_full" if lead_in else "tilt_lead_full"] - p["actual"])
            rival.append(p["tilt_lead_full" if lead_in else "tilt_run_full"] - p["actual"])
        return nb, np.array(own), np.array(rival)
    reg.add("NEscCrossNoBlocWord", lambda: word(sum(k == 0 for k in one_bloc()[0].values())),
            "cross finals in which neither contender is in a literature bloc (words)")
    reg.add("NEscCrossBothBlocWord", lambda: word(sum(k == 2 for k in one_bloc()[0].values())),
            "cross finals with both contenders in a bloc (words)")
    reg.add("NEscCrossOneBlocWord", lambda: word(sum(k == 1 for k in one_bloc()[0].values())),
            "cross finals with exactly one contender in a bloc (words)")
    reg.add("EscOwnBlocDropMin", lambda: fmt(-one_bloc()[1].max(), 2), "smallest fall in P(flip) when that contender's bloc is removed")
    reg.add("EscOwnBlocDropMax", lambda: fmt(-one_bloc()[1].min(), 2), "largest fall in P(flip) when that contender's bloc is removed")
    reg.add("EscRivalChangeCeil", lambda: fmt(np.ceil(100 * np.abs(one_bloc()[2]).max()) / 100, 2),
            "largest absolute change when the rival's group is removed, rounded up")
    reg.add("NDesignEscCrossRegionWord", lambda: word(len(cross("E1"))), "finals with contenders from two subregions (words)")
    for design, mac in (("tilt_lead_full", "TiltLeadFull"), ("tilt_run_full", "TiltRunFull")):
        for m, mm in (("B1", ""), ("B2", "Het"), ("E1", "Region")):
            reg.add(f"DesignEsc{mac}CrossTie{mm}", lambda a=design, m=m: fmt(dmean(a, "0.0", m, True), 2),
                    f"{m}, {design} (lead: runner-up's group removed; run: leader's group removed), tie, cross finals")
    conteq = lambda: d()[(d().gap.astype(str) == "0.0") & (d().design == "contenders_equal") & d().contest.isin(cross())]
    reg.add("DesignEscActualContEqTie", lambda: fmt(wmean(d()[(d().gap.astype(str) == "0.0") & (d().design == "actual")
                                                            & d().contest.isin(set(conteq().contest))]), 2),
            "B1, actual juries, tie, on the finals where the contenders' design ran")
    reg.add("NDesignEscContEqWord", lambda: word(conteq().contest.nunique()), "of these, finals with the contenders' design (words)")

    def bloc_abs(design, m="B1"):
        x = d(m)
        x = x[(x.gap.astype(str) == "0.0") & x.contest.isin(cross(m)) & x.design.str.fullmatch(design)]
        return wmean(x.assign(abs_gap=x.bloc_gap.abs()), "abs_gap")
    reg.add("DesignEscBlocAbsActual", lambda: fmt_int(100 * bloc_abs("actual")), "mean |bloc difference|, actual juries, pp")
    reg.add("DesignEscBlocAbsTiltFull", lambda: fmt_int(100 * bloc_abs("tilt_(lead|run)_full")),
            "mean |bloc difference| with one contender's group removed, pp")
    reg.add("DesignEscBlocAbsActualRegion", lambda: fmt_int(100 * bloc_abs("actual", "E1")), "the same, UN subregions, pp")
    reg.add("DesignEscBlocAbsTiltFullRegion", lambda: fmt_int(100 * bloc_abs("tilt_(lead|run)_full", "E1")),
            "mean |bloc difference| with one contender's subregion removed, pp")
    for design, mac in (("actual", "Actual"), ("border_ban", "BorderBan")):
        reg.add(f"DesignEsc{mac}TieFine", lambda a=design: fmt(dmean(a, "0.0", "C1"), 2), f"C1, {design}, tie, all finals")


def main():
    reg = Registry()
    data_numbers(reg)
    results_numbers(reg)
    bonus_numbers(reg)
    mechanism_numbers(reg)
    outcome_numbers(reg)
    robustness_numbers(reg)
    design_numbers(reg)
    jury_numbers(reg)
    esc_numbers(reg)
    pending = reg.write(OUT / "numbers.tex")
    return 0 if pending == 0 or "--allow-pending" in sys.argv else 1


if __name__ == "__main__":
    sys.exit(main())
