"""Figures of the NHB manuscript: nhb/manuscript/generated/fig*.pdf.

python -I nhb/src/make_figures.py

Colour: FIFA juries blue and Ballon d'Or juries orange (categorical slots 1-2 of the reference
palette, validated all-pairs: CVD dE 24.7, normal-vision dE 33.6); magnitudes (jury size, European
share) on the single-hue blue ramp. Juror types are labelled on the axis, never by colour alone.
A figure whose results are missing is skipped with a message.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "manuscript" / "generated"
MM = 1 / 25.4
FIFA, BDO = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
LABEL_P = 0.1          # contests above this flip probability are labelled in fig_outcomes
PLACEBO_Q = (0.025, 0.975)   # range of the placebo coefficients drawn in ed_placebo
SYMLOG_LIN = 0.1       # ed_placebo's symmetric-log axis is linear on (-SYMLOG_LIN, SYMLOG_LIN)
RAMP = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#104281"]
ROLE_LABEL = {"coach": "Coaches (FIFA)", "captain": "Captains (FIFA)", "media": "Journalists (FIFA)",
              "bdo": "Journalists (Ballon d'Or)"}
ROLE_ORDER = ["coach", "captain", "media", "bdo"]
TERM_LABEL = {"same_nation": "Compatriot", "same_confed": "Same confederation", "same_league": "Club in juror's country"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def need(path):
    if not path.exists():
        print(f"skipped: {path.relative_to(ROOT)} missing", file=sys.stderr)
        return False
    return True


def fig_bonuses():
    f = RES / "combined" / "estimates.csv"
    if not f.exists():   # preliminary draft: FIFA-only fit (identical FIFA coefficients), no Ballon d'Or rows
        f = RES / "pooled" / "estimates.csv"
        print("fig_bonuses: combined fit missing, using the FIFA-only fit", file=sys.stderr)
    if not need(f):
        return
    e = pd.read_csv(f).replace({"model": {"M1_role": "C1_role", "M4_role_confFE": "C4_role_confFE"}})
    e["lo"], e["hi"] = e.beta - 1.96 * e.se_cluster, e.beta + 1.96 * e.se_cluster
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM), sharey=True)
    for ax, term in zip(axes, TERM_LABEL):
        for k, role in enumerate(ROLE_ORDER):
            col = BDO if role == "bdo" else FIFA
            for model, filled, off in (("C1_role", True, 0.13), ("C4_role_confFE", False, -0.13)):
                r = e[(e.model == model) & (e.term == f"{term}[{role}]")]
                if len(r):
                    r = r.iloc[0]
                    y = -k + off
                    ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.1, solid_capstyle="round")
                    ax.plot(r.beta, y, "o", ms=4, mfc=col if filled else "white", mec=col, mew=1.0, zorder=3)
        ax.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(TERM_LABEL[term], loc="left", color=INK, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_xlabel("Bonus (log-odds)")
    axes[0].set_yticks([-k for k in range(len(ROLE_ORDER))], [ROLE_LABEL[r] for r in ROLE_ORDER])
    axes[0].set_ylim(-len(ROLE_ORDER) + 0.5, 0.5)
    h = [plt.Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=4, label="Contest-by-nominee effects"),
         plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=4,
                    label="Adding nominee-by-juror-region effects (same confederation not identified)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_bonuses.pdf")
    plt.close(fig)


def fig_coaches():
    f = RES / "mechanisms" / "estimates.csv"
    if not need(f):
        return
    e = pd.read_csv(f)
    e["lo"], e["hi"] = e.beta - 1.96 * e.se_cluster, e.beta + 1.96 * e.se_cluster
    a1 = e[e.model == "A1_coaches"].set_index("term")
    a3 = e[e.model == "A3_movers"].set_index("term")
    rows = [("team_domestic", "Team's country,\ndomestic coaches"), ("team_foreign", "Team's country,\nforeign coaches"),
            ("own_foreign", "Own nationality,\nforeign coaches")]
    fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 50 * MM), sharey=True)
    for ax, (pre, title) in zip(axes, (("nation", "Compatriot"), ("confed", "Same confederation"))):
        for k, (suf, _) in enumerate(rows):
            t = f"{pre}_{suf}"
            if t in a1.index:
                r = a1.loc[t]
                ax.plot([r.lo, r.hi], [-k, -k], color=FIFA, lw=1.1, solid_capstyle="round")
                ax.plot(r.beta, -k, "o", ms=4, color=FIFA, zorder=3)
        if pre == "confed" and "confed_team[mover]" in a3.index:
            r = a3.loc["confed_team[mover]"]
            ax.plot([r.lo, r.hi], [-3, -3], color=FIFA, lw=1.1)
            ax.plot(r.beta, -3, "D", ms=3.6, mfc="white", mec=FIFA, mew=1.0, zorder=3)
        ax.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(title, loc="left", color=INK, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_xlabel("Bonus (log-odds)")
    axes[0].set_yticks([0, -1, -2, -3], [lab for _, lab in rows] + ["Workplace region,\nsame coach moving"])
    axes[0].set_ylim(-3.5, 0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_coaches.pdf")
    plt.close(fig)


def fig_outcomes():
    f = RES / "combined" / "cf_C1_role_contests.csv"
    if not f.exists():   # preliminary draft: FIFA contests only
        f = RES / "pooled" / "cf_M1_role_contests.csv"
        print("fig_outcomes: combined counterfactuals missing, using FIFA contests only", file=sys.stderr)
    if not need(f):
        return
    c = pd.read_csv(f)
    c = c[c.scenario == "all"].copy()
    c["org"] = np.where(c.contest.str.startswith("BdO"), "Ballon d'Or", "FIFA")
    fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
    for org, col in (("FIFA", FIFA), ("Ballon d'Or", BDO)):
        d = c[c.org == org]
        if not len(d):
            continue
        ax.plot(d.margin, d.p_flip, "o", ms=4.2, color=col, mec="white", mew=0.6, label=f"{org} ({len(d)} contests)")
    for _, r in c[c.p_flip >= LABEL_P].iterrows():
        name = (r.contest.replace("BdO", "Ballon d'Or ") if r.contest.startswith("BdO") else "FIFA " + r.contest)
        name = name.replace("_MP", " men").replace("_WP", " women")
        ax.annotate(name, (r.margin, r.p_flip), xytext=(4, 2), textcoords="offset points", fontsize=5.5, color=INK2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Leader's margin over runner-up, percentage points: of the jury vote (FIFA to 2022),\n"
                  "of scoring points (FIFA from 2023), of all points cast (Ballon d'Or)", fontsize=5.8)
    ax.set_ylabel("Probability that the bonuses\nchanged the leader")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_outcomes.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_design(model="M1_role", name="fig_design"):
    f = RES / "design" / f"design_contests_{model}.csv"
    if not need(f):
        return
    d = pd.read_csv(f)
    d = d[d.gap.astype(str) != "obs"].copy()
    d["gap"] = d.gap.astype(float)
    m = d.groupby(["design", "gap"]).p_flip.mean().reset_index()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import design_sim
    cross = d[d.contest.isin(design_sim.cross_contests(d, 0.0))]   # contenders from two regions
    mc = cross.groupby(["design", "gap"]).p_flip.mean().reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 72 * MM))
    panels = [("Jury rules", m, [("fifa", "FIFA jury (three groups)", FIFA, "-"),
                                 ("media", "Its journalists only", BDO, "-"),
                                 ("fifa_nocompat", "FIFA jury, compatriots barred", INK2, "--")]),
              ("Jury size (random jurors)", m, [(f"size_{n}", f"{n} jurors", c, "-")
                                                for n, c in zip(design_sim.SIZES, RAMP[-len(design_sim.SIZES):])]),
              (f"Regional composition ({cross.contest.nunique()} races\nbetween two confederations)", mc,
               [("fifa", "FIFA jury", INK, "-")]
               + [(f"uefa_{s}", f"{int(round(100 * s))}% European", c, "-")
                  for s, c in zip(design_sim.UEFA_SHARES[::2], RAMP[::2])]
               + [("equal_confed", "Equal shares, six regions", BDO, "--"),
                  ("contenders_equal", "Contenders' regions equal", INK2, ":")])]
    for ax, (title, tab, series) in zip(axes[:3], panels):
        for key, lab, col, ls in series:
            s_ = tab[tab.design == key]
            if len(s_):
                ax.plot(s_.gap, s_.p_flip, ls, marker="o", ms=2.6, color=col, label=lab,
                        zorder=3 if ls != "-" else 2)
        ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=6.5)
        ax.set_xlabel("Gap in appeal (log-odds)")
        ax.grid(color=GRID, lw=0.4)
        ax.legend(frameon=False, fontsize=5, ncol=2 if len(series) > 3 else 1, columnspacing=0.8, handlelength=1.6,
                  loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ymax = max(a.get_ylim()[1] for a in axes[:3])
    for a in axes[:3]:
        a.set_ylim(0, ymax)
    axes[0].set_ylabel("Probability that similarity\ndecides the winner")
    ax = axes[3]
    t = d[(d.gap == 0) & (d.design != "fifa_nocompat")]
    same = ~t.contest.isin(cross.contest.unique())
    ax.plot(t[~same].bloc_gap.abs(), t[~same].p_flip, "o", ms=2.2, color=FIFA, alpha=0.55, mew=0,
            label="Contenders from two regions")
    ax.plot(t[same].bloc_gap.abs(), t[same].p_flip, "o", ms=2.2, color=INK2, alpha=0.55, mew=0,
            label="Contenders from one region")
    ax.set_title("Bloc asymmetry\n(tied races)", loc="left", color=INK, fontweight="bold", fontsize=6.5)
    ax.set_xlabel("Difference in the jury shares of\nthe contenders' confederations")
    ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_esc():
    """Eurovision juries: group bonuses under three groupings next to FIFA's regional bonuses, bloc-specific
    bonuses, jury design and bloc asymmetry (designs under B1, the literature blocs)."""
    E = RES / "eurovision"
    fe, fd, fc = E / "estimates.csv", E / "design_B1_contests.csv", RES / "combined" / "estimates.csv"
    if not (need(fe) and need(fd) and need(fc)):
        return
    est, d, fifa = pd.read_csv(fe), pd.read_csv(fd), pd.read_csv(fc)
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 118 * MM))
    ax = axes[0, 0]
    rows = []
    f = fifa[(fifa.model == "C1_role") & fifa.term.str.startswith("same_confed[")]
    for _, r in f.iterrows():
        role = r.term.split("[")[1][:-1]
        rows.append((f"Region, {ROLE_LABEL[role]}", r.beta, r.se_cluster, FIFA if role != "bdo" else BDO))
    for g, lab in (("B", "voting blocs"), ("E", "UN subregions"), ("C", "finer grouping")):
        for model, places in ((f"{g}1", "all places"), (f"{g}1_top10", "top ten")):
            e = est[est.model == model].set_index("term").loc["same_group"]
            rows.append((f"Eurovision, {lab} ({places})", e.beta, e.se_cluster, INK if g == "B" else INK2))
    for model, places in (("B1", "all places"), ("B1_top10", "top ten")):
        e = est[est.model == model].set_index("term").loc["same_border"]
        rows.append((f"Eurovision, neighbour ({places})", e.beta, e.se_cluster, INK2))
    for k, (lab, b, se, col) in enumerate(rows):
        ax.errorbar(b, k, xerr=1.96 * se, fmt="o", ms=3, color=col, elinewidth=1, capsize=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=5.3)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Bonus (log-odds per ranking step)")
    ax.set_title("a  Common bonuses", loc="left", color=INK, fontweight="bold", fontsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax = axes[0, 1]
    b2 = est[(est.model == "B2") & est.term.str.startswith("same_group[")].copy()
    names = {"Balkan Bloc": "Balkan bloc", "Viking Empire": "Nordic and Baltic bloc",
             "Warsaw Pact": "Russia, Ukraine, Poland", "Netherlands-Belgium": "Netherlands, Belgium"}
    b2["group"] = b2.term.str.slice(11, -1).map(lambda g: names.get(g, g))
    b2 = b2.sort_values("beta")
    for k, (_, r) in enumerate(b2.iterrows()):
        ax.errorbar(r.beta, k, xerr=1.96 * r.se_cluster, fmt="o", ms=3, elinewidth=1, capsize=0, color=INK)
    ax.set_yticks(range(len(b2)), b2.group, fontsize=5.3)
    ax.axvline(0, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Bonus within the juror's group (log-odds per ranking step)")
    ax.set_title("b  Bloc-specific bonuses", loc="left", color=INK, fontweight="bold", fontsize=6.5)
    ax.grid(axis="x", color=GRID, lw=0.4)
    # finals whose two leading songs came from different blocs in at least half the draws
    t0 = d[(d.gap.astype(str) == "0.0") & (d.design == "actual")]
    cross = set(t0[t0.share_cross >= 0.5].contest)
    ax = axes[1, 0]
    x = d[(d.gap.astype(str) != "obs") & d.contest.isin(cross)].copy()
    x["gap"] = x.gap.astype(float)
    x["w"] = x.p_flip * x.reps
    for name, lab, col, ls in (("actual", "Actual juries", INK, "-"), ("border_ban", "Neighbours barred", INK2, "--"),
                               ("equal_groups", "Equal countries per group", BDO, "-"),
                               ("contenders_equal", "Contenders' groups equal", FIFA, ":"),
                               ("tilt_lead_full", "Runner-up's group removed", INK2, "-."),
                               ("tilt_run_full", "Leader's group removed", FIFA, "-.")):
        g = x[x.design.str.fullmatch(name)].groupby("gap")[["w", "reps"]].sum()
        if len(g):
            ax.plot(g.index, g.w / g.reps, ls, marker="o", ms=2.6, color=col, label=lab)
    ax.set_xlabel("Gap in appeal (log-odds)")
    ax.set_ylabel("Probability that similarity\ndecides the jury winner")
    ax.set_ylim(0, None)
    ax.set_title("c  Jury design", loc="left", color=INK, fontweight="bold", fontsize=6.5)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax = axes[1, 1]
    t = d[(d.gap.astype(str) == "0.0") & (d.design != "border_ban")]
    same = ~t.contest.isin(cross)
    ax.plot(t[~same].bloc_gap.abs(), t[~same].p_flip, "o", ms=2.4, color=FIFA, alpha=0.7, mew=0,
            label="Contenders from two groups")
    ax.plot(t[same].bloc_gap.abs(), t[same].p_flip, "o", ms=2.4, color=INK2, alpha=0.7, mew=0,
            label="Contenders from one group")
    ax.set_xlabel("Difference in the jury shares of\nthe contenders' groups")
    ax.set_ylabel("Probability that similarity\ndecides the jury winner (tie)")
    ax.set_ylim(0, None)
    ax.set_title("d  Bloc asymmetry (tied races)", loc="left", color=INK, fontweight="bold", fontsize=6.5)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_esc.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def ed_design_het():
    fig_design("M5_role_confhet", "ed_design_het")


def fig_data():
    proc = ROOT / "data" / "processed"
    fb, bb = proc / "fifa_ballots.csv", proc / "bdo_ballots.csv"
    if not (need(fb) and need(bb)):
        return
    fifa = pd.read_csv(fb, keep_default_na=False)
    bdo = pd.read_csv(bb, keep_default_na=False)
    countries = pd.read_csv(proc / "countries.csv", keep_default_na=False)
    confed = dict(zip(countries.country, countries.confederation))
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 50 * MM), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    f = fifa.groupby(["year", "award"]).size().groupby("year").mean()
    b = bdo.groupby(["year", "award"]).size().groupby("year").mean()
    ax.plot(f.index, f.values, "-o", ms=3, color=FIFA, label="FIFA (coaches, captains, journalists)")
    ax.plot(b.index, b.values, "o", ms=4, color=BDO, label="Ballon d'Or (journalists)")
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Jurors per contest")
    ax.set_title("a  Jury size", loc="left", color=INK, fontweight="bold")
    ax.grid(color=GRID, lw=0.4)
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    ax = axes[1]
    order = ["UEFA", "CAF", "AFC", "CONCACAF", "CONMEBOL", "OFC"]
    fs = fifa.voter_country.map(confed).value_counts(normalize=True).reindex(order).fillna(0)
    bs = bdo.country.map(confed).value_counts(normalize=True).reindex(order).fillna(0)
    y = np.arange(len(order))
    ax.barh(y - 0.2, 100 * fs.values, height=0.38, color=FIFA, label="FIFA")
    ax.barh(y + 0.2, 100 * bs.values, height=0.38, color=BDO, label="Ballon d'Or")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("Share of jurors (%)")
    ax.set_title("b  Where jurors come from", loc="left", color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=5.5)
    ax = axes[2]
    cpath = proc / "fifa_coach_nationality_all.csv"
    if cpath.exists():
        c = pd.read_csv(cpath, keep_default_na=False)
        c = c[c.coach_nationality != ""].assign(confed=lambda d: d.voter_country.map(confed),
                                                foreign=lambda d: d.coach_nationality != d.voter_country)
        share = c.groupby("confed").foreign.mean().reindex(order) * 100
        n = c.groupby("confed").size().reindex(order).fillna(0).astype(int)
        ax.barh(y, share.values, height=0.55, color=FIFA)
        for k, (v, m) in enumerate(zip(share.values, n.values)):
            if not np.isnan(v):
                ax.text(v + 1.5, k, f"n = {m}", va="center", fontsize=5.5, color=INK2)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Coach ballots cast by foreign coaches (%)")
        ax.set_title("c  Foreign national-team coaches", loc="left", color=INK, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_data.pdf")
    plt.close(fig)


def ed_placebo():
    pf, mf, ef = RES / "robustness" / "placebo.csv", RES / "robustness" / "multiverse.csv", RES / "pooled" / "estimates.csv"
    if not (need(pf) and need(mf) and need(ef)):
        return
    pl, mu, est = pd.read_csv(pf), pd.read_csv(mf), pd.read_csv(ef)
    rf = RES / "robustness" / "placebo_rolefe.csv"
    pr = pd.read_csv(rf) if rf.exists() else None
    obs = est[est.model == "M1_role"].set_index("term")
    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    # placebo coefficients sit near zero and observed ones near 1-3: a symmetric-log axis shows both
    series = [(pl, "all", "Shuffled within contest and profession", INK2),
              (pr, "all", "The same, profession-specific nominee effects", RAMP[1]),
              (pl, "within_confed", "Shuffled within contest, profession and confederation", FIFA)]
    roles = ("coach", "captain", "media")
    for k, term in enumerate(("same_nation", "same_confed")):
        ax = fig.add_subplot(gs[0, k])
        for i, r in enumerate(roles):
            for j, (tab, sh, lab, col) in enumerate(series):
                if tab is None:
                    continue
                v = tab[(tab["shuffle"] == sh) & (tab.term == f"{term}[{r}]")].beta
                y = i - 0.24 + 0.16 * j
                ax.hlines(y, v.quantile(PLACEBO_Q[0]), v.quantile(PLACEBO_Q[1]), color=col, lw=1.0)
                ax.plot(v.mean(), y, "o", ms=2.6, color=col, label=lab if i == 0 else None)
            o = obs.loc[f"{term}[{r}]"]
            y = i + 0.24
            ax.hlines(y, o.beta - 1.96 * o.se_cluster, o.beta + 1.96 * o.se_cluster, color=BDO, lw=1.0)
            ax.plot(o.beta, y, "o", ms=2.6, color=BDO, label="Observed" if i == 0 else None)
        ax.set_xscale("symlog", linthresh=SYMLOG_LIN, linscale=0.6)
        ax.set_xlim(-0.3, 4 if term == "same_nation" else 1.5)
        ax.set_xticks([t for t in (-0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2, 4) if t <= ax.get_xlim()[1]])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.axvline(0, color=INK2, lw=0.5, ls=(0, (2, 2)))
        ax.set_yticks(range(len(roles)), [ROLE_LABEL[r] for r in roles])
        ax.invert_yaxis()
        ax.set_title(TERM_LABEL[term] + ": placebo and observed", loc="left", color=INK, fontweight="bold")
        ax.set_xlabel("Bonus (log-odds; symmetric log scale)")
        ax.grid(axis="x", color=GRID, lw=0.4)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=5.5, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    ax = fig.add_subplot(gs[1, :])
    m = mu[mu.term.str.startswith("same_nation")].copy()
    m["role"] = m.term.str.extract(r"\[(\w+)\]")[0]
    m["lo"], m["hi"] = m.beta - 1.96 * m.se_cluster, m.beta + 1.96 * m.se_cluster
    m = m.sort_values(["role", "beta"]).reset_index(drop=True)
    x = 0
    for role in ("coach", "captain", "media"):
        d = m[m.role == role]
        xs = np.arange(x, x + len(d))
        ax.vlines(xs, d.lo, d.hi, color=FIFA, lw=0.8)
        ax.plot(xs, d.beta, "o", ms=2.6, color=FIFA)
        ax.text(xs.mean(), ax.get_ylim()[1] if False else d.hi.max() + 0.2, ROLE_LABEL[role], ha="center",
                fontsize=6, color=INK)
        x += len(d) + 3
    ax.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xticks([])
    ax.set_ylabel("Compatriot bonus (log-odds)")
    ax.set_title("Compatriot bonus across fixed effects, samples and outcomes (sorted within profession)",
                 loc="left", color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, lw=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ed_placebo.pdf")
    plt.close(fig)


def ed_transfers():
    f = RES / "mechanisms" / "estimates.csv"
    if not need(f):
        return
    e = pd.read_csv(f)
    e["lo"], e["hi"] = e.beta - 1.96 * e.se_cluster, e.beta + 1.96 * e.se_cluster
    rows = [("B1_transfers_within", "league_current[coach]", "During, coaches (within pair)"),
            ("B1_transfers_within", "league_current[captain]", "During, captains (within pair)"),
            ("B1_transfers_within", "league_current[media]", "During, journalists (within pair)"),
            ("B1_transfers_within", "league_past", "After (within pair)"),
            ("B2_transfers_between", "league_current[coach]", "During, coaches (between pairs)"),
            ("B2_transfers_between", "league_current[captain]", "During, captains (between pairs)"),
            ("B2_transfers_between", "league_current[media]", "During, journalists (between pairs)"),
            ("B2_transfers_between", "league_past", "After (between pairs)"),
            ("B2_transfers_between", "league_future", "Before the move (between pairs)")]
    fig, ax = plt.subplots(figsize=(110 * MM, 62 * MM))
    labels = []
    for k, (model, term, lab) in enumerate(rows):
        r = e[(e.model == model) & (e.term == term)]
        if len(r):
            r = r.iloc[0]
            col = FIFA if model.startswith("B1") else INK2
            ax.plot([r.lo, r.hi], [-k, -k], color=col, lw=1.1)
            ax.plot(r.beta, -k, "o", ms=4, color=col)
        labels.append(lab)
    ax.set_yticks([-k for k in range(len(rows))], labels)
    ax.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("Bonus for a club in the juror's country (log-odds)")
    ax.grid(axis="x", color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "ed_transfers.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_data, fig_bonuses, fig_coaches, fig_outcomes, fig_design, fig_esc, ed_placebo, ed_transfers,
              ed_design_het):
        f()


if __name__ == "__main__":
    main()
