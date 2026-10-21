"""Parse the Eurovision juries' individual rankings in the grand final from Wikipedia country pages.

python -I nhb/src/parse_eurovision.py  ->  nhb/data/processed/esc_juror_ranks.csv, esc_national.csv,
                                           esc_parse_issues.csv

Each page "<Country> in the Eurovision Song Contest <year>" has a table "Detailed voting results from
<Country> (Final)" with one row per song: running order, competing country, each juror's rank, the
national jury's rank and points, and the televote's rank and points. The voting country's own song is
in the table with blank cells (jurors may not rank it).
  esc_juror_ranks.csv   year, voter, juror (1..k within the jury), song_country, rank
  esc_national.csv      year, voter, song_country, running_order, jury_rank, jury_points, tele_rank, tele_points
  esc_jurors.csv        year, voter, juror, name (names as listed on the page; left out of the deposit,
                        see make_release.py)
  esc_jury_totals.csv   year, song_country, total, jury_score, televote_score (main contest page)
  esc_sources.csv       title, revid of every downloaded page (the revision parsed)
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eurovision" / "wikipedia"
PROC = ROOT / "data" / "processed"
ALIASES = {"Czech Republic": "Czechia", "FYR Macedonia": "North Macedonia", "Macedonia": "North Macedonia",
           "F.Y.R. Macedonia": "North Macedonia", "The Netherlands": "Netherlands"}


def country(cell):
    cell = re.sub(r"\{\{\s*nowrap\s*\|([^{}]*)\}\}", r"\1", cell)          # {{nowrap|United Kingdom}}
    m = re.search(r"\{\{\s*(?:Esc|esc|Flagcountry|flagcountry|flag|Flag)\s*\|\s*([^|}]+)", cell)
    name = m.group(1) if m else re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", cell)
    name = re.sub(r"<[^>]+>|'''|''", "", name).strip()
    return ALIASES.get(name, name)


def value(cell):
    """The content of a wikitable cell: text after the attribute bar, without markup."""
    cell = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", cell, flags=re.S)
    if "|" in cell and not cell.strip().startswith("{{"):
        cell = cell.split("|")[-1]
    cell = re.sub(r"\{\{[^}]*\}\}|<[^>]+>|'''|''|&nbsp;", "", cell).strip()
    return cell


def split_cells(line):
    """'| a || b || c' -> ['a', 'b', 'c'], keeping template pipes intact."""
    line = line.lstrip("|").strip()
    out, depth, cur, i = [], 0, "", 0
    while i < len(line):
        if line.startswith("{{", i) or line.startswith("[[", i):
            depth += 1; cur += line[i:i + 2]; i += 2; continue
        if line.startswith("}}", i) or line.startswith("]]", i):
            depth -= 1; cur += line[i:i + 2]; i += 2; continue
        if depth == 0 and line.startswith("||", i):
            out.append(cur); cur = ""; i += 2; continue
        cur += line[i]; i += 1
    out.append(cur)
    return out


def final_table(text):
    """The wikitable whose caption names the final and whose header has juror columns."""
    for t in re.findall(r"\{\|.*?\n\|\}", text, flags=re.S):
        cap = re.search(r"\|\+(.*)", t)
        if not cap or not re.search(r"\((Grand )?[Ff]inal\)", cap.group(1)):
            continue
        if jury_span(t):
            return t
    return None


def jury_span(t):
    """Width of the "Jury" header: the jurors' columns plus the national rank and points."""
    m = re.search(r'^!\s*scope="col"\s*colspan="?(\d+)"?\s*\|\s*(?:\[\[[^\]]*\|)?Jury', t, flags=re.M)
    return int(m.group(1)) if m else 0


def parse_table(t):
    jurors = list(range(jury_span(t) - 2))       # juror columns, whatever their headers say
    rows = re.split(r"\n\|-[^\n]*\n", t)
    out = []
    for r in rows:
        lines = [l for l in r.split("\n") if l.strip()]
        ro = next((value(l.split("|", 1)[1] if "|" in l else l) for l in lines if l.startswith("! scope=\"row\"")), None)
        cells = [c for l in lines if l.startswith("|") and not l.startswith(("|+", "|}", "|-"))
                 for c in split_cells(l)]
        if ro is None or len(cells) < 1 + len(jurors):
            continue
        song = country(cells[0])
        vals = [value(c) for c in cells[1:]]
        out.append({"running_order": ro, "song_country": song, "jurors": vals[:len(jurors)],
                    "rest": vals[len(jurors):]})
    return jurors, out


def juror_names(text, voter):
    m = re.search(r"comprised the [^\n]*?jury[^\n]*:\s*\n((?:\*[^\n]*\n)+)", text)
    if not m:
        return []
    return [re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", l.lstrip("* ")).split("(")[0].strip()
            for l in m.group(1).strip().split("\n")]


def jury_totals(text, year):
    """Official jury score of every song in the final, from the main contest page's jury scoreboard."""
    i = text.find("jury voting results of the final")
    if i < 0:
        return []
    tab = text[i:text.find("\n|}", i)]
    heads = [next((a.strip() for a in args.split("|") if "=" not in a and a.strip()), "")
             for args in re.findall(r"vert header\|([^}]*)\}\}", tab)]
    names = [h for h in heads if h][:3]
    if "Total score" not in names:
        return []
    k_jury = next(k for k, h in enumerate(names) if h.startswith("Jury"))
    out = []
    for row in re.split(r"\n\|-[^\n]*\n", tab):
        lines = [l for l in row.split("\n") if l.strip()]
        # the song's cell is followed by three single numeric cells (total, jury, televote); the
        # winner's row has no scope="row" marker in some years
        num = lambda l: "||" not in l and l.startswith("|") and value(l).lstrip("-").isdigit()
        head = next((k for k in range(len(lines) - 3) if not num(lines[k]) and all(num(l) for l in lines[k + 1:k + 4])),
                    None)
        if head is None:
            continue
        cell = re.sub(r"\{\{\s*nowrap\s*\|([^{}]*)\}\}", r"\1", lines[head])
        song = country(cell) if re.search(r"\{\{\s*(Esc|esc|flag)", cell) else country(cell.split("|")[-1])
        nums = [value(l) for l in lines[head + 1:head + 4]]
        if len(nums) == 3 and all(n.lstrip("-").isdigit() for n in nums):
            out.append({"year": year, "song_country": song, "total": int(nums[0]),
                        "jury_score": int(nums[k_jury]), "televote_score": int(nums[3 - k_jury])})
    return out


def main():
    ranks, national, names, issues, sources = [], [], [], [], []
    for f in sorted(RAW.glob("*_in_the_Eurovision_Song_Contest_*.json")):
        d = json.loads(f.read_text())
        title, text = d["title"], d["wikitext"]
        sources.append({"title": title, "revid": d.get("revid")})
        m = re.match(r"(.+) in the Eurovision Song Contest (\d{4})$", title)
        voter, year = ALIASES.get(m.group(1), m.group(1)), int(m.group(2))
        t = final_table(text)
        if t is None:
            issues.append({"year": year, "voter": voter, "issue": "no final table with juror columns"})
            continue
        jurors, rows = parse_table(t)
        for j, nm in enumerate(juror_names(text, voter), start=1):
            names.append({"year": year, "voter": voter, "juror": j, "name": nm})
        for r in rows:
            for j, v in enumerate(r["jurors"], start=1):
                if v.isdigit():
                    ranks.append({"year": year, "voter": voter, "juror": j, "song_country": r["song_country"],
                                  "rank": int(v)})
            rest = r["rest"] + [""] * 4
            national.append({"year": year, "voter": voter, "song_country": r["song_country"],
                             "running_order": r["running_order"], "jury_rank": rest[0], "jury_points": rest[1],
                             "tele_rank": rest[2], "tele_points": rest[3]})
        if not rows:
            issues.append({"year": year, "voter": voter, "issue": "final table without rows"})
    totals = []
    for f in sorted(RAW.glob("Eurovision_Song_Contest_*.json")):
        d = json.loads(f.read_text())
        sources.append({"title": d["title"], "revid": d.get("revid")})
        y = int(re.search(r"(\d{4})$", d["title"]).group(1))
        t = jury_totals(d["wikitext"], y)
        if not t:
            issues.append({"year": y, "voter": "(main page)", "issue": "no jury scoreboard"})
        totals += t
    PROC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(totals).to_csv(PROC / "esc_jury_totals.csv", index=False)
    pd.DataFrame(sources).sort_values("title").to_csv(PROC / "esc_sources.csv", index=False)
    pd.DataFrame(ranks).to_csv(PROC / "esc_juror_ranks.csv", index=False)
    pd.DataFrame(national).to_csv(PROC / "esc_national.csv", index=False)
    pd.DataFrame(names).to_csv(PROC / "esc_jurors.csv", index=False)
    pd.DataFrame(issues, columns=["year", "voter", "issue"]).to_csv(PROC / "esc_parse_issues.csv", index=False)
    print(f"juror ranks {len(ranks)}, national rows {len(national)}, issues {len(issues)}", file=sys.stderr)


if __name__ == "__main__":
    main()
