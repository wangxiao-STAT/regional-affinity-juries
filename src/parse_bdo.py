"""Parse the Ballon d'Or journalist ballots into one row per juror, with picks as printed.

python -I nhb/src/parse_bdo.py  ->  nhb/data/processed/bdo_ballots_raw.csv, bdo_wiki_ballots.csv,
                                    bdo_official_points.csv, bdo_parse_issues.csv

Sources (downloaded by fetch_bdo.py into data/raw/ballondor/):
- L'Équipe / France Football juror lists, men 2022 and 2025, women 2023-2025 (Wayback copies). Only
  headings and paragraphs inside the article body are read; teaser widgets for other articles, image
  captions and boxed text are skipped. Two layouts occur. Men 2022 prints "Pays - Juré - Média" on one
  line, followed by the ranked picks. The other pages print the country as a heading, the juror and
  outlet ("Juré, Média") either in the same heading (men 2025, country in capitals) or in the next
  paragraph (women). A heading or line is a ballot header only if it starts with a French country name
  of countries_fr.py; other headings inside the list (links to other articles) are ignored.
- Rank lines read "N. Name". Two cases are repaired, both from the printed rank numbers alone: two
  ranks on one line ("4. Mbappé5. Hakimi", flag several_ranks_one_line) and a rank whose name sits on
  the next line ("8." / "Katoto", flag split_rank_line). A ballot whose rank numbers are not exactly
  1..k is written with its printed picks in their printed slots, flagged, and listed in the issues file.
- Men 2023 has no free juror list; Wikipedia's "Ballon D'Or Detailed Votes by Country" table gives the
  points each country's juror gave each nominee. Ranks follow from the points (6-4-3-2-1); a row whose
  points are not distinct values of the scheme goes to the issues file. The same tables for 2024 and
  2025 are kept in bdo_wiki_ballots.csv for cross-checks (validate_bdo.py); they carry no juror names.
- Official totals: each year's Wikipedia ranking table (all nominees, points), and L'Équipe's own
  ranking where the page prints one (men 2022, women 2023 and 2024).

Countries are mapped with countries_fr.py (French) or countries.resolver (English, Wikipedia). Picks
keep their printed spelling here; bdo_nominees.py maps them to persons. Nothing is guessed: a line that
fits no pattern goes to bdo_parse_issues.csv with its raw text.
"""
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from countries import resolver  # noqa: E402
from countries_fr import resolve_fr, split_country  # noqa: E402
from nominee_meta import player_link  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ballondor"
WIKI_2025 = ROOT / "data" / "raw" / "wikipedia" / "ballondor_2025.json"
PROC = ROOT / "data" / "processed"
KMAX = 10
SCHEME = {5: (6, 4, 3, 2, 1), 10: (15, 12, 10, 8, 7, 5, 4, 3, 2, 1)}
K_OF_YEAR = {2019: 5, 2022: 5, 2023: 5, 2024: 10, 2025: 10}
WIKI_YEARS = (2019, 2022, 2023, 2024, 2025)
SECTION = {"MP": "Ballon d'Or", "WP": "Ballon d'Or Féminin"}
LEQUIPE = {   # (year, award): (header layout, juror/outlet separator)
    (2022, "MP"): ("inline", "hyphen"),
    (2025, "MP"): ("heading", "comma"),
    (2023, "WP"): ("heading", "comma"),
    (2024, "WP"): ("heading", "comma"),
    (2025, "WP"): ("heading", "comma"),
}
COLUMNS = (["year", "award", "country_raw", "country", "juror", "outlet", "k"]
           + [f"pick{i}" for i in range(1, KMAX + 1)] + ["source", "flag"])
RANK_START = re.compile(r"^\d{1,2}\.(?!\d)")
RANK_SPLIT = re.compile(r"(?<!\d)(\d{1,2})\.(?!\d)")
OFFICIAL_LINE = re.compile(r"(?<!\d)(\d{1,2})\.\s*([^()]+?)\s*\(([^)]*)\)\s*,\s*(\d+)\s*(?:pts?|points)\b")


def norm_space(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def add_flag(flag, f):
    return f"{flag};{f}" if flag else f


def canon_resolver():
    return resolver(pd.read_csv(PROC / "countries.csv", keep_default_na=False).country)


# ---------------------------------------------------------------- L'Équipe pages

def article(path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    body = soup.find("div", class_="article__body")
    if body is None:
        raise ValueError(f"{path}: no article body")
    return body


def article_lines(body):
    """(line number, 'head'|'text', text) for headings and paragraphs of the article text."""
    out = []
    for e in body.find_all(["h2", "h3", "p"]):
        if e.find_parent("div", class_="Paragraph") is None:
            continue
        if e.find_parent(class_=re.compile(r"ColeaderWidget|TextFocus|Exergue|Image")) is not None:
            continue
        if e.name == "p" and "Paragraph__content" not in (e.get("class") or []):
            continue
        for br in e.find_all("br"):
            br.replace_with("\n")
        kind = "head" if e.name in ("h2", "h3") else "text"
        for line in e.get_text("").split("\n"):
            line = norm_space(line)
            if line:
                out.append((len(out) + 1, kind, line))
    return out


def header(kind, text, layout):
    """(country_raw, rest) if the line opens a ballot, else None."""
    if layout == "heading" and kind != "head":
        return None
    if layout == "inline" and kind != "text":
        return None
    country, rest = split_country(text)
    if country is None:
        return None
    if layout == "inline" and not text[len(country):].lstrip().startswith("-"):
        return None
    return country, rest


def juror_outlet(rest, sep):
    rest = norm_space(rest).lstrip(" -,.:").rstrip(" ,-")
    if sep == "hyphen":
        parts = [p for p in re.split(r"\s+-\s*|\s*-\s+", rest) if p]
    else:
        parts = [p.strip() for p in rest.split(",", 1) if p.strip()]
    flag = ""
    if len(parts) == 1:
        flag = "no_outlet_separator"
    elif len(parts) > 2:
        flag = "outlet_contains_separator"
    juror = parts[0] if parts else ""
    outlet = " - ".join(parts[1:]) if len(parts) > 1 else ""
    return juror, outlet, flag


def rank_items(text):
    parts = RANK_SPLIT.split(text)
    if parts[0].strip():
        return None
    return [(int(parts[i]), norm_space(parts[i + 1])) for i in range(1, len(parts), 2)]


def parse_lequipe(year, award):
    folder = RAW / f"{year}_{award}"
    meta = json.loads((folder / "source.json").read_text())
    source = f"lequipe:wayback/{meta['wayback_timestamp']}"
    layout, sep = LEQUIPE[(year, award)]
    k = K_OF_YEAR[year]
    body = article(folder / "page.html")
    lines = article_lines(body)
    issues, blocks, cur = [], [], None

    def issue(ln, reason, text, severity="error"):
        issues.append({"year": year, "award": award, "source": source, "line": ln, "severity": severity,
                       "reason": reason, "text": text})

    for ln, kind, text in lines:
        h = header(kind, text, layout)
        if h:
            cur = {"country_raw": h[0], "juror": "", "outlet": "", "ranks": [], "flag": "", "line": ln,
                   "raw": [text], "need_juror": True}
            blocks.append(cur)
            if h[1].strip(" -,.:"):
                cur["juror"], cur["outlet"], f = juror_outlet(h[1], sep)
                cur["flag"] = add_flag(cur["flag"], f) if f else cur["flag"]
                cur["need_juror"] = False
            continue
        if cur is None:
            continue   # introduction and rules before the first ballot
        if kind == "head":
            issue(ln, "heading inside the list ignored (not a country)", text, "info")
            continue
        cur["raw"].append(text)
        if RANK_START.match(text):
            items = rank_items(text)
            if len(items) > 1:
                cur["flag"] = add_flag(cur["flag"], "several_ranks_one_line")
            if any(not name for _, name in items[:-1]):
                issue(ln, "empty rank inside a line", text)
            cur["ranks"] += [[n, name, ln] for n, name in items]
            continue
        if cur["ranks"] and not cur["ranks"][-1][1]:
            cur["ranks"][-1][1] = text
            cur["flag"] = add_flag(cur["flag"], "split_rank_line")
            continue
        if cur["need_juror"] and not cur["ranks"]:
            cur["juror"], cur["outlet"], f = juror_outlet(text, sep)
            cur["flag"] = add_flag(cur["flag"], f) if f else cur["flag"]
            cur["need_juror"] = False
            continue
        issue(ln, "text inside the list that is neither header, juror nor rank", text)

    rows = []
    for b in blocks:
        row = {"year": year, "award": award, "country_raw": b["country_raw"],
               "country": resolve_fr(b["country_raw"]) or "", "juror": b["juror"], "outlet": b["outlet"], "k": k,
               "source": source, "flag": b["flag"]}
        for i in range(1, KMAX + 1):
            row[f"pick{i}"] = ""
        nums = [n for n, _, _ in b["ranks"]]
        if sorted(nums) != list(range(1, k + 1)) or any(not name for _, name, _ in b["ranks"]):
            row["flag"] = add_flag(row["flag"], "incomplete_ranks")
            issue(b["line"], f"rank numbers {nums} are not 1..{k} (printed picks kept in their slots)",
                  " | ".join(b["raw"]))
        elif nums != sorted(nums):
            row["flag"] = add_flag(row["flag"], "ranks_out_of_order")
        seen = set()
        for n, name, _ in b["ranks"]:
            if 1 <= n <= k and n not in seen and name:
                row[f"pick{n}"] = name
            seen.add(n)
        if not row["country"]:
            issue(b["line"], "unresolved country", b["country_raw"])
        if not b["juror"]:
            issue(b["line"], "no juror line", " | ".join(b["raw"]))
        if "outlet" in row["flag"]:
            issue(b["line"], f"juror/outlet split: {row['flag']}", b["raw"][0] if not b["need_juror"] else "",
                  "check")
        rows.append(row)
    return rows, issues, lequipe_ranking(body, year, award, source)


def lequipe_ranking(body, year, award, source):
    """L'Équipe's own final ranking (boxed text), where printed: full name, (nation, clubs), points."""
    out = []
    for box in body.find_all("div", class_="TextFocus"):
        for br in box.find_all("br"):
            br.replace_with(" ")
        text = norm_space(box.get_text(" "))
        if not re.search(r"classement", text, re.I):
            continue
        for m in OFFICIAL_LINE.finditer(text):
            out.append({"year": year, "award": award, "source": source, "rank": int(m.group(1)),
                        "name": norm_space(m.group(2)), "points": int(m.group(4)), "detail": m.group(3)})
    return out


# ---------------------------------------------------------------- Wikipedia pages

def wiki_path(year):
    return WIKI_2025 if year == 2025 else RAW / f"{year}_wiki" / "wikipedia.json"


def wiki_source(year):
    d = json.loads(wiki_path(year).read_text())["parse"]
    rev = f"rev {d['revid']}" if d.get("revid") else "saved 2026-10-17, revision id not recorded"
    return f"wikipedia:{year} Ballon d'Or ({rev})"


def wiki_soup(year):
    return BeautifulSoup(json.loads(wiki_path(year).read_text())["parse"]["text"], "html.parser")


def header_cells(t):
    return [c.get_text(" ", strip=True) for c in t.find("tr").find_all(["th", "td"])]


def wiki_tables(soup, award):
    """(ranking table, per-country table or None) under the award's section heading."""
    rank, country = None, None
    for t in soup.find_all("table", class_="wikitable"):
        h = t.find_previous(["h2", "h3"])
        if h is None or h.get_text(" ", strip=True) != SECTION[award]:
            continue
        hdr = header_cells(t)
        if hdr[0] == "Country" and country is None:
            country = t
        elif "Player" in hdr and "Points" in hdr and rank is None:
            rank = t
    return rank, country


def grid(t):
    """Table cells row by row, a cell spanning several rows (tied ranks, shared points) repeated in each."""
    rows, carry = [], {}
    for tr in t.find_all("tr"):
        cs, out, col, i = tr.find_all(["th", "td"]), [], 0, 0
        while i < len(cs) or col in carry:
            if col in carry:
                cell, n = carry.pop(col)
                if n > 1:
                    carry[col] = (cell, n - 1)
                out.append(cell)
                col += 1
                continue
            c = cs[i]
            i += 1
            span = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            for _ in range(int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)):
                out.append(c)
                if span > 1:
                    carry[col] = (c, span - 1)
                col += 1
        rows.append((tr, out))
    return rows


def ranking_rows(t):
    """Rows of a Wikipedia ranking table as (rank, {header: cell}, tr)."""
    g = grid(t)
    hdr = [c.get_text(" ", strip=True) for c in g[0][1]]
    out = []
    for tr, cs in g[1:]:
        if len(cs) != len(hdr):
            raise ValueError(f"ranking row with {len(cs)} cells for {len(hdr)} columns: {tr.get_text(' ', strip=True)}")
        out.append((int(re.search(r"\d+", cs[0].get_text()).group()), dict(zip(hdr, cs)), tr))
    return out


def wiki_official(year, award, soup, source):
    t, _ = wiki_tables(soup, award)
    out = []
    for rank, cells, _ in ranking_rows(t):
        a = player_link(cells["Player"])
        pts = re.search(r"\d+", cells["Points"].get_text())
        out.append({"year": year, "award": award, "source": source, "rank": rank,
                    "name": a.get_text(strip=True) if a else cells["Player"].get_text(" ", strip=True),
                    "points": int(pts.group()) if pts else None, "detail": ""})
    return out


def wiki_country_ballots(year, soup, source, resolve):
    _, t = wiki_tables(soup, "MP")
    k = K_OF_YEAR[year]
    rank_of = {p: i + 1 for i, p in enumerate(SCHEME[k])}
    names = header_cells(t)[1:]
    rows, issues = [], []
    for ln, tr in enumerate(t.find_all("tr")[1:], 2):
        cs = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        raw = " | ".join(cs)
        if len(cs) != len(names) + 1 or not all(re.fullmatch(r"\d+", v) for v in cs[1:]):
            issues.append({"year": year, "award": "MP", "source": source, "line": ln, "severity": "error",
                           "reason": "per-country row not parsed", "text": raw})
            continue
        given = [(int(v), names[j]) for j, v in enumerate(cs[1:]) if int(v)]
        pts = [v for v, _ in given]
        row = {"year": year, "award": "MP", "country_raw": cs[0], "country": resolve(cs[0]) or "", "juror": "",
               "outlet": "", "k": k, "source": source, "flag": ""}
        for i in range(1, KMAX + 1):
            row[f"pick{i}"] = ""
        if sorted(pts, reverse=True) != list(SCHEME[k]):
            ok = all(p in rank_of for p in pts) and len(set(pts)) == len(pts)
            row["flag"] = "incomplete_ranks" if ok else "invalid_points"
            issues.append({"year": year, "award": "MP", "source": source, "line": ln, "severity": "error",
                           "reason": f"points {sorted(pts, reverse=True)} are not one of each of {list(SCHEME[k])}"
                                     + (" (distinct valid values kept in their slots)" if ok else " (row left empty)"),
                           "text": raw})
            if not ok:
                rows.append(row)
                continue
        for p, name in given:
            row[f"pick{rank_of[p]}"] = name
        if not row["country"]:
            issues.append({"year": year, "award": "MP", "source": source, "line": ln, "severity": "error",
                           "reason": "unresolved country", "text": cs[0]})
        rows.append(row)
    return rows, issues


def write(name, rows, columns=None):
    with (PROC / name).open("w", newline="") as fh:
        cols = columns or (list(rows[0]) if rows else [])
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    resolve = canon_resolver()
    ballots, wiki_ballots, official, issues = [], [], [], []
    for (year, award) in sorted(LEQUIPE):
        r, i, o = parse_lequipe(year, award)
        ballots += r
        issues += i
        official += o
    for year in WIKI_YEARS:
        soup, source = wiki_soup(year), wiki_source(year)
        for award in ("MP", "WP"):
            if award == "MP" or year >= 2023:   # the contests with per-juror ballots
                official += wiki_official(year, award, soup, source)
        if year >= 2023:
            r, i = wiki_country_ballots(year, soup, source, resolve)
            wiki_ballots += r
            issues += i
            if year == 2023:
                ballots += r
    write("bdo_ballots_raw.csv", ballots, COLUMNS)
    write("bdo_wiki_ballots.csv", wiki_ballots, COLUMNS)
    write("bdo_official_points.csv", official,
          ["year", "award", "source", "rank", "name", "points", "detail"])
    write("bdo_parse_issues.csv", issues, ["year", "award", "source", "line", "severity", "reason", "text"])
    df = pd.DataFrame(ballots)
    print(df.groupby(["year", "award"]).size().rename("jurors").to_string(), file=sys.stderr)
    print(f"wikipedia per-country rows: {pd.DataFrame(wiki_ballots).groupby('year').size().to_dict()}",
          file=sys.stderr)
    print(f"official rows: {pd.DataFrame(official).groupby(['year', 'award', 'source']).size().to_dict()}",
          file=sys.stderr)
    sev = pd.Series([i["severity"] for i in issues]).value_counts().to_dict() if issues else {}
    print(f"issues: {sev}", file=sys.stderr)


if __name__ == "__main__":
    main()
