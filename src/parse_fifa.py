"""Parse the per-voter FIFA award PDFs (text from `pdftotext -layout`) into one row per ballot.

python -I nhb/src/parse_fifa.py  ->  nhb/data/processed/fifa_ballots_raw.csv, fifa_parse_issues.csv,
                                     fifa_official_top3.csv

Layouts: 2010 has one section per role ("Votes by captain ...") with columns Country, Name, First,
Second, Third; from 2011 a leading role column (Captain / Coach / Media). Columns are usually
separated by two or more spaces, but some rows run neighbouring cells together. Each contest has a
closed shortlist, so a row is split by matching its last three cells against the shortlist (learned
from the rows whose cells are cleanly separated) and the rest against the known country names.
Partial ballots are assigned to slots by the cell positions under the page header; picks wrapped
onto the following line are joined. Anything else goes to the issues file, never guessed.
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data" / "raw" / "fifa" / "txt"
OUT = ROOT / "data" / "processed"
ROLES = {"captain": "captain", "coach": "coach", "media": "media", "journalist": "media"}
ROLE = re.compile(r"^\s*(Captain|Coach|Media)\b\s*(.*)$")  # capitalised: rules prose starts lowercase
SECTION = re.compile(r"votes?\s+by\s+(captain|coach|media|journalist)", re.I)
HEADER = re.compile(r"first\s*\(\s*5", re.I)
SLOTS = ("first", "second", "third")
PLACEHOLDER = {"invalid vote": "INVALID", "*not counted": "NOT_COUNTED", "#n/a": "NA", "no vote": "NO_VOTE"}
MIN_COUNT = 2  # a pick must appear this often in clean rows to enter the shortlist


def cells(line):
    """Runs of text separated by two or more spaces, with their start columns."""
    return [(m.start(), m.group()) for m in re.finditer(r"\S+(?: \S+)*", line)]


def header_cols(line):
    low = line.lower()
    cols = [low.find(s) for s in SLOTS]
    return cols if min(cols) >= 0 else None


def read_lines(path):
    return [ln.replace("\f", "").rstrip() for ln in path.read_text(errors="replace").split("\n")]


def candidates(lines, year):
    """Role rows (with any wrapped continuation lines) and orphan lines."""
    rows, orphans = [], []
    section, hdr = None, None
    for ln, line in enumerate(lines, 1):
        if not line.strip():
            continue
        if HEADER.search(line):
            hdr = header_cols(line) or hdr
            continue
        m = SECTION.search(line)
        if m and len(cells(line)) <= 2:
            section = ROLES[m.group(1).lower()]
            continue
        m = ROLE.match(line)
        if m:
            off = line.index(m.group(2)) if m.group(2) else len(line)
            rows.append({"line": ln, "role": ROLES[m.group(1).lower()], "rest": m.group(2), "off": off,
                         "raw": line, "hdr": hdr, "cont": []})
        elif year == 2010 and section and len(cells(line)) >= 3:
            rows.append({"line": ln, "role": section, "rest": line, "off": 0, "raw": line, "hdr": hdr, "cont": []})
        else:
            orphans.append((ln, line, len(rows) - 1))
    return rows, orphans


def clean_split(rest):
    c = [t for _, t in cells(rest)]
    return c if len(c) == 5 else None


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def fold(s):
    return deaccent(s).lower()


def strip_nation(pick, countries):
    """2010 media ballots print picks as 'Name / Nation' (accents not always kept)."""
    m = re.match(r"^(.*?)\s*/\s*(.+)$", pick)
    folded = {fold(c) for c in countries}
    return m.group(1) if m and fold(norm_space(m.group(2))) in folded else pick


def clean_picks(rows, countries):
    for r in rows:
        if (s := clean_split(r["rest"])):
            yield from (norm_space(strip_nation(p, countries)) for p in s[2:] if p.lower() not in PLACEHOLDER)


def name_pattern(names, countries=()):
    alt = lambda xs: "|".join(r"\s+".join(map(re.escape, x.split())) for x in sorted(xs, key=len, reverse=True))
    nation = rf"(?:\s*/\s*(?:{alt(set(countries) | {deaccent(c) for c in countries})}))?" if countries else ""
    return f"(?:(?:{alt(names)}){nation}|{alt(PLACEHOLDER)})"


def norm_space(s):
    return re.sub(r"\s+", " ", s).strip()


def split_head(head, countries):
    c = [t for _, t in cells(head)]
    if len(c) == 2:
        return c[0], c[1]
    h = norm_space(head)
    for k in sorted(countries, key=len, reverse=True):
        if h == k or h.startswith(k + " "):
            return k, h[len(k):].strip()
    return None, None


def matchers(names, countries):
    nom = re.compile(name_pattern(names, countries), re.I)
    p = nom.pattern
    full = re.compile(rf"^(?P<head>.+?)\s*(?P<p1>{p})\s*(?P<p2>{p})\s*(?P<p3>{p})\s*$", re.I)
    only_picks = re.compile(rf"^\s*(?:{p}\s*){{1,3}}$", re.I)
    return nom, full, only_picks


def add_flag(flag, f):
    return f"{flag};{f}" if flag else f


def parse_contest(path, year, award, countries, universe):
    """`universe` holds every cleanly separated pick from all contests, the fallback for rare picks."""
    lines = read_lines(path)
    rows, orphans = candidates(lines, year)
    tally = Counter(clean_picks(rows, countries))
    shortlist = {n for n, k in tally.items() if k >= MIN_COUNT}
    nom, full, only_picks = matchers(shortlist, countries)
    nom_u, full_u, only_u = matchers(universe, countries)
    canon = {n.lower(): n for n in universe}
    canon.update({k: v for k, v in PLACEHOLDER.items()})

    def tidy(p):
        p = norm_space(strip_nation(norm_space(p), countries))
        return canon.get(p.lower(), p)

    issues = []

    def issue(ln, reason, text):
        issues.append({"year": year, "award": award, "line": ln, "reason": reason, "text": text.strip()})

    for ln, line, prev in orphans:   # wrapped picks, country or voter text belong to the row just above
        one = cells(line)
        if prev >= 0 and only_u.match(line) and not full_u.match(rows[prev]["rest"]):
            rows[prev]["cont"].append((ln, line))
        elif prev >= 0 and len(one) == 1 and ln == rows[prev]["line"] + 1 and not nom_u.search(line):
            starts = [pos + rows[prev]["off"] for pos, _ in cells(rows[prev]["rest"])[:2]]
            hit = [i for i, c in enumerate(starts) if abs(c - one[0][0]) <= 2]
            if hit:
                rows[prev]["wrap"] = (hit[0], one[0][1])
        elif len(cells(line)) >= 3 and not re.match(r"^\s*\d", line) and nom.search(line):
            issue(ln, "orphan line", line)

    out = []
    for r in rows:
        rest, flag = r["rest"], ""
        if r["cont"]:
            rest = rest + "  " + "  ".join(t.strip() for _, t in r["cont"])
            flag = "wrapped"
        m = full.match(rest)
        if not m and (m := full_u.match(rest)):
            flag = add_flag(flag, "rare_pick")
        if m:
            head, picks = m.group("head"), [m.group(f"p{i}") for i in (1, 2, 3)]
            cs = clean_split(r["rest"])
            if cs and [tidy(p) for p in cs[2:]] != [tidy(p) for p in picks]:
                flag = add_flag(flag, "split_disagree")
        else:   # partial ballot: place picks by the page header's column positions
            cs = cells(r["rest"])
            k = len(cs)
            while k > 0 and nom_u.fullmatch(cs[k - 1][1]):
                k -= 1
            if k == 0 or k == len(cs) or not r["hdr"]:
                issue(r["line"], "unparsed", r["raw"])
                continue
            picks = ["", "", ""]
            for pos, t in cs[k:]:
                slot = min(range(3), key=lambda j: abs(r["hdr"][j] - (pos + r["off"])))
                if picks[slot]:
                    picks = None
                    break
                picks[slot] = t
            if picks is None:
                issue(r["line"], "slot clash", r["raw"])
                continue
            first = next(j for j in range(3) if picks[j])
            pos, t = cs[k - 1]
            if any(pos + len(t) + r["off"] >= r["hdr"][j] for j in range(first)):
                issue(r["line"], "voter cell reaches an empty pick column", r["raw"])
                continue
            head = r["rest"][:cs[k][0]]
            flag = add_flag(flag, "partial")
        country, voter = split_head(head, countries)
        if country is None:
            issue(r["line"], "no country", r["raw"])
            continue
        if "wrap" in r:   # a country or voter name continued on the next line
            i, t = r["wrap"]
            country, voter = (f"{country} {t}", voter) if i == 0 else (country, f"{voter} {t}")
            flag = add_flag(flag, "wrapped_name")
        picks = [tidy(p) if p else "" for p in picks]
        if re.search(r"(?i)\btest_", voter):
            flag = add_flag(flag, "test_entry")
        out.append({"year": year, "award": award, "role": r["role"], "country": norm_space(country),
                    "voter": norm_space(voter), "pick1": picks[0], "pick2": picks[1], "pick3": picks[2],
                    "flag": flag, "line": r["line"]})
    return out, issues, sorted(shortlist)


def known_countries(paths):
    found = Counter()
    for path in paths:
        year = int(path.stem.split("_")[0])
        rows, _ = candidates(read_lines(path), year)
        for r in rows:
            if (s := clean_split(r["rest"])):
                found[norm_space(s[0])] += 1
    return {c for c in found if not re.search(r"\d", c)}


OFFICIAL = re.compile(r"^\s*(\d+)\s+(.*?)\s{2,}(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)\s+(\d+)\s*$")
SCORING = re.compile(r"^\s*(\d+)\s+(\S.*?)\s{2,}\S.*?\s{2,}(\d+)\*?\s*$")


def official_top3(path, year, award):
    """Per-group raw points for the official top three (printed only for 2023 and 2024)."""
    lines = read_lines(path)
    pts, names = {}, {}
    for line in lines[:80]:
        if (m := OFFICIAL.match(line)):
            pts[int(m.group(1))] = tuple(int(m.group(i)) for i in (3, 4, 5))
        elif (m := SCORING.match(line)) and int(m.group(1)) <= 3:
            names.setdefault(int(m.group(1)), norm_space(m.group(2)))
    return [{"year": year, "award": award, "rank": k, "name": names.get(k, ""), "coach": v[0], "captain": v[1],
             "media": v[2]} for k, v in sorted(pts.items())]


def write(name, data):
    with (OUT / name).open("w", newline="") as fh:
        if data:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(TXT.glob("*_*.txt"))
    countries = known_countries(paths)
    universe = {p for path in paths
                for p in clean_picks(candidates(read_lines(path), int(path.stem[:4]))[0], countries)}
    rows, issues, shortlists, official = [], [], [], []
    for path in paths:
        year, award = path.stem.split("_")
        r, i, s = parse_contest(path, int(year), award, countries, universe)
        rows += r
        issues += i
        shortlists += [{"year": int(year), "award": award, "name": n} for n in s]
        official += official_top3(path, int(year), award)
    write("fifa_ballots_raw.csv", rows)
    write("fifa_parse_issues.csv", issues)
    write("fifa_shortlist_seen.csv", shortlists)
    write("fifa_official_top3.csv", official)
    print(f"{len(rows)} ballots, {len(issues)} issues, {len(countries)} countries", file=sys.stderr)


if __name__ == "__main__":
    main()
