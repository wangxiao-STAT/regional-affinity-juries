"""Fetch the Wikipedia page of every men's and women's national team whose coach voted.

python -I nhb/src/fetch_team_pages.py  ->  nhb/data/raw/wikipedia/teams/{men,women}__{title}.json

Men's page titles are the links of the confederation sections of "List of men's national
association football teams" (national_teams.json). Women's titles are derived from them ("... women's
national football team" or "... women's national soccer team") and resolved through redirects.
Most pages carry a coaching history (manager, nationality flag, years), which coach_history.py
reads. Existing files are skipped; requests are spaced and back off when rate-limited.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wikipedia"
OUT = RAW / "teams"
UA = {"User-Agent": "soccer-research/0.1 (academic research)"}
PAUSE = 8
CONFEDS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")


def get(params, tries=6):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            time.sleep(PAUSE)
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(60 * (k + 1))


def men_titles():
    soup = BeautifulSoup(json.loads((RAW / "national_teams.json").read_text())["parse"]["text"], "html.parser")
    out = {}
    for t in soup.find_all("table"):
        h = t.find_previous(["h2", "h3"])
        if not h or h.get_text(" ", strip=True).split(" ")[0] not in CONFEDS or t.get("class"):
            continue
        for a in t.find_all("a"):
            href = a.get("href", "")
            if href.startswith("/wiki/") and a.find_parent("sup") is None:
                out[a.get_text(strip=True)] = urllib.parse.unquote(href[6:]).replace("_", " ")
    return out


def women_title(men):
    t = re.sub(r"^(.*?) (men's )?national (football|soccer) team$", r"\1 women's national \3 team", men)
    return t if t != men else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for country, title in men_titles().items():
        jobs.append(("men", country, title))
        w = women_title(title)
        if w:
            jobs.append(("women", country, w))
    missing = []
    for side, country, title in jobs:
        path = OUT / f"{side}__{re.sub(r'[^A-Za-z0-9]+', '_', country)}.json"
        if path.exists():
            continue
        r = get({"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2,
                 "redirects": 1})
        if "error" in r:
            missing.append((side, country, title, r["error"].get("code")))
            path.write_text(json.dumps({"country": country, "side": side, "title": title, "missing": True}))
            continue
        path.write_text(json.dumps({"country": country, "side": side, "title": r["parse"]["title"],
                                    "retrieved": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
                                    "html": r["parse"]["text"]}))
        print(side, country, r["parse"]["title"], flush=True, file=sys.stderr)
    print(f"missing pages: {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
