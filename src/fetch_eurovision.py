"""Download the Wikipedia pages that carry the Eurovision juries' individual rankings.

python -I nhb/src/fetch_eurovision.py [years ...]  ->  nhb/data/raw/eurovision/wikipedia/*.json

For each contest year: the category "Countries in the Eurovision Song Contest <year>" (the pages of
all participating countries; every participant votes in the final), the wikitext of each page, whose
"Detailed voting results" tables give every jury member's ranking of every song, and the wikitext of
the main contest page (the scoreboard used for validation). Requests are spaced by SPACING seconds,
with a generic User-Agent; a file already on disk is not fetched again. 2020 had no contest.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "eurovision" / "wikipedia"
API = "https://en.wikipedia.org/w/api.php"
UA = "soccer-research/0.1 (academic research)"
SPACING = 8.0
YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def get(params, tries=5):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    for k in range(tries):
        time.sleep(SPACING * (1 if k == 0 else 4 * k))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:          # 429 or a network error: wait longer and retry
            print(f"retry {k + 1}: {e}", file=sys.stderr, flush=True)
    raise RuntimeError(f"failed: {params}")


def save_page(title):
    f = OUT / (title.replace("/", "_").replace(" ", "_") + ".json")
    if f.exists():
        return
    d = get({"action": "parse", "page": title, "prop": "wikitext|revid", "redirects": 1})
    if "parse" not in d:
        print(f"missing page: {title}", file=sys.stderr)
        return
    f.write_text(json.dumps({"title": title, "revid": d["parse"].get("revid"),
                             "wikitext": d["parse"]["wikitext"]}, ensure_ascii=False))


def main():
    years = [int(y) for y in sys.argv[1:]] or YEARS
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        cat = OUT / f"_category_{y}.json"
        if not cat.exists():
            d = get({"action": "query", "list": "categorymembers", "cmlimit": 200,
                     "cmtitle": f"Category:Countries in the Eurovision Song Contest {y}"})
            cat.write_text(json.dumps(d, ensure_ascii=False))
        titles = [m["title"] for m in json.loads(cat.read_text())["query"]["categorymembers"]
                  if m["title"].endswith(f"in the Eurovision Song Contest {y}")]
        save_page(f"Eurovision Song Contest {y}")
        for t in titles:
            save_page(t)
        print(f"{y}: {len(titles)} country pages", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
