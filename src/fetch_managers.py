"""Fetch dated revisions of Wikipedia's "List of current national association football team managers".

python -I nhb/src/fetch_managers.py  ->  nhb/data/raw/wikipedia/managers/rev_{date}.json

One revision per quarter, 2010-2025: the latest revision on or before 1 Feb/May/Aug/Nov. Each file
holds the revision id, its timestamp and the parsed HTML. Existing files are skipped, so the script
can be rerun after an interruption. Requests are spaced and back off when rate-limited.
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "wikipedia" / "managers"
TITLE = "List of current national association football team managers"
UA = {"User-Agent": "soccer-research/0.1 (academic research)"}
PAUSE = 8


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = [f"{y}-{m:02d}-01" for y in range(2010, 2026) for m in (2, 5, 8, 11)]
    for d in dates:
        path = OUT / f"rev_{d}.json"
        if path.exists():
            continue
        rev = get({"action": "query", "titles": TITLE, "prop": "revisions", "rvlimit": 1, "rvstart": f"{d}T00:00:00Z",
                   "rvdir": "older", "rvprop": "ids|timestamp", "format": "json", "formatversion": 2})
        rev = rev["query"]["pages"][0]["revisions"][0]
        html = get({"action": "parse", "oldid": rev["revid"], "prop": "text", "format": "json", "formatversion": 2})
        path.write_text(json.dumps({"date": d, "revid": rev["revid"], "timestamp": rev["timestamp"],
                                    "html": html["parse"]["text"]}))
        print(d, rev["revid"], rev["timestamp"], flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()
