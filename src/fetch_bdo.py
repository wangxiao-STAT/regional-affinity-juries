"""Download the Ballon d'Or per-juror ballot pages and the Wikipedia award pages.

python -I nhb/src/fetch_bdo.py  ->  nhb/data/raw/ballondor/{YEAR}_{MP,WP}/page.html + source.json
                                    nhb/data/raw/ballondor/{YEAR}_wiki/wikipedia.json + source.json

L'Équipe / France Football pages (verified in inventory_agents/ballon_dor.md) are read from their
Wayback copies in the raw-capture form (timestamp + 'id_'), which returns the page as L'Équipe served
it, without the Wayback toolbar. The Wikipedia "{year} Ballon d'Or" pages are saved through the
MediaWiki parse API; one page covers both awards, so it gets its own `{year}_wiki` folder. The 2025
page was saved earlier (data/raw/wikipedia/ballondor_2025.json) and is not fetched again.

Each download goes to its own folder with a source.json recording the URL and the fetch time. Two raw
captures (women 2024 and 2025) come back still gzip-encoded; they are decoded before saving.
Existing files are skipped, so the script can be rerun after an interruption. Requests are spaced
PAUSE seconds apart and back off on HTTP 429; the User-Agent carries no personal contact.
"""
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ballondor"
UA = {"User-Agent": "soccer-research/0.1 (academic research)"}
PAUSE = 8
LEQUIPE = "https://www.lequipe.fr/France-Football/"
PAGES = {   # (year, award): (Wayback timestamp, L'Équipe path)
    (2022, "MP"): ("20250322040635", "Article/Le-detail-et-l-analyse-des-votes-de-tous-les-jures-pour-le-ballon-d-or-"
                                     "france-football-2022/1360534"),
    (2025, "MP"): ("20260510112921", "Article/Le-detail-des-votes-pour-le-ballon-d-or-masculin-2025-ousmane-dembele-"
                                     "present-dans-les-votes-de-tous-les-jures/1597021"),
    (2023, "WP"): ("20240227144843", "Actualites/Le-detail-et-l-analyse-des-votes-des-jures-du-ballon-d-or-feminin-"
                                     "france-football-2023/1429710"),
    (2024, "WP"): ("20260914175433", "Actualites/Le-detail-des-votes-pour-le-ballon-d-or-femmes-2024-remporte-par-"
                                     "aitana-bonmati/1519266"),
    (2025, "WP"): ("20260306032405", "Article/Le-detail-des-votes-du-ballon-d-or-2025-feminin-aitana-bonmati-plus-"
                                     "souvent-citee-que-mariona-caldentey/1596980"),
}
WIKI_YEARS = (2019, 2022, 2023, 2024)


def get(url, tries=6):
    for k in range(tries):
        try:
            time.sleep(PAUSE)
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or k == tries - 1:
                raise
            time.sleep(60 * (k + 1))


def save(folder, name, data, meta):
    folder.mkdir(parents=True, exist_ok=True)
    if data[:2] == b"\x1f\x8b":   # some raw Wayback captures keep the original gzip encoding
        data = gzip.decompress(data)
        meta["gzip_decoded"] = True
    (folder / name).write_bytes(data)
    meta["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["bytes"] = len(data)
    (folder / "source.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"{folder.name}/{name}: {len(data)} bytes", file=sys.stderr, flush=True)


def main():
    for (year, award), (ts, path) in sorted(PAGES.items()):
        folder = OUT / f"{year}_{award}"
        if (folder / "page.html").exists():
            continue
        original = LEQUIPE + path
        url = f"https://web.archive.org/web/{ts}id_/{original}"
        data, final = get(url)
        save(folder, "page.html", data, {"year": year, "award": award, "original_url": original,
                                         "wayback_timestamp": ts, "request_url": url, "final_url": final})
    for year in WIKI_YEARS:
        folder = OUT / f"{year}_wiki"
        if (folder / "wikipedia.json").exists():
            continue
        title = f"{year} Ballon d'Or"
        params = {"action": "parse", "page": title, "prop": "text|revid", "formatversion": 2, "format": "json",
                  "redirects": 1}
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        data, _ = get(url)
        revid = json.loads(data)["parse"].get("revid")
        save(folder, "wikipedia.json", data, {"year": year, "title": title, "request_url": url, "revid": revid})


if __name__ == "__main__":
    main()
