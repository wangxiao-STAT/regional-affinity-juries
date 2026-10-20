"""Build the download manifest (year, award, url) for the per-voter FIFA PDFs (men's and women's
player awards, 2010-2025) from the verified URLs in inventory_agents/the_best_fifa.md.
Wayback URLs are rewritten to the raw-file form (timestamp + 'id_')."""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / "inventory_agents" / "the_best_fifa.md").read_text()
    sec = text[text.index("### 3.2"):text.index("## 4.")]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"- \*\*(\d{4})\*\*:", line)
        if not m:
            continue
        year = int(m.group(1))
        for award in ("MP", "WP"):
            u = re.search(rf"\b{award} `([^`]+)`", line)
            if u:
                url = re.sub(r"(web\.archive\.org/web/\d+)/", r"\1id_/", u.group(1))
                rows.append({"year": year, "award": award, "url": url})
    out = ROOT / "data" / "raw" / "fifa" / "manifest.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "award", "url"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["year"], r["award"])))
    print(f"{len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
