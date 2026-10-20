"""Check the NHB manuscript: no hand-typed numbers in the prose, no pending macros in a final build.

python -I nhb/src/check_manuscript.py [--allow-pending]

Prose is every .tex file under manuscript/sections after removing comments, mathematics, \\ref,
\\label and citation arguments. A digit sequence there must be a year (1900-2099) or part of a year
range, or the conventional "95% confidence" level of the intervals (the code uses 1.96 standard
errors); figure widths are layout. Everything else must come from generated/numbers.tex. Statements in statements.tex and the
generated numbers are checked for \\Pending.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "manuscript"
STRIP = [r"95\\%(?= confidence)", r"\\Fig\{[^}]*\}\{[^}]*\}", r"(?<!\\)%.*", r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", r"\$[^$]*\$",
         r"\\(ref|label|cite[pt]?|citep|citet|eqref|input|includegraphics|SupRef)\{[^}]*\}", r"\\[A-Za-z]+"]


def main():
    problems = []
    for f in sorted((ROOT / "sections").glob("*.tex")):
        text = f.read_text()
        for pat in STRIP:      # only the equation pattern may span lines; "%.*" must stop at the line end
            text = re.sub(pat, " ", text, flags=re.S if "equation" in pat else 0)
        for m in re.finditer(r"\d+(?:[.,]\d+)*", text):
            tok = m.group()
            if re.fullmatch(r"(19|20)\d\d", tok):
                continue
            problems.append(f"{f.name}: hand-typed number '{tok}' near '{text[max(0, m.start() - 30):m.end() + 10].strip()}'")
    if "--allow-pending" not in sys.argv:
        for f in (ROOT / "generated" / "numbers.tex", ROOT / "statements.tex"):
            for line in f.read_text().splitlines():
                if "\\Pending" in line and not line.lstrip().startswith("%") and "newcommand{\\Pending}" not in line:
                    problems.append(f"{f.name}: {line.strip()[:100]}")
    for p in problems:
        print(p)
    print(f"check_manuscript: {len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
