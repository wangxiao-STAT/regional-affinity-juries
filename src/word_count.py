"""Words in the NHB main text (sections/main.tex), excluding comments, citations and LaTeX commands.

python -I nhb/src/word_count.py  ->  prints the count; NHB's limit is 5,000 words.
"""
import re
from pathlib import Path

text = (Path(__file__).resolve().parents[1] / "manuscript" / "sections" / "main.tex").read_text()
for pat in (r"(?<!\\)%.*", r"\\(cite[pt]?|ref|label|Pending)\{[^}]*\}", r"\\begin\{[^}]*\}|\\end\{[^}]*\}",
            r"\\[A-Za-z]+\*?(\{[^}]*\})?", r"\$[^$]*\$", r"[{}\\~]"):
    text = re.sub(pat, " " if "Pending" not in pat else " ", text)
print(len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,-]*", text)))
