"""Words in the NHB main text (sections/main.tex) as printed: generated numbers and words count, citations,
cross-references, labels and comments do not.

python -I nhb/src/word_count.py  ->  prints the count; NHB's limit is 5,000 words.

The macros of generated/numbers.tex are replaced by what they print (a value such as 0.38 or "five"
counts as a word), headings and the text inside formatting commands (\\emph{...}) are kept, and other
LaTeX commands are removed.
"""
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[1] / "manuscript"


def macros():
    out = {}
    for line in (MS / "generated" / "numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*(%.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def count(text, values):
    text = re.sub(r"(?<!\\)%.*", " ", text)
    text = re.sub(r"\\(cite[pt]?|ref|label|Pending)\{[^}]*\}", " ", text)
    for _ in range(3):                               # macros whose values contain other macros
        text = re.sub(r"\\([A-Za-z]+)(\{\})?", lambda m: f" {values[m.group(1)]} " if m.group(1) in values
                      else m.group(0), text)
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r" \1 ", text)          # headings count
    text = re.sub(r"\\(?:emph|textit|textbf|textsc|mbox|ensuremath)\{([^}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[A-Za-z]+\*?(\[[^]]*\])?(\{[^}]*\})?", " ", text)     # any other command
    text = re.sub(r"\$[^$]*\$", " x ", text)                                 # inline math as one word
    text = re.sub(r"[{}\\~]", " ", text)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,%-]*", text))


if __name__ == "__main__":
    print(count((MS / "sections" / "main.tex").read_text(), macros()))
