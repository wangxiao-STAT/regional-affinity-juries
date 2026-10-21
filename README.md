# Regional affinity, not compatriot favoritism, tips close votes of expert juries

Code and de-identified data for the manuscript of the same title (Guanyu Hu and Xiao Wang), which
analyses the published ballots of FIFA's player awards (2010-2025, men's and women's) and of France
Football's Ballon d'Or (men 2019 and 2022-2025, women 2023-2025), and the published juror rankings of
the Eurovision Song Contest's national juries in the finals of 2016-2019 and 2021-2026.

## Contents

| Folder | What it holds |
|---|---|
| `src/` | All code: fetching and parsing the ballot files, nominee and coach metadata, the ranking models, counterfactuals, jury-design and cross-jury simulations, mechanism and robustness analyses, and the scripts that write every number, table and figure of the manuscript. Each script starts with a docstring that says what it reads and writes. |
| `tests/` | Unit tests (`python -I -m pytest -q tests/`). |
| `jobs/` | The Slurm scripts that produced the reported results, in order (set `REPO` to this folder). |
| `data/processed/` | Parsed and validated tables (one row per ballot, per ballot and nominee, per nominee, validation tables). |
| `data/raw/fifa/manifest.csv` | Source URL of every FIFA result file. |
| `data/processed/esc_sources.csv` | Title and revision of every Wikipedia page the Eurovision rankings were parsed from. |
| `data/README.md` | Data sources, parsing rules, validation and known limits. |
| `results/` | Estimates, counterfactual and simulation results (CSV and JSON). |

## Reproducing the results

Requirements: Python 3 with NumPy, SciPy, pandas, matplotlib and pytest; `pdftotext` (poppler) to
parse the raw FIFA files.

1. From the processed data (included): run the jobs in `jobs/` in order (`01` fits all models and
   runs the main analyses; `02` placebo, strategic-ranking and specification analyses; `03`
   counterfactuals; `04` the rule checks for FIFA's 2023-2025 contests; `05` the football jury-design
   simulations, which replace those of `01`; `06` the Eurovision fits, counterfactuals, designs and
   aggregation-rule checks; `07` the checks with normal instead of Gumbel ranking errors). Each job
   runs the tests first. On a 16-core node the seven jobs took about an hour and three quarters in
   total. Then `src/make_numbers.py`, `src/make_tables.py` and
   `src/make_figures.py` write the numbers, tables and figures of the manuscript.
2. From the raw files: `src/manifest.py` and the fetch scripts download the ballot files and web
   pages; `parse_fifa.py`, `parse_bdo.py`, `build_fifa.py`, `build_bdo.py`, `build_bdo_long.py`,
   `nominees.py`, `nominee_meta.py`, `bdo_nominees.py`, `coach_history.py` and the `validate_*`
   scripts rebuild `data/processed/` (see `data/README.md` for the order and checks). For Eurovision,
   `fetch_eurovision.py` downloads the Wikipedia pages and `06` then reruns `parse_eurovision.py`,
   `validate_eurovision.py` and `build_eurovision.py` before the analyses.

Results depend slightly on the number of BLAS threads, through the eigendecomposition used to draw
parameters (flip probabilities change by at most 0.006); the jobs set the thread counts used for the
reported results.

## De-identification

Juror names are published by the award organizers, but the deposited tables identify jurors only by
association, role and a pseudonymous identifier (`voter_id`, `juror_id`, `coach_id`) that is stable
across files. Raw text lines of the source files, coaches' Wikipedia page titles and the two
Wikipedia manager tables (which list coaches by name) are left out; `src/fetch_managers.py` and
`src/coach_history.py` rebuild the latter. Eurovision jurors appear only by country, year and position
on the jury; the table of their names (`esc_jurors.csv`) is left out. `src/make_release.py` builds this deposit from the full
working data. FIFA's PDFs and the downloaded web pages (including the Wikipedia pages) are not
redistributed.

## License

Code: MIT license (`LICENSE`). Processed data and results: CC BY 4.0 (`data/LICENSE`). The underlying
ballots are published by FIFA and France Football; the Eurovision rankings by the European
Broadcasting Union, as reproduced on Wikipedia.
