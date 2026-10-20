# Regional affinity, not compatriot favouritism, tips close votes of expert juries

Code and de-identified data for the manuscript of the same title (Guanyu Hu and Xiao Wang), which
analyses the published ballots of FIFA's player awards (2010-2025, men's and women's) and of France
Football's Ballon d'Or (men 2019 and 2022-2025, women 2023-2025).

## Contents

| Folder | What it holds |
|---|---|
| `src/` | All code: fetching and parsing the ballot files, nominee and coach metadata, the ranking models, counterfactuals, jury-design and cross-jury simulations, mechanism and robustness analyses, and the scripts that write every number, table and figure of the manuscript. Each script starts with a docstring that says what it reads and writes. |
| `tests/` | Unit tests (`python -I -m pytest -q tests/`). |
| `jobs/` | The Slurm scripts that produced the reported results, in order (set `REPO` to this folder). |
| `data/processed/` | Parsed and validated tables (one row per ballot, per ballot and nominee, per nominee, validation tables). |
| `data/raw/fifa/manifest.csv` | Source URL of every FIFA result file. |
| `data/README.md` | Data sources, parsing rules, validation and known limits. |
| `results/` | Estimates, counterfactual and simulation results (CSV and JSON). |

## Reproducing the results

Requirements: Python 3 with NumPy, SciPy, pandas, matplotlib and pytest; `pdftotext` (poppler) to
parse the raw FIFA files.

1. From the processed data (included): run the jobs in `jobs/` in order (`01` fits all models and
   runs the main analyses; `02` placebo, strategic-ranking and specification analyses; `03`
   counterfactuals; `04` the rule checks for FIFA's 2023-2025 contests). Each job runs the tests
   first. On a 16-core node the four jobs took about an hour in total.
2. From the raw files: `src/manifest.py` and the fetch scripts download the ballot files and web
   pages; `parse_fifa.py`, `parse_bdo.py`, `build_fifa.py`, `build_bdo.py`, `build_bdo_long.py`,
   `nominees.py`, `nominee_meta.py`, `bdo_nominees.py`, `coach_history.py` and the `validate_*`
   scripts rebuild `data/processed/` (see `data/README.md` for the order and checks).

Results depend slightly on the number of BLAS threads, through the eigendecomposition used to draw
parameters (flip probabilities change by at most 0.006); the jobs set the thread counts used for the
reported results.

## De-identification

Juror names are published by the award organisers, but the deposited tables identify jurors only by
association, role and a pseudonymous identifier (`voter_id`, `juror_id`, `coach_id`) that is stable
across files. Raw text lines of the source files, coaches' Wikipedia page titles and the two
Wikipedia manager tables (which list coaches by name) are left out; `src/fetch_managers.py` and
`src/coach_history.py` rebuild the latter. `src/make_release.py` builds this deposit from the full
working data. FIFA's PDFs and the downloaded web pages are not redistributed.

## Licence

Code: MIT licence (`LICENSE`). Processed data and results: CC BY 4.0 (`data/LICENSE`). The underlying
ballots are published by FIFA and France Football.
