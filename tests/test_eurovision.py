"""Eurovision parsing, national jury aggregation and the jury-standing machinery."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esc_analysis import POINTS, national_points  # noqa: E402
from parse_eurovision import country, jury_span, parse_table, split_cells, value  # noqa: E402
from validate_eurovision import national_order  # noqa: E402

TABLE = """{| class="wikitable"
|+ {{nowrap|Detailed voting results from Sweden (Final)<ref>x</ref>}}
|-
! scope="col" rowspan="2" | R/O
! scope="col" rowspan="2" | Country
! scope="col" colspan="7" | Jury
! scope="col" colspan="2" | Televote
|-
! scope="col" | <small>Juror 1</small>
! scope="col" | <small>Juror 2</small>
! scope="col" | <small>Juror 3</small>
! scope="col" | <small>Juror 4</small>
! scope="col" | <small>Juror 5</small>
! scope="col" | Rank
! scope="col" | Points
! scope="col" | Rank
! scope="col" | Points
|-
! scope="row" style="text-align:center;" | 01
| style="text-align:left;" | {{Esc|Austria|y=2023}} || 9 || 23 || 21 || 19 || 14 || 20 || || 21 ||
|-
! scope="row" style="text-align:center;" | 02
| style="text-align:left;" | {{Esc|Czech Republic|y=2023}}
| 1
| style="background:gold;" | 2
| 3
| 1
| 1
| style="background:gold;" | 1
| style="background:gold;" | '''12'''
| 5
| 6
|}"""


def test_cells_and_names():
    assert [c.strip() for c in split_cells("| {{Esc|Austria|y=2023}} || 9 || 23")] == ["{{Esc|Austria|y=2023}}", "9", "23"]
    assert value(' style="background:gold;" | \'\'\'12\'\'\'') == "12"
    assert country("{{Esc|Czech Republic|y=2023}}") == "Czechia"


def test_final_table_rows_and_juror_columns():
    assert jury_span(TABLE) == 7
    jurors, rows = parse_table(TABLE)
    assert len(jurors) == 5 and len(rows) == 2
    assert rows[0]["song_country"] == "Austria" and rows[0]["jurors"] == ["9", "23", "21", "19", "14"]
    assert rows[1]["song_country"] == "Czechia" and rows[1]["rest"][:2] == ["1", "12"]


def test_national_rules():
    # songs A..E; A is first for three jurors and last for two, B second for all
    ranks = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                      [5, 2, 1, 3, 4], [5, 2, 1, 3, 4]], float)
    assert list(national_order(ranks, None)) == [3, 1, 2, 4, 5]   # sums 13, 10, 11, 18, 23
    assert list(national_order(ranks, 0.5)) == [1, 3, 2, 4, 5]    # exponential weights reward first places


def test_national_points_skip_own_song():
    rng = np.random.default_rng(0)
    ranks = np.tile(np.arange(1, 13, dtype=float), (5, 1))        # 5 jurors, 12 songs, same order
    own = np.zeros(12, bool)
    own[0] = True                                                  # the best-ranked song is the jury's own
    pts = national_points(ranks, own, None, rng)
    assert pts[0] == 0 and list(pts[1:11]) == list(POINTS) and pts[11] == 0


def test_groupings_cover_every_country():
    from build_eurovision import BLOCS, FINE_GROUP, GROUP, REGION
    assert set(GROUP) == set(REGION) and set(REGION) <= set(FINE_GROUP)
    assert len(BLOCS["Viking Empire"]) == 8 and GROUP["Latvia"] == "Viking Empire"   # Gatherer (2006), fig. 2
    assert GROUP["France"] == "Western Europe (other)" and GROUP["Cyprus"] == "Balkan Bloc"


def test_bloc_indicator_uses_literature_blocs_only():
    import pandas as pd
    long = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "esc_long.csv")
    resid = long.voter_bloc.str.endswith("(other)")
    assert long[resid].same_bloc.sum() == 0                       # residual groups carry no bonus
    pair = long[(long.voter_country == "Sweden") & (long.nominee_country == "Latvia")]
    assert len(pair) and pair.same_bloc.eq(1).all()


def test_group_specific_design_matrix():
    import pandas as pd
    from esc_analysis import design_matrix
    d = pd.DataFrame({"same_bloc": [1, 1, 0], "voter_bloc": ["A", "B", "A"], "nominee_bloc": ["A", "B", "C"],
                      "same_region": [1, 0, 0], "voter_region": ["r", "r", "r"], "nominee_region": ["r", "s", "s"],
                      "same_fine": [0, 0, 0], "voter_fine": ["x"] * 3, "nominee_fine": ["y"] * 3,
                      "same_border": [0, 1, 0]})
    X, names = design_matrix(d, "B2")
    assert names == ["same_group[A]", "same_group[B]", "same_border"]
    assert X.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 0]]
    X, names = design_matrix(d, "E1")
    assert names == ["same_group", "same_border"] and X[:, 0].tolist() == [1, 0, 0]
