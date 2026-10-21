# FIFA per-voter ballots, 2010–2025: data notes

Built 2026-10-17 by the scripts in `nhb/src/`. Run them in the order below. Every step is
deterministic, and only the `fetch_*` steps use the network.

| Step | Script | Output (`data/processed/`) |
|---|---|---|
| 1 | `manifest.py`, then `pdftotext -layout` | `raw/fifa/{YEAR}_{MP,WP}.pdf`, `raw/fifa/txt/` |
| 2 | `parse_fifa.py` | `fifa_ballots_raw.csv`, `fifa_parse_issues.csv`, `fifa_shortlist_seen.csv`, `fifa_official_top3.csv` |
| 3 | `nominees.py` | `fifa_nominee_names.csv` (raw spelling → person) |
| 4 | `nominee_meta.py` | `fifa_nominee_meta.csv` (nation, clubs, club countries, published score) |
| 5 | `countries.py` | `countries.csv` (name, FIFA code, confederation), `country_aliases.csv` |
| 6 | `validate_fifa.py` | `fifa_validation.csv`, `fifa_validation_shares.csv` |
| 7 | `build_fifa.py` | `fifa_ballots.csv` (one row per ballot), `fifa_long.csv` (ballot × nominee) |
| 8 | `fetch_managers.py`, then `coach_nationality.py` | `raw/wikipedia/managers/`, `manager_snapshots.csv`, `fifa_coach_nationality.csv` (men's teams, dated manager lists) |
| 9 | `fetch_team_pages.py`, then `coach_history.py` | `raw/wikipedia/teams/`, `coach_history_entries.csv`, `fifa_coach_nationality_all.csv` (men's and women's teams; used in the analysis) |

## Sources

- **Ballots:** FIFA's published per-voter PDFs for the men's (MP) and women's (WP) player awards.
  These are the FIFA Ballon d'Or for 2010–2015 and The Best FIFA Football Awards for 2016–2025.
  The URLs are in `raw/fifa/manifest.csv`, and the 2010–2017 files come from Wayback copies.
- **Nominee nationality and clubs:** the Wikipedia pages for each year's awards, saved through the
  MediaWiki parse API in `raw/wikipedia/awards_{year}.json`. The 2025 FIFA page lists only the top
  three, so most other 2025 nominees come from the 2025 Ballon d'Or page. Its voting period is
  1 August 2024 to 2 August 2025, against FIFA's 11 August 2024 to 2 August 2025. Four women
  nominees appear on neither page, and their clubs come from their own infoboxes. The `source`
  column records which page each row came from. A player who changed clubs during the voting
  period has every club listed, and each club country is that of the club's flag.
- **Countries and confederations:** Wikipedia's "List of men's national association football
  teams" (sections by confederation) and "List of FIFA country codes".
- **Coach nationality, step 1 (men's teams):** 64 quarterly revisions (2010–2025) of Wikipedia's
  "List of current national association football team managers". Each entry gives a team, its
  manager and the manager's flag. Three properties of this source shape the matching:
  - The list covered only UEFA teams (about 55) until early 2024, then all 221 members.
  - Its 2016–2018 revisions show no flags. Nationality is therefore treated as a property of the
    person, filled from any revision that shows a flag on the same manager page.
  - A coach ballot is matched by name to that team's manager in the award year's snapshots, and
    otherwise in any snapshot (column `match`). Name plus team identifies the person.

  Result: 1,130 of the 2,847 men's coach ballots have a nationality (420 coaches), and 508 of them
  come from foreign coaches. Coverage is 37–52 ballots a year in 2010–2019, 68–105 in 2020–2022 and
  144–151 in 2023–2025. Nine ballots matched a coach listed with two nationalities; their
  nationality is left blank.
- **Coach nationality, step 2 (men's and women's teams; the file used in the analysis):** the
  Wikipedia pages of 442 men's and women's national teams (`raw/wikipedia/teams/`), whose coaching
  histories and staff lists give each listed coach with a flag (4,649 entries from 339 pages). A
  coach ballot keeps its step-1 match where there is one; otherwise it is matched by name to the
  coaches listed on the page of the juror's team (men's page for the men's award, women's page for the
  women's award). Name plus team identifies the person, so no dates are needed; a name matching two
  listed people, or a person listed with two flags, stays unmatched (6 ballots). Each matched ballot
  carries a coach identifier (the coach's Wikipedia page) used to follow coaches across teams.

  Result (`fifa_coach_nationality_all.csv`): 2,727 of 5,364 coach ballots have a nationality
  (1,130 from step 1, 1,597 from step 2; 1,031 distinct coaches), 1,246 of them cast by foreign
  coaches. Men's award: 2,011 of 2,847 ballots (71%), 1,012 foreign. Women's award: 716 of 2,517
  (28%), 234 foreign.

## Parsing

The layout changes across years:
- 2010 has a section per role;
- from 2011 there is a leading role column;
- names are surname-first up to 2019;
- some rows run cells together, with one space or none between them.

Each contest has a closed shortlist, so the parser matches the last three cells of a row against
the names that appear in cleanly separated rows. A pooled list from all contests is the fallback.
It was needed once, for Benzema in 2012. Other handling:
- **Partial ballots** (2) are placed by column position under the page header.
- **Wrapped picks** (1) and wrapped country or voter names (16) are joined to their row.
- **Placeholders** are kept as codes: `INVALID` (23 cells), `NO_VOTE` (26), `NA` (3; the 2024
  Jamaica captain), `NOT_COUNTED` (1; FIFA footnote "vote ineligible") and blank (3).
- **FIFA's test record:** the 2017 file contains a test entry (voter name beginning "Test_") as a second Swiss media
  ballot. It is flagged `test_entry` and excluded from `fifa_ballots.csv`.

## Validation (`validate_fifa.py`)

- **2023 and 2024:** FIFA prints the raw per-group points of the top three. All 36 cells (four
  contests × three nominees × three juror groups) match the recomputed points exactly.
- **2010–2015:** Wikipedia lists every nominee's overall percentage. The recomputed share is the
  average of the three juror groups' shares of 5/3/1 points. Its largest gap from the published
  figure is at most 0.01 percentage points in 2013–2015 and 0.03–0.23 points in 2010–2012. The
  2013 women's percentages are not published.
- No ballot names the same player twice. Each (contest, role, country) has one ballot, except the
  test record above.

## Analysis tables

`fifa_ballots.csv` has one row per ballot: 15,385 ballots in 32 contests. Its columns:
- ballot, year, award, contest, role (captain, coach or media);
- the voter's country and confederation, and the voter's name;
- pick1–pick3, with nominees mapped to persons;
- incomplete, n_valid.

`fifa_long.csv` has one row per ballot × nominee: 226,805 rows. Its columns:
- the nominee's country and confederation, and their clubs and club countries;
- rank (1–3 for the valid picks in order, 0 if not picked);
- same_nation, same_confed and same_league (the voter's country is among the nominee's club
  countries).

It also follows these rules:
- **Choice sets:** each contest's choice set is the nominees who received at least one pick. A
  nominee never picked has a fixed effect of minus infinity, so dropping them changes no choice
  probability.
- **2010 rule:** FIFA's 2010 rules voided votes by captains and coaches for players of their own
  national team. Those compatriots are removed from 2010 captain and coach choice sets (55 pairs,
  none picked; journalists were allowed and picked 9 of 31).
- **Captains as nominees:** 115 captain ballots came from a nominee, who is removed from their own
  choice set because FIFA forbids voting for oneself.
- **Confederation affinity:** same_confed includes compatriots. In models with both terms,
  same_nation is therefore the extra bonus for a compatriot.
- **Gaps in a ballot:** when a placeholder sits between valid picks, the valid picks are
  renumbered 1, 2, … in their printed order.

## Known limits

- The Best (2016–2025) adds a fan vote as a fourth quarter. Fan votes are not published per voter,
  and their totals are published only for the top three in 2023 and 2024. Outcome analyses for
  those years therefore concern jury standings.
- Coach nationality is missing for 29% of the men's and 72% of the women's coach ballots, most often
  for small associations whose Wikipedia pages list no coaching history.
- `fifa_award_winners_manual.csv` records award winners that neither FIFA's top-three tables nor the
  scores on FIFA's award pages give (the 2025 women's award), each with its source.
- From 2023 FIFA decides the player awards by rank-based scoring points within each group (Rules of
  Allocation, art. 11-12, printed in the 2023 and 2024 result files); the analysis applies that rule
  to 2023-2025 and the group-share rule to 2010-2022 (`src/fifa_rules.py`). The rules are silent on
  ties within a group; tied nominees receive the scoring points of the best position they share.
  Subtracting the recomputed jury scores from the published totals of every nominee in 2023 and
  2024 leaves each nominee a distinct fan scoring value only under this convention (the average or
  the worst of the shared positions fails in at least one contest). Whether nominees without points
  in a group share the next place or score nothing cannot be told apart from these totals.
  `src/check_fifa_scoring.py` runs the comparison on the raw 2023 and 2024 result files and writes
  `fifa_scoring_conventions.csv`.
- Journalists' and captains' own nationalities are not recorded. Captains normally hold the
  nationality of their team.

## Ballon d'Or journalist ballots

Built 2026-10-18 by the scripts below. Only `fetch_bdo.py` uses the network. The France Football
jury is one journalist per country. Up to 2021 that meant about 176–180 countries; from 2022 the
men's jury is drawn from the FIFA top 100 and the women's from the top 50.

| Step | Script | Output (`data/processed/` unless stated) |
|---|---|---|
| 1 | `fetch_bdo.py` | `raw/ballondor/{YEAR}_{MP,WP}/page.html`, `raw/ballondor/{YEAR}_wiki/wikipedia.json` (each with `source.json`) |
| 2 | `parse_bdo.py` (uses `countries_fr.py`) | `bdo_ballots_raw.csv`, `bdo_wiki_ballots.csv`, `bdo_official_points.csv`, `bdo_parse_issues.csv` |
| 3 | `companion_bdo.py` | `bdo_ballots_companion.csv`, `bdo_companion_issues.csv` |
| 4 | `bdo_nominees.py` | `bdo_nominee_names.csv` (spelling → person), `bdo_nominee_meta.csv` |
| 5 | `build_bdo.py` | `bdo_ballots.csv` (one row per juror, picks as persons) |
| 6 | `validate_bdo.py` | `bdo_validation.csv`, `bdo_validation_summary.csv`, `bdo_crosscheck.csv` |

### Contents

`bdo_ballots.csv` holds 706 ballots. Every juror row has these columns:
- year, award (MP or WP);
- country_raw (as printed), country (canonical, as in `countries.csv`);
- juror, outlet;
- k, and pick1–pick10 (blank beyond k);
- source, flag.

`country` is the country the juror votes for, not the juror's nationality.

| Contest | Jurors | k | Juror names | Source |
|---|---|---|---|---|
| Men 2019 | 176 | 5 | no | companion-study data (France Football print issue, 4 Dec 2019) |
| Men 2022 | 93 | 5 | yes | L'Équipe/FF page, Wayback 20250322040635 |
| Men 2023 | 92 | 5 | no | Wikipedia "Ballon D'Or Detailed Votes by Country" |
| Men 2024 | 99 | 10 | no | companion-study data |
| Men 2025 | 100 | 10 | yes | L'Équipe page, Wayback 20260510112921 |
| Women 2023 | 47 | 5 | yes | L'Équipe page, Wayback 20240227144843 |
| Women 2024 | 49 | 10 | yes | L'Équipe page, Wayback 20260914175433 |
| Women 2025 | 50 | 10 | yes | L'Équipe page, Wayback 20260306032405 |

Points are 6-4-3-2-1 for k = 5 and 15-12-10-8-7-5-4-3-2-1 for k = 10.

### Sources

- **L'Équipe / France Football pages.** These are Wayback raw captures (`id_`), so they come
  without the toolbar. The women's 2024 and 2025 captures were stored gzip-encoded and are
  decoded on download.
- **Companion-study ballots, men 2019 and 2024.** They are read, never modified, from
  the companion study's `model_input/{2019,2024}/` (next to `nhb/`):
  - `rankings_long.csv` holds voter, rank, player key and points: 880 rows (176 × 5) and
    990 rows (99 × 10);
  - `voter_source.csv` holds voter, continent and source;
  - `model_players.csv` gives the 30 nominees, with player key and full_name.

  The voter is the country, in French for 2019 and in English for 2024. Each pick is written as
  the companion-study full name.
- **Wikipedia "{year} Ballon d'Or".** These pages come from the MediaWiki parse API: 2019 and
  2022–2024 are in `raw/ballondor/{YEAR}_wiki/` and 2025 in `raw/wikipedia/ballondor_2025.json`.
  They supply three things: the nominee lists, the official points for every nominee, and nominee
  nationality and clubs. Their per-country tables (men 2023–2025) give the points per juror
  country but no names.

### Parsing

- **Article text.** Only headings and paragraphs inside the article body are read. Teasers for
  other articles, image captions and boxed text are skipped.
- **Ballot headers.** A ballot opens with a French country name from `countries_fr.py`. That file
  maps 182 French spellings by hand to 179 canonical countries. Spellings are compared
  accent-, case- and punctuation-free, so "ISRAEL" and "COTE D'IVOIRE" match. One typo is listed
  explicitly ("Brésll", women 2025). "Eire" and "Irlande" both mean the Republic of Ireland. Every
  juror country resolves.
- **Layouts.** Men 2022 prints "Pays - Juré - Média" on one line. Men 2025 prints the country in
  capitals, with "Juré, Média" in the same heading. The women's pages print the country as a
  heading and "Juré, Média" in the next paragraph.
- **Ranks.** Rank lines read "N. Name". One line on the men's 2025 page holds two ranks (Mali:
  "4. Mbappé5. Hakimi"). It is split by its printed numbers and flagged `several_ranks_one_line`.
  The parser would also join a rank whose name sits on the next line. That does not occur in the
  saved HTML: the women's 2024 "8. *Katoto*" is one paragraph with the name in bold, and bold is
  used only for French players. No ballot has missing, repeated or out-of-order rank numbers, and
  `bdo_parse_issues.csv` is empty.
- **Men 2023.** Ranks come from Wikipedia's per-country points (6 → 1st, …, 1 → 5th). Every one of
  the 92 rows holds exactly one each of 6, 4, 3, 2 and 1.
- **Nominee identities.** Each spelling is matched within its contest's 30 Wikipedia nominees by
  one of three rules:
  - the same `nominees.key()`;
  - every token matching a different token of exactly one nominee, where an abbreviated token
    must be a prefix ("D. Doué", "Te. Chawinga" → Temwa, not Tabitha) and "Jr" = "Junior";
  - four hand-checked aliases (`MANUAL` in `bdo_nominees.py`): "Alisson Becker", L'Équipe's typos
    "Son Heug-min" and "Alba Redon do", and "Leonhardsen Maanum".

  The 421 spellings resolve 266 by key, 151 by tokens and 4 by alias, with none ambiguous or
  unresolved. `person` is the Wikipedia name, unified across years ("Lindsey Horan" in 2024 and
  "Lindsey Heaps" in 2025 are one person). `fifa_person` links 133 of the 240 nominee rows to the
  FIFA dataset's identities.
- **Nominee metadata** (`bdo_nominee_meta.csv`: nation, clubs, club_countries joined by ';').
  These follow `nominee_meta.py` and use the flag of each club. The 2019 nationality is the flag
  in the player cell. From 2022 Wikipedia lists one club and explains moves in footnotes:
  - a move "during the season" adds the earlier club. This affects Díaz, Vlahović and Ronaldo
    in 2022, Messi in 2023, and Banda, Ramírez and Tarciane in 2024;
  - a loan "from Lyon to Paris Saint-Germain" adds both clubs (Chawinga, 2024);
  - moves "after the end of season" add nothing.

  Debinha's undated footnote (2023) is resolved from L'Équipe's ranking, "Courage de Caroline du
  Nord puis Current de Kansas City". A footnote club's country comes from that club's flag
  elsewhere on the saved Wikipedia pages. Only Fiorentina needed a hand entry.

### Validation (`validate_bdo.py`)

- **Points.** Each nominee's total is recomputed and compared with the Wikipedia ranking table
  for all 8 contests, and with L'Équipe's own ranking for men 2022 and women 2023 and 2024.
  - Men 2022–2025 and women 2024–2025 match every nominee exactly.
  - Women 2023 matches L'Équipe exactly. It differs from Wikipedia only for Debinha (Wikipedia 1,
    ballots 0). L'Équipe gives her 0, and Wikipedia's total exceeds 47 full ballots by one point,
    so the Wikipedia cell is wrong.
  - Men 2019 differs by one point for two players: Ronaldo has 475 against the official 476, and
    Sterling 31 against 30. The companion study logs the same difference. Some 2019 ballot in the
    companion-study source must have Sterling one point too high and Ronaldo one too low. Four ballots fit:
    Australia, Norway and the Philippines (Sterling directly above Ronaldo) and Pakistan (Sterling
    fifth, no Ronaldo). The print page is needed to tell which, so the data are left unchanged.
- **Juror counts.** The parsed counts equal the counts the sources state: 176 (2019), 93 (2022:
  "89 fois sur 93"), 99 (2024), 100 (2025: "les 100 jurés"), 47 (women 2023: "42 fois sur 47")
  and 49 (women 2024: "les votes des 49 jurés"). They also equal the counts implied by the official
  points, for every contest including men 2023 (92) and women 2025 (50). The inventory's earlier
  counts of 92 (men 2022) and 48 (women 2024) were parsing shortfalls.
- **Statements in the articles.** Ten statements are reproduced exactly:
  - Benzema 89 first, 3 second and 1 third places, and De Bruyne's single first place (2022);
  - Bonmatí first 42 times, absent only from Chile's top 5 (women 2023);
  - Dembélé named on all 100 ballots, 73 times first (2025);
  - Bonmatí named 49 times and Caldentey 44 (women 2025).
- **Integrity.** No ballot names a player twice or leaves a slot empty, and there is one ballot
  per country.
- **Cross-source.** Juror by juror, the companion-study 2024 ballots (99) and the L'Équipe 2025 ballots (100)
  are identical to Wikipedia's per-country tables.

### Known limits

- **No juror names or outlets** for men 2019, 2023 and 2024. The 2023 ballots rest on Wikipedia's
  per-country table, which cites no source. It agrees with Wikipedia's ranking table, but both
  come from Wikipedia; L'Équipe's 2023 list is paywalled.
- **Missing countries.** Seven of the 100 countries invited in 2022 did not vote: Costa Rica,
  Scotland, Burkina Faso, Curaçao, Jamaica, the UAE and Uzbekistan, per Goal.fr in the inventory.
  None is in the data. Men 2023 has 92 of 100 and men 2024 has 99. Which invitees are missing in
  those years is not recorded.
- **Unresolved 2019 discrepancy.** One 2019 ballot cannot be pinned down (see Validation).
- **Clubs** follow Wikipedia's convention: clubs during the eligibility period, with moves after
  it excluded. L'Équipe's ranking box also lists post-season moves ("Liverpool puis Bayern
  Munich"); those are not used.
- **Missing contests.** Men 2016–2018 and 2021, and women 2018–2022, are not included: their
  per-juror lists are in print, images or paywalled pages (see `inventory_agents/ballon_dor.md`).

## Eurovision Song Contest juries (second domain)

**Source.** The EBU publishes, after every final, each national jury member's ranking of every song;
Wikipedia's country pages ("<Country> in the Eurovision Song Contest <year>") reproduce them in a
"Detailed voting results ... (Final)" table, with the national jury's rank and points and the
televote. `src/fetch_eurovision.py` downloads these pages and the main contest pages (wikitext, via
the MediaWiki API, 8 s between requests) for the finals of 2016-2019 and 2021-2026 (no contest in
2020) into `data/raw/eurovision/wikipedia/`. The series starts in 2016, the first contest in which each
country's jury awarded its own set of points, separately from the televote (Wikipedia, "Voting at the
Eurovision Song Contest", section "Voting systems", read 2026-10-21); until 2015 jury and televote
rankings were merged into one set of points, so no official jury standing exists to reproduce.

| Step | Script | Output |
|---|---|---|
| 1 | `fetch_eurovision.py` | `data/raw/eurovision/wikipedia/*.json` |
| 2 | `parse_eurovision.py` | `esc_juror_ranks.csv`, `esc_national.csv`, `esc_jurors.csv`, `esc_jury_totals.csv`, `esc_parse_issues.csv`, `esc_sources.csv` (page revisions) |
| 3 | `validate_eurovision.py` | `esc_rule_fit.csv`, `esc_validation.csv` |
| 4 | `build_eurovision.py` | `esc_long.csv`, `esc_regions.csv`, `esc_borders.csv` |

**Coverage.** 383 national juries in 10 finals, 1,974 juror rankings from 45 countries (juries of five
members, seven in 2026; three juries ranked with four members, 2023 Ireland and Latvia and 2025
Estonia, and four in 2026 with six). Pages without a final table
are legitimate absences (`esc_parse_issues.csv`): countries that did not take part in the final's
vote (e.g. Romania 2016, Russia 2017 and 2022, Ukraine 2019, the 2026 withdrawals), Belarus 2019
(jury dismissed) and 2021, and the six juries the EBU replaced in 2022 (Azerbaijan, Georgia,
Montenegro, Poland, Romania, San Marino; "irregular voting patterns", replaced by substitute aggregated
results in the semi-final and the final). Ten national juries voted in a final without complete
published rankings (Belarus 2019, the six of 2022, San Marino 2025, Poland and Serbia 2026); their
points enter the jury standings as published, and they are left out of the ranking model. If the
2022 patterns reflected favouritism within blocs, leaving them out makes the bloc bonuses smaller.

**Aggregation rule.** The EBU does not publish the weights it has used since 2018. 2016-2017: the
national ranking is the order of the jurors' summed ranks; every published ranking is consistent with
this order, but exact ties are broken in no consistent way (none of seven candidate tie-breaks fits),
so 20 of 84 rankings and 54 of 84 sets of points are reproduced exactly; analyses break such ties at
random. 2018-2026: each rank r scores 0.8269^(r-1) and the national ranking orders the sums; this
base (fitted on a grid of step 0.0001, pooled over years) reproduces 295 of 299 national rankings
and the points of all 299 juries. In 2024 the Netherlands was disqualified after the jury show; its
song keeps its jurors' ranks but receives no points, and the songs ranked below it move up a place
(`validate_eurovision.INELIGIBLE`).

**Jury standings.** `esc_jury_totals.csv` gives the official jury score of every song from the
scoreboards of the main contest pages; the juries without complete rankings contribute the official
total minus the complete juries' published points, so recomputed standings start from the official
numbers.

**Similarity.** Three groupings of countries (`build_eurovision.py`). (i) Voting blocs, the main one:
the five blocs that Gatherer (2006, JASSS 9(2):1) identified in the contests of 2001-2005, before
these data (the "Viking Empire" of the Nordic and Baltic states, the "Balkan Bloc" of the former
Yugoslav states with Greece, Cyprus, Romania and Albania, the "Warsaw Pact" of Russia, Ukraine and
Poland, the "Pyrenean Axis" of Spain and Andorra, and the Netherlands with Belgium; Turkey, a Balkan
member, has not competed since 2012). The bonus indicator `same_bloc` uses these blocs only; for the
jury-design simulations a country in no bloc is grouped with the other such countries of its UN M49
subregion (`voter_bloc`, `nominee_bloc`), and these residual groups carry no bonus. (ii) UN M49 subregions alone (checked against the UN Statistics
Division's table on 2026-10-20). (iii) A finer grouping chosen after the main analysis, as a check
(`FINE`). Shared land borders come from Wikipedia's "List of countries
and territories by number of land borders" (two borders through overseas territories, Cyprus-United
Kingdom and Spain-United Kingdom, excluded). Jurors may not rank their own country's song, so there is
no compatriot term. Juror names are kept in `esc_jurors.csv` only for checking and are left out of
the deposit.
