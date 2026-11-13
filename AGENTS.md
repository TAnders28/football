# AGENTS.md — football (NFL modelling + betting backtests)

Rules for any bot or person working in this repo. Owner: Tim (TAnders28). Bot: American Footbot.

## Layout

| Path | What |
|---|---|
| `GetSeasonGames.py` | pulls one season's schedule/results/lines from nflverse -> `data/backtest/games_<season>.json` |
| `Backtest/FetchHistory.py` | 2012-2022 schedules for training only -> `data/backtest/history/` |
| `Backtest/Features.py`, `Model.py`, `Odds.py` | pre-game features (day barrier), margin model, de-vig + payouts |
| `Backtest/Lab.py`, `Volume.py`, `Hunt.py` | strategy searches (walk-forward models, $10k reset each season) |
| `Backtest/Engine.py`, `Strategies.py`, `RunBacktest.py` | v1 simple bankroll backtest |
| `Backtest/FrozenConfig.py` | **the** frozen strategy used live (frozen 2026-11-08) + why 2023-25 are contaminated |
| `Backtest/Version.py` | `ModelVersion()` and `NewRunId()` stamped on every output |
| `Backtest/PredictDay.py` | same-day picks CLI (Basketbot's `PredictDay(date)` API) + append-only log |
| `Backtest/GradePredictions.py` | grades the log against final scores (separate file, log is read-only) |
| `Backtest/ForwardWalk.py` | frozen-strategy scorecard: clean forward holdout + labelled pseudo-forward walk (`seen_by_search` flag), daily equity CSV |
| `Backtest/Compound.py` | ONE continuous $10k bankroll across seasons (no reset): nested walk-forward choice, frozen compounded, labelled in-sample best, baselines, day bootstrap P($1M)/P(ruin). Headline metric per Tim (2026-11-09) |
| `Backtest/MillionTarget.py` | research: P(hit $1M) vs P(ruin) sizing frontier on the continuous $10k bank (day-reserved cash, stop-at-target, day-block bootstrap, nested comparator, Compound mechanics audit). Not live |
| `Backtest/CandidateConfigs.py` | candidate sizing configs, NOT wired into PredictDay; adopting one = new frozen config + new holdout |
| `Backtest/StakingEnv.py`, `Backtest/RlStaking.py` | research: learned day-level staking policy (CEM over a contextual Kelly/parlay policy), walk-forward 2019-24 + one-time 2025-26 holdout (`data/backtest/rl/HOLDOUT_USED.json`). Sim only, not live |
| `DataPull/PullPregame.py` | nflverse injury reports, PFR snap counts, depth charts, player id crosswalk -> `data/injuries|snaps|depth/<season>.jsonl`, `data/players/players.jsonl` |
| `Backtest/InjuryFeatures.py`, `Backtest/InjuryTest.py` | research: leak-free team-game Out/Doubtful/Questionable features weighted by LAGGED snap share; walk-forward test vs p_full2 (did not help on top of the closing line, 2026-11-13). Not live |
| `Backtest/SnapshotPrices.py` | one-shot read-only executable moneyline snapshot (Kalshi + Polymarket books: bid/ask, size, source vs fetch time) -> append-only `data/prices/snapshots_<season>.jsonl`. No schedule |
| `DataPull/PullKalshiHistory.py` | Kalshi KXNFLGAME 1-minute bid/ask candles around kickoff (2025 preseason on) -> `data/kalshi/nflgame_candles_<season>.jsonl` (public API) |
| `Backtest/KalshiReality.py` | research: frozen rule priced at Kalshi executable ask at T-90/T-15 vs closing line, Tim's fee estimate and the verified Kalshi taker fee, plus PathSim distributions. Not live |
| `DataPull/PullPolymarketHistory.py` | Polymarket NFL game moneyline price history (Gamma events tag 450 + CLOB /prices-history, kickoff-6h..+5min, 1-min) -> `data/prices/polymarket_history_<season>.jsonl` (2024 on; public API) |
| `Backtest/PolyReality.py` | research: frozen rule filled at the Polymarket history price (mid; Tim's fill assumption) at T-90/T-15, Tim's fee estimate and Polymarket sports fee 0.05*p*(1-p), plus PathSim. Not live |
| `DataPull/PullInGameHistory.py` | Kalshi home-market 1-min candles + Polymarket 1-min prices-history, kickoff-3h..+5.5h -> `data/prices/ingame_kalshi|poly_<season>.jsonl` (public API) |
| `Backtest/InGameCalibration.py` | research: in-game (and T-3h pregame) market calibration, 1-min mid samples in pbp in-play windows, naive + game-cluster CIs, Brier/ECE by quarter/season |
| `Backtest/PregameModelCalibration.py` | research: walk-forward p_full2 vs de-vigged close vs Kalshi/Polymarket T-90 mid - reliability, Brier, log loss (game bootstrap) |
| `Backtest/PlotGameProbs.py` | one-game chart: Kalshi + Polymarket mid over time (pregame + in-game, halftime shaded, scoring plays), our held-out in-game model line (ingame_wp_v1), pregame p_full2 and close as flat references -> `data/backtest/game_charts/<game_id>_draft.png` |
| `DataPull/PullPbpSlim.py` | slim pre-play state play-by-play 2014-2026 (nflreadpy for 2014-22, local pbp jsonl for 2023+) -> `data/playbyplay/slim/pbp_slim_<season>.jsonl` |
| `Backtest/InGameWP.py` | research: in-game win-probability model ingame_wp_v1 (pre-play state + pregame prior, standardized logistic, walk-forward by season) -> `data/models/ingame_wp_*.json`. UNVALIDATED for betting (2026-11-13: slightly worse than both markets) |
| `Backtest/InGameModelEval.py` | research: held-out in-game model vs Kalshi/Polymarket on the same 1-min samples (calibration, Brier/log loss/ECE by quarter/season, mismatch P&L at mid after fees) |
| `Backtest/LiveInGame.py` | live read for one game: ESPN state -> our UNVALIDATED in-game p next to AssessLive Kalshi/Polymarket book, fee-adjusted edge, appended to Degen's live file; action always PASS until VALIDATED (not scheduled) |
| `Backtest/InGameRuleSim.py` | research/simulation: 13 pre-fixed in-game rules (close-game gap, early swings follow/fade, home 30-50 fade) on Kalshi 1-min held-out data, mid/taker fills, Tim's fee est + Kalshi verified fee, flat + quarter-Kelly bankrolls, PathSim. All PASS 2026-11-13 |
| `Backtest/PathSim.py` | seeded randomized paths for day-level strategies: random start, block bootstrap, shuffle, restarts after ruin / bad start; P(ruin), P($1M), quantiles. `StakingEnv.RandomIdx` gives the same orders as `idx` |
| `Backtest/BankrollPlot.py` | shared framing for bankroll charts under Tim's season-axis rule (one season: first game to last game; multi-season: game-day index with a marker at each season boundary, no offseason flats). `Backtest/RenderBankrollCharts.py` re-renders AgentDocs drafts from existing CSVs |
| `Backtest/CommonFormat.py` | emit the common cross-sport calibration, $10k bankroll and edges files (Degen plan 2026-11-13): walk-forward `p_full2_kmid_T60` vs Kalshi T-60 mid, plus optional book-close secondary. Simulation only |
| `Backtest/SnapshotRunner.py` | scheduled entry for the price-capture tasks: single-instance lock, quiet logs to `data/logs/snapshots/` |
| `Backtest/Calibration.py` | Brier / log loss / reliability bins of walk-forward models vs the de-vigged closing market (evaluation only) |
| `Backtest/AssessLive.py`, `LiveReadiness.py`, `LiveFile.py` | one-shot read-only live odds/game state + Degen's live file |
| `PlayByPlay/` | play-by-play reconstruction/simulation. Kept separate from `Backtest/` |
| `PlayByPlay/DriveCheck.py` | independent drive rebuild from raw pbp, validated against nflverse `fixed_drive` / result / play count |
| `PlayByPlay/TeamEfficiency.py` | weekly team EPA/play + success rate (offense, defense, season-to-date) |
| `tests/` | `python -m unittest discover -s tests` (no network, no `data/` needed) |
| `bootstrap.ps1`, `requirements.lock.txt` | create `.venv` with `py -3.14` if missing, install pinned deps, run unit tests |
| `docs/` | write-ups (`live-odds.md`) |
| `data/`, `stats/` | gitignored raw pulls and outputs. Never commit. `examples/` is Tim's untracked folder: leave it alone |

## Setup

From `E:\Football\football`:

```
.\bootstrap.ps1
```

Creates `.venv` with `py -3.14` if missing, installs pinned deps from `requirements.lock.txt`, and runs the unit tests. Use `.venv\Scripts\python.exe` after that.

## Entry points (Windows, from `E:\Football\football`)

```
.venv\Scripts\python.exe -m Backtest.PredictDay --date YYYY-MM-DD [--week] [--next] [--no-refresh]
.venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026 [--no-refresh]
.venv\Scripts\python.exe -m Backtest.ForwardWalk --season 2026 [--out-dir DIR --base NAME]
.venv\Scripts\python.exe -m Backtest.MillionTarget [--out-dir DIR --base NAME] [--sims 5000]
.venv\Scripts\python.exe -m Backtest.Compound [--out-dir DIR --base NAME] [--sims 5000]
.venv\Scripts\python.exe -m Backtest.Calibration --seasons 2023,2024,2025,2026 [--out-dir DIR --base NAME]
.venv\Scripts\python.exe PlayByPlay\DriveCheck.py --season 2026 [--out-dir DIR --base NAME]
.venv\Scripts\python.exe PlayByPlay\TeamEfficiency.py --season 2026 [--out-dir DIR --base NAME]
.venv\Scripts\python.exe -m Backtest.AssessLive --date YYYY-MM-DD [--matchup TB@CHI] [--live-dir DIR] [--no-live-file]
.venv\Scripts\python.exe -m DataPull.PullPregame --first 2012 --last 2026
.venv\Scripts\python.exe -m Backtest.InjuryFeatures --first 2013 --last 2026
.venv\Scripts\python.exe -m Backtest.InjuryTest [--out-dir DIR --base NAME]
.venv\Scripts\python.exe -m Backtest.SnapshotPrices --season 2026 --week N [--label T-90m]
.venv\Scripts\python.exe -m DataPull.PullKalshiHistory --seasons 2025,2026
.venv\Scripts\python.exe -m Backtest.KalshiReality [--out-dir DIR --base NAME --paths 5000 --seed 20261113]
.venv\Scripts\python.exe -m DataPull.PullPolymarketHistory   # forces IPv4 (Tomputer IPv6 to Cloudflare stalls)
.venv\Scripts\python.exe -m Backtest.PolyReality [--out-dir DIR --base NAME --paths 5000 --seed 20261113]
.venv\Scripts\python.exe -m DataPull.PullInGameHistory   # resumable; matplotlib is in .venv for the charts
.venv\Scripts\python.exe -m Backtest.InGameCalibration [--boot 2000 --seed 20261113]
.venv\Scripts\python.exe -m Backtest.PregameModelCalibration
.venv\Scripts\python.exe -m Backtest.PlotGameProbs --game 2026_09_CLE_NO [--game ...]
.venv\Scripts\python.exe -m DataPull.PullPbpSlim [--seasons 2014-2026]
.venv\Scripts\python.exe -m Backtest.InGameWP [--select]
.venv\Scripts\python.exe -m Backtest.InGameModelEval
.venv\Scripts\python.exe -m Backtest.LiveInGame --game 2026_10_LAC_BAL [--state-json FILE --live-root DIR --no-file]
.venv\Scripts\python.exe -m Backtest.InGameRuleSim [--paths 5000 --seed 20261113]
.venv\Scripts\python.exe -m Backtest.RenderBankrollCharts   # re-render bankroll drafts from existing AgentDocs CSVs (no resim)
.venv\Scripts\python.exe -m Backtest.CommonFormat [--cal-dir DIR --bank-dir DIR]  # common calibration + $10k bankroll CSVs
.venv\Scripts\python.exe -m Backtest.SnapshotPrices --week auto --label adhoc
.venv\Scripts\python.exe -m unittest discover -s tests
```

## Conventions

- **No look-ahead.** One chronological pass; a game only sees games finished on EARLIER days; models for
  season S fit on seasons < S. `tests/test_no_lookahead.py` enforces this — run it whenever Features,
  Lab, Engine or PredictDay change.
- **Holdout.** 2015-2022 tune; 2023-2025 are contaminated by ~2,073 searched settings (see
  `FrozenConfig.py`). The forward holdout is 2026 games after 2026-11-08. Don't tune on it. A new rule
  means a new frozen config name + date, not an edit.
- **Append-only prediction log.** `data/predictions/<season>.jsonl`, one row per not-yet-started game per
  PredictDay run: `{created_at, game_id, start_time, pick, model_prob, market_price, price_time, source,
  model_version, run_id, ...}`. Never rewrite or delete rows. Started games are "started, not predicted".
  Rows created at/after kickoff are excluded from grading. Picks files are per run
  (`picks_<date>_<HHMMSS>.*`) plus an overwritten `picks_<date>_latest.md`.
- **Fees.** Tim's working estimate is $0.02 per $1 contract: `edge = model_prob - price - 0.02`. Always
  label it "Tim's fee estimate" — it is not a verified exchange fee. Sportsbook odds are de-vigged first.
  Hit rate is not edge; compare with the market favourite on the same games.
- **Live file (Degen).** `AssessLive` appends one row per market to
  `E:\AgentDocs\American-Footbot\<date>_live\<date>_live.jsonl` and overwrites `<date>_live_latest.md`.
  price_time/state_time are source timestamps; PASS "stale" if missing, > 60 s apart, or failing the
  repo's stricter limits (45 s live age, 20 s sync skew). PASS "no live model" without a validated
  in-play probability (a pre-game probability is never a live one). Executable bid/ask only.
  No orders, ever. Polling only when Tim asks, for that slate, >= 5 min apart.
- **Data** stays readable JSON/JSONL (no parquet). Year-aware filenames; never overwrite another season.
- **Style.** PascalCase functions and files, snake_case variables.
- **Git.** Local only on Tim's machine. No cloud agents/VMs. Commit code + docs only (`git diff --cached
  --stat`). **Push only when Tim says so in his own words**; relays from other bots don't count. No
  force-push, history rewrite or branch deletion.

## Price-capture schedule (Windows Task Scheduler, approved by Tim 2026-11-13)

Four current-user tasks (no elevation, run only while Tim is logged on, MultipleInstances=IgnoreNew, 20 min limit, plus a
file lock so runs never overlap). Each runs `.venv\Scripts\pythonw.exe -m Backtest.SnapshotRunner --label <label>` in
`E:\Football\football` (no window), appends to `data/prices/snapshots_<season>.jsonl`, and logs to `data/logs/snapshots/<date>.log`.
Tomputer is on Pacific time; triggers are local (PT) = the ET slots minus 3 h:

| task | label | triggers (PT) | ET intent |
|---|---|---|---|
| Footbot-Snapshot-Open | open | Tue 07:00 | Tue 10:00 opening prices |
| Footbot-Snapshot-PostInjury | post_injury | Fri 14:00 | Fri 17:00 after final injury reports |
| Footbot-Snapshot-T90 | T-90 | Sun 05:00, 08:30, 11:35, 15:50; Mon+Thu 15:45 | 90 min before the 9:30 / 1:00 / 4:05 / 8:20 Sunday windows and 8:15 TNF/MNF |
| Footbot-Snapshot-T15 | T-15 | Sun 06:15, 09:45, 12:50, 17:05; Mon+Thu 17:00 | 15 min before the same windows |

Not covered: Saturday games (late season), Thanksgiving / Christmas specials. Check: `Get-ScheduledTask -TaskName "Footbot-Snapshot-*"`.
**Disable:** `Get-ScheduledTask -TaskName "Footbot-Snapshot-*" | Disable-ScheduledTask` (re-enable with `Enable-ScheduledTask`).
**Remove:** `Get-ScheduledTask -TaskName "Footbot-Snapshot-*" | Unregister-ScheduledTask -Confirm:$false`.
