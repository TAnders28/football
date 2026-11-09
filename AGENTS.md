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
| `Backtest/AssessLive.py`, `LiveReadiness.py`, `LiveFile.py` | one-shot read-only live odds/game state + Degen's live file |
| `PlayByPlay/` | play-by-play reconstruction/simulation. Kept separate from `Backtest/` |
| `tests/` | `python -m unittest discover -s tests` (no network, no `data/` needed) |
| `docs/` | write-ups (`live-odds.md`) |
| `data/`, `stats/` | gitignored raw pulls and outputs. Never commit. `examples/` is Tim's untracked folder: leave it alone |

## Entry points (Windows, from `E:\Football\football`)

```
.venv\Scripts\python.exe -m Backtest.PredictDay --date YYYY-MM-DD [--week] [--next] [--no-refresh]
.venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026 [--no-refresh]
.venv\Scripts\python.exe -m Backtest.AssessLive --date YYYY-MM-DD [--matchup TB@CHI] [--live-dir DIR] [--no-live-file]
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
