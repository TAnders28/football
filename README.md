# football

NFL predictive modelling and backtesting. Conventions for bots and contributors: see `AGENTS.md`.

## Layout

- `data/` - raw pulls and outputs (gitignored), incl. `data/predictions/<season>.jsonl` (append-only log)
- `stats/` - derived season / play-by-play aggregates (gitignored)
- `PlayByPlay/` - simulation and evaluation
- `Backtest/` - strategy search and betting backtests
- `docs/` - notes and track writeups
- `tests/` - `python -m unittest discover -s tests` (no network)

## Commands

- `python -m Backtest.PredictDay --date YYYY-MM-DD` - same-day picks from the frozen config
  (`Backtest/FrozenConfig.py`); per-run picks files + append-only prediction log; started games are not re-scored.
- `python -m Backtest.GradePredictions --season 2026` - grades the log (pre-kickoff rows only) into `grades_<season>.json`.
- `python -m Backtest.AssessLive` - one-shot read-only NFL quote and game-state report (no orders), plus the shared
  live file `<live-dir>/<date>_live/<date>_live.jsonl` + `_live_latest.md` for Degen. See `docs/live-odds.md`.

Every output carries `model_version` (frozen config hash + git hash) and `run_id`.
