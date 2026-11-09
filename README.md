# football

NFL predictive modelling and backtesting.

## Layout

- `data/` — raw pulls (gitignored)
- `stats/` — derived season / play-by-play aggregates (gitignored)
- `PlayByPlay/` — simulation and evaluation
- `Backtest/` — strategy search and betting backtests
- `python -m Backtest.AssessLive` — one-shot read-only NFL quote and game-state report (no orders). See `docs/live-odds.md`.
- `docs/` — notes and track writeups
