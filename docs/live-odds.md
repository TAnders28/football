# Live-odds readiness

`python -m Backtest.AssessLive` is a one-shot, read-only report. It writes
`data/backtest/live/assess_<ET date>_<UTC stamp>.json` and `.md`. `data/` is
gitignored. The command does not place orders, store credentials, or schedule
itself.

```
python -m Backtest.AssessLive
python -m Backtest.AssessLive --date 2026-11-08
python -m Backtest.AssessLive --date 2026-11-08 --matchup TB@CHI
```

## What it reads

Kalshi's public Trade API (`https://external-api.kalshi.com/trade-api/v2`, no key),
as documented in the market-data quick start and the milestone / live-data references:

- `GET /milestones` with `type=football_game` (the `competition=Pro Football` filter is also called and its count is recorded; it is not required for the slate)
- `GET /markets`, `GET /markets/{ticker}/orderbook`, `GET /series/{ticker}`
- `GET /structured_targets/{id}`, `GET /live_data/batch`
- `GET /series/fee_changes`, `GET /events/fee_changes`
- the series `contract_terms_url` PDF

Polymarket's public Gamma and CLOB reads, for the moneyline only:

- `GET /sports` (NFL `ordering`, used to read `score`)
- `GET /markets?slug=nfl-{away}-{home}-{UTC date of kickoff}` — a lookup candidate in the shape returned by Gamma, not a guessed price; a miss is recorded
- `GET /book?token_id=` (best bid is the highest bid, best ask is the lowest ask)
- `GET /clob-markets/{condition_id}` for the fee fields the API actually returns

The Polymarket sports websocket is not held open. `api.nfl.com` is not called.

## What it will not do

`Backtest.Model.MarginModel` and the market-anchored full2/full3 models are pre-game.
They train on historical seasons and only use games finished on earlier days.
`AssessLive` does not import them and does not compare their probabilities to
these quotes. There is no validated game-state-adjusted model in this repo, so
the report's action is **PASS** and probability / EV stay blank.

Gross and net EV are filled only when a caller passes a validated
`game_state_adjusted` probability (including a tie probability and a zero
probability on non-binary paths), the ask is executable, the taker fee per
contract is a known number, and settlement is closed-form. The CLI never passes
that model. Kalshi's football contract pays $1 / $0.50 / $0 for win / tie / loss
and pays a discretionary last fair price on several cancellation paths, so the
parsed rules are quoted and `closed_form` stays false.

## Final boards

Polymarket documents `FT` as full time, not a quarter that is still being played.
Finished Gamma NFL events return `VFT`. That token is treated as the same kind of
ending as `FT`. A Kalshi board that is already final still shows the last quarter
played (`4` in regulation, `5` in overtime). Those agree when the score agrees.
A live quarter against `VFT`, two different live quarters, or a different score
remains a disagreement.

## Freshness

Source timestamps and the local observed time are stored separately. UTC is
stored; America/New_York is displayed. A successful HTTP call does not make an
old source timestamp current.

Tool policy (not a provider guarantee): a live quote or board older than 45
seconds is stale, a timestamp more than 90 seconds ahead of the local clock is
rejected, and quote vs accepted state skew above 20 seconds fails closed.
Kalshi documents that live data may be a cached board and that
`live_data_delayed` is set only once a feed is significantly stale, so a missing
flag is not treated as proof of freshness. Kalshi market `updated_time` is the
last non-trading metadata update and is not a quote time. The order book
payload has no quote timestamp, so those quotes fail `quote_missing_timestamp`.
Polymarket CLOB `timestamp` is used when present. Gamma `updatedAt` is not used
as the score's sample time.

## Fees

Kalshi `fee_type` / `fee_multiplier` come from `GET /series`. Docs point the
quadratic schedules at `https://kalshi.com/docs/kalshi-fee-schedule.pdf`. If that
PDF is not retrieved, or its prose is not reduced to a per-contract dollar
amount, the fee is **unknown**. Polymarket `mbf` / `tbf` / `fd` are recorded and
the dollar fee stays unknown unless a formula is actually in the payload. Unknown
fees blank EV.
