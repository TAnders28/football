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

Tool policy (not a provider guarantee): a quote or board source time older
than 45 seconds is stale, including a finished game whose provider timestamp
is hours before the retrieval. Pregame source times may be up to 15 minutes
old. A timestamp more than 90 seconds ahead of the local clock is rejected,
and quote vs accepted state skew above 20 seconds fails closed. Retrieval time
is stored as `observed_at` and is never copied into `source_timestamp`.
Kalshi documents that live data may be a cached board and that
`live_data_delayed` is set only once a feed is significantly stale, so a missing
flag is not treated as proof of freshness. Kalshi market `updated_time` is the
last non-trading metadata update and is not a quote time. The order book
payload has no quote timestamp, so those quotes fail `quote_missing_timestamp`.
Polymarket CLOB `timestamp` is used when present. Gamma `updatedAt` is not used
as the score's sample time.

## Fees

Kalshi `fee_type` / `fee_multiplier` come from `GET /series`. The series schema
maps `quadratic_with_maker_fees` to the General Trading Fees table plus the
Maker Fees section of `https://kalshi.com/docs/kalshi-fee-schedule.pdf`
(`https://docs.kalshi.com/api-reference/market/get-series`). That PDF returned
HTTP 429 on the 2026-11-08 8:57 PM ET run and is not retried. The numeric rate
and rounding unit were not read from it, so `taker_fee_per_contract` stays null.
Polymarket `GET /clob-markets/{condition_id}` returns `mbf`, `tbf`, and `fd`
(`https://docs.polymarket.com/api-reference/markets/get-clob-market-info`).
Those fields do not state a dollar function of price and size, so that fee is
unknown too. Unknown fees blank EV. No coefficient is hard-coded in their place.

## Settlement

Kalshi football game-winner terms were read from the series
`contract_terms_url`, `https://assets.kalshi.com/contract_terms/FOOTBALLGAMEWIN.pdf`.
The text includes overtime in the full game, pays $0.50 / $0.50 on a tie when
no Tie strike is listed, and pays a last fair price on several postponement,
cancellation, forfeit, and abandonment paths. The product certification is
`https://assets.kalshi.com/regulatory/product-certifications/FOOTBALLGAMEWIN.pdf`.
Because a last fair price is not a fixed payoff, `closed_form` stays false.
Polymarket moneyline text returned on the Gamma market (overtime included, tie
50-50, cancellation 50-50) is quoted the same way and is also not closed-form.
Neither text is turned into a live probability.

## What has no in-play validation

`MarginModel` uses only finished games and a season-to-date margin. `Features`
builds rows with a day barrier. `Lab` fits `p_margin`, `p_full2`, and `p_full3`
on earlier seasons against final `home_win` and the de-vigged closing moneyline
(`mkt_logit`, form, ATS, rest, divisional). `PlayByPlay.ReconstructGame` reshapes
nflverse play-by-play for reading. None of these is a model of the current
period, clock, and score against an executable venue ask, and none was checked
on historical in-play venue prices.

The missing historical set is all three of these, joined on one contract:

- in-play state (period, clock, score, and the status the venue actually settled)
- that contract's executable bid and ask, with the venue's source timestamp
- the settlement outcome under that contract's overtime, tie, and cancellation rules

nflverse schedules and closing moneylines are not that set. Until it exists,
`AssessLive` has no validated in-play probability and does not emit a pick.
