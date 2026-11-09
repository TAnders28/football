r"""
Shared live file for Degen: one JSONL row per market per run, plus a short "latest" table.

    <live_root>\<YYYY-MM-DD>_live\<YYYY-MM-DD>_live.jsonl       APPEND-ONLY, one row per market per run
    <live_root>\<YYYY-MM-DD>_live\<YYYY-MM-DD>_live_latest.md   overwritten each run: table + blockers

live_root defaults to E:\AgentDocs\American-Footbot (override with --live-dir or FOOTBOT_LIVE_DIR;
tests always pass a temp dir).

Row = {observed_at, game_id, teams, score, period, clock, state_time, venue, market, side, bid, ask,
       size, price_time, model_prob|null, edge|null, action: BET|PASS, reason, fee_basis,
       quote_qualified, ...}

Rules (fail closed)
  * price_time / state_time are the SOURCE timestamps (book timestamp, live-data last_updated_ts),
    never our retrieval time. Missing source time = stale.
  * "stale" if either is missing/stale/future, if they are more than STALE_GAP_S (60 s) apart, or if
    any stricter repo limit fails (LiveReadiness: live age 45 s, sync skew 20 s). Nothing is loosened.
  * bid/ask are the executable book's best bid / best ask, never the last trade.
  * quote_qualified = executable bid AND ask exist, the quote is fresh, synchronized with a fresh
    game board, and passes every staleness check above.
  * Without a validated in-play probability the action is PASS ("no live model"). A pre-game
    probability is never used as a live one, and this repo has no in-play model, so the CLI is PASS.
  * If a validated in-play prob is ever supplied: edge = model_prob - ask - 0.02, where 0.02 is
    Tim's fee estimate ($0.02 per $1 contract), not a verified exchange fee.
"""

import json
import os
from datetime import datetime, timezone

from Backtest.LiveReadiness import ET, ParseTime

LIVE_ROOT = os.environ.get("FOOTBOT_LIVE_DIR", r"E:\AgentDocs\American-Footbot")
STALE_GAP_S = 60
FEE_ESTIMATE = 0.02
FEE_BASIS = "tim_estimate_0.02_per_contract"
EDGE_LABEL = "model_prob - ask - 0.02 (Tim's fee estimate, not a verified exchange fee)"
STALE_REASONS = {"quote_missing_timestamp", "quote_stale_timestamp", "quote_future_timestamp",
                 "sync_skew", "sync_unknown"}


def _et_iso(value):
    dt = ParseTime(value) if not isinstance(value, datetime) else value
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).isoformat(timespec="seconds")


def _board(game):
    """The board shown for the game: the synchronized one if any, else the first one (flagged stale)."""
    states = game.get("states") or []
    venue = (game.get("sync") or {}).get("accepted_state_source")
    accepted = next((s for s in states if venue and s.get("venue") == venue and s.get("usable")), None)
    return accepted, (accepted or (states[0] if states else None))


def GameId(game, date):
    return f"{date}_{game.get('away') or '?'}@{game.get('home') or '?'}"


def QuoteRow(game, quote, date, observed_at, run_id=None, model_version=None):
    accepted, board = _board(game)
    bid, ask = quote.get("bid") or None, quote.get("ask") or None
    price_ts = ParseTime(quote.get("source_timestamp"))
    state_ts = ParseTime(board.get("source_timestamp")) if board else None
    q_reasons = set(quote.get("fail_reasons") or [])
    stale_why = sorted(q_reasons & STALE_REASONS)
    if price_ts is None:
        stale_why.append("no price source time")
    if state_ts is None:
        stale_why.append("no state source time")
    if accepted is None:
        stale_why.append("no fresh synchronized game board")
    gap = None
    if price_ts is not None and state_ts is not None:
        gap = abs((price_ts - state_ts).total_seconds())
        if gap > STALE_GAP_S:
            stale_why.append(f"price/state {gap:.0f}s apart > {STALE_GAP_S}s")
    stale = bool(stale_why)
    has_book = bool(bid and ask and bid.get("price") is not None and ask.get("price") is not None
                    and (ask.get("size") or 0) > 0)
    winner_market = quote.get("outcome_kind") == "moneyline_game_winner"
    quote_qualified = bool(has_book and quote.get("executable_ask") and not stale and winner_market
                           and "state_disagreement" not in q_reasons and "event_mismatch" not in q_reasons)
    model_prob = quote.get("probability")            # only ever set by a validated in-play model
    edge = None
    if model_prob is not None and quote_qualified:
        edge = round(model_prob - ask["price"] - FEE_ESTIMATE, 4)
    phase = game.get("phase")
    if not winner_market:
        reason = "not a game-winner market"
    elif phase == "final":
        reason = "final"
    elif stale:
        reason = "stale"
    elif {"closed", "suspended"} & q_reasons:
        reason = "market " + ("closed" if "closed" in q_reasons else "suspended")
    elif not has_book or not quote.get("executable_ask"):
        reason = "no executable bid/ask"
    elif model_prob is None:
        reason = "no live model"
    elif edge is None or edge <= 0:
        reason = "edge <= 0 after Tim's fee estimate"
    else:
        reason = "edge > 0 after Tim's fee estimate"
    action = "BET" if reason == "edge > 0 after Tim's fee estimate" else "PASS"
    score = None
    if board and board.get("away_score") is not None and board.get("home_score") is not None:
        score = f"{board['away_score']}-{board['home_score']}"
    return {
        "observed_at": _et_iso(observed_at),
        "game_id": GameId(game, date),
        "teams": f"{game.get('away') or '?'}@{game.get('home') or '?'}",
        "score": score,
        "period": None if not board else board.get("period"),
        "clock": None if not board else board.get("clock"),
        "state_time": _et_iso(state_ts),
        "state_venue": None if not board else board.get("venue"),
        "phase": phase,
        "venue": quote.get("venue"),
        "market": quote.get("market_id"),
        "side": quote.get("side_team") or quote.get("side_name"),
        "bid": None if not bid else bid.get("price"),
        "ask": None if not ask else ask.get("price"),
        "size": None if not ask else ask.get("size"),
        "bid_size": None if not bid else bid.get("size"),
        "price_time": _et_iso(price_ts),
        "price_state_gap_s": None if gap is None else round(gap, 3),
        "model_prob": model_prob,
        "edge": edge,
        "edge_basis": EDGE_LABEL if edge is not None else None,
        "action": action,
        "reason": reason,
        "stale_detail": stale_why,
        "fail_reasons": sorted(q_reasons),
        "fee_basis": FEE_BASIS,
        "quote_qualified": quote_qualified,
        "run_id": run_id,
        "model_version": model_version,
    }


def LiveRows(report, run_id=None, model_version=None):
    meta = report.get("meta") or {}
    date = meta.get("date_et")
    observed = meta.get("generated_at_utc")
    rows = []
    for game in report.get("games") or []:
        for quote in game.get("quotes") or []:
            rows.append(QuoteRow(game, quote, date, quote.get("observed_at_utc") or observed, run_id, model_version))
    return rows


def LatestMarkdown(report, rows, run_id=None, model_version=None):
    meta = report.get("meta") or {}
    date = meta.get("date_et")
    L = [f"# NFL live {date} (latest run)", "",
         f"Run {run_id} at {meta.get('generated_at_et')} | model_version `{model_version}` | read-only, no orders.",
         f"Prices are executable book bid/ask; times are source timestamps (ET). Edge would use {EDGE_LABEL}.", "",
         "| Game | Phase | Score | Q / clock | State time | Venue | Side | Bid / Ask | Size | Price time | Qualified | Action | Reason |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if not rows:
        L.append("| - | - | - | - | - | - | - | - | - | - | - | PASS | no markets found |")
    for r in rows:
        qc = " ".join(str(x) for x in (r["period"], r["clock"]) if x not in (None, "")) or "-"
        L.append(f"| {r['teams']} | {r['phase'] or '-'} | {r['score'] or '-'} | {qc} | {(r['state_time'] or '-')[11:19] or '-'} | "
                 f"{r['venue']} | {r['side'] or '-'} | {r['bid'] if r['bid'] is not None else '-'} / {r['ask'] if r['ask'] is not None else '-'} | "
                 f"{r['size'] if r['size'] is not None else '-'} | {(r['price_time'] or '-')[11:19] or '-'} | "
                 f"{'yes' if r['quote_qualified'] else 'no'} | **{r['action']}** | {r['reason']} |")
    blockers = ["No validated in-play (game-state) model in this repo: every market is PASS (\"no live model\").",
                f"Fees: exchange fees unverified; {FEE_BASIS} is Tim's working estimate."]
    counts = {}
    for r in rows:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    if counts:
        blockers.append("Reasons this run: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) + ".")
    stale_detail = sorted({d for r in rows for d in r["stale_detail"]})
    if stale_detail:
        blockers.append("Staleness detail: " + "; ".join(stale_detail) + ".")
    for err in (meta.get("source_errors") or [])[:8]:
        blockers.append(f"Source error {err.get('source')} {err.get('op')}: {err.get('error')}")
    L += ["", "## Blockers", ""] + [f"- {b}" for b in blockers] + [""]
    return "\n".join(L)


def WriteLiveFile(report, live_root=None, run_id=None, model_version=None):
    """Append one row per market to <date>_live.jsonl and overwrite <date>_live_latest.md.
    Returns {"jsonl", "latest", "rows"}."""
    live_root = live_root or LIVE_ROOT
    date = (report.get("meta") or {}).get("date_et")
    folder = os.path.join(live_root, f"{date}_live")
    os.makedirs(folder, exist_ok=True)
    rows = LiveRows(report, run_id, model_version)
    jsonl = os.path.join(folder, f"{date}_live.jsonl")
    with open(jsonl, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    latest = os.path.join(folder, f"{date}_live_latest.md")
    with open(latest, "w", encoding="utf-8") as fh:
        fh.write(LatestMarkdown(report, rows, run_id, model_version))
    return {"jsonl": jsonl, "latest": latest, "rows": len(rows)}
