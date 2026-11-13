r"""
One-shot, read-only snapshot of EXECUTABLE pre-game NFL moneyline prices (Kalshi + Polymarket public order books).

    .venv\Scripts\python.exe -m Backtest.SnapshotPrices --season 2026 --week 10 [--label T-24h] [--out-dir data\prices]

Appends one row per venue x market side to  <out-dir>/snapshots_<season>.jsonl  (append-only, never rewritten) and
(--week auto picks the week of the next unstarted game; scheduled runs use Backtest.SnapshotRunner)
and writes a per-run summary  <out-dir>/snapshot_<season>_w<week>_<UTC stamp>.md/.json.

Row fields: run_id, capture_label, fetched_at (UTC, this process's clock), season, week, game_id, away, home,
kickoff_utc, minutes_to_kickoff, venue, market_id, side_team, best_bid, bid_size, best_ask, ask_size, mid, spread,
bids_top5, asks_top5, last_trade (indicative only), volume, source_time (venue's own quote timestamp or null),
source_time_field, status, posted_ml_devig (nflverse posted moneyline, de-vigged, for comparison only), fee_basis.

Sources (public, no key, no account; same endpoints as Backtest.AssessLive):
  Kalshi   GET /markets?series_ticker=KXNFLGAME&status=open, GET /markets/{ticker}/orderbook
           The Kalshi orderbook has NO quote timestamp: source_time is null and only fetched_at is known.
  Polymarket  Gamma GET /markets?slug=nfl-{away}-{home}-{UTC kickoff date}, CLOB GET /book?token_id=
           (CLOB `timestamp` is used as source_time).
No orders, no auth, no scheduling: this runs once per call. A recurring capture needs Tim's OK.
"""

import argparse
import json
import os
import urllib.parse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from Backtest.AssessLive import (CLOB, GAMMA, KALSHI, FetchError, KalshiAsksFromNoBids, SlugPiece, UrlFetcher, _poly_quotes, _q)
from Backtest.Lab import LoadSeason
from Backtest.LiveReadiness import BookSide, Iso, ParseTime
from Backtest.Odds import FairProbs
from Backtest.Version import ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
FEE_BASIS = "tim_est_0.02_per_contract (Tim's working estimate; venue fee not verified)"
ALIASES = {"LA": ["LA", "LAR"], "JAX": ["JAX", "JAC"], "WAS": ["WAS", "WSH"], "LV": ["LV", "LVR"], "LAC": ["LAC"],
           "NO": ["NO", "NOS"], "GB": ["GB", "GNB"], "KC": ["KC", "KCC"], "NE": ["NE", "NWE"], "SF": ["SF", "SFO"], "TB": ["TB", "TAM"]}


def Kickoff(g):
    return datetime.fromisoformat(f"{g['gameday']}T{g.get('gametime') or '13:00'}").replace(tzinfo=ET).astimezone(timezone.utc)


def KalshiEventMatch(event_ticker, g):
    """KXNFLGAME-26NOV15NEDET -> date 26NOV15 (ET game date) + away code + home code."""
    try:
        tail = event_ticker.split("-", 1)[1]
    except IndexError:
        return False
    day = datetime.fromisoformat(str(g["gameday"]))
    stamp = day.strftime("%y%b%d").upper()
    if not tail.startswith(stamp):
        return False
    codes = tail[len(stamp):]
    return any(codes == a + h for a in ALIASES.get(g["away_team"], [g["away_team"]]) for h in ALIASES.get(g["home_team"], [g["home_team"]]))


def SideTeam(market_ticker, g):
    code = market_ticker.rsplit("-", 1)[-1]
    for team in (g["away_team"], g["home_team"]):
        if code in ALIASES.get(team, [team]):
            return team
    return code


def Row(base, venue, market_id, side, bids, asks, last, volume, src_time, src_field, status, fetched):
    bb, ba = (bids or {}).get("best"), (asks or {}).get("best")
    bid, ask = (bb or {}).get("price"), (ba or {}).get("price")
    return {**base, "venue": venue, "market_id": market_id, "side_team": side,
            "best_bid": bid, "bid_size": (bb or {}).get("size"), "best_ask": ask, "ask_size": (ba or {}).get("size"),
            "mid": round((bid + ask) / 2, 4) if bid is not None and ask is not None else None,
            "spread": round(ask - bid, 4) if bid is not None and ask is not None else None,
            "bids_top5": (bids or {}).get("top"), "asks_top5": (asks or {}).get("top"),
            "last_trade": last, "volume": volume,
            "source_time": Iso(src_time) if src_time else None, "source_time_field": src_field,
            "fetched_at": Iso(fetched), "status": status, "fee_basis": FEE_BASIS}


def Snapshot(season, week, label="adhoc", fetcher=None, now=None):
    fetcher = fetcher or UrlFetcher()
    now = now or datetime.now(timezone.utc)
    run_id = NewRunId()
    games = [g for g in LoadSeason(season) if int(g["week"]) == int(week)]
    errors, rows, skipped = [], [], []
    upcoming = []
    for g in games:
        ko = Kickoff(g)
        if ko <= now:
            skipped.append({"game_id": g["game_id"], "reason": "started_or_final", "kickoff_utc": Iso(ko)})
        else:
            upcoming.append((g, ko))

    kalshi_markets, cursor = [], ""
    try:
        for _ in range(10):
            p = {"series_ticker": "KXNFLGAME", "status": "open", "limit": 200}
            if cursor:
                p["cursor"] = cursor
            body = fetcher.get_json(_q(KALSHI + "/markets", p))["body"]
            kalshi_markets += body.get("markets") or []
            cursor = body.get("cursor") or ""
            if not cursor:
                break
    except FetchError as exc:
        errors.append({"venue": "kalshi", "op": "markets", "error": str(exc)})

    for g, ko in upcoming:
        p_home = None
        if g.get("home_moneyline") and g.get("away_moneyline"):
            p_home = FairProbs(g["home_moneyline"], g["away_moneyline"])[0]
        base = {"run_id": run_id, "model_version": ModelVersion(), "capture_label": label, "season": season, "week": week,
                "game_id": g["game_id"], "away": g["away_team"], "home": g["home_team"], "kickoff_utc": Iso(ko),
                "minutes_to_kickoff": round((ko - now).total_seconds() / 60, 1)}
        found = False
        for m in kalshi_markets:
            if not KalshiEventMatch(m.get("event_ticker") or "", g):
                continue
            found = True
            side = SideTeam(m["ticker"], g)
            try:
                env = fetcher.get_json(f"{KALSHI}/markets/{urllib.parse.quote(m['ticker'])}/orderbook?depth=10")
                book, fetched = (env["body"] or {}).get("orderbook_fp") or {}, env["observed_at"]
            except FetchError as exc:
                errors.append({"venue": "kalshi", "op": "orderbook", "market": m["ticker"], "error": str(exc)})
                continue
            bids = BookSide(book.get("yes_dollars"), "bid")
            asks = KalshiAsksFromNoBids(BookSide(book.get("no_dollars"), "bid"))
            r = Row(base, "kalshi", m["ticker"], side, bids, asks, m.get("last_price_dollars"), m.get("volume_fp"),
                    None, None, m.get("status"), fetched)
            r["posted_ml_devig"] = None if p_home is None else round(p_home if side == g["home_team"] else 1 - p_home, 4)
            rows.append(r)
        if not found:
            errors.append({"venue": "kalshi", "op": "match", "game_id": g["game_id"], "error": "no open KXNFLGAME event matched"})

        slug = f"nfl-{SlugPiece(g['away_team'])}-{SlugPiece(g['home_team'])}-{ko.date().isoformat()}"
        try:
            env = fetcher.get_json(_q(GAMMA + "/markets", {"slug": slug}))
            mk = [x for x in (env["body"] if isinstance(env["body"], list) else []) if (x.get("sportsMarketType") or "").lower() == "moneyline"]
        except FetchError as exc:
            errors.append({"venue": "polymarket", "op": "markets", "slug": slug, "error": str(exc)})
            mk = []
        if not mk:
            errors.append({"venue": "polymarket", "op": "match", "slug": slug, "error": "no moneyline market"})
            continue
        quotes, _fee = _poly_quotes(fetcher, mk[0], g["away_team"], g["home_team"], slug, errors)
        for q in quotes:
            side = q.get("side_team")
            r = Row(base, "polymarket", q.get("token_id"), side, q["book"]["bids"], q["book"]["asks"], q.get("last_trade"),
                    mk[0].get("volumeNum"), q.get("source_timestamp"), q.get("source_timestamp_field"), q.get("status"), q.get("observed_at"))
            r["posted_ml_devig"] = None if p_home is None or side is None else round(p_home if side == g["home_team"] else 1 - p_home, 4)
            rows.append(r)
    return {"run_id": run_id, "season": season, "week": week, "label": label, "started_at": Iso(now),
            "games_in_week": len(games), "upcoming": len(upcoming), "skipped": skipped, "rows": rows, "errors": errors,
            "http_calls": fetcher.calls}


def SummaryMarkdown(out):
    lines = [f"# Executable price snapshot: {out['season']} week {out['week']} ({out['label']})", "",
             f"run_id `{out['run_id']}`, started {out['started_at']} (UTC). Rows {len(out['rows'])}, upcoming games "
             f"{out['upcoming']}/{out['games_in_week']}, HTTP calls {out['http_calls']}, errors {len(out['errors'])}.", "",
             "Prices are dollars per $1 contract. Kalshi books carry no quote timestamp (source_time blank). "
             "Fee basis: Tim's $0.02/contract estimate, not a verified venue fee. Read-only: no orders.", "",
             "| game | kickoff ET | venue | side | bid (size) | ask (size) | spread | posted ML de-vig |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(out["rows"], key=lambda r: (r["kickoff_utc"], r["game_id"], r["venue"], r["side_team"] or "")):
        ko = ParseTime(r["kickoff_utc"]).astimezone(ET).strftime("%a %m-%d %I:%M %p")
        f = lambda p, s: "-" if p is None else f"{p:.2f} ({s:,.0f})"
        lines.append(f"| {r['away']}@{r['home']} | {ko} | {r['venue']} | {r['side_team']} | {f(r['best_bid'], r['bid_size'] or 0)} | "
                     f"{f(r['best_ask'], r['ask_size'] or 0)} | {'-' if r['spread'] is None else r['spread']} | {r.get('posted_ml_devig')} |")
    if out["skipped"]:
        lines += ["", "Skipped (already started/final): " + ", ".join(s["game_id"] for s in out["skipped"])]
    if out["errors"]:
        lines += ["", "Errors:"] + [f"- {json.dumps(e)}" for e in out["errors"]]
    return "\n".join(lines) + "\n"


def AutoSeasonWeek(now=None, season=None):
    """Season = this year from March on, else last year. Week = the week of the next game not yet kicked off."""
    now = now or datetime.now(timezone.utc)
    season = season or (now.year if now.month >= 3 else now.year - 1)
    upcoming = [g for g in LoadSeason(season) if Kickoff(g) > now]
    if not upcoming:
        return season, None
    return season, min(int(g["week"]) for g in upcoming)


def WriteOutputs(out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"snapshots_{out['season']}.jsonl"), "a", encoding="utf-8") as fh:
        for r in out["rows"]:
            fh.write(json.dumps(r, default=str) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(out_dir, f"snapshot_{out['season']}_w{out['week']}_{stamp}")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(SummaryMarkdown(out))
    return base


def Main():
    ap = argparse.ArgumentParser(description="One-shot executable NFL moneyline snapshot (read-only).")
    ap.add_argument("--season", default="auto")
    ap.add_argument("--week", default="auto", help="week number or 'auto' (week of the next unstarted game)")
    ap.add_argument("--label", default="adhoc")
    ap.add_argument("--out-dir", default=os.path.join("data", "prices"))
    a = ap.parse_args()
    season = None if a.season == "auto" else int(a.season)
    if a.week == "auto":
        season, week = AutoSeasonWeek(season=season)
        if week is None:
            print("no upcoming games for season", season)
            return
    else:
        season, week = season or AutoSeasonWeek()[0], int(a.week)
    out = Snapshot(season, week, a.label)
    base = WriteOutputs(out, a.out_dir)
    print(SummaryMarkdown(out))
    print("wrote", base + ".md")


if __name__ == "__main__":
    Main()