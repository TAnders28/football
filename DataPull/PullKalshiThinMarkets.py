r"""
Download Kalshi NFL totals and spreads 1-minute candles around kickoff.

    .venv\Scripts\python.exe -m DataPull.PullKalshiThinMarkets --seasons 2025,2026

Public API, no auth. Forces IPv4 (Tomputer IPv6 to Cloudflare can stall).
Sets BelowNormal on Windows. Resumable: tickers already in a candles file are skipped.
A successful empty candle list is written (so it is not retried); HTTP errors are
written with an "error" field and also skipped on resume unless --retry-errors.

Writes (gitignored under data/):
  data/kalshi/thin_fees.json
  data/kalshi/thin_markets.jsonl
  data/kalshi/thin_candles_<season>.jsonl
  data/kalshi/thin_pull_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

from Backtest.Lab import LoadSeason
from Backtest.MarketBlend import ClassifySeries, ForceIPv4, KalshiApi, SetBelowNormal, SURVEY_CANDIDATES
from Backtest.SnapshotPrices import KalshiEventMatch, Kickoff
from DataPull.PullKalshiHistory import NormCandle

OUT = os.path.join("data", "kalshi")
UA_NOTE = "football-PullKalshiThinMarkets/readonly"
KINDS = {"total", "spread"}
# Full-game pregame markets only. Quarter/half/team-total series are listed in the
# Job A survey but are out of scope for this pull (too many markets).
PRIMARY_SERIES = ("KXNFLSPREAD", "KXNFLTOTAL")
FEE_RE = re.compile(r"fee", re.I)


def FeeFields(obj):
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        if FEE_RE.search(str(k)):
            out[k] = v
        elif isinstance(v, dict):
            sub = FeeFields(v)
            if sub:
                out[k] = sub
    return out


def SeriesBlock(api, ticker):
    body = api.Get(f"/series/{ticker}") or {}
    series = body.get("series") or body
    fees = api.Get("/series/fee_changes", {"series_ticker": ticker, "show_historical": "true"})
    raw = series if isinstance(series, dict) else {}
    title = raw.get("title") or raw.get("name") or ""
    return {
        "ticker": ticker,
        "title": title,
        "category": raw.get("category"),
        "frequency": raw.get("frequency"),
        "market_kind": ClassifySeries(ticker, title),
        "fee_type": raw.get("fee_type"),
        "fee_multiplier": raw.get("fee_multiplier"),
        "fee_fields": FeeFields(raw),
        "fee_changes": fees,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def DiscoverThinSeries(api, candidates=SURVEY_CANDIDATES):
    found = {}
    listed = api.Get("/series", {"category": "Sports", "limit": 200}) or api.Get("/series", {"limit": 200}) or {}
    series_list = listed.get("series") or listed.get("series_list") or []
    if isinstance(listed, list):
        series_list = listed
    for s in series_list:
        ticker = (s.get("ticker") or s.get("series_ticker") or "").upper()
        title = s.get("title") or s.get("name") or ""
        kind = ClassifySeries(ticker, title)
        if kind in KINDS and ("NFL" in ticker or "NFL" in title.upper() or ticker.startswith("KXNFL")):
            found[ticker] = {"ticker": ticker, "title": title, "source": "list", "market_kind": kind, "raw": s}
    for t in candidates:
        if t in found:
            continue
        body = api.Get(f"/series/{t}") or {}
        if not body:
            continue
        s = body.get("series") or body
        if not isinstance(s, dict):
            continue
        ticker = (s.get("ticker") or t).upper()
        title = s.get("title") or s.get("name") or ""
        kind = ClassifySeries(ticker, title)
        if kind in KINDS:
            found[ticker] = {"ticker": ticker, "title": title, "source": "probe", "market_kind": kind, "raw": s}
    return found


def CollectMarkets(api, series_ticker):
    hist = api.Paged("/historical/markets", {"series_ticker": series_ticker, "limit": 1000}, key="markets", max_pages=50)
    live = []
    # /markets status is settled|closed|open (finalized is a 400 on this endpoint).
    for status in ("settled", "closed"):
        try:
            live += api.Paged("/markets", {"series_ticker": series_ticker, "status": status, "limit": 1000}, key="markets", max_pages=50)
        except Exception as exc:
            print(f"  /markets status={status} failed: {type(exc).__name__}: {exc}", flush=True)
    tier = {m["ticker"]: "historical" for m in hist if m.get("ticker")}
    for m in live:
        if m.get("ticker"):
            tier.setdefault(m["ticker"], "live")
    markets = {}
    for m in hist + live:
        t = m.get("ticker")
        if t:
            markets[t] = m
    return markets, tier


def ParseStrike(ticker, market=None):
    """Best-effort line / side from the ticker tail or market strike fields."""
    mkt = market or {}
    for k in ("custom_strike", "floor_strike", "cap_strike", "strike"):
        v = mkt.get(k)
        if v is not None and v != "":
            try:
                return float(str(v).replace("+", "")), (mkt.get("yes_sub_title") or mkt.get("subtitle") or k)
            except ValueError:
                pass
    title = " ".join(str(mkt.get(k) or "") for k in ("yes_sub_title", "subtitle", "title", "yes_title"))
    tm = re.search(r"([OU]|OVER|UNDER|O/U)?\s*([+-]?\d+(?:\.\d+)?)", title.upper())
    tail = (ticker or "").split("-")[-1]
    mm = re.search(r"([A-Z]{2,4})?([+-]?\d+(?:\.\d+)?)$", tail.replace("T", "") if tail.startswith("T") else tail)
    if tail.upper().startswith("T"):
        try:
            return float(tail[1:]), "total"
        except ValueError:
            pass
    if mm:
        try:
            return float(mm.group(2)), (mm.group(1) or title or tail)
        except (TypeError, ValueError):
            pass
    if tm:
        try:
            return float(tm.group(2)), (tm.group(1) or title)
        except (TypeError, ValueError):
            pass
    return None, title or tail


def MatchGame(event_ticker, games):
    ev = event_ticker or ""
    for g in games:
        if KalshiEventMatch(ev, g):
            return g
    return None


def LoadGames(seasons):
    out = []
    for s in seasons:
        for g in LoadSeason(s):
            if g.get("gameday"):
                out.append(g)
    return out


def DoneTickers(path, retry_errors=False):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            t = r.get("ticker")
            if not t:
                continue
            if retry_errors and r.get("error"):
                continue
            done[t] = r
    return done


def FetchCandles(api, series_ticker, ticker, tier, ko, before_min, after_min):
    params = {
        "start_ts": int((ko - timedelta(minutes=before_min)).timestamp()),
        "end_ts": int((ko + timedelta(minutes=after_min)).timestamp()),
        "period_interval": 1,
    }
    endpoints = []
    if tier == "historical":
        endpoints.append(f"/historical/markets/{ticker}/candlesticks")
        endpoints.append(f"/series/{series_ticker}/markets/{ticker}/candlesticks")
    else:
        endpoints.append(f"/series/{series_ticker}/markets/{ticker}/candlesticks")
        endpoints.append(f"/historical/markets/{ticker}/candlesticks")
    last_err = None
    for ep in endpoints:
        try:
            body = api.Get(ep, params)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        if not body:
            last_err = last_err or f"empty/404 at {ep}"
            continue
        cs = [NormCandle(c) for c in body.get("candlesticks") or []]
        return cs, ep, None
    return [], endpoints[0], last_err


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Pull Kalshi NFL totals/spreads candles (public API).")
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--series", default=",".join(PRIMARY_SERIES),
                    help="comma-separated series tickers (default: full-game spread and total)")
    ap.add_argument("--before-min", type=int, default=180)
    ap.add_argument("--after-min", type=int, default=5)
    ap.add_argument("--pause", type=float, default=0.12)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on new candle pulls (tests / smoke).")
    a = ap.parse_args(argv)
    SetBelowNormal()
    ForceIPv4()
    os.makedirs(OUT, exist_ok=True)
    seasons = [int(s) for s in a.seasons.split(",")]
    api = KalshiApi(pause=a.pause)
    # Identify the client in the UA if we go through a raw request; KalshiApi uses SURVEY_UA.
    wanted = [s.strip().upper() for s in a.series.split(",") if s.strip()]
    series_map = DiscoverThinSeries(api, candidates=tuple(wanted) or SURVEY_CANDIDATES)
    if wanted:
        series_map = {k: v for k, v in series_map.items() if k in set(wanted)}
        for tkr in wanted:
            if tkr in series_map:
                continue
            body = api.Get(f"/series/{tkr}") or {}
            s = body.get("series") or body
            if isinstance(s, dict) and (s.get("ticker") or body):
                title = s.get("title") or s.get("name") or ""
                series_map[tkr] = {"ticker": tkr, "title": title, "source": "flag",
                                   "market_kind": ClassifySeries(tkr, title), "raw": s}
            else:
                print(f"series {tkr} not found", flush=True)
    games = LoadGames(seasons)
    fees, markets_out = {}, []
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons, "series": {}, "matched_games": {},
        "candle_rows": {}, "empty_candles": {}, "errors": {}, "skipped_done": {},
        "http_calls": 0, "http_retries": 0, "unmatched_events": [],
    }
    unmatched = set()
    pulled = 0
    for ticker, info in sorted(series_map.items()):
        print(f"series {ticker} kind={info.get('market_kind')} source={info.get('source')}", flush=True)
        fees[ticker] = SeriesBlock(api, ticker)
        markets, tier = CollectMarkets(api, ticker)
        by_event = {}
        n_matched = 0
        for m in markets.values():
            ev = m.get("event_ticker") or ""
            if ev not in by_event:
                by_event[ev] = MatchGame(ev, games)
            g = by_event[ev]
            if g is None:
                unmatched.add(ev)
            else:
                n_matched += 1
            strike, side_lab = ParseStrike(m.get("ticker") or "", m)
            markets_out.append({
                "ticker": m.get("ticker"), "event_ticker": ev, "series_ticker": ticker,
                "market_kind": info.get("market_kind") or fees[ticker]["market_kind"],
                "tier": tier.get(m.get("ticker") or "", ""),
                "status": m.get("status"), "result": m.get("result"),
                "close_time": m.get("close_time") or m.get("expiration_time"),
                "volume": m.get("volume_fp") or m.get("volume"),
                "open_interest": m.get("open_interest_fp") or m.get("open_interest"),
                "title": m.get("title") or m.get("yes_sub_title") or m.get("subtitle") or "",
                "yes_sub_title": m.get("yes_sub_title") or m.get("subtitle") or "",
                "strike": strike, "strike_label": side_lab,
                "game_id": g["game_id"] if g else None,
                "season": int(g["season"]) if g else None,
                "week": g.get("week") if g else None,
                "home_team": g["home_team"] if g else None,
                "away_team": g["away_team"] if g else None,
                "gameday": g["gameday"] if g else None,
                "fee_fields": FeeFields(m),
            })
        summary["series"][ticker] = {
            "n_markets": len(markets), "n_matched_market_rows": n_matched,
            "fee_type": fees[ticker]["fee_type"], "fee_multiplier": fees[ticker]["fee_multiplier"],
            "market_kind": fees[ticker]["market_kind"],
        }
        for s in seasons:
            path = os.path.join(OUT, f"thin_candles_{s}.jsonl")
            done = DoneTickers(path, retry_errors=a.retry_errors)
            summary["skipped_done"].setdefault(s, 0)
            summary["candle_rows"].setdefault(s, 0)
            summary["empty_candles"].setdefault(s, 0)
            summary["errors"].setdefault(s, 0)
            todo = []
            for m in markets.values():
                t = m.get("ticker")
                ev = m.get("event_ticker") or ""
                g = by_event.get(ev)
                if not t or not g or int(g["season"]) != s:
                    continue
                if t in done:
                    summary["skipped_done"][s] += 1
                    continue
                todo.append((t, m, g, tier.get(t, "live")))
            with open(path, "a", encoding="utf-8") as fh:
                for t, m, g, tr in todo:
                    if a.limit and pulled >= a.limit:
                        break
                    ko = Kickoff(g)
                    cs, ep, err = FetchCandles(api, ticker, t, tr, ko, a.before_min, a.after_min)
                    empty = (not cs) and not err
                    if err:
                        summary["errors"][s] += 1
                    if empty:
                        summary["empty_candles"][s] += 1
                    fh.write(json.dumps({
                        "ticker": t, "event_ticker": m.get("event_ticker"),
                        "series_ticker": ticker, "market_kind": info.get("market_kind"),
                        "game_id": g["game_id"], "season": s, "week": g["week"],
                        "game_type": g.get("game_type"), "home_team": g["home_team"],
                        "away_team": g["away_team"],
                        "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "result": m.get("result"), "endpoint": ep, "period_minutes": 1,
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "n": len(cs), "candles": cs, "error": err,
                    }) + "\n")
                    pulled += 1
                    summary["candle_rows"][s] += 1
                    if pulled % 25 == 0:
                        print(f"  pulled {pulled} new candle rows, calls={api.calls} retries={api.retries}", flush=True)
            print(f"  {s} todo_new={len(todo)} written_this_series_season done_skip={summary['skipped_done'][s]} "
                  f"empty={summary['empty_candles'][s]} errors={summary['errors'][s]}", flush=True)
        summary["matched_games"][ticker] = len({g["game_id"] for g in by_event.values() if g})
    summary["http_calls"], summary["http_retries"] = api.calls, api.retries
    summary["unmatched_events"] = sorted(unmatched)[:400]
    summary["n_unmatched_events"] = len(unmatched)
    summary["n_markets_rows"] = len(markets_out)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(OUT, "thin_fees.json"), "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": summary["finished_at"], "series": fees}, fh, indent=1)
    with open(os.path.join(OUT, "thin_markets.jsonl"), "w", encoding="utf-8") as fh:
        for row in markets_out:
            fh.write(json.dumps(row) + "\n")
    with open(os.path.join(OUT, "thin_pull_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    brief = {k: v for k, v in summary.items() if k != "unmatched_events"}
    print(json.dumps(brief, indent=1, default=str), flush=True)
    return summary


if __name__ == "__main__":
    Main()
