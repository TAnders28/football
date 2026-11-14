r"""
Pull Kalshi NFL player-prop markets and pregame candles (public API, read-only).

    .venv\Scripts\python.exe -m DataPull.PullKalshiProps --series KXNFLTD [--seasons 2025,2026]

Series covered by default: KXNFLTD (anytime TD ladders 1+/2+/3+), KXNFLRECYDS, KXNFLREC,
KXNFLRSHYDS, KXNFLPASSYDS (player X+ ladders). Only markets listed BEFORE kickoff are kept
(Kalshi also lists in-game ladders). Two candle passes per market, both via the batch
endpoint /markets/candlesticks (falls back to per-market for historical-tier tickers):
  - hourly bars from listing to kickoff (open-to-close movement)
  - 1-minute bars kickoff-120min .. kickoff+2min (T-60 / T-15 / close snapshots)
Resumable: tickers already in the candles file are skipped.

Writes (gitignored):
  data/kalshi/props_markets_<series>.jsonl
  data/kalshi/props_candles_<series>.jsonl
  data/kalshi/props_pull_summary_<series>.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from Backtest.MarketBlend import ForceIPv4, KalshiApi, SetBelowNormal
from Backtest.SnapshotPrices import Kickoff
from DataPull.PullKalshiHistory import NormCandle
from DataPull.PullKalshiThinMarkets import CollectMarkets, DoneTickers, LoadGames, MatchGame, SeriesBlock

OUT = os.path.join("data", "kalshi")
DEFAULT_SERIES = ("KXNFLTD", "KXNFLRECYDS", "KXNFLREC", "KXNFLRSHYDS", "KXNFLPASSYDS")
BATCH = 50


def ParseTs(s):
    if not s:
        return None
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def PlayerFromMarket(m, series):
    """'Dalton Kincaid: 10+' -> ('Dalton Kincaid', 10); ticker seg 'BUFDKINCAID86' -> team guess."""
    sub = m.get("yes_sub_title") or m.get("no_sub_title") or ""
    name, _, thr = sub.partition(":")
    seg = (m.get("ticker") or "").split("-")
    pseg = seg[2] if len(seg) > 2 else ""
    thr = thr.strip().rstrip("+")
    try:
        thr_v = float(thr)
    except ValueError:
        thr_v = None
    return name.strip(), pseg, thr_v


def Batched(api, tickers, start_ts, end_ts, period):
    body = api.Get("/markets/candlesticks", {"market_tickers": ",".join(tickers), "start_ts": int(start_ts),
                                             "end_ts": int(end_ts), "period_interval": period}) or {}
    out = {}
    for row in body.get("markets") or []:
        out[row.get("market_ticker")] = [NormCandle(c) for c in row.get("candlesticks") or []]
    return out


def Single(api, series, ticker, start_ts, end_ts, period):
    for ep in (f"/historical/markets/{ticker}/candlesticks", f"/series/{series}/markets/{ticker}/candlesticks"):
        try:
            body = api.Get(ep, {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": period})
        except Exception:
            body = None
        if body and body.get("candlesticks"):
            return [NormCandle(c) for c in body["candlesticks"]]
    return []


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=",".join(DEFAULT_SERIES))
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--max-hours", type=float, default=96.0, help="cap for the hourly pass window")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hist-strikes", type=int, default=2,
                    help="historical-tier (older) markets need one API call each: keep only the middle N strikes of each player ladder")
    a = ap.parse_args(argv)
    SetBelowNormal()
    ForceIPv4()
    os.makedirs(OUT, exist_ok=True)
    api = KalshiApi(pause=a.pause)
    games = LoadGames([int(s) for s in a.seasons.split(",")])
    for series in [s.strip().upper() for s in a.series.split(",") if s.strip()]:
        t0 = datetime.now(timezone.utc)
        fees = SeriesBlock(api, series)
        markets, tier = CollectMarkets(api, series)
        rows, by_event = [], {}
        for m in markets.values():
            ev = m.get("event_ticker") or ""
            if ev not in by_event:
                by_event[ev] = MatchGame(ev, games)
            g = by_event[ev]
            name, pseg, thr = PlayerFromMarket(m, series)
            ko = Kickoff(g) if g else None
            ot = ParseTs(m.get("open_time"))
            rows.append({
                "ticker": m.get("ticker"), "event_ticker": ev, "series_ticker": series,
                "tier": tier.get(m.get("ticker"), ""), "status": m.get("status"), "result": m.get("result"),
                "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                "volume": m.get("volume_fp") or m.get("volume"), "open_interest": m.get("open_interest_fp"),
                "floor_strike": m.get("floor_strike"), "threshold": thr, "player_name": name, "player_seg": pseg,
                "yes_sub_title": m.get("yes_sub_title"),
                "game_id": g["game_id"] if g else None, "season": int(g["season"]) if g else None,
                "week": g.get("week") if g else None, "home_team": g["home_team"] if g else None,
                "away_team": g["away_team"] if g else None,
                "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ") if ko else None,
                "pregame_listed": bool(ko and ot and ot < ko),
            })
        # Sample historical-tier ladders: middle N strikes per (event, player). Pre-game, model-free rule.
        ladders = {}
        for r in rows:
            if r["tier"] == "historical" and r["pregame_listed"]:
                ladders.setdefault((r["event_ticker"], r["player_seg"]), []).append(r)
        for lad in ladders.values():
            lad.sort(key=lambda r: (r["threshold"] is None, r["threshold"] or 0))
            n = len(lad)
            lo = max(0, (n - a.hist_strikes) // 2)
            keep = {id(r) for r in lad[lo:lo + a.hist_strikes]}
            for r in lad:
                r["hist_sampled"] = id(r) in keep
        for r in rows:
            r.setdefault("hist_sampled", r["tier"] != "historical")
        with open(os.path.join(OUT, f"props_markets_{series}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cpath = os.path.join(OUT, f"props_candles_{series}.jsonl")
        done = DoneTickers(cpath)
        todo = [r for r in rows if r["pregame_listed"] and r["hist_sampled"] and r["ticker"] not in done]
        # live tier first (batch endpoint, fast), then historical (one call per market); group by kickoff
        todo.sort(key=lambda r: (r["tier"] == "historical", r["kickoff_utc"], r["ticker"]))
        print(f"{series}: markets={len(rows)} pregame_listed={sum(r['pregame_listed'] for r in rows)} "
              f"matched={sum(1 for r in rows if r['game_id'])} todo={len(todo)}", flush=True)
        n_done, n_empty = 0, 0
        with open(cpath, "a", encoding="utf-8") as fh:
            i = 0
            while i < len(todo):
                ko_s = todo[i]["kickoff_utc"]
                chunk = [todo[i]]
                i += 1
                while i < len(todo) and len(chunk) < BATCH and todo[i]["kickoff_utc"] == ko_s and todo[i]["tier"] == chunk[0]["tier"]:
                    chunk.append(todo[i]); i += 1
                ko = ParseTs(ko_s)
                opens = [ParseTs(r["open_time"]) for r in chunk]
                h_start = max(min(opens), ko - timedelta(hours=a.max_hours))
                ko_ts = ko.timestamp()
                hist_chunk = chunk[0]["tier"] == "historical"
                try:
                    if hist_chunk:
                        raise LookupError("historical tier")
                    hourly = Batched(api, [r["ticker"] for r in chunk], h_start.timestamp() - 3600, ko_ts + 120, 60)
                    minute = Batched(api, [r["ticker"] for r in chunk], ko_ts - 7200, ko_ts + 120, 1)
                    err = None
                except LookupError:
                    hourly, minute, err = {}, {}, None
                except Exception as exc:
                    hourly, minute, err = {}, {}, f"{type(exc).__name__}: {exc}"
                for r in chunk:
                    h = hourly.get(r["ticker"]) or []
                    mn = minute.get(r["ticker"]) or []
                    if not h and not mn and r["tier"] == "historical":
                        # one 1-minute call from listing (capped at KO-6h) to KO+2: covers open, T-60, T-15, close
                        m_start = max(ParseTs(r["open_time"]), ko - timedelta(hours=6)).timestamp() - 60
                        mn = Single(api, series, r["ticker"], m_start, ko_ts + 120, 1)
                    if not h and not mn:
                        n_empty += 1
                    fh.write(json.dumps({"ticker": r["ticker"], "series_ticker": series, "game_id": r["game_id"],
                                         "kickoff_utc": ko_s, "result": r["result"], "hourly": h, "minute": mn,
                                         "error": err, "fetched_at": datetime.now(timezone.utc).isoformat()}) + "\n")
                    n_done += 1
                fh.flush()
                if n_done % 500 < BATCH:
                    print(f"  {series} {n_done}/{len(todo)} empty={n_empty} calls={api.calls} retries={api.retries}", flush=True)
                if a.limit and n_done >= a.limit:
                    break
        summ = {"series": series, "started_at": t0.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
                "n_markets": len(rows), "n_pregame_listed": sum(r["pregame_listed"] for r in rows),
                "n_matched": sum(1 for r in rows if r["game_id"]), "n_candles_written": n_done, "n_empty": n_empty,
                "fee_type": fees.get("fee_type"), "fee_multiplier": fees.get("fee_multiplier"),
                "fee_changes": fees.get("fee_changes"), "calls": api.calls, "retries": api.retries}
        with open(os.path.join(OUT, f"props_pull_summary_{series}.json"), "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=1)
        print(json.dumps({k: v for k, v in summ.items() if k != "fee_changes"}), flush=True)


if __name__ == "__main__":
    Main()

