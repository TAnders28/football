r"""
Download Kalshi NFL game-winner (KXNFLGAME) bid/ask history around kickoff into readable JSONL. Public, no auth.

    .venv\Scripts\python.exe -m DataPull.PullKalshiHistory --seasons 2025,2026 [--before-min 180] [--after-min 5]

Writes (gitignored):
  data/kalshi/nflgame_markets.jsonl            every KXNFLGAME market seen (historical + live tier), with matched game_id
  data/kalshi/nflgame_candles_<season>.jsonl   one row per matched market: 1-minute candles from kickoff-180m to
                                               kickoff+5m (yes_bid / yes_ask / last-price close, volume, open interest)
  data/kalshi/fee_changes.json                 GET /series/fee_changes for KXNFLGAME (historical included)
  data/kalshi/pull_summary.json
Resumable: tickers already in a candles file are skipped.

Endpoints (https://docs.kalshi.com): GET /historical/cutoff, /historical/markets, /historical/markets/{t}/candlesticks
(markets settled before the cutoff), GET /markets?status=settled|closed, /series/{s}/markets/{t}/candlesticks (newer).
Candle yes_bid / yes_ask are top-of-book prices at the period close; there is NO size in candles.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from Backtest.Lab import LoadSeason
from Backtest.SnapshotPrices import KalshiEventMatch, Kickoff, SideTeam

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
OUT = os.path.join("data", "kalshi")
UA = "football-PullKalshiHistory/readonly"


class Api:
    def __init__(self, pause=0.12):
        self.pause, self.calls, self.retries = pause, 0, 0

    def Get(self, path, params=None):
        url = KALSHI + path + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(6):
            time.sleep(self.pause)
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
                with urllib.request.urlopen(req, timeout=40) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 5:
                    self.retries += 1
                    time.sleep(2 ** attempt)
                    continue
                if exc.code == 404:
                    return None
                raise
            except urllib.error.URLError:
                if attempt < 5:
                    self.retries += 1
                    time.sleep(2 ** attempt)
                    continue
                raise
        return None

    def Paged(self, path, params, key="markets", max_pages=50):
        out, cursor = [], ""
        for _ in range(max_pages):
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            body = self.Get(path, p) or {}
            out += body.get(key) or []
            cursor = body.get("cursor") or ""
            if not cursor or not body.get(key):
                break
        return out


def F(d, *names):
    for n in names:
        if d and d.get(n) is not None:
            try:
                return float(d[n])
            except (TypeError, ValueError):
                return None
    return None


def NormCandle(c):
    ts = int(c["end_period_ts"])
    return {"end_ts": ts, "end_utc": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "yes_bid_close": F(c.get("yes_bid"), "close_dollars", "close"),
            "yes_ask_close": F(c.get("yes_ask"), "close_dollars", "close"),
            "price_close": F(c.get("price"), "close_dollars", "close"),
            "volume": F(c, "volume_fp", "volume"), "open_interest": F(c, "open_interest_fp", "open_interest")}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025,2026")
    ap.add_argument("--before-min", type=int, default=180)
    ap.add_argument("--after-min", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    api = Api()
    cutoff = api.Get("/historical/cutoff") or {}
    fee = api.Get("/series/fee_changes", {"series_ticker": SERIES, "show_historical": "true"})
    series = api.Get(f"/series/{SERIES}") or {}
    with open(os.path.join(OUT, "fee_changes.json"), "w", encoding="utf-8") as fh:
        json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "fee_changes": fee,
                   "series_fee_type": (series.get("series") or {}).get("fee_type"),
                   "series_fee_multiplier": (series.get("series") or {}).get("fee_multiplier")}, fh, indent=1)

    hist = api.Paged("/historical/markets", {"series_ticker": SERIES, "limit": 1000})
    live = []
    for status in ("settled", "closed"):
        live += api.Paged("/markets", {"series_ticker": SERIES, "status": status, "limit": 1000})
    tier = {m["ticker"]: "historical" for m in hist}
    for m in live:
        tier.setdefault(m["ticker"], "live")
    markets = {m["ticker"]: m for m in hist + live}

    seasons = [int(s) for s in a.seasons.split(",")]
    games = [g for s in seasons for g in LoadSeason(s) if g.get("home_score") is not None]
    by_event = {}
    for m in markets.values():
        ev = m.get("event_ticker") or ""
        if ev not in by_event:
            by_event[ev] = next((g for g in games if KalshiEventMatch(ev, g)), None)
    with open(os.path.join(OUT, "nflgame_markets.jsonl"), "w", encoding="utf-8") as fh:
        for t, m in sorted(markets.items()):
            g = by_event.get(m.get("event_ticker") or "")
            fh.write(json.dumps({"ticker": t, "event_ticker": m.get("event_ticker"), "tier": tier[t], "status": m.get("status"),
                                 "result": m.get("result"), "close_time": m.get("close_time"), "volume": m.get("volume_fp") or m.get("volume"),
                                 "game_id": g["game_id"] if g else None, "side_team": SideTeam(t, g) if g else None}) + "\n")

    summary = {"cutoff": cutoff, "markets_total": len(markets), "historical_tier": len(hist), "live_tier": len(live),
               "matched_games": {}, "candle_rows": {}, "empty_candles": {}}
    for s in seasons:
        path = os.path.join(OUT, f"nflgame_candles_{s}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                done = {json.loads(l)["ticker"] for l in fh if l.strip()}
        todo = [(t, m, by_event[m.get("event_ticker") or ""]) for t, m in sorted(markets.items())
                if by_event.get(m.get("event_ticker") or "") and int(by_event[m["event_ticker"]]["season"]) == s]
        summary["matched_games"][s] = len({g["game_id"] for _, _, g in todo})
        empty = 0
        with open(path, "a", encoding="utf-8") as fh:
            for t, m, g in todo:
                if t in done:
                    continue
                ko = Kickoff(g)
                params = {"start_ts": int((ko - timedelta(minutes=a.before_min)).timestamp()),
                          "end_ts": int((ko + timedelta(minutes=a.after_min)).timestamp()), "period_interval": 1}
                ep = f"/historical/markets/{t}/candlesticks" if tier[t] == "historical" else f"/series/{SERIES}/markets/{t}/candlesticks"
                body = api.Get(ep, params) or {}
                cs = [NormCandle(c) for c in body.get("candlesticks") or []]
                empty += not cs
                fh.write(json.dumps({"ticker": t, "event_ticker": m.get("event_ticker"), "game_id": g["game_id"], "season": s,
                                     "week": g["week"], "game_type": g.get("game_type"), "side_team": SideTeam(t, g),
                                     "home_team": g["home_team"], "away_team": g["away_team"],
                                     "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"), "result": m.get("result"),
                                     "endpoint": ep, "period_minutes": 1, "fetched_at": datetime.now(timezone.utc).isoformat(),
                                     "n": len(cs), "candles": cs}) + "\n")
        summary["candle_rows"][s] = len(todo)
        summary["empty_candles"][s] = empty
        print(s, "matched games", summary["matched_games"][s], "markets", len(todo), "empty", empty, "calls", api.calls, flush=True)
    summary["http_calls"], summary["retries"] = api.calls, api.retries
    summary["unmatched_events"] = sorted(ev for ev, g in by_event.items() if g is None)[:400]
    with open(os.path.join(OUT, "pull_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "unmatched_events"}), "unmatched", len(summary["unmatched_events"]))


if __name__ == "__main__":
    Main()