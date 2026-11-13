r"""
In-game (and pregame) price history for NFL game moneylines, finest public resolution, for market-calibration research.

    .venv\Scripts\python.exe -m DataPull.PullInGameHistory [--before-min 180 --after-min 330 --workers 4]

Kalshi: 1-minute candlesticks (finest period the public candlestick endpoint offers) for the HOME-team KXNFLGAME
market of each matched game, 2025 and 2026 (markets from data/kalshi/nflgame_markets.jsonl, written by PullKalshiHistory).
Polymarket: CLOB /prices-history at fidelity=1 (1 minute; smaller fidelity values return the same 1-minute series)
for token0 of each matched game in data/prices/polymarket_history_<season>.jsonl (written by PullPolymarketHistory).
Window: kickoff - before_min .. kickoff + after_min. Output (gitignored, resumable):
  data/prices/ingame_kalshi_<season>.jsonl, data/prices/ingame_poly_<season>.jsonl, data/prices/ingame_pull_summary.json
Public endpoints only; no accounts.
"""

import argparse
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from Backtest.Lab import LoadSeason
from Backtest.SnapshotPrices import Kickoff
from DataPull.PullKalshiHistory import SERIES, Api, NormCandle
from DataPull.PullPolymarketHistory import Get as PolyGet

# Tomputer's IPv6 route to Cloudflare stalls (SYN_SENT); prefer IPv4.
_GAI = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **k: _GAI(host, port, socket.AF_INET if family == 0 else family, *a, **k)

OUT = os.path.join("data", "prices")
CLOB = "https://clob.polymarket.com/prices-history"


def Done(path, key):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {json.loads(l)[key] for l in fh if l.strip()}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-min", type=int, default=180)
    ap.add_argument("--after-min", type=int, default=330)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    games = {g["game_id"]: g for s in (2024, 2025, 2026) for g in LoadSeason(s) if g.get("home_score") is not None}
    summary = {"window_min": [-a.before_min, a.after_min], "kalshi": {}, "poly": {}}

    api = Api()
    markets = [json.loads(l) for l in open(os.path.join("data", "kalshi", "nflgame_markets.jsonl"), encoding="utf-8")]
    for s in (2025, 2026):
        path = os.path.join(OUT, f"ingame_kalshi_{s}.jsonl")
        done = Done(path, "game_id")
        todo = [m for m in markets if m.get("game_id") in games and int(games[m["game_id"]]["season"]) == s
                and m.get("side_team") == games[m["game_id"]]["home_team"] and m["game_id"] not in done]

        def FetchK(m):
            g = games[m["game_id"]]
            ko = Kickoff(g)
            ep = f"/historical/markets/{m['ticker']}/candlesticks" if m["tier"] == "historical" else f"/series/{SERIES}/markets/{m['ticker']}/candlesticks"
            body = api.Get(ep, {"start_ts": int((ko - timedelta(minutes=a.before_min)).timestamp()),
                                "end_ts": int((ko + timedelta(minutes=a.after_min)).timestamp()), "period_interval": 1}) or {}
            cs = [NormCandle(c) for c in body.get("candlesticks") or []]
            return {"game_id": g["game_id"], "season": s, "week": g["week"], "ticker": m["ticker"], "side_team": m["side_team"],
                    "home_team": g["home_team"], "away_team": g["away_team"], "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "result": m.get("result"), "period_minutes": 1, "n": len(cs), "candles": cs}

        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
            for row in ex.map(FetchK, todo):
                fh.write(json.dumps(row) + "\n")
        summary["kalshi"][s] = {"games": len(Done(path, "game_id")), "fetched_now": len(todo)}
        print("kalshi", s, summary["kalshi"][s], flush=True)

    for s in (2024, 2025, 2026):
        src = os.path.join(OUT, f"polymarket_history_{s}.jsonl")
        path = os.path.join(OUT, f"ingame_poly_{s}.jsonl")
        done = Done(path, "game_id")
        todo = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        todo = [r for r in todo if r["game_id"] not in done]

        def FetchP(r):
            ko = datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00"))
            q = {"market": r["tokens"][0], "startTs": int((ko - timedelta(minutes=a.before_min)).timestamp()),
                 "endTs": int((ko + timedelta(minutes=a.after_min)).timestamp()), "fidelity": 1}
            body = PolyGet(CLOB + "?" + "&".join(f"{k}={v}" for k, v in q.items())) or {}
            h = (body.get("history") or []) if isinstance(body, dict) else []
            return {"game_id": r["game_id"], "season": s, "week": r["week"], "home_team": r["home_team"], "away_team": r["away_team"],
                    "kickoff_utc": r["kickoff_utc"], "token0_team": r["token0_team"], "token": r["tokens"][0],
                    "fidelity_min": 1, "n": len(h), "history": h}

        with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(a.workers) as ex:
            for row in ex.map(FetchP, todo):
                fh.write(json.dumps(row) + "\n")
        summary["poly"][s] = {"games": len(Done(path, "game_id")), "fetched_now": len(todo)}
        print("poly", s, summary["poly"][s], flush=True)
    summary["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(OUT, "ingame_pull_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    Main()