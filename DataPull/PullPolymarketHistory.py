r"""
Download Polymarket public price history for NFL game moneyline markets into readable JSONL. Public, no account.

    .venv\Scripts\python.exe -m DataPull.PullPolymarketHistory [--first 2023 --last 2026] [--before-min 360] [--workers 4]

Steps
  1. Gamma GET /events?tag_id=450 (NFL), closed=true, paged 50 at a time in both date orders (the API caps offset
     near 2,050 and limit at 50), deduped by event id.
  2. Each moneyline market (sportsMarketType == "moneyline", two team outcomes) is matched to an nflverse game by the
     UNORDERED team pair (outcome nicknames) and kickoff within 36 h (slug team order is not consistent across seasons).
  3. CLOB GET /prices-history?market=<first outcome token>&startTs=kickoff-360m&endTs=kickoff+5m&fidelity=1.
     The series is Polymarket's own price for that token (one number, no bid/ask, no size). The other side = 1 - p.
Writes (gitignored): data/prices/polymarket_history_<season>.jsonl (one row per game), data/prices/polymarket_pull_summary.json
"""

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from Backtest.AssessLive import NICKNAME
from Backtest.Lab import LoadSeason
from Backtest.SnapshotPrices import Kickoff

# Tomputer's IPv6 route to Cloudflare stalls (SYN_SENT); urllib tries addresses in order, so prefer IPv4.
_GAI = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **k: _GAI(host, port, socket.AF_INET if family == 0 else family, *a, **k)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
UA = "football-PullPolymarketHistory/readonly"
OUT = os.path.join("data", "prices")


def Get(url, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404, 406, 422):
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    return None


def Q(url, params):
    return url + "?" + urllib.parse.urlencode(params)


def ListEvents():
    seen = {}
    for asc in ("true", "false"):
        off = 0
        while True:
            body = Get(Q(GAMMA + "/events", {"tag_id": 450, "closed": "true", "limit": 50, "offset": off,
                                              "order": "startDate", "ascending": asc}), tries=3)
            if not body:
                break
            for e in body:
                seen[e["id"]] = e
            off += len(body)
            if len(body) < 50:
                break
    return list(seen.values())


def JList(v):
    if isinstance(v, list):
        return v
    try:
        x = json.loads(v or "[]")
        return x if isinstance(x, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def Team(name):
    return NICKNAME.get(str(name or "").strip().lower())


def ParseTs(s):
    if not s:
        return None
    s = s.strip().replace(" ", "T").replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    try:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2023)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--before-min", type=int, default=360)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    events = ListEvents()
    games = [g for s in range(a.first, a.last + 1) for g in LoadSeason(s) if g.get("home_score") is not None]
    by_pair = {}
    for g in games:
        by_pair.setdefault(frozenset((g["home_team"], g["away_team"])), []).append(g)
    matched, unmatched = {}, Counter()
    for e in events:
        for m in e.get("markets") or []:
            if (m.get("sportsMarketType") or "").lower() != "moneyline":
                continue
            outs, toks = JList(m.get("outcomes")), JList(m.get("clobTokenIds"))
            teams = [Team(o) for o in outs]
            if len(outs) != 2 or len(toks) != 2 or None in teams:
                unmatched["not_two_nfl_teams"] += 1
                continue
            start = ParseTs(m.get("gameStartTime")) or ParseTs(e.get("startDate"))
            cands = [g for g in by_pair.get(frozenset(teams), []) if start and abs((Kickoff(g) - start).total_seconds()) <= 36 * 3600]
            if len(cands) != 1:
                unmatched["no_game_match" if not cands else "ambiguous"] += 1
                continue
            g = cands[0]
            prev = matched.get(g["game_id"])
            if prev and float(prev[1].get("volumeNum") or 0) >= float(m.get("volumeNum") or 0):
                continue
            matched[g["game_id"]] = (g, m, e, teams, toks)

    def Fetch(item):
        g, m, e, teams, toks = item
        ko = Kickoff(g)
        body = Get(Q(CLOB + "/prices-history", {"market": toks[0], "startTs": int((ko - timedelta(minutes=a.before_min)).timestamp()),
                                                 "endTs": int((ko + timedelta(minutes=5)).timestamp()), "fidelity": 1}))
        pts = [{"t": int(x["t"]), "p": float(x["p"])} for x in (body or {}).get("history") or []]
        return {"game_id": g["game_id"], "season": int(g["season"]), "week": g["week"], "game_type": g.get("game_type"),
                "home_team": g["home_team"], "away_team": g["away_team"], "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_slug": e.get("slug"), "market_slug": m.get("slug"), "condition_id": m.get("conditionId"),
                "outcomes": JList(m.get("outcomes")), "tokens": toks, "token0_team": teams[0], "token1_team": teams[1],
                "fees_enabled": m.get("feesEnabled"), "fee_type": m.get("feeType"), "fee": m.get("fee"),
                "volume": m.get("volumeNum"), "price_series": "CLOB /prices-history p for token0 (token1 = 1 - p)",
                "fidelity_min": 1, "fetched_at": datetime.now(timezone.utc).isoformat(), "n": len(pts), "history": pts}

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(Fetch, matched.values()))
    per = Counter()
    files = {}
    for r in sorted(rows, key=lambda r: (r["kickoff_utc"], r["game_id"])):
        s = r["season"]
        if s not in files:
            files[s] = open(os.path.join(OUT, f"polymarket_history_{s}.jsonl"), "w", encoding="utf-8")
        files[s].write(json.dumps(r) + "\n")
        per[(s, "games")] += 1
        per[(s, "with_points")] += r["n"] > 0
    for fh in files.values():
        fh.close()
    totals = Counter()
    for g in games:
        totals[int(g["season"])] += 1
    summary = {"events_listed": len(events), "moneyline_matched_games": len(matched), "unmatched_markets": dict(unmatched),
               "by_season": {s: {"final_games": totals[s], "matched": per[(s, "games")], "with_history": per[(s, "with_points")]}
                             for s in range(a.first, a.last + 1)},
               "event_year_counts": dict(Counter((e.get("startDate") or "")[:4] for e in events))}
    with open(os.path.join(OUT, "polymarket_pull_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    Main()