r"""
Live game view for in-progress NFL / CFB games (read-only).

    .venv\Scripts\python.exe -m Tools.Live --league cfb                       (one snapshot, today ET)
    .venv\Scripts\python.exe -m Tools.Live --league nfl --watch 60 --game DET (refresh every 60 s; Ctrl+C to stop)
    .venv\Scripts\python.exe -m Tools.Live --league cfb --all                 (include scheduled/final games)

Per game: score, clock, quarter, possession, down & distance, ESPN home win %, swing since the last update
(ESPN win % change vs. the previous snapshot this tool saw, kept in data\tools\live_state_<league>.json),
and the Kalshi live game-winner mid for the home team (public API) with its change since the last update.
Also writes E:\AgentDocs\American-Footbot\live\<league>_latest.md and .json each refresh.
ESPN win % comes from the scoreboard situation; if missing, from the game summary winprobability feed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from Tools import Common as C

OUT_DIR = os.path.join("E:\\AgentDocs", "American-Footbot", "live")


def StatePath(league):
    return os.path.join("data", "tools", f"live_state_{league}.json")


def Snapshot(league, date, game=None, include_all=False, kalshi=True):
    events, at = C.EspnScoreboard(league, date)
    kmk = C.KalshiOpen(C.KALSHI_SERIES[league]["game"])[0] if kalshi else []
    sp = StatePath(league)
    prev = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {}
    rows, new_state = [], dict(prev)
    for ev in events:
        st = (ev.get("status") or {}).get("type", {})
        if not include_all and st.get("state") != "in":
            continue
        if game and game.lower() not in (ev.get("name", "") + " " + ev.get("shortName", "")).lower():
            continue
        comp, h, a = C.Competitors(ev)
        sit = C.Situation(comp)
        wp = sit["home_win_pct"]
        if wp is None and st.get("state") == "in":
            summ, _ = C.EspnSummary(league, ev["id"])
            wpl = summ.get("winprobability") or []
            if wpl:
                wp = wpl[-1].get("homeWinPercentage")
        poss = None
        if sit["possession"]:
            poss = h["team"]["abbreviation"] if str(sit["possession"]) == str(h["team"]["id"]) else (
                a["team"]["abbreviation"] if str(sit["possession"]) == str(a["team"]["id"]) else None)
        start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        km = C.MatchKalshiGame(kmk, h["team"], a["team"], start) if kmk else None
        kmid = km["home"]["mid"] if km else None
        p = prev.get(str(ev["id"]), {})
        swing = None if (wp is None or p.get("wp") is None) else wp - p["wp"]
        kswing = None if (kmid is None or p.get("kmid") is None) else kmid - p["kmid"]
        r = {"espn_event_id": str(ev["id"]), "matchup": f"{a['team']['abbreviation']} @ {h['team']['abbreviation']}",
             "away_score": int(a.get("score") or 0), "home_score": int(h.get("score") or 0),
             "state": st.get("state"), "status": st.get("shortDetail") or st.get("description"),
             "quarter": (ev.get("status") or {}).get("period"), "clock": (ev.get("status") or {}).get("displayClock"),
             "possession": poss, "down_distance": sit["down_distance"], "home_win_pct": wp, "swing": swing,
             "since": p.get("at"), "kalshi_home_bid": km["home"]["bid"] if km else None,
             "kalshi_home_ask": km["home"]["ask"] if km else None, "kalshi_home_mid": kmid, "kalshi_swing": kswing,
             "last_play": sit["last_play"]}
        rows.append(r)
        new_state[str(ev["id"])] = {"wp": wp, "kmid": kmid, "at": C.IsoEt()}
    os.makedirs(os.path.dirname(sp), exist_ok=True)
    json.dump(new_state, open(sp, "w", encoding="utf-8"), indent=1)
    rows.sort(key=lambda r: (r["state"] != "in", r["matchup"]))
    return {"league": league, "date": date, "generated_at": C.IsoEt(), "espn_at": at, "games": rows}


def Fmt(snap):
    def pc(x):
        return "  n/a" if x is None else f"{x * 100:5.1f}%"
    def sw(x):
        return "   " if x is None else f"{x * 100:+5.1f}"
    L = [f"{snap['league'].upper()} live - {C.StampEt()} ({len(snap['games'])} games)", "",
         "| game | score | qtr/clock | poss | down & dist | ESPN home win | swing | Kalshi home bid/ask (mid) | K swing |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in snap["games"]:
        k = "n/a" if r["kalshi_home_mid"] is None else f"{r['kalshi_home_bid']}/{r['kalshi_home_ask']} ({r['kalshi_home_mid']:.3f})"
        qc = f"Q{r['quarter']} {r['clock']}" if r["state"] == "in" else r["status"]
        L.append(f"| {r['matchup']} | {r['away_score']}-{r['home_score']} | {qc} | {r['possession'] or ''} | {r['down_distance'] or ''} | "
                 f"{pc(r['home_win_pct'])} | {sw(r['swing'])} | {k} | {sw(r['kalshi_swing'])} |")
    if not snap["games"]:
        L.append("| no games in progress | | | | | | | | |")
    L += ["", "swing = change in ESPN home win % since the previous snapshot (pts); K swing = change in Kalshi home mid.",
          "Sources: ESPN scoreboard/summary, Kalshi public API. Read-only."]
    return "\n".join(L) + "\n"


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nfl", "cfb"], default="nfl")
    ap.add_argument("--date", default=datetime.now(C.ET).date().isoformat())
    ap.add_argument("--game")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--watch", type=int, default=0, help="refresh seconds (0 = one snapshot)")
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--no-kalshi", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    n = 0
    while True:
        snap = Snapshot(a.league, a.date, a.game, a.all, not a.no_kalshi)
        txt = Fmt(snap)
        print(txt, flush=True)
        os.makedirs(a.out_dir, exist_ok=True)
        open(os.path.join(a.out_dir, f"{a.league}_latest.md"), "w", encoding="utf-8").write(txt)
        json.dump(snap, open(os.path.join(a.out_dir, f"{a.league}_latest.json"), "w", encoding="utf-8"), indent=1)
        n += 1
        if a.watch <= 0 or (a.max_updates and n >= a.max_updates):
            break
        time.sleep(max(10, a.watch))


if __name__ == "__main__":
    Main()
