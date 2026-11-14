r"""
GutCall board feed: one JSON per slate for Degen Bot's cross-sport dashboard (read-only, no orders).

    E:\Football\football\.venv\Scripts\python.exe E:\Football\football\Tools\Board.py --league nfl
    (cwd-independent; --date defaults to today in ET if that day still has an unfinished game, else the next slate
     within 8 days. `python -m Tools.Board` also works from E:\Football\football.)

Writes E:\AgentDocs\American-Footbot\_dashboard\board_latest.json and board_<date>.json.
No E:\GutCall\gutcall\SCHEMA.md existed on 2026-11-14, so this follows the schema in the request:
  top: sport, generated_at (ISO -05:00), games[]
  game: game_id, start_time, home, away, status, score, market{home_yes_ask, away_yes_ask, open_home_yes, source, as_of},
        model{home_win_prob, model_name, tested}, edge_home_c, edge_away_c,
        context{injuries[{team, player, status, note}], recent_form, rest_days, head_to_head, notes[]},
        props[{player, market, line, yes_ask, model_prob, edge_c, tested}]
Rules: unknown -> null, never invented. edge_x_c = (prob - ask - 0.02) * 100 (0.02 = Tim's fee estimate).
Model = p_full2_kmid_T60 via Backtest.DailyCheck. tested=false: per GutCall, tested=true only when a model's
fee-adjusted edge passed a holdout AND SPA; this one did not (holdout edge not positive, SPA not passed).
Props: Kalshi yes_ask only, model_prob/edge_c null, tested false (no prop model passed the 2026-11-14 holdout).
Prop line = the Kalshi "k+" threshold (e.g. line 60 on receiving_yards means 60 or more yards).
open_home_yes = home yes ask on the first valid hourly Kalshi candle after listing.
CFB (--league cfb): ESPN + Kalshi KXNCAAFGAME asks, model null (no tested CFB model), props [].
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

from Tools import Common as C

OUT_DIR = os.path.join("E:\\AgentDocs", "American-Footbot", "_dashboard")
MODEL = "p_full2_kmid_T60"
MODEL_NOTE = ("model p_full2_kmid_T60 (walk-forward 2025-26) is NOT validated: its fee-adjusted edge failed the holdout and "
              "SPA (2026-11-13_bankroll-10k, 2026-11-14_prop-pricing), so tested=false; reference only")
GAME_KEYS = ["game_id", "start_time", "home", "away", "status", "score", "market", "model", "edge_home_c", "edge_away_c",
             "context", "props"]
MARKET_KEYS = ["home_yes_ask", "away_yes_ask", "open_home_yes", "source", "as_of"]
MODEL_KEYS = ["home_win_prob", "model_name", "tested"]
CONTEXT_KEYS = ["injuries", "recent_form", "rest_days", "head_to_head", "notes"]
PROP_KEYS = ["player", "market", "line", "yes_ask", "model_prob", "edge_c", "tested"]


def EdgeC(prob, ask):
    if prob is None or ask is None:
        return None
    return round((prob - ask - C.TIM_FEE) * 100, 1)


def Validate(board):
    """Raise ValueError on any schema problem. Returns number of games."""
    def need(d, keys, where):
        if not isinstance(d, dict):
            raise ValueError(f"{where}: not an object")
        miss = [k for k in keys if k not in d]
        if miss:
            raise ValueError(f"{where}: missing {miss}")
    need(board, ["sport", "generated_at", "games"], "board")
    if board["sport"] not in ("NFL", "CFB"):
        raise ValueError("sport must be 'NFL' or 'CFB' (uppercase)")
    if not board["generated_at"].endswith(("-05:00", "-04:00")):
        raise ValueError("generated_at must carry the ET offset")
    datetime.fromisoformat(board["generated_at"])
    for i, g in enumerate(board["games"]):
        w = f"games[{i}]"
        need(g, GAME_KEYS, w)
        need(g["market"], MARKET_KEYS, w + ".market")
        need(g["model"], MODEL_KEYS, w + ".model")
        need(g["context"], CONTEXT_KEYS, w + ".context")
        for j, inj in enumerate(g["context"]["injuries"] or []):
            need(inj, ["team", "player", "status", "note"], f"{w}.injuries[{j}]")
        if not isinstance(g["context"]["notes"], list):
            raise ValueError(f"{w}.notes must be a list")
        for j, p in enumerate(g["props"]):
            need(p, PROP_KEYS, f"{w}.props[{j}]")
            if p["tested"] or p["model_prob"] is not None:
                raise ValueError(f"{w}.props[{j}]: untested prop must have model_prob null and tested false")
        if g["model"]["tested"]:
            raise ValueError(f"{w}.model.tested must be false (no model passed holdout + SPA)")
        for side in ("home", "away"):
            ask, e = g["market"][f"{side}_yes_ask"], g[f"edge_{side}_c"]
            p = g["model"]["home_win_prob"]
            if e is not None:
                pr = p if side == "home" else 1 - p
                if abs(e - (pr - ask - C.TIM_FEE) * 100) > 0.11:
                    raise ValueError(f"{w}: edge_{side}_c inconsistent")
        for k in ("home_yes_ask", "away_yes_ask", "open_home_yes"):
            v = g["market"][k]
            if v is not None and not (0 < v < 1):
                raise ValueError(f"{w}.market.{k} out of (0,1)")
    return len(board["games"])


def FormString(rows):
    if not rows:
        return None
    s = "".join((r.get("result") or "?") for r in rows)
    pts = [(r["pf"], r["pa"]) for r in rows if r.get("pf") is not None and r.get("pa") is not None]
    if pts:
        s += f" (last {len(pts)}: avg {sum(a for a, _ in pts) / len(pts):.1f}-{sum(b for _, b in pts) / len(pts):.1f})"
    return s


def HeadToHead(all_games, home, away, before, n=3):
    rows = [g for g in all_games if {g.get("home_team"), g.get("away_team")} == {home, away}
            and g.get("home_score") is not None and (g.get("gameday") or "") < before]
    rows.sort(key=lambda g: g["gameday"], reverse=True)
    if not rows:
        seasons = sorted({g.get("season") for g in all_games if g.get("season")})
        return f"no meeting in nflverse schedule {seasons[0]}-{seasons[-1]}" if seasons else None
    out = []
    for g in rows[:n]:
        out.append(f"{g['gameday']} {g['away_team']} {int(g['away_score'])} @ {g['home_team']} {int(g['home_score'])}")
    return "; ".join(out) + " (nflverse schedule)"


def LoadNflGames(seasons):
    out = []
    for s in seasons:
        p = os.path.join("data", "backtest", f"games_{s}.json")
        if os.path.exists(p):
            body = json.load(open(p, encoding="utf-8"))
            out += body["games"] if isinstance(body, dict) else body
    return out


def NflProps(prop_markets, suffix, team_by_kalshi):
    out = []
    for series, kind in C.PROP_SERIES.items():
        for m in prop_markets.get(series, []):
            if (m.get("event_ticker") or "") != f"{series}-{suffix}":
                continue
            ask = C.Dollars(m, "yes_ask")
            title = m.get("yes_sub_title") or m.get("title") or ""
            player = title.split(":")[0].strip() or None
            fs = m.get("floor_strike")
            line = None if fs is None else int(round(float(fs) + 0.5))
            out.append({"player": player, "market": kind, "line": line,
                        "yes_ask": ask if ask is not None and 0 < ask < 1 else None,
                        "model_prob": None, "edge_c": None, "tested": False})
    out.sort(key=lambda p: (p["market"], p["player"] or "", p["line"] or 0))
    return out


def Build(league, date, out_dir=OUT_DIR, with_props=True, log=print):
    t0 = time.time()
    gen = C.IsoEt()
    events, espn_at = C.EspnScoreboard(league, date)
    board = {"sport": league.upper(), "generated_at": gen, "date": date, "games": [],
             "sources": {"espn_scoreboard": espn_at}, "errors": []}
    model_by_espn, quotes_by_espn, nfl_by_espn, all_nfl = {}, {}, {}, []
    if league == "nfl":
        from Backtest.DailyCheck import Check
        from Backtest.PredictDay import SeasonOf
        season = SeasonOf(date)
        all_nfl = LoadNflGames(range(season - 6, season + 1))
        for g in all_nfl:
            if g.get("espn"):
                nfl_by_espn[str(g["espn"])] = g
        try:
            dc = Check(date, out_dir=os.path.join("data", "tools", "dailycheck"))
            board["sources"]["kalshi_game"] = dc.get("generated_at")
            for rec in dc.get("games", []):
                g = next((x for x in all_nfl if x["game_id"] == rec["game_id"]), None)
                if g and g.get("espn"):
                    model_by_espn[str(g["espn"])] = rec
            board["errors"] += dc.get("errors", [])
        except Exception as exc:
            board["errors"].append(f"model/DailyCheck: {type(exc).__name__}: {exc}")
    league_inj = {}
    if league == "nfl":
        try:
            league_inj, board["sources"]["espn_injuries"] = C.LeagueInjuries(league)
        except Exception as exc:
            board["errors"].append(f"espn injuries: {type(exc).__name__}")
    game_mk, _ = C.KalshiOpen(C.KALSHI_SERIES[league]["game"])
    k_at = C.IsoEt()
    props_mk = {}
    if league == "nfl" and with_props:
        for s in C.PROP_SERIES:
            try:
                props_mk[s], _ = C.KalshiOpen(s)
            except Exception as exc:
                board["errors"].append(f"kalshi {s}: {type(exc).__name__}")
        board["sources"]["kalshi_props"] = C.IsoEt()
    for ev in events:
        comp, h, a = C.Competitors(ev)
        start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        if start.astimezone(C.ET).date().isoformat() != date:
            continue
        eid = str(ev["id"])
        ng = nfl_by_espn.get(eid)
        state = (ev.get("status") or {}).get("type", {}).get("state")
        status = {"pre": "scheduled", "in": "in_progress", "post": "final"}.get(state, state)
        score = None
        if state in ("in", "post"):
            score = {"home": int(h.get("score") or 0), "away": int(a.get("score") or 0),
                     "clock": (ev.get("status") or {}).get("displayClock"), "period": (ev.get("status") or {}).get("period")}
        km = C.MatchKalshiGame(game_mk, h["team"], a["team"], start)
        market = {"home_yes_ask": None, "away_yes_ask": None, "open_home_yes": None,
                  "source": f"Kalshi public API {C.KALSHI_SERIES[league]['game']} (yes ask, $)", "as_of": k_at}
        if km:
            market["home_yes_ask"] = km["home"]["ask"] if km["home"]["ask"] and 0 < km["home"]["ask"] < 1 else None
            market["away_yes_ask"] = km["away"]["ask"] if km["away"]["ask"] and 0 < km["away"]["ask"] < 1 else None
            mh = next((m for m in game_mk if m.get("ticker") == km["home"]["ticker"]), None)
            if mh and mh.get("open_time"):
                try:
                    fp = C.KalshiFirstPrice(C.KALSHI_SERIES[league]["game"], mh["ticker"], mh["open_time"], start)
                    market["open_home_yes"] = fp["ask"] if fp else None
                except Exception as exc:
                    board["errors"].append(f"open price {mh['ticker']}: {type(exc).__name__}")
        else:
            market["source"] += " - no matching open market"
        rec = model_by_espn.get(eid)
        p_home = rec.get("p_home") if rec else None
        model = {"home_win_prob": p_home, "model_name": MODEL if league == "nfl" else None, "tested": False}
        try:
            summ, s_at = C.EspnSummary(league, eid)
        except Exception as exc:
            summ, s_at = {}, None
            board["errors"].append(f"espn summary {eid}: {type(exc).__name__}")
        injuries = None
        if str(h["team"]["id"]) in league_inj or str(a["team"]["id"]) in league_inj:
            ih, ia = league_inj.get(str(h["team"]["id"]), []), league_inj.get(str(a["team"]["id"]), [])
        else:
            ih, ia = C.Injuries(summ, h["team"]["id"]), C.Injuries(summ, a["team"]["id"])
        if ih is not None or ia is not None:
            injuries = []
            for team, lst in ((h["team"]["abbreviation"], ih or []), (a["team"]["abbreviation"], ia or [])):
                for x in lst:
                    if (x.get("status") or "") == "Active":
                        continue  # ESPN keeps players cleared to play in the feed as 'Active'; not an injury
                    bits = [x["pos"], x["type"], x["detail"] if x["detail"] != "Not Specified" else None,
                            f"updated {x['updated'][:10]}" if x.get("updated") else None]
                    note = ", ".join(str(v) for v in bits if v) or None
                    injuries.append({"team": team, "player": x["player"], "status": x["status"], "note": note})
        fh, fa = C.LastFive(summ, h["team"]["id"]), C.LastFive(summ, a["team"]["id"])
        recent = None
        if fh or fa:
            recent = {"home": FormString(fh), "away": FormString(fa), "source": f"ESPN summary lastFiveGames, {s_at}"}
        rest = None
        if ng and ng.get("home_rest") is not None:
            rest = {"home": ng.get("home_rest"), "away": ng.get("away_rest"), "source": "nflverse schedule"}
        h2h = HeadToHead(all_nfl, ng["home_team"], ng["away_team"], date) if ng else None
        notes = []
        if league == "nfl":
            notes.append(MODEL_NOTE)
            if injuries is not None:
                notes.append(f"injuries: ESPN injuries feed ({board['sources'].get('espn_injuries')}); ESPN lists at most 25 per team, incl. IR")
            if market["open_home_yes"] is not None:
                notes.append("open_home_yes = home yes ask in the first hourly Kalshi candle after listing with a <=5c spread")
        if rec and rec.get("status") and rec.get("status") != "scored":
            notes.append(f"model not scored: {rec['status']} (DailyCheck, {board['sources'].get('kalshi_game')})")
        wx = C.Weather(comp.get("venue"), start)
        notes.append(f"weather: {wx['text']} (source {wx.get('source')}, fetched {wx.get('fetched_at') or C.IsoEt()})")
        for rk in (h, a):
            for r in rk.get("records") or []:
                if r.get("type") == "total":
                    notes.append(f"record {rk['team']['abbreviation']}: {r.get('summary')} (ESPN scoreboard, {espn_at})")
        for side, rk in (("home", h), ("away", a)):
            lf = fh if side == "home" else fa
            if lf:
                streak, last = 0, lf[0].get("result")
                for r in lf:
                    if r.get("result") == last:
                        streak += 1
                    else:
                        break
                if last in ("W", "L") and streak >= 3:
                    notes.append(f"streak {rk['team']['abbreviation']}: {last}{streak} in last-5 window (ESPN summary, {s_at})")
        pred = C.Predictor(summ)
        if pred:
            notes.append(f"ESPN FPI matchup predictor: home {pred['home']:.3f} (ESPN summary, {s_at})")
        props = []
        if league == "nfl" and km and with_props:
            props = NflProps(props_mk, km["event_suffix"], None)
        g = {"game_id": ng["game_id"] if ng else f"espn_{eid}", "start_time": C.IsoEt(start),
             "home": h["team"]["abbreviation"], "away": a["team"]["abbreviation"], "status": status, "score": score,
             "market": market, "model": model,
             "edge_home_c": EdgeC(p_home, market["home_yes_ask"]),
             "edge_away_c": EdgeC(None if p_home is None else 1 - p_home, market["away_yes_ask"]),
             "context": {"injuries": injuries, "recent_form": recent, "rest_days": rest, "head_to_head": h2h, "notes": notes},
             "props": props, "espn_event_id": eid}
        board["games"].append(g)
    board["games"].sort(key=lambda g: (g["start_time"], g["game_id"]))
    board["runtime_s"] = round(time.time() - t0, 1)
    Validate(board)
    os.makedirs(out_dir, exist_ok=True)
    for name in ("board_latest.json", f"board_{date}.json") if league == "nfl" else (f"board_{league}_latest.json", f"board_{league}_{date}.json"):
        tmp = os.path.join(out_dir, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(board, fh, indent=1)
        os.replace(tmp, os.path.join(out_dir, name))
    return board


def NextSlate(league, days=8):
    """Today (ET) if it has a game not yet final, else the next date with games."""
    today = datetime.now(C.ET).date()
    for d in range(days):
        day = (today + timedelta(days=d)).isoformat()
        evs, _ = C.EspnScoreboard(league, day)
        evs = [e for e in evs if datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(C.ET).date().isoformat() == day]
        if any((e.get("status") or {}).get("type", {}).get("state") != "post" for e in evs):
            return day
    return today.isoformat()


def Main(argv=None):
    os.chdir(ROOT)  # data/ paths are repo-relative
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nfl", "cfb"], default="nfl")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today ET if unfinished games, else next slate)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-props", action="store_true")
    a = ap.parse_args(argv)
    a.date = a.date or NextSlate(a.league)
    b = Build(a.league, a.date, a.out_dir, with_props=not a.no_props)
    n_p = sum(len(g["props"]) for g in b["games"])
    print(f"board {a.league} {a.date}: {len(b['games'])} games, {n_p} props, runtime {b['runtime_s']}s, errors {len(b['errors'])}")
    print(os.path.join(a.out_dir, "board_latest.json" if a.league == "nfl" else f"board_{a.league}_latest.json"))


if __name__ == "__main__":
    Main()
