r"""
Live in-game read for one game: our UNVALIDATED in-game win probability (Backtest.InGameWP, ingame_wp_v1) next to live
Kalshi / Polymarket executable bid/ask, with fee-adjusted edge, appended to Degen's live file. Read-only, no orders.

    .venv\Scripts\python.exe -m Backtest.LiveInGame --game 2026_11_LAC_BAL [--live-root DIR] [--state-json FILE] [--no-file]

Game state: ESPN public summary JSON (site.api.espn.com .../summary?event=<espn id>): the latest play's end state
(possession team, down, distance, yardsToEndzone, score, period, clock, wallclock) + scoreboard situation for timeouts.
--state-json replaces ESPN with a hand-written state (testing): {"home_score","away_score","period","clock","possession":
"home"|"away","down","distance","yards_to_endzone","home_timeouts","away_timeouts","opening_receiver":"home"|"away","state_time"}.
Quotes and staleness come from Backtest.AssessLive (same rules as the existing live file: executable book only, 45 s live
age, 60 s price/state gap). Rows are LiveFile.QuoteRow rows plus model_prob_unvalidated / edge_unvalidated_* fields.
model_prob stays null and action stays PASS ("no validated live model") until held-out calibration matches or beats the
markets AND Tim signs off; see E:\AgentDocs\American-Footbot\2026-11-13_ingame-model\.
Fees: edge_unvalidated_tim_est = p - ask - 0.02 (Tim's estimate); edge_unvalidated_venue_fee uses Kalshi's verified
0.07*p*(1-p) or Polymarket's 0.05*p*(1-p) (uncertain for NFL). No scheduling here.
"""

import argparse
import json
import os
import socket
import urllib.request
from datetime import datetime, timezone

from Backtest.AssessLive import AssessLive
from Backtest.InGameWP import LoadModel, MODEL_VERSION, PredictPos, StateFeatures
from Backtest.Lab import AllRows, AttachModels, LoadSeason
from Backtest.LiveFile import LIVE_ROOT, LatestMarkdown, QuoteRow

_GAI = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *a, **k: _GAI(host, port, socket.AF_INET if family == 0 else family, *a, **k)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
MODEL_PATH = os.path.join("data", "models", "ingame_wp_model_live.json")
VALIDATED = False   # flip only after held-out calibration matches/beats both markets and Tim approves
VENUE_FEE = {"kalshi": lambda p: 0.07 * p * (1 - p), "polymarket": lambda p: 0.05 * p * (1 - p)}
ALIASES = {"LA": {"LA", "LAR"}, "JAX": {"JAX", "JAC"}, "WAS": {"WAS", "WSH"}, "LV": {"LV", "LVR", "OAK"}, "LAC": {"LAC", "SD"}}


def Same(a, b):
    return a == b or b in ALIASES.get(a, set()) or a in ALIASES.get(b, set())


def GetJson(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def ClockSecs(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + float(sec)
    except (ValueError, AttributeError):
        return None


def EspnState(summary, scoreboard_event=None):
    comp = summary["header"]["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    home_id, away_id = teams["home"]["team"]["id"], teams["away"]["team"]["id"]
    status = comp.get("status") or {}
    stype = (status.get("type") or {}).get("state")
    drives = summary.get("drives") or {}
    plays = [p for d in (drives.get("previous") or []) for p in d.get("plays") or []]
    if drives.get("current"):
        plays += drives["current"].get("plays") or []
    plays.sort(key=lambda p: int(p.get("sequenceNumber") or 0))
    last = next((p for p in reversed(plays) if (p.get("end") or {}).get("team")), None)
    first_drive = (drives.get("previous") or [{}])[0]
    opening = (first_drive.get("team") or {}).get("id") if first_drive else None
    sit = (scoreboard_event or {}).get("competitions", [{}])[0].get("situation") or {}
    period = status.get("period") or ((last or {}).get("period") or {}).get("number")
    clock = status.get("displayClock") or ((last or {}).get("clock") or {}).get("displayValue")
    st = {"phase": {"pre": "pregame", "in": "live", "post": "final"}.get(stype, stype),
          "home_score": int(teams["home"].get("score") or 0), "away_score": int(teams["away"].get("score") or 0),
          "period": period, "clock": clock, "state_time": (last or {}).get("wallclock"), "source": "espn_summary",
          "home_timeouts": sit.get("homeTimeouts"), "away_timeouts": sit.get("awayTimeouts")}
    if last:
        end = last["end"]
        st.update({"possession": "home" if end["team"]["id"] == home_id else "away", "down": end.get("down") or None,
                   "distance": end.get("distance"), "yards_to_endzone": end.get("yardsToEndzone")})
    st["opening_receiver"] = None if opening is None else ("home" if opening == home_id else "away")
    return st


def ModelProbHome(st, prior_home, mj):
    """Our UNVALIDATED in-game P(home wins) from a state dict, or (None, reason)."""
    if st.get("phase") != "live":
        return None, f"phase {st.get('phase')}"
    if st.get("possession") not in ("home", "away"):
        return None, "no possession in state"
    secs, per = ClockSecs(st.get("clock")), int(st.get("period") or 0)
    if secs is None or per < 1:
        return None, "no clock"
    gsr = (4 - per) * 900 + secs if per <= 4 else secs
    hsr = (900 + secs if per in (1, 3) else secs) if per <= 4 else secs
    home = st["possession"] == "home"
    diff = (st["home_score"] - st["away_score"]) * (1 if home else -1)
    to_h, to_a = st.get("home_timeouts"), st.get("away_timeouts")
    s = {"diff": diff, "gsr": gsr, "hsr": hsr, "qtr": per, "down": st.get("down"), "ydstogo": st.get("distance"),
         "yardline_100": st.get("yards_to_endzone"), "pos_to": to_h if home else to_a, "def_to": to_a if home else to_h,
         "is_home": home, "receive_2h": st.get("opening_receiver") is not None and st["opening_receiver"] != st["possession"]}
    p_pos = float(PredictPos(mj, [StateFeatures(s, prior_home)])[0])
    return (p_pos if home else 1 - p_pos), None


def PregamePrior(g, season):
    """Our leak-free pregame p_full2 for this game, built exactly as PredictDay does for unplayed games
    (features from games finished on earlier days; model fitted on 2013..season-1). Read-only reuse."""
    from Backtest.PredictDay import SlateRows
    rows, _, _ = SlateRows(LoadSeason(season), LoadSeason(season - 1), [g])
    rbs = AllRows(2012, season - 1)
    rbs[season] = rows
    AttachModels(rbs)
    return rows[0].get("p_full2") if rows and rows[0].get("has_odds") else None


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="nflverse game_id, e.g. 2026_11_LAC_BAL")
    ap.add_argument("--live-root", default=LIVE_ROOT)
    ap.add_argument("--state-json", default=None)
    ap.add_argument("--no-file", action="store_true")
    a = ap.parse_args()
    season = int(a.game[:4])
    g = next(x for x in LoadSeason(season) if x["game_id"] == a.game)
    prior = PregamePrior(g, season)
    mj = LoadModel(MODEL_PATH)
    if a.state_json:
        st = json.load(open(a.state_json, encoding="utf-8"))
        st.setdefault("phase", "live"); st.setdefault("source", "manual_state_json")
    else:
        summ = GetJson(f"{ESPN}/summary?event={g['espn']}")
        sb = GetJson(f"{ESPN}/scoreboard?dates={g['gameday'].replace('-', '')}")
        ev = next((e for e in sb.get("events") or [] if str(e.get("id")) == str(g["espn"])), None)
        st = EspnState(summ, ev)
    p_home, why = ModelProbHome(st, prior, mj)

    # AssessLive --matchup misses some games (e.g. LAC@BAL on 2026-11-16 returned 0), so read the whole date and filter here.
    report = AssessLive(date=g["gameday"], write=False, live_file=False)
    report["games"] = [x for x in report.get("games") or [] if Same(g["home_team"], x.get("home") or "") and Same(g["away_team"], x.get("away") or "")]
    rows = []
    run_id = "liveingame_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for game in report.get("games") or []:
        for q in game.get("quotes") or []:
            r = QuoteRow(game, q, report["meta"]["date_et"], q.get("observed_at_utc") or report["meta"]["generated_at_utc"], run_id, MODEL_VERSION)
            side = q.get("side_team")
            mp = None
            if p_home is not None and side:
                mp = round(p_home if Same(g["home_team"], side) or Same(side, game.get("home")) else 1 - p_home, 4)
            ask = r.get("ask")
            r["model_prob_unvalidated"] = mp
            r["model_prob_unvalidated_note"] = why or f"{MODEL_VERSION}, UNVALIDATED; pregame prior p_full2={prior}"
            r["edge_unvalidated_tim_est"] = None if mp is None or ask is None else round(mp - ask - 0.02, 4)
            vf = VENUE_FEE.get((r.get("venue") or "").lower())
            r["edge_unvalidated_venue_fee"] = None if mp is None or ask is None or vf is None else round(mp - ask - vf(ask), 4)
            r["live_model_state"] = {k: st.get(k) for k in ("source", "phase", "period", "clock", "home_score", "away_score", "possession",
                                                            "down", "distance", "yards_to_endzone", "home_timeouts", "away_timeouts", "state_time")}
            if not VALIDATED:
                r["model_prob"], r["edge"], r["action"] = None, None, "PASS"
                if r["reason"] in ("no live model", "edge <= 0 after Tim's fee estimate", "edge > 0 after Tim's fee estimate"):
                    r["reason"] = "no validated live model (ingame_wp_v1 unvalidated)"
            rows.append(r)
    out = {"game": a.game, "espn": g["espn"], "p_home_unvalidated": p_home, "why_none": why, "prior_p_full2": prior, "state": st, "rows": len(rows), "games_on_board": len(report.get("games") or []), "source_errors": (report.get("meta") or {}).get("source_errors")}
    if not a.no_file:
        date = report["meta"]["date_et"]
        folder = os.path.join(a.live_root, f"{date}_live")
        os.makedirs(folder, exist_ok=True)
        jl = os.path.join(folder, f"{date}_live.jsonl")
        with open(jl, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        md = LatestMarkdown(report, rows, run_id, MODEL_VERSION)
        md += "\n## Our in-game model (ingame_wp_v1, UNVALIDATED - action stays PASS)\n\n"
        md += f"State ({st.get('source')}, {st.get('state_time')}): Q{st.get('period')} {st.get('clock')}, {g['away_team']} {st.get('away_score')} - {g['home_team']} {st.get('home_score')}, ball {st.get('possession')}, down {st.get('down')} & {st.get('distance')}, {st.get('yards_to_endzone')} yds to goal.\n\n"
        md += "| Venue | Side | Bid / Ask | Our p (unvalidated) | Edge, Tim's $0.02 est | Edge, venue fee | Action |\n|---|---|---|---|---|---|---|\n"
        for r in rows:
            md += f"| {r['venue']} | {r['side']} | {r['bid']} / {r['ask']} | {r['model_prob_unvalidated']} | {r['edge_unvalidated_tim_est']} | {r['edge_unvalidated_venue_fee']} | {r['action']} |\n"
        with open(os.path.join(folder, f"{date}_live_latest.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        out["jsonl"], out["latest"] = jl, os.path.join(folder, f"{date}_live_latest.md")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    Main()