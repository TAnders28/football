r"""
Daily pre-kickoff check (NFL game days only): Kalshi moneylines scored with the TESTED model only.

    .venv\Scripts\python.exe -m Backtest.DailyCheck --date YYYY-MM-DD [--refresh] [--out-dir DIR]

What it does (read-only, no orders, no prediction-log writes):
  1. Loads the day's NFL slate (data/backtest/games_<season>.json; --refresh re-pulls the nflverse schedule
     via PredictDay.LoadSchedule). No games that day -> writes a one-line "PASS - no NFL games".
  2. Pulls open Kalshi KXNFLGAME markets (public API): yes bid/ask per team, mid, de-vigged home mid.
  3. Scores each not-yet-started game with p_full2_kmid_T60 (the walk-forward model tested in
     2026-11-13_bankroll-10k and 2026-11-14_prop-pricing): p_full2 features from PredictDay.SlateRows,
     weights refit on 2013..season-1 (KalshiReality.RefitWeights), market logit = Kalshi de-vigged mid.
  4. A play is listed only if model_prob - (mid + 0.02 tim_est) >= 0.03. Ask-based and Kalshi-fee edges shown too.
  5. Writes E:\AgentDocs\American-Footbot\daily-check\<date>.md (+ .json). One line "PASS" when nothing clears.

Props: NOT scored. No prop model passed the 2026-11-14 holdout (see 2026-11-14_prop-pricing), so per Tim's
rule only tested models are used. The moneyline model is "tested", not "proven": its holdout edge CI includes 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")
OUT_DIR = os.path.join("E:\\AgentDocs", "American-Footbot", "daily-check")
FEE = 0.02
PLAY_EDGE = 0.03
MODEL = "p_full2_kmid_T60"
PROPS_STATUS = "not covered: no prop model passed the 2026-11-14 holdout (2026-11-14_prop-pricing)"


def KalshiFee(p):
    return 0.07 * p * (1 - p)


def KalshiMoneylines(api):
    out, cursor = [], ""
    for _ in range(10):
        p = {"series_ticker": "KXNFLGAME", "status": "open", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        body = api.Get("/markets", p) or {}
        out += body.get("markets") or []
        cursor = body.get("cursor") or ""
        if not cursor:
            break
    return out


def Dollars(m, key):
    v = m.get(f"{key}_dollars")
    if v is None:
        v = m.get(key)
        if v is None:
            return None
        v = float(v) / (100.0 if float(v) > 1 else 1.0)
    return float(v)


def Check(date, refresh=False, out_dir=OUT_DIR, now=None, api=None):
    from Backtest.KalshiReality import RefitWeights
    from Backtest.Lab import AllRows, AttachModels, LoadSeason, Sig, Vec, WIN_FEATS2
    from Backtest.MarketBlend import KalshiApi
    from Backtest.PredictDay import LoadSchedule, PickSlate, SeasonOf, SlateRows
    from Backtest.SnapshotPrices import KalshiEventMatch, Kickoff, SideTeam
    t0 = datetime.now(timezone.utc)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    season = SeasonOf(date)
    sched = LoadSchedule(season, refresh)
    games = sched["games"]
    slate = PickSlate(games, date, False)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, date)
    gen = now.astimezone(ET).strftime("%Y-%m-%d %I:%M %p ET")
    result = {"date": date, "generated_at": gen, "model": MODEL, "fee_basis": "tim_est_0.02", "play_edge": PLAY_EDGE,
              "props": PROPS_STATUS, "games": [], "plays": [], "errors": []}
    if not slate:
        line = f"PASS - no NFL games on {date} (checked {gen})."
        result["summary"] = line
    else:
        prior = LoadSeason(season - 1)
        rows, _last, _n = SlateRows(games, prior, slate)
        rbs = AllRows(2012, season - 1)
        rbs[season] = rows
        AttachModels(rbs)
        w2 = RefitWeights(rbs, season)[0]
        api = api or KalshiApi(pause=0.15)
        try:
            mk = KalshiMoneylines(api)
        except Exception as exc:
            mk = []
            result["errors"].append(f"kalshi markets: {type(exc).__name__}: {exc}")
        by_game = {}
        for r in rows:
            g = next(x for x in slate if x["game_id"] == r["game_id"])
            ko = Kickoff(g)
            rec = {"game_id": r["game_id"], "matchup": f"{r['away_team']} @ {r['home_team']}",
                   "kickoff_et": ko.astimezone(ET).strftime("%a %I:%M %p ET"), "started": ko <= now, "quotes": {}}
            for m in mk:
                if KalshiEventMatch(m.get("event_ticker") or "", g):
                    side = SideTeam(m["ticker"], g)
                    b, a = Dollars(m, "yes_bid"), Dollars(m, "yes_ask")
                    if b and a and 0 < b < a < 1:
                        rec["quotes"][side] = {"bid": b, "ask": a, "mid": (a + b) / 2, "ticker": m["ticker"]}
            by_game[r["game_id"]] = (r, rec)
        for gid, (r, rec) in by_game.items():
            qh, qa = rec["quotes"].get(r["home_team"]), rec["quotes"].get(r["away_team"])
            if rec["started"]:
                rec["status"] = "started - not scored"
            elif not (qh and qa):
                rec["status"] = "no two-sided Kalshi quote for both teams"
            else:
                pk = qh["mid"] / (qh["mid"] + qa["mid"])
                saved = r["mkt_logit"]
                r["mkt_logit"] = math.log(pk / (1 - pk))
                p_home = float(Sig(np.dot(w2, Vec(r, WIN_FEATS2))))
                r["mkt_logit"] = saved
                rec.update({"kalshi_home_devig": round(pk, 4), "p_home": round(p_home, 4), "status": "scored"})
                for side, p, q in ((r["home_team"], p_home, qh), (r["away_team"], 1 - p_home, qa)):
                    e_mid = p - q["mid"] - FEE
                    e_ask = p - q["ask"] - FEE
                    e_k = p - q["ask"] - KalshiFee(q["ask"])
                    rec[f"edge_mid_{side}"] = round(e_mid, 4)
                    if e_mid >= PLAY_EDGE:
                        result["plays"].append({"game_id": gid, "matchup": rec["matchup"], "kickoff_et": rec["kickoff_et"],
                                                "side": side, "model_prob": round(p, 4), "mid": q["mid"], "ask": q["ask"],
                                                "edge_mid_tim": round(e_mid, 4), "edge_ask_tim": round(e_ask, 4),
                                                "edge_ask_kalshi_fee": round(e_k, 4), "ticker": q["ticker"]})
            result["games"].append(rec)
        n_scored = sum(1 for g in result["games"] if g.get("status") == "scored")
        if result["plays"]:
            result["summary"] = f"{len(result['plays'])} moneyline play(s) clear +3c at mid on {date} ({n_scored} games scored; props not covered)."
        else:
            result["summary"] = f"PASS - no moneyline clears +3c fee-adjusted edge on {date} ({n_scored}/{len(rows)} games scored; props not covered)."
    result["runtime_s"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    L = [result["summary"], "",
         f"Generated {gen}. Model {MODEL} (tested 2025-26; holdout edge not proven, CI includes 0). "
         f"Fill = Kalshi mid; edge = model - (mid + $0.02 Tim's fee estimate); play threshold +3c. Props: {PROPS_STATUS}.", ""]
    if result["plays"]:
        L += ["| game | kickoff | side | model | mid | ask | edge mid (tim) | edge ask (tim) | edge ask (Kalshi fee) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p in result["plays"]:
            L.append(f"| {p['matchup']} | {p['kickoff_et']} | {p['side']} | {p['model_prob']:.3f} | {p['mid']:.3f} | {p['ask']:.3f} | "
                     f"{p['edge_mid_tim'] * 100:+.1f}c | {p['edge_ask_tim'] * 100:+.1f}c | {p['edge_ask_kalshi_fee'] * 100:+.1f}c |")
        L += ["", "Simulation-grade signal only. Tim places any wager; no orders are sent."]
    if result["games"]:
        L += ["", "| game | kickoff | status | Kalshi home (de-vig) | model home |", "|---|---|---|---|---|"]
        for g in result["games"]:
            L.append(f"| {g['matchup']} | {g['kickoff_et']} | {g['status']} | {g.get('kalshi_home_devig', 'n/a')} | {g.get('p_home', 'n/a')} |")
    if result["errors"]:
        L += ["", "Errors: " + "; ".join(result["errors"])]
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    return result


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(ET).date().isoformat())
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    res = Check(a.date, refresh=a.refresh, out_dir=a.out_dir)
    print(res["summary"])
    print(os.path.join(a.out_dir, a.date + ".md"))


if __name__ == "__main__":
    Main()
