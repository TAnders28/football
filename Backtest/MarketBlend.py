"""
Market-blend study: does p_full2_kmid_T60 add anything beyond Kalshi T-60?

    .venv/Scripts/python.exe -m Backtest.MarketBlend --out-dir DIR [--skip-ratings] [--skip-survey]
    .venv/Scripts/python.exe -m Backtest.MarketBlend --out-dir DIR --ratings-only
    .venv/Scripts/python.exe -m Backtest.MarketBlend --out-dir DIR --survey-only

Tasks (simulation / evaluation only; no orders):
  1. Relative Ignorance vs Kalshi T-60 de-vigged mid, in bits per game, with a 90%
     game-cluster bootstrap CI and N, for 2025 (contaminated), 2026 weeks 1-9 (forward),
     and the 20 unseen 2026 games.
  2. Logistic blend logit(p) = a + b_model*logit(model) + b_kalshi*logit(kalshi),
     fitted on prior data only. Kalshi history starts in 2025, so the default is to
     fit on 2025 and apply to 2026 (said explicitly in the report).
  3. Edges off p_blend with the 3c qualify line; flat100 and qkelly from one $10k
     bank (2% cap). PRIMARY fill = mid + tim_est_0.02 (Tim's fill rule and fee
     estimate). SECONDARY fill = ask + tim_est_0.02. Verified Kalshi taker fee is
     reported alongside. Season-axis charts via BankrollPlot.
  4. Prototype yards-per-play and success-rate ratings from local play-by-play
     (season-to-date, earlier days only). One pre-specified variant; trial count logged.
  5. Survey of Kalshi NFL totals and spreads: series, markets, and a light sample of
     public candle history. Hits the network only when --survey / default Main is run.

Do not run this module on real data or the Kalshi API until Tim says to.
Main sets the process to BelowNormal on Windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.BankrollPlot import PlotBankroll
from Backtest.CommonFormat import (
    ALPHA, BANK_FIELDS, BOOT, EDGE_EXTRA, EDGE_FIELDS, FEE, FEE_BASIS, FLAT,
    MIN_EDGE, MODEL, PRICE_KALSHI, RUIN, SEED, START, AllGameDays, AttachKmid,
    BootstrapMean, BootstrapRoi, ClipP, FlatPnl, IsoKickoff, KalshiRows,
    SeasonLabel, SimpleRuin, SplitLabel, Walk, Widest, WriteCsv,
)
from Backtest.ForwardWalk import SeenBySearch
from Backtest.FrozenConfig import FROZEN, IsForwardHoldout
from Backtest.KalshiReality import KalshiFee
from Backtest.Lab import AllRows, AttachModels, FitLogit, Sig, Vec, WIN_FEATS2
from Backtest.Version import CodeVersion, ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
SPORT = "nfl"
BLEND_MODEL = "p_blend_kmid_T60"
LN2 = math.log(2.0)
MIN_RATING_PLAYS = 80
RATINGS_TRIALS = 1
RATINGS_VARIANT = "full2_ypp_sr"
SURVEY_CANDIDATES = (
    "KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTOTALGAME", "KXNFLOU",
    "KXNFLPTS", "KXNFLPOINTSPREAD", "KXNFLGAMETOTAL", "KXNFLSPREADGAME",
)
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
SURVEY_UA = "football-MarketBlend-survey/readonly"


def SetBelowNormal():
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def Logit(p):
    p = ClipP(p)
    return math.log(p / (1.0 - p))


def LogLossNats(p, y):
    p = ClipP(p)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def LogLossBits(p, y):
    return LogLossNats(p, y) / LN2


def MeanLogLoss(rows, key, bits=False):
    vals = []
    for r in rows:
        if r.get(key) is None or r.get("home_win") is None:
            continue
        fn = LogLossBits if bits else LogLossNats
        vals.append(fn(r[key], float(r["home_win"])))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def MeanBrier(rows, key):
    vals = []
    for r in rows:
        if r.get(key) is None or r.get("home_win") is None:
            continue
        p, y = float(r[key]), float(r["home_win"])
        vals.append((p - y) ** 2)
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def IgnRelBits(rows, model_key, market_key):
    """mean(IGN_model) - mean(IGN_market) in bits. Negative means the model beats the market."""
    vals = IgnRelValues(rows, model_key, market_key)
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def IgnRelValues(rows, model_key, market_key):
    out = []
    for r in rows:
        if r.get(model_key) is None or r.get(market_key) is None or r.get("home_win") is None:
            continue
        y = float(r["home_win"])
        out.append(LogLossBits(r[model_key], y) - LogLossBits(r[market_key], y))
    return out


def GrowthPerEvent(ign_rel_bits):
    if ign_rel_bits is None:
        return None
    return float(2.0 ** (-ign_rel_bits))


def BootstrapIgnRel(rows, model_key, market_key, rng, boot=BOOT, alpha=ALPHA):
    """Game-cluster bootstrap (one probability per game). Returns (lo, hi, n)."""
    vals = np.asarray(IgnRelValues(rows, model_key, market_key), dtype=float)
    n = int(len(vals))
    if n == 0:
        return None, None, 0
    lo, hi = BootstrapMean(vals, rng, boot=boot, alpha=alpha)
    return lo, hi, n


def GapBlock(rows, model_key, market_key, season, split, rng, boot=BOOT):
    point, n = IgnRelBits(rows, model_key, market_key)
    lo, hi, n_ci = BootstrapIgnRel(rows, model_key, market_key, rng, boot=boot)
    ll_m, _ = MeanLogLoss(rows, model_key, bits=True)
    ll_k, _ = MeanLogLoss(rows, market_key, bits=True)
    ll_m_nats, _ = MeanLogLoss(rows, model_key, bits=False)
    ll_k_nats, _ = MeanLogLoss(rows, market_key, bits=False)
    return {
        "season": season, "split": split, "n": n,
        "ign_rel_bits": None if point is None else round(point, 6),
        "ign_rel_bits_ci90": None if lo is None else [round(lo, 6), round(hi, 6)],
        "growth_per_event": None if point is None else round(GrowthPerEvent(point), 6),
        "model_log_loss_bits": None if ll_m is None else round(ll_m, 6),
        "market_log_loss_bits": None if ll_k is None else round(ll_k, 6),
        "model_log_loss_nats": None if ll_m_nats is None else round(ll_m_nats, 6),
        "market_log_loss_nats": None if ll_k_nats is None else round(ll_k_nats, 6),
        "n_bootstrap": n_ci,
    }


def SplitRows(kalshi_rows):
    y25 = [r for r in kalshi_rows if int(r["season"]) == 2025]
    y26 = [r for r in kalshi_rows if int(r["season"]) == 2026 and int(r["week"]) <= 9]
    unseen = [r for r in y26 if not SeenBySearch(r)]
    return {"2025": y25, "2026": y26, "2026_unseen": unseen}


def BlendVec(r, model_key="p_kmid", market_key="kmid"):
    return [1.0, Logit(r[model_key]), Logit(r[market_key])]


def FitBlend(rows, model_key="p_kmid", market_key="kmid", l2=1.0):
    train = [r for r in rows
             if r.get("home_win") is not None and r.get(model_key) is not None and r.get(market_key) is not None]
    if len(train) < 8:
        return None
    X = [BlendVec(r, model_key, market_key) for r in train]
    y = [float(r["home_win"]) for r in train]
    w = FitLogit(X, y, l2=l2)
    return {"a": float(w[0]), "b_model": float(w[1]), "b_kalshi": float(w[2]), "n": len(train), "l2": l2}


def ApplyBlend(rows, weights, model_key="p_kmid", market_key="kmid", out_key="p_blend"):
    if not weights:
        for r in rows:
            r[out_key] = None
        return rows
    for r in rows:
        if r.get(model_key) is None or r.get(market_key) is None:
            r[out_key] = None
            continue
        z = (weights["a"] + weights["b_model"] * Logit(r[model_key])
             + weights["b_kalshi"] * Logit(r[market_key]))
        r[out_key] = float(Sig(z))
    return rows


def BootstrapBlendWeights(rows, rng, boot=BOOT, alpha=ALPHA, model_key="p_kmid", market_key="kmid", l2=1.0):
    train = [r for r in rows
             if r.get("home_win") is not None and r.get(model_key) is not None and r.get(market_key) is not None]
    n = len(train)
    empty = {"a": (None, None), "b_model": (None, None), "b_kalshi": (None, None), "n_ok": 0, "n": n}
    if n < 8:
        return empty
    recs = []
    idx = rng.integers(0, n, size=(boot, n))
    for i in range(boot):
        sample = [train[j] for j in idx[i]]
        try:
            fit = FitBlend(sample, model_key=model_key, market_key=market_key, l2=l2)
        except np.linalg.LinAlgError:
            fit = None
        if fit:
            recs.append(fit)
    if not recs:
        return empty

    def ci(key):
        arr = np.array([r[key] for r in recs], dtype=float)
        return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))

    return {"a": ci("a"), "b_model": ci("b_model"), "b_kalshi": ci("b_kalshi"), "n_ok": len(recs), "n": n}


def FrozenSeasonBlend(train_rows, apply_rows, model_key="p_kmid", market_key="kmid", l2=1.0, out_key="p_blend"):
    """Fit on train_rows only, apply to apply_rows. No outcomes from apply_rows enter the fit."""
    weights = FitBlend(train_rows, model_key=model_key, market_key=market_key, l2=l2)
    ApplyBlend(apply_rows, weights, model_key=model_key, market_key=market_key, out_key=out_key)
    return weights


def DayClusterRoi(bet_log, rng, boot=BOOT, alpha=ALPHA):
    by_day = defaultdict(list)
    for b in bet_log:
        by_day[b["date"]].append(b)
    days = list(by_day)
    if not days:
        return None, None
    pnl_d, st_d = [], []
    for d in days:
        pnl_d.append(sum(b["pnl"] for b in by_day[d]))
        st_d.append(sum(b["stake"] for b in by_day[d]))
    return BootstrapRoi(pnl_d, st_d, rng, boot=boot, alpha=alpha)


def WidestRoi(bet_log, rng, boot=BOOT):
    ev = BootstrapRoi([b["pnl"] for b in bet_log], [b["stake"] for b in bet_log], rng, boot=boot)
    day = DayClusterRoi(bet_log, rng, boot=boot)
    return Widest(ev, day), ev, day


def EdgeClaimable(roi_ci_tim, roi_ci_kalshi):
    """Claim only if both fee bases have a lower 90% bound strictly above 0."""
    if not roi_ci_tim or not roi_ci_kalshi or roi_ci_tim[0] is None or roi_ci_kalshi[0] is None:
        return False
    return roi_ci_tim[0] > 0 and roi_ci_kalshi[0] > 0


def SideFromPair(p_home, p_away, price_home, price_away, home_team, away_team, home_win, fee=FEE):
    e_home = p_home - price_home - fee
    e_away = p_away - price_away - fee
    if e_home >= e_away:
        return home_team, p_home, price_home, int(home_win), e_home
    return away_team, p_away, price_away, int(1 - home_win), e_away


def BlendEdge(r, fill_kind="mid", p_key="p_blend"):
    """One edge row for p_blend. fill_kind is 'mid' (primary) or 'ask' (secondary)."""
    qh, qa = r["kq_t60"]["home"], r["kq_t60"]["away"]
    p_home, p_away = r[p_key], 1.0 - r[p_key]
    home_mkt, away_mkt = r["kmid"], 1.0 - r["kmid"]
    if fill_kind == "ask":
        price_home, price_away = qh["ask"], qa["ask"]
        price_time = qh["price_time"] if (p_home - qh["ask"]) >= (p_away - qa["ask"]) else qa["price_time"]
        basis = "kalshi_T60_ask"
    else:
        price_home, price_away = qh["mid"], qa["mid"]
        price_time = qh["price_time"] if (p_home - home_mkt) >= (p_away - away_mkt) else qa["price_time"]
        basis = PRICE_KALSHI
    if fill_kind == "ask":
        side, model_p, fill, result, _e = SideFromPair(
            p_home, p_away, price_home, price_away, r["home_team"], r["away_team"], r["home_win"])
        market_p = home_mkt if side == r["home_team"] else away_mkt
    else:
        side, model_p, fill, result, _e = SideFromPair(
            p_home, p_away, home_mkt, away_mkt, r["home_team"], r["away_team"], r["home_win"])
        market_p = fill if fill_kind != "mid" else (home_mkt if side == r["home_team"] else away_mkt)
        fill = price_home if side == r["home_team"] else price_away
        market_p = home_mkt if side == r["home_team"] else away_mkt
    if home_mkt >= away_mkt:
        fav_fill, fav_result = (qh["ask"] if fill_kind == "ask" else qh["mid"]), int(r["home_win"])
    else:
        fav_fill, fav_result = (qa["ask"] if fill_kind == "ask" else qa["mid"]), int(1 - r["home_win"])
    fee_k = KalshiFee(fill)
    edge = model_p - market_p - FEE
    edge_fill = model_p - fill - FEE
    qualifies = bool(edge_fill >= MIN_EDGE) if fill_kind == "ask" else bool(edge >= MIN_EDGE)
    pnl = FlatPnl(fill, result, FEE)
    pnl_k = FlatPnl(fill, result, fee_k)
    return {
        "date": r["gameday"], "start_time": IsoKickoff(r), "sport": SPORT,
        "season": SeasonLabel(r), "split": SplitLabel(r), "event_id": r["game_id"],
        "event": f"{r['away_team']} @ {r['home_team']}", "side": side, "model": BLEND_MODEL,
        "model_prob": round(model_p, 6), "market_prob_devig": round(market_p, 6),
        "fill_price": round(fill, 6), "price_basis": basis, "price_time": price_time,
        "fee": FEE, "fee_basis": FEE_BASIS, "edge": round(edge, 6),
        "edge_vs_fill": round(edge_fill, 6), "qualifies": qualifies,
        "result": int(result), "pnl_at_flat_stake": None if pnl is None else round(pnl, 4),
        "fee_kalshi_verified": None if fee_k is None else round(fee_k, 6),
        "edge_vs_fill_kalshi_fee": None if fee_k is None else round(model_p - fill - fee_k, 6),
        "pnl_at_flat_stake_kalshi_fee": None if pnl_k is None else round(pnl_k, 4),
        "fav_side_fill": round(fav_fill, 6), "fav_result": int(fav_result),
        "seen_by_search": bool(SeenBySearch(r)), "forward_holdout": bool(IsForwardHoldout(r["gameday"])),
        "_p": model_p, "_fill": fill, "_fee_k": fee_k,
        "_fav_fill": fav_fill, "_fav_result": fav_result,
        "fill_kind": fill_kind,
    }


def BlendEdges(rows, fill_kind="mid", p_key="p_blend"):
    edges = [BlendEdge(r, fill_kind=fill_kind, p_key=p_key) for r in rows if r.get(p_key) is not None and r.get("kq_t60")]
    edges.sort(key=lambda e: (e["start_time"], e["event_id"]))
    return edges


def BankStats(daily, bet_log, rng, strategy):
    pnls = [b["pnl"] for b in bet_log]
    stakes = [b["stake"] for b in bet_log]
    staked = sum(stakes)
    pnl = sum(pnls)
    roi = (pnl / staked) if staked else None
    widest, ev, day = WidestRoi(bet_log, rng)
    end = daily[-1]["equity"] if daily else START
    dd = min((d["drawdown_pct"] for d in daily), default=0.0)
    return {
        "strategy": strategy, "n_bets": len(bet_log), "final_equity": end,
        "staked": round(staked, 2), "pnl": round(pnl, 2),
        "roi": roi,
        "roi_ci90_event": None if ev[0] is None else [ev[0], ev[1]],
        "roi_ci90_day": None if day[0] is None else [day[0], day[1]],
        "roi_ci90_widest": None if widest[0] is None else [widest[0], widest[1]],
        "max_drawdown_pct": dd, "ruined": SimpleRuin(daily) or (end < RUIN),
    }


def RunBankroll(edges, days, rng, fee_kind="tim"):
    out = {}
    dailies = {}
    logs = {}
    for strat in ("flat100", "qkelly", "market_fav"):
        daily, log, ruined, _end = Walk(edges, strat, fee_kind=fee_kind, days=days)
        stats = BankStats(daily, log, rng, strat)
        stats["ruined"] = bool(stats["ruined"] or ruined)
        out[strat] = stats
        dailies[strat] = daily
        logs[strat] = log
    return out, dailies, logs
def IsScrimmagePlay(p):
    if p.get("pass") in (1, 1.0) or p.get("rush") in (1, 1.0):
        return True
    return p.get("play_type") in ("pass", "run", "rush")


def PlayYards(p):
    y = p.get("yards_gained")
    if y is None:
        return None
    try:
        return float(y)
    except (TypeError, ValueError):
        return None


def PlaySuccess(p):
    s = p.get("success")
    if s in (1, 1.0, True):
        return 1
    if s in (0, 0.0, False):
        return 0
    yards, down, togo = PlayYards(p), p.get("down"), p.get("ydstogo")
    if yards is None or not down or not togo:
        return None
    try:
        down, togo = int(down), float(togo)
    except (TypeError, ValueError):
        return None
    if togo <= 0:
        return 1 if yards > 0 else 0
    if down == 1:
        need = 0.5 * togo
    elif down == 2:
        need = 0.7 * togo
    else:
        need = togo
    return 1 if yards >= need else 0


def GameStamp(p):
    return {
        "game_id": p.get("game_id"),
        "season": p.get("season"),
        "week": p.get("week"),
        "gameday": str(p.get("game_date") or p.get("gameday") or "")[:10],
        "home_team": p.get("home_team"),
        "away_team": p.get("away_team"),
    }


def AccruedGameStats(plays):
    """One-pass game-level offense totals from scrimmage plays that have yards_gained."""
    units = {}
    meta = {}
    skipped_no_yards = 0
    kept = 0
    for p in plays:
        if not IsScrimmagePlay(p):
            continue
        team = p.get("posteam")
        if not team:
            continue
        yards = PlayYards(p)
        if yards is None:
            skipped_no_yards += 1
            continue
        succ = PlaySuccess(p)
        if succ is None:
            skipped_no_yards += 1
            continue
        gid = p.get("game_id")
        key = (gid, team)
        u = units.get(key)
        if u is None:
            u = units[key] = {"plays": 0, "yards": 0.0, "succ": 0, "opp": p.get("defteam")}
            meta[key] = GameStamp(p)
        u["plays"] += 1
        u["yards"] += yards
        u["succ"] += succ
        kept += 1
    rows = []
    for (gid, team), u in units.items():
        m = meta[(gid, team)]
        opp = u["opp"]
        d = units.get((gid, opp), {"plays": 0, "yards": 0.0, "succ": 0})
        rows.append({
            "game_id": gid, "season": m["season"], "week": m["week"], "gameday": m["gameday"],
            "team": team, "opponent": opp, "home_team": m["home_team"], "away_team": m["away_team"],
            "off_plays": u["plays"], "off_yards": u["yards"], "off_succ": u["succ"],
            "def_plays": d["plays"], "def_yards": d["yards"], "def_succ": d["succ"],
        })
    rows.sort(key=lambda r: (str(r["season"] or ""), r["gameday"] or "", r["game_id"] or "", r["team"] or ""))
    return rows, {"kept_plays": kept, "skipped_no_yards": skipped_no_yards, "team_games": len(rows)}


def PriorStdRatings(game_stats, min_plays=MIN_RATING_PLAYS):
    """Season-to-date ratings using only earlier gamedays in the same season (day barrier)."""
    by_team = defaultdict(list)
    for g in game_stats:
        by_team[(g["season"], g["team"])].append(g)
    for lst in by_team.values():
        lst.sort(key=lambda r: (r["gameday"] or "", r["game_id"] or ""))
    out = {}
    for g in game_stats:
        prior = [x for x in by_team[(g["season"], g["team"])]
                 if (x["gameday"] or "") < (g["gameday"] or "")]
        op = sum(x["off_plays"] for x in prior)
        oy = sum(x["off_yards"] for x in prior)
        os_ = sum(x["off_succ"] for x in prior)
        dp = sum(x["def_plays"] for x in prior)
        dy = sum(x["def_yards"] for x in prior)
        ds = sum(x["def_succ"] for x in prior)
        ready = op >= min_plays and dp >= min_plays
        off_ypp = (oy / op) if op else None
        def_ypp = (dy / dp) if dp else None
        off_sr = (os_ / op) if op else None
        def_sr = (ds / dp) if dp else None
        rec = {
            "off_ypp": off_ypp, "def_ypp": def_ypp, "off_sr": off_sr, "def_sr": def_sr,
            "off_plays": op, "def_plays": dp, "prior_games": len(prior), "ready": ready,
            "net_ypp": None if not ready or off_ypp is None or def_ypp is None else off_ypp - def_ypp,
            "net_sr": None if not ready or off_sr is None or def_sr is None else off_sr - def_sr,
        }
        out[(g["game_id"], g["team"])] = rec
    return out


def AttachRatings(rows, ratings, min_plays=MIN_RATING_PLAYS):
    """Add ypp_diff and sr_diff (home net - away net). Missing / not-ready -> None."""
    n_ready = 0
    for r in rows:
        h = ratings.get((r["game_id"], r["home_team"]))
        a = ratings.get((r["game_id"], r["away_team"]))
        r["ypp_diff"] = r["sr_diff"] = None
        r["ratings_ready"] = False
        r["ratings_home_plays"] = 0 if not h else h["off_plays"]
        r["ratings_away_plays"] = 0 if not a else a["off_plays"]
        if h and a and h.get("ready") and a.get("ready"):
            r["ypp_diff"] = h["net_ypp"] - a["net_ypp"]
            r["sr_diff"] = h["net_sr"] - a["net_sr"]
            r["ratings_ready"] = True
            n_ready += 1
    return n_ready


def VecPlus(r, extra):
    base = Vec(r, WIN_FEATS2)
    for f in extra:
        val = r.get(f)
        base.append(0.0 if val is None else float(val))
    return base


def FitRatingsModel(rows_by_season, extra=("ypp_diff", "sr_diff"), l2=5.0, min_season=2023):
    """Walk-forward by season: fit on earlier seasons only, only games with ratings_ready."""
    coefs = {}
    for s, rows in rows_by_season.items():
        train = [r for t, rs in rows_by_season.items() if min_season <= t < s for r in rs
                 if r.get("home_win") is not None and r.get("ratings_ready") and r.get("p_full2") is not None]
        w = None
        if len(train) >= 40:
            w = FitLogit([VecPlus(r, extra) for r in train], [float(r["home_win"]) for r in train], l2=l2)
            coefs[s] = w.tolist()
        for r in rows:
            if w is None or not r.get("ratings_ready"):
                r["p_full2_yppsr"] = None
            else:
                r["p_full2_yppsr"] = float(Sig(np.dot(w, VecPlus(r, extra))))
    return coefs


def RatingsGain(rows, model_key="p_full2_yppsr", base_key="p_full2"):
    both = [r for r in rows if r.get(model_key) is not None and r.get(base_key) is not None and r.get("home_win") is not None]
    if not both:
        return {"n": 0, "d_log_loss_nats": None, "d_brier": None, "d_log_loss_bits": None}
    ll_m, n = MeanLogLoss(both, model_key, bits=False)
    ll_b, _ = MeanLogLoss(both, base_key, bits=False)
    br_m, _ = MeanBrier(both, model_key)
    br_b, _ = MeanBrier(both, base_key)
    ll_m_b, _ = MeanLogLoss(both, model_key, bits=True)
    ll_b_b, _ = MeanLogLoss(both, base_key, bits=True)
    return {
        "n": n,
        "model_log_loss_nats": None if ll_m is None else round(ll_m, 6),
        "base_log_loss_nats": None if ll_b is None else round(ll_b, 6),
        "d_log_loss_nats": None if ll_m is None or ll_b is None else round(ll_m - ll_b, 6),
        "model_brier": None if br_m is None else round(br_m, 6),
        "base_brier": None if br_b is None else round(br_b, 6),
        "d_brier": None if br_m is None or br_b is None else round(br_m - br_b, 6),
        "d_log_loss_bits": None if ll_m_b is None or ll_b_b is None else round(ll_m_b - ll_b_b, 6),
    }


def PbpPath(season, root="data"):
    full = os.path.join(root, "playbyplay", f"pbp_{season}.jsonl")
    if os.path.exists(full):
        return full, "full"
    slim = os.path.join(root, "playbyplay", "slim", f"pbp_slim_{season}.jsonl")
    if os.path.exists(slim):
        return slim, "slim"
    return None, None


def IterJsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def LoadGameStats(seasons, root="data"):
    all_rows, info = [], {}
    for s in seasons:
        path, kind = PbpPath(s, root=root)
        if not path:
            info[s] = {"path": None, "kind": None, "kept_plays": 0, "skipped_no_yards": 0, "team_games": 0}
            continue
        rows, meta = AccruedGameStats(IterJsonl(path))
        meta["path"] = path
        meta["kind"] = kind
        info[s] = meta
        all_rows.extend(rows)
    return all_rows, info


# ----------------------------------------------------------------------------- Kalshi survey (network only when called with a live Api)


class KalshiApi:
    """Thin public-API client. Tests inject a fake with the same Get/Paged methods."""

    def __init__(self, pause=0.12, get=None):
        self.pause, self.calls, self.retries = pause, 0, 0
        self._get = get

    def Get(self, path, params=None):
        if self._get is not None:
            self.calls += 1
            return self._get(path, params)
        import time
        import urllib.error
        import urllib.parse
        import urllib.request
        url = KALSHI_BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(6):
            time.sleep(self.pause)
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": SURVEY_UA})
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

    def Paged(self, path, params, key="markets", max_pages=20):
        out, cursor = [], ""
        for _ in range(max_pages):
            p = dict(params, **({"cursor": cursor} if cursor else {}))
            body = self.Get(path, p) or {}
            chunk = body.get(key) or []
            out += chunk
            cursor = body.get("cursor") or ""
            if not cursor or not chunk:
                break
        return out


def ClassifySeries(ticker, title=""):
    t = f"{ticker} {title}".upper()
    if any(s in t for s in ("TOTAL", "OVER/UNDER", "OVERUNDER", "O/U", "OU", "PTS TOTAL")):
        return "total"
    if "SPREAD" in t or "HANDICAP" in t or "ATS" in t:
        return "spread"
    if "GAME" in t and "WIN" in t:
        return "moneyline"
    if ticker.upper().endswith("GAME") or ticker.upper() == "KXNFLGAME":
        return "moneyline"
    return "other"


def SeasonWeekFromTicker(ticker):
    """Best-effort parse of Kalshi NFL tickers like KXNFLGAME-25SEP07KC... or -26NOV09..."""
    import re
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", (ticker or "").upper())
    if not m:
        return None, None, None
    yy, mon, dd = int(m.group(1)), m.group(2), int(m.group(3))
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month = months.get(mon)
    if not month:
        return None, None, None
    year = 2000 + yy
    season = year if month >= 8 else year - 1
    return season, f"{year:04d}-{month:02d}-{dd:02d}", None


def DiscoverSeries(api, candidates=SURVEY_CANDIDATES):
    found = {}
    listed = api.Get("/series", {"category": "Sports", "limit": 200}) or api.Get("/series", {"limit": 200}) or {}
    series_list = listed.get("series") or listed.get("series_list") or []
    if isinstance(listed, list):
        series_list = listed
    for s in series_list:
        ticker = (s.get("ticker") or s.get("series_ticker") or "").upper()
        title = s.get("title") or s.get("name") or ""
        if "NFL" in ticker or "NFL" in title.upper() or ticker.startswith("KXNFL"):
            found[ticker] = {"ticker": ticker, "title": title, "source": "list",
                             "category": s.get("category"), "frequency": s.get("frequency"),
                             "market_kind": ClassifySeries(ticker, title), "raw": s}
    for t in candidates:
        if t in found:
            continue
        body = api.Get(f"/series/{t}") or {}
        s = body.get("series") or body
        ticker = (s.get("ticker") or t).upper() if isinstance(s, dict) else t
        if not body:
            found[t] = {"ticker": t, "title": "", "source": "probe_missing", "category": None,
                        "frequency": None, "market_kind": ClassifySeries(t, ""), "raw": None}
            continue
        title = s.get("title") or s.get("name") or ""
        found[ticker] = {"ticker": ticker, "title": title, "source": "probe",
                         "category": s.get("category"), "frequency": s.get("frequency"),
                         "market_kind": ClassifySeries(ticker, title), "raw": s}
    return found


def CollectMarkets(api, series_ticker):
    hist = api.Paged("/historical/markets", {"series_ticker": series_ticker, "limit": 1000}, max_pages=10)
    live = []
    for status in ("settled", "closed", "open"):
        live += api.Paged("/markets", {"series_ticker": series_ticker, "status": status, "limit": 1000}, max_pages=5)
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


def SampleCandles(api, series_ticker, ticker, tier, period=1):
    """One light candle pull. Returns (n, first_ts, last_ts, endpoint) or zeros on miss."""
    if tier == "historical":
        ep = f"/historical/markets/{ticker}/candlesticks"
    else:
        ep = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
    body = api.Get(ep, {"start_ts": 0, "end_ts": 2_000_000_000, "period_interval": period}) or {}
    cs = body.get("candlesticks") or []
    if not cs:
        return 0, None, None, ep
    times = [int(c.get("end_period_ts") or 0) for c in cs if c.get("end_period_ts") is not None]
    return len(cs), (min(times) if times else None), (max(times) if times else None), ep


def SurveyCoverage(series_map, markets_by_series, candle_samples):
    """Build tidy survey rows (no network)."""
    rows = []
    for ticker, info in sorted(series_map.items()):
        mkts = markets_by_series.get(ticker) or {}
        if not mkts:
            rows.append({
                "series_ticker": ticker, "series_title": info.get("title") or "",
                "market_kind": info.get("market_kind"), "source": info.get("source"),
                "n_markets": 0, "market_ticker": "", "event_ticker": "",
                "status": "", "volume": "", "close_time": "",
                "season": "", "week": "", "gameday": "",
                "candle_tier": "", "n_candles_sample": "", "candle_first_ts": "",
                "candle_last_ts": "", "candle_endpoint": "",
                "notes": "series missing" if info.get("source") == "probe_missing" else "no markets returned",
            })
            continue
        # coverage summary row
        seasons = defaultdict(int)
        for m in mkts.values():
            season, day, _w = SeasonWeekFromTicker(m.get("ticker") or m.get("event_ticker") or "")
            if season:
                seasons[season] += 1
        sample_tickers = candle_samples.get(ticker) or {}
        n_with_candles = sum(1 for v in sample_tickers.values() if v.get("n", 0) > 0)
        notes = f"seasons={dict(seasons)}; candle_samples={len(sample_tickers)} with_history={n_with_candles}"
        rows.append({
            "series_ticker": ticker, "series_title": info.get("title") or "",
            "market_kind": info.get("market_kind"), "source": info.get("source"),
            "n_markets": len(mkts), "market_ticker": "", "event_ticker": "",
            "status": "", "volume": "", "close_time": "",
            "season": "", "week": "", "gameday": "",
            "candle_tier": "", "n_candles_sample": n_with_candles,
            "candle_first_ts": "", "candle_last_ts": "", "candle_endpoint": "",
            "notes": notes,
        })
        # one row per sampled market (not every market — keep the CSV light)
        for mt, samp in sample_tickers.items():
            m = mkts.get(mt) or {}
            season, day, week = SeasonWeekFromTicker(mt or m.get("event_ticker") or "")
            rows.append({
                "series_ticker": ticker, "series_title": info.get("title") or "",
                "market_kind": info.get("market_kind"), "source": info.get("source"),
                "n_markets": len(mkts), "market_ticker": mt, "event_ticker": m.get("event_ticker") or "",
                "status": m.get("status") or "", "volume": m.get("volume") or m.get("volume_fp") or "",
                "close_time": m.get("close_time") or "",
                "season": season or "", "week": week or "", "gameday": day or "",
                "candle_tier": samp.get("tier") or "", "n_candles_sample": samp.get("n") or 0,
                "candle_first_ts": samp.get("first_ts") or "", "candle_last_ts": samp.get("last_ts") or "",
                "candle_endpoint": samp.get("endpoint") or "",
                "notes": "candle sample",
            })
    return rows


def RunSurvey(api, candidates=SURVEY_CANDIDATES, samples_per_series=3, kinds=("total", "spread")):
    series_map = DiscoverSeries(api, candidates=candidates)
    markets_by_series, candle_samples = {}, {}
    for ticker, info in series_map.items():
        if info.get("source") == "probe_missing":
            markets_by_series[ticker] = {}
            continue
        if kinds and info.get("market_kind") not in set(kinds) | {"other", "moneyline"}:
            # still collect totals/spreads; also record moneyline as a known baseline
            pass
        want = (not kinds) or info.get("market_kind") in set(kinds) or ticker in candidates
        if not want:
            continue
        markets, tier = CollectMarkets(api, ticker)
        markets_by_series[ticker] = markets
        # sample a few markets, preferring totals/spreads coverage across seasons
        picked = list(markets.values())[:samples_per_series]
        candle_samples[ticker] = {}
        for m in picked:
            mt = m["ticker"]
            n, lo, hi, ep = SampleCandles(api, ticker, mt, tier.get(mt, "live"))
            candle_samples[ticker][mt] = {"n": n, "first_ts": lo, "last_ts": hi, "endpoint": ep,
                                          "tier": tier.get(mt, "")}
    rows = SurveyCoverage(series_map, markets_by_series, candle_samples)
    return {
        "series": {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in series_map.items()},
        "n_series": len(series_map),
        "n_markets": {k: len(v) for k, v in markets_by_series.items()},
        "http_calls": api.calls, "http_retries": api.retries,
        "rows": rows,
    }
SURVEY_FIELDS = [
    "series_ticker", "series_title", "market_kind", "source", "n_markets",
    "market_ticker", "event_ticker", "status", "volume", "close_time",
    "season", "week", "gameday", "candle_tier", "n_candles_sample",
    "candle_first_ts", "candle_last_ts", "candle_endpoint", "notes",
]
GAP_FIELDS = [
    "date", "sport", "series", "season", "split", "n",
    "ign_rel_bits", "ci90_lo", "ci90_hi", "growth_per_event",
    "model_log_loss_bits", "market_log_loss_bits",
]


def FmtPct(x, digits=1):
    if x is None:
        return ""
    return f"{100 * x:.{digits}f}%"


def FmtCi(ci):
    if not ci or ci[0] is None:
        return ""
    return f"[{FmtPct(ci[0])}, {FmtPct(ci[1])}]"


def FmtMoney(x):
    if x is None:
        return ""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def UnseenList(rows_2026):
    out = []
    for r in rows_2026:
        if SeenBySearch(r):
            continue
        out.append({
            "game_id": r["game_id"], "gameday": r["gameday"], "week": int(r["week"]),
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "home_win": r["home_win"], "forward_holdout": bool(IsForwardHoldout(r["gameday"])),
        })
    out.sort(key=lambda x: (x["gameday"], x["game_id"]))
    return out


def TidyGaps(blocks, last_date):
    rows = []
    for b in blocks:
        ci = b.get("ign_rel_bits_ci90") or [None, None]
        rows.append({
            "date": last_date, "sport": SPORT, "series": "nfl_ign_rel_bits",
            "season": b["season"], "split": b["split"], "n": b["n"],
            "ign_rel_bits": b["ign_rel_bits"],
            "ci90_lo": None if ci[0] is None else ci[0],
            "ci90_hi": None if ci[1] is None else ci[1],
            "growth_per_event": b.get("growth_per_event"),
            "model_log_loss_bits": b.get("model_log_loss_bits"),
            "market_log_loss_bits": b.get("market_log_loss_bits"),
        })
    return rows


def WriteBankChart(dailies, path, title, boundaries):
    rows = []
    for series, daily in dailies:
        if not daily:
            continue
        rows.append({"date": daily[0]["date"], "series": series, "equity": START,
                     "drawdown_pct": 0.0, "season": daily[0]["season"], "x": 0, "x_kind": "index"})
        for d in daily:
            rows.append({**d, "series": series, "x_kind": "index"})
    if not rows:
        return
    frame = {"rows": rows, "boundaries": boundaries, "x_kind": "index",
             "first_date": rows[1]["date"] if len(rows) > 1 else rows[0]["date"],
             "last_date": daily[-1]["date"] if daily else rows[0]["date"]}
    PlotBankroll(frame, path=path, title=title, ylabel="Bankroll ($)",
                 hlines=[(START, "$10,000 start")], figsize=(11, 6.5),
                 xlabel="Game-day index (offseason omitted; dashed line = new season)")
    import matplotlib.pyplot as plt
    plt.close("all")


def MdReport(meta):
    gaps = {b["season"]: b for b in meta.get("gaps", [])}
    w = meta.get("blend_weights") or {}
    wci = meta.get("blend_weight_ci90") or {}
    b26 = meta.get("blend_2026") or {}
    mid = meta.get("bank_mid_tim") or {}
    ask = meta.get("bank_ask_tim") or {}
    mid_k = meta.get("bank_mid_kalshi") or {}
    ask_k = meta.get("bank_ask_kalshi") or {}
    rat = meta.get("ratings") or {}
    survey = meta.get("survey_summary") or {}

    def gap_row(key, label):
        b = gaps.get(key) or {}
        ci = b.get("ign_rel_bits_ci90")
        ci_s = "" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        g = b.get("ign_rel_bits")
        gs = "" if g is None else f"{g:+.4f}"
        return f"| {label} | {b.get('n', '')} | {gs} | {ci_s} | {b.get('growth_per_event', '')} | {b.get('model_log_loss_bits', '')} | {b.get('market_log_loss_bits', '')} |"

    def bank_row(stats, label):
        if not stats:
            return f"| {label} |  |  |  |  |  |  |"
        ci = stats.get("roi_ci90_widest")
        return (f"| {label} | {stats.get('n_bets', '')} | {FmtMoney(stats.get('final_equity'))} "
                f"| {FmtMoney(stats.get('pnl'))} | {FmtPct(stats.get('roi'))} | {FmtCi(ci)} "
                f"| {FmtPct(stats.get('max_drawdown_pct'))} | {'yes' if stats.get('ruined') else 'no'} |")

    def wci_s(key):
        c = wci.get(key)
        if not c or c[0] is None:
            return ""
        return f"90% CI [{c[0]:+.3f}, {c[1]:+.3f}]"

    claim_mid = meta.get("edge_claimable_mid")
    claim_ask = meta.get("edge_claimable_ask")
    lines = [
        "# NFL market blend: does p_full2_kmid_T60 add anything beyond Kalshi? (2026-11-13)",
        "",
        f"Made by American Footbot, {meta.get('written_at')}, America/New_York. "
        f"Code `{meta.get('code')}`, run_id `{meta.get('run_id')}`, "
        f"model_version `{meta.get('model_version')}`. Simulation only; no orders, no real wagers.",
        "",
        "## Headline",
        meta.get("headline") or "",
        "",
        "## 1. Relative Ignorance vs Kalshi T-60 (bits per game)",
        "",
        "`ign_rel_bits = mean(-log2 p_model(y)) - mean(-log2 p_kalshi_T60(y))`. "
        "Negative means the model beats the market. "
        "`growth_per_event = 2^(-ign_rel_bits)` is the Roulston & Smith weather-roulette factor "
        "for a proportional bettor who trusts the model against the market. "
        "90% CIs are a game-cluster bootstrap (one row per game; 2000 resamples).",
        "",
        "| Split | N | ign_rel_bits | 90% CI | growth / event | model bits | Kalshi bits |",
        "|---|---|---|---|---|---|---|",
        gap_row("2025", "2025, contaminated"),
        gap_row("2026", "2026 weeks 1-9, forward"),
        gap_row("2026_unseen", "2026 truly unseen"),
        "",
        "The unseen slice is small (about 20 games) and will almost certainly be inconclusive; that is expected. "
        "Existing calibration log loss was in nats; bits are nats / ln(2). "
        "Nats are in the JSON for comparison with `2026-11-13_calibration`.",
        "",
        "## 2. Walk-forward logistic blend",
        "",
        f"Form: `logit(p_blend) = a + b_model * logit(p_full2_kmid_T60) + b_kalshi * logit(Kalshi T-60 mid)`. "
        f"Fitted by minimising log loss (L2={w.get('l2', 1)} on the slopes, intercept free), "
        f"**on 2025 only**, then applied to 2026. Kalshi pregame history starts in 2025, so there is no earlier "
        f"Kalshi season to walk forward from. 2025 outcomes are in the fit; do not read 2025 blend scores as out-of-sample. "
        f"The 2026 scores below are the out-of-sample check.",
        "",
        f"- N fit (2025): {w.get('n', '')}",
        f"- a (intercept): {w.get('a', '')} {wci_s('a')}",
        f"- b_model: {w.get('b_model', '')} {wci_s('b_model')}",
        f"- b_kalshi: {w.get('b_kalshi', '')} {wci_s('b_kalshi')}",
        "",
        "If b_model is near 0, or the CI includes 0 and is unstable, the model adds nothing beyond Kalshi.",
        "",
        f"On 2026 weeks 1-9 (N={b26.get('n', '')}): blend log loss {b26.get('blend_log_loss_nats', '')} nats "
        f"({b26.get('blend_log_loss_bits', '')} bits), Brier {b26.get('blend_brier', '')}; "
        f"Kalshi log loss {b26.get('market_log_loss_nats', '')} nats "
        f"({b26.get('market_log_loss_bits', '')} bits), Brier {b26.get('market_brier', '')}.",
        "",
        "## 3. Blend bankroll, one $10k bank, 3c qualify line",
        "",
        "Primary fill is the **mid**, per Tim (\"just assume that the middle of the market is the filled price\"), "
        "plus Tim's $0.02 fee estimate (`tim_est_0.02` — an estimate, not a verified exchange fee). "
        "Qualify when `p_blend - de-vigged mid - 0.02 >= 0.03`. "
        "Secondary fill is the **ask** plus the same fee; qualify when `p_blend - ask - 0.02 >= 0.03`. "
        "flat100 bets $100; qkelly is 0.25 x Kelly at cost = fill + fee, capped at 2% of start-of-day equity. "
        "One continuous $10k bank across 2025 (contaminated, and the blend was fit on it) and 2026 weeks 1-9. "
        "2025 P&L for the blend is in-sample and should not be read as a test. "
        "CIs are the widest of event and day-cluster bootstraps. "
        "No edge is claimed unless the lower bound is above 0 under both tim_est_0.02 and the verified Kalshi "
        "taker fee 0.07·p·(1-p).",
        "",
        "### Primary: mid + tim_est_0.02",
        "",
        "| Strategy | N bets | End bank | P&L | ROI | 90% widest CI | Max drawdown | Ruin? |",
        "|---|---|---|---|---|---|---|---|",
        bank_row(mid.get("flat100"), "flat100 / blend"),
        bank_row(mid.get("qkelly"), "qkelly / blend"),
        bank_row(mid.get("market_fav"), "market_fav"),
        "",
        f"Same bets, verified Kalshi taker fee — claimable mid? **{'yes' if claim_mid else 'no'}**.",
        "",
        "| Strategy | N bets | End bank | ROI | 90% widest CI | Max drawdown |",
        "|---|---|---|---|---|---|",
        bank_row(mid_k.get("flat100"), "flat100 / blend").rsplit('|', 3)[0] + " |" if False else
        f"| flat100 / blend | {(mid_k.get('flat100') or {}).get('n_bets', '')} | {FmtMoney((mid_k.get('flat100') or {}).get('final_equity'))} | {FmtPct((mid_k.get('flat100') or {}).get('roi'))} | {FmtCi((mid_k.get('flat100') or {}).get('roi_ci90_widest'))} | {FmtPct((mid_k.get('flat100') or {}).get('max_drawdown_pct'))} |",
        f"| qkelly / blend | {(mid_k.get('qkelly') or {}).get('n_bets', '')} | {FmtMoney((mid_k.get('qkelly') or {}).get('final_equity'))} | {FmtPct((mid_k.get('qkelly') or {}).get('roi'))} | {FmtCi((mid_k.get('qkelly') or {}).get('roi_ci90_widest'))} | {FmtPct((mid_k.get('qkelly') or {}).get('max_drawdown_pct'))} |",
        "",
        "### Secondary: ask + tim_est_0.02",
        "",
        "| Strategy | N bets | End bank | P&L | ROI | 90% widest CI | Max drawdown | Ruin? |",
        "|---|---|---|---|---|---|---|---|",
        bank_row(ask.get("flat100"), "flat100 / blend"),
        bank_row(ask.get("qkelly"), "qkelly / blend"),
        bank_row(ask.get("market_fav"), "market_fav"),
        "",
        f"Verified Kalshi taker fee on the ask bet set — claimable ask? **{'yes' if claim_ask else 'no'}**.",
        "",
        f"| flat100 / blend | {(ask_k.get('flat100') or {}).get('n_bets', '')} | {FmtMoney((ask_k.get('flat100') or {}).get('final_equity'))} | {FmtPct((ask_k.get('flat100') or {}).get('roi'))} | {FmtCi((ask_k.get('flat100') or {}).get('roi_ci90_widest'))} |",
        f"| qkelly / blend | {(ask_k.get('qkelly') or {}).get('n_bets', '')} | {FmtMoney((ask_k.get('qkelly') or {}).get('final_equity'))} | {FmtPct((ask_k.get('qkelly') or {}).get('roi'))} | {FmtCi((ask_k.get('qkelly') or {}).get('roi_ci90_widest'))} |",
        "",
        "## 4. Prototype yards-per-play and success-rate ratings",
        "",
        f"One pre-specified variant (`{RATINGS_VARIANT}`): walk-forward logit of WIN_FEATS2 plus `ypp_diff` and "
        f"`sr_diff` (home net minus away net), leak-free season-to-date on earlier gamedays only, "
        f"minimum {MIN_RATING_PLAYS} prior off and def plays. Local full play-by-play (yards_gained, success) "
        f"is used when present; slim pre-play files have no yards or success and do not contribute. "
        f"**Trials logged: {RATINGS_TRIALS}.** No threshold search.",
        "",
        rat.get("md") or "Ratings were not run in this invocation.",
        "",
        "## 5. Kalshi NFL totals and spreads survey",
        "",
        survey.get("md") or "Survey was not run in this invocation (no API calls).",
        "",
        "## Caveats for Graphtard to caption",
        "",
        "- The model uses the market line as an input (it is a recalibrated market, not an independent forecast). `spread_gap` still uses the closing spread, a residual look-ahead at T-60.",
        "- 2025 is contaminated (search tally at least 2,819, plus this blend evaluation and the ratings trial). The blend is fit on 2025, so 2025 blend P&L is in-sample.",
        "- 2026 weeks 1-9 are labelled forward, but only about 20 games are truly unseen (kickoff + 3h15m after the 2026-11-01 16:30:11 ET search pull). Only BUF @ MIN (2026-11-09) is after the 2026-11-08 freeze. Small-N CIs are wide.",
        "- Fill primary is the mid, per Tim; candles have no size, so fills are optimistic for large stakes. Fee is Tim's $0.02 estimate unless the verified-Kalshi path is named.",
        "- Do not claim an edge unless the lower 90% bound is above 0 under both fee bases, on a serious sample.",
        "",
        "## Files",
        "",
        "- `2026-11-13_market-blend.md`, `.csv`, `.json`",
        "- `2026-11-13_market-blend_mid_draft.png`, `2026-11-13_market-blend_ask_draft.png`",
        "- `2026-11-13_market-blend_survey.csv`",
        "- `2026-11-13_market-blend_edges.csv`",
        "",
    ]
    return "\n".join(lines) + "\n"


def RatingsMd(rat):
    if not rat:
        return "Ratings were not run."
    lines = [
        f"Play-by-play sources: {json.dumps(rat.get('pbp_info_short') or rat.get('pbp_kind') or {}, sort_keys=True)}.",
        f"Team-games with prior ratings ready: {rat.get('n_ready', '')}.",
        "",
        "| Split | N | d log loss (nats) | d log loss (bits) | d Brier |",
        "|---|---|---|---|---|",
    ]
    for split, g in (rat.get("gains") or {}).items():
        lines.append(f"| {split} | {g.get('n', '')} | {g.get('d_log_loss_nats', '')} | {g.get('d_log_loss_bits', '')} | {g.get('d_brier', '')} |")
    lines.append("")
    lines.append("Negative d log loss / d Brier means the ratings-augmented model improved on p_full2 on the same games. "
                 "This is a prototype, not a claimed edge.")
    return "\n".join(lines)


def SurveyMd(summary):
    if not summary:
        return "Survey was not run."
    lines = [
        f"Public API only, {summary.get('http_calls', '')} HTTP calls ({summary.get('http_retries', 0)} retries). "
        "Candle history is a small sample per series, not a full pull.",
        "",
        "| Series | Kind | Source | Markets | Sampled markets with candles |",
        "|---|---|---|---|---|",
    ]
    for t, info in (summary.get("series") or {}).items():
        n = (summary.get("n_markets") or {}).get(t, 0)
        lines.append(f"| {t} | {info.get('market_kind')} | {info.get('source')} | {n} | see survey CSV |")
    return "\n".join(lines)


def FirstSeasonX(daily, season="2026"):
    for d in daily or []:
        if d.get("season") == season:
            return d["x"], d["date"]
    return None, None


def HeadlineFrom(meta):
    gaps = {b["season"]: b for b in meta.get("gaps", [])}
    w = meta.get("blend_weights") or {}
    g26 = gaps.get("2026") or {}
    bits = g26.get("ign_rel_bits")
    ci = g26.get("ign_rel_bits_ci90") or [None, None]
    b_model = w.get("b_model")
    claim = meta.get("edge_claimable_mid") or meta.get("edge_claimable_ask")
    bit_s = "n/a" if bits is None else f"{bits:+.4f} bits/game"
    ci_s = "" if ci[0] is None else f" (90% CI {ci[0]:+.4f} to {ci[1]:+.4f}, N={g26.get('n', '')})"
    w_s = "n/a" if b_model is None else f"{b_model:+.3f}"
    claim_s = "no edge is claimed" if not claim else "an edge claim is recorded in the tables below (check both fee bases)"
    return (f"On 2026 weeks 1-9 the model's relative Ignorance vs Kalshi T-60 is {bit_s}{ci_s}. "
            f"The 2025-fit blend weight on the model is b_model={w_s}. {claim_s.capitalize()}.")


def RunGapsAndBlend(kalshi_rows, rng, boot=BOOT):
    splits = SplitRows(kalshi_rows)
    gaps = [
        GapBlock(splits["2025"], "p_kmid", "kmid", "2025", "contaminated", rng, boot=boot),
        GapBlock(splits["2026"], "p_kmid", "kmid", "2026", "forward", rng, boot=boot),
        GapBlock(splits["2026_unseen"], "p_kmid", "kmid", "2026_unseen", "forward", rng, boot=boot),
    ]
    weights = FrozenSeasonBlend(splits["2025"], splits["2026"] + splits["2025"])
    wci = BootstrapBlendWeights(splits["2025"], rng, boot=boot)
    wci_round = {k: (None if v[0] is None else [round(v[0], 4), round(v[1], 4)])
                 for k, v in wci.items() if k in ("a", "b_model", "b_kalshi")}
    wci_round["n_ok"] = wci["n_ok"]
    wci_round["n"] = wci["n"]
    y26 = splits["2026"]
    blend_2026 = {
        "n": len(y26),
        "blend_log_loss_nats": (lambda t: None if t[0] is None else round(t[0], 6))(MeanLogLoss(y26, "p_blend", False)),
        "blend_log_loss_bits": (lambda t: None if t[0] is None else round(t[0], 6))(MeanLogLoss(y26, "p_blend", True)),
        "market_log_loss_nats": (lambda t: None if t[0] is None else round(t[0], 6))(MeanLogLoss(y26, "kmid", False)),
        "market_log_loss_bits": (lambda t: None if t[0] is None else round(t[0], 6))(MeanLogLoss(y26, "kmid", True)),
        "blend_brier": (lambda t: None if t[0] is None else round(t[0], 6))(MeanBrier(y26, "p_blend")),
        "market_brier": (lambda t: None if t[0] is None else round(t[0], 6))(MeanBrier(y26, "kmid")),
        "model_log_loss_nats": (lambda t: None if t[0] is None else round(t[0], 6))(MeanLogLoss(y26, "p_kmid", False)),
        "model_brier": (lambda t: None if t[0] is None else round(t[0], 6))(MeanBrier(y26, "p_kmid")),
    }
    return splits, gaps, weights, wci_round, blend_2026


def LoadKalshiModelRows():
    rbs = AllRows()
    AttachModels(rbs)
    check = AttachKmid(rbs)
    kalshi_rows, dropped, ties = KalshiRows(rbs)
    return rbs, kalshi_rows, dropped, ties, check


def RunRatings(rbs, root="data"):
    seasons = sorted({int(s) for s in rbs if int(s) >= 2023})
    game_stats, pbp_info = LoadGameStats(seasons, root=root)
    ratings = PriorStdRatings(game_stats)
    flat = [r for s in rbs for r in rbs[s]]
    n_ready = AttachRatings(flat, ratings)
    coefs = FitRatingsModel(rbs)
    gains = {}
    for label, pred in (
        ("2024", lambda r: int(r["season"]) == 2024),
        ("2025_contaminated", lambda r: int(r["season"]) == 2025),
        ("2026_forward", lambda r: int(r["season"]) == 2026 and int(r["week"]) <= 9),
        ("2026_unseen", lambda r: int(r["season"]) == 2026 and int(r["week"]) <= 9 and not SeenBySearch(r)),
    ):
        subset = [r for s in rbs for r in rbs[s] if pred(r)]
        gains[label] = RatingsGain(subset)
    pbp_kind = {str(s): {"kind": (pbp_info.get(s) or {}).get("kind"),
                         "kept_plays": (pbp_info.get(s) or {}).get("kept_plays"),
                         "skipped_no_yards": (pbp_info.get(s) or {}).get("skipped_no_yards"),
                         "team_games": (pbp_info.get(s) or {}).get("team_games")}
                for s in seasons}
    rat = {
        "variant": RATINGS_VARIANT, "trials": RATINGS_TRIALS, "min_plays": MIN_RATING_PLAYS,
        "n_ready": n_ready, "gains": gains, "pbp_info": pbp_info, "pbp_kind": pbp_kind,
        "coefs_seasons": sorted(coefs),
    }
    rat["pbp_info_short"] = pbp_kind
    rat["md"] = RatingsMd(rat)
    return rat


def Main():
    ap = argparse.ArgumentParser(description="Market-blend study (run only when Tim approves).")
    ap.add_argument("--out-dir", default=r"E:\AgentDocs\American-Footbot\2026-11-13_market-blend")
    ap.add_argument("--base", default="2026-11-13_market-blend")
    ap.add_argument("--skip-ratings", action="store_true")
    ap.add_argument("--skip-survey", action="store_true")
    ap.add_argument("--ratings-only", action="store_true")
    ap.add_argument("--survey-only", action="store_true")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--survey-samples", type=int, default=3)
    a = ap.parse_args()
    SetBelowNormal()
    run_id = NewRunId()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.base)
    written = datetime.now(ET).strftime("%Y-%m-%d %-I:%M %p %Z") if os.name != "nt" else datetime.now(ET).strftime("%Y-%m-%d %I:%M %p %Z")

    meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "written_at": written, "fee_basis": FEE_BASIS, "fill_primary": "mid",
        "fill_secondary": "ask", "min_edge": MIN_EDGE, "start": START,
        "headline_model": MODEL, "blend_model": BLEND_MODEL,
        "contamination_tally_at_least": 2819, "ratings_trials": RATINGS_TRIALS,
        "frozen_at": FROZEN["frozen_at"], "search_pulled_at": "2026-11-01T16:30:11-05:00",
        "blend_fit": "2025 only, applied to 2026",
    }

    if a.survey_only:
        survey = RunSurvey(KalshiApi(), samples_per_series=a.survey_samples)
        survey["md"] = SurveyMd(survey)
        meta["survey_summary"] = survey
        WriteCsv(base + "_survey.csv", SURVEY_FIELDS, survey["rows"])
        meta["headline"] = "Kalshi totals/spreads survey only; gaps and bankrolls were not computed."
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, default=str)
        with open(base + ".md", "w", encoding="utf-8") as fh:
            fh.write(MdReport(meta))
        print(json.dumps({"run_id": run_id, "survey_calls": survey["http_calls"],
                          "n_series": survey["n_series"], "n_markets": survey["n_markets"],
                          "out": a.out_dir}, indent=1, default=str))
        return

    rbs, kalshi_rows, dropped, ties, check = LoadKalshiModelRows()
    splits, gaps, weights, wci, blend_2026 = RunGapsAndBlend(kalshi_rows, rng, boot=a.boot)
    meta["gaps"] = gaps
    meta["blend_weights"] = None if weights is None else {k: (round(v, 6) if isinstance(v, float) else v) for k, v in weights.items()}
    meta["blend_weight_ci90"] = wci
    meta["blend_2026"] = blend_2026
    meta["refit_check"] = check
    meta["dropped_missing_prices"] = dropped
    meta["ties"] = ties
    meta["unseen"] = UnseenList(splits["2026"])
    meta["unseen_n"] = len(meta["unseen"])
    meta["n_kalshi"] = len(kalshi_rows)

    if a.ratings_only:
        rat = RunRatings(rbs)
        meta["ratings"] = rat
        meta["headline"] = "Ratings prototype only; bankrolls were not computed."
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, default=str)
        with open(base + ".md", "w", encoding="utf-8") as fh:
            fh.write(MdReport(meta))
        print(json.dumps({"run_id": run_id, "ratings_trials": RATINGS_TRIALS, "gains": rat["gains"],
                          "n_ready": rat["n_ready"], "out": a.out_dir}, indent=1, default=str))
        return

    days = AllGameDays(rbs)
    mid_edges = BlendEdges(kalshi_rows, fill_kind="mid")
    ask_edges = BlendEdges(kalshi_rows, fill_kind="ask")
    bank_mid, daily_mid, _log_mid = RunBankroll(mid_edges, days, rng, fee_kind="tim")
    bank_mid_k, daily_mid_k, log_mid_k = RunBankroll(mid_edges, days, rng, fee_kind="kalshi")
    bank_ask, daily_ask, _log_ask = RunBankroll(ask_edges, days, rng, fee_kind="tim")
    bank_ask_k, daily_ask_k, log_ask_k = RunBankroll(ask_edges, days, rng, fee_kind="kalshi")
    meta["bank_mid_tim"] = bank_mid
    meta["bank_mid_kalshi"] = bank_mid_k
    meta["bank_ask_tim"] = bank_ask
    meta["bank_ask_kalshi"] = bank_ask_k
    meta["edge_claimable_mid"] = EdgeClaimable(
        (bank_mid.get("flat100") or {}).get("roi_ci90_widest"),
        (bank_mid_k.get("flat100") or {}).get("roi_ci90_widest"),
    ) and EdgeClaimable(
        (bank_mid.get("qkelly") or {}).get("roi_ci90_widest"),
        (bank_mid_k.get("qkelly") or {}).get("roi_ci90_widest"),
    )
    meta["edge_claimable_ask"] = EdgeClaimable(
        (bank_ask.get("flat100") or {}).get("roi_ci90_widest"),
        (bank_ask_k.get("flat100") or {}).get("roi_ci90_widest"),
    ) and EdgeClaimable(
        (bank_ask.get("qkelly") or {}).get("roi_ci90_widest"),
        (bank_ask_k.get("qkelly") or {}).get("roi_ci90_widest"),
    )

    rat = None
    if not a.skip_ratings:
        rat = RunRatings(rbs)
        meta["ratings"] = rat
    survey = None
    if not a.skip_survey:
        survey = RunSurvey(KalshiApi(), samples_per_series=a.survey_samples)
        survey["md"] = SurveyMd(survey)
        meta["survey_summary"] = {k: survey[k] for k in ("series", "n_series", "n_markets", "http_calls", "http_retries", "md")}

    meta["headline"] = HeadlineFrom(meta)

    last = days[-1] if days else datetime.now(ET).date().isoformat()
    tidy = TidyGaps(gaps, last)
    bank_rows = []
    for strat, daily in daily_mid.items():
        bank_rows.extend(daily)
    WriteCsv(base + ".csv", GAP_FIELDS, tidy)
    WriteCsv(base + "_bankroll.csv", BANK_FIELDS, bank_rows)
    WriteCsv(base + "_edges.csv", EDGE_FIELDS + EDGE_EXTRA + ["fill_kind"], mid_edges + ask_edges)
    if survey:
        WriteCsv(base + "_survey.csv", SURVEY_FIELDS, survey["rows"])
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdReport(meta))

    x2026, d2026 = FirstSeasonX(daily_mid.get("flat100"), "2026")
    bounds = []
    if x2026 is not None:
        bounds.append({"series": None, "season": "2026", "x": x2026, "date": d2026})
    WriteBankChart(
        [("flat100 / mid", daily_mid.get("flat100") or []),
         ("qkelly / mid", daily_mid.get("qkelly") or []),
         ("market_fav / mid", daily_mid.get("market_fav") or [])],
        base + "_mid_draft.png",
        "NFL $10k bankroll from p_blend at Kalshi T-60 mid, tim_est_0.02",
        bounds,
    )
    xa, da = FirstSeasonX(daily_ask.get("flat100"), "2026")
    abounds = []
    if xa is not None:
        abounds.append({"series": None, "season": "2026", "x": xa, "date": da})
    WriteBankChart(
        [("flat100 / ask", daily_ask.get("flat100") or []),
         ("qkelly / ask", daily_ask.get("qkelly") or []),
         ("market_fav / ask", daily_ask.get("market_fav") or [])],
        base + "_ask_draft.png",
        "NFL $10k bankroll from p_blend at Kalshi T-60 ask, tim_est_0.02",
        abounds,
    )
    print(json.dumps({
        "run_id": run_id, "code": CodeVersion(),
        "gaps": gaps, "blend_weights": meta["blend_weights"], "blend_2026": blend_2026,
        "bank_mid": {k: {m: bank_mid[k][m] for m in ("n_bets", "final_equity", "roi", "ruined")} for k in bank_mid},
        "bank_ask": {k: {m: bank_ask[k][m] for m in ("n_bets", "final_equity", "roi", "ruined")} for k in bank_ask},
        "edge_claimable_mid": meta["edge_claimable_mid"], "edge_claimable_ask": meta["edge_claimable_ask"],
        "ratings": None if not rat else {"trials": RATINGS_TRIALS, "gains": rat["gains"], "n_ready": rat["n_ready"]},
        "survey_calls": None if not survey else survey["http_calls"],
        "out": a.out_dir,
    }, indent=1, default=str))


if __name__ == "__main__":
    Main()
