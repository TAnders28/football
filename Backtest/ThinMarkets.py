r"""
Thin-markets study: are Kalshi NFL totals and spreads softer than moneylines?

Pre-registered (Degen Bot, 2026-11-13, before looking at 2026 results):
  H1: totals/spreads at T-60 are softer than moneylines after bid/ask width and fees.
  Fit/selection on 2025 only, freeze, test on 2026.
  Favourite-longshot in 0.05 price bins. Edge = bin hit-rate - (fill + 0.02).
  A bin is selected only if N >= MIN_BIN_N (20, pre-specified) and edge >= 0.03.
  PRIMARY fill = ask + tim_est_0.02. SECONDARY fill = mid + tim_est_0.02 (Tim's mid-fill rule).
  Claim an edge only if the widest 95% CI lower bound is > 0 under both tim_est_0.02
  and the verified Kalshi taker fee 0.07*p*(1-p).
  Trial log: 1 hypothesis; 2 fill variants x 2 families (lines, moneylines) = 4
  bankroll evaluations. Fee-basis recomputes and descriptive kind splits are not extra searches.

    .venv\Scripts\python.exe -m Backtest.ThinMarkets --out-dir DIR [--skip-pull]

Simulation only; no orders. BelowNormal on Windows. Season-axis charts via BankrollPlot.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.BankrollPlot import FrameTidy, PlotBankroll
from Backtest.CommonFormat import (
    BANK_FIELDS, ClopperPearson, FEE, FEE_BASIS, FLAT, MIN_EDGE, QKELLY_CAP, QKELLY_FRAC,
    QKellyStake, RUIN, START, Walk, Widest, WriteCsv,
)
from Backtest.KalshiReality import KalshiFee, QuoteAt, STALE_MIN
from Backtest.MarketBlend import ForceIPv4, SetBelowNormal
from Backtest.Version import CodeVersion, ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
SPORT = "nfl"
T60 = 60
BIN = 0.05
MIN_BIN_N = 20
ALPHA = 0.05  # 95% CIs, as pre-registered
BOOT = 2000
SEED = 20261113
PRICE_ASK = "kalshi_T60_ask"
PRICE_MID = "kalshi_T60_mid"
TRIALS_HYPOTHESES = 1
TRIALS_BANKROLL = 4  # ask/mid x lines/moneyline
TRIALS_NOTE = (
    "1 pre-registered hypothesis (thin markets softer than moneylines after spread cost). "
    "4 bankroll evaluations: ask and mid fills, each on totals+spreads and on moneylines. "
    "Kalshi-fee P&L is a recompute of the same bets, not a new search. "
    "Per-kind (total vs spread) tables are descriptive."
)
N_BINS = int(round(1.0 / BIN))


def YesResult(result):
    if result is None:
        return None
    s = str(result).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1
    if s in ("no", "n", "false", "0"):
        return 0
    return None  # void / scalar / push / unknown


def BinLo(price):
    if price is None or price < 0 or price > 1:
        return None
    if price >= 1.0:
        return round(1.0 - BIN, 2)
    return round(math.floor(price / BIN + 1e-12) * BIN, 2)


def BinHi(lo):
    return 1.0 if abs(lo - (1.0 - BIN)) < 1e-12 else round(lo + BIN, 2)


def BinLabel(lo):
    return f"{lo:.2f}-{BinHi(lo):.2f}"


def IterJsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping bad jsonl line {i} in {path}", flush=True)
                continue


def LoadMarkets(root="data"):
    path = os.path.join(root, "kalshi", "thin_markets.jsonl")
    out = {}
    for r in IterJsonl(path) or []:
        if r.get("ticker"):
            out[r["ticker"]] = r
    return out


def LoadFees(root="data"):
    path = os.path.join(root, "kalshi", "thin_fees.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def LoadThinEntries(seasons=(2025, 2026), root="data"):
    """game-like quote entries: ticker -> {ko, candles, meta}."""
    out = []
    for s in seasons:
        path = os.path.join(root, "kalshi", f"thin_candles_{s}.jsonl")
        for r in IterJsonl(path) or []:
            ko = r.get("kickoff_utc")
            if not ko:
                continue
            out.append({
                **r,
                "ko": datetime.fromisoformat(ko.replace("Z", "+00:00")),
                "candles": sorted(r.get("candles") or [], key=lambda c: c["end_ts"]),
            })
    return out


def QuoteBundle(entry):
    q = QuoteAt(entry, T60)
    if not q:
        return None
    width = q["ask"] - q["bid"]
    if width < 0:
        return None
    mid = (q["bid"] + q["ask"]) / 2
    # volume / OI from the T-60 candle (same candle QuoteAt picked)
    target = (entry["ko"].timestamp()) - T60 * 60
    candle = None
    for c in entry["candles"]:
        if c["end_ts"] > target:
            break
        if c.get("yes_ask_close") is not None and c.get("yes_bid_close") is not None and 0 < c["yes_ask_close"] < 1:
            candle = c
    vol = None if not candle else candle.get("volume")
    oi = None if not candle else candle.get("open_interest")
    return {
        "bid": q["bid"], "ask": q["ask"], "mid": mid, "width": width,
        "age_min": q["age_min"], "volume_t60": vol, "open_interest_t60": oi,
    }


def ContractRows(entries, markets=None, kinds=None):
    markets = markets or {}
    rows = []
    dropped = {"no_quote": 0, "no_result": 0, "no_game": 0, "error": 0}
    for e in entries:
        if e.get("error") and not e.get("candles"):
            dropped["error"] += 1
            continue
        if not e.get("game_id"):
            dropped["no_game"] += 1
            continue
        kind = e.get("market_kind") or (markets.get(e["ticker"]) or {}).get("market_kind")
        if kinds and kind not in kinds:
            continue
        y = YesResult(e.get("result"))
        q = QuoteBundle(e)
        if not q:
            dropped["no_quote"] += 1
            continue
        if y is None:
            dropped["no_result"] += 1
            continue
        m = markets.get(e["ticker"]) or {}
        gday = m.get("gameday")
        if not gday:
            try:
                gday = e["ko"].astimezone(ET).date().isoformat()
            except Exception:
                gday = (e.get("kickoff_utc") or "")[:10]
        rows.append({
            "date": gday,
            "start_time": e.get("kickoff_utc"),
            "sport": SPORT, "season": str(int(e["season"])), "week": e.get("week"),
            "split": "contaminated" if int(e["season"]) == 2025 else "forward",
            "event_id": e.get("game_id"), "ticker": e["ticker"],
            "event": f"{e.get('away_team')} @ {e.get('home_team')}",
            "market_kind": kind, "series_ticker": e.get("series_ticker") or m.get("series_ticker"),
            "result": y, "bid": q["bid"], "ask": q["ask"], "mid": q["mid"],
            "width": q["width"], "age_min": q["age_min"],
            "volume_t60": q["volume_t60"], "open_interest_t60": q["open_interest_t60"],
            "market_volume": m.get("volume"),
        })
    return rows, dropped


def MoneylineRows(seasons=(2025, 2026)):
    """One yes-contract row per moneyline side, same T-60 / stale rule."""
    rows, dropped = [], {"no_quote": 0, "no_result": 0}
    for s in seasons:
        path = os.path.join("data", "kalshi", f"nflgame_candles_{s}.jsonl")
        for r in IterJsonl(path) or []:
            y = YesResult(r.get("result"))
            ko = r.get("kickoff_utc")
            entry = {
                "ko": datetime.fromisoformat(ko.replace("Z", "+00:00")),
                "candles": sorted(r.get("candles") or [], key=lambda c: c["end_ts"]),
            }
            q = QuoteBundle(entry)
            if not q:
                dropped["no_quote"] += 1
                continue
            if y is None:
                dropped["no_result"] += 1
                continue
            gday = ko[:10]
            # prefer schedule gameday via kickoff ET
            try:
                gday = entry["ko"].astimezone(ET).date().isoformat()
            except Exception:
                pass
            rows.append({
                "date": gday, "start_time": ko, "sport": SPORT, "season": str(int(r["season"])),
                "week": r.get("week"),
                "split": "contaminated" if int(r["season"]) == 2025 else "forward",
                "event_id": r.get("game_id"), "ticker": r.get("ticker"),
                "event": f"{r.get('away_team')} @ {r.get('home_team')}",
                "market_kind": "moneyline", "series_ticker": "KXNFLGAME",
                "result": y, "bid": q["bid"], "ask": q["ask"], "mid": q["mid"],
                "width": q["width"], "age_min": q["age_min"],
                "volume_t60": q.get("volume_t60"), "open_interest_t60": q.get("open_interest_t60"),
                "market_volume": None, "side_team": r.get("side_team"),
            })
    return rows, dropped


def Coverage(rows, fees=None, dropped=None):
    by = defaultdict(list)
    for r in rows:
        by[(r["market_kind"], r["season"])].append(r)
    out = []
    for (kind, season), grp in sorted(by.items()):
        widths = [r["width"] for r in grp if r.get("width") is not None]
        vols = [r["volume_t60"] for r in grp if r.get("volume_t60") is not None]
        out.append({
            "market_kind": kind, "season": season, "n": len(grp),
            "n_games": len({r["event_id"] for r in grp}),
            "n_days": len({r["date"] for r in grp}),
            "median_width": None if not widths else float(np.median(widths)),
            "p25_width": None if not widths else float(np.quantile(widths, 0.25)),
            "p75_width": None if not widths else float(np.quantile(widths, 0.75)),
            "median_volume_t60": None if not vols else float(np.median(vols)),
            "mean_width": None if not widths else float(np.mean(widths)),
        })
    return out


def MedianWidth(rows):
    w = [r["width"] for r in rows if r.get("width") is not None]
    return None if not w else float(np.median(w))


def FillOf(row, fill_kind):
    return row["ask"] if fill_kind == "ask" else row["mid"]


def FitBins(rows, fill_kind, fee=FEE, min_n=MIN_BIN_N, min_edge=MIN_EDGE):
    """2025-only empirical hit rates. Does not look at any other season."""
    buckets = {round(i * BIN, 2): [] for i in range(N_BINS)}
    for r in rows:
        fill = FillOf(r, fill_kind)
        lo = BinLo(fill)
        if lo is None:
            continue
        buckets[lo].append(r)
    stats, selected = [], []
    for lo in sorted(buckets):
        grp = buckets[lo]
        n = len(grp)
        k = int(sum(r["result"] for r in grp))
        mean_px = None if not grp else float(np.mean([FillOf(r, fill_kind) for r in grp]))
        obs = None if not n else k / n
        # edge vs fee: hit rate minus (mean fill + fee). Also fraction that would qualify
        # at the individual-contract 3c line using this bin's hit rate as p.
        edge = None if obs is None or mean_px is None else obs - (mean_px + fee)
        choose = bool(n >= min_n and edge is not None and edge >= min_edge)
        fair = None if obs is None else round(obs, 2)
        rec = {
            "bin_lo": lo, "bin_hi": BinHi(lo), "n": n, "k": k,
            "mean_price": mean_px, "obs_rate": obs, "edge": edge,
            "fair_value": fair, "selected": choose, "fill_kind": fill_kind,
        }
        stats.append(rec)
        if choose:
            selected.append(lo)
    p_by_lo = {s["bin_lo"]: s["obs_rate"] for s in stats if s["obs_rate"] is not None}
    return stats, selected, p_by_lo


def MakeEdges(rows, selected, p_by_lo, fill_kind, price_basis, fee_kind="tim"):
    """Build CommonFormat-compatible edge rows for Walk(). p is the frozen 2025 bin rate."""
    edges = []
    for r in rows:
        fill = FillOf(r, fill_kind)
        lo = BinLo(fill)
        if lo is None:
            continue
        p = p_by_lo.get(lo)
        fee = KalshiFee(fill) if fee_kind == "kalshi" else FEE
        edge = None if p is None else p - fill - fee
        qualifies = bool(lo in selected and p is not None and (p - fill - FEE) >= MIN_EDGE)
        # Qualification always uses the $0.02 rule and the frozen 2025 selection so
        # the Kalshi-fee path is a recompute of the same bets, not a new search.
        e = {
            "date": r["date"], "start_time": r.get("start_time"), "sport": SPORT,
            "season": r["season"], "split": r["split"], "event_id": r["event_id"],
            "event": r.get("event"), "side": r.get("ticker"), "model": "flb_bin_2025",
            "price_basis": price_basis, "market_kind": r.get("market_kind"),
            "model_prob": p, "market_prob_devig": fill, "fill": fill,
            "edge": edge, "qualifies": qualifies, "result": r["result"],
            "bin_lo": lo, "width": r.get("width"),
            "_p": p, "_fill": fill, "_fav_fill": fill, "_fav_result": r["result"],
            "fee_kind": fee_kind,
        }
        edges.append(e)
    return edges


def DayBlockRoi(bet_log, rng, boot=BOOT, alpha=ALPHA):
    by = defaultdict(list)
    for b in bet_log:
        by[b["date"]].append(b)
    days = list(by)
    if not days:
        return None, None
    pnl_d = [sum(x["pnl"] for x in by[d]) for d in days]
    st_d = [sum(x["stake"] for x in by[d]) for d in days]
    return RoiBoot(pnl_d, st_d, rng, boot, alpha)


def GameClusterRoi(bet_log, edges, rng, boot=BOOT, alpha=ALPHA):
    ev_of = {}
    for e in edges:
        ev_of[(e["date"], e["event_id"], e["side"])] = e["event_id"]
    by = defaultdict(list)
    for b in bet_log:
        gid = ev_of.get((b["date"], b["event_id"], b.get("side"))) or b.get("event_id") or b["date"]
        by[gid].append(b)
    games = list(by)
    if not games:
        return None, None
    pnl_g = [sum(x["pnl"] for x in by[g]) for g in games]
    st_g = [sum(x["stake"] for x in by[g]) for g in games]
    return RoiBoot(pnl_g, st_g, rng, boot, alpha)


def RoiBoot(pnls, stakes, rng, boot=BOOT, alpha=ALPHA):
    pnls = np.asarray(pnls, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    n = len(pnls)
    if n == 0 or stakes.sum() == 0:
        return None, None
    idx = rng.integers(0, n, size=(boot, n))
    den = stakes[idx].sum(1)
    rois = pnls[idx].sum(1) / np.where(den == 0, np.nan, den)
    rois = rois[np.isfinite(rois)]
    if len(rois) == 0:
        return None, None
    return float(np.quantile(rois, alpha / 2)), float(np.quantile(rois, 1 - alpha / 2))


def BinomialRoi(bet_log):
    """Map a 95% Clopper-Pearson hit-rate CI through mean win/loss amounts to an ROI interval."""
    if not bet_log:
        return None, None, None
    wins = [b for b in bet_log if b["pnl"] > 0]
    losses = [b for b in bet_log if b["pnl"] <= 0]
    n = len(bet_log)
    k = len(wins)
    lo_p, hi_p = ClopperPearson(k, n, alpha=ALPHA)
    staked = sum(b["stake"] for b in bet_log)
    if not staked or lo_p is None:
        return None, None, (lo_p, hi_p)
    avg_win = (sum(b["pnl"] for b in wins) / k) if k else 0.0
    avg_loss = (sum(b["pnl"] for b in losses) / (n - k)) if (n - k) else 0.0

    def roi_at(p):
        return (p * n * avg_win + (1 - p) * n * avg_loss) / staked

    return roi_at(lo_p), roi_at(hi_p), (lo_p, hi_p)


def WidestRoiCi(bet_log, edges, rng):
    ev = RoiBoot([b["pnl"] for b in bet_log], [b["stake"] for b in bet_log], rng, BOOT, ALPHA)
    day = DayBlockRoi(bet_log, rng)
    game = GameClusterRoi(bet_log, edges, rng)
    b_lo, b_hi, hit_ci = BinomialRoi(bet_log)
    binomial = (b_lo, b_hi)
    widest = ev
    for other in (day, game, binomial):
        if other and other[0] is not None:
            widest = Widest(widest, other) if widest and widest[0] is not None else other
    return {
        "event_boot_95": ev, "day_block_95": day, "game_cluster_95": game,
        "binomial_mapped_95": binomial, "hit_rate_cp_95": hit_ci, "widest_95": widest,
    }


def BankBundle(edges, days, rng, fee_kind, strategies=("flat100", "qkelly")):
    out = {}
    dailies = {}
    logs = {}
    for strat in strategies:
        daily, blog, ruined, end = Walk(edges, strat, fee_kind=fee_kind, days=days)
        staked = sum(b["stake"] for b in blog)
        pnl = sum(b["pnl"] for b in blog)
        roi = None if not staked else pnl / staked
        ci = WidestRoiCi(blog, edges, rng)
        dd = min((d["drawdown_pct"] for d in daily), default=0.0)
        out[strat] = {
            "strategy": strat, "fee_kind": fee_kind, "n_bets": len(blog),
            "final_equity": end, "staked": round(staked, 2), "pnl": round(pnl, 2),
            "roi": roi, "max_drawdown_pct": dd, "ruined": bool(ruined or end < RUIN),
            "ci": {k: (None if v is None or (isinstance(v, tuple) and v[0] is None) else [v[0], v[1]])
                   for k, v in ci.items()},
        }
        dailies[strat] = daily
        logs[strat] = blog
    return out, dailies, logs


def PerBinHoldout(rows, fit_stats, selected, p_by_lo, fill_kind, rng):
    """2026 realized results in each 0.05 bin, including unselected bins."""
    by = defaultdict(list)
    for r in rows:
        lo = BinLo(FillOf(r, fill_kind))
        if lo is not None:
            by[lo].append(r)
    fit_map = {s["bin_lo"]: s for s in fit_stats}
    out = []
    for i in range(N_BINS):
        lo = round(i * BIN, 2)
        grp = by.get(lo, [])
        n = len(grp)
        k = int(sum(r["result"] for r in grp))
        mean_px = None if not grp else float(np.mean([FillOf(r, fill_kind) for r in grp]))
        obs = None if not n else k / n
        # unit-stake ROI at this fill + $0.02, using actual yes/no
        pnls, stakes = [], []
        for r in grp:
            fill = FillOf(r, fill_kind)
            cost = fill + FEE
            if cost <= 0 or cost >= 1:
                continue
            stake = 1.0
            pnl = (stake / cost) * r["result"] - stake
            pnls.append(pnl)
            stakes.append(stake)
        staked = sum(stakes)
        roi = None if not staked else sum(pnls) / staked
        # CIs on observed rate and on ROI (treat each contract; also game cluster)
        cp = ClopperPearson(k, n, alpha=ALPHA) if n else (None, None)
        # game cluster for hit rate
        by_g = defaultdict(list)
        for r in grp:
            by_g[r["event_id"]].append(r)
        g_means = [np.mean([x["result"] for x in v]) for v in by_g.values()]
        g_ci = (None, None)
        if g_means:
            arr = np.asarray(g_means, dtype=float)
            idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
            means = arr[idx].mean(1)
            g_ci = (float(np.quantile(means, ALPHA / 2)), float(np.quantile(means, 1 - ALPHA / 2)))
        day_means = []
        by_d = defaultdict(list)
        for r in grp:
            by_d[r["date"]].append(r)
        for v in by_d.values():
            day_means.append(np.mean([x["result"] for x in v]))
        d_ci = (None, None)
        if day_means:
            arr = np.asarray(day_means, dtype=float)
            idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
            means = arr[idx].mean(1)
            d_ci = (float(np.quantile(means, ALPHA / 2)), float(np.quantile(means, 1 - ALPHA / 2)))
        rate_w = cp
        for other in (g_ci, d_ci):
            if other[0] is not None:
                rate_w = Widest(rate_w, other) if rate_w[0] is not None else other
        roi_ci = RoiBoot(pnls, stakes, rng, BOOT, ALPHA) if pnls else (None, None)
        # game-cluster ROI
        g_p, g_s = [], []
        for v in by_g.values():
            gp, gs = 0.0, 0.0
            for r in v:
                fill = FillOf(r, fill_kind)
                cost = fill + FEE
                if cost <= 0 or cost >= 1:
                    continue
                gp += (1.0 / cost) * r["result"] - 1.0
                gs += 1.0
            if gs:
                g_p.append(gp)
                g_s.append(gs)
        roi_g = RoiBoot(g_p, g_s, rng, BOOT, ALPHA) if g_s else (None, None)
        # day-block ROI
        d_p, d_s = [], []
        for v in by_d.values():
            dp, ds = 0.0, 0.0
            for r in v:
                fill = FillOf(r, fill_kind)
                cost = fill + FEE
                if cost <= 0 or cost >= 1:
                    continue
                dp += (1.0 / cost) * r["result"] - 1.0
                ds += 1.0
            if ds:
                d_p.append(dp)
                d_s.append(ds)
        roi_d = RoiBoot(d_p, d_s, rng, BOOT, ALPHA) if d_s else (None, None)
        blog = [{"pnl": p, "stake": s} for p, s in zip(pnls, stakes)]
        b_lo, b_hi, _ = BinomialRoi(blog) if blog else (None, None, None)
        roi_w = roi_ci
        for other in (roi_g, roi_d, (b_lo, b_hi)):
            if other and other[0] is not None:
                roi_w = Widest(roi_w, other) if roi_w and roi_w[0] is not None else other
        fit = fit_map.get(lo) or {}
        out.append({
            "bin_lo": lo, "bin_hi": BinHi(lo), "fill_kind": fill_kind,
            "n_2025": fit.get("n", 0), "obs_2025": fit.get("obs_rate"),
            "edge_2025": fit.get("edge"), "selected": lo in selected,
            "fair_value_2025": fit.get("fair_value"),
            "n": n, "k": k, "mean_price": mean_px, "obs_rate": obs, "roi": roi,
            "obs_ci95_widest": None if rate_w[0] is None else [rate_w[0], rate_w[1]],
            "obs_cp_95": None if cp[0] is None else [cp[0], cp[1]],
            "obs_game_95": None if g_ci[0] is None else [g_ci[0], g_ci[1]],
            "obs_day_95": None if d_ci[0] is None else [d_ci[0], d_ci[1]],
            "roi_ci95_widest": None if not roi_w or roi_w[0] is None else [roi_w[0], roi_w[1]],
            "p_frozen": p_by_lo.get(lo),
        })
    return out


def Claimable(bank_tim, bank_kalshi):
    """Claim only if widest 95% lower bound > 0 under both fee bases, for both strategies."""
    for strat in ("flat100", "qkelly"):
        a = ((bank_tim.get(strat) or {}).get("ci") or {}).get("widest_95")
        b = ((bank_kalshi.get(strat) or {}).get("ci") or {}).get("widest_95")
        if not a or not b or a[0] is None or b[0] is None:
            return False
        if not (a[0] > 0 and b[0] > 0):
            return False
    return True


def FamilyEval(rows, days, fill_kind, price_basis, rng, label):
    fit_rows = [r for r in rows if r["season"] == "2025"]
    test_rows = [r for r in rows if r["season"] == "2026"]
    fit_stats, selected, p_by_lo = FitBins(fit_rows, fill_kind)
    edges_t = MakeEdges(test_rows, selected, p_by_lo, fill_kind, price_basis, fee_kind="tim")
    edges_k = MakeEdges(test_rows, selected, p_by_lo, fill_kind, price_basis, fee_kind="kalshi")
    days_26 = [d for d in days if d >= "2026-08-01"] or sorted({r["date"] for r in test_rows})
    bank_t, daily_t, log_t = BankBundle(edges_t, days_26, rng, "tim")
    bank_k, daily_k, log_k = BankBundle(edges_k, days_26, rng, "kalshi")
    bins_26 = PerBinHoldout(test_rows, fit_stats, selected, p_by_lo, fill_kind, rng)
    return {
        "label": label, "fill_kind": fill_kind, "price_basis": price_basis,
        "n_2025": len(fit_rows), "n_2026": len(test_rows),
        "n_games_2025": len({r["event_id"] for r in fit_rows}),
        "n_games_2026": len({r["event_id"] for r in test_rows}),
        "fit_bins": fit_stats, "selected_bins": selected,
        "holdout_bins": bins_26,
        "bank_tim": bank_t, "bank_kalshi": bank_k,
        "daily_tim": daily_t, "claimable": Claimable(bank_t, bank_k),
        "n_bets_flat": bank_t["flat100"]["n_bets"],
    }


def MakerFeeFinding(fees):
    series = (fees or {}).get("series") or fees or {}
    rows = []
    maker = []
    for t, blk in series.items():
        if t in ("fetched_at",) or not isinstance(blk, dict):
            continue
        ft = blk.get("fee_type")
        changes = blk.get("fee_changes") or {}
        arr = changes.get("series_fee_change_arr") if isinstance(changes, dict) else None
        types = [ft] if ft else []
        if arr:
            types += [c.get("fee_type") for c in arr if c.get("fee_type")]
        has_maker = any(t_ and "maker" in str(t_).lower() for t_ in types)
        maker.append(has_maker)
        rows.append({
            "series_ticker": t, "title": blk.get("title"), "market_kind": blk.get("market_kind"),
            "fee_type": ft, "fee_multiplier": blk.get("fee_multiplier"),
            "fee_fields": blk.get("fee_fields"), "fee_change_types": types,
            "charges_maker_fees": has_maker,
        })
    if not rows:
        verdict = "No series fee fields were returned by the API."
    elif all(r["charges_maker_fees"] for r in rows):
        verdict = "Every totals/spreads series reports a fee_type that includes maker fees (quadratic_with_maker_fees or similar)."
    elif any(r["charges_maker_fees"] for r in rows):
        verdict = "Some, but not all, totals/spreads series report maker fees in the API fee fields."
    else:
        verdict = "API fee fields for these series do not mention maker fees (fee_type has no 'maker' token)."
    return {"rows": rows, "verdict": verdict, "n_series": len(rows)}


def SoftnessVerdict(line_ask, ml_ask, cov_lines, cov_ml):
    """Pre-registered comparison: do totals/spreads look softer than moneylines after spread cost?"""
    def roi(bundle):
        return (bundle.get("bank_tim") or {}).get("flat100", {}).get("roi")

    def lo(bundle):
        ci = ((bundle.get("bank_tim") or {}).get("flat100") or {}).get("ci") or {}
        w = ci.get("widest_95")
        return None if not w else w[0]

    r_line, r_ml = roi(line_ask), roi(ml_ask)
    parts = []
    claim_line = line_ask.get("claimable")
    claim_ml = ml_ask.get("claimable")
    if claim_line and not claim_ml:
        parts.append("Totals/spreads meet the claim bar on the 2026 holdout; moneylines do not.")
        softer = True
    elif claim_line and claim_ml:
        parts.append("Both families meet the claim bar; this is not evidence that lines are uniquely soft.")
        softer = False
    elif not claim_line:
        parts.append("Totals/spreads do not meet the claim bar (widest 95% CI lower bound is not above 0 under both fee bases).")
        softer = False
    else:
        softer = False
    if r_line is not None and r_ml is not None:
        parts.append(
            f"Holdout flat100 ROI (ask + $0.02, Tim's estimate): lines {r_line:.1%} vs moneylines {r_ml:.1%}."
        )
        if r_line > r_ml + 0.02 and not claim_line:
            parts.append("Lines had a higher point ROI than moneylines, but the interval does not clear zero, so this is not a claim of edge.")
    if not line_ask.get("selected_bins"):
        parts.append("No 2025 price bin cleared the pre-specified n>=20 and 3c edge screen at ask+$0.02, so the 2026 bankroll has no selected bets.")
        softer = False
    return {
        "softer_than_moneylines": softer,
        "claim_edge_lines": bool(claim_line),
        "claim_edge_moneylines": bool(claim_ml),
        "text": " ".join(parts) if parts else "Insufficient data for a verdict.",
    }


def WriteBankChart(daily_map, path, title):
    rows = []
    for series, daily in daily_map:
        if not daily:
            continue
        for d in daily:
            rows.append({
                "date": d["date"], "series": series, "equity": d["equity"],
                "drawdown_pct": d["drawdown_pct"], "season": d.get("season") or "2026",
            })
    if not rows:
        return
    frame = FrameTidy(rows, date_key="date", equity_key="equity", series_key="series", mode="auto")
    PlotBankroll(frame, path=path, title=title, ylabel="Bankroll ($)",
                 hlines=[(START, "$10,000 start")], figsize=(11, 6.5))
    import matplotlib.pyplot as plt
    plt.close("all")


def WriteBinChart(holdout, fit, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, stats, lab in (
        (axes[0], fit, "2025 fit (contaminated)"),
        (axes[1], holdout, "2026 holdout (forward)"),
    ):
        xs = [s["bin_lo"] + BIN / 2 for s in stats if s.get("n")]
        ys = [s.get("obs_rate") for s in stats if s.get("n")]
        ns = [s.get("n") for s in stats if s.get("n")]
        sel = [s.get("selected") for s in stats if s.get("n")]
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1, label="perfect")
        if xs:
            colors = ["#d62728" if s else "#1f77b4" for s in sel]
            sizes = [max(20, min(200, n)) for n in ns]
            ax.scatter(xs, ys, s=sizes, c=colors, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("T-60 fill price")
        ax.set_ylabel("Observed yes rate")
        ax.set_title(lab, loc="left")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def FmtPct(x, d=1):
    if x is None:
        return "n/a"
    return f"{100 * x:.{d}f}%"


def FmtCi(ci):
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{FmtPct(ci[0])}, {FmtPct(ci[1])}]"


def FmtMoney(x):
    if x is None:
        return "n/a"
    return f"${x:,.2f}"


def FmtFair(x):
    if x is None:
        return "n/a"
    return f"${x:.2f}"


def MdReport(meta):
    line = meta["line_ask"]
    mid = meta["line_mid"]
    ml = meta["ml_ask"]
    mlmid = meta["ml_mid"]
    ver = meta["verdict"]
    fee = meta["maker_fees"]
    cov = meta["coverage"]
    dropped = meta.get("dropped") or {}
    b = line["bank_tim"]
    bk = line["bank_kalshi"]
    lines = [
        "# NFL Kalshi thin-markets test (totals and spreads vs moneylines)",
        "",
        f"Made by American Footbot, {meta['written_at']}. Code `{meta['code']}`, run_id `{meta['run_id']}`, "
        f"model_version `{meta['model_version']}`. Simulation only; no orders, no real wagers.",
        "",
        "## Pre-registered setup",
        "- **Hypothesis:** Kalshi NFL totals and spreads at T-60 are softer than moneylines after bid/ask width and fees.",
        "- **Fit** on 2025 only (contaminated; freeze after selection). **Test** on 2026 completed games with a T-60 quote.",
        "- **Favourite-longshot** in 0.05 price bins. The frozen probability is the 2025 hit rate in that bin. "
        f"A bin is selected only if N >= {MIN_BIN_N} and edge = hit rate − (fill + $0.02) >= {MIN_EDGE:.2f}.",
        "- **Primary fill = ask** + Tim's $0.02 fee estimate (the registered test). "
        "**Secondary fill = mid** + $0.02, per Tim's mid-fill rule. Qualifying always uses the $0.02 rule so the "
        "verified Kalshi taker-fee path is a recompute of the same bets.",
        "- **CIs** are 95%. We report day-block bootstrap, game-cluster bootstrap, and an exact binomial interval "
        "(Clopper-Pearson on hits, mapped through mean win/loss sizes for ROI) and use the **widest**.",
        f"- **Bankroll:** one $10,000 bank on the 2026 holdout only. `flat100` is $100 a bet; `qkelly` is 0.25×Kelly "
        f"capped at 2% of start-of-day equity. Ruin at equity below ${RUIN:.0f} for flat100.",
        f"- **Trials:** {meta['trials']}. {TRIALS_NOTE}",
        "- Fees: `tim_est_0.02` is **Tim's fee estimate**, not a verified exchange fee. The verified Kalshi taker fee "
        "is 0.07×p×(1−p), at most $0.0175 per contract, labelled separately. Maker-fee findings are in their own section.",
        "",
        "## Verdict",
        ver["text"],
        f"- Claim an edge on totals/spreads? **{'yes' if ver['claim_edge_lines'] else 'no'}**.",
        f"- Claim an edge on moneylines under the same rule? **{'yes' if ver['claim_edge_moneylines'] else 'no'}**.",
        f"- Softer than moneylines, with a claimable edge? **{'yes' if ver['softer_than_moneylines'] else 'no'}**.",
        "",
        "## Coverage and T-60 bid/ask width",
        "",
        "| Kind | Season | N contracts | N games | Median width | p25–p75 width | Median T-60 candle volume |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cov:
        med = "n/a" if c["median_width"] is None else f"{c['median_width']:.3f}"
        iqr = "n/a" if c["p25_width"] is None else f"{c['p25_width']:.3f}–{c['p75_width']:.3f}"
        vol = "n/a" if c["median_volume_t60"] is None else f"{c['median_volume_t60']:.0f}"
        lines.append(f"| {c['market_kind']} | {c['season']} | {c['n']} | {c['n_games']} | {med} | {iqr} | {vol} |")
    lines += [
        "",
        f"Overall median T-60 width, totals+spreads: **{meta['median_width_lines']:.3f}** dollars. "
        f"Moneylines: **{meta['median_width_ml']:.3f}**." if meta.get("median_width_lines") is not None and meta.get("median_width_ml") is not None
        else "",
        f"Dropped (lines): {dropped.get('lines')}. Dropped (moneylines): {dropped.get('moneyline')}.",
        "Width is yes_ask − yes_bid on the last 1-minute candle at or before T-60, stale if older than 30 minutes. "
        "Candles have no size; T-60 volume is the candle's volume field (definition as Kalshi ships it).",
        "",
        "## Maker fees",
        fee["verdict"],
    ]
    for r in fee["rows"]:
        lines.append(f"- `{r['series_ticker']}` ({r.get('market_kind')}): fee_type=`{r.get('fee_type')}`, "
                     f"multiplier=`{r.get('fee_multiplier')}`, maker_fees={r['charges_maker_fees']}.")
    lines += [
        "",
        "## 2025 selection (ask + $0.02), frozen before 2026",
        "",
        "| Bin | N | Hit rate | Mean ask | Edge vs ask+$0.02 | Fair value | Selected? |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in line["fit_bins"]:
        if s["n"] == 0:
            continue
        lines.append(
            f"| {s['bin_lo']:.2f}–{s['bin_hi']:.2f} | {s['n']} | {FmtPct(s['obs_rate'])} | "
            f"{s['mean_price']:.3f} | {FmtPct(s['edge'])} | {FmtFair(s['fair_value'])} | "
            f"{'yes' if s['selected'] else 'no'} |"
        )
    lines += [
        f"",
        f"Selected bins: {line['selected_bins'] or 'none'}.",
        "",
        "## 2026 holdout per-bin results (ask + $0.02, Tim's estimate)",
        "",
        "| Bin | Selected | N | Hit rate (widest 95% CI) | ROI (widest 95% CI) | 2025 fair value |",
        "|---|---|---|---|---|---|",
    ]
    for s in line["holdout_bins"]:
        if s["n"] == 0:
            continue
        lines.append(
            f"| {s['bin_lo']:.2f}–{s['bin_hi']:.2f} | {'yes' if s['selected'] else 'no'} | {s['n']} | "
            f"{FmtPct(s['obs_rate'])} {FmtCi(s['obs_ci95_widest'])} | "
            f"{FmtPct(s.get('roi'))} {FmtCi(s.get('roi_ci95_widest'))} | {FmtFair(s.get('fair_value_2025'))} |"
        )
    def bank_table(title, bundle, bundle_k):
        out = [
            f"## {title}",
            "",
            "| Strategy | Fee | N bets | End bank | P&L | ROI | Widest 95% CI | Day-block 95% | Max DD | Ruin? |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, bank, fee_lab in (
            ("flat100", bundle["flat100"], "tim_est_0.02"),
            ("qkelly", bundle["qkelly"], "tim_est_0.02"),
            ("flat100", bundle_k["flat100"], "kalshi_verified"),
            ("qkelly", bundle_k["qkelly"], "kalshi_verified"),
        ):
            ci = (bank.get("ci") or {}).get("widest_95")
            day = (bank.get("ci") or {}).get("day_block_95")
            out.append(
                f"| {name} | {fee_lab} | {bank['n_bets']} | {FmtMoney(bank['final_equity'])} | "
                f"{FmtMoney(bank['pnl'])} | {FmtPct(bank['roi'])} | {FmtCi(ci)} | {FmtCi(day)} | "
                f"{FmtPct(bank['max_drawdown_pct'])} | {'yes' if bank['ruined'] else 'no'} |"
            )
        return out
    lines += [""] + bank_table(
        "2026 bankroll, totals+spreads, ask fill (primary)",
        line["bank_tim"], line["bank_kalshi"],
    )
    lines += [""] + bank_table(
        "2026 bankroll, totals+spreads, mid fill (secondary, Tim's fill rule)",
        mid["bank_tim"], mid["bank_kalshi"],
    )
    lines += [""] + bank_table(
        "2026 bankroll, moneylines, ask fill (comparison)",
        ml["bank_tim"], ml["bank_kalshi"],
    )
    lines += [
        "",
        "## Mid-fill selection and moneyline notes",
        f"- Lines, mid fill, selected 2025 bins: {mid['selected_bins'] or 'none'}; 2026 flat100 bets={mid['n_bets_flat']}.",
        f"- Moneylines, ask fill, selected 2025 bins: {ml['selected_bins'] or 'none'}; 2026 flat100 bets={ml['n_bets_flat']}.",
        f"- Moneylines, mid fill, selected 2025 bins: {mlmid['selected_bins'] or 'none'}; 2026 flat100 bets={mlmid['n_bets_flat']}.",
        "",
        "## Caveats for Graphtard to caption",
        "- Simulated. 2026 holdout only for the bankroll; $10,000 start, no reset mid-season.",
        "- 2025 is contaminated by earlier strategy searches (tally at least 2,819) even though this exact rule was pre-registered today.",
        "- Many contracts can share one game; CIs are the widest of day-block, game-cluster, and binomial. Small N is still small N.",
        "- Candles have no size, so fills at the top of book (ask or mid) are optimistic for large stakes.",
        "- `tim_est_0.02` is Tim's fee estimate. If a series also charges maker fees, a maker fill would cost less than the taker schedule, but this test fills at the ask or mid and still applies the fee basis named above.",
        "- Do not claim an edge unless the widest 95% lower bound is above 0 under both fee bases on a serious sample.",
        f"- Weeks covered: {meta.get('week_range')}.",
    ]
    return "\n".join(lines) + "\n"


def TidyBinRows(meta, last_date):
    rows = []
    for fam, bundle in (("lines", meta["line_ask"]), ("moneyline", meta["ml_ask"])):
        for split, key in (("contaminated", "fit_bins"), ("forward", "holdout_bins")):
            for s in bundle[key]:
                if not s.get("n"):
                    continue
                rows.append({
                    "date": last_date, "sport": SPORT,
                    "series": f"nfl_{fam}_{bundle['fill_kind']}_{split}",
                    "fill_kind": bundle["fill_kind"], "family": fam, "split": s.get("split") or split,
                    "bin_lo": s["bin_lo"], "bin_hi": s["bin_hi"], "n": s["n"],
                    "mean_price": s.get("mean_price"), "obs_rate": s.get("obs_rate") or s.get("obs_rate"),
                    "roi": s.get("roi"), "selected": int(bool(s.get("selected"))),
                    "fair_value_2025": s.get("fair_value") or s.get("fair_value_2025"),
                    "edge_2025": s.get("edge") or s.get("edge_2025"),
                })
    return rows


def WeekRange(rows):
    if not rows:
        return "none"
    by = defaultdict(list)
    for r in rows:
        by[r["season"]].append(r.get("week"))
    parts = []
    for s in sorted(by):
        weeks = [w for w in by[s] if w is not None]
        if not weeks:
            parts.append(f"{s} (no week labels)")
        else:
            parts.append(f"{s} weeks {min(weeks)}–{max(weeks)}")
    return "; ".join(parts)


def Main():
    global BOOT
    ap = argparse.ArgumentParser(description="Thin-markets favourite-longshot test (run when Tim approves).")
    ap.add_argument("--out-dir", default=r"E:\AgentDocs\American-Footbot\2026-11-13_thin-markets")
    ap.add_argument("--base", default="2026-11-13_thin-markets")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--data-root", default="data",
                    help="directory that contains kalshi/thin_*.jsonl (moneyline candles still use data/)")
    ap.add_argument("--skip-rules", action="store_true",
                    help="skip the model-free rule search (original FLB study only)")
    ap.add_argument("--normal", action="store_true", help="leave process at Normal priority")
    ap.add_argument("--workers", type=int, default=max(1, (__import__('os').cpu_count() or 4) // 4),
                    help="workers for the model-free rule search (default cpu/4)")
    ap.add_argument("--pull-limit", type=int, default=0)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if not a.normal:
        SetBelowNormal()
    ForceIPv4()
    BOOT = a.boot
    run_id = NewRunId()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.base)
    written = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p %Z")

    pull_summary = None
    if not a.skip_pull:
        from DataPull.PullKalshiThinMarkets import Main as PullMain
        argv = ["--seasons", "2025,2026"]
        if a.pull_limit:
            argv += ["--limit", str(a.pull_limit)]
        pull_summary = PullMain(argv)

    fees = LoadFees(root=a.data_root)
    markets = LoadMarkets(root=a.data_root)
    entries = LoadThinEntries(root=a.data_root)
    line_rows, drop_lines = ContractRows(entries, markets, kinds=("total", "spread"))
    ml_rows, drop_ml = MoneylineRows()
    # Keep 2026 completed games only (those with a result already required).
    cov = Coverage(line_rows + ml_rows)
    maker = MakerFeeFinding(fees)
    days_lines = sorted({r["date"] for r in line_rows})
    days_ml = sorted({r["date"] for r in ml_rows})

    line_ask = FamilyEval(line_rows, days_lines, "ask", PRICE_ASK, rng, "lines_ask")
    line_mid = FamilyEval(line_rows, days_lines, "mid", PRICE_MID, rng, "lines_mid")
    ml_ask = FamilyEval(ml_rows, days_ml, "ask", PRICE_ASK, rng, "ml_ask")
    ml_mid = FamilyEval(ml_rows, days_ml, "mid", PRICE_MID, rng, "ml_mid")
    verdict = SoftnessVerdict(line_ask, ml_ask, cov, cov)

    med_lines = MedianWidth(line_rows)
    med_ml = MedianWidth(ml_rows)
    meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "written_at": written, "fee_basis": FEE_BASIS, "fill_primary": "ask",
        "fill_secondary": "mid", "min_edge": MIN_EDGE, "min_bin_n": MIN_BIN_N,
        "bin_width": BIN, "ci_alpha": ALPHA, "boot": BOOT, "seed": a.seed,
        "start": START, "trials": TRIALS_BANKROLL, "trials_hypotheses": TRIALS_HYPOTHESES,
        "trials_note": TRIALS_NOTE, "contamination_tally_at_least": 2819,
        "coverage": cov, "maker_fees": maker,
        "line_ask": _StripDaily(line_ask), "line_mid": _StripDaily(line_mid),
        "ml_ask": _StripDaily(ml_ask), "ml_mid": _StripDaily(ml_mid),
        "verdict": verdict, "median_width_lines": med_lines, "median_width_ml": med_ml,
        "dropped": {"lines": drop_lines, "moneyline": drop_ml},
        "week_range": WeekRange(line_rows + ml_rows),
        "n_line_rows": len(line_rows), "n_ml_rows": len(ml_rows),
        "pull_http_calls": None if not pull_summary else pull_summary.get("http_calls"),
        "headline": verdict["text"],
    }

    last = (days_lines[-1] if days_lines else days_ml[-1] if days_ml else written[:10])
    WriteCsv(base + ".csv", [
        "date", "sport", "series", "fill_kind", "family", "split", "bin_lo", "bin_hi",
        "n", "mean_price", "obs_rate", "roi", "selected", "fair_value_2025", "edge_2025",
    ], TidyBinRows(meta, last))
    bank_rows = []
    for series, daily in (
        ("flat100 / lines ask", line_ask["daily_tim"].get("flat100") or []),
        ("qkelly / lines ask", line_ask["daily_tim"].get("qkelly") or []),
        ("flat100 / lines mid", line_mid["daily_tim"].get("flat100") or []),
        ("qkelly / lines mid", line_mid["daily_tim"].get("qkelly") or []),
        ("flat100 / moneyline ask", ml_ask["daily_tim"].get("flat100") or []),
        ("qkelly / moneyline ask", ml_ask["daily_tim"].get("qkelly") or []),
    ):
        for d in daily:
            bank_rows.append({**d, "series": series})
    WriteCsv(base + "_bankroll.csv", BANK_FIELDS + (["x"] if "x" not in BANK_FIELDS else []), bank_rows)
    WriteCsv(base + "_coverage.csv", [
        "market_kind", "season", "n", "n_games", "n_days",
        "median_width", "p25_width", "p75_width", "median_volume_t60", "mean_width",
    ], cov)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=_JsonDefault)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdReport(meta))

    WriteBankChart(
        [("flat100 / lines ask", line_ask["daily_tim"].get("flat100") or []),
         ("qkelly / lines ask", line_ask["daily_tim"].get("qkelly") or []),
         ("flat100 / moneyline ask", ml_ask["daily_tim"].get("flat100") or []),
         ("qkelly / moneyline ask", ml_ask["daily_tim"].get("qkelly") or [])],
        base + "_bankroll_draft.png",
        "NFL $10k 2026 holdout: totals/spreads vs moneylines at Kalshi T-60 ask, tim_est_0.02",
    )
    WriteBinChart(line_ask["holdout_bins"], line_ask["fit_bins"], base + "_bins_draft.png",
                  "Favourite-longshot calibration, totals+spreads, ask fill")

    rules_meta = None
    if not a.skip_rules:
        from Backtest.ThinMarketRules import RunRules
        print("starting model-free rule search...", flush=True)
        rules_meta = RunRules(
            a.out_dir, base=base, entries=entries, markets=markets, fees=fees,
            workers=a.workers, seed=a.seed,
        )
        # Point the original report at the rule-search companion files.
        try:
            pointer = [
                "",
                "",
                "## Model-free rule search",
            ]
            if rules_meta.get("partial"):
                missing = ", ".join(rules_meta.get("missing") or [])
                pointer.append(f"**PARTIAL COVERAGE: missing {missing}.**")
                pointer.append("")
            pointer.append(
                f"See `{a.base}_rules.md`, `_rules_grid.csv`, `_rules_top50.csv`, and `_rules_mc.csv`. "
                f"Grid size {rules_meta.get('n_configs')}; new lower-bound tally {rules_meta.get('new_tally')}."
            )
            pointer.append(f"Verdict: {rules_meta.get('verdict')}")
            with open(base + ".md", "a", encoding="utf-8") as fh:
                fh.write("\n".join(pointer) + "\n")
        except OSError as exc:
            print("could not append rule-search pointer to md:", exc, flush=True)

    print(json.dumps({
        "run_id": run_id, "code": CodeVersion(),
        "median_width_lines": med_lines, "median_width_ml": med_ml,
        "selected_bins_ask": line_ask["selected_bins"],
        "n_bets_flat_ask": line_ask["n_bets_flat"],
        "bank_flat_ask": line_ask["bank_tim"]["flat100"],
        "verdict": verdict, "maker_fees": maker["verdict"],
        "out": a.out_dir,
        "rules_best": None if not rules_meta else (rules_meta.get("best") or {}).get("id"),
        "rules_spa_p": None if not rules_meta else (rules_meta.get("spa") or {}).get("spa_p"),
        "rules_tally": None if not rules_meta else rules_meta.get("new_tally"),
        "rules_partial": None if not rules_meta else rules_meta.get("partial"),
    }, indent=1, default=_JsonDefault), flush=True)


def _StripDaily(bundle):
    out = dict(bundle)
    out.pop("daily_tim", None)
    return out


def _JsonDefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    Main()
