"""
Common cross-sport calibration, $10k bankroll and edge files (Degen plan, 2026-11-13).

Headline model: p_full2_kmid_T60 -- walk-forward WIN_FEATS2 weights (fitted on seasons < S,
as in Lab.AttachModels) with the market input swapped for the Kalshi de-vigged mid at T-60.
That is leak-safer than p_full2 as-fitted, whose mkt_logit is the nflverse closing moneyline
(not known at T-60). Residual look-ahead remains in spread_gap (it still uses the closing spread).

    .venv/Scripts/python.exe -m Backtest.CommonFormat --cal-dir DIR --bank-dir DIR

Fill price is the side's mid (Tim, 2026-11-13), not the plan's ask. Fee label is tim_est_0.02
(Tim's estimate, not a verified exchange fee). The verified Kalshi taker fee 0.07*p*(1-p) is
shown alongside, never instead. Simulation only; no orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import groupby
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.BankrollPlot import PlotBankroll
from Backtest.Calibration import ReliabilityBins, Scores as CloseScores
from Backtest.ForwardWalk import Kickoff as FwKickoff, SeenBySearch
from Backtest.FrozenConfig import FROZEN, IsForwardHoldout
from Backtest.KalshiReality import KalshiFee, LoadQuotes, QuoteAt, RefitWeights
from Backtest.Lab import AllRows, AttachModels, Sig, Vec, WIN_FEATS2
from Backtest.Version import CodeVersion, ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
SPORT = "nfl"
MODEL = "p_full2_kmid_T60"
MARKET = "market_kalshi_T60"
BOOK_MODEL = "p_full2"
BOOK_MARKET = "market_book_close"
PRICE_KALSHI = "kalshi_T60"
PRICE_BOOK = "book_close_devig"
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
MIN_EDGE = 0.03
START = 10_000.0
FLAT = 100.0
QKELLY_FRAC = 0.25
QKELLY_CAP = 0.02
T60 = 60
STALE_MIN = 30
BOOT = 2000
SEED = 20261113
ALPHA = 0.10
RUIN = 100.0

CAL_FIELDS = ["date", "sport", "series", "model", "season", "split", "bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"]
SUM_FIELDS = ["date", "sport", "series", "model", "season", "split", "n", "accuracy", "brier", "log_loss", "ece"]
BANK_FIELDS = ["date", "sport", "series", "strategy", "price_basis", "season", "split", "x",
               "bets_day", "bets_cum", "equity", "drawdown_pct"]
EDGE_FIELDS = ["date", "start_time", "sport", "season", "split", "event_id", "event", "side", "model",
               "model_prob", "market_prob_devig", "fill_price", "price_basis", "price_time", "fee",
               "fee_basis", "edge", "edge_vs_fill", "qualifies", "result", "pnl_at_flat_stake"]
EDGE_EXTRA = ["fee_kalshi_verified", "edge_vs_fill_kalshi_fee", "pnl_at_flat_stake_kalshi_fee",
              "fav_side_fill", "fav_result", "seen_by_search", "forward_holdout"]


def ClipP(p):
    return min(max(float(p), 1e-6), 1 - 1e-6)


def Mid(bid, ask):
    return (float(bid) + float(ask)) / 2


def Devig(home_mid, away_mid):
    tot = home_mid + away_mid
    if tot <= 0:
        return None
    return ClipP(home_mid / tot)


def QuoteWithTs(entry, minutes=T60):
    q = QuoteAt(entry, minutes)
    if not q:
        return None
    ko = entry["ko"]
    target = (ko - timedelta(minutes=minutes)).timestamp()
    end_ts = target - q["age_min"] * 60
    q = dict(q)
    q["mid"] = Mid(q["bid"], q["ask"])
    q["end_ts"] = end_ts
    q["price_time"] = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()
    return q


def IsoKickoff(r):
    return FwKickoff(r).isoformat()


def SplitLabel(r):
    if int(r["season"]) == 2025:
        return "contaminated"
    if int(r["season"]) == 2026:
        return "forward"
    if int(r["season"]) in FROZEN["contaminated_test_seasons"]:
        return "contaminated"
    lo, hi = FROZEN["tune_seasons"]
    if lo <= int(r["season"]) <= hi:
        return "tune"
    return "validation"


def SeasonLabel(r):
    return str(int(r["season"]))


def ClopperPearson(k, n, alpha=ALPHA):
    """Exact binomial (Clopper-Pearson) interval. Returns (lo, hi) or (None, None) if n=0."""
    if n <= 0:
        return None, None
    from math import isfinite
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        if not isfinite(lo):
            lo = 0.0
        if not isfinite(hi):
            hi = 1.0
        return lo, hi
    except Exception:
        # Wilson fallback if scipy is missing
        z = 1.6448536269514722
        ph = k / n
        d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return max(0.0, c - h), min(1.0, c + h)


def Widest(a, b):
    """Return the wider of two (lo, hi) intervals, or the one that exists."""
    if a[0] is None and b[0] is None:
        return None, None
    if a[0] is None:
        return b
    if b[0] is None:
        return a
    lo = min(a[0], b[0])
    hi = max(a[1], b[1])
    return lo, hi


def BootstrapMean(vals, rng, boot=BOOT, alpha=ALPHA):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None, None
    idx = rng.integers(0, len(vals), size=(boot, len(vals)))
    means = vals[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def BootstrapRoi(pnls, stakes, rng, boot=BOOT, alpha=ALPHA):
    pnls = np.asarray(pnls, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    n = len(pnls)
    if n == 0 or stakes.sum() == 0:
        return None, None
    idx = rng.integers(0, n, size=(boot, n))
    rois = pnls[idx].sum(1) / np.where(stakes[idx].sum(1) == 0, np.nan, stakes[idx].sum(1))
    rois = rois[np.isfinite(rois)]
    if len(rois) == 0:
        return None, None
    return float(np.quantile(rois, alpha / 2)), float(np.quantile(rois, 1 - alpha / 2))


def Ece(rows, key):
    bins = ReliabilityBins(rows, key)
    n = sum(b["n"] for b in bins)
    if not n:
        return None
    return sum(abs(b["mean_pred"] - b["obs_rate"]) * b["n"] for b in bins if b["n"]) / n


def ScoreKey(rows, key):
    mapped = [{"home_win": r["home_win"], key: r[key]} for r in rows if r["home_win"] is not None]
    s = CloseScores(mapped, key)
    s["ece"] = None if not mapped else round(Ece(mapped, key), 6)
    if s["brier"] is not None:
        s["brier"] = round(s["brier"], 6)
    if s["log_loss"] is not None:
        s["log_loss"] = round(s["log_loss"], 6)
    if s["accuracy"] is not None:
        s["accuracy"] = round(s["accuracy"], 6)
    return s, ReliabilityBins(mapped, key)


def ClusterObsCi(rows, key, lo, hi, rng, boot=BOOT):
    """Game-cluster (here: one row per game) bootstrap 90% CI for the bin's observed rate."""
    inb = [r for r in rows if lo <= r[key] < hi or (hi == 1.0 and r[key] == 1.0)]
    ys = np.array([r["home_win"] for r in inb], dtype=float)
    if len(ys) == 0:
        return None, None, 0, 0
    return (*BootstrapMean(ys, rng, boot), int(ys.sum()), len(ys))


def FlatPnl(fill, result, fee=FEE, stake=FLAT):
    cost = fill + fee
    if cost <= 0 or cost >= 1:
        return None
    contracts = stake / cost
    return contracts * result - stake


def QKellyStake(p, fill, bank, fee=FEE, frac=QKELLY_FRAC, cap=QKELLY_CAP):
    cost = fill + fee
    if cost <= 0 or cost >= 1 or bank <= 0:
        return 0.0
    kelly = (p - cost) / (1 - cost)
    return max(0.0, min(frac * kelly, cap) * bank)


def AttachKmid(rbs, seasons=(2025, 2026)):
    """Add p_kmid and kalshi T-60 quotes. Restores mkt_logit after each game."""
    quotes = {s: LoadQuotes(s) for s in seasons}
    check = {}
    for s in seasons:
        base = {r["game_id"]: r["p_full2"] for r in rbs[s]}
        w2, _w3 = RefitWeights(rbs, s)
        diffs = [abs(float(Sig(np.dot(w2, Vec(r, WIN_FEATS2)))) - base[r["game_id"]]) for r in rbs[s]]
        check[s] = {"refit_max_abs_diff_vs_attachmodels": round(max(diffs) if diffs else 0.0, 8)}
        for r in rbs[s]:
            g = quotes[s].get(r["game_id"], {})
            qh, qa = QuoteWithTs(g.get(r["home_team"])), QuoteWithTs(g.get(r["away_team"]))
            r["kq_t60"] = None
            r["p_kmid"] = None
            r["kmid"] = None
            if not (qh and qa):
                continue
            r["kq_t60"] = {"home": qh, "away": qa}
            pk = Devig(qh["mid"], qa["mid"])
            r["kmid"] = pk
            saved = r["mkt_logit"]
            r["mkt_logit"] = math.log(pk / (1 - pk))
            r["p_kmid"] = float(Sig(np.dot(w2, Vec(r, WIN_FEATS2))))
            r["mkt_logit"] = saved
    return check


def Eligible(r, week_cap=9):
    if r["home_win"] is None:
        return False
    if int(r["season"]) == 2026 and int(r["week"]) > week_cap:
        return False
    return True


def KalshiRows(rbs):
    out = []
    dropped_missing, ties = [], 0
    for s in (2025, 2026):
        for r in rbs[s]:
            if int(r["season"]) == 2026 and int(r["week"]) > 9:
                continue
            if r["home_win"] is None:
                if r.get("result") == 0:
                    ties += 1
                continue
            if not r.get("kq_t60"):
                dropped_missing.append({"game_id": r["game_id"], "gameday": r["gameday"],
                                        "week": r["week"], "matchup": f"{r['away_team']} @ {r['home_team']}"})
                continue
            out.append(r)
    return out, dropped_missing, ties


def BookRows(rbs, first=2014, last=2026):
    out, ties, dropped = [], 0, 0
    for s in range(first, last + 1):
        for r in rbs[s]:
            if int(r["season"]) == 2026 and int(r["week"]) > 9:
                continue
            if r["home_win"] is None:
                if r.get("result") == 0:
                    ties += 1
                continue
            if r.get("mkt_home") is None or r.get("p_full2") is None:
                dropped += 1
                continue
            out.append(r)
    return out, ties, dropped


def MakeEdge(r, model_p, market_p, fill, price_basis, price_time, side, result, fav_fill, fav_result):
    fee_k = KalshiFee(fill) if price_basis == PRICE_KALSHI else None
    edge = model_p - market_p - FEE
    edge_fill = model_p - fill - FEE
    pnl = FlatPnl(fill, result, FEE)
    pnl_k = FlatPnl(fill, result, fee_k) if fee_k is not None else None
    edge_k = (model_p - fill - fee_k) if fee_k is not None else None
    return {
        "date": r["gameday"], "start_time": IsoKickoff(r), "sport": SPORT,
        "season": SeasonLabel(r), "split": SplitLabel(r), "event_id": r["game_id"],
        "event": f"{r['away_team']} @ {r['home_team']}", "side": side, "model": MODEL if price_basis == PRICE_KALSHI else BOOK_MODEL,
        "model_prob": round(model_p, 6), "market_prob_devig": round(market_p, 6),
        "fill_price": round(fill, 6), "price_basis": price_basis, "price_time": price_time,
        "fee": FEE, "fee_basis": FEE_BASIS, "edge": round(edge, 6),
        "edge_vs_fill": round(edge_fill, 6), "qualifies": bool(edge >= MIN_EDGE),
        "result": int(result), "pnl_at_flat_stake": None if pnl is None else round(pnl, 4),
        "fee_kalshi_verified": None if fee_k is None else round(fee_k, 6),
        "edge_vs_fill_kalshi_fee": None if edge_k is None else round(edge_k, 6),
        "pnl_at_flat_stake_kalshi_fee": None if pnl_k is None else round(pnl_k, 4),
        "fav_side_fill": round(fav_fill, 6), "fav_result": int(fav_result),
        "seen_by_search": bool(SeenBySearch(r)), "forward_holdout": bool(IsForwardHoldout(r["gameday"])),
        "_p": model_p, "_fill": fill, "_fee_k": fee_k,
        "_fav_fill": fav_fill, "_fav_result": fav_result, "_fav_p_mkt": None,
    }


def KalshiEdges(rows):
    edges = []
    for r in rows:
        qh, qa = r["kq_t60"]["home"], r["kq_t60"]["away"]
        home_mkt = r["kmid"]
        away_mkt = 1 - home_mkt
        p_home, p_away = r["p_kmid"], 1 - r["p_kmid"]
        e_home = p_home - home_mkt - FEE
        e_away = p_away - away_mkt - FEE
        if e_home >= e_away:
            side, model_p, market_p, fill, result = r["home_team"], p_home, home_mkt, qh["mid"], int(r["home_win"])
            price_time = qh["price_time"]
        else:
            side, model_p, market_p, fill, result = r["away_team"], p_away, away_mkt, qa["mid"], int(1 - r["home_win"])
            price_time = qa["price_time"]
        if home_mkt >= away_mkt:
            fav_fill, fav_result = qh["mid"], int(r["home_win"])
        else:
            fav_fill, fav_result = qa["mid"], int(1 - r["home_win"])
        rec = MakeEdge(r, model_p, market_p, fill, PRICE_KALSHI,
                       price_time, side, result, fav_fill, fav_result)
        rec["_fav_side"] = r["home_team"] if home_mkt >= away_mkt else r["away_team"]
        edges.append(rec)
    edges.sort(key=lambda e: (e["start_time"], e["event_id"]))
    return edges


def BookEdges(rows):
    edges = []
    for r in rows:
        home_mkt = float(r["mkt_home"])
        away_mkt = 1 - home_mkt
        p_home, p_away = float(r["p_full2"]), 1 - float(r["p_full2"])
        e_home = p_home - home_mkt - FEE
        e_away = p_away - away_mkt - FEE
        if e_home >= e_away:
            side, model_p, market_p, fill, result = r["home_team"], p_home, home_mkt, home_mkt, int(r["home_win"])
        else:
            side, model_p, market_p, fill, result = r["away_team"], p_away, away_mkt, away_mkt, int(1 - r["home_win"])
        fav_fill = home_mkt if home_mkt >= away_mkt else away_mkt
        fav_result = int(r["home_win"]) if home_mkt >= away_mkt else int(1 - r["home_win"])
        rec = MakeEdge(r, model_p, market_p, fill, PRICE_BOOK, "",
                       side, result, fav_fill, fav_result)
        rec["price_time"] = ""
        edges.append(rec)
    edges.sort(key=lambda e: (e["date"], e["start_time"], e["event_id"]))
    return edges


def Walk(edges, strategy, fee_kind="tim", days=None):
    """One continuous $10k bank. days = all game dates to emit (bets or not)."""
    by_day = defaultdict(list)
    for e in edges:
        by_day[e["date"]].append(e)
    if days is None:
        days = sorted(by_day)
    bank, peak, bets_cum, ruined = START, START, 0, False
    out, bet_log = [], []
    for i, day in enumerate(days, start=1):
        grp = by_day.get(day, [])
        start_bank = bank
        season = grp[0]["season"] if grp else ("2026" if day >= "2026-08-01" else "2025")
        split = grp[0]["split"] if grp else ("forward" if season == "2026" else "contaminated")
        basis = grp[0]["price_basis"] if grp else PRICE_KALSHI
        picks = []
        if not ruined:
            for e in grp:
                if strategy == "market_fav":
                    fill, result, p = e["_fav_fill"], e["_fav_result"], None
                elif not e["qualifies"]:
                    continue
                else:
                    fill, result, p = e["_fill"], e["result"], e["_p"]
                fee = KalshiFee(fill) if fee_kind == "kalshi" else FEE
                if strategy == "qkelly":
                    stake = QKellyStake(p, fill, start_bank, fee=fee)
                else:
                    if start_bank < RUIN:
                        ruined = True
                        picks = []
                        break
                    stake = FLAT
                if stake < 0.01:
                    continue
                cost = fill + fee
                if cost <= 0 or cost >= 1:
                    continue
                picks.append((e, fill, result, stake, fee, cost))
            total = sum(p[3] for p in picks)
            if total > start_bank > 0:
                scale = start_bank / total
                picks = [(e, f, r, s * scale, fee, c) for e, f, r, s, fee, c in picks]
        pnl_day = 0.0
        for e, fill, result, stake, fee, cost in picks:
            pnl = (stake / cost) * result - stake
            bank += pnl
            pnl_day += pnl
            bets_cum += 1
            bet_log.append({"date": day, "event_id": e["event_id"], "strategy": strategy,
                            "stake": round(stake, 2), "pnl": round(pnl, 2), "fee_kind": fee_kind})
        if strategy == "flat100" and bank < RUIN:
            ruined = True
        if strategy == "qkelly" and bank <= 0:
            ruined = True
            bank = max(bank, 0.0)
        peak = max(peak, bank)
        dd = bank / peak - 1.0 if peak else 0.0
        out.append({
            "date": day, "sport": SPORT,
            "series": f"{SPORT}_{strategy}_{basis}",
            "strategy": strategy, "price_basis": basis, "season": season, "split": split,
            "x": i, "bets_day": len(picks), "bets_cum": bets_cum,
            "equity": round(bank, 2), "drawdown_pct": 0.0 if dd >= 0 else round(dd, 6),
        })
    return out, bet_log, ruined, bank


def AllGameDays(rbs, seasons=(2025, 2026)):
    days = set()
    for s in seasons:
        for r in rbs[s]:
            if int(r["season"]) == 2026 and int(r["week"]) > 9:
                continue
            days.add(r["gameday"])
    return sorted(days)


def BookGameDays(rows):
    return sorted({r["gameday"] for r in rows})


def CalBlock(rows, key, model, season, split, last, rng):
    # rows must already have home_win and the probability key
    scores, bins = ScoreKey(rows, key)
    tidy = []
    bin_ci = []
    for b in bins:
        tidy.append({
            "date": last, "sport": SPORT, "series": f"{SPORT}_{model}", "model": model,
            "season": season, "split": split, "bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"],
            "n": b["n"], "mean_pred": b["mean_pred"], "obs_rate": b["obs_rate"],
        })
        lo_cp, hi_cp = ClopperPearson(int(round((b["obs_rate"] or 0) * b["n"])), b["n"]) if b["n"] else (None, None)
        # use exact k from rows, not rounded
        inb = [r for r in rows if b["bin_lo"] <= r[key] < b["bin_hi"] or (b["bin_hi"] == 1.0 and r[key] == 1.0)]
        k = int(sum(r["home_win"] for r in inb))
        lo_cp, hi_cp = ClopperPearson(k, len(inb)) if inb else (None, None)
        lo_cl, hi_cl, _, n = ClusterObsCi(rows, key, b["bin_lo"], b["bin_hi"], rng)
        lo_w, hi_w = Widest((lo_cp, hi_cp), (lo_cl, hi_cl))
        bin_ci.append({"bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"], "n": b["n"], "k": k,
                       "clopper_pearson_90": None if lo_cp is None else [round(lo_cp, 4), round(hi_cp, 4)],
                       "cluster_boot_90": None if lo_cl is None else [round(lo_cl, 4), round(hi_cl, 4)],
                       "widest_90": None if lo_w is None else [round(lo_w, 4), round(hi_w, 4)]})
    # metric CIs
    p = np.clip(np.array([r[key] for r in rows]), 1e-6, 1 - 1e-6)
    y = np.array([r["home_win"] for r in rows], dtype=float)
    br, ll = (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))
    acc = ((p >= 0.5) == (y == 1)).astype(float)
    brier_ci = BootstrapMean(br, rng)
    ll_ci = BootstrapMean(ll, rng)
    acc_cp = ClopperPearson(int(acc.sum()), len(acc))
    acc_cl = BootstrapMean(acc, rng)
    acc_w = Widest(acc_cp, acc_cl)
    summary = {
        "date": last, "sport": SPORT, "series": f"{SPORT}_{model}", "model": model,
        "season": season, "split": split, "n": scores["n"],
        "accuracy": scores["accuracy"], "brier": scores["brier"],
        "log_loss": scores["log_loss"], "ece": scores["ece"],
    }
    extra = {
        "brier_ci90": None if brier_ci[0] is None else [round(brier_ci[0], 4), round(brier_ci[1], 4)],
        "log_loss_ci90": None if ll_ci[0] is None else [round(ll_ci[0], 4), round(ll_ci[1], 4)],
        "accuracy_clopper_pearson_90": None if acc_cp[0] is None else [round(acc_cp[0], 4), round(acc_cp[1], 4)],
        "accuracy_cluster_boot_90": None if acc_cl[0] is None else [round(acc_cl[0], 4), round(acc_cl[1], 4)],
        "accuracy_widest_90": None if acc_w[0] is None else [round(acc_w[0], 4), round(acc_w[1], 4)],
        "bins_ci": bin_ci,
    }
    return tidy, summary, extra


def WriteCsv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fields}
            w.writerow(row)


def BankStats(daily, bet_log, edges, strategy, rng):
    pnls = [b["pnl"] for b in bet_log]
    stakes = [b["stake"] for b in bet_log]
    staked = sum(stakes)
    pnl = sum(pnls)
    roi = (pnl / staked) if staked else None
    lo, hi = BootstrapRoi(pnls, stakes, rng)
    end = daily[-1]["equity"] if daily else START
    dd = min((d["drawdown_pct"] for d in daily), default=0.0)
    return {
        "strategy": strategy, "n_bets": len(bet_log), "final_equity": end,
        "staked": round(staked, 2), "pnl": round(pnl, 2),
        "roi": roi, "roi_ci90": None if lo is None else [lo, hi],
        "max_drawdown_pct": dd, "ruined": bool(end < RUIN or (daily and daily[-1]["equity"] < RUIN and any(d["bets_day"] == 0 and d["x"] > 1 and daily[d["x"] - 2]["equity"] < RUIN for d in daily))),
    }


def SimpleRuin(daily):
    flagged = False
    for d in daily:
        if d["equity"] < RUIN:
            flagged = True
            break
    return flagged


def ChartCalibration(blocks, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(blocks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), squeeze=False)
    colors = {MODEL: "#2ca02c", MARKET: "#1f77b4", BOOK_MODEL: "#2ca02c", BOOK_MARKET: "#333333"}
    names = {MODEL: "p_full2 kmid T-60", MARKET: "Kalshi T-60 mid", BOOK_MODEL: "p_full2 (close input)",
             BOOK_MARKET: "book close, de-vigged"}
    for ax, (label, series) in zip(axes[0], blocks):
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        for j, (model, tidy, summary) in enumerate(series):
            bs = [b for b in tidy if b["n"]]
            off = (j - (len(series) - 1) / 2) * 0.008
            x = [b["mean_pred"] + off for b in bs]
            y = [b["obs_rate"] for b in bs]
            ax.plot(x, y, "o-", ms=5, color=colors.get(model, "black"),
                    label=f"{names.get(model, model)}  Brier {summary['brier']:.4f}, log loss {summary['log_loss']:.4f}, ECE {summary['ece']:.3f} (N={summary['n']})")
            for b in bs:
                ax.annotate(str(b["n"]), (b["mean_pred"] + off, b["obs_rate"]), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7, color=colors.get(model, "black"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("predicted home win probability (bin mean)")
        ax.set_ylabel("observed home win rate")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ChartBankroll(dailies, path, title, boundaries):
    import matplotlib
    matplotlib.use("Agg")
    rows = []
    for series, daily in dailies:
        rows.append({"date": daily[0]["date"] if daily else None, "series": series, "equity": START,
                     "drawdown_pct": 0.0, "season": daily[0]["season"] if daily else "", "x": 0, "x_kind": "index"})
        for d in daily:
            rows.append({**d, "series": series, "x_kind": "index"})
    frame = {"rows": rows, "boundaries": boundaries, "x_kind": "index",
             "first_date": dailies[0][1][0]["date"] if dailies and dailies[0][1] else None,
             "last_date": dailies[0][1][-1]["date"] if dailies and dailies[0][1] else None}
    PlotBankroll(frame, path=path, title=title, ylabel="Bankroll ($)",
                 hlines=[(START, "$10,000 start")], figsize=(11, 6.5),
                 xlabel="Game-day index (offseason omitted; dashed line = new season)")
    import matplotlib.pyplot as plt
    plt.close("all")


def UnseenList(rbs):
    out = []
    for r in rbs[2026]:
        if int(r["week"]) > 9:
            continue
        if SeenBySearch(r):
            continue
        out.append({
            "game_id": r["game_id"], "gameday": r["gameday"], "week": int(r["week"]),
            "gametime_et": r.get("gametime"), "matchup": f"{r['away_team']} @ {r['home_team']}",
            "home_win": r["home_win"], "forward_holdout": bool(IsForwardHoldout(r["gameday"])),
            "seen_by_search": False,
        })
    out.sort(key=lambda x: (x["gameday"], x["game_id"]))
    return out


def MdCalibration(meta):
    u = meta["unseen"]
    hold = [g for g in u if g["forward_holdout"]]
    pre = [g for g in u if not g["forward_holdout"]]
    unseen_hold = ", ".join(f"{g['matchup']} ({g['gameday']})" for g in hold) or "none"
    lines = [
        "# NFL calibration in the common cross-sport format (2026-11-13)",
        "",
        f"Made by American Footbot, 2026-11-13, America/Los_Angeles. Code `{meta['code']}`, run_id `{meta['run_id']}`, "
        f"model_version `{meta['model_version']}`. Evaluation only: nothing was fitted or tuned on these numbers. Simulation, no wagers.",
        "",
        "## Headline model",
        f"**`{MODEL}`** is the walk-forward `p_full2` logit (WIN_FEATS2, fitted on seasons 2013..S-1, L2=5) with the market input replaced by the Kalshi de-vigged mid at T-60 (staleness <= 30 min). "
        "That is the leak-safer of the two candidates. Raw `p_full2` uses the nflverse closing moneyline as `mkt_logit`, which is not known at T-60, so a T-60 evaluation of as-fitted `p_full2` would look ahead. "
        "`kmid` keeps the same frozen weights and only swaps the market feature for the T-60 mid, which is what we could have computed then. "
        "**Residual look-ahead:** `spread_gap` still uses the closing spread, so a little post-T-60 information can leak through that term. The model also uses the market line as an input: it is a market recalibration, not an independent forecast.",
        "",
        "## Splits",
        f"- **2025, `contaminated`.** This season was used during strategy search. The search-contamination tally is at least **2,819** trials (FrozenConfig's 2,073 on 2026-11-08, plus later Lab/Volume/Hunt recuts and the in-game rule study). Kalshi history starts in 2025.",
        f"- **2026 weeks 1-9, `forward`.** Labelled forward in this common file, but most of these games were already final in the 2026-11-01 16:30:11 ET data pull that the searches read (`SEARCH_PULLED_AT` in `Backtest/ForwardWalk.py`). A game is **seen by the search** if kickoff + 3h15m was at or before that pull (the same rule as ForwardWalk; conservative on the two borderline Nov 1 afternoon games).",
        (f"- **Truly unseen games: {len(u)}.** These are the 2026 week 1-9 games with kickoff + 3h15m after the 2026-11-01 16:30:11 ET pull. {len(pre)} were played on or before the freeze date {FROZEN['frozen_at']} (unseen by the search, but not a pre-registered holdout). {len(hold)} official forward holdout game(s) have gameday after {FROZEN['frozen_at']}: " + unseen_hold + "."),
        "- **`pooled_oos`** is 2026 weeks 1-9 only (split `forward`). 2025 is kept out of that pool because it is contaminated.",
        "- A `2026_unseen` block (same 20 games, split `forward`) is included so the clean slice can be read on its own. It is small.",
        "",
        "## Primary results, Kalshi T-60 mid, home-win probability, 10% bins",
        "",
        "| Season | Split | N | Model Brier | Market Brier | Model log loss | Market log loss | Model ECE | Market ECE | Model acc. | Market acc. |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in meta["cal_table"]:
        lines.append(
            f"| {row['season']} | {row['split']} | {row['n']} | {row['model_brier']:.4f} | {row['market_brier']:.4f} | "
            f"{row['model_ll']:.4f} | {row['market_ll']:.4f} | {row['model_ece']:.4f} | {row['market_ece']:.4f} | "
            f"{row['model_acc']:.1%} | {row['market_acc']:.1%} |"
        )
    lines += [
        "",
        "Lower Brier and log loss are better. 90% CIs (game-cluster bootstrap and exact binomial / Clopper-Pearson; the widest is the one to read) are in the JSON. With one probability per game the two intervals almost coincide; we still report the wider one.",
        "",
        f"## Games dropped",
        f"- 2025: {meta['dropped_2025']} finished games had no two-sided Kalshi T-60 quote (no older than 30 min). They are listed in the JSON. The 1 tied game in 2025 was dropped, as the plan requires.",
        f"- 2026 weeks 1-9: {meta['dropped_2026']} finished games missing T-60; {meta['ties_2026']} tie(s) dropped. Every scored non-tie in this window had a T-60 quote.",
        "",
        "## Optional secondary: `book_close_devig` and `p_full2` (close input), 2014-2026",
        "Cheap from existing walk-forward rows. Fill = the de-vigged closing moneyline (optimistic: it is a game-time price, not a T-60 quote, and there is no bid/ask). 2014 is labelled `validation` (it is outside the recorded 2015-2022 tune window, but may have been seen in earlier work). 2015-2022 are `tune`. 2023-2025 are `contaminated`. 2026 weeks 1-9 are `forward`. `pooled_oos` is again 2026 only. Headline numbers for the book-close 2026 window are in the summary CSV.",
        "",
        "## Caveats for Graphtard to caption",
        "- The model uses the market line as an input.",
        "- 2025 is contaminated (search tally at least 2,819). Only 20 of the 2026 week 1-9 games are truly unseen, and only BUF @ MIN (2026-11-09) is after the 2026-11-08 freeze.",
        "- Kalshi pregame history starts in 2025. Six 2025 games (late regular season / playoffs / Super Bowl) have no T-60 quote.",
        "- Small forward and unseen samples: CIs are wide. Read the widest of the game-cluster bootstrap and exact binomial intervals.",
        "- Candles have no size, so any fill is a top-of-book assumption, optimistic for large stakes.",
        "- Fill is the mid, per Tim, not the plan's ask. `spread_gap` still uses the closing spread.",
        "- Home-side probability, ten fixed 10% bins; empty bins have n=0 and blank rates.",
        "",
        "## Files",
        "- `2026-11-13_calibration.csv` and `2026-11-13_calibration_summary.csv`: common schema (plan section 5a).",
        "- `2026-11-13_calibration.json`: CIs, unseen list, drop list, model choice.",
        "- `2026-11-13_calibration_draft.png`: Kalshi T-60 reliability, 2025 / 2026 / unseen.",
        "- `2026-11-13_calibration_book-close_draft.png`: optional book-close reliability, tune-pooled / contaminated-pooled / 2026.",
    ]
    lines += ["", "### Truly unseen 2026 games", "",
              "| Date (ET game day) | Week | Game | Official holdout? |",
              "|---|---|---|---|"]
    for g in u:
        lines.append(f"| {g['gameday']} | {g['week']} | {g['matchup']} | {'yes' if g['forward_holdout'] else 'no'} |")
    return "\n".join(lines) + "\n"


def MdBankroll(meta):
    lines = [
        "# NFL $10k bankroll in the common cross-sport format (2026-11-13)",
        "",
        f"Made by American Footbot, 2026-11-13, America/Los_Angeles. Code `{meta['code']}`, run_id `{meta['run_id']}`, "
        f"model_version `{meta['model_version']}`. Simulation only; no orders, no real wagers.",
        "",
        "## Setup",
        f"- **Model:** `{MODEL}` (see the calibration report for why this is the leak-safer headline).",
        "- **Price basis:** `kalshi_T60`. `market_prob_devig` is the side's mid, normalised so both sides sum to 1. **Fill = mid**, per Tim (\"just assume that the middle of the market is the filled price\"). This is a deliberate departure from plan section 3, which used the ask. Candles have no size, so fills are optimistic for big stakes.",
        f"- **Fee, primary:** `fee_basis={FEE_BASIS}`, $0.02 per $1 contract. This is **Tim's fee estimate**, not a verified exchange fee. Breakeven is fill + $0.02. Value only at `edge = model_prob - market_prob_devig - 0.02 >= 0.03`; anything less is PASS.",
        "- **Verified Kalshi taker fee** 0.07*p*(1-p), at most $0.0175 (KXNFLGAME multiplier 1), is shown as a second path and as extra edge-file columns. It is never used instead of Tim's estimate for the headline. Qualifying still uses the $0.02 edge rule, so the bet set is the same.",
        "- **Bankroll:** one continuous $10,000 bank across 2025 (contaminated) and 2026 weeks 1-9 (forward). No seasonal reset. `flat100` bets a flat $100 (1% of the starting bank) on every qualifying game, no compounding; it stops if equity falls below $100 (ruin). `qkelly` stakes 0.25 x Kelly at cost c = fill + fee, capped at 2% of start-of-day equity, and compounds. `market_fav` is a flat $100 on the Kalshi favourite every priced game (the cost of fees alone). Same-day qkelly stakes share the start-of-day bank and are scaled down if they would exceed it.",
        "- **x** is a game-day index starting at 1 (the $10,000 start is implicit at x=0). Offseason days are omitted. A marker sits at the first 2026 game day. Every game day in the window is a row, bets or not.",
        "- The frozen 8-point / divisional rule, RL staking, and in-game rules are out of scope here.",
        "",
        "## Headline, `tim_est_0.02`, mid fills",
        "",
        "| Strategy | N bets | End bank | P&L | ROI per $ staked | 90% event-bootstrap CI | Max drawdown | Ruin? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in meta["bank_table"]:
        ci = row["roi_ci90"]
        ci_s = "--" if not ci else f"[{ci[0]:+.1%}, {ci[1]:+.1%}]"
        roi_s = "--" if row["roi"] is None else f"{row['roi']:+.1%}"
        lines.append(
            f"| {row['label']} | {row['n_bets']} | ${row['final_equity']:,.2f} | ${row['pnl']:,.2f} | {roi_s} | {ci_s} | {row['max_drawdown_pct']:.1%} | {'yes' if row['ruined'] else 'no'} |"
        )
    lines += [
        "",
        "## Same bets, verified Kalshi taker fee",
        "",
        "| Strategy | N bets | End bank | ROI per $ staked | 90% CI | Max drawdown |",
        "|---|---|---|---|---|---|",
    ]
    for row in meta["bank_table_verified"]:
        ci = row["roi_ci90"]
        ci_s = "--" if not ci else f"[{ci[0]:+.1%}, {ci[1]:+.1%}]"
        roi_s = "--" if row["roi"] is None else f"{row['roi']:+.1%}"
        lines.append(
            f"| {row['label']} | {row['n_bets']} | ${row['final_equity']:,.2f} | {roi_s} | {ci_s} | {row['max_drawdown_pct']:.1%} |"
        )
    lines += [
        "",
        f"## Games dropped and ties",
        f"- Kalshi T-60: {meta['dropped_n']} finished games dropped for a missing two-sided quote; {meta['ties']} tie(s) dropped from edges and P&L.",
        f"- Optional `book_close_devig` bankroll is a separate series on 2023-2026 weeks 1-9 (contaminated + forward, one $10k bank, no tune seasons, so in-sample compounding cannot inflate it). Fill = de-vigged close (optimistic, no timestamp). {meta['book_dropped']} scored games in 2014-2026 lacked a close; ties are in the JSON.",
        "",
        "## Caveats for Graphtard to caption",
        "- Simulated. One $10,000 bank, no reset, mid fills, Tim's $0.02 fee estimate.",
        "- 2025 is contaminated (search tally at least 2,819). 2026 weeks 1-9 are labelled forward, but only 20 games are truly unseen (kickoff + 3h15m after the 2026-11-01 16:30:11 ET search pull); only BUF @ MIN (2026-11-09) is after the 2026-11-08 freeze. Small-N CIs are wide; we report the widest of the event/game bootstrap and, for rates, exact binomial.",
        "- The model uses the market line as an input. Kalshi history starts in 2025. Candles have no size.",
        "- `market_fav` is not a claim of edge; it shows fee drag on the favourite.",
        "- Do not read a short winning or losing path as an edge. Claim an edge only if the lower CI bound is above 0 under both fee bases, on a serious sample. That bar is not met here unless the numbers below say otherwise.",
        "",
        "## Files",
        "- `2026-11-13_bankroll-10k.csv`: daily equity, common schema (plan section 5b).",
        "- `2026-11-13_bankroll-10k_edges.csv`: one row per priced event, qualifying or not (plan section 5c), plus verified-fee and unseen flags.",
        "- `2026-11-13_bankroll-10k.json`: summaries, CIs, unseen list, drops.",
        "- `2026-11-13_bankroll-10k_draft.png`: Kalshi T-60 flat100 / qkelly / market_fav, game-day index, 2026 marker.",
        "- `2026-11-13_bankroll-10k_book-close_draft.png`: optional book-close 2023-2026 path.",
    ]
    return "\n".join(lines) + "\n"


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-dir", default=os.path.join("E:\\AgentDocs", "American-Footbot", "2026-11-13_calibration"))
    ap.add_argument("--bank-dir", default=os.path.join("E:\\AgentDocs", "American-Footbot", "2026-11-13_bankroll-10k"))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    run_id = NewRunId()
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    check = AttachKmid(rbs)
    kalshi_rows, dropped, ties_k = KalshiRows(rbs)
    unseen = UnseenList(rbs)
    days = AllGameDays(rbs)
    edges = KalshiEdges(kalshi_rows)
    ties_2026 = sum(1 for r in rbs[2026] if int(r["week"]) <= 9 and r.get("result") == 0)
    dropped_2025 = [d for d in dropped if d["gameday"] < "2026-08-01"]
    dropped_2026 = [d for d in dropped if d["gameday"] >= "2026-08-01"]

    def subset(rows, pred):
        return [r for r in rows if pred(r)]

    # probability rows for calibration (home side)
    def as_cal(rs, pkey, mktkey):
        out = []
        for r in rs:
            rec = dict(r)
            rec["_model"] = rec[pkey]
            rec["_mkt"] = rec[mktkey]
            out.append(rec)
        return out

    cal_sets = [
        ("2025", "contaminated", subset(kalshi_rows, lambda r: int(r["season"]) == 2025)),
        ("2026", "forward", subset(kalshi_rows, lambda r: int(r["season"]) == 2026)),
        ("2026_unseen", "forward", subset(kalshi_rows, lambda r: int(r["season"]) == 2026 and not SeenBySearch(r))),
        ("pooled_oos", "forward", subset(kalshi_rows, lambda r: int(r["season"]) == 2026)),
    ]
    cal_tidy, cal_sum, cal_extra = [], [], {}
    for season, split, rs in cal_sets:
        if not rs:
            continue
        last = max(r["gameday"] for r in rs)
        for key, model in (("_model", MODEL), ("_mkt", MARKET)):
            mapped = [{"home_win": r["home_win"], key: (r["p_kmid"] if key == "_model" else r["kmid"])} for r in rs]
            # ScoreKey expects the key on the row
            tidy, summary, extra = CalBlock(mapped, key, model, season, split, last, rng)
            cal_tidy.extend(tidy)
            cal_sum.append(summary)
            cal_extra[f"{model}|{season}"] = extra

    # book close calibration 2014-2026
    book_rows, book_ties, book_dropped = BookRows(rbs)
    book_sets = []
    by_season = defaultdict(list)
    for r in book_rows:
        by_season[int(r["season"])].append(r)
    for s, rs in sorted(by_season.items()):
        split = SplitLabel(rs[0])
        book_sets.append((str(s), split, rs))
    book_sets.append(("tune_2015_2022", "tune", [r for r in book_rows if 2015 <= int(r["season"]) <= 2022]))
    book_sets.append(("contaminated_2023_2025", "contaminated", [r for r in book_rows if 2023 <= int(r["season"]) <= 2025]))
    book_sets.append(("pooled_oos", "forward", [r for r in book_rows if int(r["season"]) == 2026]))
    for season, split, rs in book_sets:
        if not rs:
            continue
        last = max(r["gameday"] for r in rs)
        for key, model in (("p_full2", BOOK_MODEL), ("mkt_home", BOOK_MARKET)):
            mapped = [{"home_win": r["home_win"], key: r[key]} for r in rs]
            tidy, summary, extra = CalBlock(mapped, key, model, season, split, last, rng)
            cal_tidy.extend(tidy)
            cal_sum.append(summary)
            cal_extra[f"{model}|{season}"] = extra

    # bankroll kalshi
    bank_rows, bank_stats, bank_stats_k, bet_logs = [], {}, {}, {}
    for strat in ("flat100", "qkelly", "market_fav"):
        daily, log, ruined, end = Walk(edges, strat, fee_kind="tim", days=days)
        bank_rows.extend(daily)
        st = {
            "strategy": strat, "label": f"{strat} / model" if strat != "market_fav" else "market_fav",
            "n_bets": len(log), "final_equity": daily[-1]["equity"] if daily else START,
            "staked": round(sum(b["stake"] for b in log), 2),
            "pnl": round(sum(b["pnl"] for b in log), 2),
            "roi": (sum(b["pnl"] for b in log) / sum(b["stake"] for b in log)) if log and sum(b["stake"] for b in log) else None,
            "roi_ci90": (lambda t: None if t[0] is None else [t[0], t[1]])(BootstrapRoi([b["pnl"] for b in log], [b["stake"] for b in log], rng)),
            "max_drawdown_pct": min((d["drawdown_pct"] for d in daily), default=0.0),
            "ruined": SimpleRuin(daily) or ruined,
        }
        bank_stats[strat] = st
        bet_logs[strat] = log
        daily_k, log_k, ruined_k, _end_k = Walk(edges, strat, fee_kind="kalshi", days=days)
        bank_stats_k[strat] = {
            "strategy": strat, "label": f"{strat} / model" if strat != "market_fav" else "market_fav",
            "n_bets": len(log_k), "final_equity": daily_k[-1]["equity"] if daily_k else START,
            "staked": round(sum(b["stake"] for b in log_k), 2),
            "pnl": round(sum(b["pnl"] for b in log_k), 2),
            "roi": (sum(b["pnl"] for b in log_k) / sum(b["stake"] for b in log_k)) if log_k and sum(b["stake"] for b in log_k) else None,
            "roi_ci90": (lambda t: None if t[0] is None else [t[0], t[1]])(BootstrapRoi([b["pnl"] for b in log_k], [b["stake"] for b in log_k], rng)),
            "max_drawdown_pct": min((d["drawdown_pct"] for d in daily_k), default=0.0),
            "ruined": SimpleRuin(daily_k) or ruined_k,
        }

    # book close bankroll 2023-2026 only
    book_bank_rows = [r for r in book_rows if int(r["season"]) >= 2023]
    book_edges = BookEdges(book_bank_rows)
    book_days = BookGameDays(book_bank_rows)
    book_stats = {}
    for strat in ("flat100", "qkelly", "market_fav"):
        daily, log, ruined, _ = Walk(book_edges, strat, fee_kind="tim", days=book_days)
        bank_rows.extend(daily)
        book_stats[strat] = {
            "strategy": strat, "label": f"{strat} / book_close",
            "n_bets": len(log), "final_equity": daily[-1]["equity"] if daily else START,
            "staked": round(sum(b["stake"] for b in log), 2),
            "pnl": round(sum(b["pnl"] for b in log), 2),
            "roi": (sum(b["pnl"] for b in log) / sum(b["stake"] for b in log)) if log and sum(b["stake"] for b in log) else None,
            "roi_ci90": (lambda t: None if t[0] is None else [t[0], t[1]])(BootstrapRoi([b["pnl"] for b in log], [b["stake"] for b in log], rng)),
            "max_drawdown_pct": min((d["drawdown_pct"] for d in daily), default=0.0),
            "ruined": SimpleRuin(daily) or ruined,
        }

    # write calibration
    cal_base = os.path.join(a.cal_dir, "2026-11-13_calibration")
    WriteCsv(cal_base + ".csv", CAL_FIELDS, cal_tidy)
    WriteCsv(cal_base + "_summary.csv", SUM_FIELDS, cal_sum)

    def pick(season, model):
        for s in cal_sum:
            if s["season"] == season and s["model"] == model:
                return s
        return None

    cal_table = []
    for season, split in (("2025", "contaminated"), ("2026", "forward"), ("2026_unseen", "forward"), ("pooled_oos", "forward")):
        m, k = pick(season, MODEL), pick(season, MARKET)
        if not (m and k):
            continue
        cal_table.append({"season": season, "split": split, "n": m["n"],
                          "model_brier": m["brier"], "market_brier": k["brier"],
                          "model_ll": m["log_loss"], "market_ll": k["log_loss"],
                          "model_ece": m["ece"], "market_ece": k["ece"],
                          "model_acc": m["accuracy"], "market_acc": k["accuracy"]})
    cal_meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "headline_model": MODEL, "model_choice": "p_full2_kmid_T60 over as-fitted p_full2 because the latter's market input is the closing line, unknown at T-60.",
        "fee_basis": FEE_BASIS, "price_basis": PRICE_KALSHI, "fill": "mid",
        "frozen_at": FROZEN["frozen_at"], "search_pulled_at": "2026-11-01T16:30:11-05:00",
        "contamination_tally_at_least": 2819, "refit_check": check,
        "unseen": unseen, "unseen_n": len(unseen),
        "unseen_forward_holdout_n": sum(g["forward_holdout"] for g in unseen),
        "dropped_missing_prices": dropped, "dropped_2025": len(dropped_2025), "dropped_2026": len(dropped_2026),
        "ties_kalshi": ties_k, "ties_2026": ties_2026, "book_ties": book_ties, "book_dropped": book_dropped,
        "cal_table": cal_table, "summary": cal_sum, "ci": cal_extra,
        "n_kalshi_2025": sum(int(r["season"]) == 2025 for r in kalshi_rows),
        "n_kalshi_2026": sum(int(r["season"]) == 2026 for r in kalshi_rows),
    }
    with open(cal_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(cal_meta, fh, indent=1)
    with open(cal_base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdCalibration(cal_meta))

    def tidy_for(model, season):
        return [r for r in cal_tidy if r["model"] == model and r["season"] == season]

    def summ(model, season):
        return pick(season, model)

    cal_blocks = []
    for season, label in (("2025", "2025, contaminated (Kalshi T-60)"),
                          ("2026", "2026 weeks 1-9, forward (mostly seen by search)"),
                          ("2026_unseen", "2026 truly unseen (N=20)")):
        series = []
        for model in (MODEL, MARKET):
            s = summ(model, season)
            if s:
                series.append((model, tidy_for(model, season), s))
        if series:
            cal_blocks.append((label, series))
    ChartCalibration(cal_blocks, cal_base + "_draft.png",
                     "NFL pregame calibration: p_full2 kmid vs Kalshi T-60 mid (home win, 10% bins)")
    book_blocks = []
    for season, label in (("tune_2015_2022", "2015-2022, tune, book close"),
                          ("contaminated_2023_2025", "2023-2025, contaminated, book close"),
                          ("2026", "2026 weeks 1-9, forward, book close")):
        series = []
        for model in (BOOK_MODEL, BOOK_MARKET):
            s = summ(model, season)
            if s:
                series.append((model, tidy_for(model, season), s))
        if series:
            book_blocks.append((label, series))
    ChartCalibration(book_blocks, cal_base + "_book-close_draft.png",
                     "NFL pregame calibration: p_full2 vs de-vigged closing moneyline (optional secondary)")

    # write bankroll
    bank_base = os.path.join(a.bank_dir, "2026-11-13_bankroll-10k")
    WriteCsv(bank_base + ".csv", BANK_FIELDS, bank_rows)
    edge_fields = EDGE_FIELDS + EDGE_EXTRA
    WriteCsv(bank_base + "_edges.csv", edge_fields, edges + book_edges)
    first_2026 = next((d["x"] for d in bank_rows if d["price_basis"] == PRICE_KALSHI and d["season"] == "2026"), None)
    bank_meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "headline_model": MODEL, "price_basis": PRICE_KALSHI, "fill": "mid", "fee_basis": FEE_BASIS,
        "start": START, "min_edge": MIN_EDGE, "contamination_tally_at_least": 2819,
        "unseen": unseen, "dropped_missing_prices": dropped, "dropped_n": len(dropped), "ties": ties_k,
        "book_ties": book_ties, "book_dropped": book_dropped,
        "bank_tim_est_0.02": bank_stats, "bank_kalshi_verified": bank_stats_k,
        "bank_book_close_2023_2026": book_stats,
        "n_priced_kalshi": len(edges), "n_game_days_kalshi": len(days),
        "first_2026_x": first_2026, "refit_check": check,
    }
    # flatten tables for md
    bank_meta["bank_table"] = [
        {**bank_stats["flat100"], "label": "flat100 / model"},
        {**bank_stats["qkelly"], "label": "qkelly / model"},
        {**bank_stats["market_fav"], "label": "market_fav"},
    ]
    bank_meta["bank_table_verified"] = [
        {**bank_stats_k["flat100"], "label": "flat100 / model"},
        {**bank_stats_k["qkelly"], "label": "qkelly / model"},
        {**bank_stats_k["market_fav"], "label": "market_fav"},
    ]
    with open(bank_base + ".json", "w", encoding="utf-8") as fh:
        json.dump(bank_meta, fh, indent=1, default=str)
    with open(bank_base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdBankroll(bank_meta))

    def series_daily(strategy, basis=PRICE_KALSHI):
        return [d for d in bank_rows if d["strategy"] == strategy and d["price_basis"] == basis]

    bounds = []
    if first_2026 is not None:
        d2026 = next(d for d in bank_rows if d["price_basis"] == PRICE_KALSHI and d["season"] == "2026")
        bounds.append({"series": None, "season": "2026", "x": first_2026, "date": d2026["date"]})
    ChartBankroll(
        [("flat100", series_daily("flat100")),
         ("qkelly", series_daily("qkelly")),
         ("market_fav", series_daily("market_fav"))],
        bank_base + "_draft.png",
        "NFL $10k bankroll at Kalshi T-60 mid, tim_est_0.02 (2025 + 2026 weeks 1-9)",
        bounds,
    )
    bfirst = next((d["x"] for d in bank_rows if d["price_basis"] == PRICE_BOOK and d["season"] == "2024"), None)
    # season markers for 2024, 2025, 2026 on the book-close path (2023 is the start)
    bbounds = []
    seen = set()
    for d in bank_rows:
        if d["price_basis"] != PRICE_BOOK:
            continue
        if d["season"] in seen:
            continue
        seen.add(d["season"])
        if d["season"] != "2023":
            bbounds.append({"series": None, "season": d["season"], "x": d["x"], "date": d["date"]})
    ChartBankroll(
        [("flat100 / book close", series_daily("flat100", PRICE_BOOK)),
         ("qkelly / book close", series_daily("qkelly", PRICE_BOOK)),
         ("market_fav / book close", series_daily("market_fav", PRICE_BOOK))],
        bank_base + "_book-close_draft.png",
        "NFL $10k bankroll at de-vigged close, tim_est_0.02 (2023-2026 weeks 1-9, optional)",
        bbounds,
    )
    print(json.dumps({
        "run_id": run_id, "code": CodeVersion(), "model": MODEL,
        "n_kalshi": len(edges), "dropped": len(dropped), "unseen": len(unseen),
        "cal": cal_table, "bank_tim": {k: {m: bank_stats[k][m] for m in ("n_bets", "final_equity", "roi", "max_drawdown_pct", "ruined")} for k in bank_stats},
        "bank_verified": {k: {m: bank_stats_k[k][m] for m in ("n_bets", "final_equity", "roi", "max_drawdown_pct")} for k in bank_stats_k},
        "bank_book": {k: {m: book_stats[k][m] for m in ("n_bets", "final_equity", "roi", "max_drawdown_pct")} for k in book_stats},
        "cal_dir": a.cal_dir, "bank_dir": a.bank_dir, "refit_check": check,
    }, indent=1, default=str))


if __name__ == "__main__":
    Main()
