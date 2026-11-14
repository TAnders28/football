r"""
Model-free rule search on Kalshi KXNFLSPREAD and KXNFLTOTAL (Job B addition).

Pre-registered before looking at 2026 results:
  Grid is the cartesian product of:
    price bin (0.05 wide, 20 bins on the fill price) x
    side (yes, no) x
    entry time (T-180, T-120, T-60) x
    key-number proximity (spreads: any / prox_3 / prox_7 / prox_3_or_7;
      totals: any / prox_41 / prox_44 / prox_47 / prox_41_44_47;
      "prox" means |strike| within 0.5 of the key, using |strike| for spreads) x
    fill type (mid = Tim's primary rule, ask = taker, bid = maker).
  That is 20*2*3*4*3 = 1,440 spread configs plus 20*2*3*5*3 = 1,800 total
  configs = 3,240 rules. Added to the prior lower-bound tally of 5,379.
  Maker-at-bid bets count only when a later 1-min candle, up to kickoff,
  trades at or through the bid (adverse-selection check: compare those
  fills with the quotes that would not have been filled).
  Fit/select on 2025 only (n >= 20 and mean edge >= 3c). Freeze. Hold out 2026.
  Edge = outcome - (price + tim_est_0.02). Also recompute with Kalshi
  taker fee 0.07*p*(1-p). White RC + Hansen SPA, 10,000 day-block boots.
  Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) with N = grid size.
  Holdout 95% CIs: day-block, game-cluster, binomial; report the widest.
  10,000-path $10k PathSim, flat100 and qkelly (2% cap), on the frozen best.

    .venv\Scripts\python.exe -m Backtest.ThinMarketRules --out-dir DIR [--workers 7]

Simulation only; no orders. Normal priority is allowed for this analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.CommonFormat import (
    BANK_FIELDS, FEE, FLAT, MIN_EDGE, QKELLY_CAP, RUIN, START, Walk, Widest, WriteCsv,
)
from Backtest.KalshiReality import KalshiFee, QuoteAt, STALE_MIN
from Backtest.PathSim import DailyReturns, Evaluate as PathEvaluate
from Backtest.ThinMarkets import (
    ALPHA, BIN, BankBundle, BinHi, BinLo, IterJsonl, LoadFees, LoadMarkets,
    LoadThinEntries, MIN_BIN_N, SEED, WidestRoiCi, YesResult, _JsonDefault,
)
from Backtest.Version import CodeVersion, ModelVersion, NewRunId
from DataPull.PullKalshiThinMarkets import ParseStrike

ET = ZoneInfo("America/New_York")
TIMES = (180, 120, 60)
SIDES = ("yes", "no")
FILLS = ("mid", "ask", "bid")
SPREAD_KEYS = ("any", "prox_3", "prox_7", "prox_3_or_7")
TOTAL_KEYS = ("any", "prox_41", "prox_44", "prox_47", "prox_41_44_47")
KEY_SETS = {
    "any": None,
    "prox_3": (3.0,),
    "prox_7": (7.0,),
    "prox_3_or_7": (3.0, 7.0),
    "prox_41": (41.0,),
    "prox_44": (44.0,),
    "prox_47": (47.0,),
    "prox_41_44_47": (41.0, 44.0, 47.0),
}
PROX_TOL = 0.5
N_BINS = int(round(1.0 / BIN))
PRICE_BINS = tuple(round(i * BIN, 2) for i in range(N_BINS))
N_SPA = 10_000
N_PATHS = 10_000
PRIOR_TALLY = 5379
MIN_N = MIN_BIN_N  # 20
SPORT = "nfl"
BLOCK = 4
EULER = 0.5772156649015329

_RECS = None
_DAYS = None


def ConfigGrid():
    """Pre-registered cartesian grid. Spreads and totals have their own key lists."""
    out = []
    for kind, keys in (("spread", SPREAD_KEYS), ("total", TOTAL_KEYS)):
        for side in SIDES:
            for tmin in TIMES:
                for key in keys:
                    for fill in FILLS:
                        for lo in PRICE_BINS:
                            out.append({
                                "kind": kind, "side": side, "tmin": int(tmin),
                                "key": key, "fill": fill, "bin_lo": lo,
                            })
    return out


def ConfigId(cfg):
    return (f"{cfg['kind']}|{cfg['side']}|T{cfg['tmin']}|{cfg['key']}|"
            f"{cfg['fill']}|{cfg['bin_lo']:.2f}-{BinHi(cfg['bin_lo']):.2f}")


def GridSize():
    return len(ConfigGrid())


def NearKey(strike, key, kind):
    keys = KEY_SETS[key]
    if keys is None:
        return True
    if strike is None:
        return False
    try:
        x = float(strike)
    except (TypeError, ValueError):
        return False
    val = abs(x) if kind == "spread" else x
    return any(abs(val - k) <= PROX_TOL + 1e-12 for k in keys)


def StrikeOf(entry, markets=None):
    markets = markets or {}
    m = markets.get(entry.get("ticker")) or {}
    if m.get("strike") is not None:
        try:
            return float(m["strike"])
        except (TypeError, ValueError):
            pass
    strike, _ = ParseStrike(entry.get("ticker") or "", m)
    return strike


def SideQuotes(q, side):
    """Map a yes-side quote bundle to the traded side. q has bid/ask/mid."""
    if not q:
        return None
    if side == "yes":
        bid, ask = q["bid"], q["ask"]
    else:
        bid, ask = 1.0 - q["ask"], 1.0 - q["bid"]
    if bid is None or ask is None or bid < 0 or ask > 1 or ask < bid:
        return None
    mid = (bid + ask) / 2
    return {
        "bid": bid, "ask": ask, "mid": mid, "width": ask - bid,
        "age_min": q.get("age_min"), "volume": q.get("volume"),
        "open_interest": q.get("open_interest"),
    }


def FillPrice(sq, fill):
    if not sq:
        return None
    return sq.get(fill)


def QuoteBundleAt(entry, minutes):
    q = QuoteAt(entry, minutes)
    if not q:
        return None
    mid = (q["bid"] + q["ask"]) / 2
    target = (entry["ko"].timestamp()) - minutes * 60
    candle = None
    for c in entry.get("candles") or []:
        if c["end_ts"] > target:
            break
        if c.get("yes_ask_close") is not None and c.get("yes_bid_close") is not None and 0 < c["yes_ask_close"] < 1:
            candle = c
    return {
        "bid": q["bid"], "ask": q["ask"], "mid": mid, "width": q["ask"] - q["bid"],
        "age_min": q["age_min"],
        "volume": None if not candle else candle.get("volume"),
        "open_interest": None if not candle else candle.get("open_interest"),
    }


def WouldFill(entry, side, bid, minutes):
    """True if a later 1-min candle up to kickoff trades at or through the posted bid."""
    if bid is None or not entry.get("ko"):
        return False
    start = (entry["ko"].timestamp()) - minutes * 60
    ko = entry["ko"].timestamp()
    for c in entry.get("candles") or []:
        ts = c.get("end_ts")
        if ts is None or ts <= start or ts > ko:
            continue
        yb, ya = c.get("yes_bid_close"), c.get("yes_ask_close")
        if yb is None or ya is None:
            continue
        if side == "yes":
            if ya <= bid + 1e-12 or yb < bid - 1e-15:
                return True
        else:
            no_ask = 1.0 - yb
            if no_ask <= bid + 1e-12:
                return True
    return False


def CoverageGaps(entries):
    seasons = {int(e["season"]) for e in entries if e.get("season") is not None}
    kinds = {e.get("market_kind") for e in entries}
    missing = []
    if 2025 not in seasons:
        missing.append("2025 candles")
    if 2026 not in seasons:
        missing.append("2026 candles")
    if "spread" not in kinds:
        missing.append("KXNFLSPREAD")
    if "total" not in kinds:
        missing.append("KXNFLTOTAL")
    return missing


def BuildRecs(entries, markets=None):
    """One compact record per market with quotes at each entry time and maker fill flags."""
    markets = markets or {}
    recs = []
    for e in entries:
        if e.get("error") and not e.get("candles"):
            continue
        if not e.get("game_id") or not e.get("ko"):
            continue
        kind = e.get("market_kind") or (markets.get(e.get("ticker")) or {}).get("market_kind")
        if kind not in ("spread", "total"):
            continue
        y = YesResult(e.get("result"))
        if y is None:
            continue
        gday = (markets.get(e.get("ticker")) or {}).get("gameday")
        if not gday:
            try:
                gday = e["ko"].astimezone(ET).date().isoformat()
            except Exception:
                gday = (e.get("kickoff_utc") or "")[:10]
        quotes = {t: QuoteBundleAt(e, t) for t in TIMES}
        if not any(quotes.values()):
            continue
        rec = {
            "ticker": e.get("ticker"),
            "season": int(e["season"]),
            "date": gday,
            "game_id": e.get("game_id"),
            "week": e.get("week"),
            "kind": kind,
            "strike": StrikeOf(e, markets),
            "result_yes": int(y),
            "event": f"{e.get('away_team')} @ {e.get('home_team')}",
            "start_time": e.get("kickoff_utc"),
            "quotes": quotes,
            "would_fill": {},
        }
        for t in TIMES:
            q = quotes[t]
            if not q:
                rec["would_fill"][(t, "yes")] = False
                rec["would_fill"][(t, "no")] = False
                continue
            rec["would_fill"][(t, "yes")] = WouldFill(e, "yes", q["bid"], t)
            rec["would_fill"][(t, "no")] = WouldFill(e, "no", 1.0 - q["ask"], t)
        recs.append(rec)
    return recs


def BetsFor(recs, cfg, fee_kind="tim"):
    """Return bet dicts for one config. Maker-at-bid keeps only would-fill quotes."""
    out = []
    lo = cfg["bin_lo"]
    hi = BinHi(lo)
    for r in recs:
        if r["kind"] != cfg["kind"]:
            continue
        if not NearKey(r["strike"], cfg["key"], cfg["kind"]):
            continue
        q = r["quotes"].get(cfg["tmin"])
        sq = SideQuotes(q, cfg["side"])
        if not sq:
            continue
        fill = FillPrice(sq, cfg["fill"])
        if fill is None or fill <= 0 or fill >= 1:
            continue
        if cfg["fill"] == "bid" and not r["would_fill"].get((cfg["tmin"], cfg["side"])):
            continue
        if not (lo <= fill < hi or (hi == 1.0 and fill == 1.0)):
            continue
        result = r["result_yes"] if cfg["side"] == "yes" else 1 - r["result_yes"]
        fee = KalshiFee(fill) if fee_kind == "kalshi" else FEE
        edge = result - fill - fee
        out.append({
            "date": r["date"], "start_time": r.get("start_time"), "sport": SPORT,
            "season": str(r["season"]),
            "split": "contaminated" if r["season"] == 2025 else "forward",
            "event_id": r["game_id"], "event": r.get("event"),
            "side": cfg["side"], "ticker": r["ticker"],
            "model": "thin_rule_2025", "price_basis": f"kalshi_T{cfg['tmin']}_{cfg['fill']}",
            "model_prob": None, "market_prob_devig": fill, "fill": fill,
            "edge": edge, "result": int(result), "fee_kind": fee_kind,
            "width": sq["width"], "volume": sq.get("volume"),
            "open_interest": sq.get("open_interest"),
            "would_fill": True if cfg["fill"] != "bid" else True,
            "kind": r["kind"], "strike": r["strike"],
        })
    return out


def ScoreBets(bets, fee_kind="tim"):
    if not bets:
        return {"n": 0, "k": 0, "mean_edge": None, "mean_fill": None, "mean_width": None,
                "obs_rate": None, "selected": False, "fee_kind": fee_kind}
    n = len(bets)
    k = int(sum(b["result"] for b in bets))
    fills = [b["fill"] for b in bets]
    edges = [b["result"] - b["fill"] - (KalshiFee(b["fill"]) if fee_kind == "kalshi" else FEE) for b in bets]
    widths = [b["width"] for b in bets if b.get("width") is not None]
    mean_edge = float(np.mean(edges))
    return {
        "n": n, "k": k, "mean_edge": mean_edge,
        "mean_fill": float(np.mean(fills)),
        "mean_width": None if not widths else float(np.mean(widths)),
        "obs_rate": k / n,
        "selected": bool(n >= MIN_N and mean_edge >= MIN_EDGE),
        "fee_kind": fee_kind,
    }


def ScoreCfg(recs, cfg):
    bets = BetsFor(recs, cfg, fee_kind="tim")
    fit = [b for b in bets if b["season"] == "2025"]
    test = [b for b in bets if b["season"] == "2026"]
    s25 = ScoreBets(fit, "tim")
    s26 = ScoreBets(test, "tim")
    s25k = ScoreBets(fit, "kalshi")
    s26k = ScoreBets(test, "kalshi")
    return {
        "id": ConfigId(cfg), **cfg,
        "n_2025": s25["n"], "k_2025": s25["k"], "edge_2025": s25["mean_edge"],
        "fill_2025": s25["mean_fill"], "obs_2025": s25["obs_rate"],
        "selected": s25["selected"],
        "edge_2025_kalshi": s25k["mean_edge"],
        "n_2026": s26["n"], "k_2026": s26["k"], "edge_2026": s26["mean_edge"],
        "fill_2026": s26["mean_fill"], "obs_2026": s26["obs_rate"],
        "edge_2026_kalshi": s26k["mean_edge"],
        "mean_width_2025": s25["mean_width"], "mean_width_2026": s26["mean_width"],
    }


def InitWorker(recs):
    global _RECS
    _RECS = recs


def _EvalOne(cfg):
    return ScoreCfg(_RECS, cfg)


def RunGrid(recs, configs, workers=1):
    if workers <= 1 or len(configs) < 8:
        return [ScoreCfg(recs, c) for c in configs]
    out = []
    with ProcessPoolExecutor(max_workers=workers, initializer=InitWorker, initargs=(recs,)) as ex:
        futs = {ex.submit(_EvalOne, c): i for i, c in enumerate(configs)}
        buf = [None] * len(configs)
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            buf[i] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(configs):
                print(f"  rules grid {done}/{len(configs)}", flush=True)
        out = buf
    return out


def PickBest(rows, min_n=MIN_N):
    """Highest 2025 mean edge among configs with n >= min_n. 2026 is ignored."""
    eligible = [r for r in rows if (r.get("n_2025") or 0) >= min_n and r.get("edge_2025") is not None]
    if not eligible:
        return None
    # Tie-break (pre-specified): higher edge, then larger n, then mid over ask over bid, then later entry (T-60).
    fill_rank = {"mid": 0, "ask": 1, "bid": 2}

    def key(r):
        return (r["edge_2025"], r["n_2025"], -fill_rank.get(r["fill"], 9), r["tmin"])

    return max(eligible, key=key)


def AdverseSplit(recs, cfg):
    """Maker adverse-selection: would-fill vs not, on the same quotes (no price-bin cut)."""
    if cfg["fill"] != "bid":
        return None
    buckets = {"2025": {"fill": [], "no_fill": []}, "2026": {"fill": [], "no_fill": []}}
    for r in recs:
        if r["kind"] != cfg["kind"] or not NearKey(r["strike"], cfg["key"], cfg["kind"]):
            continue
        q = r["quotes"].get(cfg["tmin"])
        sq = SideQuotes(q, cfg["side"])
        if not sq:
            continue
        fill = sq["bid"]
        lo, hi = cfg["bin_lo"], BinHi(cfg["bin_lo"])
        if not (lo <= fill < hi or (hi == 1.0 and fill == 1.0)):
            continue
        result = r["result_yes"] if cfg["side"] == "yes" else 1 - r["result_yes"]
        edge = result - fill - FEE
        season = "2025" if r["season"] == 2025 else ("2026" if r["season"] == 2026 else None)
        if season is None:
            continue
        flag = "fill" if r["would_fill"].get((cfg["tmin"], cfg["side"])) else "no_fill"
        buckets[season][flag].append(edge)
    out = {}
    for season, parts in buckets.items():
        rec = {}
        for name, xs in parts.items():
            rec[f"n_{name}"] = len(xs)
            rec[f"mean_edge_{name}"] = None if not xs else float(np.mean(xs))
        out[season] = rec
    return out


def DailyEdges(recs, cfg, season=2025, fee_kind="tim"):
    bets = [b for b in BetsFor(recs, cfg, fee_kind=fee_kind) if b["season"] == str(season)]
    by = defaultdict(list)
    for b in bets:
        by[b["date"]].append(b["edge"])
    return {d: float(np.mean(xs)) for d, xs in by.items()}


def SpaAndRc(recs, configs, rows, season=2025, n_boot=N_SPA, seed=SEED, block=BLOCK):
    """White RC and Hansen SPA p-values for max 2025 mean edge, day-block bootstrap."""
    days = sorted({r["date"] for r in recs if int(r["season"]) == season})
    if not days or not configs:
        return {"rc_p": None, "spa_p": None, "n_days": len(days), "n_configs": len(configs),
                "n_boot": n_boot, "block": block}
    idx = {d: i for i, d in enumerate(days)}
    D = np.zeros((len(configs), len(days)))
    for k, cfg in enumerate(configs):
        de = DailyEdges(recs, cfg, season=season)
        for d, v in de.items():
            D[k, idx[d]] = v
    mean = D.mean(1)
    obs = float(mean.max()) if len(mean) else 0.0
    # White: fully recenter. Hansen SPA: subtract min(mean, 0).
    center_rc = mean
    center_spa = np.minimum(mean, 0.0)
    blocks = [days[i:i + block] for i in range(0, len(days), block)]
    rng = np.random.default_rng(seed)
    hits_rc = hits_spa = 0
    n_days = len(days)
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        cols = []
        for b in chosen:
            for d in blocks[b]:
                cols.append(idx[d])
        if not cols:
            continue
        arr = np.array(cols[:n_days] if len(cols) >= n_days else cols)
        boot = D[:, arr]
        stat_rc = (boot.mean(1) - center_rc).max()
        stat_spa = (boot.mean(1) - center_spa).max()
        if stat_rc >= obs - 1e-15:
            hits_rc += 1
        if stat_spa >= obs - 1e-15:
            hits_spa += 1
    return {
        "rc_p": hits_rc / n_boot, "spa_p": hits_spa / n_boot,
        "observed_max_mean_edge": obs, "n_days": n_days, "n_configs": len(configs),
        "n_boot": n_boot, "n_hits_rc": hits_rc, "n_hits_spa": hits_spa, "block": block,
    }


def _NormCdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _NormPpf(p):
    """Acklam's inverse normal CDF."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577509590705e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def DeflatedSharpe(returns, n_trials, ann=1.0):
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio. returns are per-period."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    t = len(x)
    if t < 3 or n_trials < 2:
        return {"dsr": None, "sr": None, "sr0": None, "n": t, "n_trials": n_trials,
                "skew": None, "kurtosis": None, "note": "not enough observations"}
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 0:
        return {"dsr": None, "sr": 0.0 if mu == 0 else math.inf, "sr0": None, "n": t,
                "n_trials": n_trials, "skew": None, "kurtosis": None,
                "note": "zero variance"}
    sr = (mu / sd) * math.sqrt(ann)
    # skew / kurtosis of the non-annualized per-period returns (paper convention)
    sr_bar = mu / sd
    m3 = float(np.mean((x - mu) ** 3))
    m4 = float(np.mean((x - mu) ** 4))
    skew = m3 / (sd ** 3)
    kurt = m4 / (sd ** 4)  # Pearson, 3 for normal
    den = 1.0 - skew * sr_bar + ((kurt - 1.0) / 4.0) * (sr_bar ** 2)
    if den <= 0:
        return {"dsr": None, "sr": sr, "sr0": None, "n": t, "n_trials": n_trials,
                "skew": skew, "kurtosis": kurt, "note": "non-positive variance formula"}
    var_sr = den / (t - 1)
    z1 = _NormPpf(1.0 - 1.0 / n_trials)
    z2 = _NormPpf(1.0 - 1.0 / (n_trials * math.e))
    sr0 = math.sqrt(var_sr) * ((1.0 - EULER) * z1 + EULER * z2)
    z = (sr_bar - sr0) * math.sqrt(t - 1) / math.sqrt(den)
    return {
        "dsr": float(_NormCdf(z)), "sr": sr, "sr_per_period": sr_bar, "sr0": sr0,
        "z": z, "n": t, "n_trials": n_trials, "skew": skew, "kurtosis": kurt,
        "ann": ann, "note": "DSR is Phi(z); values near 1 suggest the SR survives the trial count.",
    }


def LiquidityTable(recs):
    rows = []
    for kind in ("spread", "total"):
        for season in (2025, 2026):
            for tmin in TIMES:
                widths, vols, ois = [], [], []
                n_q = 0
                n = 0
                games = set()
                for r in recs:
                    if r["kind"] != kind or r["season"] != season:
                        continue
                    n += 1
                    games.add(r["game_id"])
                    q = r["quotes"].get(tmin)
                    if not q:
                        continue
                    n_q += 1
                    if q.get("width") is not None:
                        widths.append(q["width"])
                    if q.get("volume") is not None:
                        vols.append(q["volume"])
                    if q.get("open_interest") is not None:
                        ois.append(q["open_interest"])
                rows.append({
                    "market_kind": kind, "season": season, "entry": f"T-{tmin}",
                    "n_markets": n, "n_games": len(games), "n_with_quote": n_q,
                    "median_width": None if not widths else float(np.median(widths)),
                    "p25_width": None if not widths else float(np.quantile(widths, 0.25)),
                    "p75_width": None if not widths else float(np.quantile(widths, 0.75)),
                    "median_volume": None if not vols else float(np.median(vols)),
                    "n_with_volume": len(vols),
                    "median_open_interest": None if not ois else float(np.median(ois)),
                    "n_with_oi": len(ois),
                    "depth_available": False,
                })
    return rows


def EdgesForWalk(bets, fee_kind="tim", p_frozen=None):
    """CommonFormat / Walk-compatible edge rows. Qualify already applied by BetsFor.

    p_frozen is the 2025 hit rate of the frozen rule, used for qkelly sizing only.
    """
    out = []
    for b in bets:
        fill = b["fill"]
        fee = KalshiFee(fill) if fee_kind == "kalshi" else FEE
        cost = fill + fee
        if cost <= 0 or cost >= 1:
            continue
        e = dict(b)
        e["edge"] = b["result"] - fill - fee
        e["qualifies"] = True
        e["fee"] = fee
        p = float(p_frozen) if p_frozen is not None else float(fill)
        e["model_prob"] = p
        e["_p"] = p
        e["_fill"] = fill
        e["_fav_fill"] = fill
        e["_fav_result"] = int(b["result"])
        out.append(e)
    return out


def BankAndMc(edges, days, rng, fee_kind="tim", n_paths=N_PATHS):
    bundle, dailies, logs = BankBundle(edges, days, rng, fee_kind)
    note = {}
    for strat, daily in dailies.items():
        if not daily:
            continue
        rets = DailyReturns(daily, start=START)
        horizon = max(1, len(rets))
        paths = PathEvaluate(rets, seed=SEED, paths=n_paths, horizons={"window": horizon})
        bundle[strat]["pathsim"] = paths
        block = ((paths or {}).get("randomized") or {}).get("window|block_bootstrap|absorbing")
        if block:
            note[strat] = {
                "median_final": block.get("median_final"),
                "p10_final": block.get("p10_final"),
                "p90_final": block.get("p90_final"),
                "p_ruin": block.get("p_ruin"),
                "p_final_below_start": block.get("p_final_below_start"),
                "paths": block.get("paths"),
                "horizon": block.get("horizon"),
            }
    return bundle, dailies, logs, note


def HoldoutCi(bets, recs, rng):
    """95% CIs on holdout mean edge / ROI; widest of day-block, game-cluster, binomial."""
    if not bets:
        return {"widest_95": None, "day_block_95": None, "game_cluster_95": None,
                "binomial_mapped_95": None}
    edges = EdgesForWalk(bets, "tim")
    # dummy days list
    days = sorted({b["date"] for b in bets})
    ci = WidestRoiCi(
        [{"pnl": ((1.0 / (b["fill"] + FEE)) * b["result"] - 1.0) if 0 < b["fill"] + FEE < 1 else 0.0,
          "stake": 1.0, "event_id": b["event_id"], "date": b["date"]} for b in bets],
        edges, rng,
    )
    return ci


def Verdict(best, spa, dsr, bank, missing):
    parts = []
    claim = False
    if missing:
        parts.append("PARTIAL COVERAGE: missing " + ", ".join(missing) +
                     ". Results below are from the candles that were on disk; they are not a complete 2025-fit / 2026-holdout.")
    if not best:
        parts.append("No config had 20 or more 2025 bets, so nothing was frozen.")
        return " ".join(parts), False
    parts.append(
        f"Best 2025 rule is `{best['id']}` with mean edge {best['edge_2025']:.4f} "
        f"(N={best['n_2025']}, selected={best['selected']})."
    )
    if not best["selected"]:
        parts.append("It did not clear the pre-specified 3c / n>=20 screen, so it is reported but not a betting rule.")
    p_spa, p_rc = (spa or {}).get("spa_p"), (spa or {}).get("rc_p")
    if p_spa is not None and p_spa > 0.05:
        parts.append(f"Hansen SPA p={p_spa:.3f} (White RC p={p_rc:.3f}); the top of a {spa.get('n_configs')} "
                     "config search does not survive multiple testing at 5%.")
    elif p_spa is not None:
        parts.append(f"SPA p={p_spa:.3f}, White RC p={p_rc:.3f} on the 2025 search sample.")
    d = (dsr or {}).get("dsr")
    if d is not None:
        parts.append(f"Deflated Sharpe on 2025 daily mean edges of the winner is {d:.3f} "
                     f"(N_trials={dsr.get('n_trials')}).")
    if best.get("edge_2026") is not None:
        parts.append(f"2026 holdout mean edge (tim_est_0.02) is {best['edge_2026']:.4f} (N={best['n_2026']}).")
    # claim bar: selected, SPA <= 0.05, holdout widest CI > 0 under both fees
    def lo(bundle):
        ci = ((bundle or {}).get("flat100") or {}).get("ci") or {}
        w = ci.get("widest_95")
        return None if not w else w[0]
    lo_t = lo((bank or {}).get("tim"))
    lo_k = lo((bank or {}).get("kalshi"))
    if (best["selected"] and p_spa is not None and p_spa <= 0.05 and lo_t is not None and lo_k is not None
            and lo_t > 0 and lo_k > 0 and not missing):
        parts.append("Widest 95% holdout CI is above 0 under both fee bases and SPA is significant; "
                     "this still needs a serious sample and is a simulation only.")
        claim = True
    else:
        parts.append("No betting edge is claimed (need a 3c 2025 screen, SPA at 5%, and a widest "
                     "holdout CI above 0 under both tim_est_0.02 and the Kalshi taker fee, on complete coverage).")
    return " ".join(parts), claim


def _FmtPct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _FmtCi(ci):
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{_FmtPct(ci[0])}, {_FmtPct(ci[1])}]"


def _FmtMoney(x):
    return "n/a" if x is None else f"${x:,.2f}"


def MdRules(meta):
    best = meta.get("best") or {}
    spa = meta.get("spa") or {}
    dsr = meta.get("dsr") or {}
    bank = (meta.get("bank_tim") or {}).get("flat100") or {}
    lines = [
        "# NFL thin-markets model-free rule search (2026-11-13)",
        "",
        f"Made by American Footbot, {meta['written_at']}. Code `{meta['code']}`, run_id `{meta['run_id']}`. "
        "Simulation only; no orders, no real wagers.",
        "",
    ]
    if meta.get("missing"):
        lines += [
            f"**PARTIAL COVERAGE: missing {', '.join(meta['missing'])}.** "
            "Do not treat this as a complete 2025-fit / 2026-holdout.",
            "",
        ]
    lines += [
        "## Pre-registered grid",
        f"- {meta['n_configs']} configs: 20 price bins × 2 sides × 3 entry times "
        f"(T-180/T-120/T-60) × key-number proximity × 3 fills (mid, ask, bid).",
        "- Spreads: keys `any`, `prox_3`, `prox_7`, `prox_3_or_7` (1,440). "
        "Totals: `any`, `prox_41`, `prox_44`, `prox_47`, `prox_41_44_47` (1,800). "
        f"Proximity is |strike| (spreads) or strike (totals) within {PROX_TOL} of the key.",
        "- Mid is Tim's primary fill rule. Ask is taker. Bid is maker, counted only when a later "
        "1-min candle up to kickoff trades at or through the bid.",
        "- Fit/select on 2025 only (n >= 20 and mean edge >= $0.03). 2026 is the holdout.",
        "- Edge = outcome − (price + tim_est_0.02). `tim_est_0.02` is Tim's fee estimate. "
        "Kalshi taker fee 0.07p(1-p) is a recompute of the same bets.",
        f"- Prior search tally was at least {PRIOR_TALLY}. This grid adds {meta['n_configs']}. "
        f"New lower-bound tally: **{meta['new_tally']}**.",
        f"- White's Reality Check and Hansen SPA: {N_SPA:,} day-block (block={BLOCK}) bootstraps "
        "over all configs on 2025 daily mean edges. Deflated Sharpe Ratio uses N = grid size.",
        f"- Monte Carlo: {N_PATHS:,} PathSim paths, $10k start, flat100 and qkelly (2% cap), frozen best only.",
        "",
        "## Frozen best (picked on 2025 only)",
    ]
    if not best:
        lines.append("- No config reached n >= 20 on 2025.")
    else:
        lines += [
            f"- id: `{best.get('id')}`",
            f"- 2025: N={best.get('n_2025')}, hit rate={best.get('obs_2025')}, "
            f"mean edge (tim_est_0.02)={best.get('edge_2025')}, "
            f"mean edge (Kalshi fee)={best.get('edge_2025_kalshi')}, selected={best.get('selected')}",
            f"- 2026 holdout: N={best.get('n_2026')}, hit rate={best.get('obs_2026')}, "
            f"mean edge (tim_est_0.02)={best.get('edge_2026')}, "
            f"mean edge (Kalshi fee)={best.get('edge_2026_kalshi')}",
        ]
    lines += [
        "",
        "## SPA, Reality Check, Deflated Sharpe",
        f"- White RC p-value (2025): **{spa.get('rc_p')}**. "
        f"Hansen SPA p-value: **{spa.get('spa_p')}** "
        f"(N configs={spa.get('n_configs')}, N days={spa.get('n_days')}, boots={spa.get('n_boot')}).",
        f"- DSR (2025 daily mean edges of the winner, N_trials={dsr.get('n_trials')}): "
        f"**{dsr.get('dsr')}** (SR_per_period={dsr.get('sr_per_period')}, SR_0={dsr.get('sr0')}, "
        f"skew={dsr.get('skew')}, kurtosis={dsr.get('kurtosis')}, n_days={dsr.get('n')}).",
        "",
        "## Verdict",
        meta.get("verdict") or "",
        "",
        "## 2026 holdout bankroll, frozen best, mid/ask/bid as frozen, tim_est_0.02",
        "",
        "| Strategy | N bets | End bank | P&L | ROI | Widest 95% CI | day-block 95% | Ruin? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("flat100", "qkelly"):
        b = ((meta.get("bank_tim") or {}).get(name)) or {}
        ci = (b.get("ci") or {})
        lines.append(
            f"| {name} | {b.get('n_bets')} | {_FmtMoney(b.get('final_equity'))} | "
            f"{_FmtMoney(b.get('pnl'))} | {_FmtPct(b.get('roi'))} | "
            f"{_FmtCi(ci.get('widest_95'))} | {_FmtCi(ci.get('day_block_95'))} | "
            f"{'yes' if b.get('ruined') else 'no'} |"
        )
    if meta.get("bank_kalshi"):
        lines += ["", "Kalshi taker fee 0.07p(1-p), same bets:", "",
                  "| Strategy | N bets | End bank | P&L | ROI | Widest 95% CI |",
                  "|---|---|---|---|---|---|"]
        for name in ("flat100", "qkelly"):
            b = (meta["bank_kalshi"].get(name)) or {}
            ci = (b.get("ci") or {})
            lines.append(
                f"| {name} | {b.get('n_bets')} | {_FmtMoney(b.get('final_equity'))} | "
                f"{_FmtMoney(b.get('pnl'))} | {_FmtPct(b.get('roi'))} | {_FmtCi(ci.get('widest_95'))} |"
            )
    lines += ["", "## Maker adverse selection (frozen best, if maker-at-bid)"]
    adv = meta.get("adverse")
    if not adv:
        lines.append("Frozen best is not a maker-at-bid rule, or no comparison was available.")
    else:
        for season in ("2025", "2026"):
            a = adv.get(season) or {}
            lines.append(
                f"- {season}: would-fill N={a.get('n_fill')}, mean edge={a.get('mean_edge_fill')}; "
                f"would-not N={a.get('n_no_fill')}, mean edge={a.get('mean_edge_no_fill')}."
            )
        lines.append("A worse mean edge on would-fill than would-not is the usual adverse-selection pattern.")
    lines += ["", "## Width and liquidity by kind, season, entry time",
              "",
              "| Kind | Season | Entry | N markets | N quotes | Median width | p25–p75 | Median volume | Median OI |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in meta.get("liquidity") or []:
        def f(x, d=3):
            return "n/a" if x is None else f"{x:.{d}f}"
        vw = "n/a" if r.get("median_volume") is None else f"{r['median_volume']:.0f}"
        oi = "n/a" if r.get("median_open_interest") is None else f"{r['median_open_interest']:.0f}"
        p25, p75 = f(r.get("p25_width")), f(r.get("p75_width"))
        lines.append(
            f"| {r['market_kind']} | {r['season']} | {r['entry']} | {r['n_markets']} | "
            f"{r['n_with_quote']} | {f(r.get('median_width'))} | {p25}–{p75} | {vw} | {oi} |"
        )
    lines += [
        "",
        "Candles have no size, so depth is not available. Volume and open interest are whatever "
        "Kalshi ships on the 1-min candle used for the quote (often missing). T-180 quotes can be "
        "sparse because the pull window starts at kickoff−180, and QuoteAt needs a bar at or before that stamp "
        f"(stale if older than {STALE_MIN} minutes).",
        "",
        "## PathSim (frozen best, tim_est_0.02, block-bootstrap absorbing)",
        json.dumps(meta.get("pathsim_note") or {}, indent=2, default=str),
        "",
        "## Caveats",
        "- Model-free, simulated, 2025 contaminated by this search. 2026 is the holdout only if coverage is complete.",
        "- Maker fills inferred from 1-min bid/ask, not from trade prints or queue position; optimistic.",
        "- `tim_est_0.02` is Tim's estimate, not a verified exchange fee.",
        f"- Workers={meta.get('workers')}, seed={SEED}. FrozenConfig was not edited.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _StripBank(bundle):
    out = {}
    for k, v in (bundle or {}).items():
        if not isinstance(v, dict):
            out[k] = v
            continue
        d = {kk: vv for kk, vv in v.items() if kk != "pathsim"}
        if "pathsim" in v:
            block = ((v["pathsim"] or {}).get("randomized") or {}).get("window|block_bootstrap|absorbing")
            d["pathsim_block_abs"] = block
        out[k] = d
    return out


def RunRules(out_dir, base=None, entries=None, markets=None, fees=None,
             workers=7, spa_boot=N_SPA, n_paths=N_PATHS, seed=SEED, smoke=False):
    written = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p %Z")
    run_id = NewRunId()
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    base = base or os.path.join(out_dir, "2026-11-13_thin-markets")
    if not base.startswith(out_dir) and not os.path.isabs(base):
        base = os.path.join(out_dir, base)
    if entries is None:
        entries = LoadThinEntries()
    if markets is None:
        markets = LoadMarkets()
    if fees is None:
        fees = LoadFees()
    missing = CoverageGaps(entries)
    if missing:
        print("PARTIAL COVERAGE:", ", ".join(missing), flush=True)
    print("building rule-search records...", flush=True)
    recs = BuildRecs(entries, markets)
    configs = ConfigGrid()
    if smoke:
        configs = configs[:24]
        spa_boot = min(spa_boot, 50)
        n_paths = min(n_paths, 50)
    n_cfg = len(configs)
    print(f"rules grid={n_cfg} workers={workers} recs={len(recs)}", flush=True)
    rows = RunGrid(recs, configs, workers=1 if smoke else workers)
    best = PickBest(rows)
    print("best", None if not best else best["id"],
          None if not best else best.get("edge_2025"), flush=True)
    print("running SPA / White RC...", flush=True)
    # SPA over the full (non-smoke-truncated if we want; on smoke use the short list)
    spa = SpaAndRc(recs, configs, rows, season=2025, n_boot=spa_boot, seed=seed)
    print("spa", spa.get("spa_p"), "rc", spa.get("rc_p"), flush=True)

    bank_t = bank_k = {}
    dailies = {}
    path_note = {}
    adv = None
    dsr = {}
    holdout_ci = {}
    if best:
        cfg = {k: best[k] for k in ("kind", "side", "tmin", "key", "fill", "bin_lo")}
        bets_t = [b for b in BetsFor(recs, cfg, "tim") if b["season"] == "2026"]
        bets_k = [b for b in BetsFor(recs, cfg, "kalshi") if b["season"] == "2026"]
        days = sorted({b["date"] for b in bets_t})
        edges_t = EdgesForWalk(bets_t, "tim", p_frozen=best.get("obs_2025"))
        edges_k = EdgesForWalk(bets_k, "kalshi", p_frozen=best.get("obs_2025"))
        bank_t, dailies, _, path_note = BankAndMc(edges_t, days, rng, "tim", n_paths=n_paths)
        bank_k, _, _, _ = BankAndMc(edges_k, days, rng, "kalshi", n_paths=max(1, n_paths // 5) if smoke else n_paths)
        holdout_ci = HoldoutCi(bets_t, recs, rng)
        if cfg["fill"] == "bid":
            adv = AdverseSplit(recs, cfg)
        # DSR on 2025 daily mean edges of the winner (selection sample)
        de = DailyEdges(recs, cfg, season=2025)
        all_days = sorted({r["date"] for r in recs if r["season"] == 2025})
        series = [de.get(d, 0.0) for d in all_days]
        dsr = DeflatedSharpe(series, n_trials=n_cfg, ann=1.0)

    liq = LiquidityTable(recs)
    n_sel = sum(1 for r in rows if r.get("selected"))
    verdict, claim = Verdict(best, spa, dsr, {"tim": bank_t, "kalshi": bank_k}, missing)
    ranked = sorted(
        [r for r in rows if r.get("edge_2025") is not None],
        key=lambda r: (r["edge_2025"], r.get("n_2025") or 0), reverse=True,
    )
    top50 = ranked[:50]
    meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "written_at": written, "workers": workers, "seed": seed,
        "n_configs": n_cfg, "new_tally": PRIOR_TALLY + n_cfg,
        "prior_tally": PRIOR_TALLY, "n_recs": len(recs),
        "n_selected_2025": n_sel, "missing": missing, "partial": bool(missing),
        "best": best, "spa": spa, "dsr": dsr, "adverse": adv,
        "bank_tim": _StripBank(bank_t), "bank_kalshi": _StripBank(bank_k),
        "holdout_ci": {k: (None if v is None or (isinstance(v, tuple) and v[0] is None) else list(v))
                       for k, v in (holdout_ci or {}).items()},
        "liquidity": liq, "pathsim_note": path_note,
        "verdict": verdict, "claim": claim,
        "fee_note": "tim_est_0.02 is Tim's estimate; Kalshi taker is 0.07*p*(1-p).",
        "depth_note": "Candles have no size; depth is unavailable.",
    }
    WriteCsv(base + "_rules_grid.csv", [
        "id", "kind", "side", "tmin", "key", "fill", "bin_lo",
        "n_2025", "k_2025", "obs_2025", "edge_2025", "edge_2025_kalshi", "selected",
        "n_2026", "k_2026", "obs_2026", "edge_2026", "edge_2026_kalshi",
        "fill_2025", "fill_2026", "mean_width_2025", "mean_width_2026",
    ], rows)
    WriteCsv(base + "_rules_top50.csv", [
        "id", "kind", "side", "tmin", "key", "fill", "bin_lo",
        "n_2025", "edge_2025", "edge_2025_kalshi", "selected",
        "n_2026", "edge_2026", "edge_2026_kalshi", "obs_2025", "obs_2026",
    ], top50)
    mc_rows = []
    for fee_name, bundle in (("tim_est_0.02", bank_t), ("kalshi_taker", bank_k)):
        for strat, b in (bundle or {}).items():
            if not isinstance(b, dict):
                continue
            ci = b.get("ci") or {}
            block = (b.get("pathsim") or {}).get("randomized", {}).get("window|block_bootstrap|absorbing") or {}
            if "pathsim" not in b:
                # already stripped in meta; look at path_note
                block = (path_note.get(strat) if fee_name == "tim_est_0.02" else {}) or {}
            mc_rows.append({
                "fee_basis": fee_name, "strategy": strat, "n_bets": b.get("n_bets"),
                "final_equity": b.get("final_equity"), "pnl": b.get("pnl"), "roi": b.get("roi"),
                "ruined": b.get("ruined"),
                "widest_95_lo": None if not ci.get("widest_95") else ci["widest_95"][0],
                "widest_95_hi": None if not ci.get("widest_95") else ci["widest_95"][1],
                "day_block_95_lo": None if not ci.get("day_block_95") else ci["day_block_95"][0],
                "day_block_95_hi": None if not ci.get("day_block_95") else ci["day_block_95"][1],
                "game_cluster_95_lo": None if not ci.get("game_cluster_95") else ci["game_cluster_95"][0],
                "game_cluster_95_hi": None if not ci.get("game_cluster_95") else ci["game_cluster_95"][1],
                "binomial_95_lo": None if not ci.get("binomial_mapped_95") else ci["binomial_mapped_95"][0],
                "binomial_95_hi": None if not ci.get("binomial_mapped_95") else ci["binomial_mapped_95"][1],
                "mc_paths": block.get("paths") or n_paths,
                "mc_median_final": block.get("median_final"),
                "mc_p10_final": block.get("p10_final"),
                "mc_p90_final": block.get("p90_final"),
                "mc_p_ruin": block.get("p_ruin"),
                "mc_p_final_below_start": block.get("p_final_below_start"),
            })
    WriteCsv(base + "_rules_mc.csv", [
        "fee_basis", "strategy", "n_bets", "final_equity", "pnl", "roi", "ruined",
        "widest_95_lo", "widest_95_hi", "day_block_95_lo", "day_block_95_hi",
        "game_cluster_95_lo", "game_cluster_95_hi", "binomial_95_lo", "binomial_95_hi",
        "mc_paths", "mc_median_final", "mc_p10_final", "mc_p90_final", "mc_p_ruin",
        "mc_p_final_below_start",
    ], mc_rows)
    with open(base + "_rules.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=_JsonDefault)
    with open(base + "_rules.md", "w", encoding="utf-8") as fh:
        fh.write(MdRules(meta))
    print(json.dumps({
        "run_id": run_id, "best": None if not best else best.get("id"),
        "n_configs": n_cfg, "new_tally": PRIOR_TALLY + n_cfg,
        "spa_p": spa.get("spa_p"), "rc_p": spa.get("rc_p"),
        "dsr": dsr.get("dsr"), "claim": claim, "partial": bool(missing),
        "missing": missing, "out": out_dir,
    }, indent=1, default=_JsonDefault), flush=True)
    return meta


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Model-free thin-market rule search (run when Tim approves).")
    ap.add_argument("--out-dir", default=r"E:\AgentDocs\American-Footbot\2026-11-13_thin-markets")
    ap.add_argument("--base", default="2026-11-13_thin-markets")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--spa-boot", type=int, default=N_SPA)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    base = os.path.join(a.out_dir, a.base)
    return RunRules(a.out_dir, base=base, workers=a.workers,
                    spa_boot=a.spa_boot, n_paths=a.paths, smoke=a.smoke)


if __name__ == "__main__":
    Main()
