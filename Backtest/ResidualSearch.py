r"""
Residual-on-market walk-forward search (Job C, 2026-11-13).

Pre-registered before looking at 2023-2026 results:
  Offset logistic: logit(p) = logit(market) + intercept + X b, walk-forward by season.
  Market is book_close_devig (primary) or the opening line (variant, only if open
  lines are on disk). The open-to-close move is a TARGET (CLV / predict-the-close),
  never a feature when the offset is the close.
  Features, all season-to-date on earlier gamedays only:
    epa, sr, ypp, qb, rest, travel, weather, injury.
  Grid (close-offset, pre-specified): every subset of the 8 groups (256) x l2 in
  {0.5, 1, 2, 5, 10} x standardize in {0, 1} = 2,560 configs. If opening lines
  exist, the same 2,560 are also run with offset=open (no extra selection into
  the close-target freeze).
  Tune seasons 2015-2022 only pick the winner (lowest mean ign_rel_bits vs the
  close). 2023-2025 are labelled contaminated. 2026 is reported forward but is
  not used to pick. New trial count is added to the 2,819+ tally.
  SPA / White Reality Check: 10,000 day-block bootstraps over all close-offset
  configs on the tune games. Label-permutation null (10,000) on the frozen best.

    .venv\Scripts\python.exe -m Backtest.ResidualSearch --out-dir DIR

Normal priority. About cpu/4 workers. Simulation only; no orders.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.BankrollPlot import FrameTidy, PlotBankroll
from Backtest.CommonFormat import (
    ALPHA as ALPHA90, BANK_FIELDS, ClopperPearson, Devig, FEE, FEE_BASIS, FLAT, MIN_EDGE,
    QKellyStake, QuoteWithTs, RUIN, START, Walk, Widest, WriteCsv, AllGameDays,
    KalshiRows, QKELLY_CAP,
)
from Backtest.Features import Logit, BuildRows
from Backtest.FrozenConfig import FROZEN
from Backtest.KalshiReality import KalshiFee, LoadQuotes
from Backtest.Lab import AllRows, LoadSeason, Sig
from Backtest.PathSim import DailyReturns, Evaluate as PathEvaluate
from Backtest.Version import CodeVersion, ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
SPORT = "nfl"
TUNE = list(range(FROZEN["tune_seasons"][0], FROZEN["tune_seasons"][1] + 1))  # 2015-2022
CONTAMINATED = list(FROZEN["contaminated_test_seasons"])  # 2023-2025
FIRST, LAST = 2014, 2026
GROUPS = ("epa", "sr", "ypp", "qb", "rest", "travel", "weather", "injury")
L2_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
STD_GRID = (0, 1)
N_SPA = 10_000
N_PERM = 10_000
N_PATHS = 10_000
SEED = 20261113
PRIOR_TALLY = 2819
MIN_PLAYS = 40
_RBS = None
_SEASONS = None


def InitWorker(rbs, seasons):
    """Load walk-forward inputs once per process (Windows spawn)."""
    global _RBS, _SEASONS
    _RBS = rbs
    _SEASONS = seasons


def EvalCfg(cfg):
    recs = WalkForwardConfig(_RBS, cfg, _SEASONS)
    return ConfigId(cfg), cfg, recs

FEAT_COLS = {
    "epa": ["epa_diff"],
    "sr": ["sr_diff"],
    "ypp": ["ypp_diff"],
    "qb": ["qb_epa_diff"],
    "rest": ["rest_diff"],
    "travel": ["travel_diff"],
    "weather": ["outdoor", "temp_z", "wind_z"],
    "injury": ["inj_qb_diff", "inj_off_diff", "inj_def_diff"],
}

# Approximate home-stadium coordinates for travel (decimal degrees).
TEAM_XY = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7553, -84.4006), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0950, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7478, -97.0928),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3390), "LAC": (33.9535, -118.3390),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9736, -93.2575),
    "NE": (42.0909, -71.2643), "NO": (29.9509, -90.0812), "NYG": (40.8136, -74.0744),
    "NYJ": (40.8136, -74.0744), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
    "OAK": (37.7516, -122.2005), "SD": (32.7831, -117.1196), "STL": (38.6313, -90.1887),
}


def WorkerCount():
    n = os.cpu_count() or 4
    return max(1, n // 4)


def ClipP(p):
    return min(max(float(p), 1e-6), 1 - 1e-6)


def HaversineMiles(a, b):
    if not a or not b:
        return 0.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3958.8 * 2 * math.asin(min(1.0, math.sqrt(h)))


def LoadJsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def LoadTeamEff(first=FIRST, last=LAST):
    by = defaultdict(list)
    for s in range(first, last + 1):
        for r in LoadJsonl(os.path.join("data", "backtest", f"team_game_eff_{s}.jsonl")):
            by[r["game_id"]].append(r)
    return by


def LoadInjuries(first=FIRST, last=LAST):
    out = {}
    for s in range(first - 1, last + 1):
        for r in LoadJsonl(os.path.join("data", "injuries", f"features_{s}.jsonl")):
            out[(r["game_id"], r["team"])] = r
    return out


def LoadOpenLines():
    """Optional. Expected JSONL keys: game_id, open_home_ml, open_away_ml (or open_home)."""
    rows = {}
    for s in range(FIRST, LAST + 1):
        for r in LoadJsonl(os.path.join("data", "backtest", f"open_lines_{s}.jsonl")):
            rows[r["game_id"]] = r
    path = os.path.join("data", "backtest", "open_lines.jsonl")
    for r in LoadJsonl(path):
        rows[r["game_id"]] = r
    return rows


def OpenProb(rec):
    if not rec:
        return None
    if rec.get("open_home") is not None:
        return ClipP(rec["open_home"])
    from Backtest.Odds import FairProbs
    h, a = rec.get("open_home_ml"), rec.get("open_away_ml")
    if h is None or a is None:
        return None
    mh, _ = FairProbs(h, a)
    return ClipP(mh)


def WeatherVec(g):
    roof = (g.get("roof") or "").lower()
    outdoor = 0.0 if roof in ("dome", "closed", "indoor") else 1.0
    temp, wind = g.get("temp"), g.get("wind")
    temp_z = 0.0 if (not outdoor or temp is None) else (float(temp) - 70.0) / 20.0
    wind_z = 0.0 if (not outdoor or wind is None) else float(wind) / 10.0
    return outdoor, temp_z, wind_z


def Rate(num, den):
    return 0.0 if not den else num / den


def PriorRates(history):
    """history: list of team-game eff rows already in gameday order, earlier only."""
    op = oe = osu = oy = dp = de = dsu = dy = 0.0
    qb = defaultdict(lambda: [0.0, 0.0])
    for r in history:
        op += r["off_plays"]; oe += r["off_epa"]; osu += r["off_succ"]; oy += r["off_yards"]
        dp += r["def_plays"]; de += r["def_epa"]; dsu += r["def_succ"]; dy += r["def_yards"]
        if r.get("qb_id"):
            qb[r["qb_id"]][0] += r.get("qb_plays") or 0
            qb[r["qb_id"]][1] += r.get("qb_epa") or 0.0
    return {
        "epa": Rate(oe, op) - Rate(de, dp),
        "sr": Rate(osu, op) - Rate(dsu, dp),
        "ypp": Rate(oy, op) - Rate(dy, dp),
        "off_plays": op, "def_plays": dp, "qb": qb,
    }


def AttachResidualFeatures(rows_by_season, eff_by_game, inj, opens):
    """Add residual-search features in place. Season-to-date, earlier gamedays only."""
    games = {}
    for s, rows in rows_by_season.items():
        for r in rows:
            games[r["game_id"]] = r
    # last stadium each team played at, and eff history
    hist = defaultdict(list)
    last_xy = {}
    # need game stadium: use home team's coordinates when location is Home; neutral ~ home anyway
    ordered = []
    for s in sorted(rows_by_season):
        for r in rows_by_season[s]:
            ordered.append(r)
    ordered.sort(key=lambda r: (str(r["gameday"]), r.get("gametime") or "", r["game_id"]))
    # Build a raw schedule lookup for weather / qb / rest already on the feature row
    raw = {}
    for s in range(FIRST, LAST + 1):
        try:
            for g in LoadSeason(s):
                raw[g["game_id"]] = g
        except FileNotFoundError:
            continue
    for _day, group in _GroupDay(ordered):
        for r in group:
            g = raw.get(r["game_id"], {})
            h, a = r["home_team"], r["away_team"]
            ph, pa = PriorRates(hist[h]), PriorRates(hist[a])
            qb_h = r.get("home_qb_id") or g.get("home_qb_id")
            qb_a = r.get("away_qb_id") or g.get("away_qb_id")
            qh = ph["qb"].get(qb_h, [0.0, 0.0])
            qa = pa["qb"].get(qb_a, [0.0, 0.0])
            xy_h_prev, xy_a_prev = last_xy.get(h), last_xy.get(a)
            venue = TEAM_XY.get(h)
            travel_h = 0.0 if not xy_h_prev else HaversineMiles(xy_h_prev, venue)
            travel_a = 0.0 if not xy_a_prev else HaversineMiles(xy_a_prev, venue)
            outdoor, temp_z, wind_z = WeatherVec(g)
            ih = inj.get((r["game_id"], h), {})
            ia = inj.get((r["game_id"], a), {})
            open_p = OpenProb(opens.get(r["game_id"]))
            r["epa_diff"] = ph["epa"] - pa["epa"]
            r["sr_diff"] = ph["sr"] - pa["sr"]
            r["ypp_diff"] = ph["ypp"] - pa["ypp"]
            r["qb_epa_diff"] = Rate(qh[1], qh[0]) - Rate(qa[1], qa[0])
            r["rest_diff"] = r.get("rest_diff") or 0.0
            r["travel_diff"] = (travel_a - travel_h) / 500.0
            r["outdoor"] = outdoor
            r["temp_z"] = temp_z
            r["wind_z"] = wind_z
            r["inj_qb_diff"] = float(ih.get("qb_out") or 0) - float(ia.get("qb_out") or 0)
            r["inj_off_diff"] = float(ih.get("off_out") or 0) - float(ia.get("off_out") or 0)
            r["inj_def_diff"] = float(ih.get("def_out") or 0) - float(ia.get("def_out") or 0)
            r["prior_plays"] = min(ph["off_plays"], pa["off_plays"])
            r["open_p"] = open_p
            r["close_p"] = ClipP(r["mkt_home"]) if r.get("mkt_home") is not None else None
            r["close_logit"] = Logit(r["close_p"]) if r["close_p"] is not None else None
            r["open_logit"] = Logit(open_p) if open_p is not None else None
            r["oc_move"] = None if (open_p is None or r["close_p"] is None) else (r["close_p"] - open_p)
        # update history after the whole day
        for r in group:
            g = raw.get(r["game_id"], {})
            venue = TEAM_XY.get(r["home_team"])
            for team, side in ((r["home_team"], "home"), (r["away_team"], "away")):
                last_xy[team] = venue
                for er in eff_by_game.get(r["game_id"], []):
                    if er["team"] == team:
                        hist[team].append(er)
                        # reset season-to-date at season boundaries: keep only this season
                        hist[team] = [x for x in hist[team] if int(x["season"]) == int(r["season"])]
    return rows_by_season


def _GroupDay(rows):
    from itertools import groupby
    for day, grp in groupby(rows, key=lambda r: str(r["gameday"])):
        yield day, list(grp)


def FeatureNames(groups):
    out = []
    for g in groups:
        out.extend(FEAT_COLS[g])
    return out


def ConfigGrid(include_open=False, smoke=False):
    """Pre-specified grid. smoke=True returns a tiny subset for tests."""
    if smoke:
        specs = []
        for groups in (("epa",), ("epa", "rest"), GROUPS):
            specs.append({"groups": groups, "l2": 1.0, "standardize": 1, "offset": "close"})
        return specs
    specs = []
    for bits in itertools.product((0, 1), repeat=len(GROUPS)):
        groups = tuple(g for g, on in zip(GROUPS, bits) if on)
        # empty feature set is intercept-only residual: keep it (market calibration)
        for l2 in L2_GRID:
            for std in STD_GRID:
                specs.append({"groups": groups, "l2": float(l2), "standardize": int(std), "offset": "close"})
    if include_open:
        extras = [dict(s, offset="open") for s in specs]
        specs = specs + extras
    return specs


def ConfigId(c):
    g = ",".join(c["groups"]) or "intercept"
    return f"{c['offset']}|l2={c['l2']}|std={c['standardize']}|{g}"


def FitOffsetLogit(X, y, offset, l2=1.0):
    """logit(p) = offset + intercept + X w. Intercept is unpenalized."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.asarray(offset, dtype=float)
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    w = np.zeros(k + 1)
    reg = l2 * np.eye(k + 1)
    reg[0, 0] = 0.0
    for _ in range(25):
        p = Sig(off + Z @ w)
        g = Z.T @ (p - y) + reg @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
        try:
            w = w - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            w = w - np.linalg.lstsq(H, g, rcond=None)[0]
    return w


def ApplyOffset(X, offset, w):
    Z = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    return Sig(np.asarray(offset, dtype=float) + Z @ w)


def Standardize(train, test):
    mu = train.mean(0)
    sd = train.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def SeasonMatrix(rows, names, offset_key):
    X, y, off, keep = [], [], [], []
    for r in rows:
        if r.get("home_win") is None or r.get(offset_key) is None:
            continue
        X.append([float(r.get(n) or 0.0) for n in names])
        y.append(float(r["home_win"]))
        off.append(float(r[offset_key]))
        keep.append(r)
    return np.asarray(X, dtype=float), np.asarray(y), np.asarray(off), keep


def WalkForwardConfig(rows_by_season, cfg, seasons):
    """Fit on seasons < S, predict S. Returns per-game records for scored seasons."""
    names = FeatureNames(cfg["groups"])
    off_key = "close_logit" if cfg["offset"] == "close" else "open_logit"
    out = []
    for s in seasons:
        train_rows = [r for t, rs in rows_by_season.items() if t < s for r in rs]
        test_rows = rows_by_season.get(s) or []
        Xtr, ytr, otr, _ = SeasonMatrix(train_rows, names, off_key)
        Xte, yte, ote, keep = SeasonMatrix(test_rows, names, off_key)
        if len(ytr) < 50 or len(keep) == 0:
            continue
        if cfg["standardize"] and names:
            Xtr, Xte, _, _ = Standardize(Xtr, Xte)
        elif not names:
            Xtr = np.zeros((len(ytr), 0))
            Xte = np.zeros((len(keep), 0))
        w = FitOffsetLogit(Xtr, ytr, otr, l2=cfg["l2"])
        p = ApplyOffset(Xte, ote, w)
        mkt = Sig(ote)
        for i, r in enumerate(keep):
            out.append({
                "game_id": r["game_id"], "season": int(r["season"]), "gameday": r["gameday"],
                "week": r.get("week"), "home_win": int(yte[i]),
                "p": float(p[i]), "mkt": float(mkt[i]),
                "offset": cfg["offset"], "close_p": r.get("close_p"),
                "open_p": r.get("open_p"), "oc_move": r.get("oc_move"),
            })
    return out


def LogLossBits(p, y):
    p = ClipP(p)
    return -(y * math.log(p, 2) + (1 - y) * math.log(1 - p, 2))


def IgnRel(rows):
    if not rows:
        return None
    d = [LogLossBits(r["p"], r["home_win"]) - LogLossBits(r["mkt"], r["home_win"]) for r in rows]
    return float(np.mean(d))


def ClvStats(rows):
    """Predict-the-close diagnostics on games that have an opening line."""
    have = [r for r in rows if r.get("open_p") is not None and r.get("close_p") is not None]
    if not have:
        return {"n": 0, "mean_oc_move_when_long": None, "corr_edge_vs_move": None,
                "close_logloss_bits": None}
    edge = np.array([r["p"] - r["open_p"] for r in have])
    move = np.array([r["close_p"] - r["open_p"] for r in have], dtype=float)
    long = edge > 0
    mean_move = None if not long.any() else float(move[long].mean())
    corr = None
    if len(have) > 2 and edge.std() > 1e-12 and move.std() > 1e-12:
        corr = float(np.corrcoef(edge, move)[0, 1])
    # model p as a forecast of the close probability (not of the outcome)
    ll = float(np.mean([LogLossBits(r["p"], ClipP(r["close_p"])) for r in have]))
    return {"n": len(have), "mean_oc_move_when_long": mean_move,
            "corr_edge_vs_move": corr, "close_logloss_bits": ll}


def ScoreBlock(rows, seasons):
    sub = [r for r in rows if int(r["season"]) in seasons]
    return {"n": len(sub), "ign_rel_bits": IgnRel(sub), "clv": ClvStats(sub)}


def RunGrid(rows_by_season, configs, seasons, workers):
    results = []
    if workers <= 1 or len(configs) <= 2:
        InitWorker(rows_by_season, seasons)
        for c in configs:
            results.append(EvalCfg(c))
        return results
    with ProcessPoolExecutor(max_workers=workers, initializer=InitWorker,
                             initargs=(rows_by_season, seasons)) as ex:
        futs = [ex.submit(EvalCfg, c) for c in configs]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == len(configs):
                print(f"  grid {done}/{len(configs)}", flush=True)
    return results


def PickBest(results, tune=TUNE):
    """Lowest (most negative) ign_rel_bits vs close on tune seasons, close-offset only."""
    scored = []
    for cid, cfg, recs in results:
        if cfg["offset"] != "close":
            continue
        block = ScoreBlock(recs, tune)
        if block["n"] < 50 or block["ign_rel_bits"] is None:
            continue
        scored.append((block["ign_rel_bits"], cid, cfg, recs, block))
    scored.sort(key=lambda x: (x[0], cid_stable(x[1])))
    return scored[0] if scored else None


def cid_stable(s):
    return s


def SpaPValue(results, tune=TUNE, n_boot=N_SPA, seed=SEED, block=4):
    """Hansen SPA p-value for max ign_rel improvement vs market, tune games, close-offset."""
    # Build game-aligned loss differentials d = L_mkt - L_model (positive = model better)
    # Use only games present in every close-offset config (should be all close-eligible).
    close = [(cid, cfg, recs) for cid, cfg, recs in results if cfg["offset"] == "close"]
    if not close:
        return {"p": None, "n_configs": 0, "n_games": 0, "n_boot": n_boot}
    # index by game_id
    games = [r["game_id"] for r in close[0][2] if int(r["season"]) in tune]
    days = [r["gameday"] for r in close[0][2] if int(r["season"]) in tune]
    by_cfg = []
    idx = {gid: i for i, gid in enumerate(games)}
    n = len(games)
    if n == 0:
        return {"p": None, "n_configs": len(close), "n_games": 0, "n_boot": n_boot}
    D = np.full((len(close), n), np.nan)
    for k, (cid, cfg, recs) in enumerate(close):
        for r in recs:
            i = idx.get(r["game_id"])
            if i is None:
                continue
            D[k, i] = LogLossBits(r["mkt"], r["home_win"]) - LogLossBits(r["p"], r["home_win"])
    # drop configs with all-nan; fill remaining nan with 0 (shouldn't happen)
    D = np.nan_to_num(D, nan=0.0)
    mean = D.mean(1)
    # consistent recentering: Hansen, subtract mean if mean <= 0 (poor models)
    center = np.minimum(mean, 0.0)
    obs = float(mean.max())
    # day-block bootstrap
    unique_days = sorted(set(days))
    day_idx = defaultdict(list)
    for i, d in enumerate(days):
        day_idx[d].append(i)
    day_list = unique_days
    # form contiguous blocks of `block` days
    blocks = [day_list[i:i + block] for i in range(0, len(day_list), block)]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        cols = []
        for b in chosen:
            for d in blocks[b]:
                cols.extend(day_idx[d])
        if not cols:
            continue
        boot = D[:, np.array(cols[:n]) if len(cols) >= n else np.array(cols)]
        # recenter
        stat = (boot.mean(1) - center).max()
        if stat >= obs - 1e-15:
            hits += 1
    return {"p": hits / n_boot, "observed_mean_d": obs, "n_configs": len(close),
            "n_games": n, "n_boot": n_boot, "n_hits": hits, "block": block}


def PermutationP(recs, tune=TUNE, n_perm=N_PERM, seed=SEED):
    sub = [r for r in recs if int(r["season"]) in tune]
    if len(sub) < 20:
        return {"p": None, "n": len(sub), "n_perm": n_perm}
    y = np.array([r["home_win"] for r in sub], dtype=float)
    p = np.array([r["p"] for r in sub], dtype=float)
    m = np.array([r["mkt"] for r in sub], dtype=float)
    def gap(yy):
        return float(np.mean([LogLossBits(p[i], yy[i]) - LogLossBits(m[i], yy[i]) for i in range(len(yy))]))
    obs = gap(y)
    rng = np.random.default_rng(seed + 1)
    hits = 0
    for _ in range(n_perm):
        yy = rng.permutation(y)
        if gap(yy) <= obs + 1e-15:  # more negative or equal is as good
            hits += 1
    return {"p": hits / n_perm, "observed_ign_rel_bits": obs, "n": len(sub),
            "n_perm": n_perm, "n_hits": hits}


def AttachKalshiT60(rows_by_season, seasons=(2025, 2026)):
    """Attach Kalshi T-60 quotes and de-vigged mid. Does not use p_full2 or a full2 refit."""
    coverage = {}
    for s in seasons:
        quotes = LoadQuotes(s)
        rows = rows_by_season.get(s) or []
        n_ok = 0
        missing = 0
        for r in rows:
            r["kq_t60"] = None
            r["kmid"] = None
            g = quotes.get(r.get("game_id"), {})
            qh = QuoteWithTs(g.get(r.get("home_team")))
            qa = QuoteWithTs(g.get(r.get("away_team")))
            if not (qh and qa):
                missing += 1
                continue
            r["kq_t60"] = {"home": qh, "away": qa}
            r["kmid"] = Devig(qh["mid"], qa["mid"])
            if r["kmid"] is not None:
                n_ok += 1
            else:
                missing += 1
        coverage[int(s)] = {"n": len(rows), "n_with_quote": n_ok, "n_missing": missing}
    return coverage


def ApplyKalshi(rows_by_season, cfg):
    """Walk-forward the frozen config with Kalshi T-60 mid as the offset when present."""
    check = AttachKalshiT60(rows_by_season, seasons=(2025, 2026))
    for s in (2025, 2026):
        for r in rows_by_season.get(s) or []:
            if r.get("kmid") is not None:
                r["kalshi_logit"] = Logit(r["kmid"])
                r["kalshi_p"] = r["kmid"]
    # temporarily use kalshi logit as the close offset by swapping into close_logit copy
    names = FeatureNames(cfg["groups"])
    recs = []
    for s in (2025, 2026):
        train = [r for t, rs in rows_by_season.items() if t < s for r in rs]
        test = [r for r in (rows_by_season.get(s) or []) if r.get("kalshi_logit") is not None and r.get("home_win") is not None]
        if int(s) == 2026:
            test = [r for r in test if int(r.get("week") or 0) <= 10]
        Xtr, ytr, otr, _ = SeasonMatrix(train, names, "close_logit")
        Xte, yte, ote, keep = [], [], [], []
        for r in test:
            Xte.append([float(r.get(n) or 0.0) for n in names])
            yte.append(float(r["home_win"]))
            ote.append(float(r["kalshi_logit"]))
            keep.append(r)
        if len(ytr) < 50 or not keep:
            continue
        Xte, yte, ote = np.asarray(Xte, float), np.asarray(yte), np.asarray(ote)
        if cfg["standardize"] and names:
            Xtr, Xte, _, _ = Standardize(Xtr, Xte)
        elif not names:
            Xtr = np.zeros((len(ytr), 0)); Xte = np.zeros((len(keep), 0))
        w = FitOffsetLogit(Xtr, ytr, otr, l2=cfg["l2"])
        p = ApplyOffset(Xte, ote, w)
        for i, r in enumerate(keep):
            recs.append({
                "game_id": r["game_id"], "season": int(r["season"]), "gameday": r["gameday"],
                "week": r.get("week"), "home_team": r["home_team"], "away_team": r["away_team"],
                "home_win": int(yte[i]), "p": float(p[i]), "kmid": float(r["kmid"]),
                "kq": r.get("kq_t60"), "date": r["gameday"], "split": "contaminated" if s == 2025 else "forward",
            })
    return recs, check


def KalshiEdges(recs, fill_kind="mid"):
    edges = []
    for r in recs:
        qh, qa = (r.get("kq") or {}).get("home") or {}, (r.get("kq") or {}).get("away") or {}
        if not qh or not qa or qh.get(fill_kind if fill_kind == "ask" else "ask") is None:
            # mid may be computed
            pass
        def mid(q):
            if q.get("mid") is not None:
                return q["mid"]
            if q.get("bid") is not None and q.get("ask") is not None:
                return (q["bid"] + q["ask"]) / 2
            return None
        fill_h = qh.get("ask") if fill_kind == "ask" else mid(qh)
        fill_a = qa.get("ask") if fill_kind == "ask" else mid(qa)
        if fill_h is None or fill_a is None:
            continue
        # de-vig mid for qualify on mid path; ask path uses raw ask
        if fill_kind == "mid":
            tot = fill_h + fill_a
            mkt_h = fill_h / tot if tot else fill_h
            mkt_a = fill_a / tot if tot else fill_a
            fill_side_h, fill_side_a = fill_h, fill_a
        else:
            mkt_h, mkt_a = fill_h, fill_a
            fill_side_h, fill_side_a = fill_h, fill_a
        p_h, p_a = r["p"], 1.0 - r["p"]
        e_h, e_a = p_h - mkt_h - FEE, p_a - mkt_a - FEE
        if e_h >= e_a:
            side, p, fill, res, edge, mkt = r["home_team"], p_h, fill_side_h, r["home_win"], e_h, mkt_h
        else:
            side, p, fill, res, edge, mkt = r["away_team"], p_a, fill_side_a, 1 - r["home_win"], e_a, mkt_a
        qualifies = edge >= MIN_EDGE
        edges.append({
            "date": r["date"], "start_time": None, "sport": SPORT, "season": str(r["season"]),
            "split": r["split"], "event_id": r["game_id"],
            "event": f"{r['away_team']} @ {r['home_team']}", "side": side,
            "model": "residual_best", "price_basis": f"kalshi_T60_{fill_kind}",
            "model_prob": p, "market_prob_devig": mkt, "fill": fill, "edge": edge,
            "qualifies": qualifies, "result": res, "fair_value": round(p, 2),
            "_p": p, "_fill": fill, "_fav_fill": fill, "_fav_result": res,
        })
    return edges


def WidestRoi(bet_log, edges, rng, alpha=0.05):
    from Backtest.ThinMarkets import WidestRoiCi
    return WidestRoiCi(bet_log, edges, rng)


def BankAndMc(edges, days, rng, fee_kind="tim", n_paths=N_PATHS):
    out = {}
    dailies = {}
    for strat in ("flat100", "qkelly"):
        daily, blog, ruined, end = Walk(edges, strat, fee_kind=fee_kind, days=days)
        staked = sum(b["stake"] for b in blog)
        pnl = sum(b["pnl"] for b in blog)
        roi = None if not staked else pnl / staked
        ci = WidestRoi(blog, edges, rng)
        dailies[strat] = daily
        rets = DailyReturns(daily, start=START)
        paths = PathEvaluate(rets, seed=SEED, paths=n_paths, horizons={"window": max(1, len(rets))})
        out[strat] = {
            "n_bets": len(blog), "final_equity": end, "staked": round(staked, 2),
            "pnl": round(pnl, 2), "roi": roi, "ruined": bool(ruined or end < RUIN),
            "ci": ci, "pathsim": paths,
        }
    return out, dailies


def EnsureEff(first=FIRST, last=LAST):
    missing = [s for s in range(first, last + 1)
               if not os.path.exists(os.path.join("data", "backtest", f"team_game_eff_{s}.jsonl"))]
    if not missing:
        return []
    from DataPull.BuildTeamGameEff import Main as BuildMain
    return BuildMain(["--first", str(min(missing)), "--last", str(max(missing))])


def FmtPct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def FmtCi(ci):
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{FmtPct(ci[0])}, {FmtPct(ci[1])}]"


def FmtMoney(x):
    return "n/a" if x is None else f"${x:,.2f}"


def MdReport(meta):
    best = meta["best"]
    spa = meta["spa"]
    perm = meta["perm"]
    bank = meta.get("bank_mid_tim") or {}
    lines = [
        "# NFL residual-on-market walk-forward search (2026-11-13)",
        "",
        f"Made by American Footbot, {meta['written_at']}. Code `{meta['code']}`, run_id `{meta['run_id']}`. "
        "Simulation only; no orders, no real wagers.",
        "",
        "## Pre-registered setup",
        "- Model: `logit(p) = logit(market) + intercept + Xb`, walk-forward by season (fit on seasons < S).",
        "- Primary market: de-vigged sportsbook close (`book_close_devig`). Opening-line offset is a variant, "
        "run only when opening lines are on disk.",
        "- Features (season-to-date, earlier gamedays only): EPA/play, success rate, YPP, QB EPA, rest, travel, "
        "weather, injury counts. The open-to-close move is never a feature when predicting against the close.",
        f"- Grid: {meta['n_configs']} configs ({meta['grid_note']}).",
        f"- Tune seasons {TUNE[0]}-{TUNE[-1]} pick the winner. 2023-2025 are labelled contaminated. "
        "2026 is reported forward and was not used to pick.",
        f"- Prior search tally was at least {PRIOR_TALLY}. This search adds {meta['n_configs']} trials. "
        f"New lower-bound tally: **{meta['new_tally']}**.",
        f"- SPA: {N_SPA:,} day-block bootstraps over all close-offset configs on tune games. "
        f"Label permutation: {N_PERM:,} shuffles of outcomes on the frozen best (tune games).",
        "- Kalshi application uses the frozen *config* (not frozen weights), still walk-forward, with Kalshi T-60 "
        "as the offset. PRIMARY fill = mid + Tim's $0.02 fee estimate. SECONDARY = ask + $0.02. Value at ≥3c.",
        f"- Monte Carlo: {N_PATHS:,} PathSim paths on the Kalshi $10k bankroll (flat100 and qkelly, 2% cap).",
        "",
        "## Frozen best config (picked on 2015-2022 only)",
        f"- id: `{best['id']}`",
        f"- offset: {best['cfg']['offset']}, l2={best['cfg']['l2']}, standardize={best['cfg']['standardize']}",
        f"- feature groups: {', '.join(best['cfg']['groups']) or 'intercept only'}",
        f"- tune N={best['tune']['n']}, ign_rel_bits vs close = {best['tune']['ign_rel_bits']}",
        f"- 2023-2025 contaminated: N={best['contaminated']['n']}, ign_rel_bits = {best['contaminated']['ign_rel_bits']}",
        f"- 2026 forward: N={best['forward']['n']}, ign_rel_bits = {best['forward']['ign_rel_bits']}",
        "",
        "## SPA and permutation",
        f"- SPA p-value (tune, close-offset universe): **{spa.get('p')}** "
        f"(N configs={spa.get('n_configs')}, N games={spa.get('n_games')}, boots={spa.get('n_boot')}).",
        f"- Label-permutation p-value on the best config (tune): **{perm.get('p')}** "
        f"(N={perm.get('n')}, perms={perm.get('n_perm')}).",
        "",
        "## Verdict",
        meta["verdict"],
        "",
        "## Kalshi 2025-26 bankroll, mid + tim_est_0.02 (primary)",
    ]
    def bank_tbl(bundle, title):
        out = [f"### {title}", "",
               "| Strategy | N bets | End bank | P&L | ROI | Widest 95% CI | Ruin? |",
               "|---|---|---|---|---|---|---|"]
        for name in ("flat100", "qkelly"):
            b = bundle.get(name) or {}
            ci = ((b.get("ci") or {}).get("widest_95"))
            out.append(f"| {name} | {b.get('n_bets')} | {FmtMoney(b.get('final_equity'))} | "
                       f"{FmtMoney(b.get('pnl'))} | {FmtPct(b.get('roi'))} | {FmtCi(ci)} | "
                       f"{'yes' if b.get('ruined') else 'no'} |")
        return out
    lines += bank_tbl(bank, "tim_est_0.02, mid fill")
    if meta.get("bank_ask_tim"):
        lines += [""] + bank_tbl(meta["bank_ask_tim"], "tim_est_0.02, ask fill (secondary)")
    att = meta.get("kalshi_attach") or {}
    if att:
        bits = []
        for s in sorted(att):
            c = att[s]
            bits.append(f"{s}: {c.get('n_with_quote')}/{c.get('n')} games with a T-60 quote ({c.get('n_missing')} missing)")
        lines += ["", f"- Kalshi T-60 attach (existing moneyline candles, no full2 refit): {'; '.join(bits)}."]
    lines += [
        "",
        "## Open-line coverage",
        meta.get("open_note") or "No opening-line file was present; open-offset variants and CLV vs book open were not scored.",
        "",
        "## Caveats for Graphtard",
        "- Simulated. Walk-forward residuals, not a claimed live model.",
        f"- 2023-2025 are contaminated by at least {meta['new_tally']} searched settings after this run.",
        "- 2026 forward N is small. Weather uses recorded temp/wind (a mild forecast-vs-actual caveat).",
        "- Travel uses approximate stadium coordinates. Week-1 season-to-date features are zero.",
        "- `tim_est_0.02` is Tim's fee estimate, not a verified exchange fee.",
        "- Claim an edge only if the widest 95% lower bound is above 0 under both fee bases, and the SPA / permutation tests support it, on a serious sample.",
        f"- Workers={meta.get('workers')}, seed={SEED}.",
    ]
    if meta.get("pathsim_note"):
        lines += ["", "## PathSim", meta["pathsim_note"]]
    return "\n".join(lines) + "\n"


def Verdict(best, spa, perm, bank):
    parts = []
    ign_t = (best or {}).get("tune", {}).get("ign_rel_bits")
    ign_f = (best or {}).get("forward", {}).get("ign_rel_bits")
    p_spa, p_perm = spa.get("p"), perm.get("p")
    claim = False
    if ign_t is not None and ign_t >= 0:
        parts.append(f"On tune seasons the best residual is not better than the close (ign_rel_bits={ign_t:.4f}).")
    elif p_spa is not None and p_spa > 0.05:
        parts.append(f"The tune gap looks good at a point estimate, but the SPA p-value is {p_spa:.3f} "
                     "(not below 0.05), so the apparent edge does not survive a test of the full search.")
    elif p_perm is not None and p_perm > 0.05:
        parts.append(f"SPA is suggestive but the label-permutation p-value is {p_perm:.3f}.")
    else:
        parts.append("Tune-set tests (SPA and permutation) are significant at 5%. This is necessary, not sufficient.")
        # still require Kalshi CI to claim a betting edge
    flat = (bank or {}).get("flat100") or {}
    ci = (flat.get("ci") or {}).get("widest_95")
    if ci and ci[0] is not None and ci[0] > 0:
        parts.append("The Kalshi 2025-26 flat100 widest 95% CI is above 0 at Tim's $0.02 fee; see the fee-basis pair before claiming.")
    else:
        parts.append("No betting edge is claimed on the Kalshi sample.")
        claim = False
    if ign_f is not None:
        parts.append(f"2026 forward ign_rel_bits = {ign_f:.4f} (small N, not used to pick).")
    return " ".join(parts), claim


def WriteCharts(dailies, path, title):
    rows = []
    for series, daily in dailies:
        for d in daily:
            rows.append({"date": d["date"], "series": series, "equity": d["equity"],
                         "drawdown_pct": d.get("drawdown_pct") or 0.0, "season": d.get("season")})
    if not rows:
        return
    frame = FrameTidy(rows, date_key="date", equity_key="equity", series_key="series", mode="auto")
    PlotBankroll(frame, path=path, title=title, ylabel="Bankroll ($)",
                 hlines=[(START, "$10,000 start")], figsize=(11, 6.5))
    import matplotlib.pyplot as plt
    plt.close("all")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=r"E:\AgentDocs\American-Footbot\2026-11-13_residual-search")
    ap.add_argument("--base", default="2026-11-13_residual-search")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-eff", action="store_true")
    ap.add_argument("--spa-boot", type=int, default=N_SPA)
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    a = ap.parse_args()
    workers = a.workers or WorkerCount()
    run_id = NewRunId()
    rng = np.random.default_rng(SEED)
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.base)
    written = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p %Z")
    if not a.skip_eff:
        print("ensuring team-game efficiency tables...", flush=True)
        EnsureEff()
    print("loading rows and features...", flush=True)
    rbs = AllRows(2012, LAST)
    # BuildRows already ran inside AllRows? AllRows returns feature rows.
    # Confirm rest_diff exists; attach residual features.
    eff = LoadTeamEff()
    inj = LoadInjuries()
    opens = LoadOpenLines()
    AttachResidualFeatures(rbs, eff, inj, opens)
    has_open = sum(1 for s, rs in rbs.items() for r in rs if r.get("open_logit") is not None)
    include_open = has_open >= 100
    configs = ConfigGrid(include_open=include_open, smoke=a.smoke)
    print(f"grid={len(configs)} workers={workers} open_games={has_open}", flush=True)
    seasons = list(range(FIRST, LAST + 1))
    results = RunGrid(rbs, configs, seasons, workers=1 if a.smoke else workers)
    picked = PickBest(results)
    if not picked:
        raise SystemExit("no close-offset config produced a tune score")
    ign, cid, cfg, recs, tune_block = picked
    best = {
        "id": cid, "cfg": cfg,
        "tune": tune_block,
        "contaminated": ScoreBlock(recs, CONTAMINATED),
        "forward": ScoreBlock(recs, [2026]),
        "all": ScoreBlock(recs, list(range(FIRST, LAST + 1))),
    }
    print("best", cid, "tune ign", tune_block["ign_rel_bits"], flush=True)
    spa = SpaPValue(results, n_boot=a.spa_boot if not a.smoke else 50, seed=SEED)
    perm = PermutationP(recs, n_perm=a.perm if not a.smoke else 50, seed=SEED)
    print("spa", spa.get("p"), "perm", perm.get("p"), flush=True)

    # Kalshi apply
    kalshi_recs, kcheck = ApplyKalshi(rbs, cfg)
    days = AllGameDays(rbs, seasons=(2025, 2026))
    mid_edges = KalshiEdges(kalshi_recs, "mid")
    ask_edges = KalshiEdges(kalshi_recs, "ask")
    n_paths = 50 if a.smoke else a.paths
    bank_mid, daily_mid = BankAndMc(mid_edges, days, rng, "tim", n_paths=n_paths)
    bank_mid_k, _ = BankAndMc(mid_edges, days, rng, "kalshi", n_paths=n_paths)
    bank_ask, daily_ask = BankAndMc(ask_edges, days, rng, "tim", n_paths=n_paths)
    bank_ask_k, _ = BankAndMc(ask_edges, days, rng, "kalshi", n_paths=n_paths)

    n_cfg = len(configs)
    new_tally = PRIOR_TALLY + n_cfg
    open_note = (f"Opening lines were found for {has_open} games; open-offset configs were scored for CLV "
                 "but did not enter the close-target freeze." if include_open else
                 "No usable opening-line file was found under data/backtest/open_lines*.jsonl, so the "
                 "open-offset variant and book CLV were not scored. The close-offset grid still has "
                 f"{sum(1 for c in configs if c['offset']=='close')} configs.")
    verdict, claim = Verdict(best, spa, perm, bank_mid)
    # PathSim headline from flat100 mid
    ps = ((bank_mid.get("flat100") or {}).get("pathsim") or {}).get("randomized") or {}
    path_note = None
    if ps:
        # pick window|block|absorbing if present
        key = next((k for k in ps if "block" in k and "absorbing" in k), next(iter(ps), None))
        if key:
            s = ps[key]
            path_note = f"`{key}`: {json.dumps({k:s[k] for k in s if k in ('p_ruin','p_target','q05','q50','q95','n_paths')}, default=str)}"

    meta = {
        "run_id": run_id, "code": CodeVersion(), "model_version": ModelVersion(),
        "written_at": written, "workers": workers, "n_configs": n_cfg,
        "grid_note": "8 feature groups on/off x 5 l2 x 2 standardize, close offset"
                     + ("; same grid also at open offset" if include_open else ""),
        "new_tally": new_tally, "prior_tally": PRIOR_TALLY,
        "best": best, "spa": spa, "perm": perm,
        "bank_mid_tim": _strip_ci(bank_mid), "bank_mid_kalshi": _strip_ci(bank_mid_k),
        "bank_ask_tim": _strip_ci(bank_ask), "bank_ask_kalshi": _strip_ci(bank_ask_k),
        "open_note": open_note, "verdict": verdict, "claim_edge": claim,
        "n_kalshi_games": len(kalshi_recs), "kalshi_attach": kcheck,
        "pathsim_note": path_note, "headline": verdict,
        "contamination_tally_at_least": new_tally,
    }
    # tidy CSV: per-season ign for the best config + bankroll
    last = days[-1] if days else written[:10]
    tidy = []
    for recs_s, split, season in (
        ( [r for r in recs if int(r["season"]) in TUNE], "tune", "2015-2022"),
        ( [r for r in recs if int(r["season"]) in CONTAMINATED], "contaminated", "2023-2025"),
        ( [r for r in recs if int(r["season"]) == 2026], "forward", "2026"),
    ):
        tidy.append({
            "date": last, "sport": SPORT, "series": f"nfl_residual_{split}",
            "season": season, "split": split, "n": len(recs_s),
            "ign_rel_bits": IgnRel(recs_s),
        })
    WriteCsv(base + ".csv", ["date", "sport", "series", "season", "split", "n", "ign_rel_bits"], tidy)
    bank_rows = []
    for series, daily in (("flat100 / mid", daily_mid.get("flat100") or []),
                          ("qkelly / mid", daily_mid.get("qkelly") or []),
                          ("flat100 / ask", daily_ask.get("flat100") or []),
                          ("qkelly / ask", daily_ask.get("qkelly") or [])):
        for d in daily:
            bank_rows.append({**d, "series": series})
    WriteCsv(base + "_bankroll.csv", list(BANK_FIELDS), bank_rows)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=_json)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdReport(meta))
    WriteCharts(
        [("flat100 / mid", daily_mid.get("flat100") or []),
         ("qkelly / mid", daily_mid.get("qkelly") or []),
         ("flat100 / ask", daily_ask.get("flat100") or [])],
        base + "_bankroll_draft.png",
        "NFL $10k residual-best at Kalshi T-60, tim_est_0.02",
    )
    # also write a compact leaderboard of top 15 tune configs
    board = []
    for cid2, cfg2, recs2 in results:
        if cfg2["offset"] != "close":
            continue
        b = ScoreBlock(recs2, TUNE)
        board.append({"id": cid2, "ign_rel_bits_tune": b["ign_rel_bits"], "n": b["n"],
                      "groups": ",".join(cfg2["groups"]), "l2": cfg2["l2"], "standardize": cfg2["standardize"]})
    board.sort(key=lambda r: (r["ign_rel_bits_tune"] is None, r["ign_rel_bits_tune"] if r["ign_rel_bits_tune"] is not None else 9))
    WriteCsv(base + "_leaderboard.csv", ["id", "ign_rel_bits_tune", "n", "groups", "l2", "standardize"], board[:50])
    print(json.dumps({"run_id": run_id, "best": cid, "tune_ign": tune_block["ign_rel_bits"],
                      "spa_p": spa.get("p"), "perm_p": perm.get("p"), "n_configs": n_cfg,
                      "new_tally": new_tally, "out": a.out_dir}, indent=1, default=_json), flush=True)


def _strip_ci(bank):
    out = {}
    for k, v in (bank or {}).items():
        d = dict(v)
        if "ci" in d:
            d["ci"] = {kk: (None if vv is None or (isinstance(vv, tuple) and vv[0] is None) else list(vv) if isinstance(vv, tuple) else vv)
                       for kk, vv in d["ci"].items()}
        if "pathsim" in d:
            d["pathsim"] = _jsonable(d["pathsim"])
        out[k] = d
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def _json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    Main()
