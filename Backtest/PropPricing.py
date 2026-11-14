r"""
NFL player-prop pricing study on Kalshi (+ selective moneylines). Simulation only, no orders.

    .venv\Scripts\python.exe -m Backtest.PropPricing --out-dir E:\AgentDocs\American-Footbot\2026-11-14_prop-pricing

PRE-REGISTERED (written 2026-11-14 ~4:20 PM ET, before any holdout result was computed):
  Markets (Kalshi, public API, pulled by DataPull.PullKalshiProps; pregame-listed markets only):
    td1 = KXNFLTD "1+" (anytime TD), td2p = KXNFLTD "2+/3+/4+", rec = KXNFLREC (X+ receptions),
    recyds = KXNFLRECYDS, rshyds = KXNFLRSHYDS, passyds = KXNFLPASSYDS (X+ yards ladders).
  Fit / holdout:
    rec, recyds, rshyds, passyds: fit = 2025 season (Kalshi listed these from Oct 2025), holdout = 2026 season.
    td1, td2p: KXNFLTD only exists from the 2026 season, so fit = 2026 games before 2026-10-08 and
    holdout = 2026 games from 2026-10-08. The date is the operative rule (it falls between nflverse weeks 4 and 5;
    an earlier draft of this note said 'weeks 1-5'). Not a full-season holdout; stated in the report.
  Projection (leak-free: only nflverse player_week rows from games strictly before this game):
    last 16 games played, weights 0.85^age; p(X >= k) = smoothed weighted empirical exceedance, mixed with
    2 pseudo-games of a parametric fit (Poisson for TDs/receptions, Normal for yards). Needs >= 3 prior games.
  Models: proj (raw); recal (Platt on logit(proj), per kind, fit period); resid (offset logistic
    logit(p) = logit(mid_T60) + a + b*(logit(proj) - logit(mid_T60)), per kind, fit period).
  Grid: kind (6) x side (yes, no) x model (3) x min model edge (0, .03, .05, .08) x entry (T-60, T-15)
    x price band on the side's fill (all, <.35, .35-.65, >.65) = 1,152 prop configs.
  Moneylines: p_full2_kmid_T60 (walk-forward, from 2026-11-13_bankroll-10k edges) at Kalshi T-60 mid,
    2025 fit / 2026 holdout (weeks 1-9): side filter (any, fav, dog) x min edge (0, .03, .05) = 9 configs.
  Total 1,161 configs added to the trial tally (prior lower bound 8,619 -> 9,780).
  Fill: mid (Tim's primary rule). Ask (taker) reported for the frozen config as a secondary, not searched.
  Model edge = model_prob - (fill + 0.02), 0.02 = Tim's fee estimate (tim_est_0.02).
  Realized edge per $1 contract = result - fill - fee; also with Kalshi taker fee 0.07*p*(1-p).
  Selection: best fit-period mean realized edge (tim fee), n_fit >= 30; freeze; score holdout once.
  Multiple testing: White RC + Hansen SPA (10,000 day-block bootstraps, 4-day blocks) on the holdout
  (and the fit) across all 1,161 configs; Deflated Sharpe with N = 1,161.
  Holdout CIs on mean edge: day-block, game-cluster, exact binomial (Clopper-Pearson mapped); widest used.
  Claim an edge only if the widest 95% lower bound > 0 under both fee bases.
  Bankroll: one continuous $10k over the holdout, flat100 and qkelly (2% cap), 10,000-path PathSim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from Backtest.CommonFormat import ClopperPearson, FEE, FEE_BASIS, START, WriteCsv
from Backtest.KalshiReality import KalshiFee
from Backtest.SnapshotPrices import ALIASES
from Backtest.ThinMarketRules import BankAndMc, DeflatedSharpe
from Backtest.ThinMarkets import WriteBankChart
from Backtest.Version import CodeVersion, ModelVersion, NewRunId

SEED = 20261114
N_SPA = 10_000
N_PATHS = 10_000
BLOCK = 4
PRIOR_TALLY = 8619
MIN_FIT_N = 30
PLAY_EDGE = 0.03
SERIES_KIND = {"KXNFLREC": "rec", "KXNFLRECYDS": "recyds", "KXNFLRSHYDS": "rshyds", "KXNFLPASSYDS": "passyds"}
KINDS = ("td1", "td2p", "rec", "recyds", "rshyds", "passyds")
STAT_OF = {"td1": "tds", "td2p": "tds", "rec": "receptions", "recyds": "receiving_yards",
           "rshyds": "rushing_yards", "passyds": "passing_yards"}
MODELS = ("proj", "recal", "resid")
MIN_EDGES = (0.0, 0.03, 0.05, 0.08)
TIMES = (60, 15)
BANDS = ("all", "lt35", "35to65", "gt65")
SIDES = ("yes", "no")
ML_SIDES = ("any", "fav", "dog")
ML_EDGES = (0.0, 0.03, 0.05)
TD_SPLIT = "2026-10-08"
STALE_S = 180 * 60
DECAY = 0.85
LOOKBACK = 16
PSEUDO = 2.0
ML_EDGES_CSV = os.path.join("E:\\AgentDocs", "American-Footbot", "2026-11-13_bankroll-10k", "2026-11-13_bankroll-10k_edges.csv")


# ---------------------------------------------------------------- helpers
def IterJsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def NormName(s):
    s = (s or "").lower().replace(".", "").replace("'", "").replace("-", " ").replace(",", " ")
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def Clip(p, lo=0.005, hi=0.995):
    return min(hi, max(lo, p))


def Logit(p):
    p = Clip(p)
    return math.log(p / (1 - p))


def Sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def NormCdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def PoissonSf(k, lam):
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    lam = max(lam, 1e-6)
    term, cdf = math.exp(-lam), 0.0
    for i in range(int(k)):
        if i > 0:
            term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def KindOf(series, thr):
    if series == "KXNFLTD":
        return "td1" if thr is not None and thr <= 1 else "td2p"
    return SERIES_KIND.get(series)


def Valid(c):
    b, a = c.get("yes_bid_close"), c.get("yes_ask_close")
    return b is not None and a is not None and 0 < b < a < 1


def QuoteAt(candles, target_ts, stale_s=STALE_S):
    best = None
    for c in candles:
        if c["end_ts"] > target_ts:
            break
        if Valid(c):
            best = c
    if best is None or target_ts - best["end_ts"] > stale_s:
        return None
    b, a = best["yes_bid_close"], best["yes_ask_close"]
    return {"bid": b, "ask": a, "mid": (a + b) / 2, "width": a - b, "age_min": round((target_ts - best["end_ts"]) / 60, 1)}


# ---------------------------------------------------------------- player history
class Players:
    def __init__(self, seasons=(2023, 2024, 2025, 2026)):
        self.hist = defaultdict(list)       # pid -> [(season, week, statdict)]
        self.by_team = {}                   # (season, team, norm) -> pid
        self.by_name = defaultdict(set)     # (season, norm) -> pids
        self.pos = {}
        for s in seasons:
            for r in IterJsonl(os.path.join("data", "playbyplay", f"player_week_{s}.jsonl")):
                pid = r.get("player_id")
                if not pid:
                    continue
                st = {"tds": (r.get("rushing_tds") or 0) + (r.get("receiving_tds") or 0),
                      "receptions": r.get("receptions") or 0, "receiving_yards": r.get("receiving_yards") or 0,
                      "rushing_yards": r.get("rushing_yards") or 0, "passing_yards": r.get("passing_yards") or 0,
                      "attempts": r.get("attempts") or 0, "carries": r.get("carries") or 0,
                      "targets": r.get("targets") or 0}
                self.hist[pid].append((int(r["season"]), int(r["week"]), st))
                nm = NormName(r.get("player_display_name") or r.get("player_name"))
                self.by_team[(s, r.get("team"), nm)] = pid
                self.by_name[(s, nm)].add(pid)
                self.pos[pid] = r.get("position") or r.get("position_group")
        for pid in self.hist:
            self.hist[pid].sort(key=lambda t: (t[0], t[1]))
        self._cache = {}

    def Match(self, season, name, team_codes):
        nm = NormName(name)
        for t in team_codes:
            pid = self.by_team.get((season, t, nm))
            if pid:
                return pid
        ids = self.by_name.get((season, nm)) or set()
        if len(ids) == 1:
            return next(iter(ids))
        # fallback: previous season (traded players change team)
        ids = self.by_name.get((season - 1, nm)) or set()
        if len(ids) == 1:
            return next(iter(ids))
        return None

    def Prior(self, pid, season, week):
        key = (pid, season, week)
        if key not in self._cache:
            rows = [st for (s, w, st) in self.hist.get(pid, []) if (s, w) < (season, week)]
            self._cache[key] = rows[-LOOKBACK:]
        return self._cache[key]


def Project(prior, kind, thr):
    """Leak-free P(stat >= thr) from prior games (pre-registered recipe)."""
    if len(prior) < 3 or thr is None:
        return None
    stat = STAT_OF[kind]
    xs = np.array([g[stat] for g in prior], dtype=float)
    w = DECAY ** np.arange(len(xs))[::-1]
    W = w.sum()
    emp = float((w * (xs >= thr)).sum() / W)
    m = float((w * xs).sum() / W)
    if kind in ("td1", "td2p", "rec"):
        par = PoissonSf(int(math.ceil(thr)), m)
    else:
        var = float((w * (xs - m) ** 2).sum() / W)
        sd = max(math.sqrt(var), 0.35 * m + 5.0)
        par = 1.0 - NormCdf((thr - 0.5 - m) / sd)
    return Clip((W * emp + PSEUDO * par) / (W + PSEUDO))


# ---------------------------------------------------------------- records
def TeamCodes(seg, g):
    out = []
    for t in (g["home_team"], g["away_team"]):
        for al in ALIASES.get(t, [t]):
            if seg.startswith(al):
                out.append((len(al), t))
    out.sort(reverse=True)
    return [t for _, t in out] or [g["home_team"], g["away_team"]]


def LoadRecords(series_list, players, games_by_id):
    recs, cov = [], defaultdict(lambda: defaultdict(int))
    for series in series_list:
        mk = {m["ticker"]: m for m in IterJsonl(os.path.join("data", "kalshi", f"props_markets_{series}.jsonl"))}
        for m in mk.values():
            cov[series]["markets_listed"] += 1
            if m.get("game_id"):
                cov[series]["markets_matched_game"] += 1
            if m.get("pregame_listed"):
                cov[series]["markets_pregame_listed"] += 1
        seen = set()
        for c in IterJsonl(os.path.join("data", "kalshi", f"props_candles_{series}.jsonl")):
            if c["ticker"] in seen:
                continue
            seen.add(c["ticker"])
            m = mk.get(c["ticker"])
            # historical-tier (2025) ladders were sampled to the middle 2 strikes per player; keep only those
            if not m or not m.get("game_id") or not m.get("hist_sampled", True):
                continue
            g = games_by_id.get(m["game_id"])
            if not g:
                continue
            cov[series]["candle_rows"] += 1
            res = {"yes": 1, "no": 0}.get(m.get("result"))
            if res is None:
                cov[series]["void_or_scalar"] += 1
                continue
            thr = m.get("threshold")
            if thr is None and m.get("floor_strike") is not None:
                thr = float(m["floor_strike"]) + 0.5
            kind = KindOf(series, thr)
            ko = datetime.fromisoformat(c["kickoff_utc"].replace("Z", "+00:00")).timestamp()
            cand = sorted((c.get("hourly") or []) + (c.get("minute") or []), key=lambda x: x["end_ts"])
            q = {t: QuoteAt(cand, ko - t * 60) for t in TIMES}
            q["close"] = QuoteAt(cand, ko + 60, stale_s=STALE_S)
            hv = [x for x in sorted(c.get("hourly") or [], key=lambda x: x["end_ts"]) if Valid(x) and x["end_ts"] <= ko]
            mv = [x for x in sorted(c.get("minute") or [], key=lambda x: x["end_ts"]) if Valid(x) and x["end_ts"] <= ko]
            first = hv[0] if hv else (mv[0] if mv else None)
            q["open"] = None if first is None else {"bid": first["yes_bid_close"], "ask": first["yes_ask_close"],
                                                    "mid": (first["yes_bid_close"] + first["yes_ask_close"]) / 2,
                                                    "width": first["yes_ask_close"] - first["yes_bid_close"],
                                                    "hours_before": round((ko - first["end_ts"]) / 3600, 2)}
            season, week = int(g["season"]), int(g["week"])
            seg = m.get("player_seg") or ""
            pid = players.Match(season, m.get("player_name"), TeamCodes(seg, g))
            proj = Project(players.Prior(pid, season, week), kind, thr) if pid else None
            if kind in ("td1", "td2p"):
                split = "fit" if g["gameday"] < TD_SPLIT else "hold"
            else:
                split = "fit" if season == 2025 else "hold"
            cov[series]["usable"] += 1
            if pid:
                cov[series]["player_matched"] += 1
            if proj is not None:
                cov[series]["with_projection"] += 1
            recs.append({"ticker": c["ticker"], "series": series, "kind": kind, "thr": thr, "season": season,
                         "week": week, "date": g["gameday"], "game_id": g["game_id"], "split": split,
                         "player": m.get("player_name"), "pid": pid, "pos": players.pos.get(pid) if pid else None,
                         "result": res, "q": q, "proj": proj, "volume": float(m.get("volume") or 0)})
    return recs, {k: dict(v) for k, v in cov.items()}


# ---------------------------------------------------------------- models
def FitLogit(X, y, offset=None, l2=0.5, iters=50):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    w = np.zeros(X.shape[1])
    if offset is None:
        w[1] = 1.0
    for _ in range(iters):
        z = X @ w + off
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:] - (1.0 if offset is None else 0.0)]
        H = (X.T * (p * (1 - p))) @ X + l2 * np.diag(np.r_[0, np.ones(X.shape[1] - 1)])
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def FitModels(recs):
    fits = {}
    for k in KINDS:
        rs = [r for r in recs if r["kind"] == k and r["split"] == "fit" and r["proj"] is not None and r["q"][60]]
        if len(rs) < 50:
            fits[k] = {"n": len(rs), "recal": None, "resid": None}
            continue
        y = [r["result"] for r in rs]
        lp = [Logit(r["proj"]) for r in rs]
        lm = [Logit(r["q"][60]["mid"]) for r in rs]
        recal = FitLogit(np.array(lp)[:, None], y)
        resid = FitLogit((np.array(lp) - np.array(lm))[:, None], y, offset=lm)
        fits[k] = {"n": len(rs), "recal": recal.tolist(), "resid": resid.tolist()}
    return fits


def ModelP(r, model, t, fits):
    if r["proj"] is None:
        return None
    if model == "proj":
        return r["proj"]
    f = fits.get(r["kind"]) or {}
    w = f.get(model)
    if w is None:
        return None
    lp = Logit(r["proj"])
    if model == "recal":
        return Clip(Sig(w[0] + w[1] * lp))
    q = r["q"].get(t)
    if not q:
        return None
    lm = Logit(q["mid"])
    return Clip(Sig(lm + w[0] + w[1] * (lp - lm)))


# ---------------------------------------------------------------- moneylines
def LoadMl(path=ML_EDGES_CSV):
    import csv
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != "p_full2_kmid_T60":
                continue
            p, fill, res = float(row["model_prob"]), float(row["fill_price"]), int(row["result"])
            home, away = row["event"].split(" @ ")[1], row["event"].split(" @ ")[0]
            other = away if row["side"] == home else home
            season = int(row["season"])
            for side, ps, fs, rs in ((row["side"], p, fill, res), (other, 1 - p, 1 - fill, 1 - res)):
                out.append({"kind": "ml", "date": row["date"], "game_id": row["event_id"], "side": side,
                            "season": season, "split": "fit" if season == 2025 else "hold",
                            "p": ps, "fill": fs, "result": rs})
    return out


# ---------------------------------------------------------------- grid
def PropConfigs():
    out = []
    for k in KINDS:
        for s in SIDES:
            for m in MODELS:
                for e in MIN_EDGES:
                    for t in TIMES:
                        for b in BANDS:
                            out.append({"family": "prop", "kind": k, "side": s, "model": m, "min_edge": e,
                                        "tmin": t, "band": b})
    return out


def MlConfigs():
    return [{"family": "ml", "kind": "ml", "side": s, "model": "p_full2_kmid_T60", "min_edge": e, "tmin": 60, "band": "all"}
            for s in ML_SIDES for e in ML_EDGES]


def Cid(c):
    return f"{c['family']}|{c['kind']}|{c['side']}|{c['model']}|e{c['min_edge']:.2f}|T{c['tmin']}|{c['band']}"


def InBand(f, band):
    return band == "all" or (band == "lt35" and f < 0.35) or (band == "35to65" and 0.35 <= f <= 0.65) or (band == "gt65" and f > 0.65)


def BetsFor(cfg, recs, ml, fits, fill_kind="mid"):
    bets = []
    if cfg["family"] == "ml":
        for r in ml:
            f = r["fill"]
            if cfg["side"] == "fav" and f < 0.5:
                continue
            if cfg["side"] == "dog" and f >= 0.5:
                continue
            if r["p"] - f - FEE < cfg["min_edge"]:
                continue
            bets.append({"date": r["date"], "event_id": r["game_id"], "side": r["side"], "split": r["split"],
                         "fill": f, "p": r["p"], "result": r["result"], "ticker": f"ML:{r['game_id']}:{r['side']}"})
        best = {}
        for b in bets:
            k = b["event_id"]
            if k not in best or b["p"] - b["fill"] > best[k]["p"] - best[k]["fill"]:
                best[k] = b
        return list(best.values())
    for r in recs:
        if r["kind"] != cfg["kind"]:
            continue
        q = r["q"].get(cfg["tmin"])
        if not q:
            continue
        p = ModelP(r, cfg["model"], cfg["tmin"], fits)
        if p is None:
            continue
        if cfg["side"] == "yes":
            f = q["mid"] if fill_kind == "mid" else q["ask"]
            ps, res = p, r["result"]
        else:
            f = (1 - q["mid"]) if fill_kind == "mid" else (1 - q["bid"])
            ps, res = 1 - p, 1 - r["result"]
        if not (0 < f < 1) or not InBand(f, cfg["band"]):
            continue
        if ps - f - FEE < cfg["min_edge"]:
            continue
        bets.append({"date": r["date"], "event_id": r["game_id"], "side": cfg["side"], "split": r["split"],
                     "fill": f, "p": ps, "result": res, "ticker": r["ticker"], "kind": r["kind"],
                     "player": r["player"], "thr": r["thr"], "pos": r["pos"]})
    return bets


def EdgeOf(b, fee_kind="tim"):
    return b["result"] - b["fill"] - (KalshiFee(b["fill"]) if fee_kind == "kalshi" else FEE)


def Score(bets):
    out = {}
    for sp in ("fit", "hold"):
        bs = [b for b in bets if b["split"] == sp]
        out[f"n_{sp}"] = len(bs)
        out[f"edge_{sp}"] = float(np.mean([EdgeOf(b) for b in bs])) if bs else None
        out[f"edge_{sp}_kalshi"] = float(np.mean([EdgeOf(b, "kalshi") for b in bs])) if bs else None
        out[f"hit_{sp}"] = float(np.mean([b["result"] for b in bs])) if bs else None
        out[f"fill_{sp}"] = float(np.mean([b["fill"] for b in bs])) if bs else None
    return out


def DailyMatrix(all_bets, split, days):
    idx = {d: i for i, d in enumerate(days)}
    D = np.zeros((len(all_bets), len(days)))
    for k, bets in enumerate(all_bets):
        by = defaultdict(list)
        for b in bets:
            if b["split"] == split and b["date"] in idx:
                by[b["date"]].append(EdgeOf(b))
        for d, xs in by.items():
            D[k, idx[d]] = float(np.mean(xs))
    return D


def SpaRc(D, n_boot=N_SPA, block=BLOCK, seed=SEED):
    if D.size == 0:
        return {"rc_p": None, "spa_p": None}
    n_days = D.shape[1]
    mean = D.mean(1)
    obs = float(mean.max())
    blocks = [list(range(i, min(i + block, n_days))) for i in range(0, n_days, block)]
    rng = np.random.default_rng(seed)
    hits_rc = hits_spa = 0
    c_spa = np.minimum(mean, 0.0)
    for _ in range(n_boot):
        ch = rng.integers(0, len(blocks), size=len(blocks))
        cols = np.array([d for b in ch for d in blocks[b]][:n_days])
        bm = D[:, cols].mean(1)
        if (bm - mean).max() >= obs - 1e-15:
            hits_rc += 1
        if (bm - c_spa).max() >= obs - 1e-15:
            hits_spa += 1
    return {"rc_p": hits_rc / n_boot, "spa_p": hits_spa / n_boot, "observed_max_mean_daily_edge": obs,
            "argmax": int(mean.argmax()), "n_days": n_days, "n_configs": int(D.shape[0]), "n_boot": n_boot, "block": block}


def EdgeCis(bets, fee_kind, rng, boot=4000):
    if not bets:
        return {}
    e = np.array([EdgeOf(b, fee_kind) for b in bets])
    n = len(e)
    out = {"n": n, "mean_edge": float(e.mean()), "sd": float(e.std(ddof=1)) if n > 1 else None}
    days = sorted({b["date"] for b in bets})
    di = {d: i // BLOCK for i, d in enumerate(days)}
    day_groups = defaultdict(list)
    game_groups = defaultdict(list)
    for x, b in zip(e, bets):
        day_groups[di[b["date"]]].append(x)
        game_groups[b["event_id"]].append(x)
    for name, grp in (("day_block_95", day_groups), ("game_cluster_95", game_groups)):
        sums = np.array([sum(v) for v in grp.values()])
        cnts = np.array([len(v) for v in grp.values()])
        G = len(sums)
        ix = rng.integers(0, G, size=(boot, G))
        means = sums[ix].sum(1) / cnts[ix].sum(1)
        out[name] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    k = int(sum(b["result"] for b in bets))
    lo, hi = ClopperPearson(k, n, alpha=0.05)
    fee_mean = float(np.mean([(KalshiFee(b["fill"]) if fee_kind == "kalshi" else FEE) for b in bets]))
    fill_mean = float(np.mean([b["fill"] for b in bets]))
    out["binomial_95"] = [lo - fill_mean - fee_mean, hi - fill_mean - fee_mean]
    widest = None
    for kname in ("day_block_95", "game_cluster_95", "binomial_95"):
        ci = out[kname]
        if widest is None or (ci[1] - ci[0]) > (widest[1] - widest[0]):
            widest = ci
            out["widest_name"] = kname
    out["widest_95"] = widest
    if out["sd"]:
        half = (widest[1] - widest[0]) / 2
        se_naive = out["sd"] / math.sqrt(n)
        deff = (half / (1.96 * se_naive)) ** 2 if se_naive > 0 else 1.0
        se = se_naive * math.sqrt(max(deff, 1.0))
        out["design_effect"] = deff
        out["mde_80pct_power"] = 2.8 * se
        out["power_at_3c"] = NormCdf(0.03 / se - 1.96)
    return out


def WalkEdges(bets, price_basis):
    out = []
    for b in bets:
        out.append({"date": b["date"], "season": "2026", "split": "forward", "price_basis": price_basis,
                    "event_id": b["event_id"], "side": b["side"], "qualifies": True, "_fill": b["fill"],
                    "result": int(b["result"]), "_p": b["p"], "_fav_fill": b["fill"], "_fav_result": int(b["result"])})
    out.sort(key=lambda x: x["date"])
    return out


# ---------------------------------------------------------------- pricing analyses
def Agg(rs, fn):
    xs = [fn(r) for r in rs]
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)) if xs else None), len(xs)


def PricingTables(recs):
    rows, cal = [], []
    for k in KINDS:
        for season in (2025, 2026, "all"):
            rs = [r for r in recs if r["kind"] == k and (season == "all" or r["season"] == season)]
            if not rs:
                continue

            def w(t):
                return lambda r: r["q"][t]["width"] if r["q"].get(t) else None

            def brier(t):
                return lambda r: (r["q"][t]["mid"] - r["result"]) ** 2 if r["q"].get(t) else None
            both = [r for r in rs if r["q"].get("open") and r["q"].get("close")]
            mv = [abs(r["q"]["close"]["mid"] - r["q"]["open"]["mid"]) for r in both]
            d_imp = [(r["q"]["open"]["mid"] - r["result"]) ** 2 - (r["q"]["close"]["mid"] - r["result"]) ** 2 for r in both]
            w60 = [r["q"][60]["width"] for r in rs if r["q"].get(60)]
            row = {"kind": k, "season": season, "n_markets": len(rs), "n_games": len({r["game_id"] for r in rs}),
                   "n_players": len({(r["player"], r["season"]) for r in rs}),
                   "n_with_T60": len(w60), "median_width_T60": float(np.median(w60)) if w60 else None,
                   "mean_width_T60": Agg(rs, w(60))[0], "mean_width_open": Agg(rs, w("open"))[0],
                   "mean_width_close": Agg(rs, w("close"))[0],
                   "mean_hours_listed_before_ko": Agg(rs, lambda r: r["q"]["open"]["hours_before"] if r["q"].get("open") else None)[0],
                   "mean_abs_move_open_close": float(np.mean(mv)) if mv else None,
                   "share_moved_ge_3c": float(np.mean([x >= 0.03 for x in mv])) if mv else None,
                   "brier_open": Agg(both, brier("open"))[0], "brier_T60": Agg(rs, brier(60))[0],
                   "brier_close": Agg(both, brier("close"))[0],
                   "close_minus_open_brier_gain": float(np.mean(d_imp)) if d_imp else None,
                   "mean_mid_T60": Agg(rs, lambda r: r["q"][60]["mid"] if r["q"].get(60) else None)[0],
                   "yes_rate": float(np.mean([r["result"] for r in rs])),
                   "median_volume": float(np.median([r["volume"] for r in rs]))}
            lo = [r["result"] - r["q"][60]["mid"] for r in rs if r["q"].get(60) and r["q"][60]["mid"] < 0.2]
            hi = [r["result"] - r["q"][60]["mid"] for r in rs if r["q"].get(60) and r["q"][60]["mid"] > 0.8]
            row["longshot_lt20_result_minus_mid"] = float(np.mean(lo)) if lo else None
            row["fav_gt80_result_minus_mid"] = float(np.mean(hi)) if hi else None
            row["n_lt20"], row["n_gt80"] = len(lo), len(hi)
            rows.append(row)
        rs = [r for r in recs if r["kind"] == k and r["q"].get(60)]
        for i in range(10):
            lo_b, hi_b = i / 10, (i + 1) / 10
            b = [r for r in rs if lo_b <= r["q"][60]["mid"] < hi_b]
            if b:
                cal.append({"kind": k, "bin_lo": lo_b, "bin_hi": hi_b, "n": len(b),
                            "mean_mid": float(np.mean([r["q"][60]["mid"] for r in b])),
                            "obs_rate": float(np.mean([r["result"] for r in b])),
                            "mean_proj": Agg(b, lambda r: r["proj"])[0]})
    td = defaultdict(list)
    for r in recs:
        if r["kind"] == "td1" and r["q"].get(60):
            td[r["game_id"]].append(r)
    sums = [(sum(r["q"][60]["mid"] for r in v), sum(r["result"] for r in v), len(v)) for v in td.values() if len(v) >= 10]
    ladder = {"n_games": len(sums),
              "mean_sum_mid_T60": float(np.mean([s[0] for s in sums])) if sums else None,
              "mean_realized_scorers": float(np.mean([s[1] for s in sums])) if sums else None,
              "mean_listed_players": float(np.mean([s[2] for s in sums])) if sums else None}
    if sums and ladder["mean_realized_scorers"]:
        ladder["implied_over_realized"] = ladder["mean_sum_mid_T60"] / ladder["mean_realized_scorers"]
    return rows, cal, ladder


def Disagreement(recs):
    out = []

    def block(rs, dim, val, t=60):
        rs = [r for r in rs if r["proj"] is not None and r["q"].get(t)]
        if len(rs) < 20:
            return None
        d = np.array([r["proj"] - r["q"][t]["mid"] for r in rs])
        y = np.array([r["result"] for r in rs])
        pj = np.array([r["proj"] for r in rs])
        md = np.array([r["q"][t]["mid"] for r in rs])
        big = np.abs(d) >= 0.10
        return {"dim": dim, "value": str(val), "time": f"T-{t}" if isinstance(t, int) else t, "n": len(rs),
                "mean_proj_minus_mid": float(d.mean()), "mean_abs_diff": float(np.abs(d).mean()),
                "share_abs_diff_ge_10c": float(big.mean()),
                "brier_proj": float(((pj - y) ** 2).mean()), "brier_mid": float(((md - y) ** 2).mean()),
                "brier_proj_big_disagree": float(((pj[big] - y[big]) ** 2).mean()) if big.any() else None,
                "brier_mid_big_disagree": float(((md[big] - y[big]) ** 2).mean()) if big.any() else None,
                "yes_rate": float(y.mean())}
    for k in KINDS:
        rk = [r for r in recs if r["kind"] == k]
        for t in (60, 15, "open", "close"):
            b = block(rk, "kind_time", k, t)
            if b:
                out.append(b)
        for pos in sorted({r["pos"] for r in rk if r["pos"]}):
            b = block([r for r in rk if r["pos"] == pos], f"{k}_position", pos)
            if b:
                out.append(b)
        for th in sorted({r["thr"] for r in rk if r["thr"] is not None}):
            b = block([r for r in rk if r["thr"] == th], f"{k}_line", th)
            if b:
                out.append(b)
        for sp in ("fit", "hold"):
            b = block([r for r in rk if r["split"] == sp], f"{k}_split", sp)
            if b:
                out.append(b)
    return out


def Ignorance(recs, fits):
    """Log-loss gap (bits) model vs Kalshi T-60 mid on the holdout, by kind and model."""
    out = []
    for k in KINDS:
        rs = [r for r in recs if r["kind"] == k and r["split"] == "hold" and r["q"].get(60) and r["proj"] is not None]
        if not rs:
            continue
        y = np.array([r["result"] for r in rs])
        md = np.array([Clip(r["q"][60]["mid"]) for r in rs])
        ll_m = -(y * np.log2(md) + (1 - y) * np.log2(1 - md)).mean()
        for m in MODELS:
            ps = [ModelP(r, m, 60, fits) for r in rs]
            if any(p is None for p in ps):
                continue
            p = np.array(ps)
            ll = -(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean()
            out.append({"kind": k, "model": m, "n": len(rs), "logloss_bits_model": float(ll),
                        "logloss_bits_mid": float(ll_m), "ignorance_gap_bits": float(ll - ll_m),
                        "brier_model": float(((p - y) ** 2).mean()), "brier_mid": float(((md - y) ** 2).mean())})
    return out


def CalChart(cal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect")
    for k in KINDS:
        b = [c for c in cal if c["kind"] == k and c["n"] >= 20]
        if b:
            ax.plot([c["mean_mid"] for c in b], [c["obs_rate"] for c in b], marker="o", label=k)
    ax.set_xlabel("Kalshi mid at T-60")
    ax.set_ylabel("Realized YES rate")
    ax.set_title("Kalshi NFL prop calibration at T-60 (bins with n>=20) - draft")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close("all")


# ---------------------------------------------------------------- report helpers
def F(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def C(x):
    return "n/a" if x is None else f"{x * 100:+.1f}c"


def U(x):
    return "n/a" if x is None else f"{x * 100:.1f}c"


def Ci(ci):
    return "n/a" if not ci else f"[{ci[0] * 100:+.1f}c, {ci[1] * 100:+.1f}c]"


def Money(x):
    return "n/a" if x is None else f"${x:,.0f}"


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default="2026-11-14_prop-pricing")
    ap.add_argument("--series", default="KXNFLTD,KXNFLREC,KXNFLRECYDS,KXNFLRSHYDS,KXNFLPASSYDS")
    ap.add_argument("--boot", type=int, default=N_SPA)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--partial-note", default="")
    a = ap.parse_args(argv)
    t0 = datetime.now(timezone.utc)
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.base)
    rng = np.random.default_rng(SEED)
    from Backtest.Lab import LoadSeason
    games = {g["game_id"]: g for s in (2025, 2026) for g in LoadSeason(s) if g.get("gameday")}
    players = Players()
    series = [s.strip() for s in a.series.split(",") if s.strip()]
    recs, cov = LoadRecords(series, players, games)
    print(f"records={len(recs)}", flush=True)
    fits = FitModels(recs)
    ml = LoadMl()
    configs = PropConfigs() + MlConfigs()
    all_bets, rows = [], []
    for c in configs:
        bets = BetsFor(c, recs, ml, fits)
        all_bets.append(bets)
        rows.append({"id": Cid(c), **c, **Score(bets)})
    print(f"configs={len(configs)}", flush=True)
    days = {sp: sorted({r["date"] for r in recs if r["split"] == sp} | {m["date"] for m in ml if m["split"] == sp})
            for sp in ("fit", "hold")}
    ml_hold_days = sorted({m["date"] for m in ml if m["split"] == "hold"})
    prop_idx = [i for i, c in enumerate(configs) if c["family"] == "prop"]
    ml_idx = [i for i, c in enumerate(configs) if c["family"] == "ml"]
    spa_hold = SpaRc(DailyMatrix(all_bets, "hold", days["hold"]), n_boot=a.boot)
    spa_fit = SpaRc(DailyMatrix(all_bets, "fit", days["fit"]), n_boot=a.boot)
    spa_hold_prop = SpaRc(DailyMatrix([all_bets[i] for i in prop_idx], "hold", days["hold"]), n_boot=a.boot)
    spa_hold_ml = SpaRc(DailyMatrix([all_bets[i] for i in ml_idx], "hold", ml_hold_days), n_boot=a.boot)
    print("spa done", flush=True)

    def Pick(idxs):
        cand = [i for i in idxs if rows[i]["n_fit"] >= MIN_FIT_N and rows[i]["edge_fit"] is not None]
        return max(cand, key=lambda i: rows[i]["edge_fit"]) if cand else None

    tally = PRIOR_TALLY + len(configs)
    frozen = {}
    for fam, idxs in (("prop", prop_idx), ("ml", ml_idx)):
        i = Pick(idxs)
        if i is None:
            frozen[fam] = None
            continue
        bets = all_bets[i]
        hold = [b for b in bets if b["split"] == "hold"]
        info = {"config": configs[i], "id": rows[i]["id"], "score": rows[i],
                "ci_tim": EdgeCis(hold, "tim", rng), "ci_kalshi": EdgeCis(hold, "kalshi", rng)}
        de = defaultdict(list)
        for b in hold:
            de[b["date"]].append(EdgeOf(b))
        info["dsr"] = DeflatedSharpe([float(np.mean(v)) for _, v in sorted(de.items())], len(configs))
        if fam == "prop":
            ask = [b for b in BetsFor(configs[i], recs, ml, fits, fill_kind="ask") if b["split"] == "hold"]
            info["ask_secondary"] = {"n": len(ask), "edge_tim": float(np.mean([EdgeOf(b) for b in ask])) if ask else None,
                                     "edge_kalshi": float(np.mean([EdgeOf(b, "kalshi") for b in ask])) if ask else None}
        hdays = sorted({r["date"] for r in recs if r["split"] == "hold" and r["kind"] == configs[i]["kind"]}) if fam == "prop" else ml_hold_days
        banks = {}
        for fk in ("tim", "kalshi"):
            bundle, dailies, logs, note = BankAndMc(WalkEdges(hold, f"kalshi_T{configs[i]['tmin']}_mid"), hdays, rng,
                                                    fee_kind=fk, n_paths=a.paths)
            banks[fk] = {"bundle": {s: {k: v for k, v in d.items() if k != "pathsim"} for s, d in bundle.items()},
                         "pathsim": note, "dailies": dailies}
        info["bank"] = banks
        lo_t = (info["ci_tim"].get("widest_95") or [None])[0]
        lo_k = (info["ci_kalshi"].get("widest_95") or [None])[0]
        info["claimable"] = bool(lo_t is not None and lo_k is not None and lo_t > 0 and lo_k > 0)
        info["holdout_bets"] = hold
        frozen[fam] = info
    print("frozen done", flush=True)
    pricing, cal, ladder = PricingTables(recs)
    disagree = Disagreement(recs)
    ign = Ignorance(recs, fits)

    fp = frozen.get("prop")
    verdict_exists = bool(fp and fp["claimable"] and (spa_hold.get("spa_p") if spa_hold.get("spa_p") is not None else 1) < 0.05)
    if fp:
        ct = fp["ci_tim"]
        vline = (f"VERDICT: {'prop edge exists' if verdict_exists else 'no prop edge'} | holdout edge "
                 f"{C(ct.get('mean_edge'))}/contract (mid, tim_est_0.02) | widest 95% CI {Ci(ct.get('widest_95'))} "
                 f"({ct.get('widest_name')}) | N={ct.get('n', 0)} | SPA p={F(spa_hold.get('spa_p'))} "
                 f"(RC p={F(spa_hold.get('rc_p'))}, {len(configs)} configs)")
    else:
        vline = f"VERDICT: no prop edge | no prop config reached n>={MIN_FIT_N} on the fit period | SPA p={F(spa_hold.get('spa_p'))}"
    tidy = []
    for fam in ("prop", "ml"):
        fz = frozen.get(fam)
        if not fz:
            continue
        for fk in ("tim", "kalshi"):
            for strat, daily in fz["bank"][fk]["dailies"].items():
                for d in daily:
                    tidy.append({"date": d["date"], "series": f"{fam}_{strat}_{fk}", "family": fam, "strategy": strat,
                                 "fee_basis": FEE_BASIS if fk == "tim" else "kalshi_taker_0.07p(1-p)",
                                 "season": d["season"], "bets_day": d["bets_day"], "bets_cum": d["bets_cum"],
                                 "equity": d["equity"], "drawdown_pct": d["drawdown_pct"]})
    WriteCsv(base + ".csv", ["date", "series", "family", "strategy", "fee_basis", "season", "bets_day", "bets_cum",
                             "equity", "drawdown_pct"], tidy)
    cfg_fields = ["id", "family", "kind", "side", "model", "min_edge", "tmin", "band", "n_fit", "edge_fit",
                  "edge_fit_kalshi", "hit_fit", "fill_fit", "n_hold", "edge_hold", "edge_hold_kalshi", "hit_hold", "fill_hold"]
    WriteCsv(base + "_configs.csv", cfg_fields, rows)
    WriteCsv(base + "_pricing.csv", list(pricing[0].keys()) if pricing else ["kind"], pricing)
    WriteCsv(base + "_calibration.csv", ["kind", "bin_lo", "bin_hi", "n", "mean_mid", "obs_rate", "mean_proj"], cal)
    WriteCsv(base + "_disagreement.csv", list(disagree[0].keys()) if disagree else ["dim"], disagree)
    if fp:
        WriteCsv(base + "_holdout_bets.csv", ["date", "event_id", "ticker", "kind", "player", "pos", "thr", "side", "fill", "p",
                                             "result"], fp["holdout_bets"])
    try:
        series_charts = []
        for fam in ("prop", "ml"):
            fz = frozen.get(fam)
            if fz:
                for strat, daily in fz["bank"]["tim"]["dailies"].items():
                    series_charts.append((f"{fam} {strat} (mid, tim_est_0.02)", daily))
        WriteBankChart(series_charts, base + "_draft.png", "NFL props + moneylines: frozen configs on the holdout, $10k (draft)")
        CalChart(cal, base + "_calibration_draft.png")
    except Exception as exc:
        print(f"chart failed: {type(exc).__name__}: {exc}", flush=True)

    def Strip(fz):
        if not fz:
            return None
        o = {k: v for k, v in fz.items() if k not in ("holdout_bets", "bank")}
        o["bank"] = {fk: {"bundle": v["bundle"], "pathsim": v["pathsim"]} for fk, v in fz["bank"].items()}
        return o

    meta = {"run_id": NewRunId(), "model_version": ModelVersion(), "code_version": CodeVersion(),
            "generated_at": datetime.now(timezone.utc).isoformat(), "verdict_line": vline,
            "verdict_prop_edge": verdict_exists, "partial_note": a.partial_note,
            "fee_basis": FEE_BASIS, "kalshi_fee": "0.07*p*(1-p) taker", "fill_primary": "mid", "fill_secondary": "ask",
            "coverage": cov, "n_records": len(recs),
            "records_by_kind_split": {f"{k}|{sp}": sum(1 for r in recs if r["kind"] == k and r["split"] == sp)
                                      for k in KINDS for sp in ("fit", "hold")},
            "games_by_kind_split": {f"{k}|{sp}": len({r["game_id"] for r in recs if r["kind"] == k and r["split"] == sp})
                                    for k in KINDS for sp in ("fit", "hold")},
            "weeks_by_kind_split": {f"{k}|{sp}": sorted({(r["season"], r["week"]) for r in recs if r["kind"] == k and r["split"] == sp})
                                    for k in KINDS for sp in ("fit", "hold")},
            "fits": fits, "n_configs": len(configs), "n_prop_configs": len(prop_idx), "n_ml_configs": len(ml_idx),
            "trial_tally_prior": PRIOR_TALLY, "trial_tally_now": tally,
            "spa_hold_all": spa_hold, "spa_fit_all": spa_fit, "spa_hold_prop": spa_hold_prop, "spa_hold_ml": spa_hold_ml,
            "frozen_prop": Strip(frozen.get("prop")), "frozen_ml": Strip(frozen.get("ml")),
            "pricing": pricing, "td_ladder": ladder, "ignorance_holdout": ign,
            "n_ml_rows": len(ml)}
    meta["top10_fit"] = sorted([r for r in rows if r["n_fit"] >= MIN_FIT_N and r["edge_fit"] is not None],
                               key=lambda r: -r["edge_fit"])[:10]
    meta["runtime_s"] = (datetime.now(timezone.utc) - t0).total_seconds()
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(MdReport(meta, vline, frozen, pricing, ladder, disagree, ign, cov))
    print(vline, flush=True)
    return meta


def MdReport(meta, vline, frozen, pricing, ladder, disagree, ign, cov):
    L = [vline, ""]
    if meta.get("partial_note"):
        L += [f"**PARTIAL:** {meta['partial_note']}", ""]
    L += ["# NFL prop pricing (Kalshi) + selective moneylines - American Footbot, 2026-11-14", "",
          "Simulation only. No orders. Fill = Kalshi mid (Tim's primary rule); ask shown as a secondary. "
          "Fee = $0.02/contract, **Tim's fee estimate (tim_est_0.02)**, plus the verified Kalshi taker fee 0.07*p*(1-p). "
          "Times ET unless marked UTC.", "",
          "## Caveats and assumptions (read first)", "",
          "- Historical **sportsbook** prop odds: no free source found (nflverse has none; The Odds API and similar keep history "
          "behind paid plans). All price data here is Kalshi. Book hold figures below are cited web sources, not our data.",
          "- KXNFLTD (anytime/multi TD) exists on Kalshi only from the 2026 season, so TD fit = 2026 games before 2026-10-08 (weeks 1-4) and TD "
          "holdout = games from 2026-10-08 (weeks 5-10); split date fixed before results (not a full-season holdout). Yards/receptions: fit = 2025 (listed from Oct 2025), holdout = 2026. 2025 (historical-tier) ladders were sampled to "
          "the middle 2 strikes per player (one API call per market; time budget); 2026 keeps every pregame-listed strike.",
          "- Only markets listed before kickoff are used (Kalshi also lists in-game ladders). Void/scalar results dropped.",
          "- Quotes come from 1-minute and hourly candles; a quote is the last two-sided (0<bid<ask<1) candle at or before the "
          "entry time, stale limit 180 min (prop books are quiet). Mid fills assume you can trade at mid; at ask the result is worse.",
          "- Projections are deliberately simple (weighted history of the player's own games; no injuries, no game script, "
          "no team totals). A player with <3 prior games gets no projection and is not bet.",
          "- Player matching is by name + team (Kalshi ticker prefix) against nflverse player_week. Unmatched players are excluded from model tests.",
          "- nflverse player_week for 2026 runs through week 9; Kalshi results settle the bets, so week-10 markets still score.",
          "- Moneylines reuse the walk-forward p_full2_kmid_T60 edges from 2026-11-13_bankroll-10k (one side per game in that file; "
          "the other side is taken as 1-p at 1-mid, an approximation). ML holdout = 2026 weeks 1-9. No ask secondary for ML (file has mid only).",
          f"- Trial tally: prior lower bound {meta['trial_tally_prior']:,} + {meta['n_configs']:,} configs here = **{meta['trial_tally_now']:,}**.",
          "- Bankroll: one continuous $10k over the holdout only (the fit period is in-sample for the chosen config).", ""]
    L += ["## Data coverage", "", "| series | listed | pregame-listed | usable (settled yes/no, game matched) | player matched | with projection |",
          "|---|---|---|---|---|---|"]
    for s, c in cov.items():
        L.append(f"| {s} | {c.get('markets_listed', 0):,} | {c.get('markets_pregame_listed', 0):,} | {c.get('usable', 0):,} | "
                 f"{c.get('player_matched', 0):,} | {c.get('with_projection', 0):,} |")
    L += ["", "| kind | split | contracts | games | season-weeks |", "|---|---|---|---|---|"]
    for k in KINDS:
        for sp in ("fit", "hold"):
            wk = meta["weeks_by_kind_split"][f"{k}|{sp}"]
            span = f"{wk[0][0]} wk{wk[0][1]} - {wk[-1][0]} wk{wk[-1][1]}" if wk else "none"
            L.append(f"| {k} | {sp} | {meta['records_by_kind_split'][f'{k}|{sp}']:,} | {meta['games_by_kind_split'][f'{k}|{sp}']} | {span} |")
    L += ["", "## 1. How props are priced", "", "**Sourced (web research, not our data):**", "",
          "- Lines start from a projection of the player's stat distribution (role/usage x efficiency x matchup/game script); the book posts "
          "a threshold near the middle and adds margin. Many US books buy prices from shared odds providers; a few originate. Limits open low and rise "
          "toward kickoff, so late prices are sharper. Sources: intellipickz.com/guides/how-sportsbooks-set-prop-lines; proplabsports.com/learn/how-sportsbooks-set-prop-lines; "
          "establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props.",
          "- Typical sportsbook hold: spreads/totals ~4.5%, player props ~8-14%, anytime TD ~10-20%+ (implied probabilities across a game "
          "sum to ~115-125%), first TD 25-30%. Sources: bettingonline.org/sports/football/nfl-player-props; prophetaiq.com/blog/nfl-player-props-betting-guide-2026; "
          "nflsportsbettingstats.com/articles/nfl-anytime-vs-first-td-scorer. These are industry/blog estimates, not audited figures.", "",
          "**Our Kalshi data (width = ask - bid; buying at the ask costs half the width above mid, so hold-equivalent per side ~ width/2):**", "",
          "| kind | season | contracts | games | median width T-60 | mean width open / T-60 / close | hrs listed pre-KO | mean abs move open->close | moved >=3c | Brier open / close | longshot <20c: result-mid | fav >80c: result-mid |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pricing:
        L.append(f"| {p['kind']} | {p['season']} | {p['n_markets']:,} | {p['n_games']} | {U(p['median_width_T60'])} | "
                 f"{U(p['mean_width_open'])} / {U(p['mean_width_T60'])} / {U(p['mean_width_close'])} | {F(p['mean_hours_listed_before_ko'], 1)} | "
                 f"{U(p['mean_abs_move_open_close'])} | {F(p['share_moved_ge_3c'], 2)} | {F(p['brier_open'], 4)} / {F(p['brier_close'], 4)} | "
                 f"{C(p['longshot_lt20_result_minus_mid'])} (n={p['n_lt20']}) | {C(p['fav_gt80_result_minus_mid'])} (n={p['n_gt80']}) |")
    if ladder.get("n_games"):
        L += ["", f"Anytime-TD ladder overround: across {ladder['n_games']} games, the T-60 mids of listed players sum to "
              f"{F(ladder['mean_sum_mid_T60'], 2)} expected scorers vs {F(ladder['mean_realized_scorers'], 2)} realized "
              f"(ratio {F(ladder.get('implied_over_realized'), 3)}; ~{F(ladder['mean_listed_players'], 0)} players listed per game)."]
    L += ["", "## 2. Projections vs Kalshi lines (T-60 unless noted)", "",
          "| dim | value | time | n | mean proj-mid | mean abs diff | share >=10c | Brier proj | Brier mid | Brier proj / mid when >=10c apart |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in disagree:
        if d["dim"].endswith("_line") and d["n"] < 200:
            continue
        L.append(f"| {d['dim']} | {d['value']} | {d['time']} | {d['n']:,} | {C(d['mean_proj_minus_mid'])} | {U(d['mean_abs_diff'])} | "
                 f"{F(d['share_abs_diff_ge_10c'], 2)} | {F(d['brier_proj'], 4)} | {F(d['brier_mid'], 4)} | "
                 f"{F(d['brier_proj_big_disagree'], 4)} / {F(d['brier_mid_big_disagree'], 4)} |")
    L += ["", "(Line rows with n<200 omitted here; all rows are in `_disagreement.csv`.)", "",
          "Holdout log-loss gap vs Kalshi T-60 mid (positive = model worse than the market):", "",
          "| kind | model | n | Ignorance gap (bits) | Brier model | Brier mid |", "|---|---|---|---|---|---|"]
    for g in ign:
        L.append(f"| {g['kind']} | {g['model']} | {g['n']:,} | {g['ignorance_gap_bits']:+.4f} | {g['brier_model']:.4f} | {g['brier_mid']:.4f} |")
    L += ["", "## 3. Backtest (pre-registered grid, fit -> freeze -> holdout)", "",
          f"Configs: {meta['n_prop_configs']:,} prop + {meta['n_ml_configs']} moneyline = {meta['n_configs']:,}.", "",
          "| test | RC p | SPA p | days | best mean daily edge |", "|---|---|---|---|---|"]
    for lab, key in (("holdout, all configs", "spa_hold_all"), ("holdout, props only", "spa_hold_prop"),
                     ("holdout, moneylines only", "spa_hold_ml"), ("fit, all configs", "spa_fit_all")):
        s = meta[key]
        L.append(f"| {lab} | {F(s.get('rc_p'))} | {F(s.get('spa_p'))} | {s.get('n_days')} | {C(s.get('observed_max_mean_daily_edge'))} |")
    for fam, title in (("prop", "Frozen best prop config"), ("ml", "Frozen best moneyline config")):
        fz = frozen.get(fam)
        L += ["", f"### {title}", ""]
        if not fz:
            L.append(f"No config reached n_fit >= {MIN_FIT_N}.")
            continue
        sc, ct, ck = fz["score"], fz["ci_tim"], fz["ci_kalshi"]
        L += [f"`{fz['id']}`", "",
              f"- Fit: n={sc['n_fit']}, mean edge {C(sc['edge_fit'])} (tim), {C(sc['edge_fit_kalshi'])} (Kalshi fee), hit {F(sc['hit_fit'])} at mean fill {F(sc['fill_fit'])}.",
              f"- **Holdout: n={ct.get('n', 0)}, mean edge {C(ct.get('mean_edge'))} (tim) / {C(ck.get('mean_edge'))} (Kalshi fee), hit {F(sc['hit_hold'])} at mean fill {F(sc['fill_hold'])}.**",
              f"- Holdout 95% CIs (tim): day-block {Ci(ct.get('day_block_95'))}, game-cluster {Ci(ct.get('game_cluster_95'))}, binomial {Ci(ct.get('binomial_95'))}; widest = {ct.get('widest_name')}.",
              f"- Holdout 95% CIs (Kalshi fee): widest {Ci(ck.get('widest_95'))}.",
              f"- Power: MDE at 80% power ~{U(ct.get('mde_80pct_power'))}/contract; power to detect a true +3c edge = {F(ct.get('power_at_3c'), 2)} (design effect {F(ct.get('design_effect'), 2)}).",
              f"- Deflated Sharpe (N={meta['n_configs']}): {F((fz.get('dsr') or {}).get('dsr'))}.",
              f"- Claimable (widest lower bound > 0 under both fees): **{fz['claimable']}**."]
        if fz.get("ask_secondary"):
            a = fz["ask_secondary"]
            L.append(f"- Ask fill (secondary): n={a['n']}, mean edge {C(a['edge_tim'])} (tim) / {C(a['edge_kalshi'])} (Kalshi fee).")
        L += ["", "| fee | strategy | bets | final | ROI | max DD | PathSim median / p10 / p90 | P(final < $10k) | P(ruin) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for fk in ("tim", "kalshi"):
            for strat, b in fz["bank"][fk]["bundle"].items():
                ps = fz["bank"][fk]["pathsim"].get(strat) or {}
                L.append(f"| {fk} | {strat} | {b['n_bets']} | {Money(b['final_equity'])} | {F(b['roi'])} | {F(b['max_drawdown_pct'])} | "
                         f"{Money(ps.get('median_final'))} / {Money(ps.get('p10_final'))} / {Money(ps.get('p90_final'))} | "
                         f"{F(ps.get('p_final_below_start'), 2)} | {F(ps.get('p_ruin'), 3)} |")
    L += ["", "### Top 10 configs on the fit period (and what they did on the holdout)", "",
          "| config | n fit | edge fit | n hold | edge hold (tim) | edge hold (Kalshi fee) |", "|---|---|---|---|---|---|"]
    for r in meta["top10_fit"]:
        L.append(f"| `{r['id']}` | {r['n_fit']} | {C(r['edge_fit'])} | {r['n_hold']} | {C(r['edge_hold'])} | {C(r['edge_hold_kalshi'])} |")
    L += ["", "## Files", "",
          "- `<base>.csv`: tidy holdout bankroll (date, series, equity, drawdown_pct) for the frozen prop and ML configs, both fee bases.",
          "- `<base>_configs.csv`: every config, fit and holdout. `<base>_pricing.csv`, `<base>_calibration.csv`, `<base>_disagreement.csv`, `<base>_holdout_bets.csv`.",
          "- `<base>.json`: all numbers, coverage, fits, SPA/RC, DSR, PathSim. `<base>_draft.png` bankroll (season-axis rule), `<base>_calibration_draft.png`.",
          "", f"Run {meta['run_id']}, code {meta['code_version']}, runtime {meta['runtime_s']:.0f}s."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    Main()
