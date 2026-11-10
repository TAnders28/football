r"""
MartingaleParlay: simulation-only study of a martingale ladder on NFL moneyline parlays (Tim via Degen,
2026-11-10). Objective: maximise P(first reach $1,000,000 before ruin) subject to P(ruin) < 50%, separately
for starting banks $100, $1,000 and $10,000. No wagers, nothing live; FrozenConfig/PredictDay untouched.

    .venv\Scripts\python.exe -m Backtest.MartingaleParlay [--out-dir DIR --base NAME --sims 5000]

Strategy (one parlay per ET game day at most)
  * Legs: n legs (2-4) from n DISTINCT games of that day, one moneyline side each, chosen PRE-GAME only.
    PickLegs sees a pre-game view of each side (posted ML, de-vigged market prob, walk-forward p_full2 fit on
    seasons < S); it never sees results. The parlay must price within x/÷1.35 of the target decimal odds.
      selection "market": combo whose posted price is closest to the target (tie -> higher de-vig prob).
      selection "model":  combo in the band with the highest model EV, sum(log p_model + log decimal).
  * Settlement: whole day, at day end (ET). Tied legs are voided (dropped from the price); all-void = push.
    The stake is sized from the QUOTED price before kickoff (voids are not known then).
  * Progression: base = base_pct x bankroll at the last reset (start, after a win, or after a stop-loss).
    After a loss: stake = base x mult^losses, or for "recover" the stake that wins back the streak's losses
    plus base x (quoted price - 1). Win -> reset. Optional max-stake cap (book limit) and stop-loss after k losses.
  * No reuse of unsettled money: one parlay a day, stake <= bankroll at the start of the day.
Absorbing outcomes (first passage, cash held afterwards)
  * TARGET: bankroll >= $1,000,000 at a day end.
  * RUIN: the next required stake (after any cap, floored at MIN_STAKE $1) exceeds the bankroll, or the
    bankroll is < $1. "low10" = ever at or below 10% of the start before absorption (reported, not absorbing).
Pricing bases
  * posted: payout = product of posted decimal odds (closing lines, vig included).
  * fee_est: Tim's working $0.02/contract ESTIMATE (not verified): each leg bought as a $1 event contract at
    de-vigged prob + $0.02, so parlay decimal = prod 1/(p_mkt + 0.02). Same legs as the posted basis.
Monte Carlo: day-block bootstrap of whole ET game days (days with no possible parlay included, so the clock
is real), N_SIMS paths, one common index matrix per pool (seed SEED). Horizon T = number of game days in the
replay window (2023-09 .. last finished day). These are conditional replay rates, not forecasts. A martingale
cannot create edge: if the legs are -EV after vig, bigger progressions only reshape the distribution.
"""

import argparse
import csv
import json
import math
import os
from itertools import combinations, groupby, product

import numpy as np

from Backtest.Lab import AllRows, AttachModels, Decimal
from Backtest.Version import CodeVersion, NewRunId

TARGET = 1_000_000.0
MIN_STAKE = 1.0
FEE_EST = 0.02
N_SIMS = 5000
SEED = 20261110
BAND = 1.35
BANKS = (100.0, 1_000.0, 10_000.0)
POOL_FIRST = 2014
EARLY = list(range(2014, 2020))
LATE = list(range(2020, 2027))
REPLAY_FROM = "2023-09-01"
PRIOR_SETTINGS = 2367
TARGETS = {"p100": 2.0, "p200": 3.0, "p300": 4.0, "p400": 5.0, "p600": 7.0}
LEGS = (2, 3, 4)
SELECTIONS = ("market", "model")
BASE_PCTS = (0.02, 0.05, 0.10)
MULTS = (1.5, 2.0, 3.0, "recover")
CAPS = (1_000.0, 5_000.0, 25_000.0, 100_000.0)
STOPS = (3, 5)
BASES = ("posted", "fee_est")
DEFAULT_OUT = os.path.join("data", "backtest", "martingale")


# ----------------------------------------------------------------------------- grid
def Series():
    return [{"name": f"{sel}_{t}_{n}leg", "target_name": t, "target": d, "legs": n, "selection": sel}
            for t, d in TARGETS.items() for n in LEGS for sel in SELECTIONS]


def Cfg(series, base, mult, cap=None, stop=None):
    m = mult if mult == "recover" else f"{mult:g}x"
    name = f"{series}|b{base * 100:g}|{m}" + (f"|cap{int(cap)}" if cap else "") + (f"|stop{stop}" if stop else "")
    return {"name": name, "series": series, "base": base, "mult": mult, "cap": cap, "stop": stop}


def MainGrid():
    return [Cfg(s["name"], b, m) for s in Series() for b in BASE_PCTS for m in MULTS]


def Variants(cfg):
    return ([Cfg(cfg["series"], cfg["base"], cfg["mult"], cap=c) for c in CAPS] +
            [Cfg(cfg["series"], cfg["base"], cfg["mult"], stop=k) for k in STOPS])


# ----------------------------------------------------------------------------- pre-game leg choice (no outcomes)
def SideView(r, side):
    pm = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
    pf = r.get("p_full2", r["mkt_home"])
    return {"game_id": r["game_id"], "side": side, "team": r[f"{side}_team"], "dec": Decimal(r[f"{side}_ml"]),
            "p_mkt": pm, "p_model": pf if side == "home" else 1 - pf}


def Pregame(day_rows):
    """Pre-game view of a day: posted prices and pre-game probabilities only. No result fields."""
    games = []
    for r in sorted(day_rows, key=lambda r: r["game_id"]):
        if r.get("home_ml") is None or r.get("away_ml") is None:
            continue
        games.append({"game_id": r["game_id"], "sides": [SideView(r, "home"), SideView(r, "away")]})
    return games


def PickLegs(games, n, target, selection):
    """Best n-leg combo over n distinct games, one side each, priced within the band. None if impossible."""
    k = len(games)
    if k < n:
        return None
    LD = np.array([[math.log(s["dec"]) for s in g["sides"]] for g in games])
    LM = np.array([[math.log(max(s["p_mkt"], 1e-9)) for s in g["sides"]] for g in games])
    LP = np.array([[math.log(max(s["p_model"], 1e-9)) for s in g["sides"]] for g in games])
    C = np.array(list(combinations(range(k), n)))
    S = np.array(list(product((0, 1), repeat=n)))
    gi, si = C[:, None, :], S[None, :, :]
    lp = LD[gi, si].sum(-1)
    dist = np.abs(lp - math.log(target))
    ok = dist <= math.log(BAND)
    if not ok.any():
        return None
    if selection == "market":
        score = dist - 1e-6 * LM[gi, si].sum(-1)
    else:
        score = -(LP[gi, si].sum(-1) + lp) + 1e-6 * dist
    score = np.where(ok, score, np.inf)
    c, s = np.unravel_index(int(np.argmin(score)), score.shape)
    return [games[g]["sides"][x] for g, x in zip(C[c], S[s])]


def LegResult(r, side):
    """True/False, or None for a tie (void). Settlement only; never used to choose legs."""
    if r.get("result") == 0 or r.get("home_win") is None:
        return None
    return (r["home_win"] == 1) == (side == "home")


# ----------------------------------------------------------------------------- days and parlay series
def BuildDays(rows_by_season, seasons):
    """One entry per ET game day whose games are all finished (unplayed future days are dropped)."""
    days = []
    for s in seasons:
        rows = sorted(rows_by_season.get(s, []), key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"]))
        for day, grp in groupby(rows, key=lambda r: r["gameday"]):
            grp = list(grp)
            if any(r.get("result") is None for r in grp):
                continue
            days.append({"date": day, "season": s, "rows": grp})
    return days


def BuildSeries(days, sr):
    n = len(days)
    out = {"has": np.zeros(n, bool), "q_post": np.ones(n), "q_fee": np.ones(n), "s_post": np.ones(n),
           "s_fee": np.ones(n), "s_fair": np.ones(n), "out": np.zeros(n, int), "legs": [None] * n, "dates": [d["date"] for d in days],
           "seasons": np.array([d["season"] for d in days])}
    for i, d in enumerate(days):
        legs = PickLegs(Pregame(d["rows"]), sr["legs"], sr["target"], sr["selection"])
        if not legs:
            continue
        byid = {r["game_id"]: r for r in d["rows"]}
        res = [LegResult(byid[l["game_id"]], l["side"]) for l in legs]
        fee = [1.0 / min(l["p_mkt"] + FEE_EST, 0.999) for l in legs]
        out["has"][i] = True
        out["q_post"][i] = math.prod(l["dec"] for l in legs)
        out["q_fee"][i] = math.prod(fee)
        live = [j for j, w in enumerate(res) if w is not None]
        out["s_post"][i] = math.prod(legs[j]["dec"] for j in live) if live else 1.0
        out["s_fee"][i] = math.prod(fee[j] for j in live) if live else 1.0
        out["s_fair"][i] = math.prod(1.0 / max(legs[j]["p_mkt"], 1e-9) for j in live) if live else 1.0
        out["out"][i] = 0 if not live else (1 if all(res[j] for j in live) else -1)
        out["legs"][i] = "+".join(f"{l['team']}({l['dec']:.2f})" for l in legs)
    return out


def SubSeries(ser, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else [x for x, m in zip(v, mask) if m]) for k, v in ser.items()}


# ----------------------------------------------------------------------------- engine (vectorised over paths)
def Simulate(ser, idx, cfg, start, basis="posted", record=False):
    P, steps = idx.shape
    q = ser["q_post"] if basis == "posted" else ser["q_fee"]
    st = ser["s_post"] if basis == "posted" else ser["s_fee"]
    bank = np.full(P, float(start))
    base = np.full(P, max(cfg["base"] * start, MIN_STAKE))
    streak = np.zeros(P, int)
    cum = np.zeros(P)
    hit = np.zeros(P, bool)
    ruin = np.zeros(P, bool)
    low10 = np.zeros(P, bool)
    t_abs = np.full(P, -1)
    bets = np.zeros(P, int)
    bets_abs = np.full(P, -1)
    peak = bank.copy()
    maxdd = np.zeros(P)
    max_stake = np.zeros(P)
    rec = []
    for j in range(steps):
        d = idx[:, j]
        live = ~hit & ~ruin
        has = ser["has"][d] & live
        qd = q[d]
        if cfg["mult"] == "recover":
            stake = (cum + base * (qd - 1)) / np.maximum(qd - 1, 1e-9)
        else:
            stake = base * float(cfg["mult"]) ** streak
        if cfg.get("cap"):
            stake = np.minimum(stake, cfg["cap"])
        stake = np.maximum(stake, MIN_STAKE)
        short = has & (stake > bank + 1e-9)
        place = has & ~short
        o = ser["out"][d]
        win, loss = place & (o == 1), place & (o == -1)
        bank = bank + np.where(win, stake * (st[d] - 1), 0.0) - np.where(loss, stake, 0.0)
        bets += place
        max_stake = np.maximum(max_stake, np.where(place, stake, 0.0))
        streak = np.where(win, 0, np.where(loss, streak + 1, streak))
        cum = np.where(win, 0.0, np.where(loss, cum + stake, cum))
        reset = win
        if cfg.get("stop"):
            stopped = loss & (streak >= cfg["stop"])
            streak, cum = np.where(stopped, 0, streak), np.where(stopped, 0.0, cum)
            reset = win | stopped
        base = np.where(reset, np.maximum(cfg["base"] * bank, MIN_STAKE), base)
        new_hit = live & ~short & (bank >= TARGET)
        new_ruin = live & ~new_hit & (short | (bank < MIN_STAKE))
        t_abs = np.where((new_hit | new_ruin) & (t_abs < 0), j, t_abs)
        bets_abs = np.where((new_hit | new_ruin) & (bets_abs < 0), bets, bets_abs)
        hit |= new_hit
        ruin |= new_ruin
        low10 |= live & (bank <= 0.1 * start)
        peak = np.maximum(peak, bank)
        maxdd = np.maximum(maxdd, 1 - bank / peak)
        if record:
            rec.append((bank.copy(), 1 - bank / peak, place.copy(), stake.copy(), o.copy()))
    out = {"end": bank, "hit": hit, "ruin": ruin, "low10": low10, "t_abs": t_abs, "bets": bets,
           "bets_abs": bets_abs, "maxdd": maxdd, "max_stake": max_stake}
    if record:
        out["path"] = rec
    return out


def CommonIndex(n_days, steps, n_sims=N_SIMS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n_days, size=(n_sims, steps))


def Med(x):
    return float(np.median(x)) if len(x) else None


def BootStats(res, n, start):
    h, r = res["hit"], res["ruin"]
    assert not (h & r).any()
    p = float(h.mean())
    se = float(np.sqrt(p * (1 - p) / n))
    absorbed = h | r
    e = res["end"]
    return {"p_target": round(p, 4), "p_target_mc_se": round(se, 4), "p_ruin": round(float(r.mean()), 4),
            "p_neither": round(float((~h & ~r).mean()), 4), "p_low10": round(float(res["low10"].mean()), 4),
            "median_days_to_either": Med(res["t_abs"][absorbed] + 1), "median_days_to_target": Med(res["t_abs"][h] + 1),
            "median_bets_to_target": Med(res["bets_abs"][h]), "median_days_to_ruin": Med(res["t_abs"][r] + 1),
            "median_end": round(float(np.median(e)), 2), "p5_end": round(float(np.percentile(e, 5)), 2),
            "p95_end": round(float(np.percentile(e, 95)), 2), "p_end_above_start": round(float((e > start).mean()), 4),
            "median_max_stake": round(float(np.median(res["max_stake"])), 2)}


def Replay(ser, cfg, start, basis):
    res = Simulate(ser, np.arange(len(ser["has"]))[None, :], cfg, start, basis, record=True)
    eq = [round(float(b[0]), 2) for b, *_ in res["path"]]
    dd = [round(float(-x[0]), 4) for _, x, *_ in res["path"]]
    t = int(res["t_abs"][0])
    return {"end": eq[-1] if eq else start, "hit": bool(res["hit"][0]), "ruin": bool(res["ruin"][0]),
            "absorbed_date": ser["dates"][t] if t >= 0 else None, "bets": int(res["bets"][0]),
            "max_dd": round(float(-res["maxdd"][0]), 4), "max_stake": round(float(res["max_stake"][0]), 2),
            "peak": max(eq) if eq else start, "equity": eq, "drawdown": dd}


# ----------------------------------------------------------------------------- descriptive series stats
def LongestLosing(out_bets):
    best = cur = 0
    for o in out_bets:
        if o == -1:
            cur += 1
            best = max(best, cur)
        elif o == 1:
            cur = 0
    return best


def SurvivalUnits(mult, L, q):
    """Total stake (in base units) of L losses plus the next bet, at a constant quoted price q."""
    stakes, cum = [], 0.0
    for k in range(L + 1):
        s = (cum + (q - 1)) / (q - 1) if mult == "recover" else float(mult) ** k
        stakes.append(s)
        cum += s
    return sum(stakes)


def SeriesStats(ser, sr):
    m = ser["has"]
    o = ser["out"][m]
    st = {"bets": int(m.sum()), "days": int(len(m)), "win_rate": round(float((o == 1).mean()), 4) if m.any() else None,
          "mean_quoted_decimal": round(float(ser["q_post"][m].mean()), 3) if m.any() else None,
          "longest_losing_streak": LongestLosing(o)}
    for b, s in (("posted", ser["s_post"]), ("fee_est", ser["s_fee"]), ("fair_devig", ser["s_fair"])):
        pnl = np.where(o == 1, s[m] - 1, np.where(o == -1, -1.0, 0.0))
        st[f"flat_roi_{b}"] = round(float(pnl.mean()), 4) if m.any() else None
    for era, seasons in (("early", EARLY), ("late", LATE)):
        mm = m & np.isin(ser["seasons"], seasons)
        oo = ser["out"][mm]
        st[f"flat_roi_posted_{era}"] = round(float(np.where(oo == 1, ser["s_post"][mm] - 1, np.where(oo == -1, -1.0, 0.0)).mean()), 4) if mm.any() else None
        st[f"longest_losing_streak_{era}"] = LongestLosing(oo)
    st["base_units_to_survive_longest"] = {str(mu): round(SurvivalUnits(mu, st["longest_losing_streak"], sr["target"]), 1) for mu in MULTS}
    st["max_base_pct_to_survive_longest"] = {k: round(1.0 / v, 5) for k, v in st["base_units_to_survive_longest"].items()}
    return st


def Choose(rows):
    ok = [r for r in rows if r["p_ruin"] < 0.5]
    if ok:
        return max(ok, key=lambda r: (r["p_target"], r["p95_end"], -r["p_ruin"])), True
    return min(rows, key=lambda r: (r["p_ruin"], -r["p_target"])), False


# ----------------------------------------------------------------------------- run
def Run(rows_by_season=None, out_dir=None, base=None, write=True, n_sims=N_SIMS, last=2026):
    if rows_by_season is None:
        rows_by_season = AllRows(2012, last)
        AttachModels(rows_by_season)
    days = BuildDays(rows_by_season, list(range(POOL_FIRST, last + 1)))
    dates = np.array([d["date"] for d in days])
    seasons = np.array([d["season"] for d in days])
    m_rep = dates >= REPLAY_FROM
    m_early, m_late = np.isin(seasons, EARLY), np.isin(seasons, LATE)
    T = int(m_rep.sum())
    series = {sr["name"]: (sr, BuildSeries(days, sr)) for sr in Series()}
    pools = {"full": np.ones(len(days), bool), "early": m_early, "late": m_late}
    sub = {(name, p): SubSeries(ser, mk) for name, (_, ser) in series.items() for p, mk in pools.items()}
    idx = {p: CommonIndex(int(mk.sum()), T, n_sims) for p, mk in pools.items()}
    rep = {name: SubSeries(ser, m_rep) for name, (_, ser) in series.items()}
    out = {"run_id": NewRunId(), "code": CodeVersion(), "target": TARGET, "min_stake": MIN_STAKE, "fee_est": FEE_EST,
           "n_sims": n_sims, "seed": SEED, "band": BAND, "horizon_days": T, "replay_from": REPLAY_FROM,
           "replay_to": str(dates[m_rep][-1]), "pool_days": {p: int(mk.sum()) for p, mk in pools.items()},
           "pool_first_date": str(dates[0]), "early": EARLY, "late": LATE, "banks": list(BANKS),
           "series_stats": {name: SeriesStats(ser, sr) for name, (sr, ser) in series.items()}}
    grid = MainGrid()

    def Eval(cfg, bank, basis, pool="full"):
        return BootStats(Simulate(sub[(cfg["series"], pool)], idx[pool], cfg, bank, basis), n_sims, bank)

    rows = []
    for cfg in grid:
        for bank in BANKS:
            for basis in BASES:
                rows.append({"config": cfg["name"], "start_bank": bank, "basis": basis, "set": "main", **Eval(cfg, bank, basis)})
    # cap / stop-loss sensitivity around the 5 best qualifying posted-basis configs per bank
    extra = {}
    for bank in BANKS:
        cand = sorted([r for r in rows if r["start_bank"] == bank and r["basis"] == "posted" and r["p_ruin"] < 0.5],
                      key=lambda r: (-r["p_target"], -r["p95_end"]))[:5]
        for r in cand:
            for v in Variants(next(c for c in grid if c["name"] == r["config"])):
                extra[v["name"]] = v
    for cfg in extra.values():
        for bank in BANKS:
            for basis in BASES:
                rows.append({"config": cfg["name"], "start_bank": bank, "basis": basis, "set": "sensitivity", **Eval(cfg, bank, basis)})
    allcfg = {c["name"]: c for c in grid} | extra
    out["settings_main"] = len(grid)
    out["settings_sensitivity"] = len(extra)
    out["settings_this_module"] = len(grid) + len(extra)
    out["settings_prior_lower_bound"] = PRIOR_SETTINGS
    out["settings_total_lower_bound"] = PRIOR_SETTINGS + len(grid) + len(extra)
    tidy, best = [], {}
    for bank in BANKS:
        for basis in BASES:
            mine = [r for r in rows if r["start_bank"] == bank and r["basis"] == basis]
            pick, qualifies = Choose(mine)
            upick = max(mine, key=lambda r: (r["p_target"], -r["p_ruin"]))
            ucfg = allcfg[upick["config"]]
            urep = Replay(rep[ucfg["series"]], ucfg, bank, basis)
            cfg = allcfg[pick["config"]]
            era = {p: Eval(cfg, bank, basis, p) for p in ("early", "late")}
            replay = Replay(rep[cfg["series"]], cfg, bank, basis)
            # nested: choose on the early pool (main grid), score on the late pool + replay window
            early_rows = [{"config": c["name"], **Eval(c, bank, basis, "early")} for c in grid]
            npick, nq = Choose(early_rows)
            ncfg = allcfg[npick["config"]]
            nested = {"choice": ncfg["name"], "qualified_on_early": nq, "early": {k: npick[k] for k in ("p_target", "p_ruin")},
                      "late": Eval(ncfg, bank, basis, "late")}
            nrep = Replay(rep[ncfg["series"]], ncfg, bank, basis)
            nested["replay"] = {k: v for k, v in nrep.items() if k not in ("equity", "drawdown")}
            key = f"{int(bank)}_{basis}"
            best[key] = {"in_sample_best": pick, "meets_ruin_lt_50": qualifies, "config": cfg, "era": era,
                         "replay": {k: v for k, v in replay.items() if k not in ("equity", "drawdown")}, "nested": nested,
                         "unconstrained_max_p_target": {"row": upick, "config": ucfg, "replay": {k: v for k, v in urep.items() if k not in ("equity", "drawdown")}}}
            for label, rp, c in (("insample", replay, cfg), ("nested", nrep, ncfg), ("unconstrained", urep, ucfg)):
                for dte, e, x in zip(rep[c["series"]]["dates"], rp["equity"], rp["drawdown"]):
                    tidy.append({"date": dte, "series": f"{label}:{c['name']}", "equity": e, "drawdown_pct": x,
                                 "start_bank": int(bank), "basis": basis})
    out["best"] = best
    out["rows"] = rows
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or "martingale_parlay"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "series", "equity", "drawdown_pct", "start_bank", "basis"])
            w.writeheader()
            w.writerows(tidy)
        fr = [{"date": out["replay_to"], "series": r["config"], "start_bank": int(r["start_bank"]), "basis": r["basis"],
               "set": r["set"], "p_target": r["p_target"], "p_ruin": r["p_ruin"], "p_neither": r["p_neither"],
               "p_low10": r["p_low10"], "median_days_to_either": r["median_days_to_either"],
               "median_days_to_target": r["median_days_to_target"], "median_end": r["median_end"],
               "meets_ruin_lt_50": int(r["p_ruin"] < 0.5)} for r in rows]
        with open(os.path.join(out_dir, f"{base}_frontier.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fr[0].keys()))
            w.writeheader()
            w.writerows(fr)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    return out


def Main():
    ap = argparse.ArgumentParser(description="Martingale moneyline-parlay simulation (no wagers).")
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    a = ap.parse_args()
    out = Run(out_dir=a.out_dir, base=a.base, n_sims=a.sims)
    print(f"run {out['run_id']} code {out['code']} T={out['horizon_days']} pools={out['pool_days']} settings={out['settings_this_module']}")
    for k, b in out["best"].items():
        p = b["in_sample_best"]
        print(k, p["config"], "Ptarget", p["p_target"], "Pruin", p["p_ruin"], "meets", b["meets_ruin_lt_50"],
              "| replay", b["replay"]["end"], "| nested", b["nested"]["choice"], b["nested"]["late"]["p_target"], b["nested"]["late"]["p_ruin"],
              "| uncon", b["unconstrained_max_p_target"]["row"]["config"], b["unconstrained_max_p_target"]["row"]["p_target"], b["unconstrained_max_p_target"]["row"]["p_ruin"])


if __name__ == "__main__":
    Main()