r"""
One continuous $10,000 bankroll compounding across seasons (no reset), with a NESTED walk-forward
choice of strategy + sizing, the frozen strategy compounded, an in-sample best (labelled overfit),
baselines, and a day-level bootstrap for P($1M) / P(ruin).

    .venv\Scripts\python.exe -m Backtest.Compound
    .venv\Scripts\python.exe -m Backtest.Compound --out-dir E:\AgentDocs\American-Footbot\2026-11-09_compounding-bankroll --base 2026-11-09_compounding-bankroll

Reads   data/backtest/games_<season>.json + history (Backtest.Lab.AllRows; no network)
Writes  <out-dir>/<base>.csv   tidy daily equity: date, series, equity, drawdown_pct (+ basis, bank_start)
        <out-dir>/<base>.json  every grid row, nested choices, bootstrap results, settings count, versions

Rules (fixed BEFORE looking at any result of this module)
  * Bankroll: START at the first game day of EVAL_SEASONS[0] and never reset. Stakes for a day are sized
    from the bank at the start of that day and scaled down if they exceed it (as Lab.RunSingles).
  * Leak-free inputs: features use earlier days only; models for season S are fit on seasons < S;
    the volume cut for season S comes from earlier seasons' edges only.
  * NESTED walk-forward (the honest number): before season S, every GRID config is compounded from
    START over TRAIN_FROM..S-1 (one continuous bankroll); the config with the highest ending bankroll
    whose training bankroll never fell below RUIN wins (ties -> smaller average stake). It is then
    played on season S only, continuing the real bankroll. Season S's results are never seen when
    choosing for S. A "short" variant trains on EVAL seasons only (2023..S-1) with no-bet in the first one.
  * The grid's FAMILIES (full2/full3 models, divisional add-on, volume) were themselves found by earlier
    searches that viewed 2023-2025, so even the nested number carries some optimism.
  * P&L basis "closing_ml" pays posted closing American odds (vig included). Basis "tim_fee_est" pays
    1/(de-vigged prob + 0.02) - 1, Tim's $0.02/contract working estimate, NOT a verified exchange fee.
    Stakes (Kelly) are always sized from the posted odds.
  * Bootstrap: resample whole betting DAYS (daily bank multipliers) with replacement, same number of days
    as the evaluation window, N_SIMS paths, fixed seed. Reports P(end >= $1M), median end, P(min < RUIN).
    It treats days as independent and inherits any selection bias of the strategy it resamples.
"""

import argparse
import csv
import json
import math
import os
from datetime import date
from itertools import groupby

import numpy as np

from Backtest.FrozenConfig import FROZEN
from Backtest.Hunt import FILTERS, Sel
from Backtest.Lab import START, AllRows, AttachModels, Decimal, MLCandidate
from Backtest.Version import CodeVersion, NewRunId
from Backtest.Volume import WalkForwardCut

EVAL_SEASONS = [2023, 2024, 2025, 2026]
TRAIN_FROM = 2015
RUIN = 1_000.0
TARGET = 1_000_000.0
FEE_EST = 0.02
N_SIMS = 5000
SEED = 20261109
PRIOR_SETTINGS = 2073
DEFAULT_OUT = os.path.join("data", "backtest", "compound")


# ----------------------------------------------------------------------------- selectors
def Best(r, model):
    p = r[model]
    return max((MLCandidate(r, "home", p), MLCandidate(r, "away", 1 - p)), key=lambda c: c["edge"])


def SelValue(model, min_edge):
    def factory(rows_by_season, season):
        def f(r):
            c = Best(r, model)
            return [dict(c, rule=f"value_{model}_e{int(round(min_edge * 100))}")] if c["edge"] >= min_edge else []
        return f
    return factory


def SelDiv(min_edge):
    def factory(rows_by_season, season):
        s = Sel("p_full3", min_edge, filt=FILTERS["div"])
        return lambda r: [dict(c, rule=f"div_full3_e{int(round(min_edge * 100))}") for c in s(r)]
    return factory


def SelFrozenUnion(config=None):
    """FrozenConfig's two rules, value first then divisional add-on, one bet per game."""
    cfg = config or FROZEN
    def factory(rows_by_season, season):
        v = SelValue(cfg["value"]["model"], cfg["value"]["min_edge"])(rows_by_season, season)
        d = Sel(cfg["div_addon"]["model"], cfg["div_addon"]["min_edge"], filt=FILTERS[cfg["div_addon"]["filter"]])
        def f(r):
            got = v(r)
            if got:
                return [dict(got[0], rule=cfg["value"]["rule"], tier=0)]
            got = d(r)
            return [dict(got[0], rule=cfg["div_addon"]["rule"], tier=1)] if got else []
        return f
    return factory


def SelVolume(q, model="p_full2"):
    """Tim's volume idea: bet roughly the top q share of games by model edge. The edge cut for season S
    is the (1-q) quantile of edges in the 3 seasons before S (Volume.WalkForwardCut), so no look-ahead."""
    def factory(rows_by_season, season):
        cut = WalkForwardCut(rows_by_season, model, season, q, False)
        def f(r):
            c = Best(r, model)
            return [dict(c, rule=f"volume_q{int(q * 100)}_{model}")] if c["edge"] >= cut else []
        return f
    return factory


def SelMarketFav():
    def factory(rows_by_season, season):
        def f(r):
            side = "home" if r["mkt_home"] >= 0.5 else "away"
            p = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
            return [dict(MLCandidate(r, side, p), rule="market_fav")]
        return f
    return factory


def SelNone():
    def factory(rows_by_season, season):
        return lambda r: []
    return factory


# ----------------------------------------------------------------------------- sizers (fraction of day-start bank)
def SizeKelly(frac, cap):
    def f(c):
        b = Decimal(c["odds"]) - 1
        k = (b * c["p"] - (1 - c["p"])) / b
        return max(0.0, min(k * frac, cap))
    f.label = f"kelly{frac:g}_cap{int(round(cap * 100))}"
    return f


def SizePct(pct):
    def f(c):
        return pct
    f.label = f"flat{pct * 100:g}pct"
    return f


def SizeFrozen(config=None):
    """FrozenConfig sizing: half Kelly capped 10% on the value rule, flat 2% on the divisional add-on."""
    cfg = config or FROZEN
    k = SizeKelly(cfg["value"]["stake"]["fraction"], cfg["value"]["stake"]["cap"])
    def f(c):
        return k(c) if c.get("tier", 0) == 0 else cfg["div_addon"]["stake"]["pct"]
    f.label = "frozen_sizing"
    return f


SELECTORS = {}
for _m in ("p_full2", "p_full3"):
    for _e in (0.04, 0.06, 0.08, 0.10):
        SELECTORS[f"value_{_m}_e{int(_e * 100)}"] = SelValue(_m, _e)
for _e in (0.02, 0.04):
    SELECTORS[f"div_full3_e{int(_e * 100)}"] = SelDiv(_e)
SELECTORS["frozen_union"] = SelFrozenUnion()
SELECTORS["volume_q30_p_full2"] = SelVolume(0.30)
SELECTORS["volume_q20_p_full2"] = SelVolume(0.20)
SELECTORS["market_fav"] = SelMarketFav()

SIZERS = {}
for _f in (0.25, 0.5, 1.0):
    for _c in (0.05, 0.10, 0.20, 1.0):
        s = SizeKelly(_f, _c)
        SIZERS[s.label] = s
for _p in (0.01, 0.02, 0.05, 0.10, 0.20):
    s = SizePct(_p)
    SIZERS[s.label] = s

GRID = [(sel, siz) for sel in SELECTORS for siz in SIZERS]


# ----------------------------------------------------------------------------- engine
def Payout(c, basis):
    if basis == "closing_ml":
        return Decimal(c["odds"]) - 1
    return 1.0 / min(c["p_mkt"] + FEE_EST, 0.999) - 1


def Compound(rows_by_season, seasons, plan, basis="closing_ml", start=START):
    """plan: {season: (selector_factory, sizer)}. One continuous bankroll over `seasons` in order.
    Returns dict(end, daily=[{date, season, bank_start, equity}], bets=[...], min_bank, avg_stake)."""
    bank, daily, bets, min_bank = start, [], [], start
    for season in seasons:
        factory, sizer = plan[season]
        select = factory(rows_by_season, season)
        rows = sorted(rows_by_season.get(season, []), key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"]))
        for day, group in groupby(rows, key=lambda r: r["gameday"]):
            start_bank = bank
            picks = []
            for r in group:
                if r.get("home_ml") is None or r.get("away_ml") is None:
                    continue
                for c in select(r):
                    stake = sizer(c) * start_bank
                    if stake >= 1.0:
                        picks.append((r, c, stake))
            total = sum(s for _, _, s in picks)
            scale = min(1.0, start_bank / total) if total > 0 else 1.0
            for r, c, s in picks:
                s *= scale
                pnl = 0.0 if c["won"] is None else (s * Payout(c, basis) if c["won"] else -s)
                bank += pnl
                bets.append({"date": day, "season": season, "game_id": r["game_id"], "rule": c["rule"],
                             "pick": c["team"], "odds": c["odds"], "p": round(c["p"], 4), "p_mkt": round(c["p_mkt"], 4),
                             "stake_frac": round(s / start_bank, 5) if start_bank > 0 else 0.0,
                             "won": c["won"], "pnl": round(pnl, 2)})
            bank = max(bank, 0.0)
            min_bank = min(min_bank, bank)
            daily.append({"date": day, "season": season, "bank_start": round(start_bank, 2), "equity": round(bank, 2)})
    avg = sum(b["stake_frac"] for b in bets) / len(bets) if bets else 0.0
    return {"end": bank, "daily": daily, "bets": bets, "min_bank": min_bank, "avg_stake": avg}


def Metrics(res, start=START, seasons_elapsed=None):
    daily = res["daily"]
    peak, peak_day, maxdd, longest, cur_start = start, None, 0.0, 0, None
    for d in daily:
        e = d["equity"]
        if e >= peak:
            if cur_start is not None:
                longest = max(longest, (date.fromisoformat(d["date"]) - date.fromisoformat(cur_start)).days)
            peak, peak_day, cur_start = e, d["date"], None
        else:
            if cur_start is None:
                cur_start = peak_day or d["date"]
            maxdd = max(maxdd, 1 - e / peak if peak else 1.0)
    if cur_start is not None and daily:
        longest = max(longest, (date.fromisoformat(daily[-1]["date"]) - date.fromisoformat(cur_start)).days)
    end = res["end"]
    per_season = None
    if seasons_elapsed and end > 0:
        per_season = (end / start) ** (1 / seasons_elapsed) - 1
    wins = sum(b["won"] is True for b in res["bets"])
    done = sum(b["won"] is not None for b in res["bets"])
    return {"end": round(end, 2), "multiple": round(end / start, 4), "growth_per_season": None if per_season is None else round(per_season, 4),
            "max_drawdown_pct": round(-maxdd, 4), "longest_drawdown_days": longest, "min_bank": round(res["min_bank"], 2),
            "ruined": res["min_bank"] < RUIN, "bets": len(res["bets"]), "wins": wins, "losses": done - wins,
            "avg_stake_pct": round(res["avg_stake"] * 100, 2)}


def Drawdowns(daily, start=START):
    peak, out = start, []
    for d in daily:
        peak = max(peak, d["equity"])
        out.append(round(d["equity"] / peak - 1, 4) if peak else 0.0)
    return out


def Bootstrap(daily, n_sims=N_SIMS, seed=SEED, start=START):
    """Resample daily bank multipliers with replacement (same number of days). Days with no bets count (x1)."""
    mult = np.array([d["equity"] / d["bank_start"] if d["bank_start"] > 0 else 1.0 for d in daily])
    if not len(mult):
        return {"n_days": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mult), size=(n_sims, len(mult)))
    paths = start * np.cumprod(mult[idx], axis=1)
    ends, mins = paths[:, -1], np.minimum(paths.min(axis=1), start)
    return {"n_days": int(len(mult)), "n_sims": n_sims, "p_reach_1m": round(float((ends >= TARGET).mean()), 4),
            "p_touch_1m": round(float((paths.max(axis=1) >= TARGET).mean()), 4),
            "median_end": round(float(np.median(ends)), 2), "p5_end": round(float(np.percentile(ends, 5)), 2),
            "p95_end": round(float(np.percentile(ends, 95)), 2), "p_ruin": round(float((mins < RUIN).mean()), 4),
            "p_below_start": round(float((ends < start).mean()), 4)}


def Plan(seasons, sel, siz):
    return {s: (SELECTORS[sel] if isinstance(sel, str) else sel, SIZERS[siz] if isinstance(siz, str) else siz) for s in seasons}


def ChooseConfig(rows_by_season, train_seasons, basis, grid=None):
    """Best GRID config on one continuous bankroll over train_seasons (all strictly before the season played)."""
    grid = grid or GRID
    best, best_key = ("none", "none"), None
    for sel, siz in grid:
        res = Compound(rows_by_season, train_seasons, Plan(train_seasons, sel, siz), basis)
        if res["min_bank"] < RUIN or not res["bets"]:
            continue
        key = (round(res["end"], 2), -res["avg_stake"])
        if best_key is None or key > best_key:
            best, best_key = (sel, siz), key
    return best, (best_key[0] if best_key else None)


def Nested(rows_by_season, eval_seasons, train_from, basis, grid=None, short=False):
    """Choose per season from strictly earlier seasons, then play that season on the continuing bankroll."""
    plan, choices = {}, []
    for s in eval_seasons:
        train = list(range(eval_seasons[0] if short else train_from, s))
        if not train:
            (sel, siz), train_end = ("none", "none"), None
        else:
            (sel, siz), train_end = ChooseConfig(rows_by_season, train, basis, grid)
        plan[s] = (SelNone() if sel == "none" else SELECTORS[sel], SizePct(0.0) if siz == "none" else SIZERS[siz])
        choices.append({"season": s, "train_seasons": [train[0], train[-1]] if train else [], "selector": sel,
                        "sizer": siz, "train_end_bankroll": train_end})
    return Compound(rows_by_season, eval_seasons, plan, basis), choices


# ----------------------------------------------------------------------------- report
def Run(rows_by_season=None, eval_seasons=None, train_from=TRAIN_FROM, out_dir=None, base=None, write=True, n_sims=N_SIMS):
    eval_seasons = eval_seasons or EVAL_SEASONS
    if rows_by_season is None:
        rows_by_season = AllRows(2012, max(eval_seasons))
        AttachModels(rows_by_season)
    n_last = len(rows_by_season.get(eval_seasons[-1], []))
    full = max(len(rows_by_season.get(s, [])) for s in eval_seasons[:-1]) if len(eval_seasons) > 1 else n_last
    seasons_elapsed = (len(eval_seasons) - 1) + (n_last / full if full else 1)
    out = {"run_id": NewRunId(), "code": CodeVersion(), "frozen_config": FROZEN["name"], "eval_seasons": eval_seasons,
           "train_from": train_from, "start": START, "ruin": RUIN, "target": TARGET, "seasons_elapsed": round(seasons_elapsed, 3),
           "settings_this_module": len(GRID), "settings_prior_searches": PRIOR_SETTINGS,
           "settings_total_lower_bound": len(GRID) + PRIOR_SETTINGS, "bases": {}}
    series_rows = []
    for basis in ("closing_ml", "tim_fee_est"):
        b = {}
        def add(name, res, kind, extra=None):
            m = Metrics(res, seasons_elapsed=seasons_elapsed)
            m["bootstrap"] = Bootstrap(res["daily"], n_sims)
            m["kind"] = kind
            if extra:
                m.update(extra)
            b[name] = m
            dd = Drawdowns(res["daily"])
            for d, x in zip(res["daily"], dd):
                series_rows.append({"date": d["date"], "series": name, "equity": d["equity"], "drawdown_pct": x,
                                    "basis": basis, "bank_start": d["bank_start"]})
        nres, choices = Nested(rows_by_season, eval_seasons, train_from, basis)
        add("nested_walkforward", nres, "HONEST: nested walk-forward", {"choices": choices})
        sres, schoices = Nested(rows_by_season, eval_seasons, train_from, basis, short=True)
        add("nested_walkforward_short", sres, "nested, trained on eval seasons only (first season no-bet)", {"choices": schoices})
        add("frozen_compounded", Compound(rows_by_season, eval_seasons, Plan(eval_seasons, "frozen_union", SizeFrozen()), basis),
            "FrozenConfig compounded (2023-25 contaminated)")
        add("baseline_market_fav_2pct", Compound(rows_by_season, eval_seasons, Plan(eval_seasons, "market_fav", "flat2pct"), basis), "baseline")
        add("baseline_no_bet", Compound(rows_by_season, eval_seasons, Plan(eval_seasons, SelNone(), SizePct(0.0)), basis), "baseline")
        grid_rows = []
        for sel, siz in GRID:
            res = Compound(rows_by_season, eval_seasons, Plan(eval_seasons, sel, siz), basis)
            m = Metrics(res, seasons_elapsed=seasons_elapsed)
            m["bootstrap"] = Bootstrap(res["daily"], n_sims)
            m.update(selector=sel, sizer=siz)
            grid_rows.append((m, res))
        grid_rows.sort(key=lambda t: -t[0]["end"])
        best_m, best_res = grid_rows[0]
        add("best_in_sample", best_res, f"OVERFIT: best of {len(GRID)} on the eval window itself",
            {"selector": best_m["selector"], "sizer": best_m["sizer"]})
        for sel, siz in [("frozen_union", "kelly0.25_cap5"), ("frozen_union", "kelly0.5_cap10"), ("frozen_union", "kelly1_cap20"),
                         ("frozen_union", "kelly1_cap100"), ("frozen_union", "flat2pct"), ("frozen_union", "flat5pct"),
                         ("volume_q30_p_full2", "flat20pct"), ("volume_q30_p_full2", "flat2pct")]:
            m, res = next(t for t in grid_rows if t[0]["selector"] == sel and t[0]["sizer"] == siz)
            add(f"{sel}__{siz}", res, "sizing comparison (in-sample window)", {"selector": sel, "sizer": siz})
        b["grid"] = [m for m, _ in grid_rows]
        out["bases"][basis] = b
    # what the nested procedure would pick next (trained through the latest data) -> candidate only
    nxt, nxt_end = ChooseConfig(rows_by_season, list(range(train_from, eval_seasons[-1] + 1)), "closing_ml")
    out["next_season_pick_closing_ml"] = {"selector": nxt[0], "sizer": nxt[1], "train_end_bankroll": nxt_end,
                                          "train_seasons": [train_from, eval_seasons[-1]]}
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or "compound"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "series", "equity", "drawdown_pct", "basis", "bank_start"])
            w.writeheader()
            w.writerows(series_rows)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return out


def Main():
    ap = argparse.ArgumentParser(description="Continuous compounding bankroll: nested walk-forward, frozen, overfit best, bootstrap.")
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    a = ap.parse_args()
    out = Run(out_dir=a.out_dir, base=a.base, n_sims=a.sims)
    for basis, b in out["bases"].items():
        print(f"== {basis}")
        for k, m in b.items():
            if k == "grid":
                continue
            bs = m["bootstrap"]
            print(f"{k:40s} end {m['end']:>14,.0f} dd {m['max_drawdown_pct']*100:6.1f}% bets {m['bets']:4d}"
                  f" P1M {bs.get('p_reach_1m')} med {bs.get('median_end', 0):,.0f} Pruin {bs.get('p_ruin')}")
    print(json.dumps(out["next_season_pick_closing_ml"]))


if __name__ == "__main__":
    Main()
