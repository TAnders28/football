r"""
High-volume moneyline lab: bet ~30% of games at ~20% of the bankroll per bet.

    .venv\Scripts\python.exe -m Backtest.Volume          # writes data/backtest/reports/volume_leaderboard.csv + volume_results.json

Same fairness rules as Backtest.Lab (reuses its rows + walk-forward models):
  * features from earlier days only, models fit on earlier seasons only, closing moneylines only;
  * every strategy starts every season at $10,000 (no carry-over), stakes are sized from the bankroll at
    the start of the gameday, and if a day's stakes exceed the bankroll they are scaled down;
  * TUNE 2015-2022 picks the rule, TEST 2023-2025 is reported out-of-sample, 2026 partial.

How a rule hits ~30% volume without look-ahead
  * wfq  (walk-forward quantile): for season S the edge cut-off is the (1-q) quantile of the model's
    best-side edge over seasons S-3..S-1. Any game at/above it is bet. Deterministic, one run.
  * day  (daily top-q): on each gameday rank that day's games by model edge (all pre-kickoff info) and
    bet the top q*n. The fractional part is rounded at random (so a 1-game Thursday is bet ~30% of the
    time), which makes the rule stochastic -> run over SEEDS and average.
  * rand (random 30% baseline): bet a random q of games on the home / favourite side. Averaged over SEEDS.
  * fav  suffix: only ever bet the market favourite (the model only chooses WHICH favourites).
"""

import csv
import json
import math
import os
import random
import statistics
from itertools import groupby

from Backtest.Lab import (START, TUNE, TEST, PARTIAL, REPORT_DIR, AllRows, AttachModels, Decimal, Flat, Kelly,
                          MLCandidate, RunSingles, SelAlwaysHome, SelEdge)

SEASONS = TUNE + TEST + PARTIAL
SEEDS = list(range(20))
MODELS = ("p_full2", "p_full3", "p_full", "p_calib", "p_sprdemp", "p_margin")


# ----------------------------------------------------------------------------- picking
def Best(r, model, favs_only=False):
    p = r[model]
    cands = [MLCandidate(r, "home", p), MLCandidate(r, "away", 1 - p)]
    if favs_only:
        cands = [c for c in cands if not c["dog"]] or cands
    return max(cands, key=lambda c: c["edge"])


def WalkForwardCut(rows_by_season, model, season, q, favs_only):
    hist = [Best(r, model, favs_only)["edge"] for s in range(max(2014, season - 3), season)
            for r in rows_by_season.get(s, [])]
    if not hist:
        return float("inf")
    hist.sort()
    return hist[min(len(hist) - 1, int(math.floor((1 - q) * len(hist))))]


def PickerWFQ(cut, model, favs_only):
    def f(day, rng):
        out = []
        for r in day:
            c = Best(r, model, favs_only)
            if c["edge"] >= cut:
                out.append((r, c))
        return out
    return f


def PickerDaily(q, model, favs_only):
    def f(day, rng):
        k = q * len(day)
        k = int(k) + (1 if rng.random() < k - int(k) else 0)
        ranked = sorted(((r, Best(r, model, favs_only)) for r in day), key=lambda x: -x[1]["edge"])
        return ranked[:k]
    return f


def PickerRandom(q, side):
    def f(day, rng):
        out = []
        for r in day:
            if rng.random() < q:
                if side == "home":
                    s = "home"
                else:
                    s = "home" if r["mkt_home"] >= 0.5 else "away"
                p = r["mkt_home"] if s == "home" else 1 - r["mkt_home"]
                out.append((r, MLCandidate(r, s, p)))
        return out
    return f


# ----------------------------------------------------------------------------- engine
def RunDays(rows, picker, stake_pct, seed=0):
    rng = random.Random(seed)
    bank, peak, maxdd, bets = START, START, 0.0, []
    for _d, day in groupby(rows, key=lambda r: r["gameday"]):
        day = list(day)
        start_bank = bank
        picks = picker(day, rng)
        stakes = [start_bank * stake_pct for _ in picks]
        total = sum(stakes)
        scale = min(1.0, start_bank / total) if total > 0 else 1.0
        for (r, c), s in zip(picks, stakes):
            s *= scale
            if s < 1.0:
                continue
            pnl = 0.0 if c["won"] is None else (s * (Decimal(c["odds"]) - 1) if c["won"] else -s)
            bank += pnl
            bets.append({"game_id": r["game_id"], "gameday": r["gameday"], "pick": c["team"], "odds": c["odds"],
                         "p_model": round(c["p"], 4), "p_mkt": round(c["p_mkt"], 4), "edge": round(c["edge"], 4),
                         "dog": c["dog"], "stake": round(s, 2), "won": c["won"], "pnl": round(pnl, 2),
                         "bankroll": round(bank, 2)})
        peak = max(peak, bank)
        maxdd = max(maxdd, (peak - bank) / peak if peak else 0)
    staked = sum(b["stake"] for b in bets)
    return {"end": bank, "profit": bank - START, "bets": len(bets), "games": len(rows),
            "vol": len(bets) / max(1, len(rows)), "staked": staked, "wins": sum(b["won"] is True for b in bets),
            "maxdd": maxdd, "bust": bank < 100, "rows": bets}


def Agg(runs):
    """Average a list of per-seed season results."""
    ends = [x["end"] for x in runs]
    return {"end": statistics.mean(ends), "end_median": statistics.median(ends), "end_min": min(ends),
            "end_max": max(ends), "profit": statistics.mean(ends) - START,
            "bets": statistics.mean(x["bets"] for x in runs), "games": runs[0]["games"],
            "vol": statistics.mean(x["vol"] for x in runs), "staked": statistics.mean(x["staked"] for x in runs),
            "wins": statistics.mean(x["wins"] for x in runs),
            "maxdd": statistics.mean(x["maxdd"] for x in runs), "maxdd_worst": max(x["maxdd"] for x in runs),
            "bust_rate": sum(x["bust"] for x in runs) / len(runs), "n_runs": len(runs),
            "seed_ends": [round(e, 2) for e in ends]}


# ----------------------------------------------------------------------------- grid
def Evaluate(rows_by_season):
    res = {}
    for model in MODELS:
        for favs in (False, True):
            for q in (0.25, 0.30, 0.35):
                for st in (0.15, 0.20, 0.25):
                    tag = f"{model[2:]}{'_fav' if favs else ''}_q{int(q*100)} | {int(st*100)}%bank"
                    per = {}
                    for s in SEASONS:
                        rows = rows_by_season.get(s) or []
                        cut = WalkForwardCut(rows_by_season, model, s, q, favs)
                        r = RunDays(rows, PickerWFQ(cut, model, favs), st)
                        r["cut"] = cut
                        r.update({"n_runs": 1, "bust_rate": float(r["bust"])})
                        per[s] = r
                    res["wfq_" + tag] = per
                    per = {}
                    for s in SEASONS:
                        rows = rows_by_season.get(s) or []
                        per[s] = Agg([RunDays(rows, PickerDaily(q, model, favs), st, seed) for seed in SEEDS])
                    res["day_" + tag] = per
    for side in ("home", "fav"):
        for q in (0.30,):
            for st in (0.15, 0.20, 0.25):
                per = {s: Agg([RunDays(rows_by_season[s], PickerRandom(q, side), st, seed) for seed in SEEDS])
                       for s in SEASONS}
                res[f"rand_{side}_q{int(q*100)} | {int(st*100)}%bank"] = per
    # sparse / all-games baselines from Lab (single deterministic run)
    base = {"BASE value_full2_edge8 | half-Kelly cap10% (sparse)": (SelEdge("p_full2", 0.08), Kelly(0.5, 0.10)),
            "BASE value_full2_edge8 | full-Kelly cap20% (sparse)": (SelEdge("p_full2", 0.08), Kelly(1.0, 0.20)),
            "BASE always_home | flat$200": (SelAlwaysHome, Flat(200))}
    for name, (sel, sz) in base.items():
        per = {}
        for s in SEASONS:
            r = RunSingles(rows_by_season[s], sel, sz)
            r.update({"games": len(rows_by_season[s]), "vol": r["bets"] / len(rows_by_season[s]),
                      "n_runs": 1, "bust_rate": float(r["end"] < 100)})
            per[s] = r
        res[name] = per
    per = {}
    for s in SEASONS:
        r = RunDays(rows_by_season[s], lambda day, rng: [(x, MLCandidate(x, "home", x["mkt_home"])) for x in day], 0.20)
        r.update({"n_runs": 1, "bust_rate": float(r["bust"])})
        per[s] = r
    res["BASE always_home | 20%bank (100% volume)"] = per
    return res


def Board(res):
    board = []
    for name, per in res.items():
        tune = [per[s] for s in TUNE]
        test = [per[s] for s in TEST]
        board.append({
            "strategy": name,
            "tune_avg_end": statistics.mean(t["end"] for t in tune), "tune_up": sum(t["end"] > START for t in tune),
            "tune_vol": sum(t["bets"] for t in tune) / sum(t["games"] for t in tune),
            **{f"end_{s}": per[s]["end"] for s in TEST + PARTIAL},
            **{f"median_{s}": per[s].get("end_median", per[s]["end"]) for s in TEST},
            **{f"bets_{s}": per[s]["bets"] for s in TEST},
            "test_vol": sum(t["bets"] for t in test) / sum(t["games"] for t in test),
            "test_profit": sum(t["profit"] for t in test),
            "test_up": sum(t["end"] > START for t in test),
            "test_maxdd_avg": statistics.mean(t["maxdd"] for t in test),
            "test_maxdd_worst": max(t.get("maxdd_worst", t["maxdd"]) for t in test),
            "test_bust_rate": statistics.mean(t["bust_rate"] for t in test),
            "tune_bust_rate": statistics.mean(t["bust_rate"] for t in tune),
            "n_runs": per[TEST[0]]["n_runs"],
        })
    return board


def Main():
    rows_by_season = AllRows()
    AttachModels(rows_by_season)
    res = Evaluate(rows_by_season)
    board = Board(res)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "volume_leaderboard.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(board[0]))
        w.writeheader()
        w.writerows(sorted(board, key=lambda b: -b["tune_avg_end"]))
    with open(os.path.join(REPORT_DIR, "volume_results.json"), "w", encoding="utf-8") as fh:
        json.dump({n: {s: {k: v for k, v in r.items() if k != "rows"} for s, r in per.items()} for n, per in res.items()},
                  fh, indent=1)
    return rows_by_season, res, board


def Line(b):
    return (f"{b['strategy'][:52]:52s} tune {b['tune_avg_end']:8,.0f} up{b['tune_up']}/8 vol{b['tune_vol']*100:3.0f}% bust{b['tune_bust_rate']*100:3.0f}% |"
            f" 23 {b['end_2023']:8,.0f} 24 {b['end_2024']:8,.0f} 25 {b['end_2025']:8,.0f} test {b['test_profit']:+8,.0f}"
            f" up{b['test_up']} vol{b['test_vol']*100:3.0f}% dd{b['test_maxdd_avg']*100:3.0f}/{b['test_maxdd_worst']*100:3.0f}%"
            f" bust{b['test_bust_rate']*100:3.0f}% 26p {b['end_2026']:8,.0f}")


if __name__ == "__main__":
    _rows, _res, board = Main()
    print(f"{len(board)} strategies")
    print("## by TUNE avg end (selection, no hindsight)")
    for b in sorted(board, key=lambda b: -b["tune_avg_end"])[:30]:
        print(Line(b))
    print("## Tim spec q30 | 20%bank, by tune")
    for b in sorted([b for b in board if "q30 | 20%" in b["strategy"]], key=lambda b: -b["tune_avg_end"]):
        print(Line(b))
    print("## baselines")
    for b in board:
        if b["strategy"].startswith("BASE"):
            print(Line(b))
    print("## by TEST profit (hindsight)")
    for b in sorted(board, key=lambda b: -b["test_profit"])[:15]:
        print(Line(b))