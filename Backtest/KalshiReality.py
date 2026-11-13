r"""
Research: the FROZEN rule (FrozenConfig, unchanged) priced at Kalshi's EXECUTABLE ask at T-90 / T-15 minutes before
kickoff instead of the nflverse closing line. 2025-2026 only (all the Kalshi history there is). Simulation, no orders.

    .venv\Scripts\python.exe -m Backtest.KalshiReality --out-dir DIR --base NAME

Inputs: data/kalshi/nflgame_candles_<season>.jsonl (DataPull.PullKalshiHistory), schedules, walk-forward models.

Quote at T: the last 1-minute candle closing at or before T (and no older than STALE_MIN) with both yes_bid and yes_ask.
Kalshi candles carry no size, so fills are assumed at the top-of-book ask for the whole stake (optimistic for big stakes).

Model variants (both use the frozen value + divisional rules and sizing):
  asis    p_full2 / p_full3 exactly as walk-forward-fitted (their market inputs are the nflverse closing line, which is
          NOT known at T-90 - a mild look-ahead, kept as the like-for-like comparison)
  kmid    same fitted weights, but mkt_logit (and the terms built from it) recomputed from Kalshi's de-vigged mid at T
          (closing spread is still used in spread_gap - residual look-ahead noted in the report)
Fee bases: tim_est_0.02 (Tim's $0.02/contract working estimate) and kalshi_verified (taker fee 0.07 x P x (1-P) per
contract, KXNFLGAME multiplier 1, https://kalshi.com/docs/kalshi-fee-schedule.pdf; centicent round-up ignored).
Value rule: best side with p_model - ask >= 0.08, stake = start-of-day bank x min(0.5 x Kelly, 0.10), Kelly from the
executable payout 1/(ask+fee)-1. Else divisional add-on: div game, p_full3 - ask >= 0.02, 2% of bank.
Ties pay $0.50 per contract (Kalshi terms). $10k bank, reset each season.
Closing-line comparison: Backtest.ForwardWalk.FrozenPicker/Walk on the same games (closing_ml and tim_fee_est bases).
Randomized paths (Backtest.PathSim, seeded): each series' daily returns (2025 then 2026 wk1-9) re-ordered by random start,
block bootstrap (4 game days) and shuffle, with and without restarts, on one continuous $10k bank; horizons = one season (2025 game-day count), the real
window and 10 seasons (10 x the 2025 game-day count). Real-sequence result kept alongside.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import groupby

import numpy as np

from Backtest.ForwardWalk import FrozenPicker, Kickoff as FwKickoff, SeenBySearch, Split, Walk
from Backtest.FrozenConfig import FROZEN
from Backtest.Lab import (AllRows, AttachModels, EmpProb, EmpiricalSpreadTable, FitLogit, Sig, Vec, WIN_FEATS2, WIN_FEATS3)
from Backtest.PathSim import DailyReturns, Evaluate
from Backtest.Version import ModelVersion, NewRunId

SEASONS = [2025, 2026]
TIMES = {"T-90": 90, "T-15": 15}
STALE_MIN = 30
START = 10_000.0
FEE_EST = 0.02


def KalshiFee(price):
    return 0.07 * price * (1 - price)


FEES = {"tim_est_0.02": lambda a: FEE_EST, "kalshi_verified": KalshiFee}


def LoadQuotes(season):
    path = os.path.join("data", "kalshi", f"nflgame_candles_{season}.jsonl")
    out = defaultdict(dict)   # game_id -> side_team -> candles
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["game_id"]][r["side_team"]] = {"ko": datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00")),
                                                 "candles": sorted(r["candles"], key=lambda c: c["end_ts"])}
    return out


def QuoteAt(entry, minutes_before):
    if not entry:
        return None
    target = (entry["ko"] - timedelta(minutes=minutes_before)).timestamp()
    best = None
    for c in entry["candles"]:
        if c["end_ts"] > target:
            break
        if c["yes_ask_close"] is not None and c["yes_bid_close"] is not None and 0 < c["yes_ask_close"] < 1:
            best = c
    if best is None or target - best["end_ts"] > STALE_MIN * 60:
        return None
    return {"bid": best["yes_bid_close"], "ask": best["yes_ask_close"], "age_min": round((target - best["end_ts"]) / 60, 1)}


def RefitWeights(rbs, s):
    """Exactly Lab.AttachModels' fit for season s (returns w_full2, w_full3 and the spread helpers)."""
    train = [r for t, rs in rbs.items() if 2013 <= t < s for r in rs]
    win = [r for r in train if r["home_win"] is not None]
    w_sp = FitLogit([Vec(r, ["spread"]) for r in win], [r["home_win"] for r in win])
    for r in train + rbs[s]:
        r["spread_logit"] = float(np.dot(w_sp, Vec(r, ["spread"])))
    table = EmpiricalSpreadTable(train)
    for r in train + rbs[s]:
        pe = EmpProb(table, r["spread"])
        r["spread_emp_logit"] = math.log(pe / (1 - pe))
    w3 = FitLogit([Vec(r, WIN_FEATS3) for r in win], [r["home_win"] for r in win], l2=5.0)
    w2 = FitLogit([Vec(r, WIN_FEATS2) for r in win], [r["home_win"] for r in win], l2=5.0)
    return w2, w3


def KalshiPick(r, p2, p3, q, fee_fn, bank):
    """Frozen rules on executable asks. q = {'home': quote, 'away': quote}."""
    cands = []
    for side, p in (("home", p2), ("away", 1 - p2)):
        a = q[side]["ask"]
        cands.append((p - a, side, p, a))
    edge, side, p, a = max(cands)
    if edge >= FROZEN["value"]["min_edge"]:
        b = 1.0 / min(a + fee_fn(a), 0.999) - 1
        k = (b * p - (1 - p)) / b
        stake = max(0.0, min(k * FROZEN["value"]["stake"]["fraction"], FROZEN["value"]["stake"]["cap"])) * bank
        return {"rule": FROZEN["value"]["rule"], "side": side, "p": p, "ask": a, "edge": edge, "b": b}, stake
    if r.get("div"):
        cands = [(pp - q[sd]["ask"], sd, pp, q[sd]["ask"]) for sd, pp in (("home", p3), ("away", 1 - p3))]
        edge, side, p, a = max(cands)
        if edge >= FROZEN["div_addon"]["min_edge"]:
            b = 1.0 / min(a + fee_fn(a), 0.999) - 1
            return {"rule": FROZEN["div_addon"]["rule"], "side": side, "p": p, "ask": a, "edge": edge, "b": b}, bank * FROZEN["div_addon"]["stake"]["pct"]
    return None


def KalshiWalk(rows, tkey, variant, fee_name, fee_fn=None):
    fee_fn = fee_fn or FEES[fee_name]
    rows = sorted(rows, key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"]))
    bank, peak, bets, daily = START, START, [], []
    for day, grp in groupby(rows, key=lambda r: r["gameday"]):
        grp = list(grp)
        start_bank = bank
        picks = []
        for r in grp:
            q = r["kq"][tkey]
            p2, p3 = (r["p_full2"], r["p_full3"]) if variant == "asis" else (r[f"p2_kmid_{tkey}"], r[f"p3_kmid_{tkey}"])
            pk = KalshiPick(r, p2, p3, q, fee_fn, start_bank)
            if pk and pk[1] >= 1.0:
                picks.append((r, *pk))
        total = sum(s for _, _, s in picks)
        scale = min(1.0, start_bank / total) if total > 0 else 1.0
        for r, c, s in picks:
            s *= scale
            hw = r["home_win"]
            if hw is None:
                won, pnl = None, s * (0.5 * (1 + c["b"]) - 1)    # tie pays $0.50 per contract
            else:
                won = (hw == 1) == (c["side"] == "home")
                pnl = s * c["b"] if won else -s
            bank += pnl
            bets.append({"date": day, "game_id": r["game_id"], "week": r["week"], "season": r["season"],
                         "matchup": f"{r['away_team']} @ {r['home_team']}", "rule": c["rule"], "side": c["side"],
                         "team": r[f"{c['side']}_team"], "p_model": round(c["p"], 4), "ask": c["ask"],
                         "edge_vs_ask": round(c["edge"], 4), "fee_per_contract": round(fee_fn(c["ask"]), 4),
                         "stake": round(s, 2), "won": won, "pnl": round(pnl, 2), "bankroll": round(bank, 2),
                         "seen_by_search": SeenBySearch(r)})
        peak = max(peak, bank)
        daily.append({"date": day, "equity": round(bank, 2), "drawdown_pct": round(bank / peak - 1, 4)})
    return bets, daily


def Summ(bets, daily):
    s = Split(bets)
    s["end_bank"] = daily[-1]["equity"] if daily else START
    s["max_drawdown"] = min((d["drawdown_pct"] for d in daily), default=0.0)
    return s


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "kalshi"))
    ap.add_argument("--base", default="kalshi_reality")
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    run_id = NewRunId()
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    check = {}
    for s in SEASONS:
        base_p2 = {r["game_id"]: r["p_full2"] for r in rbs[s]}
        w2, w3 = RefitWeights(rbs, s)
        quotes = LoadQuotes(s)
        diffs = []
        for r in rbs[s]:
            diffs.append(abs(float(Sig(np.dot(w2, Vec(r, WIN_FEATS2)))) - base_p2[r["game_id"]]))
            g = quotes.get(r["game_id"], {})
            r["kq"] = {}
            for tkey, mins in TIMES.items():
                qh, qa = QuoteAt(g.get(r["home_team"]), mins), QuoteAt(g.get(r["away_team"]), mins)
                if not (qh and qa):
                    r["kq"][tkey] = None
                    continue
                r["kq"][tkey] = {"home": qh, "away": qa}
                mh, ma = (qh["bid"] + qh["ask"]) / 2, (qa["bid"] + qa["ask"]) / 2
                pk = min(max(mh / (mh + ma), 1e-4), 1 - 1e-4)
                r[f"kmid_{tkey}"] = pk
                saved = r["mkt_logit"]
                r["mkt_logit"] = math.log(pk / (1 - pk))
                r[f"p2_kmid_{tkey}"] = float(Sig(np.dot(w2, Vec(r, WIN_FEATS2))))
                r[f"p3_kmid_{tkey}"] = float(Sig(np.dot(w3, Vec(r, WIN_FEATS3))))
                r["mkt_logit"] = saved
        check[s] = {"refit_max_abs_diff_vs_attachmodels": round(max(diffs), 8)}

    sets = {"2025 (contaminated)": [r for r in rbs[2025]],
            "2026 wk1-9 (cleaner)": [r for r in rbs[2026] if int(r["week"]) <= 9]}
    results, tidy, all_bets, sanity, dailies = {}, [], [], {}, {}
    for sname, rows in sets.items():
        rows = [r for r in rows if r["home_win"] is not None or r["result"] == 0]
        info = {"games": len(rows), "seen_by_search": sum(SeenBySearch(r) for r in rows)}
        for tkey in TIMES:
            have = [r for r in rows if r["kq"].get(tkey)]
            info[f"with_quotes_{tkey}"] = len(have)
            d = [abs(r[f"kmid_{tkey}"] - r["mkt_home"]) for r in have]
            spreads = [r["kq"][tkey][sd]["ask"] - r["kq"][tkey][sd]["bid"] for r in have for sd in ("home", "away")]
            ages = [r["kq"][tkey][sd]["age_min"] for r in have for sd in ("home", "away")]
            info[f"mean_abs_kalshi_mid_minus_nflverse_close_{tkey}"] = round(sum(d) / len(d), 4) if d else None
            info[f"median_spread_{tkey}"] = sorted(spreads)[len(spreads) // 2] if spreads else None
            info[f"median_quote_age_min_{tkey}"] = sorted(ages)[len(ages) // 2] if ages else None
            res = {}
            for basis in ("closing_ml", "tim_fee_est"):
                b, dly = Walk(have, FrozenPicker(), basis)
                res[f"closing_line|{basis}"] = Summ(b, dly)
                dailies[(sname, tkey, f"closing_line|{basis}")] = dly
                for x in b:
                    all_bets.append({"set": sname, "time": tkey, "series": f"closing_line|{basis}", **{k: x[k] for k in ("date", "game_id", "rule", "side", "stake", "won", "pnl", "bankroll")}})
                tidy += [{"date": dd["date"], "series": f"closing_line|{basis}", "set": sname, "time": tkey, **{k: dd[k] for k in ("equity", "drawdown_pct")}} for dd in dly]
            for variant in ("asis", "kmid"):
                for fee in FEES:
                    b, dly = KalshiWalk(have, tkey, variant, fee)
                    key = f"kalshi_ask|{variant}|{fee}"
                    res[key] = Summ(b, dly)
                    dailies[(sname, tkey, key)] = dly
                    for x in b:
                        all_bets.append({"set": sname, "time": tkey, "series": key, **{k: x[k] for k in ("date", "game_id", "rule", "side", "stake", "won", "pnl", "bankroll")}})
                    tidy += [{"date": dd["date"], "series": key, "set": sname, "time": tkey, **{k: dd[k] for k in ("equity", "drawdown_pct")}} for dd in dly]
            results[f"{sname}|{tkey}"] = res
        # closing-line result on ALL games of the set (not only those with Kalshi quotes)
        for basis in ("closing_ml", "tim_fee_est"):
            b, dly = Walk(rows, FrozenPicker(), basis)
            info[f"closing_line_all_games|{basis}"] = Summ(b, dly)
        sanity[sname] = info

    # Randomized paths on ONE continuous $10k bank (2025 then 2026 wk1-9 daily returns; PathSim)
    s25, s26 = list(sets)
    paths_out, path_tidy = {}, []
    for (sname, tkey, key), dly in dailies.items():
        if sname != s25:
            continue
        rets = DailyReturns(dly) + DailyReturns(dailies[(s26, tkey, key)])
        n25 = len(dly)
        ev = Evaluate(rets, seed=a.seed, paths=a.paths, horizons={"one_season": n25, "window": len(rets), "10_seasons": 10 * n25})
        paths_out[f"{tkey}|{key}"] = ev
        for name, sm in [("real_sequence", ev["real_sequence"])] + list(ev["randomized"].items()):
            path_tidy.append({"series": f"{tkey}|{key}", "path_set": name, **{k: sm.get(k) for k in
                              ("median_final", "p10_final", "p90_final", "p_ruin", "p_1m", "median_days_to_ruin", "mean_restarts", "p_final_below_start")}})
    report = {"run_id": run_id, "model_version": ModelVersion(), "created_at": datetime.now(timezone.utc).isoformat(),
              "refit_check": check, "coverage_and_sanity": sanity, "results": results,
              "settings_counted": 8, "stale_min": STALE_MIN, "path_sim": paths_out, "path_sim_seed": a.seed, "path_sim_paths": a.paths,
              "fee_bases": {"tim_est_0.02": "Tim's working estimate, $0.02 per contract",
                            "kalshi_verified": "0.07 x P x (1-P) per contract (taker), KXNFLGAME multiplier 1, Kalshi fee schedule PDF last updated/effective 2026-07-07"}}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "series", "set", "time", "equity", "drawdown_pct"])
        w.writeheader(); w.writerows(tidy)
    with open(os.path.join(a.out_dir, f"{a.base}_paths.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(path_tidy[0].keys()))
        w.writeheader(); w.writerows(path_tidy)
    if all_bets:
        with open(os.path.join(a.out_dir, f"{a.base}_bets.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_bets[0].keys()))
            w.writeheader(); w.writerows(all_bets)
    brief = {k: {kk: {m: vv[m] for m in ("bets", "wins", "pnl", "roi_staked", "end_bank", "max_drawdown")} for kk, vv in v.items()} for k, v in results.items()}
    print(json.dumps({"refit_check": check, "sanity": sanity, "results": brief}, indent=1, default=str))


if __name__ == "__main__":
    Main()