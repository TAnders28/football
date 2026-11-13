r"""
Research: the FROZEN rule (unchanged) priced at Polymarket's public price history at T-90 / T-15 before kickoff.
Fill assumption (Tim's choice, 2026-11-13): the market's middle price is the fill price. Polymarket /prices-history
gives one price per token, used as that mid; the other team's price is 1 - p. Simulation only, no orders.

    .venv\Scripts\python.exe -m Backtest.PolyReality [--out-dir DIR --base NAME --paths 5000 --seed 20261113]

Same machinery as Backtest.KalshiReality (frozen value rule + divisional add-on, Kelly from the payout
1/(price+fee)-1, ties pay $0.50, $10k reset per season, closing-line comparison via ForwardWalk on the same games):
  asis  walk-forward p_full2 / p_full3 as fitted (market input = nflverse closing line; mild look-ahead at T)
  kmid  same weights with the market input replaced by the Polymarket price at T ("market-mid model")
Fee bases: tim_est_0.02 (Tim's $0.02/contract estimate) and polymarket_sports_0.05 (taker fee 0.05 x p x (1-p) per share,
https://docs.polymarket.com/trading/fees; UNCERTAIN for NFL: polymarket.com/fees listed NFL at 0.03 for Oct 2026,
Polymarket US uses a different tiered rate, and sports markets were fee-free before 2026).
Price at T: the last history point at or before T and no older than STALE_MIN.
PathSim: daily returns 2024 -> 2025 -> 2026 wk1-9 on one continuous $10k bank (random start, block bootstrap, shuffle,
with/without restarts), horizons one season (2025 game days), the real window, 10 seasons.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np

from Backtest.ForwardWalk import FrozenPicker, SeenBySearch, Walk
from Backtest.KalshiReality import FEES, KalshiWalk, RefitWeights, Summ
from Backtest.Lab import AllRows, AttachModels, Sig, Vec, WIN_FEATS2, WIN_FEATS3
from Backtest.PathSim import DailyReturns, Evaluate
from Backtest.Version import ModelVersion, NewRunId

SEASONS = [2024, 2025, 2026]
TIMES = {"T-90": 90, "T-15": 15}
STALE_MIN = 30
POLY_RATE = 0.05
POLY_FEES = {"tim_est_0.02": FEES["tim_est_0.02"], "polymarket_sports_0.05": lambda p: POLY_RATE * p * (1 - p)}


def LoadHistory(season):
    path = os.path.join("data", "prices", f"polymarket_history_{season}.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[r["game_id"]] = r
    return out


def PriceAt(row, minutes_before, stale_min=STALE_MIN):
    """(home_price, away_price, age_min) from the last point at or before kickoff - minutes_before, or None."""
    if not row or not row.get("history"):
        return None
    ko = datetime.fromisoformat(row["kickoff_utc"].replace("Z", "+00:00"))
    target = (ko - timedelta(minutes=minutes_before)).timestamp()
    pts = [x for x in row["history"] if x["t"] <= target]
    if not pts:
        return None
    last = max(pts, key=lambda x: x["t"])
    if target - last["t"] > stale_min * 60 or not 0 < last["p"] < 1:
        return None
    p0 = last["p"]
    home = p0 if row["token0_team"] == row["home_team"] else 1 - p0
    return home, 1 - home, round((target - last["t"]) / 60, 1)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "polymarket"))
    ap.add_argument("--base", default="poly_reality")
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    run_id = NewRunId()
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    for s in SEASONS:
        w2, w3 = RefitWeights(rbs, s)
        hist = LoadHistory(s)
        for r in rbs[s]:
            r["kq"] = {}
            for tkey, mins in TIMES.items():
                q = PriceAt(hist.get(r["game_id"]), mins)
                if not q:
                    r["kq"][tkey] = None
                    continue
                ph, pa, age = q
                r["kq"][tkey] = {"home": {"bid": ph, "ask": ph, "age_min": age}, "away": {"bid": pa, "ask": pa, "age_min": age}}
                pk = min(max(ph, 1e-4), 1 - 1e-4)
                r[f"kmid_{tkey}"] = pk
                saved = r["mkt_logit"]
                r["mkt_logit"] = math.log(pk / (1 - pk))
                r[f"p2_kmid_{tkey}"] = float(Sig(np.dot(w2, Vec(r, WIN_FEATS2))))
                r[f"p3_kmid_{tkey}"] = float(Sig(np.dot(w3, Vec(r, WIN_FEATS3))))
                r["mkt_logit"] = saved

    sets = {"2024 (contaminated)": rbs[2024], "2025 (contaminated)": rbs[2025],
            "2026 wk1-9 (cleaner)": [r for r in rbs[2026] if int(r["week"]) <= 9]}
    results, sanity, dailies, tidy, all_bets = {}, {}, {}, [], []
    for sname, rows in sets.items():
        rows = [r for r in rows if r["home_win"] is not None or r["result"] == 0]
        info = {"games": len(rows), "seen_by_search": sum(SeenBySearch(r) for r in rows),
                "with_history": sum(1 for r in rows if any(r["kq"].values()))}
        for tkey in TIMES:
            have = [r for r in rows if r["kq"].get(tkey)]
            info[f"with_price_{tkey}"] = len(have)
            d = [abs(r[f"kmid_{tkey}"] - r["mkt_home"]) for r in have]
            ages = [r["kq"][tkey]["home"]["age_min"] for r in have]
            info[f"mean_abs_poly_minus_nflverse_close_{tkey}"] = round(sum(d) / len(d), 4) if d else None
            info[f"median_price_age_min_{tkey}"] = sorted(ages)[len(ages) // 2] if ages else None
            res = {}
            for basis in ("closing_ml", "tim_fee_est"):
                b, dly = Walk(have, FrozenPicker(), basis)
                key = f"closing_line|{basis}"
                res[key], dailies[(sname, tkey, key)] = Summ(b, dly), dly
                tidy += [{"date": x["date"], "series": key, "set": sname, "time": tkey, "equity": x["equity"], "drawdown_pct": x["drawdown_pct"]} for x in dly]
            for variant in ("asis", "kmid"):
                for fee, fn in POLY_FEES.items():
                    b, dly = KalshiWalk(have, tkey, variant, fee, fee_fn=fn)
                    key = f"poly_mid|{variant}|{fee}"
                    res[key], dailies[(sname, tkey, key)] = Summ(b, dly), dly
                    all_bets += [{"set": sname, "time": tkey, "series": key, **{k: x[k] for k in ("date", "game_id", "rule", "team", "ask", "stake", "won", "pnl", "bankroll")}} for x in b]
                    tidy += [{"date": x["date"], "series": key, "set": sname, "time": tkey, "equity": x["equity"], "drawdown_pct": x["drawdown_pct"]} for x in dly]
            results[f"{sname}|{tkey}"] = res
        sanity[sname] = info

    names = list(sets)
    paths_out, path_tidy = {}, []
    for (sname, tkey, key), dly in dailies.items():
        if sname != names[0]:
            continue
        seq = [dailies[(n, tkey, key)] for n in names]
        rets = [x for dl in seq for x in DailyReturns(dl)]
        n_season = len(seq[1]) or len(seq[0])
        if not rets:
            continue
        ev = Evaluate(rets, seed=a.seed, paths=a.paths, horizons={"one_season": n_season, "window": len(rets), "10_seasons": 10 * n_season})
        paths_out[f"{tkey}|{key}"] = ev
        for nm, sm in [("real_sequence", ev["real_sequence"])] + list(ev["randomized"].items()):
            path_tidy.append({"series": f"{tkey}|{key}", "path_set": nm, **{k: sm.get(k) for k in
                              ("median_final", "p10_final", "p90_final", "p_ruin", "p_1m", "median_days_to_ruin", "mean_restarts", "p_final_below_start")}})

    report = {"run_id": run_id, "model_version": ModelVersion(), "created_at": datetime.now(timezone.utc).isoformat(),
              "fill_assumption": "Tim's choice: the market mid (Polymarket history price) is the fill price",
              "coverage_and_sanity": sanity, "results": results, "path_sim": paths_out, "settings_counted": 8,
              "stale_min": STALE_MIN, "fee_bases": {"tim_est_0.02": "Tim's working estimate", "polymarket_sports_0.05": "0.05 x p x (1-p), uncertain for NFL"}}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    for suffix, rows_, fields in (("", tidy, ["date", "series", "set", "time", "equity", "drawdown_pct"]),
                                  ("_paths", path_tidy, list(path_tidy[0].keys()) if path_tidy else []),
                                  ("_bets", all_bets, list(all_bets[0].keys()) if all_bets else [])):
        if rows_:
            with open(os.path.join(a.out_dir, f"{a.base}{suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader(); w.writerows(rows_)
    brief = {k: {kk: {m: vv[m] for m in ("bets", "pnl", "end_bank")} for kk, vv in v.items()} for k, v in results.items()}
    print(json.dumps({"sanity": sanity, "results": brief}, indent=1))


if __name__ == "__main__":
    Main()