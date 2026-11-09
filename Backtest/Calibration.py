r"""
Calibration check of the walk-forward win-probability models against the de-vigged closing market.
Evaluation only: nothing is fitted or tuned here.

    .venv\Scripts\python.exe -m Backtest.Calibration --seasons 2026
    .venv\Scripts\python.exe -m Backtest.Calibration --seasons 2023,2024,2025,2026 --out-dir DIR --base NAME

For every finished, non-tied game with a moneyline:
  * Brier score and log loss of each model's HOME win probability (lower is better), next to the market's
    (de-vigged closing moneyline = `p_mkt`). Brier skill vs market = 1 - Brier_model / Brier_market.
  * Reliability bins: home-win probability cut into 10 equal-width bins; mean predicted vs observed rate, with N.
  * Picked-side accuracy (model favourite wins) vs market favourite on the same games.
Models p_full2 / p_full3 are fitted on seasons < S only (Lab.AttachModels); p_margin is the in-season
margin model. The closing line is a game-time price, so beating it is a high bar.

Writes  <out-dir>/<base>.csv   tidy: date, series, season_set, bin_lo, bin_hi, n, mean_pred, obs_rate
        <out-dir>/<base>.json  scores per season set and model, with N
        default out-dir: data/backtest/calibration/
"""

import argparse
import csv
import json
import math
import os

from Backtest.Lab import AllRows, AttachModels
from Backtest.Version import ModelVersion, NewRunId

MODELS = ["p_mkt", "p_full2", "p_full3", "p_margin"]
DEFAULT_OUT = os.path.join("data", "backtest", "calibration")


def Scores(rows, model) -> dict:
    rows = [r for r in rows if r["home_win"] is not None]
    n = len(rows)
    if not n:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    brier = sum((r[model] - r["home_win"]) ** 2 for r in rows) / n
    eps = 1e-6
    ll = -sum(math.log(min(max(r[model] if r["home_win"] else 1 - r[model], eps), 1 - eps)) for r in rows) / n
    acc = sum((r[model] >= 0.5) == (r["home_win"] == 1) for r in rows) / n
    return {"n": n, "brier": round(brier, 5), "log_loss": round(ll, 5), "accuracy": round(acc, 4)}


def ReliabilityBins(rows, model, bins=10) -> list:
    rows = [r for r in rows if r["home_win"] is not None]
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inb = [r for r in rows if lo <= r[model] < hi or (i == bins - 1 and r[model] == 1.0)]
        out.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": len(inb),
                    "mean_pred": round(sum(r[model] for r in inb) / len(inb), 4) if inb else None,
                    "obs_rate": round(sum(r["home_win"] for r in inb) / len(inb), 4) if inb else None})
    return out


def Calibration(seasons, rows_by_season=None, out_dir=None, base=None, write=True):
    if rows_by_season is None:
        rows_by_season = AllRows(2012, max(seasons))
        AttachModels(rows_by_season)
    for s in seasons:
        for r in rows_by_season[s]:
            r.setdefault("p_mkt", r["mkt_home"])
    sets = {str(s): rows_by_season[s] for s in seasons}
    if len(seasons) > 1:
        sets["+".join(str(s) for s in seasons)] = [r for s in seasons for r in rows_by_season[s]]
    report = {"run_id": NewRunId(), "model_version": ModelVersion(), "sets": {}}
    tidy = []
    for name, rows in sets.items():
        last = max((r["gameday"] for r in rows), default=None)
        scores = {m: Scores(rows, m) for m in MODELS}
        mb = scores["p_mkt"]["brier"]
        for m in MODELS:
            s = scores[m]
            s["brier_skill_vs_market"] = None if not mb or s["brier"] is None else round(1 - s["brier"] / mb, 4)
        report["sets"][name] = {"games": len(rows), "ties_excluded": sum(r["home_win"] is None for r in rows),
                                "last_game": last, "scores": scores}
        for m in MODELS:
            for b in ReliabilityBins(rows, m):
                tidy.append({"date": last, "series": m, "season_set": name, **b})
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or "calibration"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "series", "season_set", "bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"])
            w.writeheader()
            w.writerows(tidy)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return {"report": report, "tidy": tidy}


def Main():
    ap = argparse.ArgumentParser(description="Brier / log loss / reliability of walk-forward models vs the closing market.")
    ap.add_argument("--seasons", default="2026")
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    a = ap.parse_args()
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    out = Calibration(seasons, out_dir=a.out_dir, base=a.base)
    print(json.dumps(out["report"], indent=1))


if __name__ == "__main__":
    Main()
