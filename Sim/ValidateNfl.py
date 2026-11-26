r"""Validate the NFL drive resimulator on held-out seasons (Sportsbot, 2026-11-26; research, tested:false).

Leak-free: base tables for season S are fitted on drives of seasons < S; team factors use only games on earlier
dates (TeamWalk). 2023 is warm-up only. Tune = 2024 (small grid over n0 x carry, picked by Brier), holdout = 2025
and 2026 (pbp on disk ends 2026 week 9). Also computes near-miss ('slightly different') win probabilities.

Usage: .venv\Scripts\python.exe -m Sim.ValidateNfl --out-dir <dir> --base <name> [--sims 4000] [--panel <nfl_games.csv>]
Writes <base>_nfl_games.csv (one row per game) and <base>_nfl_summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from Sim.Drives import LoadDrives
from Sim.DriveModel import DriveArrays, FitBase, MatchupProbs, TeamWalk
from Sim.GameSim import SimulateGames, Summarise
from Sim.NearMiss import NearMiss

GRID = [(n0, carry) for n0 in (25.0, 60.0, 150.0) for carry in (0.3, 0.6)]
EPS = [("model", 0.25), ("model", 0.5), ("boot", 0.25), ("boot", 0.5), ("boot", 1.0)]
DEFAULT_PANEL = Path(r"E:\AgentDocs\Sportsbot\2026-11-26_cross-sport-history\2026-11-26_cross-sport-history_nfl_games.csv")


def Walk(drives, eval_seasons, n0, carry, sims, seed, near_miss=False, tilt=None):
    rng = np.random.default_rng(seed)
    games = (drives.group_by("game_id").agg(pl.col("game_date").first(), pl.col("season").first(), pl.col("season_type").first(),
                                            pl.col("home").first(), pl.col("away").first(), pl.col("home_final").first(),
                                            pl.col("away_final").first()).sort(["game_date", "game_id"]))
    bases = {}
    for s in sorted(set(games["season"].to_list())):
        prior = drives.filter(pl.col("season") < s)
        bases[s] = FitBase(prior if prior.height else drives.filter(pl.col("season") == s))
    walk = TeamWalk(n0=n0, carry=carry)
    by_game = {k[0]: g for k, g in drives.group_by(["game_id"])}
    out = []
    for (date,), day in games.group_by(["game_date"], maintain_order=True):
        rows = day.to_dicts()
        for r in rows:
            walk.NewSeason(r["season"])
        for r in rows:
            if r["season"] not in eval_seasons:
                continue
            base = bases[r["season"]]
            ho, hd = walk.Factors(r["home"], base.rate)
            ao, ad = walk.Factors(r["away"], base.rate)
            probs = MatchupProbs(base, ho, hd, ao, ad)
            h, a = SimulateGames(probs, base, sims, rng, playoff=r["season_type"] == "POST")
            rec = {"game_id": r["game_id"], "date": r["game_date"], "season": r["season"], "season_type": r["season_type"],
                   "home": r["home"], "away": r["away"], "home_final": r["home_final"], "away_final": r["away_final"],
                   "y": 1.0 if r["home_final"] > r["away_final"] else (0.5 if r["home_final"] == r["away_final"] else 0.0)}
            rec.update(Summarise(h, a, r["home_final"], r["away_final"]))
            if near_miss:
                g = by_game[r["game_id"]].sort("drive_idx")
                st, fb, cat = DriveArrays(g)
                side = 1 - g["is_home_off"].to_numpy()
                for src, eps in EPS:
                    rec[f"p_nm_{src}_{eps:g}"] = NearMiss(side, st, fb, cat, g["off_pts"].to_numpy(), g["def_pts"].to_numpy(),
                                                          probs, base, 2000, eps, rng, source=src)
            if tilt:
                for t in tilt:
                    pt = MatchupProbs(base, ho, hd, ao, ad, tilt_home=t)
                    hh, aa = SimulateGames(pt, base, sims, rng, playoff=r["season_type"] == "POST")
                    rec[f"p_tilt_{t:+g}"] = float(((hh - aa) > 0).mean() + 0.5 * ((hh - aa) == 0).mean())
            out.append(rec)
        for r in rows:
            walk.Update(by_game[r["game_id"]].sort("drive_idx"), bases[r["season"]])
    return pl.DataFrame(out, infer_schema_length=None)


def Brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--tune-sims", type=int, default=1000)
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    t0 = time.time()
    drives = LoadDrives([2023, 2024, 2025, 2026])
    grid = []
    for n0, carry in GRID:
        g = Walk(drives, {2024}, n0, carry, a.tune_sims, a.seed)
        grid.append({"n0": n0, "carry": carry, "n": g.height, "brier": Brier(g["p_home"], g["y"])})
        print("tune", grid[-1], flush=True)
    best = min(grid, key=lambda r: r["brier"])
    res = Walk(drives, {2024, 2025, 2026}, best["n0"], best["carry"], a.sims, a.seed + 1, near_miss=True,
               tilt=[-0.10, -0.05, 0.05, 0.10])
    res = res.with_columns(pl.when(pl.col("season") == 2024).then(pl.lit("tune")).otherwise(pl.lit("holdout")).alias("split"))
    panel = pl.read_csv(a.panel, infer_schema_length=10000).select(
        ["game_id", "p_book", "p_kalshi", "k_ask", "ka_ask", "p_full2_kmid", "p_margin", "ctx_roof", "ctx_spread"])
    res = res.join(panel, on="game_id", how="left")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.write_csv(out_dir / f"{a.base}_nfl_games.csv")
    summary = {"grid": grid, "chosen": best, "sims": a.sims, "games": res.height, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{a.base}_nfl_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    Main()
