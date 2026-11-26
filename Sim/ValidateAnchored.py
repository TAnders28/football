r"""Validate the market-anchored NFL resimulator on shape (Sportsbot, 2026-11-26; research, tested:false).

For every game in 2024 (tune) and 2025-2026 wk 9 (holdout): anchor to the de-vigged closing moneyline (p_book from the
cross-sport panel) and the closing total (nflverse total_line), using league-average drive tables fitted on prior
seasons only; simulate `--sims` versions; record margin / total PIT, key numbers, spread-cover probability and the
'near-miss' re-roll probabilities from the anchored tables (eps 0.25 / 0.5, model source).

Usage: .venv\Scripts\python.exe -m Sim.ValidateAnchored --out-dir <dir> --base <name> [--sims 4000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from Sim.Anchor import Anchor, AnchoredProbs
from Sim.Drives import LoadDrives
from Sim.DriveModel import DriveArrays, FitBase
from Sim.GameSim import SimulateGames, Summarise
from Sim.NearMiss import NearMiss
from Sim.ValidateNfl import DEFAULT_PANEL

REPO = Path(__file__).resolve().parents[1]


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--seed", type=int, default=20261127)
    a = ap.parse_args()
    t0 = time.time()
    drives = LoadDrives([2023, 2024, 2025, 2026])
    lines = {}
    for s in (2024, 2025, 2026):
        import json as _j
        for g in _j.load(open(REPO / "data" / "backtest" / f"games_{s}.json")):
            lines[g["game_id"]] = (g.get("total_line"), g.get("spread_line"))
    panel = pl.read_csv(a.panel, infer_schema_length=10000).select(["game_id", "p_book", "p_kalshi"])
    pb = dict(zip(panel["game_id"].to_list(), panel["p_book"].to_list()))
    pk = dict(zip(panel["game_id"].to_list(), panel["p_kalshi"].to_list()))
    bases = {s: FitBase(drives.filter(pl.col("season") < s)) for s in (2024, 2025, 2026)}
    rng = np.random.default_rng(a.seed)
    out = []
    for (gid,), g in drives.filter(pl.col("season") >= 2024).group_by(["game_id"], maintain_order=True):
        g = g.sort("drive_idx")
        r = g.row(0, named=True)
        p_book = pb.get(gid)
        tl, sl = lines.get(gid, (None, None))
        if p_book is None or tl is None:
            continue
        base = bases[r["season"]]
        playoff = r["season_type"] == "POST"
        s, k = Anchor(base, p_book, tl, seed=a.seed, playoff=playoff)
        probs = AnchoredProbs(base, s, k)
        h, aw = SimulateGames(probs, base, a.sims, rng, playoff=playoff)
        rec = {"game_id": gid, "date": r["game_date"], "season": r["season"], "season_type": r["season_type"],
               "split": "tune" if r["season"] == 2024 else "holdout", "home": r["home"], "away": r["away"],
               "home_final": r["home_final"], "away_final": r["away_final"],
               "y": 1.0 if r["home_final"] > r["away_final"] else (0.5 if r["home_final"] == r["away_final"] else 0.0),
               "p_book": p_book, "p_kalshi": pk.get(gid), "total_line": tl, "spread_line": sl, "anchor_s": s, "anchor_k": k}
        rec.update(Summarise(h, aw, r["home_final"], r["away_final"]))
        m = h - aw
        if sl is not None:
            rec["sim_p_cover"] = float((m > sl).mean() / max((m != sl).mean(), 1e-9))
            am = r["home_final"] - r["away_final"]
            rec["real_cover"] = None if am == sl else float(am > sl)
            rec["sim_p_over"] = float(((h + aw) > tl).mean() / max(((h + aw) != tl).mean(), 1e-9))
            at = r["home_final"] + r["away_final"]
            rec["real_over"] = None if at == tl else float(at > tl)
        st, fb, cat = DriveArrays(g)
        side = 1 - g["is_home_off"].to_numpy()
        for eps in (0.25, 0.5):
            rec[f"p_nm_anch_{eps:g}"] = NearMiss(side, st, fb, cat, g["off_pts"].to_numpy(), g["def_pts"].to_numpy(),
                                                 probs, base, 2000, eps, rng, source="model")
        out.append(rec)
        if len(out) % 100 == 0:
            print(len(out), round(time.time() - t0), flush=True)
    res = pl.DataFrame(out, infer_schema_length=None)
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    res.write_csv(od / f"{a.base}_nfl_anchored_games.csv")
    print(json.dumps({"games": res.height, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    Main()
