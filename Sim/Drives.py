r"""Drive table from nflverse play-by-play (data/playbyplay/pbp_<season>.jsonl, 2023 on).

One row per offensive drive (nflverse `fixed_drive`), with:
  game_id, season, week, season_type, game_date, home, away, posteam, defteam, is_home_off,
  half (1, 2, 3 = overtime), half_secs (seconds left in the half at the first scrimmage play),
  game_secs, start_fp (yardline_100 at the first scrimmage play: yards to the opponent end zone),
  score_diff (posteam minus defteam at the drive start), cat (drive result, see CATS),
  off_pts / def_pts (points the offense / defense scored from this drive start to the next drive start,
  so extra points, two-point tries, return touchdowns and safeties are counted where they happen),
  dur (drive time of possession, seconds), home_final / away_final.

Usage (Tomputer, repo root): .venv\Scripts\python.exe -m Sim.Drives --seasons 2023 2024 2025 2026
Writes data/sim/drives_<season>.parquet (gitignored).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
PBP = REPO / "data" / "playbyplay"
OUT = REPO / "data" / "sim"

CATS = ["TD", "FG", "PUNT", "TO", "DOWNS", "MISSFG", "OPPTD", "SAFETY", "EOH"]
RESULT_TO_CAT = {"Touchdown": "TD", "Field goal": "FG", "Punt": "PUNT", "Turnover": "TO",
                 "Turnover on downs": "DOWNS", "Missed field goal": "MISSFG", "Opp touchdown": "OPPTD",
                 "Safety": "SAFETY", "End of half": "EOH"}
SCRIMMAGE = {"run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"}
COLS = ["game_id", "season_type", "week", "game_date", "home_team", "away_team", "posteam", "defteam", "fixed_drive",
        "fixed_drive_result", "drive_time_of_possession", "game_seconds_remaining", "half_seconds_remaining", "game_half",
        "yardline_100", "total_home_score", "total_away_score", "home_score", "away_score", "play_type", "play_id"]


def ParseClock(s):
    """'5:31' -> 331 seconds; None/invalid -> nan."""
    if s is None or not isinstance(s, str) or ":" not in s:
        return float("nan")
    m, sec = s.split(":", 1)
    try:
        return float(int(m) * 60 + int(sec))
    except ValueError:
        return float("nan")


def HalfOf(game_half):
    return {"Half1": 1, "Half2": 2, "Overtime": 3}.get(game_half, 0)


def DrivesFromPlays(plays: pl.DataFrame) -> pl.DataFrame:
    """Plays of any number of games (nflverse columns COLS) -> one row per drive. Pure function (tested)."""
    plays = plays.sort(["game_id", "play_id"])
    rows = []
    for (gid,), g in plays.group_by(["game_id"], maintain_order=True):
        g = g.to_dicts()
        home, away = g[0]["home_team"], g[0]["away_team"]
        hf, af = g[0]["home_score"], g[0]["away_score"]
        drives = {}
        order = []
        for p in g:
            d = p["fixed_drive"]
            if d is None:
                continue
            if d not in drives:
                drives[d] = []
                order.append(d)
            drives[d].append(p)
        recs = []
        for d in order:
            ps = drives[d]
            first = next((p for p in ps if p["play_type"] in SCRIMMAGE and p["posteam"] and p["yardline_100"] is not None), None)
            if first is None:
                continue
            res = next((p["fixed_drive_result"] for p in ps if p["fixed_drive_result"]), None)
            cat = RESULT_TO_CAT.get(res)
            if cat is None:
                continue
            recs.append({"first": first, "cat": cat, "dur": ParseClock(next((p["drive_time_of_possession"] for p in ps if p["drive_time_of_possession"]), None)),
                         "h0": first["total_home_score"] or 0.0, "a0": first["total_away_score"] or 0.0})
        for i, r in enumerate(recs):
            f = r["first"]
            if i + 1 < len(recs):
                h1, a1 = recs[i + 1]["h0"], recs[i + 1]["a0"]
            else:
                h1, a1 = float(hf), float(af)
            dh, da = h1 - r["h0"], a1 - r["a0"]
            is_home = f["posteam"] == home
            off_pts, def_pts = (dh, da) if is_home else (da, dh)
            sd = (r["h0"] - r["a0"]) * (1 if is_home else -1)
            rows.append({"game_id": gid, "season": int(str(gid)[:4]), "week": int(f["week"] or 0), "season_type": f["season_type"],
                         "game_date": f["game_date"], "home": home, "away": away, "posteam": f["posteam"],
                         "defteam": away if is_home else home, "is_home_off": int(is_home), "drive_idx": i,
                         "half": HalfOf(f["game_half"]), "half_secs": float(f["half_seconds_remaining"] or 0.0),
                         "game_secs": float(f["game_seconds_remaining"] or 0.0), "start_fp": float(f["yardline_100"]),
                         "score_diff": float(sd), "cat": r["cat"], "off_pts": float(max(off_pts, 0.0)), "def_pts": float(max(def_pts, 0.0)),
                         "dur": r["dur"], "home_final": float(hf), "away_final": float(af)})
    return pl.DataFrame(rows)


def BuildSeason(season: int) -> pl.DataFrame:
    path = PBP / f"pbp_{season}.jsonl"
    plays = pl.scan_ndjson(path, infer_schema_length=20000).select(COLS).collect()
    plays = plays.filter(pl.col("home_score").is_not_null())
    return DrivesFromPlays(plays)


def LoadDrives(seasons) -> pl.DataFrame:
    frames = [pl.read_parquet(OUT / f"drives_{s}.parquet") for s in seasons if (OUT / f"drives_{s}.parquet").exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.seasons:
        d = BuildSeason(s)
        d.write_parquet(OUT / f"drives_{s}.parquet")
        g = d.group_by("game_id").agg(pl.col("off_pts").filter(pl.col("is_home_off") == 1).sum().alias("h_off"),
                                      pl.col("def_pts").filter(pl.col("is_home_off") == 0).sum().alias("h_def"),
                                      pl.col("home_final").first())
        recon = float(((g["h_off"] + g["h_def"]) == g["home_final"]).mean())
        print(f"{s}: drives={d.height} games={d['game_id'].n_unique()} home-points reconciled={recon:.3f}")


if __name__ == "__main__":
    Main()
