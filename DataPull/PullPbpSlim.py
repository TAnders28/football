r"""
Slim play-by-play for the in-game win-probability model: one JSONL per season with only the pre-play state columns.

    .venv\Scripts\python.exe -m DataPull.PullPbpSlim [--seasons 2014-2026]

2014-2022 come from nflverse via nflreadpy.load_pbp (public GitHub release files); 2023-2026 from the local
data/playbyplay/pbp_<season>.jsonl already used elsewhere. Output (gitignored): data/playbyplay/slim/pbp_slim_<season>.jsonl (readable JSONL per AGENTS.md).
nflverse wp / vegas_wp / home_wp / vegas_home_wp are kept ONLY as a reference benchmark, never as model features.
"""

import argparse
import json
import os

import polars as pl

COLS = ["game_id", "season", "week", "season_type", "home_team", "away_team", "posteam", "defteam", "qtr", "down", "ydstogo",
        "yardline_100", "game_seconds_remaining", "half_seconds_remaining", "game_half", "posteam_score", "defteam_score",
        "score_differential", "total_home_score", "total_away_score", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "home_timeouts_remaining", "away_timeouts_remaining", "play_type", "time_of_day", "end_clock_time", "sp", "play_id",
        "wp", "vegas_wp", "home_wp", "vegas_home_wp"]
OUT = os.path.join("data", "playbyplay", "slim")


def FromJsonl(season):
    rows = []
    with open(os.path.join("data", "playbyplay", f"pbp_{season}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({c: r.get(c) for c in COLS})
    return pl.DataFrame(rows, infer_schema_length=None)


def FromNflverse(season):
    import nflreadpy
    df = nflreadpy.load_pbp([season])
    return df.select([c for c in COLS if c in df.columns])


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2014-2026")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seasons.split("-")) if "-" in a.seasons else (int(a.seasons),) * 2
    os.makedirs(OUT, exist_ok=True)
    for s in range(lo, hi + 1):
        df = FromJsonl(s) if os.path.exists(os.path.join("data", "playbyplay", f"pbp_{s}.jsonl")) else FromNflverse(s)
        for c in ("time_of_day", "end_clock_time", "posteam", "defteam", "play_type", "game_id", "home_team", "away_team", "season_type", "game_half"):
            if c in df.columns:
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
        num = [c for c in COLS if c in df.columns and c not in ("time_of_day", "end_clock_time", "posteam", "defteam", "play_type", "game_id",
                                                                 "home_team", "away_team", "season_type", "game_half")]
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
        df.write_ndjson(os.path.join(OUT, f"pbp_slim_{s}.jsonl"))
        print(s, df.height, "games", df["game_id"].n_unique(), flush=True)


if __name__ == "__main__":
    Main()