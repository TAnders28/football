r"""
Pull pre-game availability data from nflverse (via nflreadpy) into readable JSONL, one file per season.

    .venv\Scripts\python.exe -m DataPull.PullPregame --first 2012 --last 2026

Writes (all gitignored under data/):
    data/injuries/<season>.jsonl   weekly injury report, one row per player per team-week (FINAL status of that week)
    data/snaps/<season>.jsonl      PFR snap counts per player-game (post-game data; use ONLY lagged, earlier games)
    data/depth/<season>.jsonl      depth charts. <=2024: weekly NFL depth charts (no publish timestamp).
                                   >=2025: ESPN daily snapshots (dt, UTC), thinned to the LAST snapshot per team per
                                   UTC date and pos_rank <= 2 to keep the file readable.
    data/players/players.jsonl     gsis_id <-> pfr_id / espn_id crosswalk (+ position) for joining snaps to injuries
    data/injuries/pull_summary.json   rows per season and source schema notes

Timing notes (see Backtest/InjuryFeatures.py and the 2026-11-13 report):
  * report_status (Out / Doubtful / Questionable) is the final game-status designation the club publishes before the
    game (Friday for Sunday games, earlier for TNF; Saturday/Sunday updates possible). nflverse keeps only one row per
    player-week, i.e. the LAST version.
  * 2009-2024 rows carry date_modified (UTC): the report pull can be checked against kickoff. 2025+ rows have no
    timestamp at all, so their pre-game timing cannot be verified from the data.
  * Snap counts are recorded after the game; they are never a same-game feature.
"""

import argparse
import json
import os

import nflreadpy as nfl
import polars as pl

OUT = "data"


def _write(df: pl.DataFrame, path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.write_ndjson(tmp)
    os.replace(tmp, path)
    return df.height


def PullInjuries(season: int) -> dict:
    df = nfl.load_injuries([season])
    cols = df.columns
    if "date_modified" in cols:
        df = df.with_columns(pl.col("date_modified").dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    n = _write(df, os.path.join(OUT, "injuries", f"{season}.jsonl"))
    return {"rows": n, "has_date_modified": "date_modified" in cols, "columns": cols}


def PullSnaps(season: int) -> dict:
    df = nfl.load_snap_counts([season])
    return {"rows": _write(df, os.path.join(OUT, "snaps", f"{season}.jsonl"))}


def PullDepth(season: int) -> dict:
    df = nfl.load_depth_charts([season])
    raw = df.height
    if "dt" in df.columns:   # 2025+ daily ESPN snapshots
        df = df.with_columns(pl.col("dt").str.slice(0, 10).alias("_day"))
        last = df.group_by(["team", "_day"]).agg(pl.col("dt").max().alias("_last"))
        df = (df.join(last, on=["team", "_day"]).filter((pl.col("dt") == pl.col("_last")) & (pl.col("pos_rank") <= 2))
                .drop(["_day", "_last"]))
        fmt = "espn_daily_snapshots(dt UTC), last per team per UTC day, pos_rank<=2"
    else:
        fmt = "weekly nfl depth chart, no publish timestamp"
    return {"rows_raw": raw, "rows_kept": _write(df, os.path.join(OUT, "depth", f"{season}.jsonl")), "format": fmt}


def PullPlayers() -> dict:
    df = nfl.load_players().select(["gsis_id", "display_name", "position", "position_group", "pfr_id", "espn_id"])
    return {"rows": _write(df, os.path.join(OUT, "players", "players.jsonl"))}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2012)
    ap.add_argument("--last", type=int, default=2026)
    a = ap.parse_args()
    summary = {"players": PullPlayers(), "seasons": {}}
    for s in range(a.first, a.last + 1):
        row = {}
        for name, fn in (("injuries", PullInjuries), ("snaps", PullSnaps), ("depth", PullDepth)):
            try:
                row[name] = fn(s)
            except Exception as exc:   # record the gap, keep going
                row[name] = {"error": repr(exc)[:300]}
        summary["seasons"][s] = row
        print(s, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "columns"} for k, v in row.items()}), flush=True)
    os.makedirs(os.path.join(OUT, "injuries"), exist_ok=True)
    with open(os.path.join(OUT, "injuries", "pull_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    Main()