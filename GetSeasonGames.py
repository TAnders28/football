r"""
Pull NFL seasons from nflverse (via nflreadpy) and store them as readable JSON / JSONL.

    .venv\Scripts\python.exe GetSeasonGames.py --season 2023,2024,2025,2026 [--refresh]

Data is pulled into polars DataFrames, converted to plain Python records, and only then
written to disk -- no parquet is stored. Mirrors the baseball repo (games_<season>.json,
.jsonl for big per-row tables).

Per season writes:
    data/backtest/games_<season>.json           JSON array, one object per game, pretty-printed
                                                (scores, spread/total/ML, weather, QBs, ...)
    data/backtest/pull_summary_<season>.json    counts + pull timestamp
    data/playbyplay/pbp_<season>.jsonl          JSONL, one play per line (~370 nflverse cols:
                                                EPA, WPA, air yards, ...). JSONL rather than one
                                                big JSON array so it stays streamable/greppable.
    data/playbyplay/player_week_<season>.jsonl  JSONL, one player-week per line

Format choices:
    - games are small (~285/season) so they get an indented JSON array for easy reading.
    - pbp / player_week are large, so they are compact JSONL (one record per line).
    - in the JSONL files, null fields are omitted from each record to keep size and noise
      down; a missing key means null. games JSON keeps every key (nulls included).
    - dates/datetimes are written as ISO strings.

Seasons that already have every output are skipped unless --refresh is given.
"""

import argparse
import json
import os
from datetime import date, datetime, timezone

import nflreadpy as nfl
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKTEST_DIR = os.path.join(ROOT, "data", "backtest")
PBP_DIR = os.path.join(ROOT, "data", "playbyplay")


def OutputPaths(season: int) -> dict:
    return {
        "games": os.path.join(BACKTEST_DIR, f"games_{season}.json"),
        "summary": os.path.join(BACKTEST_DIR, f"pull_summary_{season}.json"),
        "pbp": os.path.join(PBP_DIR, f"pbp_{season}.jsonl"),
        "player_week": os.path.join(PBP_DIR, f"player_week_{season}.jsonl"),
    }


def JsonDefault(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def WriteJson(df: pl.DataFrame, path: str) -> None:
    """Pretty JSON array of every row (all keys kept)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(df.to_dicts(), f, indent=2, default=JsonDefault)


def WriteJsonl(df: pl.DataFrame, path: str) -> None:
    """One compact JSON object per line, null fields dropped."""
    with open(path, "w", encoding="utf-8") as f:
        for row in df.iter_rows(named=True):
            record = {k: v for k, v in row.items() if v is not None}
            f.write(json.dumps(record, separators=(",", ":"), default=JsonDefault))
            f.write("\n")


def PullSeason(season: int, refresh: bool) -> None:
    paths = OutputPaths(season)
    if not refresh and all(os.path.exists(p) for p in paths.values()):
        print(f"[{season}] all outputs exist, skipping (use --refresh to re-pull)")
        return

    print(f"[{season}] schedules...")
    games = nfl.load_schedules(seasons=[season])
    print(f"[{season}] play-by-play...")
    pbp = nfl.load_pbp(seasons=[season])
    print(f"[{season}] weekly player stats...")
    players = nfl.load_player_stats(seasons=[season], summary_level="week")

    WriteJson(games, paths["games"])
    WriteJsonl(pbp, paths["pbp"])
    WriteJsonl(players, paths["player_week"])

    completed = games.filter(pl.col("result").is_not_null())
    summary = {
        "season": season,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "games_scheduled": games.height,
        "games_completed": completed.height,
        "games_by_type": dict(games.group_by("game_type").len().iter_rows()),
        "last_completed_week": completed["week"].max(),
        "pbp_rows": pbp.height,
        "pbp_cols": pbp.width,
        "pbp_games": pbp["game_id"].n_unique() if pbp.height else 0,
        "player_week_rows": players.height,
        "player_week_players": players["player_id"].n_unique() if players.height else 0,
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{season}] games {summary['games_completed']}/{summary['games_scheduled']} completed"
          f" (through week {summary['last_completed_week']}), pbp rows {summary['pbp_rows']:,},"
          f" player-week rows {summary['player_week_rows']:,}")


def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=str(nfl.get_current_season()),
                        help="comma-separated seasons, e.g. 2023,2024,2025,2026")
    parser.add_argument("--refresh", action="store_true", help="re-pull even if outputs exist")
    args = parser.parse_args()

    os.makedirs(BACKTEST_DIR, exist_ok=True)
    os.makedirs(PBP_DIR, exist_ok=True)
    for season in (int(s) for s in args.season.split(",") if s.strip()):
        PullSeason(season, args.refresh)


if __name__ == "__main__":
    Main()
