r"""
Fetch historical NFL schedules (with closing moneylines/spreads) for TRAINING ONLY.

    .venv\Scripts\python.exe -m Backtest.FetchHistory            # 2012-2022 -> data/backtest/history/

The backtest seasons (2023+) come from GetSeasonGames.py; these older seasons are only used to
fit/tune models walk-forward (a season is only ever predicted from seasons strictly before it).
"""

import argparse
import json
import math
import os

import nflreadpy

OUT_DIR = os.path.join("data", "backtest", "history")


def Clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def Main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=2012)
    parser.add_argument("--last", type=int, default=2022)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    frame = nflreadpy.load_schedules(list(range(args.first, args.last + 1))).to_pandas()
    for season, part in frame.groupby("season"):
        rows = [{k: Clean(v) for k, v in rec.items()} for rec in part.to_dict("records")]
        path = os.path.join(OUT_DIR, f"games_{season}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, default=str)
        print(path, len(rows))


if __name__ == "__main__":
    Main()
