r"""
Grade the append-only prediction log against final scores. The log is only read, never changed.

    .venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026
    .venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026 --no-refresh

Reads   data/predictions/<season>.jsonl            (written by Backtest.PredictDay)
Writes  data/predictions/grades_<season>.json      {game_id: outcome row}, rebuilt each run
        data/predictions/grades_<season>.md        short summary (N, accuracy, holdout split)

Rules
  * A log row counts only if created_at < start_time (made before kickoff). Rows created at or after
    kickoff are excluded and counted in `excluded_late_rows`.
  * Per game, the LAST pre-kickoff row is graded (the freshest honest prediction). All earlier rows
    stay in the log untouched.
  * Ties grade as `correct: null`. Unfinished games are listed as pending.
  * Hit rate is not edge: this reports accuracy and the market favourite's accuracy on the same games.
    No P&L is invented.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from Backtest.FrozenConfig import FROZEN, IsForwardHoldout

ET = ZoneInfo("America/New_York")
LOG_DIR = os.path.join("data", "predictions")


def ReadLog(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def Grade(log_rows: list, games: list, now=None) -> dict:
    """Returns {game_id: outcome}. `games` are schedule rows with home/away scores (None if unplayed)."""
    now = (now or datetime.now(ET)).astimezone(ET)
    by_id = {g["game_id"]: g for g in games}
    pre, late = defaultdict(list), defaultdict(int)
    for r in log_rows:
        if datetime.fromisoformat(r["created_at"]) < datetime.fromisoformat(r["start_time"]):
            pre[r["game_id"]].append(r)
        else:
            late[r["game_id"]] += 1
    out = {}
    for gid in sorted(set(pre) | set(late)):
        rows = sorted(pre.get(gid, []), key=lambda r: r["created_at"])
        g = by_id.get(gid) or {}
        base = {"game_id": gid, "excluded_late_rows": late.get(gid, 0), "pre_kickoff_rows": len(rows),
                "graded_at": now.isoformat(timespec="seconds")}
        if not rows:
            out[gid] = dict(base, status="no_pre_kickoff_prediction")
            continue
        r = rows[-1]
        hs, as_ = g.get("home_score"), g.get("away_score")
        row = dict(base, start_time=r["start_time"], matchup=r.get("matchup"), graded_row_created_at=r["created_at"],
                   run_id=r.get("run_id"), model_version=r.get("model_version"), pick=r.get("pick"),
                   pick_side=r.get("pick_side"), model_prob=r.get("model_prob"), action=r.get("action"),
                   market_prob_novig=r.get("market_prob_novig"),
                   forward_holdout=IsForwardHoldout(r["start_time"][:10]))
        if hs is None or as_ is None:
            out[gid] = dict(row, status="pending")
            continue
        winner = None if hs == as_ else ("home" if hs > as_ else "away")
        fav = None
        if r.get("market_prob_novig") is not None and r.get("pick_side"):
            p_pick = r["market_prob_novig"]
            fav = r["pick_side"] if p_pick >= 0.5 else ("away" if r["pick_side"] == "home" else "home")
        out[gid] = dict(row, status="final", home_score=hs, away_score=as_, winner=winner,
                        correct=None if winner is None else (r.get("pick_side") == winner),
                        market_fav_correct=None if winner is None or fav is None else (fav == winner))
    return out


def Summary(grades: dict) -> dict:
    def acc(rows, key):
        done = [g for g in rows if g.get(key) is not None]
        return {"n": len(done), "correct": sum(bool(g[key]) for g in done),
                "rate": round(sum(bool(g[key]) for g in done) / len(done), 4) if done else None}
    final = [g for g in grades.values() if g.get("status") == "final"]
    hold = [g for g in final if g.get("forward_holdout")]
    bets = [g for g in final if g.get("action") == "BET"]
    return {"games_logged": len(grades), "final": len(final),
            "pending": sum(g.get("status") == "pending" for g in grades.values()),
            "excluded_late_rows": sum(g.get("excluded_late_rows", 0) for g in grades.values()),
            "all": acc(final, "correct"), "all_market_fav": acc(final, "market_fav_correct"),
            "forward_holdout": acc(hold, "correct"), "forward_holdout_market_fav": acc(hold, "market_fav_correct"),
            "bets": acc(bets, "correct"), "frozen_config": FROZEN["name"], "frozen_at": FROZEN["frozen_at"]}


def SummaryMarkdown(season, s):
    def f(a):
        return "-" if a["rate"] is None else f"{a['correct']}/{a['n']} ({a['rate']*100:.1f}%)"
    return "\n".join([
        f"# Prediction grades {season}", "",
        f"Frozen config {s['frozen_config']} (frozen {s['frozen_at']}). Rows made at/after kickoff excluded: {s['excluded_late_rows']}.", "",
        "| Set | Model pick | Market favourite (same games) |", "|---|---|---|",
        f"| All final games | {f(s['all'])} | {f(s['all_market_fav'])} |",
        f"| Forward holdout (after freeze) | {f(s['forward_holdout'])} | {f(s['forward_holdout_market_fav'])} |",
        f"| BET rows only | {f(s['bets'])} | - |", "",
        f"Games logged {s['games_logged']}, final {s['final']}, pending {s['pending']}. "
        "Accuracy only: hit rate is not edge, and no P&L is computed here. Treat +-7 pts at N~200 as noise.", ""])


def GradePredictions(season: int, games: list = None, log_dir: str = None, refresh: bool = True, write: bool = True, now=None):
    log_dir = log_dir or LOG_DIR
    rows = ReadLog(os.path.join(log_dir, f"{season}.jsonl"))
    if games is None:
        from Backtest.PredictDay import LoadSchedule
        games = LoadSchedule(season, refresh)["games"]
    grades = Grade(rows, games, now)
    summary = Summary(grades)
    if write:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"grades_{season}.json"), "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "games": grades}, fh, indent=1)
        with open(os.path.join(log_dir, f"grades_{season}.md"), "w", encoding="utf-8") as fh:
            fh.write(SummaryMarkdown(season, summary))
    return {"summary": summary, "games": grades}


def Main():
    ap = argparse.ArgumentParser(description="Grade data/predictions/<season>.jsonl (read-only) against final scores.")
    ap.add_argument("--season", type=int, default=datetime.now(ET).year if datetime.now(ET).month >= 3 else datetime.now(ET).year - 1)
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    out = GradePredictions(a.season, refresh=not a.no_refresh)
    print(SummaryMarkdown(a.season, out["summary"]))


if __name__ == "__main__":
    Main()
