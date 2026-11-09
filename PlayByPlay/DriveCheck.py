r"""
Rebuild drives from raw play-by-play on our own and validate them against nflverse's drive fields.

    .venv\Scripts\python.exe PlayByPlay\DriveCheck.py --season 2026
    .venv\Scripts\python.exe PlayByPlay\DriveCheck.py --season 2023,2024,2025,2026 --out-dir DIR --base NAME

Reads   data/playbyplay/pbp_<season>.jsonl   (nflverse play-by-play, one play per line; no network)

Our rebuild (independent of nflverse's fixed_drive):
  * Only rows with a possession team (`posteam`) take part; timeouts / quarter ends have none.
  * A new drive starts when posteam changes or the game half changes (Half1 -> Half2 -> Overtime).
    nflverse gives kickoffs to the RECEIVING team, so a kickoff opens the receiver's drive.
  * Result from the drive's own plays, first match wins:
        touchdown by offense -> Touchdown | touchdown by defense -> Opp touchdown | safety -> Safety
        made FG -> Field goal | missed/blocked FG -> Missed field goal | punt -> Punt
        interception or lost fumble -> Turnover | failed 4th down -> Turnover on downs | else End of half

Checks against nflverse
  * boundary match: share of nflverse drives (game_id, fixed_drive) whose set of plays equals one of our drives.
  * drive count per game: ours minus nflverse.
  * result agreement on boundary-matched drives (confusion counts by nflverse `fixed_drive_result`).
  * play count: our count of `play == 1` rows vs nflverse `drive_play_count` (NFL API value) on matched drives,
    exact and within +-1 (the NFL's official count treats penalties / kickoffs / tries differently).

Writes  <out-dir>/<base>.csv   tidy per week: date, series(season), week, nfl_drives, matched, match_rate,
                               result_agree_rate, play_count_agree_rate,
                               play_count_within1_rate, games
        <out-dir>/<base>.json  totals, confusion table, mismatch examples
        default out-dir: stats/playbyplay/
"""

import argparse
import csv
import json
import os
from collections import Counter, OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(ROOT, "data", "playbyplay")
DEFAULT_OUT = os.path.join(ROOT, "stats", "playbyplay")
KEEP = ("game_id", "game_date", "week", "season", "play_id", "order_sequence", "posteam", "defteam", "game_half",
        "fixed_drive", "fixed_drive_result", "drive_play_count", "play", "touchdown", "td_team", "safety",
        "field_goal_result", "punt_attempt", "interception", "fumble_lost", "fourth_down_failed", "desc", "play_type")


def ReadPbp(season: int):
    """Yield trimmed play dicts in file order."""
    with open(os.path.join(PBP_DIR, f"pbp_{season}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                yield {k: row.get(k) for k in KEEP}


def _on(v) -> bool:
    return v in (1, 1.0, True)


def RebuildDrives(plays: list) -> list:
    """plays: one game's rows in order. Returns list of drives, each a list of indexes into `plays`."""
    drives, cur, last_team, last_half = [], [], None, None
    for i, p in enumerate(plays):
        team = p.get("posteam")
        if not team:
            continue
        half = p.get("game_half")
        if cur and (team != last_team or half != last_half):
            drives.append(cur)
            cur = []
        cur.append(i)
        last_team, last_half = team, half
    if cur:
        drives.append(cur)
    return drives


def DriveResult(rows: list) -> str:
    offense = next((r["posteam"] for r in rows if r.get("posteam")), None)
    for r in rows:
        if _on(r.get("touchdown")):
            return "Touchdown" if (r.get("td_team") or offense) == offense else "Opp touchdown"
    if any(_on(r.get("safety")) for r in rows):
        return "Safety"
    fg = [r.get("field_goal_result") for r in rows if r.get("field_goal_result")]
    if fg:
        return "Field goal" if fg[-1] == "made" else "Missed field goal"
    if any(_on(r.get("punt_attempt")) for r in rows):
        return "Punt"
    if any(_on(r.get("interception")) or _on(r.get("fumble_lost")) for r in rows):
        return "Turnover"
    if any(_on(r.get("fourth_down_failed")) for r in rows):
        return "Turnover on downs"
    return "End of half"


def CheckGame(plays: list) -> dict:
    """Compare our drives with nflverse fixed_drive for one game."""
    ours = RebuildDrives(plays)
    our_sets = {frozenset(d): d for d in ours}
    nfl = OrderedDict()
    for i, p in enumerate(plays):
        if p.get("posteam") and p.get("fixed_drive") is not None:
            nfl.setdefault(p["fixed_drive"], []).append(i)
    out = {"game_id": plays[0]["game_id"], "week": plays[0].get("week"), "date": plays[0].get("game_date"),
           "nfl_drives": len(nfl), "our_drives": len(ours), "matched": 0, "result_checked": 0, "result_agree": 0,
           "play_count_checked": 0, "play_count_agree": 0, "play_count_within1": 0, "confusion": Counter(), "examples": []}
    for num, idx in nfl.items():
        rows = [plays[i] for i in idx]
        if frozenset(idx) in our_sets:
            out["matched"] += 1
            nres = rows[0].get("fixed_drive_result")
            ores = DriveResult(rows)
            if nres:
                out["result_checked"] += 1
                out["result_agree"] += nres == ores
                out["confusion"][(nres, ores)] += 1
            dpc = rows[0].get("drive_play_count")
            if dpc is not None:
                out["play_count_checked"] += 1
                ours_n = sum(_on(r.get("play")) for r in rows)
                out["play_count_agree"] += int(dpc) == ours_n
                out["play_count_within1"] += abs(int(dpc) - ours_n) <= 1
        elif len(out["examples"]) < 3:
            first = rows[0]
            out["examples"].append({"game_id": out["game_id"], "fixed_drive": num, "posteam": first.get("posteam"),
                                    "plays": len(idx), "first_desc": (first.get("desc") or "")[:120],
                                    "nfl_result": first.get("fixed_drive_result")})
    return out


def CheckSeason(season: int, plays_iter=None) -> list:
    games = OrderedDict()
    for p in (plays_iter if plays_iter is not None else ReadPbp(season)):
        games.setdefault(p["game_id"], []).append(p)
    results = []
    for gid, plays in games.items():
        plays.sort(key=lambda p: (p.get("order_sequence") or 0, p.get("play_id") or 0))
        results.append(CheckGame(plays))
    return results


def Totals(results: list) -> dict:
    def rate(a, b):
        return round(a / b, 4) if b else None
    nd = sum(r["nfl_drives"] for r in results)
    m = sum(r["matched"] for r in results)
    rc, ra = sum(r["result_checked"] for r in results), sum(r["result_agree"] for r in results)
    pc, pa = sum(r["play_count_checked"] for r in results), sum(r["play_count_agree"] for r in results)
    p1 = sum(r.get("play_count_within1", 0) for r in results)
    diff = Counter(r["our_drives"] - r["nfl_drives"] for r in results)
    return {"games": len(results), "nfl_drives": nd, "our_drives": sum(r["our_drives"] for r in results),
            "matched": m, "match_rate": rate(m, nd), "result_checked": rc, "result_agree_rate": rate(ra, rc),
            "play_count_checked": pc, "play_count_agree_rate": rate(pa, pc),
            "play_count_within1_rate": rate(p1, pc), "games_same_drive_count": diff.get(0, 0), "drive_count_diff_hist": dict(sorted(diff.items()))}


def DriveCheck(seasons, out_dir=None, base=None, write=True, results_by_season=None):
    results_by_season = results_by_season or {s: CheckSeason(s) for s in seasons}
    report = {"seasons": {}, "examples": []}
    tidy = []
    confusion = Counter()
    for s, results in results_by_season.items():
        report["seasons"][str(s)] = Totals(results)
        for r in results:
            confusion.update(r["confusion"])
            report["examples"].extend(r["examples"][:1])
        by_week = defaultdict(list)
        for r in results:
            by_week[r["week"]].append(r)
        for wk in sorted(by_week, key=lambda w: (w is None, w)):
            t = Totals(by_week[wk])
            tidy.append({"date": min(r["date"] for r in by_week[wk] if r["date"]), "series": str(s), "week": wk,
                         "nfl_drives": t["nfl_drives"], "matched": t["matched"], "match_rate": t["match_rate"],
                         "result_agree_rate": t["result_agree_rate"], "play_count_agree_rate": t["play_count_agree_rate"],
                         "play_count_within1_rate": t["play_count_within1_rate"], "games": t["games"]})
    report["all"] = Totals([r for rs in results_by_season.values() for r in rs])
    report["confusion"] = [{"nflverse": a, "ours": b, "n": n} for (a, b), n in confusion.most_common()]
    report["examples"] = report["examples"][:40]
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or "drive_check"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "series", "week", "nfl_drives", "matched", "match_rate",
                                               "result_agree_rate", "play_count_agree_rate",
                                               "play_count_within1_rate", "games"])
            w.writeheader()
            w.writerows(tidy)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return {"report": report, "tidy": tidy}


def Main():
    ap = argparse.ArgumentParser(description="Independent drive rebuild vs nflverse drive fields.")
    ap.add_argument("--season", required=True, help="one season or comma list")
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    a = ap.parse_args()
    seasons = [int(x) for x in a.season.split(",") if x.strip()]
    out = DriveCheck(seasons, a.out_dir, a.base)
    print(json.dumps({k: v for k, v in out["report"].items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    Main()
