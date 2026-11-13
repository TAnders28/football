r"""
Build leak-free team-game efficiency tables (EPA/play, success rate, YPP, QB EPA).

    .venv\Scripts\python.exe -m DataPull.BuildTeamGameEff --first 2014 --last 2026

Uses local `data/playbyplay/pbp_<season>.jsonl` when present (2023-2026).
For earlier seasons, loads pbp from nflverse (nflreadpy) and keeps only the
team-game aggregate — full pbp is not written. Resumable per season.
Writes (gitignored): data/backtest/team_game_eff_<season>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from Backtest.Lab import LoadSeason
from Backtest.MarketBlend import SetBelowNormal

OUT = os.path.join("data", "backtest")


def IsScrimmage(p):
    if not p.get("posteam"):
        return False
    if p.get("epa") is None:
        return False
    return p.get("pass") in (1, 1.0, True) or p.get("rush") in (1, 1.0, True)


def IterLocalPbp(season):
    path = os.path.join("data", "playbyplay", f"pbp_{season}.jsonl")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def IterNflversePbp(season):
    import nflreadpy as nfl
    df = nfl.load_pbp(seasons=[season])
    if hasattr(df, "iter_rows"):
        for row in df.iter_rows(named=True):
            yield row
        return
    # pandas fallback
    for rec in df.to_dict("records"):
        yield rec


def IterPbp(season):
    local = os.path.join("data", "playbyplay", f"pbp_{season}.jsonl")
    if os.path.exists(local):
        yield from IterLocalPbp(season)
        return
    yield from IterNflversePbp(season)


def Accumulators():
    return {
        "off_plays": 0, "off_epa": 0.0, "off_succ": 0, "off_yards": 0.0,
        "def_plays": 0, "def_epa": 0.0, "def_succ": 0, "def_yards": 0.0,
        "qb": defaultdict(lambda: {"plays": 0, "epa": 0.0}),
    }


def AddPlay(units, p):
    off, de = p.get("posteam"), p.get("defteam")
    if not off:
        return
    gid = p["game_id"]
    u = units.setdefault((gid, off), Accumulators())
    yards = p.get("yards_gained")
    try:
        yards = 0.0 if yards is None else float(yards)
    except (TypeError, ValueError):
        yards = 0.0
    epa = float(p["epa"])
    succ = 1 if p.get("success") in (1, 1.0, True) else 0
    u["off_plays"] += 1
    u["off_epa"] += epa
    u["off_succ"] += succ
    u["off_yards"] += yards
    if de:
        d = units.setdefault((gid, de), Accumulators())
        d["def_plays"] += 1
        d["def_epa"] += epa
        d["def_succ"] += succ
        d["def_yards"] += yards
    qb = p.get("passer_id") or p.get("passer_player_id")
    if p.get("rush") in (1, 1.0, True):
        qb = p.get("rusher_id") or p.get("rusher_player_id") or qb
    if qb:
        slot = u["qb"][str(qb)]
        slot["plays"] += 1
        slot["epa"] += epa


def BuildSeason(season, games=None):
    games = games if games is not None else LoadSeason(season)
    gmeta = {g["game_id"]: g for g in games}
    units = {}
    n_plays = 0
    for p in IterPbp(season):
        if not IsScrimmage(p):
            continue
        AddPlay(units, p)
        n_plays += 1
    rows = []
    by_game = defaultdict(dict)
    for (gid, team), u in units.items():
        by_game[gid][team] = u
    for g in games:
        gid = g["game_id"]
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            u = by_game.get(gid, {}).get(team) or Accumulators()
            qb_id = g.get(f"{side}_qb_id")
            qb_stat = (u["qb"] or {}).get(str(qb_id)) if qb_id else None
            if qb_stat is None and u["qb"]:
                # fall back to the team's busiest passer/rusher that game
                qb_id, qb_stat = max(u["qb"].items(), key=lambda kv: kv[1]["plays"])
            rows.append({
                "game_id": gid, "season": int(g["season"]), "week": g.get("week"),
                "gameday": str(g["gameday"]), "team": team, "side": side,
                "off_plays": u["off_plays"], "off_epa": round(u["off_epa"], 6),
                "off_succ": u["off_succ"], "off_yards": round(u["off_yards"], 3),
                "def_plays": u["def_plays"], "def_epa": round(u["def_epa"], 6),
                "def_succ": u["def_succ"], "def_yards": round(u["def_yards"], 3),
                "qb_id": qb_id, "qb_plays": 0 if not qb_stat else qb_stat["plays"],
                "qb_epa": 0.0 if not qb_stat else round(qb_stat["epa"], 6),
            })
    return rows, n_plays


def WriteSeason(season, rows, n_plays, source):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"team_game_eff_{season}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta_path = os.path.join(OUT, f"team_game_eff_{season}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({
            "season": season, "n_rows": len(rows), "n_scrimmage_plays": n_plays,
            "source": source, "fetched_at": datetime.now(timezone.utc).isoformat(),
        }, fh, indent=1)
    return path


def SeasonSource(season):
    if os.path.exists(os.path.join("data", "playbyplay", f"pbp_{season}.jsonl")):
        return "local_pbp_jsonl"
    return "nflreadpy"


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2014)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--below-normal", action="store_true")
    a = ap.parse_args(argv)
    if a.below_normal:
        SetBelowNormal()
    written = []
    for s in range(a.first, a.last + 1):
        path = os.path.join(OUT, f"team_game_eff_{s}.jsonl")
        if os.path.exists(path) and not a.refresh:
            print(f"[{s}] exists, skip", flush=True)
            continue
        src = SeasonSource(s)
        print(f"[{s}] building from {src}", flush=True)
        rows, n = BuildSeason(s)
        WriteSeason(s, rows, n, src)
        print(f"[{s}] rows={len(rows)} scrimmage={n}", flush=True)
        written.append(s)
    print(json.dumps({"written": written, "first": a.first, "last": a.last}), flush=True)
    return written


if __name__ == "__main__":
    Main()
