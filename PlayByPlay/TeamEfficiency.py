r"""
Team efficiency by week from nflverse play-by-play: EPA per play and success rate, offense and defense.

    .venv\Scripts\python.exe PlayByPlay\TeamEfficiency.py --season 2026
    .venv\Scripts\python.exe PlayByPlay\TeamEfficiency.py --season 2026 --out-dir DIR --base NAME

Reads   data/playbyplay/pbp_<season>.jsonl   (no network)

Play filter (the usual nflfastR convention): `pass == 1 or rush == 1` and `epa` present. That keeps
scrambles and sacks as pass plays and keeps penalty no-plays that nflverse still flags as pass/rush;
it drops kicks, other special-teams plays and timeouts. Success = nflverse `success` (epa > 0).
Defense numbers are the opponent's offense against that team, so LOWER defensive EPA/play is better.
`std_*` columns are season-to-date through that week (play-weighted), which is what a pre-game model
for the NEXT week could use. Weekly numbers for one game are noisy (~60 plays).

Writes  <out-dir>/<base>.csv   tidy: date, series(team), week, opponent, off_plays, off_epa_play,
                               off_success_rate, off_pass_epa_play, off_rush_epa_play, def_plays, def_epa_play,
                               def_success_rate, std_off_epa_play, std_def_epa_play, std_net_epa_play
        <out-dir>/<base>.json  season-to-date table sorted by net EPA/play
        default out-dir: stats/season/
"""

import argparse
import csv
import json
import os
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(ROOT, "data", "playbyplay")
DEFAULT_OUT = os.path.join(ROOT, "stats", "season")
KEEP = ("game_id", "game_date", "week", "posteam", "defteam", "home_team", "away_team", "pass", "rush", "epa", "success")


def ReadPbp(season: int):
    with open(os.path.join(PBP_DIR, f"pbp_{season}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                yield {k: row.get(k) for k in KEEP}


def IsScrimmage(p) -> bool:
    return (p.get("pass") in (1, 1.0) or p.get("rush") in (1, 1.0)) and p.get("epa") is not None and bool(p.get("posteam"))


def _new():
    return {"plays": 0, "epa": 0.0, "succ": 0, "pass_n": 0, "pass_epa": 0.0, "rush_n": 0, "rush_epa": 0.0}


def GameUnits(plays) -> dict:
    """{(game_id, team): offense accumulator + meta} from scrimmage plays."""
    units = OrderedDict()
    for p in plays:
        if not IsScrimmage(p):
            continue
        key = (p["game_id"], p["posteam"])
        u = units.get(key)
        if u is None:
            u = units[key] = dict(_new(), game_id=p["game_id"], team=p["posteam"], opponent=p.get("defteam"),
                                  week=p.get("week"), date=p.get("game_date"))
        u["plays"] += 1
        u["epa"] += p["epa"]
        u["succ"] += 1 if p.get("success") in (1, 1.0) else 0
        if p.get("pass") in (1, 1.0):
            u["pass_n"] += 1
            u["pass_epa"] += p["epa"]
        else:
            u["rush_n"] += 1
            u["rush_epa"] += p["epa"]
    return units


def _div(a, b):
    return round(a / b, 4) if b else None


def WeeklyTable(plays) -> list:
    units = GameUnits(plays)
    rows = []
    for (gid, team), u in units.items():
        d = units.get((gid, u["opponent"])) or _new()
        rows.append({"date": u["date"], "series": team, "week": u["week"], "opponent": u["opponent"], "game_id": gid,
                     "off_plays": u["plays"], "off_epa": u["epa"], "off_succ": u["succ"],
                     "off_pass_n": u["pass_n"], "off_pass_epa": u["pass_epa"], "off_rush_n": u["rush_n"], "off_rush_epa": u["rush_epa"],
                     "def_plays": d["plays"], "def_epa": d["epa"], "def_succ": d["succ"]})
    rows.sort(key=lambda r: (r["series"], r["date"] or "", r["game_id"]))
    cum = defaultdict(lambda: {"op": 0, "oe": 0.0, "dp": 0, "de": 0.0})
    out = []
    for r in rows:
        c = cum[r["series"]]
        c["op"] += r["off_plays"]; c["oe"] += r["off_epa"]; c["dp"] += r["def_plays"]; c["de"] += r["def_epa"]
        so, sd = _div(c["oe"], c["op"]), _div(c["de"], c["dp"])
        out.append({"date": r["date"], "series": r["series"], "week": r["week"], "opponent": r["opponent"],
                    "off_plays": r["off_plays"], "off_epa_play": _div(r["off_epa"], r["off_plays"]),
                    "off_success_rate": _div(r["off_succ"], r["off_plays"]),
                    "off_pass_epa_play": _div(r["off_pass_epa"], r["off_pass_n"]),
                    "off_rush_epa_play": _div(r["off_rush_epa"], r["off_rush_n"]),
                    "def_plays": r["def_plays"], "def_epa_play": _div(r["def_epa"], r["def_plays"]),
                    "def_success_rate": _div(r["def_succ"], r["def_plays"]),
                    "std_off_epa_play": so, "std_def_epa_play": sd,
                    "std_net_epa_play": None if so is None or sd is None else round(so - sd, 4)})
    out.sort(key=lambda r: (r["date"] or "", r["series"]))
    return out


def SeasonTable(weekly) -> list:
    last = {}
    agg = defaultdict(lambda: {"games": 0, "op": 0, "dp": 0, "osr": 0.0, "dsr": 0.0})
    for r in weekly:
        last[r["series"]] = r
        a = agg[r["series"]]
        a["games"] += 1
        a["op"] += r["off_plays"]; a["dp"] += r["def_plays"]
        a["osr"] += (r["off_success_rate"] or 0) * r["off_plays"]
        a["dsr"] += (r["def_success_rate"] or 0) * r["def_plays"]
    table = [{"team": t, "games": agg[t]["games"], "off_plays": agg[t]["op"], "off_epa_play": r["std_off_epa_play"],
              "def_epa_play": r["std_def_epa_play"], "net_epa_play": r["std_net_epa_play"],
              "off_success_rate": _div(agg[t]["osr"], agg[t]["op"]), "def_success_rate": _div(agg[t]["dsr"], agg[t]["dp"]),
              "through": r["date"]} for t, r in last.items()]
    table.sort(key=lambda r: -(r["net_epa_play"] or -9))
    return table


COLS = ["date", "series", "week", "opponent", "off_plays", "off_epa_play", "off_success_rate", "off_pass_epa_play",
        "off_rush_epa_play", "def_plays", "def_epa_play", "def_success_rate", "std_off_epa_play", "std_def_epa_play",
        "std_net_epa_play"]


def TeamEfficiency(season: int, plays=None, out_dir=None, base=None, write=True):
    weekly = WeeklyTable(plays if plays is not None else ReadPbp(season))
    table = SeasonTable(weekly)
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or f"team_efficiency_{season}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(weekly)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump({"season": season, "team_games": len(weekly), "teams": len(table), "table": table}, fh, indent=1)
    return {"weekly": weekly, "table": table}


def Main():
    ap = argparse.ArgumentParser(description="Weekly team EPA/play and success rate from nflverse play-by-play.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    a = ap.parse_args()
    out = TeamEfficiency(a.season, out_dir=a.out_dir, base=a.base)
    for r in out["table"]:
        print(f"{r['team']:4s} g{r['games']:2d} off {r['off_epa_play']:+.3f} def {r['def_epa_play']:+.3f} net {r['net_epa_play']:+.3f}"
              f"  sr {r['off_success_rate']:.3f}/{r['def_success_rate']:.3f}")


if __name__ == "__main__":
    Main()
