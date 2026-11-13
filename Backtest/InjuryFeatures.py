r"""
Leak-free team-game availability features from nflverse injury reports, weighted by LAGGED snap usage.

    .venv\Scripts\python.exe -m Backtest.InjuryFeatures --first 2013 --last 2026      # writes data/injuries/features_<season>.jsonl

Inputs (DataPull.PullPregame): data/injuries/<season>.jsonl, data/snaps/<season>.jsonl, data/players/players.jsonl,
schedules from Backtest.Lab.LoadSeason.

Per team-game (team T, game G on gameday D, season S):
  usage(p)  = mean of player p's snap share over T's last <=3 games of season S with gameday < D (0 for a game
              p did not play). If T has no earlier game in S (week 1), p's last <=3 appearances in season S-1
              (any team). Offense share for offensive players, defense share for defensive players (whichever
              of p's lagged offense/defense share is larger). Same-game snaps are NEVER used.
  status    = report_status on T's injury report for (S, week of G): the final pre-game designation.
  qb_out    = max usage of a QB listed Out/Doubtful (starter out ~1, backup ~0)
  off_out   = sum of usage of non-QB offensive players listed Out/Doubtful (player-equivalents)
  def_out   = sum of usage of defensive players listed Out/Doubtful
  qb_q, off_q, def_q = the same for Questionable
  report_rows = injury-report rows found for T that week (0 = no report in the data -> coverage gap, features 0)

Known limits: players already on IR / PUP drop off the report, so long-term absences show only through lagged usage
(which is then ~0 and the feature is ~0 - by design this measures NEW absences). Doubtful counts as out.
"""

import argparse
import json
import os
import re
from collections import defaultdict

from Backtest.Lab import LoadSeason

OUT_STATUSES = {"Out", "Doubtful"}
Q_STATUSES = {"Questionable"}
TEAM_FIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
LAG_GAMES = 3


def Team(t):
    return TEAM_FIX.get(t, t) if t else t


def NameKey(name):
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", (name or "").lower()))


def ReadJsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def PfrToGsis(players=None):
    players = players if players is not None else ReadJsonl(os.path.join("data", "players", "players.jsonl"))
    return {p["pfr_id"]: p["gsis_id"] for p in players if p.get("pfr_id") and p.get("gsis_id")}


class SnapHistory:
    """Snap shares by team-game, keyed so a lookup only ever sees games strictly before a date."""

    def __init__(self, snap_rows, games, pfr_map):
        day_of = {g["game_id"]: str(g["gameday"]) for g in games}
        self.team_games = defaultdict(dict)        # team -> {gameday: {pkey: (off, def)}}
        self.player_apps = defaultdict(list)       # pkey -> [(gameday, off, def)]
        for r in snap_rows:
            day = day_of.get(r["game_id"])
            if day is None:
                continue
            team = Team(r["team"])
            off, de = float(r.get("offense_pct") or 0.0), float(r.get("defense_pct") or 0.0)
            keys = self.Keys(r.get("pfr_player_id"), team, r.get("player"), pfr_map)
            slot = self.team_games[team].setdefault(day, {})
            for k in keys:
                slot[k] = (off, de)
                self.player_apps[k].append((day, off, de))
        for k in self.player_apps:
            self.player_apps[k].sort()

    @staticmethod
    def Keys(pfr_id, team, name, pfr_map):
        keys = [f"nm:{team}:{NameKey(name)}"]
        g = pfr_map.get(pfr_id) if pfr_id else None
        if g:
            keys.insert(0, f"gs:{g}")
        return keys

    def RecentTeamUsage(self, team, before_day, pkeys):
        days = sorted(d for d in self.team_games.get(team, {}) if d < before_day)[-LAG_GAMES:]
        if not days:
            return None
        for k in pkeys:   # first key that appears in any lag game
            if any(k in self.team_games[team][d] for d in days):
                vals = [self.team_games[team][d].get(k, (0.0, 0.0)) for d in days]
                return (sum(v[0] for v in vals) / len(days), sum(v[1] for v in vals) / len(days))
        return (0.0, 0.0)

    def LastApps(self, pkeys, before_day):
        for k in pkeys:
            apps = [a for a in self.player_apps.get(k, []) if a[0] < before_day][-LAG_GAMES:]
            if apps:
                return (sum(a[1] for a in apps) / len(apps), sum(a[2] for a in apps) / len(apps))
        return None


def BuildSeason(season, pfr_map=None, games=None, prev_games=None, injuries=None, snaps=None, prev_snaps=None):
    pfr_map = pfr_map if pfr_map is not None else PfrToGsis()
    games = games if games is not None else LoadSeason(season)
    prev_games = prev_games if prev_games is not None else LoadSeason(season - 1)
    injuries = injuries if injuries is not None else ReadJsonl(os.path.join("data", "injuries", f"{season}.jsonl"))
    snaps = snaps if snaps is not None else ReadJsonl(os.path.join("data", "snaps", f"{season}.jsonl"))
    prev_snaps = prev_snaps if prev_snaps is not None else ReadJsonl(os.path.join("data", "snaps", f"{season - 1}.jsonl"))
    cur = SnapHistory(snaps, games, pfr_map)
    prev = SnapHistory(prev_snaps, prev_games, pfr_map)
    report = defaultdict(list)
    for r in injuries:
        report[(int(r["week"]), Team(r["team"]))].append(r)
    out = []
    for g in sorted(games, key=lambda x: (str(x["gameday"]), x["game_id"])):
        day = str(g["gameday"])
        for side in ("home", "away"):
            team = Team(g[f"{side}_team"])
            rows = report.get((int(g["week"]), team), [])
            f = {"game_id": g["game_id"], "season": season, "week": g["week"], "gameday": day, "team": team, "side": side,
                 "report_rows": len(rows), "qb_out": 0.0, "off_out": 0.0, "def_out": 0.0, "qb_q": 0.0, "off_q": 0.0,
                 "def_q": 0.0, "n_out": 0, "n_q": 0, "n_unweighted": 0, "usage_source": None}
            has_prior = any(d < day for d in cur.team_games.get(team, {}))
            f["usage_source"] = "team_last3_this_season" if has_prior else "player_last3_prev_season"
            for r in rows:
                status = r.get("report_status")
                if status not in OUT_STATUSES and status not in Q_STATUSES:
                    continue
                pkeys = ([f"gs:{r['gsis_id']}"] if r.get("gsis_id") else []) + [f"nm:{team}:{NameKey(r.get('full_name'))}"]
                use = cur.RecentTeamUsage(team, day, pkeys) if has_prior else prev.LastApps(pkeys, "9999")
                if use is None:
                    use = (0.0, 0.0)
                    f["n_unweighted"] += 1
                off, de = use
                is_out = status in OUT_STATUSES
                f["n_out" if is_out else "n_q"] += 1
                sfx = "out" if is_out else "q"
                if (r.get("position") or "") == "QB":
                    f[f"qb_{sfx}"] = max(f[f"qb_{sfx}"], off)
                elif off >= de:
                    f[f"off_{sfx}"] += off
                else:
                    f[f"def_{sfx}"] += de
            for k in ("qb_out", "off_out", "def_out", "qb_q", "off_q", "def_q"):
                f[k] = round(f[k], 4)
            out.append(f)
    return out


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2013)
    ap.add_argument("--last", type=int, default=2026)
    a = ap.parse_args()
    pfr_map = PfrToGsis()
    for s in range(a.first, a.last + 1):
        rows = BuildSeason(s, pfr_map=pfr_map)
        path = os.path.join("data", "injuries", f"features_{s}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        cov = sum(r["report_rows"] > 0 for r in rows)
        print(s, "team-games", len(rows), "with report", cov, "qb_out>0.5", sum(r["qb_out"] > 0.5 for r in rows),
              "unweighted", sum(r["n_unweighted"] for r in rows), "of", sum(r["n_out"] + r["n_q"] for r in rows), flush=True)


if __name__ == "__main__":
    Main()