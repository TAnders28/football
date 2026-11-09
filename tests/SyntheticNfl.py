"""Deterministic fake NFL seasons for tests (no network, no data/ folder)."""

import copy
import math
import random
from datetime import date, timedelta

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC"]
SUNDAY_TIMES = ["13:00", "13:00", "13:00", "13:00", "13:00", "16:25", "16:25"]


def American(p):
    p = min(max(p, 0.03), 0.97)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def SyntheticSeason(season, weeks=17, seed=0):
    rng = random.Random(seed * 1000 + season)
    strength = {t: rng.gauss(0, 4) for t in TEAMS}
    first_sunday = date(season, 9, 7)
    while first_sunday.weekday() != 6:
        first_sunday += timedelta(days=1)
    games = []
    for week in range(1, weeks + 1):
        teams = TEAMS[:]
        rng.shuffle(teams)
        sunday = first_sunday + timedelta(days=7 * (week - 1))
        pairs = [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]
        for k, (away, home) in enumerate(pairs):
            if k == 0:
                day, t = sunday - timedelta(days=3), "20:15"      # Thursday night
            elif k == len(pairs) - 1:
                day, t = sunday, "20:20"                          # Sunday night
            else:
                day, t = sunday, SUNDAY_TIMES[(k - 1) % len(SUNDAY_TIMES)]
            diff = strength[home] - strength[away] + 1.5
            p_home = 1 / (1 + math.exp(-(diff + rng.gauss(0, 1.5)) / 7.4))
            margin = round(rng.gauss(diff, 13))
            hs = max(0, 21 + margin // 2 + margin % 2)
            as_ = max(0, hs - margin)
            games.append({
                "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "week": week,
                "gameday": day.isoformat(), "gametime": t, "game_type": "REG",
                "home_team": home, "away_team": away,
                "home_moneyline": American(p_home + 0.02), "away_moneyline": American(1 - p_home + 0.02),
                "spread_line": round(diff * 2) / 2, "home_spread_odds": -110, "away_spread_odds": -110,
                "home_rest": 7, "away_rest": 7, "div_game": int(rng.random() < 0.3),
                "home_score": hs, "away_score": as_,
            })
    return games


def Perturb(games, cutoff_day, seed=99, drop_after=True):
    """Copy of `games` where every game ON OR AFTER cutoff_day gets random scores, and games strictly
    after cutoff_day are removed (drop_after). Games before cutoff_day are untouched."""
    rng = random.Random(seed)
    out = []
    for g in games:
        g = copy.deepcopy(g)
        if g["gameday"] >= cutoff_day:
            if drop_after and g["gameday"] > cutoff_day:
                continue
            g["home_score"], g["away_score"] = rng.randint(0, 50), rng.randint(0, 50)
        out.append(g)
    return out
