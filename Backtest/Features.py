"""
Pre-game feature rows for every bettable game, built with a strict day barrier: a game's features
only use games played on EARLIER days of the same season (plus the finished previous season).

Feature meanings (all from the home team's point of view, "diff" = home minus away):
    mkt_home        de-vigged closing moneyline win prob (PRESENTED odds)
    mkt_logit       log(mkt_home / (1 - mkt_home))
    margin_logit    season-to-date margin model (Model.MarginModel) expressed as a logit
    form3_diff      last-3-game avg point margin minus season rating (momentum: hot/cold streak)
    ats_diff        season-to-date avg (margin - closing spread): who has been beating the market
    ats3_diff       same, last 3 games only (short-term market momentum)
    rest_diff       days of rest difference / 7
    div             divisional game flag
"""

import math
from collections import defaultdict
from itertools import groupby

from Backtest.Model import MarginModel, SeasonAverageMargins
from Backtest.Odds import FairProbs


def Logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def Bettable(g: dict) -> bool:
    return all(g.get(k) is not None for k in ("home_score", "away_score", "home_moneyline", "away_moneyline"))


def SlateOf(g: dict) -> str:
    """Kickoff window in ET: early (London/9:30), 1pm, 4pm, night."""
    t = g.get("gametime") or "13:00"
    hour = int(t.split(":")[0])
    if hour < 12:
        return "early"
    if hour < 15:
        return "1pm"
    if hour < 18:
        return "4pm"
    return "night"


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def BuildRows(games: list, prior_games: list) -> list:
    model = MarginModel(SeasonAverageMargins(prior_games) if prior_games else {})
    margins = defaultdict(list)   # team -> list of point margins (team perspective)
    ats = defaultdict(list)       # team -> list of (margin - spread for that team)
    rows = []
    ordered = sorted((g for g in games if Bettable(g)), key=lambda g: (str(g["gameday"]), g.get("gametime") or "", g["game_id"]))
    for _day, day_games in groupby(ordered, key=lambda g: str(g["gameday"])):
        day_games = list(day_games)
        for g in day_games:
            h, a = g["home_team"], g["away_team"]
            mh, ma = FairProbs(g["home_moneyline"], g["away_moneyline"])
            rh, ra = model.Rating(h), model.Rating(a)
            pred_margin = model.PredictedHomeMargin(h, a)
            result = g["home_score"] - g["away_score"]
            spread = g.get("spread_line")
            rows.append({
                "game_id": g["game_id"], "season": g["season"], "week": g["week"], "gameday": str(g["gameday"]),
                "gametime": g.get("gametime"), "slate": f"{g['gameday']}|{SlateOf(g)}", "game_type": g.get("game_type"),
                "home_team": h, "away_team": a,
                "home_ml": g["home_moneyline"], "away_ml": g["away_moneyline"],
                "home_spread_odds": g.get("home_spread_odds") or -110, "away_spread_odds": g.get("away_spread_odds") or -110,
                "spread": spread,
                "mkt_home": mh, "mkt_logit": Logit(mh),
                "margin_logit": pred_margin / 7.4,
                "form3_diff": (_avg(margins[h][-3:]) - rh if margins[h] else 0.0) - (_avg(margins[a][-3:]) - ra if margins[a] else 0.0),
                "ats_diff": _avg(ats[h]) - _avg(ats[a]),
                "ats3_diff": _avg(ats[h][-3:]) - _avg(ats[a][-3:]),
                "rest_diff": ((g.get("home_rest") or 7) - (g.get("away_rest") or 7)) / 7.0,
                "div": float(g.get("div_game") or 0),
                "games_played": min(len(margins[h]), len(margins[a])),
                # outcomes (never used as features)
                "result": result,
                "home_win": None if result == 0 else (1 if result > 0 else 0),
                "home_cover": None if spread is None or result == spread else (1 if result > spread else 0),
            })
        for g in day_games:   # results enter the history only after the whole day is decided
            h, a = g["home_team"], g["away_team"]
            result = g["home_score"] - g["away_score"]
            margins[h].append(result)
            margins[a].append(-result)
            if g.get("spread_line") is not None:
                ats[h].append(result - g["spread_line"])
                ats[a].append(-result + g["spread_line"])
            model.Apply(g)
    return rows
