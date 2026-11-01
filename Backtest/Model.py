"""
CALCULATED odds: a transparent season-to-date point-margin model (no betting lines used).

For every team we keep its point margin per game from games ALREADY FINISHED (prior days only).
    rating = (sum of margins + PRIOR_GAMES * prior_center) / (games + PRIOR_GAMES)
prior_center is last season's average margin, regressed halfway to 0 (0 if we have no prior season),
so week 1 isn't blank and one blowout doesn't dominate early weeks - same idea as baseball's
ShrunkPythagWithPrior.

Predicted home margin = home rating - away rating + HOME_FIELD_POINTS
Home win probability  = 1 / (1 + exp(-predicted_margin / MARGIN_SCALE))
MARGIN_SCALE ~7.4 makes a logistic match the classic "NFL margin ~ normal with sd 13.5" rule,
e.g. a 3-point favourite wins ~60%, a 7-point favourite ~72%.
"""

import math
from collections import defaultdict
from typing import Optional

PRIOR_GAMES = 4            # how many "virtual games" of prior_center each team starts with
PRIOR_REGRESS = 0.5        # keep half of last season's average margin
HOME_FIELD_POINTS = 1.5    # modern NFL home-field advantage in points
MARGIN_SCALE = 7.4         # points -> probability (logistic scale)


class MarginModel:
    def __init__(self, prior_margins: Optional[dict] = None):
        """prior_margins: {team: avg margin per game last season} (complete season, so no leak)."""
        self.prior = {t: PRIOR_REGRESS * m for t, m in (prior_margins or {}).items()}
        self.margin_sum = defaultdict(float)
        self.games = defaultdict(int)

    def Rating(self, team: str) -> float:
        center = self.prior.get(team, 0.0)
        return (self.margin_sum[team] + PRIOR_GAMES * center) / (self.games[team] + PRIOR_GAMES)

    def PredictedHomeMargin(self, home: str, away: str) -> float:
        return self.Rating(home) - self.Rating(away) + HOME_FIELD_POINTS

    def HomeWinProb(self, home: str, away: str) -> float:
        return 1.0 / (1.0 + math.exp(-self.PredictedHomeMargin(home, away) / MARGIN_SCALE))

    def Apply(self, game: dict) -> None:
        """Feed a FINISHED game. The engine only calls this after every bet for the day is placed."""
        margin = game["home_score"] - game["away_score"]
        self.margin_sum[game["home_team"]] += margin
        self.margin_sum[game["away_team"]] -= margin
        self.games[game["home_team"]] += 1
        self.games[game["away_team"]] += 1


def SeasonAverageMargins(games: list) -> dict:
    """{team: average point margin per game} over a whole (finished) season's games."""
    total, count = defaultdict(float), defaultdict(int)
    for g in games:
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        margin = g["home_score"] - g["away_score"]
        total[g["home_team"]] += margin
        total[g["away_team"]] -= margin
        count[g["home_team"]] += 1
        count[g["away_team"]] += 1
    return {t: total[t] / count[t] for t in total}
