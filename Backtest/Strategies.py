"""
Strategies look at one game's pre-game view and return "home", "away" or None (no bet).

    view = {"home_team", "away_team", "home_moneyline", "away_moneyline",
            "market_home_prob", "market_away_prob",   # PRESENTED odds, de-vigged
            "model_home_prob", "model_away_prob"}     # CALCULATED odds (Model.MarginModel)

The view never contains the score, so a strategy can't peek at the result.
"""

from typing import Optional


class AlwaysHome:
    """Baseline: bet the home team's moneyline in every game."""
    name = "always_home"
    description = "Bet the home team on the moneyline every game."

    def Decide(self, view: dict) -> Optional[str]:
        return "home"


class ModelEdge:
    """Bet the side whose calculated win probability beats the presented (de-vigged) probability
    by at least `min_edge`. If neither side clears the bar, skip the game."""
    name = "model_edge"

    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge
        self.name = f"model_edge_{int(round(min_edge * 100))}pct"
        self.description = (f"Bet the side where model win prob - market (no-vig) win prob >= "
                            f"{min_edge * 100:.0f} percentage points; otherwise no bet.")

    def Decide(self, view: dict) -> Optional[str]:
        home_edge = view["model_home_prob"] - view["market_home_prob"]
        away_edge = view["model_away_prob"] - view["market_away_prob"]
        if home_edge >= self.min_edge and home_edge >= away_edge:
            return "home"
        if away_edge >= self.min_edge:
            return "away"
        return None


class HomeOnSameGames:
    """Control: bet home only on the games `inner` chose to bet - isolates the side-picking skill."""

    def __init__(self, inner):
        self.inner = inner
        self.name = f"always_home_on_{inner.name}_games"
        self.description = f"Bet home, but only on the games {inner.name} bet."

    def Decide(self, view: dict) -> Optional[str]:
        return "home" if self.inner.Decide(view) else None
