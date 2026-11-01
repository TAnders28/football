"""American-odds helpers: presented odds -> implied probability, de-vig, and bet payouts."""

from typing import Optional, Tuple


def ImpliedProb(american: float) -> float:
    """Raw implied probability of American odds, vig included. -198 -> 0.664, +164 -> 0.379."""
    if american < 0:
        return -american / (-american + 100.0)
    return 100.0 / (american + 100.0)


def FairProbs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    """Remove the bookmaker's vig by scaling both sides so they sum to 1 (proportional de-vig)."""
    home, away = ImpliedProb(home_ml), ImpliedProb(away_ml)
    total = home + away
    return home / total, away / total


def Profit(stake: float, american: float) -> float:
    """Profit (not including the returned stake) if a bet at these odds WINS."""
    return stake * (american / 100.0 if american > 0 else 100.0 / -american)


def Settle(stake: float, american: float, won: Optional[bool]) -> float:
    """Bankroll change for a settled moneyline bet. won=None means a tie -> push (stake returned)."""
    if won is None:
        return 0.0
    return Profit(stake, american) if won else -stake
