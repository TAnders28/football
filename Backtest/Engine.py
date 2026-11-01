"""
Bankroll backtest for one season. Walks games day by day (like baseball's Engine):

    for each game day D (chronological):
        1. every game from days before D is already in the model (the day barrier)
        2. build each game's pre-game view (presented + calculated odds, no scores) and let every
           strategy decide; stake = flat STAKE (capped by money not already on the table today)
        3. settle D's bets, THEN feed D's results to the model

Only completed games with both scores and both moneylines are bet. Each strategy has its own purse,
reset to STARTING_PURSE every season. A tie is a push (stake back).
"""

from itertools import groupby

from Backtest.Model import MarginModel
from Backtest.Odds import FairProbs, Settle

STARTING_PURSE = 10_000.0
STAKE = 200.0   # flat $200 per bet = 2% of the starting purse; never more than what's left


def Bettable(game: dict) -> bool:
    return all(game.get(k) is not None for k in ("home_score", "away_score", "home_moneyline", "away_moneyline"))


def PreGameView(game: dict, model: MarginModel) -> dict:
    market_home, market_away = FairProbs(game["home_moneyline"], game["away_moneyline"])
    model_home = model.HomeWinProb(game["home_team"], game["away_team"])
    return {"game_id": game["game_id"], "week": game["week"], "gameday": game["gameday"],
            "home_team": game["home_team"], "away_team": game["away_team"],
            "home_moneyline": game["home_moneyline"], "away_moneyline": game["away_moneyline"],
            "market_home_prob": market_home, "market_away_prob": market_away,
            "model_home_prob": model_home, "model_away_prob": 1.0 - model_home}


def RunSeason(games: list, strategies: list, prior_margins: dict = None) -> dict:
    """Returns {strategy name: {"purse": final, "bets": [bet rows with running bankroll]}}."""
    model = MarginModel(prior_margins)
    books = {s.name: {"purse": STARTING_PURSE, "bets": []} for s in strategies}
    ordered = sorted((g for g in games if Bettable(g)), key=lambda g: (g["gameday"], g.get("gametime") or "", g["game_id"]))
    for _day, day_games in groupby(ordered, key=lambda g: g["gameday"]):
        day_games = list(day_games)
        views = [PreGameView(g, model) for g in day_games]          # decided before any result is known
        for strategy in strategies:
            book = books[strategy.name]
            available = book["purse"]
            placed = []
            for game, view in zip(day_games, views):
                side = strategy.Decide(view)
                stake = min(STAKE, available)
                if side is None or stake <= 0:
                    continue
                available -= stake
                placed.append((game, view, side, stake))
            for game, view, side, stake in placed:
                margin = game["home_score"] - game["away_score"]
                won = None if margin == 0 else (margin > 0) == (side == "home")
                odds = view[f"{side}_moneyline"]
                change = Settle(stake, odds, won)
                book["purse"] += change
                book["bets"].append({
                    "game_id": game["game_id"], "week": game["week"], "gameday": game["gameday"],
                    "matchup": f"{game['away_team']} @ {game['home_team']}",
                    "pick": side, "pick_team": game[f"{side}_team"], "odds": odds, "stake": stake,
                    "market_prob": round(view[f"market_{side}_prob"], 4),
                    "model_prob": round(view[f"model_{side}_prob"], 4),
                    "edge": round(view[f"model_{side}_prob"] - view[f"market_{side}_prob"], 4),
                    "score": f"{game['away_score']}-{game['home_score']}",
                    "result": "push" if won is None else ("win" if won else "loss"),
                    "profit": round(change, 2), "bankroll": round(book["purse"], 2)})
        for game in day_games:
            model.Apply(game)
    return books


def Summarize(book: dict) -> dict:
    bets = book["bets"]
    staked = sum(b["stake"] for b in bets)
    wins = sum(b["result"] == "win" for b in bets)
    losses = sum(b["result"] == "loss" for b in bets)
    profit = book["purse"] - STARTING_PURSE
    return {"start": STARTING_PURSE, "end": round(book["purse"], 2), "profit": round(profit, 2),
            "roi_on_purse": round(profit / STARTING_PURSE, 4),
            "roi_on_staked": round(profit / staked, 4) if staked else None,
            "bets": len(bets), "wins": wins, "losses": losses, "pushes": len(bets) - wins - losses,
            "win_rate": round(wins / (wins + losses), 4) if wins + losses else None,
            "total_staked": round(staked, 2),
            "min_bankroll": round(min([STARTING_PURSE] + [b["bankroll"] for b in bets]), 2)}
