r"""
One command, from E:\Football\football:

    .venv\Scripts\python.exe -m Backtest.RunBacktest                      # 2023-2026, 5-pt edge
    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2025 --edge 0.03

Prints a per-season table and writes to data/backtest/reports/:
    report_<season>.json        summary per strategy
    bets_<season>_<strategy>.csv every bet with the running bankroll (the bankroll path)
    REPORT.md                   plain-language report for all seasons run
"""

import argparse
import csv
import json
import os

from Backtest.Engine import STAKE, STARTING_PURSE, RunSeason, Summarize
from Backtest.Model import HOME_FIELD_POINTS, MARGIN_SCALE, PRIOR_GAMES, SeasonAverageMargins
from Backtest.Strategies import AlwaysHome, HomeOnSameGames, ModelEdge

DATA_DIR = os.path.join("data", "backtest")
REPORT_DIR = os.path.join(DATA_DIR, "reports")


def LoadGames(season: int) -> list:
    path = os.path.join(DATA_DIR, f"games_{season}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def RunOne(season: int, edge: float) -> dict:
    games = LoadGames(season)
    if not games:
        raise SystemExit(f"missing data/backtest/games_{season}.json - run GetSeasonGames.py first")
    prior = SeasonAverageMargins(LoadGames(season - 1))  # last season is fully finished -> no leak
    edge_strategy = ModelEdge(edge)
    strategies = [AlwaysHome(), edge_strategy, HomeOnSameGames(edge_strategy)]
    books = RunSeason(games, strategies, prior)
    scheduled = len(games)
    completed = sum(g.get("home_score") is not None for g in games)
    result = {"season": season, "partial": completed < scheduled, "games_scheduled": scheduled,
              "games_completed": completed, "prior_season_used": bool(prior),
              "strategies": {s.name: {"description": s.description, **Summarize(books[s.name])} for s in strategies}}
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, f"report_{season}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    for name, book in books.items():
        if book["bets"]:
            with open(os.path.join(REPORT_DIR, f"bets_{season}_{name}.csv"), "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(book["bets"][0]))
                writer.writeheader()
                writer.writerows(book["bets"])
    return result


def TableLines(result: dict) -> list:
    label = f"{result['season']}" + (f" (PARTIAL: {result['games_completed']}/{result['games_scheduled']} games)"
                                     if result["partial"] else f" ({result['games_completed']} games incl. playoffs)")
    lines = [f"### {label}", "", "| Strategy | Start | End | Profit | ROI (purse) | ROI (staked) | Bets | W-L-P | Win rate | Low point |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in result["strategies"].items():
        win_rate = f"{s['win_rate'] * 100:.1f}%" if s["win_rate"] is not None else "n/a"
        roi_staked = f"{s['roi_on_staked'] * 100:+.1f}%" if s["roi_on_staked"] is not None else "n/a"
        lines.append(f"| {name} | ${s['start']:,.0f} | ${s['end']:,.2f} | ${s['profit']:+,.2f} | {s['roi_on_purse'] * 100:+.1f}% "
                     f"| {roi_staked} | {s['bets']} | {s['wins']}-{s['losses']}-{s['pushes']} | {win_rate} | ${s['min_bankroll']:,.0f} |")
    return lines + [""]


def WriteMarkdown(results: list, edge: float) -> str:
    lines = ["# NFL moneyline backtest", "",
             "## Rules in plain language", "",
             f"- **Purse:** every strategy starts each season with ${STARTING_PURSE:,.0f} (reset per season).",
             f"- **Stake:** flat ${STAKE:,.0f} per bet (2% of the starting purse), never more than what's left.",
             "- **Games:** only completed games (regular season + playoffs) with both scores and both moneylines, "
             "in date order. Ties are pushes.",
             "- **Presented odds:** the schedule's closing moneylines, turned into win probabilities and de-vigged "
             "(both sides scaled to sum to 100%).",
             "- **Calculated odds:** each team's season-to-date average point margin from games on *earlier days only*, "
             f"padded with {PRIOR_GAMES} virtual games at half of last season's average margin. Predicted home margin = "
             f"home rating - away rating + {HOME_FIELD_POINTS} pts home field; win prob = 1 / (1 + e^(-margin / {MARGIN_SCALE})).",
             f"- **always_home:** bet the home moneyline every game.",
             f"- **model_edge_{int(round(edge * 100))}pct:** bet the side whose calculated win prob beats the presented "
             f"win prob by at least {edge * 100:.0f} points; otherwise no bet.",
             "- **always_home_on_..._games:** control - bet home only on the games the edge strategy bet "
             "(same games, same stakes), so you can see whether picking the side added anything.",
             "- **Low point** = lowest bankroll during the season. Full bankroll paths are in `bets_<season>_<strategy>.csv`.",
             ""]
    for result in results:
        lines += TableLines(result)
    path = os.path.join(REPORT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def Main() -> None:
    parser = argparse.ArgumentParser(description="NFL moneyline bankroll backtest")
    parser.add_argument("--season", default="2023,2024,2025,2026", help="comma-separated seasons")
    parser.add_argument("--edge", type=float, default=0.05, help="min model - market win prob to bet (0.05 = 5 pts)")
    args = parser.parse_args()
    results = [RunOne(int(s), args.edge) for s in args.season.split(",") if s.strip()]
    for result in results:
        print("\n".join(TableLines(result)))
    print(f"Report: {WriteMarkdown(results, args.edge)}")


if __name__ == "__main__":
    Main()
