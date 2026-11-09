r"""
Frozen-strategy scorecard for one season: the clean forward holdout plus a clearly-labelled
pseudo-forward walk over every finished game of the season.

    .venv\Scripts\python.exe -m Backtest.ForwardWalk --season 2026
    .venv\Scripts\python.exe -m Backtest.ForwardWalk --season 2026 --out-dir E:\AgentDocs\American-Footbot\2026-11-09_x --base 2026-11-09_x

Reads   data/backtest/games_<season>.json (+ history) via Backtest.Lab.AllRows   (no network)
Writes  <out-dir>/<base>.csv          tidy daily equity: date, series, equity, drawdown_pct, bets, staked
        <out-dir>/<base>_bets.csv     one row per simulated bet (both fee bases)
        <out-dir>/<base>.json         summary (N, splits, consistency check, model_version, run_id)
        default out-dir: data/backtest/forward/

What this is and is not
  * Rules come ONLY from Backtest.FrozenConfig.FROZEN, applied exactly as PredictDay does (value first,
    divisional add-on second, one bet per game). Nothing here tunes anything.
  * FORWARD HOLDOUT = games with gameday > FROZEN["frozen_at"]. Only these are clean.
  * Everything else is a PSEUDO-FORWARD walk: the models are walk-forward (fit on earlier seasons only) and
    features use earlier days only, but the strategy searches printed this season's partial results while
    rules were being chosen, so games the search could see are NOT out-of-sample.
  * `seen_by_search` marks games already final in the data the Lab/Volume/Hunt searches read
    (games_<season>.json pulled SEARCH_PULLED_AT). Games after that are unseen by the search but were
    still played before the freeze, so they are "unseen", not a pre-registered holdout.
  * Prices are nflverse CLOSING moneylines (a game-time price, not one you could necessarily get earlier).
    Two P&L bases: "closing_ml" pays the posted American odds (sportsbook vig included, no extra fee);
    "tim_fee_est" buys $1 contracts at de-vigged prob + $0.02 (Tim's working fee estimate, NOT a verified
    exchange fee). Stakes are identical in both; only the payout differs.
  * Bankroll: START ($10,000) at the first game of the season, no reset inside the season.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from itertools import groupby
from zoneinfo import ZoneInfo

from Backtest.FrozenConfig import FROZEN, IsForwardHoldout
from Backtest.Hunt import FILTERS, Combo, Sel, Union
from Backtest.Lab import START, AllRows, AttachModels, Decimal, Kelly, RunSingles, SelAlwaysFav, SelEdge
from Backtest.Version import ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
FEE_EST = 0.02
FEE_BASIS = "tim_est_0.02"
# The committed searches (Lab 2026-11-01 17:13 ET, Volume 18:45 ET, Hunt 2026-11-02 13:34 ET) all read
# data/backtest/games_2026.json as pulled 2026-11-01 21:30:11 UTC; pull_summary then said 114 games final.
# Kickoff + 3h15m <= pull time marks 116 games as seen: conservative, since nflverse posts results with a lag,
# so the 2 extra (Nov 1, 1pm) are counted as seen rather than risk calling a seen game "unseen".
SEARCH_PULLED_AT = {2026: datetime(2026, 11, 1, 16, 30, 11, tzinfo=ET)}
GAME_LENGTH = timedelta(hours=3, minutes=15)
DEFAULT_OUT = os.path.join("data", "backtest", "forward")


def Kickoff(r) -> datetime:
    return datetime.fromisoformat(f"{r['gameday']}T{r.get('gametime') or '13:00'}").replace(tzinfo=ET)


def SeenBySearch(r) -> bool:
    """True if the game was (very likely) already final in the data the strategy searches read.
    Seasons without a recorded search pull time were fully visible to the search."""
    pulled = SEARCH_PULLED_AT.get(int(r["season"]))
    return True if pulled is None else Kickoff(r) + GAME_LENGTH <= pulled


def FrozenPicker(config=None):
    """(row, bank) -> (candidate, stake) or None; same order and sizing as PredictDay.ScoreRows."""
    cfg = config or FROZEN
    cv, cd = cfg["value"], cfg["div_addon"]
    value = SelEdge(cv["model"], cv["min_edge"])
    div = Sel(cd["model"], cd["min_edge"], filt=FILTERS[cd["filter"]])
    half = Kelly(cv["stake"]["fraction"], cv["stake"]["cap"])

    def pick(r, bank):
        if r.get("home_ml") is None or r.get("away_ml") is None:
            return None
        c = value(r)
        if c:
            return dict(c[0], rule=cv["rule"]), half(c[0], bank)
        c = div(r)
        if c:
            return dict(c[0], rule=cd["rule"]), bank * cd["stake"]["pct"]
        return None
    return pick


def MarketFavPicker(pct=0.02):
    """Baseline: back the market favourite every game, flat pct of bank."""
    def pick(r, bank):
        c = SelAlwaysFav(r)[0]
        return dict(c, rule=f"market_fav_{pct*100:g}pct"), bank * pct
    return pick


def Payout(c, basis) -> float:
    """Profit per $1 staked if the bet wins."""
    if basis == "closing_ml":
        return Decimal(c["odds"]) - 1
    return 1.0 / min(c["p_mkt"] + FEE_EST, 0.999) - 1


def Walk(rows, picker, basis="closing_ml", start=START):
    """One chronological pass. Stakes for a day are sized from the bank at the start of that day and
    scaled down if they exceed it (same as Lab.RunSingles). Returns (bets, daily)."""
    rows = sorted(rows, key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"]))
    bank, peak, bets, daily, staked_total = start, start, [], [], 0.0
    for day, group in groupby(rows, key=lambda r: r["gameday"]):
        group = list(group)
        start_bank = bank
        picks = [(r, *p) for r in group if (p := picker(r, start_bank))]
        picks = [(r, c, s) for r, c, s in picks if s >= 1.0]
        total = sum(s for _, _, s in picks)
        scale = min(1.0, start_bank / total) if total > 0 else 1.0
        for r, c, s in picks:
            s *= scale
            pnl = 0.0 if c["won"] is None else (s * Payout(c, basis) if c["won"] else -s)
            bank += pnl
            staked_total += s
            bets.append({"date": r["gameday"], "game_id": r["game_id"], "week": r["week"],
                         "matchup": f"{r['away_team']} @ {r['home_team']}", "rule": c["rule"], "pick": c["team"],
                         "side": c["side"], "odds": c["odds"], "p_model": round(c["p"], 4), "p_mkt_novig": round(c["p_mkt"], 4),
                         "edge_gross": round(c["edge"], 4), "edge_after_fee_est": round(c["edge"] - FEE_EST, 4),
                         "stake": round(s, 2), "won": c["won"], "pnl": round(pnl, 2), "bankroll": round(bank, 2),
                         "basis": basis, "seen_by_search": SeenBySearch(r),
                         "forward_holdout": IsForwardHoldout(r["gameday"])})
        peak = max(peak, bank)
        daily.append({"date": day, "equity": round(bank, 2), "drawdown_pct": round(bank / peak - 1, 4),
                      "bets": len(bets), "staked": round(staked_total, 2)})
    return bets, daily


def Split(bets, key=None):
    sel = [b for b in bets if key is None or key(b)]
    done = [b for b in sel if b["won"] is not None]
    staked = sum(b["stake"] for b in sel)
    pnl = sum(b["pnl"] for b in sel)
    wins = sum(b["won"] is True for b in done)
    return {"bets": len(sel), "wins": wins, "losses": len(done) - wins,
            "hit_rate": round(wins / len(done), 4) if done else None,
            "staked": round(staked, 2), "pnl": round(pnl, 2), "roi_staked": round(pnl / staked, 4) if staked else None}


def ForwardWalk(season=2026, rows_by_season=None, out_dir=None, base=None, write=True, now=None):
    if rows_by_season is None:
        rows_by_season = AllRows(2012, season)
        AttachModels(rows_by_season)
    rows = rows_by_season[season]
    run_id, version = NewRunId(now), ModelVersion()
    series = {}
    for name, picker, basis in (("frozen_closing_ml", FrozenPicker(), "closing_ml"),
                                ("frozen_tim_fee_est", FrozenPicker(), "tim_fee_est"),
                                ("market_fav_2pct_closing_ml", MarketFavPicker(0.02), "closing_ml")):
        series[name] = Walk(rows, picker, basis)
    bets = series["frozen_closing_ml"][0]
    fee_bets = series["frozen_tim_fee_est"][0]
    # consistency: the Hunt engine's version of the same portfolio must land on the same bankroll
    hunt = RunSingles(sorted(rows, key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"])),
                      Union(Sel("p_full2", FROZEN["value"]["min_edge"]),
                            Sel("p_full3", FROZEN["div_addon"]["min_edge"], filt=FILTERS["div"])),
                      Combo(FROZEN["value"]["min_edge"], Kelly(0.5, 0.10), FROZEN["div_addon"]["stake"]["pct"]))
    end = series["frozen_closing_ml"][1][-1]["equity"] if rows else START
    summary = {
        "season": season, "run_id": run_id, "model_version": version, "frozen_config": FROZEN["name"],
        "frozen_at": FROZEN["frozen_at"], "fee_basis_alt": FEE_BASIS,
        "games_final": len(rows), "first_game": rows[0]["gameday"] if rows else None,
        "last_game": max(r["gameday"] for r in rows) if rows else None,
        "games_seen_by_search": sum(SeenBySearch(r) for r in rows),
        "games_unseen_pre_freeze": sum(not SeenBySearch(r) and not IsForwardHoldout(r["gameday"]) for r in rows),
        "games_forward_holdout": sum(IsForwardHoldout(r["gameday"]) for r in rows),
        "closing_ml": {"all": Split(bets), "seen_by_search": Split(bets, lambda b: b["seen_by_search"]),
                       "unseen_pre_freeze": Split(bets, lambda b: not b["seen_by_search"] and not b["forward_holdout"]),
                       "forward_holdout": Split(bets, lambda b: b["forward_holdout"]),
                       "value_rule": Split(bets, lambda b: b["rule"] == FROZEN["value"]["rule"]),
                       "div_rule": Split(bets, lambda b: b["rule"] == FROZEN["div_addon"]["rule"]),
                       "end_bankroll": end,
                       "max_drawdown_pct": min((d["drawdown_pct"] for d in series["frozen_closing_ml"][1]), default=0.0)},
        "tim_fee_est": {"all": Split(fee_bets),
                        "end_bankroll": series["frozen_tim_fee_est"][1][-1]["equity"] if rows else START},
        "market_fav_2pct": {"all": Split(series["market_fav_2pct_closing_ml"][0]),
                            "end_bankroll": series["market_fav_2pct_closing_ml"][1][-1]["equity"] if rows else START},
        "consistency_hunt_engine_end": round(hunt["end"], 2),
        "consistency_ok": abs(hunt["end"] - end) < 0.05,
    }
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or f"forward_walk_{season}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "series", "equity", "drawdown_pct", "bets", "staked"])
            for name, (_b, daily) in series.items():
                for d in daily:
                    w.writerow([d["date"], name, d["equity"], d["drawdown_pct"], d["bets"], d["staked"]])
        with open(os.path.join(out_dir, f"{base}_bets.csv"), "w", newline="", encoding="utf-8") as fh:
            cols = list(bets[0].keys()) if bets else ["date"]
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for b in bets + fee_bets:
                w.writerow(b)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    return {"summary": summary, "series": series}


def Main():
    ap = argparse.ArgumentParser(description="Frozen-strategy forward scorecard + labelled pseudo-forward walk.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    a = ap.parse_args()
    out = ForwardWalk(a.season, out_dir=a.out_dir, base=a.base)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    Main()
