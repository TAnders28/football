r"""
Strategy lab: many fair strategies x bet-sizing rules, walk-forward, one leaderboard.

    .venv\Scripts\python.exe -m Backtest.FetchHistory     # once: 2012-2022 schedules for training/tuning
    .venv\Scripts\python.exe -m Backtest.Lab              # writes data/backtest/reports/REPORT_v2.md + lab_*.csv

Fairness rules
  * Features use only games from EARLIER days (Features.BuildRows day barrier).
  * Learned models for season S are fit only on seasons < S (walk-forward). 2012 is never bet.
  * Odds are the closing lines in the data (moneyline, spread + spread juice). Nothing invented,
    except the clearly-labelled FAIR-PARLAY HYPOTHETICAL (a no-vig parlay price no book offers).
  * Each strategy starts every season with $10,000. Stakes for a day are sized from the bankroll at
    the start of that day; if a day's stakes exceed the bankroll they are scaled down.
  * TUNE window = 2015-2022 (pick rules here), TEST window = 2023-2025 (report), 2026 = partial.
"""

import csv
import json
import math
import os
from collections import defaultdict
from itertools import groupby

import numpy as np

from Backtest.Features import BuildRows

START = 10_000.0
TUNE = list(range(2015, 2023))
TEST = [2023, 2024, 2025]
PARTIAL = [2026]
REPORT_DIR = os.path.join("data", "backtest", "reports")


# ----------------------------------------------------------------------------- data
def LoadSeason(season: int) -> list:
    for path in (os.path.join("data", "backtest", f"games_{season}.json"),
                 os.path.join("data", "backtest", "history", f"games_{season}.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return []


def AllRows(first=2012, last=2026) -> dict:
    out, prev = {}, []
    for s in range(first, last + 1):
        games = LoadSeason(s)
        out[s] = BuildRows(games, prev)
        prev = games
    return out


def Decimal(american: float) -> float:
    return 1 + (american / 100.0 if american > 0 else 100.0 / -american)


def Sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------- walk-forward models
WIN_FEATS = ["mkt_logit", "edge_logit", "form3_diff", "ats_diff", "ats3_diff", "rest_diff", "div"]
WIN_FEATS2 = WIN_FEATS + ["spread_gap"]   # + how far the moneyline disagrees with the closing spread
WIN_FEATS3 = WIN_FEATS + ["spread_gap_emp"]  # same, but spread -> win% from an empirical half-point table (respects key numbers 3/7)


def EmpiricalSpreadTable(train: list) -> dict:
    """{|spread| rounded to 0.5: favourite win rate} from earlier seasons, smoothed with neighbours (+-1 pt)."""
    wins, n = defaultdict(float), defaultdict(int)
    for r in train:
        if r["spread"] is None or r["home_win"] is None:
            continue
        k = round(abs(r["spread"]) * 2) / 2
        fav_won = r["home_win"] if r["spread"] >= 0 else 1 - r["home_win"]
        wins[k] += fav_won; n[k] += 1
    table = {}
    for k in [x / 2 for x in range(0, 41)]:
        ws = sum(wins[k + d] * wt for d, wt in ((-1, .25), (-.5, .5), (0, 1), (.5, .5), (1, .25)))
        ns = sum(n[k + d] * wt for d, wt in ((-1, .25), (-.5, .5), (0, 1), (.5, .5), (1, .25)))
        table[k] = (ws + 0.5 * 5) / (ns + 5) if k == 0 else (ws + 5 * (0.5 + k * 0.03)) / (ns + 5)
    table[0.0] = 0.5
    return table


def EmpProb(table: dict, spread) -> float:
    if spread is None:
        return 0.5
    k = min(round(abs(spread) * 2) / 2, 20.0)
    p = min(max(table.get(k, 0.5), 0.02), 0.98)
    return p if spread >= 0 else 1 - p
COVER_FEATS = ["pts_vs_spread", "form3_diff", "ats_diff", "ats3_diff", "rest_diff", "div"]


def Vec(r: dict, feats: list) -> list:
    x = {"mkt_logit": r["mkt_logit"], "edge_logit": r["margin_logit"] - r["mkt_logit"],
         "form3_diff": r["form3_diff"] / 7, "ats_diff": r["ats_diff"] / 7, "ats3_diff": r["ats3_diff"] / 7,
         "rest_diff": r["rest_diff"], "div": r["div"],
         "pts_vs_spread": (r["margin_logit"] * 7.4 - (r["spread"] or 0)) / 7,
         "spread": (r["spread"] or 0) / 7,
         "spread_gap": r.get("spread_logit", r["mkt_logit"]) - r["mkt_logit"],
         "spread_gap_emp": r.get("spread_emp_logit", r["mkt_logit"]) - r["mkt_logit"]}
    return [1.0] + [x[f] for f in feats]


def FitLogit(X, y, l2=1.0):
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1]); reg[0, 0] = 0
    for _ in range(25):
        p = Sig(X @ w)
        g = X.T @ (p - y) + reg @ w
        H = X.T @ (X * (p * (1 - p))[:, None]) + reg
        w -= np.linalg.solve(H, g)
    return w


def AttachModels(rows_by_season: dict) -> dict:
    """Adds p_margin, p_calib, p_full (home win) and p_cover (home covers) to every row."""
    coefs = {}
    for s, rows in rows_by_season.items():
        train = [r for t, rs in rows_by_season.items() if 2013 <= t < s for r in rs]
        win_train = [r for r in train if r["home_win"] is not None]
        cov_train = [r for r in train if r["home_cover"] is not None and r["spread"] is not None]
        w_cal = w_full = w_cov = w_sp = w_full2 = w_full3 = None
        if len(win_train) > 200:
            w_cal = FitLogit([Vec(r, ["mkt_logit"]) for r in win_train], [r["home_win"] for r in win_train])
            w_full = FitLogit([Vec(r, WIN_FEATS) for r in win_train], [r["home_win"] for r in win_train], l2=5.0)
            w_sp = FitLogit([Vec(r, ["spread"]) for r in win_train], [r["home_win"] for r in win_train])
            for r in train + rows:
                r["spread_logit"] = float(np.dot(w_sp, Vec(r, ["spread"])))
            table = EmpiricalSpreadTable(train)
            for r in train + rows:
                pe = EmpProb(table, r["spread"])
                r["spread_emp_logit"] = math.log(pe / (1 - pe))
            w_full3 = FitLogit([Vec(r, WIN_FEATS3) for r in win_train], [r["home_win"] for r in win_train], l2=5.0)
            w_full2 = FitLogit([Vec(r, WIN_FEATS2) for r in win_train], [r["home_win"] for r in win_train], l2=5.0)
            w_cov = FitLogit([Vec(r, COVER_FEATS) for r in cov_train], [r["home_cover"] for r in cov_train], l2=5.0)
            coefs[s] = {"calib": w_cal.round(3).tolist(), "full": dict(zip(["const"] + WIN_FEATS, w_full.round(3).tolist())),
                        "cover": dict(zip(["const"] + COVER_FEATS, w_cov.round(3).tolist()))}
        for r in rows:
            r["p_mkt"] = r["mkt_home"]
            r["p_margin"] = float(Sig(r["margin_logit"]))
            r["p_calib"] = float(Sig(np.dot(w_cal, Vec(r, ["mkt_logit"])))) if w_cal is not None else r["mkt_home"]
            r["p_full"] = float(Sig(np.dot(w_full, Vec(r, WIN_FEATS)))) if w_full is not None else r["mkt_home"]
            r["p_sprd"] = float(Sig(np.dot(w_sp, Vec(r, ["spread"])))) if w_sp is not None else r["mkt_home"]
            r["p_full2"] = float(Sig(np.dot(w_full2, Vec(r, WIN_FEATS2)))) if w_full2 is not None else r["mkt_home"]
            r["p_sprdemp"] = float(Sig(r["spread_emp_logit"])) if w_full3 is not None else r["mkt_home"]
            r["p_full3"] = float(Sig(np.dot(w_full3, Vec(r, WIN_FEATS3)))) if w_full3 is not None else r["mkt_home"]
            r["p_cover"] = float(Sig(np.dot(w_cov, Vec(r, COVER_FEATS)))) if w_cov is not None else 0.5
    return coefs


# ----------------------------------------------------------------------------- bet candidates
def MLCandidate(r, side, p_model):
    odds = r["home_ml"] if side == "home" else r["away_ml"]
    p_mkt = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
    won = None if r["home_win"] is None else (r["home_win"] == 1) == (side == "home")
    return {"kind": "ML", "side": side, "team": r[f"{side}_team"], "odds": odds, "p": p_model, "p_mkt": p_mkt,
            "edge": p_model - p_mkt, "won": won, "dog": p_mkt < 0.5}


def ATSCandidate(r, side, p_model):
    odds = r[f"{side}_spread_odds"]
    won = None if r["home_cover"] is None else (r["home_cover"] == 1) == (side == "home")
    p_mkt = 1 / Decimal(odds) / (1 / Decimal(r["home_spread_odds"]) + 1 / Decimal(r["away_spread_odds"]))
    return {"kind": "ATS", "side": side, "team": r[f"{side}_team"], "odds": odds, "p": p_model, "p_mkt": p_mkt,
            "edge": p_model - p_mkt, "won": won, "dog": (r["spread"] or 0) > 0 and side == "away" or (r["spread"] or 0) < 0 and side == "home"}


# selection rules: row -> list of candidates
def SelAlwaysHome(r):
    return [MLCandidate(r, "home", r["mkt_home"])]


def SelAlwaysFav(r):
    side = "home" if r["mkt_home"] >= 0.5 else "away"
    return [MLCandidate(r, side, max(r["mkt_home"], 1 - r["mkt_home"]))]


def SelAlwaysDog(r):
    side = "away" if r["mkt_home"] >= 0.5 else "home"
    return [MLCandidate(r, side, min(r["mkt_home"], 1 - r["mkt_home"]))]


def SelEdge(model, min_edge, dogs_only=False, favs_only=False, min_games=0, max_odds=None):
    def f(r):
        if r["games_played"] < min_games:
            return []
        p = r[model]
        best = max((MLCandidate(r, "home", p), MLCandidate(r, "away", 1 - p)), key=lambda c: c["edge"])
        if best["edge"] < min_edge or (dogs_only and not best["dog"]) or (favs_only and best["dog"]):
            return []
        if max_odds is not None and best["odds"] > max_odds:
            return []
        return [best]
    return f


def SelATS(min_edge, dogs_only=False):
    def f(r):
        if r["spread"] is None or r["home_cover"] is None and r["result"] != r["spread"]:
            return []
        p = r["p_cover"]
        best = max((ATSCandidate(r, "home", p), ATSCandidate(r, "away", 1 - p)), key=lambda c: c["edge"])
        if best["edge"] < min_edge or (dogs_only and not best["dog"]):
            return []
        return [best]
    return f


def SelMomentumDog(min_ats3=3.0):
    """Underdog that has been beating the spread lately (ats3_diff from the dog's side)."""
    def f(r):
        dog = "away" if r["mkt_home"] >= 0.5 else "home"
        mom = r["ats3_diff"] if dog == "home" else -r["ats3_diff"]
        if r["games_played"] < 3 or mom < min_ats3:
            return []
        return [MLCandidate(r, dog, min(r["mkt_home"], 1 - r["mkt_home"]))]
    return f


def SelFadeMomentum(min_ats3=3.0):
    """Bet AGAINST the team that has been beating the spread lately (mean reversion)."""
    def f(r):
        if r["games_played"] < 3 or abs(r["ats3_diff"]) < min_ats3:
            return []
        side = "away" if r["ats3_diff"] > 0 else "home"
        p = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
        return [MLCandidate(r, side, p)]
    return f


# sizing rules: (candidate, bankroll at start of day) -> stake
def Flat(amount=200.0):
    def f(c, bank):
        return amount
    f.label = f"flat${amount:.0f}"
    return f


def PctFlat(pct=0.02):
    def f(c, bank):
        return bank * pct
    f.label = f"{pct*100:.0f}%bank"
    return f


def Tiered(unit=0.01, cuts=(0.04, 0.08), mults=(1, 2, 4)):
    def f(c, bank):
        m = mults[0] if c["edge"] < cuts[0] else (mults[1] if c["edge"] < cuts[1] else mults[2])
        return bank * unit * m
    f.label = f"tiered{unit*100:.0f}%x{'/'.join(map(str, mults))}"
    return f


def Kelly(frac=0.25, cap=0.05):
    def f(c, bank):
        b = Decimal(c["odds"]) - 1
        k = (b * c["p"] - (1 - c["p"])) / b
        return max(0.0, min(k * frac, cap)) * bank
    f.label = f"kelly{frac:g}_cap{cap*100:.0f}%"
    return f


# ----------------------------------------------------------------------------- engines
def RunSingles(rows, select, size):
    bank, peak, maxdd = START, START, 0.0
    bets = []
    for _d, day in groupby(rows, key=lambda r: r["gameday"]):
        day = list(day)
        start_bank = bank
        picks = [(r, c, size(c, start_bank)) for r in day for c in select(r)]
        picks = [(r, c, s) for r, c, s in picks if s >= 1.0]
        total = sum(s for _, _, s in picks)
        scale = min(1.0, start_bank / total) if total > 0 else 1.0
        for r, c, s in picks:
            s *= scale
            pnl = 0.0 if c["won"] is None else (s * (Decimal(c["odds"]) - 1) if c["won"] else -s)
            bank += pnl
            bets.append({"game_id": r["game_id"], "gameday": r["gameday"], "kind": c["kind"], "pick": c["team"],
                         "odds": c["odds"], "p_model": round(c["p"], 4), "p_mkt": round(c["p_mkt"], 4),
                         "edge": round(c["edge"], 4), "stake": round(s, 2), "won": c["won"],
                         "pnl": round(pnl, 2), "bankroll": round(bank, 2)})
        peak = max(peak, bank)
        maxdd = max(maxdd, (peak - bank) / peak if peak else 0)
    return Stats(bank, bets, maxdd)


def RunParlays(rows, min_prob=0.5, max_legs=None, pricing="fair", stake_pct=0.01, model_filter=None, min_legs=2):
    """One favorites parlay per kickoff slate (day | early/1pm/4pm/night).
    pricing='fair' -> payout = prod(1 / de-vigged prob)   [FAIR-PARLAY HYPOTHETICAL, no book offers this]
    pricing='book' -> payout = prod(posted decimal odds)   [what a standard book parlay pays]"""
    bank, peak, maxdd, bets = START, START, 0.0, []
    for _d, day in groupby(rows, key=lambda r: r["gameday"]):
        day = list(day)
        start_bank = bank
        for _s, slate in groupby(sorted(day, key=lambda r: r["slate"]), key=lambda r: r["slate"]):
            legs = []
            for r in slate:
                side = "home" if r["mkt_home"] >= 0.5 else "away"
                p = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
                if p < min_prob:
                    continue
                if model_filter:
                    pm = r[model_filter] if side == "home" else 1 - r[model_filter]
                    if pm < p:
                        continue
                won = None if r["home_win"] is None else (r["home_win"] == 1) == (side == "home")
                legs.append((p, Decimal(r[f"{side}_ml"]), won, r[f"{side}_team"]))
            legs.sort(key=lambda x: -x[0])
            if max_legs:
                legs = legs[:max_legs]
            if len(legs) < min_legs:
                continue
            live = [l for l in legs if l[2] is not None]          # tied legs are voided
            price = math.prod(1 / l[0] if pricing == "fair" else l[1] for l in live)
            won = all(l[2] for l in live)
            stake = min(start_bank * stake_pct, bank)
            pnl = stake * (price - 1) if won else -stake
            bank += pnl
            bets.append({"slate": legs and _s, "legs": "+".join(l[3] for l in legs), "n_legs": len(legs),
                         "price_decimal": round(price, 3), "fair_prob": round(math.prod(l[0] for l in live), 4),
                         "stake": round(stake, 2), "won": won, "pnl": round(pnl, 2), "bankroll": round(bank, 2)})
        peak = max(peak, bank)
        maxdd = max(maxdd, (peak - bank) / peak if peak else 0)
    return Stats(bank, bets, maxdd)


def Stats(bank, bets, maxdd):
    staked = sum(b["stake"] for b in bets)
    wins = sum(b["won"] is True for b in bets)
    losses = sum(b["won"] is False for b in bets)
    return {"end": bank, "profit": bank - START, "staked": staked, "roi_staked": (bank - START) / staked if staked else 0.0,
            "bets": len(bets), "wins": wins, "losses": losses, "maxdd": maxdd, "rows": bets}


# ----------------------------------------------------------------------------- strategy grid
def Grid():
    g = {}
    flat = Flat(200)
    g["always_home | flat$200"] = ("singles", SelAlwaysHome, flat)
    g["always_fav | flat$200"] = ("singles", SelAlwaysFav, flat)
    g["always_dog | flat$200"] = ("singles", SelAlwaysDog, flat)
    g["model_edge_5pct (v1 margin) | flat$200"] = ("singles", SelEdge("p_margin", 0.05), flat)
    for model in ("p_margin", "p_full", "p_calib", "p_sprd", "p_full2", "p_sprdemp", "p_full3"):
        for t in (0.02, 0.03, 0.05, 0.065, 0.08, 0.10):
            for dogs in (False, True):
                name = f"value_{model[2:]}_edge{int(t*100)}{'_dogs' if dogs else ''}"
                sel = SelEdge(model, t, dogs_only=dogs, min_games=0 if model != "p_margin" else 2)
                for sz in (flat, Tiered(), Kelly(0.25, 0.05), Kelly(0.5, 0.10), Kelly(1.0, 0.20)):
                    g[f"{name} | {sz.label}"] = ("singles", sel, sz)
    for t in (0.02, 0.03, 0.05):
        for dogs in (False, True):
            sel = SelATS(t, dogs)
            for sz in (Flat(200), Kelly(0.25, 0.05), Kelly(0.5, 0.10)):
                g[f"ats_cover_edge{int(t*100)}{'_dogs' if dogs else ''} | {sz.label}"] = ("singles", sel, sz)
    for m in (3.0, 6.0):
        g[f"momentum_dog_ats3>{m:g} | flat$200"] = ("singles", SelMomentumDog(m), flat)
        g[f"fade_momentum_ats3>{m:g} | flat$200"] = ("singles", SelFadeMomentum(m), flat)
    for pricing in ("fair", "book"):
        for minp in (0.5, 0.65, 0.75):
            for legs in (None, 2, 3, 4):
                for mf in (None, "p_full", "p_full2"):
                    name = (f"parlay_{pricing}{'_HYPOTHETICAL' if pricing == 'fair' else ''}_fav>={int(minp*100)}"
                            f"_{'all' if legs is None else str(legs)}legs{('_modelOK' + mf[2:]) if mf else ''} | 1%bank/slate")
                    g[name] = ("parlay", dict(min_prob=minp, max_legs=legs, pricing=pricing, model_filter=mf), None)
    return g


def Evaluate(rows_by_season, grid):
    out = {}
    for name, (kind, a, b) in grid.items():
        per = {}
        for s in TUNE + TEST + PARTIAL:
            rows = rows_by_season.get(s) or []
            if not rows:
                continue
            per[s] = RunSingles(rows, a, b) if kind == "singles" else RunParlays(rows, **a)
        out[name] = per
    return out


def Fmt(x):
    return f"${x:,.0f}"


def Main():
    rows_by_season = AllRows()
    coefs = AttachModels(rows_by_season)
    grid = Grid()
    res = Evaluate(rows_by_season, grid)
    board = []
    for name, per in res.items():
        tune_ends = [per[s]["end"] for s in TUNE if s in per]
        test = [per[s] for s in TEST if s in per]
        board.append({
            "strategy": name,
            "tune_avg_end": sum(tune_ends) / len(tune_ends), "tune_up_seasons": sum(e > START for e in tune_ends),
            **{f"end_{s}": per[s]["end"] for s in TEST + PARTIAL if s in per},
            **{f"bets_{s}": per[s]["bets"] for s in TEST + PARTIAL if s in per},
            **{f"dd_{s}": per[s]["maxdd"] for s in TEST if s in per},
            "test_total_profit": sum(t["profit"] for t in test),
            "test_up_seasons": sum(t["profit"] > 0 for t in test),
            "test_roi_staked": sum(t["profit"] for t in test) / max(1, sum(t["staked"] for t in test)),
            "test_max_dd": max(t["maxdd"] for t in test),
        })
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "lab_leaderboard.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(board[0]))
        w.writeheader()
        w.writerows(sorted(board, key=lambda b: -b["test_total_profit"]))
    with open(os.path.join(REPORT_DIR, "lab_coefficients.json"), "w", encoding="utf-8") as fh:
        json.dump(coefs, fh, indent=1)
    with open(os.path.join(REPORT_DIR, "lab_results.json"), "w", encoding="utf-8") as fh:
        json.dump({n: {s: {k: v for k, v in r.items() if k != "rows"} for s, r in per.items()} for n, per in res.items()}, fh, indent=1)
    return rows_by_season, res, board


if __name__ == "__main__":
    _rows, _res, board = Main()
    print(f"{len(board)} strategies")
    hdr = f"{'strategy':70s} {'tuneAvg':>9s} {'tUp':>3s} {'2023':>9s} {'2024':>9s} {'2025':>9s} {'test$':>9s} {'up':>2s} {'roi':>6s} {'dd':>5s} {'2026p':>9s}"
    print(hdr)
    for b in sorted(board, key=lambda b: -b["test_total_profit"])[:60]:
        print(f"{b['strategy'][:70]:70s} {b['tune_avg_end']:9,.0f} {b['tune_up_seasons']:3d} {b['end_2023']:9,.0f} {b['end_2024']:9,.0f} "
              f"{b['end_2025']:9,.0f} {b['test_total_profit']:+9,.0f} {b['test_up_seasons']:2d} {b['test_roi_staked']*100:+5.1f}% "
              f"{b['test_max_dd']*100:4.0f}% {b.get('end_2026', 0):9,.0f}")
