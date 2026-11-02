r"""
Strategy hunt: sane-stake volume, underdog EV, momentum, ML-vs-spread, rest/division, slate timing,
and a portfolio (sparse Kelly core + volume satellite).

    .venv\Scripts\python.exe -m Backtest.Hunt      # writes data/backtest/reports/hunt_leaderboard.csv + hunt_results.json

Fairness (same as Backtest.Lab / Backtest.Volume):
  * features from earlier days only; models fit on earlier seasons only (walk-forward); closing moneylines only;
  * every strategy starts each season at $10,000; stakes sized from the bankroll at the start of the gameday;
  * every rule is deterministic (no seeds);
  * SELECTION uses 2015-2022 only (tune). 2023-2025 is reported out-of-sample. 2026 partial = context.
  * Section 7 (divisional complement) was added after a first pass; treat it as one extra look at the data.
  * Selection gate ("tune-qualified"): up in >= 6 of 8 tune seasons, worst tune season >= $8,500,
    tune pooled ROI on stakes > 0, and >= 15 tune bets per season on average. Ranked by tune avg end.
"""

import csv
import json
import os
import statistics

from Backtest.Lab import (START, TUNE, TEST, PARTIAL, REPORT_DIR, AllRows, AttachModels, Decimal, Kelly,
                          MLCandidate, RunSingles, SelAlwaysHome, Flat, PctFlat)
from Backtest.Volume import WalkForwardCut, Best

SEASONS = TUNE + TEST + PARTIAL


def Ensemble(rows_by_season):
    for rows in rows_by_season.values():
        for r in rows:
            r["p_ens"] = (r["p_full2"] + r["p_full3"]) / 2


# ----------------------------------------------------------------------------- selectors
def Sel(model, min_edge=0.0, max_edge=None, filt=None, side_rule="best"):
    """Generic: candidate = model's best side (or a forced side), keep if edge window + filter pass."""
    def f(r):
        if side_rule == "fav":
            s = "home" if r["mkt_home"] >= 0.5 else "away"
        elif side_rule == "dog":
            s = "away" if r["mkt_home"] >= 0.5 else "home"
        else:
            s = None
        p = r[model]
        if s is None:
            c = max((MLCandidate(r, "home", p), MLCandidate(r, "away", 1 - p)), key=lambda c: c["edge"])
        else:
            c = MLCandidate(r, s, p if s == "home" else 1 - p)
        if c["edge"] < min_edge or (max_edge is not None and c["edge"] >= max_edge):
            return []
        if filt is not None and not filt(r, c):
            return []
        return [c]
    return f


def SideVal(r, c, key):
    """A home-minus-away feature from the picked side's point of view."""
    return r[key] if c["side"] == "home" else -r[key]


FILTERS = {
    "": None,
    "fav": lambda r, c: not c["dog"],
    "dog": lambda r, c: c["dog"],
    "dog_upto+250": lambda r, c: c["dog"] and c["odds"] <= 250,
    "homedog": lambda r, c: c["dog"] and c["side"] == "home",
    "fav<=-150": lambda r, c: not c["dog"] and c["odds"] <= -150,
    "wk>=5": lambda r, c: r["week"] >= 5,
    "fav_wk>=5": lambda r, c: not c["dog"] and r["week"] >= 5,
    "rested": lambda r, c: SideVal(r, c, "rest_diff") > 0,
    "div": lambda r, c: r["div"] == 1,
    "nondiv": lambda r, c: r["div"] == 0,
    "night": lambda r, c: r["slate"].endswith("night"),
    "sunday_day": lambda r, c: r["slate"].endswith(("1pm", "4pm")),
    "cold_fade": lambda r, c: SideVal(r, c, "ats3_diff") < 0,      # picked side has been covering LESS lately
    "hot_form": lambda r, c: SideVal(r, c, "form3_diff") > 0,
    "sprd_agrees": lambda r, c: (r["p_sprdemp"] - r["mkt_home"]) * (1 if c["side"] == "home" else -1) > 0,
    "reg": lambda r, c: r["game_type"] == "REG",
}


def SelSpreadGap(min_gap, filt=None):
    """ML vs key-number-aware spread disagreement: back the side the spread likes more than the moneyline."""
    def f(r):
        if r["spread"] is None:
            return []
        g = r["p_sprdemp"] - r["mkt_home"]
        side = "home" if g > 0 else "away"
        p = r["p_sprdemp"] if side == "home" else 1 - r["p_sprdemp"]
        c = MLCandidate(r, side, p)
        if abs(g) < min_gap or (filt and not filt(r, c)):
            return []
        return [c]
    return f


def SelFadeAts3(min_ats3, filt=None):
    def f(r):
        if r["games_played"] < 3 or abs(r["ats3_diff"]) < min_ats3:
            return []
        side = "away" if r["ats3_diff"] > 0 else "home"
        c = MLCandidate(r, side, r["p_full2"] if side == "home" else 1 - r["p_full2"])
        return [] if filt and not filt(r, c) else [c]
    return f


def SelHeavyFav(min_p, filt=None):
    def f(r):
        side = "home" if r["mkt_home"] >= 0.5 else "away"
        pm = r["mkt_home"] if side == "home" else 1 - r["mkt_home"]
        if pm < min_p:
            return []
        c = MLCandidate(r, side, r["p_full2"] if side == "home" else 1 - r["p_full2"])
        return [] if filt and not filt(r, c) else [c]
    return f


def Union(*sels):
    def f(r):
        out = []
        for i, s in enumerate(sels):
            got = s(r)
            if got:
                out = [dict(got[0], tier=i)]   # one bet per game: first rule that fires wins (core listed first)
                break
        return out
    return f


# ----------------------------------------------------------------------------- sizing
def Combo(core_edge=0.08, core=None, sat_pct=0.03):
    core = core or Kelly(0.5, 0.10)
    def f(c, bank):
        return core(c, bank) if c.get("tier", 0) == 0 and c["edge"] >= core_edge else bank * sat_pct
    f.label = f"core[{core.label}]+sat{sat_pct*100:g}%"
    return f


SIZERS = [PctFlat(0.02), PctFlat(0.03), PctFlat(0.05), Kelly(0.25, 0.05), Kelly(0.5, 0.10)]


# ----------------------------------------------------------------------------- grid (strategy = season -> selector)
def Grid(rows_by_season):
    g = {}
    def add(name, factory, sizers=SIZERS):
        for sz in sizers:
            g[f"{name} | {sz.label}"] = (factory, sz)

    # 1) volume via walk-forward quantile cut-off (25/30/35% of games), sane stakes
    for model in ("p_full2", "p_full3", "p_ens", "p_full"):
        for fav in (False, True):
            for q in (0.25, 0.30, 0.35):
                fname = "fav" if fav else ""
                def fac(s, model=model, fav=fav, q=q, fname=fname):
                    cut = WalkForwardCut(rows_by_season, model, s, q, fav)
                    return Sel(model, cut, filt=FILTERS[fname], side_rule="fav" if fav else "best")
                add(f"vol_{model[2:]}{'_fav' if fav else ''}_q{int(q*100)}", fac)
    # 2) fixed edge windows x context filters (model = full2/full3/ens)
    for model in ("p_full2", "p_full3", "p_ens"):
        for lo in (0.0, 0.02, 0.03, 0.05):
            for fname in FILTERS:
                add(f"edge_{model[2:]}>={int(lo*100)}{('_' + fname) if fname else ''}",
                    lambda s, model=model, lo=lo, fname=fname: Sel(model, lo, filt=FILTERS[fname]))
    # 3) ML vs spread disagreement (key-number aware)
    for gap in (0.01, 0.02, 0.03, 0.05):
        for fname in ("", "fav", "dog", "wk>=5"):
            add(f"sprdgap>={gap*100:g}{('_' + fname) if fname else ''}",
                lambda s, gap=gap, fname=fname: SelSpreadGap(gap, FILTERS[fname]))
    # 4) momentum fade (ML on the side that has NOT been covering)
    for m in (3.0, 6.0, 10.0):
        for fname in ("", "fav", "dog", "sprd_agrees"):
            add(f"fade_ats3>{m:g}{('_' + fname) if fname else ''}",
                lambda s, m=m, fname=fname: SelFadeAts3(m, FILTERS[fname]), [PctFlat(0.02), PctFlat(0.03), PctFlat(0.05)])
    # 5) heavy favourites singles (favourite-longshot bias), optionally model must agree
    for mp in (0.65, 0.70, 0.75, 0.80):
        for fname in ("", "sprd_agrees", "wk>=5"):
            add(f"heavyfav>={int(mp*100)}{('_' + fname) if fname else ''}",
                lambda s, mp=mp, fname=fname: SelHeavyFav(mp, FILTERS[fname]), [PctFlat(0.03), PctFlat(0.05), PctFlat(0.10)])
    # 6) portfolio: sparse core (full2 edge>=8, half/full Kelly) + volume satellite at 2-3%
    for sat_model, sat_lo, sat_f in (("p_full3", 0.02, "fav"), ("p_full3", 0.03, "fav"), ("p_full2", 0.03, "fav"),
                                      ("p_ens", 0.02, "fav"), ("p_ens", 0.03, ""), ("p_full3", 0.03, "fav_wk>=5")):
        for core in (Kelly(0.5, 0.10), Kelly(1.0, 0.20)):
            for pct in (0.02, 0.03):
                name = f"PORT core=full2_edge8 + sat={sat_model[2:]}>={int(sat_lo*100)}_{sat_f or 'any'}"
                g[f"{name} | {Combo(0.08, core, pct).label}"] = (
                    lambda s, sm=sat_model, lo=sat_lo, sf=sat_f: Union(Sel("p_full2", 0.08), Sel(sm, lo, filt=FILTERS[sf])),
                    Combo(0.08, core, pct))
    # 7) divisional complement (added after the first hunt pass showed divisional rows topping the TUNE ranking):
    #    sparse core (full2 edge>=8) + divisional games where full3 likes a side by >= lo
    for lo in (0.0, 0.02, 0.05):
        div = lambda s, lo=lo: Union(Sel("p_full2", 0.08), Sel("p_full3", lo, filt=FILTERS["div"]))
        g[f"PORT core=full2_edge8 + div_full3>={int(lo*100)} | kelly0.5_cap10% (all)"] = (div, Kelly(0.5, 0.10))
        g[f"PORT core=full2_edge8 + div_full3>={int(lo*100)} | kelly0.25_cap5% (all)"] = (div, Kelly(0.25, 0.05))
        g[f"PORT core=full2_edge8 + div_full3>={int(lo*100)} | {Combo(0.08, Kelly(0.5, 0.10), 0.02).label}"] = (div, Combo(0.08, Kelly(0.5, 0.10), 0.02))
    # baselines
    g["BASE value_full2_edge8 | kelly0.5_cap10%"] = (lambda s: Sel("p_full2", 0.08), Kelly(0.5, 0.10))
    g["BASE value_full2_edge8 | kelly1_cap20%"] = (lambda s: Sel("p_full2", 0.08), Kelly(1.0, 0.20))
    g["BASE always_home | flat$200"] = (lambda s: SelAlwaysHome, Flat(200))
    def spec(s):
        cut = WalkForwardCut(rows_by_season, "p_full2", s, 0.30, True)
        return Sel("p_full2", cut, side_rule="fav")
    g["BASE losing spec vol_full2_fav_q30 | 20%bank"] = (spec, PctFlat(0.20))
    return g


def Evaluate(rows_by_season, grid):
    res = {}
    for name, (factory, sz) in grid.items():
        per = {}
        for s in SEASONS:
            rows = rows_by_season.get(s) or []
            r = RunSingles(rows, factory(s), sz)
            r["games"] = len(rows)
            per[s] = r
        res[name] = per
    return res


def Board(res):
    board = []
    for name, per in res.items():
        tune, test = [per[s] for s in TUNE], [per[s] for s in TEST]
        b = {
            "strategy": name,
            "tune_avg_end": statistics.mean(t["end"] for t in tune), "tune_up": sum(t["end"] > START for t in tune),
            "tune_min_end": min(t["end"] for t in tune),
            "tune_roi": sum(t["profit"] for t in tune) / max(1, sum(t["staked"] for t in tune)),
            "tune_bets_per_season": statistics.mean(t["bets"] for t in tune),
            "tune_maxdd": max(t["maxdd"] for t in tune),
            **{f"end_{s}": per[s]["end"] for s in TEST + PARTIAL},
            **{f"bets_{s}": per[s]["bets"] for s in TEST},
            "test_vol": sum(t["bets"] for t in test) / sum(t["games"] for t in test),
            "test_profit": sum(t["profit"] for t in test),
            "test_up": sum(t["end"] > START for t in test),
            "test_roi": sum(t["profit"] for t in test) / max(1, sum(t["staked"] for t in test)),
            "test_maxdd": max(t["maxdd"] for t in test),
        }
        b["qualified"] = (b["tune_up"] >= 6 and b["tune_min_end"] >= 8500 and b["tune_roi"] > 0
                          and b["tune_bets_per_season"] >= 15)
        board.append(b)
    return board


def Line(b):
    return (f"{b['strategy'][:78]:78s} tune {b['tune_avg_end']:7,.0f} up{b['tune_up']} min{b['tune_min_end']:6,.0f} roi{b['tune_roi']*100:+5.1f}% n{b['tune_bets_per_season']:4.0f} dd{b['tune_maxdd']*100:3.0f}% |"
            f" 23 {b['end_2023']:7,.0f} 24 {b['end_2024']:7,.0f} 25 {b['end_2025']:7,.0f} test {b['test_profit']:+7,.0f} up{b['test_up']}"
            f" vol{b['test_vol']*100:3.0f}% roi{b['test_roi']*100:+5.1f}% dd{b['test_maxdd']*100:3.0f}% 26p {b['end_2026']:7,.0f}")


def Main():
    rows_by_season = AllRows()
    AttachModels(rows_by_season)
    Ensemble(rows_by_season)
    res = Evaluate(rows_by_season, Grid(rows_by_season))
    board = Board(res)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "hunt_leaderboard.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(board[0]))
        w.writeheader()
        w.writerows(sorted(board, key=lambda b: (not b["qualified"], -b["tune_avg_end"])))
    with open(os.path.join(REPORT_DIR, "hunt_results.json"), "w", encoding="utf-8") as fh:
        json.dump({n: {s: {k: v for k, v in r.items() if k != "rows"} for s, r in per.items()} for n, per in res.items()},
                  fh, indent=1)
    return rows_by_season, res, board


if __name__ == "__main__":
    _rows, _res, board = Main()
    q = [b for b in board if b["qualified"]]
    print(f"{len(board)} strategies, {len(q)} tune-qualified")
    print("## TUNE-QUALIFIED, ranked by tune avg end (no hindsight)")
    for b in sorted(q, key=lambda b: -b["tune_avg_end"])[:50]:
        print(Line(b))
    print("## tune-qualified by family (best tune row per family)")
    fams = {}
    for b in sorted(q, key=lambda b: -b["tune_avg_end"]):
        fam = b["strategy"].split("|")[0].split(">=")[0].split("_q")[0].strip()
        fams.setdefault(fam, b)
    for b in fams.values():
        print(Line(b))
    print("## baselines")
    for b in board:
        if b["strategy"].startswith("BASE"):
            print(Line(b))
    print("## by TEST profit (hindsight, any)")
    for b in sorted(board, key=lambda b: -b["test_profit"])[:15]:
        print(Line(b))
