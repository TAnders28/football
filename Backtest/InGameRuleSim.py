r"""
Simulation only: in-game trading rules on held-out data (ingame_wp_v1 walk-forward predictions; no orders, ever).

    .venv\Scripts\python.exe -m Backtest.InGameRuleSim [--out-dir DIR --base NAME --boot 2000 --paths 5000 --seed 20261113]

Primary series: Kalshi KXNFLGAME home-market 1-minute candles (yes bid/ask close), 2025 + 2026 wk1-9.
Secondary check: Polymarket 1-minute prices-history (one price, NOT an executable book; mid fills only), 2024-2026 wk1-9.
Samples: 1-minute UTC grid inside the pbp in-play window (halftime and >10-min gaps excluded), latest quote <= 3 min old;
model = held-out ingame_wp_v1 P(home) for the latest snap at or before the minute (<= 10 min old).
Rules (all thresholds fixed BEFORE running; every variant counts toward the search tally):
  (a) close_<band>_gap<X>: home mid in band (40-60 or 35-65), Q1-Q2, |model - mid| > X (0.03 / 0.05); buy the model's side.
  (b) swing_<follow|fade>_<X>_<agree|any>: Q1-Q2, home mid moved >= X (0.10 / 0.15) over the last 5 minutes; follow = buy
      the side that rose, fade = buy the side that fell; "agree" also requires the model to be on that side of the mid.
  (c) fade_home_30_50: home mid in [0.30, 0.50) at any in-play minute; buy the away side (tests the calibration dip found
      on these same 2025-26 samples: fully in-sample / contaminated).
First qualifying minute per game only; hold to settlement (no exit rule tested). Ties dropped.
Fills: mid, and taker (buy at the ask: home ask, or 1 - home bid for the away side). Fees: Tim's $0.02/contract estimate,
and Kalshi's verified taker fee 0.07*p*(1-p) (continuous; Kalshi rounds up per order, so slightly understated).
Bankroll: continuous $10k from the first bet, bets grouped by game day (ET), stakes sized from the bank at the start of
the day: flat 2% of the starting $10k ($200/bet) and quarter Kelly from the model probability (cap 5%, skip if <= 0).
PathSim (Backtest.PathSim) on the daily returns of the primary fill (taker + Kalshi verified fee).
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import polars as pl

from Backtest.InGameCalibration import InPlayGrid, SeriesPoly, Ts, Windows
from Backtest.Lab import LoadSeason
from Backtest.PathSim import Evaluate

STALE_S, STATE_STALE_S = 180, 600
START = 10_000.0
FEES = {"tim_est_0.02": lambda p: 0.02, "kalshi_verified": lambda p: 0.07 * p * (1 - p)}
EARLY = (1, 2)
SWING_MIN = 5


def Rules():
    rules = {}
    for lo, hi in ((0.40, 0.60), (0.35, 0.65)):
        for X in (0.03, 0.05):
            rules[f"close_{int(lo*100)}_{int(hi*100)}_gap{int(X*100)}"] = {"kind": "close", "lo": lo, "hi": hi, "X": X}
    for X in (0.10, 0.15):
        for d in ("follow", "fade"):
            for ag in ("agree", "any"):
                rules[f"swing_{d}_{int(X*100)}_{ag}"] = {"kind": "swing", "X": X, "dir": d, "agree": ag == "agree"}
    rules["fade_home_30_50"] = {"kind": "fade_home", "lo": 0.30, "hi": 0.50}
    return rules


def Games():
    out = {}
    for s in (2024, 2025, 2026):
        for g in LoadSeason(s):
            if g.get("home_score") is None or (s == 2026 and int(g["week"]) > 9) or g["home_score"] == g["away_score"]:
                continue
            out[g["game_id"]] = g
    return out


def ModelPlays():
    plays = {}
    for S in (2024, 2025, 2026):
        df = pl.read_ndjson(os.path.join("data", "backtest", "ingame_model", f"heldout_plays_{S}.jsonl"), infer_schema_length=None)
        for gid, sub in df.group_by("game_id"):
            gid = gid[0] if isinstance(gid, tuple) else gid
            ts = np.array([Ts(x) or np.nan for x in sub["time_of_day"].to_list()], dtype=float)
            p = sub["p_home_model"].to_numpy().astype(float)
            ok = ~np.isnan(ts) & ~np.isnan(p)
            o = np.argsort(ts[ok], kind="stable")
            plays[gid] = (ts[ok][o], p[ok][o])
    return plays


def KalshiQuotes(row):
    """[(t, home_bid, home_ask)] from the stored home-market candles."""
    out = []
    for c in row["candles"]:
        b, a = c.get("yes_bid_close"), c.get("yes_ask_close")
        if b is None or a is None or a < b:
            continue
        if row["side_team"] != row["home_team"]:
            b, a = 1 - a, 1 - b
        out.append((c["end_ts"], b, a))
    return sorted(out)


def Samples(venue, games, plays, win):
    """Per game: list of dicts {t, qtr, mid, bid, ask, model} in time order."""
    stem, seasons = ("ingame_kalshi", (2025, 2026)) if venue == "Kalshi" else ("ingame_poly", (2024, 2025, 2026))
    out = {}
    for s in seasons:
        for l in open(os.path.join("data", "prices", f"{stem}_{s}.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            gid = r["game_id"]
            if gid not in games or gid not in win or gid not in plays:
                continue
            q = KalshiQuotes(r) if venue == "Kalshi" else [(t, p, p) for t, p in sorted(SeriesPoly(r))]
            if not q:
                continue
            qt = np.array([x[0] for x in q], dtype=float)
            pt, pp = plays[gid]
            rows = []
            for t, qtr in InPlayGrid(win[gid]):
                i = np.searchsorted(qt, t, side="right") - 1
                j = np.searchsorted(pt, t, side="right") - 1
                if i < 0 or t - qt[i] > STALE_S or j < 0 or t - pt[j] > STATE_STALE_S:
                    continue
                _, b, a = q[i]
                rows.append({"t": t, "qtr": qtr, "bid": b, "ask": a, "mid": (b + a) / 2, "model": float(pp[j])})
            if rows:
                out[gid] = rows
    return out


def Trigger(rule, rows):
    """First qualifying minute -> (sample, side 'home'|'away') or None."""
    mids = {r["t"]: r["mid"] for r in rows}
    for r in rows:
        m, p = r["mid"], r["model"]
        if rule["kind"] == "close":
            if r["qtr"] in EARLY and rule["lo"] <= m <= rule["hi"] and abs(p - m) > rule["X"]:
                return r, "home" if p > m else "away"
        elif rule["kind"] == "swing":
            if r["qtr"] not in EARLY:
                continue
            past = mids.get(r["t"] - SWING_MIN * 60)
            if past is None or abs(m - past) < rule["X"]:
                continue
            rose = "home" if m > past else "away"
            side = rose if rule["dir"] == "follow" else ("away" if rose == "home" else "home")
            if rule["agree"] and not ((side == "home" and p > m) or (side == "away" and p < m)):
                continue
            return r, side
        elif rule["kind"] == "fade_home":
            if rule["lo"] <= m < rule["hi"]:
                return r, "away"
    return None


def Price(r, side, fill):
    if fill == "mid":
        return r["mid"] if side == "home" else 1 - r["mid"]
    return r["ask"] if side == "home" else 1 - r["bid"]


def Bets(rule, samples, games):
    out = []
    for gid, rows in samples.items():
        hit = Trigger(rule, rows)
        if not hit:
            continue
        r, side = hit
        g = games[gid]
        home_won = g["home_score"] > g["away_score"]
        out.append({"game_id": gid, "season": int(g["season"]), "date": str(g["gameday"]), "t": r["t"], "qtr": r["qtr"], "side": side,
                    "won": int(home_won == (side == "home")), "mid_side": round(Price(r, side, "mid"), 4),
                    "ask_side": round(Price(r, side, "taker"), 4), "model_side": round(r["model"] if side == "home" else 1 - r["model"], 4)})
    return sorted(out, key=lambda b: (b["date"], b["t"]))


def Stats(bets, fill, fee, boot, rng):
    if not bets:
        return {"bets": 0}
    price = np.array([b["mid_side"] if fill == "mid" else b["ask_side"] for b in bets])
    f = np.array([FEES[fee](p) for p in price]) if fee else np.zeros(len(bets))
    won = np.array([b["won"] for b in bets], float)
    pnl = won - price - f
    bs = pnl[rng.integers(0, len(pnl), (boot, len(pnl)))].mean(1)
    return {"bets": len(bets), "hit_rate": round(float(won.mean()), 4), "avg_price": round(float(price.mean()), 4),
            "cents_per_contract": round(float(pnl.mean() * 100), 2),
            "cents_ci90": [round(float(np.quantile(bs, 0.05) * 100), 2), round(float(np.quantile(bs, 0.95) * 100), 2)],
            "roi": round(float(pnl.sum() / (price + f).sum()), 4)}


def Bankroll(bets, fill, fee, stake):
    """Continuous bank; bets settle by game day, stakes from the bank at the start of the day. -> daily [{date, equity, ret}]."""
    by_day = defaultdict(list)
    for b in bets:
        by_day[b["date"]].append(b)
    bank, peak, out = START, START, []
    for d in sorted(by_day):
        start_bank, pnl_day = bank, 0.0
        for b in by_day[d]:
            p = b["mid_side"] if fill == "mid" else b["ask_side"]
            cost = p + FEES[fee](p)
            if stake == "flat":
                amt = min(0.02 * START, max(start_bank, 0) / max(len(by_day[d]), 1))
            else:
                q = b["model_side"]
                k = (q - cost) / (1 - cost) if cost < 1 else 0
                amt = min(0.25 * k, 0.05) * start_bank if k > 0 else 0.0
            n = amt / cost if cost > 0 else 0
            pnl_day += n * (b["won"] - cost)
        bank = start_bank + pnl_day
        peak = max(peak, bank)
        out.append({"date": d, "equity": round(bank, 2), "drawdown_pct": round(100 * (bank - peak) / peak, 2) if peak > 0 else 0.0,
                    "ret": pnl_day / start_bank if start_bank > 0 else 0.0})
    return out


def Charts(summary, curves, path_bar, path_eq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Backtest.BankrollPlot import FrameBankroll, PlotBankroll
    names = list(summary)
    fig, ax = plt.subplots(figsize=(14, 7))
    combos = [("mid", "tim_est_0.02", "#9467bd"), ("taker", "tim_est_0.02", "#ff7f0e"), ("taker", "kalshi_verified", "#1f77b4")]
    y = np.arange(len(names))
    for k, (fill, fee, c) in enumerate(combos):
        vals = [summary[n]["kalshi"][f"{fill}|{fee}"] for n in names]
        x = np.array([v.get("cents_per_contract", np.nan) for v in vals])
        lo = np.array([v.get("cents_ci90", [np.nan, np.nan])[0] for v in vals]); hi = np.array([v.get("cents_ci90", [np.nan, np.nan])[1] for v in vals])
        ax.errorbar(x, y + (k - 1) * 0.25, xerr=[x - lo, hi - x], fmt="o", capsize=3, color=c, label=f"Kalshi, fill {fill}, fee {fee}")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{n} (n={summary[n]['kalshi']['taker|kalshi_verified'].get('bets', 0)})" for n in names], fontsize=8)
    ax.set_xlabel("cents per $1 contract, first bet per game, held to settlement (90% bootstrap CI over games)")
    ax.set_title("In-game rule simulation, Kalshi 2025 + 2026 wk1-9 (held-out model). Simulation only; draft", fontsize=11)
    ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path_bar, dpi=130, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for ax, stake in zip(axes, ("flat", "qkelly")):
        dates, equity, series = [], [], []
        for n in names:
            c = curves.get(f"{n}|taker|kalshi_verified|{stake}") or []
            dates.extend(r["date"] for r in c)
            equity.extend(r["equity"] for r in c)
            series.extend([n] * len(c))
        frame = FrameBankroll(dates, equity, series, mode="auto")
        title = f"Continuous $10k, {'flat $200/bet' if stake == 'flat' else 'quarter Kelly (cap 5%)'}; taker fill + Kalshi verified fee"
        PlotBankroll(frame, ax=ax, title=title, ylabel="bank ($)",
                     hlines=[(START, "", {"color": "black", "lw": 0.8, "ls": "-", "alpha": 0.8})],
                     legend={"fontsize": 6.5, "ncol": 2} if stake == "flat" else False)
    fig.tight_layout()
    fig.savefig(path_eq, dpi=130, bbox_inches="tight"); plt.close(fig)

def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "ingame_sim"))
    ap.add_argument("--base", default="ingame_rule_sim")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    games, plays = Games(), ModelPlays()
    win = {}
    for s in (2024, 2025, 2026):
        win.update(Windows(s, {g for g in games if int(games[g]["season"]) == s}))
    samp = {"kalshi": Samples("Kalshi", games, plays, win), "polymarket": Samples("Polymarket", games, plays, win)}
    rules = Rules()
    summary, curves, paths, tidy_curves, all_bets = {}, {}, {}, [], []
    for name, rule in rules.items():
        summary[name] = {"rule": rule}
        for venue, sm in samp.items():
            bets = Bets(rule, sm, games)
            res = {"gross_mid": Stats(bets, "mid", None, a.boot, rng)}
            fills = ("mid", "taker") if venue == "kalshi" else ("mid",)
            for fill in fills:
                for fee in FEES:
                    res[f"{fill}|{fee}"] = Stats(bets, fill, fee, a.boot, rng)
            if venue == "kalshi":
                res["by_season_taker_kalshi_verified"] = {str(s): Stats([b for b in bets if b["season"] == s], "taker", "kalshi_verified", a.boot, rng) for s in (2025, 2026)}
                spreads = [b["ask_side"] - b["mid_side"] for b in bets]
                res["mean_half_spread_cents"] = round(100 * float(np.mean(spreads)), 2) if spreads else None
                for stake in ("flat", "qkelly"):
                    for fill, fee in (("taker", "kalshi_verified"), ("mid", "tim_est_0.02")):
                        c = Bankroll(bets, fill, fee, stake)
                        key = f"{name}|{fill}|{fee}|{stake}"
                        curves[key] = c
                        tidy_curves += [{"date": r["date"], "series": key, "equity": r["equity"], "drawdown_pct": r["drawdown_pct"]} for r in c]
                        if fill == "taker" and c:
                            rets = [r["ret"] for r in c]
                            n25 = sum(1 for r in c if r["date"] < "2026-03-01") or len(c)
                            ev = Evaluate(rets, seed=a.seed, paths=a.paths, horizons={"one_season": n25, "window": len(rets), "10_seasons": 10 * n25})
                            paths[key] = {"real_final": c[-1]["equity"], "max_dd_pct": min(r["drawdown_pct"] for r in c),
                                          "one_season_random_start_restart": ev["randomized"].get("one_season|random_start|restart"),
                                          "10_seasons_block_absorbing": ev["randomized"].get("10_seasons|block_bootstrap|absorbing")}
                all_bets += [{"rule": name, **b} for b in bets]
            summary[name][venue] = res
        prim = summary[name]["kalshi"]["taker|kalshi_verified"]
        tim = summary[name]["kalshi"]["taker|tim_est_0.02"]
        ok = prim.get("bets", 0) >= 30 and prim["cents_ci90"][0] > 0 and tim["cents_ci90"][0] > 0
        summary[name]["verdict"] = "HYPOTHESIS ONLY (CI > 0 after taker fill + both fee bases; multiple testing, not validated)" if ok else "PASS (no supportable edge)"
        print(name, summary[name]["verdict"], {k: summary[name]["kalshi"][k].get("cents_per_contract") for k in ("gross_mid", "mid|tim_est_0.02", "taker|kalshi_verified")},
              "n", prim.get("bets"), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "rules": rules, "summary": summary, "path_sim": paths,
              "n_games": {v: len(s) for v, s in samp.items()}, "settings_counted": len(rules) + 2,
              "note": __doc__}
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    flat = []
    for name, s in summary.items():
        for venue in ("kalshi", "polymarket"):
            for k, v in s[venue].items():
                if isinstance(v, dict) and "bets" in v and "|" in k or k == "gross_mid":
                    flat.append({"rule": name, "venue": venue, "fill_fee": k, "bets": v.get("bets"), "hit_rate": v.get("hit_rate"),
                                 "avg_price": v.get("avg_price"), "cents_per_contract": v.get("cents_per_contract"),
                                 "cents_ci90_lo": (v.get("cents_ci90") or [None, None])[0], "cents_ci90_hi": (v.get("cents_ci90") or [None, None])[1],
                                 "roi": v.get("roi"), "verdict": s["verdict"] if venue == "kalshi" else "secondary check (not executable)"})
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys())); w.writeheader(); w.writerows(flat)
    with open(os.path.join(a.out_dir, f"{a.base}_bankroll.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "series", "equity", "drawdown_pct"]); w.writeheader(); w.writerows(tidy_curves)
    with open(os.path.join(a.out_dir, f"{a.base}_bets.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_bets[0].keys())); w.writeheader(); w.writerows(all_bets)
    Charts(summary, curves, os.path.join(a.out_dir, f"{a.base}_rules_draft.png"), os.path.join(a.out_dir, f"{a.base}_bankroll_draft.png"))


if __name__ == "__main__":
    Main()