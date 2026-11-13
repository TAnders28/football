r"""
Held-out evaluation of the in-game WP model (Backtest.InGameWP) against Kalshi and Polymarket on the same 1-minute
in-play samples used by Backtest.InGameCalibration. Research only.

    .venv\Scripts\python.exe -m Backtest.InGameModelEval [--out-dir DIR --base NAME --boot 1000 --seed 20261113]

Each market sample (1-minute UTC grid inside the pbp in-play window, halftime / >10-min gaps excluded, latest mid no older
than 3 min, home side) is paired with the model probability for the pre-play state of the LATEST play snapped at or before
that minute (no older than 10 min). That state can lag the market by up to one play (conservative for the model).
Model predictions for season S come from the walk-forward model trained on 2014..S-1 (held out).
Mismatch: when |model - market| > X, who was right more often, and would a taker at the mid (Tim's fill assumption)
have made money after fees? Fees: tim_est_0.02 (Tim's estimate, per contract); Kalshi verified taker 0.07*p*(1-p);
Polymarket sports 0.05*p*(1-p) (uncertain for NFL). One bet per game (the first trigger) is the cleaner unit; the
per-sample numbers are heavily correlated within games. CIs are game-level cluster bootstraps.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import polars as pl

from Backtest.InGameCalibration import Calibrate, InPlayGrid, Sample, SeriesKalshi, SeriesPoly, Ts, Windows
from Backtest.Lab import LoadSeason

STATE_STALE_S = 600
THRESHOLDS = (0.03, 0.05, 0.10)
FEES = {"tim_est_0.02": lambda p: 0.02, "kalshi_verified": lambda p: 0.07 * p * (1 - p), "poly_sports_0.05": lambda p: 0.05 * p * (1 - p)}
VENUE_FEE = {"Kalshi": "kalshi_verified", "Polymarket": "poly_sports_0.05"}
MODELS = {"model": "p_home_model", "model_mktprior": "p_home_market", "model_noprior": "p_home_none", "nflverse_wp_ref": "home_wp"}


def ClusterMean(gids, vals, boot, rng):
    """mean of vals and a 90% game-cluster bootstrap CI."""
    if len(vals) == 0:
        return None, [None, None]
    u, inv = np.unique(gids, return_inverse=True)
    s = np.bincount(inv, weights=vals); c = np.bincount(inv).astype(float)
    W = rng.multinomial(len(u), [1 / len(u)] * len(u), size=boot).astype(float)
    bs = (W @ s) / (W @ c)
    return round(float(s.sum() / c.sum()), 5), [round(float(np.quantile(bs, 0.05)), 5), round(float(np.quantile(bs, 0.95)), 5)]


def LL(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def Collect():
    games = {}
    for s in (2024, 2025, 2026):
        for g in LoadSeason(s):
            if g.get("home_score") is None or (s == 2026 and int(g["week"]) > 9) or g["home_score"] == g["away_score"]:
                continue
            games[g["game_id"]] = g
    plays = {}
    for S in (2024, 2025, 2026):
        df = pl.read_ndjson(os.path.join("data", "backtest", "ingame_model", f"heldout_plays_{S}.jsonl"), infer_schema_length=None)
        for gid, sub in df.group_by("game_id"):
            gid = gid[0] if isinstance(gid, tuple) else gid
            ts = np.array([Ts(x) or np.nan for x in sub["time_of_day"].to_list()], dtype=float)
            ok = ~np.isnan(ts)
            order = np.argsort(ts[ok], kind="stable")
            plays[gid] = {"t": ts[ok][order], **{k: sub[c].to_numpy()[ok][order] for k, c in MODELS.items()}}
    venues = {"Kalshi": ("ingame_kalshi", SeriesKalshi, (2025, 2026)), "Polymarket": ("ingame_poly", SeriesPoly, (2024, 2025, 2026))}
    rows = {v: {} for v in venues}
    for v, (stem, fn, seasons) in venues.items():
        for s in seasons:
            for l in open(os.path.join("data", "prices", f"{stem}_{s}.jsonl"), encoding="utf-8"):
                r = json.loads(l)
                if r["game_id"] in games:
                    rows[v][r["game_id"]] = (r, fn)
    win = {}
    for s in (2024, 2025, 2026):
        win.update(Windows(s, {g for d in rows.values() for g in d if int(games[g]["season"]) == s}))
    out = []
    for v, d in rows.items():
        for gid, (r, fn) in d.items():
            w, pls, g = win.get(gid), plays.get(gid), games[gid]
            if not w or pls is None or not len(pls["t"]):
                continue
            grid = InPlayGrid(w)
            qmap = dict(grid)
            y = 1 if g["home_score"] > g["away_score"] else 0
            for t, p in Sample(fn(r), [t for t, _ in grid]):
                i = np.searchsorted(pls["t"], t, side="right") - 1
                if i < 0 or t - pls["t"][i] > STATE_STALE_S:
                    continue
                rec = {"venue": v, "game_id": gid, "season": int(g["season"]), "t": t, "qtr": qmap[t], "y": y, "market": p}
                for k in MODELS:
                    val = pls[k][i]
                    rec[k] = None if val is None or (isinstance(val, float) and math.isnan(val)) else float(val)
                out.append(rec)
    return out


def Metrics(sub, key, boot, rng):
    gids = np.array([s["game_id"] for s in sub]); y = np.array([s["y"] for s in sub], dtype=float)
    p = np.array([s[key] for s in sub], dtype=float); pm = np.array([s["market"] for s in sub], dtype=float)
    br, ll = (p - y) ** 2, LL(p, y)
    b, bci = ClusterMean(gids, br, boot, rng)
    l, lci = ClusterMean(gids, ll, boot, rng)
    db, dbci = ClusterMean(gids, br - (pm - y) ** 2, boot, rng)
    dl, dlci = ClusterMean(gids, ll - LL(pm, y), boot, rng)
    cal = Calibrate([(g, pp, yy) for g, pp, yy in zip(gids, p, y)], 0, rng)
    return {"n_samples": len(sub), "n_games": len(set(gids)), "brier": b, "brier_ci90": bci, "log_loss": l, "log_loss_ci90": lci,
            "ece": cal["ece"], "brier_minus_market": db, "brier_minus_market_ci90": dbci, "log_loss_minus_market": dl,
            "log_loss_minus_market_ci90": dlci}


def Mismatch(sub, key, X, boot, rng, venue):
    m = [s for s in sub if abs(s[key] - s["market"]) > X]
    if not m:
        return {"threshold": X, "n_samples": 0}
    gids = np.array([s["game_id"] for s in m]); y = np.array([s["y"] for s in m], dtype=float)
    p = np.array([s[key] for s in m]); pm = np.array([s["market"] for s in m])
    model_home = p > pm
    hit = np.where(model_home, y, 1 - y)                      # outcome went the model's way
    mkt_implied = np.where(model_home, pm, 1 - pm)            # market's probability for that same side
    model_implied = np.where(model_home, p, 1 - p)
    res = {"threshold": X, "n_samples": len(m), "n_games": len(set(gids)),
           "model_side_won": ClusterMean(gids, hit, boot, rng), "market_implied": round(float(mkt_implied.mean()), 4),
           "model_implied": round(float(model_implied.mean()), 4),
           "brier_model": round(float(np.mean((p - y) ** 2)), 4), "brier_market": round(float(np.mean((pm - y) ** 2)), 4),
           "brier_model_minus_market": ClusterMean(gids, (p - y) ** 2 - (pm - y) ** 2, boot, rng)}
    for fee in ("tim_est_0.02", VENUE_FEE[venue]):
        f = FEES[fee]
        pnl, gg, first = [], [], {}
        for s in sorted(m, key=lambda s: (s["game_id"], s["t"])):
            ph, pmk = s[key], s["market"]
            if ph - pmk > f(pmk):
                v = s["y"] - pmk - f(pmk)
            elif (1 - ph) - (1 - pmk) > f(1 - pmk):
                v = (1 - s["y"]) - (1 - pmk) - f(1 - pmk)
            else:
                continue
            pnl.append(v); gg.append(s["game_id"])
            first.setdefault(s["game_id"], v)
        fv = np.array(list(first.values()))
        res[f"pnl_per_contract_{fee}"] = {"per_sample": ClusterMean(np.array(gg), np.array(pnl), boot, rng) if pnl else None,
                                          "n_samples": len(pnl),
                                          "first_trigger_per_game": {"n_games": len(fv), "mean": round(float(fv.mean()), 4) if len(fv) else None,
                                                                     "ci90": [round(float(np.quantile(rng.choice(fv, (boot, len(fv))).mean(1), q)), 4) for q in (0.05, 0.95)] if len(fv) > 1 else None}}
    return res


def Chart(samples, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.18)
    style = {"market": ("venue mid", None), "model": ("our in-game model (held out)", "#2ca02c"), "nflverse_wp_ref": ("nflverse wp (reference only)", "grey")}
    vc = {"Kalshi": "#1f77b4", "Polymarket": "#d62728"}
    for col, v in enumerate(("Kalshi", "Polymarket")):
        sub = [s for s in samples if s["venue"] == v and s["nflverse_wp_ref"] is not None]
        ax = fig.add_subplot(gs[0, col])
        ax.plot([0, 1], [0, 1], color="lightgrey", ls="--", lw=1)
        for j, (k, (lab, c)) in enumerate(style.items()):
            cal = res["calibration"][f"{v}|{k}"]
            bs = [b for b in cal["bins"] if b.get("n_samples")]
            x = np.array([b["mean_pred"] for b in bs]) + (j - 1) * 0.007
            y = np.array([b["obs_freq"] for b in bs]); lo = np.array([b["cluster_ci90_lo"] for b in bs]); hi = np.array([b["cluster_ci90_hi"] for b in bs])
            mm = res["metrics"][f"{v}|all|{k}"]
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-" if k != "nflverse_wp_ref" else "s:", ms=4, capsize=3, color=c or vc[v],
                        label=f"{v if k == 'market' else lab}: Brier {mm['brier']:.4f}, LL {mm['log_loss']:.4f}, ECE {mm['ece']:.3f}")
        n_g = len({s["game_id"] for s in sub})
        ax.set_title(f"{v} samples: {len(sub):,} one-minute in-play samples, {n_g} games\n90% game-cluster bootstrap CIs", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlabel("predicted home win probability (bin mean)"); ax.set_ylabel("observed home win frequency")
        axh = fig.add_subplot(gs[1, col])
        bins = np.linspace(0, 1, 11)
        axh.hist([[s["market"] for s in sub], [s["model"] for s in sub]], bins=bins, color=[vc[v], "#2ca02c"], label=[v, "our model"])
        axh.set_ylabel("samples"); axh.legend(fontsize=7); axh.grid(alpha=0.3, axis="y")
    fig.suptitle("Held-out in-game calibration: our model vs the market on the same samples (2024-2026 wk1-9; draft)", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "ingame_model"))
    ap.add_argument("--base", default="ingame_model_eval")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    samples = Collect()
    res = {"metrics": {}, "calibration": {}, "mismatch": {}, "n": {}}
    tidy = []
    for v in ("Kalshi", "Polymarket"):
        base = [s for s in samples if s["venue"] == v and all(s[k] is not None for k in MODELS)]
        res["n"][v] = {"samples": len(base), "games": len({s["game_id"] for s in base})}
        splits = {"all": base}
        for q in (1, 2, 3, 4, 5):
            splits["Q%d" % q if q < 5 else "OT"] = [s for s in base if s["qtr"] == q]
        for yr in (2024, 2025, 2026):
            splits[f"season_{yr}"] = [s for s in base if s["season"] == yr]
        for sp, sub in splits.items():
            if not sub:
                continue
            for k in ("market", *MODELS):
                if k == "market":
                    gids = np.array([s["game_id"] for s in sub]); y = np.array([s["y"] for s in sub], dtype=float); pm = np.array([s["market"] for s in sub])
                    b, bci = ClusterMean(gids, (pm - y) ** 2, a.boot, rng); l, lci = ClusterMean(gids, LL(pm, y), a.boot, rng)
                    res["metrics"][f"{v}|{sp}|market"] = {"n_samples": len(sub), "n_games": len(set(gids)), "brier": b, "brier_ci90": bci,
                                                          "log_loss": l, "log_loss_ci90": lci,
                                                          "ece": Calibrate([(g, p, yy) for g, p, yy in zip(gids, pm, y)], 0, rng)["ece"]}
                else:
                    res["metrics"][f"{v}|{sp}|{k}"] = Metrics(sub, k, a.boot, rng)
        for k in ("market", "model", "model_mktprior", "nflverse_wp_ref"):
            cal = Calibrate([(s["game_id"], s[k], s["y"]) for s in base], a.boot, rng)
            res["calibration"][f"{v}|{k}"] = cal
            for b in cal["bins"]:
                tidy.append({"venue_samples": v, "predictor": k, **b})
        for k in ("model", "model_mktprior"):
            for X in THRESHOLDS:
                res["mismatch"][f"{v}|{k}|{X}"] = Mismatch(base, k, X, a.boot, rng, v)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    fields = ["venue_samples", "predictor", "bin", "n_samples", "n_games", "mean_pred", "obs_freq", "naive_ci90_lo", "naive_ci90_hi",
              "cluster_ci90_lo", "cluster_ci90_hi", "gw_mean_pred", "gw_obs_freq", "gw_cluster_ci90_lo", "gw_cluster_ci90_hi"]
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(tidy)
    with open(os.path.join(a.out_dir, f"{a.base}_scores.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["key", "n_samples", "n_games", "brier", "brier_ci90", "log_loss", "log_loss_ci90", "ece", "brier_minus_market", "brier_minus_market_ci90", "log_loss_minus_market", "log_loss_minus_market_ci90"])
        for k, m in res["metrics"].items():
            w.writerow([k, m["n_samples"], m["n_games"], m["brier"], m["brier_ci90"], m["log_loss"], m["log_loss_ci90"], m["ece"],
                        m.get("brier_minus_market"), m.get("brier_minus_market_ci90"), m.get("log_loss_minus_market"), m.get("log_loss_minus_market_ci90")])
    Chart(samples, res, os.path.join(a.out_dir, f"{a.base}_draft.png"))
    print(json.dumps({"n": res["n"], "all": {k: {m: v[m] for m in ("brier", "log_loss", "ece")} for k, v in res["metrics"].items() if "|all|" in k}}, indent=1))


if __name__ == "__main__":
    Main()