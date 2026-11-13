r"""
Pregame calibration: our walk-forward pregame model vs the pregame market, one probability per game (home side).

    .venv\Scripts\python.exe -m Backtest.PregameModelCalibration [--out-dir DIR --base NAME --boot 2000 --seed 20261113]

Sources (all home win probability):
  model_p_full2      walk-forward p_full2 (Lab.AttachModels, fitted on seasons 2013..S-1). Its market input is the nflverse
                     de-vigged CLOSING moneyline, so it is not independent of the close and uses information from after T-90.
  model_p_full2_polyT90  same weights, market input swapped for the Polymarket mid at T-90 (what we could compute at T-90)
  close_devig        nflverse closing moneyline, de-vigged (mkt_home)
  kalshi_T90         Kalshi home-market mid ((bid+ask)/2, 1-min candles) at kickoff-90 min, staleness <= 30 min
  poly_T90           Polymarket prices-history mid at kickoff-90 min, staleness <= 30 min
Sets: "long" 2014..2026 wk9 (model vs close), "venues_common" 2025..2026 wk9 games priced by all sources, and
"poly_2024_2026" 2024..2026 wk9 games priced by Polymarket (Kalshi excluded). Ties dropped. One sample per game, so the
naive binomial and game-cluster intervals coincide up to bootstrap noise. Not a strategy search: nothing is tuned.
"""

import argparse
import csv
import json
import math
import os
from datetime import timedelta

import numpy as np

from Backtest.InGameCalibration import Calibrate, Sample, SeriesKalshi
from Backtest.KalshiReality import RefitWeights
from Backtest.Lab import AllRows, AttachModels, Sig, Vec, WIN_FEATS2
from Backtest.PolyReality import LoadHistory, PriceAt
from Backtest.SnapshotPrices import Kickoff

SOURCES = ["model_p_full2", "model_p_full2_polyT90", "close_devig", "kalshi_T90", "poly_T90"]


def KalshiT90(gid_rows, g):
    r = gid_rows.get(g["game_id"])
    if not r:
        return None
    t = (Kickoff(g) - timedelta(minutes=90)).timestamp()
    ser = [(ts, p) for ts, p in SeriesKalshi(r) if ts <= t and t - ts <= 1800]
    return max(ser)[1] if ser else None


def Scores(rows, key, boot, rng):
    p = np.clip(np.array([r[key] for r in rows]), 1e-6, 1 - 1e-6)
    y = np.array([r["y"] for r in rows], dtype=float)
    br, ll = (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))
    idx = rng.integers(0, len(rows), size=(boot, len(rows)))
    q = lambda a: [round(float(np.quantile(a[idx].mean(1), 0.05)), 4), round(float(np.quantile(a[idx].mean(1), 0.95)), 4)]
    return {"n": len(rows), "brier": round(float(br.mean()), 4), "brier_ci90": q(br), "log_loss": round(float(ll.mean()), 4),
            "log_loss_ci90": q(ll), "_br": br, "_ll": ll, "_idx": idx}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "calibration"))
    ap.add_argument("--base", default="pregame_model_calibration")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    kal = {}
    for s in (2025, 2026):
        p = os.path.join("data", "prices", f"ingame_kalshi_{s}.jsonl")
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            kal[r["game_id"]] = r
    rows = []
    for s in range(2014, 2027):
        poly = LoadHistory(s) if s >= 2024 else {}
        w2 = RefitWeights(rbs, s)[0] if s >= 2024 else None
        for r in rbs[s]:
            if r["home_win"] is None or (s == 2026 and int(r["week"]) > 9) or r.get("mkt_home") is None:
                continue
            row = {"game_id": r["game_id"], "season": s, "week": r["week"], "y": int(r["home_win"]),
                   "model_p_full2": r["p_full2"], "close_devig": r["mkt_home"], "model_p_full2_polyT90": None, "kalshi_T90": None, "poly_T90": None}
            if s >= 2024:
                q = PriceAt(poly.get(r["game_id"]), 90)
                if q:
                    pk = min(max(q[0], 1e-4), 1 - 1e-4)
                    row["poly_T90"] = q[0]
                    saved = r["mkt_logit"]
                    r["mkt_logit"] = math.log(pk / (1 - pk))
                    row["model_p_full2_polyT90"] = float(Sig(np.dot(w2, Vec(r, WIN_FEATS2))))
                    r["mkt_logit"] = saved
                row["kalshi_T90"] = KalshiT90(kal, r) if s >= 2025 else None
            rows.append(row)
    sets = {"long": (range(2014, 2027), ["model_p_full2", "close_devig"]),
            "poly_2024_2026": (range(2024, 2027), ["model_p_full2", "model_p_full2_polyT90", "close_devig", "poly_T90"]),
            "venues_common": (range(2025, 2027), SOURCES)}
    out, tidy, scores_flat = {}, [], []
    for sname, (yrs, srcs) in sets.items():
        rs = [r for r in rows if r["season"] in yrs and all(r[k] is not None for k in srcs)]
        res = {}
        base = None
        for k in srcs:
            cal = Calibrate([(r["game_id"], r[k], r["y"]) for r in rs], a.boot, rng)
            sc = Scores(rs, k, a.boot, rng) if base is None else None
            if base is None:
                base = sc
            # paired differences vs close_devig on the same bootstrap draws
            p = np.clip(np.array([r[k] for r in rs]), 1e-6, 1 - 1e-6)
            y = np.array([r["y"] for r in rs], dtype=float)
            br, ll = (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))
            pc = np.clip(np.array([r["close_devig"] for r in rs]), 1e-6, 1 - 1e-6)
            brc, llc = (pc - y) ** 2, -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
            idx = base["_idx"]
            qq = lambda arr: [round(float(np.quantile(arr[idx].mean(1), 0.05)), 4), round(float(np.quantile(arr[idx].mean(1), 0.95)), 4)]
            res[k] = {"n_games": len(rs), "brier": round(float(br.mean()), 4), "brier_ci90": qq(br), "log_loss": round(float(ll.mean()), 4),
                      "log_loss_ci90": qq(ll), "brier_minus_close": round(float((br - brc).mean()), 4), "brier_minus_close_ci90": qq(br - brc),
                      "log_loss_minus_close": round(float((ll - llc).mean()), 4), "log_loss_minus_close_ci90": qq(ll - llc),
                      "ece": cal["ece"], "bins": cal["bins"]}
            scores_flat.append({"set": sname, "source": k, **{m: res[k][m] for m in ("n_games", "brier", "log_loss", "brier_minus_close", "log_loss_minus_close", "ece")},
                                "brier_ci90": res[k]["brier_ci90"], "log_loss_ci90": res[k]["log_loss_ci90"],
                                "brier_minus_close_ci90": res[k]["brier_minus_close_ci90"], "log_loss_minus_close_ci90": res[k]["log_loss_minus_close_ci90"]})
            for b in cal["bins"]:
                tidy.append({"set": sname, "source": k, **b})
        out[sname] = {"seasons": [min(yrs), max(yrs)], "n_games": len(rs), "sources": res}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": __doc__, "results": out, "scores": scores_flat}, fh, indent=1, default=str)
    fields = ["set", "source", "bin", "n_samples", "n_games", "mean_pred", "obs_freq", "naive_ci90_lo", "naive_ci90_hi", "cluster_ci90_lo",
              "cluster_ci90_hi", "gw_mean_pred", "gw_obs_freq", "gw_cluster_ci90_lo", "gw_cluster_ci90_hi"]
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(tidy)
    Chart(out, os.path.join(a.out_dir, f"{a.base}_draft.png"))
    print(json.dumps(scores_flat, indent=1))


def Chart(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"model_p_full2": "#2ca02c", "model_p_full2_polyT90": "#9467bd", "close_devig": "black", "kalshi_T90": "#1f77b4", "poly_T90": "#d62728"}
    names = {"model_p_full2": "our model p_full2 (close input)", "model_p_full2_polyT90": "our model, Polymarket T-90 input",
             "close_devig": "closing line, de-vigged", "kalshi_T90": "Kalshi mid T-90", "poly_T90": "Polymarket mid T-90"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, sname in zip(axes, ("long", "venues_common")):
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        srcs = out[sname]["sources"]
        for j, (k, r) in enumerate(srcs.items()):
            bs = [b for b in r["bins"] if b.get("n_samples", 0) >= 5]
            off = (j - (len(srcs) - 1) / 2) * 0.006
            x = np.array([b["mean_pred"] for b in bs]) + off
            y = np.array([b["obs_freq"] for b in bs])
            lo = np.array([b["cluster_ci90_lo"] for b in bs]); hi = np.array([b["cluster_ci90_hi"] for b in bs])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", ms=4, capsize=3, color=colors[k],
                        label=f"{names[k]}: Brier {r['brier']:.4f}, log loss {r['log_loss']:.4f}")
        s0, s1 = out[sname]["seasons"]
        ax.set_title(f"{sname}: {s0}-{s1} (2026 wk1-9), {out[sname]['n_games']} games\n90% game bootstrap CIs; bins with <5 games hidden", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("pregame home win probability (bin mean)"); ax.set_ylabel("observed home win frequency")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Pregame calibration: our walk-forward model vs the market (no in-game model exists)", fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    Main()