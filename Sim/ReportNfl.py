r"""Score the NFL resimulator output (Sim.ValidateNfl) on tune (2024) and holdout (2025-2026 wk 9).

Pre-declared pass rules for 'usable as a variance / stress-test engine' (holdout):
  V1 margin and total interval coverage at 50/80/90% each within +-5 pts of nominal;
  V2 win-probability ECE (10 bins) <= 0.05;
  V3 key-number frequencies (|margin| = 3, 7) and the tie rate within +-3 pts of reality;
  V4 upset rate (sim favourite loses) within +-4 pts of the sim's own expectation.
'Usable for pricing' additionally needs Brier <= the closing line on the same games (not expected).
Near-miss ('slightly different') win probabilities are usable for variance reduction only if on the holdout
mean(y - p_nm) has a CI covering 0 and the slope of (y - p_nm) on (p_close - 0.5) has a CI covering 0.

Usage: .venv\Scripts\python.exe -m Sim.ReportNfl --games <csv> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

NM_COLS = ["p_nm_model_0.25", "p_nm_model_0.5", "p_nm_boot_0.25", "p_nm_boot_0.5", "p_nm_boot_1"]


def Ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def BootCi(x, fn=np.mean, n=2000, seed=7, groups=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if groups is None:
        idx = rng.integers(0, len(x), (n, len(x)))
        vals = np.array([fn(x[i]) for i in idx])
    else:
        g = np.asarray(groups)
        ug = np.unique(g)
        members = [np.flatnonzero(g == u) for u in ug]
        vals = []
        for _ in range(n):
            pick = rng.integers(0, len(ug), len(ug))
            ii = np.concatenate([members[j] for j in pick])
            vals.append(fn(x[ii]))
        vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def Slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def SlopeCi(x, y, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        vals.append(Slope(x[i], y[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def Block(d: pl.DataFrame):
    y = d["y"].to_numpy()
    p = d["p_home"].to_numpy()
    real_m = (d["home_final"] - d["away_final"]).to_numpy()
    out = {"n": d.height, "brier_sim": float(np.mean((p - y) ** 2)), "ece_sim": Ece(p, y)}
    for name, col in (("margin", "pit_margin"), ("total", "pit_total")):
        pit = d[col].to_numpy()
        for c in (0.5, 0.8, 0.9):
            out[f"cover_{name}_{int(c*100)}"] = float((np.abs(pit - 0.5) <= c / 2).mean())
        out[f"pit_mean_{name}"] = float(pit.mean())
    out["resid_sd_margin"] = float(np.std(real_m - d["margin_mean"].to_numpy()))
    out["sim_sd_margin"] = float(np.sqrt(np.mean(d["margin_sd"].to_numpy() ** 2)))
    real_t = (d["home_final"] + d["away_final"]).to_numpy()
    out["resid_sd_total"] = float(np.std(real_t - d["total_mean"].to_numpy()))
    out["sim_sd_total"] = float(np.sqrt(np.mean(d["total_sd"].to_numpy() ** 2)))
    out["real_total_mean"] = float(real_t.mean())
    out["sim_total_mean"] = float(d["total_mean"].mean())
    out["real_key3"] = float((np.abs(real_m) == 3).mean())
    out["sim_key3"] = float(d["p_key3"].mean())
    out["real_key7"] = float((np.abs(real_m) == 7).mean())
    out["sim_key7"] = float(d["p_key7"].mean())
    out["real_tie"] = float((real_m == 0).mean())
    out["sim_tie"] = float(d["p_tie"].mean())
    fav_home = p >= 0.5
    upset = np.where(fav_home, y < 0.5, y > 0.5)
    out["real_upset"] = float(upset.mean())
    out["sim_expected_upset"] = float(np.mean(np.minimum(p, 1 - p)))
    ok = d.filter(pl.col("p_book").is_not_null())
    if ok.height:
        yy, pb, ps = ok["y"].to_numpy(), ok["p_book"].to_numpy(), ok["p_home"].to_numpy()
        out["n_book"] = ok.height
        out["brier_book"] = float(np.mean((pb - yy) ** 2))
        out["brier_sim_same"] = float(np.mean((ps - yy) ** 2))
        lo, hi = BootCi(np.c_[ps, pb, yy], fn=lambda a: np.mean((a[:, 0] - a[:, 2]) ** 2 - (a[:, 1] - a[:, 2]) ** 2)) if False else (None, None)
        diff = (ps - yy) ** 2 - (pb - yy) ** 2
        out["brier_sim_minus_book"] = float(diff.mean())
        out["brier_sim_minus_book_ci"] = BootCi(diff)
        out["corr_sim_book"] = float(np.corrcoef(ps, pb)[0, 1])
    ok = d.filter(pl.col("p_kalshi").is_not_null())
    if ok.height:
        yy, pk, ps = ok["y"].to_numpy(), ok["p_kalshi"].to_numpy(), ok["p_home"].to_numpy()
        diff = (ps - yy) ** 2 - (pk - yy) ** 2
        out["n_kalshi"] = ok.height
        out["brier_kalshi"] = float(np.mean((pk - yy) ** 2))
        out["brier_sim_minus_kalshi"] = float(diff.mean())
        out["brier_sim_minus_kalshi_ci"] = BootCi(diff)
    # near-miss versions
    nm = {}
    okb = d.filter(pl.col("p_book").is_not_null())
    yy, pb = okb["y"].to_numpy(), okb["p_book"].to_numpy()
    base_var = float(np.var(yy - pb))
    for c in NM_COLS:
        if c not in d.columns:
            continue
        q = okb[c].to_numpy()
        r = yy - q
        nm[c] = {"mean_y_minus_p": float(r.mean()), "ci": BootCi(r), "ece": Ece(q, yy),
                 "slope_on_close": Slope(pb - 0.5, r), "slope_ci": SlopeCi(pb - 0.5, r),
                 "var_y_minus_close": base_var, "var_p_minus_close": float(np.var(q - pb)),
                 "n_multiplier": float(base_var / max(np.var(q - pb), 1e-9)),
                 "flip_rate": float(np.mean(np.abs(q - yy) > 0.5))}
        if "p_full2_kmid" in okb.columns:
            mk = okb.filter(pl.col("p_full2_kmid").is_not_null())
            if mk.height > 30:
                rr = mk["y"].to_numpy() - mk[c].to_numpy()
                nm[c]["slope_on_model_minus_close"] = Slope(mk["p_home"].to_numpy() - mk["p_book"].to_numpy(), rr)
                nm[c]["slope_on_model_minus_close_ci"] = SlopeCi(mk["p_home"].to_numpy() - mk["p_book"].to_numpy(), rr)
    out["near_miss"] = nm
    # stress: tilt sensitivity
    tilts = [c for c in d.columns if c.startswith("p_tilt_")]
    out["tilt"] = {c: float((d[c] - d["p_home"]).mean()) for c in tilts}
    out["tilt_flip_fav"] = {c: float(((d[c] >= 0.5) != (d["p_home"] >= 0.5)).mean()) for c in tilts}
    return out


def Verdict(h):
    v1 = all(abs(h[f"cover_{n}_{c}"] - c / 100) <= 0.05 for n in ("margin", "total") for c in (50, 80, 90))
    v2 = h["ece_sim"] <= 0.05
    v3 = abs(h["real_key3"] - h["sim_key3"]) <= 0.03 and abs(h["real_key7"] - h["sim_key7"]) <= 0.03 and abs(h["real_tie"] - h["sim_tie"]) <= 0.03
    v4 = abs(h["real_upset"] - h["sim_expected_upset"]) <= 0.04
    pricing = h.get("brier_sim_minus_book", 1) <= 0
    return {"V1_coverage": v1, "V2_calibration": v2, "V3_key_numbers": v3, "V4_upsets": v4,
            "usable_variance_engine": bool(v1 and v2 and v4), "usable_shape_incl_key_numbers": bool(v1 and v2 and v3 and v4),
            "usable_for_pricing": bool(pricing)}


def AnchoredBlock(d: pl.DataFrame):
    """Shape checks for the market-anchored resimulator (win probability is the closing line by construction)."""
    real_m = (d["home_final"] - d["away_final"]).to_numpy()
    out = {"n": d.height, "anchor_err_mean_abs": float(np.mean(np.abs(d["p_home"].to_numpy() - d["p_book"].to_numpy())))}
    for name, col in (("margin", "pit_margin"), ("total", "pit_total")):
        pit = d[col].to_numpy()
        for c in (0.5, 0.8, 0.9):
            out[f"cover_{name}_{int(c*100)}"] = float((np.abs(pit - 0.5) <= c / 2).mean())
    out.update({"real_key3": float((np.abs(real_m) == 3).mean()), "sim_key3": float(d["p_key3"].mean()),
                "real_key7": float((np.abs(real_m) == 7).mean()), "sim_key7": float(d["p_key7"].mean()),
                "real_tie": float((real_m == 0).mean()), "sim_tie": float(d["p_tie"].mean()),
                "real_total_mean": float((d["home_final"] + d["away_final"]).mean()), "sim_total_mean": float(d["total_mean"].mean()),
                "total_line_mean": float(d["total_line"].mean())})
    for what, p_col, y_col in (("spread", "sim_p_cover", "real_cover"), ("total", "sim_p_over", "real_over")):
        k = d.filter(pl.col(y_col).is_not_null() & pl.col(p_col).is_not_null())
        p, y = k[p_col].to_numpy(), k[y_col].to_numpy()
        out[f"{what}_n"] = k.height
        out[f"{what}_sim_p_mean"] = float(p.mean())
        out[f"{what}_real_rate"] = float(y.mean())
        out[f"{what}_brier_sim"] = float(np.mean((p - y) ** 2))
        out[f"{what}_brier_half"] = float(np.mean((0.5 - y) ** 2))
        out[f"{what}_ece"] = Ece(p, y, bins=5)
    nm = {}
    for c in ("p_nm_anch_0.25", "p_nm_anch_0.5"):
        r = d["y"].to_numpy() - d[c].to_numpy()
        x = d["p_book"].to_numpy() - 0.5
        nm[c] = {"mean_y_minus_p": float(r.mean()), "ci": BootCi(r), "slope_on_close": Slope(x, r), "slope_ci": SlopeCi(x, r),
                 "n_multiplier": float(np.var(d["y"].to_numpy() - d["p_book"].to_numpy()) / max(np.var(d[c].to_numpy() - d["p_book"].to_numpy()), 1e-9))}
    out["near_miss"] = nm
    return out


def AnchoredVerdict(h):
    v1 = all(abs(h[f"cover_{n}_{c}"] - c / 100) <= 0.05 for n in ("margin", "total") for c in (50, 80, 90))
    v3 = abs(h["real_key3"] - h["sim_key3"]) <= 0.03 and abs(h["real_key7"] - h["sim_key7"]) <= 0.03 and abs(h["real_tie"] - h["sim_tie"]) <= 0.03
    v5 = h["spread_ece"] <= 0.05 and h["total_ece"] <= 0.05
    nm_ok = {c: bool(v["ci"][0] <= 0 <= v["ci"][1] and v["slope_ci"][0] <= 0 <= v["slope_ci"][1]) for c, v in h["near_miss"].items()}
    return {"V1_coverage": v1, "V3_key_numbers": v3, "V5_spread_total_calibration": v5, "usable_shape_engine": bool(v1 and v5),
            "near_miss_unbiased": nm_ok}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchored", default=None)
    a = ap.parse_args()
    if a.anchored:
        d = pl.read_csv(a.anchored, infer_schema_length=10000)
        res = {"anchored_tune_2024": AnchoredBlock(d.filter(pl.col("split") == "tune")),
               "anchored_holdout_2025_26": AnchoredBlock(d.filter(pl.col("split") == "holdout"))}
        res["anchored_verdict_tune"] = AnchoredVerdict(res["anchored_tune_2024"])
        res["anchored_verdict_holdout"] = AnchoredVerdict(res["anchored_holdout_2025_26"])
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(json.dumps(res, indent=1))
        return
    d = pl.read_csv(a.games, infer_schema_length=10000)
    res = {"tune_2024": Block(d.filter(pl.col("split") == "tune")), "holdout_2025_26": Block(d.filter(pl.col("split") == "holdout")),
           "holdout_2025": Block(d.filter(pl.col("season") == 2025)), "holdout_2026": Block(d.filter(pl.col("season") == 2026))}
    res["verdict_holdout"] = Verdict(res["holdout_2025_26"])
    res["verdict_tune"] = Verdict(res["tune_2024"])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    Main()
