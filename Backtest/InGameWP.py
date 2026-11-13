r"""
In-game win-probability model (research; NOT validated for betting until held-out calibration matches the markets).

    .venv\Scripts\python.exe -m Backtest.InGameWP [--select] [--out-dir data/backtest/ingame_model]

State BEFORE each play only (posteam perspective): score differential, time remaining (game / half), elapsed share,
quarter / OT, down, distance, field position (yardline_100), timeouts each side, posteam is home, posteam receives the
2nd-half kickoff, and a pregame prior logit (+ the prior decayed with time, like nflfastR's spread_time).
Prior variants: "model" = our walk-forward pregame p_full2 (Lab.AttachModels, fitted on seasons < S),
"market" = nflverse de-vigged closing moneyline, "none" = no prior.
Model: sklearn HistGradientBoostingClassifier with monotone constraints (score diff +, diff x time +, prior +,
yardline_100 -, own timeouts +, opponent timeouts -). Walk-forward: model for season S is trained on 2014..S-1 only.
Selection (--select): a few configs scored once on 2023 (trained 2014-2022); the chosen one is frozen in CONFIG.
nflverse wp / vegas_wp / home_wp are NEVER features; they are carried through only as a reference benchmark.
Outputs (gitignored, readable JSON): data/models/ingame_wp_<variant>_wf<S>.json, data/models/ingame_wp_<variant>_live.json
(trained 2014..2026 wk9; logistic coefficients + scaler), and held-out per-play predictions data/backtest/ingame_model/heldout_plays_<S>.jsonl.
"""

import argparse
import json
import math
import os

import numpy as np
import polars as pl

from Backtest.Lab import AllRows, AttachModels

FEATS = ["diff", "diff_time_ratio", "gsr", "hsr", "elapsed", "qtr", "ot", "down", "ydstogo", "yardline_100",
         "pos_to", "def_to", "is_home", "receive_2h", "prior_logit", "prior_time"]
MONO = {"diff": 1, "diff_time_ratio": 1, "yardline_100": -1, "pos_to": 1, "def_to": -1, "prior_logit": 1, "prior_time": 1}
CONFIGS = {
    "gbm_31": dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=400, l2_regularization=1.0),
    "gbm_15": dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=15, min_samples_leaf=800, l2_regularization=1.0),
    "logit": None,
}
CONFIG = "logit"   # frozen 2026-11-13 after --select on 2023 (log loss: logit 0.45649 < gbm_15 0.45898 < gbm_31 0.45956)
VARIANTS = ("model", "market", "none")
TEST_SEASONS = (2024, 2025, 2026)
MODEL_VERSION = "ingame_wp_v1"
MODEL_DIR = os.path.join("data", "models")


def Priors():
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    out = {}
    for s, rows in rbs.items():
        for r in rows:
            out[r["game_id"]] = {"model": r.get("p_full2"), "market": r.get("mkt_home"), "result": r.get("result"),
                                 "season": s, "week": r.get("week")}
    return out


def LoadPlays(seasons):
    dfs = [pl.read_ndjson(os.path.join("data", "playbyplay", "slim", f"pbp_slim_{s}.jsonl"), infer_schema_length=None) for s in seasons]
    return pl.concat(dfs, how="diagonal_relaxed")


def Logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def StateFeatures(st, prior_home):
    """One state dict (posteam perspective) -> feature vector. st keys: diff, gsr, hsr, qtr, down, ydstogo, yardline_100,
    pos_to, def_to, is_home, receive_2h. prior_home: pregame home win prob or None."""
    gsr = max(float(st["gsr"]), 0.0)
    elapsed = min(max((3600 - gsr) / 3600, 0.0), 1.0) if st["qtr"] <= 4 else 1.0
    pl_ = 0.0
    if prior_home is not None:
        lh = Logit(prior_home)
        pl_ = lh if st["is_home"] else -lh
    return [st["diff"], st["diff"] * math.exp(4 * elapsed), gsr, float(st["hsr"]), elapsed, min(st["qtr"], 5), 1.0 if st["qtr"] >= 5 else 0.0,
            st["down"] or 0.0, st["ydstogo"] if st["ydstogo"] is not None else 10.0, st["yardline_100"] if st["yardline_100"] is not None else 75.0,
            st["pos_to"] if st["pos_to"] is not None else 3.0, st["def_to"] if st["def_to"] is not None else 3.0,
            1.0 if st["is_home"] else 0.0, 1.0 if st["receive_2h"] else 0.0, pl_, pl_ * math.exp(-4 * elapsed)]


def Build(df, priors, variant):
    """Rows with a posteam and a decided (non-tie) game -> (X, y, meta DataFrame)."""
    df = df.filter(pl.col("posteam").is_not_null() & pl.col("qtr").is_not_null() & pl.col("game_seconds_remaining").is_not_null())
    # opening-kickoff receiver: posteam on the first qtr-1 kickoff row
    ko = (df.filter((pl.col("play_type") == "kickoff") & (pl.col("qtr") == 1)).sort("play_id")
            .group_by("game_id").agg(pl.col("posteam").first().alias("open_recv")))
    df = df.join(ko, on="game_id", how="left")
    res = {g: v["result"] for g, v in priors.items()}
    pri = {g: v[variant] for g, v in priors.items()} if variant != "none" else {}
    rows = df.to_dicts()
    X, y, keep = [], [], []
    for i, r in enumerate(rows):
        result = res.get(r["game_id"])
        if result is None or result == 0:
            continue
        is_home = r["posteam"] == r["home_team"]
        st = {"diff": (r["posteam_score"] or 0) - (r["defteam_score"] or 0), "gsr": r["game_seconds_remaining"],
              "hsr": r["half_seconds_remaining"] if r["half_seconds_remaining"] is not None else 0, "qtr": int(r["qtr"]),
              "down": r["down"], "ydstogo": r["ydstogo"], "yardline_100": r["yardline_100"],
              "pos_to": r["posteam_timeouts_remaining"], "def_to": r["defteam_timeouts_remaining"], "is_home": is_home,
              "receive_2h": r["open_recv"] is not None and r["open_recv"] != r["posteam"]}
        X.append(StateFeatures(st, pri.get(r["game_id"])))
        home_won = result > 0
        y.append(1 if home_won == is_home else 0)
        keep.append(i)
    meta = pl.DataFrame([{k: rows[i].get(k) for k in ("game_id", "season", "week", "qtr", "posteam", "home_team", "time_of_day",
                                                       "end_clock_time", "play_id", "home_wp", "vegas_home_wp")} for i in keep],
                        infer_schema_length=None)
    return np.array(X, dtype=float), np.array(y, dtype=int), meta


def Fit(X, y, config):
    if CONFIGS[config] is None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(**CONFIGS[config], monotonic_cst=[MONO.get(f, 0) for f in FEATS],
                                           early_stopping=False, random_state=20261113)
    m.fit(X, y)
    return m


def SaveModel(m, path, **info):
    """Logistic pipeline -> readable JSON (scaler mean/scale, coefficients, intercept)."""
    sc, lr = m.steps[0][1], m.steps[-1][1]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"version": MODEL_VERSION, "kind": "standardized_logistic", "features": FEATS, "mean": sc.mean_.tolist(),
                   "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]), **info}, fh, indent=1)


def LoadModel(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def PredictPos(mj, X):
    """P(posteam wins) from a SaveModel JSON."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    z = ((X - np.array(mj["mean"])) / np.array(mj["scale"])) @ np.array(mj["coef"]) + mj["intercept"]
    return 1 / (1 + np.exp(-z))


def Scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5)}


def HomeProb(p_pos, meta):
    is_home = (meta["posteam"] == meta["home_team"]).to_numpy()
    return np.where(is_home, p_pos, 1 - p_pos)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="score CONFIGS on 2023 (train 2014-2022) and print; does not change CONFIG")
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "ingame_model"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    priors = Priors()
    allp = LoadPlays(range(2014, 2027))
    allp = allp.filter(~((pl.col("season") == 2026) & (pl.col("week") > 9)))
    built = {v: Build(allp, priors, v) for v in VARIANTS}
    report = {"model_version": MODEL_VERSION, "config": CONFIG, "configs": {k: v for k, v in CONFIGS.items()}, "features": FEATS, "monotone": MONO}

    if a.select:
        sel = {}
        X, y, meta = built["model"]
        s = meta["season"].to_numpy()
        tr, va = s <= 2022, s == 2023
        for c in CONFIGS:
            m = Fit(X[tr], y[tr], c)
            sel[c] = Scores(m.predict_proba(X[va])[:, 1], y[va])
            print("select", c, sel[c], flush=True)
        report["selection_2023"] = sel
        with open(os.path.join(a.out_dir, "selection_2023.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        return

    heldout = {}
    for S in TEST_SEASONS:
        cols = {}
        meta_s = None
        for v in VARIANTS:
            X, y, meta = built[v]
            s = meta["season"].to_numpy()
            tr, te = s < S, s == S
            m = Fit(X[tr], y[tr], CONFIG)
            SaveModel(m, os.path.join(MODEL_DIR, f"ingame_wp_{v}_wf{S}.json"), variant=v, trained_seasons=[2014, S - 1])
            p = m.predict_proba(X[te])[:, 1]
            meta_s = meta.filter(pl.Series(te))
            cols[f"p_home_{v}"] = HomeProb(p, meta_s)
            yh = HomeProb(y[te].astype(float), meta_s)
            heldout.setdefault(str(S), {})[v] = Scores(cols[f"p_home_{v}"], yh)
            print(S, v, heldout[str(S)][v], flush=True)
        out = meta_s.with_columns([pl.Series(k, vv) for k, vv in cols.items()] + [pl.Series("home_win", yh.astype(int))])
        for ref in ("home_wp", "vegas_home_wp"):
            r = out.filter(pl.col(ref).is_not_null())
            heldout[str(S)][f"nflverse_{ref}_reference"] = Scores(r[ref].to_numpy(), r["home_win"].to_numpy().astype(float))
        out.write_ndjson(os.path.join(a.out_dir, f"heldout_plays_{S}.jsonl"))
    for v in VARIANTS:
        X, y, meta = built[v]
        m = Fit(X, y, CONFIG)
        SaveModel(m, os.path.join(MODEL_DIR, f"ingame_wp_{v}_live.json"), variant=v, trained_seasons=[2014, 2026], through="2026 wk9")
    report["heldout_plays"] = heldout
    with open(os.path.join(a.out_dir, "heldout_plays_scores.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(heldout, indent=1))


if __name__ == "__main__":
    Main()