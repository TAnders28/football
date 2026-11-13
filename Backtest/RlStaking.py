r"""
RlStaking: train an RL-style day-level staking policy on StakingEnv and compare it with fixed baselines on
ONE continuous $10k bankroll (Tim, 2026-11-13). Simulation only: no wagers, PredictDay/FrozenConfig untouched.

    .venv\Scripts\python.exe -m Backtest.RlStaking --stage train      # all walk-forward fits (training data only)
    .venv\Scripts\python.exe -m Backtest.RlStaking --stage val        # chained walk-forward 2019-2024 + baselines
    .venv\Scripts\python.exe -m Backtest.RlStaking --stage holdout    # ONE-TIME frozen 2025-2026 holdout

Learner: cross-entropy policy search (CEM) over the 51 parameters of StakingEnv's contextual policy
(Kelly multiplier, edge threshold, parlay stake and leg-count scores as functions of the day state).
Each CEM iteration draws fresh day-bootstrap paths (H days) from the TRAINING seasons; half the paths redraw
outcomes from a probability calibrated on training days only (synthetic augmentation). Common random numbers
across the population. The policy is the final CEM mean. Hyperparameters below were fixed before any run.
Walk-forward: the policy for season S is trained on 2014..S-1. Seasons 2025-2026 are the frozen holdout,
scored once (a marker file records the first use; reruns are refused without --force and are flagged).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import Backtest.MartingaleParlay as MP
import Backtest.MillionTarget as MT
import Backtest.StakingEnv as SE
from Backtest.Lab import AllRows, AttachModels
from Backtest.Version import CodeVersion, NewRunId

POOL_FIRST = 2014
VAL_SEASONS = list(range(2019, 2025))
HOLDOUT = [2025, 2026]
TRAIN_LASTS = list(range(2018, 2025))
OBJECTIVES = ("million", "growth")
POP, ELITE, ITERS, PATHS, H = 96, 12, 40, 256, 220
SYNTH_FRAC, SIGMA0, SMOOTH, SIGMA_MIN = 0.5, 1.0, 0.7, 0.05
SEED = 20261113
N_SIMS, BOOT_H = 5000, 220
PRIOR_SETTINGS = 2757
KELLY = (0.25, 0.5, 1.0)
MART = {"mart_best_qualifying": ("model_p100_2leg", 0.02, 1.5), "mart_max_p1m": ("market_p300_3leg", 0.10, "recover")}
MP_BASIS = {"closing_ml": "posted", "tim_fee_est": "fee_est"}
WORK = os.path.join("data", "backtest", "rl")
THETAS = os.path.join(WORK, "thetas.json")
HOLDOUT_MARK = os.path.join(WORK, "HOLDOUT_USED.json")
OUT_DIR = r"E:\AgentDocs\American-Footbot\2026-11-13_rl-staking-design"
BASE = "2026-11-13_rl-staking-design"

G = {}


def Load():
    if not G:
        rbs = AllRows(2012, 2026)
        AttachModels(rbs)
        days = SE.BuildDays(rbs, list(range(POOL_FIRST, 2027)))
        G.update(rbs=rbs, days=days, packs={b: SE.Pack(days, b) for b in SE.BASES})
    return G


# ----------------------------------------------------------------------------- training (training seasons only)
def Train(job):
    basis, objective, train_last, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pack = Load()["packs"][basis]
    mask = (pack["seasons"] >= POOL_FIRST) & (pack["seasons"] <= train_last)
    assert not np.isin(pack["seasons"][mask], HOLDOUT).any()
    pool = np.flatnonzero(mask)
    w = SE.CalibFit(pack, mask)
    pcal = SE.PCal(pack, w)
    rng = np.random.default_rng(seed)
    mu, sigma = SE.InitTheta(), np.full(SE.N_PARAM, SIGMA0)
    hist, t0 = [], time.time()
    for it in range(ITERS):
        cand = mu + sigma * rng.standard_normal((POP, SE.N_PARAM))
        cand[0] = mu
        idx = rng.choice(pool, size=(PATHS, H))
        synth = rng.random(PATHS) < SYNTH_FRAC
        U = np.random.default_rng(int(rng.integers(1 << 62))).random((H, PATHS, SE.MAX_C))
        res = SE.Simulate(pack, np.tile(idx, (POP, 1)), np.repeat(cand, PATHS, 0), synth=np.tile(synth, POP), pcal=pcal,
                          u=lambda j: np.tile(U[j], (POP, 1)))
        score = SE.Objective(res, objective).reshape(POP, PATHS).mean(1)
        order = np.argsort(-score)
        el = cand[order[:ELITE]]
        mu = SMOOTH * el.mean(0) + (1 - SMOOTH) * mu
        sigma = np.maximum(SMOOTH * el.std(0) + (1 - SMOOTH) * sigma, SIGMA_MIN)
        hist.append({"iter": it, "best": float(score[order[0]]), "elite_mean": float(score[order[:ELITE]].mean()),
                     "mean_cand": float(score[0]), "p_hit_mean_cand": float(res["hit"].reshape(POP, PATHS)[0].mean())})
    # in-sample check of the final mean: 2,000 fresh real-outcome bootstrap paths from the training days
    ev = np.random.default_rng(seed + 1).choice(pool, size=(2000, H))
    r = SE.Simulate(pack, ev, np.repeat(mu[None, :], 2000, 0))
    return {"basis": basis, "objective": objective, "train_last": train_last, "seed": seed, "theta": mu.tolist(),
            "calib_w": w.tolist(), "evaluated_vectors": POP * ITERS, "seconds": round(time.time() - t0, 1), "history": hist,
            "train_fit": Summ(r)}


def Summ(r, start=SE.START):
    e, h, ru = r["end"], r["hit"], r["ruin"]
    n = len(e)
    p = float(h.mean())
    out = {"n_paths": n, "p_1m": round(p, 4), "p_1m_mc_se": round(math.sqrt(p * (1 - p) / n), 4), "p_ruin": round(float(ru.mean()), 4),
           "median_end": round(float(np.median(e)), 2), "p5_end": round(float(np.percentile(e, 5)), 2),
           "p95_end": round(float(np.percentile(e, 95)), 2), "p_end_above_start": round(float((e > start).mean()), 4)}
    if "maxdd" in r:
        out["median_max_dd"] = round(float(-np.median(r["maxdd"])), 4)
    for k in ("mean_exposure", "bet_day_share", "parlay_day_share"):
        if k in r:
            out[k] = round(float(np.mean(r[k])), 4)
    return out


def StageTrain(procs):
    jobs = [(b, o, t, SEED + 1000 * i) for i, (b, o, t) in enumerate((b, o, t) for b in SE.BASES for o in OBJECTIVES for t in TRAIN_LASTS)]
    t0 = time.time()
    with Pool(procs) as p:
        out = p.map(Train, jobs, chunksize=1)
    blob = {"run_id": NewRunId(), "code": CodeVersion(), "python": sys.version.split()[0], "jobs": out,
            "hyper": {"pop": POP, "elite": ELITE, "iters": ITERS, "paths": PATHS, "H": H, "synth_frac": SYNTH_FRAC,
                      "sigma0": SIGMA0, "smooth": SMOOTH, "seed": SEED}, "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(WORK, exist_ok=True)
    with open(THETAS, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    print(f"trained {len(out)} policies in {blob['wall_seconds']} s; code {blob['code']}")
    for j in out:
        print(j["basis"], j["objective"], j["train_last"], j["seconds"], "s", j["train_fit"])


def Theta(blob, basis, objective, train_last):
    j = next(x for x in blob["jobs"] if x["basis"] == basis and x["objective"] == objective and x["train_last"] == train_last)
    return np.array(j["theta"])


# ----------------------------------------------------------------------------- baselines
def MtPack(rbs, seasons, basis, dates):
    days = [d for d in MT.BuildDays(rbs, seasons, basis) if d["date"] in dates]
    return MT.Pack(days)


def Baselines(g, basis, mask, idx):
    """{name: (sim result, replay equity list)} on the given day mask; idx None = historical replay only."""
    pack = g["packs"][basis]
    dates = [d for d, m in zip(pack["dates"], mask) if m]
    sel = np.flatnonzero(mask)
    out = {}
    seasons = sorted(set(pack["seasons"][mask].tolist()))
    mtp = MtPack(g["rbs"], seasons, basis, set(dates))
    assert mtp["dates"] == dates, "frozen-rule day list differs from StakingEnv day list"
    mt_cfgs = {"frozen_rules": MT.REFERENCES[0]} | {f"frozen_picks_kelly{k:g}": MT.Cfg(f"k{k}", "kelly", "kelly", k, cap=1.0) for k in KELLY}
    for name, cfg in mt_cfgs.items():
        rep = MT.Historical(mtp, cfg)
        res = MT.Simulate(mtp, idx, cfg) if idx is not None else None
        out[name] = (res, rep["equity"])
    for k in KELLY:
        th = SE.FixedTheta(k)
        rep = SE.Simulate(pack, sel[None, :], th, record=True)
        res = SE.Simulate(pack, sel[idx], SE.FixedTheta(k, n=len(idx))) if idx is not None else None
        out[f"all_edge_kelly{k:g}"] = (res, [float(s["bank"][0]) for s in rep["path"]])
    out["no_bet"] = (SE.Simulate(pack, sel[idx], SE.FixedTheta(0.0, n=len(idx))) if idx is not None else None, [SE.START] * len(sel))
    mpd = MP.BuildDays(g["rbs"], list(range(POOL_FIRST, 2027)))
    assert [d["date"] for d in mpd] == pack["dates"]
    for name, (series, base, mult) in MART.items():
        sr = next(s for s in MP.Series() if s["name"] == series)
        sub = MP.SubSeries(MP.BuildSeries(mpd, sr), mask)
        cfg = MP.Cfg(series, base, mult)
        rep = MP.Replay(sub, cfg, SE.START, MP_BASIS[basis])
        res = MP.Simulate(sub, idx, cfg, SE.START, MP_BASIS[basis]) if idx is not None else None
        out[name] = (res, rep["equity"])
    return out


def Dd(eq):
    pk, out = -1.0, []
    for e in eq:
        pk = max(pk, e)
        out.append(round(-(1 - e / pk) if pk > 0 else 0.0, 4))
    return out


def Chain(g, blob, basis, objective, seasons, final_last=None):
    """Continuous bank through `seasons`; season S uses the policy trained on <= S-1 (or final_last if set)."""
    pack = g["packs"][basis]
    state, eq, per = None, [], []
    for s in seasons:
        sel = np.flatnonzero(pack["seasons"] == s)
        tl = final_last if (final_last is not None and s in HOLDOUT) else s - 1
        r = SE.Simulate(pack, sel[None, :], Theta(blob, basis, objective, tl)[None, :], state=state, record=True)
        b0 = float(state["bank"][0]) if state else SE.START
        state = {k: r[k] for k in ("peak", "ema", "hit", "ruin")} | {"bank": r["end"]}
        eq += [float(x["bank"][0]) for x in r["path"]]
        per.append({"season": s, "policy_trained_through": tl, "start": round(b0, 2), "end": round(float(r["end"][0]), 2),
                    "days": len(sel), "bet_day_share": round(float(r["bet_day_share"][0]), 3),
                    "parlay_day_share": round(float(r["parlay_day_share"][0]), 3), "mean_exposure": round(float(r["mean_exposure"][0]), 4),
                    "hit": bool(r["hit"][0]), "ruin": bool(r["ruin"][0])})
    return eq, per


def Tidy(rows, dates, eq, series, basis, stage):
    for d, e, x in zip(dates, eq, Dd(eq)):
        rows.append({"date": d, "series": series, "equity": round(e, 2), "drawdown_pct": x, "basis": basis, "stage": stage})


def Write(name, payload, tidy, summary):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{BASE}_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    for suffix, rows in (("", tidy), ("_summary", summary)):
        if not rows:
            continue
        with open(os.path.join(OUT_DIR, f"{BASE}_{name}{suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def StageVal():
    g, blob = Load(), json.load(open(THETAS, encoding="utf-8"))
    tidy, summ, payload = [], [], {"run_id": NewRunId(), "code": CodeVersion(), "thetas_run": blob["run_id"], "chains": {}}
    for basis in SE.BASES:
        pack = g["packs"][basis]
        mask = np.isin(pack["seasons"], VAL_SEASONS)
        dates = [d for d, m in zip(pack["dates"], mask) if m]
        for obj in OBJECTIVES:
            eq, per = Chain(g, blob, basis, obj, VAL_SEASONS)
            name = f"learned_{obj}"
            Tidy(tidy, dates, eq, name, basis, "val_chain_2019_2024")
            payload["chains"][f"{basis}|{name}"] = per
            summ.append({"date": dates[-1], "series": name, "basis": basis, "stage": "val_chain_2019_2024", "end": round(eq[-1], 2),
                         "max_dd": min(Dd(eq)), "hit_1m": int(per[-1]["hit"]), "ruin": int(per[-1]["ruin"])})
        for name, (_, eq) in Baselines(g, basis, mask, None).items():
            Tidy(tidy, dates, eq, name, basis, "val_chain_2019_2024")
            summ.append({"date": dates[-1], "series": name, "basis": basis, "stage": "val_chain_2019_2024", "end": round(eq[-1], 2),
                         "max_dd": min(Dd(eq)), "hit_1m": int(max(eq) >= SE.TARGET), "ruin": int(min(eq) <= SE.RUIN)})
    payload["summary"] = summ
    Write("validation", payload, tidy, summ)
    for s in summ:
        print(s)
    for k, v in payload["chains"].items():
        print(k, [(p["season"], p["end"], p["bet_day_share"], p["parlay_day_share"], p["mean_exposure"]) for p in v])


def StageHoldout(force):
    if os.path.exists(HOLDOUT_MARK) and not force:
        sys.exit(f"holdout already used ({HOLDOUT_MARK}); rerun only with --force (it will be flagged)")
    g, blob = Load(), json.load(open(THETAS, encoding="utf-8"))
    mark = {"first_used_run": NewRunId(), "code": CodeVersion(), "thetas_run": blob["run_id"], "forced_rerun": bool(force)}
    os.makedirs(WORK, exist_ok=True)
    with open(HOLDOUT_MARK, "w", encoding="utf-8") as fh:
        json.dump(mark, fh, indent=1)
    tidy, summ = [], []
    payload = {"run_id": mark["first_used_run"], "code": mark["code"], "thetas_run": blob["run_id"], "forced_rerun": bool(force),
               "n_sims": N_SIMS, "boot_h": BOOT_H, "seed": SEED, "holdout": HOLDOUT, "policy_trained_through": 2024,
               "learned": {}, "baselines": {}}
    for basis in SE.BASES:
        pack = g["packs"][basis]
        mask = np.isin(pack["seasons"], HOLDOUT)
        sel = np.flatnonzero(mask)
        dates = [pack["dates"][i] for i in sel]
        idx = np.random.default_rng(SEED).integers(0, len(sel), size=(N_SIMS, BOOT_H))
        payload["holdout_days"] = len(sel)
        payload["holdout_dates"] = [dates[0], dates[-1]]
        for obj in OBJECTIVES:
            th = Theta(blob, basis, obj, 2024)
            rep = SE.Simulate(pack, sel[None, :], th[None, :], record=True)
            eq = [float(x["bank"][0]) for x in rep["path"]]
            boot = Summ(SE.Simulate(pack, sel[idx], np.repeat(th[None, :], N_SIMS, 0)))
            name = f"learned_{obj}"
            Tidy(tidy, dates, eq, name, basis, "holdout_replay")
            legs = [int(x["legs"][0]) for x in rep["path"]]
            payload["learned"][f"{basis}|{name}"] = {"replay_end": round(eq[-1], 2), "replay_max_dd": min(Dd(eq)), "boot": boot,
                                                     "replay_bet_days": int(sum(1 for x in rep["path"] if x["exposure"][0] > 0)),
                                                     "replay_parlay_days": int(sum(1 for n in legs if n > 0)),
                                                     "replay_mean_lambda": round(float(np.mean([x["lam"][0] for x in rep["path"]])), 3)}
            summ.append({"date": dates[-1], "series": name, "basis": basis, "replay_end": round(eq[-1], 2), "replay_max_dd": min(Dd(eq)), **boot})
            feq, fper = Chain(g, blob, basis, obj, VAL_SEASONS + HOLDOUT, final_last=2024)
            fd = [d for d, m in zip(pack["dates"], np.isin(pack["seasons"], VAL_SEASONS + HOLDOUT)) if m]
            Tidy(tidy, fd, feq, name, basis, "full_chain_2019_2026")
            payload["learned"][f"{basis}|{name}"]["full_chain"] = fper
        for name, (res, eq) in Baselines(g, basis, mask, idx).items():
            boot = Summ(res)
            Tidy(tidy, dates, eq, name, basis, "holdout_replay")
            payload["baselines"][f"{basis}|{name}"] = {"replay_end": round(eq[-1], 2), "replay_max_dd": min(Dd(eq)), "boot": boot}
            summ.append({"date": dates[-1], "series": name, "basis": basis, "replay_end": round(eq[-1], 2), "replay_max_dd": min(Dd(eq)), **boot})
    keys = sorted({k for s in summ for k in s}, key=lambda k: list(summ[0].keys()).index(k) if k in summ[0] else 99)
    summ = [{k: s.get(k) for k in keys} for s in summ]
    payload["summary"] = summ
    payload["settings_scored_on_holdout"] = len(summ)
    Write("holdout", payload, tidy, summ)
    for s in summ:
        print(s)


def Main():
    ap = argparse.ArgumentParser(description="RL-style staking policy (simulation only, no wagers).")
    ap.add_argument("--stage", choices=("train", "val", "holdout", "smoke"), required=True)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(f"python {sys.version.split()[0]} code {CodeVersion()} args {vars(a)}", flush=True)
    if a.stage == "train":
        StageTrain(a.procs)
    elif a.stage == "val":
        StageVal()
    elif a.stage == "holdout":
        StageHoldout(a.force)
    else:
        global ITERS
        ITERS = 2
        t0 = time.time()
        j = Train(("tim_fee_est", "million", 2018, SEED))
        print("smoke 2 iters", round(time.time() - t0, 1), "s", j["train_fit"], j["history"][-1])


if __name__ == "__main__":
    Main()
