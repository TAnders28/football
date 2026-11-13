r"""
Research: do pre-game injury-report features improve the walk-forward win model? Evaluation only - FrozenConfig,
PredictDay sizing and the prediction log are untouched.

    .venv\Scripts\python.exe -m Backtest.InjuryTest --out-dir DIR --base NAME

Steps
  1. Timing audit: share of injury rows whose nflverse date_modified (UTC, 2012-2024 only) is AFTER kickoff.
  2. Coverage per season (team-games with an injury report, rows, statuses, usage-join gaps).
  3. Walk-forward logits for season S fit on 2013..S-1 (same FitLogit / l2 as Lab.AttachModels), scored on
     2015-2026: log loss, Brier, accuracy, ECE vs the current p_full2 and the de-vigged closing market.
  4. Betting: the frozen value rule (best side edge vs de-vigged ML >= 8 pts, half Kelly capped at 10%,
     $10k reset each season, day stakes sized from start-of-day bank) run on each model's probabilities.
     Bases: closing_ml (posted odds, vig in) and tim_fee_est (contracts at de-vigged prob + $0.02,
     Tim's working fee estimate, NOT a verified exchange fee). The divisional add-on is not run.
Every variant here counts toward the search-contamination tally (VARIANTS below).
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.InjuryFeatures import ReadJsonl, Team
from Backtest.Lab import AllRows, AttachModels, Decimal, FitLogit, Sig, Vec, WIN_FEATS2
from Backtest.Version import ModelVersion, NewRunId

ET = ZoneInfo("America/New_York")
EVAL = list(range(2015, 2027))
HOLDOUT_AFTER = "2026-11-08"
FEE_EST = 0.02
INJ3 = ["qb_out_d", "off_out_d", "def_out_d"]
INJQ = INJ3 + ["qb_q_d", "oth_q_d"]
VARIANTS = {   # name -> (base features, extra features). Each is one searched setting.
    "full2_inj": ("full2", INJ3),
    "full2_injq": ("full2", INJQ),
    "mkt_inj": ("mkt", INJ3),
    "margin_only": ("margin", []),
    "margin_inj": ("margin", INJ3),
}


def Kickoff(g):
    t = g.get("gametime") or "13:00"
    return datetime.fromisoformat(f"{g['gameday']}T{t}").replace(tzinfo=ET).astimezone(timezone.utc)


def TimingAudit(seasons, games_by_season):
    out = {}
    for s in seasons:
        inj = ReadJsonl(os.path.join("data", "injuries", f"{s}.jsonl"))
        ko = defaultdict(list)
        for g in games_by_season.get(s, []):
            for side in ("home_team", "away_team"):
                ko[(int(g["week"]), Team(g[side]))].append(Kickoff(g))
        n = n_ts = after = after_od = n_od = 0
        lag_h = []
        for r in inj:
            n += 1
            if not r.get("date_modified"):
                continue
            n_ts += 1
            k = ko.get((int(r["week"]), Team(r["team"])))
            if not k:
                continue
            dm = datetime.fromisoformat(r["date_modified"].replace("Z", "+00:00"))
            od = r.get("report_status") in ("Out", "Doubtful")
            n_od += od
            hrs = (min(k) - dm).total_seconds() / 3600
            lag_h.append(hrs)
            if dm > min(k):
                after += 1
                after_od += od
        lag_h.sort()
        out[s] = {"rows": n, "rows_with_date_modified": n_ts, "modified_after_kickoff": after,
                  "pct_after_kickoff": round(after / n_ts, 4) if n_ts else None,
                  "out_doubtful_rows": n_od, "out_doubtful_after_kickoff": after_od,
                  "median_hours_before_kickoff": round(lag_h[len(lag_h) // 2], 1) if lag_h else None}
    return out


def Coverage(seasons):
    out = {}
    for s in seasons:
        inj = ReadJsonl(os.path.join("data", "injuries", f"{s}.jsonl"))
        feats = ReadJsonl(os.path.join("data", "injuries", f"features_{s}.jsonl"))
        st = defaultdict(int)
        for r in inj:
            st[r.get("report_status") or "none"] += 1
        played = [f for f in feats if f["gameday"] <= datetime.now(ET).date().isoformat()]
        out[s] = {"injury_rows": len(inj), "weeks": len({r["week"] for r in inj}), "status_counts": dict(st),
                  "gsis_missing": sum(not r.get("gsis_id") for r in inj),
                  "team_games": len(played), "team_games_with_report": sum(f["report_rows"] > 0 for f in played),
                  "listed_out_or_q": sum(f["n_out"] + f["n_q"] for f in played),
                  "listed_without_lagged_usage": sum(f["n_unweighted"] for f in played),
                  "qb_out_gt_half": sum(f["qb_out"] > 0.5 for f in played)}
    return out


def AttachInjury(rows_by_season):
    for s, rows in rows_by_season.items():
        feats = {(f["game_id"], f["side"]): f for f in ReadJsonl(os.path.join("data", "injuries", f"features_{s}.jsonl"))}
        for r in rows:
            h, a = feats.get((r["game_id"], "home")), feats.get((r["game_id"], "away"))
            r["inj_ok"] = bool(h and a and h["report_rows"] > 0 and a["report_rows"] > 0)
            z = lambda f, k: (f or {}).get(k, 0.0)
            r["qb_out_d"] = z(h, "qb_out") - z(a, "qb_out")
            r["off_out_d"] = (z(h, "off_out") - z(a, "off_out")) / 2
            r["def_out_d"] = (z(h, "def_out") - z(a, "def_out")) / 2
            r["qb_q_d"] = z(h, "qb_q") - z(a, "qb_q")
            r["oth_q_d"] = (z(h, "off_q") + z(h, "def_q") - z(a, "off_q") - z(a, "def_q")) / 2


def X(r, base, extra):
    if base == "full2":
        v = Vec(r, WIN_FEATS2)
    elif base == "mkt":
        v = Vec(r, ["mkt_logit"])
    else:
        v = [1.0, r["margin_logit"]]
    return v + [r[k] for k in extra]


def FitVariants(rows_by_season):
    coefs = {}
    for s in EVAL:
        train = [r for t, rs in rows_by_season.items() if 2013 <= t < s for r in rs if r["home_win"] is not None]
        for name, (base, extra) in VARIANTS.items():
            w = FitLogit([X(r, base, extra) for r in train], [r["home_win"] for r in train], l2=5.0)
            for r in rows_by_season[s]:
                r[f"p_{name}"] = float(Sig(np.dot(w, X(r, base, extra))))
            coefs.setdefault(name, {})[s] = [round(float(c), 4) for c in w]
    return coefs


def Scores(rows, key):
    rows = [r for r in rows if r["home_win"] is not None]
    n = len(rows)
    if not n:
        return {"n": 0}
    p = np.array([min(max(r[key], 1e-6), 1 - 1e-6) for r in rows]); y = np.array([r["home_win"] for r in rows])
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / n
    return {"n": n, "log_loss": round(float(ll), 5), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "accuracy": round(float(np.mean((p >= 0.5) == (y == 1))), 4), "ece": round(float(ece), 4)}


def PairedDelta(rows, a, b, sims=2000, seed=20261113):
    rows = [r for r in rows if r["home_win"] is not None]
    if not rows:
        return None
    def ll(r, k):
        p = min(max(r[k], 1e-6), 1 - 1e-6)
        return -math.log(p if r["home_win"] else 1 - p)
    d = [ll(r, a) - ll(r, b) for r in rows]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(sims))
    return {"mean_logloss_delta": round(sum(d) / len(d), 5), "ci90": [round(boots[int(.05 * sims)], 5), round(boots[int(.95 * sims)], 5)]}


def BetSeason(rows, key, basis, min_edge=0.08, frac=0.5, cap=0.10):
    bank, bets = 10_000.0, []
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["gameday"]].append(r)
    for day in sorted(by_day):
        start = bank
        picks = []
        for r in by_day[day]:
            p_home = r[key]
            best = None
            for side, p, pm, odds in (("home", p_home, r["mkt_home"], r["home_ml"]), ("away", 1 - p_home, 1 - r["mkt_home"], r["away_ml"])):
                e = p - pm
                if best is None or e > best[1]:
                    best = (side, e, p, pm, odds)
            side, e, p, pm, odds = best
            if e < min_edge:
                continue
            b = Decimal(odds) - 1 if basis == "closing_ml" else 1.0 / min(pm + FEE_EST, 0.999) - 1
            k = (b * p - (1 - p)) / b
            if k <= 0:
                continue
            won = None if r["home_win"] is None else (r["home_win"] == 1) == (side == "home")
            picks.append([r, side, min(frac * k, cap) * start, b, won, e])
        tot = sum(x[2] for x in picks)
        scale = start / tot if tot > start else 1.0
        for r, side, stake, b, won, e in picks:
            stake *= scale
            pnl = 0.0 if won is None else (stake * b if won else -stake)
            bank += pnl
            bets.append({"game_id": r["game_id"], "gameday": day, "side": side, "stake": round(stake, 2), "won": won,
                         "pnl": round(pnl, 2), "edge": round(e, 4)})
    return bank, bets


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "injury"))
    ap.add_argument("--base", default="injury_test")
    a = ap.parse_args()
    run_id = NewRunId()
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    AttachInjury(rbs)
    coefs = FitVariants(rbs)
    from Backtest.Lab import LoadSeason
    games = {s: LoadSeason(s) for s in range(2012, 2027)}
    timing = TimingAudit(range(2012, 2027), games)
    coverage = Coverage(range(2013, 2027))

    models = ["p_mkt", "p_calib", "p_full2"] + [f"p_{v}" for v in VARIANTS]
    sets = {str(s): [r for r in rbs[s] if r["gameday"] <= HOLDOUT_AFTER] for s in EVAL}
    sets["2015-2022"] = [r for s in range(2015, 2023) for r in rbs[s]]
    sets["2023-2025"] = [r for s in range(2023, 2026) for r in rbs[s]]
    sets["2026_pre_holdout"] = sets["2026"]
    sets["2015-2026_pre_holdout"] = sets["2015-2022"] + sets["2023-2025"] + sets["2026"]
    sets["2026_forward_holdout"] = [r for r in rbs[2026] if r["gameday"] > HOLDOUT_AFTER]
    sets["2015-2026_inj_ok_only"] = [r for r in sets["2015-2026_pre_holdout"] if r["inj_ok"]]
    scores = {k: {m: Scores(v, m) for m in models} for k, v in sets.items()}
    deltas = {k: {"full2_inj_vs_full2": PairedDelta(v, "p_full2_inj", "p_full2"),
                  "full2_injq_vs_full2": PairedDelta(v, "p_full2_injq", "p_full2"),
                  "mkt_inj_vs_calib": PairedDelta(v, "p_mkt_inj", "p_calib"),
                  "margin_inj_vs_margin_only": PairedDelta(v, "p_margin_inj", "p_margin_only")}
              for k, v in sets.items() if k in ("2015-2022", "2023-2025", "2026_pre_holdout", "2015-2026_pre_holdout")}

    betting, tidy, all_bets = {}, [], []
    for key in ("p_full2", "p_full2_inj", "p_full2_injq"):
        for basis in ("closing_ml", "tim_fee_est"):
            per = {}
            for s in EVAL:
                rows = sets[str(s)]
                end, bets = BetSeason(rows, key, basis)
                per[s] = {"end_bank": round(end, 2), "bets": len(bets), "won": sum(1 for b in bets if b["won"]),
                          "staked": round(sum(b["stake"] for b in bets), 2)}
                for b in bets:
                    all_bets.append({"model": key, "basis": basis, "season": s, **b})
                tidy.append({"date": max(r["gameday"] for r in rows), "series": f"{key}|{basis}", "season": s,
                             "end_bank": round(end, 2), "bets": len(bets)})
            ends = [v["end_bank"] for v in per.values()]
            betting[f"{key}|{basis}"] = {"per_season": per, "sum_profit": round(sum(e - 10_000 for e in ends), 2),
                                         "worst_season_end": min(ends), "median_end": sorted(ends)[len(ends) // 2]}

    last_coefs = {v: {"features": ["const"] + {"full2": WIN_FEATS2, "mkt": ["mkt_logit"], "margin": ["margin_logit"]}[VARIANTS[v][0]] + VARIANTS[v][1],
                      "fit_for_2026": coefs[v][2026]} for v in VARIANTS}
    report = {"run_id": run_id, "model_version": ModelVersion(), "created_at": datetime.now(ET).isoformat(),
              "variants_counted": len(VARIANTS), "variants": {k: list(v) for k, v in VARIANTS.items()},
              "timing_audit": timing, "coverage": coverage, "scores": scores, "paired_logloss_deltas": deltas,
              "betting": betting, "coefficients_2026_fit": last_coefs,
              "fee_basis": "tim_est_0.02 (Tim's working estimate, not a verified exchange fee) and closing_ml (vig in)"}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "series", "metric", "season_set", "value", "n"])
        w.writeheader()
        for k, mm in scores.items():
            for m, sc in mm.items():
                for metric in ("log_loss", "brier", "accuracy", "ece"):
                    if sc.get("n"):
                        w.writerow({"date": max(r["gameday"] for r in sets[k]), "series": m, "metric": metric,
                                    "season_set": k, "value": sc[metric], "n": sc["n"]})
        for t in tidy:
            w.writerow({"date": t["date"], "series": t["series"], "metric": "end_bank_10k_reset", "season_set": t["season"],
                        "value": t["end_bank"], "n": t["bets"]})
    with open(os.path.join(a.out_dir, f"{a.base}_bets.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_bets[0].keys()))
        w.writeheader(); w.writerows(all_bets)
    print(json.dumps({"scores": {k: scores[k] for k in ("2015-2022", "2023-2025", "2026_pre_holdout", "2015-2026_pre_holdout", "2026_forward_holdout", "2015-2026_inj_ok_only")},
                      "deltas": deltas, "betting": {k: {kk: vv for kk, vv in v.items() if kk != "per_season"} for k, v in betting.items()},
                      "coefs": last_coefs}, indent=1, default=str))


if __name__ == "__main__":
    Main()