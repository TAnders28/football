r"""
Aggressive sizing for Tim's $1M goal: the frontier of P(hit $1M) versus P(ruin) on ONE continuous $10,000
bankroll over the 2023 -> 2026-to-date window (no deposits, no resets). Research only: nothing here changes
FrozenConfig, PredictDay or any live rule.

    .venv\Scripts\python.exe -m Backtest.MillionTarget
    .venv\Scripts\python.exe -m Backtest.MillionTarget --out-dir E:\AgentDocs\American-Footbot\2026-11-09_aggressive-million-target --base 2026-11-09_aggressive-million-target

Bets: the FROZEN selection rules only (value full2 >= 8 pts, else divisional full3 >= 2 pts, one bet per game).
Only the STAKE SIZE is searched (finite GRID below). Model probabilities are not touched.

Cash / settlement mechanics (also audited against Backtest.Compound by AuditCompound)
  * Settlement unit = one ET game day (nflverse has kickoff times but no final-whistle times). Every stake for a
    day is sized from the cash at the START of that day and all of them are reserved at once; nothing is
    settled until the day ends. So a 4pm stake never uses 1pm winnings and no outcome leaks into another
    same-day stake. This is conservative (no intra-day compounding).
  * Reservation: stakes are fractions f_i of day-start cash. If sum f_i > 1 they are all scaled by 1/sum f_i,
    so the day stakes exactly 100% of cash at most. No borrowing, no negative cash, no credit.
  * Payout bases (separate runs): "closing_ml" pays the posted closing American odds (vig included);
    "tim_fee_est" buys $1 contracts at (de-vigged prob + $0.02), Tim's working fee estimate, so a win
    returns stake / (p + 0.02). That assumes a fair-priced exchange fill, which is optimistic.
  * Pushes (ties) return the stake. Bank below $1 = exhausted (no stake possible) and stops betting.
  * No size limits are modelled: historical available liquidity/limits are absent from the data, so stakes of
    $100k+ on one NFL moneyline are assumed fillable at the closing price. They might not be.

Target and ruin
  * PRIMARY (first passage, both barriers absorbing): a path stops betting and holds cash at the FIRST day end
    with bank >= $1,000,000 (HIT) or bank <= $1,000 (RUIN), whichever comes first. HIT-before-ruin and RUIN are
    mutually exclusive. p_hit_before_ruin + p_ruin + p_neither = 1. Ending-bank quantiles use these absorbed paths.
  * SECONDARY, kept for transparency: p_hit_ever = ever reaching $1M with recovery allowed after a fall to <= $1k
    (ruin NOT absorbing, target still absorbing). p_ruin_then_hit = p_hit_ever - p_hit_before_ruin.
    (Study run 20261109T125827-ebf63a reported p_hit_ever under the name p_hit.)
  * A separate "play-on" run (neither barrier absorbing) reports terminal >= $1M.
  * ZERO = bank < $1 (true exhaustion; under absorbing ruin this can only happen on the ruin day itself).

Bootstrap: resample whole game DAYS (all of a day's bets together, keeping same-slate dependence) with
replacement, T = number of days in the evaluation window, N_SIMS paths, one common index matrix per pool
(common random numbers across configs). These are conditional historical replay frequencies, NOT future odds:
they can't create an edge the sample doesn't have, and they inherit its selection bias.

Nested comparator: before each season S, every GRID config is bootstrapped on the days of TRAIN_FROM..S-1 with
the same horizon T; the highest first-passage P(hit before ruin) wins (ties -> higher median ending bank) and is played on season S only,
continuing the real bank. The bet-selection families were themselves searched on 2023-25, so even this is not
untouched data.
"""

import argparse
import csv
import json
import os
from itertools import groupby

import numpy as np

from Backtest.Compound import EVAL_SEASONS, TRAIN_FROM, Compound, Plan, SelFrozenUnion, SizeFrozen
from Backtest.Lab import START, AllRows, AttachModels, Decimal
from Backtest.Version import CodeVersion, NewRunId

TARGET = 1_000_000.0
RUIN = 1_000.0
ZERO = 1.0
FEE_EST = 0.02
N_SIMS = 5000
SEED = 20261109
DERISK_AT = 100_000.0
PRIOR_SETTINGS = {"lab_volume_hunt": 2073, "compound_2026_11_09": 238}
DEFAULT_OUT = os.path.join("data", "backtest", "million")
BASES = ("closing_ml", "tim_fee_est")


# ----------------------------------------------------------------------------- sizing grid (finite, predeclared)
def Cfg(name, family, weight, hi, lo=None, cap=1.0, rule="const"):
    return {"name": name, "family": family, "weight": weight, "hi": hi, "lo": hi if lo is None else lo,
            "cap": cap, "rule": rule}


def BuildGrid():
    g = []
    for s in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        g.append(Cfg(f"flat{s * 100:g}", "flat", "flat", s))
    for m in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for cap in (0.2, 0.3, 0.5, 1.0):
            g.append(Cfg(f"kelly{m:g}x_cap{int(cap * 100)}", "over_kelly", "kelly", m, cap=cap))
    for hi in (0.15, 0.20, 0.30, 0.40):
        for lo in (0.05, 0.10):
            g.append(Cfg(f"derisk_flat{hi * 100:g}to{lo * 100:g}", "target_derisk", "flat", hi, lo, rule="derisk"))
    for hi in (2.0, 3.0, 4.0):
        g.append(Cfg(f"derisk_kelly{hi:g}xto1x_cap50", "target_derisk", "kelly", hi, 1.0, cap=0.5, rule="derisk"))
    for lo, hi in ((0.10, 0.20), (0.10, 0.30), (0.15, 0.30), (0.20, 0.40)):
        g.append(Cfg(f"schedule_flat{lo * 100:g}or{hi * 100:g}", "target_schedule", "flat", hi, lo, rule="schedule"))
    for lo, hi in ((1.0, 2.0), (1.0, 3.0), (2.0, 4.0)):
        g.append(Cfg(f"schedule_kelly{lo:g}xor{hi:g}x_cap50", "target_schedule", "kelly", hi, lo, cap=0.5, rule="schedule"))
    return g


GRID = BuildGrid()
REFERENCES = [Cfg("frozen_sizing", "reference", "frozen", 0.0), Cfg("no_bet", "reference", "flat", 0.0)]


# ----------------------------------------------------------------------------- days
def BuildDays(rows_by_season, seasons, basis):
    """One entry per ET game day (days without a bet included, so the horizon counts real time)."""
    days = []
    for s in seasons:
        select = SelFrozenUnion()(rows_by_season, s)
        rows = sorted(rows_by_season.get(s, []), key=lambda r: (r["gameday"], r.get("gametime") or "", r["game_id"]))
        for day, group in groupby(rows, key=lambda r: r["gameday"]):
            k, ret, tier, ids = [], [], [], []
            for r in group:
                if r.get("home_ml") is None or r.get("away_ml") is None:
                    continue
                for c in select(r):
                    b = Decimal(c["odds"]) - 1
                    pay = b if basis == "closing_ml" else 1.0 / min(c["p_mkt"] + FEE_EST, 0.999) - 1
                    k.append(max(0.0, (b * c["p"] - (1 - c["p"])) / b))
                    ret.append(0.0 if c["won"] is None else (pay if c["won"] else -1.0))
                    tier.append(c.get("tier", 0))
                    ids.append(r["game_id"])
            days.append({"date": day, "season": s, "k": k, "r": ret, "tier": tier, "game_ids": ids})
    return days


def Pack(days):
    nb = max([len(d["k"]) for d in days] + [1])
    K, R, M, TR = (np.zeros((len(days), nb)) for _ in range(4))
    for i, d in enumerate(days):
        n = len(d["k"])
        K[i, :n], R[i, :n], M[i, :n], TR[i, :n] = d["k"], d["r"], 1.0, d["tier"]
    return {"K": K, "R": R, "M": M, "TIER": TR, "dates": [d["date"] for d in days], "n": len(days)}


# ----------------------------------------------------------------------------- engine (vectorised over paths)
def Fractions(cfg, bank, t, T, K, M, TIER):
    """Stake fractions of day-start cash, shape (paths, bets); scaled so each path stakes <= 100% of cash."""
    if cfg["weight"] == "frozen":
        f = np.where(TIER == 0, np.minimum(0.5 * K, 0.10), 0.02) * M
    else:
        if cfg["rule"] == "derisk":
            s = np.where(bank < DERISK_AT, cfg["hi"], cfg["lo"])
        elif cfg["rule"] == "schedule":
            sched = START * (TARGET / START) ** (t / max(T, 1))
            s = np.where(bank < sched, cfg["hi"], cfg["lo"])
        else:
            s = np.full(bank.shape, cfg["hi"])
        w = M if cfg["weight"] == "flat" else K
        f = np.minimum(s[:, None] * w, cfg["cap"])
    total = f.sum(axis=1)
    f = f * np.where(total > 1.0, 1.0 / np.maximum(total, 1e-12), 1.0)[:, None]
    return f


def Simulate(pack, idx, cfg, stop=True, start=START, T=None, t0=0, record=False, absorb_ruin=True):
    """idx: (paths, steps) day indices into pack. Returns per-path outcomes (and the path if record).
    stop: target absorbing (hold cash after first hit). absorb_ruin: ruin absorbing (hold cash after first
    bank <= RUIN); then hit and ruin are mutually exclusive first-passage events."""
    paths, steps = idx.shape
    T = T or steps
    bank = np.full(paths, float(start))
    peak = bank.copy()
    maxdd = np.zeros(paths)
    lo = bank.copy()
    hit = np.zeros(paths, bool)
    ruin = np.zeros(paths, bool)
    zero = np.zeros(paths, bool)
    hit_step = np.full(paths, -1)
    staked_max = np.zeros(paths)
    rec = []
    for j in range(steps):
        d = idx[:, j]
        K, R, M, TIER = pack["K"][d], pack["R"][d], pack["M"][d], pack["TIER"][d]
        f = Fractions(cfg, bank, t0 + j, T, K, M, TIER)
        live = ~zero & ~(hit & stop) & ~(ruin & absorb_ruin)
        f = f * live[:, None]
        staked_max = np.maximum(staked_max, f.sum(axis=1))
        bank = np.maximum(bank * (1.0 + (f * R).sum(axis=1)), 0.0)
        can_hit = ~hit & ~(ruin & absorb_ruin)
        new_hit = (bank >= TARGET) & can_hit
        hit_step = np.where(new_hit, j, hit_step)
        hit |= new_hit
        ruin |= (bank <= RUIN) & ~hit
        zero |= bank < ZERO
        peak = np.maximum(peak, bank)
        maxdd = np.maximum(maxdd, 1 - bank / peak)
        lo = np.minimum(lo, bank)
        if record:
            rec.append((bank.copy(), 1 - bank / peak))
    out = {"end": bank, "hit": hit, "ruin": ruin, "zero": zero, "maxdd": maxdd, "min": lo, "hit_step": hit_step,
           "max_day_exposure": staked_max}
    if record:
        out["path"] = rec
    return out


def CommonIndex(n_days, steps, n_sims=N_SIMS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n_days, size=(n_sims, steps))


def BootStats(res, n, ever=None):
    """res: first-passage run (absorb_ruin=True). ever: same seeds with ruin not absorbing (recovery allowed).
    MC SE is the binomial standard error in probability units (x100 = percentage points), 1 SE ~ 68%;
    the 95% interval is +-1.96 SE (normal approx). It covers simulation noise only, not sample uncertainty."""
    e = res["end"]
    assert not (res["hit"] & res["ruin"]).any()
    p = float(res["hit"].mean())
    se = float(np.sqrt(p * (1 - p) / n))
    out = {"p_hit_before_ruin": round(p, 4), "p_hit_before_ruin_mc_se": round(se, 4),
           "p_hit_before_ruin_mc95": [round(max(0.0, p - 1.96 * se), 4), round(min(1.0, p + 1.96 * se), 4)],
           "p_ruin": round(float(res["ruin"].mean()), 4), "p_zero": round(float(res["zero"].mean()), 4),
           "p_neither": round(float((~res["hit"] & ~res["ruin"]).mean()), 4),
           "median_end": round(float(np.median(e)), 2), "p5_end": round(float(np.percentile(e, 5)), 2),
           "p25_end": round(float(np.percentile(e, 25)), 2), "p75_end": round(float(np.percentile(e, 75)), 2),
           "p95_end": round(float(np.percentile(e, 95)), 2), "p_below_start": round(float((e < START).mean()), 4),
           "median_max_dd": round(float(-np.median(res["maxdd"])), 4)}
    if ever is not None:
        pe = float(ever["hit"].mean())
        out.update({"p_hit_ever": round(pe, 4), "p_hit_ever_mc_se": round(float(np.sqrt(pe * (1 - pe) / n)), 4),
                    "p_ruin_then_hit": round(float((ever["ruin"] & ever["hit"]).mean()), 4),
                    "median_end_ever_policy": round(float(np.median(ever["end"])), 2),
                    "p5_end_ever_policy": round(float(np.percentile(ever["end"], 5)), 2),
                    "p95_end_ever_policy": round(float(np.percentile(ever["end"], 95)), 2)})
    return out


def Historical(pack, cfg, stop=True, absorb_ruin=True):
    res = Simulate(pack, np.arange(pack["n"])[None, :], cfg, stop=stop, record=True, absorb_ruin=absorb_ruin)
    eq = [round(float(b[0]), 2) for b, _ in res["path"]]
    dd = [round(float(-x[0]), 4) for _, x in res["path"]]
    hs = int(res["hit_step"][0])
    return {"end": eq[-1] if eq else START, "max_dd": round(float(-res["maxdd"][0]), 4), "hit": bool(res["hit"][0]),
            "hit_date": pack["dates"][hs] if hs >= 0 else None, "ruin": bool(res["ruin"][0]), "zero": bool(res["zero"][0]),
            "min_bank": round(float(res["min"][0]), 2), "max_day_exposure": round(float(res["max_day_exposure"][0]), 4),
            "equity": eq, "drawdown": dd}


def Frontier(rows, key="p_hit_before_ruin"):
    """Nondominated: no other row has key >= and p_ruin <= with one strictly better."""
    out = []
    for a in rows:
        dominated = any((b[key] >= a[key] and b["p_ruin"] <= a["p_ruin"]) and
                        (b[key] > a[key] or b["p_ruin"] < a["p_ruin"]) for b in rows)
        if not dominated:
            out.append(a["name"])
    return out


# ----------------------------------------------------------------------------- mechanics audit of Backtest.Compound
def AuditCompound(rows_by_season, seasons, sizers=("kelly1_cap100", "flat20pct", "flat10pct")):
    """Re-derive every Compound day from its own bet log: stakes <= day-start cash, cash never negative,
    day end = start + sum(pnl), and the vectorised engine here gives the same bank."""
    report = []
    for basis in BASES:
        for siz in sizers + ("frozen_sizing",):
            plan = Plan(seasons, "frozen_union", SizeFrozen() if siz == "frozen_sizing" else siz)
            res = Compound(rows_by_season, seasons, plan, basis)
            by_day = {}
            for b in res["bets"]:
                by_day.setdefault(b["date"], []).append(b)
            worst_exposure, worst_gap, negative = 0.0, 0.0, False
            for d in res["daily"]:
                bets = by_day.get(d["date"], [])
                exposure = sum(b["stake_frac"] for b in bets)
                worst_exposure = max(worst_exposure, exposure)
                worst_gap = max(worst_gap, abs(d["bank_start"] + sum(b["pnl"] for b in bets) - d["equity"]))
                negative |= d["equity"] < 0
            cfg = REFERENCES[0] if siz == "frozen_sizing" else (
                Cfg(siz, "audit", "kelly", 1.0, cap=1.0) if siz == "kelly1_cap100" else
                Cfg(siz, "audit", "flat", float(siz.replace("flat", "").replace("pct", "")) / 100))
            pack = Pack(BuildDays(rows_by_season, seasons, basis))
            mine = Simulate(pack, np.arange(pack["n"])[None, :], cfg, stop=False, absorb_ruin=False)["end"][0]
            report.append({"basis": basis, "sizer": siz, "compound_end": round(res["end"], 2), "vectorised_end": round(float(mine), 2),
                           "max_day_exposure": round(worst_exposure, 5), "max_ledger_gap": round(worst_gap, 4),
                           "negative_cash": negative, "bets": len(res["bets"])})
    return report


# ----------------------------------------------------------------------------- nested
def Nested(rows_by_season, eval_seasons, train_from, basis, grid=None, n_sims=N_SIMS, T=None):
    grid = grid or GRID
    eval_pack_days = BuildDays(rows_by_season, eval_seasons, basis)
    T = T or len(eval_pack_days)
    bank, eq, dd, peak, choices, hit, hit_date, ruin, t = START, [], [], START, [], False, None, False, 0
    for s in eval_seasons:
        train = BuildDays(rows_by_season, list(range(train_from, s)), basis)
        tp = Pack(train)
        idx = CommonIndex(tp["n"], T, n_sims)
        scored = []
        for cfg in grid:
            r = Simulate(tp, idx, cfg, stop=True, T=T, absorb_ruin=True)
            scored.append((float(r["hit"].mean()), float(np.median(r["end"])), cfg["name"], cfg))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        p, med, name, cfg = scored[0]
        choices.append({"season": s, "train": [train_from, s - 1], "train_days": tp["n"], "choice": name,
                        "train_p_hit_before_ruin": round(p, 4), "train_median_end": round(med, 2)})
        days = [d for d in eval_pack_days if d["season"] == s]
        sp = Pack(days)
        for j in range(sp["n"]):
            if hit or ruin:   # both barriers absorbing: hold cash
                f_r = 0.0
            else:
                f = Fractions(cfg, np.array([bank]), t, T, sp["K"][j:j + 1], sp["M"][j:j + 1], sp["TIER"][j:j + 1])
                f_r = float((f * sp["R"][j:j + 1]).sum())
            bank = max(bank * (1 + f_r), 0.0)
            if bank >= TARGET and not hit:
                hit, hit_date = True, sp["dates"][j]
            ruin |= bank <= RUIN and not hit
            peak = max(peak, bank)
            eq.append({"date": sp["dates"][j], "equity": round(bank, 2), "drawdown_pct": round(bank / peak - 1, 4)})
            t += 1
    return {"end": round(bank, 2), "hit": hit, "hit_date": hit_date, "ruin": ruin,
            "max_dd": min((e["drawdown_pct"] for e in eq), default=0.0), "choices": choices, "equity": eq}


# ----------------------------------------------------------------------------- report
def Run(rows_by_season=None, eval_seasons=None, train_from=TRAIN_FROM, out_dir=None, base=None, write=True, n_sims=N_SIMS):
    eval_seasons = eval_seasons or EVAL_SEASONS
    if rows_by_season is None:
        rows_by_season = AllRows(2012, max(eval_seasons))
        AttachModels(rows_by_season)
    out = {"run_id": NewRunId(), "code": CodeVersion(), "eval_seasons": eval_seasons, "train_from": train_from,
           "start": START, "target": TARGET, "ruin": RUIN, "zero": ZERO, "n_sims": n_sims, "seed": SEED,
           "settings_this_module": len(GRID), "settings_prior": PRIOR_SETTINGS,
           "settings_total_lower_bound": len(GRID) + sum(PRIOR_SETTINGS.values()),
           "audit_compound": AuditCompound(rows_by_season, eval_seasons), "bases": {}}
    tidy, frontier_rows = [], []
    for basis in BASES:
        days = BuildDays(rows_by_season, eval_seasons, basis)
        pack = Pack(days)
        T = pack["n"]
        idx = CommonIndex(T, T, n_sims)
        early = Pack([d for d in days if d["season"] in eval_seasons[:2]])
        late = Pack([d for d in days if d["season"] in eval_seasons[2:]])
        idx_e, idx_l = CommonIndex(early["n"], T, n_sims), CommonIndex(late["n"], T, n_sims)
        rows = []
        for cfg in GRID + REFERENCES:
            stop = Simulate(pack, idx, cfg, stop=True, absorb_ruin=True)
            ever = Simulate(pack, idx, cfg, stop=True, absorb_ruin=False)
            play = Simulate(pack, idx, cfg, stop=False, absorb_ruin=False)
            hist = Historical(pack, cfg)
            hist_ever = Historical(pack, cfg, absorb_ruin=False)
            hist_play = Historical(pack, cfg, stop=False, absorb_ruin=False)
            row = {"name": cfg["name"], "family": cfg["family"], "config": cfg, **BootStats(stop, n_sims, ever),
                   "p_terminal_1m_playon": round(float((play["end"] >= TARGET).mean()), 4),
                   "median_end_playon": round(float(np.median(play["end"])), 2),
                   "p_hit_before_ruin_if_only_2023_24_days": round(float(Simulate(early, idx_e, cfg, T=T)["hit"].mean()), 4),
                   "p_hit_before_ruin_if_only_2025_26_days": round(float(Simulate(late, idx_l, cfg, T=T)["hit"].mean()), 4),
                   "p_hit_ever_if_only_2023_24_days": round(float(Simulate(early, idx_e, cfg, T=T, absorb_ruin=False)["hit"].mean()), 4),
                   "p_hit_ever_if_only_2025_26_days": round(float(Simulate(late, idx_l, cfg, T=T, absorb_ruin=False)["hit"].mean()), 4),
                   "hist": {k: v for k, v in hist.items() if k not in ("equity", "drawdown")},
                   "hist_ever_policy_end": hist_ever["end"], "hist_playon_end": hist_play["end"]}
            rows.append((row, hist))
        front = set(Frontier([r for r, _ in rows if r["family"] != "reference"]))
        for r, _ in rows:
            r["on_frontier"] = r["name"] in front
        nested = Nested(rows_by_season, eval_seasons, train_from, basis, n_sims=n_sims, T=T)
        ranked = sorted([r for r, _ in rows if r["family"] != "reference"], key=lambda r: (-r["p_hit_before_ruin"], r["p_ruin"]))
        out["bases"][basis] = {"days": T, "bet_days": int((pack["M"].sum(axis=1) > 0).sum()), "bets": int(pack["M"].sum()),
                               "top_target_hit": ranked[0]["name"], "frontier": sorted(front),
                               "nested": {k: v for k, v in nested.items() if k != "equity"},
                               "rows": [r for r, _ in rows]}
        last_date = pack["dates"][-1]
        for r, hist in rows:
            if r["on_frontier"] or r["family"] == "reference" or r["name"] == ranked[0]["name"]:
                for d, e, x in zip(pack["dates"], hist["equity"], hist["drawdown"]):
                    tidy.append({"date": d, "series": r["name"], "equity": e, "drawdown_pct": x, "basis": basis})
            frontier_rows.append({"date": last_date, "series": r["name"], "basis": basis, "family": r["family"],
                                  "p_hit_before_ruin": r["p_hit_before_ruin"], "p_ruin": r["p_ruin"],
                                  "p_hit_ever": r["p_hit_ever"], "p_zero": r["p_zero"],
                                  "median_end": r["median_end"], "p5_end": r["p5_end"], "p95_end": r["p95_end"],
                                  "hist_end": r["hist"]["end"], "hist_max_dd": r["hist"]["max_dd"],
                                  "on_frontier": int(r["on_frontier"])})
        for e in nested["equity"]:
            tidy.append({"date": e["date"], "series": "nested_walkforward", "equity": e["equity"],
                         "drawdown_pct": e["drawdown_pct"], "basis": basis})
    if write:
        out_dir = out_dir or DEFAULT_OUT
        base = base or "million_target"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{base}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["date", "series", "equity", "drawdown_pct", "basis"])
            w.writeheader()
            w.writerows(tidy)
        with open(os.path.join(out_dir, f"{base}_frontier.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(frontier_rows[0].keys()))
            w.writeheader()
            w.writerows(frontier_rows)
        with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    return out


def Main():
    ap = argparse.ArgumentParser(description="P(hit $1M) vs P(ruin) sizing frontier on one continuous $10k bank.")
    ap.add_argument("--out-dir")
    ap.add_argument("--base")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    a = ap.parse_args()
    out = Run(out_dir=a.out_dir, base=a.base, n_sims=a.sims)
    print(json.dumps(out["audit_compound"], indent=0))
    for basis, b in out["bases"].items():
        print(f"== {basis}: days {b['days']} bet_days {b['bet_days']} bets {b['bets']} top {b['top_target_hit']}")
        for r in sorted(b["rows"], key=lambda r: -r["p_hit_before_ruin"]):
            h = r["hist"]
            print(f"{'*' if r['on_frontier'] else ' '} {r['name']:30s} Pfirst {r['p_hit_before_ruin']:.4f}+-{r['p_hit_before_ruin_mc_se']:.4f}"
                  f" Pever {r['p_hit_ever']:.4f} Pruin {r['p_ruin']:.4f}"
                  f" Pzero {r['p_zero']:.4f} med {r['median_end']:>12,.0f} p5 {r['p5_end']:>10,.0f} p95 {r['p95_end']:>12,.0f}"
                  f" | hist end {h['end']:>12,.0f} dd {h['max_dd']:.3f} hit {h['hit']} | playon P1M {r['p_terminal_1m_playon']:.4f}"
                  f" | early {r['p_hit_before_ruin_if_only_2023_24_days']:.4f} late {r['p_hit_before_ruin_if_only_2025_26_days']:.4f}")
        print("nested:", json.dumps(b["nested"]))


if __name__ == "__main__":
    Main()