r"""
StakingEnv: leak-free, day-by-day staking environment for LEARNED sizing policies (Tim, 2026-11-13).
Simulation only: no wagers, not wired into PredictDay, FrozenConfig untouched.

One step = one ET game day whose games are all finished (MartingaleParlay.BuildDays, so the day list is
identical to the martingale study). Before the day the policy sees ONLY pre-game information:
  * per game: the side with the larger model edge (walk-forward p_full2, fit on seasons < S), its posted
    moneyline, de-vigged market prob, payout on the chosen pricing basis, fee-adjusted edge and Kelly fraction;
  * day context: slate size, week, playoff flag; path context: bankroll progress toward $1M, drawdown from
    peak and an EMA of recent daily log returns (all from days already settled).
Action (vectorised over paths): singles stake f_i = min(lambda(s) * kelly_i, PER_BET_CAP) on candidates with
edge > tau, plus at most one parlay of n in {2,3,4} legs (top-n candidates by model EV, positive-EV legs
only) staked at g(s) of the bank; lambda = 0 and g = 0 is "abstain / preserve bankroll". Total day exposure
is scaled to <= 100% of day-start cash (no reuse of unsettled money).
Settlement: day end. Ties void (single returns 0; parlay drops the leg; all-void parlay pushes).
Pricing bases: "closing_ml" (posted closing moneyline, vig included) or "tim_fee_est" (each $1 contract
bought at de-vigged prob + $0.02 -- Tim's working fee ESTIMATE, not a verified exchange fee).
Absorbing: bank >= $1,000,000 (target) or bank <= $1,000 (ruin), cash held afterwards (first passage).
Synthetic mode redraws each candidate's outcome from a calibrated probability (logit fit on TRAINING days
only), so a policy cannot simply memorise which historical days won.
"""

import math

import numpy as np

from Backtest.Lab import Decimal, FitLogit
from Backtest.MartingaleParlay import BuildDays as FinishedDays

START = 10_000.0
TARGET = 1_000_000.0
RUIN = 1_000.0
FEE_EST = 0.02
MAX_C = 16
LEGS = (2, 3, 4)
MAX_LEGS = max(LEGS)
PER_BET_CAP = 0.25
LMAX = 2.0
GMAX = 0.25
TAU_MAX = 0.10
G_MIN = 0.002
BASES = ("closing_ml", "tim_fee_est")
FEATS = ["bias", "bank_progress", "drawdown", "recent_ema", "n_pos_edge", "max_edge", "sum_kelly", "week",
         "playoff", "slate_size"]
NF = len(FEATS)
N_PARAM = NF + 1 + NF + len(LEGS) * NF


def Sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def Logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def Payout(dec, p_mkt, basis):
    return dec - 1 if basis == "closing_ml" else 1.0 / min(p_mkt + FEE_EST, 0.999) - 1


def Candidate(r, model="p_full2"):
    """Pre-game: the side with the larger model edge vs de-vigged market. No result fields."""
    p, ph = r.get(model, r["mkt_home"]), r["mkt_home"]
    sides = [{"side": "home", "team": r["home_team"], "dec": Decimal(r["home_ml"]), "p_model": p, "p_mkt": ph},
             {"side": "away", "team": r["away_team"], "dec": Decimal(r["away_ml"]), "p_model": 1 - p, "p_mkt": 1 - ph}]
    return max(sides, key=lambda s: s["p_model"] - s["p_mkt"])


def Outcome(r, side):
    """+1 win, -1 loss, 0 void (tie). Settlement only; never read before the day is decided."""
    if r.get("result") == 0 or r.get("home_win") is None:
        return 0
    return 1 if (r["home_win"] == 1) == (side == "home") else -1


def BuildDays(rows_by_season, seasons, model="p_full2"):
    out = []
    for d in FinishedDays(rows_by_season, seasons):
        rows = [r for r in sorted(d["rows"], key=lambda r: r["game_id"]) if r.get("home_ml") is not None and r.get("away_ml") is not None]
        cands = [dict(Candidate(r, model), game_id=r["game_id"]) for r in rows][:MAX_C]
        res = [Outcome(r, c["side"]) for r, c in zip(rows, cands)]
        out.append({"date": d["date"], "season": d["season"], "week": max([r.get("week") or 0 for r in d["rows"]] + [0]),
                    "playoff": int(any((r.get("game_type") or "REG") != "REG" for r in d["rows"])),
                    "n_games": len(d["rows"]), "cands": cands, "res": res})
    return out


def Pack(days, basis):
    D = len(days)
    z = lambda: np.zeros((D, MAX_C))
    P = {"K": z(), "E": z(), "B": z(), "W": z(), "M": z(), "PM": z(), "PK": z(),
         "PH": np.zeros((D, len(LEGS)), bool), "PL": np.zeros((D, len(LEGS), MAX_LEGS), int),
         "PLM": np.zeros((D, len(LEGS), MAX_LEGS)), "CTX": np.zeros((D, 3)), "basis": basis,
         "dates": [d["date"] for d in days], "seasons": np.array([d["season"] for d in days]), "n": D}
    for i, d in enumerate(days):
        ev = []
        for j, (c, w) in enumerate(zip(d["cands"], d["res"])):
            b = Payout(c["dec"], c["p_mkt"], basis)
            p = c["p_model"]
            P["B"][i, j], P["M"][i, j], P["W"][i, j] = b, 1.0, w
            P["E"][i, j] = p - 1.0 / (1.0 + b)
            P["K"][i, j] = max(0.0, (b * p - (1 - p)) / b)
            P["PM"][i, j], P["PK"][i, j] = p, c["p_mkt"]
            if p * (1 + b) > 1:
                ev.append((math.log(p * (1 + b)), j))
        ev.sort(reverse=True)
        for k, n in enumerate(LEGS):
            if len(ev) >= n:
                P["PH"][i, k] = True
                P["PL"][i, k, :n] = [j for _, j in ev[:n]]
                P["PLM"][i, k, :n] = 1.0
        P["CTX"][i] = [min(d["week"], 22) / 22.0, d["playoff"], d["n_games"] / 16.0]
    return P


# ----------------------------------------------------------------------------- calibration (training days only)
def CalibFit(pack, mask):
    X, y = [], []
    for i in np.flatnonzero(mask):
        for j in range(MAX_C):
            if pack["M"][i, j] and pack["W"][i, j] != 0:
                lm, lp = Logit(pack["PK"][i, j]), Logit(pack["PM"][i, j])
                X.append([1.0, lm, lp - lm])
                y.append(1 if pack["W"][i, j] > 0 else 0)
    return FitLogit(X, y, l2=1.0)


def PCal(pack, w):
    lm = np.log(np.clip(pack["PK"], 1e-6, 1 - 1e-6) / (1 - np.clip(pack["PK"], 1e-6, 1 - 1e-6)))
    lp = np.log(np.clip(pack["PM"], 1e-6, 1 - 1e-6) / (1 - np.clip(pack["PM"], 1e-6, 1 - 1e-6)))
    return Sig(w[0] + w[1] * lm + w[2] * (lp - lm)) * pack["M"]


# ----------------------------------------------------------------------------- policy parameters
def Split(theta):
    return theta[:, :NF], theta[:, NF], theta[:, NF + 1:2 * NF + 1], theta[:, 2 * NF + 1:].reshape(-1, len(LEGS), NF)


def FixedTheta(lam=0.0, tau=0.0, n=None):
    """Constant policy: singles at lam x Kelly on edge > tau, no parlays (baselines). lam=0 -> never bets."""
    t = np.zeros(N_PARAM)
    t[0] = Logit(lam / LMAX) if lam > 0 else -40.0
    t[NF] = Logit(tau / TAU_MAX) if tau > 0 else -40.0
    t[NF + 1] = -40.0
    return t[None, :].repeat(n or 1, 0)


def InitTheta():
    t = np.zeros(N_PARAM)
    t[0] = Logit(0.25 / LMAX)
    t[NF] = Logit(0.02 / TAU_MAX)
    t[NF + 1] = Logit(0.005 / GMAX)
    return t


# ----------------------------------------------------------------------------- engine (vectorised over paths)
def Features(pack, d, bank, peak, ema, fs_K, fs_E):
    pos = (fs_E > 0) * pack["M"][d]
    ctx = pack["CTX"][d]
    return np.column_stack([np.ones(len(d)), np.log(np.maximum(bank, 1.0) / START) / math.log(TARGET / START),
                            1 - bank / np.maximum(peak, 1e-9), np.clip(ema * 10, -3, 3), pos.sum(1) / 8.0,
                            np.clip((fs_E * pack["M"][d]).max(1) * 10, -2, 3), np.clip((fs_K * pos).sum(1), 0, 2),
                            ctx[:, 0], ctx[:, 1], ctx[:, 2]])


def Simulate(pack, idx, theta, synth=None, pcal=None, u=None, state=None, record=False):
    """idx (N, steps) day indices; theta (N, N_PARAM). synth: bool (N,) rows whose outcomes are redrawn from
    pcal using uniforms u(j) -> (N, MAX_C) (common random numbers). state: dict(bank, peak, ema) to continue a
    bankroll across calls. Returns per-path outcomes."""
    N, steps = idx.shape
    tl, tt, tg, tn = Split(theta)
    st = state or {}
    bank = np.array(st.get("bank", np.full(N, START)), float).reshape(N)
    peak = np.array(st.get("peak", bank), float).reshape(N)
    ema = np.array(st.get("ema", np.zeros(N)), float).reshape(N)
    hit = np.array(st.get("hit", np.zeros(N, bool))).reshape(N)
    ruin = np.array(st.get("ruin", np.zeros(N, bool))).reshape(N)
    maxdd = np.zeros(N)
    expo, nbet_days, par_days = np.zeros(N), np.zeros(N), np.zeros(N)
    tau = TAU_MAX * Sig(tt)
    rows = np.arange(N)
    rec = []
    for j in range(steps):
        d = idx[:, j]
        K, E, B, M = pack["K"][d], pack["E"][d], pack["B"][d], pack["M"][d]
        W = pack["W"][d]
        if synth is not None and synth.any():
            Ws = np.where(u(j) < pcal[d], 1.0, -1.0) * M
            W = np.where(synth[:, None], Ws, W)
        phi = Features(pack, d, bank, peak, ema, K, E)
        lam = LMAX * Sig((tl * phi).sum(1))
        lam = np.where(lam < 1e-4, 0.0, lam)
        fs = np.minimum(lam[:, None] * K, PER_BET_CAP) * (E > tau[:, None]) * M
        has = pack["PH"][d]
        sc = np.where(has, np.einsum("nkf,nf->nk", tn, phi), -np.inf)
        ch = np.argmax(sc, 1)
        g = GMAX * Sig((tg * phi).sum(1)) * has.any(1)
        g = np.where(g < G_MIN, 0.0, g)
        L, LM = pack["PL"][d, ch], pack["PLM"][d, ch]
        lw, lb = np.take_along_axis(W, L, 1), np.take_along_axis(B, L, 1)
        lost = ((lw < 0) & (LM > 0)).any(1)
        mult = np.prod(np.where((lw > 0) & (LM > 0), 1 + lb, 1.0), 1)
        pret = np.where(lost, -1.0, mult - 1)
        live = ~hit & ~ruin
        tot = fs.sum(1) + g
        sc_ = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0) * live
        fs, g = fs * sc_[:, None], g * sc_
        sret = np.where(W > 0, B, np.where(W < 0, -1.0, 0.0))
        r = (fs * sret).sum(1) + g * pret
        new = np.maximum(bank * (1 + r), 0.0)
        ema = np.where(live, 0.75 * ema + 0.25 * np.log(np.maximum(new, 1.0) / np.maximum(bank, 1.0)), ema)
        bank = new
        expo += fs.sum(1) + g
        nbet_days += (fs.sum(1) + g) > 0
        par_days += g > 0
        new_hit = live & (bank >= TARGET)
        hit |= new_hit
        ruin |= live & ~new_hit & (bank <= RUIN)
        peak = np.maximum(peak, bank)
        maxdd = np.maximum(maxdd, 1 - bank / peak)
        if record:
            rec.append({"bank": bank.copy(), "dd": 1 - bank / peak, "lam": lam * live, "g": g, "legs": np.array(LEGS)[ch] * (g > 0),
                        "exposure": fs.sum(1) + g, "n_singles": (fs > 0).sum(1)})
    out = {"end": bank, "hit": hit, "ruin": ruin, "maxdd": maxdd, "peak": peak, "ema": ema,
           "mean_exposure": expo / max(steps, 1), "bet_day_share": nbet_days / max(steps, 1), "parlay_day_share": par_days / max(steps, 1)}
    if record:
        out["path"] = rec
    return out


def Objective(res, kind):
    """growth: mean capped log wealth (Kelly-style, upside capped at $1M). million: P(first hit $1M before
    ruin) + 0.05 x normalised capped log wealth as a tie-break (Tim's goal; accepts more ruin)."""
    lw = np.log(np.clip(res["end"], 1.0, TARGET) / START)
    if kind == "growth":
        return lw
    return res["hit"].astype(float) + 0.05 * lw / math.log(TARGET / START)


# ----------------------------------------------------------------------------- randomized day orders (PathSim)
def RandomIdx(n_days, paths, steps, method="block_bootstrap", seed=20261113, block=4):
    """(paths, steps) day-index array for Simulate(pack, idx, ...): random start dates, block-bootstrapped or shuffled
    game-day order (Backtest.PathSim.DayOrders). Seeded. For restarts after ruin / a bad start, call Simulate in
    segments with `state` or score daily returns with Backtest.PathSim.RunPaths(restart=True)."""
    from Backtest.PathSim import DayOrders
    return DayOrders(n_days, paths, steps, method, np.random.default_rng(seed), block)