r"""Market-anchored NFL resimulation (research, tested:false).

The drive model's own team strengths lose to the market (see Sim.ReportNfl), so for 'what else could have happened'
work the simulator takes its strength and scoring level from the market instead: league-average drive tables, then
the home offense's TD and FG probabilities are multiplied by k*(1+s) and the away offense's by k/(1+s), with (s, k)
solved so that the simulated P(home wins) matches a target price and the simulated mean total matches the closing
total. The simulator then only supplies the shape (margin, total, key numbers, near-misses), never the price.
Synthetic games: no market prices of their own, never real outcomes, never proof of an edge.
"""
from __future__ import annotations

import numpy as np

from Sim.DriveModel import CAT_ID, N_CAT
from Sim.GameSim import SimulateGames

TD, FG = CAT_ID["TD"], CAT_ID["FG"]


def AnchoredProbs(base, s, k):
    out = np.empty((2,) + base.p0.shape)
    for side, mult in ((0, k * (1 + s)), (1, k / (1 + s))):
        m = np.ones(N_CAT)
        m[TD] = m[FG] = mult
        hf = base.home if side == 0 else np.ones(N_CAT)
        p = base.p0 * m * hf
        out[side] = p / p.sum(axis=2, keepdims=True)
    return out


def _Eval(base, s, k, n, seed, playoff):
    h, a = SimulateGames(AnchoredProbs(base, s, k), base, n, np.random.default_rng(seed), playoff=playoff)
    m = h - a
    return float((m > 0).mean() + 0.5 * (m == 0).mean()), float((h + a).mean()), h, a


def Anchor(base, p_target, total_target, n=3000, seed=1, playoff=False, rounds=3):
    """Solve (s, k) by alternating bisection with common random numbers. Returns (s, k)."""
    s, k = 0.0, 1.0
    p_target = float(np.clip(p_target, 0.02, 0.98))
    for _ in range(rounds):
        lo, hi = -0.85, 6.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            p, _, _, _ = _Eval(base, mid, k, n, seed, playoff)
            lo, hi = (mid, hi) if p < p_target else (lo, mid)
        s = 0.5 * (lo + hi)
        if total_target is None or not np.isfinite(total_target):
            break
        lo, hi = 0.4, 2.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            _, t, _, _ = _Eval(base, s, mid, n, seed, playoff)
            lo, hi = (mid, hi) if t < total_target else (lo, mid)
        k = 0.5 * (lo + hi)
    return s, k
