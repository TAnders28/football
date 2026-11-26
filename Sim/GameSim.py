r"""Vectorised drive-level Monte Carlo of NFL games (research, tested:false).

SimulateGames(probs, base, n, rng, playoff=False) -> (home_pts, away_pts) arrays of length n.
Every simulated game: coin-toss opening receiver (other team receives the 2nd half), kickoff start from the
empirical kickoff pool, then drive after drive: result ~ probs[offense side, state, fp bin], duration from the
(state, result) pool, points from the result pool, next start (and whether the same team keeps the ball) from the
(result, fp bin) pool. A drive longer than the clock left ends the half; a scoring result keeps its points with
probability clock_left / duration. Overtime: 10:00 (15:00 playoffs), each team gets a possession unless the first one
is a touchdown, then next score wins; regular-season games can end tied; playoff games repeat overtime.
Synthetic games have no market prices: never real outcomes, never proof of an edge.
"""
from __future__ import annotations

import numpy as np

from Sim.DriveModel import CAT_ID, LATE_H1, LATE_H2, FpBin, N_CAT, N_FP, StateOf

EOH = CAT_ID["EOH"]
TD = CAT_ID["TD"]
SCORING = np.zeros(N_CAT, dtype=bool)
SCORING[[CAT_ID["TD"], CAT_ID["FG"], CAT_ID["OPPTD"], CAT_ID["SAFETY"]]] = True
MAX_STEPS = 80


def SimulateGames(probs, base, n, rng, playoff=False):
    cum = np.cumsum(probs, axis=3)
    cum[..., -1] = 1.0
    pts = np.zeros((n, 2))                       # [:,0] home, [:,1] away
    first = (rng.random(n) < 0.5).astype(int)     # side receiving the opening kickoff
    poss = first.copy()
    fp = base.kick.Sample(np.zeros(n, dtype=int), rng)[:, 0]
    half = np.ones(n, dtype=int)
    hs = np.full(n, 1800.0)
    done = np.zeros(n, dtype=bool)
    ot_poss = np.zeros((n, 2), dtype=int)
    ot_first_td = np.zeros(n, dtype=bool)
    rows = np.arange(n)
    for _ in range(MAX_STEPS):
        a = ~done
        if not a.any():
            break
        idx = rows[a]
        p = poss[idx]
        sd = pts[idx, p] - pts[idx, 1 - p]
        st = StateOf(half[idx], hs[idx], sd)
        fb = FpBin(fp[idx])
        c = cum[p, st, fb]
        u = rng.random(len(idx))
        cat = (u[:, None] > c).sum(axis=1)
        cat = np.minimum(cat, N_CAT - 1)
        hb = np.minimum((hs[idx] // 30).astype(int), 11)
        lt = np.where(sd > 0, 0, np.where(sd == 0, 1, 2))
        hh = np.clip(half[idx] - 1, 0, 1)
        lim = np.where(half[idx] == 1, LATE_H1, LATE_H2)
        p_eoh = np.where((half[idx] >= 3) | (hs[idx] >= lim), 0.0, base.eoh[hh, hb, lt])
        cat = np.where(rng.random(len(idx)) < p_eoh, EOH, cat)
        dur = base.dur.Sample(st * N_CAT + cat, rng)[:, 0]
        left = hs[idx]
        over = dur > left
        keep = rng.random(len(idx)) < np.where(over, left / np.maximum(dur, 1.0), 1.0)
        cat = np.where(over & ~(SCORING[cat] & keep), EOH, cat)
        dur = np.where(over, left, dur)
        pp = base.pts.Sample(cat, rng)
        pts[idx, p] += pp[:, 0]
        pts[idx, 1 - p] += pp[:, 1]
        hs[idx] = left - dur
        nx = base.nxt.Sample(cat * N_FP + fb, rng)
        in_ot = half[idx] == 3
        if in_ot.any():
            oi = idx[in_ot]
            ot_poss[oi, p[in_ot]] += 1
            ot_first_td[oi] |= (ot_poss[oi].sum(axis=1) == 1) & (cat[in_ot] == TD)
        new_fp = nx[:, 0]
        same = nx[:, 1] > 0.5
        poss[idx] = np.where(same, p, 1 - p)
        fp[idx] = new_fp
        # half transitions
        end = (hs[idx] <= 0) | (cat == EOH)
        if end.any():
            e = idx[end]
            h_before = half[e].copy()
            e1 = e[h_before == 1]
            half[e1], hs[e1], poss[e1] = 2, 1800.0, 1 - first[e1]
            fp[e1] = base.kick.Sample(np.zeros(len(e1), dtype=int), rng)[:, 0]
            e2 = e[h_before == 2]
            tied = pts[e2, 0] == pts[e2, 1]
            done[e2[~tied]] = True
            t2 = e2[tied]
            half[t2], hs[t2] = 3, (900.0 if playoff else 600.0)
            poss[t2] = (rng.random(len(t2)) < 0.5).astype(int)
            fp[t2] = base.kick.Sample(np.zeros(len(t2), dtype=int), rng)[:, 0]
            e3 = e[h_before == 3]
            if len(e3):
                if playoff:
                    tied3 = pts[e3, 0] == pts[e3, 1]
                    done[e3[~tied3]] = True
                    hs[e3[tied3]] = 900.0
                else:
                    done[e3] = True
        # overtime decisions
        ot = idx[(half[idx] == 3) & ~done[idx]]
        if len(ot):
            both = (ot_poss[ot] >= 1).all(axis=1)
            diff = pts[ot, 0] != pts[ot, 1]
            done[ot[diff & (both | ot_first_td[ot])]] = True
    return pts[:, 0], pts[:, 1]


def Summarise(home, away, actual_home=None, actual_away=None):
    m = home - away
    t = home + away
    out = {"p_home": float((m > 0).mean() + 0.5 * (m == 0).mean()), "p_tie": float((m == 0).mean()),
           "margin_mean": float(m.mean()), "margin_sd": float(m.std()), "total_mean": float(t.mean()),
           "total_sd": float(t.std()), "p_key3": float((np.abs(m) == 3).mean()), "p_key7": float((np.abs(m) == 7).mean()),
           "p_onescore": float((np.abs(m) <= 8).mean())}
    if actual_home is not None:
        am, at = actual_home - actual_away, actual_home + actual_away
        out["pit_margin"] = float((m < am).mean() + 0.5 * (m == am).mean())
        out["pit_total"] = float((t < at).mean() + 0.5 * (t == at).mean())
    return out
