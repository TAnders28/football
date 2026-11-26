r"""'Slightly different versions' of a real NFL game (research, tested:false).

Keep the game's real drive sequence (who had the ball, where each drive started, game state) and re-roll each
drive's result with probability `eps`:
  source="model": from the pre-game drive model (team factors known before kickoff), points from the result pool;
  source="boot":  from the same team's other drives in this same game (no model input at all).
Field-position and clock knock-on effects of a changed drive are ignored (documented limit).

NearMiss(...) -> P(home wins) over n versions (ties count half). The real result is eps=0.
These versions are synthetic: no market prices, never real outcomes, never proof of an edge.
"""
from __future__ import annotations

import numpy as np

from Sim.DriveModel import CAT_ID, N_CAT

EOH = CAT_ID["EOH"]


def NearMiss(side, st, fb, cat, off_pts, def_pts, probs, base, n, eps, rng, source="model"):
    side = np.asarray(side, dtype=int)
    D = len(side)
    off_pts = np.asarray(off_pts, dtype=float)
    def_pts = np.asarray(def_pts, dtype=float)
    cat = np.asarray(cat, dtype=int)
    reroll = (rng.random((n, D)) < eps) & (cat != EOH)[None, :]
    new_off = np.broadcast_to(off_pts, (n, D)).copy()
    new_def = np.broadcast_to(def_pts, (n, D)).copy()
    if source == "model":
        cum = np.cumsum(probs, axis=3)
        cum[..., -1] = 1.0
        c = cum[side, st, fb].copy()                      # [D, cat]
        # never draw EOH as a re-rolled result
        p = np.diff(np.c_[np.zeros(D), c], axis=1)
        p[:, EOH] = 0.0
        c = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
        u = rng.random((n, D))
        new_cat = np.minimum((u[:, :, None] > c[None, :, :]).sum(axis=2), N_CAT - 1)
        pp = base.pts.Sample(new_cat.ravel(), rng).reshape(n, D, 2)
        new_off = np.where(reroll, pp[..., 0], new_off)
        new_def = np.where(reroll, pp[..., 1], new_def)
    elif source == "boot":
        for s in (0, 1):
            own = np.where((side == s) & (cat != EOH))[0]
            if len(own) == 0:
                continue
            pick = own[(rng.random((n, D)) * len(own)).astype(int)]
            m = reroll & (side == s)[None, :]
            new_off = np.where(m, off_pts[pick], new_off)
            new_def = np.where(m, def_pts[pick], new_def)
    else:
        raise ValueError(source)
    home = np.where(side == 0, new_off, new_def).sum(axis=1)
    away = np.where(side == 1, new_off, new_def).sum(axis=1)
    m = home - away
    return float((m > 0).mean() + 0.5 * (m == 0).mean())
