r"""Drive-outcome model for the NFL resimulator (research, tested:false).

Base table p0[state, fp_bin, cat]: league drive-result frequencies by game state and starting field position,
fitted only on seasons before the season being simulated. States: 0 normal, 1 late & offense leading, 2 late & tied,
late states by offense score diff and overtime (see StateOf). Sparse late cells are shrunk with
SHRINK_LATE pseudo-drives to the normal row tilted by that state's overall result mix.

Team factors (leak-free walk): for each team and adjusted category k, offense factor
  f = (observed_k + n0 * rate_k) / (expected_k + n0 * rate_k)
where expected_k sums p0 over the drives the team actually had (same for defense, and a global home-offense factor).
A matchup's table is p0 * f_off * f_def (* f_home) renormalised within each (state, fp_bin). Counts carry across
seasons multiplied by `carry`.

Empirical pools (also fitted on prior seasons only): drive duration by (state, cat), points (offense, defense)
by cat, next drive start (field position, same-team flag) by (cat, fp_bin), and kickoff starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from Sim.Drives import CATS

N_CAT = len(CATS)
CAT_ID = {c: i for i, c in enumerate(CATS)}
ADJ = [CAT_ID[c] for c in ("TD", "FG", "PUNT", "TO", "DOWNS", "MISSFG", "OPPTD")]
N_FP = 10
N_STATE = 9
SHRINK_LATE = 30.0
LATE_H2, LATE_H1 = 360.0, 120.0


def FpBin(fp):
    fp = np.clip(np.asarray(fp, dtype=float), 1, 99)
    return np.minimum(((fp - 1) // 10).astype(int), N_FP - 1)


def StateOf(half, half_secs, score_diff):
    """0 normal; late (2nd half < 6:00 or 1st half < 2:00) by offense score diff: 1 lead 9+, 2 lead 4-8,
    3 lead 1-3, 4 tied, 5 trail 1-3, 6 trail 4-8, 7 trail 9+; 8 overtime."""
    half = np.asarray(half)
    hs = np.asarray(half_secs, dtype=float)
    sd = np.asarray(score_diff, dtype=float)
    late = ((half == 2) & (hs < LATE_H2)) | ((half == 1) & (hs < LATE_H1))
    st = np.select([sd >= 9, sd >= 4, sd >= 1, sd == 0, sd >= -3, sd >= -8], [1, 2, 3, 4, 5, 6], 7)
    out = np.where(late, st, 0)
    return np.where(half >= 3, 8, out).astype(int)


class Pool:
    """Ragged pools flattened for vectorised sampling: key id -> rows of `vals` (2-D)."""

    def __init__(self, groups: dict, n_keys: int, width: int, fallback=None):
        self.start = np.zeros(n_keys, dtype=np.int64)
        self.length = np.zeros(n_keys, dtype=np.int64)
        chunks = []
        pos = 0
        for k in range(n_keys):
            v = groups.get(k)
            if (v is None or len(v) == 0) and fallback is not None:
                v = fallback
            v = np.zeros((1, width)) if v is None or len(v) == 0 else np.asarray(v, dtype=float).reshape(-1, width)
            chunks.append(v)
            self.start[k], self.length[k] = pos, len(v)
            pos += len(v)
        self.vals = np.concatenate(chunks, axis=0)

    def Sample(self, key, rng):
        key = np.asarray(key, dtype=np.int64)
        idx = self.start[key] + (rng.random(key.shape) * self.length[key]).astype(np.int64)
        return self.vals[idx]


@dataclass
class BaseTables:
    p0: np.ndarray            # [state, fp, cat], EOH column zero (EOH comes from `eoh`)
    eoh: np.ndarray           # [half(0: 1st, 1: 2nd), 30-second bin of clock left (12), 0 lead / 1 tied / 2 trail]
    rate: np.ndarray          # [cat] league per-drive rate (normal state)
    home: np.ndarray          # [cat] home-offense factor
    dur: Pool                 # key state*N_CAT+cat -> [dur]
    pts: Pool                 # key cat -> [off_pts, def_pts]
    nxt: Pool                 # key cat*N_FP+fp -> [next_fp, same_team]
    kick: Pool                # key 0 -> [fp]
    n_drives: int = 0
    seasons: tuple = field(default_factory=tuple)


def DriveArrays(d: pl.DataFrame):
    st = StateOf(d["half"].to_numpy(), d["half_secs"].to_numpy(), d["score_diff"].to_numpy())
    fb = FpBin(d["start_fp"].to_numpy())
    cat = np.array([CAT_ID[c] for c in d["cat"].to_list()], dtype=int)
    return st, fb, cat


def FitBase(d: pl.DataFrame) -> BaseTables:
    d = d.sort(["game_id", "drive_idx"])
    st, fb, cat = DriveArrays(d)
    cnt = np.zeros((N_STATE, N_FP, N_CAT))
    np.add.at(cnt, (st, fb, cat), 1.0)
    normal = cnt[0] + 0.5
    p_norm = normal / normal.sum(axis=1, keepdims=True)
    p0 = np.empty_like(cnt)
    p0[0] = p_norm
    for s in range(1, N_STATE):
        # late states: prior row = normal row tilted by the state's own overall result mix (obs / expected
        # from the normal rows over the same fp bins), then shrunk with SHRINK_LATE pseudo-drives per cell
        n_s = cnt[s].sum(axis=1)
        exp_s = (p_norm * n_s[:, None]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (cnt[s].sum(axis=0) + 5.0 * exp_s / max(exp_s.sum(), 1.0)) / (exp_s + 5.0 * exp_s / max(exp_s.sum(), 1.0))
        ratio[~np.isfinite(ratio)] = 1.0
        q = p_norm * ratio
        q /= q.sum(axis=1, keepdims=True)
        p0[s] = (cnt[s] + SHRINK_LATE * q) / (n_s[:, None] + SHRINK_LATE)
    # End of half is clock-driven: P(EOH | half, clock left, lead/tie/trail) from late drives, then the
    # other results come from p0 with the EOH column removed.
    eoh = np.zeros((2, 12, 3))
    hv = d["half"].to_numpy()
    hsv = d["half_secs"].to_numpy()
    sdv = d["score_diff"].to_numpy()
    lt = np.where(sdv > 0, 0, np.where(sdv == 0, 1, 2))
    is_eoh = cat == CAT_ID["EOH"]
    for hi, h in enumerate((1, 2)):
        lim = LATE_H1 if h == 1 else LATE_H2
        for b in range(12):
            lo_s, hi_s = b * 30.0, (b + 1) * 30.0
            for l in range(3):
                m = (hv == h) & (hsv >= lo_s) & (hsv < hi_s) & (lt == l) & (hsv < lim)
                eoh[hi, b, l] = (is_eoh[m].sum() + 0.5 * 0.3) / (m.sum() + 0.5) if m.sum() else 0.0
    p0[..., CAT_ID["EOH"]] = 0.0
    p0 /= p0.sum(axis=2, keepdims=True)
    rate = cnt[0].sum(axis=0) / cnt[0].sum()
    # home-offense factor (global)
    is_home = d["is_home_off"].to_numpy() == 1
    p0e = p0.copy()
    p0e[..., CAT_ID["EOH"]] = 0.0
    p0e /= p0e.sum(axis=2, keepdims=True)
    ne = cat != CAT_ID["EOH"]
    is_home = is_home & ne
    exp_h = p0e[st[is_home], fb[is_home]].sum(axis=0)
    obs_h = np.bincount(cat[is_home], minlength=N_CAT).astype(float)
    is_away = (d["is_home_off"].to_numpy() == 0) & ne
    exp_a = p0e[st[is_away], fb[is_away]].sum(axis=0)
    obs_a = np.bincount(cat[is_away], minlength=N_CAT).astype(float)
    home = np.ones(N_CAT)
    for k in ADJ:
        rh = (obs_h[k] + 50 * rate[k]) / (exp_h[k] + 50 * rate[k])
        ra = (obs_a[k] + 50 * rate[k]) / (exp_a[k] + 50 * rate[k])
        home[k] = rh / ra
    dur = d["dur"].to_numpy()
    dur_groups = {}
    for key in np.unique(st * N_CAT + cat):
        m = (st * N_CAT + cat) == key
        v = dur[m & np.isfinite(dur)]
        if len(v):
            dur_groups[int(key)] = v
    dur_pool = Pool(dur_groups, N_STATE * N_CAT, 1, fallback=dur[np.isfinite(dur)])
    pts = np.c_[d["off_pts"].to_numpy(), d["def_pts"].to_numpy()]
    pts_pool = Pool({k: pts[cat == k] for k in range(N_CAT)}, N_CAT, 2)
    # next start: consecutive drives in the same game and half
    gid = d["game_id"].to_numpy()
    half = d["half"].to_numpy()
    pos = d["posteam"].to_numpy()
    fp = d["start_fp"].to_numpy()
    same_g = (gid[1:] == gid[:-1]) & (half[1:] == half[:-1])
    nxt_rows = np.c_[fp[1:], (pos[1:] == pos[:-1]).astype(float)][same_g]
    key = (cat[:-1] * N_FP + fb[:-1])[same_g]
    nxt_groups = {}
    for k in np.unique(key):
        v = nxt_rows[key == k]
        if len(v) >= 15:
            nxt_groups[int(k)] = v
    # fallback per category (any fp bin)
    cat_prev = cat[:-1][same_g]
    for c in range(N_CAT):
        v = nxt_rows[cat_prev == c]
        for b in range(N_FP):
            if c * N_FP + b not in nxt_groups and len(v):
                nxt_groups[c * N_FP + b] = v
    nxt_pool = Pool(nxt_groups, N_CAT * N_FP, 2, fallback=nxt_rows)
    after_score = np.isin(cat_prev, [CAT_ID["TD"], CAT_ID["FG"]])
    kick_pool = Pool({0: nxt_rows[after_score][:, :1]}, 1, 1)
    return BaseTables(p0=p0, eoh=eoh, rate=rate, home=home, dur=dur_pool, pts=pts_pool, nxt=nxt_pool, kick=kick_pool,
                      n_drives=d.height, seasons=tuple(sorted(set(d["season"].to_list()))))


class TeamWalk:
    """Leak-free team offense/defense counts. Call Factors() before a date's games, Update() after."""

    def __init__(self, n0=50.0, carry=0.5):
        self.n0, self.carry = n0, carry
        self.off = {}
        self.dfn = {}
        self.season = None

    def NewSeason(self, season):
        if self.season is not None and season != self.season:
            for tab in (self.off, self.dfn):
                for t in tab:
                    tab[t] *= self.carry
        self.season = season

    def _F(self, tab, team, rate):
        v = tab.get(team)
        f = np.ones(N_CAT)
        if v is None:
            return f
        obs, exp = v[0], v[1]
        for k in ADJ:
            f[k] = (obs[k] + self.n0 * rate[k]) / (exp[k] + self.n0 * rate[k])
        return f

    def Factors(self, team, rate):
        return self._F(self.off, team, rate), self._F(self.dfn, team, rate)

    def Update(self, d: pl.DataFrame, base: BaseTables):
        st, fb, cat = DriveArrays(d)
        exp = base.p0[st, fb]
        keep = cat != CAT_ID["EOH"]
        offs = d["posteam"].to_list()
        defs = d["defteam"].to_list()
        for i in range(d.height):
            if not keep[i]:
                continue
            for tab, t in ((self.off, offs[i]), (self.dfn, defs[i])):
                v = tab.get(t)
                if v is None:
                    v = np.zeros((2, N_CAT))
                    tab[t] = v
                v[0, cat[i]] += 1.0
                v[1] += exp[i]


def MatchupProbs(base: BaseTables, f_home_off, f_home_def, f_away_off, f_away_def, tilt_home=0.0, tilt_away=0.0):
    """-> probs[side(0 home offense, 1 away offense), state, fp, cat]. tilt multiplies the offense's TD and FG
    factors by (1 + tilt) (stress tests only)."""
    out = np.empty((2, N_STATE, N_FP, N_CAT))
    for side, (fo, fd, hf, tilt) in enumerate(((f_home_off, f_away_def, base.home, tilt_home), (f_away_off, f_home_def, np.ones(N_CAT), tilt_away))):
        m = fo * fd * hf
        if tilt:
            m = m.copy()
            m[CAT_ID["TD"]] *= 1 + tilt
            m[CAT_ID["FG"]] *= 1 + tilt
        p = base.p0 * m
        out[side] = p / p.sum(axis=2, keepdims=True)
    return out
