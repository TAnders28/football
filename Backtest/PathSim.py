r"""
PathSim: randomized-path scoring for day-level strategies (Tim, 2026-11-13: "add some randomness - if a strategy fails
after wrong bets in week 1 we never see how it plays out longer term"). Simulation only.

A strategy is reduced to its sequence of DAILY bank returns r_d = (bank_end / bank_start - 1) from a real walk.
Frozen/Kelly/pct-of-bank sizing is proportional to the bank, so these returns can be re-ordered and compounded on one
continuous $10k bank (the $1 minimum stake is the only non-proportional piece, ignored).

Path generators (all seeded, numpy Generator):
  real             the real order (one path)
  random_start     start on a random game day, walk forward circularly (wraps to the window start)
  block_bootstrap  circular moving blocks of `block` consecutive game days drawn with replacement
  shuffle          random permutations of the game days (concatenated when the horizon is longer than the window)
Each generator returns an (paths, horizon) array of day indices; the same arrays can be passed as `idx` to
Backtest.StakingEnv.Simulate (see StakingEnv.RandomIdx).

Absorbing first passage: bank >= TARGET ($1M) is a hit, bank <= RUIN ($1,000, same as MillionTarget/StakingEnv) is
ruin. With restart=True a "life" also ends at ruin OR when the bank is below `bad_start_level` x start after
`bad_start_days` game days (~a bad first month); the next game day starts a fresh $10k life. Reported: restarts per
path, final bank of the last life, P(any life hits $1M).
"""

import numpy as np

START = 10_000.0
TARGET = 1_000_000.0
RUIN = 1_000.0
METHODS = ("random_start", "block_bootstrap", "shuffle")


def DayOrders(n_days, paths, horizon, method, rng, block=4):
    if n_days <= 0:
        raise ValueError("no days")
    if method == "real":
        return np.tile(np.arange(horizon) % n_days, (paths, 1))
    if method == "random_start":
        start = rng.integers(0, n_days, paths)
        return (start[:, None] + np.arange(horizon)[None, :]) % n_days
    if method == "block_bootstrap":
        nb = -(-horizon // block)
        starts = rng.integers(0, n_days, (paths, nb))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n_days
        return idx.reshape(paths, nb * block)[:, :horizon]
    if method == "shuffle":
        reps = -(-horizon // n_days)
        perms = np.concatenate([rng.permuted(np.tile(np.arange(n_days), (paths, 1)), axis=1) for _ in range(reps)], axis=1)
        return perms[:, :horizon]
    raise ValueError(method)


def RunPaths(returns, idx, start=START, target=TARGET, ruin=RUIN, restart=False, bad_start_days=4, bad_start_level=0.8):
    r = np.asarray(returns, float)
    paths, horizon = idx.shape
    bank = np.full(paths, start)
    life_day = np.zeros(paths, int)
    hit = np.zeros(paths, bool)
    ruined = np.zeros(paths, bool)
    t_hit = np.full(paths, -1)
    t_ruin = np.full(paths, -1)
    restarts = np.zeros(paths, int)
    for j in range(horizon):
        live = ~hit & (restart | ~ruined)
        bank = np.where(live, bank * (1.0 + r[idx[:, j]]), bank)
        life_day = life_day + live
        new_hit = live & (bank >= target)
        hit |= new_hit
        t_hit = np.where(new_hit & (t_hit < 0), j + 1, t_hit)
        new_ruin = live & ~new_hit & (bank <= ruin)
        t_ruin = np.where(new_ruin & (t_ruin < 0), j + 1, t_ruin)
        ruined |= new_ruin
        if restart:
            bad = live & ~new_hit & (life_day == bad_start_days) & (bank < bad_start_level * start)
            reset = (new_ruin | bad) & (j < horizon - 1)
            restarts += reset
            bank = np.where(reset, start, bank)
            life_day = np.where(reset, 0, life_day)
    return {"final": bank, "hit": hit, "ruin": ruined, "t_hit": t_hit, "t_ruin": t_ruin, "restarts": restarts}


def Summary(res):
    f = res["final"]
    tr = res["t_ruin"][res["t_ruin"] > 0]
    th = res["t_hit"][res["t_hit"] > 0]
    return {"paths": int(len(f)), "median_final": round(float(np.median(f)), 2),
            "p10_final": round(float(np.percentile(f, 10)), 2), "p90_final": round(float(np.percentile(f, 90)), 2),
            "p_ruin": round(float(res["ruin"].mean()), 4), "p_1m": round(float(res["hit"].mean()), 4),
            "median_days_to_ruin": float(np.median(tr)) if len(tr) else None,
            "median_days_to_1m": float(np.median(th)) if len(th) else None,
            "mean_restarts": round(float(res["restarts"].mean()), 3),
            "p_final_below_start": round(float((f < START).mean()), 4)}


def Evaluate(returns, seed=20261113, paths=5000, horizons=None, block=4, bad_start_days=4, bad_start_level=0.8):
    """Real sequence + every method x {absorbing, restart} x horizon. Deterministic for a given seed."""
    r = np.asarray(returns, float)
    n = len(r)
    horizons = horizons or {"window": n}
    real = RunPaths(r, DayOrders(n, 1, n, "real", None))
    out = {"n_days": n, "seed": seed, "block": block, "real_sequence": Summary(real), "randomized": {}}
    for hname, h in horizons.items():
        for k, method in enumerate(METHODS):
            for restart in (False, True):
                rng = np.random.default_rng([seed, k, int(h), int(restart)])
                idx = DayOrders(n, paths, h, method, rng, block)
                res = RunPaths(r, idx, restart=restart, bad_start_days=bad_start_days, bad_start_level=bad_start_level)
                out["randomized"][f"{hname}|{method}|{'restart' if restart else 'absorbing'}"] = dict(Summary(res), horizon=int(h))
    return out


def DailyReturns(daily, start=START, season_of=None):
    """[{date, equity}] from a walk -> daily returns. If season_of(date) changes, the bank was reset to `start`."""
    out, prev, prev_season = [], start, None
    for d in daily:
        s = season_of(d["date"]) if season_of else None
        if season_of and s != prev_season:
            prev = start
            prev_season = s
        out.append(d["equity"] / prev - 1.0)
        prev = d["equity"]
    return out