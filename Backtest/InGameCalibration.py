r"""
Market-calibration research (not a strategy search): how well do Kalshi and Polymarket NFL moneyline prices
predict the final result, sampled while the game is in play (and, for comparison, pregame T-3h..kickoff)?

    .venv\Scripts\python.exe -m Backtest.InGameCalibration [--out-dir DIR --base NAME --boot 2000 --seed 20261113]

Inputs: data/prices/ingame_kalshi_<s>.jsonl, data/prices/ingame_poly_<s>.jsonl (DataPull.PullInGameHistory),
data/playbyplay/pbp_<s>.jsonl (nflverse; time_of_day / end_clock_time are UTC ISO strings ending in Z).
Probability = mid (Tim's choice): Kalshi (yes_bid_close + yes_ask_close) / 2 of the home-team market; Polymarket the
prices-history price of token0 converted to the home side. One side per game, so each timestamp counts once.
In-play window: first snap (pbp) .. last play end, minus halftime (last Q2 play end .. first Q3 snap) and minus any other
gap of more than GAP_MIN minutes between consecutive plays. Samples on a 1-minute UTC grid, using the latest price no older
than STALE_S seconds. Quarter = qtr of the latest play at or before the sample (5 = OT). Ties are dropped.
Error bars, both shown: naive Wilson 90% interval treating every sample as independent (too narrow: samples within a
game share one outcome) and a game-level cluster bootstrap 90% interval (resample games).
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from Backtest.Lab import LoadSeason

GAP_MIN = 10
STALE_S = 180
STEP_S = 60
PRE_S = 3 * 3600
BINS = 10
Z90 = 1.6448536269514722


def Ts(s):
    if not s or "T" not in str(s):
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def Windows(season, wanted):
    """game_id -> {kickoff, end, excluded [(a,b)], plays [(t, qtr)]} from nflverse pbp wall-clock times."""
    plays = defaultdict(list)
    with open(os.path.join("data", "playbyplay", f"pbp_{season}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            gid = r.get("game_id")
            if gid not in wanted or r.get("qtr") is None:
                continue
            t0, t1 = Ts(r.get("time_of_day")), Ts(r.get("end_clock_time"))
            if t0 is None:
                continue
            plays[gid].append((t0, max(t0, t1 or t0), int(r["qtr"])))
    out = {}
    for gid, ps in plays.items():
        ps.sort()
        q2 = [p for p in ps if p[2] == 2]
        q3 = [p for p in ps if p[2] == 3]
        exc = []
        if q2 and q3:
            exc.append((max(p[1] for p in q2), min(p[0] for p in q3)))
        for a, b in zip(ps, ps[1:]):
            if b[0] - a[1] > GAP_MIN * 60 and not (a[2] == 2 and b[2] == 3):
                exc.append((a[1], b[0]))
        out[gid] = {"kickoff": ps[0][0], "end": max(p[1] for p in ps), "excluded": exc,
                    "plays": [(p[0], p[2]) for p in ps], "halftime_min": round((exc[0][1] - exc[0][0]) / 60, 1) if q2 and q3 else None}
    return out


def SeriesKalshi(row):
    pts = []
    for c in row["candles"]:
        b, a = c.get("yes_bid_close"), c.get("yes_ask_close")
        if b is None or a is None or a < b:
            continue
        p = (b + a) / 2
        pts.append((c["end_ts"], p if row["side_team"] == row["home_team"] else 1 - p))
    return pts


def SeriesPoly(row):
    flip = row["token0_team"] != row["home_team"]
    return [(x["t"], 1 - x["p"] if flip else x["p"]) for x in row["history"] if 0 <= x["p"] <= 1]


def Sample(series, grid):
    """latest price at or before each grid time, no older than STALE_S; returns list of (t, p) for grid points with a price."""
    if not series:
        return []
    ts = np.array([s[0] for s in series], dtype=float)
    ps = np.array([s[1] for s in series], dtype=float)
    order = np.argsort(ts, kind="stable")
    ts, ps = ts[order], ps[order]
    g = np.asarray(grid, dtype=float)
    idx = np.searchsorted(ts, g, side="right") - 1
    ok = (idx >= 0)
    ok[ok] &= (g[ok] - ts[idx[ok]]) <= STALE_S
    return [(float(t), float(ps[i])) for t, i, k in zip(g, idx, ok) if k]


def InPlayGrid(w):
    out = []
    t = math.ceil(w["kickoff"] / STEP_S) * STEP_S
    pt = np.array([p[0] for p in w["plays"]])
    pq = np.array([p[1] for p in w["plays"]])
    while t <= w["end"]:
        if not any(a < t < b for a, b in w["excluded"]):
            q = int(pq[max(0, np.searchsorted(pt, t, side="right") - 1)])
            out.append((t, min(q, 5)))
        t += STEP_S
    return out


def Wilson(k, n, z=Z90):
    if n == 0:
        return None, None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def BinOf(p):
    return min(int(p * BINS), BINS - 1)


def Calibrate(samples, boot, rng):
    """samples: list of (game_id, p, y). Returns per-bin rows + brier/ece (+ cluster CIs)."""
    if not samples:
        return {"bins": [], "n_samples": 0, "n_games": 0}
    gids = sorted({s[0] for s in samples})
    gi = {g: i for i, g in enumerate(gids)}
    G = len(gids)
    cnt, sy, sp, sbr = np.zeros((G, BINS)), np.zeros((G, BINS)), np.zeros((G, BINS)), np.zeros(G)
    ns = np.zeros(G)
    for g, p, y in samples:
        i, b = gi[g], BinOf(p)
        cnt[i, b] += 1; sy[i, b] += y; sp[i, b] += p; sbr[i] += (p - y) ** 2; ns[i] += 1
    # game-weighted: each game contributes its own bin mean once
    has = cnt > 0
    gw_y = np.where(has, sy / np.maximum(cnt, 1), 0)
    gw_p = np.where(has, sp / np.maximum(cnt, 1), 0)
    W = rng.multinomial(G, [1 / G] * G, size=boot).astype(float) if boot else np.zeros((0, G))

    def Stats(w):
        c, y, p = w @ cnt, w @ sy, w @ sp
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = y / c
            meanp = p / c
            gwn = w @ has
            gwo = (w @ gw_y) / gwn
        tot = c.sum()
        ece = np.nansum(np.abs(meanp - obs) * c) / tot
        brier = (w @ sbr) / (w @ ns)
        return obs, gwo, ece, brier

    obs, gwo, ece, brier = Stats(np.ones(G))
    bo = [Stats(w) for w in W]
    q = lambda arr: (np.nanquantile(arr, 0.05, axis=0), np.nanquantile(arr, 0.95, axis=0)) if len(arr) else (None, None)
    olo, ohi = q(np.array([b[0] for b in bo]))
    glo, ghi = q(np.array([b[1] for b in bo]))
    elo, ehi = q(np.array([b[2] for b in bo]))
    blo, bhi = q(np.array([b[3] for b in bo]))
    c_all, y_all, p_all = cnt.sum(0), sy.sum(0), sp.sum(0)
    rows = []
    for b in range(BINS):
        n = int(c_all[b])
        if n == 0:
            rows.append({"bin": f"{b*10}-{b*10+10}%", "n_samples": 0, "n_games": 0})
            continue
        nlo, nhi = Wilson(y_all[b], n)
        ng = int(has[:, b].sum())
        rows.append({"bin": f"{b*10}-{b*10+10}%", "n_samples": n, "n_games": ng,
                     "mean_pred": round(p_all[b] / n, 4), "obs_freq": round(y_all[b] / n, 4),
                     "naive_ci90_lo": round(nlo, 4), "naive_ci90_hi": round(nhi, 4),
                     "cluster_ci90_lo": round(float(olo[b]), 4), "cluster_ci90_hi": round(float(ohi[b]), 4),
                     "gw_mean_pred": round(float(gw_p[:, b].sum() / ng), 4), "gw_obs_freq": round(float(gwo[b]), 4),
                     "gw_cluster_ci90_lo": round(float(glo[b]), 4), "gw_cluster_ci90_hi": round(float(ghi[b]), 4)})
    return {"bins": rows, "n_samples": int(ns.sum()), "n_games": G, "brier": round(float(brier), 4),
            "brier_ci90": [round(float(blo), 4), round(float(bhi), 4)], "ece": round(float(ece), 4),
            "ece_ci90": [round(float(elo), 4), round(float(ehi), 4)]}


def Chart(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.28, wspace=0.18)
    colors = {"Kalshi": "#1f77b4", "Polymarket": "#d62728"}
    offs = {"Kalshi": -0.006, "Polymarket": 0.006}
    for col, (kind, label) in enumerate((("naive", "(a) naive per-sample binomial 90% CI\n(treats every 1-min sample as independent: too narrow)"),
                                         ("cluster", "(b) game-level cluster bootstrap 90% CI\n(resamples whole games: honest)"))):
        ax = fig.add_subplot(gs[0, col])
        ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect calibration")
        for venue, r in res.items():
            bs = [b for b in r["bins"] if b.get("n_samples")]
            x = np.array([b["mean_pred"] for b in bs]) + offs[venue]
            y = np.array([b["obs_freq"] for b in bs])
            lo = np.array([b[f"{kind}_ci90_lo"] for b in bs])
            hi = np.array([b[f"{kind}_ci90_hi"] for b in bs])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", ms=4, capsize=3, color=colors[venue],
                        label=f"{venue} ({r['n_games']} games, {r['n_samples']:,} samples)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("market probability (mid), mean within 10% bin"); ax.set_ylabel("observed home win frequency")
        ax.set_title(label, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(BINS)
    for k, (venue, r) in enumerate(res.items()):
        ax.bar(xs + (k - 0.5) * 0.4, [b.get("n_samples", 0) for b in r["bins"]], width=0.4, color=colors[venue], label=venue)
    ax.set_xticks(xs); ax.set_xticklabels([b["bin"] for b in next(iter(res.values()))["bins"]], fontsize=8)
    ax.set_ylabel("samples per bin"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "calibration"))
    ap.add_argument("--base", default="ingame_calibration")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261113)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    games = {}
    for s in (2024, 2025, 2026):
        for g in LoadSeason(s):
            if g.get("home_score") is None or (s == 2026 and int(g["week"]) > 9):
                continue
            games[g["game_id"]] = g
    venues = {"Kalshi": ("ingame_kalshi", SeriesKalshi, (2025, 2026)), "Polymarket": ("ingame_poly", SeriesPoly, (2024, 2025, 2026))}
    rows_by_venue = {}
    for v, (stem, fn, seasons) in venues.items():
        rows_by_venue[v] = {}
        for s in seasons:
            p = os.path.join("data", "prices", f"{stem}_{s}.jsonl")
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    r = json.loads(l)
                    if r["game_id"] in games:
                        rows_by_venue[v][r["game_id"]] = r
    need = {gid for d in rows_by_venue.values() for gid in d}
    win = {}
    for s in (2024, 2025, 2026):
        win.update(Windows(s, {g for g in need if int(games[g]["season"]) == s}))

    checks = {"kickoff_minus_schedule_min": [], "ties_dropped": 0, "halftime_min": [], "no_pbp": 0}
    from Backtest.SnapshotPrices import Kickoff
    samples = {v: {"ingame": [], "pregame": []} for v in venues}
    meta = {v: {"empty_series_games": 0, "games_with_ingame_samples": set()} for v in venues}
    for gid in sorted(need):
        g = games[gid]
        if g["home_score"] == g["away_score"]:
            checks["ties_dropped"] += 1
            continue
        w = win.get(gid)
        if not w:
            checks["no_pbp"] += 1
            continue
        y = 1 if g["home_score"] > g["away_score"] else 0
        checks["kickoff_minus_schedule_min"].append((w["kickoff"] - Kickoff(g).timestamp()) / 60)
        if w["halftime_min"] is not None:
            checks["halftime_min"].append(w["halftime_min"])
        grid = InPlayGrid(w)
        pre = [t for t in range(int(math.ceil((w["kickoff"] - PRE_S) / STEP_S) * STEP_S), int(w["kickoff"]), STEP_S)]
        for v, (_, fn, _) in venues.items():
            r = rows_by_venue[v].get(gid)
            if not r:
                continue
            ser = fn(r)
            if not ser:
                meta[v]["empty_series_games"] += 1
                continue
            qmap = dict(grid)
            for t, p in Sample(ser, [t for t, _ in grid]):
                samples[v]["ingame"].append((gid, p, y, qmap[t], int(g["season"])))
            for t, p in Sample(ser, pre):
                samples[v]["pregame"].append((gid, p, y, 0, int(g["season"])))

    subsets = {"all": lambda s: True, "excl_tails_3_97": lambda s: 0.03 <= s[1] <= 0.97}
    for q in (1, 2, 3, 4, 5):
        subsets[f"Q{q}" if q < 5 else "OT"] = (lambda qq: lambda s: s[3] == qq)(q)
    for yr in (2024, 2025, 2026):
        subsets[f"season_{yr}"] = (lambda yy: lambda s: s[4] == yy)(yr)
    results = {}
    tidy = []
    for window in ("ingame", "pregame"):
        for sub, f in subsets.items():
            if window == "pregame" and sub not in ("all", "excl_tails_3_97") and not sub.startswith("season"):
                continue
            for v in venues:
                ss = [(s[0], s[1], s[2]) for s in samples[v][window] if f(s)]
                r = Calibrate(ss, a.boot, rng)
                results[f"{window}|{sub}|{v}"] = r
                for b in r["bins"]:
                    tidy.append({"window": window, "subset": sub, "venue": v, **b})
    # Folded to the favourite side (p >= 0.5): direct over/under-confidence check independent of home/away framing.
    for window in ("ingame", "pregame"):
        for v in venues:
            ss = [(s[0], s[1], s[2]) if s[1] >= 0.5 else (s[0], 1 - s[1], 1 - s[2]) for s in samples[v][window]]
            r = Calibrate(ss, a.boot, rng)
            results[f"{window}|folded_favourite|{v}"] = r
            for b in r["bins"]:
                tidy.append({"window": window, "subset": "folded_favourite", "venue": v, **b})
    hw = {v: round(float(np.mean([y for _, y in {s[0]: s[2] for s in samples[v]["ingame"]}.items()])), 4) for v in venues}
    chk = {"kickoff_minus_schedule_min_median": round(float(np.median(checks["kickoff_minus_schedule_min"])), 2),
           "kickoff_minus_schedule_min_p05_p95": [round(float(x), 2) for x in np.quantile(checks["kickoff_minus_schedule_min"], [0.05, 0.95])],
           "home_win_rate_games": hw, "halftime_min_median": round(float(np.median(checks["halftime_min"])), 1), "ties_dropped": checks["ties_dropped"], "no_pbp": checks["no_pbp"],
           "per_venue": {v: {"games_in_files": len(rows_by_venue[v]), "empty_series_games": meta[v]["empty_series_games"],
                             "ingame_games": len({s[0] for s in samples[v]["ingame"]}), "ingame_samples": len(samples[v]["ingame"]),
                             "pregame_games": len({s[0] for s in samples[v]["pregame"]}), "pregame_samples": len(samples[v]["pregame"]),
                             "ingame_by_season": {yr: len({s[0] for s in samples[v]["ingame"] if s[4] == yr}) for yr in (2024, 2025, 2026)}}
                         for v in venues}}
    os.makedirs(a.out_dir, exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "resolution": {"Kalshi": "1-minute candlesticks (bid/ask close)", "Polymarket": "1-minute prices-history (fidelity=1)", "grid_s": STEP_S, "stale_s": STALE_S},
              "params": {"gap_min": GAP_MIN, "pregame_s": PRE_S, "boot": a.boot, "seed": a.seed}, "checks": chk, "results": results}
    with open(os.path.join(a.out_dir, f"{a.base}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    fields = ["window", "subset", "venue", "bin", "n_samples", "n_games", "mean_pred", "obs_freq", "naive_ci90_lo", "naive_ci90_hi",
              "cluster_ci90_lo", "cluster_ci90_hi", "gw_mean_pred", "gw_obs_freq", "gw_cluster_ci90_lo", "gw_cluster_ci90_hi"]
    with open(os.path.join(a.out_dir, f"{a.base}.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader(); wr.writerows(tidy)
    for window, sub, suffix, title in (("ingame", "all", "_draft.png", "NFL in-game calibration, 1-minute mid samples, halftime and >10-min stoppages excluded"),
                                       ("ingame", "excl_tails_3_97", "_excl_tails_draft.png", "NFL in-game calibration, samples with price 3%-97% only"),
                                       ("pregame", "all", "_pregame_draft.png", "NFL pregame calibration (T-3h to kickoff), 1-minute mid samples")):
        Chart({v: results[f"{window}|{sub}|{v}"] for v in venues}, os.path.join(a.out_dir, a.base + suffix), title)
    print(json.dumps({"checks": chk, "summary": {k: {m: v.get(m) for m in ("n_games", "n_samples", "brier", "ece")} for k, v in results.items()}}, indent=1))


if __name__ == "__main__":
    Main()