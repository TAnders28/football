r"""
One-game chart: Kalshi and Polymarket mid (home win probability) over time, pregame and in-game, halftime shaded,
scoring plays marked, with our walk-forward PREGAME model (p_full2) and the de-vigged closing line as horizontal
reference lines. We have no in-game model; the model line is a pregame number carried across for reference only.

    .venv\Scripts\python.exe -m Backtest.PlotGameProbs --game 2026_09_KC_BUF [--out-dir DIR]

Needs data/prices/ingame_kalshi_<s>.jsonl / ingame_poly_<s>.jsonl (DataPull.PullInGameHistory) and
data/playbyplay/pbp_<s>.jsonl. Writes <out-dir>/<game_id>_draft.png (default data/backtest/game_charts).
X axis: minutes from the first snap (nflverse time_of_day, UTC) with Eastern clock labels.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from Backtest.InGameCalibration import SeriesKalshi, SeriesPoly, Ts, Windows
from Backtest.Lab import AllRows, AttachModels, LoadSeason


def Eastern(ts):
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(ts, ZoneInfo("America/New_York")), "ET"
    except Exception:
        # US Eastern without tzdata: EDT (UTC-4) from the 2nd Sunday of March to the 1st Sunday of November.
        u = datetime.fromtimestamp(ts, timezone.utc)
        mar = datetime(u.year, 3, 8, 7, tzinfo=timezone.utc); mar += timedelta(days=(6 - mar.weekday()) % 7)
        nov = datetime(u.year, 11, 1, 6, tzinfo=timezone.utc); nov += timedelta(days=(6 - nov.weekday()) % 7)
        off = -4 if mar <= u < nov else -5
        return u + timedelta(hours=off), "ET"


def Row(stem, season, gid):
    p = os.path.join("data", "prices", f"{stem}_{season}.jsonl")
    if not os.path.exists(p):
        return None
    for l in open(p, encoding="utf-8"):
        if f'"game_id": "{gid}"' in l[:200]:
            return json.loads(l)
    return None


def Scoring(season, gid):
    out = []
    with open(os.path.join("data", "playbyplay", f"pbp_{season}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if gid not in line[:400]:
                continue
            r = json.loads(line)
            if r.get("game_id") == gid and r.get("sp") == 1 and Ts(r.get("time_of_day")):
                out.append((Ts(r.get("end_clock_time")) or Ts(r["time_of_day"]), int(r.get("total_away_score") or 0), int(r.get("total_home_score") or 0), r.get("posteam")))
    return out


def Plot(gid, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    season = int(gid[:4])
    rbs = AllRows(2012, season)
    AttachModels(rbs)
    g = next(r for r in rbs[season] if r["game_id"] == gid)
    w = Windows(season, {gid})[gid]
    k0 = w["kickoff"]
    fig, ax = plt.subplots(figsize=(13, 6))
    have = []
    for name, stem, fn, color in (("Kalshi mid", "ingame_kalshi", SeriesKalshi, "#1f77b4"), ("Polymarket mid", "ingame_poly", SeriesPoly, "#d62728")):
        r = Row(stem, season, gid)
        ser = sorted(fn(r)) if r else []
        if ser:
            ax.step([(t - k0) / 60 for t, _ in ser], [p for _, p in ser], where="post", color=color, lw=1.2, label=name)
            have.append(name)
    ax.axhline(g["p_full2"], color="#2ca02c", lw=2, ls="-", label=f"our pregame model p_full2 = {g['p_full2']:.3f} (pregame only)")
    ax.axhline(g["mkt_home"], color="black", lw=1, ls=":", label=f"closing line de-vigged = {g['mkt_home']:.3f}")
    for a, b in w["excluded"]:
        ax.axvspan((a - k0) / 60, (b - k0) / 60, color="grey", alpha=0.18, lw=0)
    if w["excluded"]:
        a, b = w["excluded"][0]
        ax.text(((a + b) / 2 - k0) / 60, 0.02, "halftime" if w["halftime_min"] else "stoppage", ha="center", fontsize=8, color="dimgrey")
    ax.axvline(0, color="grey", lw=1)
    ax.axvline((w["end"] - k0) / 60, color="grey", lw=1)
    for t, aw, hm, pos in Scoring(season, gid):
        x = (t - k0) / 60
        ax.axvline(x, color="orange", lw=0.7, alpha=0.7)
        ax.text(x, 1.01, f"{aw}-{hm}", rotation=90, fontsize=7, va="bottom", ha="center", color="darkorange")
    ticks = list(range(-180, int((w["end"] - k0) / 60) + 31, 30))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{m}\n{Eastern(k0 + m * 60)[0]:%H:%M}" for m in ticks], fontsize=8)
    ax.set_xlim(-180, (w["end"] - k0) / 60 + 15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("minutes from first snap / Eastern clock (ET)")
    ax.set_ylabel(f"{g['home_team']} (home) win probability")
    sg = next(x for x in LoadSeason(season) if x["game_id"] == gid)
    res = f"{sg['away_team']} {sg['away_score']} at {sg['home_team']} {sg['home_score']}"
    ko_et = Eastern(k0)[0]
    ax.set_title(f"{gid}: {res}   first snap {ko_et:%a %b %d %H:%M} ET\n"
                 "mid = (bid+ask)/2 (Kalshi) / prices-history (Polymarket), 1-minute resolution; grey = halftime/long stoppage; orange = scoring plays (away-home)",
                 fontsize=9, pad=28)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{gid}_draft.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path, have


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, action="append", help="nflverse game_id, e.g. 2026_09_KC_BUF (repeatable)")
    ap.add_argument("--out-dir", default=os.path.join("data", "backtest", "game_charts"))
    a = ap.parse_args()
    for gid in a.game:
        path, have = Plot(gid, a.out_dir)
        print(gid, path, "series:", ", ".join(have) or "none")


if __name__ == "__main__":
    Main()