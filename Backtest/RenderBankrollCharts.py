r"""Re-render American Footbot bankroll drafts under Tim's season-axis rule.

Reads existing AgentDocs tidy CSVs and does not rerun simulations. Headline
equity numbers are taken from those CSVs unchanged.

    .venv\Scripts\python.exe -m Backtest.RenderBankrollCharts
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Backtest.BankrollPlot import FrameTidy, NflSeasonLabel, PlotBankroll

ROOT = Path(r"E:\AgentDocs\American-Footbot")
NOTE = (
    "Charts were re-rendered on Nov 13 under the season-axis rule with numbers unchanged."
)


def LoadCsv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def BackupOld(src, dest):
    """Copy an existing PNG once, so a re-run does not overwrite the backup."""
    src, dest = Path(src), Path(dest)
    if src.exists() and not dest.exists():
        shutil.copy2(src, dest)
        return str(dest)
    return None


def SaveFig(fig, path, dpi=150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def NoteMd(path):
    path = Path(path)
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8")
    if "season-axis rule" in text:
        return False
    if not text.endswith(newline):
        text += newline
    path.write_bytes((text + NOTE + newline).encode("utf-8"))
    return True


def Rename(rows, naming):
    """naming(row) -> label or None to drop."""
    out = []
    for row in rows:
        label = naming(row)
        if not label:
            continue
        item = dict(row)
        item["series"] = label
        out.append(item)
    return out


def PlotOne(rows, path, title, ylabel="Bankroll ($)", logy=False, hlines=None,
            figsize=(11, 6.5), legend=True, start_line=True):
    frame = FrameTidy(rows, mode="auto")
    lines = list(hlines or [])
    if start_line:
        lines = [(10000, "$10k start", {"color": "black", "lw": 0.8, "ls": "-", "alpha": 0.8})] + lines
    fig, ax = plt.subplots(figsize=figsize)
    PlotBankroll(frame, ax=ax, title=title, ylabel=ylabel, logy=logy,
                 hlines=lines, legend=legend)
    return frame, SaveFig(fig, path)


def PlotSeasonPanels(rows, path, title, ylabel="Bankroll ($)", figsize=None, start_line=True):
    """One calendar panel per season (for bankrolls that reset each season)."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[NflSeasonLabel(row["date"])].append(row)
    labels = sorted(buckets)
    if not labels:
        return {}, None
    w = max(5.4 * len(labels), 8)
    fig, axes = plt.subplots(1, len(labels), figsize=figsize or (w, 6.2), sharey=True)
    if len(labels) == 1:
        axes = [axes]
    frames = {}
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        frame = FrameTidy(buckets[lab], mode="single")
        lines = ([(10000, "", {"color": "black", "lw": 0.8, "ls": "-", "alpha": 0.8})]
                 if start_line else None)
        PlotBankroll(
            frame, ax=ax, title=f"{lab} season", ylabel=ylabel if i == 0 else "",
            hlines=lines, legend=(i == len(labels) - 1),
        )
        frames[lab] = frame
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    return frames, SaveFig(fig, path)


def LastEquity(rows, series=None):
    picked = [r for r in rows if series is None or r["series"] == series]
    if not picked:
        return None
    return float(picked[-1]["equity"])


def MaxDrawdownPct(rows, series=None):
    picked = [float(r["equity"]) for r in rows if series is None or r["series"] == series]
    if not picked:
        return None
    peak, worst = picked[0], 0.0
    for val in picked:
        peak = max(peak, val)
        if peak:
            worst = min(worst, val / peak - 1.0)
    return worst


# ---------------------------------------------------------------------------
# Per-folder renderers
# ---------------------------------------------------------------------------

def RenderIngameLiveSim():
    folder = ROOT / "2026-11-13_ingame-live-sim"
    base = "2026-11-13_ingame-live-sim"
    rows = LoadCsv(folder / f"{base}_bankroll.csv")
    draft = folder / f"{base}_bankroll_draft.png"
    BackupOld(draft, folder / f"{base}_bankroll_draft_old-axis.png")

    names = []
    seen = set()
    for row in rows:
        rule = row["series"].split("|")[0]
        if rule not in seen:
            seen.add(rule)
            names.append(rule)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    frames = {}
    for ax, stake, stake_title in zip(
        axes,
        ("flat", "qkelly"),
        ("flat $200/bet", "quarter Kelly (cap 5%)"),
    ):
        suffix = f"|taker|kalshi_verified|{stake}"
        sub = Rename(
            [r for r in rows if r["series"].endswith(suffix)],
            lambda r, s=suffix: r["series"][:-len(s)] if r["series"].endswith(s) else None,
        )
        # Keep the original rule order.
        order = {n: i for i, n in enumerate(names)}
        sub.sort(key=lambda r: (order.get(r["series"], 99), r["date"]))
        frame = FrameTidy(sub, mode="auto")
        PlotBankroll(
            frame, ax=ax,
            title=f"Continuous $10k, {stake_title}; taker fill + Kalshi verified fee",
            ylabel="bank ($)",
            hlines=[(10000, "", {"color": "black", "lw": 0.8, "ls": "-", "alpha": 0.8})],
            legend={"fontsize": 6.5, "ncol": 2} if stake == "flat" else False,
        )
        frames[stake] = frame
    written = SaveFig(fig, draft, dpi=130)
    key = "close_35_65_gap3|taker|kalshi_verified|qkelly"
    check = {
        "end_close_35_65_gap3_qkelly": LastEquity(rows, key),
        "max_dd_close_35_65_gap3_qkelly": MaxDrawdownPct(rows, key),
        "mode": frames["qkelly"]["mode"],
        "seasons": frames["qkelly"]["seasons"],
        "boundaries": [b["season"] for b in frames["qkelly"]["boundaries"]],
    }
    NoteMd(folder / f"{base}.md")
    return {"draft": written, "check": check}


def RenderRlStaking():
    folder = ROOT / "2026-11-13_rl-staking-design"
    base = "2026-11-13_rl-staking-design"
    BackupOld(folder / f"{base}_fig1.png", folder / f"{base}_fig1_draft_old-axis.png")
    BackupOld(folder / f"{base}_fig2.png", folder / f"{base}_fig2_draft_old-axis.png")

    labels = {
        "learned_million": "learned_million",
        "learned_growth": "learned_growth",
        "frozen_rules": "frozen_rules",
        "frozen_picks_kelly0.5": "frozen_picks_kelly0.5",
        "all_edge_kelly0.25": "all_edge_kelly0.25",
        "no_bet": "no_bet",
        "mart_best_qualifying": "mart_best_qualifying",
        "mart_max_p1m": "mart_max_p1m",
    }

    def pick(rows, stage, basis="tim_fee_est"):
        return Rename(
            [r for r in rows if r.get("stage") == stage and r.get("basis") == basis
             and r["series"] in labels],
            lambda r: labels[r["series"]],
        )

    val = LoadCsv(folder / f"{base}_validation.csv")
    hold = LoadCsv(folder / f"{base}_holdout.csv")
    _, p1 = PlotOne(
        pick(val, "val_chain_2019_2024"),
        folder / f"{base}_validation_draft.png",
        title="Walk-forward validation, one continuous $10k bank, 2019-2024 (Tim's $0.02 fee estimate)",
        figsize=(12, 6.5),
        legend={"fontsize": 8},
    )
    _, p2 = PlotOne(
        pick(hold, "holdout_replay"),
        folder / f"{base}_holdout_draft.png",
        title="Frozen holdout replay, 2025-2026, $10k start (Tim's $0.02 fee estimate)",
        figsize=(12, 6.5),
        legend={"fontsize": 8},
    )

    def end(rows, name, stage, basis="tim_fee_est"):
        return LastEquity(
            [r for r in rows if r["series"] == name and r.get("stage") == stage and r.get("basis") == basis],
        )

    check = {
        "val_learned_million_fee": end(val, "learned_million", "val_chain_2019_2024"),
        "val_frozen_rules_fee": end(val, "frozen_rules", "val_chain_2019_2024"),
        "hold_learned_million_fee": end(hold, "learned_million", "holdout_replay"),
        "hold_learned_growth_fee": end(hold, "learned_growth", "holdout_replay"),
        "hold_frozen_rules_fee": end(hold, "frozen_rules", "holdout_replay"),
        "hold_learned_million_close": end(hold, "learned_million", "holdout_replay", "closing_ml"),
        "hold_learned_growth_close": end(hold, "learned_growth", "holdout_replay", "closing_ml"),
    }
    NoteMd(folder / f"{base}.md")
    return {"drafts": [p1, p2], "check": check}


def RenderCompounding():
    folder = ROOT / "2026-11-09_compounding-bankroll"
    base = "2026-11-09_compounding-bankroll"
    BackupOld(folder / f"{base}.png", folder / f"{base}_draft_old-axis.png")
    rows = LoadCsv(folder / f"{base}.csv")
    labels = {
        "nested_walkforward": "Nested walk-forward (honest)",
        "frozen_compounded": "FrozenConfig compounded",
        "best_in_sample": "Best in-sample (overfit)",
        "baseline_market_fav_2pct": "Market favourite 2%",
        "baseline_no_bet": "No bet",
    }
    sub = Rename(
        [r for r in rows if r.get("basis") == "closing_ml" and r["series"] in labels],
        lambda r: labels[r["series"]],
    )
    frame, path = PlotOne(
        sub,
        folder / f"{base}_draft.png",
        title="One continuous $10k bankroll, 2023 through 2026 week 9 (closing moneylines)",
        ylabel="Bankroll ($, log scale)",
        logy=True,
        hlines=[(1_000_000, "$1M target", {"color": "#555555", "lw": 1, "ls": "--", "alpha": 0.8})],
        figsize=(12, 6.8),
        legend={"fontsize": 8},
    )
    def end(name, basis):
        return LastEquity([r for r in rows if r["series"] == name and r.get("basis") == basis])
    check = {
        "nested_closing": end("nested_walkforward", "closing_ml"),
        "nested_fee": end("nested_walkforward", "tim_fee_est"),
        "frozen_closing": end("frozen_compounded", "closing_ml"),
        "best_closing": end("best_in_sample", "closing_ml"),
        "mode": frame["mode"],
        "seasons": frame["seasons"],
        "last_date": frame["last_date"],
    }
    NoteMd(folder / f"{base}.md")
    return {"draft": path, "check": check}


def RenderKalshi():
    folder = ROOT / "2026-11-13_kalshi-reality-check"
    base = "2026-11-13_kalshi-reality-check"
    BackupOld(folder / f"{base}_fig1.png", folder / f"{base}_fig1_draft_old-axis.png")
    rows = LoadCsv(folder / f"{base}.csv")

    def label(row):
        ser, time = row["series"], row["time"]
        if ser == "closing_line|tim_fee_est":
            # Closing line does not depend on T; keep one copy.
            return "Closing + Tim fee est" if time == "T-90" else None
        want = {
            ("kalshi_ask|asis|kalshi_verified", "T-90"): "T-90 asis, Kalshi fee",
            ("kalshi_ask|asis|kalshi_verified", "T-15"): "T-15 asis, Kalshi fee",
            ("kalshi_ask|kmid|kalshi_verified", "T-90"): "T-90 kmid, Kalshi fee",
            ("kalshi_ask|kmid|kalshi_verified", "T-15"): "T-15 kmid, Kalshi fee",
        }
        return want.get((ser, time))

    sub = Rename(rows, label)
    frames, path = PlotSeasonPanels(
        sub,
        folder / f"{base}_draft.png",
        title="Frozen rule at Kalshi executable asks, $10k reset each season (verified Kalshi fee)",
    )
    # Per-season ends used in the report (reset bank): 2025 T-90 asis verified 10965.70, etc.
    def end(ser, sname, time):
        return LastEquity([r for r in rows if r["series"] == ser and r["set"] == sname and r["time"] == time])
    check = {
        "2025_t90_asis_verified": end("kalshi_ask|asis|kalshi_verified", "2025 (contaminated)", "T-90"),
        "2026_t90_asis_verified": end("kalshi_ask|asis|kalshi_verified", "2026 wk1-9 (cleaner)", "T-90"),
        "2026_t15_kmid_verified": end("kalshi_ask|kmid|kalshi_verified", "2026 wk1-9 (cleaner)", "T-15"),
        "modes": {k: fr["mode"] for k, fr in frames.items()},
    }
    NoteMd(folder / f"{base}.md")
    return {"draft": path, "check": check}


def RenderPolymarket():
    folder = ROOT / "2026-11-13_polymarket-history"
    base = "2026-11-13_polymarket-history"
    BackupOld(folder / f"{base}_fig1.png", folder / f"{base}_fig1_draft_old-axis.png")
    rows = LoadCsv(folder / f"{base}.csv")

    def label(row):
        ser, time = row["series"], row["time"]
        if ser == "closing_line|tim_fee_est":
            return "Closing + Tim fee est" if time == "T-90" else None
        want = {
            ("poly_mid|asis|tim_est_0.02", "T-90"): "T-90 asis, Tim fee est",
            ("poly_mid|asis|tim_est_0.02", "T-15"): "T-15 asis, Tim fee est",
            ("poly_mid|kmid|tim_est_0.02", "T-90"): "T-90 mktmid, Tim fee est",
            ("poly_mid|kmid|tim_est_0.02", "T-15"): "T-15 mktmid, Tim fee est",
        }
        return want.get((ser, time))

    sub = Rename(rows, label)
    frames, path = PlotSeasonPanels(
        sub,
        folder / f"{base}_draft.png",
        title="Frozen rule at Polymarket mid (Tim's fill assumption), $10k reset each season, Tim's $0.02 fee estimate",
        figsize=(16, 6.2),
    )
    def end(ser, sname, time):
        return LastEquity([r for r in rows if r["series"] == ser and r["set"] == sname and r["time"] == time])
    check = {
        "2024_t90_asis_est": end("poly_mid|asis|tim_est_0.02", "2024 (contaminated)", "T-90"),
        "2026_t90_asis_est": end("poly_mid|asis|tim_est_0.02", "2026 wk1-9 (cleaner)", "T-90"),
        "modes": {k: fr["mode"] for k, fr in frames.items()},
    }
    NoteMd(folder / f"{base}.md")
    return {"draft": path, "check": check}


EXPECTED = {
    # In-game live-sim headlines (real sequence, taker + Kalshi fee, quarter Kelly).
    "ingame_end": 20210.61,
    "ingame_dd": -0.389,          # -38.9%
    # Compounding, closing ML.
    "nested_closing": 40448.80,
    "frozen_closing": 28654.99,
    "best_closing": 63010.74,
    "nested_fee": 20114.27,
    # RL holdout replay (report rounds these).
    "hold_learned_million_close": 9630.57,   # $9,631
    "hold_learned_million_fee": 8759.07,     # $8,759
    "hold_learned_growth_close": 10371.0,    # will fill from CSV at runtime if needed
    "hold_frozen_rules_fee": 11631.69,       # $11,632
    # RL validation fee-est ends in the table: learned_million $8,690, frozen $28,568
    "val_learned_million_fee": 8689.55,
    "val_frozen_rules_fee": 28567.51,
}


def CheckHeadlines(report):
    """Compare CSV-derived ends to the published report figures. CSVs are not modified."""
    issues = []
    ing = report["ingame"]["check"]
    if abs(ing["end_close_35_65_gap3_qkelly"] - EXPECTED["ingame_end"]) > 0.02:
        issues.append(f"ingame end {ing['end_close_35_65_gap3_qkelly']} != {EXPECTED['ingame_end']}")
    if abs(ing["max_dd_close_35_65_gap3_qkelly"] - EXPECTED["ingame_dd"]) > 0.0015:
        issues.append(f"ingame dd {ing['max_dd_close_35_65_gap3_qkelly']} != {EXPECTED['ingame_dd']}")
    comp = report["compounding"]["check"]
    for key in ("nested_closing", "frozen_closing", "best_closing", "nested_fee"):
        if abs(comp[key] - EXPECTED[key]) > 0.02:
            issues.append(f"compounding {key} {comp[key]} != {EXPECTED[key]}")
    rl = report["rl"]["check"]
    for key in ("hold_learned_million_close", "hold_learned_million_fee",
                "hold_frozen_rules_fee", "val_learned_million_fee", "val_frozen_rules_fee"):
        if abs(rl[key] - EXPECTED[key]) > 0.5:  # report rounds to the dollar
            issues.append(f"rl {key} {rl[key]} != {EXPECTED[key]}")
    # PathSim figures live in the (untouched) JSON, not the equity CSV.
    js = json.loads((ROOT / "2026-11-13_ingame-live-sim" / "2026-11-13_ingame-live-sim.json").read_text(encoding="utf-8"))
    report["ingame"]["path_sim"] = _PathSimHeadline(js)
    return issues


def _PathSimHeadline(js):
    """Pull the published PathSim cell for close 35-65 gap3, quarter Kelly."""
    paths = js.get("path_sim") or js
    # Be tolerant of nesting.
    found = {}

    def walk(obj, trail=()):
        if not isinstance(obj, dict):
            return
        keys = set(obj)
        if "median" in keys or "p_1m" in keys or "P_1m" in keys or "median_final" in keys:
            found[trail] = {k: obj[k] for k in obj if k in (
                "median", "median_final", "p_1m", "P_1m", "p_hit_1m", "p_ruin",
                "real_final", "max_dd_pct", "p10", "p90",
            ) or "median" in k or "1m" in k or "1M" in k}
            return
        for k, v in obj.items():
            if isinstance(v, dict):
                walk(v, trail + (str(k),))

    walk(paths)
    return {str(k): v for k, v in found.items() if "close_35_65_gap3" in "".join(k) and "qkelly" in "".join(k)}


def Main():
    report = {
        "ingame": RenderIngameLiveSim(),
        "rl": RenderRlStaking(),
        "compounding": RenderCompounding(),
        "kalshi": RenderKalshi(),
        "polymarket": RenderPolymarket(),
    }
    issues = CheckHeadlines(report)
    report["headline_issues"] = issues
    out = ROOT / "2026-11-13_ingame-live-sim" / "_season_axis_rerender_check.json"
    # Keep the check next to a report, but don't pollute; print only.
    print(json.dumps(report, indent=2, default=str))
    if issues:
        print("HEADLINE MISMATCHES:")
        for item in issues:
            print(" -", item)
        return 2
    print("HEADLINES MATCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())