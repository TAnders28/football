"""Shared framing for bankroll and equity-curve charts.

Tim's rule (2026-11-13): a one-season bankroll chart runs from that season's
first game to its last game, with no flat offseason stretches. A multi-season
chart skips the summer gaps: put the x-axis on a game-day index, mark each
season boundary, and end at the last game of the final season.

This module is matplotlib-free. Callers can plot from FrameBankroll output, or
use PlotBankroll if matplotlib is installed.

NFL season labels are the kickoff year (the 2024 season runs Aug 2024 through
the February 2025 Super Bowl).
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

# Longer than a typical bye or the Super-Bowl fortnight; shorter than a normal
# NFL offseason (~200 days). Used only to describe gaps, not to assign labels.
OFFSEASON_GAP_DAYS = 56


def AsDate(value) -> dt.date:
    """Coerce a date, datetime, pandas Timestamp, numpy datetime64, or ISO string."""
    if value is None:
        raise TypeError("date value is None")
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if hasattr(value, "date") and not isinstance(value, (str, bytes)):
        try:
            out = value.date()
            if isinstance(out, dt.date):
                return out
        except Exception:
            pass
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" in text:
        return dt.datetime.fromisoformat(text).date()
    return dt.date.fromisoformat(text[:10])


def NflSeasonLabel(value) -> str:
    """Return the NFL season year ('2024') for a game date (Aug-Feb = kickoff year)."""
    day = AsDate(value)
    if day.month >= 8:
        return str(day.year)
    return str(day.year - 1)


# Back-compat alias used in a few call sites that just want "the season label".
SeasonLabel = NflSeasonLabel


def DrawdownPct(equity) -> list[float]:
    """Running drawdown as a negative fraction of the running peak (0 if flat/up)."""
    out = []
    peak = None
    for raw in equity:
        val = float(raw)
        peak = val if peak is None or val > peak else peak
        if peak == 0:
            out.append(0.0)
        else:
            dd = val / peak - 1.0
            out.append(dd if dd < 0 else 0.0)
    return out


def EndOfDayEquity(dates, equity, series=None):
    """Keep the last equity value on each (series, date), preserving series order."""
    last = {}
    order = []
    n = len(dates)
    if len(equity) != n:
        raise ValueError("dates and equity must be the same length")
    if series is None:
        series = [None] * n
    elif len(series) != n:
        raise ValueError("series must be the same length as dates")
    for i in range(n):
        key = (series[i], AsDate(dates[i]))
        if key not in last:
            order.append(key)
        last[key] = float(equity[i])
    out_dates = [k[1] for k in order]
    out_eq = [last[k] for k in order]
    out_series = [k[0] for k in order]
    return out_dates, out_eq, out_series


def SeasonWindows(dates) -> list[dict]:
    """Unique seasons in order, each with first and last game date."""
    windows = []
    index = {}
    for value in dates:
        day = AsDate(value)
        label = NflSeasonLabel(day)
        if label not in index:
            index[label] = len(windows)
            windows.append({"season": label, "first": day, "last": day})
        else:
            slot = windows[index[label]]
            if day < slot["first"]:
                slot["first"] = day
            if day > slot["last"]:
                slot["last"] = day
    return windows


def _SeriesVals(dates, equity, series):
    dates = list(dates)
    equity = [float(v) for v in equity]
    if len(dates) != len(equity):
        raise ValueError("dates and equity must be the same length")
    if series is None:
        series_vals = ["bankroll"] * len(dates)
    elif isinstance(series, (str, bytes)):
        series_vals = [str(series)] * len(dates)
    else:
        series_vals = [str(v) for v in series]
        if len(series_vals) != len(dates):
            raise ValueError("series must be the same length as dates")
    return dates, equity, series_vals


def FrameBankroll(dates, equity, series=None, mode="auto"):
    """Build a chart frame that does not leave flat offseason stretches.

    mode:
      - 'single': calendar dates, first game to last game of the one season.
      - 'index': shared game-day index (unique dates, sorted) with a marker at
        each new season. The same calendar date has the same x on every series.
      - 'cut_calendar': intra-season calendar spacing, offseason days removed.
      - 'auto': 'single' if every point is in one season, else 'index'.

    Returns a dict with rows (date, series, equity, drawdown_pct, season, x,
    x_kind), season windows, boundary markers, and the mode actually used.
    """
    dates, equity, series_vals = _SeriesVals(dates, equity, series)

    points = sorted(
        (
            (AsDate(d), i, series_vals[i], equity[i])
            for i, d in enumerate(dates)
        ),
        key=lambda row: (row[2], row[0], row[1]),
    )
    if not points:
        return {
            "rows": [],
            "seasons": [],
            "windows": [],
            "boundaries": [],
            "mode": "single" if mode in ("auto", "single") else mode,
            "x_kind": "date" if mode != "index" else "index",
            "first_date": None,
            "last_date": None,
        }

    seasons = []
    seen = set()
    for day, _i, _s, _e in points:
        label = NflSeasonLabel(day)
        if label not in seen:
            seen.add(label)
            seasons.append(label)

    if mode == "auto":
        mode_used = "single" if len(seasons) == 1 else "index"
    else:
        if mode not in ("single", "index", "cut_calendar"):
            raise ValueError(f"unknown mode: {mode}")
        if mode == "single" and len(seasons) > 1:
            raise ValueError(
                "mode='single' needs one season; got "
                + ", ".join(seasons)
                + ". Use mode='index' or 'cut_calendar', or split by season."
            )
        mode_used = mode

    # Per-series running drawdown on the sorted path (no offseason fill).
    by_series_eq = defaultdict(list)
    draft = []
    for day, _i, ser, eq in points:
        by_series_eq[ser].append(eq)
        draft.append((day, ser, eq, NflSeasonLabel(day)))
    dd_iters = {ser: iter(DrawdownPct(vals)) for ser, vals in by_series_eq.items()}

    unique_days = sorted({day for day, _ser, _eq, _label in draft})

    rows = []
    if mode_used == "index":
        day_x = {day: i for i, day in enumerate(unique_days)}
        for day, ser, eq, label in draft:
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": day_x[day],
                "x_kind": "index",
            })
        x_kind = "index"
    elif mode_used == "cut_calendar":
        compact = {}
        compact[unique_days[0]] = 0
        cursor = 0
        for prev, cur in zip(unique_days, unique_days[1:]):
            gap = (cur - prev).days
            if gap >= OFFSEASON_GAP_DAYS:
                cursor += 1
            else:
                cursor += gap
            compact[cur] = cursor
        for day, ser, eq, label in draft:
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": compact[day],
                "x_kind": "cut_calendar",
            })
        x_kind = "cut_calendar"
    else:
        for day, ser, eq, label in draft:
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": day.isoformat(),
                "x_kind": "date",
            })
        x_kind = "date"

    boundaries = []
    if mode_used != "single":
        first_x = {}
        for row in rows:
            if row["season"] not in first_x:
                first_x[row["season"]] = row["x"]
        for label in seasons[1:]:
            boundaries.append({
                "series": None,
                "season": label,
                "x": first_x[label],
                "date": next(r["date"] for r in rows if r["season"] == label),
            })

    return {
        "rows": rows,
        "seasons": seasons,
        "windows": SeasonWindows([p[0] for p in points]),
        "boundaries": boundaries,
        "mode": mode_used,
        "x_kind": x_kind,
        "first_date": min(p[0] for p in points).isoformat(),
        "last_date": max(p[0] for p in points).isoformat(),
    }


def SplitBySeason(dates, equity, series=None) -> dict:
    """Return FrameBankroll(..., mode='single') for each season, keyed by label."""
    dates, equity, series_vals = _SeriesVals(dates, equity, series)
    buckets = defaultdict(lambda: {"dates": [], "equity": [], "series": []})
    for i, value in enumerate(dates):
        label = NflSeasonLabel(value)
        buckets[label]["dates"].append(value)
        buckets[label]["equity"].append(equity[i])
        buckets[label]["series"].append(series_vals[i])
    return {
        label: FrameBankroll(b["dates"], b["equity"], b["series"], mode="single")
        for label, b in buckets.items()
    }


def TidyRows(frame) -> list[dict]:
    """Rows ready for a Graphtard-style tidy CSV (date, series, equity, drawdown_pct, ...)."""
    return [dict(row) for row in frame["rows"]]


def FrameTidy(rows, date_key="date", equity_key="equity", series_key="series", mode="auto"):
    """FrameBankroll from a list of tidy dicts (existing AgentDocs CSVs)."""
    rows = list(rows)
    if not rows:
        return FrameBankroll([], [], mode=mode)
    return FrameBankroll(
        [r[date_key] for r in rows],
        [r[equity_key] for r in rows],
        [r[series_key] for r in rows],
        mode=mode,
    )


def PlotBankroll(frame, path=None, title=None, ylabel="Bankroll", ax=None,
                 figsize=(11, 6.5), logy=False, hlines=None, legend=True,
                 xlabel=None):
    """Optional matplotlib renderer. Returns the Axes. Saves PNG if path is set."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, AutoDateLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed in this interpreter; "
            "write the tidy CSV and render with a venv that has it"
        ) from exc

    created = ax is None
    fig = None
    if created:
        fig, ax = plt.subplots(figsize=figsize)

    groups = defaultdict(list)
    for row in frame["rows"]:
        groups[row["series"]].append(row)

    kind = frame["x_kind"]
    for ser, rows in groups.items():
        xs = []
        ys = []
        for row in rows:
            if kind == "date":
                xs.append(AsDate(row["x"]))
            else:
                xs.append(row["x"])
            ys.append(row["equity"])
        ax.plot(xs, ys, lw=1.8, label=ser)

    if logy:
        ax.set_yscale("log")

    for y, lab, *rest in (hlines or []):
        style = rest[0] if rest else {}
        kw = {"color": "#333333", "lw": 1, "ls": ":", "alpha": 0.8}
        kw.update(style)
        ax.axhline(y, **kw)
        if lab:
            ax.text(
                0.01, y, f"  {lab}", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=8, color="#333333",
            )

    for bound in frame.get("boundaries") or []:
        ax.axvline(bound["x"], color="#555555", lw=1, ls="--", alpha=0.7)
        ymax = ax.get_ylim()[1]
        ax.text(
            bound["x"], ymax, f"  {bound['season']}",
            rotation=90, va="top", ha="left", fontsize=9, color="#333333",
        )

    if kind == "date":
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.set_xlabel(xlabel or "Game date")
        if frame.get("first_date") and frame.get("last_date"):
            ax.set_xlim(AsDate(frame["first_date"]), AsDate(frame["last_date"]))
    elif kind == "index":
        ax.set_xlabel(xlabel or "Game-day index (offseason omitted; dashed line = new season)")
    else:
        ax.set_xlabel(xlabel or "Compacted calendar days (summer gaps cut; dashed line = new season)")

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if legend and len(groups) > 1:
        legend_kw = legend if isinstance(legend, dict) else {}
        ax.legend(frameon=False, **legend_kw)
    ax.grid(alpha=0.3)
    if fig is not None:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
    return ax