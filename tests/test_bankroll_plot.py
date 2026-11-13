"""Tests for season-aware bankroll chart framing (no matplotlib required)."""

import datetime as dt
import unittest

from Backtest.BankrollPlot import (
    AsDate,
    DrawdownPct,
    EndOfDayEquity,
    FrameBankroll,
    FrameTidy,
    NflSeasonLabel,
    SeasonWindows,
    SplitBySeason,
)


class SeasonLabelTests(unittest.TestCase):
    def test_season_spans_calendar_year(self):
        self.assertEqual(NflSeasonLabel("2024-09-05"), "2024")
        self.assertEqual(NflSeasonLabel("2025-02-09"), "2024")
        self.assertEqual(NflSeasonLabel("2024-08-15"), "2024")
        self.assertEqual(NflSeasonLabel(dt.date(2025, 9, 4)), "2025")
        self.assertEqual(NflSeasonLabel("2026-02-08"), "2025")
        self.assertEqual(NflSeasonLabel("2026-09-09"), "2026")
        self.assertEqual(NflSeasonLabel("2026-11-09"), "2026")

    def test_as_date_accepts_iso_and_datetime(self):
        self.assertEqual(AsDate("2025-02-09T18:30:00-05:00"), dt.date(2025, 2, 9))
        self.assertEqual(AsDate(dt.datetime(2025, 2, 9, 23, 0, 0)), dt.date(2025, 2, 9))


class FrameTests(unittest.TestCase):
    def test_single_season_uses_calendar_and_stops_at_last_game(self):
        dates = ["2024-09-05", "2024-12-25", "2025-02-09"]
        equity = [10000.0, 9800.0, 10100.0]
        frame = FrameBankroll(dates, equity, mode="auto")
        self.assertEqual(frame["mode"], "single")
        self.assertEqual(frame["x_kind"], "date")
        self.assertEqual(frame["seasons"], ["2024"])
        self.assertEqual(frame["first_date"], "2024-09-05")
        self.assertEqual(frame["last_date"], "2025-02-09")
        self.assertEqual(frame["boundaries"], [])
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs, ["2024-09-05", "2024-12-25", "2025-02-09"])
        # No invented offseason points between February and September.
        self.assertTrue(all(not row["date"][5:].startswith("06-")
                            and not row["date"][5:].startswith("07-")
                            for row in frame["rows"]))

    def test_single_mode_rejects_two_seasons(self):
        dates = ["2024-09-05", "2025-02-09", "2025-09-04"]
        with self.assertRaises(ValueError):
            FrameBankroll(dates, [1, 2, 3], mode="single")

    def test_multi_season_index_skips_summer_and_marks_boundaries(self):
        dates = [
            "2023-09-07", "2024-01-10", "2024-02-11",
            "2024-09-05", "2025-02-09",
            "2025-09-04", "2026-02-08",
            "2026-09-09", "2026-11-09",
        ]
        equity = [100, 110, 105, 105, 90, 90, 80, 80, 70]
        frame = FrameBankroll(dates, equity, series="official", mode="auto")
        self.assertEqual(frame["mode"], "index")
        self.assertEqual(frame["seasons"], ["2023", "2024", "2025", "2026"])
        self.assertEqual(frame["last_date"], "2026-11-09")
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs, list(range(9)))
        # The Feb-to-Sep gaps are not represented as x jumps of ~200 days.
        self.assertEqual(xs[-1] - xs[0], 8)
        bounds = {b["season"]: b["x"] for b in frame["boundaries"]}
        self.assertEqual(bounds["2024"], 3)
        self.assertEqual(bounds["2025"], 5)
        self.assertEqual(bounds["2026"], 7)
        self.assertNotIn("2023", bounds)

    def test_shared_index_aligns_series_on_the_same_date(self):
        dates = ["2024-09-05", "2025-02-09", "2025-09-04", "2024-09-05", "2025-09-04"]
        equity = [10, 12, 11, 20, 22]
        series = ["a", "a", "a", "b", "b"]
        frame = FrameBankroll(dates, equity, series, mode="index")
        by = {(r["series"], r["date"]): r["x"] for r in frame["rows"]}
        self.assertEqual(by[("a", "2024-09-05")], by[("b", "2024-09-05")])
        self.assertEqual(by[("a", "2025-09-04")], by[("b", "2025-09-04")])
        self.assertEqual(by[("a", "2025-02-09")], 1)
        self.assertEqual(by[("b", "2025-09-04")], 2)
        # One shared marker, not one per series.
        self.assertEqual([b["season"] for b in frame["boundaries"]], ["2025"])
        self.assertEqual(frame["boundaries"][0]["x"], 2)

    def test_cut_calendar_removes_offseason_span(self):
        dates = ["2025-02-09", "2025-09-04", "2025-09-06"]
        equity = [100.0, 100.0, 99.0]
        naive_gap = (dt.date(2025, 9, 4) - dt.date(2025, 2, 9)).days
        self.assertGreater(naive_gap, 100)
        frame = FrameBankroll(dates, equity, mode="cut_calendar")
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs[0], 0)
        # Summer is cut to a one-step boundary, then Sep 4->6 keeps 2 calendar days.
        self.assertEqual(xs[1], 1)
        self.assertEqual(xs[2], 3)
        self.assertLess(xs[2], naive_gap)
        self.assertEqual(len(frame["boundaries"]), 1)
        self.assertEqual(frame["boundaries"][0]["season"], "2025")

    def test_split_by_season_ends_at_each_seasons_last_game(self):
        dates = ["2024-09-05", "2025-02-09", "2025-09-04", "2026-02-08"]
        equity = [10, 12, 12, 8]
        parts = SplitBySeason(dates, equity, series="path")
        self.assertEqual(set(parts), {"2024", "2025"})
        self.assertEqual(parts["2024"]["first_date"], "2024-09-05")
        self.assertEqual(parts["2024"]["last_date"], "2025-02-09")
        self.assertEqual(parts["2025"]["first_date"], "2025-09-04")
        self.assertEqual(parts["2025"]["last_date"], "2026-02-08")
        self.assertEqual(parts["2024"]["mode"], "single")

    def test_drawdown_and_end_of_day(self):
        self.assertEqual(DrawdownPct([100, 120, 90, 90]), [0.0, 0.0, -0.25, -0.25])
        dates, eq, ser = EndOfDayEquity(
            ["2025-01-01", "2025-01-01", "2025-01-02"],
            [100.0, 95.0, 97.0],
            ["a", "a", "a"],
        )
        self.assertEqual([d.isoformat() for d in dates], ["2025-01-01", "2025-01-02"])
        self.assertEqual(eq, [95.0, 97.0])

    def test_empty_frame(self):
        frame = FrameBankroll([], [])
        self.assertEqual(frame["rows"], [])
        self.assertEqual(frame["seasons"], [])

    def test_season_windows(self):
        windows = SeasonWindows(["2024-09-05", "2025-02-09", "2025-09-04"])
        self.assertEqual(windows[0]["season"], "2024")
        self.assertEqual(windows[0]["first"], dt.date(2024, 9, 5))
        self.assertEqual(windows[0]["last"], dt.date(2025, 2, 9))
        self.assertEqual(windows[1]["season"], "2025")

    def test_frame_tidy_keeps_equity_values(self):
        rows = [
            {"date": "2025-09-04", "series": "a", "equity": 10000},
            {"date": "2026-02-08", "series": "a", "equity": 20210.61},
            {"date": "2026-09-09", "series": "a", "equity": 20210.61},
            {"date": "2026-11-09", "series": "a", "equity": 20210.61},
        ]
        frame = FrameTidy(rows)
        self.assertEqual(frame["mode"], "index")
        self.assertEqual(frame["rows"][-1]["equity"], 20210.61)
        self.assertEqual(frame["last_date"], "2026-11-09")
        self.assertEqual(frame["boundaries"][0]["season"], "2026")


if __name__ == "__main__":
    unittest.main()