import unittest

from Backtest.SnapshotPrices import KalshiEventMatch, Row, SideTeam


class SnapshotPriceTests(unittest.TestCase):
    def test_kalshi_event_match(self):
        g = {"gameday": "2026-11-15", "away_team": "NE", "home_team": "DET"}
        self.assertTrue(KalshiEventMatch("KXNFLGAME-26NOV15NEDET", g))
        self.assertFalse(KalshiEventMatch("KXNFLGAME-26NOV16NEDET", g))
        self.assertFalse(KalshiEventMatch("KXNFLGAME-26NOV15DETNE", g))
        rams = {"gameday": "2026-11-15", "away_team": "LA", "home_team": "ARI"}
        self.assertTrue(KalshiEventMatch("KXNFLGAME-26NOV15LAARI", rams))
        self.assertTrue(KalshiEventMatch("KXNFLGAME-26NOV15LARARI", rams))
        self.assertEqual(SideTeam("KXNFLGAME-26NOV15LARARI-LAR", rams), "LA")

    def test_row_mid_spread_and_no_invented_time(self):
        bids = {"best": {"price": 0.57, "size": 10.0}, "top": []}
        asks = {"best": {"price": 0.59, "size": 5.0}, "top": []}
        r = Row({"game_id": "x"}, "kalshi", "T", "DET", bids, asks, "0.58", "1", None, None, "active", None)
        self.assertAlmostEqual(r["mid"], 0.58)
        self.assertAlmostEqual(r["spread"], 0.02)
        self.assertIsNone(r["source_time"])
        empty = Row({"game_id": "x"}, "kalshi", "T", "DET", {"best": None}, asks, None, None, None, None, "active", None)
        self.assertIsNone(empty["mid"])


class AutoWeekAndLockTests(unittest.TestCase):
    def test_auto_week_is_next_unstarted(self):
        from datetime import datetime, timezone
        from unittest import mock
        import Backtest.SnapshotPrices as SP
        games = [{"week": 9, "gameday": "2026-11-08", "gametime": "13:00"}, {"week": 10, "gameday": "2026-11-15", "gametime": "13:00"},
                 {"week": 11, "gameday": "2026-11-22", "gametime": "13:00"}]
        with mock.patch.object(SP, "LoadSeason", return_value=games):
            self.assertEqual(SP.AutoSeasonWeek(datetime(2026, 11, 13, 15, tzinfo=timezone.utc)), (2026, 10))
            self.assertEqual(SP.AutoSeasonWeek(datetime(2027, 1, 30, tzinfo=timezone.utc)), (2026, None))

    def test_runner_lock_is_exclusive(self):
        import os, tempfile
        from Backtest.SnapshotRunner import Lock
        path = os.path.join(tempfile.mkdtemp(), "x.lock")
        with Lock(path) as a:
            self.assertTrue(a)
            with Lock(path) as b:
                self.assertFalse(b)
        with Lock(path) as c:
            self.assertTrue(c)

if __name__ == "__main__":
    unittest.main()