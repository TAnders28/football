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


if __name__ == "__main__":
    unittest.main()