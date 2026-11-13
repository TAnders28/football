import unittest

import numpy as np

from Backtest.InGameRuleSim import Bankroll, KalshiQuotes, Price, Rules, Stats, Trigger


def Row(t, qtr, mid, model, half=0.01):
    return {"t": t, "qtr": qtr, "mid": mid, "bid": mid - half, "ask": mid + half, "model": model}


class RuleSimTest(unittest.TestCase):
    def test_rule_count(self):
        self.assertEqual(len(Rules()), 13)

    def test_close_rule_first_trigger_and_side(self):
        r = Rules()["close_40_60_gap5"]
        rows = [Row(0, 1, 0.70, 0.80), Row(60, 1, 0.50, 0.53), Row(120, 2, 0.50, 0.42), Row(180, 2, 0.50, 0.60)]
        hit, side = Trigger(r, rows)
        self.assertEqual((hit["t"], side), (120, "away"))
        self.assertIsNone(Trigger(r, [Row(0, 3, 0.5, 0.9)]))   # Q3 is not early

    def test_swing_follow_fade_agree(self):
        rows = [Row(0, 1, 0.50, 0.50), Row(300, 1, 0.62, 0.55)]
        rs = Rules()
        self.assertEqual(Trigger(rs["swing_follow_10_any"], rows)[1], "home")
        self.assertEqual(Trigger(rs["swing_fade_10_any"], rows)[1], "away")
        self.assertEqual(Trigger(rs["swing_fade_10_agree"], rows)[1], "away")   # model 0.55 < mid 0.62 agrees with away
        self.assertIsNone(Trigger(rs["swing_follow_10_agree"], rows))
        self.assertIsNone(Trigger(rs["swing_follow_15_any"], rows))

    def test_fade_home_band(self):
        r = Rules()["fade_home_30_50"]
        self.assertEqual(Trigger(r, [Row(0, 4, 0.55, 0.5), Row(60, 4, 0.45, 0.5)])[1], "away")
        self.assertIsNone(Trigger(r, [Row(0, 1, 0.50, 0.5)]))

    def test_prices_and_quotes(self):
        r = Row(0, 1, 0.40, 0.5, half=0.02)
        self.assertAlmostEqual(Price(r, "home", "taker"), 0.42)
        self.assertAlmostEqual(Price(r, "away", "taker"), 0.62)
        self.assertAlmostEqual(Price(r, "away", "mid"), 0.60)
        q = KalshiQuotes({"side_team": "BUF", "home_team": "KC", "candles": [{"end_ts": 5, "yes_bid_close": 0.30, "yes_ask_close": 0.32}]})
        self.assertAlmostEqual(q[0][1], 0.68); self.assertAlmostEqual(q[0][2], 0.70)

    def test_stats_and_bankroll(self):
        rng = np.random.default_rng(0)
        bets = [{"date": "2025-09-07", "won": 1, "mid_side": 0.5, "ask_side": 0.51, "model_side": 0.6},
                {"date": "2025-09-14", "won": 0, "mid_side": 0.5, "ask_side": 0.51, "model_side": 0.6}]
        s = Stats(bets, "mid", "tim_est_0.02", 200, rng)
        self.assertEqual(s["bets"], 2)
        self.assertAlmostEqual(s["cents_per_contract"], -2.0)
        c = Bankroll(bets, "mid", "tim_est_0.02", "flat")
        self.assertAlmostEqual(c[0]["equity"], 10000 + 200 / 0.52 * 0.48, places=1)
        self.assertLess(c[1]["drawdown_pct"], 0)


if __name__ == "__main__":
    unittest.main()