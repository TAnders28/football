import unittest
from datetime import datetime, timezone

from Backtest.KalshiReality import KalshiFee, KalshiPick, QuoteAt, FEES
from DataPull.PullKalshiHistory import NormCandle


def C(ts, bid, ask):
    return {"end_ts": ts, "yes_bid_close": bid, "yes_ask_close": ask}


class KalshiRealityTests(unittest.TestCase):
    def setUp(self):
        self.ko = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)
        k = int(self.ko.timestamp())
        self.entry = {"ko": self.ko, "candles": [C(k - 100 * 60, 0.40, 0.41), C(k - 91 * 60, 0.42, 0.43),
                                                 C(k - 89 * 60, 0.60, 0.61), C(k - 15 * 60, 0.45, 0.46)]}

    def test_quote_never_uses_later_candle(self):
        q = QuoteAt(self.entry, 90)
        self.assertEqual((q["bid"], q["ask"]), (0.42, 0.43))
        self.assertEqual(q["age_min"], 1.0)
        self.assertEqual(QuoteAt(self.entry, 15)["ask"], 0.46)

    def test_stale_quote_rejected(self):
        k = int(self.ko.timestamp())
        old = {"ko": self.ko, "candles": [C(k - 200 * 60, 0.4, 0.41)]}
        self.assertIsNone(QuoteAt(old, 90))

    def test_verified_fee_formula(self):
        self.assertAlmostEqual(KalshiFee(0.5), 0.0175)
        self.assertAlmostEqual(KalshiFee(0.2), 0.0112)
        self.assertEqual(FEES["tim_est_0.02"](0.3), 0.02)

    def test_pick_uses_ask_and_threshold(self):
        q = {"home": {"ask": 0.50, "bid": 0.49}, "away": {"ask": 0.51, "bid": 0.50}}
        self.assertIsNone(KalshiPick({"div": 0}, 0.57, 0.57, q, FEES["tim_est_0.02"], 10000))
        c, stake = KalshiPick({"div": 0}, 0.60, 0.60, q, FEES["tim_est_0.02"], 10000)
        self.assertEqual(c["side"], "home")
        self.assertGreater(stake, 0)
        self.assertLessEqual(stake, 1000)
        c, stake = KalshiPick({"div": 1}, 0.53, 0.53, q, FEES["tim_est_0.02"], 10000)
        self.assertEqual(c["rule"], "div_full3_edge2")
        self.assertAlmostEqual(stake, 200)

    def test_norm_candle_both_schemas(self):
        a = NormCandle({"end_period_ts": 1789009200, "yes_bid": {"close_dollars": "0.34"}, "yes_ask": {"close_dollars": "0.35"},
                        "price": {"close_dollars": "0.35"}, "volume_fp": "10", "open_interest_fp": "5"})
        b = NormCandle({"end_period_ts": 1753934400, "yes_bid": {"close": "0.44"}, "yes_ask": {"close": "0.45"},
                        "price": {"close": "0.45"}, "volume": "110", "open_interest": "4503"})
        self.assertEqual((a["yes_bid_close"], a["yes_ask_close"]), (0.34, 0.35))
        self.assertEqual((b["yes_bid_close"], b["yes_ask_close"], b["volume"]), (0.44, 0.45, 110.0))


if __name__ == "__main__":
    unittest.main()