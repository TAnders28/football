import unittest

from Backtest.PolyReality import POLY_FEES, PriceAt

KO = "2025-09-07T17:00:00Z"
KO_TS = 1757264400


def Row(hist, token0="KC", home="KC"):
    return {"kickoff_utc": KO, "history": hist, "token0_team": token0, "home_team": home}


class PolyRealityTest(unittest.TestCase):
    def test_price_at_uses_last_point_before_t(self):
        h = [{"t": KO_TS - 100 * 60, "p": 0.6}, {"t": KO_TS - 95 * 60, "p": 0.62}, {"t": KO_TS - 80 * 60, "p": 0.7}]
        ph, pa, age = PriceAt(Row(h), 90)
        self.assertEqual(ph, 0.62)
        self.assertAlmostEqual(pa, 0.38)
        self.assertEqual(age, 5.0)

    def test_price_at_flips_when_token0_is_away(self):
        ph, pa, _ = PriceAt(Row([{"t": KO_TS - 91 * 60, "p": 0.3}], token0="BAL"), 90)
        self.assertAlmostEqual(ph, 0.7)
        self.assertAlmostEqual(pa, 0.3)

    def test_price_at_stale_or_missing(self):
        self.assertIsNone(PriceAt(Row([{"t": KO_TS - 200 * 60, "p": 0.5}]), 90))
        self.assertIsNone(PriceAt(Row([{"t": KO_TS - 10 * 60, "p": 0.5}]), 90))
        self.assertIsNone(PriceAt(Row([]), 90))
        self.assertIsNone(PriceAt(None, 90))

    def test_poly_fee(self):
        self.assertAlmostEqual(POLY_FEES["polymarket_sports_0.05"](0.5), 0.0125)
        self.assertAlmostEqual(POLY_FEES["polymarket_sports_0.05"](0.9), 0.0045)


if __name__ == "__main__":
    unittest.main()