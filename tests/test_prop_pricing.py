import math
import unittest

import numpy as np

from Backtest import PropPricing as P
from Backtest.DailyCheck import Dollars


class PropPricingTests(unittest.TestCase):
    def test_poisson_sf(self):
        self.assertAlmostEqual(P.PoissonSf(1, 0.5), 1 - math.exp(-0.5), places=9)
        self.assertEqual(P.PoissonSf(0, 0.5), 1.0)

    def test_kind_of(self):
        self.assertEqual(P.KindOf("KXNFLTD", 1.0), "td1")
        self.assertEqual(P.KindOf("KXNFLTD", 2.0), "td2p")
        self.assertEqual(P.KindOf("KXNFLRECYDS", 60.0), "recyds")

    def test_quote_staleness_and_validity(self):
        c = [{"end_ts": 100, "yes_bid_close": 0.4, "yes_ask_close": 0.5},
             {"end_ts": 200, "yes_bid_close": 0.0, "yes_ask_close": 0.5}]
        q = P.QuoteAt(c, 250, stale_s=1000)
        self.assertAlmostEqual(q["mid"], 0.45)
        self.assertIsNone(P.QuoteAt(c, 5000, stale_s=1000))
        self.assertIsNone(P.QuoteAt(c, 50, stale_s=1000))

    def test_projection_uses_only_prior_games(self):
        pl = P.Players.__new__(P.Players)
        pl.hist = {"x": [(2026, w, {"tds": 1 if w < 5 else 0, "receptions": 0, "receiving_yards": 0,
                                    "rushing_yards": 0, "passing_yards": 0}) for w in range(1, 10)]}
        pl._cache = {}
        prior = pl.Prior("x", 2026, 5)
        self.assertEqual(len(prior), 4)
        self.assertTrue(all(g["tds"] == 1 for g in prior))
        self.assertIsNone(P.Project(prior[:2], "td1", 1.0))
        self.assertGreater(P.Project(prior, "td1", 1.0), 0.5)

    def test_grid_size(self):
        self.assertEqual(len(P.PropConfigs()), 1152)
        self.assertEqual(len(P.MlConfigs()), 9)

    def test_edge_and_spa_shapes(self):
        bets = [{"date": f"2026-10-{d:02d}", "event_id": f"g{d}", "fill": 0.5, "result": d % 2, "split": "hold"}
                for d in range(1, 21)]
        ci = P.EdgeCis(bets, "tim", np.random.default_rng(1), boot=200)
        self.assertAlmostEqual(ci["mean_edge"], 0.5 - 0.5 - 0.02)
        self.assertLessEqual(ci["widest_95"][0], ci["mean_edge"])
        D = P.DailyMatrix([bets, bets[:5]], "hold", sorted({b["date"] for b in bets}))
        self.assertEqual(D.shape, (2, 20))
        s = P.SpaRc(D, n_boot=50)
        self.assertTrue(0 <= s["spa_p"] <= 1 and 0 <= s["rc_p"] <= 1)

    def test_norm_name(self):
        self.assertEqual(P.NormName("Odell Beckham Jr."), "odell beckham")
        self.assertEqual(P.NormName("Amon-Ra St. Brown"), "amon ra st brown")

    def test_dollars(self):
        self.assertAlmostEqual(Dollars({"yes_bid_dollars": "0.4500"}, "yes_bid"), 0.45)
        self.assertAlmostEqual(Dollars({"yes_bid": 45}, "yes_bid"), 0.45)


if __name__ == "__main__":
    unittest.main()
