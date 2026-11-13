import unittest

import numpy as np

from Backtest.InGameCalibration import BinOf, Calibrate, InPlayGrid, Sample, SeriesKalshi, SeriesPoly, Ts, Wilson


class InGameCalibrationTest(unittest.TestCase):
    def test_ts_parses_utc_z(self):
        self.assertEqual(Ts("2025-09-07T17:02:38.000Z"), 1757264558.0)
        self.assertIsNone(Ts("17:02:38"))

    def test_grid_skips_halftime_and_tags_quarter(self):
        w = {"kickoff": 0, "end": 600, "excluded": [(200, 400)], "plays": [(0, 1), (150, 2), (400, 3), (500, 5)]}
        g = InPlayGrid(w)
        ts = [t for t, _ in g]
        self.assertNotIn(300, ts)
        self.assertIn(180, ts)
        self.assertEqual(dict(g)[180], 2)
        self.assertEqual(dict(g)[420], 3)
        self.assertEqual(dict(g)[540], 5)

    def test_sample_staleness(self):
        out = Sample([(0, 0.4), (100, 0.6)], [50, 100, 500])
        self.assertEqual(out, [(50.0, 0.4), (100.0, 0.6)])

    def test_series_home_side(self):
        k = {"side_team": "KC", "home_team": "KC", "candles": [{"end_ts": 1, "yes_bid_close": 0.6, "yes_ask_close": 0.62},
                                                               {"end_ts": 2, "yes_bid_close": None, "yes_ask_close": 0.5}]}
        self.assertEqual(len(SeriesKalshi(k)), 1)
        self.assertAlmostEqual(SeriesKalshi(k)[0][1], 0.61)
        p = {"token0_team": "BAL", "home_team": "KC", "history": [{"t": 1, "p": 0.3}]}
        self.assertAlmostEqual(SeriesPoly(p)[0][1], 0.7)

    def test_bins_and_wilson(self):
        self.assertEqual(BinOf(0.0), 0)
        self.assertEqual(BinOf(0.999), 9)
        self.assertEqual(BinOf(1.0), 9)
        lo, hi = Wilson(50, 100)
        self.assertTrue(lo < 0.5 < hi)

    def test_calibrate_cluster_wider_than_naive(self):
        rng = np.random.default_rng(1)
        samples = []
        for g in range(60):
            y = int(rng.random() < 0.65)
            samples += [(g, 0.65, y)] * 50
        r = Calibrate(samples, 300, rng)
        b = r["bins"][6]
        self.assertEqual(b["n_games"], 60)
        self.assertEqual(b["n_samples"], 3000)
        self.assertGreater(b["cluster_ci90_hi"] - b["cluster_ci90_lo"], 2 * (b["naive_ci90_hi"] - b["naive_ci90_lo"]))
        self.assertAlmostEqual(b["obs_freq"], b["gw_obs_freq"])


class PlotGameProbsTest(unittest.TestCase):
    def test_eastern_label(self):
        from Backtest.PlotGameProbs import Eastern
        dt, lab = Eastern(1757264558)  # 2025-09-07 17:02:38Z -> 13:02 EDT
        self.assertEqual((dt.hour, dt.minute, lab), (13, 2, "ET"))
        dt, _ = Eastern(1765126800)  # 2025-12-07 17:00Z -> 12:00 EST
        self.assertEqual(dt.hour, 12)


if __name__ == "__main__":
    unittest.main()