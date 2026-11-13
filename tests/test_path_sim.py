import unittest

import numpy as np

from Backtest.PathSim import DailyReturns, DayOrders, Evaluate, RunPaths, START


class PathSimTests(unittest.TestCase):
    def test_orders_shapes_and_ranges(self):
        rng = np.random.default_rng(1)
        for m in ("random_start", "block_bootstrap", "shuffle"):
            idx = DayOrders(10, 7, 25, m, rng, block=3)
            self.assertEqual(idx.shape, (7, 25))
            self.assertTrue(((idx >= 0) & (idx < 10)).all())
        sh = DayOrders(10, 3, 10, "shuffle", np.random.default_rng(2))
        for row in sh:
            self.assertEqual(sorted(row.tolist()), list(range(10)))
        rs = DayOrders(10, 5, 4, "random_start", np.random.default_rng(3))
        self.assertTrue(((np.diff(rs, axis=1) % 10) == 1).all())

    def test_seeded_and_deterministic(self):
        r = [0.05, -0.04, 0.0, 0.1, -0.2, 0.03]
        a = Evaluate(r, seed=7, paths=200)
        b = Evaluate(r, seed=7, paths=200)
        self.assertEqual(a, b)
        self.assertNotEqual(a["randomized"], Evaluate(r, seed=8, paths=200)["randomized"])

    def test_real_sequence_compounds(self):
        out = Evaluate([0.1, -0.5, 0.2], paths=10)
        self.assertAlmostEqual(out["real_sequence"]["median_final"], round(START * 1.1 * 0.5 * 1.2, 2))

    def test_ruin_absorbing_vs_restart(self):
        r = np.array([-0.95, 0.5])
        idx = np.array([[0, 1, 1, 1]])
        a = RunPaths(r, idx)
        self.assertTrue(a["ruin"][0])
        self.assertAlmostEqual(a["final"][0], START * 0.05)
        self.assertEqual(a["t_ruin"][0], 1)
        b = RunPaths(r, idx, restart=True)
        self.assertEqual(b["restarts"][0], 1)
        self.assertAlmostEqual(b["final"][0], START * 1.5 ** 3)

    def test_bad_start_restart_and_target(self):
        r = np.array([-0.1, 3.0])
        b = RunPaths(r, np.array([[0, 0, 1, 1, 1]]), restart=True, bad_start_days=2, bad_start_level=0.9)
        self.assertEqual(b["restarts"][0], 1)
        hit = RunPaths(np.array([200.0]), np.array([[0, 0]]))
        self.assertTrue(hit["hit"][0])
        self.assertEqual(hit["t_hit"][0], 1)

    def test_daily_returns_with_season_reset(self):
        daily = [{"date": "2025-09-07", "equity": 11000}, {"date": "2025-09-14", "equity": 9900},
                 {"date": "2026-09-13", "equity": 10500}]
        r = DailyReturns(daily, season_of=lambda d: d[:4])
        self.assertAlmostEqual(r[0], 0.1)
        self.assertAlmostEqual(r[1], -0.1)
        self.assertAlmostEqual(r[2], 0.05)


if __name__ == "__main__":
    unittest.main()