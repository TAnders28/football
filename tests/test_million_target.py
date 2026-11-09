"""MillionTarget: cash reservation, target stop, ruin/zero flags, Compound parity, nested isolation (synthetic)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from SyntheticNfl import Perturb, SyntheticSeason  # noqa: E402

import Backtest.MillionTarget as MT  # noqa: E402
from Backtest.Features import BuildRows  # noqa: E402
from Backtest.Lab import START, AttachModels  # noqa: E402

LAST = 2016
EVAL = [2015, 2016]


def RowsBySeason(replace=None):
    replace = replace or {}
    out, prev = {}, []
    for s in range(2012, LAST + 1):
        games = replace.get(s) or SyntheticSeason(s)
        out[s] = BuildRows(games, prev)
        prev = games
    AttachModels(out)
    return out


def Day(k, r, tier=None, date="2020-01-01"):
    return {"date": date, "season": 2020, "k": k, "r": r, "tier": tier or [0] * len(k), "game_ids": [str(i) for i in range(len(k))]}


class MillionTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rbs = RowsBySeason()

    def test_audit_compound_reservation_and_parity(self):
        for row in MT.AuditCompound(self.rbs, EVAL):
            self.assertLessEqual(row["max_day_exposure"], 1.0001, row)
            self.assertLess(row["max_ledger_gap"], 0.1, row)
            self.assertFalse(row["negative_cash"], row)
            self.assertAlmostEqual(row["compound_end"], row["vectorised_end"], delta=max(1.0, 1e-4 * row["compound_end"]))

    def test_same_day_stakes_scaled_to_cash_and_never_negative(self):
        pack = MT.Pack([Day([0, 0, 0], [-1.0, -1.0, -1.0])])
        cfg = MT.Cfg("flat50", "flat", "flat", 0.5)
        f = MT.Fractions(cfg, np.array([START]), 0, 1, pack["K"][:1], pack["M"][:1], pack["TIER"][:1])
        self.assertAlmostEqual(float(f.sum()), 1.0)
        res = MT.Simulate(pack, np.zeros((1, 1), int), cfg)
        self.assertEqual(float(res["end"][0]), 0.0)
        self.assertTrue(res["zero"][0] and res["ruin"][0])

    def test_same_day_outcomes_do_not_change_same_day_stakes(self):
        a = MT.Pack([Day([0.1, 0.2], [1.0, -1.0])])
        b = MT.Pack([Day([0.1, 0.2], [-1.0, 1.0])])
        cfg = MT.Cfg("k1", "over_kelly", "kelly", 1.0)
        fa = MT.Fractions(cfg, np.array([START]), 0, 1, a["K"][:1], a["M"][:1], a["TIER"][:1])
        fb = MT.Fractions(cfg, np.array([START]), 0, 1, b["K"][:1], b["M"][:1], b["TIER"][:1])
        self.assertTrue(np.allclose(fa, fb))

    def test_stop_at_target_holds_cash_and_playon_does_not(self):
        days = [Day([0], [9.0]), Day([0], [9.0]), Day([0], [-1.0]), Day([0], [-1.0])]
        pack = MT.Pack(days)
        cfg = MT.Cfg("flat100", "flat", "flat", 1.0)
        idx = np.arange(4)[None, :]
        stop = MT.Simulate(pack, idx, cfg, stop=True)
        play = MT.Simulate(pack, idx, cfg, stop=False)
        self.assertTrue(stop["hit"][0])
        self.assertEqual(float(stop["end"][0]), START * 100)
        self.assertEqual(int(stop["hit_step"][0]), 1)
        self.assertEqual(float(play["end"][0]), 0.0)
        self.assertFalse(stop["ruin"][0])

    def test_derisk_and_schedule_rules(self):
        pack = MT.Pack([Day([0], [0.0])])
        K, M, TR = pack["K"][:1], pack["M"][:1], pack["TIER"][:1]
        d = MT.Cfg("d", "target_derisk", "flat", 0.4, 0.1, rule="derisk")
        self.assertAlmostEqual(float(MT.Fractions(d, np.array([5e4]), 0, 10, K, M, TR).sum()), 0.4)
        self.assertAlmostEqual(float(MT.Fractions(d, np.array([2e5]), 0, 10, K, M, TR).sum()), 0.1)
        s = MT.Cfg("s", "target_schedule", "flat", 0.3, 0.1, rule="schedule")
        self.assertAlmostEqual(float(MT.Fractions(s, np.array([START]), 0, 10, K, M, TR).sum()), 0.1)   # on schedule at t=0
        self.assertAlmostEqual(float(MT.Fractions(s, np.array([START]), 5, 10, K, M, TR).sum()), 0.3)   # behind at t=5

    def test_common_random_numbers(self):
        self.assertTrue(np.array_equal(MT.CommonIndex(50, 20, 100), MT.CommonIndex(50, 20, 100)))

    def test_frontier_is_nondominated(self):
        k = "p_hit_before_ruin"
        rows = [{"name": "a", k: 0.1, "p_ruin": 0.5}, {"name": "b", k: 0.05, "p_ruin": 0.1},
                {"name": "c", k: 0.04, "p_ruin": 0.2}, {"name": "d", k: 0.1, "p_ruin": 0.6}]
        self.assertEqual(sorted(MT.Frontier(rows)), ["a", "b"])

    def test_nested_choice_ignores_the_season_played(self):
        grid = MT.GRID[:6]
        a = MT.Nested(self.rbs, EVAL, 2013, "closing_ml", grid=grid, n_sims=200)
        games = SyntheticSeason(LAST)
        b = MT.Nested(RowsBySeason({LAST: Perturb(games, games[0]["gameday"], seed=5, drop_after=False)}),
                      EVAL, 2013, "closing_ml", grid=grid, n_sims=200, T=None)
        self.assertEqual(a["choices"][1]["choice"], b["choices"][1]["choice"])
        self.assertEqual(a["choices"][1]["train"], [2013, 2015])

    def test_ruin_is_absorbing_first_passage_and_ever_hit_is_kept_separately(self):
        # 10k -> 500 (ruin) -> 10x days -> 5M: recovery after ruin counts only as ever-hit
        days = [Day([0], [-0.95]), Day([0], [9.0]), Day([0], [9.0]), Day([0], [9.0]), Day([0], [9.0])]
        pack = MT.Pack(days)
        cfg = MT.Cfg("flat100", "flat", "flat", 1.0)
        idx = np.arange(5)[None, :]
        first = MT.Simulate(pack, idx, cfg)
        ever = MT.Simulate(pack, idx, cfg, absorb_ruin=False)
        self.assertTrue(first["ruin"][0])
        self.assertFalse(first["hit"][0])
        self.assertAlmostEqual(float(first["end"][0]), 500.0)          # cash held at the ruin level
        self.assertTrue(ever["ruin"][0] and ever["hit"][0])
        self.assertEqual(int(ever["hit_step"][0]), 4)
        st = MT.BootStats(first, 1, ever)
        self.assertEqual((st["p_hit_before_ruin"], st["p_ruin"], st["p_hit_ever"], st["p_ruin_then_hit"]), (0.0, 1.0, 1.0, 1.0))

    def test_bootstrap_hit_and_ruin_are_mutually_exclusive_and_ever_is_a_superset(self):
        rng = np.random.default_rng(3)
        days = [Day(list(rng.uniform(0, 0.3, 3)), list(rng.choice([-1.0, 1.2], 3))) for _ in range(30)]
        pack = MT.Pack(days)
        idx = MT.CommonIndex(30, 120, 2000)
        cfg = MT.Cfg("k4", "over_kelly", "kelly", 4.0)
        first = MT.Simulate(pack, idx, cfg)
        ever = MT.Simulate(pack, idx, cfg, absorb_ruin=False)
        self.assertFalse((first["hit"] & first["ruin"]).any())
        self.assertTrue(np.array_equal(first["ruin"], ever["ruin"]))        # ruin-before-hit is the same event
        self.assertTrue(np.array_equal(first["hit"], ever["hit"] & ~ever["ruin"]))
        self.assertTrue(first["ruin"].any() and first["hit"].any())
        st = MT.BootStats(first, 2000, ever)
        self.assertAlmostEqual(st["p_hit_before_ruin"] + st["p_ruin"] + st["p_neither"], 1.0, places=3)
        self.assertAlmostEqual(st["p_hit_ever"] - st["p_hit_before_ruin"], st["p_ruin_then_hit"], places=3)

    def test_grid_is_finite_and_small(self):
        self.assertLessEqual(len(MT.GRID), 150)
        self.assertEqual(len({c["name"] for c in MT.GRID}), len(MT.GRID))


if __name__ == "__main__":
    unittest.main()