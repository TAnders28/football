"""Continuous compounding engine, nested walk-forward and bootstrap on synthetic seasons (no network, no data/)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from SyntheticNfl import Perturb, SyntheticSeason  # noqa: E402

import Backtest.Compound as C  # noqa: E402
from Backtest.Features import BuildRows  # noqa: E402
from Backtest.Lab import START, AttachModels  # noqa: E402

LAST = 2016
SMALL_GRID = [("value_p_full2_e4", "flat2pct"), ("value_p_full2_e8", "kelly0.5_cap10"), ("frozen_union", "flat5pct"),
              ("market_fav", "flat2pct")]


def RowsBySeason(replace=None):
    replace = replace or {}
    out, prev = {}, []
    for s in range(2012, LAST + 1):
        games = replace.get(s) or SyntheticSeason(s)
        out[s] = BuildRows(games, prev)
        prev = games
    AttachModels(out)
    return out


class CompoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rbs = RowsBySeason()

    def test_bankroll_carries_over_season_boundary(self):
        res = C.Compound(self.rbs, [2015, 2016], C.Plan([2015, 2016], "market_fav", "flat2pct"))
        first16 = next(d for d in res["daily"] if d["season"] == 2016)
        last15 = [d for d in res["daily"] if d["season"] == 2015][-1]
        self.assertEqual(first16["bank_start"], last15["equity"])
        self.assertNotEqual(first16["bank_start"], START)

    def test_no_bet_stays_flat_and_never_ruins(self):
        res = C.Compound(self.rbs, [2015, 2016], C.Plan([2015, 2016], C.SelNone(), C.SizePct(0.0)))
        self.assertEqual(res["end"], START)
        bs = C.Bootstrap(res["daily"], n_sims=200)
        self.assertEqual((bs["p_ruin"], bs["p_reach_1m"], bs["median_end"]), (0.0, 0.0, START))

    def test_no_lookahead_later_results_do_not_change_earlier_equity(self):
        games = SyntheticSeason(LAST)
        days = sorted({g["gameday"] for g in games})
        cut = days[len(days) // 2]
        changed = RowsBySeason({LAST: Perturb(games, cut, drop_after=False)})
        plan = C.Plan([2015, 2016], "value_p_full2_e4", "kelly0.5_cap10")
        a = C.Compound(self.rbs, [2015, 2016], plan)["daily"]
        b = C.Compound(changed, [2015, 2016], plan)["daily"]
        self.assertEqual([d for d in a if d["date"] < cut], [d for d in b if d["date"] < cut])

    def test_nested_choice_for_a_season_ignores_that_season(self):
        _res, ch = C.Nested(self.rbs, [2015, 2016], 2013, "closing_ml", grid=SMALL_GRID)
        games = SyntheticSeason(LAST)
        scrambled = RowsBySeason({LAST: Perturb(games, games[0]["gameday"], seed=7, drop_after=False)})
        _res2, ch2 = C.Nested(scrambled, [2015, 2016], 2013, "closing_ml", grid=SMALL_GRID)
        self.assertEqual(ch[1]["selector"], ch2[1]["selector"])
        self.assertEqual(ch[1]["sizer"], ch2[1]["sizer"])
        self.assertEqual(ch[1]["train_seasons"], [2013, 2015])

    def test_short_nested_first_season_is_no_bet(self):
        res, ch = C.Nested(self.rbs, [2015, 2016], 2013, "closing_ml", grid=SMALL_GRID, short=True)
        self.assertEqual(ch[0]["selector"], "none")
        self.assertFalse([b for b in res["bets"] if b["season"] == 2015])

    def test_kelly_caps_and_metrics(self):
        k = C.SizeKelly(1.0, 0.05)
        self.assertAlmostEqual(k({"odds": 100, "p": 0.9}), 0.05)
        self.assertEqual(k({"odds": -200, "p": 0.5}), 0.0)
        res = {"end": 5000.0, "min_bank": 4000.0, "avg_stake": 0.1, "bets": [],
               "daily": [{"date": "2020-01-01", "equity": 12000.0, "bank_start": 10000.0},
                         {"date": "2020-01-11", "equity": 4000.0, "bank_start": 12000.0},
                         {"date": "2020-01-21", "equity": 5000.0, "bank_start": 4000.0}]}
        m = C.Metrics(res, seasons_elapsed=1.0)
        self.assertAlmostEqual(m["max_drawdown_pct"], -2 / 3, places=4)
        self.assertEqual(m["longest_drawdown_days"], 20)
        self.assertAlmostEqual(m["growth_per_season"], -0.5)

    def test_bootstrap_is_seeded(self):
        res = C.Compound(self.rbs, [2015, 2016], C.Plan([2015, 2016], "market_fav", "flat5pct"))
        self.assertEqual(C.Bootstrap(res["daily"], n_sims=300), C.Bootstrap(res["daily"], n_sims=300))


if __name__ == "__main__":
    unittest.main()
