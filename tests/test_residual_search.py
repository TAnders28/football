import math
import unittest

import numpy as np

from unittest.mock import patch

from Backtest.ResidualSearch import (
    GROUPS, L2_GRID, STD_GRID, TUNE, CONTAMINATED,
    ApplyOffset, AttachKalshiT60, ConfigGrid, ConfigId, FeatureNames, FitOffsetLogit,
    HaversineMiles, IgnRel, LogLossBits, PickBest, PriorRates, Rate,
    ScoreBlock, SpaPValue, WeatherVec,
)
from DataPull.BuildTeamGameEff import Accumulators, AddPlay, IsScrimmage


class GridTests(unittest.TestCase):
    def test_close_grid_has_at_least_2000(self):
        g = ConfigGrid(include_open=False, smoke=False)
        self.assertEqual(len(g), (2 ** len(GROUPS)) * len(L2_GRID) * len(STD_GRID))
        self.assertGreaterEqual(len(g), 2000)
        self.assertTrue(all(c["offset"] == "close" for c in g))
        # intercept-only (no groups) is included
        self.assertTrue(any(c["groups"] == () for c in g))

    def test_open_grid_is_separate_and_feature_names_exclude_oc_move(self):
        g = ConfigGrid(include_open=True, smoke=False)
        self.assertEqual(len(g), 2 * 2560)
        names = FeatureNames(GROUPS)
        self.assertNotIn("oc_move", names)
        self.assertNotIn("close_logit", names)
        self.assertNotIn("open_logit", names)


class FitTests(unittest.TestCase):
    def test_offset_logit_recovers_residual_slope(self):
        rng = np.random.default_rng(0)
        n = 800
        mkt = rng.uniform(0.2, 0.8, n)
        x = rng.normal(0, 1, size=(n, 1))
        # true: logit(p) = logit(mkt) + 0.8 x
        off = np.log(mkt / (1 - mkt))
        p = 1 / (1 + np.exp(-(off + 0.8 * x[:, 0])))
        y = (rng.random(n) < p).astype(float)
        w = FitOffsetLogit(x, y, off, l2=0.1)
        # intercept near 0, slope near 0.8
        self.assertEqual(len(w), 2)
        self.assertLess(abs(w[1] - 0.8), 0.25)

    def test_apply_is_sigmoid_of_offset_plus_xb(self):
        w = np.array([0.0, 1.0])
        p = ApplyOffset([[0.0], [1.0]], [0.0, 0.0], w)
        self.assertAlmostEqual(p[0], 0.5, places=6)
        self.assertGreater(p[1], 0.7)


class LeakAndSelectionTests(unittest.TestCase):
    def test_prior_rates_are_history_only(self):
        hist = [
            {"off_plays": 10, "off_epa": 2.0, "off_succ": 5, "off_yards": 40,
             "def_plays": 10, "def_epa": -1.0, "def_succ": 3, "def_yards": 20,
             "qb_id": "Q1", "qb_plays": 8, "qb_epa": 1.6},
        ]
        pr = PriorRates(hist)
        self.assertAlmostEqual(pr["epa"], 0.2 - (-0.1))
        self.assertAlmostEqual(pr["ypp"], 4.0 - 2.0)
        self.assertAlmostEqual(Rate(pr["qb"]["Q1"][1], pr["qb"]["Q1"][0]), 0.2)

    def test_pick_best_uses_tune_seasons_only(self):
        # Config A is worse on tune, better on 2026; must NOT win.
        def recs(ign_by_season):
            out = []
            for s, ign in ign_by_season.items():
                # construct p, mkt, y so ign_rel_bits matches the sign we want
                # use constant p=mkt+delta
                n = 80
                for i in range(n):
                    y = 1 if i % 2 == 0 else 0
                    mkt = 0.5
                    # chosen so mean(ll_p - ll_mkt) ~ ign
                    p = 0.7 if (y == 1 and ign < 0) or (y == 0 and ign > 0) else 0.3
                    if ign == 0:
                        p = 0.5
                    out.append({"game_id": f"{s}_{i}", "season": s, "gameday": f"{s}-09-01",
                                "week": 1, "home_win": y, "p": p, "mkt": mkt,
                                "offset": "close", "close_p": mkt, "open_p": None, "oc_move": None})
            return out
        cfg_a = {"groups": ("epa",), "l2": 1.0, "standardize": 1, "offset": "close"}
        cfg_b = {"groups": ("rest",), "l2": 1.0, "standardize": 1, "offset": "close"}
        # A: bad on 2015-2022, great on 2026
        recs_a = recs({s: 0.05 for s in TUNE} | {2026: -0.2, 2025: -0.2})
        recs_b = recs({s: -0.01 for s in TUNE} | {2026: 0.05, 2025: 0.05})
        results = [(ConfigId(cfg_a), cfg_a, recs_a), (ConfigId(cfg_b), cfg_b, recs_b)]
        picked = PickBest(results, tune=TUNE)
        self.assertIsNotNone(picked)
        self.assertEqual(picked[2]["groups"], ("rest",))
        # confirm A's 2026 score is better (more negative) so the test is meaningful
        self.assertLess(ScoreBlock(recs_a, [2026])["ign_rel_bits"],
                        ScoreBlock(recs_b, [2026])["ign_rel_bits"])

    def test_haversine_and_weather_dome(self):
        miles = HaversineMiles((40.7, -74.0), (34.0, -118.2))
        self.assertGreater(miles, 2400)
        self.assertLess(miles, 2800)
        o, t, w = WeatherVec({"roof": "dome", "temp": 40, "wind": 20})
        self.assertEqual(o, 0.0)
        self.assertEqual(t, 0.0)
        self.assertEqual(w, 0.0)
        o2, t2, w2 = WeatherVec({"roof": "outdoors", "temp": 50, "wind": 20})
        self.assertEqual(o2, 1.0)
        self.assertAlmostEqual(t2, (50 - 70) / 20)
        self.assertAlmostEqual(w2, 2.0)

    def test_scrimmage_helper(self):
        self.assertFalse(IsScrimmage({"posteam": "KC", "epa": 0.1}))
        self.assertTrue(IsScrimmage({"posteam": "KC", "epa": 0.1, "pass": 1}))
        u = Accumulators()
        units = {("g", "KC"): u}
        # AddPlay uses setdefault on a dict of units keyed by (gid, team)
        store = {}
        AddPlay(store, {"game_id": "g", "posteam": "KC", "defteam": "DEN", "epa": 1.0,
                        "success": 1, "yards_gained": 10, "pass": 1, "passer_id": "Q"})
        self.assertEqual(store[("g", "KC")]["off_plays"], 1)
        self.assertEqual(store[("g", "DEN")]["def_plays"], 1)


class BitsTests(unittest.TestCase):
    def test_logloss_bits_and_ign_zero_when_equal(self):
        rows = [{"p": 0.6, "mkt": 0.6, "home_win": 1}]
        self.assertAlmostEqual(IgnRel(rows), 0.0)
        # perfect p is better than a 0.5 market
        self.assertLess(LogLossBits(0.99, 1), LogLossBits(0.5, 1))


class AttachKalshiTests(unittest.TestCase):
    def test_attach_does_not_require_p_full2_and_sets_kmid(self):
        row = {"game_id": "g1", "home_team": "KC", "away_team": "DEN", "season": 2025}
        missing = {"game_id": "g2", "home_team": "BUF", "away_team": "MIA", "season": 2025}
        rbs = {2025: [row, missing], 2026: []}
        qh = {"bid": 0.60, "ask": 0.64, "mid": 0.62, "age_min": 1.0, "end_ts": 0, "price_time": "t"}
        qa = {"bid": 0.36, "ask": 0.40, "mid": 0.38, "age_min": 1.0, "end_ts": 0, "price_time": "t"}

        def fake_quotes(season):
            return {"g1": {"KC": "H", "DEN": "A"}}

        def fake_qts(entry, minutes=60):
            if entry == "H":
                return qh
            if entry == "A":
                return qa
            return None

        with patch("Backtest.ResidualSearch.LoadQuotes", side_effect=fake_quotes),              patch("Backtest.ResidualSearch.QuoteWithTs", side_effect=fake_qts):
            cov = AttachKalshiT60(rbs, seasons=(2025, 2026))
        self.assertNotIn("p_full2", row)
        self.assertAlmostEqual(row["kmid"], 0.62 / (0.62 + 0.38))
        self.assertEqual(row["kq_t60"]["home"]["ask"], 0.64)
        self.assertIsNone(missing["kmid"])
        self.assertEqual(cov[2025]["n_with_quote"], 1)
        self.assertEqual(cov[2025]["n_missing"], 1)
        self.assertEqual(cov[2026]["n"], 0)

    def test_attach_kmid_survives_rows_without_p_full2(self):
        # Empty quotes: no KeyError, kmid stays None.
        rbs = {2025: [{"game_id": "x", "home_team": "KC", "away_team": "DEN"}]}
        with patch("Backtest.ResidualSearch.LoadQuotes", return_value={}):
            cov = AttachKalshiT60(rbs, seasons=(2025,))
        self.assertIsNone(rbs[2025][0]["kmid"])
        self.assertEqual(cov[2025]["n_missing"], 1)


if __name__ == "__main__":
    unittest.main()
