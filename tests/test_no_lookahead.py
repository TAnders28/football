"""Leak tests: a game's features, model probabilities and picks may only use data from before its
gameday. Perturb or delete every game on/after day D and assert day-D outputs do not move.
Run whenever Features, Lab, Engine or PredictDay change. No network, no data/ folder."""

import copy
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from SyntheticNfl import Perturb, SyntheticSeason  # noqa: E402

from Backtest.Engine import RunSeason  # noqa: E402
from Backtest.Features import BuildRows  # noqa: E402
from Backtest.Lab import AttachModels  # noqa: E402
from Backtest.PredictDay import PickSlate, ScoreRows, SlateRows  # noqa: E402

OUTCOMES = {"result", "home_win", "home_cover"}
SEASON = 2016
HISTORY = [2013, 2014, 2015]


def FeatureView(rows, day):
    return {r["game_id"]: {k: v for k, v in r.items() if k not in OUTCOMES}
            for r in rows if r["gameday"] == day}


def MidSeasonSunday(games):
    days = sorted({g["gameday"] for g in games})
    return next(d for d in days[len(days) // 2:] if sum(g["gameday"] == d for g in games) > 3)


def History():
    out, prev = {}, []
    for s in HISTORY:
        games = SyntheticSeason(s)
        out[s] = BuildRows(games, prev)
        prev = games
    return out, prev


def PredictSlate(games, prior, day):
    rbs, _ = History()
    slate = PickSlate(games, day)
    rows, _last, _n = SlateRows(games, prior, slate)
    rbs[SEASON] = rows
    AttachModels(rbs)
    return {p["game_id"]: (p["recommended"], p["full2_home_prob"], p["full3_home_prob"], p["margin_model_home_prob"])
            for p in ScoreRows(rows)}


class NoLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.games = SyntheticSeason(SEASON)
        self.prior = SyntheticSeason(SEASON - 1)
        self.day = MidSeasonSunday(self.games)

    def test_features_ignore_same_day_and_later_results(self):
        full = FeatureView(BuildRows(self.games, self.prior), self.day)
        cut = FeatureView(BuildRows(Perturb(self.games, self.day), self.prior), self.day)
        self.assertTrue(full)
        self.assertEqual(full, cut)

    def test_slate_rows_ignore_same_day_and_later_results(self):
        slate = PickSlate(self.games, self.day)
        full, _, n_full = SlateRows(self.games, self.prior, slate)
        pert = Perturb(self.games, self.day)
        cut, _, n_cut = SlateRows(pert, self.prior, PickSlate(pert, self.day))
        self.assertEqual(n_full, n_cut)
        self.assertEqual(full, cut)
        self.assertTrue(all(r["result"] is None and r["home_win"] is None for r in full))

    def test_slate_rows_match_backtest_features(self):
        """PredictDay's live feature builder equals the backtest's for the same day."""
        slate = PickSlate(self.games, self.day)
        live, _, _ = SlateRows(self.games, self.prior, slate)
        back = FeatureView(BuildRows(self.games, self.prior), self.day)
        keys = ["mkt_home", "margin_logit", "form3_diff", "ats_diff", "ats3_diff", "rest_diff", "div", "games_played"]
        for r in live:
            for k in keys:
                self.assertAlmostEqual(r[k], back[r["game_id"]][k], places=12, msg=k)

    def test_picks_and_model_probs_ignore_same_day_and_later_results(self):
        full = PredictSlate(self.games, self.prior, self.day)
        cut = PredictSlate(Perturb(self.games, self.day), self.prior, self.day)
        self.assertTrue(full)
        self.assertEqual(full, cut)

    def test_canary_an_earlier_result_does_change_the_features(self):
        """Proves the tests above are sensitive: changing a result from BEFORE day D must move day D."""
        games = copy.deepcopy(self.games)
        earlier = [g for g in games if g["gameday"] < self.day]
        for g in earlier[-8:]:
            g["home_score"], g["away_score"] = g["away_score"] + 30, g["home_score"]
        self.assertNotEqual(FeatureView(BuildRows(self.games, self.prior), self.day),
                            FeatureView(BuildRows(games, self.prior), self.day))

    def test_walk_forward_models_ignore_the_season_being_scored_and_later(self):
        def p2015(future_seed):
            rbs, prev = {}, []
            for s in (2013, 2014, 2015, 2016):
                games = SyntheticSeason(s, seed=future_seed if s == 2016 else 0)
                rbs[s] = BuildRows(games, prev)
                prev = games
            AttachModels(rbs)
            return [(r["game_id"], r["p_full2"], r["p_full3"]) for r in rbs[2015]]
        self.assertEqual(p2015(1), p2015(7))

    def test_engine_walks_days_in_order_and_ignores_later_results(self):
        class Recorder:
            name = "rec"

            def __init__(self):
                self.seen = []

            def Decide(self, view):
                self.seen.append((view["gameday"], view["game_id"], round(view["model_home_prob"], 12)))
                assert "home_score" not in view and "away_score" not in view
                return None

        shuffled = self.games[:]
        random.Random(3).shuffle(shuffled)
        rec = Recorder()
        RunSeason(shuffled, [rec])
        days = [d for d, _, _ in rec.seen]
        self.assertEqual(days, sorted(days))
        rec2 = Recorder()
        RunSeason(Perturb(shuffled, self.day, drop_after=False), [rec2])
        on_or_before = lambda seen: [x for x in seen if x[0] <= self.day]
        self.assertEqual(on_or_before(rec.seen), on_or_before(rec2.seen))


if __name__ == "__main__":
    unittest.main()
