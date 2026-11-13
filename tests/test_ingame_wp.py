import unittest

import numpy as np

from Backtest.InGameModelEval import Mismatch
from Backtest.InGameWP import FEATS, PredictPos, StateFeatures
from Backtest.LiveInGame import ClockSecs, EspnState, ModelProbHome


def St(**kw):
    s = {"diff": 0, "gsr": 1800, "hsr": 0, "qtr": 3, "down": 1, "ydstogo": 10, "yardline_100": 75, "pos_to": 3, "def_to": 3,
         "is_home": True, "receive_2h": False}
    s.update(kw)
    return s


def ToyModel():
    # P(pos wins) rises with diff_time_ratio and prior; falls with yardline_100
    coef = np.zeros(len(FEATS)); coef[FEATS.index("diff_time_ratio")] = 1.0; coef[FEATS.index("prior_logit")] = 1.0
    coef[FEATS.index("yardline_100")] = -0.2
    return {"mean": [0.0] * len(FEATS), "scale": [1.0] * len(FEATS), "coef": coef.tolist(), "intercept": 0.0}


class InGameWPTest(unittest.TestCase):
    def test_state_features_prior_flips_for_away(self):
        f_home = StateFeatures(St(is_home=True), 0.7)
        f_away = StateFeatures(St(is_home=False), 0.7)
        i = FEATS.index("prior_logit")
        self.assertAlmostEqual(f_home[i], -f_away[i])
        self.assertGreater(f_home[i], 0)
        self.assertEqual(len(f_home), len(FEATS))

    def test_state_features_defaults_and_ot(self):
        f = StateFeatures(St(down=None, ydstogo=None, yardline_100=None, pos_to=None, def_to=None, qtr=5, gsr=300), None)
        self.assertEqual(f[FEATS.index("ot")], 1.0)
        self.assertEqual(f[FEATS.index("elapsed")], 1.0)
        self.assertEqual(f[FEATS.index("yardline_100")], 75.0)
        self.assertEqual(f[FEATS.index("prior_logit")], 0.0)

    def test_predict_pos_monotone_in_score(self):
        mj = ToyModel()
        lo = PredictPos(mj, [StateFeatures(St(diff=-7), 0.5)])[0]
        hi = PredictPos(mj, [StateFeatures(St(diff=7), 0.5)])[0]
        self.assertLess(lo, 0.5); self.assertGreater(hi, 0.5)

    def test_clock_and_model_prob_home(self):
        self.assertEqual(ClockSecs("4:12"), 252)
        self.assertIsNone(ClockSecs(None))
        mj = ToyModel()
        st = {"phase": "live", "possession": "away", "period": 4, "clock": "2:00", "home_score": 10, "away_score": 17,
              "down": 1, "distance": 10, "yards_to_endzone": 50, "home_timeouts": 3, "away_timeouts": 3, "opening_receiver": "home"}
        p, why = ModelProbHome(st, 0.5, mj)
        self.assertIsNone(why)
        self.assertLess(p, 0.5)          # home trails by 7 late, away has the ball
        self.assertEqual(ModelProbHome({**st, "phase": "pregame"}, 0.5, mj)[0], None)

    def test_espn_state_parse(self):
        play = lambda seq, team, down, dist, ytg: {"sequenceNumber": str(seq), "wallclock": f"2026-11-16T02:0{seq}:00Z", "period": {"number": 2},
                                                   "clock": {"displayValue": "1:00"}, "end": {"team": {"id": team}, "down": down, "distance": dist, "yardsToEndzone": ytg}}
        summary = {"header": {"competitions": [{"competitors": [{"homeAway": "home", "team": {"id": "1"}, "score": "7"},
                                                                {"homeAway": "away", "team": {"id": "2"}, "score": "3"}],
                                                "status": {"type": {"state": "in"}, "period": 2, "displayClock": "0:45"}}]},
                   "drives": {"previous": [{"team": {"id": "2"}, "plays": [play(1, "2", 1, 10, 75)]}],
                              "current": {"team": {"id": "1"}, "plays": [play(3, "1", 2, 4, 30), play(2, "1", 1, 10, 34)]}}}
        sb = {"competitions": [{"situation": {"homeTimeouts": 2, "awayTimeouts": 1}}]}
        st = EspnState(summary, sb)
        self.assertEqual((st["phase"], st["possession"], st["down"], st["distance"], st["yards_to_endzone"]), ("live", "home", 2, 4, 30))
        self.assertEqual((st["home_score"], st["away_score"], st["period"], st["clock"]), (7, 3, 2, "0:45"))
        self.assertEqual((st["home_timeouts"], st["away_timeouts"], st["opening_receiver"]), (2, 1, "away"))
        self.assertEqual(st["state_time"], "2026-11-16T02:03:00Z")

    def test_mismatch_pnl_sign(self):
        rng = np.random.default_rng(0)
        # model says home 0.8, market 0.5, home always wins -> positive P&L after fees
        sub = [{"game_id": f"g{i}", "t": float(j), "y": 1, "market": 0.5, "model": 0.8} for i in range(20) for j in range(3)]
        m = Mismatch(sub, "model", 0.05, 200, rng, "Kalshi")
        self.assertEqual(m["n_games"], 20)
        self.assertAlmostEqual(m["pnl_per_contract_tim_est_0.02"]["first_trigger_per_game"]["mean"], 0.48)
        self.assertAlmostEqual(m["pnl_per_contract_kalshi_verified"]["first_trigger_per_game"]["mean"], 0.5 - 0.07 * 0.25, places=4)


if __name__ == "__main__":
    unittest.main()