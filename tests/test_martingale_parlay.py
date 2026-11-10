"""MartingaleParlay: pre-game leg choice (no outcome leak), distinct games, progression/reset, recover stake,
cap, stop-loss, absorbing target and ruin, grid size (synthetic data)."""

import copy
import unittest

import numpy as np

import Backtest.MartingaleParlay as MP


def Row(gid, home_ml, away_ml, mkt_home, result, team="H"):
    return {"game_id": gid, "gameday": "2020-01-05", "gametime": "13:00", "home_ml": home_ml, "away_ml": away_ml,
            "mkt_home": mkt_home, "p_full2": mkt_home, "home_team": team + gid, "away_team": "A" + gid,
            "result": result, "home_win": None if result == 0 else int(result > 0)}


def Ser(outs, q=2.0, has=None):
    n = len(outs)
    return {"has": np.array(has if has is not None else [True] * n), "q_post": np.full(n, q), "q_fee": np.full(n, q),
            "s_post": np.full(n, q), "s_fee": np.full(n, q), "out": np.array(outs), "dates": [f"d{i}" for i in range(n)],
            "legs": [None] * n, "seasons": np.full(n, 2020)}


def Path(ser):
    return np.arange(len(ser["has"]))[None, :]


class MartingaleParlayTests(unittest.TestCase):
    def setUp(self):
        self.rows = [Row("g1", -150, 130, 0.58, 3), Row("g2", -200, 170, 0.65, -7), Row("g3", 120, -140, 0.43, 10),
                     Row("g4", -110, -110, 0.5, 0), Row("g5", -300, 250, 0.73, 14)]

    def test_leg_choice_ignores_outcomes_and_uses_distinct_games(self):
        flipped = copy.deepcopy(self.rows)
        for r in flipped:
            r["result"] = -r["result"]
            r["home_win"] = None if r["result"] == 0 else int(r["result"] > 0)
        for n in (2, 3, 4):
            for sel in MP.SELECTIONS:
                a = MP.PickLegs(MP.Pregame(self.rows), n, 2.0 * n, sel)
                b = MP.PickLegs(MP.Pregame(flipped), n, 2.0 * n, sel)
                self.assertIsNotNone(a)
                self.assertEqual([(l["game_id"], l["side"]) for l in a], [(l["game_id"], l["side"]) for l in b])
                self.assertEqual(len({l["game_id"] for l in a}), n)
        self.assertNotIn("result", MP.Pregame(self.rows)[0]["sides"][0])

    def test_band_and_settlement_with_void_leg(self):
        legs = MP.PickLegs(MP.Pregame(self.rows), 2, 3.0, "market")
        q = np.prod([l["dec"] for l in legs])
        self.assertLessEqual(abs(np.log(q / 3.0)), np.log(MP.BAND))
        self.assertIsNone(MP.PickLegs(MP.Pregame(self.rows[:1]), 2, 3.0, "market"))
        self.assertIsNone(MP.LegResult(self.rows[3], "home"))

    def test_multiplier_progression_and_reset(self):
        ser = Ser([-1, -1, 1, -1])
        cfg = MP.Cfg("s", 0.10, 2.0)
        res = MP.Replay(ser, cfg, 100.0, "posted")
        self.assertEqual(res["equity"][:3], [90.0, 70.0, 110.0])     # stakes 10, 20, 40
        self.assertEqual(res["equity"][3], 99.0)                     # reset: base = 10% of 110 = 11

    def test_recover_stake_wins_back_losses_plus_base_profit(self):
        ser = Ser([-1, -1, 1], q=3.0)
        res = MP.Replay(ser, MP.Cfg("s", 0.10, "recover"), 100.0, "posted")
        self.assertEqual(res["equity"], [90.0, 75.0, 120.0])        # 10, 15, 22.5 -> +base*(q-1)

    def test_cap_and_stop_loss(self):
        ser = Ser([-1, -1, -1, -1])
        capped = MP.Replay(ser, MP.Cfg("s", 0.10, 2.0, cap=15.0), 1000.0, "posted")
        self.assertEqual(capped["max_stake"], 15.0)
        stopped = MP.Replay(Ser([-1, -1, -1]), MP.Cfg("s", 0.10, 2.0, stop=2), 100.0, "posted")
        self.assertEqual(stopped["equity"], [90.0, 70.0, 63.0])     # 10, 20, then reset to 10% of 70

    def test_absorbing_ruin_and_target(self):
        ruin = MP.Simulate(Ser([-1, -1, -1, 1]), Path(Ser([0] * 4)), MP.Cfg("s", 0.30, 2.0), 100.0)
        self.assertTrue(ruin["ruin"][0] and not ruin["hit"][0])     # stakes 30, 60; next 120 > 10 left -> cannot cover
        self.assertEqual(float(ruin["end"][0]), 10.0)
        self.assertEqual(int(ruin["bets"][0]), 2)
        win = Ser([1, 1, -1], q=1000.0)
        hit = MP.Simulate(win, Path(win), MP.Cfg("s", 1.0, 2.0), 1000.0)
        self.assertTrue(hit["hit"][0] and not hit["ruin"][0])
        self.assertEqual(int(hit["t_abs"][0]), 0)
        self.assertEqual(float(hit["end"][0]), 1_000_000.0)         # held after the first hit

    def test_no_bet_days_and_push_keep_state(self):
        ser = Ser([-1, 0, 0, 1], has=[True, True, False, True])
        res = MP.Replay(ser, MP.Cfg("s", 0.10, 2.0), 100.0, "posted")
        self.assertEqual(res["equity"], [90.0, 90.0, 90.0, 110.0])  # push and no-bet day: streak kept

    def test_choose_respects_ruin_limit_and_breaks_ties_on_upside(self):
        rows = [{"config": "a", "p_target": 0.0, "p_ruin": 0.1, "p95_end": 200.0},
                {"config": "b", "p_target": 0.0, "p_ruin": 0.3, "p95_end": 900.0},
                {"config": "c", "p_target": 0.2, "p_ruin": 0.7, "p95_end": 1e6}]
        self.assertEqual(MP.Choose(rows), (rows[1], True))
        self.assertEqual(MP.Choose(rows[2:]), (rows[2], False))

    def test_grid_is_modest(self):
        self.assertEqual(len(MP.MainGrid()), 360)
        self.assertEqual(len({c["name"] for c in MP.MainGrid()}), 360)
        self.assertEqual(len(MP.Variants(MP.MainGrid()[0])), 6)


if __name__ == "__main__":
    unittest.main()