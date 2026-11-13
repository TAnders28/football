"""StakingEnv: pre-game state (no outcome leak), payouts on both bases, parlay voids, exposure cap,
absorbing target/ruin, synthetic outcomes, train-only calibration, constant baselines (synthetic data)."""

import copy
import math
import unittest

import numpy as np

import Backtest.StakingEnv as SE


def Row(gid, day, home_ml, away_ml, mkt_home, p_model, result, week=1):
    return {"game_id": gid, "gameday": day, "gametime": "13:00", "home_ml": home_ml, "away_ml": away_ml,
            "mkt_home": mkt_home, "p_full2": p_model, "home_team": "H" + gid, "away_team": "A" + gid, "week": week,
            "game_type": "REG", "season": 2020, "result": result, "home_win": None if result == 0 else int(result > 0)}


def Pack(rows, basis="closing_ml"):
    return SE.Pack(SE.BuildDays({2020: rows}, [2020]), basis)


def Run(pack, theta, **kw):
    idx = np.arange(pack["n"])[None, :]
    return SE.Simulate(pack, idx, theta, **kw)


def ParlayTheta(g=0.1, legs=2):
    t = SE.FixedTheta(0.0)[0]
    t[SE.NF + 1] = SE.Logit(g / SE.GMAX)
    k = SE.LEGS.index(legs)
    t[2 * SE.NF + 1 + k * SE.NF] = 10.0
    return t[None, :]


class StakingEnvTests(unittest.TestCase):
    def test_single_win_loss_void_closing(self):
        for res, want in ((7, 12_000.0), (-7, 8_000.0), (0, 10_000.0)):
            p = Pack([Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, res)])
            self.assertAlmostEqual(p["K"][0, 0], 0.2)
            self.assertAlmostEqual(p["E"][0, 0], 0.1)
            out = Run(p, SE.FixedTheta(1.0))
            self.assertAlmostEqual(float(out["end"][0]), want, places=6)

    def test_fee_basis_payout(self):
        p = Pack([Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, 7)], basis="tim_fee_est")
        b = 1 / 0.52 - 1
        self.assertAlmostEqual(p["B"][0, 0], b)
        self.assertAlmostEqual(p["E"][0, 0], 0.6 - 0.52)

    def test_parlay_settlement_with_void_and_loss(self):
        base = [Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, 7), Row("g2", "2020-09-13", 150, -170, 0.4, 0.5, 3)]
        d1, d2 = 2.0, 2.5
        for r2, want in ((3, 10_000 * (1 + 0.1 * (d1 * d2 - 1))), (0, 10_000 * (1 + 0.1 * (d1 - 1))), (-3, 9_000.0)):
            rows = copy.deepcopy(base)
            rows[1]["result"], rows[1]["home_win"] = r2, None if r2 == 0 else int(r2 > 0)
            out = Run(Pack(rows), ParlayTheta(0.1, 2))
            self.assertAlmostEqual(float(out["end"][0]), want, places=4)

    def test_no_lookahead_actions_ignore_future_outcomes(self):
        rows = []
        for w in range(6):
            day = f"2020-09-{13 + 7 * w:02d}"
            rows += [Row(f"w{w}a", day, 120, -140, 0.44, 0.52, 7, w + 1), Row(f"w{w}b", day, -200, 170, 0.65, 0.70, -3, w + 1),
                     Row(f"w{w}c", day, 200, -240, 0.32, 0.40, 10, w + 1)]
        theta = SE.InitTheta()[None, :] + np.random.default_rng(1).normal(0, 0.5, (1, SE.N_PARAM))
        a = Run(Pack(rows), theta, record=True)
        cut = 3
        flipped = copy.deepcopy(rows)
        for r in flipped:
            if int(r["game_id"][1]) >= cut:
                r["result"] = -r["result"]
                r["home_win"] = None if r["result"] == 0 else int(r["result"] > 0)
        b = Run(Pack(flipped), theta, record=True)
        for t in range(cut + 1):
            for k in ("lam", "g", "legs", "exposure", "n_singles"):
                np.testing.assert_allclose(a["path"][t][k], b["path"][t][k])
        self.assertNotAlmostEqual(float(a["end"][0]), float(b["end"][0]))

    def test_exposure_capped_and_abstain(self):
        rows = [Row(f"g{i}", "2020-09-13", 300, -400, 0.24, 0.9, 7) for i in range(8)]
        p = Pack(rows)
        out = Run(p, SE.FixedTheta(2.0), record=True)
        self.assertLessEqual(float(out["path"][0]["exposure"][0]), 1.0 + 1e-9)
        none = Run(p, SE.FixedTheta(0.0))
        self.assertEqual(float(none["end"][0]), SE.START)
        self.assertEqual(float(none["bet_day_share"][0]), 0.0)

    def test_absorbing_target_and_ruin(self):
        rows = [Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, 7), Row("g2", "2020-09-20", 100, -100, 0.5, 0.6, -7)]
        p = Pack(rows)
        out = Run(p, SE.FixedTheta(1.0), state={"bank": np.array([900_000.0])})
        self.assertTrue(out["hit"][0])
        self.assertAlmostEqual(float(out["end"][0]), 1_080_000.0)
        out = Run(p, SE.FixedTheta(1.0), state={"bank": np.array([1_100.0])})
        self.assertFalse(out["hit"][0] or out["ruin"][0])
        lost = [Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, -7), Row("g2", "2020-09-20", 100, -100, 0.5, 0.6, 7)]
        out = Run(Pack(lost), SE.FixedTheta(1.0), state={"bank": np.array([1_200.0])})
        self.assertTrue(out["ruin"][0])
        self.assertAlmostEqual(float(out["end"][0]), 960.0)

    def test_synthetic_outcomes_do_not_touch_real(self):
        p = Pack([Row("g1", "2020-09-13", 100, -100, 0.5, 0.6, -7)])
        w_before = p["W"].copy()
        pcal = np.ones_like(p["M"])
        out = SE.Simulate(p, np.zeros((2, 1), int), SE.FixedTheta(1.0, n=2), synth=np.array([True, False]), pcal=pcal,
                          u=lambda j: np.zeros((2, SE.MAX_C)))
        self.assertAlmostEqual(float(out["end"][0]), 12_000.0)
        self.assertAlmostEqual(float(out["end"][1]), 8_000.0)
        np.testing.assert_array_equal(p["W"], w_before)

    def test_calibration_uses_only_masked_days(self):
        rows = []
        for w in range(10):
            day = f"2020-{9 + w // 4:02d}-{1 + 7 * (w % 4):02d}"
            rows += [Row(f"{w}a", day, 120, -140, 0.44, 0.50, 7 if w % 2 else -7), Row(f"{w}b", day, -200, 170, 0.65, 0.6, -3 if w % 3 else 3)]
        p = Pack(rows)
        mask = np.arange(p["n"]) < 6
        w1 = SE.CalibFit(p, mask)
        p2 = copy.deepcopy(p)
        p2["W"][~mask] *= -1
        np.testing.assert_allclose(w1, SE.CalibFit(p2, mask))


if __name__ == "__main__":
    unittest.main()
