"""Unit tests for the NFL drive resimulator (Sim/). No network, no data/ needed."""
import unittest

import numpy as np
import polars as pl

from Sim.Anchor import Anchor, AnchoredProbs
from Sim.DriveModel import CAT_ID, FitBase, MatchupProbs, N_CAT, StateOf, TeamWalk
from Sim.Drives import CATS, DrivesFromPlays, ParseClock
from Sim.GameSim import SimulateGames, Summarise
from Sim.NearMiss import NearMiss


def SyntheticDrives(n_games=60, seed=0):
    rng = np.random.default_rng(seed)
    probs = np.array([0.24, 0.16, 0.40, 0.10, 0.05, 0.03, 0.01, 0.005, 0.005])
    probs /= probs.sum()
    pts = {"TD": (7, 0), "FG": (3, 0), "OPPTD": (0, 7), "SAFETY": (0, 2)}
    rows = []
    for g in range(n_games):
        hs, half, sd, pos = 1800.0, 1, 0.0, int(rng.integers(0, 2))
        for i in range(22):
            if i == 11:
                half, hs = 2, 1800.0
            c = CATS[int(rng.choice(len(probs), p=probs))]
            if i in (10, 21):
                c = "EOH"
            o, d = pts.get(c, (0, 0))
            dur = float(rng.integers(30, 300))
            rows.append({"game_id": f"G{g}", "season": 2020, "drive_idx": i, "half": half, "half_secs": max(hs, 1.0),
                         "score_diff": sd, "start_fp": float(rng.integers(20, 90)), "cat": c, "off_pts": float(o),
                         "def_pts": float(d), "dur": dur, "posteam": "H" if pos == 0 else "A", "defteam": "A" if pos == 0 else "H",
                         "is_home_off": 1 - pos, "home_final": 0.0, "away_final": 0.0})
            hs = max(hs - dur, 1.0)
            sd = -(sd + o - d)
            pos = 1 - pos
    return pl.DataFrame(rows)


class TestDrives(unittest.TestCase):
    def test_parse_clock(self):
        self.assertEqual(ParseClock("5:31"), 331.0)
        self.assertTrue(np.isnan(ParseClock(None)))

    def test_drives_from_plays_points(self):
        base = {"game_id": "2025_01_A_H", "season_type": "REG", "week": 1, "game_date": "2025-09-07", "home_team": "H",
                "away_team": "A", "home_score": 7, "away_score": 3, "game_half": "Half1", "drive_time_of_possession": "2:00"}
        plays = [
            {**base, "play_id": 1, "posteam": "H", "defteam": "A", "fixed_drive": 1, "fixed_drive_result": "Touchdown", "play_type": "run",
             "yardline_100": 75, "game_seconds_remaining": 3600, "half_seconds_remaining": 1800, "total_home_score": 0, "total_away_score": 0},
            {**base, "play_id": 2, "posteam": "A", "defteam": "H", "fixed_drive": 2, "fixed_drive_result": "Field goal", "play_type": "pass",
             "yardline_100": 70, "game_seconds_remaining": 3400, "half_seconds_remaining": 1600, "total_home_score": 7, "total_away_score": 0},
        ]
        d = DrivesFromPlays(pl.DataFrame(plays))
        self.assertEqual(d.height, 2)
        self.assertEqual(d["off_pts"].to_list(), [7.0, 3.0])
        self.assertEqual(d["cat"].to_list(), ["TD", "FG"])
        self.assertEqual(d["score_diff"].to_list(), [0.0, -7.0])


class TestModelAndSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = SyntheticDrives()
        cls.base = FitBase(cls.d)

    def test_state_of(self):
        st = StateOf(np.array([1, 2, 2, 2, 3]), np.array([1000, 100, 100, 100, 500]), np.array([0, 10, 0, -2, 0]))
        self.assertEqual(st.tolist(), [0, 1, 4, 5, 8])

    def test_tables_normalised(self):
        self.assertTrue(np.allclose(self.base.p0.sum(axis=2), 1.0))
        self.assertTrue(np.all(self.base.p0[..., CAT_ID["EOH"]] == 0))

    def test_sim_runs_and_is_balanced(self):
        one = np.ones(N_CAT)
        probs = MatchupProbs(self.base, one, one, one, one)
        probs[:] = probs[0]  # identical tables for both sides
        h, a = SimulateGames(probs, self.base, 6000, np.random.default_rng(1))
        s = Summarise(h, a)
        self.assertTrue(np.all(h >= 0) and np.all(a >= 0))
        self.assertAlmostEqual(s["p_home"], 0.5, delta=0.03)
        self.assertGreater(s["total_mean"], 10)

    def test_tilt_moves_win_prob(self):
        one = np.ones(N_CAT)
        p0 = MatchupProbs(self.base, one, one, one, one)
        p1 = MatchupProbs(self.base, one, one, one, one, tilt_home=0.3)
        h0, a0 = SimulateGames(p0, self.base, 6000, np.random.default_rng(2))
        h1, a1 = SimulateGames(p1, self.base, 6000, np.random.default_rng(2))
        self.assertGreater(Summarise(h1, a1)["p_home"], Summarise(h0, a0)["p_home"] + 0.03)

    def test_near_miss_eps0_is_real_result(self):
        g = self.d.filter(pl.col("game_id") == "G0").sort("drive_idx")
        from Sim.DriveModel import DriveArrays
        st, fb, cat = DriveArrays(g)
        side = 1 - g["is_home_off"].to_numpy()
        one = np.ones(N_CAT)
        probs = MatchupProbs(self.base, one, one, one, one)
        home = float(np.where(side == 0, g["off_pts"], g["def_pts"]).sum())
        away = float(np.where(side == 1, g["off_pts"], g["def_pts"]).sum())
        want = 1.0 if home > away else (0.5 if home == away else 0.0)
        for src in ("model", "boot"):
            self.assertEqual(NearMiss(side, st, fb, cat, g["off_pts"], g["def_pts"], probs, self.base, 50, 0.0, np.random.default_rng(3), src), want)

    def test_team_walk_factors(self):
        w = TeamWalk(n0=10, carry=0.5)
        w.NewSeason(2020)
        w.Update(self.d.filter(pl.col("game_id").is_in(["G1", "G2", "G3"])), self.base)
        fo, fd = w.Factors("H", self.base.rate)
        self.assertEqual(fo.shape, (N_CAT,))
        self.assertTrue(np.all(fo > 0))

    def test_anchor_hits_target(self):
        s, k = Anchor(self.base, 0.75, None, n=3000, seed=4)
        h, a = SimulateGames(AnchoredProbs(self.base, s, k), self.base, 6000, np.random.default_rng(9))
        self.assertAlmostEqual(Summarise(h, a)["p_home"], 0.75, delta=0.03)


if __name__ == "__main__":
    unittest.main()
