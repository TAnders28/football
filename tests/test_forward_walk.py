"""Frozen-strategy forward walk and calibration on synthetic seasons (no network, no data/ folder)."""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))

from SyntheticNfl import Perturb, SyntheticSeason  # noqa: E402

import Backtest.ForwardWalk as FW  # noqa: E402
from Backtest.Calibration import Calibration, ReliabilityBins, Scores  # noqa: E402
from Backtest.Features import BuildRows  # noqa: E402
from Backtest.FrozenConfig import FROZEN  # noqa: E402
from Backtest.Lab import AttachModels  # noqa: E402

ET = ZoneInfo("America/New_York")
SEASON = 2016


def RowsBySeason(last_games=None):
    out, prev = {}, []
    for s in range(2012, SEASON + 1):
        games = last_games if (s == SEASON and last_games is not None) else SyntheticSeason(s)
        out[s] = BuildRows(games, prev)
        prev = games
    AttachModels(out)
    return out


def Row(**kw):
    r = {"game_id": "2026_09_AAA_BBB", "season": 2026, "week": 9, "gameday": "2026-11-08", "gametime": "13:00",
         "home_team": "BBB", "away_team": "AAA", "home_ml": -150, "away_ml": 130, "mkt_home": 0.58,
         "p_full2": 0.58, "p_full3": 0.58, "div": 0.0, "games_played": 5, "home_win": 1}
    r.update(kw)
    return r


class ForwardWalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rbs = RowsBySeason()
        cls.out = FW.ForwardWalk(SEASON, rows_by_season=cls.rbs, write=False)

    def test_matches_hunt_engine_bankroll(self):
        s = self.out["summary"]
        self.assertTrue(s["consistency_ok"], (s["closing_ml"]["end_bankroll"], s["consistency_hunt_engine_end"]))
        self.assertGreater(s["closing_ml"]["all"]["bets"], 0)

    def test_equity_and_drawdown_columns(self):
        for name, (_bets, daily) in self.out["series"].items():
            self.assertEqual([d["date"] for d in daily], sorted(d["date"] for d in daily), name)
            self.assertTrue(all(d["drawdown_pct"] <= 0 for d in daily), name)
            self.assertTrue(all(d["equity"] >= 0 for d in daily), name)

    def test_no_lookahead_later_results_do_not_change_earlier_bets(self):
        games = SyntheticSeason(SEASON)
        days = sorted({g["gameday"] for g in games})
        cut = days[len(days) // 2]
        changed = FW.ForwardWalk(SEASON, rows_by_season=RowsBySeason(Perturb(games, cut, drop_after=False)), write=False)
        def before(out):
            return [(b["game_id"], b["pick"], b["stake"], b["pnl"]) for b in out["series"]["frozen_closing_ml"][0] if b["date"] < cut]
        self.assertEqual(before(self.out), before(changed))

    def test_value_rule_has_priority_and_div_needs_div_game(self):
        pick = FW.FrozenPicker()
        c, s = pick(Row(p_full2=0.70, p_full3=0.70, div=1.0), 10_000)
        self.assertEqual(c["rule"], FROZEN["value"]["rule"])
        c, s = pick(Row(p_full2=0.60, p_full3=0.62, div=1.0), 10_000)
        self.assertEqual(c["rule"], FROZEN["div_addon"]["rule"])
        self.assertAlmostEqual(s, 200.0)
        self.assertIsNone(pick(Row(p_full2=0.60, p_full3=0.62, div=0.0), 10_000))
        self.assertIsNone(pick(Row(home_ml=None), 10_000))

    def test_fee_estimate_payout_is_worse_than_fair(self):
        c = {"odds": 130, "p_mkt": 0.42}
        self.assertAlmostEqual(FW.Payout(c, "tim_fee_est"), 1 / 0.44 - 1)
        self.assertAlmostEqual(FW.Payout(c, "closing_ml"), 1.30)

    def test_seen_by_search_cutoff(self):
        self.assertTrue(FW.SeenBySearch(Row(gameday="2026-11-01", gametime="13:00")))
        self.assertFalse(FW.SeenBySearch(Row(gameday="2026-11-01", gametime="16:25")))
        self.assertFalse(FW.SeenBySearch(Row(gameday="2026-11-05", gametime="20:15")))
        self.assertTrue(FW.SeenBySearch(Row(season=2025, gameday="2025-12-01")))

    def test_writes_tidy_csv(self):
        with tempfile.TemporaryDirectory() as d:
            FW.ForwardWalk(SEASON, rows_by_season=self.rbs, out_dir=d, base="x")
            with open(os.path.join(d, "x.csv"), encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
            self.assertEqual(header[:4], ["date", "series", "equity", "drawdown_pct"])
            self.assertTrue(os.path.exists(os.path.join(d, "x.json")))


class CalibrationTests(unittest.TestCase):
    def test_scores_and_bins(self):
        rows = [{"p": 0.8, "home_win": 1}, {"p": 0.8, "home_win": 0}, {"p": 0.2, "home_win": 0}, {"p": 0.5, "home_win": None}]
        s = Scores(rows, "p")
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["brier"], (0.04 + 0.64 + 0.04) / 3, places=5)
        bins = ReliabilityBins(rows, "p")
        self.assertEqual(sum(b["n"] for b in bins), 3)
        b8 = next(b for b in bins if b["bin_lo"] == 0.8)
        self.assertEqual((b8["n"], b8["obs_rate"]), (2, 0.5))

    def test_market_skill_is_zero_for_market(self):
        out = Calibration([SEASON], rows_by_season=RowsBySeason(), write=False)
        sc = out["report"]["sets"][str(SEASON)]["scores"]
        self.assertEqual(sc["p_mkt"]["brier_skill_vs_market"], 0.0)
        self.assertGreater(sc["p_full2"]["n"], 50)
        self.assertEqual(list(out["tidy"][0].keys())[:2], ["date", "series"])


if __name__ == "__main__":
    unittest.main()
