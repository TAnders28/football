"""Drive rebuild / validation and team efficiency on hand-made play-by-play (no network, no data/ folder)."""

import os
import tempfile
import unittest

from PlayByPlay.DriveCheck import CheckGame, DriveCheck, DriveResult, RebuildDrives
from PlayByPlay.TeamEfficiency import SeasonTable, TeamEfficiency, WeeklyTable

G = "2026_01_AAA_BBB"


def P(i, team, drive, half="Half1", result=None, **kw):
    p = {"game_id": G, "game_date": "2026-09-13", "week": 1, "season": 2026, "play_id": i, "order_sequence": i,
         "posteam": team, "defteam": None if team is None else ("BBB" if team == "AAA" else "AAA"),
         "game_half": half, "fixed_drive": drive, "fixed_drive_result": result, "play": 1, "desc": f"play {i}"}
    p.update(kw)
    return p


def Game():
    return [
        P(1, "AAA", 1, result="Touchdown", play=0, play_type="kickoff"),     # kickoff to AAA opens AAA's drive
        P(2, "AAA", 1, result="Touchdown", drive_play_count=2),
        P(3, "AAA", 1, result="Touchdown", touchdown=1, td_team="AAA", drive_play_count=2),
        P(4, None, None, play=0),                                             # timeout: no posteam
        P(5, "BBB", 2, result="Punt", play=0, play_type="kickoff"),
        P(6, "BBB", 2, result="Punt", drive_play_count=2),
        P(7, "BBB", 2, result="Punt", punt_attempt=1, drive_play_count=2),
        P(8, "AAA", 3, result="Turnover", interception=1, drive_play_count=1),
        P(9, "BBB", 4, result="End of half", drive_play_count=1),
        P(10, "BBB", 5, half="Half2", result="Field goal", field_goal_result="made", drive_play_count=1),
    ]


class DriveTests(unittest.TestCase):
    def test_rebuild_splits_on_possession_and_half(self):
        self.assertEqual(RebuildDrives(Game()), [[0, 1, 2], [4, 5, 6], [7], [8], [9]])

    def test_results(self):
        g = Game()
        self.assertEqual(DriveResult(g[0:3]), "Touchdown")
        self.assertEqual(DriveResult(g[4:7]), "Punt")
        self.assertEqual(DriveResult([g[7]]), "Turnover")
        self.assertEqual(DriveResult([g[8]]), "End of half")
        self.assertEqual(DriveResult([g[9]]), "Field goal")
        self.assertEqual(DriveResult([P(1, "AAA", 1, interception=1, touchdown=1, td_team="BBB")]), "Opp touchdown")
        self.assertEqual(DriveResult([P(1, "AAA", 1, field_goal_result="blocked")]), "Missed field goal")
        self.assertEqual(DriveResult([P(1, "AAA", 1, fourth_down_failed=1)]), "Turnover on downs")

    def test_check_game_full_agreement(self):
        out = CheckGame(Game())
        self.assertEqual((out["nfl_drives"], out["our_drives"], out["matched"]), (5, 5, 5))
        self.assertEqual(out["result_agree"], out["result_checked"])
        self.assertEqual(out["play_count_agree"], out["play_count_checked"])
        self.assertEqual(out["play_count_within1"], out["play_count_checked"])

    def test_check_game_flags_a_boundary_mismatch(self):
        g = Game()
        g[8]["fixed_drive"] = 3          # pretend nflverse glued BBB's play to AAA's drive 3
        out = CheckGame(g)
        self.assertLess(out["matched"], out["nfl_drives"])
        self.assertTrue(out["examples"])

    def test_report_writes_tidy_csv(self):
        with tempfile.TemporaryDirectory() as d:
            res = DriveCheck([2026], out_dir=d, base="dc", results_by_season={2026: [CheckGame(Game())]})
            self.assertEqual(res["report"]["all"]["match_rate"], 1.0)
            with open(os.path.join(d, "dc.csv"), encoding="utf-8") as fh:
                self.assertTrue(fh.readline().startswith("date,series,"))


def S(team, opp, epa, success, kind="pass", week=1, gid=G, date="2026-09-13"):
    return {"game_id": gid, "game_date": date, "week": week, "posteam": team, "defteam": opp,
            "pass": 1 if kind == "pass" else 0, "rush": 1 if kind == "rush" else 0, "epa": epa, "success": success}


class EfficiencyTests(unittest.TestCase):
    def plays(self):
        return [S("AAA", "BBB", 0.5, 1), S("AAA", "BBB", -0.1, 0, "rush"), S("BBB", "AAA", -0.4, 0),
                {"game_id": G, "game_date": "2026-09-13", "week": 1, "posteam": "AAA", "defteam": "BBB",
                 "pass": 0, "rush": 0, "epa": 3.0, "success": 1},                  # kick: excluded
                S("AAA", "CCC", 0.2, 1, week=2, gid="g2", date="2026-09-20"),
                S("CCC", "AAA", 0.6, 1, week=2, gid="g2", date="2026-09-20")]

    def test_weekly_offense_defense_and_season_to_date(self):
        w = WeeklyTable(self.plays())
        a1 = next(r for r in w if r["series"] == "AAA" and r["week"] == 1)
        self.assertEqual(a1["off_plays"], 2)
        self.assertAlmostEqual(a1["off_epa_play"], 0.2)
        self.assertAlmostEqual(a1["off_success_rate"], 0.5)
        self.assertAlmostEqual(a1["def_epa_play"], -0.4)
        self.assertAlmostEqual(a1["off_rush_epa_play"], -0.1)
        a2 = next(r for r in w if r["series"] == "AAA" and r["week"] == 2)
        self.assertAlmostEqual(a2["std_off_epa_play"], round(0.6 / 3, 4))
        self.assertAlmostEqual(a2["std_def_epa_play"], 0.1)
        self.assertEqual(list(w[0].keys())[:2], ["date", "series"])

    def test_season_table_sorted_by_net(self):
        t = SeasonTable(WeeklyTable(self.plays()))
        nets = [r["net_epa_play"] for r in t]
        self.assertEqual(nets, sorted(nets, reverse=True))

    def test_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            TeamEfficiency(2026, plays=self.plays(), out_dir=d, base="te")
            self.assertTrue(os.path.exists(os.path.join(d, "te.csv")))
            self.assertTrue(os.path.exists(os.path.join(d, "te.json")))


if __name__ == "__main__":
    unittest.main()
