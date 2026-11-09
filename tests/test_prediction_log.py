"""Append-only prediction log, started-game handling, per-run picks files and grading.
Uses synthetic seasons and temp dirs only (no network, no data/ folder)."""

import copy
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(__file__))

from SyntheticNfl import SyntheticSeason  # noqa: E402

import Backtest.PredictDay as PD  # noqa: E402
from Backtest.Features import BuildRows  # noqa: E402
from Backtest.FrozenConfig import FROZEN, ConfigHash  # noqa: E402
from Backtest.GradePredictions import GradePredictions  # noqa: E402
from Backtest.Version import ModelVersion, NewRunId  # noqa: E402

ET = ZoneInfo("America/New_York")
SEASON = 2016



def ReadText(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()

def Fixture():
    hist, prev = {}, []
    for s in (2013, 2014, 2015):
        g = SyntheticSeason(s)
        hist[s] = BuildRows(g, prev)
        prev = g
    games = SyntheticSeason(SEASON)
    days = sorted({g["gameday"] for g in games})
    day = next(d for d in days[len(days) // 2:] if datetime.fromisoformat(d).weekday() == 6)
    schedule = copy.deepcopy(games)
    for g in schedule:                       # the slate day and later are unplayed in the schedule
        if g["gameday"] >= day:
            g["home_score"] = g["away_score"] = None
    return hist, prev, games, schedule, day


class PredictionLogTests(unittest.TestCase):
    def setUp(self):
        self.hist, self.prior, self.final_games, self.schedule, self.day = Fixture()
        self.tmp = tempfile.TemporaryDirectory()
        self.pred_dir = os.path.join(self.tmp.name, "picks")
        self.log_dir = os.path.join(self.tmp.name, "predictions")
        sched = {"season": SEASON, "pulled_at_utc": "2016-11-06T12:00:00+00:00", "source": "synthetic", "games": self.schedule}
        self.patches = [
            mock.patch.object(PD, "LoadSchedule", lambda season, refresh: copy.deepcopy(sched)),
            mock.patch.object(PD, "LoadSeason", lambda season: copy.deepcopy(self.prior) if season == SEASON - 1 else []),
            mock.patch.object(PD, "AllRows", lambda first, last: copy.deepcopy(self.hist)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def Run(self, hhmm):
        now = datetime.fromisoformat(f"{self.day}T{hhmm}").replace(tzinfo=ET)
        return PD.PredictDay(self.day, refresh=False, now=now, pred_dir=self.pred_dir, log_dir=self.log_dir)

    def LogLines(self):
        with open(os.path.join(self.log_dir, f"{SEASON}.jsonl"), encoding="utf-8") as fh:
            return [json.loads(x) for x in fh if x.strip()]

    def test_started_games_are_not_scored_or_logged(self):
        out = self.Run("14:00:00")
        started = [p for p in out["games"] if p["started"]]
        open_ = [p for p in out["games"] if not p["started"]]
        self.assertTrue(started and open_)
        for p in started:
            self.assertEqual(p["recommended"]["action"], "STARTED")
            self.assertIsNone(p["recommended"]["prob"])
            self.assertTrue(all(v is None for v in p["strategies"].values()))
        self.assertIn("started, not predicted", out["markdown"])
        logged = {r["game_id"] for r in self.LogLines()}
        self.assertEqual(logged, {p["game_id"] for p in open_})

    def test_log_rows_have_required_fields_and_offsets(self):
        self.Run("12:00:00")
        rows = self.LogLines()
        self.assertTrue(rows)
        need = {"created_at", "game_id", "start_time", "pick", "model_prob", "market_price", "price_time",
                "source", "model_version", "run_id"}
        for r in rows:
            self.assertTrue(need <= set(r))
            self.assertIn(r["created_at"][-6:], ("-05:00", "-04:00"))
            self.assertIn(r["start_time"][-6:], ("-05:00", "-04:00"))
            self.assertLess(datetime.fromisoformat(r["created_at"]), datetime.fromisoformat(r["start_time"]))
            self.assertEqual(r["model_version"], ModelVersion())
            self.assertTrue(r["model_version"].startswith(f"{FROZEN['name']}@{ConfigHash()}+code:"))

    def test_rerun_appends_and_never_overwrites(self):
        a = self.Run("12:00:00")
        first = ReadText(os.path.join(self.log_dir, f"{SEASON}.jsonl"))
        b = self.Run("14:00:00")
        second = ReadText(os.path.join(self.log_dir, f"{SEASON}.jsonl"))
        self.assertTrue(second.startswith(first))
        self.assertGreater(len(second), len(first))
        self.assertNotEqual(a["meta"]["run_id"], b["meta"]["run_id"])
        files = sorted(os.listdir(self.pred_dir))
        self.assertIn(f"picks_{self.day}_120000.json", files)
        self.assertIn(f"picks_{self.day}_140000.json", files)
        self.assertIn(f"picks_{self.day}_latest.md", files)
        c = self.Run("14:00:00")                          # same second again: new file, nothing overwritten
        self.assertEqual(len([f for f in os.listdir(self.pred_dir) if f.endswith(".json")]), 3)
        self.assertNotEqual(c["meta"]["written"], b["meta"]["written"])
        latest = ReadText(os.path.join(self.pred_dir, f"picks_{self.day}_latest.md"))
        self.assertIn(c["meta"]["run_id"], latest)

    def test_grading_uses_only_pre_kickoff_rows(self):
        self.Run("12:00:00")
        log = os.path.join(self.log_dir, f"{SEASON}.jsonl")
        rows = self.LogLines()
        late = dict(rows[0], created_at=rows[0]["start_time"], run_id="late", pick="LATE")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(late) + "\n")
        before = ReadText(log)
        out = GradePredictions(SEASON, games=self.final_games, log_dir=self.log_dir, write=True)
        self.assertEqual(ReadText(log), before)   # grading never edits the log
        g = out["games"][rows[0]["game_id"]]
        self.assertEqual(g["excluded_late_rows"], 1)
        self.assertNotEqual(g["pick"], "LATE")
        self.assertEqual(g["status"], "final")
        self.assertIn(g["correct"], (True, False, None))
        self.assertTrue(os.path.exists(os.path.join(self.log_dir, f"grades_{SEASON}.json")))
        self.assertEqual(out["summary"]["excluded_late_rows"], 1)

    def test_frozen_config_drives_scoring(self):
        loose = copy.deepcopy(FROZEN)
        loose["value"]["min_edge"] = -1.0           # everything fires under a looser config
        out = self.Run("12:00:00")
        rows, _, _ = PD.SlateRows(self.schedule, self.prior, PD.PickSlate(self.schedule, self.day))
        rbs = copy.deepcopy(self.hist)
        rbs[SEASON] = rows
        PD.AttachModels(rbs)
        loose_bets = sum(p["recommended"]["action"] == "BET" for p in PD.ScoreRows(rows, loose))
        frozen_bets = sum(p["recommended"]["action"] == "BET" for p in out["games"] if not p["started"])
        self.assertEqual(loose_bets, len(rows))
        self.assertLessEqual(frozen_bets, loose_bets)
        self.assertEqual(out["meta"]["frozen_at"], "2026-11-08")

    def test_run_ids_are_unique(self):
        self.assertNotEqual(NewRunId(), NewRunId())


if __name__ == "__main__":
    unittest.main()
