"""Shared live file for Degen: rows, rules and the writer. Temp dirs only, no network."""

import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

from test_live_readiness import NOW, _game, _quote, _state  # noqa: E402

import Backtest.AssessLive as AL  # noqa: E402
from Backtest.AssessLive import FetchError  # noqa: E402
from Backtest.LiveFile import FEE_BASIS, LiveRows, WriteLiveFile  # noqa: E402
from Backtest.LiveReadiness import AssessGame, EmptyReport  # noqa: E402

FIELDS = {"observed_at", "game_id", "teams", "score", "period", "clock", "state_time", "venue", "market", "side",
          "bid", "ask", "size", "price_time", "model_prob", "edge", "action", "reason", "fee_basis", "quote_qualified"}



def ReadText(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()

def Report(*games):
    rep = EmptyReport(NOW, "2026-11-08", [])
    rep["games"] = list(games)
    return rep


def Row(game, **kw):
    return LiveRows(Report(game), **kw)[0]


class LiveFileTests(unittest.TestCase):
    def test_fresh_quote_without_model_is_pass_no_live_model(self):
        row = Row(AssessGame(_game(), NOW))
        self.assertTrue(FIELDS <= set(row))
        self.assertEqual((row["action"], row["reason"]), ("PASS", "no live model"))
        self.assertTrue(row["quote_qualified"])
        self.assertEqual(row["fee_basis"], FEE_BASIS)
        self.assertEqual((row["bid"], row["ask"], row["size"]), (0.85, 0.86, 80.0))
        self.assertIsNone(row["model_prob"])
        self.assertIsNone(row["edge"])
        self.assertEqual(row["teams"], "TB@CHI")
        self.assertEqual(row["score"], "0-7")
        self.assertTrue(row["price_time"].endswith("-05:00"))   # source time, shown in ET
        self.assertEqual(row["price_time"], "2026-11-08T20:44:55-05:00")

    def test_stale_quote_is_pass_stale(self):
        old = NOW - timedelta(minutes=10)
        row = Row(AssessGame(_game(quote=_quote(source_timestamp=old)), NOW))
        self.assertEqual((row["action"], row["reason"]), ("PASS", "stale"))
        self.assertFalse(row["quote_qualified"])

    def test_missing_price_time_is_stale_not_filled_from_retrieval(self):
        row = Row(AssessGame(_game(quote=_quote(source_timestamp=None)), NOW))
        self.assertEqual(row["reason"], "stale")
        self.assertIsNone(row["price_time"])
        self.assertFalse(row["quote_qualified"])

    def test_sixty_second_gap_rule_even_when_repo_limits_are_relaxed(self):
        relaxed = {"max_live_age_s": 300, "max_sync_skew_s": 300, "max_future_s": 90, "max_pregame_age_s": 900}
        game = AssessGame(_game(state=_state(source_timestamp=NOW - timedelta(seconds=75)),
                                quote=_quote(source_timestamp=NOW - timedelta(seconds=5))), NOW, thresholds=relaxed)
        self.assertTrue(game["quotes"][0]["executable_ask"])     # the repo check alone would have passed it
        row = Row(game)
        self.assertEqual(row["reason"], "stale")
        self.assertFalse(row["quote_qualified"])
        self.assertAlmostEqual(row["price_state_gap_s"], 70.0)

    def test_stricter_repo_limits_are_kept(self):
        # 30 s apart is inside 60 s but outside the repo's 20 s sync limit: still stale.
        game = AssessGame(_game(state=_state(source_timestamp=NOW - timedelta(seconds=35)),
                                quote=_quote(source_timestamp=NOW - timedelta(seconds=5))), NOW)
        row = Row(game)
        self.assertEqual(row["reason"], "stale")
        self.assertIn("sync_skew", row["stale_detail"])
        # 50 s old board breaks the 45 s live-age limit.
        row = Row(AssessGame(_game(state=_state(source_timestamp=NOW - timedelta(seconds=50)),
                                   quote=_quote(source_timestamp=NOW - timedelta(seconds=50))), NOW))
        self.assertEqual(row["reason"], "stale")
        self.assertFalse(row["quote_qualified"])

    def test_finished_game_with_stale_boards_is_final_and_pass(self):
        old = NOW - timedelta(hours=4)
        game = AssessGame(_game(states=[
            _state(status="closed", period=4, clock="00:00", away_score=34, home_score=17, source_timestamp=old),
            _state(venue="polymarket", status="ended", ended=True, period="VFT", clock=None,
                   away_score=34, home_score=17, source_timestamp=None),
        ], quote=_quote(status="closed", source_timestamp=old)), NOW)
        self.assertEqual(game["phase"], "final")
        row = Row(game)
        self.assertEqual((row["phase"], row["action"], row["reason"]), ("final", "PASS", "final"))
        self.assertFalse(row["quote_qualified"])

    def test_validated_live_model_edge_uses_tims_fee_estimate(self):
        model = {"scope": "game_state_adjusted", "validated": True, "side_team": "CHI",
                 "p_yes": 0.5, "p_tie": 0.0, "p_nonbinary": 0.0}
        quote = _quote(book={"bids": {"best": {"price": 0.39, "size": 5}, "top": [], "level_count": 1},
                             "asks": {"best": {"price": 0.40, "size": 8}, "top": [], "level_count": 1}},
                       summary_bid=0.39, summary_ask=0.40)
        row = Row(AssessGame(_game(quote=quote), NOW, model=model))
        self.assertAlmostEqual(row["edge"], 0.5 - 0.40 - 0.02)
        self.assertIn("Tim's fee estimate", row["edge_basis"])
        self.assertEqual(row["action"], "BET")
        pregame = {"scope": "pregame", "validated": True, "name": "MarginModel", "p_yes": 0.9, "side_team": "CHI"}
        row = Row(AssessGame(_game(quote=quote), NOW, model=pregame))
        self.assertEqual((row["action"], row["reason"], row["edge"]), ("PASS", "no live model", None))

    def test_writer_appends_jsonl_and_overwrites_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = Report(AssessGame(_game(), NOW), AssessGame(_game(quote=_quote(source_timestamp=None)), NOW))
            a = WriteLiveFile(rep, tmp, run_id="run-a", model_version="mv")
            b = WriteLiveFile(rep, tmp, run_id="run-b", model_version="mv")
            self.assertEqual(a["jsonl"], os.path.join(tmp, "2026-11-08_live", "2026-11-08_live.jsonl"))
            self.assertEqual(a["latest"], os.path.join(tmp, "2026-11-08_live", "2026-11-08_live_latest.md"))
            lines = [json.loads(x) for x in ReadText(a["jsonl"]).splitlines()]
            self.assertEqual(len(lines), 4)
            self.assertEqual([r["run_id"] for r in lines], ["run-a", "run-a", "run-b", "run-b"])
            latest = ReadText(b["latest"])
            self.assertIn("run-b", latest)
            self.assertNotIn("run-a", latest)
            self.assertIn("## Blockers", latest)
            self.assertIn("no live model", latest)

    def test_assess_live_writes_live_file_to_the_given_dir(self):
        class Boom:
            calls = 0

            def get_json(self, url):
                raise FetchError(f"HTTP 503 {url}")

            def get_bytes(self, url):
                raise FetchError(f"HTTP 503 {url}")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(AL, "OUT_DIR", os.path.join(tmp, "assess")):
            rep = AL.AssessLive(date="2026-11-08", write=True, fetcher=Boom(), now=NOW, live_dir=os.path.join(tmp, "live"))
            lf = rep["meta"]["live_file"]
            self.assertTrue(lf["jsonl"].startswith(tmp))
            self.assertTrue(os.path.exists(lf["latest"]))
            self.assertEqual(lf["rows"], 0)
            self.assertIn("503", ReadText(lf["latest"]))
            self.assertTrue(rep["meta"]["run_id"] and rep["meta"]["model_version"])
            saved = json.loads(ReadText(rep["meta"]["written"].split(".json")[0] + ".json"))
            self.assertEqual(saved["meta"]["run_id"], rep["meta"]["run_id"])


if __name__ == "__main__":
    unittest.main()
