"""Fail-closed cases for the read-only live report. No network."""

import json
import unittest
from datetime import datetime, timedelta, timezone

from Backtest.AssessLive import FetchError, _kalshi_quote
from Backtest.LiveReadiness import AssessGame, BookSide, PREGAME_MODEL_NOTE, SettlementFromRules
from Backtest.PredictDay import SeasonOf


NOW = datetime(2026, 11, 9, 1, 45, tzinfo=timezone.utc)


def _state(**over):
    row = {
        "venue": "kalshi",
        "url": "https://external-api.kalshi.com/trade-api/v2/live_data/milestone/m1",
        "away": "TB", "home": "CHI",
        "status": "inprogress", "period": 1, "clock": "7:24",
        "away_score": 0, "home_score": 7,
        "source_timestamp": NOW - timedelta(seconds=5),
        "observed_at": NOW,
        "delayed": None,
    }
    row.update(over)
    return row


def _quote(**over):
    row = {
        "venue": "kalshi",
        "event_id": "KXNFLGAME-26NOV08TBCHI",
        "event_url": "https://external-api.kalshi.com/trade-api/v2/events/KXNFLGAME-26NOV08TBCHI",
        "market_id": "KXNFLGAME-26NOV08TBCHI-CHI",
        "market_url": "https://external-api.kalshi.com/trade-api/v2/markets/KXNFLGAME-26NOV08TBCHI-CHI",
        "outcome_kind": "moneyline_game_winner",
        "side_team": "CHI", "side_name": "Chicago",
        "away": "TB", "home": "CHI",
        "status": "active",
        "book": {
            "bids": {"best": {"price": 0.85, "size": 100.0}, "top": [{"price": 0.85, "size": 100.0}], "level_count": 1},
            "asks": {"best": {"price": 0.86, "size": 80.0}, "top": [{"price": 0.86, "size": 80.0}], "level_count": 1},
        },
        "summary_bid": 0.85, "summary_ask": 0.86,
        "last_trade": 0.86,
        "source_timestamp": NOW - timedelta(seconds=5),
        "observed_at": NOW,
        "fee": {"status": "unknown", "taker_fee_per_contract": None},
        "settlement": {"status": "unknown", "closed_form": False, "source_text_present": False},
    }
    row.update(over)
    return row


def _game(quote=None, state=None, **over):
    row = {"away": "TB", "home": "CHI", "quotes": [_quote()] if quote is None else [quote],
           "states": [_state()] if state is None else [state]}
    row.update(over)
    return row


class ReadinessTests(unittest.TestCase):
    def test_book_best_price_ignores_array_order(self):
        bids = BookSide([["0.1000", "10"], ["0.1400", "20"]], "bid")
        asks = BookSide([{"price": "0.99", "size": "5"}, {"price": "0.10", "size": "7"}], "ask")
        self.assertEqual(bids["best"]["price"], 0.14)
        self.assertEqual(asks["best"]["price"], 0.10)

    def test_metadata_updated_time_is_not_a_quote_timestamp(self):
        market = {
            "ticker": "KXNFLGAME-26NOV08TBCHI-TB", "yes_sub_title": "Tampa Bay",
            "title": "Tampa Bay wins", "status": "active",
            "rules_primary": "If Tampa Bay wins the Tampa Bay vs Chicago Pro Football game, then the market resolves to Yes.",
            "yes_bid_dollars": "0.1400", "yes_ask_dollars": "0.1500",
            "yes_bid_size_fp": "10.00", "yes_ask_size_fp": "12.00",
            "last_price_dollars": "0.1500", "updated_time": "2026-10-27T04:06:00.276368Z",
            "volume_fp": "1.00", "open_interest_fp": "1.00",
        }
        book = {"orderbook_fp": {"yes_dollars": [["0.1400", "10.00"]], "no_dollars": [["0.8500", "12.00"]]}}
        quote = _kalshi_quote(market, book, "TB", "TB", "CHI", "KXNFLGAME-26NOV08TBCHI", NOW, {"fee": {}, "settlement": {}})
        self.assertIsNone(quote["source_timestamp"])
        self.assertTrue(quote["metadata_updated_at"].startswith("2026-10-27"))
        assessed = AssessGame(_game(quote=quote), NOW, model=None)
        self.assertIn("quote_missing_timestamp", assessed["fail_reasons"])
        self.assertNotIn("quote_stale_timestamp", assessed["fail_reasons"])
        self.assertIsNone(assessed["probability"])

    def test_stale_missing_and_future_timestamps(self):
        stale = AssessGame(_game(state=_state(source_timestamp=NOW - timedelta(minutes=10)),
                                 quote=_quote(source_timestamp=NOW - timedelta(minutes=10))), NOW)
        self.assertIn("state_stale_timestamp", stale["fail_reasons"])
        self.assertIn("quote_stale_timestamp", stale["fail_reasons"])
        self.assertIsNone(stale["probability"])
        self.assertIsNone(stale["quotes"][0]["gross_ev"])

        missing = AssessGame(_game(state=_state(source_timestamp=None), quote=_quote(source_timestamp=None)), NOW)
        self.assertIn("state_missing_timestamp", missing["fail_reasons"])
        self.assertIn("quote_missing_timestamp", missing["fail_reasons"])
        self.assertIsNone(missing["quotes"][0]["gross_ev"])

        future = AssessGame(_game(state=_state(source_timestamp=NOW + timedelta(hours=2)),
                                  quote=_quote(source_timestamp=NOW + timedelta(hours=2))), NOW)
        self.assertIn("state_future_timestamp", future["fail_reasons"])
        self.assertIn("quote_future_timestamp", future["fail_reasons"])
        self.assertEqual(future["action"], "PASS")

    def test_quote_state_skew_and_event_mismatch(self):
        skewed = AssessGame(
            _game(state=_state(source_timestamp=NOW - timedelta(seconds=70)),
                  quote=_quote(source_timestamp=NOW - timedelta(seconds=10))),
            NOW, thresholds={"max_live_age_s": 200, "max_sync_skew_s": 20, "max_future_s": 90, "max_pregame_age_s": 900},
        )
        self.assertIn("sync_skew", skewed["fail_reasons"])
        self.assertFalse(skewed["quotes"][0]["executable_ask"])
        self.assertIsNone(skewed["quotes"][0]["gross_ev"])

        mismatched = AssessGame(_game(quote=_quote(away="DAL", home="PHI", side_team="PHI")), NOW)
        self.assertIn("event_mismatch", mismatched["fail_reasons"])
        self.assertIsNone(mismatched["probability"])

    def test_no_depth_wrong_outcome_suspended_and_closed(self):
        empty = AssessGame(_game(quote=_quote(book={"bids": {"best": None, "level_count": 0, "top": []},
                                                    "asks": {"best": None, "level_count": 0, "top": []}},
                                            summary_bid=None, summary_ask=None)), NOW)
        self.assertIn("no_depth", empty["fail_reasons"])
        self.assertIsNone(empty["quotes"][0]["gross_ev"])

        spread = AssessGame(_game(quote=_quote(outcome_kind="spread")), NOW)
        self.assertIn("wrong_outcome", spread["fail_reasons"])
        self.assertFalse(spread["quotes"][0]["executable_ask"])

        suspended = AssessGame(_game(quote=_quote(status="paused")), NOW)
        self.assertIn("suspended", suspended["fail_reasons"])
        self.assertFalse(suspended["quotes"][0]["executable_ask"])

        closed = AssessGame(_game(quote=_quote(status="closed"), state=_state(status="closed", clock=None)), NOW)
        self.assertIn("closed", closed["fail_reasons"])
        self.assertEqual(closed["action"], "PASS")

    def test_pregame_model_is_not_compared_to_the_quote(self):
        # The number 0.72 is the shape of a pre-game lean. It must not come back as a live probability.
        model = {"scope": "pregame", "validated": True, "name": "MarginModel", "p_yes": 0.72, "side_team": "CHI"}
        out = AssessGame(_game(), NOW, model=model)
        self.assertIn("pregame_model_misuse", out["fail_reasons"])
        self.assertIsNone(out["probability"])
        self.assertIsNone(out["quotes"][0]["probability"])
        self.assertIsNone(out["quotes"][0]["gross_ev"])
        self.assertIsNone(out["quotes"][0]["net_ev"])
        self.assertNotIn("0.72", json.dumps(out))
        self.assertIn("pre-game", out["model"]["reason"])
        self.assertFalse(out["model"]["applied"])

    def test_synced_quote_still_has_no_ev_without_a_live_model(self):
        out = AssessGame(_game(), NOW, model=None)
        self.assertIn("no_validated_live_model", out["fail_reasons"])
        self.assertEqual(out["action"], "PASS")
        self.assertEqual(out["phase"], "live")
        self.assertIsNone(out["probability"])
        self.assertIsNone(out["quotes"][0]["gross_ev"])
        self.assertIsNone(out["quotes"][0]["net_ev"])

    def test_fee_or_settlement_unknown_blanks_ev_even_with_a_validated_model(self):
        model = {"scope": "game_state_adjusted", "validated": True, "side_team": "CHI",
                 "p_yes": 0.55, "p_tie": 0.0, "p_nonbinary": 0.0}
        settlement = {"status": "known", "closed_form": True, "source_text_present": True,
                      "yes_payoff": 1.0, "tie_payoff": 0.5, "no_payoff": 0.0}
        fee_unknown = AssessGame(_game(quote=_quote(fee={"status": "unknown", "taker_fee_per_contract": None},
                                                    settlement=settlement)), NOW, model=model)
        self.assertIn("fee_unknown", fee_unknown["fail_reasons"])
        self.assertIsNone(fee_unknown["quotes"][0]["gross_ev"])
        self.assertIsNone(fee_unknown["quotes"][0]["net_ev"])

        settle_unknown = AssessGame(_game(quote=_quote(
            fee={"status": "known", "taker_fee_per_contract": 0.01},
            settlement={"status": "unknown", "closed_form": False, "source_text_present": False},
        )), NOW, model=model)
        self.assertIn("settlement_unknown", settle_unknown["fail_reasons"])
        self.assertIsNone(settle_unknown["quotes"][0]["gross_ev"])
        self.assertIsNone(settle_unknown["quotes"][0]["net_ev"])

    def test_ev_only_when_every_gate_passes_and_action_stays_pass(self):
        model = {"scope": "game_state_adjusted", "validated": True, "side_team": "CHI",
                 "p_yes": 0.5, "p_tie": 0.0, "p_nonbinary": 0.0}
        quote = _quote(
            fee={"status": "known", "taker_fee_per_contract": 0.01},
            settlement={"status": "known", "closed_form": True, "source_text_present": True,
                        "yes_payoff": 1.0, "tie_payoff": 0.5, "no_payoff": 0.0},
            book={"bids": {"best": {"price": 0.39, "size": 5}, "top": [], "level_count": 1},
                  "asks": {"best": {"price": 0.40, "size": 8}, "top": [], "level_count": 1}},
            summary_bid=0.39, summary_ask=0.40,
        )
        out = AssessGame(_game(quote=quote), NOW, model=model)
        self.assertEqual(out["action"], "PASS")
        self.assertAlmostEqual(out["quotes"][0]["gross_ev"], 0.10)
        self.assertAlmostEqual(out["quotes"][0]["net_ev"], 0.09)
        self.assertEqual(out["quotes"][0]["probability"], 0.5)

    def test_final_vft_matches_closed_quarter_without_hiding_a_real_split(self):
        def board(status, period, away_score, home_score):
            return _state(status=status, period=period, clock="00:00", away_score=away_score, home_score=home_score)

        regulation = AssessGame(_game(states=[board("closed", 4, 34, 17), board("ended", "VFT", 34, 17)]), NOW)
        self.assertNotIn("state_disagreement", regulation["fail_reasons"])
        self.assertNotIn("state_mismatch", regulation["fail_reasons"])

        overtime = AssessGame(_game(states=[board("closed", 5, 22, 23), board("ended", "VFT", 22, 23)]), NOW)
        self.assertNotIn("state_disagreement", overtime["fail_reasons"])

        live_against_final = AssessGame(_game(states=[board("inprogress", 1, 0, 14), board("ended", "VFT", 0, 14)]), NOW)
        self.assertIn("state_disagreement", live_against_final["fail_reasons"])

        different_score = AssessGame(_game(states=[board("closed", 4, 34, 17), board("ended", "VFT", 31, 17)]), NOW)
        self.assertIn("state_disagreement", different_score["fail_reasons"])

        live_quarters = AssessGame(_game(states=[board("inprogress", "Q1", 0, 7), board("inprogress", "Q2", 0, 7)]), NOW)
        self.assertIn("state_disagreement", live_quarters["fail_reasons"])

        early_final = AssessGame(_game(states=[board("closed", 1, 0, 0), board("ended", "VFT", 0, 0)]), NOW)
        self.assertIn("state_disagreement", early_final["fail_reasons"])

    def test_fresh_quote_still_passes_without_a_live_model(self):
        out = AssessGame(_game(), NOW, model=None)
        self.assertEqual(out["action"], "PASS")
        self.assertIsNone(out["probability"])
        self.assertIsNone(out["quotes"][0]["gross_ev"])
        self.assertIsNone(out["quotes"][0]["net_ev"])

    def test_old_final_source_time_is_not_made_current_by_retrieval(self):
        old = NOW - timedelta(hours=5)
        out = AssessGame(_game(state=_state(
            status="closed", period=4, clock="00:00", away_score=34, home_score=17,
            source_timestamp=old, observed_at=NOW,
        ), quote=_quote(source_timestamp=old, observed_at=NOW)), NOW)
        state = out["states"][0]
        self.assertIn("state_stale_timestamp", out["fail_reasons"])
        self.assertFalse(state["usable"])
        self.assertIsNone(out["sync"]["accepted_state_source"])
        self.assertEqual(state["source_timestamp"], "2026-11-08T20:45:00.000Z")
        self.assertEqual(state["observed_at_utc"], "2026-11-09T01:45:00.000Z")
        self.assertNotEqual(state["source_timestamp"], state["observed_at_utc"])
        self.assertIsNone(out["probability"])
        self.assertIsNone(out["quotes"][0]["gross_ev"])

        fresh = AssessGame(_game(state=_state(
            status="closed", period=4, clock="00:00", away_score=34, home_score=17,
            source_timestamp=NOW - timedelta(seconds=10), observed_at=NOW,
        )), NOW)
        self.assertNotIn("state_stale_timestamp", fresh["fail_reasons"])
        self.assertTrue(fresh["states"][0]["usable"])

    def test_cancellation_and_tie_paths_are_not_closed_form(self):
        text = (
            "All overtime periods are included in the winner. "
            "If no Tie strike is listed, Yes holders are paid $0.50 and No holders are paid $0.50. "
            "If the game is cancelled, the market resolves at the last fair price."
        )
        parsed = SettlementFromRules(text)
        self.assertEqual(parsed["status"], "known")
        self.assertFalse(parsed["closed_form"])
        self.assertEqual(parsed["tie_payoff"], 0.5)
        self.assertTrue(parsed["overtime"].startswith("All overtime"))
        self.assertIn("last fair price", parsed["cancellation"])
        fifty = SettlementFromRules(
            "Overtime is included if played. If the game ends in a tie, this market will resolve 50-50. "
            "If the game is canceled entirely, this market will resolve 50-50."
        )
        self.assertFalse(fifty["closed_form"])
        self.assertEqual(fifty["tie_payoff"], 0.5)

    def test_last_trade_outside_the_book_is_not_the_ask(self):
        out = AssessGame(_game(quote=_quote(last_trade=0.13)), NOW)
        quote = out["quotes"][0]
        self.assertIn("last_trade_outside_book", quote["notes"])
        self.assertEqual(quote["ask"]["price"], 0.86)
        self.assertIsNone(quote["gross_ev"])
        self.assertIsNone(out["probability"])

    def test_provider_delay_flag_fails_closed(self):
        out = AssessGame(_game(state=_state(delayed=True)), NOW)
        self.assertIn("state_provider_marked_delayed", out["fail_reasons"])
        self.assertIsNone(out["sync"]["accepted_state_source"])

    def test_source_failure_writes_no_fabricated_ev(self):
        from Backtest.AssessLive import AssessLive

        class Boom:
            calls = 0

            def get_json(self, url):
                raise FetchError(f"HTTP 503 {url}")

            def get_bytes(self, url):
                raise FetchError(f"HTTP 503 {url}")

        report = AssessLive(date="2026-11-08", write=False, fetcher=Boom(), now=NOW)
        self.assertEqual(report["games"], [])
        self.assertTrue(report["meta"]["source_errors"])
        self.assertIn("503", report["markdown"])
        self.assertNotIn("gross_ev", json.dumps(report["games"]))
        self.assertFalse(report["meta"]["validated"]["ev"])
        self.assertFalse(report["meta"]["validated"]["pregame_model_compared_to_quotes"])

    def test_predict_day_season_helper_unchanged(self):
        self.assertEqual(SeasonOf("2026-11-08"), 2026)
        self.assertEqual(SeasonOf("2026-02-01"), 2025)
        self.assertIn("pre-game", PREGAME_MODEL_NOTE)


if __name__ == "__main__":
    unittest.main()
