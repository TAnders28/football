import math
import unittest
from datetime import datetime, timezone

from Backtest.ThinMarketRules import (
    TOTAL_KEYS, SPREAD_KEYS, TIMES, SIDES, FILLS, PRICE_BINS, PROX_TOL, PRIOR_TALLY,
    AdverseSplit, BetsFor, BuildRecs, ConfigGrid, ConfigId, CoverageGaps, DeflatedSharpe,
    GridSize, NearKey, PickBest, QuoteBundleAt, ScoreCfg, SideQuotes, WouldFill,
)


def _candle(end_ts, bid, ask, volume=None):
    return {"end_ts": end_ts, "yes_bid_close": bid, "yes_ask_close": ask, "volume": volume}


def _entry(ko, candles, **extra):
    e = {
        "ko": ko, "candles": candles, "ticker": extra.get("ticker", "KXNFLSPREAD-25SEP21ARISF-SF3"),
        "season": extra.get("season", 2025), "game_id": extra.get("game_id", "g1"),
        "market_kind": extra.get("kind", "spread"), "result": extra.get("result", "yes"),
        "home_team": "SF", "away_team": "ARI",
        "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "week": 3,
    }
    e.update({k: v for k, v in extra.items() if k not in ("ticker", "season", "game_id", "kind", "result")})
    return e


KO = datetime(2025, 9, 21, 20, 25, tzinfo=timezone.utc)


class GridTests(unittest.TestCase):
    def test_grid_size_is_pre_registered_product(self):
        n = (len(PRICE_BINS) * len(SIDES) * len(TIMES) * len(SPREAD_KEYS) * len(FILLS)
             + len(PRICE_BINS) * len(SIDES) * len(TIMES) * len(TOTAL_KEYS) * len(FILLS))
        self.assertEqual(n, 3240)
        self.assertEqual(GridSize(), 3240)
        self.assertEqual(len(ConfigGrid()), 3240)
        self.assertEqual(PRIOR_TALLY + GridSize(), 8619)
        kinds = {c["kind"] for c in ConfigGrid()}
        self.assertEqual(kinds, {"spread", "total"})
        # no total config carries a spread-only key
        for c in ConfigGrid():
            if c["kind"] == "total":
                self.assertIn(c["key"], TOTAL_KEYS)
            else:
                self.assertIn(c["key"], SPREAD_KEYS)

    def test_config_id_is_stable(self):
        cfg = {"kind": "spread", "side": "yes", "tmin": 60, "key": "prox_3",
               "fill": "mid", "bin_lo": 0.45}
        self.assertEqual(ConfigId(cfg), "spread|yes|T60|prox_3|mid|0.45-0.50")


class KeyAndQuoteTests(unittest.TestCase):
    def test_near_key_uses_abs_for_spreads_and_tolerance(self):
        self.assertTrue(NearKey(3.0, "prox_3", "spread"))
        self.assertTrue(NearKey(-3.0, "prox_3", "spread"))
        self.assertTrue(NearKey(3.5, "prox_3", "spread"))
        self.assertTrue(NearKey(2.5, "prox_3", "spread"))
        self.assertFalse(NearKey(4.0, "prox_3", "spread"))
        self.assertTrue(NearKey(7.0, "prox_3_or_7", "spread"))
        self.assertTrue(NearKey(44.0, "prox_44", "total"))
        self.assertTrue(NearKey(43.5, "prox_44", "total"))
        self.assertFalse(NearKey(-44.0, "prox_44", "total"))  # totals use signed strike as-is
        self.assertTrue(NearKey(99.0, "any", "total"))
        self.assertFalse(NearKey(None, "prox_3", "spread"))
        self.assertEqual(PROX_TOL, 0.5)

    def test_side_no_flips_bid_ask_and_result_via_bets(self):
        q = {"bid": 0.40, "ask": 0.50, "mid": 0.45, "width": 0.10}
        nq = SideQuotes(q, "no")
        self.assertAlmostEqual(nq["bid"], 0.50)
        self.assertAlmostEqual(nq["ask"], 0.60)
        self.assertAlmostEqual(nq["mid"], 0.55)


class MakerFillTests(unittest.TestCase):
    def test_would_fill_when_ask_trades_at_or_through_bid(self):
        t0 = KO.timestamp() - 60 * 60
        candles = [
            _candle(t0 - 30, 0.40, 0.44),
            _candle(t0 + 60, 0.39, 0.40),  # ask hits our 0.40 bid
        ]
        e = _entry(KO, candles)
        self.assertTrue(WouldFill(e, "yes", 0.40, 60))
        # no later trade at/through: ask stays above bid
        candles2 = [
            _candle(t0 - 30, 0.40, 0.44),
            _candle(t0 + 60, 0.41, 0.45),
        ]
        e2 = _entry(KO, candles2)
        self.assertFalse(WouldFill(e2, "yes", 0.40, 60))
        # through: later bid prints below our bid
        candles3 = [
            _candle(t0 - 30, 0.40, 0.44),
            _candle(t0 + 120, 0.38, 0.42),
        ]
        e3 = _entry(KO, candles3)
        self.assertTrue(WouldFill(e3, "yes", 0.40, 60))
        # candles after kickoff do not count
        candles4 = [
            _candle(t0 - 30, 0.40, 0.44),
            _candle(KO.timestamp() + 60, 0.10, 0.12),
        ]
        e4 = _entry(KO, candles4)
        self.assertFalse(WouldFill(e4, "yes", 0.40, 60))

    def test_would_fill_no_side_uses_flipped_quotes(self):
        # yes ask at entry 0.60 → no bid = 0.40. Fill no if later no_ask=1-yes_bid <= 0.40
        # i.e. yes_bid >= 0.60
        t0 = KO.timestamp() - 60 * 60
        candles = [
            _candle(t0 - 30, 0.50, 0.60),
            _candle(t0 + 60, 0.61, 0.65),
        ]
        e = _entry(KO, candles)
        self.assertTrue(WouldFill(e, "no", 0.40, 60))
        candles2 = [
            _candle(t0 - 30, 0.50, 0.60),
            _candle(t0 + 60, 0.51, 0.59),
        ]
        e2 = _entry(KO, candles2)
        self.assertFalse(WouldFill(e2, "no", 0.40, 60))

    def test_adverse_split_separates_fill_from_no_fill(self):
        t60 = KO.timestamp() - 60 * 60
        # would fill
        e1 = _entry(KO, [_candle(t60 - 30, 0.40, 0.44, 10), _candle(t60 + 60, 0.39, 0.40)],
                    game_id="g1", result="yes", ticker="KXNFLSPREAD-25SEP21ARISF-SF3")
        # would not
        e2 = _entry(KO, [_candle(t60 - 30, 0.40, 0.44, 10), _candle(t60 + 60, 0.41, 0.45)],
                    game_id="g2", result="no", ticker="KXNFLSPREAD-25SEP21ARISF-SF3")
        recs = BuildRecs([e1, e2], markets={})
        self.assertEqual(len(recs), 2)
        cfg = {"kind": "spread", "side": "yes", "tmin": 60, "key": "prox_3",
               "fill": "bid", "bin_lo": 0.40}
        # bids are 0.40, bin 0.40-0.45
        adv = AdverseSplit(recs, cfg)
        self.assertEqual(adv["2025"]["n_fill"], 1)
        self.assertEqual(adv["2025"]["n_no_fill"], 1)
        # maker bets keep only would-fill
        bets = BetsFor(recs, cfg)
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0]["event_id"], "g1")


class SelectionTests(unittest.TestCase):
    def test_pick_best_uses_2025_only(self):
        weak_2025 = {
            "id": "a", "kind": "spread", "side": "yes", "tmin": 60, "key": "any",
            "fill": "mid", "bin_lo": 0.5, "n_2025": 40, "edge_2025": 0.01,
            "n_2026": 40, "edge_2026": 0.50,
        }
        strong_2025 = {
            "id": "b", "kind": "total", "side": "yes", "tmin": 60, "key": "any",
            "fill": "mid", "bin_lo": 0.5, "n_2025": 40, "edge_2025": 0.04,
            "n_2026": 40, "edge_2026": -0.20,
        }
        picked = PickBest([weak_2025, strong_2025])
        self.assertEqual(picked["id"], "b")
        # n < 20 is ineligible even with a huge 2025 edge
        tiny = dict(strong_2025, id="c", n_2025=19, edge_2025=0.80)
        picked2 = PickBest([weak_2025, tiny])
        self.assertEqual(picked2["id"], "a")

    def test_selected_flag_requires_min_n_and_3c(self):
        t60 = KO.timestamp() - 60 * 60
        recs = []
        for i in range(25):
            e = _entry(KO, [_candle(t60 - 30, 0.40, 0.42)],
                       game_id=f"g{i}", result="yes",
                       ticker="KXNFLSPREAD-25SEP21ARISF-SF3")
            recs.extend(BuildRecs([e]))
        cfg = {"kind": "spread", "side": "yes", "tmin": 60, "key": "prox_3",
               "fill": "mid", "bin_lo": 0.40}
        row = ScoreCfg(recs, cfg)
        # mid = 0.41, all wins, edge = 1-0.41-0.02 = 0.57, n=25 → selected
        self.assertEqual(row["n_2025"], 25)
        self.assertGreater(row["edge_2025"], 0.03)
        self.assertTrue(row["selected"])
        cfg_empty_bin = dict(cfg, bin_lo=0.90)
        row2 = ScoreCfg(recs, cfg_empty_bin)
        self.assertEqual(row2["n_2025"], 0)
        self.assertFalse(row2["selected"])


class DsrAndCoverageTests(unittest.TestCase):
    def test_dsr_falls_as_trial_count_rises(self):
        rng = __import__("numpy").random.default_rng(0)
        rets = rng.normal(0.02, 0.05, size=200)
        a = DeflatedSharpe(rets, n_trials=2)
        b = DeflatedSharpe(rets, n_trials=3000)
        self.assertIsNotNone(a["dsr"])
        self.assertIsNotNone(b["dsr"])
        self.assertGreater(a["dsr"], b["dsr"])
        self.assertGreaterEqual(a["dsr"], 0.0)
        self.assertLessEqual(a["dsr"], 1.0)

    def test_coverage_gaps_flag_missing_season_and_kind(self):
        e = _entry(KO, [_candle(KO.timestamp() - 90, 0.4, 0.5)], season=2025, kind="spread")
        missing = CoverageGaps([e])
        self.assertIn("2026 candles", missing)
        self.assertIn("KXNFLTOTAL", missing)
        self.assertNotIn("2025 candles", missing)
        self.assertNotIn("KXNFLSPREAD", missing)
        self.assertEqual(CoverageGaps([]), ["2025 candles", "2026 candles", "KXNFLSPREAD", "KXNFLTOTAL"])


if __name__ == "__main__":
    unittest.main()
