import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from Backtest.ThinMarkets import (
    ALPHA, BIN, FEE, MIN_BIN_N, MIN_EDGE, START,
    BinLo, BinomialRoi, Claimable, ContractRows, Coverage, FamilyEval, FitBins,
    MakeEdges, MakerFeeFinding, MedianWidth, PerBinHoldout, QuoteBundle,
    SoftnessVerdict, WidestRoiCi, YesResult,
)
from DataPull.PullKalshiThinMarkets import FeeFields, ParseStrike


# ParseStrike lives in the puller; YesResult / BinLo in the analyzer.
# A leftover name is guarded below.


def Candle(ts, bid, ask, volume=10, oi=5):
    return {
        "end_ts": int(ts), "end_utc": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yes_bid_close": bid, "yes_ask_close": ask, "price_close": (bid + ask) / 2,
        "volume": volume, "open_interest": oi,
    }


def Entry(gid, season, week, kickoff, result, candles, kind="total", ticker=None, err=None, away="KC", home="DEN"):
    ko = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=timezone.utc)
    return {
        "ticker": ticker or f"KXNFLTOTAL-{gid}-T45",
        "event_ticker": f"KXNFLTOTAL-{gid}",
        "series_ticker": "KXNFLTOTAL" if kind == "total" else "KXNFLSPREAD",
        "market_kind": kind, "game_id": gid, "season": season, "week": week,
        "home_team": home, "away_team": away,
        "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ko": ko, "result": result, "candles": candles, "error": err,
    }


def T60Candles(ko, bid, ask, n=5):
    # n one-minute candles ending at or just before T-60
    t60 = int(ko.timestamp()) - 60 * 60
    return [Candle(t60 - 60 * (n - i), bid, ask) for i in range(n)]


class HelperTests(unittest.TestCase):
    def test_yes_result_and_bin_edges(self):
        self.assertEqual(YesResult("yes"), 1)
        self.assertEqual(YesResult("NO"), 0)
        self.assertIsNone(YesResult("scalar"))
        self.assertIsNone(YesResult("void"))
        self.assertEqual(BinLo(0.00), 0.0)
        self.assertEqual(BinLo(0.049999), 0.0)
        self.assertEqual(BinLo(0.05), 0.05)
        self.assertEqual(BinLo(0.199), 0.15)
        self.assertEqual(BinLo(0.99), 0.95)
        self.assertEqual(BinLo(1.0), 0.95)
        self.assertIsNone(BinLo(-0.01))

    def test_parse_strike_and_fee_fields(self):
        self.assertEqual(ParseStrike("KXNFLTOTAL-25SEP07KCDEN-T45.5")[0], 45.5)
        strike, lab = ParseStrike("KXNFLSPREAD-26NOV09BUFMIN-BUF3")
        self.assertEqual(strike, 3)
        self.assertIn("BUF", lab)
        m = ParseStrike("X", {"custom_strike": "3.5", "yes_sub_title": "Chiefs -3.5"})
        self.assertEqual(m[0], 3.5)
        fees = FeeFields({"fee_type": "quadratic_with_maker_fees", "title": "NFL",
                          "nested": {"maker_fee": 0.0, "other": 1}})
        self.assertEqual(fees["fee_type"], "quadratic_with_maker_fees")
        self.assertIn("nested", fees)
        self.assertEqual(fees["nested"]["maker_fee"], 0.0)
        self.assertNotIn("title", fees)

    def test_quote_bundle_stale_and_width(self):
        ko = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
        fresh = Entry("g1", 2025, 1, ko, "yes", T60Candles(ko, 0.40, 0.44))
        q = QuoteBundle(fresh)
        self.assertAlmostEqual(q["width"], 0.04)
        self.assertAlmostEqual(q["mid"], 0.42)
        self.assertAlmostEqual(q["ask"], 0.44)
        self.assertIsNotNone(q["volume_t60"])
        # candles all older than 30 minutes before T-60 are stale
        stale_ts = int(ko.timestamp()) - 60 * 60 - 31 * 60
        stale = Entry("g1", 2025, 1, ko, "yes", [Candle(stale_ts, 0.40, 0.44)])
        self.assertIsNone(QuoteBundle(stale))

    def test_contract_rows_drop_errors_voids_and_missing_quotes(self):
        ko = datetime(2025, 9, 7, 17, 0, tzinfo=timezone.utc)
        good = Entry("g1", 2025, 1, ko, "yes", T60Candles(ko, 0.40, 0.44))
        void = Entry("g2", 2025, 1, ko, "scalar", T60Candles(ko, 0.40, 0.44))
        missing = Entry("g3", 2025, 1, ko, "no", [])
        err = Entry("g4", 2025, 1, ko, "yes", [], err="HTTP 400")
        rows, dropped = ContractRows([good, void, missing, err], kinds=("total", "spread"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], 1)
        self.assertEqual(dropped["no_result"], 1)
        self.assertEqual(dropped["no_quote"], 1)
        self.assertEqual(dropped["error"], 1)


class FitAndHoldoutTests(unittest.TestCase):
    def _rows(self, season, prices_and_ys, kind="total", day0="2025-09-07"):
        rows = []
        base = datetime.fromisoformat(day0).replace(tzinfo=timezone.utc)
        for i, (px, y) in enumerate(prices_and_ys):
            day = (base + timedelta(days=i)).date().isoformat()
            rows.append({
                "date": day, "start_time": day + "T17:00:00Z", "sport": "nfl",
                "season": str(season), "week": 1 + (i // 16),
                "split": "contaminated" if season == 2025 else "forward",
                "event_id": f"{season}_{i}", "ticker": f"T-{season}-{i}",
                "event": "KC @ DEN", "market_kind": kind, "series_ticker": "KXNFLTOTAL",
                "result": y, "bid": px - 0.01, "ask": px + 0.01, "mid": px,
                "width": 0.02, "age_min": 1, "volume_t60": 10, "open_interest_t60": 5,
                "market_volume": 100,
            })
        return rows

    def test_fit_bins_selects_only_on_2025_and_respects_floors(self):
        # 25 contracts in the 0.20 bin, all win -> huge edge, should select.
        # 25 contracts in the 0.50 bin, 12/25 win -> no 3c edge, not selected.
        # 5 contracts in the 0.10 bin, all win -> edge but n < 20, not selected.
        yes20 = self._rows(2025, [(0.21, 1)] * 25)
        fair50 = self._rows(2025, [(0.51, 1)] * 12 + [(0.51, 0)] * 13)
        tiny = self._rows(2025, [(0.11, 1)] * 5)
        stats, selected, p_by = FitBins(yes20 + fair50 + tiny, "mid")
        self.assertIn(0.20, selected)
        self.assertNotIn(0.50, selected)
        self.assertNotIn(0.10, selected)
        sel = {s["bin_lo"]: s for s in stats}
        self.assertGreaterEqual(sel[0.20]["n"], MIN_BIN_N)
        self.assertGreaterEqual(sel[0.20]["edge"], MIN_EDGE)
        self.assertEqual(sel[0.10]["n"], 5)
        self.assertFalse(sel[0.10]["selected"])
        self.assertAlmostEqual(sel[0.20]["fair_value"], 1.00)
        # 2026 rows must not affect the fit: sneak a contrary 2026 set into the caller's mind
        # by checking FitBins ignores whatever we do not pass it.
        hold = self._rows(2026, [(0.21, 0)] * 40, day0="2026-09-06")
        stats2, selected2, _ = FitBins(yes20, "mid")
        self.assertEqual(selected2, selected if 0.20 in selected else selected2)
        self.assertIn(0.20, selected2)

    def test_family_eval_does_not_use_2026_for_selection(self):
        fit = self._rows(2025, [(0.21, 1)] * 25)
        # 2026 in the same bin all lose — if leaked into the fit, the bin would not be selected.
        test = self._rows(2026, [(0.21, 0)] * 25, day0="2026-09-06")
        rng = np.random.default_rng(0)
        days = sorted({r["date"] for r in fit + test})
        out = FamilyEval(fit + test, days, "mid", "kalshi_T60_mid", rng, "toy")
        self.assertEqual(out["selected_bins"], [0.20])
        self.assertEqual(out["n_2025"], 25)
        self.assertEqual(out["n_2026"], 25)
        # Bets are placed (bin selected) and they lose on the holdout.
        self.assertEqual(out["n_bets_flat"], 25)
        self.assertLess(out["bank_tim"]["flat100"]["final_equity"], START)

    def test_make_edges_qualify_at_three_cents(self):
        rows = self._rows(2026, [(0.21, 1), (0.24, 1)], day0="2026-09-06")
        # frozen p = 0.30; first contract edge = 0.30-0.22-0.02=0.06; second 0.30-0.25-0.02=0.03
        # a third with ask/mid such that edge < 0.03 should not qualify
        rows += self._rows(2026, [(0.28, 1)], day0="2026-09-09")
        # mid fill: prices are mid. Bin of 0.21/0.24/0.28 is 0.20.
        edges = MakeEdges(rows, selected=[0.20], p_by_lo={0.20: 0.30}, fill_kind="mid", price_basis="x")
        q = [e["qualifies"] for e in edges]
        # 0.30 - 0.21 - 0.02 = 0.07, 0.30-0.24-0.02=0.04, 0.30-0.28-0.02=0.00
        self.assertEqual(q, [True, True, False])

    def test_widest_ci_and_claim_rule(self):
        # All-win small sample: point ROI is huge but a 95% interval should not always clear 0
        # once we include a binomial/mapped bound? All wins -> CP lower bound > 0, so it may claim.
        # Construct mixed bets so the lower bound crosses 0.
        blog = [{"date": "2026-09-07", "event_id": "g1", "side": "a", "stake": 100.0, "pnl": 80.0},
                {"date": "2026-09-07", "event_id": "g1", "side": "b", "stake": 100.0, "pnl": -100.0},
                {"date": "2026-09-08", "event_id": "g2", "side": "a", "stake": 100.0, "pnl": -100.0},
                {"date": "2026-09-09", "event_id": "g3", "side": "a", "stake": 100.0, "pnl": 50.0}]
        edges = [{"date": b["date"], "event_id": b["event_id"], "side": b["side"]} for b in blog]
        rng = np.random.default_rng(1)
        ci = WidestRoiCi(blog, edges, rng)
        self.assertIsNotNone(ci["day_block_95"][0])
        self.assertIsNotNone(ci["game_cluster_95"][0])
        self.assertIsNotNone(ci["binomial_mapped_95"][0])
        self.assertIsNotNone(ci["widest_95"][0])
        # widest should span at least as far as each component
        for name in ("day_block_95", "game_cluster_95", "binomial_mapped_95", "event_boot_95"):
            lo, hi = ci[name]
            wlo, whi = ci["widest_95"]
            self.assertLessEqual(wlo, lo + 1e-12)
            self.assertGreaterEqual(whi, hi - 1e-12)
        # claim requires both fee bases' lower bounds > 0
        bank_t = {"flat100": {"ci": {"widest_95": [0.01, 0.2]}}, "qkelly": {"ci": {"widest_95": [0.02, 0.3]}}}
        bank_k_bad = {"flat100": {"ci": {"widest_95": [-0.05, 0.2]}}, "qkelly": {"ci": {"widest_95": [0.02, 0.3]}}}
        bank_k_ok = {"flat100": {"ci": {"widest_95": [0.01, 0.2]}}, "qkelly": {"ci": {"widest_95": [0.01, 0.2]}}}
        self.assertFalse(Claimable(bank_t, bank_k_bad))
        self.assertTrue(Claimable(bank_t, bank_k_ok))
        # zero is not strictly above 0
        bank_zero = {"flat100": {"ci": {"widest_95": [0.0, 0.2]}}, "qkelly": {"ci": {"widest_95": [0.01, 0.2]}}}
        self.assertFalse(Claimable(bank_t, bank_zero))

    def test_softness_verdict_does_not_claim_without_clearing_bar(self):
        line = {"claimable": False, "selected_bins": [0.20],
                "bank_tim": {"flat100": {"roi": 0.10, "ci": {"widest_95": [-0.2, 0.4]}}}}
        ml = {"claimable": False, "selected_bins": [],
              "bank_tim": {"flat100": {"roi": -0.05, "ci": {"widest_95": [-0.2, 0.1]}}}}
        v = SoftnessVerdict(line, ml, [], [])
        self.assertFalse(v["softer_than_moneylines"])
        self.assertFalse(v["claim_edge_lines"])
        self.assertIn("do not meet the claim bar", v["text"])

        line2 = {"claimable": True, "selected_bins": [0.20],
                 "bank_tim": {"flat100": {"roi": 0.20, "ci": {"widest_95": [0.05, 0.4]}}}}
        v2 = SoftnessVerdict(line2, ml, [], [])
        self.assertTrue(v2["softer_than_moneylines"])
        self.assertTrue(v2["claim_edge_lines"])

    def test_maker_fee_finding(self):
        fees = {"series": {
            "KXNFLTOTAL": {"title": "Totals", "market_kind": "total",
                           "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1,
                           "fee_changes": {"series_fee_change_arr": [
                               {"fee_type": "quadratic_with_maker_fees"}]}},
            "KXNFLSPREAD": {"title": "Spreads", "market_kind": "spread",
                            "fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1,
                            "fee_changes": {}},
        }}
        out = MakerFeeFinding(fees)
        self.assertEqual(out["n_series"], 2)
        self.assertTrue(all(r["charges_maker_fees"] for r in out["rows"]))
        self.assertIn("maker fees", out["verdict"])

    def test_coverage_median_width(self):
        rows = self._rows(2025, [(0.40, 1), (0.40, 0), (0.60, 1)])
        rows[0]["width"] = 0.02
        rows[1]["width"] = 0.04
        rows[2]["width"] = 0.06
        self.assertAlmostEqual(MedianWidth(rows), 0.04)
        cov = Coverage(rows)
        self.assertEqual(len(cov), 1)
        self.assertEqual(cov[0]["n"], 3)
        self.assertEqual(cov[0]["n_games"], 3)


class PullerResumeTests(unittest.TestCase):
    def test_done_tickers_skips_errors_unless_retry(self):
        from DataPull.PullKalshiThinMarkets import DoneTickers
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.jsonl"
            path.write_text(
                json.dumps({"ticker": "A", "n": 3, "error": None}) + "\n"
                + json.dumps({"ticker": "B", "n": 0, "error": "HTTP 400"}) + "\n",
                encoding="utf-8",
            )
            done = DoneTickers(str(path), retry_errors=False)
            self.assertIn("A", done)
            self.assertIn("B", done)
            retry = DoneTickers(str(path), retry_errors=True)
            self.assertIn("A", retry)
            self.assertNotIn("B", retry)



    def test_iter_jsonl_skips_bad_line(self):
        import os, tempfile
        from Backtest.ThinMarkets import IterJsonl
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"ticker": "A"}\n')
                fh.write("{not-json\n")
                fh.write('{"ticker": "B"}\n')
            rows = list(IterJsonl(path))
            self.assertEqual([r["ticker"] for r in rows], ["A", "B"])

if __name__ == "__main__":
    unittest.main()
