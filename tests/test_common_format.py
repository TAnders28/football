import unittest
from datetime import datetime, timezone

import numpy as np

from Backtest.CommonFormat import (
    ALPHA, BANK_FIELDS, CAL_FIELDS, EDGE_FIELDS, FEE, FLAT, MIN_EDGE, MODEL, START, SUM_FIELDS,
    ClopperPearson, Devig, FlatPnl, Mid, QKellyStake, QuoteWithTs, Walk, Widest,
)
from Backtest.KalshiReality import QuoteAt


def C(ts, bid, ask):
    return {"end_ts": ts, "yes_bid_close": bid, "yes_ask_close": ask}


def Edge(date, gid, qualifies, fill, result, model_p=0.70, start="2026-11-08T13:00:00-05:00",
         season="2026", split="forward", fav_fill=None, fav_result=None):
    fill = float(fill)
    fav_fill = fill if fav_fill is None else float(fav_fill)
    fav_result = result if fav_result is None else fav_result
    mkt = fill  # tests often use a tight market
    return {
        "date": date, "start_time": start, "event_id": gid, "season": season, "split": split,
        "price_basis": "kalshi_T60", "qualifies": qualifies, "result": result,
        "_p": model_p, "_fill": fill, "_fav_fill": fav_fill, "_fav_result": fav_result,
    }


class CommonFormatTests(unittest.TestCase):
    def test_mid_and_devig(self):
        self.assertAlmostEqual(Mid(0.40, 0.44), 0.42)
        self.assertAlmostEqual(Devig(0.44, 0.44), 0.5)
        self.assertAlmostEqual(Devig(0.60, 0.30), 0.60 / 0.90)

    def test_quote_t60_does_not_use_later_candle(self):
        ko = datetime(2026, 11, 15, 18, 0, tzinfo=timezone.utc)
        k = int(ko.timestamp())
        entry = {"ko": ko, "candles": [C(k - 100 * 60, 0.40, 0.42), C(k - 61 * 60, 0.48, 0.52),
                                       C(k - 59 * 60, 0.80, 0.82), C(k - 15 * 60, 0.30, 0.32)]}
        q = QuoteWithTs(entry, 60)
        self.assertEqual((q["bid"], q["ask"]), (0.48, 0.52))
        self.assertAlmostEqual(q["mid"], 0.50)
        self.assertEqual(q["age_min"], 1.0)
        # the T-59 candle must not be used
        self.assertEqual(QuoteAt(entry, 60)["ask"], 0.52)

    def test_flat_pnl_and_qkelly(self):
        # cost = 0.50 + 0.02 = 0.52; win: 100/0.52 - 100
        self.assertAlmostEqual(FlatPnl(0.50, 1), 100 / 0.52 - 100, places=6)
        self.assertAlmostEqual(FlatPnl(0.50, 0), -100)
        self.assertIsNone(FlatPnl(0.99, 1))  # 0.99+0.02 >= 1
        # quarter Kelly at p=0.70, c=0.52: 0.25 * (0.18/0.48) * 10000 = 937.5, cap 2% = 200
        self.assertAlmostEqual(QKellyStake(0.70, 0.50, 10_000), 200.0)
        # no bet when p < cost
        self.assertEqual(QKellyStake(0.40, 0.50, 10_000), 0.0)

    def test_clopper_pearson_and_widest(self):
        lo, hi = ClopperPearson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual(ClopperPearson(0, 10)[0], 0.0)
        self.assertAlmostEqual(ClopperPearson(10, 10)[1], 1.0)
        self.assertEqual(ClopperPearson(0, 0), (None, None))
        w = Widest((0.2, 0.6), (0.1, 0.55))
        self.assertEqual(w, (0.1, 0.6))

    def test_walk_flat100_formula_index_and_zero_bet_days(self):
        # Day 1: one qualifying win at fill 0.50; Day 2: a game that does not qualify; Day 3: a loss
        e1 = Edge("2025-09-07", "g1", True, 0.50, 1, season="2025", split="contaminated")
        e2 = Edge("2025-09-14", "g2", False, 0.50, 1, model_p=0.51, season="2025", split="contaminated")
        e3 = Edge("2026-09-10", "g3", True, 0.50, 0, season="2026", split="forward")
        days = ["2025-09-07", "2025-09-14", "2026-09-10"]
        daily, log, ruined, end = Walk([e1, e2, e3], "flat100", fee_kind="tim", days=days)
        self.assertEqual([d["x"] for d in daily], [1, 2, 3])
        self.assertEqual([d["bets_day"] for d in daily], [1, 0, 1])
        self.assertEqual(daily[1]["bets_cum"], 1)
        self.assertEqual(daily[2]["bets_cum"], 2)
        self.assertEqual(daily[2]["season"], "2026")
        win = FLAT / (0.50 + FEE) - FLAT
        self.assertAlmostEqual(daily[0]["equity"], round(START + win, 2))
        self.assertAlmostEqual(daily[1]["equity"], daily[0]["equity"])
        self.assertAlmostEqual(daily[2]["equity"], round(START + win - FLAT, 2))
        self.assertFalse(ruined)
        self.assertEqual(daily[0]["drawdown_pct"], 0.0)
        self.assertLess(daily[2]["drawdown_pct"], 0)

    def test_walk_stops_at_ruin(self):
        # lose enough flat $100 bets to go under $100, then a later qualifying game must not be bet
        edges = [Edge("2025-09-0%d" % (i + 1,) if i < 9 else "2025-09-%d" % (i + 1,),
                      "g%d" % i, True, 0.80, 0, model_p=0.95, season="2025", split="contaminated")
                 for i in range(100)]
        # 100 losses of $100 would go to 0; ruin triggers once bank < 100 after a day
        days = [e["date"] for e in edges] + ["2025-12-28"]
        edges.append(Edge("2025-12-28", "late", True, 0.40, 1, model_p=0.80, season="2025", split="contaminated"))
        daily, log, ruined, end = Walk(edges, "flat100", fee_kind="tim", days=days)
        self.assertTrue(ruined)
        self.assertLess(end, 100)
        # the last day must not add a bet
        self.assertEqual(daily[-1]["bets_day"], 0)
        self.assertEqual(daily[-1]["bets_cum"], daily[-2]["bets_cum"])

    def test_qkelly_uses_start_of_day_and_cap(self):
        # two qualifying games the same day: stakes must be computed from the start-of-day bank (and cap 2%)
        e1 = Edge("2026-09-10", "a", True, 0.50, 1, model_p=0.70)
        e2 = Edge("2026-09-10", "b", True, 0.50, 0, model_p=0.70)
        daily, log, ruined, end = Walk([e1, e2], "qkelly", fee_kind="tim", days=["2026-09-10"])
        self.assertEqual(daily[0]["bets_day"], 2)
        self.assertTrue(all(abs(b["stake"] - 200.0) < 0.01 for b in log))  # 2% cap
        # win 200/0.52 - 200, lose -200
        win = 200.0 / 0.52 - 200.0
        self.assertAlmostEqual(daily[0]["equity"], round(START + win - 200.0, 2))

    def test_schema_column_order(self):
        self.assertEqual(CAL_FIELDS[:11], ["date", "sport", "series", "model", "season", "split",
                                           "bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"])
        self.assertEqual(SUM_FIELDS, ["date", "sport", "series", "model", "season", "split",
                                      "n", "accuracy", "brier", "log_loss", "ece"])
        self.assertEqual(BANK_FIELDS[0], "date")
        self.assertEqual(BANK_FIELDS[-1], "drawdown_pct")
        self.assertIn("x", BANK_FIELDS)
        self.assertEqual(EDGE_FIELDS[0], "date")
        self.assertEqual(EDGE_FIELDS[-1], "pnl_at_flat_stake")
        self.assertEqual(MODEL, "p_full2_kmid_T60")
        self.assertEqual(MIN_EDGE, 0.03)
        self.assertEqual(FEE, 0.02)
        self.assertEqual(ALPHA, 0.10)


if __name__ == "__main__":
    unittest.main()
