import math
import unittest

from Tools import Common as C
from Tools import Live
from Tools import Slip as S


class DecisionToolTests(unittest.TestCase):
    def test_parse_legs(self):
        self.assertEqual(S.ParseLeg("DET ml")["kind"], "ml")
        l = S.ParseLeg("BUF -6.5 @0.48")
        self.assertEqual((l["kind"], l["team"], l["line"], l["price"]), ("spread", "BUF", -6.5, 0.48))
        l = S.ParseLeg("KC@ATL over 47.5")
        self.assertEqual((l["kind"], l["team"], l["side"], l["line"]), ("total", "ATL", "over", 47.5))
        self.assertEqual(S.ParseLeg("Texas A&M u 52")["side"], "under")
        self.assertEqual(S.ParseLeg("ticker:KXNFLTD-26NOV15KCATL-X")["kind"], "ticker")
        self.assertEqual(S.ParseLeg("PIT pk")["line"], 0.0)
        with self.assertRaises(ValueError):
            S.ParseLeg("nonsense")

    def test_american_price(self):
        self.assertAlmostEqual(S.AmericanToPrice("+500"), 1 / 6)
        self.assertAlmostEqual(S.AmericanToPrice(-200), 2 / 3)

    def test_slip_math(self):
        m = C.SlipMath(0.60, 0.55, 10)
        self.assertEqual(m["cost"], 5.5)
        self.assertEqual(m["payout_if_win"], 10.0)
        self.assertAlmostEqual(m["ev_at_fair_before_fees"], 0.5)
        self.assertAlmostEqual(m["ev_after_tim_fee"], 0.3)
        self.assertAlmostEqual(m["fee_kalshi_taker"], math.ceil(0.07 * 0.55 * 0.45 * 10 * 100) / 100)

    def test_shift_line(self):
        p = C.ShiftLineProb(0.5, -3.0, -3.0, 13.5)
        self.assertAlmostEqual(p, 0.5)
        self.assertLess(C.ShiftLineProb(0.5, -3.0, -6.5, 13.5), 0.5)  # laying more points is worse
        self.assertGreater(C.ShiftLineProb(0.5, 47.5, 44.5, 10, higher_is_better=False), 0.5)  # lower over line is easier
        self.assertAlmostEqual(C.ComboFair([0.5, 0.5]), 0.25)
        self.assertIsNone(C.ComboFair([0.5, None]))

    def test_devig(self):
        fa, fb, over, hold = C.DevigMultiplicative(C.AmericanToProb(-110), C.AmericanToProb(-110))
        self.assertAlmostEqual(fa, 0.5)
        self.assertAlmostEqual(hold, 1 - 1 / (2 * 110 / 210))

    def test_live_fmt_empty(self):
        txt = Live.Fmt({"league": "nfl", "games": []})
        self.assertIn("no games in progress", txt)


if __name__ == "__main__":
    unittest.main()
