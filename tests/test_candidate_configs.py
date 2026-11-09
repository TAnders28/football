"""Candidate configs exist but are NOT wired into PredictDay, and FrozenConfig is untouched by them."""

import os
import unittest

from Backtest.CandidateConfigs import CANDIDATES
from Backtest.FrozenConfig import FROZEN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CandidateConfigTests(unittest.TestCase):
    def test_not_imported_by_live_code(self):
        for name in ("PredictDay.py", "GradePredictions.py", "AssessLive.py"):
            with open(os.path.join(ROOT, "Backtest", name), encoding="utf-8") as fh:
                self.assertNotIn("CandidateConfigs", fh.read(), name)

    def test_same_rules_only_sizing_changes_and_frozen_untouched(self):
        self.assertEqual(FROZEN["name"], "sparse_value8_plus_div2")
        self.assertEqual(FROZEN["value"]["stake"], {"kind": "kelly", "fraction": 0.5, "cap": 0.10})
        for c in CANDIDATES.values():
            self.assertIsNone(c["frozen_at"])
            for part in ("value", "div_addon"):
                self.assertEqual(c[part]["min_edge"], FROZEN[part]["min_edge"])
                self.assertEqual(c[part]["model"], FROZEN[part]["model"])


if __name__ == "__main__":
    unittest.main()
