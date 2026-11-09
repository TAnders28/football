"""
Frozen strategy config for live/forward use. PredictDay reads ONLY this. Do not edit the rules in
place: if a rule changes, add a new config with a new name and frozen_at date, and start a new
forward holdout from that date.

Why frozen
  * TUNE seasons 2015-2022 picked the rules; 2023-2025 were meant to be the out-of-sample TEST.
  * 2023-2025 are now CONTAMINATED: the Lab, Volume and Hunt searches were each re-ranked against
    those seasons many times, so their numbers are optimistic (post-selection bias; see Bailey et al.,
    "The Probability of Backtest Overfitting").
  * Named settings evaluated by the committed search code (counted from the grids on 2026-11-08):
        Backtest.Lab.Grid()      518
        Backtest.Hunt.Grid()    1329
        Backtest.Volume          226  (108 configs x {walk-forward cut, daily quantile} + 6 random + 4 baselines;
                                       daily/random rows are 20-seed averages)
        total                  2,073  (plus earlier ad hoc RunBacktest edge thresholds; lower bound)
  * The only untouched data left is the future: 2026 games with gameday AFTER frozen_at are the
    FORWARD HOLDOUT. Grade them with Backtest.GradePredictions from the append-only log.
"""

import hashlib
import json

FROZEN = {
    "name": "sparse_value8_plus_div2",
    "frozen_at": "2026-11-08",
    # 1) core: walk-forward full2 logit (WIN_FEATS2), bet the side whose edge vs de-vigged ML >= 8 pts
    "value": {"rule": "value_full2_edge8", "model": "p_full2", "min_edge": 0.08,
              "stake": {"kind": "kelly", "fraction": 0.5, "cap": 0.10}},
    # 2) divisional add-on: full3 logit, divisional games only, edge >= 2 pts, flat 2% of bank
    "div_addon": {"rule": "div_full3_edge2", "model": "p_full3", "min_edge": 0.02, "filter": "div",
                  "stake": {"kind": "pct_bank", "pct": 0.02}},
    # shown as a comparison column only (never bet)
    "v1_column": {"rule": "model_edge_5pct_v1", "model": "p_margin", "min_edge": 0.05},
    "train_seasons_from": 2013,
    "tune_seasons": [2015, 2022],
    "contaminated_test_seasons": [2023, 2024, 2025],
    "forward_holdout": "2026 regular/post-season games with gameday > 2026-11-08",
    "settings_tried": {"lab": 518, "hunt": 1329, "volume": 226, "total": 2073, "note": "lower bound"},
}


def ConfigHash(config=None) -> str:
    """Short, stable hash of the frozen rules (key order independent)."""
    blob = json.dumps(config or FROZEN, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def IsForwardHoldout(gameday: str, config=None) -> bool:
    return str(gameday) > (config or FROZEN)["frozen_at"]
