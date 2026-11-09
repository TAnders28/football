"""
CANDIDATE configs only. Nothing imports this: PredictDay reads Backtest.FrozenConfig.FROZEN and nothing else.
A candidate goes live only if Tim says so; then it becomes a NEW frozen config with its own frozen_at date and
a fresh forward holdout from that date (2026 games after it).

Both candidates keep the frozen SELECTION rules (value full2 >= 8 pts, divisional full3 >= 2 pts) and change
only SIZING, which is what compounds a single bankroll. Evidence (Backtest.Compound, run 20261109T115526-a805b3,
one continuous $10k bankroll 2023 -> 2026 wk 9, closing moneylines; 2023-25 contaminated by >= 2,311 settings):
    frozen as-is (half Kelly cap 10% + flat 2% div)   $28,655   max DD -22%
    A  full Kelly, no cap (what the nested walk-forward picks for next season)  $63,011   max DD -55%
    B  half Kelly, cap 20% on both rules              $29,269   max DD -31%
Full Kelly assumes the model probabilities are right; they are barely better calibrated than the closing
market (Brier skill +0.4%, N = 988), so A's real risk is higher than its backtest shows.
"""

import copy

from Backtest.FrozenConfig import FROZEN


def _Sized(name, fraction, cap, div_kind):
    cfg = copy.deepcopy(FROZEN)
    cfg["name"] = name
    cfg["frozen_at"] = None            # set only if Tim adopts it
    cfg["status"] = "candidate (not wired into PredictDay)"
    cfg["value"]["stake"] = {"kind": "kelly", "fraction": fraction, "cap": cap}
    cfg["div_addon"]["stake"] = ({"kind": "kelly", "fraction": fraction, "cap": cap} if div_kind == "kelly"
                                 else dict(cfg["div_addon"]["stake"]))
    cfg["evidence"] = "Backtest.Compound run 20261109T115526-a805b3; see E:/AgentDocs/American-Footbot/2026-11-09_compounding-bankroll"
    return cfg


CANDIDATES = {
    "A_sparse_value8_plus_div2_kelly1_nocap": _Sized("A_sparse_value8_plus_div2_kelly1_nocap", 1.0, 1.0, "kelly"),
    "B_sparse_value8_plus_div2_kelly0.5_cap20": _Sized("B_sparse_value8_plus_div2_kelly0.5_cap20", 0.5, 0.20, "kelly"),
}
