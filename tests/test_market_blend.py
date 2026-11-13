import math
import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np

from Backtest.CommonFormat import FEE, MIN_EDGE, START, Walk
from Backtest.Lab import Sig
from Backtest.MarketBlend import (
    AccruedGameStats, ApplyBlend, AttachRatings, BlendEdge, BlendEdges, BootstrapBlendWeights,
    BootstrapIgnRel, ClassifySeries, ClipP, DiscoverSeries, EdgeClaimable, FitBlend, FitRatingsModel,
    FrozenSeasonBlend, GrowthPerEvent, IgnRelBits, KalshiApi, LogLossBits, LogLossNats, LN2,
    MIN_RATING_PLAYS, PlaySuccess, PlayYards, PriorStdRatings, RATINGS_TRIALS, RatingsGain,
    MarketCloseTs, RunSurvey, SampleCandles, SeasonWeekFromTicker, SplitRows, SurveyCoverage, WidestRoi,
)


def Row(gid, season, week, gameday, home, away, home_win, p_kmid, kmid, home_bid=None, home_ask=None,
        away_bid=None, away_ask=None, seen=True):
    if home_bid is None:
        home_bid = max(0.01, kmid - 0.02)
    if home_ask is None:
        home_ask = min(0.99, kmid + 0.02)
    if away_bid is None:
        away_bid = max(0.01, (1 - kmid) - 0.02)
    if away_ask is None:
        away_ask = min(0.99, (1 - kmid) + 0.02)
    mid_h = (home_bid + home_ask) / 2
    mid_a = (away_bid + away_ask) / 2
    ts = "2025-09-07T17:00:00+00:00"
    r = {
        "game_id": gid, "season": season, "week": week, "gameday": gameday,
        "home_team": home, "away_team": away, "home_win": home_win,
        "p_kmid": p_kmid, "kmid": kmid, "result": None if home_win is None else (1 if home_win else -1),
        "kq_t60": {
            "home": {"bid": home_bid, "ask": home_ask, "mid": mid_h, "price_time": ts, "age_min": 1},
            "away": {"bid": away_bid, "ask": away_ask, "mid": mid_a, "price_time": ts, "age_min": 1},
        },
        "gametime": "13:00",
    }
    r["_seen"] = seen
    return r


class FakeApi:
    def __init__(self):
        self.calls = 0
        self.retries = 0

    def Get(self, path, params=None):
        self.calls += 1
        if path == "/series":
            return {"series": [
                {"ticker": "KXNFLTOTAL", "title": "NFL Game Total Points", "category": "Sports"},
                {"ticker": "KXNFLSPREAD", "title": "NFL Point Spread", "category": "Sports"},
                {"ticker": "KXNBAGame", "title": "NBA Game", "category": "Sports"},
            ]}
        if path.startswith("/series/") and "/markets/" not in path:
            t = path.rsplit("/", 1)[-1]
            if t in ("KXNFLTOTAL", "KXNFLSPREAD", "KXNFLGAME"):
                return {"series": {"ticker": t, "title": t, "category": "Sports"}}
            return None
        if "candlesticks" in path:
            return {"candlesticks": [{"end_period_ts": 1_725_000_000}, {"end_period_ts": 1_725_003_600}]}
        return {}

    def Paged(self, path, params, key="markets", max_pages=20):
        self.calls += 1
        st = (params or {}).get("series_ticker")
        if st == "KXNFLTOTAL":
            return [{"ticker": "KXNFLTOTAL-25SEP07KCDEN-T45", "event_ticker": "KXNFLTOTAL-25SEP07KCDEN",
                     "status": "settled", "volume": "12", "close_time": "2025-09-08T00:00:00Z"}]
        if st == "KXNFLSPREAD":
            return [{"ticker": "KXNFLSPREAD-26NOV09BUFMIN-BUF3", "event_ticker": "KXNFLSPREAD-26NOV09BUFMIN",
                     "status": "open", "volume": "3"}]
        if st == "KXNFLGAME":
            return [{"ticker": "KXNFLGAME-25SEP07KCDEN-KC", "event_ticker": "KXNFLGAME-25SEP07KCDEN",
                     "status": "settled"}]
        return []


class MarketBlendScoreTests(unittest.TestCase):
    def test_bits_are_nats_over_ln2(self):
        self.assertAlmostEqual(LogLossBits(0.8, 1), LogLossNats(0.8, 1) / LN2)
        self.assertAlmostEqual(LogLossBits(0.2, 0), -math.log2(0.8))

    def test_ign_rel_negative_when_model_better(self):
        rows = [
            {"home_win": 1, "p_kmid": 0.80, "kmid": 0.55},
            {"home_win": 0, "p_kmid": 0.20, "kmid": 0.45},
        ]
        gap, n = IgnRelBits(rows, "p_kmid", "kmid")
        self.assertEqual(n, 2)
        self.assertLess(gap, 0)
        self.assertAlmostEqual(GrowthPerEvent(gap), 2 ** (-gap))

    def test_ign_rel_zero_when_identical(self):
        rows = [{"home_win": 1, "p_kmid": 0.6, "kmid": 0.6},
                {"home_win": 0, "p_kmid": 0.3, "kmid": 0.3}]
        gap, n = IgnRelBits(rows, "p_kmid", "kmid")
        self.assertEqual(n, 2)
        self.assertAlmostEqual(gap, 0.0, places=12)

    def test_game_bootstrap_ci_contains_point_and_reports_n(self):
        rng = np.random.default_rng(0)
        rows = [{"home_win": int(i % 3 != 0), "p_kmid": 0.55 + 0.01 * (i % 5), "kmid": 0.50 + 0.01 * (i % 7)}
                for i in range(40)]
        point, n = IgnRelBits(rows, "p_kmid", "kmid")
        lo, hi, n_ci = BootstrapIgnRel(rows, "p_kmid", "kmid", rng, boot=200)
        self.assertEqual(n, 40)
        self.assertEqual(n_ci, 40)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


class MarketBlendFitTests(unittest.TestCase):
    def test_fit_blend_recovers_market_heavier_weight(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(300):
            k = float(np.clip(rng.uniform(0.2, 0.8), 0.05, 0.95))
            # model is a noisy copy; market is the true probability
            p = float(np.clip(k + rng.normal(0, 0.12), 0.05, 0.95))
            y = int(rng.random() < k)
            rows.append({"home_win": y, "p_kmid": p, "kmid": k})
        fit = FitBlend(rows, l2=1.0)
        self.assertIsNotNone(fit)
        self.assertGreater(fit["b_kalshi"], fit["b_model"])
        self.assertGreater(fit["b_kalshi"], 0)

    def test_frozen_blend_does_not_use_apply_outcomes(self):
        train = [{"home_win": 1, "p_kmid": 0.7, "kmid": 0.6},
                 {"home_win": 0, "p_kmid": 0.3, "kmid": 0.4},
                 {"home_win": 1, "p_kmid": 0.65, "kmid": 0.55},
                 {"home_win": 0, "p_kmid": 0.35, "kmid": 0.45},
                 {"home_win": 1, "p_kmid": 0.8, "kmid": 0.7},
                 {"home_win": 0, "p_kmid": 0.25, "kmid": 0.35},
                 {"home_win": 1, "p_kmid": 0.6, "kmid": 0.58},
                 {"home_win": 0, "p_kmid": 0.4, "kmid": 0.42},
                 {"home_win": 1, "p_kmid": 0.75, "kmid": 0.62},
                 {"home_win": 0, "p_kmid": 0.28, "kmid": 0.38}]
        apply_a = [{"home_win": 1, "p_kmid": 0.66, "kmid": 0.60},
                   {"home_win": 0, "p_kmid": 0.33, "kmid": 0.40}]
        apply_b = [{"home_win": 0, "p_kmid": 0.66, "kmid": 0.60},
                   {"home_win": 1, "p_kmid": 0.33, "kmid": 0.40}]
        w_a = FrozenSeasonBlend(train, apply_a)
        w_b = FrozenSeasonBlend(train, apply_b)
        self.assertEqual(w_a["a"], w_b["a"])
        self.assertEqual(w_a["b_model"], w_b["b_model"])
        self.assertEqual(w_a["b_kalshi"], w_b["b_kalshi"])
        # apply rows got a probability from the frozen weights
        self.assertIsNotNone(apply_a[0]["p_blend"])
        self.assertAlmostEqual(apply_a[0]["p_blend"], apply_b[0]["p_blend"])

    def test_bootstrap_weight_ci_shape(self):
        rng = np.random.default_rng(2)
        rows = [{"home_win": int(i % 2 == 0), "p_kmid": 0.4 + 0.02 * (i % 6), "kmid": 0.45 + 0.01 * (i % 5)}
                for i in range(40)]
        ci = BootstrapBlendWeights(rows, rng, boot=50)
        self.assertEqual(ci["n"], 40)
        self.assertGreater(ci["n_ok"], 0)
        for key in ("a", "b_model", "b_kalshi"):
            lo, hi = ci[key]
            self.assertLessEqual(lo, hi)


class RatingsLeakTests(unittest.TestCase):
    def test_play_success_standard_thresholds(self):
        self.assertEqual(PlaySuccess({"success": 1, "yards_gained": 1, "down": 1, "ydstogo": 10}), 1)
        self.assertEqual(PlaySuccess({"yards_gained": 5, "down": 1, "ydstogo": 10}), 1)
        self.assertEqual(PlaySuccess({"yards_gained": 4, "down": 1, "ydstogo": 10}), 0)
        self.assertEqual(PlaySuccess({"yards_gained": 7, "down": 2, "ydstogo": 10}), 1)
        self.assertEqual(PlaySuccess({"yards_gained": 6, "down": 2, "ydstogo": 10}), 0)
        self.assertEqual(PlaySuccess({"yards_gained": 8, "down": 3, "ydstogo": 8}), 1)
        self.assertEqual(PlaySuccess({"yards_gained": 7, "down": 3, "ydstogo": 8}), 0)
        self.assertIsNone(PlayYards({"yards_gained": None}))

    def test_accrued_skips_plays_without_yards(self):
        plays = [
            {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-07",
             "posteam": "KC", "defteam": "DEN", "home_team": "DEN", "away_team": "KC",
             "pass": 1, "yards_gained": 10, "success": 1},
            {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-07",
             "posteam": "KC", "defteam": "DEN", "home_team": "DEN", "away_team": "KC",
             "pass": 1, "yards_gained": None, "success": 1},
            {"game_id": "g1", "season": 2025, "week": 1, "game_date": "2025-09-07",
             "posteam": "DEN", "defteam": "KC", "home_team": "DEN", "away_team": "KC",
             "rush": 1, "yards_gained": 4, "success": 0},
        ]
        rows, meta = AccruedGameStats(plays)
        self.assertEqual(meta["kept_plays"], 2)
        self.assertEqual(meta["skipped_no_yards"], 1)
        by = {(r["game_id"], r["team"]): r for r in rows}
        self.assertEqual(by[("g1", "KC")]["off_plays"], 1)
        self.assertEqual(by[("g1", "KC")]["off_yards"], 10)
        self.assertEqual(by[("g1", "DEN")]["def_yards"], 10)

    def test_prior_ratings_use_earlier_days_only(self):
        # KC: 100-play game on 09-07, then another on 09-14. DEN is the opponent both times
        # (synthetic; we only need team-game rows).
        games = []
        for gid, day, team, opp, yards, plays in (
            ("g1", "2025-09-07", "KC", "DEN", 500.0, 100),
            ("g1", "2025-09-07", "DEN", "KC", 200.0, 100),
            ("g2", "2025-09-14", "KC", "DEN", 300.0, 80),
            ("g2", "2025-09-14", "DEN", "KC", 240.0, 80),
            ("g3", "2025-09-14", "BUF", "MIA", 400.0, 90),  # same day as g2; must not leak into KC
            ("g3", "2025-09-14", "MIA", "BUF", 270.0, 90),
        ):
            games.append({"game_id": gid, "season": 2025, "week": 1 if day.endswith("07") else 2,
                          "gameday": day, "team": team, "opponent": opp,
                          "home_team": opp, "away_team": team,
                          "off_plays": plays, "off_yards": yards, "off_succ": 40,
                          "def_plays": plays, "def_yards": 0.0, "def_succ": 20})
        # fix def stats from the opponent's offense
        by = {(g["game_id"], g["team"]): g for g in games}
        for g in games:
            opp = by[(g["game_id"], g["opponent"])]
            g["def_plays"], g["def_yards"], g["def_succ"] = opp["off_plays"], opp["off_yards"], opp["off_succ"]
        ratings = PriorStdRatings(games, min_plays=80)
        first = ratings[("g1", "KC")]
        self.assertFalse(first["ready"])
        self.assertEqual(first["prior_games"], 0)
        self.assertEqual(first["off_plays"], 0)
        second = ratings[("g2", "KC")]
        self.assertTrue(second["ready"])
        self.assertEqual(second["prior_games"], 1)
        self.assertAlmostEqual(second["off_ypp"], 5.0)
        self.assertAlmostEqual(second["def_ypp"], 2.0)
        # g3 on the same day as g2 must not be in KC's prior
        self.assertEqual(second["off_plays"], 100)
        # BUF's first game is not ready
        self.assertFalse(ratings[("g3", "BUF")]["ready"])

    def test_attach_and_gain_on_ready_games_only(self):
        ratings = {
            ("g2", "KC"): {"net_ypp": 3.0, "net_sr": 0.1, "off_plays": 100, "ready": True},
            ("g2", "DEN"): {"net_ypp": -1.0, "net_sr": -0.05, "off_plays": 100, "ready": True},
            ("g1", "KC"): {"net_ypp": None, "net_sr": None, "off_plays": 0, "ready": False},
            ("g1", "DEN"): {"net_ypp": None, "net_sr": None, "off_plays": 0, "ready": False},
        }
        rows = [
            {"game_id": "g1", "home_team": "DEN", "away_team": "KC", "home_win": 0, "p_full2": 0.4},
            {"game_id": "g2", "home_team": "DEN", "away_team": "KC", "home_win": 0, "p_full2": 0.4},
        ]
        n = AttachRatings(rows, ratings)
        self.assertEqual(n, 1)
        self.assertFalse(rows[0]["ratings_ready"])
        self.assertTrue(rows[1]["ratings_ready"])
        self.assertAlmostEqual(rows[1]["ypp_diff"], -1.0 - 3.0)
        self.assertAlmostEqual(rows[1]["sr_diff"], -0.05 - 0.1)
        rows[1]["p_full2_yppsr"] = 0.2
        gain = RatingsGain([rows[1]])
        self.assertEqual(gain["n"], 1)
        # home lost; lower p is better, so d_log_loss should be negative
        self.assertLess(gain["d_log_loss_nats"], 0)

    def test_fit_ratings_walk_forward_is_season_prior_only(self):
        # 2024 ready games used to fit 2025; a 2025 feature change must not leak into 2025 weights
        def base(season, gid, ypp, win, extra=0):
            return {
                "game_id": gid, "season": season, "week": 3, "gameday": f"{season}-09-20",
                "home_team": "H", "away_team": "A", "home_win": win,
                "ratings_ready": True, "ypp_diff": ypp, "sr_diff": 0.0, "p_full2": 0.5,
                "mkt_logit": 0.0, "margin_logit": 0.0, "form3_diff": 0, "ats_diff": 0,
                "ats3_diff": 0, "rest_diff": 0, "div": 0, "spread": 0, "spread_logit": 0,
            }
        rbs = {
            2024: [base(2024, f"a{i}", 2.0 if i % 2 == 0 else -2.0, 1 if i % 2 == 0 else 0) for i in range(50)],
            2025: [base(2025, f"b{i}", 2.0 if i % 2 == 0 else -2.0, 1 if i % 2 == 0 else 0) for i in range(10)],
        }
        FitRatingsModel(rbs, min_season=2024)
        p_before = [r["p_full2_yppsr"] for r in rbs[2025]]
        # flip 2025 labels and refit: 2025 probabilities must stay the same (train is 2024 only)
        for r in rbs[2025]:
            r["home_win"] = 1 - r["home_win"]
        FitRatingsModel(rbs, min_season=2024)
        p_after = [r["p_full2_yppsr"] for r in rbs[2025]]
        for a, b in zip(p_before, p_after):
            self.assertAlmostEqual(a, b)
        self.assertEqual(RATINGS_TRIALS, 1)


class BlendEdgeTests(unittest.TestCase):
    def test_mid_qualifies_on_devig_edge_ask_on_ask_edge(self):
        # model 0.80 vs mid 0.50 -> edge vs mid 0.28, clearly qualifies on mid
        # ask is 0.90, so p - ask - 0.02 = 0.80-0.90-0.02 < 0, does not qualify on ask
        r = Row("g", 2026, 1, "2026-09-10", "KC", "DEN", 1, 0.80, 0.50,
                home_bid=0.49, home_ask=0.90, away_bid=0.30, away_ask=0.50)
        mid = BlendEdge(r, fill_kind="mid", p_key="p_kmid")
        ask = BlendEdge(r, fill_kind="ask", p_key="p_kmid")
        self.assertTrue(mid["qualifies"])
        self.assertAlmostEqual(mid["fill_price"], (0.49 + 0.90) / 2)
        self.assertEqual(mid["fill_kind"], "mid")
        self.assertFalse(ask["qualifies"])
        self.assertAlmostEqual(ask["fill_price"], 0.90)
        self.assertEqual(ask["price_basis"], "kalshi_T60_ask")
        self.assertLess(ask["edge_vs_fill"], MIN_EDGE)

    def test_walk_uses_blend_edge_fields(self):
        r = Row("g", 2026, 1, "2026-09-10", "KC", "DEN", 1, 0.80, 0.50,
                home_bid=0.48, home_ask=0.52, away_bid=0.48, away_ask=0.52)
        r["p_blend"] = 0.80
        edges = BlendEdges([r], fill_kind="mid", p_key="p_blend")
        self.assertEqual(len(edges), 1)
        daily, log, ruined, end = Walk(edges, "flat100", fee_kind="tim", days=["2026-09-10"])
        self.assertFalse(ruined)
        self.assertEqual(daily[0]["bets_day"], 1)
        self.assertGreater(end, START)

    def test_edge_claimable_requires_both_lower_bounds_positive(self):
        self.assertTrue(EdgeClaimable((0.01, 0.2), (0.001, 0.3)))
        self.assertFalse(EdgeClaimable((-0.01, 0.2), (0.01, 0.3)))
        self.assertFalse(EdgeClaimable((0.01, 0.2), (-0.01, 0.3)))
        self.assertFalse(EdgeClaimable((None, None), (0.01, 0.3)))

    def test_widest_roi_uses_day_and_event(self):
        rng = np.random.default_rng(3)
        log = [{"date": "2026-09-10", "pnl": 20.0, "stake": 100.0},
               {"date": "2026-09-10", "pnl": -100.0, "stake": 100.0},
               {"date": "2026-09-17", "pnl": 50.0, "stake": 100.0}]
        widest, ev, day = WidestRoi(log, rng, boot=200)
        self.assertIsNotNone(widest[0])
        self.assertLessEqual(widest[0], ev[0])
        self.assertGreaterEqual(widest[1], ev[1])


class SurveyTests(unittest.TestCase):
    def test_classify_and_ticker_parse(self):
        self.assertEqual(ClassifySeries("KXNFLTOTAL", "NFL totals"), "total")
        self.assertEqual(ClassifySeries("KXNFLSPREAD", "NFL point spread"), "spread")
        self.assertEqual(ClassifySeries("KXNFLGAME", "NFL Game Winner"), "moneyline")
        season, day, _w = SeasonWeekFromTicker("KXNFLTOTAL-25SEP07KCDEN-T45")
        self.assertEqual(season, 2025)
        self.assertEqual(day, "2025-09-07")
        season2, day2, _ = SeasonWeekFromTicker("KXNFLSPREAD-26NOV09BUFMIN-BUF3")
        self.assertEqual(season2, 2026)
        self.assertEqual(day2, "2026-11-09")

    def test_survey_uses_injected_api_and_records_candle_sample(self):
        api = FakeApi()
        out = RunSurvey(api, candidates=("KXNFLGAME", "KXNFLTOTAL", "KXNFLSPREAD", "KXNFLMISSING"),
                        samples_per_series=1, kinds=("total", "spread"))
        self.assertGreater(api.calls, 0)
        self.assertEqual(out["http_calls"], api.calls)
        self.assertEqual(out["series"]["KXNFLTOTAL"]["market_kind"], "total")
        self.assertEqual(out["series"]["KXNFLSPREAD"]["market_kind"], "spread")
        self.assertEqual(out["series"]["KXNFLMISSING"]["source"], "probe_missing")
        self.assertGreater(out["n_markets"]["KXNFLTOTAL"], 0)
        tickers = {r["market_ticker"] for r in out["rows"] if r["market_ticker"]}
        self.assertTrue(any("KXNFLTOTAL" in t for t in tickers))
        sampled = [r for r in out["rows"] if r["notes"] == "candle sample"]
        self.assertTrue(sampled)
        self.assertGreater(sampled[0]["n_candles_sample"], 0)

    def test_kalshi_api_callback_is_used_instead_of_network(self):
        hits = []

        def getter(path, params):
            hits.append(path)
            return {"series": []}

        api = KalshiApi(get=getter)
        body = api.Get("/series", {"limit": 1})
        self.assertEqual(body, {"series": []})
        self.assertEqual(hits, ["/series"])

    def test_survey_coverage_handles_empty_series(self):
        series_map = {"KXNFLPTS": {"ticker": "KXNFLPTS", "title": "", "source": "probe_missing",
                                   "market_kind": "other"}}
        rows = SurveyCoverage(series_map, {"KXNFLPTS": {}}, {})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["notes"], "series missing")
        self.assertEqual(rows[0]["n_markets"], 0)

    def test_sample_candles_bounded_window_survives_http_400(self):
        import io
        import urllib.error
        seen = []

        class BoomApi:
            calls = retries = 0

            def Get(self, path, params=None):
                seen.append(dict(params or {}))
                raise urllib.error.HTTPError(path, 400, 'Bad Request', hdrs=None, fp=io.BytesIO())

        n, lo, hi, ep, err = SampleCandles(
            BoomApi(), 'KXNFLTOTAL', 'KXNFLTOTAL-25SEP07KCDEN-T45', 'historical',
            market={'close_time': '2025-09-08T00:00:00Z'},
        )
        self.assertEqual(n, 0)
        self.assertIsNone(lo)
        self.assertIsNotNone(err)
        self.assertIn('400', err)
        self.assertTrue(seen)
        span = seen[0]['end_ts'] - seen[0]['start_ts']
        self.assertLessEqual(span, 8 * 24 * 3600)
        self.assertGreater(seen[0]['start_ts'], 0)
        self.assertNotEqual(seen[0]['period_interval'], 1)
        close_ts = MarketCloseTs({'close_time': '2025-09-08T00:00:00Z'})
        self.assertEqual(close_ts, 1757289600)

    def test_sample_candles_retries_daily_after_hourly_400(self):
        import io
        import urllib.error
        periods = []

        class OnceApi:
            calls = retries = 0

            def Get(self, path, params=None):
                periods.append((params or {}).get('period_interval'))
                if (params or {}).get('period_interval') != 1440:
                    raise urllib.error.HTTPError(path, 400, 'Bad Request', hdrs=None, fp=io.BytesIO())
                return {'candlesticks': [{'end_period_ts': 1757289600}, {'end_period_ts': 1757293200}]}

        n, lo, hi, ep, err = SampleCandles(
            OnceApi(), 'KXNFLTOTAL', 'KXNFLTOTAL-25SEP07KCDEN-T45', 'live',
            market={'close_time': '2025-09-08T00:00:00Z'},
        )
        self.assertEqual(n, 2)
        self.assertIsNone(err)
        self.assertEqual(periods, [60, 1440])
        self.assertEqual(lo, 1757289600)
        self.assertEqual(hi, 1757293200)



if __name__ == "__main__":
    unittest.main()
