import unittest
from datetime import datetime, timezone
from unittest import mock

from Tools import Board as B
from Tools import Common as C


def approx(a, b):
    return abs(a - b) < 1e-9


def _game(**kw):
    g = {"game_id": "2026_10_NE_DET", "start_time": "2026-11-15T09:30:00-05:00", "home": "DET", "away": "NE",
         "status": "scheduled", "score": None,
         "market": {"home_yes_ask": 0.58, "away_yes_ask": 0.43, "open_home_yes": 0.53, "source": "Kalshi", "as_of": "x"},
         "model": {"home_win_prob": 0.594, "model_name": B.MODEL, "tested": False},
         "edge_home_c": B.EdgeC(0.594, 0.58), "edge_away_c": B.EdgeC(1 - 0.594, 0.43),
         "context": {"injuries": [{"team": "DET", "player": "X", "status": "Out", "note": None}], "recent_form": None,
                     "rest_days": None, "head_to_head": None, "notes": [B.MODEL_NOTE]},
         "props": [{"player": "A", "market": "anytime_td", "line": 1, "yes_ask": 0.5, "model_prob": None, "edge_c": None,
                    "tested": False}]}
    g.update(kw)
    return g


def _board(games):
    return {"sport": "NFL", "generated_at": "2026-11-14T18:00:00-05:00", "games": games}


def test_edge_formula():
    assert B.EdgeC(0.60, 0.55) == 3.0
    assert B.EdgeC(None, 0.5) is None and B.EdgeC(0.5, None) is None


def test_validate_ok_and_failures():
    assert B.Validate(_board([_game()])) == 1
    bad = _game(); del bad["market"]["open_home_yes"]
    with T.assertRaises(ValueError):
        B.Validate(_board([bad]))
    bad = _game(); bad["props"][0]["tested"] = True
    with T.assertRaises(ValueError):
        B.Validate(_board([bad]))
    bad = _game(edge_home_c=9.9)
    with T.assertRaises(ValueError):
        B.Validate(_board([bad]))
    b = _board([_game()]); b["generated_at"] = "2026-11-14T23:00:00Z"
    with T.assertRaises(ValueError):
        B.Validate(b)
    b = _board([_game()]); b["sport"] = "nfl"
    with T.assertRaises(ValueError):
        B.Validate(b)
    bad = _game(); bad["model"]["tested"] = True
    with T.assertRaises(ValueError):
        B.Validate(_board([bad]))


def test_nulls_allowed():
    g = _game(model={"home_win_prob": None, "model_name": None, "tested": False}, edge_home_c=None, edge_away_c=None)
    g["market"].update(home_yes_ask=None, away_yes_ask=None, open_home_yes=None)
    g["context"]["injuries"] = None
    assert B.Validate(_board([g])) == 1


def test_props_parse_line_and_untested():
    mk = {"KXNFLRECYDS": [{"event_ticker": "KXNFLRECYDS-26NOV15CARNO", "yes_sub_title": "Alvin Kamara: 60+",
                           "floor_strike": 59.5, "yes_ask_dollars": "0.0700"},
                          {"event_ticker": "KXNFLRECYDS-26NOV15KCATL", "yes_sub_title": "Other: 50+", "floor_strike": 49.5,
                           "yes_ask_dollars": "0.30"}],
          "KXNFLTD": [{"event_ticker": "KXNFLTD-26NOV15CARNO", "yes_sub_title": "Bryce Young: 1+", "floor_strike": 0.5,
                       "yes_ask_dollars": "1.0000"}]}
    out = B.NflProps(mk, "26NOV15CARNO", None)
    assert len(out) == 2
    rec = next(p for p in out if p["market"] == "receiving_yards")
    assert rec == {"player": "Alvin Kamara", "market": "receiving_yards", "line": 60, "yes_ask": 0.07,
                   "model_prob": None, "edge_c": None, "tested": False}
    td = next(p for p in out if p["market"] == "anytime_td")
    assert td["line"] == 1 and td["yes_ask"] is None  # $1.00 ask = no real offer


def test_head_to_head():
    gs = [{"season": 2025, "gameday": "2025-10-01", "home_team": "DET", "away_team": "NE", "home_score": 20, "away_score": 10},
          {"season": 2026, "gameday": "2026-11-15", "home_team": "DET", "away_team": "NE", "home_score": None, "away_score": None}]
    assert B.HeadToHead(gs, "DET", "NE", "2026-11-15").startswith("2025-10-01 NE 10 @ DET 20")
    assert B.HeadToHead(gs, "DET", "KC", "2026-11-15") == "no meeting in nflverse schedule 2025-2026"


def test_first_price_skips_wide_placeholder():
    cs = {"candlesticks": [
        {"end_period_ts": 1793685600, "yes_ask": {"close_dollars": "0.74"}, "yes_bid": {"close_dollars": "0.24"}},
        {"end_period_ts": 1793689200, "yes_ask": {"close_dollars": "0.54"}, "yes_bid": {"close_dollars": "0.52"}}]}
    with mock.patch.object(C, "GetJson", lambda *a, **k: cs):
        fp = C.KalshiFirstPrice("KXNFLGAME", "T", "2026-11-03T05:06:00Z", datetime(2026, 11, 15, tzinfo=timezone.utc))
    assert fp["ask"] == 0.54 and approx(fp["mid"], 0.53)


def test_devig_and_match():
    fa, fb, over, hold = C.DevigMultiplicative(C.AmericanToProb(-150), C.AmericanToProb(130))
    assert approx(fa + fb, 1.0) and over > 0 and 0 < hold < over
    mk = [{"event_ticker": "KXNFLGAME-26NOV15NEDET", "ticker": "KXNFLGAME-26NOV15NEDET-DET", "yes_sub_title": "Detroit",
           "yes_ask_dollars": "0.58", "yes_bid_dollars": "0.57"},
          {"event_ticker": "KXNFLGAME-26NOV15NEDET", "ticker": "KXNFLGAME-26NOV15NEDET-NE", "yes_sub_title": "New England",
           "yes_ask_dollars": "0.43", "yes_bid_dollars": "0.42"}]
    m = C.MatchKalshiGame(mk, {"location": "Detroit", "displayName": "Detroit Lions"},
                          {"location": "New England", "displayName": "New England Patriots"},
                          datetime(2026, 11, 15, 14, 30, tzinfo=timezone.utc))
    assert m["event_suffix"] == "26NOV15NEDET" and m["home"]["ask"] == 0.58 and m["away"]["ask"] == 0.43


T = unittest.TestCase()


class BoardTests(unittest.TestCase):
    pass


for _n, _f in list(globals().items()):
    if _n.startswith("test_") and callable(_f):
        setattr(BoardTests, _n, (lambda f: lambda self: f())(_f))


if __name__ == "__main__":
    unittest.main()
