import unittest

from Backtest.InjuryFeatures import BuildSeason, NameKey


def G(gid, week, day, home, away):
    return {"game_id": gid, "season": 2026, "week": week, "gameday": day, "gametime": "13:00", "home_team": home, "away_team": away}


def S(gid, team, pfr, name, off, de=0.0):
    return {"game_id": gid, "team": team, "pfr_player_id": pfr, "player": name, "offense_pct": off, "defense_pct": de}


class InjuryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.games = [G("g1", 1, "2026-09-13", "AAA", "BBB"), G("g2", 2, "2026-09-20", "AAA", "CCC")]
        self.pfr = {"QB1": "00-1", "WR1": "00-2", "LB1": "00-3"}
        self.snaps = [S("g1", "AAA", "QB1", "Q One", 1.0), S("g1", "AAA", "WR1", "W One", 0.8), S("g1", "AAA", "LB1", "L One", 0.0, 0.9),
                      # same-game (week 2) snaps must never be used for week 2 features
                      S("g2", "AAA", "QB1", "Q One", 0.05), S("g2", "AAA", "WR1", "W One", 0.0)]
        self.inj = [{"week": 2, "team": "AAA", "gsis_id": "00-1", "full_name": "Q One", "position": "QB", "report_status": "Out"},
                    {"week": 2, "team": "AAA", "gsis_id": "00-2", "full_name": "W One", "position": "WR", "report_status": "Doubtful"},
                    {"week": 2, "team": "AAA", "gsis_id": "00-3", "full_name": "L One", "position": "LB", "report_status": "Questionable"}]

    def Build(self, snaps=None):
        rows = BuildSeason(2026, pfr_map=self.pfr, games=self.games, prev_games=[], injuries=self.inj,
                           snaps=snaps if snaps is not None else self.snaps, prev_snaps=[])
        return {(r["game_id"], r["team"]): r for r in rows}

    def test_uses_only_earlier_games(self):
        f = self.Build()[("g2", "AAA")]
        self.assertAlmostEqual(f["qb_out"], 1.0)
        self.assertAlmostEqual(f["off_out"], 0.8)
        self.assertAlmostEqual(f["def_q"], 0.9)
        self.assertEqual(f["n_out"], 2)
        self.assertEqual(f["usage_source"], "team_last3_this_season")

    def test_same_game_snaps_do_not_change_features(self):
        a = self.Build()[("g2", "AAA")]
        changed = [dict(s) for s in self.snaps]
        for s in changed:
            if s["game_id"] == "g2":
                s["offense_pct"] = 0.99
        b = self.Build(changed)[("g2", "AAA")]
        self.assertEqual({k: a[k] for k in ("qb_out", "off_out", "def_q")}, {k: b[k] for k in ("qb_out", "off_out", "def_q")})

    def test_missing_report_is_a_gap_not_zero_signal(self):
        f = self.Build()[("g2", "CCC")]
        self.assertEqual(f["report_rows"], 0)
        self.assertEqual(f["qb_out"], 0.0)

    def test_name_key(self):
        self.assertEqual(NameKey("Odell Beckham Jr."), NameKey("odell beckham"))


if __name__ == "__main__":
    unittest.main()