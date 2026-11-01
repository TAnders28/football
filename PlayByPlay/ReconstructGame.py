r"""
Reconstruct NFL games from nflverse play-by-play JSONL into readable, drive-grouped JSON.

    .venv\Scripts\python.exe PlayByPlay\ReconstructGame.py --season 2026 --game-id 2026_05_BUF_LA
    .venv\Scripts\python.exe PlayByPlay\ReconstructGame.py --season 2026 --game-id 2026_05_BUF_LA --out examples\2026_05_BUF_LA.json
    .venv\Scripts\python.exe PlayByPlay\ReconstructGame.py --season 2023,2024,2025,2026 --all

Reads  data/playbyplay/pbp_<season>.jsonl  (+ data/backtest/games_<season>.json for schedule/betting info)
Writes (default locations):
    data/playbyplay/<season>/<game_id>.json   one indented, drive-grouped JSON per game
    data/playbyplay/<season>.jsonl            (--all only) the whole season, one full game object per line,
                                              in the same order as the pbp file (by game_id)

Nothing from the raw pbp is thrown away -- it is reshaped:
    game        columns that are constant for the whole game (teams, stadium, weather, final score, lines)
    drives[]    grouped by `fixed_drive`; columns constant within a drive (drive_*) live on the drive
    plays[]     every remaining column, nested into groups (clock, situation, score, pass, rush,
                kicking, players, penalty, epa, wp, ...)
    flags       every 0/1 indicator column becomes a list of the flag names that were 1, grouped by
                category, e.g. "flags": {"formation": ["shotgun", "no_huddle"], "outcome": ["first_down"]}.
                A flag not listed was 0 (or null in the raw data).
    players     role -> {id, name, jersey}; numbered columns (solo_tackle_1_*, _2_*) become lists.
                Roles nflverse fills twice (passer_player_* and passer/passer_id, etc.) are merged
                when they name the same player and kept separately when they differ.
    cumulative  running game totals (total_home_epa, total_away_pass_wpa, ...) as
                cumulative.home/away.<stat>
    ticker      ESPN-style line, e.g. "2nd & 7 at LA 33: 9-M.Stafford pass short left to ..."
    other       any pbp column not covered by the mapping below lands here, so new nflverse columns
                are never silently dropped.

Integral floats (1.0, 25.0) are written as ints; nulls are omitted.
"""

import argparse
import json
import os
import re
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP_DIR = os.path.join(ROOT, "data", "playbyplay")
BACKTEST_DIR = os.path.join(ROOT, "data", "backtest")

# ---------------------------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------------------------

# Columns that describe the game; lifted to the top-level "game" object when constant across plays.
GAME_COLS = [
    "game_id", "old_game_id", "nfl_api_id", "season", "season_type", "week", "game_date", "start_time",
    "home_team", "away_team", "home_score", "away_score", "result", "total", "location", "div_game",
    "spread_line", "total_line", "home_coach", "away_coach", "stadium", "stadium_id", "game_stadium",
    "roof", "surface", "temp", "wind", "weather", "home_opening_kickoff",
]

# Columns that describe the drive; lifted to the drive object when constant within the drive.
DRIVE_COLS = [
    "fixed_drive", "fixed_drive_result", "drive_real_start_time", "drive_play_count",
    "drive_time_of_possession", "drive_first_downs", "drive_inside20", "drive_ended_with_score",
    "drive_quarter_start", "drive_quarter_end", "drive_yards_penalized", "drive_start_transition",
    "drive_end_transition", "drive_game_clock_start", "ydsnet", "drive_game_clock_end", "drive_start_yard_line",
    "drive_end_yard_line", "drive_play_id_started", "drive_play_id_ended",
]
DRIVE_FLAG_COLS = {"drive_inside20", "drive_ended_with_score"}

# 0/1 indicator columns -> flag category. Each play gets flags[category] = [names that were 1].
FLAG_GROUPS = OrderedDict([
    ("formation", ["shotgun", "no_huddle"]),
    ("qb", ["qb_dropback", "qb_kneel", "qb_spike", "qb_scramble", "qb_hit"]),
    ("play_kind", ["pass", "rush", "special", "play", "pass_attempt", "rush_attempt", "special_teams_play",
                   "aborted_play", "play_deleted", "sp", "quarter_end", "timeout", "goal_to_go"]),
    ("outcome", ["success", "first_down", "first_down_rush", "first_down_pass", "first_down_penalty",
                 "third_down_converted", "third_down_failed", "fourth_down_converted", "fourth_down_failed",
                 "complete_pass", "incomplete_pass", "out_of_bounds", "sack", "tackled_for_loss",
                 "solo_tackle", "assist_tackle", "tackle_with_assist", "series_success"]),
    ("scoring", ["touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown", "safety",
                 "extra_point_attempt", "two_point_attempt", "field_goal_attempt",
                 "defensive_two_point_attempt", "defensive_two_point_conv",
                 "defensive_extra_point_attempt", "defensive_extra_point_conv"]),
    ("turnover", ["interception", "fumble", "fumble_forced", "fumble_not_forced", "fumble_out_of_bounds",
                  "fumble_lost"]),
    ("kicking", ["kickoff_attempt", "punt_attempt", "punt_blocked", "touchback", "punt_inside_twenty",
                 "punt_in_endzone", "punt_out_of_bounds", "punt_downed", "punt_fair_catch",
                 "kickoff_inside_twenty", "kickoff_in_endzone", "kickoff_out_of_bounds", "kickoff_downed",
                 "kickoff_fair_catch", "own_kickoff_recovery", "own_kickoff_recovery_td"]),
    ("lateral", ["lateral_reception", "lateral_rush", "lateral_return", "lateral_recovery"]),
    ("review", ["penalty", "replay_or_challenge"]),
])

# Plain (non-flag) play columns -> nested group. Keys inside a group drop redundant prefixes.
PLAY_GROUPS = OrderedDict([
    ("ids", {"play_id": "play_id", "order_sequence": "order_sequence", "drive": "nfl_drive",
             "time_of_day": "time_of_day", "end_clock_time": "end_clock_time", "play_clock": "play_clock",
             "end_yard_line": "end_yard_line"}),
    ("clock", {"qtr": "qtr", "time": "time", "game_half": "half",
               "quarter_seconds_remaining": "quarter_sec_left", "half_seconds_remaining": "half_sec_left",
               "game_seconds_remaining": "game_sec_left"}),
    ("situation", {"posteam": "offense", "posteam_type": "offense_side", "defteam": "defense",
                   "down": "down", "ydstogo": "ydstogo", "yrdln": "yrdln", "yardline_100": "yardline_100",
                   "side_of_field": "side_of_field"}),
    ("series", {"series": "number", "series_result": "result"}),
    ("timeouts", {"home_timeouts_remaining": "home", "away_timeouts_remaining": "away",
                  "posteam_timeouts_remaining": "offense", "defteam_timeouts_remaining": "defense",
                  "timeout_team": "called_by"}),
    ("score_before", {"total_home_score": "home", "total_away_score": "away", "posteam_score": "offense",
                      "defteam_score": "defense", "score_differential": "offense_diff"}),
    ("score_after", {"posteam_score_post": "offense", "defteam_score_post": "defense",
                     "score_differential_post": "offense_diff"}),
    ("type", {"play_type": "play_type", "play_type_nfl": "nfl", "st_play_type": "special_teams"}),
    ("result", {"yards_gained": "yards_gained", "ydsnet": "drive_net_yards"}),
    ("pass", {"pass_length": "length", "pass_location": "location", "air_yards": "air_yards",
              "yards_after_catch": "yards_after_catch", "passing_yards": "passing_yards",
              "receiving_yards": "receiving_yards", "cp": "cp", "cpoe": "cpoe", "xpass": "xpass",
              "pass_oe": "pass_oe", "xyac_epa": "xyac_epa", "xyac_mean_yardage": "xyac_mean_yardage",
              "xyac_median_yardage": "xyac_median_yardage", "xyac_success": "xyac_success",
              "xyac_fd": "xyac_fd"}),
    ("rush", {"run_location": "location", "run_gap": "gap", "rushing_yards": "rushing_yards"}),
    ("kicking", {"kick_distance": "kick_distance", "field_goal_result": "field_goal_result",
                 "extra_point_result": "extra_point_result", "two_point_conv_result": "two_point_result",
                 "return_team": "return_team", "return_yards": "return_yards"}),
    ("scoring", {"td_team": "td_team"}),
    ("penalty", {"penalty_team": "team", "penalty_type": "type", "penalty_yards": "yards"}),
    ("review", {"replay_or_challenge_result": "result"}),
    ("epa", {"ep": "ep", "epa": "epa", "qb_epa": "qb_epa", "air_epa": "air_epa", "yac_epa": "yac_epa",
             "comp_air_epa": "comp_air_epa", "comp_yac_epa": "comp_yac_epa"}),
    ("wp", {"wp": "wp", "def_wp": "def_wp", "home_wp": "home_wp", "away_wp": "away_wp",
            "home_wp_post": "home_wp_post", "away_wp_post": "away_wp_post", "wpa": "wpa",
            "vegas_wp": "vegas_wp", "vegas_home_wp": "vegas_home_wp", "vegas_wpa": "vegas_wpa",
            "vegas_home_wpa": "vegas_home_wpa", "air_wpa": "air_wpa", "yac_wpa": "yac_wpa",
            "comp_air_wpa": "comp_air_wpa", "comp_yac_wpa": "comp_yac_wpa"}),
    ("next_score_prob", {"no_score_prob": "no_score", "td_prob": "td", "fg_prob": "fg", "safety_prob": "safety",
                         "opp_td_prob": "opp_td", "opp_fg_prob": "opp_fg", "opp_safety_prob": "opp_safety",
                         "extra_point_prob": "extra_point", "two_point_conversion_prob": "two_point"}),
])

# Running game totals: total_<home|away>_<stat> -> cumulative.<home|away>.<stat>
CUMULATIVE_STATS = ["epa", "rush_epa", "pass_epa", "comp_air_epa", "comp_yac_epa", "raw_air_epa",
                    "raw_yac_epa", "rush_wpa", "pass_wpa", "comp_air_wpa", "comp_yac_wpa", "raw_air_wpa",
                    "raw_yac_wpa"]

# Player roles: role -> {field: column}. Each role becomes players.<role> = {id, name, ...}.
PLAYER_ROLES = OrderedDict([
    ("passer", {"id": "passer_player_id", "name": "passer_player_name"}),
    ("dropback_qb", {"id": "passer_id", "name": "passer", "jersey": "passer_jersey_number"}),
    ("rusher", {"id": "rusher_player_id", "name": "rusher_player_name"}),
    ("ball_carrier", {"id": "rusher_id", "name": "rusher", "jersey": "rusher_jersey_number"}),
    ("receiver", {"id": "receiver_player_id", "name": "receiver_player_name"}),
    ("target", {"id": "receiver_id", "name": "receiver", "jersey": "receiver_jersey_number"}),
    ("primary", {"id": "id", "name": "name", "jersey": "jersey_number"}),
    ("fantasy", {"id": "fantasy_player_id", "name": "fantasy_player_name"}),
    ("fantasy_alt", {"id": "fantasy_id", "name": "fantasy"}),
    ("td_scorer", {"id": "td_player_id", "name": "td_player_name"}),
    ("lateral_receiver", {"id": "lateral_receiver_player_id", "name": "lateral_receiver_player_name",
                          "yards": "lateral_receiving_yards"}),
    ("lateral_rusher", {"id": "lateral_rusher_player_id", "name": "lateral_rusher_player_name",
                        "yards": "lateral_rushing_yards"}),
    ("lateral_sack", {"id": "lateral_sack_player_id", "name": "lateral_sack_player_name"}),
    ("interceptor", {"id": "interception_player_id", "name": "interception_player_name"}),
    ("lateral_interceptor", {"id": "lateral_interception_player_id", "name": "lateral_interception_player_name"}),
    ("punt_returner", {"id": "punt_returner_player_id", "name": "punt_returner_player_name"}),
    ("lateral_punt_returner", {"id": "lateral_punt_returner_player_id", "name": "lateral_punt_returner_player_name"}),
    ("kickoff_returner", {"id": "kickoff_returner_player_id", "name": "kickoff_returner_player_name"}),
    ("lateral_kickoff_returner", {"id": "lateral_kickoff_returner_player_id",
                                  "name": "lateral_kickoff_returner_player_name"}),
    ("punter", {"id": "punter_player_id", "name": "punter_player_name"}),
    ("kicker", {"id": "kicker_player_id", "name": "kicker_player_name"}),
    ("own_kickoff_recoverer", {"id": "own_kickoff_recovery_player_id", "name": "own_kickoff_recovery_player_name"}),
    ("blocked_by", {"id": "blocked_player_id", "name": "blocked_player_name"}),
    ("sacked_by", {"id": "sack_player_id", "name": "sack_player_name"}),
    ("safety_by", {"id": "safety_player_id", "name": "safety_player_name"}),
    ("penalized", {"id": "penalty_player_id", "name": "penalty_player_name"}),
])

# Numbered player lists: list name -> (column template, fields, count). Become players.<list> = [ {...}, ... ].
PLAYER_LISTS = OrderedDict([
    ("solo_tacklers", ("solo_tackle_{n}_{f}", ["team", "player_id", "player_name"], 2)),
    ("assist_tacklers", ("assist_tackle_{n}_{f}", ["team", "player_id", "player_name"], 4)),
    ("tacklers_with_assist", ("tackle_with_assist_{n}_{f}", ["team", "player_id", "player_name"], 2)),
    ("tackle_for_loss", ("tackle_for_loss_{n}_{f}", ["player_id", "player_name"], 2)),
    ("qb_hitters", ("qb_hit_{n}_{f}", ["player_id", "player_name"], 2)),
    ("half_sackers", ("half_sack_{n}_{f}", ["player_id", "player_name"], 2)),
    ("pass_defenders", ("pass_defense_{n}_{f}", ["player_id", "player_name"], 2)),
    ("forced_fumble_by", ("forced_fumble_player_{n}_{f}", ["team", "player_id", "player_name"], 2)),
    ("fumbled_by", ("fumbled_{n}_{f}", ["team", "player_id", "player_name"], 2)),
    ("fumble_recovered_by", ("fumble_recovery_{n}_{f}", ["team", "yards", "player_id", "player_name"], 2)),
])
LIST_FIELD_NAMES = {"player_id": "id", "player_name": "name", "team": "team", "yards": "yards"}

ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

def Clean(value):
    """Integral floats -> int (and -0.0 -> 0); everything else unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def IsFlagValue(value) -> bool:
    return value in (0, 1, 0.0, 1.0) and not isinstance(value, bool)


def ReadGameRows(season: int, game_id: str | None):
    """Yield raw pbp dicts (one game, or all games if game_id is None). Cheap string pre-filter."""
    path = os.path.join(PBP_DIR, f"pbp_{season}.jsonl")
    needle = f'"game_id":"{game_id}"' if game_id else None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if needle and needle not in line:
                continue
            row = json.loads(line)
            if game_id is None or row.get("game_id") == game_id:
                yield row


def LoadSchedule(season: int) -> dict:
    path = os.path.join(BACKTEST_DIR, f"games_{season}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {g["game_id"]: g for g in json.load(f)}


def ConstantColumns(rows, cols) -> dict:
    """Columns from `cols` whose (non-null) value is identical on every row that has it."""
    out = {}
    for col in cols:
        values = {json.dumps(r[col]) for r in rows if col in r}
        if len(values) == 1:
            out[col] = Clean(rows[0].get(col, next(r[col] for r in rows if col in r)))
    return out


def Situation(play: dict) -> str:
    """ESPN-style situation text, e.g. '1st & 10 at BUF 25' / '3rd & Goal at LA 4'."""
    s = play.get("situation", {})
    down, togo, spot = s.get("down"), s.get("ydstogo"), s.get("yrdln")
    if down is None:
        return f"at {spot}" if spot else ""
    goal = "goal_to_go" in play.get("flags", {}).get("play_kind", [])
    return f"{ORDINAL.get(down, down)} & {'Goal' if goal else togo} at {spot}"


# ---------------------------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------------------------

def BuildPlay(row: dict, skip: set) -> dict:
    used = set(skip)
    play = OrderedDict()
    play["desc"] = row.get("desc")
    used.add("desc")

    for group, mapping in PLAY_GROUPS.items():
        block = OrderedDict((key, Clean(row[col])) for col, key in mapping.items() if row.get(col) is not None)
        used.update(mapping)
        if block:
            play[group] = block

    flags = OrderedDict()
    for group, cols in FLAG_GROUPS.items():
        on = [c for c in cols if row.get(c) in (1, 1.0)]
        used.update(cols)
        if on:
            flags[group] = on
    play["flags"] = flags

    cumulative = OrderedDict()
    for side in ("home", "away"):
        block = OrderedDict()
        for stat in CUMULATIVE_STATS:
            col = f"total_{side}_{stat}"
            used.add(col)
            if row.get(col) is not None:
                block[stat] = Clean(row[col])
        if block:
            cumulative[side] = block
    if cumulative:
        play["cumulative"] = cumulative

    players = OrderedDict()
    for role, fields in PLAYER_ROLES.items():
        used.update(fields.values())
        person = OrderedDict((f, Clean(row[c])) for f, c in fields.items() if row.get(c) is not None)
        if person:
            players[role] = person
    for name, (template, fields, count) in PLAYER_LISTS.items():
        entries = []
        for n in range(1, count + 1):
            cols = {LIST_FIELD_NAMES[f]: template.format(n=n, f=f) for f in fields}
            used.update(cols.values())
            entry = OrderedDict((k, Clean(row[c])) for k, c in cols.items() if row.get(c) is not None)
            if entry:
                entries.append(entry)
        if entries:
            players[name] = entries
    if players:
        play["players"] = players

    # Safety net: anything unmapped is kept verbatim (flags still become tags).
    other = OrderedDict()
    extra_flags = []
    for col, value in row.items():
        if col in used or value is None:
            continue
        if IsFlagValue(value) and col not in ("down",):
            if value in (1, 1.0):
                extra_flags.append(col)
        else:
            other[col] = Clean(value)
    if extra_flags:
        flags["other"] = extra_flags
    if other:
        play["other"] = other

    MergeDuplicateRoles(players)

    situation = Situation(play)
    text = re.sub(r"^\(\d{0,2}:\d{2}\)\s*", "", play.get("desc") or "")
    qtr = row.get("qtr")
    play["ticker"] = OrderedDict([
        ("clock", f"{'OT' if qtr and qtr >= 5 else 'Q' + str(int(qtr)) if qtr else ''} {row.get('time', '')}".strip()),
        ("situation", situation),
        ("line", f"{situation}: {text}" if situation else text),
    ])
    order = ["ticker", "desc", "type", "flags"]
    return OrderedDict([(k, play[k]) for k in order if k in play] +
                       [(k, v) for k, v in play.items() if k not in order])


# Role pairs that nflverse often fills identically; merged into the first when id+name match.
DUPLICATE_ROLES = [("passer", "dropback_qb"), ("rusher", "ball_carrier"), ("receiver", "target"),
                   ("fantasy", "fantasy_alt")]


def MergeDuplicateRoles(players: dict) -> None:
    """Fold e.g. passer/dropback_qb into one entry when they are the same person (adds jersey).
    When they differ (scramble, sack, lateral, ...) both are kept."""
    for keep, dup in DUPLICATE_ROLES:
        a, b = players.get(keep), players.get(dup)
        if a and b and a.get("id") == b.get("id") and a.get("name") == b.get("name"):
            for k, v in b.items():
                a.setdefault(k, v)
            del players[dup]


def BuildDrive(rows: list, game_skip: set) -> dict:
    drive_info = ConstantColumns(rows, DRIVE_COLS)
    skip = game_skip | set(drive_info)

    drive = OrderedDict()
    drive["number"] = drive_info.pop("fixed_drive", None)
    offenses = Counter(r["posteam"] for r in rows if r.get("posteam"))
    drive["offense"] = offenses.most_common(1)[0][0] if offenses else None
    defenses = Counter(r["defteam"] for r in rows if r.get("defteam"))
    drive["defense"] = defenses.most_common(1)[0][0] if defenses else None
    drive["result"] = drive_info.pop("fixed_drive_result", None)

    net = [r["ydsnet"] for r in rows if r.get("ydsnet") is not None]
    yards = drive_info.pop("ydsnet", Clean(net[-1]) if net else 0)
    plays_count = drive_info.get("drive_play_count", sum(1 for r in rows if r.get("play") in (1, 1.0)))
    top = drive_info.get("drive_time_of_possession", "0:00")
    drive["summary"] = (f"{drive['offense']} Drive: {plays_count} plays, {yards} yards, {top}"
                        f" -- {drive['result']}")
    drive["yards"] = yards

    info = OrderedDict()
    for col, value in drive_info.items():
        key = col.removeprefix("drive_")
        info[key] = (value == 1) if col in DRIVE_FLAG_COLS else value
    drive["info"] = info
    drive["plays"] = [BuildPlay(r, skip) for r in rows]
    return drive


def BuildGame(rows: list, schedule: dict) -> dict:
    rows = sorted(rows, key=lambda r: (r.get("order_sequence", 0), r.get("play_id", 0)))
    game_info = ConstantColumns(rows, GAME_COLS)
    game_skip = set(game_info)

    game = OrderedDict()
    gid = game_info.get("game_id") or rows[0]["game_id"]
    game["game_id"] = gid
    home, away = game_info.get("home_team"), game_info.get("away_team")
    game["matchup"] = f"{away} @ {home}"
    game["final"] = f"{away} {game_info.get('away_score')} - {home} {game_info.get('home_score')}"
    info = OrderedDict()
    for col, value in game_info.items():
        if col == "game_id":
            continue
        info[col] = (value == 1) if col in ("div_game", "home_opening_kickoff") else value
    game["info"] = info

    sched = schedule.get(gid)
    if sched:
        game["schedule"] = OrderedDict((k, v) for k, v in sched.items() if v is not None)

    drives, loose = OrderedDict(), []
    for r in rows:
        key = r.get("fixed_drive")
        if key is None:
            loose.append(r)
        else:
            drives.setdefault(key, []).append(r)

    built = [BuildDrive(d, game_skip) for d in drives.values()]
    game["drive_count"] = len(built)
    game["play_count"] = len(rows)
    game["scoring_summary"] = [
        OrderedDict([("drive", d["number"]), ("qtr", last.get("clock", {}).get("qtr")),
                     ("time", last.get("clock", {}).get("time")), ("team", _scoring_team(d)),
                     ("result", d["result"]), ("score_after", f"{away} {_post(last, home, away)}")])
        for d in built if d["info"].get("ended_with_score") and (last := _scoring_play(d))
    ]
    game["drives"] = built
    if loose:
        game["non_drive_plays"] = [BuildPlay(r, game_skip) for r in loose]
    return game


def _scoring_team(drive: dict):
    for p in drive["plays"]:
        if p.get("scoring", {}).get("td_team"):
            return p["scoring"]["td_team"]
    return drive["defense"] if drive["result"] == "Safety" else drive["offense"]


def _scoring_play(drive: dict):
    for p in reversed(drive["plays"]):
        if "score_before" in p:
            return p
    return None


def _post(play: dict, home: str, away: str) -> str:
    """Away/home score after the drive's last play (from offense-relative post scores)."""
    sb, sa, sit = play.get("score_before", {}), play.get("score_after", {}), play.get("situation", {})
    if sit.get("offense_side") == "home":
        h, a = sa.get("offense", sb.get("home")), sa.get("defense", sb.get("away"))
    else:
        a, h = sa.get("offense", sb.get("away")), sa.get("defense", sb.get("home"))
    return f"{a}, {home} {h}"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def WriteGame(game: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(game, f, indent=2, ensure_ascii=False)


def SeasonDir(season: int) -> str:
    return os.path.join(PBP_DIR, str(season))


def SeasonJsonl(season: int) -> str:
    return os.path.join(PBP_DIR, f"{season}.jsonl")


def ReconstructSeason(season: int, out_dir: str | None = None) -> dict:
    """Write every game of a season as <out_dir>/<game_id>.json and the season JSONL next to it."""
    schedule = LoadSchedule(season)
    by_game = OrderedDict()
    for r in ReadGameRows(season, None):
        by_game.setdefault(r["game_id"], []).append(r)

    out_dir = out_dir or SeasonDir(season)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), f"{os.path.basename(os.path.normpath(out_dir))}.jsonl")
    tmp_path = jsonl_path + ".tmp"
    written, failed = 0, []
    with open(tmp_path, "w", encoding="utf-8") as jl:
        for gid in list(by_game):
            rows = by_game.pop(gid)  # free memory as we go
            try:
                game = BuildGame(rows, schedule)
            except Exception as e:  # keep going; report at the end
                failed.append((gid, f"{type(e).__name__}: {e}"))
                continue
            WriteGame(game, os.path.join(out_dir, f"{gid}.json"))
            jl.write(json.dumps(game, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    os.replace(tmp_path, jsonl_path)
    print(f"{season}: {written} games -> {out_dir}  |  season JSONL -> {jsonl_path}")
    for gid, err in failed:
        print(f"  FAILED {gid}: {err}")
    return {"season": season, "written": written, "failed": failed, "dir": out_dir, "jsonl": jsonl_path}


def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", required=True, help="one season or comma list, e.g. 2023,2024,2025,2026")
    parser.add_argument("--game-id", help="e.g. 2026_05_BUF_LA")
    parser.add_argument("--all", action="store_true",
                        help="reconstruct every game -> data/playbyplay/<season>/ + data/playbyplay/<season>.jsonl")
    parser.add_argument("--out", help="output file (single game) or directory (--all, single season)")
    args = parser.parse_args()
    if not args.game_id and not args.all:
        parser.error("give --game-id or --all")
    seasons = [int(x) for x in str(args.season).split(",") if x.strip()]

    if args.game_id:
        if len(seasons) != 1:
            parser.error("--game-id takes a single --season")
        season = seasons[0]
        rows = list(ReadGameRows(season, args.game_id))
        if not rows:
            raise SystemExit(f"no plays for {args.game_id} in pbp_{season}.jsonl")
        out = args.out or os.path.join(SeasonDir(season), f"{args.game_id}.json")
        WriteGame(BuildGame(rows, LoadSchedule(season)), out)
        print(f"{args.game_id}: {len(rows)} plays -> {out}")
        return

    if args.out and len(seasons) > 1:
        parser.error("--out with --all only works for a single season")
    results = [ReconstructSeason(season, args.out) for season in seasons]
    if any(r["failed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    Main()
