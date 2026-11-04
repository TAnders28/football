r"""
PredictDay(date): NFL picks for one ET gameday with the backtested strategies (same API as basketball's
Backtest.PredictDay). Same-day = pass today's date.

    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-05            # that ET gameday
    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-05 --week     # that day + rest of its NFL week
    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-04 --next     # no games that day -> next slate
    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-05 --no-refresh   # last saved schedule
    (--date defaults to today in America/New_York; --season defaults from the date)

    from Backtest.PredictDay import PredictDay; out = PredictDay("2026-11-05")   # {"meta", "games", "markdown"}

What it does
  1. Refreshes the season's schedule + posted lines from nflverse (nflreadpy.load_schedules, schedule only)
     into data/backtest/today/schedule_<season>.json with a pulled-at timestamp.
     Without a fresh pull it falls back to data/backtest/games_<season>.json (GetSeasonGames.py).
  2. Slate = games whose ET gameday is --date. If none: says so and names the next gameday
     (or scores it with --next).
  3. Builds pre-game features exactly like Features.BuildRows: only games FINISHED on days before the
     slate feed the margin model, form, ATS and records. Slate games' scores are never read (they are
     stripped before anything else touches them).
  4. Fits the Lab's walk-forward models on seasons before this one (same as the backtest did for this
     season) and scores each game:
       always_home              baseline
       market_fav               side the de-vigged moneyline favours
       model_edge_5pct (v1)     margin model vs market, bet if >= 5 pts
       value_full2_edge8        SPARSE VALUE (the backtest winner); stake = half Kelly, cap 10% of bank
       div_full3_edge2          divisional add-on from the hunt; stake 2% of bank
  5. One recommendation per game: BET the value pick if it fires, else BET the divisional add-on if it
     fires, else PASS with the model's lean. Odds shown are the nflverse posted lines at pull time
     (the backtest used closing lines, so prices may still move).

Writes data/backtest/today/predictions_<date>[_week].json (readable JSON) and .md.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from zoneinfo import ZoneInfo

from Backtest.Features import Bettable, Logit, SlateOf, _avg
from Backtest.Lab import AllRows, AttachModels, Kelly, LoadSeason, SelEdge
from Backtest.Hunt import FILTERS, Sel
from Backtest.Model import MarginModel, SeasonAverageMargins
from Backtest.Odds import FairProbs

ET = ZoneInfo("America/New_York")
TODAY_DIR = os.path.join("data", "backtest", "today")
OUTCOME_KEYS = ("home_score", "away_score", "result", "total", "overtime")


# ----------------------------------------------------------------------------- data
def Clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def PullSchedule(season: int) -> dict:
    import nflreadpy as nfl
    df = nfl.load_schedules(seasons=[season])
    games = [{k: Clean(v) for k, v in rec.items()} for rec in df.to_dicts()]
    out = {"season": season, "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": "nflverse schedules via nflreadpy.load_schedules", "games": games}
    os.makedirs(TODAY_DIR, exist_ok=True)
    with open(os.path.join(TODAY_DIR, f"schedule_{season}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def LoadSchedule(season: int, refresh: bool) -> dict:
    if refresh:
        try:
            return PullSchedule(season)
        except Exception as exc:   # offline etc. -> fall back to what is on disk
            print(f"[warn] schedule refresh failed ({exc}); using saved data")
    path = os.path.join(TODAY_DIR, f"schedule_{season}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    games = LoadSeason(season)
    summ = os.path.join("data", "backtest", f"pull_summary_{season}.json")
    pulled = json.load(open(summ, encoding="utf-8")).get("pulled_at_utc") if os.path.exists(summ) else None
    return {"season": season, "pulled_at_utc": pulled, "source": f"data/backtest/games_{season}.json", "games": games}


def Done(g) -> bool:
    return g.get("home_score") is not None and g.get("away_score") is not None


def NextGameday(games, date):
    later = sorted({str(g["gameday"]) for g in games if str(g["gameday"]) > date and not Done(g)})
    return later[0] if later else None


def PickSlate(games, date, week=False):
    slate = [g for g in games if str(g["gameday"]) == date]
    if week and slate:
        wk = slate[0]["week"]
        slate = [g for g in games if g["week"] == wk and str(g["gameday"]) >= date]
    slate.sort(key=lambda g: (str(g["gameday"]), g.get("gametime") or "", g["game_id"]))
    return slate


# ----------------------------------------------------------------------------- leak-free features
def SlateRows(games, prior_games, slate):
    """Same features as Features.BuildRows, for not-yet-played games. Only games that finished on a
    day BEFORE each slate game's day are fed to the ratings."""
    model = MarginModel(SeasonAverageMargins(prior_games) if prior_games else {})
    margins, ats = defaultdict(list), defaultdict(list)
    first_day = min(str(g["gameday"]) for g in slate)
    hist = sorted((g for g in games if Bettable(g) and str(g["gameday"]) < first_day),
                  key=lambda g: (str(g["gameday"]), g.get("gametime") or "", g["game_id"]))
    for g in hist:
        h, a = g["home_team"], g["away_team"]
        res = g["home_score"] - g["away_score"]
        margins[h].append(res); margins[a].append(-res)
        if g.get("spread_line") is not None:
            ats[h].append(res - g["spread_line"]); ats[a].append(-res + g["spread_line"])
        model.Apply(g)
    rows = []
    for g0 in slate:
        g = {k: v for k, v in g0.items() if k not in OUTCOME_KEYS}       # never look at the score
        h, a = g["home_team"], g["away_team"]
        has_odds = g.get("home_moneyline") is not None and g.get("away_moneyline") is not None
        mh = FairProbs(g["home_moneyline"], g["away_moneyline"])[0] if has_odds else 0.5
        rh, ra = model.Rating(h), model.Rating(a)
        rows.append({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"], "gameday": str(g["gameday"]),
            "gametime": g.get("gametime"), "slate": f"{g['gameday']}|{SlateOf(g)}", "game_type": g.get("game_type"),
            "home_team": h, "away_team": a, "home_ml": g.get("home_moneyline"), "away_ml": g.get("away_moneyline"),
            "home_spread_odds": g.get("home_spread_odds") or -110, "away_spread_odds": g.get("away_spread_odds") or -110,
            "spread": g.get("spread_line"), "has_odds": has_odds,
            "mkt_home": mh, "mkt_logit": Logit(mh), "margin_logit": model.PredictedHomeMargin(h, a) / 7.4,
            "home_rating": rh, "away_rating": ra,
            "form3_diff": (_avg(margins[h][-3:]) - rh if margins[h] else 0.0) - (_avg(margins[a][-3:]) - ra if margins[a] else 0.0),
            "ats_diff": _avg(ats[h]) - _avg(ats[a]), "ats3_diff": _avg(ats[h][-3:]) - _avg(ats[a][-3:]),
            "rest_diff": ((g.get("home_rest") or 7) - (g.get("away_rest") or 7)) / 7.0,
            "div": float(g.get("div_game") or 0),
            "games_played": min(len(margins[h]), len(margins[a])),
            "home_record": Record(margins[h]), "away_record": Record(margins[a]),
            "result": None, "home_win": None, "home_cover": None,
        })
    return rows, (hist[-1]["gameday"] if hist else None), len(hist)


def Record(ms):
    w, l, t = sum(m > 0 for m in ms), sum(m < 0 for m in ms), sum(m == 0 for m in ms)
    return f"{w}-{l}" + (f"-{t}" if t else "")


# ----------------------------------------------------------------------------- scoring
def Team(r, side):
    return r[f"{side}_team"] if side else None


def ScoreRows(rows):
    v1, value, div = SelEdge("p_margin", 0.05), SelEdge("p_full2", 0.08), Sel("p_full3", 0.02, filt=FILTERS["div"])
    half = Kelly(0.5, 0.10)
    out = []
    for r in rows:
        fav = "home" if r["mkt_home"] >= 0.5 else "away"
        lean = "home" if r["p_full2"] >= 0.5 else "away"
        p_lean = r["p_full2"] if lean == "home" else 1 - r["p_full2"]
        eh = r["p_full2"] - r["mkt_home"]
        best_edge, edge_side = (eh, "home") if eh >= -eh else (-eh, "away")
        c1 = v1(r) if r["has_odds"] else []
        cv = value(r) if r["has_odds"] else []
        cd = div(r) if r["has_odds"] else []
        s = {"always_home": r["home_team"],
             "market_fav": Team(r, fav) if r["has_odds"] else None,
             "margin_model_fav": Team(r, "home" if r["margin_logit"] >= 0 else "away"),
             "model_edge_5pct_v1": c1[0]["team"] if c1 else None,
             "value_full2_edge8": cv[0]["team"] if cv else None,
             "div_full3_edge2": cd[0]["team"] if cd else None}
        if not r["has_odds"]:
            rec = {"action": "PASS", "team": s["margin_model_fav"], "stake_pct": 0.0,
                   "why": "No moneyline posted yet; margin model lean only (strategies need prices)."}
        elif cv:
            c = cv[0]
            st = half(c, 1.0)
            rec = {"action": "BET", "team": c["team"], "odds": c["odds"], "stake_pct": round(st * 100, 1),
                   "why": f"Sparse value fires: model {c['p']*100:.0f}% vs market {c['p_mkt']*100:.0f}% (+{c['edge']*100:.1f} pts >= 8); half Kelly {st*100:.1f}% of bank."}
        elif cd:
            c = cd[0]
            rec = {"action": "BET", "team": c["team"], "odds": c["odds"], "stake_pct": 2.0,
                   "why": f"Divisional add-on: full3 model {c['p']*100:.0f}% vs market {c['p_mkt']*100:.0f}% (+{c['edge']*100:.1f} pts); flat 2% of bank."}
        else:
            rec = {"action": "PASS", "team": Team(r, lean), "stake_pct": 0.0,
                   "why": f"No edge: best model edge {best_edge*100:+.1f} pts on {Team(r, edge_side)} (< 8 needed); lean {Team(r, lean)} {p_lean*100:.0f}% to win."}
        out.append({
            "game_id": r["game_id"], "week": r["week"], "gameday": r["gameday"], "kickoff_et": r["gametime"],
            "matchup": f"{r['away_team']} @ {r['home_team']}", "records": f"{r['away_record']} @ {r['home_record']}",
            "away_ml": r["away_ml"], "home_ml": r["home_ml"], "spread_home": r["spread"], "divisional": bool(r["div"]),
            "market_home_prob": round(r["mkt_home"], 3) if r["has_odds"] else None,
            "margin_model_home_prob": round(1 / (1 + math.exp(-r["margin_logit"])), 3),
            "full2_home_prob": round(r["p_full2"], 3) if r["has_odds"] else None,
            "full3_home_prob": round(r["p_full3"], 3) if r["has_odds"] else None,
            "strategies": s, "recommended": rec})
    return out


# ----------------------------------------------------------------------------- output
def Kick(day, t):
    d = datetime.fromisoformat(f"{day}T{t or '13:00'}")
    return d.strftime("%a %b ") + str(d.day) + " " + d.strftime("%I:%M %p").lstrip("0") + " ET"


def Ml(x):
    return "-" if x is None else (f"+{x:.0f}" if x > 0 else f"{x:.0f}")


def Markdown(meta, preds):
    L = [f"# NFL picks: {meta['slate_label']}", "",
         f"- Date (ET): {meta['requested_date']} -- {meta['tonight_note']} (run {meta['generated_at_et']})",
         f"- Data: {meta['source']}, pulled {meta['pulled_at_et']}; ratings use {meta['finished_games_used']} finished games through {meta['last_finished_day']}.",
         f"- Models fit on {meta['model_train_seasons']} (walk-forward, as in the backtest). Lines are nflverse posted lines at pull time, not closing lines.", "",
         "| Game | Kickoff | Lines (away/home ML, home spread) | Records | always_home | market_fav | margin_model | model_edge_5% (v1) | value_edge8 (half Kelly) | div_add-on | **Recommended** | Why |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in preds:
        s, r = p["strategies"], p["recommended"]
        sp = "-" if p["spread_home"] is None else f"{p['spread_home']:+g}"
        rec = f"**{r['action']} {r['team']}**" + (f" {Ml(r.get('odds'))} @ {r['stake_pct']:g}%" if r["action"] == "BET" else "")
        L.append(f"| {p['matchup']}{' (div)' if p['divisional'] else ''} | {Kick(p['gameday'], p['kickoff_et'])} | {Ml(p['away_ml'])} / {Ml(p['home_ml'])}, {sp} | {p['records']} | "
                 f"{s['always_home']} | {s['market_fav'] or '-'} | {s['margin_model_fav']} | {s['model_edge_5pct_v1'] or 'no bet'} | "
                 f"{s['value_full2_edge8'] or 'no bet'} | {s['div_full3_edge2'] or 'no bet'} | {rec} | {r['why']} |")
    L += ["", "Stakes are % of the current bankroll. BET = a rule that made money out-of-sample in 2023-25 fired; PASS = no backtested edge (lean shown for interest only)."]
    return "\n".join(L) + "\n"


def SeasonOf(date):
    y, m = int(date[:4]), int(date[5:7])
    return y if m >= 3 else y - 1


def PredictDay(date=None, week=False, refresh=True, season=None, next_slate=False, write=True):
    """Score the NFL games on ET gameday `date` (YYYY-MM-DD; default today ET). Returns
    {"meta", "games", "markdown"}; games is [] when nothing is scheduled that day."""
    now = datetime.now(ET)
    today = now.date().isoformat()
    date = date or today
    season = season or SeasonOf(date)
    sched = LoadSchedule(season, refresh)
    games = sched["games"]
    asked = date
    slate = PickSlate(games, date, week)
    nxt = NextGameday(games, date) if not slate else None
    if not slate and next_slate and nxt:
        date = nxt
        slate = PickSlate(games, date, week)
    pulled = sched.get("pulled_at_utc")
    pulled_et = datetime.fromisoformat(pulled).astimezone(ET).strftime("%a %b %d %Y %I:%M %p ET") if pulled else "unknown"
    if not slate:
        msg = (f"No NFL games on {asked} (ET), season {season}. "
               + (f"Next slate: {nxt} -> run with --date {nxt} (or add --next)." if nxt else "No later unplayed games."))
        return {"meta": {"date": asked, "today_et": today, "note": msg, "next_gameday": nxt,
                         "pulled_at_et": pulled_et}, "games": [], "markdown": msg}
    note = (f"{len(slate)} game(s) on {asked}" if date == asked
            else f"no NFL game on {asked}; scoring the next slate ({date})")
    prior = LoadSeason(season - 1)
    rows, last_day, n_hist = SlateRows(games, prior, slate)
    rbs = AllRows(2012, season - 1)
    rbs[season] = rows
    AttachModels(rbs)
    preds = ScoreRows(rows)
    days = sorted({p["gameday"] for p in preds})
    meta = {"date": date, "requested_date": asked, "today_et": today, "tonight_note": note,
            "slate_label": f"week {slate[0]['week']}, {days[0]}" + (f" to {days[-1]}" if len(days) > 1 else ""),
            "source": sched.get("source"), "pulled_at_utc": pulled, "pulled_at_et": pulled_et,
            "last_finished_day": last_day, "finished_games_used": n_hist,
            "model_train_seasons": f"2013-{season - 1}", "generated_at_et": now.strftime("%Y-%m-%d %H:%M ET")}
    md = Markdown(meta, preds)
    if write:
        os.makedirs(TODAY_DIR, exist_ok=True)
        stem = os.path.join(TODAY_DIR, f"predictions_{date}{'_week' if week else ''}")
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "games": preds}, fh, indent=1)
        with open(stem + ".md", "w", encoding="utf-8") as fh:
            fh.write(md)
        meta["written"] = stem + ".json/.md"
    return {"meta": meta, "games": preds, "markdown": md}


def Main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="NFL picks for one ET gameday. Same-day = pass today's date.")
    ap.add_argument("--date", default=datetime.now(ET).strftime("%Y-%m-%d"), help="ET gameday YYYY-MM-DD (default: today ET)")
    ap.add_argument("--season", type=int, default=None, help="season start year (default from date)")
    ap.add_argument("--week", action="store_true", help="also score the rest of that NFL week")
    ap.add_argument("--next", action="store_true", help="if no games on --date, score the next gameday")
    ap.add_argument("--no-refresh", action="store_true", help="use the last saved schedule, no download")
    a = ap.parse_args()
    out = PredictDay(a.date, a.week, not a.no_refresh, a.season, a.next)
    print(out["markdown"])
    if out["meta"].get("written"):
        print(f"wrote {out['meta']['written']}")


if __name__ == "__main__":
    Main()
