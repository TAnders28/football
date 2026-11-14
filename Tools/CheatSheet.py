r"""
Game-day cheat sheets (decision support for Tim's own bets; read-only, no orders, no picks).

    .venv\Scripts\python.exe -m Tools.CheatSheet --league nfl --date 2026-11-15
    .venv\Scripts\python.exe -m Tools.CheatSheet --league cfb --date 2026-11-14 [--game "Texas A&M"]

Writes E:\AgentDocs\American-Footbot\cheatsheets\<date>_<league>\ : slate.md, slate.json, games\<away>-at-<home>.md

Per game: book odds ESPN shows (provider named) with multiplicative no-vig fair prices and hold for moneyline,
spread and total; Kalshi game-winner / spread / total quotes (mid, de-vig) near the book lines; line movement
(book open -> now, Kalshi first tight hourly quote -> now); injuries (ESPN; NFL only, ESPN has no CFB feed);
kickoff weather (Open-Meteo, domes marked); last-5 form with points for/against; season scoring and ATS (ESPN);
NFL model p_full2_kmid_T60 home win prob where it can be scored (reference only: failed holdout + SPA).
Unknown values print as "n/a"; nothing is invented.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import re
import time
from datetime import datetime

from Tools import Common as C

OUT_ROOT = os.path.join("E:\\AgentDocs", "American-Footbot", "cheatsheets")


def Pct(p, d=1):
    return "n/a" if p is None else f"{p * 100:.{d}f}%"


def Am(o):
    if o is None:
        return "n/a"
    return f"{o:+.0f}"


def Ln(x):
    return "n/a" if x is None else (f"{x:+g}" if isinstance(x, (int, float)) else str(x))


def FairAmerican(p):
    if p is None or p <= 0 or p >= 1:
        return "n/a"
    return f"{-100 * p / (1 - p):+.0f}" if p >= 0.5 else f"{100 * (1 - p) / p:+.0f}"


def KalshiLadder(markets, suffix, kind):
    """Spread: {team_name: [(strike, quote)]}; total: [(strike, quote)]."""
    out = {} if kind == "spread" else []
    for m in markets:
        ev = m.get("event_ticker") or ""
        if not ev.endswith("-" + suffix):
            continue
        q = C.Quote(m)
        k = m.get("floor_strike")
        if k is None:
            continue
        if kind == "spread":
            t = (m.get("yes_sub_title") or "").split(" wins by")[0].strip()
            out.setdefault(t, []).append((float(k), q))
        else:
            out.append((float(k), q))
    if kind == "spread":
        for v in out.values():
            v.sort(key=lambda x: x[0])
    else:
        out.sort(key=lambda x: x[0])
    return out


def Near(ladder, target, width=3.0):
    if target is None:
        return []
    return [(k, q) for k, q in ladder if abs(k - target) <= width + 1e-9]


def SeasonScoring(all_games, team, before):
    rows = [g for g in all_games if team in (g.get("home_team"), g.get("away_team")) and g.get("home_score") is not None
            and (g.get("gameday") or "") < before and str(g.get("season")) == before[:4] and g.get("game_type") == "REG"]
    if not rows:
        return None
    pf = pa = w = l = t = 0
    aw = al = ap = 0
    for g in rows:
        home = g["home_team"] == team
        f, a = (g["home_score"], g["away_score"]) if home else (g["away_score"], g["home_score"])
        pf += f; pa += a
        w += f > a; l += f < a; t += f == a
        sl = g.get("spread_line")  # nflverse: points the home team is favored by (closing)
        if sl is not None:
            m = (f - a) - (sl if home else -sl)
            aw += m > 0; al += m < 0; ap += m == 0
    n = len(rows)
    return {"games": n, "record": f"{w}-{l}" + (f"-{t}" if t else ""), "pf_pg": round(pf / n, 1), "pa_pg": round(pa / n, 1),
            "diff_pg": round((pf - pa) / n, 1),
            "ats": f"{aw}-{al}" + (f"-{ap}" if ap else "") if (aw + al + ap) else None}


def Ats(summary, team_id):
    for t in summary.get("againstTheSpread") or []:
        if str((t.get("team") or {}).get("id")) == str(team_id):
            recs = t.get("records") or []
            return ", ".join(f"{r.get('type') or r.get('name')}: {r.get('summary')}" for r in recs if r.get("summary")) or None
    return None


def Slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")


def BuildGame(league, ev, ctx):
    comp, h, a = C.Competitors(ev)
    start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
    eid = str(ev["id"])
    ha, aa = h["team"]["abbreviation"], a["team"]["abbreviation"]
    odds = C.EspnOdds(comp)
    summ, s_at = C.EspnSummary(league, eid)
    if not odds and summ.get("pickcenter"):
        odds = C.EspnOdds({"odds": summ.get("pickcenter")})
    fair = C.FairFromOdds(odds)
    km = C.MatchKalshiGame(ctx["k_game"], h["team"], a["team"], start)
    kal = {"game": None, "spread": {}, "total": [], "first_home": None}
    if km:
        qh, qa = km["home"], km["away"]
        dv = C.DevigMultiplicative(qh["mid"], qa["mid"]) if qh["mid"] and qa["mid"] else None
        kal["game"] = {"home": qh, "away": qa, "home_devig": dv[0] if dv else None, "event": km["event_ticker"]}
        kal["spread"] = KalshiLadder(ctx["k_spread"], km["event_suffix"], "spread")
        kal["total"] = KalshiLadder(ctx["k_total"], km["event_suffix"], "total")
        mh = next((m for m in ctx["k_game"] if m.get("ticker") == qh["ticker"]), None)
        if mh and mh.get("open_time"):
            try:
                kal["first_home"] = C.KalshiFirstPrice(C.KALSHI_SERIES[league]["game"], mh["ticker"], mh["open_time"], start)
            except Exception:
                kal["first_home"] = None
    inj = {}
    for side, t in (("home", h), ("away", a)):
        rows = ctx["inj"].get(str(t["team"]["id"]))
        inj[side] = rows if rows is not None else C.Injuries(summ, t["team"]["id"])
    form = {"home": C.LastFive(summ, h["team"]["id"]), "away": C.LastFive(summ, a["team"]["id"])}
    rec = {}
    for side, t in (("home", h), ("away", a)):
        rec[side] = {r.get("type"): r.get("summary") for r in t.get("records") or []}
    nfl_g = ctx["nfl_by_espn"].get(eid)
    season = {}
    if nfl_g:
        season = {"home": SeasonScoring(ctx["all_nfl"], nfl_g["home_team"], ctx["date"]),
                  "away": SeasonScoring(ctx["all_nfl"], nfl_g["away_team"], ctx["date"])}
    model = ctx["model"].get(eid)
    state = (ev.get("status") or {}).get("type", {})
    g = {"espn_event_id": eid, "game_id": nfl_g["game_id"] if nfl_g else None, "matchup": f"{aa} @ {ha}",
         "home": {"abbr": ha, "name": h["team"].get("displayName"), "rank": (h.get("curatedRank") or {}).get("current")},
         "away": {"abbr": aa, "name": a["team"].get("displayName"), "rank": (a.get("curatedRank") or {}).get("current")},
         "kickoff_et": C.IsoEt(start), "status": state.get("description"), "state": state.get("state"),
         "score": {"home": h.get("score"), "away": a.get("score")} if state.get("state") != "pre" else None,
         "broadcast": ", ".join(n for b in comp.get("broadcasts") or [] for n in b.get("names") or []) or None,
         "venue": (comp.get("venue") or {}).get("fullName"), "neutral_site": comp.get("neutralSite"),
         "book": odds, "fair": fair, "kalshi": kal,
         "model": None if not model else {"name": "p_full2_kmid_T60", "home_win_prob": model.get("p_home"),
                                          "status": model.get("status"), "note": "not validated: fee-adjusted edge failed holdout + SPA; reference only"},
         "espn_predictor": C.Predictor(summ), "injuries": inj, "form": form, "records": rec, "season": season,
         "ats": {"home": Ats(summ, h["team"]["id"]), "away": Ats(summ, a["team"]["id"])},
         "weather": C.Weather(comp.get("venue"), start), "fetched": {"espn_summary": s_at}}
    return g


def GameMd(g, league):
    H, A = g["home"]["abbr"], g["away"]["abbr"]
    rk = lambda t: f"#{t['rank']} " if t.get("rank") and t["rank"] < 99 else ""
    L = [f"# {rk(g['away'])}{g['away']['name']} @ {rk(g['home'])}{g['home']['name']}",
         f"Kickoff {datetime.fromisoformat(g['kickoff_et']).strftime('%a %b %d %I:%M %p ET')} | {g['venue'] or 'venue n/a'}"
         + (" (neutral)" if g.get("neutral_site") else "") + f" | TV {g['broadcast'] or 'n/a'} | status {g['status']}", ""]
    b, f = g["book"], g["fair"]
    L.append("## Market (no-vig = multiplicative de-vig of the two sides)")
    if b:
        L.append(f"Book: {b['provider'] or 'n/a'} via ESPN ({b.get('details') or 'n/a'}).")
        L += ["", "| market | side | book price | no-vig fair | fair odds | hold |", "|---|---|---|---|---|---|"]
        ml = f.get("ml")
        L.append(f"| moneyline | {H} | {Am(b['ml_home']['cur'])} | {Pct(ml and ml['home'])} | {FairAmerican(ml and ml['home'])} | {Pct(ml and ml['hold'])} |")
        L.append(f"| moneyline | {A} | {Am(b['ml_away']['cur'])} | {Pct(ml and ml['away'])} | {FairAmerican(ml and ml['away'])} | |")
        sp = f.get("spread")
        L.append(f"| spread | {H} {Ln(b['spread_home']['line'])} | {Am(b['spread_home']['odds'])} | {Pct(sp and sp['home'])} | {FairAmerican(sp and sp['home'])} | {Pct(sp and sp['hold'])} |")
        L.append(f"| spread | {A} {Ln(b['spread_away']['line'])} | {Am(b['spread_away']['odds'])} | {Pct(sp and sp['away'])} | {FairAmerican(sp and sp['away'])} | |")
        tt = f.get("total")
        tl = b["total"]["line"]
        L.append(f"| total | over {tl if tl is not None else 'n/a'} | {Am(b['total']['over_odds'])} | {Pct(tt and tt['over'])} | {FairAmerican(tt and tt['over'])} | {Pct(tt and tt['hold'])} |")
        L.append(f"| total | under {tl if tl is not None else 'n/a'} | {Am(b['total']['under_odds'])} | {Pct(tt and tt['under'])} | {FairAmerican(tt and tt['under'])} | |")
    else:
        L.append("Book odds: n/a (ESPN shows none for this game).")
    k = g["kalshi"]
    L += ["", "Kalshi (public API, yes bid/ask in $; mid = (bid+ask)/2):"]
    if k["game"]:
        qh, qa = k["game"]["home"], k["game"]["away"]
        L.append(f"- Game winner: {H} {qh['bid']}/{qh['ask']} (mid {qh['mid'] if qh['mid'] is not None else 'n/a'}), "
                 f"{A} {qa['bid']}/{qa['ask']} (mid {qa['mid'] if qa['mid'] is not None else 'n/a'}); de-vig {H} {Pct(k['game']['home_devig'])}")
        bl = b["spread_home"]["line"] if b else None
        for team, lad in (k["spread"] or {}).items():
            if bl is None:
                near = lad[:5]
            else:
                fav_margin = abs(bl)
                near = Near(lad, fav_margin, 3.0)
            if near:
                L.append(f"- Spread '{team} wins by over X': " + ", ".join(
                    f"{x:g}: {q['bid']}/{q['ask']}" for x, q in near))
        tl = b["total"]["line"] if b else None
        nt = Near(k["total"], tl, 3.0) if tl is not None else k["total"][:5]
        if nt:
            L.append("- Total 'over X': " + ", ".join(f"{x:g}: {q['bid']}/{q['ask']}" for x, q in nt))
    else:
        L.append("- no matching open Kalshi game market")
    L += ["", "## Line movement"]
    if b:
        mo = f.get("ml_open")
        L.append(f"- Moneyline {H}: open {Am(b['ml_home']['open'])} -> now {Am(b['ml_home']['cur'])} "
                 f"(fair {Pct(mo and mo['home'])} -> {Pct(f.get('ml') and f['ml']['home'])})")
        L.append(f"- Spread {H}: open {Ln(b['spread_home']['open_line'])} -> now {Ln(b['spread_home']['line'])}")
        L.append(f"- Total: open {b['total']['open_line'] if b['total']['open_line'] is not None else 'n/a'} -> now {b['total']['line'] if b['total']['line'] is not None else 'n/a'}")
    fh = k.get("first_home")
    if fh and k["game"]:
        L.append(f"- Kalshi {H} yes: first tight quote {fh['bid']}/{fh['ask']} ({fh['at']}) -> now {k['game']['home']['bid']}/{k['game']['home']['ask']}")
    if g["model"] or g["espn_predictor"]:
        L += ["", "## Models (reference only)"]
        if g["model"]:
            L.append(f"- p_full2_kmid_T60 {H} win: {Pct(g['model']['home_win_prob'])} ({g['model']['status']}; {g['model']['note']})")
        if g["espn_predictor"]:
            L.append(f"- ESPN FPI matchup predictor {H} win: {Pct(g['espn_predictor']['home'])}")
    L += ["", "## Injuries"]
    for side, ab in (("away", A), ("home", H)):
        rows = g["injuries"][side]
        if rows is None:
            L.append(f"- {ab}: n/a (ESPN publishes no injury report for this league/game)")
            continue
        rows = [r for r in rows if (r.get("status") or "") != "Active"]
        act = [r for r in rows if (r.get("status") or "") not in ("Injured Reserve",)]
        ir = len(rows) - len(act)
        txt = "; ".join(f"{r['player']} ({r.get('pos') or '?'}) {r['status']}" + (f" - {r['type']}" if r.get('type') else "") for r in act) or "none listed"
        L.append(f"- {ab}: {txt}" + (f" | +{ir} on IR" if ir else ""))
    L += ["", "## Weather", f"- {g['weather']['text']}"]
    L += ["", "## Form and matchup"]
    for side, ab in (("away", A), ("home", H)):
        rows = g["form"][side]
        r5 = " ".join(f"{r['result'] or '?'} {r['at_vs'] or ''}{r['opp'] or ''} {r['score'] or ''}".strip() for r in rows) or "n/a"
        recs = g["records"][side]
        rec = ", ".join(f"{k2}: {v}" for k2, v in recs.items() if v) or "n/a"
        L.append(f"- {ab} last 5: {r5}")
        L.append(f"  record {rec}; ATS {g['ats'][side] or 'n/a'}")
        s = (g["season"] or {}).get(side)
        if s:
            L.append(f"  season (nflverse, REG): {s['record']}, PF/G {s['pf_pg']}, PA/G {s['pa_pg']}, diff {s['diff_pg']:+}, "
                     f"ATS vs nflverse closing line {s.get('ats') or 'n/a'}")
        elif rows:
            pts = [(r["pf"], r["pa"]) for r in rows if r.get("pf") is not None]
            if pts:
                L.append(f"  last-{len(pts)} PF/G {sum(x for x, _ in pts) / len(pts):.1f}, PA/G {sum(y for _, y in pts) / len(pts):.1f}")
    L += ["", f"_Sources: ESPN site API (summary fetched {g['fetched']['espn_summary']}), Kalshi public API, "
          f"Open-Meteo, nflverse schedule. Decision support only - Tim places any bet._"]
    return "\n".join(L) + "\n"


def Build(league, date, out_root=OUT_ROOT, game_filter=None, log=print):
    t0 = time.time()
    events, espn_at = C.EspnScoreboard(league, date)
    events = [e for e in events if datetime.fromisoformat(e["date"].replace("Z", "+00:00")).astimezone(C.ET).date().isoformat() == date]
    if game_filter:
        gf = game_filter.lower()
        events = [e for e in events if gf in e.get("name", "").lower() or gf in e.get("shortName", "").lower()]
    ser = C.KALSHI_SERIES[league]
    ctx = {"date": date, "k_game": C.KalshiOpen(ser["game"])[0], "k_spread": C.KalshiOpen(ser["spread"])[0],
           "k_total": C.KalshiOpen(ser["total"])[0], "inj": {}, "nfl_by_espn": {}, "all_nfl": [], "model": {}}
    errors = []
    if league == "nfl":
        from Tools.Board import LoadNflGames
        try:
            ctx["inj"], _ = C.LeagueInjuries("nfl")
        except Exception as exc:
            errors.append(f"injuries: {type(exc).__name__}")
        from Backtest.PredictDay import SeasonOf
        ctx["all_nfl"] = LoadNflGames(range(SeasonOf(date) - 1, SeasonOf(date) + 1))
        ctx["nfl_by_espn"] = {str(g["espn"]): g for g in ctx["all_nfl"] if g.get("espn")}
        try:
            from Backtest.DailyCheck import Check
            dc = Check(date, out_dir=os.path.join("data", "tools", "dailycheck"))
            for rec in dc.get("games", []):
                g = next((x for x in ctx["all_nfl"] if x["game_id"] == rec["game_id"]), None)
                if g and g.get("espn"):
                    ctx["model"][str(g["espn"])] = rec
        except Exception as exc:
            errors.append(f"model: {type(exc).__name__}: {exc}")
    games = []
    for ev in events:
        try:
            games.append(BuildGame(league, ev, ctx))
        except Exception as exc:
            errors.append(f"{ev.get('shortName')}: {type(exc).__name__}: {exc}")
    games.sort(key=lambda g: (g["kickoff_et"], g["matchup"]))
    out = os.path.join(out_root, f"{date}_{league}")
    os.makedirs(os.path.join(out, "games"), exist_ok=True)
    gen = C.StampEt()
    S = [f"# {league.upper()} slate {date} - cheat sheet", f"Generated {gen}. {len(games)} games. Fair = multiplicative no-vig of the book ESPN shows; "
         "Kalshi mid = (bid+ask)/2. Decision support only, no picks.", "",
         "| kickoff (ET) | game | book | fav fair ML | hold | spread (home) | total | Kalshi home bid/ask | Kalshi de-vig home | model home | wx |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in games:
        f, b, k = g["fair"], g["book"], g["kalshi"]
        ml = f.get("ml")
        fav = None
        if ml:
            fav = (g["home"]["abbr"], ml["home"]) if ml["home"] >= ml["away"] else (g["away"]["abbr"], ml["away"])
        kg = k["game"]
        wx = g["weather"]
        wxs = "dome" if wx.get("indoor") else (f"{wx['temp_f']:.0f}F w{wx['wind_mph']:.0f}" if wx.get("temp_f") is not None else "n/a")
        S.append(f"| {datetime.fromisoformat(g['kickoff_et']).strftime('%I:%M %p')} | {g['matchup']} | {(b or {}).get('provider') or 'n/a'} | "
                 f"{(fav[0] + ' ' + Pct(fav[1])) if fav else 'n/a'} | {Pct(ml and ml['hold'])} | "
                 f"{Ln(b['spread_home']['line']) if b else 'n/a'} | {(b['total']['line'] if b and b['total']['line'] is not None else 'n/a')} | "
                 f"{(str(kg['home']['bid']) + '/' + str(kg['home']['ask'])) if kg else 'n/a'} | {Pct(kg and kg['home_devig'])} | "
                 f"{Pct(g['model'] and g['model']['home_win_prob'])} | {wxs} |")
        name = f"{Slug(g['away']['abbr'])}-at-{Slug(g['home']['abbr'])}.md"
        with open(os.path.join(out, "games", name), "w", encoding="utf-8") as fh:
            fh.write(GameMd(g, league))
    S += ["", "Per-game sheets: games\\<away>-at-<home>.md"]
    if errors:
        S += ["", "Errors: " + "; ".join(errors)]
    S.append("")
    S += [GameMd(g, league) for g in games]
    with open(os.path.join(out, "slate.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(S))
    res = {"league": league, "date": date, "generated_at": C.IsoEt(), "games": games, "errors": errors,
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(out, "slate.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    return res, out


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nfl", "cfb"], default="nfl")
    ap.add_argument("--date", default=datetime.now(C.ET).date().isoformat())
    ap.add_argument("--game", help="substring filter on the ESPN game name, e.g. 'Texas A&M' or 'DET'")
    ap.add_argument("--out-root", default=OUT_ROOT)
    a = ap.parse_args(argv)
    os.chdir(ROOT)  # data/ paths are repo-relative
    res, out = Build(a.league, a.date, a.out_root, a.game)
    print(f"cheat sheet {a.league} {a.date}: {len(res['games'])} games, {res['runtime_s']}s, errors {len(res['errors'])}")
    print(os.path.join(out, "slate.md"))


if __name__ == "__main__":
    Main()
