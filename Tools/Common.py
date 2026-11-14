"""Shared fetchers and price math for the decision tools (read-only, free public sources).

Every fetch records its source URL family and an ET timestamp so outputs can cite them.
Sources: ESPN public site API (site.api.espn.com), Kalshi public trade API v2, Open-Meteo
(geocoding + forecast), nflverse schedule already in data/backtest/games_<season>.json.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UA = "football-decision-tools/readonly (Tim's local research)"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
LEAGUE_PATH = {"nfl": "nfl", "cfb": "college-football"}
KALSHI_SERIES = {"nfl": {"game": "KXNFLGAME", "spread": "KXNFLSPREAD", "total": "KXNFLTOTAL"},
                 "cfb": {"game": "KXNCAAFGAME", "spread": "KXNCAAFSPREAD", "total": "KXNCAAFTOTAL"}}
PROP_SERIES = {"KXNFLTD": "anytime_td", "KXNFLREC": "receptions", "KXNFLRECYDS": "receiving_yards",
               "KXNFLRSHYDS": "rushing_yards", "KXNFLPASSYDS": "passing_yards"}
TIM_FEE = 0.02
MARGIN_SD = {"nfl": 13.5, "cfb": 16.0}   # assumption for line-shift approximations (labelled where used)
TOTAL_SD = {"nfl": 10.0, "cfb": 13.0}
_IPV4 = False


def NowEt():
    return datetime.now(ET)


def IsoEt(dt=None):
    return (dt or NowEt()).astimezone(ET).isoformat(timespec="seconds")


def StampEt(dt=None):
    return (dt or NowEt()).astimezone(ET).strftime("%a %b %d %I:%M %p ET")


def ForceIPv4():
    global _IPV4
    if _IPV4:
        return
    import socket
    gai = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **k: gai(host, port, socket.AF_INET if family == 0 else family, *a, **k)
    _IPV4 = True


def GetJson(url, params=None, retries=4, pause=0.0):
    ForceIPv4()
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(retries):
        try:
            if pause:
                time.sleep(pause)
            req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(1.5 * (i + 1))
                continue
            if e.code == 404:
                return None
            raise
        except urllib.error.URLError:
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
                continue
            raise
    return None


# ------------------------------------------------------------------ price math
def AmericanToProb(odds):
    """Implied probability (with vig) from American odds."""
    if odds is None or odds == "":
        return None
    o = float(str(odds).replace("+", "")) if not isinstance(odds, (int, float)) else float(odds)
    if o == 0:
        return None
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def DevigMultiplicative(pa, pb):
    """Multiplicative (proportional) de-vig of a two-way market.
    Returns (fair_a, fair_b, overround, hold) with overround = pa+pb-1 and hold = 1 - 1/(pa+pb)."""
    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return None
    s = pa + pb
    return pa / s, pb / s, s - 1.0, 1.0 - 1.0 / s


def NormCdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def NormPpf(p):
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if NormCdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def ShiftLineProb(p_at_book, book_line, new_line, sd, higher_is_better=True):
    """Normal approximation: given fair P(cover) at the book line, estimate P at another line.
    For a spread taken as 'team +L' (or 'team wins by over -L'), a bigger L helps (higher_is_better=True).
    For an Over at total T, a bigger T hurts: pass higher_is_better=False."""
    if p_at_book is None or book_line is None or new_line is None:
        return None
    if abs(new_line - book_line) < 1e-9:
        return p_at_book
    z = NormPpf(min(max(p_at_book, 1e-6), 1 - 1e-6))
    d = (new_line - book_line) / sd
    return NormCdf(z + d if higher_is_better else z - d)


def ComboFair(probs):
    out = 1.0
    for p in probs:
        if p is None:
            return None
        out *= p
    return out


def KalshiFee(price):
    return 0.07 * price * (1.0 - price)


def SlipMath(fair, price, contracts):
    """Price-vs-fair numbers for a $1-payout ticket bought at `price` per contract."""
    cost = price * contracts
    payout = 1.0 * contracts
    ev = (fair - price) * contracts if fair is not None else None
    fee_tim = TIM_FEE * contracts
    fee_k = math.ceil(KalshiFee(price) * contracts * 100 - 1e-9) / 100.0
    return {"cost": round(cost, 2), "payout_if_win": round(payout, 2), "profit_if_win": round(payout - cost, 2),
            "fair_value": None if fair is None else round(fair * contracts, 2),
            "ev_at_fair_before_fees": None if ev is None else round(ev, 2),
            "fee_tim_est_0.02": round(fee_tim, 2), "ev_after_tim_fee": None if ev is None else round(ev - fee_tim, 2),
            "fee_kalshi_taker": round(fee_k, 2), "ev_after_kalshi_fee": None if ev is None else round(ev - fee_k, 2),
            "breakeven_prob_tim": round(price + TIM_FEE, 4), "breakeven_prob_kalshi": round(price + KalshiFee(price), 4)}


# ------------------------------------------------------------------ names
def NormTeam(s):
    s = (s or "").lower().replace("&", " and ").replace("st.", "state").replace("'", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    stop = {"university", "the", "of", "u"}
    return " ".join(w for w in s.split() if w not in stop)


def NameMatch(kalshi_name, espn_team):
    k = NormTeam(kalshi_name)
    if not k:
        return 0.0
    best = 0.0
    for key in ("location", "shortDisplayName", "displayName", "name", "abbreviation"):
        v = NormTeam(espn_team.get(key))
        if not v:
            continue
        if k == v:
            return 1.0
        kt, vt = set(k.split()), set(v.split())
        j = len(kt & vt) / len(kt | vt)
        best = max(best, j)
    return best


# ------------------------------------------------------------------ ESPN
def EspnScoreboard(league, date):
    params = {"dates": date.replace("-", ""), "limit": 400}
    if league == "cfb":
        params["groups"] = 80
    body = GetJson(f"{ESPN}/{LEAGUE_PATH[league]}/scoreboard", params) or {}
    return body.get("events") or [], IsoEt()


def EspnSummary(league, event_id):
    return GetJson(f"{ESPN}/{LEAGUE_PATH[league]}/summary", {"event": event_id}) or {}, IsoEt()


def Competitors(event):
    c = event["competitions"][0]
    home = next(x for x in c["competitors"] if x["homeAway"] == "home")
    away = next(x for x in c["competitors"] if x["homeAway"] == "away")
    return c, home, away


def EspnOdds(comp):
    """Book odds ESPN shows (first provider): moneyline/spread/total open and current."""
    odds = (comp.get("odds") or [None])[0]
    if not odds:
        return None
    def g(*path):
        x = odds
        for p in path:
            x = (x or {}).get(p)
        return x
    def num(s):
        if s in (None, "", "OFF", "EVEN"):
            return 100.0 if s == "EVEN" else None
        try:
            return float(str(s).replace("o", "").replace("u", "").replace("+", ""))
        except ValueError:
            return None
    return {
        "provider": (odds.get("provider") or {}).get("name"),
        "details": odds.get("details"),
        "ml_home": {"open": num(g("moneyline", "home", "open", "odds")), "cur": num(g("moneyline", "home", "close", "odds"))},
        "ml_away": {"open": num(g("moneyline", "away", "open", "odds")), "cur": num(g("moneyline", "away", "close", "odds"))},
        "spread_home": {"open_line": num(g("pointSpread", "home", "open", "line")), "open_odds": num(g("pointSpread", "home", "open", "odds")),
                        "line": num(g("pointSpread", "home", "close", "line")), "odds": num(g("pointSpread", "home", "close", "odds"))},
        "spread_away": {"open_line": num(g("pointSpread", "away", "open", "line")), "open_odds": num(g("pointSpread", "away", "open", "odds")),
                        "line": num(g("pointSpread", "away", "close", "line")), "odds": num(g("pointSpread", "away", "close", "odds"))},
        "total": {"open_line": num(g("total", "over", "open", "line")), "line": num(g("total", "over", "close", "line")) or odds.get("overUnder"),
                  "over_odds": num(g("total", "over", "close", "odds")), "under_odds": num(g("total", "under", "close", "odds")),
                  "open_over_odds": num(g("total", "over", "open", "odds")), "open_under_odds": num(g("total", "under", "open", "odds"))},
    }


def FairFromOdds(o):
    """Multiplicative no-vig fair probabilities for ML / spread / total from EspnOdds()."""
    if not o:
        return {}
    out = {}
    d = DevigMultiplicative(AmericanToProb(o["ml_home"]["cur"]), AmericanToProb(o["ml_away"]["cur"]))
    if d:
        out["ml"] = {"home": d[0], "away": d[1], "overround": d[2], "hold": d[3]}
    d0 = DevigMultiplicative(AmericanToProb(o["ml_home"]["open"]), AmericanToProb(o["ml_away"]["open"]))
    if d0:
        out["ml_open"] = {"home": d0[0], "away": d0[1]}
    d = DevigMultiplicative(AmericanToProb(o["spread_home"]["odds"]), AmericanToProb(o["spread_away"]["odds"]))
    if d and o["spread_home"]["line"] is not None:
        out["spread"] = {"home_line": o["spread_home"]["line"], "home": d[0], "away": d[1], "overround": d[2], "hold": d[3]}
    d = DevigMultiplicative(AmericanToProb(o["total"]["over_odds"]), AmericanToProb(o["total"]["under_odds"]))
    if d and o["total"]["line"] is not None:
        out["total"] = {"line": o["total"]["line"], "over": d[0], "under": d[1], "overround": d[2], "hold": d[3]}
    return out


def Situation(comp):
    s = comp.get("situation") or {}
    prob = ((s.get("lastPlay") or {}).get("probability")) or {}
    return {"possession": s.get("possession"), "down_distance": s.get("downDistanceText") or s.get("shortDownDistanceText"),
            "home_win_pct": prob.get("homeWinPercentage"), "last_play": (s.get("lastPlay") or {}).get("text")}


# ------------------------------------------------------------------ Kalshi
def KalshiOpen(series, pause=0.1):
    out, cursor = [], ""
    for _ in range(30):
        p = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        body = GetJson(f"{KALSHI}/markets", p, pause=pause) or {}
        out += body.get("markets") or []
        cursor = body.get("cursor") or ""
        if not cursor:
            break
    return out, IsoEt()


def Dollars(m, key):
    v = m.get(f"{key}_dollars")
    if v is not None:
        return float(v)
    v = m.get(key)
    return None if v is None else float(v) / 100.0


def Quote(m):
    b, a = Dollars(m, "yes_bid"), Dollars(m, "yes_ask")
    mid = (a + b) / 2 if (a is not None and b is not None and 0 < b < a < 1) else None
    return {"ticker": m.get("ticker"), "bid": b, "ask": a, "mid": mid, "width": None if mid is None else round(a - b, 4),
            "title": m.get("yes_sub_title") or m.get("title"), "strike": m.get("floor_strike")}


def EventStamp(start_utc):
    """Kalshi event tickers carry the ET calendar date, e.g. 26NOV15."""
    return start_utc.astimezone(ET).strftime("%y%b%d").upper()


def MatchKalshiGame(game_markets, home_team, away_team, start_utc):
    """Return {'event_suffix', 'home': Quote, 'away': Quote} by matching yes_sub_title names on the ET date."""
    stamp = EventStamp(start_utc)
    by_ev = {}
    for m in game_markets:
        ev = m.get("event_ticker") or ""
        if "-" in ev and ev.split("-", 1)[1].startswith(stamp):
            by_ev.setdefault(ev, []).append(m)
    best, best_score = None, 0.0
    for ev, ms in by_ev.items():
        if len(ms) < 2:
            continue
        sh = max(ms, key=lambda m: NameMatch(m.get("yes_sub_title"), home_team))
        sa = max(ms, key=lambda m: NameMatch(m.get("yes_sub_title"), away_team))
        if sh is sa:
            continue
        score = NameMatch(sh.get("yes_sub_title"), home_team) + NameMatch(sa.get("yes_sub_title"), away_team)
        if score > best_score:
            best, best_score = (ev, sh, sa), score
    if not best or best_score < 1.0:
        return None
    ev, sh, sa = best
    return {"event_ticker": ev, "event_suffix": ev.split("-", 1)[1], "home": Quote(sh), "away": Quote(sa), "match_score": round(best_score, 2)}


def KalshiFirstPrice(series, ticker, open_time, end_utc, max_width=0.05):
    """First hourly candle after listing with a tight (<= max_width) two-sided quote: yes ask / bid / mid."""
    try:
        st = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
    except Exception:
        return None
    body = GetJson(f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks",
                   {"start_ts": int(st.timestamp()), "end_ts": int(min(end_utc, datetime.now(timezone.utc)).timestamp()),
                    "period_interval": 60})
    for c in (body or {}).get("candlesticks") or []:
        a = (c.get("yes_ask") or {}).get("close_dollars")
        b = (c.get("yes_bid") or {}).get("close_dollars")
        # skip the wide placeholder quotes Kalshi shows right after listing: first hour with a tight spread
        if a is not None and b is not None and 0 < float(b) < float(a) < 1 and float(a) - float(b) <= max_width + 1e-9:
            return {"ask": float(a), "bid": float(b), "mid": (float(a) + float(b)) / 2,
                    "at": datetime.fromtimestamp(c["end_period_ts"], timezone.utc).astimezone(ET).isoformat(timespec="minutes")}
    return None


# ------------------------------------------------------------------ weather
_GEO_CACHE = os.path.join("data", "tools", "geocode.json")


def Geocode(city, state, country=None):
    os.makedirs(os.path.dirname(_GEO_CACHE), exist_ok=True)
    cache = json.load(open(_GEO_CACHE, encoding="utf-8")) if os.path.exists(_GEO_CACHE) else {}
    key = f"{city}|{state}|{country or 'US'}"
    if key in cache:
        return cache[key]
    q = {"name": city, "count": 10}
    if not country or country in ("USA", "US", "United States"):
        q["countryCode"] = "US"
    body = GetJson("https://geocoding-api.open-meteo.com/v1/search", q) or {}
    res = body.get("results") or []
    pick = next((r for r in res if state and (r.get("admin1_code") == state or (r.get("admin1") or "").lower().startswith(state.lower()))), res[0] if res else None)
    val = None if not pick else {"lat": pick["latitude"], "lon": pick["longitude"], "name": f"{pick.get('name')}, {pick.get('admin1')}"}
    cache[key] = val
    json.dump(cache, open(_GEO_CACHE, "w", encoding="utf-8"), indent=1)
    return val


STATE_ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
              "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
              "Kentucky": "KY", "Louisiana": "LA", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
              "Mississippi": "MS", "Missouri": "MO", "Nebraska": "NE", "Nevada": "NV", "New Jersey": "NJ", "New Mexico": "NM",
              "New York": "NY", "North Carolina": "NC", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
              "South Carolina": "SC", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Virginia": "VA", "Washington": "WA",
              "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Hawaii": "HI", "Delaware": "DE", "Montana": "MT",
              "North Dakota": "ND", "South Dakota": "SD", "Vermont": "VT", "New Hampshire": "NH", "Maine": "ME", "Rhode Island": "RI",
              "Alaska": "AK", "District of Columbia": "DC"}


def Weather(venue, start_utc):
    """Open-Meteo hourly forecast at kickoff. Domes/indoor venues are marked, no forecast."""
    if not venue:
        return {"text": "venue unknown - weather not available", "source": None}
    addr = venue.get("address") or {}
    name = venue.get("fullName")
    if venue.get("indoor"):
        return {"text": f"{name}: indoor/dome (ESPN venue.indoor=true) - weather not a factor", "indoor": True, "source": "ESPN venue"}
    city, state = addr.get("city"), addr.get("state")
    if not city:
        return {"text": f"{name}: no city on ESPN venue - weather not available", "source": None}
    try:
        st = state if (state or "").__len__() == 2 else None
        geo = Geocode(city, st or state, addr.get("country"))
        if not geo:
            return {"text": f"{name} ({city}): geocode failed - weather not available", "source": None}
        day = start_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")
        body = GetJson("https://api.open-meteo.com/v1/forecast", {
            "latitude": geo["lat"], "longitude": geo["lon"], "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC", "start_date": day, "end_date": day}) or {}
        h = body.get("hourly") or {}
        times = h.get("time") or []
        tgt = start_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")
        if tgt not in times:
            return {"text": f"{name}: forecast hour not available (Open-Meteo)", "source": "Open-Meteo"}
        i = times.index(tgt)
        temp, pop, wind, gust = (h["temperature_2m"][i], h["precipitation_probability"][i], h["wind_speed_10m"][i], h["wind_gusts_10m"][i])
        where = ", ".join(x for x in (city, state or addr.get("country")) if x)
        return {"text": f"{name} ({where}): {temp:.0f}F, wind {wind:.0f} mph (gusts {gust:.0f}), precip chance {pop}% at kickoff",
                "temp_f": temp, "wind_mph": wind, "gust_mph": gust, "precip_pct": pop, "indoor": False,
                "source": "Open-Meteo forecast", "fetched_at": IsoEt()}
    except Exception as exc:
        return {"text": f"{name}: weather fetch failed ({type(exc).__name__})", "source": "Open-Meteo"}


# ------------------------------------------------------------------ ESPN context
def Injuries(summary, team_id):
    """ESPN summary injuries for one team (NFL). Returns None when ESPN has no injuries block (e.g. CFB)."""
    inj = summary.get("injuries")
    if not inj:
        return None
    out = []
    for t in inj:
        if str((t.get("team") or {}).get("id")) != str(team_id):
            continue
        for x in t.get("injuries") or []:
            a = x.get("athlete") or {}
            det = x.get("details") or {}
            out.append({"player": a.get("displayName"), "pos": (a.get("position") or {}).get("abbreviation"),
                        "status": x.get("status"), "type": det.get("type"), "detail": det.get("detail"),
                        "side": det.get("side"), "return": det.get("returnDate"), "updated": x.get("date")})
    return out


def LeagueInjuries(league):
    """ESPN league injuries feed (NFL): {espn_team_id: [rows]}. ESPN caps each team at 25 rows (incl. IR)."""
    body = GetJson(f"{ESPN}/{LEAGUE_PATH[league]}/injuries") or {}
    out = {}
    for t in body.get("injuries") or []:
        rows = []
        for x in t.get("injuries") or []:
            a = x.get("athlete") or {}
            det = x.get("details") or {}
            rows.append({"player": a.get("displayName"), "pos": (a.get("position") or {}).get("abbreviation"),
                         "status": x.get("status"), "type": det.get("type"), "detail": det.get("detail"),
                         "side": det.get("side"), "return": det.get("returnDate"), "updated": x.get("date"),
                         "comment": x.get("shortComment")})
        out[str(t.get("id"))] = rows
    return out, IsoEt()


def LastFive(summary, team_id):
    for t in summary.get("lastFiveGames") or []:
        if str((t.get("team") or {}).get("id")) == str(team_id):
            evs = t.get("events") or []
            rows = []
            for e in evs:
                home = str(e.get("homeTeamId")) == str(team_id)
                pf = e.get("homeTeamScore") if home else e.get("awayTeamScore")
                pa = e.get("awayTeamScore") if home else e.get("homeTeamScore")
                rows.append({"date": (e.get("gameDate") or "")[:10], "opp": (e.get("opponent") or {}).get("abbreviation"),
                             "at_vs": e.get("atVs"), "result": e.get("gameResult"), "score": e.get("score"),
                             "pf": int(pf) if pf not in (None, "") else None, "pa": int(pa) if pa not in (None, "") else None})
            return rows
    return []


def Predictor(summary):
    p = summary.get("predictor") or {}
    try:
        return {"home": float(p["homeTeam"]["gameProjection"]) / 100.0, "away": float(p["awayTeam"]["gameProjection"]) / 100.0}
    except (KeyError, TypeError, ValueError):
        return None
