"""
AssessLive: one-shot, read-only NFL quote and game-state report.

    python -m Backtest.AssessLive
    python -m Backtest.AssessLive --date 2026-11-08
    python -m Backtest.AssessLive --date 2026-11-08 --matchup TB@CHI

Writes data/backtest/live/assess_<date>_<utc>.json and .md. Those paths are
gitignored with the rest of data/. Nothing here places an order, stores a
credential, or refreshes itself on a timer.

Public reads used, all from the provider's own docs (no auth):

  Kalshi Trade API  https://external-api.kalshi.com/trade-api/v2
    GET /milestones, /live_data/batch, /markets, /markets/{ticker}/orderbook,
    /events/{event_ticker}, /series/{series_ticker}, /structured_targets/{id},
    /series/fee_changes, /events/fee_changes
    Quick start: https://docs.kalshi.com/getting_started/quick_start_market_data
  The series payload's contract_terms_url (football game winner PDF).

  Polymarket Gamma  https://gamma-api.polymarket.com
    GET /sports, GET /markets?slug=
    Docs: https://docs.polymarket.com/api-reference/markets/list-markets
  Polymarket CLOB   https://clob.polymarket.com
    GET /book?token_id=  and  GET /clob-markets/{condition_id}
    Docs: https://docs.polymarket.com/market-data/prices-order-books

Game state is the Kalshi live-data payload (details.status, quarter, clock,
away_points, home_points, last_updated_ts). Polymarket's event score/period/elapsed
are reported beside it. Polymarket's sports websocket is not held open; this
command is one shot. NFL.com's live feed is not called: it is not a public
unauthenticated API.

The pre-game margin model is not loaded. See Backtest.LiveReadiness.
"""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from Backtest.LiveReadiness import (
    ET, AsFloat, BookSide, Canon, EmptyReport, EtText, Iso, ParseTime, PREGAME_MODEL_NOTE,
    SettlementFromRules, AssessGame, Thresholds,
)

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
OUT_DIR = os.path.join("data", "backtest", "live")
UA = "football-AssessLive/readonly"
FEE_SCHEDULE_URL = "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
# Gamma game slugs observed from GET /events?tag_id=<nfl> look like
# nfl-{away}-{home}-{UTC date of kickoff}. This is a lookup candidate for the
# documented slug filter, not a guessed price. A 404 is recorded as a miss.
SLUG_ABBREV = {
    "LAR": "la", "LA": "la", "JAC": "jax", "JAX": "jax",
    "WSH": "was", "WAS": "was", "NOLA": "no", "NO": "no", "GNB": "gb", "GB": "gb",
}
NICKNAME = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF",
    "panthers": "CAR", "bears": "CHI", "bengals": "CIN", "browns": "CLE",
    "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC",
    "raiders": "LV", "chargers": "LAC", "rams": "LA", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "49ers": "SF",
    "seahawks": "SEA", "buccaneers": "TB", "titans": "TEN", "commanders": "WAS",
}


class FetchError(Exception):
    pass


class UrlFetcher:
    """Sequential public GETs. A 429 stops further calls on that host."""

    def __init__(self):
        self.calls = 0
        self.blocked_hosts = set()

    def get_json(self, url):
        env = self._open(url)
        try:
            env["body"] = json.loads(env.pop("raw").decode())
        except json.JSONDecodeError as exc:
            raise FetchError(f"non-JSON body from {url}: {exc}") from exc
        return env

    def get_bytes(self, url):
        env = self._open(url)
        env["body"] = env.pop("raw")
        return env

    def _open(self, url):
        host = urllib.parse.urlparse(url).netloc
        if host in self.blocked_hosts:
            raise FetchError(f"skipped {url} after HTTP 429 from {host}")
        self.calls += 1
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = resp.read()
                return {
                    "url": url, "status": resp.status,
                    "http_date": resp.headers.get("Date"),
                    "cache_control": resp.headers.get("Cache-Control"),
                    "observed_at": datetime.now(timezone.utc),
                    "raw": raw,
                }
        except urllib.error.HTTPError as exc:
            detail = exc.read(300).decode("utf-8", "replace")
            if exc.code == 429:
                self.blocked_hosts.add(host)
            raise FetchError(f"HTTP {exc.code} {urllib.parse.urlparse(url).netloc}{urllib.parse.urlparse(url).path}") from exc
        except urllib.error.URLError as exc:
            raise FetchError(f"request failed {url}: {exc.reason}") from exc


def _q(url, params):
    return url + "?" + urllib.parse.urlencode(params, doseq=True)


def SlugPiece(abbrev):
    return SLUG_ABBREV.get((abbrev or "").upper(), (abbrev or "").lower())


def WinnerMarket(market):
    text = " ".join(str(market.get(k) or "") for k in ("title", "yes_sub_title", "rules_primary")).lower()
    if any(word in text for word in ("spread", "total", "touchdown", "yards", "player", "points")):
        # "points" appears in cancellation prose less often than in totals. The
        # game-winner rules say "wins the ... game", not a points line.
        if "wins" in text and "game" in text and "spread" not in text and "touchdown" not in text and "yards" not in text:
            return True
        return False
    return "wins" in text and "game" in text


def KalshiAsksFromNoBids(no_bids):
    """A YES ask is the reciprocal of a NO bid (Kalshi orderbook docs)."""
    rows = [{"price": round(1.0 - level["price"], 4), "size": level["size"]} for level in no_bids.get("top") or []]
    rows.sort(key=lambda r: r["price"])
    return {"best": rows[0] if rows else None, "top": rows[:5], "level_count": no_bids.get("level_count") or 0,
            "size_sum": no_bids.get("size_sum") or 0.0}


def FeeRecord(series, fee_changes, event_changes, pdf_status):
    """Labels from the series payload. Dollar fee stays unknown unless the
    official fee-schedule PDF was fetched and parsed, which this tool does not
    invent from memory when the fetch fails."""
    fee_type = (series or {}).get("fee_type")
    multiplier = (series or {}).get("fee_multiplier")
    known = False
    return {
        "status": "known" if known else "unknown",
        "taker_fee_per_contract": None,
        "fee_type": fee_type,
        "fee_multiplier": multiplier,
        "fee_type_doc": "https://docs.kalshi.com/api-reference/market/get-series",
        "fee_schedule_url": FEE_SCHEDULE_URL,
        "fee_schedule_fetch": pdf_status,
        "series_fee_changes": fee_changes,
        "event_fee_changes": event_changes,
        "why": (
            "Series fee_type and fee_multiplier are from GET /series. "
            "Docs map quadratic_with_maker_fees to the General Trading Fees table plus the Maker Fees section "
            "of the fee-schedule PDF. This run did not obtain a numeric formula from that PDF "
            f"({pdf_status}), so the taker fee in dollars is unknown and EV stays blank."
        ),
    }


def LoadKalshi(fetcher, date, now, matchup):
    """Return (games_by_key, notes, errors). games_by_key values are AssessGame inputs."""
    notes, errors = [], []
    start_et = datetime.fromisoformat(date).replace(tzinfo=ET)
    minimum = (start_et - timedelta(hours=12)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        probe = fetcher.get_json(_q(KALSHI + "/milestones", {
            "limit": 1, "category": "Sports", "competition": "Pro Football", "minimum_start_date": minimum}))
        notes.append({"competition_filter": "Pro Football", "returned": len((probe["body"] or {}).get("milestones") or []),
                      "url": probe["url"]})
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "milestones_competition", "error": str(exc)})

    milestones, cursor, pages = [], "", 0
    try:
        while pages < 8:
            params = {"limit": 200, "type": "football_game", "category": "Sports", "minimum_start_date": minimum}
            if cursor:
                params["cursor"] = cursor
            env = fetcher.get_json(_q(KALSHI + "/milestones", params))
            body = env["body"]
            batch = body.get("milestones") or []
            milestones.extend(batch)
            cursor = body.get("cursor") or ""
            pages += 1
            if not cursor or not batch:
                break
            last = ParseTime(batch[-1].get("start_date"))
            if last and last > start_et.astimezone(timezone.utc) + timedelta(days=2):
                break
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "milestones", "error": str(exc)})
        return {}, notes, errors

    wanted = []
    for ms in milestones:
        details = ms.get("details") or {}
        if str(details.get("league") or "").upper() != "NFL":
            continue
        start = ParseTime(ms.get("start_date"))
        if start is None or start.astimezone(ET).date().isoformat() != date:
            continue
        title_away, title_home = _title_teams(ms.get("title") or "")
        if matchup and title_away and title_home and not _matchup_ok(matchup, title_away, title_home):
            continue
        wanted.append(ms)
    notes.append({"milestones_scanned": len(milestones), "pages": pages, "nfl_on_date": len(wanted)})

    targets = {}
    games = {}
    series_cache = {}
    pdf_text_cache = {}
    pdf_status = "not_requested"

    ids = [ms["id"] for ms in wanted if ms.get("id")]
    live_by_id = {}
    if ids:
        try:
            params = [("milestone_ids", i) for i in ids]
            env = fetcher.get_json(_q(KALSHI + "/live_data/batch", params))
            for row in (env["body"] or {}).get("live_datas") or []:
                live_by_id[row.get("milestone_id")] = (row, env["observed_at"], env["url"])
        except FetchError as exc:
            errors.append({"source": "kalshi", "op": "live_data", "error": str(exc)})

    for ms in wanted:
        details = ms.get("details") or {}
        event = details.get("main_game_event_ticker") or ((ms.get("primary_event_tickers") or [None])[0])
        if not event:
            errors.append({"source": "kalshi", "op": "event_ticker", "milestone": ms.get("id"), "error": "no event ticker"})
            continue
        try:
            markets_env = fetcher.get_json(_q(KALSHI + "/markets", {"event_ticker": event, "limit": 20}))
        except FetchError as exc:
            errors.append({"source": "kalshi", "op": "markets", "event": event, "error": str(exc)})
            continue
        markets = markets_env["body"].get("markets") or []
        series_ticker = event.split("-")[0]
        if series_ticker not in series_cache:
            series_cache[series_ticker] = _load_series(fetcher, series_ticker, pdf_text_cache, errors)
            pdf_status = series_cache[series_ticker].get("pdf_status") or pdf_status
        series_info = series_cache[series_ticker]
        away_id, home_id = details.get("away_team_id"), details.get("home_team_id")
        for tid in (away_id, home_id):
            _ensure_target(fetcher, targets, tid, errors)
        away = (targets.get(away_id) or {}).get("abbreviation")
        home = (targets.get(home_id) or {}).get("abbreviation")
        if not away or not home:
            title_away, title_home = _title_teams(ms.get("title") or "")
            away = away or title_away
            home = home or title_home
        start = ParseTime(ms.get("start_date"))
        live_row = live_by_id.get(ms.get("id"))
        state = None
        if live_row:
            state = _kalshi_state(live_row[0], away, home, targets, live_row[1], ms.get("id"))
        quotes = []
        for market in markets:
            team_id = ((market.get("custom_strike") or {}).get("football_team"))
            _ensure_target(fetcher, targets, team_id, errors)
            side = (targets.get(team_id) or {}).get("abbreviation")
            book_body, book_obs = None, markets_env["observed_at"]
            try:
                book_env = fetcher.get_json(f"{KALSHI}/markets/{urllib.parse.quote(market['ticker'])}/orderbook?depth=10")
                book_body, book_obs = book_env["body"], book_env["observed_at"]
            except FetchError as exc:
                errors.append({"source": "kalshi", "op": "orderbook", "market": market.get("ticker"), "error": str(exc)})
            quote = _kalshi_quote(market, book_body, side, away, home, event, book_obs, series_info)
            fee = dict(series_info.get("fee") or {})
            fee["event_fee_changes"] = _event_fee_changes(fetcher, event, errors)
            quote["fee"] = fee
            quotes.append(quote)
        key = (Canon(away), Canon(home), Iso(start))
        games[key] = {
            "away": away, "home": home, "start_utc": Iso(start),
            "quotes": quotes, "states": [state] if state else [],
            "fee": series_info.get("fee"), "settlement": series_info.get("settlement"),
            "season": details.get("season"), "milestone_id": ms.get("id"),
            "milestone_status_lagging": details.get("status"),
            "event_id": event,
        }
    notes.append({"fee_schedule_fetch": pdf_status})
    return games, notes, errors


def _load_series(fetcher, ticker, pdf_cache, errors):
    info = {"pdf_status": "not_requested", "fee": None, "settlement": None}
    try:
        env = fetcher.get_json(f"{KALSHI}/series/{urllib.parse.quote(ticker)}")
        series = (env["body"] or {}).get("series") or {}
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "series", "series": ticker, "error": str(exc)})
        info["fee"] = FeeRecord(None, None, None, "series_fetch_failed")
        info["settlement"] = SettlementFromRules("")
        return info
    changes, event_note = None, "not_checked_per_event"
    try:
        ch = fetcher.get_json(_q(KALSHI + "/series/fee_changes", {"series_ticker": ticker}))
        changes = ch["body"].get("series_fee_change_arr")
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "series_fee_changes", "series": ticker, "error": str(exc)})
    terms_url = series.get("contract_terms_url")
    text = ""
    if terms_url:
        if terms_url in pdf_cache:
            text, info["pdf_status"] = pdf_cache[terms_url]
        else:
            try:
                raw = fetcher.get_bytes(terms_url)
                text = _pdf_text(raw["body"])
                info["pdf_status"] = f"fetched {len(raw['body'])} bytes"
            except FetchError as exc:
                info["pdf_status"] = str(exc)
                errors.append({"source": "kalshi", "op": "contract_terms", "error": str(exc)})
            except Exception as exc:
                info["pdf_status"] = f"pdf parse failed: {exc}"
            pdf_cache[terms_url] = (text, info["pdf_status"])
    # Fee schedule PDF is a separate document. Try once per process via the cache key.
    if FEE_SCHEDULE_URL not in pdf_cache:
        try:
            raw = fetcher.get_bytes(FEE_SCHEDULE_URL)
            schedule = _pdf_text(raw["body"])
            pdf_cache[FEE_SCHEDULE_URL] = (schedule, f"fetched {len(raw['body'])} bytes")
        except FetchError as exc:
            pdf_cache[FEE_SCHEDULE_URL] = ("", str(exc))
        except Exception as exc:
            pdf_cache[FEE_SCHEDULE_URL] = ("", f"pdf parse failed: {exc}")
    schedule_text, schedule_status = pdf_cache[FEE_SCHEDULE_URL]
    info["pdf_status"] = f"contract_terms={info['pdf_status']}; fee_schedule={schedule_status}"
    info["fee"] = FeeRecord(series, changes, event_note, info["pdf_status"] if not schedule_text else "fetched_but_not_applied")
    if schedule_text and not _formula_verified(schedule_text):
        info["fee"]["fee_schedule_fetch"] = "fetched; numeric formula not applied (rounding unit is not hard-coded)"
        info["fee"]["status"] = "unknown"
        info["fee"]["taker_fee_per_contract"] = None
    elif schedule_text and _formula_verified(schedule_text):
        # Still unknown per contract until a count and a rounding implementation
        # that matches the PDF are both present. Recording the excerpt is enough.
        info["fee"]["formula_excerpt"] = _excerpt_formula(schedule_text)
        info["fee"]["status"] = "unknown"
        info["fee"]["taker_fee_per_contract"] = None
        info["fee"]["why"] += " The PDF was fetched and an excerpt is stored, but this tool does not turn that prose into a per-contract dollar fee."
    rules = text or ""
    info["settlement"] = SettlementFromRules(rules)
    info["settlement"]["contract_terms_url"] = terms_url
    info["settlement"]["contract_url"] = series.get("contract_url")
    info["settlement"]["settlement_sources"] = series.get("settlement_sources")
    info["series"] = {"ticker": series.get("ticker"), "title": series.get("title"),
                      "fee_type": series.get("fee_type"), "fee_multiplier": series.get("fee_multiplier")}
    return info


def _formula_verified(text):
    folded = " ".join(text.split()).lower()
    return "0.07" in folded and "fee" in folded


def _excerpt_formula(text):
    folded = " ".join(text.split())
    idx = folded.lower().find("0.07")
    if idx < 0:
        return None
    return folded[max(0, idx - 80):idx + 120]


def _pdf_text(data):
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _ensure_target(fetcher, cache, tid, errors):
    if not tid or tid in cache:
        return
    try:
        env = fetcher.get_json(f"{KALSHI}/structured_targets/{urllib.parse.quote(tid)}")
        target = (env["body"] or {}).get("structured_target") or {}
        details = target.get("details") or {}
        cache[tid] = {"abbreviation": details.get("abbreviation"), "name": details.get("team_name"),
                      "market": details.get("market"), "id": tid}
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "structured_target", "id": tid, "error": str(exc)})
        cache[tid] = {}


def _event_fee_changes(fetcher, event, errors):
    try:
        env = fetcher.get_json(_q(KALSHI + "/events/fee_changes", {"event_ticker": event, "limit": 20}))
        return (env["body"] or {}).get("event_fee_changes")
    except FetchError as exc:
        errors.append({"source": "kalshi", "op": "event_fee_changes", "event": event, "error": str(exc)})
        return None


def _title_teams(title):
    """'TB Buccaneers at CHI Bears' -> (TB, CHI). Used only when a structured target is missing."""
    if " at " not in title:
        return None, None
    left, right = title.split(" at ", 1)
    def abbr(side):
        token = side.strip().split(" ")[0]
        if 2 <= len(token) <= 4 and token.isupper():
            return token
        return None
    return abbr(left), abbr(right)


def _matchup_ok(matchup, away, home):
    want_away, want_home = _parse_matchup(matchup)
    return Canon(away) == Canon(want_away) and Canon(home) == Canon(want_home)


def _parse_matchup(matchup):
    text = matchup.replace(" ", "")
    if "@" not in text:
        raise SystemExit("--matchup must look like TB@CHI")
    away, home = text.split("@", 1)
    return away, home


def _kalshi_state(row, away, home, targets, observed, milestone_id):
    details = row.get("details") or {}
    situation = details.get("situation") or {}
    poss = (targets.get(situation.get("possession_team_id")) or {}).get("abbreviation")
    delayed = details.get("live_data_delayed") if "live_data_delayed" in details else None
    ts_field = "details.last_updated_ts" if "last_updated_ts" in details else None
    return {
        "venue": "kalshi",
        "url": f"{KALSHI}/live_data/milestone/{milestone_id}",
        "event_id": milestone_id,
        "away": away, "home": home,
        "status": details.get("status"),
        "period": details.get("quarter"),
        "clock": details.get("clock"),
        "clock_note": "live_data.details.clock as returned. Docs do not say whether the string is time remaining or elapsed.",
        "away_score": details.get("away_points"),
        "home_score": details.get("home_points"),
        "source_timestamp": ParseTime(details.get("last_updated_ts")) if ts_field else None,
        "source_timestamp_field": ts_field,
        "delayed": delayed,
        "observed_at": observed,
        "possession": poss,
        "down": situation.get("down"),
        "distance": situation.get("yfd"),
        "yardline": situation.get("yardline"),
        "timeouts_away": details.get("away_timeouts_remaining"),
        "timeouts_home": details.get("home_timeouts_remaining"),
        "under_review": details.get("is_under_review"),
        "provider": details.get("provider"),
        "last_play": (details.get("last_play") or {}).get("description"),
        "winner": details.get("winner") or None,
    }


def _kalshi_quote(market, book_body, side, away, home, event, observed, series_info):
    book = (book_body or {}).get("orderbook_fp") or {}
    yes_bids = BookSide(book.get("yes_dollars"), "bid")
    no_bids = BookSide(book.get("no_dollars"), "bid")
    rules = " ".join(filter(None, [market.get("rules_primary"), market.get("rules_secondary")]))
    settlement = series_info.get("settlement") or SettlementFromRules(rules)
    if market.get("rules_primary") or market.get("rules_secondary"):
        merged = SettlementFromRules(" ".join(filter(None, [
            market.get("rules_primary"), market.get("rules_secondary"),
            (series_info.get("settlement") or {}).get("overtime") and "overtime",
        ])))
        # Keep the contract-terms settlement, which has the OT sentence, and
        # always attach the instance rules verbatim.
        settlement = dict(series_info.get("settlement") or merged)
        settlement["rules_primary"] = market.get("rules_primary")
        settlement["rules_secondary"] = market.get("rules_secondary")
    return {
        "venue": "kalshi",
        "event_id": event,
        "event_url": f"{KALSHI}/events/{event}",
        "market_id": market.get("ticker"),
        "market_url": f"{KALSHI}/markets/{market.get('ticker')}",
        "orderbook_url": f"{KALSHI}/markets/{market.get('ticker')}/orderbook",
        "series_ui_url": "https://kalshi.com/markets/" + event.split("-")[0].lower(),
        "series_ui_url_note": "Documented series UI pattern is https://kalshi.com/markets/{series}. This is not an event-specific link; the event URL above is the API resource.",
        "outcome_kind": "moneyline_game_winner" if WinnerMarket(market) else "other",
        "side_team": side,
        "side_name": market.get("yes_sub_title"),
        "away": away, "home": home,
        "status": market.get("status"),
        "book": {"bids": yes_bids, "asks": KalshiAsksFromNoBids(no_bids), "depth_requested": 10},
        "summary_bid": AsFloat(market.get("yes_bid_dollars")),
        "summary_ask": AsFloat(market.get("yes_ask_dollars")),
        "summary_bid_size": AsFloat(market.get("yes_bid_size_fp")),
        "summary_ask_size": AsFloat(market.get("yes_ask_size_fp")),
        "last_trade": AsFloat(market.get("last_price_dollars")),
        "last_trade_role": "indicative last YES trade (last_price_dollars), not an executable quote",
        "volume": market.get("volume_fp"),
        "open_interest": market.get("open_interest_fp"),
        "notional_dollars": market.get("notional_value_dollars"),
        "source_timestamp": None,
        "source_timestamp_field": None,
        "metadata_updated_at": market.get("updated_time"),
        "metadata_updated_at_role": "Market.updated_time is documented as the last non-trading metadata update, not a quote time. It is not used for freshness.",
        "observed_at": observed,
        "http_note": "Orderbook and market payloads do not include a quote timestamp. observed_at is when this process received the body.",
        "fee": series_info.get("fee"),
        "settlement": settlement,
    }


def LoadPolymarket(fetcher, games, errors):
    """Attach Polymarket moneyline quotes for games already keyed by Kalshi.

    Slug lookup only. No prices are invented when the slug 404s.
    """
    ordering = None
    try:
        sports = fetcher.get_json(GAMMA + "/sports")
        for row in sports["body"] or []:
            if isinstance(row, dict) and row.get("sport") == "nfl":
                ordering = row.get("ordering")
                break
    except FetchError as exc:
        errors.append({"source": "polymarket", "op": "sports", "error": str(exc)})
    for key, game in list(games.items()):
        start = ParseTime(game.get("start_utc"))
        if start is None or not game.get("away") or not game.get("home"):
            continue
        slug = f"nfl-{SlugPiece(game['away'])}-{SlugPiece(game['home'])}-{start.date().isoformat()}"
        try:
            env = fetcher.get_json(_q(GAMMA + "/markets", {"slug": slug, "closed": "any"}))
        except FetchError as exc:
            errors.append({"source": "polymarket", "op": "markets", "slug": slug, "error": str(exc)})
            continue
        markets = env["body"] if isinstance(env["body"], list) else []
        moneyline = [m for m in markets if (m.get("sportsMarketType") or "").lower() == "moneyline"]
        if not moneyline:
            errors.append({"source": "polymarket", "op": "markets", "slug": slug,
                           "error": f"no moneyline market (HTTP {env['status']}, {len(markets)} rows)"})
            continue
        market = moneyline[0]
        event = ((market.get("events") or [None])[0]) or {}
        state = _poly_state(event, game["away"], game["home"], ordering, env["observed_at"], slug)
        if state:
            game["states"].append(state)
        quotes, poly_fee = _poly_quotes(fetcher, market, game["away"], game["home"], slug, errors)
        for quote in quotes:
            quote["fee"] = poly_fee
            quote["settlement"] = SettlementFromRules(market.get("description") or "")
            quote["settlement"]["description"] = (market.get("description") or "")[:4000]
        game["quotes"].extend(quotes)
    return games


def _poly_state(event, away, home, ordering, observed, slug):
    if not event:
        return None
    away_score = home_score = None
    score = event.get("score")
    if isinstance(score, str) and "-" in score and ordering in ("away", "home"):
        left, right = score.split("-", 1)
        try:
            a, b = int(left.strip()), int(right.strip())
        except ValueError:
            a = b = None
        if a is not None:
            away_score, home_score = (a, b) if ordering == "away" else (b, a)
    live = event.get("live")
    ended = event.get("ended")
    status = "live" if live else ("ended" if ended else "pregame")
    return {
        "venue": "polymarket",
        "url": f"{GAMMA}/events/slug/{slug}",
        "event_id": event.get("id") or slug,
        "slug": slug,
        "away": away, "home": home,
        "status": status, "live": live, "ended": ended,
        "period": event.get("period"),
        "clock": event.get("elapsed"),
        "clock_note": "REST event field `elapsed` as returned. The sports websocket docs define elapsed as time elapsed in the period; the Gamma event schema lists the field without restating that definition.",
        "away_score": away_score, "home_score": home_score,
        "score_ordering": ordering,
        "source_timestamp": None,
        "source_timestamp_field": None,
        "metadata_updated_at": event.get("updatedAt"),
        "metadata_updated_at_role": "Event.updatedAt is a record timestamp. It is not documented as the score sample time, so it is not used as source_timestamp.",
        "observed_at": observed,
        "delayed": None,
    }


def _poly_quotes(fetcher, market, away, home, slug, errors):
    outcomes = _json_list(market.get("outcomes"))
    condition = market.get("conditionId")
    token_names = {}
    fee = {"status": "unknown", "taker_fee_per_contract": None, "venue": "polymarket",
           "why": "CLOB market info was not read, so the taker fee in dollars is unknown."}
    if condition:
        try:
            info = fetcher.get_json(f"{CLOB}/clob-markets/{condition}")
            body = info["body"] or {}
            for token in body.get("t") or []:
                if isinstance(token, dict) and token.get("t"):
                    token_names[str(token["t"])] = token.get("o")
            fee = {
                "status": "unknown",
                "taker_fee_per_contract": None,
                "venue": "polymarket",
                "maker_base_fee": body.get("mbf"),
                "taker_base_fee": body.get("tbf"),
                "fee_details": body.get("fd"),
                "min_tick": body.get("mts"),
                "min_order_size": body.get("mos"),
                "source_url": info["url"],
                "why": ("GET /clob-markets returns maker/taker base fee fields and a fee-details object. "
                        "The published field list does not state the dollar formula those numbers plug into, "
                        "so the taker fee per contract is unknown."),
            }
        except FetchError as exc:
            errors.append({"source": "polymarket", "op": "clob_market", "condition": condition, "error": str(exc)})
    quotes = []
    # One book per outcome token named by the CLOB payload. Fall back to clobTokenIds order.
    pairs = [(name, tok) for tok, name in token_names.items()] if token_names else []
    if not pairs:
        for idx, tok in enumerate(_json_list(market.get("clobTokenIds"))):
            name = outcomes[idx] if idx < len(outcomes) else None
            pairs.append((name, tok))
    for name, tok in pairs:
        side = NICKNAME.get(str(name or "").strip().lower())
        book_body, observed, book_url = None, None, f"{CLOB}/book?token_id={urllib.parse.quote(str(tok))}"
        try:
            env = fetcher.get_json(book_url)
            book_body, observed = env["body"], env["observed_at"]
        except FetchError as exc:
            errors.append({"source": "polymarket", "op": "book", "token": str(tok)[:12], "error": str(exc)})
            observed = datetime.now(timezone.utc)
        book = book_body or {}
        bids = BookSide(book.get("bids"), "bid")
        asks = BookSide(book.get("asks"), "ask")
        kind = "moneyline_game_winner" if (market.get("sportsMarketType") or "").lower() == "moneyline" else "other"
        status = "active" if market.get("acceptingOrders") and not market.get("closed") else (
            "closed" if market.get("closed") else "suspended")
        quotes.append({
            "venue": "polymarket",
            "event_id": slug,
            "event_url": f"{GAMMA}/events/slug/{slug}",
            "market_id": market.get("conditionId"),
            "market_url": book_url,
            "gamma_market_slug": market.get("slug"),
            "outcome_kind": kind,
            "side_team": side,
            "side_name": name,
            "token_id": str(tok),
            "away": away, "home": home,
            "status": status,
            "book": {"bids": bids, "asks": asks, "hash": book.get("hash"), "min_order_size": book.get("min_order_size"),
                     "tick_size": book.get("tick_size")},
            "summary_bid": AsFloat(market.get("bestBid")) if side and Canon(side) == Canon(away) else None,
            "summary_ask": AsFloat(market.get("bestAsk")) if side and Canon(side) == Canon(away) else None,
            "summary_note": "Gamma bestBid/bestAsk are market-level summary fields. They are compared to the book only for the away outcome, which is the side Gamma's moneyline summary tracked in sampled responses. The executable prices are the CLOB book.",
            "last_trade": AsFloat(book.get("last_trade_price")),
            "last_trade_role": "indicative last_trade_price on the book, not an executable quote",
            "source_timestamp": ParseTime(book.get("timestamp")),
            "source_timestamp_field": "timestamp" if book.get("timestamp") is not None else None,
            "observed_at": observed,
            "liquidity": market.get("liquidityClob") or market.get("liquidityNum") or market.get("liquidity"),
        })
    return quotes, fee


def _json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def MergeAndAssess(games, now):
    assessed = []
    for game in games.values():
        assessed.append(AssessGame(game, now, model=None))
    assessed.sort(key=lambda g: (g.get("start_utc") or "", g.get("matchup") or ""))
    return assessed


def Markdown(meta, games):
    lines = [
        f"# NFL live-odds readiness {meta['date_et']} ET",
        "",
        "Read-only. No order is sent. The pre-game margin model is not applied. "
        "Probability and EV stay blank unless a validated game-state model, an executable ask, "
        "a known fee, and a closed-form settlement are all present. This repository has no such model.",
        "",
        f"Generated {meta.get('generated_at_et')} ({meta.get('generated_at_utc')}).",
        "",
        "| Game | Phase | Score | Clock | Venue quote | Freshness | Action | Prob | EV |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    if not games:
        lines.append("| — | — | — | — | — | — | PASS | — | — |")
    for game in games:
        state = next((s for s in game["states"] if s.get("venue") == game["sync"].get("accepted_state_source")), None)
        if state is None and game["states"]:
            state = game["states"][0]
        score = "—"
        clock = "—"
        if state:
            if state.get("away_score") is not None:
                score = f"{state['away_score']}-{state['home_score']}"
            clock = " ".join(str(x) for x in (state.get("period"), state.get("clock")) if x not in (None, ""))
        quote_bits = []
        for quote in game["quotes"]:
            if quote.get("outcome_kind") != "moneyline_game_winner":
                continue
            ask = quote.get("ask") or {}
            bid = quote.get("bid") or {}
            timed = "book" if quote.get("source_timestamp") else "book, no source time, not executable"
            quote_bits.append(
                f"{quote.get('venue')} {quote.get('side_team') or quote.get('side_name')} "
                f"{timed} {bid.get('price') if bid else '—'}/{ask.get('price') if ask else '—'}"
            )
        fresh = ", ".join(game["fail_reasons"]) if game["fail_reasons"] else "timestamps inside the tool window"
        prob = "—" if game.get("probability") is None else f"{game['probability']:.3f}"
        ev = "—" if all(q.get("gross_ev") is None for q in game["quotes"]) else "per contract"
        lines.append(
            f"| {game['matchup']} | {game['phase']} | {score} | {clock or '—'} | "
            f"{'; '.join(quote_bits) or '—'} | {fresh} | PASS | {prob} | {ev} |"
        )
    lines += ["", "## Games", ""]
    for game in games:
        lines.append(f"### {game['matchup']} ({game['phase']})")
        lines.append("")
        prob = "unavailable" if game.get("probability") is None else f"{game['probability']:.4f}"
        lines.append(f"Action **PASS**. Probability {prob}. Gross EV blank at game level. Net EV blank at game level.")
        lines.append(f"Fail-closed: {', '.join(game['fail_reasons']) or 'none'}.")
        lines.append("")
        lines.append(game["model"]["reason"])
        lines.append("")
        for state in game["states"]:
            lines += [
                f"- State `{state.get('venue')}` phase {state.get('phase')} status {state.get('status')} "
                f"score {state.get('away_score')}-{state.get('home_score')} period {state.get('period')} clock {state.get('clock')}.",
                f"  Source time {state.get('source_timestamp') or 'missing'} ({state.get('source_timestamp_et') or 'n/a'}); "
                f"observed {state.get('observed_at_utc')} ({state.get('observed_at_et')}). "
                f"Field: {state.get('source_timestamp_field') or 'none'}.",
                f"  URL: {state.get('url')}",
                "  Synchronized board." if state.get("usable") else
                "  Not a synchronized game state. A missing or undocumented sample time is not filled in from the fetch clock.",
            ]
            if state.get("down") is not None:
                lines.append(f"  Down {state.get('down')} distance {state.get('distance')} yardline {state.get('yardline')} "
                             f"possession {state.get('possession')} timeouts {state.get('timeouts_away')}/{state.get('timeouts_home')}.")
            if state.get("clock_note"):
                lines.append(f"  {state['clock_note']}")
            if state.get("metadata_updated_at_role"):
                lines.append(f"  Metadata {state.get('metadata_updated_at')}: {state['metadata_updated_at_role']}")
        for quote in game["quotes"]:
            bid, ask = quote.get("bid") or {}, quote.get("ask") or {}
            lines += [
                f"- Quote `{quote.get('venue')}` {quote.get('side_name') or quote.get('side_team')} "
                f"({quote.get('outcome_kind')}) status {quote.get('status')}.",
                f"  Event {quote.get('event_id')} {quote.get('event_url')}",
                f"  Market {quote.get('market_id')} {quote.get('market_url')}",
                f"  Book bid {bid.get('price')} size {bid.get('size')}; book ask {ask.get('price')} size {ask.get('size')}; "
                f"executable_ask={quote.get('executable_ask')}.",
                f"  Notes: {', '.join(quote.get('notes') or []) or 'none'}.",
                f"  Last trade {quote.get('last_trade')} ({quote.get('last_trade_role')}).",
                f"  Source time {quote.get('source_timestamp') or 'missing'} ({quote.get('source_timestamp_et') or 'n/a'}); "
                f"observed {quote.get('observed_at_utc')} ({quote.get('observed_at_et')}).",
            ]
            book = quote.get("book") or {}
            for side in ("bids", "asks"):
                levels = (book.get(side) or {}).get("top") or []
                count = (book.get(side) or {}).get("level_count")
                if levels:
                    shown = ", ".join(f"{lvl['price']} x {lvl['size']}" for lvl in levels[:5])
                    lines.append(f"  Depth {side} ({count} levels, showing {len(levels[:5])}): {shown}.")
                elif count == 0:
                    lines.append(f"  Depth {side}: none returned.")
            if quote.get("volume") is not None:
                lines.append(f"  Volume {quote.get('volume')} open interest {quote.get('open_interest')}.")
            if quote.get("metadata_updated_at"):
                lines.append(f"  Metadata updated_at {quote.get('metadata_updated_at')}: {quote.get('metadata_updated_at_role')}")
            settlement = quote.get("settlement") or {}
            if settlement:
                lines.append(f"  Settlement status {settlement.get('status')} closed_form {settlement.get('closed_form')}.")
                for label in ("overtime", "tie", "cancellation"):
                    if settlement.get(label):
                        lines.append(f"  {label}: {settlement[label]}")
            fee = quote.get("fee") or {}
            if fee:
                lines.append(f"  Fee status {fee.get('status')}. {fee.get('why') or ''}")
        lines.append("")
    if meta.get("source_errors"):
        lines += ["## Source errors", ""]
        for err in meta["source_errors"]:
            lines.append(f"- {err.get('source')} {err.get('op')}: {err.get('error')}")
        lines.append("")
    lines += [
        "## What this report did not validate",
        "",
        PREGAME_MODEL_NOTE,
        "",
        "No live probability and no EV were validated. A stale or missing source timestamp is not treated as current just because the HTTP call succeeded today. "
        "Provider bids and asks above are whatever the public read returned at the observed time; they are not a standing price.",
        "",
    ]
    return "\n".join(lines)


def AssessLive(date=None, matchup=None, write=True, fetcher=None, now=None):
    supplied_now = now is not None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    date = date or now.astimezone(ET).date().isoformat()
    fetcher = fetcher or UrlFetcher()
    errors = []
    notes = []
    try:
        games, notes, errors = LoadKalshi(fetcher, date, now, matchup)
        if games:
            LoadPolymarket(fetcher, games, errors)
    except Exception as exc:
        errors.append({"source": "assess", "op": "load", "error": f"{type(exc).__name__}: {exc}"})
        games = {}
    # Age is measured against the clock after the reads return, unless a test
    # pinned `now`. A fetch that takes a minute must not make a fresh board look stale.
    assessed_at = now if supplied_now else datetime.now(timezone.utc)
    assessed = MergeAndAssess(games, assessed_at) if games else []
    report = EmptyReport(assessed_at, date, errors)
    report["meta"]["notes"] = notes
    report["meta"]["http_calls"] = getattr(fetcher, "calls", None)
    report["meta"]["thresholds_seconds"] = Thresholds()
    report["meta"]["validated"] = {
        "public_reads_attempted": True,
        "live_probability_model": False,
        "ev": False,
        "orders": False,
        "pregame_model_compared_to_quotes": False,
    }
    report["games"] = assessed
    report["markdown"] = Markdown(report["meta"], assessed)
    if write:
        os.makedirs(OUT_DIR, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        stem = os.path.join(OUT_DIR, f"assess_{date}_{stamp}")
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump({"meta": report["meta"], "games": assessed}, fh, indent=1, default=str)
        with open(stem + ".md", "w", encoding="utf-8") as fh:
            fh.write(report["markdown"])
        report["meta"]["written"] = stem + ".json/.md"
    return report


def Main():
    parser = argparse.ArgumentParser(description="One-shot read-only NFL live quote and game-state report. No orders.")
    parser.add_argument("--date", default=None, help="ET date YYYY-MM-DD (default: today ET)")
    parser.add_argument("--matchup", default=None, help="Optional away@home filter, e.g. TB@CHI")
    args = parser.parse_args()
    report = AssessLive(args.date, args.matchup)
    print(report["markdown"])
    if report["meta"].get("written"):
        print(f"wrote {report['meta']['written']}")


if __name__ == "__main__":
    Main()
