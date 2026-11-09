"""
Fail-closed readiness for one NFL game's quotes and game state.

This module does not place orders and does not score Backtest.Model.MarginModel.
That model is pre-game: it is fit on historical seasons and only consumes games
that finished on earlier days (see Backtest.Features). A pre-game probability is
not comparable to an in-play quote. There is no game-state-adjusted model in
this repository, so probability and EV stay empty unless a caller passes an
explicitly validated game-state model AND the quote, fee, and settlement gates
below all pass. The CLI never does that.

Gross/net EV, when the gates pass, is per one contract bought at the executable
ask:

    gross = p_yes * yes_payoff + p_tie * tie_payoff + p_no * no_payoff - ask
    net   = gross - taker_fee_per_contract

p_no = 1 - p_yes - p_tie - p_nonbinary, and p_nonbinary must be 0. A tie payoff
of 0.50 or a cancellation that pays "last fair price" is not a closed-form
0/1 settlement, so EV stays blank unless the caller marks the settlement
closed-form and supplies the missing probabilities.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Tool policy, not a provider guarantee. Kalshi documents that live data can be
# a cached board and that live_data_delayed is set only once a feed is
# significantly stale, so a missing flag is not treated as proof of freshness.
MAX_FUTURE_SKEW_S = 90
MAX_LIVE_AGE_S = 45
MAX_PREGAME_AGE_S = 15 * 60
MAX_SYNC_SKEW_S = 20

PREGAME_MODEL_NOTE = (
    "Backtest.Model.MarginModel and the market-anchored full2/full3 models are "
    "pre-game. They are trained on historical seasons and use only games finished "
    "on earlier days. They are not a game-state-adjusted model and are not "
    "compared to these quotes, in-play or otherwise."
)

# Matching aids for the same NFL club. The raw abbreviation from the source is
# kept on the record; only equality uses the canonical form.
CANON_ABBREV = {
    "LAR": "LA", "LA": "LA",
    "JAC": "JAX", "JAX": "JAX",
    "WSH": "WAS", "WAS": "WAS",
    "NOLA": "NO", "NO": "NO",
    "GNB": "GB", "GB": "GB",
}

LIVE_STATUS = {"inprogress", "in_progress", "live", "halftime", "ht"}
PREGAME_STATUS = {"scheduled", "pregame", "not_started", "created", "upcoming", "unopened"}
FINAL_STATUS = {"closed", "final", "complete", "completed", "ended", "finalized"}
OPEN_MARKET = {"active", "open"}
SUSPENDED_MARKET = {"inactive", "initialized", "paused", "suspended", "halted", "disputed"}
CLOSED_MARKET = {"closed", "determined", "finalized", "settled", "amended"}
CLOCK_OPTIONAL_PERIODS = {"HT", "HALFTIME", "FT", "FINAL", "END"}


def Thresholds(**over):
    base = {
        "max_future_s": MAX_FUTURE_SKEW_S,
        "max_live_age_s": MAX_LIVE_AGE_S,
        "max_pregame_age_s": MAX_PREGAME_AGE_S,
        "max_sync_skew_s": MAX_SYNC_SKEW_S,
    }
    base.update(over)
    return base


def Canon(abbrev):
    if not abbrev:
        return None
    key = str(abbrev).strip().upper()
    return CANON_ABBREV.get(key, key)


def ParseTime(value):
    """ISO-8601, unix seconds, or unix milliseconds. None if the value is absent
    or not a timestamp. A metadata field the caller does not pass in stays None."""
    if value is None or value is False or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) or (isinstance(value, str) and _numeric(value)):
        num = float(value)
        if num > 1e12:
            num = num / 1000.0
        if num < 1e9 or num > 1e11:
            return None
        return datetime.fromtimestamp(num, timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _numeric(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def AsFloat(value):
    if value is None or value is False or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    return num


def Iso(dt):
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def EtText(dt):
    if dt is None:
        return None
    local = dt.astimezone(ET)
    clock = local.strftime("%I:%M:%S %p").lstrip("0")
    return local.strftime("%Y-%m-%d ") + clock + " ET"


def BookSide(levels, kind, keep=5):
    """Best bid is the highest price; best ask is the lowest. Array order is not
    trusted: Kalshi's quick start slices the front of the array as the top of
    book, while the book returned for a live NFL market was sorted ascending,
    and Polymarket documents bids ascending and asks descending (best = last).
    Selecting by price is the same in both layouts."""
    rows = []
    for level in levels or []:
        if isinstance(level, dict):
            price, size = AsFloat(level.get("price")), AsFloat(level.get("size"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = AsFloat(level[0]), AsFloat(level[1])
        else:
            continue
        if price is None or size is None:
            continue
        rows.append({"price": price, "size": size})
    if kind == "bid":
        rows.sort(key=lambda r: r["price"], reverse=True)
    else:
        rows.sort(key=lambda r: r["price"])
    best = rows[0] if rows else None
    return {
        "best": best,
        "top": rows[:keep],
        "level_count": len(rows),
        "size_sum": round(sum(r["size"] for r in rows), 4) if rows else 0.0,
    }


def ClassifyAge(source_ts, now, max_age_s, max_future_s):
    """Reasons for one source timestamp. Observed fetch time is not a substitute."""
    if source_ts is None:
        return ["missing_timestamp"], None
    age = (now - source_ts).total_seconds()
    if -age > max_future_s:
        return ["future_timestamp"], round(age, 3)
    if age > max_age_s:
        return ["stale_timestamp"], round(age, 3)
    return [], round(age, 3)


def MarketState(status):
    key = (status or "").strip().lower()
    if key in OPEN_MARKET:
        return "open"
    if key in SUSPENDED_MARKET:
        return "suspended"
    if key in CLOSED_MARKET:
        return "closed"
    return "unknown"


def PhaseOf(status, live=None, ended=None):
    if ended is True:
        return "final"
    key = (status or "").strip().lower()
    if key in LIVE_STATUS or live is True:
        return "live"
    if key in PREGAME_STATUS:
        return "pregame"
    if key in FINAL_STATUS:
        return "final"
    if live is False and ended is False:
        return "pregame"
    return "unknown"


def PeriodKey(period):
    if period is None or period == "":
        return None
    text = str(period).strip().upper()
    aliases = {"Q1": "1", "Q2": "2", "Q3": "3", "Q4": "4", "1ST": "1", "2ND": "2",
               "3RD": "3", "4TH": "4", "HALFTIME": "HT", "HALF": "HT", "OVERTIME": "OT"}
    return aliases.get(text, text)


# Polymarket documents FT as full time, not an in-quarter clock
# (https://docs.polymarket.com/market-data/realtime-data ; sports websocket NFL
# periods end in FT). Finished Gamma NFL events return VFT, the same final-board
# family. A Kalshi board that is already final still displays the last quarter
# played (4 in regulation, 5 in overtime). Those are one ending. They are not
# the same as a live quarter, and a different score is still a disagreement.
FINAL_BOARD_PERIODS = {"FT", "VFT", "FT OT", "FT NR", "F/OT", "FOT", "END", "END_OF_GAME", "FINAL"}
TERMINAL_QUARTERS = {"4", "5", "OT"}


def _final_period_equivalent(left, right):
    def kind(key):
        if key in FINAL_BOARD_PERIODS:
            return "full"
        if key in TERMINAL_QUARTERS:
            return "terminal"
        return "other"
    kinds = {kind(left), kind(right)}
    return kinds == {"full"} or kinds == {"full", "terminal"}


def BoardsAgree(left, right):
    """True when two boards are the same score and the same point in the game.

    Full-time tokens agree with quarter 4 or 5 only when both boards are already
    final. A live quarter against VFT/FT, two different live quarters, or any
    score mismatch stays a disagreement.
    """
    if left.get("away_score") != right.get("away_score") or left.get("home_score") != right.get("home_score"):
        return False
    if not SameTeams(left.get("away"), left.get("home"), right.get("away"), right.get("home")):
        return False
    left_period, right_period = PeriodKey(left.get("period")), PeriodKey(right.get("period"))
    if left_period == right_period:
        return True
    if left.get("phase") == "final" and right.get("phase") == "final":
        return _final_period_equivalent(left_period, right_period)
    return False


def _boards_conflict(states):
    for i, left in enumerate(states):
        for right in states[i + 1:]:
            if not BoardsAgree(left, right):
                return True
    return False


def SameTeams(a_away, a_home, b_away, b_home):
    return Canon(a_away) == Canon(b_away) and Canon(a_home) == Canon(b_home)


def _without_times(row, extra):
    out = {k: v for k, v in row.items() if k not in ("source_timestamp", "observed_at")}
    out.update(extra)
    return out


def _excerpt(text, needle, span=320):
    """Quote the sentence that contains needle. Do not start mid-word."""
    if not text:
        return None
    lower = text.lower()
    idx = lower.find(needle.lower())
    if idx < 0:
        return None
    start = max(text.rfind(". ", 0, idx), text.rfind("? ", 0, idx), text.rfind("! ", 0, idx))
    start = 0 if start < 0 else start + 2
    end = len(text)
    for mark in (". ", "? ", "! "):
        found = text.find(mark, idx)
        if found >= 0:
            end = min(end, found + 1)
    clip = " ".join(text[start:end].split())
    if len(clip) > span:
        clip = clip[:span].rsplit(" ", 1)[0]
    return clip or None


def SettlementFromRules(text):
    """Pull OT / tie / cancellation language out of provider rule text.

    closed_form is false when a named path pays a discretionary last fair price
    or when OT/tie are not both stated. Known prose is still returned so the
    report can quote it; that is not a number EV can use.
    """
    raw = " ".join((text or "").split())
    if not raw:
        return {"status": "unknown", "closed_form": False, "overtime": None, "tie": None,
                "cancellation": None, "yes_payoff": None, "tie_payoff": None, "no_payoff": None,
                "source_text_present": False}
    overtime = _excerpt(raw, "overtime")
    tie = _excerpt(raw, "tie")
    cancel = None
    for needle in ("postponed", "cancelled", "canceled", "abandoned", "forfeit", "fair price", "fair market"):
        cancel = cancel or _excerpt(raw, needle)
    discretionary = any(phrase in raw.lower() for phrase in ("fair price", "fair market", "sole discretion", "50-50", "50/50"))
    yes_payoff, tie_payoff, no_payoff = None, None, None
    lower = raw.lower()
    if "overtime" in lower and ("$0.50" in raw or "0.50" in raw or "50-50" in lower or "50/50" in lower):
        yes_payoff, tie_payoff, no_payoff = 1.0, 0.5, 0.0
    status = "known" if overtime and tie and cancel else "partial"
    # A tie at 0.50, a 50-50 cancellation, or a last-fair-price path is not a
    # 0/1 payoff. closed_form stays false so EV cannot treat it as one.
    closed = False
    return {
        "status": status,
        "closed_form": closed,
        "overtime": overtime,
        "tie": tie,
        "cancellation": cancel,
        "discretionary_payoff": discretionary,
        "yes_payoff": yes_payoff,
        "tie_payoff": tie_payoff,
        "no_payoff": no_payoff,
        "source_text_present": True,
    }


def _clock_required(phase, period):
    if phase != "live":
        return False
    key = PeriodKey(period)
    if key in CLOCK_OPTIONAL_PERIODS:
        return False
    return True


def _state_problems(state, now, limits):
    reasons = []
    phase = PhaseOf(state.get("status"), state.get("live"), state.get("ended"))
    if phase == "unknown" and state.get("winner"):
        phase = "final"   # the board names a winner: the game is over even if the status word is new to us
    if phase == "unknown":
        reasons.append("game_status_unknown")
    # A finished score is still shown. It is a synchronized board only when its
    # own source time is recent. The retrieval clock is not that source time.
    max_age = limits["max_pregame_age_s"] if phase == "pregame" else limits["max_live_age_s"]
    age_reasons, age = ClassifyAge(state.get("source_timestamp"), now, max_age, limits["max_future_s"])
    if state.get("delayed") is True:
        age_reasons = sorted(set(age_reasons + ["provider_marked_delayed"]))
    reasons.extend(f"state_{r}" if not r.startswith("state_") else r for r in age_reasons)
    if state.get("delayed") is True and "state_provider_marked_delayed" not in reasons:
        reasons.append("state_provider_marked_delayed")
    if not state.get("away") or not state.get("home"):
        reasons.append("state_teams_missing")
    if phase == "live":
        if state.get("away_score") is None or state.get("home_score") is None:
            reasons.append("missing_score")
        if not PeriodKey(state.get("period")):
            reasons.append("missing_period")
        if _clock_required(phase, state.get("period")) and not state.get("clock"):
            reasons.append("missing_clock")
    return phase, reasons, age


def _quote_problems(quote, now, limits, phase):
    reasons = []
    kind = quote.get("outcome_kind")
    if kind != "moneyline_game_winner":
        reasons.append("wrong_outcome")
    market = MarketState(quote.get("status"))
    if market == "suspended":
        reasons.append("suspended")
    elif market == "closed":
        reasons.append("closed")
    elif market != "open":
        reasons.append("market_status_unknown")
    max_age = limits["max_pregame_age_s"] if phase == "pregame" else limits["max_live_age_s"]
    age_reasons, age = ClassifyAge(quote.get("source_timestamp"), now, max_age, limits["max_future_s"])
    reasons.extend(f"quote_{r}" for r in age_reasons)
    book = quote.get("book") or {}
    bid = (book.get("bids") or {}).get("best")
    ask = (book.get("asks") or {}).get("best")
    if bid is None and ask is None and not book.get("bids", {}).get("level_count") and not book.get("asks", {}).get("level_count"):
        reasons.append("no_depth")
    summary_bid, summary_ask = quote.get("summary_bid"), quote.get("summary_ask")
    notes = []
    if bid and summary_bid is not None and abs(bid["price"] - summary_bid) > 0.001:
        notes.append("summary_disagrees_with_book")
    if ask and summary_ask is not None and abs(ask["price"] - summary_ask) > 0.001:
        notes.append("summary_disagrees_with_book")
    last_trade = AsFloat(quote.get("last_trade"))
    if last_trade is not None and bid and ask and (last_trade < bid["price"] - 0.02 or last_trade > ask["price"] + 0.02):
        notes.append("last_trade_outside_book")
    executable = market == "open" and ask is not None and ask.get("size") is not None and ask["size"] > 0 and "wrong_outcome" not in reasons
    if market == "open" and not executable and "wrong_outcome" not in reasons and "no_depth" not in reasons:
        if ask is None or not ask.get("size"):
            reasons.append("no_depth")
    return reasons, age, executable, bid, ask, notes


def _model_gate(model):
    """Return (applied_model_or_None, reasons). A pre-game probability is dropped
    and is not copied into the result."""
    if not model:
        return None, ["no_validated_live_model"]
    scope = model.get("scope")
    if scope != "game_state_adjusted" or model.get("validated") is not True:
        if scope == "pregame" or model.get("name") in {"MarginModel", "full2", "full3", "value_full2_edge8"}:
            return None, ["pregame_model_misuse"]
        return None, ["no_validated_live_model"]
    p_yes, p_tie = model.get("p_yes"), model.get("p_tie")
    p_other = model.get("p_nonbinary", 0.0)
    if p_yes is None or p_tie is None or p_other not in (0, 0.0):
        return None, ["no_validated_live_model"]
    if min(p_yes, p_tie, p_other) < 0 or p_yes + p_tie + p_other > 1 + 1e-9:
        return None, ["no_validated_live_model"]
    return {"p_yes": float(p_yes), "p_tie": float(p_tie), "p_nonbinary": float(p_other or 0)}, []


def _ev(applied, ask, fee, settlement):
    reasons = []
    if applied is None:
        reasons.append("probability_unavailable")
    if ask is None:
        reasons.append("executable_ask_unavailable")
    fee_ok = isinstance(fee, dict) and fee.get("status") == "known" and AsFloat(fee.get("taker_fee_per_contract")) is not None
    if not fee_ok:
        reasons.append("fee_unknown")
    settle_ok = isinstance(settlement, dict) and settlement.get("status") == "known" and settlement.get("closed_form") is True
    if not isinstance(settlement, dict) or settlement.get("status") == "unknown" or not settlement.get("source_text_present"):
        reasons.append("settlement_unknown")
    elif not settle_ok:
        reasons.append("settlement_not_closed_form")
    if reasons:
        return None, None, reasons
    p_no = 1.0 - applied["p_yes"] - applied["p_tie"] - applied["p_nonbinary"]
    gross = (applied["p_yes"] * settlement["yes_payoff"]
             + applied["p_tie"] * settlement["tie_payoff"]
             + p_no * settlement["no_payoff"]
             - ask["price"])
    net = gross - float(fee["taker_fee_per_contract"])
    return round(gross, 6), round(net, 6), []


def BoardPhase(prepared_states):
    """Phase shown when no board is synchronized (e.g. every source time is stale).

    Staleness decides whether a board is usable, not what phase it reports. If every board that
    names a phase names the same one, show it: a finished game with an old source time is still
    "final" (and still PASS). Boards that name different phases stay "unknown".
    """
    known = {s["phase"] for s in prepared_states if s.get("phase") not in (None, "unknown")}
    return known.pop() if len(known) == 1 else "unknown"


def AssessGame(game, now, model=None, thresholds=None):
    """Assess one matched game. `game` carries normalized quotes, states, fees, settlement.

    `model` is ignored unless scope is game_state_adjusted and validated is true.
    The pre-game model must not be passed in by the CLI. Tests cover a caller who does.
    """
    limits = Thresholds(**(thresholds or {}))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    quotes = list(game.get("quotes") or [])
    states = list(game.get("states") or [])
    reasons = []
    if not quotes:
        reasons.append("missing_quote")
    if not states:
        reasons.append("missing_game_state")

    prepared_states = []
    for state in states:
        phase, state_reasons, age = _state_problems(state, now, limits)
        prepared_states.append(_without_times(state, {
            "source_timestamp": Iso(state.get("source_timestamp")),
            "observed_at_utc": Iso(state.get("observed_at")),
            "observed_at_et": EtText(state.get("observed_at")),
            "source_timestamp_et": EtText(state.get("source_timestamp")),
            "phase": phase, "fail_reasons": state_reasons, "age_seconds": age,
            "usable": not state_reasons and phase != "unknown",
        }))

    usable = [s for s in prepared_states if s["usable"]]
    accepted = None
    if len(usable) == 1:
        accepted = usable[0]
    elif len(usable) > 1:
        if _boards_conflict(usable):
            reasons.append("state_mismatch")
        else:
            # Same score and period. Prefer the newest source timestamp.
            accepted = max(usable, key=lambda s: s.get("source_timestamp") or "")
    if states and not usable and not any(s["fail_reasons"] for s in prepared_states):
        reasons.append("missing_game_state")
    phase = accepted["phase"] if accepted else BoardPhase(prepared_states)
    if not accepted and prepared_states and "state_mismatch" not in reasons:
        reasons.append("no_synchronized_state")

    away = (accepted or prepared_states[0] if prepared_states else game).get("away") if (accepted or prepared_states) else game.get("away")
    home = (accepted or (prepared_states[0] if prepared_states else game)).get("home") if (accepted or prepared_states) else game.get("home")
    if not away:
        away = game.get("away")
    if not home:
        home = game.get("home")

    prepared_quotes = []
    for quote in quotes:
        q_reasons, age, executable, bid, ask, notes = _quote_problems(quote, now, limits, phase if phase != "unknown" else "live")
        if quote.get("away") and quote.get("home") and away and home and not SameTeams(quote.get("away"), quote.get("home"), away, home):
            q_reasons.append("event_mismatch")
        skew_seconds = None
        if accepted and quote.get("source_timestamp") and accepted.get("source_timestamp"):
            q_ts = quote["source_timestamp"]
            s_ts = ParseTime(accepted["source_timestamp"])
            if s_ts is not None:
                skew = abs((q_ts - s_ts).total_seconds())
                if skew > limits["max_sync_skew_s"]:
                    q_reasons.append("sync_skew")
                skew_seconds = round(skew, 3)
        elif accepted and quote.get("source_timestamp") is None:
            q_reasons.append("sync_unknown")
        elif not accepted:
            q_reasons.append("sync_unknown")
        if "wrong_outcome" in q_reasons or "suspended" in q_reasons or "closed" in q_reasons or "event_mismatch" in q_reasons:
            executable = False
        if any(r in q_reasons for r in ("quote_missing_timestamp", "quote_stale_timestamp", "quote_future_timestamp", "sync_skew", "sync_unknown")):
            executable = False
        prepared_quotes.append(_without_times(quote, {
            "source_timestamp": Iso(quote.get("source_timestamp")),
            "source_timestamp_et": EtText(quote.get("source_timestamp")),
            "observed_at_utc": Iso(quote.get("observed_at")),
            "observed_at_et": EtText(quote.get("observed_at")),
            "age_seconds": age,
            "sync_skew_seconds": skew_seconds,
            "bid": bid,
            "ask": ask,
            "executable_ask": executable,
            "fail_reasons": sorted(set(q_reasons)),
            "notes": sorted(set(notes)),
        }))

    scored = [s for s in prepared_states if s.get("away_score") is not None and s.get("home_score") is not None]
    if _boards_conflict(scored):
        reasons.append("state_disagreement")
        for q in prepared_quotes:
            q["executable_ask"] = False
            q["fail_reasons"] = sorted(set(q["fail_reasons"] + ["state_disagreement"]))
    for q in prepared_quotes:
        reasons.extend(q["fail_reasons"])
    for s in prepared_states:
        reasons.extend(s["fail_reasons"])

    applied, model_reasons = _model_gate(model)
    if applied is not None and (not accepted or phase == "pregame"):
        # A game-state model still cannot be applied without a fresh live board,
        # and it is the wrong instrument for a pre-game price.
        applied = None
        model_reasons = ["no_validated_live_model"]
        if phase == "pregame":
            model_reasons = ["pregame_model_misuse"]
    reasons.extend(model_reasons)

    # EV is per executable contract. With no validated model every contract is blank.
    # When a model is applied, it is P(YES) for the contract named by model["side_team"].
    ev_rows = []
    for quote in prepared_quotes:
        fee = quote.get("fee") if quote.get("fee") is not None else game.get("fee")
        settlement = quote.get("settlement") if quote.get("settlement") is not None else game.get("settlement")
        side_ok = applied is not None and Canon(quote.get("side_team")) == Canon((model or {}).get("side_team"))
        use_model = applied if side_ok else None
        ask = quote["ask"] if quote["executable_ask"] else None
        gross, net, ev_reasons = _ev(use_model, ask, fee, settlement)
        if not side_ok and applied is not None:
            ev_reasons = sorted(set(ev_reasons + ["model_side_mismatch"]))
            gross, net = None, None
        quote["gross_ev"] = gross
        quote["net_ev"] = net
        # Probability is published only with a validated game-state model on an
        # executable, synchronized quote. It is never the pre-game model's number.
        quote["probability"] = applied["p_yes"] if side_ok and quote["executable_ask"] else None
        ev_rows.append((gross, net, ev_reasons))
        reasons.extend(ev_reasons)

    reasons = sorted(set(reasons))
    any_ev = any(row[0] is not None for row in ev_rows)
    published = next((q["probability"] for q in prepared_quotes if q["probability"] is not None), None)
    return {
        "matchup": f"{away or '?'} @ {home or '?'}",
        "away": away,
        "home": home,
        "start_utc": game.get("start_utc"),
        "phase": phase,
        "action": "PASS",
        "probability": published,
        "gross_ev": None,
        "net_ev": None,
        "ev_note": "blank" if not any_ev else "per-contract gross_ev and net_ev are on quotes; this tool does not recommend a wager",
        "fail_reasons": reasons,
        "model": {
            "applied": published is not None,
            "scope_accepted": "game_state_adjusted" if published is not None else None,
            "reason": _model_reason(reasons, phase, published, any_ev),
        },
        "sync": {
            "max_skew_seconds": limits["max_sync_skew_s"],
            "max_live_age_seconds": limits["max_live_age_s"],
            "accepted_state_source": None if not accepted else accepted.get("venue"),
        },
        "quotes": prepared_quotes,
        "states": prepared_states,
        "fee": game.get("fee"),
        "settlement": game.get("settlement"),
    }


def _model_reason(reasons, phase, published, any_ev):
    if "pregame_model_misuse" in reasons or phase == "pregame":
        return PREGAME_MODEL_NOTE
    if published is None:
        return "No validated game-state-adjusted probability is available."
    if not any_ev:
        return ("A validated game-state probability is on the matching quote. "
                "Gross and net EV stay blank until the executable ask, the fee, and a closed-form settlement are all known.")
    return "Caller supplied a validated game-state model and every EV gate passed. Action stays PASS; this tool does not recommend a wager."


def EmptyReport(now, date, errors):
    return {
        "meta": {
            "date_et": date,
            "generated_at_utc": Iso(now),
            "generated_at_et": EtText(now),
            "read_only": True,
            "orders": False,
            "pregame_model_applied": False,
            "pregame_model_note": PREGAME_MODEL_NOTE,
            "live_probability": "unavailable",
            "thresholds_seconds": Thresholds(),
            "source_errors": errors,
        },
        "games": [],
        "markdown": "",
    }
