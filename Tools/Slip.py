r"""
Slip checker: what a ticket is worth at no-vig fair prices, before Tim buys it. Read-only, no orders.

    .venv\Scripts\python.exe -m Tools.Slip --league nfl --date 2026-11-15 --leg "DET ml" --leg "BUF -6.5" --price 0.30 --stake 10
    .venv\Scripts\python.exe -m Tools.Slip --slip E:\AgentDocs\American-Footbot\slips\example_slip.json
    .venv\Scripts\python.exe -m Tools.Slip --leg "KC ml @0.39" --contracts 25          (single, Kalshi price)
    .venv\Scripts\python.exe -m Tools.Slip --leg "KC@ATL over 47.5" --leg "ticker:KXNFLTD-26NOV15KCATL-..." --american +450 --stake 20

Leg syntax (team = ESPN abbreviation or name):  "<team> ml" | "<team> <line>" (spread, e.g. "BUF -6.5")
  | "<team> over|under <line>" or "<away>@<home> over|under <line>" (game total) | "ticker:<Kalshi ticker>" (any Kalshi
  market, e.g. a prop) ; optional " @<price>" = what Tim pays for that leg in $ per $1 contract.
Slip file: JSON (or YAML if PyYAML is installed) {"league","date","legs":[...strings or {"leg": "...", "price": x}],
  "price"|"american"|"decimal", "stake"|"contracts"}.

Fair probability per leg:
  ml / spread / total at the book line: multiplicative no-vig of the book ESPN shows (provider named).
  spread / total at a different line: book fair shifted with a Normal margin model (sd 13.5 NFL / 16 CFB for margins,
    10 / 13 for totals - an approximation, labelled "shifted"); the exact Kalshi strike mid is shown as a cross-check.
  no book line: Kalshi mid (de-vigged across both teams for ml).  ticker legs: Kalshi mid.
Combo fair = product of leg fairs (independence). Same-game legs are flagged: the product can be off when legs
are correlated. Price: --price ($ per $1 payout for the whole ticket), or --american / --decimal odds, or per-leg
@prices (a single leg, or legs priced separately -> each scored as its own ticket and summed).
Fees: Tim's estimate $0.02 per contract, and Kalshi's taker fee ceil(0.07 * p * (1-p) * contracts) in cents.
For a sportsbook ticket the vig is already in the price; fee lines are shown for comparison only.
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
from datetime import datetime

from Tools import Common as C

OUT_DIR = os.path.join("E:\\AgentDocs", "American-Footbot", "slips")


def FindGame(events, token):
    tok = token.strip()
    def m(team):
        t = team["team"]
        if tok.upper() == (t.get("abbreviation") or "").upper():
            return 2.0
        return C.NameMatch(tok, t)
    best, score, side = None, 0.0, None
    for ev in events:
        comp, h, a = C.Competitors(ev)
        for sd, t in (("home", h), ("away", a)):
            s = m(t)
            if s > score:
                best, score, side = ev, s, sd
    return (best, side) if score >= 0.6 else (None, None)


def ParseLeg(raw):
    s = raw.strip()
    price = None
    mp = re.search(r"\s@\s*([0-9]*\.?[0-9]+)\s*$", s)
    if mp:
        price = float(mp.group(1)); s = s[:mp.start()].strip()
    if s.lower().startswith("ticker:"):
        return {"kind": "ticker", "ticker": s.split(":", 1)[1].strip(), "price": price, "raw": raw}
    mt = re.match(r"^(.+?)\s+(over|under|o|u)\s*([0-9]+\.?[0-9]*)$", s, re.I)
    if mt:
        side = "over" if mt.group(2).lower().startswith("o") else "under"
        team = mt.group(1).split("@")[-1].strip()
        return {"kind": "total", "team": team, "side": side, "line": float(mt.group(3)), "price": price, "raw": raw}
    mm = re.match(r"^(.+?)\s+(ml|moneyline|win)$", s, re.I)
    if mm:
        return {"kind": "ml", "team": mm.group(1).strip(), "price": price, "raw": raw}
    ms = re.match(r"^(.+?)\s+([+-]?[0-9]+\.?[0-9]*|pk|pick)$", s, re.I)
    if ms:
        ln = 0.0 if ms.group(2).lower() in ("pk", "pick") else float(ms.group(2))
        return {"kind": "spread", "team": ms.group(1).strip(), "line": ln, "price": price, "raw": raw}
    raise ValueError(f"cannot parse leg: {raw!r}")


def ResolveLeg(leg, league, events, kmk, cache):
    out = dict(leg)
    out.update({"fair": None, "fair_source": None, "kalshi": None, "game": None, "notes": []})
    if leg["kind"] == "ticker":
        body = C.GetJson(f"{C.KALSHI}/markets/{leg['ticker']}") or {}
        m = body.get("market")
        if not m:
            out["notes"].append("ticker not found on Kalshi")
            return out
        q = C.Quote(m)
        out["kalshi"] = q
        out["label"] = m.get("title") or leg["ticker"]
        out["game"] = (m.get("event_ticker") or "").split("-", 1)[-1]
        if q["mid"] is not None:
            out["fair"], out["fair_source"] = q["mid"], "Kalshi mid (bid/ask midpoint)"
        if out["price"] is None and q["ask"] is not None:
            out["price"], out["price_source"] = q["ask"], "Kalshi yes ask"
        if q["width"] is not None and q["width"] > 0.06:
            out["notes"].append(f"wide Kalshi market ({q['bid']}/{q['ask']}) - mid is a rough fair")
        return out
    ev, side = FindGame(events, leg["team"])
    if not ev:
        out["notes"].append(f"no {league} game found for '{leg['team']}' on this date")
        return out
    comp, h, a = C.Competitors(ev)
    start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
    me, opp = (h, a) if side == "home" else (a, h)
    out["game"] = f"{a['team']['abbreviation']}@{h['team']['abbreviation']}"
    eid = str(ev["id"])
    if eid not in cache:
        odds = C.EspnOdds(comp)
        if not odds:
            summ, _ = C.EspnSummary(league, eid)
            odds = C.EspnOdds({"odds": summ.get("pickcenter")}) if summ.get("pickcenter") else None
        km = C.MatchKalshiGame(kmk["game"], h["team"], a["team"], start)
        cache[eid] = (odds, C.FairFromOdds(odds), km)
    odds, fair, km = cache[eid]
    prov = (odds or {}).get("provider") or "book"
    if leg["kind"] == "ml":
        out["label"] = f"{me['team']['abbreviation']} moneyline"
        if fair.get("ml"):
            out["fair"], out["fair_source"] = fair["ml"][side], f"{prov} no-vig moneyline (hold {fair['ml']['hold'] * 100:.1f}%)"
        if km:
            q, qo = km[side], km["away" if side == "home" else "home"]
            out["kalshi"] = q
            if q["mid"] and qo["mid"]:
                dv = q["mid"] / (q["mid"] + qo["mid"])
                out["kalshi_devig"] = dv
                if out["fair"] is None:
                    out["fair"], out["fair_source"] = dv, "Kalshi de-vigged mid"
            if out["price"] is None and q["ask"] is not None:
                out["price"], out["price_source"] = q["ask"], "Kalshi yes ask"
    elif leg["kind"] == "spread":
        L = leg["line"]
        out["label"] = f"{me['team']['abbreviation']} {L:+g}"
        sp = fair.get("spread")
        if sp:
            book_line = sp["home_line"] if side == "home" else -sp["home_line"]
            p_book = sp[side]
            p = C.ShiftLineProb(p_book, book_line, L, C.MARGIN_SD[league], True)
            out["fair"] = p
            out["fair_source"] = f"{prov} no-vig spread at {book_line:+g}" + ("" if abs(L - book_line) < 1e-9 else
                                 f", shifted to {L:+g} (Normal sd {C.MARGIN_SD[league]})")
        if km:
            # Kalshi: '<team> wins by over X'  ==  team -X. A team +L ticket = NOT(opp wins by over L).
            lad = [m for m in kmk["spread"] if (m.get("event_ticker") or "").endswith("-" + km["event_suffix"])]
            for m in lad:
                k = m.get("floor_strike")
                t = (m.get("yes_sub_title") or "").split(" wins by")[0]
                if k is None:
                    continue
                q = C.Quote(m)
                if L < 0 and abs(float(k) + L) < 1e-9 and C.NameMatch(t, me["team"]) >= 0.6:
                    out["kalshi"] = q
                    if out["fair"] is None and q["mid"] is not None:
                        out["fair"], out["fair_source"] = q["mid"], "Kalshi spread mid"
                    if out["price"] is None and q["ask"] is not None:
                        out["price"], out["price_source"] = q["ask"], "Kalshi yes ask"
                if L > 0 and abs(float(k) - L) < 1e-9 and C.NameMatch(t, opp["team"]) >= 0.6:
                    no = {"ticker": q["ticker"] + " (NO side)", "bid": None if q["ask"] is None else round(1 - q["ask"], 4),
                          "ask": None if q["bid"] is None else round(1 - q["bid"], 4),
                          "mid": None if q["mid"] is None else round(1 - q["mid"], 4)}
                    out["kalshi"] = no
                    if out["fair"] is None and no["mid"] is not None:
                        out["fair"], out["fair_source"] = no["mid"], "Kalshi spread NO mid"
                    if out["price"] is None and no["ask"] is not None:
                        out["price"], out["price_source"] = no["ask"], "Kalshi NO ask"
    elif leg["kind"] == "total":
        L, sd = leg["line"], leg["side"]
        out["label"] = f"{out['game']} {sd} {L:g}"
        tt = fair.get("total")
        if tt:
            p = C.ShiftLineProb(tt[sd], tt["line"], L, C.TOTAL_SD[league], higher_is_better=(sd == "under"))
            out["fair"] = p
            out["fair_source"] = f"{prov} no-vig total at {tt['line']:g}" + ("" if abs(L - tt["line"]) < 1e-9 else
                                 f", shifted to {L:g} (Normal sd {C.TOTAL_SD[league]})")
        if km:
            for m in kmk["total"]:
                if (m.get("event_ticker") or "").endswith("-" + km["event_suffix"]) and m.get("floor_strike") is not None \
                        and abs(float(m["floor_strike"]) - L) < 1e-9:
                    q = C.Quote(m)
                    if sd == "under":
                        q = {"ticker": q["ticker"] + " (NO side)", "bid": None if q["ask"] is None else round(1 - q["ask"], 4),
                             "ask": None if q["bid"] is None else round(1 - q["bid"], 4), "mid": None if q["mid"] is None else round(1 - q["mid"], 4)}
                    out["kalshi"] = q
                    if out["fair"] is None and q["mid"] is not None:
                        out["fair"], out["fair_source"] = q["mid"], "Kalshi total mid"
                    if out["price"] is None and q["ask"] is not None:
                        out["price"], out["price_source"] = q["ask"], "Kalshi ask"
    if out["fair"] is None:
        out["notes"].append("no fair price available (no book line and no two-sided Kalshi quote)")
    return out


def AmericanToPrice(a):
    a = float(str(a).replace("+", ""))
    dec = 1 + (a / 100.0 if a > 0 else 100.0 / -a)
    return 1.0 / dec


def Check(legs_raw, league="nfl", date=None, price=None, american=None, decimal=None, stake=None, contracts=None):
    date = date or datetime.now(C.ET).date().isoformat()
    events, espn_at = C.EspnScoreboard(league, date)
    ser = C.KALSHI_SERIES[league]
    kmk = {"game": C.KalshiOpen(ser["game"])[0], "spread": C.KalshiOpen(ser["spread"])[0], "total": C.KalshiOpen(ser["total"])[0]}
    cache, legs = {}, []
    for raw in legs_raw:
        p_override = None
        if isinstance(raw, dict):
            p_override = raw.get("price"); raw = raw["leg"]
        lg = ParseLeg(raw)
        if p_override is not None:
            lg["price"] = float(p_override)
        legs.append(ResolveLeg(lg, league, events, kmk, cache))
    games = [l["game"] for l in legs if l.get("game")]
    same_game = sorted({g for g in games if games.count(g) > 1})
    combo_fair = C.ComboFair([l["fair"] for l in legs])
    ticket_price, price_source = None, None
    if price is not None:
        ticket_price, price_source = float(price), "given ($ per $1 payout)"
    elif american is not None:
        ticket_price, price_source = AmericanToPrice(american), f"American {american}"
    elif decimal is not None:
        ticket_price, price_source = 1.0 / float(decimal), f"decimal {decimal}"
    res = {"league": league, "date": date, "generated_at": C.IsoEt(), "legs": legs, "combo_fair": combo_fair,
           "same_game_legs": same_game, "tickets": []}
    if ticket_price is None and len(legs) > 1 and all(l.get("price") is not None for l in legs):
        # legs priced separately -> independent tickets
        for l in legs:
            n = contracts if contracts else (stake / l["price"] if stake else 1.0)
            res["tickets"].append({"what": l.get("label", l["raw"]), "price": l["price"], "contracts": round(n, 2),
                                   "fair": l["fair"], **C.SlipMath(l["fair"], l["price"], n)})
        res["mode"] = "separate singles"
    else:
        if ticket_price is None and len(legs) == 1:
            ticket_price, price_source = legs[0].get("price"), legs[0].get("price_source", "leg @price")
        if ticket_price is None and len(legs) > 1 and all(l.get("price") for l in legs):
            ticket_price = C.ComboFair([l["price"] for l in legs]); price_source = "product of leg prices"
        res["mode"] = "single ticket" if len(legs) == 1 else "parlay"
        if ticket_price is not None:
            n = contracts if contracts else (stake / ticket_price if stake else 1.0)
            res["tickets"].append({"what": " + ".join(l.get("label", l["raw"]) for l in legs), "price": round(ticket_price, 4),
                                   "price_source": price_source, "contracts": round(n, 2), "fair": combo_fair,
                                   **C.SlipMath(combo_fair, ticket_price, n)})
        else:
            res["notes"] = ["no ticket price: pass --price, --american, --decimal or leg @prices"]
    return res


def Md(res):
    L = [f"# Slip check - {res['league'].upper()} {res['date']}", f"Generated {C.StampEt()}. Decision support only; no orders sent.", "",
         "| leg | fair prob | fair source | Kalshi bid/ask (mid) | leg price | notes |", "|---|---|---|---|---|---|"]
    for l in res["legs"]:
        k = l.get("kalshi")
        ks = "n/a" if not k else f"{k.get('bid')}/{k.get('ask')} ({k.get('mid')})"
        L.append(f"| {l.get('label', l['raw'])} | {'n/a' if l['fair'] is None else f'{l['fair'] * 100:.1f}%'} | {l.get('fair_source') or 'n/a'} | "
                 f"{ks} | {l.get('price') if l.get('price') is not None else 'n/a'} | {'; '.join(l['notes']) or ''} |")
    cf = res["combo_fair"]
    L += ["", f"Combo fair probability: {'n/a' if cf is None else f'{cf * 100:.2f}%'} (product of legs, independence assumed)"]
    if res["same_game_legs"]:
        L.append(f"WARNING same-game legs ({', '.join(res['same_game_legs'])}): legs are correlated, the product is only an approximation.")
    for t in res["tickets"]:
        L += ["", f"## {t['what']}", f"price ${t['price']:.4f} per $1 ({t.get('price_source', 'leg price')}), contracts {t['contracts']}",
              "", "| | value |", "|---|---|"]
        for k2 in ("cost", "payout_if_win", "profit_if_win", "fair_value", "ev_at_fair_before_fees", "fee_tim_est_0.02",
                   "ev_after_tim_fee", "fee_kalshi_taker", "ev_after_kalshi_fee", "breakeven_prob_tim", "breakeven_prob_kalshi"):
            v = t.get(k2)
            L.append(f"| {k2} | {'n/a' if v is None else (f'${v:,.2f}' if not k2.startswith('breakeven') else f'{v * 100:.1f}%')} |")
        if t.get("fair") is not None:
            verdict = "positive" if (t["ev_after_tim_fee"] or 0) > 0 else "negative"
            L.append(f"\nAt fair prices this ticket is EV-{verdict} after Tim's fee estimate "
                     f"(fair {t['fair'] * 100:.1f}% vs breakeven {t['breakeven_prob_tim'] * 100:.1f}%).")
    for n in res.get("notes", []):
        L.append(f"\n{n}")
    return "\n".join(L) + "\n"


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=["nfl", "cfb"], default="nfl")
    ap.add_argument("--date")
    ap.add_argument("--leg", action="append", default=[])
    ap.add_argument("--slip", help="JSON (or YAML) slip file")
    ap.add_argument("--price", type=float)
    ap.add_argument("--american")
    ap.add_argument("--decimal", type=float)
    ap.add_argument("--stake", type=float)
    ap.add_argument("--contracts", type=float)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    if a.slip:
        a.slip = os.path.abspath(a.slip)
    os.chdir(ROOT)  # data/ paths are repo-relative
    spec = {}
    if a.slip:
        txt = open(a.slip, encoding="utf-8").read()
        if a.slip.lower().endswith((".yml", ".yaml")):
            import yaml  # optional dependency
            spec = yaml.safe_load(txt)
        else:
            spec = json.loads(txt)
    legs = spec.get("legs", []) + a.leg
    if not legs:
        ap.error("give at least one --leg or a --slip file")
    res = Check(legs, spec.get("league", a.league), spec.get("date", a.date),
                a.price if a.price is not None else spec.get("price"), a.american or spec.get("american"),
                a.decimal or spec.get("decimal"), a.stake if a.stake is not None else spec.get("stake"),
                a.contracts if a.contracts is not None else spec.get("contracts"))
    md = Md(res)
    print(md)
    if not a.no_save:
        os.makedirs(a.out_dir, exist_ok=True)
        base = os.path.join(a.out_dir, datetime.now(C.ET).strftime("%Y-%m-%d_%H%M%S"))
        open(base + ".md", "w", encoding="utf-8").write(md)
        json.dump(res, open(base + ".json", "w", encoding="utf-8"), indent=1, default=str)
        print(base + ".md")


if __name__ == "__main__":
    Main()
