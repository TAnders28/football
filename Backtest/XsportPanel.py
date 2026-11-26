"""Cross-sport look-back panel for the NFL (Sportsbot, 2026-11-26). Read-only on data; writes one CSV.

One row per finished game (ties dropped), 2014 onward. Columns follow the cross-sport panel contract read by
basketball `Analysis.CrossSportHistory`:
  date (ET game day), sport, game_id, season, split, home, away, y (1 = home won),
  k_bid/k_ask (Kalshi home YES at T-60, 1-min candle close), k2_bid/k2_ask (T-15), ka_bid/ka_ask (away YES at T-60),
  p_* (home-win probabilities): p_kalshi (T-60 devig of home/away mids), p_kalshi15, p_poly (Polymarket T-90 price),
  p_book (nflverse closing moneyline, multiplicative no-vig; quote time unknown), p_full2 (walk-forward, uses the
  closing line as a feature), p_full2_kmid (same model refit with the Kalshi T-60 mid; the board model),
  p_margin (ratings-only margin model, no market input), bk_home_ml/bk_away_ml (nflverse closing American odds),
  ss90_* / ss15_* (executable snapshot books from data/prices/snapshots_2026.jsonl, only rows whose
  minutes_to_kickoff is within 10 min of the label and whose kickoff matches the final schedule), ctx_* situational fields.
Models are walk-forward by season (fit on seasons before). Everything here is tested:false.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from Backtest.CommonFormat import AttachKmid, Devig, QuoteWithTs
from Backtest.KalshiReality import LoadQuotes
from Backtest.Lab import AllRows, AttachModels, LoadSeason
from Backtest.PolyReality import LoadHistory, PriceAt

SPLIT_SEASON = 2026  # Kalshi-era tune = 2025 season, holdout = 2026 season
LABEL_TOL_MIN = 10.0
KICKOFF_TOL_MIN = 5.0
ET = ZoneInfo("America/New_York")


def LoadRaw(season):
    """Raw nflverse rows (data/backtest or data/backtest/history) for rest, roof, weather and weekday."""
    return {g["game_id"]: g for g in LoadSeason(season)}


def ScheduleKickoff(g):
    """Final-schedule kickoff (UTC) from nflverse gameday + gametime (ET), or None."""
    try:
        return datetime.fromisoformat(f"{g['gameday']}T{g['gametime']}:00").replace(tzinfo=ET).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None


def SnapshotBooks(path, raw):
    """game_id -> label -> (venue, side_team) -> (bid, ask, bid_size, ask_size, fetched_at, minutes_to_kickoff) for rows whose
    label matches minutes_to_kickoff (within LABEL_TOL_MIN) and whose kickoff agrees with the final schedule (KICKOFF_TOL_MIN)."""
    out = defaultdict(lambda: defaultdict(dict))
    flags = []
    if not os.path.exists(path):
        return out, flags
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            lab = r.get("capture_label") or ""
            mtk = r.get("minutes_to_kickoff")
            target = {"T-90": 90.0, "T-15": 15.0}.get(lab)
            if target is None:
                continue
            ok = mtk is not None and abs(float(mtk) - target) <= LABEL_TOL_MIN
            sk = ScheduleKickoff(raw.get(r.get("game_id"), {}))
            snap_ko = datetime.fromisoformat(r["kickoff_utc"].replace("Z", "+00:00")) if r.get("kickoff_utc") else None
            ko_ok = sk is None or snap_ko is None or abs((snap_ko - sk).total_seconds()) <= KICKOFF_TOL_MIN * 60
            if not (ok and ko_ok):
                flag = "label_mismatch_not_%s" % lab if not ok else "kickoff_differs_from_final_schedule"
                if ok and not ko_ok:
                    flag += "_(schedule %s)" % sk.isoformat()
                flags.append({"line": i + 1, "run_id": r.get("run_id"), "game_id": r.get("game_id"), "venue": r.get("venue"),
                              "side_team": r.get("side_team"), "capture_label": lab, "minutes_to_kickoff": mtk,
                              "fetched_at": r.get("fetched_at"), "flag": flag})
                continue
            if r.get("best_bid") is None or r.get("best_ask") is None:
                continue
            key = "ss90" if lab == "T-90" else "ss15"
            prev = out[r["game_id"]][key].get((r["venue"], r["side_team"]))
            # keep the row closest to the label time
            if prev is None or abs(float(mtk) - target) < abs(prev[5] - target):
                out[r["game_id"]][key][(r["venue"], r["side_team"])] = (r["best_bid"], r["best_ask"], r.get("bid_size"),
                                                                       r.get("ask_size"), r.get("fetched_at"), float(mtk))
    return out, flags


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--flags-out", default=None, help="CSV of snapshot rows whose capture_label does not match minutes_to_kickoff")
    ap.add_argument("--first", type=int, default=2014)
    a = ap.parse_args()
    rbs = AllRows(2012, 2026)
    AttachModels(rbs)
    AttachKmid(rbs, seasons=(2025, 2026))
    raw = {}
    for s in range(a.first, 2027):
        raw.update(LoadRaw(s))
    kq = {s: LoadQuotes(s) for s in (2025, 2026)}
    poly = {s: LoadHistory(s) for s in (2024, 2025, 2026)}
    snaps, flags = SnapshotBooks(os.path.join("data", "prices", "snapshots_2026.jsonl"), raw)
    rows = []
    for s in range(a.first, 2027):
        for r in rbs[s]:
            if r["home_win"] is None:
                continue
            g = raw.get(r["game_id"], {})
            row = {"date": r["gameday"], "sport": "NFL", "game_id": r["game_id"], "season": s,
                   "split": ("holdout" if s >= SPLIT_SEASON else "tune"), "home": r["home_team"], "away": r["away_team"],
                   "y": int(r["home_win"]), "week": r["week"], "game_type": r["game_type"]}
            row["p_book"] = r.get("mkt_home")
            row["bk_home_ml"], row["bk_away_ml"] = r.get("home_ml"), r.get("away_ml")
            row["p_full2"] = r.get("p_full2")
            row["p_margin"] = r.get("p_margin")
            row["p_full2_kmid"] = r.get("p_kmid")
            row["p_kalshi"] = r.get("kmid")
            q = (r.get("kq_t60") or {})
            if q:
                row["k_bid"], row["k_ask"] = q["home"]["bid"], q["home"]["ask"]
                row["ka_bid"], row["ka_ask"] = q["away"]["bid"], q["away"]["ask"]
                row["k_time"] = q["home"]["price_time"]
            if s in kq:
                ent = kq[s].get(r["game_id"], {})
                h15, a15 = QuoteWithTs(ent.get(r["home_team"]), 15), QuoteWithTs(ent.get(r["away_team"]), 15)
                if h15 and a15:
                    row["k2_bid"], row["k2_ask"] = h15["bid"], h15["ask"]
                    row["p_kalshi15"] = Devig(h15["mid"], a15["mid"])
                h90, a90 = QuoteWithTs(ent.get(r["home_team"]), 90), QuoteWithTs(ent.get(r["away_team"]), 90)
                if h90 and a90:
                    row["p_kalshi90"] = Devig(h90["mid"], a90["mid"])
            if s in poly:
                pq = PriceAt(poly[s].get(r["game_id"]), 90)
                if pq:
                    row["p_poly"] = pq[0]
            sb = snaps.get(r["game_id"])
            if sb:
                for key in ("ss90", "ss15"):
                    for venue in ("kalshi", "polymarket"):
                        h = sb[key].get((venue, r["home_team"]))
                        aw = sb[key].get((venue, r["away_team"]))
                        if h:
                            row[f"{key}_{venue}_home_bid"], row[f"{key}_{venue}_home_ask"] = h[0], h[1]
                            row[f"{key}_{venue}_home_ask_size"] = h[3]
                        if aw:
                            row[f"{key}_{venue}_away_bid"], row[f"{key}_{venue}_away_ask"] = aw[0], aw[1]
            hr, ar = g.get("home_rest"), g.get("away_rest")
            row["ctx_home_rest"], row["ctx_away_rest"] = hr, ar
            row["ctx_rest_diff"] = r.get("rest_diff")
            row["ctx_div"] = int(r.get("div") or 0)
            row["ctx_spread"] = r.get("spread")
            row["ctx_weekday"] = g.get("weekday")
            row["ctx_gametime"] = r.get("gametime")
            row["ctx_roof"] = g.get("roof")
            row["ctx_wind"] = g.get("wind")
            row["ctx_temp"] = g.get("temp")
            row["ctx_neutral"] = int(g.get("location") == "Neutral")
            rows.append(row)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if a.flags_out:
        with open(a.flags_out, "w", newline="", encoding="utf-8") as fh:
            ff = ["line", "run_id", "game_id", "venue", "side_team", "capture_label", "minutes_to_kickoff", "fetched_at", "flag"]
            w = csv.DictWriter(fh, fieldnames=ff)
            w.writeheader()
            w.writerows(flags)
    n_k = sum(1 for r in rows if r.get("k_ask") is not None)
    n_ss = sum(1 for r in rows if r.get("ss90_kalshi_home_ask") is not None or r.get("ss15_kalshi_home_ask") is not None)
    print(json.dumps({"rows": len(rows), "kalshi_t60": n_k, "snapshot_true_label_games": n_ss, "label_flags": len(flags),
                      "seasons": [a.first, 2026]}))


if __name__ == "__main__":
    Main()
