r"""Flatten simulator report JSONs into one tidy CSV for Graphtard (Sportsbot, 2026-11-26).

Columns: date (run time, ISO with offset), sport, series (JSON path of the block), metric, value, ci_lo, ci_hi.
A list of two numbers under a key ending in 'ci' (or named 'ci') becomes ci_lo / ci_hi of the matching metric row.

Usage: .venv\Scripts\python.exe -m Sim.TidyCsv --out <csv> SPORT=<json> [SPORT=<json> ...]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from zoneinfo import ZoneInfo


def Walk(node, path, rows):
    if isinstance(node, dict):
        cis = {}
        for k, v in node.items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and (k == "ci" or k.endswith("_ci") or k.endswith("ci")):
                cis[k] = v
        for k, v in node.items():
            if k in cis:
                continue
            if isinstance(v, (dict, list)):
                Walk(v, path + [str(k)], rows)
            elif isinstance(v, (int, float, bool)) and not isinstance(v, str):
                ci = cis.get(f"{k}_ci") or (cis.get("ci") if k.startswith("mean_") else None)
                if ci is None and k == "boot_width":
                    ci = None
                rows.append({"series": "/".join(path), "metric": k, "value": float(v), "ci_lo": ci[0] if ci else "", "ci_hi": ci[1] if ci else ""})
        for k, v in cis.items():
            base = k[:-3] if k.endswith("_ci") else k
            if not any(r["series"] == "/".join(path) and r["metric"] == base for r in rows):
                rows.append({"series": "/".join(path), "metric": base, "value": "", "ci_lo": v[0], "ci_hi": v[1]})
    elif isinstance(node, list):
        for i, v in enumerate(node):
            label = v.get("pattern") if isinstance(v, dict) and "pattern" in v else str(i)
            Walk(v, path + [label], rows)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()
    now = dt.datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
    out = []
    for spec in a.inputs:
        sport, path = spec.split("=", 1)
        rows = []
        Walk(json.load(open(path)), [], rows)
        for r in rows:
            out.append({"date": now, "sport": sport, **r})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "sport", "series", "metric", "value", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows")


if __name__ == "__main__":
    Main()
