r"""
Scheduled entry point for the price-capture tasks (Windows Task Scheduler "Footbot-Snapshot-*", approved by Tim
2026-11-13). Runs ONE Backtest.SnapshotPrices snapshot for the week of the next unstarted game, quietly:

    .venv\Scripts\pythonw.exe -m Backtest.SnapshotRunner --label T-90

  * single instance: an exclusive lock on data/logs/snapshots/snapshot.lock (a second run exits at once, logged)
  * no console / popups (pythonw), every outcome and traceback goes to data/logs/snapshots/<YYYY-MM-DD>.log
  * appends rows to data/prices/snapshots_<season>.jsonl (gitignored); read-only HTTP GETs, no orders
Disable: see AGENTS.md ("Price-capture schedule").
"""

import argparse
import io
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

LOG_DIR = os.path.join("data", "logs", "snapshots")


def Log(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now().astimezone()
    with open(os.path.join(LOG_DIR, f"{now:%Y-%m-%d}.log"), "a", encoding="utf-8") as fh:
        fh.write(f"{now.isoformat(timespec='seconds')} {msg}\n")


class Lock:
    def __init__(self, path):
        self.path, self.fh = path, None

    def __enter__(self):
        import msvcrt
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.fh.close()
            self.fh = None
        return self.fh is not None

    def __exit__(self, *exc):
        if self.fh:
            import msvcrt
            try:
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.fh.close()


def Run(label, out_dir=os.path.join("data", "prices")):
    from Backtest.SnapshotPrices import AutoSeasonWeek, Snapshot, WriteOutputs
    season, week = AutoSeasonWeek()
    if week is None:
        Log(f"[{label}] no upcoming games for season {season}; nothing to do")
        return 0
    out = Snapshot(season, week, label)
    base = WriteOutputs(out, out_dir)
    Log(f"[{label}] season {season} week {week}: rows {len(out['rows'])}, upcoming {out['upcoming']}/{out['games_in_week']}, "
        f"errors {len(out['errors'])}, http {out['http_calls']} -> {base}.md")
    for e in out["errors"][:20]:
        Log(f"[{label}]   error {e}")
    return 0


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="scheduled")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sink = io.StringIO()
    code = 1
    with Lock(os.path.join(LOG_DIR, "snapshot.lock")) as got:
        if not got:
            Log(f"[{a.label}] another snapshot is running; skipped")
            return 0
        Log(f"[{a.label}] start (python {sys.version.split()[0]})")
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                code = Run(a.label)
        except Exception:
            Log(f"[{a.label}] FAILED\n{traceback.format_exc()}")
            code = 1
        Log(f"[{a.label}] end code {code}")
    return code


if __name__ == "__main__":
    sys.exit(Main())