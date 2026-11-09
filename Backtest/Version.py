"""
One place for the version stamp on every output.

    model_version = "<frozen config name>@<config hash>+code:<git short hash>[-dirty]"
    run_id        = "<YYYYMMDDTHHMMSS ET>-<6 hex>"   (unique per invocation)

PredictDay and AssessLive put both into every JSON/JSONL/MD they write, so any number can be traced
back to the exact rules and code that made it.
"""

import os
import subprocess
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from Backtest.FrozenConfig import FROZEN, ConfigHash

ET = ZoneInfo("America/New_York")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = None


def CodeVersion() -> str:
    """git short hash of HEAD, with -dirty if tracked files differ. 'nogit' if git is unavailable."""
    global _CODE
    if _CODE is None:
        try:
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            _CODE = (head or "nogit") + ("-dirty" if head and dirty else "")
        except Exception:
            _CODE = "nogit"
    return _CODE


def ModelVersion() -> str:
    return f"{FROZEN['name']}@{ConfigHash()}+code:{CodeVersion()}"


def NewRunId(now=None) -> str:
    now = (now or datetime.now(ET)).astimezone(ET)
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
