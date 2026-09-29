"""Merge what a staged session learned into the live library: ultraquant.distill.merge's command line.

Usage:
    python tools/merge_session.py STAGING_ROOT LIVE_ROOT --backup-dir DIR [--base DIR] [--check]

The merge lives in the package (§11.171), where the GUI takes a finished
session in itself; --help describes it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultraquant.distill.merge import main  # noqa: E402 - importable once ROOT is on the path

if __name__ == "__main__":
    raise SystemExit(main())
