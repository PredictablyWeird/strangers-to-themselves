"""Check every behavior_raw.json against its committed targets.json (see rawsync module).

Run:  .venv/bin/python scripts/check_raw_targets_sync.py [--quarantine] [--results DIR]

Exit status 1 when any stale cell remains on disk, 0 otherwise. ``--quarantine`` renames each
stale file to ``behavior_raw.stale.json`` in place (item-level analyses look for
``behavior_raw.json``, so a quarantined run can no longer be joined against targets silently;
the file is kept for run-to-run agreement estimates, e.g. item-level noise ceilings).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from behavior_prediction import rawsync


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--quarantine", action="store_true",
                    help="rename stale behavior_raw.json -> behavior_raw.stale.json")
    args = ap.parse_args()

    reports = rawsync.check_all(args.results)
    stale = [r for r in reports if r.stale]
    for r in reports:
        print(r.line())
    if not reports:
        print("No checkable (eval, model) cells found.")
    if args.quarantine:
        for r in stale:
            dest = r.raw_path.with_name("behavior_raw.stale.json")
            r.raw_path.rename(dest)
            print(f"quarantined: {r.raw_path} -> {dest.name}")
        stale = []
    print(f"\n{len([r for r in reports if not r.stale])} ok, {len(stale)} stale remaining.")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
