#!/usr/bin/env python3
"""One-off migration: fold the standalone ``third_person`` method into ``self_report`` as the
``perspective=third_person`` argument.

Renames every on-disk ``third_person*`` artifact to the new ``self_report-third_person*`` method id
and rewrites its in-file fields:

  predictions/third_person.json         -> predictions/self_report-third_person.json
  predictions/third_person-honest.json  -> predictions/self_report-third_person-honest.json
  reasoning/third_person*.json          -> reasoning/self_report-third_person*.json

Prediction files also get ``base_method: self_report`` and ``perspective: third_person`` added to
``hyperparameters`` (preserving ``honesty_nudge``); reasoning files only carry ``method``.

Idempotent: a file whose target already exists is skipped. Use ``--dry-run`` to preview.

    python scripts/migrate_third_person.py --dry-run
    python scripts/migrate_third_person.py                 # all evals
    python scripts/migrate_third_person.py --eval discrimeval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_prediction import common


def _new_id(old_id: str) -> str:
    """``third_person`` -> ``self_report-third_person``; ``third_person-honest`` ->
    ``self_report-third_person-honest``."""
    assert old_id.startswith("third_person"), old_id
    return "self_report-" + old_id


def _new_hyperparameters(old_id: str) -> dict[str, object]:
    hp: dict[str, object] = {"perspective": "third_person"}
    if old_id.endswith("-honest"):
        hp["honesty_nudge"] = True
    return hp


def _migrate_prediction(path: Path, *, dry_run: bool) -> None:
    old_id = path.stem
    new_id = _new_id(old_id)
    target = path.with_name(new_id + ".json")
    hp = _new_hyperparameters(old_id)
    # Cross-check the new id is exactly what the YAML-driven method_id derives.
    assert common.method_id("self_report", hp) == new_id, (new_id, hp)
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    doc = json.loads(path.read_text())
    doc["method"] = new_id
    doc["base_method"] = "self_report"
    doc["hyperparameters"] = {**(doc.get("hyperparameters") or {}), **hp}
    print(f"[pred] {path}  ->  {target}")
    if dry_run:
        return
    common.save_json(doc, target)
    path.unlink()


def _migrate_reasoning(path: Path, *, dry_run: bool) -> None:
    old_id = path.stem
    new_id = _new_id(old_id)
    target = path.with_name(new_id + ".json")
    if target.exists():
        print(f"[skip] {target} already exists")
        return
    doc = json.loads(path.read_text())
    if "method" in doc:
        doc["method"] = new_id
    print(f"[reas] {path}  ->  {target}")
    if dry_run:
        return
    common.save_json(doc, target)
    path.unlink()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default=None,
                   help="restrict to one eval (default: every eval under the results root)")
    p.add_argument("--dry-run", action="store_true", help="print planned renames, change nothing")
    args = p.parse_args()

    root = common.results_root()
    base = root / args.eval if args.eval else root
    preds = sorted(base.glob("**/predictions/third_person*.json"))
    reas = sorted(base.glob("**/reasoning/third_person*.json"))
    if not preds and not reas:
        print(f"No third_person* files under {base}")
        return 0
    for f in preds:
        _migrate_prediction(f, dry_run=args.dry_run)
    for f in reas:
        _migrate_reasoning(f, dry_run=args.dry_run)
    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} "
          f"{len(preds)} prediction + {len(reas)} reasoning file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
