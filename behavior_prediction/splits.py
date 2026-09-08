#!/usr/bin/env python3
"""Dev/test splits for the benchmark — keyed by *split unit*, hand-editable, frozen.

The split discipline is what lets the benchmark compare many methods (and methods with fitted
parameters) without selection bias: fitting and hyperparameter/method selection happen on a
**dev** development pool via cross-validation; the held-out **test** split is scored once.
Splits are **per eval** and keyed by the eval's *split unit* (``spec.split_unit(cond)`` — e.g. a
PropensityBench workspace or a DiscrimEval decision template), so a unit is never spread across
dev and test.

Cross-validation within the dev pool holds out whole *fold groups* at a time
(``spec.fold_group(unit)``): leave-one-group-out when the number of groups already lands in
``[MIN_FOLDS, MAX_FOLDS]`` (PropensityBench: one fold per scenario → 3 folds), else groups are
binned to ``MAX_FOLDS`` folds (DiscrimEval: 42 templates → 5 folds), or units are binned to
``MIN_FOLDS`` folds when there are too few groups. A trainable method is fit on the other folds
and predicts the held-out fold (out-of-fold), so its dev correlation is honestly
cross-validated; stateless methods are fold-independent.

A manifest is plain JSON at ``results/<eval>/splits.json`` and is the **hand-editable source
of truth** — inspect it with ``python splits.py --eval <eval>``, tweak by hand, re-run. The
benchmark never silently regenerates it; ``--build`` does so explicitly.

Manifest shape::

    {
      "eval": "propensitybench",
      "split_unit_kind": "workspace",     # doc only (spec.scenario_granularity)
      "seed": 1234,
      "fractions": {"dev": 0.6, "test": 0.4},
      "assignments": {"<split_unit_id>": "dev"|"test", ...},
      "forced": {"test": ["<split_unit_id>", ...]}   # provenance for forced placements
    }
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from behavior_prediction import common
from behavior_prediction.evals import get_spec
from behavior_prediction.evals.base import EvalSpec

SPLITS = ("dev", "test")
MIN_FOLDS, MAX_FOLDS = 3, 5


def manifest_path(eval_name: str) -> str:
    return f"{common.results_root()}/{eval_name}/splits.json"


def load_manifest(eval_name: str) -> dict[str, Any]:
    p = manifest_path(eval_name)
    if not Path(p).exists():
        raise SystemExit(f"no split manifest at {p}; build one with "
                         f"`python splits.py --eval {eval_name} --build`")
    return common.load_json(p)


# --- Applying a manifest -------------------------------------------------------

def assignment(spec: EvalSpec, cond: dict[str, Any],
               manifest: dict[str, Any]) -> str | None:
    """The split ("dev"/"test") a condition belongs to, or None if its split unit is
    unassigned. ``cond`` is available at every join site as ``targets[k]["condition"]``."""
    return manifest.get("assignments", {}).get(spec.split_unit(cond))


def split_keys(spec: EvalSpec, targets: dict[str, Any], manifest: dict[str, Any],
               which: str) -> set[str]:
    """Condition keys in ``targets`` whose split unit is assigned to split ``which``."""
    return {k for k, t in targets.items()
            if assignment(spec, t["condition"], manifest) == which}


def filter_targets(spec: EvalSpec, targets: dict[str, Any], manifest: dict[str, Any],
                   which: str) -> dict[str, Any]:
    """A ``targets``-shaped subset restricted to split ``which``."""
    keep = split_keys(spec, targets, manifest, which)
    return {k: v for k, v in targets.items() if k in keep}


def cv_folds(spec: EvalSpec, dev_units: list[str], *, seed: int = 1234) -> list[list[str]]:
    """Cross-validation folds over the dev pool, as a list of held-out split-unit lists.

    Units are first bucketed by ``spec.fold_group(unit)``. Then:

    - ``MIN_FOLDS <= #groups <= MAX_FOLDS`` → **leave-one-group-out** (one fold per group);
    - ``#groups > MAX_FOLDS`` → groups binned into ``MAX_FOLDS`` size-balanced folds (groups
      stay intact — a group is never split across folds);
    - ``#groups < MIN_FOLDS`` → fall back to binning individual units into ``MIN_FOLDS`` folds.

    Deterministic given ``seed``. The union of the returned folds is exactly ``dev_units``.
    """
    groups: dict[str, list[str]] = {}
    for u in dev_units:
        groups.setdefault(spec.fold_group(u), []).append(u)
    keys = sorted(groups)
    if MIN_FOLDS <= len(keys) <= MAX_FOLDS:
        return [sorted(groups[g]) for g in keys]
    if len(keys) > MAX_FOLDS:
        bins: list[list[str]] = [[] for _ in range(MAX_FOLDS)]
        sizes = [0] * MAX_FOLDS
        for g in sorted(keys, key=lambda g: (-len(groups[g]), g)):  # largest group first
            i = min(range(MAX_FOLDS), key=lambda j: (sizes[j], j))  # into the smallest bin
            bins[i].extend(groups[g])
            sizes[i] += len(groups[g])
        return [sorted(b) for b in bins if b]
    units = sorted(dev_units)
    random.Random(seed).shuffle(units)
    k = min(MIN_FOLDS, len(units))
    return [sorted(units[i::k]) for i in range(k)] if k else []


def folds(spec: EvalSpec, targets: dict[str, Any],
          manifest: dict[str, Any]) -> list[tuple[set[str], set[str]]]:
    """Out-of-fold CV folds over the dev pool as ``(fit_keys, score_keys)`` pairs (condition
    keys in ``targets``): each pair fits on the rest of the dev pool and scores the held-out
    fold. ``test`` is excluded — it is fit on the whole dev pool and scored separately.
    """
    dev_keys = split_keys(spec, targets, manifest, "dev")
    unit_of = {k: spec.split_unit(targets[k]["condition"]) for k in dev_keys}
    dev_units = sorted(set(unit_of.values()))
    out: list[tuple[set[str], set[str]]] = []
    for held in cv_folds(spec, dev_units, seed=manifest.get("seed", 1234)):
        held_set = set(held)
        score = {k for k in dev_keys if unit_of[k] in held_set}
        out.append((dev_keys - score, score))
    return out


# --- Building a manifest -------------------------------------------------------

def units_from_targets(spec: EvalSpec, targets: dict[str, Any]) -> list[str]:
    """Distinct split units present in a targets doc, in stable (sorted) order."""
    return sorted({spec.split_unit(t["condition"]) for t in targets.values()})


def build_manifest(spec: EvalSpec, units: list[str], *, seed: int = 1234,
                   fractions: tuple[float, float] = (0.6, 0.4)) -> dict[str, Any]:
    """Seeded dev/test assignment over ``units``.

    Units the eval forces (``spec.forced_split(unit)`` — e.g. every cyber-security workspace
    into test) are placed first; the rest are shuffled deterministically and partitioned by
    ``fractions`` into the dev pool and the held-out test set. Cross-validation folds within the
    dev pool are derived at scoring time (``cv_folds``), not stored here.
    """
    assignments: dict[str, str] = {}
    forced: dict[str, list[str]] = {}
    for u in units:
        split = spec.forced_split(u)
        if split is not None:
            split = "dev" if split == "train" else split  # legacy: train folded into the dev pool
            assignments[u] = split
            forced.setdefault(split, []).append(u)
    rest = [u for u in units if u not in assignments]
    random.Random(seed).shuffle(rest)
    f_dev, _ = fractions
    n_dev = round(len(rest) * f_dev)
    for i, u in enumerate(rest):
        assignments[u] = "dev" if i < n_dev else "test"
    return {
        "eval": spec.name,
        "split_unit_kind": spec.scenario_granularity,
        "seed": seed,
        "fractions": dict(zip(SPLITS, fractions)),
        "forced": {k: sorted(v) for k, v in sorted(forced.items())},
        "assignments": dict(sorted(assignments.items())),
    }


# --- CLI: inspect / build ------------------------------------------------------

def _find_targets(spec: EvalSpec, override: str | None) -> dict[str, Any]:
    """Load a targets doc to enumerate split units. Splits are model-independent, so any
    model's targets gives the same condition set. Pick the newest targets file whose conditions
    are compatible with the eval's current ``split_unit``."""
    if override:
        return common.load_json(override)["targets"]
    hits = sorted(Path(f"{common.results_root()}/{spec.name}").glob("*/targets.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise SystemExit(f"no results/{spec.name}/*/targets.json found; pass --targets PATH")
    for p in hits:
        targets = common.load_json(p)["targets"]
        try:
            for t in targets.values():
                spec.split_unit(t["condition"])
            return targets
        except KeyError:
            continue  # incompatible grain for this eval's split_unit; try the next file
    raise SystemExit(f"no compatible targets under results/{spec.name}/ for split_unit "
                     f"({spec.scenario_granularity}); re-run extract_targets.py")


def summarize(spec: EvalSpec, manifest: dict[str, Any]) -> str:
    """Readable table of the split assignment for manual review."""
    assignments = manifest.get("assignments", {})
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for unit, split in sorted(assignments.items()):
        by_split.setdefault(split, []).append(unit)
    lines = [f"# {spec.name} split ({manifest.get('split_unit_kind')}, "
             f"seed={manifest.get('seed')})"]
    forced = manifest.get("forced", {})
    if forced:
        lines.append(f"forced: {forced}")
    total = len(assignments)
    for split in SPLITS:
        units = by_split.get(split, [])
        pct = f"{100 * len(units) / total:.0f}%" if total else "0%"
        lines.append(f"\n## {split}: {len(units)} units ({pct})")
        for u in units:
            lines.append(f"  {u}")
    dev_units = by_split.get("dev", [])
    cvf = cv_folds(spec, dev_units, seed=manifest.get("seed", 1234))
    lines.append(f"\n## CV folds over dev ({len(cvf)}, leave-group-out by fold_group)")
    for i, held in enumerate(cvf):
        lines.append(f"  fold {i}: {len(held)} held-out — {', '.join(held)}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True, help="eval name (key in evals.REGISTRY)")
    p.add_argument("--build", action="store_true",
                   help="(re)generate the manifest from a targets doc (otherwise just summarize)")
    p.add_argument("--targets", default=None,
                   help="targets.json to enumerate split units from (default: first under results/<eval>/)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fractions", default="0.6,0.4",
                   help="dev,test fractions of the non-forced units (default: 0.6,0.4)")
    args = p.parse_args()

    spec = get_spec(args.eval)
    if args.build:
        # Prefer the eval's canonical, model-independent unit universe; fall back to the units
        # present in a measured targets doc only if the eval can't enumerate them.
        units = spec.split_units()
        if units is None:
            units = units_from_targets(spec, _find_targets(spec, args.targets))
            print("(canonical split_units() unavailable; built from measured targets)")
        fractions = tuple(float(x) for x in args.fractions.split(","))
        manifest = build_manifest(spec, units, seed=args.seed, fractions=fractions)
        common.save_json(manifest, manifest_path(args.eval))
        print(f"wrote {manifest_path(args.eval)} ({len(units)} units)\n")
    else:
        manifest = load_manifest(args.eval)
    print(summarize(spec, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
