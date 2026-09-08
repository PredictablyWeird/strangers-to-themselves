"""Shared results-IO + aggregation helpers used by the tuning and evaluation steps.

Loads the frozen ``targets.json`` + per-method ``predictions/*.json`` for a (eval, model) cell,
enforcing model-identity (predictions run against a different model/reasoning than the targets are
skipped, never silently mixed). Lifted out of the retired ``leaderboard`` so both ``tuning`` and
``evaluate`` read cells the same way.
"""

from __future__ import annotations

import random
from pathlib import Path
from statistics import mean
from typing import Any

from behavior_prediction import common


def load_cell(eval_name: str, model_slug: str
              ) -> tuple[dict[str, Any], dict[str, dict[str, float | None]]] | None:
    """(targets, {method_id: {key: predicted_rate}}) for one (eval, model), or None if absent.

    Predictions whose stored model identity differs from the targets' (different model / reasoning)
    are skipped and logged — never mixed into the cell."""
    tpath = Path(f"{common.results_root()}/{eval_name}/{model_slug}/targets.json")
    if not tpath.exists():
        return None
    tdoc = common.load_json(tpath)
    targets = tdoc["targets"]
    tident = common.identity_of(tdoc)
    methods: dict[str, dict[str, float | None]] = {}
    pred_dir = Path(f"{common.results_root()}/{eval_name}/{model_slug}/predictions")
    for pf in sorted(pred_dir.glob("*.json")):
        doc = common.load_json(pf)
        bad = common.identity_mismatch(tident, common.identity_of(doc))
        if bad:
            print(f"[skip {eval_name}/{model_slug}/{pf.stem}] {bad} identity differs from targets "
                  f"(predictions and targets were run with different {bad}); not mixing")
            continue
        methods[doc.get("method", pf.stem)] = {
            k: v.get("predicted_rate") for k, v in doc["predictions"].items()}
    return targets, methods


def discover_models(eval_name: str) -> list[str]:
    """Model slugs with a ``targets.json`` under an eval's results dir."""
    return sorted(p.parent.name
                  for p in Path(f"{common.results_root()}/{eval_name}").glob("*/targets.json"))


def bootstrap_cells(vals: list[float], *, n: int = 1000, alpha: float = 0.05,
                    seed: int = 0) -> tuple[float, float] | None:
    """Percentile CI for the mean of per-cell correlations (resamples cells)."""
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(mean([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(n))
    return means[int(alpha / 2 * n)], means[min(n - 1, int((1 - alpha / 2) * n))]
