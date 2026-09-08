"""Development-time tuning reports: per-setting dev correlations (A) and selection-CV (B).

A method *family* is a base method plus the grid of settings it was run with; each setting is a
distinct ``method_id`` (e.g. ``self_report`` vs ``self_report-honest``) with its own cached
prediction file. Splitting is owned by the eval geometry (``splits`` / ``EvalSpec``); these reports
only consume cached *per-key* predictions, so for stateless methods they are exact arithmetic with
no re-elicitation.

- **(A) ``per_setting_dev``** — each setting's correlation over the whole dev pool. Use it to see
  how every grid setting does on dev (the "10 correlations" view).
- **(B) ``selection_cv``** — k-fold over the dev pool; each fold picks the best candidate setting on
  the *train* part and predicts the *held-out* fold; the held-out predictions are pooled and
  correlated once — an honest estimate of what "select-best-on-dev" actually buys.

For stateless methods predictions are split-independent, so (B) is exact. For trained methods the
cached predictions are the out-of-fold dev predictions; selection over them is a reasonable estimate
but the per-fold *selection* step is not fully nested (a true nested CV would re-fit per outer fold,
which belongs to the trained/probe phase). ``selection_cv`` flags this via ``exact``.
"""

from __future__ import annotations

from typing import Any

from behavior_prediction import metrics, splits
from behavior_prediction.evals.base import EvalSpec


def _corr(spec: EvalSpec, targets: dict[str, Any], preds: dict[str, float | None],
          keys: set[str]) -> float | None:
    """Correlation of ``preds`` vs actual over ``keys`` (the shared scoring used by tuning/evaluate:
    contrast semantics for bias evals, 0.0 for a constant predictor, None when no overlap)."""
    if spec.scoring_semantics == "bias_contrast":
        ps = metrics.contrast_pairs(spec.score_contrasts(targets, preds, keys=keys))
    else:
        ps = metrics.pairs(targets, preds, keys)
    if not ps:
        return None
    r = metrics.pearson(ps)
    return 0.0 if r is None else r


def family_of(method_id: str) -> str:
    """The base method a setting belongs to (``self_report-honest`` -> ``self_report``)."""
    return method_id.split("-")[0]


def families(method_ids) -> dict[str, list[str]]:
    """Group setting ids by base method, each list sorted."""
    out: dict[str, list[str]] = {}
    for m in sorted(method_ids):
        out.setdefault(family_of(m), []).append(m)
    return out


def per_setting_dev(spec: EvalSpec, targets: dict[str, Any],
                    methods: dict[str, dict[str, float | None]],
                    dev_keys: set[str]) -> dict[str, float | None]:
    """(A) ``{method_id: dev correlation}`` over the whole dev pool, for every setting with data."""
    return {mid: _corr(spec, targets, preds, dev_keys) for mid, preds in methods.items()}


def selection_cv(spec: EvalSpec, targets: dict[str, Any],
                 methods: dict[str, dict[str, float | None]], manifest: dict[str, Any],
                 *, candidates: list[str] | None = None) -> dict[str, Any] | None:
    """(B) Pooled selection-CV over the dev pool for a candidate set of settings.

    k-fold over dev units (``splits.cv_folds``); each fold selects the candidate with the best
    correlation on the train part (the rest of dev) and predicts the held-out fold; held-out
    predictions are pooled and correlated once. Returns ``{"pooled_r", "picks", "n_folds"}`` or
    ``None`` when there is too little data / fewer than two folds.
    """
    cands = [c for c in (candidates if candidates is not None else list(methods)) if c in methods]
    if not cands:
        return None
    dev_keys = splits.split_keys(spec, targets, manifest, "dev")
    if not dev_keys:
        return None
    unit_of = {k: spec.split_unit(targets[k]["condition"]) for k in dev_keys}
    dev_units = sorted(set(unit_of.values()))
    folds = splits.cv_folds(spec, dev_units, seed=manifest.get("seed", 1234))
    if len(folds) < 2:
        return None
    bias = spec.scoring_semantics == "bias_contrast"
    combined: dict[str, float | None] = {}              # absolute-rate path: pooled per-key preds
    pooled_pairs: list[tuple[float, float]] = []        # bias path: pooled (actual, predicted) gaps
    picks: list[str] = []
    for held in folds:
        held_set = set(held)
        held_keys = {k for k in dev_keys if unit_of[k] in held_set}
        train_keys = dev_keys - held_keys
        scored = [(c, _corr(spec, targets, methods[c], train_keys)) for c in cands]
        scored = [(c, r) for c, r in scored if r is not None]
        if not scored or not held_keys:
            continue
        best = max(scored, key=lambda cr: cr[1])[0]
        picks.append(best)
        if bias:
            # Pool the chosen method's contrast pairs for this held fold. Contrasts are disjoint
            # across folds (split is by unit), so this is keying-agnostic: score_contrasts uses the
            # direct contrast key or differences cells, whichever the chosen method provides.
            pooled_pairs.extend(
                metrics.contrast_pairs(spec.score_contrasts(targets, methods[best], keys=held_keys)))
        else:
            for k in held_keys:
                combined[k] = methods[best].get(k)
    if bias:
        if not pooled_pairs:
            return None
        r = metrics.pearson(pooled_pairs)
        return {"pooled_r": 0.0 if r is None else r, "picks": picks, "n_folds": len(picks)}
    if not combined:
        return None
    return {"pooled_r": _corr(spec, targets, combined, set(combined)),
            "picks": picks, "n_folds": len(picks)}
