#!/usr/bin/env python3
"""Hyperparameter tuning: pick the single best setting per (method, eval), aggregated over the
model selection pool, and persist it.

For each eval and each method, every grid setting (a distinct ``method_id`` with its own cached
predictions) is scored on the **dev** pool of every pool model that has predictions, and the scores
are aggregated (macro-mean of the per-model correlations) into one number per setting. The setting
with the highest aggregate is the method's ``best_setting`` for that eval.

- **Training-free** methods: the dev-pool correlation is the score directly (predictions are
  split-independent).
- **Training-based** methods (``method.trained``): the cached dev predictions are already
  *out-of-fold* (``benchmark._predict_method`` fits on the other CV folds and predicts the held-out
  fold), so the same dev-pool correlation is the cross-validated score — the dev folds were used at
  generation time. The ``criterion`` label records which path a method took.

Result is written to ``results/<eval>/tuning.json``: the chosen ``best_setting`` per method plus a
**per-setting score cache** (keyed by prediction-file mtime per model). On re-run only settings whose
prediction files changed are re-scored, and ``best_setting`` is always re-derived as the argmax over
the merged cache — so expanding a method's grid later only scores the new settings (predictions are
already cached per ``method_id``). ``--force`` rescoring ignores the cache.

This step calls **no models** — it only reads cached ``predictions/*.json`` + ``targets.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from behavior_prediction import common, results_io, selection, splits
from behavior_prediction.evals import REGISTRY as EVALS, get_spec
from behavior_prediction.methods import REGISTRY as METHODS, get_method

#: The benchmark's active eval set — the default for a bare ``bp-tune`` AND (imported by
#: ``evaluate.py``) for a bare ``bp-evaluate``, so the two can never drift apart again.
#: 2026-08-11: ``tau2_transfer`` and ``mask_subdomain_pressure`` promoted to default
#: benchmarks (small-pool predictions + tuning committed; frontier coverage still open).
#: ``discrimeval_implicit`` is benchmarked on the small pool but NOT active (pending decision).
ACTIVE_EVALS = ["sycophancy_pushback", "discrimeval", "capability_mmlu",
                "reward_hacking", "tau2_policy", "tau2_transfer",
                "propensitybench", "mask_subdomain_pressure"]


def _candidate_settings(spec, method) -> dict[str, dict[str, Any]]:
    """``{method_id: hyperparameters}`` for every setting in a method's grid on this eval (the eval
    may prune args that don't apply — see ``EvalSpec.method_grid``)."""
    return {method.method_id(hp): hp for hp in spec.method_grid(method)}


def _pred_path(spec, model_name: str, method_id: str) -> Path:
    return Path(common.default_pred_out(spec, model_name, method_id))


def _donor_tuning(spec, manifest, method_names: list[str] | None) -> dict[str, Any]:
    """Frozen settings for a ``test_only`` eval (no dev pool — nothing to tune on): each
    applicable method's ``best_setting`` is the MODAL tuned best across the other active evals'
    committed tuning docs, restricted to the setting ids valid on this eval; ties and methods
    with no donor vote fall back to grid order (the default setting first). Deterministic and
    computable before a single prediction exists for the eval, so the choice is pre-registered
    on the donors, never selected on the eval's own (test) data."""
    donors = [e for e in ACTIVE_EVALS if e != spec.name]
    donor_best: dict[str, list[str]] = {}
    for ev in donors:
        path = tuning_path(ev)
        if not path.exists():
            continue
        for base, m in common.load_json(path).get("methods", {}).items():
            if m.get("best_setting"):
                donor_best.setdefault(base, []).append(m["best_setting"])
    names = method_names or [n for n in METHODS]
    out_methods: dict[str, Any] = {}
    for base in names:
        method = get_method(base)
        if not method.applies_to(spec):
            continue
        candidates = _candidate_settings(spec, method)          # {mid: hp}, grid order
        order = list(candidates)
        votes = [mid for mid in donor_best.get(base, []) if mid in candidates]
        best_id = (max(dict.fromkeys(votes),
                       key=lambda mid: (votes.count(mid), -order.index(mid)))
                   if votes else order[0])
        out_methods[base] = {
            "trained": bool(method.trained),
            "criterion": "donor_modal_best",
            "best_setting": best_id,
            "best_args": candidates[best_id],
            "donor_votes": {mid: votes.count(mid) for mid in dict.fromkeys(votes)},
            "settings": {mid: {"args": hp, "agg_r": None, "n_models": 0, "per_model": {}}
                         for mid, hp in candidates.items()},
        }
    return {
        "eval": spec.name,
        "split": "test",
        "selection_pool": [],
        "donor_evals": donors,
        "manifest_seed": manifest.get("seed", 1234),
        "methods": out_methods,
    }


def tune_eval(eval_name: str, *, models: list[str] | None = None,
              method_names: list[str] | None = None, force: bool = False,
              prior: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the tuning doc for one eval. ``prior`` is the previous ``tuning.json`` (its cached
    per-model scores are reused when the prediction file mtime is unchanged and not ``force``).
    A ``test_only`` eval takes the donor path instead (:func:`_donor_tuning`)."""
    spec = get_spec(eval_name)
    manifest = splits.load_manifest(eval_name)
    if spec.test_only:
        return _donor_tuning(spec, manifest, method_names)
    pool = models or common.selection_pool()
    prior_methods = (prior or {}).get("methods", {})

    # Load each pool model's cell once: (targets, {method_id: {key: rate}}) + dev keys.
    cells: dict[str, tuple[dict[str, Any], dict[str, dict[str, float | None]], set[str]]] = {}
    for m in pool:
        slug = common.model_slug(m)
        loaded = results_io.load_cell(eval_name, slug)
        if not loaded:
            print(f"[tune {eval_name}] no cell for pool model {m} ({slug}); skipping")
            continue
        targets, preds = loaded
        try:
            dev_keys = splits.split_keys(spec, targets, manifest, "dev")
        except KeyError:
            print(f"[tune {eval_name}] {m}: targets incompatible with split_unit; skipping")
            continue
        cells[m] = (targets, preds, dev_keys)

    names = method_names or [n for n in METHODS]
    out_methods: dict[str, Any] = {}
    for base in names:
        method = get_method(base)
        if not method.applies_to(spec):
            continue
        prior_settings = (prior_methods.get(base, {}) or {}).get("settings", {})
        settings_out: dict[str, Any] = {}
        for mid, hp in _candidate_settings(spec, method).items():
            prior_pm = (prior_settings.get(mid, {}) or {}).get("per_model", {})
            per_model: dict[str, Any] = {}
            for m, (targets, preds, dev_keys) in cells.items():
                path = _pred_path(spec, m, mid)
                if not path.exists():
                    continue
                mtime = path.stat().st_mtime
                cached = prior_pm.get(m)
                if not force and cached and cached.get("mtime") == mtime:
                    per_model[m] = cached
                    continue
                r = selection._corr(spec, targets, preds.get(mid, {}), dev_keys)
                per_model[m] = {"r": r, "mtime": mtime}
            rs = [v["r"] for v in per_model.values() if v.get("r") is not None]
            settings_out[mid] = {
                "args": hp,
                "agg_r": (mean(rs) if rs else None),
                "n_models": len(rs),
                "per_model": per_model,   # {model: {"r": ..., "mtime": ...}} — mtime feeds the cache
            }
        # Re-derive the best setting as the argmax over the (merged) cache, in grid order so a tie
        # keeps the earlier/default setting.
        best_id, best_args, best_r = None, None, None
        for mid, s in settings_out.items():
            if s["agg_r"] is None:
                continue
            if best_r is None or s["agg_r"] > best_r:
                best_id, best_args, best_r = mid, s["args"], s["agg_r"]
        out_methods[base] = {
            "trained": bool(method.trained),
            "criterion": "dev_cv_macro_r" if method.trained else "dev_pool_macro_r",
            "best_setting": best_id,
            "best_args": best_args,
            "settings": settings_out,
        }

    return {
        "eval": eval_name,
        "split": "dev",
        "selection_pool": pool,
        "manifest_seed": manifest.get("seed", 1234),
        "methods": out_methods,
    }


def tuning_path(eval_name: str) -> Path:
    return Path(f"{common.results_root()}/{eval_name}/tuning.json")


def _example_conditions(spec, manifest, pool: list[str], n: int = 2
                        ) -> list[tuple[str, dict]]:
    """A few representative dev conditions (label, condition) for the tuning report's prompt
    exhibits — taken from the first pool model that has targets."""
    for m in pool:
        loaded = results_io.load_cell(spec.name, common.model_slug(m))
        if not loaded:
            continue
        targets, _ = loaded
        dev = sorted(splits.split_keys(spec, targets, manifest, "dev"))
        if dev:
            return [(spec.condition_label(targets[k]["condition"]), targets[k]["condition"])
                    for k in dev[:n]]
    return []


def run(eval_names: list[str], *, models: list[str] | None, method_names: list[str] | None,
        force: bool) -> None:
    for ev in eval_names:
        path = tuning_path(ev)
        prior = common.load_json(path) if path.exists() else None
        try:
            doc = tune_eval(ev, models=models, method_names=method_names, force=force, prior=prior)
        except SystemExit as e:
            print(f"[skip {ev}] {e}")
            continue
        common.save_json(doc, path)
        print(f"\n[{ev}] best settings (criterion in parens):")
        for base, m in doc["methods"].items():
            best = m["best_setting"]
            r = m["settings"].get(best, {}).get("agg_r") if best else None
            rs = f"{r:+.3f}" if isinstance(r, float) else "—"
            print(f"  {base:20s} -> {best or '(no data)':35s} {rs}  ({m['criterion']})")
        print(f"Wrote {path}")
        # HTML tuning report: per method, each setting's args + dev correlation + rendered prompt.
        from behavior_prediction import report_html
        spec = get_spec(ev)
        examples = _example_conditions(spec, splits.load_manifest(ev), doc["selection_pool"])
        html_out = f"{common.results_root()}/reports/tuning-{ev}.html"
        report_html.build_tuning_report(spec, doc, examples, html_out)
        print(f"Wrote {html_out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", default=",".join(ACTIVE_EVALS),
                   help=f"comma-separated eval names (default: {','.join(ACTIVE_EVALS)})")
    p.add_argument("--models", default=None,
                   help="comma-separated pool override (default: methods.yaml selection_pool)")
    p.add_argument("--methods", default=None,
                   help="comma-separated method names to tune (default: all applicable)")
    p.add_argument("--force", action="store_true", help="rescore every setting (ignore the cache)")
    args = p.parse_args()

    eval_names = [e.strip() for e in args.evals.split(",") if e.strip() in EVALS]
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    method_names = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run(eval_names, models=models, method_names=method_names, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
