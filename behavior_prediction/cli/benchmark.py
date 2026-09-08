#!/usr/bin/env python3
"""Benchmark runner: models × methods × evals -> cross-validate on dev, predict, score on test.

This is the top-level entry point. It orchestrates the existing pieces:
- **targets** come from the existing ``run_sweep.py`` + ``extract_targets.py`` (this runner does
  not re-measure behavior; it requires ``results/<eval>/<model>/targets.json`` to exist);
- **predictions** come from the method adapters (``methods/``), which delegate to the
  single-sourced elicitation primitives; conditions are derived from the targets, so every eval
  is predicted at its own grain (PropensityBench roles, DiscrimEval templates);
- **scoring** is the split-aware tuning + evaluation steps (``tuning.py`` picks the best setting per
  method on dev over the selection pool; ``evaluate.py`` scores those frozen settings on a split).

Only (model × method × eval) combos that pass capability negotiation are run; the rest are
skipped and logged. Keep development runs small (few models, small ``--runs``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from behavior_prediction import common
from behavior_prediction import elicitation
from behavior_prediction import evaluate
from behavior_prediction import splits
from behavior_prediction.evals import REGISTRY, get_spec, require_runnable
from behavior_prediction.methods import (DEFAULT_METHODS, REGISTRY as METHODS, RunConfig,
                                         get_method, valid_combo)
from behavior_prediction.models_adapter import ModelAdapter
# Single-sourced from tuning.py (2026-08-17): this module carried its own stale two-eval
# literal, silently limiting every bare `bp-benchmark` to propensitybench+discrimeval —
# masked for months because drivers passed explicit --evals, until the frozen-test run
# relied on the default. Same incident class as 2026-07-24; never redeclare this list.
from behavior_prediction.tuning import ACTIVE_EVALS


def _predict_method(method, spec, model, targets: dict, manifest: dict,
                    hp: dict, cfg: RunConfig, *, include_test: bool) -> dict[str, dict]:
    """All predictions for one (method, hp). By default only the **dev** pool is predicted — the
    frozen **test** split is left untouched during development (pass ``include_test`` for the
    explicit, manual final run on selected settings).

    Stateless methods predict each dev condition once (their predictions don't depend on the
    split). Trained methods are cross-validated: dev-pool conditions get **out-of-fold** predictions
    (fit on the other CV folds, predict the held-out fold); test conditions, when requested, are
    predicted from a single fit on the whole dev pool — so the test set never leaks into fit."""
    dev_keys = splits.split_keys(spec, targets, manifest, "dev")
    test_keys = splits.split_keys(spec, targets, manifest, "test") if include_test else set()
    if not method.trained:
        conds = [targets[k]["condition"] for k in (dev_keys | test_keys)]
        return method.predict(spec, model, conds, hp, cfg, method.fit({}, model, spec, hp))
    preds: dict[str, dict] = {}
    for fit_keys, score_keys in splits.folds(spec, targets, manifest):
        fs = method.fit({k: targets[k] for k in fit_keys}, model, spec, hp)
        fold_conds = [targets[k]["condition"] for k in score_keys]
        preds.update(method.predict(spec, model, fold_conds, hp, cfg, fs))
    if test_keys:
        fs = method.fit({k: targets[k] for k in dev_keys}, model, spec, hp)
        test_conds = [targets[k]["condition"] for k in test_keys]
        preds.update(method.predict(spec, model, test_conds, hp, cfg, fs))
    return preds


def _coerce(v: str):
    """Coerce a --method-config string value to None / int / float / bool / str."""
    low = v.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_method_config(items: list[str]) -> dict[str, dict[str, object]]:
    """Parse repeated ``--method-config name:key=val,key=val`` into ``{name: {key: value}}``."""
    cfg: dict[str, dict[str, object]] = {}
    for item in items or []:
        if ":" not in item:
            raise SystemExit(f"--method-config must be 'name:key=val[,key=val]'; got {item!r}")
        name, kvs = item.split(":", 1)
        for kv in kvs.split(","):
            if not kv.strip():
                continue
            if "=" not in kv:
                raise SystemExit(f"--method-config entry must be key=val; got {kv!r}")
            k, v = kv.split("=", 1)
            cfg.setdefault(name.strip(), {})[k.strip()] = _coerce(v)
    return cfg


def _apply_config(method, conf: dict[str, object]) -> None:
    """Set configured attributes on a method instance (only ones it already declares)."""
    for k, v in conf.items():
        if not hasattr(method, k):
            raise SystemExit(f"{method.base_method!r} has no configurable attribute {k!r}")
        setattr(method, k, v)


def _tuned_grid(method, spec, tuned_best: dict[str, str] | None
                ) -> list[dict] | None:
    """The hyperparameter settings to generate for a method on ``spec``'s eval. Normally its whole
    (eval-pruned) grid; when ``tuned_best`` is given (``--tuned-only``), just the one setting tuning
    picked as best for this eval (so the costly test run only elicits the chosen settings). Returns
    ``None`` to skip the method (it has no tuned best setting yet)."""
    grid = spec.method_grid(method)
    if tuned_best is None:
        return grid
    best_id = tuned_best.get(method.base_method)
    if not best_id:
        return None
    return [hp for hp in grid if method.method_id(hp) == best_id]


def _tuned_best_for(eval_name: str) -> dict[str, str]:
    """``{base_method: best_setting_id}`` from ``results/<eval>/tuning.json`` (errors if absent —
    ``--tuned-only`` requires a prior ``bp-tune`` run)."""
    from behavior_prediction.tuning import tuning_path
    path = tuning_path(eval_name)
    if not path.exists():
        raise SystemExit(f"--tuned-only needs {path}; run `bp-tune --evals {eval_name}` first")
    doc = common.load_json(path)
    return {b: m["best_setting"] for b, m in doc.get("methods", {}).items() if m.get("best_setting")}


def generate(eval_names: list[str], model_names: list[str], method_names: list[str],
             *, concurrency: int, runs: int | None, include_test: bool = False,
             tuned_only: bool = False, fresh: bool = False,
             method_config: dict[str, dict[str, object]] | None = None) -> None:
    cfg = RunConfig(runs=runs, concurrency=concurrency)
    method_config = method_config or {}
    for ev in eval_names:
        spec = get_spec(ev)
        try:
            manifest = splits.load_manifest(ev)
        except SystemExit as e:
            print(f"[skip {ev}] {e}")
            continue
        tuned_best = _tuned_best_for(ev) if tuned_only else None
        for name in model_names:
            model = ModelAdapter(name)
            tpath = common.default_targets_out(spec, name)
            if not Path(tpath).exists():
                print(f"[skip {ev}/{model.slug}] no targets at {tpath} "
                      f"(run run_sweep.py + extract_targets.py first)")
                continue
            targets = common.load_json(tpath)["targets"]
            for mname in method_names:
                method = get_method(mname)
                _apply_config(method, method_config.get(mname, {}))
                if not valid_combo(method, model, spec):
                    print(f"[skip {ev}/{model.slug}/{mname}] capability mismatch")
                    continue
                grid = _tuned_grid(method, spec, tuned_best)
                if grid is None:
                    print(f"[skip {ev}/{model.slug}/{mname}] no tuned best setting in tuning.json")
                    continue
                if not grid:
                    # LOUD: an empty resolved grid means the stored best_setting id no longer
                    # matches any current grid setting (grid drift) — a silent variant of this
                    # skipped work unnoticed during the 2026-08-16 frozen-test run.
                    print(f"[WARN {ev}/{model.slug}/{mname}] tuned best_setting does not "
                          f"resolve against the current grid; SKIPPED — re-run bp-tune")
                    continue
                for hp in grid:
                    mid = method.method_id(hp)
                    scope = "dev+test" if include_test else "dev"
                    out = common.default_pred_out(spec, name, mid)
                    # A completed prediction file is left alone by default — re-running the
                    # benchmark continues unfinished work, it does not redo finished work. Pass
                    # --fresh to regenerate. (An interrupted run has no file yet, only a partial/
                    # checkpoint, so it still resumes below.)
                    if Path(out).exists() and not fresh:
                        if include_test:
                            # FROZEN-TEST TOP-UP (2026-08-16): the final confirmatory run must
                            # add test-split predictions WITHOUT touching the reviewed dev
                            # predictions (a --fresh rerun would resample them). Elicit only the
                            # test conditions missing from the existing file and merge. Trained
                            # methods fit on the full dev pool (the same fit the from-scratch
                            # include_test path uses for test); needs_all_conditions methods
                            # (paired comparisons) compute their matchings WITHIN the test split,
                            # mirroring the dev file's within-split matchings.
                            doc = common.load_json(out)
                            existing = doc.get("predictions", {})
                            test_keys = splits.split_keys(spec, targets, manifest, "test")
                            missing = sorted(k for k in test_keys
                                             if k in targets and k not in existing)
                            if not missing:
                                print(f"[skip] {ev}/{model.slug}/{mid}: test already present")
                                elicitation._cleanup_checkpoints(out)
                                continue
                            print(f"[test-topup] {ev}/{model.slug}/{mid}: "
                                  f"+{len(missing)} test conditions")
                            if method.trained:
                                dev_keys = splits.split_keys(spec, targets, manifest, "dev")
                                fs = method.fit({k: targets[k] for k in dev_keys},
                                                model, spec, hp)
                            else:
                                fs = method.fit({}, model, spec, hp)
                            conds = [targets[k]["condition"] for k in missing]
                            cfg.checkpoint_base = out
                            new_preds = method.predict(spec, model, conds, hp, cfg, fs)
                            added = {k: v for k, v in new_preds.items() if k not in existing}
                            if not added or all(p.get("predicted_rate") is None
                                                for p in added.values()):
                                print("  no parseable test predictions; file left unchanged")
                                continue
                            existing.update(added)
                            elicitation._save_outputs(
                                method=mid, base_method=method.base_method, model=model.full,
                                metric=spec.default_metric, predictions=existing,
                                hyperparameters=hp, reasoning_config=model.reasoning_cfg,
                                out=out)
                            elicitation._cleanup_checkpoints(out)
                            continue
                        print(f"[skip] {ev}/{model.slug}/{mid}: already generated "
                              f"(--fresh to regenerate)")
                        elicitation._cleanup_checkpoints(out)  # drop any stale aborted-regen partial
                        continue
                    print(f"[run] {ev}/{model.slug}/{mid} ({scope})")
                    # Derive the resume sidecar from the out path and hand it to the method via cfg
                    # (a kill mid-run continues on the next identical invocation).
                    cfg.checkpoint_base = out
                    if fresh:
                        elicitation._cleanup_checkpoints(out)
                    preds = _predict_method(method, spec, model, targets, manifest,
                                            hp, cfg, include_test=include_test)
                    if all(p.get("predicted_rate") is None for p in preds.values()):
                        print(f"  all predictions failed to parse; nothing written")
                        continue
                    elicitation._save_outputs(
                        method=mid, base_method=method.base_method, model=model.full,
                        metric=spec.default_metric, predictions=preds, hyperparameters=hp,
                        reasoning_config=model.reasoning_cfg, out=out)
                    elicitation._cleanup_checkpoints(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", default=",".join(ACTIVE_EVALS),
                   help=f"comma-separated eval names (default: {','.join(ACTIVE_EVALS)})")
    p.add_argument("--models", default=None,
                   help="comma-separated --model values (shortcuts or full strings)")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                   help="comma-separated method names (default: all registered except the "
                        "train_scenario_mean demo baseline)")
    p.add_argument("--phase", choices=["generate", "score", "all"], default="all")
    p.add_argument("--split", choices=["dev", "test"], default="dev",
                   help="which split to report (default: dev — keep test frozen during development)")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--runs", type=int, default=None,
                   help="override per-method default runs (keep small in development)")
    p.add_argument("--include-test", action="store_true",
                   help="also elicit/predict the frozen test split (manual final run on selected "
                        "settings; by default only the dev pool is predicted)")
    p.add_argument("--tuned-only", action="store_true",
                   help="generate only each method's tuned best setting (from results/<eval>/"
                        "tuning.json); use with --include-test so the final test run elicits only "
                        "the chosen settings")
    p.add_argument("--force", action="store_true",
                   help="generate even for evals marked broken (pending review / costly)")
    p.add_argument("--fresh", action="store_true",
                   help="regenerate even prediction files that already exist, discarding any resume "
                        "checkpoints (default: skip already-generated files and resume only "
                        "interrupted ones)")
    p.add_argument("--method-config", action="append", default=[], metavar="NAME:K=V,...",
                   help="configure a method instance, e.g. "
                        "'llm_prediction:description_mode=gist,max_train_examples=80' (repeatable)")
    p.add_argument("--out", default="results/reports/evaluation.md")
    args = p.parse_args()

    requested = [e.strip() for e in args.evals.split(",") if e.strip()]
    eval_names = [e for e in requested if e in REGISTRY]
    for e in requested:
        if e not in REGISTRY:
            # LOUD: silently dropping evals hid a frozen-test gap on 2026-08-16.
            print(f"[WARN] --evals entry {e!r} not in the eval registry; SKIPPED")
    method_names = [m.strip() for m in args.methods.split(",") if m.strip() in METHODS]
    method_config = parse_method_config(args.method_config)

    if args.phase in ("generate", "all"):
        if not args.models:
            raise SystemExit("--models is required for the generate phase")
        require_runnable([get_spec(e) for e in eval_names], args.force)
        model_names = [m.strip() for m in args.models.split(",")]
        generate(eval_names, model_names, method_names,
                 concurrency=args.concurrency, runs=args.runs, include_test=args.include_test,
                 tuned_only=args.tuned_only, fresh=args.fresh, method_config=method_config)

    if args.phase in ("score", "all"):
        from behavior_prediction import tuning
        model_slugs = ([common.model_slug(m.strip()) for m in args.models.split(",")]
                       if args.models else None)
        # Tune (pick the best setting per method on dev, aggregated over the selection pool) then
        # score those frozen settings on the chosen split — the two new steps, end-to-end.
        tuning.run(eval_names, models=None, method_names=method_names or None, force=False)
        c = evaluate.write_reports(eval_names, model_slugs, args.split, args.out)
        print(evaluate.render_markdown(c, args.split))
        print(f"Wrote {args.out} + HTML reports under results/reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
