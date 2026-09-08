#!/usr/bin/env python3
"""Run ONE method with constructor-attribute overrides to a SIDE output path.

The benchmark writes every method's canonical file to ``predictions/<method_id>.json``, and
``--method-config`` overrides attributes without changing the id — so a variant run (e.g. the
sampling budget ablation at ``scenarios_per_condition=50``) would clobber the canonical file.
This runner reuses the benchmark's own prediction internals (same split handling, checkpoints,
output schema) but saves wherever ``--out`` points, keeping variants out of the canonical tree.

Examples:
    # budget-scaling ablation (k=50 x r=2) for the appendix robustness note
    .venv/bin/python scripts/run_method_variant.py --method informed_sampling \
        --eval sycophancy_pushback --model llama-3.3-70b \
        --set scenarios_per_condition=50 \
        --out results/_pilot/budget_k50_syc_llama-3.3-70b.json

    # cheap smoke of a new sampling variant (tiny k)
    .venv/bin/python scripts/run_method_variant.py --method protocol_sampling \
        --eval capability_mmlu --model llama-3.3-70b --set scenarios_per_condition=2 \
        --out results/_pilot/protocol_sampling_smoke.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from behavior_prediction import common, elicitation, splits
from behavior_prediction.cli.benchmark import _coerce, _predict_method
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import RunConfig, get_method, valid_combo
from behavior_prediction.models_adapter import ModelAdapter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--eval", required=True, dest="eval_name")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="ATTR=VALUE",
                    help="constructor-attribute override (repeatable), e.g. "
                         "scenarios_per_condition=50")
    ap.add_argument("--out", required=True, help="side output path (required — this runner "
                    "exists to keep variants away from the canonical predictions/ tree)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--include-test", action="store_true")
    args = ap.parse_args()

    spec = get_spec(args.eval_name)
    model = ModelAdapter(args.model)
    method = get_method(args.method)
    overrides = dict(kv.split("=", 1) for kv in args.set)
    for attr, val in overrides.items():
        if not hasattr(method, attr):
            raise SystemExit(f"{args.method} has no attribute {attr!r}")
        setattr(method, attr, _coerce(val))
    if not valid_combo(method, model, spec):
        raise SystemExit(f"{args.method} does not apply to {args.eval_name} / {args.model}")

    tpath = common.default_targets_out(spec, args.model)
    if not Path(tpath).exists():
        raise SystemExit(f"no targets at {tpath}")
    targets = common.load_json(tpath)["targets"]
    manifest = splits.load_manifest(args.eval_name)

    cfg = RunConfig(runs=None, concurrency=args.concurrency)
    cfg.checkpoint_base = args.out
    hp: dict = {}
    label = method.method_id(hp) + "".join(f"+{a}={v}" for a, v in sorted(overrides.items()))
    print(f"[variant] {args.eval_name}/{common.model_slug(args.model)}/{label} -> {args.out}")
    preds = _predict_method(method, spec, model, targets, manifest, hp, cfg,
                            include_test=args.include_test)
    if all(p.get("predicted_rate") is None for p in preds.values()):
        print("  all predictions failed to parse; nothing written")
        return 1
    elicitation._save_outputs(
        method=label, base_method=method.base_method, model=model.full,
        metric=spec.default_metric, predictions=preds, hyperparameters=overrides,
        reasoning_config=model.reasoning_cfg, out=args.out)
    elicitation._cleanup_checkpoints(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
