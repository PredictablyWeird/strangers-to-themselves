#!/usr/bin/env python3
"""PILOT (2026-08-10): optimize n_exemplars for informed_sampling on sycophancy + rh.

Runs llama, k=25 r=2, contract v3, n in {3, 5, 10}; writes predictions to results/_pilot/
(canonical files untouched) and prints dev r per arm. Exemplars are the only difficulty/
temptation anchor (explicit difficulty instructions were rejected), so this measures how many
anchors the generator needs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import RunConfig
from behavior_prediction.methods.pilot import InformedSampling
from behavior_prediction.models_adapter import ModelAdapter

ARMS = [3, 5, 10]
EVALS = ["sycophancy_pushback", "reward_hacking"]
MODEL = "llama-3.3-70b"


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y)) ** 0.5
    return float("nan") if sx == 0 or sy == 0 else \
        sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=30)
    args = ap.parse_args()
    model = ModelAdapter(MODEL)
    out_dir = Path("results/_pilot")
    for ev in EVALS:
        spec = get_spec(ev)
        manifest = splits.load_manifest(ev)
        targets = common.load_json(f"results/{ev}/{model.slug}/targets.json")["targets"]
        dev = splits.split_keys(spec, targets, manifest, "dev")
        tr = {k: targets[k]["rate"] for k in dev if targets[k].get("rate") is not None}
        conds = [targets[k]["condition"] for k in dev]
        for n in ARMS:
            out = out_dir / f"nexemp_{ev}_{model.slug}_n{n}.json"
            if out.exists():
                preds = json.loads(out.read_text())
            else:
                m = InformedSampling(n_exemplars=n, scenarios_per_condition=25,
                                     samples_per_scenario=2)
                cfg = RunConfig(concurrency=args.concurrency)
                cfg.checkpoint_base = str(out)
                preds = m.predict(spec, model, conds, {}, cfg)
                for e in preds.values():   # keep the file lean
                    e.pop("raw", None), e.pop("reasoning", None), e.pop("transcripts", None)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(preds, indent=1, default=str))
            pr = {k: v.get("predicted_rate") for k, v in preds.items()
                  if k in tr and v.get("predicted_rate") is not None}
            ks = sorted(pr)
            r = _pearson([pr[k] for k in ks], [tr[k] for k in ks])
            mean_ = sum(pr.values()) / len(pr) if pr else float("nan")
            print(f"{ev:22s} n={n:2d}  r={r:+.3f}  sampled mean={mean_:.3f}  "
                  f"(target mean {sum(tr.values())/len(tr):.3f}, {len(ks)} conds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
