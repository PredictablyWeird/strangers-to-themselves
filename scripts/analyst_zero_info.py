"""Analyst zero-information baseline (analysis 2 of the analyst-gain set).

The paper's fixed analyst (few_shot_other / llm_prediction's predictor, Claude Sonnet 4) only ever
answered WITH the subject's history. To compare information GAINS — subject: few_shot - self_report
vs analyst: few_shot_other - (analyst asked the same abstract question about the subject by name)
— the analyst needs the no-history cell. This runs ``other_report_matrix.named_report_prompt``
with the analyst as predictor about every pool subject, stored alongside the pool matrix as
``results/<eval>/_other_report/<analyst-slug>__about__<B>.json``, then prints the gain table.

    python scripts/analyst_zero_info.py elicit [--evals ...] [--concurrency N]
    python scripts/analyst_zero_info.py analyze [--evals ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from behavior_prediction import common, elicitation, metrics, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods.trained import LlmPrediction
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from other_report_matrix import RUNS, dev_conditions, named_report_prompt  # noqa: E402
from fewshot_analyst_matrix import POOL, out_path as fewshot_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANALYST = LlmPrediction.DEFAULT_PREDICTOR
ANALYST_SLUG = "analyst-" + ANALYST.split("/")[-1]
EVALS = ["capability_mmlu", "propensitybench", "sycophancy_pushback", "reward_hacking"]


def out_path(ev: str, b: str) -> Path:
    return ROOT / f"results/{ev}/_other_report/{ANALYST_SLUG}__about__{b}.json"


def elicit(evals: list[str], concurrency: int) -> None:
    a_full, a_reasoning = common.resolve_model(ANALYST)
    for ev in evals:
        spec = get_spec(ev)
        for b in POOL:
            out = out_path(ev, b)
            if out.exists():
                print(f"skip (exists): {out.relative_to(ROOT)}")
                continue
            b_full, _ = common.resolve_model(b)
            b_name = common.model_display_name(b_full)
            conds, _ = dev_conditions(spec, ev, b)
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"[analyst report] {ev}: {ANALYST} about {b_name} ({len(conds)} conds)", flush=True)
            raw = elicitation.elicit_over_conditions(
                spec, a_full, conds, named_report_prompt(b_name, self_arm=False),
                runs=RUNS, temperature=1.0, max_parse_retries=2, concurrency=concurrency,
                parse_fn=common.parse_percentage, reasoning_config=a_reasoning,
                checkpoint_path=str(out.with_suffix(".partial.jsonl")))
            preds = {k: {kk: vv for kk, vv in e.items() if kk != "condition"} for k, e in raw.items()}
            common.save_json({"method": "other_report", "predictor": ANALYST, "subject": b,
                              "subject_name": b_name, "eval": ev, "runs": RUNS,
                              "predictions": preds}, out)


def _preds(path: Path) -> dict:
    if not path.exists():
        return {}
    return {k: e.get("predicted_rate") for k, e in common.load_json(path)["predictions"].items()}


def _score(spec, ev, subject, preds):
    if not preds:
        return None
    targets = common.load_json(ROOT / f"results/{ev}/{subject}/targets.json")["targets"]
    dev = splits.split_keys(spec, targets, splits.load_manifest(ev), "dev")
    return metrics.corr_spec(spec, targets, preds, dev)


def fmt(v):
    return "   --" if v is None else f"{v:+.2f}"


def analyze(evals: list[str]) -> str:
    L = ["##### information gain from the subject's history: subject vs fixed analyst vs pool readers (dev)",
         "# subject gain  = few_shot[B]            - self_report[B]",
         "# analyst gain  = few_shot_other[B]      - analyst other_report about B (this script)",
         "# pool gain     = mean_A!=B fewshot[A,B] - mean_A!=B other_report[A,B]  (pool matrices, where both exist)",
         ""]
    tot = {"subject": [], "analyst": [], "pool": []}
    for ev in evals:
        spec = get_spec(ev)
        L.append(f"===== {ev} =====")
        L.append(f"  {'subject B':26s} {'sr':>6s} {'fs':>6s} {'gain':>6s} | {'an0':>6s} {'fso':>6s} {'gain':>6s} | {'pool0':>6s} {'poolfs':>6s} {'gain':>6s}")
        for b in POOL:
            sr = _score(spec, ev, b, _preds(ROOT / f"results/{ev}/{b}/predictions/self_report.json"))
            fs = _score(spec, ev, b, _preds(ROOT / f"results/{ev}/{b}/predictions/few_shot.json"))
            an0 = _score(spec, ev, b, _preds(out_path(ev, b)))
            fso = _score(spec, ev, b, _preds(ROOT / f"results/{ev}/{b}/predictions/few_shot_other.json"))
            p0 = [_score(spec, ev, b, _preds(ROOT / f"results/{ev}/_other_report/{a}__about__{b}.json"))
                  for a in POOL if a != b]
            pfs = [_score(spec, ev, b, _preds(fewshot_path(ev, a, b))) for a in POOL if a != b]
            p0 = [x for x in p0 if x is not None]; pfs = [x for x in pfs if x is not None]
            g_s = fs - sr if None not in (fs, sr) else None
            g_a = fso - an0 if None not in (fso, an0) else None
            g_p = mean(pfs) - mean(p0) if p0 and pfs else None
            for k, g in (("subject", g_s), ("analyst", g_a), ("pool", g_p)):
                if g is not None:
                    tot[k].append(g)
            L.append(f"  {b:26s} {fmt(sr):>6s} {fmt(fs):>6s} {fmt(g_s):>6s} | {fmt(an0):>6s} {fmt(fso):>6s} {fmt(g_a):>6s} | "
                     f"{fmt(mean(p0) if p0 else None):>6s} {fmt(mean(pfs) if pfs else None):>6s} {fmt(g_p):>6s}")
        L.append("")
    # Does the analyst's zero-info answer actually differentiate SUBJECTS, or just know the eval?
    import itertools
    L.append("Analyst zero-info genericness: answer agreement across named subjects, and its")
    L.append("about-B vector scored on B (named) vs on the other models (transfer):")
    for ev in evals:
        spec = get_spec(ev)
        vecs = {b: _preds(out_path(ev, b)) for b in POOL}
        rs = []
        for b1, b2 in itertools.combinations(POOL, 2):
            ks = sorted(set(vecs[b1]) & set(vecs[b2]))
            r = metrics.pearson([(vecs[b1][k], vecs[b2][k]) for k in ks])
            if r is not None:
                rs.append(r)
        diag = [x for x in (_score(spec, ev, b, vecs[b]) for b in POOL) if x is not None]
        off = [x for x in (_score(spec, ev, b2, vecs[b1])
                           for b1, b2 in itertools.product(POOL, POOL) if b1 != b2)
               if x is not None]
        if rs and diag and off:
            L.append(f"  {ev:22s} subject-subject answer r {mean(rs):+.2f}  |  on named subject "
                     f"{mean(diag):+.2f} vs on others {mean(off):+.2f} (advantage {mean(diag)-mean(off):+.2f})")
    L.append("POOLED mean gains: " + "  ".join(f"{k} {fmt(mean(v) if v else None)} (n={len(v)})" for k, v in tot.items()))
    L.append("A privileged-access signature requires subject gain > analyst/pool gain once reader skill is controlled.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["elicit", "analyze"])
    ap.add_argument("--evals", nargs="+", default=EVALS)
    ap.add_argument("--concurrency", type=int, default=20)
    a = ap.parse_args()
    if a.cmd == "elicit":
        elicit(a.evals, a.concurrency)
    else:
        t = analyze(a.evals); print(t)
        (ROOT / "results/reports/analyst_zero_info.txt").write_text(t + "\n")


if __name__ == "__main__":
    main()
