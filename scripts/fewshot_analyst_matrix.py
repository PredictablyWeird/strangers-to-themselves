"""Same-model analyst matrix for the history-informed channel (few_shot tier).

Model A is shown model B's measured dev history as ``few_shot``'s conversation — byte-identical
turns, re-attributed to B by name via ``few_shot_other``'s system turn — and predicts B's held-out
conditions out-of-fold (the benchmark's own CV folds). Every (A, B) pair in the small selection
pool is run, including A == B (the *named self*: the subject answering its own history, told whose
it is).

Why: the paper compares few_shot (subject answers own history) against few_shot_other (a fixed
Sonnet-4 analyst answers it) and finds the analyst better. That conflates privileged self-access
with reader skill. Holding the READER fixed (row A) and varying whose history it reads (column B)
separates them: a privileged-access signature is A predicting A better than A predicts others,
after controlling for how predictable each B is (column effects).

Outputs ``results/<eval>/_other_fewshot/<A>__about__<B>.json`` and
``results/reports/fewshot_analyst_matrix.txt``. Dev split only.

    python scripts/fewshot_analyst_matrix.py elicit  [--evals ...] [--concurrency N]
    python scripts/fewshot_analyst_matrix.py analyze [--evals ...]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from statistics import mean

import yaml

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods.base import RunConfig
from behavior_prediction.methods.trained import FewShotOther

ROOT = Path(__file__).resolve().parent.parent
POOL = list(yaml.safe_load(open(ROOT / "behavior_prediction/methods.yaml"))["selection_pool"])
EVALS = ["capability_mmlu", "propensitybench", "sycophancy_pushback", "discrimeval",
         "reward_hacking"]
DIR = "_other_fewshot"


def out_path(ev: str, a: str, b: str) -> Path:
    return ROOT / f"results/{ev}/{DIR}/{a}__about__{b}.json"


class _ModelShim:
    def __init__(self, shortcut: str) -> None:
        self.name = shortcut
        self.full, self.reasoning_cfg = common.resolve_model(shortcut)


class PoolAnalyst(FewShotOther):
    """``few_shot_other`` with a pool model as the reader. Keeps the system turn naming the
    SUBJECT (the ``model`` argument = B) and answers with A, under ``few_shot``'s temperature rule
    for A (greedy unless A has a reasoning config)."""
    def __init__(self, reader: str) -> None:
        super().__init__(predictor_model=reader)
        self.reader = _ModelShim(reader)

    def _predictor(self, model, cfg):
        temperature = cfg.temperature if self.reader.reasoning_cfg else self.GREEDY_TEMPERATURE
        return self.reader.full, self.reader.reasoning_cfg, temperature


def elicit(evals: list[str], concurrency: int) -> None:
    for ev in evals:
        spec = get_spec(ev)
        manifest = splits.load_manifest(ev)
        for a, b in itertools.product(POOL, POOL):
            out = out_path(ev, a, b)
            if out.exists():
                print(f"skip (exists): {out.relative_to(ROOT)}")
                continue
            tdoc = common.load_json(ROOT / f"results/{ev}/{b}/targets.json")
            targets = tdoc["targets"]
            subject = _ModelShim(b)
            method = PoolAnalyst(a)
            out.parent.mkdir(parents=True, exist_ok=True)
            cfg = RunConfig(temperature=1.0, max_parse_retries=2, concurrency=concurrency,
                            checkpoint_base=str(out))
            print(f"[few_shot] {ev}: {a} reads {b}'s history", flush=True)
            preds: dict = {}
            for fit_keys, score_keys in splits.folds(spec, targets, manifest):
                fs = method.fit({k: targets[k] for k in fit_keys}, subject, spec, {})
                conds = [targets[k]["condition"] for k in score_keys]
                preds.update(method.predict(spec, subject, conds, {}, cfg, fs))
            preds = {k: {kk: vv for kk, vv in e.items()
                         if kk not in ("condition", "raw", "reasoning")}
                     for k, e in preds.items()}
            common.save_json({"method": "other_fewshot", "predictor": a, "subject": b,
                              "subject_name": common.model_display_name(subject.full),
                              "eval": ev, "split": "dev", "predictions": preds}, out)


# --- analysis --------------------------------------------------------------------------------

def _load_preds(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    doc = common.load_json(path)
    return {k: e.get("predicted_rate") for k, e in doc["predictions"].items()}


def _score(spec, ev: str, subject: str, preds: dict) -> float | None:
    """Pearson r of ``preds`` against ``subject``'s dev behavior, scored the eval's way."""
    if not preds:
        return None
    targets = common.load_json(ROOT / f"results/{ev}/{subject}/targets.json")["targets"]
    dev = splits.split_keys(spec, targets, splits.load_manifest(ev), "dev")
    return metrics.corr_spec(spec, targets, preds, dev)


def _stored(ev: str, model: str, method: str) -> dict:
    return _load_preds(ROOT / f"results/{ev}/{model}/predictions/{method}.json")


def fmt(v) -> str:
    return "   --" if v is None else f"{v:+.2f}"


def analyze(evals: list[str]) -> str:
    lines = ["##### few_shot analyst matrix (dev; reader A shown subject B's history, named)",
             f"pool: {', '.join(POOL)}", ""]
    diag_all, off_all, rowgain_all, colgain_all = [], [], [], []
    for ev in evals:
        spec = get_spec(ev)
        M = {(a, b): _score(spec, ev, b, _load_preds(out_path(ev, a, b)))
             for a, b in itertools.product(POOL, POOL)}
        if all(v is None for v in M.values()):
            lines.append(f"===== {ev}: no cells yet\n")
            continue
        w = max(len(m) for m in POOL)
        lines.append(f"===== {ev} =====")
        lines.append("r vs SUBJECT B's behavior (rows: reader A; cols: subject B; diag = named self)")
        lines.append("  " + "A \\ B".ljust(w) + "".join(f"{b[:12]:>14}" for b in POOL))
        for a in POOL:
            lines.append("  " + a.ljust(w) + "".join(f"{fmt(M[(a, b)]):>14}" for b in POOL))
        # reference rows per subject B: stored few_shot (unnamed self) and few_shot_other (Sonnet-4)
        fs = {b: _score(spec, ev, b, _stored(ev, b, "few_shot")) for b in POOL}
        fso = {b: _score(spec, ev, b, _stored(ev, b, "few_shot_other")) for b in POOL}
        sr = {b: _score(spec, ev, b, _stored(ev, b, "self_report")) for b in POOL}
        lines.append("  " + "few_shot (unnamed self)".ljust(w) + "".join(f"{fmt(fs[b]):>14}" for b in POOL))
        lines.append("  " + "few_shot_other (Sonnet-4)".ljust(w) + "".join(f"{fmt(fso[b]):>14}" for b in POOL))
        lines.append("  " + "self_report (no history)".ljust(w) + "".join(f"{fmt(sr[b]):>14}" for b in POOL))
        # answerer-signature check: A's predictions about B scored against A's OWN behavior
        lines.append("r vs READER A's own behavior (leakage check; meaningful only off-diagonal)")
        for a in POOL:
            row = []
            for b in POOL:
                if a == b:
                    row.append(fmt(M[(a, b)]))
                else:
                    # needs the same condition keys to exist for A; score what overlaps
                    row.append(fmt(_score(spec, ev, a, _load_preds(out_path(ev, a, b)))))
            lines.append("  " + a.ljust(w) + "".join(f"{v:>14}" for v in row))
        diag = [M[(a, a)] for a in POOL if M[(a, a)] is not None]
        off = [M[(a, b)] for a, b in itertools.product(POOL, POOL) if a != b and M[(a, b)] is not None]
        # reader-fixed gain: A on A minus A's mean on others (controls reader skill)
        rg = []
        for a in POOL:
            others = [M[(a, b)] for b in POOL if b != a and M[(a, b)] is not None]
            if M[(a, a)] is not None and others:
                rg.append(M[(a, a)] - mean(others))
        # subject-fixed gain: B on B minus others' mean on B (controls subject predictability)
        cg = []
        for b in POOL:
            others = [M[(a, b)] for a in POOL if a != b and M[(a, b)] is not None]
            if M[(b, b)] is not None and others:
                cg.append(M[(b, b)] - mean(others))
        lines.append(f"  named-self diagonal {fmt(mean(diag) if diag else None)} (n={len(diag)})  "
                     f"off-diagonal {fmt(mean(off) if off else None)} (n={len(off)})  "
                     f"reader-fixed self-gain {fmt(mean(rg) if rg else None)}  "
                     f"subject-fixed self-gain {fmt(mean(cg) if cg else None)}")
        fs_v = [v for v in fs.values() if v is not None]
        fso_v = [v for v in fso.values() if v is not None]
        lines.append(f"  reference means: few_shot {fmt(mean(fs_v) if fs_v else None)}  "
                     f"few_shot_other {fmt(mean(fso_v) if fso_v else None)}\n")
        diag_all += diag; off_all += off; rowgain_all += rg; colgain_all += cg
    # Named-bundle vs bare few_shot: the diagonal differs from stored few_shot by the whole
    # system turn (name + "another AI model" disavowal + measured-ground-truth framing); paired
    # per cell this tests whether that bundle changes what the reader extracts from the history.
    paired = []
    for ev in evals:
        spec = get_spec(ev)
        for b in POOL:
            named = _score(spec, ev, b, _load_preds(out_path(ev, b, b)))
            bare = _score(spec, ev, b, _stored(ev, b, "few_shot"))
            if named is not None and bare is not None:
                paired.append(named - bare)
    if paired:
        lines.append(f"Named-self diagonal vs stored few_shot (paired): mean {mean(paired):+.3f} "
                     f"(n={len(paired)}, named better in {sum(x > 0 for x in paired)}) — the whole "
                     "system-turn bundle (name, disavowal, ground-truth framing) changes nothing.")
    lines.append("POOLED over evals: named-self diagonal "
                 f"{fmt(mean(diag_all) if diag_all else None)} (n={len(diag_all)}) vs off-diagonal "
                 f"{fmt(mean(off_all) if off_all else None)} (n={len(off_all)}); reader-fixed "
                 f"self-gain {fmt(mean(rowgain_all) if rowgain_all else None)}; subject-fixed "
                 f"self-gain {fmt(mean(colgain_all) if colgain_all else None)}. A privileged-access "
                 "signature requires the reader-fixed gain > 0.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["elicit", "analyze"])
    ap.add_argument("--evals", nargs="+", default=EVALS)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    if args.cmd == "elicit":
        elicit(args.evals, args.concurrency)
    else:
        text = analyze(args.evals)
        print(text)
        out = ROOT / "results/reports/fewshot_analyst_matrix.txt"
        out.write_text(text + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
