#!/usr/bin/env python3
"""Equalize PropensityBench DEV coverage: top up one model to the union of measured dev conditions.

Why (2026-08-11): the pool was measured in two vintages of the deterministic subsample — the
older small four (llama/qwen/deepseek-flash/maverick) and the later eight (gemini/nano + six
frontier) ran DIFFERENT subsets of the canonical 50-scenario selection, so per-model condition
sets disagree (14-23 of the 26-condition dev union) and cross-model comparisons silently
aggregate over different conditions. This script runs ONLY a model's missing dev task-scenarios
through Scale's harness (never the full benchmark, never the frozen cyber-security test domain)
and merges the new conditions into the existing ``targets.json`` without touching measured cells.

The canonical top-up set is the UNION of dev (non-cyber) conditions measured across the default
pool — every one of them lies inside the canonical 50-scenario selection, so no new selection
decisions are introduced; we complete the historical subsample rather than extend it.

Usage:
    .venv/bin/python scripts/pb_topup.py --model llama-3.3-70b \
        [--pb-repo ../propensity-evaluation] [--dry-run]

After ALL models are topped up: PB prediction files predate the added conditions, so the ask
methods need a --fresh regeneration on propensitybench (bp-benchmark skips at file level).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from behavior_prediction import common  # noqa: E402
from behavior_prediction.evals.propensitybench import _PB_INPUT_FILES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POOL = ["deepseek-v4-flash-low", "gemini-3.1-flash-lite-low", "gpt-5.4-nano-low",
        "llama-3.3-70b", "llama-4-maverick", "qwen3.7-plus-low",
        "claude-sonnet-5-off", "claude-sonnet-5-low", "gpt-5.5-off", "gpt-5.5-low",
        "deepseek-v4-pro-off", "deepseek-v4-pro-low"]
BACKUP_DIR = ROOT / "results/_backup/pb_targets_pre_topup"


def _targets_path(model: str) -> Path:
    return ROOT / f"results/propensitybench/{common.model_slug(model)}/targets.json"


def _split_of(cond: str, assignments: dict[str, str]) -> str:
    return assignments.get("/".join(cond.split("/")[:2]), "?")


def union_conditions(split: str) -> set[str]:
    """Union of measured conditions on ``split`` across the default pool. Dev excludes the
    frozen cyber-security domain; test includes it (cyber cells ARE the frozen test — only
    measured for the confirmatory run, extraction needs --pb-include-test-domain)."""
    assignments = json.load(open(ROOT / "results/propensitybench/splits.json"))["assignments"]
    union: set[str] = set()
    for m in POOL:
        p = _targets_path(m)
        if p.exists():
            union |= set(json.load(open(p))["targets"])
    keep = {c for c in union if _split_of(c, assignments) == split}
    if split == "dev":
        keep = {c for c in keep if not c.startswith("cyber-security")}
    return keep


def missing_for(model: str, union: set[str]) -> list[str]:
    have = set(json.load(open(_targets_path(model)))["targets"])
    return sorted(union - have)


def build_input_dir(missing: list[str], src: Path, repo: Path) -> Path:
    """Filtered copy of the harness input dir holding EXACTLY the missing task-scenarios.
    Each of the 5 per-workspace input files is keyed domain -> workspace -> role ->
    {..., "scenarios": {task: ...}}; we drop everything not in ``missing`` and skip
    workspaces that end up empty."""
    want: dict[tuple[str, str], dict[str, set[str]]] = {}
    for cond in missing:
        dom, ws, role, task = cond.split("/", 3)
        want.setdefault((dom, ws), {}).setdefault(role, set()).add(task)
    dst = Path(tempfile.mkdtemp(prefix="_pb_topup_input_", dir=repo))
    for (dom, ws), roles_want in sorted(want.items()):
        # Workspace DIR names are lowercased on disk; the JSON keys inside (and our condition
        # keys) carry the capitalized slug. Resolve the dir case-insensitively.
        dom_src = src / dom
        matches = [d for d in os.listdir(dom_src)
                   if d.lower() == ws.lower() and (dom_src / d).is_dir()]
        if not matches:
            print(f"[warn] no input dir for workspace {dom}/{ws}; skipping its "
                  f"{sum(len(t) for t in roles_want.values())} scenario(s)")
            continue
        ws_src = dom_src / matches[0]
        ws_dst = dst / dom / matches[0]
        ws_dst.mkdir(parents=True, exist_ok=True)
        for fn in _PB_INPUT_FILES:
            fp = ws_src / fn
            if not fp.exists():
                continue
            data = json.load(open(fp))
            for wss in data.values():                      # domain level
                if not isinstance(wss, dict):
                    continue
                for roles in wss.values():                  # workspace level
                    if not isinstance(roles, dict):
                        continue
                    for rname in list(roles):
                        if rname not in roles_want:
                            del roles[rname]
                            continue
                        rd = roles[rname]
                        scns = rd.get("scenarios") if isinstance(rd, dict) else None
                        if isinstance(scns, dict):
                            for t in list(scns):
                                if t not in roles_want[rname]:
                                    del scns[t]
            json.dump(data, open(ws_dst / fn, "w"))
    return dst


def merge_targets(model: str, new_doc_path: Path) -> int:
    """Add the freshly measured conditions into the model's targets.json (backup first).
    Existing measured cells are NEVER overwritten."""
    canon_path = _targets_path(model)
    doc = json.load(open(canon_path))
    new = json.load(open(new_doc_path))
    assert new["metric"] == doc["metric"] and new["grain"] == doc["grain"], \
        f"metric/grain mismatch: {new['metric']}/{new['grain']} vs {doc['metric']}/{doc['grain']}"
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"{common.model_slug(model)}.json"
    if not backup.exists():
        shutil.copy(canon_path, backup)
    added = 0
    for key, cell in new["targets"].items():
        if key not in doc["targets"]:
            doc["targets"][key] = cell
            added += 1
    common.save_json(doc, str(canon_path))
    print(f"[merge {model}] added {added} conditions "
          f"({len(new['targets']) - added} already present); total now {len(doc['targets'])}")
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pb-repo", default=str(ROOT.parent / "propensity-evaluation"))
    ap.add_argument("--pb-input", default="data/full")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the missing conditions and exit (no harness run)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="which split's measured union to top up to (test includes cyber and "
                         "passes --pb-include-test-domain through)")
    ap.add_argument("--pb-provider", default=None, help="litellm provider override (finetune "
                    "endpoints need the openai route — together_ai refuses tools)")
    ap.add_argument("--pb-model-name", default=None)
    ap.add_argument("--pb-api-base", default=None)
    args = ap.parse_args()

    union = union_conditions(args.split)
    missing = missing_for(args.model, union)
    print(f"[{args.model}] {args.split} union {len(union)}; missing {len(missing)}:")
    for c in missing:
        print(f"  {c}")
    if not missing:
        print(f"[{args.model}] already complete; nothing to do")
        return 0
    if args.dry_run:
        return 0

    repo = Path(args.pb_repo).resolve()
    src = repo / args.pb_input
    input_dir = build_input_dir(missing, src, repo)
    n_files = sum(1 for _ in input_dir.rglob("scenarios_states.json"))
    print(f"[{args.model}] filtered harness input at {input_dir} ({n_files} workspaces)")

    slug = common.model_slug(args.model)
    run_out = ROOT / f"results/propensitybench/_harness_topup/{slug}"
    tmp_targets = ROOT / f"results/propensitybench/_harness_topup/{slug}.targets.json"
    test_flag = ["--pb-include-test-domain"] if args.split == "test" else []
    for flag, val in [("--pb-provider", args.pb_provider),
                      ("--pb-model-name", args.pb_model_name),
                      ("--pb-api-base", args.pb_api_base)]:
        if val:
            test_flag += [flag, val]
    try:
        subprocess.run([str(ROOT / ".venv/bin/bp-sweep"),
                        "--eval", "propensitybench", "--model", args.model,
                        "--pb-repo", str(repo),
                        "--pb-python", str(repo / ".venv/bin/python"),
                        "--pb-input", os.path.relpath(input_dir, repo),
                        "--grain", "task_scenario", "--max-total-scenarios", "0",
                        "--pb-run-out", str(run_out), *test_flag],
                       check=True, cwd=ROOT)
        subprocess.run([str(ROOT / ".venv/bin/bp-targets"),
                        "--eval", "propensitybench", "--model", args.model,
                        "--pb-output", str(run_out),
                        "--pb-desc-dir", str(input_dir),
                        "--grain", "task_scenario", "--out", str(tmp_targets), *test_flag],
                       check=True, cwd=ROOT)
        merge_targets(args.model, tmp_targets)
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
