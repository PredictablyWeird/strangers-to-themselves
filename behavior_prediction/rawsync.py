"""Consistency check between committed targets.json and the local behavior_raw.json.

``behavior_raw.json`` is gitignored scratch: on any given checkout it is whatever the last
LOCAL measurement run left behind, while ``targets.json`` is the committed, canonical
measurement (possibly produced on another host or in a later run). The two can silently
diverge — found 2026-08-03 on capability_mmlu/llama-3.3-70b, where 29/34 dev subjects
disagreed — and any item-level analysis that joins per-item raw entries against
target-derived quantities is then scrambled by run-to-run sampling noise.

This module reconstructs each condition's target rate from the raw file (for the evals whose
raw format supports it) and compares against targets.json. A cell whose reconstruction does
not match is STALE: its raw file belongs to a different sampling run and must not be used
for item-level work. ``bp-evaluate`` prints a loud warning for stale cells;
``scripts/check_raw_targets_sync.py`` is the standalone CLI (with ``--quarantine`` to rename
stale files to ``behavior_raw.stale.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TOL = 1e-6


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _target_rates(targets_path: Path) -> dict[str, dict[str, Any]]:
    t = _load(targets_path)
    tg = t.get("targets", t)
    out = {}
    for v in (tg.values() if isinstance(tg, dict) else tg):
        if isinstance(v, dict) and v.get("rate") is not None and "scenario" in v:
            out[v["scenario"]] = v
    return out


def _recon_capability(raw: dict[str, Any]) -> dict[str, float]:
    # NOTE: item ``predicted_rate``/``samples`` hold the CHOSEN OPTION INDEX (0-3), not
    # correctness — correctness lives in ``n_correct``/``n_answered`` (added by the parse
    # step against the MMLU answer key). Reconstruct accuracy from those.
    correct: dict[str, int] = {}
    answered: dict[str, int] = {}
    for item in raw["results"].values():
        s = item.get("subject")
        if s is None or item.get("n_answered") in (None, 0):
            continue
        correct[s] = correct.get(s, 0) + int(item.get("n_correct") or 0)
        answered[s] = answered.get(s, 0) + int(item["n_answered"])
    return {s: correct[s] / answered[s] for s in correct if answered.get(s)}


def _recon_sycophancy(raw: dict[str, Any]) -> dict[str, float]:
    by: dict[str, list[bool]] = {}
    for item in raw["results"].values():
        if item.get("first_idx") == item.get("answer_idx"):
            by.setdefault(item["subject"], []).append(bool(item.get("flip")))
    return {s: sum(v) / len(v) for s, v in by.items() if v}


def _recon_reward_hacking(raw: dict[str, Any]) -> dict[str, float]:
    hacks: dict[str, int] = {}
    answered: dict[str, int] = {}
    for item in raw["results"].values():
        if item.get("variant") != "wp":
            continue
        s = item["subject"]
        hacks[s] = hacks.get(s, 0) + int(item.get("n_hack") or 0)
        answered[s] = answered.get(s, 0) + int(item.get("n_answered") or 0)
    return {s: hacks[s] / answered[s] for s in hacks if answered.get(s)}


#: Evals whose behavior_raw carries enough per-item structure to reconstruct target rates.
RECONSTRUCTORS: dict[str, Callable[[dict[str, Any]], dict[str, float]]] = {
    "capability_mmlu": _recon_capability,
    "sycophancy_pushback": _recon_sycophancy,
    "reward_hacking": _recon_reward_hacking,
}


@dataclass
class SyncReport:
    eval_name: str
    model: str
    raw_path: Path
    n_compared: int
    n_mismatched: int
    n_missing_in_raw: int
    max_abs_diff: float

    @property
    def stale(self) -> bool:
        return self.n_mismatched > 0 or (self.n_compared == 0 and self.n_missing_in_raw > 0)

    def line(self) -> str:
        status = "STALE" if self.stale else "ok"
        return (f"[{status}] {self.eval_name}/{self.model}: {self.n_mismatched}/"
                f"{self.n_compared} conditions mismatch (max |Δ|={self.max_abs_diff:.3f}, "
                f"{self.n_missing_in_raw} conditions absent from raw)")


def check_cell(eval_name: str, model_dir: Path) -> SyncReport | None:
    """Compare one model dir's behavior_raw against its targets; None when not checkable."""
    recon_fn = RECONSTRUCTORS.get(eval_name)
    raw_path = model_dir / "behavior_raw.json"
    targets_path = model_dir / "targets.json"
    if recon_fn is None or not raw_path.exists() or not targets_path.exists():
        return None
    try:
        recon = recon_fn(_load(raw_path))
        targets = _target_rates(targets_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return SyncReport(eval_name, model_dir.name, raw_path, 0, 0, 0, 0.0)
    diffs = [abs(recon[s] - t["rate"]) for s, t in targets.items() if s in recon]
    missing = sum(1 for s in targets if s not in recon)
    return SyncReport(
        eval_name, model_dir.name, raw_path,
        n_compared=len(diffs),
        n_mismatched=sum(1 for d in diffs if d > TOL),
        n_missing_in_raw=missing,
        max_abs_diff=max(diffs, default=0.0),
    )


def check_all(results_root: Path | str = "results") -> list[SyncReport]:
    """Sweep every checkable (eval, model) cell under ``results_root``."""
    reports = []
    root = Path(results_root)
    for eval_name in RECONSTRUCTORS:
        eval_dir = root / eval_name
        if not eval_dir.is_dir():
            continue
        for model_dir in sorted(eval_dir.iterdir()):
            if model_dir.is_dir():
                r = check_cell(eval_name, model_dir)
                if r is not None:
                    reports.append(r)
    return reports


def warn_if_stale(results_root: Path | str = "results") -> list[SyncReport]:
    """Loud-warning hook for bp-evaluate: prints a banner when any raw file is stale."""
    stale = [r for r in check_all(results_root) if r.stale]
    if stale:
        print("=" * 78)
        print("WARNING: behavior_raw.json out of sync with committed targets.json for "
              f"{len(stale)} cell(s) — these raw files are from a DIFFERENT sampling run.")
        print("Do NOT use them for item-level analysis. Quarantine with:")
        print("  .venv/bin/python scripts/check_raw_targets_sync.py --quarantine")
        for r in stale:
            print("  " + r.line())
        print("=" * 78)
    return stale
