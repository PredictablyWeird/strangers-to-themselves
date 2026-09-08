"""PropensityBench frozen-test-domain guard: cyber-security is excluded from measurement
(input-dir pruning) and extraction (record filtering) by default, and only
``--pb-include-test-domain`` lifts it. The split seam (cyber forced into test) is unchanged.
Offline: everything runs on tmp trees.
"""

from __future__ import annotations

import json
import os

import pytest

from behavior_prediction.evals.propensitybench import (
    PB_TEST_DOMAINS,
    PropensityBenchSpec,
)


@pytest.fixture
def input_tree(tmp_path):
    """A minimal harness input dir: one workspace per domain, one states file each."""
    for domain in ("cyber-security", "bio-security"):
        ws = tmp_path / "input" / domain / "ws1"
        ws.mkdir(parents=True)
        (ws / "scenarios_states.json").write_text(json.dumps(
            {"ws1": {"ws1": {"r1": {"scenarios": {"t1": {}}}}}}))
    return tmp_path / "input"


def test_prune_excludes_test_domain_without_caps(input_tree, tmp_path):
    spec = PropensityBenchSpec()
    dst = spec._prune_input_dir(str(input_tree), None, None, None,
                                base_dir=str(tmp_path), exclude_domains=set(PB_TEST_DOMAINS))
    assert dst != str(input_tree), "exclusion alone must trigger a pruned copy"
    assert sorted(os.listdir(dst)) == ["bio-security"]


def test_prune_without_caps_or_exclusion_is_passthrough(input_tree):
    spec = PropensityBenchSpec()
    assert spec._prune_input_dir(str(input_tree), None, None, None) == str(input_tree)


def test_include_test_domain_keeps_all_domains(input_tree, tmp_path):
    spec = PropensityBenchSpec()
    # the final confirmatory run passes an empty exclusion but still wants the caps machinery
    dst = spec._prune_input_dir(str(input_tree), 1, None, None,
                                base_dir=str(tmp_path), exclude_domains=set())
    assert sorted(os.listdir(dst)) == ["bio-security", "cyber-security"]


def _results_dir(tmp_path):
    out = tmp_path / "harness_out"
    out.mkdir()
    rows = [
        {"domain": "cyber-security", "workspace": "ws1", "role": "r1", "scenario": "t1",
         "category": "time", "trigger_called": True, "pressure_level": 2},
        {"domain": "bio-security", "workspace": "ws1", "role": "r1", "scenario": "t1",
         "category": "time", "trigger_called": False, "pressure_level": 1},
    ]
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(out)


def test_aggregate_drops_test_domain_by_default(tmp_path):
    spec = PropensityBenchSpec()
    doc = spec._aggregate(_results_dir(tmp_path), "m", "trigger_called", grain="role")
    domains = {t["scenario"] for t in doc["targets"].values()}
    assert domains == {"bio-security"}


def test_aggregate_keeps_test_domain_with_flag(tmp_path):
    spec = PropensityBenchSpec()
    doc = spec._aggregate(_results_dir(tmp_path), "m", "trigger_called", grain="role",
                          include_test=True)
    domains = {t["scenario"] for t in doc["targets"].values()}
    assert domains == {"bio-security", "cyber-security"}


def test_forced_split_still_pins_test_domain():
    spec = PropensityBenchSpec()
    assert spec.forced_split("cyber-security/anything") == "test"
    assert spec.forced_split("bio-security/anything") is None
