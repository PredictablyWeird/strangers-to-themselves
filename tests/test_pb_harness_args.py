"""PropensityBench harness invocation: out-dir path handling and worker/timestamp passthrough.

The harness strips a leading "/" from --output_dir and resolves it against its own cwd, so
``_run_harness`` must hand it a repo-relative path (an absolute one would be silently nested
inside the harness repo AND break --pb-timestamp resume). Offline: subprocess.run is captured.
"""

from __future__ import annotations

import argparse

import pytest

from behavior_prediction.evals.propensitybench import PropensityBenchSpec


def _args(**over):
    ns = argparse.Namespace(
        model="llama-3.3-70b", pb_repo="/repo", pb_python=None, pb_input="data/_devpool",
        pb_run_out=None, pb_log_dir="evaluation/output/traj/", pb_provider=None,
        pb_model_name=None, pb_api_base=None, pb_max_workers=None, pb_timestamp=None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def run(monkeypatch):
    calls = {}

    def fake_run(cmd, cwd=None, check=None, env=None):
        calls.update(cmd=cmd, cwd=cwd, env=env)

    monkeypatch.setattr("subprocess.run", fake_run)

    def _go(**over):
        spec = PropensityBenchSpec()
        out_dir = spec._run_harness(_args(**over), "openrouter/meta-llama/llama-3.3-70b-instruct")
        opt = dict(zip(calls["cmd"], calls["cmd"][1:]))
        return out_dir, opt, calls

    return _go


def test_absolute_run_out_is_passed_repo_relative(run):
    out_dir, opt, _ = run(pb_run_out="/repo/results/propensitybench/_harness/llama-3.3-70b")
    assert opt["--output_dir"] == "results/propensitybench/_harness/llama-3.3-70b"  # no leading /
    assert out_dir == "/repo/results/propensitybench/_harness/llama-3.3-70b"        # caller view


def test_relative_run_out_and_default_flags_unchanged(run):
    out_dir, opt, calls = run(pb_run_out="results/x")
    assert opt["--output_dir"] == "results/x"
    assert "--max_workers" not in calls["cmd"] and "--timestamp" not in calls["cmd"]


def test_max_workers_and_timestamp_passthrough(run):
    _, opt, _ = run(pb_max_workers=200, pb_timestamp="2026-07-01-18-04-48.712264")
    assert opt["--max_workers"] == "200"
    assert opt["--timestamp"] == "2026-07-01-18-04-48.712264"
