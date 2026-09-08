"""Interruption-resume, concurrency wiring, and transport-retry behaviour of the shared
``common.elicit_rates`` engine. The model is faked, so these stay offline."""

from __future__ import annotations

import asyncio
import json

import pytest

from behavior_prediction import common
from behavior_prediction.elicitation import _checkpoint_path, _cleanup_checkpoints


class _Output:
    def __init__(self, completion: str) -> None:
        self.completion = completion  # no .choices -> _extract_reasoning returns None


class _FakeModel:
    """Records every generate call. ``reply`` maps (prompt, call_index) -> completion text."""

    def __init__(self, reply) -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.last_config = None

    async def generate(self, prompt, config=None):
        self.calls.append(prompt)
        self.last_config = config
        return _Output(self.reply(prompt, len(self.calls)))


def _use_model(monkeypatch, model) -> None:
    import inspect_ai.model as im

    monkeypatch.setattr(im, "get_model", lambda name: model)


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def test_checkpoint_written_per_sample_and_concurrency_wired(tmp_path, monkeypatch):
    model = _FakeModel(lambda p, n: "50%")
    _use_model(monkeypatch, model)
    cp = str(tmp_path / "partial" / "x.jsonl")   # parent dir does not exist yet

    res = common.elicit_rates("fake", {"a": "pa", "b": "pb"}, runs=2, concurrency=3,
                              checkpoint_path=cp, verbose=False)

    assert res["a"]["n"] == 2 and res["a"]["predicted_rate"] == 0.5
    assert len(model.calls) == 4                       # 2 keys x 2 runs
    assert len(_lines(cp)) == 4                         # one complete sample per line
    # concurrency is wired into inspect's own connection pool (the silent cap fix)
    assert model.last_config.max_connections == 3


def test_full_resume_uses_checkpoint_without_new_calls(tmp_path, monkeypatch):
    model = _FakeModel(lambda p, n: "50%")
    _use_model(monkeypatch, model)
    cp = str(tmp_path / "x.jsonl")
    common.elicit_rates("fake", {"a": "pa", "b": "pb"}, runs=2, checkpoint_path=cp, verbose=False)

    # Re-run the identical call with a model that would answer differently: nothing should re-issue.
    model2 = _FakeModel(lambda p, n: "90%")
    _use_model(monkeypatch, model2)
    res = common.elicit_rates("fake", {"a": "pa", "b": "pb"}, runs=2, checkpoint_path=cp,
                              verbose=False)
    assert model2.calls == []                           # fully resumed
    assert res["a"]["predicted_rate"] == 0.5            # from checkpoint, not the new 90%


def test_partial_resume_schedules_only_remaining(tmp_path, monkeypatch):
    cp = tmp_path / "x.jsonl"
    cp.write_text(json.dumps({"key": "a", "frac": 0.5, "text": "50%", "reasoning": None}) + "\n")
    model = _FakeModel(lambda p, n: "10%")
    _use_model(monkeypatch, model)

    res = common.elicit_rates("fake", {"a": "pa", "b": "pb"}, runs=2, checkpoint_path=str(cp),
                              verbose=False)
    assert len(model.calls) == 3                        # a needs 1 more, b needs 2
    assert res["a"]["n"] == 2
    assert abs(res["a"]["predicted_rate"] - 0.3) < 1e-9  # mean(0.5 resumed, 0.1 fresh)
    assert res["b"]["predicted_rate"] == 0.1


def test_torn_final_line_is_tolerated(tmp_path, monkeypatch):
    cp = tmp_path / "x.jsonl"
    cp.write_text(json.dumps({"key": "a", "frac": 0.5, "text": "50%", "reasoning": None}) + "\n"
                  + '{"key": "a", "frac": 0.5, "te')   # torn write from a hard kill
    model = _FakeModel(lambda p, n: "10%")
    _use_model(monkeypatch, model)
    res = common.elicit_rates("fake", {"a": "pa"}, runs=2, checkpoint_path=str(cp), verbose=False)
    assert len(model.calls) == 1                        # one good sample loaded, one re-issued
    assert res["a"]["n"] == 2


def test_transport_error_retries_with_backoff_then_succeeds(tmp_path, monkeypatch):
    state = {"n": 0}

    class _Flaky:
        last_config = None

        async def generate(self, prompt, config=None):
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("boom")
            return _Output("50%")

    _use_model(monkeypatch, _Flaky())
    slept: list[float] = []

    async def _no_sleep(d):
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    res = common.elicit_rates("fake", {"a": "pa"}, runs=1, max_transport_retries=3, verbose=False)
    assert res["a"]["predicted_rate"] == 0.5
    assert state["n"] == 3                               # 2 failures + 1 success
    assert slept == [1, 2]                               # exponential backoff between retries


def test_transport_failure_recorded_as_error_sample(tmp_path, monkeypatch):
    class _Dead:
        async def generate(self, prompt, config=None):
            raise RuntimeError("nope")

    _use_model(monkeypatch, _Dead())

    async def _no_sleep(d):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    res = common.elicit_rates("fake", {"a": "pa"}, runs=1, max_transport_retries=1, verbose=False)
    assert res["a"]["n"] == 0 and res["a"]["parse_failures"] == 1
    assert res["a"]["raw"][0].startswith("<error:")


def test_checkpoint_path_and_cleanup(tmp_path):
    pred = tmp_path / "results" / "ev" / "m" / "predictions" / "self_report.json"
    pred.parent.mkdir(parents=True)
    cp = _checkpoint_path(str(pred))
    assert cp.endswith("partial/self_report.jsonl")
    behave = _checkpoint_path(str(pred), "behave")
    assert behave.endswith("partial/self_report.behave.jsonl")

    partial = tmp_path / "results" / "ev" / "m" / "partial"
    partial.mkdir()
    (partial / "self_report.jsonl").write_text("x\n")
    (partial / "self_report.behave.jsonl").write_text("x\n")
    sibling = partial / "self_report-concrete.jsonl"   # different method sharing the prefix
    sibling.write_text("x\n")

    _cleanup_checkpoints(str(pred))
    assert not (partial / "self_report.jsonl").exists()
    assert not (partial / "self_report.behave.jsonl").exists()
    assert sibling.exists()                              # prefix-collision guard holds
