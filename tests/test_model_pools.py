"""Named model pools (models.yaml `pools:`) and the artifact-ownership rule in bp-evaluate.

The rule that matters: only the `default` pool writes the canonical evaluation.* artifacts and the
paper's LaTeX macros. Any other pool — including an ad-hoc --models list — must write its own
suffixed files, so scoring a subset of models can never overwrite the numbers the paper cites.
"""

from __future__ import annotations

import pytest

from behavior_prediction import common, evaluate


def test_declared_pools_reference_known_models():
    """Every pool member must resolve in the models.yaml registry (catches typos and stale slugs)."""
    registry = common.load_model_registry()
    for pool, models in common.load_model_pools().items():
        assert models, f"pool {pool!r} is empty"
        unknown = [m for m in models if m not in registry]
        assert not unknown, f"pool {pool!r} references unknown models: {unknown}"


def test_default_pool_is_the_paper_pool():
    # 2026-08-07 pool merge: the paper pool is the six small models plus the three frontier
    # bases at two reasoning settings each; `small` keeps the original six for tier contrasts.
    assert evaluate.DEFAULT_POOL == "default"
    assert common.model_pool("default") == [
        "deepseek-v4-flash-low", "gemini-3.1-flash-lite-low", "gpt-5.4-nano-low",
        "llama-3.3-70b", "llama-4-maverick", "qwen3.7-plus-low",
        "claude-sonnet-5-off", "claude-sonnet-5-low",
        "gpt-5.5-off", "gpt-5.5-low",
        "deepseek-v4-pro-off", "deepseek-v4-pro-low",
    ]
    assert common.model_pool("small") == common.model_pool("default")[:6]


def test_introspection_pool_pairs_each_base_with_its_finetune():
    intro = common.model_pool("introspection")
    assert {"qwen3-30b-a3b", "qwen3-30b-a3b-intro30k"} <= set(intro)
    assert "qwen3-30b-a3b-intro" not in intro  # k=1000 sanity-check LoRA, superseded by k=30000
    assert {"llama-3.3-70b-tg", "llama-3.3-70b-intro30k-tg"} <= set(intro)


def test_combined_pool_is_small_plus_introspection():
    # Despite its (pre-pool-merge) name, the combined pool pairs the introspection models with
    # the SMALL tier — the six models the finetunes are comparable to — not the merged twelve.
    combined = common.model_pool("default_plus_introspection")
    assert set(combined) == set(common.model_pool("small")) | set(common.model_pool("introspection"))


def test_discover_pool_defers_to_disk():
    """`all` means "whatever has predictions" — resolved by the caller as model_names=None."""
    assert common.model_pool(common.DISCOVER_POOL) is None


def test_unknown_pool_lists_the_valid_names():
    with pytest.raises(KeyError) as exc:
        common.model_pool("nope")
    assert "introspection" in str(exc.value) and common.DISCOVER_POOL in str(exc.value)


def _capture_writes(monkeypatch, tmp_path, pool):
    """Run write_reports over an empty results tree, recording every path written."""
    written: list[str] = []
    monkeypatch.setattr(common, "results_root", lambda: tmp_path)
    monkeypatch.setattr(evaluate, "compute", lambda *a, **k: {
        "cells": {}, "methods": [], "evals": [], "models": [], "settings": {},
        "dropped": 0, "missing_tuning": []})
    from behavior_prediction import export
    monkeypatch.setattr(export, "write_latex_macros", lambda doc, p: written.append(str(p)))
    monkeypatch.setattr(export, "write_method_model_table", lambda doc, p: written.append(str(p)))
    monkeypatch.setattr(export, "write_calibration_table", lambda doc, p: written.append(str(p)))
    monkeypatch.setattr(export, "write_json", lambda doc, p: written.append(str(p)))

    md = tmp_path / "out.md"
    evaluate.write_reports([], [], "dev", str(md), pool)
    written += [str(p) for p in (tmp_path / "reports").glob("*")]
    return written


def test_default_pool_writes_canonical_artifacts_and_paper_macros(monkeypatch, tmp_path):
    written = _capture_writes(monkeypatch, tmp_path, "default")
    assert any(w.endswith("paper/generated/results.tex") for w in written)
    assert any(w.endswith("paper/generated/table_method_model.tex") for w in written)
    assert any(w.endswith("paper/generated/table_bias.tex") for w in written)
    assert any(w.endswith("reports/evaluation.html") for w in written)
    assert any(w.endswith("reports/evaluation.json") for w in written)


@pytest.mark.parametrize("pool", ["introspection", "custom", "all"])
def test_non_default_pool_never_touches_paper_or_canonical_files(monkeypatch, tmp_path, pool):
    written = _capture_writes(monkeypatch, tmp_path, pool)
    assert not any("paper/generated" in w for w in written), f"{pool} wrote a paper file"
    assert not any(w.endswith("reports/evaluation.html") for w in written)
    assert not any(w.endswith("reports/evaluation.json") for w in written)
    assert any(w.endswith(f"reports/evaluation_{pool}.html") for w in written)
    assert any(w.endswith(f"reports/evaluation_{pool}.json") for w in written)


def test_extra_body_passthrough():
    """Together ignores `reasoning_effort`; only extra_body.chat_template_kwargs disables thinking.
    resolve_model must carry it into GenerateConfig, and it must count as part of model identity."""
    full, cfg = common.resolve_model("gemma-4-31b")
    assert full == "together/google/gemma-4-31B-it"
    assert cfg["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    # a raw provider string must NOT collapse onto a shortcut that carries per-call config,
    # or thinking-on and thinking-off runs would share one results directory
    assert common.model_slug("together/google/gemma-4-31B-it") != "gemma-4-31b"
    assert common.model_slug("gemma-4-31b") == "gemma-4-31b"

    # identity records the config, so the mismatch guard separates thinking-on from thinking-off
    ident = common.model_identity("gemma-4-31b")
    assert ident["reasoning"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert common.identity_mismatch(ident, {"model": full, "reasoning": {}}) == "reasoning"


def test_canonical_model_strips_rotating_endpoint_suffix():
    """A Together dedicated endpoint = finetuned model id + a fresh 8-hex suffix per deploy. Targets
    measured through one deploy must still match predictions made through another."""
    ft = "together/acct/Llama-3.3-70B-Instruct-Reference-selfpred-llama-v2-e57d5a06"
    ep1, ep2 = f"{ft}-6ade019f", f"{ft}-a16f62c6"
    assert common.canonical_model(ep1) == ft
    assert common.canonical_model(ep2) == ft
    assert common.identity_mismatch({"model": ep1}, {"model": ep2}) is None
    assert common.identity_mismatch({"model": ep1}, {"model": ft}) is None

    # plain models and bare finetune ids are untouched
    for m in ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "together/google/gemma-4-31B-it",
              "together/acct/gemma-4-31B-it-selfpred-gemma-v1-8b7c705d"):
        assert common.canonical_model(m) == m

    # genuinely different models still mismatch
    other = "together/acct/Llama-3.3-70B-Instruct-Reference-selfpred-llama-v1-03d958a8-deadbeef"
    assert common.identity_mismatch({"model": ep1}, {"model": other}) == "model"
