"""The activation-probe seam: capability negotiation skips API-only models, accepts local ones."""

from __future__ import annotations

from behavior_prediction.methods import ActivationProbe, valid_combo
from behavior_prediction.models_adapter import ActivationModelAdapter, ModelAdapter
from behavior_prediction.evals import get_spec


def test_probe_skipped_for_api_model_and_selectable_for_activation_model():
    spec = get_spec("discrimeval")
    probe = ActivationProbe()
    assert valid_combo(probe, ModelAdapter("llama-3.3-70b"), spec) is False
    assert valid_combo(probe, ActivationModelAdapter("local/whatever"), spec) is True


def test_probe_declares_trained_and_needs_activations():
    probe = ActivationProbe()
    assert probe.trained is True
    assert probe.capabilities.needs_activations is True
