#!/usr/bin/env python3
"""Model adapter: a thin handle over a model under test that the benchmark drives.

The generation engine stays ``common.elicit_rates`` (which uses ``inspect_ai.model.get_model``
+ ``GenerateConfig``, already handling every provider + reasoning effort/tokens/summary from
``models.yaml``). This wrapper adds only (a) resolution of a ``--model`` shortcut to its full
string + reasoning config, (b) **capability flags** so the benchmark can run only valid
(model × method) combos, and (c) a stable ``slug`` for result paths.

``ActivationModelAdapter`` is the seam for locally-hosted models that expose hidden states for
mechanistic-interpretability methods (e.g. activation probes); it is a declared stub here and
is implemented in the activation-probe phase.
"""

from __future__ import annotations

from typing import Any

from behavior_prediction import common


class ModelAdapter:
    """Wraps a model under test. ``name`` is a ``--model`` value (a ``models.yaml`` shortcut or
    a full provider string)."""

    #: Capability flags — methods declare requirements; the harness checks them.
    has_logprobs: bool = False
    has_activations: bool = False
    has_reasoning_trace: bool = True   # most chat models can return reasoning if configured

    def __init__(self, name: str) -> None:
        self.name = name
        self.full, self.reasoning_cfg = common.resolve_model(name)

    @property
    def slug(self) -> str:
        return common.model_slug(self.name)

    def elicit(self, prompts: dict[str, str], **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Pooled ``ask-N-times-and-parse`` over ``prompts`` (delegates to ``elicit_rates``).
        The model's reasoning config is applied unless the caller overrides it."""
        kwargs.setdefault("reasoning_config", self.reasoning_cfg)
        return common.elicit_rates(self.full, prompts, **kwargs)

    def __repr__(self) -> str:
        return f"ModelAdapter({self.name!r} -> {self.full!r})"


class ActivationModelAdapter(ModelAdapter):
    """Stub for a locally-hosted model exposing activations (HF/nnsight/TransformerLens).

    Implemented in the activation-probe phase. Declares ``has_activations = True`` and will add
    an ``activations(prompt, layer)`` method that probe-based methods consume in ``fit``/``predict``.
    """

    has_activations = True

    def activations(self, prompt: str, layer: int) -> Any:  # pragma: no cover - stub
        raise NotImplementedError("ActivationModelAdapter is a stub; implement in the probe phase")
