"""Method-adapter interface: a prediction method as ``fit`` + ``predict`` + declared
capability requirements.

This is the seam the benchmark drives. Existing elicitation methods are *stateless* — their
``fit`` is a no-op and ``predict`` delegates to the single-sourced elicitation primitives in
``elicitation.py`` (no prompt logic is duplicated). Methods that learn parameters (e.g. an
activation probe) set ``trained=True``, implement a real ``fit`` over a fit-split of the dev
pool, and carry a ``FitState`` into ``predict`` (the benchmark cross-validates them).

A method declares its ``Capabilities``; the registry's ``valid_combo`` runs a method only
against models/evals that satisfy them, so e.g. a probe method is skipped for an API-only model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from behavior_prediction import common


@dataclass(frozen=True)
class Capabilities:
    """What a method needs from the model / eval to run."""
    needs_activations: bool = False        # requires model hidden states (mech-interp)
    needs_logprobs: bool = False           # requires token logprobs
    needs_all_conditions: bool = False     # must see the whole sweep at once (e.g. ranking)
    abstract_description_only: bool = False  # builds only from the Frame, never the raw eval prompt


@dataclass
class RunConfig:
    """Sampling knobs for one ``predict`` call. ``runs=None`` uses the method's ``default_runs``."""
    runs: int | None = None
    temperature: float = 1.0
    max_parse_retries: int = 2
    concurrency: int = 5
    #: Prediction out path for this call, used to derive an interruption-resume checkpoint sidecar
    #: (``None`` disables checkpointing). Set per (eval, model, method, hp) by the benchmark runner.
    checkpoint_base: str | None = None


class MethodAdapter:
    """Interface every prediction method implements."""

    base_method: str = ""                       # prompt-builder key; feeds common.method_id
    capabilities: Capabilities = Capabilities()
    default_runs: int = 10
    #: Whether ``fit`` learns from the dev pool. When True the benchmark generates *out-of-fold*
    #: dev predictions (fit on the other CV folds, predict the held-out fold) so the dev score is
    #: cross-validated; when False ``predict`` is fold-independent and is run once.
    trained: bool = False
    #: Whether ``predicted_rate`` is a **calibrated rate** on the eval's own scale (comparable to
    #: the measured rate) or merely a monotone **score** used for ranking. Correlation is defined
    #: for both; the calibration metrics (absolute error, signed level bias) are only meaningful
    #: for the former, so ``bp-evaluate`` skips them when this is False. Score-emitting methods:
    #: the tournament/ordinal ones (pairwise, oracle_pairwise), the sign-decoded cot_flip, the
    #: z-scored ensembles, and the in-context regressors whose per-fold calibration re-centres
    #: predictions on a fixed anchor (llm_prediction, few_shot*).
    calibrated: bool = True

    def grid_key(self) -> str:
        """Key used to look up this method's argument grid in ``methods.yaml`` (its ``base_method``
        by default; override to share / diverge a grid)."""
        return self.base_method

    def hyperparameter_grid(self) -> list[dict[str, Any]]:
        """Hyperparameter dicts to search over on dev, from ``methods.yaml`` (``[{}]`` when the
        method has no swept args)."""
        return common.method_arg_grid(self.grid_key())

    def fit(self, train_targets: dict[str, Any], model, spec,
            hp: dict[str, Any]) -> Any:
        """Fit parameters on a set of fit-split targets. No-op (returns None) for stateless
        methods (``trained=False``)."""
        return None

    def predict(self, spec, model, conds: list[dict[str, Any]], hp: dict[str, Any],
                cfg: RunConfig, fit_state: Any = None) -> dict[str, dict[str, Any]]:
        """Return ``{condition_key: {predicted_rate, ...}}`` for ``conds``."""
        raise NotImplementedError

    def applies_to(self, spec) -> bool:
        """Whether this method should run against eval ``spec`` at all — the relevance gate.

        The benchmark calls this (via ``methods.valid_combo``) for every (method, model, eval)
        combination and **silently skips** the pairings where it returns ``False``: a skip is
        "this method makes no sense for this eval", not an error. Override it to restrict a
        method to the evals it can actually predict. Examples in this codebase: the
        value method needs a driver/brake *value* pair, so it overrides this to return
        ``spec.has_value_framing`` (skipping e.g. capability, which has none). The default below
        also skips every trained method on a ``test_only`` eval (empty dev pool — nothing to fit
        on); non-trained methods apply everywhere."""
        return not (self.trained and spec.test_only)

    def method_id(self, hp: dict[str, Any]) -> str:
        """Stable id (base + ``-concrete``/``-honest`` suffixes), matching existing filenames."""
        return common.method_id(self.base_method, hp)

    def runs(self, cfg: RunConfig) -> int:
        return cfg.runs if cfg and cfg.runs else self.default_runs
