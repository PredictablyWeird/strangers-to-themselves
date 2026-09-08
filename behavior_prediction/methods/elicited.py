"""Adapters wrapping the existing stateless elicitation methods.

Each adapter's ``predict`` delegates to the single-sourced primitives in ``elicitation.py``
(``elicit_over_conditions``, ``elicit_value_over_conditions``,
``behavioral_sampling_predict``), so the prompts and aggregation are identical to the
standalone runner scripts — and the prediction dicts they return match the committed
``predictions/*.json`` shape. ``fit`` is the inherited no-op.
"""

from __future__ import annotations

from typing import Any

from behavior_prediction import common, elicitation
from behavior_prediction.methods.base import Capabilities, MethodAdapter, RunConfig


def _is_bias_contrast(spec) -> bool:
    return getattr(spec, "scoring_semantics", "") == "bias_contrast"


def _checkpoint(cfg) -> str | None:
    """Resume sidecar for this run, derived from the prediction out path the benchmark set on
    ``cfg`` (``None`` when checkpointing is disabled, e.g. direct library use)."""
    base = getattr(cfg, "checkpoint_base", None)
    return elicitation._checkpoint_path(base) if base else None


def _contrast_scope(conds):
    """Infer the categories and per-axis comparison values present in ``conds`` so the comparative
    bias prediction covers only the contrasts the measurement can score (carries the dev/test split
    and any axis/value subsampling through to prediction)."""
    from collections import defaultdict
    scenarios = sorted({c["scenario"] for c in conds})
    comps: dict[str, set] = defaultdict(set)
    for c in conds:
        if c.get("axis") not in (None, "baseline"):
            comps[c["axis"]].add(c["value"])
    comparisons = {ax: sorted(vs, key=str) for ax, vs in comps.items()} or None
    return scenarios, comparisons


def _predict_comparative_bias(method, spec, model, conds, hp, cfg):
    """Predict each contrast's signed gap directly (contrast-keyed), via ``method``'s framing.
    The introspective methods route here on ``bias_contrast`` evals; their ``default_runs`` are
    even, as the A/B-order swap requires."""
    scenarios, comparisons = _contrast_scope(conds)
    return elicitation.elicit_comparative_bias(
        spec, model.full, scenarios, runs=method.runs(cfg), comparisons=comparisons,
        base_method=method.base_method, params=hp, temperature=cfg.temperature,
        max_parse_retries=cfg.max_parse_retries, concurrency=cfg.concurrency,
        reasoning_config=model.reasoning_cfg, checkpoint_path=_checkpoint(cfg))


class _PromptMethod(MethodAdapter):
    """One prompt per condition; its swept args (e.g. self_report's ``honesty_nudge``) come from
    ``methods.yaml`` via the base ``hyperparameter_grid``. On a ``bias_contrast`` eval it instead
    predicts each contrast's gap directly (comparative prompt)."""
    builder = None

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        if _is_bias_contrast(spec):
            return _predict_comparative_bias(self, spec, model, conds, hp, cfg)
        return elicitation.elicit_over_conditions(
            spec, model.full, conds, self.builder, params=hp,
            runs=self.runs(cfg), temperature=cfg.temperature,
            max_parse_retries=cfg.max_parse_retries, concurrency=cfg.concurrency,
            reasoning_config=model.reasoning_cfg, checkpoint_path=_checkpoint(cfg))


class SelfReport(_PromptMethod):
    """Direct self-report. ``honesty_nudge`` is swept; see ``methods.yaml``. On a ``bias_contrast``
    eval (DiscrimEval) it predicts the demographic gap directly."""
    base_method = "self_report"
    builder = staticmethod(elicitation.self_report_prompt)
    default_runs = 10


class GenericReport(_PromptMethod):
    """self_report with the subject swapped to *capable AI agents in general* — the paired control
    that isolates the indexical component of testimony (does "you" add anything over "a typical
    assistant"?). Same conditions, runs, and parsing as self_report; unswept. On a
    ``bias_contrast`` eval it predicts the demographic gap comparatively, with the third-person
    framing handled in ``discrimeval.bias_prompt``."""
    base_method = "generic_report"
    builder = staticmethod(elicitation.generic_report_prompt)
    default_runs = 10


class Value(MethodAdapter):
    """Merged value-conflict method (even runs; A/B order swapped on half). Frames the condition's
    two short, canonical values and asks which the model prefers as a graded estimate; ``aggregate``,
    ``value_detail`` and ``context`` are swept (see ``methods.yaml``). Conditions that resolve to the
    same value pair render an identical prompt and collapse (elicited once, reused). On a
    ``bias_contrast`` eval it predicts the demographic gap directly with the value-scale framing."""
    base_method = "value"
    default_runs = 10

    def applies_to(self, spec) -> bool:
        return spec.has_value_framing

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        if _is_bias_contrast(spec):
            return _predict_comparative_bias(self, spec, model, conds, hp, cfg)
        return elicitation.elicit_value_over_conditions(
            spec, model.full, conds, params=hp,
            runs=self.runs(cfg), temperature=cfg.temperature,
            max_parse_retries=cfg.max_parse_retries, concurrency=cfg.concurrency,
            reasoning_config=model.reasoning_cfg, checkpoint_path=_checkpoint(cfg))


class ListExperiment(_PromptMethod):
    """Item-count (list-experiment) technique. Instead of asking directly (which the model denies),
    it asks how *many* actions from a list it would take — control (neutrals) vs treatment
    (neutrals + the sensitive item) — and estimates the rate as the difference in mean counts.
    Neutral subsets are re-drawn per run (deniability) and the control is shared across conditions
    (variance: a shared control is a constant offset that correlation ignores). ``list_size``
    is swept (see ``methods.yaml``).
    Stateless / target-queried, like self_report."""
    base_method = "list_experiment"
    builder = staticmethod(elicitation.list_experiment_prompt)  # for report exhibits only
    #: The estimate is a difference of mean counts (per-run count SD ~0.9, partly the deliberate
    #: neutral-subset randomisation), so its SD shrinks as 1/sqrt(runs). At 30 runs the estimate
    #: noise still halved the observable correlation on the higher-variance cells; 100 runs
    #: recovers ~0.9 of the true r there (target-model calls are cheap — the analyst isn't used).
    default_runs = 100

    def applies_to(self, spec) -> bool:
        # The count question is about *willingness to act*, so it needs a value-framed behaviour
        # (like value/behavioral_sampling; capability evals have none). And it estimates an
        # absolute per-condition rate — the comparative-bias (contrast) framing has no natural
        # list form — so bias_contrast evals (e.g. DiscrimEval) are skipped too.
        return spec.has_value_framing and not _is_bias_contrast(spec)

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        return elicitation.list_experiment_predict(
            spec, model.full, conds, params=hp, runs=self.runs(cfg),
            temperature=cfg.temperature, max_parse_retries=cfg.max_parse_retries,
            concurrency=cfg.concurrency, reasoning_config=model.reasoning_cfg,
            checkpoint_base=getattr(cfg, "checkpoint_base", None))


class Pairwise(MethodAdapter):
    """Comparative elicitation: show two conditions' situations and force a choice of where the
    action is more likely; a condition's prediction is its win rate over ``rounds`` seeded random
    matchings (both A/B orders asked, position bias cancels). The win rate is a monotone *score*,
    not a calibrated rate — correlation is the target metric; MAE is meaningless for it.

    ``indifference`` (swept; default off) adds an EQUAL verdict worth 0.5 to both sides. Off by
    default because the forced choice exists to close the "no difference" escape hatch that
    collapses absolute self-reports to a constant; the tuner decides whether allowing it helps.
    ``rounds``/``seed`` are ``--method-config`` attributes."""
    base_method = "pairwise"
    calibrated = False
    capabilities = Capabilities(needs_all_conditions=True)  # ranks the sweep; no per-cond prompt
    default_runs = 2   # per pair: one sample per A/B order; comparisons/condition come from rounds

    def __init__(self, *, rounds: int = 8, seed: int = 1234) -> None:
        self.rounds = rounds
        self.seed = seed

    def applies_to(self, spec) -> bool:
        # Head-to-head comparison of two demographic variants IS the discrimination question —
        # models flatten or refuse it — so bias_contrast evals (e.g. DiscrimEval) are skipped;
        # they keep their dedicated comparative-bias path.
        return not _is_bias_contrast(spec)

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        return elicitation.pairwise_predict(
            spec, model.full, conds, params=hp, runs=self.runs(cfg), rounds=self.rounds,
            seed=self.seed, temperature=cfg.temperature, max_parse_retries=cfg.max_parse_retries,
            concurrency=cfg.concurrency, reasoning_config=model.reasoning_cfg,
            checkpoint_path=_checkpoint(cfg))


class _OracleMethod(MethodAdapter):
    """Shared plumbing for the methods built on ``EvalSpec.oracle_items`` (the verbatim measured
    items): the applicability gate and the ``k_items`` cap, which comes from ``methods.yaml``
    unless overridden per instance (``--method-config``) or swept."""
    DEFAULT_K = 20

    def __init__(self, *, k_items: int | None = None) -> None:
        self.k_items = k_items  # --method-config override (smoke runs); None -> methods.yaml

    def applies_to(self, spec) -> bool:
        from behavior_prediction.evals.base import EvalSpec
        return type(spec).oracle_items is not EvalSpec.oracle_items

    def _k(self, hp) -> int:
        if hp and "k_items" in hp:              # only if the yaml value is ever swept/non-default
            return int(hp["k_items"])
        if self.k_items is not None:
            return int(self.k_items)
        from behavior_prediction import common
        vals = common._method_args(self.base_method).get("k_items") or []
        return int(vals[0]) if vals else self.DEFAULT_K


class InformedOracle(_OracleMethod):
    """Informed-oracle upper bound on verbal self-prediction: for each measured item of a
    condition, show the model (almost) full information — the verbatim prompt(s) sent during
    measurement plus a description of the protocol, WITHOUT eliciting the behavior itself —
    and ask it to predict its own per-item score. ``predicted_rate`` = mean over the
    condition's parsed item predictions. Deliberately over-informed: a diagnostic ceiling for
    what verbal self-prediction could achieve with maximal item information, not a deployable
    method. Applies only to evals implementing ``EvalSpec.oracle_items``.

    One call per item (``default_runs = 1``), greedy where the model allows it (models run
    WITH a reasoning config keep the default sampling temperature — reasoning endpoints may
    reject or ignore temperature 0). Run it in its own ``bp-benchmark`` invocation: a global
    ``--runs N`` would multiply the per-item calls. ``k_items`` (max items per condition) is
    configured in ``methods.yaml``, not swept.

    Grain note (sycophancy): the target is flips/initially-correct pooled over all measured
    questions, while this method equal-weights its first-k items' *conditional* per-item flip
    predictions (including items the model would in fact answer wrong, which the measurement
    drops from the denominator). Fine for an upper bound on correlation; don't read the MAE
    too literally.

    Grain note (propensitybench): items are (task-scenario, pressure-tactic) runs, so at the
    default task_scenario grain the item mean pools exactly what the target pools."""
    base_method = "informed_oracle"
    default_runs = 1            # one call per item; runs>1 would average samples per item
    GREEDY_TEMPERATURE = 0.0

    def _item_prompt(self, protocol, exhibit, ask, instruction):
        """The per-item prompt. The seam ``GenericOracle`` overrides to swap the SUBJECT while
        keeping every exhibit, ask, parser, and grain identical."""
        return elicitation.informed_oracle_prompt(protocol, exhibit, ask,
                                                  instruction=instruction)

    def _item_prompt_from(self, view, item, instruction):
        """View-level hook (the whole ``oracle_items`` view + one item), for arms that need more
        than (protocol, exhibit, ask) — the phrasing-ablation arms read ``protocol_3p`` /
        ``ask_gen``. Default: the classic ``_item_prompt`` seam."""
        return self._item_prompt(view["protocol"], item["exhibit"], item["ask"], instruction)

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        from statistics import mean
        from behavior_prediction import common
        k = self._k(hp)
        # On a bias_contrast eval the oracle predicts the signed group-vs-baseline gap directly
        # (contrast-keyed, signed tag/parser); score_contrasts consumes it as-is. Elsewhere it
        # predicts an absolute per-condition rate.
        bias = _is_bias_contrast(spec)
        instruction = (elicitation.ORACLE_SIGNED_PREDICTION_INSTRUCTION if bias
                       else elicitation.ORACLE_PREDICTION_INSTRUCTION)
        parse_fn = common.parse_signed_prediction_tag if bias else common.parse_prediction_tag
        prompts: dict[str, str] = {}
        unit_items: dict[str, list[str]] = {}
        unit_cond: dict[str, dict[str, Any]] = {}
        signs: dict[str, int] = {}      # per prompt key; +1 unless an item flips it (bias swap)
        for c in conds:
            if bias:
                contrast = spec.condition_contrast(c)
                if contrast is None:            # baseline cell: no gap to predict
                    continue
                unit = contrast["key"]
            else:
                unit = spec.condition_key(c)
            # oracle_items applies its own k cap in whatever unit keeps its items balanced (bias
            # returns two applicant orders per template; PB returns one item per pressure tactic of
            # a whole task-scenario), so we do not re-slice here.
            view = spec.oracle_items(c, k) or {"items": []}
            keys = []
            for it in view["items"]:
                key = f"{unit}//{it['item_id']}"
                prompts[key] = self._item_prompt_from(view, it, instruction)
                signs[key] = it.get("sign", 1)
                keys.append(key)
            unit_items[unit] = keys
            unit_cond[unit] = c
        temperature = cfg.temperature if model.reasoning_cfg else self.GREEDY_TEMPERATURE
        res = common.elicit_rates(
            model.full, prompts, runs=self.runs(cfg), temperature=temperature,
            max_parse_retries=cfg.max_parse_retries, concurrency=cfg.concurrency,
            parse_fn=parse_fn, reasoning_config=model.reasoning_cfg,
            checkpoint_path=_checkpoint(cfg))
        out: dict[str, dict[str, Any]] = {}
        for unit, keys in unit_items.items():
            c = unit_cond[unit]
            entries = [(key, res[key]) for key in keys]
            # Apply each item's sign so both applicant orders express the same canonical gap
            # (group−baseline); a constant position bias cancels in the mean.
            item_rates = {key.split("//", 1)[1]:
                          (None if e["predicted_rate"] is None else e["predicted_rate"] * signs[key])
                          for key, e in entries}
            vals = [v for v in item_rates.values() if v is not None]
            out[unit] = {
                # For bias this is the signed gap (fraction, group−baseline); else an absolute rate.
                "predicted_rate": mean(vals) if vals else None,
                "n": len(vals),                       # items that parsed
                "k_items": len(entries),              # items asked
                "item_rates": item_rates,             # per-item diagnostics
                "parse_failures": sum(e.get("parse_failures", 0) for _, e in entries),
                # raw/reasoning flattened in item order stay index-aligned (runs=1 per item),
                # which split_reasoning relies on.
                "raw": [t for _, e in entries for t in (e.get("raw") or [])],
                "reasoning": [r for _, e in entries for r in (e.get("reasoning") or [])],
                "scenario": c["scenario"], "condition": c,
            }
        return out


class GenericOracle(InformedOracle):
    """``informed_oracle`` with the SUBJECT swapped to a generic capable AI agent — the paired
    control at the top of the asking channel.

    Everything is held fixed: the same verbatim exhibits from ``spec.oracle_items``, the same
    protocol text, the same per-item ask, the same signed/absolute parser, the same item grain
    and ``k_items``. Only the framing changes (``elicitation.other_oracle_prompt``), so the
    difference between the two methods isolates exactly what the model's *self*-reference adds
    once maximal item information is on the table. If the informed oracle's advantage over the
    outside view were self-knowledge, this control should lose to it; if the two match, even the
    best form of asking is reading a generic prior off the item.

    Scored against the answering model's own measured behavior, exactly like the informed
    oracle (this is a prediction of the asker, made without asking about the asker)."""
    base_method = "generic_oracle"

    def _item_prompt(self, protocol, exhibit, ask, instruction):
        return elicitation.other_oracle_prompt(protocol, exhibit, ask, instruction=instruction)


class OraclePairwise(_OracleMethod):
    """Pairwise version of ``informed_oracle``: same maximal-information exhibits (the verbatim
    measured items of a condition), but the model is asked for an ORDERING instead of a number —
    two units' items side by side, forced choice of which item its own answer would be higher on.
    A unit's prediction is its win rate over ``rounds`` seeded matchings, both sides' orders asked.

    Motivation: the oracle's absolute per-item numbers are badly quantized (and on a bias eval most
    items come back a flat 0 — the model won't put a number on its own discrimination), which caps
    the correlation the *upper bound* can reach for reasons that have nothing to do with
    self-knowledge. A forced choice needs only ordinal self-knowledge, so it separates "the model
    cannot rank its own items" from "the model will not quantify them". Like ``pairwise``, the
    output is a monotone score in [0, 1]: correlation is the metric, MAE is meaningless for it.

    On a bias_contrast eval (DiscrimEval) the units are contrasts and the item ask is the signed
    gap question, so "higher" means a larger group-vs-baseline gap; the prediction is contrast-keyed
    and ``score_contrasts`` consumes it directly, exactly as for ``informed_oracle``.

    ``indifference`` (an EQUAL verdict worth 0.5) is off and not swept — like ``pairwise``, the
    forced choice exists to close the "no difference" escape hatch. ``rounds``/``seed``/``k_items``
    are ``--method-config`` attributes."""
    base_method = "oracle_pairwise"
    calibrated = False
    capabilities = Capabilities(needs_all_conditions=True)  # ranks the sweep; no per-cond prompt
    default_runs = 2            # per pair: one sample per X/Y order

    def __init__(self, *, k_items: int | None = None, rounds: int = 8, seed: int = 1234,
                 indifference: bool = False) -> None:
        super().__init__(k_items=k_items)
        self.rounds = rounds
        self.seed = seed
        self.indifference = indifference

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        params = {"indifference": self.indifference, **(hp or {})}
        return elicitation.oracle_pairwise_predict(
            spec, model.full, conds, params=params, runs=self.runs(cfg), k_items=self._k(hp),
            rounds=self.rounds, seed=self.seed, temperature=cfg.temperature,
            max_parse_retries=cfg.max_parse_retries, concurrency=cfg.concurrency,
            reasoning_config=model.reasoning_cfg, checkpoint_path=_checkpoint(cfg))


class BehavioralSampling(MethodAdapter):
    """Measure-don't-introspect: generate scenario SCRIPTS (the generator decides 1..3 user
    turns from the Frame; later turns are injected after the model's own replies — 2026-08-08,
    previously single-turn), sample behavior, grade the full exchange. Needs a generator +
    grader model. Output is inherently stochastic, so it reproduces the committed file's
    *shape*, not its exact rates."""
    base_method = "behavioral_sampling"
    capabilities = Capabilities(abstract_description_only=True)

    def applies_to(self, spec) -> bool:
        # 2026-08-09: the value-framing gate is dropped — the generation prompt uses only the
        # Frame's setting/situation/action fields, never the value pair, so any eval with a
        # Frame supports sampling (capability_mmlu included: proxy questions, graded for
        # correctness). Every registered eval implements ``frame``.
        return True

    #: 2026-08-09: GRADING switched from Claude Sonnet 4 — on 220 stored Sonnet-graded exchanges
    #: across four evals, this grader agrees 97.7% (kappa 0.94) with matched yes-rates at
    #: ~30-60x lower cost. When the SUBJECT model is gemini, pass a different cheap grader
    #: (--method-config behavioral_sampling:grader_model=... / informed_sampling:...) so no model
    #: grades its own outputs.
    DEFAULT_GRADER = common.DEFAULT_GRADER  # the project-wide standard grader (see common.py)
    #: GENERATION no longer follows the grader default (the k=25 run that silently generated
    #: with flash-lite produced contract violations, points on correct answers, and ceiling-easy
    #: questions — generation quality is the method's backbone). 2026-08-09 generator audit
    #: (scripts/pilot_generator_audit.py) on the observed failure modes: gpt-5.5-off is the
    #: cleanest candidate — 0 format violations (sonnet-4: 16, sonnet-5-off: 17), 100% correct
    #: turn-structure choices (sonnet-5-off: 47%), points-on-correct 3% (flash-lite: 46%), and
    #: generated-question difficulty matching the subject's measured accuracy exactly (81% vs
    #: target 81%). Model shortcuts resolve via models.yaml; the generator's reasoning config is
    #: carried through to the generation calls.
    DEFAULT_GENERATOR = "gpt-5.5-off"

    #: k=25 x r=2 (50 samples/condition) is the paper-grade budget: at the historical 10x1 a
    #: condition estimate had run-to-run reliability ~0.2 and floor-rate behaviors (2-6%)
    #: were unresolvable.
    def __init__(self, *, grader_model: str | None = None, generator_model: str | None = None,
                 scenarios_per_condition: int = 25, samples_per_scenario: int = 2,
                 generation_retries: int = 3) -> None:
        self.grader_model = grader_model or self.DEFAULT_GRADER
        self.generator_model = generator_model or self.DEFAULT_GENERATOR
        self.k = scenarios_per_condition
        self.r = samples_per_scenario
        self.retries = generation_retries

    def predict(self, spec, model, conds, hp, cfg, fit_state=None):
        from behavior_prediction import common
        grader, _ = common.resolve_model(self.grader_model)
        generator, gen_reasoning = common.resolve_model(self.generator_model)
        base = getattr(cfg, "checkpoint_base", None)
        if _is_bias_contrast(spec):
            # Paired/comparative: same generated case under each demographic, differenced per case
            # (the only version on bias_contrast evals — see comparative_behavioral_sampling_predict).
            return elicitation.comparative_behavioral_sampling_predict(
                spec, model.full, generator, grader, conds,
                k=self.k, r=self.r, retries=self.retries,
                temperature=cfg.temperature, concurrency=cfg.concurrency,
                reasoning_cfg=model.reasoning_cfg, checkpoint_base=base,
                generator_reasoning=gen_reasoning)
        return elicitation.behavioral_sampling_predict(
            spec, model.full, generator, grader, conds,
            k=self.k, r=self.r, retries=self.retries,
            temperature=cfg.temperature, concurrency=cfg.concurrency,
            reasoning_cfg=model.reasoning_cfg, checkpoint_base=base,
            generator_reasoning=gen_reasoning)
