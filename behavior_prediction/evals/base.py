"""The eval-adapter seam: ``EvalSpec`` + the ``Frame`` struct the prompt templates consume.

A *behavior-prediction eval* (agentic_misalignment, propensitybench, ...) is described by an
``EvalSpec`` subclass. The generic core (``elicitation.py``, the report scripts) is written
entirely against this interface, so adding an eval means adding one ``EvalSpec`` — never
touching the prediction methods (which are the research contribution and stay single-sourced).

An ``EvalSpec`` owns four things:
  1. **Metadata** — the scenarios, the condition axes + defaults, the metrics, the named
     sweeps, and the report ordering.
  2. **Condition helpers** — build/serialize/label a *condition* (one cell we predict).
  3. **The prompt seam** — ``frame(cond)`` returns a ``Frame`` of eval-specific text slots;
     the three shared prompt templates in ``elicitation.py`` read those slots.
  4. **Targets** — ``run_behavior(args)`` measures actual behavior; ``produce_targets(args)``
     turns that into the standard ``targets.json`` doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Frame:
    """The eval-specific text slots the prompt templates compose.

    Every method prompt is ``(eval-specific situation) + (shared scaffolding)``; this struct
    is the eval-specific part for one condition. The slots:

    - ``setting`` — concrete situation paragraph (the scene the model is in).
    - ``target_action`` — the concrete action whose rate we predict (used by self_report and
      the value method at ``context=concrete``).
    - ``action_frame`` — an abstract one-line paraphrase of that action.
    - ``driver_value`` — Value A: a **short, canonical value label** that pushes toward the
      harmful action *for this condition*. Resolved via the eval's own hierarchy fallback: a
      condition gets its value pair at the finest level the eval defines it, and inherits the
      parent value conflict when its more-specific setting doesn't change which values are
      involved (so such conditions render an identical value prompt and collapse).
    - ``driver_value_detail`` — an optional parenthetical gloss for ``driver_value`` (appended
      when the value method's ``value_detail`` hyperparameter is on; ``""`` for none).
    - ``brake_value`` — Value B: the short, canonical safety/refusal value label.
    - ``brake_value_detail`` — optional parenthetical gloss for ``brake_value`` (``""`` for none).
    - ``situation_context`` — an abstract "In this particular case, ..." clause distinguishing
      this condition (its non-value-changing specifics). The value method appends it at
      ``context=abstract``/``concrete`` to make otherwise-collapsed conditions distinct
      (``""`` when there is nothing distinguishing).
    - ``self_report_situation`` — the middle paragraph of the self_report prompt, between
      ``setting`` and the "Out of 100 ..." question (e.g. the goal+threat sentence).
    """

    setting: str
    target_action: str
    action_frame: str = ""
    #: The value pair (Value A / Value B). Default ``""`` for evals whose value method predicts
    #: contrasts directly (``bias_contrast``, e.g. DiscrimEval) and so never reads these slots.
    driver_value: str = ""
    brake_value: str = ""
    driver_value_detail: str = ""
    brake_value_detail: str = ""
    situation_context: str = ""
    self_report_situation: str = ""
    #: Short name of the knowledge domain this condition exercises (e.g. the MMLU subject
    #: phrase), used by ``cot_flip_prompt``'s self-assessment clause ("how solid is your
    #: {domain} knowledge"). Naming the domain explicitly is load-bearing there — a generic
    #: "the subject matter of this situation" loses most of the signal (see
    #: docs/single-model-methods-sycophancy.md). ``""`` falls back to a domain-free phrasing.
    domain: str = ""
    #: Optional clause restricting which "situations" ``cot_flip_prompt``'s rate question is
    #: over, e.g. sycophancy's " where your first answer was actually correct" (leading space,
    #: appended to "Out of 100 such situations"). Matches the measurement's conditioning — and
    #: is load-bearing for some models: dropping it cost deepseek most of its sycophancy signal
    #: (raw r +0.46 with vs +0.11 without, same batch). ``""`` for unconditioned rates.
    rate_qualifier: str = ""
    #: Third-person twins of ``setting`` / ``self_report_situation`` / ``target_action`` for the
    #: self-vs-generic phrasing ablation: the same facts in the
    #: same order with the reader absent — the subject is "an AI assistant" / "the assistant" (or
    #: "the agent" on the agentic evals) and no "you/your/yourself" appears. Hand-written per eval
    #: (a pronoun regex would break agreement and possessives); ``""`` = no 3p version, and the
    #: ``self_report_3p*`` arms then skip the eval. ``target_action_3p`` is also the
    #: generic-subject action for the ``genmin`` arm ("would a typical capable AI assistant
    #: {target_action_3p}"), so it must read after a third-person subject ("its answer").
    setting_3p: str = ""
    self_report_situation_3p: str = ""
    target_action_3p: str = ""

    def has_3p(self) -> bool:
        """Whether this frame carries the full third-person rewrite."""
        return bool(self.setting_3p and self.self_report_situation_3p and self.target_action_3p)


class EvalSpec:
    """Interface every eval adapter implements. Subclasses set the metadata attributes and
    override the condition / prompt / targets methods. Generic defaults are provided where
    the behavior is genuinely eval-independent."""

    # --- Metadata (subclasses set these) --------------------------------------
    name: str = ""
    #: Default model under test for this eval's runner scripts.
    default_model: str = ""
    scenarios: list[str] = []
    #: Every key is a condition axis; ``make_condition`` fills these defaults.
    condition_defaults: dict[str, Any] = {}
    metrics: list[str] = []
    default_metric: str = ""
    #: Scorer name in the behavior logs (``None`` for evals that don't score via a scorer).
    scorer_name: str | None = None
    #: name -> ``fn(scenarios) -> list[condition]``.
    sweeps: dict[str, Callable[[list[str]], list[dict[str, Any]]]] = {}
    default_sweep: str = ""
    #: Canonical display order per axis for the report (axis -> ordered values).
    report_axis_order: dict[str, list[Any]] = {}
    #: Axes the report may produce a by-axis breakdown for.
    report_candidate_axes: list[str] = []
    #: The condition axis used as the secondary sort in report key ordering.
    report_secondary_axis: str | None = None
    #: Report header text.
    report_title: str = "Behavior-prediction report"
    #: When False, the detailed report omits the per-condition table (one row per condition) and
    #: relies on the by-axis rollups (``report_candidate_axes``) instead — for evals with hundreds
    #: of conditions where a per-condition dump is unreadable (e.g. MASK's per-proposition grain).
    report_per_condition: bool = True

    #: When True, the eval is not in a trustworthy/affordable state to *run* (mis-specified grain,
    #: pending redesign, prohibitive measurement cost, ...). The eval-running CLIs refuse to
    #: generate/measure it unless ``--force`` is passed; ``broken_reason`` explains why. Read-only
    #: inspection/scoring of existing results is unaffected. Library access via ``get_spec`` is also
    #: unguarded — only CLI entry enforces the gate.
    broken: bool = False
    broken_reason: str = ""

    #: Self-vs-generic phrasing ablation: ``frame_3p`` declares
    #: that ``frame`` fills the ``*_3p`` slots for every condition; ``oracle_3p`` that every
    #: ``oracle_items`` view carries ``protocol_3p`` and per-item ``ask_gen``. The ablation arms
    #: apply only where the flag is set; tests/test_selfgeneric_ablation.py verifies the flags.
    frame_3p: bool = False
    oracle_3p: bool = False

    # --- Split / scoring metadata ---------------------------------------------
    #: Human grain of the split unit (doc-only, e.g. "scenario", "workspace", "template").
    scenario_granularity: str = "scenario"
    #: How predictions are scored: "absolute_rate" (correlate predicted vs actual rate per
    #: condition) or "bias_contrast" (correlate signed group-vs-baseline gaps; see
    #: ``score_contrasts``). For a "bias_contrast" eval the introspective methods predict each
    #: contrast's gap directly via ``comparative_bias_prompts``.
    scoring_semantics: str = "absolute_rate"
    #: Whether the eval has a driver/brake *value* pair. False for non-harm evals (e.g.
    #: capability), so the value / value_aggregate methods opt out via ``applies_to``.
    has_value_framing: bool = True
    #: Too few split units to split: every unit is forced to ``test`` and only stateless
    #: (untrained) methods are scored. Trained methods opt out via ``applies_to``.
    test_only: bool = False

    def split_unit(self, cond: dict[str, Any]) -> str:
        """The dev/test split group a condition belongs to. Conditions sharing a split unit are
        never spread across dev and test. Default: the scenario."""
        return cond["scenario"]

    def fold_group(self, unit: str) -> str:
        """The cross-validation grouping a split *unit* belongs to within the dev pool. CV holds
        out whole groups at a time, so correlated units (e.g. all PropensityBench workspaces in
        one scenario) stay together. Default: the part before the first ``/`` — which makes
        ``scenario/workspace`` units group by scenario and slash-free unit ids (e.g. a DiscrimEval
        ``q123`` template) each their own group. Override to group differently."""
        return unit.split("/", 1)[0]

    def split_units(self) -> list[str] | None:
        """The **canonical** list of all split-unit ids (the split universe), enumerated from the
        eval definition (dataset / harness inputs) — **model-independent**, so everyone who builds
        a manifest gets the same split. Returns None when the universe can't be enumerated without
        measurement (then the builder falls back to the units present in a targets doc). The
        ids must match what :meth:`split_unit` returns for a condition."""
        return None

    def forced_split(self, unit: str) -> str | None:
        """If a split unit must be placed in a specific split regardless of the random
        partition, return that split; else None. Used e.g. to force every PropensityBench
        ``cyber-security/*`` workspace into ``test`` (a domain-generalization signal).
        Data-driven (per unit id) so the eval needn't enumerate units. Default: None."""
        return None

    def score_contrasts(self, targets: dict[str, Any],
                        preds: dict[str, float | None]) -> dict[str, tuple[float, float]]:
        """For ``scoring_semantics == "bias_contrast"`` only: map each contrast key to an
        ``(actual_delta, predicted_delta)`` pair (group rate minus baseline rate, on each
        side). The leaderboard feeds these through ``metrics.contrast_pairs``. Default: none."""
        return {}

    def condition_contrast(self, cond: dict[str, Any]) -> dict[str, Any] | None:
        """For ``bias_contrast`` evals only: the bias contrast a measured condition (cell)
        belongs to, as ``{"key": contrast_key, "label": human description, "scenario": ...}`` —
        or ``None`` for the baseline cell (it anchors every contrast but is none itself).
        Lets contrast-grain methods (e.g. ``llm_prediction``) map cells to the contrasts they
        predict. Default: ``None`` (no contrast grain)."""
        return None

    def comparative_bias_prompts(self, scenarios: list[str] | None = None,
                                 comparisons: dict[str, list[Any]] | None = None, *,
                                 base_method: str = "self_report",
                                 params: dict[str, Any] | None = None
                                 ) -> dict[str, dict[str, str]] | None:
        """For ``bias_contrast`` evals only: the comparative prompts the introspective methods send
        to predict each contrast's gap directly, as ``{contrast_key: {"o0": prompt, "o1":
        swapped_prompt}}`` (each contrast asked in both A/B orders; the swapped order's parsed delta
        is negated and pooled). ``base_method`` + ``params`` select the prompt framing and apply that
        method's hyperparameters. Restrict to ``scenarios`` / ``comparisons`` (an ``{axis: [values]}``
        map) when given, so a subsampled run predicts only what it measured. Default: ``None`` (the
        comparative path doesn't apply to this eval)."""
        return None

    def parse_bias_gap(self, text: str | None) -> float | None:
        """Parse a comparative-bias reply into a signed gap fraction in [-1, 1] (``None`` if
        unparseable). Used as the ``parse_fn`` for the comparative-bias elicitation. Default: None."""
        return None

    # --- Condition helpers ----------------------------------------------------
    def make_condition(self, scenario: str, **overrides: Any) -> dict[str, Any]:
        """Build a full condition dict with defaults filled. Override to add validation."""
        cond: dict[str, Any] = {"scenario": scenario, **self.condition_defaults}
        cond.update(overrides)
        return cond

    def condition_key(self, cond: dict[str, Any]) -> str:
        """Stable, readable key used to align predictions with targets."""
        raise NotImplementedError

    def condition_label(self, cond: dict[str, Any]) -> str:
        """Short human label for tables."""
        raise NotImplementedError

    def task_args(self, cond: dict[str, Any]) -> dict[str, Any]:
        """The condition as eval task args (default: the condition verbatim)."""
        return dict(cond)

    def build_sweep(self, name: str, scenarios: list[str],
                    model: str | None = None) -> list[dict[str, Any]]:
        """The conditions to predict over. ``model`` is the model under test; most evals ignore it
        (their sweeps are model-independent), but evals whose condition set is derived from measured
        targets (PropensityBench) use it to read *that* model's targets so predictions align with
        what the model actually measured."""
        if name not in self.sweeps:
            raise ValueError(f"unknown sweep {name!r}; choose from {sorted(self.sweeps)}")
        return self.sweeps[name](scenarios)

    def method_grid(self, method) -> list[dict[str, Any]]:
        """The hyperparameter settings to search for ``method`` on *this* eval. Default: the
        method's full grid from ``methods.yaml``, minus any ``value`` setting whose prompt is
        **identical across every condition** of this eval — a constant prompt yields a constant,
        non-ranking prediction, so running it only adds a noisy zero to the leaderboard. This is
        data-driven: a setting is kept iff its rendered prompt actually varies (e.g. on Sycophancy
        the value pair is the same for every subject, so plain ``value`` / ``-aggregate`` /
        ``-detailed`` collapse to one prompt and are dropped, while ``-context`` / ``-concrete``,
        which inject the per-subject situation, are kept). An eval may still override to prune
        further (e.g. DiscrimEval's value-scale prompt), so the benchmark and tuner agree on the
        same per-eval settings."""
        grid = method.hyperparameter_grid()
        if getattr(method, "base_method", None) != "value":
            return grid
        try:
            conds = self.build_sweep(self.default_sweep, self.scenarios)
        except (Exception, SystemExit):
            return grid                          # can't introspect conditions -> keep the full grid
        if len(conds) < 2:
            return grid                          # nothing to rank -> don't prune on one condition
        from behavior_prediction.elicitation import value_prompt
        kept: list[dict[str, Any]] = []
        for hp in grid:
            try:
                rendered = {value_prompt(self, c, hp) for c in conds}
            except (Exception, SystemExit):
                return grid                      # rendering failed -> don't guess, keep the grid
            if len(rendered) > 1:                # varies across conditions -> can rank -> keep
                kept.append(hp)
        kept = kept or grid                      # never prune the method away entirely
        # PINNED 2026-07-24 (sweep retired): wherever the concrete-context setting survives the
        # variance prune it is the tuned winner on every selection-pool eval, so future runs
        # generate only it instead of re-sweeping the survivors. (DiscrimEval, where `context`
        # is moot, pins `value-aggregate` in its own method_grid override.)
        concrete = [hp for hp in kept if hp.get("context") == "concrete"]
        return concrete or kept

    # --- Prompt seam ----------------------------------------------------------
    def frame(self, cond: dict[str, Any]) -> Frame:
        """Build the eval-specific text slots for one condition. Resolves the condition's value
        pair via the eval's hierarchy fallback and fills ``situation_context`` with an abstract
        clause distinguishing this condition (its non-value-changing specifics)."""
        raise NotImplementedError

    # --- Behavioral prompt (for the report's Example-prompts section) ---------
    def behavioral_prompt(self, cond: dict[str, Any]) -> dict[str, str]:
        """The prompt(s) the model saw when behavior was measured, as ``{label: text}``.
        The report renders each entry generically, so the key set is free."""
        raise NotImplementedError

    def behavioral_group(self, cond: dict[str, Any]) -> str | None:
        """Optional group label used to cluster the report's 'behavioral (eval)' example prompts
        into nested sections (e.g. MASK groups by ``domain``, mirroring how DiscrimEval's list
        clusters by category). Return ``None`` (default) for a flat, ungrouped list."""
        return None

    def oracle_items(self, cond: dict[str, Any], k: int) -> dict[str, Any] | None:
        """Per-item exhibits for the ``informed_oracle`` upper-bound method: the verbatim
        material of up to ``k`` of this condition's measured items — WITHOUT eliciting the
        behavior itself — plus a description of the measurement protocol and a per-item
        prediction ask. Returns ``{"protocol": str, "items": [{"item_id": str,
        "exhibit": {label: verbatim text}, "ask": str}, ...]}``. The ``ask`` is per item
        because it may reference item specifics (e.g. sycophancy names the item's correct
        letter); item ids must be stable and deterministic (they key the resume checkpoint
        and the per-item diagnostics kept in the prediction record).

        For the self-vs-generic phrasing ablation a view may
        additionally carry ``"protocol_3p"`` (the protocol in the third person) and each item an
        ``"ask_gen"`` — the ask with the subject as a ``{subject}`` placeholder and every later
        pronoun third-person ("… would it answer …") — used by the ``informed_oracle_genmin`` /
        ``informed_oracle_3p*`` arms. Items lacking them make those arms skip the eval.

        An eval whose behavior emerges from a multi-turn interaction rather than one scored prompt
        may exhibit that interaction's *opening* material and describe the rest in ``protocol``:
        PropensityBench shows a run's system prompt and first user turn verbatim, and explains the
        escalating-pressure ladder that follows in prose.

        Default ``None``: the eval does not expose its concrete measurement items this way —
        ``informed_oracle`` then skips the eval via ``applies_to``. (Every currently registered
        eval implements it; a single-fixed-prompt eval returns one item carrying the whole
        condition, as agentic_misalignment and tau2_policy do.)"""
        return None

    # --- Targets --------------------------------------------------------------
    def add_run_args(self, parser) -> None:
        """Register the args for the behavioral run (run_sweep.py). Subclasses call
        ``super().add_run_args(parser)`` then add their eval-specific args. The generic
        ``--max-*`` caps subsample the measurement to control cost; an eval honors whichever
        caps make sense for its hierarchy (PropensityBench prunes its harness inputs by
        workspace/role/task-scenario; evals that don't subsample simply ignore them)."""
        parser.add_argument("--max-workspaces", type=int, default=None,
                            help="cap the number of workspaces measured per domain (default: all)")
        parser.add_argument("--max-roles", type=int, default=None,
                            help="cap the number of roles per workspace measured (default: all)")
        parser.add_argument("--max-scenarios", type=int, default=None,
                            help="cap the number of task-scenarios per role measured (default: all)")
        parser.add_argument("--max-total-scenarios", type=int, default=50,
                            help="cap the TOTAL number of task-scenarios measured across the run, "
                                 "spread round-robin across domains so the split stays valid "
                                 "(default: 50 — a cost guardrail; pass 0 to disable)")

    def run_behavior(self, args) -> int:
        """Measure actual behavior, leaving artifacts for ``produce_targets`` to read."""
        raise NotImplementedError

    def add_extract_args(self, parser) -> None:
        """Register the args for target extraction (extract_targets.py). Subclasses call
        ``super().add_extract_args(parser)`` then add their eval-specific args."""
        parser.add_argument("--metric", choices=self.metrics, default=self.default_metric,
                            help=f"target metric to extract (default: {self.default_metric})")
        parser.add_argument("--out", default=None,
                            help="output path (default: results/<eval>/<model>/targets.json)")

    def produce_targets(self, args) -> dict[str, Any]:
        """Build the targets doc:
        ``{"metric", "model", "targets": {key: {rate,count,n,scenario,condition,logs}}}``."""
        raise NotImplementedError

    # --- Example prompts ------------------------------------------------------
    def extra_prompt_exhibits(self, split: str = "dev") -> list[dict[str, Any]]:
        """Optional eval-specific exhibits for the report's Example-prompts section, for methods
        that aren't standard per-condition builders (e.g. DiscrimEval's direct-contrast bias
        prompt). Each exhibit is ``{"title": str, "items": [(label, key, {slot: text}), ...]}``.
        Restrict to the report's ``split`` so test prompts stay hidden. Default: none."""
        return []

    # --- Report blurb ---------------------------------------------------------
    def report_extra_html(self, model_name: str, results_dir: str = "results",
                          split: str = "dev") -> str:
        """Optional eval-specific HTML appended to each model's report section (e.g. a bias
        table built from extra artifacts). ``split`` is the report's dev/test split, so the eval
        can list only that split's conditions. Default: nothing."""
        return ""

    def report_tldr_html(self, metric: str, elicitation_methods: list[str]) -> str:
        """The TL;DR ``<div class="tldr">`` for the HTML report. Override for eval-specific
        prose; the default is a generic one-liner."""
        methods = ", ".join(elicitation_methods)
        return (
            '<div class="tldr"><b>TL;DR.</b> Actual behavior on <b>'
            f'{self.name}</b> (metric <b>{metric}</b>) vs. each model\'s own prediction of '
            f'that rate, elicited via {len(elicitation_methods)} methods ({methods}). '
            'Tables below: cross-model Overview, then per-model per-condition accuracy and '
            'by-axis breakdowns.</div>'
        )
