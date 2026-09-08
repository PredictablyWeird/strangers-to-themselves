"""Method-adapter registry. Mirrors ``evals/__init__.py``: the benchmark selects methods by
name and runs only the (model × method × eval) combos that pass capability negotiation."""

from __future__ import annotations

from behavior_prediction.methods.base import Capabilities, MethodAdapter, RunConfig
from behavior_prediction.methods.elicited import (BehavioralSampling, GenericOracle,
                                                  GenericReport, InformedOracle, ListExperiment,
                                                  OraclePairwise, Pairwise, SelfReport, Value)
from behavior_prediction.methods.pilot import (FewShotOracle, InformedOracle3p,
                                               InformedOracle3pGen, InformedOracleGenmin,
                                               InformedSampling, ProtocolReport,
                                               ProtocolSampling, SelfReport3p, SelfReport3pGen,
                                               SelfReportGenmin)
from behavior_prediction.methods.trained import (ActivationProbe, CotFlip, CrossModelMean,
                                                  FewShot, FewShotOther, LlmPrediction,
                                                  OracleReportMean, OracleXmm, OracleXmmLearned,
                                                  ReportMean, TrainScenarioMean)

REGISTRY: dict[str, MethodAdapter] = {
    "self_report": SelfReport(),
    "generic_report": GenericReport(),     # self_report about "capable AI agents in general"
    "value": Value(),
    "list_experiment": ListExperiment(),
    "pairwise": Pairwise(),
    "behavioral_sampling": BehavioralSampling(),
    "informed_oracle": InformedOracle(),
    "generic_oracle": GenericOracle(),      # the oracle's exhibits, asked about a generic agent
    "oracle_pairwise": OraclePairwise(),   # the oracle's exhibits, asked as a forced choice
    "train_scenario_mean": TrainScenarioMean(),
    "llm_prediction": LlmPrediction(),
    "few_shot": FewShot(),                 # llm_prediction's cases as turns, answered by the model
    "few_shot_other": FewShotOther(),      # ...the same turns, answered by a named other model
    "cross_model_mean": CrossModelMean(),
    "report_mean": ReportMean(),           # mean of the other pool models' self_reports
    "oracle_report_mean": OracleReportMean(),   # ...and of their informed_oracle predictions
    "cot_flip": CotFlip(),
    "few_shot_oracle": FewShotOracle(),   # PILOT: few_shot history + informed_oracle item asks
    "informed_sampling": InformedSampling(),   # PILOT: behavioral_sampling, informed generator
    "protocol_report": ProtocolReport(),  # PILOT: self_report + measurement-protocol prose
    "protocol_sampling": ProtocolSampling(),  # PILOT: behavioral_sampling, protocol-only generator
    # ABLATION: the self/generic x 2p/3p phrasing arms.
    "self_report_genmin": SelfReportGenmin(),
    "self_report_3p": SelfReport3p(),
    "self_report_3p_gen": SelfReport3pGen(),
    "informed_oracle_genmin": InformedOracleGenmin(),
    "informed_oracle_3p": InformedOracle3p(),
    "informed_oracle_3p_gen": InformedOracle3pGen(),
    # Ensembles of the two above — must stay LAST so their component prediction files already
    # exist when they run (they read them off disk rather than recomputing). oracle_xmm is the
    # equal-weight baseline; oracle_xmm_learned learns the per-fold weighting.
    "oracle_xmm": OracleXmm(),
    "oracle_xmm_learned": OracleXmmLearned(),
}
# ``ActivationProbe`` is intentionally NOT registered: it is a stubbed contract (the seam for the
# locally-hosted-model phase), so it would only ever skip (no model exposes activations yet).

#: Methods the benchmark runs by default. Excludes ``train_scenario_mean`` — a debugging-only
#: plumbing baseline that stays registered (runnable by name, used by tests) but is not run by
#: default and is never reported in the paper —
#: and ``list_experiment``: null on every selection-pool model (dev mean r +0.01, table
#: tab:method-model) while accounting for ~half of all prediction calls (100 runs/unit + a
#: shared control), so new models skip it; the existing pool results stand as the null.
#: ``few_shot_oracle`` / ``protocol_report`` are PILOTs (methods/pilot.py): runnable by name
#: only until promoted. ``informed_sampling`` was promoted to the canonical suite, with
#: reuse-allowed n_exemplars=10 defaults.
#: ``oracle_xmm`` / ``oracle_xmm_learned`` are excluded by default: the
#: oracle-xmm ensemble combinations are not part of the standard suite — runnable by name for
#: targeted analyses, but no default run should generate them.
#: ``cot_flip`` is excluded (2026-08-11): not run at all in the final suite — the
#: fold-learned sign mechanism is reported for ALL methods via scripts/learned_flip_sweep.py
#: (appendix; main body if it works out very well), so the standalone method would be redundant
#: spend. Existing prediction files stay on disk for that sweep's history.
DEFAULT_METHODS: list[str] = [m for m in REGISTRY
                              if m not in ("train_scenario_mean", "list_experiment",
                                           "few_shot_oracle", "protocol_report",
                                           "protocol_sampling",
                                           "self_report_genmin", "self_report_3p",
                                           "self_report_3p_gen", "informed_oracle_genmin",
                                           "informed_oracle_3p", "informed_oracle_3p_gen",
                                           "oracle_xmm", "oracle_xmm_learned", "cot_flip")]


def get_method(name: str) -> MethodAdapter:
    if name not in REGISTRY:
        raise SystemExit(f"unknown method {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def valid_combo(method: MethodAdapter, model, spec) -> bool:
    """A (method, model, eval) combo is runnable iff the model satisfies every capability the
    method requires and the method applies to the eval. Invalid combos are skipped, not errored."""
    c = method.capabilities
    if c.needs_activations and not getattr(model, "has_activations", False):
        return False
    if c.needs_logprobs and not getattr(model, "has_logprobs", False):
        return False
    return method.applies_to(spec)


__all__ = ["REGISTRY", "DEFAULT_METHODS", "get_method", "valid_combo",
           "MethodAdapter", "Capabilities", "RunConfig", "ActivationProbe"]
