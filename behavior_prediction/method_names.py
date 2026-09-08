"""Single source of the paper-facing display names for prediction methods.

The code ids (``informed_oracle``, ``cot_flip``, ...) stay the on-disk and macro identifiers;
everything the reader sees — tables, figures, prose — uses these names. Every generator
(export.py, plots.py, scripts/significance_tests.py, scripts/spearman_robustness.py,
scripts/identity_transfer_figure.py) imports from here so a rename is one edit.
"""

#: Full display names (LaTeX- and matplotlib-safe: no underscores).
DISPLAY = {
    "self_report": "self-report",
    "generic_report": "generic report",
    "protocol_report": "protocol-informed self-report",
    "value": "value framing",
    "list_experiment": "list experiment",
    "pairwise": "paired comparison",
    "cot_flip": "reasoned self-assessment",
    "few_shot": "history-informed self-prediction",
    "few_shot_other": "history-informed, analyst answers",
    "llm_prediction": "analyst forecast",
    "informed_oracle": "item-informed self-prediction",
    "generic_oracle": "item-informed, generic subject",
    "oracle_report_mean": "item-informed, others' mean",
    "oracle_pairwise": "item-informed paired comparison",
    "behavioral_sampling": "proxy-scenario sampling",
    "informed_sampling": "item-informed proxy sampling",
    "protocol_sampling": "protocol-informed proxy sampling",
    "cross_model_mean": "cross-model behavior mean",
    "report_mean": "mean of others' self-reports",
    "oracle_xmm": "equal-weight ensemble",
    "oracle_xmm_learned": "learned ensemble",
    "train_scenario_mean": "train-scenario mean",
}

#: Compact variants for tight table columns / figure ticks.
SHORT = {
    "self_report": "self-report",
    "generic_report": "generic report",
    "protocol_report": "protocol-informed self",
    "value": "value framing",
    "list_experiment": "list experiment",
    "pairwise": "paired comp.",
    "cot_flip": "reasoned self-assess.",
    "few_shot": "history-informed self",
    "few_shot_other": "history, analyst",
    "llm_prediction": "analyst forecast",
    "informed_oracle": "item-informed self",
    "generic_oracle": "item-inf., generic",
    "oracle_report_mean": "item-inf., others",
    "oracle_pairwise": "item-inf. paired comp.",
    "behavioral_sampling": "proxy sampling",
    "informed_sampling": "item-inf. proxy",
    "protocol_sampling": "protocol-inf. proxy",
    "cross_model_mean": "cross-model mean",
    "report_mean": "others' self-reports",
    "oracle_xmm": "equal ensemble",
    "oracle_xmm_learned": "learned ensemble",
    "train_scenario_mean": "train-scenario mean",
}


def display(method: str) -> str:
    return DISPLAY.get(method, method.replace("_", " "))


def short(method: str) -> str:
    return SHORT.get(method, display(method))
