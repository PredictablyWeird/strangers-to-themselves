"""The package imports cleanly and the registries are intact (catches import-rewrite breakage)."""

from __future__ import annotations

import importlib


def test_library_modules_import():
    for m in ["common", "elicitation", "metrics", "splits", "models_adapter",
              "results_io", "selection", "tuning", "evaluate", "report_html"]:
        importlib.import_module(f"behavior_prediction.{m}")
    for m in ["tuning", "evaluate"]:
        assert hasattr(importlib.import_module(f"behavior_prediction.{m}"), "main")


def test_cli_modules_import():
    for m in ["benchmark", "run_sweep", "extract_targets", "compare_predictions", "elicit"]:
        mod = importlib.import_module(f"behavior_prediction.cli.{m}")
        assert hasattr(mod, "main"), f"{m} is missing a main() entry point"


def test_registries():
    from behavior_prediction.evals import REGISTRY as EVALS
    from behavior_prediction.methods import REGISTRY as METHODS
    assert {"agentic_misalignment", "discrimeval", "propensitybench"} <= set(EVALS)
    assert {"self_report", "value", "train_scenario_mean"} <= set(METHODS)
    assert "third_person" not in METHODS  # never a standalone method (self_report is the only self-report)


def test_models_yaml_loads_as_package_data():
    from behavior_prediction import common
    reg = common.load_model_registry()
    assert "llama-3.3-70b" in reg and "deepseek-v4-flash-low" in reg


def test_methods_yaml_loads_as_package_data():
    from behavior_prediction import common
    cfg = common.load_methods_config()
    assert "self_report" in cfg["methods"]
    assert common.selection_pool()  # non-empty selection pool declared
