"""behavior_prediction — measure model behavior and predict it with several elicitation methods."""

# Register the Tinker inspect_ai provider if its SDK is installed (venv-only extra).
try:
    from behavior_prediction import tinker_provider as _tinker_provider  # noqa: F401
except ImportError:
    pass
