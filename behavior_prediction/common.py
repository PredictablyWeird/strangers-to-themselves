"""Shared, eval-agnostic helpers for the behavior-prediction experiment.

This module holds everything the prediction methods share *regardless of which eval* is
being predicted (the eval-specific parts — scenarios, condition axes, value framing — live in
``evals/``):

- model registry (``resolve_model`` from ``models.yaml``),
- ``parse_percentage`` — robustly turn a model reply into a fraction in [0, 1],
- JSON IO + ``make_prediction_record`` — the *uniform prediction-file schema* every method
  writes (so ``compare_predictions.py`` / ``tuning.py`` / ``evaluate.py`` pick up any new method
  file with no code change),
- ``elicit_rates`` — the async "ask a prompt N times and average a parsed percentage" engine,
- path/slug helpers for the ``results/<eval>/<model>/`` layout.

Prediction-file schema (one file per method, under ``results/<eval>/<model>/predictions/``)::

    {
      "method": "self_report",                 # unique id -> one column group (base + suffixes)
      "base_method": "self_report",            # builder key (id minus hyperparameter suffixes)
      "hyperparameters": {},                   # e.g. {"detail": "concrete", "honesty_nudge": true}
      "model":  "openrouter/meta-llama/...",   # model whose behavior is predicted
      "metric": "harmful",                     # which target metric this predicts
      "predictions": {
        "<condition_key>": {
          "predicted_rate": 0.05,              # REQUIRED: fraction in [0, 1]
          ...                                  # method-specific extras (n, samples, ...)
        },
        ...
      }
    }

Target-file schema (written by extract_targets.py)::

    {
      "metric": "harmful",
      "model":  "openrouter/meta-llama/...",
      "targets": {
        "<condition_key>": {"rate": 0.0, "count": 0, "n": 10,
                            "scenario": "...", "condition": {...}, "logs": [...]},
        ...
      }
    }
"""

from __future__ import annotations

import itertools
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from behavior_prediction.evals.base import EvalSpec

# --- Model registry (models.yaml shortcuts) ------------------------------------
# Maps a short name -> full provider model string + reasoning config. See models.yaml.

_MODELS_DOC: dict[str, Any] | None = None
# Reasoning keys an entry may set; they map directly to inspect_ai GenerateConfig fields.
_REASONING_KEYS = ("reasoning_effort", "reasoning_tokens", "reasoning_summary")
#: Provider-specific passthrough. Together IGNORES `reasoning_effort`; the only way to run its
#: models non-reasoning is `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`
#: (docs/together-finetunable-models.md). inspect_ai forwards GenerateConfig.extra_body verbatim.
_PASSTHROUGH_KEYS = ("extra_body",)
#: Everything a models.yaml entry may carry into GenerateConfig, and what makes two entries distinct.
_CONFIG_KEYS = _REASONING_KEYS + _PASSTHROUGH_KEYS
#: Pool name meaning "every model with predictions on disk" (never listed in models.yaml).
DISCOVER_POOL = "all"

#: THE standard grader, project-wide (2026-08-10): the SAME model grades every benchmark —
#: measurement judges (MASK, agentic_misalignment) and the sampling methods' exchange graders
#: alike. claude-sonnet-4 is retired as a grader everywhere: on 220 stored sonnet-4-graded
#: exchanges across four evals this model agreed 97.7% (kappa 0.94) with matched yes-rates at
#: ~30-60x lower cost (2026-08-09 validation). ONE exception — self-grading hygiene: when the
#: SUBJECT model is the grader model's family (gemini), pass a different cheap grader explicitly
#: (gpt-5.4-nano-low is the validated alternate) so no model judges its own outputs.
DEFAULT_GRADER = "openrouter/google/gemini-3.1-flash-lite"


def load_models_doc(path: str | Path | None = None) -> dict[str, Any]:
    """Load (and cache) the whole ``models.yaml`` document (``models`` + ``pools``).

    By default reads the copy shipped as package data (``behavior_prediction/models.yaml``) via
    ``importlib.resources`` so it works regardless of CWD. Pass ``path`` to override (returns ``{}``
    if that file is absent)."""
    global _MODELS_DOC
    if _MODELS_DOC is None:
        import yaml

        if path is not None:
            p = Path(path)
            text = p.read_text() if p.exists() else None
        else:
            from importlib.resources import files
            text = (files("behavior_prediction") / "models.yaml").read_text()
        _MODELS_DOC = (yaml.safe_load(text) if text else None) or {}
    return _MODELS_DOC


def load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    """The ``models.yaml`` shortcut table (short name -> provider string + reasoning config)."""
    return load_models_doc(path).get("models") or {}


def load_model_pools(path: str | Path | None = None) -> dict[str, list[str]]:
    """The named model pools declared in ``models.yaml`` (``pools:``).

    A pool is just a list of model shortcuts that ``bp-evaluate`` scores together into one overview
    table. ``DISCOVER_POOL`` ("all") is implicit — it means "every model found on disk" — and is
    resolved by the caller, not stored here."""
    pools = load_models_doc(path).get("pools") or {}
    return {name: list(models or []) for name, models in pools.items()}


def model_pool(name: str, path: str | Path | None = None) -> list[str] | None:
    """Resolve a pool name to its model shortcuts. ``None`` for ``DISCOVER_POOL`` (= discover from
    disk); raises ``KeyError`` for an unknown name, listing the valid ones."""
    if name == DISCOVER_POOL:
        return None
    pools = load_model_pools(path)
    if name not in pools:
        known = ", ".join(sorted([*pools, DISCOVER_POOL]))
        raise KeyError(f"unknown model pool {name!r} (models.yaml declares: {known})")
    return pools[name]


def results_root() -> Path:
    """Base directory for all generated results. Defaults to ``./results`` (relative to the CWD,
    i.e. the repo root); override with the ``BP_RESULTS_DIR`` environment variable."""
    return Path(os.environ.get("BP_RESULTS_DIR", "results"))


# --- Methods config (methods.yaml: per-method arg grids + the model selection pool) ------------

_METHODS_CONFIG: dict[str, Any] | None = None


def load_methods_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load (and cache) ``methods.yaml`` — the per-method argument grids and the selection pool.

    By default reads the copy shipped as package data via ``importlib.resources`` (so it works
    regardless of CWD); pass ``path`` to override (returns ``{}`` if that file is absent)."""
    global _METHODS_CONFIG
    if _METHODS_CONFIG is None:
        import yaml

        if path is not None:
            p = Path(path)
            text = p.read_text() if p.exists() else None
        else:
            from importlib.resources import files
            text = (files("behavior_prediction") / "methods.yaml").read_text()
        _METHODS_CONFIG = yaml.safe_load(text) if text else {}
    return _METHODS_CONFIG or {}


def selection_pool() -> list[str]:
    """The model shortcuts ``bp-tune`` aggregates over to pick one shared best setting per
    (method, eval). Empty list if ``methods.yaml`` declares none."""
    return list(load_methods_config().get("selection_pool") or [])


def donor_pool() -> list[str]:
    """The fixed model set the pool-derived predictors (cross_model_mean, report_mean,
    oracle_report_mean) draw donors from. Empty list if ``methods.yaml`` declares none —
    callers then fall back to their legacy behavior (targets glob / selection_pool)."""
    return list(load_methods_config().get("donor_pool") or [])


def _method_args(base_method: str, config: dict[str, Any] | None = None) -> dict[str, list[Any]]:
    """The ordered ``arg -> [values]`` map for a base method (empty if it has no swept args /
    is absent from ``methods.yaml``)."""
    cfg = config if config is not None else load_methods_config()
    mcfg = (cfg.get("methods") or {}).get(base_method) or {}
    return mcfg.get("args") or {}


#: How a method's ``args`` are expanded into settings (``methods.<name>.sweep``).
_SWEEP_STRATEGIES = ("cross_product", "one_at_a_time")


def _method_sweep(base_method: str, config: dict[str, Any] | None = None) -> str:
    """The sweep strategy for a base method (``methods.<name>.sweep``; default
    ``cross_product``). Raises on an unknown value so a typo fails loudly."""
    cfg = config if config is not None else load_methods_config()
    mcfg = (cfg.get("methods") or {}).get(base_method) or {}
    strategy = mcfg.get("sweep", "cross_product")
    if strategy not in _SWEEP_STRATEGIES:
        raise ValueError(f"methods.{base_method}.sweep={strategy!r} is not one of "
                         f"{list(_SWEEP_STRATEGIES)}")
    return strategy


def method_arg_grid(base_method: str, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Hyperparameter dicts to search over for a method, from ``methods.yaml``.

    Each arg's first listed value is its default; every returned dict contains only the args whose
    value differs from their default, so the default setting is ``{}`` (its id == ``base_method``)
    and every existing on-disk id is reproduced. Methods with no swept args (or absent from the
    YAML) return ``[{}]``.

    The expansion follows ``methods.<name>.sweep``:
      - ``cross_product`` (default) — the cartesian product of every arg's values.
      - ``one_at_a_time`` — the all-default setting plus, for each arg, each of its non-default
        values with every other arg held at its default (only one arg moved off-default at a time).
        This is the OFAT grid: ``1 + sum(len(values) - 1)`` settings instead of the full product.
    """
    args = _method_args(base_method, config)
    if not args:
        return [{}]
    names = list(args)
    defaults = {n: args[n][0] for n in names}
    if _method_sweep(base_method, config) == "one_at_a_time":
        grid: list[dict[str, Any]] = [{}]
        for n in names:
            grid.extend({n: v} for v in args[n][1:])   # skip the default (first) value
        return grid
    grid = []
    for combo in itertools.product(*[args[n] for n in names]):
        grid.append({n: v for n, v in zip(names, combo) if v != defaults[n]})
    return grid


def resolve_model(name: str) -> tuple[str, dict[str, Any]]:
    """Resolve a --model value to (full_model_name, generate_config).

    If ``name`` is a key in models.yaml, return its ``model`` plus any reasoning keys and any
    provider passthrough (``extra_body``). Otherwise pass the name through verbatim with no config
    (raw provider strings keep working)."""
    spec = load_model_registry().get(name)
    if not spec:
        return name, {}
    return spec["model"], {k: spec[k] for k in _CONFIG_KEYS if k in spec}


def model_slug(name: str) -> str:
    """Directory slug for a model. A ``models.yaml`` shortcut keeps its own name (so reasoning
    variants like ``deepseek-v4-flash`` vs ``deepseek-v4-flash-low`` get distinct directories). A raw full
    provider string adopts a shortcut's name only when exactly one shortcut maps to it *and* that
    shortcut carries no per-call config — otherwise distinct settings (a reasoning effort, or an
    ``extra_body`` that disables thinking) could silently share a directory, so it falls back to a
    filesystem-safe form of the full string."""
    reg = load_model_registry()
    if name in reg:
        return name
    matches = [s for s, spec in reg.items() if spec.get("model") == name]
    if len(matches) == 1 and not any(k in reg[matches[0]] for k in _CONFIG_KEYS):
        return matches[0]
    return name.replace("/", "_")


def model_display_name(full: str) -> str:
    """A model's name **as written into a prompt** — the full provider string minus its routing
    prefix. ``few_shot_other`` names the model whose behavior it predicts, and the provider segment
    is plumbing (where the request is routed), not identity:

        openrouter/meta-llama/llama-3.3-70b-instruct  -> meta-llama/llama-3.3-70b-instruct
        together/meta-llama/Llama-3.3-70B-Instruct-Turbo -> meta-llama/Llama-3.3-70B-Instruct-Turbo

    An ``inspect_ai`` model string is ``<provider>/<model path>``, so this drops the first segment
    and keeps the rest — which stays informative (vendor + model + size/variant) for every hosted
    model. Caveat: a finetune addressed by opaque id (``tinker/tinker://<uuid>/…``) has no
    informative name to recover, and reduces to that id."""
    return full.split("/", 1)[1] if "/" in full else full


def model_identity(name: str) -> dict[str, Any]:
    """Canonical identity of a ``--model`` value: the resolved full provider string, the
    ``models.yaml`` shortcut (if ``name`` is one), and the per-call config applied (reasoning keys
    and any ``extra_body``). Persisted in every result file so runs made with different settings —
    including thinking on vs off — are never silently mixed. The dict is stored under the historical
    key ``reasoning``."""
    full, config = resolve_model(name)
    return {"model": full, "shortcut": name if name in load_model_registry() else None,
            "reasoning": config}


def identity_of(doc: dict[str, Any]) -> dict[str, Any]:
    """Read the model-identity fields from a stored result doc (targets or prediction), tolerating
    older files that omit ``model_shortcut``/``reasoning``."""
    return {"model": doc.get("model"), "shortcut": doc.get("model_shortcut"),
            "reasoning": doc.get("reasoning")}


#: A Together *dedicated endpoint* name is the finetuned model id plus a fresh 8-hex suffix, minted
#: anew on every deploy. A finetuned model id already ends in one 8-hex group, so an endpoint ends in
#: two. Collapsing the last one recovers the stable model id; plain models (``…-Instruct-Turbo``) and
#: bare finetune ids are left alone.
_ENDPOINT_SUFFIX = re.compile(r"^(.*-[0-9a-f]{8})-[0-9a-f]{8}$")


def canonical_model(model: str | None) -> str | None:
    """Model string with a rotating dedicated-endpoint suffix stripped.

    Redeploying a finetuned model gives it a new endpoint name, so targets measured through one
    deploy and predictions made through another would otherwise look like different models."""
    if not model:
        return model
    m = _ENDPOINT_SUFFIX.match(model)
    return m.group(1) if m else model


def identity_mismatch(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """The field where two stored model identities disagree, or ``None`` if compatible. Compares
    the full model string always (modulo a rotating endpoint suffix — see ``canonical_model``) and
    the reasoning config when *both* sides recorded one (older files may omit it); the shortcut is
    cosmetic and ignored."""
    if a.get("model") and b.get("model") \
            and canonical_model(a["model"]) != canonical_model(b["model"]):
        return "model"
    ra, rb = a.get("reasoning"), b.get("reasoning")
    if ra is not None and rb is not None and ra != rb:
        return "reasoning"
    return None


def default_pred_out(spec: "EvalSpec", model: str, method_id: str) -> str:
    """Default prediction-file path: results/<eval>/<model-slug>/predictions/<method_id>.json."""
    return f"{results_root()}/{spec.name}/{model_slug(model)}/predictions/{method_id}.json"


def default_targets_out(spec: "EvalSpec", model: str) -> str:
    """Default targets-file path: results/<eval>/<model-slug>/targets.json."""
    return f"{results_root()}/{spec.name}/{model_slug(model)}/targets.json"


# --- Percentage parsing --------------------------------------------------------

# First number in the reply (handles "5", "5%", "5.0", "I would say 5%").
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


def parse_percentage(text: str | None) -> float | None:
    """Extract the first 0-100 percentage from ``text`` and return it as a fraction.

    Returns ``None`` if no number is found. Numbers are clamped to [0, 100].
    """
    if not text:
        return None
    match = _PCT_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    value = max(0.0, min(100.0, value))
    return value / 100.0


# Matches the tagged final answer an analyst is asked to end on, e.g. "PREDICTION: 80".
_PREDICTION_TAG_RE = re.compile(r"PREDICTION:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_prediction_tag(text: str | None) -> float | None:
    """Parse a forecasting reply that may reason out loud before its answer.

    Unlike :func:`parse_percentage` (which takes the *first* number and so is fooled by a leading
    list marker like ``1. ...`` in verbose reasoning), this reads the **last** ``PREDICTION: <n>``
    tag the analyst was told to end on; failing that, it falls back to the **last** number in the
    text. Returns ``None`` if no number is found; values are clamped to [0, 100] and returned as a
    fraction.
    """
    if not text:
        return None
    tags = _PREDICTION_TAG_RE.findall(text)
    if tags:
        value = float(tags[-1])
    else:
        nums = _PCT_RE.findall(text)
        if not nums:
            return None
        value = float(nums[-1])
    value = max(0.0, min(100.0, value))
    return value / 100.0


# Signed variants for bias-gap forecasts: an optional +/−/- before the number.
_SIGNED_PREDICTION_TAG_RE = re.compile(r"PREDICTION:\s*([+\-−]?\s*\d+(?:\.\d+)?)", re.IGNORECASE)
_SIGNED_NUM_RE = re.compile(r"[+\-−]?\d+(?:\.\d+)?")


def parse_signed_prediction_tag(text: str | None) -> float | None:
    """Signed :func:`parse_prediction_tag`, for gap forecasts on a bias_contrast eval (e.g.
    ``llm_prediction`` on DiscrimEval): read the **last** ``PREDICTION: <n>`` tag, honoring an
    explicit +/- sign (falling back to the last signed number in the text), clamp to [-100, 100]
    and return a fraction in [-1, 1]. ``None`` if no number is found."""
    if not text:
        return None
    tags = _SIGNED_PREDICTION_TAG_RE.findall(text)
    if tags:
        token = tags[-1]
    else:
        nums = _SIGNED_NUM_RE.findall(text)
        if not nums:
            return None
        token = nums[-1]
    value = float(token.replace("−", "-").replace(" ", ""))
    value = max(-100.0, min(100.0, value))
    return value / 100.0


# Matches the tagged final count a list-experiment respondent is asked to end on, e.g. "COUNT: 3".
_COUNT_TAG_RE = re.compile(r"COUNT:\s*(\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"\d+")


def count_parser(max_count: int) -> "Callable[[str | None], float | None]":
    """A parse_fn for the item-count (list-experiment) technique: read the integer number of
    items the respondent said it would take, clamped to ``[0, max_count]`` and returned as a raw
    count (a float, *not* a fraction — the caller differences control vs treatment counts).

    Reads the ``COUNT: <n>`` tag the prompt asks for (last occurrence, so a reasoning model may
    think out loud first); falls back to the last bare integer. ``None`` when no integer is found
    so ``elicit_rates`` retries."""
    def _parse(text: str | None) -> float | None:
        if not text:
            return None
        tags = _COUNT_TAG_RE.findall(text)
        nums = tags or _INT_RE.findall(text)
        if not nums:
            return None
        return float(max(0, min(max_count, int(nums[-1]))))
    return _parse


# A standalone yes/no token (used by parse_yes_no).
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(text: str | None) -> float | None:
    """Turn a yes/no reply into ``1.0`` (affirmative) / ``0.0`` (negative).

    Returns ``None`` when the answer is ambiguous (no yes/no, or both appear) so the caller
    retries. Prefers a yes/no at the very start of the (stripped) reply — the expected shape
    after the "Answer:" cue — then falls back to a single unambiguous occurrence anywhere.
    """
    if not text:
        return None
    s = text.strip().lower()
    lead = re.match(r"[^a-z]*(yes|no)\b", s)
    if lead:
        return 1.0 if lead.group(1) == "yes" else 0.0
    found = set(_YES_NO_RE.findall(s))
    if found == {"yes"}:
        return 1.0
    if found == {"no"}:
        return 0.0
    return None


# Pairwise-comparison verdict tokens (parse_pair_choice). The option letters are matched
# case-sensitively so prose articles ("a situation") can't be mistaken for a verdict; EQUAL is
# case-insensitive.
_PAIR_EQUAL_RE = re.compile(r"\bEQUAL\b", re.IGNORECASE)


@lru_cache(maxsize=None)
def _pair_option_re(options: tuple[str, str]) -> "re.Pattern[str]":
    return re.compile(rf"\b({options[0]}|{options[1]})\b")


def parse_pair_choice(text: str | None, options: tuple[str, str] = ("A", "B")) -> float | None:
    """Parse a pairwise-comparison reply into the FIRST-listed option's win value: ``1.0`` for the
    first ``options`` token, ``0.0`` for the second, ``0.5`` for EQUAL (the optional indifference
    verdict; the caller inverts for the swapped order, where 0.5 is the fixed point).

    ``options`` are the two verdict tokens. A/B by default; ``oracle_pairwise`` passes X/Y because
    its prompt shows verbatim measured items that may themselves contain "A"/"B" labels (e.g.
    DiscrimEval's two applicant VERSIONs), which would make an A/B verdict ambiguous.

    The prompt asks for exactly one word and nothing else, but a (reasoning) model may still talk
    first ("Situation A is tense… ANSWER: B"), so the LAST verdict token in the reply wins.
    ``None`` when no verdict token is found (``elicit_rates`` retries)."""
    if not text:
        return None
    last_ab = None
    for last_ab in _pair_option_re(tuple(options)).finditer(text):
        pass
    last_eq = None
    for last_eq in _PAIR_EQUAL_RE.finditer(text):
        pass
    if last_eq is not None and (last_ab is None or last_eq.start() > last_ab.start()):
        return 0.5
    if last_ab is not None:
        return 1.0 if last_ab.group(1) == options[0] else 0.0
    return None


# --- JSON IO -------------------------------------------------------------------


def save_json(obj: Any, path: str | Path) -> None:
    # Write to a temp file in the same dir, then atomically rename, so a process killed mid-write
    # can't leave a truncated/corrupt JSON file (expensive runs are interruption-prone).
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _legacy_method_id(base_method: str, hyperparameters: dict[str, Any]) -> str:
    """Fallback id logic for methods absent from ``methods.yaml`` (kept so nothing breaks if the
    config is missing a method): fixed-order ``-concrete``/``-honest`` suffixes."""
    suffixes: list[str] = []
    if hyperparameters.get("detail") == "concrete":
        suffixes.append("concrete")
    if hyperparameters.get("honesty_nudge"):
        suffixes.append("honest")
    return "-".join([base_method, *suffixes])


def method_id(base_method: str, hyperparameters: dict[str, Any],
              config: dict[str, Any] | None = None) -> str:
    """Deterministic, collision-free method id: base + descriptive hyperparameter suffixes.

    Driven by ``methods.yaml``: each non-default arg value appends its ``id_suffix`` token (or the
    value itself if untokened), in the YAML arg order, so ids are stable regardless of dict order.
    Default values (the first listed for an arg) contribute no suffix, so the id equals
    ``base_method`` — keeping existing ids/filenames unchanged. Methods absent from the YAML fall
    back to the legacy ``-concrete``/``-honest`` logic."""
    cfg = config if config is not None else load_methods_config()
    mcfg = (cfg.get("methods") or {}).get(base_method)
    if mcfg is None:
        return _legacy_method_id(base_method, hyperparameters)
    args = mcfg.get("args") or {}
    id_suffix = mcfg.get("id_suffix") or {}
    suffixes: list[str] = []
    for arg, values in args.items():
        if arg not in hyperparameters:
            continue
        val = hyperparameters[arg]
        if values and val == values[0]:        # default value -> no suffix
            continue
        token = (id_suffix.get(arg) or {}).get(val, str(val))
        suffixes.append(token)
    return "-".join([base_method, *suffixes])


def make_prediction_record(
    method: str,
    model: str,
    metric: str,
    predictions: dict[str, dict[str, Any]],
    *,
    base_method: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a prediction-file record conforming to the shared schema.

    ``predictions`` maps scenario -> dict that MUST contain ``predicted_rate``
    (a fraction in [0, 1]); any additional keys (n, samples, raw, ...) are kept.

    ``base_method`` is the prompt-builder key (defaults to ``method``); ``hyperparameters``
    records the knobs the prompt was built with so the report can reconstruct it;
    ``reasoning`` records the reasoning config the model was run with (full traces, if any,
    live in the sibling reasoning/ file)."""
    for scenario, pred in predictions.items():
        if "predicted_rate" not in pred:
            raise ValueError(
                f"prediction for {scenario!r} is missing required 'predicted_rate'"
            )
    return {
        "method": method,
        "base_method": base_method or method,
        "hyperparameters": hyperparameters or {},
        "reasoning": reasoning or {},
        "model": model,
        "metric": metric,
        "predictions": predictions,
    }


def _extract_reasoning(output: Any) -> dict[str, Any] | None:
    """Pull reasoning out of a model output: full trace (``ContentReasoning.reasoning``) when
    available, otherwise the summary. Returns None when the model produced no reasoning."""
    try:
        content = output.choices[0].message.content
    except (AttributeError, IndexError):
        return None
    if isinstance(content, str):
        return None
    parts = [c for c in content if getattr(c, "type", None) == "reasoning"]
    if not parts:
        return None
    reasoning = "\n".join(p.reasoning for p in parts if getattr(p, "reasoning", None)) or None
    summary = "\n".join(p.summary for p in parts if getattr(p, "summary", None)) or None
    redacted = any(getattr(p, "redacted", False) for p in parts)
    if not (reasoning or summary or redacted):
        return None
    return {"reasoning": reasoning, "summary": summary, "redacted": redacted}


def split_reasoning(
    predictions: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Pop the per-call ``reasoning`` list out of each prediction entry (keeping the lean
    prediction data) and build a ``{key: [{answer, reasoning, summary, redacted}, ...]}``
    bundle, including only calls that actually produced reasoning."""
    bundle: dict[str, list[dict[str, Any]]] = {}
    for key, entry in predictions.items():
        reasoning = entry.pop("reasoning", None)
        if not reasoning:
            continue
        raw = entry.get("raw") or []
        items = [{"answer": (raw[i] if i < len(raw) else None), **r}
                 for i, r in enumerate(reasoning) if r]
        if items:
            bundle[key] = items
    return predictions, bundle


def save_reasoning(method: str, model: str, reasoning_config: dict[str, Any] | None,
                   traces: dict[str, list[dict[str, Any]]], path: str | Path) -> None:
    """Write the reasoning traces/summaries to a sibling reasoning file."""
    save_json({"method": method, "model": model, "reasoning": reasoning_config or {},
               "traces": traces}, path)


# --- Shared elicitation engine -------------------------------------------------
# Used by every prediction method that works by asking the model a prompt N times and
# averaging a parsed percentage (self_report, value, future variants).

#: One turn of a multi-turn prompt: ``{"role": "user"|"assistant"|"system", "content": text}``.
#: Turns are plain dicts rather than ``inspect_ai`` ChatMessage objects so the prompt *builders*
#: (elicitation.py) stay free of an inspect_ai import — this module owns that dependency and
#: imports it lazily inside ``elicit_rates``.
Turn = dict[str, str]


def _as_model_input(prompt: str | list[Turn]) -> Any:
    """Turn one ``prompts`` value into an ``inspect_ai`` model input: a plain string passes through
    unchanged; a conversation becomes the matching list of ChatMessages. Raises on an unknown role
    so a typo fails loudly rather than silently dropping a turn."""
    if isinstance(prompt, str):
        return prompt
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

    roles = {"user": ChatMessageUser, "assistant": ChatMessageAssistant,
             "system": ChatMessageSystem}
    out = []
    for turn in prompt:
        role = turn["role"]
        if role not in roles:
            raise ValueError(f"unknown conversation role {role!r}; expected one of {sorted(roles)}")
        out.append(roles[role](content=turn["content"]))
    return out


def _load_checkpoint(path: str, keys: set[str]) -> dict[str, list[tuple[float | None, str, Any]]]:
    """Load a JSONL checkpoint sidecar into ``key -> [(frac, text, reasoning), ...]``.

    Only keys in ``keys`` are kept (stale keys from a different prompt set are ignored). A torn
    final line (process killed mid-append) is tolerated by skipping unparseable lines."""
    done: dict[str, list[tuple[float | None, str, Any]]] = {k: [] for k in keys}
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a hard kill — skip it
            k = rec.get("key")
            if k in done:
                done[k].append((rec.get("frac"), rec.get("text", ""), rec.get("reasoning")))
    return done


def elicit_rates(
    model_name: str,
    prompts: dict[str, str | list[Turn]],
    *,
    runs: int = 10,
    temperature: float = 1.0,
    max_parse_retries: int = 2,
    max_transport_retries: int = 3,
    concurrency: int = 16,
    reasoning_config: dict[str, Any] | None = None,
    parse_fn: Callable[[str | None], float | None] = parse_percentage,
    verbose: bool = True,
    checkpoint_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Ask ``model_name`` each prompt ``runs`` times; parse a rate; average.

    ``prompts`` maps a key (e.g. scenario) to either the prompt text **or a conversation** — a list
    of ``{"role", "content"}`` turns, which ``few_shot`` uses to show its labelled training cases as
    alternating user/assistant exchanges. Only the *input* differs: everything below (sampling,
    parsing, retries, checkpointing) reads the model's reply text and is identical either way.
    Returns a dict key -> {predicted_rate, n, samples, raw, reasoning, parse_failures} suitable to
    pass to ``make_prediction_record`` (``reasoning`` is a per-call list, aligned with ``raw``, of
    {reasoning, summary, redacted} dicts or None). Loads ``.env`` so provider API keys are
    available (the ``inspect`` CLI does this automatically, but library use does not).

    ``parse_fn`` turns one reply into a fraction in [0, 1] (``None`` = unparseable, retried up to
    ``max_parse_retries`` times); the default reads a percentage, but a binary eval can pass
    ``parse_yes_no`` to sample a yes/no rate through the same engine (``predicted_rate`` is then
    the yes-fraction). Transport/provider errors are a *separate* concern: each ``generate`` call
    is retried up to ``max_transport_retries`` times with exponential backoff before the sample is
    recorded as a ``"<error: Type>"`` failure (so a flaky connection no longer burns the parse
    budget, and rate limits get a backoff rather than an instant re-hammer).

    ``reasoning_config`` (e.g. ``{"reasoning_effort": "low"}``) is merged into the
    GenerateConfig, so reasoning mode is controlled per model via models.yaml.

    All key x run calls are pooled under a single ``concurrency`` semaphore (no barrier
    between keys), so a slow model's calls overlap across conditions — important for
    reasoning models with high per-call latency. ``concurrency`` is also wired into the inspect
    ``GenerateConfig.max_connections`` so inspect's own connection pool doesn't silently cap us at
    its default of 10.

    ``checkpoint_path`` enables interruption-robust resume: each completed sample is appended to a
    JSONL sidecar as it finishes, and on start any prior samples are loaded so only the remaining
    ``runs`` per key are re-issued. The caller owns the sidecar's lifecycle (deletes it after the
    final record is written, or up-front for a forced fresh run).
    """
    import asyncio
    import sys

    from dotenv import load_dotenv
    from inspect_ai.model import GenerateConfig, get_model

    load_dotenv()
    model = get_model(model_name)
    config = GenerateConfig(temperature=temperature, max_connections=concurrency,
                            **(reasoning_config or {}))

    keys = list(prompts)
    # Build each key's model input once, not per (key x run) call: a few_shot conversation is k
    # turns long, so rebuilding it for every sample would be pure waste.
    inputs = {k: _as_model_input(p) for k, p in prompts.items()}
    # Resume: load any samples already persisted for these keys, capped at ``runs`` per key.
    done = _load_checkpoint(checkpoint_path, set(keys)) if checkpoint_path else {k: [] for k in keys}
    done = {k: v[:runs] for k, v in done.items()}
    n_loaded = sum(len(v) for v in done.values())
    if n_loaded and verbose:
        print(f"[resume] {checkpoint_path}: {n_loaded} prior samples loaded; "
              f"scheduling {sum(max(0, runs - len(done[k])) for k in keys)} more", file=sys.stderr)

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt_file = open(checkpoint_path, "a") if checkpoint_path else None
    write_lock = asyncio.Lock()

    async def _persist(key: str, res: tuple[float | None, str, Any]) -> None:
        if ckpt_file is None:
            return
        frac, text, reasoning = res
        line = json.dumps({"key": key, "frac": frac, "text": text, "reasoning": reasoning})
        async with write_lock:  # one complete sample per line, serialized across tasks
            ckpt_file.write(line + "\n")
            ckpt_file.flush()

    async def one_query(
        key: str, model_input: Any, sem: "asyncio.Semaphore"
    ) -> tuple[float | None, str, dict[str, Any] | None]:
        last_text = ""
        last_reasoning: dict[str, Any] | None = None
        async with sem:
            for _ in range(max_parse_retries + 1):
                output = None
                for attempt in range(max_transport_retries + 1):
                    try:
                        output = await model.generate(model_input, config=config)
                        break
                    except Exception as exc:  # transport/provider error: back off and retry
                        last_text = f"<error: {type(exc).__name__}>"
                        if attempt < max_transport_retries:
                            await asyncio.sleep(min(2 ** attempt, 30))
                        else:
                            print(f"[error] {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if output is None:  # transport retries exhausted — not a parse problem, stop here
                    break
                last_text = (output.completion or "").strip()
                last_reasoning = _extract_reasoning(output)
                frac = parse_fn(last_text)
                if frac is not None:
                    res = (frac, last_text, last_reasoning)
                    await _persist(key, res)
                    return res
        res = (None, last_text, last_reasoning)
        await _persist(key, res)
        return res

    async def run_all() -> dict[str, dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        # Pool every (remaining) key x run call into one gather (no per-key barrier).
        task_keys = [k for k in keys for _ in range(max(0, runs - len(done[k])))]
        flat = await asyncio.gather(
            *(one_query(k, inputs[k], sem) for k in task_keys)
        )
        grouped: dict[str, list[tuple[float | None, str, dict[str, Any] | None]]] = {
            k: list(done[k]) for k in keys  # seed with resumed samples, merge by key (order-free)
        }
        for k, res in zip(task_keys, flat):
            grouped[k].append(res)

        results: dict[str, dict[str, Any]] = {}
        for key in keys:
            queries = grouped[key]
            fracs = [f for f, _, _ in queries if f is not None]
            raw = [t for _, t, _ in queries]
            reasoning = [r for _, _, r in queries]
            failures = sum(1 for f, _, _ in queries if f is None)
            predicted = mean(fracs) if fracs else None
            results[key] = {
                "predicted_rate": predicted,
                "n": len(fracs),
                "samples": fracs,
                "raw": raw,
                "reasoning": reasoning,
                "parse_failures": failures,
            }
            if verbose:
                pct = "n/a" if predicted is None else f"{predicted * 100:.1f}%"
                print(f"{key:<28} predicted={pct:>7}  (n={len(fracs)}, failures={failures})")
        return results

    try:
        return asyncio.run(run_all())
    finally:
        if ckpt_file is not None:
            ckpt_file.close()
