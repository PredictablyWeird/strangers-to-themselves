"""Model identity: reasoning-aware slugs, identity blocks, and the score-time mismatch guard."""

from __future__ import annotations

from behavior_prediction import common
from behavior_prediction.evals import get_spec


def test_reasoning_variants_get_distinct_slugs_and_dirs():
    assert common.model_slug("deepseek-v4-flash") != common.model_slug("deepseek-v4-flash-low")
    spec = get_spec("discrimeval")
    assert common.default_targets_out(spec, "deepseek-v4-flash") \
        != common.default_targets_out(spec, "deepseek-v4-flash-low")


def test_raw_full_string_is_not_misattributed_to_a_reasoning_variant():
    # The full string is shared by 5 deepseek shortcuts -> must NOT silently pick one of them.
    assert common.model_slug("openrouter/deepseek/deepseek-v4-flash") == "openrouter_deepseek_deepseek-v4-flash"
    # An unambiguous, reasoning-free shortcut still resolves from its full string.
    assert common.model_slug("openrouter/meta-llama/llama-3.3-70b-instruct") == "llama-3.3-70b"


def test_model_identity_block():
    mi = common.model_identity("deepseek-v4-flash-low")
    assert mi == {"model": "openrouter/deepseek/deepseek-v4-flash",
                  "shortcut": "deepseek-v4-flash-low",
                  "reasoning": {"reasoning_effort": "low"}}


def test_identity_mismatch_guard():
    base = common.identity_of({"model": "M", "reasoning": {}})
    same = common.identity_of({"model": "M", "reasoning": {}})
    diff_reasoning = common.identity_of({"model": "M", "reasoning": {"reasoning_effort": "low"}})
    diff_model = common.identity_of({"model": "N", "reasoning": {}})
    assert common.identity_mismatch(base, same) is None
    assert common.identity_mismatch(base, diff_reasoning) == "reasoning"
    assert common.identity_mismatch(base, diff_model) == "model"
    # Older files that omit reasoning are tolerated (no false mismatch).
    assert common.identity_mismatch(common.identity_of({"model": "M"}), diff_reasoning) is None
