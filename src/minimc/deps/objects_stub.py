"""Stub: object generation — originally from exp004_5a3_tool_material_transfer.objects.

Provide the real implementation or install the package to use.
"""

# Visible feature lists (used by subtype_objects.py)
ACTIVE_FEATURES_PHASE_A = []
PREDICTOR_AUGMENTED_FEATURES = []
SPURIOUS_VISIBLE_FEATURES = []
CORE_ACTION_FEATURES = []
BASIC_ACTION_DISTRACTORS = []
NOISE_FEATURES = []
WEAK_CORRELATED_FEATURES = []
SPURIOUS_FEATURES = []
IRRELEVANT_VALID_FEATURES = []
ALL_CANDIDATE_FUNCTION_FEATURES = []
PROBE_ACTIONS = []
TASK_QUERIES = {}
TASK_QUERY_ANSWERS = {}

try:
    from objects import (
        ACTIVE_FEATURES_PHASE_A,
        PREDICTOR_AUGMENTED_FEATURES,
        SPURIOUS_VISIBLE_FEATURES,
        CORE_ACTION_FEATURES,
        BASIC_ACTION_DISTRACTORS,
        NOISE_FEATURES,
        WEAK_CORRELATED_FEATURES,
        SPURIOUS_FEATURES,
        IRRELEVANT_VALID_FEATURES,
        ALL_CANDIDATE_FUNCTION_FEATURES,
        PROBE_ACTIONS,
        TASK_QUERIES,
        TASK_QUERY_ANSWERS,
        _randomize_spurious_visible,
        _generate_function_feature_values,
        _visible_feature_probs,
        _randomize_color_variants,
    )
except ImportError:
    # Stub the required functions so the module can be imported
    import random as _random

    def _randomize_spurious_visible(feats, rng=None):
        pass

    def _generate_function_feature_values(hidden_profile, category, prefix, rng=None):
        return {}

    def _visible_feature_probs(category):
        return {}

    def _randomize_color_variants(feats, color_probs, rng=None):
        pass
