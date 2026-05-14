"""
Subtype-based object generation for C4_instance_subtype_cued_v1.

Each major category has two hidden subtypes with different affordance profiles.
Visible features partially indicate subtype but do not perfectly determine it.
Probing confirms affordance.

Design principle: category alone is insufficient; probing has marginal value.
"""
import random

# Reuse existing visible feature lists from objects.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "exp004_5a3_tool_material_transfer"))
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
)

# =============================================================================
# Subtype definitions
# =============================================================================

SUBTYPE_DEFINITIONS = {
    "wood_log": {
        "subtypes": ["craftable", "rotten_or_brittle"],
        "subtype_ratio": {"craftable": 0.80, "rotten_or_brittle": 0.20},
        "affordance_profiles": {
            "craftable": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "rotten_or_brittle": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
        },
        # Visible feature probabilities by subtype (partial cue, not deterministic)
        "visible_cue_probs": {
            "craftable": {
                "has_wood_grain": 0.75,
                "has_bark_texture": 0.72,
                "rough_texture": 0.62,
                "solid": 0.85,
                "brownish": 0.58,
                "long_shape": 0.55,
                "block_like": 0.55,
                "movable": 0.65,
                "heavy_weight": 0.55,
            },
            "rotten_or_brittle": {
                "has_wood_grain": 0.38,
                "has_bark_texture": 0.42,
                "rough_texture": 0.48,  # decayed surface may be smoother
                "solid": 0.70,           # less solid
                "brownish": 0.48,
                "long_shape": 0.48,
                "block_like": 0.52,
                "movable": 0.60,
                "heavy_weight": 0.45,
            },
        },
    },
    "stone_block": {
        "subtypes": ["brittle", "hard"],
        "subtype_ratio": {"brittle": 0.20, "hard": 0.80},
        "affordance_profiles": {
            "brittle": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "hard": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
        },
        "visible_cue_probs": {
            "brittle": {
                "has_crystal_flecks": 0.70,
                "has_granular_surface": 0.65,
                "grayish": 0.60,
                "solid": 0.78,
                "rough_texture": 0.55,
                "block_like": 0.58,
                "heavy_weight": 0.72,
            },
            "hard": {
                "has_crystal_flecks": 0.42,
                "has_granular_surface": 0.42,
                "grayish": 0.52,
                "solid": 0.90,
                "rough_texture": 0.50,
                "block_like": 0.54,
                "heavy_weight": 0.78,
            },
        },
    },
    "apple": {
        "subtypes": ["ripe", "unripe_or_spoiled"],
        "subtype_ratio": {"ripe": 0.80, "unripe_or_spoiled": 0.20},
        "affordance_profiles": {
            "ripe": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "unripe_or_spoiled": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
        },
        "visible_cue_probs": {
            "ripe": {
                "has_stem_remnant": 0.72,
                "has_peel_texture": 0.70,
                "greenish": 0.55,
                "round_small": 0.60,
                "smooth_texture": 0.58,
                "solid": 0.72,
                "light_weight": 0.55,
            },
            "unripe_or_spoiled": {
                "has_stem_remnant": 0.40,
                "has_peel_texture": 0.42,
                "greenish": 0.40,
                "round_small": 0.46,
                "smooth_texture": 0.42,
                "solid": 0.65,
                "light_weight": 0.48,
            },
        },
    },
    "wooden_pickaxe": {
        "subtypes": ["intact", "damaged"],
        "subtype_ratio": {"intact": 0.80, "damaged": 0.20},
        "affordance_profiles": {
            "intact": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "success",
            },
            "damaged": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
        },
        "visible_cue_probs": {
            "intact": {
                "has_grip_area": 0.75,
                "has_shaft_shape": 0.72,
                "elongated_with_handle": 0.65,
                "solid": 0.80,
                "movable": 0.72,
                "long_shape": 0.60,
                "smooth_texture": 0.52,
            },
            "damaged": {
                "has_grip_area": 0.38,
                "has_shaft_shape": 0.42,
                "elongated_with_handle": 0.48,
                "solid": 0.65,
                "movable": 0.65,
                "long_shape": 0.50,
                "smooth_texture": 0.45,  # splintered
            },
        },
    },
}


def _make_hidden_profile_subtype(category, subtype, rng):
    """Build hidden_affordance_profile from subtype definition."""
    subtype_def = SUBTYPE_DEFINITIONS[category]
    profile = dict(subtype_def["affordance_profiles"][subtype])
    # Distractor actions vary per object (same as original)
    profile["tap_sound"] = "clear" if rng.random() < 0.5 else "dull"
    profile["push"] = "success" if rng.random() < 0.7 else "fail"
    profile["roll"] = "success" if rng.random() < 0.4 else "fail"
    profile["inspect"] = "success"
    return profile


def _sample_subtype(category, rng):
    """Sample a hidden subtype for a category based on subtype_ratio."""
    subtype_def = SUBTYPE_DEFINITIONS[category]
    subtypes = subtype_def["subtypes"]
    ratios = subtype_def["subtype_ratio"]
    p = rng.random()
    cum = 0.0
    for st in subtypes:
        cum += ratios[st]
        if p < cum:
            return st
    return subtypes[-1]


def _generate_visible_features_subtype(category, subtype, rng):
    """Generate visible features with subtype-dependent probabilities.

    Features not specified in visible_cue_probs fall back to category-level
    defaults from objects._visible_feature_probs.
    """
    from objects import _visible_feature_probs as _cat_probs

    subtype_def = SUBTYPE_DEFINITIONS[category]
    cue_probs = subtype_def["visible_cue_probs"].get(subtype, {})
    cat_probs = _cat_probs(category)

    feats = {}
    for f in ACTIVE_FEATURES_PHASE_A:
        if f in cue_probs:
            feats[f] = rng.random() < cue_probs[f]
        elif f in cat_probs:
            feats[f] = rng.random() < cat_probs[f]
        else:
            feats[f] = rng.random() < 0.5

    _randomize_spurious_visible(feats, rng)

    # Color variants with subtype influence
    from objects import _randomize_color_variants
    color_probs = dict(cat_probs)
    if "brownish" in cue_probs:
        color_probs["brownish"] = cue_probs["brownish"]
    if "grayish" in cue_probs:
        color_probs["grayish"] = cue_probs["grayish"]
    if "greenish" in cue_probs:
        color_probs["greenish"] = cue_probs["greenish"]
    _randomize_color_variants(feats, color_probs, rng)

    # Generate augmented visible features with subtype-dependent probabilities
    _generate_augmented_visible_subtype(feats, category, subtype, rng)

    return feats


def _generate_augmented_visible_subtype(feats, category, subtype, rng):
    """Generate predictor-augmented visible features with subtype influence.

    Augmented features serve as partial subtype cues:
    - For the 'positive' subtype (craftable/ripe/hard/intact), diagnostic features
      are more likely to be true.
    - For the 'negative' subtype, diagnostic features are less likely.
    - Non-diagnostic features remain at baseline (0.15-0.20).
    """
    subtype_def = SUBTYPE_DEFINITIONS[category]
    cue_probs = subtype_def["visible_cue_probs"].get(subtype, {})

    # Feature-to-category diagnostic mapping (same as original)
    feature_cat_map = {
        "has_bark_texture": "wood_log", "has_wood_grain": "wood_log",
        "has_crystal_flecks": "stone_block", "has_granular_surface": "stone_block",
        "has_stem_remnant": "apple", "has_peel_texture": "apple",
        "has_grip_area": "wooden_pickaxe", "has_shaft_shape": "wooden_pickaxe",
    }

    for f in PREDICTOR_AUGMENTED_FEATURES:
        is_diagnostic = feature_cat_map.get(f) == category
        if is_diagnostic:
            # Use subtype cue probability if available, else fall back
            if f in cue_probs:
                feats[f] = rng.random() < cue_probs[f]
            else:
                # Default: higher for positive subtype, lower for negative
                subtype_def = SUBTYPE_DEFINITIONS[category]
                subtypes = subtype_def["subtypes"]
                pos_subtype = subtypes[0]  # first subtype is "positive"
                p = 0.78 if subtype == pos_subtype else 0.38
                feats[f] = rng.random() < p
        else:
            # Non-diagnostic: low baseline probability
            feats[f] = rng.random() < 0.18


def generate_subtype_object(obj_id, category, rng):
    """Generate a single object with hidden subtype."""
    subtype = _sample_subtype(category, rng)
    feats = _generate_visible_features_subtype(category, subtype, rng)
    hidden_profile = _make_hidden_profile_subtype(category, subtype, rng)

    prefix = obj_id.split("_")[0]
    ff_vals = _generate_function_feature_values(
        hidden_profile, category, prefix, rng)

    return {
        "id": obj_id,
        "hidden_category": category,
        "hidden_subtype": subtype,
        "hidden_affordance_profile": hidden_profile,
        "visible_features": feats,
        "function_feature_values": ff_vals,
    }


def generate_subtype_objects(num_logs, num_stones, num_apples, num_pickaxes,
                              rng, prefix="train"):
    """Generate objects with hidden subtypes.

    Same signature as objects.generate_objects() for drop-in compatibility.
    """
    objects = {}
    counts = {
        "wood_log": num_logs,
        "stone_block": num_stones,
        "apple": num_apples,
        "wooden_pickaxe": num_pickaxes,
    }
    for cat, count in counts.items():
        for i in range(count):
            oid = f"{prefix}_{cat}_{i:03d}"
            objects[oid] = generate_subtype_object(oid, cat, rng)
    return objects


def generate_subtype_objects_deterministic(num_logs, num_stones, num_apples,
                                            num_pickaxes, rng, prefix="train"):
    """Generate objects with exact subtype ratios (not random per-object).

    Assigns subtypes deterministically to hit exact target counts, then
    shuffles the assignment order so object IDs don't reveal subtype.
    Visible features are still sampled probabilistically.
    """
    objects = {}
    counts = {
        "wood_log": num_logs,
        "stone_block": num_stones,
        "apple": num_apples,
        "wooden_pickaxe": num_pickaxes,
    }

    for cat, count in counts.items():
        subtype_def = SUBTYPE_DEFINITIONS[cat]
        subtypes = subtype_def["subtypes"]
        ratios = subtype_def["subtype_ratio"]

        # Compute exact subtype counts
        subtype_counts = []
        remaining = count
        for j, st in enumerate(subtypes):
            if j == len(subtypes) - 1:
                n = remaining
            else:
                n = round(count * ratios[st])
                remaining -= n
            subtype_counts.append((st, n))

        # Build ordered list of subtypes and shuffle
        subtype_list = []
        for st, n in subtype_counts:
            subtype_list.extend([st] * n)
        rng.shuffle(subtype_list)

        for i, st in enumerate(subtype_list):
            oid = f"{prefix}_{cat}_{i:03d}"
            feats = _generate_visible_features_subtype(cat, st, rng)
            hidden_profile = _make_hidden_profile_subtype(cat, st, rng)
            ff_vals = _generate_function_feature_values(
                hidden_profile, cat, prefix, rng)
            objects[oid] = {
                "id": oid,
                "hidden_category": cat,
                "hidden_subtype": st,
                "hidden_affordance_profile": hidden_profile,
                "visible_features": feats,
                "function_feature_values": ff_vals,
            }

    return objects


def compute_query_ground_truth(objects):
    """Compute ground truth query answers for a set of objects.

    Returns dict: query_name -> {"positive_oids": [...], "negative_oids": [...]}
    """
    queries = {}
    for qname, conditions in TASK_QUERIES.items():
        pos = []
        neg = []
        for oid, obj in objects.items():
            profile = obj["hidden_affordance_profile"]
            match = True
            for feat_key, expected_val in conditions.items():
                action = feat_key.replace("_success", "")
                actual = profile.get(action, "fail")
                actual_val = 1.0 if actual == "success" else 0.0
                if actual_val != expected_val:
                    match = False
                    break
            if match:
                pos.append(oid)
            else:
                neg.append(oid)
        queries[qname] = {"positive_oids": pos, "negative_oids": neg}
    return queries
