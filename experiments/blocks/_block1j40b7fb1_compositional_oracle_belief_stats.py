"""
Block 1J40b-7f-b-1 -- Compositional Oracle-Belief Statistic Coverage Audit.

Scope boundary:
- Compute audit-side compositional oracle-belief statistics only.
- Do NOT train a learner.
- Do NOT run policy evaluation.
- Do NOT interpret the research outcome beyond statistic coverage evidence.

Question:
For held-out Test A / Test B signatures, do compositional oracle-belief levels
produce non-global estimates when exact signature lookup must fall back?

Levels:
1. cue_marginal: mean over per-cue training statistics
2. knn_jaccard_signature: weighted nearest training signatures by cue overlap
3. cue_subset_candidate: maximal training signature subsets of the test signature
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Config safety (same as 7e / 7f)
# =============================================================================
C6_LABEL = "C6_observe_try_counterfactual_v0"
existing_labels = [c["label"] for c in config.CUE_CONDITIONS]
if C6_LABEL not in existing_labels:
    config.CUE_CONDITIONS.append({
        "label": C6_LABEL,
        "p_target": 0.60,
        "p_other": 0.40,
        "absent": False,
        "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True,
        "observe_depth_schedules": True,
    })

# =============================================================================
# Constants (from env3b / 7e / 7f)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
OBSERVE_COSTS = [0.01, 0.03, 0.05, 0.07]
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]
COVERAGE_FRACTIONS = [0.25, 0.50]
KNN_K = 3

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel",
    "craft_plank",
    "eat",
    "mine_by_hand",
    "mine_with_pickaxe",
    "use_as_tool",
]

AMBIENT_GROUP_FEATURES = {
    "A": ["brownish", "rough_texture", "long_shape", "heavy_weight"],
    "B": ["greenish", "smooth_texture", "round_small", "light_weight"],
}
CATEGORY_AMBIENT_GROUP = {
    "wood_log": "A",
    "stone_block": "A",
    "apple": "B",
    "wooden_pickaxe": "B",
}
ALL_AMBIENT_FEATURE_NAMES = sorted(set(
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats
))

CATEGORY_POST_OBSERVE_FEATURES = {
    "wood_log": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
    "stone_block": [
        "has_crystal_flecks",
        "has_granular_surface",
        "block_like",
        "cold_to_touch",
        "scratch_resistant",
    ],
    "apple": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
    "wooden_pickaxe": [
        "has_grip_area",
        "has_shaft_shape",
        "elongated_with_handle",
        "movable",
        "has_metal_head",
        "jointed",
    ],
}

SHARED_SURFACE_FEATURES = [
    "spotted",
    "weathered",
    "coarse_texture",
    "discolored",
    "has_veins",
    "pitted",
]

ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    [f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats]
    + [f for feats in CATEGORY_POST_OBSERVE_FEATURES.values() for f in feats]
    + list(SHARED_SURFACE_FEATURES)
))

INSTANCE_SUBTYPE_DEFS = {
    "wood_log": {
        "P": {
            "label": "ambiguous_log",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": [
                "has_bark_texture",
                "fibrous",
                "flammable",
                "porous_surface",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_log",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["spotted"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": [
                "has_bark_texture",
                "fibrous",
                "flammable",
                "porous_surface",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_log",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_bark_texture", "discolored"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "success",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "success",
            },
            "post_observe_revealed": ["fibrous", "flammable", "porous_surface"],
            "state_features": {
                "fresh": False,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
    "stone_block": {
        "P": {
            "label": "ambiguous_stone",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_crystal_flecks",
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
                "scratch_resistant",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_stone",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["spotted"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_crystal_flecks",
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_stone",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_crystal_flecks"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "success",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_granular_surface",
                "block_like",
                "cold_to_touch",
                "scratch_resistant",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": True,
                "clean": False,
                "hot": False,
                "open": False,
            },
        },
    },
    "apple": {
        "P": {
            "label": "ambiguous_fruit",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_stem_remnant",
                "has_peel_texture",
                "fruity_scent",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_apple",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["discolored"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_stem_remnant",
                "has_peel_texture",
                "fruity_scent",
            ],
            "state_features": {
                "fresh": True,
                "wet": True,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_apple",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_stem_remnant", "spotted"],
            "affordance_profile": {
                "mine_by_hand": "success",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "success",
                "use_as_tool": "fail",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": ["has_peel_texture", "fruity_scent"],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
    "wooden_pickaxe": {
        "P": {
            "label": "ambiguous_tool",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": [],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_grip_area",
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
                "has_metal_head",
                "jointed",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "Z": {
            "label": "positive_shared_cue_tool",
            "audit_rationale_class": "positive_change",
            "pre_surface_extra": ["discolored"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_grip_area",
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
        "N": {
            "label": "nonpos_identified_tool",
            "audit_rationale_class": "wasteful_cost_only",
            "pre_surface_extra": ["has_grip_area"],
            "affordance_profile": {
                "mine_by_hand": "fail",
                "mine_with_pickaxe": "fail",
                "craft_plank": "fail",
                "eat": "fail",
                "use_as_tool": "success",
                "burn_as_fuel": "fail",
            },
            "post_observe_revealed": [
                "has_shaft_shape",
                "elongated_with_handle",
                "movable",
                "has_metal_head",
                "jointed",
            ],
            "state_features": {
                "fresh": True,
                "wet": False,
                "damaged": False,
                "clean": True,
                "hot": False,
                "open": False,
            },
        },
    },
}


# =============================================================================
# Object generation (same env3b semantics as 7f)
# =============================================================================
def generate_env3b_objects(seed):
    objects = {}
    audit_labels = {}
    oid_counter = [0]

    def make_oid():
        oid_counter[0] += 1
        return f"env3b_obj_{oid_counter[0]:04d}"

    for category in CATEGORIES:
        ambient_group = CATEGORY_AMBIENT_GROUP[category]
        ambient_features_list = AMBIENT_GROUP_FEATURES[ambient_group]

        for subtype_key in ["P", "Z", "N"]:
            subtype_def = INSTANCE_SUBTYPE_DEFS[category][subtype_key]
            pre_extra = subtype_def["pre_surface_extra"]
            affordance = subtype_def["affordance_profile"]
            post_revealed = subtype_def["post_observe_revealed"]
            state_feats = dict(subtype_def["state_features"])

            pre_surface = {}
            for feat in ambient_features_list:
                pre_surface[feat] = True
            for feat in pre_extra:
                pre_surface[feat] = True

            post_observe = dict(pre_surface)
            for feat in post_revealed:
                post_observe[feat] = True

            hidden = {}
            for feat in ALL_VISIBLE_FEATURE_NAMES:
                if feat not in pre_surface and feat not in post_revealed:
                    hidden[feat] = True

            for depth_schedule in ["no_observe", "one_observe", "repeated_observe"]:
                oid = make_oid()
                objects[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "hidden_affordance_profile": dict(affordance),
                    "pre_surface_features": dict(pre_surface),
                    "post_observe_features": dict(post_observe),
                    "hidden_features": dict(hidden),
                    "visible_state": dict(state_feats),
                    "depth_schedule": depth_schedule,
                }
                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(affordance),
                    "ambient_group": ambient_group,
                    "pre_surface_extra": list(pre_extra),
                    "post_observe_revealed": list(post_revealed),
                    "seed": seed,
                }

    return objects, audit_labels


def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))


def sig_to_display_name(sig):
    feats = list(sig)
    ambient_part = [f for f in feats if f in ALL_AMBIENT_FEATURE_NAMES]
    extra_part = [f for f in feats if f not in ALL_AMBIENT_FEATURE_NAMES]
    amb = ",".join(sorted(ambient_part)[:2])
    ext = ",".join(sorted(extra_part)[:3])
    if ext:
        return f"amb({amb})+{ext}"
    return f"amb({amb})_only"


def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    if success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0


per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels


# =============================================================================
# Signature analysis / stress splits (same semantics as 7c / 7f)
# =============================================================================
sig_map = defaultdict(list)
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        sig_map[sig].append((seed, oid))

SIGNATURE_LIST = sorted(sig_map.keys(), key=lambda s: (-len(sig_map[s]), s))

shared_sigs = []
unique_sigs = []
for sig in SIGNATURE_LIST:
    cats_in_sig = set()
    for seed, oid in sig_map[sig]:
        cats_in_sig.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if len(cats_in_sig) >= 2:
        shared_sigs.append(sig)
    else:
        unique_sigs.append(sig)


def make_test_variant(variant_id, train_sigs, test_sigs, description):
    train_oids = set()
    test_oids = set()
    for sig in train_sigs:
        for seed, oid in sig_map[sig]:
            train_oids.add((seed, oid))
    for sig in test_sigs:
        for seed, oid in sig_map[sig]:
            test_oids.add((seed, oid))
    return {
        "id": variant_id,
        "train_sigs": train_sigs,
        "test_sigs": test_sigs,
        "train_oids": train_oids,
        "test_oids": test_oids,
        "n_train": len(train_oids),
        "n_test": len(test_oids),
        "description": description,
    }


TEST_A_VARIANTS = []
for i in range(min(len(shared_sigs), len(unique_sigs))):
    held_shared = shared_sigs[i]
    held_unique = unique_sigs[i]
    held_sigs = [held_shared, held_unique]
    train_sigs = [s for s in SIGNATURE_LIST if s not in held_sigs]
    TEST_A_VARIANTS.append(make_test_variant(
        variant_id=f"A{i+1}",
        train_sigs=train_sigs,
        test_sigs=held_sigs,
        description=(
            f"Hold out shared {sig_to_display_name(held_shared)} + "
            f"unique {sig_to_display_name(held_unique)}"
        ),
    ))

apple_n_sig = None
woodlog_n_sig = None
for sig in unique_sigs:
    cats_in = set()
    for seed, oid in sig_map[sig]:
        cats_in.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if cats_in == {"apple"}:
        apple_n_sig = sig
    if cats_in == {"wood_log"}:
        woodlog_n_sig = sig

ambient_b_only_sig = None
ambient_a_only_sig = None
for sig in shared_sigs:
    some_seed, some_oid = sig_map[sig][0]
    ambient_group = per_seed_audit_labels[some_seed][some_oid]["ambient_group"]
    is_ambient_only = len(sig) == 4
    if ambient_group == "B" and is_ambient_only:
        ambient_b_only_sig = sig
    if ambient_group == "A" and is_ambient_only:
        ambient_a_only_sig = sig

TEST_B_VARIANTS = []
b1_test_sigs = [s for s in [apple_n_sig, ambient_b_only_sig] if s is not None]
b2_test_sigs = [s for s in [woodlog_n_sig, ambient_a_only_sig] if s is not None]
if b1_test_sigs:
    TEST_B_VARIANTS.append(make_test_variant(
        variant_id="B1",
        train_sigs=[s for s in SIGNATURE_LIST if s not in b1_test_sigs],
        test_sigs=b1_test_sigs,
        description="Hold out apple-N spotted non-positive + ambient-B shared signature",
    ))
if b2_test_sigs:
    TEST_B_VARIANTS.append(make_test_variant(
        variant_id="B2",
        train_sigs=[s for s in SIGNATURE_LIST if s not in b2_test_sigs],
        test_sigs=b2_test_sigs,
        description="Hold out wood_log-N discolored non-positive + ambient-A shared signature",
    ))


# =============================================================================
# Counterfactual intervention helpers (same 7e / 7f logic)
# =============================================================================
AMBIENT_GROUP_CATEGORIES = {
    "A": ["wood_log", "stone_block"],
    "B": ["apple", "wooden_pickaxe"],
}


def get_flat_affordance(all_success=True):
    outcome = "success" if all_success else "fail"
    return {a: outcome for a in ALL_TRY_AFFORDANCES}


def get_lookalike_affordance(seed, oid):
    lbl = per_seed_audit_labels[seed][oid]
    original_cat = lbl["hidden_category"]
    ambient_group = lbl["ambient_group"]
    other_cats = [c for c in AMBIENT_GROUP_CATEGORIES[ambient_group] if c != original_cat]
    if other_cats:
        other_cat = other_cats[0]
        return dict(INSTANCE_SUBTYPE_DEFS[other_cat]["P"]["affordance_profile"]), other_cat
    return get_flat_affordance(True), None


def build_counterfactual_affordance_overrides(train_oids, coverage_fraction):
    rng = random.Random(42)
    train_by_sig = defaultdict(list)
    for seed, oid in train_oids:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        train_by_sig[sig].append((seed, oid))

    overrides = {}
    intervention_labels = {}
    lookalike_targets = {}
    for sig, instances in sorted(train_by_sig.items()):
        n_flip = max(1, int(len(instances) * coverage_fraction))
        flipped = rng.sample(instances, n_flip)
        some_lbl = per_seed_audit_labels[instances[0][0]][instances[0][1]]
        is_shared = some_lbl["hidden_audit_rationale_class"] == "positive_change"

        for seed, oid in flipped:
            if is_shared:
                flat_val = "success" if rng.random() < 0.5 else "fail"
                new_aff = get_flat_affordance(flat_val == "success")
            else:
                new_aff, target_cat = get_lookalike_affordance(seed, oid)
                lookalike_targets[(seed, oid)] = target_cat
            overrides[(seed, oid)] = new_aff
            intervention_labels[(seed, oid)] = (
                "decision_irrelevant_flat" if is_shared else "lookalike_deceptive_exception"
            )

    return overrides, intervention_labels, lookalike_targets


# =============================================================================
# True/oracle VOI (same 7f semantics)
# =============================================================================
def compute_true_oracle_voi(oid_set, observe_cost, affordance_overrides=None):
    sig_instances = defaultdict(list)
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if aff is None:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        sig_instances[sig].append((seed, oid, aff))

    sig_expected = {}
    for sig, instances in sig_instances.items():
        action_sums = {a: 0.0 for a in ALL_TRY_AFFORDANCES}
        for _, _, aff in instances:
            for action in ALL_TRY_AFFORDANCES:
                action_sums[action] += try_net_value(aff.get(action, "fail") == "success")
        n = len(instances)
        action_exp = {a: action_sums[a] / n for a in ALL_TRY_AFFORDANCES}
        best_action = max(action_exp, key=action_exp.get)
        sig_expected[sig] = {
            "best_expected_value": action_exp[best_action],
            "best_action": best_action,
            "n_instances": n,
        }

    result = {}
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        true_pre_best = sig_expected[sig]["best_expected_value"]
        aff = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if aff is None:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        true_post_best = max(
            try_net_value(aff.get(action, "fail") == "success")
            for action in ALL_TRY_AFFORDANCES
        )
        true_voi = true_post_best - true_pre_best - observe_cost
        result[(seed, oid)] = {
            "true_pre_best": true_pre_best,
            "true_post_best": true_post_best,
            "true_voi": true_voi,
            "true_optimal": true_voi > 0,
        }
    return result


# =============================================================================
# Statistic builders
# =============================================================================
def summarize_rows(rows):
    n = len(rows)
    n_positive = sum(1 for r in rows if r["true_optimal"])
    return {
        "n": n,
        "n_positive": n_positive,
        "p_positive": (n_positive / n) if n else 0.0,
        "e_observe_net": (sum(r["true_voi"] for r in rows) / n) if n else 0.0,
    }


def build_train_rows(train_oids, true_voi):
    rows = []
    for seed, oid in sorted(train_oids):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        rows.append({
            "seed": seed,
            "oid": oid,
            "signature": sig,
            "true_voi": true_voi[(seed, oid)]["true_voi"],
            "true_optimal": true_voi[(seed, oid)]["true_optimal"],
        })
    return rows


def build_signature_stats(train_rows):
    stats = defaultdict(list)
    for row in train_rows:
        stats[row["signature"]].append(row)
    return {sig: summarize_rows(rows) for sig, rows in stats.items()}


def build_cue_stats(train_rows):
    cue_rows = defaultdict(list)
    for row in train_rows:
        for cue in row["signature"]:
            cue_rows[cue].append(row)
    return {cue: summarize_rows(rows) for cue, rows in cue_rows.items()}


def jaccard_similarity(sig_a, sig_b):
    set_a = set(sig_a)
    set_b = set(sig_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def level1_cue_marginal_estimate(test_sig, cue_stats, global_stats):
    matched_cues = [cue for cue in test_sig if cue in cue_stats]
    if not matched_cues:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "matched_cues": [],
            **global_stats,
        }

    matched_stats = [cue_stats[cue] for cue in matched_cues]
    return {
        "source_type": "cue_marginal",
        "used_global_fallback": False,
        "matched_cues": matched_cues,
        "matched_non_ambient_cues": [
            cue for cue in matched_cues if cue not in ALL_AMBIENT_FEATURE_NAMES
        ],
        "n": sum(s["n"] for s in matched_stats),
        "n_positive": sum(s["n_positive"] for s in matched_stats),
        "p_positive": sum(s["p_positive"] for s in matched_stats) / len(matched_stats),
        "e_observe_net": sum(s["e_observe_net"] for s in matched_stats) / len(matched_stats),
    }


def level2_knn_estimate(test_sig, signature_stats, global_stats, k=KNN_K):
    neighbors = []
    for train_sig, stats in signature_stats.items():
        sim = jaccard_similarity(test_sig, train_sig)
        if sim > 0.0:
            neighbors.append((sim, train_sig, stats))

    neighbors.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
    top_neighbors = neighbors[:k]
    if not top_neighbors:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "neighbors": [],
            **global_stats,
        }

    total_w = sum(sim for sim, _, _ in top_neighbors)
    if total_w <= 0:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "neighbors": [],
            **global_stats,
        }

    return {
        "source_type": "knn_jaccard_signature",
        "used_global_fallback": False,
        "neighbors": [
            {
                "signature": list(train_sig),
                "signature_name": sig_to_display_name(train_sig),
                "jaccard": round(sim, 4),
                "n": stats["n"],
            }
            for sim, train_sig, stats in top_neighbors
        ],
        "n": sum(stats["n"] for _, _, stats in top_neighbors),
        "n_positive": sum(stats["n_positive"] for _, _, stats in top_neighbors),
        "p_positive": sum(sim * stats["p_positive"] for sim, _, stats in top_neighbors) / total_w,
        "e_observe_net": sum(sim * stats["e_observe_net"] for sim, _, stats in top_neighbors) / total_w,
    }


def level3_subset_candidate_estimate(test_sig, signature_stats, global_stats):
    test_set = set(test_sig)
    best_size = -1
    matched = []
    for train_sig, stats in signature_stats.items():
        train_set = set(train_sig)
        if train_set.issubset(test_set):
            if len(train_set) > best_size:
                best_size = len(train_set)
                matched = [(train_sig, stats)]
            elif len(train_set) == best_size:
                matched.append((train_sig, stats))

    if not matched:
        return {
            "source_type": "global_fallback",
            "used_global_fallback": True,
            "matched_subset_size": 0,
            "matched_signatures": [],
            **global_stats,
        }

    return {
        "source_type": "cue_subset_candidate",
        "used_global_fallback": False,
        "matched_subset_size": best_size,
        "matched_signatures": [
            {
                "signature": list(train_sig),
                "signature_name": sig_to_display_name(train_sig),
                "n": stats["n"],
            }
            for train_sig, stats in matched
        ],
        "n": sum(stats["n"] for _, stats in matched),
        "n_positive": sum(stats["n_positive"] for _, stats in matched),
        "p_positive": sum(stats["p_positive"] for _, stats in matched) / len(matched),
        "e_observe_net": sum(stats["e_observe_net"] for _, stats in matched) / len(matched),
    }


def exact_signature_lookup_available(test_sig, signature_stats):
    return test_sig in signature_stats


def build_condition_overrides(train_oids, condition_name):
    if condition_name == "original":
        return None
    if condition_name == "counterfactual_cov0.25":
        return build_counterfactual_affordance_overrides(train_oids, 0.25)[0]
    if condition_name == "counterfactual_cov0.50":
        return build_counterfactual_affordance_overrides(train_oids, 0.50)[0]
    raise ValueError(f"Unknown condition: {condition_name}")


def audit_variant(variant, test_group_name, condition_name, observe_cost):
    train_oids = variant["train_oids"]
    overrides = build_condition_overrides(train_oids, condition_name)
    true_voi = compute_true_oracle_voi(train_oids, observe_cost, overrides)
    train_rows = build_train_rows(train_oids, true_voi)
    global_stats = summarize_rows(train_rows)
    signature_stats = build_signature_stats(train_rows)
    cue_stats = build_cue_stats(train_rows)

    held_signature_results = []
    for test_sig in variant["test_sigs"]:
        sample_seed, sample_oid = sig_map[test_sig][0]
        held_signature_results.append({
            "test_group": test_group_name,
            "variant_id": variant["id"],
            "condition": condition_name,
            "observe_cost": observe_cost,
            "held_signature": list(test_sig),
            "held_signature_name": sig_to_display_name(test_sig),
            "held_signature_size": len(test_sig),
            "held_signature_ambient_group": per_seed_audit_labels[sample_seed][sample_oid]["ambient_group"],
            "exact_signature_lookup_available": exact_signature_lookup_available(test_sig, signature_stats),
            "global_stats": global_stats,
            "level1_cue_marginal": level1_cue_marginal_estimate(test_sig, cue_stats, global_stats),
            "level2_knn_jaccard": level2_knn_estimate(test_sig, signature_stats, global_stats),
            "level3_subset_candidate": level3_subset_candidate_estimate(test_sig, signature_stats, global_stats),
        })

    return {
        "variant_id": variant["id"],
        "description": variant["description"],
        "n_train": variant["n_train"],
        "n_test": variant["n_test"],
        "held_signatures": held_signature_results,
    }


def summarize_group(variant_results):
    level_keys = [
        "level1_cue_marginal",
        "level2_knn_jaccard",
        "level3_subset_candidate",
    ]
    all_sigs = []
    for vr in variant_results:
        all_sigs.extend(vr["held_signatures"])

    summary = {
        "n_variant_signatures": len(all_sigs),
        "n_exact_signature_non_global": sum(
            1 for row in all_sigs if row["exact_signature_lookup_available"]
        ),
        "levels": {},
    }
    for level_key in level_keys:
        level_rows = [row[level_key] for row in all_sigs]
        summary["levels"][level_key] = {
            "n_non_global": sum(1 for row in level_rows if not row["used_global_fallback"]),
            "n_global_fallback": sum(1 for row in level_rows if row["used_global_fallback"]),
            "n_positive_estimate": sum(1 for row in level_rows if row["e_observe_net"] > 0.0),
        }
    return summary


print("[1/4] Auditing compositional statistic coverage...")
condition_names = ["original", "counterfactual_cov0.25", "counterfactual_cov0.50"]
output = {
    "block_id": "1J40b-7f-b-1",
    "implementation_status": "COMPLETED",
    "elapsed_seconds": None,
    "scope": "compositional_oracle_belief_statistics_only",
    "observe_costs": OBSERVE_COSTS,
    "condition_names": condition_names,
    "level_definitions": {
        "level1_cue_marginal": "Average of per-cue training statistics over legal pre_surface cues present in the held-out signature.",
        "level2_knn_jaccard": f"Top-{KNN_K} nearest training signatures by Jaccard overlap over legal pre_surface cues.",
        "level3_subset_candidate": "Aggregate over maximal training signatures whose cue set is a subset of the held-out signature.",
        "global_fallback": "Training-wide aggregate over all training objects in the split/condition/cost.",
    },
    "results": {},
}

for observe_cost in OBSERVE_COSTS:
    print(f"  observe_cost={observe_cost:.2f}")
    for condition_name in condition_names:
        print(f"    condition={condition_name}")
        for test_group_name, variants in [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS)]:
            variant_results = [
                audit_variant(variant, test_group_name, condition_name, observe_cost)
                for variant in variants
            ]
            output["results"][f"cost{observe_cost:.2f}_{condition_name}_{test_group_name}"] = {
                "observe_cost": observe_cost,
                "condition": condition_name,
                "test_group": test_group_name,
                "summary": summarize_group(variant_results),
                "variants": variant_results,
            }

output["elapsed_seconds"] = round(time.time() - t0, 2)


# =============================================================================
# Write outputs
# =============================================================================
print("[2/4] Writing JSON output...")
runs_dir = os.path.join(CURRENT_DIR, "runs")
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(protocols_dir, exist_ok=True)

json_path = os.path.join(
    runs_dir, "block1j40b7fb1_compositional_oracle_belief_stats.json"
)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("[3/4] Writing compact markdown protocol...")
md_lines = []
md_lines.append("# Block 1J40b-7f-b-1: Compositional Oracle-Belief Statistic Coverage")
md_lines.append("")
md_lines.append(f"- **Implementation Status**: {output['implementation_status']}")
md_lines.append(f"- **Elapsed**: {output['elapsed_seconds']:.2f}s")
md_lines.append(f"- **Observe Costs**: {OBSERVE_COSTS}")
md_lines.append(f"- **Conditions**: {condition_names}")
md_lines.append("")
md_lines.append("## Scope")
md_lines.append("")
md_lines.append("- Compute compositional oracle-belief statistics only.")
md_lines.append("- No learner training.")
md_lines.append("- No policy evaluation.")
md_lines.append("- Key question: for held-out Test A/B signatures, which levels avoid global fallback?")
md_lines.append("")
md_lines.append("## Level Definitions")
md_lines.append("")
for level_key, description in output["level_definitions"].items():
    md_lines.append(f"- **{level_key}**: {description}")
md_lines.append("")

for observe_cost in OBSERVE_COSTS:
    md_lines.append(f"## Results at observe_cost = {observe_cost:.2f}")
    md_lines.append("")
    md_lines.append("| Test | Condition | exact_sig_non_global | total_sig_cases | L1 non_global | L2 non_global | L3 non_global |")
    md_lines.append("|------|-----------|----------------------|-----------------|---------------|---------------|---------------|")
    for condition_name in condition_names:
        for test_group_name in ["A", "B"]:
            key = f"cost{observe_cost:.2f}_{condition_name}_{test_group_name}"
            summary = output["results"][key]["summary"]
            levels = summary["levels"]
            md_lines.append(
                f"| {test_group_name} | {condition_name} | "
                f"{summary['n_exact_signature_non_global']} | {summary['n_variant_signatures']} | "
                f"{levels['level1_cue_marginal']['n_non_global']} | "
                f"{levels['level2_knn_jaccard']['n_non_global']} | "
                f"{levels['level3_subset_candidate']['n_non_global']} |"
            )
    md_lines.append("")

md_lines.append("## Per-Signature Evidence")
md_lines.append("")
for key in sorted(output["results"].keys()):
    result = output["results"][key]
    md_lines.append(
        f"### {key} "
        f"(exact_sig_non_global={result['summary']['n_exact_signature_non_global']}/"
        f"{result['summary']['n_variant_signatures']})"
    )
    md_lines.append("")
    md_lines.append("| Variant | Held Signature | Exact Lookup | L1 Source | L2 Source | L3 Source |")
    md_lines.append("|---------|----------------|--------------|-----------|-----------|-----------|")
    for variant in result["variants"]:
        for row in variant["held_signatures"]:
            l1 = row["level1_cue_marginal"]
            l2 = row["level2_knn_jaccard"]
            l3 = row["level3_subset_candidate"]
            l1_src = "global" if l1["used_global_fallback"] else f"cue[{len(l1.get('matched_cues', []))}]"
            l2_src = "global" if l2["used_global_fallback"] else f"knn[{len(l2.get('neighbors', []))}]"
            l3_src = "global" if l3["used_global_fallback"] else f"subset[{l3.get('matched_subset_size', 0)}]"
            md_lines.append(
                f"| {variant['variant_id']} | {row['held_signature_name']} | "
                f"{'yes' if row['exact_signature_lookup_available'] else 'no'} | "
                f"{l1_src} | {l2_src} | {l3_src} |"
            )
    md_lines.append("")

md_path = os.path.join(
    protocols_dir, "block1j40b7fb1_compositional_oracle_belief_stats.md"
)
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("[4/4] Done.")
print(f"  JSON: {json_path}")
print(f"  MD:   {md_path}")
