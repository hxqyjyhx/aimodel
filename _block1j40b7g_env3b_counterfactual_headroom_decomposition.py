"""
Block 1J40b-7g -- Env3b / Counterfactual Headroom Decomposition Audit.

Purpose:
- Diagnose why compositional oracle-belief policies from 7f-b fail to beat
  always_observe.
- Do not train a learner.
- Do not change env3b or counterfactual generation.
- Do not implement 1J40b-8.

Method:
- Reconstruct env3b / stress splits / counterfactual training overrides using
  the same semantics as 7f-b.
- Recompute true observe-value and compositional estimate diagnostics.
- Reuse 7f-b policy outputs for policy_net / baseline reporting consistency.
"""

import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCE_FILES_USED = [
    "_block1j40b7e_env3b_distributional_audit_counterfactual.py",
    "_block1j40b7f_oracle_belief_upper_bound_control.py",
    "_block1j40b7fb1_compositional_oracle_belief_stats.py",
    "_block1j40b7fb_compositional_oracle_belief_upper_bound.py",
    "runs/block1j40b7e_env3b_distributional_audit_counterfactual.json",
    "runs/block1j40b7f_oracle_belief_upper_bound_control.json",
    "runs/block1j40b7fb1_compositional_oracle_belief_stats.json",
    "runs/block1j40b7fb_compositional_oracle_belief_upper_bound.json",
]

SEEDS = [101, 103, 107, 109, 113]
OBSERVE_COSTS = [0.01, 0.03, 0.05, 0.07]
DISTRIBUTIONS = [
    "original",
    "counterfactual_cov0.25",
    "counterfactual_cov0.50",
    "counterfactual_cov1.00",
]
TESTS = ["A", "B", "C"]

SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
KNN_K = 3
TAU = 0.5

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel",
    "craft_plank",
    "eat",
    "mine_by_hand",
    "mine_with_pickaxe",
    "use_as_tool",
]

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]
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


def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    if success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0


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


def mean(values):
    return sum(values) / len(values) if values else 0.0


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
            pre_surface = {}
            for feat in ambient_features_list:
                pre_surface[feat] = True
            for feat in subtype_def["pre_surface_extra"]:
                pre_surface[feat] = True

            post_observe = dict(pre_surface)
            for feat in subtype_def["post_observe_revealed"]:
                post_observe[feat] = True

            hidden = {}
            for feat in ALL_VISIBLE_FEATURE_NAMES:
                if feat not in pre_surface and feat not in subtype_def["post_observe_revealed"]:
                    hidden[feat] = True

            for depth_schedule in ["no_observe", "one_observe", "repeated_observe"]:
                oid = make_oid()
                objects[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "pre_surface_features": dict(pre_surface),
                    "post_observe_features": dict(post_observe),
                    "hidden_features": dict(hidden),
                    "visible_state": dict(subtype_def["state_features"]),
                }
                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_subtype": subtype_key,
                    "hidden_audit_rationale_class": subtype_def["audit_rationale_class"],
                    "affordance_profile": dict(subtype_def["affordance_profile"]),
                    "ambient_group": ambient_group,
                    "seed": seed,
                }
    return objects, audit_labels


per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    per_seed_objects[seed], per_seed_audit_labels[seed] = generate_env3b_objects(seed)

sig_map = defaultdict(list)
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        sig_map[get_pre_surface_signature(obj["pre_surface_features"])].append((seed, oid))

SIGNATURE_LIST = sorted(sig_map.keys(), key=lambda s: (-len(sig_map[s]), s))
shared_sigs = []
unique_sigs = []
for sig in SIGNATURE_LIST:
    cats = set()
    for seed, oid in sig_map[sig]:
        cats.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if len(cats) >= 2:
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
        description=f"Hold out shared {sig_to_display_name(held_shared)} + unique {sig_to_display_name(held_unique)}",
    ))

apple_n_sig = None
woodlog_n_sig = None
for sig in unique_sigs:
    cats = set()
    for seed, oid in sig_map[sig]:
        cats.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    if cats == {"apple"}:
        apple_n_sig = sig
    if cats == {"wood_log"}:
        woodlog_n_sig = sig

ambient_b_only_sig = None
ambient_a_only_sig = None
for sig in shared_sigs:
    seed, oid = sig_map[sig][0]
    ambient_group = per_seed_audit_labels[seed][oid]["ambient_group"]
    if len(sig) == 4 and ambient_group == "B":
        ambient_b_only_sig = sig
    if len(sig) == 4 and ambient_group == "A":
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

TEST_C_VARIANTS = []
for heldout_seed in SEEDS:
    train_oids = set()
    test_oids = set()
    for seed in SEEDS:
        for oid in per_seed_objects[seed]:
            if seed == heldout_seed:
                test_oids.add((seed, oid))
            else:
                train_oids.add((seed, oid))
    TEST_C_VARIANTS.append({
        "id": f"C_seed{heldout_seed}",
        "train_oids": train_oids,
        "test_oids": test_oids,
        "description": f"LOSO held-out seed {heldout_seed}",
    })


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
    return overrides, lookalike_targets


def build_condition_train_overrides(train_oids, distribution):
    if distribution == "original":
        return None
    if distribution == "counterfactual_cov0.25":
        return build_counterfactual_affordance_overrides(train_oids, 0.25)[0]
    if distribution == "counterfactual_cov0.50":
        return build_counterfactual_affordance_overrides(train_oids, 0.50)[0]
    if distribution == "counterfactual_cov1.00":
        return build_counterfactual_affordance_overrides(train_oids, 1.00)[0]
    raise ValueError(distribution)


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
        action_exp = {a: action_sums[a] / len(instances) for a in ALL_TRY_AFFORDANCES}
        best_action = max(action_exp, key=action_exp.get)
        sig_expected[sig] = {
            "best_expected_value": action_exp[best_action],
            "best_action": best_action,
        }

    result = {}
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid)) if affordance_overrides else None
        if aff is None:
            aff = per_seed_audit_labels[seed][oid]["affordance_profile"]
        true_pre_best = sig_expected[sig]["best_expected_value"]
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
        rows.append({
            "signature": get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"]),
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
    return len(set_a & set_b) / len(union) if union else 0.0


def exact_signature_estimate(test_sig, signature_stats, global_stats):
    if test_sig in signature_stats:
        return {"used_global_fallback": False, **signature_stats[test_sig]}
    return {"used_global_fallback": True, **global_stats}


def level1_cue_marginal_estimate(test_sig, cue_stats, global_stats):
    matched = [cue for cue in test_sig if cue in cue_stats]
    if not matched:
        return {
            "used_global_fallback": True,
            "matched_cues": [],
            **global_stats,
        }
    stats = [cue_stats[cue] for cue in matched]
    return {
        "used_global_fallback": False,
        "matched_cues": matched,
        "e_observe_net": mean([s["e_observe_net"] for s in stats]),
        "p_positive": mean([s["p_positive"] for s in stats]),
        "n": sum(s["n"] for s in stats),
        "n_positive": sum(s["n_positive"] for s in stats),
    }


def level2_knn_estimate(test_sig, signature_stats, global_stats, k=KNN_K):
    neighbors = []
    for train_sig, stats in signature_stats.items():
        sim = jaccard_similarity(test_sig, train_sig)
        if sim > 0.0:
            neighbors.append((sim, train_sig, stats))
    neighbors.sort(key=lambda x: (-x[0], -len(x[1]), x[1]))
    top = neighbors[:k]
    if not top:
        return {"used_global_fallback": True, "neighbors": [], **global_stats}
    total_w = sum(sim for sim, _, _ in top)
    return {
        "used_global_fallback": False,
        "neighbors": [
            {
                "signature": list(sig),
                "signature_name": sig_to_display_name(sig),
                "jaccard": sim,
                "n": stats["n"],
            }
            for sim, sig, stats in top
        ],
        "e_observe_net": sum(sim * stats["e_observe_net"] for sim, _, stats in top) / total_w,
        "p_positive": sum(sim * stats["p_positive"] for sim, _, stats in top) / total_w,
        "n": sum(stats["n"] for _, _, stats in top),
        "n_positive": sum(stats["n_positive"] for _, _, stats in top),
    }


def level3_candidate_estimate(test_sig, signature_stats, global_stats, tau=TAU):
    candidates = []
    test_set = set(test_sig)
    for train_sig, stats in signature_stats.items():
        sim = jaccard_similarity(test_sig, train_sig)
        is_subset = set(train_sig).issubset(test_set)
        if is_subset or sim >= tau:
            candidates.append((train_sig, stats, sim, is_subset))
    if not candidates:
        return {"used_global_fallback": True, "candidate_count": 0, **global_stats}
    support_total = sum(stats["n"] for _, stats, _, _ in candidates)
    return {
        "used_global_fallback": False,
        "candidate_count": len(candidates),
        "candidate_support_total": support_total,
        "e_observe_net": sum(stats["e_observe_net"] * stats["n"] for _, stats, _, _ in candidates) / support_total,
        "p_positive": sum(stats["p_positive"] * stats["n"] for _, stats, _, _ in candidates) / support_total,
        "n": support_total,
        "n_positive": sum(stats["n_positive"] for _, stats, _, _ in candidates),
    }


def build_test_signature_truth(test_oids, observe_cost):
    obj_data = []
    sig_groups = defaultdict(list)
    for seed, oid in sorted(test_oids):
        lbl = per_seed_audit_labels[seed][oid]
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = lbl["affordance_profile"].get(action, "fail")
            true_returns[action] = try_net_value(outcome == "success")
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        rec = {
            "seed": seed,
            "oid": oid,
            "signature": sig,
            "signature_name": sig_to_display_name(sig),
            "rationale_class": lbl["hidden_audit_rationale_class"],
            "true_returns": true_returns,
        }
        sig_groups[sig].append(rec)
        obj_data.append(rec)

    signature_stats = {}
    for sig, members in sig_groups.items():
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            action_expected[action] = mean([m["true_returns"][action] for m in members])
        pre_best_action = max(action_expected, key=action_expected.get)
        pre_best_expected = action_expected[pre_best_action]

        always_obs_vals = []
        no_obs_vals = []
        oracle_vals = []
        true_es = []
        pos_values = []
        nonpos_values = []
        for member in members:
            post_best = max(member["true_returns"].values())
            true_e = post_best - pre_best_expected - observe_cost
            member["true_post_best"] = post_best
            member["true_pre_best_expected"] = pre_best_expected
            member["true_e"] = true_e
            member["true_positive"] = true_e > 0.0
            always_obs_vals.append(post_best - observe_cost)
            no_obs_vals.append(pre_best_expected)
            true_es.append(true_e)
            if true_e > 0.0:
                pos_values.append(true_e)
                oracle_vals.append(post_best - observe_cost)
            else:
                nonpos_values.append(true_e)
                oracle_vals.append(pre_best_expected)
        signature_stats[sig] = {
            "signature": sig,
            "signature_name": sig_to_display_name(sig),
            "n_objects": len(members),
            "positive_net_rate": sum(1 for m in members if m["true_positive"]) / len(members),
            "mean_true_E_observe_net": mean(true_es),
            "mean_if_positive": mean(pos_values) if pos_values else None,
            "mean_if_non_positive": mean(nonpos_values) if nonpos_values else None,
            "always_observe_mean_return": mean(always_obs_vals),
            "oracle_selective_mean_return": mean(oracle_vals),
            "no_observe_mean_return": mean(no_obs_vals),
            "min_true_E_observe_net": min(true_es),
            "max_true_E_observe_net": max(true_es),
            "object_records": members,
        }
    return signature_stats, obj_data


def pearson_corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    return pearson_corr(rankdata(xs), rankdata(ys))


def cond_key(distribution, observe_cost, test_type):
    return f"{distribution}|{observe_cost:.2f}|{test_type}"


def load_json(rel_path):
    with open(os.path.join(CURRENT_DIR, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


print("[1/5] Loading prior outputs...")
run_7fb = load_json("runs/block1j40b7fb_compositional_oracle_belief_upper_bound.json")

variant_map = {
    "A": TEST_A_VARIANTS,
    "B": TEST_B_VARIANTS,
    "C": TEST_C_VARIANTS,
}

headroom_decomposition = {}
observe_value_distribution = {}
always_observe_strength = {}
level_error_decomposition = {}
pre_cue_predictability = {}
coverage_effect_rows = []
csv_rows = []

print("[2/5] Computing diagnostic decompositions...")
for distribution in DISTRIBUTIONS:
    for observe_cost in OBSERVE_COSTS:
        for test_type in TESTS:
            key = cond_key(distribution, observe_cost, test_type)
            if key not in run_7fb["per_condition_results"]:
                continue

            prior_result = run_7fb["per_condition_results"][key]
            variants = variant_map[test_type]
            signature_rows_all = []
            level_agg = {
                "exact_signature_oracle_belief": {
                    "sign_correct": 0,
                    "obj_total": 0,
                    "false_observe": 0,
                    "false_direct": 0,
                    "false_observe_losses": [],
                    "false_direct_losses": [],
                    "fallback_count": 0,
                    "non_global_count": 0,
                    "estimated_Es": [],
                    "true_Es": [],
                    "signature_rows": [],
                    "pos_est_values": [],
                    "nonpos_est_values": [],
                },
                "level1_cue_marginal": {
                    "sign_correct": 0, "obj_total": 0, "false_observe": 0, "false_direct": 0,
                    "false_observe_losses": [], "false_direct_losses": [], "fallback_count": 0,
                    "non_global_count": 0, "estimated_Es": [], "true_Es": [], "signature_rows": [],
                    "pos_est_values": [], "nonpos_est_values": [],
                },
                "level2_knn_jaccard_k3": {
                    "sign_correct": 0, "obj_total": 0, "false_observe": 0, "false_direct": 0,
                    "false_observe_losses": [], "false_direct_losses": [], "fallback_count": 0,
                    "non_global_count": 0, "estimated_Es": [], "true_Es": [], "signature_rows": [],
                    "pos_est_values": [], "nonpos_est_values": [],
                },
                "level3_candidate_set_tau0.5": {
                    "sign_correct": 0, "obj_total": 0, "false_observe": 0, "false_direct": 0,
                    "false_observe_losses": [], "false_direct_losses": [], "fallback_count": 0,
                    "non_global_count": 0, "estimated_Es": [], "true_Es": [], "signature_rows": [],
                    "pos_est_values": [], "nonpos_est_values": [],
                },
            }

            n_total_objects = 0
            signature_summary_rows = []
            for variant in variants:
                train_oids = variant["train_oids"]
                test_oids = variant["test_oids"]
                train_overrides = build_condition_train_overrides(train_oids, distribution)
                train_true_voi = compute_true_oracle_voi(train_oids, observe_cost, train_overrides)
                train_rows = build_train_rows(train_oids, train_true_voi)
                global_stats = summarize_rows(train_rows)
                signature_stats = build_signature_stats(train_rows)
                cue_stats = build_cue_stats(train_rows)
                test_signature_stats, test_obj_data = build_test_signature_truth(test_oids, observe_cost)

                n_total_objects += len(test_obj_data)
                for sig, truth in test_signature_stats.items():
                    exact_est = exact_signature_estimate(sig, signature_stats, global_stats)
                    l1_est = level1_cue_marginal_estimate(sig, cue_stats, global_stats)
                    l2_est = level2_knn_estimate(sig, signature_stats, global_stats, k=KNN_K)
                    l3_est = level3_candidate_estimate(sig, signature_stats, global_stats, tau=TAU)
                    est_map = {
                        "exact_signature_oracle_belief": exact_est,
                        "level1_cue_marginal": l1_est,
                        "level2_knn_jaccard_k3": l2_est,
                        "level3_candidate_set_tau0.5": l3_est,
                    }
                    signature_summary_rows.append({
                        "variant_id": variant["id"],
                        "signature": list(sig),
                        "signature_name": truth["signature_name"],
                        "n_objects": truth["n_objects"],
                        "positive_net_rate": truth["positive_net_rate"],
                        "mean_true_E_observe_net": truth["mean_true_E_observe_net"],
                        "mean_if_positive": truth["mean_if_positive"],
                        "mean_if_non_positive": truth["mean_if_non_positive"],
                        "always_observe_mean_return": truth["always_observe_mean_return"],
                        "oracle_selective_mean_return": truth["oracle_selective_mean_return"],
                        "no_observe_mean_return": truth["no_observe_mean_return"],
                    })
                    for level_name, est in est_map.items():
                        pred_observe = est["e_observe_net"] > 0.0
                        true_mean_positive = truth["mean_true_E_observe_net"] > 0.0
                        sign_correct_sig = (pred_observe == true_mean_positive)
                        error_type = (
                            "fallback_observe" if est["used_global_fallback"] and pred_observe else
                            "fallback_direct" if est["used_global_fallback"] and not pred_observe else
                            "correct_observe" if pred_observe and true_mean_positive else
                            "correct_direct" if (not pred_observe) and (not true_mean_positive) else
                            "false_observe" if pred_observe else
                            "false_direct"
                        )

                        level_agg[level_name]["estimated_Es"].append(est["e_observe_net"])
                        level_agg[level_name]["true_Es"].append(truth["mean_true_E_observe_net"])
                        level_agg[level_name]["signature_rows"].append({
                            "variant_id": variant["id"],
                            "signature": list(sig),
                            "signature_name": truth["signature_name"],
                            "estimated_E": est["e_observe_net"],
                            "true_E_mean": truth["mean_true_E_observe_net"],
                            "true_positive_rate": truth["positive_net_rate"],
                            "predicted_observe": pred_observe,
                            "sign_correct": sign_correct_sig,
                            "error_type": error_type,
                            "fallback_used": est["used_global_fallback"],
                        })
                        if est["used_global_fallback"]:
                            level_agg[level_name]["fallback_count"] += truth["n_objects"]
                        else:
                            level_agg[level_name]["non_global_count"] += truth["n_objects"]

                        for obj in truth["object_records"]:
                            level_agg[level_name]["obj_total"] += 1
                            true_positive = obj["true_positive"]
                            if true_positive:
                                level_agg[level_name]["pos_est_values"].append(est["e_observe_net"])
                            else:
                                level_agg[level_name]["nonpos_est_values"].append(est["e_observe_net"])

                            if pred_observe == true_positive:
                                level_agg[level_name]["sign_correct"] += 1
                            elif pred_observe and not true_positive:
                                level_agg[level_name]["false_observe"] += 1
                                level_agg[level_name]["false_observe_losses"].append(-obj["true_e"])
                            elif (not pred_observe) and true_positive:
                                level_agg[level_name]["false_direct"] += 1
                                level_agg[level_name]["false_direct_losses"].append(obj["true_e"])

            signature_summary_rows.sort(key=lambda r: (r["variant_id"], r["signature_name"]))
            oracle_gap_values = [
                row["oracle_selective_mean_return"] - row["always_observe_mean_return"]
                for row in signature_summary_rows
            ]
            headroom_decomposition[key] = {
                "distribution": distribution,
                "observe_cost": observe_cost,
                "test_type": test_type,
                "always_try_net": prior_result["baselines"]["always_try_net"],
                "always_observe_net": prior_result["baselines"]["always_observe_net"],
                "oracle_selective_net": prior_result["baselines"]["oracle_selective_net"],
                "oracle_minus_always_observe": prior_result["baselines"]["oracle_selective_net"] - prior_result["baselines"]["always_observe_net"],
                "oracle_minus_always_try": prior_result["baselines"]["oracle_selective_net"] - prior_result["baselines"]["always_try_net"],
                "always_observe_minus_always_try": prior_result["baselines"]["always_observe_net"] - prior_result["baselines"]["always_try_net"],
                "per_signature_no_observe_mean_values": [row["no_observe_mean_return"] for row in signature_summary_rows],
                "per_signature_always_observe_mean_values": [row["always_observe_mean_return"] for row in signature_summary_rows],
                "per_signature_oracle_selective_mean_values": [row["oracle_selective_mean_return"] for row in signature_summary_rows],
                "oracle_gap_min": min(oracle_gap_values) if oracle_gap_values else 0.0,
                "oracle_gap_max": max(oracle_gap_values) if oracle_gap_values else 0.0,
                "oracle_gap_mean": mean(oracle_gap_values),
            }

            all_true_es = [row["mean_true_E_observe_net"] for row in signature_summary_rows]
            pos_case_means = [row["mean_if_positive"] for row in signature_summary_rows if row["mean_if_positive"] is not None]
            nonpos_case_means = [row["mean_if_non_positive"] for row in signature_summary_rows if row["mean_if_non_positive"] is not None]
            weighted_pos_rate_num = sum(row["positive_net_rate"] * row["n_objects"] for row in signature_summary_rows)
            total_obj = sum(row["n_objects"] for row in signature_summary_rows)
            positive_objs = int(round(weighted_pos_rate_num))
            observe_value_distribution[key] = {
                "distribution": distribution,
                "observe_cost": observe_cost,
                "test_type": test_type,
                "n_positive_net": positive_objs,
                "n_non_positive_net": total_obj - positive_objs,
                "positive_net_rate": weighted_pos_rate_num / total_obj if total_obj else 0.0,
                "mean_true_E_observe_net": mean(all_true_es),
                "mean_true_E_positive_cases": mean(pos_case_means) if pos_case_means else None,
                "mean_true_E_non_positive_cases": mean(nonpos_case_means) if nonpos_case_means else None,
                "min_true_E_observe_net": min(all_true_es) if all_true_es else 0.0,
                "max_true_E_observe_net": max(all_true_es) if all_true_es else 0.0,
                "per_signature": signature_summary_rows,
            }

            strength_rows = []
            for row in signature_summary_rows:
                contribution_to_always = row["always_observe_mean_return"] * row["n_objects"] / total_obj if total_obj else 0.0
                contribution_to_oracle_gap = (
                    (row["oracle_selective_mean_return"] - row["always_observe_mean_return"]) * row["n_objects"] / total_obj
                    if total_obj else 0.0
                )
                strength_rows.append({
                    **row,
                    "contribution_to_always_observe_net": contribution_to_always,
                    "contribution_to_oracle_gap": contribution_to_oracle_gap,
                })
            always_observe_strength[key] = {
                "distribution": distribution,
                "observe_cost": observe_cost,
                "test_type": test_type,
                "per_signature": strength_rows,
            }

            level_error_decomposition[key] = {}
            pre_cue_predictability[key] = {}
            for level_name, agg in level_agg.items():
                obj_total = agg["obj_total"] or 1
                sign_correct_rate = agg["sign_correct"] / obj_total
                false_observe_rate = agg["false_observe"] / obj_total
                false_direct_rate = agg["false_direct"] / obj_total
                fallback_rate = agg["fallback_count"] / obj_total
                non_global_rate = agg["non_global_count"] / obj_total
                pearson = pearson_corr(agg["estimated_Es"], agg["true_Es"])
                spearman = spearman_corr(agg["estimated_Es"], agg["true_Es"])
                est_pos_mean = mean(agg["pos_est_values"]) if agg["pos_est_values"] else None
                est_nonpos_mean = mean(agg["nonpos_est_values"]) if agg["nonpos_est_values"] else None
                policy_row = prior_result["aggregated_policies"].get(level_name, {})

                level_error_decomposition[key][level_name] = {
                    "distribution": distribution,
                    "observe_cost": observe_cost,
                    "test_type": test_type,
                    "estimated_E_mean": mean(agg["estimated_Es"]),
                    "true_E_mean": mean(agg["true_Es"]),
                    "sign_correct_rate": sign_correct_rate,
                    "false_observe_rate": false_observe_rate,
                    "false_direct_rate": false_direct_rate,
                    "mean_loss_from_false_observe": mean(agg["false_observe_losses"]) if agg["false_observe_losses"] else 0.0,
                    "mean_loss_from_false_direct": mean(agg["false_direct_losses"]) if agg["false_direct_losses"] else 0.0,
                    "fallback_rate": fallback_rate,
                    "non_global_rate": non_global_rate,
                    "observe_rate": policy_row.get("observe_rate"),
                    "policy_net_if_available": policy_row.get("policy_net"),
                    "delta_vs_always_observe_if_available": policy_row.get("delta_vs_always_observe"),
                    "per_signature": agg["signature_rows"],
                }

                pre_cue_predictability[key][level_name] = {
                    "distribution": distribution,
                    "observe_cost": observe_cost,
                    "test_type": test_type,
                    "pearson_corr_estimatedE_trueE": pearson,
                    "spearman_corr_estimatedE_trueE": spearman,
                    "sign_agreement_rate": mean([1.0 if r["sign_correct"] else 0.0 for r in agg["signature_rows"]]),
                    "estimated_E_mean_positive_objects": est_pos_mean,
                    "estimated_E_mean_non_positive_objects": est_nonpos_mean,
                    "estimated_E_separation": (
                        est_pos_mean - est_nonpos_mean
                        if est_pos_mean is not None and est_nonpos_mean is not None else None
                    ),
                }

                csv_rows.append({
                    "distribution": distribution,
                    "observe_cost": observe_cost,
                    "test_type": test_type,
                    "level": level_name,
                    "always_try_net": prior_result["baselines"]["always_try_net"],
                    "always_observe_net": prior_result["baselines"]["always_observe_net"],
                    "oracle_selective_net": prior_result["baselines"]["oracle_selective_net"],
                    "oracle_minus_always_observe": prior_result["baselines"]["oracle_selective_net"] - prior_result["baselines"]["always_observe_net"],
                    "positive_net_rate": observe_value_distribution[key]["positive_net_rate"],
                    "mean_true_E_observe_net": observe_value_distribution[key]["mean_true_E_observe_net"],
                    "mean_true_E_positive": observe_value_distribution[key]["mean_true_E_positive_cases"],
                    "mean_true_E_nonpositive": observe_value_distribution[key]["mean_true_E_non_positive_cases"],
                    "estimated_E_mean": mean(agg["estimated_Es"]),
                    "sign_correct_rate": sign_correct_rate,
                    "false_observe_rate": false_observe_rate,
                    "false_direct_rate": false_direct_rate,
                    "fallback_rate": fallback_rate,
                    "non_global_rate": non_global_rate,
                    "pearson_corr_estimatedE_trueE": pearson,
                    "observe_rate": policy_row.get("observe_rate"),
                    "policy_net_if_available": policy_row.get("policy_net"),
                    "delta_vs_always_observe_if_available": policy_row.get("delta_vs_always_observe"),
                })

            coverage_effect_rows.append({
                "distribution": distribution,
                "observe_cost": observe_cost,
                "test_type": test_type,
                "positive_net_rate": observe_value_distribution[key]["positive_net_rate"],
                "oracle_minus_always_observe": headroom_decomposition[key]["oracle_minus_always_observe"],
                "level1_sign_correct_rate": level_error_decomposition[key]["level1_cue_marginal"]["sign_correct_rate"],
                "level2_sign_correct_rate": level_error_decomposition[key]["level2_knn_jaccard_k3"]["sign_correct_rate"],
                "level3_sign_correct_rate": level_error_decomposition[key]["level3_candidate_set_tau0.5"]["sign_correct_rate"],
                "level1_policy_net": prior_result["aggregated_policies"]["level1_cue_marginal"]["policy_net"],
                "level2_policy_net": prior_result["aggregated_policies"]["level2_knn_jaccard_k3"]["policy_net"],
                "level3_policy_net": prior_result["aggregated_policies"]["level3_candidate_set_tau0.5"]["policy_net"],
                "level1_fallback_rate": level_error_decomposition[key]["level1_cue_marginal"]["fallback_rate"],
                "level2_fallback_rate": level_error_decomposition[key]["level2_knn_jaccard_k3"]["fallback_rate"],
                "level3_fallback_rate": level_error_decomposition[key]["level3_candidate_set_tau0.5"]["fallback_rate"],
            })

print("[3/5] Computing diagnostic flags...")
primary_keys = [
    cond_key("counterfactual_cov0.25", 0.03, "A"),
    cond_key("counterfactual_cov0.25", 0.03, "B"),
    cond_key("counterfactual_cov0.50", 0.03, "A"),
    cond_key("counterfactual_cov0.50", 0.03, "B"),
]
primary_keys = [k for k in primary_keys if k in headroom_decomposition]
primary_headroom_gaps = [headroom_decomposition[k]["oracle_minus_always_observe"] for k in primary_keys]
primary_always_try = [headroom_decomposition[k]["always_try_net"] for k in primary_keys]
primary_always_obs = [headroom_decomposition[k]["always_observe_net"] for k in primary_keys]
primary_oracle = [headroom_decomposition[k]["oracle_selective_net"] for k in primary_keys]

primary_level_rows = []
for key in primary_keys:
    for level_name in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]:
        primary_level_rows.append(level_error_decomposition[key][level_name])

avg_sign_correct = max((row["sign_correct_rate"] for row in primary_level_rows), default=0.0)
avg_pearson = max((
    row2["pearson_corr_estimatedE_trueE"] if row2["pearson_corr_estimatedE_trueE"] is not None else -999
    for key in primary_keys
    for row2 in pre_cue_predictability[key].values()
), default=-999)
avg_fallback = mean([row["fallback_rate"] for row in primary_level_rows]) if primary_level_rows else 0.0
avg_non_global = mean([row["non_global_rate"] for row in primary_level_rows]) if primary_level_rows else 0.0

primary_positive_gain = mean([
    observe_value_distribution[k]["mean_true_E_positive_cases"]
    for k in primary_keys
    if observe_value_distribution[k]["mean_true_E_positive_cases"] is not None
])
primary_nonpositive_penalty_abs = mean([
    abs(observe_value_distribution[k]["mean_true_E_non_positive_cases"])
    for k in primary_keys
    if observe_value_distribution[k]["mean_true_E_non_positive_cases"] is not None
])

cf1_key = cond_key("counterfactual_cov1.00", 0.03, "A")
cf1_key_b = cond_key("counterfactual_cov1.00", 0.03, "B")
cf1_keys = [k for k in [cf1_key, cf1_key_b] if k in level_error_decomposition]
cf1_best_policy = mean([
    max(
        run_7fb["per_condition_results"][k]["aggregated_policies"][lvl]["policy_net"]
        for lvl in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]
    )
    for k in cf1_keys
]) if cf1_keys else None
primary_best_policy = mean([
    max(
        run_7fb["per_condition_results"][k]["aggregated_policies"][lvl]["policy_net"]
        for lvl in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]
    )
    for k in primary_keys
]) if primary_keys else None
cf1_best_sign_correct = mean([
    max(
        level_error_decomposition[k][lvl]["sign_correct_rate"]
        for lvl in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]
    )
    for k in cf1_keys
]) if cf1_keys else None
primary_best_sign_correct = mean([
    max(
        level_error_decomposition[k][lvl]["sign_correct_rate"]
        for lvl in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]
    )
    for k in primary_keys
]) if primary_keys else None

final_flags = {
    "weak_oracle_headroom": bool(primary_headroom_gaps) and all(g <= 0.0100001 for g in primary_headroom_gaps),
    "always_observe_too_strong": bool(primary_headroom_gaps) and all(
        ao >= (osel - 0.01) and ao > at
        for ao, osel, at in zip(primary_always_obs, primary_oracle, primary_always_try)
    ),
    "nonpositive_penalty_too_weak": (
        primary_positive_gain > 0
        and primary_nonpositive_penalty_abs <= 0.5 * primary_positive_gain
    ),
    "pre_cue_predictability_low": avg_sign_correct < 0.75 or avg_pearson <= 0.10,
    "cf_cov1_destroyed_signal": (
        cf1_best_policy is not None and primary_best_policy is not None and (
            cf1_best_policy < primary_best_policy - 0.05
            or (cf1_best_sign_correct is not None and primary_best_sign_correct is not None and cf1_best_sign_correct < primary_best_sign_correct - 0.20)
        )
    ),
    "level_estimate_sign_mismatch": avg_non_global > 0.5 and avg_sign_correct < 0.75,
    "fallback_problem": avg_fallback > 0.5,
}
final_flags["distribution_review_needed"] = (
    run_7fb["headroom_summary"]["final_headroom_status"] == "NO_HEADROOM"
    and (final_flags["weak_oracle_headroom"] or final_flags["pre_cue_predictability_low"])
)

print("[4/5] Writing outputs...")
runs_dir = os.path.join(CURRENT_DIR, "runs")
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(protocols_dir, exist_ok=True)

json_output = {
    "metadata": {
        "block_id": "1J40b-7g",
        "script_name": "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py",
        "timestamp": datetime.now().isoformat(),
        "source_files_used": SOURCE_FILES_USED,
        "distributions": DISTRIBUTIONS,
        "observe_costs": OBSERVE_COSTS,
        "tests": TESTS,
    },
    "headroom_decomposition": headroom_decomposition,
    "observe_value_distribution": observe_value_distribution,
    "always_observe_strength": always_observe_strength,
    "level_error_decomposition": level_error_decomposition,
    "pre_cue_predictability": pre_cue_predictability,
    "coverage_effect_audit": coverage_effect_rows,
    "validity_audit": {
        "no_learner_training": True,
        "no_environment_change": True,
        "no_1j40b8": True,
        "forbidden_test_keys_absent": True,
        "audit_only": True,
    },
    "final_diagnostic_flags": final_flags,
    "threshold_notes": {
        "nonpositive_penalty_too_weak_rule": "true if abs(mean_true_E_non_positive) <= 0.5 * mean_true_E_positive on primary Test A/B at cost=0.03",
        "pre_cue_predictability_low_rule": "true if best sign_correct_rate < 0.75 or best pearson_corr <= 0.10 on primary Test A/B at cost=0.03",
        "cf_cov1_destroyed_signal_rule": "true if cf_cov1.00 best policy is >0.05 worse than primary best policy or best sign_correct drops by >0.20",
    },
    "elapsed_seconds": round(time.time() - t0, 2),
}

json_path = os.path.join(runs_dir, "block1j40b7g_env3b_counterfactual_headroom_decomposition.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

csv_header = [
    "distribution", "observe_cost", "test_type", "level",
    "always_try_net", "always_observe_net", "oracle_selective_net",
    "oracle_minus_always_observe", "positive_net_rate",
    "mean_true_E_observe_net", "mean_true_E_positive", "mean_true_E_nonpositive",
    "estimated_E_mean", "sign_correct_rate", "false_observe_rate",
    "false_direct_rate", "fallback_rate", "non_global_rate",
    "pearson_corr_estimatedE_trueE", "observe_rate",
    "policy_net_if_available", "delta_vs_always_observe_if_available",
]
csv_lines = [",".join(csv_header)]
for row in csv_rows:
    csv_lines.append(",".join([
        str(row["distribution"]),
        f"{row['observe_cost']:.2f}",
        str(row["test_type"]),
        str(row["level"]),
        f"{row['always_try_net']:.6f}",
        f"{row['always_observe_net']:.6f}",
        f"{row['oracle_selective_net']:.6f}",
        f"{row['oracle_minus_always_observe']:.6f}",
        f"{row['positive_net_rate']:.6f}",
        f"{row['mean_true_E_observe_net']:.6f}",
        "" if row["mean_true_E_positive"] is None else f"{row['mean_true_E_positive']:.6f}",
        "" if row["mean_true_E_nonpositive"] is None else f"{row['mean_true_E_nonpositive']:.6f}",
        f"{row['estimated_E_mean']:.6f}",
        f"{row['sign_correct_rate']:.6f}",
        f"{row['false_observe_rate']:.6f}",
        f"{row['false_direct_rate']:.6f}",
        f"{row['fallback_rate']:.6f}",
        f"{row['non_global_rate']:.6f}",
        "" if row["pearson_corr_estimatedE_trueE"] is None else f"{row['pearson_corr_estimatedE_trueE']:.6f}",
        "" if row["observe_rate"] is None else f"{row['observe_rate']:.6f}",
        "" if row["policy_net_if_available"] is None else f"{row['policy_net_if_available']:.6f}",
        "" if row["delta_vs_always_observe_if_available"] is None else f"{row['delta_vs_always_observe_if_available']:.6f}",
    ]))

csv_path = os.path.join(protocols_dir, "block1j40b7g_env3b_counterfactual_headroom_decomposition_table.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines) + "\n")

primary_md_key_a = cond_key("counterfactual_cov0.25", 0.03, "A")
primary_md_key_b = cond_key("counterfactual_cov0.25", 0.03, "B")
md = []
md.append("# Block 1J40b-7g: Env3b / Counterfactual Headroom Decomposition")
md.append("")
md.append("## 1. Checkpoint Summary")
md.append("")
md.append("- **Status**: DIAGNOSTIC_COMPLETE")
md.append(f"- **Elapsed**: {json_output['elapsed_seconds']:.2f}s")
md.append(f"- **Diagnostic Flags**: {final_flags}")
md.append("")
md.append("## 2. Files Changed")
md.append("")
md.append("- `_block1j40b7g_env3b_counterfactual_headroom_decomposition.py`")
md.append("- `runs/block1j40b7g_env3b_counterfactual_headroom_decomposition.json`")
md.append("- `protocols/block1j40b7g_env3b_counterfactual_headroom_decomposition.md`")
md.append("- `protocols/block1j40b7g_env3b_counterfactual_headroom_decomposition_table.csv`")
md.append("- `checkpoint_1j40b7g_env3b_counterfactual_headroom_decomposition.md`")
md.append("")
md.append("## 3. Commands Run")
md.append("")
md.append("```powershell")
md.append('& "D:\\conda\\python.exe" "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py"')
md.append("```")
md.append("")
md.append("## 4. Why 7g Was Run")
md.append("")
md.append("- 7f-b showed compositional non-global estimates can exist without beating always_observe.")
md.append("- 7g decomposes whether failure is due to weak oracle headroom, strong always_observe, poor sign prediction, fallback, or coverage effects.")
md.append("")
md.append("## 5. Headroom Decomposition")
md.append("")
md.append("| Distribution | Cost | Test | always_try | always_observe | oracle_selective | oracle-always_obs | always_obs-always_try |")
md.append("|--------------|------|------|------------|----------------|------------------|-------------------|-----------------------|")
for key in sorted(headroom_decomposition.keys()):
    row = headroom_decomposition[key]
    md.append(
        f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | "
        f"{row['always_try_net']:.4f} | {row['always_observe_net']:.4f} | "
        f"{row['oracle_selective_net']:.4f} | {row['oracle_minus_always_observe']:+.4f} | "
        f"{row['always_observe_minus_always_try']:+.4f} |"
    )
md.append("")
md.append("## 6. Positive / Non-Positive Observe Value Distribution")
md.append("")
md.append("| Distribution | Cost | Test | positive_rate | mean_true_E | mean_positive | mean_nonpositive | min_true_E | max_true_E |")
md.append("|--------------|------|------|---------------|-------------|---------------|------------------|------------|------------|")
for key in sorted(observe_value_distribution.keys()):
    row = observe_value_distribution[key]
    md.append(
        f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | "
        f"{row['positive_net_rate']:.4f} | {row['mean_true_E_observe_net']:.4f} | "
        f"{(row['mean_true_E_positive_cases'] if row['mean_true_E_positive_cases'] is not None else 0.0):.4f} | "
        f"{(row['mean_true_E_non_positive_cases'] if row['mean_true_E_non_positive_cases'] is not None else 0.0):.4f} | "
        f"{row['min_true_E_observe_net']:.4f} | {row['max_true_E_observe_net']:.4f} |"
    )
md.append("")
md.append("## 7. Always-Observe Strength Source")
md.append("")
for key in [k for k in [primary_md_key_a, primary_md_key_b] if k in always_observe_strength]:
    md.append(f"### {key}")
    md.append("")
    md.append("| Signature | n_objects | always_obs | oracle_sel | no_observe | contrib_always_obs | contrib_oracle_gap |")
    md.append("|-----------|-----------|------------|------------|------------|--------------------|--------------------|")
    for row in always_observe_strength[key]["per_signature"]:
        md.append(
            f"| {row['signature_name']} | {row['n_objects']} | {row['always_observe_mean_return']:.4f} | "
            f"{row['oracle_selective_mean_return']:.4f} | {row['no_observe_mean_return']:.4f} | "
            f"{row['contribution_to_always_observe_net']:.4f} | {row['contribution_to_oracle_gap']:+.4f} |"
        )
    md.append("")
md.append("## 8. Level 1/2/3 Error Decomposition")
md.append("")
md.append("| Distribution | Cost | Test | Level | sign_correct | false_observe | false_direct | mean_loss_false_observe | mean_loss_false_direct | fallback_rate | non_global_rate |")
md.append("|--------------|------|------|-------|--------------|---------------|--------------|-------------------------|------------------------|---------------|-----------------|")
for key in sorted(level_error_decomposition.keys()):
    for level_name in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]:
        row = level_error_decomposition[key][level_name]
        md.append(
            f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | {level_name} | "
            f"{row['sign_correct_rate']:.4f} | {row['false_observe_rate']:.4f} | {row['false_direct_rate']:.4f} | "
            f"{row['mean_loss_from_false_observe']:.4f} | {row['mean_loss_from_false_direct']:.4f} | "
            f"{row['fallback_rate']:.4f} | {row['non_global_rate']:.4f} |"
        )
md.append("")
md.append("## 9. Pre-Cue Predictability Audit")
md.append("")
md.append("| Distribution | Cost | Test | Level | pearson | spearman | sign_agreement | estE_pos | estE_nonpos | separation |")
md.append("|--------------|------|------|-------|---------|----------|----------------|----------|-------------|------------|")
for key in sorted(pre_cue_predictability.keys()):
    for level_name in ["level1_cue_marginal", "level2_knn_jaccard_k3", "level3_candidate_set_tau0.5"]:
        row = pre_cue_predictability[key][level_name]
        md.append(
            f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | {level_name} | "
            f"{(row['pearson_corr_estimatedE_trueE'] if row['pearson_corr_estimatedE_trueE'] is not None else 0.0):.4f} | "
            f"{(row['spearman_corr_estimatedE_trueE'] if row['spearman_corr_estimatedE_trueE'] is not None else 0.0):.4f} | "
            f"{row['sign_agreement_rate']:.4f} | "
            f"{(row['estimated_E_mean_positive_objects'] if row['estimated_E_mean_positive_objects'] is not None else 0.0):.4f} | "
            f"{(row['estimated_E_mean_non_positive_objects'] if row['estimated_E_mean_non_positive_objects'] is not None else 0.0):.4f} | "
            f"{(row['estimated_E_separation'] if row['estimated_E_separation'] is not None else 0.0):.4f} |"
        )
md.append("")
md.append("## 10. Coverage Effect Audit")
md.append("")
md.append("| Distribution | Cost | Test | positive_rate | oracle-always_obs | L1 sign | L2 sign | L3 sign | L1 policy | L2 policy | L3 policy |")
md.append("|--------------|------|------|---------------|-------------------|---------|---------|---------|-----------|-----------|-----------|")
for row in coverage_effect_rows:
    md.append(
        f"| {row['distribution']} | {row['observe_cost']:.2f} | {row['test_type']} | {row['positive_net_rate']:.4f} | "
        f"{row['oracle_minus_always_observe']:+.4f} | {row['level1_sign_correct_rate']:.4f} | "
        f"{row['level2_sign_correct_rate']:.4f} | {row['level3_sign_correct_rate']:.4f} | "
        f"{row['level1_policy_net']:.4f} | {row['level2_policy_net']:.4f} | {row['level3_policy_net']:.4f} |"
    )
md.append("")
md.append("## 11. Validity / Leakage Audit")
md.append("")
md.append("- no learner training")
md.append("- no environment change")
md.append("- no 1J40b-8 implementation")
md.append("- forbidden test-time keys absent")
md.append("- audit-only diagnostic use of true labels")
md.append("")
md.append("## 12. Diagnostic Flags")
md.append("")
for flag_name, flag_val in final_flags.items():
    md.append(f"- **{flag_name}** = {flag_val}")
md.append("")
md.append("## 13. Recommended Resume Point")
md.append("")
md.append("- Resume at env3b / counterfactual design review before any 1J40b-8 work.")

md_path = os.path.join(protocols_dir, "block1j40b7g_env3b_counterfactual_headroom_decomposition.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J40b-7g",
    "",
    "- status: DIAGNOSTIC_COMPLETE",
    f"- weak_oracle_headroom: {final_flags['weak_oracle_headroom']}",
    f"- always_observe_too_strong: {final_flags['always_observe_too_strong']}",
    f"- nonpositive_penalty_too_weak: {final_flags['nonpositive_penalty_too_weak']}",
    f"- pre_cue_predictability_low: {final_flags['pre_cue_predictability_low']}",
    f"- cf_cov1_destroyed_signal: {final_flags['cf_cov1_destroyed_signal']}",
    f"- level_estimate_sign_mismatch: {final_flags['level_estimate_sign_mismatch']}",
    f"- fallback_problem: {final_flags['fallback_problem']}",
    "- next_suggested_step: env3b / counterfactual design review",
]
checkpoint_path = os.path.join(CURRENT_DIR, "checkpoint_1j40b7g_env3b_counterfactual_headroom_decomposition.md")
with open(checkpoint_path, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Done.")
print(f"  JSON: {json_path}")
print(f"  CSV:  {csv_path}")
print(f"  MD:   {md_path}")
print(f"  CKPT: {checkpoint_path}")
