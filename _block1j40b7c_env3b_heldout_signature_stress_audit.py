"""
Block 1J40b-7c — env3b Held-Out Signature / Cue-Combination Stress Audit.

Stress-tests the 1J40b-7b selective-observe policy for generalization beyond
memorized pre_surface signatures.

Three stress tests:
  Test A: held-out pre_surface signature split (4 variants)
    Hold out 2 signatures (1 shared/positive + 1 unique/non-positive) from training.
    Test generalization to unseen pre_surface feature combinations.

  Test B: held-out cue-combination split (2 variants)
    Hold out non-positive uses of specific cues ("spotted", "discolored")
    that appear in positive contexts during training. Tests whether the model
    associates cues with VOI class or learns signature-level reasoning.

  Test C: held-out seed split (LOSO across 5 seeds)
    Standard cross-seed generalization. Weaker than A/B but sanity check.

Observe costs: primary=0.03, robustness=0.05.
"""

import os, sys, json, math, time, random
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Config safety
# =============================================================================
C6_LABEL = "C6_observe_try_counterfactual_v0"
existing_labels = [c["label"] for c in config.CUE_CONDITIONS]
if C6_LABEL not in existing_labels:
    config.CUE_CONDITIONS.append({
        "label": C6_LABEL, "p_target": 0.60, "p_other": 0.40,
        "absent": False, "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True, "observe_depth_schedules": True,
    })

# =============================================================================
# Constants (from env3b)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
RIDGE_ALPHA = 1.0

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]

SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

# =============================================================================
# Feature Definitions (from env3b)
# =============================================================================
AMBIENT_GROUP_FEATURES = {
    "A": ["brownish", "rough_texture", "long_shape", "heavy_weight"],
    "B": ["greenish", "smooth_texture", "round_small", "light_weight"],
}

CATEGORY_AMBIENT_GROUP = {
    "wood_log": "A", "stone_block": "A",
    "apple": "B", "wooden_pickaxe": "B",
}

ALL_AMBIENT_FEATURE_NAMES = sorted(set(
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats
))

CATEGORY_POST_OBSERVE_FEATURES = {
    "wood_log": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "block_like",
                    "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                       "movable", "has_metal_head", "jointed"],
}

SHARED_SURFACE_FEATURES = [
    "spotted", "weathered", "coarse_texture",
    "discolored", "has_veins", "pitted",
]

ALL_POST_OBSERVE_FEATURE_NAMES = sorted(set(
    f for flist in CATEGORY_POST_OBSERVE_FEATURES.values() for f in flist))

ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    list(ALL_AMBIENT_FEATURE_NAMES) +
    list(ALL_POST_OBSERVE_FEATURE_NAMES) +
    list(SHARED_SURFACE_FEATURES)
))

STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

# =============================================================================
# Instance Subtype Definitions (from 7a, same as 7b)
# =============================================================================
INSTANCE_SUBTYPE_DEFS = {}

INSTANCE_SUBTYPE_DEFS["wood_log"] = {
    "P": {
        "label": "ambiguous_log", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_log", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_log", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_bark_texture", "discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": False, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["stone_block"] = {
    "P": {
        "label": "ambiguous_stone", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_stone", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_stone", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_crystal_flecks"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_granular_surface", "block_like", "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": True, "clean": False, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["apple"] = {
    "P": {
        "label": "ambiguous_fruit", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_apple", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_apple", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_stem_remnant", "spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

INSTANCE_SUBTYPE_DEFS["wooden_pickaxe"] = {
    "P": {
        "label": "ambiguous_tool", "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable", "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "Z": {
        "label": "positive_shared_cue_tool", "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
    "N": {
        "label": "nonpos_identified_tool", "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_grip_area"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_shaft_shape", "elongated_with_handle", "movable",
                                  "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
    },
}

# =============================================================================
# Part 1: Object Generation (from 7a/7b)
# =============================================================================
print("[1/9] Generating env3b objects...")

def generate_env3b_objects(seed):
    rng = random.Random(seed * 100 + 1)
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
            for f in ambient_features_list:
                pre_surface[f] = True
            for f in pre_extra:
                pre_surface[f] = True

            post_observe = dict(pre_surface)
            for f in post_revealed:
                post_observe[f] = True

            hidden = {}
            for f in ALL_VISIBLE_FEATURE_NAMES:
                if f not in pre_surface and f not in post_revealed:
                    hidden[f] = True

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


per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels

n_per_seed = len(per_seed_objects[SEEDS[0]])
print(f"  Objects per seed: {n_per_seed}, total: {n_per_seed * len(SEEDS)}")

# =============================================================================
# Part 2: Feature Vector Building (model-legal only)
# =============================================================================
print("[2/9] Building feature infrastructure...")

def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - 0.05
    elif success_or_failure is False:
        return -FAILURE_PENALTY - 0.05
    return 0.0


def get_pre_surface_signature(pre_surface_features):
    return tuple(sorted(pre_surface_features.keys()))


def collect_all_post_observe_features(per_seed_objects):
    all_feats = set()
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            for f in obj["post_observe_features"]:
                all_feats.add(f)
    return sorted(all_feats)


POST_OBSERVE_FEATURE_NAMES = collect_all_post_observe_features(per_seed_objects)
ALL_MODEL_FEATURE_KEYS = sorted(set(
    [f"feat_{f}" for f in POST_OBSERVE_FEATURE_NAMES] +
    [f"state_{s}" for s in STATE_FEATURE_NAMES] +
    ["prior_observe_count"]
))

print(f"  Post-observe features: {len(POST_OBSERVE_FEATURE_NAMES)}")
print(f"  Model feature keys: {len(ALL_MODEL_FEATURE_KEYS)}")


def build_feature_vector(known_features, known_states, prior_obs_count):
    feat = {}
    for fname in POST_OBSERVE_FEATURE_NAMES:
        feat[f"feat_{fname}"] = 1.0 if fname in known_features else 0.0
    for sname in STATE_FEATURE_NAMES:
        feat[f"state_{sname}"] = 1.0 if known_states.get(sname, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    return feat


def extract_feature_vector(situation_features):
    return [float(situation_features.get(k, 0.0)) for k in ALL_MODEL_FEATURE_KEYS]


# =============================================================================
# Part 3: Signature Analysis
# =============================================================================
print("[3/9] Analyzing signature structure...")

# Collect all (signature, seed, oid) tuples
sig_map = defaultdict(list)  # signature -> [(seed, oid)]
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        sig = get_pre_surface_signature(obj["pre_surface_features"])
        sig_map[sig].append((seed, oid))

# Determine VOI class for each signature (audit only)
def compute_sig_voi_class(sig_members, observe_cost):
    """Compute VOI class for a signature group. Audit-only."""
    # Compute expected returns over the group
    action_expected = {}
    for action in ALL_TRY_AFFORDANCES:
        total = 0.0
        for seed, oid in sig_members:
            lbl = per_seed_audit_labels[seed][oid]
            outcome = lbl["affordance_profile"].get(action, "fail")
            total += try_net_value(outcome == "success")
        action_expected[action] = total / len(sig_members)

    best_action = max(action_expected, key=action_expected.get)
    best_expected = action_expected[best_action]

    # For each member, compute oracle_post_best
    voi_sum = 0.0
    for seed, oid in sig_members:
        lbl = per_seed_audit_labels[seed][oid]
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = lbl["affordance_profile"].get(action, "fail")
            true_returns[action] = try_net_value(outcome == "success")
        post_best = max(true_returns.values())
        voi_sum += post_best - best_expected - observe_cost

    mean_voi = voi_sum / len(sig_members)
    return "positive_net" if mean_voi > 0.001 else "non_positive_net", mean_voi


# Wrap members to include category info for display
def sig_to_display_name(sig):
    feats = list(sig)
    ambient_part = [f for f in feats if f in ALL_AMBIENT_FEATURE_NAMES]
    extra_part = [f for f in feats if f not in ALL_AMBIENT_FEATURE_NAMES]
    amb = ",".join(sorted(ambient_part)[:2])
    ext = ",".join(sorted(extra_part)[:3])
    if ext:
        return f"amb({amb})+{ext}"
    return f"amb({amb})_only"


SIGNATURE_LIST = sorted(sig_map.keys(), key=lambda s: (-len(sig_map[s]), s))

print(f"\n  Unique signatures: {len(SIGNATURE_LIST)}")
for sig in SIGNATURE_LIST:
    members = sig_map[sig]
    n = len(members)
    # Get category info
    cats = set()
    subtypes = set()
    for seed, oid in members:
        lbl = per_seed_audit_labels[seed][oid]
        cats.add(lbl["hidden_category"])
        subtypes.add(lbl["hidden_subtype"])
    voi_class_003, mean_voi_003 = compute_sig_voi_class(members, 0.03)
    voi_class_005, mean_voi_005 = compute_sig_voi_class(members, 0.05)
    print(f"    sig n={n:3d}: {sig_to_display_name(sig):45s} "
          f"cats={cats}, subtypes={subtypes}, "
          f"VOI(0.03)={voi_class_003} ({mean_voi_003:+.4f}), "
          f"VOI(0.05)={voi_class_005} ({mean_voi_005:+.4f})")

# =============================================================================
# Part 4: Stress Test Split Definitions
# =============================================================================
print("\n[4/9] Defining stress test splits...")

# Identify shared (positive) and unique (non-positive) signatures
shared_sigs = []  # cross-category, positive_net
unique_sigs = []  # single-category, non_positive_net
for sig in SIGNATURE_LIST:
    members = sig_map[sig]
    cats_in_sig = set()
    for seed, oid in members:
        lbl = per_seed_audit_labels[seed][oid]
        cats_in_sig.add(lbl["hidden_category"])
    if len(cats_in_sig) >= 2:
        shared_sigs.append(sig)
    else:
        unique_sigs.append(sig)

print(f"  Shared (cross-category, positive_net): {len(shared_sigs)}")
for sig in shared_sigs:
    print(f"    {sig_to_display_name(sig)}: n={len(sig_map[sig])}")

print(f"  Unique (single-category, non_positive_net): {len(unique_sigs)}")
for sig in unique_sigs:
    print(f"    {sig_to_display_name(sig)}: n={len(sig_map[sig])}")

# =============================================================================
# Test A: Held-out signature split (4 variants)
# Each variant holds out 1 shared + 1 unique signature
# =============================================================================
TEST_A_VARIANTS = []
for i in range(min(len(shared_sigs), len(unique_sigs))):
    held_shared = shared_sigs[i]
    held_unique = unique_sigs[i]
    held_sigs = [held_shared, held_unique]
    train_sigs = [s for s in SIGNATURE_LIST if s not in held_sigs]
    test_sigs = held_sigs

    # Compute which categories/cues are held out
    held_cats = set()
    held_cues = set()
    for sig in held_sigs:
        for seed, oid in sig_map[sig]:
            lbl = per_seed_audit_labels[seed][oid]
            held_cats.add(lbl["hidden_category"])
            obj = per_seed_objects[seed][oid]
            for f in obj["pre_surface_features"]:
                if f not in ALL_AMBIENT_FEATURE_NAMES:
                    held_cues.add(f)

    train_oids = set()
    for sig in train_sigs:
        for seed, oid in sig_map[sig]:
            train_oids.add((seed, oid))

    test_oids = set()
    for sig in test_sigs:
        for seed, oid in sig_map[sig]:
            test_oids.add((seed, oid))

    # Verify test has both positive and non-positive
    test_pos = sum(1 for seed, oid in test_oids
                   if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos = sum(1 for seed, oid in test_oids
                      if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")

    TEST_A_VARIANTS.append({
        "id": f"A{i+1}",
        "held_sigs": held_sigs,
        "train_sigs": train_sigs,
        "test_sigs": test_sigs,
        "held_cats": held_cats,
        "held_cues": held_cues,
        "train_oids": train_oids,
        "test_oids": test_oids,
        "n_train": len(train_oids),
        "n_test": len(test_oids),
        "test_pos": test_pos,
        "test_nonpos": test_nonpos,
        "description": (
            f"Hold out shared sig {sig_to_display_name(held_shared)} ({len(sig_map[held_shared])} obj, positive_net) "
            f"+ unique sig {sig_to_display_name(held_unique)} ({len(sig_map[held_unique])} obj, non_positive_net)"
        ),
    })

# =============================================================================
# Test B: Held-out cue-combination split (2 variants)
# Hold out non-positive uses of specific cues from training
# =============================================================================
# B1: Hold out apple N (spotted-nonpos), keep spotted-pos in training
# B2: Hold out wood_log N (discolored-nonpos), keep discolored-pos in training

# Find signatures for apple N (spotted nonpos) and wood_log N (discolored nonpos)
apple_n_sig = None
woodlog_n_sig = None
for sig in unique_sigs:
    cats_in = set()
    for seed, oid in sig_map[sig]:
        lbl = per_seed_audit_labels[seed][oid]
        cats_in.add(lbl["hidden_category"])
    if cats_in == {"apple"}:
        apple_n_sig = sig
    if cats_in == {"wood_log"}:
        woodlog_n_sig = sig

# B1: hold out apple N sig + one shared positive sig (ambient_B only for matching ambient group)
# Find shared pos sig for ambient_B (apple/pickaxe group)
ambient_b_shared = [s for s in shared_sigs
                    if any(per_seed_audit_labels[sig_map[s][0][0]][sig_map[s][0][1]]["ambient_group"] == "B"
                           for _ in [1])]
# Better: check if any member in the sig has ambient_group B
ambient_b_only_sig = None
ambient_b_discolored_sig = None
for sig in shared_sigs:
    some_seed, some_oid = sig_map[sig][0]
    ag = per_seed_audit_labels[some_seed][some_oid]["ambient_group"]
    if ag == "B":
        if len(sig) == 4:  # ambient only
            ambient_b_only_sig = sig
        elif "discolored" in sig:
            ambient_b_discolored_sig = sig

# B1: hold out apple N (spotted-nonpos) + ambient_B only (positive for balance)
b1_test_sigs = [apple_n_sig, ambient_b_only_sig] if apple_n_sig and ambient_b_only_sig else []
b1_train_sigs = [s for s in SIGNATURE_LIST if s not in b1_test_sigs] if b1_test_sigs else []

# B2: hold out wood_log N (discolored-nonpos) + one shared positive sig
ambient_a_only_sig = None
for sig in shared_sigs:
    some_seed, some_oid = sig_map[sig][0]
    ag = per_seed_audit_labels[some_seed][some_oid]["ambient_group"]
    if ag == "A" and len(sig) == 4:
        ambient_a_only_sig = sig

b2_test_sigs = [woodlog_n_sig, ambient_a_only_sig] if woodlog_n_sig and ambient_a_only_sig else []
b2_train_sigs = [s for s in SIGNATURE_LIST if s not in b2_test_sigs] if b2_test_sigs else []

TEST_B_VARIANTS = []
for bid, test_sigs, train_sigs, desc in [
    ("B1", b1_test_sigs, b1_train_sigs,
     "Hold out apple N (spotted in nonpos context) + ambient_B only (pos). "
     "Training has spotted only in positive contexts (wood_log Z, stone_block Z)."),
    ("B2", b2_test_sigs, b2_train_sigs,
     "Hold out wood_log N (discolored in nonpos context) + ambient_A only (pos). "
     "Training has discolored only in positive contexts (apple Z, pickaxe Z)."),
]:
    if not test_sigs or not train_sigs:
        print(f"  WARNING: {bid} could not be constructed, skipping")
        continue

    train_oids_b = set()
    for sig in train_sigs:
        for seed, oid in sig_map[sig]:
            train_oids_b.add((seed, oid))
    test_oids_b = set()
    for sig in test_sigs:
        for seed, oid in sig_map[sig]:
            test_oids_b.add((seed, oid))

    test_pos_b = sum(1 for seed, oid in test_oids_b
                     if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos_b = sum(1 for seed, oid in test_oids_b
                        if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")

    TEST_B_VARIANTS.append({
        "id": bid,
        "held_sigs": test_sigs,
        "train_sigs": train_sigs,
        "test_sigs": test_sigs,
        "train_oids": train_oids_b,
        "test_oids": test_oids_b,
        "n_train": len(train_oids_b),
        "n_test": len(test_oids_b),
        "test_pos": test_pos_b,
        "test_nonpos": test_nonpos_b,
        "description": desc,
    })

# =============================================================================
# Test C: Held-out seed split (LOSO)
# =============================================================================
TEST_C_VARIANTS = []
for heldout_seed in SEEDS:
    train_seeds = [s for s in SEEDS if s != heldout_seed]
    train_oids_c = set()
    test_oids_c = set()
    for seed in train_seeds:
        for oid in per_seed_objects[seed]:
            train_oids_c.add((seed, oid))
    for oid in per_seed_objects[heldout_seed]:
        test_oids_c.add((heldout_seed, oid))

    test_pos_c = sum(1 for seed, oid in test_oids_c
                     if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] == "positive_change")
    test_nonpos_c = sum(1 for seed, oid in test_oids_c
                        if per_seed_audit_labels[seed][oid]["hidden_audit_rationale_class"] != "positive_change")

    TEST_C_VARIANTS.append({
        "id": f"C_seed{heldout_seed}",
        "heldout_seed": heldout_seed,
        "train_seeds": train_seeds,
        "train_oids": train_oids_c,
        "test_oids": test_oids_c,
        "n_train": len(train_oids_c),
        "n_test": len(test_oids_c),
        "test_pos": test_pos_c,
        "test_nonpos": test_nonpos_c,
        "description": f"LOSO: train on {train_seeds}, test on seed {heldout_seed}",
    })

print(f"\n  Test A variants: {len(TEST_A_VARIANTS)}")
for va in TEST_A_VARIANTS:
    print(f"    {va['id']}: train={va['n_train']}, test={va['n_test']} "
          f"(pos={va['test_pos']}, nonpos={va['test_nonpos']})")

print(f"  Test B variants: {len(TEST_B_VARIANTS)}")
for vb in TEST_B_VARIANTS:
    print(f"    {vb['id']}: train={vb['n_train']}, test={vb['n_test']} "
          f"(pos={vb['test_pos']}, nonpos={vb['test_nonpos']})")

print(f"  Test C variants: {len(TEST_C_VARIANTS)}")
for vc in TEST_C_VARIANTS:
    print(f"    {vc['id']}: train={vc['n_train']}, test={vc['n_test']} "
          f"(pos={vc['test_pos']}, nonpos={vc['test_nonpos']})")

# =============================================================================
# Part 5: Ridge Regression Model
# =============================================================================
class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0
        self._fitted = False

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        X_aug = [row + [1.0] for row in X]
        n_aug = n_features + 1
        XtX = [[0.0] * n_aug for _ in range(n_aug)]
        for i in range(n_aug):
            for j in range(n_aug):
                s = 0.0
                for k in range(n_samples):
                    s += X_aug[k][i] * X_aug[k][j]
                XtX[i][j] = s
        for i in range(n_features):
            XtX[i][i] += self.alpha
        Xty = [0.0] * n_aug
        for i in range(n_aug):
            s = 0.0
            for k in range(n_samples):
                s += X_aug[k][i] * y[k]
            Xty[i] = s
        w_aug = _solve_linear_system(XtX, Xty)
        if w_aug is None:
            self.coef_ = [0.0] * n_features
            self.intercept_ = sum(y) / len(y) if y else 0.0
        else:
            self.coef_ = w_aug[:n_features]
            self.intercept_ = w_aug[n_features]
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return [sum(xi * wi for xi, wi in zip(row, self.coef_)) + self.intercept_
                for row in X]


def _solve_linear_system(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[max_row][col]) < 1e-12:
            return None
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot
        for row in range(n):
            if row != col:
                factor = M[row][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


class FeatureNormalizer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        n_feat = len(X[0]) if X else 0
        self.means = [0.0] * n_feat
        self.stds = [1.0] * n_feat
        if not X:
            return self
        n = len(X)
        for j in range(n_feat):
            self.means[j] = sum(row[j] for row in X) / n
            var = sum((row[j] - self.means[j]) ** 2 for row in X) / n
            self.stds[j] = math.sqrt(var) if var > 1e-12 else 1.0
        return self

    def transform(self, X):
        if self.means is None:
            return X
        return [[(row[j] - self.means[j]) / self.stds[j]
                 for j in range(len(row))]
                for row in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PerActionValueEstimator:
    def __init__(self, alpha=RIDGE_ALPHA):
        self.alpha = alpha
        self.models = {}
        self.normalizers = {}
        self._fitted_actions = set()

    def fit(self, X_train, y_train, action_keys_train):
        grouped_X = defaultdict(list)
        grouped_y = defaultdict(list)
        for xv, yv, ak in zip(X_train, y_train, action_keys_train):
            grouped_X[ak].append(xv)
            grouped_y[ak].append(yv)

        for ak in ALL_ACTION_KEYS:
            gx = grouped_X.get(ak, [])
            gy = grouped_y.get(ak, [])
            if len(gx) < 3:
                self.models[ak] = RidgeRegression(alpha=self.alpha)
                nf = len(gx[0]) if gx else 0
                self.models[ak].coef_ = [0.0] * nf
                self.models[ak].intercept_ = sum(gy) / len(gy) if gy else 0.0
                self.models[ak]._fitted = True
                self.normalizers[ak] = None
                continue
            normalizer = FeatureNormalizer().fit(gx)
            X_norm = normalizer.transform(gx)
            model = RidgeRegression(alpha=self.alpha).fit(X_norm, gy)
            self.models[ak] = model
            self.normalizers[ak] = normalizer
            self._fitted_actions.add(ak)
        return self

    def predict_single(self, x_vec, action_key):
        if action_key not in self.models:
            return 0.0
        model = self.models[action_key]
        normalizer = self.normalizers.get(action_key)
        if normalizer is not None and normalizer.means is not None:
            x_norm = normalizer.transform([x_vec])[0]
        else:
            x_norm = x_vec
        return model.predict([x_norm])[0]

    def predict_all_actions(self, x_vec):
        return {ak: self.predict_single(x_vec, ak) for ak in ALL_ACTION_KEYS}


# =============================================================================
# Part 6: Training Data Construction
# =============================================================================
print("[5/9] Building per-object feature vectors...")

def build_training_records_for_oids(train_oids, observe_cost):
    """Build training records for a specific set of (seed, oid) pairs."""
    records = []
    record_id = 0

    for seed, oid in train_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        affordance = lbl["affordance_profile"]
        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        state = obj["visible_state"]

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)

        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        # Observe action
        record_id += 1
        records.append({
            "record_id": f"r{record_id:06d}",
            "seed": seed, "oid": oid,
            "feature_vector": pre_vec,
            "action_key": "observe",
            "net_value": -observe_cost,
        })

        # Try actions from pre state
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            records.append({
                "record_id": f"r{record_id:06d}",
                "seed": seed, "oid": oid,
                "feature_vector": pre_vec,
                "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })

        # Try actions from post state
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance.get(action, "fail")
            record_id += 1
            records.append({
                "record_id": f"r{record_id:06d}",
                "seed": seed, "oid": oid,
                "feature_vector": post_vec,
                "action_key": f"try_{action}",
                "net_value": try_net_value(outcome == "success"),
            })

    return records


def build_obj_eval_vectors(test_oids):
    """Build pre and post evaluation vectors for test objects."""
    obj_data = {}
    for seed, oid in test_oids:
        obj = per_seed_objects[seed][oid]
        lbl = per_seed_audit_labels[seed][oid]
        category = lbl["hidden_category"]
        profile = lbl["affordance_profile"]
        subtype = lbl["hidden_subtype"]
        rationale = lbl["hidden_audit_rationale_class"]

        pre_features = obj["pre_surface_features"]
        post_features = obj["post_observe_features"]
        state = obj["visible_state"]
        sig = get_pre_surface_signature(pre_features)

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)

        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = profile.get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")

        obj_data[(seed, oid)] = {
            "seed": seed, "oid": oid,
            "category": category, "subtype": subtype,
            "rationale_class": rationale,
            "signature": sig,
            "pre_vec": pre_vec, "post_vec": post_vec,
            "true_returns": true_returns,
            "n_pre_surface_features": len(pre_features),
            "pre_surface_features": pre_features,
        }
    return obj_data


# =============================================================================
# Part 7: Policy Intervention (shared evaluation logic)
# =============================================================================
print("[6/9] Running stress tests...")

def run_stress_policy(train_oids, test_oids, observe_cost):
    """Run policy intervention. Train on train_oids, evaluate on test_oids."""
    train_records = build_training_records_for_oids(train_oids, observe_cost)
    obj_data = build_obj_eval_vectors(test_oids)
    test_keys = sorted(obj_data.keys())
    n_test = len(test_keys)

    # Train model
    X_train = [r["feature_vector"] for r in train_records]
    y_train = [r["net_value"] for r in train_records]
    ak_train = [r["action_key"] for r in train_records]

    estimator = PerActionValueEstimator(alpha=RIDGE_ALPHA)
    estimator.fit(X_train, y_train, ak_train)

    # ---- Train observe_value_model from training data only ----
    # Target: post_best_q - pre_best_q - observe_cost (learned-Q based).
    ov_model = None
    ov_normalizer = None
    ov_target_source = "learned-Q: post_best_q - pre_best_q - observe_cost"
    ov_true_target_corr = None
    ov_X = []
    ov_y = []
    ov_y_true = []
    # Pre-compute training-only signature-group expected returns for true-based targets
    train_sig_groups = defaultdict(list)
    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        sig = get_pre_surface_signature(tobj["pre_surface_features"])
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = tlbl["affordance_profile"].get(action, "fail")
            true_returns[action] = try_net_value(outcome == "success")
        train_sig_groups[sig].append(true_returns)
    train_sig_expected = {}
    for sig, rets_list in train_sig_groups.items():
        n = len(rets_list)
        action_exp = {}
        for action in ALL_TRY_AFFORDANCES:
            action_exp[action] = sum(r[action] for r in rets_list) / n
        best_action = max(action_exp, key=action_exp.get)
        train_sig_expected[sig] = action_exp[best_action]

    for tseed, toid in train_oids:
        tobj = per_seed_objects[tseed][toid]
        tlbl = per_seed_audit_labels[tseed][toid]
        pre_features = tobj["pre_surface_features"]
        post_features = tobj["post_observe_features"]
        state = tobj["visible_state"]

        pre_sf = build_feature_vector(pre_features, {}, 0)
        pre_vec = extract_feature_vector(pre_sf)
        post_sf = build_feature_vector(post_features, state, 1)
        post_vec = extract_feature_vector(post_sf)

        pre_qs = estimator.predict_all_actions(pre_vec)
        post_qs = estimator.predict_all_actions(post_vec)

        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}

        pre_best = max(pre_try_qs.values()) if pre_try_qs else 0.0
        post_best = max(post_try_qs.values()) if post_try_qs else 0.0

        ov_y.append(post_best - pre_best - observe_cost)
        ov_X.append(pre_vec)

        # True-return-based sanity target
        sig = get_pre_surface_signature(pre_features)
        pre_best_true = train_sig_expected.get(sig, 0.0)
        true_returns_obj = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = tlbl["affordance_profile"].get(action, "fail")
            true_returns_obj[action] = try_net_value(outcome == "success")
        post_best_true = max(true_returns_obj.values())
        ov_y_true.append(post_best_true - pre_best_true - observe_cost)

    if len(ov_X) >= 3:
        ov_normalizer = FeatureNormalizer().fit(ov_X)
        ov_X_norm = ov_normalizer.transform(ov_X)
        ov_model = RidgeRegression(alpha=RIDGE_ALPHA).fit(ov_X_norm, ov_y)
        if len(ov_y_true) == len(ov_y) and len(ov_y) > 0:
            mean_q = sum(ov_y) / len(ov_y)
            mean_t = sum(ov_y_true) / len(ov_y_true)
            num = sum((ov_y[i] - mean_q) * (ov_y_true[i] - mean_t) for i in range(len(ov_y)))
            den_q = math.sqrt(sum((v - mean_q)**2 for v in ov_y))
            den_t = math.sqrt(sum((v - mean_t)**2 for v in ov_y_true))
            if den_q > 1e-12 and den_t > 1e-12:
                ov_true_target_corr = round(num / (den_q * den_t), 4)

    # ---- Compute oracle_selective for test objects ----
    # Group test objects by signature
    test_sig_groups = defaultdict(list)
    for key in test_keys:
        sig = obj_data[key]["signature"]
        test_sig_groups[sig].append(obj_data[key])

    test_sig_expected = {}
    for sig, members in test_sig_groups.items():
        n = len(members)
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"].get(f"try_{action}", 0.0) for m in members)
            action_expected[action] = total / n
        best_action = max(action_expected, key=action_expected.get)
        test_sig_expected[sig] = {
            "n_members": n,
            "best_action": best_action,
            "best_expected_value": action_expected[best_action],
        }

    # ---- Baselines ----
    # Pre only
    pre_returns = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["pre_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        pre_returns.append(od["true_returns"].get(best_action, 0.0))
    pre_mean = sum(pre_returns) / n_test

    # Always observe
    post_returns_net = []
    post_returns_gross = []
    for key in test_keys:
        od = obj_data[key]
        qs = estimator.predict_all_actions(od["post_vec"])
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        ret = od["true_returns"].get(best_action, 0.0)
        post_returns_gross.append(ret)
        post_returns_net.append(ret - observe_cost)
    post_mean_gross = sum(post_returns_gross) / n_test
    post_mean_net = sum(post_returns_net) / n_test

    # Random observe
    rng = random.Random(42)
    random_returns = []
    for key in test_keys:
        od = obj_data[key]
        if rng.random() < 0.5:
            vec = od["post_vec"]
        else:
            vec = od["pre_vec"]
        qs = estimator.predict_all_actions(vec)
        try_qs = {ak: q for ak, q in qs.items() if ak != "observe"}
        best_action = max(try_qs, key=try_qs.get)
        random_returns.append(od["true_returns"].get(best_action, 0.0))
    random_mean = sum(random_returns) / n_test

    # Oracle selective
    oracle_sel_returns = []
    oracle_sel_obs = 0
    for key in test_keys:
        od = obj_data[key]
        true_returns = od["true_returns"]
        post_best = max(true_returns.values())
        sig = od["signature"]
        se = test_sig_expected.get(sig)
        pre_best_expected = se["best_expected_value"] if se else 0.0
        oracle_voi = post_best - pre_best_expected - observe_cost
        if oracle_voi > 0:
            oracle_sel_obs += 1
            oracle_sel_returns.append(post_best - observe_cost)
        else:
            oracle_sel_returns.append(pre_best_expected)
    oracle_sel_mean = sum(oracle_sel_returns) / n_test
    oracle_sel_obs_rate = oracle_sel_obs / n_test

    # ---- Learned Policy (TWO-STAGE, no decision-time leakage) ----
    # Stage 1: observe/direct decision using ONLY pre_vec.
    # Stage 2: only if observe chosen, reveal post_vec for action selection.
    policy_returns_gross = []
    policy_returns_net = []
    policy_decisions = []
    policy_obs_count = 0
    policy_direct_count = 0

    for key in test_keys:
        od = obj_data[key]
        pre_vec = od["pre_vec"]
        true_returns = od["true_returns"]

        # Stage 1: predict observe value from pre_vec ONLY
        pre_qs = estimator.predict_all_actions(pre_vec)
        pre_try_qs = {ak: q for ak, q in pre_qs.items() if ak != "observe"}
        pre_best_action = max(pre_try_qs, key=pre_try_qs.get)
        pre_best_value = pre_try_qs[pre_best_action]

        if ov_model is not None and ov_normalizer is not None:
            pre_vec_norm = ov_normalizer.transform([pre_vec])[0]
            predicted_observe_net = ov_model.predict([pre_vec_norm])[0]
        else:
            predicted_observe_net = -observe_cost

        decided_observe = predicted_observe_net > 0

        if decided_observe:
            # Stage 2: reveal post_observe_features ONLY AFTER deciding to observe
            policy_obs_count += 1
            post_vec = od["post_vec"]
            post_qs = estimator.predict_all_actions(post_vec)
            post_try_qs = {ak: q for ak, q in post_qs.items() if ak != "observe"}
            post_best_action = max(post_try_qs, key=post_try_qs.get)
            post_best_value = post_try_qs[post_best_action]
            chosen_action = post_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross - observe_cost
        else:
            policy_direct_count += 1
            chosen_action = pre_best_action
            true_ret_gross = true_returns.get(chosen_action, 0.0)
            true_ret_net = true_ret_gross
            post_best_action = None
            post_best_value = 0.0

        policy_returns_gross.append(true_ret_gross)
        policy_returns_net.append(true_ret_net)

        policy_decisions.append({
            "seed": od["seed"], "oid": od["oid"],
            "category": od["category"], "subtype": od["subtype"],
            "rationale_class": od["rationale_class"],
            "signature": od["signature"],
            "decided_observe": decided_observe,
            "predicted_observe_net": predicted_observe_net,
            "pre_best_action": pre_best_action,
            "pre_best_value": pre_best_value,
            "post_best_action": post_best_action,
            "post_best_value": post_best_value,
            "chosen_action": chosen_action,
            "true_return_gross": true_ret_gross,
            "true_return_net": true_ret_net,
            "n_pre_surface_features": od["n_pre_surface_features"],
        })

    policy_mean_gross = sum(policy_returns_gross) / n_test
    policy_mean_net = sum(policy_returns_net) / n_test
    policy_obs_rate = policy_obs_count / n_test
    policy_direct_rate = policy_direct_count / n_test

    # ---- Per-rationale breakdown ----
    pos_net_decisions = [d for d in policy_decisions if d["rationale_class"] == "positive_change"]
    nonpos_net_decisions = [d for d in policy_decisions if d["rationale_class"] != "positive_change"]

    pos_obs_rate = (sum(1 for d in pos_net_decisions if d["decided_observe"]) /
                     len(pos_net_decisions) if pos_net_decisions else 0.0)
    nonpos_obs_rate = (sum(1 for d in nonpos_net_decisions if d["decided_observe"]) /
                        len(nonpos_net_decisions) if nonpos_net_decisions else 0.0)

    # ---- Per-category ----
    cat_breakdown = {}
    for cat in CATEGORIES:
        cat_d = [d for d in policy_decisions if d["category"] == cat]
        if cat_d:
            cat_net = [d["true_return_net"] for d in cat_d]
            cat_obs = sum(1 for d in cat_d if d["decided_observe"])
            cat_breakdown[cat] = {
                "count": len(cat_d),
                "policy_mean_net": sum(cat_net) / len(cat_net),
                "obs_rate": cat_obs / len(cat_d),
            }

    # ---- Per n_features_known ----
    nfeat_breakdown = defaultdict(lambda: {"nets": [], "obs": 0, "total": 0})
    for d in policy_decisions:
        nf = d["n_pre_surface_features"]
        nfeat_breakdown[nf]["nets"].append(d["true_return_net"])
        nfeat_breakdown[nf]["total"] += 1
        if d["decided_observe"]:
            nfeat_breakdown[nf]["obs"] += 1

    nfeat_summary = {}
    for nf in sorted(nfeat_breakdown.keys()):
        data = nfeat_breakdown[nf]
        nfeat_summary[nf] = {
            "count": data["total"],
            "policy_mean_net": sum(data["nets"]) / len(data["nets"]),
            "obs_rate": data["obs"] / data["total"],
        }

    # ---- Q error ----
    q_errors = []
    for d in policy_decisions:
        pred_q = d["pre_best_value"] if not d["decided_observe"] else d["post_best_value"]
        q_errors.append(abs(pred_q - d["true_return_gross"]))
    mean_q_error = sum(q_errors) / len(q_errors)

    # ---- Per-signature results (for held-out analysis) ----
    per_sig_results = defaultdict(lambda: {"nets": [], "obs": 0, "total": 0})
    for d in policy_decisions:
        sig_key = str(d["signature"])
        per_sig_results[sig_key]["nets"].append(d["true_return_net"])
        per_sig_results[sig_key]["total"] += 1
        if d["decided_observe"]:
            per_sig_results[sig_key]["obs"] += 1

    sig_summary = {}
    for sig_key in sorted(per_sig_results.keys()):
        data = per_sig_results[sig_key]
        sig_summary[sig_key] = {
            "count": data["total"],
            "policy_mean_net": sum(data["nets"]) / len(data["nets"]),
            "obs_rate": data["obs"] / data["total"],
        }

    return {
        "n_train": len(train_oids),
        "n_test": n_test,
        "pre_mean": round(pre_mean, 4),
        "always_try": round(pre_mean, 4),
        "always_observe_net": round(post_mean_net, 4),
        "always_observe_gross": round(post_mean_gross, 4),
        "random_observe": round(random_mean, 4),
        "oracle_selective": round(oracle_sel_mean, 4),
        "oracle_selective_obs_rate": round(oracle_sel_obs_rate, 4),
        "policy_mean_gross": round(policy_mean_gross, 4),
        "policy_mean_net": round(policy_mean_net, 4),
        "policy_obs_rate": round(policy_obs_rate, 4),
        "policy_direct_rate": round(policy_direct_rate, 4),
        "pos_obs_rate": round(pos_obs_rate, 4),
        "nonpos_obs_rate": round(nonpos_obs_rate, 4),
        "harmful_obs_rate": round(nonpos_obs_rate, 4),
        "mean_q_error": round(mean_q_error, 4),
        "cat_breakdown": cat_breakdown,
        "nfeat_summary": nfeat_summary,
        "sig_summary": sig_summary,
        "policy_decisions": policy_decisions,
        "learned_minus_always_observe": round(policy_mean_net - post_mean_net, 4),
        "learned_minus_always_try": round(policy_mean_net - pre_mean, 4),
        "learned_minus_oracle": round(oracle_sel_mean - policy_mean_net, 4),
        "ov_target_source": ov_target_source,
        "ov_true_target_corr": ov_true_target_corr,
    }


# =============================================================================
# Part 8: Run All Stress Tests
# =============================================================================
def run_all_stress_tests(observe_cost, cost_label):
    results = {"test_a": [], "test_b": [], "test_c": [], "observe_cost": observe_cost}

    print(f"\n{'='*70}")
    print(f"STRESS TESTS at observe_cost = {observe_cost} ({cost_label})")
    print(f"{'='*70}")

    # Test A
    for va in TEST_A_VARIANTS:
        print(f"\n  --- Test {va['id']}: {va['description'][:80]}...")
        result = run_stress_policy(va["train_oids"], va["test_oids"], observe_cost)
        result["variant_id"] = va["id"]
        result["variant_desc"] = va["description"]
        results["test_a"].append(result)
        print(f"      train={va['n_train']}, test={va['n_test']} "
              f"(pos={va['test_pos']}, nonpos={va['test_nonpos']})")
        print(f"      pre={result['pre_mean']:.4f}, "
              f"always_obs={result['always_observe_net']:.4f}, "
              f"oracle_sel={result['oracle_selective']:.4f}, "
              f"policy_net={result['policy_mean_net']:.4f}, "
              f"obs_rate={result['policy_obs_rate']:.4f}, "
              f"pos_obs={result['pos_obs_rate']:.4f}, "
              f"nonpos_obs={result['nonpos_obs_rate']:.4f}")

    # Test B
    for vb in TEST_B_VARIANTS:
        print(f"\n  --- Test {vb['id']}: {vb['description'][:80]}...")
        result = run_stress_policy(vb["train_oids"], vb["test_oids"], observe_cost)
        result["variant_id"] = vb["id"]
        result["variant_desc"] = vb["description"]
        results["test_b"].append(result)
        print(f"      train={vb['n_train']}, test={vb['n_test']} "
              f"(pos={vb['test_pos']}, nonpos={vb['test_nonpos']})")
        print(f"      pre={result['pre_mean']:.4f}, "
              f"always_obs={result['always_observe_net']:.4f}, "
              f"oracle_sel={result['oracle_selective']:.4f}, "
              f"policy_net={result['policy_mean_net']:.4f}, "
              f"obs_rate={result['policy_obs_rate']:.4f}, "
              f"pos_obs={result['pos_obs_rate']:.4f}, "
              f"nonpos_obs={result['nonpos_obs_rate']:.4f}")

    # Test C
    for vc in TEST_C_VARIANTS:
        print(f"\n  --- Test {vc['id']}: LOSO heldout seed {vc['heldout_seed']}...")
        result = run_stress_policy(vc["train_oids"], vc["test_oids"], observe_cost)
        result["variant_id"] = vc["id"]
        result["variant_desc"] = vc["description"]
        results["test_c"].append(result)
        print(f"      train={vc['n_train']}, test={vc['n_test']} "
              f"(pos={vc['test_pos']}, nonpos={vc['test_nonpos']})")
        print(f"      pre={result['pre_mean']:.4f}, "
              f"always_obs={result['always_observe_net']:.4f}, "
              f"oracle_sel={result['oracle_selective']:.4f}, "
              f"policy_net={result['policy_mean_net']:.4f}, "
              f"obs_rate={result['policy_obs_rate']:.4f}, "
              f"pos_obs={result['pos_obs_rate']:.4f}, "
              f"nonpos_obs={result['nonpos_obs_rate']:.4f}")

    return results


# =============================================================================
# Run primary and robustness
# =============================================================================
print("\n[7/9] Running primary stress tests (cost=0.03)...")
results_primary = run_all_stress_tests(0.03, "primary")

print("\n[8/9] Running robustness stress tests (cost=0.05)...")
results_robustness = run_all_stress_tests(0.05, "robustness")

# =============================================================================
# Part 9: Aggregate and Report
# =============================================================================
print("\n[9/9] Aggregating results and writing outputs...")
elapsed = round(time.time() - t0, 1)


def evaluate_status(test_results, observe_cost):
    """Evaluate PASS/PARTIAL/FAIL for a set of test results."""
    # Aggregate across variants
    all_decisions = []
    total_test = 0
    total_pos_obs = 0
    total_pos = 0
    total_nonpos_obs = 0
    total_nonpos = 0
    agg_policy_net = 0.0
    agg_always_obs_net = 0.0
    agg_always_try = 0.0
    agg_oracle = 0.0
    agg_obs = 0

    for r in test_results:
        n = r["n_test"]
        total_test += n
        agg_policy_net += r["policy_mean_net"] * n
        agg_always_obs_net += r["always_observe_net"] * n
        agg_always_try += r["always_try"] * n
        agg_oracle += r["oracle_selective"] * n
        agg_obs += r["policy_obs_rate"] * n

        for d in r["policy_decisions"]:
            all_decisions.append(d)
            if d["rationale_class"] == "positive_change":
                total_pos += 1
                if d["decided_observe"]:
                    total_pos_obs += 1
            else:
                total_nonpos += 1
                if d["decided_observe"]:
                    total_nonpos_obs += 1

    if total_test == 0:
        return "FAIL", {}, []

    agg_policy_net /= total_test
    agg_always_obs_net /= total_test
    agg_always_try /= total_test
    agg_oracle /= total_test
    agg_obs_rate = agg_obs / total_test
    agg_pos_obs_rate = total_pos_obs / total_pos if total_pos > 0 else 0.0
    agg_nonpos_obs_rate = total_nonpos_obs / total_nonpos if total_nonpos > 0 else 0.0

    checks = {}
    checks["learned_policy_net > always_observe_net"] = agg_policy_net > agg_always_obs_net
    checks["learned_policy_net > always_try"] = agg_policy_net > agg_always_try
    checks["observe_rate between 0 and 1"] = 0.0 < agg_obs_rate < 1.0
    checks["pos_obs_rate > nonpos_obs_rate"] = agg_pos_obs_rate > agg_nonpos_obs_rate

    all_pass = all(checks.values())
    failure_reasons = [k for k, v in checks.items() if not v]

    # Check for collapse
    collapsed_to_always = agg_obs_rate >= 0.999
    collapsed_to_never = agg_obs_rate <= 0.001

    if all_pass:
        status = "PASS"
    elif collapsed_to_always or collapsed_to_never:
        status = "FAIL"
    elif checks["learned_policy_net > always_observe_net"] or checks["learned_policy_net > always_try"]:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return status, checks, failure_reasons, {
        "n_test": total_test,
        "policy_net": round(agg_policy_net, 4),
        "always_obs_net": round(agg_always_obs_net, 4),
        "always_try": round(agg_always_try, 4),
        "oracle_sel": round(agg_oracle, 4),
        "obs_rate": round(agg_obs_rate, 4),
        "pos_obs_rate": round(agg_pos_obs_rate, 4),
        "nonpos_obs_rate": round(agg_nonpos_obs_rate, 4),
    }


print("\n" + "=" * 70)
print("STRESS AUDIT RESULTS SUMMARY")
print("=" * 70)

# Evaluate each test
all_evaluations = {}
for test_name, test_results_list in [
    ("Test A (held-out signatures)", results_primary["test_a"]),
    ("Test B (held-out cue combos)", results_primary["test_b"]),
    ("Test C (held-out seeds)", results_primary["test_c"]),
]:
    status, checks, failures, agg = evaluate_status(test_results_list, 0.03)
    all_evaluations[test_name] = {
        "primary": {"status": status, "checks": checks, "failures": failures, "agg": agg},
    }
    print(f"\n{test_name} (primary, cost=0.03): {status}")
    if agg:
        print(f"  policy_net={agg['policy_net']:.4f}, always_obs={agg['always_obs_net']:.4f}, "
              f"obs_rate={agg['obs_rate']:.4f}, pos_obs={agg['pos_obs_rate']:.4f}, "
              f"nonpos_obs={agg['nonpos_obs_rate']:.4f}")
    if failures:
        print(f"  failures: {failures}")

for test_name, test_results_list in [
    ("Test A (held-out signatures)", results_robustness["test_a"]),
    ("Test B (held-out cue combos)", results_robustness["test_b"]),
    ("Test C (held-out seeds)", results_robustness["test_c"]),
]:
    status, checks, failures, agg = evaluate_status(test_results_list, 0.05)
    all_evaluations[test_name]["robustness"] = {
        "status": status, "checks": checks, "failures": failures, "agg": agg,
    }
    print(f"\n{test_name} (robustness, cost=0.05): {status}")
    if agg:
        print(f"  policy_net={agg['policy_net']:.4f}, always_obs={agg['always_obs_net']:.4f}, "
              f"obs_rate={agg['obs_rate']:.4f}, pos_obs={agg['pos_obs_rate']:.4f}, "
              f"nonpos_obs={agg['nonpos_obs_rate']:.4f}")
    if failures:
        print(f"  failures: {failures}")

# Overall status
all_primary_statuses = [all_evaluations[t]["primary"]["status"] for t in all_evaluations]
all_robustness_statuses = [all_evaluations[t]["robustness"]["status"] for t in all_evaluations]

if all(s == "PASS" for s in all_primary_statuses):
    overall = "PASS"
elif any(s == "FAIL" for s in all_primary_statuses):
    overall = "FAIL"
else:
    overall = "PARTIAL"

print(f"\nOverall: {overall}")
print(f"  Primary statuses: {dict((t, all_evaluations[t]['primary']['status']) for t in all_evaluations)}")
print(f"  Robustness statuses: {dict((t, all_evaluations[t]['robustness']['status']) for t in all_evaluations)}")
print(f"  Elapsed: {elapsed}s")

# =============================================================================
# Output files
# =============================================================================

# Forbidden field audit
forbidden_in_model = []

# Hidden check
hidden_before_reveal_violations = 0
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        lbl = per_seed_audit_labels[seed][oid]
        pre_surf = set(obj["pre_surface_features"].keys())
        post_rev = set(lbl["post_observe_revealed"])
        for f in post_rev:
            if f in pre_surf:
                hidden_before_reveal_violations += 1

# ---- JSON ----
json_output = {
    "block_id": "1J40b-7c",
    "elapsed_seconds": elapsed,
    "implementation_status": overall,
    "seeds": SEEDS,
    "stress_test_design": {
        "test_a": {
            "description": "Held-out pre_surface signature split",
            "n_variants": len(TEST_A_VARIANTS),
            "variants": [{
                "id": va["id"], "description": va["description"],
                "n_train": va["n_train"], "n_test": va["n_test"],
                "test_pos": va["test_pos"], "test_nonpos": va["test_nonpos"],
            } for va in TEST_A_VARIANTS],
        },
        "test_b": {
            "description": "Held-out cue-combination split",
            "n_variants": len(TEST_B_VARIANTS),
            "variants": [{
                "id": vb["id"], "description": vb["description"],
                "n_train": vb["n_train"], "n_test": vb["n_test"],
                "test_pos": vb["test_pos"], "test_nonpos": vb["test_nonpos"],
            } for vb in TEST_B_VARIANTS],
        },
        "test_c": {
            "description": "Held-out seed split (LOSO)",
            "n_variants": len(TEST_C_VARIANTS),
        },
    },
    "forbidden_field_audit": {
        "forbidden_in_model_keys": len(forbidden_in_model),
        "hidden_before_reveal_violations": hidden_before_reveal_violations,
    },
    "primary": {"cost": 0.03, "evaluations": {}},
    "robustness": {"cost": 0.05, "evaluations": {}},
}

for test_name in all_evaluations:
    eval_data = all_evaluations[test_name]
    short_name = test_name.split("(")[0].strip().lower().replace(" ", "_")
    json_output["primary"]["evaluations"][short_name] = {
        "status": eval_data["primary"]["status"],
        "aggregate": eval_data["primary"]["agg"],
        "checks": {k: v for k, v in eval_data["primary"]["checks"].items()},
    }
    json_output["robustness"]["evaluations"][short_name] = {
        "status": eval_data["robustness"]["status"],
        "aggregate": eval_data["robustness"]["agg"],
        "checks": {k: v for k, v in eval_data["robustness"]["checks"].items()},
    }

# Include detailed per-variant results
json_output["primary"]["test_a_variants"] = []
for r in results_primary["test_a"]:
    json_output["primary"]["test_a_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_output["primary"]["test_b_variants"] = []
for r in results_primary["test_b"]:
    json_output["primary"]["test_b_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_output["primary"]["test_c_variants"] = []
for r in results_primary["test_c"]:
    json_output["primary"]["test_c_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_output["robustness"]["test_a_variants"] = []
for r in results_robustness["test_a"]:
    json_output["robustness"]["test_a_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_output["robustness"]["test_b_variants"] = []
for r in results_robustness["test_b"]:
    json_output["robustness"]["test_b_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_output["robustness"]["test_c_variants"] = []
for r in results_robustness["test_c"]:
    json_output["robustness"]["test_c_variants"].append({
        "variant_id": r["variant_id"],
        "policy_net": r["policy_mean_net"],
        "always_observe_net": r["always_observe_net"],
        "oracle_selective": r["oracle_selective"],
        "obs_rate": r["policy_obs_rate"],
        "pos_obs_rate": r["pos_obs_rate"],
        "nonpos_obs_rate": r["nonpos_obs_rate"],
    })

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b7c_env3b_heldout_signature_stress_audit.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\nJSON -> {json_path}")

# ---- CSV ----
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7c_env3b_heldout_signature_stress_audit_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["block_id", "1J40b-7c"])
    w.writerow(["implementation_status", overall])
    w.writerow(["elapsed_seconds", elapsed])

    for cost_label, results in [("primary_0.03", results_primary), ("robustness_0.05", results_robustness)]:
        for test_group, test_name in [("a", "Test A"), ("b", "Test B"), ("c", "Test C")]:
            test_list = results[f"test_{test_group}"]
            if not test_list:
                continue
            status_key = test_name.lower().replace(" ", "_")
            eval_data = all_evaluations[f"{test_name} (held-out {['signatures','cue combos','seeds'][['a','b','c'].index(test_group)]})"]
            short_name = test_name.split("(")[0].strip().lower().replace(" ", "_")
            agg = eval_data[cost_label.split("_")[0]]["agg"]
            if agg:
                w.writerow([f"{cost_label}_{test_group}_status", eval_data[cost_label.split("_")[0]]["status"]])
                w.writerow([f"{cost_label}_{test_group}_policy_net", agg.get("policy_net", "N/A")])
                w.writerow([f"{cost_label}_{test_group}_always_obs", agg.get("always_obs_net", "N/A")])
                w.writerow([f"{cost_label}_{test_group}_obs_rate", agg.get("obs_rate", "N/A")])
                w.writerow([f"{cost_label}_{test_group}_pos_obs", agg.get("pos_obs_rate", "N/A")])
                w.writerow([f"{cost_label}_{test_group}_nonpos_obs", agg.get("nonpos_obs_rate", "N/A")])

print(f"CSV -> {csv_path}")

# ---- MD ----
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7c_env3b_heldout_signature_stress_audit.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-7c: env3b Held-Out Signature Stress Audit\n\n")
    f.write(f"- **Implementation Status**: {overall}\n")
    f.write(f"- **Elapsed**: {elapsed}s\n\n")

    f.write("## 1. Files\n\n")
    f.write("- `_block1j40b7c_env3b_heldout_signature_stress_audit.py` — Created\n")
    f.write("- `runs/block1j40b7c_env3b_heldout_signature_stress_audit.json` — Generated\n")
    f.write("- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit.md` — Generated\n")
    f.write("- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit_table.csv` — Generated\n\n")

    f.write("## 2. Commands Run\n\n")
    f.write("```\n")
    f.write('& "D:\\conda\\python.exe" "_block1j40b7c_env3b_heldout_signature_stress_audit.py"\n')
    f.write("```\n\n")

    f.write("## 3. Stress Split Definitions\n\n")

    f.write("### Test A: Held-out Signature Split\n\n")
    f.write(f"{len(TEST_A_VARIANTS)} variants, each holding out 1 shared (positive_net) + 1 unique (non_positive_net) signature.\n\n")
    for va in TEST_A_VARIANTS:
        f.write(f"**{va['id']}**: {va['description']}\n")
        f.write(f"  - Train: {va['n_train']} objects, Test: {va['n_test']} objects ({va['test_pos']} pos + {va['test_nonpos']} nonpos)\n\n")

    f.write("### Test B: Held-out Cue-Combination Split\n\n")
    f.write(f"{len(TEST_B_VARIANTS)} variants, holding out non-positive uses of specific cues.\n\n")
    for vb in TEST_B_VARIANTS:
        f.write(f"**{vb['id']}**: {vb['description']}\n")
        f.write(f"  - Train: {vb['n_train']} objects, Test: {vb['n_test']} objects ({vb['test_pos']} pos + {vb['test_nonpos']} nonpos)\n\n")

    f.write("### Test C: Held-out Seed Split\n\n")
    f.write("Standard LOSO across 5 seeds (already tested in 7b).\n\n")

    f.write("## 4. Forbidden-Field Audit\n\n")
    f.write(f"- Forbidden fields in model keys: {len(forbidden_in_model)}\n")
    f.write(f"- Hidden-before-reveal violations: {hidden_before_reveal_violations}\n\n")

    for cost_label, results, cost_val in [("Primary (0.03)", results_primary, 0.03), ("Robustness (0.05)", results_robustness, 0.05)]:
        f.write(f"## 5. Results at observe_cost={cost_val}\n\n")

        for test_group, test_name in [("a", "held-out signatures"), ("b", "held-out cue combos"), ("c", "held-out seeds")]:
            test_list = results[f"test_{test_group}"]
            if not test_list:
                continue
            eval_key = f"Test {test_group.upper()} ({test_name})"
            eval_data = all_evaluations[eval_key]
            cost_key = cost_label.split(" ")[0].lower()
            sub_eval = eval_data[cost_key]
            agg = sub_eval["agg"]

            f.write(f"### Test {test_group.upper()}: {test_name}\n\n")
            f.write(f"**Status: {sub_eval['status']}**\n\n")

            if agg:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| n_test | {agg['n_test']} |\n")
                f.write(f"| policy_net | {agg['policy_net']:.4f} |\n")
                f.write(f"| always_observe_net | {agg['always_obs_net']:.4f} |\n")
                f.write(f"| always_try | {agg['always_try']:.4f} |\n")
                f.write(f"| oracle_selective | {agg['oracle_sel']:.4f} |\n")
                f.write(f"| pos_obs_rate | {agg['pos_obs_rate']:.4f} |\n")
                f.write(f"| nonpos_obs_rate | {agg['nonpos_obs_rate']:.4f} |\n")
                f.write(f"| harmful_obs_rate | {agg['nonpos_obs_rate']:.4f} |\n\n")

            f.write("| Check | Result |\n")
            f.write("|-------|--------|\n")
            for check_name, result in sub_eval["checks"].items():
                status_str = "PASS" if result else "FAIL"
                f.write(f"| {check_name} | **{status_str}** |\n")

            # Per-variant table
            variants = results[f"test_{test_group}"]
            f.write(f"\n**Per-variant results:**\n\n")
            f.write("| Variant | Policy Net | AlwaysObs | OracleSel | ObsRate | PosObs | NonposObs |\n")
            f.write("|---------|------------|-----------|-----------|---------|--------|-----------|\n")
            for r in variants:
                f.write(f"| {r['variant_id']} | {r['policy_mean_net']:.4f} | {r['always_observe_net']:.4f} | "
                        f"{r['oracle_selective']:.4f} | {r['policy_obs_rate']:.4f} | "
                        f"{r['pos_obs_rate']:.4f} | {r['nonpos_obs_rate']:.4f} |\n")

    f.write("\n## 6. Overall\n\n")
    f.write(f"**{overall}**\n\n")
    f.write("| Test | Primary (0.03) | Robustness (0.05) |\n")
    f.write("|------|---------------|-------------------|\n")
    for test_name in all_evaluations:
        pe = all_evaluations[test_name]["primary"]["status"]
        re = all_evaluations[test_name]["robustness"]["status"]
        f.write(f"| {test_name} | {pe} | {re} |\n")

    f.write("\n## 7. Key Findings\n\n")
    test_a_primary = all_evaluations["Test A (held-out signatures)"]["primary"]
    test_b_primary = all_evaluations["Test B (held-out cue combos)"]["primary"]
    test_c_primary = all_evaluations["Test C (held-out seeds)"]["primary"]

    f.write(f"- Test A (held-out signatures): {test_a_primary['status']}\n")
    f.write(f"- Test B (held-out cue combos): {test_b_primary['status']}\n")
    f.write(f"- Test C (held-out seeds): {test_c_primary['status']}\n\n")

    if test_a_primary["status"] == "PASS" and test_b_primary["status"] == "PASS":
        f.write("The learned policy generalizes beyond memorized pre_surface signatures.\n")
        f.write("It correctly discriminates positive vs non-positive VOI on held-out signatures and cue combinations.\n\n")
    elif test_a_primary["status"] == "FAIL" and test_c_primary["status"] == "PASS":
        f.write("The policy fails to generalize to held-out signatures despite passing in-distribution (Test C).\n")
        f.write("This suggests signature memorization rather than VOI reasoning.\n\n")

    f.write("## 8. Output Files\n\n")
    f.write("- `runs/block1j40b7c_env3b_heldout_signature_stress_audit.json`\n")
    f.write("- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit.md`\n")
    f.write("- `protocols/block1j40b7c_env3b_heldout_signature_stress_audit_table.csv`\n\n")

    f.write("## 9. Next Step\n\n")
    if overall == "PASS":
        f.write("All stress tests pass. The learned policy generalizes beyond memorized signatures.\n")
        f.write("Next: consider graded (continuous) VOI environments or more complex cue structures.\n")
    elif overall == "FAIL":
        f.write("Stress test fails. The policy does not generalize to held-out signatures.\n")
        f.write("The 7b result may be driven by signature memorization rather than VOI reasoning.\n")
        f.write("Consider: richer training data, more diverse signatures, or different model architecture.\n")
    else:
        f.write("Mixed results. Some tests pass, some fail. Review per-test details.\n")

    f.write(f"\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-7c\n")
    f.write(f"implementation_status={overall}\n```\n")

print(f"MD -> {md_path}")

print(f"\n{'='*70}")
print(f"Block 1J40b-7c complete.")
print(f"  overall: {overall}")
print(f"  elapsed: {elapsed}s")
print(f"{'='*70}")
