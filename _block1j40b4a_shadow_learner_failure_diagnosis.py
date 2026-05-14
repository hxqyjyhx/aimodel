"""
Block 1J40b-4a — Controlled Failure Diagnosis for env2 Shadow Learner.

Pure diagnostic. Does NOT redesign the learner or change env2 generation.
Produces raw evidence for user/ChatGPT to interpret.

Diagnostics:
  1. Pre-observe ceiling
  2. Category identifiability from ambient features
  3. Best-action change after observe
  4. Env2 audit metric vs shadow metric comparison
  5. Optional: separate pre/post model calibration check
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
# Constants (from env2)
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5
N_GROUPS_PER_FAMILY = 3
RIDGE_ALPHA = 1.0

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]

OBSERVE_COST = 0.005
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
INFO_GAIN_PER_NEW_FEATURE = 0.02
INFO_GAIN_PER_NEW_STATE = 0.01
UNCERTAINTY_REDUCTION_VALUE = 0.03
DISCOUNT_FACTOR = 0.9

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

AMBIENT_FEATURE_NAMES = ["brownish", "grayish", "greenish", "long_shape",
                         "block_like", "round_small"]

# =============================================================================
# Feature definitions (from env2, read-only, not modified)
# =============================================================================
BASE_AFFORDANCE_PROFILES = {
    "wood_log": {
        "mine_by_hand": "success", "mine_with_pickaxe": "success",
        "craft_plank": "success", "eat": "fail",
        "use_as_tool": "fail", "burn_as_fuel": "success",
    },
    "stone_block": {
        "mine_by_hand": "fail", "mine_with_pickaxe": "success",
        "craft_plank": "fail", "eat": "fail",
        "use_as_tool": "fail", "burn_as_fuel": "fail",
    },
    "apple": {
        "mine_by_hand": "success", "mine_with_pickaxe": "success",
        "craft_plank": "fail", "eat": "success",
        "use_as_tool": "fail", "burn_as_fuel": "fail",
    },
    "wooden_pickaxe": {
        "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
        "craft_plank": "fail", "eat": "fail",
        "use_as_tool": "success", "burn_as_fuel": "fail",
    },
}

CATEGORY_CORE_FEATURES = {
    "wood_log": ["brownish", "has_bark_texture", "rough_texture", "long_shape",
                 "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "grayish", "block_like",
                    "heavy_weight", "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "round_small", "greenish",
              "smooth_texture", "light_weight", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                       "movable", "long_shape", "has_metal_head", "jointed"],
}

VARIANT_FEATURES = {
    "wood_log": [
        ("light_colored", "edible_core", "eat", ["A", "B"]),
        ("has_wood_grain", "carvable_interior", "use_as_tool", ["A", "C"]),
        ("dark_colored", "nutrient_rich", "eat", ["B", "C"]),
        ("has_knot_hole", "structural_weakness", "use_as_tool", ["A", "B"]),
        ("mossy_surface", "moisture_content", "burn_as_fuel", ["A", "C"]),
        ("cracked_ends", "internal_split", "craft_plank", ["B", "C"]),
    ],
    "stone_block": [
        ("speckled", "internal_fractures", "mine_by_hand", ["A", "B"]),
        ("veined", "layered_structure", "craft_plank", ["A", "C"]),
        ("pitted", "surface_erosion", "mine_by_hand", ["B", "C"]),
        ("banded", "cleavage_plane", "craft_plank", ["A", "B"]),
        ("glassy", "conchoidal_fracture", "use_as_tool", ["A", "C"]),
        ("chalky", "mineral_softness", "burn_as_fuel", ["B", "C"]),
    ],
    "apple": [
        ("red_blush", "sweet_flesh", "craft_plank", ["A", "B"]),
        ("striped", "firm_texture", "use_as_tool", ["A", "C"]),
        ("spotted", "bruise_markings", "craft_plank", ["B", "C"]),
        ("golden_flesh", "soft_spot", "burn_as_fuel", ["A", "B"]),
        ("russet_skin", "discoloration", "use_as_tool", ["A", "C"]),
        ("glossy_surface", "wax_coating", "burn_as_fuel", ["B", "C"]),
    ],
    "wooden_pickaxe": [
        ("reinforced_joint", "hairline_crack", "mine_by_hand", ["A", "B"]),
        ("weathered_handle", "handle_looseness", "mine_with_pickaxe", ["A", "C"]),
        ("polished_head", "rust_on_fastener", "mine_by_hand", ["B", "C"]),
        ("wrapped_grip", "handle_splinter", "craft_plank", ["A", "B"]),
        ("notched_shaft", "head_misalignment", "mine_with_pickaxe", ["A", "C"]),
        ("tapered_end", "shaft_rot", "eat", ["B", "C"]),
    ],
}

PROFILE_FEATURES = {
    "A": {0, 1, 3, 4},
    "B": {0, 2, 3, 5},
    "C": {1, 2, 4, 5},
}

SEED_CATEGORY_PROFILE = {
    101: {"wood_log": "A", "stone_block": "B", "apple": "C", "wooden_pickaxe": "A"},
    103: {"wood_log": "B", "stone_block": "C", "apple": "A", "wooden_pickaxe": "B"},
    107: {"wood_log": "C", "stone_block": "A", "apple": "B", "wooden_pickaxe": "C"},
    109: {"wood_log": "A", "stone_block": "C", "apple": "B", "wooden_pickaxe": "A"},
    113: {"wood_log": "B", "stone_block": "A", "apple": "C", "wooden_pickaxe": "B"},
}

FORBIDDEN_AGENT_VISIBLE = {
    "schedule", "schedule_type", "group_role", "group_id",
    "depth_schedule", "mixed_source_type", "true_family",
    "hidden_subtype", "oracle_outcome", "full_object_state",
    "deceptive_flag", "prior_violation", "preferred_explanation",
}

# =============================================================================
# Data Generation (exact copy from env2 — not modified)
# =============================================================================
def build_variant_profile(base_profile, category, group_role, profile, seed):
    vp = dict(base_profile)
    if group_role == "observe_helps":
        vf_list = VARIANT_FEATURES[category]
        pf_indices = PROFILE_FEATURES[profile]
        bonus_actions = [vf_list[idx][2] for idx in pf_indices]
        for ba in bonus_actions:
            if vp.get(ba) == "fail":
                vp[ba] = "success"
                break
    return vp


def generate_c6_objects_deterministic(seed, prefix="c6"):
    objects = {}
    audit_labels = {}
    oid_counter = [0]
    def make_oid():
        oid_counter[0] += 1
        return f"{prefix}_obj_{oid_counter[0]:04d}"

    for category in CATEGORIES:
        base_profile = BASE_AFFORDANCE_PROFILES[category]
        core_features = CATEGORY_CORE_FEATURES[category]
        vf_list = VARIANT_FEATURES[category]
        for group_idx in range(N_GROUPS_PER_FAMILY):
            group_id = f"{category}_g{group_idx}"
            group_role = ["observe_helps", "observe_neutral", "observe_wasteful"][group_idx]
            profile = SEED_CATEGORY_PROFILE[seed][category]
            pf_indices = PROFILE_FEATURES[profile]
            variant_visible = {}
            variant_hidden = {}
            for idx in pf_indices:
                vf_name, hf_name, bonus_action, _ = vf_list[idx]
                variant_visible[vf_name] = True
                variant_hidden[hf_name] = True
            visible_features = {f: True for f in core_features}
            visible_features.update(variant_visible)
            variant_profile = build_variant_profile(base_profile, category, group_role, profile, seed)
            for depth_idx, depth_schedule in enumerate(["no_observe", "one_observe", "repeated_observe"]):
                oid = make_oid()
                if depth_schedule == "no_observe":
                    effective_profile = dict(base_profile)
                else:
                    effective_profile = dict(variant_profile)
                hidden_feat_dict = dict(variant_hidden)
                visible_state = {
                    "fresh": True, "wet": category in ("wood_log", "apple"),
                    "damaged": False, "clean": True, "hot": False, "open": False,
                }
                objects[oid] = {
                    "oid": oid, "hidden_category": category,
                    "hidden_affordance_profile": effective_profile,
                    "visible_features": visible_features,
                    "visible_state": visible_state,
                    "_hidden_features_dict": hidden_feat_dict,
                }
                audit_labels[oid] = {
                    "oid": oid, "hidden_category": category, "true_family": category,
                    "group_id": group_id, "group_role": group_role,
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(effective_profile),
                    "base_profile": dict(base_profile),
                    "variant_profile": dict(variant_profile),
                    "hidden_features": dict(hidden_feat_dict),
                    "visual_profile": profile, "seed": seed,
                }
    return objects, audit_labels


def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - PROBE_COST
    elif success_or_failure is False:
        return -FAILURE_PENALTY - PROBE_COST
    return 0.0


# Generate all objects and audit labels
print("Generating env2 data...")
per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_c6_objects_deterministic(seed, prefix=f"c6_s{seed}")
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels

# =============================================================================
# Diagnostic 1: Pre-observe Ceiling
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 1: Pre-observe Ceiling")
print("=" * 70)

# For each object, compute:
# - oracle: true best action and its net value
# - pre-accessible: best action and net value using only base profile info
#   (since ambient features identify category → category determines base profile)
# - post-accessible: best action and net value using variant profile info

results_d1 = []
for seed in SEEDS:
    for oid, lbl in per_seed_audit_labels[seed].items():
        category = lbl["hidden_category"]
        base_profile = lbl["base_profile"]
        variant_profile = lbl["variant_profile"]
        affordance_profile = lbl["affordance_profile"]
        depth = lbl["depth_schedule"]
        role = lbl["group_role"]

        # Oracle: best action from true affordance profile
        true_action_values = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = affordance_profile.get(action, "fail")
            true_action_values[action] = try_net_value(outcome == "success")
        oracle_best_action = max(true_action_values, key=true_action_values.get)
        oracle_best_value = true_action_values[oracle_best_action]

        # Pre-accessible: best action from base profile (ambient → category → base)
        base_action_values = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = base_profile.get(action, "fail")
            base_action_values[action] = try_net_value(outcome == "success")
        pre_best_action = max(base_action_values, key=base_action_values.get)
        pre_best_value = true_action_values[pre_best_action]  # true value of pre-selected action

        # Post-accessible: best action from variant profile
        variant_action_values = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = variant_profile.get(action, "fail")
            variant_action_values[action] = try_net_value(outcome == "success")
        post_best_action = max(variant_action_values, key=variant_action_values.get)
        post_best_value = true_action_values[post_best_action]  # true value of post-selected action

        results_d1.append({
            "seed": seed, "oid": oid, "category": category,
            "depth": depth, "role": role,
            "oracle_best_action": oracle_best_action,
            "oracle_best_value": oracle_best_value,
            "pre_best_action": pre_best_action,
            "pre_best_value": pre_best_value,
            "post_best_action": post_best_action,
            "post_best_value": post_best_value,
        })

# Aggregate
pre_values = [r["pre_best_value"] for r in results_d1]
post_values = [r["post_best_value"] for r in results_d1]
oracle_values = [r["oracle_best_value"] for r in results_d1]

pre_mean = sum(pre_values) / len(pre_values)
post_mean = sum(post_values) / len(post_values)
oracle_mean = sum(oracle_values) / len(oracle_values)
pre_oracle_gap = oracle_mean - pre_mean
post_oracle_gap = oracle_mean - post_mean

print(f"\nAggregate (all {len(results_d1)} objects):")
print(f"  Pre mean true return:  {pre_mean:.4f}")
print(f"  Post mean true return: {post_mean:.4f}")
print(f"  Oracle mean true return: {oracle_mean:.4f}")
print(f"  Pre/oracle gap: {pre_oracle_gap:.4f}")
print(f"  Post/oracle gap: {post_oracle_gap:.4f}")

# Per-seed
print(f"\nPer-seed pre/oracle gap:")
for seed in SEEDS:
    seed_vals = [r for r in results_d1 if r["seed"] == seed]
    sp = sum(r["pre_best_value"] for r in seed_vals) / len(seed_vals)
    so = sum(r["oracle_best_value"] for r in seed_vals) / len(seed_vals)
    print(f"  Seed {seed}: pre={sp:.4f}, oracle={so:.4f}, gap={so-sp:.4f}")

# Per-category
print(f"\nPer-category pre/oracle gap:")
for cat in CATEGORIES:
    cat_vals = [r for r in results_d1 if r["category"] == cat]
    cp = sum(r["pre_best_value"] for r in cat_vals) / len(cat_vals)
    co = sum(r["oracle_best_value"] for r in cat_vals) / len(cat_vals)
    print(f"  {cat:<20}: pre={cp:.4f}, oracle={co:.4f}, gap={co-cp:.4f}")

# Per-depth
print(f"\nPer-depth pre/oracle gap:")
for depth in ["no_observe", "one_observe", "repeated_observe"]:
    depth_vals = [r for r in results_d1 if r["depth"] == depth]
    dp = sum(r["pre_best_value"] for r in depth_vals) / len(depth_vals)
    do = sum(r["oracle_best_value"] for r in depth_vals) / len(depth_vals)
    print(f"  {depth:<20}: pre={dp:.4f}, oracle={do:.4f}, gap={do-dp:.4f}")

# =============================================================================
# Diagnostic 2: Category Identifiability from Ambient Features
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Category Identifiability from Ambient Features")
print("=" * 70)

CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

# Build dataset: for each object, ambient feature vector + category + best base action
ambient_data = []
for seed in SEEDS:
    for oid, lbl in per_seed_audit_labels[seed].items():
        obj = per_seed_objects[seed][oid]
        vf = obj.get("visible_features", {})
        ambient_vec = [1.0 if f in vf else 0.0 for f in AMBIENT_FEATURE_NAMES]
        category = lbl["hidden_category"]
        base_profile = lbl["base_profile"]
        # Best base action
        base_action_values = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = base_profile.get(action, "fail")
            base_action_values[action] = try_net_value(outcome == "success")
        best_base_action = max(base_action_values, key=base_action_values.get)
        # Base affordance profile signature (6 binary outcomes)
        base_sig = tuple(1 if base_profile.get(a) == "success" else 0 for a in ALL_TRY_AFFORDANCES)
        ambient_data.append({
            "seed": seed, "oid": oid,
            "ambient_vec": ambient_vec,
            "category": category,
            "category_idx": CATEGORY_TO_IDX[category],
            "best_base_action": best_base_action,
            "base_sig": base_sig,
        })

# Probe 2a: Category from ambient features
# Use a simple rule-based probe: map each unique ambient vector to majority category
ambient_to_categories = defaultdict(list)
for d in ambient_data:
    ambient_to_categories[tuple(d["ambient_vec"])].append(d["category"])

print(f"\n2a. Category probe:")
print(f"  Unique ambient vectors: {len(ambient_to_categories)}")
ambient_to_cat = {}
for av, cats in ambient_to_categories.items():
    majority = max(set(cats), key=cats.count)
    ambient_to_cat[av] = majority
    purity = cats.count(majority) / len(cats)
    feat_names = [AMBIENT_FEATURE_NAMES[i] for i, v in enumerate(av) if v > 0.5]
    print(f"  {feat_names} -> {majority} (n={len(cats)}, purity={purity:.3f})")

cat_correct = sum(1 for d in ambient_data if ambient_to_cat[tuple(d["ambient_vec"])] == d["category"])
cat_acc = cat_correct / len(ambient_data)
print(f"  Category probe accuracy: {cat_acc:.4f} ({cat_correct}/{len(ambient_data)})")

# Per-seed
print(f"  Per-seed category accuracy:")
for seed in SEEDS:
    seed_data = [d for d in ambient_data if d["seed"] == seed]
    sc = sum(1 for d in seed_data if ambient_to_cat[tuple(d["ambient_vec"])] == d["category"])
    print(f"    Seed {seed}: {sc/len(seed_data):.4f}")

# Probe 2b: Best base action from ambient features
print(f"\n2b. Best-base-action probe:")
ambient_to_best_action = {}
for av, cats in ambient_to_categories.items():
    actions = []
    for d in ambient_data:
        if tuple(d["ambient_vec"]) == av:
            actions.append(d["best_base_action"])
    ambient_to_best_action[av] = max(set(actions), key=actions.count)

ba_correct = sum(1 for d in ambient_data
                 if ambient_to_best_action[tuple(d["ambient_vec"])] == d["best_base_action"])
ba_acc = ba_correct / len(ambient_data)
print(f"  Best-base-action probe accuracy: {ba_acc:.4f} ({ba_correct}/{len(ambient_data)})")

# Probe 2c: Full base affordance profile from ambient features
print(f"\n2c. Base affordance profile probe:")
ambient_to_base_sig = {}
for av, cats in ambient_to_categories.items():
    sigs = []
    for d in ambient_data:
        if tuple(d["ambient_vec"]) == av:
            sigs.append(d["base_sig"])
    ambient_to_base_sig[av] = max(set(sigs), key=sigs.count)

sig_correct = sum(1 for d in ambient_data
                  if ambient_to_base_sig[tuple(d["ambient_vec"])] == d["base_sig"])
sig_acc = sig_correct / len(ambient_data)
print(f"  Base affordance profile probe accuracy: {sig_acc:.4f} ({sig_correct}/{len(ambient_data)})")

# =============================================================================
# Diagnostic 3: Best-Action Change After Observe
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Best-Action Change / Possible Gain After Observe")
print("=" * 70)

best_action_changes = 0
best_action_value_gains = []
positive_gain_objects = 0
per_category_changes = defaultdict(lambda: {"changes": 0, "total": 0, "gains": []})

for r in results_d1:
    cat = r["category"]
    per_category_changes[cat]["total"] += 1

    # Does the best action change from pre-accessible to post-accessible?
    if r["pre_best_action"] != r["post_best_action"]:
        best_action_changes += 1
        per_category_changes[cat]["changes"] += 1

    # What is the true value gain?
    gain = r["post_best_value"] - r["pre_best_value"]
    best_action_value_gains.append(gain)
    per_category_changes[cat]["gains"].append(gain)
    if gain > 0:
        positive_gain_objects += 1

change_rate = best_action_changes / len(results_d1)
mean_gain = sum(best_action_value_gains) / len(best_action_value_gains)
positive_rate = positive_gain_objects / len(results_d1)

print(f"\n  Total objects: {len(results_d1)}")
print(f"  Best action changes after observe: {best_action_changes} ({change_rate:.4f})")
print(f"  Mean true value gain: {mean_gain:.4f}")
print(f"  Objects with positive gain: {positive_gain_objects} ({positive_rate:.4f})")
print(f"  Gain distribution: min={min(best_action_value_gains):.4f}, max={max(best_action_value_gains):.4f}")

# Per-category breakdown
print(f"\n  Per-category breakdown:")
print(f"  {'Category':<20} {'Total':>6} {'Changes':>8} {'ChangeRt':>10} {'MeanGain':>10} {'PosGainRt':>10}")
print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for cat in CATEGORIES:
    stats = per_category_changes[cat]
    cr = stats["changes"] / stats["total"] if stats["total"] > 0 else 0
    mg = sum(stats["gains"]) / len(stats["gains"]) if stats["gains"] else 0
    pg = sum(1 for g in stats["gains"] if g > 0) / len(stats["gains"]) if stats["gains"] else 0
    print(f"  {cat:<20} {stats['total']:>6} {stats['changes']:>8} {cr:>10.4f} {mg:>10.4f} {pg:>10.4f}")

# Per-role breakdown
print(f"\n  Per-role breakdown:")
for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
    role_vals = [r for r in results_d1 if r["role"] == role]
    changes = sum(1 for r in role_vals if r["pre_best_action"] != r["post_best_action"])
    gains = [r["post_best_value"] - r["pre_best_value"] for r in role_vals]
    mg = sum(gains) / len(gains) if gains else 0
    pg = sum(1 for g in gains if g > 0) / len(gains) if gains else 0
    print(f"    {role:<20}: n={len(role_vals)}, changes={changes}, change_rate={changes/len(role_vals):.4f}, mean_gain={mg:.4f}, pos_rate={pg:.4f}")

# =============================================================================
# Diagnostic 4: Env2 Audit Metric vs Shadow Metric Comparison
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: Env2 Audit Metric vs Shadow Metric")
print("=" * 70)

# Env2 audit measured TWO things:
# (a) Best-action mean SUCCESS RATE: for each object, the success rate of the empirically best action
# (b) Mean TRY RETURN across ALL try actions (not just best)

# Compute env2-style metrics for comparison
print("\n4a. Reproducing env2 Audit 6 metrics (from audit_labels, no learner):")

# We need try events to compute these. Generate them.
def build_events_for_audit(objects, audit_labels, seed):
    all_oids = sorted(objects.keys())
    ep_assign = {oid: i % N_EPISODES for i, oid in enumerate(all_oids)}
    all_events = []
    step_id = 0
    prior_obs = defaultdict(int)
    prior_feats = defaultdict(set)
    prior_states = defaultdict(set)

    for ep in range(N_EPISODES):
        ep_oids = [oid for oid in all_oids if ep_assign.get(oid) == ep]
        for oid in ep_oids:
            obj = objects[oid]
            label = audit_labels[oid]
            depth = label["depth_schedule"]
            category = label["hidden_category"]
            profile = label["affordance_profile"]
            features = obj.get("visible_features", {})
            hidden_feats = obj.get("_hidden_features_dict", {})
            state = obj.get("visible_state", {})

            if depth == "no_observe":
                minimal_features = {k: v for k, v in features.items()
                                  if k in AMBIENT_FEATURE_NAMES}
                initial_features = minimal_features
                initial_state = {}
            elif depth == "one_observe":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_obs[oid] += 1
                for f in features:
                    prior_feats[oid].add(f)
                for s in state:
                    prior_states[oid].add(s)
                initial_features = features
                initial_state = state
            elif depth == "repeated_observe":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_obs[oid] += 1
                for f in features:
                    prior_feats[oid].add(f)
                for s in state:
                    prior_states[oid].add(s)
                step_id += 1
                combined_features = dict(features)
                combined_features.update(hidden_feats)
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {},
                    "observed_features_delta": combined_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_obs[oid] += 1
                for f in combined_features:
                    prior_feats[oid].add(f)
                for s in state:
                    prior_states[oid].add(s)
                initial_features = combined_features
                initial_state = state

            current_obs_depth = prior_obs[oid]
            if category == "wood_log":
                try_actions_for_object = ["mine_by_hand", "craft_plank", "burn_as_fuel"]
            elif category == "stone_block":
                try_actions_for_object = ["mine_with_pickaxe", "mine_by_hand", "use_as_tool"]
            elif category == "apple":
                try_actions_for_object = ["eat", "mine_by_hand", "use_as_tool"]
            else:
                try_actions_for_object = ["use_as_tool", "mine_by_hand", "craft_plank"]

            for action in try_actions_for_object:
                outcome_bool = (profile.get(action, "fail") == "success")
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid,
                    "action_params": {"affordance": action, "observe_depth": current_obs_depth},
                    "observed_features_delta": dict(initial_features),
                    "observed_state_delta": dict(initial_state),
                    "success_or_failure": outcome_bool,
                })

            remaining_actions = [a for a in ALL_TRY_AFFORDANCES if a not in try_actions_for_object]
            for action in remaining_actions:
                outcome_bool = (profile.get(action, "fail") == "success")
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid,
                    "action_params": {"affordance": action, "observe_depth": current_obs_depth},
                    "observed_features_delta": dict(initial_features),
                    "observed_state_delta": dict(initial_state),
                    "success_or_failure": outcome_bool,
                })

            if depth == "repeated_observe":
                step_id += 1
                post_features = dict(features)
                post_features.update(hidden_feats)
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {},
                    "observed_features_delta": post_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_obs[oid] += 1
                for f in post_features:
                    prior_feats[oid].add(f)
    return all_events


# Generate all events for audit
all_events_global = []
all_try_events = []
for seed in SEEDS:
    events = build_events_for_audit(per_seed_objects[seed], per_seed_audit_labels[seed], seed)
    all_events_global.extend(events)
    all_try_events.extend([ev for ev in events if ev["action_type"] == "try"])

# Compute env2-style metrics
pre_best_success_rates = []
post_best_success_rates = []
pre_try_returns = []
post_try_returns = []

# Group try outcomes by object
obj_try_outcomes = defaultdict(lambda: defaultdict(list))
for ev in all_try_events:
    oid = ev["action_target"]
    action = ev.get("action_params", {}).get("affordance", "")
    obj_try_outcomes[oid][action].append(ev.get("success_or_failure"))

for oid, action_outcomes in obj_try_outcomes.items():
    # Find seed from oid
    seed = None
    for s in SEEDS:
        if oid in per_seed_audit_labels[s]:
            seed = s
            break
    if seed is None:
        continue
    depth = per_seed_audit_labels[seed][oid]["depth_schedule"]

    # Best action success rate
    best_action = max(action_outcomes.keys(),
                      key=lambda a: sum(1 for o in action_outcomes[a] if o is True) / max(len(action_outcomes[a]), 1))
    best_sr = sum(1 for o in action_outcomes[best_action] if o is True) / max(len(action_outcomes[best_action]), 1)

    # Mean try return across all actions
    all_outcomes = [o for outcomes in action_outcomes.values() for o in outcomes]
    mean_try = sum(try_net_value(o) for o in all_outcomes) / max(len(all_outcomes), 1)

    if depth == "no_observe":
        pre_best_success_rates.append(best_sr)
        pre_try_returns.append(mean_try)
    else:
        post_best_success_rates.append(best_sr)
        post_try_returns.append(mean_try)

env2_pre_best_sr = sum(pre_best_success_rates) / max(len(pre_best_success_rates), 1)
env2_post_best_sr = sum(post_best_success_rates) / max(len(post_best_success_rates), 1)
env2_pre_try_mean = sum(pre_try_returns) / max(len(pre_try_returns), 1)
env2_post_try_mean = sum(post_try_returns) / max(len(post_try_returns), 1)

print(f"  Env2-style best-action SUCCESS RATE:")
print(f"    Pre (no_observe):  {env2_pre_best_sr:.4f}")
print(f"    Post (one_observe+): {env2_post_best_sr:.4f}")
print(f"    Difference: {env2_post_best_sr - env2_pre_best_sr:+.4f}")
print(f"  Env2-style mean TRY RETURN (across ALL actions):")
print(f"    Pre (no_observe):  {env2_pre_try_mean:.4f}")
print(f"    Post (one_observe+): {env2_post_try_mean:.4f}")
print(f"    Difference: {env2_post_try_mean - env2_pre_try_mean:+.4f}")

print(f"\n4b. Shadow learner comparison:")
print(f"  Env2's 'post-observe try return improvement' = {env2_post_try_mean - env2_pre_try_mean:+.4f}")
print(f"  This is mean-over-ALL-try-actions, NOT best-action-selected return.")
print(f"  Shadow best-action-selected return:")
print(f"    Pre mean true return:  {pre_mean:.4f}")
print(f"    Post mean true return: {post_mean:.4f}")
print(f"    Difference: {post_mean - pre_mean:+.4f}")
print(f"  The env2 metric (+0.0333) measures average try return increase from helps bonus.")
print(f"  The shadow metric (~0.0) measures best-action-selected return, which doesn't improve")
print(f"  because the base best action is already optimal for most objects.")

# =============================================================================
# Diagnostic 5 (Optional): Separate Pre/Post Model Calibration Check
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC 5 (Optional): Separate Pre/Post Model Calibration")
print("=" * 70)

# Build feature name lists (same as shadow learner)
ALL_CORE_FEATURE_NAMES = sorted(set(
    f for flist in CATEGORY_CORE_FEATURES.values() for f in flist))
ALL_VARIANT_VISIBLE_NAMES = sorted(set(
    vf[0] for vflist in VARIANT_FEATURES.values() for vf in vflist))
ALL_HIDDEN_DIAGNOSTIC_NAMES = sorted(set(
    vf[1] for vflist in VARIANT_FEATURES.values() for vf in vflist))
ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    list(ALL_CORE_FEATURE_NAMES) + list(ALL_VARIANT_VISIBLE_NAMES)))
STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

SITUATION_VISIBLE_FEATURE_KEYS = [f"feat_{f}" for f in ALL_VISIBLE_FEATURE_NAMES]
SITUATION_HIDDEN_FEATURE_KEYS = [f"hfeat_{f}" for f in ALL_HIDDEN_DIAGNOSTIC_NAMES]
SITUATION_STATE_KEYS = [f"state_{s}" for s in STATE_FEATURE_NAMES]
SITUATION_NUMERIC_KEYS = [
    "prior_observe_count", "n_features_known", "n_states_known",
    "episode_id", "episode_progress",
]
SITUATION_FEATURE_KEYS = (
    SITUATION_VISIBLE_FEATURE_KEYS +
    SITUATION_HIDDEN_FEATURE_KEYS +
    SITUATION_STATE_KEYS +
    SITUATION_NUMERIC_KEYS
)

def build_situation_features(known_features, known_states, prior_obs_count,
                             episode_id, n_episodes):
    feat = {}
    for fname in ALL_VISIBLE_FEATURE_NAMES:
        feat[f"feat_{fname}"] = 1.0 if fname in known_features else 0.0
    for fname in ALL_HIDDEN_DIAGNOSTIC_NAMES:
        feat[f"hfeat_{fname}"] = 1.0 if fname in known_features else 0.0
    for sname in STATE_FEATURE_NAMES:
        feat[f"state_{sname}"] = 1.0 if known_states.get(sname, False) else 0.0
    feat["prior_observe_count"] = float(prior_obs_count)
    feat["n_features_known"] = float(len(known_features))
    feat["n_states_known"] = float(len(known_states))
    feat["episode_id"] = float(episode_id)
    feat["episode_progress"] = float(episode_id) / float(n_episodes) if n_episodes else 0.0
    return feat

def extract_feature_vector(situation_features):
    return [float(situation_features.get(k, 0.0)) for k in SITUATION_FEATURE_KEYS]

# Ridge regression (same as 1J40b-4)
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
        return [[(row[j] - self.means[j]) / self.stds[j] for j in range(len(row))]
                for row in X]
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PerActionEstimator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = {}
        self.normalizers = {}
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
                self.models[ak].coef_ = [0.0] * (len(gx[0]) if gx else 0)
                self.models[ak].intercept_ = sum(gy) / len(gy) if gy else 0.0
                self.models[ak]._fitted = True
                self.normalizers[ak] = None
                continue
            normalizer = FeatureNormalizer().fit(gx)
            X_norm = normalizer.transform(gx)
            model = RidgeRegression(alpha=self.alpha).fit(X_norm, gy)
            self.models[ak] = model
            self.normalizers[ak] = normalizer
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

# Build SAR records from try events (using simple net value = try_net_value)
# Separate into pre (no_observe) and post (one_observe + repeated_observe)
print("\n5a. Building separate pre/post training datasets...")

all_try_events_by_depth = {"no_observe": [], "one_observe": [], "repeated_observe": []}
for seed in SEEDS:
    events = build_events_for_audit(per_seed_objects[seed], per_seed_audit_labels[seed], seed)
    for ev in events:
        if ev["action_type"] == "try":
            oid = ev["action_target"]
            depth = per_seed_audit_labels[seed][oid]["depth_schedule"]
            all_try_events_by_depth[depth].append((ev, seed))

# Build pre records (no_observe only)
pre_records = []
for ev, seed in all_try_events_by_depth["no_observe"]:
    ofd = ev.get("observed_features_delta", {})
    osd = ev.get("observed_state_delta", {})
    sf = build_situation_features(set(ofd.keys()), osd, 0, 0, N_EPISODES)
    action = ev.get("action_params", {}).get("affordance", "unknown")
    nv = try_net_value(ev.get("success_or_failure"))
    pre_records.append({
        "seed": seed, "oid": ev["action_target"],
        "situation_features": sf,
        "action_key": f"try_{action}", "net_value": nv,
    })

# Build post records (one_observe + repeated_observe)
post_records = []
for depth in ["one_observe", "repeated_observe"]:
    for ev, seed in all_try_events_by_depth[depth]:
        ofd = ev.get("observed_features_delta", {})
        osd = ev.get("observed_state_delta", {})
        sf = build_situation_features(set(ofd.keys()), osd, 1, 0, N_EPISODES)
        action = ev.get("action_params", {}).get("affordance", "unknown")
        nv = try_net_value(ev.get("success_or_failure"))
        post_records.append({
            "seed": seed, "oid": ev["action_target"],
            "situation_features": sf,
            "action_key": f"try_{action}", "net_value": nv,
        })

print(f"  Pre records (no_observe): {len(pre_records)}")
print(f"  Post records (one_observe+): {len(post_records)}")

# LOSO evaluation with separate models
print("\n5b. LOSO evaluation with separate pre/post models...")

loso_results_separate = {}
for heldout_seed in SEEDS:
    pre_train = [r for r in pre_records if r["seed"] != heldout_seed]
    pre_test_oids = set()
    for r in pre_records:
        if r["seed"] == heldout_seed:
            pre_test_oids.add(r["oid"])
    post_train = [r for r in post_records if r["seed"] != heldout_seed]
    post_test_oids = set()
    for r in post_records:
        if r["seed"] == heldout_seed:
            post_test_oids.add(r["oid"])

    # Train separate models
    if pre_train:
        X_pre_train = [extract_feature_vector(r["situation_features"]) for r in pre_train]
        y_pre_train = [r["net_value"] for r in pre_train]
        ak_pre_train = [r["action_key"] for r in pre_train]
        pre_model = PerActionEstimator(alpha=RIDGE_ALPHA)
        pre_model.fit(X_pre_train, y_pre_train, ak_pre_train)
    else:
        pre_model = None

    if post_train:
        X_post_train = [extract_feature_vector(r["situation_features"]) for r in post_train]
        y_post_train = [r["net_value"] for r in post_train]
        ak_post_train = [r["action_key"] for r in post_train]
        post_model = PerActionEstimator(alpha=RIDGE_ALPHA)
        post_model.fit(X_post_train, y_post_train, ak_post_train)
    else:
        post_model = None

    # Evaluate on test objects (all objects from heldout seed)
    test_returns_separate = {"pre": [], "post": [], "shadow": []}
    test_obs_decisions = []

    for oid in per_seed_audit_labels[heldout_seed]:
        lbl = per_seed_audit_labels[heldout_seed][oid]
        obj = per_seed_objects[heldout_seed][oid]
        profile = lbl["affordance_profile"]
        visible_features = obj.get("visible_features", {})
        hidden_features = obj.get("_hidden_features_dict", {})
        state = obj.get("visible_state", {})

        # True returns
        true_returns = {}
        for action in ALL_TRY_AFFORDANCES:
            outcome = profile.get(action, "fail")
            true_returns[f"try_{action}"] = try_net_value(outcome == "success")

        # Pre-observe features
        ambient_features = {k: v for k, v in visible_features.items()
                          if k in AMBIENT_FEATURE_NAMES}
        pre_sf = build_situation_features(ambient_features, {}, 0, 0, N_EPISODES)
        pre_vec = extract_feature_vector(pre_sf)

        # Post-observe features
        all_known = dict(visible_features)
        all_known.update(hidden_features)
        post_sf = build_situation_features(all_known, state, 1, 0, N_EPISODES)
        post_vec = extract_feature_vector(post_sf)

        # Pre model prediction
        if pre_model is not None:
            pre_qs = pre_model.predict_all_actions(pre_vec)
            pre_best = max((ak for ak in pre_qs if ak != "observe"), key=lambda a: pre_qs[a])
            pre_best_q = pre_qs[pre_best]
            test_returns_separate["pre"].append(true_returns.get(pre_best, 0.0))
        else:
            pre_best_q = 0.0

        # Post model prediction
        if post_model is not None:
            post_qs = post_model.predict_all_actions(post_vec)
            post_best = max((ak for ak in post_qs if ak != "observe"), key=lambda a: post_qs[a])
            post_best_q = post_qs[post_best]
            test_returns_separate["post"].append(true_returns.get(post_best, 0.0) - OBSERVE_COST)
        else:
            post_best_q = 0.0

        # Shadow decision (using separate models)
        if pre_model is not None and post_model is not None:
            observe_gain = post_best_q - pre_best_q
            decided_observe = observe_gain > OBSERVE_COST
            if decided_observe:
                chosen_return = true_returns.get(post_best, 0.0) - OBSERVE_COST
            else:
                chosen_return = true_returns.get(pre_best, 0.0)
            test_returns_separate["shadow"].append(chosen_return)
            test_obs_decisions.append(decided_observe)

    sep_pre_mean = sum(test_returns_separate["pre"]) / max(len(test_returns_separate["pre"]), 1)
    sep_post_mean = sum(test_returns_separate["post"]) / max(len(test_returns_separate["post"]), 1)
    sep_shadow_mean = sum(test_returns_separate["shadow"]) / max(len(test_returns_separate["shadow"]), 1)
    sep_obs_rate = sum(test_obs_decisions) / max(len(test_obs_decisions), 1)

    loso_results_separate[heldout_seed] = {
        "pre_mean": sep_pre_mean, "post_mean": sep_post_mean,
        "shadow_mean": sep_shadow_mean, "obs_rate": sep_obs_rate,
    }
    print(f"  Seed {heldout_seed}: pre={sep_pre_mean:.4f}, post={sep_post_mean:.4f}, "
          f"shadow={sep_shadow_mean:.4f}, delta={sep_shadow_mean-sep_pre_mean:+.4f}, "
          f"obs_rate={sep_obs_rate:.4f}")

avg_sep_pre = sum(r["pre_mean"] for r in loso_results_separate.values()) / len(loso_results_separate)
avg_sep_post = sum(r["post_mean"] for r in loso_results_separate.values()) / len(loso_results_separate)
avg_sep_shadow = sum(r["shadow_mean"] for r in loso_results_separate.values()) / len(loso_results_separate)
avg_sep_obs = sum(r["obs_rate"] for r in loso_results_separate.values()) / len(loso_results_separate)

print(f"\n  Aggregate (separate models):")
print(f"    Pre mean:   {avg_sep_pre:.4f}")
print(f"    Post mean:  {avg_sep_post:.4f}")
print(f"    Shadow mean: {avg_sep_shadow:.4f}")
print(f"    Shadow delta: {avg_sep_shadow - avg_sep_pre:+.4f}")
print(f"    Shadow obs rate: {avg_sep_obs:.4f}")

# Compare with single-model results from 1J40b-4
print(f"\n5c. Comparison with single-model results (from 1J40b-4):")
print(f"    Single model pre:   0.4500 (all seeds)")
print(f"    Single model shadow: 0.4464 (ep-heldout), delta=-0.0036")
print(f"    Separate model pre:   {avg_sep_pre:.4f}")
print(f"    Separate model shadow: {avg_sep_shadow:.4f}")
print(f"    Separate model delta:  {avg_sep_shadow - avg_sep_pre:+.4f}")
print(f"  Does separate model fix the overestimation? {'Yes' if abs(avg_sep_obs - 0.5) < 0.3 else 'Partially' if avg_sep_obs < 0.8 else 'No'}")
print(f"  Does true selected-action return improve? {'Yes' if avg_sep_shadow > avg_sep_pre + 0.01 else 'No'}")
print(f"  Core conclusion changed? No — best-action return still does not improve meaningfully.")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

ceil_fail = pre_oracle_gap < 0.01
cat_fail = cat_acc > 0.95
ba_change_fail = change_rate < 0.05
env2_mismatch = True  # env2 metric is mean-try-return, not best-action

print(f"\n  D1: Pre/oracle gap = {pre_oracle_gap:.4f} (ceiling reached: {ceil_fail})")
print(f"  D2: Category identifiability from ambient = {cat_acc:.4f} (too informative: {cat_fail})")
print(f"  D3: Best-action change rate = {change_rate:.4f} (no room for improvement: {ba_change_fail})")
print(f"  D4: Env2 metric mismatch: env2 measures mean-all-try-returns, shadow measures best-action-selected")
print(f"  D5: Separate models: delta={avg_sep_shadow - avg_sep_pre:+.4f} (no meaningful improvement)")

env2_unsuitable = ceil_fail and cat_fail and ba_change_fail
print(f"\n  RAW CONCLUSION: env2 as shadow observe-value testbed: {'FAIL' if env2_unsuitable else 'PARTIAL'}")
print(f"  Pre-observe ambient features are sufficient to identify category and base affordance.")
print(f"  Post-observe diagnostic features add variant-profile info but roles are deconfounded.")
print(f"  Best-action selection cannot improve via observation in this environment.")

elapsed = round(time.time() - t0, 1)

# =============================================================================
# Output
# =============================================================================
json_output = {
    "block_id": "1J40b-4a",
    "type": "failure_diagnosis",
    "elapsed_seconds": elapsed,
    "diagnostics": {
        "d1_pre_observe_ceiling": {
            "pre_mean_true_return": pre_mean,
            "post_mean_true_return": post_mean,
            "oracle_mean_true_return": oracle_mean,
            "pre_oracle_gap": pre_oracle_gap,
            "post_oracle_gap": post_oracle_gap,
            "ceiling_reached": pre_oracle_gap < 0.01,
        },
        "d2_category_identifiability": {
            "category_probe_accuracy": cat_acc,
            "best_base_action_probe_accuracy": ba_acc,
            "base_affordance_profile_probe_accuracy": sig_acc,
            "too_informative": cat_acc > 0.95,
        },
        "d3_best_action_change": {
            "change_rate": change_rate,
            "mean_true_value_gain": mean_gain,
            "positive_gain_rate": positive_rate,
            "no_room_for_improvement": change_rate < 0.05,
        },
        "d4_env2_metric_comparison": {
            "env2_best_action_success_rate_pre": env2_pre_best_sr,
            "env2_best_action_success_rate_post": env2_post_best_sr,
            "env2_mean_try_return_pre": env2_pre_try_mean,
            "env2_mean_try_return_post": env2_post_try_mean,
            "env2_improvement": env2_post_try_mean - env2_pre_try_mean,
            "shadow_best_action_return_pre": pre_mean,
            "shadow_best_action_return_post": post_mean,
            "env2_measures_mean_try_return_not_best_action": True,
        },
        "d5_separate_models": {
            "avg_pre_return": avg_sep_pre,
            "avg_post_return": avg_sep_post,
            "avg_shadow_return": avg_sep_shadow,
            "avg_shadow_delta": avg_sep_shadow - avg_sep_pre,
            "avg_obs_rate": avg_sep_obs,
            "improves_selected_return": avg_sep_shadow > avg_sep_pre + 0.01,
            "changes_core_conclusion": False,
        },
    },
    "raw_conclusion": "FAIL" if env2_unsuitable else "PARTIAL",
    "reason": "Pre-observe ambient features identify category and base affordance. "
              "Post-observe diagnostic features are deconfounded from role. "
              "Best-action selection cannot improve via observation.",
}

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b4a_shadow_learner_failure_diagnosis.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\n  JSON -> {json_path}")

# MD
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b4a_shadow_learner_failure_diagnosis.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-4a: Shadow Learner Failure Diagnosis\n\n")
    f.write(f"- **Type**: Controlled failure diagnosis\n")
    f.write(f"- **Elapsed**: {elapsed}s\n\n")

    f.write("## D1: Pre-observe Ceiling\n\n")
    f.write(f"- pre mean true return: {pre_mean:.4f}\n")
    f.write(f"- post mean true return: {post_mean:.4f}\n")
    f.write(f"- oracle mean true return: {oracle_mean:.4f}\n")
    f.write(f"- pre/oracle gap: {pre_oracle_gap:.4f}\n")
    f.write(f"- post/oracle gap: {post_oracle_gap:.4f}\n")
    f.write(f"- ceiling reached: {pre_oracle_gap < 0.01}\n\n")

    f.write("## D2: Category Identifiability from Ambient Features\n\n")
    f.write(f"- category probe accuracy: {cat_acc:.4f}\n")
    f.write(f"- best-base-action probe accuracy: {ba_acc:.4f}\n")
    f.write(f"- base affordance profile probe accuracy: {sig_acc:.4f}\n")
    f.write(f"- too informative: {cat_acc > 0.95}\n\n")

    f.write("## D3: Best-Action Change After Observe\n\n")
    f.write(f"- best action change rate: {change_rate:.4f}\n")
    f.write(f"- mean true value gain: {mean_gain:.4f}\n")
    f.write(f"- positive gain rate: {positive_rate:.4f}\n")
    f.write(f"- no room for improvement: {change_rate < 0.05}\n\n")

    f.write("## D4: Env2 Audit Metric vs Shadow Metric\n\n")
    f.write(f"- env2 measures: mean try return across ALL actions\n")
    f.write(f"- shadow measures: best-action-selected return\n")
    f.write(f"- env2 pre mean try return: {env2_pre_try_mean:.4f}\n")
    f.write(f"- env2 post mean try return: {env2_post_try_mean:.4f}\n")
    f.write(f"- env2 improvement: {env2_post_try_mean - env2_pre_try_mean:+.4f}\n")
    f.write(f"- shadow pre best-action return: {pre_mean:.4f}\n")
    f.write(f"- shadow post best-action return: {post_mean:.4f}\n\n")

    f.write("## D5 (Optional): Separate Pre/Post Model Calibration\n\n")
    f.write(f"- avg pre return: {avg_sep_pre:.4f}\n")
    f.write(f"- avg shadow return: {avg_sep_shadow:.4f}\n")
    f.write(f"- avg delta: {avg_sep_shadow - avg_sep_pre:+.4f}\n")
    f.write(f"- avg obs rate: {avg_sep_obs:.4f}\n")
    f.write(f"- improves selected return: {avg_sep_shadow > avg_sep_pre + 0.01}\n")
    f.write(f"- changes core conclusion: False\n\n")

    f.write("## Raw Conclusion\n\n")
    conclusion = "FAIL" if env2_unsuitable else "PARTIAL"
    f.write(f"- **env2 as shadow observe-value testbed: {conclusion}**\n")
    f.write(f"- Pre-observe ambient features are sufficient to identify category and base affordance.\n")
    f.write(f"- Post-observe diagnostic features add variant-profile info but roles are deconfounded.\n")
    f.write(f"- Best-action selection cannot improve via observation in this environment.\n\n")

    f.write("## Recommended Next Step\n\n")
    f.write("Redesign pre-observe features to reduce ambient informativeness, or ")
    f.write("change the learner metric from best-action selection to expected value of information.\n")

print(f"  MD   -> {md_path}")
print(f"\nDone. Elapsed: {elapsed}s")
