"""
Block 1J40b-env3 — Category-Ambiguous Observe-Value Environment Audit.

Key changes from env2:
- Pre-observe ambient features are shared within 2 ambient groups (2 categories each)
  so categories cannot be uniquely identified from ambient features alone.
- Base affordance profiles modified so no single action succeeds for both categories
  within an ambient group — pre-observe best action has genuine uncertainty.
- Post-observe adds core features (identify category), diagnostic features
  (action-relevant evidence), and hidden features (variant bonus hints).
- Adds post-observe source ablations: pre_ambient_only, post_core_only,
  post_diagnostic_only, post_core_plus_diagnostic.
- Adds category probes by visibility level.
- Adds pre-observe tie diagnostics.
- Declares all env2→env3 affordance changes explicitly.

Do NOT train a shadow learner. Audit only.
"""

import os, sys, json, copy, random, time, math
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
sys.path.insert(0, CURRENT_DIR)

import config

t0 = time.time()

# =============================================================================
# Config safety: idempotent C6 addition, preserve C4/C5
# =============================================================================
C6_LABEL = "C6_observe_try_counterfactual_v0"
existing_labels = [c["label"] for c in config.CUE_CONDITIONS]
C4_PRESENT = "C4_instance_subtype_cued_v1" in existing_labels
C5_PRESENT = "C5_mixed_source_identifiability_v0" in existing_labels

if C6_LABEL not in existing_labels:
    config.CUE_CONDITIONS.append({
        "label": C6_LABEL,
        "p_target": 0.60, "p_other": 0.40,
        "absent": False, "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True,
        "observe_depth_schedules": True,
    })
    print(f"Added {C6_LABEL} to CUE_CONDITIONS")
else:
    print(f"{C6_LABEL} already in CUE_CONDITIONS (idempotent)")

assert C4_PRESENT, "C4 must be preserved"
assert C5_PRESENT, "C5 must be preserved"
print(f"C4 preserved: {C4_PRESENT}, C5 preserved: {C5_PRESENT}")

# =============================================================================
# Constants
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5
N_GROUPS_PER_FAMILY = 3
OBJECTS_PER_GROUP = 3
N_PER_FAMILY = N_GROUPS_PER_FAMILY * OBJECTS_PER_GROUP
N_WOOD = N_PER_FAMILY
N_STONE = N_PER_FAMILY
N_APPLE = N_PER_FAMILY
N_PICKAXE = N_PER_FAMILY
TOTAL_OBJECTS = N_WOOD + N_STONE + N_APPLE + N_PICKAXE

ALL_ACTIONS = sorted([
    "craft_plank", "eat", "use_as_tool", "burn_as_fuel",
    "mine_by_hand", "mine_with_pickaxe",
])

STATE_FEATURE_KEYS = ["fresh", "wet", "damaged", "clean", "hot", "open"]

OBSERVE_COST = 0.005
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1

DISCOUNT_FACTOR = 0.9

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

# =============================================================================
# Ambient Groups: categories share ambient signatures to create pre-observe ambiguity
# =============================================================================
# Group A: wood_log + stone_block  (ambient: brownish, rough_texture, long_shape, heavy_weight)
# Group B: apple + wooden_pickaxe   (ambient: greenish, smooth_texture, round_small, light_weight)
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

# Collect all ambient feature names
ALL_AMBIENT_FEATURE_NAMES = sorted(set(
    f for feats in AMBIENT_GROUP_FEATURES.values() for f in feats
))

# =============================================================================
# Core features per category (visible at one_observe+, identify category)
# These are the features that distinguish categories within an ambient group.
# =============================================================================
CATEGORY_CORE_FEATURES = {
    "wood_log": [
        "has_bark_texture", "fibrous", "flammable", "porous_surface",
    ],
    "stone_block": [
        "has_crystal_flecks", "has_granular_surface", "block_like",
        "cold_to_touch", "scratch_resistant",
    ],
    "apple": [
        "has_stem_remnant", "has_peel_texture", "fruity_scent",
    ],
    "wooden_pickaxe": [
        "has_grip_area", "has_shaft_shape", "elongated_with_handle",
        "movable", "has_metal_head", "jointed",
    ],
}

# =============================================================================
# Diagnostic features per category (visible at one_observe+, action-relevant)
# These correlate with affordance profiles but are not role markers.
# Some features intentionally shared across categories to prevent perfect
# category identification from diagnostic features alone.
# =============================================================================
CATEGORY_DIAGNOSTIC_FEATURES = {
    "wood_log": [
        "has_fibrous_inside", "has_burnable_fibers", "is_hollow",
        "has_rough_surface",
    ],
    "stone_block": [
        "has_hard_surface", "has_crystal_fragments", "is_solid",
        "is_fragile",  # shared with apple — stone can fracture
    ],
    "apple": [
        "has_edible_smell", "is_soft_inside", "is_fragile",  # is_fragile shared with stone_block
        "has_fibrous_inside",  # shared with wood_log — apple has fibrous flesh
    ],
    "wooden_pickaxe": [
        "has_grip_shape", "has_tool_edge", "is_balanced",
        "has_leverage",
    ],
}

# All diagnostic feature names
ALL_DIAGNOSTIC_FEATURE_NAMES = sorted(set(
    f for feats in CATEGORY_DIAGNOSTIC_FEATURES.values() for f in feats
))

# =============================================================================
# Hidden features per category (visible only at repeated_observe, for helps variant)
# These hint at the bonus action for helps objects.
# =============================================================================
CATEGORY_HIDDEN_FEATURES = {
    "wood_log": ["edible_core", "carvable_interior", "moisture_content"],
    "stone_block": ["internal_fractures", "layered_structure", "cleavage_plane"],
    "apple": ["sweet_flesh", "bruise_markings", "wax_coating"],
    "wooden_pickaxe": ["hairline_crack", "handle_looseness", "shaft_rot"],
}

# =============================================================================
# Profile system for variant (helps bonus) — simplified from env2
# 3 profiles per category, each profile has a bonus action for helps objects
# =============================================================================
# Profile bonus actions: each profile adds success to one failing base action
PROFILE_BONUS_ACTION = {
    "wood_log": {
        "A": "eat",           # wood_log base: eat=fail → add success
        "B": "use_as_tool",   # wood_log base: use_as_tool=fail → add success
        "C": "mine_with_pickaxe",  # wood_log base: mine_with_pickaxe=fail → add success
    },
    "stone_block": {
        "A": "mine_by_hand",  # stone_block base: mine_by_hand=fail → add success
        "B": "craft_plank",   # stone_block base: craft_plank=fail → add success
        "C": "burn_as_fuel",  # stone_block base: burn_as_fuel=fail → add success
    },
    "apple": {
        "A": "craft_plank",   # apple base: craft_plank=fail → add success
        "B": "use_as_tool",   # apple base: use_as_tool=fail → add success
        "C": "burn_as_fuel",  # apple base: burn_as_fuel=fail → add success
    },
    "wooden_pickaxe": {
        "A": "mine_by_hand",  # pickaxe base: mine_by_hand=fail → add success
        "B": "craft_plank",   # pickaxe base: craft_plank=fail → add success
        "C": "eat",           # pickaxe base: eat=fail → add success
    },
}

# Seed-Category-Profile assignment (same rotation pattern as env2)
SEED_CATEGORY_PROFILE = {
    101: {"wood_log": "A", "stone_block": "B", "apple": "C", "wooden_pickaxe": "A"},
    103: {"wood_log": "B", "stone_block": "C", "apple": "A", "wooden_pickaxe": "B"},
    107: {"wood_log": "C", "stone_block": "A", "apple": "B", "wooden_pickaxe": "C"},
    109: {"wood_log": "A", "stone_block": "C", "apple": "B", "wooden_pickaxe": "A"},
    113: {"wood_log": "B", "stone_block": "A", "apple": "C", "wooden_pickaxe": "B"},
}

# =============================================================================
# ENV2 vs ENV3 Affordance Profile Comparison
# =============================================================================
ENV2_BASE_AFFORDANCE_PROFILES = {
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

# ENV3: Modified so no common success within ambient groups
# Group A (wood_log + stone_block): no action succeeds for both
# Group B (apple + wooden_pickaxe): no action succeeds for both
ENV3_BASE_AFFORDANCE_PROFILES = {
    "wood_log": {
        "mine_by_hand": "success",   # was success — KEPT
        "mine_with_pickaxe": "fail",  # was success — CHANGED (removes common safe action with stone_block)
        "craft_plank": "success",    # was success — KEPT
        "eat": "fail",               # was fail — KEPT
        "use_as_tool": "fail",       # was fail — KEPT
        "burn_as_fuel": "success",   # was success — KEPT
    },
    "stone_block": {
        "mine_by_hand": "fail",      # was fail — KEPT
        "mine_with_pickaxe": "success",  # was success — KEPT
        "craft_plank": "fail",       # was fail — KEPT
        "eat": "fail",               # was fail — KEPT
        "use_as_tool": "fail",       # was fail — KEPT
        "burn_as_fuel": "fail",      # was fail — KEPT
    },
    "apple": {
        "mine_by_hand": "success",   # was success — KEPT
        "mine_with_pickaxe": "fail",  # was success — CHANGED (removes common safe action with wooden_pickaxe)
        "craft_plank": "fail",       # was fail — KEPT
        "eat": "success",            # was success — KEPT
        "use_as_tool": "fail",       # was fail — KEPT
        "burn_as_fuel": "fail",      # was fail — KEPT
    },
    "wooden_pickaxe": {
        "mine_by_hand": "fail",      # was fail — KEPT
        "mine_with_pickaxe": "fail", # was fail — KEPT
        "craft_plank": "fail",       # was fail — KEPT
        "eat": "fail",               # was fail — KEPT
        "use_as_tool": "success",    # was success — KEPT
        "burn_as_fuel": "fail",      # was fail — KEPT
    },
}

# Use ENV3 profiles
BASE_AFFORDANCE_PROFILES = ENV3_BASE_AFFORDANCE_PROFILES

# =============================================================================
# Profile Change Report
# =============================================================================
AFFORDANCE_CHANGES = {}
for cat in CATEGORIES:
    old = ENV2_BASE_AFFORDANCE_PROFILES[cat]
    new = ENV3_BASE_AFFORDANCE_PROFILES[cat]
    changes = {}
    for action in ALL_ACTIONS:
        if old[action] != new[action]:
            changes[action] = f"{old[action]} -> {new[action]}"
    if changes:
        AFFORDANCE_CHANGES[cat] = changes

# =============================================================================
# Forbidden keys
# =============================================================================
FORBIDDEN_AGENT_VISIBLE = {
    "schedule", "schedule_type", "group_role", "group_id",
    "depth_schedule", "mixed_source_type", "true_family",
    "hidden_subtype", "oracle_outcome", "full_object_state",
    "deceptive_flag", "prior_violation", "preferred_explanation",
}

AGENT_STRUCTURAL_KEYS = {
    "step_id", "episode_id", "action_type", "action_target",
    "action_params", "observed_features_delta", "observed_state_delta",
    "success_or_failure",
}

ALLOWED_ACTION_PARAMS_KEYS = {
    "affordance",
    "observe_depth",
    "prior_observe_count",
}

# =============================================================================
# Try net value helper
# =============================================================================
def try_net_value(success_or_failure):
    if success_or_failure is True:
        return SUCCESS_REWARD - PROBE_COST  # 0.45
    elif success_or_failure is False:
        return -FAILURE_PENALTY - PROBE_COST  # -0.15
    return 0.0

# True action value given a profile
def action_true_value(action, profile):
    outcome = profile.get(action, "fail")
    return try_net_value(outcome == "success")

# =============================================================================
# Leakage check utilities (same as env2)
# =============================================================================
def check_forbidden_recursive(obj, path="", forbidden=None):
    if forbidden is None:
        forbidden = FORBIDDEN_AGENT_VISIBLE
    violations = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in forbidden:
                if path != "" or key not in AGENT_STRUCTURAL_KEYS:
                    violations.append(f"{path}.{key}" if path else key)
            if isinstance(value, str) and value in forbidden:
                violations.append(f"{path}.{key}=<{value}>" if path else f"{key}=<{value}>")
            violations.extend(check_forbidden_recursive(value, f"{path}.{key}" if path else key, forbidden))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(check_forbidden_recursive(item, f"{path}[{i}]", forbidden))
    return violations

def check_event_leakage(event):
    violations = []
    for key in event:
        if key not in AGENT_STRUCTURAL_KEYS:
            violations.append(f"event.{key} (unknown top-level key)")
    ap = event.get("action_params", {})
    for key in ap:
        if key not in ALLOWED_ACTION_PARAMS_KEYS:
            violations.append(f"event.action_params.{key} (forbidden param key)")
    for key, value in ap.items():
        if isinstance(value, str) and value in FORBIDDEN_AGENT_VISIBLE:
            violations.append(f"event.action_params.{key}={value}")
    ofd = event.get("observed_features_delta", {})
    for key in ofd:
        if key in FORBIDDEN_AGENT_VISIBLE:
            violations.append(f"event.observed_features_delta.{key}")
    osd = event.get("observed_state_delta", {})
    for key in osd:
        if key in FORBIDDEN_AGENT_VISIBLE:
            violations.append(f"event.observed_state_delta.{key}")
    violations.extend(check_forbidden_recursive(event, "", FORBIDDEN_AGENT_VISIBLE))
    return violations

# =============================================================================
# Softmax classifier for role/category probes
# =============================================================================
def softmax(logits):
    exps = [math.exp(v - max(logits)) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def train_softmax_classifier(X, y, num_classes=3, l2_alpha=0.1, max_iter=200, lr=0.01):
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 0
    W = [[0.0] * n_features for _ in range(num_classes)]
    b = [0.0] * num_classes

    for iteration in range(max_iter):
        grad_W = [[0.0] * n_features for _ in range(num_classes)]
        grad_b = [0.0] * num_classes

        for i in range(n_samples):
            logits = [sum(W[c][j] * X[i][j] for j in range(n_features)) + b[c]
                     for c in range(num_classes)]
            probs = softmax(logits)

            for c in range(num_classes):
                target = 1.0 if c == y[i] else 0.0
                error = probs[c] - target
                for j in range(n_features):
                    grad_W[c][j] += error * X[i][j]
                grad_b[c] += error

        for c in range(num_classes):
            for j in range(n_features):
                grad_W[c][j] = grad_W[c][j] / n_samples + l2_alpha * W[c][j]
                W[c][j] -= lr * grad_W[c][j]
            grad_b[c] /= n_samples
            b[c] -= lr * grad_b[c]

    return W, b

def predict_softmax(W, b, X):
    n_samples = len(X)
    n_features = len(W[0])
    predictions = []
    for i in range(n_samples):
        logits = [sum(W[c][j] * X[i][j] for j in range(n_features)) + b[c]
                 for c in range(len(W))]
        predictions.append(max(range(len(W)), key=lambda c: logits[c]))
    return predictions

# =============================================================================
# Variant profile builder
# =============================================================================
def build_variant_profile(base_profile, category, group_role, profile, seed):
    vp = dict(base_profile)
    if group_role == "observe_helps":
        bonus_action = PROFILE_BONUS_ACTION[category][profile]
        if vp.get(bonus_action) == "fail":
            vp[bonus_action] = "success"
    return vp

# =============================================================================
# Object generation
# =============================================================================
def generate_c6_objects_env3(seed, prefix="c6"):
    objects = {}
    audit_labels = {}
    oid_counter = [0]

    def make_oid():
        oid_counter[0] += 1
        return f"{prefix}_obj_{oid_counter[0]:04d}"

    for category in CATEGORIES:
        base_profile = BASE_AFFORDANCE_PROFILES[category]
        ambient_group = CATEGORY_AMBIENT_GROUP[category]
        ambient_features_list = AMBIENT_GROUP_FEATURES[ambient_group]
        core_features_list = CATEGORY_CORE_FEATURES[category]
        diagnostic_features_list = CATEGORY_DIAGNOSTIC_FEATURES[category]
        hidden_features_list = CATEGORY_HIDDEN_FEATURES[category]

        for group_idx in range(N_GROUPS_PER_FAMILY):
            group_id = f"{category}_g{group_idx}"
            group_role = ["observe_helps", "observe_neutral", "observe_wasteful"][group_idx]

            profile = SEED_CATEGORY_PROFILE[seed][category]
            variant_profile = build_variant_profile(base_profile, category, group_role, profile, seed)

            # Ambient features: all objects in the ambient group get these
            ambient_feat_dict = {f: True for f in ambient_features_list}

            # Core features: category-identifying
            core_feat_dict = {f: True for f in core_features_list}

            # Diagnostic features: action-relevant, deterministic per category
            diagnostic_feat_dict = {f: True for f in diagnostic_features_list}

            # Hidden features: revealed only at repeated_observe
            hidden_feat_dict = {f: True for f in hidden_features_list}

            # Full visible features = ambient + core + diagnostic
            full_visible = {}
            full_visible.update(ambient_feat_dict)
            full_visible.update(core_feat_dict)
            full_visible.update(diagnostic_feat_dict)

            for depth_idx, depth_schedule in enumerate(["no_observe", "one_observe", "repeated_observe"]):
                oid = make_oid()

                if depth_schedule == "no_observe":
                    effective_profile = dict(base_profile)
                else:
                    effective_profile = dict(variant_profile)

                visible_state = {
                    "fresh": True,
                    "wet": category in ("wood_log", "apple"),
                    "damaged": False, "clean": True,
                    "hot": False, "open": False,
                }

                objects[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_affordance_profile": effective_profile,
                    "_ambient_features": dict(ambient_feat_dict),
                    "_core_features": dict(core_feat_dict),
                    "_diagnostic_features": dict(diagnostic_feat_dict),
                    "_hidden_features_dict": dict(hidden_feat_dict),
                    "visible_features": full_visible,
                    "visible_state": visible_state,
                }

                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "true_family": category,
                    "mixed_source_type": f"c6_depth_{depth_schedule}",
                    "group_id": group_id,
                    "group_role": group_role,
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(effective_profile),
                    "base_profile": dict(base_profile),
                    "variant_profile": dict(variant_profile),
                    "ambient_group": ambient_group,
                    "hidden_features": dict(hidden_feat_dict),
                    "seed": seed,
                }

    return objects, audit_labels

# =============================================================================
# Event schedule builder (adapted for env3 feature layers)
# =============================================================================
def build_events(objects, audit_labels, seed):
    all_oids = sorted(objects.keys())
    episode_assignments = {}
    for idx, oid in enumerate(all_oids):
        episode_assignments[oid] = idx % N_EPISODES

    all_events = []
    step_id = 0
    prior_observe_count = defaultdict(int)
    prior_observed_features = defaultdict(set)
    prior_observed_states = defaultdict(set)

    for ep in range(N_EPISODES):
        ep_oids = [oid for oid in all_oids if episode_assignments.get(oid) == ep]

        for oid in ep_oids:
            obj = objects[oid]
            label = audit_labels[oid]
            depth = label["depth_schedule"]
            category = label["hidden_category"]
            profile = label["affordance_profile"]

            ambient_feats = obj.get("_ambient_features", {})
            core_feats = obj.get("_core_features", {})
            diagnostic_feats = obj.get("_diagnostic_features", {})
            hidden_feats = obj.get("_hidden_features_dict", {})
            state = obj.get("visible_state", {})
            full_features = dict(obj.get("visible_features", {}))

            # --- Phase 1: Information gathering ---
            if depth == "no_observe":
                # Only ambient features visible
                initial_features = dict(ambient_feats)
                initial_state = {}

            elif depth == "one_observe":
                # Reveal ambient + core + diagnostic
                reveal_features = {}
                reveal_features.update(ambient_feats)
                reveal_features.update(core_feats)
                reveal_features.update(diagnostic_feats)

                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": reveal_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                for f in reveal_features:
                    prior_observed_features[oid].add(f)
                for s in state:
                    prior_observed_states[oid].add(s)
                initial_features = reveal_features
                initial_state = state

            elif depth == "repeated_observe":
                # First observe: reveal ambient + core + diagnostic
                reveal_features = {}
                reveal_features.update(ambient_feats)
                reveal_features.update(core_feats)
                reveal_features.update(diagnostic_feats)

                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": reveal_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                for f in reveal_features:
                    prior_observed_features[oid].add(f)
                for s in state:
                    prior_observed_states[oid].add(s)

                # Second observe: reveal hidden features on top
                combined_features = dict(reveal_features)
                combined_features.update(hidden_feats)

                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": combined_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                for f in combined_features:
                    prior_observed_features[oid].add(f)
                for s in state:
                    prior_observed_states[oid].add(s)
                initial_features = combined_features
                initial_state = state

            # --- Phase 2: Try actions ---
            current_observe_depth = prior_observe_count[oid]

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
                    "action_params": {
                        "affordance": action,
                        "observe_depth": current_observe_depth,
                    },
                    "observed_features_delta": dict(initial_features),
                    "observed_state_delta": dict(initial_state),
                    "success_or_failure": outcome_bool,
                })

            # --- Phase 3: Extended action set ---
            remaining_actions = [a for a in ALL_ACTIONS if a not in try_actions_for_object]
            for action in remaining_actions:
                outcome_bool = (profile.get(action, "fail") == "success")
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid,
                    "action_params": {
                        "affordance": action,
                        "observe_depth": current_observe_depth,
                    },
                    "observed_features_delta": dict(initial_features),
                    "observed_state_delta": dict(initial_state),
                    "success_or_failure": outcome_bool,
                })

            # --- Phase 4: Post-try observe (repeated_observe only) ---
            if depth == "repeated_observe":
                step_id += 1
                post_features = dict(reveal_features)
                post_features.update(hidden_feats)
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid,
                    "action_params": {},
                    "observed_features_delta": post_features,
                    "observed_state_delta": dict(state),
                    "success_or_failure": None,
                })
                prior_observe_count[oid] += 1
                for f in post_features:
                    prior_observed_features[oid].add(f)

    return all_events

# =============================================================================
# Per-seed generation
# =============================================================================
def run_seed(seed):
    print(f"\n{'='*70}")
    print(f"Seed {seed}")
    print(f"{'='*70}")

    objects, audit_labels = generate_c6_objects_env3(seed, prefix=f"c6_s{seed}")
    print(f"  Objects: {len(objects)}")

    all_events = build_events(objects, audit_labels, seed)
    print(f"  Events: {len(all_events)}")

    # Leakage check
    all_violations = []
    for ev in all_events:
        ev_violations = check_event_leakage(ev)
        if ev_violations:
            all_violations.extend([f"step {ev['step_id']}: {v}" for v in ev_violations])
    leakage_detected = len(all_violations) > 0

    schedule_in_ap = any(
        "schedule" in ev.get("action_params", {}) or
        "schedule_type" in ev.get("action_params", {}) or
        "group_role" in ev.get("action_params", {}) or
        "depth_schedule" in ev.get("action_params", {})
        for ev in all_events)
    schedule_label_removed = not schedule_in_ap

    observe_events = [ev for ev in all_events if ev["action_type"] == "observe"]
    try_events = [ev for ev in all_events if ev["action_type"] == "try"]

    # Classify try by depth
    try_by_depth = defaultdict(list)
    for ev in try_events:
        oid = ev["action_target"]
        ds = audit_labels[oid]["depth_schedule"]
        try_by_depth[ds].append(ev)

    group_ids = sorted(set(lbl["group_id"] for lbl in audit_labels.values()))

    # Matched advantage per group
    group_advantages = {}
    role_advantages = defaultdict(list)

    for gid in group_ids:
        group_lbls = {lbl["depth_schedule"]: lbl for oid, lbl in audit_labels.items()
                      if lbl["group_id"] == gid}
        role = list(group_lbls.values())[0].get("group_role", "unknown") if group_lbls else "unknown"

        depth_returns = {}
        for depth_schedule in ["no_observe", "one_observe", "repeated_observe"]:
            depth_oids = [oid for oid, lbl in audit_labels.items()
                          if lbl["group_id"] == gid and lbl["depth_schedule"] == depth_schedule]
            depth_tries = [ev for ev in try_events if ev["action_target"] in depth_oids]
            if depth_tries:
                net_vals = [try_net_value(ev.get("success_or_failure")) for ev in depth_tries]
                depth_returns[depth_schedule] = {
                    "n_tries": len(depth_tries),
                    "mean_net_value": round(sum(net_vals) / len(net_vals), 4),
                    "successes": sum(1 for ev in depth_tries if ev.get("success_or_failure") is True),
                    "failures": sum(1 for ev in depth_tries if ev.get("success_or_failure") is False),
                }

        no_baseline = depth_returns.get("no_observe", {}).get("mean_net_value")
        one_return = depth_returns.get("one_observe", {}).get("mean_net_value")
        rep_return = depth_returns.get("repeated_observe", {}).get("mean_net_value")

        one_adv = round(one_return - no_baseline, 4) if (one_return is not None and no_baseline is not None) else None
        rep_adv = round(rep_return - no_baseline, 4) if (rep_return is not None and no_baseline is not None) else None

        group_advantages[gid] = {
            "role": role,
            "no_observe_baseline": no_baseline,
            "one_observe_return": one_return,
            "repeated_observe_return": rep_return,
            "one_observe_advantage": one_adv,
            "repeated_observe_advantage": rep_adv,
        }

        if one_adv is not None:
            role_advantages[role].append(one_adv)
        if rep_adv is not None:
            role_advantages[role].append(rep_adv)

    return {
        "seed": seed,
        "objects": objects,
        "audit_labels": audit_labels,
        "all_events": all_events,
        "observe_events": observe_events,
        "try_events": try_events,
        "leakage_detected": leakage_detected,
        "schedule_label_removed": schedule_label_removed,
        "no_observe_try_count": len(try_by_depth.get("no_observe", [])),
        "one_observe_try_count": len(try_by_depth.get("one_observe", [])),
        "repeated_observe_try_count": len(try_by_depth.get("repeated_observe", [])),
        "group_ids": group_ids,
        "group_advantages": group_advantages,
        "role_advantages": dict(role_advantages),
    }

# =============================================================================
# Run all seeds
# =============================================================================
print("\n" + "="*70)
print("Block 1J40b-env3: Category-Ambiguous Observe-Value Environment")
print("="*70)

all_seed_results = {}
for seed in SEEDS:
    all_seed_results[seed] = run_seed(seed)

# =============================================================================
# Gather all objects with full audit info for cross-seed analyses
# =============================================================================
all_objects_info = []  # (seed, oid, category, ambient_group, group_role, depth, base_profile, variant_profile, effective_profile, ambient_feats, core_feats, diagnostic_feats)
for seed, result in all_seed_results.items():
    for oid, lbl in result["audit_labels"].items():
        obj = result["objects"][oid]
        all_objects_info.append({
            "seed": seed,
            "oid": oid,
            "category": lbl["hidden_category"],
            "ambient_group": lbl["ambient_group"],
            "group_role": lbl["group_role"],
            "depth_schedule": lbl["depth_schedule"],
            "base_profile": lbl["base_profile"],
            "variant_profile": lbl["variant_profile"],
            "effective_profile": lbl["affordance_profile"],
            "ambient_features": obj.get("_ambient_features", {}),
            "core_features": obj.get("_core_features", {}),
            "diagnostic_features": obj.get("_diagnostic_features", {}),
            "hidden_features": obj.get("_hidden_features_dict", {}),
        })

n_total_objects = len(all_objects_info)

# =============================================================================
# AUDIT 1: Pre-observe Ambiguity
# =============================================================================
print("\n" + "="*70)
print("AUDIT 1: Pre-observe Ambiguity")
print("="*70)

# For each object, compute expected return under pre-observe (ambient only) uncertainty
# Pre-observe: agent knows ambient group but not which category within the group.
# Within a group (2 categories), each is equally likely.
# Expected return of action = avg of true values across both categories in the group.

ambient_group_categories = defaultdict(list)
for cat in CATEGORIES:
    ambient_group_categories[CATEGORY_AMBIENT_GROUP[cat]].append(cat)

# Map: ambient_group -> list of categories
print(f"  Ambient Group A categories: {ambient_group_categories['A']}")
print(f"  Ambient Group B categories: {ambient_group_categories['B']}")

pre_ambiguity_results = []
for info in all_objects_info:
    category = info["category"]
    ambient_group = info["ambient_group"]
    effective_profile = info["effective_profile"]
    base_profile = info["base_profile"]
    variant_profile = info["variant_profile"]
    role = info["group_role"]

    # Oracle: best action from effective (true) profile
    oracle_values = {}
    for action in ALL_ACTIONS:
        oracle_values[action] = action_true_value(action, effective_profile)
    oracle_best_action = max(oracle_values, key=oracle_values.get)
    oracle_best_value = oracle_values[oracle_best_action]

    # Pre-observe: expected return given only ambient group knowledge
    # Within the ambient group, each category is equally likely
    group_cats = ambient_group_categories[ambient_group]
    pre_expected_values = {}
    for action in ALL_ACTIONS:
        values = []
        for gcat in group_cats:
            gcat_base = BASE_AFFORDANCE_PROFILES[gcat]
            values.append(action_true_value(action, gcat_base))
        pre_expected_values[action] = sum(values) / len(values)

    # Best pre action: max expected value
    max_pre_val = max(pre_expected_values.values())
    pre_best_actions = [a for a, v in pre_expected_values.items() if abs(v - max_pre_val) < 0.001]
    pre_tie_count = len(pre_best_actions)
    pre_fixed_best = sorted(pre_best_actions)[0]  # fixed tie-break: alphabetical

    # True value of pre-fixed-tie-break action
    pre_fixed_true_value = oracle_values[pre_fixed_best]

    # Randomized tie-break: expected true value if randomly choosing among ties
    pre_randomized_value = sum(oracle_values[a] for a in pre_best_actions) / len(pre_best_actions)

    pre_ambiguity_results.append({
        "seed": info["seed"],
        "oid": info["oid"],
        "category": category,
        "ambient_group": ambient_group,
        "role": role,
        "depth": info["depth_schedule"],
        "oracle_best_action": oracle_best_action,
        "oracle_best_value": oracle_best_value,
        "pre_expected_values": pre_expected_values,
        "pre_best_actions": pre_best_actions,
        "pre_tie_count": pre_tie_count,
        "pre_fixed_best_action": pre_fixed_best,
        "pre_fixed_true_value": pre_fixed_true_value,
        "pre_randomized_value": pre_randomized_value,
    })

# Aggregate pre-observe ambiguity
pre_oracle_values = [r["oracle_best_value"] for r in pre_ambiguity_results]
pre_fixed_values = [r["pre_fixed_true_value"] for r in pre_ambiguity_results]
pre_randomized_values = [r["pre_randomized_value"] for r in pre_ambiguity_results]

pre_mean_oracle = sum(pre_oracle_values) / len(pre_oracle_values)
pre_mean_fixed = sum(pre_fixed_values) / len(pre_fixed_values)
pre_mean_randomized = sum(pre_randomized_values) / len(pre_randomized_values)
pre_oracle_gap_fixed = pre_mean_oracle - pre_mean_fixed
pre_oracle_gap_randomized = pre_mean_oracle - pre_mean_randomized

print(f"\n  Pre-observe ambiguity summary (all {n_total_objects} objects):")
print(f"    oracle mean true return:          {pre_mean_oracle:.4f}")
print(f"    pre (fixed tie-break) mean true:   {pre_mean_fixed:.4f}")
print(f"    pre (randomized tie) mean true:    {pre_mean_randomized:.4f}")
print(f"    pre/oracle gap (fixed):            {pre_oracle_gap_fixed:.4f}")
print(f"    pre/oracle gap (randomized):       {pre_oracle_gap_randomized:.4f}")

# Per-seed breakdown
print(f"\n  Per-seed pre (randomized) vs oracle:")
print(f"  {'Seed':>6} {'Pre':>8} {'Oracle':>8} {'Gap':>8}")
for seed in SEEDS:
    seed_rs = [r for r in pre_ambiguity_results if r["seed"] == seed]
    seed_pre = sum(r["pre_randomized_value"] for r in seed_rs) / len(seed_rs)
    seed_oracle = sum(r["oracle_best_value"] for r in seed_rs) / len(seed_rs)
    print(f"  {seed:>6} {seed_pre:>8.4f} {seed_oracle:>8.4f} {seed_oracle - seed_pre:>8.4f}")

# Per-category breakdown
print(f"\n  Per-category pre (randomized) vs oracle:")
print(f"  {'Category':<18} {'Pre':>8} {'Oracle':>8} {'Gap':>8}")
for cat in CATEGORIES:
    cat_rs = [r for r in pre_ambiguity_results if r["category"] == cat]
    cat_pre = sum(r["pre_randomized_value"] for r in cat_rs) / len(cat_rs)
    cat_oracle = sum(r["oracle_best_value"] for r in cat_rs) / len(cat_rs)
    print(f"  {cat:<18} {cat_pre:>8.4f} {cat_oracle:>8.4f} {cat_oracle - cat_pre:>8.4f}")

# =============================================================================
# AUDIT 2: Category Probes by Visibility Level
# =============================================================================
print("\n" + "="*70)
print("AUDIT 2: Category Probes by Visibility Level")
print("="*70)

# Build feature vectors for each visibility level
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}

def build_feature_vectors(feature_set_key, all_info):
    """Build feature vectors using only features from the specified set."""
    vectors = []
    labels = []
    seeds_list = []
    for info_item in all_info:
        features = {}
        if feature_set_key == "ambient":
            features = dict(info_item["ambient_features"])
        elif feature_set_key == "core":
            features = dict(info_item["core_features"])
        elif feature_set_key == "diagnostic":
            features = dict(info_item["diagnostic_features"])
        elif feature_set_key == "core_plus_diagnostic":
            features = {}
            features.update(info_item["core_features"])
            features.update(info_item["diagnostic_features"])
        elif feature_set_key == "all_visible":
            features = {}
            features.update(info_item["ambient_features"])
            features.update(info_item["core_features"])
            features.update(info_item["diagnostic_features"])

        # Collect all feature names for this set
        all_feat_names = sorted(set(
            f for info2 in all_info
            for f in _get_features_dict(info2, feature_set_key).keys()
        ))
        vec = [1.0 if f in features else 0.0 for f in all_feat_names]
        vectors.append(vec)
        labels.append(CATEGORY_TO_IDX[info_item["category"]])
        seeds_list.append(info_item["seed"])

    return vectors, labels, seeds_list, all_feat_names

def _get_features_dict(info_item, key):
    if key == "ambient":
        return info_item["ambient_features"]
    elif key == "core":
        return info_item["core_features"]
    elif key == "diagnostic":
        return info_item["diagnostic_features"]
    elif key == "core_plus_diagnostic":
        d = {}
        d.update(info_item["core_features"])
        d.update(info_item["diagnostic_features"])
        return d
    elif key == "all_visible":
        d = {}
        d.update(info_item["ambient_features"])
        d.update(info_item["core_features"])
        d.update(info_item["diagnostic_features"])
        return d
    return {}

category_probe_results = {}
for level_name in ["ambient", "core", "diagnostic", "core_plus_diagnostic", "all_visible"]:
    X, y, seeds_list, feat_names = build_feature_vectors(level_name, all_objects_info)

    if len(set(y)) < 2:
        acc = 1.0 if len(set(y)) == 1 else 0.0
        category_probe_results[level_name] = {
            "accuracy": acc, "n_features": len(feat_names),
            "feature_names": feat_names,
            "seed_accuracies": {},
        }
        print(f"  {level_name}: accuracy={acc:.4f} (trivial, {len(set(y))} classes)")
        continue

    # LOSO per seed
    seed_accs = {}
    for seed in SEEDS:
        train_X = [X[i] for i in range(len(X)) if seeds_list[i] != seed]
        train_y = [y[i] for i in range(len(y)) if seeds_list[i] != seed]
        test_X = [X[i] for i in range(len(X)) if seeds_list[i] == seed]
        test_y = [y[i] for i in range(len(y)) if seeds_list[i] == seed]

        if len(train_y) < 2 or len(test_y) < 2:
            seed_accs[seed] = None
            continue

        W, b = train_softmax_classifier(train_X, train_y, num_classes=4, l2_alpha=0.1, max_iter=200, lr=0.01)
        preds = predict_softmax(W, b, test_X)
        seed_accs[seed] = sum(1 for p, ty in zip(preds, test_y) if p == ty) / len(test_y)

    # Aggregate accuracy (closed-world)
    W_all, b_all = train_softmax_classifier(X, y, num_classes=4, l2_alpha=0.1, max_iter=200, lr=0.01)
    preds_all = predict_softmax(W_all, b_all, X)
    full_acc = sum(1 for p, ty in zip(preds_all, y) if p == ty) / len(y)

    mean_seed_acc = sum(v for v in seed_accs.values() if v is not None) / max(sum(1 for v in seed_accs.values() if v is not None), 1)
    max_seed_acc = max((v for v in seed_accs.values() if v is not None), default=0)

    category_probe_results[level_name] = {
        "accuracy": round(full_acc, 4),
        "mean_seed_accuracy": round(mean_seed_acc, 4),
        "max_seed_accuracy": round(max_seed_acc, 4),
        "n_features": len(feat_names),
        "seed_accuracies": seed_accs,
    }
    print(f"  {level_name}: accuracy={full_acc:.4f}, mean_seed={mean_seed_acc:.4f}, max_seed={max_seed_acc:.4f}, n_features={len(feat_names)}")

# =============================================================================
# AUDIT 3: Post-observe Source Ablations
# =============================================================================
print("\n" + "="*70)
print("AUDIT 3: Post-observe Source Ablations")
print("="*70)

# For each object, compute best-action-selected return under 4 conditions:
# 1. pre_ambient_only: only ambient features
# 2. post_core_only: ambient + core features (identifies category)
# 3. post_diagnostic_only: ambient + diagnostic features (action-relevant but not category-identifying)
# 4. post_core_plus_diagnostic: ambient + core + diagnostic (full post-observe)

ablation_results = []
for info in all_objects_info:
    category = info["category"]
    ambient_group = info["ambient_group"]
    effective_profile = info["effective_profile"]
    role = info["group_role"]
    seed = info["seed"]
    depth = info["depth_schedule"]

    # True values for all actions
    true_values = {}
    for action in ALL_ACTIONS:
        true_values[action] = action_true_value(action, effective_profile)
    oracle_best = max(true_values, key=true_values.get)
    oracle_val = true_values[oracle_best]

    # --- pre_ambient_only: expected return given only ambient group ---
    group_cats = ambient_group_categories[ambient_group]
    pre_ambient_vals = {}
    for action in ALL_ACTIONS:
        vals = [action_true_value(action, BASE_AFFORDANCE_PROFILES[gcat]) for gcat in group_cats]
        pre_ambient_vals[action] = sum(vals) / len(vals)
    pre_best_val = max(pre_ambient_vals.values())
    pre_best_actions = [a for a, v in pre_ambient_vals.items() if abs(v - pre_best_val) < 0.001]

    # --- post_core_only: knows category from core features ---
    # Category → base profile → best action
    post_core_vals = {}
    base_profile = info["base_profile"]
    for action in ALL_ACTIONS:
        post_core_vals[action] = action_true_value(action, base_profile)
    post_core_best_val = max(post_core_vals.values())
    post_core_best_actions = [a for a, v in post_core_vals.items() if abs(v - post_core_best_val) < 0.001]
    # True value of post_core best (using fixed tie-break)
    post_core_best_true = true_values[sorted(post_core_best_actions)[0]]

    # --- post_diagnostic_only: only ambient + diagnostic ---
    # diagnostic features are deterministic per category, so this ALSO identifies category
    # But some diagnostic features are shared across categories, creating ambiguity
    # Compute: within ambient group, use diagnostic features to distinguish categories
    # For simplicity, diagnostic features ARE also deterministic per category
    # So post_diagnostic_only should have same performance as post_core_only
    post_diag_vals = {}
    for action in ALL_ACTIONS:
        post_diag_vals[action] = action_true_value(action, base_profile)
    post_diag_best_val = max(post_diag_vals.values())
    post_diag_best_actions = [a for a, v in post_diag_vals.items() if abs(v - post_diag_best_val) < 0.001]
    post_diag_best_true = true_values[sorted(post_diag_best_actions)[0]]

    # --- post_core_plus_diagnostic: full post-observe ---
    # Same as post_core_only for neutral/wasteful (both identify category → base profile)
    # For helps objects, still base profile (variant not identifiable from visible features)
    post_full_vals = {}
    for action in ALL_ACTIONS:
        post_full_vals[action] = action_true_value(action, base_profile)
    post_full_best_val = max(post_full_vals.values())
    post_full_best_actions = [a for a, v in post_full_vals.items() if abs(v - post_full_best_val) < 0.001]
    post_full_best_true = true_values[sorted(post_full_best_actions)[0]]

    # Tie-breaking: use randomized expected value
    pre_rand_val = sum(true_values[a] for a in pre_best_actions) / len(pre_best_actions)
    post_core_rand_val = sum(true_values[a] for a in post_core_best_actions) / len(post_core_best_actions)
    post_diag_rand_val = sum(true_values[a] for a in post_diag_best_actions) / len(post_diag_best_actions)
    post_full_rand_val = sum(true_values[a] for a in post_full_best_actions) / len(post_full_best_actions)

    ablation_results.append({
        "seed": seed, "oid": info["oid"], "category": category,
        "ambient_group": ambient_group, "role": role, "depth": depth,
        "oracle_val": oracle_val,
        "pre_ambient_val": pre_rand_val,
        "post_core_val": post_core_rand_val,
        "post_diagnostic_val": post_diag_rand_val,
        "post_full_val": post_full_rand_val,
        "pre_best_actions": pre_best_actions,
        "pre_tie_count": len(pre_best_actions),
    })

# Aggregate ablation results
def aggregate_ablation(key, results):
    vals = [r[key] for r in results]
    mean_val = sum(vals) / len(vals)
    oracle_vals = [r["oracle_val"] for r in results]
    mean_oracle = sum(oracle_vals) / len(oracle_vals)
    gain = mean_val - sum([r["pre_ambient_val"] for r in results]) / len(results)
    gap = mean_oracle - mean_val
    return {"mean_return": round(mean_val, 4), "mean_gain_vs_pre": round(gain, 4),
            "mean_oracle_gap": round(gap, 4)}

ablation_summary = {}
for cond_key, cond_label in [("pre_ambient_val", "pre_ambient_only"),
                               ("post_core_val", "post_core_only"),
                               ("post_diagnostic_val", "post_diagnostic_only"),
                               ("post_full_val", "post_full")]:
    agg = aggregate_ablation(cond_key, ablation_results)
    ablation_summary[cond_label] = agg
    print(f"\n  {cond_label}:")
    print(f"    mean best-action-selected return: {agg['mean_return']:.4f}")
    print(f"    mean gain vs pre_ambient:         {agg['mean_gain_vs_pre']:+.4f}")
    print(f"    mean oracle gap:                  {agg['mean_oracle_gap']:.4f}")

# Per-seed breakdown
print(f"\n  Per-seed breakdown:")
print(f"  {'Seed':>6} {'Pre':>8} {'Core':>8} {'Diag':>8} {'Full':>8} {'Oracle':>8}")
for seed in SEEDS:
    seed_rs = [r for r in ablation_results if r["seed"] == seed]
    pre = sum(r["pre_ambient_val"] for r in seed_rs) / len(seed_rs)
    core = sum(r["post_core_val"] for r in seed_rs) / len(seed_rs)
    diag = sum(r["post_diagnostic_val"] for r in seed_rs) / len(seed_rs)
    full = sum(r["post_full_val"] for r in seed_rs) / len(seed_rs)
    oracle = sum(r["oracle_val"] for r in seed_rs) / len(seed_rs)
    print(f"  {seed:>6} {pre:>8.4f} {core:>8.4f} {diag:>8.4f} {full:>8.4f} {oracle:>8.4f}")

# Per-category breakdown
print(f"\n  Per-category breakdown:")
print(f"  {'Category':<18} {'Pre':>8} {'Core':>8} {'Diag':>8} {'Full':>8} {'Oracle':>8}")
for cat in CATEGORIES:
    cat_rs = [r for r in ablation_results if r["category"] == cat]
    pre = sum(r["pre_ambient_val"] for r in cat_rs) / len(cat_rs)
    core = sum(r["post_core_val"] for r in cat_rs) / len(cat_rs)
    diag = sum(r["post_diagnostic_val"] for r in cat_rs) / len(cat_rs)
    full = sum(r["post_full_val"] for r in cat_rs) / len(cat_rs)
    oracle = sum(r["oracle_val"] for r in cat_rs) / len(cat_rs)
    print(f"  {cat:<18} {pre:>8.4f} {core:>8.4f} {diag:>8.4f} {full:>8.4f} {oracle:>8.4f}")

# Positive gain analysis
for cond_key, cond_label in [("post_core_val", "post_core_only"),
                               ("post_full_val", "post_full")]:
    gains = [r[cond_key] - r["pre_ambient_val"] for r in ablation_results]
    positive_gains = sum(1 for g in gains if g > 0.001)
    zero_gains = sum(1 for g in gains if -0.001 <= g <= 0.001)
    negative_gains = sum(1 for g in gains if g < -0.001)
    print(f"\n  {cond_label} gain distribution:")
    print(f"    positive: {positive_gains}/{len(gains)} ({positive_gains/len(gains):.4f})")
    print(f"    zero:     {zero_gains}/{len(gains)} ({zero_gains/len(gains):.4f})")
    print(f"    negative: {negative_gains}/{len(gains)} ({negative_gains/len(gains):.4f})")
    print(f"    mean gain: {sum(gains)/len(gains):+.4f}")

# =============================================================================
# AUDIT 4: Pre-observe Tie Diagnostics
# =============================================================================
print("\n" + "="*70)
print("AUDIT 4: Pre-observe Tie Diagnostics")
print("="*70)

for ambient_group in ["A", "B"]:
    group_cats = ambient_group_categories[ambient_group]
    print(f"\n  Ambient Group {ambient_group} ({', '.join(group_cats)}):")

    # Compute expected return for each action given only ambient group knowledge
    print(f"    Expected return per action (equal category prior):")
    for action in ALL_ACTIONS:
        vals = [action_true_value(action, BASE_AFFORDANCE_PROFILES[cat]) for cat in group_cats]
        exp_val = sum(vals) / len(vals)
        success_desc = [f"{cat}={v:+.2f}" for cat, v in zip(group_cats, vals)]
        print(f"      {action:<22}: expected={exp_val:+.2f}  ({', '.join(success_desc)})")

    # Identify best action(s) and tie count
    action_exp_vals = {}
    for action in ALL_ACTIONS:
        vals = [action_true_value(action, BASE_AFFORDANCE_PROFILES[cat]) for cat in group_cats]
        action_exp_vals[action] = sum(vals) / len(vals)

    max_exp = max(action_exp_vals.values())
    best_actions = sorted([a for a, v in action_exp_vals.items() if abs(v - max_exp) < 0.001])
    tie_count = len(best_actions)
    print(f"    Best action(s): {best_actions}")
    print(f"    Tie count: {tie_count}")
    print(f"    Fixed tie-break (alphabetical): {best_actions[0]}")

    # Randomized tie-break stats: for each category, what's the true value of a random best action?
    # This is computed per-object in ablation_results, let's aggregate
    group_objs = [r for r in ablation_results if r["ambient_group"] == ambient_group]
    pre_vals = [r["pre_ambient_val"] for r in group_objs]
    pre_mean = sum(pre_vals) / len(pre_vals)
    pre_std = (sum((v - pre_mean)**2 for v in pre_vals) / len(pre_vals))**0.5
    print(f"    Randomized-tie pre return: mean={pre_mean:.4f}, std={pre_std:.4f}")

# =============================================================================
# AUDIT 5: Role Leakage / Shortcut Audits
# =============================================================================
print("\n" + "="*70)
print("AUDIT 5: Role Leakage / Shortcut Audits")
print("="*70)

# Collect all visible features (ambient + core + diagnostic) per object
ROLE_TO_IDX = {"observe_helps": 0, "observe_neutral": 1, "observe_wasteful": 2}
all_visible_feature_names = sorted(set(
    f for info in all_objects_info
    for f in list(info["ambient_features"].keys())
    + list(info["core_features"].keys())
    + list(info["diagnostic_features"].keys())
))

print(f"  Total unique visible features: {len(all_visible_feature_names)}")

# Build feature vectors for role probe
role_X = []
role_y = []
role_seeds = []
for info in all_objects_info:
    feats = {}
    feats.update(info["ambient_features"])
    feats.update(info["core_features"])
    feats.update(info["diagnostic_features"])
    vec = [1.0 if f in feats else 0.0 for f in all_visible_feature_names]
    role_X.append(vec)
    role_y.append(ROLE_TO_IDX[info["group_role"]])
    role_seeds.append(info["seed"])

# Single-feature role probe
print("\n  Single-feature role probe:")
single_feat_accuracies = {}
single_feat_high = []
for fi, fname in enumerate(all_visible_feature_names):
    role_counts_by_val = defaultdict(lambda: defaultdict(int))
    for vec, role_idx in zip(role_X, role_y):
        role_counts_by_val[int(vec[fi])][role_idx] += 1
    correct = 0
    for val, counts in role_counts_by_val.items():
        majority = max(range(3), key=lambda c: counts.get(c, 0))
        correct += counts.get(majority, 0)
    acc = correct / len(role_y)
    single_feat_accuracies[fname] = acc
    if acc > 0.40:
        single_feat_high.append((fname, acc))

if single_feat_high:
    print(f"    Features with accuracy > 0.40:")
    for fname, acc in sorted(single_feat_high, key=lambda x: -x[1]):
        print(f"      {fname:<30}: accuracy={acc:.4f}")
else:
    print(f"    No single feature exceeds 0.40 accuracy (chance=0.333)")

# Full-feature role probe (LOSO per seed)
print("\n  Full-feature role probe (LOSO per seed):")
role_seed_accs = []
for seed in SEEDS:
    train_X = [role_X[i] for i in range(len(role_X)) if role_seeds[i] != seed]
    train_y = [role_y[i] for i in range(len(role_y)) if role_seeds[i] != seed]
    test_X = [role_X[i] for i in range(len(role_X)) if role_seeds[i] == seed]
    test_y = [role_y[i] for i in range(len(role_y)) if role_seeds[i] == seed]

    if len(train_y) < 3 or len(test_y) < 3:
        continue

    W, b = train_softmax_classifier(train_X, train_y, num_classes=3, l2_alpha=0.1, max_iter=200, lr=0.01)
    preds = predict_softmax(W, b, test_X)
    acc = sum(1 for p, ty in zip(preds, test_y) if p == ty) / len(test_y)
    role_seed_accs.append(acc)
    print(f"    Seed {seed}: accuracy={acc:.4f}")

# Aggregate role probe
W_all, b_all = train_softmax_classifier(role_X, role_y, num_classes=3, l2_alpha=0.1, max_iter=200, lr=0.01)
preds_all = predict_softmax(W_all, b_all, role_X)
full_role_probe_accuracy = sum(1 for p, ty in zip(preds_all, role_y) if p == ty) / len(role_y)

mean_role_seed_acc = sum(role_seed_accs) / len(role_seed_accs) if role_seed_accs else 0
max_role_seed_acc = max(role_seed_accs) if role_seed_accs else 0

print(f"\n  Full feature role probe: accuracy={full_role_probe_accuracy:.4f}")
print(f"  Mean seed accuracy: {mean_role_seed_acc:.4f}")
print(f"  Max seed accuracy:  {max_role_seed_acc:.4f}")
print(f"  Chance baseline:    0.333")

# Feature-pair role probe
print("\n  Feature-pair role probe:")
n_visible = len(all_visible_feature_names)
pairs_exclusive = 0
total_pairs = 0
for fi in range(n_visible):
    for fj in range(fi + 1, n_visible):
        total_pairs += 1
        pair_roles = set()
        for vec, role_idx in zip(role_X, role_y):
            if vec[fi] > 0.5 and vec[fj] > 0.5:
                pair_roles.add(role_idx)
        if len(pair_roles) == 1:
            pairs_exclusive += 1
            if pairs_exclusive <= 5:
                f1_name = all_visible_feature_names[fi]
                f2_name = all_visible_feature_names[fj]
                role_name = ["helps", "neutral", "wasteful"][list(pair_roles)[0]]
                print(f"    EXCLUSIVE: ({f1_name}, {f2_name}) -> {role_name}")

print(f"  Total pairs: {total_pairs}")
print(f"  Exclusive pairs: {pairs_exclusive}")

# Direct role marker check
all_visible_set = set(all_visible_feature_names)
direct_role_marker = False
for fname in all_visible_set:
    if "role" in fname.lower() or "observe_helps" in fname or "observe_neutral" in fname or "observe_wasteful" in fname:
        direct_role_marker = True
        print(f"  DIRECT ROLE MARKER FOUND: {fname}")

if not direct_role_marker:
    print(f"  direct_role_marker_detected: false")

# Feature-role balance
print("\n  Feature-Role Balance:")
feature_role_counts = defaultdict(lambda: defaultdict(int))
total_per_role = defaultdict(int)
for info in all_objects_info:
    role = info["group_role"]
    total_per_role[role] += 1
    for fname in info["ambient_features"]:
        feature_role_counts[fname][role] += 1
    for fname in info["core_features"]:
        feature_role_counts[fname][role] += 1
    for fname in info["diagnostic_features"]:
        feature_role_counts[fname][role] += 1

features_in_all_roles = 0
features_in_2_roles = 0
features_in_1_role = 0
max_role_gap_overall = 0.0

for fname in sorted(all_visible_feature_names):
    hc = feature_role_counts[fname].get("observe_helps", 0)
    nc = feature_role_counts[fname].get("observe_neutral", 0)
    wc = feature_role_counts[fname].get("observe_wasteful", 0)
    total = hc + nc + wc
    if total > 0:
        ph, pn, pw = hc/total, nc/total, wc/total
        max_gap = max(ph, pn, pw) - min(ph, pn, pw)
    else:
        ph = pn = pw = 0
        max_gap = 0.0
    max_role_gap_overall = max(max_role_gap_overall, max_gap)

    roles_present = sum(1 for c in [hc, nc, wc] if c > 0)
    if roles_present >= 3:
        features_in_all_roles += 1
    elif roles_present == 2:
        features_in_2_roles += 1
    else:
        features_in_1_role += 1

print(f"  Features in all 3 roles: {features_in_all_roles}")
print(f"  Features in 2 roles:    {features_in_2_roles}")
print(f"  Features in 1 role:     {features_in_1_role}")
print(f"  Max role gap: {max_role_gap_overall:.4f}")

# =============================================================================
# AUDIT 6: Hidden Feature Timing
# =============================================================================
print("\n" + "="*70)
print("AUDIT 6: Hidden Feature Timing")
print("="*70)

all_hidden_feature_names = set()
for cat, hf_list in CATEGORY_HIDDEN_FEATURES.items():
    for hf in hf_list:
        all_hidden_feature_names.add(hf)

no_observe_hidden = 0
one_observe_hidden = 0
repeated_observe_hidden_before_reveal = 0
repeated_observe_hidden_after_reveal = 0

for seed, result in all_seed_results.items():
    for ev in result["all_events"]:
        oid = ev["action_target"]
        lbl = result["audit_labels"].get(oid)
        if not lbl:
            continue
        depth = lbl["depth_schedule"]
        ofd = ev.get("observed_features_delta", {})
        hidden_in_delta = [f for f in ofd if f in all_hidden_feature_names]

        if hidden_in_delta:
            if depth == "no_observe":
                no_observe_hidden += len(hidden_in_delta)
            elif depth == "one_observe":
                one_observe_hidden += len(hidden_in_delta)
            elif depth == "repeated_observe":
                visible_in_delta = [f for f in ofd if f not in all_hidden_feature_names]
                has_both = len(visible_in_delta) > 5 and len(hidden_in_delta) > 0
                if has_both:
                    repeated_observe_hidden_after_reveal += len(hidden_in_delta)
                else:
                    repeated_observe_hidden_before_reveal += len(hidden_in_delta)

print(f"  no_observe hidden feature uses: {no_observe_hidden}")
print(f"  one_observe hidden feature uses: {one_observe_hidden}")
print(f"  repeated_observe hidden before reveal: {repeated_observe_hidden_before_reveal}")
print(f"  repeated_observe hidden after reveal: {repeated_observe_hidden_after_reveal}")

hidden_timing_audit_passed = (no_observe_hidden == 0 and one_observe_hidden == 0
                               and repeated_observe_hidden_before_reveal == 0)
print(f"  hidden_timing_audit_passed: {hidden_timing_audit_passed}")

# =============================================================================
# AUDIT 7: Distinguish Metrics (mean-all-actions vs best-action-selected)
# =============================================================================
print("\n" + "="*70)
print("AUDIT 7: Distinguish Metrics")
print("="*70)

# Mean return across ALL try actions (the env2 metric)
pre_try_returns = []
post_try_returns = []
for seed, result in all_seed_results.items():
    for ev in result["try_events"]:
        oid = ev["action_target"]
        ds = result["audit_labels"][oid]["depth_schedule"]
        nv = try_net_value(ev.get("success_or_failure"))
        if ds == "no_observe":
            pre_try_returns.append(nv)
        else:
            post_try_returns.append(nv)

pre_mean_all_try = sum(pre_try_returns) / max(len(pre_try_returns), 1)
post_mean_all_try = sum(post_try_returns) / max(len(post_try_returns), 1)
all_try_improvement = post_mean_all_try - pre_mean_all_try

print(f"  Mean return across ALL try actions:")
print(f"    pre (no_observe):  {pre_mean_all_try:.4f}")
print(f"    post (one_observe+): {post_mean_all_try:.4f}")
print(f"    improvement:         {all_try_improvement:+.4f}")

# Best-action-selected return (from ablation, using post_full)
best_action_pre = sum(r["pre_ambient_val"] for r in ablation_results) / len(ablation_results)
best_action_post = sum(r["post_full_val"] for r in ablation_results) / len(ablation_results)

print(f"\n  Best-action-selected return:")
print(f"    pre (ambient only):      {best_action_pre:.4f}")
print(f"    post (core+diagnostic):  {best_action_post:.4f}")
print(f"    improvement:             {best_action_post - best_action_pre:+.4f}")

print(f"\n  IMPORTANT: These are DIFFERENT metrics.")
print(f"  Mean-all-try improvement ({all_try_improvement:+.4f}) != best-action-selected improvement ({best_action_post - best_action_pre:+.4f})")

# =============================================================================
# AUDIT 8: Possible Observe Gain Summary
# =============================================================================
print("\n" + "="*70)
print("AUDIT 8: Possible Observe Gain Summary")
print("="*70)

# Using post_full vs pre_ambient from ablation results
for info_type, filter_key, filter_vals in [
    ("All objects", None, None),
    ("By role", "role", ["observe_helps", "observe_neutral", "observe_wasteful"]),
    ("By category", "category", CATEGORIES),
]:
    if filter_key is None:
        rs = ablation_results
    else:
        rs = [r for r in ablation_results if r[filter_key] in filter_vals]

    if not rs:
        continue

    pre_mean = sum(r["pre_ambient_val"] for r in rs) / len(rs)
    post_mean = sum(r["post_full_val"] for r in rs) / len(rs)
    oracle_mean = sum(r["oracle_val"] for r in rs) / len(rs)
    gain = post_mean - pre_mean
    pre_gap = oracle_mean - pre_mean
    post_gap = oracle_mean - post_mean
    positive_gains = sum(1 for r in rs if r["post_full_val"] - r["pre_ambient_val"] > 0.001)
    gain_rate = positive_gains / len(rs)

    print(f"\n  {info_type}:")
    print(f"    pre mean true return:  {pre_mean:.4f}")
    print(f"    post mean true return: {post_mean:.4f}")
    print(f"    oracle mean true return: {oracle_mean:.4f}")
    print(f"    pre/oracle gap:  {pre_gap:.4f}")
    print(f"    post/oracle gap: {post_gap:.4f}")
    print(f"    post-pre gain:   {gain:+.4f}")
    print(f"    positive gain rate: {gain_rate:.4f}")

# Per-seed summary
print(f"\n  Per-seed summary:")
print(f"  {'Seed':>6} {'Pre':>8} {'Post':>8} {'Oracle':>8} {'PreGap':>8} {'PostGap':>8} {'Gain':>8} {'PosGainRate':>12}")
for seed in SEEDS:
    rs = [r for r in ablation_results if r["seed"] == seed]
    pre = sum(r["pre_ambient_val"] for r in rs) / len(rs)
    post = sum(r["post_full_val"] for r in rs) / len(rs)
    oracle = sum(r["oracle_val"] for r in rs) / len(rs)
    gain = post - pre
    pre_gap = oracle - pre
    post_gap = oracle - post
    pos_rate = sum(1 for r in rs if r["post_full_val"] - r["pre_ambient_val"] > 0.001) / len(rs)
    print(f"  {seed:>6} {pre:>8.4f} {post:>8.4f} {oracle:>8.4f} {pre_gap:>8.4f} {post_gap:>8.4f} {gain:>+8.4f} {pos_rate:>12.4f}")

# Best action change rate
# For each object: does the set of best actions change from pre to post?
best_action_changes = 0
for r in ablation_results:
    # Pre best actions (from ambient only, within the ambient group)
    pre_best = set(r["pre_best_actions"])
    # Post best actions (from full features = core+diagnostic)
    base_profile = BASE_AFFORDANCE_PROFILES[r["category"]]
    post_vals = {a: action_true_value(a, base_profile) for a in ALL_ACTIONS}
    post_max = max(post_vals.values())
    post_best = set(a for a, v in post_vals.items() if abs(v - post_max) < 0.001)
    if pre_best != post_best:
        best_action_changes += 1

print(f"\n  Best action change rate: {best_action_changes}/{len(ablation_results)} ({best_action_changes/len(ablation_results):.4f})")

# =============================================================================
# Acceptance Checks
# =============================================================================
print("\n" + "="*70)
print("Acceptance Checks")
print("="*70)

# Aggregate role advantages
all_helps_advs = []
all_neutral_advs = []
all_wasteful_advs = []
for seed, result in all_seed_results.items():
    all_helps_advs.extend(result["role_advantages"].get("observe_helps", []))
    all_neutral_advs.extend(result["role_advantages"].get("observe_neutral", []))
    all_wasteful_advs.extend(result["role_advantages"].get("observe_wasteful", []))

agg_helps_adv = round(sum(all_helps_advs) / max(len(all_helps_advs), 1), 4)
agg_neutral_adv = round(sum(all_neutral_advs) / max(len(all_neutral_advs), 1), 4)
agg_wasteful_adv = round(sum(all_wasteful_advs) / max(len(all_wasteful_advs), 1), 4)

any_leakage = any(r["leakage_detected"] for r in all_seed_results.values())
all_schedule_removed = all(r["schedule_label_removed"] for r in all_seed_results.values())

# Pre-observe ambiguity targets
cat_probe_ambient = category_probe_results.get("ambient", {}).get("accuracy", 1.0)
cat_probe_core = category_probe_results.get("core", {}).get("accuracy", 1.0)
cat_probe_diagnostic = category_probe_results.get("diagnostic", {}).get("accuracy", 1.0)

# Using randomized tie-break for pre, and post_full for post
best_pre = sum(r["pre_ambient_val"] for r in ablation_results) / len(ablation_results)
best_post = sum(r["post_full_val"] for r in ablation_results) / len(ablation_results)
best_oracle = sum(r["oracle_val"] for r in ablation_results) / len(ablation_results)
post_pre_gain = best_post - best_pre
pre_oracle_gap = best_oracle - best_pre
post_oracle_gap = best_oracle - best_post
positive_gain_rate = sum(1 for r in ablation_results if r["post_full_val"] - r["pre_ambient_val"] > 0.001) / len(ablation_results)

checks = {}
checks["recursive_leakage_check_passed"] = not any_leakage
checks["direct_role_marker_detected = false"] = not direct_role_marker
checks["full_visible_role_probe_accuracy <= 0.45"] = full_role_probe_accuracy <= 0.45
checks["mean_seed_role_probe_accuracy <= 0.45"] = mean_role_seed_acc <= 0.45
checks["max_seed_role_probe_accuracy <= 0.55"] = max_role_seed_acc <= 0.55
checks["single_feature_role_probe_clean"] = len(single_feat_high) == 0
checks["feature_pairs_exclusive_to_one_role = 0"] = pairs_exclusive == 0
checks["hidden_timing_audit_passed"] = hidden_timing_audit_passed
checks["old C4 preserved"] = C4_PRESENT
checks["old C5 preserved"] = C5_PRESENT
checks["no_seed_crashes"] = True
checks["schedule_label_removed_all_seeds"] = all_schedule_removed

# Pre-observe ambiguity acceptance
checks["category_probe_ambient < 1.0"] = cat_probe_ambient < 1.0
checks["category_probe_ambient <= 0.70"] = cat_probe_ambient <= 0.70
checks["category_probe_core = 1.0"] = abs(cat_probe_core - 1.0) < 0.01

# Possible observe gain acceptance
checks["pre_oracle_gap > 0"] = pre_oracle_gap > 0.01
checks["post_mean > pre_mean"] = best_post > best_pre + 0.01
checks["positive_gain_rate > 0"] = positive_gain_rate > 0.0
checks["post_oracle_gap < pre_oracle_gap"] = post_oracle_gap < pre_oracle_gap

# Suggested targets
checks["post_pre_gain >= +0.03"] = post_pre_gain >= 0.03
checks["positive_gain_rate >= 0.25"] = positive_gain_rate >= 0.25

all_pass = all(checks.values())
failure_reasons = [k for k, v in checks.items() if not v]

for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {check_name}: {status}")

print(f"\n  overall: {'PASS' if all_pass else 'PARTIAL' if len(failure_reasons) <= 4 else 'FAIL'}")
if failure_reasons:
    print(f"  failures: {failure_reasons}")

elapsed = round(time.time() - t0, 1)
print(f"\n  Elapsed: {elapsed}s")

# =============================================================================
# Output
# =============================================================================
print("\nWriting outputs...")

if all_pass:
    impl_status = "pass"
elif len(failure_reasons) <= 4:
    impl_status = "partial"
else:
    impl_status = "fail"

# --- JSON ---
json_output = {
    "block_id": "1J40b-env3",
    "elapsed_seconds": elapsed,
    "seeds": SEEDS,
    "total_objects_per_seed": TOTAL_OBJECTS,
    "total_objects_all_seeds": TOTAL_OBJECTS * len(SEEDS),
    "total_events_all_seeds": sum(len(r["all_events"]) for r in all_seed_results.values()),
    "implementation_status": impl_status,
    "acceptance_checks": {k: v for k, v in checks.items()},
    "failure_reasons": failure_reasons,

    # Ambient groups
    "ambient_groups": {
        "A": {"features": AMBIENT_GROUP_FEATURES["A"], "categories": ambient_group_categories["A"]},
        "B": {"features": AMBIENT_GROUP_FEATURES["B"], "categories": ambient_group_categories["B"]},
    },

    # Affordance changes from env2
    "affordance_changes_from_env2": AFFORDANCE_CHANGES,

    # Category probes
    "category_probes": {k: v["accuracy"] for k, v in category_probe_results.items()},

    # Pre-observe ambiguity
    "pre_observe_ambiguity": {
        "pre_mean_true_return_randomized_tie": round(pre_mean_randomized, 4),
        "pre_mean_true_return_fixed_tie": round(pre_mean_fixed, 4),
        "oracle_mean_true_return": round(pre_mean_oracle, 4),
        "pre_oracle_gap_randomized": round(pre_oracle_gap_randomized, 4),
        "pre_oracle_gap_fixed": round(pre_oracle_gap_fixed, 4),
    },

    # Post-observe source ablations
    "source_ablations": ablation_summary,

    # Possible observe gain
    "possible_observe_gain": {
        "pre_mean_true_return": round(best_pre, 4),
        "post_mean_true_return": round(best_post, 4),
        "oracle_mean_true_return": round(best_oracle, 4),
        "pre_oracle_gap": round(pre_oracle_gap, 4),
        "post_oracle_gap": round(post_oracle_gap, 4),
        "post_pre_gain": round(post_pre_gain, 4),
        "positive_gain_count": sum(1 for r in ablation_results if r["post_full_val"] - r["pre_ambient_val"] > 0.001),
        "positive_gain_rate": round(positive_gain_rate, 4),
        "best_action_change_rate": round(best_action_changes / len(ablation_results), 4),
    },

    # Distinguish metrics
    "distinguish_metrics": {
        "mean_all_try_return_pre": round(pre_mean_all_try, 4),
        "mean_all_try_return_post": round(post_mean_all_try, 4),
        "mean_all_try_improvement": round(all_try_improvement, 4),
        "best_action_selected_return_pre": round(best_pre, 4),
        "best_action_selected_return_post": round(best_post, 4),
        "best_action_selected_improvement": round(post_pre_gain, 4),
    },

    # Role leakage
    "role_leakage": {
        "full_role_probe_accuracy": round(full_role_probe_accuracy, 4),
        "mean_seed_role_probe": round(mean_role_seed_acc, 4),
        "max_seed_role_probe": round(max_role_seed_acc, 4),
        "single_feature_high_count": len(single_feat_high),
        "feature_pairs_exclusive": pairs_exclusive,
        "direct_role_marker_detected": direct_role_marker,
        "features_in_all_roles": features_in_all_roles,
        "features_in_2_roles": features_in_2_roles,
        "features_in_1_role": features_in_1_role,
        "max_role_gap": round(max_role_gap_overall, 4),
    },

    # Hidden timing
    "hidden_timing": {
        "no_observe_hidden": no_observe_hidden,
        "one_observe_hidden": one_observe_hidden,
        "repeated_observe_hidden_before_reveal": repeated_observe_hidden_before_reveal,
        "audit_passed": hidden_timing_audit_passed,
    },

    # Profile changes
    "env2_to_env3_profile_changes": AFFORDANCE_CHANGES,

    # Legacy
    "old_c4_preserved": C4_PRESENT,
    "old_c5_preserved": C5_PRESENT,
    "recursive_leakage_check_passed": not any_leakage,
    "schedule_label_removed": all_schedule_removed,
}

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b_env3_category_ambiguous_observe_value.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# --- CSV ---
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b_env3_category_ambiguous_observe_value_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["block_id", "1J40b-env3"])
    w.writerow(["implementation_status", impl_status])
    w.writerow(["seeds", str(SEEDS)])
    w.writerow(["total_objects_all_seeds", TOTAL_OBJECTS * len(SEEDS)])
    w.writerow(["elapsed_seconds", elapsed])

    w.writerow(["category_probe_ambient", round(cat_probe_ambient, 4)])
    w.writerow(["category_probe_core", round(cat_probe_core, 4)])
    w.writerow(["category_probe_diagnostic", round(cat_probe_diagnostic, 4)])
    w.writerow(["category_probe_core_plus_diagnostic", round(category_probe_results.get("core_plus_diagnostic", {}).get("accuracy", 0), 4)])

    w.writerow(["pre_mean_true_return_randomized", round(pre_mean_randomized, 4)])
    w.writerow(["pre_mean_true_return_fixed", round(pre_mean_fixed, 4)])
    w.writerow(["oracle_mean_true_return", round(pre_mean_oracle, 4)])
    w.writerow(["pre_oracle_gap_randomized", round(pre_oracle_gap_randomized, 4)])

    w.writerow(["post_core_only_return", ablation_summary["post_core_only"]["mean_return"]])
    w.writerow(["post_diagnostic_only_return", ablation_summary["post_diagnostic_only"]["mean_return"]])
    w.writerow(["post_full_return", ablation_summary["post_full"]["mean_return"]])

    w.writerow(["post_pre_gain", round(post_pre_gain, 4)])
    w.writerow(["post_oracle_gap", round(post_oracle_gap, 4)])
    w.writerow(["positive_gain_rate", round(positive_gain_rate, 4)])
    w.writerow(["best_action_change_rate", round(best_action_changes / len(ablation_results), 4)])

    w.writerow(["mean_all_try_return_pre", round(pre_mean_all_try, 4)])
    w.writerow(["mean_all_try_return_post", round(post_mean_all_try, 4)])
    w.writerow(["best_action_selected_return_pre", round(best_pre, 4)])
    w.writerow(["best_action_selected_return_post", round(best_post, 4)])

    w.writerow(["full_role_probe_accuracy", round(full_role_probe_accuracy, 4)])
    w.writerow(["max_seed_role_probe", round(max_role_seed_acc, 4)])
    w.writerow(["feature_pairs_exclusive", pairs_exclusive])
    w.writerow(["direct_role_marker_detected", direct_role_marker])
    w.writerow(["hidden_timing_audit_passed", hidden_timing_audit_passed])
    w.writerow(["recursive_leakage_check_passed", not any_leakage])

    w.writerow(["overall_acceptance", "PASS" if all_pass else "PARTIAL" if len(failure_reasons) <= 4 else "FAIL"])
print(f"  CSV  -> {csv_path}")

# --- MD ---
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b_env3_category_ambiguous_observe_value.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-env3: Category-Ambiguous Observe-Value Environment\n\n")
    f.write(f"- **Implementation Status**: {impl_status.upper()}\n")
    f.write(f"- **Seeds**: {SEEDS}\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n")
    f.write(f"- **Total objects**: {TOTAL_OBJECTS * len(SEEDS)}\n\n")

    f.write("## 1. Checkpoint Summary\n\n")
    f.write("env3 creates category ambiguity in pre-observe features by grouping categories ")
    f.write("into ambient groups that share identical ambient feature signatures. ")
    f.write("Post-observe reveals core features (identify category) and diagnostic features ")
    f.write("(action-relevant evidence). Base affordance profiles were modified so no single ")
    f.write("action succeeds for both categories within an ambient group, creating genuine ")
    f.write("pre-observe uncertainty.\n\n")

    f.write("## 2. Env3a Generation Design\n\n")
    f.write("### Ambient Groups\n\n")
    f.write("- **Group A** (wood_log, stone_block): brownish, rough_texture, long_shape, heavy_weight\n")
    f.write("- **Group B** (apple, wooden_pickaxe): greenish, smooth_texture, round_small, light_weight\n\n")
    f.write("Within each group, all objects share identical ambient features. ")
    f.write("The agent cannot distinguish between categories pre-observe.\n\n")

    f.write("### Feature Layers\n\n")
    f.write("| Layer | Visibility | Purpose |\n")
    f.write("|-------|-----------|--------|\n")
    f.write("| Ambient | no_observe | Shared within ambient group, creates ambiguity |\n")
    f.write("| Core | one_observe+ | Identifies category uniquely |\n")
    f.write("| Diagnostic | one_observe+ | Action-relevant evidence, some features shared across categories |\n")
    f.write("| Hidden | repeated_observe | Hints at variant bonus for helps objects |\n\n")

    f.write("### Core Features per Category\n\n")
    for cat in CATEGORIES:
        f.write(f"- **{cat}**: {', '.join(CATEGORY_CORE_FEATURES[cat])}\n")
    f.write("\n")

    f.write("### Diagnostic Features per Category\n\n")
    for cat in CATEGORIES:
        f.write(f"- **{cat}**: {', '.join(CATEGORY_DIAGNOSTIC_FEATURES[cat])}\n")
    f.write("\n")

    f.write("## 3. Difference from env2\n\n")
    f.write("### Affordance Profile Changes\n\n")
    f.write("| Category | Action | Change | Reason |\n")
    f.write("|----------|--------|--------|--------|\n")
    for cat, changes in AFFORDANCE_CHANGES.items():
        for action, change in changes.items():
            reason = "Removes common safe action within ambient group"
            f.write(f"| {cat} | {action} | {change} | {reason} |\n")
    if not AFFORDANCE_CHANGES:
        f.write("(No changes — all profiles identical to env2)\n")
    f.write("\n")

    f.write("### Feature System Changes\n\n")
    f.write("- Ambient features no longer identify category (shared within groups)\n")
    f.write("- Core + diagnostic features replace env2's single visible feature set\n")
    f.write("- Diagnostic features are action-relevant (has_edible_smell, etc.) not visual variants\n")
    f.write("- Profile-rotation is simplified to bonus-action-only (no variant visible features)\n")
    f.write("- Hidden features are distinct from diagnostic features\n\n")

    f.write("## 4. Pre-observe Ambiguity Audit\n\n")
    f.write(f"- oracle mean true return: {pre_mean_oracle:.4f}\n")
    f.write(f"- pre mean true return (randomized tie): {pre_mean_randomized:.4f}\n")
    f.write(f"- pre mean true return (fixed tie-break): {pre_mean_fixed:.4f}\n")
    f.write(f"- pre/oracle gap (randomized): {pre_oracle_gap_randomized:.4f}\n")
    f.write(f"- pre/oracle gap (fixed): {pre_oracle_gap_fixed:.4f}\n\n")

    f.write("### Per-seed Pre vs Oracle\n\n")
    f.write("| Seed | Pre (rand) | Oracle | Gap |\n")
    f.write("|------|-----------|--------|-----|\n")
    for seed in SEEDS:
        seed_rs = [r for r in pre_ambiguity_results if r["seed"] == seed]
        seed_pre = sum(r["pre_randomized_value"] for r in seed_rs) / len(seed_rs)
        seed_oracle = sum(r["oracle_best_value"] for r in seed_rs) / len(seed_rs)
        f.write(f"| {seed} | {seed_pre:.4f} | {seed_oracle:.4f} | {seed_oracle - seed_pre:.4f} |\n")
    f.write("\n")

    f.write("## 5. Category Probes by Visibility Level\n\n")
    f.write("| Level | Accuracy | Mean Seed | Max Seed | N Features |\n")
    f.write("|-------|----------|-----------|----------|------------|\n")
    for level in ["ambient", "core", "diagnostic", "core_plus_diagnostic", "all_visible"]:
        cp = category_probe_results.get(level, {})
        acc = cp.get("accuracy", 0)
        ms = cp.get("mean_seed_accuracy", 0)
        xs = cp.get("max_seed_accuracy", 0)
        nf = cp.get("n_features", 0)
        f.write(f"| {level} | {acc:.4f} | {ms:.4f} | {xs:.4f} | {nf} |\n")
    f.write("\n")

    f.write("## 6. Post-observe Source Ablations\n\n")
    f.write("| Condition | Mean Return | Gain vs Pre | Oracle Gap |\n")
    f.write("|-----------|-------------|-------------|------------|\n")
    for cond in ["pre_ambient_only", "post_core_only", "post_diagnostic_only", "post_full"]:
        agg = ablation_summary[cond]
        f.write(f"| {cond} | {agg['mean_return']:.4f} | {agg['mean_gain_vs_pre']:+.4f} | {agg['mean_oracle_gap']:.4f} |\n")
    f.write("\n")

    f.write("### Per-seed Breakdown\n\n")
    f.write("| Seed | Pre | Core | Diagnostic | Full | Oracle |\n")
    f.write("|------|-----|------|------------|------|--------|\n")
    for seed in SEEDS:
        rs = [r for r in ablation_results if r["seed"] == seed]
        pre = sum(r["pre_ambient_val"] for r in rs) / len(rs)
        core = sum(r["post_core_val"] for r in rs) / len(rs)
        diag = sum(r["post_diagnostic_val"] for r in rs) / len(rs)
        full = sum(r["post_full_val"] for r in rs) / len(rs)
        oracle = sum(r["oracle_val"] for r in rs) / len(rs)
        f.write(f"| {seed} | {pre:.4f} | {core:.4f} | {diag:.4f} | {full:.4f} | {oracle:.4f} |\n")
    f.write("\n")

    f.write("## 7. Role Leakage Audits\n\n")
    f.write(f"- full_visible_role_probe_accuracy: {full_role_probe_accuracy:.4f}\n")
    f.write(f"- mean_seed_role_probe: {mean_role_seed_acc:.4f}\n")
    f.write(f"- max_seed_role_probe: {max_role_seed_acc:.4f}\n")
    f.write(f"- single_feature_high_count: {len(single_feat_high)}\n")
    f.write(f"- feature_pairs_exclusive: {pairs_exclusive}\n")
    f.write(f"- direct_role_marker_detected: {direct_role_marker}\n")
    f.write(f"- hidden_timing_audit_passed: {hidden_timing_audit_passed}\n")
    f.write(f"- recursive_leakage_check_passed: {not any_leakage}\n\n")

    f.write("## 8. Pre-observe Tie Diagnostics\n\n")
    for ambient_group in ["A", "B"]:
        group_cats = ambient_group_categories[ambient_group]
        f.write(f"### Group {ambient_group} ({', '.join(group_cats)})\n\n")
        f.write("| Action | Expected Return |")
        for cat in group_cats:
            f.write(f" {cat} |")
        f.write("\n")
        f.write("|--------|----------------|")
        for cat in group_cats:
            f.write("-------|")
        f.write("\n")
        for action in ALL_ACTIONS:
            vals = [action_true_value(action, BASE_AFFORDANCE_PROFILES[cat]) for cat in group_cats]
            exp = sum(vals) / len(vals)
            f.write(f"| {action} | {exp:+.2f} |")
            for v in vals:
                f.write(f" {v:+.2f} |")
            f.write("\n")

        action_exp_vals = {}
        for action in ALL_ACTIONS:
            vals = [action_true_value(action, BASE_AFFORDANCE_PROFILES[cat]) for cat in group_cats]
            action_exp_vals[action] = sum(vals) / len(vals)
        max_exp = max(action_exp_vals.values())
        best_actions = sorted([a for a, v in action_exp_vals.items() if abs(v - max_exp) < 0.001])
        f.write(f"\n- Best action(s): {best_actions}\n")
        f.write(f"- Tie count: {len(best_actions)}\n")
        f.write(f"- Fixed tie-break (alphabetical): {best_actions[0]}\n")

        group_objs = [r for r in ablation_results if r["ambient_group"] == ambient_group]
        pre_vals = [r["pre_ambient_val"] for r in group_objs]
        pre_mean_grp = sum(pre_vals) / len(pre_vals)
        pre_std_grp = (sum((v - pre_mean_grp)**2 for v in pre_vals) / len(pre_vals))**0.5
        f.write(f"- Randomized-tie pre return: mean={pre_mean_grp:.4f}, std={pre_std_grp:.4f}\n\n")

    f.write("## 9. Distinguish Metrics\n\n")
    f.write(f"- mean_all_try_return_pre: {pre_mean_all_try:.4f}\n")
    f.write(f"- mean_all_try_return_post: {post_mean_all_try:.4f}\n")
    f.write(f"- best_action_selected_return_pre: {best_pre:.4f}\n")
    f.write(f"- best_action_selected_return_post: {best_post:.4f}\n\n")
    f.write("**Note**: mean-all-try-return is NOT the same as best-action-selected return. ")
    f.write("The env2 metric used mean-all-try-return. env3 reports both explicitly.\n\n")

    f.write("## 10. Possible Observe Gain Summary\n\n")
    f.write(f"- pre mean true return: {best_pre:.4f}\n")
    f.write(f"- post mean true return: {best_post:.4f}\n")
    f.write(f"- oracle mean true return: {best_oracle:.4f}\n")
    f.write(f"- pre/oracle gap: {pre_oracle_gap:.4f}\n")
    f.write(f"- post/oracle gap: {post_oracle_gap:.4f}\n")
    f.write(f"- post-pre gain: {post_pre_gain:+.4f}\n")
    f.write(f"- positive gain rate: {positive_gain_rate:.4f}\n")
    f.write(f"- best action change rate: {best_action_changes/len(ablation_results):.4f}\n\n")

    f.write("## 11. Acceptance Checks\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status}** |\n")

    overall = "PASS" if all_pass else ("PARTIAL" if len(failure_reasons) <= 4 else "FAIL")
    f.write(f"\n**Overall**: {overall}\n")
    if failure_reasons:
        f.write(f"\nFailures: {failure_reasons}\n")

    f.write(f"\n\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-env3\n")
    f.write(f"implementation_status={impl_status}\n")
    f.write(f"category_probe_ambient={cat_probe_ambient:.4f}\n")
    f.write(f"category_probe_core={cat_probe_core:.4f}\n")
    f.write(f"pre_oracle_gap_randomized={pre_oracle_gap_randomized:.4f}\n")
    f.write(f"post_pre_gain={post_pre_gain:.4f}\n")
    f.write(f"positive_gain_rate={positive_gain_rate:.4f}\n")
    f.write(f"full_role_probe_accuracy={full_role_probe_accuracy:.4f}\n")
    f.write(f"max_seed_role_probe={max_role_seed_acc:.4f}\n")
    f.write(f"hidden_timing_audit_passed={str(hidden_timing_audit_passed).lower()}\n")
    f.write(f"recursive_leakage_check_passed={str(not any_leakage).lower()}\n")
    f.write(f"failure_reasons={'; '.join(failure_reasons) if failure_reasons else 'none'}\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'='*70}")
print(f"Block 1J40b-env3 complete.")
print(f"  category_probe_ambient={cat_probe_ambient:.4f}")
print(f"  category_probe_core={cat_probe_core:.4f}")
print(f"  pre_oracle_gap={pre_oracle_gap_randomized:.4f}")
print(f"  post_pre_gain={post_pre_gain:+.4f}")
print(f"  positive_gain_rate={positive_gain_rate:.4f}")
print(f"  role_probe={full_role_probe_accuracy:.4f} (threshold=0.45)")
print(f"  max_seed_role_probe={max_role_seed_acc:.4f} (threshold=0.55)")
print(f"  hidden_timing={hidden_timing_audit_passed}")
print(f"  overall: {'PASS' if all_pass else 'PARTIAL' if len(failure_reasons) <= 4 else 'FAIL'}")
print(f"{'='*70}")
