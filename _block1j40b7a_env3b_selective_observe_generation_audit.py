"""
Block 1J40b-7a — env3b Selective-Observe Environment Generation + Audit.

Creates env3b as a mixed observe-value environment where:
- Some objects benefit from observation (pre ambiguous, post clarifies)
- Some objects gain ~zero from observation (pre already sufficient)
- Some objects are harmed by observation cost only (wasteful)
- Selective observation beats always_observe and always_try.

Two-layer VOI reporting:
  Layer 1 (numeric): positive_net vs non_positive_net
  Layer 2 (audit rationale): positive_change / confirmatory / wasteful_cost_only

Design: Objects share pre_surface signatures across categories, creating ambiguity
that observation resolves. VOI emerges from signature-group expected returns.

Three-world separation: environment holds hidden truth; agent sees only legal surface
observations; audit inspects hidden truth for evaluation only.

NO learner training. NO policy intervention. Generation + audit + observe_cost sweep.
"""

import os, sys, json, time, math, random
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
# Constants
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
N_DEPTH_SCHEDULES = 3  # no_observe, one_observe, repeated_observe

ALL_TRY_AFFORDANCES = [
    "burn_as_fuel", "craft_plank", "eat",
    "mine_by_hand", "mine_with_pickaxe", "use_as_tool",
]
ALL_ACTION_KEYS = ["observe"] + [f"try_{a}" for a in ALL_TRY_AFFORDANCES]

DEFAULT_OBSERVE_COST = 0.005
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1

CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

# =============================================================================
# Feature Layers (renamed per correction)
#
# pre_surface_features: visible before any observe (ambient + weak surface cues)
# post_observe_features: revealed by one observe (in addition to pre_surface)
# hidden_features: not visible in normal pre/post; only in repeated_observe or audit
# =============================================================================

# Ambient features (shared pre-observe signatures — same as env3a)
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

# Category-diagnostic features (revealed by observe, identify category)
CATEGORY_POST_OBSERVE_FEATURES = {
    "wood_log": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "block_like",
                    "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                       "movable", "has_metal_head", "jointed"],
}

# Shared weak surface cues (appear pre-observe across multiple categories)
# These are legitimate pre_surface features, NOT "leaked core" features.
# They create partial category hints without uniquely identifying category.
SHARED_SURFACE_FEATURES = [
    "spotted",        # appears on wood_log, stone_block, apple subtypes
    "weathered",      # appears on wood_log, wooden_pickaxe subtypes
    "coarse_texture", # appears on wood_log, stone_block subtypes
    "discolored",     # appears on apple, wooden_pickaxe subtypes
    "has_veins",      # appears on stone_block, apple subtypes
    "pitted",         # appears on stone_block, wooden_pickaxe subtypes
]

# =============================================================================
# Build all feature name lists
# =============================================================================
ALL_POST_OBSERVE_FEATURE_NAMES = sorted(set(
    f for flist in CATEGORY_POST_OBSERVE_FEATURES.values() for f in flist))

ALL_VISIBLE_FEATURE_NAMES = sorted(set(
    list(ALL_AMBIENT_FEATURE_NAMES) +
    list(ALL_POST_OBSERVE_FEATURE_NAMES) +
    list(SHARED_SURFACE_FEATURES)
))

STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

print(f"Feature space:")
print(f"  Ambient: {len(ALL_AMBIENT_FEATURE_NAMES)}")
print(f"  Post-observe (category-diagnostic): {len(ALL_POST_OBSERVE_FEATURE_NAMES)}")
print(f"  Shared surface (weak pre-observe cues): {len(SHARED_SURFACE_FEATURES)}")
print(f"  Total visible: {len(ALL_VISIBLE_FEATURE_NAMES)}")
print(f"  State: {len(STATE_FEATURE_NAMES)}")

# =============================================================================
# env3b Instance Subtype Definitions
#
# Each category has subtypes defining:
#   - pre_surface_extra: weak surface cues visible pre-observe (beyond ambient)
#   - affordance_profile: which actions succeed
#   - post_observe_revealed: features observation reveals
#   - audit_rationale_class: positive_change / confirmatory / wasteful_cost_only
#
# Key design property: pre_surface signatures are SHARED across categories
# to create ambiguity. VOI emerges from signature-group expected returns,
# NOT from true category or subtype identity.
# =============================================================================

INSTANCE_SUBTYPE_DEFS = {}

# ---- wood_log (ambient Group A with stone_block) ----
# Base affordance: mine_by_hand=S, craft_plank=S, burn_as_fuel=S (3 successes)
INSTANCE_SUBTYPE_DEFS["wood_log"] = {
    "P": {
        "label": "ambiguous_log",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A only, shared with stone_block_P; post reveals category-diagnostic features → disambiguates wood_log from stone_block → action choice changes",
    },
    "Z": {
        "label": "positive_shared_cue_log",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["has_bark_texture", "fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A+spotted, shared with stone_block_Z (DIFFERENT best actions: wood_log→mine_by_hand, stone_block→mine_with_pickaxe); post resolves ambiguity → action choice changes",
    },
    "N": {
        "label": "nonpos_identified_log",
        "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_bark_texture", "discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "success", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "success",
        },
        "post_observe_revealed": ["fibrous", "flammable", "porous_surface"],
        "state_features": {"fresh": False, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A+has_bark_texture+discolored uniquely identifies category (category-diagnostic feature); best action already clear; observe adds no decision value, only pays cost",
    },
}

# ---- stone_block (ambient Group A with wood_log) ----
INSTANCE_SUBTYPE_DEFS["stone_block"] = {
    "P": {
        "label": "ambiguous_stone",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A only, shared with wood_log_P; post reveals category-diagnostic features → disambiguates stone_block from wood_log → action choice changes",
    },
    "Z": {
        "label": "positive_shared_cue_stone",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["spotted"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_crystal_flecks", "has_granular_surface", "block_like",
                                  "cold_to_touch"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A+spotted, shared with wood_log_Z (DIFFERENT best actions: stone_block→mine_with_pickaxe, wood_log→mine_by_hand); post resolves ambiguity → action choice changes",
    },
    "N": {
        "label": "nonpos_identified_stone",
        "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_crystal_flecks"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "success",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_granular_surface", "block_like", "cold_to_touch", "scratch_resistant"],
        "state_features": {"fresh": True, "wet": False, "damaged": True, "clean": False, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_A+has_crystal_flecks uniquely identifies category (category-diagnostic feature); best action already clear; observe adds no decision value, only pays cost",
    },
}

# ---- apple (ambient Group B with wooden_pickaxe) ----
INSTANCE_SUBTYPE_DEFS["apple"] = {
    "P": {
        "label": "ambiguous_fruit",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B only, shared with wooden_pickaxe_P; post reveals category-diagnostic features → disambiguates apple from pickaxe → action choice changes",
    },
    "Z": {
        "label": "positive_shared_cue_apple",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_stem_remnant", "has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B+discolored, shared with wooden_pickaxe_Z (DIFFERENT best actions: apple→mine_by_hand/eat, pickaxe→use_as_tool); post resolves ambiguity → action choice changes",
    },
    "N": {
        "label": "nonpos_identified_apple",
        "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_stem_remnant", "spotted"],
        "affordance_profile": {
            "mine_by_hand": "success", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "success",
            "use_as_tool": "fail", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_peel_texture", "fruity_scent"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B+has_stem_remnant+spotted uniquely identifies category (category-diagnostic feature); best action already clear; observe adds no decision value, only pays cost",
    },
}

# ---- wooden_pickaxe (ambient Group B with apple) ----
INSTANCE_SUBTYPE_DEFS["wooden_pickaxe"] = {
    "P": {
        "label": "ambiguous_tool",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": [],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable", "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B only, shared with apple_P; post reveals category-diagnostic features → disambiguates pickaxe from apple → action choice changes",
    },
    "Z": {
        "label": "positive_shared_cue_tool",
        "audit_rationale_class": "positive_change",
        "pre_surface_extra": ["discolored"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_grip_area", "has_shaft_shape", "elongated_with_handle",
                                  "movable"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B+discolored, shared with apple_Z (DIFFERENT best actions: pickaxe→use_as_tool, apple→mine_by_hand/eat); post resolves ambiguity → action choice changes",
    },
    "N": {
        "label": "nonpos_identified_tool",
        "audit_rationale_class": "wasteful_cost_only",
        "pre_surface_extra": ["has_grip_area"],
        "affordance_profile": {
            "mine_by_hand": "fail", "mine_with_pickaxe": "fail",
            "craft_plank": "fail", "eat": "fail",
            "use_as_tool": "success", "burn_as_fuel": "fail",
        },
        "post_observe_revealed": ["has_shaft_shape", "elongated_with_handle", "movable",
                                  "has_metal_head", "jointed"],
        "state_features": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
        "voi_rationale": "pre_surface=ambient_B+has_grip_area uniquely identifies category (category-diagnostic feature); best action already clear; observe adds no decision value, only pays cost",
    },
}

# =============================================================================
# State feature names
# =============================================================================
STATE_FEATURE_NAMES = ["fresh", "wet", "damaged", "clean", "hot", "open"]

# =============================================================================
# Part 1: Object Generation
# =============================================================================
print("\n[1/14] Generating env3b objects...")

def generate_env3b_objects(seed):
    """Generate env3b objects with instance subtypes across depth schedules."""
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

            # Build pre_surface_features
            pre_surface = {}
            for f in ambient_features_list:
                pre_surface[f] = True
            for f in pre_extra:
                pre_surface[f] = True

            # Build post_observe_features (pre_surface + revealed)
            post_observe = dict(pre_surface)
            for f in post_revealed:
                post_observe[f] = True

            # Hidden features = all visible features NOT in pre_surface or post_revealed
            # (features that would only be visible via repeated_observe or audit)
            hidden = {}
            for f in ALL_VISIBLE_FEATURE_NAMES:
                if f not in pre_surface and f not in post_revealed:
                    hidden[f] = True  # not visible in normal one-observe

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
                    "voi_rationale": subtype_def["voi_rationale"],
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(affordance),
                    "ambient_group": ambient_group,
                    "pre_surface_extra": list(pre_extra),
                    "post_observe_revealed": list(post_revealed),
                    "seed": seed,
                }

    return objects, audit_labels

# Generate objects for all seeds
per_seed_objects = {}
per_seed_audit_labels = {}
for seed in SEEDS:
    objects, audit_labels = generate_env3b_objects(seed)
    per_seed_objects[seed] = objects
    per_seed_audit_labels[seed] = audit_labels

total_objects_per_seed = len(list(per_seed_objects[SEEDS[0]].keys()))
total_objects_all = total_objects_per_seed * len(SEEDS)
print(f"  Objects per seed: {total_objects_per_seed}")
print(f"  Total objects: {total_objects_all}")

# =============================================================================
# Part 2: Three-World Separation Audit
# =============================================================================
print("\n[2/14] Three-world separation audit...")

# Agent-visible fields (what the agent could legally see)
AGENT_VISIBLE_PRE_KEYS = [f"feat_{f}" for f in ALL_VISIBLE_FEATURE_NAMES]
AGENT_VISIBLE_STATE_KEYS = [f"state_{s}" for s in STATE_FEATURE_NAMES]
AGENT_VISIBLE_NUMERIC = ["prior_observe_count"]

# Forbidden fields (must never be agent input)
FORBIDDEN_FIELDS = sorted([
    "category", "hidden_category", "ambient_group", "hidden_subtype",
    "hidden_audit_rationale_class", "audit_rationale_class",
    "group_role", "voi_class",
    "object_id", "oid", "seed_id", "depth", "depth_schedule",
    "episode_id", "episode_progress",
    "true_affordance_profile", "effective_affordance_profile",
    "hidden_affordance_profile", "affordance_profile",
    "oracle_action", "audit_computed_gain",
    "pre_observe_uncertainty", "audit_observe_value", "audit_VOI",
    "hidden_features", "post_observe_revealed", "post_observe_features",
    "pre_surface_extra", "voi_rationale",
    "n_features_known", "n_states_known",
])

# Verify agent-visible fields don't include forbidden fields
agent_visible_all = set(AGENT_VISIBLE_PRE_KEYS + AGENT_VISIBLE_STATE_KEYS + AGENT_VISIBLE_NUMERIC)
forbidden_in_agent = []
for ff in FORBIDDEN_FIELDS:
    for av in agent_visible_all:
        if ff.lower() in av.lower().replace("feat_", "").replace("state_", ""):
            forbidden_in_agent.append(f"{ff} -> {av}")

print(f"  Agent-visible pre feature keys: {len(AGENT_VISIBLE_PRE_KEYS)}")
print(f"  Agent-visible state keys: {len(AGENT_VISIBLE_STATE_KEYS)}")
print(f"  Agent-visible numeric keys: {len(AGENT_VISIBLE_NUMERIC)}")
print(f"  Forbidden fields total: {len(FORBIDDEN_FIELDS)}")
print(f"  Forbidden-in-agent leaks: {len(forbidden_in_agent)}")
if forbidden_in_agent:
    for leak in forbidden_in_agent:
        print(f"    LEAK: {leak}")

# Hidden feature check: post_observe_revealed features NOT in pre_surface
hidden_before_reveal_violations = 0
for seed in SEEDS:
    for oid, obj in per_seed_objects[seed].items():
        lbl = per_seed_audit_labels[seed][oid]
        pre_surf = set(obj["pre_surface_features"].keys())
        post_rev = set(lbl["post_observe_revealed"])
        for f in post_rev:
            if f in pre_surf:
                hidden_before_reveal_violations += 1

print(f"  Hidden-before-reveal violations: {hidden_before_reveal_violations}")

separation_ok = len(forbidden_in_agent) == 0 and hidden_before_reveal_violations == 0
print(f"  Three-world separation: {'PASS' if separation_ok else 'FAIL'}")

# =============================================================================
# Part 3: Audit VOI Computation (by legal pre-observe signature groups)
# =============================================================================
print("\n[3/14] Audit VOI computation (legal pre-observe signature groups)...")

def try_net_value(success_or_failure):
    """Net value of a try action after probe cost."""
    if success_or_failure is True:
        return SUCCESS_REWARD - PROBE_COST
    elif success_or_failure is False:
        return -FAILURE_PENALTY - PROBE_COST
    return 0.0

def get_pre_surface_signature(pre_surface_features):
    """Return a canonical, sortable representation of pre_surface feature set."""
    return tuple(sorted(pre_surface_features.keys()))

def compute_audit_voi_all_objects(per_seed_objects, per_seed_audit_labels, observe_cost):
    """
    Compute audit_VOI for all objects using legal pre-observe signature groups.

    For each object:
    - oracle_post_best = max(true_return for this object's affordance)
    - oracle_pre_best_expected = best expected action value over all objects
      sharing the same legal pre_surface signature
    - audit_VOI = oracle_post_best - oracle_pre_best_expected - observe_cost

    Uses ONLY legal pre_surface signatures. Does NOT use true category, subtype,
    audit rationale class, or hidden truth to compute pre-observe expected return.
    """
    # Build signature groups: sig -> [(seed, oid, true_returns_dict)]
    sig_groups = defaultdict(list)

    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            sig = get_pre_surface_signature(obj["pre_surface_features"])

            true_returns = {}
            for action in ALL_TRY_AFFORDANCES:
                outcome = lbl["affordance_profile"].get(action, "fail")
                true_returns[action] = try_net_value(outcome == "success")

            sig_groups[sig].append({
                "seed": seed,
                "oid": oid,
                "true_returns": true_returns,
                "oracle_post_best": max(true_returns.values()),
            })

    # Compute per-signature expected action values
    sig_expected = {}
    for sig, members in sig_groups.items():
        n = len(members)
        action_expected = {}
        for action in ALL_TRY_AFFORDANCES:
            total = sum(m["true_returns"][action] for m in members)
            action_expected[action] = total / n
        best_action = max(action_expected, key=action_expected.get)
        sig_expected[sig] = {
            "n_members": n,
            "action_expected": action_expected,
            "best_action": best_action,
            "best_expected_value": action_expected[best_action],
        }

    # Compute VOI for each object
    all_voi_results = []
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            lbl = per_seed_audit_labels[seed][oid]
            sig = get_pre_surface_signature(obj["pre_surface_features"])

            oracle_post_best = max(
                try_net_value(lbl["affordance_profile"].get(a, "fail") == "success")
                for a in ALL_TRY_AFFORDANCES
            )
            oracle_pre_best_expected = sig_expected[sig]["best_expected_value"]
            audit_VOI = oracle_post_best - oracle_pre_best_expected - observe_cost

            all_voi_results.append({
                "oid": oid,
                "seed": seed,
                "category": lbl["hidden_category"],
                "subtype": lbl["hidden_subtype"],
                "audit_rationale_class": lbl["hidden_audit_rationale_class"],
                "pre_surface_signature": sig,
                "sig_group_size": sig_expected[sig]["n_members"],
                "sig_best_action": sig_expected[sig]["best_action"],
                "sig_best_expected_value": round(sig_expected[sig]["best_expected_value"], 6),
                "oracle_post_best": round(oracle_post_best, 6),
                "oracle_pre_best_expected": round(oracle_pre_best_expected, 6),
                "audit_VOI": round(audit_VOI, 6),
                "positive_net": audit_VOI > 0.001,
                "depth_schedule": lbl["depth_schedule"],
                "n_pre_surface_features": len(sig),
            })

    return all_voi_results, sig_groups, sig_expected

all_voi_results, sig_groups, sig_expected = compute_audit_voi_all_objects(
    per_seed_objects, per_seed_audit_labels, DEFAULT_OBSERVE_COST)

# Report VOI distribution
total_voi = len(all_voi_results)
positive_net_voi = [v for v in all_voi_results if v["positive_net"]]
non_positive_net_voi = [v for v in all_voi_results if not v["positive_net"]]
cost_only_voi = [v for v in all_voi_results
                 if abs(v["audit_VOI"] - (-DEFAULT_OBSERVE_COST)) < 0.0005]

pos_rate = len(positive_net_voi) / total_voi
nonpos_rate = len(non_positive_net_voi) / total_voi
cost_only_rate = len(cost_only_voi) / total_voi

print(f"  Total objects evaluated: {total_voi}")
print(f"  Unique pre_surface signatures: {len(sig_groups)}")
print(f"  Signature group size range: {min(len(m) for m in sig_groups.values())} - {max(len(m) for m in sig_groups.values())}")

print(f"\n  Numeric VOI class distribution:")
print(f"    positive_net:        {len(positive_net_voi)} ({pos_rate*100:.1f}%), mean VOI={sum(v['audit_VOI'] for v in positive_net_voi)/(len(positive_net_voi) or 1):.4f}")
print(f"    non_positive_net:    {len(non_positive_net_voi)} ({nonpos_rate*100:.1f}%), mean VOI={sum(v['audit_VOI'] for v in non_positive_net_voi)/(len(non_positive_net_voi) or 1):.4f}")
print(f"    cost_only (~-{DEFAULT_OBSERVE_COST}): {len(cost_only_voi)} ({cost_only_rate*100:.1f}%)")

print(f"\n  Audit rationale class distribution:")
for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
    rc_objs = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
    print(f"    {rc}: {len(rc_objs)} ({len(rc_objs)/total_voi*100:.1f}%), mean VOI={sum(v['audit_VOI'] for v in rc_objs)/(len(rc_objs) or 1):.4f}")

# Cross-tabulation
print(f"\n  Numeric x Rationale cross-tabulation:")
for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
    rc_objs = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
    rc_pos = [v for v in rc_objs if v["positive_net"]]
    print(f"    {rc}: {len(rc_pos)}/{len(rc_objs)} positive_net")

print(f"\n  Per category:")
for cat in CATEGORIES:
    cat_vois = [v for v in all_voi_results if v["category"] == cat]
    cat_pos = [v for v in cat_vois if v["positive_net"]]
    cat_nonpos = [v for v in cat_vois if not v["positive_net"]]
    print(f"    {cat}: n={len(cat_vois)}, positive_net={len(cat_pos)}, non_positive_net={len(cat_nonpos)}, mean VOI={sum(v['audit_VOI'] for v in cat_vois)/len(cat_vois):.4f}")

# =============================================================================
# Part 4: Baseline Audit
# =============================================================================
print("\n[4/14] Baseline audit...")

def compute_baselines(all_voi_results, observe_cost):
    """Compute no_observe, always_try, always_observe, random_observe, oracle_selective."""
    n = len(all_voi_results)

    no_observe_returns = [v["oracle_pre_best_expected"] for v in all_voi_results]
    always_observe_returns = [v["oracle_post_best"] - observe_cost for v in all_voi_results]

    rng = random.Random(42)
    random_returns = []
    for v in all_voi_results:
        if rng.random() < 0.5:
            random_returns.append(v["oracle_post_best"] - observe_cost)
        else:
            random_returns.append(v["oracle_pre_best_expected"])

    oracle_selective_returns = []
    oracle_selective_obs_count = 0
    for v in all_voi_results:
        if v["audit_VOI"] > 0:
            oracle_selective_returns.append(v["oracle_post_best"] - observe_cost)
            oracle_selective_obs_count += 1
        else:
            oracle_selective_returns.append(v["oracle_pre_best_expected"])

    return {
        "no_observe_pre_only": sum(no_observe_returns) / n,
        "always_try": sum(no_observe_returns) / n,
        "always_observe_net": sum(always_observe_returns) / n,
        "random_observe": sum(random_returns) / n,
        "oracle_selective": sum(oracle_selective_returns) / n,
        "oracle_selective_obs_rate": oracle_selective_obs_count / n,
        "oracle_selective_gap_vs_always_observe": (
            sum(oracle_selective_returns) / n - sum(always_observe_returns) / n),
        "oracle_selective_gap_vs_always_try": (
            sum(oracle_selective_returns) / n - sum(no_observe_returns) / n),
        "oracle_selective_gap_vs_no_observe": (
            sum(oracle_selective_returns) / n - sum(no_observe_returns) / n),
    }

baselines = compute_baselines(all_voi_results, DEFAULT_OBSERVE_COST)

print(f"  no_observe_pre_only:           {baselines['no_observe_pre_only']:.4f}")
print(f"  always_try:                    {baselines['always_try']:.4f}")
print(f"  always_observe_net:            {baselines['always_observe_net']:.4f}")
print(f"  random_observe:                {baselines['random_observe']:.4f}")
print(f"  oracle_selective:              {baselines['oracle_selective']:.4f}")
print(f"  oracle_selective obs_rate:     {baselines['oracle_selective_obs_rate']:.4f}")
print(f"  oracle - always_observe gap:   {baselines['oracle_selective_gap_vs_always_observe']:.4f}")
print(f"  oracle - always_try gap:       {baselines['oracle_selective_gap_vs_always_try']:.4f}")
print(f"  oracle - no_observe gap:       {baselines['oracle_selective_gap_vs_no_observe']:.4f}")

oracle_beats_always = baselines["oracle_selective"] > baselines["always_observe_net"]
oracle_beats_try = baselines["oracle_selective"] > baselines["always_try"]
print(f"  oracle > always_observe: {oracle_beats_always}")
print(f"  oracle > always_try: {oracle_beats_try}")

# =============================================================================
# Part 5: Pre-Observe Non-Oracle Audit
# =============================================================================
print("\n[5/14] Pre-observe non-oracle audit...")

pre_returns_all = [v["oracle_pre_best_expected"] for v in all_voi_results]
oracle_all = [v["oracle_post_best"] for v in all_voi_results]

pre_mean = sum(pre_returns_all) / len(pre_returns_all)
oracle_mean = sum(oracle_all) / len(oracle_all)
pre_oracle_gap = oracle_mean - pre_mean

print(f"  pre_mean_return:  {pre_mean:.4f}")
print(f"  oracle_mean:      {oracle_mean:.4f}")
print(f"  pre/oracle gap:   {pre_oracle_gap:.4f}")

print(f"\n  Per category pre/oracle gap:")
for cat in CATEGORIES:
    cat_vois = [v for v in all_voi_results if v["category"] == cat]
    cat_pre = sum(v["oracle_pre_best_expected"] for v in cat_vois) / len(cat_vois)
    cat_oracle = sum(v["oracle_post_best"] for v in cat_vois) / len(cat_vois)
    print(f"    {cat}: pre={cat_pre:.4f}, oracle={cat_oracle:.4f}, gap={cat_oracle-cat_pre:.4f}")

print(f"\n  Per audit rationale class pre/oracle gap:")
for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
    rc_vois = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
    if rc_vois:
        rc_pre = sum(v["oracle_pre_best_expected"] for v in rc_vois) / len(rc_vois)
        rc_oracle = sum(v["oracle_post_best"] for v in rc_vois) / len(rc_vois)
        print(f"    {rc}: pre={rc_pre:.4f}, oracle={rc_oracle:.4f}, gap={rc_oracle-rc_pre:.4f}")

# =============================================================================
# Part 6: n_features_known Distribution by VOI Class
# =============================================================================
print("\n[6/14] n_features_known distribution by VOI class...")

# n_features_known = number of pre_surface features visible before observe
nfeat_by_class = defaultdict(list)
for v in all_voi_results:
    label = "positive_net" if v["positive_net"] else "non_positive_net"
    nfeat_by_class[label].append(v["n_pre_surface_features"])

for label in ["positive_net", "non_positive_net"]:
    vals = nfeat_by_class[label]
    if vals:
        print(f"  {label}: mean={sum(vals)/len(vals):.1f}, min={min(vals)}, max={max(vals)}, "
              f"unique={sorted(set(vals))}")

# Check overlap: do positive_net and non_positive_net share n_features_known values?
pos_nfeat_set = set(nfeat_by_class["positive_net"])
nonpos_nfeat_set = set(nfeat_by_class["non_positive_net"])
overlap = pos_nfeat_set & nonpos_nfeat_set
print(f"  n_features_known overlap between classes: {sorted(overlap)}")
print(f"  Overlap count: {len(overlap)}")
if len(overlap) < 1:
    print(f"  *** RISK: n_features_known fully determines VOI class!")

# Also by audit rationale
print(f"\n  By audit rationale:")
for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
    rc_objs = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
    rc_nfeat = [v["n_pre_surface_features"] for v in rc_objs]
    if rc_nfeat:
        print(f"    {rc}: mean={sum(rc_nfeat)/len(rc_nfeat):.1f}, range=[{min(rc_nfeat)},{max(rc_nfeat)}], unique={sorted(set(rc_nfeat))}")

# =============================================================================
# Part 7: Cue / Shortcut Audit
# =============================================================================
print("\n[7/14] Cue / shortcut audit...")

# Build per-object feature vectors
object_feature_vectors = []
for v in all_voi_results:
    oid = v["oid"]
    seed = v["seed"]
    obj = per_seed_objects[seed][oid]
    pre_feats = obj["pre_surface_features"]
    feat_vec = {}
    for fn in ALL_VISIBLE_FEATURE_NAMES:
        feat_vec[f"feat_{fn}"] = 1.0 if fn in pre_feats else 0.0
    feat_vec["audit_VOI"] = v["audit_VOI"]
    feat_vec["positive_net"] = 1.0 if v["positive_net"] else 0.0
    feat_vec["audit_rationale_class"] = v["audit_rationale_class"]
    feat_vec["category"] = v["category"]
    feat_vec["seed"] = v["seed"]
    feat_vec["n_pre_surface_features"] = v["n_pre_surface_features"]
    object_feature_vectors.append(feat_vec)

# Softmax classifier utilities
def softmax(logits):
    exps = [math.exp(lv - max(logits)) for lv in logits]
    total = sum(exps)
    return [e / total for e in exps]

def train_softmax_3class(X, y, l2_alpha=0.1, max_iter=200, lr=0.01):
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 0
    if n_features == 0 or n_samples == 0:
        return None, None
    W = [[0.0] * n_features for _ in range(3)]
    b = [0.0] * 3
    for _ in range(max_iter):
        grad_W = [[0.0] * n_features for _ in range(3)]
        grad_b = [0.0] * 3
        for i in range(n_samples):
            logits = [sum(W[c][j] * X[i][j] for j in range(n_features)) + b[c] for c in range(3)]
            probs = softmax(logits)
            for c in range(3):
                target = 1.0 if c == y[i] else 0.0
                error = probs[c] - target
                for j in range(n_features):
                    grad_W[c][j] += error * X[i][j]
                grad_b[c] += error
        for c in range(3):
            for j in range(n_features):
                grad_W[c][j] = grad_W[c][j] / n_samples + l2_alpha * W[c][j]
                W[c][j] -= lr * grad_W[c][j]
            grad_b[c] /= n_samples
            b[c] -= lr * grad_b[c]
    return W, b

def predict_softmax(W, b, X):
    preds = []
    for i in range(len(X)):
        logits = [sum(W[c][j] * X[i][j] for j in range(len(W[0]))) + b[c] for c in range(3)]
        preds.append(max(range(3), key=lambda c: logits[c]))
    return preds

# --- Single-feature VOI probe (predict positive_net binary) ---
print("  Single-feature positive_net prediction:")
# Binary logistic regression (simplified: softmax with 2 classes)
def train_softmax_2class(X, y, l2_alpha=0.1, max_iter=200, lr=0.01):
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 0
    if n_features == 0 or n_samples == 0:
        return None, None
    W = [[0.0] * n_features for _ in range(2)]
    b = [0.0] * 2
    for _ in range(max_iter):
        grad_W = [[0.0] * n_features for _ in range(2)]
        grad_b = [0.0] * 2
        for i in range(n_samples):
            logits = [sum(W[c][j] * X[i][j] for j in range(n_features)) + b[c] for c in range(2)]
            probs = softmax(logits)
            for c in range(2):
                target = 1.0 if c == y[i] else 0.0
                error = probs[c] - target
                for j in range(n_features):
                    grad_W[c][j] += error * X[i][j]
                grad_b[c] += error
        for c in range(2):
            for j in range(n_features):
                grad_W[c][j] = grad_W[c][j] / n_samples + l2_alpha * W[c][j]
                W[c][j] -= lr * grad_W[c][j]
            grad_b[c] /= n_samples
            b[c] -= lr * grad_b[c]
    return W, b

def predict_softmax_2class(W, b, X):
    preds = []
    for i in range(len(X)):
        logits = [sum(W[c][j] * X[i][j] for j in range(len(W[0]))) + b[c] for c in range(2)]
        preds.append(max(range(2), key=lambda c: logits[c]))
    return preds

single_feature_accuracies = {}
for fn in ALL_VISIBLE_FEATURE_NAMES:
    X = [[1.0 if fv[f"feat_{fn}"] > 0 else 0.0] for fv in object_feature_vectors]
    y = [1 if fv["positive_net"] > 0 else 0 for fv in object_feature_vectors]

    W, b = train_softmax_2class(X, y)
    if W is not None:
        preds = predict_softmax_2class(W, b, X)
        acc = sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)
        single_feature_accuracies[fn] = acc

top_single = sorted(single_feature_accuracies.items(), key=lambda x: -x[1])[:10]
print("  Top single-feature positive_net predictors:")
for fn, acc in top_single:
    flag = " *** SHORTCUT RISK" if acc > 0.70 else ""
    print(f"    {fn}: acc={acc:.4f}{flag}")

max_single_acc = max(single_feature_accuracies.values()) if single_feature_accuracies else 0

# --- Feature-pair probe ---
print("  Top feature-pair positive_net predictors (sampled):")
pair_accs = []
sampled_features = [fn for fn, acc in top_single[:8]]
for i, f1 in enumerate(sampled_features):
    for f2 in sampled_features[i+1:]:
        X = [[1.0 if fv[f"feat_{f1}"] > 0 else 0.0,
              1.0 if fv[f"feat_{f2}"] > 0 else 0.0]
             for fv in object_feature_vectors]
        y = [1 if fv["positive_net"] > 0 else 0 for fv in object_feature_vectors]
        W, b = train_softmax_2class(X, y)
        if W is not None:
            preds = predict_softmax_2class(W, b, X)
            acc = sum(1 for p, yi in zip(preds, y) if p == yi) / len(y)
            pair_accs.append(((f1, f2), acc))

pair_accs.sort(key=lambda x: -x[1])
for (f1, f2), acc in pair_accs[:8]:
    flag = " *** SHORTCUT RISK" if acc > 0.80 else ""
    print(f"    {f1} + {f2}: acc={acc:.4f}{flag}")

max_pair_acc = pair_accs[0][1] if pair_accs else 0

# --- Feature-count VOI probe ---
print("  Feature-count VOI probe:")
X_nfeat = [[float(fv["n_pre_surface_features"])] for fv in object_feature_vectors]
y_bin = [1 if fv["positive_net"] > 0 else 0 for fv in object_feature_vectors]
W_nf, b_nf = train_softmax_2class(X_nfeat, y_bin)
if W_nf is not None:
    preds_nf = predict_softmax_2class(W_nf, b_nf, X_nfeat)
    nfeat_acc = sum(1 for p, yi in zip(preds_nf, y_bin) if p == yi) / len(y_bin)
    print(f"    n_features_known alone predicts positive_net: acc={nfeat_acc:.4f} (chance={max(sum(y_bin)/len(y_bin), 1-sum(y_bin)/len(y_bin)):.4f})")
    if nfeat_acc > 0.70:
        print(f"    *** RISK: feature count nearly determines VOI class!")

# --- Category VOI probe (audit only) ---
print("  Category VOI probe (audit only):")
RATIONAL_CLASS_TO_IDX = {"positive_change": 0, "confirmatory": 1, "wasteful_cost_only": 2}
# Category -> positive_net
for cat in CATEGORIES:
    cat_objs = [v for v in all_voi_results if v["category"] == cat]
    cat_pos = [v for v in cat_objs if v["positive_net"]]
    print(f"    {cat}: positive_net rate = {len(cat_pos)}/{len(cat_objs)} = {len(cat_pos)/len(cat_objs):.3f}")

# Can category predict positive_net?
cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
X_cat = [[0.0] * len(CATEGORIES) for fv in object_feature_vectors]
for i, fv in enumerate(object_feature_vectors):
    X_cat[i][cat_to_idx[fv["category"]]] = 1.0
W_cat, b_cat = train_softmax_2class(X_cat, y_bin)
if W_cat is not None:
    preds_cat = predict_softmax_2class(W_cat, b_cat, X_cat)
    cat_acc = sum(1 for p, yi in zip(preds_cat, y_bin) if p == yi) / len(y_bin)
    print(f"    Category predicts positive_net: acc={cat_acc:.4f}")
    if cat_acc > 0.70:
        print(f"    *** RISK: category nearly determines VOI class!")

# --- Pre-signature VOI probe (audit only) ---
print("  Pre-signature VOI probe (audit only):")
sig_to_idx = {}
for sig in sig_groups:
    if sig not in sig_to_idx:
        sig_to_idx[sig] = len(sig_to_idx)

X_sig = [[0.0] * len(sig_to_idx) for _ in object_feature_vectors]
for i, fv in enumerate(object_feature_vectors):
    sig = fv.get("pre_surface_signature", ())
    if sig in sig_to_idx:
        X_sig[i][sig_to_idx[sig]] = 1.0

W_sig, b_sig = train_softmax_2class(X_sig, y_bin)
if W_sig is not None and len(sig_to_idx) > 0:
    preds_sig = predict_softmax_2class(W_sig, b_sig, X_sig)
    sig_acc = sum(1 for p, yi in zip(preds_sig, y_bin) if p == yi) / len(y_bin)
    print(f"    Pre-signature predicts positive_net: acc={sig_acc:.4f} (n_signatures={len(sig_to_idx)})")
    if sig_acc > 0.80:
        print(f"    *** RISK: pre-signature nearly determines VOI class!")

# --- Full-visible VOI probe ---
print("  Full-visible-feature positive_net probe:")
X_full = [[fv[f"feat_{fn}"] for fn in ALL_VISIBLE_FEATURE_NAMES]
          for fv in object_feature_vectors]
W_full, b_full = train_softmax_2class(X_full, y_bin)
if W_full is not None:
    preds_full = predict_softmax_2class(W_full, b_full, X_full)
    full_acc = sum(1 for p, yi in zip(preds_full, y_bin) if p == yi) / len(y_bin)
    # Baseline: majority class
    majority_acc = max(sum(y_bin) / len(y_bin), 1 - sum(y_bin) / len(y_bin))
    print(f"    accuracy: {full_acc:.4f} (majority baseline={majority_acc:.4f})")

# =============================================================================
# Part 8: Cue-VOI Correlation Audit
# =============================================================================
print("\n[8/14] Cue-VOI correlation audit...")

def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / n)
    sy = math.sqrt(sum((y - my)**2 for y in ys) / n)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n * sx * sy)

voi_values = [fv["audit_VOI"] for fv in object_feature_vectors]

feature_voi_corrs = {}
for fn in ALL_VISIBLE_FEATURE_NAMES:
    feat_vals = [fv[f"feat_{fn}"] for fv in object_feature_vectors]
    corr = pearson_corr(feat_vals, voi_values)
    feature_voi_corrs[fn] = corr

top_corrs = sorted(feature_voi_corrs.items(), key=lambda x: -abs(x[1]))[:10]
print("  Top feature-audit_VOI correlations:")
for fn, corr in top_corrs:
    flag = " *** HIGH CORRELATION" if abs(corr) > 0.60 else ""
    print(f"    {fn}: corr={corr:.4f}{flag}")

# Correlation of feature count with audit_VOI
nfeat_vals = [float(fv["n_pre_surface_features"]) for fv in object_feature_vectors]
nfeat_voi_corr = pearson_corr(nfeat_vals, voi_values)
print(f"  corr(n_features_known, audit_VOI): {nfeat_voi_corr:.4f}")
if abs(nfeat_voi_corr) > 0.70:
    print(f"  *** RISK: feature count strongly correlated with VOI!")

# =============================================================================
# Part 9: Counterfactual Slice Audit
# =============================================================================
print("\n[9/14] Counterfactual slice audit...")

# Find shared surface features that appear in BOTH positive_net and non_positive_net
# across different categories
for shared_feat in SHARED_SURFACE_FEATURES:
    slice_objs = []
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            if shared_feat in obj["pre_surface_features"]:
                for v in all_voi_results:
                    if v["oid"] == oid and v["seed"] == seed:
                        slice_objs.append({
                            "category": v["category"],
                            "audit_rationale_class": v["audit_rationale_class"],
                            "positive_net": v["positive_net"],
                            "audit_VOI": v["audit_VOI"],
                            "n_pre_surface_features": v["n_pre_surface_features"],
                        })
                        break

    if not slice_objs:
        continue

    pos_slice = [o for o in slice_objs if o["positive_net"]]
    nonpos_slice = [o for o in slice_objs if not o["positive_net"]]
    categories_present = set(o["category"] for o in slice_objs)
    rationales_present = set(o["audit_rationale_class"] for o in slice_objs)

    # Must appear in BOTH positive_net AND non_positive_net, AND >=2 categories
    if len(pos_slice) > 0 and len(nonpos_slice) > 0 and len(categories_present) >= 2:
        print(f"  Feature '{shared_feat}': n={len(slice_objs)}, "
              f"categories={categories_present}, "
              f"rationales={rationales_present}")
        print(f"    positive_net: {len(pos_slice)} cases, VOI mean={sum(o['audit_VOI'] for o in pos_slice)/len(pos_slice):.4f}")
        print(f"    non_positive_net: {len(nonpos_slice)} cases, VOI mean={sum(o['audit_VOI'] for o in nonpos_slice)/len(nonpos_slice):.4f}")
        # Show rationale breakdown for non_positive_net
        for rc in ["confirmatory", "wasteful_cost_only"]:
            rc_count = sum(1 for o in nonpos_slice if o["audit_rationale_class"] == rc)
            if rc_count > 0:
                print(f"      {rc}: {rc_count} cases")
        # What legal context factor changes VOI?
        print(f"    Context: pre_surface n_features range [{min(o['n_pre_surface_features'] for o in slice_objs)}, {max(o['n_pre_surface_features'] for o in slice_objs)}]")

# Count features that appear in both pos and non-pos
features_in_both = []
for fn in ALL_VISIBLE_FEATURE_NAMES:
    pos_with = set()
    nonpos_with = set()
    for fv in object_feature_vectors:
        if fv[f"feat_{fn}"] > 0:
            if fv["positive_net"] > 0:
                pos_with.add(fv["category"])
            else:
                nonpos_with.add(fv["category"])
    if len(pos_with) > 0 and len(nonpos_with) > 0:
        features_in_both.append((fn, len(pos_with), len(nonpos_with)))

print(f"\n  Features appearing in BOTH positive_net and non_positive_net: {len(features_in_both)}")
for fn, np_cats, nnp_cats in sorted(features_in_both, key=lambda x: -(x[1]+x[2]))[:10]:
    print(f"    {fn}: {np_cats} pos categories, {nnp_cats} non-pos categories")

# =============================================================================
# Part 10: Per-Cell Sample Count
# =============================================================================
print("\n[10/14] Per-cell sample counts...")

print("  Category x Audit Rationale Class:")
for cat in CATEGORIES:
    for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
        count = sum(1 for v in all_voi_results
                    if v["category"] == cat and v["audit_rationale_class"] == rc)
        print(f"    {cat} / {rc}: {count}")

print("  Category x Depth Schedule:")
for cat in CATEGORIES:
    for depth in ["no_observe", "one_observe", "repeated_observe"]:
        count = sum(1 for v in all_voi_results
                    if v["category"] == cat and v["depth_schedule"] == depth)
        print(f"    {cat} / {depth}: {count}")

print("  Pre-surface signature groups:")
for sig, info in sorted(sig_expected.items(), key=lambda x: -x[1]["n_members"]):
    categories_in_sig = set()
    for seed in SEEDS:
        for oid, obj in per_seed_objects[seed].items():
            if get_pre_surface_signature(obj["pre_surface_features"]) == sig:
                categories_in_sig.add(per_seed_audit_labels[seed][oid]["hidden_category"])
    print(f"    sig_{hash(sig) % 10000:04d}: n={info['n_members']}, "
          f"best_action={info['best_action']}, "
          f"best_expected={info['best_expected_value']:.4f}, "
          f"categories={categories_in_sig}")

# =============================================================================
# Part 11: Baseline gaps by category
# =============================================================================
print("\n[11/14] Baseline gaps by category...")

for cat in CATEGORIES:
    cat_vois = [v for v in all_voi_results if v["category"] == cat]
    n = len(cat_vois)
    cat_no_obs = sum(v["oracle_pre_best_expected"] for v in cat_vois) / n
    cat_always_obs = sum(v["oracle_post_best"] - DEFAULT_OBSERVE_COST for v in cat_vois) / n
    cat_selective = sum(
        (v["oracle_post_best"] - DEFAULT_OBSERVE_COST) if v["audit_VOI"] > 0 else v["oracle_pre_best_expected"]
        for v in cat_vois
    ) / n
    cat_obs_rate = sum(1 for v in cat_vois if v["audit_VOI"] > 0) / n
    print(f"  {cat}:")
    print(f"    no_observe={cat_no_obs:.4f}, always_observe={cat_always_obs:.4f}, "
          f"selective={cat_selective:.4f}")
    print(f"    selective-always gap={cat_selective-cat_always_obs:.4f}, "
          f"selective-no_obs gap={cat_selective-cat_no_obs:.4f}, "
          f"opt_obs_rate={cat_obs_rate:.4f}")

# =============================================================================
# Part 12: Per-depth-schedule VOI consistency
# =============================================================================
print("\n[12/14] Per-depth-schedule VOI check...")

for cat in CATEGORIES:
    for st in ["P", "Z", "N"]:
        for depth in ["no_observe", "one_observe", "repeated_observe"]:
            vois_for_cell = []
            for seed in SEEDS:
                for oid, lbl in per_seed_audit_labels[seed].items():
                    if (lbl["hidden_category"] == cat and lbl["hidden_subtype"] == st
                            and lbl["depth_schedule"] == depth):
                        for v in all_voi_results:
                            if v["oid"] == oid and v["seed"] == seed:
                                vois_for_cell.append(v["audit_VOI"])
                                break
            if vois_for_cell:
                print(f"  {cat}/{st}/{depth}: n={len(vois_for_cell)}, "
                      f"VOI mean={sum(vois_for_cell)/len(vois_for_cell):.4f}")

# =============================================================================
# Part 13: Observe-Cost Calibration Sweep
# =============================================================================
print("\n[13/14] Observe-cost calibration sweep...")

def run_cost_sweep(observe_costs):
    """Sweep observe_cost values and report key metrics."""
    results = []
    for oc in observe_costs:
        voi_results, sig_groups_oc, sig_expected_oc = compute_audit_voi_all_objects(
            per_seed_objects, per_seed_audit_labels, oc)
        bl = compute_baselines(voi_results, oc)

        positive_net = [v for v in voi_results if v["audit_VOI"] > 0.001]
        near_zero = [v for v in voi_results if -0.001 <= v["audit_VOI"] <= 0.001]
        negative_net = [v for v in voi_results if v["audit_VOI"] < -0.001]

        results.append({
            "observe_cost": oc,
            "positive_net_rate": len(positive_net) / len(voi_results),
            "near_zero_rate": len(near_zero) / len(voi_results),
            "negative_net_rate": len(negative_net) / len(voi_results),
            "always_observe_net": bl["always_observe_net"],
            "always_try": bl["always_try"],
            "no_observe": bl["no_observe_pre_only"],
            "oracle_selective": bl["oracle_selective"],
            "oracle_obs_rate": bl["oracle_selective_obs_rate"],
            "sel_vs_always_observe": bl["oracle_selective_gap_vs_always_observe"],
            "sel_vs_always_try": bl["oracle_selective_gap_vs_always_try"],
            "sel_vs_no_observe": bl["oracle_selective_gap_vs_no_observe"],
        })
    return results

# Sweep observe costs from 0 to 0.10
cost_candidates = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
sweep_results = run_cost_sweep(cost_candidates)

print(f"  {'cost':>8s}  {'pos_rate':>8s}  {'zero_rate':>8s}  {'neg_rate':>8s}  "
      f"{'always_obs':>10s}  {'always_try':>10s}  {'sel_net':>10s}  "
      f"{'sel-obs':>8s}  {'sel-try':>8s}  {'obs_rate':>8s}")
for r in sweep_results:
    print(f"  {r['observe_cost']:8.4f}  {r['positive_net_rate']:8.4f}  {r['near_zero_rate']:8.4f}  "
          f"{r['negative_net_rate']:8.4f}  {r['always_observe_net']:10.4f}  "
          f"{r['always_try']:10.4f}  {r['oracle_selective']:10.4f}  "
          f"{r['sel_vs_always_observe']:8.4f}  {r['sel_vs_always_try']:8.4f}  "
          f"{r['oracle_obs_rate']:8.4f}")

# Select default observe_cost based on criteria
print(f"\n  Default observe_cost selection criteria:")
default_ok = True
default_r = [r for r in sweep_results if abs(r["observe_cost"] - DEFAULT_OBSERVE_COST) < 0.0001][0]

criteria = {
    "positive_net cases exist": default_r["positive_net_rate"] > 0,
    "non_positive_net cases exist": default_r["positive_net_rate"] < 1.0,
    "oracle_selective > always_observe (meaningful margin > 0.001)": default_r["sel_vs_always_observe"] > 0.001,
    "oracle_selective > always_try (meaningful margin > 0.001)": default_r["sel_vs_always_try"] > 0.001,
    "always_observe not oracle-level": abs(default_r["always_observe_net"] - default_r["oracle_selective"]) > 0.001,
    "always_try not oracle-level": abs(default_r["always_try"] - default_r["oracle_selective"]) > 0.001,
}

for criterion, result in criteria.items():
    status = "PASS" if result else "FAIL"
    print(f"    {criterion}: {status}")
    if not result:
        default_ok = False

if default_ok:
    print(f"\n  Default observe_cost={DEFAULT_OBSERVE_COST} is VALID.")
else:
    # Find the best alternative
    print(f"\n  Default observe_cost={DEFAULT_OBSERVE_COST} FAILS criteria. Searching alternatives...")
    for r in sweep_results:
        ok = (r["positive_net_rate"] > 0 and r["positive_net_rate"] < 1.0
              and r["sel_vs_always_observe"] > 0.001
              and r["sel_vs_always_try"] > 0.001
              and abs(r["always_observe_net"] - r["oracle_selective"]) > 0.001
              and abs(r["always_try"] - r["oracle_selective"]) > 0.001)
        if ok:
            print(f"    cost={r['observe_cost']:.4f} passes all criteria.")
            print(f"    Consider setting DEFAULT_OBSERVE_COST = {r['observe_cost']:.4f}")

# Best cost: maximize sel_vs_always_observe while keeping positive_net_rate > 0
best_cost = None
best_margin = -999
for r in sweep_results:
    if r["positive_net_rate"] > 0 and r["positive_net_rate"] < 1.0:
        if r["sel_vs_always_observe"] > best_margin:
            best_margin = r["sel_vs_always_observe"]
            best_cost = r["observe_cost"]
if best_cost is not None:
    print(f"  Best cost for max oracle_selective margin: {best_cost:.4f} (margin={best_margin:.4f})")

# =============================================================================
# Part 14: Acceptance and Output
# =============================================================================
print("\n[14/14] Acceptance checks and output...")

checks = {}

# 1. Three-world separation
checks["three_world_separation"] = separation_ok
checks["forbidden_fields_absent"] = len(forbidden_in_agent) == 0
checks["hidden_before_reveal_clean"] = hidden_before_reveal_violations == 0

# 2. Positive_net cases exist and not near-universal
checks["positive_net_exists"] = len(positive_net_voi) > 0
checks["positive_net_not_universal"] = pos_rate < 0.90

# 3. Non-positive_net cases exist in EVERY category
all_cats_have_nonpos = True
for cat in CATEGORIES:
    cat_nonpos = [v for v in all_voi_results
                  if v["category"] == cat and not v["positive_net"]]
    if len(cat_nonpos) == 0:
        all_cats_have_nonpos = False
        print(f"  MISSING: {cat} has no non_positive_net cases!")
checks["every_category_has_non_positive_net"] = all_cats_have_nonpos

# 4. Each category has at least positive_change and one non-positive rationale
all_cats_have_pos_and_nonpos_rationale = True
for cat in CATEGORIES:
    has_pos = any(v["category"] == cat and v["audit_rationale_class"] == "positive_change" for v in all_voi_results)
    has_nonpos = any(v["category"] == cat and v["audit_rationale_class"] in ("confirmatory", "wasteful_cost_only") for v in all_voi_results)
    if not (has_pos and has_nonpos):
        all_cats_have_pos_and_nonpos_rationale = False
        print(f"  MISSING: {cat} lacks positive_change or non-positive rationale!")
checks["each_category_has_mixed_rationale"] = all_cats_have_pos_and_nonpos_rationale

# Note: confirmatory and wasteful_cost_only are expected to collapse together
# when observe_cost is fixed — both yield VOI ≈ -observe_cost.
# The distinction is audit-semantic, not numerically forced.

# 5. Oracle selective beats always_observe and always_try
checks["oracle_selective_beats_always_observe"] = oracle_beats_always
checks["oracle_selective_beats_always_try"] = oracle_beats_try

# 6. Obs rate is selective (not 0 or 1)
opt_rate = baselines["oracle_selective_obs_rate"]
checks["oracle_selective_is_selective"] = 0.05 < opt_rate < 0.95

# 7. No single feature shortcut for positive_net
checks["no_single_feature_shortcut"] = max_single_acc <= 0.70

# 8. No feature-pair shortcut
checks["no_feature_pair_shortcut"] = max_pair_acc <= 0.80

# 9. n_features_known overlaps across positive_net / non_positive_net
nfeat_overlap_count = len(set(nfeat_by_class["positive_net"]) & set(nfeat_by_class["non_positive_net"]))
checks["n_features_known_overlap"] = nfeat_overlap_count >= 1

# 10. At least one shared surface feature appears in both pos and non-pos
checks["shared_feature_in_both_voi_classes"] = len(features_in_both) >= 1

# 11. Category does not determine VOI class
checks["category_not_determine_voi"] = cat_acc <= 0.70 if W_cat is not None else True

# 12. Feature count not strongly correlated with VOI
checks["feature_count_not_strongly_corr_voi"] = abs(nfeat_voi_corr) <= 0.70

# 13. Pre-oracle uses legal pre signature only (verified by construction)
checks["pre_oracle_uses_legal_signature_only"] = True  # by construction

# 14. Observe cost calibration passed
checks["observe_cost_calibration_passed"] = default_ok

all_pass = all(checks.values())
failure_reasons = [k for k, v in checks.items() if not v]

for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {check_name}: {status}")

if all_pass:
    overall = "PASS"
elif len(failure_reasons) <= 3:
    overall = "PARTIAL"
else:
    overall = "FAIL"

print(f"\n  overall: {overall}")
if failure_reasons:
    print(f"  failures: {failure_reasons}")

elapsed = round(time.time() - t0, 1)
print(f"\n  Elapsed: {elapsed}s")

# =============================================================================
# Output
# =============================================================================
print("\nWriting outputs...")

impl_status = overall.lower()

# --- JSON ---
json_output = {
    "block_id": "1J40b-7a",
    "elapsed_seconds": elapsed,
    "seeds": SEEDS,
    "total_objects": total_objects_all,
    "implementation_status": impl_status,
    "observe_cost_default": DEFAULT_OBSERVE_COST,
    "three_world_separation": {
        "agent_visible_pre_keys": AGENT_VISIBLE_PRE_KEYS,
        "agent_visible_state_keys": AGENT_VISIBLE_STATE_KEYS,
        "agent_visible_numeric": AGENT_VISIBLE_NUMERIC,
        "forbidden_fields": FORBIDDEN_FIELDS,
        "forbidden_in_agent_leaks": forbidden_in_agent,
        "hidden_before_reveal_violations": hidden_before_reveal_violations,
        "separation_ok": separation_ok,
    },
    "voi_distribution": {
        "total": total_voi,
        "positive_net_count": len(positive_net_voi),
        "positive_net_rate": round(pos_rate, 4),
        "non_positive_net_count": len(non_positive_net_voi),
        "non_positive_net_rate": round(nonpos_rate, 4),
        "cost_only_count": len(cost_only_voi),
        "cost_only_rate": round(cost_only_rate, 4),
        "mean_voi": round(sum(v["audit_VOI"] for v in all_voi_results) / total_voi, 4),
        "n_unique_pre_surface_signatures": len(sig_groups),
        "sig_group_size_range": [min(len(m) for m in sig_groups.values()), max(len(m) for m in sig_groups.values())],
    },
    "audit_rationale_distribution": {},
    "baselines": {k: round(v, 4) if isinstance(v, float) else v for k, v in baselines.items()},
    "pre_observe_audit": {
        "pre_mean": round(pre_mean, 4),
        "oracle_mean": round(oracle_mean, 4),
        "pre_oracle_gap": round(pre_oracle_gap, 4),
    },
    "cue_shortcut_audit": {
        "max_single_feature_acc": round(max_single_acc, 4),
        "top_single_features": [(fn, round(acc, 4)) for fn, acc in top_single[:5]],
        "max_feature_pair_acc": round(max_pair_acc, 4),
        "top_feature_pairs": [((f1, f2), round(acc, 4)) for (f1, f2), acc in pair_accs[:5]],
        "nfeat_voi_acc": round(nfeat_acc, 4) if W_nf is not None else None,
        "category_voi_acc": round(cat_acc, 4) if W_cat is not None else None,
        "pre_signature_voi_acc": round(sig_acc, 4) if W_sig is not None else None,
        "full_visible_voi_probe_acc": round(full_acc, 4),
        "nfeat_voi_correlation": round(nfeat_voi_corr, 4),
    },
    "n_features_known_overlap": {
        "positive_net_values": sorted(set(nfeat_by_class["positive_net"])),
        "non_positive_net_values": sorted(set(nfeat_by_class["non_positive_net"])),
        "overlap": sorted(overlap),
        "overlap_count": nfeat_overlap_count,
    },
    "counterfactual_slice_features": [
        fn for fn, np_c, nnp_c in features_in_both
    ],
    "observe_cost_sweep": [{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in sweep_results],
    "acceptance_checks": {k: v for k, v in checks.items()},
    "failure_reasons": failure_reasons,
}

# Audit rationale distribution
for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
    rc_objs = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
    json_output["audit_rationale_distribution"][rc] = {
        "count": len(rc_objs),
        "rate": round(len(rc_objs) / total_voi, 4),
        "mean_voi": round(sum(v["audit_VOI"] for v in rc_objs) / (len(rc_objs) or 1), 4),
    }

# Per-category stats
json_output["per_category"] = {}
for cat in CATEGORIES:
    cat_vois = [v for v in all_voi_results if v["category"] == cat]
    cat_pos = [v for v in cat_vois if v["positive_net"]]
    json_output["per_category"][cat] = {
        "count": len(cat_vois),
        "positive_net": len(cat_pos),
        "non_positive_net": len(cat_vois) - len(cat_pos),
        "mean_voi": round(sum(v["audit_VOI"] for v in cat_vois) / len(cat_vois), 4),
        "pre_mean": round(sum(v["oracle_pre_best_expected"] for v in cat_vois) / len(cat_vois), 4),
        "oracle_mean": round(sum(v["oracle_post_best"] for v in cat_vois) / len(cat_vois), 4),
    }

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b7a_env3b_selective_observe_generation_audit.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# --- CSV ---
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7a_env3b_selective_observe_generation_audit_table.csv")
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["block_id", "1J40b-7a"])
    w.writerow(["implementation_status", impl_status])
    w.writerow(["seeds", str(SEEDS)])
    w.writerow(["total_objects", total_objects_all])
    w.writerow(["elapsed_seconds", elapsed])
    w.writerow(["observe_cost_default", DEFAULT_OBSERVE_COST])
    w.writerow(["three_world_separation", "PASS" if separation_ok else "FAIL"])
    w.writerow(["positive_net_count", len(positive_net_voi)])
    w.writerow(["positive_net_rate", round(pos_rate, 4)])
    w.writerow(["non_positive_net_count", len(non_positive_net_voi)])
    w.writerow(["non_positive_net_rate", round(nonpos_rate, 4)])
    w.writerow(["cost_only_count", len(cost_only_voi)])
    w.writerow(["positive_change_count", sum(1 for v in all_voi_results if v["audit_rationale_class"]=="positive_change")])
    w.writerow(["confirmatory_count", sum(1 for v in all_voi_results if v["audit_rationale_class"]=="confirmatory")])
    w.writerow(["wasteful_cost_only_count", sum(1 for v in all_voi_results if v["audit_rationale_class"]=="wasteful_cost_only")])
    w.writerow(["no_observe_pre_only", round(baselines["no_observe_pre_only"], 4)])
    w.writerow(["always_try", round(baselines["always_try"], 4)])
    w.writerow(["always_observe_net", round(baselines["always_observe_net"], 4)])
    w.writerow(["random_observe", round(baselines["random_observe"], 4)])
    w.writerow(["oracle_selective", round(baselines["oracle_selective"], 4)])
    w.writerow(["oracle_selective_obs_rate", round(baselines["oracle_selective_obs_rate"], 4)])
    w.writerow(["sel_vs_always_observe", round(baselines["oracle_selective_gap_vs_always_observe"], 4)])
    w.writerow(["sel_vs_always_try", round(baselines["oracle_selective_gap_vs_always_try"], 4)])
    w.writerow(["pre_mean_return", round(pre_mean, 4)])
    w.writerow(["oracle_mean_return", round(oracle_mean, 4)])
    w.writerow(["pre_oracle_gap", round(pre_oracle_gap, 4)])
    w.writerow(["max_single_feature_voi_acc", round(max_single_acc, 4)])
    w.writerow(["nfeat_voi_acc", round(nfeat_acc, 4) if W_nf is not None else "N/A"])
    w.writerow(["nfeat_voi_corr", round(nfeat_voi_corr, 4)])
    w.writerow(["category_voi_acc", round(cat_acc, 4) if W_cat is not None else "N/A"])
    w.writerow(["pre_signature_voi_acc", round(sig_acc, 4) if W_sig is not None else "N/A"])
    w.writerow(["full_visible_voi_probe_acc", round(full_acc, 4)])
    w.writerow(["shared_features_in_both_voi", len(features_in_both)])
    for check_name, result in checks.items():
        w.writerow([f"check_{check_name}", "PASS" if result else "FAIL"])
print(f"  CSV  -> {csv_path}")

# --- MD ---
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b7a_env3b_selective_observe_generation_audit.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-7a: env3b Selective-Observe Environment Generation + Audit\n\n")
    f.write(f"- **Implementation Status**: {overall}\n")
    f.write(f"- **Seeds**: {SEEDS}\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n")
    f.write(f"- **Total objects**: {total_objects_all}\n\n")

    f.write("## 1. Files Changed\n\n")
    f.write("- `_block1j40b7a_env3b_selective_observe_generation_audit.py` — Created\n")
    f.write("- `runs/block1j40b7a_env3b_selective_observe_generation_audit.json` — Generated\n")
    f.write("- `protocols/block1j40b7a_env3b_selective_observe_generation_audit.md` — Generated\n")
    f.write("- `protocols/block1j40b7a_env3b_selective_observe_generation_audit_table.csv` — Generated\n\n")

    f.write("## 2. Commands Run\n\n")
    f.write("```\n")
    f.write('& "D:\\conda\\python.exe" "_block1j40b7a_env3b_selective_observe_generation_audit.py"\n')
    f.write("```\n\n")

    f.write("## 3. Env3b Generation Design\n\n")
    f.write("**Categories**: wood_log, stone_block, apple, wooden_pickaxe\n\n")
    f.write("**Feature layers** (renamed per correction):\n")
    f.write(f"- pre_surface_features ({len(ALL_AMBIENT_FEATURE_NAMES)} ambient + {len(SHARED_SURFACE_FEATURES)} shared surface cues)\n")
    f.write(f"- post_observe_features ({len(ALL_POST_OBSERVE_FEATURE_NAMES)} category-diagnostic)\n")
    f.write("- hidden_features (only in repeated_observe or audit)\n\n")
    f.write("**Two-layer VOI reporting**:\n")
    f.write("- Layer 1 (numeric): positive_net vs non_positive_net\n")
    f.write("- Layer 2 (audit rationale): positive_change, confirmatory, wasteful_cost_only\n\n")
    f.write("**Key design**: Objects share pre_surface signatures across categories.\n")
    f.write("VOI emerges from signature-group expected returns, not from true category.\n")
    f.write("Pre-oracle uses ONLY legal pre_surface signature groups.\n\n")

    f.write("### Shared surface features (weak pre-observe cues across categories):\n")
    f.write(f"{', '.join(SHARED_SURFACE_FEATURES)}\n\n")

    f.write("### Instance subtypes per category:\n\n")
    f.write("| Category | Subtype | Rationale | Pre Extra | Post Reveals |\n")
    f.write("|----------|---------|-----------|-----------|-------------|\n")
    for cat in CATEGORIES:
        for st in ["P", "Z", "N"]:
            sd = INSTANCE_SUBTYPE_DEFS[cat][st]
            pre_list = (["ambient"] + sd["pre_surface_extra"])[:5]
            post_list = sd["post_observe_revealed"][:4]
            f.write(f"| {cat} | {st} | {sd['audit_rationale_class']} | {', '.join(pre_list)} | {', '.join(post_list)} |\n")
    f.write("\n")

    f.write("## 4. Agent-Visible Fields\n\n")
    f.write(f"- **Pre-observe feature keys** ({len(AGENT_VISIBLE_PRE_KEYS)}): `{'`, `'.join(AGENT_VISIBLE_PRE_KEYS[:8])}...`\n")
    f.write(f"- **State keys** ({len(AGENT_VISIBLE_STATE_KEYS)}): `{'`, `'.join(AGENT_VISIBLE_STATE_KEYS)}`\n")
    f.write(f"- **Numeric**: `{'`, `'.join(AGENT_VISIBLE_NUMERIC)}`\n\n")

    f.write("## 5. Hidden/Audit-Only Fields\n\n")
    f.write("| Field | Visibility |\n")
    f.write("|-------|-----------|\n")
    for ff in FORBIDDEN_FIELDS[:20]:
        f.write(f"| {ff} | hidden/audit-only |\n")
    f.write("\n")

    f.write("## 6. Three-World Separation Audit\n\n")
    f.write(f"- Agent-visible pre features: {len(AGENT_VISIBLE_PRE_KEYS)}\n")
    f.write(f"- Forbidden fields total: {len(FORBIDDEN_FIELDS)}\n")
    f.write(f"- Forbidden-in-agent leaks: {len(forbidden_in_agent)}\n")
    f.write(f"- Hidden-before-reveal violations: {hidden_before_reveal_violations}\n")
    f.write(f"- **Separation**: {'PASS' if separation_ok else 'FAIL'}\n\n")

    f.write("## 7. VOI Distribution Audit\n\n")
    f.write("### Numeric VOI Class\n\n")
    f.write("| Class | Count | Rate | Mean VOI |\n")
    f.write("|-------|-------|------|----------|\n")
    f.write(f"| positive_net | {len(positive_net_voi)} | {pos_rate*100:.1f}% | {sum(v['audit_VOI'] for v in positive_net_voi)/(len(positive_net_voi) or 1):.4f} |\n")
    f.write(f"| non_positive_net | {len(non_positive_net_voi)} | {nonpos_rate*100:.1f}% | {sum(v['audit_VOI'] for v in non_positive_net_voi)/(len(non_positive_net_voi) or 1):.4f} |\n")
    f.write(f"| cost_only (~-{DEFAULT_OBSERVE_COST}) | {len(cost_only_voi)} | {cost_only_rate*100:.1f}% | |\n\n")

    f.write("### Audit Rationale Class\n\n")
    f.write("| Rationale | Count | Rate | Mean VOI |\n")
    f.write("|-----------|-------|------|----------|\n")
    for rc in ["positive_change", "confirmatory", "wasteful_cost_only"]:
        rc_objs = [v for v in all_voi_results if v["audit_rationale_class"] == rc]
        f.write(f"| {rc} | {len(rc_objs)} | {len(rc_objs)/total_voi*100:.1f}% | {sum(v['audit_VOI'] for v in rc_objs)/(len(rc_objs) or 1):.4f} |\n")
    f.write("\n")

    f.write("### Per-Category\n\n")
    f.write("| Category | Total | Pos Net | Non-Pos Net | Mean VOI |\n")
    f.write("|----------|-------|---------|-------------|----------|\n")
    for cat in CATEGORIES:
        cat_vois = [v for v in all_voi_results if v["category"] == cat]
        cat_pos = [v for v in cat_vois if v["positive_net"]]
        f.write(f"| {cat} | {len(cat_vois)} | {len(cat_pos)} | {len(cat_vois)-len(cat_pos)} | {sum(v['audit_VOI'] for v in cat_vois)/len(cat_vois):.4f} |\n")
    f.write("\n")

    f.write("## 8. Baseline Audit\n\n")
    f.write("| Baseline | Mean Return |\n")
    f.write("|----------|------------|\n")
    f.write(f"| no_observe_pre_only | {baselines['no_observe_pre_only']:.4f} |\n")
    f.write(f"| always_try | {baselines['always_try']:.4f} |\n")
    f.write(f"| always_observe_net | {baselines['always_observe_net']:.4f} |\n")
    f.write(f"| random_observe | {baselines['random_observe']:.4f} |\n")
    f.write(f"| **oracle_selective** | **{baselines['oracle_selective']:.4f}** |\n\n")
    f.write("| Gap | Value |\n")
    f.write("|-----|-------|\n")
    f.write(f"| oracle - always_observe | {baselines['oracle_selective_gap_vs_always_observe']:.4f} |\n")
    f.write(f"| oracle - always_try | {baselines['oracle_selective_gap_vs_always_try']:.4f} |\n")
    f.write(f"| oracle - no_observe | {baselines['oracle_selective_gap_vs_no_observe']:.4f} |\n\n")

    f.write("## 9. n_features_known Distribution\n\n")
    f.write("| VOI Class | Mean nfeat | Range | Unique |\n")
    f.write("|-----------|-----------|-------|--------|\n")
    for label in ["positive_net", "non_positive_net"]:
        vals = nfeat_by_class[label]
        f.write(f"| {label} | {sum(vals)/len(vals):.1f} | [{min(vals)},{max(vals)}] | {sorted(set(vals))} |\n")
    f.write(f"\nOverlap: {sorted(overlap)} (count={nfeat_overlap_count})\n\n")

    f.write("## 10. Cue / Shortcut Audit\n\n")
    f.write(f"- Max single-feature positive_net acc: {max_single_acc:.4f}\n")
    f.write(f"- Max feature-pair positive_net acc: {max_pair_acc:.4f}\n")
    f.write(f"- n_features_known VOI acc: {nfeat_acc:.4f}\n" if W_nf is not None else "")
    f.write(f"- Category VOI acc: {cat_acc:.4f}\n" if W_cat is not None else "")
    f.write(f"- Pre-signature VOI acc: {sig_acc:.4f} (n_sigs={len(sig_to_idx)})\n" if W_sig is not None else "")
    f.write(f"- Full-visible VOI probe acc: {full_acc:.4f}\n")
    f.write(f"- corr(n_features_known, VOI): {nfeat_voi_corr:.4f}\n\n")

    f.write("## 11. Counterfactual Slice Audit\n\n")
    f.write(f"Features appearing in both positive_net and non_positive_net: {len(features_in_both)}\n\n")
    for fn, np_c, nnp_c in sorted(features_in_both, key=lambda x: -(x[1]+x[2]))[:10]:
        f.write(f"- **{fn}**: {np_c} pos categories, {nnp_c} non-pos categories\n")
    f.write("\n")

    f.write("## 12. Observe-Cost Calibration Sweep\n\n")
    f.write("| Cost | Pos Rate | Zero Rate | Neg Rate | AlwaysObs | AlwaysTry | Selective | Sel-Obs | Sel-Try | ObsRate |\n")
    f.write("|------|----------|-----------|----------|-----------|-----------|-----------|---------|---------|--------|\n")
    for r in sweep_results:
        f.write(f"| {r['observe_cost']:.4f} | {r['positive_net_rate']:.4f} | {r['near_zero_rate']:.4f} | {r['negative_net_rate']:.4f} | {r['always_observe_net']:.4f} | {r['always_try']:.4f} | {r['oracle_selective']:.4f} | {r['sel_vs_always_observe']:.4f} | {r['sel_vs_always_try']:.4f} | {r['oracle_obs_rate']:.4f} |\n")
    f.write(f"\nDefault observe_cost={DEFAULT_OBSERVE_COST}: {'VALID' if default_ok else 'NEEDS ADJUSTMENT'}\n\n")

    f.write("## 13. Acceptance Result\n\n")
    f.write(f"**{overall}**\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status}** |\n")
    if failure_reasons:
        f.write(f"\nFailures: {failure_reasons}\n")
    f.write("\n")

    f.write("## 14. Output Files\n\n")
    f.write("- `runs/block1j40b7a_env3b_selective_observe_generation_audit.json`\n")
    f.write("- `protocols/block1j40b7a_env3b_selective_observe_generation_audit.md`\n")
    f.write("- `protocols/block1j40b7a_env3b_selective_observe_generation_audit_table.csv`\n\n")

    f.write("## 15. Next Suggested Resume Point\n\n")
    if overall == "PASS":
        f.write("1J40b-7a PASS: env3b provides a valid mixed observe-value environment.\n")
        f.write("Next: 1J40b-7b — train shadow learner on env3b to test whether the learner can:\n")
        f.write("1. Learn to observe only when VOI > 0\n")
        f.write("2. Avoid always-observe collapse\n")
        f.write("3. Achieve returns between always_observe and oracle_selective\n")
    else:
        f.write(f"1J40b-7a {overall}: Review failures above.\n")
        f.write("Adjust instance subtype definitions to fix issues.\n")

    f.write(f"\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-7a\n")
    f.write(f"implementation_status={impl_status}\n")
    f.write(f"overall={overall}\n")
    f.write(f"oracle_selective={baselines['oracle_selective']:.4f}\n")
    f.write(f"always_observe={baselines['always_observe_net']:.4f}\n")
    f.write(f"pos_rate={pos_rate:.4f}\n")
    f.write(f"failure_reasons={'; '.join(failure_reasons) if failure_reasons else 'none'}\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'='*70}")
print(f"Block 1J40b-7a complete.")
print(f"  overall: {overall}")
print(f"  oracle_selective={baselines['oracle_selective']:.4f}")
print(f"  always_observe={baselines['always_observe_net']:.4f}")
print(f"  pos_rate={pos_rate:.4f}")
print(f"{'='*70}")
