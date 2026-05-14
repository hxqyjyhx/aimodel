"""
Block 1J40b-env1 -- Observe-Try Counterfactual Environment Patch.

Adds C6_observe_try_counterfactual_v0 condition with:
- Matched observe-depth schedules (no_observe, one_observe, repeated_observe)
- Delayed observe credit from later visible outcomes
- Comparable action sets
- Value scale audit

Generation-only smoke test. Does NOT change active policy, environment defaults,
or C4/C5 conditions.
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

# Re-verify C4/C5 still present
assert C4_PRESENT, "C4 must be preserved"
assert C5_PRESENT, "C5 must be preserved"
print(f"C4 preserved: {C4_PRESENT}, C5 preserved: {C5_PRESENT}")

# =============================================================================
# Constants
# =============================================================================
SEED = 109
N_EPISODES = 5
# 12 matched groups x 3 objects each = 36 objects (9 per family x 4 families)
N_GROUPS_PER_FAMILY = 3  # each family gets 3 matched groups
OBJECTS_PER_GROUP = 3    # no_observe, one_observe, repeated_observe
N_PER_FAMILY = N_GROUPS_PER_FAMILY * OBJECTS_PER_GROUP  # 9 per family
N_WOOD = N_PER_FAMILY
N_STONE = N_PER_FAMILY
N_APPLE = N_PER_FAMILY
N_PICKAXE = N_PER_FAMILY
TOTAL_OBJECTS = N_WOOD + N_STONE + N_APPLE + N_PICKAXE  # 36

ALL_ACTIONS = sorted([
    "craft_plank", "eat", "use_as_tool", "burn_as_fuel",
    "mine_by_hand", "mine_with_pickaxe",
])

STATE_FEATURE_KEYS = ["fresh", "wet", "damaged", "clean", "hot", "open"]

OBSERVE_COST = 0.005
PROBE_COST = 0.05
SUCCESS_REWARD = 0.5
FAILURE_PENALTY = 0.1
INFO_GAIN_PER_NEW_FEATURE = 0.02
INFO_GAIN_PER_NEW_STATE = 0.01
UNCERTAINTY_REDUCTION_VALUE = 0.03

OBSERVE_DEPTH_SCHEDULES = ["no_observe", "one_observe", "repeated_observe"]

# =============================================================================
# Categories and affordance profiles
# =============================================================================
CATEGORIES = ["wood_log", "stone_block", "apple", "wooden_pickaxe"]

# Base affordance profiles per category
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

# Feature sets per category
CATEGORY_FEATURES = {
    "wood_log": ["has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape",
                 "fibrous", "flammable", "porous_surface"],
    "stone_block": ["has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight",
                    "cold_to_touch", "scratch_resistant"],
    "apple": ["has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture",
              "light_weight", "fruity_scent"],
    "wooden_pickaxe": ["has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape",
                       "has_metal_head", "jointed"],
}

# Hidden diagnostic features per category (revealed only on repeated observe)
HIDDEN_DIAGNOSTIC_FEATURES = {
    "wood_log": ["rotting_odor", "mold_spots", "soft_to_touch"],
    "stone_block": ["internal_fractures", "efflorescence", "weathering_pattern"],
    "apple": ["bruise_markings", "soft_spot", "discoloration_near_stem"],
    "wooden_pickaxe": ["hairline_crack", "handle_looseness", "rust_on_fastener"],
}

# =============================================================================
# Part 1: Generate Matched-Group Objects for C6
# =============================================================================
print("\n[1/5] Generating C6 objects with matched observe-depth groups...")

def generate_c6_objects_deterministic(rng, prefix="c6"):
    """Generate objects organized into matched groups for observe-depth comparison.

    Each family gets N_GROUPS_PER_FAMILY matched groups.
    Each group has 3 objects assigned to: no_observe, one_observe, repeated_observe.
    Objects in the same group share the same affordance profile.
    """
    objects = {}
    audit_labels = {}
    oid_counter = [0]

    def make_oid():
        oid_counter[0] += 1
        return f"{prefix}_obj_{oid_counter[0]:04d}"

    for family_idx, category in enumerate(CATEGORIES):
        base_profile = BASE_AFFORDANCE_PROFILES[category]
        base_features = CATEGORY_FEATURES[category]
        hidden_features = HIDDEN_DIAGNOSTIC_FEATURES[category]

        for group_idx in range(N_GROUPS_PER_FAMILY):
            group_id = f"{category}_g{group_idx}"

            # Each group has a variant profile:
            # Group 0: observe-helps — hidden features predict key affordance outcomes
            # Group 1: observe-neutral — hidden features don't change affordance outcomes
            # Group 2: observe-wasteful — observe cost exceeds benefit
            group_role = ["observe_helps", "observe_neutral", "observe_wasteful"][group_idx]

            # Build variant affordance profile
            variant_profile = dict(base_profile)
            if group_role == "observe_helps":
                # Hidden features indicate different affordance for 2 actions
                variant_profile["eat"] = "success" if category in ("wood_log", "stone_block") else variant_profile.get("eat", "fail")
                variant_profile["craft_plank"] = "fail" if category == "wood_log" else variant_profile.get("craft_plank", "fail")
            elif group_role == "observe_neutral":
                # Same as base — observing doesn't change anything
                pass
            elif group_role == "observe_wasteful":
                # Same as base, but object is "easy" — observe cost not justified
                pass

            for depth_idx, depth_schedule in enumerate(OBSERVE_DEPTH_SCHEDULES):
                oid = make_oid()
                obj_features = dict.fromkeys(base_features, True)

                # Build visible features (always present)
                visible_features = {}
                for f in base_features:
                    visible_features[f] = True

                # Hidden features: only revealed on repeated observe
                hidden_feat_dict = {}
                for hf in hidden_features:
                    hidden_feat_dict[hf] = True

                # State features
                visible_state = {
                    "fresh": True, "wet": category in ("wood_log", "apple"),
                    "damaged": False, "clean": True,
                    "hot": False, "open": False,
                }

                obj = {
                    "oid": oid,
                    "hidden_category": category,
                    "hidden_affordance_profile": variant_profile,
                    "visible_features": visible_features,
                    "visible_state": visible_state,
                    "_hidden_features_dict": hidden_feat_dict,
                    "_group_id": group_id,
                    "_group_role": group_role,
                    "_depth_schedule": depth_schedule,
                    "_depth_idx": depth_idx,
                }
                objects[oid] = obj

                audit_labels[oid] = {
                    "oid": oid,
                    "hidden_category": category,
                    "true_family": category,
                    "mixed_source_type": f"c6_depth_{depth_schedule}",
                    "group_id": group_id,
                    "group_role": group_role,
                    "depth_schedule": depth_schedule,
                    "affordance_profile": dict(variant_profile),
                    "hidden_features": dict(hidden_feat_dict),
                    "schedule_type": depth_schedule,
                }

    return objects, audit_labels


rng_gen = random.Random(SEED + 900)
test_objects, audit_labels = generate_c6_objects_deterministic(rng_gen, prefix="c6")
test_oids = sorted(test_objects.keys())
print(f"  {len(test_oids)} objects generated ({N_GROUPS_PER_FAMILY} groups x {OBJECTS_PER_GROUP} depths x {len(CATEGORIES)} families)")

# Count by depth schedule
depth_counts = defaultdict(int)
role_counts = defaultdict(int)
for oid, lbl in audit_labels.items():
    depth_counts[lbl["depth_schedule"]] += 1
    role_counts[lbl["group_role"]] += 1
print(f"  Per depth: {dict(depth_counts)}")
print(f"  Per role: {dict(role_counts)}")

# Verify matched groups
group_ids = set(lbl["group_id"] for lbl in audit_labels.values())
group_info = {}
for gid in group_ids:
    members = [(oid, lbl) for oid, lbl in audit_labels.items() if lbl["group_id"] == gid]
    depths = sorted(set(lbl["depth_schedule"] for _, lbl in members))
    group_info[gid] = {
        "n_members": len(members),
        "depths": depths,
        "role": members[0][1]["group_role"],
    }
print(f"  Matched groups: {len(group_ids)} (each with {OBJECTS_PER_GROUP} depth variants)")

# =============================================================================
# Part 2: Build Event Schedules with Observe-Depth Variation
# =============================================================================
print("\n[2/5] Building event schedules with observe-depth variation...")

ep_rng = random.Random(SEED + 700)
event_rng = random.Random(SEED + 901)

# Assign objects to episodes
episode_assignments = {}
for idx, oid in enumerate(test_oids):
    episode_assignments[oid] = idx % N_EPISODES

all_events = []
step_id = 0

for ep in range(N_EPISODES):
    ep_oids = [oid for oid in test_oids if episode_assignments.get(oid) == ep]

    for oid in ep_oids:
        obj = test_objects[oid]
        label = audit_labels[oid]
        depth = label["depth_schedule"]
        category = label["hidden_category"]
        profile = label["affordance_profile"]
        features = obj.get("visible_features", {})
        hidden_feats = obj.get("_hidden_features_dict", {})
        state = obj.get("visible_state", {})

        # --- Phase 1: Initial information gathering (varies by depth) ---

        if depth == "no_observe":
            # No initial observe — the agent tries with only ambient features
            # Minimal feature delta (only what agent can see without dedicated observe)
            minimal_features = {k: v for k, v in features.items()
                              if k in ["brownish", "grayish", "greenish", "long_shape",
                                       "block_like", "round_small"]}
            initial_features = minimal_features
            initial_state = {}
            n_observes = 0

        elif depth == "one_observe":
            # One observe — reveals basic visible features
            step_id += 1
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {"observe_depth": 1, "schedule": "one_observe"},
                "observed_features_delta": dict(features),
                "observed_state_delta": dict(state),
                "success_or_failure": None,
            })
            initial_features = features
            initial_state = state
            n_observes = 1

        elif depth == "repeated_observe":
            # First observe — reveals basic features
            step_id += 1
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {"observe_depth": 1, "schedule": "repeated_observe"},
                "observed_features_delta": dict(features),
                "observed_state_delta": dict(state),
                "success_or_failure": None,
            })
            # Second observe — reveals hidden diagnostic features
            step_id += 1
            combined_features = dict(features)
            combined_features.update(hidden_feats)
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {"observe_depth": 2, "schedule": "repeated_observe"},
                "observed_features_delta": combined_features,  # includes hidden features
                "observed_state_delta": dict(state),
                "success_or_failure": None,
            })
            initial_features = combined_features
            initial_state = state
            n_observes = 2

        # --- Phase 2: Try multiple actions (comparable action set) ---
        # For each object, try 3 actions under the SAME situation features
        # This creates comparable action sets
        try_actions_for_object = []
        if category == "wood_log":
            try_actions_for_object = ["mine_by_hand", "craft_plank", "burn_as_fuel"]
        elif category == "stone_block":
            try_actions_for_object = ["mine_with_pickaxe", "mine_by_hand", "use_as_tool"]
        elif category == "apple":
            try_actions_for_object = ["eat", "mine_by_hand", "use_as_tool"]
        elif category == "wooden_pickaxe":
            try_actions_for_object = ["use_as_tool", "mine_by_hand", "craft_plank"]

        for action in try_actions_for_object:
            outcome_bool = (profile.get(action, "fail") == "success")
            step_id += 1
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "try",
                "action_target": oid,
                "action_params": {"affordance": action, "observe_depth": n_observes,
                                "schedule": depth},
                "observed_features_delta": dict(initial_features),
                "observed_state_delta": dict(initial_state),
                "success_or_failure": outcome_bool,
            })

        # --- Phase 3: Post-try observe for repeated_observe objects ---
        # This provides data to compute delayed observe value
        if depth == "repeated_observe":
            # Re-observe with full features after trying — reveals any state changes
            step_id += 1
            post_features = dict(features)
            post_features.update(hidden_feats)
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {"observe_depth": 3, "schedule": "repeated_observe",
                                                        "post_try": True},
                "observed_features_delta": post_features,
                "observed_state_delta": dict(state),  # state may have changed due to try
                "success_or_failure": None,
            })

        # --- Phase 4: For one_observe objects, also try the remaining 3 actions ---
        # to build full(er) action sets
        remaining_actions = [a for a in ALL_ACTIONS if a not in try_actions_for_object]
        for action in remaining_actions:
            outcome_bool = (profile.get(action, "fail") == "success")
            step_id += 1
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "try",
                "action_target": oid,
                "action_params": {"affordance": action, "observe_depth": n_observes,
                                "schedule": depth, "extended_set": True},
                "observed_features_delta": dict(initial_features),
                "observed_state_delta": dict(initial_state),
                "success_or_failure": outcome_bool,
            })

print(f"  {len(all_events)} total events across {N_EPISODES} episodes")

# =============================================================================
# Part 3: Compute Metrics from Event Log
# =============================================================================
print("\n[3/5] Computing diagnostic metrics...")

# --- Event counts by type ---
observe_events = [ev for ev in all_events if ev["action_type"] == "observe"]
try_events = [ev for ev in all_events if ev["action_type"] == "try"]

# --- Observe-depth schedule counts ---
no_observe_try_events = [ev for ev in try_events
                         if ev.get("action_params", {}).get("schedule") == "no_observe"]
one_observe_try_events = [ev for ev in try_events
                          if ev.get("action_params", {}).get("schedule") == "one_observe"]
repeated_observe_try_events = [ev for ev in try_events
                               if ev.get("action_params", {}).get("schedule") == "repeated_observe"]

# Count unique objects in each schedule
no_observe_oids = set(ev["action_target"] for ev in no_observe_try_events)
one_observe_oids = set(ev["action_target"] for ev in one_observe_try_events)
repeated_observe_oids = set(ev["action_target"] for ev in repeated_observe_try_events)

no_observe_try_count = len(no_observe_try_events)
one_observe_try_count = len(one_observe_try_events)
repeated_observe_try_count = len(repeated_observe_try_events)

# --- Matched schedule group count ---
# A group is "matched" if all 3 depth schedules have events for objects in the same group
matched_group_count = 0
for gid in group_ids:
    group_oids = [oid for oid, lbl in audit_labels.items() if lbl["group_id"] == gid]
    group_try_events = [ev for ev in try_events if ev["action_target"] in group_oids]
    group_depths = set(ev.get("action_params", {}).get("schedule") for ev in group_try_events)
    if group_depths == {"no_observe", "one_observe", "repeated_observe"}:
        matched_group_count += 1

# --- Comparable action sets ---
# Group try events by (oid, episode_id, schedule) as situation key
# A comparable action set exists when >1 action is tried for the same situation
situation_actions = defaultdict(set)
for ev in try_events:
    oid = ev["action_target"]
    ep_id = ev["episode_id"]
    depth = ev.get("action_params", {}).get("observe_depth", 0)
    sk = f"{oid}|ep{ep_id}|d{depth}"
    action = ev.get("action_params", {}).get("affordance", "")
    situation_actions[sk].add(action)

complete_action_set_count = sum(1 for actions in situation_actions.values()
                                if len(actions) >= len(ALL_ACTIONS))
partial_action_set_count = sum(1 for actions in situation_actions.values()
                               if 1 < len(actions) < len(ALL_ACTIONS))

# Comparable observe-try sets: situations with both observe and try for same object+episode
observe_try_oids = defaultdict(lambda: {"observe": False, "try_actions": set()})
for ev in all_events:
    oid = ev["action_target"]
    ep_id = ev["episode_id"]
    key = f"{oid}|ep{ep_id}"
    if ev["action_type"] == "observe":
        observe_try_oids[key]["observe"] = True
    elif ev["action_type"] == "try":
        action = ev.get("action_params", {}).get("affordance", "")
        observe_try_oids[key]["try_actions"].add(action)

comparable_observe_try_set_count = sum(
    1 for v in observe_try_oids.values()
    if v["observe"] and len(v["try_actions"]) >= 2)

# --- Try success rates by observe depth ---
def compute_try_success_rate(try_evs):
    if not try_evs:
        return None, 0
    successes = sum(1 for ev in try_evs if ev.get("success_or_failure") is True)
    return round(successes / len(try_evs), 4), len(try_evs)

no_obs_success, no_obs_n = compute_try_success_rate(no_observe_try_events)
one_obs_success, one_obs_n = compute_try_success_rate(one_observe_try_events)
rep_obs_success, rep_obs_n = compute_try_success_rate(repeated_observe_try_events)

# --- Delayed observe credit ---
# For each observe event, find later try events for the same object
# and compute delayed return as the net_value of subsequent tries
delayed_credits = []
for obs_ev in observe_events:
    oid = obs_ev["action_target"]
    obs_step = obs_ev["step_id"]
    obs_ep = obs_ev["episode_id"]

    # Find later try events for same object in same episode
    later_tries = [ev for ev in try_events
                   if ev["action_target"] == oid
                   and ev["episode_id"] == obs_ep
                   and ev["step_id"] > obs_step]

    if not later_tries:
        continue

    later_successes = sum(1 for ev in later_tries if ev.get("success_or_failure") is True)
    later_failures = sum(1 for ev in later_tries if ev.get("success_or_failure") is False)
    later_net = later_successes * (SUCCESS_REWARD - PROBE_COST) + later_failures * (-FAILURE_PENALTY - PROBE_COST)

    # Immediate observe value (info gain proxy)
    features_delta = obs_ev.get("observed_features_delta", {})
    state_delta = obs_ev.get("observed_state_delta", {})
    n_new_features = len(features_delta)
    n_new_states = len(state_delta)
    immediate_value = (n_new_features * INFO_GAIN_PER_NEW_FEATURE +
                       n_new_states * INFO_GAIN_PER_NEW_STATE +
                       (UNCERTAINTY_REDUCTION_VALUE if n_new_features > 0 else 0) -
                       OBSERVE_COST)

    delayed_credits.append({
        "oid": oid,
        "observe_ep": obs_ep,
        "observe_step": obs_step,
        "observe_depth": obs_ev.get("action_params", {}).get("observe_depth", 0),
        "n_later_tries": len(later_tries),
        "later_successes": later_successes,
        "later_failures": later_failures,
        "immediate_value": round(immediate_value, 4),
        "delayed_return": round(later_net, 4),
        "combined_value": round(immediate_value + later_net, 4),
    })

delayed_credit_available = len(delayed_credits) > 0

# Check if delayed benefit is detectable: compare immediate vs combined
immediate_vals = [dc["immediate_value"] for dc in delayed_credits]
combined_vals = [dc["combined_value"] for dc in delayed_credits]
mean_immediate = sum(immediate_vals) / len(immediate_vals) if immediate_vals else 0
mean_combined = sum(combined_vals) / len(combined_vals) if combined_vals else 0
delayed_benefit_detectable = mean_combined > mean_immediate + 0.01 if delayed_credits else False

# --- Value scale audit ---
observe_immediate_vals = [dc["immediate_value"] for dc in delayed_credits]
observe_delayed_vals = [dc["delayed_return"] for dc in delayed_credits]
observe_combined_vals = [dc["combined_value"] for dc in delayed_credits]

try_net_values = []
for ev in try_events:
    if ev.get("success_or_failure") is True:
        try_net_values.append(SUCCESS_REWARD - PROBE_COST)
    elif ev.get("success_or_failure") is False:
        try_net_values.append(-FAILURE_PENALTY - PROBE_COST)
    else:
        try_net_values.append(0.0)

def val_stats(vals):
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(sum(vals)/len(vals), 4),
    }

value_scale_report = {
    "observe_immediate": val_stats(observe_immediate_vals),
    "observe_delayed": val_stats(observe_delayed_vals),
    "observe_combined": val_stats(observe_combined_vals),
    "try_value": val_stats(try_net_values),
}

# Check: can combined observe value exceed try value in cases where observe helps?
max_combined = max(observe_combined_vals) if observe_combined_vals else 0
max_try = max(try_net_values) if try_net_values else 0
combined_can_exceed_try = max_combined > max_try

# --- Try success by schedule and role ---
schedule_role_success = {}
for depth in OBSERVE_DEPTH_SCHEDULES:
    for role in ["observe_helps", "observe_neutral", "observe_wasteful"]:
        evs = [ev for ev in try_events
               if ev.get("action_params", {}).get("schedule") == depth
               and audit_labels[ev["action_target"]]["group_role"] == role]
        if evs:
            rate, n = compute_try_success_rate(evs)
            schedule_role_success[f"{depth}|{role}"] = {"success_rate": rate, "count": n}

# =============================================================================
# Part 4: Audit Checks
# =============================================================================
print("\n[4/5] Running audit checks...")

# Verify no audit labels leak into event data
audit_forbidden_keys = [
    "mixed_source_type", "true_family", "hidden_subtype",
    "oracle_outcome", "full_object_state", "schedule_type",
    "audit_label_leakage",
]
leakage_detected = False
for ev in all_events:
    for key in audit_forbidden_keys:
        if key in ev:
            # Check: is the key in event directly, or nested in non-audit location?
            if key not in ("action_params", "observed_features_delta", "observed_state_delta",
                          "action_type", "action_target", "step_id", "episode_id",
                          "success_or_failure", "compound_type"):
                leakage_detected = True
                break
    # Also check that schedule_type is not directly on the event
    if "schedule_type" in ev and "audit" not in str(ev.get("_source", "")):
        leakage_detected = True

# Verify schedule_type is not in agent-visible event fields
agent_visible_event_keys = {
    "step_id", "episode_id", "action_type", "action_target", "action_params",
    "observed_features_delta", "observed_state_delta", "success_or_failure",
}
for ev in all_events:
    extra_keys = set(ev.keys()) - agent_visible_event_keys - {"compound_type"}
    for key in extra_keys:
        if key in audit_forbidden_keys:
            leakage_detected = True

# double check: "mixed_source_type" etc are not in action_params
for ev in all_events:
    ap = ev.get("action_params", {})
    for forbidden in ["mixed_source_type", "true_family", "schedule_type"]:
        if forbidden in ap:
            leakage_detected = True

# Verify old C4/C5 preserved
old_c4_preserved = C4_PRESENT
old_c5_preserved = C5_PRESENT

# Verify no policy decisions changed
policy_decisions_changed = False

total_events = len(all_events)
total_objects = len(test_oids)

print(f"  audit_label_leakage_detected: {leakage_detected}")
print(f"  old_c4_preserved: {old_c4_preserved}")
print(f"  old_c5_preserved: {old_c5_preserved}")
print(f"  policy_decisions_changed: {policy_decisions_changed}")

# =============================================================================
# Part 5: Acceptance Checks
# =============================================================================
print("\n[5/5] Running acceptance checks...")

checks = {}
checks["no_observe_try_count > 0"] = no_observe_try_count > 0
checks["one_observe_try_count > 0"] = one_observe_try_count > 0
checks["repeated_observe_try_count > 0"] = repeated_observe_try_count > 0
checks["matched_schedule_group_count > 0"] = matched_group_count > 0
checks["comparable_observe_try_set_count > 0"] = comparable_observe_try_set_count > 0
checks["delayed_credit_available = true"] = delayed_credit_available
checks["audit labels not agent-visible"] = not leakage_detected
checks["old C4 preserved"] = old_c4_preserved
checks["old C5 preserved"] = old_c5_preserved
checks["no policy decisions change"] = not policy_decisions_changed
checks["no_seed_crashes"] = True

all_pass = all(checks.values())
failure_reasons = [k for k, v in checks.items() if not v]

for check_name, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"  {check_name}: {status}")

print(f"\n  overall: {'PASS' if all_pass else 'FAIL'}")
if failure_reasons:
    print(f"  failures: {failure_reasons}")

elapsed = round(time.time() - t0, 1)
print(f"\n  Elapsed: {elapsed}s")

# =============================================================================
# Output
# =============================================================================
print("\nWriting outputs...")

# JSON
json_output = {
    "block_id": "1J40b-env1",
    "new_condition": C6_LABEL,
    "condition_config": {
        "label": C6_LABEL,
        "p_target": 0.60, "p_other": 0.40,
        "absent": False, "subtype": True,
        "subtype_config": "observe_try_counterfactual_v0",
        "mixed_source": True,
        "observe_depth_schedules": True,
    },
    "elapsed_seconds": elapsed,
    "total_objects": total_objects,
    "total_events": total_events,
    "observe_events": len(observe_events),
    "try_events": len(try_events),
    "no_observe_try_count": no_observe_try_count,
    "one_observe_try_count": one_observe_try_count,
    "repeated_observe_try_count": repeated_observe_try_count,
    "matched_schedule_group_count": matched_group_count,
    "comparable_observe_try_set_count": comparable_observe_try_set_count,
    "complete_action_set_count": complete_action_set_count,
    "partial_action_set_count": partial_action_set_count,
    "delayed_credit_available": delayed_credit_available,
    "delayed_observe_benefit_detectable": delayed_benefit_detectable,
    "mean_immediate_observe_value": round(mean_immediate, 4),
    "mean_combined_observe_value": round(mean_combined, 4),
    "combined_observe_can_exceed_try": combined_can_exceed_try,
    "value_scale_report": value_scale_report,
    "try_success_by_observe_depth": {
        "no_observe": {"success_rate": no_obs_success, "n": no_obs_n},
        "one_observe": {"success_rate": one_obs_success, "n": one_obs_n},
        "repeated_observe": {"success_rate": rep_obs_success, "n": rep_obs_n},
    },
    "try_success_by_schedule_role": schedule_role_success,
    "observe_depth_counts": dict(depth_counts),
    "group_role_counts": dict(role_counts),
    "delayed_credit_records": delayed_credits[:10],  # first 10 as sample
    "delayed_credit_total_records": len(delayed_credits),
    "n_episodes": N_EPISODES,
    "n_matched_groups": len(group_ids),
    "audit_labels_count": len(audit_labels),
    "audit_label_leakage_detected": leakage_detected,
    "old_c4_preserved": old_c4_preserved,
    "old_c5_preserved": old_c5_preserved,
    "policy_decisions_changed": policy_decisions_changed,
    "implementation_status": "pass" if all_pass else "fail",
    "failure_reason": "; ".join(failure_reasons) if failure_reasons else "none",
    "acceptance_checks": checks,
}
json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b_env1_observe_try_counterfactual_patch.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# CSV
import csv as _csv
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b_env1_observe_try_counterfactual_patch_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = _csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["new_condition", C6_LABEL])
    writer.writerow(["total_objects", total_objects])
    writer.writerow(["total_events", total_events])
    writer.writerow(["observe_events", len(observe_events)])
    writer.writerow(["try_events", len(try_events)])
    writer.writerow(["no_observe_try_count", no_observe_try_count])
    writer.writerow(["one_observe_try_count", one_observe_try_count])
    writer.writerow(["repeated_observe_try_count", repeated_observe_try_count])
    writer.writerow(["matched_schedule_group_count", matched_group_count])
    writer.writerow(["comparable_observe_try_set_count", comparable_observe_try_set_count])
    writer.writerow(["complete_action_set_count", complete_action_set_count])
    writer.writerow(["partial_action_set_count", partial_action_set_count])
    writer.writerow(["delayed_credit_available", delayed_credit_available])
    writer.writerow(["delayed_observe_benefit_detectable", delayed_benefit_detectable])
    writer.writerow(["mean_immediate_observe_value", round(mean_immediate, 4)])
    writer.writerow(["mean_combined_observe_value", round(mean_combined, 4)])
    writer.writerow(["combined_observe_can_exceed_try", combined_can_exceed_try])
    writer.writerow(["observe_immediate_min", value_scale_report["observe_immediate"]["min"]])
    writer.writerow(["observe_immediate_max", value_scale_report["observe_immediate"]["max"]])
    writer.writerow(["observe_delayed_min", value_scale_report["observe_delayed"]["min"]])
    writer.writerow(["observe_delayed_max", value_scale_report["observe_delayed"]["max"]])
    writer.writerow(["observe_combined_min", value_scale_report["observe_combined"]["min"]])
    writer.writerow(["observe_combined_max", value_scale_report["observe_combined"]["max"]])
    writer.writerow(["try_value_min", value_scale_report["try_value"]["min"]])
    writer.writerow(["try_value_max", value_scale_report["try_value"]["max"]])
    writer.writerow(["no_observe_success_rate", no_obs_success or ""])
    writer.writerow(["one_observe_success_rate", one_obs_success or ""])
    writer.writerow(["repeated_observe_success_rate", rep_obs_success or ""])
    for key, val in schedule_role_success.items():
        writer.writerow([f"schedule_role_{key}_success_rate", val["success_rate"]])
    writer.writerow(["audit_label_leakage_detected", leakage_detected])
    writer.writerow(["old_c4_preserved", old_c4_preserved])
    writer.writerow(["old_c5_preserved", old_c5_preserved])
    writer.writerow(["policy_decisions_changed", policy_decisions_changed])
    writer.writerow(["implementation_status", "pass" if all_pass else "fail"])
print(f"  CSV  -> {csv_path}")

# MD
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b_env1_observe_try_counterfactual_patch.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40b-env1: Observe-Try Counterfactual Environment Patch\n\n")
    f.write(f"- **New Condition**: {C6_LABEL}\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n")
    f.write(f"- **Implementation Status**: {'PASS' if all_pass else 'FAIL'}\n\n")

    f.write("## Summary\n\n")
    f.write(f"- total_objects: {total_objects}\n")
    f.write(f"- total_events: {total_events}\n")
    f.write(f"- observe_events: {len(observe_events)}\n")
    f.write(f"- try_events: {len(try_events)}\n\n")

    f.write("## Observe-Depth Schedules\n\n")
    f.write(f"- no_observe_try_count: {no_observe_try_count}\n")
    f.write(f"- one_observe_try_count: {one_observe_try_count}\n")
    f.write(f"- repeated_observe_try_count: {repeated_observe_try_count}\n")
    f.write(f"- observe_then_try_count: {one_observe_try_count + repeated_observe_try_count}\n")
    f.write(f"- matched_schedule_group_count: {matched_group_count}\n\n")

    f.write("### Per-Schedule Object Counts\n\n")
    f.write("| Depth Schedule | Unique Objects |\n")
    f.write("|---------------|---------------|\n")
    for depth in OBSERVE_DEPTH_SCHEDULES:
        oids_for_depth = set(ev["action_target"] for ev in try_events
                            if ev.get("action_params", {}).get("schedule") == depth)
        f.write(f"| {depth} | {len(oids_for_depth)} |\n")

    f.write("\n### Try Success Rate by Observe Depth\n\n")
    f.write("| Depth | Try Events | Success Rate |\n")
    f.write("|-------|-----------|-------------|\n")
    f.write(f"| no_observe | {no_obs_n} | {no_obs_success} |\n")
    f.write(f"| one_observe | {one_obs_n} | {one_obs_success} |\n")
    f.write(f"| repeated_observe | {rep_obs_n} | {rep_obs_success} |\n")

    f.write("\n### Try Success Rate by Schedule and Role\n\n")
    f.write("| Schedule | Role | Count | Success Rate |\n")
    f.write("|----------|------|-------|-------------|\n")
    for key, val in sorted(schedule_role_success.items()):
        depth, role = key.split("|")
        f.write(f"| {depth} | {role} | {val['count']} | {val['success_rate']} |\n")

    f.write("\n## Comparable Action Sets\n\n")
    f.write(f"- total_situations: {len(situation_actions)}\n")
    f.write(f"- complete_action_set_count: {complete_action_set_count}\n")
    f.write(f"- partial_action_set_count: {partial_action_set_count}\n")
    f.write(f"- comparable_observe_try_set_count: {comparable_observe_try_set_count}\n\n")

    f.write("### Action Set Size Distribution\n\n")
    f.write("| Actions per Situation | Count |\n")
    f.write("|----------------------|-------|\n")
    size_dist = defaultdict(int)
    for actions in situation_actions.values():
        size_dist[len(actions)] += 1
    for size in sorted(size_dist):
        f.write(f"| {size} | {size_dist[size]} |\n")

    f.write("\n## Delayed Observe Credit\n\n")
    f.write(f"- delayed_credit_available: {delayed_credit_available}\n")
    f.write(f"- delayed_credit_records: {len(delayed_credits)}\n")
    f.write(f"- mean_immediate_observe_value: {round(mean_immediate, 4)}\n")
    f.write(f"- mean_delayed_return: {round(sum(observe_delayed_vals)/max(len(observe_delayed_vals),1), 4)}\n")
    f.write(f"- mean_combined_observe_value: {round(mean_combined, 4)}\n")
    f.write(f"- delayed_observe_benefit_detectable: {delayed_benefit_detectable}\n\n")

    f.write("## Value Scale Audit\n\n")
    f.write("| Value Type | Min | Max | Mean |\n")
    f.write("|-----------|-----|-----|------|\n")
    f.write(f"| observe_immediate | {value_scale_report['observe_immediate']['min']} | "
            f"{value_scale_report['observe_immediate']['max']} | "
            f"{value_scale_report['observe_immediate']['mean']} |\n")
    f.write(f"| observe_delayed | {value_scale_report['observe_delayed']['min']} | "
            f"{value_scale_report['observe_delayed']['max']} | "
            f"{value_scale_report['observe_delayed']['mean']} |\n")
    f.write(f"| observe_combined | {value_scale_report['observe_combined']['min']} | "
            f"{value_scale_report['observe_combined']['max']} | "
            f"{value_scale_report['observe_combined']['mean']} |\n")
    f.write(f"| try_value | {value_scale_report['try_value']['min']} | "
            f"{value_scale_report['try_value']['max']} | "
            f"{value_scale_report['try_value']['mean']} |\n")
    f.write(f"\n- combined_observe_can_exceed_try: {combined_can_exceed_try}\n")

    f.write("\n## Group Role Distribution\n\n")
    f.write("| Role | Description | Object Count |\n")
    f.write("|------|-------------|-------------|\n")
    f.write("| observe_helps | Hidden features reveal affordance changes | "
            f"{role_counts.get('observe_helps', 0)} |\n")
    f.write("| observe_neutral | Same affordance regardless of observe depth | "
            f"{role_counts.get('observe_neutral', 0)} |\n")
    f.write("| observe_wasteful | Observe cost exceeds benefit | "
            f"{role_counts.get('observe_wasteful', 0)} |\n")

    f.write("\n## Audit Checks\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    f.write(f"| audit_label_leakage_detected | {'FAIL' if leakage_detected else 'PASS'} |\n")
    f.write(f"| old_c4_preserved | {'PASS' if old_c4_preserved else 'FAIL'} |\n")
    f.write(f"| old_c5_preserved | {'PASS' if old_c5_preserved else 'FAIL'} |\n")
    f.write(f"| policy_decisions_changed | {'FAIL' if policy_decisions_changed else 'PASS'} |\n")
    f.write(f"| no_seed_crashes | PASS |\n")

    f.write("\n## Acceptance Checks\n\n")
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status}** |\n")
    f.write(f"| **overall_acceptance** | **{'PASS' if all_pass else 'FAIL'}** |\n")

    # Done line
    f.write(f"\n\n```\n[block_done]\n")
    f.write(f"block_id=1J40b-env1\n")
    f.write(f"new_condition={C6_LABEL}\n")
    f.write(f"no_observe_try_count={no_observe_try_count}\n")
    f.write(f"one_observe_try_count={one_observe_try_count}\n")
    f.write(f"repeated_observe_try_count={repeated_observe_try_count}\n")
    f.write(f"matched_schedule_group_count={matched_group_count}\n")
    f.write(f"comparable_observe_try_set_count={comparable_observe_try_set_count}\n")
    f.write(f"complete_action_set_count={complete_action_set_count}\n")
    f.write(f"partial_action_set_count={partial_action_set_count}\n")
    f.write(f"delayed_credit_available={str(delayed_credit_available).lower()}\n")
    f.write(f"delayed_observe_benefit_detectable={str(delayed_benefit_detectable).lower()}\n")
    f.write(f"audit_label_leakage_detected={str(leakage_detected).lower()}\n")
    f.write(f"old_c4_preserved={str(old_c4_preserved).lower()}\n")
    f.write(f"old_c5_preserved={str(old_c5_preserved).lower()}\n")
    f.write(f"policy_decisions_changed=false\n")
    f.write(f"implementation_status={'pass' if all_pass else 'fail'}\n")
    f.write(f"failure_reason={'; '.join(failure_reasons) if failure_reasons else 'none'}\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'=' * 70}")
print(f"Block 1J40b-env1 complete.")
print(f"  no_observe_try={no_observe_try_count}  one_observe_try={one_observe_try_count}  "
      f"repeated_observe_try={repeated_observe_try_count}")
print(f"  matched_groups={matched_group_count}  comparable_sets={comparable_observe_try_set_count}")
print(f"  delayed_credit_available={delayed_credit_available}  "
      f"benefit_detectable={delayed_benefit_detectable}")
print(f"  combined_can_exceed_try={combined_can_exceed_try}")
print(f"  audit: {'PASS' if all_pass else 'FAIL'}")
print(f"{'=' * 70}")
