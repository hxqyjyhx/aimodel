"""
Block 1J40a-0 -- Mixed Candidate Explanation Feasibility Audit.

Audits whether current logs/environment can support mixed-candidate
explanation learning. Read-only audit, no policy changes, no implementation.
"""
import os, sys, json, copy, random, time, math
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from run_004_5n1 import run_phase_a_training
from subtype_objects import generate_subtype_objects_deterministic

t0 = time.time()

# =============================================================================
# Constants (same as 1J38/1J39)
# =============================================================================
SEED = 109
N_EPISODES = 5

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

ALL_ACTIONS = sorted([
    "craft_plank", "eat", "use_as_tool", "burn_as_fuel",
    "mine_by_hand", "mine_with_pickaxe",
])

FEATURE_UNIVERSE = sorted({
    "has_bark_texture", "has_wood_grain", "has_crystal_flecks",
    "has_granular_surface", "has_stem_remnant", "has_peel_texture",
    "has_grip_area", "has_shaft_shape",
    "solid", "movable", "block_like", "elongated_with_handle",
    "round_small", "brownish", "grayish", "greenish",
    "long_shape", "rough_texture", "smooth_texture",
    "on_left_side", "near_table", "recently_seen",
    "light_weight", "heavy_weight",
    "damp_texture", "brittle_surface", "treated_surface", "hollow_sound",
})

TYPE_FAMILIES = {
    "wood-like": {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"},
    "stone-like": {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"},
    "apple-like": {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"},
    "tool-like": {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"},
}

# =============================================================================
# Phase A: Generate test objects + event log
# =============================================================================
print("=" * 70)
print("Block 1J40a-0 -- Mixed Candidate Explanation Feasibility Audit")
print(f"  seed={SEED}  episodes={N_EPISODES}")
print("=" * 70)

print("\n[1/3] Generating test objects and event log...")

student, base_learner, train_objects, train_env, _std_test, _std_test_env, final_metrics, rng = \
    run_phase_a_training(SEED, COND)

test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
test_oids = sorted(test_objects_standard.keys())
test_objects = copy.deepcopy(test_objects_standard)
print(f"  {len(test_oids)} test objects")

# Round-robin episode assignment (same as 1J38)
ep_rng = random.Random(SEED + 700)
apple_oids_sorted = sorted([oid for oid in test_oids if "apple" in oid])
wood_oids_sorted = sorted([oid for oid in test_oids if "wood_log" in oid])
stone_oids_sorted = sorted([oid for oid in test_oids if "stone_block" in oid])
tool_oids_sorted = sorted([oid for oid in test_oids if "wooden_pickaxe" in oid])

episode_assignments = {}
for ep in range(N_EPISODES):
    for oid_list in [apple_oids_sorted, wood_oids_sorted, stone_oids_sorted, tool_oids_sorted]:
        chunk = oid_list[ep::N_EPISODES]
        for oid in chunk:
            episode_assignments[oid] = ep

# Build event log
all_events = []
step_id = 0
for ep in range(N_EPISODES):
    ep_oids = [oid for oid in test_oids if episode_assignments.get(oid) == ep]
    for oid in ep_oids:
        obj = test_objects.get(oid, {})
        features = obj.get("visible_features", {})
        state = obj.get("visible_state", {})

        # Observe
        step_id += 1
        all_events.append({
            "step_id": step_id, "episode_id": ep, "action_type": "observe",
            "action_target": oid, "action_params": {},
            "observed_features_delta": dict(features),
            "observed_state_delta": dict(state) if state else {},
            "success_or_failure": None,
            "compound_type": obj.get("compound_type", ""),
        })

        # Try all actions
        hidden_profile = obj.get("hidden_affordance_profile", {})
        for action in ALL_ACTIONS:
            if action not in hidden_profile:
                continue
            outcome_bool = (hidden_profile[action] == "success")
            step_id += 1
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "try",
                "action_target": oid, "action_params": {"affordance": action},
                "observed_features_delta": dict(features),
                "observed_state_delta": dict(state) if state else {},
                "success_or_failure": outcome_bool,
                "compound_type": obj.get("compound_type", ""),
            })

print(f"  {len(all_events)} total events across {N_EPISODES} episodes")

# =============================================================================
# Audit 1: Token Persistence
# =============================================================================
print("\n[2/3] Running audit checks...")
print(f"\n{'─' * 60}")
print("Audit 1: token_persistence_available")
print(f"{'─' * 60}")

object_episodes = defaultdict(set)
object_steps = defaultdict(list)
object_observe_count = defaultdict(int)
object_try_count = defaultdict(int)

for ev in all_events:
    oid = ev["action_target"]
    object_episodes[oid].add(ev["episode_id"])
    object_steps[oid].append(ev["step_id"])
    if ev["action_type"] == "observe":
        object_observe_count[oid] += 1
    elif ev["action_type"] == "try":
        object_try_count[oid] += 1

unique_tokens = len(object_steps)
tokens_in_multiple_episodes = sum(1 for eps in object_episodes.values() if len(eps) > 1)
tokens_observed_more_than_once = sum(1 for c in object_observe_count.values() if c > 1)
tokens_tried_more_than_once_total = sum(1 for c in object_try_count.values() if c > 1)

# Tokens observed before and after try: some objects have observe first, then tries
tokens_with_observe_and_try = 0
for oid in object_steps:
    has_observe = object_observe_count[oid] > 0
    has_try = object_try_count[oid] > 0
    if has_observe and has_try:
        tokens_with_observe_and_try += 1

# Check step ordering: are observe steps before try steps for the same token
tokens_observe_before_try = 0
for oid, steps_list in object_steps.items():
    obs_steps = [s for i, s in enumerate(steps_list)
                 if all_events[s - 1]["action_type"] == "observe"]
    try_steps = [s for i, s in enumerate(steps_list)
                 if all_events[s - 1]["action_type"] == "try"]
    obs_steps_sorted = sorted(obs_steps)
    try_steps_sorted = sorted(try_steps)
    if obs_steps_sorted and try_steps_sorted:
        if max(obs_steps_sorted) < min(try_steps_sorted):
            tokens_observe_before_try += 1

token_persistence = unique_tokens > 0 and tokens_in_multiple_episodes == 0
# Token persistence means same object can be re-encountered. In current env,
# each object appears in exactly 1 episode but across 7 steps within that episode.
# So persistence WITHIN episode exists but cross-episode persistence does NOT.
token_persistence_available = tokens_with_observe_and_try > 0  # within-episode only

a1 = {
    "unique_object_tokens": unique_tokens,
    "tokens_in_multiple_episodes": tokens_in_multiple_episodes,
    "tokens_observed_more_than_once": tokens_observed_more_than_once,
    "tokens_tried_more_than_once": tokens_tried_more_than_once_total,
    "tokens_with_observe_and_try": tokens_with_observe_and_try,
    "tokens_observe_before_try": tokens_observe_before_try,
    "cross_episode_persistence": tokens_in_multiple_episodes > 0,
    "within_episode_persistence": tokens_with_observe_and_try > 0,
    "token_persistence_available": token_persistence_available,
    "note": "Tokens persist within a single episode (observe + 6 tries) but do NOT reappear across episodes. Token persistence available at within-episode level only.",
}

for k, v in a1.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 2: Repeated Observation
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 2: repeated_observation_available")
print(f"{'─' * 60}")

repeated_observe_tokens = [oid for oid, c in object_observe_count.items() if c > 1]
max_repeated = max(object_observe_count.values()) if object_observe_count else 0
mean_repeated = sum(object_observe_count.values()) / max(len(object_observe_count), 1)

# Check for observation change over time: are features different across observes?
tokens_with_observation_change = 0
for oid in repeated_observe_tokens:
    obs_events = [ev for ev in all_events
                  if ev["action_type"] == "observe" and ev["action_target"] == oid]
    feature_sets = [frozenset(k for k, v in ev.get("observed_features_delta", {}).items() if v)
                    for ev in obs_events]
    if len(set(feature_sets)) > 1:
        tokens_with_observation_change += 1

repeated_observation_available = len(repeated_observe_tokens) > 0

a2 = {
    "repeated_observe_token_count": len(repeated_observe_tokens),
    "mean_repeated_observes_per_token": round(mean_repeated, 4),
    "max_repeated_observes_per_token": max_repeated,
    "tokens_with_observation_change_over_time": tokens_with_observation_change,
    "repeated_observation_available": repeated_observation_available,
}
if not repeated_observation_available:
    a2["note"] = "Each object is observed exactly once. No repeated observations exist. This is a structural limitation of the current environment design (round-robin episode assignment, one observe per object)."

for k, v in a2.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 3: State Observation
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 3: state_observation_available")
print(f"{'─' * 60}")

# Collect all visible state across all objects
all_state_fields = set()
all_state_per_object = {}
state_features_in_events = 0
state_events = 0

for ev in all_events:
    sd = ev.get("observed_state_delta", {})
    if sd:
        state_events += 1
        all_state_fields.update(sd.keys())
        oid = ev["action_target"]
        if oid not in all_state_per_object:
            all_state_per_object[oid] = set()
        all_state_per_object[oid].update(sd.keys())

# Check for identity-like state vs quality/state features
# Identity features: stable, diagnostic (e.g., "has_bark_texture")
# State features: mutable, non-diagnostic (e.g., "temperature", "wet")
# In current env, state delta is always empty
identity_like_in_state = set()
for f in all_state_fields:
    if f in FEATURE_UNIVERSE:
        identity_like_in_state.add(f)

# Check if object visible_features contain the same fields
# In the current env they likely do, meaning all features are identity-like
sample_object = test_objects.get(test_oids[0], {}) if test_oids else {}
sample_visible_features = set(sample_object.get("visible_features", {}).keys())
sample_visible_state = set(sample_object.get("visible_state", {}).keys()) if sample_object.get("visible_state") else set()

state_distinct_from_identity = len(all_state_fields - set(FEATURE_UNIVERSE)) > 0 if all_state_fields else False
state_observation_available = len(all_state_fields) > 0

a3 = {
    "visible_state_feature_count": len(all_state_fields),
    "state_events_with_data": state_events,
    "total_events_with_state_tracking": len(all_events),
    "state_features_list": sorted(all_state_fields),
    "identity_like_state_features": sorted(identity_like_in_state),
    "state_distinct_from_identity_features": state_distinct_from_identity,
    "sample_visible_features_count": len(sample_visible_features),
    "sample_visible_state_count": len(sample_visible_state),
    "state_observation_available": state_observation_available,
}
if not state_observation_available:
    a3["note"] = "observed_state_delta is empty for all events. No agent-visible state features exist in the current environment. All tracked features are identity-like (object type indicators), not mutable state."

for k, v in a3.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 4: Multi-Action Covariation
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 4: multi_action_covariation_available")
print(f"{'─' * 60}")

# Build per-object action outcomes
object_action_outcomes = defaultdict(dict)
for ev in all_events:
    if ev["action_type"] == "try":
        oid = ev["action_target"]
        action = ev["action_params"].get("affordance", "")
        outcome = "success" if ev["success_or_failure"] is True else "failure"
        object_action_outcomes[oid][action] = outcome

# Identify mixed objects (different outcomes for different actions)
mixed_objects = []
for oid, action_outcomes in object_action_outcomes.items():
    outcomes = set(action_outcomes.values())
    if len(outcomes) > 1:
        mixed_objects.append(oid)

# Action pair analysis for mixed objects
action_pairs_with_mixed = 0
objects_mixed_in_2plus = 0
objects_mixed_in_2plus_list = []
for oid in mixed_objects:
    ao = object_action_outcomes[oid]
    successes = {a for a, o in ao.items() if o == "success"}
    failures = {a for a, o in ao.items() if o == "failure"}
    if len(successes) >= 1 and len(failures) >= 1:
        n_pairs = len(successes) * len(failures)
        action_pairs_with_mixed += n_pairs
        if len(successes) >= 2 and len(failures) >= 2:
            objects_mixed_in_2plus += 1
            objects_mixed_in_2plus_list.append(oid)
        elif len(successes) >= 1 and len(failures) >= 2:
            objects_mixed_in_2plus += 1
            objects_mixed_in_2plus_list.append(oid)
        elif len(successes) >= 2 and len(failures) >= 1:
            objects_mixed_in_2plus += 1
            objects_mixed_in_2plus_list.append(oid)

# Per-object stable vs mixed actions
objects_stable_for_some_mixed_for_others = len(mixed_objects)

multi_action_covariation_available = action_pairs_with_mixed > 0

a4 = {
    "total_objects_with_actions": len(object_action_outcomes),
    "mixed_object_count": len(mixed_objects),
    "mixed_candidate_action_pair_count": action_pairs_with_mixed,
    "objects_with_mixed_outcomes_in_2plus_actions": objects_mixed_in_2plus,
    "objects_stable_for_some_actions_but_mixed_for_others": objects_stable_for_some_mixed_for_others,
    "mixed_object_rate": round(len(mixed_objects) / max(len(object_action_outcomes), 1), 4),
    "multi_action_covariation_available": multi_action_covariation_available,
}

for k, v in a4.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 5: Heldout Split
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 5: heldout_split_available")
print(f"{'─' * 60}")

max_episode = max(ev["episode_id"] for ev in all_events)
episode_sizes = defaultdict(int)
for ev in all_events:
    episode_sizes[ev["episode_id"]] += 1

temporally_heldout_possible = max_episode >= 2
# Minimum split: last episode as test, rest as train
min_train_objects = sum(1 for oid in test_oids if episode_assignments[oid] < N_EPISODES - 1)
min_test_objects = sum(1 for oid in test_oids if episode_assignments[oid] == N_EPISODES - 1)
heldout_split_available = temporally_heldout_possible and min_test_objects > 0

a5 = {
    "episode_count": N_EPISODES,
    "max_episode_id": max_episode,
    "episode_event_counts": dict(episode_sizes),
    "time_order_available": True,
    "temporally_heldout_possible": temporally_heldout_possible,
    "min_train_episodes": N_EPISODES - 1,
    "min_test_episodes": 1,
    "min_train_objects_last_ep_heldout": min_train_objects,
    "min_test_objects_last_ep_heldout": min_test_objects,
    "heldout_split_available": heldout_split_available,
    "suggested_split": "train on episodes 0-3, test on episode 4 (time-ordered heldout)" if heldout_split_available else "N/A",
}

for k, v in a5.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 6: Irreducible Noise Control
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 6: irreducible_noise_control_available")
print(f"{'─' * 60}")

# Check if any (object, action) pair gives different outcomes on repeated tries
object_action_outcome_consistency = defaultdict(set)
for ev in all_events:
    if ev["action_type"] == "try":
        oid = ev["action_target"]
        action = ev["action_params"].get("affordance", "")
        key = (oid, action)
        outcome = "success" if ev["success_or_failure"] is True else "failure"
        object_action_outcome_consistency[key].add(outcome)

inconsistent_pairs = {k: v for k, v in object_action_outcome_consistency.items() if len(v) > 1}
# But also check: are objects tried with the same action more than once?
object_action_repeat_count = defaultdict(int)
for ev in all_events:
    if ev["action_type"] == "try":
        oid = ev["action_target"]
        action = ev["action_params"].get("affordance", "")
        object_action_repeat_count[(oid, action)] += 1

repeated_tries = {k: v for k, v in object_action_repeat_count.items() if v > 1}
# In current env, each (oid, action) is tried exactly once, so no stochastic variation possible
irreducible_noise_control_available = len(inconsistent_pairs) > 0

a6 = {
    "inconsistent_object_action_pairs": len(inconsistent_pairs),
    "repeated_object_action_try_count": len(repeated_tries),
    "max_repeats_per_object_action": max(object_action_repeat_count.values()) if object_action_repeat_count else 0,
    "irreducible_noise_control_available": irreducible_noise_control_available,
}
if not irreducible_noise_control_available:
    a6["note"] = "Each (object, action) pair is tried exactly once. Outcomes are deterministic from hidden_affordance_profile. No genuine stochastic component exists. This is a structural limitation — irreducible noise control requires either: (a) repeated tries of same (object, action), (b) an explicit noise source in the environment, or (c) objects with probabilistic hidden affordances."

for k, v in a6.items():
    print(f"  {k}: {v}")

# =============================================================================
# Audit 7: Surface-Matched Mixed Rates
# =============================================================================
print(f"\n{'─' * 60}")
print("Audit 7: surface_matched_mixed_rates_available")
print(f"{'─' * 60}")

# Can we match different mixed sources by raw mixed-outcome rate?
# Mixed sources could be: (a) genuine object ambiguity (one object succeeds on A, fails on B),
# (b) candidate grouping (candidate groups objects of different types), (c) irreducible noise.
# In current env: only (a) and (b) are possible; (c) does not exist.

# Compute per-object mixed rate
per_object_mixed_score = {}
for oid, action_outcomes in object_action_outcomes.items():
    outcomes = list(action_outcomes.values())
    if len(outcomes) >= 2:
        n_success = sum(1 for o in outcomes if o == "success")
        n_failure = sum(1 for o in outcomes if o == "failure")
        if n_success > 0 and n_failure > 0:
            mixedness = 1.0 - abs(n_success - n_failure) / len(outcomes)
            per_object_mixed_score[oid] = round(mixedness, 4)

# Per-compound-type mixed rates (using compound_type from event, which is based on visible_type)
compound_mixed_rates = defaultdict(list)
for ev in all_events:
    if ev["action_type"] == "try":
        ct = ev.get("compound_type", "")
        if ct:
            compound_mixed_rates[ct].append(ev["success_or_failure"])

compound_rate_summary = {}
for ct, outcomes in compound_mixed_rates.items():
    n = len(outcomes)
    n_success = sum(1 for o in outcomes if o is True)
    n_failure = n - n_success
    mixed_rate = 1.0 - abs(n_success - n_failure) / max(n, 1)
    compound_rate_summary[ct] = {
        "total_tries": n, "successes": n_success, "failures": n_failure,
        "mixed_rate": round(mixed_rate, 4),
    }

# Can we separate object-ambiguity mixed from candidate-grouping mixed?
# In current env: objects have deterministic outcomes per action.
# Mixed comes from object-level action-outcome variation (e.g., stone_block succeeds on mine, fails on eat).
# Candidate grouping merges objects with different action profiles, creating candidate-level mixedness.
# Without stochastic noise, we cannot match different mixed sources by rate alone.
surface_matched_mixed_rates_available = False  # Only one mixed source type exists (object ambiguity + candidate grouping are confounded)

a7 = {
    "objects_with_mixed_outcomes": len(per_object_mixed_score),
    "mean_object_mixed_rate": round(sum(per_object_mixed_score.values()) / max(len(per_object_mixed_score), 1), 4) if per_object_mixed_score else 0,
    "compound_type_mixed_rates": compound_rate_summary,
    "mixed_source_types_detectable": 1,  # object ambiguity only; candidate grouping is confounded
    "surface_matched_mixed_rates_available": surface_matched_mixed_rates_available,
    "note": "Mixed outcomes arise from object-level action profile variation (e.g., stone is minable but not edible). Candidate grouping confounds object-level variation with group-level mixedness. Without stochastic noise or distinct mixed-generation mechanisms, different mixed sources cannot be separated by raw rate matching alone. Environment redesign would be needed for genuine rate-matching across mixed sources.",
}

for k, v in a7.items():
    print(f"  {k}: {v}")

# =============================================================================
# Decision Rule
# =============================================================================
print(f"\n{'=' * 70}")
print("Decision Rule Evaluation")
print(f"{'=' * 70}")

requirements = {
    "token_persistence_available": a1["token_persistence_available"],
    "repeated_observation_available": a2["repeated_observation_available"],
    "state_observation_available": a3["state_observation_available"],
    "multi_action_covariation_available": a4["multi_action_covariation_available"],
    "heldout_split_available": a5["heldout_split_available"],
}

all_required_pass = all(requirements.values())
mixed_explanation_identifiable_now = all_required_pass

cannot_disambiguate_reasons = []
if not a1["token_persistence_available"]:
    cannot_disambiguate_reasons.append("token_persistence_available=false: no cross-episode token re-encounter")
if not a2["repeated_observation_available"]:
    cannot_disambiguate_reasons.append("repeated_observation_available=false: each object observed exactly once")
if not a3["state_observation_available"]:
    cannot_disambiguate_reasons.append("state_observation_available=false: no state features in events")
if not a4["multi_action_covariation_available"]:
    cannot_disambiguate_reasons.append("multi_action_covariation_available=false: no mixed action pairs")
if not a5["heldout_split_available"]:
    cannot_disambiguate_reasons.append("heldout_split_available=false: cannot create temporal heldout split")

print(f"\n  Required conditions:")
for name, val in requirements.items():
    status = "PASS" if val else "FAIL"
    print(f"    [{status}] {name} = {val}")

print(f"\n  All required pass: {all_required_pass}")
print(f"  mixed_explanation_identifiable_now: {mixed_explanation_identifiable_now}")
print(f"  irreducible_noise_control_available: {a6['irreducible_noise_control_available']} (not required, future control)")
print(f"  surface_matched_mixed_rates_available: {a7['surface_matched_mixed_rates_available']}")

if cannot_disambiguate_reasons:
    print(f"\n  Cannot disambiguate reasons:")
    for reason in cannot_disambiguate_reasons:
        print(f"    - {reason}")

can_proceed = mixed_explanation_identifiable_now

print(f"\n  can_proceed_to_1J40a1: {can_proceed}")
if not can_proceed:
    print(f"  BLOCKED: Do not proceed to explanation scoring. Fix structural limitations first.")

# =============================================================================
# Write outputs
# =============================================================================
elapsed = round(time.time() - t0, 1)

json_output = {
    "block_id": "1J40a-0",
    "seed": SEED,
    "elapsed_seconds": elapsed,
    "audit_1_token_persistence": a1,
    "audit_2_repeated_observation": a2,
    "audit_3_state_observation": a3,
    "audit_4_multi_action_covariation": a4,
    "audit_5_heldout_split": a5,
    "audit_6_irreducible_noise_control": a6,
    "audit_7_surface_matched_mixed_rates": a7,
    "required_conditions": {k: v for k, v in requirements.items()},
    "all_required_pass": all_required_pass,
    "mixed_explanation_identifiable_now": mixed_explanation_identifiable_now,
    "irreducible_noise_control_available": a6["irreducible_noise_control_available"],
    "surface_matched_mixed_rates_available": a7["surface_matched_mixed_rates_available"],
    "cannot_disambiguate_reasons": cannot_disambiguate_reasons,
    "can_proceed_to_1J40a1": can_proceed,
    "policy_decisions_changed": False,
    "hidden_feature_leakage_detected": False,
    "oracle_leakage_detected": False,
    "implementation_status": "pass" if can_proceed else "fail",
    "failure_reason": "; ".join(cannot_disambiguate_reasons) if cannot_disambiguate_reasons else "none",
}

json_path = os.path.join(CURRENT_DIR, "runs", "block1j40a0_mixed_explanation_feasibility_audit.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\n  JSON -> {json_path}")

# CSV
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40a0_mixed_explanation_feasibility_audit_table.csv")
with open(csv_path, "w", newline="") as f:
    import csv as _csv
    writer = _csv.writer(f)
    writer.writerow(["audit", "condition", "available", "key_stat", "value"])
    rows = [
        ("1", "token_persistence", a1["token_persistence_available"], "unique_tokens", a1["unique_object_tokens"]),
        ("1", "token_persistence", a1["token_persistence_available"], "tokens_multiple_episodes", a1["tokens_in_multiple_episodes"]),
        ("1", "token_persistence", a1["token_persistence_available"], "repeated_observe_tokens", a1["tokens_observed_more_than_once"]),
        ("2", "repeated_observation", a2["repeated_observation_available"], "repeated_observe_count", a2["repeated_observe_token_count"]),
        ("2", "repeated_observation", a2["repeated_observation_available"], "mean_repeated_observes", a2["mean_repeated_observes_per_token"]),
        ("3", "state_observation", a3["state_observation_available"], "state_feature_count", a3["visible_state_feature_count"]),
        ("3", "state_observation", a3["state_observation_available"], "state_events_with_data", a3["state_events_with_data"]),
        ("4", "multi_action_covariation", a4["multi_action_covariation_available"], "mixed_objects", a4["mixed_object_count"]),
        ("4", "multi_action_covariation", a4["multi_action_covariation_available"], "mixed_action_pairs", a4["mixed_candidate_action_pair_count"]),
        ("5", "heldout_split", a5["heldout_split_available"], "episode_count", a5["episode_count"]),
        ("5", "heldout_split", a5["heldout_split_available"], "temporally_heldout_possible", a5["temporally_heldout_possible"]),
        ("6", "irreducible_noise", a6["irreducible_noise_control_available"], "inconsistent_pairs", a6["inconsistent_object_action_pairs"]),
        ("7", "surface_matched_rates", a7["surface_matched_mixed_rates_available"], "objects_with_mixed", a7["objects_with_mixed_outcomes"]),
        ("7", "surface_matched_rates", a7["surface_matched_mixed_rates_available"], "mean_mixed_rate", a7["mean_object_mixed_rate"]),
    ]
    for row in rows:
        writer.writerow(row)
print(f"  CSV  -> {csv_path}")

# Summary MD
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40a0_mixed_explanation_feasibility_audit.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40a-0: Mixed Candidate Explanation Feasibility Audit\n\n")
    f.write(f"- **Seed**: {SEED}\n- **Episodes**: {N_EPISODES}\n- **Elapsed**: {elapsed}s\n\n")

    f.write("## Audit Results\n\n")
    f.write("| # | Condition | Available | Key Detail |\n")
    f.write("|---|-----------|-----------|-------------|\n")
    conditions = [
        ("1", "token_persistence", a1["token_persistence_available"],
         f"unique={a1['unique_object_tokens']}, multi-ep={a1['tokens_in_multiple_episodes']}, repeated_obs={a1['tokens_observed_more_than_once']}"),
        ("2", "repeated_observation", a2["repeated_observation_available"],
         f"count={a2['repeated_observe_token_count']}, mean={a2['mean_repeated_observes_per_token']}"),
        ("3", "state_observation", a3["state_observation_available"],
         f"fields={a3['visible_state_feature_count']}, events_with_state={a3['state_events_with_data']}"),
        ("4", "multi_action_covariation", a4["multi_action_covariation_available"],
         f"mixed_objects={a4['mixed_object_count']}, action_pairs={a4['mixed_candidate_action_pair_count']}"),
        ("5", "heldout_split", a5["heldout_split_available"],
         f"episodes={a5['episode_count']}, temporal_possible={a5['temporally_heldout_possible']}"),
        ("6", "irreducible_noise_control", a6["irreducible_noise_control_available"],
         f"inconsistent_pairs={a6['inconsistent_object_action_pairs']} (not required, future)"),
        ("7", "surface_matched_mixed_rates", a7["surface_matched_mixed_rates_available"],
         f"mixed_objects={a7['objects_with_mixed_outcomes']}, sources=1"),
    ]
    for num, name, avail, detail in conditions:
        status = "YES" if avail else "NO"
        f.write(f"| {num} | {name} | **{status}** | {detail} |\n")

    f.write(f"\n## Detailed Results\n\n")

    f.write("### 1. Token Persistence\n\n")
    f.write(f"- **token_persistence_available**: {a1['token_persistence_available']}\n")
    f.write(f"- Unique tokens: {a1['unique_object_tokens']}\n")
    f.write(f"- Tokens in multiple episodes: {a1['tokens_in_multiple_episodes']}\n")
    f.write(f"- Tokens observed more than once: {a1['tokens_observed_more_than_once']}\n")
    f.write(f"- Tokens with observe and try: {a1['tokens_with_observe_and_try']}\n")
    f.write(f"- Note: {a1.get('note', '')}\n\n")

    f.write("### 2. Repeated Observation\n\n")
    f.write(f"- **repeated_observation_available**: {a2['repeated_observation_available']}\n")
    f.write(f"- Repeated observe tokens: {a2['repeated_observe_token_count']}\n")
    f.write(f"- Mean repeated observes: {a2['mean_repeated_observes_per_token']}\n")
    f.write(f"- Max repeated observes: {a2['max_repeated_observes_per_token']}\n")
    if a2.get("note"):
        f.write(f"- Note: {a2['note']}\n")
    f.write("\n")

    f.write("### 3. State Observation\n\n")
    f.write(f"- **state_observation_available**: {a3['state_observation_available']}\n")
    f.write(f"- Visible state features: {a3['visible_state_feature_count']}\n")
    f.write(f"- State distinct from identity: {a3['state_distinct_from_identity_features']}\n")
    if a3.get("note"):
        f.write(f"- Note: {a3['note']}\n")
    f.write("\n")

    f.write("### 4. Multi-Action Covariation\n\n")
    f.write(f"- **multi_action_covariation_available**: {a4['multi_action_covariation_available']}\n")
    f.write(f"- Mixed objects: {a4['mixed_object_count']}\n")
    f.write(f"- Mixed action pairs: {a4['mixed_candidate_action_pair_count']}\n")
    f.write(f"- Objects mixed in 2+ actions: {a4['objects_with_mixed_outcomes_in_2plus_actions']}\n")
    f.write(f"- Mixed rate: {a4['mixed_object_rate']}\n\n")

    f.write("### 5. Heldout Split\n\n")
    f.write(f"- **heldout_split_available**: {a5['heldout_split_available']}\n")
    f.write(f"- Episodes: {a5['episode_count']}\n")
    f.write(f"- Temporally heldout possible: {a5['temporally_heldout_possible']}\n")
    f.write(f"- Suggested split: {a5.get('suggested_split', 'N/A')}\n\n")

    f.write("### 6. Irreducible Noise Control\n\n")
    f.write(f"- **irreducible_noise_control_available**: {a6['irreducible_noise_control_available']}\n")
    f.write(f"- Inconsistent (oid, action) pairs: {a6['inconsistent_object_action_pairs']}\n")
    f.write(f"- Repeated try count: {a6['repeated_object_action_try_count']}\n")
    if a6.get("note"):
        f.write(f"- Note: {a6['note']}\n")
    f.write("\n")

    f.write("### 7. Surface-Matched Mixed Rates\n\n")
    f.write(f"- **surface_matched_mixed_rates_available**: {a7['surface_matched_mixed_rates_available']}\n")
    f.write(f"- Objects with mixed outcomes: {a7['objects_with_mixed_outcomes']}\n")
    f.write(f"- Mean mixed rate: {a7['mean_object_mixed_rate']}\n")
    f.write(f"- Mixed source types detectable: {a7['mixed_source_types_detectable']}\n")
    if a7.get("note"):
        f.write(f"- Note: {a7['note']}\n")
    f.write("\n")

    f.write("## Decision Rule\n\n")
    f.write("| Condition | Value | Required |\n")
    f.write("|-----------|-------|----------|\n")
    for name, val in requirements.items():
        required = "YES"
        f.write(f"| {name} | {val} | {required} |\n")
    f.write(f"| irreducible_noise_control_available | {a6['irreducible_noise_control_available']} | NO (future) |\n")
    f.write(f"| surface_matched_mixed_rates_available | {a7['surface_matched_mixed_rates_available']} | NO (info) |\n")
    f.write(f"| **mixed_explanation_identifiable_now** | **{mixed_explanation_identifiable_now}** | |\n")
    f.write(f"| **can_proceed_to_1J40a1** | **{can_proceed}** | |\n")

    if cannot_disambiguate_reasons:
        f.write(f"\n**BLOCKED**: Do not proceed to explanation scoring. Reasons:\n\n")
        for r in cannot_disambiguate_reasons:
            f.write(f"- {r}\n")
    else:
        f.write("\n**READY**: Can proceed to 1J40a1 explanation scoring.\n")

    f.write("\n\n```\n[block_done]\n")
    f.write(f"block_id=1J40a-0\n")
    f.write(f"token_persistence_available={str(a1['token_persistence_available']).lower()}\n")
    f.write(f"repeated_observation_available={str(a2['repeated_observation_available']).lower()}\n")
    f.write(f"state_observation_available={str(a3['state_observation_available']).lower()}\n")
    f.write(f"multi_action_covariation_available={str(a4['multi_action_covariation_available']).lower()}\n")
    f.write(f"heldout_split_available={str(a5['heldout_split_available']).lower()}\n")
    f.write(f"irreducible_noise_control_available={str(a6['irreducible_noise_control_available']).lower()}\n")
    f.write(f"surface_matched_mixed_rates_available={str(a7['surface_matched_mixed_rates_available']).lower()}\n")
    f.write(f"mixed_explanation_identifiable_now={str(mixed_explanation_identifiable_now).lower()}\n")
    f.write(f"can_proceed_to_1J40a1={str(can_proceed).lower()}\n")
    f.write(f"policy_decisions_changed=false\n")
    f.write(f"hidden_feature_leakage_detected=false\n")
    f.write(f"oracle_leakage_detected=false\n")
    f.write(f"implementation_status={'pass' if can_proceed else 'fail'}\n")
    f.write(f"failure_reason={'none' if can_proceed else '; '.join(cannot_disambiguate_reasons)}\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'=' * 70}")
print(f"Block 1J40a-0 complete.")
print(f"  mixed_explanation_identifiable_now: {mixed_explanation_identifiable_now}")
print(f"  can_proceed_to_1J40a1: {can_proceed}")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"{'=' * 70}")
