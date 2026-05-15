"""
Block 1J18 -- Goal-Conditioned Soft Probe Prior Memory.

Adds a minimal goal-conditioned soft prior memory layer:
visible attributes should raise or lower probe priority under a given goal,
without hard exclusion.

Three policy variants:
  1. soft_prior_only: rank by goal_soft_prior + exploration_floor + cost penalty
  2. c15b_soft_prior_mul: multiply normalized C15b score by goal_soft_prior
  3. c15b_soft_prior_memory: variant 2 + pattern_priority_memory adjustment

Single seed (101) only. No multi-seed.
"""
import os, sys, json, copy, random, time, math
import numpy as np
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
K5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5k_episodic_sparse_outcome")
sys.path.insert(0, K5_DIR)
I5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5i_object_uncertainty_probe")
sys.path.insert(0, I5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from simulator_truth import MiniMCSimulatorTruth
from policies import (
    MiniMCPolicy, C0b_ObserveOnlyPolicy,
    C14a_TruthAnswerOracle,
    MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE, CORE_ACTION_FEATURES,
    _compute_utility, _compute_entropy,
)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes
from subtype_objects import (
    generate_subtype_objects_deterministic, compute_query_ground_truth,
    SUBTYPE_DEFINITIONS,
)

t0 = time.time()

SEED = 101
BUDGET = 1.5
COST_WEIGHT = 0.5

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J18 -- Goal-Conditioned Soft Probe Prior Memory")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print(f"  variants=soft_prior_only+c15b_soft_prior_mul+c15b_soft_prior_memory")
print(f"  method=minimal_goal_conditioned_soft_prior")
print(f"  hard_exclusion=false  exploration_floor>0")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/9] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED)
posterior_visible_features = list(base_learner.visible_feature_names)
visible_feature_names_sorted = sorted(posterior_visible_features)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}  "
      f"coverage={actual_coverage:.4f}  IOM ready")

# =============================================================================
# 2. Test objects
# =============================================================================
print("\n[2/9] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)

ALL_GT_AFFORDANCES = {}
for oid in test_oids:
    ALL_GT_AFFORDANCES[oid] = sim_gt.get_ground_truth_affordances(oid)

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

print(f"  Generated {N_OBJECTS} test objects, {len(QUERY_NAMES)} queries")

# =============================================================================
# 3. Metric helpers
# =============================================================================
def _check_query(probs, query_name):
    if query_name == "need_planks":
        return probs.get("craft_plank_success", 0.5) >= 0.5
    elif query_name == "need_stone":
        return (probs.get("mine_by_hand_success", 0.5) < 0.5
                and probs.get("mine_with_pickaxe_success", 0.5) >= 0.5)
    elif query_name == "need_food":
        return probs.get("eat_success", 0.5) >= 0.5
    elif query_name == "need_tool":
        return probs.get("use_as_tool_success", 0.5) >= 0.5
    elif query_name == "need_fuel":
        return probs.get("burn_as_fuel_success", 0.5) >= 0.5
    return False


def compute_full_metrics(predictions, query_gt):
    per_query = {}
    for qname in sorted(query_gt.keys()):
        pos_set = set(query_gt[qname]["positive_oids"])
        neg_set = set(query_gt[qname]["negative_oids"])
        tp = fp = tn = fn = 0
        for oid in predictions:
            pred_true = _check_query(predictions[oid], qname)
            actual_true = oid in pos_set
            if pred_true and actual_true: tp += 1
            elif pred_true and not actual_true: fp += 1
            elif not pred_true and not actual_true: tn += 1
            else: fn += 1
        pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal = 0.5 * (pos_recall + neg_recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        per_query[qname] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_recall": round(pos_recall, 6),
            "negative_recall": round(neg_recall, 6),
            "balanced_accuracy": round(bal, 6),
            "accuracy": round(accuracy, 6),
        }
    macro_bal = sum(v["balanced_accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    macro_acc = sum(v["accuracy"] for v in per_query.values()) / max(len(per_query), 1)
    mean_pos_recall = sum(v["positive_recall"] for v in per_query.values()) / max(len(per_query), 1)
    mean_neg_recall = sum(v["negative_recall"] for v in per_query.values()) / max(len(per_query), 1)
    return {
        "per_query": per_query,
        "macro_query_balanced_accuracy": round(macro_bal, 6),
        "macro_query_accuracy": round(macro_acc, 6),
        "mean_positive_recall": round(mean_pos_recall, 6),
        "mean_negative_recall": round(mean_neg_recall, 6),
    }


def get_cost_metrics(event_log, obs):
    visit_count = sum(1 for oid in obs.get_all_object_ids() if obs.is_visited(oid))
    probe_count = sum(1 for oid in obs.get_all_object_ids() if obs.get_probe_results(oid))
    total = obs.budget_spent
    return {
        "visit_count": visit_count,
        "observe_count": visit_count,
        "probe_count": probe_count,
        "total_cost": round(total, 4),
        "normalized_cost": round(total / max(obs.initial_budget, 0.001), 4),
    }


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 4. Default Goal Prior Table
# =============================================================================
print("\n[3/9] Building default goal-conditioned soft prior table...")

# Visible feature groups for type-family detection
# These are the characteristic visible features for each object category.
# We use predictor-augmented features as the cleanest discriminators,
# plus some ACTIVE_FEATURES_PHASE_A that correlate with categories.

WOOD_FEATURES = {
    "has_bark_texture", "has_wood_grain",
    "brownish", "rough_texture", "long_shape",
}

STONE_FEATURES = {
    "has_crystal_flecks", "has_granular_surface",
    "grayish", "block_like", "heavy_weight",
}

APPLE_FEATURES = {
    "has_stem_remnant", "has_peel_texture",
    "round_small", "greenish", "smooth_texture", "light_weight",
}

TOOL_FEATURES = {
    "has_grip_area", "has_shaft_shape",
    "elongated_with_handle", "movable", "long_shape",
}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES,
    "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES,
    "tool-like": TOOL_FEATURES,
}

EXPLORATION_FLOOR = 0.05  # No hard exclusion

# Default goal prior: (type_family, query, action) -> prior in [0.05, 0.95]
# Values encode: "under this goal, how promising is probing this action on this type?"
#
# HIGH  (0.85): goal-relevant action on matching type
# MED   (0.60): action on type with some relevance
# LOW   (0.25): action on type with weak relevance
# MIN   (0.10): action on mismatched type (but never zero)
#
# mine_by_hand and mine_with_pickaxe share need_stone query.

DEFAULT_GOAL_PRIOR = {}

# need_food: eat is primary
# apple-like + eat = high, others low
for fam in TYPE_FAMILIES:
    for action in MAIN_CANDIDATE_ACTIONS:
        qname = ACTION_TO_QUERY.get(action, "")
        prior = EXPLORATION_FLOOR  # absolute floor

        if qname == "need_food":
            if action == "eat":
                if fam == "apple-like":
                    prior = 0.85
                elif fam == "tool-like":
                    prior = 0.10
                else:
                    prior = 0.15
            else:
                if fam == "apple-like":
                    prior = 0.30  # other actions on apple: moderate (might help indirectly)
                else:
                    prior = 0.10

        elif qname == "need_fuel":
            if action == "burn_as_fuel":
                if fam == "wood-like":
                    prior = 0.85
                elif fam == "tool-like":
                    prior = 0.60  # wooden tools can burn too
                elif fam == "stone-like":
                    prior = 0.10  # stone doesn't burn
                else:
                    prior = 0.15  # apples don't burn well
            else:
                if fam == "wood-like":
                    prior = 0.25
                else:
                    prior = 0.10

        elif qname == "need_planks":
            if action == "craft_plank":
                if fam == "wood-like":
                    prior = 0.85
                elif fam == "tool-like":
                    prior = 0.15  # tool, not suitable for planks
                else:
                    prior = 0.10
            else:
                if fam == "wood-like":
                    prior = 0.25
                else:
                    prior = 0.10

        elif qname == "need_stone":
            if action in ("mine_by_hand", "mine_with_pickaxe"):
                if fam == "stone-like":
                    prior = 0.85
                elif fam == "wood-like":
                    prior = 0.25  # wood can be mined too (both subtypes succeed)
                elif fam == "tool-like":
                    prior = 0.25
                else:
                    prior = 0.15  # apples: mine succeeds for both subtypes
            else:
                if fam == "stone-like":
                    prior = 0.30
                else:
                    prior = 0.10

        elif qname == "need_tool":
            if action == "use_as_tool":
                if fam == "tool-like":
                    prior = 0.85
                elif fam == "wood-like":
                    prior = 0.15
                else:
                    prior = 0.10
            else:
                if fam == "tool-like":
                    prior = 0.25
                else:
                    prior = 0.10

        # Ensure within bounds
        prior = max(EXPLORATION_FLOOR, min(0.95, prior))
        DEFAULT_GOAL_PRIOR[(fam, qname, action)] = prior


def detect_type_family(features):
    """Determine the visible type family of an object from its observed features.

    Returns the family name with the most matching characteristic features.
    Ties broken by wood > stone > apple > tool order.
    If no features match any family, returns "unknown".
    """
    if features is None:
        return "unknown"

    scores = {}
    for fam, feat_set in TYPE_FAMILIES.items():
        score = 0
        for f in feat_set:
            if features.get(f, False):
                score += 1
        scores[fam] = score

    best_fam = max(scores, key=scores.get)
    if scores[best_fam] == 0:
        return "unknown"
    return best_fam


def get_goal_soft_prior(features, action):
    """Get the soft prior for probing (object, action) under detected type family.

    Returns a single prior value in [EXPLORATION_FLOOR, 0.95].
    The prior is goal-conditioned: each action maps to a query, and the
    prior encodes how promising this action is for that query given the
    object's visible type family.
    """
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "")
    return DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)


def get_per_query_goal_soft_prior(features):
    """Get soft priors for all actions on an object, keyed by action."""
    fam = detect_type_family(features)
    result = {}
    for action in MAIN_CANDIDATE_ACTIONS:
        qname = ACTION_TO_QUERY.get(action, "")
        result[action] = DEFAULT_GOAL_PRIOR.get((fam, qname, action), EXPLORATION_FLOOR)
    return result


# Print prior table for transparency
print("  Default Goal Prior Table (type_family, query, action -> prior):")
print(f"  {'Family':<14} {'Query':<14} {'Action':<22} Prior")
print(f"  {'-'*14} {'-'*14} {'-'*22} -----")
for (fam, qname, action), prior in sorted(DEFAULT_GOAL_PRIOR.items()):
    print(f"  {fam:<14} {qname:<14} {action:<22} {prior:.2f}")

# Verify: no prior is exactly 0
min_prior = min(DEFAULT_GOAL_PRIOR.values())
assert min_prior > 0.0, f"Found zero prior! min={min_prior}"
print(f"  Min prior={min_prior:.4f} (all > 0 [OK])")

# Prior sanity checks
checks_passed = 0
checks_failed = 0

# need_food: eat should rank higher than burn_as_fuel for apple-like
apple_eat = DEFAULT_GOAL_PRIOR[("apple-like", "need_food", "eat")]
apple_burn = DEFAULT_GOAL_PRIOR[("apple-like", "need_fuel", "burn_as_fuel")]
if apple_eat > apple_burn:
    checks_passed += 1
else:
    checks_failed += 1
    print(f"  FAIL: apple-like need_food/eat={apple_eat} <= need_fuel/burn={apple_burn}")

# need_fuel: burn_as_fuel should rank higher than eat for wood-like
wood_burn = DEFAULT_GOAL_PRIOR[("wood-like", "need_fuel", "burn_as_fuel")]
wood_eat = DEFAULT_GOAL_PRIOR[("wood-like", "need_food", "eat")]
if wood_burn > wood_eat:
    checks_passed += 1
else:
    checks_failed += 1
    print(f"  FAIL: wood-like need_fuel/burn={wood_burn} <= need_food/eat={wood_eat}")

# need_tool: use_as_tool should be high for tool-like
tool_use = DEFAULT_GOAL_PRIOR[("tool-like", "need_tool", "use_as_tool")]
if tool_use > 0.7:
    checks_passed += 1
else:
    checks_failed += 1
    print(f"  FAIL: tool-like need_tool/use_as_tool={tool_use} <= 0.7")

# need_planks: craft_plank should be high for wood-like
wood_craft = DEFAULT_GOAL_PRIOR[("wood-like", "need_planks", "craft_plank")]
if wood_craft > 0.7:
    checks_passed += 1
else:
    checks_failed += 1
    print(f"  FAIL: wood-like need_planks/craft_plank={wood_craft} <= 0.7")

# need_stone: mine actions should be high for stone-like
stone_mine = DEFAULT_GOAL_PRIOR[("stone-like", "need_stone", "mine_by_hand")]
if stone_mine > 0.7:
    checks_passed += 1
else:
    checks_failed += 1
    print(f"  FAIL: stone-like need_stone/mine_by_hand={stone_mine} <= 0.7")

print(f"  Prior sanity checks: {checks_passed} passed, {checks_failed} failed")

# =============================================================================
# 5. Probe Event Memory + Pattern Priority Memory
# =============================================================================
print("\n[4/9] Initializing probe event memory and pattern priority memory...")

# probe_event_memory: list of per-probe event records
probe_event_memory = []  # populated as probes happen

# pattern_priority_memory: aggregated by (type_family, goal, action)
# Key format: "family|goal|action" -> aggregate stats
pattern_priority_memory = {}

# Reliability weighting parameter
ALPHA = 3.0  # support_count / (support_count + alpha)


def get_pattern_key(features, action):
    """Build pattern key from visible features and action."""
    fam = detect_type_family(features)
    qname = ACTION_TO_QUERY.get(action, "unknown")
    return f"{fam}|{qname}|{action}"


def record_probe_event(features, action, prior_before, final_probe_score,
                       probe_outcome, query_changed, effect_label, cost):
    """Record a probe event in the probe_event_memory."""
    key = get_pattern_key(features, action)
    probe_event_memory.append({
        "visible_pattern": detect_type_family(features),
        "goal": ACTION_TO_QUERY.get(action, ""),
        "action": action,
        "prior_before": prior_before,
        "final_probe_score": final_probe_score,
        "probe_outcome": probe_outcome,
        "query_changed": query_changed,
        "effect_label": effect_label,
        "cost": cost,
        "pattern_key": key,
    })


def update_pattern_priority_memory():
    """Aggregate probe_event_memory into pattern_priority_memory.

    Groups by (visible_pattern, goal, action) and computes:
    - support_count, effective_count, zero_count, harmful_count
    - effective_rate, harmful_rate, query_changed_rate
    - confidence, priority_adjustment
    """
    pattern_priority_memory.clear()

    # Group events
    groups = {}
    for event in probe_event_memory:
        key = event["pattern_key"]
        if key not in groups:
            groups[key] = []
        groups[key].append(event)

    for key, events in groups.items():
        parts = key.split("|")
        fam = parts[0]
        goal = parts[1]
        action = parts[2]

        support_count = len(events)
        effective_count = sum(1 for e in events if e["effect_label"] == "effective")
        zero_count = sum(1 for e in events if e["effect_label"] == "zero")
        harmful_count = sum(1 for e in events if e["effect_label"] == "harmful")
        query_changed_count = sum(1 for e in events if e["query_changed"])

        effective_rate = effective_count / max(support_count, 1)
        harmful_rate = harmful_count / max(support_count, 1)
        query_changed_rate = query_changed_count / max(support_count, 1)

        # Reliability weighting
        reliability = support_count / (support_count + ALPHA)

        # Priority adjustment: effective_rate drives upward, harmful_rate downward
        # Baseline = 1.0 (no adjustment)
        # effective_rate > 0.5 -> adjust up; harmful_rate > 0.3 -> adjust down
        raw_adjustment = 1.0 + 0.3 * (effective_rate - harmful_rate - 0.3)
        priority_adjustment = 1.0 + reliability * (raw_adjustment - 1.0)
        priority_adjustment = max(0.5, min(2.0, priority_adjustment))

        # Confidence in the adjustment
        confidence = reliability

        pattern_priority_memory[key] = {
            "type_family": fam,
            "goal": goal,
            "action": action,
            "support_count": support_count,
            "effective_count": effective_count,
            "zero_count": zero_count,
            "harmful_count": harmful_count,
            "effective_rate": round(effective_rate, 4),
            "harmful_rate": round(harmful_rate, 4),
            "query_changed_rate": round(query_changed_rate, 4),
            "confidence": round(confidence, 4),
            "priority_adjustment": round(priority_adjustment, 4),
        }


def get_experience_adjustment(features, action):
    """Get experience-based priority adjustment for (object, action).

    Returns adjustment factor in [0.5, 2.0]. 1.0 = no experience yet.
    """
    key = get_pattern_key(features, action)
    if key in pattern_priority_memory:
        return pattern_priority_memory[key]["priority_adjustment"]
    return 1.0


# =============================================================================
# 6. C15b Reference Run (defines candidate universe and probe budget)
# =============================================================================
print("\n[5/9] Running C15b reference...")

class C15b_ProbeReservePolicy(MiniMCPolicy):
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            affordable = view.can_afford(total_cost)
            features = view.get_observed_features(oid)
            if features is None: continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            current_utility = _compute_utility(probs)
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            best_action_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)
                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
                utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
                e_post = p_success * utility_succ + (1 - p_success) * utility_fail
                net_voi = e_post - current_utility - self._cost_weight * norm_cost
                if net_voi > best_action_voi: best_action_voi = net_voi
            if affordable and best_action_voi > best_score:
                best_score = best_action_voi; best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        current_utility = _compute_utility(probs)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        norm_probe_cost = self._normalized_probe_cost(view)
        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)
            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
            e_post = p_success * utility_succ + (1 - p_success) * utility_fail
            net_voi = e_post - current_utility - self._cost_weight * norm_probe_cost
            if net_voi > best_net_voi:
                best_net_voi = net_voi; best_action = action
        should_probe = best_net_voi > 0
        if should_probe: self._probed_oids.add(object_id)
        return should_probe, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            self._im.incorporate_probe(object_id, action, outcome)
            self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})
        else:
            self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}


c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_visited_oids = [oid for oid in test_oids if c15b_obs.is_visited(oid)]
c15b_probe_pairs = [(e["oid"], e["action"]) for e in c15b_policy._probe_tracking]
c15b_probe_pair_set = set(c15b_probe_pairs)
c15b_metrics = compute_full_metrics(c15b_preds["per_object"], query_gt)
c15b_mb = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in test_oids if c15b_obs.get_probe_results(oid))
c15b_visited_n = len(c15b_visited_oids)
print(f"  C15b macro_bal={c15b_mb:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")
print(f"  C15b probe pairs: {c15b_probe_pairs}")

# =============================================================================
# 7. Oracle k=8 Reference (Non-Deployable)
# =============================================================================
print("\n[6/9] Computing oracle k=8 reference...")

# Build oracle k=8 from pre-probe IOM state (C15b observe phase, no probes)
observe_only_im = im_base.clone()
observe_only_policy = C0b_ObserveOnlyPolicy(observe_only_im, random.Random(SEED + 500))
observe_only_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
observe_only_preds, observe_only_log, observe_only_obs, _, _ = run_policy(
    observe_only_policy, observe_only_env)
pre_probe_visited_oids = [oid for oid in test_oids if observe_only_obs.is_visited(oid)]

oracle_errors_all = []
for oid in pre_probe_visited_oids:
    features = observe_only_obs.get_observed_features(oid)
    if features is None: continue
    fake_obj = {"id": oid, "visible_features": features}
    probs = observe_only_im.predict_all_affordances(fake_obj)
    for action in MAIN_CANDIDATE_ACTIONS:
        feature = ACTION_TO_FEATURE[action]
        pred = probs.get(feature, 0.5)
        gt = ALL_GT_AFFORDANCES[oid].get(feature, 0.0)
        error = abs(pred - gt)
        oracle_errors_all.append((error, oid, action, feature, pred, gt))
oracle_errors_all.sort(key=lambda x: x[0], reverse=True)
oracle_k8_pairs_list = oracle_errors_all[:8]
oracle_k8_pair_set = set((oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list)
print(f"  Oracle k=8 pairs: {[(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]}")

def inject_and_predict(im, injections, test_oids):
    im_clone = im.clone()
    for oid, action, outcome in injections:
        feature = ACTION_TO_FEATURE[action]
        im_clone._direct_outcomes[(oid, feature)] = outcome
    predictions = {}
    for oid in test_oids:
        obj = test_objects[oid]
        fake_obj = {"id": oid, "visible_features": obj.get("visible_features", {})}
        predictions[oid] = im_clone.predict_all_affordances(fake_obj)
    return predictions

oracle_k8_injections = [(oid, action, ALL_GT_AFFORDANCES[oid][feature])
                        for _, oid, action, feature, _, _ in oracle_k8_pairs_list]
oracle_k8_preds = inject_and_predict(im_base, oracle_k8_injections, test_oids)
oracle_k8_mb = compute_full_metrics(oracle_k8_preds, query_gt)["macro_query_balanced_accuracy"]
print(f"  Oracle k=8 macro_bal={oracle_k8_mb:.4f}")

# =============================================================================
# 8. Three Soft Prior Policy Variants
# =============================================================================
print("\n[7/9] Running soft prior policy variants...")

# =============================================================================
# 8a. Base observer-only policy (C15b observe phase, no probes)
# =============================================================================
class C15b_ObserveOnlyPhasePolicy(MiniMCPolicy):
    PHASE_OBSERVE = 1
    PHASE_DONE = 2

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE

    def _should_transition(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        if self._phase != self.PHASE_OBSERVE: return None
        if self._should_transition(view):
            self._phase = self.PHASE_DONE
            return None
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total; best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}


# =============================================================================
# 8b. Shared C15b observe-phase probe target selector (used by variants 2 and 3)
# =============================================================================

def _c15b_voi_for_pair(im, view, oid, action, cost_weight):
    """Compute C15b net VOI for a specific (oid, action) pair."""
    features = view.get_observed_features(oid)
    if features is None:
        return -999.0, None
    fake_obj = {"id": oid, "visible_features": features}
    probs = im.predict_all_affordances(fake_obj)
    current_utility = _compute_utility(probs)
    p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

    im_succ = im.clone()
    im_succ.incorporate_probe(oid, action, 1.0)
    utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
    im_fail = im.clone()
    im_fail.incorporate_probe(oid, action, 0.0)
    utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))

    e_post = p_success * utility_succ + (1 - p_success) * utility_fail
    total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
    norm_cost = total_cost / max(view.initial_budget, 0.001)
    net_voi = e_post - current_utility - cost_weight * norm_cost
    return net_voi, probs


def _sigmoid(x, scale=10.0):
    """Sigmoid to map VOI to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x * scale))))


# =============================================================================
# 8c. Variant 1: Soft Prior Only
# =============================================================================
class SoftPriorOnlyPolicy(MiniMCPolicy):
    """Rank probes by goal_soft_prior + exploration_floor + cost penalty only.
    No C15b VOI. No experience memory.
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._selected_prior_scores = []
        self._variant = "soft_prior_only"

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._selected_prior_scores = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            for action in MAIN_CANDIDATE_ACTIONS:
                prior = get_goal_soft_prior(features, action)
                # Soft prior only: score = prior + floor - cost_penalty
                score = prior + EXPLORATION_FLOOR - COST_WEIGHT * norm_cost
                if score > best_score:
                    best_score = score; best_oid = oid
        if best_score <= 0.0: return None
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_score = float('-inf')
        for action in MAIN_CANDIDATE_ACTIONS:
            prior = get_goal_soft_prior(features, action)
            norm_cost = view.probe_cost / max(view.initial_budget, 0.001)
            score = prior + EXPLORATION_FLOOR - COST_WEIGHT * norm_cost
            if score > best_score:
                best_score = score; best_action = action

        self._probed_oids.add(object_id)
        self._selected_prior_scores.append({
            "oid": object_id, "action": best_action,
            "prior": get_goal_soft_prior(features, best_action),
            "score": best_score,
        })
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})

            pre_utility = _compute_utility(pre_probs)
            post_utility = _compute_utility(post_probs)
            utility_delta = post_utility - pre_utility
            is_eff = utility_delta > 0.001
            is_zero = abs(utility_delta) <= 0.001
            is_harm = utility_delta < -0.001

            pre_query = _check_query(pre_probs, ACTION_TO_QUERY.get(action, ""))
            post_query = _check_query(post_probs, ACTION_TO_QUERY.get(action, ""))
            query_changed = pre_query != post_query

            if is_eff: effect = "effective"
            elif is_zero: effect = "zero"
            else: effect = "harmful"

            record_probe_event(features, action,
                              get_goal_soft_prior(features, action),
                              self._selected_prior_scores[-1]["score"] if self._selected_prior_scores else 0.0,
                              float(outcome), query_changed, effect,
                              float(view.probe_cost))

            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "is_zero": is_zero, "is_harmful": is_harm,
                "query_changed": query_changed,
            })
        else:
            self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}

    def get_probe_effects(self):
        return list(self._probe_effects)

    def get_selected_scores(self):
        return list(self._selected_prior_scores)


# =============================================================================
# 8d. Variant 2: C15b * Soft Prior (Multiplicative)
# =============================================================================
class C15bSoftPriorMulPolicy(MiniMCPolicy):
    """Multiply normalized C15b VOI score by goal_soft_prior.
    No experience memory.
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5, target_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._selected_scores = []
        self._variant = "c15b_soft_prior_mul"

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._selected_scores = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            for action in MAIN_CANDIDATE_ACTIONS:
                net_voi, _ = _c15b_voi_for_pair(self._im, view, oid, action, self._cost_weight)
                if net_voi is None: continue
                norm_voi = _sigmoid(net_voi, scale=10.0)
                prior = get_goal_soft_prior(features, action)
                # Multiplicative: normalized C15b * soft prior + floor - cost
                score = norm_voi * prior + EXPLORATION_FLOOR - self._cost_weight * norm_cost
                if score > best_score:
                    best_score = score; best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_score = float('-inf')
        norm_probe_cost = view.probe_cost / max(view.initial_budget, 0.001)
        action_details = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            net_voi, _ = _c15b_voi_for_pair(self._im, view, object_id, action, self._cost_weight)
            if net_voi is None: continue
            norm_voi = _sigmoid(net_voi, scale=10.0)
            prior = get_goal_soft_prior(features, action)
            score = norm_voi * prior + EXPLORATION_FLOOR - self._cost_weight * norm_probe_cost
            action_details[action] = {"net_voi": round(net_voi, 6), "norm_voi": round(norm_voi, 4),
                                      "prior": round(prior, 4), "final_score": round(score, 6)}
            if score > best_score:
                best_score = score; best_action = action

        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "score": best_score, "action_details": action_details,
        })
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})

            pre_utility = _compute_utility(pre_probs)
            post_utility = _compute_utility(post_probs)
            utility_delta = post_utility - pre_utility
            is_eff = utility_delta > 0.001
            is_zero = abs(utility_delta) <= 0.001
            is_harm = utility_delta < -0.001

            pre_query = _check_query(pre_probs, ACTION_TO_QUERY.get(action, ""))
            post_query = _check_query(post_probs, ACTION_TO_QUERY.get(action, ""))
            query_changed = pre_query != post_query

            if is_eff: effect = "effective"
            elif is_zero: effect = "zero"
            else: effect = "harmful"

            record_probe_event(features, action,
                              get_goal_soft_prior(features, action),
                              self._selected_scores[-1]["score"] if self._selected_scores else 0.0,
                              float(outcome), query_changed, effect,
                              float(view.probe_cost))

            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "utility_delta": round(utility_delta, 6),
                "is_effective": is_eff, "is_zero": is_zero, "is_harmful": is_harm,
                "query_changed": query_changed,
            })
        else:
            self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances(
                {"id": oid, "visible_features": features})
        return {"per_object": per_object}

    def get_probe_effects(self):
        return list(self._probe_effects)

    def get_selected_scores(self):
        return list(self._selected_scores)


# =============================================================================
# 8e. Variant 3: C15b * Soft Prior * Experience Memory
# =============================================================================
class C15bSoftPriorMemoryPolicy(C15bSoftPriorMulPolicy):
    """Same as variant 2, but also includes pattern_priority_memory adjustment.
    If no prior events exist for a pattern, this reduces to variant 2.
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5, target_probe_count=8):
        super().__init__(instance_memory, rng, cost_weight, target_probe_count)
        self._variant = "c15b_soft_prior_memory"

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            for action in MAIN_CANDIDATE_ACTIONS:
                net_voi, _ = _c15b_voi_for_pair(self._im, view, oid, action, self._cost_weight)
                if net_voi is None: continue
                norm_voi = _sigmoid(net_voi, scale=10.0)
                prior = get_goal_soft_prior(features, action)
                exp_adj = get_experience_adjustment(features, action)
                score = norm_voi * prior * exp_adj + EXPLORATION_FLOOR - self._cost_weight * norm_cost
                if score > best_score:
                    best_score = score; best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        best_action = None
        best_score = float('-inf')
        norm_probe_cost = view.probe_cost / max(view.initial_budget, 0.001)
        action_details = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            net_voi, _ = _c15b_voi_for_pair(self._im, view, object_id, action, self._cost_weight)
            if net_voi is None: continue
            norm_voi = _sigmoid(net_voi, scale=10.0)
            prior = get_goal_soft_prior(features, action)
            exp_adj = get_experience_adjustment(features, action)
            score = norm_voi * prior * exp_adj + EXPLORATION_FLOOR - self._cost_weight * norm_probe_cost
            action_details[action] = {"net_voi": round(net_voi, 6), "norm_voi": round(norm_voi, 4),
                                      "prior": round(prior, 4),
                                      "exp_adj": round(exp_adj, 4),
                                      "final_score": round(score, 6)}
            if score > best_score:
                best_score = score; best_action = action

        self._probed_oids.add(object_id)
        self._selected_scores.append({
            "oid": object_id, "action": best_action,
            "score": best_score, "action_details": action_details,
        })
        return True, best_action


# =============================================================================
# 9. Run All Variants
# =============================================================================
print("\n  Running Variant 1: soft_prior_only...")
sp_im = im_base.clone()
sp_policy = SoftPriorOnlyPolicy(sp_im, random.Random(SEED + 1000),
                                 target_probe_count=c15b_probed_n)
sp_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
sp_preds, sp_log, sp_obs, _, _ = run_policy(sp_policy, sp_env)
sp_metrics = compute_full_metrics(sp_preds["per_object"], query_gt)
sp_mb = sp_metrics["macro_query_balanced_accuracy"]
sp_probed = sum(1 for oid in test_oids if sp_obs.get_probe_results(oid))
sp_visited = len([oid for oid in test_oids if sp_obs.is_visited(oid)])
sp_effects = sp_policy.get_probe_effects()
sp_probe_pairs = [(e["oid"], e["action"]) for e in sp_effects]
sp_effective = sum(1 for e in sp_effects if e["is_effective"])
sp_zero = sum(1 for e in sp_effects if e["is_zero"])
sp_harmful = sum(1 for e in sp_effects if e["is_harmful"])
sp_overlap_oracle = len(set(sp_probe_pairs) & oracle_k8_pair_set)
sp_overlap_c15b = len(set(sp_probe_pairs) & c15b_probe_pair_set)
sp_delta = sp_mb - c15b_mb
sp_scores = sp_policy.get_selected_scores()

# Compute soft prior statistics for selected vs unselected
all_visited_oids = [oid for oid in test_oids if sp_obs.is_visited(oid)]
probed_oids_set = set(e["oid"] for e in sp_effects)
selected_priors = []
unselected_priors = []
for oid in all_visited_oids:
    features = sp_obs.get_observed_features(oid)
    if features is None: continue
    action_priors = get_per_query_goal_soft_prior(features)
    if oid in probed_oids_set:
        for a, p in action_priors.items():
            selected_priors.append(p)
    else:
        for a, p in action_priors.items():
            unselected_priors.append(p)

sp_avg_sel_prior = sum(selected_priors) / max(len(selected_priors), 1)
sp_avg_unsel_prior = sum(unselected_priors) / max(len(unselected_priors), 1)

# Check for hard exclusion
sp_any_zero = any(get_goal_soft_prior(sp_obs.get_observed_features(oid), action) == 0.0
                  for oid in all_visited_oids
                  for action in MAIN_CANDIDATE_ACTIONS
                  if sp_obs.get_observed_features(oid) is not None)

# Check: any candidate has nonzero priority after exploration floor
sp_low_prior_selected = sum(1 for s in sp_scores
                            if s.get("prior", EXPLORATION_FLOOR) <= EXPLORATION_FLOOR + 0.05)
sp_high_prior_selected = sum(1 for s in sp_scores
                             if s.get("prior", EXPLORATION_FLOOR) >= 0.7)

print(f"  soft_prior_only macro_bal={sp_mb:.4f}  visited={sp_visited}  probed={sp_probed}")
print(f"  soft_prior_only delta_vs_C15b={sp_delta:+.4f}")
print(f"  soft_prior_only effective={sp_effective}  zero={sp_zero}  harmful={sp_harmful}")
print(f"  soft_prior_only overlap_C15b={sp_overlap_c15b}  overlap_oracle_k8={sp_overlap_oracle}")
print(f"  soft_prior_only avg_sel_prior={sp_avg_sel_prior:.4f}  avg_unsel_prior={sp_avg_unsel_prior:.4f}")
print(f"  soft_prior_only low_prior_selected={sp_low_prior_selected}  high_prior_selected={sp_high_prior_selected}")
print(f"  soft_prior_only any_zero_prior_in_candidates={sp_any_zero}  hard_exclusion=false")

print("\n  Running Variant 2: c15b_soft_prior_mul...")
mul_im = im_base.clone()
mul_policy = C15bSoftPriorMulPolicy(mul_im, random.Random(SEED + 1100),
                                      cost_weight=COST_WEIGHT,
                                      target_probe_count=c15b_probed_n)
mul_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
mul_preds, mul_log, mul_obs, _, _ = run_policy(mul_policy, mul_env)
mul_metrics = compute_full_metrics(mul_preds["per_object"], query_gt)
mul_mb = mul_metrics["macro_query_balanced_accuracy"]
mul_probed = sum(1 for oid in test_oids if mul_obs.get_probe_results(oid))
mul_visited = len([oid for oid in test_oids if mul_obs.is_visited(oid)])
mul_effects = mul_policy.get_probe_effects()
mul_probe_pairs = [(e["oid"], e["action"]) for e in mul_effects]
mul_effective = sum(1 for e in mul_effects if e["is_effective"])
mul_zero = sum(1 for e in mul_effects if e["is_zero"])
mul_harmful = sum(1 for e in mul_effects if e["is_harmful"])
mul_overlap_oracle = len(set(mul_probe_pairs) & oracle_k8_pair_set)
mul_overlap_c15b = len(set(mul_probe_pairs) & c15b_probe_pair_set)
mul_delta = mul_mb - c15b_mb
mul_scores = mul_policy.get_selected_scores()

mul_probed_oids_set = set(e["oid"] for e in mul_effects)
mul_sel_priors = []
mul_unsel_priors = []
for oid in all_visited_oids:
    features = mul_obs.get_observed_features(oid)
    if features is None: continue
    action_priors = get_per_query_goal_soft_prior(features)
    if oid in mul_probed_oids_set:
        for a, p in action_priors.items():
            mul_sel_priors.append(p)
    else:
        for a, p in action_priors.items():
            mul_unsel_priors.append(p)

mul_avg_sel_prior = sum(mul_sel_priors) / max(len(mul_sel_priors), 1)
mul_avg_unsel_prior = sum(mul_unsel_priors) / max(len(mul_unsel_priors), 1)
mul_low_prior_selected = sum(1 for s in mul_scores
                              if s.get("action_details", {}).get(
                                  s.get("action", ""), {}).get("prior", EXPLORATION_FLOOR) <= EXPLORATION_FLOOR + 0.05)
mul_high_prior_selected = sum(1 for s in mul_scores
                               if s.get("action_details", {}).get(
                                   s.get("action", ""), {}).get("prior", EXPLORATION_FLOOR) >= 0.7)

print(f"  c15b_soft_prior_mul macro_bal={mul_mb:.4f}  visited={mul_visited}  probed={mul_probed}")
print(f"  c15b_soft_prior_mul delta_vs_C15b={mul_delta:+.4f}")
print(f"  c15b_soft_prior_mul effective={mul_effective}  zero={mul_zero}  harmful={mul_harmful}")
print(f"  c15b_soft_prior_mul overlap_C15b={mul_overlap_c15b}  overlap_oracle_k8={mul_overlap_oracle}")
print(f"  c15b_soft_prior_mul avg_sel_prior={mul_avg_sel_prior:.4f}  avg_unsel_prior={mul_avg_unsel_prior:.4f}")
print(f"  c15b_soft_prior_mul low_prior_selected={mul_low_prior_selected}  high_prior_selected={mul_high_prior_selected}")

# Update pattern_priority_memory from variant 2's probe events
print("\n  Updating pattern priority memory from variant 2 probe events...")
update_pattern_priority_memory()
print(f"  Pattern entries: {len(pattern_priority_memory)}")
for key, entry in sorted(pattern_priority_memory.items()):
    print(f"    {key}: support={entry['support_count']}  eff_rate={entry['effective_rate']:.2f}  "
          f"harm_rate={entry['harmful_rate']:.2f}  adj={entry['priority_adjustment']:.3f}  "
          f"conf={entry['confidence']:.3f}")

print("\n  Running Variant 3: c15b_soft_prior_memory...")
mem_im = im_base.clone()
# Clear probe event memory so variant 3 starts fresh but uses pattern_priority_memory
# from variant 2's probe events
# Actually, pattern_priority_memory is already populated from variant 2.
# But variant 3 starts with its own im (clean), so the pattern memory is pre-loaded
# with variant 2's experience. This is intentional - it tests whether the memory
# layer can improve over variant 2.
mem_policy = C15bSoftPriorMemoryPolicy(mem_im, random.Random(SEED + 1200),
                                         cost_weight=COST_WEIGHT,
                                         target_probe_count=c15b_probed_n)
mem_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
mem_preds, mem_log, mem_obs, _, _ = run_policy(mem_policy, mem_env)
mem_metrics = compute_full_metrics(mem_preds["per_object"], query_gt)
mem_mb = mem_metrics["macro_query_balanced_accuracy"]
mem_probed = sum(1 for oid in test_oids if mem_obs.get_probe_results(oid))
mem_visited = len([oid for oid in test_oids if mem_obs.is_visited(oid)])
mem_effects = mem_policy.get_probe_effects()
mem_probe_pairs = [(e["oid"], e["action"]) for e in mem_effects]
mem_effective = sum(1 for e in mem_effects if e["is_effective"])
mem_zero = sum(1 for e in mem_effects if e["is_zero"])
mem_harmful = sum(1 for e in mem_effects if e["is_harmful"])
mem_overlap_oracle = len(set(mem_probe_pairs) & oracle_k8_pair_set)
mem_overlap_c15b = len(set(mem_probe_pairs) & c15b_probe_pair_set)
mem_delta = mem_mb - c15b_mb
mem_scores = mem_policy.get_selected_scores()

mem_probed_oids_set = set(e["oid"] for e in mem_effects)
mem_sel_priors = []
mem_unsel_priors = []
for oid in all_visited_oids:
    features = mem_obs.get_observed_features(oid)
    if features is None: continue
    action_priors = get_per_query_goal_soft_prior(features)
    if oid in mem_probed_oids_set:
        for a, p in action_priors.items():
            mem_sel_priors.append(p)
    else:
        for a, p in action_priors.items():
            mem_unsel_priors.append(p)

mem_avg_sel_prior = sum(mem_sel_priors) / max(len(mem_sel_priors), 1)
mem_avg_unsel_prior = sum(mem_unsel_priors) / max(len(mem_unsel_priors), 1)
mem_low_prior_selected = sum(1 for s in mem_scores
                              if s.get("action_details", {}).get(
                                  s.get("action", ""), {}).get("prior", EXPLORATION_FLOOR) <= EXPLORATION_FLOOR + 0.05)
mem_high_prior_selected = sum(1 for s in mem_scores
                               if s.get("action_details", {}).get(
                                   s.get("action", ""), {}).get("prior", EXPLORATION_FLOOR) >= 0.7)

print(f"  c15b_soft_prior_memory macro_bal={mem_mb:.4f}  visited={mem_visited}  probed={mem_probed}")
print(f"  c15b_soft_prior_memory delta_vs_C15b={mem_delta:+.4f}")
print(f"  c15b_soft_prior_memory effective={mem_effective}  zero={mem_zero}  harmful={mem_harmful}")
print(f"  c15b_soft_prior_memory overlap_C15b={mem_overlap_c15b}  overlap_oracle_k8={mem_overlap_oracle}")
print(f"  c15b_soft_prior_memory avg_sel_prior={mem_avg_sel_prior:.4f}  avg_unsel_prior={mem_avg_unsel_prior:.4f}")
print(f"  c15b_soft_prior_memory low_prior_selected={mem_low_prior_selected}  high_prior_selected={mem_high_prior_selected}")

# Determine best soft prior variant
sp_variants = {
    "soft_prior_only": (sp_mb, sp_delta, sp_probed, sp_overlap_oracle, sp_overlap_c15b,
                         sp_effective, sp_harmful),
    "c15b_soft_prior_mul": (mul_mb, mul_delta, mul_probed, mul_overlap_oracle, mul_overlap_c15b,
                             mul_effective, mul_harmful),
    "c15b_soft_prior_memory": (mem_mb, mem_delta, mem_probed, mem_overlap_oracle, mem_overlap_c15b,
                                mem_effective, mem_harmful),
}
best_sp_name = max(sp_variants, key=lambda k: sp_variants[k][0])
(best_sp_mb, best_sp_delta, best_sp_probed, best_sp_overlap_oracle, best_sp_overlap_c15b,
 best_sp_eff, best_sp_harm) = sp_variants[best_sp_name]

print(f"\n  Best soft prior variant: {best_sp_name}")
print(f"  Best variant macro_bal={best_sp_mb:.4f}  delta_vs_C15b={best_sp_delta:+.4f}")
print(f"  Best variant overlap_oracle_k8={best_sp_overlap_oracle}  overlap_C15b={best_sp_overlap_c15b}")

# =============================================================================
# 10. Run Additional Baselines
# =============================================================================
print("\n[8/9] Running baselines...")

# C0b
c0b_im = im_base.clone()
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, random.Random(SEED + 500))
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_mb = compute_full_metrics(c0b_preds["per_object"], query_gt)["macro_query_balanced_accuracy"]
print(f"  C0b macro_bal={c0b_mb:.4f}")

# C18b (from 1J15)
class C18b_QueryRelevantFixedBudgetPolicy(MiniMCPolicy):
    PHASE_OBSERVE = 1; PHASE_PROBE = 2
    def __init__(self, instance_memory, rng, target_probe_count=8):
        self._im = instance_memory; self._rng = rng
        self._phase = self.PHASE_OBSERVE; self._probe_tracking = []
        self._pre_probe_entropies = {}; self._probed_oids = set()
        self._observe_pass_visit_count = 0; self._target_probe_count = target_probe_count

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE; self._probe_tracking = []
        self._pre_probe_entropies = {}; self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited: return True
        cheapest_observe = min(view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
        if not view.can_afford(cheapest_observe): return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve: return True
        if len(unvisited) <= 3: return True
        return False

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                return self._select_probe_target(view)
            best_oid = None; best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total; best_oid = oid
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count: return None
        best_oid = None; best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids: continue
            features = view.get_observed_features(oid)
            if features is None: continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost): continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            for action in MAIN_CANDIDATE_ACTIONS:
                p = probs.get(ACTION_TO_FEATURE[action], 0.5)
                score = 1.0 - max(p, 1.0 - max(0.001, min(0.999, p)))
                if score > best_score: best_score = score; best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE: return False, None
        if len(self._probed_oids) >= self._target_probe_count: return False, None
        features = view.get_observed_features(object_id)
        if features is None: return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        best_action = None; best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p = probs.get(ACTION_TO_FEATURE[action], 0.5)
            score = 1.0 - max(p, 1.0 - max(0.001, min(0.999, p)))
            if score > best_score: best_score = score; best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)
        self._probe_tracking.append({"oid": object_id, "action": action, "outcome": outcome})

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances({"id": oid, "visible_features": features})
        return {"per_object": per_object}

c18b_im = im_base.clone()
c18b_policy = C18b_QueryRelevantFixedBudgetPolicy(c18b_im, random.Random(SEED + 850),
                                                    target_probe_count=c15b_probed_n)
c18b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18b_preds, c18b_log, c18b_obs, _, _ = run_policy(c18b_policy, c18b_env)
c18b_mb = compute_full_metrics(c18b_preds["per_object"], query_gt)["macro_query_balanced_accuracy"]
print(f"  C18b macro_bal={c18b_mb:.4f}")

# Oracle
oracle_preds = C14a_TruthAnswerOracle(sim_gt).get_answer()
oracle_mb = compute_full_metrics(oracle_preds["per_object"], query_gt)["macro_query_balanced_accuracy"]
print(f"  C_oracle macro_bal={oracle_mb:.4f}")

# =============================================================================
# 11. Interpretation + Output
# =============================================================================
print("\n[9/9] Interpreting results...")

target_10pct = 0.6284
oracle_c0b_gap = oracle_mb - c0b_mb

best_improves_c15b = best_sp_mb > c15b_mb + 0.001
best_hits_10pct = best_sp_mb >= target_10pct

# Effect on C15b: does soft prior improve, preserve, or hurt?
if best_sp_delta > 0.001:
    effect_on_c15b = "improves"
elif best_sp_delta < -0.001:
    effect_on_c15b = "hurts"
else:
    effect_on_c15b = "preserves"

# soft_priority_only performance interpretation
sp_ok = sp_mb > c0b_mb  # At least beats observe-only

if best_improves_c15b:
    soft_goal_prior_supported = True
    if best_hits_10pct:
        interpretation = (
            f"Best soft prior variant ({best_sp_name}) reaches {best_sp_mb:.4f} "
            f"(+{best_sp_delta:+.4f} vs C15b), exceeding the 10% threshold ({target_10pct}). "
            f"Goal-conditioned soft prior is USEFUL as a probe-ranking modifier."
        )
    else:
        interpretation = (
            f"Best soft prior variant ({best_sp_name}) reaches {best_sp_mb:.4f} "
            f"(+{best_sp_delta:+.4f} vs C15b), improving over C15b but not reaching "
            f"10% threshold ({target_10pct}). Soft goal prior shows partial signal."
        )
    memory_layer_keep = mem_mb > mul_mb + 0.001
else:
    soft_goal_prior_supported = False
    memory_layer_keep = False
    if best_sp_delta > -0.005:
        interpretation = (
            f"Best soft prior variant ({best_sp_name}) reaches {best_sp_mb:.4f} "
            f"({best_sp_delta:+.4f} vs C15b). Soft prior does not improve C15b but "
            f"preserves it. Keep prior as coarse candidate-priority layer only."
        )
    else:
        interpretation = (
            f"Best soft prior variant ({best_sp_name}) reaches {best_sp_mb:.4f} "
            f"({best_sp_delta:+.4f} vs C15b). Soft prior HURTS C15b. "
            f"Reduce prior weight and use only as diagnostic."
        )

# Determine next route
if soft_goal_prior_supported and best_hits_10pct:
    next_recommended_route = "small_multiseed_goal_conditioned_soft_prior"
elif soft_goal_prior_supported:
    next_recommended_route = "tune_soft_prior_weights_and_retry"
else:
    next_recommended_route = "environment_or_teacher_representation_redesign"

print(f"  Best variant: {best_sp_name}")
print(f"  Best macro_bal: {best_sp_mb:.4f}  delta_vs_C15b: {best_sp_delta:+.4f}")
print(f"  C15b effect: {effect_on_c15b}")
print(f"  soft_goal_prior_supported: {soft_goal_prior_supported}")
print(f"  memory_layer_keep: {memory_layer_keep}")
print(f"  next_recommended_route: {next_recommended_route}")

# =============================================================================
# 12. Save outputs
# =============================================================================
os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

output = {
    "block_id": "1J18",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Goal-Conditioned Soft Probe Prior Memory. Adds minimal goal-conditioned visible-type->priority mapping.",
    "design_validation": {
        "no_environment_change": True,
        "no_budget_cost_change": True,
        "no_multi_seed": True,
        "no_hidden_subtype_category_in_deployment": True,
        "no_oracle_outcomes_in_deployment": True,
        "no_hard_exclusion": True,
        "no_cross_test_object_propagation": True,
        "exploration_floor": EXPLORATION_FLOOR,
    },
    "default_goal_prior": {
        "type_families": list(TYPE_FAMILIES.keys()),
        "family_features": {fam: sorted(list(fs)) for fam, fs in TYPE_FAMILIES.items()},
        "prior_table": [
            {"type_family": fam, "query": qname, "action": action, "prior": prior}
            for (fam, qname, action), prior in sorted(DEFAULT_GOAL_PRIOR.items())
        ],
        "min_prior": min_prior,
        "exploration_floor": EXPLORATION_FLOOR,
    },
    "prior_sanity_checks": {
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "all_pass": checks_failed == 0,
    },
    "variant_results": {
        "soft_prior_only": {
            "macro_bal": sp_mb,
            "delta_vs_C15b": round(sp_delta, 6),
            "visited": sp_visited,
            "probed": sp_probed,
            "probe_pairs": str(sp_probe_pairs),
            "effective": sp_effective,
            "zero": sp_zero,
            "harmful": sp_harmful,
            "overlap_C15b": sp_overlap_c15b,
            "overlap_oracle_k8": sp_overlap_oracle,
            "avg_selected_prior": round(sp_avg_sel_prior, 4),
            "avg_unselected_prior": round(sp_avg_unsel_prior, 4),
            "low_prior_selected": sp_low_prior_selected,
            "high_prior_selected": sp_high_prior_selected,
            "any_zero_prior": sp_any_zero,
            "per_query": sp_metrics["per_query"],
        },
        "c15b_soft_prior_mul": {
            "macro_bal": mul_mb,
            "delta_vs_C15b": round(mul_delta, 6),
            "visited": mul_visited,
            "probed": mul_probed,
            "probe_pairs": str(mul_probe_pairs),
            "effective": mul_effective,
            "zero": mul_zero,
            "harmful": mul_harmful,
            "overlap_C15b": mul_overlap_c15b,
            "overlap_oracle_k8": mul_overlap_oracle,
            "avg_selected_prior": round(mul_avg_sel_prior, 4),
            "avg_unselected_prior": round(mul_avg_unsel_prior, 4),
            "low_prior_selected": mul_low_prior_selected,
            "high_prior_selected": mul_high_prior_selected,
            "per_query": mul_metrics["per_query"],
            "selected_scores": [
                {"oid": s["oid"], "action": s["action"], "score": s["score"],
                 "details": {a: {"norm_voi": d["norm_voi"], "prior": d["prior"],
                                 "final_score": d["final_score"]}
                            for a, d in s.get("action_details", {}).items()}}
                for s in mul_scores
            ],
        },
        "c15b_soft_prior_memory": {
            "macro_bal": mem_mb,
            "delta_vs_C15b": round(mem_delta, 6),
            "visited": mem_visited,
            "probed": mem_probed,
            "probe_pairs": str(mem_probe_pairs),
            "effective": mem_effective,
            "zero": mem_zero,
            "harmful": mem_harmful,
            "overlap_C15b": mem_overlap_c15b,
            "overlap_oracle_k8": mem_overlap_oracle,
            "avg_selected_prior": round(mem_avg_sel_prior, 4),
            "avg_unselected_prior": round(mem_avg_unsel_prior, 4),
            "low_prior_selected": mem_low_prior_selected,
            "high_prior_selected": mem_high_prior_selected,
            "per_query": mem_metrics["per_query"],
            "selected_scores": [
                {"oid": s["oid"], "action": s["action"], "score": s["score"],
                 "details": {a: {"norm_voi": d["norm_voi"], "prior": d["prior"],
                                 "exp_adj": d.get("exp_adj", 1.0),
                                 "final_score": d["final_score"]}
                            for a, d in s.get("action_details", {}).items()}}
                for s in mem_scores
            ],
        },
    },
    "pattern_priority_memory": {
        str(k): v for k, v in pattern_priority_memory.items()
    },
    "probe_event_memory_sample": probe_event_memory[:20],  # first 20 events
    "baselines": {
        "C0b": {"macro_bal": c0b_mb},
        "C15b": {"macro_bal": c15b_mb, "probed": c15b_probed_n,
                 "probe_pairs": str(c15b_probe_pairs)},
        "C18b": {"macro_bal": c18b_mb},
        "C19a_ref": {"macro_bal": 0.6067, "note": "Best C19 from 1J16"},
        "crossfit_MLP_ref": {"macro_bal": 0.5942, "note": "From 1J17_patch"},
        "oracle_k8": {"macro_bal": oracle_k8_mb, "pairs": str([(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]),
                      "note": "Non-deployable reference"},
        "oracle_full": {"macro_bal": oracle_mb},
    },
    "best_soft_prior_variant": best_sp_name,
    "gap_capture": {
        "C0b_base": c0b_mb,
        "oracle_ceiling": oracle_mb,
        "oracle_c0b_gap": oracle_c0b_gap,
        "C15b_pct": round((c15b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
        "soft_prior_only_pct": round((sp_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
        "c15b_soft_prior_mul_pct": round((mul_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
        "c15b_soft_prior_memory_pct": round((mem_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
        "oracle_k8_pct": round((oracle_k8_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    },
    "soft_prior_selection_analysis": {
        "soft_prior_only": {
            "avg_sel_prior": round(sp_avg_sel_prior, 4),
            "avg_unsel_prior": round(sp_avg_unsel_prior, 4),
            "low_prior_selected_count": sp_low_prior_selected,
            "high_prior_selected_count": sp_high_prior_selected,
        },
        "c15b_soft_prior_mul": {
            "avg_sel_prior": round(mul_avg_sel_prior, 4),
            "avg_unsel_prior": round(mul_avg_unsel_prior, 4),
            "low_prior_selected_count": mul_low_prior_selected,
            "high_prior_selected_count": mul_high_prior_selected,
        },
        "c15b_soft_prior_memory": {
            "avg_sel_prior": round(mem_avg_sel_prior, 4),
            "avg_unsel_prior": round(mem_avg_unsel_prior, 4),
            "low_prior_selected_count": mem_low_prior_selected,
            "high_prior_selected_count": mem_high_prior_selected,
        },
    },
    "per_query_comparison": {},
    "interpretation": {
        "soft_goal_prior_supported": soft_goal_prior_supported,
        "memory_layer_keep": memory_layer_keep,
        "effect_on_C15b": effect_on_c15b,
        "best_variant": best_sp_name,
        "best_macro_bal": best_sp_mb,
        "best_delta_vs_C15b": round(best_sp_delta, 6),
        "best_overlap_oracle_k8": best_sp_overlap_oracle,
        "best_overlap_C15b": best_sp_overlap_c15b,
        "best_effective": best_sp_eff,
        "best_harmful": best_sp_harm,
        "soft_prior_only_acceptable": sp_ok,
        "hard_exclusion_used": False,
        "candidate_ready_for_small_multiseed": best_improves_c15b and best_hits_10pct,
        "ready_for_multiseed": False,
        "next_recommended_route": next_recommended_route,
        "text": interpretation,
    },
}

# Per-query comparison
for qname in QUERY_NAMES:
    output["per_query_comparison"][qname] = {
        "C15b_bacc": c15b_metrics["per_query"][qname]["balanced_accuracy"],
        "soft_prior_only_bacc": sp_metrics["per_query"][qname]["balanced_accuracy"],
        "c15b_soft_prior_mul_bacc": mul_metrics["per_query"][qname]["balanced_accuracy"],
        "c15b_soft_prior_memory_bacc": mem_metrics["per_query"][qname]["balanced_accuracy"],
        "soft_prior_only_delta": round(
            sp_metrics["per_query"][qname]["balanced_accuracy"] -
            c15b_metrics["per_query"][qname]["balanced_accuracy"], 6),
        "c15b_soft_prior_mul_delta": round(
            mul_metrics["per_query"][qname]["balanced_accuracy"] -
            c15b_metrics["per_query"][qname]["balanced_accuracy"], 6),
        "c15b_soft_prior_memory_delta": round(
            mem_metrics["per_query"][qname]["balanced_accuracy"] -
            c15b_metrics["per_query"][qname]["balanced_accuracy"], 6),
    }

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j18_goal_conditioned_soft_prior_memory_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

# =============================================================================
# 13. Protocol MD
# =============================================================================
md_path = os.path.join(CURRENT_DIR, "protocols",
                       "block1j18_goal_conditioned_soft_prior_memory_c4_seed101.md")
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

# Build probe selection action analysis
sp_actions = dict(Counter(e["action"] for e in sp_effects))
mul_actions = dict(Counter(e["action"] for e in mul_effects))
mem_actions = dict(Counter(e["action"] for e in mem_effects))

md = f"""# Block 1J18 — Goal-Conditioned Soft Probe Prior Memory

## 1. Objective

Add a minimal goal-conditioned soft prior memory layer:
visible attributes should raise or lower probe priority under a given goal,
without hard exclusion. Never use hidden subtype/category as input.
Never use oracle outcomes as deployment input.

## 2. Setup

| Parameter | Value |
|-----------|-------|
| Condition | C4_instance_subtype_cued_v1 |
| Seed | {SEED} |
| Budget | {BUDGET} |
| Cost Weight | {COST_WEIGHT} |
| Probe Cost | {config.PROBE_COST} |
| Observe Cost | {config.OBSERVE_COST} |
| Reach Cost/unit | {config.REACH_COST_PER_UNIT} |
| IOM k | {config.INSTANCE_K} |
| Exploration Floor | {EXPLORATION_FLOOR} |
| Type Families | {len(TYPE_FAMILIES)} ({', '.join(TYPE_FAMILIES.keys())}) |
| Prior Sanity Checks | {checks_passed}/{checks_passed + checks_failed} passed |

## 3. Default Goal Prior Table

Visible type families are detected from characteristic features:

| Family | Characteristic Features |
|--------|------------------------|
"""
for fam, feats in TYPE_FAMILIES.items():
    md += f"| {fam} | {', '.join(sorted(feats))} |\n"

md += f"""
### Prior Values (type_family x goal x action)

| Family | Query | Action | Prior |
|--------|-------|--------|-------|
"""
for (fam, qname, action), prior in sorted(DEFAULT_GOAL_PRIOR.items()):
    md += f"| {fam} | {qname} | {action} | {prior:.2f} |\n"

md += f"""
**Min prior: {min_prior:.2f}** (never zero — no hard exclusion)

## 4. C15b Reference

| Metric | Value |
|--------|-------|
| Macro BAcc | {c15b_mb:.4f} |
| Visited | {c15b_visited_n} |
| Probed | {c15b_probed_n} |
| Probe pairs | {c15b_probe_pairs} |

## 5. Variant Results

### 5.1 Aggregate Metrics

| Policy | Macro BAcc | Pos Rec | Neg Rec | Visited | Probed | Delta vs C15b |
|--------|-----------|---------|---------|---------|--------|---------------|
| C0b (observe only) | {c0b_mb:.4f} | — | — | — | 0 | — |
| C15b (VOI reserve) | {c15b_mb:.4f} | {c15b_metrics['mean_positive_recall']:.4f} | {c15b_metrics['mean_negative_recall']:.4f} | {c15b_visited_n} | {c15b_probed_n} | 0 |
| C18b (query-rel) | {c18b_mb:.4f} | — | — | — | — | {c18b_mb - c15b_mb:+.4f} |
| C19a (best from 1J16) | 0.6067 | — | — | — | 5 | -0.0075 |
| crossfit MLP (1J17p) | 0.5942 | — | — | — | — | -0.0200 |
| **soft_prior_only** | **{sp_mb:.4f}** | {sp_metrics['mean_positive_recall']:.4f} | {sp_metrics['mean_negative_recall']:.4f} | {sp_visited} | {sp_probed} | **{sp_delta:+.4f}** |
| **c15b_soft_prior_mul** | **{mul_mb:.4f}** | {mul_metrics['mean_positive_recall']:.4f} | {mul_metrics['mean_negative_recall']:.4f} | {mul_visited} | {mul_probed} | **{mul_delta:+.4f}** |
| **c15b_soft_prior_memory** | **{mem_mb:.4f}** | {mem_metrics['mean_positive_recall']:.4f} | {mem_metrics['mean_negative_recall']:.4f} | {mem_visited} | {mem_probed} | **{mem_delta:+.4f}** |
| Oracle k=8 | {oracle_k8_mb:.4f} | — | — | — | 8 | {oracle_k8_mb - c15b_mb:+.4f} |
| Oracle (full) | {oracle_mb:.4f} | — | — | — | — | — |

### 5.2 Probe Selection

| Metric | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|--------|----------------|---------------------|------------------------|
| Probe count | {sp_probed} | {mul_probed} | {mem_probed} |
| Effective | {sp_effective} | {mul_effective} | {mem_effective} |
| Zero gain | {sp_zero} | {mul_zero} | {mem_zero} |
| Harmful | {sp_harmful} | {mul_harmful} | {mem_harmful} |
| Avg sel prior | {sp_avg_sel_prior:.4f} | {mul_avg_sel_prior:.4f} | {mem_avg_sel_prior:.4f} |
| Avg unsel prior | {sp_avg_unsel_prior:.4f} | {mul_avg_unsel_prior:.4f} | {mem_avg_unsel_prior:.4f} |
| Low prior selected | {sp_low_prior_selected} | {mul_low_prior_selected} | {mem_low_prior_selected} |
| High prior selected | {sp_high_prior_selected} | {mul_high_prior_selected} | {mem_high_prior_selected} |
| Selected actions | {sp_actions} | {mul_actions} | {mem_actions} |

### 5.3 Overlap Analysis

| Comparison | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|------------|----------------|---------------------|------------------------|
| Overlap C15b | {sp_overlap_c15b} | {mul_overlap_c15b} | {mem_overlap_c15b} |
| Overlap oracle k=8 | {sp_overlap_oracle} | {mul_overlap_oracle} | {mem_overlap_oracle} |

| Reference | Pairs |
|-----------|-------|
| C15b | {c15b_probe_pairs} |
| Oracle k=8 | {[(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]} |
| soft_prior_only | {sp_probe_pairs} |
| c15b_soft_prior_mul | {mul_probe_pairs} |
| c15b_soft_prior_memory | {mem_probe_pairs} |

### 5.4 Per-Query Breakdown

| Query | C15b BAcc | soft_prior_only | c15b_soft_prior_mul | c15b_soft_prior_memory |
|-------|----------|----------------|---------------------|------------------------|
"""
for qname in QUERY_NAMES:
    c15b_b = c15b_metrics["per_query"][qname]["balanced_accuracy"]
    sp_b = sp_metrics["per_query"][qname]["balanced_accuracy"]
    mul_b = mul_metrics["per_query"][qname]["balanced_accuracy"]
    mem_b = mem_metrics["per_query"][qname]["balanced_accuracy"]
    md += f"| {qname} | {c15b_b:.4f} | {sp_b:.4f} ({sp_b - c15b_b:+.4f}) | {mul_b:.4f} ({mul_b - c15b_b:+.4f}) | {mem_b:.4f} ({mem_b - c15b_b:+.4f}) |\n"

md += f"""
## 6. Gap Capture

| Policy | Gap Capture % |
|--------|--------------|
| C15b | {round((c15b_mb - c0b_mb) / oracle_c0b_gap * 100, 1):.1f}% |
| soft_prior_only | {round((sp_mb - c0b_mb) / oracle_c0b_gap * 100, 1):.1f}% |
| c15b_soft_prior_mul | {round((mul_mb - c0b_mb) / oracle_c0b_gap * 100, 1):.1f}% |
| c15b_soft_prior_memory | {round((mem_mb - c0b_mb) / oracle_c0b_gap * 100, 1):.1f}% |
| Oracle k=8 | {round((oracle_k8_mb - c0b_mb) / oracle_c0b_gap * 100, 1):.1f}% |

## 7. Pattern Priority Memory

After variant 2 probes, the pattern_priority_memory contains:

| Pattern | Support | Eff Rate | Harm Rate | Adj | Conf |
|---------|---------|----------|-----------|-----|------|
"""
for key, entry in sorted(pattern_priority_memory.items()):
    md += (f"| {key} | {entry['support_count']} | {entry['effective_rate']:.2f} | "
           f"{entry['harmful_rate']:.2f} | {entry['priority_adjustment']:.3f} | "
           f"{entry['confidence']:.3f} |\n")

if not pattern_priority_memory:
    md += "| (none) | — | — | — | — | — |\n"

md += f"""
## 8. Design Compliance

| Check | Status |
|-------|--------|
| No environment change | [OK] |
| No budget/cost change | [OK] |
| Single seed only | [OK] |
| No hidden subtype/category in deployment | [OK] |
| No oracle outcomes in deployment | [OK] |
| No hard exclusion (min prior = {min_prior:.2f} > 0) | [OK] |
| Exploration floor > 0 | [OK] |
| No same-style learned critic | [OK] |
| C15b observe phase unchanged | [OK] |

## 9. Interpretation

{interpretation}

### Tiered Rules

- **If c15b_soft_prior_mul > C15b**: soft goal prior is useful as a probe-ranking modifier.
  -> **Result**: {"SUPPORTED" if mul_delta > 0.001 else "NOT SUPPORTED"} (delta={mul_delta:+.4f})
- **If macro is similar to C15b but probes are better targeted**:
  keep prior as a coarse candidate-priority layer.
  -> **Result**: effect_on_C15b={effect_on_c15b}
- **If soft prior hurts C15b**:
  reduce prior weight and use it only as diagnostic.
  -> **Result**: hurts={effect_on_c15b == 'hurts'}
- **If soft_prior_only performs poorly**:
  acceptable; prior is not expected to replace C15b, only to guide it.
  -> **Result**: soft_prior_only_macro_bal={sp_mb:.4f} (beats C0b={sp_ok})

## 10. Summary

```
[block_done]
block_id=1J18
c15b_macro_bal={c15b_mb:.4f}
best_soft_prior_variant={best_sp_name}
best_soft_prior_macro_bal={best_sp_mb:.4f}
best_soft_prior_delta_vs_c15b={best_sp_delta:+.4f}
best_soft_prior_probe_count={best_sp_probed}
best_soft_prior_overlap_with_c15b={best_sp_overlap_c15b}
best_soft_prior_overlap_with_oracle_k8={best_sp_overlap_oracle}
best_soft_prior_effective_probe_count={best_sp_eff}
best_soft_prior_harmful_probe_count={best_sp_harm}
hard_exclusion_used=false
soft_goal_prior_supported={'true' if soft_goal_prior_supported else 'false'}
memory_layer_keep={'true' if memory_layer_keep else 'false'}
next_recommended_route={next_recommended_route}
candidate_ready_for_small_multiseed={'true' if (best_improves_c15b and best_hits_10pct) else 'false'}
ready_for_multiseed=false
```
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"  Saved: {md_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J18")
print(f"c15b_macro_bal={c15b_mb:.4f}")
print(f"best_soft_prior_variant={best_sp_name}")
print(f"best_soft_prior_macro_bal={best_sp_mb:.4f}")
print(f"best_soft_prior_delta_vs_c15b={best_sp_delta:+.4f}")
print(f"best_soft_prior_probe_count={best_sp_probed}")
print(f"best_soft_prior_overlap_with_c15b={best_sp_overlap_c15b}")
print(f"best_soft_prior_overlap_with_oracle_k8={best_sp_overlap_oracle}")
print(f"best_soft_prior_effective_probe_count={best_sp_eff}")
print(f"best_soft_prior_harmful_probe_count={best_sp_harm}")
print(f"hard_exclusion_used=false")
print(f"soft_goal_prior_supported={'true' if soft_goal_prior_supported else 'false'}")
print(f"memory_layer_keep={'true' if memory_layer_keep else 'false'}")
print(f"next_recommended_route={next_recommended_route}")
print(f"candidate_ready_for_small_multiseed={'true' if (best_improves_c15b and best_hits_10pct) else 'false'}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
