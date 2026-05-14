"""
Block 1J15 -- Query-Level VOI Probe Selection Diagnostic.

Three C18 variants testing query-relevant direct probe ranking:

  C18a_local_query_confidence_VOI — ablation:
    local query-confidence proxy, VOI-gated with cost penalty.
    Tests whether query-confidence framing alone differs from C15b.

  C18b_query_relevant_fixed_probe_budget — PRIMARY:
    query-relevant ranking, fixed k = C15b probe count, no cost threshold.
    Isolates probe selection quality from stop-rule/cost-calibration issues.

  C18c_query_relevant_netVOI:
    same ranking as C18b, but VOI-gated with cost penalty.
    Tests whether the deployable stop rule works.

Architecture note: IOM has no cross-test-object propagation.
A probe on test object A only affects predictions for A
(via _direct_outcomes). Therefore global recomputation is
equivalent to local direct-effect scoring.

Single seed (101) only. No multi-seed.
"""
import os, sys, json, copy, random, time, math
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
    C14a_TruthAnswerOracle, C13_InstanceVOIPolicy,
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
print("Block 1J15 -- Query-Level VOI Probe Selection Diagnostic")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print(f"  no_cross_test_object_propagation=true")
print(f"  global_recompute_equivalent_to_local=true")
print("=" * 60)

# =============================================================================
# 1. Training + IOM building
# =============================================================================
print("\n[1/7] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED)
posterior_visible_features = list(base_learner.visible_feature_names)
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
print("\n[2/7] Generating test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

QUERY_NAMES = sorted(_TASK_QUERIES.keys())
N_OBJECTS = len(test_oids)

# Action-to-query mapping (which query each action is relevant to)
ACTION_TO_QUERY = {
    "craft_plank": "need_planks",
    "eat": "need_food",
    "use_as_tool": "need_tool",
    "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone",
    "mine_with_pickaxe": "need_stone",
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
        "per_query_positive_recall": {q: per_query[q]["positive_recall"] for q in per_query},
        "per_query_negative_recall": {q: per_query[q]["negative_recall"] for q in per_query},
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
# 4. Query-Relevant Probe Scoring
# =============================================================================

def compute_query_relevant_gain(probs, action):
    """Expected gain in query decision confidence from probing (object, action).

    The gain is computed as the expected improvement in the query decision
    confidence for the SINGLE query that this action is relevant to.

    For action probing feature F relevant to query Q:
      - p = current prediction for feature F
      - After probing: p becomes 0 or 1, confidence becomes 1.0
      - Expected post-confidence = 1.0
      - Current confidence = max(p, 1-p)
      - Expected gain = 1.0 - max(p, 1-p) = min(p, 1-p)

    This is the expected reduction in query decision uncertainty
    from resolving this specific (object, feature) pair.
    """
    feature = ACTION_TO_FEATURE[action]
    p = probs.get(feature, 0.5)
    p = max(0.001, min(0.999, p))
    return 1.0 - max(p, 1.0 - p)  # = min(p, 1-p)


def compute_query_relevant_score(probs, action):
    """Query-relevant score for a candidate (object, action) probe.

    Uses the expected gain in query decision confidence.
    This is a local direct-effect score — it only considers the probed
    object's prediction change, since the IOM has no cross-test-object
    propagation.
    """
    return compute_query_relevant_gain(probs, action)


# =============================================================================
# 5. C15b Inline Policy (baseline)
# =============================================================================
print("\n[3/7] Defining C15b, C18a, C18b, C18c policies...")

class C15b_ProbeReservePolicy(MiniMCPolicy):
    """C15b: two-pass VOI with 25% budget reserve."""
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            affordable = view.can_afford(total_cost)
            features = view.get_observed_features(oid)
            if features is None:
                continue
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
                if net_voi > best_action_voi:
                    best_action_voi = net_voi
            if affordable and best_action_voi > best_score:
                best_score = best_action_voi
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
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
                best_net_voi = net_voi
                best_action = action
        should_probe = best_net_voi > 0
        if should_probe:
            self._probed_oids.add(object_id)
        return should_probe, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_probs": dict(pre_probs), "post_probs": dict(post_probs),
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)


# =============================================================================
# 6. C18a — Local Query Confidence VOI (ablation)
# =============================================================================
class C18a_LocalQueryConfidenceVOIPolicy(MiniMCPolicy):
    """C18a: Local query-confidence VOI, VOI-gated with cost penalty.

    Ablation — replaces C15b's affordance utility with per-object
    query-confidence scoring. Uses uniform query weights.
    VOI-gated (same stop rule as C15b).
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            affordable = view.can_afford(total_cost)
            features = view.get_observed_features(oid)
            if features is None:
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            norm_cost = total_cost / max(view.initial_budget, 0.001)

            best_action_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                # Expected gain = min(p, 1-p) for the feature this action probes
                # This is identical to C15b's per-feature utility gain, just
                # framed as query-relevant rather than affordance-utility.
                expected_gain = compute_query_relevant_gain(probs, action)
                # Scale similarly to C15b: divide by n_features for comparability
                scaled_gain = expected_gain / len(CORE_ACTION_FEATURES)
                net_voi = scaled_gain - self._cost_weight * norm_cost
                if net_voi > best_action_voi:
                    best_action_voi = net_voi

            if affordable and best_action_voi > best_score:
                best_score = best_action_voi
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        norm_probe_cost = self._normalized_probe_cost(view)

        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            expected_gain = compute_query_relevant_gain(probs, action)
            scaled_gain = expected_gain / len(CORE_ACTION_FEATURES)
            net_voi = scaled_gain - self._cost_weight * norm_probe_cost
            if net_voi > best_net_voi:
                best_net_voi = net_voi
                best_action = action

        should_probe = best_net_voi > 0
        if should_probe:
            self._probed_oids.add(object_id)
        return should_probe, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_probs": dict(pre_probs), "post_probs": dict(post_probs),
            })
            pre_gain = compute_query_relevant_gain(pre_probs, action)
            post_gain = compute_query_relevant_gain(post_probs, action)
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_query_gain": round(pre_gain, 6),
                "post_query_gain": round(post_gain, 6),
                "delta": round(post_gain - pre_gain, 6),
                "is_zero_gain": abs(post_gain - pre_gain) < 0.0001,
                "is_harmful": post_gain < pre_gain - 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


# =============================================================================
# 7. C18b — Query-Relevant Fixed Probe Budget (PRIMARY)
# =============================================================================
class C18b_QueryRelevantFixedBudgetPolicy(MiniMCPolicy):
    """C18b: Query-relevant ranking, fixed probe budget.

    Primary diagnostic. Keeps C15b observe phase. In probe phase:
    1. Rank all visited+unprobed objects by query-relevant score
    2. Probe top k objects (where k = C15b's observed probe count, ~8)
    3. No cost threshold for probe decision — isolates selection quality
       from stop-rule/cost-calibration issues.

    For each object, the best action is selected by query-relevant gain.
    Objects are ranked by their best action's gain. Top k are probed.
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5, fixed_probe_count=8):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._fixed_probe_count = fixed_probe_count
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._probe_queue = []  # pre-computed ranked list

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._probe_queue = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
        return False

    def _build_probe_queue(self, view):
        """Rank all visited+unprobed objects by query-relevant score.

        Returns list of (oid, action, score), sorted by score descending.
        """
        candidates = []
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)

            best_action = None
            best_score = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                score = compute_query_relevant_score(probs, action)
                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is not None:
                candidates.append((oid, best_action, best_score, probs))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if self._phase == self.PHASE_OBSERVE:
            if self._should_transition_to_probe_phase(view):
                self._phase = self.PHASE_PROBE
                self._observe_pass_visit_count = sum(
                    1 for oid in view.get_all_object_ids() if view.is_visited(oid))
                # Build probe queue once at phase transition
                self._probe_queue = self._build_probe_queue(view)
                return self._select_probe_target(view)
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        # Return next object from probe queue (up to fixed_probe_count)
        probes_done = len(self._probed_oids)
        if probes_done >= self._fixed_probe_count:
            return None

        # Find the highest-ranked candidate that is still affordable
        for oid, action, score, probs in self._probe_queue:
            if oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if view.can_afford(total_cost):
                self._visit_order.append(("probe", oid))
                return oid

        return None

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None

        # Always probe if in probe phase and object is in the top-k queue
        probes_done = len(self._probed_oids)
        if probes_done >= self._fixed_probe_count:
            return False, None

        features = view.get_observed_features(object_id)
        if features is None:
            return False, None

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)

        # Find best action by query-relevant score
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            score = compute_query_relevant_score(probs, action)
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is not None:
            self._probed_oids.add(object_id)
            return True, best_action
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_probs": dict(pre_probs), "post_probs": dict(post_probs),
            })
            pre_score = compute_query_relevant_score(pre_probs, action)
            post_score = compute_query_relevant_score(post_probs, action)
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_query_score": round(pre_score, 6),
                "post_query_score": round(post_score, 6),
                "delta": round(post_score - pre_score, 6),
                "is_zero_gain": abs(post_score - pre_score) < 0.0001,
                "is_harmful": post_score < pre_score - 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


# =============================================================================
# 8. C18c — Query-Relevant Net VOI
# =============================================================================
class C18c_QueryRelevantNetVOIPolicy(MiniMCPolicy):
    """C18c: Query-relevant ranking with VOI gate + cost penalty.

    Same ranking score as C18b (query-relevant gain), but uses
    positive net VOI threshold with cost penalty (like C15b).
    Tests whether the deployable stop rule works with query-relevant scoring.
    """
    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited)
        if not view.can_afford(cheapest_observe):
            return True
        reserve = 0.25 * view.initial_budget
        if view.budget_remaining - cheapest_observe < reserve:
            return True
        if len(unvisited) <= 3:
            return True
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
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        best_oid = None
        best_score = float('-inf')
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            affordable = view.can_afford(total_cost)
            features = view.get_observed_features(oid)
            if features is None:
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            norm_cost = total_cost / max(view.initial_budget, 0.001)

            best_action_voi = 0.0
            for action in MAIN_CANDIDATE_ACTIONS:
                expected_gain = compute_query_relevant_gain(probs, action)
                net_voi = expected_gain - self._cost_weight * norm_cost
                if net_voi > best_action_voi:
                    best_action_voi = net_voi

            if affordable and best_action_voi > best_score:
                best_score = best_action_voi
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        norm_probe_cost = self._normalized_probe_cost(view)

        best_action = None
        best_net_voi = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            expected_gain = compute_query_relevant_gain(probs, action)
            net_voi = expected_gain - self._cost_weight * norm_probe_cost
            if net_voi > best_net_voi:
                best_net_voi = net_voi
                best_action = action

        should_probe = best_net_voi > 0
        if should_probe:
            self._probed_oids.add(object_id)
        return should_probe, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_probs": dict(pre_probs), "post_probs": dict(post_probs),
            })
            pre_score = compute_query_relevant_score(pre_probs, action)
            post_score = compute_query_relevant_score(post_probs, action)
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_query_score": round(pre_score, 6),
                "post_query_score": round(post_score, 6),
                "delta": round(post_score - pre_score, 6),
                "is_zero_gain": abs(post_score - pre_score) < 0.0001,
                "is_harmful": post_score < pre_score - 0.0001,
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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


# =============================================================================
# 9. Run Baselines + C18a + C18b + C18c
# =============================================================================
print("\n[4/7] Running baselines...")
results = {}

# --- all_false ---
all_false_preds = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
results["all_false"] = {
    "predictions": all_false_preds,
    "metrics": compute_full_metrics(all_false_preds, query_gt),
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
    "policy_type": "floor",
}
print(f"  all_false macro_bal={results['all_false']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C_minus_prior_only ---
c_minus_preds = {oid: {f: 0.5 for f in CORE_ACTION_FEATURES}
                 for oid in test_oids}
results["C_minus_prior_only"] = {
    "predictions": c_minus_preds,
    "metrics": compute_full_metrics(c_minus_preds, query_gt),
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
    "policy_type": "no_interaction",
}
print(f"  C_minus_prior_only macro_bal={results['C_minus_prior_only']['metrics']['macro_query_balanced_accuracy']:.4f}")

# --- C0b_original ---
c0b_im = im_base.clone()
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, random.Random(SEED + 500))
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_visited_oids = [oid for oid in test_oids if c0b_obs.is_visited(oid)]
results["C0b_original"] = {
    "predictions": c0b_preds["per_object"],
    "metrics": compute_full_metrics(c0b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c0b_log, c0b_obs),
    "policy_type": "observe_only",
    "visited_oids": c0b_visited_oids,
}
c0b_mb = results["C0b_original"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C0b_original macro_bal={c0b_mb:.4f}  visited={len(c0b_visited_oids)}")

# --- C13_instance_VOI_original ---
c13_im = im_base.clone()
c13_policy = C13_InstanceVOIPolicy(c13_im, random.Random(SEED + 600), cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
c13_visited_oids = [oid for oid in test_oids if c13_obs.is_visited(oid)]
results["C13_instance_VOI_original"] = {
    "predictions": c13_preds["per_object"],
    "metrics": compute_full_metrics(c13_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c13_log, c13_obs),
    "policy_type": "instance_VOI",
    "visited_oids": c13_visited_oids,
}
c13_mb = results["C13_instance_VOI_original"]["metrics"]["macro_query_balanced_accuracy"]
c13_visited_n = len(c13_visited_oids)
c13_probed_n = results["C13_instance_VOI_original"]["cost"]["probe_count"]
print(f"  C13_original macro_bal={c13_mb:.4f}  visited={c13_visited_n}  probed={c13_probed_n}")

# --- C15b_probe_reserve_original ---
c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_visited_oids = [oid for oid in test_oids if c15b_obs.is_visited(oid)]
results["C15b_probe_reserve_original"] = {
    "predictions": c15b_preds["per_object"],
    "metrics": compute_full_metrics(c15b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "policy_type": "two_pass_VOI",
    "visited_oids": c15b_visited_oids,
}
c15b_mb = results["C15b_probe_reserve_original"]["metrics"]["macro_query_balanced_accuracy"]
c15b_pos_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"]
c15b_neg_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_negative_recall"]
c15b_visited_n = len(c15b_visited_oids)
c15b_probed_n = results["C15b_probe_reserve_original"]["cost"]["probe_count"]
c15b_probe_pairs = [(e["oid"], e["action"]) for e in c15b_policy._probe_tracking]
c15b_probe_pair_set = set(c15b_probe_pairs)
c15b_selected_actions_from_tracking = [e["action"] for e in c15b_policy._probe_tracking]
print(f"  C15b_original macro_bal={c15b_mb:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")
print(f"  C15b probe pairs: {c15b_probe_pairs}")
print(f"  C15b probe actions: {dict(Counter(c15b_selected_actions_from_tracking))}")

# --- C18a_local_query_confidence_VOI ---
print("\n  Running C18a_local_query_confidence_VOI (ablation)...")
t_c18a = time.time()
c18a_im = im_base.clone()
c18a_policy = C18a_LocalQueryConfidenceVOIPolicy(c18a_im, random.Random(SEED + 800), cost_weight=COST_WEIGHT)
c18a_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18a_preds, c18a_log, c18a_obs, _, _ = run_policy(c18a_policy, c18a_env)
c18a_visited_oids = [oid for oid in test_oids if c18a_obs.is_visited(oid)]
c18a_probe_effects = c18a_policy.get_probe_effects()
c18a_selected_actions = [e["action"] for e in c18a_probe_effects]
c18a_selected_objects = [e["oid"] for e in c18a_probe_effects]
c18a_zero_gain = sum(1 for e in c18a_probe_effects if e["is_zero_gain"])
c18a_harmful = sum(1 for e in c18a_probe_effects if e["is_harmful"])
c18a_effective = len(c18a_probe_effects) - c18a_zero_gain - c18a_harmful

results["C18a_local_query_confidence_VOI"] = {
    "predictions": c18a_preds["per_object"],
    "metrics": compute_full_metrics(c18a_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c18a_log, c18a_obs),
    "policy_type": "local_query_confidence_VOI_ablation",
    "visited_oids": c18a_visited_oids,
    "probe_effects": c18a_probe_effects,
    "selected_probe_objects": c18a_selected_objects,
    "selected_probe_actions": c18a_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18a_selected_actions)),
    "effective_probe_count": c18a_effective,
    "zero_gain_probe_count": c18a_zero_gain,
    "harmful_probe_count": c18a_harmful,
}
c18a_mb = results["C18a_local_query_confidence_VOI"]["metrics"]["macro_query_balanced_accuracy"]
c18a_pos_recall = results["C18a_local_query_confidence_VOI"]["metrics"]["mean_positive_recall"]
c18a_neg_recall = results["C18a_local_query_confidence_VOI"]["metrics"]["mean_negative_recall"]
c18a_visited_n = len(c18a_visited_oids)
c18a_probed_n = results["C18a_local_query_confidence_VOI"]["cost"]["probe_count"]
c18a_delta = c18a_mb - c15b_mb
c18a_runtime = time.time() - t_c18a
print(f"  C18a macro_bal={c18a_mb:.4f}  visited={c18a_visited_n}  probed={c18a_probed_n}  "
      f"delta_vs_C15b={c18a_delta:+.4f}  runtime={c18a_runtime:.1f}s")

# --- C18b_query_relevant_fixed_probe_budget (PRIMARY) ---
print("\n  Running C18b_query_relevant_fixed_probe_budget (PRIMARY)...")
t_c18b = time.time()
c18b_im = im_base.clone()
c18b_policy = C18b_QueryRelevantFixedBudgetPolicy(
    c18b_im, random.Random(SEED + 900), cost_weight=COST_WEIGHT,
    fixed_probe_count=c15b_probed_n)  # match C15b's probe count
c18b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18b_preds, c18b_log, c18b_obs, _, _ = run_policy(c18b_policy, c18b_env)
c18b_visited_oids = [oid for oid in test_oids if c18b_obs.is_visited(oid)]
c18b_probe_effects = c18b_policy.get_probe_effects()
c18b_selected_actions = [e["action"] for e in c18b_probe_effects]
c18b_selected_objects = [e["oid"] for e in c18b_probe_effects]
c18b_zero_gain = sum(1 for e in c18b_probe_effects if e["is_zero_gain"])
c18b_harmful = sum(1 for e in c18b_probe_effects if e["is_harmful"])
c18b_effective = len(c18b_probe_effects) - c18b_zero_gain - c18b_harmful

results["C18b_query_relevant_fixed_probe_budget"] = {
    "predictions": c18b_preds["per_object"],
    "metrics": compute_full_metrics(c18b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c18b_log, c18b_obs),
    "policy_type": "query_relevant_fixed_budget_primary",
    "visited_oids": c18b_visited_oids,
    "probe_effects": c18b_probe_effects,
    "selected_probe_objects": c18b_selected_objects,
    "selected_probe_actions": c18b_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18b_selected_actions)),
    "effective_probe_count": c18b_effective,
    "zero_gain_probe_count": c18b_zero_gain,
    "harmful_probe_count": c18b_harmful,
    "target_probe_count": c15b_probed_n,
}
c18b_mb = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["macro_query_balanced_accuracy"]
c18b_pos_recall = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["mean_positive_recall"]
c18b_neg_recall = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["mean_negative_recall"]
c18b_visited_n = len(c18b_visited_oids)
c18b_probed_n = results["C18b_query_relevant_fixed_probe_budget"]["cost"]["probe_count"]
c18b_delta = c18b_mb - c15b_mb
c18b_runtime = time.time() - t_c18b
c18b_probe_pairs = [(e["oid"], e["action"]) for e in c18b_probe_effects]
c18b_probe_pair_set = set(c18b_probe_pairs)
c18b_c15b_pair_overlap = c18b_probe_pair_set & c15b_probe_pair_set
c18b_c15b_pair_overlap_n = len(c18b_c15b_pair_overlap)
c18b_unique_to_c18b = c18b_probe_pair_set - c15b_probe_pair_set
c18b_unique_to_c15b = c15b_probe_pair_set - c18b_probe_pair_set

print(f"  C18b macro_bal={c18b_mb:.4f}  visited={c18b_visited_n}  probed={c18b_probed_n}  "
      f"delta_vs_C15b={c18b_delta:+.4f}  runtime={c18b_runtime:.1f}s")
print(f"  C18b probe actions: {dict(Counter(c18b_selected_actions))}")
print(f"  C18b effective={c18b_effective}  zero_gain={c18b_zero_gain}  harmful={c18b_harmful}")
print(f"  C18b vs C15b pair overlap: {c18b_c15b_pair_overlap_n}/{c18b_probed_n} = {c18b_c15b_pair_overlap_n/max(c18b_probed_n,1)*100:.1f}%")
if c18b_unique_to_c18b:
    print(f"  Unique to C18b: {sorted(c18b_unique_to_c18b)}")
if c18b_unique_to_c15b:
    print(f"  Unique to C15b: {sorted(c18b_unique_to_c15b)}")

# --- C18c_query_relevant_netVOI ---
print("\n  Running C18c_query_relevant_netVOI...")
t_c18c = time.time()
c18c_im = im_base.clone()
c18c_policy = C18c_QueryRelevantNetVOIPolicy(c18c_im, random.Random(SEED + 1000), cost_weight=COST_WEIGHT)
c18c_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18c_preds, c18c_log, c18c_obs, _, _ = run_policy(c18c_policy, c18c_env)
c18c_visited_oids = [oid for oid in test_oids if c18c_obs.is_visited(oid)]
c18c_probe_effects = c18c_policy.get_probe_effects()
c18c_selected_actions = [e["action"] for e in c18c_probe_effects]
c18c_selected_objects = [e["oid"] for e in c18c_probe_effects]
c18c_zero_gain = sum(1 for e in c18c_probe_effects if e["is_zero_gain"])
c18c_harmful = sum(1 for e in c18c_probe_effects if e["is_harmful"])
c18c_effective = len(c18c_probe_effects) - c18c_zero_gain - c18c_harmful

results["C18c_query_relevant_netVOI"] = {
    "predictions": c18c_preds["per_object"],
    "metrics": compute_full_metrics(c18c_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c18c_log, c18c_obs),
    "policy_type": "query_relevant_netVOI",
    "visited_oids": c18c_visited_oids,
    "probe_effects": c18c_probe_effects,
    "selected_probe_objects": c18c_selected_objects,
    "selected_probe_actions": c18c_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18c_selected_actions)),
    "effective_probe_count": c18c_effective,
    "zero_gain_probe_count": c18c_zero_gain,
    "harmful_probe_count": c18c_harmful,
}
c18c_mb = results["C18c_query_relevant_netVOI"]["metrics"]["macro_query_balanced_accuracy"]
c18c_pos_recall = results["C18c_query_relevant_netVOI"]["metrics"]["mean_positive_recall"]
c18c_neg_recall = results["C18c_query_relevant_netVOI"]["metrics"]["mean_negative_recall"]
c18c_visited_n = len(c18c_visited_oids)
c18c_probed_n = results["C18c_query_relevant_netVOI"]["cost"]["probe_count"]
c18c_delta = c18c_mb - c15b_mb
c18c_runtime = time.time() - t_c18c
print(f"  C18c macro_bal={c18c_mb:.4f}  visited={c18c_visited_n}  probed={c18c_probed_n}  "
      f"delta_vs_C15b={c18c_delta:+.4f}  runtime={c18c_runtime:.1f}s")

# --- Oracle ---
oracle_preds = C14a_TruthAnswerOracle(sim_gt).get_answer()
results["C_oracle_full_information"] = {
    "predictions": oracle_preds["per_object"],
    "metrics": compute_full_metrics(oracle_preds["per_object"], query_gt),
    "policy_type": "oracle",
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0},
}
oracle_mb = results["C_oracle_full_information"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C_oracle macro_bal={oracle_mb:.4f}")


# =============================================================================
# 10. Comparisons
# =============================================================================
print("\n[5/7] Computing comparisons...")
target_10pct = 0.6284
oracle_c0b_gap = oracle_mb - c0b_mb

comparisons = {}

# Primary: C18b vs C15b
comparisons["C18b_vs_C15b_primary"] = {
    "delta_macro_bal": round(c18b_delta, 6),
    "delta_positive_recall": round(c18b_pos_recall - c15b_pos_recall, 6),
    "delta_negative_recall": round(c18b_neg_recall - c15b_neg_recall, 6),
    "delta_visit_count": c18b_visited_n - c15b_visited_n,
    "delta_probe_count": c18b_probed_n - c15b_probed_n,
    "C18b_hits_10pct_target": c18b_mb >= target_10pct,
    "C18b_gap_capture_pct": round((c18b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
}

# C18a ablation vs C15b
comparisons["C18a_vs_C15b_ablation"] = {
    "delta_macro_bal": round(c18a_delta, 6),
    "delta_visit_count": c18a_visited_n - c15b_visited_n,
    "delta_probe_count": c18a_probed_n - c15b_probed_n,
}

# C18c vs C15b
comparisons["C18c_vs_C15b"] = {
    "delta_macro_bal": round(c18c_delta, 6),
    "delta_probe_count": c18c_probed_n - c15b_probed_n,
}

# C18b vs C18c (fixed budget vs net VOI stop rule)
comparisons["C18b_vs_C18c_stop_rule"] = {
    "C18b_macro_bal": c18b_mb,
    "C18c_macro_bal": c18c_mb,
    "delta": round(c18b_mb - c18c_mb, 6),
    "C18b_probe_count": c18b_probed_n,
    "C18c_probe_count": c18c_probed_n,
    "fixed_budget_beats_netVOI": c18b_mb > c18c_mb + 0.001,
    "interpretation": (
        "Fixed budget outperforms net VOI → stop rule / cost calibration is the issue."
        if c18b_mb > c18c_mb + 0.001
        else "Net VOI matches or beats fixed budget → stop rule is adequate."
        if c18c_mb >= c18b_mb - 0.001
        else "Both underperform → ranking itself is insufficient."
    ),
}

# Gap capture summary
comparisons["gap_capture"] = {
    "C0b_base": c0b_mb,
    "oracle_ceiling": oracle_mb,
    "oracle_c0b_gap": oracle_c0b_gap,
    "target_10pct_threshold": target_10pct,
    "C15b_gap_capture_pct": round((c15b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C18a_gap_capture_pct": round((c18a_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C18b_gap_capture_pct": round((c18b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C18c_gap_capture_pct": round((c18c_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
}

# Per-query breakdown
comparisons["per_query_breakdown"] = {}
for qname in QUERY_NAMES:
    c15b_q = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]
    c18b_q = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["per_query"][qname]
    comparisons["per_query_breakdown"][qname] = {
        "C15b_bal_acc": c15b_q["balanced_accuracy"],
        "C15b_pos_recall": c15b_q["positive_recall"],
        "C15b_neg_recall": c15b_q["negative_recall"],
        "C18b_bal_acc": c18b_q["balanced_accuracy"],
        "C18b_pos_recall": c18b_q["positive_recall"],
        "C18b_neg_recall": c18b_q["negative_recall"],
        "C18b_delta_vs_C15b": round(c18b_q["balanced_accuracy"] - c15b_q["balanced_accuracy"], 6),
    }

# Probe selection comparison
comparisons["probe_selection"] = {
    "C15b_probe_count": c15b_probed_n,
    "C15b_probe_pairs": str(c15b_probe_pairs),
    "C15b_action_distribution": dict(Counter(c15b_selected_actions_from_tracking)),
    "C18a_probe_count": c18a_probed_n,
    "C18a_action_distribution": dict(Counter(c18a_selected_actions)),
    "C18b_probe_count": c18b_probed_n,
    "C18b_probe_pairs": str(c18b_probe_pairs),
    "C18b_action_distribution": dict(Counter(c18b_selected_actions)),
    "C18b_target_count": c15b_probed_n,
    "C18c_probe_count": c18c_probed_n,
    "C18c_action_distribution": dict(Counter(c18c_selected_actions)),
    "c18b_c15b_pair_overlap": {
        "C15b_probe_pairs": str(sorted(c15b_probe_pairs)),
        "C18b_probe_pairs": str(sorted(c18b_probe_pairs)),
        "overlap_pairs": str(sorted(c18b_c15b_pair_overlap)),
        "overlap_count": c18b_c15b_pair_overlap_n,
        "overlap_pct_of_c18b": round(c18b_c15b_pair_overlap_n / max(c18b_probed_n, 1) * 100, 1),
        "unique_to_C18b": str(sorted(c18b_unique_to_c18b)),
        "unique_to_C18b_count": len(c18b_unique_to_c18b),
        "unique_to_C15b": str(sorted(c18b_unique_to_c15b)),
        "unique_to_C15b_count": len(c18b_unique_to_c15b),
        "ranking_differs_mainly_due_to_cost_penalty_removal": c18b_c15b_pair_overlap_n <= c18b_probed_n * 0.5,
    },
}

# IOM architecture notes
comparisons["iom_architecture_notes"] = {
    "no_cross_test_object_propagation": True,
    "global_recompute_equivalent_to_local": True,
    "note": (
        "IOM._direct_outcomes only affects the probed test object. "
        "Test objects are not in _train_oids, so k-NN retrieval for other "
        "test objects does not include probed test objects. "
        "Cross-object propagation requires a different IOM architecture."
    ),
}

# 1J14 oracle UB reference
comparisons["1J14_oracle_UB_reference"] = {
    "C15b_UB_macro_bal": 0.8546,
    "C15b_UB_delta_vs_C15b": 0.2404,
    "IOM_k8_macro_bal": 0.6488,
    "IOM_k8_delta_vs_C15b": 0.0346,
    "note": "Non-deployable oracle upper bounds from 1J14.",
}


# =============================================================================
# 11. Interpretation
# =============================================================================
print("\n[6/7] Interpreting results...")

c18b_hits_10pct = c18b_mb >= target_10pct
c18b_improves_c15b = c18b_mb > c15b_mb + 0.001

pair_overlap_note = (
    f"C15b–C18b (object, action) pair overlap: {c18b_c15b_pair_overlap_n}/{max(c18b_probed_n,1)} = "
    f"{c18b_c15b_pair_overlap_n/max(c18b_probed_n,1)*100:.1f}%. "
    f"Overlap pairs: {sorted(c18b_c15b_pair_overlap)}. "
    f"Unique to C18b: {sorted(c18b_unique_to_c18b)}. "
    f"Unique to C15b: {sorted(c18b_unique_to_c15b)}."
) if c18b_probed_n > 0 else "No C18b probes executed — cannot compute overlap."

if c18b_hits_10pct:
    c18b_interpretation = (
        f"C18b (query-relevant fixed budget) reaches {c18b_mb:.4f}, exceeding the 10% "
        f"gap threshold ({target_10pct}). Query-relevant probe ranking is STRONGLY "
        f"SUPPORTED. The ranking successfully identifies more diagnostically valuable "
        f"probes than C15b's VOI, even with the same probe count. "
        f"{pair_overlap_note} "
        f"IMPORTANT: Under current IOM architecture, test-object probes do not propagate "
        f"to other test objects, and local query-confidence gain min(p,1-p) is nearly "
        f"monotonic with C15b's affordance-utility gain. Improvement likely comes from "
        f"fixed probe budget / reduced cost penalty, not fundamentally new query-level "
        f"value modeling. Do not claim C18 is full query-level VOI unless it selects "
        f"DIFFERENT object-action pairs and improves macro_bal."
    )
    query_relevant_ranking_supported = True
    candidate_ready_for_small_multiseed = True
elif c18b_improves_c15b:
    c18b_interpretation = (
        f"C18b reaches {c18b_mb:.4f} (+{c18b_delta:+.4f} vs C15b), improving over "
        f"C15b but not reaching the 10% threshold ({target_10pct}). "
        f"{pair_overlap_note} "
        f"Improvement likely comes from fixed probe budget / reduced cost penalty "
        f"rather than query-level reasoning, since min(p,1-p) is nearly monotonic "
        f"with C15b's affordance-utility gain under current IOM. "
        f"Do not claim C18 is full query-level VOI."
    )
    query_relevant_ranking_supported = True
    candidate_ready_for_small_multiseed = False
elif abs(c18b_mb - c15b_mb) < 0.001:
    c18b_interpretation = (
        f"C18b reaches {c18b_mb:.4f}, essentially identical to C15b ({c15b_mb:.4f}). "
        f"{pair_overlap_note} "
        f"The query-relevant ranking does NOT differentiate from C15b's VOI ranking. "
        f"This is expected: for single-feature queries, min(p,1-p) is monotonically "
        f"equivalent to C15b's per-feature utility gain. "
        f"Query-confidence ranking is equivalent to affordance-utility ranking under "
        f"current IOM. Do not claim C18 is full query-level VOI."
    )
    query_relevant_ranking_supported = False
    candidate_ready_for_small_multiseed = False
else:
    c18b_interpretation = (
        f"C18b reaches {c18b_mb:.4f} ({c18b_delta:+.4f} vs C15b), below C15b. "
        f"{pair_overlap_note} "
        f"Query-relevant ranking DEGRADES performance. Under current IOM architecture, "
        f"test-object probes do not propagate to other objects, and local query-confidence "
        f"gain min(p,1-p) is nearly monotonic with C15b's affordance-utility gain. "
        f"The fixed budget forces probes that the VOI cost gate would have skipped, "
        f"resulting in budget wasted on low-value probes. "
        f"The missing ingredient is likely confident-error detection / calibration-aware "
        f"probe value, not simple uncertainty. Do not claim C18 is full query-level VOI."
    )
    query_relevant_ranking_supported = False
    candidate_ready_for_small_multiseed = False

# C18a ablation interpretation
if abs(c18a_mb - c15b_mb) < 0.001 and abs(c18a_mb - c18b_mb) < 0.001:
    c18a_interpretation = (
        f"C18a (local query confidence VOI) at {c18a_mb:.4f} is effectively "
        f"identical to both C15b ({c15b_mb:.4f}) and C18b ({c18b_mb:.4f}). "
        f"All three ranking functions (affordance utility, query confidence, "
        f"query-relevant gain) are monotonically equivalent for per-action "
        f"ranking within an object. Probe selection is determined by which "
        f"objects are visited, not by the scoring function."
    )
else:
    c18a_interpretation = (
        f"C18a at {c18a_mb:.4f} (delta vs C15b: {c18a_delta:+.4f}). "
        f"See per-query breakdown for differences."
    )

# C18c vs C18b stop-rule interpretation
if c18b_mb > c18c_mb + 0.001:
    stop_rule_interpretation = (
        f"C18b (fixed budget, {c18b_probed_n} probes) outperforms C18c (net VOI, "
        f"{c18c_probed_n} probes) by {c18b_mb - c18c_mb:+.4f}. The net VOI stop rule "
        f"is too conservative — it blocks probes that would be beneficial. "
        f"Cost calibration needs adjustment."
    )
elif c18c_mb > c18b_mb + 0.001:
    stop_rule_interpretation = (
        f"C18c (net VOI, {c18c_probed_n} probes) outperforms C18b (fixed budget, "
        f"{c18b_probed_n} probes) by {c18c_mb - c18b_mb:+.4f}. The net VOI stop rule "
        f"effectively filters out low-value probes."
    )
else:
    stop_rule_interpretation = (
        f"C18b and C18c perform similarly (delta={c18b_mb - c18c_mb:+.4f}). "
        f"The stop rule does not materially affect outcomes at current cost settings."
    )

print(f"  C18b macro_bal={c18b_mb:.4f}  delta_vs_C15b={c18b_delta:+.4f}")
print(f"  C18b hits 10% target: {c18b_hits_10pct}")
print(f"  C18b improves C15b: {c18b_improves_c15b}")
print(f"  query_relevant_ranking_supported: {query_relevant_ranking_supported}")
print(f"  candidate_ready_for_small_multiseed: {candidate_ready_for_small_multiseed}")

# Per-query
per_query_interp = {}
for qname in QUERY_NAMES:
    c15b_bal = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]["balanced_accuracy"]
    c18b_bal = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["per_query"][qname]["balanced_accuracy"]
    delta = c18b_bal - c15b_bal
    if delta > 0.01:
        per_query_interp[qname] = f"improved (+{delta:.4f})"
    elif delta < -0.01:
        per_query_interp[qname] = f"degraded ({delta:.4f})"
    else:
        per_query_interp[qname] = f"unchanged ({delta:.4f})"
    print(f"  {qname}: {per_query_interp[qname]}")


# =============================================================================
# 12. Save outputs
# =============================================================================
print("\n[7/7] Saving outputs...")

output = {
    "block_id": "1J15",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Query-Level VOI Probe Selection Diagnostic. Tests query-relevant direct probe ranking with three variants: C18a (ablation, VOI-gated), C18b (primary, fixed budget), C18c (net VOI gate).",
    "iom_architecture": {
        "no_cross_test_object_propagation": True,
        "global_recompute_equivalent_to_local": True,
    },
    "diagnostic_caveats": {
        "test_object_probes_do_not_propagate": True,
        "local_query_confidence_nearly_monotonic_with_affordance_utility": True,
        "c18_not_full_query_level_VOI": True,
        "c18_primarily_tests_cost_stop_calibration_not_value_model": True,
    },
    "c15b_c18b_pair_overlap": {
        "c15b_probe_pairs": str(c15b_probe_pairs),
        "c18b_probe_pairs": str(c18b_probe_pairs),
        "overlap_pairs": str(sorted(c18b_c15b_pair_overlap)),
        "overlap_count": c18b_c15b_pair_overlap_n,
        "overlap_pct": round(c18b_c15b_pair_overlap_n / max(c18b_probed_n, 1) * 100, 1),
        "unique_to_c18b": str(sorted(c18b_unique_to_c18b)),
        "unique_to_c18b_count": len(c18b_unique_to_c18b),
        "unique_to_c15b": str(sorted(c18b_unique_to_c15b)),
        "unique_to_c15b_count": len(c18b_unique_to_c15b),
        "ranking_differs_mainly_due_to_cost_penalty_removal": c18b_c15b_pair_overlap_n <= c18b_probed_n * 0.5,
    },
    "design_validation": {
        "no_environment_change": True,
        "no_budget_cost_change": True,
        "no_multi_seed": True,
        "no_hidden_labels_in_policy": True,
        "no_oracle_probe_outcomes": True,
        "no_IOM_update_change": True,
        "no_Route_F_F2_global_coarse_map": True,
    },
    "baselines": {},
    "C18a_details": {},
    "C18b_details": {},
    "C18c_details": {},
    "comparisons": comparisons,
    "interpretation": {
        "primary_variant": "C18b_query_relevant_fixed_probe_budget",
        "ablation_variant": "C18a_local_query_confidence_VOI",
        "netVOI_variant": "C18c_query_relevant_netVOI",
        "c18b_macro_bal": c18b_mb,
        "c18b_delta_vs_c15b": round(c18b_delta, 6),
        "c18b_gap_capture_vs_c0b": round((c18b_mb - c0b_mb) / oracle_c0b_gap, 6) if oracle_c0b_gap > 0 else 0.0,
        "c18b_hits_10pct_target": c18b_hits_10pct,
        "query_relevant_ranking_supported": query_relevant_ranking_supported,
        "candidate_ready_for_small_multiseed": candidate_ready_for_small_multiseed,
        "ready_for_multiseed": False,
        "no_cross_test_object_propagation": True,
        "c18_global_recompute_equivalent_to_local": True,
        "c18b_interpretation": c18b_interpretation,
        "c18a_interpretation": c18a_interpretation,
        "stop_rule_interpretation": stop_rule_interpretation,
        "per_query_interpretation": per_query_interp,
    },
}

# Serialize baselines
for name, r in results.items():
    output["baselines"][name] = {
        "macro_query_balanced_accuracy": r["metrics"]["macro_query_balanced_accuracy"],
        "macro_query_accuracy": r["metrics"]["macro_query_accuracy"],
        "mean_positive_recall": r["metrics"]["mean_positive_recall"],
        "mean_negative_recall": r["metrics"]["mean_negative_recall"],
        "per_query": r["metrics"]["per_query"],
        "visit_count": r["cost"]["visit_count"],
        "observe_count": r["cost"]["observe_count"],
        "probe_count": r["cost"]["probe_count"],
        "total_cost": r["cost"]["total_cost"],
        "normalized_cost": r["cost"]["normalized_cost"],
        "policy_type": r["policy_type"],
    }

# C18a details
c18a_r = results["C18a_local_query_confidence_VOI"]
output["C18a_details"] = {
    "variant": "ablation",
    "scoring": "local_query_confidence_VOI_gated",
    "macro_query_balanced_accuracy": c18a_mb,
    "mean_positive_recall": c18a_pos_recall,
    "mean_negative_recall": c18a_neg_recall,
    "per_query": c18a_r["metrics"]["per_query"],
    "visit_count": c18a_visited_n,
    "probe_count": c18a_probed_n,
    "total_cost": c18a_r["cost"]["total_cost"],
    "normalized_cost": c18a_r["cost"]["normalized_cost"],
    "selected_probe_objects": c18a_selected_objects,
    "selected_probe_actions": c18a_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18a_selected_actions)),
    "effective_probe_count": c18a_effective,
    "zero_gain_probe_count": c18a_zero_gain,
    "harmful_probe_count": c18a_harmful,
    "probe_effects": c18a_probe_effects,
    "delta_vs_C15b": round(c18a_delta, 6),
    "runtime_seconds": round(c18a_runtime, 1),
}

# C18b details
c18b_r = results["C18b_query_relevant_fixed_probe_budget"]
output["C18b_details"] = {
    "variant": "primary",
    "scoring": "query_relevant_fixed_probe_budget",
    "stop_rule": "fixed_k_equal_C15b_probe_count",
    "target_probe_count": c15b_probed_n,
    "macro_query_balanced_accuracy": c18b_mb,
    "mean_positive_recall": c18b_pos_recall,
    "mean_negative_recall": c18b_neg_recall,
    "per_query": c18b_r["metrics"]["per_query"],
    "visit_count": c18b_visited_n,
    "probe_count": c18b_probed_n,
    "total_cost": c18b_r["cost"]["total_cost"],
    "normalized_cost": c18b_r["cost"]["normalized_cost"],
    "selected_probe_objects": c18b_selected_objects,
    "selected_probe_actions": c18b_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18b_selected_actions)),
    "effective_probe_count": c18b_effective,
    "zero_gain_probe_count": c18b_zero_gain,
    "harmful_probe_count": c18b_harmful,
    "probe_effects": c18b_probe_effects,
    "delta_vs_C15b": round(c18b_delta, 6),
    "runtime_seconds": round(c18b_runtime, 1),
}

# C18c details
c18c_r = results["C18c_query_relevant_netVOI"]
output["C18c_details"] = {
    "variant": "netVOI",
    "scoring": "query_relevant_netVOI_gated",
    "stop_rule": "positive_net_VOI_with_cost_penalty",
    "macro_query_balanced_accuracy": c18c_mb,
    "mean_positive_recall": c18c_pos_recall,
    "mean_negative_recall": c18c_neg_recall,
    "per_query": c18c_r["metrics"]["per_query"],
    "visit_count": c18c_visited_n,
    "probe_count": c18c_probed_n,
    "total_cost": c18c_r["cost"]["total_cost"],
    "normalized_cost": c18c_r["cost"]["normalized_cost"],
    "selected_probe_objects": c18c_selected_objects,
    "selected_probe_actions": c18c_selected_actions,
    "selected_probe_action_summary": dict(Counter(c18c_selected_actions)),
    "effective_probe_count": c18c_effective,
    "zero_gain_probe_count": c18c_zero_gain,
    "harmful_probe_count": c18c_harmful,
    "probe_effects": c18c_probe_effects,
    "delta_vs_C15b": round(c18c_delta, 6),
    "runtime_seconds": round(c18c_runtime, 1),
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j15_query_level_voi_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J15")
print(f"c15b_macro_bal={c15b_mb:.6f}")
print(f"c15b_probe_count={c15b_probed_n}")
print(f"c15b_probe_pairs={c15b_probe_pairs}")
print(f"c18a_macro_bal={c18a_mb:.6f}")
print(f"c18a_delta_vs_c15b={c18a_delta:+.6f}")
print(f"c18b_macro_bal={c18b_mb:.6f}")
print(f"c18b_delta_vs_c15b={c18b_delta:+.6f}")
print(f"c18b_gap_capture_vs_c0b={(c18b_mb - c0b_mb) / oracle_c0b_gap:.6f}")
print(f"c18b_hits_10pct_target={'true' if c18b_hits_10pct else 'false'}")
print(f"c18b_probe_pairs={c18b_probe_pairs}")
print(f"c18b_c15b_pair_overlap={c18b_c15b_pair_overlap_n}/{c18b_probed_n}={c18b_c15b_pair_overlap_n/max(c18b_probed_n,1)*100:.1f}%")
print(f"c18b_unique_to_c18b={sorted(c18b_unique_to_c18b)}")
print(f"c18b_unique_to_c15b={sorted(c18b_unique_to_c15b)}")
print(f"c18c_macro_bal={c18c_mb:.6f}")
print(f"c18c_delta_vs_c15b={c18c_delta:+.6f}")
print(f"query_relevant_ranking_supported={'true' if query_relevant_ranking_supported else 'false'}")
print(f"candidate_ready_for_small_multiseed={'true' if candidate_ready_for_small_multiseed else 'false'}")
print(f"ready_for_multiseed=false")
print(f"no_cross_test_object_propagation=true")
print(f"c18_global_recompute_equivalent_to_local=true")
print(f"c18_not_full_query_level_VOI=true")
print(f"c18a_runtime_seconds={c18a_runtime:.1f}")
print(f"c18b_runtime_seconds={c18b_runtime:.1f}")
print(f"c18c_runtime_seconds={c18c_runtime:.1f}")
print(f"elapsed_seconds={elapsed:.1f}")

# =============================================================================
# 13. Protocol MD
# =============================================================================
print("\nWriting protocol MD...")

per_query_rows = []
for qname in QUERY_NAMES:
    c15b_q = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]
    c18b_q = results["C18b_query_relevant_fixed_probe_budget"]["metrics"]["per_query"][qname]
    c18c_q = results["C18c_query_relevant_netVOI"]["metrics"]["per_query"][qname]
    per_query_rows.append(
        f"| {qname} | {c15b_q['balanced_accuracy']:.4f} | {c18b_q['balanced_accuracy']:.4f} | "
        f"{c18c_q['balanced_accuracy']:.4f} | {c18b_q['balanced_accuracy'] - c15b_q['balanced_accuracy']:+.4f} |"
    )

protocol_md = f"""# Block 1J15 — Query-Relevant Probe Ranking Diagnostic

**Date:** 2026-05-11
**Block:** 1J15 — Query-relevant probe ranking diagnostic
**Condition:** C4_instance_subtype_cued_v1
**Seed:** 101
**Budget:** 1.5

---

## 1. Executive Summary

1J15 tests whether query-relevant probe ranking (preferring object-action pairs near query decision boundaries) improves macro_bal over C15b's affordance-utility VOI. Three variants isolate ranking quality from stop-rule/cost-calibration issues:

- **C18a (ablation)**: Local query-confidence VOI, VOI-gated with cost penalty.
- **C18b (PRIMARY)**: Query-relevant ranking, fixed probe budget (= C15b's probe count, k≈{c15b_probed_n}). No cost threshold — isolates probe selection quality.
- **C18c**: Same ranking as C18b, but VOI-gated with cost penalty. Tests whether the deployable stop rule works.

**Architecture finding**: The current IOM has no cross-test-object propagation (`_direct_outcomes` only affects the probed object; test objects are not in `_train_oids`). Global recomputation is equivalent to local direct-effect scoring. Therefore C18 tests query-relevant direct probe ranking, not propagation-based global VOI.

**C18b reaches macro_bal={c18b_mb:.4f} (delta_vs_C15b={c18b_delta:+.4f}).** {c18b_interpretation}

---

## 2. Motivation

| Prior Finding | Implication |
|--------------|-------------|
| 1J14: C15b_UB reaches 0.8546 with oracle probe selection | Probe information has substantial usable value |
| 1J14: Oracle k=8 (same budget) reaches 0.6488 vs C15b 0.6142 | C15b's VOI selects wrong probes — +0.0346 from better selection |
| 1J14: C15b VOI = affordance utility; oracle ranking = \|pred - gt\| | Different ranking functions produce different probe sets |
| IOM: _direct_outcomes only affects probed object | No cross-object propagation; global = local |

**Key question**: Can a query-relevant ranking (preferring probes near query decision boundaries) improve over C15b's affordance-utility ranking, even with the same probe budget?

---

## 3. C18 Variant Designs

### 3.1 Common Architecture

All C18 variants share C15b's observe phase:
- Nearest-first object visitation
- 25% budget reserve
- Same transition logic to probe phase

### 3.2 Query-Relevant Ranking Score

For candidate (object, action), score = `min(p, 1-p)` where p = current IOM prediction for the feature that this action probes.

This is the expected gain in query decision confidence from resolving this (object, feature):
- Current query confidence = max(p, 1-p)
- After probing (resolved to 0 or 1): confidence = 1.0
- Expected gain = 1.0 - max(p, 1-p) = min(p, 1-p)

Action-to-query mapping:
| Action | Query |
|--------|-------|
| craft_plank | need_planks |
| eat | need_food |
| use_as_tool | need_tool |
| burn_as_fuel | need_fuel |
| mine_by_hand | need_stone |
| mine_with_pickaxe | need_stone |

### 3.3 C18a — Local Query Confidence VOI (Ablation)

- Score: query-relevant gain / 6 (scaled like C15b's per-feature utility)
- Stop rule: positive net VOI with cost penalty
- Purpose: test whether query-confidence framing alone differs from C15b

### 3.4 C18b — Query-Relevant Fixed Probe Budget (PRIMARY)

- Score: query-relevant gain (no scaling)
- Stop rule: probe exactly k objects (k = C15b's observed probe count = {c15b_probed_n})
- No cost threshold — isolates probe selection quality
- Purpose: if C18b > C15b with same probe count, the ranking is better

### 3.5 C18c — Query-Relevant Net VOI

- Score: same as C18b
- Stop rule: positive net VOI with cost penalty (like C15b)
- Purpose: if C18b > C15b but C18c ≤ C15b, the ranking works but the stop rule / cost calibration blocks it

---

## 4. Forbidden-Information Compliance

| Check | Status |
|-------|--------|
| No environment change | ✓ |
| No budget/cost change | ✓ |
| Single seed only | ✓ |
| No hidden labels in policy | ✓ |
| No oracle probe outcomes | ✓ |
| No IOM update change | ✓ |
| No Route F/F2/global coarse map | ✓ |
| No ground truth in ranking | ✓ (uses only current IOM predictions) |

---

## 5. Baseline Results

| Policy | macro_bal | pos_recall | neg_recall | visited | probed | ncost | Type |
|--------|-----------|------------|------------|---------|--------|-------|------|
| all_false | {results['all_false']['metrics']['macro_query_balanced_accuracy']:.4f} | {results['all_false']['metrics']['mean_positive_recall']:.4f} | {results['all_false']['metrics']['mean_negative_recall']:.4f} | 0 | 0 | 0.0000 | floor |
| C_minus_prior_only | {results['C_minus_prior_only']['metrics']['macro_query_balanced_accuracy']:.4f} | — | — | 0 | 0 | 0.0000 | no_interaction |
| C0b_original | {c0b_mb:.4f} | {results['C0b_original']['metrics']['mean_positive_recall']:.4f} | {results['C0b_original']['metrics']['mean_negative_recall']:.4f} | {results['C0b_original']['cost']['visit_count']} | 0 | {results['C0b_original']['cost']['normalized_cost']:.4f} | observe_only |
| C13_instance_VOI_original | {c13_mb:.4f} | — | — | {c13_visited_n} | {c13_probed_n} | {results['C13_instance_VOI_original']['cost']['normalized_cost']:.4f} | instance_VOI |
| C15b_probe_reserve_original | {c15b_mb:.4f} | {c15b_pos_recall:.4f} | {c15b_neg_recall:.4f} | {c15b_visited_n} | {c15b_probed_n} | {results['C15b_probe_reserve_original']['cost']['normalized_cost']:.4f} | two_pass_VOI |
| C_oracle_full_information | {oracle_mb:.4f} | 1.0000 | 1.0000 | — | — | — | oracle |

### 1J14 Oracle UB Reference (non-deployable)

| Variant | macro_bal | delta_vs_C15b |
|---------|-----------|---------------|
| C15b_UB | 0.8546 | +0.2404 |
| IOM k=8 | 0.6488 | +0.0346 |
| C13_full_UB | 0.6867 | +0.0725 |
| Full_UB | 1.0000 | +0.3858 |

Oracle-C0b gap: **{oracle_c0b_gap:.4f}**. C15b captures {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}%.

---

## 6. C18 Results

### 6.1 Aggregate Metrics

| Metric | C15b | C18a (ablation) | C18b (primary) | C18c (netVOI) |
|--------|------|-----------------|----------------|---------------|
| macro_bal | {c15b_mb:.4f} | {c18a_mb:.4f} | **{c18b_mb:.4f}** | {c18c_mb:.4f} |
| mean_positive_recall | {c15b_pos_recall:.4f} | {c18a_pos_recall:.4f} | {c18b_pos_recall:.4f} | {c18c_pos_recall:.4f} |
| mean_negative_recall | {c15b_neg_recall:.4f} | {c18a_neg_recall:.4f} | {c18b_neg_recall:.4f} | {c18c_neg_recall:.4f} |
| visit_count | {c15b_visited_n} | {c18a_visited_n} | {c18b_visited_n} | {c18c_visited_n} |
| probe_count | {c15b_probed_n} | {c18a_probed_n} | {c18b_probed_n} | {c18c_probed_n} |
| total_cost | {results['C15b_probe_reserve_original']['cost']['total_cost']:.4f} | {c18a_r['cost']['total_cost']:.4f} | {c18b_r['cost']['total_cost']:.4f} | {c18c_r['cost']['total_cost']:.4f} |
| delta_vs_C15b | — | {c18a_delta:+.4f} | **{c18b_delta:+.4f}** | {c18c_delta:+.4f} |
| gap_capture | {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | {(c18a_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% | **{(c18b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}%** | {(c18c_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| runtime_s | — | {c18a_runtime:.1f} | {c18b_runtime:.1f} | {c18c_runtime:.1f} |

### 6.2 Per-Query Breakdown

| Query | C15b Bal | C18b Bal | C18c Bal | C18b Δ vs C15b |
|-------|----------|----------|----------|----------------|
{chr(10).join(per_query_rows)}

### 6.3 Probe Selection

| Metric | C15b | C18a | C18b | C18c |
|--------|------|------|------|------|
| probe_count | {c15b_probed_n} | {c18a_probed_n} | {c18b_probed_n} | {c18c_probed_n} |
| actions | {dict(Counter(c15b_selected_actions_from_tracking))} | {dict(Counter(c18a_selected_actions))} | {dict(Counter(c18b_selected_actions))} | {dict(Counter(c18c_selected_actions))} |
| effective | — | {c18a_effective} | {c18b_effective} | {c18c_effective} |
| zero_gain | — | {c18a_zero_gain} | {c18b_zero_gain} | {c18c_zero_gain} |
| harmful | — | {c18a_harmful} | {c18b_harmful} | {c18c_harmful} |

### 6.4 C15b vs C18b (Object, Action) Pair Overlap

| Metric | Value |
|--------|-------|
| C15b probe pairs | {sorted(c15b_probe_pairs)} |
| C18b probe pairs | {sorted(c18b_probe_pairs)} |
| overlap pairs | {sorted(c18b_c15b_pair_overlap)} |
| overlap count | {c18b_c15b_pair_overlap_n} / {max(c18b_probed_n,1)} = {c18b_c15b_pair_overlap_n/max(c18b_probed_n,1)*100:.1f}% |
| unique to C18b | {sorted(c18b_unique_to_c18b)} ({len(c18b_unique_to_c18b)}) |
| unique to C15b | {sorted(c18b_unique_to_c15b)} ({len(c18b_unique_to_c15b)}) |
| ranking differs due to cost penalty removal | **{'Yes' if c18b_c15b_pair_overlap_n <= c18b_probed_n * 0.5 else 'No — rankings are substantially similar'}** |

---

## 7. Key Comparisons

### 7.1 C18b vs C15b (PRIMARY)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **{c18b_delta:+.4f}** |
| delta_positive_recall | {c18b_pos_recall - c15b_pos_recall:+.4f} |
| delta_negative_recall | {c18b_neg_recall - c15b_neg_recall:+.4f} |
| delta_visit_count | {c18b_visited_n - c15b_visited_n} |
| delta_probe_count | {c18b_probed_n - c15b_probed_n} |
| C18b hits 10% threshold ({target_10pct}) | **{'Yes' if c18b_hits_10pct else 'No'}** |

### 7.2 C18b vs C18c (Fixed Budget vs Net VOI Stop Rule)

| Metric | C18b | C18c | Delta |
|--------|------|------|-------|
| macro_bal | {c18b_mb:.4f} | {c18c_mb:.4f} | {c18b_mb - c18c_mb:+.4f} |
| probe_count | {c18b_probed_n} | {c18c_probed_n} | {c18b_probed_n - c18c_probed_n} |

**{stop_rule_interpretation}**

### 7.3 C18a vs C15b (Ablation)

| Metric | Delta |
|--------|-------|
| delta_macro_bal | **{c18a_delta:+.4f}** |
| C18a equivalent to C15b | **{'Yes' if abs(c18a_mb - c15b_mb) < 0.001 else 'No'}** |

### 7.4 Gap Capture

| Policy | gap_capture |
|--------|-------------|
| C15b | {(c15b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C18a | {(c18a_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C18b | {(c18b_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| C18c | {(c18c_mb - c0b_mb) / oracle_c0b_gap * 100:.2f}% |
| 1J14 C15b_UB | 64.78% |
| 1J14 IOM k=8 | 14.93% |

---

## 8. Interpretation

```
C18b improves over C15b:     {'True' if c18b_improves_c15b else 'False'} ({c18b_delta:+.4f})
C18b hits 10% threshold:     {'True' if c18b_hits_10pct else 'False'} ({c18b_mb:.4f} vs {target_10pct})
query_relevant_ranking_supported: {query_relevant_ranking_supported}
candidate_ready_for_small_multiseed: {candidate_ready_for_small_multiseed}
ready_for_multiseed:          False
no_cross_test_object_propagation: true
c18_global_recompute_equivalent_to_local: true
```

**C18b interpretation: {c18b_interpretation}**

**C18a ablation: {c18a_interpretation}**

**Stop rule: {stop_rule_interpretation}**

### Per-Query Interpretation

{chr(10).join(f"- **{q}**: {interp}" for q, interp in per_query_interp.items())}

---

## 9. Why Query-Relevant Ranking May Not Differentiate

### 9.1 Monotonic Equivalence Under Current IOM

For single-feature queries (need_planks, need_food, need_tool, need_fuel), the query decision depends on a single feature crossing the 0.5 threshold. The query-relevant gain `min(p, 1-p)` and C15b's per-feature utility gain `(1 - max(p, 1-p)) / 6` are **monotonically equivalent** — they produce the same per-action ranking within an object.

The only structural differences between C15b and C18b are:
1. **Per-object aggregation**: Both use max over actions → same best action per object
2. **Cost penalty**: C18b ignores reach cost in ranking; C15b penalizes distant objects
3. **Stop rule**: C18b probes exactly k objects; C15b stops when no net VOI > 0

### 9.2 Diagnostic Caveats

**Under current IOM architecture:**
- Test-object probes do NOT propagate to other test objects (`_direct_outcomes` only affects the probed object; test objects are not in `_train_oids`).
- Local query-confidence gain using `min(p, 1-p)` is nearly monotonic with C15b's affordance-utility gain.
- Therefore C18a/C18c mainly test cost/stop calibration, not a fundamentally new query-level value model.

**C18 is NOT claimed as full query-level VOI.** The C18 variants test query-relevant direct probe ranking, not propagation-based global VOI.

### 9.3 Pair Overlap Findings

| Metric | Value |
|--------|-------|
| C15b probe pairs | {sorted(c15b_probe_pairs)} |
| C18b probe pairs | {sorted(c18b_probe_pairs)} |
| Overlap | {c18b_c15b_pair_overlap_n}/{c18b_probed_n} = {c18b_c15b_pair_overlap_n/max(c18b_probed_n,1)*100:.1f}% |
| Ranking differs due to cost penalty removal | **{'Yes — cost penalty removal changes selected pairs substantially' if c18b_c15b_pair_overlap_n <= c18b_probed_n * 0.5 else 'No — rankings are substantially similar'}** |

### 9.4 Interpreting C18b Results

- **If C18b improves over C15b**: improvement likely comes from fixed probe budget / reduced cost penalty, not necessarily query-level reasoning.
- **If C18b equals C15b**: query-confidence ranking is equivalent to affordance-utility ranking under current IOM.
- **If C18b fails to approach 1J14 oracle k=8 (0.6488)**: the missing ingredient is likely confident-error detection / calibration-aware probe value, not simple uncertainty.

**Current result: C18b = {c18b_mb:.4f} vs 1J14 oracle k=8 = 0.6488 → gap = {0.6488 - c18b_mb:.4f}.** {("C18b is substantially below oracle k=8. Confident-error detection / calibration-aware probe value is the missing ingredient." if 0.6488 - c18b_mb > 0.01 else "C18b approaches oracle k=8 level.")}

---

## 10. Final Recommendation

**{"Query-relevant ranking with fixed budget is supported. Candidate for small multiseed (3 seeds)." if candidate_ready_for_small_multiseed else "Query-relevant ranking does not differentiate from C15b's VOI. The ranking functions are monotonically equivalent for the current IOM architecture." if not query_relevant_ranking_supported else "Query-relevant ranking shows marginal improvement. Consider: (1) designing a better probe value surrogate using 1J14 oracle k=8 patterns, (2) multi-step VOI, or (3) IOM architecture changes to enable cross-object propagation."}**

### Consolidated Block 1J11-1J15 Findings

| Block | Approach | Result | vs C15b | Key Finding |
|-------|----------|--------|---------|-------------|
| 1J11 | F_mid aggregate peripheral | 0.6142 | +0.0000 | Safe but not useful |
| 1J12 | Global coarse scene map (C15G) | 0.5992 | -0.0150 | Scene priority misaligns with diagnostic value |
| 1J13 | Interleaved breadth-depth (C17) | 0.5667 | -0.0475 | VOI doesn't activate at low breadth |
| 1J14 | Oracle probe injection (C15b_UB) | 0.8546 | +0.2404 | Substantial probe value exists; VOI fails to find it |
| **1J15** | **Query-relevant ranking (C18b)** | **{c18b_mb:.4f}** | **{c18b_delta:+.4f}** | **{c18b_interpretation[:120]}...** |

---

*Generated by Block 1J15 — Query-Relevant Probe Ranking Diagnostic.*
"""

protocol_path = os.path.join(CURRENT_DIR, "protocols",
                             "block1j15_query_level_voi_c4_seed101.md")
with open(protocol_path, "w", encoding="utf-8") as f:
    f.write(protocol_md)
print(f"  Saved: {protocol_path}")

print(f"\nDone! elapsed={elapsed:.1f}s")
