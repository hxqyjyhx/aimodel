"""
Block 1J13 -- Interleaved Breadth-Depth Diagnostic.

Tests whether interleaving observation and probing (instead of strict
two-pass) improves object selection / probe targeting.

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
from event_log import EventLog
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
INTERLEAVED_K_VALUES = [3, 5, 8]

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J13 -- Interleaved Breadth-Depth Diagnostic")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
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

# Object metadata
object_meta = {}
for oid in test_oids:
    obj = test_objects[oid]
    cat = obj["hidden_category"]
    subtype = obj["hidden_subtype"]
    gt_aff = sim_gt.get_ground_truth_affordances(oid)
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    ratios = subtype_def["subtype_ratio"]
    is_majority = ratios.get(subtype, 0.0) >= 0.5
    object_meta[oid] = {
        "category": cat, "subtype": subtype, "is_majority": is_majority,
        "gt_affordances": gt_aff,
    }

# Public visible features for all objects
ALL_VISIBLE_FEATURES = {}
for oid in test_oids:
    ALL_VISIBLE_FEATURES[oid] = dict(test_objects[oid].get("visible_features", {}))

# Global base rates
GLOBAL_FEATURE_PREVALENCE = {}
for f in posterior_visible_features:
    prevalence = sum(1 for oid in test_oids if ALL_VISIBLE_FEATURES[oid].get(f, False)) / len(test_oids)
    GLOBAL_FEATURE_PREVALENCE[f] = prevalence

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


def compute_probe_efficiency_from_tracking(tracking_list):
    effective = 0
    zero_gain = 0
    harmful = 0
    for pt in tracking_list:
        pre_u = _compute_utility(pt["pre_probs"])
        post_u = _compute_utility(pt["post_probs"])
        delta = post_u - pre_u
        if delta > 0.001:
            effective += 1
        elif delta < -0.001:
            harmful += 1
        else:
            zero_gain += 1
    total = effective + zero_gain + harmful
    return {
        "effective_probe_count": effective,
        "zero_gain_probe_count": zero_gain,
        "harmful_probe_count": harmful,
        "total_probe_count": total,
        "effective_probe_rate": effective / max(total, 1),
    }


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


def pareto_frontier(items):
    nondominated = set()
    for name1, (acc1, cost1) in items.items():
        dominated = False
        for name2, (acc2, cost2) in items.items():
            if name1 == name2:
                continue
            if acc2 >= acc1 and cost2 <= cost1 and (acc2 > acc1 or cost2 < cost1):
                dominated = True
                break
        if not dominated:
            nondominated.add(name1)
    return nondominated


def compute_coverage(view, query_gt):
    """Compute positive/negative coverage of visited objects per query."""
    visited_oids = [oid for oid in view.get_all_object_ids() if view.is_visited(oid)]
    coverage = {}
    for qname, qdata in query_gt.items():
        pos_visited = sum(1 for oid in qdata["positive_oids"] if oid in visited_oids)
        neg_visited = sum(1 for oid in qdata["negative_oids"] if oid in visited_oids)
        coverage[qname] = {
            "positive_visited": pos_visited,
            "positive_total": len(qdata["positive_oids"]),
            "negative_visited": neg_visited,
            "negative_total": len(qdata["negative_oids"]),
        }
    return coverage

# =============================================================================
# 4. C15b Inline (for baseline and policy reference)
# =============================================================================
print("\n[3/7] Defining policy variants...")

class C15b_ProbeReservePolicy(MiniMCPolicy):
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
# 5. C17 Interleaved Policies
# =============================================================================

class C17_InterleavedKPolicy(MiniMCPolicy):
    """C17: Interleaved observe-then-probe with fixed K.

    Each cycle: observe K new objects, then probe one high-VOI visited object.
    Repeat until budget exhausted.

    Uses nearest-first for observation order within each cycle.
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5, K=5):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._K = K
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_count_since_last_probe = 0
        self._first_probe_at_visited_count = None
        self._interleaving_pattern = []

    def reset(self, view):
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_count_since_last_probe = 0
        self._first_probe_at_visited_count = None
        self._interleaving_pattern = []

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _select_probe_target(self, view):
        """Find best probe target among visited+unprobed objects."""
        best_oid = None
        best_voi = 0.0
        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            current_utility = _compute_utility(probs)
            norm_cost = total_cost / max(view.initial_budget, 0.001)

            best_action_voi = 0.0
            best_action = None
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
                    best_action = action

            if best_action_voi > best_voi:
                best_voi = best_action_voi
                best_oid = oid

        return best_oid, best_voi

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited and not [o for o in view.get_all_object_ids()
                                    if view.is_visited(o) and o not in self._probed_oids]:
            return None

        # Try to probe if we've observed K new objects since last probe
        # OR if we can't afford any more observes
        if self._observe_count_since_last_probe >= self._K or not unvisited:
            probe_target, probe_voi = self._select_probe_target(view)
            if probe_target is not None and probe_voi > 0:
                self._observe_count_since_last_probe = 0
                if self._first_probe_at_visited_count is None:
                    self._first_probe_at_visited_count = sum(
                        1 for o in view.get_all_object_ids() if view.is_visited(o))
                self._visit_order.append(("probe", probe_target))
                self._interleaving_pattern.append("P")
                return probe_target
            # If no viable probe target, continue observing if possible
            if not unvisited:
                return None
            self._observe_count_since_last_probe = 0

        # Observe: nearest-first among unvisited
        if not unvisited:
            # Last resort: try probing
            probe_target, probe_voi = self._select_probe_target(view)
            if probe_target is not None and probe_voi > 0:
                self._visit_order.append(("probe", probe_target))
                self._interleaving_pattern.append("P")
                return probe_target
            return None

        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid

        if best_oid is None:
            probe_target, probe_voi = self._select_probe_target(view)
            if probe_target is not None and probe_voi > 0:
                self._visit_order.append(("probe", probe_target))
                self._interleaving_pattern.append("P")
                return probe_target
            return None

        self._observe_count_since_last_probe += 1
        self._visit_order.append(("observe", best_oid))
        self._interleaving_pattern.append("O")
        return best_oid

    def decide_probe(self, view, object_id):
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

    def get_probe_efficiency(self, view):
        return compute_probe_efficiency_from_tracking(self._probe_tracking)

    def get_selected_object_stats(self, view):
        majority = 0
        minority = 0
        per_query_pos = {q: 0 for q in query_gt}
        per_query_neg = {q: 0 for q in query_gt}

        for oid in self._probed_oids:
            meta = object_meta.get(oid, {})
            if meta.get("is_majority", True):
                majority += 1
            else:
                minority += 1
            for qname, qdata in query_gt.items():
                if oid in qdata["positive_oids"]:
                    per_query_pos[qname] += 1
                elif oid in qdata["negative_oids"]:
                    per_query_neg[qname] += 1

        return {
            "selected_probe_objects_majority": majority,
            "selected_probe_objects_minority": minority,
            "selected_probe_objects_per_query_positive": per_query_pos,
            "selected_probe_objects_per_query_negative": per_query_neg,
        }

    def get_timing_stats(self, view):
        """Probe timing and interleaving pattern stats."""
        first_probe_at = self._first_probe_at_visited_count
        total_visited = sum(1 for o in view.get_all_object_ids() if view.is_visited(o))
        total_probed = len(self._probed_oids)
        pattern_str = "".join(self._interleaving_pattern) if self._interleaving_pattern else ""

        return {
            "first_probe_at_visited_count": first_probe_at,
            "total_visited": total_visited,
            "total_probed": total_probed,
            "interleaving_pattern": pattern_str,
            "pattern_length": len(pattern_str),
            "K_value": self._K,
        }


class C17_AdaptiveInterleavedPolicy(C17_InterleavedKPolicy):
    """C17 adaptive: at each step, compare best probe VOI vs continue-observing threshold.

    If best_probe_net_voi > threshold: probe it
    Otherwise: observe next nearest unvisited object
    Continue until budget exhausted.
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5,
                 probe_threshold=0.0):
        super().__init__(instance_memory, rng, cost_weight, K=1)  # K unused
        self._probe_threshold = probe_threshold

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()

        # Check if there's a viable probe target
        probe_target, probe_voi = self._select_probe_target(view)
        has_viable_probe = (probe_target is not None and probe_voi > self._probe_threshold)

        # Check if we can afford to observe
        can_observe = False
        if unvisited:
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total
                    best_oid = oid
            can_observe = best_oid is not None

        # Decision: probe if viable probe exists AND (no observe possible OR probe is better)
        if has_viable_probe and not can_observe:
            self._visit_order.append(("probe", probe_target))
            self._interleaving_pattern.append("P")
            if self._first_probe_at_visited_count is None:
                self._first_probe_at_visited_count = sum(
                    1 for o in view.get_all_object_ids() if view.is_visited(o))
            return probe_target

        if has_viable_probe and can_observe:
            # Both are possible — probe if probe_voi is clearly beneficial
            # Default threshold 0.0: probe whenever net VOI > 0
            self._visit_order.append(("probe", probe_target))
            self._interleaving_pattern.append("P")
            if self._first_probe_at_visited_count is None:
                self._first_probe_at_visited_count = sum(
                    1 for o in view.get_all_object_ids() if view.is_visited(o))
            return probe_target

        # No viable probe: observe if possible
        if can_observe:
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
                self._interleaving_pattern.append("O")
                return best_oid

        return None


class C17_OracleSwitchTimingPolicy(C17_InterleavedKPolicy):
    """DIAGNOSTIC ONLY — NON-DEPLOYABLE.

    Uses oracle knowledge to decide when to switch from observing to probing.
    After each observe, checks if the best probe target object would be
    selected by oracle's optimal object ordering. If so, probes it.

    Purpose: Estimate the upper bound on interleaved timing benefit.
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5,
                 oracle_selected_oids=None):
        super().__init__(instance_memory, rng, cost_weight, K=1)
        self._oracle_selected_oids = set(oracle_selected_oids or [])
        self._label = "diagnostic_non_deployable"

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()

        # Find best probe target
        probe_target, probe_voi = self._select_probe_target(view)

        # Oracle rule: probe if this object is in the oracle-selected set
        if probe_target is not None and probe_target in self._oracle_selected_oids:
            self._visit_order.append(("probe", probe_target))
            self._interleaving_pattern.append("P")
            if self._first_probe_at_visited_count is None:
                self._first_probe_at_visited_count = sum(
                    1 for o in view.get_all_object_ids() if view.is_visited(o))
            return probe_target

        # Otherwise, observe if possible
        if unvisited:
            best_oid = None
            best_cost = float('inf')
            for oid in unvisited:
                total = view.compute_reach_cost(oid) + view.observe_cost
                if view.can_afford(total) and total < best_cost:
                    best_cost = total
                    best_oid = oid
            if best_oid is not None:
                self._visit_order.append(("observe", best_oid))
                self._interleaving_pattern.append("O")
                return best_oid

        return None


# =============================================================================
# 6. Baselines
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
c_minus_preds = {oid: {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5) for f in CORE_ACTION_FEATURES}
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
results["C0b_original"] = {
    "predictions": c0b_preds["per_object"],
    "metrics": compute_full_metrics(c0b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c0b_log, c0b_obs),
    "policy_type": "observe_only_no_peripheral",
}
c0b_mb = results["C0b_original"]["metrics"]["macro_query_balanced_accuracy"]
print(f"  C0b_original macro_bal={c0b_mb:.4f}  visited={results['C0b_original']['cost']['visit_count']}")

# --- C13_instance_VOI_original ---
c13_im = im_base.clone()
c13_policy = C13_InstanceVOIPolicy(c13_im, random.Random(SEED + 600), cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
results["C13_instance_VOI_original"] = {
    "predictions": c13_preds["per_object"],
    "metrics": compute_full_metrics(c13_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c13_log, c13_obs),
    "policy_type": "instance_VOI_no_peripheral",
}
c13_mb = results["C13_instance_VOI_original"]["metrics"]["macro_query_balanced_accuracy"]
c13_visited = results["C13_instance_VOI_original"]["cost"]["visit_count"]
c13_probed = results["C13_instance_VOI_original"]["cost"]["probe_count"]
print(f"  C13_instance_VOI_original macro_bal={c13_mb:.4f}  "
      f"visited={c13_visited}  probed={c13_probed}")

# --- C15b_probe_reserve_original ---
c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_efficiency = compute_probe_efficiency_from_tracking(c15b_policy._probe_tracking)
c15b_probed_oids = set(pt["oid"] for pt in c15b_policy._probe_tracking)
results["C15b_probe_reserve_original"] = {
    "predictions": c15b_preds["per_object"],
    "metrics": compute_full_metrics(c15b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "efficiency": c15b_efficiency,
    "policy_type": "two_pass_VOI_probe_no_peripheral",
}
c15b_mb = results["C15b_probe_reserve_original"]["metrics"]["macro_query_balanced_accuracy"]
c15b_pos_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_positive_recall"]
c15b_neg_recall = results["C15b_probe_reserve_original"]["metrics"]["mean_negative_recall"]
c15b_visited = results["C15b_probe_reserve_original"]["cost"]["visit_count"]
c15b_probed = results["C15b_probe_reserve_original"]["cost"]["probe_count"]
print(f"  C15b_original macro_bal={c15b_mb:.4f}  visited={c15b_visited}  "
      f"probed={c15b_probed}  eff_rate={c15b_efficiency['effective_probe_rate']:.4f}")

# --- C15G from 1J12 (load from JSON) ---
c15g_json_path = os.path.join(CURRENT_DIR, "runs",
                               "calibration_block1j12_global_coarse_scene_info_c4_seed101.json")
c15g_mb = None
if os.path.exists(c15g_json_path):
    with open(c15g_json_path, "r", encoding="utf-8") as f:
        c15g_json = json.load(f)
    c15g_mb = c15g_json["scene_variants"]["C15G_global_coarse_map"]["macro_query_balanced_accuracy"]
    print(f"  C15G_global_coarse_map (from 1J12) macro_bal={c15g_mb:.4f}")
else:
    c15g_mb = 0.5992  # fallback from 1J12 result
    print(f"  C15G_global_coarse_map (1J12 JSON not found, using known value={c15g_mb:.4f})")

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
oracle_c0b_gap = oracle_mb - c0b_mb
print(f"  C_oracle macro_bal={oracle_mb:.4f}  oracle_c0b_gap={oracle_c0b_gap:.4f}")

# =============================================================================
# 7. C17 Variants
# =============================================================================
print("\n[5/7] Running C17 interleaved variants...")
c17_results = {}

# --- C17 K=3, 5, 8 ---
for K in INTERLEAVED_K_VALUES:
    vname = f"C17_interleaved_K{K}"
    print(f"  {vname}...")
    c17k_im = im_base.clone()
    c17k_policy = C17_InterleavedKPolicy(
        c17k_im, random.Random(SEED + 1000 + K),
        cost_weight=COST_WEIGHT, K=K)
    c17k_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
    c17k_preds, c17k_log, c17k_obs, _, _ = run_policy(c17k_policy, c17k_env)
    c17k_metrics = compute_full_metrics(c17k_preds["per_object"], query_gt)
    c17k_cost = get_cost_metrics(c17k_log, c17k_obs)
    c17k_efficiency = c17k_policy.get_probe_efficiency(c17k_obs)
    c17k_obj_stats = c17k_policy.get_selected_object_stats(c17k_obs)
    c17k_timing = c17k_policy.get_timing_stats(c17k_obs)
    c17k_mb = c17k_metrics["macro_query_balanced_accuracy"]

    # Coverage at first probe
    first_probe_visited = c17k_timing["first_probe_at_visited_count"] or 0

    print(f"    macro_bal={c17k_mb:.4f}  visited={c17k_cost['visit_count']}  "
          f"probed={c17k_cost['probe_count']}  ncost={c17k_cost['normalized_cost']:.4f}")
    print(f"    eff_rate={c17k_efficiency['effective_probe_rate']:.4f}  "
          f"eff={c17k_efficiency['effective_probe_count']}  "
          f"zero={c17k_efficiency['zero_gain_probe_count']}  "
          f"harmful={c17k_efficiency['harmful_probe_count']}")
    print(f"    majority_probed={c17k_obj_stats['selected_probe_objects_majority']}  "
          f"minority_probed={c17k_obj_stats['selected_probe_objects_minority']}")
    print(f"    first_probe_at_visited={first_probe_visited}  "
          f"pattern={c17k_timing['interleaving_pattern'][:60]}...")

    c17_results[vname] = {
        "predictions": c17k_preds["per_object"],
        "metrics": c17k_metrics,
        "cost": c17k_cost,
        "efficiency": c17k_efficiency,
        "object_stats": c17k_obj_stats,
        "timing": c17k_timing,
        "policy_type": "C17_interleaved_K",
        "K": K,
        "probed_object_ids": [pt["oid"] for pt in c17k_policy._probe_tracking],
        "probed_actions": {pt["oid"]: pt["action"] for pt in c17k_policy._probe_tracking},
        "probe_outcomes": {pt["oid"]: pt["outcome"] for pt in c17k_policy._probe_tracking},
    }

# --- C17 adaptive ---
print("  C17_adaptive_interleaved...")
c17a_im = im_base.clone()
c17a_policy = C17_AdaptiveInterleavedPolicy(
    c17a_im, random.Random(SEED + 1100),
    cost_weight=COST_WEIGHT, probe_threshold=0.0)
c17a_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c17a_preds, c17a_log, c17a_obs, _, _ = run_policy(c17a_policy, c17a_env)
c17a_metrics = compute_full_metrics(c17a_preds["per_object"], query_gt)
c17a_cost = get_cost_metrics(c17a_log, c17a_obs)
c17a_efficiency = c17a_policy.get_probe_efficiency(c17a_obs)
c17a_obj_stats = c17a_policy.get_selected_object_stats(c17a_obs)
c17a_timing = c17a_policy.get_timing_stats(c17a_obs)
c17a_mb = c17a_metrics["macro_query_balanced_accuracy"]
c17a_first_probe_visited = c17a_timing["first_probe_at_visited_count"] or 0

print(f"    macro_bal={c17a_mb:.4f}  visited={c17a_cost['visit_count']}  "
      f"probed={c17a_cost['probe_count']}  ncost={c17a_cost['normalized_cost']:.4f}")
print(f"    eff_rate={c17a_efficiency['effective_probe_rate']:.4f}  "
      f"eff={c17a_efficiency['effective_probe_count']}  "
      f"zero={c17a_efficiency['zero_gain_probe_count']}  "
      f"harmful={c17a_efficiency['harmful_probe_count']}")
print(f"    majority_probed={c17a_obj_stats['selected_probe_objects_majority']}  "
      f"minority_probed={c17a_obj_stats['selected_probe_objects_minority']}")
print(f"    first_probe_at_visited={c17a_first_probe_visited}  "
      f"pattern={c17a_timing['interleaving_pattern'][:60]}...")

c17_results["C17_adaptive_interleaved"] = {
    "predictions": c17a_preds["per_object"],
    "metrics": c17a_metrics,
    "cost": c17a_cost,
    "efficiency": c17a_efficiency,
    "object_stats": c17a_obj_stats,
    "timing": c17a_timing,
    "policy_type": "C17_adaptive_interleaved",
    "probe_threshold": 0.0,
    "probed_object_ids": [pt["oid"] for pt in c17a_policy._probe_tracking],
    "probed_actions": {pt["oid"]: pt["action"] for pt in c17a_policy._probe_tracking},
    "probe_outcomes": {pt["oid"]: pt["outcome"] for pt in c17a_policy._probe_tracking},
}

# --- C17 oracle switch timing (diagnostic, non-deployable) ---
print("  C17_oracle_switch_timing_upper_bound (non-deployable)...")

# Compute oracle-selected objects (same method as 1J12 oracle UB)
def compute_oracle_selected_objects(sim_gt, positions, budget):
    oracle_preds = C14a_TruthAnswerOracle(sim_gt).get_answer()
    oracle_per_object = oracle_preds["per_object"]
    object_scores = {}
    for oid in test_oids:
        o_probs = oracle_per_object[oid]
        o_utility = _compute_utility(o_probs)
        prior_probs = {f: GLOBAL_FEATURE_PREVALENCE.get(f, 0.5) for f in CORE_ACTION_FEATURES}
        prior_utility = _compute_utility(prior_probs)
        object_scores[oid] = o_utility - prior_utility

    sorted_oids = sorted(object_scores.keys(), key=lambda o: object_scores[o], reverse=True)
    selected = []
    total_cost = 0.0
    agent_pos = config.AGENT_START
    for oid in sorted_oids:
        pos = positions[oid]
        reach_cost = manhattan(agent_pos, pos) * config.REACH_COST_PER_UNIT
        step_cost = reach_cost + config.OBSERVE_COST + config.PROBE_COST
        if total_cost + step_cost <= budget:
            selected.append(oid)
            total_cost += step_cost
            agent_pos = pos
    return selected

oracle_selected_oids = compute_oracle_selected_objects(sim_gt, positions, BUDGET)
print(f"    oracle_selected_oids: {len(oracle_selected_oids)} objects")

c17o_im = im_base.clone()
c17o_policy = C17_OracleSwitchTimingPolicy(
    c17o_im, random.Random(SEED + 1200),
    cost_weight=COST_WEIGHT, oracle_selected_oids=oracle_selected_oids)
c17o_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c17o_preds, c17o_log, c17o_obs, _, _ = run_policy(c17o_policy, c17o_env)
c17o_metrics = compute_full_metrics(c17o_preds["per_object"], query_gt)
c17o_cost = get_cost_metrics(c17o_log, c17o_obs)
c17o_efficiency = c17o_policy.get_probe_efficiency(c17o_obs)
c17o_obj_stats = c17o_policy.get_selected_object_stats(c17o_obs)
c17o_timing = c17o_policy.get_timing_stats(c17o_obs)
c17o_mb = c17o_metrics["macro_query_balanced_accuracy"]
c17o_first_probe_visited = c17o_timing["first_probe_at_visited_count"] or 0

print(f"    macro_bal={c17o_mb:.4f}  visited={c17o_cost['visit_count']}  "
      f"probed={c17o_cost['probe_count']}  ncost={c17o_cost['normalized_cost']:.4f}")
print(f"    pattern={c17o_timing['interleaving_pattern'][:60]}...")

c17_results["C17_oracle_switch_timing_upper_bound"] = {
    "predictions": c17o_preds["per_object"],
    "metrics": c17o_metrics,
    "cost": c17o_cost,
    "efficiency": c17o_efficiency,
    "object_stats": c17o_obj_stats,
    "timing": c17o_timing,
    "policy_type": "C17_oracle_switch_timing",
    "label": "diagnostic_non_deployable",
    "probed_object_ids": [pt["oid"] for pt in c17o_policy._probe_tracking],
    "probed_actions": {pt["oid"]: pt["action"] for pt in c17o_policy._probe_tracking},
    "probe_outcomes": {pt["oid"]: pt["outcome"] for pt in c17o_policy._probe_tracking},
}

# =============================================================================
# 8. Comparisons
# =============================================================================
print("\n[6/7] Computing comparisons...")
comparisons = {}

# Find best C17 variant (deployable only)
deployable_c17 = {k: v for k, v in c17_results.items()
                  if v["policy_type"] in ("C17_interleaved_K", "C17_adaptive_interleaved")}
best_c17_name = max(deployable_c17, key=lambda k: deployable_c17[k]["metrics"]["macro_query_balanced_accuracy"])
best_c17 = deployable_c17[best_c17_name]
best_c17_mb = best_c17["metrics"]["macro_query_balanced_accuracy"]

print(f"  Best C17 variant: {best_c17_name} (macro_bal={best_c17_mb:.4f})")

# 1. Each C17 vs C15b
for vname, vdata in c17_results.items():
    comparisons[f"{vname}_vs_C15b"] = {
        "delta_macro_bal": round(vdata["metrics"]["macro_query_balanced_accuracy"] - c15b_mb, 6),
        "delta_positive_recall": round(
            vdata["metrics"]["mean_positive_recall"] - c15b_pos_recall, 6),
        "delta_negative_recall": round(
            vdata["metrics"]["mean_negative_recall"] - c15b_neg_recall, 6),
        "delta_visit_count": vdata["cost"]["visit_count"] - c15b_visited,
        "delta_probe_count": vdata["cost"]["probe_count"] - c15b_probed,
        "delta_effective_probe_rate": round(
            vdata["efficiency"]["effective_probe_rate"] - c15b_efficiency["effective_probe_rate"], 6),
        "pareto_relation": "C17_dominates" if (
            vdata["metrics"]["macro_query_balanced_accuracy"] > c15b_mb and
            vdata["cost"]["normalized_cost"] < results["C15b_probe_reserve_original"]["cost"]["normalized_cost"]
        ) else "C15b_dominates" if (
            c15b_mb > vdata["metrics"]["macro_query_balanced_accuracy"] and
            results["C15b_probe_reserve_original"]["cost"]["normalized_cost"] < vdata["cost"]["normalized_cost"]
        ) else "tradeoff",
    }

# 2. Best C17 vs C13
c17_vs_c13 = {
    "C17_preserves_more_breadth_than_C13": best_c17["cost"]["visit_count"] > c13_visited,
    "C17_avoids_C13_narrow_depth_failure": best_c17_mb > c13_mb,
    "C17_visit_count_vs_C13": best_c17["cost"]["visit_count"] - c13_visited,
    "C17_probe_count_vs_C13": best_c17["cost"]["probe_count"] - c13_probed,
}
comparisons["best_C17_vs_C13"] = c17_vs_c13

# 3. Best C17 vs C15G
c17_vs_c15g = {
    "delta_macro_bal": round(best_c17_mb - c15g_mb, 6) if c15g_mb else None,
    "temporal_interleaving_more_useful_than_global_coarse_map": (
        best_c17_mb > c15g_mb if c15g_mb else None),
}
comparisons["best_C17_vs_C15G"] = c17_vs_c15g

# 4. Gap capture
target_10pct = 0.5871 + 0.0413  # 0.6284
best_c17_hits_10pct = best_c17_mb >= target_10pct
best_c17_gap_capture = (best_c17_mb - 0.5871) / 0.4129

comparisons["gap_capture"] = {
    "C0b_base": 0.5871,
    "oracle_ceiling": 1.0000,
    "target_10pct_threshold": target_10pct,
    "C15b_gap_capture_pct": round((c15b_mb - 0.5871) / 0.4129 * 100, 2),
    f"best_C17_gap_capture_pct": round(best_c17_gap_capture * 100, 2),
    f"best_C17_hits_10pct_threshold": best_c17_hits_10pct,
}

# 5. C17 vs oracle switch timing
oracle_switch = c17_results.get("C17_oracle_switch_timing_upper_bound")
if oracle_switch:
    comparisons["C17_vs_oracle_switch_timing"] = {
        "delta_macro_bal": round(best_c17_mb - oracle_switch["metrics"]["macro_query_balanced_accuracy"], 6),
        "oracle_switch_macro_bal": oracle_switch["metrics"]["macro_query_balanced_accuracy"],
        "remaining_headroom": round(
            oracle_switch["metrics"]["macro_query_balanced_accuracy"] - best_c17_mb, 6),
    }

# Pareto frontier
deployable_items = {}
for name, r in results.items():
    if r.get("policy_type") in ("observe_only_no_peripheral", "instance_VOI_no_peripheral",
                                 "two_pass_VOI_probe_no_peripheral"):
        deployable_items[name] = (r["metrics"]["macro_query_balanced_accuracy"], r["cost"]["normalized_cost"])
for vname, vdata in deployable_c17.items():
    deployable_items[vname] = (vdata["metrics"]["macro_query_balanced_accuracy"], vdata["cost"]["normalized_cost"])

nondominated = pareto_frontier(deployable_items)
best_c17_on_frontier = best_c17_name in nondominated
comparisons["pareto"] = {
    "frontier_policies": sorted(nondominated),
    "best_C17_on_frontier": best_c17_on_frontier,
}

# 6. Interleaving pattern comparison
comparisons["interleaving_patterns"] = {
    f"C17_interleaved_K{K}": c17_results[f"C17_interleaved_K{K}"]["timing"]["interleaving_pattern"]
    for K in INTERLEAVED_K_VALUES
}
comparisons["interleaving_patterns"]["C17_adaptive"] = c17a_timing["interleaving_pattern"]
if oracle_switch:
    comparisons["interleaving_patterns"]["C17_oracle_switch"] = c17o_timing["interleaving_pattern"]

# =============================================================================
# 9. Interpretation
# =============================================================================
print("\n[7/7] Interpreting results...")

best_c17_pos_recall = best_c17["metrics"]["mean_positive_recall"]
improves_over_c15b = best_c17_mb > c15b_mb
improves_pos_recall = best_c17_pos_recall > c15b_pos_recall
improves_minority = best_c17["object_stats"]["selected_probe_objects_minority"] > 2

c17_preserves_breadth = c17_vs_c13["C17_preserves_more_breadth_than_C13"]
c17_beats_c13 = c17_vs_c13["C17_avoids_C13_narrow_depth_failure"]

if improves_over_c15b and best_c17_hits_10pct:
    interleaving_interp = (
        "Interleaved breadth-depth is strongly supported. "
        "Candidate for small multi-seed."
    )
    interleaving_supported = True
    candidate_multiseed = True
elif improves_over_c15b and not best_c17_hits_10pct:
    interleaving_interp = (
        "Interleaving helps but remains weak — above C15b but below 10% threshold."
    )
    interleaving_supported = True
    candidate_multiseed = False
elif abs(best_c17_mb - c15b_mb) < 0.001:
    interleaving_interp = (
        "Strict two-pass is already sufficient under current signals. "
        "Bottleneck likely probe informativeness / IOM update, not timing."
    )
    interleaving_supported = False
    candidate_multiseed = False
elif c17_preserves_breadth and c17_beats_c13:
    interleaving_interp = (
        "Interleaving preserves more breadth than C13 and avoids narrow-depth failure, "
        "but does not beat C15b. C15b's two-pass with reserve is well-calibrated."
    )
    interleaving_supported = False
    candidate_multiseed = False
else:
    interleaving_interp = (
        "Interleaving hurts breadth or wastes probes. Do not pursue."
    )
    interleaving_supported = False
    candidate_multiseed = False

print(f"  Best C17 variant: {best_c17_name}")
print(f"  Best C17 macro_bal: {best_c17_mb:.4f}")
print(f"  Best C17 vs C15b delta: {best_c17_mb - c15b_mb:.6f}")
print(f"  improves_over_c15b: {improves_over_c15b}")
print(f"  improves_pos_recall: {improves_pos_recall}")
print(f"  improves_minority_targeting: {improves_minority}")
print(f"  preserves_breadth_vs_C13: {c17_preserves_breadth}")
print(f"  beats_C13: {c17_beats_c13}")
print(f"  hits_10pct: {best_c17_hits_10pct}")
print(f"  best_c17_on_frontier: {best_c17_on_frontier}")
print(f"  interleaving_supported: {interleaving_supported}")
print(f"  interpretation: {interleaving_interp}")

# =============================================================================
# 10. Save outputs
# =============================================================================

output = {
    "block_id": "1J13",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Interleaved breadth-depth diagnostic. Tests whether interleaving observation and probing improves over strict two-pass C15b. Single seed.",
    "design_validation": {
        "no_hidden_subtype_access": True,
        "no_true_affordance_access": True,
        "no_hidden_category_access": True,
        "no_query_label_access": True,
        "no_route_f_usage": True,
        "no_global_coarse_map_usage": True,
        "oracle_variants_labeled_non_deployable": True,
        "multi_seed_deferred": True,
    },
    "baselines": {
        name: {
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
        for name, r in results.items()
    },
    "c17_variants": {
        vname: {
            "macro_query_balanced_accuracy": vdata["metrics"]["macro_query_balanced_accuracy"],
            "macro_query_accuracy": vdata["metrics"]["macro_query_accuracy"],
            "mean_positive_recall": vdata["metrics"]["mean_positive_recall"],
            "mean_negative_recall": vdata["metrics"]["mean_negative_recall"],
            "per_query": vdata["metrics"]["per_query"],
            "visit_count": vdata["cost"]["visit_count"],
            "observe_count": vdata["cost"]["observe_count"],
            "probe_count": vdata["cost"]["probe_count"],
            "total_cost": vdata["cost"]["total_cost"],
            "normalized_cost": vdata["cost"]["normalized_cost"],
            "policy_type": vdata["policy_type"],
            "effective_probe_count": vdata["efficiency"]["effective_probe_count"],
            "zero_gain_probe_count": vdata["efficiency"]["zero_gain_probe_count"],
            "harmful_probe_count": vdata["efficiency"]["harmful_probe_count"],
            "effective_probe_rate": vdata["efficiency"]["effective_probe_rate"],
            "selected_probe_objects_majority": vdata["object_stats"]["selected_probe_objects_majority"],
            "selected_probe_objects_minority": vdata["object_stats"]["selected_probe_objects_minority"],
            "first_probe_at_visited_count": vdata["timing"]["first_probe_at_visited_count"],
            "interleaving_pattern": vdata["timing"]["interleaving_pattern"],
            "probed_object_ids": vdata["probed_object_ids"],
            "probed_actions": vdata["probed_actions"],
            "probe_outcomes": vdata["probe_outcomes"],
            "K": vdata.get("K"),
            "probe_threshold": vdata.get("probe_threshold"),
            "label": vdata.get("label"),
        }
        for vname, vdata in c17_results.items()
    },
    "best_c17_variant": best_c17_name,
    "comparisons": comparisons,
    "interpretation": {
        "best_C17_variant": best_c17_name,
        "best_C17_macro_bal": best_c17_mb,
        "best_C17_delta_vs_C15b": round(best_c17_mb - c15b_mb, 6),
        "best_C17_improves_over_C15b": improves_over_c15b,
        "best_C17_improves_pos_recall": improves_pos_recall,
        "best_C17_improves_minority_targeting": improves_minority,
        "best_C17_preserves_breadth_vs_C13": c17_preserves_breadth,
        "best_C17_beats_C13": c17_beats_c13,
        "best_C17_hits_10pct_threshold": best_c17_hits_10pct,
        "best_C17_on_pareto_frontier": best_c17_on_frontier,
        "interleaving_supported": interleaving_supported,
        "candidate_ready_for_small_multiseed": candidate_multiseed,
        "ready_for_multiseed": False,
        "interpretation": interleaving_interp,
    },
}

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j13_interleaved_breadth_depth_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J13")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"c15b_macro_bal={c15b_mb:.6f}")
print(f"best_c17_variant={best_c17_name}")
print(f"best_c17_macro_bal={best_c17_mb:.6f}")
print(f"best_c17_delta_vs_c15b={best_c17_mb - c15b_mb:.6f}")
print(f"best_c17_gap_capture_vs_c0b={best_c17_gap_capture:.6f}")
print(f"best_c17_positive_recall={best_c17_pos_recall:.6f}")
print(f"best_c17_probe_count={best_c17['cost']['probe_count']}")
print(f"best_c17_visit_count={best_c17['cost']['visit_count']}")
print(f"interleaving_supported={interleaving_supported}")
print(f"candidate_ready_for_small_multiseed={candidate_multiseed}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
