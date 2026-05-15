"""
Block 1J7 — C15B Cross-Action Diagnostic VOI Evaluation on C4_instance_subtype_cued_v1.

Adds Route B (differentiating-action / cross-action VOI) on top of C15 two-pass.
Pass 1: observe broadly (same as C15b). Pass 2: probe object-action pairs
ranked by diagnostic score (uncertainty × query_relevance × boundary_relevance
× neighbor_variance / cost).
"""
import os
import sys
import json
import copy
import random
import time

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
from evaluator import MiniMCEvaluator
from agent_obs import AgentObs
from event_log import EventLog
from simulator_truth import MiniMCSimulatorTruth
from policies import (
    MiniMCPolicy, C0a_NoInteractionPolicy, C0b_ObserveOnlyPolicy,
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
from objects import AFFORDANCE_PROFILES

t0 = time.time()

SEED = 101
BUDGET = 1.5
COST_WEIGHT = 0.5
COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J7 — C15B Cross-Action Diagnostic VOI Evaluation")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}  CW={COST_WEIGHT}")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM
# =============================================================================
print("\n[1/6] Phase A: Training + IOM building...")
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
# 2. Test objects + environment
# =============================================================================
print("\n[2/6] Generating subtype test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
print(f"  test_objects={len(test_objects)}  positions assigned")

sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

CATEGORY_DIFFERENTIATING_ACTION = {
    "wood_log": "craft_plank",
    "stone_block": "mine_by_hand",
    "apple": "eat",
    "wooden_pickaxe": "use_as_tool",
}

ACTION_TO_QUERY = {
    "mine_by_hand": "need_stone",
    "mine_with_pickaxe": "need_stone",
    "craft_plank": "need_planks",
    "eat": "need_food",
    "use_as_tool": "need_tool",
    "burn_as_fuel": "need_fuel",
}

object_meta = {}
for oid in test_oids:
    obj = test_objects[oid]
    cat = obj["hidden_category"]
    subtype = obj["hidden_subtype"]
    gt_aff = sim_gt.get_ground_truth_affordances(oid)
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    ratios = subtype_def["subtype_ratio"]
    is_majority = ratios.get(subtype, 0.0) >= 0.5
    diff_action = CATEGORY_DIFFERENTIATING_ACTION.get(cat, None)
    query_membership = {}
    for qname in query_gt:
        query_membership[qname] = oid in query_gt[qname]["positive_oids"]
    object_meta[oid] = {
        "category": cat, "subtype": subtype, "is_majority": is_majority,
        "gt_affordances": gt_aff, "differentiating_action": diff_action,
        "query_membership": query_membership,
    }

# =============================================================================
# 3. C15B Cross-Action Diagnostic VOI Policy
# =============================================================================

def _predict_query_prob(probs, query_name):
    """Continuous probability that query is true given affordance probabilities."""
    if query_name == "need_planks":
        return probs.get("craft_plank_success", 0.5)
    elif query_name == "need_stone":
        p_mbh = probs.get("mine_by_hand_success", 0.5)
        p_mwp = probs.get("mine_with_pickaxe_success", 0.5)
        return (1.0 - p_mbh) * p_mwp
    elif query_name == "need_food":
        return probs.get("eat_success", 0.5)
    elif query_name == "need_tool":
        return probs.get("use_as_tool_success", 0.5)
    elif query_name == "need_fuel":
        return probs.get("burn_as_fuel_success", 0.5)
    return 0.5


class C15B_CrossActionDiagnosticVOIPolicy(MiniMCPolicy):
    """Two-pass C15 + Route B: cross-action diagnostic VOI scoring.

    Pass 1: observe broadly (C15b-style, nearest-first, budget-reserve rule).
    Pass 2: rank object-action pairs by diagnostic score and probe best.

    Diagnostic score = query_relevance * outcome_uncertainty
                        * decision_boundary_relevance * neighbor_variance
                        / normalized_cost
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
        self._pre_decision_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0

        # C15B-specific tracking
        self._all_scored_pairs = []       # all (oid, action, score) evaluated
        self._selected_scores = []        # scores for pairs actually probed
        self._selected_action = None      # action for current target
        self._observed_object_count = 0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_scored_pairs = []
        self._selected_scores = []
        self._selected_action = None
        self._observed_object_count = 0

    def _diagnostic_score(self, oid, action, features, view):
        """Score an object-action pair for diagnostic probing value.

        Components (all realizable, no hidden info):
          1. action_outcome_uncertainty = p * (1-p)
          2. query_relevance = 1.0 if action maps to a task query
          3. decision_boundary_relevance = 1 - 2*|p_query - 0.5|
          4. training_neighbor_variance = outcome_variance among IOM neighbors
          5. cost penalty = 1 / normalized_total_cost
        """
        fake_obj = {"id": oid, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)

        # 1. Outcome uncertainty
        uncertainty = p_success * (1.0 - p_success)
        if uncertainty < 0.001:
            uncertainty = 0.001

        # 2. Query relevance
        query_name = ACTION_TO_QUERY.get(action)
        query_rel = 1.0 if query_name else 0.0

        # 3. Decision boundary relevance
        boundary_rel = 0.0
        if query_name:
            query_prob = _predict_query_prob(probs, query_name)
            boundary_rel = 1.0 - 2.0 * abs(query_prob - 0.5)
            boundary_rel = max(boundary_rel, 0.01)

        # 4. Training neighbor variance
        _, _, info = self._im.predict_outcome(fake_obj, action)
        neighbor_var = info.get("outcome_variance", 0.0)
        neighbor_var = max(neighbor_var, 0.05)  # floor so all actions get some signal

        # 5. Cost penalty
        total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
        norm_cost = total_cost / max(view.initial_budget, 0.001)
        if norm_cost < 0.001:
            norm_cost = 0.001

        score = query_rel * uncertainty * boundary_rel * neighbor_var / norm_cost
        return score

    def _should_transition_to_probe_phase(self, view):
        """C15b transition rule: reserve 25% budget for probes."""
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited
        )
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
                self._observe_pass_cost = view.budget_spent
                self._remaining_budget_after_observe = view.budget_remaining
                self._observed_object_count = self._observe_pass_visit_count
                return self._select_probe_target(view)

            # C0b-like nearest-first
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
        """Select best (object, action) pair by diagnostic score."""
        best_oid = None
        best_action = None
        best_score = float('-inf')

        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue

            features = view.get_observed_features(oid)
            if features is None:
                continue

            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue

            for action in MAIN_CANDIDATE_ACTIONS:
                score = self._diagnostic_score(oid, action, features, view)
                self._all_scored_pairs.append({
                    "oid": oid, "action": action, "score": score,
                })
                if score > best_score:
                    best_score = score
                    best_oid = oid
                    best_action = action

        self._selected_action = best_action
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

        if self._selected_action is None:
            return False, None

        # Recompute the diagnostic score for the selected action to gate
        score = self._diagnostic_score(object_id, self._selected_action, features, view)
        if score <= 0:
            return False, None

        self._selected_scores.append(score)
        self._probed_oids.add(object_id)
        return True, self._selected_action

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

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =============================================================================
# 3b. C15B Oracle Diff-Action Upper Bound (diagnostic only)
# =============================================================================
class C15B_OracleDiffActionUpperBound(C15B_CrossActionDiagnosticVOIPolicy):
    """Oracle upper bound: probes the KNOWN differentiating action per category.

    Diagnostic only — uses hidden_category to look up differentiating action.
    NOT a deployable policy. Excluded from normal-policy Pareto frontier.
    """

    def __init__(self, instance_memory, rng, cost_weight=0.5,
                 category_diff_action=None):
        super().__init__(instance_memory, rng, cost_weight)
        self._cat_diff_action = category_diff_action or {}

    def _select_probe_target(self, view):
        """Rank objects by uncertainty of their category's differentiating action."""
        best_oid = None
        best_score = float('-inf')

        for oid in view.get_all_object_ids():
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue

            # Use hidden_category to pick the differentiating action — ORACLE LEAK
            cat = object_meta[oid]["category"]
            diff_action = self._cat_diff_action.get(cat)
            if diff_action is None:
                continue

            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            p_success = probs.get(ACTION_TO_FEATURE[diff_action], 0.5)
            uncertainty = p_success * (1.0 - p_success)
            norm_cost = total_cost / max(view.initial_budget, 0.001)
            score = uncertainty / (norm_cost + 0.001)

            if score > best_score:
                best_score = score
                best_oid = oid
                self._selected_action = diff_action

        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid


# =============================================================================
# 4. RandomProbeTrackedPolicy (same as 1J6)
# =============================================================================
class RandomProbeTrackedPolicy(MiniMCPolicy):
    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng
        self.probe_tracking = []
        self.visit_order = []

    def reset(self, view):
        self.probe_tracking = []
        self.visit_order = []

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid
        if best_oid is not None:
            self.visit_order.append(best_oid)
        return best_oid

    def decide_probe(self, view, object_id):
        return True, self._rng.choice(MAIN_CANDIDATE_ACTIONS)

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self.probe_tracking.append({
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


# =============================================================================
# 5. Metric helpers
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
            if pred_true and actual_true:
                tp += 1
            elif pred_true and not actual_true:
                fp += 1
            elif not pred_true and not actual_true:
                tn += 1
            else:
                fn += 1
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


def get_cost_metrics(event_log, agent_obs):
    events = event_log.to_list()
    total_reach = sum(e.get("cost", 0.0) for e in events if e.get("event") == "reach")
    total_observe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "observe")
    total_probe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "probe")
    total_cost = event_log.total_cost()
    n_visited = sum(1 for oid in agent_obs.get_all_object_ids() if agent_obs.is_visited(oid))
    n_probed = sum(1 for oid in agent_obs.get_all_object_ids() if agent_obs.get_probe_results(oid))
    init_budget = agent_obs.initial_budget
    return {
        "visit_count": n_visited,
        "observe_count": n_visited,
        "probe_count": n_probed,
        "total_cost": total_cost,
        "normalized_cost": total_cost / max(init_budget, 0.001),
        "total_reach_cost": total_reach,
        "total_observe_cost": total_observe,
        "total_probe_cost": total_probe,
        "budget_remaining": agent_obs.budget_remaining,
        "budget_spent": agent_obs.budget_spent,
        "initial_budget": init_budget,
    }


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 6. Run all policies
# =============================================================================
print("\n[3/6] Running all policies...")

all_results = {}

# --- all_false ---
print("  all_false (all predictions = False)...")
all_false_preds = {}
for oid in test_oids:
    all_false_preds[oid] = {f: 0.0 for f in CORE_ACTION_FEATURES}
all_false_metrics = compute_full_metrics(all_false_preds, query_gt)
all_results["all_false"] = {
    "metrics": all_false_metrics,
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0,
             "total_reach_cost": 0.0, "total_observe_cost": 0.0,
             "total_probe_cost": 0.0, "budget_remaining": BUDGET,
             "budget_spent": 0.0, "initial_budget": BUDGET},
}

# --- C_minus_prior_only ---
print("  C_minus_prior_only...")
c_minus_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c_minus_policy = C0a_NoInteractionPolicy(im_base.clone(), random.Random(SEED))
c_minus_preds, c_minus_log, c_minus_obs, _, _ = run_policy(c_minus_policy, c_minus_env)
c_minus_metrics = compute_full_metrics(c_minus_preds["per_object"], query_gt)
all_results["C_minus_prior_only"] = {
    "metrics": c_minus_metrics, "cost": get_cost_metrics(c_minus_log, c_minus_obs)}
print(f"    macro_bal={c_minus_metrics['macro_query_balanced_accuracy']:.4f}")

# --- C0b_observe_only ---
print("  C0b_observe_only...")
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(im_base.clone(), random.Random(SEED + 100))
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(c0b_preds["per_object"], query_gt)
c0b_cost = get_cost_metrics(c0b_log, c0b_obs)
all_results["C0b_observe_only"] = {"metrics": c0b_metrics, "cost": c0b_cost}
print(f"    macro_bal={c0b_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c0b_cost['visit_count']}  ncost={c0b_cost['normalized_cost']:.4f}")

# --- random_probe_budgeted ---
print("  random_probe_budgeted...")
rprobe_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
rprobe_policy = RandomProbeTrackedPolicy(im_base.clone(), random.Random(SEED + 300))
rprobe_preds, rprobe_log, rprobe_obs, _, _ = run_policy(rprobe_policy, rprobe_env)
rprobe_metrics = compute_full_metrics(rprobe_preds["per_object"], query_gt)
rprobe_cost = get_cost_metrics(rprobe_log, rprobe_obs)
all_results["random_probe_budgeted"] = {
    "metrics": rprobe_metrics, "cost": rprobe_cost}
print(f"    macro_bal={rprobe_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={rprobe_cost['visit_count']}  probed={rprobe_cost['probe_count']}")

# --- C13_instance_VOI_original ---
print("  C13_instance_VOI (CW=0.5)...")
c13_im = im_base.clone()
c13_policy = C13_InstanceVOIPolicy(
    c13_im, random.Random(SEED + 500 + int(COST_WEIGHT * 100)),
    cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
c13_metrics = compute_full_metrics(c13_preds["per_object"], query_gt)
c13_cost = get_cost_metrics(c13_log, c13_obs)
all_results["C13_instance_VOI_original"] = {"metrics": c13_metrics, "cost": c13_cost}
print(f"    macro_bal={c13_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c13_cost['visit_count']}  probed={c13_cost['probe_count']}  "
      f"ncost={c13_cost['normalized_cost']:.4f}")

# --- C15b_probe_reserve ---
print("  C15b_probe_reserve...")
c15b_im = im_base.clone()
# Reuse C15_TwoPassObserveThenProbePolicy from 1J6 pattern
# We inline the definition here for self-containedness
class C15b_ProbeReservePolicy(MiniMCPolicy):
    """C15b: Two-pass, reserve 25% for probes. Original C15 VOI-based probe selection."""

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
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0

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
                self._observe_pass_cost = view.budget_spent
                self._remaining_budget_after_observe = view.budget_remaining
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
            if best_action_voi > best_score:
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


c15b_policy = C15b_ProbeReservePolicy(
    c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, c15b_pre_p, c15b_pre_d = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(c15b_preds["per_object"], query_gt)
c15b_cost = get_cost_metrics(c15b_log, c15b_obs)
all_results["C15b_probe_reserve"] = {"metrics": c15b_metrics, "cost": c15b_cost}
print(f"    macro_bal={c15b_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c15b_cost['visit_count']}  probed={c15b_cost['probe_count']}  "
      f"ncost={c15b_cost['normalized_cost']:.4f}")

# --- C15B_cross_action_diagnostic_VOI ---
print("  C15B_cross_action_diagnostic_VOI...")
c15B_im = im_base.clone()
c15B_policy = C15B_CrossActionDiagnosticVOIPolicy(
    c15B_im, random.Random(SEED + 800), cost_weight=COST_WEIGHT)
c15B_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15B_preds, c15B_log, c15B_obs, c15B_pre_p, c15B_pre_d = run_policy(c15B_policy, c15B_env)
c15B_metrics = compute_full_metrics(c15B_preds["per_object"], query_gt)
c15B_cost = get_cost_metrics(c15B_log, c15B_obs)
all_results["C15B_cross_action_diagnostic_VOI"] = {
    "metrics": c15B_metrics, "cost": c15B_cost}
print(f"    macro_bal={c15B_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c15B_cost['visit_count']}  probed={c15B_cost['probe_count']}  "
      f"ncost={c15B_cost['normalized_cost']:.4f}")

# --- C15B_oracle_diff_action_upper_bound (diagnostic) ---
print("  C15B_oracle_diff_action_upper_bound (diagnostic, oracle)...")
c15B_oracle_im = im_base.clone()
c15B_oracle_policy = C15B_OracleDiffActionUpperBound(
    c15B_oracle_im, random.Random(SEED + 900), cost_weight=COST_WEIGHT,
    category_diff_action=CATEGORY_DIFFERENTIATING_ACTION)
c15B_oracle_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15B_oracle_preds, c15B_oracle_log, c15B_oracle_obs, _, _ = run_policy(
    c15B_oracle_policy, c15B_oracle_env)
c15B_oracle_metrics = compute_full_metrics(c15B_oracle_preds["per_object"], query_gt)
c15B_oracle_cost = get_cost_metrics(c15B_oracle_log, c15B_oracle_obs)
print(f"    macro_bal={c15B_oracle_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c15B_oracle_cost['visit_count']}  probed={c15B_oracle_cost['probe_count']}  "
      f"ncost={c15B_oracle_cost['normalized_cost']:.4f}")

# --- C_oracle_full_information ---
print("  C_oracle_full_information (truth answer)...")
c_oracle = C14a_TruthAnswerOracle(sim_gt)
c_oracle_preds = c_oracle.get_answer()
oracle_metrics = compute_full_metrics(c_oracle_preds["per_object"], query_gt)
print(f"    macro_bal={oracle_metrics['macro_query_balanced_accuracy']:.4f}")

# --- category_majority diagnostic ---
print("  category_majority (diagnostic)...")
cat_majority_preds = {}
for oid in test_oids:
    cat = object_meta[oid]["category"]
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    ratios = subtype_def["subtype_ratio"]
    majority_subtype = max(ratios, key=ratios.get)
    majority_profile = subtype_def["affordance_profiles"][majority_subtype]
    cat_majority_preds[oid] = {
        f: float(v) for f, v in majority_profile.items()
        if f in CORE_ACTION_FEATURES
    }
cat_maj_metrics = compute_full_metrics(cat_majority_preds, query_gt)
print(f"    macro_bal={cat_maj_metrics['macro_query_balanced_accuracy']:.4f}")

print("  all policies done.\n")

# =============================================================================
# 7. Compute probe diagnostics
# =============================================================================
print("[4/6] Computing probe diagnostics...")


def classify_probes(probe_tracking, agent_obs):
    """Classify each probe by efficiency and differentiating action."""
    details = []
    visited_oids = {oid for oid in test_oids if agent_obs.is_visited(oid)}

    for probe in probe_tracking:
        oid = probe["oid"]
        meta = object_meta[oid]
        action = probe["action"]
        is_diff = (action == meta["differentiating_action"])

        pre_correct = 0
        post_correct = 0
        gt = meta["gt_affordances"]
        for f in CORE_ACTION_FEATURES:
            pre_pred = 1.0 if probe["pre_probs"].get(f, 0.5) >= 0.5 else 0.0
            post_pred = 1.0 if probe["post_probs"].get(f, 0.5) >= 0.5 else 0.0
            true_val = gt.get(f, 0.0)
            if pre_pred == true_val:
                pre_correct += 1
            if post_pred == true_val:
                post_correct += 1
        if post_correct > pre_correct:
            eff_label = "effective"
        elif post_correct == pre_correct:
            eff_label = "zero_gain"
        else:
            eff_label = "harmful"

        details.append({
            "oid": oid,
            "category": meta["category"],
            "subtype": meta["subtype"],
            "is_majority": meta["is_majority"],
            "action": action,
            "is_differentiating": is_diff,
            "outcome": probe["outcome"],
            "delta_correct": post_correct - pre_correct,
            "efficiency": eff_label,
        })

    n_probes = len(details)
    n_eff = sum(1 for d in details if d["efficiency"] == "effective")
    n_zg = sum(1 for d in details if d["efficiency"] == "zero_gain")
    n_hm = sum(1 for d in details if d["efficiency"] == "harmful")
    n_diff = sum(1 for d in details if d["is_differentiating"])
    n_non_diff = n_probes - n_diff
    n_non_diff_zg = sum(1 for d in details
                        if not d["is_differentiating"] and d["efficiency"] == "zero_gain")

    # visited coverage per query
    visited_coverage = {}
    for qname in sorted(query_gt.keys()):
        pos_set = set(query_gt[qname]["positive_oids"])
        neg_set = set(query_gt[qname]["negative_oids"])
        visited_pos = len(pos_set & visited_oids)
        visited_neg = len(neg_set & visited_oids)
        total_pos = len(pos_set)
        total_neg = len(neg_set)
        visited_coverage[qname] = {
            "total_positive": total_pos,
            "total_negative": total_neg,
            "visited_positive": visited_pos,
            "visited_negative": visited_neg,
            "positive_coverage": round(visited_pos / max(total_pos, 1), 4),
            "negative_coverage": round(visited_neg / max(total_neg, 1), 4),
        }

    return {
        "probe_count": n_probes,
        "effective_probe_count": n_eff,
        "zero_gain_probe_count": n_zg,
        "harmful_probe_count": n_hm,
        "effective_probe_rate": round(n_eff / max(n_probes, 1), 4),
        "differentiating_action_probe_count": n_diff,
        "differentiating_action_probe_rate": round(n_diff / max(n_probes, 1), 4),
        "non_differentiating_action_probe_count": n_non_diff,
        "non_differentiating_action_zero_gain_rate": (
            round(n_non_diff_zg / max(n_non_diff, 1), 4) if n_non_diff > 0 else 0.0
        ),
        "per_probe": details,
        "visited_positive_coverage_per_query": visited_coverage,
    }


c15b_probe_diag = classify_probes(c15b_policy._probe_tracking, c15b_obs)
c15B_probe_diag = classify_probes(c15B_policy._probe_tracking, c15B_obs)
c15B_oracle_probe_diag = classify_probes(c15B_oracle_policy._probe_tracking, c15B_oracle_obs)

# C15B-specific diagnostic score stats
c15B_all_scores = c15B_policy._all_scored_pairs
c15B_selected_scores = c15B_policy._selected_scores
mean_score_selected = (sum(c15B_selected_scores) / max(len(c15B_selected_scores), 1))
# Unselected: all scored pairs except the selected ones
selected_set = set()
for i, _ in enumerate(c15B_selected_scores):
    # We don't have exact correspondence, so we compute from scored pairs
    pass
# Better approach: group by probe order
unselected_scores = [s["score"] for s in c15B_all_scores
                     if s["score"] not in c15B_selected_scores]
# Actually use a simpler approach: mean of all scored minus selected
mean_score_all = (sum(s["score"] for s in c15B_all_scores)
                  / max(len(c15B_all_scores), 1)) if c15B_all_scores else 0.0
mean_score_unselected = (
    (sum(s["score"] for s in c15B_all_scores) - sum(c15B_selected_scores))
    / max(len(c15B_all_scores) - len(c15B_selected_scores), 1)
) if len(c15B_all_scores) > len(c15B_selected_scores) else 0.0

c15B_probe_extra = {
    "observed_object_count": c15B_policy._observed_object_count,
    "candidate_object_action_count": len(c15B_all_scores),
    "selected_probe_count": len(c15B_policy._probed_oids),
    "mean_diagnostic_score_selected": round(mean_score_selected, 6),
    "mean_diagnostic_score_unselected": round(mean_score_unselected, 6),
}

# =============================================================================
# 8. Build comparisons and output
# =============================================================================
print("[5/6] Building output...")

oracle_macro_bal = oracle_metrics["macro_query_balanced_accuracy"]
oracle_c0b_gap = oracle_macro_bal - c0b_metrics["macro_query_balanced_accuracy"]
c0b_mb = c0b_metrics["macro_query_balanced_accuracy"]
c0b_pos_r = c0b_metrics["mean_positive_recall"]


def make_comparison(policy_label, metrics, cost, extra_info=None, probe_diag=None):
    entry = {
        "policy": policy_label,
        "macro_query_balanced_accuracy": metrics["macro_query_balanced_accuracy"],
        "macro_query_accuracy": metrics["macro_query_accuracy"],
        "mean_positive_recall": metrics["mean_positive_recall"],
        "mean_negative_recall": metrics["mean_negative_recall"],
        "per_query": metrics["per_query"],
    }
    if cost is not None:
        entry.update(cost)

    # vs C0b
    if policy_label != "C0b_observe_only" and "C0b_observe_only" in all_results:
        c0bm = all_results["C0b_observe_only"]["metrics"]
        c0bc = all_results["C0b_observe_only"]["cost"]
        entry["vs_C0b"] = {
            "delta_macro_bal": round(metrics["macro_query_balanced_accuracy"] - c0bm["macro_query_balanced_accuracy"], 6),
            "delta_mean_positive_recall": round(metrics["mean_positive_recall"] - c0bm["mean_positive_recall"], 6),
            "delta_mean_negative_recall": round(metrics["mean_negative_recall"] - c0bm["mean_negative_recall"], 6),
            "delta_normalized_cost": round(cost["normalized_cost"] - c0bc["normalized_cost"], 6) if cost else None,
        }

    # vs C13
    if policy_label != "C13_instance_VOI_original" and "C13_instance_VOI_original" in all_results:
        c13m = all_results["C13_instance_VOI_original"]["metrics"]
        c13c = all_results["C13_instance_VOI_original"]["cost"]
        entry["vs_C13"] = {
            "delta_macro_bal": round(metrics["macro_query_balanced_accuracy"] - c13m["macro_query_balanced_accuracy"], 6),
            "delta_visit_count": (cost.get("visit_count", 0) - c13c["visit_count"]) if cost else None,
            "delta_probe_count": (cost.get("probe_count", 0) - c13c["probe_count"]) if cost else None,
            "delta_positive_recall": round(metrics["mean_positive_recall"] - c13m["mean_positive_recall"], 6),
            "delta_negative_recall": round(metrics["mean_negative_recall"] - c13m["mean_negative_recall"], 6),
        }

    # vs C15b (for C15B only)
    if policy_label == "C15B_cross_action_diagnostic_VOI" and "C15b_probe_reserve" in all_results:
        c15bm = all_results["C15b_probe_reserve"]["metrics"]
        c15bc = all_results["C15b_probe_reserve"]["cost"]
        entry["vs_C15b"] = {
            "delta_macro_bal": round(metrics["macro_query_balanced_accuracy"] - c15bm["macro_query_balanced_accuracy"], 6),
            "delta_mean_positive_recall": round(metrics["mean_positive_recall"] - c15bm["mean_positive_recall"], 6),
            "delta_mean_negative_recall": round(metrics["mean_negative_recall"] - c15bm["mean_negative_recall"], 6),
            "delta_normalized_cost": round(cost["normalized_cost"] - c15bc["normalized_cost"], 6),
        }

    # Gap capture
    if policy_label not in ("C0b_observe_only",):
        gap = metrics["macro_query_balanced_accuracy"] - c0b_mb
        entry["gap_capture_vs_c0b"] = round(gap / max(oracle_c0b_gap, 0.001), 6)

    if extra_info is not None:
        entry["c15B_extra"] = extra_info
    if probe_diag is not None:
        entry["probe_diagnostics"] = {
            k: v for k, v in probe_diag.items() if k != "per_probe"
        }
        entry["probe_diagnostics"]["per_probe"] = probe_diag["per_probe"]

    return entry


policies_output = {}
for label in ["all_false", "C_minus_prior_only", "C0b_observe_only",
              "random_probe_budgeted", "C13_instance_VOI_original"]:
    r = all_results[label]
    policies_output[label] = make_comparison(label, r["metrics"], r["cost"])

# C15b
policies_output["C15b_probe_reserve"] = make_comparison(
    "C15b_probe_reserve", c15b_metrics, c15b_cost, probe_diag=c15b_probe_diag)

# C15B
policies_output["C15B_cross_action_diagnostic_VOI"] = make_comparison(
    "C15B_cross_action_diagnostic_VOI", c15B_metrics, c15B_cost,
    extra_info=c15B_probe_extra, probe_diag=c15B_probe_diag)

# =============================================================================
# 9. Compute Pareto frontier
# =============================================================================
normal_policies = [
    "all_false", "C_minus_prior_only", "C0b_observe_only",
    "random_probe_budgeted", "C13_instance_VOI_original",
    "C15b_probe_reserve", "C15B_cross_action_diagnostic_VOI",
]

pareto_points = []
for label in normal_policies:
    r = all_results[label]
    m = r["metrics"]
    c = r["cost"]
    pareto_points.append({
        "policy": label,
        "macro_bal": m["macro_query_balanced_accuracy"],
        "normalized_cost": c["normalized_cost"],
    })

pareto_frontier = []
for p in pareto_points:
    dominated = False
    for q in pareto_points:
        if p["policy"] == q["policy"]:
            continue
        if (q["macro_bal"] >= p["macro_bal"]
                and q["normalized_cost"] <= p["normalized_cost"]
                and (q["macro_bal"] > p["macro_bal"] or q["normalized_cost"] < p["normalized_cost"])):
            dominated = True
            break
    if not dominated:
        pareto_frontier.append(p["policy"])

# =============================================================================
# 10. C15B assessment
# =============================================================================
c15B_mb = c15B_metrics["macro_query_balanced_accuracy"]
c15B_pos_r = c15B_metrics["mean_positive_recall"]
c15B_neg_r = c15B_metrics["mean_negative_recall"]
c15b_mb = c15b_metrics["macro_query_balanced_accuracy"]
c13_mb = c13_metrics["macro_query_balanced_accuracy"]
rprobe_mb = rprobe_metrics["macro_query_balanced_accuracy"]

c15B_gap_capture = (c15B_mb - c0b_mb) / max(oracle_c0b_gap, 0.001)
TEN_PCT_THRESHOLD = 0.0413
PRACTICAL_TARGET = 0.6284

c15B_dominated_by_c0b = (
    c0b_mb >= c15B_mb
    and c0b_cost["normalized_cost"] <= c15B_cost["normalized_cost"]
    and (c0b_mb > c15B_mb or c0b_cost["normalized_cost"] < c15B_cost["normalized_cost"])
)

c15B_dominated_by_c15b = (
    c15b_mb >= c15B_mb
    and c15b_cost["normalized_cost"] <= c15B_cost["normalized_cost"]
    and (c15b_mb > c15B_mb or c15b_cost["normalized_cost"] < c15B_cost["normalized_cost"])
)

c15B_neg_only = (
    c15B_pos_r < c0b_pos_r
    and c15B_neg_r > c0b_metrics["mean_negative_recall"]
)

c15B_diff_rate = c15B_probe_diag["differentiating_action_probe_rate"]
c15b_diff_rate = c15b_probe_diag["differentiating_action_probe_rate"]
c15B_eff_rate = c15B_probe_diag["effective_probe_rate"]
c15b_eff_rate = c15b_probe_diag["effective_probe_rate"]

c15B_strongly_promising = (
    c15B_mb > c15b_mb                                             # A
    and c15B_mb > c0b_mb                                          # B
    and c15B_mb > rprobe_mb                                       # C
    and c15B_pos_r > c0b_pos_r                                    # D
    and c15B_mb >= PRACTICAL_TARGET                               # E
    and not c15B_dominated_by_c0b                                 # F
    and not c15B_dominated_by_c15b                                # F
    and c15B_diff_rate > c15b_diff_rate                           # G
    and not c15B_neg_only                                         # H
)

# =============================================================================
# 11. Build output JSON
# =============================================================================
output = {
    "block_id": "1J7",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "C15B cross-action diagnostic VOI evaluation — Route B on top of C15 two-pass",
    "policies": policies_output,
    "oracle": {
        "policy": "C_oracle_full_information",
        "macro_query_balanced_accuracy": oracle_macro_bal,
        "per_query": oracle_metrics["per_query"],
    },
    "diagnostic_category_majority": {
        "policy": "diagnostic_category_majority_upper_bound",
        "macro_query_balanced_accuracy": cat_maj_metrics["macro_query_balanced_accuracy"],
        "note": "Leakage-risk diagnostic only. Uses hidden_category. NOT a deployable policy. Excluded from normal-policy Pareto frontier.",
    },
    "diagnostic_oracle_diff_action_upper_bound": {
        "policy": "C15B_oracle_diff_action_upper_bound",
        "macro_query_balanced_accuracy": c15B_oracle_metrics["macro_query_balanced_accuracy"],
        "visit_count": c15B_oracle_cost["visit_count"],
        "probe_count": c15B_oracle_cost["probe_count"],
        "normalized_cost": c15B_oracle_cost["normalized_cost"],
        "note": "Diagnostic only: probes KNOWN differentiating action per category. Uses hidden_category. NOT deployable. Excluded from normal-policy Pareto frontier.",
    },
    "oracle_c0b_gap": round(oracle_c0b_gap, 6),
    "pareto_frontier": {
        "normal_policies_only": True,
        "policies_on_frontier": pareto_frontier,
        "all_pareto_points": pareto_points,
        "category_majority_excluded_as_diagnostic": True,
        "oracle_diff_action_excluded_as_diagnostic": True,
    },
    "c15B_assessment": {
        "c15B_macro_bal": c15B_mb,
        "c15B_ncost": c15B_cost["normalized_cost"],
        "c15B_visit_count": c15B_cost["visit_count"],
        "c15B_probe_count": c15B_cost["probe_count"],
        "c15B_delta_vs_c15b": round(c15B_mb - c15b_mb, 6),
        "c15B_delta_vs_c0b": round(c15B_mb - c0b_mb, 6),
        "c15B_delta_vs_c13": round(c15B_mb - c13_mb, 6),
        "c15B_gap_capture_vs_c0b": round(c15B_gap_capture, 6),
        "c15B_improves_positive_recall_over_c0b": c15B_pos_r > c0b_pos_r,
        "c15B_diff_action_probe_rate": c15B_diff_rate,
        "c15B_effective_probe_rate": c15B_eff_rate,
        "c15B_diff_rate_delta_vs_c15b": round(c15B_diff_rate - c15b_diff_rate, 4),
        "c15B_eff_rate_delta_vs_c15b": round(c15B_eff_rate - c15b_eff_rate, 4),
        "c15B_neg_only_tradeoff": c15B_neg_only,
        "c15B_pareto_dominated_by_c0b": c15B_dominated_by_c0b,
        "c15B_pareto_dominated_by_c15b": c15B_dominated_by_c15b,
        "c15B_single_seed_strongly_promising": c15B_strongly_promising,
        "candidate_ready_for_small_multiseed": c15B_strongly_promising,
        "conditions_for_strongly_promising": {
            "A_beats_C15b": c15B_mb > c15b_mb,
            "B_beats_C0b": c15B_mb > c0b_mb,
            "C_beats_random": c15B_mb > rprobe_mb,
            "D_improves_pos_recall_over_C0b": c15B_pos_r > c0b_pos_r,
            "E_gap_capture_ge_10pct": c15B_mb >= PRACTICAL_TARGET,
            "F_not_pareto_dominated_by_C0b_or_C15b": (
                not c15B_dominated_by_c0b and not c15B_dominated_by_c15b
            ),
            "G_diff_action_rate_improves_over_C15b": c15B_diff_rate > c15b_diff_rate,
            "H_no_neg_only_tradeoff": not c15B_neg_only,
        },
    },
    "c15b_reference": {
        "c15b_macro_bal": c15b_mb,
        "c15b_ncost": c15b_cost["normalized_cost"],
        "c15b_visit_count": c15b_cost["visit_count"],
        "c15b_probe_count": c15b_cost["probe_count"],
        "c15b_diff_action_probe_rate": c15b_diff_rate,
        "c15b_effective_probe_rate": c15b_eff_rate,
    },
    "route_recommendation": {
        "route_a_directionally_validated": True,
        "route_b_evaluated_in_1J7": True,
        "route_b_improves_over_c15b": c15B_mb > c15b_mb,
        "route_b_diff_rate_improved": c15B_diff_rate > c15b_diff_rate,
        "route_f_needed_later": True,
        "ready_for_multiseed": False,
        "block_1J7_summary": "C15B cross-action diagnostic VOI adds Route B to C15 two-pass.",
    },
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j7_c15b_cross_action_voi_c4_seed101.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {json_path}")

# =============================================================================
# 12. Print block_done
# =============================================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"block_id=1J7")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"c0b_macro_bal={c0b_mb:.4f}")
print(f"c15b_macro_bal={c15b_mb:.4f}")
print(f"c15B_macro_bal={c15B_mb:.4f}")
print(f"c15B_ncost={c15B_cost['normalized_cost']:.4f}")
print(f"c15B_delta_vs_c15b={c15B_mb - c15b_mb:+.4f}")
print(f"c15B_delta_vs_c0b={c15B_mb - c0b_mb:+.4f}")
print(f"c15B_gap_capture_vs_c0b={c15B_gap_capture:.4f}")
print(f"c15B_improves_positive_recall={c15B_pos_r > c0b_pos_r}")
print(f"c15B_diff_action_probe_rate={c15B_diff_rate:.4f}")
print(f"c15B_effective_probe_rate={c15B_eff_rate:.4f}")
print(f"c15B_pareto_status={'dominated' if (c15B_dominated_by_c0b or c15B_dominated_by_c15b) else 'not_dominated'}")
print(f"c15B_single_seed_strongly_promising={c15B_strongly_promising}")
print(f"candidate_ready_for_small_multiseed={c15B_strongly_promising}")
print(f"route_f_needed_later=true")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*60}")
