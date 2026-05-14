"""
Block 1J8 — Bottleneck Localization Before Route F.

Four diagnostics to determine where the remaining C0b-to-oracle headroom is:
  1. Oracle-object-selection ablation
  2. Environment ceiling check
  3. Category-only / shortcut diagnostic
  4. Budget/probe binding audit
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
print("Block 1J8 — Bottleneck Localization Before Route F")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print("=" * 60)

# =============================================================================
# 1. Training + IOM
# =============================================================================
print("\n[1/5] Phase A: Training + IOM building...")
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
print("\n[2/5] Generating subtype test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

CATEGORY_DIFFERENTIATING_ACTION = {
    "wood_log": "craft_plank",
    "stone_block": "mine_by_hand",
    "apple": "eat",
    "wooden_pickaxe": "use_as_tool",
}

# Build object metadata
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
# 3. Metric helpers (same as 1J6/1J7)
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
# 4. C15b reference policy (same as 1J7)
# =============================================================================
class C15b_ProbeReservePolicy(MiniMCPolicy):
    """C15b: Two-pass, reserve 25% for probes, VOI-based probe selection."""

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
        # Track VOI scores for budget audit
        self._all_voi_scores = []  # (oid, voi_score, affordable)

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0
        self._all_voi_scores = []

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
            self._all_voi_scores.append({
                "oid": oid, "voi": best_action_voi, "affordable": affordable,
            })
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
# 5. Run baselines: C0b and C15b
# =============================================================================
print("\n[3/5] Running baselines...")

# C0b
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(im_base.clone(), random.Random(SEED + 100))
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(c0b_preds["per_object"], query_gt)
c0b_cost = get_cost_metrics(c0b_log, c0b_obs)
c0b_mb = c0b_metrics["macro_query_balanced_accuracy"]
print(f"  C0b macro_bal={c0b_mb:.4f}  visited={c0b_cost['visit_count']}")

# C15b
c15b_im = im_base.clone()
c15b_policy = C15b_ProbeReservePolicy(
    c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(c15b_preds["per_object"], query_gt)
c15b_cost = get_cost_metrics(c15b_log, c15b_obs)
c15b_mb = c15b_metrics["macro_query_balanced_accuracy"]
print(f"  C15b macro_bal={c15b_mb:.4f}  visited={c15b_cost['visit_count']}  "
      f"probed={c15b_cost['probe_count']}  ncost={c15b_cost['normalized_cost']:.4f}")

# Oracle
c_oracle = C14a_TruthAnswerOracle(sim_gt)
oracle_preds = c_oracle.get_answer()
oracle_metrics = compute_full_metrics(oracle_preds["per_object"], query_gt)
oracle_mb = oracle_metrics["macro_query_balanced_accuracy"]
oracle_c0b_gap = oracle_mb - c0b_mb
print(f"  Oracle macro_bal={oracle_mb:.4f}  oracle_c0b_gap={oracle_c0b_gap:.4f}")

# =============================================================================
# 6. Diagnostic 1: Oracle-object-selection ablation
# =============================================================================
print("\n[4/5] Running bottleneck diagnostics...")
print("  Diagnostic 1: Oracle-object-selection ablation...")

class OracleObjectSelectionPolicy(MiniMCPolicy):
    """Diagnostic: uses oracle knowledge to select best objects to probe.

    After Pass 1 (observe, same as C15b), in Pass 2 ranks observed objects
    by prediction error (how many affordance predictions are wrong) / cost.
    Probes the differentiating action on selected objects.

    This tests: if we could perfectly select WHICH objects to probe (oracle
    object selection), how much improvement would we get?

    DIAGNOSTIC ONLY — uses ground-truth affordances for object ranking.
    NOT a deployable policy.
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
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._probed_oids = set()
        self._observe_pass_visit_count = 0

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
                return self._select_probe_target_oracle(view)
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
            return self._select_probe_target_oracle(view)
        return None

    def _select_probe_target_oracle(self, view):
        """Oracle object ranking: select objects with highest prediction error / cost."""
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
            gt = object_meta[oid]["gt_affordances"]
            # Count prediction errors
            errors = 0
            for f in CORE_ACTION_FEATURES:
                pred = 1.0 if probs.get(f, 0.5) >= 0.5 else 0.0
                true_val = gt.get(f, 0.0)
                if pred != true_val:
                    errors += 1
            score = errors / (total_cost / max(view.initial_budget, 0.001) + 0.001)
            if score > best_score:
                best_score = score
                best_oid = oid
        if best_oid is not None:
            self._visit_order.append(("probe", best_oid))
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        # Always probe the differentiating action for the selected object
        diff_action = object_meta[object_id]["differentiating_action"]
        if diff_action is None:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        # Check if probing this action would be affordable
        if not view.can_afford(view.probe_cost):
            return False, None
        self._probed_oids.add(object_id)
        return True, diff_action

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


oracle_obj_sel_im = im_base.clone()
oracle_obj_sel_policy = OracleObjectSelectionPolicy(
    oracle_obj_sel_im, random.Random(SEED + 1000), cost_weight=COST_WEIGHT)
oracle_obj_sel_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
oracle_obj_preds, oracle_obj_log, oracle_obj_obs, _, _ = run_policy(
    oracle_obj_sel_policy, oracle_obj_sel_env)
oracle_obj_metrics = compute_full_metrics(oracle_obj_preds["per_object"], query_gt)
oracle_obj_cost = get_cost_metrics(oracle_obj_log, oracle_obj_obs)
oracle_obj_mb = oracle_obj_metrics["macro_query_balanced_accuracy"]
print(f"    macro_bal={oracle_obj_mb:.4f}  delta_vs_C15b={oracle_obj_mb - c15b_mb:+.4f}  "
      f"visited={oracle_obj_cost['visit_count']}  probed={oracle_obj_cost['probe_count']}")

# Analyze selected objects: majority vs minority, positive vs negative
oracle_obj_selection_details = []
for probe in oracle_obj_sel_policy._probe_tracking:
    oid = probe["oid"]
    meta = object_meta[oid]
    is_minority = not meta["is_majority"]
    positive_queries = [q for q, v in meta["query_membership"].items() if v]
    oracle_obj_selection_details.append({
        "oid": oid,
        "category": meta["category"],
        "subtype": meta["subtype"],
        "is_majority": meta["is_majority"],
        "is_minority": is_minority,
        "positive_for_queries": positive_queries,
        "action": probe["action"],
        "outcome": probe["outcome"],
    })

n_minority_selected = sum(1 for d in oracle_obj_selection_details if d["is_minority"])
n_majority_selected = sum(1 for d in oracle_obj_selection_details if not d["is_minority"])

# =============================================================================
# 7. Diagnostic 3B: Public-surface-category-like baseline
# =============================================================================
print("  Diagnostic 3: Category-only diagnostics...")

# 3A: hidden_category_majority
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
cat_maj_mb = cat_maj_metrics["macro_query_balanced_accuracy"]
print(f"    3A: hidden_category_majority macro_bal={cat_maj_mb:.4f}")

# 3B: public_surface_category_like
# Predict category from visible features using training-neighbor majority vote,
# then use that predicted category's majority subtype affordance profile.
# Uses training hidden_category (available during training) but NOT test hidden_category.
public_cat_preds = {}
for oid in test_oids:
    obj = test_objects[oid]
    test_vis = obj.get("visible_features", {})

    # Find k-nearest training neighbors by visible feature similarity
    neighbors = []
    for train_oid in im_base._train_oids:
        train_vis = im_base._train_vis[train_oid]
        # Jaccard similarity
        intersection = 0
        union = 0
        for f in posterior_visible_features:
            v1 = bool(test_vis.get(f, False))
            v2 = bool(train_vis.get(f, False))
            if v1 and v2:
                intersection += 1
                union += 1
            elif v1 or v2:
                union += 1
        sim = intersection / union if union > 0 else 0.0
        if sim > 0:
            neighbors.append((sim, train_oid))

    neighbors.sort(key=lambda x: x[0], reverse=True)
    top_k = neighbors[:max(config.INSTANCE_K, 3)]

    if top_k:
        # Majority vote on training hidden_category
        cat_votes = {}
        for sim, train_oid in top_k:
            train_cat = train_objects_dict[train_oid].get("hidden_category", None)
            if train_cat:
                cat_votes[train_cat] = cat_votes.get(train_cat, 0) + sim
        predicted_cat = max(cat_votes, key=cat_votes.get) if cat_votes else None
    else:
        predicted_cat = None

    if predicted_cat and predicted_cat in SUBTYPE_DEFINITIONS:
        subtype_def = SUBTYPE_DEFINITIONS[predicted_cat]
        ratios = subtype_def["subtype_ratio"]
        majority_subtype = max(ratios, key=ratios.get)
        majority_profile = subtype_def["affordance_profiles"][majority_subtype]
        public_cat_preds[oid] = {
            f: float(v) for f, v in majority_profile.items()
            if f in CORE_ACTION_FEATURES
        }
    else:
        # Fallback: use global base rates
        public_cat_preds[oid] = {f: 0.5 for f in CORE_ACTION_FEATURES}

public_cat_metrics = compute_full_metrics(public_cat_preds, query_gt)
public_cat_mb = public_cat_metrics["macro_query_balanced_accuracy"]
print(f"    3B: public_category_like macro_bal={public_cat_mb:.4f}")

# Determine category shortcut risk
if public_cat_mb > 0.70:
    cat_shortcut_risk = "high"
elif public_cat_mb > 0.60:
    cat_shortcut_risk = "medium"
else:
    cat_shortcut_risk = "low"
print(f"    category_shortcut_risk={cat_shortcut_risk}")

def _classify_efficiency(probe):
    oid = probe["oid"]
    gt = object_meta[oid]["gt_affordances"]
    pre_correct = 0
    post_correct = 0
    for f in CORE_ACTION_FEATURES:
        pre_pred = 1.0 if probe["pre_probs"].get(f, 0.5) >= 0.5 else 0.0
        post_pred = 1.0 if probe["post_probs"].get(f, 0.5) >= 0.5 else 0.0
        true_val = gt.get(f, 0.0)
        if pre_pred == true_val:
            pre_correct += 1
        if post_pred == true_val:
            post_correct += 1
    if post_correct > pre_correct:
        return "effective"
    elif post_correct == pre_correct:
        return "zero_gain"
    return "harmful"


# =============================================================================
# 8. Diagnostic 4: Budget/probe binding audit
# =============================================================================
print("  Diagnostic 4: Budget/probe binding audit...")

# Recompute VOI from scratch at episode end using final IOM state.
# This avoids duplication issues with _all_voi_scores (which accumulates
# across multiple _select_probe_target calls) and gives a clean picture
# of what's left on the table with actual remaining budget.
voi_recomputed = {}
for oid in c15b_obs.get_all_object_ids():
    if not c15b_obs.is_visited(oid) or oid in c15b_policy._probed_oids:
        continue
    features = c15b_obs.get_observed_features(oid)
    if features is None:
        continue
    total_cost = c15b_obs.compute_reach_cost(oid) + c15b_obs.observe_cost + c15b_obs.probe_cost
    affordable = c15b_obs.can_afford(total_cost)
    fake_obj = {"id": oid, "visible_features": features}
    probs = c15b_policy._im.predict_all_affordances(fake_obj)
    current_utility = _compute_utility(probs)
    norm_cost = total_cost / max(BUDGET, 0.001)
    best_action_voi = 0.0
    for action in MAIN_CANDIDATE_ACTIONS:
        p_success = probs.get(ACTION_TO_FEATURE[action], 0.5)
        im_succ = c15b_policy._im.clone()
        im_succ.incorporate_probe(oid, action, 1.0)
        utility_succ = _compute_utility(im_succ.predict_all_affordances(fake_obj))
        im_fail = c15b_policy._im.clone()
        im_fail.incorporate_probe(oid, action, 0.0)
        utility_fail = _compute_utility(im_fail.predict_all_affordances(fake_obj))
        e_post = p_success * utility_succ + (1 - p_success) * utility_fail
        net_voi = e_post - current_utility - COST_WEIGHT * norm_cost
        if net_voi > best_action_voi:
            best_action_voi = net_voi
    voi_recomputed[oid] = {"oid": oid, "voi": best_action_voi, "affordable": affordable}

voi_scores_unique = list(voi_recomputed.values())

selected_voi = [s for s in voi_scores_unique if s["oid"] in c15b_policy._probed_oids]
unselected_affordable = sorted(
    [s for s in voi_scores_unique
     if s["oid"] not in c15b_policy._probed_oids
     and s["affordable"]],
    key=lambda x: x["voi"], reverse=True)
unselected_unaffordable = sorted(
    [s for s in voi_scores_unique
     if s["oid"] not in c15b_policy._probed_oids
     and not s["affordable"]],
    key=lambda x: x["voi"], reverse=True)
unselected_positive_voi_affordable = [s for s in unselected_affordable if s["voi"] > 0]
unselected_positive_voi_unaffordable = [s for s in unselected_unaffordable if s["voi"] > 0]

actual_probe_count = len(c15b_policy._probed_oids)
budget_audit = {
    "c15b_remaining_budget": c15b_obs.budget_remaining,
    "c15b_initial_budget": BUDGET,
    "c15b_budget_spent": c15b_obs.budget_spent,
    "total_visited_unprobed": len(voi_scores_unique),
    "probes_selected": actual_probe_count,
    "unselected_affordable_with_positive_voi": len(unselected_positive_voi_affordable),
    "unselected_unaffordable_with_positive_voi": len(unselected_positive_voi_unaffordable),
    "unselected_affordable_total": len(unselected_affordable),
    "unselected_unaffordable_total": len(unselected_unaffordable),
    "mean_voi_selected": "N/A (recomputed at episode end with final IOM; probed objects excluded)",
    "mean_voi_unselected_affordable": (
        sum(s["voi"] for s in unselected_affordable) / max(len(unselected_affordable), 1)
    ) if unselected_affordable else 0.0,
    "mean_voi_unselected_unaffordable": (
        sum(s["voi"] for s in unselected_unaffordable) / max(len(unselected_unaffordable), 1)
    ) if unselected_unaffordable else 0.0,
    "max_voi_unselected_affordable": unselected_affordable[0]["voi"] if unselected_affordable else 0.0,
    "max_voi_unselected_unaffordable": unselected_unaffordable[0]["voi"] if unselected_unaffordable else 0.0,
    "voi_recomputation_method": "fresh_computation_at_episode_end_with_final_IOM_state",
}

# Count probe efficiency
n_eff = sum(1 for p in c15b_policy._probe_tracking
            if _classify_efficiency(p) == "effective")
n_zg = sum(1 for p in c15b_policy._probe_tracking
           if _classify_efficiency(p) == "zero_gain")
n_hm = sum(1 for p in c15b_policy._probe_tracking
           if _classify_efficiency(p) == "harmful")
n_probes = max(len(c15b_policy._probe_tracking), 1)

budget_audit.update({
    "effective_probe_count": n_eff,
    "zero_gain_probe_count": n_zg,
    "harmful_probe_count": n_hm,
    "effective_probe_rate": n_eff / n_probes,
    "zero_gain_rate": n_zg / n_probes,
    "marginal_voi_of_last_probe": (
        c15b_policy._all_voi_scores[-1]["voi"] if c15b_policy._all_voi_scores else 0.0
    ),
})

# "One extra probe" diagnostic: which object would have been next?
next_best = unselected_affordable[0] if unselected_affordable else None
budget_audit["next_best_unselected_oid"] = next_best["oid"] if next_best else None
budget_audit["next_best_unselected_voi"] = next_best["voi"] if next_best else 0.0

print(f"    selected={budget_audit['probes_selected']}  "
      f"unsel_affordable_pos_voi={budget_audit['unselected_affordable_with_positive_voi']}  "
      f"unsel_unaffordable_pos_voi={budget_audit['unselected_unaffordable_with_positive_voi']}  "
      f"eff_rate={budget_audit['effective_probe_rate']:.2f}  "
      f"remaining_budget={budget_audit['c15b_remaining_budget']:.4f}")


# =============================================================================
# 9. Bottleneck determination
# =============================================================================
print("\n[5/5] Determining bottlenecks...")

# Decision rules
obj_sel_improvement = oracle_obj_mb - c15b_mb
object_selection_bottleneck = obj_sel_improvement > 0.005  # meaningful improvement

env_ceiling = oracle_mb - c15b_mb
environment_ceiling_bottleneck = env_ceiling < 0.05  # oracle is close to C15b

category_shortcut_bottleneck = public_cat_mb > 0.70

# Budget binding: many positive-VOI unaffordable, OR remaining budget too small
# but effective_probe_rate is decent
budget_binding = (
    budget_audit["unselected_unaffordable_with_positive_voi"] > 3
    and budget_audit["effective_probe_rate"] >= 0.4
)

probe_informativeness_bottleneck = (
    budget_audit["effective_probe_rate"] < 0.6
    and budget_audit["unselected_affordable_with_positive_voi"] < 5
)

# Object selection bottleneck means observation interface might help with better targeting
observation_interface_bottleneck = object_selection_bottleneck

# Determine recommended next route
if object_selection_bottleneck:
    recommended_next = "Route_F_blurred_peripheral_observation"
    implement_route_f_now = True
elif category_shortcut_bottleneck:
    recommended_next = "environment_redesign_reduce_category_shortcut"
    implement_route_f_now = False
elif budget_binding and not probe_informativeness_bottleneck:
    recommended_next = "budget_or_cost_redesign"
    implement_route_f_now = False
elif probe_informativeness_bottleneck:
    recommended_next = "probe_informativeness_investigation_or_environment_redesign"
    implement_route_f_now = False
else:
    recommended_next = "mixed_or_inconclusive"
    implement_route_f_now = False

print(f"  object_selection_bottleneck={object_selection_bottleneck}")
print(f"  environment_ceiling_bottleneck={environment_ceiling_bottleneck}")
print(f"  category_shortcut_bottleneck={category_shortcut_bottleneck}")
print(f"  budget_binding={budget_binding}")
print(f"  probe_informativeness_bottleneck={probe_informativeness_bottleneck}")
print(f"  observation_interface_bottleneck={observation_interface_bottleneck}")
print(f"  recommended_next={recommended_next}")
print(f"  implement_route_f_now={implement_route_f_now}")

# =============================================================================
# 10. Build output JSON
# =============================================================================
output = {
    "block_id": "1J8",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Bottleneck localization before Route F — four diagnostics",
    "baselines": {
        "C0b_observe_only": {
            "macro_bal": c0b_mb,
            "mean_positive_recall": c0b_metrics["mean_positive_recall"],
            "mean_negative_recall": c0b_metrics["mean_negative_recall"],
            "visit_count": c0b_cost["visit_count"],
            "normalized_cost": c0b_cost["normalized_cost"],
        },
        "C15b_probe_reserve": {
            "macro_bal": c15b_mb,
            "mean_positive_recall": c15b_metrics["mean_positive_recall"],
            "mean_negative_recall": c15b_metrics["mean_negative_recall"],
            "visit_count": c15b_cost["visit_count"],
            "probe_count": c15b_cost["probe_count"],
            "normalized_cost": c15b_cost["normalized_cost"],
            "per_query": c15b_metrics["per_query"],
        },
        "C_oracle_full_information": {
            "macro_bal": oracle_mb,
            "mean_positive_recall": oracle_metrics["mean_positive_recall"],
            "mean_negative_recall": oracle_metrics["mean_negative_recall"],
        },
    },
    "oracle_c0b_gap": round(oracle_c0b_gap, 6),
    "diagnostic_1_oracle_object_selection": {
        "macro_bal": oracle_obj_mb,
        "mean_positive_recall": oracle_obj_metrics["mean_positive_recall"],
        "mean_negative_recall": oracle_obj_metrics["mean_negative_recall"],
        "visit_count": oracle_obj_cost["visit_count"],
        "probe_count": oracle_obj_cost["probe_count"],
        "normalized_cost": oracle_obj_cost["normalized_cost"],
        "delta_vs_C15b": round(oracle_obj_mb - c15b_mb, 6),
        "delta_vs_C0b": round(oracle_obj_mb - c0b_mb, 6),
        "gap_capture_vs_C0b": round((oracle_obj_mb - c0b_mb) / max(oracle_c0b_gap, 0.001), 6),
        "per_query": oracle_obj_metrics["per_query"],
        "selected_objects": oracle_obj_selection_details,
        "minority_selected_count": n_minority_selected,
        "majority_selected_count": n_majority_selected,
        "minority_enrichment_vs_population": round(
            (n_minority_selected / max(n_minority_selected + n_majority_selected, 1))
            / max(12.0 / 60.0, 0.001), 4  # 12 minority out of 60 total
        ) if (n_minority_selected + n_majority_selected) > 0 else 0.0,
        "note": "DIAGNOSTIC ONLY. Uses ground-truth affordances for object ranking. NOT deployable.",
    },
    "diagnostic_2_environment_ceiling": {
        "full_oracle_macro_bal": oracle_mb,
        "full_oracle_positive_recall": oracle_metrics["mean_positive_recall"],
        "full_oracle_negative_recall": oracle_metrics["mean_negative_recall"],
        "oracle_minus_C15b": round(oracle_mb - c15b_mb, 6),
        "oracle_minus_C0b": round(oracle_mb - c0b_mb, 6),
        "headroom_remaining": round(oracle_mb - c15b_mb, 6),
        "environment_ceiling_bottleneck": environment_ceiling_bottleneck,
    },
    "diagnostic_3_category_shortcut": {
        "hidden_category_majority_macro_bal": cat_maj_mb,
        "hidden_category_majority_note": "Uses true hidden_category. Diagnostic only, NOT deployable. Value 0.975 from 1J2/1J3 (script recomputation may differ due to diagnostic implementation mismatch).",
        "public_category_like_macro_bal": public_cat_mb,
        "public_category_like_note": "Predicts category from visible features via training-neighbor majority vote, then uses majority subtype affordance profile. No hidden_category at test time. Deployable in principle.",
        "public_category_like_per_query": public_cat_metrics["per_query"],
        "category_shortcut_deployable": public_cat_mb > 0.70,
        "category_shortcut_risk_level": cat_shortcut_risk,
    },
    "diagnostic_4_budget_probe_binding": budget_audit,
    "bottleneck_conclusion": {
        "object_selection_bottleneck": object_selection_bottleneck,
        "observation_interface_bottleneck": observation_interface_bottleneck,
        "environment_ceiling_bottleneck": environment_ceiling_bottleneck,
        "category_shortcut_bottleneck": category_shortcut_bottleneck,
        "budget_binding_bottleneck": budget_binding,
        "probe_informativeness_bottleneck": probe_informativeness_bottleneck,
        "primary_bottleneck": (
            "object_selection" if object_selection_bottleneck
            else "category_shortcut" if category_shortcut_bottleneck
            else "budget_binding" if budget_binding
            else "probe_informativeness" if probe_informativeness_bottleneck
            else "mixed"
        ),
        "recommended_next_route": recommended_next,
        "implement_route_f_now": implement_route_f_now,
        "ready_for_multiseed": False,
    },
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j8_bottleneck_localization_c4_seed101.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {json_path}")

# =============================================================================
# 11. Print block_done
# =============================================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"block_id=1J8")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"oracle_object_selection_macro_bal={oracle_obj_mb:.4f}")
print(f"oracle_object_selection_delta_vs_C15b={oracle_obj_mb - c15b_mb:+.4f}")
print(f"full_oracle_macro_bal={oracle_mb:.4f}")
print(f"hidden_category_majority_macro_bal={cat_maj_mb:.4f}")
print(f"public_category_like_macro_bal={public_cat_mb:.4f}")
print(f"budget_binding_supported={budget_binding}")
print(f"object_selection_bottleneck_supported={object_selection_bottleneck}")
print(f"observation_interface_bottleneck_supported={observation_interface_bottleneck}")
print(f"category_shortcut_bottleneck_supported={category_shortcut_bottleneck}")
print(f"recommended_next_route={recommended_next}")
print(f"implement_route_f_now={implement_route_f_now}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*60}")
