"""
Block 1J4 — C13 Failure Decomposition on C4_instance_subtype_cued_v1.

Diagnoses why C13 loses to C0b. Re-runs the 1J3 simulation with detailed
per-object and per-probe tracking. Produces 8 diagnostic sections.
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
print("Block 1J4 — C13 Failure Decomposition")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}  CW={COST_WEIGHT}")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM (same as 1J3)
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
# 2. Subtype test objects + environment
# =============================================================================
print("\n[2/6] Generating subtype test objects + environment...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
print(f"  test_objects={len(test_objects)}  positions assigned")

# =============================================================================
# 3. Query-helpers and probe-classification utilities
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


# Subtype-differentiating action mapping
CATEGORY_DIFFERENTIATING_ACTION = {
    "wood_log": "craft_plank",
    "stone_block": "mine_by_hand",
    "apple": "eat",
    "wooden_pickaxe": "use_as_tool",
}

# Which queries each differentiating action affects
DIFFERENTIATING_ACTION_QUERIES = {
    "craft_plank": ["need_planks"],
    "mine_by_hand": ["need_stone"],
    "eat": ["need_food"],
    "use_as_tool": ["need_tool"],
    "burn_as_fuel": ["need_fuel"],
}

# =============================================================================
# 4. Tracked C13 policy (extended: records visit order, probe details)
# =============================================================================
class C13_FullTrackedPolicy(C13_InstanceVOIPolicy):
    """C13 with full tracking: visit order, per-probe pre/post predictions,
    and object-level metadata for diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_tracking = []
        self.visit_order = []
        self._last_pre_probs = None

    def reset(self, view):
        super().reset(view)
        self.probe_tracking = []
        self.visit_order = []
        self._last_pre_probs = None

    def select_next_object(self, view):
        oid = super().select_next_object(view)
        if oid is not None:
            self.visit_order.append(oid)
        return oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            self._last_pre_probs = self._im.predict_all_affordances(fake_obj)
        else:
            self._last_pre_probs = None
        return super().decide_probe(view, object_id)

    def on_probe_result(self, view, object_id, action, outcome):
        pre_probs = self._last_pre_probs
        super().on_probe_result(view, object_id, action, outcome)
        if pre_probs is not None:
            features = view.get_observed_features(object_id)
            if features is not None:
                fake_obj = {"id": object_id, "visible_features": features}
                post_probs = self._im.predict_all_affordances(fake_obj)
                self.probe_tracking.append({
                    "oid": object_id, "action": action, "outcome": outcome,
                    "pre_probs": dict(pre_probs),
                    "post_probs": dict(post_probs),
                })


# =============================================================================
# 5. RandomProbeBudgetedPolicy (same as 1J3)  — with tracking
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
                "pre_probs": dict(pre_probs),
                "post_probs": dict(post_probs),
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
# 6. Run C13, C0b, random_probe with full tracking
# =============================================================================
print("\n[3/6] Running policies with full tracking...")

# --- C13 ---
print("  C13_instance_VOI (CW=0.5, full tracking)...")
c13_im = im_base.clone()
c13_policy = C13_FullTrackedPolicy(
    c13_im, random.Random(SEED + 500 + int(COST_WEIGHT * 100)),
    cost_weight=COST_WEIGHT)
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_preds, c13_log, c13_obs, _, _ = run_policy(c13_policy, c13_env)
c13_metrics = compute_full_metrics(c13_preds["per_object"], query_gt)
c13_cost = get_cost_metrics(c13_log, c13_obs)
print(f"    macro_bal={c13_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c13_cost['visit_count']}  probed={c13_cost['probe_count']}")

# --- C0b ---
print("  C0b_observe_only...")
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(im_base.clone(), random.Random(SEED + 100))
c0b_preds, c0b_log, c0b_obs, _, _ = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(c0b_preds["per_object"], query_gt)
c0b_cost = get_cost_metrics(c0b_log, c0b_obs)
print(f"    macro_bal={c0b_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c0b_cost['visit_count']}")

# --- random_probe ---
print("  random_probe_budgeted (with tracking)...")
rprobe_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
rprobe_policy = RandomProbeTrackedPolicy(im_base.clone(), random.Random(SEED + 300))
rprobe_preds, rprobe_log, rprobe_obs, _, _ = run_policy(rprobe_policy, rprobe_env)
rprobe_metrics = compute_full_metrics(rprobe_preds["per_object"], query_gt)
rprobe_cost = get_cost_metrics(rprobe_log, rprobe_obs)
print(f"    macro_bal={rprobe_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={rprobe_cost['visit_count']}  probed={rprobe_cost['probe_count']}")

# --- C_minus ---
print("  C_minus_prior_only...")
c_minus_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c_minus_policy = C0a_NoInteractionPolicy(im_base.clone(), random.Random(SEED))
c_minus_preds, c_minus_log, c_minus_obs, _, _ = run_policy(c_minus_policy, c_minus_env)
c_minus_metrics = compute_full_metrics(c_minus_preds["per_object"], query_gt)

print("  all policies done.\n")

# =============================================================================
# 7. Build per-object metadata (category, subtype, ground truth)
# =============================================================================
print("[4/6] Building per-object metadata...")

# Ground truth simulator
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

object_meta = {}
for oid in test_oids:
    obj = test_objects[oid]
    cat = obj["hidden_category"]
    subtype = obj["hidden_subtype"]
    gt_aff = sim_gt.get_ground_truth_affordances(oid)

    # Which subtype is majority vs minority
    subtype_def = SUBTYPE_DEFINITIONS[cat]
    subtypes = subtype_def["subtypes"]
    ratios = subtype_def["subtype_ratio"]
    is_majority = ratios.get(subtype, 0.0) >= 0.5

    # Differentiating action for this category
    diff_action = CATEGORY_DIFFERENTIATING_ACTION.get(cat, None)
    diff_feature = ACTION_TO_FEATURE.get(diff_action, None) if diff_action else None

    # Whether this is a positive object for each query
    query_membership = {}
    for qname in query_gt:
        query_membership[qname] = oid in query_gt[qname]["positive_oids"]

    object_meta[oid] = {
        "category": cat,
        "subtype": subtype,
        "is_majority": is_majority,
        "gt_affordances": gt_aff,
        "differentiating_action": diff_action,
        "differentiating_feature": diff_feature,
        "query_membership": query_membership,
    }

# =============================================================================
# 8. Identify C13 visited/probed sets
# =============================================================================
c13_visited_oids = set()
c13_probed_oids = set()
for oid in test_oids:
    if c13_obs.is_visited(oid):
        c13_visited_oids.add(oid)
    if c13_obs.get_probe_results(oid):
        c13_probed_oids.add(oid)
c13_unvisited_oids = set(test_oids) - c13_visited_oids

rprobe_visited_oids = set()
rprobe_probed_oids = set()
for oid in test_oids:
    if rprobe_obs.is_visited(oid):
        rprobe_visited_oids.add(oid)
    if rprobe_obs.get_probe_results(oid):
        rprobe_probed_oids.add(oid)

# =============================================================================
# 9. Classify C13 probes (efficiency + differentiating action)
# =============================================================================
c13_probe_details = []
for probe in c13_policy.probe_tracking:
    oid = probe["oid"]
    meta = object_meta[oid]
    action = probe["action"]
    is_diff = (action == meta["differentiating_action"])
    affected_queries = DIFFERENTIATING_ACTION_QUERIES.get(action, [])

    # Compute probe efficiency
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

    c13_probe_details.append({
        "oid": oid,
        "category": meta["category"],
        "subtype": meta["subtype"],
        "is_majority": meta["is_majority"],
        "action": action,
        "is_differentiating": is_diff,
        "affected_queries": affected_queries,
        "outcome": probe["outcome"],
        "delta_correct": post_correct - pre_correct,
        "efficiency": eff_label,
    })

# Same for random probe
rprobe_probe_details = []
for probe in rprobe_policy.probe_tracking:
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

    rprobe_probe_details.append({
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

# =============================================================================
# DIAGNOSTIC 1: Visit-breadth loss
# =============================================================================
print("[5/6] Computing diagnostics...")

diag1 = {}
for qname in sorted(query_gt.keys()):
    pos_set = set(query_gt[qname]["positive_oids"])
    neg_set = set(query_gt[qname]["negative_oids"])
    total_pos = len(pos_set)
    total_neg = len(neg_set)

    visited_pos = pos_set & c13_visited_oids
    visited_neg = neg_set & c13_visited_oids
    missed_pos = pos_set - c13_visited_oids
    missed_neg = neg_set - c13_visited_oids

    pos_cov = len(visited_pos) / max(total_pos, 1)
    neg_cov = len(visited_neg) / max(total_neg, 1)

    diag1[qname] = {
        "total_positive": total_pos,
        "total_negative": total_neg,
        "c13_visited_positive": len(visited_pos),
        "c13_visited_negative": len(visited_neg),
        "c13_missed_positive": len(missed_pos),
        "c13_missed_negative": len(missed_neg),
        "positive_coverage_rate": round(pos_cov, 4),
        "negative_coverage_rate": round(neg_cov, 4),
    }

# How much positive recall loss is due to not visiting?
total_pos_across_queries = sum(d["total_positive"] for d in diag1.values())
total_visited_pos = sum(d["c13_visited_positive"] for d in diag1.values())
total_missed_pos = sum(d["c13_missed_positive"] for d in diag1.values())
# C0b visits all 60 objects, so its theoretical pos_recall ceiling from visits alone is 100%
# C13 can at most achieve pos_recall = visited_pos / total_pos (if predictions are perfect on visited)
c13_max_pos_recall_from_visits = total_visited_pos / max(total_pos_across_queries, 1)
c0b_pos_recall = c0b_metrics["mean_positive_recall"]

visit_breadth_summary = {
    "c13_visit_count": len(c13_visited_oids),
    "c0b_visit_count": len(test_oids),
    "c13_unvisited_count": len(c13_unvisited_oids),
    "total_positive_across_queries": total_pos_across_queries,
    "c13_max_possible_pos_recall_from_visits": round(c13_max_pos_recall_from_visits, 4),
    "c0b_actual_pos_recall": round(c0b_pos_recall, 4),
    "pos_recall_loss_from_missed_visits": round(c0b_pos_recall - c13_max_pos_recall_from_visits, 4),
    "per_query": diag1,
}

# =============================================================================
# DIAGNOSTIC 2: Visited-object prediction diagnostic
# =============================================================================
# Evaluate C0b predictions restricted to C13-visited subset
# Evaluate C13 predictions restricted to C13-visited subset
# Also random_probe on same subset (if visited)

diag2 = {}
for qname in sorted(query_gt.keys()):
    pos_set = set(query_gt[qname]["positive_oids"])
    neg_set = set(query_gt[qname]["negative_oids"])

    # Restrict to C13-visited
    c13_vis_pos = [oid for oid in c13_visited_oids if oid in pos_set]
    c13_vis_neg = [oid for oid in c13_visited_oids if oid in neg_set]

    def eval_on_subset(preds, oid_subset, pos_set):
        tp = fp = tn = fn = 0
        for oid in oid_subset:
            pred_true = _check_query(preds.get(oid, {}), qname)
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
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "positive_recall": round(pos_recall, 4),
            "negative_recall": round(neg_recall, 4),
            "balanced_accuracy": round(bal, 4),
            "n_objects": len(oid_subset),
        }

    # Also evaluate on the union of C13-visited and rprobe-visited
    # for comparison where both visited
    both_visited = c13_visited_oids & rprobe_visited_oids
    both_vis_pos = [oid for oid in both_visited if oid in pos_set]
    both_vis_neg = [oid for oid in both_visited if oid in neg_set]

    c13_subset = list(c13_visited_oids)
    rprobe_subset = list(rprobe_visited_oids)

    diag2[qname] = {
        "c13_visited_subset": {
            "c13_predictions": eval_on_subset(c13_preds["per_object"], c13_subset, pos_set),
            "c0b_predictions": eval_on_subset(c0b_preds["per_object"], c13_subset, pos_set),
            "random_probe_predictions": eval_on_subset(rprobe_preds["per_object"], c13_subset, pos_set),
        },
        "n_c13_visited": len(c13_visited_oids),
        "n_c13_visited_positive": len(c13_vis_pos),
        "n_c13_visited_negative": len(c13_vis_neg),
    }

# Summary
c13_vis_macro_bal = sum(d["c13_visited_subset"]["c13_predictions"]["balanced_accuracy"]
                        for d in diag2.values()) / max(len(diag2), 1)
c0b_on_c13vis_macro_bal = sum(d["c13_visited_subset"]["c0b_predictions"]["balanced_accuracy"]
                              for d in diag2.values()) / max(len(diag2), 1)
rprobe_on_c13vis_macro_bal = sum(d["c13_visited_subset"]["random_probe_predictions"]["balanced_accuracy"]
                                 for d in diag2.values()) / max(len(diag2), 1)

c13_vis_pos_r = sum(d["c13_visited_subset"]["c13_predictions"]["positive_recall"]
                    for d in diag2.values()) / max(len(diag2), 1)
c0b_on_c13vis_pos_r = sum(d["c13_visited_subset"]["c0b_predictions"]["positive_recall"]
                          for d in diag2.values()) / max(len(diag2), 1)

visited_subset_summary = {
    "c13_macro_bal_on_visited": round(c13_vis_macro_bal, 4),
    "c0b_macro_bal_on_same_subset": round(c0b_on_c13vis_macro_bal, 4),
    "random_probe_macro_bal_on_same_subset": round(rprobe_on_c13vis_macro_bal, 4),
    "c13_mean_pos_recall_on_visited": round(c13_vis_pos_r, 4),
    "c0b_mean_pos_recall_on_same_subset": round(c0b_on_c13vis_pos_r, 4),
    "c13_performs_better_on_visited": c13_vis_macro_bal > c0b_on_c13vis_macro_bal,
    "per_query": diag2,
}

# =============================================================================
# DIAGNOSTIC 3: Probe action relevance
# =============================================================================
n_probes = len(c13_probe_details)
n_diff_probes = sum(1 for p in c13_probe_details if p["is_differentiating"])
n_nondiff_probes = n_probes - n_diff_probes

diff_effective = sum(1 for p in c13_probe_details if p["is_differentiating"] and p["efficiency"] == "effective")
diff_zero = sum(1 for p in c13_probe_details if p["is_differentiating"] and p["efficiency"] == "zero_gain")
diff_harmful = sum(1 for p in c13_probe_details if p["is_differentiating"] and p["efficiency"] == "harmful")

nondiff_effective = sum(1 for p in c13_probe_details if not p["is_differentiating"] and p["efficiency"] == "effective")
nondiff_zero = sum(1 for p in c13_probe_details if not p["is_differentiating"] and p["efficiency"] == "zero_gain")
nondiff_harmful = sum(1 for p in c13_probe_details if not p["is_differentiating"] and p["efficiency"] == "harmful")

diag3 = {
    "n_probes": n_probes,
    "n_differentiating_action_probes": n_diff_probes,
    "n_non_differentiating_action_probes": n_nondiff_probes,
    "differentiating_action_probe_rate": round(n_diff_probes / max(n_probes, 1), 4),
    "differentiating_action_probes": {
        "n_effective": diff_effective,
        "n_zero_gain": diff_zero,
        "n_harmful": diff_harmful,
        "effective_rate": round(diff_effective / max(n_diff_probes, 1), 4),
        "zero_gain_rate": round(diff_zero / max(n_diff_probes, 1), 4),
        "harmful_rate": round(diff_harmful / max(n_diff_probes, 1), 4),
    },
    "non_differentiating_action_probes": {
        "n_effective": nondiff_effective,
        "n_zero_gain": nondiff_zero,
        "n_harmful": nondiff_harmful,
        "effective_rate": round(nondiff_effective / max(n_nondiff_probes, 1), 4),
        "zero_gain_rate": round(nondiff_zero / max(n_nondiff_probes, 1), 4),
        "harmful_rate": round(nondiff_harmful / max(n_nondiff_probes, 1), 4),
    },
    "per_probe": c13_probe_details,
}

# =============================================================================
# DIAGNOSTIC 4: Minority-subtype targeting
# =============================================================================
minority_oids = set()
majority_oids = set()
for oid in test_oids:
    if object_meta[oid]["is_majority"]:
        majority_oids.add(oid)
    else:
        minority_oids.add(oid)

n_majority = len(majority_oids)
n_minority = len(minority_oids)

c13_visited_majority = len(majority_oids & c13_visited_oids)
c13_visited_minority = len(minority_oids & c13_visited_oids)
c13_probed_majority = len(majority_oids & c13_probed_oids)
c13_probed_minority = len(minority_oids & c13_probed_oids)

majority_visit_rate = c13_visited_majority / max(n_majority, 1)
minority_visit_rate = c13_visited_minority / max(n_minority, 1)
majority_probe_rate = c13_probed_majority / max(c13_visited_majority, 1)
minority_probe_rate = c13_probed_minority / max(c13_visited_minority, 1)
minority_enrichment = minority_probe_rate / max(majority_probe_rate, 0.001)

# Per-category
diag4_categories = {}
for cat in ["wood_log", "stone_block", "apple", "wooden_pickaxe"]:
    cat_oids = {oid for oid in test_oids if object_meta[oid]["category"] == cat}
    cat_majority = {oid for oid in cat_oids if object_meta[oid]["is_majority"]}
    cat_minority = {oid for oid in cat_oids if not object_meta[oid]["is_majority"]}

    n_cat_maj = len(cat_majority)
    n_cat_min = len(cat_minority)

    vis_maj = len(cat_majority & c13_visited_oids)
    vis_min = len(cat_minority & c13_visited_oids)
    prb_maj = len(cat_majority & c13_probed_oids)
    prb_min = len(cat_minority & c13_probed_oids)

    cat_maj_visit_rate = vis_maj / max(n_cat_maj, 1)
    cat_min_visit_rate = vis_min / max(n_cat_min, 1)
    cat_maj_probe_rate = prb_maj / max(vis_maj, 1)
    cat_min_probe_rate = prb_min / max(vis_min, 1)
    cat_enrichment = cat_min_probe_rate / max(cat_maj_probe_rate, 0.001)

    diag4_categories[cat] = {
        "n_majority": n_cat_maj,
        "n_minority": n_cat_min,
        "c13_visited_majority": vis_maj,
        "c13_visited_minority": vis_min,
        "c13_probed_majority": prb_maj,
        "c13_probed_minority": prb_min,
        "majority_visit_rate": round(cat_maj_visit_rate, 4),
        "minority_visit_rate": round(cat_min_visit_rate, 4),
        "majority_probe_rate": round(cat_maj_probe_rate, 4),
        "minority_probe_rate": round(cat_min_probe_rate, 4),
        "minority_enrichment_ratio": round(cat_enrichment, 4),
    }

diag4 = {
    "total_objects": len(test_oids),
    "n_majority": n_majority,
    "n_minority": n_minority,
    "c13_visited_majority": c13_visited_majority,
    "c13_visited_minority": c13_visited_minority,
    "c13_probed_majority": c13_probed_majority,
    "c13_probed_minority": c13_probed_minority,
    "majority_visit_rate": round(majority_visit_rate, 4),
    "minority_visit_rate": round(minority_visit_rate, 4),
    "majority_probe_rate": round(majority_probe_rate, 4),
    "minority_probe_rate": round(minority_probe_rate, 4),
    "minority_enrichment_ratio": round(minority_enrichment, 4),
    "c13_preferentially_targets_minority": minority_enrichment > 1.1,
    "per_category": diag4_categories,
}

# =============================================================================
# DIAGNOSTIC 5: Probe value by subtype
# =============================================================================
maj_probes = [p for p in c13_probe_details if p["is_majority"]]
min_probes = [p for p in c13_probe_details if not p["is_majority"]]

def subtype_probe_stats(probes):
    n = len(probes)
    if n == 0:
        return {"n_probes": 0, "n_effective": 0, "n_zero_gain": 0, "n_harmful": 0,
                "effective_rate": 0.0, "zero_gain_rate": 0.0, "harmful_rate": 0.0}
    eff = sum(1 for p in probes if p["efficiency"] == "effective")
    zg = sum(1 for p in probes if p["efficiency"] == "zero_gain")
    hm = sum(1 for p in probes if p["efficiency"] == "harmful")
    return {
        "n_probes": n,
        "n_effective": eff,
        "n_zero_gain": zg,
        "n_harmful": hm,
        "effective_rate": round(eff / n, 4),
        "zero_gain_rate": round(zg / n, 4),
        "harmful_rate": round(hm / n, 4),
    }

diag5 = {
    "majority_subtype_objects": subtype_probe_stats(maj_probes),
    "minority_subtype_objects": subtype_probe_stats(min_probes),
    "minority_probes_more_effective": (
        subtype_probe_stats(min_probes)["effective_rate"]
        > subtype_probe_stats(maj_probes)["effective_rate"]
    ),
}

# =============================================================================
# DIAGNOSTIC 6: Positive recall loss decomposition
# =============================================================================
diag6 = {}
for qname in sorted(query_gt.keys()):
    pos_set = set(query_gt[qname]["positive_oids"])
    total_pos = len(pos_set)

    unvisited_pos = []
    visited_not_probed_pos = []
    probed_wrong_pos = []
    probe_changed_to_false_pos = []
    other_pos = []

    for oid in pos_set:
        if oid not in c13_visited_oids:
            unvisited_pos.append(oid)
        elif oid not in c13_probed_oids:
            # visited but not probed — check if prediction is correct
            pred_true = _check_query(c13_preds["per_object"].get(oid, {}), qname)
            if not pred_true:
                visited_not_probed_pos.append(oid)
        else:
            # probed — check if prediction is correct
            pred_true = _check_query(c13_preds["per_object"].get(oid, {}), qname)
            if not pred_true:
                # Check if probe pushed probability in wrong direction
                # Find the probe for this object (use full tracking data)
                obj_probes = [p for p in c13_policy.probe_tracking if p["oid"] == oid]
                changed_to_false = False
                for p in obj_probes:
                    feature = ACTION_TO_FEATURE.get(p["action"])
                    if feature:
                        pre_val = p["pre_probs"].get(feature, 0.5)
                        post_val = p["post_probs"].get(feature, 0.5)
                        # If pre was >= 0.5 (predicting True) and post < 0.5 (False)
                        if pre_val >= 0.5 and post_val < 0.5:
                            changed_to_false = True
                            break
                        # If pre was < 0.5 and probe didn't raise it enough
                        if pre_val < 0.5 and post_val < 0.5:
                            changed_to_false = True
                            break
                if changed_to_false:
                    probe_changed_to_false_pos.append(oid)
                else:
                    probed_wrong_pos.append(oid)
            # else: prediction is correct, no FN — don't count

    # Count actual FNs from metrics
    q_metrics = c13_metrics["per_query"][qname]
    actual_fn = q_metrics["fn"]

    diag6[qname] = {
        "total_positive": total_pos,
        "c13_fn_total": actual_fn,
        "A_unvisited": len(unvisited_pos),
        "B_visited_not_probed_wrong": len(visited_not_probed_pos),
        "C_probed_but_wrong": len(probed_wrong_pos),
        "D_probe_changed_to_false": len(probe_changed_to_false_pos),
        "A_pct": round(len(unvisited_pos) / max(actual_fn, 1), 4),
        "B_pct": round(len(visited_not_probed_pos) / max(actual_fn, 1), 4),
        "C_pct": round(len(probed_wrong_pos) / max(actual_fn, 1), 4),
        "D_pct": round(len(probe_changed_to_false_pos) / max(actual_fn, 1), 4),
    }

# Total across queries
total_fn = sum(d["c13_fn_total"] for d in diag6.values())
total_A = sum(d["A_unvisited"] for d in diag6.values())
total_B = sum(d["B_visited_not_probed_wrong"] for d in diag6.values())
total_C = sum(d["C_probed_but_wrong"] for d in diag6.values())
total_D = sum(d["D_probe_changed_to_false"] for d in diag6.values())

diag6_summary = {
    "total_fn_across_queries": total_fn,
    "total_A_unvisited": total_A,
    "total_B_visited_not_probed_wrong": total_B,
    "total_C_probed_but_wrong": total_C,
    "total_D_probe_changed_to_false": total_D,
    "A_pct_of_fn": round(total_A / max(total_fn, 1), 4),
    "B_pct_of_fn": round(total_B / max(total_fn, 1), 4),
    "C_pct_of_fn": round(total_C / max(total_fn, 1), 4),
    "D_pct_of_fn": round(total_D / max(total_fn, 1), 4),
    "per_query": diag6,
}

# =============================================================================
# DIAGNOSTIC 7: C13 vs random_probe comparison
# =============================================================================
diag7 = {}
for qname in sorted(query_gt.keys()):
    pos_set = set(query_gt[qname]["positive_oids"])

    c13_vis_pos = len(pos_set & c13_visited_oids)
    rprobe_vis_pos = len(pos_set & rprobe_visited_oids)
    total_pos = len(pos_set)

    diag7[qname] = {
        "c13_macro_bal": c13_metrics["per_query"][qname]["balanced_accuracy"],
        "random_macro_bal": rprobe_metrics["per_query"][qname]["balanced_accuracy"],
        "c13_positive_coverage": round(c13_vis_pos / max(total_pos, 1), 4),
        "random_positive_coverage": round(rprobe_vis_pos / max(total_pos, 1), 4),
        "c13_better": c13_metrics["per_query"][qname]["balanced_accuracy"]
                      > rprobe_metrics["per_query"][qname]["balanced_accuracy"],
    }

# Compare object selection: both visit same number of objects (22)
both_visit_count = len(c13_visited_oids & rprobe_visited_oids)
c13_only_visit = len(c13_visited_oids - rprobe_visited_oids)
rprobe_only_visit = len(rprobe_visited_oids - c13_visited_oids)

# Differentiating action rate
c13_diff_rate = n_diff_probes / max(n_probes, 1)
rprobe_n_diff = sum(1 for p in rprobe_probe_details if p["is_differentiating"])
rprobe_diff_rate = rprobe_n_diff / max(len(rprobe_probe_details), 1)

# Minority targeting
rprobe_visited_minority = len(minority_oids & rprobe_visited_oids)
rprobe_probed_minority = len(minority_oids & rprobe_probed_oids)
rprobe_minority_probe_rate = rprobe_probed_minority / max(rprobe_visited_minority, 1)

# Random probe efficiency
rprobe_n_probes = len(rprobe_probe_details)
rprobe_eff = sum(1 for p in rprobe_probe_details if p["efficiency"] == "effective")
rprobe_zg = sum(1 for p in rprobe_probe_details if p["efficiency"] == "zero_gain")
rprobe_hm = sum(1 for p in rprobe_probe_details if p["efficiency"] == "harmful")

diag7_summary = {
    "c13_vs_random_macro_bal_delta": round(
        c13_metrics["macro_query_balanced_accuracy"]
        - rprobe_metrics["macro_query_balanced_accuracy"], 4),
    "both_visit_same_count": c13_cost["visit_count"] == rprobe_cost["visit_count"],
    "c13_visit_count": c13_cost["visit_count"],
    "random_visit_count": rprobe_cost["visit_count"],
    "overlap_visited_objects": both_visit_count,
    "c13_only_visited": c13_only_visit,
    "random_only_visited": rprobe_only_visit,
    "c13_differentiating_action_probe_rate": round(c13_diff_rate, 4),
    "random_differentiating_action_probe_rate": round(rprobe_diff_rate, 4),
    "c13_minority_probe_rate": round(minority_probe_rate, 4),
    "random_minority_probe_rate": round(rprobe_minority_probe_rate, 4),
    "c13_effective_probe_rate": round(
        sum(1 for p in c13_probe_details if p["efficiency"] == "effective") / max(n_probes, 1), 4),
    "random_effective_probe_rate": round(rprobe_eff / max(rprobe_n_probes, 1), 4),
    "c13_advantage_from_better_action_selection": c13_diff_rate > rprobe_diff_rate,
    "per_query": diag7,
}

# =============================================================================
# DIAGNOSTIC 8: Category shortcut risk note
# =============================================================================
cat_maj_macro_bal = 0.9750  # from 1J3
diag8 = {
    "category_majority_macro_bal": cat_maj_macro_bal,
    "c_minus_macro_bal": c_minus_metrics["macro_query_balanced_accuracy"],
    "category_shortcut_risk": "strong",
    "agent_exploits_shortcut": c_minus_metrics["macro_query_balanced_accuracy"] > 0.5,
    "interpretation": (
        "category_majority = 0.9750 shows strong category-level shortcut risk. "
        "However, C_minus_prior_only = 0.5000 confirms the current zero-interaction "
        "agent does NOT exploit this shortcut. The shortcut is latent, not active."
    ),
}

# =============================================================================
# FINAL SYNTHESIS: Determine main diagnosis
# =============================================================================
print("[6/6] Final synthesis...")

# Check each diagnosis criterion

# 1. visit_breadth_loss: C13 performs well on visited objects but misses many positives
c13_good_on_visited = c13_vis_macro_bal >= 0.55  # reasonable threshold
misses_many_positives = total_A > 0  # any unvisited positives
visit_breadth_loss = c13_good_on_visited and misses_many_positives

# 2. poor_differentiating_action_selection: many probes are non-differentiating
poor_diff_action = c13_diff_rate < 0.5

# 3. poor_minority_subtype_targeting: probes majority at similar or higher rate
poor_minority_targeting = minority_enrichment <= 1.1

# 4. posterior_prediction_bias: C13 visited-subset positive recall < C0b
posterior_bias = c13_vis_pos_r < c0b_on_c13vis_pos_r

# 5. persistent_negative_only_tradeoff
neg_only_tradeoff = (
    c13_metrics["mean_positive_recall"] < c0b_metrics["mean_positive_recall"]
    and c13_metrics["mean_negative_recall"] > c0b_metrics["mean_negative_recall"]
)

# Build main diagnosis
diagnoses = []
if visit_breadth_loss:
    diagnoses.append("visit_breadth_loss")
if poor_diff_action:
    diagnoses.append("poor_differentiating_action_selection")
if poor_minority_targeting:
    diagnoses.append("poor_minority_subtype_targeting")
if posterior_bias:
    diagnoses.append("posterior_prediction_bias")
if neg_only_tradeoff:
    diagnoses.append("persistent_negative_only_tradeoff")
diagnoses.append("environment_category_shortcut_risk")

main_diagnosis = "+".join(diagnoses) if diagnoses else "mixed"

print(f"\n  Main diagnosis: {main_diagnosis}")
print(f"  visit_breadth_loss: {visit_breadth_loss}")
print(f"  poor_differentiating_action_selection: {poor_diff_action}  (diff_rate={c13_diff_rate:.3f})")
print(f"  poor_minority_subtype_targeting: {poor_minority_targeting}  (enrichment={minority_enrichment:.3f})")
print(f"  posterior_prediction_bias: {posterior_bias}  (C13_vis_pos_r={c13_vis_pos_r:.4f} vs C0b={c0b_on_c13vis_pos_r:.4f})")
print(f"  persistent_negative_only_tradeoff: {neg_only_tradeoff}")

# =============================================================================
# BUILD OUTPUT JSON
# =============================================================================
output = {
    "block_id": "1J4",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "c13_metrics": {
        "macro_query_balanced_accuracy": c13_metrics["macro_query_balanced_accuracy"],
        "mean_positive_recall": c13_metrics["mean_positive_recall"],
        "mean_negative_recall": c13_metrics["mean_negative_recall"],
        "per_query": c13_metrics["per_query"],
    },
    "c0b_metrics": {
        "macro_query_balanced_accuracy": c0b_metrics["macro_query_balanced_accuracy"],
        "mean_positive_recall": c0b_metrics["mean_positive_recall"],
        "mean_negative_recall": c0b_metrics["mean_negative_recall"],
        "per_query": c0b_metrics["per_query"],
    },
    "c13_cost": c13_cost,
    "c0b_cost": c0b_cost,
    "diagnostics": {
        "1_visit_breadth_loss": {
            "summary": visit_breadth_summary,
        },
        "2_visited_object_prediction": {
            "summary": visited_subset_summary,
        },
        "3_probe_action_relevance": {
            "summary": {k: v for k, v in diag3.items() if k != "per_probe"},
            "per_probe": diag3["per_probe"],
        },
        "4_minority_subtype_targeting": diag4,
        "5_probe_value_by_subtype": diag5,
        "6_positive_recall_loss_decomposition": {
            "summary": diag6_summary,
        },
        "7_c13_vs_random_probe": {
            "summary": diag7_summary,
            "random_probe_metrics": {
                "macro_query_balanced_accuracy": rprobe_metrics["macro_query_balanced_accuracy"],
                "per_query": rprobe_metrics["per_query"],
            },
        },
        "8_category_shortcut_risk_note": diag8,
    },
    "final_diagnosis": {
        "main_diagnosis": main_diagnosis,
        "visit_breadth_loss_supported": visit_breadth_loss,
        "poor_differentiating_action_selection_supported": poor_diff_action,
        "poor_minority_subtype_targeting_supported": poor_minority_targeting,
        "posterior_prediction_bias_supported": posterior_bias,
        "persistent_negative_only_tradeoff_supported": neg_only_tradeoff,
        "c13_fix_ready": False,
        "environment_change_ready": False,
        "ready_for_multiseed": False,
    },
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j4_c13_failure_decomposition_c4_seed101.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved: {json_path}")

# =============================================================================
# PRINT BLOCK DONE
# =============================================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"block_id=1J4")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"main_diagnosis={main_diagnosis}")
print(f"visit_breadth_loss_supported={visit_breadth_loss}")
print(f"poor_differentiating_action_selection_supported={poor_diff_action}")
print(f"poor_minority_subtype_targeting_supported={poor_minority_targeting}")
print(f"posterior_prediction_bias_supported={posterior_bias}")
print(f"persistent_negative_only_tradeoff_supported={neg_only_tradeoff}")
print(f"c13_fix_ready=false")
print(f"environment_change_ready=false")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*60}")
