"""
Block 1J2 — Marginal-Value Audit for C4_instance_subtype_cued_v1.

Evaluates 6 baselines to determine whether probing has measurable marginal value
above observation, and observation above prior, under balanced metrics.

No C13. No tuning. No multi-seed.
"""
import os
import sys
import json
import copy
import random
import math
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path setup
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
from harness import EpisodeHarness, HarnessResult
from evaluator import MiniMCEvaluator
from agent_obs import AgentObs, AgentObsView
from event_log import EventLog
from simulator_truth import MiniMCSimulatorTruth
from policies import (
    MiniMCPolicy, C0a_NoInteractionPolicy, C0b_ObserveOnlyPolicy,
    C14a_TruthAnswerOracle, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
    CORE_ACTION_FEATURES, _compute_utility,
)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes
from subtype_objects import (
    SUBTYPE_DEFINITIONS,
    generate_subtype_objects_deterministic,
    compute_query_ground_truth,
)

t0 = time.time()

# =============================================================================
# 0. Constants
# =============================================================================
SEED = 101
BUDGET = 1.5
COND = copy.deepcopy(config.CUE_CONDITIONS[3])  # C4_instance_subtype_cued_v1
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J2 — Marginal-Value Audit")
print(f"  condition={COND['label']}")
print(f"  seed={SEED}")
print(f"  budget={BUDGET}")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training (standard objects, C4 cue condition)
# =============================================================================
print("\n[1/6] Phase A: Training student on standard objects...")
(student, base_learner, train_objects, train_env,
 _std_test_objects, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
print(f"  train_objects={len(train_objects_dict)}")
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}")

# =============================================================================
# 2. Phase A': Sparse outcomes + Phase B: IOM
# =============================================================================
print("\n[2/6] Building InstanceOutcomeMemory...")
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED
)
print(f"  outcome_rows={len(outcome_rows)}  actual_coverage={actual_coverage:.4f}")

posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)
print(f"  IOM built: k={config.INSTANCE_K}  mode={config.SIMILARITY_MODE}")

# =============================================================================
# 3. Generate subtype test objects
# =============================================================================
print("\n[3/6] Generating subtype test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")

# Compute query ground truth
query_gt = compute_query_ground_truth(test_objects)

# Verify subtype distribution
subtype_counts = {}
for oid, obj in test_objects.items():
    key = f"{obj['hidden_category']}::{obj['hidden_subtype']}"
    subtype_counts[key] = subtype_counts.get(key, 0) + 1
print(f"  test_objects={len(test_objects)}")
for key in sorted(subtype_counts.keys()):
    print(f"    {key}: {subtype_counts[key]}")

# Verify query positive rates
print("  Query positive rates:")
for qname in sorted(query_gt.keys()):
    data = query_gt[qname]
    n_pos = len(data["positive_oids"])
    n_neg = len(data["negative_oids"])
    pos_rate = n_pos / (n_pos + n_neg)
    print(f"    {qname}: pos={n_pos} neg={n_neg} pos_rate={pos_rate:.4f}")

# =============================================================================
# 4. Assign positions and build environment
# =============================================================================
print("\n[4/6] Assigning positions and building environment...")
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng
)
env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
sim = env.simulator
evaluator = MiniMCEvaluator(env)
print(f"  grid={config.GRID_ROWS}x{config.GRID_COLS}  agent_start={config.AGENT_START}")

# =============================================================================
# 5. Helper: query checking (same logic as evaluator._check_query)
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


# =============================================================================
# 6. Macro query balanced accuracy computation
# =============================================================================
def compute_macro_bal(predictions, query_gt):
    """Compute macro-averaged query balanced accuracy.

    predictions: {oid: {feature_name: probability}}
    query_gt: {query_name: {"positive_oids": [...], "negative_oids": [...]}}
    """
    per_query = {}
    for qname in sorted(query_gt.keys()):
        pos_set = set(query_gt[qname]["positive_oids"])
        neg_set = set(query_gt[qname]["negative_oids"])
        tp, fp, tn, fn = 0, 0, 0, 0
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
    return {
        "per_query": per_query,
        "macro_query_balanced_accuracy": round(macro_bal, 6),
        "macro_query_accuracy": round(macro_acc, 6),
    }


# =============================================================================
# 7. Helper: run a policy through the harness (or get predictions directly)
# =============================================================================
def run_policy(policy, env):
    """Run policy through EpisodeHarness. Returns (predictions, event_log, agent_obs)."""
    if isinstance(policy, C14a_TruthAnswerOracle):
        predictions = policy.get_answer()
        obs = env.reset()
        obs.event_log.append({"event": "C14a_truth_oracle",
                              "note": "no budget, no harness"})
        return predictions, obs.event_log, obs, {}, {}
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


def get_cost_metrics(event_log, agent_obs):
    """Extract cost metrics from event log."""
    events = event_log.to_list()
    total_reach = sum(e.get("cost", 0.0) for e in events if e.get("event") == "reach")
    total_observe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "observe")
    total_probe = sum(e.get("cost", 0.0) for e in events if e.get("event") == "probe")
    total_cost = event_log.total_cost()
    n_visited = sum(1 for oid in agent_obs.get_all_object_ids()
                    if agent_obs.is_visited(oid))
    n_probed = sum(1 for oid in agent_obs.get_all_object_ids()
                   if agent_obs.get_probe_results(oid))
    initial_budget = agent_obs.initial_budget
    normalized_cost = total_cost / max(initial_budget, 0.001)
    return {
        "visit_count": n_visited,
        "probe_count": n_probed,
        "total_cost": total_cost,
        "normalized_cost": normalized_cost,
        "total_reach_cost": total_reach,
        "total_observe_cost": total_observe,
        "total_probe_cost": total_probe,
        "budget_remaining": agent_obs.budget_remaining,
        "budget_spent": agent_obs.budget_spent,
        "initial_budget": initial_budget,
    }


# =============================================================================
# 8. Random Probe Budgeted policy
# =============================================================================
class RandomProbeBudgetedPolicy(MiniMCPolicy):
    """Visits nearest affordable objects, randomly probes one action each."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng

    def reset(self, view):
        pass

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
        return best_oid

    def decide_probe(self, view, object_id):
        action = self._rng.choice(MAIN_CANDIDATE_ACTIONS)
        return True, action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


# =============================================================================
# 9. Run all 6 baselines
# =============================================================================
print("\n[5/6] Running baselines...")
baseline_results = {}

# ----- 9a. all_false -----
print("  all_false...")
all_false_predictions = {}
for oid in test_oids:
    all_false_predictions[oid] = {f: 0.0 for f in CORE_ACTION_FEATURES}
all_false_bal = compute_macro_bal(all_false_predictions, query_gt)
# Cost: zero (no visits)
all_false_cost = {
    "visit_count": 0, "probe_count": 0,
    "total_cost": 0.0, "normalized_cost": 0.0,
    "total_reach_cost": 0.0, "total_observe_cost": 0.0, "total_probe_cost": 0.0,
    "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET,
}
# raw_comp using evaluator
dummy_log = EventLog()
dummy_obs = AgentObs(sim, BUDGET, config.REACH_COST_PER_UNIT,
                     config.OBSERVE_COST, config.PROBE_COST, dummy_log)
all_false_eval = evaluator.evaluate(
    {"per_object": all_false_predictions}, dummy_log, dummy_obs
)
all_false_raw_comp = all_false_eval["composite_task_accuracy"]
baseline_results["all_false"] = {
    "raw_comp": all_false_raw_comp,
    "macro_query_balanced_accuracy": all_false_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": all_false_bal["macro_query_accuracy"],
    "per_query": all_false_bal["per_query"],
    "cost": all_false_cost,
}
print(f"    macro_bal={all_false_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={all_false_raw_comp:.4f}  cost=0")

# ----- 9b. C_minus_prior_only -----
print("  C_minus_prior_only...")
c_minus_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c_minus_im = im_base.clone()
c_minus_policy = C0a_NoInteractionPolicy(c_minus_im, random.Random(SEED))
c_minus_preds, c_minus_log, c_minus_obs, _, _ = run_policy(c_minus_policy, c_minus_env)
c_minus_bal = compute_macro_bal(c_minus_preds["per_object"], query_gt)
c_minus_cost = get_cost_metrics(c_minus_log, c_minus_obs)
c_minus_eval = evaluator.evaluate(c_minus_preds, c_minus_log, c_minus_obs)
baseline_results["C_minus_prior_only"] = {
    "raw_comp": c_minus_eval["composite_task_accuracy"],
    "macro_query_balanced_accuracy": c_minus_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": c_minus_bal["macro_query_accuracy"],
    "per_query": c_minus_bal["per_query"],
    "cost": c_minus_cost,
}
print(f"    macro_bal={c_minus_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={c_minus_eval['composite_task_accuracy']:.4f}  "
      f"visited={c_minus_cost['visit_count']}  probed={c_minus_cost['probe_count']}  "
      f"cost={c_minus_cost['total_cost']:.4f}")

# ----- 9c. C0b_observe_only -----
print("  C0b_observe_only...")
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_im = im_base.clone()
c0b_policy = C0b_ObserveOnlyPolicy(c0b_im, random.Random(SEED + 100))
c0b_preds, c0b_log, c0b_obs, c0b_pre_probe, c0b_pre_decision = run_policy(c0b_policy, c0b_env)
c0b_bal = compute_macro_bal(c0b_preds["per_object"], query_gt)
c0b_cost = get_cost_metrics(c0b_log, c0b_obs)
c0b_eval = evaluator.evaluate(c0b_preds, c0b_log, c0b_obs,
                               c0b_pre_probe, c0b_pre_decision)
baseline_results["C0b_observe_only"] = {
    "raw_comp": c0b_eval["composite_task_accuracy"],
    "macro_query_balanced_accuracy": c0b_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": c0b_bal["macro_query_accuracy"],
    "per_query": c0b_bal["per_query"],
    "cost": c0b_cost,
}
print(f"    macro_bal={c0b_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={c0b_eval['composite_task_accuracy']:.4f}  "
      f"visited={c0b_cost['visit_count']}  probed={c0b_cost['probe_count']}  "
      f"cost={c0b_cost['total_cost']:.4f}")

# ----- 9d. C_oracle_full_information -----
print("  C_oracle_full_information...")
oracle_sim = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)
oracle_policy = C14a_TruthAnswerOracle(oracle_sim)
oracle_preds = oracle_policy.get_answer()
oracle_bal = compute_macro_bal(oracle_preds["per_object"], query_gt)
# Oracle is unconstrained — cost tracked separately
oracle_log = EventLog()
oracle_log.append({"event": "C14a_truth_oracle", "note": "unconstrained full information"})
oracle_obs = AgentObs(oracle_sim, BUDGET, config.REACH_COST_PER_UNIT,
                      config.OBSERVE_COST, config.PROBE_COST, oracle_log)
oracle_eval = evaluator.evaluate(oracle_preds, oracle_log, oracle_obs)
oracle_cost = {
    "visit_count": 0, "probe_count": 0,
    "total_cost": 0.0, "normalized_cost": 0.0,
    "total_reach_cost": 0.0, "total_observe_cost": 0.0, "total_probe_cost": 0.0,
    "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET,
    "note": "oracle_unconstrained — cost is zero, full information ceiling",
}
baseline_results["C_oracle_full_information"] = {
    "raw_comp": oracle_eval["composite_task_accuracy"],
    "macro_query_balanced_accuracy": oracle_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": oracle_bal["macro_query_accuracy"],
    "per_query": oracle_bal["per_query"],
    "cost": oracle_cost,
}
print(f"    macro_bal={oracle_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={oracle_eval['composite_task_accuracy']:.4f}  "
      f"cost=0 (unconstrained)")

# ----- 9e. random_probe_budgeted -----
print("  random_probe_budgeted...")
rprobe_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
rprobe_im = im_base.clone()
rprobe_policy = RandomProbeBudgetedPolicy(rprobe_im, random.Random(SEED + 300))
rprobe_preds, rprobe_log, rprobe_obs, rprobe_pre_probe, rprobe_pre_decision = \
    run_policy(rprobe_policy, rprobe_env)
rprobe_bal = compute_macro_bal(rprobe_preds["per_object"], query_gt)
rprobe_cost = get_cost_metrics(rprobe_log, rprobe_obs)
rprobe_eval = evaluator.evaluate(rprobe_preds, rprobe_log, rprobe_obs,
                                  rprobe_pre_probe, rprobe_pre_decision)
baseline_results["random_probe_budgeted"] = {
    "raw_comp": rprobe_eval["composite_task_accuracy"],
    "macro_query_balanced_accuracy": rprobe_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": rprobe_bal["macro_query_accuracy"],
    "per_query": rprobe_bal["per_query"],
    "cost": rprobe_cost,
}
print(f"    macro_bal={rprobe_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={rprobe_eval['composite_task_accuracy']:.4f}  "
      f"visited={rprobe_cost['visit_count']}  probed={rprobe_cost['probe_count']}  "
      f"cost={rprobe_cost['total_cost']:.4f}")

# ----- 9f. category_majority -----
print("  category_majority...")
# Use training-set category majority (from standard AFFORDANCE_PROFILES)
# Per-query category majority from training data:
# need_planks: wood_log=True (all have craft_plank=success), others=False
# need_stone: stone_block=True (all have mine_by_hand=fail AND mine_with_pickaxe=success), others=False
# need_food: apple=True (all have eat=success), others=False
# need_tool: wooden_pickaxe=True (all have use_as_tool=success), others=False
# need_fuel: wood_log=True, wooden_pickaxe=True (both have burn_as_fuel=success), others=False
from objects import AFFORDANCE_PROFILES
cat_maj_predictions = {}
for oid in test_oids:
    cat = test_objects[oid]["hidden_category"]
    std_profile = AFFORDANCE_PROFILES.get(cat, {})
    probs = {}
    for action, result in std_profile.items():
        if action in ("tap_sound", "push", "roll", "inspect"):
            continue
        feat = f"{action}_success"
        probs[feat] = 1.0 if result == "success" else 0.0
    for f in CORE_ACTION_FEATURES:
        if f not in probs:
            probs[f] = 0.5
    cat_maj_predictions[oid] = probs

cat_maj_bal = compute_macro_bal(cat_maj_predictions, query_gt)
cat_maj_eval = evaluator.evaluate(
    {"per_object": cat_maj_predictions}, dummy_log, dummy_obs
)
cat_maj_cost = {
    "visit_count": 0, "probe_count": 0,
    "total_cost": 0.0, "normalized_cost": 0.0,
    "total_reach_cost": 0.0, "total_observe_cost": 0.0, "total_probe_cost": 0.0,
    "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET,
}
baseline_results["category_majority"] = {
    "raw_comp": cat_maj_eval["composite_task_accuracy"],
    "macro_query_balanced_accuracy": cat_maj_bal["macro_query_balanced_accuracy"],
    "macro_query_accuracy": cat_maj_bal["macro_query_accuracy"],
    "per_query": cat_maj_bal["per_query"],
    "cost": cat_maj_cost,
}
print(f"    macro_bal={cat_maj_bal['macro_query_balanced_accuracy']:.4f}  "
      f"raw_comp={cat_maj_eval['composite_task_accuracy']:.4f}  cost=0")

# =============================================================================
# 10. Compute deltas and acceptance criteria
# =============================================================================
print("\n[6/6] Computing deltas and acceptance checks...")

def get_bal(result):
    return result["macro_query_balanced_accuracy"]

def get_pos_recall(result):
    return sum(v["positive_recall"] for v in result["per_query"].values()) / max(len(result["per_query"]), 1)

def get_neg_recall(result):
    return sum(v["negative_recall"] for v in result["per_query"].values()) / max(len(result["per_query"]), 1)

all_false_res = baseline_results["all_false"]
c_minus_res = baseline_results["C_minus_prior_only"]
c0b_res = baseline_results["C0b_observe_only"]
oracle_res = baseline_results["C_oracle_full_information"]
rprobe_res = baseline_results["random_probe_budgeted"]
cat_maj_res = baseline_results["category_majority"]

# Core metrics
all_false_macro_bal = get_bal(all_false_res)
c_minus_macro_bal = get_bal(c_minus_res)
c0b_macro_bal = get_bal(c0b_res)
oracle_macro_bal = get_bal(oracle_res)
rprobe_macro_bal = get_bal(rprobe_res)
cat_maj_macro_bal = get_bal(cat_maj_res)

# Deltas
delta_c0b_minus_cminus = c0b_macro_bal - c_minus_macro_bal
delta_oracle_minus_c0b = oracle_macro_bal - c0b_macro_bal
delta_oracle_minus_cminus = oracle_macro_bal - c_minus_macro_bal
delta_rprobe_minus_c0b = rprobe_macro_bal - c0b_macro_bal
delta_cat_maj_minus_c0b = cat_maj_macro_bal - c0b_macro_bal
delta_cat_maj_minus_cminus = cat_maj_macro_bal - c_minus_macro_bal

# Positive/negative recall deltas
c0b_pos_recall = get_pos_recall(c0b_res)
c0b_neg_recall = get_neg_recall(c0b_res)
oracle_pos_recall = get_pos_recall(oracle_res)
oracle_neg_recall = get_neg_recall(oracle_res)
delta_pos_recall = oracle_pos_recall - c0b_pos_recall
delta_neg_recall = oracle_neg_recall - c0b_neg_recall

deltas = {
    "C0b_minus_Cminus": round(delta_c0b_minus_cminus, 6),
    "C_oracle_minus_C0b": round(delta_oracle_minus_c0b, 6),
    "C_oracle_minus_Cminus": round(delta_oracle_minus_cminus, 6),
    "random_probe_minus_C0b": round(delta_rprobe_minus_c0b, 6),
    "category_majority_minus_C0b": round(delta_cat_maj_minus_c0b, 6),
    "category_majority_minus_Cminus": round(delta_cat_maj_minus_cminus, 6),
    "oracle_pos_recall_minus_C0b_pos_recall": round(delta_pos_recall, 6),
    "oracle_neg_recall_minus_C0b_neg_recall": round(delta_neg_recall, 6),
}

# =============================================================================
# 11. Acceptance criteria checks
# =============================================================================
checks = {}

# Mandatory
checks["1_all_false_macro_bal_approx_0.5"] = abs(all_false_macro_bal - 0.5) < 0.01
checks["2_oracle_macro_bal_ge_0.7"] = oracle_macro_bal >= 0.70
checks["3_oracle_minus_c0b_ge_0.1"] = delta_oracle_minus_c0b >= 0.10
checks["4_c0b_minus_cminus_ge_0.05"] = delta_c0b_minus_cminus >= 0.05
checks["5_oracle_pos_recall_minus_c0b_pos_recall_ge_0.1"] = delta_pos_recall >= 0.10
checks["6_oracle_neg_recall_minus_c0b_neg_recall_ge_0.1"] = delta_neg_recall >= 0.10

# Recommended
checks["7_c0b_macro_bal_0.55_to_0.65"] = 0.55 <= c0b_macro_bal <= 0.65
checks["8_oracle_improves_both_pos_and_neg"] = (
    delta_pos_recall > 0 and delta_neg_recall > 0
)
checks["9_random_probe_reported"] = rprobe_res is not None  # always true

# Special checks
checks["category_majority_dominates_environment"] = (
    cat_maj_macro_bal >= oracle_macro_bal - 0.02
)
checks["probe_marginal_value_insufficient"] = delta_oracle_minus_c0b < 0.10
checks["observation_value_insufficient"] = delta_c0b_minus_cminus < 0.05
# Check if positive recall is near zero for C_minus or C0b
c_minus_pos_recall = get_pos_recall(c_minus_res)
checks["positive_discovery_still_hard"] = (
    c_minus_pos_recall < 0.10 or c0b_pos_recall < 0.10
)

# Overall pass/fail (all mandatory must pass)
mandatory_pass = all(checks[k] for k in [
    "1_all_false_macro_bal_approx_0.5",
    "2_oracle_macro_bal_ge_0.7",
    "3_oracle_minus_c0b_ge_0.1",
    "4_c0b_minus_cminus_ge_0.05",
    "5_oracle_pos_recall_minus_c0b_pos_recall_ge_0.1",
    "6_oracle_neg_recall_minus_c0b_neg_recall_ge_0.1",
])

# =============================================================================
# 12. Print audit summary
# =============================================================================
print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)

print(f"\n  {'Baseline':<30} {'macro_bal':<12} {'raw_comp':<12} {'visits':<8} {'probes':<8} {'cost':<10} {'ncost':<10}")
print("  " + "-" * 90)
for name in ["all_false", "C_minus_prior_only", "C0b_observe_only",
             "C_oracle_full_information", "random_probe_budgeted", "category_majority"]:
    res = baseline_results[name]
    c = res["cost"]
    print(f"  {name:<30} {get_bal(res):<12.4f} {res['raw_comp']:<12.4f} "
          f"{c['visit_count']:<8} {c['probe_count']:<8} "
          f"{c['total_cost']:<10.4f} {c['normalized_cost']:<10.4f}")

print(f"\n  Deltas:")
print(f"    C0b - C_minus:              {delta_c0b_minus_cminus:+.4f}")
print(f"    C_oracle - C0b:             {delta_oracle_minus_c0b:+.4f}")
print(f"    C_oracle - C_minus:         {delta_oracle_minus_cminus:+.4f}")
print(f"    random_probe - C0b:         {delta_rprobe_minus_c0b:+.4f}")
print(f"    category_majority - C0b:    {delta_cat_maj_minus_c0b:+.4f}")
print(f"    category_majority - Cminus: {delta_cat_maj_minus_cminus:+.4f}")
print(f"    oracle pos_recall - C0b pos_recall: {delta_pos_recall:+.4f}")
print(f"    oracle neg_recall - C0b neg_recall: {delta_neg_recall:+.4f}")

print(f"\n  Per-query details:")
for name in ["all_false", "C_minus_prior_only", "C0b_observe_only",
             "C_oracle_full_information", "random_probe_budgeted", "category_majority"]:
    res = baseline_results[name]
    print(f"\n  [{name}]  macro_bal={get_bal(res):.4f}")
    for qname in sorted(res["per_query"].keys()):
        q = res["per_query"][qname]
        print(f"    {qname:<20} pos_rec={q['positive_recall']:.4f}  "
              f"neg_rec={q['negative_recall']:.4f}  bal={q['balanced_accuracy']:.4f}  "
              f"acc={q['accuracy']:.4f}")

print(f"\n  Acceptance Criteria:")
for check_name in sorted(checks.keys()):
    result = checks[check_name]
    flag = "PASS" if result else "FAIL"
    print(f"    [{flag}] {check_name}")

print(f"\n  Overall:")
print(f"    mandatory_checks_pass: {mandatory_pass}")
print(f"    environment_passed_marginal_audit: {mandatory_pass}")

# =============================================================================
# 13. Save JSON
# =============================================================================
print("\n" + "=" * 60)
print("Saving outputs...")

# Build comprehensive output
audit_output = {
    "block_id": "1J2",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "config": {
        "grid_rows": config.GRID_ROWS,
        "grid_cols": config.GRID_COLS,
        "agent_start": config.AGENT_START,
        "reach_cost_per_unit": config.REACH_COST_PER_UNIT,
        "observe_cost": config.OBSERVE_COST,
        "probe_cost": config.PROBE_COST,
        "instance_k": config.INSTANCE_K,
        "similarity_mode": config.SIMILARITY_MODE,
        "coverage": config.COVERAGE,
    },
    "baselines": {},
    "deltas": deltas,
    "acceptance_checks": {k: v for k, v in checks.items()},
    "mandatory_checks_pass": mandatory_pass,
    "environment_passed_marginal_audit": mandatory_pass,
    "ready_for_policy_comparison": mandatory_pass,
}

for name, res in baseline_results.items():
    audit_output["baselines"][name] = {
        "macro_query_balanced_accuracy": get_bal(res),
        "macro_query_accuracy": res["macro_query_accuracy"],
        "raw_comp": res["raw_comp"],
        "per_query": res["per_query"],
        "cost": res["cost"],
    }

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j2_marginal_value_audit_c4.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(audit_output, f, indent=2)
print(f"Saved: {json_path}")

# =============================================================================
# 14. Block done
# =============================================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"  block_id=1J2")
print(f"  condition={COND['label']}")
print(f"  all_false_macro_bal={all_false_macro_bal:.4f}")
print(f"  c_minus_macro_bal={c_minus_macro_bal:.4f}")
print(f"  c0b_macro_bal={c0b_macro_bal:.4f}")
print(f"  oracle_macro_bal={oracle_macro_bal:.4f}")
print(f"  random_probe_macro_bal={rprobe_macro_bal:.4f}")
print(f"  category_majority_macro_bal={cat_maj_macro_bal:.4f}")
print(f"  delta_c0b_minus_cminus={delta_c0b_minus_cminus:+.4f}")
print(f"  delta_oracle_minus_c0b={delta_oracle_minus_c0b:+.4f}")
print(f"  environment_passed_marginal_audit={mandatory_pass}")
print(f"  ready_for_policy_comparison={mandatory_pass}")
print(f"  c13_runs=false")
print(f"  elapsed={elapsed:.1f}s")
print(f"{'='*60}")
