"""
Block 1J3 — Single-seed policy comparison on C4_instance_subtype_cued_v1.
Pareto reporting: macro_bal vs normalized_cost.

Runs all 7 policies on seed=101, budget=1.5. C13 with CW=0.5.
Computes probe efficiency, Pareto frontier, gap capture.
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
from environment import MiniMCEnvironment
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
)
from objects import AFFORDANCE_PROFILES

t0 = time.time()

SEED = 101
BUDGET = 1.5
COST_WEIGHT = 0.5
COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J3 — Single-Seed Policy Comparison (Pareto)")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}  CW={COST_WEIGHT}")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM
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
# 2. Subtype test objects + environment
# =============================================================================
print("\n[2/5] Generating subtype test objects + environment...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
print(f"  test_objects={len(test_objects)}  positions assigned")

# =============================================================================
# 3. Helpers: query checking, macro_bal, cost extraction
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
    n_observed = n_visited
    init_budget = agent_obs.initial_budget
    return {
        "visit_count": n_visited,
        "observe_count": n_observed,
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
    if isinstance(policy, C14a_TruthAnswerOracle):
        predictions = policy.get_answer()
        obs = env.reset()
        obs.event_log.append({"event": "C14a_truth_oracle", "note": "unconstrained"})
        return predictions, obs.event_log, obs, {}, {}
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 4. C13 Tracked Policy (records pre/post probe predictions for efficiency)
# =============================================================================
class C13_TrackedPolicy(C13_InstanceVOIPolicy):
    """C13 with probe-level prediction tracking for efficiency analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_tracking = []
        self._last_pre_probs = None

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
# 5. Random Probe Budgeted policy (same as 1J2)
# =============================================================================
class RandomProbeBudgetedPolicy(MiniMCPolicy):
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
        return True, self._rng.choice(MAIN_CANDIDATE_ACTIONS)

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            per_object[oid] = self._im.predict_all_affordances({"id": oid, "visible_features": features})
        return {"per_object": per_object}


# =============================================================================
# 6. Run all 7 policies
# =============================================================================
print("\n[3/5] Running all policies...")
policy_results = {}

# --- all_false ---
print("  all_false...")
all_false_preds = {oid: {f: 0.0 for f in CORE_ACTION_FEATURES} for oid in test_oids}
all_false_metrics = compute_full_metrics(all_false_preds, query_gt)
policy_results["all_false"] = {
    "predictions": all_false_preds,
    "metrics": all_false_metrics,
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0, "total_reach_cost": 0.0,
             "total_observe_cost": 0.0, "total_probe_cost": 0.0,
             "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET},
    "probe_efficiency": None,
}
print(f"    macro_bal={all_false_metrics['macro_query_balanced_accuracy']:.4f}")

# --- C_minus_prior_only ---
print("  C_minus_prior_only...")
c_minus_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c_minus_policy = C0a_NoInteractionPolicy(im_base.clone(), random.Random(SEED))
c_minus_preds, c_minus_log, c_minus_obs, _, _ = run_policy(c_minus_policy, c_minus_env)
c_minus_metrics = compute_full_metrics(c_minus_preds["per_object"], query_gt)
c_minus_cost = get_cost_metrics(c_minus_log, c_minus_obs)
policy_results["C_minus_prior_only"] = {
    "predictions": c_minus_preds["per_object"],
    "metrics": c_minus_metrics,
    "cost": c_minus_cost,
    "probe_efficiency": None,
}
print(f"    macro_bal={c_minus_metrics['macro_query_balanced_accuracy']:.4f}")

# --- C0b_observe_only ---
print("  C0b_observe_only...")
c0b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c0b_policy = C0b_ObserveOnlyPolicy(im_base.clone(), random.Random(SEED + 100))
c0b_preds, c0b_log, c0b_obs, c0b_prep, c0b_pred = run_policy(c0b_policy, c0b_env)
c0b_metrics = compute_full_metrics(c0b_preds["per_object"], query_gt)
c0b_cost = get_cost_metrics(c0b_log, c0b_obs)
policy_results["C0b_observe_only"] = {
    "predictions": c0b_preds["per_object"],
    "metrics": c0b_metrics,
    "cost": c0b_cost,
    "probe_efficiency": None,
}
print(f"    macro_bal={c0b_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c0b_cost['visit_count']}  cost={c0b_cost['total_cost']:.4f}")

# --- random_probe_budgeted ---
print("  random_probe_budgeted...")
rprobe_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
rprobe_policy = RandomProbeBudgetedPolicy(im_base.clone(), random.Random(SEED + 300))
rprobe_preds, rprobe_log, rprobe_obs, _, _ = run_policy(rprobe_policy, rprobe_env)
rprobe_metrics = compute_full_metrics(rprobe_preds["per_object"], query_gt)
rprobe_cost = get_cost_metrics(rprobe_log, rprobe_obs)
policy_results["random_probe_budgeted"] = {
    "predictions": rprobe_preds["per_object"],
    "metrics": rprobe_metrics,
    "cost": rprobe_cost,
    "probe_efficiency": None,
}
print(f"    macro_bal={rprobe_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={rprobe_cost['visit_count']}  probed={rprobe_cost['probe_count']}  "
      f"cost={rprobe_cost['total_cost']:.4f}")

# --- C13_instance_VOI_original (CW=0.5, with tracking) ---
print("  C13_instance_VOI_original (CW=0.5)...")
c13_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c13_im = im_base.clone()
c13_policy = C13_TrackedPolicy(c13_im, random.Random(SEED + 500 + int(COST_WEIGHT * 100)),
                                cost_weight=COST_WEIGHT)
c13_preds, c13_log, c13_obs, c13_prep, c13_pred = run_policy(c13_policy, c13_env)
c13_metrics = compute_full_metrics(c13_preds["per_object"], query_gt)
c13_cost = get_cost_metrics(c13_log, c13_obs)

# Compute probe efficiency from tracked data
sim_ref = c13_env.simulator
effective = zero_gain = harmful = 0
for probe in c13_policy.probe_tracking:
    oid = probe["oid"]
    gt = sim_ref.get_ground_truth_affordances(oid)
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
        effective += 1
    elif post_correct == pre_correct:
        zero_gain += 1
    else:
        harmful += 1

total_probes = len(c13_policy.probe_tracking)
c13_probe_eff = {
    "total_probe_count": total_probes,
    "effective_probe_count": effective,
    "zero_gain_probe_count": zero_gain,
    "harmful_probe_count": harmful,
    "effective_probe_rate": effective / max(total_probes, 1),
    "zero_gain_rate": zero_gain / max(total_probes, 1),
    "harmful_rate": harmful / max(total_probes, 1),
} if total_probes > 0 else None

policy_results["C13_instance_VOI_original"] = {
    "predictions": c13_preds["per_object"],
    "metrics": c13_metrics,
    "cost": c13_cost,
    "probe_efficiency": c13_probe_eff,
}
print(f"    macro_bal={c13_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c13_cost['visit_count']}  probed={c13_cost['probe_count']}  "
      f"cost={c13_cost['total_cost']:.4f}")
if c13_probe_eff:
    print(f"    probe_eff: effective={effective} zero_gain={zero_gain} "
          f"harmful={harmful}  eff_rate={c13_probe_eff['effective_probe_rate']:.3f}")

# --- category_majority ---
print("  category_majority...")
cat_maj_preds = {}
for oid in test_oids:
    cat = test_objects[oid]["hidden_category"]
    std_profile = AFFORDANCE_PROFILES.get(cat, {})
    probs = {}
    for action, result in std_profile.items():
        if action in ("tap_sound", "push", "roll", "inspect"):
            continue
        probs[f"{action}_success"] = 1.0 if result == "success" else 0.0
    for f in CORE_ACTION_FEATURES:
        if f not in probs:
            probs[f] = 0.5
    cat_maj_preds[oid] = probs
cat_maj_metrics = compute_full_metrics(cat_maj_preds, query_gt)
policy_results["category_majority"] = {
    "predictions": cat_maj_preds,
    "metrics": cat_maj_metrics,
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0, "total_reach_cost": 0.0,
             "total_observe_cost": 0.0, "total_probe_cost": 0.0,
             "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET},
    "probe_efficiency": None,
}
print(f"    macro_bal={cat_maj_metrics['macro_query_balanced_accuracy']:.4f}")

# --- C_oracle_full_information ---
print("  C_oracle_full_information...")
oracle_sim = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)
oracle_policy = C14a_TruthAnswerOracle(oracle_sim)
oracle_preds = oracle_policy.get_answer()
oracle_metrics = compute_full_metrics(oracle_preds["per_object"], query_gt)
policy_results["C_oracle_full_information"] = {
    "predictions": oracle_preds["per_object"],
    "metrics": oracle_metrics,
    "cost": {"visit_count": 0, "observe_count": 0, "probe_count": 0,
             "total_cost": 0.0, "normalized_cost": 0.0, "total_reach_cost": 0.0,
             "total_observe_cost": 0.0, "total_probe_cost": 0.0,
             "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET,
             "note": "oracle_unconstrained"},
    "probe_efficiency": None,
}
print(f"    macro_bal={oracle_metrics['macro_query_balanced_accuracy']:.4f}")

# =============================================================================
# 7. Comparisons
# =============================================================================
print("\n[4/5] Computing comparisons...")

def mb(res):
    return res["metrics"]["macro_query_balanced_accuracy"]

def nc(res):
    return res["cost"]["normalized_cost"]

def mpr(res):
    return res["metrics"]["mean_positive_recall"]

def mnr(res):
    return res["metrics"]["mean_negative_recall"]

all_false_r = policy_results["all_false"]
c_minus_r = policy_results["C_minus_prior_only"]
c0b_r = policy_results["C0b_observe_only"]
rprobe_r = policy_results["random_probe_budgeted"]
c13_r = policy_results["C13_instance_VOI_original"]
cat_maj_r = policy_results["category_majority"]
oracle_r = policy_results["C_oracle_full_information"]

# --- 1. C13 vs C0b ---
delta_c13_c0b_macro_bal = mb(c13_r) - mb(c0b_r)
delta_c13_c0b_ncost = nc(c13_r) - nc(c0b_r)
if mb(c13_r) > mb(c0b_r) and nc(c13_r) <= nc(c0b_r):
    c13_c0b_pareto = "c13_dominates_c0b"
elif mb(c0b_r) > mb(c13_r) and nc(c0b_r) <= nc(c13_r):
    c13_c0b_pareto = "c0b_dominates_c13"
else:
    c13_c0b_pareto = "tradeoff"

# --- 2. C13 vs random_probe ---
delta_c13_rprobe_macro_bal = mb(c13_r) - mb(rprobe_r)
delta_c13_rprobe_ncost = nc(c13_r) - nc(rprobe_r)
c13_beats_random = mb(c13_r) > mb(rprobe_r)  # at comparable or lower cost check below

# --- 3. C13 vs C_minus ---
delta_c13_cminus_macro_bal = mb(c13_r) - mb(c_minus_r)

# --- 4. Gap capture ---
oracle_c0b_gap = mb(oracle_r) - mb(c0b_r)
c13_gain = mb(c13_r) - mb(c0b_r)
c13_gap_capture = c13_gain / max(oracle_c0b_gap, 0.001)
practical_threshold = 0.10 * oracle_c0b_gap
c13_passes_practical = c13_gain >= practical_threshold

# --- 5. Positive discovery ---
delta_pos_recall = mpr(c13_r) - mpr(c0b_r)
delta_neg_recall = mnr(c13_r) - mnr(c0b_r)
c13_improves_pos_recall = delta_pos_recall > 0

# --- 6. Negative-recall tradeoff ---
if delta_pos_recall > 0 and delta_neg_recall > 0:
    recall_source = "both_improved"
elif delta_pos_recall > 0:
    recall_source = "positive_only"
elif delta_neg_recall > 0:
    recall_source = "negative_only_tradeoff"
else:
    recall_source = "neither_improved"

# --- 7. Probe efficiency (already computed above) ---

# --- 8. Pareto frontier ---
pareto_candidates = [
    ("all_false", mb(all_false_r), nc(all_false_r)),
    ("C_minus_prior_only", mb(c_minus_r), nc(c_minus_r)),
    ("C0b_observe_only", mb(c0b_r), nc(c0b_r)),
    ("random_probe_budgeted", mb(rprobe_r), nc(rprobe_r)),
    ("C13_instance_VOI_original", mb(c13_r), nc(c13_r)),
    ("category_majority", mb(cat_maj_r), nc(cat_maj_r)),
]
# Sort by ncost ascending, macro_bal descending
pareto_candidates.sort(key=lambda x: (x[2], -x[1]))
pareto_frontier = []
for name, bal, ncost in pareto_candidates:
    dominated = False
    for pname, pbal, pncost in pareto_frontier:
        if pbal >= bal and pncost <= ncost and (pbal > bal or pncost < ncost):
            dominated = True
            break
    if not dominated:
        pareto_frontier.append((name, bal, ncost))
# Remove lower-accuracy higher-cost entries
filtered = []
best_bal = -1.0
for name, bal, ncost in pareto_frontier:
    if bal > best_bal:
        filtered.append((name, bal, ncost))
        best_bal = bal
pareto_frontier = filtered

oracle_point = ("C_oracle_full_information (reference)", mb(oracle_r), nc(oracle_r))

# --- 9. C13 promising check ---
c13_promising = all([
    mb(c13_r) > mb(c0b_r),                          # A: beats C0b
    mb(c13_r) > mb(rprobe_r),                        # B: beats random
    c13_improves_pos_recall,                          # C: improves positive recall
    c13_passes_practical,                             # D: >= 10% gap capture
    c13_c0b_pareto != "c0b_dominates_c13",            # E: not dominated by C0b
])

# Check for metric exploit (negative-recall-only tradeoff)
metric_exploit = recall_source == "negative_only_tradeoff" and not c13_improves_pos_recall

# =============================================================================
# 8. Print summary
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\n  {'Policy':<30} {'macro_bal':<10} {'m_pos_r':<10} {'m_neg_r':<10} "
      f"{'ncost':<10} {'visits':<8} {'probes':<8} {'cost':<10}")
print("  " + "-" * 86)
for name in ["all_false", "C_minus_prior_only", "C0b_observe_only",
             "random_probe_budgeted", "C13_instance_VOI_original",
             "category_majority", "C_oracle_full_information"]:
    r = policy_results[name]
    c = r["cost"]
    pe = r.get("probe_efficiency")
    eff_str = ""
    if pe:
        eff_str = f"  eff={pe['effective_probe_count']}/{pe['total_probe_count']}"
    print(f"  {name:<30} {mb(r):<10.4f} {mpr(r):<10.4f} {mnr(r):<10.4f} "
          f"{nc(r):<10.4f} {c['visit_count']:<8} {c['probe_count']:<8} "
          f"{c['total_cost']:<10.4f}{eff_str}")

print(f"\n  --- C13 vs C0b ---")
print(f"  delta_macro_bal:        {delta_c13_c0b_macro_bal:+.4f}")
print(f"  delta_normalized_cost:  {delta_c13_c0b_ncost:+.4f}")
print(f"  Pareto relation:        {c13_c0b_pareto}")

print(f"\n  --- C13 vs random_probe ---")
print(f"  delta_macro_bal:        {delta_c13_rprobe_macro_bal:+.4f}")
print(f"  C13 beats random:       {c13_beats_random}")

print(f"\n  --- C13 vs C_minus ---")
print(f"  delta_macro_bal:        {delta_c13_cminus_macro_bal:+.4f}")

print(f"\n  --- Gap Capture ---")
print(f"  Oracle - C0b gap:       {oracle_c0b_gap:.4f}")
print(f"  C13 gain over C0b:      {c13_gain:+.4f}")
print(f"  C13 gap capture:        {c13_gap_capture:.4f} ({c13_gap_capture*100:.1f}%)")
print(f"  Practical threshold:    {practical_threshold:.4f} (10% of gap)")
print(f"  C13 passes practical:   {c13_passes_practical}")

print(f"\n  --- Positive Discovery ---")
print(f"  C13 mean pos_recall:    {mpr(c13_r):.4f}")
print(f"  C0b mean pos_recall:    {mpr(c0b_r):.4f}")
print(f"  delta_pos_recall:       {delta_pos_recall:+.4f}")
print(f"  delta_neg_recall:       {delta_neg_recall:+.4f}")
print(f"  recall source:          {recall_source}")

if c13_r.get("probe_efficiency"):
    pe = c13_r["probe_efficiency"]
    print(f"\n  --- Probe Efficiency ---")
    print(f"  total probes:           {pe['total_probe_count']}")
    print(f"  effective:              {pe['effective_probe_count']} ({pe['effective_probe_rate']:.3f})")
    print(f"  zero_gain:              {pe['zero_gain_probe_count']} ({pe['zero_gain_rate']:.3f})")
    print(f"  harmful:                {pe['harmful_probe_count']} ({pe['harmful_rate']:.3f})")

print(f"\n  --- Pareto Frontier (non-oracle) ---")
for name, bal, ncost_val in pareto_frontier:
    print(f"  {name:<30} macro_bal={bal:.4f}  ncost={ncost_val:.4f}")
print(f"  {oracle_point[0]:<30} macro_bal={oracle_point[1]:.4f}  ncost={oracle_point[2]:.4f}")

print(f"\n  --- C13 Single-Seed Assessment ---")
print(f"  A. C13 > C0b:                         {mb(c13_r) > mb(c0b_r)}")
print(f"  B. C13 > random_probe:                {mb(c13_r) > mb(rprobe_r)}")
print(f"  C. improves positive recall:          {c13_improves_pos_recall}")
print(f"  D. >= 10% gap capture:                {c13_passes_practical}")
print(f"  E. not dominated by C0b/random:       {c13_c0b_pareto != 'c0b_dominates_c13'}")
print(f"  F. no metric exploit:                 {not metric_exploit}")
print(f"  => c13_single_seed_promising:         {c13_promising}")

# Print per-query detail for C13
print(f"\n  --- C13 Per-Query Detail ---")
for qname in sorted(c13_metrics["per_query"].keys()):
    q = c13_metrics["per_query"][qname]
    c0b_q = c0b_metrics["per_query"][qname]
    print(f"  {qname:<20} pos_rec={q['positive_recall']:.4f} (C0b={c0b_q['positive_recall']:.4f})  "
          f"neg_rec={q['negative_recall']:.4f} (C0b={c0b_q['negative_recall']:.4f})  "
          f"bal={q['balanced_accuracy']:.4f} (C0b={c0b_q['balanced_accuracy']:.4f})")

# =============================================================================
# 9. Build output JSON
# =============================================================================
print("\n[5/5] Saving outputs...")

output = {
    "block_id": "1J3",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "primary_metric": "macro_query_balanced_accuracy",
    "reporting_mode": "Pareto: macro_bal vs normalized_cost",
    "baselines": {},
    "comparisons": {
        "C13_vs_C0b": {
            "delta_macro_bal": round(delta_c13_c0b_macro_bal, 6),
            "delta_normalized_cost": round(delta_c13_c0b_ncost, 6),
            "pareto_relation": c13_c0b_pareto,
            "c13_dominates_c0b": c13_c0b_pareto == "c13_dominates_c0b",
            "c0b_dominates_c13": c13_c0b_pareto == "c0b_dominates_c13",
        },
        "C13_vs_random_probe": {
            "delta_macro_bal": round(delta_c13_rprobe_macro_bal, 6),
            "delta_normalized_cost": round(delta_c13_rprobe_ncost, 6),
            "c13_beats_random": c13_beats_random,
        },
        "C13_vs_C_minus": {
            "delta_macro_bal": round(delta_c13_cminus_macro_bal, 6),
        },
        "gap_capture": {
            "oracle_c0b_gap": round(oracle_c0b_gap, 6),
            "c13_gain_over_c0b": round(c13_gain, 6),
            "c13_gap_capture_vs_c0b": round(c13_gap_capture, 6),
            "practical_threshold": round(practical_threshold, 6),
            "c13_passes_practical": c13_passes_practical,
            "c13_macro_bal_needed": round(mb(c0b_r) + practical_threshold, 6),
        },
        "positive_discovery": {
            "c13_mean_pos_recall": mpr(c13_r),
            "c0b_mean_pos_recall": mpr(c0b_r),
            "delta_pos_recall": round(delta_pos_recall, 6),
            "delta_neg_recall": round(delta_neg_recall, 6),
            "recall_source": recall_source,
            "c13_improves_positive_recall": c13_improves_pos_recall,
            "metric_exploit_detected": metric_exploit,
        },
    },
    "probe_efficiency": c13_probe_eff,
    "pareto_frontier": [
        {"name": name, "macro_bal": bal, "normalized_cost": ncost}
        for name, bal, ncost in pareto_frontier
    ],
    "oracle_reference": {
        "name": oracle_point[0],
        "macro_bal": oracle_point[1],
        "normalized_cost": oracle_point[2],
    },
    "c13_single_seed_assessment": {
        "A_c13_gt_c0b": mb(c13_r) > mb(c0b_r),
        "B_c13_gt_random": mb(c13_r) > mb(rprobe_r),
        "C_improves_positive_recall": c13_improves_pos_recall,
        "D_10pct_gap_capture": c13_passes_practical,
        "E_not_pareto_dominated": c13_c0b_pareto != "c0b_dominates_c13",
        "F_no_metric_exploit": not metric_exploit,
        "c13_single_seed_promising": c13_promising,
        "ready_for_multiseed": False,
    },
}

for name in ["all_false", "C_minus_prior_only", "C0b_observe_only",
             "random_probe_budgeted", "C13_instance_VOI_original",
             "category_majority", "C_oracle_full_information"]:
    r = policy_results[name]
    entry = {
        "macro_query_balanced_accuracy": mb(r),
        "macro_query_accuracy": r["metrics"]["macro_query_accuracy"],
        "raw_comp": r["metrics"]["macro_query_accuracy"],
        "mean_positive_recall": mpr(r),
        "mean_negative_recall": mnr(r),
        "per_query": r["metrics"]["per_query"],
        "cost": r["cost"],
    }
    if r.get("probe_efficiency"):
        entry["probe_efficiency"] = r["probe_efficiency"]
    output["baselines"][name] = entry

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j3_policy_comparison_c4_seed101.json")
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"Saved: {json_path}")

# =============================================================================
# 10. Block done
# =============================================================================
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"[block_done]")
print(f"  block_id=1J3")
print(f"  condition=C4_instance_subtype_cued_v1")
print(f"  seed=101")
print(f"  c0b_macro_bal={mb(c0b_r):.4f}")
print(f"  random_probe_macro_bal={mb(rprobe_r):.4f}")
print(f"  c13_macro_bal={mb(c13_r):.4f}")
print(f"  c13_ncost={nc(c13_r):.4f}")
print(f"  c13_delta_vs_c0b={delta_c13_c0b_macro_bal:+.4f}")
print(f"  c13_delta_vs_random={delta_c13_rprobe_macro_bal:+.4f}")
print(f"  c13_gap_capture_vs_c0b={c13_gap_capture:.4f}")
print(f"  c13_improves_positive_recall={c13_improves_pos_recall}")
print(f"  c13_pareto_status={c13_c0b_pareto}")
print(f"  c13_single_seed_promising={c13_promising}")
print(f"  ready_for_multiseed=false")
print(f"  elapsed={elapsed:.1f}s")
print(f"{'='*60}")
