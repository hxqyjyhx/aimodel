"""
Block 1J6 — C15 Two-Pass Observe-Then-Probe Evaluation on C4_instance_subtype_cued_v1.

Implements C15a (observe_all_then_probe) and C15b (probe_reserve).
Evaluates against all baselines: all_false, C_minus, C0b, random_probe, C13, C_oracle.
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
print("Block 1J6 — C15 Two-Pass Observe-Then-Probe Evaluation")
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
# 2. Test objects + environment
# =============================================================================
print("\n[2/5] Generating subtype test objects...")
test_objects = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
query_gt = compute_query_ground_truth(test_objects)
test_oids = sorted(test_objects.keys())
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
print(f"  test_objects={len(test_objects)}  positions assigned")

# Ground truth simulator for oracle and diagnostics
sim_gt = MiniMCSimulatorTruth(test_objects, positions, config.AGENT_START)

# Object metadata
CATEGORY_DIFFERENTIATING_ACTION = {
    "wood_log": "craft_plank",
    "stone_block": "mine_by_hand",
    "apple": "eat",
    "wooden_pickaxe": "use_as_tool",
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
# 3. C15 Two-Pass Policy Definition
# =============================================================================
class C15_TwoPassObserveThenProbePolicy(MiniMCPolicy):
    """Two-pass: observe broad first, then probe selectively.

    Variants:
      C15a — observe as many objects as possible, then probe with remaining budget
      C15b — reserve probe budget, observe until reserve threshold, then probe
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, cost_weight=0.5, variant="C15a"):
        self._im = instance_memory
        self._rng = rng
        self._cost_weight = cost_weight
        self._variant = variant
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._visit_order = []
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._observe_pass_cost = 0.0
        self._remaining_budget_after_observe = 0.0

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

    def _normalized_probe_cost(self, view):
        return view.probe_cost / max(view.initial_budget, 0.001)

    def _should_transition_to_probe_phase(self, view):
        """Check whether to end the observe pass and begin probing."""
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True

        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost
            for oid in unvisited
        )
        if not view.can_afford(cheapest_observe):
            return True

        # Estimate minimum cost to do one probe cycle:
        # re-reach to an already-observed object + observe (harness re-does it) + probe
        # After observing many objects, agent is far from center; estimate reach ~0.07
        min_probe_cycle = 0.07 + view.observe_cost + view.probe_cost

        if self._variant == "C15a":
            # Transition BEFORE last object to avoid is_terminal() killing the episode
            # Leave at least 1 object unvisited so harness doesn't auto-terminate
            if len(unvisited) <= 5:
                return True
            if not view.can_afford(cheapest_observe + min_probe_cycle):
                return True
            # Also transition if remaining budget is low enough that probing
            # needs to happen now or never
            reserve = 0.30 * view.initial_budget
            if view.budget_remaining - cheapest_observe < reserve:
                return True
            return False
        else:  # C15b
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
        """Select best already-observed, not-yet-probed object by post-observation VOI."""
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

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =============================================================================
# 4. RandomProbeTrackedPolicy (same as 1J4)
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
print("\n[3/5] Running all policies...")

all_results = {}

# --- all_false ---
print("  all_false (all predictions = False)...")
all_false_preds = {}
for oid in test_oids:
    all_false_preds[oid] = {f: 0.0 for f in CORE_ACTION_FEATURES}
all_false_metrics = compute_full_metrics(all_false_preds, query_gt)
all_false_cost = {
    "visit_count": 0, "observe_count": 0, "probe_count": 0,
    "total_cost": 0.0, "normalized_cost": 0.0,
    "total_reach_cost": 0.0, "total_observe_cost": 0.0, "total_probe_cost": 0.0,
    "budget_remaining": BUDGET, "budget_spent": 0.0, "initial_budget": BUDGET,
}
all_results["all_false"] = {"metrics": all_false_metrics, "cost": all_false_cost}

# --- C_minus_prior_only ---
print("  C_minus_prior_only...")
c_minus_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c_minus_policy = C0a_NoInteractionPolicy(im_base.clone(), random.Random(SEED))
c_minus_preds, c_minus_log, c_minus_obs, _, _ = run_policy(c_minus_policy, c_minus_env)
c_minus_metrics = compute_full_metrics(c_minus_preds["per_object"], query_gt)
c_minus_cost = get_cost_metrics(c_minus_log, c_minus_obs)
all_results["C_minus_prior_only"] = {"metrics": c_minus_metrics, "cost": c_minus_cost}
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
all_results["random_probe_budgeted"] = {"metrics": rprobe_metrics, "cost": rprobe_cost}
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

# --- C15a_observe_all_then_probe ---
print("  C15a_observe_all_then_probe...")
c15a_im = im_base.clone()
c15a_policy = C15_TwoPassObserveThenProbePolicy(
    c15a_im, random.Random(SEED + 600), cost_weight=COST_WEIGHT, variant="C15a")
c15a_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15a_preds, c15a_log, c15a_obs, c15a_pre_probe, c15a_pre_dec = run_policy(c15a_policy, c15a_env)
c15a_metrics = compute_full_metrics(c15a_preds["per_object"], query_gt)
c15a_cost = get_cost_metrics(c15a_log, c15a_obs)
c15a_extra = {
    "observe_pass_visit_count": c15a_policy._observe_pass_visit_count,
    "observe_pass_cost": c15a_policy._observe_pass_cost,
    "remaining_budget_after_observe_pass": c15a_policy._remaining_budget_after_observe,
    "probe_pass_candidate_count": sum(
        1 for oid in test_oids
        if c15a_obs.is_visited(oid) and oid not in c15a_policy._probed_oids
    ) + len(c15a_policy._probed_oids),
    "probe_pass_probe_count": len(c15a_policy._probed_oids),
    "visit_order": c15a_policy._visit_order,
    "phase_transition_reason": (
        "all_observed" if c15a_policy._observe_pass_visit_count >= len(test_oids)
        else "budget_conservation"
    ),
}
all_results["C15a_observe_all_then_probe"] = {
    "metrics": c15a_metrics, "cost": c15a_cost, "c15_extra": c15a_extra}
print(f"    macro_bal={c15a_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c15a_cost['visit_count']}  probed={len(c15a_policy._probed_oids)}  "
      f"observe_pass_visits={c15a_policy._observe_pass_visit_count}  "
      f"ncost={c15a_cost['normalized_cost']:.4f}")

# --- C15b_probe_reserve ---
print("  C15b_probe_reserve...")
c15b_im = im_base.clone()
c15b_policy = C15_TwoPassObserveThenProbePolicy(
    c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT, variant="C15b")
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, c15b_pre_probe, c15b_pre_dec = run_policy(c15b_policy, c15b_env)
c15b_metrics = compute_full_metrics(c15b_preds["per_object"], query_gt)
c15b_cost = get_cost_metrics(c15b_log, c15b_obs)
c15b_extra = {
    "observe_pass_visit_count": c15b_policy._observe_pass_visit_count,
    "observe_pass_cost": c15b_policy._observe_pass_cost,
    "remaining_budget_after_observe_pass": c15b_policy._remaining_budget_after_observe,
    "probe_pass_candidate_count": sum(
        1 for oid in test_oids
        if c15b_obs.is_visited(oid) and oid not in c15b_policy._probed_oids
    ) + len(c15b_policy._probed_oids),
    "probe_pass_probe_count": len(c15b_policy._probed_oids),
    "visit_order": c15b_policy._visit_order,
    "phase_transition_reason": (
        "all_observed" if c15b_policy._observe_pass_visit_count >= len(test_oids)
        else "budget_reserve_reached"
    ),
}
all_results["C15b_probe_reserve"] = {
    "metrics": c15b_metrics, "cost": c15b_cost, "c15_extra": c15b_extra}
print(f"    macro_bal={c15b_metrics['macro_query_balanced_accuracy']:.4f}  "
      f"visited={c15b_cost['visit_count']}  probed={len(c15b_policy._probed_oids)}  "
      f"observe_pass_visits={c15b_policy._observe_pass_visit_count}  "
      f"ncost={c15b_cost['normalized_cost']:.4f}")

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
    subtypes = subtype_def["subtypes"]
    ratios = subtype_def["subtype_ratio"]
    # Find majority subtype per category, use its affordance profile
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
# 7. Compute C15-specific probe diagnostics
# =============================================================================
print("[4/5] Computing C15 probe diagnostics...")


def classify_c15_probes(policy_obj, agent_obs):
    """Classify each C15 probe by efficiency and differentiating action."""
    details = []
    visited_oids = {oid for oid in test_oids if agent_obs.is_visited(oid)}
    probed_oids = {oid for oid in test_oids if agent_obs.get_probe_results(oid)}

    for probe in policy_obj._probe_tracking:
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

    # visited positive/negative coverage per query
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
        "per_probe": details,
        "visited_positive_coverage_per_query": visited_coverage,
    }


c15a_probe_diag = classify_c15_probes(c15a_policy, c15a_obs)
c15b_probe_diag = classify_c15_probes(c15b_policy, c15b_obs)

# =============================================================================
# 8. Build comparisons and output
# =============================================================================
print("[5/5] Building output...")

oracle_macro_bal = oracle_metrics["macro_query_balanced_accuracy"]
oracle_c0b_gap = oracle_macro_bal - c0b_metrics["macro_query_balanced_accuracy"]


def make_comparison(policy_label, metrics, cost, c15_extra=None, probe_diag=None):
    """Build comprehensive comparison record for a policy."""
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
        entry["vs_C0b"] = {
            "delta_macro_bal": round(metrics["macro_query_balanced_accuracy"] - c0bm["macro_query_balanced_accuracy"], 6),
            "delta_mean_positive_recall": round(metrics["mean_positive_recall"] - c0bm["mean_positive_recall"], 6),
            "delta_mean_negative_recall": round(metrics["mean_negative_recall"] - c0bm["mean_negative_recall"], 6),
            "delta_normalized_cost": round(cost["normalized_cost"] - all_results["C0b_observe_only"]["cost"]["normalized_cost"], 6) if cost else None,
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

    # Gap capture vs C0b-to-oracle
    if policy_label not in ("C0b_observe_only",):
        gap = metrics["macro_query_balanced_accuracy"] - c0b_metrics["macro_query_balanced_accuracy"]
        entry["gap_capture_vs_c0b"] = round(gap / max(oracle_c0b_gap, 0.001), 6)

    # C15 extra
    if c15_extra is not None:
        entry["c15_specific"] = c15_extra
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
    c15_extra = r.get("c15_extra")
    probe_diag = None
    policies_output[label] = make_comparison(label, r["metrics"], r["cost"],
                                             c15_extra=c15_extra, probe_diag=probe_diag)

# C15a
policies_output["C15a_observe_all_then_probe"] = make_comparison(
    "C15a_observe_all_then_probe", c15a_metrics, c15a_cost,
    c15_extra=c15a_extra, probe_diag=c15a_probe_diag)

# C15b
policies_output["C15b_probe_reserve"] = make_comparison(
    "C15b_probe_reserve", c15b_metrics, c15b_cost,
    c15_extra=c15b_extra, probe_diag=c15b_probe_diag)

# =============================================================================
# 9. Compute Pareto frontier
# =============================================================================
normal_policies = [
    "all_false", "C_minus_prior_only", "C0b_observe_only",
    "random_probe_budgeted", "C13_instance_VOI_original",
    "C15a_observe_all_then_probe", "C15b_probe_reserve",
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

# Determine Pareto frontier
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

# Check C15a vs C0b Pareto relation
c15a_dominated_by_c0b = (
    c0b_metrics["macro_query_balanced_accuracy"] >= c15a_metrics["macro_query_balanced_accuracy"]
    and c0b_cost["normalized_cost"] <= c15a_cost["normalized_cost"]
    and (c0b_metrics["macro_query_balanced_accuracy"] > c15a_metrics["macro_query_balanced_accuracy"]
         or c0b_cost["normalized_cost"] < c15a_cost["normalized_cost"])
)

# =============================================================================
# 10. Determine if C15 is promising on single seed
# =============================================================================
c15a_mb = c15a_metrics["macro_query_balanced_accuracy"]
c15a_pos_r = c15a_metrics["mean_positive_recall"]
c15a_neg_r = c15a_metrics["mean_negative_recall"]
c0b_mb = c0b_metrics["macro_query_balanced_accuracy"]
c0b_pos_r = c0b_metrics["mean_positive_recall"]
c0b_neg_r = c0b_metrics["mean_negative_recall"]
c13_mb = c13_metrics["macro_query_balanced_accuracy"]
rprobe_mb = rprobe_metrics["macro_query_balanced_accuracy"]

c15a_gap_capture = (c15a_mb - c0b_mb) / max(oracle_c0b_gap, 0.001)

neg_only_tradeoff = (
    c15a_pos_r < c0b_pos_r
    and c15a_neg_r > c0b_neg_r
)

c15a_promising = (
    c15a_mb > c0b_mb
    and c15a_mb > c13_mb
    and c15a_mb > rprobe_mb
    and c15a_pos_r > c0b_pos_r
    and c15a_gap_capture >= 0.10
    and not c15a_dominated_by_c0b
    and not neg_only_tradeoff
)

# Determine if Route B or Route F recommended
route_b_needed = (
    not c15a_promising
    or c15a_gap_capture < 0.10
    or c15a_pos_r <= c0b_pos_r
)

# Route F: blurred observation still unknown until A+B exhausted
route_f_needed = True  # deferred per 1J5_patch

# =============================================================================
# 11. Build output JSON
# =============================================================================
output = {
    "block_id": "1J6",
    "condition": COND["label"],
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "C15_two_pass_observe_then_probe evaluation",
    "policies": policies_output,
    "oracle": {
        "policy": "C_oracle_full_information",
        "macro_query_balanced_accuracy": oracle_macro_bal,
        "per_query": oracle_metrics["per_query"],
    },
    "diagnostic_category_majority": {
        "policy": "diagnostic_category_majority_upper_bound",
        "macro_query_balanced_accuracy": cat_maj_metrics["macro_query_balanced_accuracy"],
        "note": "Leakage-risk diagnostic only. Uses hidden_category. NOT a deployable policy.",
    },
    "oracle_c0b_gap": round(oracle_c0b_gap, 6),
    "pareto_frontier": {
        "normal_policies_only": True,
        "policies_on_frontier": pareto_frontier,
        "all_pareto_points": pareto_points,
        "category_majority_excluded_as_diagnostic": True,
    },
    "c15a_assessment": {
        "c15a_macro_bal": c15a_mb,
        "c15a_ncost": c15a_cost["normalized_cost"],
        "c15a_visit_count": c15a_cost["visit_count"],
        "c15a_probe_count": len(c15a_policy._probed_oids),
        "c15a_delta_vs_c0b": round(c15a_mb - c0b_mb, 6),
        "c15a_delta_vs_c13": round(c15a_mb - c13_mb, 6),
        "c15a_gap_capture_vs_c0b": round(c15a_gap_capture, 6),
        "c15a_improves_positive_recall": c15a_pos_r > c0b_pos_r,
        "c15a_neg_only_tradeoff": neg_only_tradeoff,
        "c15a_pareto_dominated_by_c0b": c15a_dominated_by_c0b,
        "c15a_single_seed_promising": c15a_promising,
        "conditions_for_promising": {
            "A_beats_C0b": c15a_mb > c0b_mb,
            "B_beats_C13": c15a_mb > c13_mb,
            "C_beats_random": c15a_mb > rprobe_mb,
            "D_improves_pos_recall": c15a_pos_r > c0b_pos_r,
            "E_gap_capture_ge_10pct": c15a_gap_capture >= 0.10,
            "F_not_pareto_dominated_by_C0b": not c15a_dominated_by_c0b,
            "G_no_neg_only_tradeoff": not neg_only_tradeoff,
        },
    },
    "c15b_assessment": {
        "c15b_macro_bal": c15b_metrics["macro_query_balanced_accuracy"],
        "c15b_ncost": c15b_cost["normalized_cost"],
        "c15b_visit_count": c15b_cost["visit_count"],
        "c15b_probe_count": len(c15b_policy._probed_oids),
        "c15b_delta_vs_c0b": round(c15b_metrics["macro_query_balanced_accuracy"] - c0b_mb, 6),
        "c15b_delta_vs_c13": round(c15b_metrics["macro_query_balanced_accuracy"] - c13_mb, 6),
    },
    "route_recommendation": {
        "route_b_differentiating_action_needed_next": route_b_needed,
        "route_f_blurred_observation_needed_later": route_f_needed,
        "ready_for_multiseed": False,
        "block_1J6_summary": (
            "C15 breadth-preserving two-pass design evaluated. "
            "Primary question: does separating observation from probing improve over C13?"
        ),
    },
}

# Save JSON
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j6_c15_two_pass_c4_seed101.json")
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
print(f"block_id=1J6")
print(f"condition=C4_instance_subtype_cued_v1")
print(f"seed=101")
print(f"c0b_macro_bal={c0b_mb:.4f}")
print(f"c13_macro_bal={c13_mb:.4f}")
print(f"c15a_macro_bal={c15a_mb:.4f}")
print(f"c15a_ncost={c15a_cost['normalized_cost']:.4f}")
print(f"c15a_visit_count={c15a_cost['visit_count']}")
print(f"c15a_probe_count={len(c15a_policy._probed_oids)}")
print(f"c15a_delta_vs_c0b={c15a_mb - c0b_mb:+.4f}")
print(f"c15a_delta_vs_c13={c15a_mb - c13_mb:+.4f}")
print(f"c15a_gap_capture_vs_c0b={c15a_gap_capture:.4f}")
print(f"c15a_improves_positive_recall={c15a_pos_r > c0b_pos_r}")
print(f"c15a_pareto_status={'dominated_by_C0b' if c15a_dominated_by_c0b else 'not_dominated_by_C0b'}")
print(f"c15a_single_seed_promising={c15a_promising}")
print(f"c15b_macro_bal={c15b_metrics['macro_query_balanced_accuracy']:.4f}")
print(f"c15b_visit_count={c15b_cost['visit_count']}")
print(f"c15b_probe_count={len(c15b_policy._probed_oids)}")
print(f"route_b_needed_next={route_b_needed}")
print(f"route_f_needed_later={route_f_needed}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'='*60}")
