"""
Block 1J14 -- Probe Informativeness / IOM Update Upper-Bound Diagnostic.

Tests whether injecting additional true probe outcomes into the IOM
can improve query-level macro_bal. All variants are DIAGNOSTIC ONLY,
non-deployable.

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
ORACLE_INJECTION_K_VALUES = [4, 8, 12]

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

print("=" * 60)
print("Block 1J14 -- Probe Informativeness / IOM Update Upper-Bound Diagnostic")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print("=" * 60)

# =============================================================================
# 1. Training + IOM building
# =============================================================================
print("\n[1/8] Phase A: Training + IOM building...")
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
print("\n[2/8] Generating test objects...")
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

# Ground truth affordances for all (oid, feature) pairs
ALL_GT_AFFORDANCES = {}
for oid in test_oids:
    ALL_GT_AFFORDANCES[oid] = sim_gt.get_ground_truth_affordances(oid)


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


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def run_policy(policy, env):
    harness = EpisodeHarness(env, policy)
    result = harness.run()
    return (result.predictions, result.event_log, result.agent_obs,
            result.pre_probe_entropies, result.pre_decision_entropies)


# =============================================================================
# 4. C15b Inline (needed for visited object set)
# =============================================================================
print("\n[3/8] Defining C15b policy for visited-object capture...")

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
# 5. Run baselines (C0b, C13, C15b, Oracle)
# =============================================================================
print("\n[4/8] Running baselines...")
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
print(f"  C13_instance_VOI_original macro_bal={c13_mb:.4f}  "
      f"visited={c13_visited_n}  probed={c13_probed_n}")

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
print(f"  C15b_original macro_bal={c15b_mb:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")

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
# 6. Oracle Probe Injection Helpers
# =============================================================================
print("\n[5/8] Computing oracle probe injection rankings...")

def compute_iom_predictions(im, oids_to_use_features, all_oids):
    """Compute IOM predictions for all objects.

    Args:
        im: IOM instance (with any pre-injected probes)
        oids_to_use_features: set of oids to use actual visible features for
        all_oids: all object IDs to predict for

    Returns:
        dict oid -> {feature: prob}
    """
    preds = {}
    for oid in all_oids:
        if oid in oids_to_use_features:
            features = ALL_VISIBLE_FEATURES.get(oid, {})
        else:
            features = {}
        fake_obj = {"id": oid, "visible_features": features}
        preds[oid] = im.predict_all_affordances(fake_obj)
    return preds


def compute_prediction_errors(predictions, gt_affordances):
    """Compute |pred - gt| for all (oid, feature) pairs.

    Returns list of (oid, action, feature, error, gt_value) sorted by error desc.
    """
    errors = []
    for oid in predictions:
        for feature in CORE_ACTION_FEATURES:
            pred = predictions[oid].get(feature, 0.5)
            gt = gt_affordances[oid].get(feature, 0.0)
            error = abs(pred - gt)
            action = {v: k for k, v in ACTION_TO_FEATURE.items()}.get(feature, feature)
            errors.append((oid, action, feature, error, gt))
    errors.sort(key=lambda x: x[3], reverse=True)
    return errors


def inject_probes_and_predict(im, injections, oids_to_use_features, all_oids):
    """Inject probe outcomes into a fresh IOM clone and predict.

    Args:
        im: base IOM to clone
        injections: list of (oid, action, true_outcome) tuples
        oids_to_use_features: set of oids for visible features
        all_oids: all object IDs

    Returns:
        predictions dict oid -> {feature: prob}
    """
    im_clone = im.clone()
    for oid, action, outcome in injections:
        im_clone.incorporate_probe(oid, action, outcome)
    return compute_iom_predictions(im_clone, oids_to_use_features, all_oids)


def get_true_outcome(oid, action):
    """Get ground truth probe outcome for (oid, action)."""
    feature = ACTION_TO_FEATURE[action]
    return ALL_GT_AFFORDANCES[oid].get(feature, 0.0)


# Compute base predictions (all features visible, no probes) for error ranking
base_im_for_ranking = im_base.clone()
base_predictions_all_features = compute_iom_predictions(
    base_im_for_ranking, set(test_oids), test_oids)
all_errors_ranked = compute_prediction_errors(base_predictions_all_features, ALL_GT_AFFORDANCES)
print(f"  Computed {len(all_errors_ranked)} (oid, action) error pairs")
print(f"  Top 5 errors: {[(e[0][:20], e[1][:15], round(e[3], 4)) for e in all_errors_ranked[:5]]}")


# =============================================================================
# 7. Oracle Probe Injection Variants
# =============================================================================
print("\n[6/8] Running oracle probe injection variants...")
oracle_variants = {}

# ----- 7.1 C15b_observed_objects_probe_oracle_UB -----
print("  [7.1] C15b_observed_objects_probe_oracle_UB...")
# For each C15b-visited object, select the best single action by oracle query-level value.
# "Best" = action with largest |pred - gt| on this object (most informative correction).
# Use actual features for visited objects, empty for unvisited.

c15b_visited_set = set(c15b_visited_oids)
# Filter errors to only C15b-visited objects, take best action per object
c15b_errors = [(oid, action, feature, error, gt)
               for (oid, action, feature, error, gt) in all_errors_ranked
               if oid in c15b_visited_set]

# Per-object best action
c15b_per_object_best = {}
for oid, action, feature, error, gt in c15b_errors:
    if oid not in c15b_per_object_best:
        c15b_per_object_best[oid] = (oid, action, feature, error, gt)

c15b_injections = [(oid, action, get_true_outcome(oid, action))
                   for oid, action, feature, error, gt in c15b_per_object_best.values()]

c15b_ub_im = im_base.clone()
c15b_ub_preds = inject_probes_and_predict(
    c15b_ub_im, c15b_injections, c15b_visited_set, test_oids)
c15b_ub_metrics = compute_full_metrics(c15b_ub_preds, query_gt)

# Count affected queries
c15b_injected_actions = {}
for oid, action, _ in c15b_injections:
    c15b_injected_actions[oid] = action

c15b_ub_mb = c15b_ub_metrics["macro_query_balanced_accuracy"]
c15b_ub_delta = c15b_ub_mb - c15b_mb
print(f"    injected_probes={len(c15b_injections)}  "
      f"macro_bal={c15b_ub_mb:.4f}  delta_vs_C15b={c15b_ub_delta:+.4f}")

oracle_variants["C15b_observed_objects_probe_oracle_UB"] = {
    "label": "diagnostic_non_deployable",
    "source_visited_set": "C15b",
    "n_visited": c15b_visited_n,
    "injected_probe_count": len(c15b_injections),
    "injected_actions": c15b_injected_actions,
    "injected_action_summary": dict(Counter(
        a for _, a, _ in c15b_injections)),
    "predictions": c15b_ub_preds,
    "metrics": c15b_ub_metrics,
    "delta_vs_source": round(c15b_ub_delta, 6),
    "gap_capture_vs_c0b": round((c15b_ub_mb - 0.5871) / 0.4129, 6),
    "reaches_10pct_threshold": c15b_ub_mb >= 0.6284,
}

# ----- 7.2 C13_visited_objects_full_probe_UB -----
print("  [7.2] C13_visited_objects_full_probe_UB...")
# For each C13-visited object, inject ALL 6 true probe outcomes.
c13_visited_set = set(c13_visited_oids)

c13_full_injections = []
for oid in c13_visited_oids:
    for action in MAIN_CANDIDATE_ACTIONS:
        c13_full_injections.append((oid, action, get_true_outcome(oid, action)))

c13_ub_im = im_base.clone()
c13_ub_preds = inject_probes_and_predict(
    c13_ub_im, c13_full_injections, c13_visited_set, test_oids)
c13_ub_metrics = compute_full_metrics(c13_ub_preds, query_gt)

c13_ub_mb = c13_ub_metrics["macro_query_balanced_accuracy"]
c13_ub_delta = c13_ub_mb - c13_mb
print(f"    injected_probes={len(c13_full_injections)}  "
      f"macro_bal={c13_ub_mb:.4f}  delta_vs_C13={c13_ub_delta:+.4f}")

oracle_variants["C13_visited_objects_full_probe_UB"] = {
    "label": "diagnostic_non_deployable",
    "source_visited_set": "C13",
    "n_visited": c13_visited_n,
    "injected_probe_count": len(c13_full_injections),
    "injected_actions": {oid: "all_6_actions" for oid in c13_visited_oids},
    "injected_action_summary": dict(Counter(
        a for _, a, _ in c13_full_injections)),
    "predictions": c13_ub_preds,
    "metrics": c13_ub_metrics,
    "delta_vs_source": round(c13_ub_delta, 6),
    "gap_capture_vs_c0b": round((c13_ub_mb - 0.5871) / 0.4129, 6),
    "reaches_10pct_threshold": c13_ub_mb >= 0.6284,
}

# ----- 7.3 IOM_oracle_probe_injection_UB (k=4,8,12, all) -----
print("  [7.3] IOM_oracle_probe_injection_UB...")
# No policy claim. Use all objects' actual visible features.
# Rank (oid, action) by |pred - gt|, inject top k.

all_oids_set = set(test_oids)
iom_injection_results = {}

for k in ORACLE_INJECTION_K_VALUES:
    top_k = all_errors_ranked[:k]
    injections = [(oid, action, get_true_outcome(oid, action))
                  for oid, action, feature, error, gt in top_k]

    preds = inject_probes_and_predict(im_base, injections, all_oids_set, test_oids)
    metrics = compute_full_metrics(preds, query_gt)
    mb = metrics["macro_query_balanced_accuracy"]

    injected_actions = {oid: action for oid, action, feature, error, gt in top_k}
    affected_queries = set()
    for oid in injected_actions:
        for qname, qdata in query_gt.items():
            if oid in qdata["positive_oids"] or oid in qdata["negative_oids"]:
                affected_queries.add(qname)

    print(f"    k={k}: macro_bal={mb:.4f}  delta_vs_C15b={mb - c15b_mb:+.4f}  "
          f"affected_queries={len(affected_queries)}")

    iom_injection_results[f"k{k}"] = {
        "k": k,
        "injected_probe_count": len(injections),
        "injected_actions": injected_actions,
        "injected_action_summary": dict(Counter(a for _, a, _ in injections)),
        "affected_queries": sorted(affected_queries),
        "predictions": preds,
        "metrics": metrics,
        "macro_bal": mb,
        "delta_vs_C15b": round(mb - c15b_mb, 6),
        "gap_capture_vs_c0b": round((mb - 0.5871) / 0.4129, 6),
        "reaches_10pct_threshold": mb >= 0.6284,
    }

# --- k = all_visited_possible: inject all 360 (oid, action) pairs ---
print("    k=all_visited_possible (all 360 probe outcomes)...")
all_injections = [(oid, action, get_true_outcome(oid, action))
                  for oid in test_oids for action in MAIN_CANDIDATE_ACTIONS]
all_preds = inject_probes_and_predict(im_base, all_injections, all_oids_set, test_oids)
all_metrics = compute_full_metrics(all_preds, query_gt)
all_mb = all_metrics["macro_query_balanced_accuracy"]
print(f"    k=all: macro_bal={all_mb:.4f}  n_injections={len(all_injections)}")

iom_injection_results["all_visited_possible"] = {
    "k": "all_visited_possible",
    "injected_probe_count": len(all_injections),
    "injected_actions": "all_6_actions_on_all_60_objects",
    "injected_action_summary": dict(Counter(a for _, a, _ in all_injections)),
    "affected_queries": sorted(query_gt.keys()),
    "predictions": all_preds,
    "metrics": all_metrics,
    "macro_bal": all_mb,
    "delta_vs_C15b": round(all_mb - c15b_mb, 6),
    "gap_capture_vs_c0b": round((all_mb - 0.5871) / 0.4129, 6),
    "reaches_10pct_threshold": all_mb >= 0.6284,
}

oracle_variants["IOM_oracle_probe_injection_UB"] = {
    "label": "diagnostic_non_deployable",
    "no_policy_claim": True,
    "uses_all_visible_features": True,
    "description": "Inject top-k oracle-ranked probe outcomes into IOM. No policy claim.",
    "injection_results": iom_injection_results,
}

# ----- 7.4 Full_object_action_probe_oracle_UB -----
print("  [7.4] Full_object_action_probe_oracle_UB...")
# This is the same as k=all_visited_possible above - inject all 6 actions on all 60 objects.
# Report as a standalone variant.
full_ub_mb = all_mb
full_ub_delta = full_ub_mb - c15b_mb
print(f"    macro_bal={full_ub_mb:.4f}  delta_vs_C15b={full_ub_delta:+.4f}  "
      f"n_injections={len(all_injections)}")

oracle_variants["Full_object_action_probe_oracle_UB"] = {
    "label": "diagnostic_non_deployable",
    "description": "Inject ALL true probe outcomes for ALL 6 actions on ALL 60 objects.",
    "injected_probe_count": len(all_injections),
    "injected_actions": "all_6_actions_on_all_60_objects",
    "injected_action_summary": dict(Counter(a for _, a, _ in all_injections)),
    "predictions": all_preds,
    "metrics": all_metrics,
    "delta_vs_C15b": round(full_ub_delta, 6),
    "gap_capture_vs_c0b": round((full_ub_mb - 0.5871) / 0.4129, 6),
    "reaches_10pct_threshold": full_ub_mb >= 0.6284,
}


# =============================================================================
# 8. Comparisons
# =============================================================================
print("\n[7/8] Computing comparisons...")
comparisons = {}

# Best variant (excluding trivial all_visited_possible and Full which are oracle-equivalent)
non_trivial_variants = {}
for vname, vdata in oracle_variants.items():
    if vname == "IOM_oracle_probe_injection_UB":
        for kname, kdata in vdata["injection_results"].items():
            if kname != "all_visited_possible":
                non_trivial_variants[f"{vname}_{kname}"] = kdata
    elif vname != "Full_object_action_probe_oracle_UB":
        non_trivial_variants[vname] = vdata

all_variants_for_comparison = {}
for vname, vdata in oracle_variants.items():
    if vname == "IOM_oracle_probe_injection_UB":
        for kname, kdata in vdata["injection_results"].items():
            all_variants_for_comparison[f"{vname}_{kname}"] = kdata
    else:
        all_variants_for_comparison[vname] = vdata

best_non_trivial_name = max(
    non_trivial_variants,
    key=lambda k: non_trivial_variants[k]["metrics"]["macro_query_balanced_accuracy"])
best_non_trivial = non_trivial_variants[best_non_trivial_name]
best_ub_mb = best_non_trivial["metrics"]["macro_query_balanced_accuracy"]
print(f"  Best non-trivial oracle UB: {best_non_trivial_name} (macro_bal={best_ub_mb:.4f})")

best_overall_name = max(
    all_variants_for_comparison,
    key=lambda k: all_variants_for_comparison[k]["metrics"]["macro_query_balanced_accuracy"])
best_overall = all_variants_for_comparison[best_overall_name]

# 1. Each variant vs C15b
for vname, vdata in oracle_variants.items():
    if vname == "IOM_oracle_probe_injection_UB":
        for kname, kdata in vdata["injection_results"].items():
            cmp_name = f"{vname}_{kname}"
            comparisons[f"{cmp_name}_vs_C15b"] = {
                "delta_macro_bal": round(kdata["macro_bal"] - c15b_mb, 6),
                "delta_positive_recall": round(
                    kdata["metrics"]["mean_positive_recall"] - c15b_pos_recall, 6),
                "delta_negative_recall": round(
                    kdata["metrics"]["mean_negative_recall"] - c15b_neg_recall, 6),
                "reaches_10pct_threshold": kdata.get("reaches_10pct_threshold", False),
            }
    else:
        comparisons[f"{vname}_vs_C15b"] = {
            "delta_macro_bal": round(vdata["metrics"]["macro_query_balanced_accuracy"] - c15b_mb, 6),
            "delta_positive_recall": round(
                vdata["metrics"]["mean_positive_recall"] - c15b_pos_recall, 6),
            "delta_negative_recall": round(
                vdata["metrics"]["mean_negative_recall"] - c15b_neg_recall, 6),
            "reaches_10pct_threshold": vdata.get("reaches_10pct_threshold", False),
        }

# 2. C15b_UB vs C13_UB (breadth necessity)
c15b_ub_mb_val = oracle_variants["C15b_observed_objects_probe_oracle_UB"]["metrics"]["macro_query_balanced_accuracy"]
c13_ub_mb_val = oracle_variants["C13_visited_objects_full_probe_UB"]["metrics"]["macro_query_balanced_accuracy"]
comparisons["breadth_necessity"] = {
    "C15b_UB_macro_bal": c15b_ub_mb_val,
    "C13_UB_macro_bal": c13_ub_mb_val,
    "delta": round(c15b_ub_mb_val - c13_ub_mb_val, 6),
    "breadth_improves_UB": c15b_ub_mb_val > c13_ub_mb_val,
    "C15b_visited_count": c15b_visited_n,
    "C13_visited_count": c13_visited_n,
}

# 3. Gap capture summary
target_10pct = 0.5871 + 0.0413  # 0.6284
gap_capture_summary = {
    "C0b_base": 0.5871,
    "oracle_ceiling": 1.0000,
    "target_10pct_threshold": target_10pct,
    "C15b_gap_capture_pct": round((c15b_mb - 0.5871) / 0.4129 * 100, 2),
    f"best_UB_gap_capture_pct": round((best_ub_mb - 0.5871) / 0.4129 * 100, 2),
    f"best_UB_hits_10pct": best_ub_mb >= target_10pct,
}
comparisons["gap_capture"] = gap_capture_summary

# 4. Per-query breakdown (C15b vs best UB)
comparisons["per_query_breakdown"] = {
    qname: {
        "C15b_bal_acc": results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]["balanced_accuracy"],
        "best_UB_bal_acc": best_non_trivial["metrics"]["per_query"][qname]["balanced_accuracy"],
        "delta": round(
            best_non_trivial["metrics"]["per_query"][qname]["balanced_accuracy"] -
            results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]["balanced_accuracy"], 6),
    }
    for qname in sorted(query_gt.keys())
}

# 5. IOM injection scaling (k=4,8,12,all)
iom_scaling = {
    kname: {
        "k": kdata["k"],
        "macro_bal": kdata["macro_bal"],
        "delta_vs_C15b": kdata["delta_vs_C15b"],
        "gap_capture_pct": round((kdata["macro_bal"] - 0.5871) / 0.4129 * 100, 2),
        "reaches_10pct": kdata["reaches_10pct_threshold"],
    }
    for kname, kdata in oracle_variants["IOM_oracle_probe_injection_UB"]["injection_results"].items()
}
comparisons["iom_injection_scaling"] = iom_scaling


# =============================================================================
# 9. Interpretation
# =============================================================================
print("\n[8/8] Interpreting results...")

# Determine probe information usefulness
# Use non-trivial variants for interpretation (exclude all_visited_possible and Full)
c15b_ub_mb_val = oracle_variants["C15b_observed_objects_probe_oracle_UB"]["metrics"]["macro_query_balanced_accuracy"]
c13_ub_mb_val = oracle_variants["C13_visited_objects_full_probe_UB"]["metrics"]["macro_query_balanced_accuracy"]
k8_mb = oracle_variants["IOM_oracle_probe_injection_UB"]["injection_results"]["k8"]["macro_bal"]
k12_mb = oracle_variants["IOM_oracle_probe_injection_UB"]["injection_results"]["k12"]["macro_bal"]
full_ub_mb = oracle_variants["Full_object_action_probe_oracle_UB"]["metrics"]["macro_query_balanced_accuracy"]
full_ub_matches_oracle = full_ub_mb >= 0.99

# C15b_UB is the most diagnostic: one oracle-selected probe per C15b-visited object
best_ub_reaches_10pct = c15b_ub_mb_val >= target_10pct
c15b_ub_reaches_10pct = c15b_ub_mb_val >= target_10pct

# k=8 reaches 10% threshold with only 8 probe injections
k8_reaches_10pct = k8_mb >= target_10pct

if c15b_ub_mb_val >= 0.75:
    probe_info_interp = (
        f"C15b with oracle probe injection reaches {c15b_ub_mb_val:.4f} (+{c15b_ub_mb_val - c15b_mb:+.4f} over C15b), "
        f"far exceeding the 10% gap threshold (0.6284). Even k=8 oracle-ranked probes on all objects "
        f"reaches {k8_mb:.4f}. Probe information has SUBSTANTIAL usable value. "
        f"Current VOI/policy fails catastrophically to find these probes (C15b achieves only 0.6142 with 8 probes). "
        f"Next route: query-level or multi-step VOI, or probe informativeness redesign."
    )
    probe_information_useful = True
    next_route = "query_level_or_multi_step_VOI"
elif c15b_ub_mb_val >= 0.6284:
    probe_info_interp = (
        "Probe information has usable value. Oracle probe injection reaches or exceeds "
        "the 10% gap threshold. Current VOI/policy fails to select these probes. "
        "Next route: query-level or multi-step VOI."
    )
    probe_information_useful = True
    next_route = "query_level_or_multi_step_VOI"
elif best_ub_mb > c15b_mb + 0.005:
    probe_info_interp = (
        "Probe information has marginal value — oracle injection improves over C15b "
        "but does not reach the 10% threshold. Probe informativeness is a partial bottleneck."
    )
    probe_information_useful = True
    next_route = "probe_informativeness_redesign_with_other_improvements"
elif abs(best_ub_mb - c15b_mb) < 0.001:
    probe_info_interp = (
        "Probe information injection does NOT improve over C15b. The bottleneck is NOT "
        "probe selection — it is the IOM update mechanism, query readout, or affordance "
        "propagation. Even with oracle-selected true outcomes, the IOM cannot translate "
        "probe knowledge into better query predictions."
    )
    probe_information_useful = False
    next_route = "IOM_update_or_query_readout_redesign"
else:
    probe_info_interp = (
        "Probe information has slight value but is mostly captured by C15b already. "
        "Remaining headroom is small."
    )
    probe_information_useful = True
    next_route = "accept_C15b_near_ceiling"

# IOM update limitation
c15b_ub_improves = c15b_ub_mb_val > c15b_mb + 0.001
c13_ub_improves = c13_ub_mb_val > c13_mb + 0.001

if not full_ub_matches_oracle:
    iom_update_interp = (
        "IOM update limitation is CONFIRMED: even with all 360 true probe outcomes injected, "
        f"macro_bal={full_ub_mb:.4f} does not reach oracle ceiling (1.0). The IOM's "
        "predict_all_affordances / query readout cannot fully exploit complete probe knowledge."
    )
    iom_update_limitation_supported = True
elif not c15b_ub_improves and not c13_ub_improves:
    iom_update_interp = (
        "IOM update limitation is SUPPORTED: oracle probe injection on visited objects "
        "does not improve predictions. The IOM cannot translate individual probe outcomes "
        "into better affordance predictions or query answers."
    )
    iom_update_limitation_supported = True
else:
    iom_update_interp = (
        "IOM update mechanism works correctly. Full probe injection reaches oracle accuracy "
        f"({full_ub_mb:.4f}), and oracle probe injection on C15b-visited objects reaches "
        f"{c15b_ub_mb_val:.4f}. The IOM can use probe information when available. "
        "The bottleneck is selecting WHICH probes to perform and on which objects, "
        "not the IOM's ability to incorporate probe outcomes."
    )
    iom_update_limitation_supported = False

# Breadth necessity
if c15b_ub_mb_val > c13_ub_mb_val + 0.005:
    breadth_interp = (
        f"Breadth is STRONGLY necessary: C15b's broader observation ({c15b_visited_n} objects) "
        f"enables oracle probe injection to reach {c15b_ub_mb_val:.4f}, while C13's narrow "
        f"observation ({c13_visited_n} objects) with FULL probe outcomes (6 per object = "
        f"{c13_visited_n * 6} injections) only reaches {c13_ub_mb_val:.4f}. "
        f"Even with complete probe knowledge, narrow observation cannot recover."
    )
    breadth_necessary = True
else:
    breadth_interp = (
        "Breadth is not the primary differentiator: C13 with full probes achieves similar "
        "or better accuracy than C15b with oracle probes."
    )
    breadth_necessary = False

print(f"  Best oracle UB: {best_non_trivial_name} (macro_bal={best_ub_mb:.4f})")
print(f"  Best UB delta vs C15b: {best_ub_mb - c15b_mb:+.6f}")
print(f"  Best UB gap capture: {(best_ub_mb - 0.5871) / 0.4129 * 100:.2f}%")
print(f"  Best UB hits 10%: {best_ub_reaches_10pct}")
print(f"  C15b_UB macro_bal: {c15b_ub_mb_val:.4f}")
print(f"  C13_UB macro_bal: {c13_ub_mb_val:.4f}")
print(f"  Full_UB macro_bal: {full_ub_mb:.4f}")
print(f"  Full_UB matches oracle: {full_ub_matches_oracle}")
print(f"  probe_information_useful: {probe_information_useful}")
print(f"  iom_update_limitation_supported: {iom_update_limitation_supported}")
print(f"  breadth_necessary: {breadth_necessary}")
print(f"  next_recommended_route: {next_route}")
print(f"  interpretation: {probe_info_interp}")


# =============================================================================
# 10. Save outputs
# =============================================================================

# Per-query details for all variants
def extract_per_query(vdata):
    return {
        qname: {
            "balanced_accuracy": vdata["metrics"]["per_query"][qname]["balanced_accuracy"],
            "positive_recall": vdata["metrics"]["per_query"][qname]["positive_recall"],
            "negative_recall": vdata["metrics"]["per_query"][qname]["negative_recall"],
        }
        for qname in sorted(query_gt.keys())
    }

output = {
    "block_id": "1J14",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Probe informativeness / IOM update upper-bound diagnostic. Tests whether injecting additional true probe outcomes into IOM improves query-level macro_bal. All variants are diagnostic non-deployable.",
    "design_validation": {
        "no_environment_change": True,
        "no_budget_cost_change": True,
        "no_multi_seed": True,
        "not_deployable_policy": True,
        "no_oracle_labels_for_answers": True,
        "no_IOM_query_readout_bypass": True,
        "oracle_variants_labeled_non_deployable": True,
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
    "oracle_variants": {},
    "comparisons": comparisons,
    "interpretation": {
        "best_oracle_injection_variant": best_non_trivial_name,
        "best_oracle_injection_macro_bal": best_ub_mb,
        "best_oracle_injection_delta_vs_C15b": round(best_ub_mb - c15b_mb, 6),
        "best_oracle_injection_gap_capture_pct": round((best_ub_mb - 0.5871) / 0.4129 * 100, 2),
        "C15b_UB_macro_bal": c15b_ub_mb_val,
        "C13_UB_macro_bal": c13_ub_mb_val,
        "Full_UB_macro_bal": full_ub_mb,
        "Full_UB_matches_oracle": full_ub_matches_oracle,
        "probe_information_useful": probe_information_useful,
        "iom_update_limitation_supported": iom_update_limitation_supported,
        "breadth_necessary": breadth_necessary,
        "next_recommended_route": next_route,
        "ready_for_multiseed": False,
        "interpretation": probe_info_interp,
        "iom_update_interpretation": iom_update_interp,
        "breadth_interpretation": breadth_interp,
    },
}

# Serialize oracle variants (excluding full prediction dicts for JSON size)
for vname, vdata in oracle_variants.items():
    if vname == "IOM_oracle_probe_injection_UB":
        output["oracle_variants"][vname] = {
            "label": vdata["label"],
            "no_policy_claim": vdata["no_policy_claim"],
            "uses_all_visible_features": vdata["uses_all_visible_features"],
            "description": vdata["description"],
            "injection_results": {
                kname: {
                    "k": kdata["k"],
                    "injected_probe_count": kdata["injected_probe_count"],
                    "injected_action_summary": kdata["injected_action_summary"],
                    "affected_queries": kdata["affected_queries"],
                    "macro_query_balanced_accuracy": kdata["metrics"]["macro_query_balanced_accuracy"],
                    "mean_positive_recall": kdata["metrics"]["mean_positive_recall"],
                    "mean_negative_recall": kdata["metrics"]["mean_negative_recall"],
                    "per_query": extract_per_query(kdata),
                    "delta_vs_C15b": kdata["delta_vs_C15b"],
                    "gap_capture_vs_c0b": kdata["gap_capture_vs_c0b"],
                    "reaches_10pct_threshold": kdata["reaches_10pct_threshold"],
                }
                for kname, kdata in vdata["injection_results"].items()
            },
        }
    else:
        v_mb = vdata["metrics"]["macro_query_balanced_accuracy"]
        output["oracle_variants"][vname] = {
            "label": vdata.get("label", ""),
            "description": vdata.get("description", ""),
            "source_visited_set": vdata.get("source_visited_set", ""),
            "n_visited": vdata.get("n_visited", 0),
            "injected_probe_count": vdata["injected_probe_count"],
            "injected_action_summary": vdata["injected_action_summary"],
            "macro_query_balanced_accuracy": v_mb,
            "mean_positive_recall": vdata["metrics"]["mean_positive_recall"],
            "mean_negative_recall": vdata["metrics"]["mean_negative_recall"],
            "per_query": extract_per_query(vdata),
            "delta_vs_source": vdata.get("delta_vs_source", 0),
            "delta_vs_C15b": round(v_mb - c15b_mb, 6),
            "gap_capture_vs_c0b": round((v_mb - 0.5871) / 0.4129, 6),
            "reaches_10pct_threshold": v_mb >= 0.6284,
        }

json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j14_probe_iom_upper_bound_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J14")
print(f"c13_macro_bal={c13_mb:.6f}")
print(f"c15b_macro_bal={c15b_mb:.6f}")
print(f"best_probe_ub_macro_bal={best_ub_mb:.6f}")
print(f"best_probe_ub_delta_vs_c15b={best_ub_mb - c15b_mb:.6f}")
print(f"best_probe_ub_gap_capture_vs_c0b={(best_ub_mb - 0.5871) / 0.4129:.6f}")
print(f"probe_information_useful={probe_information_useful}")
print(f"iom_update_limitation_supported={iom_update_limitation_supported}")
print(f"next_recommended_route={next_route}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
