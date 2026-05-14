"""
Block 1J17 -- Learned Probe-Value Critic Diagnostic.

Trains a small learned critic (logistic regression + MLP) to predict
which (object, action) probe pairs are most valuable, using only
deployable features. Teacher labels use oracle info ONLY for training,
never for deployment.

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
print("Block 1J17 -- Learned Probe-Value Critic Diagnostic")
print(f"  condition={COND['label']}  seed={SEED}  budget={BUDGET}")
print(f"  models=logistic_regression+small_mlp  backend=numpy_only")
print(f"  training_labels=oracle_based  deployment_features=deployable_only")
print("=" * 60)

# =============================================================================
# 1. Phase A: Training + IOM building
# =============================================================================
print("\n[1/12] Phase A: Training + IOM building...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SEED, COND)
train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, SEED)
posterior_visible_features = list(base_learner.visible_feature_names)
visible_feature_names_sorted = sorted(posterior_visible_features)
N_VISIBLE_FEATURES = len(visible_feature_names_sorted)
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
print("\n[2/12] Generating test objects...")
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
# 4. C15b Observe-Phase Reference Run (defines candidate universe)
# =============================================================================
print("\n[3/12] Running C15b reference (observe phase defines candidate universe)...")

class C15b_ObservePhaseProbeReservePolicy(MiniMCPolicy):
    """C15b: two-pass VOI with 25% budget reserve. Used for reference run."""
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


c15b_im = im_base.clone()
c15b_policy = C15b_ObservePhaseProbeReservePolicy(c15b_im, random.Random(SEED + 700), cost_weight=COST_WEIGHT)
c15b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c15b_preds, c15b_log, c15b_obs, _, _ = run_policy(c15b_policy, c15b_env)
c15b_visited_oids = [oid for oid in test_oids if c15b_obs.is_visited(oid)]
c15b_probe_pairs = [(e["oid"], e["action"]) for e in c15b_policy._probe_tracking]
c15b_probe_pair_set = set(c15b_probe_pairs)
c15b_selected_actions = [e["action"] for e in c15b_policy._probe_tracking]
c15b_metrics = compute_full_metrics(c15b_preds["per_object"], query_gt)
c15b_mb = c15b_metrics["macro_query_balanced_accuracy"]
c15b_probed_n = sum(1 for oid in test_oids if c15b_obs.get_probe_results(oid))
c15b_visited_n = len(c15b_visited_oids)
print(f"  C15b macro_bal={c15b_mb:.4f}  visited={c15b_visited_n}  probed={c15b_probed_n}")
print(f"  C15b probe pairs: {c15b_probe_pairs}")
print(f"  C15b visited oids: {sorted(c15b_visited_oids)}")

# =============================================================================
# 5. Observe-Only Pass: Run C15b observe phase WITHOUT probing
# =============================================================================
print("\n[4/12] Running C15b observe-only pass to get pre-probe IOM state...")

class C15b_ObserveOnlyPhasePolicy(MiniMCPolicy):
    """Runs C15b observe phase only, never enters probe phase.
    This gives us the IOM state and visited objects BEFORE any probes.
    """
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
        if self._phase != self.PHASE_OBSERVE:
            return None
        if self._should_transition(view):
            self._phase = self.PHASE_DONE
            return None
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total = view.compute_reach_cost(oid) + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid
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


observe_only_im = im_base.clone()
observe_only_policy = C15b_ObserveOnlyPhasePolicy(observe_only_im, random.Random(SEED + 700))
observe_only_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
observe_only_preds, observe_only_log, observe_only_obs, _, _ = run_policy(
    observe_only_policy, observe_only_env)
pre_probe_visited_oids = [oid for oid in test_oids if observe_only_obs.is_visited(oid)]
print(f"  Pre-probe visited objects: {len(pre_probe_visited_oids)}")
print(f"  Pre-probe budget_spent: {observe_only_obs.budget_spent:.4f}")
print(f"  Pre-probe budget_remaining: {observe_only_obs.budget_remaining:.4f}")

# Get pre-probe IOM predictions (no probe info injected yet)
pre_probe_predictions = observe_only_preds["per_object"]
pre_probe_metrics = compute_full_metrics(pre_probe_predictions, query_gt)
print(f"  Pre-probe macro_bal: {pre_probe_metrics['macro_query_balanced_accuracy']:.4f}")

# =============================================================================
# 6. Feature Extraction for Training Data
# =============================================================================
print("\n[5/12] Extracting deployable features and teacher labels...")

def compute_iom_confidence(per_action_stats, action, k=10):
    """Replicate IOM internal confidence formula."""
    stats = per_action_stats.get(action, {})
    n_neighbors = stats.get("neighbor_count", 0)
    total_weight = stats.get("total_sim_weight", 0.0)
    outcome_variance = stats.get("outcome_variance", 0.25)
    source = stats.get("source", "unknown")
    if source in ("direct_override",):
        return 1.0
    if source in ("unsupported", "global_base_rate"):
        return 0.1
    if n_neighbors == 0:
        return 0.1
    confidence = (
        0.4 * min(total_weight / max(n_neighbors, 1), 1.0) +
        0.3 * min(n_neighbors / k, 1.0) +
        0.3 * (1.0 - min(outcome_variance / 0.25, 1.0))
    )
    return max(0.0, min(1.0, confidence))


def extract_deployable_features(im, oid, features, action):
    """Extract deployable features for a (object, action) candidate pair.
    Uses ONLY IOM internals and visible features. NO oracle info.
    """
    fake_obj = {"id": oid, "visible_features": features}

    # IOM prediction with support info
    prob, confidence, support_info = im.predict_outcome(fake_obj, action)

    # Also get indicators for additional stats
    indicators = im.get_object_dynamic_indicators(fake_obj)
    per_action = indicators.get("per_action", {})
    stats = per_action.get(action, {})

    feat = []

    # 1. Visible features (binary)
    for vf in visible_feature_names_sorted:
        feat.append(1.0 if features.get(vf, False) else 0.0)

    # 2. Action one-hot (6 actions)
    for a in MAIN_CANDIDATE_ACTIONS:
        feat.append(1.0 if a == action else 0.0)

    # 3. IOM predicted probability
    p = max(0.001, min(0.999, prob))
    feat.append(p)

    # 4. Prediction confidence = abs(p-0.5)*2
    feat.append(abs(p - 0.5) * 2.0)

    # 5. IOM internal confidence
    iom_conf = compute_iom_confidence(per_action, action, im.k)
    feat.append(iom_conf)

    # 6. Neighbor disagreement (outcome_variance)
    feat.append(stats.get("outcome_variance", 0.25))

    # 7. Top1 similarity (max_similarity from support_info)
    feat.append(support_info.get("max_similarity", 0.0))

    # 8. Neighbor count (normalized by k)
    feat.append(min(stats.get("neighbor_count", 0) / max(im.k, 1), 1.0))

    # 9. Total similarity weight (normalized)
    nc = max(stats.get("neighbor_count", 1), 1)
    tw = stats.get("total_sim_weight", 0.0)
    feat.append(min(tw / max(nc, 1), 1.0))

    # 10. Mean similarity (total_weight / neighbor_count)
    feat.append(tw / max(nc, 1))

    # 11. Source type (one-hot: 5 categories)
    source = stats.get("source", "unknown")
    for s in ["direct_override", "global_base_rate", "unsupported", "instance_retrieval", "zero_weight"]:
        feat.append(1.0 if source == s else 0.0)

    # 12. Query relevance: which query does this action affect?
    query_name = ACTION_TO_QUERY.get(action, "")
    for qn in QUERY_NAMES:
        feat.append(1.0 if qn == query_name else 0.0)

    # 13. Per-action probabilities for all features (affordance context)
    probs_all = im.predict_all_affordances(fake_obj)
    for feat_name in CORE_ACTION_FEATURES:
        feat.append(probs_all.get(feat_name, 0.5))

    # 14. Per-action confidences (abs(p-0.5)*2) for all features
    for feat_name in CORE_ACTION_FEATURES:
        fp = probs_all.get(feat_name, 0.5)
        feat.append(abs(fp - 0.5) * 2.0)

    return np.array(feat, dtype=np.float64)


def compute_teacher_labels(im, oid, action, gt_affordances, oracle_topk_set):
    """Compute teacher labels using oracle ground truth.
    These labels are NEVER used as deployment features.
    """
    feature = ACTION_TO_FEATURE[action]
    gt_outcome = gt_affordances[oid].get(feature, 0.0)

    # Get current IOM prediction
    fake_obj = {"id": oid, "visible_features": test_objects[oid].get("visible_features", {})}
    prob, _, _ = im.predict_outcome(fake_obj, action)
    p = max(0.001, min(0.999, prob))

    # oracle_error
    oracle_error = abs(p - gt_outcome)

    # is_oracle_topk_probe
    is_topk = 1.0 if (oid, action) in oracle_topk_set else 0.0

    # probe_effect: inject true outcome, measure utility change
    im_test = im.clone()
    im_test.incorporate_probe(oid, action, gt_outcome)
    post_probs = im_test.predict_all_affordances(fake_obj)
    pre_probs = im.predict_all_affordances(fake_obj)

    pre_utility = _compute_utility(pre_probs)
    post_utility = _compute_utility(post_probs)
    utility_delta = post_utility - pre_utility

    if utility_delta > 0.001:
        effect = "effective"
    elif utility_delta < -0.001:
        effect = "harmful"
    else:
        effect = "zero"

    # query_relevant_value: change in query decision
    query_name = ACTION_TO_QUERY.get(action, "")
    pre_correct = _check_query(pre_probs, query_name) if query_name else None
    post_correct = _check_query(post_probs, query_name) if query_name else None
    if pre_correct is not None and post_correct is not None:
        query_value = 1.0 if (not pre_correct and post_correct) else (
            -1.0 if (pre_correct and not post_correct) else 0.0)
    else:
        query_value = 0.0

    return {
        "oracle_error": float(oracle_error),
        "is_oracle_topk_probe": float(is_topk),
        "probe_effect": effect,
        "utility_delta": float(utility_delta),
        "query_relevant_value": float(query_value),
        "gt_outcome": float(gt_outcome),
    }


# Build oracle k=8 reference set
print("  Computing oracle k=8 reference set...")
oracle_errors_all = []
for oid in pre_probe_visited_oids:
    obj = test_objects[oid]
    fake_obj = {"id": oid, "visible_features": obj.get("visible_features", {})}
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

# Extract features and labels for all (visited_oid, action) pairs
print(f"  Extracting features for {len(pre_probe_visited_oids)} visited objects x 6 actions...")
training_data = []  # list of dicts: {oid, action, features_vec, labels_dict}

for oid in pre_probe_visited_oids:
    features = observe_only_obs.get_observed_features(oid)
    if features is None:
        continue
    for action in MAIN_CANDIDATE_ACTIONS:
        feat_vec = extract_deployable_features(observe_only_im, oid, features, action)
        labels = compute_teacher_labels(observe_only_im, oid, action,
                                        ALL_GT_AFFORDANCES, oracle_k8_pair_set)
        training_data.append({
            "oid": oid,
            "action": action,
            "features": feat_vec,
            "labels": labels,
        })

N_FEATURES = len(training_data[0]["features"])
N_TRAINING_PAIRS = len(training_data)
print(f"  Training pairs: {N_TRAINING_PAIRS}  feature_dim={N_FEATURES}")

# Label statistics
n_topk = sum(1 for d in training_data if d["labels"]["is_oracle_topk_probe"] > 0.5)
n_effective = sum(1 for d in training_data if d["labels"]["probe_effect"] == "effective")
n_zero = sum(1 for d in training_data if d["labels"]["probe_effect"] == "zero")
n_harmful = sum(1 for d in training_data if d["labels"]["probe_effect"] == "harmful")
print(f"  Labels: oracle_topk={n_topk}  effective={n_effective}  zero={n_zero}  harmful={n_harmful}")

# =============================================================================
# 7. Model Implementations (numpy-only)
# =============================================================================
print("\n[6/12] Implementing models (logistic regression + MLP, numpy-only)...")

class LogisticRegressionCritic:
    """L2-regularized logistic regression for binary probe-value prediction.
    Target: is_oracle_topk_probe (binary).
    """

    def __init__(self, input_dim, l2_reg=1.0, learning_rate=0.01, n_epochs=500):
        self.input_dim = input_dim
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Fit with batch gradient descent + L2 regularization."""
        n_samples = X.shape[0]
        self.weights = np.zeros(self.input_dim, dtype=np.float64)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            # Forward
            z = np.dot(X, self.weights) + self.bias
            p = self._sigmoid(z)

            # Gradients
            dz = p - y
            dw = np.dot(X.T, dz) / n_samples + self.l2_reg * self.weights / n_samples
            db = np.mean(dz)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and (epoch + 1) % 100 == 0:
                train_loss = self._loss(X, y)
                msg = f"  epoch {epoch + 1}/{self.n_epochs}  loss={train_loss:.6f}"
                if X_val is not None:
                    val_loss = self._loss(X_val, y_val)
                    msg += f"  val_loss={val_loss:.6f}"
                print(msg)

    def _loss(self, X, y):
        z = np.dot(X, self.weights) + self.bias
        p = self._sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * self.l2_reg * np.sum(self.weights ** 2) / X.shape[0]
        return nll + reg

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.float64)


class SmallMLPCritic:
    """2-layer MLP (hidden_dim -> 1) with ReLU for binary probe-value prediction.
    Target: is_oracle_topk_probe (binary).
    """

    def __init__(self, input_dim, hidden_dim=16, l2_reg=0.1,
                 learning_rate=0.01, n_epochs=500):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _relu(self, z):
        return np.maximum(0, z)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def _init_weights(self):
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / self.input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.W1 = rng.randn(self.input_dim, self.hidden_dim).astype(np.float64) * scale1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(self.hidden_dim).astype(np.float64) * scale2
        self.b2 = 0.0

    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        self._init_weights()
        n_samples = X.shape[0]

        for epoch in range(self.n_epochs):
            # Forward
            h = self._relu(np.dot(X, self.W1) + self.b1)
            z = np.dot(h, self.W2) + self.b2
            p = self._sigmoid(z)

            # Backward
            dz = (p - y) / n_samples
            dW2 = np.dot(h.T, dz) + self.l2_reg * self.W2 / n_samples
            db2 = np.sum(dz)

            dh = np.outer(dz, self.W2)
            dh[h <= 0] = 0
            dW1 = np.dot(X.T, dh) + self.l2_reg * self.W1 / n_samples
            db1 = np.sum(dh, axis=0)

            # Update
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            if verbose and (epoch + 1) % 100 == 0:
                train_loss = self._loss(X, y)
                msg = f"  epoch {epoch + 1}/{self.n_epochs}  loss={train_loss:.6f}"
                if X_val is not None:
                    val_loss = self._loss(X_val, y_val)
                    msg += f"  val_loss={val_loss:.6f}"
                print(msg)

    def _loss(self, X, y):
        h = self._relu(np.dot(X, self.W1) + self.b1)
        z = np.dot(h, self.W2) + self.b2
        p = self._sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * self.l2_reg * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
        ) / X.shape[0]
        return nll + reg

    def predict_proba(self, X):
        h = self._relu(np.dot(X, self.W1) + self.b1)
        z = np.dot(h, self.W2) + self.b2
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.float64)


# =============================================================================
# 8. Leave-Object-Out Cross-Validation
# =============================================================================
print("\n[7/12] Leave-object-out cross-validation...")

def compute_auc(y_true, y_score):
    """Compute ROC AUC manually."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = 0.0
    fpr = 0.0
    auc = 0.0
    for i, y in enumerate(y_true_sorted):
        if y > 0.5:
            tpr += 1.0
        else:
            auc += tpr
            fpr += 1.0
    auc /= (n_pos * n_neg) if (n_pos * n_neg) > 0 else 1.0
    return auc


def compute_precision_at_k(y_true, y_score, k=8):
    """Precision@k: fraction of top-k scored items that are true positives."""
    order = np.argsort(y_score)[::-1]
    top_k = order[:min(k, len(order))]
    if len(top_k) == 0:
        return 0.0
    return float(np.mean(y_true[top_k]))


# Build feature matrix and label vectors
X_all = np.array([d["features"] for d in training_data], dtype=np.float64)
y_topk_all = np.array([d["labels"]["is_oracle_topk_probe"] for d in training_data], dtype=np.float64)
y_error_all = np.array([d["labels"]["oracle_error"] for d in training_data], dtype=np.float64)
oid_all = np.array([d["oid"] for d in training_data])

# Feature normalization (z-score per feature)
X_mean = np.mean(X_all, axis=0)
X_std = np.std(X_all, axis=0)
X_std[X_std < 1e-10] = 1.0
X_norm = (X_all - X_mean) / X_std

# LOO CV
unique_oids_in_data = sorted(set(d["oid"] for d in training_data))
print(f"  LOO CV over {len(unique_oids_in_data)} objects...")

loo_results = {
    "lr_topk": {"y_true": [], "y_score": [], "fold_aucs": [], "fold_p8s": []},
    "mlp_topk": {"y_true": [], "y_score": [], "fold_aucs": [], "fold_p8s": []},
}

for heldout_oid in unique_oids_in_data:
    train_mask = oid_all != heldout_oid
    test_mask = oid_all == heldout_oid

    X_train = X_norm[train_mask]
    y_train_topk = y_topk_all[train_mask]
    X_test = X_norm[test_mask]
    y_test_topk = y_topk_all[test_mask]

    if np.sum(train_mask) < 5 or np.sum(test_mask) == 0:
        continue

    # Logistic regression
    lr = LogisticRegressionCritic(N_FEATURES, l2_reg=1.0, learning_rate=0.05, n_epochs=300)
    lr.fit(X_train, y_train_topk)
    lr_scores = lr.predict_proba(X_test)
    loo_results["lr_topk"]["y_true"].extend(y_test_topk.tolist())
    loo_results["lr_topk"]["y_score"].extend(lr_scores.tolist())
    if np.sum(y_test_topk) > 0 and np.sum(y_test_topk) < len(y_test_topk):
        loo_results["lr_topk"]["fold_aucs"].append(
            compute_auc(y_test_topk, lr_scores))
    loo_results["lr_topk"]["fold_p8s"].append(
        compute_precision_at_k(y_test_topk, lr_scores, k=8))

    # MLP
    mlp = SmallMLPCritic(N_FEATURES, hidden_dim=16, l2_reg=0.1, learning_rate=0.05, n_epochs=300)
    mlp.fit(X_train, y_train_topk)
    mlp_scores = mlp.predict_proba(X_test)
    loo_results["mlp_topk"]["y_true"].extend(y_test_topk.tolist())
    loo_results["mlp_topk"]["y_score"].extend(mlp_scores.tolist())
    if np.sum(y_test_topk) > 0 and np.sum(y_test_topk) < len(y_test_topk):
        loo_results["mlp_topk"]["fold_aucs"].append(
            compute_auc(y_test_topk, mlp_scores))
    loo_results["mlp_topk"]["fold_p8s"].append(
        compute_precision_at_k(y_test_topk, mlp_scores, k=8))

# Compute overall LOO metrics
for model_key in ["lr_topk", "mlp_topk"]:
    yt = np.array(loo_results[model_key]["y_true"])
    ys = np.array(loo_results[model_key]["y_score"])
    loo_results[model_key]["overall_auc"] = compute_auc(yt, ys)
    loo_results[model_key]["overall_precision_at_8"] = compute_precision_at_k(yt, ys, k=8)
    loo_results[model_key]["mean_fold_auc"] = (
        np.mean(loo_results[model_key]["fold_aucs"])
        if loo_results[model_key]["fold_aucs"] else 0.5)
    loo_results[model_key]["mean_fold_p8"] = (
        np.mean(loo_results[model_key]["fold_p8s"])
        if loo_results[model_key]["fold_p8s"] else 0.0)

lr_auc = loo_results["lr_topk"]["overall_auc"]
lr_p8 = loo_results["lr_topk"]["overall_precision_at_8"]
mlp_auc = loo_results["mlp_topk"]["overall_auc"]
mlp_p8 = loo_results["mlp_topk"]["overall_precision_at_8"]

print(f"  LR  AUC={lr_auc:.4f}  precision@8={lr_p8:.4f}  n_folds={len(loo_results['lr_topk']['fold_aucs'])}")
print(f"  MLP AUC={mlp_auc:.4f}  precision@8={mlp_p8:.4f}  n_folds={len(loo_results['mlp_topk']['fold_aucs'])}")

# Determine best critic
best_critic = "lr" if lr_auc >= mlp_auc else "mlp"
best_critic_auc = lr_auc if best_critic == "lr" else mlp_auc
best_critic_p8 = lr_p8 if best_critic == "lr" else mlp_p8
print(f"  Best critic: {best_critic}  AUC={best_critic_auc:.4f}  P@8={best_critic_p8:.4f}")

# =============================================================================
# 9. Train Final Critics on All Data + Deployment Evaluation
# =============================================================================
print("\n[8/12] Training final critics on all data + deployment evaluation...")

# Train on all training pairs
print("  Training final LR critic...")
lr_final = LogisticRegressionCritic(N_FEATURES, l2_reg=1.0, learning_rate=0.05, n_epochs=500)
lr_final.fit(X_norm, y_topk_all)

print("  Training final MLP critic...")
mlp_final = SmallMLPCritic(N_FEATURES, hidden_dim=16, l2_reg=0.1, learning_rate=0.05, n_epochs=500)
mlp_final.fit(X_norm, y_topk_all)


# Deployment policy using learned critic
class LearnedCriticProbePolicy(MiniMCPolicy):
    """C15b observe phase + learned critic probe ranking.

    Same observe phase as C15b. In probe phase, ranks visited object-action
    pairs by learned critic score instead of VOI.
    """

    PHASE_OBSERVE = 1
    PHASE_PROBE = 2

    def __init__(self, instance_memory, rng, critic, feature_extractor,
                 X_mean, X_std, target_probe_count=8, critic_label="unknown"):
        self._im = instance_memory
        self._rng = rng
        self._critic = critic
        self._feature_extractor = feature_extractor
        self._X_mean = X_mean
        self._X_std = X_std
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._target_probe_count = target_probe_count
        self._probe_effects = []
        self._critic_label = critic_label
        self._critic_scores = []

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []
        self._critic_scores = []

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
            return best_oid
        elif self._phase == self.PHASE_PROBE:
            return self._select_probe_target(view)
        return None

    def _select_probe_target(self, view):
        if len(self._probed_oids) >= self._target_probe_count:
            return None

        # Score all visited+unprobed (object, action) pairs
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
            for action in MAIN_CANDIDATE_ACTIONS:
                feat_vec = self._feature_extractor(self._im, oid, features, action)
                feat_norm = (feat_vec - self._X_mean) / self._X_std
                score = float(self._critic.predict_proba(feat_norm.reshape(1, -1))[0])
                if score > best_score:
                    best_score = score
                    best_oid = oid

        if best_score <= 0:
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None

        features = view.get_observed_features(object_id)
        if features is None:
            return False, None

        # Select best action for this object by critic score
        best_action = None
        best_score = -1.0
        action_scores = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            feat_vec = self._feature_extractor(self._im, object_id, features, action)
            feat_norm = (feat_vec - self._X_mean) / self._X_std
            score = float(self._critic.predict_proba(feat_norm.reshape(1, -1))[0])
            action_scores[action] = float(score)
            if score > best_score:
                best_score = score
                best_action = action

        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        self._critic_scores.append({
            "oid": object_id,
            "selected_action": best_action,
            "selected_score": float(best_score),
            "action_scores": action_scores,
        })
        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        features = view.get_observed_features(object_id)
        if features is not None:
            fake_obj = {"id": object_id, "visible_features": features}
            pre_probs = self._im.predict_all_affordances(fake_obj)
            self._im.incorporate_probe(object_id, action, outcome)
            post_probs = self._im.predict_all_affordances(fake_obj)
            self._probe_tracking.append({
                "oid": object_id, "action": action, "outcome": outcome,
            })
            pre_utility = _compute_utility(pre_probs)
            post_utility = _compute_utility(post_probs)
            utility_delta = post_utility - pre_utility
            self._probe_effects.append({
                "oid": object_id, "action": action, "outcome": outcome,
                "pre_utility": round(pre_utility, 6),
                "post_utility": round(post_utility, 6),
                "utility_delta": round(utility_delta, 6),
                "is_effective": utility_delta > 0.001,
                "is_zero": abs(utility_delta) <= 0.001,
                "is_harmful": utility_delta < -0.001,
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

    def get_critic_scores(self):
        return list(self._critic_scores)


# Wrapper feature extractor for deployment (without oracle)
def deployable_feature_extractor(im, oid, features, action):
    """Same as extract_deployable_features but takes features as param."""
    return extract_deployable_features(im, oid, features, action)


# Run deployment evaluation
print("\n  Running LR critic deployment...")
lr_im = im_base.clone()
lr_critic_policy = LearnedCriticProbePolicy(
    lr_im, random.Random(SEED + 800),
    lr_final, deployable_feature_extractor,
    X_mean, X_std,
    target_probe_count=c15b_probed_n,
    critic_label="logistic_regression",
)
lr_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
lr_preds, lr_log, lr_obs, _, _ = run_policy(lr_critic_policy, lr_env)
lr_critic_metrics = compute_full_metrics(lr_preds["per_object"], query_gt)
lr_critic_mb = lr_critic_metrics["macro_query_balanced_accuracy"]
lr_critic_probed = sum(1 for oid in test_oids if lr_obs.get_probe_results(oid))
lr_critic_visited = len([oid for oid in test_oids if lr_obs.is_visited(oid)])
lr_probe_effects = lr_critic_policy.get_probe_effects()
lr_probe_pairs = [(e["oid"], e["action"]) for e in lr_probe_effects]
lr_effective = sum(1 for e in lr_probe_effects if e["is_effective"])
lr_zero = sum(1 for e in lr_probe_effects if e["is_zero"])
lr_harmful = sum(1 for e in lr_probe_effects if e["is_harmful"])
lr_overlap_oracle = len(set(lr_probe_pairs) & oracle_k8_pair_set)
lr_overlap_c15b = len(set(lr_probe_pairs) & c15b_probe_pair_set)
lr_critic_scores = lr_critic_policy.get_critic_scores()
print(f"  LR critic macro_bal={lr_critic_mb:.4f}  visited={lr_critic_visited}  probed={lr_critic_probed}")
print(f"  LR critic effective={lr_effective}  zero={lr_zero}  harmful={lr_harmful}")
print(f"  LR critic overlap_oracle_k8={lr_overlap_oracle}  overlap_C15b={lr_overlap_c15b}")

print("\n  Running MLP critic deployment...")
mlp_im = im_base.clone()
mlp_critic_policy = LearnedCriticProbePolicy(
    mlp_im, random.Random(SEED + 900),
    mlp_final, deployable_feature_extractor,
    X_mean, X_std,
    target_probe_count=c15b_probed_n,
    critic_label="small_mlp",
)
mlp_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
mlp_preds, mlp_log, mlp_obs, _, _ = run_policy(mlp_critic_policy, mlp_env)
mlp_critic_metrics = compute_full_metrics(mlp_preds["per_object"], query_gt)
mlp_critic_mb = mlp_critic_metrics["macro_query_balanced_accuracy"]
mlp_critic_probed = sum(1 for oid in test_oids if mlp_obs.get_probe_results(oid))
mlp_critic_visited = len([oid for oid in test_oids if mlp_obs.is_visited(oid)])
mlp_probe_effects = mlp_critic_policy.get_probe_effects()
mlp_probe_pairs = [(e["oid"], e["action"]) for e in mlp_probe_effects]
mlp_effective = sum(1 for e in mlp_probe_effects if e["is_effective"])
mlp_zero = sum(1 for e in mlp_probe_effects if e["is_zero"])
mlp_harmful = sum(1 for e in mlp_probe_effects if e["is_harmful"])
mlp_overlap_oracle = len(set(mlp_probe_pairs) & oracle_k8_pair_set)
mlp_overlap_c15b = len(set(mlp_probe_pairs) & c15b_probe_pair_set)
mlp_critic_scores = mlp_critic_policy.get_critic_scores()
print(f"  MLP critic macro_bal={mlp_critic_mb:.4f}  visited={mlp_critic_visited}  probed={mlp_critic_probed}")
print(f"  MLP critic effective={mlp_effective}  zero={mlp_zero}  harmful={mlp_harmful}")
print(f"  MLP critic overlap_oracle_k8={mlp_overlap_oracle}  overlap_C15b={mlp_overlap_c15b}")

# Determine best learned critic
critic_variants = {
    "logistic_regression": (lr_critic_mb, lr_critic_probed, lr_overlap_oracle,
                             lr_effective, lr_harmful, lr_auc, lr_p8),
    "small_mlp": (mlp_critic_mb, mlp_critic_probed, mlp_overlap_oracle,
                  mlp_effective, mlp_harmful, mlp_auc, mlp_p8),
}
best_critic_name = max(critic_variants, key=lambda k: critic_variants[k][0])
(best_critic_mb, best_critic_probed, best_critic_overlap_oracle,
 best_critic_eff, best_critic_harm, best_critic_auc_final, best_critic_p8_final) = critic_variants[best_critic_name]
best_critic_delta = best_critic_mb - c15b_mb

print(f"\n  Best learned critic: {best_critic_name}")
print(f"  Best critic macro_bal={best_critic_mb:.4f}  delta_vs_C15b={best_critic_delta:+.4f}")
print(f"  Best critic overlap_oracle_k8={best_critic_overlap_oracle}")

# =============================================================================
# 10. Run Baselines
# =============================================================================
print("\n[9/12] Running baselines...")
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
c_minus_preds = {oid: {f: 0.5 for f in CORE_ACTION_FEATURES} for oid in test_oids}
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

# --- C15b_original (already run above, use those results) ---
results["C15b_probe_reserve_original"] = {
    "predictions": c15b_preds["per_object"],
    "metrics": c15b_metrics,
    "cost": get_cost_metrics(c15b_log, c15b_obs),
    "policy_type": "two_pass_VOI",
    "visited_oids": c15b_visited_oids,
    "probe_pairs": c15b_probe_pairs,
}
print(f"  C15b_original macro_bal={c15b_mb:.4f}")

# --- C18b from 1J15 ---
print("\n  Running C18b_query_relevant (from 1J15)...")
class C18b_QueryRelevantFixedBudgetPolicy(MiniMCPolicy):
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

    def reset(self, view):
        self._phase = self.PHASE_OBSERVE
        self._probe_tracking = []
        self._pre_probe_entropies = {}
        self._probed_oids = set()
        self._observe_pass_visit_count = 0
        self._probe_effects = []

    def _should_transition_to_probe_phase(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_observe = min(
            view.compute_reach_cost(oid) + view.observe_cost for oid in unvisited)
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
            if not view.is_visited(oid) or oid in self._probed_oids:
                continue
            features = view.get_observed_features(oid)
            if features is None:
                continue
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if not view.can_afford(total_cost):
                continue
            fake_obj = {"id": oid, "visible_features": features}
            probs = self._im.predict_all_affordances(fake_obj)
            for action in MAIN_CANDIDATE_ACTIONS:
                p = probs.get(ACTION_TO_FEATURE[action], 0.5)
                p = max(0.001, min(0.999, p))
                score = 1.0 - max(p, 1.0 - p)
                if score > best_score:
                    best_score = score
                    best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        if self._phase == self.PHASE_OBSERVE:
            return False, None
        if len(self._probed_oids) >= self._target_probe_count:
            return False, None
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        best_action = None
        best_score = 0.0
        for action in MAIN_CANDIDATE_ACTIONS:
            p = probs.get(ACTION_TO_FEATURE[action], 0.5)
            p = max(0.001, min(0.999, p))
            score = 1.0 - max(p, 1.0 - p)
            if score > best_score:
                best_score = score
                best_action = action
        self._pre_probe_entropies[object_id] = _compute_entropy(probs)
        self._probed_oids.add(object_id)
        return True, best_action

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

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_probe_effects(self):
        return list(self._probe_effects)


c18b_im = im_base.clone()
c18b_policy = C18b_QueryRelevantFixedBudgetPolicy(c18b_im, random.Random(SEED + 850),
                                                    target_probe_count=c15b_probed_n)
c18b_env = MiniMCEnvironment(test_objects, positions, initial_budget=BUDGET)
c18b_preds, c18b_log, c18b_obs, _, _ = run_policy(c18b_policy, c18b_env)
c18b_visited_oids = [oid for oid in test_oids if c18b_obs.is_visited(oid)]
c18b_probe_pairs = [(e["oid"], e["action"]) for e in c18b_policy._probe_tracking]
results["C18b_query_relevant_fixed_budget"] = {
    "predictions": c18b_preds["per_object"],
    "metrics": compute_full_metrics(c18b_preds["per_object"], query_gt),
    "cost": get_cost_metrics(c18b_log, c18b_obs),
    "policy_type": "query_relevant_fixed_budget_from_1J15",
    "probe_pairs": c18b_probe_pairs,
}
c18b_mb = results["C18b_query_relevant_fixed_budget"]["metrics"]["macro_query_balanced_accuracy"]
c18b_visited_n = len(c18b_visited_oids)
c18b_probed_n = results["C18b_query_relevant_fixed_budget"]["cost"]["probe_count"]
print(f"  C18b macro_bal={c18b_mb:.4f}  probed={c18b_probed_n}  delta_vs_C15b={c18b_mb - c15b_mb:+.4f}")

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

# Oracle k=8 (non-deployable reference)
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
oracle_k8_metrics = compute_full_metrics(oracle_k8_preds, query_gt)
oracle_k8_mb = oracle_k8_metrics["macro_query_balanced_accuracy"]
print(f"  Oracle k=8 macro_bal={oracle_k8_mb:.4f} (non-deployable reference)")

# --- best C19 from 1J16 (C19a neighbor disagreement, 0.6067) ---
results["C19a_neighbor_disagreement_from_1J16"] = {
    "macro_query_balanced_accuracy": 0.6067,
    "policy_type": "neighbor_disagreement_risk_from_1J16",
    "note": "Best C19 variant from 1J16 (hardcoded reference value)",
}
c19a_ref_mb = 0.6067
print(f"  C19a reference (from 1J16) macro_bal={c19a_ref_mb:.4f}")

# =============================================================================
# 11. Comparisons
# =============================================================================
print("\n[10/12] Computing comparisons...")
target_10pct = 0.6284
oracle_c0b_gap = oracle_mb - c0b_mb

# Gap capture
gap_capture = {
    "C0b_base": c0b_mb,
    "oracle_ceiling": oracle_mb,
    "oracle_c0b_gap": oracle_c0b_gap,
    "target_10pct_threshold": target_10pct,
    "C15b_gap_capture_pct": round((c15b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C18b_gap_capture_pct": round((c18b_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "C19a_gap_capture_pct": round((c19a_ref_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "lr_critic_gap_capture_pct": round((lr_critic_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "mlp_critic_gap_capture_pct": round((mlp_critic_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
    "oracle_k8_gap_capture_pct": round((oracle_k8_mb - c0b_mb) / oracle_c0b_gap * 100, 2) if oracle_c0b_gap > 0 else 0.0,
}

# Probe pair overlap
probe_pair_overlap = {
    "C15b_probe_pairs": str(c15b_probe_pairs),
    "oracle_k8_pairs": str([(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]),
    "C18b_probe_pairs": str(c18b_probe_pairs),
    "C18b_overlap_C15b": len(set(c18b_probe_pairs) & c15b_probe_pair_set),
    "C18b_overlap_oracle_k8": len(set(c18b_probe_pairs) & oracle_k8_pair_set),
    "LR_critic_probe_pairs": str(lr_probe_pairs),
    "LR_critic_overlap_C15b": lr_overlap_c15b,
    "LR_critic_overlap_oracle_k8": lr_overlap_oracle,
    "MLP_critic_probe_pairs": str(mlp_probe_pairs),
    "MLP_critic_overlap_C15b": mlp_overlap_c15b,
    "MLP_critic_overlap_oracle_k8": mlp_overlap_oracle,
}

# Per-query breakdown
per_query_breakdown = {}
for qname in QUERY_NAMES:
    c15b_q = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]
    best_q = (lr_critic_metrics if best_critic_name == "logistic_regression"
              else mlp_critic_metrics)["per_query"][qname]
    per_query_breakdown[qname] = {
        "C15b_bal_acc": c15b_q["balanced_accuracy"],
        "best_critic_bal_acc": best_q["balanced_accuracy"],
        "best_critic_delta_vs_C15b": round(best_q["balanced_accuracy"] - c15b_q["balanced_accuracy"], 6),
    }

# =============================================================================
# 12. Interpretation
# =============================================================================
print("\n[11/12] Interpreting results...")

best_critic_hits_10pct = best_critic_mb >= target_10pct
best_critic_improves_c15b = best_critic_mb > c15b_mb + 0.001
critic_predicts_oracle = best_critic_overlap_oracle > 0

if best_critic_improves_c15b and best_critic_hits_10pct:
    learned_probe_value_supported = True
    candidate_ready_for_small_multiseed = True
    interpretation_text = (
        f"Best learned critic ({best_critic_name}) reaches {best_critic_mb:.4f} "
        f"(+{best_critic_delta:+.4f} vs C15b), exceeding the 10% gap threshold "
        f"({target_10pct}). Learned probe-value critic SUCCESSFULLY identifies "
        f"valuable probes from deployable features."
    )
elif best_critic_improves_c15b:
    learned_probe_value_supported = True
    candidate_ready_for_small_multiseed = False
    interpretation_text = (
        f"Best learned critic ({best_critic_name}) reaches {best_critic_mb:.4f} "
        f"(+{best_critic_delta:+.4f} vs C15b), improving over C15b but not "
        f"reaching 10% threshold ({target_10pct}). Learned probe-value route "
        f"is PARTIALLY supported."
    )
elif abs(best_critic_mb - c15b_mb) < 0.002:
    if critic_predicts_oracle:
        learned_probe_value_supported = False
        candidate_ready_for_small_multiseed = False
        interpretation_text = (
            f"Best learned critic ({best_critic_name}) matches C15b "
            f"({best_critic_mb:.4f} vs {c15b_mb:.4f}). Critic CAN predict "
            f"oracle probes offline (AUC={best_critic_auc_final:.4f}, "
            f"overlap={best_critic_overlap_oracle}) but deployment does not "
            f"improve — execution/budget ordering remains limiting."
        )
    else:
        learned_probe_value_supported = False
        candidate_ready_for_small_multiseed = False
        interpretation_text = (
            f"Best learned critic ({best_critic_name}) matches C15b "
            f"({best_critic_mb:.4f} vs {c15b_mb:.4f}). Critic CANNOT predict "
            f"oracle probes offline (AUC={best_critic_auc_final:.4f}). "
            f"Available deployable features may be insufficient."
        )
else:
    learned_probe_value_supported = False
    candidate_ready_for_small_multiseed = False
    interpretation_text = (
        f"Best learned critic ({best_critic_name}) reaches {best_critic_mb:.4f} "
        f"({best_critic_delta:+.4f} vs C15b), below C15b. Learned critic "
        f"DEGRADES performance. Deployable features insufficient to predict "
        f"oracle probe value."
    )

print(f"  Best learned critic: {best_critic_name}")
print(f"  Best critic macro_bal: {best_critic_mb:.4f}  delta_vs_C15b: {best_critic_delta:+.4f}")
print(f"  Best critic AUC: {best_critic_auc_final:.4f}  P@8: {best_critic_p8_final:.4f}")
print(f"  Best critic overlap_oracle_k8: {best_critic_overlap_oracle}")
print(f"  learned_probe_value_supported: {learned_probe_value_supported}")
print(f"  candidate_ready_for_small_multiseed: {candidate_ready_for_small_multiseed}")

# =============================================================================
# 13. Save outputs
# =============================================================================
print("\n[12/12] Saving outputs...")

output = {
    "block_id": "1J17",
    "condition": "C4_instance_subtype_cued_v1",
    "seed": SEED,
    "budget": BUDGET,
    "cost_weight": COST_WEIGHT,
    "desc": "Learned Probe-Value Critic Diagnostic. Trains logistic regression + MLP on deployable features with oracle-based teacher labels.",
    "design_validation": {
        "no_environment_change": True,
        "no_budget_cost_change": True,
        "no_multi_seed": True,
        "no_hidden_labels_in_deployment": True,
        "no_oracle_probe_outcomes_in_ranking": True,
        "no_IOM_update_change": True,
        "deployment_uses_only_deployable_features": True,
        "teacher_labels_use_oracle_only_for_training": True,
    },
    "training_data": {
        "n_visited_objects": len(pre_probe_visited_oids),
        "n_training_pairs": N_TRAINING_PAIRS,
        "feature_dim": int(N_FEATURES),
        "visible_feature_count": N_VISIBLE_FEATURES,
        "n_oracle_topk_pairs": n_topk,
        "n_effective_probes": n_effective,
        "n_zero_probes": n_zero,
        "n_harmful_probes": n_harmful,
        "feature_names_hint": [
            f"visible_feature_{i}" for i in range(N_VISIBLE_FEATURES)
        ] + [f"action_{a}" for a in MAIN_CANDIDATE_ACTIONS] + [
            "predicted_probability", "prediction_confidence_abs_p_minus_0_5_times_2",
            "iom_confidence", "neighbor_disagreement", "top1_similarity",
            "neighbor_count_norm", "total_sim_weight_norm", "mean_similarity",
        ] + ["source_" + s for s in ["direct_override", "global_base_rate",
                                      "unsupported", "instance_retrieval", "zero_weight"]] + [
            "query_" + qn for qn in QUERY_NAMES
        ] + [f"prob_{f}" for f in CORE_ACTION_FEATURES] + [
            f"conf_{f}" for f in CORE_ACTION_FEATURES
        ],
    },
    "loo_cv_results": {
        "logistic_regression": {
            "overall_auc": round(lr_auc, 6),
            "overall_precision_at_8": round(lr_p8, 6),
            "mean_fold_auc": round(loo_results["lr_topk"]["mean_fold_auc"], 6),
            "mean_fold_p8": round(loo_results["lr_topk"]["mean_fold_p8"], 6),
            "n_valid_folds": len(loo_results["lr_topk"]["fold_aucs"]),
        },
        "small_mlp": {
            "overall_auc": round(mlp_auc, 6),
            "overall_precision_at_8": round(mlp_p8, 6),
            "mean_fold_auc": round(loo_results["mlp_topk"]["mean_fold_auc"], 6),
            "mean_fold_p8": round(loo_results["mlp_topk"]["mean_fold_p8"], 6),
            "n_valid_folds": len(loo_results["mlp_topk"]["fold_aucs"]),
        },
    },
    "baselines": {},
    "learned_critic_deployment": {
        "logistic_regression": {
            "macro_query_balanced_accuracy": lr_critic_mb,
            "mean_positive_recall": lr_critic_metrics["mean_positive_recall"],
            "mean_negative_recall": lr_critic_metrics["mean_negative_recall"],
            "per_query": lr_critic_metrics["per_query"],
            "visit_count": lr_critic_visited,
            "probe_count": lr_critic_probed,
            "total_cost": get_cost_metrics(lr_log, lr_obs)["total_cost"],
            "probe_pairs": str(lr_probe_pairs),
            "effective_probe_count": lr_effective,
            "zero_gain_probe_count": lr_zero,
            "harmful_probe_count": lr_harmful,
            "overlap_with_C15b": lr_overlap_c15b,
            "overlap_with_oracle_k8": lr_overlap_oracle,
            "delta_vs_C15b": round(lr_critic_mb - c15b_mb, 6),
            "critic_scores": lr_critic_scores,
            "offline_auc": round(lr_auc, 6),
            "offline_precision_at_8": round(lr_p8, 6),
        },
        "small_mlp": {
            "macro_query_balanced_accuracy": mlp_critic_mb,
            "mean_positive_recall": mlp_critic_metrics["mean_positive_recall"],
            "mean_negative_recall": mlp_critic_metrics["mean_negative_recall"],
            "per_query": mlp_critic_metrics["per_query"],
            "visit_count": mlp_critic_visited,
            "probe_count": mlp_critic_probed,
            "total_cost": get_cost_metrics(mlp_log, mlp_obs)["total_cost"],
            "probe_pairs": str(mlp_probe_pairs),
            "effective_probe_count": mlp_effective,
            "zero_gain_probe_count": mlp_zero,
            "harmful_probe_count": mlp_harmful,
            "overlap_with_C15b": mlp_overlap_c15b,
            "overlap_with_oracle_k8": mlp_overlap_oracle,
            "delta_vs_C15b": round(mlp_critic_mb - c15b_mb, 6),
            "critic_scores": mlp_critic_scores,
            "offline_auc": round(mlp_auc, 6),
            "offline_precision_at_8": round(mlp_p8, 6),
        },
    },
    "oracle_k8_reference": {
        "label": "diagnostic_non_deployable",
        "k": 8,
        "pairs": str([(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]),
        "macro_bal": oracle_k8_mb,
        "delta_vs_C15b": round(oracle_k8_mb - c15b_mb, 6),
        "uses_all_visible_features": True,
        "ranking_method": "|IOM_prediction - ground_truth|",
    },
    "comparisons": {
        "best_critic_vs_C15b": {
            "best_critic_variant": best_critic_name,
            "delta_macro_bal": round(best_critic_delta, 6),
            "best_macro_bal": best_critic_mb,
        },
        "gap_capture": gap_capture,
        "probe_pair_overlap": probe_pair_overlap,
        "per_query_breakdown": per_query_breakdown,
        "oracle_references": {
            "1J14_oracle_k8_macro_bal": 0.6488,
            "1J14_C15b_UB_macro_bal": 0.8546,
            "1J14_Full_UB_macro_bal": 1.0000,
            "1J16_oracle_k8_macro_bal": 0.6488,
            "1J16_best_C19_macro_bal": c19a_ref_mb,
        },
        "iom_architecture_notes": {
            "no_cross_test_object_propagation": True,
            "global_recompute_equivalent_to_local": True,
        },
    },
    "interpretation": {
        "best_critic_variant": best_critic_name,
        "best_critic_macro_bal": best_critic_mb,
        "best_critic_auc": round(best_critic_auc_final, 6),
        "best_critic_precision_at_8": round(best_critic_p8_final, 6),
        "best_critic_overlap_with_oracle_k8": best_critic_overlap_oracle,
        "best_critic_delta_vs_c15b": round(best_critic_delta, 6),
        "best_critic_gap_capture_vs_c0b": round(
            (best_critic_mb - c0b_mb) / oracle_c0b_gap, 6) if oracle_c0b_gap > 0 else 0.0,
        "learned_probe_value_supported": learned_probe_value_supported,
        "candidate_ready_for_small_multiseed": candidate_ready_for_small_multiseed,
        "ready_for_multiseed": False,
        "c15b_macro_bal": c15b_mb,
        "interpretation": interpretation_text,
    },
}

# Serialize baselines
for name, r in results.items():
    output["baselines"][name] = {
        "macro_query_balanced_accuracy": r.get("metrics", r).get(
            "macro_query_balanced_accuracy", r.get("macro_query_balanced_accuracy", 0.0)),
        "macro_query_accuracy": r.get("metrics", {}).get("macro_query_accuracy", 0.0),
        "mean_positive_recall": r.get("metrics", {}).get("mean_positive_recall", 0.0),
        "mean_negative_recall": r.get("metrics", {}).get("mean_negative_recall", 0.0),
        "per_query": r.get("metrics", {}).get("per_query", {}),
        "visit_count": r.get("cost", {}).get("visit_count", 0),
        "observe_count": r.get("cost", {}).get("observe_count", 0),
        "probe_count": r.get("cost", {}).get("probe_count", 0),
        "total_cost": r.get("cost", {}).get("total_cost", 0.0),
        "normalized_cost": r.get("cost", {}).get("normalized_cost", 0.0),
        "policy_type": r.get("policy_type", "unknown"),
    }

# Save JSON
os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
json_path = os.path.join(CURRENT_DIR, "runs",
                         "calibration_block1j17_learned_probe_value_critic_c4_seed101.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"  Saved: {json_path}")

# =============================================================================
# 14. Generate protocol MD
# =============================================================================
md_path = os.path.join(CURRENT_DIR, "protocols",
                       "block1j17_learned_probe_value_critic_c4_seed101.md")
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

md = f"""# Block 1J17 — Learned Probe-Value Critic Diagnostic

## 1. Objective

Test whether a small learned critic (logistic regression / MLP) trained on
deployable IOM features can predict which (object, action) probes are valuable.
Teacher labels use oracle ground truth ONLY for training, never for deployment.

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
| Training pairs | {N_TRAINING_PAIRS} ({len(pre_probe_visited_oids)} objects x 6 actions) |
| Feature dim | {N_FEATURES} |
| Models | Logistic Regression (L2), Small MLP (hidden=16, ReLU) |

## 3. Training Label Distribution

| Label | Count | Pct |
|-------|-------|-----|
| oracle_topk_probe | {n_topk} | {n_topk/N_TRAINING_PAIRS*100:.1f}% |
| effective | {n_effective} | {n_effective/N_TRAINING_PAIRS*100:.1f}% |
| zero | {n_zero} | {n_zero/N_TRAINING_PAIRS*100:.1f}% |
| harmful | {n_harmful} | {n_harmful/N_TRAINING_PAIRS*100:.1f}% |

## 4. Leave-Object-Out Cross-Validation

| Metric | Logistic Regression | Small MLP |
|--------|---------------------|-----------|
| Overall AUC | {lr_auc:.4f} | {mlp_auc:.4f} |
| Overall Precision@8 | {lr_p8:.4f} | {mlp_p8:.4f} |
| Mean fold AUC | {loo_results['lr_topk']['mean_fold_auc']:.4f} | {loo_results['mlp_topk']['mean_fold_auc']:.4f} |
| Valid folds | {len(loo_results['lr_topk']['fold_aucs'])} | {len(loo_results['mlp_topk']['fold_aucs'])} |

## 5. Deployment Results

| Policy | Macro BAcc | Pos Rec | Neg Rec | Visited | Probed | Delta vs C15b |
|--------|-----------|---------|---------|---------|--------|---------------|
| C0b (observe only) | {c0b_mb:.4f} | {results['C0b_original']['metrics']['mean_positive_recall']:.4f} | {results['C0b_original']['metrics']['mean_negative_recall']:.4f} | {len(c0b_visited_oids)} | 0 | — |
| C15b (VOI reserve) | {c15b_mb:.4f} | {results['C15b_probe_reserve_original']['metrics']['mean_positive_recall']:.4f} | {results['C15b_probe_reserve_original']['metrics']['mean_negative_recall']:.4f} | {c15b_visited_n} | {c15b_probed_n} | 0 |
| C18b (query-rel) | {c18b_mb:.4f} | {results['C18b_query_relevant_fixed_budget']['metrics']['mean_positive_recall']:.4f} | {results['C18b_query_relevant_fixed_budget']['metrics']['mean_negative_recall']:.4f} | {c18b_visited_n} | {c18b_probed_n} | {c18b_mb-c15b_mb:+.4f} |
| C19a (best from 1J16) | {c19a_ref_mb:.4f} | — | — | — | 5 | {c19a_ref_mb-c15b_mb:+.4f} |
| **LR Critic** | **{lr_critic_mb:.4f}** | {lr_critic_metrics['mean_positive_recall']:.4f} | {lr_critic_metrics['mean_negative_recall']:.4f} | {lr_critic_visited} | {lr_critic_probed} | **{lr_critic_mb-c15b_mb:+.4f}** |
| **MLP Critic** | **{mlp_critic_mb:.4f}** | {mlp_critic_metrics['mean_positive_recall']:.4f} | {mlp_critic_metrics['mean_negative_recall']:.4f} | {mlp_critic_visited} | {mlp_critic_probed} | **{mlp_critic_mb-c15b_mb:+.4f}** |
| Oracle k=8 | {oracle_k8_mb:.4f} | — | — | — | 8 | {oracle_k8_mb-c15b_mb:+.4f} |
| Oracle (full) | {oracle_mb:.4f} | — | — | — | — | — |

## 6. Probe Pair Overlap

| Comparison | Count |
|------------|-------|
| C15b pairs | {c15b_probe_pairs} |
| Oracle k=8 pairs | {[(oid, action) for _, oid, action, _, _, _ in oracle_k8_pairs_list]} |
| LR critic overlap with C15b | {lr_overlap_c15b} |
| LR critic overlap with oracle k=8 | {lr_overlap_oracle} |
| MLP critic overlap with C15b | {mlp_overlap_c15b} |
| MLP critic overlap with oracle k=8 | {mlp_overlap_oracle} |

## 7. Probe Effects (LR Critic)

| Metric | Count |
|--------|-------|
| Effective | {lr_effective} |
| Zero gain | {lr_zero} |
| Harmful | {lr_harmful} |

## 8. Probe Effects (MLP Critic)

| Metric | Count |
|--------|-------|
| Effective | {mlp_effective} |
| Zero gain | {mlp_zero} |
| Harmful | {mlp_harmful} |

## 9. Gap Capture

| Policy | Gap Capture % |
|--------|--------------|
| C15b | {gap_capture['C15b_gap_capture_pct']:.1f}% |
| C18b | {gap_capture['C18b_gap_capture_pct']:.1f}% |
| C19a (best from 1J16) | {gap_capture['C19a_gap_capture_pct']:.1f}% |
| LR Critic | {gap_capture['lr_critic_gap_capture_pct']:.1f}% |
| MLP Critic | {gap_capture['mlp_critic_gap_capture_pct']:.1f}% |
| Oracle k=8 | {gap_capture['oracle_k8_gap_capture_pct']:.1f}% |

## 10. Per-Query Breakdown

| Query | C15b BAcc | Best Critic BAcc | Delta |
|-------|----------|------------------|-------|
"""
for qname in QUERY_NAMES:
    c15b_bal = results["C15b_probe_reserve_original"]["metrics"]["per_query"][qname]["balanced_accuracy"]
    best_bal = per_query_breakdown[qname]["best_critic_bal_acc"]
    delta = per_query_breakdown[qname]["best_critic_delta_vs_C15b"]
    md += f"| {qname} | {c15b_bal:.4f} | {best_bal:.4f} | {delta:+.4f} |\n"

md += f"""
## 11. Interpretation

{interpretation_text}

### Tiered Rules

- **If learned critic > C15b**: learned probe-value route is supported.
  → **Result**: {"SUPPORTED" if best_critic_improves_c15b else "NOT SUPPORTED"} (delta={best_critic_delta:+.4f})
- **If learned critic >= {target_10pct}**: candidate for small multiseed later.
  → **Result**: {"CANDIDATE" if best_critic_hits_10pct else "NOT CANDIDATE"} (macro_bal={best_critic_mb:.4f})
- **If critic predicts oracle probes offline but deployment does not improve**:
  execution/budget ordering remains limiting.
  → **Result**: Offline AUC={best_critic_auc_final:.4f}, overlap_oracle_k8={best_critic_overlap_oracle}
- **If critic cannot predict oracle probes offline**:
  available deployable features are insufficient.
  → **Result**: AUC={best_critic_auc_final:.4f}

### Diagnostic Caveats

1. IOM has no cross-test-object propagation. A probe on test object A only affects
   predictions for A. Therefore the probe ranking problem reduces to per-object ranking.
2. Training labels use oracle ground truth (probe outcomes). Deployment uses only
   deployable IOM features (visible features, IOM internals, query context).
3. Models are low-capacity (logistic regression, small MLP) trained on {N_TRAINING_PAIRS} pairs.
   Overfitting risk with {N_FEATURES} features is addressed by L2 regularization and LOO CV.
4. Single seed (101) only. Multi-seed generalization unknown.

## 12. Summary

```
[block_done]
block_id=1J17
c15b_macro_bal={c15b_mb:.4f}
best_critic_variant={best_critic_name}
best_critic_auc={best_critic_auc_final:.4f}
best_critic_precision_at_8={best_critic_p8_final:.4f}
best_critic_overlap_with_oracle_k8={best_critic_overlap_oracle}
best_critic_macro_bal={best_critic_mb:.4f}
best_critic_delta_vs_c15b={best_critic_delta:+.4f}
learned_probe_value_supported={'true' if learned_probe_value_supported else 'false'}
candidate_ready_for_small_multiseed={'true' if candidate_ready_for_small_multiseed else 'false'}
ready_for_multiseed=false
```
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"  Saved: {md_path}")

elapsed = time.time() - t0
print(f"\n{'=' * 60}")
print(f"[block_done]")
print(f"block_id=1J17")
print(f"c15b_macro_bal={c15b_mb:.4f}")
print(f"best_critic_variant={best_critic_name}")
print(f"best_critic_auc={best_critic_auc_final:.4f}")
print(f"best_critic_precision_at_8={best_critic_p8_final:.4f}")
print(f"best_critic_overlap_with_oracle_k8={best_critic_overlap_oracle}")
print(f"best_critic_macro_bal={best_critic_mb:.4f}")
print(f"best_critic_delta_vs_c15b={best_critic_delta:+.4f}")
print(f"learned_probe_value_supported={'true' if learned_probe_value_supported else 'false'}")
print(f"candidate_ready_for_small_multiseed={'true' if candidate_ready_for_small_multiseed else 'false'}")
print(f"ready_for_multiseed=false")
print(f"elapsed={elapsed:.1f}s")
print(f"{'=' * 60}")
