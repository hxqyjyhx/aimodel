"""Block 1H1: Re-score Existing Policies Under Balanced Metrics.

Re-runs the policies from Blocks 1F1-1F3 using the balanced evaluation
framework from Block 1H0. Primary metric: macro_query_balanced_accuracy.
"""
import sys, os, json, time, copy, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

# ---- Import paths ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_5L = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
PARENT_5A3 = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
PARENT_5K = os.path.join(CURRENT_DIR, "..", "exp004_5k_episodic_sparse_outcome")
sys.path.insert(0, PARENT_5L)
sys.path.insert(0, PARENT_5A3)
sys.path.insert(0, PARENT_5K)

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from evaluator import MiniMCEvaluator
from policies import (
    C0b_ObserveOnlyPolicy,
    C13_InstanceVOIPolicy,
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
    _compute_utility,
)
from simulator_truth import MiniMCSimulatorTruth
from instance_outcome_memory import InstanceOutcomeMemory
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
budget = 1.5
cw = 0.5
BETA = 0.25
DISTANCE_THRESHOLD = 0.094829  # from Block 1F0c
QUERY_NAMES = ["need_planks", "need_stone", "need_food", "need_tool", "need_fuel"]

print("=" * 70)
print("Block 1H1: Re-score Existing Policies Under Balanced Metrics")
print(f"  cue=C3_P060_O040, budget={budget}, CW={cw}, seed={seed}")
print(f"  beta={BETA}, distance_threshold={DISTANCE_THRESHOLD}")
print("=" * 70)


# ============================================================
# Helpers
# ============================================================
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


def _task_correct_count(probs, gt):
    correct = 0
    for qname, qfn in _TASK_QUERIES.items():
        if _check_query(probs, qname) == qfn(gt):
            correct += 1
    return correct


def _mean(xs):
    return sum(xs) / max(len(xs), 1)


# ============================================================
# Balanced evaluation framework (from Block 1H0)
# ============================================================
def compute_balanced_metrics(gt_labels, preds_by_oid_query, query_names, test_oids):
    n_total_pairs = len(test_oids) * len(query_names)
    per_query = {}
    for qname in query_names:
        per_query[qname] = {
            "correct": 0, "total": 0,
            "pos_correct": 0, "pos_total": 0,
            "neg_correct": 0, "neg_total": 0,
        }
    raw_correct = 0
    for oid in test_oids:
        for qname in query_names:
            true_label = gt_labels[oid][qname]
            pred_label = preds_by_oid_query[oid][qname]
            if pred_label == true_label:
                raw_correct += 1
                per_query[qname]["correct"] += 1
            per_query[qname]["total"] += 1
            if true_label:
                per_query[qname]["pos_total"] += 1
                if pred_label:
                    per_query[qname]["pos_correct"] += 1
            else:
                per_query[qname]["neg_total"] += 1
                if not pred_label:
                    per_query[qname]["neg_correct"] += 1

    raw_comp_acc = raw_correct / max(n_total_pairs, 1)
    per_query_acc = {}
    per_query_bal_acc = {}
    per_query_pos_recall = {}
    per_query_neg_recall = {}
    per_query_class_coverage = {}
    for qname in query_names:
        pq = per_query[qname]
        pos_total = pq["pos_total"]
        neg_total = pq["neg_total"]
        pos_recall = pq["pos_correct"] / max(pos_total, 1)
        neg_recall = pq["neg_correct"] / max(neg_total, 1)
        per_query_acc[qname] = pq["correct"] / max(pq["total"], 1)
        per_query_pos_recall[qname] = pos_recall
        per_query_neg_recall[qname] = neg_recall
        has_pos = pos_total > 0
        has_neg = neg_total > 0
        per_query_class_coverage[qname] = {
            "n_positive": pos_total, "n_negative": neg_total,
            "has_both_classes": has_pos and has_neg,
        }
        if has_pos and has_neg:
            per_query_bal_acc[qname] = 0.5 * (pos_recall + neg_recall)
        elif has_pos:
            per_query_bal_acc[qname] = pos_recall
        elif has_neg:
            per_query_bal_acc[qname] = neg_recall
        else:
            per_query_bal_acc[qname] = 0.0
    macro_query_acc = sum(per_query_acc.values()) / len(query_names)
    macro_query_bal_acc = sum(per_query_bal_acc.values()) / len(query_names)
    return {
        "raw_comp_acc": raw_comp_acc,
        "macro_query_acc": macro_query_acc,
        "macro_query_bal_acc": macro_query_bal_acc,
        "per_query": {
            qname: {
                "accuracy": per_query_acc[qname],
                "balanced_accuracy": per_query_bal_acc[qname],
                "positive_recall": per_query_pos_recall[qname],
                "negative_recall": per_query_neg_recall[qname],
                "class_coverage": per_query_class_coverage[qname],
            }
            for qname in query_names
        },
    }


# ============================================================
# ProbeLabelingTracker (shared — wraps any policy for GT-based labeling)
# ============================================================
class ProbeLabelingTracker:
    """Wraps a policy to capture per-probe before/after IOM state for GT labeling.
    GT is ONLY used for post-hoc diagnostic labeling — NOT as filter input."""

    def __init__(self, inner_policy, im, gt_per_object):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self.probe_records = []

    def reset(self, view):
        self._inner.reset(view)

    def select_next_object(self, view):
        return self._inner.select_next_object(view)

    def get_answer(self, view):
        return self._inner.get_answer(view)

    def get_pre_probe_entropies(self):
        return getattr(self._inner, 'get_pre_probe_entropies', lambda: {})()

    def get_pre_decision_entropies(self):
        return getattr(self._inner, 'get_pre_decision_entropies', lambda: {})()

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}
        before_probs = self._im.predict_all_affordances(fake_obj)
        before_correct = _task_correct_count(before_probs, gt)

        should_probe, action = self._inner.decide_probe(view, object_id)

        if should_probe and action is not None:
            p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
            self._pending_label = {
                "object_id": object_id, "action": action, "gt": gt,
                "before_correct": before_correct,
                "distance_to_threshold": abs(p_success - 0.5),
                "p_success": p_success,
            }
        else:
            self._pending_label = None
        return should_probe, action

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)
        pending = getattr(self, "_pending_label", None)
        if pending is not None and pending["object_id"] == object_id:
            gt = pending["gt"]
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            after_correct = _task_correct_count(after_probs, gt)
            actual_delta = after_correct - pending["before_correct"]
            self.probe_records.append({
                "object_id": object_id, "action": action, "outcome": outcome,
                "before_correct": pending["before_correct"],
                "after_correct": after_correct,
                "actual_delta_task_correct": actual_delta,
                "is_effective": actual_delta > 0,
                "is_zero_gain": actual_delta == 0,
                "is_harmful": actual_delta < 0,
                "distance_to_threshold": pending["distance_to_threshold"],
                "p_success": pending["p_success"],
            })
        self._pending_label = None


# ============================================================
# C13DistanceFilterPolicy (from Block 1F1)
# ============================================================
class C13DistanceFilterPolicy:
    """Wraps C13. After C13 selects a candidate probe action, only executes
    the probe if distance_to_threshold <= fixed threshold."""

    def __init__(self, inner_policy, im, gt_per_object, threshold):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self._threshold = threshold
        self.probe_records = []
        self.filter_stats = {
            "n_candidates_seen": 0, "n_kept": 0, "n_skipped_by_distance": 0,
            "kept_distances": [], "skipped_distances": [],
            "kept_actions": [], "skipped_actions": [],
            "skipped_candidate_records": [],
        }

    def reset(self, view):
        self._inner.reset(view)

    def select_next_object(self, view):
        return self._inner.select_next_object(view)

    def get_answer(self, view):
        return self._inner.get_answer(view)

    def get_pre_probe_entropies(self):
        return getattr(self._inner, 'get_pre_probe_entropies', lambda: {})()

    def get_pre_decision_entropies(self):
        return getattr(self._inner, 'get_pre_decision_entropies', lambda: {})()

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}
        before_probs = self._im.predict_all_affordances(fake_obj)
        before_correct = _task_correct_count(before_probs, gt)

        should_probe, action = self._inner.decide_probe(view, object_id)
        if not should_probe or action is None:
            self._pending_label = None
            return False, None

        p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
        dist = abs(p_success - 0.5)
        keep = dist <= self._threshold
        self.filter_stats["n_candidates_seen"] += 1

        if keep:
            self.filter_stats["n_kept"] += 1
            self.filter_stats["kept_distances"].append(dist)
            self.filter_stats["kept_actions"].append(action)
            self._pending_label = {
                "object_id": object_id, "action": action, "gt": gt,
                "before_correct": before_correct, "distance_to_threshold": dist,
                "p_success": p_success, "kept": True,
            }
            return True, action
        else:
            self.filter_stats["n_skipped_by_distance"] += 1
            self.filter_stats["skipped_distances"].append(dist)
            self.filter_stats["skipped_actions"].append(action)
            self.filter_stats["skipped_candidate_records"].append({
                "object_id": object_id, "action": action,
                "distance_to_threshold": round(dist, 6),
                "p_success": round(p_success, 6),
                "threshold": self._threshold,
            })
            self._pending_label = None
            return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)
        pending = getattr(self, "_pending_label", None)
        if pending is not None and pending["object_id"] == object_id:
            gt = pending["gt"]
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            after_correct = _task_correct_count(after_probs, gt)
            actual_delta = after_correct - pending["before_correct"]
            self.probe_records.append({
                "object_id": object_id, "action": action, "outcome": outcome,
                "before_correct": pending["before_correct"],
                "after_correct": after_correct,
                "actual_delta_task_correct": actual_delta,
                "is_effective": actual_delta > 0,
                "is_zero_gain": actual_delta == 0,
                "is_harmful": actual_delta < 0,
                "distance_to_threshold": pending["distance_to_threshold"],
                "p_success": pending["p_success"],
            })
        self._pending_label = None


# ============================================================
# C13DistanceAwareSelectionPolicy (from Block 1F2)
# ============================================================
class C13DistanceAwareSelectionPolicy:
    """Distance-aware object selection + distance-gated probe decisions."""

    def __init__(self, inner_policy, im, gt_per_object, threshold, cost_weight):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self._threshold = threshold
        self._cost_weight = cost_weight
        self.probe_records = []
        self.selection_stats = {
            "n_objects_considered": 0,
            "n_objects_rejected_no_eligible_action": 0,
            "n_objects_selected_with_eligible_action": 0,
        }
        self.filter_stats = {
            "n_candidates_seen": 0, "n_kept": 0, "n_skipped_by_distance": 0,
            "kept_distances": [], "skipped_distances": [],
            "kept_actions": [], "skipped_actions": [],
            "skipped_candidate_records": [],
        }

    def reset(self, view):
        self._inner.reset(view)

    def get_answer(self, view):
        return self._inner.get_answer(view)

    def get_pre_probe_entropies(self):
        return getattr(self._inner, 'get_pre_probe_entropies', lambda: {})()

    def get_pre_decision_entropies(self):
        return getattr(self._inner, 'get_pre_decision_entropies', lambda: {})()

    def _normalized_step_cost(self, oid, view):
        raw = (view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost)
        return raw / max(view.initial_budget, 0.001)

    def _get_eligible_actions(self, probs):
        eligible = []
        for action in MAIN_CANDIDATE_ACTIONS:
            feat = ACTION_TO_FEATURE[action]
            p = probs.get(feat, 0.5)
            if abs(p - 0.5) <= self._threshold:
                eligible.append((action, p))
        return eligible

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        best_oid = None
        best_score = float('-inf')
        for oid in unvisited:
            total_cost = (view.compute_reach_cost(oid) + view.observe_cost
                          + view.probe_cost)
            if not view.can_afford(total_cost):
                continue
            self.selection_stats["n_objects_considered"] += 1
            fake_obj = {"id": oid, "visible_features": {}}
            probs = self._im.predict_all_affordances(fake_obj)
            eligible = self._get_eligible_actions(probs)
            if not eligible:
                self.selection_stats["n_objects_rejected_no_eligible_action"] += 1
                continue
            self.selection_stats["n_objects_selected_with_eligible_action"] += 1
            current_utility = _compute_utility(probs)
            norm_cost = self._normalized_step_cost(oid, view)
            best_net_voi = 0.0
            for action, p_success in eligible:
                im_succ = self._im.clone()
                im_succ.incorporate_probe(oid, action, 1.0)
                utility_succ = _compute_utility(
                    im_succ.predict_all_affordances(fake_obj))
                im_fail = self._im.clone()
                im_fail.incorporate_probe(oid, action, 0.0)
                utility_fail = _compute_utility(
                    im_fail.predict_all_affordances(fake_obj))
                e_post = p_success * utility_succ + (1 - p_success) * utility_fail
                net_voi = e_post - current_utility - self._cost_weight * norm_cost
                if net_voi > best_net_voi:
                    best_net_voi = net_voi
            if best_net_voi > best_score:
                best_score = best_net_voi
                best_oid = oid
        if best_score <= 0:
            return None
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}
        before_probs = self._im.predict_all_affordances(fake_obj)
        before_correct = _task_correct_count(before_probs, gt)
        should_probe, action = self._inner.decide_probe(view, object_id)
        if not should_probe or action is None:
            self._pending_label = None
            return False, None
        p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
        dist = abs(p_success - 0.5)
        keep = dist <= self._threshold
        self.filter_stats["n_candidates_seen"] += 1
        if keep:
            self.filter_stats["n_kept"] += 1
            self.filter_stats["kept_distances"].append(dist)
            self.filter_stats["kept_actions"].append(action)
            self._pending_label = {
                "object_id": object_id, "action": action, "gt": gt,
                "before_correct": before_correct, "distance_to_threshold": dist,
                "p_success": p_success, "kept": True,
            }
            return True, action
        else:
            self.filter_stats["n_skipped_by_distance"] += 1
            self.filter_stats["skipped_distances"].append(dist)
            self.filter_stats["skipped_actions"].append(action)
            self.filter_stats["skipped_candidate_records"].append({
                "object_id": object_id, "action": action,
                "distance_to_threshold": round(dist, 6),
                "p_success": round(p_success, 6),
                "threshold": self._threshold,
            })
            self._pending_label = None
            return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)
        pending = getattr(self, "_pending_label", None)
        if pending is not None and pending["object_id"] == object_id:
            gt = pending["gt"]
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            after_correct = _task_correct_count(after_probs, gt)
            actual_delta = after_correct - pending["before_correct"]
            self.probe_records.append({
                "object_id": object_id, "action": action, "outcome": outcome,
                "before_correct": pending["before_correct"],
                "after_correct": after_correct,
                "actual_delta_task_correct": actual_delta,
                "is_effective": actual_delta > 0,
                "is_zero_gain": actual_delta == 0,
                "is_harmful": actual_delta < 0,
                "distance_to_threshold": pending["distance_to_threshold"],
                "p_success": pending["p_success"],
            })
        self._pending_label = None


# ============================================================
# C13DistanceFilterStopOnSkip (from Block 1F3)
# ============================================================
class C13DistanceFilterStopOnSkip:
    """Distance-gated probe filter that STOPS the episode on first distance skip."""

    def __init__(self, inner_policy, im, gt_per_object, threshold):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self._threshold = threshold
        self.probe_records = []
        self._stopped = False
        self.stop_info = {}
        self.filter_stats = {
            "n_candidates_seen": 0, "n_kept": 0, "n_skipped_by_distance": 0,
            "kept_distances": [], "skipped_distances": [],
            "kept_actions": [], "skipped_actions": [],
            "skipped_candidate_records": [],
        }

    def reset(self, view):
        self._inner.reset(view)
        self._stopped = False
        self.stop_info = {}

    def select_next_object(self, view):
        if self._stopped:
            return None
        return self._inner.select_next_object(view)

    def get_answer(self, view):
        return self._inner.get_answer(view)

    def get_pre_probe_entropies(self):
        return getattr(self._inner, 'get_pre_probe_entropies', lambda: {})()

    def get_pre_decision_entropies(self):
        return getattr(self._inner, 'get_pre_decision_entropies', lambda: {})()

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}
        before_probs = self._im.predict_all_affordances(fake_obj)
        before_correct = _task_correct_count(before_probs, gt)

        should_probe, action = self._inner.decide_probe(view, object_id)
        if not should_probe or action is None:
            self._pending_label = None
            return False, None

        p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
        dist = abs(p_success - 0.5)
        keep = dist <= self._threshold
        self.filter_stats["n_candidates_seen"] += 1

        if keep:
            self.filter_stats["n_kept"] += 1
            self.filter_stats["kept_distances"].append(dist)
            self.filter_stats["kept_actions"].append(action)
            self._pending_label = {
                "object_id": object_id, "action": action, "gt": gt,
                "before_correct": before_correct, "distance_to_threshold": dist,
                "p_success": p_success, "kept": True,
            }
            return True, action
        else:
            self.filter_stats["n_skipped_by_distance"] += 1
            self.filter_stats["skipped_distances"].append(dist)
            self.filter_stats["skipped_actions"].append(action)
            self.filter_stats["skipped_candidate_records"].append({
                "object_id": object_id, "action": action,
                "distance_to_threshold": round(dist, 6),
                "p_success": round(p_success, 6),
                "threshold": self._threshold,
            })
            self._stopped = True
            self.stop_info = {
                "stopped_due_to_distance_skip": True,
                "stop_object_id": object_id,
                "stop_action": action,
                "stop_distance": round(dist, 6),
                "stop_p_success": round(p_success, 6),
                "budget_remaining_at_stop": round(view.budget_remaining, 6),
            }
            self._pending_label = None
            return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)
        pending = getattr(self, "_pending_label", None)
        if pending is not None and pending["object_id"] == object_id:
            gt = pending["gt"]
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            after_correct = _task_correct_count(after_probs, gt)
            actual_delta = after_correct - pending["before_correct"]
            self.probe_records.append({
                "object_id": object_id, "action": action, "outcome": outcome,
                "before_correct": pending["before_correct"],
                "after_correct": after_correct,
                "actual_delta_task_correct": actual_delta,
                "is_effective": actual_delta > 0,
                "is_zero_gain": actual_delta == 0,
                "is_harmful": actual_delta < 0,
                "distance_to_threshold": pending["distance_to_threshold"],
                "p_success": pending["p_success"],
            })
        self._pending_label = None


# ============================================================
# Phase A: Training
# ============================================================
print("\n  Phase A: Training (C3_P060_O040)...")
from run_004_5n1 import run_phase_a_training

cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60,
                      "p_other": 0.40, "absent": False})
(student, base_learner, train_objects, train_env,
 test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
test_oids = sorted(test_objects)
test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}
n_test = len(test_oids)
print(f"  train={len(train_objects_dict)}, test={n_test}")

# ---- Build IOM ----
outcome_rows, _ = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, seed)
posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)

# ---- Positions + GT ----
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)
_temp_sim = MiniMCSimulatorTruth(test_objects_dict, positions, config.AGENT_START)
gt_per_object = {oid: _temp_sim.get_ground_truth_affordances(oid) for oid in test_oids}

# ---- GT labels for balanced metrics ----
gt_labels = {}
for oid in test_oids:
    gt_labels[oid] = {}
    for qname in QUERY_NAMES:
        gt_labels[oid][qname] = _TASK_QUERIES[qname](gt_per_object[oid])


def run_policy(policy, env):
    """Run a policy through harness and return metrics + result."""
    result = EpisodeHarness(env, policy).run()
    evaluator = MiniMCEvaluator(env)
    metrics = evaluator.evaluate(
        result.predictions, result.event_log, result.agent_obs,
        result.pre_probe_entropies, result.pre_decision_entropies,
    )
    return metrics, result


def probe_summary(records):
    return {
        "n_probe": len(records),
        "n_effective": sum(1 for r in records if r.get("is_effective", False)),
        "n_zero_gain": sum(1 for r in records if r.get("is_zero_gain", False)),
        "n_harmful": sum(1 for r in records if r.get("is_harmful", False)),
        "actual_delta_sum": sum(r.get("actual_delta_task_correct", 0) for r in records),
    }


# ============================================================
# Policy 1: all_false (static)
# ============================================================
print("\n  Computing all_false...")
all_false_preds = {oid: {q: False for q in QUERY_NAMES} for oid in test_oids}
all_false_cost = {"objects_visited": 0, "total_cost": 0.0, "normalized_cost": 0.0,
                  "budget_remaining": budget}
all_false_probe = {"n_probe": 0, "n_effective": 0, "n_zero_gain": 0,
                   "n_harmful": 0, "actual_delta_sum": 0}

# ============================================================
# Policy 2: IOM_prior_empty_features (static)
# ============================================================
print("  Computing IOM_prior_empty_features...")
im_prior_static = im_base.clone()
iom_prior_preds = {}
for oid in test_oids:
    fake_obj = {"id": oid, "visible_features": {}}
    probs = im_prior_static.predict_all_affordances(fake_obj)
    iom_prior_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
iom_prior_cost = {"objects_visited": 0, "total_cost": 0.0, "normalized_cost": 0.0,
                  "budget_remaining": budget}
iom_prior_probe = {"n_probe": 0, "n_effective": 0, "n_zero_gain": 0,
                   "n_harmful": 0, "actual_delta_sum": 0}

# ============================================================
# Policy 3: C0b / C_obs_full (harness)
# ============================================================
print("  Running C0b / C_obs_full...")
env_c0b = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c0b = im_base.clone()
rng_c0b = random.Random(seed + 100)
c0b_inner = C0b_ObserveOnlyPolicy(im_c0b, rng_c0b)
c0b_policy = ProbeLabelingTracker(c0b_inner, im_c0b, gt_per_object)
c0b_metrics, c0b_result = run_policy(c0b_policy, env_c0b)
c0b_per_object = c0b_result.predictions.get("per_object", {})
c0b_preds = {}
for oid in test_oids:
    probs = c0b_per_object.get(oid, {})
    c0b_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
c0b_ps = probe_summary(c0b_policy.probe_records)

# ============================================================
# Policy 4: original C13 (harness)
# ============================================================
print("  Running original C13 (CW=0.5)...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
rng_c13 = random.Random(seed + 550)
c13_inner = C13_InstanceVOIPolicy(im_c13, rng_c13, cost_weight=cw)
c13_policy = ProbeLabelingTracker(c13_inner, im_c13, gt_per_object)
c13_metrics, c13_result = run_policy(c13_policy, env_c13)
c13_per_object = c13_result.predictions.get("per_object", {})
c13_preds = {}
for oid in test_oids:
    probs = c13_per_object.get(oid, {})
    c13_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
c13_ps = probe_summary(c13_policy.probe_records)

# ============================================================
# Policy 5: C13_distance_filter (harness)
# ============================================================
print(f"  Running C13_distance_filter (threshold={DISTANCE_THRESHOLD})...")
env_filt = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_filt = im_base.clone()
rng_filt = random.Random(seed + 550)
filt_inner = C13_InstanceVOIPolicy(im_filt, rng_filt, cost_weight=cw)
filt_policy = C13DistanceFilterPolicy(filt_inner, im_filt, gt_per_object, DISTANCE_THRESHOLD)
filt_metrics, filt_result = run_policy(filt_policy, env_filt)
filt_per_object = filt_result.predictions.get("per_object", {})
filt_preds = {}
for oid in test_oids:
    probs = filt_per_object.get(oid, {})
    filt_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
filt_ps = probe_summary(filt_policy.probe_records)

# ============================================================
# Policy 6: C13_distance_aware_selection (harness)
# ============================================================
print(f"  Running C13_distance_aware_selection (threshold={DISTANCE_THRESHOLD})...")
env_aware = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_aware = im_base.clone()
rng_aware = random.Random(seed + 550)
aware_inner = C13_InstanceVOIPolicy(im_aware, rng_aware, cost_weight=cw)
aware_policy = C13DistanceAwareSelectionPolicy(
    aware_inner, im_aware, gt_per_object, DISTANCE_THRESHOLD, cw)
aware_metrics, aware_result = run_policy(aware_policy, env_aware)
aware_per_object = aware_result.predictions.get("per_object", {})
aware_preds = {}
for oid in test_oids:
    probs = aware_per_object.get(oid, {})
    aware_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
aware_ps = probe_summary(aware_policy.probe_records)

# ============================================================
# Policy 7: C13_distance_filter_stop_on_skip (harness)
# ============================================================
print(f"  Running C13_distance_filter_stop_on_skip (threshold={DISTANCE_THRESHOLD})...")
env_stop = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_stop = im_base.clone()
rng_stop = random.Random(seed + 550)
stop_inner = C13_InstanceVOIPolicy(im_stop, rng_stop, cost_weight=cw)
stop_policy = C13DistanceFilterStopOnSkip(stop_inner, im_stop, gt_per_object, DISTANCE_THRESHOLD)
stop_metrics, stop_result = run_policy(stop_policy, env_stop)
stop_per_object = stop_result.predictions.get("per_object", {})
stop_preds = {}
for oid in test_oids:
    probs = stop_per_object.get(oid, {})
    stop_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
stop_ps = probe_summary(stop_policy.probe_records)

# ============================================================
# Policy 8: oracle_unconstrained (static)
# ============================================================
print("  Computing oracle_unconstrained...")
oracle_preds = {oid: {q: gt_labels[oid][q] for q in QUERY_NAMES} for oid in test_oids}
oracle_cost = {"objects_visited": 0, "total_cost": 0.0, "normalized_cost": 0.0,
               "budget_remaining": budget}
oracle_probe = {"n_probe": 0, "n_effective": 0, "n_zero_gain": 0,
                "n_harmful": 0, "actual_delta_sum": 0}

# ============================================================
# Compute balanced metrics for all 8 policies
# ============================================================
policy_names = [
    "all_false",
    "iom_prior",
    "c0b",
    "c13",
    "c13_distance_filter",
    "c13_distance_aware_selection",
    "c13_distance_filter_stop_on_skip",
    "oracle_unconstrained",
]

policy_preds = {
    "all_false": all_false_preds,
    "iom_prior": iom_prior_preds,
    "c0b": c0b_preds,
    "c13": c13_preds,
    "c13_distance_filter": filt_preds,
    "c13_distance_aware_selection": aware_preds,
    "c13_distance_filter_stop_on_skip": stop_preds,
    "oracle_unconstrained": oracle_preds,
}

policy_costs = {
    "all_false": all_false_cost,
    "iom_prior": iom_prior_cost,
    "c0b": {"objects_visited": c0b_metrics["objects_visited"],
            "total_cost": c0b_metrics["total_cost"],
            "normalized_cost": c0b_metrics["normalized_cost"],
            "budget_remaining": c0b_metrics["budget_remaining"]},
    "c13": {"objects_visited": c13_metrics["objects_visited"],
            "total_cost": c13_metrics["total_cost"],
            "normalized_cost": c13_metrics["normalized_cost"],
            "budget_remaining": c13_metrics["budget_remaining"]},
    "c13_distance_filter": {"objects_visited": filt_metrics["objects_visited"],
                            "total_cost": filt_metrics["total_cost"],
                            "normalized_cost": filt_metrics["normalized_cost"],
                            "budget_remaining": filt_metrics["budget_remaining"]},
    "c13_distance_aware_selection": {"objects_visited": aware_metrics["objects_visited"],
                                      "total_cost": aware_metrics["total_cost"],
                                      "normalized_cost": aware_metrics["normalized_cost"],
                                      "budget_remaining": aware_metrics["budget_remaining"]},
    "c13_distance_filter_stop_on_skip": {"objects_visited": stop_metrics["objects_visited"],
                                          "total_cost": stop_metrics["total_cost"],
                                          "normalized_cost": stop_metrics["normalized_cost"],
                                          "budget_remaining": stop_metrics["budget_remaining"]},
    "oracle_unconstrained": oracle_cost,
}

policy_probes = {
    "all_false": all_false_probe,
    "iom_prior": iom_prior_probe,
    "c0b": c0b_ps,
    "c13": c13_ps,
    "c13_distance_filter": filt_ps,
    "c13_distance_aware_selection": aware_ps,
    "c13_distance_filter_stop_on_skip": stop_ps,
    "oracle_unconstrained": oracle_probe,
}

all_metrics = {}
for pname in policy_names:
    all_metrics[pname] = compute_balanced_metrics(
        gt_labels, policy_preds[pname], QUERY_NAMES, test_oids)

# ============================================================
# Compute utility scores
# ============================================================
policy_results = {}
for pname in policy_names:
    m = all_metrics[pname]
    c = policy_costs[pname]
    p = policy_probes[pname]
    raw_comp = m["raw_comp_acc"]
    macro_bal = m["macro_query_bal_acc"]
    ncost = c["normalized_cost"]
    raw_ub25 = raw_comp - BETA * ncost
    bal_ub25 = macro_bal - BETA * ncost
    policy_results[pname] = {
        "raw_comp_acc": raw_comp,
        "macro_query_acc": m["macro_query_acc"],
        "macro_query_bal_acc": macro_bal,
        "normalized_cost": ncost,
        "raw_utility_beta0.25": raw_ub25,
        "balanced_utility_beta0.25": bal_ub25,
        "visited_count": c["objects_visited"],
        "probe_count": p["n_probe"],
        "effective_probe_count": p["n_effective"],
        "zero_gain_probe_count": p["n_zero_gain"],
        "harmful_probe_count": p["n_harmful"],
        "actual_probe_delta_sum": p["actual_delta_sum"],
        "total_cost": c["total_cost"],
        "budget_remaining": c["budget_remaining"],
        "per_query_accuracy": {q: m["per_query"][q]["accuracy"] for q in QUERY_NAMES},
        "per_query_balanced_accuracy": {q: m["per_query"][q]["balanced_accuracy"]
                                         for q in QUERY_NAMES},
        "per_query_positive_recall": {q: m["per_query"][q]["positive_recall"]
                                       for q in QUERY_NAMES},
        "per_query_negative_recall": {q: m["per_query"][q]["negative_recall"]
                                       for q in QUERY_NAMES},
    }

# ============================================================
# Print results
# ============================================================

# ---- Table 1: Policy summary ----
print(f"\n{'='*70}")
print(f"Policy Summary: Raw vs Balanced Metrics")
print(f"{'='*70}")
header = (f"  {'policy':<32} {'raw_comp':>9} {'macro_bal':>9} "
          f"{'ncost':>7} {'raw_ub25':>9} {'bal_ub25':>9} "
          f"{'visited':>7} {'probes':>7} {'eff':>5} {'harm':>5}")
print(header)
print(f"  {'-'*32} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*9} "
      f"{'-'*7} {'-'*7} {'-'*5} {'-'*5}")
for pname in policy_names:
    r = policy_results[pname]
    print(f"  {pname:<32} {r['raw_comp_acc']:>9.4f} {r['macro_query_bal_acc']:>9.4f} "
          f"{r['normalized_cost']:>7.4f} {r['raw_utility_beta0.25']:>9.4f} "
          f"{r['balanced_utility_beta0.25']:>9.4f} "
          f"{r['visited_count']:>7} {r['probe_count']:>7} "
          f"{r['effective_probe_count']:>5} {r['harmful_probe_count']:>5}")

# ---- Table 2: Per-query balanced accuracy ----
print(f"\n{'='*70}")
print(f"Per-Query Balanced Accuracy")
print(f"{'='*70}")
header = f"  {'policy':<32}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
header += f" {'macro_bal':>10}"
print(header)
print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
for pname in policy_names:
    r = policy_results[pname]
    row = f"  {pname:<32}"
    for q in QUERY_NAMES:
        row += f" {r['per_query_balanced_accuracy'][q]:>12.4f}"
    row += f" {r['macro_query_bal_acc']:>10.4f}"
    print(row)

# ---- Table 3: Per-query positive recall ----
print(f"\n{'='*70}")
print(f"Per-Query Positive Recall")
print(f"{'='*70}")
header = f"  {'policy':<32}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
print(header)
print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for pname in policy_names:
    r = policy_results[pname]
    row = f"  {pname:<32}"
    for q in QUERY_NAMES:
        row += f" {r['per_query_positive_recall'][q]:>12.4f}"
    print(row)

# ---- Table 4: Per-query negative recall ----
print(f"\n{'='*70}")
print(f"Per-Query Negative Recall")
print(f"{'='*70}")
header = f"  {'policy':<32}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
print(header)
print(f"  {'-'*32} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for pname in policy_names:
    r = policy_results[pname]
    row = f"  {pname:<32}"
    for q in QUERY_NAMES:
        row += f" {r['per_query_negative_recall'][q]:>12.4f}"
    print(row)

# ---- Key comparisons ----
print(f"\n{'='*70}")
print(f"Key Comparisons: Balanced Evaluation of Existing Policies")
print(f"{'='*70}")

c0b_r = policy_results["c0b"]
c13_r = policy_results["c13"]
filt_r = policy_results["c13_distance_filter"]
aware_r = policy_results["c13_distance_aware_selection"]
stop_r = policy_results["c13_distance_filter_stop_on_skip"]
all_false_r = policy_results["all_false"]
iom_r = policy_results["iom_prior"]
oracle_r = policy_results["oracle_unconstrained"]

print(f"\n  --- C13 vs C0b ---")
print(f"  c0b  raw_comp={c0b_r['raw_comp_acc']:.4f}  macro_bal={c0b_r['macro_query_bal_acc']:.4f}"
      f"  raw_ub25={c0b_r['raw_utility_beta0.25']:.4f}  bal_ub25={c0b_r['balanced_utility_beta0.25']:.4f}")
print(f"  c13  raw_comp={c13_r['raw_comp_acc']:.4f}  macro_bal={c13_r['macro_query_bal_acc']:.4f}"
      f"  raw_ub25={c13_r['raw_utility_beta0.25']:.4f}  bal_ub25={c13_r['balanced_utility_beta0.25']:.4f}")
raw_comp_gain_c13 = c13_r['raw_comp_acc'] - c0b_r['raw_comp_acc']
macro_bal_gain_c13 = c13_r['macro_query_bal_acc'] - c0b_r['macro_query_bal_acc']
raw_ub25_gain_c13 = c13_r['raw_utility_beta0.25'] - c0b_r['raw_utility_beta0.25']
bal_ub25_gain_c13 = c13_r['balanced_utility_beta0.25'] - c0b_r['balanced_utility_beta0.25']
pos_recall_gain_c13 = (sum(c13_r['per_query_positive_recall'].values()) -
                        sum(c0b_r['per_query_positive_recall'].values()))
print(f"  raw_comp gain: {raw_comp_gain_c13:+.4f}")
print(f"  macro_bal gain: {macro_bal_gain_c13:+.4f}")
print(f"  raw_ub25 gain: {raw_ub25_gain_c13:+.4f}")
print(f"  bal_ub25 gain: {bal_ub25_gain_c13:+.4f}")
print(f"  sum(pos_recall) gain: {pos_recall_gain_c13:+.4f}")

print(f"\n  --- C13_distance_filter vs C13 ---")
print(f"  filter  raw_comp={filt_r['raw_comp_acc']:.4f}  macro_bal={filt_r['macro_query_bal_acc']:.4f}"
      f"  raw_ub25={filt_r['raw_utility_beta0.25']:.4f}  bal_ub25={filt_r['balanced_utility_beta0.25']:.4f}")
print(f"  bal_ub25 diff: {filt_r['balanced_utility_beta0.25'] - c13_r['balanced_utility_beta0.25']:+.4f}")
print(f"  macro_bal diff: {filt_r['macro_query_bal_acc'] - c13_r['macro_query_bal_acc']:+.4f}")
print(f"  probe_count diff: {filt_r['probe_count'] - c13_r['probe_count']}")
print(f"  sum(pos_recall) diff: "
      f"{sum(filt_r['per_query_positive_recall'].values()) - sum(c13_r['per_query_positive_recall'].values()):+.4f}")

print(f"\n  --- C13_distance_aware_selection vs C13 ---")
print(f"  aware  raw_comp={aware_r['raw_comp_acc']:.4f}  macro_bal={aware_r['macro_query_bal_acc']:.4f}"
      f"  raw_ub25={aware_r['raw_utility_beta0.25']:.4f}  bal_ub25={aware_r['balanced_utility_beta0.25']:.4f}")

print(f"\n  --- stop_on_skip vs C13 ---")
print(f"  stop   raw_comp={stop_r['raw_comp_acc']:.4f}  macro_bal={stop_r['macro_query_bal_acc']:.4f}"
      f"  raw_ub25={stop_r['raw_utility_beta0.25']:.4f}  bal_ub25={stop_r['balanced_utility_beta0.25']:.4f}")
print(f"  bal_ub25 diff: {stop_r['balanced_utility_beta0.25'] - c13_r['balanced_utility_beta0.25']:+.4f}")
print(f"  macro_bal drop: {stop_r['macro_query_bal_acc'] - c13_r['macro_query_bal_acc']:+.4f}")
print(f"  cost saving: {stop_r['total_cost'] - c13_r['total_cost']:+.4f}")

# ---- Interpretation flags ----
print(f"\n{'='*70}")
print(f"Interpretation")
print(f"{'='*70}")

c13_improves_macro_bal = macro_bal_gain_c13 > 0.01
c13_loses_bal_utility = bal_ub25_gain_c13 < 0.0
c13_no_macro_bal_improvement = macro_bal_gain_c13 <= 0.01
stop_high_bal_ub25_low_macro_bal = (
    stop_r['balanced_utility_beta0.25'] > c13_r['balanced_utility_beta0.25']
    and stop_r['macro_query_bal_acc'] <= 0.52
)
filt_improves_bal_per_probe = (
    filt_r['macro_query_bal_acc'] > c13_r['macro_query_bal_acc']
    and filt_r['probe_count'] < c13_r['probe_count']
)
filt_loses_bal_utility = filt_r['balanced_utility_beta0.25'] < c13_r['balanced_utility_beta0.25']

if c13_improves_macro_bal and c13_loses_bal_utility:
    print(f"  C13: real signal but too costly "
          f"(macro_bal={macro_bal_gain_c13:+.4f}, bal_ub25={bal_ub25_gain_c13:+.4f})")
elif c13_no_macro_bal_improvement:
    print(f"  C13: previous raw-comp gain was majority-class artifact "
          f"(macro_bal gain={macro_bal_gain_c13:+.4f})")
else:
    print(f"  C13: mixed result "
          f"(macro_bal={macro_bal_gain_c13:+.4f}, bal_ub25={bal_ub25_gain_c13:+.4f})")

if stop_high_bal_ub25_low_macro_bal:
    print(f"  stop_on_skip: still metric gaming / low-cost prior exploitation"
          f" (bal_ub25={stop_r['balanced_utility_beta0.25']:.4f},"
          f" macro_bal={stop_r['macro_query_bal_acc']:.4f})")
else:
    print(f"  stop_on_skip: bal_ub25={stop_r['balanced_utility_beta0.25']:.4f},"
          f" macro_bal={stop_r['macro_query_bal_acc']:.4f}")

if filt_improves_bal_per_probe and filt_loses_bal_utility:
    print(f"  distance_filter: improves bal per probe but loses bal utility "
          f"(budget reallocation diagnosis holds)")
elif not filt_improves_bal_per_probe:
    print(f"  distance_filter: does not improve macro_bal per probe")

# ---- Diagnostic checks ----
# Check if stop_on_skip is just metric exploit (high bal_ub25 but macro_bal near 0.5)
stop_is_metric_exploit = (
    stop_r['balanced_utility_beta0.25'] > c0b_r['balanced_utility_beta0.25'] + 0.05
    and stop_r['macro_query_bal_acc'] < c0b_r['macro_query_bal_acc'] + 0.03
)
c13_beats_c0b_balanced = c13_r['balanced_utility_beta0.25'] > c0b_r['balanced_utility_beta0.25']
distance_filter_beats_c13 = filt_r['balanced_utility_beta0.25'] > c13_r['balanced_utility_beta0.25']

print(f"\n  Diagnostic flags:")
print(f"    c13_beats_c0b_balanced:           {c13_beats_c0b_balanced}")
print(f"    distance_filter_beats_c13:         {distance_filter_beats_c13}")
print(f"    stop_on_skip_is_metric_exploit:    {stop_is_metric_exploit}")

# ============================================================
# [block_done]
# ============================================================
print()
print("[block_done]")
print(f"  block_id=1H1")
print(f"  c0b_macro_bal={c0b_r['macro_query_bal_acc']:.4f}")
print(f"  c0b_bal_ub25={c0b_r['balanced_utility_beta0.25']:.4f}")
print(f"  c13_macro_bal={c13_r['macro_query_bal_acc']:.4f}")
print(f"  c13_bal_ub25={c13_r['balanced_utility_beta0.25']:.4f}")
print(f"  distance_filter_macro_bal={filt_r['macro_query_bal_acc']:.4f}")
print(f"  distance_filter_bal_ub25={filt_r['balanced_utility_beta0.25']:.4f}")
print(f"  stop_on_skip_macro_bal={stop_r['macro_query_bal_acc']:.4f}")
print(f"  stop_on_skip_bal_ub25={stop_r['balanced_utility_beta0.25']:.4f}")
print(f"  c13_beats_c0b_balanced={'true' if c13_beats_c0b_balanced else 'false'}")
print(f"  distance_filter_beats_c13_balanced={'true' if distance_filter_beats_c13 else 'false'}")
print(f"  stop_on_skip_is_metric_exploit={'true' if stop_is_metric_exploit else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ============================================================
# Save
# ============================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)

block_out = {
    "block_id": "1H1",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040",
    "budget": budget,
    "cw": cw,
    "seed": seed,
    "beta": BETA,
    "distance_threshold": DISTANCE_THRESHOLD,
    "query_names": QUERY_NAMES,
    "n_test_objects": n_test,
    "policies": {},
    "key_comparisons": {
        "c13_vs_c0b": {
            "raw_comp_gain": raw_comp_gain_c13,
            "macro_bal_gain": macro_bal_gain_c13,
            "raw_ub25_gain": raw_ub25_gain_c13,
            "bal_ub25_gain": bal_ub25_gain_c13,
            "sum_pos_recall_gain": pos_recall_gain_c13,
        },
        "distance_filter_vs_c13": {
            "bal_ub25_diff": filt_r['balanced_utility_beta0.25'] - c13_r['balanced_utility_beta0.25'],
            "macro_bal_diff": filt_r['macro_query_bal_acc'] - c13_r['macro_query_bal_acc'],
            "probe_count_diff": filt_r['probe_count'] - c13_r['probe_count'],
        },
        "stop_on_skip_vs_c13": {
            "bal_ub25_diff": stop_r['balanced_utility_beta0.25'] - c13_r['balanced_utility_beta0.25'],
            "macro_bal_drop": stop_r['macro_query_bal_acc'] - c13_r['macro_query_bal_acc'],
            "cost_saving": stop_r['total_cost'] - c13_r['total_cost'],
        },
    },
    "diagnostic_flags": {
        "c13_beats_c0b_balanced": c13_beats_c0b_balanced,
        "distance_filter_beats_c13": distance_filter_beats_c13,
        "stop_on_skip_is_metric_exploit": stop_is_metric_exploit,
    },
    "interpretation": {
        "c13_improves_macro_bal": c13_improves_macro_bal,
        "c13_loses_bal_utility": c13_loses_bal_utility,
        "stop_high_bal_ub25_low_macro_bal": stop_high_bal_ub25_low_macro_bal,
        "filt_improves_bal_per_probe": filt_improves_bal_per_probe,
        "filt_loses_bal_utility": filt_loses_bal_utility,
    },
}

for pname in policy_names:
    r = policy_results[pname]
    block_out["policies"][pname] = {
        "raw_comp_acc": r["raw_comp_acc"],
        "macro_query_acc": r["macro_query_acc"],
        "macro_query_bal_acc": r["macro_query_bal_acc"],
        "normalized_cost": r["normalized_cost"],
        "raw_utility_beta0.25": r["raw_utility_beta0.25"],
        "balanced_utility_beta0.25": r["balanced_utility_beta0.25"],
        "visited_count": r["visited_count"],
        "probe_count": r["probe_count"],
        "effective_probe_count": r["effective_probe_count"],
        "zero_gain_probe_count": r["zero_gain_probe_count"],
        "harmful_probe_count": r["harmful_probe_count"],
        "actual_probe_delta_sum": r["actual_probe_delta_sum"],
        "total_cost": r["total_cost"],
        "budget_remaining": r["budget_remaining"],
        "per_query": {
            q: {
                "accuracy": r["per_query_accuracy"][q],
                "balanced_accuracy": r["per_query_balanced_accuracy"][q],
                "positive_recall": r["per_query_positive_recall"][q],
                "negative_recall": r["per_query_negative_recall"][q],
            }
            for q in QUERY_NAMES
        },
    }

with open("runs/calibration_block1h1_balanced_policy_rescore.json", "w") as f:
    json.dump(make_serializable(block_out), f, indent=2)
print(f"  saved: runs/calibration_block1h1_balanced_policy_rescore.json")
