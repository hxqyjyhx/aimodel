"""Block 1F2: Distance-aware object selection diagnostic.

Tests whether aligning object selection with the distance gate avoids
the wasted visit/observe cost seen in Block 1F1.
"""
import sys, os, json, time, copy, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

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

PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, PARENT_DIR)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False})
budget = 1.5
cw = 0.5

# Fixed threshold from Block 1F0c diagnostic_4 distance_filter median.
# Same-condition pilot threshold — NOT generalization evidence.
DISTANCE_THRESHOLD = 0.094829

print("=" * 70)
print("Block 1F2: Distance-Aware Object Selection Diagnostic")
print(f"  cue=C3_P060_O040, budget={budget}, CW={cw}, seed={seed}")
print(f"  distance_threshold={DISTANCE_THRESHOLD}")
print("=" * 70)


# ============ Helpers ============
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


# ============ ProbeLabelingTracker (shared) ============
class ProbeLabelingTracker:
    """Wraps a policy to capture per-probe before/after IOM state for GT labeling.

    GT is ONLY used for post-hoc diagnostic labeling — NOT as filter input.
    """

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
                "object_id": object_id,
                "action": action,
                "gt": gt,
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
                "object_id": object_id,
                "action": action,
                "outcome": outcome,
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


# ============ C13DistanceFilterPolicy (from Block 1F1) ============
class C13DistanceFilterPolicy:
    """Wraps C13. After C13 selects a candidate probe action, only executes
    the probe if distance_to_threshold <= fixed threshold.
    """

    def __init__(self, inner_policy, im, gt_per_object, threshold):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self._threshold = threshold
        self.probe_records = []
        self.filter_stats = {
            "n_candidates_seen": 0,
            "n_kept": 0,
            "n_skipped_by_distance": 0,
            "kept_distances": [],
            "skipped_distances": [],
            "kept_actions": [],
            "skipped_actions": [],
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


# ============ C13DistanceAwareSelectionPolicy ============
class C13DistanceAwareSelectionPolicy:
    """Distance-aware object selection + distance-gated probe decisions.

    select_next_object: computes VOI using only distance-eligible actions
    (prior probs). Objects with zero eligible actions are skipped.

    decide_probe: same distance gate as C13DistanceFilterPolicy.
    """

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
            "n_candidates_seen": 0,
            "n_kept": 0,
            "n_skipped_by_distance": 0,
            "kept_distances": [],
            "skipped_distances": [],
            "kept_actions": [],
            "skipped_actions": [],
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
        """Return list of (action, p_success) for actions with distance <= threshold."""
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

            # Compute VOI using only eligible actions
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

        # Call inner C13 for its VOI-based action selection
        should_probe, action = self._inner.decide_probe(view, object_id)

        if not should_probe or action is None:
            self._pending_label = None
            return False, None

        # Distance gate (same threshold as select_next_object)
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


# ======== Phase A: Training ========
print("\n  Phase A: Training...")
(student, base_learner, train_objects, train_env,
 test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
test_oids = sorted(test_objects)
test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

outcome_rows, _ = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, seed)

posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)

from simulator_truth import MiniMCSimulatorTruth
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

_temp_sim = MiniMCSimulatorTruth(test_objects_dict, positions, config.AGENT_START)
gt_per_object = {oid: _temp_sim.get_ground_truth_affordances(oid) for oid in test_oids}


def run_policy(policy, env, label):
    """Run a policy through harness and evaluate."""
    result = EpisodeHarness(env, policy).run()
    evaluator = MiniMCEvaluator(env)
    metrics = evaluator.evaluate(
        result.predictions, result.event_log, result.agent_obs,
        result.pre_probe_entropies, result.pre_decision_entropies,
    )
    metrics["label"] = label
    return metrics, result


def probe_summary(records):
    """Return dict of probe summary stats from labeled records."""
    return {
        "n_probe": len(records),
        "n_effective": sum(1 for r in records if r["is_effective"]),
        "n_zero_gain": sum(1 for r in records if r["is_zero_gain"]),
        "n_harmful": sum(1 for r in records if r["is_harmful"]),
        "actual_delta_sum": sum(r["actual_delta_task_correct"] for r in records),
    }


# ======== Run all 4 policies ========

# 1. C0b
print("\n  Running C0b...")
env_c0b = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c0b = im_base.clone()
rng_c0b = random.Random(seed + 100)
c0b_inner = C0b_ObserveOnlyPolicy(im_c0b, rng_c0b)
c0b_policy = ProbeLabelingTracker(c0b_inner, im_c0b, gt_per_object)
c0b_metrics, _ = run_policy(c0b_policy, env_c0b, "C0b")
c0b_ps = probe_summary(c0b_policy.probe_records)

# 2. C13
print("  Running original C13...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
rng_c13 = random.Random(seed + 550)
c13_inner = C13_InstanceVOIPolicy(im_c13, rng_c13, cost_weight=cw)
c13_policy = ProbeLabelingTracker(c13_inner, im_c13, gt_per_object)
c13_metrics, _ = run_policy(c13_policy, env_c13, "C13")
c13_ps = probe_summary(c13_policy.probe_records)

# 3. C13_distance_filter
print("  Running C13_distance_filter...")
env_filt = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_filt = im_base.clone()
rng_filt = random.Random(seed + 550)
filt_inner = C13_InstanceVOIPolicy(im_filt, rng_filt, cost_weight=cw)
filt_policy = C13DistanceFilterPolicy(filt_inner, im_filt, gt_per_object, DISTANCE_THRESHOLD)
filt_metrics, _ = run_policy(filt_policy, env_filt, "C13_distance_filter")
filt_ps = probe_summary(filt_policy.probe_records)
filt_fs = filt_policy.filter_stats

# 4. C13_distance_aware_selection
print("  Running C13_distance_aware_selection...")
env_aware = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_aware = im_base.clone()
rng_aware = random.Random(seed + 550)
aware_inner = C13_InstanceVOIPolicy(im_aware, rng_aware, cost_weight=cw)
aware_policy = C13DistanceAwareSelectionPolicy(
    aware_inner, im_aware, gt_per_object, DISTANCE_THRESHOLD, cw)
aware_metrics, _ = run_policy(aware_policy, env_aware, "C13_distance_aware_selection")
aware_ps = probe_summary(aware_policy.probe_records)
aware_ss = aware_policy.selection_stats
aware_fs = aware_policy.filter_stats

# ======== Report ========
BETA = 0.25

print()
print("=" * 70)
print("Block 1F2 — Distance-Aware Object Selection Results")
print("=" * 70)


def policy_block(label, metrics, ps_dict):
    comp = metrics["composite_task_accuracy"]
    ncost = metrics["normalized_cost"]
    ub25 = comp - BETA * ncost
    print(f"\n--- {label} ---")
    print(f"  comp={comp:.4f}  ncost={ncost:.4f}  ub25={ub25:.4f}")
    print(f"  visited={metrics['objects_visited']}  observe={metrics['objects_visited']}"
          f"  probe_count={ps_dict['n_probe']}  probe_coverage={metrics['probe_coverage']:.4f}")
    print(f"  effective={ps_dict['n_effective']}  zero_gain={ps_dict['n_zero_gain']}"
          f"  harmful={ps_dict['n_harmful']}  delta_sum={ps_dict['actual_delta_sum']}")
    print(f"  total_cost={metrics['total_cost']:.4f}"
          f"  budget_remaining={metrics['budget_remaining']:.4f}")
    return {"comp": comp, "ncost": ncost, "ub25": ub25,
            "visited": metrics["objects_visited"],
            "total_cost": metrics["total_cost"],
            "budget_remaining": metrics["budget_remaining"],
            "probe_count": ps_dict["n_probe"],
            "probe_coverage": metrics["probe_coverage"]}


r_c0b = policy_block("C0b", c0b_metrics, c0b_ps)
r_c13 = policy_block("C13", c13_metrics, c13_ps)
r_filt = policy_block("C13_distance_filter", filt_metrics, filt_ps)
r_aware = policy_block("C13_distance_aware_selection", aware_metrics, aware_ps)

# Additional diagnostics for C13_distance_filter
print(f"\n--- C13_distance_filter: Filter Stats ---")
print(f"  n_candidates_seen={filt_fs['n_candidates_seen']}  kept={filt_fs['n_kept']}"
      f"  skipped={filt_fs['n_skipped_by_distance']}")

# Additional diagnostics for C13_distance_aware_selection
print(f"\n--- C13_distance_aware_selection: Selection Stats ---")
print(f"  n_objects_considered={aware_ss['n_objects_considered']}")
print(f"  n_objects_rejected_no_eligible_action={aware_ss['n_objects_rejected_no_eligible_action']}")
print(f"  n_objects_selected_with_eligible_action={aware_ss['n_objects_selected_with_eligible_action']}")
print(f"  visited_count_vs_C13: {r_aware['visited']} vs {r_c13['visited']}")
print(f"  total_cost_vs_C13: {r_aware['total_cost']:.4f} vs {r_c13['total_cost']:.4f}")
print(f"  utility_gain_over_C13: {r_aware['ub25'] - r_c13['ub25']:+.4f}")
print(f"  utility_gain_over_C0b: {r_aware['ub25'] - r_c0b['ub25']:+.4f}")

print(f"\n--- C13_distance_aware_selection: Filter Stats ---")
print(f"  n_probe_candidates_seen={aware_fs['n_candidates_seen']}"
      f"  n_probe_kept={aware_fs['n_kept']}"
      f"  n_probe_skipped_by_distance={aware_fs['n_skipped_by_distance']}")
print(f"  kept_distance_mean={_mean(aware_fs['kept_distances']):.6f}")
print(f"  skipped_distance_mean={_mean(aware_fs['skipped_distances']):.6f}")
print(f"  kept_actions={aware_fs['kept_actions']}")
print(f"  skipped_actions={aware_fs['skipped_actions']}")
if aware_fs["skipped_candidate_records"]:
    print(f"  skipped_probe_candidate_records:")
    for r in aware_fs["skipped_candidate_records"]:
        print(f"    {r['object_id']}  {r['action']}  dist={r['distance_to_threshold']:.6f}"
              f"  p={r['p_success']:.4f}")

# ======== Acceptance ========
passed_primary = r_aware["ub25"] > r_c13["ub25"]
passed_strong = r_aware["ub25"] > r_c0b["ub25"]
comp_drop = r_c13["comp"] - r_aware["comp"]
probes_lt_c13 = r_aware["probe_count"] < r_c13["probe_count"]
visited_ok = r_aware["visited"] <= r_c13["visited"] + 3  # not substantially more
cost_le_c13 = r_aware["total_cost"] <= r_c13["total_cost"] + 0.001
harmful_zero = aware_ps["n_harmful"] == 0

print()
print(f"--- Acceptance ---")
print(f"  Primary: aware ub25 > C13 ub25:  {r_aware['ub25']:.4f} > {r_c13['ub25']:.4f}  "
      f"{'OK' if passed_primary else 'FAIL'}")
print(f"  Strong:  aware ub25 > C0b ub25:  {r_aware['ub25']:.4f} > {r_c0b['ub25']:.4f}  "
      f"{'OK' if passed_strong else 'FAIL'}")
print(f"  Mechanism:")
print(f"    probes < C13:  {r_aware['probe_count']} < {r_c13['probe_count']}  "
      f"{'OK' if probes_lt_c13 else 'FAIL'}")
print(f"    visited <= C13+3:  {r_aware['visited']} <= {r_c13['visited'] + 3}  "
      f"{'OK' if visited_ok else 'FAIL'}")
print(f"    total_cost <= C13:  {r_aware['total_cost']:.4f} <= {r_c13['total_cost']:.4f}  "
      f"{'OK' if cost_le_c13 else 'FAIL'}")
print(f"    harmful=0:  {aware_ps['n_harmful']} == 0  "
      f"{'OK' if harmful_zero else 'FAIL'}")
print(f"    comp_drop_vs_C13:  {comp_drop:.4f}")

print()
print("[block_done]")
print(f"  block_id=1F2")
print(f"  c0b_ub25={r_c0b['ub25']:.4f}")
print(f"  c13_ub25={r_c13['ub25']:.4f}")
print(f"  c13_distance_filter_ub25={r_filt['ub25']:.4f}")
print(f"  c13_distance_aware_ub25={r_aware['ub25']:.4f}")
print(f"  passed_primary={'true' if passed_primary else 'false'}")
print(f"  passed_strong={'true' if passed_strong else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ======== Save ========
block_out = {
    "block_id": "1F2",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": cw, "seed": seed,
    "distance_threshold": DISTANCE_THRESHOLD,
    "threshold_source": "Block 1F0c diagnostic_4 dist_median — same-condition pilot, NOT generalization evidence",
    "beta": BETA,
    "C0b": {
        "comp": r_c0b["comp"], "ncost": r_c0b["ncost"], "ub25": r_c0b["ub25"],
        "visited_count": r_c0b["visited"], "total_cost": r_c0b["total_cost"],
        "probe_count": r_c0b["probe_count"], "probe_coverage": r_c0b["probe_coverage"],
        "effective_probe_count": c0b_ps["n_effective"],
        "zero_gain_probe_count": c0b_ps["n_zero_gain"],
        "harmful_probe_count": c0b_ps["n_harmful"],
        "actual_probe_delta_sum": c0b_ps["actual_delta_sum"],
    },
    "C13": {
        "comp": r_c13["comp"], "ncost": r_c13["ncost"], "ub25": r_c13["ub25"],
        "visited_count": r_c13["visited"], "total_cost": r_c13["total_cost"],
        "probe_count": r_c13["probe_count"], "probe_coverage": r_c13["probe_coverage"],
        "effective_probe_count": c13_ps["n_effective"],
        "zero_gain_probe_count": c13_ps["n_zero_gain"],
        "harmful_probe_count": c13_ps["n_harmful"],
        "actual_probe_delta_sum": c13_ps["actual_delta_sum"],
        "probe_records": c13_policy.probe_records,
    },
    "C13_distance_filter": {
        "comp": r_filt["comp"], "ncost": r_filt["ncost"], "ub25": r_filt["ub25"],
        "visited_count": r_filt["visited"], "total_cost": r_filt["total_cost"],
        "probe_count": r_filt["probe_count"], "probe_coverage": r_filt["probe_coverage"],
        "effective_probe_count": filt_ps["n_effective"],
        "zero_gain_probe_count": filt_ps["n_zero_gain"],
        "harmful_probe_count": filt_ps["n_harmful"],
        "actual_probe_delta_sum": filt_ps["actual_delta_sum"],
        "probe_records": filt_policy.probe_records,
        "filter_stats": {
            "n_candidates_seen": filt_fs["n_candidates_seen"],
            "n_kept": filt_fs["n_kept"],
            "n_skipped_by_distance": filt_fs["n_skipped_by_distance"],
            "kept_distance_mean": _mean(filt_fs["kept_distances"]),
            "skipped_distance_mean": _mean(filt_fs["skipped_distances"]),
            "kept_actions": filt_fs["kept_actions"],
            "skipped_actions": filt_fs["skipped_actions"],
        },
    },
    "C13_distance_aware_selection": {
        "comp": r_aware["comp"], "ncost": r_aware["ncost"], "ub25": r_aware["ub25"],
        "visited_count": r_aware["visited"], "total_cost": r_aware["total_cost"],
        "probe_count": r_aware["probe_count"], "probe_coverage": r_aware["probe_coverage"],
        "effective_probe_count": aware_ps["n_effective"],
        "zero_gain_probe_count": aware_ps["n_zero_gain"],
        "harmful_probe_count": aware_ps["n_harmful"],
        "actual_probe_delta_sum": aware_ps["actual_delta_sum"],
        "probe_records": aware_policy.probe_records,
        "selection_stats": {
            "n_objects_considered": aware_ss["n_objects_considered"],
            "n_objects_rejected_no_eligible_action": aware_ss["n_objects_rejected_no_eligible_action"],
            "n_objects_selected_with_eligible_action": aware_ss["n_objects_selected_with_eligible_action"],
        },
        "filter_stats": {
            "n_probe_candidates_seen": aware_fs["n_candidates_seen"],
            "n_probe_kept": aware_fs["n_kept"],
            "n_probe_skipped_by_distance": aware_fs["n_skipped_by_distance"],
            "kept_distance_mean": _mean(aware_fs["kept_distances"]),
            "skipped_distance_mean": _mean(aware_fs["skipped_distances"]),
            "kept_actions": aware_fs["kept_actions"],
            "skipped_actions": aware_fs["skipped_actions"],
            "skipped_probe_candidate_records": aware_fs["skipped_candidate_records"],
        },
        "comparison": {
            "visited_count_vs_C13": f"{r_aware['visited']} vs {r_c13['visited']}",
            "total_cost_vs_C13": f"{r_aware['total_cost']:.4f} vs {r_c13['total_cost']:.4f}",
            "utility_gain_over_C13": round(r_aware["ub25"] - r_c13["ub25"], 6),
            "utility_gain_over_C0b": round(r_aware["ub25"] - r_c0b["ub25"], 6),
        },
    },
    "acceptance": {
        "passed_primary": passed_primary,
        "passed_strong": passed_strong,
        "primary_check": f"aware_ub25 ({r_aware['ub25']:.4f}) > c13_ub25 ({r_c13['ub25']:.4f})",
        "strong_check": f"aware_ub25 ({r_aware['ub25']:.4f}) > c0b_ub25 ({r_c0b['ub25']:.4f})",
        "probes_lt_c13": probes_lt_c13,
        "visited_within_range": visited_ok,
        "cost_le_c13": cost_le_c13,
        "harmful_zero": harmful_zero,
        "comp_drop_vs_c13": round(comp_drop, 6),
    },
}
with open("runs/calibration_block1f2_distance_aware_selection.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1f2_distance_aware_selection.json")
