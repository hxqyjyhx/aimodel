"""Block 1F3: Post-observe stop/continue diagnostic.

Tests whether conserving saved probe budget (by stopping instead of
spending it on more visits) recovers the posthoc utility gain.
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

DISTANCE_THRESHOLD = 0.094829

print("=" * 70)
print("Block 1F3: Post-Observe Stop/Continue Diagnostic")
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


# ============ C13DistanceFilterStopOnSkip ============
class C13DistanceFilterStopOnSkip:
    """Distance-gated probe filter that STOPS the episode on first distance skip.

    Conserves saved probe budget instead of spending it on additional
    visits/observes. Tests whether budget conservation recovers utility.
    """

    def __init__(self, inner_policy, im, gt_per_object, threshold):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object
        self._threshold = threshold
        self.probe_records = []
        self._stopped = False
        self.stop_info = {}
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
            # Distance gate skip → STOP the episode
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
        "n_effective": sum(1 for r in records if r["is_effective"]),
        "n_zero_gain": sum(1 for r in records if r["is_zero_gain"]),
        "n_harmful": sum(1 for r in records if r["is_harmful"]),
        "actual_delta_sum": sum(r["actual_delta_task_correct"] for r in records),
    }


# ======== Run all 5 policies ========

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

# 4. C13_distance_filter_stop_on_skip
print("  Running C13_distance_filter_stop_on_skip...")
env_stop = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_stop = im_base.clone()
rng_stop = random.Random(seed + 550)
stop_inner = C13_InstanceVOIPolicy(im_stop, rng_stop, cost_weight=cw)
stop_policy = C13DistanceFilterStopOnSkip(stop_inner, im_stop, gt_per_object, DISTANCE_THRESHOLD)
stop_metrics, _ = run_policy(stop_policy, env_stop, "C13_distance_filter_stop_on_skip")
stop_ps = probe_summary(stop_policy.probe_records)
stop_fs = stop_policy.filter_stats
stop_info = stop_policy.stop_info

# 5. C13_distance_filter_stop_after_first_skip_optional
# Same implementation as stop_on_skip for this diagnostic.
# The "optional" designation means future work could make the stop
# conditional (e.g., only stop if no probes have been kept yet).
print("  Running C13_distance_filter_stop_after_first_skip_optional...")
env_opt = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_opt = im_base.clone()
rng_opt = random.Random(seed + 550)
opt_inner = C13_InstanceVOIPolicy(im_opt, rng_opt, cost_weight=cw)
opt_policy = C13DistanceFilterStopOnSkip(opt_inner, im_opt, gt_per_object, DISTANCE_THRESHOLD)
opt_metrics, _ = run_policy(opt_policy, env_opt,
                            "C13_distance_filter_stop_after_first_skip_optional")
opt_ps = probe_summary(opt_policy.probe_records)
opt_fs = opt_policy.filter_stats
opt_stop_info = opt_policy.stop_info

# ======== Report ========
BETA = 0.25

print()
print("=" * 70)
print("Block 1F3 — Post-Observe Stop/Continue Results")
print("=" * 70)


def policy_block(label, metrics, ps_dict):
    comp = metrics["composite_task_accuracy"]
    ncost = metrics["normalized_cost"]
    ub25 = comp - BETA * ncost
    print(f"\n--- {label} ---")
    print(f"  comp={comp:.4f}  ncost={ncost:.4f}  ub25={ub25:.4f}")
    print(f"  visited={metrics['objects_visited']}"
          f"  probe_count={ps_dict['n_probe']}"
          f"  probe_coverage={metrics['probe_coverage']:.4f}")
    print(f"  effective={ps_dict['n_effective']}"
          f"  zero_gain={ps_dict['n_zero_gain']}"
          f"  harmful={ps_dict['n_harmful']}"
          f"  delta_sum={ps_dict['actual_delta_sum']}")
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
r_stop = policy_block("C13_distance_filter_stop_on_skip", stop_metrics, stop_ps)
r_opt = policy_block("C13_distance_filter_stop_after_first_skip_optional",
                     opt_metrics, opt_ps)

# Stop-on-skip details
print(f"\n--- C13_distance_filter_stop_on_skip: Stop Details ---")
print(f"  stopped_due_to_distance_skip={stop_info.get('stopped_due_to_distance_skip', False)}")
print(f"  stop_object_id={stop_info.get('stop_object_id', 'N/A')}")
print(f"  stop_action={stop_info.get('stop_action', 'N/A')}")
print(f"  stop_distance={stop_info.get('stop_distance', 'N/A')}")
print(f"  budget_remaining_at_stop={stop_info.get('budget_remaining_at_stop', 'N/A')}")
print(f"  n_probe_candidates_seen={stop_fs['n_candidates_seen']}")
print(f"  n_probe_kept={stop_fs['n_kept']}")
print(f"  n_probe_skipped_by_distance={stop_fs['n_skipped_by_distance']}")
print(f"  kept_actions={stop_fs['kept_actions']}")
print(f"  skipped_actions={stop_fs['skipped_actions']}")

# ======== Acceptance ========
passed_primary = r_stop["ub25"] > r_c13["ub25"]
passed_strong = r_stop["ub25"] > r_c0b["ub25"]
cost_lt_c13 = r_stop["total_cost"] < r_c13["total_cost"] - 0.0001
probes_lt_c13 = r_stop["probe_count"] < r_c13["probe_count"]
harmful_zero = stop_ps["n_harmful"] == 0
comp_drop = r_c13["comp"] - r_stop["comp"]

print()
print(f"--- Acceptance ---")
print(f"  Primary: stop ub25 > C13 ub25:  {r_stop['ub25']:.4f} > {r_c13['ub25']:.4f}  "
      f"{'OK' if passed_primary else 'FAIL'}")
print(f"  Strong:  stop ub25 > C0b ub25:  {r_stop['ub25']:.4f} > {r_c0b['ub25']:.4f}  "
      f"{'OK' if passed_strong else 'FAIL'}")
print(f"  Mechanism:")
print(f"    total_cost < C13:  {r_stop['total_cost']:.4f} < {r_c13['total_cost']:.4f}  "
      f"{'OK' if cost_lt_c13 else 'FAIL'}")
print(f"    probes < C13:  {r_stop['probe_count']} < {r_c13['probe_count']}  "
      f"{'OK' if probes_lt_c13 else 'FAIL'}")
print(f"    harmful=0:  {stop_ps['n_harmful']} == 0  "
      f"{'OK' if harmful_zero else 'FAIL'}")
print(f"    comp_drop_vs_C13:  {comp_drop:.4f}  "
      f"{'<=0.02' if comp_drop <= 0.02 else '>0.02'}")

print()
print("[block_done]")
print(f"  block_id=1F3")
print(f"  c0b_ub25={r_c0b['ub25']:.4f}")
print(f"  c13_ub25={r_c13['ub25']:.4f}")
print(f"  c13_distance_filter_ub25={r_filt['ub25']:.4f}")
print(f"  stop_on_skip_ub25={r_stop['ub25']:.4f}")
print(f"  passed_primary={'true' if passed_primary else 'false'}")
print(f"  passed_strong={'true' if passed_strong else 'false'}")
print(f"  label=same-condition diagnostic of budget-conservation effect")
print(f"  elapsed={time.time() - t0:.0f}s")

# ======== Save ========
block_out = {
    "block_id": "1F3",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": cw, "seed": seed,
    "distance_threshold": DISTANCE_THRESHOLD,
    "threshold_source": "Block 1F0c diagnostic_4 dist_median — same-condition pilot, NOT generalization evidence",
    "label": "same-condition diagnostic of budget-conservation effect",
    "beta": BETA,
    "C0b": {
        "comp": r_c0b["comp"], "ncost": r_c0b["ncost"], "ub25": r_c0b["ub25"],
        "visited_count": r_c0b["visited"], "total_cost": r_c0b["total_cost"],
        "probe_count": r_c0b["probe_count"], "probe_coverage": r_c0b["probe_coverage"],
        "effective_probe_count": c0b_ps["n_effective"],
        "zero_gain_probe_count": c0b_ps["n_zero_gain"],
        "harmful_probe_count": c0b_ps["n_harmful"],
        "actual_probe_delta_sum": c0b_ps["actual_delta_sum"],
        "budget_remaining": r_c0b["budget_remaining"],
    },
    "C13": {
        "comp": r_c13["comp"], "ncost": r_c13["ncost"], "ub25": r_c13["ub25"],
        "visited_count": r_c13["visited"], "total_cost": r_c13["total_cost"],
        "probe_count": r_c13["probe_count"], "probe_coverage": r_c13["probe_coverage"],
        "effective_probe_count": c13_ps["n_effective"],
        "zero_gain_probe_count": c13_ps["n_zero_gain"],
        "harmful_probe_count": c13_ps["n_harmful"],
        "actual_probe_delta_sum": c13_ps["actual_delta_sum"],
        "budget_remaining": r_c13["budget_remaining"],
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
        "budget_remaining": r_filt["budget_remaining"],
        "probe_records": filt_policy.probe_records,
        "filter_stats": {
            "n_candidates_seen": filt_fs["n_candidates_seen"],
            "n_kept": filt_fs["n_kept"],
            "n_skipped_by_distance": filt_fs["n_skipped_by_distance"],
        },
    },
    "C13_distance_filter_stop_on_skip": {
        "comp": r_stop["comp"], "ncost": r_stop["ncost"], "ub25": r_stop["ub25"],
        "visited_count": r_stop["visited"], "total_cost": r_stop["total_cost"],
        "probe_count": r_stop["probe_count"], "probe_coverage": r_stop["probe_coverage"],
        "effective_probe_count": stop_ps["n_effective"],
        "zero_gain_probe_count": stop_ps["n_zero_gain"],
        "harmful_probe_count": stop_ps["n_harmful"],
        "actual_probe_delta_sum": stop_ps["actual_delta_sum"],
        "budget_remaining": r_stop["budget_remaining"],
        "probe_records": stop_policy.probe_records,
        "filter_stats": {
            "n_probe_candidates_seen": stop_fs["n_candidates_seen"],
            "n_probe_kept": stop_fs["n_kept"],
            "n_probe_skipped_by_distance": stop_fs["n_skipped_by_distance"],
            "kept_distance_mean": _mean(stop_fs["kept_distances"]),
            "skipped_distance_mean": _mean(stop_fs["skipped_distances"]),
            "kept_actions": stop_fs["kept_actions"],
            "skipped_actions": stop_fs["skipped_actions"],
        },
        "stop_info": stop_info,
        "utility_gain_over_C13": round(r_stop["ub25"] - r_c13["ub25"], 6),
        "utility_gain_over_C0b": round(r_stop["ub25"] - r_c0b["ub25"], 6),
    },
    "C13_distance_filter_stop_after_first_skip_optional": {
        "comp": r_opt["comp"], "ncost": r_opt["ncost"], "ub25": r_opt["ub25"],
        "visited_count": r_opt["visited"], "total_cost": r_opt["total_cost"],
        "probe_count": r_opt["probe_count"], "probe_coverage": r_opt["probe_coverage"],
        "effective_probe_count": opt_ps["n_effective"],
        "zero_gain_probe_count": opt_ps["n_zero_gain"],
        "harmful_probe_count": opt_ps["n_harmful"],
        "actual_probe_delta_sum": opt_ps["actual_delta_sum"],
        "note": "Same implementation as stop_on_skip for this diagnostic. "
                "Optional designation = future work could make stop conditional.",
    },
    "acceptance": {
        "passed_primary": passed_primary,
        "passed_strong": passed_strong,
        "primary_check": f"stop_ub25 ({r_stop['ub25']:.4f}) > c13_ub25 ({r_c13['ub25']:.4f})",
        "strong_check": f"stop_ub25 ({r_stop['ub25']:.4f}) > c0b_ub25 ({r_c0b['ub25']:.4f})",
        "cost_lt_c13": cost_lt_c13,
        "probes_lt_c13": probes_lt_c13,
        "harmful_zero": harmful_zero,
        "comp_drop_vs_c13": round(comp_drop, 6),
        "label": "same-condition diagnostic of budget-conservation effect",
    },
}
with open("runs/calibration_block1f3_stop_on_skip_diagnostic.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1f3_stop_on_skip_diagnostic.json")
