"""Block 1F1: Real rollout pilot for distance-based probe filter.

Same-condition pilot: C3_P060_O040, budget=1.5, CW=0.5, seed=101.
Threshold = median distance_to_threshold from Block 1F0c (0.094829).
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

# Fixed threshold from Block 1F0c diagnostic_4 distance_filter median
# This is a same-condition pilot threshold — NOT generalization evidence.
DISTANCE_THRESHOLD = 0.094829

print("=" * 70)
print("Block 1F1: Distance Filter Rollout Pilot")
print(f"  cue=C3_P060_O040, budget={budget}, CW={cw}, seed={seed}")
print(f"  distance_threshold={DISTANCE_THRESHOLD} (median from Block 1F0c)")
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


# ============ ProbeLabelingTracker ============
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

        # Snapshot IOM state before probe
        im_before = self._im.clone()

        should_probe, action = self._inner.decide_probe(view, object_id)

        if should_probe and action is not None:
            p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
            self._pending_label = {
                "object_id": object_id,
                "action": action,
                "gt": gt,
                "before_correct": before_correct,
                "im_before": im_before,
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


# ============ C13DistanceFilterPolicy ============
class C13DistanceFilterPolicy:
    """Wraps C13. After C13 selects a candidate probe action, only executes
    the probe if distance_to_threshold <= fixed threshold.

    The threshold is fixed before rollout begins. No adaptation.
    No GT access. No posterior VOI.
    """

    def __init__(self, inner_policy, im, gt_per_object, threshold):
        self._inner = inner_policy
        self._im = im
        self._gt = gt_per_object  # post-hoc labeling only
        self._threshold = threshold
        self.probe_records = []   # all probe candidates (kept + skipped)
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

        # Call inner C13
        should_probe, action = self._inner.decide_probe(view, object_id)

        if not should_probe or action is None:
            self._pending_label = None
            return False, None

        # Distance gate
        p_success = before_probs.get(ACTION_TO_FEATURE[action], 0.5)
        dist = abs(p_success - 0.5)
        keep = dist <= self._threshold

        self.filter_stats["n_candidates_seen"] += 1

        if keep:
            self.filter_stats["n_kept"] += 1
            self.filter_stats["kept_distances"].append(dist)
            self.filter_stats["kept_actions"].append(action)
            self._pending_label = {
                "object_id": object_id,
                "action": action,
                "gt": gt,
                "before_correct": before_correct,
                "distance_to_threshold": dist,
                "p_success": p_success,
                "kept": True,
            }
            return True, action
        else:
            self.filter_stats["n_skipped_by_distance"] += 1
            self.filter_stats["skipped_distances"].append(dist)
            self.filter_stats["skipped_actions"].append(action)
            self.filter_stats["skipped_candidate_records"].append({
                "object_id": object_id,
                "action": action,
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
    """Run a policy through harness and evaluate. Returns metrics dict."""
    result = EpisodeHarness(env, policy).run()
    evaluator = MiniMCEvaluator(env)
    metrics = evaluator.evaluate(
        result.predictions, result.event_log, result.agent_obs,
        result.pre_probe_entropies, result.pre_decision_entropies,
    )
    metrics["label"] = label
    return metrics, result


# ======== Run C0b ========
print("\n  Running C0b observe-only baseline...")
env_c0b = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c0b = im_base.clone()
rng_c0b = random.Random(seed + 100)
c0b_policy = C0b_ObserveOnlyPolicy(im_c0b, rng_c0b)
c0b_labeled = ProbeLabelingTracker(c0b_policy, im_c0b, gt_per_object)
c0b_metrics, c0b_result = run_policy(c0b_labeled, env_c0b, "C0b")

n_probe_c0b = len(c0b_labeled.probe_records)
# C0b never probes, but just in case:
c0b_eff = sum(1 for r in c0b_labeled.probe_records if r["is_effective"])
c0b_zg = sum(1 for r in c0b_labeled.probe_records if r["is_zero_gain"])
c0b_harm = sum(1 for r in c0b_labeled.probe_records if r["is_harmful"])
c0b_delta_sum = sum(r["actual_delta_task_correct"] for r in c0b_labeled.probe_records)


# ======== Run C13 ========
print("  Running original C13 (CW=0.5)...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
rng_c13 = random.Random(seed + 550)
c13_inner = C13_InstanceVOIPolicy(im_c13, rng_c13, cost_weight=cw)
c13_labeled = ProbeLabelingTracker(c13_inner, im_c13, gt_per_object)
c13_metrics, c13_result = run_policy(c13_labeled, env_c13, "C13")

n_probe_c13 = len(c13_labeled.probe_records)
c13_eff = sum(1 for r in c13_labeled.probe_records if r["is_effective"])
c13_zg = sum(1 for r in c13_labeled.probe_records if r["is_zero_gain"])
c13_harm = sum(1 for r in c13_labeled.probe_records if r["is_harmful"])
c13_delta_sum = sum(r["actual_delta_task_correct"] for r in c13_labeled.probe_records)


# ======== Run C13_distance_filter ========
print(f"  Running C13_distance_filter (threshold={DISTANCE_THRESHOLD})...")
env_dist = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_dist = im_base.clone()
rng_dist = random.Random(seed + 550)  # same seed as C13 for reproducibility of decisions
dist_inner = C13_InstanceVOIPolicy(im_dist, rng_dist, cost_weight=cw)
dist_policy = C13DistanceFilterPolicy(dist_inner, im_dist, gt_per_object, DISTANCE_THRESHOLD)
dist_metrics, dist_result = run_policy(dist_policy, env_dist, "C13_distance_filter")

n_probe_dist = len(dist_policy.probe_records)
dist_eff = sum(1 for r in dist_policy.probe_records if r["is_effective"])
dist_zg = sum(1 for r in dist_policy.probe_records if r["is_zero_gain"])
dist_harm = sum(1 for r in dist_policy.probe_records if r["is_harmful"])
dist_delta_sum = sum(r["actual_delta_task_correct"] for r in dist_policy.probe_records)

# ======== Report ========
print()
print("=" * 70)
print("Block 1F1 — Distance Filter Rollout Results")
print("=" * 70)

BETA = 0.25
TOTAL_COMPOSITE_DECISIONS = 60 * 5


def print_policy_row(name, metrics, n_probe, n_eff, n_zg, n_harm, delta_sum):
    comp = metrics["composite_task_accuracy"]
    ncost = metrics["normalized_cost"]
    ub25 = comp - BETA * ncost
    print(f"\n--- {name} ---")
    print(f"  comp={comp:.4f}  ncost={ncost:.4f}  ub25={ub25:.4f}")
    print(f"  visited={metrics['objects_visited']}  observe={metrics['objects_visited']}"
          f"  probe_count={n_probe}  probe_coverage={metrics['probe_coverage']:.4f}")
    print(f"  effective={n_eff}  zero_gain={n_zg}  harmful={n_harm}"
          f"  actual_probe_delta_sum={delta_sum}")
    print(f"  total_cost={metrics['total_cost']:.4f}  budget_remaining={metrics['budget_remaining']:.4f}")
    return comp, ncost, ub25


c0b_comp, c0b_ncost, c0b_ub25 = print_policy_row(
    "C0b", c0b_metrics, n_probe_c0b, c0b_eff, c0b_zg, c0b_harm, c0b_delta_sum)
c13_comp, c13_ncost, c13_ub25 = print_policy_row(
    "C13", c13_metrics, n_probe_c13, c13_eff, c13_zg, c13_harm, c13_delta_sum)
dist_comp, dist_ncost, dist_ub25 = print_policy_row(
    "C13_distance_filter", dist_metrics, n_probe_dist, dist_eff, dist_zg, dist_harm,
    dist_delta_sum)

# C13_distance_filter additional stats
fs = dist_policy.filter_stats
print(f"\n--- C13_distance_filter: Filter Stats ---")
print(f"  threshold={DISTANCE_THRESHOLD} (same-condition pilot, NOT generalization evidence)")
print(f"  n_probe_candidates_seen={fs['n_candidates_seen']}")
print(f"  n_probe_kept={fs['n_kept']}")
print(f"  n_probe_skipped_by_distance={fs['n_skipped_by_distance']}")
print(f"  kept_distance_mean={_mean(fs['kept_distances']):.6f}")
print(f"  skipped_distance_mean={_mean(fs['skipped_distances']):.6f}")
print(f"  kept_actions={fs['kept_actions']}")
print(f"  skipped_actions={fs['skipped_actions']}")
print(f"  skipped_probe_candidate_records:")
for r in fs["skipped_candidate_records"]:
    print(f"    {r['object_id']}  {r['action']}  dist={r['distance_to_threshold']:.6f}  "
          f"p={r['p_success']:.4f}")

# ======== Acceptance checks ========
comp_drop = c13_comp - dist_comp
passed = (
    dist_ub25 > c0b_ub25
    and dist_ub25 > c13_ub25
    and n_probe_dist < n_probe_c13
    and comp_drop <= 0.02
)

print()
print(f"--- Acceptance ---")
print(f"  C13_distance ub25 > C0b ub25:  {dist_ub25:.4f} > {c0b_ub25:.4f}  {'OK' if dist_ub25 > c0b_ub25 else 'FAIL'}")
print(f"  C13_distance ub25 > C13 ub25:  {dist_ub25:.4f} > {c13_ub25:.4f}  {'OK' if dist_ub25 > c13_ub25 else 'FAIL'}")
print(f"  C13_distance probes < C13 probes:  {n_probe_dist} < {n_probe_c13}  {'OK' if n_probe_dist < n_probe_c13 else 'FAIL'}")
print(f"  comp_drop_vs_C13 <= 0.02:  {comp_drop:.4f} <= 0.02  {'OK' if comp_drop <= 0.02 else 'FAIL'}")

print()
print("[block_done]")
print(f"  block_id=1F1")
print(f"  c0b_ub25={c0b_ub25:.4f}")
print(f"  c13_ub25={c13_ub25:.4f}")
print(f"  c13_distance_ub25={dist_ub25:.4f}")
print(f"  passed_pilot={'true' if passed else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ======== Save ========
block_out = {
    "block_id": "1F1",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040",
    "budget": budget,
    "cw": cw,
    "seed": seed,
    "distance_threshold": DISTANCE_THRESHOLD,
    "threshold_source": "Block 1F0c diagnostic_4 dist_median — same-condition pilot, NOT generalization evidence",
    "beta": BETA,
    "TOTAL_COMPOSITE_DECISIONS": TOTAL_COMPOSITE_DECISIONS,
    "policies": {
        "C0b": {
            "comp": c0b_comp,
            "ncost": c0b_ncost,
            "ub25": c0b_ub25,
            "total_cost": c0b_metrics["total_cost"],
            "normalized_cost": c0b_metrics["normalized_cost"],
            "visited_count": c0b_metrics["objects_visited"],
            "observe_count": c0b_metrics["objects_visited"],
            "probe_count": n_probe_c0b,
            "probe_coverage": c0b_metrics["probe_coverage"],
            "effective_probe_count": c0b_eff,
            "zero_gain_probe_count": c0b_zg,
            "harmful_probe_count": c0b_harm,
            "actual_probe_delta_sum": c0b_delta_sum,
        },
        "C13": {
            "comp": c13_comp,
            "ncost": c13_ncost,
            "ub25": c13_ub25,
            "total_cost": c13_metrics["total_cost"],
            "normalized_cost": c13_metrics["normalized_cost"],
            "visited_count": c13_metrics["objects_visited"],
            "observe_count": c13_metrics["objects_visited"],
            "probe_count": n_probe_c13,
            "probe_coverage": c13_metrics["probe_coverage"],
            "effective_probe_count": c13_eff,
            "zero_gain_probe_count": c13_zg,
            "harmful_probe_count": c13_harm,
            "actual_probe_delta_sum": c13_delta_sum,
            "probe_records": c13_labeled.probe_records,
        },
        "C13_distance_filter": {
            "comp": dist_comp,
            "ncost": dist_ncost,
            "ub25": dist_ub25,
            "total_cost": dist_metrics["total_cost"],
            "normalized_cost": dist_metrics["normalized_cost"],
            "visited_count": dist_metrics["objects_visited"],
            "observe_count": dist_metrics["objects_visited"],
            "probe_count": n_probe_dist,
            "probe_coverage": dist_metrics["probe_coverage"],
            "effective_probe_count": dist_eff,
            "zero_gain_probe_count": dist_zg,
            "harmful_probe_count": dist_harm,
            "actual_probe_delta_sum": dist_delta_sum,
            "probe_records": dist_policy.probe_records,
            "filter_stats": {
                "n_probe_candidates_seen": fs["n_candidates_seen"],
                "n_probe_kept": fs["n_kept"],
                "n_probe_skipped_by_distance": fs["n_skipped_by_distance"],
                "kept_distance_mean": _mean(fs["kept_distances"]),
                "skipped_distance_mean": _mean(fs["skipped_distances"]),
                "kept_actions": fs["kept_actions"],
                "skipped_actions": fs["skipped_actions"],
                "skipped_probe_candidate_records": fs["skipped_candidate_records"],
            },
        },
    },
    "acceptance": {
        "c13_distance_ub25_gt_c0b_ub25": dist_ub25 > c0b_ub25,
        "c13_distance_ub25_gt_c13_ub25": dist_ub25 > c13_ub25,
        "c13_distance_probes_lt_c13_probes": n_probe_dist < n_probe_c13,
        "comp_drop_vs_c13": comp_drop,
        "comp_drop_le_002": comp_drop <= 0.02,
        "passed_pilot": passed,
    },
}
with open("runs/calibration_block1f1_distance_filter_rollout.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1f1_distance_filter_rollout.json")
