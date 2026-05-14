"""Block 1C: Deep diagnostic — why C13's probes yield only +0.0233 comp gain."""
import sys, os, json, time, copy, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

import config
from environment import MiniMCEnvironment
from harness import EpisodeHarness, HarnessResult
from evaluator import MiniMCEvaluator, _TASK_QUERIES
from policies import (
    C0b_ObserveOnlyPolicy, C13_InstanceVOIPolicy,
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
)

# Import training pipeline
PARENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, PARENT_DIR)
from instance_outcome_memory import InstanceOutcomeMemory
from run_004_5n1 import run_phase_a_training
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60, "p_other": 0.40, "absent": False})
budget = 1.5

print("=" * 70)
print("Block 1C: C13 Probe Efficacy Diagnostic")
print(f"  cue=C3_P060_O040, budget={budget}, seed={seed}")
print("=" * 70)

# =========================================================================
# Helpers
# =========================================================================
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

def _composite_correct_count(probs, gt):
    """Count how many of 5 task queries match between probs and gt."""
    correct = 0
    for qname, qfn in _TASK_QUERIES.items():
        pred_match = _check_query(probs, qname)
        true_match = qfn(gt)
        if pred_match == true_match:
            correct += 1
    return correct

def per_object_composite_accuracy(per_object, gt_per_object):
    correct = 0
    total = 0
    per_obj_correct = {}
    for oid, probs in per_object.items():
        gt = gt_per_object.get(oid, {})
        obj_correct = 0
        obj_total = 0
        for qname, qfn in _TASK_QUERIES.items():
            pred_match = _check_query(probs, qname)
            true_match = qfn(gt)
            if pred_match == true_match:
                correct += 1
                obj_correct += 1
            total += 1
            obj_total += 1
        per_obj_correct[oid] = obj_correct / max(obj_total, 1)
    return correct / max(total, 1), per_obj_correct

def _probs_from_im(im, oid, features):
    """Get affordance probabilities from an IOM for a given object."""
    fake_obj = {"id": oid, "visible_features": features or {}}
    return im.predict_all_affordances(fake_obj)


# =========================================================================
# ProbeTracker — instrumented C13 policy wrapper
# =========================================================================
class ProbeTracker:
    """Wraps a C13_InstanceVOIPolicy to record before/after state per probe."""

    def __init__(self, inner_policy, gt_per_object):
        self._inner = inner_policy
        self._gt = gt_per_object
        # probe_records: list of dicts per probe event
        self.probe_records = []

    # Delegate all harness-facing methods
    def reset(self, view):
        self._inner.reset(view)

    def select_next_object(self, view):
        return self._inner.select_next_object(view)

    def get_answer(self, view):
        return self._inner.get_answer(view)

    def get_pre_probe_entropies(self):
        if hasattr(self._inner, 'get_pre_probe_entropies'):
            return self._inner.get_pre_probe_entropies()
        return getattr(self._inner, '_pre_probe_entropies', {})

    def get_pre_decision_entropies(self):
        if hasattr(self._inner, 'get_pre_decision_entropies'):
            return self._inner.get_pre_decision_entropies()
        return getattr(self._inner, '_pre_decision_entropies', {})

    def decide_probe(self, view, object_id):
        # Capture before-state from inner IOM
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        before_probs = _probs_from_im(self._inner._im, object_id, features)
        before_comp = _composite_correct_count(before_probs, gt)

        # Store in a temp slot so on_probe_result can access it
        self._pending_before = {
            "object_id": object_id,
            "action": None,  # filled after decide
            "before_probs": before_probs,
            "gt": gt,
            "before_composite_correct": before_comp,
        }

        should_probe, action = self._inner.decide_probe(view, object_id)
        if should_probe and self._pending_before is not None:
            self._pending_before["action"] = action
        else:
            self._pending_before = None
        return should_probe, action

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)

        pending = self._pending_before
        if pending is not None and pending["object_id"] == object_id:
            features = view.get_observed_features(object_id)
            after_probs = _probs_from_im(self._inner._im, object_id, features)
            gt = pending["gt"]
            after_comp = _composite_correct_count(after_probs, gt)
            delta_comp = after_comp - pending["before_composite_correct"]

            # Feature-level changes
            changed_features = []
            changed_to_correct = 0
            changed_to_wrong = 0
            for f in CORE_ACTION_FEATURES:
                before_pred = 1.0 if pending["before_probs"].get(f, 0.5) >= 0.5 else 0.0
                after_pred = 1.0 if after_probs.get(f, 0.5) >= 0.5 else 0.0
                true_val = gt.get(f, 0.0)
                if before_pred != after_pred:
                    changed_features.append(f)
                    if after_pred == true_val:
                        changed_to_correct += 1
                    else:
                        changed_to_wrong += 1

            self.probe_records.append({
                "object_id": object_id,
                "action": action,
                "before_probs": {k: round(v, 6) for k, v in pending["before_probs"].items()},
                "after_probs": {k: round(v, 6) for k, v in after_probs.items()},
                "gt": {k: int(v) for k, v in gt.items()},
                "before_composite_correct_count": pending["before_composite_correct"],
                "after_composite_correct_count": after_comp,
                "delta_composite_correct": delta_comp,
                "changed_features": changed_features,
                "changed_to_correct": changed_to_correct,
                "changed_to_wrong": changed_to_wrong,
            })
            self._pending_before = None


# =========================================================================
# Phase A: Training
# =========================================================================
print("\n  Phase A: Training...")
(student, base_learner, train_objects, train_env,
 test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)

train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
test_oids = sorted(test_objects)
test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

# Phase A': Sparse outcomes
outcome_rows, actual_coverage = collect_sparse_probe_outcomes(
    train_objects_dict, student, config.COVERAGE, seed
)

# Phase B: IOM
posterior_visible_features = list(base_learner.visible_feature_names)
im_base = InstanceOutcomeMemory(
    train_objects_dict, posterior_visible_features,
    k=config.INSTANCE_K, similarity_power=config.SIMILARITY_POWER,
    similarity_mode=config.SIMILARITY_MODE,
)
im_base.build(outcome_rows)

# Phase D: Positions
from simulator_truth import MiniMCSimulatorTruth
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS,
    config.AGENT_START, rng
)

# Ground truth for all test objects (use simulator to get proper float values)
_temp_sim = MiniMCSimulatorTruth(test_objects_dict, positions, config.AGENT_START)
gt_per_object = {}
for oid in test_oids:
    gt_per_object[oid] = _temp_sim.get_ground_truth_affordances(oid)

# =========================================================================
# Run C0b
# =========================================================================
print("\n  Running C0b...")
env_c0b = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c0b = im_base.clone()
policy_c0b = C0b_ObserveOnlyPolicy(im_c0b, random.Random(seed + 100))
result_c0b = EpisodeHarness(env_c0b, policy_c0b).run()
eval_c0b = MiniMCEvaluator(env_c0b)
metrics_c0b = eval_c0b.evaluate(
    result_c0b.predictions, result_c0b.event_log, result_c0b.agent_obs)
c0b_per_object = result_c0b.predictions.get("per_object", {})
c0b_events = result_c0b.event_log.to_list()

c0b_visited_oids = set()
for e in c0b_events:
    if e.get("event") == "reach":
        c0b_visited_oids.add(e.get("object_id"))

# =========================================================================
# Run C13 with ProbeTracker (CW=0.5)
# =========================================================================
print("  Running C13 (CW=0.5, instrumented)...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
inner_c13 = C13_InstanceVOIPolicy(im_c13, random.Random(seed + 550), cost_weight=0.5)
tracker_c13 = ProbeTracker(inner_c13, gt_per_object)
result_c13 = EpisodeHarness(env_c13, tracker_c13).run()
eval_c13 = MiniMCEvaluator(env_c13)
metrics_c13 = eval_c13.evaluate(
    result_c13.predictions, result_c13.event_log, result_c13.agent_obs,
    pre_probe_entropies=result_c13.pre_probe_entropies,
    pre_decision_entropies=result_c13.pre_decision_entropies,
)
c13_per_object = result_c13.predictions.get("per_object", {})
c13_events = result_c13.event_log.to_list()
probe_records = tracker_c13.probe_records

# Parse C13 event log for visited/probed sets
c13_probed_oids = set()
c13_visited_oids = set()
c13_probe_actions = {}
for e in c13_events:
    if e.get("event") == "reach":
        c13_visited_oids.add(e.get("object_id"))
    elif e.get("event") == "probe":
        oid = e.get("object_id")
        c13_probed_oids.add(oid)
        c13_probe_actions.setdefault(oid, []).append(e.get("action", "?"))

all_test_oids = set(test_oids)
c13_unprobed_oids = all_test_oids - c13_probed_oids

# =========================================================================
# Run C13 episodes for CW=1.0 and CW=1.5 (actual full runs)
# =========================================================================
cw_results = {}
for cw_val in [1.0, 1.5]:
    print(f"  Running C13 (CW={cw_val:.1f})...")
    env_cw = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
    im_cw = im_base.clone()
    inner_cw = C13_InstanceVOIPolicy(im_cw, random.Random(seed + 550 + int(cw_val * 100)),
                                     cost_weight=cw_val)
    tracker_cw = ProbeTracker(inner_cw, gt_per_object)
    result_cw = EpisodeHarness(env_cw, tracker_cw).run()
    eval_cw = MiniMCEvaluator(env_cw)
    metrics_cw = eval_cw.evaluate(
        result_cw.predictions, result_cw.event_log, result_cw.agent_obs)
    cw_events = result_cw.event_log.to_list()
    cw_probed = set()
    cw_visited = set()
    for e in cw_events:
        if e.get("event") == "reach":
            cw_visited.add(e.get("object_id"))
        elif e.get("event") == "probe":
            cw_probed.add(e.get("object_id"))
    cw_results[cw_val] = {
        "comp": metrics_cw["composite_task_accuracy"],
        "ncost": metrics_cw["normalized_cost"],
        "probed_n": len(cw_probed),
        "probed_oids": sorted(cw_probed),
        "visited_n": len(cw_visited),
        "visited_oids": sorted(cw_visited),
        "probe_records": tracker_cw.probe_records,
    }

# =========================================================================
# Compute metrics
# =========================================================================
c0b_comp_obj, c0b_per_obj_acc = per_object_composite_accuracy(c0b_per_object, gt_per_object)
c13_comp_obj, c13_per_obj_acc = per_object_composite_accuracy(c13_per_object, gt_per_object)

# --- 1. Probed vs unprobed comp ---
probed_correct = 0
probed_total = 0
unprobed_correct = 0
unprobed_total = 0
for oid in c13_probed_oids:
    probed_correct += int(c13_per_obj_acc.get(oid, 0) * 5)
    probed_total += 5
for oid in c13_unprobed_oids:
    unprobed_correct += int(c13_per_obj_acc.get(oid, 0) * 5)
    unprobed_total += 5

probed_comp = probed_correct / max(probed_total, 1)
unprobed_comp = unprobed_correct / max(unprobed_total, 1)

# C0b comp on same object subsets
c0b_probed_correct = 0
c0b_probed_total = 0
c0b_unprobed_correct = 0
c0b_unprobed_total = 0
for oid in c13_probed_oids:
    c0b_probed_correct += int(c0b_per_obj_acc.get(oid, 0) * 5)
    c0b_probed_total += 5
for oid in c13_unprobed_oids:
    c0b_unprobed_correct += int(c0b_per_obj_acc.get(oid, 0) * 5)
    c0b_unprobed_total += 5

c0b_probed_comp = c0b_probed_correct / max(c0b_probed_total, 1)
c0b_unprobed_comp = c0b_unprobed_correct / max(c0b_unprobed_total, 1)

# --- 2a. C13 final vs C0b final differences (NOT causal probe effect) ---
n_probe_total = sum(len(v) for v in c13_probe_actions.values())
n_c13_diff_from_c0b = 0
n_c13_diff_to_correct_vs_c0b = 0
n_c13_diff_to_wrong_vs_c0b = 0

for oid in c13_probed_oids:
    c0b_probs = c0b_per_object.get(oid, {})
    c13_probs = c13_per_object.get(oid, {})
    gt = gt_per_object.get(oid, {})

    for f in CORE_ACTION_FEATURES:
        c0b_pred = 1.0 if c0b_probs.get(f, 0.5) >= 0.5 else 0.0
        c13_pred = 1.0 if c13_probs.get(f, 0.5) >= 0.5 else 0.0
        true_val = gt.get(f, 0.0)

        if c0b_pred != c13_pred:
            n_c13_diff_from_c0b += 1
            if c13_pred == true_val:
                n_c13_diff_to_correct_vs_c0b += 1
            else:
                n_c13_diff_to_wrong_vs_c0b += 1

net_c13_diff_vs_c0b = n_c13_diff_to_correct_vs_c0b - n_c13_diff_to_wrong_vs_c0b

# --- 2b. True probe before/after from ProbeTracker ---
total_probe_before_after = len(probe_records)
total_before_comp = sum(r["before_composite_correct_count"] for r in probe_records)
total_after_comp = sum(r["after_composite_correct_count"] for r in probe_records)
total_delta_comp = sum(r["delta_composite_correct"] for r in probe_records)
total_changed_features = sum(len(r["changed_features"]) for r in probe_records)
total_changed_to_correct = sum(r["changed_to_correct"] for r in probe_records)
total_changed_to_wrong = sum(r["changed_to_wrong"] for r in probe_records)
net_before_after_gain = total_changed_to_correct - total_changed_to_wrong

# --- 3. Cost decomposition ---
c0b_visit_cost = sum(e.get("cost", 0) for e in c0b_events if e.get("event") == "reach")
c0b_obs_cost = sum(e.get("cost", 0) for e in c0b_events if e.get("event") == "observe")
c13_visit_cost = sum(e.get("cost", 0) for e in c13_events if e.get("event") == "reach")
c13_obs_cost = sum(e.get("cost", 0) for e in c13_events if e.get("event") == "observe")
c13_probe_cost = sum(e.get("cost", 0) for e in c13_events if e.get("event") == "probe")
total_cost_c13 = c13_visit_cost + c13_obs_cost + c13_probe_cost
total_cost_c0b = c0b_visit_cost + c0b_obs_cost

comp_gain = metrics_c13["composite_task_accuracy"] - metrics_c0b["composite_task_accuracy"]

# --- 4. C0b baseline ---
c0b_visited_all = len(c0b_visited_oids) == len(test_oids)

# --- 5. CW sensitivity: real probed object ID comparison ---
cw05_probed = c13_probed_oids
cw10_probed = set(cw_results[1.0]["probed_oids"])
cw15_probed = set(cw_results[1.5]["probed_oids"])

cw_pairwise_jaccard = {}
for (lbl_a, set_a), (lbl_b, set_b) in [
    (("CW=0.5", cw05_probed), ("CW=1.0", cw10_probed)),
    (("CW=0.5", cw05_probed), ("CW=1.5", cw15_probed)),
    (("CW=1.0", cw10_probed), ("CW=1.5", cw15_probed)),
]:
    union = len(set_a | set_b)
    inter = len(set_a & set_b)
    cw_pairwise_jaccard[f"{lbl_a}_vs_{lbl_b}"] = {
        "intersection": inter,
        "union": union,
        "jaccard": inter / max(union, 1),
        "a_only": sorted(set_a - set_b),
        "b_only": sorted(set_b - set_a),
    }

cw_diagnostic = {
    "CW=0.5": {"n_probed": len(cw05_probed), "n_visited": len(c13_visited_oids),
               "comp": metrics_c13["composite_task_accuracy"],
               "ncost": metrics_c13["normalized_cost"]},
    "CW=1.0": {"n_probed": len(cw10_probed), "n_visited": cw_results[1.0]["visited_n"],
               "comp": cw_results[1.0]["comp"], "ncost": cw_results[1.0]["ncost"]},
    "CW=1.5": {"n_probed": len(cw15_probed), "n_visited": cw_results[1.5]["visited_n"],
               "comp": cw_results[1.5]["comp"], "ncost": cw_results[1.5]["ncost"]},
    "all_identical": (cw05_probed == cw10_probed == cw15_probed),
    "pairwise_jaccard": cw_pairwise_jaccard,
}

# =========================================================================
# Output
# =========================================================================
print()
print("=" * 70)
print("Block 1C — Diagnostic Results")
print("=" * 70)

print(f"\n1. C13 probed vs unprobed composite accuracy:")
print(f"   probed objects (n={len(c13_probed_oids)}):")
print(f"     C13 comp = {probed_comp:.4f}")
print(f"     C0b comp = {c0b_probed_comp:.4f}")
print(f"     C13 gain over C0b = {probed_comp - c0b_probed_comp:+.4f}")
print(f"   unprobed objects (n={len(c13_unprobed_oids)}):")
print(f"     C13 comp = {unprobed_comp:.4f}")
print(f"     C0b comp = {c0b_unprobed_comp:.4f}")
print(f"     C13 gain over C0b = {unprobed_comp - c0b_unprobed_comp:+.4f}")

print(f"\n2a. C13 final vs C0b final (NOT causal probe effect):")
print(f"   n_probe_total = {n_probe_total}")
print(f"   n_c13_final_diff_from_c0b = {n_c13_diff_from_c0b}")
print(f"   n_c13_final_diff_to_correct_vs_c0b = {n_c13_diff_to_correct_vs_c0b}")
print(f"   n_c13_final_diff_to_wrong_vs_c0b = {n_c13_diff_to_wrong_vs_c0b}")
print(f"   net_c13_final_gain_vs_c0b = {net_c13_diff_vs_c0b:+d}")
print(f"   diff_rate = {n_c13_diff_from_c0b}/{n_probe_total * len(CORE_ACTION_FEATURES)} "
      f"= {n_c13_diff_from_c0b / max(n_probe_total * len(CORE_ACTION_FEATURES), 1):.4f}")

print(f"\n2b. True probe before/after (instrumented):")
print(f"   n_probe_records = {total_probe_before_after}")
print(f"   total_before_composite_correct = {total_before_comp}")
print(f"   total_after_composite_correct = {total_after_comp}")
print(f"   total_delta_composite_correct = {total_delta_comp:+d}")
print(f"   total_changed_features = {total_changed_features}")
print(f"   changed_to_correct = {total_changed_to_correct}")
print(f"   changed_to_wrong = {total_changed_to_wrong}")
print(f"   net_before_after_gain = {net_before_after_gain:+d}")
if total_probe_before_after > 0:
    print(f"   avg_delta_comp_per_probe = {total_delta_comp / total_probe_before_after:+.4f}")
    print(f"   avg_changed_features_per_probe = {total_changed_features / total_probe_before_after:.2f}")
    print(f"   avg_changed_to_correct_per_probe = {total_changed_to_correct / total_probe_before_after:.2f}")
    print(f"   avg_changed_to_wrong_per_probe = {total_changed_to_wrong / total_probe_before_after:.2f}")

# Per-probe detail (first 30 and any with delta != 0)
print(f"\n   Per-probe detail:")
print(f"   {'oid':<22} {'action':<22} {'before_c':<9} {'after_c':<7} {'delta':<6} "
      f"{'#chg':<5} {'+cor':<4} {'-wrg':<4}")
nonzero_delta = [r for r in probe_records if r["delta_composite_correct"] != 0]
zero_delta = [r for r in probe_records if r["delta_composite_correct"] == 0]
for r in nonzero_delta + zero_delta[:min(10, len(zero_delta))]:
    print(f"   {r['object_id']:<22} {r['action']:<22} "
          f"{r['before_composite_correct_count']:<9} {r['after_composite_correct_count']:<7} "
          f"{r['delta_composite_correct']:+<6d} "
          f"{len(r['changed_features']):<5} {r['changed_to_correct']:<4} {r['changed_to_wrong']:<4}")
if len(zero_delta) > 10:
    print(f"   ... and {len(zero_delta) - 10} more with delta=0")

print(f"\n3. Cost decomposition:")
print(f"   C13 visit_cost   = {c13_visit_cost:.4f}  (normalized={c13_visit_cost/budget:.4f})")
print(f"   C13 observe_cost = {c13_obs_cost:.4f}  (normalized={c13_obs_cost/budget:.4f})")
print(f"   C13 probe_cost   = {c13_probe_cost:.4f}  (normalized={c13_probe_cost/budget:.4f})")
print(f"   C13 total_cost   = {total_cost_c13:.4f}")
print(f"   C0b total_cost   = {total_cost_c0b:.4f}")
print(f"   comp_gain = {comp_gain:+.4f}")
print(f"   comp_gain_per_total_cost = {comp_gain / total_cost_c13:.4f}")
print(f"   comp_gain_per_probe_cost = {comp_gain / max(c13_probe_cost, 0.001):.4f}")

print(f"\n4. C0b baseline diagnostics:")
print(f"   C0b visited_count = {len(c0b_visited_oids)}/{len(test_oids)}")
print(f"   C0b visited_all = {c0b_visited_all}")
print(f"   C0b comp = {metrics_c0b['composite_task_accuracy']:.4f}")
print(f"   C0b total_cost = {metrics_c0b['total_cost']:.4f}")
print(f"   C0b normalized_cost = {metrics_c0b['normalized_cost']:.4f}")

visible_feature_coverage = 0
for oid in c0b_visited_oids:
    obs_features = env_c0b.simulator._objects[oid].get("visible_features", {})
    visible_feature_coverage += len(obs_features)
print(f"   C0b observes visible_feature entries across {len(c0b_visited_oids)} objects "
      f"(avg {visible_feature_coverage/max(len(c0b_visited_oids),1):.1f}/obj)")

print(f"\n5. CW sensitivity diagnosis (real episode runs):")
print(f"   CW=0.5: probed={len(cw05_probed)}, visited={len(c13_visited_oids)}, "
      f"comp={metrics_c13['composite_task_accuracy']:.4f}, ncost={metrics_c13['normalized_cost']:.4f}")
print(f"   CW=1.0: probed={len(cw10_probed)}, visited={cw_results[1.0]['visited_n']}, "
      f"comp={cw_results[1.0]['comp']:.4f}, ncost={cw_results[1.0]['ncost']:.4f}")
print(f"   CW=1.5: probed={len(cw15_probed)}, visited={cw_results[1.5]['visited_n']}, "
      f"comp={cw_results[1.5]['comp']:.4f}, ncost={cw_results[1.5]['ncost']:.4f}")
print(f"   All identical probed sets: {cw_diagnostic['all_identical']}")
print(f"   Pairwise Jaccard:")
for pair, info in cw_pairwise_jaccard.items():
    print(f"     {pair}: J={info['jaccard']:.4f}  |intersection|={info['intersection']}  "
          f"|union|={info['union']}")
    if info['a_only']:
        print(f"       a_only: {info['a_only'][:5]}{'...' if len(info['a_only']) > 5 else ''}")
    if info['b_only']:
        print(f"       b_only: {info['b_only'][:5]}{'...' if len(info['b_only']) > 5 else ''}")

# Composite score change breakdown: per-action
print(f"\n6. Per-action probe efficacy (CW=0.5):")
action_stats = {}
for r in probe_records:
    act = r["action"]
    if act not in action_stats:
        action_stats[act] = {"n": 0, "delta_comp": 0, "n_changed": 0, "+cor": 0, "-wrg": 0}
    s = action_stats[act]
    s["n"] += 1
    s["delta_comp"] += r["delta_composite_correct"]
    s["n_changed"] += len(r["changed_features"])
    s["+cor"] += r["changed_to_correct"]
    s["-wrg"] += r["changed_to_wrong"]
print(f"   {'action':<24} {'n':<5} {'delta_comp':<10} {'avg_delta':<10} "
      f"{'n_changed':<10} {'+cor':<5} {'-wrg':<5}")
for act in sorted(action_stats):
    s = action_stats[act]
    print(f"   {act:<24} {s['n']:<5} {s['delta_comp']:+<10d} "
          f"{s['delta_comp']/s['n']:+.4f}     {s['n_changed']:<10} {s['+cor']:<5} {s['-wrg']:<5}")

# Summary
print()
print("[block_done]")
print(f"  block_id=1C")
print(f"  elapsed={time.time()-t0:.0f}s")
print(f"  n_probed={len(c13_probed_oids)}, n_unprobed={len(c13_unprobed_oids)}")
print(f"  probed_comp={probed_comp:.4f}, unprobed_comp={unprobed_comp:.4f}")
print(f"  comp_gain={comp_gain:+.4f}")
print(f"  n_c13_diff_from_c0b={n_c13_diff_from_c0b}, net_c13_diff_vs_c0b={net_c13_diff_vs_c0b:+d}")
print(f"  n_probe_before_after={total_probe_before_after}, net_before_after_gain={net_before_after_gain:+d}")
print(f"  probe_cost={c13_probe_cost:.4f}")
print(f"  CW_all_identical_sets={cw_diagnostic['all_identical']}")

# Save
block_out = {
    "block_id": "1C",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "seed": seed,
    "probed_n": len(c13_probed_oids), "unprobed_n": len(c13_unprobed_oids),
    "probed_comp": probed_comp, "unprobed_comp": unprobed_comp,
    "c0b_probed_comp": c0b_probed_comp, "c0b_unprobed_comp": c0b_unprobed_comp,
    "comp_gain": comp_gain,
    # C13 vs C0b differences (NOT causal probe effect)
    "c13_vs_c0b": {
        "n_diff_from_c0b": n_c13_diff_from_c0b,
        "n_diff_to_correct_vs_c0b": n_c13_diff_to_correct_vs_c0b,
        "n_diff_to_wrong_vs_c0b": n_c13_diff_to_wrong_vs_c0b,
        "net_diff_vs_c0b": net_c13_diff_vs_c0b,
    },
    # True probe before/after
    "probe_before_after": {
        "n_records": total_probe_before_after,
        "total_before_comp": total_before_comp,
        "total_after_comp": total_after_comp,
        "total_delta_comp": total_delta_comp,
        "total_changed_features": total_changed_features,
        "changed_to_correct": total_changed_to_correct,
        "changed_to_wrong": total_changed_to_wrong,
        "net_gain": net_before_after_gain,
        "per_probe": probe_records,
    },
    "costs": {
        "c13_visit": c13_visit_cost, "c13_observe": c13_obs_cost,
        "c13_probe": c13_probe_cost, "c13_total": total_cost_c13,
        "c0b_total": total_cost_c0b,
    },
    "c0b_visited_all": c0b_visited_all,
    "cw_diagnostic": cw_diagnostic,
    "per_action_efficacy": {act: {
        "n": s["n"], "delta_comp": s["delta_comp"],
        "avg_delta": s["delta_comp"] / s["n"] if s["n"] else 0,
        "n_changed": s["n_changed"], "correct": s["+cor"], "wrong": s["-wrg"],
    } for act, s in action_stats.items()},
}
with open("runs/calibration_block1c_diagnostic.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1c_diagnostic.json")
