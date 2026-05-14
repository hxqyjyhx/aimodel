"""Block 1F0: Oracle decision-relevant VOI — uses ground truth; diagnostic upper bound only."""
import sys, os, json, time, copy, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("runs", exist_ok=True)

t0 = time.time()

import config
from environment import MiniMCEnvironment, _TASK_QUERIES
from harness import EpisodeHarness
from policies import (
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

print("=" * 70)
print("Block 1F0: Decision-Relevant VOI Diagnostic")
print(f"  cue=C3_P060_O040, budget={budget}, CW={cw}, seed={seed}")
print("=" * 70)

# ============ Helpers ============
def _binary_entropy(p):
    p = max(0.0001, min(0.9999, p))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

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

def _get_task_answers(probs):
    """Return dict of query_name -> bool for all 5 task queries."""
    return {qname: _check_query(probs, qname) for qname in _TASK_QUERIES}

def _task_correct_count(probs, gt):
    correct = 0
    for qname, qfn in _TASK_QUERIES.items():
        if _check_query(probs, qname) == qfn(gt):
            correct += 1
    return correct

def _mean(xs): return sum(xs) / max(len(xs), 1)


# ============ DecisionVOITracker ============
class DecisionVOITracker:
    """Wraps C13 to compute decision-relevant VOI at each probe decision.

    oracle_decision_relevant_voi_gt =
        P(success) * (task_correct_after_success - task_correct_before)
      + P(fail)    * (task_correct_after_fail - task_correct_before)

    This is the EX ANTE expected task delta — computed BEFORE seeing the outcome.
    USES GROUND TRUTH: task_correct_count requires gt_per_object.
    This is a diagnostic upper bound, NOT an implementable policy signal.
    """

    def __init__(self, inner_policy, gt_per_object, im):
        self._inner = inner_policy
        self._gt = gt_per_object
        self._im = im
        self.voi_records = []

    def reset(self, view):       self._inner.reset(view)
    def select_next_object(self, view): return self._inner.select_next_object(view)
    def get_answer(self, view):  return self._inner.get_answer(view)
    def get_pre_probe_entropies(self):
        return getattr(self._inner, '_pre_probe_entropies', {})
    def get_pre_decision_entropies(self):
        return getattr(self._inner, '_pre_decision_entropies', {})

    def on_probe_result(self, view, object_id, action, outcome):
        self._inner.on_probe_result(view, object_id, action, outcome)

        pending = getattr(self, "_pending_voi", None)
        if pending is not None and pending["object_id"] == object_id:
            features = view.get_observed_features(object_id)
            fake_obj = {"id": object_id, "visible_features": features or {}}
            after_probs = self._im.predict_all_affordances(fake_obj)
            gt = pending["gt"]
            after_correct = _task_correct_count(after_probs, gt)
            before_correct = pending["before_correct"]
            delta_correct = after_correct - before_correct
            is_effective = delta_correct > 0

            rec = {
                "object_id": object_id,
                "action": action,
                "is_effective_probe": is_effective,
                "delta_task_correct": delta_correct,
                "before_correct": before_correct,
                "after_correct": after_correct,
            }
            # Merge pre-decision VOI fields
            for k, v in pending["voi_fields"].items():
                rec[k] = v
            self.voi_records.append(rec)
        self._pending_voi = None

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}

        before_probs = self._im.predict_all_affordances(fake_obj)
        before_correct = _task_correct_count(before_probs, gt)
        before_answers = _get_task_answers(before_probs)

        # ---- Per-action pre-probe features (same as Block 1E) ----
        norm_probe_cost = view.probe_cost / max(view.initial_budget, 0.001)

        # Compute entropy and distance for each action
        per_action_entropy = {}
        per_action_dist = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            feature = ACTION_TO_FEATURE[action]
            prob = before_probs.get(feature, 0.5)
            per_action_entropy[action] = _binary_entropy(prob)
            per_action_dist[action] = abs(prob - 0.5)

        # Now call inner.decide_probe
        should_probe, action = self._inner.decide_probe(view, object_id)

        if should_probe and action is not None:
            feature = ACTION_TO_FEATURE[action]
            p_success = before_probs.get(feature, 0.5)
            entropy = per_action_entropy[action]
            dist = per_action_dist[action]

            # ---- Simulate success outcome ----
            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            probs_succ = im_succ.predict_all_affordances(fake_obj)
            answers_succ = _get_task_answers(probs_succ)
            correct_succ = _task_correct_count(probs_succ, gt)

            # ---- Simulate fail outcome ----
            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            probs_fail = im_fail.predict_all_affordances(fake_obj)
            answers_fail = _get_task_answers(probs_fail)
            correct_fail = _task_correct_count(probs_fail, gt)

            # ---- Compute decision-relevant VOI ----
            delta_succ = correct_succ - before_correct
            delta_fail = correct_fail - before_correct

            # Expected task delta (ex ante)
            expected_task_delta = p_success * delta_succ + (1 - p_success) * delta_fail

            # Probability that at least one task answer changes
            any_change_succ = 1 if answers_succ != before_answers else 0
            any_change_fail = 1 if answers_fail != before_answers else 0
            predicted_answer_change_prob = p_success * any_change_succ + (1 - p_success) * any_change_fail

            # Oracle decision-relevant VOI (uses GT task_correct_count)
            # USES GROUND TRUTH — diagnostic upper bound only
            oracle_decision_relevant_voi_gt = expected_task_delta

            # Beta * probe_cost
            BETA = 0.25
            beta_probe_cost = BETA * config.PROBE_COST  # 0.25 * 0.05 = 0.0125

            would_probe_by_oracle_voi = oracle_decision_relevant_voi_gt > beta_probe_cost

            voi_fields = {
                "entropy": round(entropy, 6),
                "distance_to_threshold": round(dist, 6),
                "selected_action_score": "N/A (C13 net_voi not stored here)",
                "p_success": round(p_success, 6),
                "delta_if_success": delta_succ,
                "delta_if_fail": delta_fail,
                "expected_task_delta": round(expected_task_delta, 6),
                "predicted_answer_change_prob": round(predicted_answer_change_prob, 6),
                "oracle_decision_relevant_voi_gt": round(oracle_decision_relevant_voi_gt, 6),
                "oracle_uses_ground_truth": True,
                "beta_probe_cost": round(beta_probe_cost, 6),
                "beta": BETA,
                "would_probe_by_oracle_voi": would_probe_by_oracle_voi,
                "before_correct": before_correct,
                "correct_if_success": correct_succ,
                "correct_if_fail": correct_fail,
                "answers_before": before_answers,
                "answers_if_success": answers_succ,
                "answers_if_fail": answers_fail,
                "budget_remaining": round(view.budget_remaining, 6),
            }

            self._pending_voi = {
                "object_id": object_id,
                "action": action,
                "gt": gt,
                "before_correct": before_correct,
                "voi_fields": voi_fields,
            }
        else:
            self._pending_voi = None

        return should_probe, action


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

# ======== Run instrumented C13 with DecisionVOITracker ========
print("  Running C13 (CW=0.5) with DecisionVOITracker...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
inner_c13 = C13_InstanceVOIPolicy(im_c13, random.Random(seed + 550), cost_weight=cw)
tracker = DecisionVOITracker(inner_c13, gt_per_object, im_c13)
result_c13 = EpisodeHarness(env_c13, tracker).run()

voi_records = tracker.voi_records
n_probe = len(voi_records)
effective = [r for r in voi_records if r["is_effective_probe"]]
zero_gain = [r for r in voi_records if not r["is_effective_probe"]]
n_eff = len(effective)
n_zero = len(zero_gain)

# C0b and C13 references from Block 1C
with open("runs/calibration_block1c_diagnostic.json") as f:
    b1c = json.load(f)
c0b_comp = 0.7033
c0b_ncost = b1c["costs"]["c0b_total"] / budget
c0b_ub25 = c0b_comp - 0.25 * c0b_ncost

c13_ncost = b1c["costs"]["c13_total"] / budget
c13_comp = c0b_comp + b1c["comp_gain"]
c13_ub25 = c13_comp - 0.25 * c13_ncost

TOTAL_COMPOSITE_DECISIONS = 60 * 5  # 300

# ======== 1. Distribution: oracle_decision_relevant_voi_gt in effective vs zero-gain ========
voi_eff = [r["oracle_decision_relevant_voi_gt"] for r in effective]
voi_zero = [r["oracle_decision_relevant_voi_gt"] for r in zero_gain]
exp_delta_eff = [r["expected_task_delta"] for r in effective]
exp_delta_zero = [r["expected_task_delta"] for r in zero_gain]
p_change_eff = [r["predicted_answer_change_prob"] for r in effective]
p_change_zero = [r["predicted_answer_change_prob"] for r in zero_gain]

print()
print("=" * 70)
print("Block 1F0 — Decision-Relevant VOI Diagnostic")
print("=" * 70)

print(f"\n--- 1A. oracle_decision_relevant_voi_gt distribution ---")
print(f"  {'metric':<35} {'eff_mean':<10} {'zero_mean':<10} "
      f"{'eff_med':<10} {'zero_med':<10} {'eff_min':<10} {'eff_max':<10} "
      f"{'zero_min':<10} {'zero_max':<10}")
for label, eff_vals, zero_vals in [
    ("oracle_decision_relevant_voi_gt", voi_eff, voi_zero),
    ("expected_task_delta", exp_delta_eff, exp_delta_zero),
    ("predicted_answer_change_prob", p_change_eff, p_change_zero),
]:
    print(f"  {label:<35} {_mean(eff_vals):<10.4f} {_mean(zero_vals):<10.4f} "
          f"{sorted(eff_vals)[len(eff_vals)//2]:<10.4f} {sorted(zero_vals)[len(zero_vals)//2]:<10.4f} "
          f"{min(eff_vals):<10.4f} {max(eff_vals):<10.4f} "
          f"{min(zero_vals):<10.4f} {max(zero_vals):<10.4f}")

print(f"\n--- 1B. Per-probe detail ---")
print(f"  {'oid':<14} {'action':<16} {'eff':<4} {'entropy':<9} {'dist':<9} "
      f"{'exp_delta':<10} {'p_chg':<8} {'d_voi':<8} {'beta*c':<8} {'would':<5} "
      f"{'d_succ':<7} {'d_fail':<7} {'b_corr':<7}")
for r in voi_records:
    print(f"  {r['object_id']:<14} {r['action']:<16} {r['is_effective_probe']!s:<4} "
          f"{r['entropy']:<9.4f} {r['distance_to_threshold']:<9.4f} "
          f"{r['expected_task_delta']:<10.4f} {r['predicted_answer_change_prob']:<8.4f} "
          f"{r['oracle_decision_relevant_voi_gt']:<8.4f} {r['beta_probe_cost']:<8.4f} "
          f"{r['would_probe_by_oracle_voi']!s:<5} "
          f"{r['delta_if_success']:<7} {r['delta_if_fail']:<7} "
          f"{r['before_correct']:<7}")

# ======== 2. Decision-VOI filter ========
print(f"\n--- 2. oracle_decision_relevant_voi_gt filter ---")
kept_by_dvoi = [r for r in voi_records if r["would_probe_by_oracle_voi"]]
skipped_by_dvoi = [r for r in voi_records if not r["would_probe_by_oracle_voi"]]
n_kd = len(kept_by_dvoi)
n_sd = len(skipped_by_dvoi)
kd_eff = sum(1 for r in kept_by_dvoi if r["is_effective_probe"])
sd_eff = sum(1 for r in skipped_by_dvoi if r["is_effective_probe"])
kd_zero = sum(1 for r in kept_by_dvoi if not r["is_effective_probe"])
kd_delta = sum(r["delta_task_correct"] for r in kept_by_dvoi)
sd_delta = sum(r["delta_task_correct"] for r in skipped_by_dvoi)

print(f"  keeps  {n_kd}/{n_probe} probes ({kd_eff} effective, {kd_zero} zero-gain)")
print(f"  skips  {n_sd}/{n_probe} probes ({sd_eff} effective, {n_sd - sd_eff} zero-gain)")
print(f"  effective retention: {kd_eff}/{n_eff} ({kd_eff/max(n_eff,1):.1%})")
print(f"  zero-gain skipped:   {n_sd - sd_eff}/{n_zero} ({(n_sd - sd_eff)/max(n_zero,1):.1%})")
print(f"  kept_effective_delta={kd_delta}, skipped_effective_delta={sd_delta}")

# ======== 3. Conservative utility estimate ========
print(f"\n--- 3. Conservative utility_beta0.25 estimate ---")
est_probe_cost_dvoi = n_kd * config.PROBE_COST
c13_visit_obs = b1c["costs"]["c13_visit"] + b1c["costs"]["c13_observe"]
est_total_cost_dvoi = c13_visit_obs + est_probe_cost_dvoi
est_ncost_dvoi = est_total_cost_dvoi / budget
est_comp_dvoi = c13_comp - sd_delta / TOTAL_COMPOSITE_DECISIONS
est_ub25_dvoi = est_comp_dvoi - 0.25 * est_ncost_dvoi

print(f"  (conservative_trajectory_conditional_estimate)")
print(f"    n_probes = {n_kd}/{n_probe}")
print(f"    estimated_total_cost = {est_total_cost_dvoi:.4f}")
print(f"    estimated_normalized_cost = {est_ncost_dvoi:.4f}")
print(f"    estimated_comp = {est_comp_dvoi:.4f}  (= {c13_comp:.4f} - {sd_delta}/{TOTAL_COMPOSITE_DECISIONS})")
print(f"    estimated_utility_beta0.25 = {est_ub25_dvoi:.4f}")
print(f"    utility_gain_over_C0b = {est_ub25_dvoi - c0b_ub25:+.4f}")
print(f"    utility_gain_over_C13 = {est_ub25_dvoi - c13_ub25:+.4f}")

# ======== 4. Comparison: decision-VOI vs distance_to_threshold ========
print(f"\n--- 4. oracle_decision_relevant_voi_gt vs distance_to_threshold comparison ---")
# distance filter (same as Block 1E filter B)
dist_vals = [r["distance_to_threshold"] for r in voi_records]
dist_median = sorted(dist_vals)[len(dist_vals)//2]
kept_by_dist = [r for r in voi_records if r["distance_to_threshold"] <= dist_median]
skipped_by_dist = [r for r in voi_records if r["distance_to_threshold"] > dist_median]
n_kdist = len(kept_by_dist)
n_sdist = len(skipped_by_dist)
kdist_eff = sum(1 for r in kept_by_dist if r["is_effective_probe"])
sdist_eff = sum(1 for r in skipped_by_dist if r["is_effective_probe"])
kdist_delta = sum(r["delta_task_correct"] for r in kept_by_dist)
sdist_delta = sum(r["delta_task_correct"] for r in skipped_by_dist)

# Conservative for distance filter
est_pc_dist = n_kdist * config.PROBE_COST
est_tc_dist = c13_visit_obs + est_pc_dist
est_nc_dist = est_tc_dist / budget
est_comp_dist = c13_comp - sdist_delta / TOTAL_COMPOSITE_DECISIONS
est_ub25_dist = est_comp_dist - 0.25 * est_nc_dist

print(f"  {'metric':<38} {'decision_VOI':<16} {'distance':<16}")
print(f"  {'':38} {'':16} {'(median threshold)':<16}")
print(f"  {'probes_kept':<38} {n_kd:<16} {n_kdist:<16}")
print(f"  {'effective_kept':<38} {kd_eff:<16} {kdist_eff:<16}")
print(f"  {'zero_gain_skipped':<38} {n_sd - sd_eff:<16} {n_sdist - sdist_eff:<16}")
print(f"  {'effective_retention':<38} {kd_eff/max(n_eff,1):.3f}           {kdist_eff/max(n_eff,1):.3f}")
print(f"  {'zero_gain_skip_rate':<38} {(n_sd - sd_eff)/max(n_zero,1):.3f}           {(n_sdist - sdist_eff)/max(n_zero,1):.3f}")
print(f"  {'--- conservative estimates ---':<38}")
print(f"  {'estimated_comp':<38} {est_comp_dvoi:<16.4f} {est_comp_dist:<16.4f}")
print(f"  {'estimated_total_cost':<38} {est_total_cost_dvoi:<16.4f} {est_tc_dist:<16.4f}")
print(f"  {'estimated_ub25':<38} {est_ub25_dvoi:<16.4f} {est_ub25_dist:<16.4f}")
print(f"  {'utility_gain_over_C0b':<38} {est_ub25_dvoi - c0b_ub25:<+16.4f} {est_ub25_dist - c0b_ub25:<+16.4f}")
print(f"  {'utility_gain_over_C13':<38} {est_ub25_dvoi - c13_ub25:<+16.4f} {est_ub25_dist - c13_ub25:<+16.4f}")

# Overlap analysis
dvoi_oids = {r["object_id"] for r in kept_by_dvoi}
dist_oids = {r["object_id"] for r in kept_by_dist}
overlap = dvoi_oids & dist_oids
dvoi_only = dvoi_oids - dist_oids
dist_only = dist_oids - dvoi_oids
neither = {r["object_id"] for r in voi_records} - dvoi_oids - dist_oids
print(f"\n  Overlap: both={len(overlap)}, dVOI_only={len(dvoi_only)}, "
      f"dist_only={len(dist_only)}, neither={len(neither)}")

# ======== 5. Summary ========
print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
if est_ub25_dvoi > c0b_ub25:
    print(f"  decision_VOI > C0b: {est_ub25_dvoi:.4f} > {c0b_ub25:.4f}  "
          f"(delta={est_ub25_dvoi - c0b_ub25:+.4f})")
else:
    print(f"  decision_VOI <= C0b: {est_ub25_dvoi:.4f} <= {c0b_ub25:.4f}")
print(f"  C0b  ub25 = {c0b_ub25:.4f}")
print(f"  C13  ub25 = {c13_ub25:.4f}")
print(f"  dVOI ub25 = {est_ub25_dvoi:.4f}")
print(f"  dist ub25 = {est_ub25_dist:.4f}")

# Per-action: which actions have zero expected_task_delta?
print(f"\n  Per-action expected_task_delta distribution:")
action_voi = {}
for r in voi_records:
    act = r["action"]
    if act not in action_voi:
        action_voi[act] = []
    action_voi[act].append(r["expected_task_delta"])
for act in sorted(action_voi):
    vals = action_voi[act]
    n_zero_delta = sum(1 for v in vals if v == 0.0)
    print(f"    {act:<18} n={len(vals):<3} mean={_mean(vals):.4f} "
          f"n_zero_delta={n_zero_delta}/{len(vals)}")

print()
print("[block_done]")
print(f"  block_id=1F0")
print(f"  elapsed={time.time()-t0:.0f}s")
print(f"  n_probe={n_probe}, n_eff={n_eff}, n_zero={n_zero}")
print(f"  dVOI_kept={n_kd}, dVOI_skipped={n_sd}")
print(f"  dVOI_ub25={est_ub25_dvoi:.4f}")
print(f"  dist_ub25={est_ub25_dist:.4f}")
print(f"  c0b_ub25={c0b_ub25:.4f}, c13_ub25={c13_ub25:.4f}")

# Save
block_out = {
    "block_id": "1F0",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": cw, "seed": seed,
    "n_probe": n_probe, "n_effective": n_eff, "n_zero_gain": n_zero,
    "TOTAL_COMPOSITE_DECISIONS": TOTAL_COMPOSITE_DECISIONS,
    "C0b": {"comp": c0b_comp, "ncost": c0b_ncost, "ub25": c0b_ub25},
    "C13": {"comp": c13_comp, "ncost": c13_ncost, "ub25": c13_ub25},
    "beta": 0.25,
    "beta_probe_cost": 0.25 * config.PROBE_COST,
    "oracle_decision_relevant_voi_gt_formula": (
        "oracle_decision_relevant_voi_gt = "
        "P(success) * (task_correct_after_success - task_correct_before) + "
        "P(fail) * (task_correct_after_fail - task_correct_before)"
    ),
    "voi_records": voi_records,
    "decision_voi_filter": {
        "kept_count": n_kd, "skipped_count": n_sd,
        "kept_effective": kd_eff, "skipped_effective": sd_eff,
        "kept_zero_gain": kd_zero, "skipped_zero_gain": n_sd - sd_eff,
        "kept_effective_delta": kd_delta, "skipped_effective_delta": sd_delta,
        "estimated_probe_cost": round(est_probe_cost_dvoi, 6),
        "estimated_total_cost": round(est_total_cost_dvoi, 6),
        "estimated_normalized_cost": round(est_ncost_dvoi, 6),
        "estimated_comp_conservative": round(est_comp_dvoi, 6),
        "estimated_ub25_conservative": round(est_ub25_dvoi, 6),
        "utility_gain_over_C0b": round(est_ub25_dvoi - c0b_ub25, 6),
    },
    "distance_filter_comparison": {
        "kept_count": n_kdist, "skipped_count": n_sdist,
        "kept_effective": kdist_eff, "skipped_effective": sdist_eff,
        "kept_effective_delta": kdist_delta, "skipped_effective_delta": sdist_delta,
        "estimated_comp_conservative": round(est_comp_dist, 6),
        "estimated_ub25_conservative": round(est_ub25_dist, 6),
        "utility_gain_over_C0b": round(est_ub25_dist - c0b_ub25, 6),
    },
    "overlap_analysis": {
        "dvoi_only": sorted(dvoi_only),
        "dist_only": sorted(dist_only),
        "both": sorted(overlap),
        "neither": sorted(neither),
    },
    "per_action_expected_delta": {
        act: {"n": len(vals), "mean": _mean(vals), "n_zero": sum(1 for v in vals if v == 0.0)}
        for act, vals in action_voi.items()
    },
}
with open("runs/calibration_block1f0_decision_voi.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1f0_decision_voi.json")
