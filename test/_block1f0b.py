"""Block 1F0b: Realizable posterior decision VOI — NO ground truth access."""
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
print("Block 1F0b: Realizable Posterior Decision VOI")
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
    return {qname: _check_query(probs, qname) for qname in _TASK_QUERIES}

def _task_answer_confidence(probs, query_name):
    """Model's own confidence in its task answer. NO ground truth.

    For each query, confidence = max(p_correct, 1-p_correct), where
    p_correct is estimated from affordance probabilities.
    For conjunctive queries (need_stone), uses min of conjunct confidences.
    """
    if query_name == "need_planks":
        p = probs.get("craft_plank_success", 0.5)
        return max(p, 1 - p)
    elif query_name == "need_stone":
        p1 = probs.get("mine_by_hand_success", 0.5)
        p2 = probs.get("mine_with_pickaxe_success", 0.5)
        return min(max(p1, 1 - p1), max(p2, 1 - p2))
    elif query_name == "need_food":
        p = probs.get("eat_success", 0.5)
        return max(p, 1 - p)
    elif query_name == "need_tool":
        p = probs.get("use_as_tool_success", 0.5)
        return max(p, 1 - p)
    elif query_name == "need_fuel":
        p = probs.get("burn_as_fuel_success", 0.5)
        return max(p, 1 - p)
    return 0.5

def _task_utility(probs):
    """Model's own estimate of task answer quality. Returns mean confidence across 5 queries."""
    confs = [_task_answer_confidence(probs, qname) for qname in _TASK_QUERIES]
    return sum(confs) / len(confs)

def _task_correct_count(probs, gt):
    correct = 0
    for qname, qfn in _TASK_QUERIES.items():
        if _check_query(probs, qname) == qfn(gt):
            correct += 1
    return correct

def _mean(xs): return sum(xs) / max(len(xs), 1)
def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0: return float('nan')
    if n % 2: return s[n//2]
    return (s[n//2-1] + s[n//2]) / 2


# ============ RealizableDecisionVOITracker ============
class RealizableDecisionVOITracker:
    """Computes oracle (GT) VOI for labeling, AND realizable VOI metrics.

    Realizable signals (NO ground truth):
      A. answer_change_prob
         P(at least one task answer flips) =
           p_success * I(answers_success != answers_before)
         + (1-p_success) * I(answers_fail != answers_before)

      B. posterior_expected_task_utility_delta
         task_utility = mean of model's own answer confidence (max(p,1-p))
         delta = p_success * (util_success - util_before)
               + (1-p_success) * (util_fail - util_before)
    """

    def __init__(self, inner_policy, gt_per_object, im):
        self._inner = inner_policy
        self._gt = gt_per_object  # only for LABELING (is_effective_probe, oracle_voi_gt)
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
            delta_correct = after_correct - pending["before_correct"]
            is_effective = delta_correct > 0

            rec = {
                "object_id": object_id,
                "action": action,
                "is_effective_probe": is_effective,  # label only
                "delta_task_correct": delta_correct,   # label only
                "before_correct": pending["before_correct"],
                "after_correct": after_correct,
            }
            for k, v in pending["voi_fields"].items():
                rec[k] = v
            self.voi_records.append(rec)
        self._pending_voi = None

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        gt = self._gt.get(object_id, {})
        fake_obj = {"id": object_id, "visible_features": features or {}}

        before_probs = self._im.predict_all_affordances(fake_obj)
        before_answers = _get_task_answers(before_probs)
        before_correct = _task_correct_count(before_probs, gt)  # for labeling only
        before_util = _task_utility(before_probs)

        # Entropy and distance for each action
        per_action_entropy = {}
        per_action_dist = {}
        for action in MAIN_CANDIDATE_ACTIONS:
            feature = ACTION_TO_FEATURE[action]
            prob = before_probs.get(feature, 0.5)
            per_action_entropy[action] = _binary_entropy(prob)
            per_action_dist[action] = abs(prob - 0.5)

        # Call inner.decide_probe
        norm_probe_cost = view.probe_cost / max(view.initial_budget, 0.001)
        should_probe, action = self._inner.decide_probe(view, object_id)

        if should_probe and action is not None:
            feature = ACTION_TO_FEATURE[action]
            p_success = before_probs.get(feature, 0.5)
            entropy = per_action_entropy[action]
            dist = per_action_dist[action]

            # ---- Simulate success ----
            im_succ = self._im.clone()
            im_succ.incorporate_probe(object_id, action, 1.0)
            probs_succ = im_succ.predict_all_affordances(fake_obj)
            answers_succ = _get_task_answers(probs_succ)
            correct_succ = _task_correct_count(probs_succ, gt)
            util_succ = _task_utility(probs_succ)

            # ---- Simulate fail ----
            im_fail = self._im.clone()
            im_fail.incorporate_probe(object_id, action, 0.0)
            probs_fail = im_fail.predict_all_affordances(fake_obj)
            answers_fail = _get_task_answers(probs_fail)
            correct_fail = _task_correct_count(probs_fail, gt)
            util_fail = _task_utility(probs_fail)

            # ---- Oracle VOI (GT, diagnostic upper bound only) ----
            delta_succ = correct_succ - before_correct
            delta_fail = correct_fail - before_correct
            oracle_decision_relevant_voi_gt = p_success * delta_succ + (1 - p_success) * delta_fail

            # ---- A. answer_change_prob (REALIZABLE — no GT) ----
            changed_succ = 1 if answers_succ != before_answers else 0
            changed_fail = 1 if answers_fail != before_answers else 0
            answer_change_prob = p_success * changed_succ + (1 - p_success) * changed_fail

            # ---- B. posterior_expected_task_utility_delta (REALIZABLE — no GT) ----
            posterior_expected_task_utility_delta = (
                p_success * (util_succ - before_util)
                + (1 - p_success) * (util_fail - before_util)
            )

            # ---- C13 VOI for selected action (recomputed) ----
            # Same formula as C13 decide_probe: net_voi = E[post_utility] - current_utility - cw * norm_probe_cost
            def _affordance_utility(probs_dict):
                return sum(max(p, 1-p) for p in probs_dict.values()) / max(len(probs_dict), 1)
            current_aff_util = _affordance_utility(before_probs)
            e_post_aff_util = (p_success * _affordance_utility(probs_succ)
                               + (1 - p_success) * _affordance_utility(probs_fail))
            selected_action_score = e_post_aff_util - current_aff_util - cw * norm_probe_cost

            voi_fields = {
                "entropy": round(entropy, 6),
                "distance_to_threshold": round(dist, 6),
                "selected_action_score": round(selected_action_score, 6),
                "p_success": round(p_success, 6),
                # Oracle (GT — diagnostic upper bound only)
                "oracle_decision_relevant_voi_gt": round(oracle_decision_relevant_voi_gt, 6),
                "oracle_uses_ground_truth": True,
                # Realizable signals (NO ground truth)
                "answer_change_prob": round(answer_change_prob, 6),
                "posterior_expected_task_utility_delta": round(posterior_expected_task_utility_delta, 6),
                # Budget
                "budget_remaining": round(view.budget_remaining, 6),
                # Counterfactual details
                "delta_if_success_gt": delta_succ,
                "delta_if_fail_gt": delta_fail,
                "util_before": round(before_util, 6),
                "util_if_success": round(util_succ, 6),
                "util_if_fail": round(util_fail, 6),
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

# ======== Run instrumented C13 ========
print("  Running C13 (CW=0.5) with RealizableDecisionVOITracker...")
env_c13 = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_c13 = im_base.clone()
inner_c13 = C13_InstanceVOIPolicy(im_c13, random.Random(seed + 550), cost_weight=cw)
tracker = RealizableDecisionVOITracker(inner_c13, gt_per_object, im_c13)
result_c13 = EpisodeHarness(env_c13, tracker).run()

voi_records = tracker.voi_records
n_probe = len(voi_records)
effective = [r for r in voi_records if r["is_effective_probe"]]
zero_gain = [r for r in voi_records if not r["is_effective_probe"]]
n_eff = len(effective)
n_zero = len(zero_gain)

# References
with open("runs/calibration_block1c_diagnostic.json") as f:
    b1c = json.load(f)
c0b_comp = 0.7033
c0b_ncost = b1c["costs"]["c0b_total"] / budget
c0b_ub25 = c0b_comp - 0.25 * c0b_ncost
c13_ncost = b1c["costs"]["c13_total"] / budget
c13_comp = c0b_comp + b1c["comp_gain"]
c13_ub25 = c13_comp - 0.25 * c13_ncost
TOTAL_COMPOSITE_DECISIONS = 60 * 5

# ======== 1. Distribution comparison ========
print()
print("=" * 70)
print("Block 1F0b — Realizable Posterior Decision VOI")
print(f"  (is_effective_probe and oracle_* are LABELS only — use GT)")
print("=" * 70)

signals = [
    ("oracle_decision_relevant_voi_gt",    "oracle VOI (GT, diagnostic upper bound)"),
    ("answer_change_prob",                  "answer_change_prob (REALIZABLE)"),
    ("posterior_expected_task_utility_delta","posterior_task_util_delta (REALIZABLE)"),
    ("selected_action_score",               "selected_action_score (C13 affordance VOI)"),
    ("entropy",                             "entropy"),
    ("distance_to_threshold",               "distance_to_threshold"),
]

print(f"\n--- 1. Distribution comparison (n_eff={n_eff}, n_zero={n_zero}) ---")
print(f"  {'metric':<42} {'eff_mean':<10} {'zero_mean':<10} "
      f"{'eff_med':<10} {'zero_med':<10} {'eff_min':<10} {'eff_max':<10} "
      f"{'zero_min':<10} {'zero_max':<10}")
for fname, flabel in signals:
    eff_vals = [r[fname] for r in effective]
    zero_vals = [r[fname] for r in zero_gain]
    print(f"  {flabel:<42} {_mean(eff_vals):<10.4f} {_mean(zero_vals):<10.4f} "
          f"{_median(eff_vals):<10.4f} {_median(zero_vals):<10.4f} "
          f"{min(eff_vals):<10.4f} {max(eff_vals):<10.4f} "
          f"{min(zero_vals):<10.4f} {max(zero_vals):<10.4f}")

# ======== 2. Per-probe detail ========
print(f"\n--- 2. Per-probe detail ---")
print(f"  {'oid':<16} {'action':<16} {'eff':<4} {'oracle':<8} {'chg_prob':<9} "
      f"{'util_delta':<11} {'sel_score':<10} {'entropy':<8} {'dist':<8} "
      f"{'d_succ':<6} {'d_fail':<6} {'budget':<8}")
for r in voi_records:
    print(f"  {r['object_id']:<16} {r['action']:<16} {r['is_effective_probe']!s:<4} "
          f"{r['oracle_decision_relevant_voi_gt']:<8.4f} "
          f"{r['answer_change_prob']:<9.4f} "
          f"{r['posterior_expected_task_utility_delta']:<11.6f} "
          f"{r['selected_action_score']:<10.6f} "
          f"{r['entropy']:<8.4f} {r['distance_to_threshold']:<8.4f} "
          f"{r['delta_if_success_gt']:<6} {r['delta_if_fail_gt']:<6} "
          f"{r['budget_remaining']:<8.4f}")

# ======== 3. Separability analysis ========
print(f"\n--- 3. Separability analysis ---")
print(f"  Effective probe has positive signal? YES=effective, NO=zero_gain")
print(f"  {'signal':<44} {'eff>0':<7} {'zero>0':<7} {'sep':<10}")
for fname, flabel in signals:
    if fname == "distance_to_threshold":
        # distance: lower = more uncertain = more likely effective
        eff_pos = sum(1 for r in effective if r[fname] < _median([rr[fname] for rr in voi_records]))
        zero_neg = sum(1 for r in zero_gain if r[fname] >= _median([rr[fname] for rr in voi_records]))
    elif fname == "entropy":
        eff_pos = sum(1 for r in effective if r[fname] > _median([rr[fname] for rr in voi_records]))
        zero_neg = sum(1 for r in zero_gain if r[fname] <= _median([rr[fname] for rr in voi_records]))
    else:
        eff_pos = sum(1 for r in effective if r[fname] > 0)
        zero_pos = sum(1 for r in zero_gain if r[fname] > 0)
    sign_sep = "SIGN" if (fname not in ("entropy", "distance_to_threshold")
                          and all(r[fname] > 0 for r in effective)
                          and all(r[fname] <= 0 for r in zero_gain)) else ""
    if not sign_sep:
        sign_sep = "SIGN" if (fname in ("entropy", "distance_to_threshold")
                              and False) else "partial"
    print(f"  {flabel:<44} {eff_pos}/{n_eff:<5} {zero_pos}/{n_zero:<5}  ", end="")
    if fname not in ("entropy", "distance_to_threshold"):
        eff_sign = sum(1 for r in effective if r[fname] > 0)
        zero_sign = sum(1 for r in zero_gain if r[fname] > 0)
        if eff_sign == n_eff and zero_sign == 0:
            print("PERFECT SIGN SEPARATION")
        elif eff_sign == n_eff and zero_sign <= n_zero//2:
            print(f"SIGN SEP (zero>0: {zero_sign}/{n_zero})")
        elif eff_sign >= n_eff//2 and zero_sign == 0:
            print(f"SIGN SEP (eff>0: {eff_sign}/{n_eff})")
        else:
            eff_neg = n_eff - eff_sign
            print(f"mixed: eff<=0={eff_neg}, zero>0={zero_sign}")
    else:
        print("(continuous, use median)")

# ======== 4. Realizable filter tests ========
print(f"\n--- 4. Realizable filter tests (posthoc_trajectory_estimate) ---")
print(f"  Conservative estimate: comp = {c13_comp:.4f} - skipped_effective_delta / {TOTAL_COMPOSITE_DECISIONS}")

def test_realizable_filter(name, keep_condition, records):
    kept = [r for r in records if keep_condition(r)]
    skipped = [r for r in records if not keep_condition(r)]
    n_kept = len(kept)
    n_skipped = len(skipped)
    kd_eff = sum(1 for r in kept if r["is_effective_probe"])
    sd_eff = sum(1 for r in skipped if r["is_effective_probe"])
    kd_zero = sum(1 for r in kept if not r["is_effective_probe"])
    kd_delta = sum(r["delta_task_correct"] for r in kept)
    sd_delta = sum(r["delta_task_correct"] for r in skipped)

    est_probe_cost = n_kept * config.PROBE_COST
    c13_visit_obs = b1c["costs"]["c13_visit"] + b1c["costs"]["c13_observe"]
    est_total_cost = c13_visit_obs + est_probe_cost
    est_ncost = est_total_cost / budget
    est_comp_conservative = c13_comp - sd_delta / TOTAL_COMPOSITE_DECISIONS
    est_ub25_conservative = est_comp_conservative - 0.25 * est_ncost

    return {
        "name": name,
        "label_as": "posthoc_trajectory_estimate (all)",
        "kept_count": n_kept, "skipped_count": n_skipped,
        "kept_effective": kd_eff, "skipped_effective": sd_eff,
        "kept_zero_gain": kd_zero, "skipped_zero_gain": n_skipped - sd_eff,
        "kept_effective_delta": kd_delta, "skipped_effective_delta": sd_delta,
        "estimated_probe_cost": round(est_probe_cost, 6),
        "estimated_total_cost": round(est_total_cost, 6),
        "estimated_normalized_cost": round(est_ncost, 6),
        "estimated_comp_conservative": round(est_comp_conservative, 6),
        "estimated_utility_beta0.25_conservative": round(est_ub25_conservative, 6),
        "utility_gain_over_C0b_conservative": round(est_ub25_conservative - c0b_ub25, 6),
        "utility_gain_over_C13_conservative": round(est_ub25_conservative - c13_ub25, 6),
    }

filters = [
    # F1: answer_change_prob > 0
    test_realizable_filter("F1: answer_change_prob > 0",
        lambda r: r["answer_change_prob"] > 0, voi_records),
    # F2: posterior_expected_task_utility_delta > 0
    test_realizable_filter("F2: posterior_task_util_delta > 0",
        lambda r: r["posterior_expected_task_utility_delta"] > 0, voi_records),
    # F3: answer_change_prob > 0 AND posterior_task_util_delta > 0
    test_realizable_filter("F3: chg_prob>0 AND util_delta>0",
        lambda r: r["answer_change_prob"] > 0 and r["posterior_expected_task_utility_delta"] > 0,
        voi_records),
    # F4: selected_action_score > 0 (C13's own gate)
    test_realizable_filter("F4: selected_action_score > 0",
        lambda r: r["selected_action_score"] > 0, voi_records),
    # Oracle reference
    test_realizable_filter("ORACLE: oracle_voi_gt > 0",
        lambda r: r["oracle_decision_relevant_voi_gt"] > 0, voi_records),
]

# Also: current C13 (keep all) reference
current_c13_filter = {
    "name": "REF: current_C13 (keep all)",
    "label_as": "REF",
    "kept_count": n_probe, "skipped_count": 0,
    "kept_effective": n_eff, "skipped_effective": 0,
    "kept_zero_gain": n_zero, "skipped_zero_gain": 0,
    "kept_effective_delta": sum(r["delta_task_correct"] for r in voi_records),
    "skipped_effective_delta": 0,
    "estimated_probe_cost": round(n_probe * config.PROBE_COST, 6),
    "estimated_total_cost": b1c["costs"]["c13_total"],
    "estimated_normalized_cost": round(c13_ncost, 6),
    "estimated_comp_conservative": round(c13_comp, 6),
    "estimated_utility_beta0.25_conservative": round(c13_ub25, 6),
    "utility_gain_over_C0b_conservative": round(c13_ub25 - c0b_ub25, 6),
    "utility_gain_over_C13_conservative": 0.0,
}

print(f"  {'filter':<42} {'kept':<5} {'skip':<5} {'k_eff':<5} {'s_eff':<5} "
      f"{'kE_d':<5} {'sE_d':<5} {'e_cost':<8} {'c_comp':<7} {'c_ub25':<8} "
      f"{'vC0b':<8}")
print(f"  {'':42} {'':5} {'':5} {'':5} {'':5} "
      f"{'':5} {'':5} {'':8} {'(consv)':<7} {'(consv)':<8} "
      f"{'(consv)':<8}")
for f in [current_c13_filter] + filters:
    label = " posthoc" if "posthoc" in f.get("label_as", "") else " REF"
    print(f"  {f['name']:<42} {f['kept_count']:<5} {f['skipped_count']:<5} "
          f"{f['kept_effective']:<5} {f['skipped_effective']:<5} "
          f"{f['kept_effective_delta']:<5} {f['skipped_effective_delta']:<5} "
          f"{f['estimated_total_cost']:<8.4f} "
          f"{f['estimated_comp_conservative']:<7.4f} "
          f"{f['estimated_utility_beta0.25_conservative']:<8.4f} "
          f"{f['utility_gain_over_C0b_conservative']:<+8.4f}{label}")

# ======== 5. Interpretability check ========
print(f"\n--- 5. Interpretability check ---")
print(f"  Are the realizable signals zero when oracle_voi_gt <= 0?")
zero_oracle = [r for r in voi_records if r["oracle_decision_relevant_voi_gt"] <= 0]
pos_oracle = [r for r in voi_records if r["oracle_decision_relevant_voi_gt"] > 0]
for flabel, fname in [("answer_change_prob", "answer_change_prob"),
                        ("posterior_util_delta", "posterior_expected_task_utility_delta")]:
    zero_mean = _mean([r[fname] for r in zero_oracle])
    pos_mean = _mean([r[fname] for r in pos_oracle])
    zero_pos = sum(1 for r in zero_oracle if r[fname] > 0)
    pos_pos = sum(1 for r in pos_oracle if r[fname] > 0)
    print(f"  {flabel}: oracle<=0 mean={zero_mean:.4f} (pos_rate={zero_pos}/{len(zero_oracle)}), "
          f"oracle>0 mean={pos_mean:.4f} (pos_rate={pos_pos}/{len(pos_oracle)})")

# ======== 6. Comparison with Block 1E distance filter ========
print(f"\n--- 6. Comparison with Block 1E distance filter ---")
dist_vals = [r["distance_to_threshold"] for r in voi_records]
dist_median = sorted(dist_vals)[len(dist_vals)//2]
dist_filter = test_realizable_filter("dist: distance_to_threshold <= median",
    lambda r, t=dist_median: r["distance_to_threshold"] <= t, voi_records)

print(f"  {'metric':<42} {'answer_chg>0':<16} {'util_delta>0':<16} {'both>0':<16} {'distance':<16}")
for label, f in [("F1", filters[0]), ("F2", filters[1]), ("F3", filters[2]), ("dist", dist_filter)]:
    print(f"  {f['name']:<42} "
          f"eff_ret={f['kept_effective']}/{n_eff}         "
          f"eff_ret={f['kept_effective']}/{n_eff}         "
          f"eff_ret={f['kept_effective']}/{n_eff}         "
          f"eff_ret={f['kept_effective']}/{n_eff}")
    break  # just header
print(f"  {'effective_retention':<42} "
      f"{filters[0]['kept_effective']/n_eff:.3f}              "
      f"{filters[1]['kept_effective']/n_eff:.3f}              "
      f"{filters[2]['kept_effective']/n_eff:.3f}              "
      f"{dist_filter['kept_effective']/n_eff:.3f}")
print(f"  {'zero_gain_skip_rate':<42} "
      f"{filters[0]['skipped_zero_gain']/n_zero:.3f}              "
      f"{filters[1]['skipped_zero_gain']/n_zero:.3f}              "
      f"{filters[2]['skipped_zero_gain']/n_zero:.3f}              "
      f"{dist_filter['skipped_zero_gain']/n_zero:.3f}")
print(f"  {'estimated_ub25_conservative':<42} "
      f"{filters[0]['estimated_utility_beta0.25_conservative']:<16.4f} "
      f"{filters[1]['estimated_utility_beta0.25_conservative']:<16.4f} "
      f"{filters[2]['estimated_utility_beta0.25_conservative']:<16.4f} "
      f"{dist_filter['estimated_utility_beta0.25_conservative']:<16.4f}")
print(f"  {'utility_gain_over_C0b':<42} "
      f"{filters[0]['utility_gain_over_C0b_conservative']:<+16.4f} "
      f"{filters[1]['utility_gain_over_C0b_conservative']:<+16.4f} "
      f"{filters[2]['utility_gain_over_C0b_conservative']:<+16.4f} "
      f"{dist_filter['utility_gain_over_C0b_conservative']:<+16.4f}")

print()
print("[block_done]")
print(f"  block_id=1F0b")
print(f"  elapsed={time.time()-t0:.0f}s")
print(f"  n_probe={n_probe}, n_eff={n_eff}, n_zero={n_zero}")
for f in filters:
    print(f"  {f['name']}: ub25={f['estimated_utility_beta0.25_conservative']:.4f} "
          f"vC0b={f['utility_gain_over_C0b_conservative']:+.4f} "
          f"eff_ret={f['kept_effective']}/{n_eff} "
          f"zero_skip={f['skipped_zero_gain']}/{n_zero}")

# Save
block_out = {
    "block_id": "1F0b",
    "elapsed_s": time.time() - t0,
    "cue": "C3_P060_O040", "budget": budget, "cw": cw, "seed": seed,
    "n_probe": n_probe, "n_effective": n_eff, "n_zero_gain": n_zero,
    "TOTAL_COMPOSITE_DECISIONS": TOTAL_COMPOSITE_DECISIONS,
    "C0b": {"comp": c0b_comp, "ncost": c0b_ncost, "ub25": c0b_ub25},
    "C13": {"comp": c13_comp, "ncost": c13_ncost, "ub25": c13_ub25},
    "note": (
        "answer_change_prob and posterior_expected_task_utility_delta are "
        "REALIZABLE — computed without ground truth. "
        "oracle_decision_relevant_voi_gt uses GT task_correct_count and is a "
        "diagnostic upper bound only. "
        "is_effective_probe is a post-hoc label, not an input to any filter."
    ),
    "distribution_comparison": [
        {"field": fname, "label": flabel,
         "eff_mean": _mean([r[fname] for r in effective]),
         "zero_mean": _mean([r[fname] for r in zero_gain]),
         "eff_median": _median([r[fname] for r in effective]),
         "zero_median": _median([r[fname] for r in zero_gain]),
         } for fname, flabel in signals
    ],
    "filters": filters,
    "current_c13_ref": current_c13_filter,
    "distance_filter_comparison": dist_filter,
    "per_action": {},
    "voi_records": voi_records,
}
with open("runs/calibration_block1f0b_realizable_voi.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1f0b_realizable_voi.json")
