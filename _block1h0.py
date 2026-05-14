"""Block 1H0: Balanced Evaluation + Query-Balanced Environment Audit.

Part A: Adds balanced evaluation metrics (per-query balanced accuracy,
macro balanced accuracy, class coverage) to every baseline.

Part B: Implements query-balanced evaluation via reweighting (Option 1)
— per-query balanced accuracy = 0.5*(pos_recall + neg_recall) inherently
reweights each class equally without changing object generation.

Also constructs per-query balanced subsets (Option 2) for verification.
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
    CORE_ACTION_FEATURES, MAIN_CANDIDATE_ACTIONS, ACTION_TO_FEATURE,
)
from simulator_truth import MiniMCSimulatorTruth
from instance_outcome_memory import InstanceOutcomeMemory
from sparse_outcome_collector import collect_sparse_probe_outcomes

seed = 101
budget = 1.5
QUERY_NAMES = ["need_planks", "need_stone", "need_food", "need_tool", "need_fuel"]


# ============================================================
# GT helpers
# ============================================================
def _profile_to_gt_affordances(profile):
    result = {}
    for action, raw in profile.items():
        if action == "tap_sound":
            result["tap_sound_clear"] = 1.0 if raw == "clear" else 0.0
        else:
            feature = ACTION_TO_FEATURE.get(action, action)
            result[feature] = 1.0 if raw == "success" else 0.0
    for feature in CORE_ACTION_FEATURES:
        if feature not in result:
            result[feature] = 0.0
    return result


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


# ============================================================
# Balanced evaluation framework (Part A)
# ============================================================
def compute_balanced_metrics(gt_labels, preds_by_oid_query, query_names, test_oids):
    """Compute raw AND balanced metrics for a set of predictions.

    gt_labels: {oid: {qname: bool}}
    preds_by_oid_query: {oid: {qname: bool}}
    query_names: list of query names
    test_oids: list of test object IDs

    Returns dict with raw and balanced metrics.
    """
    n_total_pairs = len(test_oids) * len(query_names)

    # Per-query accumulators
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

    # Raw composite accuracy
    raw_comp_acc = raw_correct / max(n_total_pairs, 1)

    # Per-query metrics
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

        # Balanced accuracy: handle missing classes explicitly
        has_pos = pos_total > 0
        has_neg = neg_total > 0
        per_query_class_coverage[qname] = {
            "n_positive": pos_total,
            "n_negative": neg_total,
            "has_both_classes": has_pos and has_neg,
            "positive_only": has_pos and not has_neg,
            "negative_only": has_neg and not has_pos,
        }

        if has_pos and has_neg:
            per_query_bal_acc[qname] = 0.5 * (pos_recall + neg_recall)
        elif has_pos:
            per_query_bal_acc[qname] = pos_recall
        elif has_neg:
            per_query_bal_acc[qname] = neg_recall
        else:
            per_query_bal_acc[qname] = 0.0

    # Macro metrics (unweighted mean across queries)
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
# Per-query balanced subset evaluation (Part B, Option 2)
# ============================================================
def compute_balanced_subset_metrics(gt_labels, preds_by_oid_query, query_names,
                                    test_oids, rng, n_resamples=20):
    """For each query, construct a balanced subset (equal pos/neg) and evaluate.

    Since positive objects are limited (min 15), we use all positives and
    sample an equal number of negatives. Repeated resampling gives a stable estimate.
    """
    # Identify positive/negative objects per query
    pos_oids_per_query = {}
    neg_oids_per_query = {}
    for qname in query_names:
        pos_oids = [oid for oid in test_oids if gt_labels[oid][qname]]
        neg_oids = [oid for oid in test_oids if not gt_labels[oid][qname]]
        pos_oids_per_query[qname] = pos_oids
        neg_oids_per_query[qname] = neg_oids

    per_query_bal_subset_acc = {}
    per_query_bal_subset_std = {}

    for qname in query_names:
        pos_oids = pos_oids_per_query[qname]
        neg_oids = neg_oids_per_query[qname]
        n_balanced = min(len(pos_oids), len(neg_oids))
        if n_balanced == 0:
            # Single-class query: report raw accuracy
            all_oids = pos_oids + neg_oids
            acc = sum(1 for oid in all_oids
                     if preds_by_oid_query[oid][qname] == gt_labels[oid][qname]) / max(len(all_oids), 1)
            per_query_bal_subset_acc[qname] = acc
            per_query_bal_subset_std[qname] = 0.0
            continue

        accs = []
        for _ in range(n_resamples):
            sampled_pos = rng.sample(pos_oids, n_balanced)
            sampled_neg = rng.sample(neg_oids, n_balanced)
            sampled_oids = sampled_pos + sampled_neg
            correct = sum(1 for oid in sampled_oids
                         if preds_by_oid_query[oid][qname] == gt_labels[oid][qname])
            accs.append(correct / (2 * n_balanced))

        per_query_bal_subset_acc[qname] = sum(accs) / len(accs)
        # Std of the mean
        if len(accs) > 1:
            mean = per_query_bal_subset_acc[qname]
            variance = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
            per_query_bal_subset_std[qname] = (variance / len(accs)) ** 0.5
        else:
            per_query_bal_subset_std[qname] = 0.0

    macro_bal_subset_acc = sum(per_query_bal_subset_acc.values()) / len(query_names)
    return {
        "per_query_bal_subset_acc": per_query_bal_subset_acc,
        "per_query_bal_subset_std": per_query_bal_subset_std,
        "macro_bal_subset_acc": macro_bal_subset_acc,
        "per_query_n_balanced": {
            qname: min(len(pos_oids_per_query[qname]), len(neg_oids_per_query[qname]))
            for qname in query_names
        },
    }


# ============================================================
# Main
# ============================================================
print("=" * 70)
print("Block 1H0: Balanced Evaluation + Query-Balanced Environment Audit")
print(f"  budget={budget}, seed={seed}")
print("=" * 70)

# ---- Phase A (original condition) ----
print("\n  Phase A: Training (original C3_P060_O040)...")
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

# ---- GT labels ----
gt_labels = {}
gt_affordances = {}
for oid in test_oids:
    profile = test_objects_dict[oid]["hidden_affordance_profile"]
    gt = _profile_to_gt_affordances(profile)
    gt_affordances[oid] = gt
    gt_labels[oid] = {}
    for qname in QUERY_NAMES:
        gt_labels[oid][qname] = _TASK_QUERIES[qname](gt)

# ---- Per-query class distribution ----
print(f"\n  Per-query class distribution:")
print(f"  {'query':<16} {'n_pos':>6} {'n_neg':>6} {'pos_rate':>9} {'neg_rate':>9}")
per_query_stats = {}
for qname in QUERY_NAMES:
    n_pos = sum(1 for oid in test_oids if gt_labels[oid][qname])
    n_neg = n_test - n_pos
    per_query_stats[qname] = {"n_pos": n_pos, "n_neg": n_neg,
                              "pos_rate": n_pos / n_test, "neg_rate": n_neg / n_test}
    print(f"  {qname:<16} {n_pos:>6} {n_neg:>6} "
          f"{n_pos / n_test:>9.4f} {n_neg / n_test:>9.4f}")

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

# ---- Positions for C_obs_full ----
positions = MiniMCSimulatorTruth.assign_positions(
    test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

# ============================================================
# Build all 7 baselines
# ============================================================
from collections import Counter

# --- Baseline 1: all_false ---
all_false_preds = {oid: {q: False for q in QUERY_NAMES} for oid in test_oids}

# --- Baseline 2: all_true ---
all_true_preds = {oid: {q: True for q in QUERY_NAMES} for oid in test_oids}

# --- Baseline 3: per_query_majority ---
pq_majority_label = {}
for qname in QUERY_NAMES:
    pq_majority_label[qname] = per_query_stats[qname]["n_pos"] >= per_query_stats[qname]["n_neg"]
pq_majority_preds = {oid: {q: pq_majority_label[q] for q in QUERY_NAMES}
                     for oid in test_oids}

# --- Baseline 4: category_majority ---
# Compute per-(category, query) majority from training objects
cat_query_train = {}  # (cat, qname) -> (n_pos, n_total)
for tid in train_objects:
    cat = train_objects_dict[tid]["hidden_category"]
    profile = train_objects_dict[tid]["hidden_affordance_profile"]
    gt = _profile_to_gt_affordances(profile)
    for qname in QUERY_NAMES:
        key = (cat, qname)
        cat_query_train.setdefault(key, [0, 0])
        cat_query_train[key][1] += 1
        if _TASK_QUERIES[qname](gt):
            cat_query_train[key][0] += 1

cat_majority_label = {}
for (cat, qname), (n_pos, n_total) in cat_query_train.items():
    cat_majority_label[(cat, qname)] = n_pos >= (n_total - n_pos)

cat_majority_preds = {}
cat_maj_unmatched = Counter()
for oid in test_oids:
    cat = test_objects_dict[oid]["hidden_category"]
    cat_majority_preds[oid] = {}
    for qname in QUERY_NAMES:
        key = (cat, qname)
        if key in cat_majority_label:
            cat_majority_preds[oid][qname] = cat_majority_label[key]
        else:
            cat_majority_preds[oid][qname] = pq_majority_label.get(qname, False)
            cat_maj_unmatched[qname] += 1

# --- Baseline 5: IOM prior (empty features) ---
im_prior = im_base.clone()
iom_prior_preds = {}
for oid in test_oids:
    fake_obj = {"id": oid, "visible_features": {}}
    probs = im_prior.predict_all_affordances(fake_obj)
    iom_prior_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}

# --- Baseline 6: C_obs_full (C0b ObserveOnlyPolicy) ---
print("  Running C_obs_full...")
env_full = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
im_full = im_base.clone()
rng_full = random.Random(seed + 100)
full_result = EpisodeHarness(env_full, C0b_ObserveOnlyPolicy(im_full, rng_full)).run()
full_per_object = full_result.predictions.get("per_object", {})
cobs_preds = {}
for oid in test_oids:
    probs = full_per_object.get(oid, {})
    cobs_preds[oid] = {q: _check_query(probs, q) for q in QUERY_NAMES}
# Also get cost metrics from the evaluator
full_evaluator = MiniMCEvaluator(env_full)
full_cost_metrics = full_evaluator.evaluate(
    full_result.predictions, full_result.event_log, full_result.agent_obs, {}, {})

# --- Baseline 7: C_oracle_unconstrained (ground truth) ---
oracle_preds = {oid: {q: gt_labels[oid][q] for q in QUERY_NAMES} for oid in test_oids}

# ============================================================
# Compute balanced metrics for ALL baselines
# ============================================================
baseline_names = ["all_false", "all_true", "per_query_majority",
                  "category_majority", "iom_prior", "c_obs_full",
                  "oracle_unconstrained"]
baseline_preds = {
    "all_false": all_false_preds,
    "all_true": all_true_preds,
    "per_query_majority": pq_majority_preds,
    "category_majority": cat_majority_preds,
    "iom_prior": iom_prior_preds,
    "c_obs_full": cobs_preds,
    "oracle_unconstrained": oracle_preds,
}

all_metrics = {}
bal_subset_rng = random.Random(seed + 9999)
for bname in baseline_names:
    preds = baseline_preds[bname]
    metrics = compute_balanced_metrics(gt_labels, preds, QUERY_NAMES, test_oids)
    # Add balanced subset metrics (Option 2 verification)
    bal_subset = compute_balanced_subset_metrics(
        gt_labels, preds, QUERY_NAMES, test_oids, bal_subset_rng)
    metrics["bal_subset"] = bal_subset
    all_metrics[bname] = metrics

# ============================================================
# Print results
# ============================================================

# ---- Table 1: Baseline summary (raw vs balanced) ----
print(f"\n{'='*70}")
print(f"Baseline Summary: Raw vs Balanced Metrics")
print(f"{'='*70}")
print(f"  {'baseline':<22} {'raw_comp':>9} {'macro_q_acc':>11} {'macro_bal':>9} "
      f"{'bal_subset':>10} {'Δ(raw-bal)':>11}")
for bname in baseline_names:
    m = all_metrics[bname]
    delta = m["raw_comp_acc"] - m["macro_query_bal_acc"]
    print(f"  {bname:<22} {m['raw_comp_acc']:>9.4f} {m['macro_query_acc']:>11.4f} "
          f"{m['macro_query_bal_acc']:>9.4f} {m['bal_subset']['macro_bal_subset_acc']:>10.4f} "
          f"{delta:>+11.4f}")

# ---- Table 2: Per-query balanced accuracy for each baseline ----
print(f"\n{'='*70}")
print(f"Per-Query Balanced Accuracy")
print(f"{'='*70}")
header = f"  {'baseline':<22}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
header += f" {'macro_bal':>10}"
print(header)
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
for bname in baseline_names:
    m = all_metrics[bname]
    row = f"  {bname:<22}"
    for q in QUERY_NAMES:
        row += f" {m['per_query'][q]['balanced_accuracy']:>12.4f}"
    row += f" {m['macro_query_bal_acc']:>10.4f}"
    print(row)

# ---- Table 3: Per-query positive recall for each baseline ----
print(f"\n{'='*70}")
print(f"Per-Query Positive Recall")
print(f"{'='*70}")
header = f"  {'baseline':<22}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
print(header)
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for bname in baseline_names:
    m = all_metrics[bname]
    row = f"  {bname:<22}"
    for q in QUERY_NAMES:
        row += f" {m['per_query'][q]['positive_recall']:>12.4f}"
    print(row)

# ---- Table 4: Per-query negative recall for each baseline ----
print(f"\n{'='*70}")
print(f"Per-Query Negative Recall")
print(f"{'='*70}")
header = f"  {'baseline':<22}"
for q in QUERY_NAMES:
    header += f" {q:>12}"
print(header)
print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for bname in baseline_names:
    m = all_metrics[bname]
    row = f"  {bname:<22}"
    for q in QUERY_NAMES:
        row += f" {m['per_query'][q]['negative_recall']:>12.4f}"
    print(row)

# ---- Table 5: Class coverage per query ----
print(f"\n{'='*70}")
print(f"Per-Query Class Coverage (GT labels)")
print(f"{'='*70}")
sample_m = all_metrics["all_false"]  # class coverage is same for all baselines
print(f"  {'query':<16} {'n_pos':>6} {'n_neg':>6} {'has_both':>9} {'n_balanced_subset':>18}")
for qname in QUERY_NAMES:
    cc = sample_m["per_query"][qname]["class_coverage"]
    n_bal = all_metrics["all_false"]["bal_subset"]["per_query_n_balanced"][qname]
    print(f"  {qname:<16} {cc['n_positive']:>6} {cc['n_negative']:>6} "
          f"{str(cc['has_both_classes']):>9} {n_bal:>18}")

# ---- Key comparisons for balanced evaluation ----
print(f"\n{'='*70}")
print(f"Key Comparisons: Balanced Evaluation")
print(f"{'='*70}")

# Extract key values
all_false_raw = all_metrics["all_false"]["raw_comp_acc"]
all_false_macro_bal = all_metrics["all_false"]["macro_query_bal_acc"]
iom_prior_raw = all_metrics["iom_prior"]["raw_comp_acc"]
iom_prior_macro_bal = all_metrics["iom_prior"]["macro_query_bal_acc"]
cobs_raw = all_metrics["c_obs_full"]["raw_comp_acc"]
cobs_macro_bal = all_metrics["c_obs_full"]["macro_query_bal_acc"]
oracle_raw = all_metrics["oracle_unconstrained"]["raw_comp_acc"]
oracle_macro_bal = all_metrics["oracle_unconstrained"]["macro_query_bal_acc"]
cat_maj_raw = all_metrics["category_majority"]["raw_comp_acc"]
cat_maj_macro_bal = all_metrics["category_majority"]["macro_query_bal_acc"]

print(f"  all_false:                raw={all_false_raw:.4f}  macro_bal={all_false_macro_bal:.4f}")
print(f"  iom_prior:                raw={iom_prior_raw:.4f}  macro_bal={iom_prior_macro_bal:.4f}")
print(f"  category_majority:        raw={cat_maj_raw:.4f}  macro_bal={cat_maj_macro_bal:.4f}")
print(f"  c_obs_full:               raw={cobs_raw:.4f}  macro_bal={cobs_macro_bal:.4f}")
print(f"  oracle_unconstrained:     raw={oracle_raw:.4f}  macro_bal={oracle_macro_bal:.4f}")

# Delta analysis
print(f"\n  --- Balanced metric deltas ---")
print(f"  iom_prior vs all_false (macro_bal):  {iom_prior_macro_bal - all_false_macro_bal:+.4f}")
print(f"  c_obs_full vs iom_prior (macro_bal): {cobs_macro_bal - iom_prior_macro_bal:+.4f}")
print(f"  oracle vs c_obs_full (macro_bal):    {oracle_macro_bal - cobs_macro_bal:+.4f}")
print(f"  oracle vs all_false (macro_bal):     {oracle_macro_bal - all_false_macro_bal:+.4f}")
print(f"  cat_maj vs iom_prior (macro_bal):    {cat_maj_macro_bal - iom_prior_macro_bal:+.4f}")

# ---- Diagnostic checks ----
print(f"\n{'='*70}")
print(f"Diagnostic Checks")
print(f"{'='*70}")

# Check 1: all_false should have macro_bal ≈ 0.5
all_false_bal_ok = abs(all_false_macro_bal - 0.5) < 0.01
print(f"  all_false macro_bal ≈ 0.5:        {all_false_bal_ok}  ({all_false_macro_bal:.4f})")

# Check 2: IOM prior should NOT look strong under balanced metrics
iom_prior_weak = iom_prior_macro_bal < 0.55
print(f"  iom_prior not strong (bal<0.55):  {iom_prior_weak}  ({iom_prior_macro_bal:.4f})")

# Check 3: Oracle should remain high under balanced metrics
oracle_high = oracle_macro_bal >= 0.95
print(f"  oracle macro_bal >= 0.95:         {oracle_high}  ({oracle_macro_bal:.4f})")

# Check 4: C_obs_full should have room to improve (not ceiling)
cobs_has_room = cobs_macro_bal < 0.90
print(f"  c_obs_full has room (bal<0.90):   {cobs_has_room}  ({cobs_macro_bal:.4f})")

# Check 5: raw composite accuracy IS misleading (all_false raw >> all_false bal)
raw_misleading = all_false_raw > 0.65 and all_false_macro_bal < 0.55
print(f"  raw composite IS misleading:      {raw_misleading}  (raw={all_false_raw:.4f}, bal={all_false_macro_bal:.4f})")

# Check 6: observation adds value under balanced metrics
obs_adds_value_bal = cobs_macro_bal > iom_prior_macro_bal + 0.01
print(f"  observation adds value (bal):     {obs_adds_value_bal}  "
      f"(cobs={cobs_macro_bal:.4f} > iom={iom_prior_macro_bal:.4f}+0.01)")

# Check 7: category_majority under balanced metrics
# (should be high since categories are deterministic per query)
print(f"  category_majority macro_bal:      {cat_maj_macro_bal:.4f}")

# Overall readiness
balanced_metric_ready = all_false_bal_ok and oracle_high and cobs_has_room
ready_for_policy_rerun = balanced_metric_ready and obs_adds_value_bal

print(f"\n  --- Summary ---")
print(f"  balanced_metric_ready:            {balanced_metric_ready}")
print(f"  ready_for_policy_rerun:           {ready_for_policy_rerun}")

# ---- Final output ----
print(f"\n{'='*70}")
print(f"  C_obs_full cost metrics: "
      f"visited={full_cost_metrics['objects_visited']}, "
      f"total_cost={full_cost_metrics['total_cost']:.4f}, "
      f"budget_remaining={full_cost_metrics['budget_remaining']:.4f}")

print()
print("[block_done]")
print(f"  block_id=1H0")
print(f"  raw_all_false_comp={all_false_raw:.4f}")
print(f"  raw_iom_prior_comp={iom_prior_raw:.4f}")
print(f"  raw_iom_prior_macro_bal_acc={iom_prior_macro_bal:.4f}")
print(f"  balanced_all_false_macro_bal_acc={all_false_macro_bal:.4f}")
print(f"  balanced_iom_prior_macro_bal_acc={iom_prior_macro_bal:.4f}")
print(f"  balanced_c_obs_full_macro_bal_acc={cobs_macro_bal:.4f}")
print(f"  balanced_oracle_macro_bal_acc={oracle_macro_bal:.4f}")
print(f"  balanced_category_majority_macro_bal_acc={cat_maj_macro_bal:.4f}")
print(f"  observation_bal_gain={cobs_macro_bal - iom_prior_macro_bal:+.4f}")
print(f"  oracle_bal_gap={oracle_macro_bal - cobs_macro_bal:+.4f}")
print(f"  total_oracle_bal_gap={oracle_macro_bal - all_false_macro_bal:+.4f}")
print(f"  balanced_metric_ready={'true' if balanced_metric_ready else 'false'}")
print(f"  ready_for_policy_rerun={'true' if ready_for_policy_rerun else 'false'}")
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
    "block_id": "1H0",
    "elapsed_s": time.time() - t0,
    "budget": budget,
    "seed": seed,
    "query_names": QUERY_NAMES,
    "n_test_objects": n_test,
    "per_query_class_distribution": make_serializable(per_query_stats),
    "baselines": {
        bname: {
            "raw_comp_acc": all_metrics[bname]["raw_comp_acc"],
            "macro_query_acc": all_metrics[bname]["macro_query_acc"],
            "macro_query_bal_acc": all_metrics[bname]["macro_query_bal_acc"],
            "macro_bal_subset_acc": all_metrics[bname]["bal_subset"]["macro_bal_subset_acc"],
            "per_query": make_serializable(all_metrics[bname]["per_query"]),
            "bal_subset": make_serializable(all_metrics[bname]["bal_subset"]),
        }
        for bname in baseline_names
    },
    "diagnostics": {
        "all_false_macro_bal_near_0_5": all_false_bal_ok,
        "iom_prior_not_strong_under_balanced": iom_prior_weak,
        "oracle_high_under_balanced": oracle_high,
        "cobs_has_room_under_balanced": cobs_has_room,
        "raw_composite_misleading": raw_misleading,
        "observation_adds_value_bal": obs_adds_value_bal,
        "balanced_metric_ready": balanced_metric_ready,
        "ready_for_policy_rerun": ready_for_policy_rerun,
    },
    "key_values": {
        "raw_all_false_comp": all_false_raw,
        "raw_iom_prior_comp": iom_prior_raw,
        "raw_iom_prior_macro_bal_acc": iom_prior_macro_bal,
        "balanced_all_false_macro_bal_acc": all_false_macro_bal,
        "balanced_iom_prior_macro_bal_acc": iom_prior_macro_bal,
        "balanced_c_obs_full_macro_bal_acc": cobs_macro_bal,
        "balanced_oracle_macro_bal_acc": oracle_macro_bal,
        "balanced_category_majority_macro_bal_acc": cat_maj_macro_bal,
        "observation_bal_gain": cobs_macro_bal - iom_prior_macro_bal,
        "oracle_bal_gap": oracle_macro_bal - cobs_macro_bal,
        "total_oracle_bal_gap": oracle_macro_bal - all_false_macro_bal,
    },
    "c_obs_full_cost": {
        "objects_visited": full_cost_metrics["objects_visited"],
        "total_cost": full_cost_metrics["total_cost"],
        "budget_remaining": full_cost_metrics["budget_remaining"],
    },
}
with open("runs/calibration_block1h0_balanced_evaluation_audit.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1h0_balanced_evaluation_audit.json")
