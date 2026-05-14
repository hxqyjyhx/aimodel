"""Block 1G2: Label Balance / Majority-Class Baseline Audit.

Audits class balance for original condition (C3_P060_O040) and
instance-variation condition (C3v_instance_variation_v0) from Block 1G1.

Computes per-query label distributions, 6 baselines (all_false, all_true,
per_query_majority, category_majority, IOM_prior, C_obs_full),
and determines whether composite_task_accuracy is dominated by majority-class effects.
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
import objects as obj_module

seed = 101
budget = 1.5
QUERY_NAMES = ["need_planks", "need_stone", "need_food", "need_tool", "need_fuel"]

# ---- GT conversion helper ----
def _profile_to_gt_affordances(profile):
    """Convert raw string profile (success/fail/clear) to float dict."""
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


# ---- Query prediction from probs (same as MiniMCEvaluator._check_query) ----
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
# Per-condition audit
# ============================================================
def audit_condition(condition_label, variation_config=None):
    """Run full label-balance audit for one condition.

    variation_config: None for original, or dict like Block 1G1 INSTANCE_VARIATION.
    """
    print(f"\n{'='*70}")
    print(f"Condition: {condition_label}")
    print(f"{'='*70}")

    # ---- Monkey-patch if variation ----
    _original_make = None
    _variation_rng = None
    if variation_config:
        _original_make = obj_module._make_hidden_profile
        _variation_rng = random.Random(seed * 3 + 7777)

        def _patched(category, rng):
            profile = _original_make(category, rng)
            overrides = variation_config.get(category, {})
            for action, flip_prob in overrides.items():
                if _variation_rng.random() < flip_prob:
                    current = profile.get(action, "success")
                    profile[action] = "fail" if current == "success" else "success"
            return profile
        obj_module._make_hidden_profile = _patched
        print("  (monkey-patched _make_hidden_profile for instance variation)")

    # ---- Phase A ----
    try:
        from run_004_5n1 import run_phase_a_training
        cond = copy.deepcopy({"label": "C3_P060_O040", "p_target": 0.60,
                              "p_other": 0.40, "absent": False})
        print("  Running Phase A training...")
        (student, base_learner, train_objects, train_env,
         test_objects, test_env, final_metrics, rng) = run_phase_a_training(seed, cond)
    finally:
        if _original_make:
            obj_module._make_hidden_profile = _original_make
            print("  (restored original _make_hidden_profile)")

    train_objects_dict = {oid: train_env.objects[oid] for oid in train_objects}
    test_oids = sorted(test_objects)
    test_objects_dict = {oid: test_env.objects[oid] for oid in test_oids}

    n_train = len(train_objects_dict)
    n_test = len(test_oids)
    print(f"  train_objects={n_train}, test_objects={n_test}")

    # ---- Per-object GT labels ----
    gt_per_object = {}
    gt_labels = {}  # (oid, query) -> bool
    for oid in test_oids:
        profile = test_objects_dict[oid]["hidden_affordance_profile"]
        gt = _profile_to_gt_affordances(profile)
        gt_per_object[oid] = gt
        gt_labels[oid] = {}
        for qname in QUERY_NAMES:
            gt_labels[oid][qname] = _TASK_QUERIES[qname](gt)

    # ---- Per-query label distribution ----
    print(f"\n  Per-query label distribution ({condition_label}):")
    print(f"  {'query':<16} {'n_pos':>6} {'n_neg':>6} {'pos_rate':>9} {'neg_rate':>9}")
    per_query_stats = {}
    total_pos = 0
    total_neg = 0
    for qname in QUERY_NAMES:
        n_pos = sum(1 for oid in test_oids if gt_labels[oid][qname])
        n_neg = n_test - n_pos
        pos_rate = n_pos / n_test
        neg_rate = n_neg / n_test
        total_pos += n_pos
        total_neg += n_neg
        per_query_stats[qname] = {
            "n_positive": n_pos, "n_negative": n_neg,
            "positive_rate": pos_rate, "negative_rate": neg_rate,
        }
        print(f"  {qname:<16} {n_pos:>6} {n_neg:>6} {pos_rate:>9.4f} {neg_rate:>9.4f}")

    mean_neg_rate = total_neg / (total_pos + total_neg) if (total_pos + total_neg) > 0 else 0.0
    print(f"  {'OVERALL':<16} {total_pos:>6} {total_neg:>6} {1-mean_neg_rate:>9.4f} {mean_neg_rate:>9.4f}")

    # ---- Hidden category distribution ----
    from collections import Counter
    cat_counts = Counter()
    for oid in test_oids:
        cat_counts[test_objects_dict[oid]["hidden_category"]] += 1
    print(f"\n  Test object categories: {dict(cat_counts)}")

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

    # ---- Positions for C_obs_full MiniMCEnvironment ----
    positions = MiniMCSimulatorTruth.assign_positions(
        test_oids, config.GRID_ROWS, config.GRID_COLS, config.AGENT_START, rng)

    # ============================================================
    # BASELINES
    # ============================================================
    baselines = {}

    # ---- Baseline 1: all_false ----
    all_false_correct = 0
    all_false_total = 0
    all_false_per_query = {q: {"correct": 0, "total": 0} for q in QUERY_NAMES}
    for oid in test_oids:
        for qname in QUERY_NAMES:
            true_label = gt_labels[oid][qname]
            pred_label = False
            if pred_label == true_label:
                all_false_correct += 1
                all_false_per_query[qname]["correct"] += 1
            all_false_total += 1
            all_false_per_query[qname]["total"] += 1
    baselines["all_false"] = {
        "comp_acc": all_false_correct / all_false_total,
        "per_query_acc": {q: all_false_per_query[q]["correct"] / all_false_per_query[q]["total"]
                         for q in QUERY_NAMES},
        "predictions": {oid: {q: False for q in QUERY_NAMES} for oid in test_oids},
    }

    # ---- Baseline 2: all_true ----
    all_true_correct = 0
    all_true_total = 0
    all_true_per_query = {q: {"correct": 0, "total": 0} for q in QUERY_NAMES}
    for oid in test_oids:
        for qname in QUERY_NAMES:
            true_label = gt_labels[oid][qname]
            pred_label = True
            if pred_label == true_label:
                all_true_correct += 1
                all_true_per_query[qname]["correct"] += 1
            all_true_total += 1
            all_true_per_query[qname]["total"] += 1
    baselines["all_true"] = {
        "comp_acc": all_true_correct / all_true_total,
        "per_query_acc": {q: all_true_per_query[q]["correct"] / all_true_per_query[q]["total"]
                         for q in QUERY_NAMES},
        "predictions": {oid: {q: True for q in QUERY_NAMES} for oid in test_oids},
    }

    # ---- Baseline 3: per_query_majority ----
    per_query_majority_label = {}
    for qname in QUERY_NAMES:
        n_pos = per_query_stats[qname]["n_positive"]
        n_neg = per_query_stats[qname]["n_negative"]
        per_query_majority_label[qname] = n_pos >= n_neg

    pq_maj_correct = 0
    pq_maj_total = 0
    pq_maj_per_query = {q: {"correct": 0, "total": 0} for q in QUERY_NAMES}
    for oid in test_oids:
        for qname in QUERY_NAMES:
            true_label = gt_labels[oid][qname]
            pred_label = per_query_majority_label[qname]
            if pred_label == true_label:
                pq_maj_correct += 1
                pq_maj_per_query[qname]["correct"] += 1
            pq_maj_total += 1
            pq_maj_per_query[qname]["total"] += 1
    baselines["per_query_majority"] = {
        "comp_acc": pq_maj_correct / pq_maj_total,
        "per_query_acc": {q: pq_maj_per_query[q]["correct"] / pq_maj_per_query[q]["total"]
                         for q in QUERY_NAMES},
        "majority_label": per_query_majority_label,
        "predictions": {oid: {q: per_query_majority_label[q] for q in QUERY_NAMES}
                       for oid in test_oids},
    }

    # ---- Baseline 4: category_majority ----
    # Compute per-(category, query) majority from training objects
    cat_query_train_pos = {}  # (cat, q) -> n_pos
    cat_query_train_total = {}  # (cat, q) -> n_total
    for tid in train_objects:
        cat = train_objects_dict[tid]["hidden_category"]
        profile = train_objects_dict[tid]["hidden_affordance_profile"]
        gt = _profile_to_gt_affordances(profile)
        for qname in QUERY_NAMES:
            key = (cat, qname)
            cat_query_train_pos.setdefault(key, 0)
            cat_query_train_total.setdefault(key, 0)
            if _TASK_QUERIES[qname](gt):
                cat_query_train_pos[key] += 1
            cat_query_train_total[key] += 1

    cat_majority_label = {}
    for (cat, qname), n_pos in cat_query_train_pos.items():
        n_total = cat_query_train_total[(cat, qname)]
        cat_majority_label[(cat, qname)] = n_pos >= (n_total - n_pos)

    cat_maj_correct = 0
    cat_maj_total = 0
    cat_maj_per_query = {q: {"correct": 0, "total": 0, "unmatched": 0} for q in QUERY_NAMES}
    for oid in test_oids:
        cat = test_objects_dict[oid]["hidden_category"]
        for qname in QUERY_NAMES:
            key = (cat, qname)
            if key in cat_majority_label:
                pred_label = cat_majority_label[key]
            else:
                # Category not seen in training — fallback to per-query majority
                pred_label = per_query_majority_label.get(qname, False)
                cat_maj_per_query[qname]["unmatched"] += 1
            true_label = gt_labels[oid][qname]
            if pred_label == true_label:
                cat_maj_correct += 1
                cat_maj_per_query[qname]["correct"] += 1
            cat_maj_total += 1
            cat_maj_per_query[qname]["total"] += 1
    baselines["category_majority"] = {
        "comp_acc": cat_maj_correct / cat_maj_total,
        "per_query_acc": {q: cat_maj_per_query[q]["correct"] / max(cat_maj_per_query[q]["total"], 1)
                         for q in QUERY_NAMES},
        "per_query_unmatched": {q: cat_maj_per_query[q]["unmatched"] for q in QUERY_NAMES},
    }

    # ---- Baseline 5: IOM_prior_empty_features ----
    im_prior = im_base.clone()
    iom_correct = 0
    iom_total = 0
    iom_per_query = {q: {"correct": 0, "total": 0} for q in QUERY_NAMES}
    iom_per_object = {}
    for oid in test_oids:
        fake_obj = {"id": oid, "visible_features": {}}
        probs = im_prior.predict_all_affordances(fake_obj)
        iom_per_object[oid] = probs
        for qname in QUERY_NAMES:
            pred_label = _check_query(probs, qname)
            true_label = gt_labels[oid][qname]
            if pred_label == true_label:
                iom_correct += 1
                iom_per_query[qname]["correct"] += 1
            iom_total += 1
            iom_per_query[qname]["total"] += 1
    baselines["iom_prior"] = {
        "comp_acc": iom_correct / iom_total,
        "per_query_acc": {q: iom_per_query[q]["correct"] / iom_per_query[q]["total"]
                         for q in QUERY_NAMES},
        "predictions": iom_per_object,
    }

    # ---- Baseline 6: C_obs_full (C0b ObserveOnlyPolicy) ----
    print("  Running C_obs_full...")
    env_full = MiniMCEnvironment(test_objects_dict, positions, initial_budget=budget)
    im_full = im_base.clone()
    rng_full = random.Random(seed + 100)
    full_policy = C0b_ObserveOnlyPolicy(im_full, rng_full)
    full_result = EpisodeHarness(env_full, full_policy).run()
    full_evaluator = MiniMCEvaluator(env_full)
    full_metrics = full_evaluator.evaluate(
        full_result.predictions, full_result.event_log, full_result.agent_obs, {}, {})

    # Extract per-query accuracies from full result
    full_per_object = full_result.predictions.get("per_object", {})
    full_correct = 0
    full_total = 0
    full_per_query = {q: {"correct": 0, "total": 0} for q in QUERY_NAMES}
    for oid in test_oids:
        probs = full_per_object.get(oid, {})
        for qname in QUERY_NAMES:
            pred_label = _check_query(probs, qname)
            true_label = gt_labels[oid][qname]
            if pred_label == true_label:
                full_correct += 1
                full_per_query[qname]["correct"] += 1
            full_total += 1
            full_per_query[qname]["total"] += 1
    baselines["c_obs_full"] = {
        "comp_acc": full_correct / max(full_total, 1),
        "per_query_acc": {q: full_per_query[q]["correct"] / max(full_per_query[q]["total"], 1)
                         for q in QUERY_NAMES},
        "metrics": full_metrics,
    }

    # ============================================================
    # Per-query detailed metrics for each baseline
    # ============================================================
    all_baseline_names = ["all_false", "all_true", "per_query_majority",
                          "category_majority", "iom_prior", "c_obs_full"]

    def compute_per_query_detailed(baseline_name, preds_by_oid_query):
        """Compute per-query: acc, balanced_acc, pos_recall, neg_recall."""
        result = {}
        for qname in QUERY_NAMES:
            pos_correct = 0
            pos_total = 0
            neg_correct = 0
            neg_total = 0
            correct = 0
            total = 0
            for oid in test_oids:
                true_label = gt_labels[oid][qname]
                pred_label = preds_by_oid_query[oid][qname]
                if pred_label == true_label:
                    correct += 1
                total += 1
                if true_label:
                    pos_total += 1
                    if pred_label:
                        pos_correct += 1
                else:
                    neg_total += 1
                    if not pred_label:
                        neg_correct += 1

            acc = correct / max(total, 1)
            pos_recall = pos_correct / max(pos_total, 1)
            neg_recall = neg_correct / max(neg_total, 1)

            # Balanced accuracy: handle missing classes
            if pos_total > 0 and neg_total > 0:
                bal_acc = 0.5 * (pos_recall + neg_recall)
            elif pos_total > 0:
                bal_acc = pos_recall
            elif neg_total > 0:
                bal_acc = neg_recall
            else:
                bal_acc = 0.0

            result[qname] = {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "positive_recall": pos_recall,
                "negative_recall": neg_recall,
                "n_pos_gt": pos_total,
                "n_neg_gt": neg_total,
            }
        return result

    # Build oid->{qname->bool} predictions for each simple baseline
    baseline_query_details = {}
    for bname in all_baseline_names:
        if bname in ["all_false", "all_true", "per_query_majority"]:
            preds = baselines[bname]["predictions"]
            baseline_query_details[bname] = compute_per_query_detailed(bname, preds)
        elif bname == "category_majority":
            # Build predictions dict
            cm_preds = {}
            for oid in test_oids:
                cm_preds[oid] = {}
                cat = test_objects_dict[oid]["hidden_category"]
                for qname in QUERY_NAMES:
                    key = (cat, qname)
                    if key in cat_majority_label:
                        cm_preds[oid][qname] = cat_majority_label[key]
                    else:
                        cm_preds[oid][qname] = per_query_majority_label.get(qname, False)
            baseline_query_details[bname] = compute_per_query_detailed(bname, cm_preds)
        elif bname == "iom_prior":
            iom_preds = {}
            for oid in test_oids:
                iom_preds[oid] = {}
                probs = iom_per_object[oid]
                for qname in QUERY_NAMES:
                    iom_preds[oid][qname] = _check_query(probs, qname)
            baseline_query_details[bname] = compute_per_query_detailed(bname, iom_preds)
        elif bname == "c_obs_full":
            cobs_preds = {}
            for oid in test_oids:
                cobs_preds[oid] = {}
                probs = full_per_object.get(oid, {})
                for qname in QUERY_NAMES:
                    cobs_preds[oid][qname] = _check_query(probs, qname)
            baseline_query_details[bname] = compute_per_query_detailed(bname, cobs_preds)

    # ---- Aggregate metrics per baseline ----
    for bname in all_baseline_names:
        details = baseline_query_details[bname]
        comp_acc = baselines[bname]["comp_acc"]

        # Macro query accuracy (unweighted mean of per-query accuracies)
        per_q_accs = [details[q]["accuracy"] for q in QUERY_NAMES]
        macro_query_acc = sum(per_q_accs) / len(per_q_accs)

        # Macro balanced accuracy
        per_q_bal = [details[q]["balanced_accuracy"] for q in QUERY_NAMES]
        macro_bal_acc = sum(per_q_bal) / len(per_q_bal)

        baselines[bname]["macro_query_acc"] = macro_query_acc
        baselines[bname]["macro_bal_acc"] = macro_bal_acc
        baselines[bname]["query_details"] = details

    # ============================================================
    # Print results
    # ============================================================

    # Per-query label distribution with all_false/all_true acc
    print(f"\n  --- Per-query with trivial baselines ---")
    print(f"  {'query':<16} {'n_pos':>5} {'n_neg':>5} {'pos_rate':>8} {'all_false_acc':>13} {'all_true_acc':>13}")
    for qname in QUERY_NAMES:
        stats = per_query_stats[qname]
        af_acc = baselines["all_false"]["per_query_acc"][qname]
        at_acc = baselines["all_true"]["per_query_acc"][qname]
        print(f"  {qname:<16} {stats['n_positive']:>5} {stats['n_negative']:>5} "
              f"{stats['positive_rate']:>8.4f} {af_acc:>13.4f} {at_acc:>13.4f}")

    # Baseline summary table
    print(f"\n  --- Baseline summary ---")
    print(f"  {'baseline':<22} {'comp_acc':>9} {'macro_query_acc':>15} {'macro_bal_acc':>14}")
    for bname in all_baseline_names:
        b = baselines[bname]
        print(f"  {bname:<22} {b['comp_acc']:>9.4f} {b['macro_query_acc']:>15.4f} "
              f"{b['macro_bal_acc']:>14.4f}")

    # Per-query baseline detail (for each baseline, show per-query)
    for bname in all_baseline_names:
        details = baseline_query_details[bname]
        print(f"\n  --- {bname} per-query detail ---")
        print(f"  {'query':<16} {'acc':>8} {'bal_acc':>8} {'pos_recall':>10} {'neg_recall':>10} "
              f"{'n_pos':>6} {'n_neg':>6}")
        for qname in QUERY_NAMES:
            d = details[qname]
            print(f"  {qname:<16} {d['accuracy']:>8.4f} {d['balanced_accuracy']:>8.4f} "
                  f"{d['positive_recall']:>10.4f} {d['negative_recall']:>10.4f} "
                  f"{d['n_pos_gt']:>6} {d['n_neg_gt']:>6}")

    # ---- Key comparisons ----
    all_false_comp = baselines["all_false"]["comp_acc"]
    pq_maj_comp = baselines["per_query_majority"]["comp_acc"]
    iom_prior_comp = baselines["iom_prior"]["comp_acc"]
    iom_prior_macro_bal = baselines["iom_prior"]["macro_bal_acc"]
    cobs_comp = baselines["c_obs_full"]["comp_acc"]
    cobs_macro_bal = baselines["c_obs_full"]["macro_bal_acc"]

    print(f"\n  --- Key comparisons ---")
    print(f"  all_false_comp:          {all_false_comp:.4f}")
    print(f"  per_query_majority_comp:  {pq_maj_comp:.4f}")
    print(f"  iom_prior_comp:           {iom_prior_comp:.4f}")
    print(f"  iom_prior_macro_bal_acc:  {iom_prior_macro_bal:.4f}")
    print(f"  c_obs_full_comp:          {cobs_comp:.4f}")
    print(f"  c_obs_full_macro_bal_acc: {cobs_macro_bal:.4f}")
    print(f"  iom_prior vs all_false:   {iom_prior_comp - all_false_comp:+.4f}")
    print(f"  iom_prior vs pq_majority: {iom_prior_comp - pq_maj_comp:+.4f}")
    print(f"  c_obs_full vs all_false:  {cobs_comp - all_false_comp:+.4f}")
    print(f"  comp vs macro_bal (iom):  {iom_prior_comp - iom_prior_macro_bal:+.4f}")

    # ---- Interpretation flags ----
    all_false_strong = all_false_comp >= 0.70
    majority_class_dominated = mean_neg_rate >= 0.70
    raw_comp_misleading = abs(iom_prior_comp - all_false_comp) < 0.05
    balanced_metric_needed = (iom_prior_comp - iom_prior_macro_bal) >= 0.05

    print(f"\n  --- Interpretation flags ---")
    print(f"  all_false_strong:            {all_false_strong}  (all_false_comp={all_false_comp:.4f} >= 0.70)")
    print(f"  majority_class_dominated:    {majority_class_dominated}  (mean_neg_rate={mean_neg_rate:.4f} >= 0.70)")
    print(f"  raw_comp_misleading:         {raw_comp_misleading}  (|iom_prior - all_false|={abs(iom_prior_comp - all_false_comp):.4f} < 0.05)")
    print(f"  balanced_metric_needed:      {balanced_metric_needed}  (comp - macro_bal={iom_prior_comp - iom_prior_macro_bal:+.4f} >= 0.05)")

    return {
        "condition_label": condition_label,
        "n_train": n_train,
        "n_test": n_test,
        "category_counts": dict(cat_counts),
        "per_query_stats": per_query_stats,
        "mean_negative_rate": mean_neg_rate,
        "baselines": baselines,
        "baseline_query_details": baseline_query_details,
        "all_baseline_names": all_baseline_names,
        "all_false_comp": all_false_comp,
        "per_query_majority_comp": pq_maj_comp,
        "iom_prior_comp": iom_prior_comp,
        "iom_prior_macro_bal_acc": iom_prior_macro_bal,
        "c_obs_full_comp": cobs_comp,
        "c_obs_full_macro_bal_acc": cobs_macro_bal,
        "all_false_strong": all_false_strong,
        "majority_class_dominated": majority_class_dominated,
        "raw_comp_misleading": raw_comp_misleading,
        "balanced_metric_needed": balanced_metric_needed,
        "gt_labels": gt_labels,
    }


# ============================================================
# Run both conditions
# ============================================================

print("=" * 70)
print("Block 1G2: Label Balance / Majority-Class Baseline Audit")
print(f"  budget={budget}, seed={seed}")
print("=" * 70)

# Condition 1: Original C3_P060_O040
result_original = audit_condition("C3_P060_O040", variation_config=None)

# Condition 2: Instance-variation from Block 1G1
INSTANCE_VARIATION = {
    "wood_log": {"craft_plank": 0.50},
    "apple": {"eat": 0.50},
    "stone_block": {"mine_with_pickaxe": 0.50},
    "wooden_pickaxe": {"use_as_tool": 0.50},
}
result_v0 = audit_condition("C3v_instance_variation_v0", variation_config=INSTANCE_VARIATION)

# ============================================================
# Cross-condition comparison
# ============================================================
print(f"\n{'='*70}")
print(f"Cross-Condition Comparison")
print(f"{'='*70}")

print(f"\n  {'metric':<35} {'C3_P060_O040':>14} {'C3v_v0':>14} {'delta':>14}")
print(f"  {'all_false_comp':<35} {result_original['all_false_comp']:>14.4f} "
      f"{result_v0['all_false_comp']:>14.4f} "
      f"{result_v0['all_false_comp'] - result_original['all_false_comp']:>+14.4f}")
print(f"  {'per_query_majority_comp':<35} {result_original['per_query_majority_comp']:>14.4f} "
      f"{result_v0['per_query_majority_comp']:>14.4f} "
      f"{result_v0['per_query_majority_comp'] - result_original['per_query_majority_comp']:>+14.4f}")
print(f"  {'iom_prior_comp':<35} {result_original['iom_prior_comp']:>14.4f} "
      f"{result_v0['iom_prior_comp']:>14.4f} "
      f"{result_v0['iom_prior_comp'] - result_original['iom_prior_comp']:>+14.4f}")
print(f"  {'iom_prior_macro_bal_acc':<35} {result_original['iom_prior_macro_bal_acc']:>14.4f} "
      f"{result_v0['iom_prior_macro_bal_acc']:>14.4f} "
      f"{result_v0['iom_prior_macro_bal_acc'] - result_original['iom_prior_macro_bal_acc']:>+14.4f}")
print(f"  {'c_obs_full_comp':<35} {result_original['c_obs_full_comp']:>14.4f} "
      f"{result_v0['c_obs_full_comp']:>14.4f} "
      f"{result_v0['c_obs_full_comp'] - result_original['c_obs_full_comp']:>+14.4f}")
print(f"  {'c_obs_full_macro_bal_acc':<35} {result_original['c_obs_full_macro_bal_acc']:>14.4f} "
      f"{result_v0['c_obs_full_macro_bal_acc']:>14.4f} "
      f"{result_v0['c_obs_full_macro_bal_acc'] - result_original['c_obs_full_macro_bal_acc']:>+14.4f}")
print(f"  {'mean_negative_rate':<35} {result_original['mean_negative_rate']:>14.4f} "
      f"{result_v0['mean_negative_rate']:>14.4f} "
      f"{result_v0['mean_negative_rate'] - result_original['mean_negative_rate']:>+14.4f}")

print(f"\n  --- Per-query comparison (iom_prior_comp) ---")
print(f"  {'query':<16} {'original':>10} {'v0':>10} {'delta':>10}")
for qname in QUERY_NAMES:
    orig_acc = result_original['baselines']['iom_prior']['per_query_acc'][qname]
    v0_acc = result_v0['baselines']['iom_prior']['per_query_acc'][qname]
    print(f"  {qname:<16} {orig_acc:>10.4f} {v0_acc:>10.4f} {v0_acc - orig_acc:>+10.4f}")

print(f"\n  --- Interpretation flag comparison ---")
print(f"  {'flag':<28} {'original':<10} {'v0':<10}")
print(f"  {'all_false_strong':<28} {str(result_original['all_false_strong']):<10} {str(result_v0['all_false_strong']):<10}")
print(f"  {'majority_class_dominated':<28} {str(result_original['majority_class_dominated']):<10} {str(result_v0['majority_class_dominated']):<10}")
print(f"  {'raw_comp_misleading':<28} {str(result_original['raw_comp_misleading']):<10} {str(result_v0['raw_comp_misleading']):<10}")
print(f"  {'balanced_metric_needed':<28} {str(result_original['balanced_metric_needed']):<10} {str(result_v0['balanced_metric_needed']):<10}")

# ============================================================
# Overall diagnostic
# ============================================================
# Use original condition as primary (it's the one we're trying to improve)
ready = (not result_original["raw_comp_misleading"]) and result_original["balanced_metric_needed"]

print(f"\n{'='*70}")
print(f"Overall Diagnostic")
print(f"{'='*70}")
print(f"  majority_class_dominated:   {result_original['majority_class_dominated']}")
print(f"  balanced_metric_needed:     {result_original['balanced_metric_needed']}")
print(f"  raw_comp_misleading:        {result_original['raw_comp_misleading']}")
print(f"  1G1 made it worse:          {result_v0['all_false_comp'] > result_original['all_false_comp']}")
print(f"  ready_for_environment_redesign: {ready}")

print()
print("[block_done]")
print(f"  block_id=1G2")
print(f"  original_all_false_comp={result_original['all_false_comp']:.4f}")
print(f"  original_iom_prior_comp={result_original['iom_prior_comp']:.4f}")
print(f"  original_iom_prior_macro_bal_acc={result_original['iom_prior_macro_bal_acc']:.4f}")
print(f"  original_c_obs_full_comp={result_original['c_obs_full_comp']:.4f}")
print(f"  original_c_obs_full_macro_bal_acc={result_original['c_obs_full_macro_bal_acc']:.4f}")
print(f"  v0_all_false_comp={result_v0['all_false_comp']:.4f}")
print(f"  v0_iom_prior_comp={result_v0['iom_prior_comp']:.4f}")
print(f"  v0_iom_prior_macro_bal_acc={result_v0['iom_prior_macro_bal_acc']:.4f}")
print(f"  v0_c_obs_full_comp={result_v0['c_obs_full_comp']:.4f}")
print(f"  v0_mean_negative_rate={result_v0['mean_negative_rate']:.4f}")
print(f"  original_mean_negative_rate={result_original['mean_negative_rate']:.4f}")
print(f"  majority_class_dominated={'true' if result_original['majority_class_dominated'] else 'false'}")
print(f"  balanced_metric_needed={'true' if result_original['balanced_metric_needed'] else 'false'}")
print(f"  ready_for_environment_redesign={'true' if ready else 'false'}")
print(f"  elapsed={time.time() - t0:.0f}s")

# ============================================================
# Save
# ============================================================
# Helper to make results JSON-serializable
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
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
    "block_id": "1G2",
    "elapsed_s": time.time() - t0,
    "budget": budget,
    "seed": seed,
    "query_names": QUERY_NAMES,
    "original": {
        "condition_label": result_original["condition_label"],
        "n_train": result_original["n_train"],
        "n_test": result_original["n_test"],
        "category_counts": {str(k): v for k, v in result_original["category_counts"].items()},
        "per_query_stats": make_serializable(result_original["per_query_stats"]),
        "mean_negative_rate": result_original["mean_negative_rate"],
        "baselines": {k: {
            "comp_acc": v["comp_acc"],
            "macro_query_acc": v["macro_query_acc"],
            "macro_bal_acc": v["macro_bal_acc"],
            "per_query_acc": make_serializable(v["per_query_acc"]),
        } for k, v in result_original["baselines"].items()},
        "baseline_query_details": make_serializable(result_original["baseline_query_details"]),
        "all_false_comp": result_original["all_false_comp"],
        "per_query_majority_comp": result_original["per_query_majority_comp"],
        "iom_prior_comp": result_original["iom_prior_comp"],
        "iom_prior_macro_bal_acc": result_original["iom_prior_macro_bal_acc"],
        "c_obs_full_comp": result_original["c_obs_full_comp"],
        "c_obs_full_macro_bal_acc": result_original["c_obs_full_macro_bal_acc"],
        "flags": {
            "all_false_strong": result_original["all_false_strong"],
            "majority_class_dominated": result_original["majority_class_dominated"],
            "raw_comp_misleading": result_original["raw_comp_misleading"],
            "balanced_metric_needed": result_original["balanced_metric_needed"],
        },
        "per_query_stats_detail": make_serializable(result_original["per_query_stats"]),
    },
    "variation_v0": {
        "condition_label": result_v0["condition_label"],
        "n_train": result_v0["n_train"],
        "n_test": result_v0["n_test"],
        "category_counts": {str(k): v for k, v in result_v0["category_counts"].items()},
        "per_query_stats": make_serializable(result_v0["per_query_stats"]),
        "mean_negative_rate": result_v0["mean_negative_rate"],
        "baselines": {k: {
            "comp_acc": v["comp_acc"],
            "macro_query_acc": v["macro_query_acc"],
            "macro_bal_acc": v["macro_bal_acc"],
            "per_query_acc": make_serializable(v["per_query_acc"]),
        } for k, v in result_v0["baselines"].items()},
        "baseline_query_details": make_serializable(result_v0["baseline_query_details"]),
        "all_false_comp": result_v0["all_false_comp"],
        "per_query_majority_comp": result_v0["per_query_majority_comp"],
        "iom_prior_comp": result_v0["iom_prior_comp"],
        "iom_prior_macro_bal_acc": result_v0["iom_prior_macro_bal_acc"],
        "c_obs_full_comp": result_v0["c_obs_full_comp"],
        "c_obs_full_macro_bal_acc": result_v0["c_obs_full_macro_bal_acc"],
        "flags": {
            "all_false_strong": result_v0["all_false_strong"],
            "majority_class_dominated": result_v0["majority_class_dominated"],
            "raw_comp_misleading": result_v0["raw_comp_misleading"],
            "balanced_metric_needed": result_v0["balanced_metric_needed"],
        },
        "per_query_stats_detail": make_serializable(result_v0["per_query_stats"]),
    },
    "cross_condition": {
        "delta_all_false_comp": result_v0["all_false_comp"] - result_original["all_false_comp"],
        "delta_iom_prior_comp": result_v0["iom_prior_comp"] - result_original["iom_prior_comp"],
        "delta_iom_prior_macro_bal": result_v0["iom_prior_macro_bal_acc"] - result_original["iom_prior_macro_bal_acc"],
        "delta_c_obs_full_comp": result_v0["c_obs_full_comp"] - result_original["c_obs_full_comp"],
        "delta_mean_negative_rate": result_v0["mean_negative_rate"] - result_original["mean_negative_rate"],
        "v0_made_all_false_stronger": result_v0["all_false_comp"] > result_original["all_false_comp"],
    },
    "diagnostic": {
        "majority_class_dominated": result_original["majority_class_dominated"],
        "balanced_metric_needed": result_original["balanced_metric_needed"],
        "ready_for_environment_redesign": ready,
    },
}
with open("runs/calibration_block1g2_label_balance_majority_audit.json", "w") as f:
    json.dump(block_out, f, indent=2)
print(f"  saved: runs/calibration_block1g2_label_balance_majority_audit.json")
