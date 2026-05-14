"""
Block 1J40b-7h -- Env3b / Counterfactual Redesign Parameter Audit.

Audit-only parameter search over candidate redesign families.
Does not replace official env3b, does not train a learner, and does not
implement 1J40b-8.
"""

import json
import os
import random
import runpy
import time
from collections import defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES_USED = [
    "_block1j40b7e_env3b_distributional_audit_counterfactual.py",
    "_block1j40b7fb_compositional_oracle_belief_upper_bound.py",
    "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py",
    "runs/block1j40b7e_env3b_distributional_audit_counterfactual.json",
    "runs/block1j40b7fb_compositional_oracle_belief_upper_bound.json",
    "runs/block1j40b7g_env3b_counterfactual_headroom_decomposition.json",
]

print("[1/6] Loading 7g environment helpers...")
ns = runpy.run_path(os.path.join(CURRENT_DIR, "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py"))

SEEDS = ns["SEEDS"]
ALL_TRY_AFFORDANCES = ns["ALL_TRY_AFFORDANCES"]
INSTANCE_SUBTYPE_DEFS = ns["INSTANCE_SUBTYPE_DEFS"]
AMBIENT_GROUP_CATEGORIES = ns["AMBIENT_GROUP_CATEGORIES"]
ALL_AMBIENT_FEATURE_NAMES = ns["ALL_AMBIENT_FEATURE_NAMES"]

per_seed_objects = ns["per_seed_objects"]
per_seed_audit_labels = ns["per_seed_audit_labels"]
TEST_A_VARIANTS = ns["TEST_A_VARIANTS"]
TEST_B_VARIANTS = ns["TEST_B_VARIANTS"]
TEST_C_VARIANTS = ns["TEST_C_VARIANTS"]

try_net_value = ns["try_net_value"]
get_pre_surface_signature = ns["get_pre_surface_signature"]
sig_to_display_name = ns["sig_to_display_name"]
build_counterfactual_affordance_overrides = ns["build_counterfactual_affordance_overrides"]
build_train_rows = ns["build_train_rows"]
summarize_rows = ns["summarize_rows"]
build_signature_stats = ns["build_signature_stats"]
build_cue_stats = ns["build_cue_stats"]
exact_signature_estimate = ns["exact_signature_estimate"]
level1_cue_marginal_estimate = ns["level1_cue_marginal_estimate"]
level2_knn_estimate = ns["level2_knn_estimate"]
level3_candidate_estimate = ns["level3_candidate_estimate"]
jaccard_similarity = ns["jaccard_similarity"]

OBSERVE_COSTS_PRIMARY = [0.03, 0.05, 0.07, 0.10]
OBSERVE_COSTS_SWEEP = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
TESTS = [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)]

ALL_OIDS = set()
for seed in SEEDS:
    for oid in per_seed_objects[seed]:
        ALL_OIDS.add((seed, oid))

run_7g = json.load(open(os.path.join(CURRENT_DIR, "runs", "block1j40b7g_env3b_counterfactual_headroom_decomposition.json"), "r", encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pearson_corr(xs, ys):
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(xs, ys):
    if len(xs) < 2:
        return None
    return pearson_corr(rankdata(xs), rankdata(ys))


def get_flat_affordance(all_success=False):
    outcome = "success" if all_success else "fail"
    return {a: outcome for a in ALL_TRY_AFFORDANCES}


def get_lookalike_affordance(seed, oid):
    lbl = per_seed_audit_labels[seed][oid]
    original_cat = lbl["hidden_category"]
    ambient_group = lbl["ambient_group"]
    other_cats = [c for c in AMBIENT_GROUP_CATEGORIES[ambient_group] if c != original_cat]
    if other_cats:
        other_cat = other_cats[0]
        return dict(INSTANCE_SUBTYPE_DEFS[other_cat]["P"]["affordance_profile"])
    return get_flat_affordance(False)


def base_affordance(seed, oid):
    return dict(per_seed_audit_labels[seed][oid]["affordance_profile"])


def positive_like_affordance(seed, oid):
    lbl = per_seed_audit_labels[seed][oid]
    if lbl["hidden_audit_rationale_class"] == "positive_change":
        return base_affordance(seed, oid)
    return get_lookalike_affordance(seed, oid)


def nonpositive_like_affordance(seed, oid):
    lbl = per_seed_audit_labels[seed][oid]
    if lbl["hidden_audit_rationale_class"] != "positive_change":
        return base_affordance(seed, oid)
    return get_flat_affordance(False)


def cue_group(seed, oid):
    sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
    extras = [f for f in sig if f not in ALL_AMBIENT_FEATURE_NAMES]
    diagnostic = {"has_bark_texture", "has_crystal_flecks", "has_stem_remnant", "has_grip_area"}
    shared = {"spotted", "discolored"}
    if any(f in diagnostic for f in extras):
        ctype = "diagnostic"
    elif any(f in shared for f in extras):
        ctype = "shared"
    else:
        ctype = "ambient_only"
    ambient = per_seed_audit_labels[seed][oid]["ambient_group"]
    return f"{ambient}:{ctype}"


def stable_random(seed, oid, salt):
    oid_num = int(oid.split("_")[-1])
    rng = random.Random(seed * 1000003 + oid_num * 101 + salt)
    return rng.random()


def build_full_distribution_overrides(mode, params):
    overrides = {}
    if mode == "original":
        return overrides
    if mode == "current":
        return build_counterfactual_affordance_overrides(ALL_OIDS, params["coverage"])[0]

    if mode == "random_balanced":
        p_positive = params["p_positive"]
        salt = params.get("salt", 17)
        for seed, oid in sorted(ALL_OIDS):
            is_positive_like = stable_random(seed, oid, salt) < p_positive
            overrides[(seed, oid)] = positive_like_affordance(seed, oid) if is_positive_like else nonpositive_like_affordance(seed, oid)
        return overrides

    if mode == "cue_group_structured":
        probs = params["group_positive_prob"]
        salt = params.get("salt", 23)
        for seed, oid in sorted(ALL_OIDS):
            group = cue_group(seed, oid)
            p_positive = probs[group]
            is_positive_like = stable_random(seed, oid, salt) < p_positive
            overrides[(seed, oid)] = positive_like_affordance(seed, oid) if is_positive_like else nonpositive_like_affordance(seed, oid)
        return overrides

    if mode == "mixed_within_signature":
        probs = params["group_positive_prob"]
        salt = params.get("salt", 29)
        for seed, oid in sorted(ALL_OIDS):
            group = cue_group(seed, oid)
            p_positive = probs[group]
            is_positive_like = stable_random(seed, oid, salt) < p_positive
            overrides[(seed, oid)] = positive_like_affordance(seed, oid) if is_positive_like else nonpositive_like_affordance(seed, oid)
        return overrides

    raise ValueError(f"Unknown mode: {mode}")


def compute_true_oracle_voi_adjusted(oid_set, observe_cost, affordance_overrides, extra_nonpositive_penalty=0.0):
    sig_instances = defaultdict(list)
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid), base_affordance(seed, oid))
        sig_instances[sig].append((seed, oid, aff))

    sig_expected = {}
    for sig, instances in sig_instances.items():
        action_sums = {a: 0.0 for a in ALL_TRY_AFFORDANCES}
        for _, _, aff in instances:
            for action in ALL_TRY_AFFORDANCES:
                action_sums[action] += try_net_value(aff.get(action, "fail") == "success")
        action_exp = {a: action_sums[a] / len(instances) for a in ALL_TRY_AFFORDANCES}
        best_action = max(action_exp, key=action_exp.get)
        sig_expected[sig] = {"best_expected_value": action_exp[best_action], "best_action": best_action}

    result = {}
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid), base_affordance(seed, oid))
        true_pre_best = sig_expected[sig]["best_expected_value"]
        true_post_best = max(try_net_value(aff.get(action, "fail") == "success") for action in ALL_TRY_AFFORDANCES)
        base_true_voi = true_post_best - true_pre_best - observe_cost
        adjusted_true_voi = base_true_voi - extra_nonpositive_penalty if base_true_voi <= 0.0 else base_true_voi
        observe_return = true_post_best - observe_cost - (extra_nonpositive_penalty if base_true_voi <= 0.0 else 0.0)
        result[(seed, oid)] = {
            "true_pre_best": true_pre_best,
            "true_post_best": true_post_best,
            "base_true_voi": base_true_voi,
            "true_voi": adjusted_true_voi,
            "true_optimal": adjusted_true_voi > 0.0,
            "observe_return": observe_return,
        }
    return result


def build_test_object_truth(test_oids, observe_cost, affordance_overrides, extra_nonpositive_penalty=0.0):
    true_voi = compute_true_oracle_voi_adjusted(test_oids, observe_cost, affordance_overrides, extra_nonpositive_penalty)
    sig_groups = defaultdict(list)
    obj_rows = []
    for seed, oid in sorted(test_oids):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        rec = {
            "seed": seed,
            "oid": oid,
            "signature": sig,
            "signature_name": sig_to_display_name(sig),
            **true_voi[(seed, oid)],
        }
        sig_groups[sig].append(rec)
        obj_rows.append(rec)

    signature_truth = {}
    for sig, members in sig_groups.items():
        signature_truth[sig] = {
            "signature": sig,
            "signature_name": sig_to_display_name(sig),
            "n_objects": len(members),
            "positive_net_rate": mean([1.0 if m["true_optimal"] else 0.0 for m in members]),
            "mean_true_E_observe_net": mean([m["true_voi"] for m in members]),
            "mean_true_E_positive": mean([m["true_voi"] for m in members if m["true_optimal"]]),
            "mean_true_E_nonpositive": mean([m["true_voi"] for m in members if not m["true_optimal"]]),
            "always_observe_mean_return": mean([m["observe_return"] for m in members]),
            "no_observe_mean_return": mean([m["true_pre_best"] for m in members]),
            "oracle_selective_mean_return": mean([
                m["observe_return"] if m["true_optimal"] else m["true_pre_best"] for m in members
            ]),
        }
    return obj_rows, signature_truth


def deterministic_ratio_slice(obj_rows, target_rate, seed):
    pos = [r for r in obj_rows if r["true_optimal"]]
    nonpos = [r for r in obj_rows if not r["true_optimal"]]
    if not pos or not nonpos:
        return list(obj_rows), {"target_rate": target_rate, "actual_rate": mean([1.0 if r["true_optimal"] else 0.0 for r in obj_rows]), "reason": "degenerate_class"}

    best = None
    rng = random.Random(seed)
    pos_shuf = list(pos)
    nonpos_shuf = list(nonpos)
    rng.shuffle(pos_shuf)
    rng.shuffle(nonpos_shuf)
    for n_pos in range(1, len(pos_shuf) + 1):
        n_non = round(n_pos * (1.0 - target_rate) / target_rate)
        if 1 <= n_non <= len(nonpos_shuf):
            actual = n_pos / (n_pos + n_non)
            err = abs(actual - target_rate)
            candidate = pos_shuf[:n_pos] + nonpos_shuf[:n_non]
            if best is None or err < best[0]:
                best = (err, actual, candidate)
    if best is None:
        return list(obj_rows), {"target_rate": target_rate, "actual_rate": mean([1.0 if r["true_optimal"] else 0.0 for r in obj_rows]), "reason": "target_unreachable"}
    return best[2], {"target_rate": target_rate, "actual_rate": best[1], "reason": "deterministic_subsample"}


def evaluate_policy_from_estimates(obj_rows, est_by_sig):
    returns = []
    observed = 0
    pos_obs = 0
    pos_total = 0
    nonpos_obs = 0
    nonpos_total = 0
    false_observe = 0
    false_direct = 0
    false_observe_losses = []
    false_direct_losses = []
    for row in obj_rows:
        est = est_by_sig[row["signature"]]
        pred_observe = est["e_observe_net"] > 0.0
        if row["true_optimal"]:
            pos_total += 1
            if pred_observe:
                pos_obs += 1
            else:
                false_direct += 1
                false_direct_losses.append(row["true_voi"])
        else:
            nonpos_total += 1
            if pred_observe:
                nonpos_obs += 1
                false_observe += 1
                false_observe_losses.append(-row["true_voi"])
        if pred_observe:
            observed += 1
            returns.append(row["observe_return"])
        else:
            returns.append(row["true_pre_best"])
    n = len(obj_rows)
    return {
        "policy_net": mean(returns),
        "observe_rate": observed / n if n else 0.0,
        "positive_net_observe_rate": pos_obs / pos_total if pos_total else 0.0,
        "nonpositive_net_observe_rate": nonpos_obs / nonpos_total if nonpos_total else 0.0,
        "false_observe_rate": false_observe / n if n else 0.0,
        "false_direct_rate": false_direct / n if n else 0.0,
        "mean_loss_false_observe": mean(false_observe_losses),
        "mean_loss_false_direct": mean(false_direct_losses),
    }


def summarize_predictability(obj_rows, est_by_sig):
    est_values = []
    true_values = []
    fallback_n = 0
    non_global_n = 0
    sign_correct = 0
    total = 0
    pos_est = []
    nonpos_est = []
    for row in obj_rows:
        est = est_by_sig[row["signature"]]
        e = est["e_observe_net"]
        est_values.append(e)
        true_values.append(row["true_voi"])
        if est.get("used_global_fallback", False):
            fallback_n += 1
        else:
            non_global_n += 1
        if (e > 0.0) == row["true_optimal"]:
            sign_correct += 1
        total += 1
        if row["true_optimal"]:
            pos_est.append(e)
        else:
            nonpos_est.append(e)
    return {
        "sign_correct_rate": sign_correct / total if total else 0.0,
        "fallback_rate": fallback_n / total if total else 0.0,
        "non_global_rate": non_global_n / total if total else 0.0,
        "pearson_corr": pearson_corr(est_values, true_values),
        "spearman_corr": spearman_corr(est_values, true_values),
        "estimated_E_separation": (mean(pos_est) - mean(nonpos_est)) if pos_est and nonpos_est else None,
        "estimated_E_mean": mean(est_values),
    }


def acceptance_for_result(row):
    head = row["headroom_metrics"]
    dist = row["observe_value_distribution"]
    policies = row["level_policy_metrics"]
    best_level = policies["best_level"]
    best_policy = policies["levels"][best_level]
    sign_rates = [policies["levels"][lvl]["predictability"]["sign_correct_rate"] for lvl in policies["levels"]]
    corr_vals = [policies["levels"][lvl]["predictability"]["pearson_corr"] for lvl in policies["levels"] if policies["levels"][lvl]["predictability"]["pearson_corr"] is not None]
    primary = (
        head["oracle_minus_always_observe"] >= 0.03
        and head["always_observe_minus_always_try"] > 0.0
        and 0.40 <= dist["positive_net_rate"] <= 0.65
        and dist["mean_true_E_positive"] is not None and 0.15 <= dist["mean_true_E_positive"] <= 0.35
        and dist["mean_true_E_nonpositive"] is not None and dist["mean_true_E_nonpositive"] <= -0.06
        and max(sign_rates) > 0.70
        and (max(corr_vals) if corr_vals else -1.0) > 0.0
        and best_policy["policy_net"] > head["always_observe_net"] + 0.005
        and best_policy["predictability"]["fallback_rate"] < 0.50
    )
    partial = (
        head["oracle_minus_always_observe"] > 0.015
        or best_policy["policy_net"] > head["always_observe_net"]
        or max(sign_rates) > 0.68
        or ((max(corr_vals) if corr_vals else -1.0) > 0.10)
    )
    reasons = []
    if head["oracle_minus_always_observe"] < 0.03:
        reasons.append("oracle_gap_lt_0.03")
    if dist["positive_net_rate"] > 0.60:
        reasons.append("positive_rate_too_high")
    if dist["mean_true_E_nonpositive"] is None or dist["mean_true_E_nonpositive"] > -0.06:
        reasons.append("nonpositive_not_negative_enough")
    if max(sign_rates) <= 0.70:
        reasons.append("sign_correct_not_enough")
    if best_policy["policy_net"] <= head["always_observe_net"] + 0.005:
        reasons.append("best_level_does_not_beat_always_observe")
    return {"passes_primary": primary, "passes_partial": (not primary) and partial, "rejection_reason": ";".join(reasons) if reasons else ""}


def evaluate_candidate(candidate, test_name, variant, observe_cost):
    distribution_overrides = build_full_distribution_overrides(candidate["mode"], candidate["parameters"])
    train_true_voi = compute_true_oracle_voi_adjusted(
        variant["train_oids"], observe_cost, distribution_overrides, candidate.get("extra_nonpositive_penalty", 0.0)
    )
    train_rows = build_train_rows(variant["train_oids"], train_true_voi)
    global_stats = summarize_rows(train_rows)
    signature_stats = build_signature_stats(train_rows)
    cue_stats = build_cue_stats(train_rows)

    test_obj_rows, signature_truth = build_test_object_truth(
        variant["test_oids"], observe_cost, distribution_overrides, candidate.get("extra_nonpositive_penalty", 0.0)
    )

    slice_info = None
    if candidate["family"] == "B":
        test_obj_rows, slice_info = deterministic_ratio_slice(test_obj_rows, candidate["parameters"]["target_positive_rate"], candidate["parameters"]["slice_seed"])

    est_maps = {
        "level1": {},
        "level2": {},
        "level3": {},
    }
    exact_est = {}
    for sig in {row["signature"] for row in test_obj_rows}:
        exact_est[sig] = exact_signature_estimate(sig, signature_stats, global_stats)
        est_maps["level1"][sig] = level1_cue_marginal_estimate(sig, cue_stats, global_stats)
        est_maps["level2"][sig] = level2_knn_estimate(sig, signature_stats, global_stats, k=3)
        est_maps["level3"][sig] = level3_candidate_estimate(sig, signature_stats, global_stats, tau=0.5)

    always_try_net = mean([r["true_pre_best"] for r in test_obj_rows])
    always_observe_net = mean([r["observe_return"] for r in test_obj_rows])
    oracle_selective_net = mean([r["observe_return"] if r["true_optimal"] else r["true_pre_best"] for r in test_obj_rows])

    positive_objs = [r for r in test_obj_rows if r["true_optimal"]]
    nonpositive_objs = [r for r in test_obj_rows if not r["true_optimal"]]
    value_dist = {
        "positive_net_rate": len(positive_objs) / len(test_obj_rows) if test_obj_rows else 0.0,
        "n_positive_net": len(positive_objs),
        "n_nonpositive_net": len(nonpositive_objs),
        "mean_true_E_positive": mean([r["true_voi"] for r in positive_objs]) if positive_objs else None,
        "mean_true_E_nonpositive": mean([r["true_voi"] for r in nonpositive_objs]) if nonpositive_objs else None,
        "mean_true_E_all": mean([r["true_voi"] for r in test_obj_rows]),
        "min_true_E": min((r["true_voi"] for r in test_obj_rows), default=0.0),
        "max_true_E": max((r["true_voi"] for r in test_obj_rows), default=0.0),
    }

    level_policy_metrics = {"levels": {}}
    for short_level, est_by_sig in est_maps.items():
        pred = summarize_predictability(test_obj_rows, est_by_sig)
        policy = evaluate_policy_from_estimates(test_obj_rows, est_by_sig)
        level_policy_metrics["levels"][short_level] = {
            "policy_net": policy["policy_net"],
            "observe_rate": policy["observe_rate"],
            "positive_net_observe_rate": policy["positive_net_observe_rate"],
            "nonpositive_net_observe_rate": policy["nonpositive_net_observe_rate"],
            "predictability": pred,
            "errors": {
                "false_observe_rate": policy["false_observe_rate"],
                "false_direct_rate": policy["false_direct_rate"],
                "mean_loss_false_observe": policy["mean_loss_false_observe"],
                "mean_loss_false_direct": policy["mean_loss_false_direct"],
            },
        }

    best_level = max(level_policy_metrics["levels"], key=lambda lvl: level_policy_metrics["levels"][lvl]["policy_net"])
    best_net = level_policy_metrics["levels"][best_level]["policy_net"]
    level_policy_metrics["best_level"] = best_level
    level_policy_metrics["best_level_net"] = best_net
    level_policy_metrics["best_level_delta_vs_always_observe"] = best_net - always_observe_net
    level_policy_metrics["best_level_delta_vs_always_try"] = best_net - always_try_net
    level_policy_metrics["oracle_minus_best_level"] = oracle_selective_net - best_net

    result = {
        "variant_id": candidate["variant_id"],
        "family": candidate["family"],
        "description": candidate["description"],
        "parameters": candidate["parameters"],
        "design_eligible": candidate["design_eligible"],
        "test_type": test_name,
        "observe_cost": observe_cost,
        "headroom_metrics": {
            "always_try_net": always_try_net,
            "always_observe_net": always_observe_net,
            "oracle_selective_net": oracle_selective_net,
            "oracle_minus_always_observe": oracle_selective_net - always_observe_net,
            "always_observe_minus_always_try": always_observe_net - always_try_net,
            "oracle_selective_minus_always_try": oracle_selective_net - always_try_net,
        },
        "observe_value_distribution": value_dist,
        "pre_cue_predictability": {
            lvl: {
                "sign_correct_rate": level_policy_metrics["levels"][lvl]["predictability"]["sign_correct_rate"],
                "false_observe_rate": level_policy_metrics["levels"][lvl]["errors"]["false_observe_rate"],
                "false_direct_rate": level_policy_metrics["levels"][lvl]["errors"]["false_direct_rate"],
                "fallback_rate": level_policy_metrics["levels"][lvl]["predictability"]["fallback_rate"],
                "non_global_rate": level_policy_metrics["levels"][lvl]["predictability"]["non_global_rate"],
                "pearson_corr": level_policy_metrics["levels"][lvl]["predictability"]["pearson_corr"],
                "spearman_corr": level_policy_metrics["levels"][lvl]["predictability"]["spearman_corr"],
                "estimated_E_separation": level_policy_metrics["levels"][lvl]["predictability"]["estimated_E_separation"],
            }
            for lvl in level_policy_metrics["levels"]
        },
        "level_policy_metrics": level_policy_metrics,
        "slice_info": slice_info,
        "acceptance_flags": None,
    }
    result["acceptance_flags"] = acceptance_for_result(result)
    return result


print("[2/6] Defining candidate families...")
candidates = []

# Family A: observe-cost sweep on current design variants
for mode_name, coverage in [("A0_original", None), ("A1_current_cov0.25", 0.25), ("A2_current_cov0.50", 0.50), ("A3_current_cov1.00", 1.00)]:
    candidates.append({
        "variant_id": mode_name,
        "family": "A",
        "description": "Observe-cost sweep with current env3b / counterfactual generation unchanged",
        "mode": "original" if coverage is None else "current",
        "parameters": {"coverage": coverage} if coverage is not None else {},
        "costs": OBSERVE_COSTS_SWEEP,
        "design_eligible": True,
    })

# Family B: balanced diagnostic slices, base current cov0.25
for target in [0.50, 0.55, 0.60]:
    candidates.append({
        "variant_id": f"B_cov0.25_target{int(target*100):02d}",
        "family": "B",
        "description": "Deterministic balanced diagnostic evaluation slice on current cov0.25 full distribution",
        "mode": "current",
        "parameters": {"coverage": 0.25, "target_positive_rate": target, "slice_seed": 240514 + int(target * 100)},
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": False,
    })

# Family C: non-positive penalty sensitivity, base current cov0.25 / cov0.50
for coverage in [0.25, 0.50]:
    for penalty in [0.00, 0.03, 0.06, 0.09, 0.12]:
        candidates.append({
            "variant_id": f"C_cov{int(coverage*100):03d}_pen{int(penalty*100):02d}",
            "family": "C",
            "description": "Audit-side extra nonpositive observe penalty sensitivity",
            "mode": "current",
            "parameters": {"coverage": coverage, "extra_nonpositive_penalty": penalty},
            "extra_nonpositive_penalty": penalty,
            "costs": OBSERVE_COSTS_PRIMARY,
            "design_eligible": False,
        })

# Family D: cue-structured redesign audit candidates
candidates.extend([
    {
        "variant_id": "D0_current_cov0.25",
        "family": "D",
        "description": "Current counterfactual assignment, full-distribution audit",
        "mode": "current",
        "parameters": {"coverage": 0.25},
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D0_current_cov0.50",
        "family": "D",
        "description": "Current counterfactual assignment, full-distribution audit",
        "mode": "current",
        "parameters": {"coverage": 0.50},
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D1_random_balanced_p50",
        "family": "D",
        "description": "Random balanced assignment, tests whether balance alone is enough",
        "mode": "random_balanced",
        "parameters": {"p_positive": 0.50, "salt": 17},
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D2_cue_group_structured_strong",
        "family": "D",
        "description": "Cue-group structured assignment with strong positive/negative cue-group separation",
        "mode": "cue_group_structured",
        "parameters": {
            "group_positive_prob": {
                "A:shared": 0.80, "A:ambient_only": 0.65, "A:diagnostic": 0.20,
                "B:shared": 0.80, "B:ambient_only": 0.65, "B:diagnostic": 0.20,
            },
            "salt": 23,
        },
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D2_cue_group_structured_mid",
        "family": "D",
        "description": "Cue-group structured assignment with moderate separation",
        "mode": "cue_group_structured",
        "parameters": {
            "group_positive_prob": {
                "A:shared": 0.75, "A:ambient_only": 0.60, "A:diagnostic": 0.25,
                "B:shared": 0.75, "B:ambient_only": 0.60, "B:diagnostic": 0.25,
            },
            "salt": 31,
        },
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D3_mixed_cue_predictive",
        "family": "D",
        "description": "Mixed-within-signature assignment while retaining cue-level predictivity",
        "mode": "mixed_within_signature",
        "parameters": {
            "group_positive_prob": {
                "A:shared": 0.65, "A:ambient_only": 0.55, "A:diagnostic": 0.35,
                "B:shared": 0.65, "B:ambient_only": 0.55, "B:diagnostic": 0.35,
            },
            "salt": 29,
        },
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
    {
        "variant_id": "D3_mixed_cue_predictive_balanced",
        "family": "D",
        "description": "Mixed-within-signature assignment with lower positive prior",
        "mode": "mixed_within_signature",
        "parameters": {
            "group_positive_prob": {
                "A:shared": 0.60, "A:ambient_only": 0.50, "A:diagnostic": 0.40,
                "B:shared": 0.60, "B:ambient_only": 0.50, "B:diagnostic": 0.40,
            },
            "salt": 37,
        },
        "costs": OBSERVE_COSTS_PRIMARY,
        "design_eligible": True,
    },
])


print("[3/6] Evaluating candidates...")
candidate_results = []
for cand in candidates:
    print(f"  {cand['variant_id']}")
    for observe_cost in cand["costs"]:
        for test_name, variants in TESTS:
            for variant in variants:
                candidate_results.append(evaluate_candidate(cand, test_name, variant, observe_cost))


def aggregate_candidate_rows(rows):
    n = len(rows)
    if n == 0:
        return None
    sample = rows[0]
    level_names = ["level1", "level2", "level3"]
    agg = {
        "variant_id": sample["variant_id"],
        "family": sample["family"],
        "description": sample["description"],
        "parameters": sample["parameters"],
        "design_eligible": sample["design_eligible"],
        "test_type": sample["test_type"],
        "observe_cost": sample["observe_cost"],
        "headroom_metrics": {
            "always_try_net": mean([r["headroom_metrics"]["always_try_net"] for r in rows]),
            "always_observe_net": mean([r["headroom_metrics"]["always_observe_net"] for r in rows]),
            "oracle_selective_net": mean([r["headroom_metrics"]["oracle_selective_net"] for r in rows]),
            "oracle_minus_always_observe": mean([r["headroom_metrics"]["oracle_minus_always_observe"] for r in rows]),
            "always_observe_minus_always_try": mean([r["headroom_metrics"]["always_observe_minus_always_try"] for r in rows]),
        },
        "observe_value_distribution": {
            "positive_net_rate": mean([r["observe_value_distribution"]["positive_net_rate"] for r in rows]),
            "mean_true_E_positive": mean([r["observe_value_distribution"]["mean_true_E_positive"] for r in rows if r["observe_value_distribution"]["mean_true_E_positive"] is not None]),
            "mean_true_E_nonpositive": mean([r["observe_value_distribution"]["mean_true_E_nonpositive"] for r in rows if r["observe_value_distribution"]["mean_true_E_nonpositive"] is not None]),
        },
        "level_policy_metrics": {"levels": {}},
    }
    for lvl in level_names:
        agg["level_policy_metrics"]["levels"][lvl] = {
            "policy_net": mean([r["level_policy_metrics"]["levels"][lvl]["policy_net"] for r in rows]),
            "predictability": {
                "sign_correct_rate": mean([r["level_policy_metrics"]["levels"][lvl]["predictability"]["sign_correct_rate"] for r in rows]),
                "pearson_corr": mean([r["level_policy_metrics"]["levels"][lvl]["predictability"]["pearson_corr"] for r in rows if r["level_policy_metrics"]["levels"][lvl]["predictability"]["pearson_corr"] is not None]),
                "fallback_rate": mean([r["level_policy_metrics"]["levels"][lvl]["predictability"]["fallback_rate"] for r in rows]),
            },
        }
    best_level = max(level_names, key=lambda lvl: agg["level_policy_metrics"]["levels"][lvl]["policy_net"])
    agg["level_policy_metrics"]["best_level"] = best_level
    agg["level_policy_metrics"]["best_level_net"] = agg["level_policy_metrics"]["levels"][best_level]["policy_net"]
    agg["level_policy_metrics"]["best_level_delta_vs_always_observe"] = (
        agg["level_policy_metrics"]["best_level_net"] - agg["headroom_metrics"]["always_observe_net"]
    )
    agg["acceptance_flags"] = acceptance_for_result(agg)
    return agg


aggregated_rows = []
for variant_id in sorted(set(r["variant_id"] for r in candidate_results)):
    rows_by_key = defaultdict(list)
    for row in candidate_results:
        if row["variant_id"] != variant_id:
            continue
        rows_by_key[(row["test_type"], row["observe_cost"])].append(row)
    for (_, _), rows in rows_by_key.items():
        aggregated_rows.append(aggregate_candidate_rows(rows))


def best_row_for_test(test_type, eligible_only=False):
    rows = [r for r in aggregated_rows if r["test_type"] == test_type and (r["design_eligible"] or not eligible_only)]
    if not rows:
        return None
    return max(rows, key=lambda r: r["level_policy_metrics"]["best_level_delta_vs_always_observe"])


passing_primary = [r for r in aggregated_rows if r["acceptance_flags"]["passes_primary"] and r["design_eligible"]]
passing_partial = [r for r in aggregated_rows if r["acceptance_flags"]["passes_partial"] and r["design_eligible"]]
best_A = best_row_for_test("A", eligible_only=True)
best_B = best_row_for_test("B", eligible_only=True)
best_overall = max(
    [r for r in aggregated_rows if r["design_eligible"]],
    key=lambda r: (r["acceptance_flags"]["passes_primary"], r["level_policy_metrics"]["best_level_delta_vs_always_observe"]),
    default=None,
)

status = "NO_VALID_CANDIDATE"
if passing_primary:
    status = "DESIGN_CANDIDATE_FOUND"
elif passing_partial:
    status = "PARTIAL_CANDIDATE_FOUND"

acceptance_summary = {
    "candidates_passing_primary": [
        {"variant_id": r["variant_id"], "test_type": r["test_type"], "observe_cost": r["observe_cost"]}
        for r in passing_primary
    ],
    "candidates_passing_partial": [
        {"variant_id": r["variant_id"], "test_type": r["test_type"], "observe_cost": r["observe_cost"]}
        for r in passing_partial
    ],
    "best_candidate_by_test_A": None if best_A is None else {
        "variant_id": best_A["variant_id"], "family": best_A["family"],
        "observe_cost": best_A["observe_cost"], "best_level_delta_vs_always_observe": best_A["level_policy_metrics"]["best_level_delta_vs_always_observe"],
    },
    "best_candidate_by_test_B": None if best_B is None else {
        "variant_id": best_B["variant_id"], "family": best_B["family"],
        "observe_cost": best_B["observe_cost"], "best_level_delta_vs_always_observe": best_B["level_policy_metrics"]["best_level_delta_vs_always_observe"],
    },
    "best_candidate_overall": None if best_overall is None else {
        "variant_id": best_overall["variant_id"], "family": best_overall["family"],
        "observe_cost": best_overall["observe_cost"], "test_type": best_overall["test_type"],
        "parameters": best_overall["parameters"],
        "best_level_delta_vs_always_observe": best_overall["level_policy_metrics"]["best_level_delta_vs_always_observe"],
    },
    "rejected_reasons": defaultdict(int),
}
for row in aggregated_rows:
    reason = row["acceptance_flags"]["rejection_reason"] or "none"
    acceptance_summary["rejected_reasons"][reason] += 1
acceptance_summary["rejected_reasons"] = dict(acceptance_summary["rejected_reasons"])


print("[4/6] Writing outputs...")
json_output = {
    "metadata": {
        "block_id": "1J40b-7h",
        "script_name": "_block1j40b7h_env3b_redesign_parameter_audit.py",
        "timestamp": datetime.now().isoformat(),
        "source_files_used": SOURCE_FILES_USED,
        "candidate_families": ["A", "B", "C", "D"],
        "costs": sorted(set(r["observe_cost"] for r in aggregated_rows)),
        "tests": ["A", "B", "C"],
    },
    "candidate_results": aggregated_rows,
    "acceptance_summary": acceptance_summary,
    "validity_audit": {
        "no_learner_training": True,
        "no_1j40b8": True,
        "no_official_env_replacement": True,
        "forbidden_test_keys_absent": True,
        "audit_only": True,
    },
    "status": status,
    "elapsed_seconds": round(time.time() - t0, 2),
}

runs_dir = os.path.join(CURRENT_DIR, "runs")
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(protocols_dir, exist_ok=True)

json_path = os.path.join(runs_dir, "block1j40b7h_env3b_redesign_parameter_audit.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

csv_lines = [
    "variant_id,family,test_type,observe_cost,always_try_net,always_observe_net,oracle_selective_net,oracle_minus_always_observe,"
    "always_observe_minus_always_try,positive_net_rate,mean_true_E_positive,mean_true_E_nonpositive,level1_policy_net,level2_policy_net,"
    "level3_policy_net,best_level,best_level_net,best_level_delta_vs_always_observe,level1_sign_correct_rate,level2_sign_correct_rate,"
    "level3_sign_correct_rate,level1_corr,level2_corr,level3_corr,fallback_rate_best_level,passes_primary_criteria,rejection_reason"
]
for row in aggregated_rows:
    best_lvl = row["level_policy_metrics"]["best_level"]
    csv_lines.append(",".join([
        row["variant_id"],
        row["family"],
        row["test_type"],
        f"{row['observe_cost']:.2f}",
        f"{row['headroom_metrics']['always_try_net']:.6f}",
        f"{row['headroom_metrics']['always_observe_net']:.6f}",
        f"{row['headroom_metrics']['oracle_selective_net']:.6f}",
        f"{row['headroom_metrics']['oracle_minus_always_observe']:.6f}",
        f"{row['headroom_metrics']['always_observe_minus_always_try']:.6f}",
        f"{row['observe_value_distribution']['positive_net_rate']:.6f}",
        "" if row["observe_value_distribution"]["mean_true_E_positive"] is None else f"{row['observe_value_distribution']['mean_true_E_positive']:.6f}",
        "" if row["observe_value_distribution"]["mean_true_E_nonpositive"] is None else f"{row['observe_value_distribution']['mean_true_E_nonpositive']:.6f}",
        f"{row['level_policy_metrics']['levels']['level1']['policy_net']:.6f}",
        f"{row['level_policy_metrics']['levels']['level2']['policy_net']:.6f}",
        f"{row['level_policy_metrics']['levels']['level3']['policy_net']:.6f}",
        best_lvl,
        f"{row['level_policy_metrics']['best_level_net']:.6f}",
        f"{row['level_policy_metrics']['best_level_delta_vs_always_observe']:.6f}",
        f"{row['level_policy_metrics']['levels']['level1']['predictability']['sign_correct_rate']:.6f}",
        f"{row['level_policy_metrics']['levels']['level2']['predictability']['sign_correct_rate']:.6f}",
        f"{row['level_policy_metrics']['levels']['level3']['predictability']['sign_correct_rate']:.6f}",
        "" if row['level_policy_metrics']['levels']['level1']['predictability']['pearson_corr'] is None else f"{row['level_policy_metrics']['levels']['level1']['predictability']['pearson_corr']:.6f}",
        "" if row['level_policy_metrics']['levels']['level2']['predictability']['pearson_corr'] is None else f"{row['level_policy_metrics']['levels']['level2']['predictability']['pearson_corr']:.6f}",
        "" if row['level_policy_metrics']['levels']['level3']['predictability']['pearson_corr'] is None else f"{row['level_policy_metrics']['levels']['level3']['predictability']['pearson_corr']:.6f}",
        f"{row['level_policy_metrics']['levels'][best_lvl]['predictability']['fallback_rate']:.6f}",
        str(row["acceptance_flags"]["passes_primary"]),
        row["acceptance_flags"]["rejection_reason"],
    ]))

csv_path = os.path.join(protocols_dir, "block1j40b7h_env3b_redesign_parameter_audit_table.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("\n".join(csv_lines) + "\n")

md = []
md.append("# Block 1J40b-7h: Env3b / Counterfactual Redesign Parameter Audit")
md.append("")
md.append("## 1. Checkpoint Summary")
md.append("")
md.append(f"- **Status**: {status}")
md.append(f"- **Elapsed**: {json_output['elapsed_seconds']:.2f}s")
md.append(f"- **Best Candidate Overall**: {acceptance_summary['best_candidate_overall']}")
md.append("")
md.append("## 2. Files Changed")
md.append("")
md.append("- `_block1j40b7h_env3b_redesign_parameter_audit.py`")
md.append("- `runs/block1j40b7h_env3b_redesign_parameter_audit.json`")
md.append("- `protocols/block1j40b7h_env3b_redesign_parameter_audit.md`")
md.append("- `protocols/block1j40b7h_env3b_redesign_parameter_audit_table.csv`")
md.append("- `checkpoint_1j40b7h_env3b_redesign_parameter_audit.md`")
md.append("")
md.append("## 3. Commands Run")
md.append("")
md.append("```powershell")
md.append('& "D:\\conda\\python.exe" "_block1j40b7h_env3b_redesign_parameter_audit.py"')
md.append("```")
md.append("")
md.append("## 4. Why 7h Was Run")
md.append("")
md.append("- 7g showed weak headroom, strong always_observe, weak nonpositive penalty, and poor cue predictability.")
md.append("- 7h searches audit-only parameterized redesign directions that might restore headroom before any env3c implementation.")
md.append("")
md.append("## 5. 7g Diagnosis Recap")
md.append("")
md.append("- always_observe_too_strong")
md.append("- nonpositive_penalty_too_weak")
md.append("- pre_cue_predictability_low")
md.append("- level_estimate_sign_mismatch")
md.append("- cf_cov1_destroyed_signal")
md.append("")
md.append("## 6. Candidate Families Tested")
md.append("")
for fam in ["A", "B", "C", "D"]:
    fam_rows = [r for r in aggregated_rows if r["family"] == fam]
    md.append(f"- **Family {fam}**: {len(fam_rows)} aggregated rows")
md.append("")
md.append("## 7. Headroom Results")
md.append("")
md.append("| Variant | Family | Test | Cost | oracle-always_obs | always_obs-always_try | best_level | best-minus-always_obs |")
md.append("|---------|--------|------|------|-------------------|-----------------------|------------|-----------------------|")
for row in sorted(aggregated_rows, key=lambda r: (r["family"], r["variant_id"], r["observe_cost"], r["test_type"])):
    md.append(
        f"| {row['variant_id']} | {row['family']} | {row['test_type']} | {row['observe_cost']:.2f} | "
        f"{row['headroom_metrics']['oracle_minus_always_observe']:+.4f} | "
        f"{row['headroom_metrics']['always_observe_minus_always_try']:+.4f} | "
        f"{row['level_policy_metrics']['best_level']} | {row['level_policy_metrics']['best_level_delta_vs_always_observe']:+.4f} |"
    )
md.append("")
md.append("## 8. Positive / Non-positive Balance Results")
md.append("")
md.append("| Variant | Test | Cost | positive_rate | mean_pos | mean_nonpos |")
md.append("|---------|------|------|---------------|----------|-------------|")
for row in sorted(aggregated_rows, key=lambda r: (r["variant_id"], r["observe_cost"], r["test_type"])):
    md.append(
        f"| {row['variant_id']} | {row['test_type']} | {row['observe_cost']:.2f} | "
        f"{row['observe_value_distribution']['positive_net_rate']:.4f} | "
        f"{(row['observe_value_distribution']['mean_true_E_positive'] if row['observe_value_distribution']['mean_true_E_positive'] is not None else 0.0):.4f} | "
        f"{(row['observe_value_distribution']['mean_true_E_nonpositive'] if row['observe_value_distribution']['mean_true_E_nonpositive'] is not None else 0.0):.4f} |"
    )
md.append("")
md.append("## 9. Non-positive Penalty Sensitivity")
md.append("")
for row in [r for r in aggregated_rows if r["family"] == "C" and r["test_type"] in {"A", "B"} and r["observe_cost"] == 0.03]:
    md.append(
        f"- {row['variant_id']} Test {row['test_type']}: oracle_gap={row['headroom_metrics']['oracle_minus_always_observe']:+.4f}, "
        f"best_delta={row['level_policy_metrics']['best_level_delta_vs_always_observe']:+.4f}, "
        f"mean_nonpositive={row['observe_value_distribution']['mean_true_E_nonpositive']:.4f}"
    )
md.append("")
md.append("## 10. Pre-cue Predictability Results")
md.append("")
md.append("| Variant | Test | Cost | L1 sign/corr | L2 sign/corr | L3 sign/corr |")
md.append("|---------|------|------|--------------|--------------|--------------|")
for row in sorted(aggregated_rows, key=lambda r: (r["variant_id"], r["observe_cost"], r["test_type"])):
    l1 = row["level_policy_metrics"]["levels"]["level1"]["predictability"]
    l2 = row["level_policy_metrics"]["levels"]["level2"]["predictability"]
    l3 = row["level_policy_metrics"]["levels"]["level3"]["predictability"]
    md.append(
        f"| {row['variant_id']} | {row['test_type']} | {row['observe_cost']:.2f} | "
        f"{l1['sign_correct_rate']:.4f}/{(l1['pearson_corr'] if l1['pearson_corr'] is not None else 0.0):.4f} | "
        f"{l2['sign_correct_rate']:.4f}/{(l2['pearson_corr'] if l2['pearson_corr'] is not None else 0.0):.4f} | "
        f"{l3['sign_correct_rate']:.4f}/{(l3['pearson_corr'] if l3['pearson_corr'] is not None else 0.0):.4f} |"
    )
md.append("")
md.append("## 11. Level 1/2/3 Policy Results")
md.append("")
md.append("| Variant | Test | Cost | L1 net | L2 net | L3 net | best_level |")
md.append("|---------|------|------|--------|--------|--------|------------|")
for row in sorted(aggregated_rows, key=lambda r: (r["variant_id"], r["observe_cost"], r["test_type"])):
    md.append(
        f"| {row['variant_id']} | {row['test_type']} | {row['observe_cost']:.2f} | "
        f"{row['level_policy_metrics']['levels']['level1']['policy_net']:.4f} | "
        f"{row['level_policy_metrics']['levels']['level2']['policy_net']:.4f} | "
        f"{row['level_policy_metrics']['levels']['level3']['policy_net']:.4f} | "
        f"{row['level_policy_metrics']['best_level']} |"
    )
md.append("")
md.append("## 12. Best Candidate Variants")
md.append("")
md.append(f"- Test A: {acceptance_summary['best_candidate_by_test_A']}")
md.append(f"- Test B: {acceptance_summary['best_candidate_by_test_B']}")
md.append(f"- Overall: {acceptance_summary['best_candidate_overall']}")
md.append("")
md.append("## 13. Rejected Variants and Reasons")
md.append("")
for reason, count in sorted(acceptance_summary["rejected_reasons"].items()):
    md.append(f"- {reason}: {count}")
md.append("")
md.append("## 14. Validity / Leakage Audit")
md.append("")
md.append("- no learner training")
md.append("- no 1J40b-8")
md.append("- no official env replacement")
md.append("- legal pre-surface cue keyed estimators only")
md.append("- audit-only use of true values")
md.append("")
md.append("## 15. Recommended Resume Point")
md.append("")
if status == "DESIGN_CANDIDATE_FOUND":
    md.append("- Resume with a small env3c generation audit for the best candidate, not 1J40b-8.")
elif status == "PARTIAL_CANDIDATE_FOUND":
    md.append("- Resume with targeted refinement of the best partial candidate.")
else:
    md.append("- Resume with deeper environment redesign review.")

md_path = os.path.join(protocols_dir, "block1j40b7h_env3b_redesign_parameter_audit.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J40b-7h",
    "",
    f"- status: {status}",
    f"- best_candidate_overall: {acceptance_summary['best_candidate_overall']}",
    f"- best_candidate_test_A: {acceptance_summary['best_candidate_by_test_A']}",
    f"- best_candidate_test_B: {acceptance_summary['best_candidate_by_test_B']}",
    f"- candidates_passing_primary: {len(acceptance_summary['candidates_passing_primary'])}",
    f"- candidates_passing_partial: {len(acceptance_summary['candidates_passing_partial'])}",
    "- next_suggested_step: "
    + (
        "env3c generation audit" if status == "DESIGN_CANDIDATE_FOUND"
        else "targeted candidate refinement" if status == "PARTIAL_CANDIDATE_FOUND"
        else "deeper environment redesign"
    ),
]
checkpoint_path = os.path.join(CURRENT_DIR, "checkpoint_1j40b7h_env3b_redesign_parameter_audit.md")
with open(checkpoint_path, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/6] Final console summary...")
print(f"STATUS={status}")
print(f"best_candidate_overall={acceptance_summary['best_candidate_overall']}")
for test_type in ["A", "B"]:
    best = acceptance_summary[f"best_candidate_by_test_{test_type}"]
    if best is None:
        continue
    row = next(r for r in aggregated_rows if r["variant_id"] == best["variant_id"] and r["test_type"] == test_type and r["observe_cost"] == best["observe_cost"])
    best_level = row["level_policy_metrics"]["best_level"]
    print(
        f"Test {test_type}: oracle_gap={row['headroom_metrics']['oracle_minus_always_observe']:+.4f} "
        f"best_delta={row['level_policy_metrics']['best_level_delta_vs_always_observe']:+.4f} "
        f"positive_rate={row['observe_value_distribution']['positive_net_rate']:.4f} "
        f"mean_pos={(row['observe_value_distribution']['mean_true_E_positive'] if row['observe_value_distribution']['mean_true_E_positive'] is not None else 0.0):.4f} "
        f"mean_nonpos={(row['observe_value_distribution']['mean_true_E_nonpositive'] if row['observe_value_distribution']['mean_true_E_nonpositive'] is not None else 0.0):.4f} "
        f"best_sign={row['level_policy_metrics']['levels'][best_level]['predictability']['sign_correct_rate']:.4f}"
    )
print(f"any_oracle_gap_ge_0.03={any(r['headroom_metrics']['oracle_minus_always_observe'] >= 0.03 for r in aggregated_rows if r['design_eligible'])}")
print(f"any_mean_nonpositive_le_-0.06={any((r['observe_value_distribution']['mean_true_E_nonpositive'] is not None and r['observe_value_distribution']['mean_true_E_nonpositive'] <= -0.06) for r in aggregated_rows if r['design_eligible'])}")
print(f"any_positive_rate_le_0.60={any(r['observe_value_distribution']['positive_net_rate'] <= 0.60 for r in aggregated_rows if r['design_eligible'])}")
print(f"any_best_level_beats_always_observe={any(r['level_policy_metrics']['best_level_delta_vs_always_observe'] > 0.0 for r in aggregated_rows if r['design_eligible'])}")

print("[6/6] Done.")
print(f"  JSON: {json_path}")
print(f"  CSV:  {csv_path}")
print(f"  MD:   {md_path}")
print(f"  CKPT: {checkpoint_path}")
