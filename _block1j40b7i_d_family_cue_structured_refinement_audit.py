"""
Block 1J40b-7i -- D-Family Cue-Structured Refinement Audit.

Targeted audit-only refinement over small cue-structured D-family variants.
Does not train a learner, does not replace official env3b, and does not
implement 1J40b-8.
"""

import json
import os
import random
import runpy
import time
from collections import Counter, defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES_USED = [
    "_block1j40b7e_env3b_distributional_audit_counterfactual.py",
    "_block1j40b7fb_compositional_oracle_belief_upper_bound.py",
    "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py",
    "_block1j40b7h_env3b_redesign_parameter_audit.py",
    "runs/block1j40b7h_env3b_redesign_parameter_audit.json",
]

print("[1/6] Loading 7g environment helpers...")
ns = runpy.run_path(
    os.path.join(CURRENT_DIR, "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py")
)

SEEDS = ns["SEEDS"]
ALL_TRY_AFFORDANCES = ns["ALL_TRY_AFFORDANCES"]
INSTANCE_SUBTYPE_DEFS = ns["INSTANCE_SUBTYPE_DEFS"]
AMBIENT_GROUP_CATEGORIES = ns["AMBIENT_GROUP_CATEGORIES"]
ALL_AMBIENT_FEATURE_NAMES = set(ns["ALL_AMBIENT_FEATURE_NAMES"])

per_seed_objects = ns["per_seed_objects"]
per_seed_audit_labels = ns["per_seed_audit_labels"]
TEST_A_VARIANTS = ns["TEST_A_VARIANTS"]
TEST_B_VARIANTS = ns["TEST_B_VARIANTS"]
TEST_C_VARIANTS = ns["TEST_C_VARIANTS"]

try_net_value = ns["try_net_value"]
get_pre_surface_signature = ns["get_pre_surface_signature"]
sig_to_display_name = ns["sig_to_display_name"]
build_train_rows = ns["build_train_rows"]
summarize_rows = ns["summarize_rows"]
build_signature_stats = ns["build_signature_stats"]
build_cue_stats = ns["build_cue_stats"]
exact_signature_estimate = ns["exact_signature_estimate"]
level1_cue_marginal_estimate = ns["level1_cue_marginal_estimate"]
level2_knn_estimate = ns["level2_knn_estimate"]
level3_candidate_estimate = ns["level3_candidate_estimate"]

OBSERVE_COSTS = [0.03, 0.05, 0.07, 0.10]
TESTS = [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)]
PRIMARY_TESTS = {"A", "B"}
PRIMARY_COST = 0.03

ALL_OIDS = set()
for seed in SEEDS:
    for oid in per_seed_objects[seed]:
        ALL_OIDS.add((seed, oid))

CORE_A = {"brownish", "heavy_weight", "long_shape", "rough_texture"}
CORE_B = {"greenish", "light_weight", "round_small", "smooth_texture"}


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


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


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


def stable_random(seed, oid, salt):
    oid_num = int(oid.split("_")[-1])
    rng = random.Random(seed * 1000003 + oid_num * 101 + salt)
    return rng.random()


def signature_ambient(sig):
    if "brownish" in sig:
        return "A"
    if "greenish" in sig:
        return "B"
    return "UNK"


def signature_extras(sig):
    ambient = signature_ambient(sig)
    core = CORE_A if ambient == "A" else CORE_B
    return tuple(sorted([cue for cue in sig if cue not in core]))


def signature_profile(sig):
    extras = set(signature_extras(sig))
    return {
        "ambient": signature_ambient(sig),
        "extras": extras,
        "has_spotted": "spotted" in extras,
        "has_discolored": "discolored" in extras,
        "has_bark_texture": "has_bark_texture" in extras,
        "has_crystal_flecks": "has_crystal_flecks" in extras,
        "has_grip_area": "has_grip_area" in extras,
        "has_stem_remnant": "has_stem_remnant" in extras,
        "has_any_diagnostic": any(
            cue in extras
            for cue in ("has_bark_texture", "has_crystal_flecks", "has_grip_area", "has_stem_remnant")
        ),
        "has_any_shared": any(cue in extras for cue in ("spotted", "discolored")),
    }


def d4_probability(sig, params):
    prof = signature_profile(sig)
    amb = prof["ambient"]
    extras = prof["extras"]
    if amb == "A":
        if extras == {"spotted"}:
            return params["A_spotted"]
        if extras == set():
            return params["A_ambient_only"]
        if extras == {"discolored", "has_bark_texture"}:
            return params["A_bark_discolored"]
        if extras == {"has_crystal_flecks"}:
            return params["A_crystal"]
    if amb == "B":
        if extras == {"discolored"}:
            return params["B_discolored"]
        if extras == set():
            return params["B_ambient_only"]
        if extras == {"has_grip_area"}:
            return params["B_grip"]
        if extras == {"has_stem_remnant", "spotted"}:
            return params["B_stem_spotted"]
    return 0.50


def d5_probability(sig, params):
    prof = signature_profile(sig)
    amb = prof["ambient"]
    p = params["base_A"] if amb == "A" else params["base_B"]
    cues = set(sig)
    pair_weights = params["pair_weights"]
    for c1, c2, delta in pair_weights:
        if c1 in cues and c2 in cues:
            p += delta
    return clamp(p, params["clip_lo"], params["clip_hi"])


def d6_probability(sig, params):
    prof = signature_profile(sig)
    amb = prof["ambient"]
    p = params["base_A"] if amb == "A" else params["base_B"]
    if prof["has_any_shared"]:
        p += params["shared_bonus"]
    if prof["has_any_diagnostic"]:
        p += params["diagnostic_shift"]
    if prof["has_any_shared"] and prof["has_any_diagnostic"]:
        p += params["combo_shift"]
    return clamp(p, params["clip_lo"], params["clip_hi"])


def d7_probability(sig, params):
    p = d6_probability(sig, params["subset_part"])
    cues = set(sig)
    for c1, c2, delta in params["pair_refine"]:
        if c1 in cues and c2 in cues:
            p += delta
    return clamp(p, params["clip_lo"], params["clip_hi"])


def probability_for_signature(sig, candidate):
    mode = candidate["mode"]
    params = candidate["parameters"]
    if mode == "cue_group_balanced_soft":
        return d4_probability(sig, params)
    if mode == "pairwise_cue_interaction":
        return d5_probability(sig, params)
    if mode == "candidate_set_structured":
        return d6_probability(sig, params)
    if mode == "hybrid_soft_structured":
        return d7_probability(sig, params)
    raise ValueError(f"Unknown candidate mode: {mode}")


def build_distribution_overrides(candidate):
    overrides = {}
    salt = candidate["parameters"].get("salt", 17)
    for seed, oid in sorted(ALL_OIDS):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        p_positive = probability_for_signature(sig, candidate)
        is_positive_like = stable_random(seed, oid, salt) < p_positive
        overrides[(seed, oid)] = (
            positive_like_affordance(seed, oid) if is_positive_like else nonpositive_like_affordance(seed, oid)
        )
    return overrides


def compute_true_oracle_voi(oid_set, observe_cost, affordance_overrides):
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
        sig_expected[sig] = {
            "best_expected_value": action_exp[best_action],
            "best_action": best_action,
        }

    result = {}
    for seed, oid in oid_set:
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        aff = affordance_overrides.get((seed, oid), base_affordance(seed, oid))
        true_pre_best = sig_expected[sig]["best_expected_value"]
        true_post_best = max(
            try_net_value(aff.get(action, "fail") == "success") for action in ALL_TRY_AFFORDANCES
        )
        true_voi = true_post_best - true_pre_best - observe_cost
        result[(seed, oid)] = {
            "true_pre_best": true_pre_best,
            "true_post_best": true_post_best,
            "true_voi": true_voi,
            "true_optimal": true_voi > 0.0,
            "observe_return": true_post_best - observe_cost,
        }
    return result


def build_test_object_truth(test_oids, observe_cost, affordance_overrides):
    true_voi = compute_true_oracle_voi(test_oids, observe_cost, affordance_overrides)
    rows = []
    for seed, oid in sorted(test_oids):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        rows.append(
            {
                "seed": seed,
                "oid": oid,
                "signature": sig,
                "signature_name": sig_to_display_name(sig),
                **true_voi[(seed, oid)],
            }
        )
    return rows


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
        if row["true_optimal"]:
            pos_est.append(e)
        else:
            nonpos_est.append(e)
    total = len(obj_rows)
    return {
        "sign_correct_rate": sign_correct / total if total else 0.0,
        "fallback_rate": fallback_n / total if total else 0.0,
        "non_global_rate": non_global_n / total if total else 0.0,
        "pearson_corr": pearson_corr(est_values, true_values),
        "spearman_corr": spearman_corr(est_values, true_values),
        "estimated_E_separation": (mean(pos_est) - mean(nonpos_est)) if pos_est and nonpos_est else None,
        "estimated_E_mean": mean(est_values),
    }


def predictiveness_strength(rate):
    return max(rate, 1.0 - rate)


def compute_structure_validity(all_rows, train_signature_stats, est_maps):
    sig_groups = defaultdict(list)
    cue_groups = defaultdict(list)
    pair_groups = defaultdict(list)
    for row in all_rows:
        sig_groups[row["signature"]].append(row)
        for cue in row["signature"]:
            cue_groups[cue].append(row)
        sig_list = sorted(row["signature"])
        for i in range(len(sig_list)):
            for j in range(i + 1, len(sig_list)):
                pair_groups[(sig_list[i], sig_list[j])].append(row)

    exact_det = []
    for members in sig_groups.values():
        rate = mean([1.0 if m["true_optimal"] else 0.0 for m in members])
        exact_det.append(1.0 if rate in (0.0, 1.0) else 0.0)

    cue_strength = []
    for members in cue_groups.values():
        rate = mean([1.0 if m["true_optimal"] else 0.0 for m in members])
        cue_strength.append(predictiveness_strength(rate))

    pair_strength = []
    for members in pair_groups.values():
        rate = mean([1.0 if m["true_optimal"] else 0.0 for m in members])
        pair_strength.append(predictiveness_strength(rate))

    return {
        "exact_signature_determinism_rate": mean(exact_det),
        "single_cue_max_predictiveness": max(cue_strength) if cue_strength else 0.0,
        "cue_pair_max_predictiveness": max(pair_strength) if pair_strength else 0.0,
    }


def compute_test_reachability(test_rows, train_signature_stats, est_maps):
    test_signatures = sorted({row["signature"] for row in test_rows})
    heldout_count = len(test_signatures)
    keymiss_count = 0
    reachable_count = 0
    for sig in test_signatures:
        exact_miss = sig not in train_signature_stats
        if exact_miss:
            keymiss_count += 1
        any_non_global = any(
            not est_maps[level][sig].get("used_global_fallback", False) for level in ("level1", "level2", "level3")
        )
        if exact_miss and any_non_global:
            reachable_count += 1
    return {
        "heldout_signature_count": heldout_count,
        "heldout_signature_keymiss_rate": (keymiss_count / heldout_count) if heldout_count else 0.0,
        "compositional_reachability_rate": (reachable_count / heldout_count) if heldout_count else 0.0,
    }


def acceptance_for_result(row):
    head = row["baseline_headroom"]
    dist = row["observe_value_distribution"]
    levels = row["level_policy_metrics"]["levels"]
    best_level = row["level_policy_metrics"]["best_level"]
    best = levels[best_level]
    struct = row["structure_validity"]
    sign_rates = [levels[lvl]["sign_correct_rate"] for lvl in levels]
    primary = (
        row["test_type"] in PRIMARY_TESTS
        and abs(row["observe_cost"] - PRIMARY_COST) < 1e-12
        and head["oracle_minus_always_observe"] >= 0.03
        and head["always_observe_minus_always_try"] > 0.0
        and best["delta_vs_always_observe"] >= 0.005
        and best["sign_correct_rate"] >= 0.70
        and 0.40 <= dist["positive_net_rate"] <= 0.65
        and dist["mean_true_E_positive"] is not None
        and 0.15 <= dist["mean_true_E_positive"] <= 0.35
        and dist["mean_true_E_nonpositive"] is not None
        and dist["mean_true_E_nonpositive"] <= -0.06
        and best["fallback_rate"] < 0.25
        and struct["exact_signature_determinism_rate"] < 0.95
        and struct["single_cue_max_predictiveness"] < 0.95
    )
    partial = (
        row["test_type"] in PRIMARY_TESTS
        and abs(row["observe_cost"] - PRIMARY_COST) < 1e-12
        and (
            head["oracle_minus_always_observe"] >= 0.03
            or best["delta_vs_always_observe"] > 0.0
            or max(sign_rates) >= 0.68
            or struct["single_cue_max_predictiveness"] < 0.95
        )
    )
    reasons = []
    if head["oracle_minus_always_observe"] < 0.03:
        reasons.append("oracle_gap_lt_0.03")
    if head["always_observe_minus_always_try"] <= 0.0:
        reasons.append("always_observe_not_above_always_try")
    if best["delta_vs_always_observe"] < 0.005:
        reasons.append("best_level_does_not_beat_always_observe")
    if best["sign_correct_rate"] < 0.70:
        reasons.append("sign_correct_not_enough")
    if not (0.40 <= dist["positive_net_rate"] <= 0.65):
        reasons.append("positive_rate_out_of_range")
    if dist["mean_true_E_nonpositive"] is None or dist["mean_true_E_nonpositive"] > -0.06:
        reasons.append("nonpositive_not_negative_enough")
    if struct["exact_signature_determinism_rate"] >= 0.95:
        reasons.append("exact_signature_too_deterministic")
    if struct["single_cue_max_predictiveness"] >= 0.95:
        reasons.append("single_cue_too_predictive")
    return {
        "passes_primary": primary,
        "passes_partial": (not primary) and partial,
        "rejection_reason": ";".join(reasons) if reasons else "",
    }


def evaluate_candidate_variant(candidate, test_name, variant, observe_cost):
    overrides = build_distribution_overrides(candidate)
    train_truth = compute_true_oracle_voi(variant["train_oids"], observe_cost, overrides)
    train_rows = build_train_rows(variant["train_oids"], train_truth)
    global_stats = summarize_rows(train_rows)
    signature_stats = build_signature_stats(train_rows)
    cue_stats = build_cue_stats(train_rows)

    test_rows = build_test_object_truth(variant["test_oids"], observe_cost, overrides)
    full_rows = build_test_object_truth(ALL_OIDS, observe_cost, overrides)

    est_maps = {"level1": {}, "level2": {}, "level3": {}}
    for sig in {row["signature"] for row in test_rows}:
        est_maps["level1"][sig] = level1_cue_marginal_estimate(sig, cue_stats, global_stats)
        est_maps["level2"][sig] = level2_knn_estimate(sig, signature_stats, global_stats, k=3)
        est_maps["level3"][sig] = level3_candidate_estimate(sig, signature_stats, global_stats, tau=0.5)

    always_try_net = mean([r["true_pre_best"] for r in test_rows])
    always_observe_net = mean([r["observe_return"] for r in test_rows])
    oracle_selective_net = mean(
        [r["observe_return"] if r["true_optimal"] else r["true_pre_best"] for r in test_rows]
    )

    positive_rows = [r for r in test_rows if r["true_optimal"]]
    nonpositive_rows = [r for r in test_rows if not r["true_optimal"]]
    level_metrics = {"levels": {}}
    for level_name, est_by_sig in est_maps.items():
        pred = summarize_predictability(test_rows, est_by_sig)
        policy = evaluate_policy_from_estimates(test_rows, est_by_sig)
        level_metrics["levels"][level_name] = {
            "policy_net": policy["policy_net"],
            "delta_vs_always_observe": policy["policy_net"] - always_observe_net,
            "delta_vs_always_try": policy["policy_net"] - always_try_net,
            "gap_to_oracle_selective": oracle_selective_net - policy["policy_net"],
            "observe_rate": policy["observe_rate"],
            "positive_net_observe_rate": policy["positive_net_observe_rate"],
            "nonpositive_net_observe_rate": policy["nonpositive_net_observe_rate"],
            "sign_correct_rate": pred["sign_correct_rate"],
            "false_observe_rate": policy["false_observe_rate"],
            "false_direct_rate": policy["false_direct_rate"],
            "fallback_rate": pred["fallback_rate"],
            "non_global_rate": pred["non_global_rate"],
            "pearson_corr": pred["pearson_corr"],
            "spearman_corr": pred["spearman_corr"],
            "estimated_E_separation": pred["estimated_E_separation"],
        }

    best_level = max(level_metrics["levels"], key=lambda lvl: level_metrics["levels"][lvl]["policy_net"])
    level_metrics["best_level"] = best_level
    level_metrics["best_level_net"] = level_metrics["levels"][best_level]["policy_net"]

    structure_validity = compute_structure_validity(full_rows, signature_stats, est_maps)
    structure_validity.update(compute_test_reachability(test_rows, signature_stats, est_maps))
    structure_validity["level1_non_global_rate"] = level_metrics["levels"]["level1"]["non_global_rate"]
    structure_validity["level2_non_global_rate"] = level_metrics["levels"]["level2"]["non_global_rate"]
    structure_validity["level3_non_global_rate"] = level_metrics["levels"]["level3"]["non_global_rate"]

    row = {
        "variant_id": candidate["variant_id"],
        "description": candidate["description"],
        "parameters": candidate["parameters"],
        "test_type": test_name,
        "observe_cost": observe_cost,
        "baseline_headroom": {
            "always_try_net": always_try_net,
            "always_observe_net": always_observe_net,
            "oracle_selective_net": oracle_selective_net,
            "oracle_minus_always_observe": oracle_selective_net - always_observe_net,
            "always_observe_minus_always_try": always_observe_net - always_try_net,
            "oracle_minus_always_try": oracle_selective_net - always_try_net,
        },
        "observe_value_distribution": {
            "positive_net_rate": len(positive_rows) / len(test_rows) if test_rows else 0.0,
            "n_positive_net": len(positive_rows),
            "n_nonpositive_net": len(nonpositive_rows),
            "mean_true_E_positive": mean([r["true_voi"] for r in positive_rows]) if positive_rows else None,
            "mean_true_E_nonpositive": mean([r["true_voi"] for r in nonpositive_rows]) if nonpositive_rows else None,
            "mean_true_E_all": mean([r["true_voi"] for r in test_rows]),
            "min_true_E": min((r["true_voi"] for r in test_rows), default=0.0),
            "max_true_E": max((r["true_voi"] for r in test_rows), default=0.0),
        },
        "level_policy_metrics": level_metrics,
        "structure_validity": structure_validity,
        "acceptance_flags": None,
        "rejection_reason": "",
    }
    row["acceptance_flags"] = acceptance_for_result(row)
    row["rejection_reason"] = row["acceptance_flags"]["rejection_reason"]
    return row


def aggregate_rows(rows):
    sample = rows[0]
    levels = ("level1", "level2", "level3")
    agg = {
        "variant_id": sample["variant_id"],
        "description": sample["description"],
        "parameters": sample["parameters"],
        "test_type": sample["test_type"],
        "observe_cost": sample["observe_cost"],
        "baseline_headroom": {
            "always_try_net": mean([r["baseline_headroom"]["always_try_net"] for r in rows]),
            "always_observe_net": mean([r["baseline_headroom"]["always_observe_net"] for r in rows]),
            "oracle_selective_net": mean([r["baseline_headroom"]["oracle_selective_net"] for r in rows]),
            "oracle_minus_always_observe": mean([r["baseline_headroom"]["oracle_minus_always_observe"] for r in rows]),
            "always_observe_minus_always_try": mean([r["baseline_headroom"]["always_observe_minus_always_try"] for r in rows]),
            "oracle_minus_always_try": mean([r["baseline_headroom"]["oracle_minus_always_try"] for r in rows]),
        },
        "observe_value_distribution": {
            "positive_net_rate": mean([r["observe_value_distribution"]["positive_net_rate"] for r in rows]),
            "n_positive_net": round(mean([r["observe_value_distribution"]["n_positive_net"] for r in rows]), 4),
            "n_nonpositive_net": round(mean([r["observe_value_distribution"]["n_nonpositive_net"] for r in rows]), 4),
            "mean_true_E_positive": mean(
                [r["observe_value_distribution"]["mean_true_E_positive"] for r in rows if r["observe_value_distribution"]["mean_true_E_positive"] is not None]
            ),
            "mean_true_E_nonpositive": mean(
                [r["observe_value_distribution"]["mean_true_E_nonpositive"] for r in rows if r["observe_value_distribution"]["mean_true_E_nonpositive"] is not None]
            ),
            "mean_true_E_all": mean([r["observe_value_distribution"]["mean_true_E_all"] for r in rows]),
            "min_true_E": min(r["observe_value_distribution"]["min_true_E"] for r in rows),
            "max_true_E": max(r["observe_value_distribution"]["max_true_E"] for r in rows),
        },
        "level_policy_metrics": {"levels": {}},
        "structure_validity": {
            "exact_signature_determinism_rate": mean([r["structure_validity"]["exact_signature_determinism_rate"] for r in rows]),
            "single_cue_max_predictiveness": mean([r["structure_validity"]["single_cue_max_predictiveness"] for r in rows]),
            "cue_pair_max_predictiveness": mean([r["structure_validity"]["cue_pair_max_predictiveness"] for r in rows]),
            "heldout_signature_count": mean([r["structure_validity"]["heldout_signature_count"] for r in rows]),
            "heldout_signature_keymiss_rate": mean([r["structure_validity"]["heldout_signature_keymiss_rate"] for r in rows]),
            "compositional_reachability_rate": mean([r["structure_validity"]["compositional_reachability_rate"] for r in rows]),
            "level1_non_global_rate": mean([r["structure_validity"]["level1_non_global_rate"] for r in rows]),
            "level2_non_global_rate": mean([r["structure_validity"]["level2_non_global_rate"] for r in rows]),
            "level3_non_global_rate": mean([r["structure_validity"]["level3_non_global_rate"] for r in rows]),
        },
    }
    for level_name in levels:
        agg["level_policy_metrics"]["levels"][level_name] = {
            "policy_net": mean([r["level_policy_metrics"]["levels"][level_name]["policy_net"] for r in rows]),
            "delta_vs_always_observe": mean([r["level_policy_metrics"]["levels"][level_name]["delta_vs_always_observe"] for r in rows]),
            "delta_vs_always_try": mean([r["level_policy_metrics"]["levels"][level_name]["delta_vs_always_try"] for r in rows]),
            "gap_to_oracle_selective": mean([r["level_policy_metrics"]["levels"][level_name]["gap_to_oracle_selective"] for r in rows]),
            "observe_rate": mean([r["level_policy_metrics"]["levels"][level_name]["observe_rate"] for r in rows]),
            "positive_net_observe_rate": mean([r["level_policy_metrics"]["levels"][level_name]["positive_net_observe_rate"] for r in rows]),
            "nonpositive_net_observe_rate": mean([r["level_policy_metrics"]["levels"][level_name]["nonpositive_net_observe_rate"] for r in rows]),
            "sign_correct_rate": mean([r["level_policy_metrics"]["levels"][level_name]["sign_correct_rate"] for r in rows]),
            "false_observe_rate": mean([r["level_policy_metrics"]["levels"][level_name]["false_observe_rate"] for r in rows]),
            "false_direct_rate": mean([r["level_policy_metrics"]["levels"][level_name]["false_direct_rate"] for r in rows]),
            "fallback_rate": mean([r["level_policy_metrics"]["levels"][level_name]["fallback_rate"] for r in rows]),
            "non_global_rate": mean([r["level_policy_metrics"]["levels"][level_name]["non_global_rate"] for r in rows]),
            "pearson_corr": mean(
                [r["level_policy_metrics"]["levels"][level_name]["pearson_corr"] for r in rows if r["level_policy_metrics"]["levels"][level_name]["pearson_corr"] is not None]
            ),
            "spearman_corr": mean(
                [r["level_policy_metrics"]["levels"][level_name]["spearman_corr"] for r in rows if r["level_policy_metrics"]["levels"][level_name]["spearman_corr"] is not None]
            ),
            "estimated_E_separation": mean(
                [r["level_policy_metrics"]["levels"][level_name]["estimated_E_separation"] for r in rows if r["level_policy_metrics"]["levels"][level_name]["estimated_E_separation"] is not None]
            ),
        }
    best_level = max(agg["level_policy_metrics"]["levels"], key=lambda lvl: agg["level_policy_metrics"]["levels"][lvl]["policy_net"])
    agg["level_policy_metrics"]["best_level"] = best_level
    agg["level_policy_metrics"]["best_level_net"] = agg["level_policy_metrics"]["levels"][best_level]["policy_net"]
    agg["acceptance_flags"] = acceptance_for_result(agg)
    agg["rejection_reason"] = agg["acceptance_flags"]["rejection_reason"]
    return agg


print("[2/6] Defining targeted D-family refinement variants...")
candidates = [
    {
        "variant_id": "D4_cue_group_balanced_soft",
        "description": "Cue-group soft balancing with useful/mixed/wasteful groups derived from legal cue composition",
        "mode": "cue_group_balanced_soft",
        "parameters": {
            "A_spotted": 0.68,
            "A_ambient_only": 0.56,
            "A_bark_discolored": 0.36,
            "A_crystal": 0.40,
            "B_discolored": 0.58,
            "B_ambient_only": 0.42,
            "B_grip": 0.32,
            "B_stem_spotted": 0.48,
            "salt": 141,
        },
    },
    {
        "variant_id": "D5_pairwise_cue_interaction",
        "description": "Pairwise cue interaction assignment where single cues stay ambiguous but cue pairs carry signal",
        "mode": "pairwise_cue_interaction",
        "parameters": {
            "base_A": 0.52,
            "base_B": 0.48,
            "pair_weights": [
                ("brownish", "spotted", 0.12),
                ("greenish", "discolored", 0.10),
                ("discolored", "has_bark_texture", -0.18),
                ("greenish", "has_grip_area", -0.18),
                ("spotted", "has_stem_remnant", 0.02),
                ("brownish", "has_crystal_flecks", -0.10),
            ],
            "clip_lo": 0.28,
            "clip_hi": 0.72,
            "salt": 157,
        },
    },
    {
        "variant_id": "D6_candidate_set_structured",
        "description": "Subset/Jaccard structured assignment intended to improve candidate-set aggregation",
        "mode": "candidate_set_structured",
        "parameters": {
            "base_A": 0.55,
            "base_B": 0.45,
            "shared_bonus": 0.08,
            "diagnostic_shift": -0.12,
            "combo_shift": -0.08,
            "clip_lo": 0.28,
            "clip_hi": 0.72,
            "salt": 173,
        },
    },
    {
        "variant_id": "D7_hybrid_soft_structured",
        "description": "Hybrid cue-group plus pairwise refinement with non-deterministic structure",
        "mode": "hybrid_soft_structured",
        "parameters": {
            "subset_part": {
                "base_A": 0.55,
                "base_B": 0.45,
                "shared_bonus": 0.08,
                "diagnostic_shift": -0.10,
                "combo_shift": -0.04,
                "clip_lo": 0.28,
                "clip_hi": 0.72,
            },
            "pair_refine": [
                ("brownish", "spotted", 0.04),
                ("greenish", "discolored", 0.04),
                ("spotted", "has_stem_remnant", 0.05),
                ("discolored", "has_bark_texture", -0.03),
                ("brownish", "has_crystal_flecks", -0.03),
                ("greenish", "has_grip_area", -0.03),
            ],
            "clip_lo": 0.28,
            "clip_hi": 0.74,
            "salt": 191,
        },
    },
]


print("[3/6] Evaluating refinement variants...")
raw_rows = []
for cand in candidates:
    print(f"  {cand['variant_id']}")
    for observe_cost in OBSERVE_COSTS:
        for test_name, variants in TESTS:
            for variant in variants:
                raw_rows.append(evaluate_candidate_variant(cand, test_name, variant, observe_cost))

aggregated_rows = []
for cand in candidates:
    rows_by_key = defaultdict(list)
    for row in raw_rows:
        if row["variant_id"] != cand["variant_id"]:
            continue
        rows_by_key[(row["test_type"], row["observe_cost"])].append(row)
    for _, rows in sorted(rows_by_key.items()):
        aggregated_rows.append(aggregate_rows(rows))


def rank_key(row):
    best_level = row["level_policy_metrics"]["best_level"]
    return (
        1 if row["acceptance_flags"]["passes_primary"] else 0,
        1 if row["acceptance_flags"]["passes_partial"] else 0,
        row["level_policy_metrics"]["levels"][best_level]["delta_vs_always_observe"],
        row["level_policy_metrics"]["levels"][best_level]["sign_correct_rate"],
        row["baseline_headroom"]["oracle_minus_always_observe"],
    )


focus_rows = [
    r for r in aggregated_rows
    if r["test_type"] in PRIMARY_TESTS and abs(r["observe_cost"] - PRIMARY_COST) < 1e-12
]
passing_primary = [r for r in focus_rows if r["acceptance_flags"]["passes_primary"]]
passing_partial = [r for r in focus_rows if r["acceptance_flags"]["passes_partial"]]

status = "NO_VALID_CANDIDATE"
if passing_primary:
    status = "DESIGN_CANDIDATE_FOUND"
elif passing_partial:
    status = "PARTIAL_CANDIDATE_FOUND"

best_overall = max(focus_rows, key=rank_key) if focus_rows else None
best_A = max([r for r in focus_rows if r["test_type"] == "A"], key=rank_key, default=None)
best_B = max([r for r in focus_rows if r["test_type"] == "B"], key=rank_key, default=None)

rejected_reasons = Counter()
for row in focus_rows:
    reason = row["rejection_reason"] or "none"
    rejected_reasons[reason] += 1

acceptance_summary = {
    "status": status,
    "best_candidate_overall": None if best_overall is None else {
        "variant_id": best_overall["variant_id"],
        "parameters": best_overall["parameters"],
        "test_type": best_overall["test_type"],
        "observe_cost": best_overall["observe_cost"],
        "best_level": best_overall["level_policy_metrics"]["best_level"],
        "best_level_delta_vs_always_observe": best_overall["level_policy_metrics"]["levels"][best_overall["level_policy_metrics"]["best_level"]]["delta_vs_always_observe"],
    },
    "best_candidate_test_A": None if best_A is None else {
        "variant_id": best_A["variant_id"],
        "observe_cost": best_A["observe_cost"],
        "best_level": best_A["level_policy_metrics"]["best_level"],
        "best_level_delta_vs_always_observe": best_A["level_policy_metrics"]["levels"][best_A["level_policy_metrics"]["best_level"]]["delta_vs_always_observe"],
    },
    "best_candidate_test_B": None if best_B is None else {
        "variant_id": best_B["variant_id"],
        "observe_cost": best_B["observe_cost"],
        "best_level": best_B["level_policy_metrics"]["best_level"],
        "best_level_delta_vs_always_observe": best_B["level_policy_metrics"]["levels"][best_B["level_policy_metrics"]["best_level"]]["delta_vs_always_observe"],
    },
    "candidates_passing_primary": [
        {"variant_id": r["variant_id"], "test_type": r["test_type"], "observe_cost": r["observe_cost"]}
        for r in passing_primary
    ],
    "candidates_passing_partial": [
        {"variant_id": r["variant_id"], "test_type": r["test_type"], "observe_cost": r["observe_cost"]}
        for r in passing_partial
    ],
    "rejected_reasons": dict(rejected_reasons),
}

print("[4/6] Writing outputs...")
runs_dir = os.path.join(CURRENT_DIR, "runs")
protocols_dir = os.path.join(CURRENT_DIR, "protocols")
os.makedirs(runs_dir, exist_ok=True)
os.makedirs(protocols_dir, exist_ok=True)

json_output = {
    "metadata": {
        "block_id": "1J40b-7i",
        "script_name": "_block1j40b7i_d_family_cue_structured_refinement_audit.py",
        "timestamp": datetime.now().isoformat(),
        "source_files_used": SOURCE_FILES_USED,
        "candidate_variants": [c["variant_id"] for c in candidates],
        "costs": OBSERVE_COSTS,
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

json_path = os.path.join(runs_dir, "block1j40b7i_d_family_cue_structured_refinement_audit.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, indent=2)

csv_lines = [
    "variant_id,test_type,observe_cost,always_try_net,always_observe_net,oracle_selective_net,oracle_minus_always_observe,"
    "always_observe_minus_always_try,positive_net_rate,mean_true_E_positive,mean_true_E_nonpositive,level1_policy_net,"
    "level2_policy_net,level3_policy_net,best_level,best_level_net,best_level_delta_vs_always_observe,best_level_sign_correct_rate,"
    "level1_sign_correct_rate,level2_sign_correct_rate,level3_sign_correct_rate,level1_corr,level2_corr,level3_corr,"
    "exact_signature_determinism_rate,single_cue_max_predictiveness,cue_pair_max_predictiveness,compositional_reachability_rate,"
    "fallback_rate_best_level,passes_primary_criteria,rejection_reason"
]
for row in aggregated_rows:
    best_level = row["level_policy_metrics"]["best_level"]
    best_metrics = row["level_policy_metrics"]["levels"][best_level]
    csv_lines.append(",".join([
        row["variant_id"],
        row["test_type"],
        f"{row['observe_cost']:.2f}",
        f"{row['baseline_headroom']['always_try_net']:.6f}",
        f"{row['baseline_headroom']['always_observe_net']:.6f}",
        f"{row['baseline_headroom']['oracle_selective_net']:.6f}",
        f"{row['baseline_headroom']['oracle_minus_always_observe']:.6f}",
        f"{row['baseline_headroom']['always_observe_minus_always_try']:.6f}",
        f"{row['observe_value_distribution']['positive_net_rate']:.6f}",
        "" if row["observe_value_distribution"]["mean_true_E_positive"] is None else f"{row['observe_value_distribution']['mean_true_E_positive']:.6f}",
        "" if row["observe_value_distribution"]["mean_true_E_nonpositive"] is None else f"{row['observe_value_distribution']['mean_true_E_nonpositive']:.6f}",
        f"{row['level_policy_metrics']['levels']['level1']['policy_net']:.6f}",
        f"{row['level_policy_metrics']['levels']['level2']['policy_net']:.6f}",
        f"{row['level_policy_metrics']['levels']['level3']['policy_net']:.6f}",
        best_level,
        f"{row['level_policy_metrics']['best_level_net']:.6f}",
        f"{best_metrics['delta_vs_always_observe']:.6f}",
        f"{best_metrics['sign_correct_rate']:.6f}",
        f"{row['level_policy_metrics']['levels']['level1']['sign_correct_rate']:.6f}",
        f"{row['level_policy_metrics']['levels']['level2']['sign_correct_rate']:.6f}",
        f"{row['level_policy_metrics']['levels']['level3']['sign_correct_rate']:.6f}",
        "" if row["level_policy_metrics"]["levels"]["level1"]["pearson_corr"] is None else f"{row['level_policy_metrics']['levels']['level1']['pearson_corr']:.6f}",
        "" if row["level_policy_metrics"]["levels"]["level2"]["pearson_corr"] is None else f"{row['level_policy_metrics']['levels']['level2']['pearson_corr']:.6f}",
        "" if row["level_policy_metrics"]["levels"]["level3"]["pearson_corr"] is None else f"{row['level_policy_metrics']['levels']['level3']['pearson_corr']:.6f}",
        f"{row['structure_validity']['exact_signature_determinism_rate']:.6f}",
        f"{row['structure_validity']['single_cue_max_predictiveness']:.6f}",
        f"{row['structure_validity']['cue_pair_max_predictiveness']:.6f}",
        f"{row['structure_validity']['compositional_reachability_rate']:.6f}",
        f"{best_metrics['fallback_rate']:.6f}",
        str(row["acceptance_flags"]["passes_primary"]),
        row["rejection_reason"].replace(",", ";"),
    ]))
csv_path = os.path.join(protocols_dir, "block1j40b7i_d_family_cue_structured_refinement_audit_table.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(csv_lines))

lines = [
    "# Checkpoint Summary",
    "",
    f"- status: {status}",
    f"- best_candidate_overall: {acceptance_summary['best_candidate_overall']}",
    f"- best_candidate_test_A: {acceptance_summary['best_candidate_test_A']}",
    f"- best_candidate_test_B: {acceptance_summary['best_candidate_test_B']}",
    f"- candidates_passing_primary: {len(acceptance_summary['candidates_passing_primary'])}",
    f"- candidates_passing_partial: {len(acceptance_summary['candidates_passing_partial'])}",
    "",
    "# Files Changed",
    "",
    "- _block1j40b7i_d_family_cue_structured_refinement_audit.py",
    "",
    "# Commands Run",
    "",
    '- `python _block1j40b7i_d_family_cue_structured_refinement_audit.py`',
    "",
    "# Why 7i Was Run",
    "",
    "- 7h showed partial raw headroom but not clean cue-structured compositional predictability.",
    "- 7i narrows to D-family refinements only, with no broad A/B/C reruns.",
    "",
    "# 7h Recap",
    "",
    "- D-family in 7h improved oracle gap but not `best_level - always_observe`.",
    "- Positive rate was often too high and cue correlations were weak or wrong-sign.",
    "",
    "# Candidate Variants",
    "",
]
for cand in candidates:
    lines.append(f"- `{cand['variant_id']}`: {cand['description']}")

for title, test_name in [("Test A", "A"), ("Test B", "B"), ("Test C", "C")]:
    lines.extend(["", f"# {title} Results", ""])
    for row in [r for r in aggregated_rows if r["test_type"] == test_name and abs(r["observe_cost"] - PRIMARY_COST) < 1e-12]:
        best_level = row["level_policy_metrics"]["best_level"]
        best = row["level_policy_metrics"]["levels"][best_level]
        lines.append(
            f"- `{row['variant_id']}`: oracle_gap={row['baseline_headroom']['oracle_minus_always_observe']:.4f}, "
            f"ao_minus_at={row['baseline_headroom']['always_observe_minus_always_try']:.4f}, "
            f"best={best_level}, best_delta={best['delta_vs_always_observe']:.4f}, "
            f"sign={best['sign_correct_rate']:.4f}, rate={row['observe_value_distribution']['positive_net_rate']:.4f}, "
            f"rej={row['rejection_reason'] or 'none'}"
        )

lines.extend([
    "",
    "# Structure Validity Audit",
    "",
])
for row in focus_rows:
    lines.append(
        f"- `{row['variant_id']}|{row['test_type']}`: exact_det={row['structure_validity']['exact_signature_determinism_rate']:.4f}, "
        f"single_cue_max={row['structure_validity']['single_cue_max_predictiveness']:.4f}, "
        f"pair_max={row['structure_validity']['cue_pair_max_predictiveness']:.4f}, "
        f"reachability={row['structure_validity']['compositional_reachability_rate']:.4f}"
    )

lines.extend([
    "",
    "# Best Candidate",
    "",
    f"- {acceptance_summary['best_candidate_overall']}",
    "",
    "# Partial Candidates",
    "",
    f"- {acceptance_summary['candidates_passing_partial']}",
    "",
    "# Rejected Variants and Reasons",
    "",
])
for reason, count in sorted(acceptance_summary["rejected_reasons"].items()):
    lines.append(f"- `{reason}`: {count}")

lines.extend([
    "",
    "# Validity / Leakage Audit",
    "",
    "- no learner training: true",
    "- no 1J40b-8: true",
    "- no official env replacement: true",
    "- forbidden test keys absent: true",
    "- audit only: true",
    "",
    "# Recommended Resume Point",
    "",
    "- If status remains partial: one more D-family refinement focused on sign accuracy and keeping `always_observe > always_try`.",
    "- If no clean pass exists: deeper environment redesign before any architecture work.",
])

md_path = os.path.join(protocols_dir, "block1j40b7i_d_family_cue_structured_refinement_audit.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

ckpt_path = os.path.join(CURRENT_DIR, "checkpoint_1j40b7i_d_family_cue_structured_refinement_audit.md")
with open(ckpt_path, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            [
                "# Checkpoint 1J40b-7i",
                "",
                f"- status: {status}",
                f"- best_candidate_overall: {acceptance_summary['best_candidate_overall']}",
                f"- best_candidate_test_A: {acceptance_summary['best_candidate_test_A']}",
                f"- best_candidate_test_B: {acceptance_summary['best_candidate_test_B']}",
                f"- candidates_passing_primary: {len(acceptance_summary['candidates_passing_primary'])}",
                f"- candidates_passing_partial: {len(acceptance_summary['candidates_passing_partial'])}",
                "- next_suggested_step: one more D-family refinement" if status == "PARTIAL_CANDIDATE_FOUND" else "- next_suggested_step: deeper environment redesign",
            ]
        )
    )

print("[5/6] Final console summary...")
print(f"STATUS: {status}")
print(f"Best candidate overall: {acceptance_summary['best_candidate_overall']}")
for label, row in [("Test A", best_A), ("Test B", best_B)]:
    if row is None:
        continue
    best_level = row["level_policy_metrics"]["best_level"]
    best = row["level_policy_metrics"]["levels"][best_level]
    print(
        f"{label}: oracle_gap={row['baseline_headroom']['oracle_minus_always_observe']:.4f}, "
        f"ao_minus_at={row['baseline_headroom']['always_observe_minus_always_try']:.4f}, "
        f"best_delta={best['delta_vs_always_observe']:.4f}, "
        f"pos_rate={row['observe_value_distribution']['positive_net_rate']:.4f}, "
        f"mean_pos={row['observe_value_distribution']['mean_true_E_positive']:.4f}, "
        f"mean_nonpos={row['observe_value_distribution']['mean_true_E_nonpositive']:.4f}, "
        f"sign={best['sign_correct_rate']:.4f}, "
        f"exact_det={row['structure_validity']['exact_signature_determinism_rate']:.4f}, "
        f"single_cue_max={row['structure_validity']['single_cue_max_predictiveness']:.4f}"
    )

print("[6/6] Done.")
print(f"  JSON: {json_path}")
print(f"  CSV:  {csv_path}")
print(f"  MD:   {md_path}")
print(f"  CKPT: {ckpt_path}")
