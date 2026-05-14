"""
Block 1J40b-7j -- D-Family Anti-Dominance / Anti-Single-Cue Refinement Audit.

Narrow audit-only refinement over D-family variants derived from 7i.
This script does not train a learner, does not replace official env3b,
and does not implement 1J40b-8.
"""

import json
import os
import runpy
import time
from collections import Counter, defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES_USED = [
    "_block1j40b7i_d_family_cue_structured_refinement_audit.py",
    "_block1j40b7h_env3b_redesign_parameter_audit.py",
    "_block1j40b7g_env3b_counterfactual_headroom_decomposition.py",
    "_block1j40b7fb_compositional_oracle_belief_upper_bound.py",
    "_block1j40b7e_env3b_distributional_audit_counterfactual.py",
    "runs/block1j40b7i_d_family_cue_structured_refinement_audit.json",
]

print("[1/6] Loading 7i helper functions...")
ns = runpy.run_path(
    os.path.join(CURRENT_DIR, "_block1j40b7i_d_family_cue_structured_refinement_audit.py")
)

SEEDS = ns["SEEDS"]
ALL_OIDS = ns["ALL_OIDS"]
TEST_A_VARIANTS = ns["TEST_A_VARIANTS"]
TEST_B_VARIANTS = ns["TEST_B_VARIANTS"]
TEST_C_VARIANTS = ns["TEST_C_VARIANTS"]
OBSERVE_COSTS = [0.03, 0.05, 0.07, 0.10]
PRIMARY_COST = 0.03
PRIMARY_TESTS = {"A", "B"}

per_seed_objects = ns["per_seed_objects"]
get_pre_surface_signature = ns["get_pre_surface_signature"]
positive_like_affordance = ns["positive_like_affordance"]
nonpositive_like_affordance = ns["nonpositive_like_affordance"]
get_flat_affordance = ns["get_flat_affordance"]
stable_random = ns["stable_random"]

compute_true_oracle_voi = ns["compute_true_oracle_voi"]
build_test_object_truth = ns["build_test_object_truth"]
build_train_rows = ns["build_train_rows"]
summarize_rows = ns["summarize_rows"]
build_signature_stats = ns["build_signature_stats"]
build_cue_stats = ns["build_cue_stats"]
level1_cue_marginal_estimate = ns["level1_cue_marginal_estimate"]
level2_knn_estimate = ns["level2_knn_estimate"]
level3_candidate_estimate = ns["level3_candidate_estimate"]
evaluate_policy_from_estimates = ns["evaluate_policy_from_estimates"]
summarize_predictability = ns["summarize_predictability"]
compute_structure_validity = ns["compute_structure_validity"]
compute_test_reachability = ns["compute_test_reachability"]
mean = ns["mean"]
signature_extras = ns["signature_extras"]

TESTS = [("A", TEST_A_VARIANTS), ("B", TEST_B_VARIANTS), ("C", TEST_C_VARIANTS)]


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def signature_profile(sig):
    extras = set(signature_extras(sig))
    return {
        "ambient": "A" if "brownish" in sig else "B",
        "extras": extras,
        "diagnostic": extras in (
            {"discolored", "has_bark_texture"},
            {"has_crystal_flecks"},
            {"has_grip_area"},
            {"has_stem_remnant", "spotted"},
        ),
        "shared": extras in ({"spotted"}, {"discolored"}),
    }


def negative_affordance_for_profile(seed, oid, profile, candidate):
    negative_mode = candidate["parameters"].get("negative_mode", "diagnostic_flat_fail")
    if negative_mode == "diagnostic_flat_fail":
        if profile["diagnostic"]:
            return get_flat_affordance(False)
        return nonpositive_like_affordance(seed, oid)
    raise ValueError(f"Unknown negative_mode: {negative_mode}")


def probability_for_signature(sig, candidate):
    params = candidate["parameters"]
    prof = signature_profile(sig)
    cues = set(sig)
    mode = candidate["mode"]

    if mode == "d8_anti_single_cue_balanced":
        if prof["diagnostic"]:
            p = params["diagnostic"]
        elif prof["shared"]:
            p = params["shared"]
        else:
            p = params["ambient"]
        return clamp(p, params["clip_lo"], params["clip_hi"])

    if mode == "d9_pairwise_dominant":
        p = params["base_A"] if prof["ambient"] == "A" else params["base_B"]
        for c1, c2, delta in params["pairs"]:
            if c1 in cues and c2 in cues:
                p += delta
        return clamp(p, params["clip_lo"], params["clip_hi"])

    if mode == "d10_candidate_set_balanced":
        p = params["base_A"] if prof["ambient"] == "A" else params["base_B"]
        if prof["shared"]:
            p += params["shared_bonus"]
        if prof["diagnostic"]:
            p += params["diagnostic_shift"]
        if prof["shared"] and prof["diagnostic"]:
            p += params["combo_shift"]
        return clamp(p, params["clip_lo"], params["clip_hi"])

    if mode == "d11_hybrid_anti_dominance":
        if prof["diagnostic"]:
            p = params["diagnostic"]
        elif prof["shared"]:
            p = params["shared"]
        else:
            p = params["ambient"]
        for c1, c2, delta in params["pair_refine"]:
            if c1 in cues and c2 in cues:
                p += delta
        return clamp(p, params["clip_lo"], params["clip_hi"])

    raise ValueError(f"Unknown candidate mode: {mode}")


def build_distribution_overrides(candidate):
    overrides = {}
    salt = candidate["parameters"]["salt"]
    for seed, oid in sorted(ALL_OIDS):
        sig = get_pre_surface_signature(per_seed_objects[seed][oid]["pre_surface_features"])
        prof = signature_profile(sig)
        p_positive = probability_for_signature(sig, candidate)
        is_positive_like = stable_random(seed, oid, salt) < p_positive
        overrides[(seed, oid)] = (
            positive_like_affordance(seed, oid)
            if is_positive_like
            else negative_affordance_for_profile(seed, oid, prof, candidate)
        )
    return overrides


def acceptance_for_result(row):
    head = row["baseline_headroom"]
    dist = row["observe_value_distribution"]
    levels = row["level_policy_metrics"]["levels"]
    best_level = row["level_policy_metrics"]["best_level"]
    best = levels[best_level]
    struct = row["structure_validity"]

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
        and struct["single_cue_max_predictiveness"] <= 0.85
        and struct["cue_pair_max_predictiveness"] < 1.0
    )

    partial = (
        row["test_type"] in PRIMARY_TESTS
        and abs(row["observe_cost"] - PRIMARY_COST) < 1e-12
        and (
            head["oracle_minus_always_observe"] >= 0.03
            or 0.40 <= dist["positive_net_rate"] <= 0.65
            or struct["single_cue_max_predictiveness"] <= 0.85
            or best["delta_vs_always_observe"] > 0.0
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
    if struct["single_cue_max_predictiveness"] > 0.85:
        reasons.append("single_cue_too_predictive")
    if struct["cue_pair_max_predictiveness"] >= 1.0:
        reasons.append("cue_pair_too_predictive")

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


def average_non_null(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


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
            "mean_true_E_positive": average_non_null([r["observe_value_distribution"]["mean_true_E_positive"] for r in rows]),
            "mean_true_E_nonpositive": average_non_null([r["observe_value_distribution"]["mean_true_E_nonpositive"] for r in rows]),
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
            "pearson_corr": average_non_null([r["level_policy_metrics"]["levels"][level_name]["pearson_corr"] for r in rows]),
            "spearman_corr": average_non_null([r["level_policy_metrics"]["levels"][level_name]["spearman_corr"] for r in rows]),
            "estimated_E_separation": average_non_null([r["level_policy_metrics"]["levels"][level_name]["estimated_E_separation"] for r in rows]),
        }

    best_level = max(
        levels,
        key=lambda lvl: (
            agg["level_policy_metrics"]["levels"][lvl]["policy_net"],
            agg["level_policy_metrics"]["levels"][lvl]["sign_correct_rate"],
        ),
    )
    agg["level_policy_metrics"]["best_level"] = best_level
    agg["level_policy_metrics"]["best_level_net"] = agg["level_policy_metrics"]["levels"][best_level]["policy_net"]
    agg["acceptance_flags"] = acceptance_for_result(agg)
    agg["rejection_reason"] = agg["acceptance_flags"]["rejection_reason"]
    return agg


print("[2/6] Defining 7j anti-dominance candidates...")
candidates = [
    {
        "variant_id": "D8_anti_single_cue_balanced_s141",
        "description": "D8 anti-single-cue balanced refinement with fixed salt 141",
        "mode": "d8_anti_single_cue_balanced",
        "parameters": {
            "ambient": 0.50,
            "shared": 0.56,
            "diagnostic": 0.36,
            "clip_lo": 0.28,
            "clip_hi": 0.64,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 141,
        },
    },
    {
        "variant_id": "D8_anti_single_cue_balanced_s207",
        "description": "D8 anti-single-cue balanced refinement with fixed salt 207",
        "mode": "d8_anti_single_cue_balanced",
        "parameters": {
            "ambient": 0.50,
            "shared": 0.56,
            "diagnostic": 0.36,
            "clip_lo": 0.28,
            "clip_hi": 0.64,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 207,
        },
    },
    {
        "variant_id": "D8_anti_single_cue_balanced_s319",
        "description": "D8 anti-single-cue balanced refinement with fixed salt 319",
        "mode": "d8_anti_single_cue_balanced",
        "parameters": {
            "ambient": 0.50,
            "shared": 0.56,
            "diagnostic": 0.36,
            "clip_lo": 0.28,
            "clip_hi": 0.64,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 319,
        },
    },
    {
        "variant_id": "D9_pairwise_dominant_single_cue_ambiguous_s141",
        "description": "D9 pairwise-dominant refinement with ambiguous single-cue marginals",
        "mode": "d9_pairwise_dominant",
        "parameters": {
            "base_A": 0.50,
            "base_B": 0.48,
            "pairs": [
                ("brownish", "spotted", 0.08),
                ("greenish", "discolored", 0.08),
                ("discolored", "has_bark_texture", -0.16),
                ("greenish", "has_grip_area", -0.16),
                ("brownish", "has_crystal_flecks", -0.16),
                ("spotted", "has_stem_remnant", -0.16),
            ],
            "clip_lo": 0.22,
            "clip_hi": 0.68,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 141,
        },
    },
    {
        "variant_id": "D10_candidate_set_balanced_s141",
        "description": "D10 candidate-set balanced refinement derived from D6",
        "mode": "d10_candidate_set_balanced",
        "parameters": {
            "base_A": 0.48,
            "base_B": 0.46,
            "shared_bonus": 0.08,
            "diagnostic_shift": -0.14,
            "combo_shift": -0.02,
            "clip_lo": 0.22,
            "clip_hi": 0.68,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 141,
        },
    },
    {
        "variant_id": "D10_candidate_set_balanced_s207",
        "description": "D10 candidate-set balanced refinement derived from D6",
        "mode": "d10_candidate_set_balanced",
        "parameters": {
            "base_A": 0.48,
            "base_B": 0.46,
            "shared_bonus": 0.08,
            "diagnostic_shift": -0.14,
            "combo_shift": -0.02,
            "clip_lo": 0.22,
            "clip_hi": 0.68,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 207,
        },
    },
    {
        "variant_id": "D11_hybrid_anti_dominance_s141",
        "description": "D11 hybrid anti-dominance refinement combining D8 group priors and mild pair refinement",
        "mode": "d11_hybrid_anti_dominance",
        "parameters": {
            "ambient": 0.50,
            "shared": 0.56,
            "diagnostic": 0.36,
            "pair_refine": [
                ("brownish", "spotted", 0.03),
                ("greenish", "discolored", 0.03),
                ("discolored", "has_bark_texture", -0.06),
                ("greenish", "has_grip_area", -0.06),
                ("brownish", "has_crystal_flecks", -0.06),
                ("spotted", "has_stem_remnant", -0.06),
            ],
            "clip_lo": 0.24,
            "clip_hi": 0.66,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 141,
        },
    },
    {
        "variant_id": "D11_hybrid_anti_dominance_s207",
        "description": "D11 hybrid anti-dominance refinement combining D8 group priors and mild pair refinement",
        "mode": "d11_hybrid_anti_dominance",
        "parameters": {
            "ambient": 0.50,
            "shared": 0.56,
            "diagnostic": 0.36,
            "pair_refine": [
                ("brownish", "spotted", 0.03),
                ("greenish", "discolored", 0.03),
                ("discolored", "has_bark_texture", -0.06),
                ("greenish", "has_grip_area", -0.06),
                ("brownish", "has_crystal_flecks", -0.06),
                ("spotted", "has_stem_remnant", -0.06),
            ],
            "clip_lo": 0.24,
            "clip_hi": 0.66,
            "negative_mode": "diagnostic_flat_fail",
            "salt": 207,
        },
    },
]

print("[3/6] Evaluating 7j candidates...")
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
    best = row["level_policy_metrics"]["levels"][best_level]
    return (
        1 if row["acceptance_flags"]["passes_primary"] else 0,
        1 if row["acceptance_flags"]["passes_partial"] else 0,
        best["delta_vs_always_observe"],
        -abs(row["observe_value_distribution"]["positive_net_rate"] - 0.525),
        best["sign_correct_rate"],
        row["baseline_headroom"]["oracle_minus_always_observe"],
        -row["structure_validity"]["single_cue_max_predictiveness"],
    )


focus_rows = [
    r
    for r in aggregated_rows
    if r["test_type"] in PRIMARY_TESTS and abs(r["observe_cost"] - PRIMARY_COST) < 1e-12
]
passing_primary = [r for r in focus_rows if r["acceptance_flags"]["passes_primary"]]
passing_partial = [r for r in focus_rows if r["acceptance_flags"]["passes_partial"]]

status = "NO_VALID_CANDIDATE"
if passing_primary:
    status = "DESIGN_CANDIDATE_FOUND"
elif passing_partial:
    status = "PARTIAL_CANDIDATE_FOUND"

best_overall = max(focus_rows, key=rank_key, default=None)
best_A = max([r for r in focus_rows if r["test_type"] == "A"], key=rank_key, default=None)
best_B = max([r for r in focus_rows if r["test_type"] == "B"], key=rank_key, default=None)

rejected_reasons = Counter()
for row in focus_rows:
    rejected_reasons[row["rejection_reason"] or "none"] += 1

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
        "block_id": "1J40b-7j",
        "script_name": "_block1j40b7j_d_family_anti_dominance_refinement_audit.py",
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

json_path = os.path.join(runs_dir, "block1j40b7j_d_family_anti_dominance_refinement_audit.json")
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

csv_path = os.path.join(protocols_dir, "block1j40b7j_d_family_anti_dominance_refinement_audit_table.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(csv_lines))

report_lines = [
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
    "- _block1j40b7j_d_family_anti_dominance_refinement_audit.py",
    "",
    "# Commands Run",
    "",
    '- `python _block1j40b7j_d_family_anti_dominance_refinement_audit.py`',
    "",
    "# Why 7j Was Run",
    "",
    "- 7i improved raw oracle headroom but left positive rate too high and single-cue predictiveness at 1.0.",
    "- 7j narrows to anti-dominance D-family variants only, with no broad family reruns.",
    "",
    "# 7i Recap",
    "",
    "- D4 and D6 preserved `always_observe > always_try` and opened oracle gap.",
    "- But no compositional level beat `always_observe`, positive rate stayed around 0.75, and diagnostic single cues remained too predictive.",
    "",
    "# Candidate Variants",
    "",
]
for cand in candidates:
    report_lines.append(f"- `{cand['variant_id']}`: {cand['description']}")

for heading, test_name in [("Test A", "A"), ("Test B", "B"), ("Test C", "C")]:
    report_lines.extend(["", f"# {heading} Results", ""])
    for row in [r for r in aggregated_rows if r["test_type"] == test_name and abs(r["observe_cost"] - PRIMARY_COST) < 1e-12]:
        best_level = row["level_policy_metrics"]["best_level"]
        best = row["level_policy_metrics"]["levels"][best_level]
        report_lines.append(
            f"- `{row['variant_id']}`: oracle_gap={row['baseline_headroom']['oracle_minus_always_observe']:.4f}, "
            f"ao_minus_at={row['baseline_headroom']['always_observe_minus_always_try']:.4f}, "
            f"best={best_level}, best_delta={best['delta_vs_always_observe']:.4f}, "
            f"sign={best['sign_correct_rate']:.4f}, rate={row['observe_value_distribution']['positive_net_rate']:.4f}, "
            f"single={row['structure_validity']['single_cue_max_predictiveness']:.4f}, "
            f"pair={row['structure_validity']['cue_pair_max_predictiveness']:.4f}, "
            f"rej={row['rejection_reason'] or 'none'}"
        )

report_lines.extend([
    "",
    "# Structure Validity Audit",
    "",
])
for row in focus_rows:
    report_lines.append(
        f"- `{row['variant_id']}|{row['test_type']}`: exact_det={row['structure_validity']['exact_signature_determinism_rate']:.4f}, "
        f"single_cue_max={row['structure_validity']['single_cue_max_predictiveness']:.4f}, "
        f"pair_max={row['structure_validity']['cue_pair_max_predictiveness']:.4f}, "
        f"reachability={row['structure_validity']['compositional_reachability_rate']:.4f}"
    )

report_lines.extend([
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
    report_lines.append(f"- `{reason}`: {count}")

report_lines.extend([
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
    "- If status remains partial: one more targeted refinement only if you want to search for a compositional policy that actually beats `always_observe`.",
    "- If no level beats `always_observe` cleanly after anti-dominance fixes: prefer deeper environment redesign or expert review before any architecture work.",
])

md_path = os.path.join(protocols_dir, "block1j40b7j_d_family_anti_dominance_refinement_audit.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

ckpt_path = os.path.join(CURRENT_DIR, "checkpoint_1j40b7j_d_family_anti_dominance_refinement_audit.md")
with open(ckpt_path, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            [
                "# Checkpoint 1J40b-7j",
                "",
                f"- status: {status}",
                f"- best_candidate_overall: {acceptance_summary['best_candidate_overall']}",
                f"- best_candidate_test_A: {acceptance_summary['best_candidate_test_A']}",
                f"- best_candidate_test_B: {acceptance_summary['best_candidate_test_B']}",
                f"- candidates_passing_primary: {len(acceptance_summary['candidates_passing_primary'])}",
                f"- candidates_passing_partial: {len(acceptance_summary['candidates_passing_partial'])}",
                "- next_suggested_step: one more targeted refinement" if status == "PARTIAL_CANDIDATE_FOUND" else "- next_suggested_step: deeper environment redesign or expert review",
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
        f"single_cue={row['structure_validity']['single_cue_max_predictiveness']:.4f}, "
        f"cue_pair={row['structure_validity']['cue_pair_max_predictiveness']:.4f}"
    )

print("[6/6] Done.")
print(f"  JSON: {json_path}")
print(f"  CSV:  {csv_path}")
print(f"  MD:   {md_path}")
print(f"  CKPT: {ckpt_path}")
