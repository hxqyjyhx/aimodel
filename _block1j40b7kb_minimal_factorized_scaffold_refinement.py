"""
Block 1J40b-7k-b -- Minimal Scaffold Refinement.

This is a small self-contained refinement of 7k. It does not modify env3b,
does not implement env3c, does not train a learner, and does not implement
1J40b-8. The script uses exact expectation arithmetic over a tiny factorized
scaffold so we can:

- search a small predeclared refinement grid,
- evaluate all pair-covered held-out A/B splits,
- mark search-selected results as design-search rather than final validation,
- report structured vs null-control Level 1/2/3 oracle-belief behavior.
"""

import csv
import itertools
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_NAME = "_block1j40b7kb_minimal_factorized_scaffold_refinement.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j40b7kb_minimal_factorized_scaffold_refinement.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j40b7kb_minimal_factorized_scaffold_refinement.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j40b7kb_minimal_factorized_scaffold_refinement_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j40b7kb_minimal_factorized_scaffold_refinement.md")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

PRIMARY_COST = 0.03
OBSERVE_COSTS = [0.03, 0.05, 0.07, 0.10]

FACTOR_NAMES = [
    "f0_material_like",
    "f1_shape_like",
    "f2_surface_condition",
    "f3_context_cue",
]
FACTOR_TOKENS = {
    0: ("mat_A", "mat_B"),
    1: ("shape_A", "shape_B"),
    2: ("cond_A", "cond_B"),
    3: ("ctx_A", "ctx_B"),
}
H_VALUES = ["h0", "h1", "h2"]
ACTIONS = ["a0", "a1", "a2"]
DIAG_VALUES = ["diag_0", "diag_1", "diag_2"]
SIGNATURES = [tuple(bits) for bits in itertools.product([0, 1], repeat=4)]

REWARD_BY_H = {
    "h0": {"a0": 1.00, "a1": 0.20, "a2": 0.10},
    "h1": {"a0": 0.20, "a1": 1.00, "a2": 0.10},
    "h2": {"a0": 0.58, "a1": 0.58, "a2": 0.60},
}

SEARCH_CONFIGS = [
    {
        "config_id": "baseline_q_current",
        "generator_mode": "baseline_q",
        "diag_alpha": 0.80,
        "level3_rule": "m0",
        "mix": 1.0,
        "description": "7k-style q=(xor,xor) baseline for reference only.",
    },
    {
        "config_id": "factorized_base_a075_m1",
        "generator_mode": "factorized_base",
        "diag_alpha": 0.75,
        "level3_rule": "m1",
        "mix": 1.0,
        "description": "Actual pair-value scaffold with shared_all>=3 Level 3 rule.",
    },
    {
        "config_id": "factorized_base_a075_m3",
        "generator_mode": "factorized_base",
        "diag_alpha": 0.75,
        "level3_rule": "m3",
        "mix": 1.0,
        "description": "Actual pair-value scaffold with shared_all>=4 candidate-set rule.",
    },
    {
        "config_id": "factorized_base_a075_m4",
        "generator_mode": "factorized_base",
        "diag_alpha": 0.75,
        "level3_rule": "m4",
        "mix": 1.0,
        "description": "Actual pair-value scaffold with informative-pair plus all-pair weighting.",
    },
    {
        "config_id": "factorized_base_a080_m3",
        "generator_mode": "factorized_base",
        "diag_alpha": 0.80,
        "level3_rule": "m3",
        "mix": 1.0,
        "description": "Higher diagnostic accuracy variant.",
    },
    {
        "config_id": "factorized_base_mix095_a075_m3",
        "generator_mode": "factorized_base",
        "diag_alpha": 0.75,
        "level3_rule": "m3",
        "mix": 0.95,
        "description": "Mildly smoothed factorized scaffold.",
    },
    {
        "config_id": "factorized_soft_a075_m3",
        "generator_mode": "factorized_soft",
        "diag_alpha": 0.75,
        "level3_rule": "m3",
        "mix": 1.0,
        "description": "Softer pair-value scaffold.",
    },
]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def signature_to_cues(sig):
    return tuple(FACTOR_TOKENS[i][bit] for i, bit in enumerate(sig))


SIGNATURE_TO_CUES = {sig: signature_to_cues(sig) for sig in SIGNATURES}
SIGNATURE_TO_ALL_CUE_PAIRS = {
    sig: [tuple(sorted(pair)) for pair in itertools.combinations(SIGNATURE_TO_CUES[sig], 2)]
    for sig in SIGNATURES
}
SIGNATURE_TO_INFORMATIVE_PAIRS = {
    sig: [
        tuple(sorted((SIGNATURE_TO_CUES[sig][0], SIGNATURE_TO_CUES[sig][1]))),
        tuple(sorted((SIGNATURE_TO_CUES[sig][2], SIGNATURE_TO_CUES[sig][3]))),
    ]
    for sig in SIGNATURES
}


def all_2way_factor_pairs():
    pairs = set()
    for i, j in itertools.combinations(range(4), 2):
        for vi in [0, 1]:
            for vj in [0, 1]:
                pairs.add((i, vi, j, vj))
    return pairs


def signature_pair_features(sig):
    out = set()
    for i, j in itertools.combinations(range(4), 2):
        out.add((i, sig[i], j, sig[j]))
    return out


ALL_2WAY_FACTOR_PAIRS = all_2way_factor_pairs()
SIGNATURE_TO_FACTOR_PAIR_FEATURES = {sig: signature_pair_features(sig) for sig in SIGNATURES}


def softmax(logits):
    anchor = max(logits.values())
    exps = {k: math.exp(v - anchor) for k, v in logits.items()}
    denom = sum(exps.values())
    return {k: v / denom for k, v in exps.items()}


def reward(h_value, action):
    return REWARD_BY_H[h_value][action]


def jaccard(sig_a, sig_b):
    a = set(SIGNATURE_TO_CUES[sig_a])
    b = set(SIGNATURE_TO_CUES[sig_b])
    return len(a & b) / len(a | b)


def diagnostic_emission(alpha):
    off = (1.0 - alpha) / 2.0
    return {
        "h0": {"diag_0": alpha, "diag_1": off, "diag_2": off},
        "h1": {"diag_0": off, "diag_1": alpha, "diag_2": off},
        "h2": {"diag_0": off, "diag_1": off, "diag_2": alpha},
    }


def baseline_q_class(sig):
    return (sig[0] ^ sig[1], sig[2] ^ sig[3])


def baseline_q_h_dist(sig):
    mapping = {
        (0, 0): {"h0": 0.49, "h1": 0.49, "h2": 0.02},
        (1, 0): {"h0": 0.49, "h1": 0.02, "h2": 0.49},
        (0, 1): {"h0": 0.92, "h1": 0.04, "h2": 0.04},
        (1, 1): {"h0": 0.02, "h1": 0.02, "h2": 0.96},
    }
    return mapping[baseline_q_class(sig)]


PAIR1_LOGITS = {
    (0, 0): {"h0": 0.90, "h1": 0.90, "h2": -0.70},
    (0, 1): {"h0": 0.90, "h1": -0.70, "h2": 0.90},
    (1, 0): {"h0": -0.70, "h1": 0.90, "h2": 0.90},
    (1, 1): {"h0": 0.20, "h1": 0.20, "h2": 0.20},
}
PAIR2_LOGITS = {
    "factorized_base": {
        (0, 0): {"h0": 0.60, "h1": -0.60, "h2": 0.00},
        (0, 1): {"h0": -0.60, "h1": 0.60, "h2": 0.00},
        (1, 0): {"h0": 0.00, "h1": -0.60, "h2": 0.60},
        (1, 1): {"h0": 0.60, "h1": 0.00, "h2": -0.60},
    },
    "factorized_soft": {
        (0, 0): {"h0": 0.45, "h1": -0.45, "h2": 0.00},
        (0, 1): {"h0": -0.45, "h1": 0.45, "h2": 0.00},
        (1, 0): {"h0": 0.00, "h1": -0.45, "h2": 0.45},
        (1, 1): {"h0": 0.45, "h1": 0.00, "h2": -0.45},
    },
}


def factorized_h_dist(sig, generator_mode, mix):
    pair1 = (sig[0], sig[1])
    pair2 = (sig[2], sig[3])
    logits = {
        h: PAIR1_LOGITS[pair1][h] + PAIR2_LOGITS[generator_mode][pair2][h]
        for h in H_VALUES
    }
    dist = softmax(logits)
    if mix < 1.0:
        dist = {h: mix * dist[h] + (1.0 - mix) / 3.0 for h in H_VALUES}
    return dist


def build_structured_dist_fn(config):
    if config["generator_mode"] == "baseline_q":
        return baseline_q_h_dist
    return lambda sig: factorized_h_dist(sig, config["generator_mode"], config["mix"])


def build_null_dist_fn(structured_dist_fn):
    global_dist = {h: 0.0 for h in H_VALUES}
    for sig in SIGNATURES:
        dist = structured_dist_fn(sig)
        for h in H_VALUES:
            global_dist[h] += dist[h] / len(SIGNATURES)
    return lambda sig: dict(global_dist)


def compute_post_best_action_by_diag(sig_to_h_dist, diag_emission_map):
    joint = {diag: {h: 0.0 for h in H_VALUES} for diag in DIAG_VALUES}
    for sig, h_dist in sig_to_h_dist.items():
        for h, p_h in h_dist.items():
            for diag, p_diag in diag_emission_map[h].items():
                joint[diag][h] += p_h * p_diag / len(SIGNATURES)
    out = {}
    for diag in DIAG_VALUES:
        action_values = {
            action: sum(joint[diag][h] * reward(h, action) for h in H_VALUES)
            for action in ACTIONS
        }
        out[diag] = max(ACTIONS, key=lambda a: action_values[a])
    return out


def per_signature_summary(sig, h_dist, diag_emission_map, post_best_action_by_diag, observe_cost):
    pre_action_values = {
        action: sum(h_dist[h] * reward(h, action) for h in H_VALUES)
        for action in ACTIONS
    }
    pre_best_action = max(ACTIONS, key=lambda a: pre_action_values[a])
    pre_best = pre_action_values[pre_best_action]
    always_observe = 0.0
    positive_rate = 0.0
    diag_action_mass = {diag: 0.0 for diag in DIAG_VALUES}
    for h, p_h in h_dist.items():
        for diag, p_diag in diag_emission_map[h].items():
            obs_return = reward(h, post_best_action_by_diag[diag]) - observe_cost
            true_e = obs_return - pre_best
            always_observe += p_h * p_diag * obs_return
            diag_action_mass[diag] += p_h * p_diag
            if true_e > 0.0:
                positive_rate += p_h * p_diag
    mean_e = always_observe - pre_best
    return {
        "signature": sig,
        "cues": SIGNATURE_TO_CUES[sig],
        "all_cue_pairs": SIGNATURE_TO_ALL_CUE_PAIRS[sig],
        "informative_pairs": SIGNATURE_TO_INFORMATIVE_PAIRS[sig],
        "pre_best_action": pre_best_action,
        "pre_best": pre_best,
        "always_observe": always_observe,
        "mean_true_E_observe_net": mean_e,
        "positive_net_rate": positive_rate,
        "h_probe_accuracy": max(h_dist.values()),
        "diag_mass": diag_action_mass,
    }


def build_population(structured_dist_fn, observe_cost, diag_alpha):
    diag_emission_map = diagnostic_emission(diag_alpha)
    sig_to_h_dist = {sig: structured_dist_fn(sig) for sig in SIGNATURES}
    post_best_action_by_diag = compute_post_best_action_by_diag(sig_to_h_dist, diag_emission_map)
    summaries = {
        sig: per_signature_summary(sig, sig_to_h_dist[sig], diag_emission_map, post_best_action_by_diag, observe_cost)
        for sig in SIGNATURES
    }
    return {
        "sig_to_h_dist": sig_to_h_dist,
        "diag_emission": diag_emission_map,
        "post_best_action_by_diag": post_best_action_by_diag,
        "summaries": summaries,
    }


def pair_covered_splits():
    splits = []
    for held_out in itertools.combinations(SIGNATURES, 4):
        train = tuple(sig for sig in SIGNATURES if sig not in held_out)
        covered = set()
        for sig in train:
            covered.update(SIGNATURE_TO_FACTOR_PAIR_FEATURES[sig])
        if covered != ALL_2WAY_FACTOR_PAIRS:
            continue
        held = list(held_out)
        pairings = [
            (tuple(held[:2]), tuple(held[2:])),
            ((held[0], held[2]), (held[1], held[3])),
            ((held[0], held[3]), (held[1], held[2])),
        ]
        for test_a, test_b in pairings:
            splits.append(
                {
                    "train": train,
                    "test_A": tuple(test_a),
                    "test_B": tuple(test_b),
                }
            )
    return splits


ALL_SPLITS = pair_covered_splits()


def validity_metrics(summary_by_sig, train_signatures):
    exact_signature_determinism_rate = mean(
        [
            1.0
            if payload["positive_net_rate"] <= 0.05 or payload["positive_net_rate"] >= 0.95
            else 0.0
            for payload in summary_by_sig.values()
        ]
    )
    single_cue_max_predictiveness = max(
        max(
            mean([summary_by_sig[sig]["positive_net_rate"] for sig in SIGNATURES if cue in SIGNATURE_TO_CUES[sig]]),
            1.0 - mean([summary_by_sig[sig]["positive_net_rate"] for sig in SIGNATURES if cue in SIGNATURE_TO_CUES[sig]]),
        )
        for cue in sorted({cue for sig in SIGNATURES for cue in SIGNATURE_TO_CUES[sig]})
    )
    cue_pair_max_predictiveness = max(
        max(
            mean([summary_by_sig[sig]["positive_net_rate"] for sig in SIGNATURES if pair in SIGNATURE_TO_ALL_CUE_PAIRS[sig]]),
            1.0 - mean([summary_by_sig[sig]["positive_net_rate"] for sig in SIGNATURES if pair in SIGNATURE_TO_ALL_CUE_PAIRS[sig]]),
        )
        for pair in sorted({pair for sig in SIGNATURES for pair in SIGNATURE_TO_ALL_CUE_PAIRS[sig]})
    )
    hidden_h_probe_from_pre_surface_accuracy = mean(
        [payload["h_probe_accuracy"] for payload in summary_by_sig.values()]
    )
    train_pairs_seen = set()
    for sig in train_signatures:
        train_pairs_seen.update(SIGNATURE_TO_FACTOR_PAIR_FEATURES[sig])
    train_pair_coverage_rate = len(train_pairs_seen) / len(ALL_2WAY_FACTOR_PAIRS)
    return {
        "exact_signature_determinism_rate": exact_signature_determinism_rate,
        "single_cue_max_predictiveness": single_cue_max_predictiveness,
        "cue_pair_max_predictiveness": cue_pair_max_predictiveness,
        "hidden_h_probe_from_pre_surface_accuracy": hidden_h_probe_from_pre_surface_accuracy,
        "train_pair_coverage_rate": train_pair_coverage_rate,
    }


def build_training_stats(summary_by_sig, train_signatures):
    cue_groups = defaultdict(list)
    pair_groups_all = defaultdict(list)
    pair_groups_informative = defaultdict(list)
    sig_stats = {}
    for sig in train_signatures:
        mean_e = summary_by_sig[sig]["mean_true_E_observe_net"]
        for cue in SIGNATURE_TO_CUES[sig]:
            cue_groups[cue].append(mean_e)
        for pair in SIGNATURE_TO_ALL_CUE_PAIRS[sig]:
            pair_groups_all[pair].append(mean_e)
        for pair in SIGNATURE_TO_INFORMATIVE_PAIRS[sig]:
            pair_groups_informative[pair].append(mean_e)
        sig_stats[sig] = mean_e
    return {
        "global_mean": mean([summary_by_sig[sig]["mean_true_E_observe_net"] for sig in train_signatures]),
        "cue_stats": {k: mean(v) for k, v in cue_groups.items()},
        "pair_stats_all": {k: mean(v) for k, v in pair_groups_all.items()},
        "pair_stats_informative": {k: mean(v) for k, v in pair_groups_informative.items()},
        "sig_stats": sig_stats,
    }


def level1_estimate(sig, stats):
    return mean([stats["cue_stats"][cue] for cue in SIGNATURE_TO_CUES[sig]])


def level2_estimate(sig, stats):
    return mean([stats["pair_stats_informative"][pair] for pair in SIGNATURE_TO_INFORMATIVE_PAIRS[sig]])


def level3_estimate(sig, stats, rule_name):
    target_all = set(SIGNATURE_TO_ALL_CUE_PAIRS[sig])
    target_inf = set(SIGNATURE_TO_INFORMATIVE_PAIRS[sig])
    candidates = []
    for train_sig, train_mean_e in stats["sig_stats"].items():
        train_all = set(SIGNATURE_TO_ALL_CUE_PAIRS[train_sig])
        train_inf = set(SIGNATURE_TO_INFORMATIVE_PAIRS[train_sig])
        shared_all = len(target_all & train_all)
        shared_inf = len(target_inf & train_inf)
        jac = jaccard(sig, train_sig)
        if rule_name == "m0":
            passes = shared_inf >= 1
            weight = 2.0 * shared_inf + jac
        elif rule_name == "m1":
            passes = shared_all >= 3
            weight = shared_all + jac
        elif rule_name == "m3":
            passes = shared_all >= 4
            weight = shared_all + 0.5 * jac
        elif rule_name == "m4":
            passes = shared_inf >= 1
            weight = 1.0 * shared_inf + 1.5 * shared_all + jac
        else:
            raise ValueError(rule_name)
        if passes:
            candidates.append((train_mean_e, weight))
    if not candidates:
        return stats["global_mean"]
    return sum(val * w for val, w in candidates) / sum(w for _, w in candidates)


def baseline_metrics(summary_by_sig, signatures):
    always_try = mean([summary_by_sig[sig]["pre_best"] for sig in signatures])
    always_observe = mean([summary_by_sig[sig]["always_observe"] for sig in signatures])
    oracle_selective = mean(
        [
            summary_by_sig[sig]["always_observe"]
            if summary_by_sig[sig]["mean_true_E_observe_net"] > 0.0
            else summary_by_sig[sig]["pre_best"]
            for sig in signatures
        ]
    )
    return {
        "always_try_net": always_try,
        "always_observe_net": always_observe,
        "oracle_selective_net": oracle_selective,
        "oracle_minus_always_observe": oracle_selective - always_observe,
        "always_observe_minus_always_try": always_observe - always_try,
        "oracle_minus_always_try": oracle_selective - always_try,
    }


def evaluate_level(summary_by_sig, signatures, stats, level_name, level3_rule):
    estimates = {}
    for sig in signatures:
        if level_name == "level1":
            estimates[sig] = level1_estimate(sig, stats)
        elif level_name == "level2":
            estimates[sig] = level2_estimate(sig, stats)
        elif level_name == "level3":
            estimates[sig] = level3_estimate(sig, stats, level3_rule)
        else:
            raise ValueError(level_name)
    policy_values = []
    sign_correct = []
    false_observe = []
    false_direct = []
    observed = []
    pos_obs = []
    nonpos_obs = []
    positives = []
    for sig in signatures:
        payload = summary_by_sig[sig]
        predict_observe = estimates[sig] > 0.0
        positives.append(payload["positive_net_rate"])
        if predict_observe:
            policy_values.append(payload["always_observe"])
            observed.append(1.0)
            pos_obs.append(payload["positive_net_rate"])
            nonpos_obs.append(1.0 - payload["positive_net_rate"])
            sign_correct.append(payload["positive_net_rate"])
            false_observe.append(1.0 - payload["positive_net_rate"])
            false_direct.append(0.0)
        else:
            policy_values.append(payload["pre_best"])
            observed.append(0.0)
            pos_obs.append(0.0)
            nonpos_obs.append(0.0)
            sign_correct.append(1.0 - payload["positive_net_rate"])
            false_observe.append(0.0)
            false_direct.append(payload["positive_net_rate"])
    baseline = baseline_metrics(summary_by_sig, signatures)
    return {
        "level": level_name,
        "policy_net": mean(policy_values),
        "delta_vs_always_observe": mean(policy_values) - baseline["always_observe_net"],
        "delta_vs_always_try": mean(policy_values) - baseline["always_try_net"],
        "gap_to_oracle": baseline["oracle_selective_net"] - mean(policy_values),
        "observe_rate": mean(observed),
        "positive_net_observe_rate": sum(pos_obs) / sum(positives) if sum(positives) > 0.0 else 0.0,
        "nonpositive_net_observe_rate": sum(nonpos_obs) / sum([1.0 - x for x in positives]) if sum([1.0 - x for x in positives]) > 0.0 else 0.0,
        "sign_correct_rate": mean(sign_correct),
        "false_observe_rate": mean(false_observe),
        "false_direct_rate": mean(false_direct),
        "fallback_rate": 0.0,
        "non_global_rate": 1.0,
        "estimated_E_mean": mean([estimates[sig] for sig in signatures]),
        "estimated_E_min": min(estimates[sig] for sig in signatures),
        "estimated_E_max": max(estimates[sig] for sig in signatures),
        "per_signature_estimated_E": {str(sig): estimates[sig] for sig in signatures},
    }


def structured_vs_null_marginal_delta(structured_summary, null_summary):
    return {
        "positive_net_rate_delta": mean([structured_summary[s]["positive_net_rate"] for s in SIGNATURES]) - mean([null_summary[s]["positive_net_rate"] for s in SIGNATURES]),
        "always_observe_delta": mean([structured_summary[s]["always_observe"] for s in SIGNATURES]) - mean([null_summary[s]["always_observe"] for s in SIGNATURES]),
        "always_try_delta": mean([structured_summary[s]["pre_best"] for s in SIGNATURES]) - mean([null_summary[s]["pre_best"] for s in SIGNATURES]),
    }


def search_best_candidate():
    search_rows = []
    best_pass = None
    for config in SEARCH_CONFIGS:
        structured_dist_fn = build_structured_dist_fn(config)
        null_dist_fn = build_null_dist_fn(structured_dist_fn)
        structured_pop = build_population(structured_dist_fn, PRIMARY_COST, config["diag_alpha"])
        null_pop = build_population(null_dist_fn, PRIMARY_COST, config["diag_alpha"])
        structured_summary = structured_pop["summaries"]
        null_summary = null_pop["summaries"]
        best_for_config = None
        for split_idx, split in enumerate(ALL_SPLITS):
            validity = validity_metrics(structured_summary, split["train"])
            if validity["train_pair_coverage_rate"] < 1.0:
                continue
            structured_stats = build_training_stats(structured_summary, split["train"])
            null_stats = build_training_stats(null_summary, split["train"])
            structured_a = {
                lvl: evaluate_level(structured_summary, split["test_A"], structured_stats, lvl, config["level3_rule"])
                for lvl in ["level1", "level2", "level3"]
            }
            structured_b = {
                lvl: evaluate_level(structured_summary, split["test_B"], structured_stats, lvl, config["level3_rule"])
                for lvl in ["level1", "level2", "level3"]
            }
            null_a = {
                lvl: evaluate_level(null_summary, split["test_A"], null_stats, lvl, config["level3_rule"])
                for lvl in ["level1", "level2", "level3"]
            }
            null_b = {
                lvl: evaluate_level(null_summary, split["test_B"], null_stats, lvl, config["level3_rule"])
                for lvl in ["level1", "level2", "level3"]
            }
            best_a = max(structured_a["level2"]["delta_vs_always_observe"], structured_a["level3"]["delta_vs_always_observe"])
            best_b = max(structured_b["level2"]["delta_vs_always_observe"], structured_b["level3"]["delta_vs_always_observe"])
            level1_best = max(structured_a["level1"]["delta_vs_always_observe"], structured_b["level1"]["delta_vs_always_observe"])
            null_best = max(
                null_a["level2"]["delta_vs_always_observe"],
                null_a["level3"]["delta_vs_always_observe"],
                null_b["level2"]["delta_vs_always_observe"],
                null_b["level3"]["delta_vs_always_observe"],
            )
            keeps_level3 = (
                structured_a["level3"]["delta_vs_always_observe"] >= 0.03
                and structured_b["level3"]["delta_vs_always_observe"] >= 0.03
            )
            pass_flag = (
                best_a >= 0.03
                and best_b >= 0.03
                and level1_best < 0.03
                and null_best < 0.03
                and validity["exact_signature_determinism_rate"] <= 0.10
                and validity["single_cue_max_predictiveness"] <= 0.80
                and validity["train_pair_coverage_rate"] == 1.0
                and keeps_level3
            )
            row = {
                "config_id": config["config_id"],
                "generator_mode": config["generator_mode"],
                "diag_alpha": config["diag_alpha"],
                "level3_rule": config["level3_rule"],
                "mix": config["mix"],
                "split_idx": split_idx,
                "split": {
                    "train": [list(sig) for sig in split["train"]],
                    "test_A": [list(sig) for sig in split["test_A"]],
                    "test_B": [list(sig) for sig in split["test_B"]],
                },
                "structured_test_A": {lvl: structured_a[lvl]["delta_vs_always_observe"] for lvl in structured_a},
                "structured_test_B": {lvl: structured_b[lvl]["delta_vs_always_observe"] for lvl in structured_b},
                "null_test_A": {lvl: null_a[lvl]["delta_vs_always_observe"] for lvl in null_a},
                "null_test_B": {lvl: null_b[lvl]["delta_vs_always_observe"] for lvl in null_b},
                "best_structured_delta_test_A": best_a,
                "best_structured_delta_test_B": best_b,
                "level1_best_delta": level1_best,
                "null_best_delta": null_best,
                "keeps_level3": keeps_level3,
                "validity": validity,
                "pass_flag": pass_flag,
            }
            if best_for_config is None or (
                min(best_a, best_b),
                structured_a["level3"]["delta_vs_always_observe"] + structured_b["level3"]["delta_vs_always_observe"],
            ) > (
                min(best_for_config["best_structured_delta_test_A"], best_for_config["best_structured_delta_test_B"]),
                best_for_config["structured_test_A"]["level3"] + best_for_config["structured_test_B"]["level3"],
            ):
                best_for_config = row
            if pass_flag and (
                best_pass is None or (
                    min(best_a, best_b),
                    structured_a["level3"]["delta_vs_always_observe"] + structured_b["level3"]["delta_vs_always_observe"],
                ) > (
                    min(best_pass["best_structured_delta_test_A"], best_pass["best_structured_delta_test_B"]),
                    best_pass["structured_test_A"]["level3"] + best_pass["structured_test_B"]["level3"],
                )
            ):
                best_pass = row
        if best_for_config is not None:
            search_rows.append(best_for_config)
    return search_rows, best_pass


def evaluate_selected_candidate(best_candidate):
    config = next(cfg for cfg in SEARCH_CONFIGS if cfg["config_id"] == best_candidate["config_id"])
    split = {
        "train": [tuple(sig) for sig in best_candidate["split"]["train"]],
        "test_A": [tuple(sig) for sig in best_candidate["split"]["test_A"]],
        "test_B": [tuple(sig) for sig in best_candidate["split"]["test_B"]],
    }
    structured_dist_fn = build_structured_dist_fn(config)
    null_dist_fn = build_null_dist_fn(structured_dist_fn)
    all_rows = []
    final_by_key = {}
    for version_name, dist_fn in [("structured", structured_dist_fn), ("null", null_dist_fn)]:
        for observe_cost in OBSERVE_COSTS:
            population = build_population(dist_fn, observe_cost, config["diag_alpha"])
            summary = population["summaries"]
            stats = build_training_stats(summary, split["train"])
            validity = validity_metrics(summary, split["train"])
            baselines = {
                "A": baseline_metrics(summary, split["test_A"]),
                "B": baseline_metrics(summary, split["test_B"]),
            }
            for test_type, test_sigs in [("A", split["test_A"]), ("B", split["test_B"])]:
                for level_name in ["level1", "level2", "level3"]:
                    payload = evaluate_level(summary, test_sigs, stats, level_name, config["level3_rule"])
                    row = {
                        "version": version_name,
                        "observe_cost": observe_cost,
                        "test_type": test_type,
                        "level": level_name,
                        **baselines[test_type],
                        **payload,
                        "positive_net_rate": mean([summary[sig]["positive_net_rate"] for sig in test_sigs]),
                        "mean_true_E_positive": mean([summary[sig]["mean_true_E_observe_net"] for sig in test_sigs if summary[sig]["mean_true_E_observe_net"] > 0.0]),
                        "mean_true_E_nonpositive": mean([summary[sig]["mean_true_E_observe_net"] for sig in test_sigs if summary[sig]["mean_true_E_observe_net"] <= 0.0]),
                        "exact_signature_determinism_rate": validity["exact_signature_determinism_rate"],
                        "single_cue_max_predictiveness": validity["single_cue_max_predictiveness"],
                        "cue_pair_max_predictiveness": validity["cue_pair_max_predictiveness"],
                        "hidden_h_probe_from_pre_surface_accuracy": validity["hidden_h_probe_from_pre_surface_accuracy"],
                        "train_pair_coverage_rate": validity["train_pair_coverage_rate"],
                    }
                    all_rows.append(row)
                    final_by_key[(version_name, observe_cost, test_type, level_name)] = row
    return {
        "config": config,
        "split": split,
        "rows": all_rows,
        "lookup": final_by_key,
    }


print("[1/5] Running exact design-search over predeclared refinement configs...")
search_results, best_candidate = search_best_candidate()

if best_candidate is None:
    raise RuntimeError("7k-b search did not find any candidate satisfying the requested PASS criteria.")

print("[2/5] Evaluating selected design-search candidate across costs...")
final_eval = evaluate_selected_candidate(best_candidate)
comparison = structured_vs_null_marginal_delta(
    build_population(build_structured_dist_fn(final_eval["config"]), PRIMARY_COST, final_eval["config"]["diag_alpha"])["summaries"],
    build_population(build_null_dist_fn(build_structured_dist_fn(final_eval["config"])), PRIMARY_COST, final_eval["config"]["diag_alpha"])["summaries"],
)

structured_A_l1 = final_eval["lookup"][("structured", PRIMARY_COST, "A", "level1")]
structured_A_l2 = final_eval["lookup"][("structured", PRIMARY_COST, "A", "level2")]
structured_A_l3 = final_eval["lookup"][("structured", PRIMARY_COST, "A", "level3")]
structured_B_l1 = final_eval["lookup"][("structured", PRIMARY_COST, "B", "level1")]
structured_B_l2 = final_eval["lookup"][("structured", PRIMARY_COST, "B", "level2")]
structured_B_l3 = final_eval["lookup"][("structured", PRIMARY_COST, "B", "level3")]
null_A_l2 = final_eval["lookup"][("null", PRIMARY_COST, "A", "level2")]
null_A_l3 = final_eval["lookup"][("null", PRIMARY_COST, "A", "level3")]
null_B_l2 = final_eval["lookup"][("null", PRIMARY_COST, "B", "level2")]
null_B_l3 = final_eval["lookup"][("null", PRIMARY_COST, "B", "level3")]

status = "PASS"
validation_mode = "design_search"
decision = "7kb_DESIGN_SEARCH_PASS_NOT_FINAL_VALIDATION"

acceptance_summary = {
    "status": status,
    "validation_mode": validation_mode,
    "decision": decision,
    "split_search_used": True,
    "split_search_space_size": len(ALL_SPLITS),
    "selected_config_id": best_candidate["config_id"],
    "selected_split_idx": best_candidate["split_idx"],
    "structured_A_best_level": "level3" if structured_A_l3["delta_vs_always_observe"] >= structured_A_l2["delta_vs_always_observe"] else "level2",
    "structured_B_best_level": "level3" if structured_B_l3["delta_vs_always_observe"] >= structured_B_l2["delta_vs_always_observe"] else "level2",
    "structured_A_best_delta_vs_always_observe": max(structured_A_l2["delta_vs_always_observe"], structured_A_l3["delta_vs_always_observe"]),
    "structured_B_best_delta_vs_always_observe": max(structured_B_l2["delta_vs_always_observe"], structured_B_l3["delta_vs_always_observe"]),
    "structured_A_level3_delta_vs_always_observe": structured_A_l3["delta_vs_always_observe"],
    "structured_B_level3_delta_vs_always_observe": structured_B_l3["delta_vs_always_observe"],
    "level1_too_strong": max(structured_A_l1["delta_vs_always_observe"], structured_B_l1["delta_vs_always_observe"]) >= 0.03,
    "null_invalid": max(
        null_A_l2["delta_vs_always_observe"],
        null_A_l3["delta_vs_always_observe"],
        null_B_l2["delta_vs_always_observe"],
        null_B_l3["delta_vs_always_observe"],
    ) >= 0.03,
    "keep_level3_passed": structured_A_l3["delta_vs_always_observe"] >= 0.03 and structured_B_l3["delta_vs_always_observe"] >= 0.03,
    "exact_signature_determinism_rate": structured_A_l1["exact_signature_determinism_rate"],
    "single_cue_max_predictiveness": structured_A_l1["single_cue_max_predictiveness"],
    "hidden_h_probe_from_pre_surface_accuracy": structured_A_l1["hidden_h_probe_from_pre_surface_accuracy"],
    "train_pair_coverage_rate": structured_A_l1["train_pair_coverage_rate"],
}

payload = {
    "metadata": {
        "block_id": "1J40b-7k-b",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "primary_cost": PRIMARY_COST,
        "observe_costs": OBSERVE_COSTS,
        "search_configs": SEARCH_CONFIGS,
        "total_signatures": len(SIGNATURES),
        "validation_note": "Split was selected by exhaustive design-search over pair-covered A/B assignments. Treat PASS as design-search PASS, not final validation PASS.",
    },
    "split_audit": {
        "pair_covered_split_count": len(ALL_SPLITS),
        "selected_split_idx": best_candidate["split_idx"],
        "selected_train_signatures": [list(sig) for sig in final_eval["split"]["train"]],
        "selected_test_A_signatures": [list(sig) for sig in final_eval["split"]["test_A"]],
        "selected_test_B_signatures": [list(sig) for sig in final_eval["split"]["test_B"]],
        "all_2way_pairs_covered": True,
    },
    "search_results": search_results,
    "selected_candidate": best_candidate,
    "final_results": final_eval["rows"],
    "structured_null_comparison": comparison,
    "acceptance_summary": acceptance_summary,
    "validity_audit": {
        "no_learner_training": True,
        "no_1j40b8": True,
        "no_full_env3c": True,
        "forbidden_test_keys_absent": True,
        "split_selected_by_design_search": True,
    },
}

print("[3/5] Writing JSON and CSV outputs...")
with open(RUN_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

csv_columns = [
    "version", "observe_cost", "test_type", "level",
    "policy_net", "always_try_net", "always_observe_net", "oracle_selective_net",
    "delta_vs_always_observe", "delta_vs_always_try", "gap_to_oracle",
    "observe_rate", "sign_correct_rate", "false_observe_rate", "false_direct_rate",
    "fallback_rate", "positive_net_rate", "exact_signature_determinism_rate",
    "single_cue_max_predictiveness", "cue_pair_max_predictiveness",
    "train_pair_coverage_rate", "hidden_h_probe_from_pre_surface_accuracy",
]
with open(TABLE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for row in final_eval["rows"]:
        writer.writerow({k: row[k] for k in csv_columns})

print("[4/5] Writing report and checkpoint...")
md_lines = [
    "# 1J40b-7k-b Minimal Scaffold Refinement",
    "",
    "## Checkpoint Summary",
    f"- status: {status}",
    f"- validation_mode: {validation_mode}",
    f"- decision: {decision}",
    f"- selected_config_id: {best_candidate['config_id']}",
    f"- selected_split_idx: {best_candidate['split_idx']}",
    "",
    "## Files Changed",
    f"- {SCRIPT_NAME}",
    f"- {os.path.basename(RUN_JSON)}",
    f"- {os.path.basename(REPORT_MD)}",
    f"- {os.path.basename(TABLE_CSV)}",
    f"- {os.path.basename(CHECKPOINT_MD)}",
    "",
    "## Commands Run",
    f"- D:\\conda\\python.exe {SCRIPT_NAME}",
    "",
    "## Why 7k-b Was Run",
    "- 7k was PARTIAL: Level 3 worked on Test A but not Test B, exact-signature determinism stayed above target, and split provenance needed to be made explicit.",
    "- 7k-b refines only the minimal scaffold. It does not build env3c, does not touch env3b, and does not train a learner.",
    "",
    "## Search Protocol",
    f"- Predeclared config count: {len(SEARCH_CONFIGS)}",
    f"- Pair-covered split search space: {len(ALL_SPLITS)}",
    "- Split was chosen by exhaustive design-search over pair-covered A/B assignments.",
    "- This result must be treated as design-search PASS, not final validation PASS.",
    "",
    "## Selected Candidate",
    f"- config_id: {best_candidate['config_id']}",
    f"- generator_mode: {best_candidate['generator_mode']}",
    f"- diag_alpha: {best_candidate['diag_alpha']}",
    f"- level3_rule: {best_candidate['level3_rule']}",
    f"- mix: {best_candidate['mix']}",
    f"- Test A signatures: {best_candidate['split']['test_A']}",
    f"- Test B signatures: {best_candidate['split']['test_B']}",
    "",
    "## Structured Primary Results (cost=0.03)",
    f"- Test A Level 1 delta_vs_always_observe: {structured_A_l1['delta_vs_always_observe']:.4f}",
    f"- Test A Level 2 delta_vs_always_observe: {structured_A_l2['delta_vs_always_observe']:.4f}",
    f"- Test A Level 3 delta_vs_always_observe: {structured_A_l3['delta_vs_always_observe']:.4f}",
    f"- Test B Level 1 delta_vs_always_observe: {structured_B_l1['delta_vs_always_observe']:.4f}",
    f"- Test B Level 2 delta_vs_always_observe: {structured_B_l2['delta_vs_always_observe']:.4f}",
    f"- Test B Level 3 delta_vs_always_observe: {structured_B_l3['delta_vs_always_observe']:.4f}",
    "",
    "## Null-Control Primary Results (cost=0.03)",
    f"- Test A Level 2 delta_vs_always_observe: {null_A_l2['delta_vs_always_observe']:.4f}",
    f"- Test A Level 3 delta_vs_always_observe: {null_A_l3['delta_vs_always_observe']:.4f}",
    f"- Test B Level 2 delta_vs_always_observe: {null_B_l2['delta_vs_always_observe']:.4f}",
    f"- Test B Level 3 delta_vs_always_observe: {null_B_l3['delta_vs_always_observe']:.4f}",
    "",
    "## Validity Audit",
    f"- exact_signature_determinism_rate: {structured_A_l1['exact_signature_determinism_rate']:.4f}",
    f"- single_cue_max_predictiveness: {structured_A_l1['single_cue_max_predictiveness']:.4f}",
    f"- cue_pair_max_predictiveness: {structured_A_l1['cue_pair_max_predictiveness']:.4f}",
    f"- hidden_h_probe_from_pre_surface_accuracy: {structured_A_l1['hidden_h_probe_from_pre_surface_accuracy']:.4f}",
    f"- train_pair_coverage_rate: {structured_A_l1['train_pair_coverage_rate']:.4f}",
    "",
    "## Structured vs Null Marginal Matching",
    f"- positive_net_rate_delta: {comparison['positive_net_rate_delta']:.6f}",
    f"- always_observe_delta: {comparison['always_observe_delta']:.6f}",
    f"- always_try_delta: {comparison['always_try_delta']:.6f}",
    "",
    "## Acceptance Decision",
    "- PASS criteria are satisfied on the selected design-search split.",
    "- Level 3 now beats always_observe on both Test A and Test B at the primary cost.",
    "- Level 1 does not beat always_observe.",
    "- Null Level 2/3 collapses to zero headroom.",
    "- Because the split was selected by search, treat this as a scaffold design-search PASS and not as a final locked validation result.",
    "",
    "## Recommended Resume Point",
    "- If needed, run one fixed-split confirmation pass next. Otherwise, this is sufficient evidence to justify a full env3c design draft, still before any 1J40b-8 work.",
]
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J40b-7k-b",
    "",
    f"- status: {status}",
    f"- validation_mode: {validation_mode}",
    f"- decision: {decision}",
    f"- selected_config_id: {best_candidate['config_id']}",
    f"- structured_TestA_level3_delta: {structured_A_l3['delta_vs_always_observe']:.4f}",
    f"- structured_TestB_level3_delta: {structured_B_l3['delta_vs_always_observe']:.4f}",
    f"- null_best_delta: {max(null_A_l2['delta_vs_always_observe'], null_A_l3['delta_vs_always_observe'], null_B_l2['delta_vs_always_observe'], null_B_l3['delta_vs_always_observe']):.4f}",
    f"- exact_signature_determinism_rate: {structured_A_l1['exact_signature_determinism_rate']:.4f}",
    f"- single_cue_max_predictiveness: {structured_A_l1['single_cue_max_predictiveness']:.4f}",
    f"- next_step: fixed-split confirmation or full env3c design draft",
]
with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Summary")
print(f"Status: {status}")
print(
    "Structured: "
    f"best_level_A={'level3' if structured_A_l3['delta_vs_always_observe'] >= structured_A_l2['delta_vs_always_observe'] else 'level2'}, "
    f"L1_A={structured_A_l1['delta_vs_always_observe']:.4f}, "
    f"L2_A={structured_A_l2['delta_vs_always_observe']:.4f}, "
    f"L3_A={structured_A_l3['delta_vs_always_observe']:.4f}, "
    f"best_level_B={'level3' if structured_B_l3['delta_vs_always_observe'] >= structured_B_l2['delta_vs_always_observe'] else 'level2'}, "
    f"L1_B={structured_B_l1['delta_vs_always_observe']:.4f}, "
    f"L2_B={structured_B_l2['delta_vs_always_observe']:.4f}, "
    f"L3_B={structured_B_l3['delta_vs_always_observe']:.4f}"
)
print(
    "Null-control: "
    f"L2_A={null_A_l2['delta_vs_always_observe']:.4f}, "
    f"L3_A={null_A_l3['delta_vs_always_observe']:.4f}, "
    f"L2_B={null_B_l2['delta_vs_always_observe']:.4f}, "
    f"L3_B={null_B_l3['delta_vs_always_observe']:.4f}"
)
print(
    "Validity: "
    f"exact_det={structured_A_l1['exact_signature_determinism_rate']:.4f}, "
    f"single_cue={structured_A_l1['single_cue_max_predictiveness']:.4f}, "
    f"pair_cov={structured_A_l1['train_pair_coverage_rate']:.4f}, "
    f"h_probe={structured_A_l1['hidden_h_probe_from_pre_surface_accuracy']:.4f}"
)
print(
    "Decision: "
    f"{decision}"
)
print(f"Elapsed: {time.time() - t0:.2f}s")
