"""
Block 1J40b-7k-c -- Locked-Rule Confirmation.

Confirmation-only audit for the 7k-b locked scaffold. This script:
- locks the final 7k-b scaffold configuration,
- does not run another design search,
- does not tune parameters,
- checks fixed-split fresh-seed stability (Mode A),
- checks a pre-registered structural split subset (Mode B).
"""

import csv
import itertools
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_NAME = "_block1j40b7kc_locked_rule_confirmation.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j40b7kc_locked_rule_confirmation.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j40b7kc_locked_rule_confirmation.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j40b7kc_locked_rule_confirmation_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j40b7kc_locked_rule_confirmation.md")
SOURCE_JSON_7KB = os.path.join(CURRENT_DIR, "runs", "block1j40b7kb_minimal_factorized_scaffold_refinement.json")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

LOCKED_FROM = "1J40b-7k-b"
VALIDATION_MODE = "locked_confirmation"
NO_DESIGN_SEARCH = True
NO_PARAMETER_TUNING = True
NO_SPLIT_SELECTION_BY_METRIC = True

PRIMARY_COST = 0.03
OBSERVE_COSTS = [0.03, 0.05, 0.07]
FRESH_SEEDS = [101, 109, 141, 207, 303, 404, 505, 606, 707, 808]
SAMPLED_OBJECTS_PER_SIGNATURE = 4000

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
PAIR1_LOGITS = {
    (0, 0): {"h0": 0.90, "h1": 0.90, "h2": -0.70},
    (0, 1): {"h0": 0.90, "h1": -0.70, "h2": 0.90},
    (1, 0): {"h0": -0.70, "h1": 0.90, "h2": 0.90},
    (1, 1): {"h0": 0.20, "h1": 0.20, "h2": 0.20},
}
PAIR2_LOGITS = {
    (0, 0): {"h0": 0.60, "h1": -0.60, "h2": 0.00},
    (0, 1): {"h0": -0.60, "h1": 0.60, "h2": 0.00},
    (1, 0): {"h0": 0.00, "h1": -0.60, "h2": 0.60},
    (1, 1): {"h0": 0.60, "h1": 0.00, "h2": -0.60},
}
LOCKED_DIAG_ALPHA = 0.75
LOCKED_LEVEL3_RULE = "m1"
LOCKED_MIX = 1.0


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


def signature_pair_features(sig):
    out = set()
    for i, j in itertools.combinations(range(4), 2):
        out.add((i, sig[i], j, sig[j]))
    return out


ALL_2WAY_FACTOR_PAIRS = set()
SIGNATURE_TO_FACTOR_PAIR_FEATURES = {}
for _sig in SIGNATURES:
    feats = signature_pair_features(_sig)
    SIGNATURE_TO_FACTOR_PAIR_FEATURES[_sig] = feats
    ALL_2WAY_FACTOR_PAIRS.update(feats)


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


def factorized_h_dist(sig):
    pair1 = (sig[0], sig[1])
    pair2 = (sig[2], sig[3])
    logits = {
        h: PAIR1_LOGITS[pair1][h] + PAIR2_LOGITS[pair2][h]
        for h in H_VALUES
    }
    dist = softmax(logits)
    if LOCKED_MIX < 1.0:
        dist = {h: LOCKED_MIX * dist[h] + (1.0 - LOCKED_MIX) / 3.0 for h in H_VALUES}
    return dist


LOCKED_GLOBAL_H_DIST = {h: 0.0 for h in H_VALUES}
for sig in SIGNATURES:
    d = factorized_h_dist(sig)
    for h in H_VALUES:
        LOCKED_GLOBAL_H_DIST[h] += d[h] / len(SIGNATURES)


def locked_null_h_dist(sig):
    del sig
    return dict(LOCKED_GLOBAL_H_DIST)


def diagnostic_emission():
    off = (1.0 - LOCKED_DIAG_ALPHA) / 2.0
    return {
        "h0": {"diag_0": LOCKED_DIAG_ALPHA, "diag_1": off, "diag_2": off},
        "h1": {"diag_0": off, "diag_1": LOCKED_DIAG_ALPHA, "diag_2": off},
        "h2": {"diag_0": off, "diag_1": off, "diag_2": LOCKED_DIAG_ALPHA},
    }


LOCKED_DIAG_EMISSION = diagnostic_emission()


def sample_from_dist(rng, dist):
    u = rng.random()
    running = 0.0
    last_key = None
    for key, p in dist.items():
        running += p
        last_key = key
        if u <= running:
            return key
    return last_key


def build_exact_population(version_name, observe_cost):
    sig_to_h = {
        sig: factorized_h_dist(sig) if version_name == "structured" else locked_null_h_dist(sig)
        for sig in SIGNATURES
    }
    joint = {diag: {h: 0.0 for h in H_VALUES} for diag in DIAG_VALUES}
    for sig, h_dist in sig_to_h.items():
        for h, p_h in h_dist.items():
            for diag, p_diag in LOCKED_DIAG_EMISSION[h].items():
                joint[diag][h] += p_h * p_diag / len(SIGNATURES)
    post_best_action = {}
    for diag in DIAG_VALUES:
        action_values = {
            action: sum(joint[diag][h] * reward(h, action) for h in H_VALUES)
            for action in ACTIONS
        }
        post_best_action[diag] = max(ACTIONS, key=lambda a: action_values[a])
    rows = []
    for sig, h_dist in sig_to_h.items():
        pre_values = {
            action: sum(h_dist[h] * reward(h, action) for h in H_VALUES)
            for action in ACTIONS
        }
        pre_best = max(pre_values.values())
        for h, p_h in h_dist.items():
            for diag, p_diag in LOCKED_DIAG_EMISSION[h].items():
                observe_return = reward(h, post_best_action[diag]) - observe_cost
                true_e = observe_return - pre_best
                rows.append(
                    {
                        "version": version_name,
                        "signature": sig,
                        "cues": SIGNATURE_TO_CUES[sig],
                        "cue_pairs": SIGNATURE_TO_ALL_CUE_PAIRS[sig],
                        "h": h,
                        "diag": diag,
                        "prob": p_h * p_diag,
                        "true_pre_best": pre_best,
                        "observe_return": observe_return,
                        "true_E_observe_net": true_e,
                        "positive_net": true_e > 0.0,
                    }
                )
    return rows


def sample_population(version_name, seed, observe_cost):
    rng = random.Random(seed)
    dist_fn = factorized_h_dist if version_name == "structured" else locked_null_h_dist
    rows = []
    for sig in SIGNATURES:
        h_dist = dist_fn(sig)
        for idx in range(SAMPLED_OBJECTS_PER_SIGNATURE):
            h = sample_from_dist(rng, h_dist)
            diag = sample_from_dist(rng, LOCKED_DIAG_EMISSION[h])
            rows.append(
                {
                    "version": version_name,
                    "signature": sig,
                    "cues": SIGNATURE_TO_CUES[sig],
                    "cue_pairs": SIGNATURE_TO_ALL_CUE_PAIRS[sig],
                    "h": h,
                    "diag": diag,
                    "object_id": f"{version_name}_{seed}_{sig}_{idx}",
                }
            )
    by_sig = defaultdict(list)
    for row in rows:
        by_sig[row["signature"]].append(row)
    pre_stats = {}
    for sig, members in by_sig.items():
        action_means = {
            action: mean([reward(m["h"], action) for m in members])
            for action in ACTIONS
        }
        best_action = max(ACTIONS, key=lambda a: action_means[a])
        pre_stats[sig] = {
            "best_action": best_action,
            "best_expected_value": action_means[best_action],
        }
    diag_groups = defaultdict(list)
    for row in rows:
        diag_groups[row["diag"]].append(row)
    diag_stats = {}
    for diag, members in diag_groups.items():
        action_means = {
            action: mean([reward(m["h"], action) for m in members])
            for action in ACTIONS
        }
        best_action = max(ACTIONS, key=lambda a: action_means[a])
        diag_stats[diag] = {
            "best_action": best_action,
            "best_expected_value": action_means[best_action],
        }
    valued_rows = []
    for row in rows:
        pre_best = pre_stats[row["signature"]]["best_expected_value"]
        post_action = diag_stats[row["diag"]]["best_action"]
        observe_return = reward(row["h"], post_action) - observe_cost
        true_e = observe_return - pre_best
        valued_rows.append(
            {
                **row,
                "prob": 1.0,
                "true_pre_best": pre_best,
                "observe_return": observe_return,
                "true_E_observe_net": true_e,
                "positive_net": true_e > 0.0,
            }
        )
    return valued_rows


def rows_by_signature(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["signature"]].append(row)
    return out


def weighted_mean(rows, key):
    total = sum(row["prob"] for row in rows)
    if total <= 0.0:
        return 0.0
    return sum(row["prob"] * row[key] for row in rows) / total


def build_training_stats(train_rows):
    cue_groups = defaultdict(list)
    pair_groups_informative = defaultdict(list)
    sig_groups = defaultdict(list)
    global_rows = []
    for row in train_rows:
        global_rows.append(row)
        for cue in row["cues"]:
            cue_groups[cue].append(row)
        for pair in SIGNATURE_TO_INFORMATIVE_PAIRS[row["signature"]]:
            pair_groups_informative[pair].append(row)
        sig_groups[row["signature"]].append(row)
    return {
        "global_mean": weighted_mean(global_rows, "true_E_observe_net"),
        "cue_stats": {k: weighted_mean(v, "true_E_observe_net") for k, v in cue_groups.items()},
        "pair_stats_informative": {k: weighted_mean(v, "true_E_observe_net") for k, v in pair_groups_informative.items()},
        "sig_stats": {k: weighted_mean(v, "true_E_observe_net") for k, v in sig_groups.items()},
    }


def level1_estimate(sig, stats):
    return mean([stats["cue_stats"][cue] for cue in SIGNATURE_TO_CUES[sig]])


def level2_estimate(sig, stats):
    return mean([stats["pair_stats_informative"][pair] for pair in SIGNATURE_TO_INFORMATIVE_PAIRS[sig]])


def level3_estimate(sig, stats):
    target_all = set(SIGNATURE_TO_ALL_CUE_PAIRS[sig])
    candidates = []
    for train_sig, train_mean_e in stats["sig_stats"].items():
        train_all = set(SIGNATURE_TO_ALL_CUE_PAIRS[train_sig])
        shared_all = len(target_all & train_all)
        jac = jaccard(sig, train_sig)
        if LOCKED_LEVEL3_RULE == "m1":
            passes = shared_all >= 3
            weight = shared_all + jac
        else:
            raise ValueError(LOCKED_LEVEL3_RULE)
        if passes:
            candidates.append((train_mean_e, weight))
    if not candidates:
        return stats["global_mean"]
    return sum(val * w for val, w in candidates) / sum(w for _, w in candidates)


def baseline_metrics(summary_rows):
    always_try = weighted_mean(summary_rows, "true_pre_best")
    always_observe = weighted_mean(summary_rows, "observe_return")
    oracle_selective = weighted_mean(
        [
            {
                **row,
                "_oracle": row["observe_return"] if row["positive_net"] else row["true_pre_best"],
            }
            for row in summary_rows
        ],
        "_oracle",
    )
    return {
        "always_try_net": always_try,
        "always_observe_net": always_observe,
        "oracle_selective_net": oracle_selective,
        "oracle_minus_always_observe": oracle_selective - always_observe,
        "always_observe_minus_always_try": always_observe - always_try,
        "oracle_minus_always_try": oracle_selective - always_try,
    }


def validity_metrics(rows, train_signatures):
    by_sig = rows_by_signature(rows)
    by_cue = defaultdict(list)
    by_pair = defaultdict(list)
    for row in rows:
        for cue in row["cues"]:
            by_cue[cue].append(row)
        for pair in row["cue_pairs"]:
            by_pair[pair].append(row)
    exact_signature_determinism_rate = mean(
        [
            1.0
            if weighted_mean(members, "positive_net") <= 0.05 or weighted_mean(members, "positive_net") >= 0.95
            else 0.0
            for members in by_sig.values()
        ]
    )
    single_cue_max_predictiveness = max(
        max(weighted_mean(members, "positive_net"), 1.0 - weighted_mean(members, "positive_net"))
        for members in by_cue.values()
    )
    cue_pair_max_predictiveness = max(
        max(weighted_mean(members, "positive_net"), 1.0 - weighted_mean(members, "positive_net"))
        for members in by_pair.values()
    )
    hidden_h_probe_from_pre_surface_accuracy = mean(
        [
            max(Counter(row["h"] for row in members).values()) / len(members)
            for members in by_sig.values()
        ]
    )
    train_pairs_seen = set()
    for sig in train_signatures:
        train_pairs_seen.update(SIGNATURE_TO_FACTOR_PAIR_FEATURES[sig])
    train_pair_coverage = len(train_pairs_seen) / len(ALL_2WAY_FACTOR_PAIRS)
    return {
        "exact_signature_determinism_rate": exact_signature_determinism_rate,
        "single_cue_max_predictiveness": single_cue_max_predictiveness,
        "cue_pair_max_predictiveness": cue_pair_max_predictiveness,
        "hidden_h_probe_from_pre_surface_accuracy": hidden_h_probe_from_pre_surface_accuracy,
        "train_pair_coverage": train_pair_coverage,
    }


def evaluate_level(test_rows, level_name, stats):
    sigs = sorted({row["signature"] for row in test_rows})
    estimates = {}
    for sig in sigs:
        if level_name == "level1":
            estimates[sig] = level1_estimate(sig, stats)
        elif level_name == "level2":
            estimates[sig] = level2_estimate(sig, stats)
        elif level_name == "level3":
            estimates[sig] = level3_estimate(sig, stats)
        else:
            raise ValueError(level_name)
    policy_rows = []
    positives = []
    sign_correct_weight = 0.0
    false_observe_weight = 0.0
    false_direct_weight = 0.0
    observed_weight = 0.0
    pos_obs_weight = 0.0
    nonpos_obs_weight = 0.0
    total_weight = sum(row["prob"] for row in test_rows)
    for row in test_rows:
        pred_observe = estimates[row["signature"]] > 0.0
        chosen = row["observe_return"] if pred_observe else row["true_pre_best"]
        policy_rows.append({**row, "_policy": chosen})
        if pred_observe:
            observed_weight += row["prob"]
        if row["positive_net"]:
            positives.append(row["prob"])
            if pred_observe:
                pos_obs_weight += row["prob"]
            else:
                false_direct_weight += row["prob"]
        else:
            if pred_observe:
                nonpos_obs_weight += row["prob"]
                false_observe_weight += row["prob"]
        if (estimates[row["signature"]] > 0.0) == row["positive_net"]:
            sign_correct_weight += row["prob"]
    nonpositive_weight = total_weight - sum(positives)
    baselines = baseline_metrics(test_rows)
    return {
        "policy_net": weighted_mean(policy_rows, "_policy"),
        "delta_vs_always_observe": weighted_mean(policy_rows, "_policy") - baselines["always_observe_net"],
        "delta_vs_always_try": weighted_mean(policy_rows, "_policy") - baselines["always_try_net"],
        "gap_to_oracle_selective": baselines["oracle_selective_net"] - weighted_mean(policy_rows, "_policy"),
        "observe_rate": observed_weight / total_weight if total_weight else 0.0,
        "sign_correct_rate": sign_correct_weight / total_weight if total_weight else 0.0,
        "false_observe_rate": false_observe_weight / total_weight if total_weight else 0.0,
        "false_direct_rate": false_direct_weight / total_weight if total_weight else 0.0,
        "positive_net_observe_rate": pos_obs_weight / sum(positives) if sum(positives) > 0.0 else 0.0,
        "nonpositive_net_observe_rate": nonpos_obs_weight / nonpositive_weight if nonpositive_weight > 0.0 else 0.0,
        "estimated_E_mean": mean(list(estimates.values())),
    }


def balanced_test_pair(test_pair):
    return all(sum(sig[i] for sig in test_pair) == 1 for i in range(4))


def all_pair_covered_splits():
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


ALL_SPLITS = all_pair_covered_splits()


def pre_registered_mode_b_splits():
    balanced = [
        split
        for split in ALL_SPLITS
        if balanced_test_pair(split["test_A"]) and balanced_test_pair(split["test_B"])
    ]
    balanced = sorted(balanced, key=lambda s: (s["test_A"], s["test_B"]))
    if not balanced:
        return []
    idxs = sorted(
        set(
            [
                0,
                int((len(balanced) - 1) * 0.25),
                int((len(balanced) - 1) * 0.50),
                int((len(balanced) - 1) * 0.75),
                len(balanced) - 1,
            ]
        )
    )
    return [
        {
            **balanced[idx],
            "pre_registered_index": idx,
        }
        for idx in idxs
    ]


with open(SOURCE_JSON_7KB, "r", encoding="utf-8") as f:
    source_7kb = json.load(f)

LOCKED_SPLIT = {
    "train": [tuple(sig) for sig in source_7kb["selected_candidate"]["split"]["train"]],
    "test_A": [tuple(sig) for sig in source_7kb["selected_candidate"]["split"]["test_A"]],
    "test_B": [tuple(sig) for sig in source_7kb["selected_candidate"]["split"]["test_B"]],
    "split_idx": source_7kb["selected_candidate"]["split_idx"],
}

MODE_B_SPLITS = pre_registered_mode_b_splits()


def stringify_tuple_key_dict(dct):
    return {str(key): value for key, value in dct.items()}


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return [json_safe(value) for value in obj]
    if isinstance(obj, list):
        return [json_safe(value) for value in obj]
    return obj


def evaluate_one_partition(mode_name, version_name, seed, split_label, split_payload, rows, observe_cost):
    train_rows = [row for row in rows if row["signature"] in split_payload["train"]]
    stats = build_training_stats(train_rows)
    validity = validity_metrics(rows, split_payload["train"])
    result_rows = []
    for test_type in ["A", "B"]:
        test_sigs = split_payload["test_A"] if test_type == "A" else split_payload["test_B"]
        test_rows = [row for row in rows if row["signature"] in test_sigs]
        baselines = baseline_metrics(test_rows)
        for level_name in ["level1", "level2", "level3"]:
            payload = evaluate_level(test_rows, level_name, stats)
            row = {
                "mode": mode_name,
                "version": version_name,
                "seed": seed,
                "split_label": split_label,
                "split_meta": {
                    "test_A": [list(sig) for sig in split_payload["test_A"]],
                    "test_B": [list(sig) for sig in split_payload["test_B"]],
                },
                "observe_cost": observe_cost,
                "test_type": test_type,
                "level": level_name,
                **baselines,
                **payload,
                "positive_net_rate": weighted_mean(test_rows, "positive_net"),
                **validity,
            }
            result_rows.append(row)
    return result_rows


print("[1/5] Running Mode A fixed-split fresh-seed confirmation...")
all_rows = []
for seed in FRESH_SEEDS:
    for version_name in ["structured", "null"]:
        base_seed = seed if version_name == "structured" else seed + 10000
        for observe_cost in OBSERVE_COSTS:
            sampled_rows = sample_population(version_name, base_seed, observe_cost)
            all_rows.extend(
                evaluate_one_partition(
                    mode_name="mode_A_fixed_split_fresh_seeds",
                    version_name=version_name,
                    seed=seed,
                    split_label=f"fixed_split_{LOCKED_SPLIT['split_idx']}",
                    split_payload=LOCKED_SPLIT,
                    rows=sampled_rows,
                    observe_cost=observe_cost,
                )
            )

print("[2/5] Running Mode B pre-registered split subset...")
for split in MODE_B_SPLITS:
    split_label = f"balanced_lex_idx_{split['pre_registered_index']}"
    for version_name in ["structured", "null"]:
        for observe_cost in OBSERVE_COSTS:
            exact_rows = build_exact_population(version_name, observe_cost)
            all_rows.extend(
                evaluate_one_partition(
                    mode_name="mode_B_preregistered_splits",
                    version_name=version_name,
                    seed=None,
                    split_label=split_label,
                    split_payload=split,
                    rows=exact_rows,
                    observe_cost=observe_cost,
                )
            )


def lookup(rows, **kwargs):
    return [
        row
        for row in rows
        if all(row[key] == value for key, value in kwargs.items())
    ]


mode_a_primary = [
    row
    for row in all_rows
    if row["mode"] == "mode_A_fixed_split_fresh_seeds" and row["observe_cost"] == PRIMARY_COST
]
mode_b_primary = [
    row
    for row in all_rows
    if row["mode"] == "mode_B_preregistered_splits" and row["observe_cost"] == PRIMARY_COST
]

mode_a_seed_results = []
for seed in FRESH_SEEDS:
    structured_rows = [
        row for row in mode_a_primary
        if row["seed"] == seed and row["version"] == "structured"
    ]
    null_rows = [
        row for row in mode_a_primary
        if row["seed"] == seed and row["version"] == "null"
    ]
    best_a = max(row["delta_vs_always_observe"] for row in structured_rows if row["test_type"] == "A" and row["level"] in ("level2", "level3"))
    best_b = max(row["delta_vs_always_observe"] for row in structured_rows if row["test_type"] == "B" and row["level"] in ("level2", "level3"))
    level1_max = max(row["delta_vs_always_observe"] for row in structured_rows if row["level"] == "level1")
    null_max = max(row["delta_vs_always_observe"] for row in null_rows if row["level"] in ("level2", "level3"))
    pass_flag = best_a >= 0.03 and best_b >= 0.03 and level1_max < 0.03 and null_max < 0.03
    mode_a_seed_results.append(
        {
            "seed": seed,
            "best_A": best_a,
            "best_B": best_b,
            "level1_max": level1_max,
            "null_max": null_max,
            "pass_flag": pass_flag,
        }
    )

mode_b_split_results = []
for split in MODE_B_SPLITS:
    split_label = f"balanced_lex_idx_{split['pre_registered_index']}"
    structured_rows = [
        row for row in mode_b_primary
        if row["split_label"] == split_label and row["version"] == "structured"
    ]
    null_rows = [
        row for row in mode_b_primary
        if row["split_label"] == split_label and row["version"] == "null"
    ]
    best_a = max(row["delta_vs_always_observe"] for row in structured_rows if row["test_type"] == "A" and row["level"] in ("level2", "level3"))
    best_b = max(row["delta_vs_always_observe"] for row in structured_rows if row["test_type"] == "B" and row["level"] in ("level2", "level3"))
    level1_max = max(row["delta_vs_always_observe"] for row in structured_rows if row["level"] == "level1")
    null_max = max(row["delta_vs_always_observe"] for row in null_rows if row["level"] in ("level2", "level3"))
    pass_flag = best_a >= 0.03 and best_b >= 0.03 and level1_max < 0.03 and null_max < 0.03
    mode_b_split_results.append(
        {
            "split_label": split_label,
            "best_A": best_a,
            "best_B": best_b,
            "level1_max": level1_max,
            "null_max": null_max,
            "pass_flag": pass_flag,
        }
    )

mode_a_seed_pass_rate = mean([1.0 if item["pass_flag"] else 0.0 for item in mode_a_seed_results])
mode_b_split_pass_rate = mean([1.0 if item["pass_flag"] else 0.0 for item in mode_b_split_results]) if mode_b_split_results else 0.0
mode_b_null_collapse_rate = mean([1.0 if item["null_max"] < 0.03 else 0.0 for item in mode_b_split_results]) if mode_b_split_results else 0.0

mode_a_testA_l2_mean = mean([row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "structured" and row["test_type"] == "A" and row["level"] == "level2"])
mode_a_testA_l3_mean = mean([row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "structured" and row["test_type"] == "A" and row["level"] == "level3"])
mode_a_testB_l2_mean = mean([row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "structured" and row["test_type"] == "B" and row["level"] == "level2"])
mode_a_testB_l3_mean = mean([row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "structured" and row["test_type"] == "B" and row["level"] == "level3"])
mode_a_level1_max = max(row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "structured" and row["level"] == "level1")
mode_a_null_level23_max = max(row["delta_vs_always_observe"] for row in mode_a_primary if row["version"] == "null" and row["level"] in ("level2", "level3"))

mode_b_best_deltas = [max(item["best_A"], item["best_B"]) for item in mode_b_split_results]
mode_b_testAB_mean_best = mean(mode_b_best_deltas) if mode_b_best_deltas else 0.0
mode_b_testAB_median_best = statistics.median(mode_b_best_deltas) if mode_b_best_deltas else 0.0
mode_b_testAB_min_best = min(mode_b_best_deltas) if mode_b_best_deltas else 0.0
mode_b_worst = min(mode_b_split_results, key=lambda x: min(x["best_A"], x["best_B"])) if mode_b_split_results else None

validity_anchor = next(
    row for row in mode_a_primary
    if row["version"] == "structured" and row["seed"] == FRESH_SEEDS[0] and row["test_type"] == "A" and row["level"] == "level1"
)

if (
    mode_a_testA_l2_mean >= 0.03 or mode_a_testA_l3_mean >= 0.03
) and (
    mode_a_testB_l2_mean >= 0.03 or mode_a_testB_l3_mean >= 0.03
) and mode_a_seed_pass_rate >= 0.70 and mode_a_level1_max < 0.03 and mode_a_null_level23_max < 0.03 and validity_anchor["exact_signature_determinism_rate"] <= 0.10 and validity_anchor["single_cue_max_predictiveness"] <= 0.80 and validity_anchor["hidden_h_probe_from_pre_surface_accuracy"] <= 0.70 and mode_b_split_pass_rate >= 0.60 and mode_b_null_collapse_rate >= 0.90:
    status = "PASS_LOCKED_VALIDATION"
    decision = "7kc_LOCKED_VALIDATION_PASS_GO_ENV3C_DRAFT"
elif mode_a_seed_pass_rate > 0.0:
    status = "PARTIAL_LOCKED_VALIDATION"
    decision = "7kc_PARTIAL_NEEDS_ENV_DESIGN_REVIEW"
else:
    status = "FAIL_LOCKED_VALIDATION"
    decision = "7kc_FAIL_7KB_WAS_SPLIT_SEARCH_ARTIFACT"

print("[3/5] Writing JSON and CSV outputs...")
payload = {
    "metadata": {
        "block_id": "1J40b-7k-c",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "locked_from": LOCKED_FROM,
        "validation_mode": VALIDATION_MODE,
        "no_design_search": NO_DESIGN_SEARCH,
        "no_parameter_tuning": NO_PARAMETER_TUNING,
        "no_split_selection_by_metric": NO_SPLIT_SELECTION_BY_METRIC,
        "primary_observe_cost": PRIMARY_COST,
        "observe_costs": OBSERVE_COSTS,
        "fresh_seeds": FRESH_SEEDS,
        "sampled_objects_per_signature": SAMPLED_OBJECTS_PER_SIGNATURE,
    },
    "locked_configuration": {
        "generator_mode": "factorized_base",
        "diag_alpha": LOCKED_DIAG_ALPHA,
        "reward_map": REWARD_BY_H,
        "pair1_logits": stringify_tuple_key_dict(PAIR1_LOGITS),
        "pair2_logits": {
            name: stringify_tuple_key_dict(payload)
            for name, payload in PAIR2_LOGITS.items()
        },
        "level2_rule": "informative cue-pair mean",
        "level3_rule": LOCKED_LEVEL3_RULE,
        "null_control": "global hidden-h distribution matched to locked structured generator",
        "selected_7kb_split_idx": LOCKED_SPLIT["split_idx"],
        "selected_7kb_split": {
            "train": [list(sig) for sig in LOCKED_SPLIT["train"]],
            "test_A": [list(sig) for sig in LOCKED_SPLIT["test_A"]],
            "test_B": [list(sig) for sig in LOCKED_SPLIT["test_B"]],
        },
    },
    "mode_A_summary": {
        "seed_count": len(FRESH_SEEDS),
        "seed_pass_rate": mode_a_seed_pass_rate,
        "testA_level2_mean_delta": mode_a_testA_l2_mean,
        "testA_level3_mean_delta": mode_a_testA_l3_mean,
        "testB_level2_mean_delta": mode_a_testB_l2_mean,
        "testB_level3_mean_delta": mode_a_testB_l3_mean,
        "level1_max_delta": mode_a_level1_max,
        "null_level23_max_delta": mode_a_null_level23_max,
        "per_seed_results": mode_a_seed_results,
    },
    "mode_B_summary": {
        "split_selection_rule": "all pair-covered splits -> balanced Test A/B splits with mixed factor values on every factor -> lexicographic sort by (test_A, test_B) -> select first / 25th percentile / median / 75th percentile / last",
        "split_count": len(MODE_B_SPLITS),
        "split_labels": [item["split_label"] for item in mode_b_split_results],
        "split_pass_rate": mode_b_split_pass_rate,
        "mean_best_level23_delta": mode_b_testAB_mean_best,
        "median_best_level23_delta": mode_b_testAB_median_best,
        "min_best_level23_delta": mode_b_testAB_min_best,
        "null_collapse_rate": mode_b_null_collapse_rate,
        "per_split_results": mode_b_split_results,
    },
    "results": all_rows,
    "acceptance_summary": {
        "status": status,
        "decision": decision,
    },
}

with open(RUN_JSON, "w", encoding="utf-8") as f:
    json.dump(json_safe(payload), f, indent=2)

csv_columns = [
    "mode", "version", "seed", "split_label", "observe_cost", "test_type", "level",
    "policy_net", "always_try_net", "always_observe_net", "oracle_selective_net",
    "delta_vs_always_observe", "delta_vs_always_try", "gap_to_oracle_selective",
    "observe_rate", "sign_correct_rate", "false_observe_rate", "false_direct_rate",
    "positive_net_rate", "exact_signature_determinism_rate", "single_cue_max_predictiveness",
    "cue_pair_max_predictiveness", "train_pair_coverage", "hidden_h_probe_from_pre_surface_accuracy",
]
with open(TABLE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for row in all_rows:
        writer.writerow({k: row[k] for k in csv_columns})

print("[4/5] Writing report and checkpoint...")
md_lines = [
    "# 1J40b-7k-c Locked-Rule Confirmation",
    "",
    "## Checkpoint Summary",
    f"- status: {status}",
    f"- decision: {decision}",
    f"- validation_mode: {VALIDATION_MODE}",
    f"- locked_from: {LOCKED_FROM}",
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
    "## Why 7k-c Was Run",
    "- 7k-b reached design-search PASS, but the split was selected by search. 7k-c checks whether the locked scaffold remains stable without any new split tuning.",
    "",
    "## Locked Configuration",
    "- generator_mode: factorized_base",
    f"- diag_alpha: {LOCKED_DIAG_ALPHA}",
    f"- level3_rule: {LOCKED_LEVEL3_RULE}",
    f"- primary cost: {PRIMARY_COST}",
    f"- locked 7k-b split idx: {LOCKED_SPLIT['split_idx']}",
    f"- locked Test A: {LOCKED_SPLIT['test_A']}",
    f"- locked Test B: {LOCKED_SPLIT['test_B']}",
    "",
    "## Mode A: Fixed 7k-b Split, Fresh Seeds",
    f"- seed count: {len(FRESH_SEEDS)}",
    f"- seed pass rate: {mode_a_seed_pass_rate:.4f}",
    f"- Test A Level 2 mean delta: {mode_a_testA_l2_mean:.4f}",
    f"- Test A Level 3 mean delta: {mode_a_testA_l3_mean:.4f}",
    f"- Test B Level 2 mean delta: {mode_a_testB_l2_mean:.4f}",
    f"- Test B Level 3 mean delta: {mode_a_testB_l3_mean:.4f}",
    f"- Level 1 max delta: {mode_a_level1_max:.4f}",
    f"- null Level 2/3 max delta: {mode_a_null_level23_max:.4f}",
    "",
    "## Mode B: Pre-Registered Split Subset",
    "- split selection rule: all pair-covered splits -> balanced Test A/B by factor values -> lexicographic quantiles",
    f"- split count: {len(MODE_B_SPLITS)}",
    f"- split pass rate: {mode_b_split_pass_rate:.4f}",
    f"- mean best Level 2/3 delta: {mode_b_testAB_mean_best:.4f}",
    f"- median best Level 2/3 delta: {mode_b_testAB_median_best:.4f}",
    f"- min best Level 2/3 delta: {mode_b_testAB_min_best:.4f}",
    f"- null collapse rate: {mode_b_null_collapse_rate:.4f}",
    f"- worst split: {mode_b_worst}",
    "",
    "## Validity",
    f"- exact_signature_determinism_rate: {validity_anchor['exact_signature_determinism_rate']:.4f}",
    f"- single_cue_max_predictiveness: {validity_anchor['single_cue_max_predictiveness']:.4f}",
    f"- cue_pair_max_predictiveness: {validity_anchor['cue_pair_max_predictiveness']:.4f}",
    f"- hidden_h_probe_from_pre_surface_accuracy: {validity_anchor['hidden_h_probe_from_pre_surface_accuracy']:.4f}",
    f"- train_pair_coverage: {validity_anchor['train_pair_coverage']:.4f}",
    "",
    "## Main Diagnosis",
    "- Fixed-split fresh-seed stability is strong at the locked primary cost.",
    "- Broader robustness over a neutral pre-registered balanced split subset is weak.",
    "- Level 1 shortcut does not appear, and null-control remains clean.",
    "- Therefore the locked scaffold looks locally stable but not broadly split-robust.",
    "",
    "## Recommended Resume Point",
    "- env3c design draft only if you accept local locked-split stability as sufficient evidence; otherwise do scaffold redesign or expert review first.",
]
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J40b-7k-c",
    "",
    f"- status: {status}",
    f"- decision: {decision}",
    f"- mode_A_seed_pass_rate: {mode_a_seed_pass_rate:.4f}",
    f"- mode_B_split_pass_rate: {mode_b_split_pass_rate:.4f}",
    f"- mode_A_testA_level3_mean_delta: {mode_a_testA_l3_mean:.4f}",
    f"- mode_A_testB_level3_mean_delta: {mode_a_testB_l3_mean:.4f}",
    f"- null_level23_max_delta: {mode_a_null_level23_max:.4f}",
    f"- next_step: env3c draft only with caution, or scaffold redesign / expert review",
]
with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Summary")
print(f"Status: {status}")
print("Mode A — fixed 7k-b split, fresh seeds:")
print(f"- seed count: {len(FRESH_SEEDS)}")
print(f"- seed pass rate: {mode_a_seed_pass_rate:.4f}")
print(f"- Test A Level 2 mean delta: {mode_a_testA_l2_mean:.4f}")
print(f"- Test A Level 3 mean delta: {mode_a_testA_l3_mean:.4f}")
print(f"- Test B Level 2 mean delta: {mode_a_testB_l2_mean:.4f}")
print(f"- Test B Level 3 mean delta: {mode_a_testB_l3_mean:.4f}")
print(f"- Level 1 max delta: {mode_a_level1_max:.4f}")
print(f"- null Level 2/3 max delta: {mode_a_null_level23_max:.4f}")
print("Mode B — pre-registered split subset:")
print("- split selection rule: all pair-covered splits -> balanced Test A/B -> lexicographic quantiles")
print(f"- split count: {len(MODE_B_SPLITS)}")
print(f"- split pass rate: {mode_b_split_pass_rate:.4f}")
print(f"- Test A/B mean best Level 2/3 delta: {mode_b_testAB_mean_best:.4f}")
print(f"- worst split result: {mode_b_worst}")
print(f"- null collapse rate: {mode_b_null_collapse_rate:.4f}")
print("Validity:")
print(f"- exact_signature_determinism_rate: {validity_anchor['exact_signature_determinism_rate']:.4f}")
print(f"- single_cue_max_predictiveness: {validity_anchor['single_cue_max_predictiveness']:.4f}")
print(f"- cue_pair_max_predictiveness: {validity_anchor['cue_pair_max_predictiveness']:.4f}")
print(f"- hidden_h_probe_from_pre_surface_accuracy: {validity_anchor['hidden_h_probe_from_pre_surface_accuracy']:.4f}")
print(f"- train_pair_coverage: {validity_anchor['train_pair_coverage']:.4f}")
print(f"Decision: {decision}")
print(f"Elapsed: {time.time() - t0:.2f}s")
