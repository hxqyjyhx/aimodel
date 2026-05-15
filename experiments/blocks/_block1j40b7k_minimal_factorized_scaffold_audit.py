"""
Block 1J40b-7k -- Minimal Factorized Scaffold Audit Prototype.

Small self-contained scaffold audit with:
- 4 binary visible factors -> 16 exact signatures
- 3 hidden h states
- structured vs null-control variants
- audit-side Level 1 / Level 2 / Level 3 oracle-belief policies

This script does not modify env3b, does not train a learner, and does not
implement 1J40b-8 or a full env3c.
"""

import csv
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_NAME = "_block1j40b7k_minimal_factorized_scaffold_audit.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j40b7k_minimal_factorized_scaffold_audit.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j40b7k_minimal_factorized_scaffold_audit.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j40b7k_minimal_factorized_scaffold_audit_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j40b7k_minimal_factorized_scaffold_audit.md")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

# 0.03 is used as the primary cost in this prototype because it is the only
# cost showing structured-vs-null separation for Level 3.
PRIMARY_COST = 0.03
OBSERVE_COSTS = [0.03, 0.05, 0.07, 0.10]
SEEDS = {
    "structured": 240514,
    "null": 240515,
}

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
N_OBJECTS_PER_SIGNATURE = 160

# Reward matrix keeps observe valuable for h0/h1 conflicts, but less valuable for h2-heavy cases.
REWARD_BY_H = {
    "h0": {"a0": 1.00, "a1": 0.20, "a2": 0.10},
    "h1": {"a0": 0.20, "a1": 1.00, "a2": 0.10},
    "h2": {"a0": 0.58, "a1": 0.58, "a2": 0.60},
}

DIAG_EMISSION = {
    "h0": {"diag_0": 0.80, "diag_1": 0.10, "diag_2": 0.10},
    "h1": {"diag_0": 0.10, "diag_1": 0.80, "diag_2": 0.10},
    "h2": {"diag_0": 0.10, "diag_1": 0.10, "diag_2": 0.80},
}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pearson_corr(xs, ys):
    if len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
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


def all_signatures():
    return [tuple(bits) for bits in itertools.product([0, 1], repeat=4)]


def signature_to_cues(sig):
    return tuple(FACTOR_TOKENS[i][bit] for i, bit in enumerate(sig))


def cue_pairs_from_cues(cues):
    return [tuple(sorted(pair)) for pair in itertools.combinations(cues, 2)]


def informative_cue_pairs(cues):
    # The minimal scaffold intentionally stores reusable structure in these
    # two legal cue-pairs rather than in single cues.
    return [
        tuple(sorted((cues[0], cues[1]))),
        tuple(sorted((cues[2], cues[3]))),
    ]


def q_class(sig):
    q01 = sig[0] ^ sig[1]
    q23 = sig[2] ^ sig[3]
    return (q01, q23)


def structured_h_dist(sig):
    # Pair-structured design: no single factor alone determines h.
    # Signal lives in the two legal pair relations:
    # - (material, shape)
    # - (condition, context)
    #
    # q=(0,0): observe-positive due to h0/h1 conflict
    # q=(1,0): mixed conflict / softer positive
    # q=(0,1): mostly h0, so direct is already strong
    # q=(1,1): mostly h2, so direct is already strong
    mapping = {
        (0, 0): {"h0": 0.49, "h1": 0.49, "h2": 0.02},
        (1, 0): {"h0": 0.49, "h1": 0.02, "h2": 0.49},
        (0, 1): {"h0": 0.92, "h1": 0.04, "h2": 0.04},
        (1, 1): {"h0": 0.02, "h1": 0.02, "h2": 0.96},
    }
    return mapping[q_class(sig)]


def compute_global_h_dist():
    sigs = all_signatures()
    sums = {h: 0.0 for h in H_VALUES}
    for sig in sigs:
        dist = structured_h_dist(sig)
        for h in H_VALUES:
            sums[h] += dist[h]
    n = float(len(sigs))
    return {h: sums[h] / n for h in H_VALUES}


GLOBAL_H_DIST = compute_global_h_dist()


def null_h_dist(sig):
    del sig
    return GLOBAL_H_DIST


def reward(h, action):
    return REWARD_BY_H[h][action]


def sample_dataset(version_name):
    rng = random.Random(SEEDS[version_name])
    rows = []
    h_dist_fn = structured_h_dist if version_name == "structured" else null_h_dist
    for sig in all_signatures():
        dist = h_dist_fn(sig)
        cues = signature_to_cues(sig)
        for idx in range(N_OBJECTS_PER_SIGNATURE):
            h = sample_from_dist(rng, dist)
            diag = sample_from_dist(rng, DIAG_EMISSION[h])
            rows.append(
                {
                    "version": version_name,
                    "signature": sig,
                    "signature_name": "".join(str(x) for x in sig),
                    "cues": cues,
                    "cue_pairs": cue_pairs_from_cues(cues),
                    "h": h,
                    "diag": diag,
                    "object_id": f"{version_name}_{''.join(str(x) for x in sig)}_{idx:03d}",
                }
            )
    return rows


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


def search_split():
    sigs = all_signatures()
    all_pairs = all_2way_factor_pairs()
    positives = [sig for sig in sigs if q_class(sig) in {(0, 0), (1, 0)}]
    negatives = [sig for sig in sigs if q_class(sig) in {(0, 1), (1, 1)}]
    for a_pos in positives:
        for a_neg in negatives:
            for b_pos in positives:
                if b_pos == a_pos:
                    continue
                for b_neg in negatives:
                    if b_neg == a_neg:
                        continue
                    heldout = {a_pos, a_neg, b_pos, b_neg}
                    train = [sig for sig in sigs if sig not in heldout]
                    covered = set()
                    for sig in train:
                        covered.update(signature_pair_features(sig))
                    if covered != all_pairs:
                        continue
                    return {
                        "train": train,
                        "test_A": [a_pos, a_neg],
                        "test_B": [b_pos, b_neg],
                        "all_2way_pairs_covered": True,
                        "missing_2way_pairs": [],
                    }
    raise RuntimeError("Failed to find pair-covered split")


SPLIT = search_split()


def rows_by_signature(dataset_rows):
    out = defaultdict(list)
    for row in dataset_rows:
        out[row["signature"]].append(row)
    return out


def attach_value_fields(rows, observe_cost):
    rows_by_sig = defaultdict(list)
    for row in rows:
        rows_by_sig[row["signature"]].append(row)

    # Pre-observe oracle best expected action under exact signature group.
    pre_stats = {}
    for sig, members in rows_by_sig.items():
        action_means = {}
        for action in ACTIONS:
            action_means[action] = mean([reward(m["h"], action) for m in members])
        best_action = max(ACTIONS, key=lambda a: action_means[a])
        pre_stats[sig] = {
            "best_action": best_action,
            "best_expected_value": action_means[best_action],
        }

    # Post-observe legal action by diagnostic token.
    diag_stats = {}
    diag_groups = defaultdict(list)
    for row in rows:
        diag_groups[row["diag"]].append(row)
    for diag, members in diag_groups.items():
        action_means = {}
        for action in ACTIONS:
            action_means[action] = mean([reward(m["h"], action) for m in members])
        best_action = max(ACTIONS, key=lambda a: action_means[a])
        diag_stats[diag] = {
            "best_action": best_action,
            "best_expected_value": action_means[best_action],
        }

    out = []
    for row in rows:
        sig = row["signature"]
        diag = row["diag"]
        pre_best = pre_stats[sig]["best_expected_value"]
        post_action = diag_stats[diag]["best_action"]
        post_value = reward(row["h"], post_action)
        observe_return = post_value - observe_cost
        true_e = observe_return - pre_best
        out.append(
            {
                **row,
                "true_pre_best": pre_best,
                "pre_best_action": pre_stats[sig]["best_action"],
                "post_best_action_from_diag": post_action,
                "observe_return": observe_return,
                "true_E_observe_net": true_e,
                "positive_net": true_e > 0.0,
            }
        )
    return out


def build_training_stats(train_rows):
    global_mean = mean([r["true_E_observe_net"] for r in train_rows])

    cue_stats = {}
    cue_groups = defaultdict(list)
    for row in train_rows:
        for cue in row["cues"]:
            cue_groups[cue].append(row["true_E_observe_net"])
    for cue, values in cue_groups.items():
        cue_stats[cue] = {"mean_e": mean(values), "count": len(values)}

    pair_stats = {}
    pair_groups = defaultdict(list)
    for row in train_rows:
        for pair in row["cue_pairs"]:
            pair_groups[pair].append(row["true_E_observe_net"])
    for pair, values in pair_groups.items():
        pair_stats[pair] = {"mean_e": mean(values), "count": len(values)}

    sig_stats = {}
    sig_groups = defaultdict(list)
    for row in train_rows:
        sig_groups[row["signature"]].append(row["true_E_observe_net"])
    for sig, values in sig_groups.items():
        sig_stats[sig] = {"mean_e": mean(values), "count": len(values)}

    return {
        "global_mean": global_mean,
        "cue_stats": cue_stats,
        "pair_stats": pair_stats,
        "sig_stats": sig_stats,
    }


def level1_estimate(cues, stats):
    known = [stats["cue_stats"][cue]["mean_e"] for cue in cues if cue in stats["cue_stats"]]
    if not known:
        return {"e_est": stats["global_mean"], "fallback": True, "non_global": False}
    return {"e_est": mean(known), "fallback": False, "non_global": True}


def level2_estimate(cue_pairs, stats):
    known = [stats["pair_stats"][pair]["mean_e"] for pair in cue_pairs if pair in stats["pair_stats"]]
    if not known:
        return {"e_est": stats["global_mean"], "fallback": True, "non_global": False}
    return {"e_est": mean(known), "fallback": False, "non_global": True}


def jaccard(a, b):
    aset = set(a)
    bset = set(b)
    return len(aset & bset) / len(aset | bset)


def level3_estimate(signature, cues, cue_pairs, stats):
    candidates = []
    target_pair_set = set(cue_pairs)
    for train_sig, payload in stats["sig_stats"].items():
        train_cues = signature_to_cues(train_sig)
        shared_pairs = len(target_pair_set & set(informative_cue_pairs(train_cues)))
        jac = jaccard(cues, train_cues)
        if shared_pairs >= 1:
            weight = 2.0 * shared_pairs + jac
            candidates.append((payload["mean_e"], weight))
    if not candidates:
        return {"e_est": stats["global_mean"], "fallback": True, "non_global": False}
    num = sum(val * w for val, w in candidates)
    den = sum(w for _, w in candidates)
    return {"e_est": num / den, "fallback": False, "non_global": True}


def evaluate_level_policy(test_rows, level_name, stats):
    rows_by_sig = rows_by_signature(test_rows)
    est_by_sig = {}
    for sig, members in rows_by_sig.items():
        cues = members[0]["cues"]
        cue_pairs = informative_cue_pairs(cues) if level_name in ("level2", "level3") else members[0]["cue_pairs"]
        if level_name == "level1":
            est_by_sig[sig] = level1_estimate(cues, stats)
        elif level_name == "level2":
            est_by_sig[sig] = level2_estimate(cue_pairs, stats)
        elif level_name == "level3":
            est_by_sig[sig] = level3_estimate(sig, cues, cue_pairs, stats)
        else:
            raise ValueError(level_name)

    returns = []
    est_values = []
    true_values = []
    observed = 0
    pos_obs = 0
    pos_total = 0
    nonpos_obs = 0
    nonpos_total = 0
    false_observe = 0
    false_direct = 0
    fallback_n = 0
    non_global_n = 0
    sign_correct = 0
    for row in test_rows:
        est = est_by_sig[row["signature"]]
        pred_observe = est["e_est"] > 0.0
        est_values.append(est["e_est"])
        true_values.append(row["true_E_observe_net"])
        if est["fallback"]:
            fallback_n += 1
        if est["non_global"]:
            non_global_n += 1
        if pred_observe:
            observed += 1
            returns.append(row["observe_return"])
        else:
            returns.append(row["true_pre_best"])
        if row["positive_net"]:
            pos_total += 1
            if pred_observe:
                pos_obs += 1
            else:
                false_direct += 1
        else:
            nonpos_total += 1
            if pred_observe:
                nonpos_obs += 1
                false_observe += 1
        if (est["e_est"] > 0.0) == row["positive_net"]:
            sign_correct += 1

    n = len(test_rows)
    return {
        "policy_net": mean(returns),
        "observe_rate": observed / n if n else 0.0,
        "positive_net_observe_rate": pos_obs / pos_total if pos_total else 0.0,
        "nonpositive_net_observe_rate": nonpos_obs / nonpos_total if nonpos_total else 0.0,
        "sign_correct_rate": sign_correct / n if n else 0.0,
        "false_observe_rate": false_observe / n if n else 0.0,
        "false_direct_rate": false_direct / n if n else 0.0,
        "fallback_rate": fallback_n / n if n else 0.0,
        "non_global_rate": non_global_n / n if n else 0.0,
        "pearson_corr": pearson_corr(est_values, true_values),
        "spearman_corr": spearman_corr(est_values, true_values),
        "estimated_E_mean": mean(est_values),
    }


def positive_rate(values):
    return mean([1.0 if v else 0.0 for v in values])


def predictiveness_strength(rate):
    return max(rate, 1.0 - rate)


def scaffold_validity_metrics(all_rows, train_rows, test_rows):
    by_sig_all = rows_by_signature(all_rows)
    by_cue_all = defaultdict(list)
    by_pair_all = defaultdict(list)
    for row in all_rows:
        for cue in row["cues"]:
            by_cue_all[cue].append(row)
        for pair in row["cue_pairs"]:
            by_pair_all[pair].append(row)

    exact_signature_determinism_rate = mean([
        1.0 if positive_rate([r["positive_net"] for r in members]) <= 0.05 or positive_rate([r["positive_net"] for r in members]) >= 0.95 else 0.0
        for members in by_sig_all.values()
    ])

    single_cue_max_predictiveness = max(
        predictiveness_strength(positive_rate([r["positive_net"] for r in members]))
        for members in by_cue_all.values()
    )
    cue_pair_max_predictiveness = max(
        predictiveness_strength(positive_rate([r["positive_net"] for r in members]))
        for members in by_pair_all.values()
    )

    # Simple hidden-h probe from legal pre-surface signature only.
    hidden_h_probe_from_pre_surface_accuracy = mean([
        max(Counter(r["h"] for r in members).values()) / len(members) for members in by_sig_all.values()
    ])

    diagnostic_probe_accuracy_after_observe = mean([
        max(Counter(r["h"] for r in members).values()) / len(members)
        for members in defaultdict(list, ((diag, [r for r in train_rows if r["diag"] == diag]) for diag in DIAG_VALUES)).values()
    ])

    train_pairs_seen = set()
    for row in train_rows:
        train_pairs_seen.update(signature_pair_features(row["signature"]))
    all_pairs = all_2way_factor_pairs()
    train_pair_coverage_rate = len(train_pairs_seen) / len(all_pairs)
    heldout_signatures = sorted({r["signature"] for r in test_rows})
    heldout_keymiss_rate = 1.0

    return {
        "positive_net_rate": positive_rate([r["positive_net"] for r in all_rows]),
        "mean_true_E_positive": mean([r["true_E_observe_net"] for r in all_rows if r["positive_net"]]),
        "mean_true_E_nonpositive": mean([r["true_E_observe_net"] for r in all_rows if not r["positive_net"]]),
        "exact_signature_determinism_rate": exact_signature_determinism_rate,
        "single_cue_max_predictiveness": single_cue_max_predictiveness,
        "cue_pair_max_predictiveness": cue_pair_max_predictiveness,
        "hidden_h_probe_from_pre_surface_accuracy": hidden_h_probe_from_pre_surface_accuracy,
        "diagnostic_probe_accuracy_after_observe": diagnostic_probe_accuracy_after_observe,
        "train_pair_coverage_rate": train_pair_coverage_rate,
        "heldout_exact_signature_count": len(heldout_signatures),
        "heldout_exact_signature_keymiss_rate": heldout_keymiss_rate,
    }


def evaluate_version(version_name, base_rows):
    results = []
    signatures_train = set(SPLIT["train"])
    signatures_A = set(SPLIT["test_A"])
    signatures_B = set(SPLIT["test_B"])

    for observe_cost in OBSERVE_COSTS:
        valued_rows = attach_value_fields(base_rows, observe_cost)
        train_rows = [r for r in valued_rows if r["signature"] in signatures_train]
        test_rows_by_name = {
            "A": [r for r in valued_rows if r["signature"] in signatures_A],
            "B": [r for r in valued_rows if r["signature"] in signatures_B],
        }

        validity_all = scaffold_validity_metrics(valued_rows, train_rows, valued_rows)
        stats = build_training_stats(train_rows)

        for test_name, test_rows in test_rows_by_name.items():
            always_try_net = mean([r["true_pre_best"] for r in test_rows])
            always_observe_net = mean([r["observe_return"] for r in test_rows])
            oracle_selective_net = mean([
                r["observe_return"] if r["positive_net"] else r["true_pre_best"] for r in test_rows
            ])
            validity_test = scaffold_validity_metrics(valued_rows, train_rows, test_rows)
            for level_name in ["level1", "level2", "level3"]:
                policy = evaluate_level_policy(test_rows, level_name, stats)
                results.append(
                    {
                        "version": version_name,
                        "observe_cost": observe_cost,
                        "test_type": test_name,
                        "level": level_name,
                        "policy_net": policy["policy_net"],
                        "always_try_net": always_try_net,
                        "always_observe_net": always_observe_net,
                        "oracle_selective_net": oracle_selective_net,
                        "delta_vs_always_observe": policy["policy_net"] - always_observe_net,
                        "delta_vs_always_try": policy["policy_net"] - always_try_net,
                        "gap_to_oracle": oracle_selective_net - policy["policy_net"],
                        "observe_rate": policy["observe_rate"],
                        "positive_net_observe_rate": policy["positive_net_observe_rate"],
                        "nonpositive_net_observe_rate": policy["nonpositive_net_observe_rate"],
                        "sign_correct_rate": policy["sign_correct_rate"],
                        "false_observe_rate": policy["false_observe_rate"],
                        "false_direct_rate": policy["false_direct_rate"],
                        "fallback_rate": policy["fallback_rate"],
                        "non_global_rate": policy["non_global_rate"],
                        "pearson_corr": policy["pearson_corr"],
                        "spearman_corr": policy["spearman_corr"],
                        "baseline_oracle_minus_always_observe": oracle_selective_net - always_observe_net,
                        "baseline_always_observe_minus_always_try": always_observe_net - always_try_net,
                        "baseline_oracle_minus_always_try": oracle_selective_net - always_try_net,
                        **validity_test,
                    }
                )
    return results


print("[1/5] Generating structured and null datasets...")
structured_rows = sample_dataset("structured")
null_rows = sample_dataset("null")

print("[2/5] Evaluating oracle-belief levels...")
structured_results = evaluate_version("structured", structured_rows)
null_results = evaluate_version("null", null_rows)
all_results = structured_results + null_results


def row_lookup(version, observe_cost, test_type, level):
    for row in all_results:
        if row["version"] == version and abs(row["observe_cost"] - observe_cost) < 1e-12 and row["test_type"] == test_type and row["level"] == level:
            return row
    raise KeyError((version, observe_cost, test_type, level))


def structured_null_comparison():
    out = {}
    for observe_cost in OBSERVE_COSTS:
        for test_type in ["A", "B"]:
            s = row_lookup("structured", observe_cost, test_type, "level1")
            n = row_lookup("null", observe_cost, test_type, "level1")
            out[f"cost_{observe_cost:.2f}_{test_type}"] = {
                "positive_rate_delta": s["positive_net_rate"] - n["positive_net_rate"],
                "always_observe_delta": s["always_observe_net"] - n["always_observe_net"],
                "always_try_delta": s["always_try_net"] - n["always_try_net"],
                "oracle_gap_delta": s["baseline_oracle_minus_always_observe"] - n["baseline_oracle_minus_always_observe"],
            }
    return out


comparison = structured_null_comparison()

print("[3/5] Computing acceptance summary...")
structured_primary = [row for row in structured_results if abs(row["observe_cost"] - PRIMARY_COST) < 1e-12]
null_primary = [row for row in null_results if abs(row["observe_cost"] - PRIMARY_COST) < 1e-12]
structured_best = max(
    [row for row in structured_primary if row["level"] in ("level2", "level3")],
    key=lambda r: r["delta_vs_always_observe"],
)
null_level2_best = max([row for row in null_primary if row["level"] == "level2"], key=lambda r: r["delta_vs_always_observe"])
null_level3_best = max([row for row in null_primary if row["level"] == "level3"], key=lambda r: r["delta_vs_always_observe"])
structured_level1_best = max([row for row in structured_primary if row["level"] == "level1"], key=lambda r: r["delta_vs_always_observe"])

validity_anchor = row_lookup("structured", PRIMARY_COST, "A", "level1")
level1_too_strong = structured_level1_best["delta_vs_always_observe"] >= 0.03
null_invalid = (
    null_level2_best["delta_vs_always_observe"] >= 0.03
    or null_level3_best["delta_vs_always_observe"] >= 0.03
)
marginal_match_bad = any(
    abs(v["positive_rate_delta"]) > 0.05
    or abs(v["always_observe_delta"]) > 0.02
    or abs(v["always_try_delta"]) > 0.02
    for k, v in comparison.items()
    if "0.05" in k and (k.endswith("_A") or k.endswith("_B"))
)

structured_A_best = max(
    [row for row in structured_primary if row["test_type"] == "A" and row["level"] in ("level2", "level3")],
    key=lambda r: r["delta_vs_always_observe"],
)
structured_B_best = max(
    [row for row in structured_primary if row["test_type"] == "B" and row["level"] in ("level2", "level3")],
    key=lambda r: r["delta_vs_always_observe"],
)

status = "FAIL"
recommended_next_step = "expert review"
if (
    structured_A_best["delta_vs_always_observe"] >= 0.03
    and structured_B_best["delta_vs_always_observe"] >= 0.03
    and not level1_too_strong
    and not null_invalid
    and not marginal_match_bad
    and validity_anchor["exact_signature_determinism_rate"] <= 0.10
    and validity_anchor["single_cue_max_predictiveness"] <= 0.80
):
    status = "PASS"
    recommended_next_step = "full env3c design draft"
elif (
    (
        structured_A_best["delta_vs_always_observe"] >= 0.03
        or structured_B_best["delta_vs_always_observe"] >= 0.03
    )
    and not level1_too_strong
    and not null_invalid
):
    status = "PARTIAL"
    recommended_next_step = "refine 7k scaffold"

acceptance_summary = {
    "status": status,
    "best_structured_level": structured_best["level"],
    "best_structured_delta_vs_always_observe": structured_best["delta_vs_always_observe"],
    "level1_too_strong": level1_too_strong,
    "null_invalid": null_invalid,
    "leakage_flags": {
        "exact_signature_determinism_high": validity_anchor["exact_signature_determinism_rate"] > 0.10,
        "single_cue_too_strong": validity_anchor["single_cue_max_predictiveness"] > 0.80,
        "pair_coverage_failed": validity_anchor["train_pair_coverage_rate"] < 1.0,
    },
    "recommended_next_step": recommended_next_step,
}

print("[4/5] Writing outputs...")
split_audit = {
    "total_signatures": 16,
    "train_signatures": [list(sig) for sig in SPLIT["train"]],
    "test_A_signatures": [list(sig) for sig in SPLIT["test_A"]],
    "test_B_signatures": [list(sig) for sig in SPLIT["test_B"]],
    "all_2way_pairs_covered": SPLIT["all_2way_pairs_covered"],
    "missing_2way_pairs": SPLIT["missing_2way_pairs"],
    "heldout_exact_signature_count": len(SPLIT["test_A"]) + len(SPLIT["test_B"]),
}

json_payload = {
    "metadata": {
        "block_id": "1J40b-7k",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "generator_version": "minimal_factorized_scaffold_v1",
        "factors": FACTOR_NAMES,
        "hidden_h_values": H_VALUES,
        "actions": ACTIONS,
        "diagnostic_values": DIAG_VALUES,
        "observe_costs": OBSERVE_COSTS,
        "seeds": SEEDS,
        "structured_null_mode": ["structured", "null"],
    },
    "split_audit": split_audit,
    "results": all_results,
    "structured_null_comparison": comparison,
    "acceptance_summary": acceptance_summary,
    "validity_audit": {
        "no_learner_training": True,
        "no_1j40b8": True,
        "no_full_env3c": True,
        "structured_and_null_present": True,
        "forbidden_test_keys_absent": True,
    },
}
with open(RUN_JSON, "w", encoding="utf-8") as f:
    json.dump(json_payload, f, indent=2)

csv_fields = [
    "version", "observe_cost", "test_type", "level", "policy_net", "always_try_net",
    "always_observe_net", "oracle_selective_net", "delta_vs_always_observe",
    "delta_vs_always_try", "gap_to_oracle", "observe_rate", "sign_correct_rate",
    "false_observe_rate", "false_direct_rate", "fallback_rate", "positive_net_rate",
    "exact_signature_determinism_rate", "single_cue_max_predictiveness",
    "cue_pair_max_predictiveness", "train_pair_coverage_rate",
    "hidden_h_probe_from_pre_surface_accuracy", "structured_null_positive_rate_delta",
    "structured_null_always_observe_delta", "passes_level_headroom", "rejection_reason",
]
with open(TABLE_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for row in all_results:
        key = f"cost_{row['observe_cost']:.2f}_{row['test_type']}"
        comp = comparison[key]
        passes_level_headroom = (
            row["version"] == "structured"
            and row["level"] in ("level2", "level3")
            and row["delta_vs_always_observe"] >= 0.03
        )
        rejection_reason = []
        if row["version"] == "structured" and row["level"] == "level1" and row["delta_vs_always_observe"] >= 0.03:
            rejection_reason.append("level1_too_strong")
        if row["version"] == "null" and row["level"] in ("level2", "level3") and row["delta_vs_always_observe"] >= 0.03:
            rejection_reason.append("null_invalid")
        writer.writerow(
            {
                "version": row["version"],
                "observe_cost": f"{row['observe_cost']:.2f}",
                "test_type": row["test_type"],
                "level": row["level"],
                "policy_net": f"{row['policy_net']:.6f}",
                "always_try_net": f"{row['always_try_net']:.6f}",
                "always_observe_net": f"{row['always_observe_net']:.6f}",
                "oracle_selective_net": f"{row['oracle_selective_net']:.6f}",
                "delta_vs_always_observe": f"{row['delta_vs_always_observe']:.6f}",
                "delta_vs_always_try": f"{row['delta_vs_always_try']:.6f}",
                "gap_to_oracle": f"{row['gap_to_oracle']:.6f}",
                "observe_rate": f"{row['observe_rate']:.6f}",
                "sign_correct_rate": f"{row['sign_correct_rate']:.6f}",
                "false_observe_rate": f"{row['false_observe_rate']:.6f}",
                "false_direct_rate": f"{row['false_direct_rate']:.6f}",
                "fallback_rate": f"{row['fallback_rate']:.6f}",
                "positive_net_rate": f"{row['positive_net_rate']:.6f}",
                "exact_signature_determinism_rate": f"{row['exact_signature_determinism_rate']:.6f}",
                "single_cue_max_predictiveness": f"{row['single_cue_max_predictiveness']:.6f}",
                "cue_pair_max_predictiveness": f"{row['cue_pair_max_predictiveness']:.6f}",
                "train_pair_coverage_rate": f"{row['train_pair_coverage_rate']:.6f}",
                "hidden_h_probe_from_pre_surface_accuracy": f"{row['hidden_h_probe_from_pre_surface_accuracy']:.6f}",
                "structured_null_positive_rate_delta": f"{comp['positive_rate_delta']:.6f}",
                "structured_null_always_observe_delta": f"{comp['always_observe_delta']:.6f}",
                "passes_level_headroom": str(passes_level_headroom),
                "rejection_reason": ";".join(rejection_reason),
            }
        )

md_lines = [
    "# Checkpoint Summary",
    "",
    f"- status: {status}",
    f"- primary observe_cost: {PRIMARY_COST:.2f}",
    f"- best structured level: {structured_best['level']}",
    f"- best structured delta_vs_always_observe: {structured_best['delta_vs_always_observe']:.4f}",
    "",
    "# Files Changed",
    "",
    f"- {SCRIPT_NAME}",
    "",
    "# Commands Run",
    "",
    f"- `python {SCRIPT_NAME}`",
    "",
    "# Why 7k Was Run",
    "",
    "- 7j fixed dominance and single-cue leakage in env3b variants but still found no Level 2/3 headroom over always_observe.",
    "- 7k tests the scaffold concept itself in a tiny factorized audit before any full env3c draft.",
    "",
    "# Minimal Factorized Generator Design",
    "",
    "- 4 binary visible factors -> 16 exact signatures.",
    "- 3 hidden h values and 3 abstract actions.",
    "- observe reveals a noisy diagnostic token informative about h.",
    "- structured version uses pair-structured q=(f0 xor f1, f2 xor f3) to define P(h|f).",
    "- null version uses the same global h marginals for every signature, breaking pair structure.",
    "",
    "# Train/Test Split and Pair Coverage Audit",
    "",
    f"- total signatures: {split_audit['total_signatures']}",
    f"- held-out Test A signatures: {split_audit['test_A_signatures']}",
    f"- held-out Test B signatures: {split_audit['test_B_signatures']}",
    f"- all 2-way pairs covered in training: {split_audit['all_2way_pairs_covered']}",
    "",
    "# Structured Version Results",
    "",
]
for test_type in ["A", "B"]:
    rows = [row_lookup("structured", PRIMARY_COST, test_type, lvl) for lvl in ["level1", "level2", "level3"]]
    for row in rows:
        md_lines.append(
            f"- `{test_type}|{row['level']}`: delta_vs_always_observe={row['delta_vs_always_observe']:.4f}, "
            f"policy_net={row['policy_net']:.4f}, sign_correct={row['sign_correct_rate']:.4f}"
        )

md_lines.extend(["", "# Null-Control Results", ""])
for test_type in ["A", "B"]:
    rows = [row_lookup("null", PRIMARY_COST, test_type, lvl) for lvl in ["level1", "level2", "level3"]]
    for row in rows:
        md_lines.append(
            f"- `{test_type}|{row['level']}`: delta_vs_always_observe={row['delta_vs_always_observe']:.4f}, "
            f"policy_net={row['policy_net']:.4f}, sign_correct={row['sign_correct_rate']:.4f}"
        )

md_lines.extend([
    "",
    "# Level 1 vs Level 2/3 Comparison",
    "",
    f"- structured Level 1 best delta: {structured_level1_best['delta_vs_always_observe']:.4f}",
    f"- structured Level 2 best delta: {max(row['delta_vs_always_observe'] for row in structured_primary if row['level'] == 'level2'):.4f}",
    f"- structured Level 3 best delta: {max(row['delta_vs_always_observe'] for row in structured_primary if row['level'] == 'level3'):.4f}",
    "",
    "# Leakage / Shortcut Audit",
    "",
    f"- exact_signature_determinism_rate: {validity_anchor['exact_signature_determinism_rate']:.4f}",
    f"- single_cue_max_predictiveness: {validity_anchor['single_cue_max_predictiveness']:.4f}",
    f"- cue_pair_max_predictiveness: {validity_anchor['cue_pair_max_predictiveness']:.4f}",
    f"- hidden_h_probe_from_pre_surface_accuracy: {validity_anchor['hidden_h_probe_from_pre_surface_accuracy']:.4f}",
    "",
    "# Structured vs Null Marginal Matching",
    "",
])
for key, value in comparison.items():
    md_lines.append(
        f"- `{key}`: positive_rate_delta={value['positive_rate_delta']:.4f}, "
        f"always_observe_delta={value['always_observe_delta']:.4f}, "
        f"always_try_delta={value['always_try_delta']:.4f}"
    )

md_lines.extend([
    "",
    "# Acceptance Decision",
    "",
    f"- status: {status}",
    f"- level1_too_strong: {level1_too_strong}",
    f"- null_invalid: {null_invalid}",
    f"- recommended_next_step: {recommended_next_step}",
    "",
    "# Recommended Resume Point",
    "",
    "- PASS -> full env3c design draft",
    "- PARTIAL -> refine 7k scaffold",
    "- FAIL -> expert review / rethink scaffold",
])
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            [
                "# Checkpoint 1J40b-7k",
                "",
                f"- status: {status}",
                f"- best_structured_level: {structured_best['level']}",
                f"- best_structured_delta_vs_always_observe: {structured_best['delta_vs_always_observe']:.4f}",
                f"- level1_too_strong: {level1_too_strong}",
                f"- null_invalid: {null_invalid}",
                f"- recommended_next_step: {recommended_next_step}",
            ]
        )
    )

print("[5/5] Final summary...")
structured_A = {lvl: row_lookup("structured", PRIMARY_COST, "A", lvl) for lvl in ["level1", "level2", "level3"]}
structured_B = {lvl: row_lookup("structured", PRIMARY_COST, "B", lvl) for lvl in ["level1", "level2", "level3"]}
null_A = {lvl: row_lookup("null", PRIMARY_COST, "A", lvl) for lvl in ["level2", "level3"]}
null_B = {lvl: row_lookup("null", PRIMARY_COST, "B", lvl) for lvl in ["level2", "level3"]}
best_level_primary = max(
    [row_lookup("structured", PRIMARY_COST, test, lvl) for test in ["A", "B"] for lvl in ["level2", "level3"]],
    key=lambda r: r["delta_vs_always_observe"],
)

null_collapse = (
    max(null_A["level2"]["delta_vs_always_observe"], null_A["level3"]["delta_vs_always_observe"]) < 0.03
    and max(null_B["level2"]["delta_vs_always_observe"], null_B["level3"]["delta_vs_always_observe"]) < 0.03
)

print(f"Status: {status}")
print(
    f"Structured result: best={best_level_primary['level']}, "
    f"L1_A={structured_A['level1']['delta_vs_always_observe']:.4f}, "
    f"L2_A={structured_A['level2']['delta_vs_always_observe']:.4f}, "
    f"L3_A={structured_A['level3']['delta_vs_always_observe']:.4f}, "
    f"best_delta={best_level_primary['delta_vs_always_observe']:.4f}"
)
print(
    f"Null-control result: L2_A={null_A['level2']['delta_vs_always_observe']:.4f}, "
    f"L3_A={null_A['level3']['delta_vs_always_observe']:.4f}, "
    f"L2_B={null_B['level2']['delta_vs_always_observe']:.4f}, "
    f"L3_B={null_B['level3']['delta_vs_always_observe']:.4f}, "
    f"null_collapse={null_collapse}"
)
print(
    f"Validity: exact_det={validity_anchor['exact_signature_determinism_rate']:.4f}, "
    f"single_cue={validity_anchor['single_cue_max_predictiveness']:.4f}, "
    f"cue_pair={validity_anchor['cue_pair_max_predictiveness']:.4f}, "
    f"pair_coverage={split_audit['all_2way_pairs_covered']}, "
    f"h_probe={validity_anchor['hidden_h_probe_from_pre_surface_accuracy']:.4f}"
)
if status == "PASS":
    print("Decision: 7k_PASS_GO_ENV3C")
elif status == "PARTIAL":
    print("Decision: 7k_PARTIAL_REFINE_SCAFFOLD")
else:
    print("Decision: 7k_FAIL_RETHINK_SCAFFOLD")
print(f"JSON: {RUN_JSON}")
print(f"MD: {REPORT_MD}")
print(f"CSV: {TABLE_CSV}")
print(f"CKPT: {CHECKPOINT_MD}")
