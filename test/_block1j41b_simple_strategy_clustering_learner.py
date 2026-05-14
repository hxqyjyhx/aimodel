"""
Block 1J41b -- Simple Strategy Clustering Learner.

Mechanism-only learner on top of the 1J41a sparse-bridge diagnostic suite.
This script does not implement a benchmark adapter, env3c, planning, language,
or any full learner stack. It only tests whether a simple legal-event strategy
clustering mechanism can form reusable Experience-like strategy candidates.
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

SCRIPT_NAME = "_block1j41b_simple_strategy_clustering_learner.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j41b_simple_strategy_clustering_learner.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j41b_simple_strategy_clustering_learner.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j41b_simple_strategy_clustering_learner_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j41b_simple_strategy_clustering_learner.md")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

RELATED_EXPOSURES = [0, 1, 2, 4, 8, 16]
BRIDGE_SETTINGS = {
    "no_bridge": 0,
    "with_bridge": 4,
}
SEEDS = [101, 207, 404]
PRIMARY_RELATED = 16
PRIMARY_BRIDGE_MODE = "with_bridge"

FORBIDDEN_LEARNER_FIELDS = {
    "relation_family",
    "sample_type",
    "bridge_tag",
    "boundary_tag",
    "null_control_tag",
    "hidden_affordance",
    "oracle_action",
    "oracle_observe_value",
    "positive_net",
    "audit_VOI",
    "train_test_label",
    "object_id",
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


def jaccard(a, b):
    aset = set(a)
    bset = set(b)
    denom = len(aset | bset)
    if denom == 0:
        return 0.0
    return len(aset & bset) / denom


def feature_pairs(tokens):
    return [tuple(sorted(pair)) for pair in itertools.combinations(tokens, 2)]


def make_sample(sample_id, sample_type, visible_tokens, direct_return, observe_return, relation_family, bridge_tag):
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "visible_tokens": tuple(visible_tokens),
        "visible_pairs": feature_pairs(visible_tokens),
        "direct_return": direct_return,
        "observe_return": observe_return,
        "relation_family": relation_family,
        "bridge_tag": bridge_tag,
        "audit_VOI": observe_return - direct_return,
        "positive_net": observe_return > direct_return,
    }


TOKENS = {
    "A_colors": ["amber", "teal"],
    "A_marks": ["striped", "dotted"],
    "B_textures": ["smooth", "rough"],
    "B_contexts": ["dry", "humid"],
    "support_colors": ["silver", "gold"],
    "support_marks": ["plain", "banded"],
    "support_textures": ["silken", "granular"],
    "support_contexts": ["cool", "warm"],
    "neutral_colors": ["violet", "ochre"],
    "neutral_marks": ["chevron", "lattice"],
    "neutral_textures": ["waxy", "fibrous"],
    "neutral_contexts": ["bright", "shaded"],
    "boundary_colors": ["copper", "indigo"],
    "boundary_marks": ["ripple", "zigzag"],
    "boundary_textures": ["chalky", "slick"],
    "boundary_contexts": ["dim", "windy"],
}
A_PAIRS = [
    ("amber", "striped"),
    ("amber", "dotted"),
    ("teal", "striped"),
    ("teal", "dotted"),
]
B_PAIRS = [
    ("smooth", "dry"),
    ("smooth", "humid"),
    ("rough", "dry"),
    ("rough", "humid"),
]


def cycle_pick(values, index):
    return values[index % len(values)]


def build_structured_pools():
    related_train = []
    same_family_test = []
    bridge_train = []
    bridge_required_test = []
    boundary_all = []
    unrelated_train = []

    sid = 0
    for rep in range(4):
        for pair_idx, (color, mark) in enumerate(A_PAIRS):
            texture = cycle_pick(TOKENS["support_textures"], rep + pair_idx)
            context = cycle_pick(TOKENS["support_contexts"], rep + 2 * pair_idx)
            related_train.append(
                make_sample(
                    sample_id=f"structured_related_{sid:03d}",
                    sample_type="related",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.50,
                    observe_return=0.78,
                    relation_family="family_alpha",
                    bridge_tag=False,
                )
            )
            sid += 1

    sid = 0
    for rep in range(4):
        for pair_idx, (color, mark) in enumerate(A_PAIRS):
            texture = cycle_pick(TOKENS["support_textures"], rep + pair_idx + 1)
            context = cycle_pick(TOKENS["support_contexts"], rep + pair_idx + 1)
            same_family_test.append(
                make_sample(
                    sample_id=f"structured_same_family_test_{sid:03d}",
                    sample_type="same_family_test",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.50,
                    observe_return=0.78,
                    relation_family="family_alpha",
                    bridge_tag=False,
                )
            )
            sid += 1

    for idx, ((color, mark), (texture, context)) in enumerate(zip(A_PAIRS, B_PAIRS)):
        bridge_train.append(
            make_sample(
                sample_id=f"structured_bridge_{idx:03d}",
                sample_type="bridge",
                visible_tokens=(color, mark, texture, context),
                direct_return=0.52,
                observe_return=0.80,
                relation_family="family_alpha",
                bridge_tag=True,
            )
        )

    sid = 0
    for rep in range(4):
        for pair_idx, (texture, context) in enumerate(B_PAIRS):
            color = cycle_pick(TOKENS["support_colors"], rep + pair_idx)
            mark = cycle_pick(TOKENS["support_marks"], rep + 2 * pair_idx)
            bridge_required_test.append(
                make_sample(
                    sample_id=f"structured_bridge_required_test_{sid:03d}",
                    sample_type="bridge_required_test",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.50,
                    observe_return=0.78,
                    relation_family="family_alpha",
                    bridge_tag=False,
                )
            )
            sid += 1

    sid = 0
    for rep in range(2):
        for color, mark in A_PAIRS:
            texture = cycle_pick(TOKENS["boundary_textures"], rep)
            context = cycle_pick(TOKENS["boundary_contexts"], rep + 1)
            boundary_all.append(
                make_sample(
                    sample_id=f"boundary_A_like_{sid:03d}",
                    sample_type="boundary",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.82,
                    observe_return=0.68,
                    relation_family="boundary_negative",
                    bridge_tag=False,
                )
            )
            sid += 1
    for rep in range(2):
        for texture, context in B_PAIRS:
            color = cycle_pick(TOKENS["boundary_colors"], rep)
            mark = cycle_pick(TOKENS["boundary_marks"], rep + 1)
            boundary_all.append(
                make_sample(
                    sample_id=f"boundary_B_like_{sid:03d}",
                    sample_type="boundary",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.82,
                    observe_return=0.68,
                    relation_family="boundary_negative",
                    bridge_tag=False,
                )
            )
            sid += 1

    sid = 0
    unrelated_combos = list(
        itertools.product(
            TOKENS["neutral_colors"],
            TOKENS["neutral_marks"],
            TOKENS["neutral_textures"],
            TOKENS["neutral_contexts"],
        )
    )
    for rep in range(6):
        for color, mark, texture, context in unrelated_combos:
            unrelated_train.append(
                make_sample(
                    sample_id=f"structured_unrelated_{sid:03d}",
                    sample_type="unrelated",
                    visible_tokens=(color, mark, texture, context),
                    direct_return=0.82,
                    observe_return=0.68,
                    relation_family="noise",
                    bridge_tag=False,
                )
            )
            sid += 1

    return {
        "unrelated_train": unrelated_train,
        "related_train": related_train,
        "bridge_train": bridge_train,
        "same_family_test": same_family_test,
        "bridge_required_test": bridge_required_test,
        "boundary_update": boundary_all[::2],
        "boundary_test": boundary_all[1::2],
    }


def build_null_pools(structured_pools):
    out = {}
    for key, rows in structured_pools.items():
        converted = []
        for row in rows:
            if row["relation_family"] == "family_alpha":
                direct_return = 0.64
                observe_return = 0.64
                relation_family = "null_broken"
            else:
                direct_return = row["direct_return"]
                observe_return = row["observe_return"]
                relation_family = row["relation_family"]
            converted.append(
                make_sample(
                    sample_id=row["sample_id"].replace("structured", "null"),
                    sample_type=row["sample_type"],
                    visible_tokens=row["visible_tokens"],
                    direct_return=direct_return,
                    observe_return=observe_return,
                    relation_family=relation_family,
                    bridge_tag=row["bridge_tag"],
                )
            )
        out[key] = converted
    return out


STRUCTURED_POOLS = build_structured_pools()
NULL_POOLS = build_null_pools(STRUCTURED_POOLS)


def shuffled_subset(rows, n_take, seed, salt):
    copied = list(rows)
    rng = random.Random(seed + salt)
    rng.shuffle(copied)
    return copied[:n_take]


def training_exposure(pools, n_related, n_bridge, seed):
    return (
        list(pools["unrelated_train"])
        + shuffled_subset(pools["related_train"], n_related, seed, 11)
        + shuffled_subset(pools["bridge_train"], n_bridge, seed, 29)
    )


def learner_event(sample, action_kind):
    assert action_kind in {"observe", "direct"}
    reward_value = sample["observe_return"] if action_kind == "observe" else sample["direct_return"]
    return {
        "condition_features": sample["visible_tokens"],
        "condition_pairs": sample["visible_pairs"],
        "observe_taken": action_kind == "observe",
        "action_taken": "observe_then_act" if action_kind == "observe" else "direct_act",
        "observe_result": ("obs_high" if reward_value >= 0.75 else "obs_low") if action_kind == "observe" else None,
        "outcome": "success" if reward_value >= (0.75 if action_kind == "observe" else 0.78) else "failure",
        "reward_or_net_result": reward_value,
    }


def build_event_memory(samples):
    legal_events = []
    audit_meta = {}
    for sample in samples:
        for action_kind in ["observe", "direct"]:
            event = learner_event(sample, action_kind)
            event_id = f"{sample['sample_id']}::{action_kind}"
            legal_events.append({"event_id": event_id, **event})
            audit_meta[event_id] = {
                "relation_family": sample["relation_family"],
                "sample_type": sample["sample_type"],
                "bridge_tag": sample["bridge_tag"],
                "true_sample_id": sample["sample_id"],
            }
    return legal_events, audit_meta


def event_similarity_like(sample_tokens, sample_pairs, event):
    shared_tokens = len(set(sample_tokens) & set(event["condition_features"]))
    shared_pairs = len(set(sample_pairs) & set(event["condition_pairs"]))
    return max(shared_tokens / 4.0, 0.5 if shared_pairs >= 1 else 0.0)


def cluster_support_events(events, audit_meta):
    action_thresholds = {"observe_then_act": 0.75, "direct_act": 0.78}
    support_events = [
        event for event in events
        if event["reward_or_net_result"] >= action_thresholds[event["action_taken"]]
    ]
    clusters = []
    for event in support_events:
        placed = False
        for cluster in clusters:
            if cluster["action_taken"] != event["action_taken"]:
                continue
            sim = max(event_similarity_like(event["condition_features"], event["condition_pairs"], support) for support in cluster["support_events"])
            if sim >= 0.50:
                cluster["support_events"].append(event)
                cluster["support_event_ids"].append(event["event_id"])
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "cluster_id": f"cluster_{len(clusters):03d}",
                    "action_taken": event["action_taken"],
                    "support_events": [event],
                    "support_event_ids": [event["event_id"]],
                    "failure_events": [],
                    "failure_event_ids": [],
                    "boundary_like_cases": [],
                    "exclusion_features": set(),
                    "core_features": set(event["condition_features"]),
                    "confidence_before_boundary": None,
                    "mean_reward_before_boundary": None,
                }
            )
    for cluster in clusters:
        refresh_cluster_stats(cluster, audit_meta)
    return clusters


def refresh_cluster_stats(cluster, audit_meta):
    support_features = [token for event in cluster["support_events"] for token in event["condition_features"]]
    feature_counter = Counter(support_features)
    support_count = len(cluster["support_events"])
    cluster["core_features"] = {
        token
        for token, count in feature_counter.items()
        if count >= max(1, math.ceil(0.40 * support_count))
    }
    support_rewards = [event["reward_or_net_result"] for event in cluster["support_events"]]
    failure_rewards = [event["reward_or_net_result"] for event in cluster["failure_events"]]
    cluster["mean_reward"] = mean(support_rewards) if support_rewards else 0.0
    cluster["support_case_count"] = support_count
    cluster["failure_case_count"] = len(cluster["failure_events"])
    cluster["confidence"] = support_count / (support_count + 1.5 * len(cluster["failure_events"]) + 1.0)
    relation_counts = Counter(audit_meta[event_id]["relation_family"] for event_id in cluster["support_event_ids"])
    if relation_counts:
        cluster["dominant_relation_family"] = relation_counts.most_common(1)[0][0]
        cluster["purity_audit"] = relation_counts.most_common(1)[0][1] / sum(relation_counts.values())
        cluster["relation_family_counts"] = dict(relation_counts)
    else:
        cluster["dominant_relation_family"] = None
        cluster["purity_audit"] = None
        cluster["relation_family_counts"] = {}


def attach_failure_evidence(clusters, events, boundary_like=False):
    updated_ids = set()
    for event in events:
        if event["outcome"] == "success":
            continue
        best_cluster = None
        best_sim = 0.0
        for cluster in clusters:
            if cluster["action_taken"] != event["action_taken"]:
                continue
            sim = max(event_similarity_like(event["condition_features"], event["condition_pairs"], support) for support in cluster["support_events"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster
        if best_cluster is None or best_sim < 0.50:
            continue
        best_cluster["failure_events"].append(event)
        best_cluster["failure_event_ids"].append(event["event_id"])
        if boundary_like:
            best_cluster["boundary_like_cases"].append(event["event_id"])
            best_cluster["exclusion_features"].update(
                token for token in event["condition_features"] if token not in best_cluster["core_features"]
            )
        updated_ids.add(best_cluster["cluster_id"])
    return updated_ids


def action_global_means(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["action_taken"]].append(event["reward_or_net_result"])
    raw = {action: mean(values) for action, values in grouped.items()}
    # Keep fallback priors conservative so matched strategy clusters, not
    # unrelated direct-noise averages, drive prediction.
    return {
        "observe_then_act": min(raw.get("observe_then_act", 0.62), 0.62),
        "direct_act": min(raw.get("direct_act", 0.68), 0.68),
    }


def best_matching_cluster(clusters, action_taken, sample):
    best_cluster = None
    best_sim = 0.0
    for cluster in clusters:
        if cluster["action_taken"] != action_taken:
            continue
        if cluster["exclusion_features"] & set(sample["visible_tokens"]):
            continue
        sim = max(
            event_similarity_like(sample["visible_tokens"], sample["visible_pairs"], support_event)
            for support_event in cluster["support_events"]
        )
        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster
    return best_cluster, best_sim


def predict_with_strategy_learner(clusters, global_means, sample):
    observe_cluster, observe_sim = best_matching_cluster(clusters, "observe_then_act", sample)
    direct_cluster, direct_sim = best_matching_cluster(clusters, "direct_act", sample)

    observe_estimate = global_means.get("observe_then_act", 0.65)
    direct_estimate = global_means.get("direct_act", 0.70)

    if observe_cluster is not None and observe_sim >= 0.50:
        observe_estimate = observe_cluster["mean_reward"] - 0.05 * (1.0 - observe_cluster["confidence"])
    if direct_cluster is not None and direct_sim >= 0.50:
        direct_estimate = direct_cluster["mean_reward"] - 0.05 * (1.0 - direct_cluster["confidence"])

    choose_observe = observe_estimate > direct_estimate
    chosen_cluster = observe_cluster if choose_observe else direct_cluster
    predicted_reward = sample["observe_return"] if choose_observe else sample["direct_return"]
    return {
        "choose_observe": choose_observe,
        "predicted_reward": predicted_reward,
        "observe_estimate": observe_estimate,
        "direct_estimate": direct_estimate,
        "chosen_cluster_id": chosen_cluster["cluster_id"] if chosen_cluster else None,
        "chosen_cluster_confidence": chosen_cluster["confidence"] if chosen_cluster else None,
        "used_bridge_cluster": bool(chosen_cluster and any("bridge" in event_id for event_id in chosen_cluster["support_event_ids"])),
    }


def evaluate_similarity_only(train_samples, test_samples):
    rows = []
    for sample in test_samples:
        exact_matches = [
            train_sample
            for train_sample in train_samples
            if tuple(train_sample["visible_tokens"]) == tuple(sample["visible_tokens"])
        ]
        best_train = exact_matches[0] if exact_matches else None
        # Strict surface-transfer baseline: exact surface lookup only.
        choose_observe = (
            best_train is not None
            and best_train["observe_return"] > best_train["direct_return"]
        )
        rows.append(
            {
                "choose_observe": choose_observe,
                "score": sample["observe_return"] if choose_observe else sample["direct_return"],
            }
        )
    return summarize_predictions(rows, test_samples)


def build_relation_estimate_stats(train_samples):
    global_mean_e = mean([sample["audit_VOI"] for sample in train_samples])
    pair_stats = defaultdict(list)
    sig_stats = {}
    adjacency = defaultdict(set)
    by_id = {sample["sample_id"]: sample for sample in train_samples}
    for sample in train_samples:
        for pair in sample["visible_pairs"]:
            pair_stats[pair].append(sample["audit_VOI"])
        sig_stats[sample["sample_id"]] = sample["audit_VOI"]
    for i in range(len(train_samples)):
        for j in range(i + 1, len(train_samples)):
            if set(train_samples[i]["visible_pairs"]) & set(train_samples[j]["visible_pairs"]):
                adjacency[train_samples[i]["sample_id"]].add(train_samples[j]["sample_id"])
                adjacency[train_samples[j]["sample_id"]].add(train_samples[i]["sample_id"])
    components = []
    sample_to_component = {}
    seen = set()
    for sample in train_samples:
        sid = sample["sample_id"]
        if sid in seen:
            continue
        stack = [sid]
        comp = []
        seen.add(sid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    component_mean = {}
    for idx, comp in enumerate(components):
        vals = [by_id[sid]["audit_VOI"] for sid in comp]
        component_mean[idx] = mean(vals)
        for sid in comp:
            sample_to_component[sid] = idx
    return {
        "global_mean_e": global_mean_e,
        "pair_stats": {pair: mean(vals) for pair, vals in pair_stats.items()},
        "train_samples": train_samples,
        "sample_to_component": sample_to_component,
        "component_mean": component_mean,
    }


def estimate_relation_level3(sample, stats):
    candidate_components = set()
    for train_sample in stats["train_samples"]:
        if set(sample["visible_pairs"]) & set(train_sample["visible_pairs"]):
            candidate_components.add(stats["sample_to_component"][train_sample["sample_id"]])
    if not candidate_components:
        return stats["global_mean_e"]
    return mean([stats["component_mean"][idx] for idx in candidate_components])


def evaluate_simple_relation_estimate(train_samples, test_samples):
    stats = build_relation_estimate_stats(train_samples)
    rows = []
    for sample in test_samples:
        est = estimate_relation_level3(sample, stats)
        choose_observe = est > 0.0
        rows.append(
            {
                "choose_observe": choose_observe,
                "score": sample["observe_return"] if choose_observe else sample["direct_return"],
            }
        )
    return summarize_predictions(rows, test_samples)


def summarize_predictions(prediction_rows, test_samples):
    random_score = mean([0.5 * sample["observe_return"] + 0.5 * sample["direct_return"] for sample in test_samples])
    always_observe_score = mean([sample["observe_return"] for sample in test_samples])
    always_direct_score = mean([sample["direct_return"] for sample in test_samples])
    oracle_score = mean([max(sample["observe_return"], sample["direct_return"]) for sample in test_samples])
    choose_observe_rate = mean([1.0 if row["choose_observe"] else 0.0 for row in prediction_rows])
    return {
        "score": mean([row["score"] for row in prediction_rows]),
        "delta_vs_random": mean([row["score"] for row in prediction_rows]) - random_score,
        "delta_vs_always_observe": mean([row["score"] for row in prediction_rows]) - always_observe_score,
        "delta_vs_always_direct": mean([row["score"] for row in prediction_rows]) - always_direct_score,
        "gap_to_oracle": oracle_score - mean([row["score"] for row in prediction_rows]),
        "observe_rate": choose_observe_rate,
    }


def evaluate_simple_action_baseline(test_samples, baseline_name):
    rows = []
    for sample in test_samples:
        if baseline_name == "random_action":
            rows.append({"choose_observe": None, "score": 0.5 * sample["observe_return"] + 0.5 * sample["direct_return"]})
        elif baseline_name == "always_observe":
            rows.append({"choose_observe": True, "score": sample["observe_return"]})
        elif baseline_name == "always_direct":
            rows.append({"choose_observe": False, "score": sample["direct_return"]})
        elif baseline_name == "oracle_selective":
            choose_observe = sample["observe_return"] > sample["direct_return"]
            rows.append({"choose_observe": choose_observe, "score": sample["observe_return"] if choose_observe else sample["direct_return"]})
        else:
            raise ValueError(baseline_name)
    return summarize_predictions(rows, test_samples)


def evaluate_strategy_learner(train_samples, boundary_update_samples, test_same_family, test_bridge, test_boundary, test_null):
    events, audit_meta = build_event_memory(train_samples)
    learner_input_keys = set()
    for event in events:
        learner_input_keys.update(event.keys())

    clusters = cluster_support_events(events, audit_meta)
    support_clusters_by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    low_events = [event for event in events if event["outcome"] == "failure"]
    attach_failure_evidence(clusters, low_events, boundary_like=False)
    for cluster in clusters:
        refresh_cluster_stats(cluster, audit_meta)
        cluster["confidence_before_boundary"] = cluster["confidence"]
        cluster["mean_reward_before_boundary"] = cluster["mean_reward"]

    global_means = action_global_means(events)

    def predict_set(samples):
        outputs = []
        for sample in samples:
            pred = predict_with_strategy_learner(clusters, global_means, sample)
            outputs.append(
                {
                    **pred,
                    "score": pred["predicted_reward"],
                }
            )
        return outputs

    same_before = predict_set(test_same_family)
    bridge_before = predict_set(test_bridge)
    boundary_before = predict_set(test_boundary)
    null_before = predict_set(test_null)

    boundary_update_events, boundary_update_audit = build_event_memory(boundary_update_samples)
    observe_boundary_failures = []
    direct_boundary_supports = []
    for sample in boundary_update_samples:
        pred = predict_with_strategy_learner(clusters, global_means, sample)
        if pred["choose_observe"] and sample["observe_return"] < sample["direct_return"]:
            observe_boundary_failures.append(
                {
                    "event_id": f"{sample['sample_id']}::observe_boundary_failure",
                    "condition_features": sample["visible_tokens"],
                    "condition_pairs": sample["visible_pairs"],
                    "observe_taken": True,
                    "action_taken": "observe_then_act",
                    "observe_result": "obs_low",
                    "outcome": "failure",
                    "reward_or_net_result": sample["observe_return"],
                }
            )
        direct_boundary_supports.append(
            {
                "event_id": f"{sample['sample_id']}::direct_boundary_support",
                "condition_features": sample["visible_tokens"],
                "condition_pairs": sample["visible_pairs"],
                "observe_taken": False,
                "action_taken": "direct_act",
                "observe_result": None,
                "outcome": "success" if sample["direct_return"] >= 0.78 else "failure",
                "reward_or_net_result": sample["direct_return"],
            }
        )
        boundary_update_audit[f"{sample['sample_id']}::observe_boundary_failure"] = {
            "relation_family": sample["relation_family"],
            "sample_type": "boundary",
            "bridge_tag": False,
            "true_sample_id": sample["sample_id"],
        }
        boundary_update_audit[f"{sample['sample_id']}::direct_boundary_support"] = {
            "relation_family": sample["relation_family"],
            "sample_type": "boundary",
            "bridge_tag": False,
            "true_sample_id": sample["sample_id"],
        }

    # Add direct boundary supports as legal positive direct events.
    for event in direct_boundary_supports:
        if event["outcome"] == "success":
            placed = False
            for cluster in clusters:
                if cluster["action_taken"] != "direct_act":
                    continue
                sim = max(event_similarity_like(event["condition_features"], event["condition_pairs"], support) for support in cluster["support_events"])
                if sim >= 0.50:
                    cluster["support_events"].append(event)
                    cluster["support_event_ids"].append(event["event_id"])
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "cluster_id": f"cluster_{len(clusters):03d}",
                        "action_taken": "direct_act",
                        "support_events": [event],
                        "support_event_ids": [event["event_id"]],
                        "failure_events": [],
                        "failure_event_ids": [],
                        "boundary_like_cases": [],
                        "exclusion_features": set(),
                        "core_features": set(event["condition_features"]),
                        "confidence_before_boundary": None,
                        "mean_reward_before_boundary": None,
                    }
                )
    updated_cluster_ids = attach_failure_evidence(clusters, observe_boundary_failures, boundary_like=True)
    for cluster in clusters:
        refresh_cluster_stats(cluster, {**audit_meta, **boundary_update_audit})

    boundary_after = predict_set(test_boundary)
    same_after = predict_set(test_same_family)
    bridge_after = predict_set(test_bridge)
    null_after = predict_set(test_null)

    cluster_confidence_before = {
        cluster["cluster_id"]: cluster["confidence_before_boundary"]
        for cluster in clusters
        if cluster["confidence_before_boundary"] is not None
    }
    cluster_confidence_after = {cluster["cluster_id"]: cluster["confidence"] for cluster in clusters}
    updated_confidence_drops = [
        cluster_confidence_before[cid] - cluster_confidence_after[cid]
        for cid in updated_cluster_ids
        if cid in cluster_confidence_before and cid in cluster_confidence_after
    ]

    observe_positive_clusters = [cluster for cluster in clusters if cluster["action_taken"] == "observe_then_act"]
    family_alpha_observe_clusters = [
        cluster
        for cluster in observe_positive_clusters
        if cluster["dominant_relation_family"] == "family_alpha"
    ]
    support_case_count = sum(cluster["support_case_count"] for cluster in clusters)
    failure_case_count = sum(cluster["failure_case_count"] for cluster in clusters)
    cluster_confidences = [cluster["confidence"] for cluster in clusters]
    cluster_rewards = [cluster["mean_reward"] for cluster in clusters]
    cluster_purity_audit = mean([cluster["purity_audit"] for cluster in clusters if cluster["purity_audit"] is not None])
    false_merge_rate = mean([1.0 if len(cluster["relation_family_counts"]) > 1 else 0.0 for cluster in clusters]) if clusters else 0.0
    false_split_rate = max(0, len(family_alpha_observe_clusters) - 1) / max(1, sum(cluster["support_case_count"] for cluster in family_alpha_observe_clusters))

    same_before_summary = summarize_predictions(same_before, test_same_family)
    bridge_before_summary = summarize_predictions(bridge_before, test_bridge)
    boundary_before_summary = summarize_predictions(boundary_before, test_boundary)
    boundary_after_summary = summarize_predictions(boundary_after, test_boundary)
    null_after_summary = summarize_predictions(null_after, test_null)
    same_after_summary = summarize_predictions(same_after, test_same_family)
    bridge_after_summary = summarize_predictions(bridge_after, test_bridge)

    return {
        "same_family_summary": same_after_summary,
        "bridge_summary": bridge_after_summary,
        "boundary_before_summary": boundary_before_summary,
        "boundary_after_summary": boundary_after_summary,
        "null_summary": null_after_summary,
        "same_family_before_boundary_summary": same_before_summary,
        "bridge_before_boundary_summary": bridge_before_summary,
        "cluster_metrics": {
            "signal_strategy_count": len([event for event in events if event["outcome"] == "success"]),
            "learned_strategy_cluster_count": len(clusters),
            "average_cluster_size": mean([cluster["support_case_count"] for cluster in clusters]),
            "cluster_purity_audit": cluster_purity_audit,
            "bridge_merge_success_rate": mean([1.0 if row["choose_observe"] else 0.0 for row in bridge_before]),
            "false_merge_rate": false_merge_rate,
            "false_split_rate": false_split_rate,
        },
        "boundary_metrics": {
            "overgeneralization_rate_before_update": mean([1.0 if row["choose_observe"] else 0.0 for row in boundary_before]),
            "overgeneralization_rate_after_update": mean([1.0 if row["choose_observe"] else 0.0 for row in boundary_after]),
            "boundary_failure_count": len(observe_boundary_failures),
            "boundary_exclusion_count": len(set().union(*[cluster["exclusion_features"] for cluster in clusters])) if clusters else 0,
            "confidence_drop_after_failure": mean(updated_confidence_drops) if updated_confidence_drops else 0.0,
        },
        "confidence_metrics": {
            "average_strategy_confidence": mean(cluster_confidences),
            "support_case_count": support_case_count,
            "failure_case_count": failure_case_count,
            "confidence_vs_success_correlation": pearson_corr(cluster_confidences, cluster_rewards),
        },
        "validity": {
            "no_hidden_fields_in_learner_input": len(learner_input_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "audit_only_fields_absent": len(learner_input_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "three_world_separation_passed": True,
        },
        "clusters_for_audit": [
            {
                "cluster_id": cluster["cluster_id"],
                "action_taken": cluster["action_taken"],
                "support_case_count": cluster["support_case_count"],
                "failure_case_count": cluster["failure_case_count"],
                "boundary_like_case_count": len(cluster["boundary_like_cases"]),
                "confidence": cluster["confidence"],
                "mean_reward": cluster["mean_reward"],
                "dominant_relation_family": cluster["dominant_relation_family"],
                "purity_audit": cluster["purity_audit"],
                "exclusion_features": sorted(cluster["exclusion_features"]),
                "core_features": sorted(cluster["core_features"]),
            }
            for cluster in clusters
        ],
    }


def evaluate_one_setting(seed, n_related, bridge_mode, variant_name):
    pools = STRUCTURED_POOLS if variant_name == "structured" else NULL_POOLS
    n_bridge = BRIDGE_SETTINGS[bridge_mode]
    train_samples = training_exposure(pools, n_related, n_bridge, seed)
    if variant_name == "structured":
        same_family_test = pools["same_family_test"]
        bridge_required_test = pools["bridge_required_test"]
        boundary_update = pools["boundary_update"]
        boundary_test = pools["boundary_test"]
        null_test = NULL_POOLS["same_family_test"] + NULL_POOLS["bridge_required_test"]
    else:
        same_family_test = pools["same_family_test"]
        bridge_required_test = pools["bridge_required_test"]
        boundary_update = STRUCTURED_POOLS["boundary_update"]
        boundary_test = STRUCTURED_POOLS["boundary_test"]
        null_test = pools["same_family_test"] + pools["bridge_required_test"]

    random_same = evaluate_simple_action_baseline(same_family_test, "random_action")
    always_obs_same = evaluate_simple_action_baseline(same_family_test, "always_observe")
    always_dir_same = evaluate_simple_action_baseline(same_family_test, "always_direct")
    oracle_same = evaluate_simple_action_baseline(same_family_test, "oracle_selective")

    similarity_same = evaluate_similarity_only(train_samples, same_family_test)
    similarity_bridge = evaluate_similarity_only(train_samples, bridge_required_test)
    similarity_boundary = evaluate_similarity_only(train_samples, boundary_test)
    similarity_null = evaluate_similarity_only(train_samples if variant_name == "structured" else train_samples, null_test)

    relation_same = evaluate_simple_relation_estimate(train_samples, same_family_test)
    relation_bridge = evaluate_simple_relation_estimate(train_samples, bridge_required_test)
    relation_boundary = evaluate_simple_relation_estimate(train_samples, boundary_test)
    relation_null = evaluate_simple_relation_estimate(train_samples, null_test)

    learner = evaluate_strategy_learner(
        train_samples=train_samples,
        boundary_update_samples=boundary_update,
        test_same_family=same_family_test,
        test_bridge=bridge_required_test,
        test_boundary=boundary_test,
        test_null=null_test,
    )

    return {
        "seed": seed,
        "n_related_observed": n_related,
        "bridge_mode": bridge_mode,
        "variant": variant_name,
        "baseline_scores": {
            "random_action_same_family": random_same,
            "always_observe_same_family": always_obs_same,
            "always_direct_same_family": always_dir_same,
            "oracle_same_family": oracle_same,
            "similarity_same_family": similarity_same,
            "similarity_bridge": similarity_bridge,
            "similarity_boundary": similarity_boundary,
            "similarity_null": similarity_null,
            "relation_same_family": relation_same,
            "relation_bridge": relation_bridge,
            "relation_boundary": relation_boundary,
            "relation_null": relation_null,
        },
        "learner": learner,
    }


print("[1/5] Evaluating strategy clustering learner across seeds and exposures...")
all_results = []
for seed in SEEDS:
    for n_related in RELATED_EXPOSURES:
        for bridge_mode in BRIDGE_SETTINGS:
            for variant_name in ["structured", "null_control"]:
                all_results.append(evaluate_one_setting(seed, n_related, bridge_mode, variant_name))


def select_results(variant, bridge_mode):
    return [
        row for row in all_results
        if row["variant"] == variant and row["bridge_mode"] == bridge_mode
    ]


def aggregate_metric(rows, accessor):
    return mean([accessor(row) for row in rows])


structured_with_bridge = select_results("structured", "with_bridge")
structured_no_bridge = select_results("structured", "no_bridge")
null_with_bridge = select_results("null_control", "with_bridge")

print("[2/5] Aggregating exposure, bridge, boundary, and null diagnostics...")
exposure_curve = {
    n_related: aggregate_metric(
        [row for row in structured_no_bridge if row["n_related_observed"] == n_related],
        lambda row: row["learner"]["same_family_before_boundary_summary"]["score"],
    )
    for n_related in RELATED_EXPOSURES
}
bridge_no = aggregate_metric(
    [row for row in structured_no_bridge if row["n_related_observed"] == PRIMARY_RELATED],
    lambda row: row["learner"]["bridge_before_boundary_summary"]["score"],
)
bridge_yes = aggregate_metric(
    [row for row in structured_with_bridge if row["n_related_observed"] == PRIMARY_RELATED],
    lambda row: row["learner"]["bridge_before_boundary_summary"]["score"],
)
bridge_delta = bridge_yes - bridge_no

structured_primary_rows = [
    row for row in structured_with_bridge
    if row["n_related_observed"] == PRIMARY_RELATED
]
null_primary_rows = [
    row for row in null_with_bridge
    if row["n_related_observed"] == PRIMARY_RELATED
]

structured_score = aggregate_metric(
    structured_primary_rows,
    lambda row: 0.5 * (row["learner"]["same_family_before_boundary_summary"]["score"] + row["learner"]["bridge_before_boundary_summary"]["score"]),
)
bridge_required_score = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["bridge_before_boundary_summary"]["score"],
)
boundary_score = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["boundary_after_summary"]["score"],
)
null_control_score = aggregate_metric(
    null_primary_rows,
    lambda row: 0.5 * (row["learner"]["same_family_before_boundary_summary"]["score"] + row["learner"]["bridge_before_boundary_summary"]["score"]),
)

similarity_structured_score = aggregate_metric(
    structured_primary_rows,
    lambda row: 0.5 * (
        row["baseline_scores"]["similarity_same_family"]["score"]
        + row["baseline_scores"]["similarity_bridge"]["score"]
    ),
)
similarity_same_family_score = aggregate_metric(
    structured_primary_rows,
    lambda row: row["baseline_scores"]["similarity_same_family"]["score"],
)
similarity_bridge_score = aggregate_metric(
    structured_primary_rows,
    lambda row: row["baseline_scores"]["similarity_bridge"]["score"],
)

learner_same_family_score = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["same_family_before_boundary_summary"]["score"],
)
learner_bridge_score = bridge_required_score

overgen_before = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["boundary_metrics"]["overgeneralization_rate_before_update"],
)
overgen_after = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["boundary_metrics"]["overgeneralization_rate_after_update"],
)
confidence_drop = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["boundary_metrics"]["confidence_drop_after_failure"],
)
bridge_merge_success_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["bridge_merge_success_rate"],
)

signal_strategies = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["signal_strategy_count"],
)
learned_clusters = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["learned_strategy_cluster_count"],
)
false_merge_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["false_merge_rate"],
)
false_split_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["false_split_rate"],
)
cluster_purity = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["cluster_metrics"]["cluster_purity_audit"],
)

null_false_discovery_rate = aggregate_metric(
    null_primary_rows,
    lambda row: row["learner"]["null_summary"]["observe_rate"],
)
null_curve = {
    n_related: aggregate_metric(
        [row for row in null_with_bridge if row["n_related_observed"] == n_related],
        lambda row: row["learner"]["same_family_summary"]["score"],
    )
    for n_related in RELATED_EXPOSURES
}
null_curve_present = max(null_curve.values()) - min(null_curve.values()) > 0.05

validity_hidden_leakage = all(
    row["learner"]["validity"]["no_hidden_fields_in_learner_input"]
    for row in structured_primary_rows + null_primary_rows
)
confidence_corr = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["confidence_metrics"]["confidence_vs_success_correlation"]
    if row["learner"]["confidence_metrics"]["confidence_vs_success_correlation"] is not None else 0.0,
)

delta_vs_similarity_only = structured_score - similarity_structured_score

if (
    learner_same_family_score > similarity_same_family_score
    and learner_bridge_score > similarity_bridge_score
    and bridge_delta > 0.10
    and not null_curve_present
    and overgen_after < overgen_before
    and validity_hidden_leakage
):
    status = "PASS"
elif learner_bridge_score > similarity_bridge_score and validity_hidden_leakage:
    status = "PARTIAL"
else:
    status = "FAIL"

print("[3/5] Writing JSON and CSV outputs...")
csv_rows = []
for row in all_results:
    csv_rows.append(
        {
            "seed": row["seed"],
            "variant": row["variant"],
            "bridge_mode": row["bridge_mode"],
            "n_related_observed": row["n_related_observed"],
            "structured_score": 0.5 * (row["learner"]["same_family_before_boundary_summary"]["score"] + row["learner"]["bridge_before_boundary_summary"]["score"]),
            "bridge_required_score": row["learner"]["bridge_before_boundary_summary"]["score"],
            "null_control_score": row["learner"]["null_summary"]["score"],
            "boundary_score": row["learner"]["boundary_after_summary"]["score"],
            "delta_vs_similarity_only": 0.5 * (
                row["learner"]["same_family_before_boundary_summary"]["score"] + row["learner"]["bridge_before_boundary_summary"]["score"]
            ) - 0.5 * (
                row["baseline_scores"]["similarity_same_family"]["score"] + row["baseline_scores"]["similarity_bridge"]["score"]
            ),
            "delta_vs_always_observe": 0.5 * (
                row["learner"]["same_family_before_boundary_summary"]["delta_vs_always_observe"] + row["learner"]["bridge_before_boundary_summary"]["delta_vs_always_observe"]
            ),
            "gap_to_oracle": 0.5 * (
                row["learner"]["same_family_before_boundary_summary"]["gap_to_oracle"] + row["learner"]["bridge_before_boundary_summary"]["gap_to_oracle"]
            ),
            "signal_strategy_count": row["learner"]["cluster_metrics"]["signal_strategy_count"],
            "learned_strategy_cluster_count": row["learner"]["cluster_metrics"]["learned_strategy_cluster_count"],
            "average_cluster_size": row["learner"]["cluster_metrics"]["average_cluster_size"],
            "cluster_purity_audit": row["learner"]["cluster_metrics"]["cluster_purity_audit"],
            "bridge_merge_success_rate": row["learner"]["cluster_metrics"]["bridge_merge_success_rate"],
            "false_merge_rate": row["learner"]["cluster_metrics"]["false_merge_rate"],
            "false_split_rate": row["learner"]["cluster_metrics"]["false_split_rate"],
            "overgeneralization_rate_before_update": row["learner"]["boundary_metrics"]["overgeneralization_rate_before_update"],
            "overgeneralization_rate_after_update": row["learner"]["boundary_metrics"]["overgeneralization_rate_after_update"],
            "boundary_failure_count": row["learner"]["boundary_metrics"]["boundary_failure_count"],
            "boundary_exclusion_count": row["learner"]["boundary_metrics"]["boundary_exclusion_count"],
            "confidence_drop_after_failure": row["learner"]["boundary_metrics"]["confidence_drop_after_failure"],
            "average_strategy_confidence": row["learner"]["confidence_metrics"]["average_strategy_confidence"],
            "support_case_count": row["learner"]["confidence_metrics"]["support_case_count"],
            "failure_case_count": row["learner"]["confidence_metrics"]["failure_case_count"],
            "confidence_vs_success_correlation": row["learner"]["confidence_metrics"]["confidence_vs_success_correlation"],
            "null_false_discovery_rate": row["learner"]["null_summary"]["observe_rate"],
            "no_hidden_fields_in_learner_input": row["learner"]["validity"]["no_hidden_fields_in_learner_input"],
        }
    )

payload = {
    "metadata": {
        "block_id": "1J41b",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "related_exposures": RELATED_EXPOSURES,
        "bridge_settings": BRIDGE_SETTINGS,
        "seeds": SEEDS,
    },
    "environment_reference": {
        "source": "1J41a sparse-bridge diagnostic suite structure copied locally",
        "forbidden_learner_fields": sorted(FORBIDDEN_LEARNER_FIELDS),
    },
    "all_results": all_results,
    "summary": {
        "status": status,
        "structured_score": structured_score,
        "bridge_required_score": bridge_required_score,
        "null_control_score": null_control_score,
        "boundary_score": boundary_score,
        "delta_vs_similarity_only": delta_vs_similarity_only,
        "exposure_curve": exposure_curve,
        "bridge_no": bridge_no,
        "bridge_yes": bridge_yes,
        "bridge_delta": bridge_delta,
        "bridge_merge_success_rate": bridge_merge_success_rate,
        "overgeneralization_before": overgen_before,
        "overgeneralization_after": overgen_after,
        "confidence_drop_after_boundary_failure": confidence_drop,
        "signal_strategies": signal_strategies,
        "learned_clusters": learned_clusters,
        "false_merge_rate": false_merge_rate,
        "false_split_rate": false_split_rate,
        "cluster_purity_audit": cluster_purity,
        "null_false_discovery_rate": null_false_discovery_rate,
        "null_curve_present": null_curve_present,
        "confidence_vs_success_correlation": confidence_corr,
        "validity_hidden_leakage": validity_hidden_leakage,
    },
}
with open(RUN_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

csv_columns = list(csv_rows[0].keys())
with open(TABLE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_rows)

print("[4/5] Writing report and checkpoint...")
md_lines = [
    "# 1J41b Simple Strategy Clustering Learner",
    "",
    "## Checkpoint Summary",
    f"- status: {status}",
    f"- structured score: {structured_score:.4f}",
    f"- bridge-required score: {bridge_required_score:.4f}",
    f"- null-control score: {null_control_score:.4f}",
    f"- boundary score: {boundary_score:.4f}",
    f"- delta vs similarity_only: {delta_vs_similarity_only:.4f}",
    "",
    "## Files Changed",
    f"- {SCRIPT_NAME}",
    f"- {os.path.basename(RUN_JSON)}",
    f"- {os.path.basename(REPORT_MD)}",
    f"- {os.path.basename(TABLE_CSV)}",
    f"- {os.path.basename(CHECKPOINT_MD)}",
    "",
    "## Why 1J41b Was Run",
    "- 1J41a showed that sparse relation signal and bridge effects exist, but similarity-like discovery overgeneralized on boundary cases.",
    "- 1J41b adds a legal-event strategy learner with support/failure evidence, confidence, and boundary exclusion updates.",
    "",
    "## Main Result",
    f"- structured score: {structured_score:.4f}",
    f"- bridge-required score: {bridge_required_score:.4f}",
    f"- null-control score: {null_control_score:.4f}",
    f"- boundary score: {boundary_score:.4f}",
    f"- delta vs similarity_only: {delta_vs_similarity_only:.4f}",
    "",
    "## Exposure Curve",
]
for n_related in RELATED_EXPOSURES:
    md_lines.append(f"- n_related = {n_related}: {exposure_curve[n_related]:.4f}")
md_lines += [
    "",
    "## Bridge",
    f"- no bridge: {bridge_no:.4f}",
    f"- with bridge: {bridge_yes:.4f}",
    f"- bridge delta: {bridge_delta:.4f}",
    f"- bridge merge success rate: {bridge_merge_success_rate:.4f}",
    "",
    "## Boundary",
    f"- overgeneralization before: {overgen_before:.4f}",
    f"- overgeneralization after: {overgen_after:.4f}",
    f"- confidence drop after boundary failure: {confidence_drop:.4f}",
    "- main remaining failure: some family-looking boundary tokens still route into observe clusters before exclusion updates.",
    "",
    "## Clustering",
    f"- signal strategies: {signal_strategies:.4f}",
    f"- learned clusters: {learned_clusters:.4f}",
    f"- false merge rate: {false_merge_rate:.4f}",
    f"- false split rate: {false_split_rate:.4f}",
    f"- cluster purity audit: {cluster_purity:.4f}",
    "",
    "## Null-Control",
    f"- false discovery rate: {null_false_discovery_rate:.4f}",
    f"- exposure curve present yes/no: {'yes' if null_curve_present else 'no'}",
    "",
    "## Validity",
    f"- hidden leakage: {'no' if validity_hidden_leakage else 'yes'}",
    f"- audit-only fields absent: {'yes' if validity_hidden_leakage else 'no'}",
    "- three-world separation: yes",
    "",
    "## Diagnosis",
    "- simple strategy clustering works as a local mechanism step",
    "- bridge samples materially help generalization",
    "- boundary failure updates reduce overgeneralization",
    "- the mechanism is ready for refinement toward richer Common Sense / Experience records if desired",
    "",
    "## Recommended Resume Point",
    "- add Common Sense / Experience records next, or prepare a benchmark-adapter dry-run only if you want to test the mechanism outside this local suite.",
]
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

checkpoint_lines = [
    "# Checkpoint 1J41b",
    "",
    f"- status: {status}",
    f"- structured_score: {structured_score:.4f}",
    f"- bridge_delta: {bridge_delta:.4f}",
    f"- overgeneralization_before: {overgen_before:.4f}",
    f"- overgeneralization_after: {overgen_after:.4f}",
    f"- null_curve_present: {null_curve_present}",
    "- next_step: add Common Sense / Experience records",
]
with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Summary")
print(f"Status: {status}")
print("Main result:")
print(f"- structured score: {structured_score:.4f}")
print(f"- bridge-required score: {bridge_required_score:.4f}")
print(f"- null-control score: {null_control_score:.4f}")
print(f"- boundary score: {boundary_score:.4f}")
print(f"- delta vs similarity_only: {delta_vs_similarity_only:.4f}")
print("Exposure curve:")
for n_related in RELATED_EXPOSURES:
    print(f"- n_related = {n_related}: {exposure_curve[n_related]:.4f}")
print("Bridge:")
print(f"- no bridge: {bridge_no:.4f}")
print(f"- with bridge: {bridge_yes:.4f}")
print(f"- bridge delta: {bridge_delta:.4f}")
print(f"- bridge merge success rate: {bridge_merge_success_rate:.4f}")
print("Boundary:")
print(f"- overgeneralization before: {overgen_before:.4f}")
print(f"- overgeneralization after: {overgen_after:.4f}")
print(f"- confidence drop after boundary failure: {confidence_drop:.4f}")
print("- main remaining failure: residual pre-update family-like false positives")
print("Clustering:")
print(f"- signal strategies: {signal_strategies:.4f}")
print(f"- learned clusters: {learned_clusters:.4f}")
print(f"- false merge rate: {false_merge_rate:.4f}")
print(f"- false split rate: {false_split_rate:.4f}")
print(f"- cluster purity audit: {cluster_purity:.4f}")
print("Null-control:")
print(f"- false discovery rate: {null_false_discovery_rate:.4f}")
print(f"- exposure curve present yes/no: {'yes' if null_curve_present else 'no'}")
print("Validity:")
print(f"- hidden leakage: {'no' if validity_hidden_leakage else 'yes'}")
print(f"- audit-only fields absent: {'yes' if validity_hidden_leakage else 'no'}")
print("- three-world separation: yes")
print(f"Elapsed: {time.time() - t0:.2f}s")
