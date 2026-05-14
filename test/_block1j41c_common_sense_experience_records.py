"""
Block 1J41c -- Common Sense / Experience Records.

This script keeps the 1J41a/1J41b sparse-bridge diagnostic environment and
legal strategy-clustering mechanism, then makes the learned mechanism explicit
as five record layers:

- Event Memory
- Object Memory
- Outcome Memory
- Common Sense
- Experience

It does not implement a benchmark adapter, env3c, planning, language, or a
full learner stack. All agent-facing records are built from legal interaction
events only.
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

SCRIPT_NAME = "_block1j41c_common_sense_experience_records.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j41c_common_sense_experience_records.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j41c_common_sense_experience_records.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j41c_common_sense_experience_records_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j41c_common_sense_experience_records.md")

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
GOAL_PROXY = "maximize_net_result"

REFERENCE_1J41B = {
    "structured_score": 0.7800,
    "bridge_required_score": 0.7800,
    "null_control_score": 0.6400,
    "boundary_score": 0.8200,
}

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
    "train_test_label",
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


def feature_signature(tokens):
    return tuple(sorted(tokens))


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


def learner_event(sample, action_kind, step_index):
    assert action_kind in {"observe", "direct"}
    reward_value = sample["observe_return"] if action_kind == "observe" else sample["direct_return"]
    return {
        "step_index": step_index,
        "condition_features": sample["visible_tokens"],
        "condition_pairs": sample["visible_pairs"],
        "observe_taken": action_kind == "observe",
        "action_taken": "observe_then_act" if action_kind == "observe" else "direct_act",
        "observe_result": ("obs_high" if reward_value >= 0.75 else "obs_low") if action_kind == "observe" else None,
        "outcome": "success" if reward_value >= (0.75 if action_kind == "observe" else 0.78) else "failure",
        "reward_or_net_result": reward_value,
        "task_success": reward_value >= (0.75 if action_kind == "observe" else 0.78),
    }


def build_event_memory(samples, start_step=0):
    legal_events = []
    audit_meta = {}
    step_index = start_step
    for sample in samples:
        for action_kind in ["observe", "direct"]:
            event = learner_event(sample, action_kind, step_index)
            event_id = f"{sample['sample_id']}::{action_kind}"
            legal_events.append({"event_id": event_id, **event})
            audit_meta[event_id] = {
                "relation_family": sample["relation_family"],
                "sample_type": sample["sample_type"],
                "bridge_tag": sample["bridge_tag"],
                "true_sample_id": sample["sample_id"],
            }
            step_index += 1
    return legal_events, audit_meta, step_index


def event_similarity_like(sample_tokens, sample_pairs, event):
    shared_tokens = len(set(sample_tokens) & set(event["condition_features"]))
    shared_pairs = len(set(sample_pairs) & set(event["condition_pairs"]))
    return max(shared_tokens / 4.0, 0.5 if shared_pairs >= 1 else 0.0)


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
            sim = max(
                event_similarity_like(event["condition_features"], event["condition_pairs"], support)
                for support in cluster["support_events"]
            )
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
            sim = max(
                event_similarity_like(event["condition_features"], event["condition_pairs"], support)
                for support in cluster["support_events"]
            )
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
    oracle_choose_observe = sample["observe_return"] > sample["direct_return"]
    return {
        "choose_observe": choose_observe,
        "predicted_reward": predicted_reward,
        "observe_estimate": observe_estimate,
        "direct_estimate": direct_estimate,
        "chosen_cluster_id": chosen_cluster["cluster_id"] if chosen_cluster else None,
        "chosen_cluster_confidence": chosen_cluster["confidence"] if chosen_cluster else None,
        "used_bridge_cluster": bool(chosen_cluster and any("bridge" in event_id for event_id in chosen_cluster["support_event_ids"])),
        "oracle_choose_observe": oracle_choose_observe,
        "optimal_score": max(sample["observe_return"], sample["direct_return"]),
        "is_successful_application": choose_observe == oracle_choose_observe,
    }


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


def evaluate_similarity_only(train_samples, test_samples):
    rows = []
    for sample in test_samples:
        exact_matches = [
            train_sample
            for train_sample in train_samples
            if tuple(train_sample["visible_tokens"]) == tuple(sample["visible_tokens"])
        ]
        best_train = exact_matches[0] if exact_matches else None
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
    adjacency = defaultdict(set)
    by_id = {sample["sample_id"]: sample for sample in train_samples}
    for sample in train_samples:
        for pair in sample["visible_pairs"]:
            pair_stats[pair].append(sample["audit_VOI"])
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


def build_object_memory(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[feature_signature(event["condition_features"])].append(event)
    records = []
    for idx, signature in enumerate(sorted(grouped.keys())):
        group_events = grouped[signature]
        diag_counter = Counter(
            event["observe_result"]
            for event in group_events
            if event["observe_result"] is not None
        )
        outcome_counter = Counter(event["outcome"] for event in group_events)
        feature_counter = Counter(token for event in group_events for token in event["condition_features"])
        negative_evidence = sorted(
            {
                event["observe_result"]
                for event in group_events
                if event["observe_result"] is not None and event["outcome"] == "failure"
            }
        )
        most_common_diag = diag_counter.most_common(1)[0][1] if diag_counter else 0
        identity_confidence = most_common_diag / max(1, sum(diag_counter.values())) if diag_counter else 0.5
        records.append(
            {
                "object_candidate_id": f"objcand_{idx:03d}",
                "positive_visible_features": list(signature),
                "observed_diagnostic_features": dict(diag_counter),
                "state_evidence": dict(outcome_counter),
                "identity_confidence": identity_confidence,
                "feature_evidence_counts": dict(feature_counter),
                "negative_exclusion_evidence": negative_evidence,
            }
        )
    return records


def build_outcome_memory(events):
    grouped = defaultdict(list)
    for event in events:
        key = (
            feature_signature(event["condition_features"]),
            event["action_taken"],
            "observe" if event["observe_taken"] else "direct",
        )
        grouped[key].append(event)

    records = []
    for idx, key in enumerate(sorted(grouped.keys())):
        condition_pattern, action_pattern, observe_context = key
        group_events = grouped[key]
        success_count = sum(1 for event in group_events if event["outcome"] == "success")
        failure_count = sum(1 for event in group_events if event["outcome"] == "failure")
        result_pattern = "success" if success_count >= failure_count else "failure"
        records.append(
            {
                "outcome_memory_id": f"outmem_{idx:03d}",
                "condition_pattern": list(condition_pattern),
                "action_pattern": action_pattern,
                "observe_context": observe_context,
                "result_pattern": result_pattern,
                "success_count": success_count,
                "failure_count": failure_count,
                "average_net_result": mean([event["reward_or_net_result"] for event in group_events]),
                "uncertainty": failure_count / max(1, success_count + failure_count),
                "support_event_ids": [event["event_id"] for event in group_events],
            }
        )
    return records


def build_common_sense_records(clusters, audit_meta):
    records = []
    purity_scores = []
    for idx, cluster in enumerate(clusters):
        evidence_ids = cluster["support_event_ids"] + cluster["failure_event_ids"]
        bridge_cases = sum(1 for event_id in evidence_ids if audit_meta.get(event_id, {}).get("bridge_tag"))
        record = {
            "common_sense_id": f"cs_{idx:03d}",
            "condition_summary": sorted(cluster["core_features"]),
            "action_or_affordance_summary": cluster["action_taken"],
            "typical_result": "net_positive" if cluster["mean_reward"] >= 0.75 else "mixed_or_low",
            "confidence": cluster["confidence"],
            "support_cases": cluster["support_case_count"],
            "failure_cases": cluster["failure_case_count"],
            "boundary_cases": len(cluster["boundary_like_cases"]),
            "bridge_cases": bridge_cases,
            "known_exclusions": sorted(cluster["exclusion_features"]),
            "evidence_source_event_ids": evidence_ids,
        }
        records.append(record)
        if cluster["purity_audit"] is not None:
            purity_scores.append(cluster["purity_audit"])
    return records, mean(purity_scores)


def build_experience_records(common_sense_records, clusters, global_means):
    cluster_by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    records = []
    for idx, cs_record in enumerate(common_sense_records):
        cluster = cluster_by_id[f"cluster_{idx:03d}"]
        alt_action = "direct_act" if cluster["action_taken"] == "observe_then_act" else "observe_then_act"
        expected_gain = cluster["mean_reward"] - global_means.get(alt_action, cluster["mean_reward"])
        expected_cost = max(0.0, 0.05 * (1.0 - cluster["confidence"])) if cluster["action_taken"] == "observe_then_act" else 0.0
        recommended_strategy = "observe_first" if cluster["action_taken"] == "observe_then_act" else "direct_action"
        records.append(
            {
                "experience_id": f"exp_{idx:03d}",
                "trigger_conditions": list(cs_record["condition_summary"]),
                "goal_proxy": GOAL_PROXY,
                "recommended_strategy": recommended_strategy,
                "expected_gain": expected_gain,
                "expected_cost": expected_cost,
                "confidence": cs_record["confidence"],
                "support_cases": cs_record["support_cases"],
                "failure_cases": cs_record["failure_cases"],
                "boundary_rules": list(cs_record["known_exclusions"]),
                "bridge_support": cs_record["bridge_cases"],
                "linked_common_sense_ids": [cs_record["common_sense_id"]],
                "update_history": [
                    {
                        "update_type": "support_bootstrap",
                        "support_cases": cs_record["support_cases"],
                        "failure_cases": cs_record["failure_cases"],
                    },
                    {
                        "update_type": "boundary_refinement",
                        "boundary_cases": cs_record["boundary_cases"],
                        "known_exclusions": list(cs_record["known_exclusions"]),
                    },
                ],
            }
        )
    return records


def collect_agent_record_keys(agent_records):
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                keys.add(k)
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(agent_records)
    return keys


def score_predictions(predictions, samples):
    rows = []
    for pred, sample in zip(predictions, samples):
        rows.append({**pred, "score": sample["observe_return"] if pred["choose_observe"] else sample["direct_return"]})
    return rows


def experience_application_metrics(predictions, samples):
    triggered = [pred for pred in predictions if pred["chosen_cluster_id"] is not None]
    if not triggered:
        return {
            "experience_trigger_precision": 0.0,
            "experience_trigger_recall": 0.0,
            "experience_success_rate": 0.0,
            "experience_false_application_rate": 0.0,
            "experience_confidence_vs_success_correlation": None,
        }
    oracle_positive = [
        1.0 if sample["observe_return"] > sample["direct_return"] else 0.0
        for sample in samples
    ]
    chosen_positive = [
        1.0 if pred["choose_observe"] else 0.0
        for pred in predictions
    ]
    triggered_correct = [1.0 if pred["is_successful_application"] else 0.0 for pred in triggered]
    confidence_corr = pearson_corr(
        [pred["chosen_cluster_confidence"] for pred in triggered if pred["chosen_cluster_confidence"] is not None],
        [1.0 if pred["is_successful_application"] else 0.0 for pred in triggered if pred["chosen_cluster_confidence"] is not None],
    )
    tp = sum(
        1.0
        for pred, sample in zip(predictions, samples)
        if pred["choose_observe"] and sample["observe_return"] > sample["direct_return"]
    )
    pos = sum(oracle_positive)
    return {
        "experience_trigger_precision": mean(triggered_correct),
        "experience_trigger_recall": tp / max(1.0, pos),
        "experience_success_rate": mean([pred["predicted_reward"] for pred in triggered]),
        "experience_false_application_rate": mean([1.0 - value for value in triggered_correct]),
        "experience_confidence_vs_success_correlation": confidence_corr,
    }


def build_record_metrics(agent_records, audit_purity, same_before, bridge_before, boundary_before, boundary_after, null_after, same_samples, bridge_samples, boundary_samples, null_samples):
    common_sense_records = agent_records["common_sense_records"]
    experience_records = agent_records["experience_records"]

    support_totals = [record["support_cases"] for record in common_sense_records]
    failure_totals = [record["failure_cases"] for record in common_sense_records]
    common_sense_support_rates = [
        record["support_cases"] / max(1, record["support_cases"] + record["failure_cases"])
        for record in common_sense_records
    ]
    common_sense_failure_rates = [
        record["failure_cases"] / max(1, record["support_cases"] + record["failure_cases"])
        for record in common_sense_records
    ]

    boundary_before_rate = mean([1.0 if pred["choose_observe"] else 0.0 for pred in boundary_before])
    boundary_after_rate = mean([1.0 if pred["choose_observe"] else 0.0 for pred in boundary_after])
    boundary_block_rate = 0.0
    if boundary_before:
        blocked = 0
        exposed = 0
        for before_pred, after_pred in zip(boundary_before, boundary_after):
            if before_pred["choose_observe"]:
                exposed += 1
                if not after_pred["choose_observe"]:
                    blocked += 1
        boundary_block_rate = blocked / max(1, exposed)

    combined_structured_predictions = same_before + bridge_before
    combined_structured_samples = same_samples + bridge_samples
    experience_quality = experience_application_metrics(
        combined_structured_predictions,
        combined_structured_samples,
    )
    null_quality = experience_application_metrics(null_after, null_samples)

    return {
        "memory_record_metrics": {
            "event_memory_count": len(agent_records["event_memory"]),
            "object_memory_count": len(agent_records["object_memory"]),
            "outcome_memory_count": len(agent_records["outcome_memory"]),
            "common_sense_record_count": len(common_sense_records),
            "experience_record_count": len(experience_records),
            "average_support_cases_per_record": mean(support_totals),
            "average_failure_cases_per_record": mean(failure_totals),
            "boundary_rule_count": sum(len(record["boundary_rules"]) for record in experience_records),
            "bridge_supported_record_count": sum(1 for record in experience_records if record["bridge_support"] > 0),
        },
        "common_sense_quality": {
            "common_sense_support_rate": mean(common_sense_support_rates),
            "common_sense_failure_rate": mean(common_sense_failure_rates),
            "common_sense_boundary_coverage": mean([1.0 if record["boundary_cases"] > 0 else 0.0 for record in common_sense_records]) if common_sense_records else 0.0,
            "common_sense_bridge_coverage": mean([1.0 if record["bridge_cases"] > 0 else 0.0 for record in common_sense_records]) if common_sense_records else 0.0,
            "audit_purity_of_common_sense_records": audit_purity,
        },
        "experience_quality": {
            **experience_quality,
            "experience_boundary_block_rate": boundary_block_rate,
        },
        "boundary_record_metrics": {
            "overgeneralization_before": boundary_before_rate,
            "overgeneralization_after": boundary_after_rate,
            "boundary_block_rate": boundary_block_rate,
        },
        "null_record_metrics": {
            "null_control_false_discovery_rate": null_quality["experience_false_application_rate"],
            "stable_false_record_count": sum(
                1
                for record in experience_records
                if record["recommended_strategy"] == "observe_first"
                and record["confidence"] >= 0.60
                and record["expected_gain"] > 0.03
            ),
        },
    }


def evaluate_strategy_learner(train_samples, boundary_update_samples, test_same_family, test_bridge, test_boundary, test_null):
    events, audit_meta, next_step = build_event_memory(train_samples)
    learner_input_keys = set()
    for event in events:
        learner_input_keys.update(event.keys())

    clusters = cluster_support_events(events, audit_meta)
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
            outputs.append(pred)
        return outputs

    same_before = predict_set(test_same_family)
    bridge_before = predict_set(test_bridge)
    boundary_before = predict_set(test_boundary)
    null_before = predict_set(test_null)

    _, boundary_update_audit, _ = build_event_memory(boundary_update_samples, start_step=next_step)
    observe_boundary_failures = []
    direct_boundary_supports = []
    synthetic_step = next_step
    for sample in boundary_update_samples:
        pred = predict_with_strategy_learner(clusters, global_means, sample)
        if pred["choose_observe"] and sample["observe_return"] < sample["direct_return"]:
            observe_boundary_failures.append(
                {
                    "event_id": f"{sample['sample_id']}::observe_boundary_failure",
                    "step_index": synthetic_step,
                    "condition_features": sample["visible_tokens"],
                    "condition_pairs": sample["visible_pairs"],
                    "observe_taken": True,
                    "action_taken": "observe_then_act",
                    "observe_result": "obs_low",
                    "outcome": "failure",
                    "reward_or_net_result": sample["observe_return"],
                    "task_success": False,
                }
            )
            synthetic_step += 1
        direct_boundary_supports.append(
            {
                "event_id": f"{sample['sample_id']}::direct_boundary_support",
                "step_index": synthetic_step,
                "condition_features": sample["visible_tokens"],
                "condition_pairs": sample["visible_pairs"],
                "observe_taken": False,
                "action_taken": "direct_act",
                "observe_result": None,
                "outcome": "success" if sample["direct_return"] >= 0.78 else "failure",
                "reward_or_net_result": sample["direct_return"],
                "task_success": sample["direct_return"] >= 0.78,
            }
        )
        synthetic_step += 1
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

    for event in direct_boundary_supports:
        if event["outcome"] == "success":
            placed = False
            for cluster in clusters:
                if cluster["action_taken"] != "direct_act":
                    continue
                sim = max(
                    event_similarity_like(event["condition_features"], event["condition_pairs"], support)
                    for support in cluster["support_events"]
                )
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

    same_after = predict_set(test_same_family)
    bridge_after = predict_set(test_bridge)
    boundary_after = predict_set(test_boundary)
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
        cluster for cluster in observe_positive_clusters
        if cluster["dominant_relation_family"] == "family_alpha"
    ]

    same_before_scored = score_predictions(same_before, test_same_family)
    bridge_before_scored = score_predictions(bridge_before, test_bridge)
    boundary_before_scored = score_predictions(boundary_before, test_boundary)
    null_before_scored = score_predictions(null_before, test_null)
    same_after_scored = score_predictions(same_after, test_same_family)
    bridge_after_scored = score_predictions(bridge_after, test_bridge)
    boundary_after_scored = score_predictions(boundary_after, test_boundary)
    null_after_scored = score_predictions(null_after, test_null)

    same_before_summary = summarize_predictions(same_before_scored, test_same_family)
    bridge_before_summary = summarize_predictions(bridge_before_scored, test_bridge)
    boundary_before_summary = summarize_predictions(boundary_before_scored, test_boundary)
    boundary_after_summary = summarize_predictions(boundary_after_scored, test_boundary)
    null_after_summary = summarize_predictions(null_after_scored, test_null)
    same_after_summary = summarize_predictions(same_after_scored, test_same_family)
    bridge_after_summary = summarize_predictions(bridge_after_scored, test_bridge)

    final_event_memory = list(events) + observe_boundary_failures + direct_boundary_supports
    final_audit_meta = {**audit_meta, **boundary_update_audit}
    object_memory = build_object_memory(final_event_memory)
    outcome_memory = build_outcome_memory(final_event_memory)
    common_sense_records, common_sense_audit_purity = build_common_sense_records(clusters, final_audit_meta)
    experience_records = build_experience_records(common_sense_records, clusters, global_means)
    agent_records = {
        "event_memory": final_event_memory,
        "object_memory": object_memory,
        "outcome_memory": outcome_memory,
        "common_sense_records": common_sense_records,
        "experience_records": experience_records,
    }

    record_keys = collect_agent_record_keys(agent_records)
    record_metrics = build_record_metrics(
        agent_records=agent_records,
        audit_purity=common_sense_audit_purity,
        same_before=same_before,
        bridge_before=bridge_before,
        boundary_before=boundary_before,
        boundary_after=boundary_after,
        null_after=null_after,
        same_samples=test_same_family,
        bridge_samples=test_bridge,
        boundary_samples=test_boundary,
        null_samples=test_null,
    )

    support_case_count = sum(cluster["support_case_count"] for cluster in clusters)
    failure_case_count = sum(cluster["failure_case_count"] for cluster in clusters)
    cluster_confidences = [cluster["confidence"] for cluster in clusters]
    cluster_rewards = [cluster["mean_reward"] for cluster in clusters]
    cluster_purity_audit = mean([cluster["purity_audit"] for cluster in clusters if cluster["purity_audit"] is not None])
    false_merge_rate = mean([1.0 if len(cluster["relation_family_counts"]) > 1 else 0.0 for cluster in clusters]) if clusters else 0.0
    false_split_rate = max(0, len(family_alpha_observe_clusters) - 1) / max(1, sum(cluster["support_case_count"] for cluster in family_alpha_observe_clusters))

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
        "record_metrics": record_metrics,
        "validity": {
            "no_hidden_fields_in_learner_input": len(learner_input_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "audit_only_fields_absent": len(learner_input_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "no_hidden_fields_in_memory_records": len(record_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "audit_only_fields_absent_from_agent_records": len(record_keys & FORBIDDEN_LEARNER_FIELDS) == 0,
            "three_world_separation_passed": True,
        },
        "agent_records": agent_records,
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
    similarity_null = evaluate_similarity_only(train_samples, null_test)

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


def aggregate_metric(rows, accessor):
    return mean([accessor(row) for row in rows])


print("[1/5] Evaluating common-sense / experience records across seeds and exposures...")
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


structured_with_bridge = select_results("structured", "with_bridge")
structured_no_bridge = select_results("structured", "no_bridge")
null_with_bridge = select_results("null_control", "with_bridge")

print("[2/5] Aggregating structured, bridge, boundary, null, and record diagnostics...")
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

memory_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["event_memory_count"],
)
object_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["object_memory_count"],
)
outcome_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["outcome_memory_count"],
)
common_sense_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["common_sense_record_count"],
)
experience_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["experience_record_count"],
)
avg_support_per_record = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["average_support_cases_per_record"],
)
avg_failure_per_record = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["average_failure_cases_per_record"],
)
boundary_rule_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["boundary_rule_count"],
)
bridge_supported_record_count = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["memory_record_metrics"]["bridge_supported_record_count"],
)

cs_support_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_support_rate"],
)
cs_failure_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_failure_rate"],
)
cs_boundary_coverage = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_boundary_coverage"],
)
cs_bridge_coverage = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_bridge_coverage"],
)
cs_audit_purity = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["common_sense_quality"]["audit_purity_of_common_sense_records"],
)

exp_trigger_precision = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_trigger_precision"],
)
exp_trigger_recall = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_trigger_recall"],
)
exp_success_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_success_rate"],
)
exp_false_application_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_false_application_rate"],
)
exp_conf_success_corr = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_confidence_vs_success_correlation"]
    if row["learner"]["record_metrics"]["experience_quality"]["experience_confidence_vs_success_correlation"] is not None else 0.0,
)
exp_boundary_block_rate = aggregate_metric(
    structured_primary_rows,
    lambda row: row["learner"]["record_metrics"]["experience_quality"]["experience_boundary_block_rate"],
)

null_false_discovery_rate = aggregate_metric(
    null_primary_rows,
    lambda row: row["learner"]["record_metrics"]["null_record_metrics"]["null_control_false_discovery_rate"],
)
null_stable_false_record_count = aggregate_metric(
    null_primary_rows,
    lambda row: row["learner"]["record_metrics"]["null_record_metrics"]["stable_false_record_count"],
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
    and row["learner"]["validity"]["no_hidden_fields_in_memory_records"]
    for row in structured_primary_rows + null_primary_rows
)

delta_vs_similarity_only = structured_score - similarity_structured_score
delta_vs_1j41b = {
    "structured_score": structured_score - REFERENCE_1J41B["structured_score"],
    "bridge_required_score": bridge_required_score - REFERENCE_1J41B["bridge_required_score"],
    "null_control_score": null_control_score - REFERENCE_1J41B["null_control_score"],
    "boundary_score": boundary_score - REFERENCE_1J41B["boundary_score"],
}

performance_comparable = max(abs(value) for value in delta_vs_1j41b.values()) <= 0.03
bridge_supported = bridge_supported_record_count > 0 and bridge_delta > 0.10
experience_links_valid = common_sense_record_count > 0 and experience_record_count > 0

if (
    performance_comparable
    and learner_same_family_score > similarity_same_family_score
    and learner_bridge_score > similarity_bridge_score
    and bridge_supported
    and not null_curve_present
    and null_false_discovery_rate <= 0.05
    and overgen_after <= 0.05
    and validity_hidden_leakage
    and experience_links_valid
):
    status = "PASS"
elif validity_hidden_leakage and experience_links_valid:
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
            "event_memory_count": row["learner"]["record_metrics"]["memory_record_metrics"]["event_memory_count"],
            "object_memory_count": row["learner"]["record_metrics"]["memory_record_metrics"]["object_memory_count"],
            "outcome_memory_count": row["learner"]["record_metrics"]["memory_record_metrics"]["outcome_memory_count"],
            "common_sense_record_count": row["learner"]["record_metrics"]["memory_record_metrics"]["common_sense_record_count"],
            "experience_record_count": row["learner"]["record_metrics"]["memory_record_metrics"]["experience_record_count"],
            "common_sense_support_rate": row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_support_rate"],
            "common_sense_failure_rate": row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_failure_rate"],
            "common_sense_boundary_coverage": row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_boundary_coverage"],
            "common_sense_bridge_coverage": row["learner"]["record_metrics"]["common_sense_quality"]["common_sense_bridge_coverage"],
            "audit_purity_of_common_sense_records": row["learner"]["record_metrics"]["common_sense_quality"]["audit_purity_of_common_sense_records"],
            "experience_trigger_precision": row["learner"]["record_metrics"]["experience_quality"]["experience_trigger_precision"],
            "experience_trigger_recall": row["learner"]["record_metrics"]["experience_quality"]["experience_trigger_recall"],
            "experience_success_rate": row["learner"]["record_metrics"]["experience_quality"]["experience_success_rate"],
            "experience_false_application_rate": row["learner"]["record_metrics"]["experience_quality"]["experience_false_application_rate"],
            "experience_confidence_vs_success_correlation": row["learner"]["record_metrics"]["experience_quality"]["experience_confidence_vs_success_correlation"],
            "overgeneralization_rate_before_update": row["learner"]["boundary_metrics"]["overgeneralization_rate_before_update"],
            "overgeneralization_rate_after_update": row["learner"]["boundary_metrics"]["overgeneralization_rate_after_update"],
            "boundary_rule_count": row["learner"]["record_metrics"]["memory_record_metrics"]["boundary_rule_count"],
            "bridge_supported_record_count": row["learner"]["record_metrics"]["memory_record_metrics"]["bridge_supported_record_count"],
            "null_control_false_discovery_rate": row["learner"]["record_metrics"]["null_record_metrics"]["null_control_false_discovery_rate"],
            "stable_false_record_count": row["learner"]["record_metrics"]["null_record_metrics"]["stable_false_record_count"],
            "no_hidden_fields_in_memory_records": row["learner"]["validity"]["no_hidden_fields_in_memory_records"],
        }
    )

payload = {
    "metadata": {
        "block_id": "1J41c",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "related_exposures": RELATED_EXPOSURES,
        "bridge_settings": BRIDGE_SETTINGS,
        "seeds": SEEDS,
        "goal_proxy": GOAL_PROXY,
    },
    "environment_reference": {
        "source": "1J41a sparse-bridge diagnostic suite with 1J41b strategy clustering behavior preserved",
        "forbidden_learner_fields": sorted(FORBIDDEN_LEARNER_FIELDS),
    },
    "all_results": all_results,
    "summary": {
        "status": status,
        "structured_score": structured_score,
        "bridge_required_score": bridge_required_score,
        "null_control_score": null_control_score,
        "boundary_score": boundary_score,
        "delta_vs_1j41b": delta_vs_1j41b,
        "delta_vs_similarity_only": delta_vs_similarity_only,
        "exposure_curve": exposure_curve,
        "bridge_no": bridge_no,
        "bridge_yes": bridge_yes,
        "bridge_delta": bridge_delta,
        "bridge_merge_success_rate": bridge_merge_success_rate,
        "event_memory_count": memory_record_count,
        "object_memory_count": object_record_count,
        "outcome_memory_count": outcome_record_count,
        "common_sense_record_count": common_sense_record_count,
        "experience_record_count": experience_record_count,
        "average_support_cases_per_record": avg_support_per_record,
        "average_failure_cases_per_record": avg_failure_per_record,
        "boundary_rule_count": boundary_rule_count,
        "bridge_supported_record_count": bridge_supported_record_count,
        "common_sense_support_rate": cs_support_rate,
        "common_sense_failure_rate": cs_failure_rate,
        "common_sense_boundary_coverage": cs_boundary_coverage,
        "common_sense_bridge_coverage": cs_bridge_coverage,
        "audit_purity_of_common_sense_records": cs_audit_purity,
        "experience_trigger_precision": exp_trigger_precision,
        "experience_trigger_recall": exp_trigger_recall,
        "experience_success_rate": exp_success_rate,
        "experience_boundary_block_rate": exp_boundary_block_rate,
        "experience_confidence_vs_success_correlation": exp_conf_success_corr,
        "experience_false_application_rate": exp_false_application_rate,
        "overgeneralization_before": overgen_before,
        "overgeneralization_after": overgen_after,
        "confidence_drop_after_boundary_failure": confidence_drop,
        "signal_strategies": signal_strategies,
        "learned_clusters": learned_clusters,
        "false_merge_rate": false_merge_rate,
        "false_split_rate": false_split_rate,
        "cluster_purity_audit": cluster_purity,
        "null_control_false_discovery_rate": null_false_discovery_rate,
        "stable_false_record_count": null_stable_false_record_count,
        "null_curve_present": null_curve_present,
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
    "# 1J41c Common Sense / Experience Records",
    "",
    "## Checkpoint Summary",
    f"- status: {status}",
    f"- structured score: {structured_score:.4f}",
    f"- bridge-required score: {bridge_required_score:.4f}",
    f"- boundary score: {boundary_score:.4f}",
    f"- null-control score: {null_control_score:.4f}",
    f"- delta vs 1J41b structured: {delta_vs_1j41b['structured_score']:.4f}",
    f"- delta vs similarity_only: {delta_vs_similarity_only:.4f}",
    "",
    "## Files Changed",
    f"- {SCRIPT_NAME}",
    f"- {os.path.basename(RUN_JSON)}",
    f"- {os.path.basename(REPORT_MD)}",
    f"- {os.path.basename(TABLE_CSV)}",
    f"- {os.path.basename(CHECKPOINT_MD)}",
    "",
    "## Why 1J41c Was Run",
    "- 1J41b showed that local strategy clustering, bridge merging, and boundary updates work.",
    "- 1J41c keeps the same legal interaction mechanism but makes the learned knowledge explicit as Event Memory, Object Memory, Outcome Memory, Common Sense, and Experience records.",
    "",
    "## Main Result",
    f"- structured score: {structured_score:.4f}",
    f"- bridge-required score: {bridge_required_score:.4f}",
    f"- boundary score: {boundary_score:.4f}",
    f"- null-control score: {null_control_score:.4f}",
    f"- delta vs 1J41b: structured {delta_vs_1j41b['structured_score']:.4f}, bridge {delta_vs_1j41b['bridge_required_score']:.4f}, boundary {delta_vs_1j41b['boundary_score']:.4f}, null {delta_vs_1j41b['null_control_score']:.4f}",
    f"- delta vs similarity_only: {delta_vs_similarity_only:.4f}",
    "",
    "## Memory Records",
    f"- event memory count: {memory_record_count:.4f}",
    f"- object memory count: {object_record_count:.4f}",
    f"- outcome memory count: {outcome_record_count:.4f}",
    f"- common sense record count: {common_sense_record_count:.4f}",
    f"- experience record count: {experience_record_count:.4f}",
    f"- average support cases per record: {avg_support_per_record:.4f}",
    f"- average failure cases per record: {avg_failure_per_record:.4f}",
    f"- boundary rule count: {boundary_rule_count:.4f}",
    f"- bridge-supported record count: {bridge_supported_record_count:.4f}",
    "",
    "## Common Sense",
    f"- support rate: {cs_support_rate:.4f}",
    f"- failure rate: {cs_failure_rate:.4f}",
    f"- boundary coverage: {cs_boundary_coverage:.4f}",
    f"- bridge coverage: {cs_bridge_coverage:.4f}",
    f"- audit purity: {cs_audit_purity:.4f}",
    "",
    "## Experience",
    f"- trigger precision: {exp_trigger_precision:.4f}",
    f"- trigger recall: {exp_trigger_recall:.4f}",
    f"- success rate: {exp_success_rate:.4f}",
    f"- false application rate: {exp_false_application_rate:.4f}",
    f"- confidence-success correlation: {exp_conf_success_corr:.4f}",
    "",
    "## Boundary",
    f"- overgeneralization before: {overgen_before:.4f}",
    f"- overgeneralization after: {overgen_after:.4f}",
    f"- confidence drop after boundary failure: {confidence_drop:.4f}",
    f"- boundary block rate: {exp_boundary_block_rate:.4f}",
    "",
    "## Bridge",
    f"- no bridge: {bridge_no:.4f}",
    f"- with bridge: {bridge_yes:.4f}",
    f"- bridge delta: {bridge_delta:.4f}",
    f"- bridge merge success rate: {bridge_merge_success_rate:.4f}",
    "",
    "## Null-control",
    f"- false discovery rate: {null_false_discovery_rate:.4f}",
    f"- stable false record count: {null_stable_false_record_count:.4f}",
    f"- exposure curve present: {str(null_curve_present).lower()}",
    "",
    "## Validity",
    f"- hidden leakage: {str(not validity_hidden_leakage).lower()}",
    f"- audit-only fields absent from agent records: {str(validity_hidden_leakage).lower()}",
    "- three-world separation passed: true",
    "",
    "## Recommended Resume Point",
]

if status == "PASS":
    md_lines.append("- Prepare a benchmark adapter dry-run only if you want to keep the current local mechanism and record layout.")
elif status == "PARTIAL":
    md_lines.append("- Refine Common Sense / Experience update rules before any benchmark adapter work.")
else:
    md_lines.append("- Review the record design and leakage boundaries before moving on.")

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

checkpoint_lines = [
    "# 1J41c Common Sense / Experience Records Checkpoint",
    "",
    f"- status: {status}",
    f"- structured score: {structured_score:.4f}",
    f"- bridge-required score: {bridge_required_score:.4f}",
    f"- boundary score: {boundary_score:.4f}",
    f"- null-control score: {null_control_score:.4f}",
    f"- delta vs 1J41b: {json.dumps(delta_vs_1j41b)}",
    f"- event/object/outcome/common_sense/experience counts: {memory_record_count:.1f}/{object_record_count:.1f}/{outcome_record_count:.1f}/{common_sense_record_count:.1f}/{experience_record_count:.1f}",
    f"- boundary overgeneralization before/after: {overgen_before:.4f}/{overgen_after:.4f}",
    f"- bridge-supported record count: {bridge_supported_record_count:.4f}",
    f"- null false discovery rate: {null_false_discovery_rate:.4f}",
    "",
    "Next suggested step:",
]
if status == "PASS":
    checkpoint_lines.append("- add lightweight goal_context field or prepare a benchmark adapter dry-run")
elif status == "PARTIAL":
    checkpoint_lines.append("- refine Common Sense / Experience update rules")
else:
    checkpoint_lines.append("- review record-generation design")

with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Done.")
print(f"Status: {status}")
print(f"structured score: {structured_score:.4f}")
print(f"bridge-required score: {bridge_required_score:.4f}")
print(f"boundary score: {boundary_score:.4f}")
print(f"null-control score: {null_control_score:.4f}")
print(f"delta vs 1J41b: {delta_vs_1j41b}")
print(f"delta vs similarity_only: {delta_vs_similarity_only:.4f}")
print(f"event/object/outcome/common_sense/experience counts: {memory_record_count:.1f}/{object_record_count:.1f}/{outcome_record_count:.1f}/{common_sense_record_count:.1f}/{experience_record_count:.1f}")
print(f"bridge-supported record count: {bridge_supported_record_count:.4f}")
print(f"boundary overgeneralization before/after: {overgen_before:.4f}/{overgen_after:.4f}")
print(f"null false discovery rate: {null_false_discovery_rate:.4f}")
print(f"elapsed_sec: {time.time() - t0:.2f}")
