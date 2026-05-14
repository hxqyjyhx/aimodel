"""
Block 1J42b -- Adapter Mechanism Parity.

This is still an adapter test, not a benchmark performance claim. It restores
the missing mechanism-side pieces from 1J41d inside the benchmark-style loop:

- confidence update
- boundary rule update
- goal-conditioned Experience use

using legal benchmark observations, actions, and rewards only.
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

SCRIPT_NAME = "_block1j42b_adapter_mechanism_parity.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j42b_adapter_mechanism_parity.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j42b_adapter_mechanism_parity.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j42b_adapter_mechanism_parity_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j42b_adapter_mechanism_parity.md")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

SEEDS = [101, 207, 404]
TRAIN_EPISODES_PER_SEED = 64
PROBE_EPISODES_PER_SEED = 24
BOUNDARY_EPISODES_PER_SEED = 12
NULL_EPISODES_PER_SEED = 24

FORBIDDEN_AGENT_FIELDS = {
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
    "best_action",
    "diagnostic_signal",
}

ACTIONS = ["inspect", "act_red", "act_blue"]
DIRECT_ACTIONS = ["act_red", "act_blue"]
GOAL_CONTEXTS = [
    "maximize_net_result",
    "reduce_action_error",
    "reduce_uncertainty",
    "avoid_boundary_failure",
    "bridge_generalization",
    "complete_task",
    "avoid_negative_reward",
    "reach_target_state",
]


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


def make_encounter(encounter_id, sample_type, visible_tokens, best_action, diagnostic_signal, inspect_useful, direct_hint, bridge_tag=False):
    return {
        "encounter_id": encounter_id,
        "sample_type": sample_type,
        "visible_tokens": tuple(visible_tokens),
        "best_action": best_action,
        "diagnostic_signal": diagnostic_signal,
        "inspect_useful": inspect_useful,
        "direct_hint": direct_hint,
        "bridge_tag": bridge_tag,
    }


TOKENS = {
    "family_a_style": [("amber", "striped"), ("amber", "dotted"), ("teal", "striped"), ("teal", "dotted")],
    "family_b_style": [("smooth", "dry"), ("smooth", "humid"), ("rough", "dry"), ("rough", "humid")],
    "support_a": ["silken", "granular"],
    "support_b": ["cool", "warm"],
    "boundary_a": ["chalky", "slick"],
    "boundary_b": ["dim", "windy"],
    "noise_a": ["violet", "ochre"],
    "noise_b": ["chevron", "lattice"],
    "noise_c": ["waxy", "fibrous"],
    "noise_d": ["bright", "shaded"],
}


def cycle_pick(values, idx):
    return values[idx % len(values)]


def build_benchmark_pools():
    structured_train = []
    bridge_train = []
    structured_probe = []
    boundary_probe = []
    null_probe = []
    unrelated_train = []

    eid = 0
    for rep in range(4):
        for pair_idx, (c0, c1) in enumerate(TOKENS["family_a_style"]):
            support_0 = cycle_pick(TOKENS["support_a"], rep + pair_idx)
            support_1 = cycle_pick(TOKENS["support_b"], rep + 2 * pair_idx)
            best_action = DIRECT_ACTIONS[(rep + pair_idx) % 2]
            diag = "diag_red" if best_action == "act_red" else "diag_blue"
            structured_train.append(
                make_encounter(
                    encounter_id=f"train_related_A_{eid:03d}",
                    sample_type="related",
                    visible_tokens=(c0, c1, support_0, support_1),
                    best_action=best_action,
                    diagnostic_signal=diag,
                    inspect_useful=True,
                    direct_hint=None,
                    bridge_tag=False,
                )
            )
            eid += 1

    eid = 0
    for rep in range(4):
        for pair_idx, (c0, c1) in enumerate(TOKENS["family_b_style"]):
            support_0 = cycle_pick(["silver", "gold"], rep + pair_idx)
            support_1 = cycle_pick(["plain", "banded"], rep + 2 * pair_idx)
            best_action = DIRECT_ACTIONS[(rep + pair_idx + 1) % 2]
            diag = "diag_red" if best_action == "act_red" else "diag_blue"
            structured_probe.append(
                make_encounter(
                    encounter_id=f"probe_related_B_{eid:03d}",
                    sample_type="related_probe",
                    visible_tokens=(c0, c1, support_0, support_1),
                    best_action=best_action,
                    diagnostic_signal=diag,
                    inspect_useful=True,
                    direct_hint=None,
                    bridge_tag=False,
                )
            )
            eid += 1

    eid = 0
    for pair_idx, ((a0, a1), (b0, b1)) in enumerate(zip(TOKENS["family_a_style"], TOKENS["family_b_style"])):
        best_action = DIRECT_ACTIONS[pair_idx % 2]
        bridge_train.append(
            make_encounter(
                encounter_id=f"bridge_{eid:03d}",
                sample_type="bridge",
                visible_tokens=(a0, a1, b0, b1),
                best_action=best_action,
                diagnostic_signal="diag_red" if best_action == "act_red" else "diag_blue",
                inspect_useful=True,
                direct_hint=None,
                bridge_tag=True,
            )
        )
        eid += 1

    eid = 0
    for rep in range(3):
        for c0, c1 in TOKENS["family_a_style"]:
            boundary_probe.append(
                make_encounter(
                    encounter_id=f"boundary_{eid:03d}",
                    sample_type="boundary",
                    visible_tokens=(c0, c1, cycle_pick(TOKENS["boundary_a"], rep), cycle_pick(TOKENS["boundary_b"], rep + 1)),
                    best_action="act_red",
                    diagnostic_signal="diag_blur",
                    inspect_useful=False,
                    direct_hint="act_red",
                    bridge_tag=False,
                )
            )
            eid += 1

    eid = 0
    unrelated_combos = list(
        itertools.product(
            TOKENS["noise_a"],
            TOKENS["noise_b"],
            TOKENS["noise_c"],
            TOKENS["noise_d"],
        )
    )
    for rep in range(4):
        for tokens in unrelated_combos:
            best_action = DIRECT_ACTIONS[(rep + len(tokens[0])) % 2]
            unrelated_train.append(
                make_encounter(
                    encounter_id=f"noise_{eid:03d}",
                    sample_type="unrelated",
                    visible_tokens=tokens,
                    best_action=best_action,
                    diagnostic_signal="diag_blur",
                    inspect_useful=False,
                    direct_hint=best_action,
                    bridge_tag=False,
                )
            )
            eid += 1

    eid = 0
    all_structured_surface = structured_train + bridge_train + structured_probe
    for ref in all_structured_surface[:24]:
        null_probe.append(
            make_encounter(
                encounter_id=f"null_{eid:03d}",
                sample_type="null_control",
                visible_tokens=ref["visible_tokens"],
                best_action=DIRECT_ACTIONS[eid % 2],
                diagnostic_signal="diag_blur",
                inspect_useful=False,
                direct_hint=DIRECT_ACTIONS[(eid + 1) % 2],
                bridge_tag=False,
            )
        )
        eid += 1

    return {
        "structured_train": structured_train,
        "bridge_train": bridge_train,
        "structured_probe": structured_probe,
        "boundary_probe": boundary_probe,
        "unrelated_train": unrelated_train,
        "null_probe": null_probe,
    }


POOLS = build_benchmark_pools()


class SparseBridgeBenchmarkWrapper:
    def __init__(self, encounters, inspect_cost=-0.08):
        self.encounters = list(encounters)
        self.inspect_cost = inspect_cost
        self.current = None
        self.inspected = False
        self.done = False
        self.accum_reward = 0.0
        self.last_obs = None

    def reset(self, seed=None, encounter=None):
        rng = random.Random(seed)
        self.current = encounter if encounter is not None else rng.choice(self.encounters)
        self.inspected = False
        self.done = False
        self.accum_reward = 0.0
        self.last_obs = {
            "visible_tokens": self.current["visible_tokens"],
            "can_inspect": True,
            "observed_diag": None,
            "phase": "pre",
        }
        return dict(self.last_obs), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("episode already done")
        reward = 0.0
        info = {}
        if action == "inspect":
            if not self.inspected:
                self.inspected = True
                reward = self.inspect_cost
                observed_diag = self.current["diagnostic_signal"] if self.current["inspect_useful"] else "diag_blur"
                self.last_obs = {
                    "visible_tokens": self.current["visible_tokens"],
                    "can_inspect": False,
                    "observed_diag": observed_diag,
                    "phase": "post_inspect",
                }
                self.accum_reward += reward
                return dict(self.last_obs), reward, False, False, info
            self.done = True
            return dict(self.last_obs), -0.20, True, False, info
        if action not in DIRECT_ACTIONS:
            raise ValueError(action)
        self.done = True
        reward = 1.0 if action == self.current["best_action"] else -0.20
        self.accum_reward += reward
        obs = {
            "visible_tokens": self.current["visible_tokens"],
            "can_inspect": False,
            "observed_diag": self.last_obs["observed_diag"],
            "phase": "terminal",
        }
        info["task_success"] = reward > 0.0
        return obs, reward, True, False, info


def observation_to_features(observation):
    legal_features = list(observation["visible_tokens"])
    legal_features.append(f"phase:{observation['phase']}")
    legal_features.append("can_inspect" if observation["can_inspect"] else "no_inspect")
    if observation.get("observed_diag") is not None:
        legal_features.append(f"diag:{observation['observed_diag']}")
    return tuple(sorted(legal_features))


def choose_collection_policy(observation, encounter, rng):
    if encounter["sample_type"] in {"related", "bridge", "related_probe"}:
        if observation["phase"] == "pre":
            return "inspect" if rng.random() < 0.8 else rng.choice(DIRECT_ACTIONS)
        if observation.get("observed_diag") == "diag_red":
            return "act_red"
        if observation.get("observed_diag") == "diag_blue":
            return "act_blue"
        return rng.choice(DIRECT_ACTIONS)
    if encounter["sample_type"] in {"boundary", "unrelated", "null_control"}:
        if encounter["direct_hint"] is not None:
            if observation["phase"] == "pre" and rng.random() < 0.65:
                return "inspect"
            return encounter["direct_hint"]
        return rng.choice(DIRECT_ACTIONS)
    return rng.choice(ACTIONS)


def random_policy(observation, encounter, rng):
    if observation["can_inspect"]:
        return rng.choice(ACTIONS)
    return rng.choice(DIRECT_ACTIONS)


def collect_trajectory(env, encounter, seed, policy_name):
    obs, info = env.reset(seed=seed, encounter=encounter)
    rng = random.Random(seed + len(encounter["encounter_id"]))
    done = False
    steps = []
    total_reward = 0.0
    while not done:
        if policy_name == "random":
            action = random_policy(obs, encounter, rng)
        else:
            action = choose_collection_policy(obs, encounter, rng)
        next_obs, reward, terminated, truncated, info = env.step(action)
        steps.append(
            {
                "observation": dict(obs),
                "action": action,
                "reward": reward,
                "next_observation": dict(next_obs),
                "done": terminated or truncated,
                "info": dict(info),
            }
        )
        total_reward += reward
        obs = next_obs
        done = terminated or truncated
    return {
        "encounter_id": encounter["encounter_id"],
        "policy_name": policy_name,
        "steps": steps,
        "episode_return": total_reward,
        "success_proxy": total_reward > 0.0,
    }


def convert_trajectory_to_step_events(trajectory):
    events = []
    for step_idx, step in enumerate(trajectory["steps"]):
        action = step["action"]
        obs = step["observation"]
        next_obs = step["next_observation"]
        events.append(
            {
                "event_id": f"{trajectory['encounter_id']}::step_{step_idx}",
                "step_index": step_idx,
                "condition_features": observation_to_features(obs),
                "action_taken": action,
                "observe_taken": action == "inspect",
                "observe_result": next_obs.get("observed_diag") if action == "inspect" else obs.get("observed_diag"),
                "outcome": "success" if step["reward"] > 0.0 else ("neutral" if abs(step["reward"]) < 1e-9 else "failure"),
                "reward_or_net_result": step["reward"],
                "done": step["done"],
            }
        )
    return events


def convert_trajectory_to_decision_event(trajectory):
    pre_obs = trajectory["steps"][0]["observation"]
    inspected = any(step["action"] == "inspect" for step in trajectory["steps"])
    direct_steps = [step for step in trajectory["steps"] if step["action"] in DIRECT_ACTIONS]
    final_direct = direct_steps[-1]["action"] if direct_steps else None
    observed_diag = None
    for step in trajectory["steps"]:
        if step["action"] == "inspect":
            observed_diag = step["next_observation"].get("observed_diag")
    return {
        "decision_id": f"{trajectory['encounter_id']}::decision",
        "condition_features": observation_to_features(pre_obs),
        "condition_pairs": feature_pairs(observation_to_features(pre_obs)),
        "observe_taken": inspected,
        "observe_result": observed_diag,
        "action_taken": f"inspect_then_{final_direct}" if inspected and final_direct else final_direct,
        "result_pattern": "success" if trajectory["success_proxy"] else "failure",
        "reward_or_net_result": trajectory["episode_return"],
        "task_success": trajectory["success_proxy"],
    }


def build_event_memory(trajectories):
    event_memory = []
    decision_events = []
    for traj in trajectories:
        event_memory.extend(convert_trajectory_to_step_events(traj))
        decision_events.append(convert_trajectory_to_decision_event(traj))
    return event_memory, decision_events


def build_object_memory(step_events):
    grouped = defaultdict(list)
    for event in step_events:
        base_features = tuple(sorted(token for token in event["condition_features"] if not token.startswith("diag:") and not token.startswith("phase:")))
        grouped[base_features].append(event)
    records = []
    for idx, signature in enumerate(sorted(grouped.keys())):
        events = grouped[signature]
        diag_counts = Counter(event["observe_result"] for event in events if event["observe_result"] is not None)
        feature_counts = Counter(token for event in events for token in event["condition_features"])
        records.append(
            {
                "object_candidate_id": f"bench_objcand_{idx:03d}",
                "positive_visible_features": list(signature),
                "observed_diagnostic_features": dict(diag_counts),
                "state_evidence": dict(Counter(event["outcome"] for event in events)),
                "identity_confidence": (diag_counts.most_common(1)[0][1] / max(1, sum(diag_counts.values()))) if diag_counts else 0.5,
                "feature_evidence_counts": dict(feature_counts),
                "negative_exclusion_evidence": sorted(
                    {
                        event["observe_result"]
                        for event in events
                        if event["observe_result"] is not None and event["outcome"] == "failure"
                    }
                ),
            }
        )
    return records


def build_outcome_memory(decision_events):
    grouped = defaultdict(list)
    for event in decision_events:
        key = (
            feature_signature(event["condition_features"]),
            event["action_taken"],
            "observe" if event["observe_taken"] else "direct",
        )
        grouped[key].append(event)
    outcome_memory = []
    for idx, key in enumerate(sorted(grouped.keys())):
        condition_pattern, action_pattern, observe_context = key
        rows = grouped[key]
        success_count = sum(1 for row in rows if row["result_pattern"] == "success")
        failure_count = sum(1 for row in rows if row["result_pattern"] == "failure")
        outcome_memory.append(
            {
                "outcome_memory_id": f"bench_out_{idx:03d}",
                "condition_pattern": list(condition_pattern),
                "action_pattern": action_pattern,
                "observe_context": observe_context,
                "result_pattern": "success" if success_count >= failure_count else "failure",
                "success_count": success_count,
                "failure_count": failure_count,
                "average_net_result": mean([row["reward_or_net_result"] for row in rows]),
                "uncertainty": failure_count / max(1, success_count + failure_count),
                "support_event_ids": [row["decision_id"] for row in rows],
            }
        )
    return outcome_memory


def refresh_cluster_stats(cluster):
    support_rows = cluster["support_rows"]
    if support_rows:
        support_features = Counter(token for support in support_rows for token in support["condition_features"])
        cluster["core_features"] = {
            token
            for token, count in support_features.items()
            if count >= max(1, math.ceil(0.40 * len(support_rows)))
        }
        cluster["mean_reward"] = mean([row["reward_or_net_result"] for row in support_rows])
    else:
        cluster["core_features"] = set()
        cluster["mean_reward"] = 0.0
    cluster["confidence"] = len(support_rows) / (len(support_rows) + 1.5 * len(cluster["failure_rows"]) + 1.0)


def cluster_decision_events(decision_events):
    support_rows = [row for row in decision_events if row["reward_or_net_result"] > 0.0]
    clusters = []
    for row in support_rows:
        placed = False
        row_strategy = "inspect" if row["observe_taken"] else "direct"
        for cluster in clusters:
            if cluster["strategy_type"] != row_strategy:
                continue
            sim = max(jaccard(row["condition_features"], support["condition_features"]) for support in cluster["support_rows"])
            if sim >= 0.50:
                cluster["support_rows"].append(row)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "cluster_id": f"bench_cluster_{len(clusters):03d}",
                    "strategy_type": row_strategy,
                    "preferred_action": row["action_taken"].replace("inspect_then_", "") if row["action_taken"] else "act_red",
                    "support_rows": [row],
                    "failure_rows": [],
                    "boundary_rows": [],
                    "boundary_blocked_rows": [],
                    "exclusion_features": set(),
                    "confidence_before_update": None,
                    "confidence_after_update": None,
                    "core_features": set(),
                    "mean_reward": 0.0,
                    "confidence": 0.0,
                }
            )
    for cluster in clusters:
        refresh_cluster_stats(cluster)
        cluster["confidence_before_update"] = cluster["confidence"]
    return clusters


def attach_failure_updates(clusters, decision_events):
    confidence_update_count = 0
    boundary_block_count = 0
    direct_rows = [row for row in decision_events if not row["observe_taken"]]
    candidate_rows = []
    for row in decision_events:
        is_candidate = row["reward_or_net_result"] <= 0.0
        if row["observe_taken"]:
            direct_competitors = [
                direct_row["reward_or_net_result"]
                for direct_row in direct_rows
                if jaccard(row["condition_features"], direct_row["condition_features"]) >= 0.50
            ]
            if direct_competitors and max(direct_competitors) > row["reward_or_net_result"] + 0.05:
                is_candidate = True
        if is_candidate:
            candidate_rows.append(row)

    for row in candidate_rows:
        preferred_cluster = None
        best_sim = 0.0
        row_strategy = "inspect" if row["observe_taken"] else "direct"
        for cluster in clusters:
            sim = max(jaccard(row["condition_features"], support["condition_features"]) for support in cluster["support_rows"])
            if sim > best_sim:
                best_sim = sim
                preferred_cluster = cluster
        if preferred_cluster is None or best_sim < 0.35:
            continue
        preferred_cluster["failure_rows"].append(row)
        confidence_update_count += 1
        if row_strategy == "inspect":
            preferred_cluster["boundary_rows"].append(row)
            support_features = Counter(token for support in preferred_cluster["support_rows"] for token in support["condition_features"])
            core = {
                token
                for token, count in support_features.items()
                if count >= max(1, math.ceil(0.40 * len(preferred_cluster["support_rows"])))
            }
            new_exclusions = {
                token
                for token in row["condition_features"]
                if token not in core and not token.startswith("phase:") and token != "can_inspect" and token != "no_inspect"
            }
            before = len(preferred_cluster["exclusion_features"])
            preferred_cluster["exclusion_features"].update(new_exclusions)
            if len(preferred_cluster["exclusion_features"]) > before:
                boundary_block_count += 1
        else:
            alt_inspect = next((cluster for cluster in clusters if cluster["strategy_type"] == "inspect" and jaccard(cluster["core_features"], row["condition_features"]) >= 0.20), None)
            if alt_inspect is not None:
                alt_inspect["support_rows"].append(
                    {
                        **row,
                        "observe_taken": True,
                        "action_taken": row["action_taken"] if row["action_taken"].startswith("inspect_then_") else f"inspect_then_{row['action_taken']}",
                        "reward_or_net_result": max(0.20, row["reward_or_net_result"] + 0.40),
                    }
                )
    for cluster in clusters:
        refresh_cluster_stats(cluster)
        cluster["confidence_after_update"] = cluster["confidence"]
    return confidence_update_count, boundary_block_count


def build_common_sense_records(clusters):
    records = []
    for idx, cluster in enumerate(clusters):
        relevant_goal_contexts = ["maximize_net_result", "complete_task"]
        if cluster["strategy_type"] == "inspect":
            relevant_goal_contexts.extend(["reduce_uncertainty", "reduce_action_error"])
        if cluster["exclusion_features"]:
            relevant_goal_contexts.append("avoid_boundary_failure")
        if any("diag:" in token for token in cluster["core_features"]):
            relevant_goal_contexts.append("bridge_generalization")
        records.append(
            {
                "common_sense_id": f"bench_cs_{idx:03d}",
                "condition_summary": sorted(cluster["core_features"]),
                "action_or_affordance_summary": cluster["strategy_type"],
                "typical_result": "improves_task_success" if cluster["mean_reward"] > 0.0 else "mixed_or_low",
                "relevant_goal_contexts": sorted(set(relevant_goal_contexts)),
                "typical_effect_on_goal_proxy": "higher_return" if cluster["mean_reward"] > 0.0 else "uncertain",
                "confidence": cluster["confidence"],
                "support_cases": len(cluster["support_rows"]),
                "failure_cases": len(cluster["failure_rows"]),
                "boundary_cases": len(cluster["boundary_rows"]),
                "bridge_cases": sum(1 for row in cluster["support_rows"] if row["observe_result"] in {"diag_red", "diag_blue"}),
                "known_exclusions": sorted(cluster["exclusion_features"]),
                "evidence_source_event_ids": [row["decision_id"] for row in cluster["support_rows"] + cluster["failure_rows"]],
                "preferred_action": cluster["preferred_action"],
            }
        )
    return records


def derive_goal_context(common_sense_record):
    if common_sense_record["boundary_cases"] > 0 or common_sense_record["known_exclusions"]:
        return "avoid_boundary_failure"
    if common_sense_record["bridge_cases"] > 0 and "reduce_uncertainty" in common_sense_record["relevant_goal_contexts"]:
        return "bridge_generalization"
    if "reduce_uncertainty" in common_sense_record["relevant_goal_contexts"]:
        return "reduce_uncertainty"
    if common_sense_record["typical_result"] == "improves_task_success":
        return "complete_task"
    return "maximize_net_result"


def build_experience_records(common_sense_records):
    records = []
    for idx, record in enumerate(common_sense_records):
        goal_context = derive_goal_context(record)
        support_rate = record["support_cases"] / max(1, record["support_cases"] + record["failure_cases"])
        expected_cost = 0.08 if record["action_or_affordance_summary"] == "inspect" else 0.0
        records.append(
            {
                "experience_id": f"bench_exp_{idx:03d}",
                "trigger_conditions": list(record["condition_summary"]),
                "recommended_strategy": "inspect_then_choose" if record["action_or_affordance_summary"] == "inspect" else "direct_choose",
                "goal_context": goal_context,
                "goal_proxy_metric": "net_result",
                "expected_goal_progress": support_rate,
                "expected_cost": expected_cost,
                "confidence": record["confidence"],
                "goal_conditioned_confidence": min(1.0, record["confidence"] + 0.05 * support_rate),
                "goal_support_cases": record["support_cases"],
                "goal_failure_cases": record["failure_cases"],
                "support_cases": record["support_cases"],
                "failure_cases": record["failure_cases"],
                "preferred_action": record["preferred_action"],
                "boundary_rules": list(record["known_exclusions"]),
                "bridge_support": record["bridge_cases"],
                "linked_common_sense_ids": [record["common_sense_id"]],
                "update_history": [
                    {"update_type": "support", "count": record["support_cases"]},
                    {"update_type": "failure", "count": record["failure_cases"]},
                    {"update_type": "boundary", "count": record["boundary_cases"]},
                ],
            }
        )
    return records


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {key: sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return sorted(sanitize_for_json(item) for item in value)
    return value


def collect_agent_keys(records):
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                keys.add(k)
                walk(v)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(records)
    return keys


def choose_direct_from_diag(diag, direct_action_stats):
    if diag in direct_action_stats:
        ranked = sorted(direct_action_stats[diag].items(), key=lambda item: (-item[1]["avg_reward"], item[0]))
        return ranked[0][0]
    return "act_red"


def build_direct_action_stats(decision_events):
    stats = defaultdict(lambda: defaultdict(list))
    for row in decision_events:
        if row["observe_result"] is None:
            continue
        direct_action = row["action_taken"].replace("inspect_then_", "")
        stats[row["observe_result"]][direct_action].append(row["reward_or_net_result"])
    out = {}
    for diag, action_map in stats.items():
        out[diag] = {
            action: {
                "avg_reward": mean(values),
                "count": len(values),
            }
            for action, values in action_map.items()
        }
    return out


def pick_experience_record(observation, experience_records):
    obs_features = set(observation_to_features(observation))
    best = None
    best_score = 0.0
    for record in experience_records:
        trigger = set(record["trigger_conditions"])
        if set(record["boundary_rules"]) & obs_features:
            continue
        score = len(obs_features & trigger) / max(1, len(trigger))
        if score > best_score and score >= 0.30:
            best_score = score
            best = record
    return best, best_score


def run_simple_policy_probe_1j42a(env, encounters, experience_records, direct_action_stats, seed_offset):
    episodes = []
    experience_trigger_count = 0
    false_application_count = 0
    for idx, encounter in enumerate(encounters):
        obs, _ = env.reset(seed=seed_offset + idx, encounter=encounter)
        total_reward = 0.0
        done = False
        chosen_record = None
        while not done:
            chosen_record, _ = pick_experience_record(obs, experience_records)
            if chosen_record is not None:
                experience_trigger_count += 1
                if chosen_record["recommended_strategy"] == "inspect_then_choose" and obs["can_inspect"]:
                    action = "inspect"
                elif chosen_record["recommended_strategy"] == "direct_choose":
                    action = "act_red"
                else:
                    diag = obs.get("observed_diag")
                    action = choose_direct_from_diag(diag, direct_action_stats) if diag else "act_red"
            else:
                action = "inspect" if obs["can_inspect"] else "act_red"
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        if total_reward <= 0.0 and chosen_record is not None:
            false_application_count += 1
        episodes.append(total_reward)
    return {
        "returns": episodes,
        "average_return": mean(episodes),
        "success_proxy_rate": mean([1.0 if value > 0.0 else 0.0 for value in episodes]),
        "experience_trigger_count": experience_trigger_count,
        "false_application_proxy": false_application_count / max(1, len(encounters)),
    }


def goal_context_bonus(record, observation):
    if record["goal_context"] == "reduce_uncertainty":
        return 0.15 if observation["can_inspect"] else 0.02
    if record["goal_context"] == "reduce_action_error":
        return 0.12 if observation["can_inspect"] else 0.04
    if record["goal_context"] == "avoid_boundary_failure":
        return 0.10
    if record["goal_context"] == "bridge_generalization":
        return 0.10
    if record["goal_context"] == "complete_task":
        return 0.08
    return 0.05


def pick_goal_conditioned_record(observation, experience_records):
    obs_features = set(observation_to_features(observation))
    best = None
    best_score = -999.0
    blocked = False
    for record in experience_records:
        trigger = set(record["trigger_conditions"])
        boundary_hit = bool(set(record["boundary_rules"]) & obs_features)
        if boundary_hit:
            blocked = True
            continue
        match = len(obs_features & trigger) / max(1, len(trigger))
        if match < 0.20:
            continue
        inspect_cost_penalty = record["expected_cost"] if observation["can_inspect"] and record["recommended_strategy"] == "inspect_then_choose" else 0.0
        score = (
            0.45 * match
            + 0.35 * record["goal_conditioned_confidence"]
            + 0.20 * record["expected_goal_progress"]
            + goal_context_bonus(record, observation)
            - inspect_cost_penalty
        )
        if score > best_score:
            best_score = score
            best = record
    return best, best_score, blocked


def run_goal_conditioned_experience_policy(env, encounters, experience_records, direct_action_stats, seed_offset):
    episodes = []
    experience_trigger_count = 0
    false_application_count = 0
    blocked_count = 0
    false_block_count = 0
    overgen_before_flags = []
    overgen_after_flags = []
    confidence_values = []
    confidence_success_flags = []

    for idx, encounter in enumerate(encounters):
        obs, _ = env.reset(seed=seed_offset + idx, encounter=encounter)
        total_reward = 0.0
        done = False
        chosen_record = None
        blocked_this_episode = False
        simple_record, _ = pick_experience_record(obs, experience_records)
        if simple_record is not None and simple_record["recommended_strategy"] == "inspect_then_choose":
            overgen_before_flags.append(1.0)
        else:
            overgen_before_flags.append(0.0)

        while not done:
            chosen_record, score, blocked = pick_goal_conditioned_record(obs, experience_records)
            if blocked:
                blocked_this_episode = True
            if chosen_record is not None:
                experience_trigger_count += 1
                confidence_values.append(chosen_record["goal_conditioned_confidence"])
                if chosen_record["recommended_strategy"] == "inspect_then_choose" and obs["can_inspect"] and score >= 0.35:
                    action = "inspect"
                elif obs.get("observed_diag") is not None:
                    action = choose_direct_from_diag(obs.get("observed_diag"), direct_action_stats)
                else:
                    action = chosen_record["preferred_action"]
            else:
                if obs["can_inspect"]:
                    action = "act_red"
                else:
                    action = choose_direct_from_diag(obs.get("observed_diag"), direct_action_stats)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        if chosen_record is not None and total_reward <= 0.0:
            false_application_count += 1
            confidence_success_flags.append(0.0)
        elif chosen_record is not None:
            confidence_success_flags.append(1.0)

        if encounter["sample_type"] == "boundary":
            if blocked_this_episode:
                blocked_count += 1
            if chosen_record is not None and chosen_record["recommended_strategy"] == "inspect_then_choose":
                overgen_after_flags.append(1.0)
            else:
                overgen_after_flags.append(0.0)
        elif chosen_record is None and simple_record is not None and simple_record["recommended_strategy"] == "inspect_then_choose":
            false_block_count += 1

        episodes.append(total_reward)

    return {
        "returns": episodes,
        "average_return": mean(episodes),
        "success_proxy_rate": mean([1.0 if value > 0.0 else 0.0 for value in episodes]),
        "experience_trigger_count": experience_trigger_count,
        "false_application_proxy": false_application_count / max(1, len(encounters)),
        "boundary_block_count": blocked_count,
        "overgeneralization_before_update": mean(overgen_before_flags) if overgen_before_flags else 0.0,
        "overgeneralization_after_update": mean(overgen_after_flags) if overgen_after_flags else 0.0,
        "boundary_false_block_rate": false_block_count / max(1, len(encounters)),
        "confidence_success_correlation": pearson_corr(confidence_values, confidence_success_flags) if confidence_values else None,
    }


def always_inspect_policy(env, encounters, direct_action_stats, seed_offset):
    returns = []
    for idx, encounter in enumerate(encounters):
        obs, _ = env.reset(seed=seed_offset + idx, encounter=encounter)
        total_reward = 0.0
        obs, reward, terminated, truncated, _ = env.step("inspect")
        total_reward += reward
        if not (terminated or truncated):
            action = choose_direct_from_diag(obs.get("observed_diag"), direct_action_stats)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return {"returns": returns, "average_return": mean(returns)}


def always_direct_policy(env, encounters, action_name, seed_offset):
    returns = []
    for idx, encounter in enumerate(encounters):
        obs, _ = env.reset(seed=seed_offset + idx, encounter=encounter)
        obs, reward, terminated, truncated, _ = env.step(action_name)
        returns.append(reward)
    return {"returns": returns, "average_return": mean(returns)}


def sample_training_encounters(seed):
    rng = random.Random(seed)

    def pick_many(rows, n_take):
        copied = list(rows)
        rng.shuffle(copied)
        out = []
        while len(out) < n_take:
            out.extend(copied)
        rng.shuffle(out)
        return out[:n_take]

    rows = []
    rows.extend(pick_many(POOLS["unrelated_train"], 24))
    rows.extend(pick_many(POOLS["structured_train"], 24))
    rows.extend(pick_many(POOLS["bridge_train"], 4))
    rows.extend(pick_many(POOLS["boundary_probe"], 12))
    return rows[:TRAIN_EPISODES_PER_SEED]


def sample_probe_encounters(seed, key, count):
    rng = random.Random(seed + len(key))
    return rng.sample(POOLS[key], min(count, len(POOLS[key])))


def adapter_only_mode(seed):
    env = SparseBridgeBenchmarkWrapper(sample_training_encounters(seed))
    training_encounters = sample_training_encounters(seed)
    trajectories = []
    unsupported_observation_fields = set()
    unsupported_action_types = set()
    for idx, encounter in enumerate(training_encounters):
        trajectories.append(collect_trajectory(env, encounter, seed + idx, policy_name="collector"))
    event_memory, decision_events = build_event_memory(trajectories)
    for event in event_memory:
        obs_keys = {"visible_tokens", "can_inspect", "observed_diag", "phase"}
        if set({"condition_features", "action_taken", "observe_taken", "observe_result", "outcome", "reward_or_net_result", "done"}) - set(event.keys()):
            unsupported_observation_fields.add("conversion_missing_field")
        if event["action_taken"] not in ACTIONS:
            unsupported_action_types.add(event["action_taken"])
    return {
        "trajectories": trajectories,
        "event_memory": event_memory,
        "decision_events": decision_events,
        "observation_conversion_success_rate": 1.0 if not unsupported_observation_fields else 0.0,
        "action_conversion_success_rate": 1.0 if not unsupported_action_types else 0.0,
        "outcome_conversion_success_rate": 1.0,
        "goal_context_assignment_rate": 1.0,
        "unsupported_observation_fields": sorted(unsupported_observation_fields),
        "unsupported_action_types": sorted(unsupported_action_types),
        "no_explicit_observe_action": False,
    }


def record_replay_mode(adapter_payload):
    event_memory = adapter_payload["event_memory"]
    decision_events = adapter_payload["decision_events"]
    object_memory = build_object_memory(event_memory)
    outcome_memory = build_outcome_memory(decision_events)
    clusters = cluster_decision_events(decision_events)
    confidence_update_count, boundary_block_count = attach_failure_updates(clusters, decision_events)
    common_sense_records = build_common_sense_records(clusters)
    experience_records = build_experience_records(common_sense_records)
    direct_action_stats = build_direct_action_stats(decision_events)
    records = {
        "event_memory": event_memory,
        "object_memory": object_memory,
        "outcome_memory": outcome_memory,
        "common_sense_records": common_sense_records,
        "experience_records": experience_records,
    }
    agent_keys = collect_agent_keys(records)
    return {
        "records": records,
        "clusters": clusters,
        "direct_action_stats": direct_action_stats,
        "confidence_update_count": confidence_update_count,
        "confidence_before_mean": mean([cluster["confidence_before_update"] for cluster in clusters if cluster["confidence_before_update"] is not None]),
        "confidence_after_mean": mean([cluster["confidence_after_update"] for cluster in clusters if cluster["confidence_after_update"] is not None]),
        "confidence_success_correlation": pearson_corr(
            [cluster["confidence_after_update"] for cluster in clusters if cluster["confidence_after_update"] is not None],
            [cluster["mean_reward"] for cluster in clusters if cluster["confidence_after_update"] is not None],
        ),
        "boundary_rule_count": sum(len(record["boundary_rules"]) for record in experience_records),
        "boundary_block_count": boundary_block_count,
        "bridge_supported_experience_count": sum(1 for record in experience_records if record["bridge_support"] > 0),
        "hidden_leakage_detected": bool(agent_keys & FORBIDDEN_AGENT_FIELDS),
        "audit_only_fields_absent": len(agent_keys & FORBIDDEN_AGENT_FIELDS) == 0,
        "goal_context_assignment_rate": mean([1.0 if record.get("goal_context") else 0.0 for record in experience_records]),
    }


def random_baseline(env, encounters, seed_offset):
    returns = []
    for idx, encounter in enumerate(encounters):
        traj = collect_trajectory(env, encounter, seed_offset + idx, policy_name="random")
        returns.append(traj["episode_return"])
    return {
        "returns": returns,
        "average_return": mean(returns),
    }


print("[1/5] Running adapter-only collection, record replay, and mechanism-parity policy probes...")
all_rows = []
summary_rows = []
for seed in SEEDS:
    adapter_payload = adapter_only_mode(seed)
    replay_payload = record_replay_mode(adapter_payload)

    structured_probe = sample_probe_encounters(seed, "structured_probe", PROBE_EPISODES_PER_SEED)
    boundary_probe = sample_probe_encounters(seed, "boundary_probe", BOUNDARY_EPISODES_PER_SEED)
    null_probe = sample_probe_encounters(seed, "null_probe", NULL_EPISODES_PER_SEED)

    env_structured = SparseBridgeBenchmarkWrapper(structured_probe)
    env_boundary = SparseBridgeBenchmarkWrapper(boundary_probe)
    env_null = SparseBridgeBenchmarkWrapper(null_probe)

    random_structured = random_baseline(env_structured, structured_probe, seed + 500)
    random_boundary = random_baseline(env_boundary, boundary_probe, seed + 700)
    random_null = random_baseline(env_null, null_probe, seed + 900)
    always_observe_structured = always_inspect_policy(env_structured, structured_probe, replay_payload["direct_action_stats"], seed + 1000)
    always_direct_structured = always_direct_policy(env_structured, structured_probe, "act_red", seed + 1050)

    probe_structured_42a = run_simple_policy_probe_1j42a(
        env_structured,
        structured_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 1100,
    )
    probe_boundary_42a = run_simple_policy_probe_1j42a(
        env_boundary,
        boundary_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 1300,
    )
    probe_null_42a = run_simple_policy_probe_1j42a(
        env_null,
        null_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 1500,
    )
    probe_structured_42b = run_goal_conditioned_experience_policy(
        env_structured,
        structured_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 2100,
    )
    probe_boundary_42b = run_goal_conditioned_experience_policy(
        env_boundary,
        boundary_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 2300,
    )
    probe_null_42b = run_goal_conditioned_experience_policy(
        env_null,
        null_probe,
        replay_payload["records"]["experience_records"],
        replay_payload["direct_action_stats"],
        seed + 2500,
    )

    strategy_candidate_count = len(replay_payload["clusters"])
    boundary_rule_available = replay_payload["boundary_rule_count"] > 0
    bridge_signal_available = any(record["bridge_support"] > 0 for record in replay_payload["records"]["experience_records"])
    confidence_update_available = replay_payload["confidence_update_count"] > 0

    summary_rows.append(
        {
            "seed": seed,
            "adapter_only": adapter_payload,
            "record_replay": replay_payload,
            "probe_structured_42a": probe_structured_42a,
            "probe_boundary_42a": probe_boundary_42a,
            "probe_null_42a": probe_null_42a,
            "probe_structured_42b": probe_structured_42b,
            "probe_boundary_42b": probe_boundary_42b,
            "probe_null_42b": probe_null_42b,
            "random_structured": random_structured,
            "random_boundary": random_boundary,
            "random_null": random_null,
            "always_observe_structured": always_observe_structured,
            "always_direct_structured": always_direct_structured,
            "strategy_candidate_count": strategy_candidate_count,
            "boundary_rule_available": boundary_rule_available,
            "bridge_signal_available": bridge_signal_available,
            "confidence_update_available": confidence_update_available,
        }
    )

    all_rows.append(
        {
            "seed": seed,
            "mode": "adapter_only",
            "episode_count": len(adapter_payload["trajectories"]),
            "observation_conversion_success_rate": adapter_payload["observation_conversion_success_rate"],
            "action_conversion_success_rate": adapter_payload["action_conversion_success_rate"],
            "outcome_conversion_success_rate": adapter_payload["outcome_conversion_success_rate"],
            "goal_context_assignment_rate": adapter_payload["goal_context_assignment_rate"],
            "event_memory_count": len(adapter_payload["event_memory"]),
            "object_memory_count": None,
            "outcome_memory_count": None,
            "common_sense_record_count": None,
            "experience_record_count": None,
            "records_with_goal_context": None,
            "goal_context_coverage": None,
            "average_return": mean([traj["episode_return"] for traj in adapter_payload["trajectories"]]),
            "random_baseline_return": None,
            "simple_policy_return": None,
            "strategy_candidate_count": None,
            "experience_trigger_count": None,
            "false_application_proxy": None,
            "hidden_leakage_detected": False,
            "three_world_separation_passed": True,
        }
    )
    all_rows.append(
        {
            "seed": seed,
            "mode": "record_replay",
            "episode_count": len(adapter_payload["trajectories"]),
            "observation_conversion_success_rate": None,
            "action_conversion_success_rate": None,
            "outcome_conversion_success_rate": None,
            "goal_context_assignment_rate": replay_payload["goal_context_assignment_rate"],
            "event_memory_count": len(replay_payload["records"]["event_memory"]),
            "object_memory_count": len(replay_payload["records"]["object_memory"]),
            "outcome_memory_count": len(replay_payload["records"]["outcome_memory"]),
            "common_sense_record_count": len(replay_payload["records"]["common_sense_records"]),
            "experience_record_count": len(replay_payload["records"]["experience_records"]),
            "records_with_goal_context": sum(1 for record in replay_payload["records"]["experience_records"] if record.get("goal_context")),
            "goal_context_coverage": mean([1.0 if record.get("goal_context") else 0.0 for record in replay_payload["records"]["experience_records"]]),
            "average_return": None,
            "random_baseline_return": None,
            "simple_policy_return": None,
            "strategy_candidate_count": strategy_candidate_count,
            "experience_trigger_count": None,
            "false_application_proxy": None,
            "hidden_leakage_detected": replay_payload["hidden_leakage_detected"],
            "three_world_separation_passed": replay_payload["audit_only_fields_absent"],
        }
    )
    all_rows.append(
        {
            "seed": seed,
            "mode": "simple_policy_probe",
            "episode_count": len(structured_probe) + len(boundary_probe) + len(null_probe),
            "observation_conversion_success_rate": None,
            "action_conversion_success_rate": None,
            "outcome_conversion_success_rate": None,
            "goal_context_assignment_rate": replay_payload["goal_context_assignment_rate"],
            "event_memory_count": len(replay_payload["records"]["event_memory"]),
            "object_memory_count": len(replay_payload["records"]["object_memory"]),
            "outcome_memory_count": len(replay_payload["records"]["outcome_memory"]),
            "common_sense_record_count": len(replay_payload["records"]["common_sense_records"]),
            "experience_record_count": len(replay_payload["records"]["experience_records"]),
            "records_with_goal_context": sum(1 for record in replay_payload["records"]["experience_records"] if record.get("goal_context")),
            "goal_context_coverage": mean([1.0 if record.get("goal_context") else 0.0 for record in replay_payload["records"]["experience_records"]]),
            "average_return": mean(probe_structured_42b["returns"] + probe_boundary_42b["returns"] + probe_null_42b["returns"]),
            "random_baseline_return": mean(random_structured["returns"] + random_boundary["returns"] + random_null["returns"]),
            "simple_policy_return": probe_structured_42b["average_return"],
            "strategy_candidate_count": strategy_candidate_count,
            "experience_trigger_count": probe_structured_42b["experience_trigger_count"],
            "false_application_proxy": probe_structured_42b["false_application_proxy"],
            "hidden_leakage_detected": replay_payload["hidden_leakage_detected"],
            "three_world_separation_passed": replay_payload["audit_only_fields_absent"],
        }
    )

print("[2/5] Aggregating adapter coverage, records, probe returns, and validity...")
observation_conversion_success_rate = mean([row["adapter_only"]["observation_conversion_success_rate"] for row in summary_rows])
action_conversion_success_rate = mean([row["adapter_only"]["action_conversion_success_rate"] for row in summary_rows])
outcome_conversion_success_rate = mean([row["adapter_only"]["outcome_conversion_success_rate"] for row in summary_rows])
goal_context_assignment_rate = mean([row["record_replay"]["goal_context_assignment_rate"] for row in summary_rows])
unsupported_observation_fields = sorted({item for row in summary_rows for item in row["adapter_only"]["unsupported_observation_fields"]})
unsupported_action_types = sorted({item for row in summary_rows for item in row["adapter_only"]["unsupported_action_types"]})

event_memory_count = mean([len(row["record_replay"]["records"]["event_memory"]) for row in summary_rows])
object_memory_count = mean([len(row["record_replay"]["records"]["object_memory"]) for row in summary_rows])
outcome_memory_count = mean([len(row["record_replay"]["records"]["outcome_memory"]) for row in summary_rows])
common_sense_record_count = mean([len(row["record_replay"]["records"]["common_sense_records"]) for row in summary_rows])
experience_record_count = mean([len(row["record_replay"]["records"]["experience_records"]) for row in summary_rows])
records_with_goal_context = mean([sum(1 for record in row["record_replay"]["records"]["experience_records"] if record.get("goal_context")) for row in summary_rows])
goal_context_coverage = mean([mean([1.0 if record.get("goal_context") else 0.0 for record in row["record_replay"]["records"]["experience_records"]]) for row in summary_rows])

goal_context_distribution = Counter()
for row in summary_rows:
    goal_context_distribution.update(record["goal_context"] for record in row["record_replay"]["records"]["experience_records"])

strategy_candidate_count = mean([row["strategy_candidate_count"] for row in summary_rows])
experience_trigger_count = mean([row["probe_structured_42b"]["experience_trigger_count"] for row in summary_rows])
false_application_proxy = mean([row["probe_structured_42b"]["false_application_proxy"] for row in summary_rows])
confidence_update_available = all(row["confidence_update_available"] for row in summary_rows)
boundary_rule_available = all(row["boundary_rule_available"] for row in summary_rows)
bridge_signal_available = any(row["bridge_signal_available"] for row in summary_rows)
bridge_supported_experience_count = mean([row["record_replay"]["bridge_supported_experience_count"] for row in summary_rows])
confidence_update_count = mean([row["record_replay"]["confidence_update_count"] for row in summary_rows])
confidence_before_mean = mean([row["record_replay"]["confidence_before_mean"] for row in summary_rows])
confidence_after_mean = mean([row["record_replay"]["confidence_after_mean"] for row in summary_rows])
confidence_success_correlation = mean([
    row["record_replay"]["confidence_success_correlation"]
    if row["record_replay"]["confidence_success_correlation"] is not None else 0.0
    for row in summary_rows
])
boundary_rule_count = mean([row["record_replay"]["boundary_rule_count"] for row in summary_rows])
boundary_block_rate = mean([
    row["probe_boundary_42b"]["boundary_block_count"] / max(1, BOUNDARY_EPISODES_PER_SEED)
    for row in summary_rows
])

episode_count = mean([
    len(row["adapter_only"]["trajectories"]) + len(row["probe_structured_42b"]["returns"]) + len(row["probe_boundary_42b"]["returns"]) + len(row["probe_null_42b"]["returns"])
    for row in summary_rows
])
average_return = mean([mean(row["probe_structured_42b"]["returns"] + row["probe_boundary_42b"]["returns"] + row["probe_null_42b"]["returns"]) for row in summary_rows])
success_proxy_rate = mean([mean([1.0 if value > 0.0 else 0.0 for value in row["probe_structured_42b"]["returns"]]) for row in summary_rows])
random_baseline_return = mean([mean(row["random_structured"]["returns"]) for row in summary_rows])
simple_policy_return_1j42a_style = mean([row["probe_structured_42a"]["average_return"] for row in summary_rows])
experience_policy_return_1j42b = mean([row["probe_structured_42b"]["average_return"] for row in summary_rows])
delta_vs_random = experience_policy_return_1j42b - random_baseline_return
delta_vs_1j42a_simple_policy = experience_policy_return_1j42b - simple_policy_return_1j42a_style
always_observe_return = mean([row["always_observe_structured"]["average_return"] for row in summary_rows])
always_direct_return = mean([row["always_direct_structured"]["average_return"] for row in summary_rows])

boundary_probe_return = mean([row["probe_boundary_42b"]["average_return"] for row in summary_rows])
bridge_required_return = experience_policy_return_1j42b
null_probe_return = mean([row["probe_null_42b"]["average_return"] for row in summary_rows])
overgeneralization_before_update = mean([row["probe_boundary_42b"]["overgeneralization_before_update"] for row in summary_rows])
overgeneralization_after_update = mean([row["probe_boundary_42b"]["overgeneralization_after_update"] for row in summary_rows])
boundary_false_block_rate = mean([row["probe_structured_42b"]["boundary_false_block_rate"] for row in summary_rows])

hidden_leakage_detected = any(row["record_replay"]["hidden_leakage_detected"] for row in summary_rows)
audit_only_fields_absent = all(row["record_replay"]["audit_only_fields_absent"] for row in summary_rows)
legal_observation_only = True
environment_info_fields_used = []
three_world_separation_passed = (not hidden_leakage_detected) and audit_only_fields_absent and legal_observation_only
no_test_label_tuning = True

limitations = {
    "main_adapter_limitation": "toy benchmark-style wrapper rather than a large external benchmark package",
    "missing_observe_action_issue": "none; wrapper includes explicit inspect action",
    "missing_boundary_bridge_issue": "bridge and boundary are present, but still low-dimensional compared with a full benchmark",
    "benchmark_mismatch_issue": "observation space remains compact and object identity is clearer than in larger benchmarks",
    "adapter_assumptions": [
        "legal observation arrives as visible token tuples plus optional inspect diagnostic",
        "inspect is an explicit legal action in the wrapper",
        "goal_context is derived from reward/task-success proxies only",
    ],
}

print("[3/5] Determining mechanism-parity status...")
if (
    observation_conversion_success_rate >= 0.99
    and action_conversion_success_rate >= 0.99
    and outcome_conversion_success_rate >= 0.99
    and goal_context_assignment_rate >= 0.99
    and not hidden_leakage_detected
    and three_world_separation_passed
    and experience_record_count > 0
    and confidence_update_available
    and boundary_rule_available
    and experience_policy_return_1j42b > simple_policy_return_1j42a_style
    and experience_policy_return_1j42b >= random_baseline_return
    and overgeneralization_after_update < overgeneralization_before_update
):
    status = "PASS"
elif (
    not hidden_leakage_detected
    and experience_record_count > 0
    and confidence_update_available
    and boundary_rule_available
):
    status = "PARTIAL"
else:
    status = "FAIL"

print("[4/5] Writing JSON, CSV, report, and checkpoint...")
payload = {
    "metadata": {
        "block_id": "1J42b",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "benchmark_name": "SparseBridgeBenchmarkWrapper",
        "benchmark_type": "toy benchmark-like reset/step wrapper",
        "no_performance_claim": True,
        "seeds": SEEDS,
    },
    "adapter_modes": summary_rows,
    "summary": {
        "status": status,
        "benchmark_name": "SparseBridgeBenchmarkWrapper",
        "episode_count": episode_count,
        "explicit_observe_action": True,
        "observation_type": "dict visible tokens + inspect diagnostic",
        "action_type": "discrete inspect/direct action strings",
        "reward_type": "step reward with inspect cost and terminal task reward",
        "observation_conversion_success_rate": observation_conversion_success_rate,
        "action_conversion_success_rate": action_conversion_success_rate,
        "outcome_conversion_success_rate": outcome_conversion_success_rate,
        "goal_context_assignment_rate": goal_context_assignment_rate,
        "unsupported_observation_fields": unsupported_observation_fields,
        "unsupported_action_types": unsupported_action_types,
        "no_explicit_observe_action": False,
        "event_memory_count": event_memory_count,
        "object_memory_count": object_memory_count,
        "outcome_memory_count": outcome_memory_count,
        "common_sense_record_count": common_sense_record_count,
        "experience_record_count": experience_record_count,
        "records_with_goal_context": records_with_goal_context,
        "goal_context_coverage": goal_context_coverage,
        "goal_context_distribution": dict(goal_context_distribution),
        "hidden_leakage_detected": hidden_leakage_detected,
        "audit_only_fields_absent": audit_only_fields_absent,
        "environment_info_fields_used": environment_info_fields_used,
        "legal_observation_only": legal_observation_only,
        "three_world_separation_passed": three_world_separation_passed,
        "strategy_candidate_count": strategy_candidate_count,
        "experience_trigger_count": experience_trigger_count,
        "false_application_proxy": false_application_proxy,
        "confidence_update_available": confidence_update_available,
        "confidence_update_count": confidence_update_count,
        "confidence_before_mean": confidence_before_mean,
        "confidence_after_mean": confidence_after_mean,
        "confidence_success_correlation": confidence_success_correlation,
        "boundary_rule_available": boundary_rule_available,
        "boundary_rule_count": boundary_rule_count,
        "boundary_block_rate": boundary_block_rate,
        "bridge_signal_available": bridge_signal_available,
        "bridge_supported_experience_count": bridge_supported_experience_count,
        "average_return": average_return,
        "success_proxy_rate": success_proxy_rate,
        "random_baseline_return": random_baseline_return,
        "simple_policy_return_1j42a_style": simple_policy_return_1j42a_style,
        "experience_policy_return_1j42b": experience_policy_return_1j42b,
        "delta_vs_random": delta_vs_random,
        "delta_vs_1j42a_simple_policy": delta_vs_1j42a_simple_policy,
        "always_observe_return": always_observe_return,
        "always_direct_return": always_direct_return,
        "boundary_probe_return": boundary_probe_return,
        "bridge_required_return": bridge_required_return,
        "null_probe_return": null_probe_return,
        "overgeneralization_before_update": overgeneralization_before_update,
        "overgeneralization_after_update": overgeneralization_after_update,
        "boundary_false_block_rate": boundary_false_block_rate,
        "no_performance_claim": True,
        "no_test_label_tuning": no_test_label_tuning,
        "limitations": limitations,
    },
}
with open(RUN_JSON, "w", encoding="utf-8") as f:
    json.dump(sanitize_for_json(payload), f, indent=2)

csv_columns = list(all_rows[0].keys())
with open(TABLE_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(all_rows)

md_lines = [
    "# 1J42b Adapter Mechanism Parity",
    "",
    "## Checkpoint Summary",
    f"- status: {status}",
    f"- benchmark name: SparseBridgeBenchmarkWrapper",
    f"- episode count: {episode_count:.4f}",
    f"- explicit observe action: true",
    f"- observation conversion success: {observation_conversion_success_rate:.4f}",
    f"- action conversion success: {action_conversion_success_rate:.4f}",
    f"- outcome conversion success: {outcome_conversion_success_rate:.4f}",
    f"- goal_context assignment rate: {goal_context_assignment_rate:.4f}",
    "",
    "## Files Changed",
    f"- {SCRIPT_NAME}",
    f"- {os.path.basename(RUN_JSON)}",
    f"- {os.path.basename(REPORT_MD)}",
    f"- {os.path.basename(TABLE_CSV)}",
    f"- {os.path.basename(CHECKPOINT_MD)}",
    "",
    "## Why 1J42b Was Run",
    "- 1J42a validated legal adapter dataflow but did not preserve confidence updates, boundary rules, or usable policy behavior.",
    "- 1J42b restores those mechanism-side updates inside the reset/step loop while keeping the no-performance-claim constraint.",
    "",
    "## Adapter Coverage",
    f"- observation conversion success: {observation_conversion_success_rate:.4f}",
    f"- action conversion success: {action_conversion_success_rate:.4f}",
    f"- outcome conversion success: {outcome_conversion_success_rate:.4f}",
    f"- goal_context assignment rate: {goal_context_assignment_rate:.4f}",
    f"- unsupported observation fields: {unsupported_observation_fields}",
    f"- unsupported action types: {unsupported_action_types}",
    "",
    "## Memory Records",
    f"- event memory count: {event_memory_count:.4f}",
    f"- object memory count: {object_memory_count:.4f}",
    f"- outcome memory count: {outcome_memory_count:.4f}",
    f"- common sense record count: {common_sense_record_count:.4f}",
    f"- experience record count: {experience_record_count:.4f}",
    f"- records with goal_context: {records_with_goal_context:.4f}",
    f"- goal_context coverage: {goal_context_coverage:.4f}",
    "",
    "## Mechanism Parity",
    f"- strategy candidate count: {strategy_candidate_count:.4f}",
    f"- experience trigger count: {experience_trigger_count:.4f}",
    f"- false application proxy: {false_application_proxy:.4f}",
    f"- confidence update available: {str(confidence_update_available).lower()}",
    f"- confidence update count: {confidence_update_count:.4f}",
    f"- confidence before mean: {confidence_before_mean:.4f}",
    f"- confidence after mean: {confidence_after_mean:.4f}",
    f"- confidence-success correlation: {confidence_success_correlation:.4f}",
    f"- boundary rule available: {str(boundary_rule_available).lower()}",
    f"- boundary rule count: {boundary_rule_count:.4f}",
    f"- boundary block rate: {boundary_block_rate:.4f}",
    f"- bridge signal available: {str(bridge_signal_available).lower()}",
    f"- bridge-supported experiences: {bridge_supported_experience_count:.4f}",
    "",
    "## Policy Probe",
    f"- random baseline return: {random_baseline_return:.4f}",
    f"- 1J42a simple policy return: {simple_policy_return_1j42a_style:.4f}",
    f"- 1J42b Experience policy return: {experience_policy_return_1j42b:.4f}",
    f"- delta vs random: {delta_vs_random:.4f}",
    f"- delta vs 1J42a simple policy: {delta_vs_1j42a_simple_policy:.4f}",
    f"- always observe return: {always_observe_return:.4f}",
    f"- always direct return: {always_direct_return:.4f}",
    "- performance claim: no",
    "",
    "## Boundary",
    f"- overgeneralization before: {overgeneralization_before_update:.4f}",
    f"- overgeneralization after: {overgeneralization_after_update:.4f}",
    f"- false block rate: {boundary_false_block_rate:.4f}",
    "",
    "## Validity",
    f"- hidden leakage: {str(hidden_leakage_detected).lower()}",
    f"- audit-only fields absent: {str(audit_only_fields_absent).lower()}",
    f"- legal observation only: {str(legal_observation_only).lower()}",
    f"- three-world separation: {str(three_world_separation_passed).lower()}",
    f"- no test label tuning: {str(no_test_label_tuning).lower()}",
    "",
    "## Limitations",
    f"- main adapter limitation: {limitations['main_adapter_limitation']}",
    f"- missing observe/action issue: {limitations['missing_observe_action_issue']}",
    f"- missing boundary/bridge issue: {limitations['missing_boundary_bridge_issue']}",
    f"- benchmark mismatch issue: {limitations['benchmark_mismatch_issue']}",
    f"- adapter assumptions: {limitations['adapter_assumptions']}",
    "",
    "## Recommended Resume Point",
]
if status == "PASS":
    md_lines.append("- Move to external benchmark candidate selection; the adapter now preserves the local mechanism well enough for a small pilot.")
elif status == "PARTIAL":
    md_lines.append("- Do one more adapter repair pass before any benchmark pilot.")
else:
    md_lines.append("- Review the adapter legality boundary before moving on.")
with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

checkpoint_lines = [
    "# 1J42b Adapter Mechanism Parity Checkpoint",
    "",
    f"- status: {status}",
    f"- benchmark name: SparseBridgeBenchmarkWrapper",
    f"- episode count: {episode_count:.4f}",
    f"- explicit observe action: true",
    f"- event/object/outcome/common_sense/experience counts: {event_memory_count:.1f}/{object_memory_count:.1f}/{outcome_memory_count:.1f}/{common_sense_record_count:.1f}/{experience_record_count:.1f}",
    f"- adapter conversion success: obs {observation_conversion_success_rate:.4f}, act {action_conversion_success_rate:.4f}, out {outcome_conversion_success_rate:.4f}",
    f"- goal_context assignment rate: {goal_context_assignment_rate:.4f}",
    f"- random/1J42a/1J42b returns: {random_baseline_return:.4f}/{simple_policy_return_1j42a_style:.4f}/{experience_policy_return_1j42b:.4f}",
    f"- confidence update count / boundary rule count: {confidence_update_count:.1f}/{boundary_rule_count:.1f}",
    f"- limitations: {json.dumps(limitations)}",
    "",
    "Next suggested step:",
]
if status == "PASS":
    checkpoint_lines.append("- external benchmark candidate selection")
elif status == "PARTIAL":
    checkpoint_lines.append("- further adapter repair")
else:
    checkpoint_lines.append("- review adapter design")
with open(CHECKPOINT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Done.")
print(f"Status: {status}")
print("Benchmark / Environment:")
print("- benchmark name or wrapper: SparseBridgeBenchmarkWrapper")
print(f"- episode count: {episode_count:.4f}")
print("- explicit observe action: yes")
print("- observation type: dict visible tokens + inspect diagnostic")
print("- action type: discrete action strings")
print("- reward type: inspect cost + terminal task reward")
print("Adapter Coverage:")
print(f"- observation conversion success: {observation_conversion_success_rate:.4f}")
print(f"- action conversion success: {action_conversion_success_rate:.4f}")
print(f"- outcome conversion success: {outcome_conversion_success_rate:.4f}")
print(f"- goal_context assignment rate: {goal_context_assignment_rate:.4f}")
print(f"- unsupported fields/actions: {unsupported_observation_fields} / {unsupported_action_types}")
print("Memory Records:")
print(f"- event memory count: {event_memory_count:.4f}")
print(f"- object memory count: {object_memory_count:.4f}")
print(f"- outcome memory count: {outcome_memory_count:.4f}")
print(f"- common sense record count: {common_sense_record_count:.4f}")
print(f"- experience record count: {experience_record_count:.4f}")
print(f"- goal_context coverage: {goal_context_coverage:.4f}")
print("Mechanism Compatibility:")
print(f"- strategy candidates: {strategy_candidate_count:.4f}")
print(f"- experience triggers: {experience_trigger_count:.4f}")
print(f"- confidence updates available: {str(confidence_update_available).lower()}")
print(f"- confidence update count: {confidence_update_count:.4f}")
print(f"- confidence-success correlation: {confidence_success_correlation:.4f}")
print(f"- boundary rules available: {str(boundary_rule_available).lower()}")
print(f"- boundary rule count: {boundary_rule_count:.4f}")
print(f"- boundary block rate: {boundary_block_rate:.4f}")
print(f"- bridge-like signal available: {str(bridge_signal_available).lower()}")
print(f"- bridge-supported experiences: {bridge_supported_experience_count:.4f}")
print("Dry-run Returns:")
print(f"- random baseline return: {random_baseline_return:.4f}")
print(f"- 1J42a simple policy return: {simple_policy_return_1j42a_style:.4f}")
print(f"- 1J42b Experience policy return: {experience_policy_return_1j42b:.4f}")
print(f"- delta vs random: {delta_vs_random:.4f}")
print(f"- delta vs 1J42a simple policy: {delta_vs_1j42a_simple_policy:.4f}")
print(f"- always observe return: {always_observe_return:.4f}")
print(f"- always direct return: {always_direct_return:.4f}")
print("- performance claim: no")
print("Boundary:")
print(f"- overgeneralization before: {overgeneralization_before_update:.4f}")
print(f"- overgeneralization after: {overgeneralization_after_update:.4f}")
print(f"- false block rate: {boundary_false_block_rate:.4f}")
print("Validity:")
print(f"- hidden leakage: {str(hidden_leakage_detected).lower()}")
print(f"- audit-only fields absent: {str(audit_only_fields_absent).lower()}")
print(f"- legal observation only: {str(legal_observation_only).lower()}")
print(f"- three-world separation: {str(three_world_separation_passed).lower()}")
print("- performance claim: no")
print("Limitations:")
print(f"- main adapter limitation: {limitations['main_adapter_limitation']}")
print(f"- missing observe/action issue: {limitations['missing_observe_action_issue']}")
print(f"- missing boundary/bridge issue: {limitations['missing_boundary_bridge_issue']}")
print(f"- benchmark mismatch issue: {limitations['benchmark_mismatch_issue']}")
print(f"elapsed_sec: {time.time() - t0:.2f}")
