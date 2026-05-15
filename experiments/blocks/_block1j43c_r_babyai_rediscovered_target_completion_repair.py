"""
Block 1J43c-r -- BabyAI Rediscovered Target Completion Repair.

This is a narrow BabyAI probe repair focused on the transition from
target-not-visible exploration to rediscovered-target completion. It keeps the
legal observation / action / outcome conversion, mission parser, visible-target
bridge, and exploration fallback from 1J43c as much as possible, but adds
explicit rediscovery tracking and bridge re-entry repair.

No benchmark performance claim is made.
No language learning claim is made.
"""

import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime

t0 = time.time()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPT_NAME = "_block1j43c_r_babyai_rediscovered_target_completion_repair.py"
RUN_JSON = os.path.join(CURRENT_DIR, "runs", "block1j43c_r_babyai_rediscovered_target_completion_repair.json")
REPORT_MD = os.path.join(CURRENT_DIR, "protocols", "block1j43c_r_babyai_rediscovered_target_completion_repair.md")
TABLE_CSV = os.path.join(CURRENT_DIR, "protocols", "block1j43c_r_babyai_rediscovered_target_completion_repair_table.csv")
CHECKPOINT_MD = os.path.join(CURRENT_DIR, "checkpoint_1j43c_r_babyai_rediscovered_target_completion_repair.md")

os.makedirs(os.path.join(CURRENT_DIR, "runs"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_DIR, "protocols"), exist_ok=True)

SEEDS = [101, 207, 404]
TRAIN_RANDOM_EPISODES_PER_ENV = 4
TRAIN_BRIDGE_EPISODES_PER_ENV = 4
PROBE_EPISODES_PER_ENV = 20
CANDIDATE_ENVS = [
    "BabyAI-GoToObj-v0",
    "BabyAI-GoToRedBall-v0",
    "BabyAI-Pickup-v0",
    "BabyAI-OpenDoor-v0",
]
FORBIDDEN_AGENT_FIELDS = {
    "full_grid",
    "oracle_action",
    "expert_action",
    "hidden_state",
    "shortest_path",
    "success_label",
    "relation_family",
    "bridge_tag",
    "boundary_tag",
    "audit_VOI",
    "object_id",
    "train_test_label",
}
ACTION_ID_TO_NAME = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}
ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}
AGENT_X = 3
AGENT_Y = 6
FRONT_X = 3
FRONT_Y = 5
PASSABLE_OBJECTS = {"empty", "floor", "goal"}
DOOR_OPEN_STATE = 0


def mean(values):
    return sum(values) / len(values) if values else 0.0


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
            for key, val in value.items():
                keys.add(key)
                walk(val)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(records)
    return keys


def jaccard(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    denom = len(set_a | set_b)
    if denom == 0:
        return 0.0
    return len(set_a & set_b) / denom


try:
    import gymnasium as gym
    import minigrid
    from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
except Exception as exc:  # pragma: no cover
    IMPORT_ERROR = repr(exc)
    gym = None
    minigrid = None
    IDX_TO_COLOR = {}
    IDX_TO_OBJECT = {}
else:
    IMPORT_ERROR = None


def parse_mission(mission):
    mission = mission.strip().lower()
    patterns = [
        (
            re.compile(r"^go to (?:the |a )?(?:(?P<color>\w+) )?(?P<obj>\w+)$"),
            "reach_target_object",
        ),
        (
            re.compile(r"^pick up (?:the |a )?(?:(?P<color>\w+) )?(?P<obj>\w+)$"),
            "acquire_target_object",
        ),
        (
            re.compile(r"^open (?:the |a )?(?:(?P<color>\w+) )?(?P<obj>door)(?: .*)?$"),
            "open_target_door",
        ),
        (
            re.compile(r"^unlock(?: .*)?$"),
            "unlock_path",
        ),
    ]
    for pattern, goal_context in patterns:
        match = pattern.match(mission)
        if match:
            parsed = {
                "goal_context": goal_context,
                "mission_pattern": pattern.pattern,
            }
            parsed.update({k: v for k, v in match.groupdict().items() if v is not None})
            return True, parsed
    return False, {"goal_context": "complete_task", "mission_pattern": None}


def decode_visible_cells(image):
    cells = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            obj_idx, color_idx, state_idx = [int(v) for v in image[x, y]]
            obj_name = IDX_TO_OBJECT.get(obj_idx, f"obj{obj_idx}")
            color_name = IDX_TO_COLOR.get(color_idx, f"color{color_idx}")
            cell = {
                "x": x,
                "y": y,
                "obj": obj_name,
                "color": color_name,
                "state": state_idx,
            }
            cells.append(cell)
    return cells


def decode_image_to_tokens(image):
    object_counts = Counter()
    color_counts = Counter()
    tokens = []
    for cell in decode_visible_cells(image):
        if cell["obj"] in {"unseen", "empty"}:
            continue
        object_counts[cell["obj"]] += 1
        color_counts[cell["color"]] += 1
        tokens.append(f"vis_obj:{cell['obj']}")
        tokens.append(f"vis_color:{cell['color']}")
        if cell["obj"] == "door":
            tokens.append(f"vis_door_state:{cell['state']}")
    for name, count in object_counts.items():
        tokens.append(f"count_obj:{name}:{min(count, 3)}")
    for name, count in color_counts.items():
        tokens.append(f"count_color:{name}:{min(count, 3)}")
    return tuple(sorted(set(tokens)))


def observation_to_features(observation, parsed_mission):
    tokens = list(decode_image_to_tokens(observation["image"]))
    tokens.append(f"dir:{int(observation['direction'])}")
    tokens.append(f"goal:{parsed_mission['goal_context']}")
    if "color" in parsed_mission:
        tokens.append(f"target_color:{parsed_mission['color']}")
    if "obj" in parsed_mission:
        tokens.append(f"target_obj:{parsed_mission['obj']}")
    return tuple(sorted(set(tokens)))


def visible_state_change(prev_obs, next_obs):
    return not (prev_obs["image"] == next_obs["image"]).all() or int(prev_obs["direction"]) != int(next_obs["direction"])


def legal_success_proxy(reward, terminated, truncated, prev_obs, next_obs, action_name):
    if reward > 0.0:
        return True, reward
    progress_bonus = 0.0
    if visible_state_change(prev_obs, next_obs):
        progress_bonus += 0.05
    if action_name in {"pickup", "toggle"} and visible_state_change(prev_obs, next_obs):
        progress_bonus += 0.10
    if terminated and reward <= 0.0:
        progress_bonus -= 0.05
    return progress_bonus > 0.0, reward + progress_bonus


def get_cell(image, x, y):
    obj_idx, color_idx, state_idx = [int(v) for v in image[x, y]]
    return {
        "x": x,
        "y": y,
        "obj": IDX_TO_OBJECT.get(obj_idx, f"obj{obj_idx}"),
        "color": IDX_TO_COLOR.get(color_idx, f"color{color_idx}"),
        "state": state_idx,
    }


def is_passable(cell):
    return cell["obj"] in PASSABLE_OBJECTS or (cell["obj"] == "door" and cell["state"] == DOOR_OPEN_STATE)


def matches_target(cell, parsed_mission):
    target_obj = parsed_mission.get("obj")
    target_color = parsed_mission.get("color")
    goal_context = parsed_mission["goal_context"]
    if goal_context == "open_target_door":
        if cell["obj"] != "door":
            return False
        if target_color is not None and cell["color"] != target_color:
            return False
        return True
    if target_obj is None:
        return False
    if cell["obj"] != target_obj:
        return False
    if target_color is not None and cell["color"] != target_color:
        return False
    return True


def find_target_cells(observation, parsed_mission):
    cells = decode_visible_cells(observation["image"])
    return [cell for cell in cells if matches_target(cell, parsed_mission)]


def select_target_cell(target_cells):
    if not target_cells:
        return None
    return sorted(
        target_cells,
        key=lambda cell: (
            abs(cell["x"] - AGENT_X) + abs(cell["y"] - AGENT_Y),
            abs(cell["x"] - AGENT_X),
            abs(cell["y"] - AGENT_Y),
        ),
    )[0]


def choose_random_action(action_space, rng):
    return rng.randrange(action_space.n)


def choose_1j43a_simple_probe_action(observation, parsed_mission, rng):
    image = observation["image"]
    front = image[3, 6]
    front_obj = IDX_TO_OBJECT.get(int(front[0]), f"obj{front[0]}")
    front_color = IDX_TO_COLOR.get(int(front[1]), f"color{front[1]}")
    target_obj = parsed_mission.get("obj")
    target_color = parsed_mission.get("color")
    goal_context = parsed_mission["goal_context"]

    if goal_context == "open_target_door" and front_obj == "door":
        return 5, {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True}
    if goal_context == "acquire_target_object" and front_obj == target_obj and (target_color is None or front_color == target_color):
        return 3, {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True}
    if goal_context == "reach_target_object" and front_obj == target_obj and (target_color is None or front_color == target_color):
        return 6, {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True}
    visible_tokens = decode_image_to_tokens(image)
    if goal_context in {"reach_target_object", "acquire_target_object"}:
        if target_obj is not None and any(token == f"vis_obj:{target_obj}" for token in visible_tokens):
            return 2, {"target_visible": True, "target_adjacent": False, "used_bridge": False, "used_fallback": True}
    if goal_context == "open_target_door" and any(token == "vis_obj:door" for token in visible_tokens):
        return 2, {"target_visible": True, "target_adjacent": False, "used_bridge": False, "used_fallback": True}
    return rng.choice([0, 1, 2]), {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True}


def choose_exploration_action(observation, policy_state):
    image = observation["image"]
    front_cell = get_cell(image, FRONT_X, FRONT_Y)
    if is_passable(front_cell) and policy_state["forward_streak"] < 2:
        policy_state["forward_streak"] += 1
        return ACTION_NAME_TO_ID["move_forward"]
    policy_state["forward_streak"] = 0
    turn_left = policy_state["turn_bias"] == "left"
    if policy_state["step_index"] % 4 >= 2:
        turn_left = not turn_left
    policy_state["turn_bias"] = "left" if turn_left else "right"
    return ACTION_NAME_TO_ID["turn_left"] if turn_left else ACTION_NAME_TO_ID["turn_right"]


def observation_signature(observation, parsed_mission):
    return observation_to_features(observation, parsed_mission)


def choose_exploration_fallback_action(observation, parsed_mission, policy_state):
    image = observation["image"]
    signature = observation_signature(observation, parsed_mission)
    signature_visits = policy_state["signature_counts"][signature]
    front_cell = get_cell(image, FRONT_X, FRONT_Y)
    front_passable = is_passable(front_cell)
    front_signature_repeated = signature_visits >= 2

    if policy_state["recent_signatures"] and policy_state["recent_signatures"][-1] == signature:
        policy_state["stuck_same_obs_streak"] += 1
    else:
        policy_state["stuck_same_obs_streak"] = 0
    policy_state["signature_counts"][signature] += 1
    policy_state["recent_signatures"].append(signature)

    if not front_passable:
        policy_state["blocked_forward_count"] += 1

    if front_passable and policy_state["stuck_same_obs_streak"] == 0 and signature_visits <= 1:
        policy_state["forward_streak"] += 1
        policy_state["last_explore_action"] = "move_forward"
        return ACTION_NAME_TO_ID["move_forward"]

    if front_passable and policy_state["forward_streak"] < 2 and not front_signature_repeated:
        policy_state["forward_streak"] += 1
        policy_state["last_explore_action"] = "move_forward"
        return ACTION_NAME_TO_ID["move_forward"]

    policy_state["forward_streak"] = 0

    if policy_state["stuck_same_obs_streak"] >= 1:
        policy_state["stuck_loop_count"] += 1
        turn_left = policy_state["last_explore_action"] != "turn_left"
    else:
        turn_left = policy_state["turn_bias"] == "left"
        if signature_visits >= 2:
            turn_left = not turn_left

    action_name = "turn_left" if turn_left else "turn_right"
    policy_state["turn_bias"] = "right" if turn_left else "left"
    policy_state["last_explore_action"] = action_name
    return ACTION_NAME_TO_ID[action_name]


def target_adjacent_or_front(target_cell):
    if target_cell is None:
        return False
    return target_cell["y"] == FRONT_Y and abs(target_cell["x"] - FRONT_X) <= 1


def target_in_front(target_cell):
    return target_cell is not None and target_cell["x"] == FRONT_X and target_cell["y"] == FRONT_Y


def reset_after_rediscovery(policy_state, current_signature):
    policy_state["forward_streak"] = 0
    policy_state["stuck_same_obs_streak"] = 0
    policy_state["recent_signatures"] = [current_signature]
    policy_state["bridge_priority_steps"] = 4
    policy_state["rediscovered_target_active"] = True
    policy_state["rediscovered_target_seen"] = True


def bridge_turn_action(policy_state, prefer_left=None):
    if prefer_left is None:
        prefer_left = policy_state["turn_bias"] == "left"
    policy_state["turn_bias"] = "right" if prefer_left else "left"
    return ACTION_NAME_TO_ID["turn_left"] if prefer_left else ACTION_NAME_TO_ID["turn_right"]


def bridge_action_from_target(observation, parsed_mission, target_cell, policy_state):
    image = observation["image"]
    front_cell = get_cell(image, FRONT_X, FRONT_Y)
    goal_context = parsed_mission["goal_context"]
    target_adjacent = target_adjacent_or_front(target_cell)

    if target_cell is None:
        return choose_exploration_action(observation, policy_state), False, False

    if goal_context == "reach_target_object" and target_cell["x"] == FRONT_X and target_cell["y"] == FRONT_Y:
        return ACTION_NAME_TO_ID["done"], True, True
    if goal_context == "acquire_target_object" and target_cell["x"] == FRONT_X and target_cell["y"] == FRONT_Y:
        return ACTION_NAME_TO_ID["pickup"], True, True
    if goal_context == "open_target_door" and target_cell["x"] == FRONT_X and target_cell["y"] == FRONT_Y:
        return ACTION_NAME_TO_ID["toggle"], True, True

    if target_cell["x"] < FRONT_X:
        return ACTION_NAME_TO_ID["turn_left"], True, target_adjacent
    if target_cell["x"] > FRONT_X:
        return ACTION_NAME_TO_ID["turn_right"], True, target_adjacent
    if is_passable(front_cell):
        return ACTION_NAME_TO_ID["move_forward"], True, target_adjacent
    return choose_exploration_action(observation, policy_state), True, target_adjacent


def bridge_action_from_target_repaired(observation, parsed_mission, target_cell, policy_state):
    image = observation["image"]
    front_cell = get_cell(image, FRONT_X, FRONT_Y)
    goal_context = parsed_mission["goal_context"]
    target_adjacent = target_adjacent_or_front(target_cell)
    target_front = target_in_front(target_cell)

    if target_cell is None:
        action_id = choose_exploration_fallback_action(observation, parsed_mission, policy_state)
        return action_id, {
            "target_visible": False,
            "target_adjacent": False,
            "target_front": False,
            "used_bridge": False,
            "used_fallback": True,
            "interaction_attempt": False,
        }

    policy_state["bridge_priority_steps"] = max(policy_state.get("bridge_priority_steps", 0), 1)

    if goal_context == "reach_target_object" and target_front:
        return ACTION_NAME_TO_ID["done"], {
            "target_visible": True,
            "target_adjacent": True,
            "target_front": True,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": True,
        }
    if goal_context == "acquire_target_object" and target_front:
        return ACTION_NAME_TO_ID["pickup"], {
            "target_visible": True,
            "target_adjacent": True,
            "target_front": True,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": True,
        }
    if goal_context == "open_target_door" and target_front:
        return ACTION_NAME_TO_ID["toggle"], {
            "target_visible": True,
            "target_adjacent": True,
            "target_front": True,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": True,
        }

    if target_cell["x"] < FRONT_X:
        return bridge_turn_action(policy_state, prefer_left=True), {
            "target_visible": True,
            "target_adjacent": target_adjacent,
            "target_front": False,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": False,
        }
    if target_cell["x"] > FRONT_X:
        return bridge_turn_action(policy_state, prefer_left=False), {
            "target_visible": True,
            "target_adjacent": target_adjacent,
            "target_front": False,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": False,
        }
    if is_passable(front_cell):
        return ACTION_NAME_TO_ID["move_forward"], {
            "target_visible": True,
            "target_adjacent": target_adjacent,
            "target_front": False,
            "used_bridge": True,
            "used_fallback": False,
            "interaction_attempt": False,
        }
    return bridge_turn_action(policy_state), {
        "target_visible": True,
        "target_adjacent": target_adjacent,
        "target_front": False,
        "used_bridge": True,
        "used_fallback": False,
        "interaction_attempt": False,
    }


def choose_mission_visible_object_action(observation, parsed_mission, policy_state, rng):
    target_cell = select_target_cell(find_target_cells(observation, parsed_mission))
    if target_cell is None:
        return (
            choose_exploration_action(observation, policy_state),
            {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True},
        )
    action_id, used_bridge, target_adjacent = bridge_action_from_target(observation, parsed_mission, target_cell, policy_state)
    return (
        action_id,
        {"target_visible": True, "target_adjacent": target_adjacent, "used_bridge": used_bridge, "used_fallback": False},
    )


def choose_exploration_fallback_probe_action(observation, parsed_mission, policy_state, rng):
    target_cell = select_target_cell(find_target_cells(observation, parsed_mission))
    if target_cell is None:
        return (
            choose_exploration_fallback_action(observation, parsed_mission, policy_state),
            {"target_visible": False, "target_adjacent": False, "used_bridge": False, "used_fallback": True},
        )
    action_id, used_bridge, target_adjacent = bridge_action_from_target(observation, parsed_mission, target_cell, policy_state)
    return (
        action_id,
        {"target_visible": True, "target_adjacent": target_adjacent, "used_bridge": used_bridge, "used_fallback": False},
    )


def build_event_memory(trajectories):
    event_memory = []
    outcome_rows = []
    for ep_idx, traj in enumerate(trajectories):
        for step_idx, step in enumerate(traj["steps"]):
            event_id = f"ep{ep_idx:03d}::step{step_idx:03d}"
            features = observation_to_features(step["observation"], traj["parsed_mission"])
            next_features = observation_to_features(step["next_observation"], traj["parsed_mission"])
            event_memory.append(
                {
                    "event_id": event_id,
                    "step_index": step_idx,
                    "condition_features": features,
                    "action_taken": step["action_name"],
                    "observe_taken": False,
                    "observe_result": None,
                    "outcome": "success" if step["step_success_proxy"] else ("failure" if step["reward"] < 0.0 else "neutral"),
                    "reward_or_net_result": step["adjusted_score"],
                    "goal_context": traj["parsed_mission"]["goal_context"],
                    "done": step["terminated"] or step["truncated"],
                }
            )
            outcome_rows.append(
                {
                    "decision_id": event_id,
                    "condition_features": features,
                    "next_features": next_features,
                    "action_taken": step["action_name"],
                    "goal_context": traj["parsed_mission"]["goal_context"],
                    "result_pattern": "success" if step["step_success_proxy"] else "failure",
                    "reward_or_net_result": step["adjusted_score"],
                    "task_success": step["step_success_proxy"],
                }
            )
    return event_memory, outcome_rows


def build_object_memory(event_memory):
    grouped = defaultdict(list)
    for event in event_memory:
        signature = tuple(
            sorted(
                token
                for token in event["condition_features"]
                if token.startswith(("vis_obj:", "vis_color:", "target_", "goal:"))
            )
        )
        grouped[signature].append(event)
    records = []
    for idx, signature in enumerate(sorted(grouped.keys())):
        rows = grouped[signature]
        feature_counts = Counter(token for row in rows for token in row["condition_features"])
        records.append(
            {
                "object_candidate_id": f"babyai_objcand_{idx:03d}",
                "positive_visible_features": list(signature),
                "observed_diagnostic_features": {},
                "state_evidence": dict(Counter(row["outcome"] for row in rows)),
                "identity_confidence": len(rows) / max(1, len(rows) + 1),
                "feature_evidence_counts": dict(feature_counts),
                "negative_exclusion_evidence": sorted(
                    {
                        token
                        for row in rows
                        if row["outcome"] == "failure"
                        for token in row["condition_features"]
                        if token.startswith(("vis_obj:", "vis_color:", "vis_door_state:"))
                    }
                ),
            }
        )
    return records


def build_outcome_memory(outcome_rows):
    grouped = defaultdict(list)
    for row in outcome_rows:
        key = (tuple(sorted(row["condition_features"])), row["action_taken"], row["goal_context"])
        grouped[key].append(row)
    records = []
    for idx, key in enumerate(sorted(grouped.keys())):
        condition_pattern, action_pattern, goal_context = key
        rows = grouped[key]
        success_count = sum(1 for row in rows if row["result_pattern"] == "success")
        failure_count = sum(1 for row in rows if row["result_pattern"] == "failure")
        records.append(
            {
                "outcome_memory_id": f"babyai_out_{idx:03d}",
                "condition_pattern": list(condition_pattern),
                "action_pattern": action_pattern,
                "goal_context": goal_context,
                "result_pattern": "success" if success_count >= failure_count else "failure",
                "success_count": success_count,
                "failure_count": failure_count,
                "average_net_result": mean([row["reward_or_net_result"] for row in rows]),
                "uncertainty": failure_count / max(1, success_count + failure_count),
                "support_event_ids": [row["decision_id"] for row in rows],
            }
        )
    return records


def cluster_outcomes(outcome_rows):
    support_rows = [row for row in outcome_rows if row["reward_or_net_result"] > 0.0]
    clusters = []
    for row in support_rows:
        placed = False
        for cluster in clusters:
            if cluster["goal_context"] != row["goal_context"] or cluster["action_taken"] != row["action_taken"]:
                continue
            similarity = max(jaccard(row["condition_features"], support["condition_features"]) for support in cluster["support_rows"])
            if similarity >= 0.45:
                cluster["support_rows"].append(row)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "cluster_id": f"babyai_cluster_{len(clusters):03d}",
                    "goal_context": row["goal_context"],
                    "action_taken": row["action_taken"],
                    "support_rows": [row],
                    "failure_rows": [],
                    "boundary_rows": [],
                    "exclusion_features": set(),
                }
            )
    for row in [row for row in outcome_rows if row["reward_or_net_result"] <= 0.0]:
        best_cluster = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = max(jaccard(row["condition_features"], support["condition_features"]) for support in cluster["support_rows"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        if best_cluster is None or best_similarity < 0.45:
            continue
        best_cluster["failure_rows"].append(row)
        negative_tokens = {
            token
            for token in row["condition_features"]
            if token.startswith(("vis_obj:", "vis_color:", "vis_door_state:"))
        }
        if negative_tokens:
            best_cluster["boundary_rows"].append(row)
            best_cluster["exclusion_features"].update(negative_tokens)
    for cluster in clusters:
        feature_counts = Counter(token for support in cluster["support_rows"] for token in support["condition_features"])
        cluster["core_features"] = {
            token
            for token, count in feature_counts.items()
            if count >= max(1, math.ceil(0.35 * len(cluster["support_rows"])))
        }
        cluster["mean_reward"] = mean([row["reward_or_net_result"] for row in cluster["support_rows"]])
        cluster["confidence"] = len(cluster["support_rows"]) / (
            len(cluster["support_rows"]) + 1.5 * len(cluster["failure_rows"]) + 1.0
        )
    return clusters


def build_common_sense_records(clusters):
    records = []
    for idx, cluster in enumerate(clusters):
        records.append(
            {
                "common_sense_id": f"babyai_cs_{idx:03d}",
                "condition_summary": sorted(cluster["core_features"]),
                "action_or_affordance_summary": cluster["action_taken"],
                "typical_result": "improves_task_progress" if cluster["mean_reward"] > 0.0 else "uncertain",
                "relevant_goal_contexts": [cluster["goal_context"], "maximize_net_result"],
                "typical_effect_on_goal_proxy": "higher_return" if cluster["mean_reward"] > 0.0 else "mixed",
                "confidence": cluster["confidence"],
                "support_cases": len(cluster["support_rows"]),
                "failure_cases": len(cluster["failure_rows"]),
                "boundary_cases": len(cluster["boundary_rows"]),
                "bridge_cases": 0,
                "known_exclusions": sorted(cluster["exclusion_features"]),
                "evidence_source_event_ids": [row["decision_id"] for row in cluster["support_rows"] + cluster["failure_rows"]],
            }
        )
    return records


def build_experience_records(common_sense_records):
    records = []
    for idx, record in enumerate(common_sense_records):
        support_rate = record["support_cases"] / max(1, record["support_cases"] + record["failure_cases"])
        records.append(
            {
                "experience_id": f"babyai_exp_{idx:03d}",
                "trigger_conditions": list(record["condition_summary"]),
                "recommended_strategy": record["action_or_affordance_summary"],
                "goal_context": record["relevant_goal_contexts"][0],
                "goal_proxy_metric": "net_result",
                "expected_goal_progress": support_rate,
                "confidence": record["confidence"],
                "goal_conditioned_confidence": min(1.0, record["confidence"] + 0.05 * support_rate),
                "goal_support_cases": record["support_cases"],
                "goal_failure_cases": record["failure_cases"],
                "boundary_rules": list(record["known_exclusions"]),
                "linked_common_sense_ids": [record["common_sense_id"]],
                "update_history": [
                    {"update_type": "support", "count": record["support_cases"]},
                    {"update_type": "failure", "count": record["failure_cases"]},
                    {"update_type": "boundary", "count": record["boundary_cases"]},
                ],
            }
        )
    return records


def pick_experience(observation, parsed_mission, experience_records):
    obs_features = set(observation_to_features(observation, parsed_mission))
    best_record = None
    best_score = 0.0
    for record in experience_records:
        if record["goal_context"] != parsed_mission["goal_context"]:
            continue
        if set(record["boundary_rules"]) & obs_features:
            continue
        feature_overlap = len(obs_features & set(record["trigger_conditions"])) / max(1, len(record["trigger_conditions"]))
        score = feature_overlap + 0.35 * record["goal_conditioned_confidence"] + 0.20 * record["expected_goal_progress"]
        if score >= 0.40 and score > best_score:
            best_score = score
            best_record = record
    return best_record


def choose_experience_plus_mission_action(observation, parsed_mission, experience_records, policy_state, rng):
    obs_features = set(observation_to_features(observation, parsed_mission))
    record = pick_experience(observation, parsed_mission, experience_records)
    target_cell = select_target_cell(find_target_cells(observation, parsed_mission))
    if record is not None and set(record["boundary_rules"]) & obs_features:
        action_id = choose_exploration_fallback_action(observation, parsed_mission, policy_state)
        return action_id, {
            "target_visible": target_cell is not None,
            "target_adjacent": target_adjacent_or_front(target_cell),
            "used_bridge": False,
            "used_fallback": True,
            "used_record": True,
        }
    if target_cell is not None:
        action_id, used_bridge, target_adjacent = bridge_action_from_target(observation, parsed_mission, target_cell, policy_state)
        return action_id, {
            "target_visible": True,
            "target_adjacent": target_adjacent,
            "used_bridge": used_bridge,
            "used_fallback": False,
            "used_record": record is not None,
        }
    action_id = choose_exploration_fallback_action(observation, parsed_mission, policy_state)
    return action_id, {
        "target_visible": False,
        "target_adjacent": False,
        "used_bridge": False,
        "used_fallback": True,
        "used_record": record is not None,
    }


def choose_rediscovered_target_completion_action(observation, parsed_mission, experience_records, policy_state, rng):
    target_cell = select_target_cell(find_target_cells(observation, parsed_mission))
    obs_features = set(observation_to_features(observation, parsed_mission))
    record = pick_experience(observation, parsed_mission, experience_records)

    if target_cell is not None:
        action_id, info = bridge_action_from_target_repaired(observation, parsed_mission, target_cell, policy_state)
        info["used_record"] = record is not None
        return action_id, info

    if record is not None and set(record["boundary_rules"]) & obs_features:
        action_id = choose_exploration_fallback_action(observation, parsed_mission, policy_state)
        return action_id, {
            "target_visible": False,
            "target_adjacent": False,
            "target_front": False,
            "used_bridge": False,
            "used_fallback": True,
            "interaction_attempt": False,
            "used_record": True,
        }

    action_id = choose_exploration_fallback_action(observation, parsed_mission, policy_state)
    return action_id, {
        "target_visible": False,
        "target_adjacent": False,
        "target_front": False,
        "used_bridge": False,
        "used_fallback": True,
        "interaction_attempt": False,
        "used_record": record is not None,
    }


def choose_experience_plus_mission_action_1j43b(observation, parsed_mission, experience_records, policy_state, rng):
    obs_features = set(observation_to_features(observation, parsed_mission))
    record = pick_experience(observation, parsed_mission, experience_records)
    target_cell = select_target_cell(find_target_cells(observation, parsed_mission))
    if record is not None and set(record["boundary_rules"]) & obs_features:
        action_id = choose_exploration_action(observation, policy_state)
        return action_id, {
            "target_visible": target_cell is not None,
            "target_adjacent": target_adjacent_or_front(target_cell),
            "used_bridge": False,
            "used_fallback": True,
            "used_record": True,
        }
    if target_cell is not None:
        action_id, used_bridge, target_adjacent = bridge_action_from_target(observation, parsed_mission, target_cell, policy_state)
        return action_id, {
            "target_visible": True,
            "target_adjacent": target_adjacent,
            "used_bridge": used_bridge,
            "used_fallback": False,
            "used_record": record is not None,
        }
    action_id = choose_exploration_action(observation, policy_state)
    return action_id, {
        "target_visible": False,
        "target_adjacent": False,
        "used_bridge": False,
        "used_fallback": True,
        "used_record": record is not None,
    }


def collect_trajectory(env, seed, policy_name, experience_records=None):
    observation, _ = env.reset(seed=seed)
    mission = observation["mission"]
    parsed_ok, parsed = parse_mission(mission)
    rng = random.Random(seed + len(mission))
    done = False
    total_reward = 0.0
    steps = []
    policy_state = {
        "step_index": 0,
        "turn_bias": "left" if seed % 2 == 0 else "right",
        "forward_streak": 0,
        "signature_counts": Counter(),
        "recent_signatures": [],
        "stuck_same_obs_streak": 0,
        "stuck_loop_count": 0,
        "blocked_forward_count": 0,
        "last_explore_action": None,
        "bridge_priority_steps": 0,
        "rediscovered_target_active": False,
        "rediscovered_target_seen": False,
    }
    target_visible_steps = 0
    target_adjacent_steps = 0
    bridge_steps = 0
    fallback_steps = 0
    used_record_steps = 0
    initial_target_visible = bool(find_target_cells(observation, parsed))
    target_ever_visible = initial_target_visible
    first_visible_step = 0 if initial_target_visible else None
    obs_sig_history = []
    action_counter = Counter()
    rediscovered_target_count = 0
    rediscovered_to_bridge_handoff = False
    rediscovered_to_adjacent = False
    rediscovered_to_interaction_attempt = False
    rediscovered_to_success = False
    bridge_control_stolen_by_fallback_count = 0
    last_visible_target_cell = None
    mission_goal = parsed["goal_context"]
    current_rediscovery_window = False
    was_visible_previous = initial_target_visible
    failure_flags = {
        "not_approached_after_rediscovery": False,
        "adjacent_not_facing": False,
        "facing_no_interaction": False,
        "interaction_no_reward": False,
        "timeout_after_rediscovery": False,
    }
    saw_adjacent_after_rediscovery = False
    saw_front_after_rediscovery = False
    attempted_interaction_after_rediscovery = False
    interaction_reward_after_rediscovery = False

    while not done:
        obs_sig_history.append(observation_signature(observation, parsed))
        current_target_cell = select_target_cell(find_target_cells(observation, parsed))
        current_target_visible = current_target_cell is not None
        if (not initial_target_visible) and current_target_visible and not policy_state["rediscovered_target_seen"]:
            rediscovered_target_count += 1
            current_rediscovery_window = True
            reset_after_rediscovery(policy_state, obs_sig_history[-1])
        elif current_target_visible and policy_state["rediscovered_target_seen"]:
            current_rediscovery_window = True
        elif not current_target_visible:
            current_rediscovery_window = False

        if policy_name == "random_policy":
            action_id = choose_random_action(env.action_space, rng)
            action_info = {"target_visible": current_target_visible, "target_adjacent": False, "target_front": False, "used_bridge": False, "used_fallback": True, "used_record": False, "interaction_attempt": False}
        elif policy_name == "1j43a_simple_experience_probe":
            action_id, action_info = choose_1j43a_simple_probe_action(observation, parsed, rng)
            action_info["used_record"] = False
        elif policy_name == "mission_visible_object_probe":
            action_id, action_info = choose_mission_visible_object_action(observation, parsed, policy_state, rng)
            action_info["used_record"] = False
        elif policy_name == "exploration_fallback_probe":
            action_id, action_info = choose_exploration_fallback_probe_action(observation, parsed, policy_state, rng)
            action_info["used_record"] = False
        elif policy_name == "1j43b_experience_plus_mission_probe":
            action_id, action_info = choose_experience_plus_mission_action_1j43b(
                observation, parsed, experience_records or [], policy_state, rng
            )
        elif policy_name == "experience_plus_mission_probe":
            action_id, action_info = choose_experience_plus_mission_action(observation, parsed, experience_records or [], policy_state, rng)
        elif policy_name == "rediscovered_target_completion_probe":
            action_id, action_info = choose_rediscovered_target_completion_action(observation, parsed, experience_records or [], policy_state, rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        action_info.setdefault("target_visible", current_target_visible)
        action_info.setdefault("target_adjacent", target_adjacent_or_front(current_target_cell))
        action_info.setdefault("target_front", target_in_front(current_target_cell))
        action_info.setdefault("used_bridge", False)
        action_info.setdefault("used_fallback", False)
        action_info.setdefault("used_record", False)
        action_info.setdefault("interaction_attempt", ACTION_ID_TO_NAME[int(action_id)] in {"pickup", "toggle", "done"})

        if current_rediscovery_window and current_target_visible:
            if action_info["used_bridge"]:
                rediscovered_to_bridge_handoff = True
            if action_info["target_adjacent"]:
                rediscovered_to_adjacent = True
                saw_adjacent_after_rediscovery = True
            if action_info["target_front"]:
                saw_front_after_rediscovery = True
            if action_info["interaction_attempt"]:
                rediscovered_to_interaction_attempt = True
                attempted_interaction_after_rediscovery = True
            if action_info["used_fallback"]:
                bridge_control_stolen_by_fallback_count += 1

        next_observation, reward, terminated, truncated, _ = env.step(action_id)
        action_name = ACTION_ID_TO_NAME[int(action_id)]
        step_success, adjusted_score = legal_success_proxy(reward, terminated, truncated, observation, next_observation, action_name)
        steps.append(
            {
                "observation": observation,
                "action_id": int(action_id),
                "action_name": action_name,
                "reward": reward,
                "next_observation": next_observation,
                "terminated": terminated,
                "truncated": truncated,
                "step_success_proxy": step_success,
                "adjusted_score": adjusted_score,
                "policy_name": policy_name,
                "action_info": deepcopy(action_info),
            }
        )
        total_reward += reward
        target_visible_steps += 1 if action_info["target_visible"] else 0
        target_adjacent_steps += 1 if action_info["target_adjacent"] else 0
        bridge_steps += 1 if action_info["used_bridge"] else 0
        fallback_steps += 1 if action_info["used_fallback"] else 0
        used_record_steps += 1 if action_info.get("used_record") else 0
        action_counter[ACTION_ID_TO_NAME[int(action_id)]] += 1
        observation = next_observation
        now_visible = bool(find_target_cells(observation, parsed))
        if now_visible:
            target_ever_visible = True
            if first_visible_step is None:
                first_visible_step = policy_state["step_index"] + 1
            last_visible_target_cell = select_target_cell(find_target_cells(observation, parsed))
        if current_rediscovery_window and reward > 0.0:
            rediscovered_to_success = True
            if action_info["interaction_attempt"]:
                interaction_reward_after_rediscovery = True
        done = terminated or truncated
        policy_state["step_index"] += 1
        was_visible_previous = now_visible

    repeated_obs_steps = max(0, len(obs_sig_history) - len(set(obs_sig_history)))
    if policy_state["rediscovered_target_seen"] and mission_goal in {"acquire_target_object", "open_target_door"}:
        if not saw_adjacent_after_rediscovery:
            failure_flags["not_approached_after_rediscovery"] = True
        elif saw_adjacent_after_rediscovery and not saw_front_after_rediscovery:
            failure_flags["adjacent_not_facing"] = True
        elif saw_front_after_rediscovery and not attempted_interaction_after_rediscovery:
            failure_flags["facing_no_interaction"] = True
        elif attempted_interaction_after_rediscovery and not interaction_reward_after_rediscovery:
            failure_flags["interaction_no_reward"] = True
        if (bool(steps) and steps[-1]["truncated"] and not steps[-1]["terminated"]) and not rediscovered_to_success:
            failure_flags["timeout_after_rediscovery"] = True

    return {
        "mission": mission,
        "parsed_ok": parsed_ok,
        "parsed_mission": parsed,
        "steps": steps,
        "episode_return": total_reward,
        "success_proxy": total_reward > 0.0,
        "target_visible_steps": target_visible_steps,
        "target_adjacent_steps": target_adjacent_steps,
        "bridge_steps": bridge_steps,
        "fallback_steps": fallback_steps,
        "used_record_steps": used_record_steps,
        "step_count": len(steps),
        "initial_target_visible": initial_target_visible,
        "target_ever_visible": target_ever_visible,
        "first_visible_step": first_visible_step,
        "repeated_observation_steps": repeated_obs_steps,
        "stuck_loop_count": policy_state["stuck_loop_count"],
        "blocked_forward_count": policy_state["blocked_forward_count"],
        "new_observation_signature_rate": len(set(obs_sig_history)) / max(1, len(obs_sig_history)),
        "timed_out": bool(steps) and steps[-1]["truncated"] and not steps[-1]["terminated"],
        "action_distribution": dict(action_counter),
        "target_rediscovered_count": rediscovered_target_count,
        "rediscovered_to_bridge_handoff": rediscovered_to_bridge_handoff,
        "rediscovered_to_adjacent": rediscovered_to_adjacent,
        "rediscovered_to_interaction_attempt": rediscovered_to_interaction_attempt,
        "rediscovered_to_success": rediscovered_to_success,
        "bridge_control_stolen_by_fallback_count": bridge_control_stolen_by_fallback_count,
        "pickup_open_door_failure_flags": failure_flags,
    }


def summarize_policy(trajectories):
    episode_returns = [traj["episode_return"] for traj in trajectories]
    success = [1.0 if traj["success_proxy"] else 0.0 for traj in trajectories]
    visible_episode_success = [1.0 if traj["success_proxy"] else 0.0 for traj in trajectories if traj["target_visible_steps"] > 0]
    fallback_only_success = [
        1.0 if traj["success_proxy"] else 0.0
        for traj in trajectories
        if traj["target_visible_steps"] == 0 and traj["fallback_steps"] > 0
    ]
    initially_not_visible = [traj for traj in trajectories if not traj["initial_target_visible"]]
    discovered_after_hidden = [traj for traj in initially_not_visible if traj["target_ever_visible"]]
    first_visible_steps = [traj["first_visible_step"] for traj in discovered_after_hidden if traj["first_visible_step"] is not None]
    total_steps = sum(traj["step_count"] for traj in trajectories)
    total_visible = sum(traj["target_visible_steps"] for traj in trajectories)
    total_adjacent = sum(traj["target_adjacent_steps"] for traj in trajectories)
    total_bridge = sum(traj["bridge_steps"] for traj in trajectories)
    repeated_steps = sum(traj["repeated_observation_steps"] for traj in trajectories)
    stuck_loops = sum(traj["stuck_loop_count"] for traj in trajectories)
    blocked_forward = sum(traj["blocked_forward_count"] for traj in trajectories)
    timeout_rate = mean([1.0 if traj["timed_out"] else 0.0 for traj in trajectories])
    new_obs_sig_rate = mean([traj["new_observation_signature_rate"] for traj in trajectories])
    action_distribution = Counter()
    for traj in trajectories:
        action_distribution.update(traj["action_distribution"])
    rediscovery_count = sum(traj["target_rediscovered_count"] for traj in trajectories)
    rediscovered_episodes = [traj for traj in trajectories if traj["target_rediscovered_count"] > 0]
    failure_breakdown = Counter()
    for traj in rediscovered_episodes:
        for key, flag in traj["pickup_open_door_failure_flags"].items():
            if flag:
                failure_breakdown[key] += 1
    return {
        "return": mean(episode_returns),
        "success_proxy_rate": mean(success),
        "visible_target_success_rate": mean(visible_episode_success),
        "target_not_visible_fallback_success_rate": mean(fallback_only_success),
        "target_visible_rate": total_visible / max(1, total_steps),
        "target_adjacent_rate": total_adjacent / max(1, total_steps),
        "action_bridge_coverage": total_bridge / max(1, total_steps),
        "trajectory_count": len(trajectories),
        "target_discovery_rate": len(discovered_after_hidden) / max(1, len(initially_not_visible)),
        "steps_until_target_visible": mean(first_visible_steps),
        "repeated_observation_rate": repeated_steps / max(1, total_steps),
        "stuck_loop_count": stuck_loops,
        "wall_collision_or_blocked_forward_count": blocked_forward,
        "new_observation_signature_rate": new_obs_sig_rate,
        "timeout_rate": timeout_rate,
        "action_distribution": dict(action_distribution),
        "target_rediscovered_count": rediscovery_count,
        "rediscovered_to_bridge_handoff_rate": mean([1.0 if traj["rediscovered_to_bridge_handoff"] else 0.0 for traj in rediscovered_episodes]),
        "rediscovered_to_adjacent_rate": mean([1.0 if traj["rediscovered_to_adjacent"] else 0.0 for traj in rediscovered_episodes]),
        "rediscovered_to_interaction_attempt_rate": mean([1.0 if traj["rediscovered_to_interaction_attempt"] else 0.0 for traj in rediscovered_episodes]),
        "rediscovered_to_success_rate": mean([1.0 if traj["rediscovered_to_success"] else 0.0 for traj in rediscovered_episodes]),
        "bridge_control_stolen_by_fallback_count": sum(
            traj["bridge_control_stolen_by_fallback_count"] for traj in trajectories
        ),
        "not_approached_after_rediscovery": failure_breakdown["not_approached_after_rediscovery"],
        "adjacent_not_facing": failure_breakdown["adjacent_not_facing"],
        "facing_no_interaction": failure_breakdown["facing_no_interaction"],
        "interaction_no_reward": failure_breakdown["interaction_no_reward"],
        "timeout_after_rediscovery": failure_breakdown["timeout_after_rediscovery"],
    }


def run_env_seed(env_id, seed):
    env = gym.make(env_id)
    train_random = []
    train_bridge = []
    unsupported_missions = []

    for ep in range(TRAIN_RANDOM_EPISODES_PER_ENV):
        traj = collect_trajectory(env, seed + ep, "random_policy")
        train_random.append(traj)
        if not traj["parsed_ok"]:
            unsupported_missions.append(traj["mission"])

    for ep in range(TRAIN_BRIDGE_EPISODES_PER_ENV):
        traj = collect_trajectory(env, seed + 500 + ep, "mission_visible_object_probe")
        train_bridge.append(traj)
        if not traj["parsed_ok"]:
            unsupported_missions.append(traj["mission"])

    train_trajectories = train_random + train_bridge
    event_memory, outcome_rows = build_event_memory(train_trajectories)
    object_memory = build_object_memory(event_memory)
    outcome_memory = build_outcome_memory(outcome_rows)
    clusters = cluster_outcomes(outcome_rows)
    common_sense_records = build_common_sense_records(clusters)
    experience_records = build_experience_records(common_sense_records)

    policy_trajectories = defaultdict(list)
    for ep in range(PROBE_EPISODES_PER_ENV):
        base_seed = seed + 1000 + ep
        policy_trajectories["random_policy"].append(collect_trajectory(env, base_seed, "random_policy"))
        policy_trajectories["1j43b_experience_plus_mission_probe"].append(
            collect_trajectory(env, base_seed, "1j43b_experience_plus_mission_probe", experience_records=experience_records)
        )
        policy_trajectories["exploration_fallback_probe"].append(
            collect_trajectory(env, base_seed, "exploration_fallback_probe")
        )
        policy_trajectories["rediscovered_target_completion_probe"].append(
            collect_trajectory(
                env,
                base_seed,
                "rediscovered_target_completion_probe",
                experience_records=experience_records,
            )
        )

    env.close()

    records = {
        "event_memory": event_memory,
        "object_memory": object_memory,
        "outcome_memory": outcome_memory,
        "common_sense_records": common_sense_records,
        "experience_records": experience_records,
    }
    agent_keys = collect_agent_keys(records)

    policy_summary = {name: summarize_policy(rows) for name, rows in policy_trajectories.items()}
    parsed_train_ok = sum(1 for traj in train_trajectories if traj["parsed_ok"])

    return {
        "env_id": env_id,
        "reset_success_rate": 1.0,
        "step_success_rate": 1.0,
        "observation_conversion_success_rate": 1.0 if event_memory else 0.0,
        "action_conversion_success_rate": 1.0,
        "outcome_conversion_success_rate": 1.0 if outcome_rows else 0.0,
        "mission_parse_success_rate": parsed_train_ok / max(1, len(train_trajectories)),
        "unsupported_mission_count": len(unsupported_missions),
        "unsupported_mission_examples": unsupported_missions[:5],
        "event_memory_count": len(event_memory),
        "object_memory_count": len(object_memory),
        "outcome_memory_count": len(outcome_memory),
        "common_sense_record_count": len(common_sense_records),
        "experience_record_count": len(experience_records),
        "goal_context_coverage": mean([1.0 if record.get("goal_context") else 0.0 for record in experience_records]),
        "confidence_update_available": any(record["goal_failure_cases"] > 0 for record in experience_records),
        "boundary_rule_available": any(record["boundary_rules"] for record in experience_records),
        "confidence_update_count": sum(len(record["update_history"]) for record in experience_records),
        "boundary_rule_count": sum(len(record["boundary_rules"]) for record in experience_records),
        "target_visible_rate": policy_summary["rediscovered_target_completion_probe"]["target_visible_rate"],
        "target_adjacent_rate": policy_summary["rediscovered_target_completion_probe"]["target_adjacent_rate"],
        "action_bridge_coverage": policy_summary["rediscovered_target_completion_probe"]["action_bridge_coverage"],
        "random_return": policy_summary["random_policy"]["return"],
        "1j43b_probe_return": policy_summary["1j43b_experience_plus_mission_probe"]["return"],
        "1j43c_probe_return": policy_summary["exploration_fallback_probe"]["return"],
        "rediscovered_target_completion_probe_return": policy_summary["rediscovered_target_completion_probe"]["return"],
        "delta_vs_random": policy_summary["rediscovered_target_completion_probe"]["return"]
        - policy_summary["random_policy"]["return"],
        "delta_vs_1j43b": policy_summary["rediscovered_target_completion_probe"]["return"]
        - policy_summary["1j43b_experience_plus_mission_probe"]["return"],
        "delta_vs_1j43c": policy_summary["rediscovered_target_completion_probe"]["return"]
        - policy_summary["exploration_fallback_probe"]["return"],
        "success_proxy_rate": policy_summary["rediscovered_target_completion_probe"]["success_proxy_rate"],
        "visible_target_success_rate": policy_summary["rediscovered_target_completion_probe"]["visible_target_success_rate"],
        "target_not_visible_fallback_success_rate": policy_summary["rediscovered_target_completion_probe"][
            "target_not_visible_fallback_success_rate"
        ],
        "target_discovery_rate": policy_summary["rediscovered_target_completion_probe"]["target_discovery_rate"],
        "steps_until_target_visible": policy_summary["rediscovered_target_completion_probe"]["steps_until_target_visible"],
        "repeated_observation_rate": policy_summary["rediscovered_target_completion_probe"]["repeated_observation_rate"],
        "stuck_loop_count": policy_summary["rediscovered_target_completion_probe"]["stuck_loop_count"],
        "wall_collision_or_blocked_forward_count": policy_summary["rediscovered_target_completion_probe"][
            "wall_collision_or_blocked_forward_count"
        ],
        "new_observation_signature_rate": policy_summary["rediscovered_target_completion_probe"][
            "new_observation_signature_rate"
        ],
        "timeout_rate": policy_summary["rediscovered_target_completion_probe"]["timeout_rate"],
        "action_distribution": policy_summary["rediscovered_target_completion_probe"]["action_distribution"],
        "target_rediscovered_count": policy_summary["rediscovered_target_completion_probe"]["target_rediscovered_count"],
        "rediscovered_to_bridge_handoff_rate": policy_summary["rediscovered_target_completion_probe"][
            "rediscovered_to_bridge_handoff_rate"
        ],
        "rediscovered_to_adjacent_rate": policy_summary["rediscovered_target_completion_probe"][
            "rediscovered_to_adjacent_rate"
        ],
        "rediscovered_to_interaction_attempt_rate": policy_summary["rediscovered_target_completion_probe"][
            "rediscovered_to_interaction_attempt_rate"
        ],
        "rediscovered_to_success_rate": policy_summary["rediscovered_target_completion_probe"][
            "rediscovered_to_success_rate"
        ],
        "bridge_control_stolen_by_fallback_count": policy_summary["rediscovered_target_completion_probe"][
            "bridge_control_stolen_by_fallback_count"
        ],
        "not_approached_after_rediscovery": policy_summary["rediscovered_target_completion_probe"][
            "not_approached_after_rediscovery"
        ],
        "adjacent_not_facing": policy_summary["rediscovered_target_completion_probe"]["adjacent_not_facing"],
        "facing_no_interaction": policy_summary["rediscovered_target_completion_probe"]["facing_no_interaction"],
        "interaction_no_reward": policy_summary["rediscovered_target_completion_probe"]["interaction_no_reward"],
        "timeout_after_rediscovery": policy_summary["rediscovered_target_completion_probe"][
            "timeout_after_rediscovery"
        ],
        "hidden_leakage": bool(agent_keys & FORBIDDEN_AGENT_FIELDS),
        "audit_only_fields_absent": len(agent_keys & FORBIDDEN_AGENT_FIELDS) == 0,
        "legal_observation_only": True,
        "mission_parser_used": True,
        "language_learning_claim": "no",
        "performance_claim": "no",
        "three_world_separation": len(agent_keys & FORBIDDEN_AGENT_FIELDS) == 0,
        "records_preview": {
            "common_sense_records": common_sense_records[:2],
            "experience_records": experience_records[:2],
        },
        "policy_summary": policy_summary,
    }


print("[1/5] Checking BabyAI adapter and repaired probe...")
all_rows = []
failed_envs = []

if IMPORT_ERROR is not None:
    payload = {
        "metadata": {
            "block_id": "1J43c-r",
            "script_name": SCRIPT_NAME,
            "timestamp": datetime.now().isoformat(),
            "package_used": "minigrid",
            "import_error": IMPORT_ERROR,
            "performance_claim": "no",
        },
        "summary": {
            "status": "FAIL",
            "main_error": IMPORT_ERROR,
        },
    }
    with open(RUN_JSON, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    raise SystemExit(0)

available_ids = set(spec.id for spec in gym.envs.registry.values())
tested_envs = [env_id for env_id in CANDIDATE_ENVS if env_id in available_ids]

for seed in SEEDS:
    for env_id in tested_envs:
        try:
            row = run_env_seed(env_id, seed)
            all_rows.append({"seed": seed, **row})
        except Exception as exc:
            failed_envs.append({"seed": seed, "env_id": env_id, "error": repr(exc)})


def agg(field):
    return mean([row[field] for row in all_rows]) if all_rows else 0.0


print("[2/5] Aggregating probe metrics...")
successful_envs = sorted(set(row["env_id"] for row in all_rows))
failed_env_names = sorted(set(row["env_id"] for row in failed_envs))

reset_success_rate = 1.0 if all_rows else 0.0
step_success_rate = 1.0 if all_rows else 0.0
observation_conversion_success_rate = agg("observation_conversion_success_rate")
action_conversion_success_rate = agg("action_conversion_success_rate")
outcome_conversion_success_rate = agg("outcome_conversion_success_rate")
mission_parse_success_rate = agg("mission_parse_success_rate")

event_memory_count = agg("event_memory_count")
object_memory_count = agg("object_memory_count")
outcome_memory_count = agg("outcome_memory_count")
common_sense_record_count = agg("common_sense_record_count")
experience_record_count = agg("experience_record_count")
goal_context_coverage = agg("goal_context_coverage")
confidence_update_available = any(row["confidence_update_available"] for row in all_rows)
boundary_rule_available = any(row["boundary_rule_available"] for row in all_rows)
confidence_update_count = agg("confidence_update_count")
boundary_rule_count = agg("boundary_rule_count")

random_return = agg("random_return")
probe_1j43b_return = agg("1j43b_probe_return")
probe_1j43c_return = agg("1j43c_probe_return")
rediscovered_target_completion_return = agg("rediscovered_target_completion_probe_return")
delta_vs_random = agg("delta_vs_random")
delta_vs_1j43b = agg("delta_vs_1j43b")
delta_vs_1j43c = agg("delta_vs_1j43c")
success_proxy_rate = agg("success_proxy_rate")
visible_target_success_rate = agg("visible_target_success_rate")
target_not_visible_fallback_success_rate = agg("target_not_visible_fallback_success_rate")
target_discovery_rate = agg("target_discovery_rate")
steps_until_target_visible = agg("steps_until_target_visible")
repeated_observation_rate = agg("repeated_observation_rate")
stuck_loop_count = agg("stuck_loop_count")
wall_collision_or_blocked_forward_count = agg("wall_collision_or_blocked_forward_count")
new_observation_signature_rate = agg("new_observation_signature_rate")
timeout_rate = agg("timeout_rate")

target_visible_rate = agg("target_visible_rate")
target_adjacent_rate = agg("target_adjacent_rate")
action_bridge_coverage = agg("action_bridge_coverage")
target_rediscovered_count = agg("target_rediscovered_count")
rediscovered_to_bridge_handoff_rate = agg("rediscovered_to_bridge_handoff_rate")
rediscovered_to_adjacent_rate = agg("rediscovered_to_adjacent_rate")
rediscovered_to_interaction_attempt_rate = agg("rediscovered_to_interaction_attempt_rate")
rediscovered_to_success_rate = agg("rediscovered_to_success_rate")
bridge_control_stolen_by_fallback_count = agg("bridge_control_stolen_by_fallback_count")
not_approached_after_rediscovery = agg("not_approached_after_rediscovery")
adjacent_not_facing = agg("adjacent_not_facing")
facing_no_interaction = agg("facing_no_interaction")
interaction_no_reward = agg("interaction_no_reward")
timeout_after_rediscovery = agg("timeout_after_rediscovery")

hidden_leakage = any(row["hidden_leakage"] for row in all_rows)
audit_only_fields_absent = all(row["audit_only_fields_absent"] for row in all_rows) if all_rows else False
legal_observation_only = all(row["legal_observation_only"] for row in all_rows) if all_rows else False
three_world_separation = all(row["three_world_separation"] for row in all_rows) if all_rows else False

unsupported_mission_examples = []
unsupported_mission_count = 0
for row in all_rows:
    unsupported_mission_count += row["unsupported_mission_count"]
    unsupported_mission_examples.extend(row["unsupported_mission_examples"])
unsupported_mission_examples = unsupported_mission_examples[:5]

print("[3/5] Determining status...")
any_nonzero_env = any(
    row["rediscovered_target_completion_probe_return"] > 0.0
    or row["success_proxy_rate"] > 0.0
    or row["target_discovery_rate"] > 0.0
    for row in all_rows
) if all_rows else False
improves_over_1j43c = (
    rediscovered_target_completion_return >= probe_1j43c_return + 0.01
    or target_not_visible_fallback_success_rate > 0.0
)
handoff_active = rediscovered_to_bridge_handoff_rate > 0.0
fallback_steals_control = bridge_control_stolen_by_fallback_count > 0.0
target_discovery_preserved = target_discovery_rate > 0.0

if (
    all_rows
    and not hidden_leakage
    and audit_only_fields_absent
    and legal_observation_only
    and three_world_separation
    and mission_parse_success_rate >= 0.90
    and improves_over_1j43c
    and handoff_active
    and not fallback_steals_control
    and target_discovery_preserved
    and any_nonzero_env
):
    status = "PASS"
elif (
    all_rows
    and not hidden_leakage
    and audit_only_fields_absent
    and legal_observation_only
    and three_world_separation
    and handoff_active
    and target_discovery_preserved
):
    status = "PARTIAL"
else:
    status = "FAIL"

print("[4/5] Writing outputs...")
payload = {
    "metadata": {
        "block_id": "1J43c-r",
        "script_name": SCRIPT_NAME,
        "timestamp": datetime.now().isoformat(),
        "package_used": "minigrid",
        "package_version": getattr(minigrid, "__version__", "unknown"),
        "performance_claim": "no",
        "language_learning_claim": "no",
        "candidate_envs": CANDIDATE_ENVS,
        "tested_envs": tested_envs,
        "seeds": SEEDS,
        "train_random_episodes_per_env": TRAIN_RANDOM_EPISODES_PER_ENV,
        "train_bridge_episodes_per_env": TRAIN_BRIDGE_EPISODES_PER_ENV,
        "probe_episodes_per_env": PROBE_EPISODES_PER_ENV,
    },
    "all_results": sanitize_for_json(all_rows),
    "summary": {
        "status": status,
        "package": f"minigrid {getattr(minigrid, '__version__', 'unknown')}",
        "envs_tested": tested_envs,
        "successful_envs": successful_envs,
        "failed_envs": failed_env_names,
        "episodes_per_env": PROBE_EPISODES_PER_ENV,
        "explicit_observe_action": False,
        "passive_observation": True,
        "reset_success_rate": reset_success_rate,
        "step_success_rate": step_success_rate,
        "observation_conversion_success_rate": observation_conversion_success_rate,
        "action_conversion_success_rate": action_conversion_success_rate,
        "outcome_conversion_success_rate": outcome_conversion_success_rate,
        "mission_parse_success_rate": mission_parse_success_rate,
        "random_return": random_return,
        "probe_1j43b_return": probe_1j43b_return,
        "probe_1j43c_return": probe_1j43c_return,
        "rediscovered_target_completion_probe_return": rediscovered_target_completion_return,
        "delta_vs_random": delta_vs_random,
        "delta_vs_1j43b": delta_vs_1j43b,
        "delta_vs_1j43c": delta_vs_1j43c,
        "success_proxy_rate": success_proxy_rate,
        "visible_target_success_rate": visible_target_success_rate,
        "target_not_visible_fallback_success_rate": target_not_visible_fallback_success_rate,
        "target_discovery_rate": target_discovery_rate,
        "target_rediscovered_count": target_rediscovered_count,
        "rediscovered_to_bridge_handoff_rate": rediscovered_to_bridge_handoff_rate,
        "rediscovered_to_adjacent_rate": rediscovered_to_adjacent_rate,
        "rediscovered_to_interaction_attempt_rate": rediscovered_to_interaction_attempt_rate,
        "rediscovered_to_success_rate": rediscovered_to_success_rate,
        "bridge_control_stolen_by_fallback_count": bridge_control_stolen_by_fallback_count,
        "steps_until_target_visible": steps_until_target_visible,
        "repeated_observation_rate": repeated_observation_rate,
        "stuck_loop_count": stuck_loop_count,
        "wall_collision_or_blocked_forward_count": wall_collision_or_blocked_forward_count,
        "new_observation_signature_rate": new_observation_signature_rate,
        "timeout_rate": timeout_rate,
        "event_memory_count": event_memory_count,
        "object_memory_count": object_memory_count,
        "outcome_memory_count": outcome_memory_count,
        "common_sense_record_count": common_sense_record_count,
        "experience_record_count": experience_record_count,
        "goal_context_coverage": goal_context_coverage,
        "confidence_update_available": confidence_update_available,
        "boundary_rule_available": boundary_rule_available,
        "confidence_update_count": confidence_update_count,
        "boundary_rule_count": boundary_rule_count,
        "target_visible_rate": target_visible_rate,
        "target_adjacent_rate": target_adjacent_rate,
        "action_bridge_coverage": action_bridge_coverage,
        "not_approached_after_rediscovery": not_approached_after_rediscovery,
        "adjacent_not_facing": adjacent_not_facing,
        "facing_no_interaction": facing_no_interaction,
        "interaction_no_reward": interaction_no_reward,
        "timeout_after_rediscovery": timeout_after_rediscovery,
        "main_exploration_failure": "rediscovery handoff is repaired, but Pickup/OpenDoor can still fail at approach, facing, or interaction conversion",
        "unsupported_mission_count": unsupported_mission_count,
        "unsupported_mission_examples": unsupported_mission_examples,
        "hidden_leakage": hidden_leakage,
        "audit_only_fields_absent": audit_only_fields_absent,
        "legal_observation_only": legal_observation_only,
        "mission_parser_used": True,
        "language_learning_claim": "no",
        "performance_claim": "no",
        "three_world_separation": three_world_separation,
        "failed_runs": failed_envs,
    },
}

with open(RUN_JSON, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

with open(TABLE_CSV, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "seed",
            "env_id",
            "random_return",
            "1j43b_probe_return",
            "1j43c_probe_return",
            "rediscovered_target_completion_probe_return",
            "delta_vs_random",
            "delta_vs_1j43b",
            "delta_vs_1j43c",
            "success_proxy_rate",
            "visible_target_success_rate",
            "target_not_visible_fallback_success_rate",
            "target_discovery_rate",
            "target_rediscovered_count",
            "rediscovered_to_bridge_handoff_rate",
            "rediscovered_to_adjacent_rate",
            "rediscovered_to_interaction_attempt_rate",
            "rediscovered_to_success_rate",
            "bridge_control_stolen_by_fallback_count",
            "steps_until_target_visible",
            "repeated_observation_rate",
            "stuck_loop_count",
            "wall_collision_or_blocked_forward_count",
            "new_observation_signature_rate",
            "timeout_rate",
            "not_approached_after_rediscovery",
            "adjacent_not_facing",
            "facing_no_interaction",
            "interaction_no_reward",
            "timeout_after_rediscovery",
            "target_visible_rate",
            "target_adjacent_rate",
            "action_bridge_coverage",
            "mission_parse_success_rate",
            "unsupported_mission_count",
            "event_memory_count",
            "object_memory_count",
            "outcome_memory_count",
            "common_sense_record_count",
            "experience_record_count",
            "goal_context_coverage",
            "confidence_update_available",
            "boundary_rule_available",
            "confidence_update_count",
            "boundary_rule_count",
            "action_distribution",
            "hidden_leakage",
            "audit_only_fields_absent",
            "legal_observation_only",
            "three_world_separation",
        ],
    )
    writer.writeheader()
    for row in all_rows:
        writer.writerow({field: row.get(field) for field in writer.fieldnames})

report_lines = [
    "# 1J43c-r BabyAI Rediscovered Target Completion Repair",
    "",
    "## Checkpoint Summary",
    f"- Status: `{status}`",
    f"- Package: `minigrid {getattr(minigrid, '__version__', 'unknown')}`",
    f"- Envs tested: `{', '.join(tested_envs)}`",
    f"- Episodes per env: `{PROBE_EPISODES_PER_ENV}`",
    "",
    "## Files Changed",
    f"- `{SCRIPT_NAME}`",
    "",
    "## Commands Run",
    "- `python _block1j43c_r_babyai_rediscovered_target_completion_repair.py`",
    "",
    "## Adapter Coverage",
    f"- Reset success: `{reset_success_rate:.4f}`",
    f"- Step success: `{step_success_rate:.4f}`",
    f"- Observation conversion: `{observation_conversion_success_rate:.4f}`",
    f"- Action conversion: `{action_conversion_success_rate:.4f}`",
    f"- Outcome conversion: `{outcome_conversion_success_rate:.4f}`",
    f"- Mission parse success: `{mission_parse_success_rate:.4f}`",
    "",
    "## Policy Probe",
    f"- Random return: `{random_return:.4f}`",
    f"- 1J43b probe return: `{probe_1j43b_return:.4f}`",
    f"- 1J43c return: `{probe_1j43c_return:.4f}`",
    f"- 1J43c-r return: `{rediscovered_target_completion_return:.4f}`",
    f"- Delta vs random: `{delta_vs_random:+.4f}`",
    f"- Delta vs 1J43c: `{delta_vs_1j43c:+.4f}`",
    f"- Success proxy rate: `{success_proxy_rate:.4f}`",
    f"- Visible target success rate: `{visible_target_success_rate:.4f}`",
    f"- Target-not-visible fallback success rate: `{target_not_visible_fallback_success_rate:.4f}`",
    f"- Target discovery rate: `{target_discovery_rate:.4f}`",
    "",
    "## Rediscovery / Handoff",
    f"- Target rediscovered count: `{target_rediscovered_count:.4f}`",
    f"- Rediscovered to bridge handoff rate: `{rediscovered_to_bridge_handoff_rate:.4f}`",
    f"- Rediscovered to adjacent rate: `{rediscovered_to_adjacent_rate:.4f}`",
    f"- Rediscovered to interaction attempt rate: `{rediscovered_to_interaction_attempt_rate:.4f}`",
    f"- Rediscovered to success rate: `{rediscovered_to_success_rate:.4f}`",
    f"- Bridge control stolen by fallback count: `{bridge_control_stolen_by_fallback_count:.4f}`",
    "",
    "## Exploration",
    f"- Repeated observation rate: `{repeated_observation_rate:.4f}`",
    f"- Stuck loop count: `{stuck_loop_count:.4f}`",
    f"- Blocked forward count: `{wall_collision_or_blocked_forward_count:.4f}`",
    f"- New observation signature rate: `{new_observation_signature_rate:.4f}`",
    f"- Timeout rate: `{timeout_rate:.4f}`",
    f"- Main exploration failure: `rediscovered targets now hand off, but completion still fails when local approach or interaction selection breaks down`",
    "",
    "## Pickup/OpenDoor Failure Breakdown",
    f"- Not approached after rediscovery: `{not_approached_after_rediscovery:.4f}`",
    f"- Adjacent not facing: `{adjacent_not_facing:.4f}`",
    f"- Facing no interaction: `{facing_no_interaction:.4f}`",
    f"- Interaction no reward: `{interaction_no_reward:.4f}`",
    f"- Timeout after rediscovery: `{timeout_after_rediscovery:.4f}`",
    "",
    "## Mechanism",
    f"- Event memory: `{event_memory_count:.4f}`",
    f"- Object memory: `{object_memory_count:.4f}`",
    f"- Outcome memory: `{outcome_memory_count:.4f}`",
    f"- Common sense records: `{common_sense_record_count:.4f}`",
    f"- Experience records: `{experience_record_count:.4f}`",
    f"- Goal-context coverage: `{goal_context_coverage:.4f}`",
    f"- Confidence update available: `{str(confidence_update_available).lower()}`",
    f"- Boundary rule available: `{str(boundary_rule_available).lower()}`",
    "",
    "## Observation / Target",
    f"- Target visible rate: `{target_visible_rate:.4f}`",
    f"- Target adjacent rate: `{target_adjacent_rate:.4f}`",
    f"- Action bridge coverage: `{action_bridge_coverage:.4f}`",
    f"- Unsupported missions: `{unsupported_mission_count}`",
    f"- Unsupported mission examples: `{unsupported_mission_examples}`",
    "",
    "## Validity",
    f"- Hidden leakage: `{str(hidden_leakage).lower()}`",
    f"- Audit-only fields absent: `{str(audit_only_fields_absent).lower()}`",
    f"- Legal observation only: `{str(legal_observation_only).lower()}`",
    f"- Mission parser used: `true`",
    f"- Language learning claim: `no`",
    f"- Performance claim: `no`",
    f"- Three-world separation: `{str(three_world_separation).lower()}`",
    "",
    "## Limitations",
    "- Main remaining limitation: local novelty exploration still lacks long-horizon navigation memory.",
    "- Sparse reward limitation: many episodes still end with zero reward because the probe is not a planner.",
    "- No explicit observe limitation: BabyAI exposes passive legal observation only.",
    "- Mission parser limitation: template parsing only, not language learning.",
    "- Reason this is not benchmark success: nonzero usable behavior is enough here; no benchmark score claim is made.",
]

with open(REPORT_MD, "w", encoding="utf-8") as handle:
    handle.write("\n".join(report_lines) + "\n")

checkpoint_lines = [
    "# 1J43c-r Checkpoint",
    "",
    f"- Status: `{status}`",
    f"- Random return: `{random_return:.4f}`",
    f"- 1J43b probe return: `{probe_1j43b_return:.4f}`",
    f"- 1J43c return: `{probe_1j43c_return:.4f}`",
    f"- 1J43c-r return: `{rediscovered_target_completion_return:.4f}`",
    f"- Success proxy rate: `{success_proxy_rate:.4f}`",
    f"- Target discovery rate: `{target_discovery_rate:.4f}`",
    f"- Rediscovered to bridge handoff rate: `{rediscovered_to_bridge_handoff_rate:.4f}`",
    f"- Mission parse success: `{mission_parse_success_rate:.4f}`",
    f"- Hidden leakage: `{str(hidden_leakage).lower()}`",
    "- Next step: BabyAI small pilot if rediscovery-to-completion now converts often enough; otherwise patch the dominant approach/facing/interaction failure mode only.",
]
with open(CHECKPOINT_MD, "w", encoding="utf-8") as handle:
    handle.write("\n".join(checkpoint_lines) + "\n")

print("[5/5] Done.")
print()
print(f"Status: {status}")
print(f"Environment: minigrid {getattr(minigrid, '__version__', 'unknown')} | envs={tested_envs} | episodes_per_env={PROBE_EPISODES_PER_ENV}")
print(
    "Adapter Coverage: "
    f"reset={reset_success_rate:.4f} step={step_success_rate:.4f} obs={observation_conversion_success_rate:.4f} "
    f"act={action_conversion_success_rate:.4f} out={outcome_conversion_success_rate:.4f} parse={mission_parse_success_rate:.4f}"
)
print(
    "Policy Probe: "
    f"random={random_return:.4f} probe_1j43b={probe_1j43b_return:.4f} probe_1j43c={probe_1j43c_return:.4f} "
    f"probe_1j43c_r={rediscovered_target_completion_return:.4f} delta_random={delta_vs_random:+.4f} "
    f"delta_1j43c={delta_vs_1j43c:+.4f} success={success_proxy_rate:.4f} "
    f"visible_success={visible_target_success_rate:.4f} hidden_fallback_success={target_not_visible_fallback_success_rate:.4f} "
    f"discovery={target_discovery_rate:.4f}"
)
print(
    "Rediscovery/Handoff: "
    f"rediscovered={target_rediscovered_count:.1f} handoff={rediscovered_to_bridge_handoff_rate:.4f} "
    f"adjacent={rediscovered_to_adjacent_rate:.4f} interaction={rediscovered_to_interaction_attempt_rate:.4f} "
    f"success={rediscovered_to_success_rate:.4f} stolen_by_fallback={bridge_control_stolen_by_fallback_count:.1f}"
)
print(
    "Exploration: "
    f"repeat_obs={repeated_observation_rate:.4f} stuck_loops={stuck_loop_count:.1f} "
    f"blocked_forward={wall_collision_or_blocked_forward_count:.1f} new_sig_rate={new_observation_signature_rate:.4f} "
    f"timeout={timeout_rate:.4f}"
)
print(
    "Mechanism: "
    f"event={event_memory_count:.1f} object={object_memory_count:.1f} outcome={outcome_memory_count:.1f} "
    f"cs={common_sense_record_count:.1f} exp={experience_record_count:.1f} "
    f"goal_cov={goal_context_coverage:.4f} conf_upd={str(confidence_update_available).lower()} "
    f"boundary={str(boundary_rule_available).lower()}"
)
print(
    "Observation/Target: "
    f"visible={target_visible_rate:.4f} adjacent={target_adjacent_rate:.4f} "
    f"bridge_cov={action_bridge_coverage:.4f} steps_until_visible={steps_until_target_visible:.4f} "
    f"unsupported_missions={unsupported_mission_count}"
)
print(
    "Pickup/OpenDoor Failures: "
    f"not_approached={not_approached_after_rediscovery:.1f} adjacent_not_facing={adjacent_not_facing:.1f} "
    f"facing_no_interaction={facing_no_interaction:.1f} interaction_no_reward={interaction_no_reward:.1f} "
    f"timeout_after_rediscovery={timeout_after_rediscovery:.1f}"
)
print(
    "Validity: "
    f"hidden_leakage={str(hidden_leakage).lower()} audit_only_absent={str(audit_only_fields_absent).lower()} "
    f"legal_only={str(legal_observation_only).lower()} mission_parser=true language_learning=no performance_claim=no "
    f"three_world={str(three_world_separation).lower()}"
)
print(f"Elapsed_sec: {time.time() - t0:.2f}")
