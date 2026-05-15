"""Layer 1: MiniMCSimulatorTruth — owns all hidden ground truth.

Policies NEVER access this class. Only MiniMCEnvironment and MiniMCEvaluator do.
"""


class MiniMCSimulatorTruth:
    def __init__(self, objects, positions, agent_start):
        self._objects = objects
        self._positions = positions
        self._agent_start = agent_start

    # ---- Position API ----
    def get_position(self, object_id):
        return self._positions[object_id]

    def get_agent_start(self):
        return self._agent_start

    @staticmethod
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # ---- Observation (deterministic, cached in objects) ----
    def observe(self, object_id):
        return dict(self._objects[object_id]["visible_features"])

    # ---- Probe execution (ground truth) ----
    def probe(self, object_id, action):
        profile = self._objects[object_id]["hidden_affordance_profile"]
        raw = profile.get(action, "fail")
        if action == "tap_sound":
            outcome_float = 1.0 if raw == "clear" else 0.0
        else:
            outcome_float = 1.0 if raw == "success" else 0.0
        return outcome_float, str(raw)

    # ---- Oracle access (evaluator and C14 only) ----
    def get_hidden_category(self, object_id):
        return self._objects[object_id]["hidden_category"]

    def get_ground_truth_affordances(self, object_id):
        profile = self._objects[object_id]["hidden_affordance_profile"]
        result = {}
        for action, raw in profile.items():
            if action == "tap_sound":
                result["tap_sound_clear"] = 1.0 if raw == "clear" else 0.0
            elif action in _CORE_PROBE_ACTIONS:
                feat = _ACTION_TO_FEATURE.get(action, action)
                result[feat] = 1.0 if raw == "success" else 0.0
        return result

    # ---- Utility ----
    def get_all_object_ids(self):
        return sorted(self._objects.keys())

    def get_object_by_id(self, object_id):
        return self._objects[object_id]

    # ---- Position assignment ----
    @staticmethod
    def assign_positions(object_ids, grid_rows, grid_cols, agent_start, rng):
        all_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
        if agent_start in all_cells:
            all_cells.remove(agent_start)
        rng.shuffle(all_cells)
        positions = {}
        for oid, cell in zip(sorted(object_ids), all_cells):
            positions[oid] = cell
        return positions


# Local copies to avoid circular imports
_CORE_PROBE_ACTIONS = [
    "mine_by_hand", "mine_with_pickaxe", "craft_plank",
    "eat", "use_as_tool", "burn_as_fuel",
]
_ACTION_TO_FEATURE = {
    "mine_by_hand": "mine_by_hand_success",
    "mine_with_pickaxe": "mine_with_pickaxe_success",
    "craft_plank": "craft_plank_success",
    "eat": "eat_success",
    "use_as_tool": "use_as_tool_success",
    "burn_as_fuel": "burn_as_fuel_success",
}
