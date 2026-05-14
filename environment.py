"""MiniMCEnvironment — wraps SimulatorTruth + AgentObs, harness-facing API."""

from simulator_truth import MiniMCSimulatorTruth
from agent_obs import AgentObs
from event_log import EventLog
import config


class MiniMCEnvironment:
    def __init__(self, objects, positions, agent_start=None,
                 initial_budget=None, reach_cost_per_unit=None,
                 observe_cost=None, probe_cost=None):
        if agent_start is None:
            agent_start = config.AGENT_START
        if initial_budget is None:
            initial_budget = config.DEFAULT_INITIAL_BUDGET
        if reach_cost_per_unit is None:
            reach_cost_per_unit = config.REACH_COST_PER_UNIT
        if observe_cost is None:
            observe_cost = config.OBSERVE_COST
        if probe_cost is None:
            probe_cost = config.PROBE_COST

        self.simulator = MiniMCSimulatorTruth(objects, positions, agent_start)
        self._initial_budget = initial_budget
        self._reach_cost_per_unit = reach_cost_per_unit
        self._observe_cost = observe_cost
        self._probe_cost = probe_cost
        self._event_log = None

    def reset(self):
        self._event_log = EventLog()
        obs = AgentObs(
            self.simulator, self._initial_budget,
            self._reach_cost_per_unit, self._observe_cost, self._probe_cost,
            self._event_log,
        )
        obs.event_log.append({
            "event": "episode_start",
            "initial_budget": self._initial_budget,
            "agent_start": list(self.simulator.get_agent_start()),
        })
        return obs

    def reach(self, agent_obs, object_id):
        agent_obs._reach(object_id)

    def observe(self, agent_obs, object_id):
        agent_obs._observe(object_id)

    def probe(self, agent_obs, object_id, action):
        return agent_obs._probe(object_id, action)

    # Evaluation helpers (use simulator privilege)
    def get_valid_task_queries(self):
        return list(_TASK_QUERIES.keys())

    def get_task_query_ground_truth(self, query_name, object_id):
        gt = self.simulator.get_ground_truth_affordances(object_id)
        return _TASK_QUERIES[query_name](gt)

    def get_ground_truth_affordances(self, object_id):
        return self.simulator.get_ground_truth_affordances(object_id)

    def get_hidden_category(self, object_id):
        return self.simulator.get_hidden_category(object_id)


# Task queries: need_planks, need_stone, need_food, need_tool, need_fuel
_TASK_QUERIES = {
    "need_planks": lambda gt: gt.get("craft_plank_success", 0.0) == 1.0,
    "need_stone": lambda gt: (
        gt.get("mine_by_hand_success", 1.0) == 0.0
        and gt.get("mine_with_pickaxe_success", 0.0) == 1.0
    ),
    "need_food": lambda gt: gt.get("eat_success", 0.0) == 1.0,
    "need_tool": lambda gt: gt.get("use_as_tool_success", 0.0) == 1.0,
    "need_fuel": lambda gt: gt.get("burn_as_fuel_success", 0.0) == 1.0,
}
