"""Layer 2: AgentObsView (read-only policy interface) and AgentObs (mutable)."""

import config
from simulator_truth import MiniMCSimulatorTruth


class BudgetExceededError(Exception):
    pass


class AgentObsView:
    """Read-only snapshot. Policy sees ONLY this. No mutator methods."""

    def __init__(self, simulator, initial_budget, reach_cost_per_unit,
                 observe_cost, probe_cost):
        self._simulator = simulator
        self._initial_budget = initial_budget
        self._budget_remaining = initial_budget
        self._reach_cost_per_unit = reach_cost_per_unit
        self._observe_cost = observe_cost
        self._probe_cost = probe_cost

        # Mutable state (private, mutated by AgentObs subclass)
        self._agent_position = simulator.get_agent_start()
        self._observed_features = {}       # oid -> dict[str, bool]
        self._probe_results = {}           # oid -> dict[str, float]
        self._probe_history = {}           # oid -> list[dict]
        self._visited = {}                 # oid -> bool
        self._budget_spent = 0.0
        self._log = None                   # set by AgentObs

    # ---- Read-only properties ----
    @property
    def agent_position(self):
        return self._agent_position

    @property
    def budget_remaining(self):
        return self._budget_remaining

    @property
    def budget_spent(self):
        return self._budget_spent

    @property
    def initial_budget(self):
        return self._initial_budget

    @property
    def observe_cost(self):
        return self._observe_cost

    @property
    def probe_cost(self):
        return self._probe_cost

    # ---- Read-only methods ----
    def get_all_object_ids(self):
        return self._simulator.get_all_object_ids()

    def get_unvisited_objects(self):
        all_oids = self.get_all_object_ids()
        return [oid for oid in all_oids if not self._visited.get(oid, False)]

    def get_object_position(self, object_id):
        return self._simulator.get_position(object_id)

    def get_observed_features(self, object_id):
        return self._observed_features.get(object_id, None)

    def get_probe_results(self, object_id):
        return self._probe_results.get(object_id, {})

    def get_probe_history(self, object_id):
        return list(self._probe_history.get(object_id, []))

    def is_visited(self, object_id):
        return self._visited.get(object_id, False)

    def compute_reach_cost(self, object_id):
        obj_pos = self._simulator.get_position(object_id)
        dist = MiniMCSimulatorTruth.manhattan_distance(self._agent_position, obj_pos)
        return dist * self._reach_cost_per_unit

    def can_afford(self, cost):
        return self._budget_remaining >= cost - 1e-12

    def is_terminal(self):
        unvisited = self.get_unvisited_objects()
        if not unvisited:
            return True
        cheapest_cost = min(
            self.compute_reach_cost(oid) + self._observe_cost for oid in unvisited
        )
        return not self.can_afford(cheapest_cost)


class AgentObs(AgentObsView):
    """Mutable state. Owned by MiniMCEnvironment/harness. NEVER passed to policy."""

    def __init__(self, simulator, initial_budget, reach_cost_per_unit,
                 observe_cost, probe_cost, event_log):
        super().__init__(simulator, initial_budget, reach_cost_per_unit,
                         observe_cost, probe_cost)
        self._log = event_log

    @property
    def event_log(self):
        return self._log

    def _deduct_budget(self, amount, reason):
        if amount > self._budget_remaining + 1e-12:
            raise BudgetExceededError(
                f"Budget exceeded: need {amount:.4f} for {reason}, "
                f"{self._budget_remaining:.4f} remaining"
            )
        self._budget_remaining -= amount
        self._budget_spent += amount

    def _reach(self, object_id):
        old_pos = self._agent_position
        new_pos = self._simulator.get_position(object_id)
        reach_cost = self.compute_reach_cost(object_id)
        self._deduct_budget(reach_cost, f"reach {object_id}")
        self._agent_position = new_pos
        self._log.append({
            "event": "reach",
            "object_id": object_id,
            "from_position": list(old_pos),
            "to_position": list(new_pos),
            "cost": reach_cost,
            "budget_before": self._budget_remaining + reach_cost,
            "budget_after": self._budget_remaining,
        })

    def _observe(self, object_id):
        cost = self._observe_cost
        self._deduct_budget(cost, f"observe {object_id}")
        features = self._simulator.observe(object_id)
        self._observed_features[object_id] = features
        self._visited[object_id] = True
        self._log.append({
            "event": "observe",
            "object_id": object_id,
            "visible_features": features,
            "cost": cost,
            "budget_before": self._budget_remaining + cost,
            "budget_after": self._budget_remaining,
        })

    def _probe(self, object_id, action):
        features = self._observed_features.get(object_id)
        cost = self._probe_cost
        self._deduct_budget(cost, f"probe {object_id} {action}")
        outcome_float, outcome_str = self._simulator.probe(object_id, action)
        self._probe_results.setdefault(object_id, {})[action] = outcome_float
        self._probe_history.setdefault(object_id, []).append({
            "action": action, "outcome": outcome_float, "outcome_str": outcome_str
        })
        self._log.append({
            "event": "probe",
            "object_id": object_id,
            "action": action,
            "outcome": outcome_float,
            "outcome_str": outcome_str,
            "cost": cost,
            "budget_before": self._budget_remaining + cost,
            "budget_after": self._budget_remaining,
        })
        return outcome_float, outcome_str
