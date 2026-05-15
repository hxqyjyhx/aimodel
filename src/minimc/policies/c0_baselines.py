"""C0 baseline policies: no interaction and observe-only."""

from .base import MiniMCPolicy, compute_entropy


# =========================================================================
# C0a — No interaction baseline
# =========================================================================
class C0a_NoInteractionPolicy(MiniMCPolicy):
    """Never observes, never probes, never moves. Uses global base rates only."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng

    def reset(self, view):
        pass

    def select_next_object(self, view):
        return None

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            fake_obj = {"id": oid, "visible_features": {}}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}


# =========================================================================
# C0b — Observe only, no probe
# =========================================================================
class C0b_ObserveOnlyPolicy(MiniMCPolicy):
    """Visits objects (nearest first), observes, never probes."""

    def __init__(self, instance_memory, rng):
        self._im = instance_memory
        self._rng = rng
        self._pre_decision_entropies = {}

    def reset(self, view):
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            return None
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            reach_cost = view.compute_reach_cost(oid)
            total = reach_cost + view.observe_cost
            if view.can_afford(total) and total < best_cost:
                best_cost = total
                best_oid = oid
        # Record pre-decision entropies for objects we're skipping
        for oid in unvisited:
            if oid != best_oid:
                fake_obj = {"id": oid, "visible_features": {}}
                probs = self._im.predict_all_affordances(fake_obj)
                self._pre_decision_entropies[oid] = compute_entropy(probs)
        return best_oid

    def decide_probe(self, view, object_id):
        return False, None

    def on_probe_result(self, view, object_id, action, outcome):
        pass

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)
