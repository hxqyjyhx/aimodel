"""Oracle policies: C14a (truth answer) and C14b (budgeted oracle probe)."""

from .base import MiniMCPolicy, MAIN_CANDIDATE_ACTIONS, CORE_ACTION_FEATURES, compute_entropy


# =========================================================================
# C14b — Budgeted oracle probe
# =========================================================================
class C14b_BudgetedOraclePolicy(MiniMCPolicy):
    """Budgeted oracle: uses hidden truth to select best probe action,
    but respects the budget constraint. Runs in separate harness."""

    def __init__(self, instance_memory, rng, simulator):
        self._im = instance_memory
        self._rng = rng
        self._simulator = simulator
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def reset(self, view):
        self._pre_probe_entropies = {}
        self._pre_decision_entropies = {}

    def select_next_object(self, view):
        unvisited = view.get_unvisited_objects()
        if not unvisited:
            for oid in unvisited:
                fake_obj = {"id": oid, "visible_features": {}}
                probs = self._im.predict_all_affordances(fake_obj)
                self._pre_decision_entropies[oid] = compute_entropy(probs)
            return None
        # Visit nearest affordable
        best_oid = None
        best_cost = float('inf')
        for oid in unvisited:
            total_cost = view.compute_reach_cost(oid) + view.observe_cost + view.probe_cost
            if view.can_afford(total_cost) and total_cost < best_cost:
                best_cost = total_cost
                best_oid = oid
        return best_oid

    def decide_probe(self, view, object_id):
        features = view.get_observed_features(object_id)
        if features is None:
            return False, None
        fake_obj = {"id": object_id, "visible_features": features}
        probs = self._im.predict_all_affordances(fake_obj)
        self._pre_probe_entropies[object_id] = compute_entropy(probs)

        best_action = None
        best_post_acc = -1.0
        for action in MAIN_CANDIDATE_ACTIONS:
            true_outcome = self._simulator.probe(object_id, action)[0]
            im_clone = self._im.clone()
            im_clone.incorporate_probe(object_id, action, true_outcome)
            post_probs = im_clone.predict_all_affordances(fake_obj)
            gt = self._simulator.get_ground_truth_affordances(object_id)
            acc = sum(
                1.0 if (post_probs.get(f, 0.5) >= 0.5) == (gt.get(f, 0.0) >= 0.5)
                else 0.0
                for f in CORE_ACTION_FEATURES
            ) / len(CORE_ACTION_FEATURES)
            if acc > best_post_acc:
                best_post_acc = acc
                best_action = action

        return True, best_action

    def on_probe_result(self, view, object_id, action, outcome):
        self._im.incorporate_probe(object_id, action, outcome)

    def get_answer(self, view):
        per_object = {}
        for oid in view.get_all_object_ids():
            features = view.get_observed_features(oid) or {}
            fake_obj = {"id": oid, "visible_features": features}
            per_object[oid] = self._im.predict_all_affordances(fake_obj)
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return dict(self._pre_probe_entropies)

    def get_pre_decision_entropies(self):
        return dict(self._pre_decision_entropies)


# =========================================================================
# C14a — Truth answer oracle (no budget, no harness)
# =========================================================================
class C14a_TruthAnswerOracle:
    """Truth answer oracle. Returns ground truth directly.
    Does NOT use the harness — used only to verify evaluator ceiling.
    Should score ~1.0 on composite_task_accuracy."""

    def __init__(self, simulator):
        self._simulator = simulator

    def get_answer(self):
        """Return ground truth affordances as predictions."""
        per_object = {}
        for oid in self._simulator._objects:
            gt = self._simulator.get_ground_truth_affordances(oid)
            per_object[oid] = {f: gt.get(f, 0.0) for f in CORE_ACTION_FEATURES}
        return {"per_object": per_object}

    def get_pre_probe_entropies(self):
        return {}

    def get_pre_decision_entropies(self):
        return {}
