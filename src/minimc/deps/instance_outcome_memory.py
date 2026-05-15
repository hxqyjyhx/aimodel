"""Stub: InstanceOutcomeMemory — originally from exp004_5l_instance_outcome_memory.

Provide the real implementation or install the package to use.
"""


class InstanceOutcomeMemory:
    """Instance-based memory for outcome prediction.

    Methods used by policies:
        predict_all_affordances(obj) -> dict[str, float]
        clone() -> InstanceOutcomeMemory
        incorporate_probe(oid, action, outcome)
        compute_action_outcome_eig(obj, action) -> (float, ...)
        select_best_probe(obj, actions) -> dict
        build(outcome_rows)
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InstanceOutcomeMemory is an external dependency from "
            "exp004_5l_instance_outcome_memory. "
            "Provide the real implementation before running."
        )

    def build(self, outcome_rows):
        raise NotImplementedError

    def predict_all_affordances(self, obj):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def incorporate_probe(self, object_id, action, outcome):
        raise NotImplementedError

    def compute_action_outcome_eig(self, obj, action):
        raise NotImplementedError

    def select_best_probe(self, obj, candidate_actions):
        raise NotImplementedError
