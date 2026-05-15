"""Stub: CategoryPosteriorLearner — originally from exp004_5k_episodic_sparse_outcome.

Provide the real implementation or install the package to use.
"""


class CategoryPosteriorLearner:
    """Category-level posterior learner for cluster-based policies.

    Methods used by C2 policy:
        compute_posterior(obj) -> dict
        compute_max_eig(label_probs) -> float
        predict_affordances(label_probs) -> dict[str, float]
        select_best_probe(label_probs, actions) -> dict
        incorporate_probe(oid, action, outcome)
        clone() -> CategoryPosteriorLearner
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CategoryPosteriorLearner is an external dependency from "
            "exp004_5k_episodic_sparse_outcome. "
            "Provide the real implementation before running."
        )

    def compute_posterior(self, obj):
        raise NotImplementedError

    def compute_max_eig(self, label_probs):
        raise NotImplementedError

    def predict_affordances(self, label_probs):
        raise NotImplementedError

    def select_best_probe(self, label_probs, candidate_actions):
        raise NotImplementedError

    def incorporate_probe(self, object_id, action, outcome):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError
