"""External dependency stubs.

These modules were originally imported from sibling experiment directories
(exp004_5l, exp004_5a3, exp004_5k, exp004_5i) via sys.path hacks.

The real implementations need to be provided before run.py can execute.
"""

from .instance_outcome_memory import InstanceOutcomeMemory
from .category_posterior import CategoryPosteriorLearner
from .runner_helpers import run_phase_a_training, run_cluster_baselines, collect_sparse_probe_outcomes

__all__ = [
    "InstanceOutcomeMemory",
    "CategoryPosteriorLearner",
    "run_phase_a_training",
    "run_cluster_baselines",
    "collect_sparse_probe_outcomes",
]
