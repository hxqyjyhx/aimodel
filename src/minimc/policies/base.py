"""Base policy class and shared utilities for Mini-MC policies."""

import math
from abc import ABC, abstractmethod

# Reuse constants from InstanceOutcomeMemory
MAIN_CANDIDATE_ACTIONS = [
    "mine_by_hand", "mine_with_pickaxe", "craft_plank",
    "eat", "use_as_tool", "burn_as_fuel",
]
ACTION_TO_FEATURE = {
    "mine_by_hand": "mine_by_hand_success",
    "mine_with_pickaxe": "mine_with_pickaxe_success",
    "craft_plank": "craft_plank_success",
    "eat": "eat_success",
    "use_as_tool": "use_as_tool_success",
    "burn_as_fuel": "burn_as_fuel_success",
}
CORE_ACTION_FEATURES = list(ACTION_TO_FEATURE.values())


def compute_utility(probs):
    """Utility = mean over 6 core features of max(p, 1-p)."""
    return sum(max(probs.get(f, 0.5), 1.0 - probs.get(f, 0.5))
               for f in CORE_ACTION_FEATURES) / len(CORE_ACTION_FEATURES)


def compute_entropy(probs):
    """Mean binary entropy over features."""
    total = 0.0
    for f in CORE_ACTION_FEATURES:
        p = max(1e-9, min(1.0 - 1e-9, probs.get(f, 0.5)))
        total += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return total / len(CORE_ACTION_FEATURES)


class MiniMCPolicy(ABC):
    @abstractmethod
    def reset(self, view):
        pass

    @abstractmethod
    def select_next_object(self, view):
        pass

    @abstractmethod
    def decide_probe(self, view, object_id):
        pass

    @abstractmethod
    def on_probe_result(self, view, object_id, action, outcome):
        pass

    @abstractmethod
    def get_answer(self, view):
        pass

    def get_pre_probe_entropies(self):
        """Return {oid: pre-probe entropy} for entropy gap analysis."""
        return {}

    def get_pre_decision_entropies(self):
        """Return {oid: pre-decision entropy} for objects considered but skipped."""
        return {}
