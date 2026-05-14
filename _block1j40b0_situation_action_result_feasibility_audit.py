"""
Block 1J40b-0 -- Situation-Action-Result Learning Feasibility Audit.

Audit whether C5 logs can support Strategy Learning from SAR records.
Shadow-only: does NOT train or activate a policy.
Uses only agent-visible evidence + 1J40a-2 soft explanation scores.
"""
import os, sys, json, copy, random, time
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from run_004_5n1 import run_phase_a_training
from subtype_objects import (
    SUBTYPE_DEFINITIONS, _generate_visible_features_subtype,
    _make_hidden_profile_subtype, _generate_function_feature_values,
)
from objects import ACTIVE_FEATURES_PHASE_A, PREDICTOR_AUGMENTED_FEATURES

# =============================================================================
# CLI
# =============================================================================
import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--multiseed", action="store_true", default=True)
_args = _parser.parse_args()
MULTISEED_RUN = _args.multiseed
MULTISEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5

# =============================================================================
# Constants (shared with 1J40a-2)
# =============================================================================
ALL_ACTIONS = sorted([
    "craft_plank", "eat", "use_as_tool", "burn_as_fuel",
    "mine_by_hand", "mine_with_pickaxe",
])
STATE_FEATURE_KEYS = ["fresh", "wet", "damaged", "clean", "hot", "open"]

FEATURE_UNIVERSE = sorted(set(ACTIVE_FEATURES_PHASE_A) | set(PREDICTOR_AUGMENTED_FEATURES) |
    {"damp_texture", "brittle_surface", "treated_surface", "hollow_sound"})
FEATURE_UNIVERSE_SET = set(FEATURE_UNIVERSE)

TYPE_FAMILIES = {
    "wood-like": {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"},
    "stone-like": {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"},
    "apple-like": {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"},
    "tool-like": {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"},
}
CATEGORY_TO_FAMILY = {
    "wood_log": "wood-like", "stone_block": "stone-like",
    "apple": "apple-like", "wooden_pickaxe": "tool-like",
}

MIXED_SOURCE_ALLOCATION = {
    "wood_log":         {"observation_gap": 4, "state_condition": 3, "identity_split": 4, "irreducible_noise": 4},
    "stone_block":      {"observation_gap": 4, "state_condition": 4, "identity_split": 3, "irreducible_noise": 4},
    "apple":            {"observation_gap": 4, "state_condition": 4, "identity_split": 4, "irreducible_noise": 3},
    "wooden_pickaxe":   {"observation_gap": 3, "state_condition": 4, "identity_split": 4, "irreducible_noise": 4},
}

STATE_CONDITION_PROFILES = {
    "wood_log": {
        "states": ["fresh", "rotten"],
        "affordance_by_state": {
            "fresh": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "success",
                      "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "success"},
            "rotten": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                       "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "success"},
        },
        "state_features": {
            "fresh": {"fresh": True, "wet": True, "damaged": False, "clean": True, "hot": False, "open": False},
            "rotten": {"fresh": False, "wet": False, "damaged": True, "clean": False, "hot": False, "open": False},
        },
    },
    "stone_block": {
        "states": ["intact_state", "cracked"],
        "affordance_by_state": {
            "intact_state": {"mine_by_hand": "fail", "mine_with_pickaxe": "success", "craft_plank": "fail",
                             "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "fail"},
            "cracked": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                        "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "fail"},
        },
        "state_features": {
            "intact_state": {"fresh": False, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
            "cracked": {"fresh": False, "wet": False, "damaged": True, "clean": False, "hot": False, "open": True},
        },
    },
    "apple": {
        "states": ["fresh_state", "spoiled"],
        "affordance_by_state": {
            "fresh_state": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                            "eat": "success", "use_as_tool": "fail", "burn_as_fuel": "fail"},
            "spoiled": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                        "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "fail"},
        },
        "state_features": {
            "fresh_state": {"fresh": True, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
            "spoiled": {"fresh": False, "wet": True, "damaged": True, "clean": False, "hot": False, "open": False},
        },
    },
    "wooden_pickaxe": {
        "states": ["intact_state", "worn"],
        "affordance_by_state": {
            "intact_state": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                             "eat": "fail", "use_as_tool": "success", "burn_as_fuel": "success"},
            "worn": {"mine_by_hand": "success", "mine_with_pickaxe": "success", "craft_plank": "fail",
                     "eat": "fail", "use_as_tool": "fail", "burn_as_fuel": "success"},
        },
        "state_features": {
            "intact_state": {"fresh": False, "wet": False, "damaged": False, "clean": True, "hot": False, "open": False},
            "worn": {"fresh": False, "wet": False, "damaged": True, "clean": False, "hot": False, "open": False},
        },
    },
}

IRREDUCIBLE_NOISE_BASE = {
    "wood_log": {"mine_by_hand": 0.55, "mine_with_pickaxe": 0.55, "craft_plank": 0.45,
                 "eat": 0.45, "use_as_tool": 0.45, "burn_as_fuel": 0.55},
    "stone_block": {"mine_by_hand": 0.45, "mine_with_pickaxe": 0.55, "craft_plank": 0.45,
                    "eat": 0.45, "use_as_tool": 0.45, "burn_as_fuel": 0.45},
    "apple": {"mine_by_hand": 0.55, "mine_with_pickaxe": 0.55, "craft_plank": 0.45,
              "eat": 0.45, "use_as_tool": 0.45, "burn_as_fuel": 0.45},
    "wooden_pickaxe": {"mine_by_hand": 0.55, "mine_with_pickaxe": 0.55, "craft_plank": 0.45,
                       "eat": 0.45, "use_as_tool": 0.45, "burn_as_fuel": 0.55},
}

# OCM constants
PROMOTION_MIN_SUPPORT = 2
PROMOTION_MIN_EPISODE_SPREAD = 2
PROMOTION_MIN_STABILITY = 0.70
PROMOTION_MAX_CONTRADICTION_RATE = 0.30
PROMOTION_MIN_POSITIVE_FEATURES = 3
PROMOTION_MAX_OVERLAP_WITH_OTHER = 0.60
CANDIDATE_MATCH_MIN_JACCARD = 0.40
CANDIDATE_MAX_CONTRADICTION_FOR_MATCHING = 0.40

EXP_MIN_SUPPORT_FOR_STABLE = 2
EXP_MAX_CONTRADICTION_FOR_REJECT = 2
EXP_MIN_CONFIDENCE_FOR_REJECT = 0.60

# 1J40a-2 calibrated thresholds
MIN_CONFIDENCE_FOR_EXPLANATION = 0.20
MIN_SUPPORT_FOR_EXPLANATION = 3
MIN_MARGIN_FOR_DECISION = 0.08
MIN_DISAGREEING_ACTIONS_FOR_SPLIT = 2
MIN_REPEAT_PAIRS_FOR_NOISE = 1
HELDOUT_TRAIN_EPISODES = 4

# SAR constants
OBSERVE_COST = config.OBSERVE_COST       # 0.005
PROBE_COST = config.PROBE_COST            # 0.05
FAILURE_PENALTY = 0.1                     # penalty for failed try
SUCCESS_REWARD = 0.5                      # reward for successful try (task value)
SWITCH_COST = 0.01                        # cost of switching target
INFO_GAIN_PER_NEW_FEATURE = 0.02          # value of discovering a new feature
INFO_GAIN_PER_NEW_STATE = 0.01            # value of discovering a new state feature
UNCERTAINTY_REDUCTION_VALUE = 0.03        # value of reducing candidate uncertainty

FORBIDDEN_FIELDS = [
    "mixed_source_type", "true_family", "hidden_subtype", "oracle_outcome",
    "full_object_state", "deceptive_flag", "prior_violation",
]

# =============================================================================
# ObjectCandidate + ObjectCandidateMemory (verbatim from 1J40a-2)
# =============================================================================
class ObjectCandidate:
    def __init__(self, candidate_id, initial_oid, initial_features, source_step_id, episode_id):
        self.candidate_id = candidate_id
        self.object_ids = [initial_oid]
        self.positive_feature_set = set()
        self.negative_feature_set = set()
        self.contradicted_features = {}
        self.confidence_score = 0.0
        self.stability_score = 0.0
        self.support_count = 1
        self.episodes_seen = {episode_id}
        self.creation_step_id = source_step_id
        self.last_updated_step_id = source_step_id
        self.source_step_ids = [source_step_id]
        self.promotion_status = "provisional"
        self.promotion_step_id = None
        self.observed_feature_evidence = {}
        self._init_features(initial_features, source_step_id)

    def _init_features(self, features, step_id):
        for feat, present in features.items():
            if present:
                self.positive_feature_set.add(feat)
                self.observed_feature_evidence[feat] = {
                    "count": 1, "source_step_ids": [step_id],
                    "first_seen_step": step_id, "last_seen_step": step_id,
                }

    def add_evidence(self, oid, features, step_id, episode_id):
        is_new_object = oid not in self.object_ids
        if is_new_object:
            self.object_ids.append(oid)
            self.support_count += 1
        self.episodes_seen.add(episode_id)
        self.source_step_ids.append(step_id)
        self.last_updated_step_id = step_id
        for feat, present in features.items():
            if present:
                if feat in self.observed_feature_evidence:
                    self.observed_feature_evidence[feat]["count"] += 1
                    self.observed_feature_evidence[feat]["source_step_ids"].append(step_id)
                    self.observed_feature_evidence[feat]["last_seen_step"] = step_id
                else:
                    self.observed_feature_evidence[feat] = {
                        "count": 1, "source_step_ids": [step_id],
                        "first_seen_step": step_id, "last_seen_step": step_id,
                    }
                    self.positive_feature_set.add(feat)
        if is_new_object:
            for feat in list(self.positive_feature_set):
                if not features.get(feat, False):
                    if feat not in self.contradicted_features:
                        self.contradicted_features[feat] = []
                    self.contradicted_features[feat].append(
                        {"oid": oid, "present": False, "step_id": step_id})
        self._recompute_scores()

    def _recompute_scores(self):
        if not self.positive_feature_set:
            self.confidence_score = 0.0; self.stability_score = 0.0
            return
        total_checks = 0; consistent_checks = 0
        for feat in self.positive_feature_set:
            ev = self.observed_feature_evidence.get(feat, {})
            consistent_checks += ev.get("count", 0)
            total_checks += self.support_count
        self.confidence_score = round(consistent_checks / max(total_checks, 1), 4)
        n_pos = len(self.positive_feature_set)
        self.stability_score = round(1.0 - (len(self.contradicted_features) / max(n_pos, 1)), 4)

    def feature_overlap_score(self, features, diagnosticity_weights=None):
        obs_set = {f for f, v in features.items() if v}
        if not obs_set or not self.positive_feature_set:
            return 0.0
        if diagnosticity_weights:
            intersection = self.positive_feature_set & obs_set
            union = self.positive_feature_set | obs_set
            w_inter = sum(diagnosticity_weights.get(f, 1.0) for f in intersection)
            w_union = sum(diagnosticity_weights.get(f, 1.0) for f in union)
            if w_union == 0:
                return 0.0
            return w_inter / w_union
        return len(self.positive_feature_set & obs_set) / max(len(self.positive_feature_set | obs_set), 1)

    def is_closed(self):
        if self.promotion_status == "stable":
            return True
        n_pos = len(self.positive_feature_set)
        if n_pos == 0:
            return False
        return len(self.contradicted_features) / n_pos > CANDIDATE_MAX_CONTRADICTION_FOR_MATCHING

    def jaccard_similarity(self, other_candidate):
        a = self.positive_feature_set; b = other_candidate.positive_feature_set
        if not a and not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    def meets_promotion_criteria(self):
        if self.support_count < PROMOTION_MIN_SUPPORT: return False
        if len(self.episodes_seen) < PROMOTION_MIN_EPISODE_SPREAD: return False
        if self.stability_score < PROMOTION_MIN_STABILITY: return False
        if len(self.positive_feature_set) < PROMOTION_MIN_POSITIVE_FEATURES: return False
        n_pos = len(self.positive_feature_set)
        if n_pos > 0 and (len(self.contradicted_features) / n_pos) > PROMOTION_MAX_CONTRADICTION_RATE: return False
        return True


class ObjectCandidateMemory:
    def __init__(self):
        self._candidates = []
        self._candidate_buffer = []
        self._object_feature_map = {}
        self._object_candidate_map = {}
        self._event_count = 0
        self._promotion_log = []
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._global_feature_object_count = defaultdict(int)
        self._total_objects_seen = 0

    def _compute_diagnosticity_weights(self):
        if self._total_objects_seen == 0:
            return {}
        weights = {}
        for feat in FEATURE_UNIVERSE_SET:
            freq = self._global_feature_object_count.get(feat, 0) / self._total_objects_seen
            weights[feat] = round(1.0 - freq, 4)
        return weights

    def process_event(self, event_record):
        self._event_count += 1
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in event_record:
                self._leakage_checks["hidden_feature_leakage_detected"] = True
                return
        oid = event_record.get("action_target")
        features_delta = event_record.get("observed_features_delta", {})
        episode_id = event_record.get("episode_id", 0)
        step_id = event_record.get("step_id", 0)
        if not oid or not features_delta:
            return
        if oid not in self._object_feature_map:
            self._object_feature_map[oid] = {}
            self._total_objects_seen += 1
            for feat in features_delta:
                if features_delta[feat]:
                    self._global_feature_object_count[feat] += 1
        self._object_feature_map[oid].update(features_delta)
        current_features = self._object_feature_map[oid]
        diagnosticity_weights = self._compute_diagnosticity_weights()
        best_match = self._find_best_candidate(oid, current_features, diagnosticity_weights)
        if best_match:
            best_match.add_evidence(oid, current_features, step_id, episode_id)
            self._object_candidate_map[oid] = best_match.candidate_id
        else:
            cand_id = f"cand_{len(self._candidates) + len(self._candidate_buffer):04d}"
            candidate = ObjectCandidate(cand_id, oid, current_features, step_id, episode_id)
            self._candidate_buffer.append(candidate)
            self._object_candidate_map[oid] = cand_id

    def _find_best_candidate(self, oid, features, diagnosticity_weights=None):
        best_candidate = None; best_score = 0.0
        for cand in self._candidates + self._candidate_buffer:
            if oid in cand.object_ids:
                return cand
            if cand.is_closed():
                continue
            score = cand.feature_overlap_score(features, diagnosticity_weights)
            if score > best_score:
                best_score = score; best_candidate = cand
        if best_score >= CANDIDATE_MATCH_MIN_JACCARD:
            return best_candidate
        return None

    def promote_candidates(self):
        promoted = []
        for cand in list(self._candidate_buffer):
            if cand.support_count >= PROMOTION_MIN_SUPPORT:
                max_overlap = 0.0
                for stable in self._candidates:
                    max_overlap = max(max_overlap, cand.jaccard_similarity(stable))
                if max_overlap < PROMOTION_MAX_OVERLAP_WITH_OTHER or not self._candidates:
                    if cand.meets_promotion_criteria():
                        self._candidate_buffer.remove(cand)
                        self._candidates.append(cand)
                        cand.promotion_status = "stable"
                        cand.promotion_step_id = cand.last_updated_step_id
                        promoted.append(cand)
                        self._promotion_log.append({
                            "candidate_id": cand.candidate_id,
                            "support_count": cand.support_count,
                            "episode_spread": len(cand.episodes_seen),
                            "stability_score": cand.stability_score,
                        })
        return promoted

    def get_candidate_for_object(self, oid):
        return self._object_candidate_map.get(oid)

    def get_candidate_by_id(self, cand_id):
        for c in self._candidates + self._candidate_buffer:
            if c.candidate_id == cand_id:
                return c
        return None

    def get_all_candidates(self):
        return self._candidates + self._candidate_buffer

    def get_object_features(self, oid):
        return self._object_feature_map.get(oid, {})


# =============================================================================
# ExperienceRecord + ExperienceMemory (verbatim from 1J40a-2)
# =============================================================================
class ExperienceRecord:
    def __init__(self, experience_id, object_id, candidate_id, features, state,
                 action_type, action_affordance, outcome, source_step_id, episode_id):
        self.experience_id = experience_id
        self.object_anchor_id = object_id
        self.candidate_anchor_id = candidate_id
        self.source_event_ids = [source_step_id]
        self.observed_feature_snapshots = [dict(features) if features else {}]
        self.observed_state_snapshots = [dict(state) if state else {}]
        self.action_type = action_type
        self.action_affordance = action_affordance
        self.outcome_history = [{"outcome": outcome, "step_id": source_step_id, "episode_id": episode_id}]
        self.success_count = 1 if outcome == "success" else 0
        self.failure_count = 1 if outcome == "failure" else 0
        self.support_count = 1
        self.contradiction_count = 0
        self.confidence_score = 1.0 if outcome in ("success", "failure") else 0.5
        self.episodes_seen = {episode_id}
        self.last_updated_step = source_step_id
        self.status = "provisional"
        self.object_ids = [object_id]

    def update(self, object_id, candidate_id, features, state, outcome,
               source_step_id, episode_id):
        self.source_event_ids.append(source_step_id)
        self.observed_feature_snapshots.append(dict(features) if features else {})
        self.observed_state_snapshots.append(dict(state) if state else {})
        self.outcome_history.append({"outcome": outcome, "step_id": source_step_id, "episode_id": episode_id})
        self.support_count += 1
        self.episodes_seen.add(episode_id)
        self.last_updated_step = source_step_id
        if object_id not in self.object_ids:
            self.object_ids.append(object_id)
        if outcome == "success":
            self.success_count += 1
        elif outcome == "failure":
            self.failure_count += 1
        if self.success_count > 0 and self.failure_count > 0:
            self.contradiction_count = min(self.success_count, self.failure_count)
        self._recompute_status()

    def _recompute_status(self):
        if self.support_count == 0:
            self.confidence_score = 0.0
        else:
            majority = max(self.success_count, self.failure_count)
            self.confidence_score = round(majority / self.support_count, 4)
        if self.support_count >= 3 and self.contradiction_count >= EXP_MAX_CONTRADICTION_FOR_REJECT \
                and self.confidence_score < EXP_MIN_CONFIDENCE_FOR_REJECT:
            self.status = "rejected"
        elif self.contradiction_count > 0:
            self.status = "contradicted"
        elif self.support_count >= EXP_MIN_SUPPORT_FOR_STABLE and self.contradiction_count == 0:
            self.status = "stable"
        else:
            self.status = "provisional"

    def dominant_outcome(self):
        if self.success_count >= self.failure_count:
            return "success" if self.success_count > 0 else None
        return "failure" if self.failure_count > 0 else None


class ExperienceMemory:
    def __init__(self):
        self._records = []
        self._record_index = {}
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._feature_only_rule_detected = False
        self._per_object_actions = defaultdict(set)
        self._per_candidate_actions = defaultdict(set)

    def process_event(self, event_record, ocm):
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in event_record:
                self._leakage_checks["hidden_feature_leakage_detected"] = True
                return
        if event_record.get("action_type") != "try":
            return
        oid = event_record.get("action_target")
        if not oid:
            return
        action_affordance = event_record.get("action_params", {}).get("affordance", "")
        if not action_affordance:
            return
        outcome_bool = event_record.get("success_or_failure")
        outcome = "success" if outcome_bool is True else ("failure" if outcome_bool is False else "null")
        features = event_record.get("observed_features_delta", {})
        state = event_record.get("observed_state_delta", {})
        step_id = event_record.get("step_id", 0)
        episode_id = event_record.get("episode_id", 0)
        candidate_id = ocm.get_candidate_for_object(oid)
        if not candidate_id and not oid:
            self._feature_only_rule_detected = True
            return
        anchor_key = candidate_id if candidate_id else oid
        self._per_object_actions[oid].add(action_affordance)
        if candidate_id:
            self._per_candidate_actions[candidate_id].add((action_affordance, outcome))
        record_key = (anchor_key, action_affordance)
        if record_key in self._record_index:
            self._record_index[record_key].update(oid, candidate_id, features, state, outcome, step_id, episode_id)
        else:
            exp_id = f"exp_{anchor_key}_{action_affordance}"
            self._records.append(ExperienceRecord(
                exp_id, oid, candidate_id, features, state, "try", action_affordance, outcome, step_id, episode_id))
            self._record_index[record_key] = self._records[-1]

    def get_all_records(self):
        return self._records

    def get_mixed_records(self):
        return [r for r in self._records if r.success_count > 0 and r.failure_count > 0]

    def get_record(self, candidate_id, action):
        return self._record_index.get((candidate_id, action))

    def get_candidate_action_outcomes(self, candidate_id):
        """Return {action: (success_count, failure_count)} for a candidate."""
        result = {}
        for (cid, action), rec in self._record_index.items():
            if cid == candidate_id:
                result[action] = (rec.success_count, rec.failure_count)
        return result


# =============================================================================
# C5 Generator + Event Log Builder (verbatim from 1J40a-2)
# =============================================================================
def generate_c5_objects(rng):
    objects = {}
    audit_labels = {}
    counts = {"wood_log": 15, "stone_block": 15, "apple": 15, "wooden_pickaxe": 15}
    id_subtypes = {}
    for cat, count in counts.items():
        subtype_def = SUBTYPE_DEFINITIONS[cat]
        subtypes = subtype_def["subtypes"]
        ratios = subtype_def["subtype_ratio"]
        st_counts = []
        remaining = count
        for j, st in enumerate(subtypes):
            if j == len(subtypes) - 1:
                n = remaining
            else:
                n = round(count * ratios[st])
                remaining -= n
            st_counts.append((st, n))
        st_list = []
        for st, n in st_counts:
            st_list.extend([st] * n)
        id_subtypes[cat] = st_list

    for cat, count in counts.items():
        alloc = MIXED_SOURCE_ALLOCATION[cat]
        idx = 0
        n_id = alloc["identity_split"]
        cat_subtypes = id_subtypes[cat]
        id_st_list = cat_subtypes[:n_id]
        remaining_st = cat_subtypes[n_id:]
        for st in id_st_list:
            oid = f"test_{cat}_{idx:03d}"; idx += 1
            feats = _generate_visible_features_subtype(cat, st, rng)
            hidden_profile = _make_hidden_profile_subtype(cat, st, rng)
            ff_vals = _generate_function_feature_values(hidden_profile, cat, "test", rng)
            objects[oid] = {"id": oid, "hidden_category": cat, "hidden_subtype": st,
                "hidden_affordance_profile": hidden_profile,
                "visible_features": feats, "visible_state": {},
                "function_feature_values": ff_vals}
            audit_labels[oid] = {"mixed_source_type": "identity_split",
                "true_family": CATEGORY_TO_FAMILY[cat], "hidden_subtype": st}
        n_gap = alloc["observation_gap"]
        gap_st_list = remaining_st[:n_gap]
        remaining_st = remaining_st[n_gap:]
        if len(gap_st_list) < n_gap:
            subtype_opts = SUBTYPE_DEFINITIONS[cat]["subtypes"]
            while len(gap_st_list) < n_gap:
                gap_st_list.append(subtype_opts[len(gap_st_list) % len(subtype_opts)])
        for st in gap_st_list:
            oid = f"test_{cat}_{idx:03d}"; idx += 1
            feats = _generate_visible_features_subtype(cat, st, rng)
            hidden_profile = _make_hidden_profile_subtype(cat, st, rng)
            ff_vals = _generate_function_feature_values(hidden_profile, cat, "test", rng)
            family_feats = TYPE_FAMILIES.get(CATEGORY_TO_FAMILY[cat], set())
            diagnostic_true = [f for f in family_feats if feats.get(f, False)]
            if not diagnostic_true:
                diagnostic_true = [f for f in family_feats if f in feats]
            n_hide = min(2, max(1, len(diagnostic_true)))
            hidden_features = rng.sample(diagnostic_true, n_hide) if len(diagnostic_true) >= n_hide else list(diagnostic_true)
            objects[oid] = {"id": oid, "hidden_category": cat, "hidden_subtype": st,
                "hidden_affordance_profile": hidden_profile,
                "visible_features": feats, "visible_state": {},
                "function_feature_values": ff_vals,
                "_gap_hidden_features": hidden_features}
            audit_labels[oid] = {"mixed_source_type": "observation_gap",
                "true_family": CATEGORY_TO_FAMILY[cat], "hidden_subtype": st,
                "gap_hidden_features": hidden_features}
        n_state = alloc["state_condition"]
        sc_def = STATE_CONDITION_PROFILES[cat]
        state_names = sc_def["states"]
        for si in range(n_state):
            oid = f"test_{cat}_{idx:03d}"; idx += 1
            init_state = state_names[si % len(state_names)]
            alt_state = state_names[1] if init_state == state_names[0] else state_names[0]
            default_st = SUBTYPE_DEFINITIONS[cat]["subtypes"][0]
            feats = _generate_visible_features_subtype(cat, default_st, rng)
            init_profile = dict(sc_def["affordance_by_state"][init_state])
            init_profile["tap_sound"] = "clear" if rng.random() < 0.5 else "dull"
            init_profile["push"] = "success" if rng.random() < 0.7 else "fail"
            init_profile["roll"] = "success" if rng.random() < 0.4 else "fail"
            init_profile["inspect"] = "success"
            ff_vals = _generate_function_feature_values(init_profile, cat, "test", rng)
            alt_profile = dict(sc_def["affordance_by_state"][alt_state])
            alt_profile.update({k: init_profile[k] for k in ["tap_sound", "push", "roll", "inspect"]})
            objects[oid] = {"id": oid, "hidden_category": cat,
                "hidden_subtype": f"state_{init_state}",
                "hidden_affordance_profile": init_profile,
                "visible_features": feats,
                "visible_state": dict(sc_def["state_features"][init_state]),
                "function_feature_values": ff_vals,
                "_sc_initial_state": init_state, "_sc_alt_state": alt_state,
                "_sc_alt_profile": alt_profile,
                "_sc_alt_state_features": dict(sc_def["state_features"][alt_state])}
            audit_labels[oid] = {"mixed_source_type": "state_condition",
                "true_family": CATEGORY_TO_FAMILY[cat],
                "hidden_subtype": f"state_{init_state}",
                "initial_state": init_state, "alt_state": alt_state}
        n_noise = alloc["irreducible_noise"]
        noise_def = IRREDUCIBLE_NOISE_BASE[cat]
        for ni in range(n_noise):
            oid = f"test_{cat}_{idx:03d}"; idx += 1
            default_st = SUBTYPE_DEFINITIONS[cat]["subtypes"][0]
            feats = _generate_visible_features_subtype(cat, default_st, rng)
            base_profile = {}
            for action in ALL_ACTIONS:
                prob = noise_def.get(action, 0.5)
                base_profile[action] = "success" if prob >= 0.5 else "fail"
            base_profile["tap_sound"] = "clear" if rng.random() < 0.5 else "dull"
            base_profile["push"] = "success" if rng.random() < 0.7 else "fail"
            base_profile["roll"] = "success" if rng.random() < 0.4 else "fail"
            base_profile["inspect"] = "success"
            ff_vals = _generate_function_feature_values(base_profile, cat, "test", rng)
            objects[oid] = {"id": oid, "hidden_category": cat, "hidden_subtype": "noise",
                "hidden_affordance_profile": base_profile,
                "visible_features": feats, "visible_state": {},
                "function_feature_values": ff_vals,
                "_noise_probs": {a: noise_def.get(a, 0.5) for a in ALL_ACTIONS}}
            audit_labels[oid] = {"mixed_source_type": "irreducible_noise",
                "true_family": CATEGORY_TO_FAMILY[cat], "hidden_subtype": "noise",
                "noise_probs": {a: noise_def.get(a, 0.5) for a in ALL_ACTIONS}}
    return objects, audit_labels


def build_c5_event_log(test_objects, audit_labels, rng):
    test_oids = sorted(test_objects.keys())
    ep_rng = random.Random(rng.randint(0, 2**31 - 1))
    fam_to_oids = defaultdict(list)
    for oid in test_oids:
        fam = audit_labels[oid]["true_family"]
        fam_to_oids[fam].append(oid)
    for fam in fam_to_oids:
        fam_to_oids[fam].sort()
    episode_assignments = {}
    family_order = ["wood-like", "stone-like", "apple-like", "tool-like"]
    for ep in range(N_EPISODES):
        for fam in family_order:
            oid_list = fam_to_oids[fam]
            chunk = [oid for i, oid in enumerate(oid_list) if i % N_EPISODES == ep]
            for oid in chunk:
                episode_assignments[oid] = ep
    all_events = []
    step_id = 0
    event_rng = random.Random(rng.randint(0, 2**31 - 1))
    for ep in range(N_EPISODES):
        ep_oids = [oid for oid in test_oids if episode_assignments.get(oid) == ep]
        for oid in ep_oids:
            obj = test_objects[oid]
            ms_type = audit_labels[oid]["mixed_source_type"]
            features = obj.get("visible_features", {})
            state = obj.get("visible_state", {})
            category = obj.get("hidden_category", "")
            step_id += 1
            if ms_type == "observation_gap":
                obs_features_delta = dict(features)
                for hf in obj.get("_gap_hidden_features", []):
                    if hf in obs_features_delta:
                        del obs_features_delta[hf]
            else:
                obs_features_delta = dict(features)
            all_events.append({
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {},
                "observed_features_delta": obs_features_delta,
                "observed_state_delta": dict(state) if state else {},
                "success_or_failure": None,
            })
            for action in ALL_ACTIONS:
                if ms_type == "irreducible_noise":
                    prob = obj.get("_noise_probs", {}).get(action, 0.5)
                    outcome_bool = event_rng.random() < prob
                elif ms_type == "state_condition":
                    init_state = obj.get("_sc_initial_state", "")
                    sc_def = STATE_CONDITION_PROFILES.get(category, {})
                    aff_by_state = sc_def.get("affordance_by_state", {})
                    state_aff = aff_by_state.get(init_state, {})
                    outcome_bool = (state_aff.get(action, "fail") == "success")
                else:
                    outcome_bool = (obj["hidden_affordance_profile"].get(action, "fail") == "success")
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid, "action_params": {"affordance": action},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state) if state else {},
                    "success_or_failure": outcome_bool,
                })
            if ms_type == "observation_gap":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {"reobserve": True},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state) if state else {},
                    "success_or_failure": None,
                })
            if ms_type == "state_condition":
                alt_state_features = obj.get("_sc_alt_state_features", {})
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {"reobserve": True, "state_changed": True},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(alt_state_features),
                    "success_or_failure": None,
                })
                alt_profile = obj.get("_sc_alt_profile", {})
                for action in ["eat", "use_as_tool", "craft_plank"]:
                    if action in alt_profile:
                        outcome_bool = (alt_profile[action] == "success")
                        step_id += 1
                        all_events.append({
                            "step_id": step_id, "episode_id": ep, "action_type": "try",
                            "action_target": oid, "action_params": {"affordance": action, "state_changed": True},
                            "observed_features_delta": dict(features),
                            "observed_state_delta": dict(alt_state_features),
                            "success_or_failure": outcome_bool,
                        })
            if ms_type == "irreducible_noise":
                for action in ["eat", "craft_plank"]:
                    prob = obj.get("_noise_probs", {}).get(action, 0.5)
                    outcome_bool = event_rng.random() < prob
                    step_id += 1
                    all_events.append({
                        "step_id": step_id, "episode_id": ep, "action_type": "try",
                        "action_target": oid, "action_params": {"affordance": action, "repeat": True},
                        "observed_features_delta": dict(features),
                        "observed_state_delta": dict(state) if state else {},
                        "success_or_failure": outcome_bool,
                    })
    return all_events, episode_assignments


# =============================================================================
# 1J40a-2 Explanation Module (imported interface, full implementation copied)
# =============================================================================
class MixedExplanationModule:
    """Conservative shadow-only explanation from 1J40a-2.
    Used here only to produce soft explanation scores as belief-state features.
    """

    def __init__(self, all_events, ocm, experience_memory, episode_assignments,
                 confidence_threshold=0.10, margin_threshold=0.04):
        self._events = all_events
        self._ocm = ocm
        self._exp = experience_memory
        self._ep_assignments = episode_assignments
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._explanations = []
        self._candidate_split_performed = False
        self._state_condition_created = False
        self._conf_threshold = confidence_threshold
        self._margin_threshold = margin_threshold
        self._obj_events = defaultdict(list)
        for ev in all_events:
            oid = ev.get("action_target")
            if oid:
                self._obj_events[oid].append(ev)

    def explain_all_mixed(self):
        self._explanations = []
        mixed_records = self._exp.get_mixed_records()
        for rec in mixed_records:
            if rec.candidate_anchor_id is None:
                continue
            cand = self._ocm.get_candidate_by_id(rec.candidate_anchor_id)
            if cand is None:
                continue
            explanation = self._explain_one(rec, cand)
            self._explanations.append(explanation)
        return self._explanations

    def _explain_one(self, exp_rec, candidate):
        cand_id = candidate.candidate_id
        action = exp_rec.action_affordance
        obj_ids = candidate.object_ids
        source_event_ids = list(exp_rec.source_event_ids)
        source_exp_ids = [exp_rec.experience_id]
        gap_score, gap_evidence = self._score_observation_gap(obj_ids, exp_rec)
        state_score, state_evidence = self._score_state_condition(obj_ids, cand_id, action, exp_rec)
        split_score, split_evidence = self._score_identity_split(candidate, action, exp_rec)
        noise_score, noise_evidence = self._score_irreducible_noise(obj_ids, action, exp_rec)
        scores = {
            "observation_gap": gap_score, "state_condition": state_score,
            "identity_split": split_score, "irreducible_noise": noise_score,
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_scores[0]
        second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin = best_score - second_best_score
        if exp_rec.support_count < MIN_SUPPORT_FOR_EXPLANATION:
            predicted = "unresolved"; confidence = max(0.0, best_score)
        elif best_score < self._conf_threshold:
            predicted = "unresolved"; confidence = round(best_score, 4)
        elif margin < self._margin_threshold:
            predicted = "unresolved"; confidence = round(best_score, 4)
        elif best_type == "identity_split" and (noise_score >= split_score - self._margin_threshold * 1.5):
            if noise_evidence.get("inconsistent_repeat_pairs", 0) >= MIN_REPEAT_PAIRS_FOR_NOISE:
                predicted = "irreducible_noise"; confidence = round(noise_score, 4)
            else:
                predicted = "unresolved"; confidence = round(best_score, 4)
        elif best_type == "irreducible_noise" and noise_evidence.get("inconsistent_repeat_pairs", 0) < MIN_REPEAT_PAIRS_FOR_NOISE:
            predicted = "unresolved"; confidence = round(best_score, 4)
        else:
            predicted = best_type; confidence = round(best_score, 4)
        return {
            "candidate_id": cand_id, "action": action,
            "support_count": exp_rec.support_count,
            "success_count": exp_rec.success_count,
            "failure_count": exp_rec.failure_count,
            "gap_score": round(gap_score, 4),
            "state_score": round(state_score, 4),
            "split_score": round(split_score, 4),
            "noise_score": round(noise_score, 4),
            "predicted_explanation": predicted, "confidence": confidence,
            "margin": round(margin, 4),
            "source_event_ids": source_event_ids,
            "source_experience_ids": source_exp_ids,
            "gap_evidence_keys": gap_evidence,
            "state_evidence_keys": state_evidence,
            "split_evidence_keys": split_evidence,
            "noise_evidence_keys": noise_evidence,
        }

    def _score_observation_gap(self, obj_ids, exp_rec):
        tokens_with_gap = 0; total_tokens = 0; heldout_improvement = 0.0
        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            observes = [ev for ev in events if ev.get("action_type") == "observe"]
            total_tokens += 1
            if len(observes) < 2: continue
            feat_sets = []
            for obs in observes:
                fd = obs.get("observed_features_delta", {})
                feat_sets.append(set(k for k, v in fd.items() if v))
            new_features_revealed = False
            for i in range(1, len(feat_sets)):
                if feat_sets[i] - feat_sets[i - 1]:
                    new_features_revealed = True; break
            if new_features_revealed: tokens_with_gap += 1
            train_events = [ev for ev in events if ev.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES]
            test_events = [ev for ev in events if ev.get("episode_id", 0) >= HELDOUT_TRAIN_EPISODES]
            if train_events and test_events:
                early_tries = [ev for ev in train_events if ev.get("action_type") == "try"]
                late_tries = [ev for ev in test_events if ev.get("action_type") == "try"]
                if early_tries and late_tries:
                    early_acc = sum(1 for ev in early_tries if ev.get("success_or_failure", False) is True) / len(early_tries)
                    late_acc = sum(1 for ev in late_tries if ev.get("success_or_failure", False) is True) / len(late_tries)
                    heldout_improvement += max(0, late_acc - early_acc)
        score = 0.0
        if total_tokens > 0:
            gap_rate = tokens_with_gap / total_tokens
            score = gap_rate * 0.6 + min(heldout_improvement, 0.4) * 0.4
        return score, {"tokens_with_gap": tokens_with_gap, "total_tokens": total_tokens,
                       "heldout_improvement": round(heldout_improvement, 4)}

    def _score_state_condition(self, obj_ids, cand_id, action, exp_rec):
        tokens_with_state_change = 0; total_tokens = 0
        state_outcomes = defaultdict(list); no_state_outcomes = []
        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            total_tokens += 1
            observes = [ev for ev in events if ev.get("action_type") == "observe"]
            state_sets = []
            for obs in observes:
                sd = obs.get("observed_state_delta", {})
                state_sets.append(frozenset(k for k, v in sd.items() if v))
            if len(set(state_sets)) > 1: tokens_with_state_change += 1
            tries = [ev for ev in events if ev.get("action_type") == "try"
                     and ev.get("action_params", {}).get("affordance") == action]
            for ev in tries:
                sd = ev.get("observed_state_delta", {})
                state_key = frozenset(k for k, v in sd.items() if v)
                outcome_bool = ev.get("success_or_failure")
                if state_key: state_outcomes[state_key].append(outcome_bool)
                else: no_state_outcomes.append(outcome_bool)
        all_outcomes = no_state_outcomes + [o for outs in state_outcomes.values() for o in outs]
        if len(all_outcomes) < 2:
            score = 0.0
        else:
            baseline_accuracy = max(sum(1 for o in all_outcomes if o is True),
                                    sum(1 for o in all_outcomes if o is False)) / len(all_outcomes)
            if state_outcomes:
                state_accs = []
                for sk, outcomes in state_outcomes.items():
                    if len(outcomes) >= 1:
                        acc = max(sum(1 for o in outcomes if o is True),
                                  sum(1 for o in outcomes if o is False)) / len(outcomes)
                        state_accs.append(acc)
                state_accuracy = sum(state_accs) / len(state_accs) if state_accs else 0.0
                improvement = max(0, state_accuracy - baseline_accuracy)
            else:
                improvement = 0.0
            state_rate = tokens_with_state_change / max(total_tokens, 1)
            score = state_rate * 0.5 + improvement * 0.5
        return score, {"tokens_with_state_change": tokens_with_state_change,
                       "total_tokens": total_tokens, "state_distinct_outcome_sets": len(state_outcomes)}

    def _score_identity_split(self, candidate, action, exp_rec):
        obj_ids = candidate.object_ids
        if len(obj_ids) < 2: return 0.0, {"reason": "too_few_objects"}
        obj_action_outcomes = defaultdict(dict)
        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            for ev in events:
                if ev.get("action_type") == "try":
                    act = ev.get("action_params", {}).get("affordance", "")
                    outcome = "success" if ev.get("success_or_failure") is True else "failure"
                    if act not in obj_action_outcomes[oid]:
                        obj_action_outcomes[oid][act] = []
                    obj_action_outcomes[oid][act].append(outcome)
        obj_dominant = {}
        for oid, act_outcomes in obj_action_outcomes.items():
            obj_dominant[oid] = {}
            for act, outcomes in act_outcomes.items():
                successes = sum(1 for o in outcomes if o == "success")
                failures = len(outcomes) - successes
                obj_dominant[oid][act] = "success" if successes >= failures else "failure"
        disagreeing_actions = []
        for act in ALL_ACTIONS:
            outcomes_for_act = {}
            for oid in obj_ids:
                dom = obj_dominant.get(oid, {}).get(act)
                if dom: outcomes_for_act[oid] = dom
            if len(set(outcomes_for_act.values())) > 1:
                disagreeing_actions.append(act)
        if len(disagreeing_actions) < MIN_DISAGREEING_ACTIONS_FOR_SPLIT:
            return 0.0, {"disagreeing_actions": len(disagreeing_actions),
                         "reason": f"need_{MIN_DISAGREEING_ACTIONS_FOR_SPLIT}_actions"}
        n_disagree = len(disagreeing_actions)
        max_possible = min(6, len(ALL_ACTIONS))
        heldout_score = self._heldout_split_score(obj_ids, disagreeing_actions)
        multi_action_ratio = n_disagree / max_possible
        if n_disagree <= 1: complexity_penalty = 0.25
        elif n_disagree == 2: complexity_penalty = 0.45
        else: complexity_penalty = 0.70
        support_penalty = min(1.0, (len(obj_ids) - 1) / 8.0)
        heldout_weight = 0.4 if heldout_score > 0 else 0.0
        score = (multi_action_ratio * complexity_penalty * support_penalty * 0.5
                 + heldout_score * heldout_weight * 0.5)
        return score, {"disagreeing_actions": n_disagree, "object_count": len(obj_ids),
                       "heldout_split_score": round(heldout_score, 4),
                       "complexity_penalty": round(complexity_penalty, 4),
                       "support_penalty": round(support_penalty, 4)}

    def _heldout_split_score(self, obj_ids, disagreeing_actions):
        train_objs = []; test_objs = []
        for oid in obj_ids:
            ep = self._ep_assignments.get(oid, 0)
            if ep < HELDOUT_TRAIN_EPISODES: train_objs.append(oid)
            else: test_objs.append(oid)
        if not train_objs or not test_objs: return 0.0
        train_profiles = {}
        for oid in train_objs:
            events = self._obj_events.get(oid, [])
            profile = {}
            for ev in events:
                if ev.get("action_type") == "try":
                    act = ev.get("action_params", {}).get("affordance", "")
                    profile[act] = "success" if ev.get("success_or_failure") is True else "failure"
            train_profiles[oid] = profile
        subgroups = self._cluster_by_profile(train_profiles, disagreeing_actions)
        if len(subgroups) < 2: return 0.0
        baseline_correct = 0; subgroup_correct = 0; total_predictions = 0
        for oid in test_objs:
            events = self._obj_events.get(oid, [])
            test_profile = {}
            for ev in events:
                if ev.get("action_type") == "try" and ev.get("action_params", {}).get("affordance") in disagreeing_actions:
                    test_profile[ev.get("action_params", {}).get("affordance", "")] = \
                        "success" if ev.get("success_or_failure") is True else "failure"
            for act in disagreeing_actions:
                if act not in test_profile: continue
                total_predictions += 1; actual = test_profile[act]
                train_outcomes = [train_profiles.get(oid, {}).get(act) for oid in train_objs
                                  if act in train_profiles.get(oid, {})]
                if train_outcomes:
                    from collections import Counter
                    majority = Counter(train_outcomes).most_common(1)[0][0]
                    if majority == actual: baseline_correct += 1
                best_match = self._best_subgroup_for_object(test_profile, subgroups, disagreeing_actions)
                if best_match is not None:
                    sg_outcomes = [train_profiles.get(oid, {}).get(act) for oid in subgroups[best_match]
                                   if act in train_profiles.get(oid, {})]
                    if sg_outcomes:
                        from collections import Counter
                        sg_majority = Counter(sg_outcomes).most_common(1)[0][0]
                        if sg_majority == actual: subgroup_correct += 1
        if total_predictions == 0: return 0.0
        return max(0.0, subgroup_correct / total_predictions - baseline_correct / total_predictions)

    def _cluster_by_profile(self, profiles, actions):
        subgroups = defaultdict(list)
        for oid, profile in profiles.items():
            key = tuple(profile.get(act, "unknown") for act in sorted(actions))
            subgroups[key].append(oid)
        return dict(subgroups)

    def _best_subgroup_for_object(self, test_profile, subgroups, actions):
        best_match = None; best_score = -1
        test_key = tuple(test_profile.get(act) for act in sorted(actions))
        for sg_key, sg_oids in subgroups.items():
            matches = sum(1 for a, b in zip(test_key, sg_key) if a == b)
            if matches > best_score: best_score = matches; best_match = sg_key
        return best_match

    def _score_irreducible_noise(self, obj_ids, action, exp_rec):
        tokens_with_consistent_conditions = 0; total_tokens = 0
        total_inconsistent_pairs = 0; tokens_with_repeats = 0
        tokens_with_verified_inconsistency = 0
        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            total_tokens += 1
            action_tries = [ev for ev in events if ev.get("action_type") == "try"
                            and ev.get("action_params", {}).get("affordance") == action]
            if len(action_tries) < 2: continue
            tokens_with_repeats += 1
            grouped = defaultdict(list)
            for ev in action_tries:
                fd = ev.get("observed_features_delta", {})
                sd = ev.get("observed_state_delta", {})
                fkey = frozenset(k for k, v in fd.items() if v)
                skey = frozenset(k for k, v in sd.items() if v)
                grouped[(fkey, skey)].append(ev.get("success_or_failure"))
            object_has_inconsistency = False
            for (fkey, skey), outcomes in grouped.items():
                if len(outcomes) >= 2 and len(set(outcomes)) > 1:
                    total_inconsistent_pairs += 1
                    if not object_has_inconsistency:
                        object_has_inconsistency = True
                        tokens_with_consistent_conditions += 1
            if object_has_inconsistency:
                observes = [ev for ev in events if ev.get("action_type") == "observe"]
                state_sets = [frozenset(k for k, v in obs.get("observed_state_delta", {}).items() if v) for obs in observes]
                state_changes = len(set(state_sets)) > 1
                feat_sets_across_obs = [set(k for k, v in obs.get("observed_features_delta", {}).items() if v) for obs in observes]
                feat_changes = False
                if len(feat_sets_across_obs) >= 2:
                    if feat_sets_across_obs[-1] != feat_sets_across_obs[-2]:
                        feat_changes = True
                if not state_changes and not feat_changes:
                    tokens_with_verified_inconsistency += 1
        score = 0.0
        if total_tokens > 0:
            inconsistency_rate = tokens_with_consistent_conditions / total_tokens
            repeat_rate = tokens_with_repeats / total_tokens
            verified_rate = tokens_with_verified_inconsistency / max(tokens_with_consistent_conditions, 1)
            score = inconsistency_rate * 0.5 + repeat_rate * 0.3 + verified_rate * 0.2
        return score, {
            "tokens_with_inconsistency": tokens_with_consistent_conditions,
            "total_tokens": total_tokens,
            "inconsistent_repeat_pairs": total_inconsistent_pairs,
            "tokens_with_repeats": tokens_with_repeats,
            "tokens_with_verified_inconsistency": tokens_with_verified_inconsistency,
        }

    def get_explanation_scores(self, candidate_id, action):
        """Return soft explanation scores for a given (candidate, action) pair."""
        for exp in self._explanations:
            if exp["candidate_id"] == candidate_id and exp["action"] == action:
                return {
                    "observation_gap_score": exp["gap_score"],
                    "state_condition_score": exp["state_score"],
                    "identity_split_score": exp["split_score"],
                    "irreducible_noise_score": exp["noise_score"],
                    "unresolved_score": 1.0 if exp["predicted_explanation"] == "unresolved" else 0.0,
                    "confidence": exp["confidence"],
                    "top_score_margin": exp["margin"],
                    "preferred_explanation": exp["predicted_explanation"],
                }
        return {
            "observation_gap_score": 0.0, "state_condition_score": 0.0,
            "identity_split_score": 0.0, "irreducible_noise_score": 0.0,
            "unresolved_score": 1.0, "confidence": 0.0, "top_score_margin": 0.0,
            "preferred_explanation": "unresolved",
        }


# =============================================================================
# SITUATION FEATURE BUILDER (new for 1J40b-0)
# =============================================================================
class SituationFeatureBuilder:
    """Builds belief-state features from agent-visible data at each step.

    All features are derived from: Event Memory, OCM, Experience Memory,
    and 1J40a-2 soft explanation scores. NO audit labels or oracle data.
    """

    def __init__(self, ocm, exp_mem, explanation_module, episode_assignments):
        self._ocm = ocm
        self._exp = exp_mem
        self._explainer = explanation_module
        self._ep_assignments = episode_assignments
        # Running counters
        self._per_object_observe_count = defaultdict(int)
        self._per_object_try_count = defaultdict(int)
        self._per_candidate_observe_count = defaultdict(int)
        self._per_candidate_try_count = defaultdict(int)
        # Track what features have been observed per candidate
        self._candidate_observed_features = defaultdict(set)
        self._candidate_observed_states = defaultdict(set)
        # Track recent events (last N steps per candidate for recency features)
        self._recent_events = []

    def capture_before_event(self, event_record):
        """Capture situation features BEFORE the action in event_record is taken.
        Returns a dict of belief-state features.
        """
        oid = event_record.get("action_target", "")
        candidate_id = self._ocm.get_candidate_for_object(oid)
        if not candidate_id:
            candidate_id = oid

        cand = self._ocm.get_candidate_by_id(candidate_id)
        episode_id = event_record.get("episode_id", 0)

        features = {}

        # ---- Candidate-level features ----
        if cand:
            features["candidate_confidence"] = cand.confidence_score
            features["candidate_support_count"] = cand.support_count
            features["candidate_stability"] = cand.stability_score
            features["candidate_positive_feature_count"] = len(cand.positive_feature_set)
        else:
            features["candidate_confidence"] = 0.0
            features["candidate_support_count"] = 0
            features["candidate_stability"] = 0.0
            features["candidate_positive_feature_count"] = 0

        # ---- Mixed rate for this candidate ----
        if candidate_id:
            ca_outcomes = self._exp.get_candidate_action_outcomes(candidate_id)
            total_pairs = 0; mixed_pairs = 0
            for act, (succ, fail) in ca_outcomes.items():
                total_pairs += 1
                if succ > 0 and fail > 0:
                    mixed_pairs += 1
            features["candidate_mixed_rate"] = round(mixed_pairs / max(total_pairs, 1), 4)
            features["candidate_contradiction_count"] = mixed_pairs
        else:
            features["candidate_mixed_rate"] = 0.0
            features["candidate_contradiction_count"] = 0

        # ---- Feature coverage ----
        if candidate_id:
            obs_feats = self._candidate_observed_features.get(candidate_id, set())
            features["observed_feature_coverage"] = round(
                len(obs_feats) / max(len(FEATURE_UNIVERSE_SET), 1), 4)
            obs_states = self._candidate_observed_states.get(candidate_id, set())
            features["observed_state_coverage"] = round(
                len(obs_states) / max(len(STATE_FEATURE_KEYS), 1), 4)
        else:
            features["observed_feature_coverage"] = 0.0
            features["observed_state_coverage"] = 0.0

        # ---- Recency ----
        features["recent_observe_count"] = self._per_candidate_observe_count.get(candidate_id, 0)
        features["recent_try_count"] = self._per_candidate_try_count.get(candidate_id, 0)
        features["total_object_observes"] = self._per_object_observe_count.get(oid, 0)
        features["total_object_tries"] = self._per_object_try_count.get(oid, 0)

        # ---- Best current try value and margin ----
        if candidate_id:
            ca_outcomes = self._exp.get_candidate_action_outcomes(candidate_id)
            action_values = {}
            for act, (succ, fail) in ca_outcomes.items():
                total = succ + fail
                if total > 0:
                    action_values[act] = succ / total
            if action_values:
                sorted_vals = sorted(action_values.values(), reverse=True)
                features["best_current_try_value"] = round(sorted_vals[0], 4)
                if len(sorted_vals) > 1:
                    features["top_action_value_margin"] = round(sorted_vals[0] - sorted_vals[1], 4)
                else:
                    features["top_action_value_margin"] = 0.0
            else:
                features["best_current_try_value"] = 0.0
                features["top_action_value_margin"] = 0.0
        else:
            features["best_current_try_value"] = 0.0
            features["top_action_value_margin"] = 0.0

        # ---- Episode progress ----
        features["episode_id"] = episode_id
        features["episode_progress"] = round(episode_id / max(N_EPISODES, 1), 4)

        return features

    def update_after_event(self, event_record):
        """Update internal counters after event is processed."""
        oid = event_record.get("action_target", "")
        candidate_id = self._ocm.get_candidate_for_object(oid)
        action_type = event_record.get("action_type", "")

        self._per_object_observe_count[oid] = self._per_object_observe_count.get(oid, 0)
        self._per_object_try_count[oid] = self._per_object_try_count.get(oid, 0)

        if action_type == "observe":
            self._per_object_observe_count[oid] += 1
            if candidate_id:
                self._per_candidate_observe_count[candidate_id] = \
                    self._per_candidate_observe_count.get(candidate_id, 0) + 1
                # Track observed features
                fd = event_record.get("observed_features_delta", {})
                for feat, present in fd.items():
                    if present:
                        self._candidate_observed_features[candidate_id].add(feat)
                sd = event_record.get("observed_state_delta", {})
                for feat, present in sd.items():
                    if present:
                        self._candidate_observed_states[candidate_id].add(feat)

        elif action_type == "try":
            self._per_object_try_count[oid] += 1
            if candidate_id:
                self._per_candidate_try_count[candidate_id] = \
                    self._per_candidate_try_count.get(candidate_id, 0) + 1

        self._recent_events.append(event_record)


# =============================================================================
# RESULT + NET VALUE COMPUTER (new for 1J40b-0)
# =============================================================================
class ResultComputer:
    """Computes result summary and net_value for each action.

    Uses only agent-visible data: event outcomes, memory state changes.
    """

    def __init__(self, ocm, exp_mem, situation_builder):
        self._ocm = ocm
        self._exp = exp_mem
        self._situation = situation_builder

    def compute(self, event_record, situation_features_before):
        """Compute result and net_value for an event.

        Returns: (result_summary, net_value, is_proxy)
        """
        action_type = event_record.get("action_type", "")
        oid = event_record.get("action_target", "")
        candidate_id = self._ocm.get_candidate_for_object(oid)

        if action_type == "observe":
            return self._compute_observe_result(event_record, situation_features_before, oid, candidate_id)
        elif action_type == "try":
            return self._compute_try_result(event_record)
        else:
            return {"result_type": "unknown", "note": f"unhandled action_type={action_type}"}, 0.0, True

    def _compute_observe_result(self, event_record, sit_before, oid, candidate_id):
        """Compute info gain and net value for an observe action."""
        features_delta = event_record.get("observed_features_delta", {})
        state_delta = event_record.get("observed_state_delta", {})

        # Count newly revealed features (not seen before for this candidate)
        newly_revealed_features = 0
        if candidate_id:
            prev_features = self._situation._candidate_observed_features.get(candidate_id, set())
            for feat, present in features_delta.items():
                if present and feat not in prev_features:
                    newly_revealed_features += 1

        # Count newly revealed state features
        newly_revealed_states = 0
        if candidate_id:
            prev_states = self._situation._candidate_observed_states.get(candidate_id, set())
            for feat, present in state_delta.items():
                if present and feat not in prev_states:
                    newly_revealed_states += 1

        total_features_seen = len(features_delta)
        total_state_seen = len(state_delta)

        # Info gain proxy
        info_gain = (newly_revealed_features * INFO_GAIN_PER_NEW_FEATURE +
                     newly_revealed_states * INFO_GAIN_PER_NEW_STATE)

        # Uncertainty reduction: if this observe reveals features that distinguish
        # between candidates (reduces ambiguity in candidate matching)
        uncertainty_reduction = 0.0
        if newly_revealed_features > 0:
            uncertainty_reduction = UNCERTAINTY_REDUCTION_VALUE

        # Net value: benefit - cost
        net_value = round(info_gain + uncertainty_reduction - OBSERVE_COST, 4)

        result_summary = {
            "result_type": "observe",
            "total_features_seen": total_features_seen,
            "total_state_seen": total_state_seen,
            "newly_revealed_features": newly_revealed_features,
            "newly_revealed_states": newly_revealed_states,
            "info_gain_proxy": round(info_gain, 4),
            "uncertainty_reduction_proxy": round(uncertainty_reduction, 4),
            "observe_cost": OBSERVE_COST,
            "net_value": net_value,
        }

        return result_summary, net_value, False  # not a proxy

    def _compute_try_result(self, event_record):
        """Compute result and net value for a try action."""
        outcome_bool = event_record.get("success_or_failure")
        outcome = "success" if outcome_bool is True else ("failure" if outcome_bool is False else "null")

        if outcome == "success":
            raw_value = SUCCESS_REWARD
        elif outcome == "failure":
            raw_value = -FAILURE_PENALTY
        else:
            raw_value = 0.0

        net_value = round(raw_value - PROBE_COST, 4)

        result_summary = {
            "result_type": "try",
            "outcome": outcome,
            "success_reward": SUCCESS_REWARD if outcome == "success" else 0.0,
            "failure_penalty": -FAILURE_PENALTY if outcome == "failure" else 0.0,
            "probe_cost": PROBE_COST,
            "net_value": net_value,
        }

        return result_summary, net_value, False  # not a proxy

    def compute_switch_result(self, from_oid, to_oid):
        """Compute result for a switch action (only if real switch events exist).
        Returns proxy=True since we have no real switch events.
        """
        # No real switch events exist in current C5 - this is a proxy
        net_value = -SWITCH_COST  # Only cost, no benefit without real data
        result_summary = {
            "result_type": "switch",
            "from_object": from_oid,
            "to_object": to_oid,
            "switch_cost": SWITCH_COST,
            "net_value": net_value,
            "note": "proxy: no real switch events in C5 logs",
        }
        return result_summary, net_value, True  # proxy


# =============================================================================
# LEARNING RECORD BUILDER (new for 1J40b-0)
# =============================================================================
def build_learning_records(all_events, ocm, exp_mem, explainer, episode_assignments):
    """Process events in order, building SAR learning records.

    For each event:
    1. Capture situation features BEFORE the action
    2. Record the action and its result
    3. Compute net_value
    4. Create a learning record
    """
    situation_builder = SituationFeatureBuilder(ocm, exp_mem, explainer, episode_assignments)
    result_computer = ResultComputer(ocm, exp_mem, situation_builder)

    learning_records = []
    record_id_counter = 0
    leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}

    # Process events sequentially to simulate online belief state
    for ev in all_events:
        oid = ev.get("action_target", "")

        # Leakage check
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in ev:
                leakage_checks["hidden_feature_leakage_detected"] = True

        # Skip events without target
        if not oid:
            continue

        action_type = ev.get("action_type", "")
        action_params = ev.get("action_params", {})

        # Capture situation BEFORE the event
        situation_features = situation_builder.capture_before_event(ev)

        # Update OCM + Experience first (so result computation has updated state)
        # Actually, we need to compute result from the event itself.
        # The event IS the action. The situation is what the agent knew BEFORE.
        # The result is what happened AS A RESULT of the action.

        # Compute result from the event
        result_summary, net_value, is_proxy = result_computer.compute(ev, situation_features)

        # Build situation key
        candidate_id = ocm.get_candidate_for_object(oid) or oid
        if action_type == "try":
            action_key = f"try_{action_params.get('affordance', 'unknown')}"
        elif action_type == "observe":
            action_key = "observe"
        else:
            action_key = f"{action_type}_{action_params.get('affordance', '')}"

        situation_key = f"{candidate_id}|{action_key}"

        # Get explanation scores for this (candidate, action)
        expl_scores = {}
        if action_type == "try" and candidate_id:
            expl_scores = explainer.get_explanation_scores(
                candidate_id, action_params.get("affordance", ""))

        record_id_counter += 1
        record = {
            "record_id": f"sar_{record_id_counter:05d}",
            "situation_key": situation_key,
            "situation_features": situation_features,
            "action_type": action_type,
            "action_params": dict(action_params),
            "result_summary": result_summary,
            "net_value": net_value,
            "source_event_ids": [ev.get("step_id")],
            "source_candidate_ids": [candidate_id] if candidate_id else [],
            "source_experience_ids": [],
            "explanation_feature_source": expl_scores,
            "explanation_features_used": len(expl_scores) > 0,
            "evidence_mode": "observed_action",
            "is_proxy_estimate": is_proxy,
            "episode_id": ev.get("episode_id", 0),
            "step_id": ev.get("step_id", 0),
        }

        learning_records.append(record)

        # Update situation builder AFTER creating the record
        situation_builder.update_after_event(ev)

    return learning_records, leakage_checks


# =============================================================================
# AUDIT
# =============================================================================
def audit_learning_records(learning_records, all_events, ocm, exp_mem, explainer,
                           leakage_checks):
    """Run all required audit checks on learning records."""
    from collections import Counter

    # 1. Action event availability
    observe_events = [ev for ev in all_events if ev.get("action_type") == "observe"]
    try_events = [ev for ev in all_events if ev.get("action_type") == "try"]
    switch_events = [ev for ev in all_events if ev.get("action_type") == "switch"]

    observe_event_count = len(observe_events)
    try_event_count = len(try_events)
    switch_event_count = len(switch_events)

    try_actions = set()
    for ev in try_events:
        act = ev.get("action_params", {}).get("affordance", "")
        if act:
            try_actions.add(act)

    switch_learning_available = switch_event_count > 0

    # 2. Count learning records by type
    observe_records = [r for r in learning_records if r["action_type"] == "observe"]
    try_records = [r for r in learning_records if r["action_type"] == "try"]
    switch_records = [r for r in learning_records if r["action_type"] == "switch"]

    total = len(learning_records)

    # 3. Records with valid net_value
    records_with_valid_nv = [r for r in learning_records
                             if isinstance(r.get("net_value"), (int, float)) and r["net_value"] != 0.0]
    # Actually, net_value can be 0 legitimately. Let's check it's not None/NaN.
    records_with_valid_nv = [r for r in learning_records
                             if r.get("net_value") is not None and isinstance(r.get("net_value"), (int, float))]

    # 4. Records with source IDs
    records_with_source = [r for r in learning_records
                           if len(r.get("source_event_ids", [])) > 0]

    # 5. Proxy records
    proxy_records = [r for r in learning_records if r.get("is_proxy_estimate", False)]

    # 6. Records using soft explanation features
    records_with_expl = [r for r in learning_records if r.get("explanation_features_used", False)]

    # 7. Situation features check: no forbidden fields
    forbidden_in_sit = False
    for r in learning_records:
        sf = r.get("situation_features", {})
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in sf:
                forbidden_in_sit = True
                break

    # 8. Per-action-type net value stats
    observe_net_values = [r["net_value"] for r in observe_records]
    try_net_values = [r["net_value"] for r in try_records]

    audit = {
        "total_learning_records": total,
        "observe_learning_records": len(observe_records),
        "try_learning_records": len(try_records),
        "switch_learning_records": len(switch_records),

        "action_event_availability": {
            "observe_event_count": observe_event_count,
            "try_event_count": try_event_count,
            "switch_event_count": switch_event_count,
            "unique_try_action_types": sorted(try_actions),
            "switch_learning_available": switch_learning_available,
        },

        "records_with_valid_net_value": len(records_with_valid_nv),
        "records_with_source_ids": len(records_with_source),
        "records_using_soft_explanation_features": len(records_with_expl),
        "proxy_records_created": len(proxy_records),

        "net_value_stats": {
            "observe_mean_nv": round(sum(observe_net_values) / max(len(observe_net_values), 1), 4) if observe_net_values else 0.0,
            "observe_min_nv": round(min(observe_net_values), 4) if observe_net_values else 0.0,
            "observe_max_nv": round(max(observe_net_values), 4) if observe_net_values else 0.0,
            "try_mean_nv": round(sum(try_net_values) / max(len(try_net_values), 1), 4) if try_net_values else 0.0,
            "try_min_nv": round(min(try_net_values), 4) if try_net_values else 0.0,
            "try_max_nv": round(max(try_net_values), 4) if try_net_values else 0.0,
        },

        "situation_feature_counts": {},
        "hidden_feature_leakage_in_situation": forbidden_in_sit,
        "hidden_feature_leakage_detected": leakage_checks.get("hidden_feature_leakage_detected", False),
        "oracle_leakage_detected": leakage_checks.get("oracle_leakage_detected", False),
        "policy_decisions_changed": False,
    }

    # Count situation feature keys across all records
    feat_key_counts = Counter()
    for r in learning_records:
        for k in r.get("situation_features", {}).keys():
            feat_key_counts[k] += 1
    audit["situation_feature_counts"] = dict(feat_key_counts.most_common())

    # Decision rule
    can_proceed = (
        total > 0
        and len(observe_records) > 0
        and len(try_records) > 0
        and len(records_with_valid_nv) == total
        and len(records_with_source) == total
        and len(proxy_records) == 0
        and not leakage_checks.get("hidden_feature_leakage_detected", False)
        and not leakage_checks.get("oracle_leakage_detected", False)
        and not forbidden_in_sit
    )

    audit["can_proceed_to_1J40b1"] = can_proceed

    failure_reasons = []
    if total == 0: failure_reasons.append("no_learning_records")
    if len(observe_records) == 0: failure_reasons.append("no_observe_records")
    if len(try_records) == 0: failure_reasons.append("no_try_records")
    if len(records_with_valid_nv) != total: failure_reasons.append("some_records_missing_net_value")
    if len(records_with_source) != total: failure_reasons.append("some_records_missing_source_ids")
    if len(proxy_records) > 0: failure_reasons.append(f"{len(proxy_records)}_proxy_records")
    if leakage_checks.get("hidden_feature_leakage_detected"): failure_reasons.append("hidden_feature_leakage")
    if leakage_checks.get("oracle_leakage_detected"): failure_reasons.append("oracle_leakage")
    if forbidden_in_sit: failure_reasons.append("forbidden_fields_in_situation_features")

    audit["failure_reason"] = "; ".join(failure_reasons) if failure_reasons else "none"
    audit["implementation_status"] = "pass" if can_proceed else "fail"

    return audit


# =============================================================================
# Main runner for a single seed
# =============================================================================
def run_seed(seed):
    t_start = time.time()

    c5_label = "C5_mixed_source_identifiability_v0"
    c5_cond = None
    for c in config.CUE_CONDITIONS:
        if c["label"] == c5_label:
            c5_cond = copy.deepcopy(c)
            break
    if c5_cond is None:
        raise RuntimeError(f"{c5_label} not found in CUE_CONDITIONS")

    student, base_learner, train_objects, train_env, _std_test, _std_test_env, final_metrics, rng = \
        run_phase_a_training(seed, c5_cond)

    obj_rng = random.Random(rng.randint(0, 2**31 - 1))
    test_objects, audit_labels = generate_c5_objects(obj_rng)
    event_rng = random.Random(rng.randint(0, 2**31 - 1))
    all_events, episode_assignments = build_c5_event_log(test_objects, audit_labels, event_rng)

    # Process through OCM + ExperienceMemory
    ocm = ObjectCandidateMemory()
    exp_mem = ExperienceMemory()

    for ev in all_events:
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in ev:
                ocm._leakage_checks["hidden_feature_leakage_detected"] = True
                exp_mem._leakage_checks["hidden_feature_leakage_detected"] = True
        if ev.get("action_type") == "observe":
            ocm.process_event(ev)
        elif ev.get("action_type") == "try":
            ocm.process_event(ev)
            exp_mem.process_event(ev, ocm)

    ocm.promote_candidates()

    # Run 1J40a-2 explainer for soft scores
    explainer = MixedExplanationModule(
        all_events, ocm, exp_mem, episode_assignments,
        confidence_threshold=0.10, margin_threshold=0.04,
    )
    explainer.explain_all_mixed()

    # Build learning records (shadow, no policy activation)
    learning_records, leakage_checks = build_learning_records(
        all_events, ocm, exp_mem, explainer, episode_assignments)

    # Audit
    audit = audit_learning_records(
        learning_records, all_events, ocm, exp_mem, explainer, leakage_checks)

    # Merge leakage from all sources
    audit["hidden_feature_leakage_detected"] = (
        audit["hidden_feature_leakage_detected"]
        or ocm._leakage_checks["hidden_feature_leakage_detected"]
        or exp_mem._leakage_checks["hidden_feature_leakage_detected"]
        or explainer._leakage_checks["hidden_feature_leakage_detected"]
    )
    audit["oracle_leakage_detected"] = (
        audit["oracle_leakage_detected"]
        or ocm._leakage_checks["oracle_leakage_detected"]
        or exp_mem._leakage_checks["oracle_leakage_detected"]
        or explainer._leakage_checks["oracle_leakage_detected"]
    )

    elapsed = round(time.time() - t_start, 2)

    return {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "audit": audit,
        "learning_records": learning_records,
        "object_count": len(test_objects),
        "event_count": len(all_events),
        "candidate_count": len(ocm._candidates) + len(ocm._candidate_buffer),
    }


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Block 1J40b-0 -- Situation-Action-Result Learning Feasibility Audit")
    print(f"  seeds={MULTISEEDS}  episodes={N_EPISODES}")
    print(f"  condition=C5_mixed_source_identifiability_v0")
    print("=" * 70)
    
    all_results = []
    for seed in MULTISEEDS:
        print(f"\n--- Seed {seed} ---")
        result = run_seed(seed)
        all_results.append(result)
        a = result["audit"]
        print(f"  total_records={a['total_learning_records']}  "
              f"observe={a['observe_learning_records']}  "
              f"try={a['try_learning_records']}  "
              f"switch={a['switch_learning_records']}")
        print(f"  switch_available={a['action_event_availability']['switch_learning_available']}  "
              f"proxy_records={a['proxy_records_created']}")
        print(f"  valid_nv={a['records_with_valid_net_value']}/{a['total_learning_records']}  "
              f"source_ids={a['records_with_source_ids']}/{a['total_learning_records']}")
        print(f"  leakage: hidden={a['hidden_feature_leakage_detected']}  "
              f"oracle={a['oracle_leakage_detected']}  "
              f"sit_forbidden={a['hidden_feature_leakage_in_situation']}")
        print(f"  can_proceed={a['can_proceed_to_1J40b1']}")
    
    # =============================================================================
    # Aggregate and output
    # =============================================================================
    total_elapsed = round(sum(r["elapsed_seconds"] for r in all_results), 1)
    
    agg_audit = {
        "total_learning_records": sum(r["audit"]["total_learning_records"] for r in all_results),
        "observe_learning_records": sum(r["audit"]["observe_learning_records"] for r in all_results),
        "try_learning_records": sum(r["audit"]["try_learning_records"] for r in all_results),
        "switch_learning_records": sum(r["audit"]["switch_learning_records"] for r in all_results),
        "total_objects": sum(r["object_count"] for r in all_results),
        "total_events": sum(r["event_count"] for r in all_results),
        "total_candidates": sum(r["candidate_count"] for r in all_results),
    }
    
    # Aggregate switch availability
    any_switch = any(r["audit"]["action_event_availability"]["switch_learning_available"]
                     for r in all_results)
    agg_audit["switch_learning_available"] = any_switch
    
    # Aggregate net value
    all_nv_valid = all(r["audit"]["records_with_valid_net_value"] == r["audit"]["total_learning_records"]
                       for r in all_results)
    all_nv_source = all(r["audit"]["records_with_source_ids"] == r["audit"]["total_learning_records"]
                        for r in all_results)
    agg_audit["records_with_valid_net_value"] = agg_audit["total_learning_records"] if all_nv_valid else 0
    agg_audit["records_with_source_ids"] = agg_audit["total_learning_records"] if all_nv_source else 0
    
    # Aggregate proxy
    agg_proxy = sum(r["audit"]["proxy_records_created"] for r in all_results)
    agg_audit["proxy_records_created"] = agg_proxy
    
    # Aggregate explanation usage
    agg_expl_records = sum(r["audit"]["records_using_soft_explanation_features"] for r in all_results)
    agg_audit["records_using_soft_explanation_features"] = agg_expl_records
    agg_audit["soft_explanation_features_used"] = agg_expl_records > 0
    
    # Leakage
    any_hidden = any(r["audit"]["hidden_feature_leakage_detected"] for r in all_results)
    any_oracle = any(r["audit"]["oracle_leakage_detected"] for r in all_results)
    any_sit_forbidden = any(r["audit"]["hidden_feature_leakage_in_situation"] for r in all_results)
    agg_audit["hidden_feature_leakage_detected"] = any_hidden or any_sit_forbidden
    agg_audit["oracle_leakage_detected"] = any_oracle
    agg_audit["policy_decisions_changed"] = False
    
    # Situation feature counts (union across seeds)
    all_feat_keys = defaultdict(int)
    for r in all_results:
        for k, v in r["audit"].get("situation_feature_counts", {}).items():
            all_feat_keys[k] += v
    agg_audit["situation_feature_counts"] = dict(sorted(all_feat_keys.items(), key=lambda x: -x[1]))
    
    # Try action types
    all_try_actions = set()
    for r in all_results:
        for act in r["audit"]["action_event_availability"]["unique_try_action_types"]:
            all_try_actions.add(act)
    agg_audit["unique_try_actions"] = sorted(all_try_actions)
    
    # Net value stats
    all_observe_nvs = []
    all_try_nvs = []
    for r in all_results:
        for lr in r["learning_records"]:
            if lr["action_type"] == "observe":
                all_observe_nvs.append(lr["net_value"])
            elif lr["action_type"] == "try":
                all_try_nvs.append(lr["net_value"])
    
    agg_audit["net_value_stats"] = {
        "observe_mean_nv": round(sum(all_observe_nvs) / max(len(all_observe_nvs), 1), 4),
        "observe_min_nv": round(min(all_observe_nvs), 4) if all_observe_nvs else 0.0,
        "observe_max_nv": round(max(all_observe_nvs), 4) if all_observe_nvs else 0.0,
        "try_mean_nv": round(sum(all_try_nvs) / max(len(all_try_nvs), 1), 4),
        "try_min_nv": round(min(all_try_nvs), 4) if all_try_nvs else 0.0,
        "try_max_nv": round(max(all_try_nvs), 4) if all_try_nvs else 0.0,
    }
    
    # Decision
    can_proceed = (
        agg_audit["total_learning_records"] > 0
        and agg_audit["observe_learning_records"] > 0
        and agg_audit["try_learning_records"] > 0
        and all_nv_valid
        and all_nv_source
        and agg_proxy == 0
        and not agg_audit["hidden_feature_leakage_detected"]
        and not agg_audit["oracle_leakage_detected"]
        and not agg_audit["policy_decisions_changed"]
    )
    
    agg_audit["can_proceed_to_1J40b1"] = can_proceed
    
    failure_reasons = []
    if agg_audit["total_learning_records"] == 0: failure_reasons.append("no_learning_records")
    if agg_audit["observe_learning_records"] == 0: failure_reasons.append("no_observe_records")
    if agg_audit["try_learning_records"] == 0: failure_reasons.append("no_try_records")
    if not all_nv_valid: failure_reasons.append("some_records_missing_net_value")
    if not all_nv_source: failure_reasons.append("some_records_missing_source_ids")
    if agg_proxy > 0: failure_reasons.append(f"{agg_proxy}_proxy_records")
    if agg_audit["hidden_feature_leakage_detected"]: failure_reasons.append("hidden_feature_leakage")
    if agg_audit["oracle_leakage_detected"]: failure_reasons.append("oracle_leakage")
    agg_audit["failure_reason"] = "; ".join(failure_reasons) if failure_reasons else "none"
    agg_audit["implementation_status"] = "pass" if can_proceed else "fail"
    
    # Write JSON
    json_output = {
        "block_id": "1J40b-0",
        "condition": "C5_mixed_source_identifiability_v0",
        "seeds": MULTISEEDS,
        "elapsed_seconds": total_elapsed,
        "aggregate": agg_audit,
        "per_seed": [],
    }
    
    for r in all_results:
        a = r["audit"]
        json_output["per_seed"].append({
            "seed": r["seed"],
            "elapsed_seconds": r["elapsed_seconds"],
            "total_learning_records": a["total_learning_records"],
            "observe_learning_records": a["observe_learning_records"],
            "try_learning_records": a["try_learning_records"],
            "switch_learning_records": a["switch_learning_records"],
            "switch_learning_available": a["action_event_availability"]["switch_learning_available"],
            "records_with_valid_net_value": a["records_with_valid_net_value"],
            "records_with_source_ids": a["records_with_source_ids"],
            "records_using_soft_explanation_features": a["records_using_soft_explanation_features"],
            "proxy_records_created": a["proxy_records_created"],
            "hidden_feature_leakage_detected": a["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": a["oracle_leakage_detected"],
            "hidden_feature_leakage_in_situation": a["hidden_feature_leakage_in_situation"],
            "can_proceed_to_1J40b1": a["can_proceed_to_1J40b1"],
            "implementation_status": a["implementation_status"],
        })
    
    json_path = os.path.join(CURRENT_DIR, "runs", "block1j40b0_situation_action_result_feasibility_audit.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\n  JSON -> {json_path}")
    
    # Write CSV
    import csv as _csv
    csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b0_situation_action_result_feasibility_audit_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["seed", "total_records", "observe_records", "try_records", "switch_records",
                          "switch_available", "valid_nv", "source_ids", "expl_feat_records",
                          "proxy_records", "hidden_leak", "oracle_leak", "sit_forbidden", "can_proceed"])
        for r in all_results:
            a = r["audit"]
            writer.writerow([
                r["seed"], a["total_learning_records"],
                a["observe_learning_records"], a["try_learning_records"],
                a["switch_learning_records"],
                a["action_event_availability"]["switch_learning_available"],
                a["records_with_valid_net_value"], a["records_with_source_ids"],
                a["records_using_soft_explanation_features"],
                a["proxy_records_created"],
                a["hidden_feature_leakage_detected"], a["oracle_leakage_detected"],
                a["hidden_feature_leakage_in_situation"],
                a["can_proceed_to_1J40b1"],
            ])
    print(f"  CSV  -> {csv_path}")
    
    # Write MD
    md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40b0_situation_action_result_feasibility_audit.md")
    with open(md_path, "w") as f:
        f.write("# Block 1J40b-0: Situation-Action-Result Learning Feasibility Audit\n\n")
        f.write(f"- **Condition**: C5_mixed_source_identifiability_v0\n")
        f.write(f"- **Seeds**: {MULTISEEDS}\n")
        f.write(f"- **Total elapsed**: {total_elapsed}s\n")
        f.write(f"- **Total events processed**: {agg_audit['total_events']}\n")
        f.write(f"- **Total objects**: {agg_audit['total_objects']}\n")
        f.write(f"- **Total OCM candidates**: {agg_audit['total_candidates']}\n\n")
    
        f.write("## Action Event Availability\n\n")
        f.write(f"- observe_event_count: {agg_audit['total_events']} (estimated, see per-seed)\n")
        f.write(f"- switch_learning_available: {any_switch}\n")
        f.write(f"- unique_try_action_types: {agg_audit['unique_try_actions']}\n\n")
    
        f.write("## Per-Seed Results\n\n")
        f.write("| Seed | Total Recs | Observe | Try | Switch | Switch Avail | Valid NV | Source IDs | Expl Feat | Proxy | Hidden Leak | Oracle Leak | Can Proceed |\n")
        f.write("|------|-----------|---------|-----|--------|-------------|----------|------------|-----------|-------|-------------|-------------|-------------|\n")
        for r in all_results:
            a = r["audit"]
            f.write(f"| {r['seed']} | {a['total_learning_records']} | "
                    f"{a['observe_learning_records']} | {a['try_learning_records']} | "
                    f"{a['switch_learning_records']} | "
                    f"{a['action_event_availability']['switch_learning_available']} | "
                    f"{a['records_with_valid_net_value']} | {a['records_with_source_ids']} | "
                    f"{a['records_using_soft_explanation_features']} | "
                    f"{a['proxy_records_created']} | "
                    f"{a['hidden_feature_leakage_detected']} | {a['oracle_leakage_detected']} | "
                    f"{a['can_proceed_to_1J40b1']} |\n")
    
        f.write("\n## Learning Record Details\n\n")
        f.write(f"- total_learning_records: {agg_audit['total_learning_records']}\n")
        f.write(f"- observe_learning_records: {agg_audit['observe_learning_records']}\n")
        f.write(f"- try_learning_records: {agg_audit['try_learning_records']}\n")
        f.write(f"- switch_learning_records: {agg_audit['switch_learning_records']}\n")
        f.write(f"- records_with_valid_net_value: {agg_audit['records_with_valid_net_value']}\n")
        f.write(f"- records_with_source_ids: {agg_audit['records_with_source_ids']}\n")
        f.write(f"- records_using_soft_explanation_features: {agg_audit['records_using_soft_explanation_features']}\n")
        f.write(f"- proxy_records_created: {agg_audit['proxy_records_created']}\n")
        f.write(f"- soft_explanation_features_used: {agg_audit['soft_explanation_features_used']}\n\n")
    
        f.write("## Net Value Statistics\n\n")
        nv = agg_audit["net_value_stats"]
        f.write("### Observe\n")
        f.write(f"- mean: {nv['observe_mean_nv']}, min: {nv['observe_min_nv']}, max: {nv['observe_max_nv']}\n\n")
        f.write("### Try\n")
        f.write(f"- mean: {nv['try_mean_nv']}, min: {nv['try_min_nv']}, max: {nv['try_max_nv']}\n\n")
    
        f.write("## Situation Features Used\n\n")
        for feat, count in agg_audit.get("situation_feature_counts", {}).items():
            f.write(f"- {feat}: {count}\n")
    
        f.write("\n## Audit Checks\n\n")
        checks = [
            ("total_learning_records > 0", agg_audit["total_learning_records"] > 0),
            ("observe_learning_records > 0", agg_audit["observe_learning_records"] > 0),
            ("try_learning_records > 0", agg_audit["try_learning_records"] > 0),
            ("records_with_valid_net_value == total", all_nv_valid),
            ("records_with_source_ids == total", all_nv_source),
            ("proxy_records_created == 0", agg_proxy == 0),
            ("hidden_feature_leakage_detected=false", not agg_audit["hidden_feature_leakage_detected"]),
            ("oracle_leakage_detected=false", not agg_audit["oracle_leakage_detected"]),
            ("policy_decisions_changed=false", not agg_audit["policy_decisions_changed"]),
            ("no_forbidden_in_situation_features", not any_sit_forbidden),
            ("switch_learning_available (optional)", True),
            ("no_seed_crashes", len(all_results) == len(MULTISEEDS)),
        ]
        f.write("| Check | Result |\n")
        f.write("|-------|--------|\n")
        all_pass = True
        for check_name, result in checks:
            status = "PASS" if result else "FAIL"
            if not result: all_pass = False
            f.write(f"| {check_name} | **{status}** |\n")
        f.write(f"| **can_proceed_to_1J40b1** | **{'YES' if can_proceed else 'NO'}** |\n")
        f.write(f"| **overall_audit** | **{'PASS' if all_pass else 'FAIL'}** |\n")
    
        f.write("\n\n```\n[block_done]\n")
        f.write(f"block_id=1J40b-0\n")
        f.write(f"condition=C5_mixed_source_identifiability_v0\n")
        f.write(f"total_learning_records={agg_audit['total_learning_records']}\n")
        f.write(f"observe_learning_records={agg_audit['observe_learning_records']}\n")
        f.write(f"try_learning_records={agg_audit['try_learning_records']}\n")
        f.write(f"switch_learning_records={agg_audit['switch_learning_records']}\n")
        f.write(f"switch_learning_available={str(any_switch).lower()}\n")
        f.write(f"records_with_valid_net_value={agg_audit['records_with_valid_net_value']}\n")
        f.write(f"records_with_source_ids={agg_audit['records_with_source_ids']}\n")
        f.write(f"proxy_records_created={agg_proxy}\n")
        f.write(f"soft_explanation_features_used={str(agg_audit['soft_explanation_features_used']).lower()}\n")
        f.write(f"hidden_feature_leakage_detected={str(agg_audit['hidden_feature_leakage_detected']).lower()}\n")
        f.write(f"oracle_leakage_detected={str(agg_audit['oracle_leakage_detected']).lower()}\n")
        f.write(f"policy_decisions_changed=false\n")
        f.write(f"can_proceed_to_1J40b1={str(can_proceed).lower()}\n")
        f.write(f"implementation_status={'pass' if can_proceed else 'fail'}\n")
        f.write(f"failure_reason={agg_audit['failure_reason']}\n```\n")
    
    print(f"  MD   -> {md_path}")
    print(f"\n{'=' * 70}")
    print(f"Block 1J40b-0 complete.")
    print(f"  total_records: {agg_audit['total_learning_records']}")
    print(f"  observe: {agg_audit['observe_learning_records']}  try: {agg_audit['try_learning_records']}  switch: {agg_audit['switch_learning_records']}")
    print(f"  switch_available: {any_switch}")
    print(f"  proxy: {agg_proxy}  leakage: hidden={agg_audit['hidden_feature_leakage_detected']} oracle={agg_audit['oracle_leakage_detected']}")
    print(f"  can_proceed: {can_proceed}")
    print(f"{'=' * 70}")
