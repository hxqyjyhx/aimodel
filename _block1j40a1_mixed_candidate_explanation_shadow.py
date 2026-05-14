"""
Block 1J40a-1 -- Mixed Candidate Explanation Shadow on C5.

Shadow-only mixed-candidate explanation module. Classifies each mixed
candidate-action case as one of: observation_gap, state_condition,
identity_split, irreducible_noise, unresolved.

Uses C5_mixed_source_identifiability_v0 environment.
Does NOT change policy, split candidates, create state conditions,
or activate Strategy Memory.
"""
import os, sys, json, copy, random, time, math, hashlib
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
_parser.add_argument("--seed", type=int, default=109)
_parser.add_argument("--multiseed", action="store_true", default=False)
_args = _parser.parse_args()

SMOKE_SEED = _args.seed
MULTISEED_RUN = _args.multiseed
MULTISEEDS = [101, 103, 107, 109, 113]
N_EPISODES = 5

# =============================================================================
# Constants
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

# Experience constants
EXP_MIN_SUPPORT_FOR_STABLE = 2
EXP_MAX_CONTRADICTION_FOR_REJECT = 2
EXP_MIN_CONFIDENCE_FOR_REJECT = 0.60

# Explanation thresholds
MIN_CONFIDENCE_FOR_EXPLANATION = 0.15
MIN_SUPPORT_FOR_EXPLANATION = 2
HELDOUT_TRAIN_EPISODES = 4  # episodes 0-3 for train
HELDOUT_TEST_EPISODES = 1   # episode 4 for test

FORBIDDEN_FIELDS = [
    "mixed_source_type", "true_family", "hidden_subtype", "oracle_outcome",
    "full_object_state", "deceptive_flag", "prior_violation",
]

# =============================================================================
# ObjectCandidate + ObjectCandidateMemory (from 1J38)
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
        is_new_object = oid not in self._object_feature_map
        if is_new_object:
            self._total_objects_seen += 1
        if oid not in self._object_feature_map:
            self._object_feature_map[oid] = {}
        self._object_feature_map[oid].update(features_delta)
        if is_new_object:
            for feat in features_delta:
                if features_delta[feat]:
                    self._global_feature_object_count[feat] += 1
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


# =============================================================================
# ExperienceRecord + ExperienceMemory (from 1J38, extended for state tracking)
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
        """Return records with both success and failure outcomes."""
        return [r for r in self._records if r.success_count > 0 and r.failure_count > 0]


# =============================================================================
# C5 Mixed-Source Object Generator (from 1J40a-env0)
# =============================================================================
def generate_c5_objects(rng):
    """Generate 60 test objects with C5 mixed-source types."""
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

        # identity_split
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

        # observation_gap
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

        # state_condition
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

        # irreducible_noise
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


# =============================================================================
# Event Log Builder with C5 features
# =============================================================================
def build_c5_event_log(test_objects, audit_labels, rng):
    """Build event log with repeated observes, state changes, and noise repeats."""
    test_oids = sorted(test_objects.keys())

    # Round-robin episode assignment
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

            # Initial Observe
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

            # Try all 6 core actions
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

            # Repeated Observe for observation_gap
            if ms_type == "observation_gap":
                step_id += 1
                all_events.append({
                    "step_id": step_id, "episode_id": ep, "action_type": "observe",
                    "action_target": oid, "action_params": {"reobserve": True},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": dict(state) if state else {},
                    "success_or_failure": None,
                })

            # State condition: re-observe with changed state + retry
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

            # Irreducible noise: repeat 2 actions under SAME features/state
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
# Mixed Explanation Module (Shadow Only)
# =============================================================================
class MixedExplanationModule:
    """Shadow-only: classifies each mixed candidate-action case into one of 5 types.

    Uses ONLY agent-available evidence:
      - Event log (observed_features_delta, observed_state_delta)
      - OCM (candidates, object-to-candidate mapping)
      - ExperienceMemory (candidate-action outcome records)

    Held-out scoring: train on early episodes, evaluate improvement on later.
    """
    def __init__(self, all_events, ocm, experience_memory, episode_assignments):
        self._events = all_events
        self._ocm = ocm
        self._exp = experience_memory
        self._ep_assignments = episode_assignments
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._explanations = []
        self._candidate_split_performed = False
        self._state_condition_created = False

        # Pre-index events by object
        self._obj_events = defaultdict(list)
        for ev in all_events:
            oid = ev.get("action_target")
            if oid:
                self._obj_events[oid].append(ev)

    def _check_leakage(self, event_record):
        for forbidden in FORBIDDEN_FIELDS:
            if forbidden in event_record:
                self._leakage_checks["hidden_feature_leakage_detected"] = True
                return True
        return False

    def explain_all_mixed(self):
        """Generate explanations for all mixed candidate-action pairs."""
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
        """Score all 5 explanation types for one mixed candidate-action pair."""
        cand_id = candidate.candidate_id
        action = exp_rec.action_affordance
        obj_ids = candidate.object_ids
        source_event_ids = list(exp_rec.source_event_ids)
        source_exp_ids = [exp_rec.experience_id]

        # Gather evidence for each explanation type
        gap_score, gap_evidence = self._score_observation_gap(obj_ids, exp_rec)
        state_score, state_evidence = self._score_state_condition(obj_ids, cand_id, action, exp_rec)
        split_score, split_evidence = self._score_identity_split(candidate, action, exp_rec)
        noise_score, noise_evidence = self._score_irreducible_noise(obj_ids, action, exp_rec)

        # Resolve: highest scorer wins, but must meet minimum threshold
        scores = {
            "observation_gap": gap_score,
            "state_condition": state_score,
            "identity_split": split_score,
            "irreducible_noise": noise_score,
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < MIN_CONFIDENCE_FOR_EXPLANATION or exp_rec.support_count < MIN_SUPPORT_FOR_EXPLANATION:
            predicted = "unresolved"
            confidence = max(0.0, best_score)
        else:
            predicted = best_type
            confidence = round(best_score, 4)

        return {
            "candidate_id": cand_id,
            "action": action,
            "support_count": exp_rec.support_count,
            "success_count": exp_rec.success_count,
            "failure_count": exp_rec.failure_count,
            "gap_score": round(gap_score, 4),
            "state_score": round(state_score, 4),
            "split_score": round(split_score, 4),
            "noise_score": round(noise_score, 4),
            "predicted_explanation": predicted,
            "confidence": confidence,
            "source_event_ids": source_event_ids,
            "source_experience_ids": source_exp_ids,
            "gap_evidence_keys": gap_evidence,
            "state_evidence_keys": state_evidence,
            "split_evidence_keys": split_evidence,
            "noise_evidence_keys": noise_evidence,
        }

    def _score_observation_gap(self, obj_ids, exp_rec):
        """Score observation_gap: features omitted then revealed.

        Evidence: any object in this candidate was observed with different
        feature sets across observations (not False, but actually absent then present).
        Also check if later observations improved prediction accuracy.
        """
        tokens_with_gap = 0
        total_tokens = 0
        heldout_improvement = 0.0

        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            observes = [ev for ev in events if ev.get("action_type") == "observe"]
            total_tokens += 1

            if len(observes) < 2:
                continue

            # Compare features across observes: check for features that appear later
            feat_sets = []
            for obs in observes:
                fd = obs.get("observed_features_delta", {})
                feat_sets.append(set(k for k, v in fd.items() if v))

            new_features_revealed = False
            for i in range(1, len(feat_sets)):
                newly_revealed = feat_sets[i] - feat_sets[i - 1]
                if newly_revealed:
                    new_features_revealed = True
                    break

            if new_features_revealed:
                tokens_with_gap += 1

            # Held-out check: do later-episode predictions improve?
            train_events = [ev for ev in events if ev.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES]
            test_events = [ev for ev in events if ev.get("episode_id", 0) >= HELDOUT_TRAIN_EPISODES]
            if train_events and test_events:
                # Compare prediction using early features vs all features
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

        evidence_keys = {
            "tokens_with_gap": tokens_with_gap,
            "total_tokens": total_tokens,
            "heldout_improvement": round(heldout_improvement, 4),
        }
        return score, evidence_keys

    def _score_state_condition(self, obj_ids, cand_id, action, exp_rec):
        """Score state_condition: visible_state changes across time for same token.

        Evidence: objects in candidate show different visible_state across
        observations, and outcomes vary with state.
        Check if P(outcome|candidate, action, state) improves over P(outcome|candidate, action).
        State MUST come from observed_state_delta, NOT observed_features_delta.
        """
        tokens_with_state_change = 0
        total_tokens = 0

        # Collect outcomes grouped by state
        state_outcomes = defaultdict(list)  # frozenset(state) -> [outcome_bool]
        no_state_outcomes = []

        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            total_tokens += 1

            # Check for state changes across observes
            observes = [ev for ev in events if ev.get("action_type") == "observe"]
            state_sets = []
            for obs in observes:
                sd = obs.get("observed_state_delta", {})
                state_sets.append(frozenset(k for k, v in sd.items() if v))

            if len(set(state_sets)) > 1:
                tokens_with_state_change += 1

            # Group tries by state at try time
            tries = [ev for ev in events if ev.get("action_type") == "try"
                     and ev.get("action_params", {}).get("affordance") == action]
            for ev in tries:
                sd = ev.get("observed_state_delta", {})
                state_key = frozenset(k for k, v in sd.items() if v)
                outcome_bool = ev.get("success_or_failure")
                if state_key:
                    state_outcomes[state_key].append(outcome_bool)
                else:
                    no_state_outcomes.append(outcome_bool)

        # Compute P(outcome|candidate, action, state) improvement
        # Baseline: predict majority outcome from all tries
        all_outcomes = no_state_outcomes + [o for outs in state_outcomes.values() for o in outs]
        if len(all_outcomes) < 2:
            score = 0.0
        else:
            baseline_accuracy = max(
                sum(1 for o in all_outcomes if o is True),
                sum(1 for o in all_outcomes if o is False)
            ) / len(all_outcomes)

            # State-conditioned accuracy
            if state_outcomes:
                state_accs = []
                for sk, outcomes in state_outcomes.items():
                    if len(outcomes) >= 1:
                        acc = max(
                            sum(1 for o in outcomes if o is True),
                            sum(1 for o in outcomes if o is False)
                        ) / len(outcomes)
                        state_accs.append(acc)
                state_accuracy = sum(state_accs) / len(state_accs) if state_accs else 0.0
                improvement = max(0, state_accuracy - baseline_accuracy)
            else:
                improvement = 0.0

            state_rate = tokens_with_state_change / max(total_tokens, 1)
            score = state_rate * 0.5 + improvement * 0.5

        evidence_keys = {
            "tokens_with_state_change": tokens_with_state_change,
            "total_tokens": total_tokens,
            "state_distinct_outcome_sets": len(state_outcomes),
        }
        return score, evidence_keys

    def _score_identity_split(self, candidate, action, exp_rec):
        """Score identity_split: surface-similar subgroups with different affordance profiles.

        Evidence: objects in candidate show different action-outcome patterns
        across MULTIPLE actions. Complexity penalty for single-action splits.

        This does NOT split the candidate — it only scores whether the
        multi-action covariation pattern suggests a split.
        """
        obj_ids = candidate.object_ids
        if len(obj_ids) < 2:
            return 0.0, {"reason": "too_few_objects"}

        # Build per-object action-outcome map
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

        # Get dominant outcome per (oid, action)
        obj_dominant = {}
        for oid, act_outcomes in obj_action_outcomes.items():
            obj_dominant[oid] = {}
            for act, outcomes in act_outcomes.items():
                successes = sum(1 for o in outcomes if o == "success")
                failures = len(outcomes) - successes
                obj_dominant[oid][act] = "success" if successes >= failures else "failure"

        # Find actions where objects disagree
        disagreeing_actions = []
        for act in ALL_ACTIONS:
            outcomes_for_act = {}
            for oid in obj_ids:
                dom = obj_dominant.get(oid, {}).get(act)
                if dom:
                    outcomes_for_act[oid] = dom
            unique_outcomes = set(outcomes_for_act.values())
            if len(unique_outcomes) > 1:
                disagreeing_actions.append(act)

        if not disagreeing_actions:
            return 0.0, {"disagreeing_actions": 0}

        # Score based on number of disagreeing actions (multi-action = stronger)
        n_disagree = len(disagreeing_actions)
        max_possible = min(6, len(ALL_ACTIONS))

        # Held-out: train on early episodes, check if split improves prediction
        heldout_score = self._heldout_split_score(obj_ids, disagreeing_actions)

        # Multi-action bonus
        multi_action_ratio = n_disagree / max_possible
        # Complexity penalty: reduce score if only 1 action disagrees, or if support is weak
        complexity_penalty = 0.5 if n_disagree == 1 else (0.8 if n_disagree == 2 else 1.0)
        # Support penalty: reduce if this is based on few objects
        support_penalty = min(1.0, (len(obj_ids) - 1) / 4.0)  # need ~5+ objects for full score

        score = multi_action_ratio * complexity_penalty * support_penalty * 0.5 + heldout_score * 0.5

        evidence_keys = {
            "disagreeing_actions": n_disagree,
            "object_count": len(obj_ids),
            "heldout_split_score": round(heldout_score, 4),
            "complexity_penalty": round(complexity_penalty, 4),
            "support_penalty": round(support_penalty, 4),
        }
        return score, evidence_keys

    def _heldout_split_score(self, obj_ids, disagreeing_actions):
        """Compute held-out improvement from subgrouping on disagreeing actions."""
        train_objs = []
        test_objs = []
        for oid in obj_ids:
            ep = self._ep_assignments.get(oid, 0)
            if ep < HELDOUT_TRAIN_EPISODES:
                train_objs.append(oid)
            else:
                test_objs.append(oid)

        if not train_objs or not test_objs:
            return 0.0

        # Build outcome profiles on train
        train_profiles = {}
        for oid in train_objs:
            events = self._obj_events.get(oid, [])
            profile = {}
            for ev in events:
                if ev.get("action_type") == "try":
                    act = ev.get("action_params", {}).get("affordance", "")
                    profile[act] = "success" if ev.get("success_or_failure") is True else "failure"
            train_profiles[oid] = profile

        # Find subgroups based on action-outcome agreement on disagreeing actions
        subgroups = self._cluster_by_profile(train_profiles, disagreeing_actions)
        if len(subgroups) < 2:
            return 0.0

        # Baseline: predict majority outcome for each action from all train objects
        baseline_correct = 0
        subgroup_correct = 0
        total_predictions = 0

        for oid in test_objs:
            events = self._obj_events.get(oid, [])
            test_profile = {}
            for ev in events:
                if ev.get("action_type") == "try" and ev.get("action_params", {}).get("affordance") in disagreeing_actions:
                    act = ev.get("action_params", {}).get("affordance", "")
                    outcome = "success" if ev.get("success_or_failure") is True else "failure"
                    test_profile[act] = outcome

            for act in disagreeing_actions:
                if act not in test_profile:
                    continue
                total_predictions += 1
                actual = test_profile[act]

                # Baseline: majority across ALL train objects
                train_outcomes = [train_profiles.get(oid, {}).get(act) for oid in train_objs
                                  if act in train_profiles.get(oid, {})]
                if train_outcomes:
                    from collections import Counter
                    majority = Counter(train_outcomes).most_common(1)[0][0]
                    if majority == actual:
                        baseline_correct += 1

                # Subgroup: find best-matching subgroup by profile similarity
                best_match = self._best_subgroup_for_object(test_profile, subgroups, disagreeing_actions)
                if best_match is not None:
                    sg_outcomes = [train_profiles.get(oid, {}).get(act) for oid in subgroups[best_match]
                                   if act in train_profiles.get(oid, {})]
                    if sg_outcomes:
                        from collections import Counter
                        sg_majority = Counter(sg_outcomes).most_common(1)[0][0]
                        if sg_majority == actual:
                            subgroup_correct += 1

        if total_predictions == 0:
            return 0.0

        baseline_acc = baseline_correct / total_predictions
        subgroup_acc = subgroup_correct / total_predictions
        return max(0.0, subgroup_acc - baseline_acc)

    def _cluster_by_profile(self, profiles, actions):
        """Simple clustering: group objects that agree on all disagreeing actions."""
        subgroups = defaultdict(list)
        for oid, profile in profiles.items():
            key = tuple(profile.get(act, "unknown") for act in sorted(actions))
            subgroups[key].append(oid)
        return dict(subgroups)

    def _best_subgroup_for_object(self, test_profile, subgroups, actions):
        """Find the subgroup whose profile best matches the test object."""
        best_match = None
        best_score = -1
        test_key = tuple(test_profile.get(act) for act in sorted(actions))
        for sg_key, sg_oids in subgroups.items():
            matches = sum(1 for a, b in zip(test_key, sg_key) if a == b)
            if matches > best_score:
                best_score = matches
                best_match = sg_key
        return best_match

    def _score_irreducible_noise(self, obj_ids, action, exp_rec):
        """Score irreducible_noise: same (oid, action) gives different outcomes under same features/state.

        Evidence: at least one object in this candidate has repeated tries of
        the same action with SAME features and state but different outcomes.
        Also check that other explanations don't account for the variance.
        """
        tokens_with_inconsistency = 0
        total_tokens = 0
        total_inconsistent_pairs = 0

        for oid in obj_ids:
            events = self._obj_events.get(oid, [])
            total_tokens += 1

            # Find tries of this action
            action_tries = [ev for ev in events if ev.get("action_type") == "try"
                            and ev.get("action_params", {}).get("affordance") == action]
            if len(action_tries) < 2:
                continue

            # Group by (features_key, state_key)
            grouped = defaultdict(list)
            for ev in action_tries:
                fd = ev.get("observed_features_delta", {})
                sd = ev.get("observed_state_delta", {})
                fkey = frozenset(k for k, v in fd.items() if v)
                skey = frozenset(k for k, v in sd.items() if v)
                grouped[(fkey, skey)].append(ev.get("success_or_failure"))

            # Check for inconsistency within same (fkey, skey)
            for (fkey, skey), outcomes in grouped.items():
                if len(outcomes) >= 2:
                    unique_outcomes = set(outcomes)
                    if len(unique_outcomes) > 1:
                        tokens_with_inconsistency += 1
                        total_inconsistent_pairs += 1
                        break  # one inconsistent pair per token is enough

        score = 0.0
        if total_tokens > 0:
            inconsistency_rate = tokens_with_inconsistency / total_tokens
            # Bonus if other explanations are weak
            score = inconsistency_rate * 0.7  # base from inconsistency rate
            # The remaining 0.3 would come from lack of other explanations,
            # but we let the argmax handle that naturally

        evidence_keys = {
            "tokens_with_inconsistency": tokens_with_inconsistency,
            "total_tokens": total_tokens,
            "inconsistent_pairs": total_inconsistent_pairs,
        }
        return score, evidence_keys


# =============================================================================
# Main runner for a single seed
# =============================================================================
def run_seed(seed):
    t_start = time.time()

    # Get C5 condition
    c5_label = "C5_mixed_source_identifiability_v0"
    c5_cond = None
    for c in config.CUE_CONDITIONS:
        if c["label"] == c5_label:
            c5_cond = copy.deepcopy(c)
            break
    if c5_cond is None:
        raise RuntimeError(f"{c5_label} not found in CUE_CONDITIONS")

    # Phase A training
    student, base_learner, train_objects, train_env, _std_test, _std_test_env, final_metrics, rng = \
        run_phase_a_training(seed, c5_cond)

    # Generate C5 test objects
    obj_rng = random.Random(rng.randint(0, 2**31 - 1))
    test_objects, audit_labels = generate_c5_objects(obj_rng)

    # Build event log
    event_rng = random.Random(rng.randint(0, 2**31 - 1))
    all_events, episode_assignments = build_c5_event_log(test_objects, audit_labels, event_rng)

    # Process events through OCM + ExperienceMemory
    ocm = ObjectCandidateMemory()
    exp_mem = ExperienceMemory()

    # Process observe events first for leakage check, then process
    for ev in all_events:
        # Leakage checks on raw events
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

    # Run explanation module
    explainer = MixedExplanationModule(all_events, ocm, exp_mem, episode_assignments)
    explanations = explainer.explain_all_mixed()

    # Evaluate against audit labels
    confusion = defaultdict(lambda: defaultdict(int))
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    total_matched = 0

    for exp in explanations:
        cand_id = exp["candidate_id"]
        action = exp["action"]
        predicted = exp["predicted_explanation"]

        # Find audit label: get the mixed_source_type for the majority of objects in this candidate
        cand = ocm.get_candidate_by_id(cand_id)
        if cand is None:
            continue

        obj_types = []
        for oid in cand.object_ids:
            if oid in audit_labels:
                obj_types.append(audit_labels[oid]["mixed_source_type"])

        if not obj_types:
            continue

        from collections import Counter
        audit_type = Counter(obj_types).most_common(1)[0][0]

        confusion[audit_type][predicted] += 1
        per_class_total[audit_type] += 1
        if predicted == audit_type:
            per_class_correct[audit_type] += 1
        total_matched += 1

    # Compute metrics
    all_types = ["observation_gap", "state_condition", "identity_split", "irreducible_noise"]
    if total_matched > 0:
        macro_accuracy = sum(per_class_correct.get(t, 0) / max(per_class_total.get(t, 1), 1)
                            for t in all_types) / len(all_types)
    else:
        macro_accuracy = 0.0

    per_class_recall = {}
    for t in all_types:
        per_class_recall[t] = round(per_class_correct.get(t, 0) / max(per_class_total.get(t, 1), 1), 4)

    # Summary stats
    total_mixed_cases = len(explanations)
    pred_counts = defaultdict(int)
    total_conf = 0.0
    low_conf_count = 0
    for exp in explanations:
        pred_counts[exp["predicted_explanation"]] += 1
        total_conf += exp["confidence"]
        if exp["confidence"] < MIN_CONFIDENCE_FOR_EXPLANATION:
            low_conf_count += 1

    avg_conf = total_conf / max(len(explanations), 1)
    unresolved_rate = pred_counts.get("unresolved", 0) / max(total_mixed_cases, 1)

    # Over-split rate: identity_split predicted for non-identity_split audit types
    split_pred_total = pred_counts.get("identity_split", 0)
    split_wrong = sum(confusion[at].get("identity_split", 0) for at in all_types if at != "identity_split")
    over_split_rate = split_wrong / max(split_pred_total, 1) if split_pred_total > 0 else 0.0

    # Noise overfit: irreducible_noise predicted for non-noise audit types
    noise_pred_total = pred_counts.get("irreducible_noise", 0)
    noise_wrong = sum(confusion[at].get("irreducible_noise", 0) for at in all_types if at != "irreducible_noise")
    noise_overfit_rate = noise_wrong / max(noise_pred_total, 1) if noise_pred_total > 0 else 0.0

    # Source ID check
    all_have_source_ids = all(
        len(exp.get("source_event_ids", [])) > 0 or len(exp.get("source_experience_ids", [])) > 0
        for exp in explanations
    )

    # Heldout check: are there events in both train and test episodes?
    train_eps_present = set()
    test_eps_present = set()
    for ev in all_events:
        if ev.get("episode_id", 0) < HELDOUT_TRAIN_EPISODES:
            train_eps_present.add(ev.get("episode_id"))
        else:
            test_eps_present.add(ev.get("episode_id"))
    heldout_prediction_available = len(train_eps_present) > 0 and len(test_eps_present) > 0

    elapsed = round(time.time() - t_start, 2)

    return {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "total_mixed_cases": total_mixed_cases,
        "predicted_counts": dict(pred_counts),
        "confusion_matrix": {at: dict(preds) for at, preds in confusion.items()},
        "macro_accuracy": round(macro_accuracy, 4),
        "per_class_recall": per_class_recall,
        "unresolved_rate": round(unresolved_rate, 4),
        "over_split_rate": round(over_split_rate, 4),
        "noise_overfit_rate": round(noise_overfit_rate, 4),
        "average_confidence": round(avg_conf, 4),
        "low_confidence_count": low_conf_count,
        "heldout_prediction_available": heldout_prediction_available,
        "all_records_have_source_ids": all_have_source_ids,
        "hidden_feature_leakage_detected": ocm._leakage_checks["hidden_feature_leakage_detected"] or exp_mem._leakage_checks["hidden_feature_leakage_detected"] or explainer._leakage_checks["hidden_feature_leakage_detected"],
        "oracle_leakage_detected": ocm._leakage_checks["oracle_leakage_detected"] or exp_mem._leakage_checks["oracle_leakage_detected"] or explainer._leakage_checks["oracle_leakage_detected"],
        "policy_decisions_changed": False,
        "candidate_split_performed": explainer._candidate_split_performed,
        "state_condition_created": explainer._state_condition_created,
        "explanations": explanations,
        "candidate_summary": {
            "total_candidates": len(ocm._candidates),
            "buffer_candidates": len(ocm._candidate_buffer),
            "promoted": len(ocm._promotion_log),
        },
    }


# =============================================================================
# Run
# =============================================================================
if MULTISEED_RUN:
    seeds_to_run = MULTISEEDS
else:
    seeds_to_run = [SMOKE_SEED]

print("=" * 70)
print("Block 1J40a-1 -- Mixed Candidate Explanation Shadow")
print(f"  seeds={seeds_to_run}  episodes={N_EPISODES}")
print(f"  condition=C5_mixed_source_identifiability_v0")
print("=" * 70)

all_results = []
for seed in seeds_to_run:
    print(f"\n--- Seed {seed} ---")
    result = run_seed(seed)
    all_results.append(result)
    print(f"  mixed_cases={result['total_mixed_cases']}  macro_acc={result['macro_accuracy']}")
    print(f"  preds: gap={result['predicted_counts'].get('observation_gap', 0)} "
          f"state={result['predicted_counts'].get('state_condition', 0)} "
          f"split={result['predicted_counts'].get('identity_split', 0)} "
          f"noise={result['predicted_counts'].get('irreducible_noise', 0)} "
          f"unresolved={result['predicted_counts'].get('unresolved', 0)}")
    print(f"  leakage: hidden={result['hidden_feature_leakage_detected']} "
          f"oracle={result['oracle_leakage_detected']} "
          f"split_performed={result['candidate_split_performed']} "
          f"state_created={result['state_condition_created']}")
    print(f"  over_split={result['over_split_rate']}  noise_overfit={result['noise_overfit_rate']}")

# =============================================================================
# Write Outputs
# =============================================================================
seed_list_str = ",".join(str(s) for s in seeds_to_run)
total_elapsed = round(sum(r["elapsed_seconds"] for r in all_results), 1)

# Aggregate
agg_total_mixed = sum(r["total_mixed_cases"] for r in all_results)
agg_pred_counts = defaultdict(int)
agg_confusion = defaultdict(lambda: defaultdict(int))
all_leakage_hidden = False
all_leakage_oracle = False
all_have_source_ids = True
all_heldout = True
any_split = False
any_state_created = False

for r in all_results:
    for k, v in r["predicted_counts"].items():
        agg_pred_counts[k] += v
    for at, preds in r["confusion_matrix"].items():
        for pred, cnt in preds.items():
            agg_confusion[at][pred] += cnt
    if r["hidden_feature_leakage_detected"]:
        all_leakage_hidden = True
    if r["oracle_leakage_detected"]:
        all_leakage_oracle = True
    if not r["all_records_have_source_ids"]:
        all_have_source_ids = False
    if not r["heldout_prediction_available"]:
        all_heldout = False
    if r["candidate_split_performed"]:
        any_split = True
    if r["state_condition_created"]:
        any_state_created = True

# Aggregate accuracy
all_types_list = ["observation_gap", "state_condition", "identity_split", "irreducible_noise"]
agg_per_class_correct = defaultdict(int)
agg_per_class_total = defaultdict(int)
for at in all_types_list:
    for pred in all_types_list:
        cnt = agg_confusion.get(at, {}).get(pred, 0)
        agg_per_class_total[at] += cnt
        if at == pred:
            agg_per_class_correct[at] += cnt

if sum(agg_per_class_total.values()) > 0:
    agg_macro_accuracy = sum(
        agg_per_class_correct.get(t, 0) / max(agg_per_class_total.get(t, 1), 1)
        for t in all_types_list
    ) / len(all_types_list)
else:
    agg_macro_accuracy = 0.0

# Over-split / noise-overfit aggregates
agg_split_pred = agg_pred_counts.get("identity_split", 0)
agg_split_wrong = sum(agg_confusion.get(at, {}).get("identity_split", 0) for at in all_types_list if at != "identity_split")
agg_over_split = agg_split_wrong / max(agg_split_pred, 1) if agg_split_pred > 0 else 0.0

agg_noise_pred = agg_pred_counts.get("irreducible_noise", 0)
agg_noise_wrong = sum(agg_confusion.get(at, {}).get("irreducible_noise", 0) for at in all_types_list if at != "irreducible_noise")
agg_noise_overfit = agg_noise_wrong / max(agg_noise_pred, 1) if agg_noise_pred > 0 else 0.0

# Acceptance
agg_unresolved = agg_pred_counts.get("unresolved", 0) / max(agg_total_mixed, 1)
acceptance = (
    not all_leakage_hidden and
    not all_leakage_oracle and
    not any_split and
    not any_state_created and
    all_have_source_ids and
    all_heldout
)

json_output = {
    "block_id": "1J40a-1",
    "condition": "C5_mixed_source_identifiability_v0",
    "seeds": seeds_to_run,
    "elapsed_seconds": total_elapsed,
    "aggregate": {
        "total_mixed_cases": agg_total_mixed,
        "predicted_counts": dict(agg_pred_counts),
        "confusion_matrix": {at: dict(preds) for at, preds in agg_confusion.items()},
        "macro_accuracy": round(agg_macro_accuracy, 4),
        "per_class_recall": {
            t: round(agg_per_class_correct.get(t, 0) / max(agg_per_class_total.get(t, 1), 1), 4)
            for t in all_types_list
        },
        "unresolved_rate": round(agg_unresolved, 4),
        "over_split_rate": round(agg_over_split, 4),
        "noise_overfit_rate": round(agg_noise_overfit, 4),
        "heldout_prediction_available": all_heldout,
        "all_records_have_source_ids": all_have_source_ids,
        "hidden_feature_leakage_detected_any": all_leakage_hidden,
        "oracle_leakage_detected_any": all_leakage_oracle,
        "policy_decisions_changed": False,
        "candidate_split_performed": any_split,
        "state_condition_created": any_state_created,
        "implementation_status": "pass" if acceptance else "partial",
        "failure_reason": "none",  # computed below
    },
    "per_seed": [],
}

# Build failure reason
failure_parts = []
if all_leakage_hidden: failure_parts.append("hidden_feature_leakage_detected")
if all_leakage_oracle: failure_parts.append("oracle_leakage_detected")
if any_split: failure_parts.append("candidate_split_performed")
if any_state_created: failure_parts.append("state_condition_created")
if not all_have_source_ids: failure_parts.append("missing_source_ids")
if not all_heldout: failure_parts.append("heldout_not_available")
json_output["aggregate"]["failure_reason"] = "; ".join(failure_parts) if failure_parts else "none"

for r in all_results:
    json_output["per_seed"].append({
        "seed": r["seed"],
        "total_mixed_cases": r["total_mixed_cases"],
        "predicted_counts": r["predicted_counts"],
        "macro_accuracy": r["macro_accuracy"],
        "per_class_recall": r["per_class_recall"],
        "unresolved_rate": r["unresolved_rate"],
        "over_split_rate": r["over_split_rate"],
        "noise_overfit_rate": r["noise_overfit_rate"],
        "heldout_prediction_available": r["heldout_prediction_available"],
        "all_records_have_source_ids": r["all_records_have_source_ids"],
        "hidden_feature_leakage_detected": r["hidden_feature_leakage_detected"],
        "oracle_leakage_detected": r["oracle_leakage_detected"],
        "candidate_split_performed": r["candidate_split_performed"],
        "state_condition_created": r["state_condition_created"],
    })

# Write JSON
json_path = os.path.join(CURRENT_DIR, "runs", "block1j40a1_mixed_candidate_explanation_shadow.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"\n  JSON -> {json_path}")

# Write CSV
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j40a1_mixed_candidate_explanation_shadow_table.csv")
with open(csv_path, "w", newline="") as f:
    import csv as _csv
    writer = _csv.writer(f)
    writer.writerow(["seed", "total_mixed", "gap_pred", "state_pred", "split_pred", "noise_pred",
                      "unresolved_pred", "macro_accuracy", "over_split_rate", "noise_overfit_rate",
                      "unresolved_rate", "heldout_available", "all_source_ids", "hidden_leak", "oracle_leak"])
    for r in all_results:
        writer.writerow([
            r["seed"], r["total_mixed_cases"],
            r["predicted_counts"].get("observation_gap", 0),
            r["predicted_counts"].get("state_condition", 0),
            r["predicted_counts"].get("identity_split", 0),
            r["predicted_counts"].get("irreducible_noise", 0),
            r["predicted_counts"].get("unresolved", 0),
            r["macro_accuracy"], r["over_split_rate"], r["noise_overfit_rate"],
            r["unresolved_rate"], r["heldout_prediction_available"],
            r["all_records_have_source_ids"], r["hidden_feature_leakage_detected"],
            r["oracle_leakage_detected"],
        ])
print(f"  CSV  -> {csv_path}")

# Write MD
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j40a1_mixed_candidate_explanation_shadow.md")
with open(md_path, "w") as f:
    f.write("# Block 1J40a-1: Mixed Candidate Explanation Shadow\n\n")
    f.write(f"- **Condition**: C5_mixed_source_identifiability_v0\n")
    f.write(f"- **Seeds**: {seeds_to_run}\n")
    f.write(f"- **Total elapsed**: {total_elapsed}s\n\n")

    f.write("## Per-Seed Results\n\n")
    f.write("| Seed | Total Mixed | Gap | State | Split | Noise | Unresolved | Macro Acc | OverSplit | NoiseFit |\n")
    f.write("|------|-------------|-----|-------|-------|-------|------------|-----------|-----------|----------|\n")
    for r in all_results:
        f.write(f"| {r['seed']} | {r['total_mixed_cases']} | "
                f"{r['predicted_counts'].get('observation_gap', 0)} | "
                f"{r['predicted_counts'].get('state_condition', 0)} | "
                f"{r['predicted_counts'].get('identity_split', 0)} | "
                f"{r['predicted_counts'].get('irreducible_noise', 0)} | "
                f"{r['predicted_counts'].get('unresolved', 0)} | "
                f"{r['macro_accuracy']} | {r['over_split_rate']} | {r['noise_overfit_rate']} |\n")

    f.write("\n## Aggregate Confusion Matrix\n\n")
    f.write("| Audit \\ Predicted | gap | state | split | noise | unresolved |\n")
    f.write("|--------------------|-----|-------|-------|-------|------------|\n")
    for at in all_types_list:
        row = agg_confusion.get(at, {})
        f.write(f"| {at} | {row.get('observation_gap', 0)} | {row.get('state_condition', 0)} | "
                f"{row.get('identity_split', 0)} | {row.get('irreducible_noise', 0)} | "
                f"{row.get('unresolved', 0)} |\n")

    f.write("\n## Per-Class Recall\n\n")
    for t in all_types_list:
        recall = agg_per_class_correct.get(t, 0) / max(agg_per_class_total.get(t, 1), 1)
        f.write(f"- **{t}**: {round(recall, 4)} ({agg_per_class_correct.get(t, 0)}/{agg_per_class_total.get(t, 0)})\n")

    f.write("\n## Aggregate Metrics\n\n")
    f.write(f"- total_mixed_cases: {agg_total_mixed}\n")
    f.write(f"- observation_gap_pred_count: {agg_pred_counts.get('observation_gap', 0)}\n")
    f.write(f"- state_condition_pred_count: {agg_pred_counts.get('state_condition', 0)}\n")
    f.write(f"- identity_split_pred_count: {agg_pred_counts.get('identity_split', 0)}\n")
    f.write(f"- irreducible_noise_pred_count: {agg_pred_counts.get('irreducible_noise', 0)}\n")
    f.write(f"- unresolved_pred_count: {agg_pred_counts.get('unresolved', 0)}\n")
    f.write(f"- macro_accuracy: {round(agg_macro_accuracy, 4)}\n")
    f.write(f"- over_split_rate: {round(agg_over_split, 4)}\n")
    f.write(f"- noise_overfit_rate: {round(agg_noise_overfit, 4)}\n")
    f.write(f"- unresolved_rate: {round(agg_unresolved, 4)}\n")
    f.write(f"- heldout_prediction_available: {all_heldout}\n")
    f.write(f"- all_records_have_source_ids: {all_have_source_ids}\n\n")

    f.write("## Acceptance Checks\n\n")
    checks = [
        ("policy_decisions_changed=false", True),
        ("candidate_split_performed=false", not any_split),
        ("state_condition_created=false", not any_state_created),
        ("hidden_feature_leakage_detected_any=false", not all_leakage_hidden),
        ("oracle_leakage_detected_any=false", not all_leakage_oracle),
        ("all_records_have_source_ids=true", all_have_source_ids),
        ("heldout_prediction_available=true", all_heldout),
        ("unresolved_allowed", True),
        ("no_seed_crashes", len(all_results) == len(seeds_to_run)),
    ]
    f.write("| Check | Result |\n")
    f.write("|-------|--------|\n")
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        f.write(f"| {check_name} | **{status}** |\n")
    all_accept = all(r for _, r in checks)
    f.write(f"| **overall_acceptance** | **{'PASS' if all_accept else 'FAIL'}** |\n")

    f.write("\n\n```\n[block_done]\n")
    f.write(f"block_id=1J40a-1\n")
    f.write(f"condition=C5_mixed_source_identifiability_v0\n")
    f.write(f"seeds={','.join(str(s) for s in seeds_to_run)}\n")
    f.write(f"total_mixed_cases={agg_total_mixed}\n")
    f.write(f"observation_gap_pred_count={agg_pred_counts.get('observation_gap', 0)}\n")
    f.write(f"state_condition_pred_count={agg_pred_counts.get('state_condition', 0)}\n")
    f.write(f"identity_split_pred_count={agg_pred_counts.get('identity_split', 0)}\n")
    f.write(f"irreducible_noise_pred_count={agg_pred_counts.get('irreducible_noise', 0)}\n")
    f.write(f"unresolved_pred_count={agg_pred_counts.get('unresolved', 0)}\n")
    f.write(f"macro_accuracy={round(agg_macro_accuracy, 4)}\n")
    f.write(f"over_split_rate={round(agg_over_split, 4)}\n")
    f.write(f"noise_overfit_rate={round(agg_noise_overfit, 4)}\n")
    f.write(f"heldout_prediction_available={str(all_heldout).lower()}\n")
    f.write(f"all_records_have_source_ids={str(all_have_source_ids).lower()}\n")
    f.write(f"hidden_feature_leakage_detected_any={str(all_leakage_hidden).lower()}\n")
    f.write(f"oracle_leakage_detected_any={str(all_leakage_oracle).lower()}\n")
    f.write(f"policy_decisions_changed=false\n")
    f.write(f"candidate_split_performed={str(any_split).lower()}\n")
    f.write(f"state_condition_created={str(any_state_created).lower()}\n")
    f.write(f"implementation_status={'pass' if acceptance else 'partial'}\n")
    f.write(f"failure_reason={'none' if acceptance else '; '.join(failure_parts)}\n```\n")

print(f"  MD   -> {md_path}")
print(f"\n{'=' * 70}")
print(f"Block 1J40a-1 complete.")
print(f"  acceptance: {'PASS' if acceptance else 'PARTIAL/FAIL'}")
print(f"  seeds_run: {len(all_results)}/{len(seeds_to_run)}")
print(f"{'=' * 70}")
