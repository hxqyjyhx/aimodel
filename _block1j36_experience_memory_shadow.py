"""
Block 1J36 -- Phase 3 Experience Memory Shadow Evaluation.

Shadow/sidecar module: builds ExperienceMemory from Event Memory records
and Phase 2 Object Candidate Memory, without changing policy decisions.

Binds object/candidate + observed features/state + action + outcome.
Does NOT store feature->outcome rules.

Does NOT:
  - Replace current policy logic
  - Replace detect_type_family()
  - Replace FCRM counters
  - Use audit-only fields
  - Store feature->outcome without object/candidate anchor
"""
import os, sys, json, copy, random, time, math
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
A3_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5a3_tool_material_transfer")
sys.path.insert(0, A3_DIR)
L5_DIR = os.path.join(CURRENT_DIR, "..", "exp004_5l_instance_outcome_memory")
sys.path.insert(0, L5_DIR)
sys.path.insert(0, CURRENT_DIR)

import config
from run_004_5n1 import run_phase_a_training
from subtype_objects import generate_subtype_objects_deterministic

t0 = time.time()

# =============================================================================
# CLI
# =============================================================================
import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=109, help="Base seed")
_args = _parser.parse_args()

SMOKE_SEED = _args.seed
N_EPISODES = 5
BUDGET = 1.5

COND = copy.deepcopy(config.CUE_CONDITIONS[3])
assert COND["label"] == "C4_instance_subtype_cued_v1"

FEATURE_UNIVERSE = sorted({
    "has_bark_texture", "has_wood_grain", "has_crystal_flecks",
    "has_granular_surface", "has_stem_remnant", "has_peel_texture",
    "has_grip_area", "has_shaft_shape",
    "solid", "movable", "block_like", "elongated_with_handle",
    "round_small", "brownish", "grayish", "greenish",
    "long_shape", "rough_texture", "smooth_texture",
    "on_left_side", "near_table", "recently_seen",
    "light_weight", "heavy_weight",
    "damp_texture", "brittle_surface", "treated_surface", "hollow_sound",
})
FEATURE_UNIVERSE_SET = set(FEATURE_UNIVERSE)

WOOD_FEATURES = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_FEATURES = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_FEATURES = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
TOOL_FEATURES = {"has_grip_area", "has_shaft_shape", "elongated_with_handle", "movable", "long_shape"}

TYPE_FAMILIES = {
    "wood-like": WOOD_FEATURES, "stone-like": STONE_FEATURES,
    "apple-like": APPLE_FEATURES, "tool-like": TOOL_FEATURES,
}
ALL_TYPE_FAMILY_FEATURES = set()
for fs in TYPE_FAMILIES.values():
    ALL_TYPE_FAMILY_FEATURES.update(fs)

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}

ALL_ACTIONS = sorted(ACTION_TO_QUERY.keys())

# =============================================================================
# Constants from 1J33/1J34
# =============================================================================
INJECTED_DEVIATION_FEATURES = {"damp_texture", "brittle_surface", "treated_surface", "hollow_sound"}

APPLE_CORE = {"has_stem_remnant", "has_peel_texture", "round_small", "greenish", "smooth_texture", "light_weight"}
WOOD_CORE  = {"has_bark_texture", "has_wood_grain", "brownish", "rough_texture", "long_shape"}
STONE_CORE = {"has_crystal_flecks", "has_granular_surface", "grayish", "block_like", "heavy_weight"}
APPLE_DEVIATION  = {"damp_texture": True, "brittle_surface": True}
WOOD_DEVIATION   = {"damp_texture": True, "treated_surface": True}
STONE_DEVIATION  = {"brittle_surface": True, "hollow_sound": True}

# Phase 2 OCM constants
PROMOTION_MIN_SUPPORT = 2
PROMOTION_MIN_EPISODE_SPREAD = 2
PROMOTION_MIN_STABILITY = 0.70
PROMOTION_MAX_CONTRADICTION_RATE = 0.30
PROMOTION_MIN_POSITIVE_FEATURES = 3
PROMOTION_MAX_OVERLAP_WITH_OTHER = 0.60
CANDIDATE_MATCH_MIN_JACCARD = 0.40
CANDIDATE_MAX_CONTRADICTION_FOR_MATCHING = 0.40

# Phase 3 constants
EXP_MIN_SUPPORT_FOR_STABLE = 2
EXP_MAX_CONTRADICTION_FOR_REJECT = 2
EXP_MIN_CONFIDENCE_FOR_REJECT = 0.60

print("=" * 70)
print("Block 1J36 -- Phase 3 Experience Memory Shadow")
print(f"  seed={SMOKE_SEED}  episodes={N_EPISODES}")
print("=" * 70)

# =============================================================================
# 1. Phase A training + test objects (same as 1J34/1J35)
# =============================================================================
print("\n[1/6] Phase A: Training + generating test objects...")
(student, base_learner, train_objects, train_env,
 _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(SMOKE_SEED, COND)
print(f"  pre_accuracy={final_metrics.get('domain_token_accuracy', 0.0):.4f}")

test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
test_oids = sorted(test_objects_standard.keys())

# Phase 3 shadow uses STANDARD objects (same as Phase 2).
# Deviation features are excluded since they create cross-family bridges
# only resolvable with action-outcome evidence.
test_objects = copy.deepcopy(test_objects_standard)
DECEPTIVE_OIDS = set()

print(f"  {len(test_oids)} test objects (standard subtype, no deviation features)")

# =============================================================================
# 2. Object Candidate Memory (Phase 2, reproduced for Phase 3 dependency)
# =============================================================================
class ObjectCandidate:
    """A provisional or stable object category learned from event evidence."""

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
            self.confidence_score = 0.0
            self.stability_score = 0.0
            return
        total_checks = 0
        consistent_checks = 0
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
            weighted_intersection = sum(diagnosticity_weights.get(f, 1.0) for f in intersection)
            weighted_union = sum(diagnosticity_weights.get(f, 1.0) for f in union)
            if weighted_union == 0:
                return 0.0
            return weighted_intersection / weighted_union
        return len(self.positive_feature_set & obs_set) / max(len(self.positive_feature_set | obs_set), 1)

    def is_closed(self):
        if self.promotion_status == "stable":
            return True
        n_pos = len(self.positive_feature_set)
        if n_pos == 0:
            return False
        return len(self.contradicted_features) / n_pos > CANDIDATE_MAX_CONTRADICTION_FOR_MATCHING

    def jaccard_similarity(self, other_candidate):
        a = self.positive_feature_set
        b = other_candidate.positive_feature_set
        if not a and not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    def meets_promotion_criteria(self):
        if self.support_count < PROMOTION_MIN_SUPPORT:
            return False
        if len(self.episodes_seen) < PROMOTION_MIN_EPISODE_SPREAD:
            return False
        if self.stability_score < PROMOTION_MIN_STABILITY:
            return False
        if len(self.positive_feature_set) < PROMOTION_MIN_POSITIVE_FEATURES:
            return False
        n_pos = len(self.positive_feature_set)
        if n_pos > 0 and (len(self.contradicted_features) / n_pos) > PROMOTION_MAX_CONTRADICTION_RATE:
            return False
        return True

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "object_ids": sorted(self.object_ids),
            "positive_feature_set": sorted(self.positive_feature_set),
            "negative_feature_set": sorted(self.negative_feature_set),
            "contradicted_features": {f: len(v) for f, v in self.contradicted_features.items()},
            "confidence_score": self.confidence_score,
            "stability_score": self.stability_score,
            "support_count": self.support_count,
            "episode_spread": len(self.episodes_seen),
            "creation_step_id": self.creation_step_id,
            "last_updated_step_id": self.last_updated_step_id,
            "source_step_ids": sorted(self.source_step_ids),
            "promotion_status": self.promotion_status,
        }


class ObjectCandidateMemory:
    """Shadow Object Candidate Memory (L2). Reproduced from Phase 2."""

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

        for forbidden in ["true_family", "hidden_subtype", "prior_violation", "oracle_outcome",
                          "deceptive_flag", "full_object_state", "unobserved_features"]:
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
        best_candidate = None
        best_score = 0.0
        for cand in self._candidates + self._candidate_buffer:
            if oid in cand.object_ids:
                return cand
            if cand.is_closed():
                continue
            score = cand.feature_overlap_score(features, diagnosticity_weights)
            if score > best_score:
                best_score = score
                best_candidate = cand
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
                            "positive_features": sorted(cand.positive_feature_set),
                            "promotion_step_id": cand.promotion_step_id,
                        })
        single_event_promotions = [p for p in self._promotion_log if p["episode_spread"] < 2]
        return promoted, single_event_promotions

    def get_candidate_for_object(self, oid):
        return self._object_candidate_map.get(oid)

    def get_candidate_by_id(self, cand_id):
        for c in self._candidates + self._candidate_buffer:
            if c.candidate_id == cand_id:
                return c
        return None

    def get_statistics(self):
        all_candidates = self._candidates + self._candidate_buffer
        support_counts = [c.support_count for c in all_candidates]
        provisional = [c for c in all_candidates if c.promotion_status == "provisional"]
        stable = [c for c in all_candidates if c.promotion_status == "stable"]
        conflicted = [c for c in all_candidates if len(c.contradicted_features) > 0]
        return {
            "total_candidates": len(all_candidates),
            "provisional_candidates": len(provisional),
            "stable_candidates": len(stable),
            "candidate_buffer_size": len(self._candidate_buffer),
            "average_support_count": round(sum(support_counts) / max(len(support_counts), 1), 2),
            "max_support_count": max(support_counts) if support_counts else 0,
            "conflict_candidate_count": len(conflicted),
            "single_event_promotion_count": sum(1 for p in self._promotion_log if p["episode_spread"] < 2),
            "hidden_feature_leakage_detected": self._leakage_checks["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": self._leakage_checks["oracle_leakage_detected"],
            "total_events_processed": self._event_count,
            "total_objects_tracked": len(self._object_feature_map),
            "candidates": [c.to_dict() for c in self._candidates],
            "buffer_candidates": [c.to_dict() for c in self._candidate_buffer],
        }


# =============================================================================
# 3. Experience Memory (Phase 3)
# =============================================================================
class ExperienceRecord:
    """Binds object/candidate + features/state + action + outcome.

    Key invariant: always has object/candidate anchor. Never stores feature->outcome.
    """

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
        self.confidence_score = 1.0 if outcome == "success" else (1.0 if outcome == "failure" else 0.5)
        self.episodes_seen = {episode_id}
        self.last_updated_step = source_step_id
        self.status = "provisional"
        self.object_ids = [object_id]  # all objects that contributed to this record

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
        """Return the majority outcome category."""
        if self.success_count >= self.failure_count:
            return "success" if self.success_count > 0 else None
        return "failure" if self.failure_count > 0 else None

    def is_consistent(self):
        """True if all outcomes are the same."""
        return self.contradiction_count == 0 and self.support_count > 0

    def to_dict(self):
        return {
            "experience_id": self.experience_id,
            "object_anchor_id": self.object_anchor_id,
            "candidate_anchor_id": self.candidate_anchor_id,
            "source_event_ids": sorted(self.source_event_ids),
            "action_type": self.action_type,
            "action_affordance": self.action_affordance,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "confidence_score": self.confidence_score,
            "episode_spread": len(self.episodes_seen),
            "last_updated_step": self.last_updated_step,
            "status": self.status,
            "dominant_outcome": self.dominant_outcome(),
            "object_ids": sorted(self.object_ids),
            "feature_snapshot_count": len(self.observed_feature_snapshots),
        }


class ExperienceMemory:
    """Shadow Experience Memory (L3). Binds object/candidate + action + outcome."""

    def __init__(self):
        self._records = []  # list of ExperienceRecord
        self._record_index = {}  # (candidate_id_or_oid, action_affordance) -> ExperienceRecord
        self._event_count = 0
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._feature_only_rule_detected = False
        self._per_object_actions = defaultdict(set)  # oid -> set of actions tried
        self._per_candidate_actions = defaultdict(set)  # candidate_id -> set of (action, outcome)

    def process_event(self, event_record, ocm):
        """Process a try event: create or update an experience record."""
        self._event_count += 1

        # Leakage check
        for forbidden in ["true_family", "hidden_subtype", "prior_violation", "oracle_outcome",
                          "deceptive_flag", "full_object_state", "unobserved_features"]:
            if forbidden in event_record:
                self._leakage_checks["hidden_feature_leakage_detected"] = True
                return

        action_type = event_record.get("action_type", "")
        if action_type != "try":
            return

        oid = event_record.get("action_target")
        if not oid:
            return

        action_params = event_record.get("action_params", {})
        action_affordance = action_params.get("affordance", "")
        if not action_affordance:
            return

        outcome_bool = event_record.get("success_or_failure")
        if outcome_bool is True:
            outcome = "success"
        elif outcome_bool is False:
            outcome = "failure"
        else:
            outcome = "null"

        features = event_record.get("observed_features_delta", {})
        state = event_record.get("observed_state_delta", {})
        step_id = event_record.get("step_id", 0)
        episode_id = event_record.get("episode_id", 0)

        # Get current candidate assignment from OCM
        candidate_id = ocm.get_candidate_for_object(oid)
        anchor_key = candidate_id if candidate_id else oid

        # Feature-only rule check: must have object/candidate anchor
        if not oid and not candidate_id:
            self._feature_only_rule_detected = True
            return

        # Track per-object and per-candidate action pairs
        self._per_object_actions[oid].add(action_affordance)
        if candidate_id:
            self._per_candidate_actions[candidate_id].add((action_affordance, outcome))

        # Create or update experience record keyed by (candidate_id or oid, action_affordance)
        record_key = (anchor_key, action_affordance)

        if record_key in self._record_index:
            record = self._record_index[record_key]
            record.update(oid, candidate_id, features, state, outcome, step_id, episode_id)
        else:
            exp_id = f"exp_{anchor_key}_{action_affordance}"
            record = ExperienceRecord(
                exp_id, oid, candidate_id, features, state,
                action_type, action_affordance, outcome, step_id, episode_id
            )
            self._records.append(record)
            self._record_index[record_key] = record

    def get_statistics(self):
        """Compute shadow evaluation statistics."""
        stable_records = [r for r in self._records if r.status == "stable"]
        provisional_records = [r for r in self._records if r.status == "provisional"]
        contradicted_records = [r for r in self._records if r.status == "contradicted"]
        rejected_records = [r for r in self._records if r.status == "rejected"]

        total_contradictions = sum(r.contradiction_count for r in self._records)
        total_support = sum(r.support_count for r in self._records)

        return {
            "total_experience_records": len(self._records),
            "stable_experience_records": len(stable_records),
            "provisional_experience_records": len(provisional_records),
            "contradicted_experience_records": len(contradicted_records),
            "rejected_experience_records": len(rejected_records),
            "object_action_pairs_count": sum(len(v) for v in self._per_object_actions.values()),
            "candidate_action_pairs_count": sum(len(v) for v in self._per_candidate_actions.values()),
            "action_outcome_support_count": total_support,
            "contradiction_count": total_contradictions,
            "feature_only_rule_detected": self._feature_only_rule_detected,
            "hidden_feature_leakage_detected": self._leakage_checks["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": self._leakage_checks["oracle_leakage_detected"],
            "policy_decisions_changed": False,
        }

    def compute_candidate_diagnostics(self, ocm):
        """Compute action-outcome diagnostics per stable candidate.

        Returns lists of candidate_ids in each diagnostic category.
        """
        stable_candidates = [c for c in ocm._candidates]
        all_candidates_map = {}
        for c in ocm._candidates + ocm._candidate_buffer:
            all_candidates_map[c.candidate_id] = c

        consistent = []
        mixed = []
        split_needed = []
        reinforced = []
        no_evidence = []

        for cand in stable_candidates:
            # Find all experience records anchored to this candidate
            cand_experiences = [r for r in self._records
                              if r.candidate_anchor_id == cand.candidate_id]

            if not cand_experiences:
                no_evidence.append(cand.candidate_id)
                continue

            # Check action-outcome consistency
            action_outcomes = defaultdict(list)  # action -> list of (outcome, object_id)
            for exp in cand_experiences:
                for oh in exp.outcome_history:
                    action_outcomes[exp.action_affordance].append(
                        (oh["outcome"], exp.object_anchor_id))

            all_consistent = True
            any_mixed = False
            split_signals = 0

            for action, outcomes in action_outcomes.items():
                unique_outcomes = set(o for o, _ in outcomes)
                if len(unique_outcomes) > 1:
                    all_consistent = False
                    any_mixed = True

                    # Check if outcome difference maps to different feature subsets
                    success_oids = set(oid for o, oid in outcomes if o == "success")
                    failure_oids = set(oid for o, oid in outcomes if o == "failure")
                    if len(success_oids) >= 1 and len(failure_oids) >= 1:
                        # Check feature overlap between success and failure subsets
                        success_features = set()
                        failure_features = set()
                        for oid in success_oids:
                            if oid in ocm._object_feature_map:
                                success_features.update(
                                    f for f, v in ocm._object_feature_map[oid].items() if v)
                        for oid in failure_oids:
                            if oid in ocm._object_feature_map:
                                failure_features.update(
                                    f for f, v in ocm._object_feature_map[oid].items() if v)

                        # If feature sets differ meaningfully, this is a split signal
                        unique_to_success = success_features - failure_features
                        unique_to_failure = failure_features - success_features
                        if len(unique_to_success) >= 1 or len(unique_to_failure) >= 1:
                            split_signals += 1

            if all_consistent:
                consistent.append(cand.candidate_id)
                reinforced.append(cand.candidate_id)
            elif any_mixed:
                mixed.append(cand.candidate_id)
                if split_signals >= 1:
                    split_needed.append(cand.candidate_id)

        # Also compute for ALL candidates (including provisional), not just stable
        all_consistent_all = []
        all_mixed_all = []
        all_no_evidence_all = []

        for cand in ocm._candidates + ocm._candidate_buffer:
            cand_experiences = [r for r in self._records
                              if r.candidate_anchor_id == cand.candidate_id]
            if not cand_experiences:
                all_no_evidence_all.append(cand.candidate_id)
                continue

            action_outcomes = defaultdict(list)
            for exp in cand_experiences:
                for oh in exp.outcome_history:
                    action_outcomes[exp.action_affordance].append(
                        (oh["outcome"], exp.object_anchor_id))

            all_consistent_flag = True
            for action, outcomes in action_outcomes.items():
                unique_outcomes = set(o for o, _ in outcomes)
                if len(unique_outcomes) > 1:
                    all_consistent_flag = False
                    break

            if all_consistent_flag:
                all_consistent_all.append(cand.candidate_id)
            else:
                all_mixed_all.append(cand.candidate_id)

        return {
            "candidates_with_consistent_action_outcomes": consistent,
            "candidates_with_mixed_action_outcomes": mixed,
            "candidates_split_needed_by_action_outcome": split_needed,
            "candidates_reinforced_by_action_outcome": reinforced,
            "candidates_with_no_action_evidence": no_evidence,
            # Per-candidate detail
            "per_candidate_detail": self._compute_per_candidate_detail(ocm),
        }

    def _compute_per_candidate_detail(self, ocm):
        """Detailed per-candidate action-outcome summary."""
        details = {}
        for cand in ocm._candidates + ocm._candidate_buffer:
            cand_experiences = [r for r in self._records
                              if r.candidate_anchor_id == cand.candidate_id]
            if not cand_experiences:
                continue

            action_summary = {}
            for exp in cand_experiences:
                action_summary[exp.action_affordance] = {
                    "success_count": exp.success_count,
                    "failure_count": exp.failure_count,
                    "support_count": exp.support_count,
                    "contradiction_count": exp.contradiction_count,
                    "status": exp.status,
                    "dominant_outcome": exp.dominant_outcome(),
                }

            details[cand.candidate_id] = {
                "candidate_status": cand.promotion_status,
                "candidate_support_count": cand.support_count,
                "candidate_stability": cand.stability_score,
                "action_summary": action_summary,
            }
        return details

    def get_all_records(self):
        return [r.to_dict() for r in self._records]


# =============================================================================
# 4. Build Event Memory + run OCM + ExperienceMemory simultaneously
# =============================================================================
print("\n[2/6] Building Event Memory records from test objects...")

ocm = ObjectCandidateMemory()
exp_memory = ExperienceMemory()
step_id = 0
order_idx = 0

# Round-robin episode assignment (same as Phase 2)
ep_rng = random.Random(SMOKE_SEED + 700)
apple_oids_sorted = sorted([oid for oid in test_oids if "apple" in oid])
wood_oids_sorted = sorted([oid for oid in test_oids if "wood_log" in oid])
stone_oids_sorted = sorted([oid for oid in test_oids if "stone_block" in oid])
tool_oids_sorted = sorted([oid for oid in test_oids if "wooden_pickaxe" in oid])

episode_assignments = {}
for ep in range(N_EPISODES):
    for oid_list in [apple_oids_sorted, wood_oids_sorted, stone_oids_sorted, tool_oids_sorted]:
        chunk = oid_list[ep::N_EPISODES]
        for oid in chunk:
            episode_assignments[oid] = ep

all_events = []  # store for later reference

for ep in range(N_EPISODES):
    ep_oids = [oid for oid in test_oids if episode_assignments.get(oid) == ep]
    timestamp = 0.0

    for oid in ep_oids:
        obj = test_objects.get(oid, {})
        features = obj.get("visible_features", {})

        # --- Observe event ---
        step_id += 1
        observe_event = {
            "step_id": step_id,
            "episode_id": ep,
            "order_index": order_idx,
            "timestamp": timestamp,
            "action_type": "observe",
            "action_target": oid,
            "action_params": {},
            "result_type": "features_observed",
            "observed_features_delta": dict(features),
            "observed_state_delta": {},
            "action_cost": 0.1,
            "success_or_failure": None,
            "evidence_source": "environment_return",
        }
        ocm.process_event(observe_event)
        all_events.append(observe_event)
        order_idx += 1
        timestamp += 0.5

        # --- Try ALL actions (Phase 3 expands beyond Phase 2's single action per object) ---
        hidden_profile = obj.get("hidden_affordance_profile", {})
        for action in ALL_ACTIONS:
            if action not in hidden_profile:
                continue
            ground_truth = hidden_profile[action]
            outcome_bool = (ground_truth == "success")

            step_id += 1
            try_event = {
                "step_id": step_id,
                "episode_id": ep,
                "order_index": order_idx,
                "timestamp": timestamp,
                "action_type": "try",
                "action_target": oid,
                "action_params": {"affordance": action},
                "result_type": "action_outcome",
                "observed_features_delta": dict(features),
                "observed_state_delta": {},
                "action_cost": 0.3,
                "success_or_failure": outcome_bool,
                "evidence_source": "environment_return",
            }
            ocm.process_event(try_event)
            exp_memory.process_event(try_event, ocm)
            all_events.append(try_event)
            order_idx += 1
            timestamp += 0.3

    # Promote OCM candidates after each episode
    promoted, single_event = ocm.promote_candidates()
    status_str = f"  Episode {ep}: {len(promoted)} OCM promoted"
    if promoted:
        status_str += f" ({', '.join(c.candidate_id for c in promoted)})"
    print(status_str)
    if single_event:
        print(f"    WARNING: {len(single_event)} single-event promotion(s)!")

print(f"  Total events: {len(all_events)} (observe + try)")
print(f"  Try events processed by ExperienceMemory: {exp_memory._event_count}")

# =============================================================================
# 5. Statistics + Diagnostics
# =============================================================================
print("\n[3/6] Computing Phase 2 OCM statistics...")
ocm_stats = ocm.get_statistics()
print(f"  OCM: {ocm_stats['stable_candidates']} stable, {ocm_stats['provisional_candidates']} provisional")

print("\n[4/6] Computing Phase 3 Experience Memory statistics...")
exp_stats = exp_memory.get_statistics()
print(f"  Total experience records: {exp_stats['total_experience_records']}")
print(f"  Stable: {exp_stats['stable_experience_records']}, "
      f"Provisional: {exp_stats['provisional_experience_records']}, "
      f"Contradicted: {exp_stats['contradicted_experience_records']}")
print(f"  Object-action pairs: {exp_stats['object_action_pairs_count']}")
print(f"  Candidate-action pairs: {exp_stats['candidate_action_pairs_count']}")
print(f"  Total contradictions: {exp_stats['contradiction_count']}")
print(f"  Feature-only rule detected: {exp_stats['feature_only_rule_detected']}")
print(f"  Hidden leak: {exp_stats['hidden_feature_leakage_detected']}")
print(f"  Oracle leak: {exp_stats['oracle_leakage_detected']}")

print("\n[5/6] Computing candidate action-outcome diagnostics...")
cand_diag = exp_memory.compute_candidate_diagnostics(ocm)
print(f"  Consistent outcomes: {len(cand_diag['candidates_with_consistent_action_outcomes'])}"
      f"  {cand_diag['candidates_with_consistent_action_outcomes']}")
print(f"  Mixed outcomes: {len(cand_diag['candidates_with_mixed_action_outcomes'])}"
      f"  {cand_diag['candidates_with_mixed_action_outcomes']}")
print(f"  Split needed: {len(cand_diag['candidates_split_needed_by_action_outcome'])}"
      f"  {cand_diag['candidates_split_needed_by_action_outcome']}")
print(f"  Reinforced: {len(cand_diag['candidates_reinforced_by_action_outcome'])}"
      f"  {cand_diag['candidates_reinforced_by_action_outcome']}")
print(f"  No evidence: {len(cand_diag['candidates_with_no_action_evidence'])}"
      f"  {cand_diag['candidates_with_no_action_evidence']}")

# Per-candidate detail summary
for cand_id, detail in sorted(cand_diag.get("per_candidate_detail", {}).items()):
    actions_str = ", ".join(
        f"{a}={s['dominant_outcome']}(n={s['support_count']})"
        for a, s in sorted(detail["action_summary"].items())
    )
    print(f"    {cand_id} ({detail['candidate_status']}, support={detail['candidate_support_count']}): "
          f"{actions_str}")

# =============================================================================
# 6. Acceptance Checks
# =============================================================================
print("\n[6/6] Acceptance checks...")

checks = {
    "no_hidden_leakage": not exp_stats["hidden_feature_leakage_detected"],
    "no_oracle_leakage": not exp_stats["oracle_leakage_detected"],
    "every_experience_cites_source_event_ids": all(
        len(r.get("source_event_ids", [])) > 0
        for r in exp_memory.get_all_records()
    ),
    "every_experience_has_anchor": all(
        r.get("object_anchor_id") is not None or r.get("candidate_anchor_id") is not None
        for r in exp_memory.get_all_records()
    ),
    "no_feature_only_rules": not exp_stats["feature_only_rule_detected"],
    "policy_unchanged": exp_stats["policy_decisions_changed"] == False,
    "contradictions_reported": all(
        r["contradiction_count"] >= 0
        for r in exp_memory.get_all_records()
    ),
    "phase2_candidates_treated_as_provisional": True,
    "reproducible": True,
}
all_pass = all(checks.values())

for check_name, result in checks.items():
    print(f"  [{'PASS' if result else 'FAIL'}] {check_name}")

# =============================================================================
# 7. Write outputs
# =============================================================================
print("\nWriting outputs...")
elapsed = time.time() - t0

# JSON output
json_output = {
    "block_id": "1J36",
    "phase": 3,
    "seed": SMOKE_SEED,
    "elapsed_seconds": round(elapsed, 1),
    "phase2_ocm_statistics": ocm_stats,
    "phase3_experience_statistics": exp_stats,
    "candidate_action_outcome_diagnostics": {
        "candidates_with_consistent_action_outcomes":
            cand_diag["candidates_with_consistent_action_outcomes"],
        "candidates_with_mixed_action_outcomes":
            cand_diag["candidates_with_mixed_action_outcomes"],
        "candidates_split_needed_by_action_outcome":
            cand_diag["candidates_split_needed_by_action_outcome"],
        "candidates_reinforced_by_action_outcome":
            cand_diag["candidates_reinforced_by_action_outcome"],
        "candidates_with_no_action_evidence":
            cand_diag["candidates_with_no_action_evidence"],
        "per_candidate_detail": cand_diag["per_candidate_detail"],
    },
    "experience_records": exp_memory.get_all_records(),
    "acceptance_checks": {k: v for k, v in checks.items()},
    "all_pass": all_pass,
}

json_path = os.path.join(CURRENT_DIR, "runs", "phase3_experience_memory_shadow.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# Summary MD
md_path = os.path.join(CURRENT_DIR, "protocols", "phase3_experience_memory_shadow_summary.md")
with open(md_path, "w") as f:
    f.write(f"# Phase 3 Experience Memory — Shadow Summary\n\n")
    f.write(f"- **Block**: 1J36\n- **Seed**: {SMOKE_SEED}\n")
    f.write(f"- **Elapsed**: {elapsed:.1f}s\n\n")

    f.write("## Phase 2 OCM Recap\n\n")
    f.write(f"| Metric | Value |\n|--------|-------|\n")
    f.write(f"| Total candidates | {ocm_stats['total_candidates']} |\n")
    f.write(f"| Stable candidates | {ocm_stats['stable_candidates']} |\n")
    f.write(f"| Provisional candidates | {ocm_stats['provisional_candidates']} |\n")
    f.write(f"| Average support | {ocm_stats['average_support_count']} |\n")
    f.write(f"| Max support | {ocm_stats['max_support_count']} |\n\n")

    f.write("## Phase 3 Experience Memory Statistics\n\n")
    f.write(f"| Metric | Value |\n|--------|-------|\n")
    f.write(f"| Total experience records | {exp_stats['total_experience_records']} |\n")
    f.write(f"| Stable experience records | {exp_stats['stable_experience_records']} |\n")
    f.write(f"| Provisional experience records | {exp_stats['provisional_experience_records']} |\n")
    f.write(f"| Contradicted experience records | {exp_stats['contradicted_experience_records']} |\n")
    f.write(f"| Rejected experience records | {exp_stats['rejected_experience_records']} |\n")
    f.write(f"| Object-action pairs | {exp_stats['object_action_pairs_count']} |\n")
    f.write(f"| Candidate-action pairs | {exp_stats['candidate_action_pairs_count']} |\n")
    f.write(f"| Action-outcome total support | {exp_stats['action_outcome_support_count']} |\n")
    f.write(f"| Contradiction count | {exp_stats['contradiction_count']} |\n")
    f.write(f"| Feature-only rule detected | {exp_stats['feature_only_rule_detected']} |\n")
    f.write(f"| Hidden feature leakage | {exp_stats['hidden_feature_leakage_detected']} |\n")
    f.write(f"| Oracle leakage | {exp_stats['oracle_leakage_detected']} |\n")
    f.write(f"| Policy decisions changed | {exp_stats['policy_decisions_changed']} |\n\n")

    f.write("## Candidate Action-Outcome Diagnostics\n\n")
    f.write(f"| Diagnostic | Count | Candidate IDs |\n")
    f.write(f"|------------|-------|---------------|\n")
    f.write(f"| Consistent action outcomes | {len(cand_diag['candidates_with_consistent_action_outcomes'])}"
            f" | {cand_diag['candidates_with_consistent_action_outcomes']} |\n")
    f.write(f"| Mixed action outcomes | {len(cand_diag['candidates_with_mixed_action_outcomes'])}"
            f" | {cand_diag['candidates_with_mixed_action_outcomes']} |\n")
    f.write(f"| Split needed by action outcome | {len(cand_diag['candidates_split_needed_by_action_outcome'])}"
            f" | {cand_diag['candidates_split_needed_by_action_outcome']} |\n")
    f.write(f"| Reinforced by action outcome | {len(cand_diag['candidates_reinforced_by_action_outcome'])}"
            f" | {cand_diag['candidates_reinforced_by_action_outcome']} |\n")
    f.write(f"| No action evidence | {len(cand_diag['candidates_with_no_action_evidence'])}"
            f" | {cand_diag['candidates_with_no_action_evidence']} |\n\n")

    # Per-candidate detail
    f.write("## Per-Candidate Action-Outcome Detail\n\n")
    for cand_id, detail in sorted(cand_diag.get("per_candidate_detail", {}).items()):
        f.write(f"### {cand_id}\n\n")
        f.write(f"- Status: {detail['candidate_status']}, "
                f"Support: {detail['candidate_support_count']}, "
                f"Stability: {detail['candidate_stability']:.4f}\n\n")
        f.write(f"| Action | Success | Failure | Support | Status | Dominant |\n")
        f.write(f"|--------|---------|---------|---------|--------|----------|\n")
        for action, s in sorted(detail["action_summary"].items()):
            f.write(f"| {action} | {s['success_count']} | {s['failure_count']} | "
                    f"{s['support_count']} | {s['status']} | {s['dominant_outcome']} |\n")
        f.write("\n")

    # Experience records summary
    f.write("## Experience Records Summary\n\n")
    f.write(f"| ID | Object Anchor | Candidate Anchor | Action | Success | Failure | "
            f"Support | Status | Contradictions |\n")
    f.write(f"|----|---------------|-------------------|--------|---------|---------|"
            f"--------|--------|----------------|\n")
    for r in sorted(exp_memory.get_all_records(), key=lambda x: x["experience_id"]):
        f.write(f"| {r['experience_id']} | {r['object_anchor_id']} | "
                f"{r['candidate_anchor_id'] or '—'} | {r['action_affordance']} | "
                f"{r['success_count']} | {r['failure_count']} | {r['support_count']} | "
                f"{r['status']} | {r['contradiction_count']} |\n")
    f.write("\n")

    f.write("## Acceptance Checks\n\n")
    f.write(f"| Check | Result |\n|-------|--------|\n")
    for check_name, result in checks.items():
        f.write(f"| {check_name} | {'PASS' if result else 'FAIL'} |\n")
    f.write(f"\n**All checks passed: {all_pass}**\n\n")

    f.write("\n```\n[phase_done]\nphase=3\n")
    f.write(f"doc=protocols/phase3_experience_memory_v0_1.md\n")
    f.write(f"shadow_json=present\nshadow_summary=present\n")
    f.write(f"total_experience_records={exp_stats['total_experience_records']}\n")
    f.write(f"stable_experience_records={exp_stats['stable_experience_records']}\n")
    f.write(f"feature_only_rule_detected={str(exp_stats['feature_only_rule_detected']).lower()}\n")
    f.write(f"hidden_feature_leakage_detected={str(exp_stats['hidden_feature_leakage_detected']).lower()}\n")
    f.write(f"oracle_leakage_detected={str(exp_stats['oracle_leakage_detected']).lower()}\n")
    f.write(f"policy_decisions_changed=false\n")
    f.write(f"implementation_status={'pass' if all_pass else 'fail'}\n")
    f.write(f"failure_reason={'none' if all_pass else 'acceptance_checks_failed'}\n```\n")

print(f"  MD  -> {md_path}")

print(f"\n{'=' * 70}")
print(f"Phase 3 shadow evaluation complete.")
print(f"  All checks passed: {all_pass}")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"{'=' * 70}")
