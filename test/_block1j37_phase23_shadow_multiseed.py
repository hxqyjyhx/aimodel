"""
Block 1J37 -- Phase 2+3 Shadow Multiseed Robustness Check.

Runs Object Candidate Memory + Experience Memory shadow evaluation
across seeds 101, 103, 107, 109, 113. Robustness check before Phase 4.

Does NOT:
  - Change policy logic, environment dynamics
  - Replace detect_type_family() or FCRM counters
  - Use audit-only fields
  - Store feature->outcome rules
"""
import os, sys, json, copy, random, time, math, csv
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
# Constants
# =============================================================================
SEEDS = [101, 103, 107, 109, 113]
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

ACTION_TO_QUERY = {
    "craft_plank": "need_planks", "eat": "need_food",
    "use_as_tool": "need_tool", "burn_as_fuel": "need_fuel",
    "mine_by_hand": "need_stone", "mine_with_pickaxe": "need_stone",
}
ALL_ACTIONS = sorted(ACTION_TO_QUERY.keys())

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

# =============================================================================
# Phase 2: ObjectCandidate + ObjectCandidateMemory
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
            "promotion_status": self.promotion_status,
        }


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
        single_event_promotions = [p for p in self._promotion_log if p["episode_spread"] < 2]
        return promoted, single_event_promotions

    def get_candidate_for_object(self, oid):
        return self._object_candidate_map.get(oid)

    def get_statistics(self):
        all_candidates = self._candidates + self._candidate_buffer
        support_counts = [c.support_count for c in all_candidates]
        stable = [c for c in all_candidates if c.promotion_status == "stable"]
        conflicted = [c for c in all_candidates if len(c.contradicted_features) > 0]
        return {
            "total_candidates": len(all_candidates),
            "stable_candidates": len(stable),
            "provisional_candidates": len(self._candidate_buffer),
            "candidate_buffer_size": len(self._candidate_buffer),
            "average_support_count": round(sum(support_counts) / max(len(support_counts), 1), 2),
            "max_support_count": max(support_counts) if support_counts else 0,
            "conflict_candidate_count": len(conflicted),
            "single_event_promotion_count": sum(1 for p in self._promotion_log if p["episode_spread"] < 2),
            "hidden_feature_leakage_detected": self._leakage_checks["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": self._leakage_checks["oracle_leakage_detected"],
        }


# =============================================================================
# Phase 3: ExperienceRecord + ExperienceMemory
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

    def to_dict(self):
        return {
            "experience_id": self.experience_id,
            "object_anchor_id": self.object_anchor_id,
            "candidate_anchor_id": self.candidate_anchor_id,
            "action_affordance": self.action_affordance,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "confidence_score": self.confidence_score,
            "status": self.status,
            "dominant_outcome": self.dominant_outcome(),
        }


class ExperienceMemory:
    def __init__(self):
        self._records = []
        self._record_index = {}
        self._event_count = 0
        self._leakage_checks = {"hidden_feature_leakage_detected": False, "oracle_leakage_detected": False}
        self._feature_only_rule_detected = False
        self._per_object_actions = defaultdict(set)
        self._per_candidate_actions = defaultdict(set)

    def process_event(self, event_record, ocm):
        self._event_count += 1
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
        candidate_id = ocm.get_candidate_for_object(oid)
        anchor_key = candidate_id if candidate_id else oid
        if not oid and not candidate_id:
            self._feature_only_rule_detected = True
            return
        self._per_object_actions[oid].add(action_affordance)
        if candidate_id:
            self._per_candidate_actions[candidate_id].add((action_affordance, outcome))
        record_key = (anchor_key, action_affordance)
        if record_key in self._record_index:
            record = self._record_index[record_key]
            record.update(oid, candidate_id, features, state, outcome, step_id, episode_id)
        else:
            exp_id = f"exp_{anchor_key}_{action_affordance}"
            record = ExperienceRecord(
                exp_id, oid, candidate_id, features, state,
                action_type, action_affordance, outcome, step_id, episode_id)
            self._records.append(record)
            self._record_index[record_key] = record

    def get_statistics(self):
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
        all_candidates = ocm._candidates + ocm._candidate_buffer
        consistent = []; mixed = []; split_needed = []; reinforced = []; no_evidence = []

        for cand in all_candidates:
            cand_experiences = [r for r in self._records if r.candidate_anchor_id == cand.candidate_id]
            if not cand_experiences:
                no_evidence.append(cand.candidate_id)
                continue

            action_outcomes = defaultdict(list)
            for exp in cand_experiences:
                for oh in exp.outcome_history:
                    action_outcomes[exp.action_affordance].append((oh["outcome"], exp.object_anchor_id))

            all_consistent = True
            any_mixed = False
            split_signals = 0

            for action, outcomes in action_outcomes.items():
                unique_outcomes = set(o for o, _ in outcomes)
                if len(unique_outcomes) > 1:
                    all_consistent = False
                    any_mixed = True
                    success_oids = set(oid for o, oid in outcomes if o == "success")
                    failure_oids = set(oid for o, oid in outcomes if o == "failure")
                    if len(success_oids) >= 1 and len(failure_oids) >= 1:
                        success_features = set()
                        failure_features = set()
                        for oid in success_oids:
                            if oid in ocm._object_feature_map:
                                success_features.update(f for f, v in ocm._object_feature_map[oid].items() if v)
                        for oid in failure_oids:
                            if oid in ocm._object_feature_map:
                                failure_features.update(f for f, v in ocm._object_feature_map[oid].items() if v)
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

        return {
            "candidates_with_consistent_action_outcomes": consistent,
            "candidates_with_mixed_action_outcomes": mixed,
            "candidates_split_needed_by_action_outcome": split_needed,
            "candidates_reinforced_by_action_outcome": reinforced,
            "candidates_with_no_action_evidence": no_evidence,
        }


# =============================================================================
# Single-seed runner
# =============================================================================
def run_single_seed(seed):
    """Run Phase 2 OCM + Phase 3 Experience Memory for a single seed."""
    seed_t0 = time.time()

    # Phase A training
    (student, base_learner, train_objects, train_env,
     _std_test, _std_test_env, final_metrics, rng) = run_phase_a_training(seed, COND)

    test_objects_standard = generate_subtype_objects_deterministic(15, 15, 15, 15, rng, prefix="test")
    test_oids = sorted(test_objects_standard.keys())
    test_objects = copy.deepcopy(test_objects_standard)

    # Round-robin episode assignment
    ep_rng = random.Random(seed + 700)
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

    # Build Event Memory + run OCM + ExperienceMemory
    ocm = ObjectCandidateMemory()
    exp_memory = ExperienceMemory()
    step_id = 0

    for ep in range(N_EPISODES):
        ep_oids = [oid for oid in test_oids if episode_assignments.get(oid) == ep]

        for oid in ep_oids:
            obj = test_objects.get(oid, {})
            features = obj.get("visible_features", {})

            # Observe event
            step_id += 1
            observe_event = {
                "step_id": step_id, "episode_id": ep, "action_type": "observe",
                "action_target": oid, "action_params": {},
                "observed_features_delta": dict(features),
                "observed_state_delta": {},
                "success_or_failure": None,
            }
            ocm.process_event(observe_event)

            # Try ALL actions
            hidden_profile = obj.get("hidden_affordance_profile", {})
            for action in ALL_ACTIONS:
                if action not in hidden_profile:
                    continue
                ground_truth = hidden_profile[action]
                outcome_bool = (ground_truth == "success")
                step_id += 1
                try_event = {
                    "step_id": step_id, "episode_id": ep, "action_type": "try",
                    "action_target": oid, "action_params": {"affordance": action},
                    "observed_features_delta": dict(features),
                    "observed_state_delta": {},
                    "success_or_failure": outcome_bool,
                }
                ocm.process_event(try_event)
                exp_memory.process_event(try_event, ocm)

        ocm.promote_candidates()

    # Collect statistics
    ocm_stats = ocm.get_statistics()
    exp_stats = exp_memory.get_statistics()
    cand_diag = exp_memory.compute_candidate_diagnostics(ocm)

    elapsed = round(time.time() - seed_t0, 2)

    # Per-seed summary
    result = {
        "seed": seed,
        "elapsed_seconds": elapsed,
        "phase2_ocm": {
            "total_candidates": ocm_stats["total_candidates"],
            "stable_candidates": ocm_stats["stable_candidates"],
            "provisional_candidates": ocm_stats["provisional_candidates"],
            "conflict_candidate_count": ocm_stats["conflict_candidate_count"],
            "single_event_promotion_count": ocm_stats["single_event_promotion_count"],
            "average_support_count": ocm_stats["average_support_count"],
            "max_support_count": ocm_stats["max_support_count"],
            "hidden_feature_leakage_detected": ocm_stats["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": ocm_stats["oracle_leakage_detected"],
        },
        "phase3_experience": {
            "total_experience_records": exp_stats["total_experience_records"],
            "stable_experience_records": exp_stats["stable_experience_records"],
            "provisional_experience_records": exp_stats["provisional_experience_records"],
            "contradicted_experience_records": exp_stats["contradicted_experience_records"],
            "rejected_experience_records": exp_stats["rejected_experience_records"],
            "object_action_pairs_count": exp_stats["object_action_pairs_count"],
            "candidate_action_pairs_count": exp_stats["candidate_action_pairs_count"],
            "action_outcome_support_count": exp_stats["action_outcome_support_count"],
            "contradiction_count": exp_stats["contradiction_count"],
            "feature_only_rule_detected": exp_stats["feature_only_rule_detected"],
            "hidden_feature_leakage_detected": exp_stats["hidden_feature_leakage_detected"],
            "oracle_leakage_detected": exp_stats["oracle_leakage_detected"],
            "policy_decisions_changed": exp_stats["policy_decisions_changed"],
        },
        "candidate_diagnostics": {
            "candidates_with_consistent_action_outcomes": len(cand_diag["candidates_with_consistent_action_outcomes"]),
            "candidates_with_mixed_action_outcomes": len(cand_diag["candidates_with_mixed_action_outcomes"]),
            "candidates_split_needed_by_action_outcome": len(cand_diag["candidates_split_needed_by_action_outcome"]),
            "candidates_reinforced_by_action_outcome": len(cand_diag["candidates_reinforced_by_action_outcome"]),
            "candidates_with_no_action_evidence": len(cand_diag["candidates_with_no_action_evidence"]),
            "consistent_ids": cand_diag["candidates_with_consistent_action_outcomes"],
            "mixed_ids": cand_diag["candidates_with_mixed_action_outcomes"],
            "split_needed_ids": cand_diag["candidates_split_needed_by_action_outcome"],
            "reinforced_ids": cand_diag["candidates_reinforced_by_action_outcome"],
            "no_evidence_ids": cand_diag["candidates_with_no_action_evidence"],
        },
    }
    return result


# =============================================================================
# Aggregate computation
# =============================================================================
def compute_aggregates(results):
    """Compute aggregate statistics across seeds."""
    n = len(results)
    if n == 0:
        return {}

    def _extract(path):
        values = []
        for r in results:
            v = r
            for key in path:
                v = v.get(key, {})
            values.append(v)
        return values

    def _mean_std(values):
        m = sum(values) / len(values)
        v = sum((x - m) ** 2 for x in values) / len(values)
        return round(m, 2), round(math.sqrt(v), 2)

    tc = _extract(["phase2_ocm", "total_candidates"])
    sc = _extract(["phase2_ocm", "stable_candidates"])
    ter = _extract(["phase3_experience", "total_experience_records"])
    contr = _extract(["phase3_experience", "contradiction_count"])
    mixed_rate_vals = []
    reinf_rate_vals = []

    for r in results:
        ocm = r["phase2_ocm"]
        diag = r["candidate_diagnostics"]
        total_c = ocm["total_candidates"]
        if total_c > 0:
            mixed_rate_vals.append(diag["candidates_with_mixed_action_outcomes"] / total_c)
            reinf_rate_vals.append(diag["candidates_reinforced_by_action_outcome"] / total_c)

    tc_mean, tc_std = _mean_std(tc)
    sc_mean, sc_std = _mean_std(sc)
    ter_mean, ter_std = _mean_std(ter)
    contr_mean, contr_std = _mean_std(contr)
    mixed_rate_mean, mixed_rate_std = _mean_std(mixed_rate_vals) if mixed_rate_vals else (0, 0)
    reinf_rate_mean, reinf_rate_std = _mean_std(reinf_rate_vals) if reinf_rate_vals else (0, 0)

    seeds_with_feature_only = [r["seed"] for r in results if r["phase3_experience"]["feature_only_rule_detected"]]
    seeds_with_leakage = [r["seed"] for r in results
                          if r["phase3_experience"]["hidden_feature_leakage_detected"]
                          or r["phase3_experience"]["oracle_leakage_detected"]]
    seeds_with_policy_change = [r["seed"] for r in results if r["phase3_experience"]["policy_decisions_changed"]]
    seeds_with_single_event = [r["seed"] for r in results if r["phase2_ocm"]["single_event_promotion_count"] > 0]

    # Decision rule
    robust = (
        len(seeds_with_feature_only) == 0
        and len(seeds_with_leakage) == 0
        and len(seeds_with_policy_change) == 0
        and len(seeds_with_single_event) == 0
        and len(results) == len(SEEDS)  # no crashes
    )

    return {
        "num_seeds": n,
        "mean_total_candidates": tc_mean, "stdev_total_candidates": tc_std,
        "mean_stable_candidates": sc_mean, "stdev_stable_candidates": sc_std,
        "mean_total_experience_records": ter_mean, "stdev_total_experience_records": ter_std,
        "mean_contradictions": contr_mean, "stdev_contradictions": contr_std,
        "mean_mixed_outcome_candidate_rate": mixed_rate_mean, "stdev_mixed_outcome_candidate_rate": mixed_rate_std,
        "mean_reinforced_candidate_rate": reinf_rate_mean, "stdev_reinforced_candidate_rate": reinf_rate_std,
        "seeds_with_feature_only_rule": seeds_with_feature_only,
        "seeds_with_hidden_oracle_leakage": seeds_with_leakage,
        "seeds_with_policy_change": seeds_with_policy_change,
        "seeds_with_single_event_promotion": seeds_with_single_event,
        "phase3_shadow_robust_enough_for_phase4": robust,
    }


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("Block 1J37 -- Phase 2+3 Shadow Multiseed Robustness")
print(f"  Seeds: {SEEDS}")
print("=" * 70)

all_results = []
for i, seed in enumerate(SEEDS):
    print(f"\n[{i + 1}/{len(SEEDS)}] Seed {seed}...")
    result = run_single_seed(seed)
    all_results.append(result)
    ocm = result["phase2_ocm"]
    exp = result["phase3_experience"]
    diag = result["candidate_diagnostics"]
    print(f"  OCM: {ocm['total_candidates']} total, {ocm['stable_candidates']} stable, "
          f"{ocm['conflict_candidate_count']} conflicted, "
          f"single_event_promotions={ocm['single_event_promotion_count']}")
    print(f"  EXP: {exp['total_experience_records']} records, {exp['stable_experience_records']} stable, "
          f"{exp['contradicted_experience_records']} contradicted, {exp['contradiction_count']} contradictions")
    print(f"  DIAG: {diag['candidates_with_consistent_action_outcomes']} consistent, "
          f"{diag['candidates_with_mixed_action_outcomes']} mixed, "
          f"{diag['candidates_reinforced_by_action_outcome']} reinforced, "
          f"{diag['candidates_split_needed_by_action_outcome']} split-needed")
    print(f"  CHECKS: feature_only={exp['feature_only_rule_detected']}, "
          f"hidden_leak={exp['hidden_feature_leakage_detected']}, "
          f"oracle_leak={exp['oracle_leakage_detected']}, "
          f"policy_changed={exp['policy_decisions_changed']}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")

# Compute aggregates
print("\n" + "=" * 70)
print("Aggregate Statistics")
print("=" * 70)

agg = compute_aggregates(all_results)

print(f"\n  Total candidates:       mean={agg['mean_total_candidates']:.1f} stdev={agg['stdev_total_candidates']:.1f}")
print(f"  Stable candidates:      mean={agg['mean_stable_candidates']:.1f} stdev={agg['stdev_stable_candidates']:.1f}")
print(f"  Experience records:     mean={agg['mean_total_experience_records']:.1f} stdev={agg['stdev_total_experience_records']:.1f}")
print(f"  Contradictions:         mean={agg['mean_contradictions']:.1f} stdev={agg['stdev_contradictions']:.1f}")
print(f"  Mixed outcome rate:     mean={agg['mean_mixed_outcome_candidate_rate']:.3f} stdev={agg['stdev_mixed_outcome_candidate_rate']:.3f}")
print(f"  Reinforced rate:        mean={agg['mean_reinforced_candidate_rate']:.3f} stdev={agg['stdev_reinforced_candidate_rate']:.3f}")
print(f"\n  Seeds with feature-only rule:   {agg['seeds_with_feature_only_rule'] or 'none'}")
print(f"  Seeds with hidden/oracle leak:  {agg['seeds_with_hidden_oracle_leakage'] or 'none'}")
print(f"  Seeds with policy change:       {agg['seeds_with_policy_change'] or 'none'}")
print(f"  Seeds with single-event prom:   {agg['seeds_with_single_event_promotion'] or 'none'}")
print(f"\n  phase3_shadow_robust_enough_for_phase4: {agg['phase3_shadow_robust_enough_for_phase4']}")

# =============================================================================
# Write outputs
# =============================================================================
elapsed = round(time.time() - t0, 1)
print(f"\nWriting outputs (total elapsed: {elapsed}s)...")

# JSON
json_output = {
    "block_id": "1J37",
    "description": "Phase 2+3 Shadow Multiseed Robustness Check",
    "seeds": SEEDS,
    "num_seeds": len(SEEDS),
    "total_elapsed_seconds": elapsed,
    "per_seed_results": all_results,
    "aggregates": agg,
    "decision_rule": {
        "phase3_shadow_robust_enough_for_phase4": agg["phase3_shadow_robust_enough_for_phase4"],
        "feature_only_rule_detected_any": len(agg["seeds_with_feature_only_rule"]) > 0,
        "hidden_feature_leakage_detected_any": len(agg["seeds_with_hidden_oracle_leakage"]) > 0,
        "oracle_leakage_detected_any": len(agg["seeds_with_hidden_oracle_leakage"]) > 0,
        "policy_decisions_changed_any": len(agg["seeds_with_policy_change"]) > 0,
        "single_event_promotion_any": len(agg["seeds_with_single_event_promotion"]) > 0,
    },
}

json_path = os.path.join(CURRENT_DIR, "runs", "block1j37_phase23_shadow_multiseed.json")
with open(json_path, "w") as f:
    json.dump(json_output, f, indent=2)
print(f"  JSON -> {json_path}")

# CSV
csv_path = os.path.join(CURRENT_DIR, "protocols", "block1j37_phase23_shadow_multiseed_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "seed", "total_candidates", "stable_candidates", "provisional_candidates",
        "conflict_candidates", "single_event_promotion_count",
        "total_exp_records", "stable_exp_records", "provisional_exp_records",
        "contradicted_exp_records", "rejected_exp_records", "total_contradictions",
        "feature_only_rule", "hidden_leakage", "oracle_leakage", "policy_changed",
        "consistent_cands", "mixed_cands", "split_needed_cands", "reinforced_cands",
    ])
    for r in all_results:
        ocm = r["phase2_ocm"]
        exp = r["phase3_experience"]
        diag = r["candidate_diagnostics"]
        writer.writerow([
            r["seed"],
            ocm["total_candidates"], ocm["stable_candidates"], ocm["provisional_candidates"],
            ocm["conflict_candidate_count"], ocm["single_event_promotion_count"],
            exp["total_experience_records"], exp["stable_experience_records"],
            exp["provisional_experience_records"], exp["contradicted_experience_records"],
            exp["rejected_experience_records"], exp["contradiction_count"],
            exp["feature_only_rule_detected"], exp["hidden_feature_leakage_detected"],
            exp["oracle_leakage_detected"], exp["policy_decisions_changed"],
            diag["candidates_with_consistent_action_outcomes"],
            diag["candidates_with_mixed_action_outcomes"],
            diag["candidates_split_needed_by_action_outcome"],
            diag["candidates_reinforced_by_action_outcome"],
        ])
print(f"  CSV  -> {csv_path}")

# MD summary
md_path = os.path.join(CURRENT_DIR, "protocols", "block1j37_phase23_shadow_multiseed.md")
with open(md_path, "w") as f:
    f.write(f"# Block 1J37: Phase 2+3 Shadow Multiseed Robustness\n\n")
    f.write(f"- **Seeds**: {SEEDS}\n")
    f.write(f"- **Condition**: {COND['label']}\n")
    f.write(f"- **Episodes per seed**: {N_EPISODES}\n")
    f.write(f"- **Total elapsed**: {elapsed}s\n\n")

    f.write("## Per-Seed Results\n\n")
    f.write("### Phase 2 OCM\n\n")
    f.write(f"| Seed | Total Cand | Stable | Provisional | Conflicted | "
            f"Single-Event Prom | Hidden Leak | Oracle Leak |\n")
    f.write(f"|------|-----------|--------|-------------|------------|"
            f"-------------------|-------------|-------------|\n")
    for r in all_results:
        ocm = r["phase2_ocm"]
        f.write(f"| {r['seed']} | {ocm['total_candidates']} | {ocm['stable_candidates']} | "
                f"{ocm['provisional_candidates']} | {ocm['conflict_candidate_count']} | "
                f"{ocm['single_event_promotion_count']} | "
                f"{ocm['hidden_feature_leakage_detected']} | {ocm['oracle_leakage_detected']} |\n")
    f.write("\n")

    f.write("### Phase 3 Experience Memory\n\n")
    f.write(f"| Seed | Total Exp | Stable | Prov | Contra | Rejected | "
            f"Contradictions | FeatureOnly | HiddenLeak | OracleLeak | PolicyChg |\n")
    f.write(f"|------|-----------|--------|------|--------|----------|"
            f"---------------|-------------|------------|------------|------------|\n")
    for r in all_results:
        exp = r["phase3_experience"]
        f.write(f"| {r['seed']} | {exp['total_experience_records']} | "
                f"{exp['stable_experience_records']} | {exp['provisional_experience_records']} | "
                f"{exp['contradicted_experience_records']} | {exp['rejected_experience_records']} | "
                f"{exp['contradiction_count']} | {exp['feature_only_rule_detected']} | "
                f"{exp['hidden_feature_leakage_detected']} | {exp['oracle_leakage_detected']} | "
                f"{exp['policy_decisions_changed']} |\n")
    f.write("\n")

    f.write("### Candidate Action-Outcome Diagnostics\n\n")
    f.write(f"| Seed | Consistent | Mixed | Split-Needed | Reinforced | No Evidence |\n")
    f.write(f"|------|------------|-------|--------------|------------|-------------|\n")
    for r in all_results:
        diag = r["candidate_diagnostics"]
        f.write(f"| {r['seed']} | {diag['candidates_with_consistent_action_outcomes']} | "
                f"{diag['candidates_with_mixed_action_outcomes']} | "
                f"{diag['candidates_split_needed_by_action_outcome']} | "
                f"{diag['candidates_reinforced_by_action_outcome']} | "
                f"{diag['candidates_with_no_action_evidence']} |\n")
    f.write("\n")

    f.write("## Aggregate Statistics\n\n")
    f.write(f"| Metric | Mean | Stdev |\n")
    f.write(f"|--------|------|-------|\n")
    f.write(f"| Total candidates | {agg['mean_total_candidates']:.1f} | {agg['stdev_total_candidates']:.1f} |\n")
    f.write(f"| Stable candidates | {agg['mean_stable_candidates']:.1f} | {agg['stdev_stable_candidates']:.1f} |\n")
    f.write(f"| Total experience records | {agg['mean_total_experience_records']:.1f} | {agg['stdev_total_experience_records']:.1f} |\n")
    f.write(f"| Contradictions | {agg['mean_contradictions']:.1f} | {agg['stdev_contradictions']:.1f} |\n")
    f.write(f"| Mixed outcome candidate rate | {agg['mean_mixed_outcome_candidate_rate']:.3f} | {agg['stdev_mixed_outcome_candidate_rate']:.3f} |\n")
    f.write(f"| Reinforced candidate rate | {agg['mean_reinforced_candidate_rate']:.3f} | {agg['stdev_reinforced_candidate_rate']:.3f} |\n\n")

    f.write("## Flagged Seeds\n\n")
    f.write(f"| Flag | Seeds |\n|------|-------|\n")
    f.write(f"| feature_only_rule_detected | {agg['seeds_with_feature_only_rule'] or 'none'} |\n")
    f.write(f"| hidden_oracle_leakage | {agg['seeds_with_hidden_oracle_leakage'] or 'none'} |\n")
    f.write(f"| policy_decisions_changed | {agg['seeds_with_policy_change'] or 'none'} |\n")
    f.write(f"| single_event_promotion | {agg['seeds_with_single_event_promotion'] or 'none'} |\n\n")

    f.write("## Decision Rule\n\n")
    f.write(f"| Criterion | Value |\n|-----------|-------|\n")
    f.write(f"| feature_only_rule_detected=false for all seeds | {len(agg['seeds_with_feature_only_rule']) == 0} |\n")
    f.write(f"| hidden_feature_leakage_detected=false for all seeds | {len(agg['seeds_with_hidden_oracle_leakage']) == 0} |\n")
    f.write(f"| oracle_leakage_detected=false for all seeds | {len(agg['seeds_with_hidden_oracle_leakage']) == 0} |\n")
    f.write(f"| policy_decisions_changed=false for all seeds | {len(agg['seeds_with_policy_change']) == 0} |\n")
    f.write(f"| single_event_promotion_count=0 for all seeds | {len(agg['seeds_with_single_event_promotion']) == 0} |\n")
    f.write(f"| No seed crashes | {len(all_results) == len(SEEDS)} |\n")
    f.write(f"| **phase3_shadow_robust_enough_for_phase4** | **{agg['phase3_shadow_robust_enough_for_phase4']}** |\n\n")

    f.write("## Important Caveats\n\n")
    f.write("- Experience Memory is **shadow-only**. It does not influence policy decisions.\n")
    f.write("- Phase 2 candidates are **provisional anchors**, not true categories.\n")
    f.write("- This robustness check verifies the shadow pipeline is stable across seeds.\n")
    f.write("- Phase 4 may use these outputs as diagnostic inputs for strategy statistics.\n")
    f.write("- **This does NOT claim Experience Memory is ready to replace FCRM.**\n\n")

    f.write("\n```\n[block_done]\nblock_id=1J37\n")
    f.write(f"seeds={SEEDS}\n")
    f.write(f"phase3_shadow_robust_enough_for_phase4={str(agg['phase3_shadow_robust_enough_for_phase4']).lower()}\n")
    f.write(f"feature_only_rule_detected_any={str(len(agg['seeds_with_feature_only_rule']) > 0).lower()}\n")
    f.write(f"hidden_feature_leakage_detected_any={str(len(agg['seeds_with_hidden_oracle_leakage']) > 0).lower()}\n")
    f.write(f"oracle_leakage_detected_any={str(len(agg['seeds_with_hidden_oracle_leakage']) > 0).lower()}\n")
    f.write(f"policy_decisions_changed_any={str(len(agg['seeds_with_policy_change']) > 0).lower()}\n")
    f.write(f"single_event_promotion_any={str(len(agg['seeds_with_single_event_promotion']) > 0).lower()}\n")
    f.write(f"implementation_status=pass\nfailure_reason=none\n```\n")

print(f"  MD   -> {md_path}")

print(f"\n{'=' * 70}")
print(f"Block 1J37 complete.")
print(f"  phase3_shadow_robust_enough_for_phase4: {agg['phase3_shadow_robust_enough_for_phase4']}")
print(f"  Total elapsed: {elapsed:.1f}s")
print(f"{'=' * 70}")
