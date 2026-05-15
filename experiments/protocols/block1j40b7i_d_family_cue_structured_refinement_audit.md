# Checkpoint Summary

- status: PARTIAL_CANDIDATE_FOUND
- best_candidate_overall: {'variant_id': 'D4_cue_group_balanced_soft', 'parameters': {'A_spotted': 0.68, 'A_ambient_only': 0.56, 'A_bark_discolored': 0.36, 'A_crystal': 0.4, 'B_discolored': 0.58, 'B_ambient_only': 0.42, 'B_grip': 0.32, 'B_stem_spotted': 0.48, 'salt': 141}, 'test_type': 'A', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- best_candidate_test_A: {'variant_id': 'D4_cue_group_balanced_soft', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- best_candidate_test_B: {'variant_id': 'D6_candidate_set_structured', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- candidates_passing_primary: 0
- candidates_passing_partial: 8

# Files Changed

- _block1j40b7i_d_family_cue_structured_refinement_audit.py

# Commands Run

- `python _block1j40b7i_d_family_cue_structured_refinement_audit.py`

# Why 7i Was Run

- 7h showed partial raw headroom but not clean cue-structured compositional predictability.
- 7i narrows to D-family refinements only, with no broad A/B/C reruns.

# 7h Recap

- D-family in 7h improved oracle gap but not `best_level - always_observe`.
- Positive rate was often too high and cue correlations were weak or wrong-sign.

# Candidate Variants

- `D4_cue_group_balanced_soft`: Cue-group soft balancing with useful/mixed/wasteful groups derived from legal cue composition
- `D5_pairwise_cue_interaction`: Pairwise cue interaction assignment where single cues stay ambiguous but cue pairs carry signal
- `D6_candidate_set_structured`: Subset/Jaccard structured assignment intended to improve candidate-set aggregation
- `D7_hybrid_soft_structured`: Hybrid cue-group plus pairwise refinement with non-deterministic structure

# Test A Results

- `D4_cue_group_balanced_soft`: oracle_gap=0.0531, ao_minus_at=0.1567, best=level1, best_delta=0.0000, sign=0.7722, rate=0.7722, rej=best_level_does_not_beat_always_observe;positive_rate_out_of_range;single_cue_too_predictive
- `D5_pairwise_cue_interaction`: oracle_gap=0.0653, ao_minus_at=0.1367, best=level1, best_delta=0.0000, sign=0.6611, rate=0.6611, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive
- `D6_candidate_set_structured`: oracle_gap=0.0621, ao_minus_at=0.1300, best=level1, best_delta=0.0000, sign=0.6889, rate=0.6889, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive
- `D7_hybrid_soft_structured`: oracle_gap=0.0633, ao_minus_at=0.1433, best=level1, best_delta=0.0000, sign=0.6833, rate=0.6833, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive

# Test B Results

- `D4_cue_group_balanced_soft`: oracle_gap=0.0674, ao_minus_at=0.1367, best=level1, best_delta=0.0000, sign=0.6778, rate=0.6778, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive
- `D5_pairwise_cue_interaction`: oracle_gap=0.0623, ao_minus_at=0.1233, best=level1, best_delta=0.0000, sign=0.6333, rate=0.6333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive
- `D6_candidate_set_structured`: oracle_gap=0.0567, ao_minus_at=0.1567, best=level1, best_delta=0.0000, sign=0.7556, rate=0.7556, rej=best_level_does_not_beat_always_observe;positive_rate_out_of_range;single_cue_too_predictive
- `D7_hybrid_soft_structured`: oracle_gap=0.0678, ao_minus_at=0.1233, best=level1, best_delta=0.0000, sign=0.6222, rate=0.6222, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive

# Test C Results

- `D4_cue_group_balanced_soft`: oracle_gap=0.0554, ao_minus_at=0.1200, best=level1, best_delta=0.0000, sign=0.6722, rate=0.6722, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive
- `D5_pairwise_cue_interaction`: oracle_gap=0.0711, ao_minus_at=0.0933, best=level1, best_delta=0.0000, sign=0.5944, rate=0.5944, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive
- `D6_candidate_set_structured`: oracle_gap=0.0607, ao_minus_at=0.1033, best=level1, best_delta=0.0000, sign=0.5889, rate=0.5889, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive
- `D7_hybrid_soft_structured`: oracle_gap=0.0626, ao_minus_at=0.0967, best=level1, best_delta=0.0000, sign=0.6000, rate=0.6000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive

# Structure Validity Audit

- `D4_cue_group_balanced_soft|A`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D4_cue_group_balanced_soft|B`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D5_pairwise_cue_interaction|A`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D5_pairwise_cue_interaction|B`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D6_candidate_set_structured|A`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D6_candidate_set_structured|B`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D7_hybrid_soft_structured|A`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000
- `D7_hybrid_soft_structured|B`: exact_det=0.5000, single_cue_max=1.0000, pair_max=1.0000, reachability=1.0000

# Best Candidate

- {'variant_id': 'D4_cue_group_balanced_soft', 'parameters': {'A_spotted': 0.68, 'A_ambient_only': 0.56, 'A_bark_discolored': 0.36, 'A_crystal': 0.4, 'B_discolored': 0.58, 'B_ambient_only': 0.42, 'B_grip': 0.32, 'B_stem_spotted': 0.48, 'salt': 141}, 'test_type': 'A', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}

# Partial Candidates

- [{'variant_id': 'D4_cue_group_balanced_soft', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D4_cue_group_balanced_soft', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D5_pairwise_cue_interaction', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D5_pairwise_cue_interaction', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D6_candidate_set_structured', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D6_candidate_set_structured', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D7_hybrid_soft_structured', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D7_hybrid_soft_structured', 'test_type': 'B', 'observe_cost': 0.03}]

# Rejected Variants and Reasons

- `best_level_does_not_beat_always_observe;positive_rate_out_of_range;single_cue_too_predictive`: 2
- `best_level_does_not_beat_always_observe;sign_correct_not_enough;positive_rate_out_of_range;single_cue_too_predictive`: 4
- `best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive`: 2

# Validity / Leakage Audit

- no learner training: true
- no 1J40b-8: true
- no official env replacement: true
- forbidden test keys absent: true
- audit only: true

# Recommended Resume Point

- If status remains partial: one more D-family refinement focused on sign accuracy and keeping `always_observe > always_try`.
- If no clean pass exists: deeper environment redesign before any architecture work.