# Checkpoint Summary

- status: PARTIAL_CANDIDATE_FOUND
- best_candidate_overall: {'variant_id': 'D11_hybrid_anti_dominance_s141', 'parameters': {'ambient': 0.5, 'shared': 0.56, 'diagnostic': 0.36, 'pair_refine': [('brownish', 'spotted', 0.03), ('greenish', 'discolored', 0.03), ('discolored', 'has_bark_texture', -0.06), ('greenish', 'has_grip_area', -0.06), ('brownish', 'has_crystal_flecks', -0.06), ('spotted', 'has_stem_remnant', -0.06)], 'clip_lo': 0.24, 'clip_hi': 0.66, 'negative_mode': 'diagnostic_flat_fail', 'salt': 141}, 'test_type': 'A', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- best_candidate_test_A: {'variant_id': 'D11_hybrid_anti_dominance_s141', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- best_candidate_test_B: {'variant_id': 'D8_anti_single_cue_balanced_s207', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}
- candidates_passing_primary: 0
- candidates_passing_partial: 16

# Files Changed

- _block1j40b7j_d_family_anti_dominance_refinement_audit.py

# Commands Run

- `python _block1j40b7j_d_family_anti_dominance_refinement_audit.py`

# Why 7j Was Run

- 7i improved raw oracle headroom but left positive rate too high and single-cue predictiveness at 1.0.
- 7j narrows to anti-dominance D-family variants only, with no broad family reruns.

# 7i Recap

- D4 and D6 preserved `always_observe > always_try` and opened oracle gap.
- But no compositional level beat `always_observe`, positive rate stayed around 0.75, and diagnostic single cues remained too predictive.

# Candidate Variants

- `D8_anti_single_cue_balanced_s141`: D8 anti-single-cue balanced refinement with fixed salt 141
- `D8_anti_single_cue_balanced_s207`: D8 anti-single-cue balanced refinement with fixed salt 207
- `D8_anti_single_cue_balanced_s319`: D8 anti-single-cue balanced refinement with fixed salt 319
- `D9_pairwise_dominant_single_cue_ambiguous_s141`: D9 pairwise-dominant refinement with ambiguous single-cue marginals
- `D10_candidate_set_balanced_s141`: D10 candidate-set balanced refinement derived from D6
- `D10_candidate_set_balanced_s207`: D10 candidate-set balanced refinement derived from D6
- `D11_hybrid_anti_dominance_s141`: D11 hybrid anti-dominance refinement combining D8 group priors and mild pair refinement
- `D11_hybrid_anti_dominance_s207`: D11 hybrid anti-dominance refinement combining D8 group priors and mild pair refinement

# Test A Results

- `D8_anti_single_cue_balanced_s141`: oracle_gap=0.1094, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.5111, rate=0.5111, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s207`: oracle_gap=0.1109, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.5056, rate=0.5056, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s319`: oracle_gap=0.1109, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.5056, rate=0.5056, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D9_pairwise_dominant_single_cue_ambiguous_s141`: oracle_gap=0.1094, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.5111, rate=0.5111, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s141`: oracle_gap=0.1101, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.5056, rate=0.5056, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s207`: oracle_gap=0.1126, ao_minus_at=0.0667, best=level1, best_delta=0.0000, sign=0.4833, rate=0.4833, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s141`: oracle_gap=0.1053, ao_minus_at=0.0800, best=level1, best_delta=0.0000, sign=0.5167, rate=0.5167, single=0.7333, pair=0.7667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s207`: oracle_gap=0.1068, ao_minus_at=0.0733, best=level1, best_delta=0.0000, sign=0.4944, rate=0.4944, single=0.8667, pair=0.8667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive

# Test B Results

- `D8_anti_single_cue_balanced_s141`: oracle_gap=0.1166, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4556, rate=0.4556, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s207`: oracle_gap=0.1132, ao_minus_at=0.0700, best=level1, best_delta=0.0000, sign=0.4778, rate=0.4778, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s319`: oracle_gap=0.1100, ao_minus_at=0.0633, best=level1, best_delta=0.0000, sign=0.4444, rate=0.4444, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D9_pairwise_dominant_single_cue_ambiguous_s141`: oracle_gap=0.1166, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4556, rate=0.4556, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s141`: oracle_gap=0.1166, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4556, rate=0.4556, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s207`: oracle_gap=0.1167, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4444, rate=0.4444, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s141`: oracle_gap=0.1166, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4556, rate=0.4556, single=0.7333, pair=0.7667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s207`: oracle_gap=0.1094, ao_minus_at=0.0700, best=level1, best_delta=0.0000, sign=0.4556, rate=0.4556, single=0.8667, pair=0.8667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive

# Test C Results

- `D8_anti_single_cue_balanced_s141`: oracle_gap=0.0974, ao_minus_at=0.0567, best=level2, best_delta=0.0005, sign=0.5167, rate=0.5111, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s207`: oracle_gap=0.0987, ao_minus_at=0.0533, best=level1, best_delta=0.0000, sign=0.4889, rate=0.4889, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D8_anti_single_cue_balanced_s319`: oracle_gap=0.0942, ao_minus_at=0.0533, best=level1, best_delta=0.0000, sign=0.4889, rate=0.4889, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D9_pairwise_dominant_single_cue_ambiguous_s141`: oracle_gap=0.0974, ao_minus_at=0.0567, best=level2, best_delta=0.0005, sign=0.5167, rate=0.5111, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s141`: oracle_gap=0.0982, ao_minus_at=0.0567, best=level2, best_delta=0.0005, sign=0.5111, rate=0.5056, single=0.7333, pair=0.7333, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D10_candidate_set_balanced_s207`: oracle_gap=0.0993, ao_minus_at=0.0500, best=level1, best_delta=0.0000, sign=0.4667, rate=0.4667, single=0.8000, pair=0.8000, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s141`: oracle_gap=0.0945, ao_minus_at=0.0633, best=level2, best_delta=0.0005, sign=0.5222, rate=0.5167, single=0.7333, pair=0.7667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough
- `D11_hybrid_anti_dominance_s207`: oracle_gap=0.0940, ao_minus_at=0.0567, best=level1, best_delta=0.0000, sign=0.4778, rate=0.4778, single=0.8667, pair=0.8667, rej=best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive

# Structure Validity Audit

- `D8_anti_single_cue_balanced_s141|A`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D8_anti_single_cue_balanced_s141|B`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D8_anti_single_cue_balanced_s207|A`: exact_det=0.0000, single_cue_max=0.8000, pair_max=0.8000, reachability=1.0000
- `D8_anti_single_cue_balanced_s207|B`: exact_det=0.0000, single_cue_max=0.8000, pair_max=0.8000, reachability=1.0000
- `D8_anti_single_cue_balanced_s319|A`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D8_anti_single_cue_balanced_s319|B`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D9_pairwise_dominant_single_cue_ambiguous_s141|A`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D9_pairwise_dominant_single_cue_ambiguous_s141|B`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D10_candidate_set_balanced_s141|A`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D10_candidate_set_balanced_s141|B`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7333, reachability=1.0000
- `D10_candidate_set_balanced_s207|A`: exact_det=0.0000, single_cue_max=0.8000, pair_max=0.8000, reachability=1.0000
- `D10_candidate_set_balanced_s207|B`: exact_det=0.0000, single_cue_max=0.8000, pair_max=0.8000, reachability=1.0000
- `D11_hybrid_anti_dominance_s141|A`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7667, reachability=1.0000
- `D11_hybrid_anti_dominance_s141|B`: exact_det=0.0000, single_cue_max=0.7333, pair_max=0.7667, reachability=1.0000
- `D11_hybrid_anti_dominance_s207|A`: exact_det=0.0000, single_cue_max=0.8667, pair_max=0.8667, reachability=1.0000
- `D11_hybrid_anti_dominance_s207|B`: exact_det=0.0000, single_cue_max=0.8667, pair_max=0.8667, reachability=1.0000

# Best Candidate

- {'variant_id': 'D11_hybrid_anti_dominance_s141', 'parameters': {'ambient': 0.5, 'shared': 0.56, 'diagnostic': 0.36, 'pair_refine': [('brownish', 'spotted', 0.03), ('greenish', 'discolored', 0.03), ('discolored', 'has_bark_texture', -0.06), ('greenish', 'has_grip_area', -0.06), ('brownish', 'has_crystal_flecks', -0.06), ('spotted', 'has_stem_remnant', -0.06)], 'clip_lo': 0.24, 'clip_hi': 0.66, 'negative_mode': 'diagnostic_flat_fail', 'salt': 141}, 'test_type': 'A', 'observe_cost': 0.03, 'best_level': 'level1', 'best_level_delta_vs_always_observe': 0.0}

# Partial Candidates

- [{'variant_id': 'D8_anti_single_cue_balanced_s141', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D8_anti_single_cue_balanced_s141', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D8_anti_single_cue_balanced_s207', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D8_anti_single_cue_balanced_s207', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D8_anti_single_cue_balanced_s319', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D8_anti_single_cue_balanced_s319', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D9_pairwise_dominant_single_cue_ambiguous_s141', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D9_pairwise_dominant_single_cue_ambiguous_s141', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D10_candidate_set_balanced_s141', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D10_candidate_set_balanced_s141', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D10_candidate_set_balanced_s207', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D10_candidate_set_balanced_s207', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D11_hybrid_anti_dominance_s141', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D11_hybrid_anti_dominance_s141', 'test_type': 'B', 'observe_cost': 0.03}, {'variant_id': 'D11_hybrid_anti_dominance_s207', 'test_type': 'A', 'observe_cost': 0.03}, {'variant_id': 'D11_hybrid_anti_dominance_s207', 'test_type': 'B', 'observe_cost': 0.03}]

# Rejected Variants and Reasons

- `best_level_does_not_beat_always_observe;sign_correct_not_enough`: 14
- `best_level_does_not_beat_always_observe;sign_correct_not_enough;single_cue_too_predictive`: 2

# Validity / Leakage Audit

- no learner training: true
- no 1J40b-8: true
- no official env replacement: true
- forbidden test keys absent: true
- audit only: true

# Recommended Resume Point

- If status remains partial: one more targeted refinement only if you want to search for a compositional policy that actually beats `always_observe`.
- If no level beats `always_observe` cleanly after anti-dominance fixes: prefer deeper environment redesign or expert review before any architecture work.