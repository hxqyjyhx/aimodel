# 1J40b-7k-b Minimal Scaffold Refinement

## Checkpoint Summary
- status: PASS
- validation_mode: design_search
- decision: 7kb_DESIGN_SEARCH_PASS_NOT_FINAL_VALIDATION
- selected_config_id: factorized_base_a075_m1
- selected_split_idx: 3985

## Files Changed
- _block1j40b7kb_minimal_factorized_scaffold_refinement.py
- block1j40b7kb_minimal_factorized_scaffold_refinement.json
- block1j40b7kb_minimal_factorized_scaffold_refinement.md
- block1j40b7kb_minimal_factorized_scaffold_refinement_table.csv
- checkpoint_1j40b7kb_minimal_factorized_scaffold_refinement.md

## Commands Run
- D:\conda\python.exe _block1j40b7kb_minimal_factorized_scaffold_refinement.py

## Why 7k-b Was Run
- 7k was PARTIAL: Level 3 worked on Test A but not Test B, exact-signature determinism stayed above target, and split provenance needed to be made explicit.
- 7k-b refines only the minimal scaffold. It does not build env3c, does not touch env3b, and does not train a learner.

## Search Protocol
- Predeclared config count: 7
- Pair-covered split search space: 5388
- Split was chosen by exhaustive design-search over pair-covered A/B assignments.
- This result must be treated as design-search PASS, not final validation PASS.

## Selected Candidate
- config_id: factorized_base_a075_m1
- generator_mode: factorized_base
- diag_alpha: 0.75
- level3_rule: m1
- mix: 1.0
- Test A signatures: [[0, 1, 0, 0], [1, 0, 0, 0]]
- Test B signatures: [[0, 1, 0, 1], [1, 0, 0, 1]]

## Structured Primary Results (cost=0.03)
- Test A Level 1 delta_vs_always_observe: 0.0000
- Test A Level 2 delta_vs_always_observe: 0.0347
- Test A Level 3 delta_vs_always_observe: 0.0347
- Test B Level 1 delta_vs_always_observe: 0.0000
- Test B Level 2 delta_vs_always_observe: 0.0356
- Test B Level 3 delta_vs_always_observe: 0.0356

## Null-Control Primary Results (cost=0.03)
- Test A Level 2 delta_vs_always_observe: 0.0000
- Test A Level 3 delta_vs_always_observe: 0.0000
- Test B Level 2 delta_vs_always_observe: 0.0000
- Test B Level 3 delta_vs_always_observe: 0.0000

## Validity Audit
- exact_signature_determinism_rate: 0.0000
- single_cue_max_predictiveness: 0.6175
- cue_pair_max_predictiveness: 0.7420
- hidden_h_probe_from_pre_surface_accuracy: 0.5979
- train_pair_coverage_rate: 1.0000

## Structured vs Null Marginal Matching
- positive_net_rate_delta: -0.000000
- always_observe_delta: -0.000000
- always_try_delta: 0.100758

## Acceptance Decision
- PASS criteria are satisfied on the selected design-search split.
- Level 3 now beats always_observe on both Test A and Test B at the primary cost.
- Level 1 does not beat always_observe.
- Null Level 2/3 collapses to zero headroom.
- Because the split was selected by search, treat this as a scaffold design-search PASS and not as a final locked validation result.

## Recommended Resume Point
- If needed, run one fixed-split confirmation pass next. Otherwise, this is sufficient evidence to justify a full env3c design draft, still before any 1J40b-8 work.
