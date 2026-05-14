# 1J40b-7k-c Locked-Rule Confirmation

## Checkpoint Summary
- status: PARTIAL_LOCKED_VALIDATION
- decision: 7kc_PARTIAL_NEEDS_ENV_DESIGN_REVIEW
- validation_mode: locked_confirmation
- locked_from: 1J40b-7k-b

## Files Changed
- _block1j40b7kc_locked_rule_confirmation.py
- block1j40b7kc_locked_rule_confirmation.json
- block1j40b7kc_locked_rule_confirmation.md
- block1j40b7kc_locked_rule_confirmation_table.csv
- checkpoint_1j40b7kc_locked_rule_confirmation.md

## Commands Run
- D:\conda\python.exe _block1j40b7kc_locked_rule_confirmation.py

## Why 7k-c Was Run
- 7k-b reached design-search PASS, but the split was selected by search. 7k-c checks whether the locked scaffold remains stable without any new split tuning.

## Locked Configuration
- generator_mode: factorized_base
- diag_alpha: 0.75
- level3_rule: m1
- primary cost: 0.03
- locked 7k-b split idx: 3985
- locked Test A: [(0, 1, 0, 0), (1, 0, 0, 0)]
- locked Test B: [(0, 1, 0, 1), (1, 0, 0, 1)]

## Mode A: Fixed 7k-b Split, Fresh Seeds
- seed count: 10
- seed pass rate: 1.0000
- Test A Level 2 mean delta: 0.0331
- Test A Level 3 mean delta: 0.0177
- Test B Level 2 mean delta: 0.0352
- Test B Level 3 mean delta: 0.0352
- Level 1 max delta: 0.0000
- null Level 2/3 max delta: 0.0000

## Mode B: Pre-Registered Split Subset
- split selection rule: all pair-covered splits -> balanced Test A/B by factor values -> lexicographic quantiles
- split count: 5
- split pass rate: 0.0000
- mean best Level 2/3 delta: 0.0128
- median best Level 2/3 delta: 0.0000
- min best Level 2/3 delta: 0.0000
- null collapse rate: 1.0000
- worst split: {'split_label': 'balanced_lex_idx_13', 'best_A': -0.04481323343549404, 'best_B': 0.0, 'level1_max': -0.032741858863953976, 'null_max': 0.0, 'pass_flag': False}

## Validity
- exact_signature_determinism_rate: 0.0000
- single_cue_max_predictiveness: 0.6132
- cue_pair_max_predictiveness: 0.7362
- hidden_h_probe_from_pre_surface_accuracy: 0.5978
- train_pair_coverage: 1.0000

## Main Diagnosis
- Fixed-split fresh-seed stability is strong at the locked primary cost.
- Broader robustness over a neutral pre-registered balanced split subset is weak.
- Level 1 shortcut does not appear, and null-control remains clean.
- Therefore the locked scaffold looks locally stable but not broadly split-robust.

## Recommended Resume Point
- env3c design draft only if you accept local locked-split stability as sufficient evidence; otherwise do scaffold redesign or expert review first.
