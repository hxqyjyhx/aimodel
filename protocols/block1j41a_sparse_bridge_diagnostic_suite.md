# Sparse-Bridge Diagnostic Suite v0

## Checkpoint Summary
- status: PASS
- primary baseline: level3_cluster
- structured exposure curve present: True
- bridge effect present: True
- null exposure curve present: False

## Files Changed
- _block1j41a_sparse_bridge_diagnostic_suite.py
- block1j41a_sparse_bridge_diagnostic_suite.json
- block1j41a_sparse_bridge_diagnostic_suite.md
- block1j41a_sparse_bridge_diagnostic_suite_table.csv
- checkpoint_1j41a_sparse_bridge_diagnostic_suite.md

## Why This Was Run
- This suite replaces the old split-centered env3c route with a much smaller sparse-relation diagnostic scaffold.
- It measures whether recoverable relation signal appears only after enough related and bridge evidence is exposed.

## Environment
- total samples: 196
- unrelated samples: 96
- related samples: 16
- bridge samples: 4
- boundary samples: 16
- null-control present: yes

## Exposure Curve (Level 3, same_family_test, n_bridge=0)
- n_related = 0: 0.5000
- n_related = 1: 0.6750
- n_related = 2: 0.7450
- n_related = 4: 0.7800
- n_related = 8: 0.7800
- n_related = 16: 0.7800

## Bridge Effect (Level 3, bridge_required_test, n_related=16)
- no bridge: 0.5000
- with bridge: 0.7800
- delta: 0.2800

## Null-Control
- exposure curve present: no
- false discovery rate: 0.0000

## Boundary
- overgeneralization rate: 1.0000
- main failure mode: similarity-only and relation-estimate baselines over-apply the observe-worthwhile strategy to family-looking boundary cases.

## Validity
- hidden leakage: no
- audit-only fields absent from agent input: yes
- three-world separation passed: yes

## Main Diagnosis
- sparse relation signal exists in the structured suite
- bridge samples materially help bridge-required generalization
- null-control stays flat and does not show the same curve
- boundary cases expose overgeneralization of simple similarity-based discovery

## Recommended Resume Point
- add a simple strategy-clustering learner next, or do a benchmark-adapter dry-run only after that if needed.
