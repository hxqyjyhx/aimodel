# 1J41d Lightweight Goal Context

## Checkpoint Summary
- status: PASS
- structured score: 0.7800
- bridge-required score: 0.7800
- boundary score: 0.8200
- null-control score: 0.6400
- delta vs 1J41c structured: -0.0000
- delta vs similarity_only: 0.2100

## Files Changed
- _block1j41d_lightweight_goal_context.py
- block1j41d_lightweight_goal_context.json
- block1j41d_lightweight_goal_context.md
- block1j41d_lightweight_goal_context_table.csv
- checkpoint_1j41d_lightweight_goal_context.md

## Why 1J41d Was Run
- 1J41c showed that explicit Event/Object/Outcome/Common Sense/Experience records preserve 1J41b performance.
- 1J41d keeps the same legal mechanism but adds lightweight goal_context fields so Experience is goal-conditioned before any benchmark adapter dry-run.

## Main Result
- structured score: 0.7800
- bridge-required score: 0.7800
- boundary score: 0.8200
- null-control score: 0.6400
- delta vs 1J41c: structured -0.0000, bridge -0.0000, boundary 0.0000, null 0.0000
- delta vs similarity_only: 0.2100

## Memory Records
- event memory count: 248.0000
- object memory count: 36.0000
- outcome memory count: 72.0000
- common sense record count: 5.0000
- experience record count: 5.0000
- average support cases per record: 24.8000
- average failure cases per record: 1.6000
- boundary rule count: 11.0000
- bridge-supported record count: 1.0000

## Goal Context
- experience records: 5.0000
- records with goal_context: 5.0000
- goal_context coverage: 1.0000
- goal_context count: 3.0000
- goal_context distribution: {"avoid_boundary_failure": 3, "maximize_net_result": 9, "reduce_uncertainty": 3}
- goal-conditioned success rate: 0.9376
- goal-conditioned false application rate: 0.0000
- goal confidence-success correlation: 0.7089

## Common Sense
- support rate: 0.9376
- failure rate: 0.0624
- boundary coverage: 0.2000
- bridge coverage: 0.2000
- audit purity: 1.0000
- common sense records with relevant_goal_contexts: 5.0000
- average linked common sense per experience: 1.0000
- goal_context link consistency: 1.0000

## Experience
- trigger precision: 1.0000
- trigger recall: 1.0000
- success rate: 0.7800
- false application rate: 0.0000
- confidence-success correlation: 0.0000

## Boundary
- overgeneralization before: 1.0000
- overgeneralization after: 0.0000
- confidence drop after boundary failure: 0.3704
- boundary block rate: 1.0000

## Bridge
- no bridge: 0.5000
- with bridge: 0.7800
- bridge delta: 0.2800
- bridge merge success rate: 1.0000
- bridge-supported goal-context records: 1.0000

## Null-control
- false discovery rate: 0.0000
- false goal-context count: 0.0000
- stable false record count: 0.0000
- exposure curve present: false

## Validity
- hidden leakage: false
- audit-only fields absent from agent records: true
- three-world separation passed: true

## Recommended Resume Point
- Prepare a benchmark adapter dry-run if you want to test this goal-conditioned Experience layout in a thin external interface.
