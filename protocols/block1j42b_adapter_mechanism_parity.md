# 1J42b Adapter Mechanism Parity

## Checkpoint Summary
- status: PASS
- benchmark name: SparseBridgeBenchmarkWrapper
- episode count: 116.0000
- explicit observe action: true
- observation conversion success: 1.0000
- action conversion success: 1.0000
- outcome conversion success: 1.0000
- goal_context assignment rate: 1.0000

## Files Changed
- _block1j42b_adapter_mechanism_parity.py
- block1j42b_adapter_mechanism_parity.json
- block1j42b_adapter_mechanism_parity.md
- block1j42b_adapter_mechanism_parity_table.csv
- checkpoint_1j42b_adapter_mechanism_parity.md

## Why 1J42b Was Run
- 1J42a validated legal adapter dataflow but did not preserve confidence updates, boundary rules, or usable policy behavior.
- 1J42b restores those mechanism-side updates inside the reset/step loop while keeping the no-performance-claim constraint.

## Adapter Coverage
- observation conversion success: 1.0000
- action conversion success: 1.0000
- outcome conversion success: 1.0000
- goal_context assignment rate: 1.0000
- unsupported observation fields: []
- unsupported action types: []

## Memory Records
- event memory count: 109.3333
- object memory count: 62.0000
- outcome memory count: 47.6667
- common sense record count: 7.3333
- experience record count: 7.3333
- records with goal_context: 7.3333
- goal_context coverage: 1.0000

## Mechanism Parity
- strategy candidate count: 7.3333
- experience trigger count: 16.0000
- false application proxy: 0.3333
- confidence update available: true
- confidence update count: 47.0000
- confidence before mean: 0.7610
- confidence after mean: 0.5629
- confidence-success correlation: 0.1728
- boundary rule available: true
- boundary rule count: 16.0000
- boundary block rate: 1.0000
- bridge signal available: true
- bridge-supported experiences: 2.0000

## Policy Probe
- random baseline return: 0.5517
- 1J42a simple policy return: 0.3733
- 1J42b Experience policy return: 0.5733
- delta vs random: 0.0217
- delta vs 1J42a simple policy: 0.2000
- always observe return: 0.9200
- always direct return: 0.4000
- performance claim: no

## Boundary
- overgeneralization before: 0.2778
- overgeneralization after: 0.0000
- false block rate: 0.3333

## Validity
- hidden leakage: false
- audit-only fields absent: true
- legal observation only: true
- three-world separation: true
- no test label tuning: true

## Limitations
- main adapter limitation: toy benchmark-style wrapper rather than a large external benchmark package
- missing observe/action issue: none; wrapper includes explicit inspect action
- missing boundary/bridge issue: bridge and boundary are present, but still low-dimensional compared with a full benchmark
- benchmark mismatch issue: observation space remains compact and object identity is clearer than in larger benchmarks
- adapter assumptions: ['legal observation arrives as visible token tuples plus optional inspect diagnostic', 'inspect is an explicit legal action in the wrapper', 'goal_context is derived from reward/task-success proxies only']

## Recommended Resume Point
- Move to external benchmark candidate selection; the adapter now preserves the local mechanism well enough for a small pilot.
