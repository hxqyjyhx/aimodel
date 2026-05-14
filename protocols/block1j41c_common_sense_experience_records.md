# 1J41c Common Sense / Experience Records

## Checkpoint Summary
- status: PASS
- structured score: 0.7800
- bridge-required score: 0.7800
- boundary score: 0.8200
- null-control score: 0.6400
- delta vs 1J41b structured: -0.0000
- delta vs similarity_only: 0.2100

## Files Changed
- _block1j41c_common_sense_experience_records.py
- block1j41c_common_sense_experience_records.json
- block1j41c_common_sense_experience_records.md
- block1j41c_common_sense_experience_records_table.csv
- checkpoint_1j41c_common_sense_experience_records.md

## Why 1J41c Was Run
- 1J41b showed that local strategy clustering, bridge merging, and boundary updates work.
- 1J41c keeps the same legal interaction mechanism but makes the learned knowledge explicit as Event Memory, Object Memory, Outcome Memory, Common Sense, and Experience records.

## Main Result
- structured score: 0.7800
- bridge-required score: 0.7800
- boundary score: 0.8200
- null-control score: 0.6400
- delta vs 1J41b: structured -0.0000, bridge -0.0000, boundary 0.0000, null 0.0000
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

## Common Sense
- support rate: 0.9376
- failure rate: 0.0624
- boundary coverage: 0.2000
- bridge coverage: 0.2000
- audit purity: 1.0000

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

## Null-control
- false discovery rate: 0.0000
- stable false record count: 0.0000
- exposure curve present: false

## Validity
- hidden leakage: false
- audit-only fields absent from agent records: true
- three-world separation passed: true

## Recommended Resume Point
- Prepare a benchmark adapter dry-run only if you want to keep the current local mechanism and record layout.
