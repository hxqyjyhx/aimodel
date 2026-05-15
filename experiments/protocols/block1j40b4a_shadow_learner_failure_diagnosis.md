# Block 1J40b-4a: Shadow Learner Failure Diagnosis

- **Type**: Controlled failure diagnosis
- **Elapsed**: 2.4s

## D1: Pre-observe Ceiling

- pre mean true return: 0.4500
- post mean true return: 0.4067
- oracle mean true return: 0.4500
- pre/oracle gap: 0.0000
- post/oracle gap: 0.0433
- ceiling reached: True

## D2: Category Identifiability from Ambient Features

- category probe accuracy: 1.0000
- best-base-action probe accuracy: 1.0000
- base affordance profile probe accuracy: 1.0000
- too informative: True

## D3: Best-Action Change After Observe

- best action change rate: 0.2167
- mean true value gain: -0.0433
- positive gain rate: 0.0000
- no room for improvement: False

## D4: Env2 Audit Metric vs Shadow Metric

- env2 measures: mean try return across ALL actions
- shadow measures: best-action-selected return
- env2 pre mean try return: 0.0750
- env2 post mean try return: 0.1083
- env2 improvement: +0.0333
- shadow pre best-action return: 0.4500
- shadow post best-action return: 0.4067

## D5 (Optional): Separate Pre/Post Model Calibration

- avg pre return: 0.4500
- avg shadow return: 0.4490
- avg delta: -0.0010
- avg obs rate: 0.2000
- improves selected return: False
- changes core conclusion: False

## Raw Conclusion

- **env2 as shadow observe-value testbed: PARTIAL**
- Pre-observe ambient features are sufficient to identify category and base affordance.
- Post-observe diagnostic features add variant-profile info but roles are deconfounded.
- Best-action selection cannot improve via observation in this environment.

## Recommended Next Step

Redesign pre-observe features to reduce ambient informativeness, or change the learner metric from best-action selection to expected value of information.
