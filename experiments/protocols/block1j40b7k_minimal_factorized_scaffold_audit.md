# Checkpoint Summary

- status: PARTIAL
- primary observe_cost: 0.03
- best structured level: level3
- best structured delta_vs_always_observe: 0.0525

# Files Changed

- _block1j40b7k_minimal_factorized_scaffold_audit.py

# Commands Run

- `python _block1j40b7k_minimal_factorized_scaffold_audit.py`

# Why 7k Was Run

- 7j fixed dominance and single-cue leakage in env3b variants but still found no Level 2/3 headroom over always_observe.
- 7k tests the scaffold concept itself in a tiny factorized audit before any full env3c draft.

# Minimal Factorized Generator Design

- 4 binary visible factors -> 16 exact signatures.
- 3 hidden h values and 3 abstract actions.
- observe reveals a noisy diagnostic token informative about h.
- structured version uses pair-structured q=(f0 xor f1, f2 xor f3) to define P(h|f).
- null version uses the same global h marginals for every signature, breaking pair structure.

# Train/Test Split and Pair Coverage Audit

- total signatures: 16
- held-out Test A signatures: [[0, 0, 0, 0], [0, 0, 0, 1]]
- held-out Test B signatures: [[0, 0, 1, 1], [0, 1, 0, 1]]
- all 2-way pairs covered in training: True

# Structured Version Results

- `A|level1`: delta_vs_always_observe=-0.0475, policy_net=0.7869, sign_correct=0.1844
- `A|level2`: delta_vs_always_observe=-0.0475, policy_net=0.7869, sign_correct=0.1844
- `A|level3`: delta_vs_always_observe=0.0525, policy_net=0.8869, sign_correct=0.4844
- `B|level1`: delta_vs_always_observe=-0.1100, policy_net=0.6130, sign_correct=0.5344
- `B|level2`: delta_vs_always_observe=-0.1100, policy_net=0.6130, sign_correct=0.5344
- `B|level3`: delta_vs_always_observe=-0.1100, policy_net=0.6130, sign_correct=0.5344

# Null-Control Results

- `A|level1`: delta_vs_always_observe=0.0000, policy_net=0.7387, sign_correct=0.5625
- `A|level2`: delta_vs_always_observe=0.0000, policy_net=0.7387, sign_correct=0.5625
- `A|level3`: delta_vs_always_observe=0.0000, policy_net=0.7387, sign_correct=0.5625
- `B|level1`: delta_vs_always_observe=0.0000, policy_net=0.7517, sign_correct=0.5594
- `B|level2`: delta_vs_always_observe=0.0000, policy_net=0.7517, sign_correct=0.5594
- `B|level3`: delta_vs_always_observe=0.0000, policy_net=0.7517, sign_correct=0.5594

# Level 1 vs Level 2/3 Comparison

- structured Level 1 best delta: -0.0475
- structured Level 2 best delta: -0.0475
- structured Level 3 best delta: 0.0525

# Leakage / Shortcut Audit

- exact_signature_determinism_rate: 0.1875
- single_cue_max_predictiveness: 0.5523
- cue_pair_max_predictiveness: 0.8453
- hidden_h_probe_from_pre_surface_accuracy: 0.7309

# Structured vs Null Marginal Matching

- `cost_0.03_A`: positive_rate_delta=-0.0160, always_observe_delta=0.0956, always_try_delta=0.0506
- `cost_0.03_B`: positive_rate_delta=-0.0160, always_observe_delta=-0.0287, always_try_delta=-0.1362
- `cost_0.05_A`: positive_rate_delta=-0.1258, always_observe_delta=0.0956, always_try_delta=0.0506
- `cost_0.05_B`: positive_rate_delta=-0.1258, always_observe_delta=-0.0287, always_try_delta=-0.1362
- `cost_0.07_A`: positive_rate_delta=-0.2309, always_observe_delta=0.0956, always_try_delta=0.0506
- `cost_0.07_B`: positive_rate_delta=-0.2309, always_observe_delta=-0.0287, always_try_delta=-0.1362
- `cost_0.10_A`: positive_rate_delta=-0.2309, always_observe_delta=0.0956, always_try_delta=0.0506
- `cost_0.10_B`: positive_rate_delta=-0.2309, always_observe_delta=-0.0287, always_try_delta=-0.1362

# Acceptance Decision

- status: PARTIAL
- level1_too_strong: False
- null_invalid: False
- recommended_next_step: refine 7k scaffold

# Recommended Resume Point

- PASS -> full env3c design draft
- PARTIAL -> refine 7k scaffold
- FAIL -> expert review / rethink scaffold