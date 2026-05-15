# 1J42a Benchmark Adapter Dry-Run Checkpoint

- status: PASS
- benchmark name: SparseBridgeBenchmarkWrapper
- episode count: 116.0000
- explicit observe action: true
- event/object/outcome/common_sense/experience counts: 93.3/50.7/45.3/7.7/7.7
- adapter conversion success: obs 1.0000, act 1.0000, out 1.0000
- goal_context assignment rate: 1.0000
- random/simple/avg return: 0.5517/0.3733/0.4867
- limitations: {"main_adapter_limitation": "toy benchmark-style wrapper rather than a large external benchmark package", "missing_observe_action_issue": "none; wrapper includes explicit inspect action", "missing_boundary_bridge_issue": "bridge and boundary are present, but still low-dimensional compared with a full benchmark", "benchmark_mismatch_issue": "observation space remains compact and object identity is clearer than in larger benchmarks", "adapter_assumptions": ["legal observation arrives as visible token tuples plus optional inspect diagnostic", "inspect is an explicit legal action in the wrapper", "goal_context is derived from reward/task-success proxies only"]}

Next suggested step:
- choose a better benchmark or run a very small benchmark pilot
