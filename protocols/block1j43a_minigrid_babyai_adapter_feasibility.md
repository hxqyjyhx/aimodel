# 1J43a Minigrid-BabyAI Adapter Feasibility

## Checkpoint Summary
- status: PASS
- package used: minigrid 3.1.0
- envs tested: ['BabyAI-GoToObj-v0', 'BabyAI-GoToRedBall-v0', 'BabyAI-Pickup-v0', 'BabyAI-OpenDoor-v0']
- successful envs: ['BabyAI-GoToObj-v0', 'BabyAI-GoToRedBall-v0', 'BabyAI-OpenDoor-v0', 'BabyAI-Pickup-v0']
- failed envs: []

## Adapter Coverage
- reset success: 1.0000
- step success: 1.0000
- observation conversion: 1.0000
- action conversion: 1.0000
- outcome conversion: 1.0000
- mission parse success: 0.9236

## Memory Records
- event memory: 3206.2500
- object memory: 50.8333
- outcome memory: 696.9167
- common sense records: 9.6667
- experience records: 9.6667
- goal_context coverage: 1.0000

## Mechanism Compatibility
- strategy candidates: 9.6667
- experience triggers: 9.6667
- confidence update available: true
- boundary rule available: true
- explicit observe action: false
- passive observation available: true

## Policy Probe
- random return: 0.1536
- simple experience probe return: 0.0000
- delta vs random: -0.1536
- success proxy: 0.0000
- performance claim: no

## Validity
- hidden leakage: false
- audit-only fields absent: true
- legal observation only: true
- mission parser used: true
- language learning claim: no
- three-world separation: true

## Limitations
- main limitation: partial 7x7 egocentric observations and sparse reward make experience records thin
- mission limitation: template parser only; not a language model
- observe/action limitation: no explicit inspect action; passive legal observation only
- sparse reward limitation: many random episodes produce little or no terminal reward

## Recommended Resume Point
- BabyAI pilot adapter is feasible, but the next step should still be a small pilot rather than a performance benchmark claim.
