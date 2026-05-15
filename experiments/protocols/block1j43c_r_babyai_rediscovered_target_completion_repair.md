# 1J43c-r BabyAI Rediscovered Target Completion Repair

## Checkpoint Summary
- Status: `PARTIAL`
- Package: `minigrid 3.1.0`
- Envs tested: `BabyAI-GoToObj-v0, BabyAI-GoToRedBall-v0, BabyAI-Pickup-v0, BabyAI-OpenDoor-v0`
- Episodes per env: `20`

## Files Changed
- `_block1j43c_r_babyai_rediscovered_target_completion_repair.py`

## Commands Run
- `python _block1j43c_r_babyai_rediscovered_target_completion_repair.py`

## Adapter Coverage
- Reset success: `1.0000`
- Step success: `1.0000`
- Observation conversion: `1.0000`
- Action conversion: `1.0000`
- Outcome conversion: `1.0000`
- Mission parse success: `1.0000`

## Policy Probe
- Random return: `0.1603`
- 1J43b probe return: `0.1875`
- 1J43c return: `0.2151`
- 1J43c-r return: `0.2189`
- Delta vs random: `+0.0586`
- Delta vs 1J43c: `+0.0038`
- Success proxy rate: `0.2375`
- Visible target success rate: `0.3306`
- Target-not-visible fallback success rate: `0.0000`
- Target discovery rate: `0.7817`

## Rediscovery / Handoff
- Target rediscovered count: `8.4167`
- Rediscovered to bridge handoff rate: `1.0000`
- Rediscovered to adjacent rate: `0.1730`
- Rediscovered to interaction attempt rate: `0.1503`
- Rediscovered to success rate: `0.2963`
- Bridge control stolen by fallback count: `0.0000`

## Exploration
- Repeated observation rate: `0.9483`
- Stuck loop count: `60.5833`
- Blocked forward count: `477.2500`
- New observation signature rate: `0.1609`
- Timeout rate: `0.7625`
- Main exploration failure: `rediscovered targets now hand off, but completion still fails when local approach or interaction selection breaks down`

## Pickup/OpenDoor Failure Breakdown
- Not approached after rediscovery: `1.6667`
- Adjacent not facing: `0.0000`
- Facing no interaction: `0.0000`
- Interaction no reward: `0.0000`
- Timeout after rediscovery: `1.6667`

## Mechanism
- Event memory: `2182.0000`
- Object memory: `29.3333`
- Outcome memory: `271.5000`
- Common sense records: `7.1667`
- Experience records: `7.1667`
- Goal-context coverage: `1.0000`
- Confidence update available: `true`
- Boundary rule available: `true`

## Observation / Target
- Target visible rate: `0.5020`
- Target adjacent rate: `0.1074`
- Action bridge coverage: `0.5020`
- Unsupported missions: `0`
- Unsupported mission examples: `[]`

## Validity
- Hidden leakage: `false`
- Audit-only fields absent: `true`
- Legal observation only: `true`
- Mission parser used: `true`
- Language learning claim: `no`
- Performance claim: `no`
- Three-world separation: `true`

## Limitations
- Main remaining limitation: local novelty exploration still lacks long-horizon navigation memory.
- Sparse reward limitation: many episodes still end with zero reward because the probe is not a planner.
- No explicit observe limitation: BabyAI exposes passive legal observation only.
- Mission parser limitation: template parsing only, not language learning.
- Reason this is not benchmark success: nonzero usable behavior is enough here; no benchmark score claim is made.
