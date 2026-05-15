# 1J43b BabyAI Probe Repair / Minimal Usable Policy Probe

## Checkpoint Summary
- Status: `PASS`
- Package: `minigrid 3.1.0`
- Envs tested: `BabyAI-GoToObj-v0, BabyAI-GoToRedBall-v0, BabyAI-Pickup-v0, BabyAI-OpenDoor-v0`
- Episodes per env: `20`

## Files Changed
- `_block1j43b_babyai_probe_repair.py`

## Commands Run
- `python _block1j43b_babyai_probe_repair.py`

## Adapter Coverage
- Reset success: `1.0000`
- Step success: `1.0000`
- Observation conversion: `1.0000`
- Action conversion: `1.0000`
- Outcome conversion: `1.0000`
- Mission parse success: `1.0000`

## Policy Probe
- Random return: `0.1603`
- 1J43a simple probe return: `0.0229`
- Mission visible-object probe return: `0.1875`
- Experience plus mission probe return: `0.1875`
- Delta vs random: `+0.0272`
- Delta vs 1J43a probe: `+0.1646`
- Success proxy rate: `0.2000`
- Visible target success rate: `0.2770`
- Target-not-visible fallback success rate: `0.0000`

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
- Target visible rate: `0.4435`
- Target adjacent rate: `0.1036`
- Action bridge coverage: `0.4435`
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
- Main remaining limitation: local action bridge still cannot route around obstacles reliably under sparse passive observation.
- Sparse reward limitation: many episodes still end with zero reward because the probe is not a planner.
- No explicit observe limitation: BabyAI exposes passive legal observation only.
- Mission parser limitation: template parsing only, not language learning.
- Reason this is not benchmark success: nonzero usable behavior is enough here; no benchmark score claim is made.
