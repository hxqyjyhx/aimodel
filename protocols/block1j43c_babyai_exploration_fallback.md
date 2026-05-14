# 1J43c BabyAI Exploration Fallback

## Checkpoint Summary
- Status: `PASS`
- Package: `minigrid 3.1.0`
- Envs tested: `BabyAI-GoToObj-v0, BabyAI-GoToRedBall-v0, BabyAI-Pickup-v0, BabyAI-OpenDoor-v0`
- Episodes per env: `20`

## Files Changed
- `_block1j43c_babyai_exploration_fallback.py`

## Commands Run
- `python _block1j43c_babyai_exploration_fallback.py`

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
- Exploration fallback return: `0.2151`
- Delta vs random: `+0.0548`
- Delta vs 1J43b: `+0.0276`
- Success proxy rate: `0.2333`
- Visible target success rate: `0.3264`
- Target-not-visible fallback success rate: `0.0000`
- Target discovery rate: `0.7817`
- Steps until target visible: `6.3436`

## Exploration
- Repeated observation rate: `0.9480`
- Stuck loop count: `63.6667`
- Blocked forward count: `478.4167`
- New observation signature rate: `0.1594`
- Timeout rate: `0.7667`
- Main exploration failure: `target-not-visible episodes still rely on local novelty only and cannot route around obstacles globally`

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
- Target visible rate: `0.5003`
- Target adjacent rate: `0.1064`
- Action bridge coverage: `0.5003`
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
