# Phase 2 Object Candidate Memory ¡ª Shadow Summary

- **Block**: 1J35
- **Seed**: 109
- **Elapsed**: 2.0s

## Candidate Statistics

| Metric | Value |
|--------|-------|
| Total candidates | 29 |
| Provisional candidates | 22 |
| Stable candidates | 7 |
| Buffer size | 22 |
| Average support count | 2.07 |
| Max support count | 4 |
| Conflict candidates | 21 |
| Single-event promotions | 0 |
| Hidden feature leakage | False |
| Oracle leakage | False |
| Events processed | 120 |
| Objects tracked | 60 |

## Stable Candidates

### cand_0001

- Support: 2, Episode spread: 2
- Confidence: 1.6667, Stability: 0.8333
- Positive features (12): `['block_like', 'brownish', 'elongated_with_handle', 'grayish', 'has_stem_remnant', 'long_shape', 'movable', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'solid']`
- Contradicted: {'block_like': 1, 'has_stem_remnant': 1}
- Object count: 2

### cand_0005

- Support: 2, Episode spread: 2
- Confidence: 1.5000, Stability: 0.7857
- Positive features (14): `['block_like', 'grayish', 'has_granular_surface', 'has_grip_area', 'has_wood_grain', 'heavy_weight', 'light_weight', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'smooth_texture', 'solid']`
- Contradicted: {'has_wood_grain': 1, 'on_left_side': 1, 'near_table': 1}
- Object count: 2

### cand_0008

- Support: 2, Episode spread: 2
- Confidence: 1.4615, Stability: 0.7692
- Positive features (13): `['block_like', 'brownish', 'has_bark_texture', 'has_stem_remnant', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'on_left_side', 'recently_seen', 'rough_texture', 'solid']`
- Contradicted: {'solid': 1, 'has_bark_texture': 1, 'has_stem_remnant': 1}
- Object count: 2

### cand_0009

- Support: 2, Episode spread: 2
- Confidence: 1.5000, Stability: 0.7857
- Positive features (14): `['block_like', 'brownish', 'elongated_with_handle', 'grayish', 'has_grip_area', 'has_shaft_shape', 'has_wood_grain', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'rough_texture', 'solid']`
- Contradicted: {'long_shape': 1, 'movable': 1, 'near_table': 1}
- Object count: 2

### cand_0006

- Support: 2, Episode spread: 2
- Confidence: 1.4286, Stability: 0.7857
- Positive features (14): `['brownish', 'grayish', 'greenish', 'has_grip_area', 'has_shaft_shape', 'has_wood_grain', 'light_weight', 'movable', 'near_table', 'recently_seen', 'rough_texture', 'round_small', 'smooth_texture', 'solid']`
- Contradicted: {'grayish': 1, 'has_wood_grain': 1, 'light_weight': 1}
- Object count: 2

### cand_0015

- Support: 2, Episode spread: 2
- Confidence: 1.5000, Stability: 0.7143
- Positive features (14): `['brownish', 'has_crystal_flecks', 'has_peel_texture', 'has_shaft_shape', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'on_left_side', 'recently_seen', 'round_small', 'smooth_texture', 'solid']`
- Contradicted: {'has_peel_texture': 1, 'smooth_texture': 1, 'heavy_weight': 1, 'has_crystal_flecks': 1}
- Object count: 2

### cand_0020

- Support: 2, Episode spread: 2
- Confidence: 1.5000, Stability: 0.8333
- Positive features (12): `['brownish', 'has_bark_texture', 'has_crystal_flecks', 'has_granular_surface', 'has_grip_area', 'has_shaft_shape', 'heavy_weight', 'long_shape', 'movable', 'round_small', 'smooth_texture', 'solid']`
- Contradicted: {'has_grip_area': 1, 'has_shaft_shape': 1}
- Object count: 2

## Buffer (Provisional) Candidates

- **cand_0000**: support=3, stability=0.5263, features=['block_like', 'brownish', 'elongated_with_handle', 'greenish', 'has_bark_texture', 'has_granular_surface', 'has_grip_area', 'has_peel_texture', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'round_small', 'smooth_texture', 'solid']
- **cand_0002**: support=3, stability=0.5625, features=['block_like', 'brownish', 'elongated_with_handle', 'greenish', 'has_bark_texture', 'has_granular_surface', 'has_stem_remnant', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'on_left_side', 'recently_seen', 'smooth_texture', 'solid']
- **cand_0003**: support=1, stability=1.0000, features=['has_granular_surface', 'heavy_weight', 'long_shape', 'near_table', 'recently_seen', 'solid']
- **cand_0004**: support=1, stability=1.0000, features=['grayish', 'has_shaft_shape', 'has_wood_grain', 'near_table', 'recently_seen', 'rough_texture', 'solid']
- **cand_0007**: support=3, stability=0.5556, features=['block_like', 'brownish', 'elongated_with_handle', 'grayish', 'has_bark_texture', 'has_grip_area', 'has_peel_texture', 'has_stem_remnant', 'has_wood_grain', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'smooth_texture', 'solid']
- **cand_0010**: support=2, stability=0.6364, features=['block_like', 'brownish', 'grayish', 'greenish', 'light_weight', 'long_shape', 'movable', 'recently_seen', 'round_small', 'smooth_texture', 'solid']
- **cand_0011**: support=4, stability=0.5238, features=['block_like', 'brownish', 'elongated_with_handle', 'grayish', 'greenish', 'has_bark_texture', 'has_granular_surface', 'has_peel_texture', 'has_shaft_shape', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'smooth_texture', 'solid']
- **cand_0012**: support=1, stability=1.0000, features=['greenish', 'heavy_weight', 'near_table', 'smooth_texture', 'solid']
- **cand_0013**: support=2, stability=0.7500, features=['block_like', 'elongated_with_handle', 'grayish', 'has_bark_texture', 'has_crystal_flecks', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'movable', 'near_table', 'on_left_side', 'solid']
- **cand_0014**: support=1, stability=1.0000, features=['greenish', 'has_granular_surface', 'has_shaft_shape', 'light_weight', 'long_shape', 'near_table', 'rough_texture', 'round_small', 'solid']
- **cand_0016**: support=3, stability=0.5385, features=['brownish', 'has_grip_area', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'solid']
- **cand_0017**: support=3, stability=0.4286, features=['block_like', 'brownish', 'elongated_with_handle', 'grayish', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'recently_seen', 'rough_texture', 'smooth_texture', 'solid']
- **cand_0018**: support=2, stability=0.8571, features=['block_like', 'brownish', 'elongated_with_handle', 'has_grip_area', 'has_shaft_shape', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'solid']
- **cand_0019**: support=4, stability=0.3529, features=['block_like', 'elongated_with_handle', 'greenish', 'has_peel_texture', 'has_shaft_shape', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'recently_seen', 'rough_texture', 'round_small', 'smooth_texture', 'solid']
- **cand_0021**: support=1, stability=1.0000, features=['block_like', 'brownish', 'greenish', 'has_bark_texture', 'has_granular_surface', 'has_shaft_shape', 'has_stem_remnant', 'on_left_side', 'recently_seen', 'solid']
- **cand_0022**: support=2, stability=0.9231, features=['block_like', 'brownish', 'elongated_with_handle', 'greenish', 'has_bark_texture', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'on_left_side', 'round_small']
- **cand_0023**: support=2, stability=0.7857, features=['brownish', 'has_crystal_flecks', 'has_granular_surface', 'has_grip_area', 'has_peel_texture', 'has_wood_grain', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'on_left_side', 'recently_seen', 'rough_texture', 'solid']
- **cand_0024**: support=1, stability=1.0000, features=['brownish', 'elongated_with_handle', 'grayish', 'has_crystal_flecks', 'has_grip_area', 'has_shaft_shape', 'movable', 'round_small']
- **cand_0025**: support=3, stability=0.4000, features=['block_like', 'brownish', 'greenish', 'has_bark_texture', 'has_peel_texture', 'has_shaft_shape', 'has_stem_remnant', 'heavy_weight', 'light_weight', 'long_shape', 'movable', 'near_table', 'recently_seen', 'rough_texture', 'smooth_texture']
- **cand_0026**: support=1, stability=1.0000, features=['elongated_with_handle', 'has_stem_remnant', 'has_wood_grain', 'heavy_weight', 'light_weight', 'movable', 'round_small', 'solid']
- **cand_0027**: support=2, stability=0.9091, features=['block_like', 'brownish', 'grayish', 'has_crystal_flecks', 'has_shaft_shape', 'heavy_weight', 'light_weight', 'long_shape', 'near_table', 'smooth_texture', 'solid']
- **cand_0028**: support=1, stability=1.0000, features=['elongated_with_handle', 'greenish', 'has_grip_area', 'has_shaft_shape', 'heavy_weight', 'movable', 'rough_texture', 'smooth_texture', 'solid']

## Feature Overlap with TYPE_FAMILIES (Evaluation Only)

| Candidate | Best Family | Jaccard | Learned Features | Family Features |
|-----------|-------------|--------|-----------------|----------------|
| cand_0001 | wood-like | 0.2143 | 12 | 5 |
| cand_0005 | stone-like | 0.2667 | 14 | 5 |
| cand_0006 | apple-like | 0.2500 | 14 | 6 |
| cand_0008 | wood-like | 0.3846 | 13 | 5 |
| cand_0009 | tool-like | 0.3571 | 14 | 5 |
| cand_0015 | apple-like | 0.2500 | 14 | 6 |
| cand_0020 | tool-like | 0.3077 | 12 | 5 |

## Acceptance Checks

| Check | Result |
|-------|--------|
| no_hidden_leakage | PASS |
| no_oracle_leakage | PASS |
| no_single_event_promotion | PASS |
| candidates_cite_source_ids | PASS |
| provisional_unless_criteria_met | PASS |
| stable_meets_all_criteria | PASS |
| policy_unchanged | PASS |
| reproducible | PASS |

**All checks passed: True**


```
[phase_done]
phase=2
doc=protocols/phase2_object_candidate_memory_v0_1.md
shadow_json=present
shadow_summary=present
provisional_candidates=22
stable_candidates=7
single_event_promotion_count=0
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
