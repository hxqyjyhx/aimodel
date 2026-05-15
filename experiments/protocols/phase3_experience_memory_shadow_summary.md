# Phase 3 Experience Memory ¡ª Shadow Summary

- **Block**: 1J36
- **Seed**: 109
- **Elapsed**: 3.4s

## Phase 2 OCM Recap

| Metric | Value |
|--------|-------|
| Total candidates | 29 |
| Stable candidates | 7 |
| Provisional candidates | 22 |
| Average support | 2.07 |
| Max support | 4 |

## Phase 3 Experience Memory Statistics

| Metric | Value |
|--------|-------|
| Total experience records | 174 |
| Stable experience records | 88 |
| Provisional experience records | 48 |
| Contradicted experience records | 35 |
| Rejected experience records | 3 |
| Object-action pairs | 360 |
| Candidate-action pairs | 212 |
| Action-outcome total support | 360 |
| Contradiction count | 41 |
| Feature-only rule detected | False |
| Hidden feature leakage | False |
| Oracle leakage | False |
| Policy decisions changed | False |

## Candidate Action-Outcome Diagnostics

| Diagnostic | Count | Candidate IDs |
|------------|-------|---------------|
| Consistent action outcomes | 4 | ['cand_0005', 'cand_0008', 'cand_0009', 'cand_0020'] |
| Mixed action outcomes | 3 | ['cand_0001', 'cand_0006', 'cand_0015'] |
| Split needed by action outcome | 0 | [] |
| Reinforced by action outcome | 4 | ['cand_0005', 'cand_0008', 'cand_0009', 'cand_0020'] |
| No action evidence | 0 | [] |

## Per-Candidate Action-Outcome Detail

### cand_0000

- Status: provisional, Support: 3, Stability: 0.5263

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 1 | 3 | contradicted | success |
| craft_plank | 0 | 3 | 3 | stable | failure |
| eat | 1 | 2 | 3 | contradicted | failure |
| mine_by_hand | 3 | 0 | 3 | stable | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 1 | 2 | 3 | contradicted | failure |

### cand_0001

- Status: stable, Support: 2, Stability: 0.8333

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 1 | 1 | 2 | contradicted | success |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0002

- Status: provisional, Support: 3, Stability: 0.5625

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 1 | 3 | contradicted | success |
| craft_plank | 2 | 1 | 3 | contradicted | success |
| eat | 1 | 2 | 3 | contradicted | failure |
| mine_by_hand | 3 | 0 | 3 | stable | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 0 | 3 | 3 | stable | failure |

### cand_0003

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 0 | 1 | 1 | provisional | failure |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0004

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 0 | 1 | 1 | provisional | failure |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0005

- Status: stable, Support: 2, Stability: 0.7857

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 2 | 2 | stable | failure |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 0 | 2 | 2 | stable | failure |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0006

- Status: stable, Support: 2, Stability: 0.7857

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 0 | 2 | stable | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 1 | 1 | 2 | contradicted | success |

### cand_0007

- Status: provisional, Support: 3, Stability: 0.5556

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 3 | 0 | 3 | stable | success |
| craft_plank | 2 | 1 | 3 | contradicted | success |
| eat | 0 | 3 | 3 | stable | failure |
| mine_by_hand | 3 | 0 | 3 | stable | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 0 | 3 | 3 | stable | failure |

### cand_0008

- Status: stable, Support: 2, Stability: 0.7692

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 0 | 2 | stable | success |
| craft_plank | 2 | 0 | 2 | stable | success |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0009

- Status: stable, Support: 2, Stability: 0.7857

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 0 | 2 | stable | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 2 | 0 | 2 | stable | success |

### cand_0010

- Status: provisional, Support: 2, Stability: 0.6364

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0011

- Status: provisional, Support: 4, Stability: 0.5238

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 3 | 4 | contradicted | failure |
| craft_plank | 1 | 3 | 4 | contradicted | failure |
| eat | 3 | 1 | 4 | contradicted | success |
| mine_by_hand | 4 | 0 | 4 | stable | success |
| mine_with_pickaxe | 4 | 0 | 4 | stable | success |
| use_as_tool | 0 | 4 | 4 | stable | failure |

### cand_0012

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0013

- Status: provisional, Support: 2, Stability: 0.7500

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 1 | 1 | 2 | contradicted | success |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 1 | 1 | 2 | contradicted | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0014

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0015

- Status: stable, Support: 2, Stability: 0.7143

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 1 | 1 | 2 | contradicted | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 1 | 1 | 2 | contradicted | success |

### cand_0016

- Status: provisional, Support: 3, Stability: 0.5385

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 3 | 3 | stable | failure |
| craft_plank | 0 | 3 | 3 | stable | failure |
| eat | 2 | 1 | 3 | contradicted | success |
| mine_by_hand | 2 | 1 | 3 | contradicted | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 0 | 3 | 3 | stable | failure |

### cand_0017

- Status: provisional, Support: 3, Stability: 0.4286

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 1 | 3 | contradicted | success |
| craft_plank | 2 | 1 | 3 | contradicted | success |
| eat | 0 | 3 | 3 | stable | failure |
| mine_by_hand | 2 | 1 | 3 | contradicted | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 0 | 3 | 3 | stable | failure |

### cand_0018

- Status: provisional, Support: 2, Stability: 0.8571

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 0 | 2 | stable | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 2 | 0 | 2 | stable | success |

### cand_0019

- Status: provisional, Support: 4, Stability: 0.3529

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 2 | 2 | 4 | rejected | success |
| craft_plank | 0 | 4 | 4 | stable | failure |
| eat | 2 | 2 | 4 | rejected | success |
| mine_by_hand | 4 | 0 | 4 | stable | success |
| mine_with_pickaxe | 4 | 0 | 4 | stable | success |
| use_as_tool | 2 | 2 | 4 | rejected | success |

### cand_0020

- Status: stable, Support: 2, Stability: 0.8333

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 2 | 2 | stable | failure |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 0 | 2 | 2 | stable | failure |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0021

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0022

- Status: provisional, Support: 2, Stability: 0.9231

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 1 | 1 | 2 | contradicted | success |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 1 | 1 | 2 | contradicted | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0023

- Status: provisional, Support: 2, Stability: 0.7857

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 1 | 1 | 2 | contradicted | success |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 2 | 0 | 2 | stable | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0024

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 0 | 1 | provisional | success |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 1 | 0 | 1 | provisional | success |

### cand_0025

- Status: provisional, Support: 3, Stability: 0.4000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 2 | 3 | contradicted | failure |
| craft_plank | 0 | 3 | 3 | stable | failure |
| eat | 1 | 2 | 3 | contradicted | failure |
| mine_by_hand | 3 | 0 | 3 | stable | success |
| mine_with_pickaxe | 3 | 0 | 3 | stable | success |
| use_as_tool | 1 | 2 | 3 | contradicted | failure |

### cand_0026

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 0 | 1 | 1 | provisional | failure |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 1 | 0 | 1 | provisional | success |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 0 | 1 | 1 | provisional | failure |

### cand_0027

- Status: provisional, Support: 2, Stability: 0.9091

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 1 | 2 | contradicted | success |
| craft_plank | 0 | 2 | 2 | stable | failure |
| eat | 0 | 2 | 2 | stable | failure |
| mine_by_hand | 1 | 1 | 2 | contradicted | success |
| mine_with_pickaxe | 2 | 0 | 2 | stable | success |
| use_as_tool | 0 | 2 | 2 | stable | failure |

### cand_0028

- Status: provisional, Support: 1, Stability: 1.0000

| Action | Success | Failure | Support | Status | Dominant |
|--------|---------|---------|---------|--------|----------|
| burn_as_fuel | 1 | 0 | 1 | provisional | success |
| craft_plank | 0 | 1 | 1 | provisional | failure |
| eat | 0 | 1 | 1 | provisional | failure |
| mine_by_hand | 1 | 0 | 1 | provisional | success |
| mine_with_pickaxe | 1 | 0 | 1 | provisional | success |
| use_as_tool | 1 | 0 | 1 | provisional | success |

## Experience Records Summary

| ID | Object Anchor | Candidate Anchor | Action | Success | Failure | Support | Status | Contradictions |
|----|---------------|-------------------|--------|---------|---------|--------|--------|----------------|
| exp_cand_0000_burn_as_fuel | test_apple_000 | cand_0000 | burn_as_fuel | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0000_craft_plank | test_apple_000 | cand_0000 | craft_plank | 0 | 3 | 3 | stable | 0 |
| exp_cand_0000_eat | test_apple_000 | cand_0000 | eat | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0000_mine_by_hand | test_apple_000 | cand_0000 | mine_by_hand | 3 | 0 | 3 | stable | 0 |
| exp_cand_0000_mine_with_pickaxe | test_apple_000 | cand_0000 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0000_use_as_tool | test_apple_000 | cand_0000 | use_as_tool | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0001_burn_as_fuel | test_apple_005 | cand_0001 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0001_craft_plank | test_apple_005 | cand_0001 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0001_eat | test_apple_005 | cand_0001 | eat | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0001_mine_by_hand | test_apple_005 | cand_0001 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0001_mine_with_pickaxe | test_apple_005 | cand_0001 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0001_use_as_tool | test_apple_005 | cand_0001 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0002_burn_as_fuel | test_apple_010 | cand_0002 | burn_as_fuel | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0002_craft_plank | test_apple_010 | cand_0002 | craft_plank | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0002_eat | test_apple_010 | cand_0002 | eat | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0002_mine_by_hand | test_apple_010 | cand_0002 | mine_by_hand | 3 | 0 | 3 | stable | 0 |
| exp_cand_0002_mine_with_pickaxe | test_apple_010 | cand_0002 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0002_use_as_tool | test_apple_010 | cand_0002 | use_as_tool | 0 | 3 | 3 | stable | 0 |
| exp_cand_0003_burn_as_fuel | test_stone_block_000 | cand_0003 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0003_craft_plank | test_stone_block_000 | cand_0003 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0003_eat | test_stone_block_000 | cand_0003 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0003_mine_by_hand | test_stone_block_000 | cand_0003 | mine_by_hand | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0003_mine_with_pickaxe | test_stone_block_000 | cand_0003 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0003_use_as_tool | test_stone_block_000 | cand_0003 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0004_burn_as_fuel | test_stone_block_005 | cand_0004 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0004_craft_plank | test_stone_block_005 | cand_0004 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0004_eat | test_stone_block_005 | cand_0004 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0004_mine_by_hand | test_stone_block_005 | cand_0004 | mine_by_hand | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0004_mine_with_pickaxe | test_stone_block_005 | cand_0004 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0004_use_as_tool | test_stone_block_005 | cand_0004 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0005_burn_as_fuel | test_stone_block_010 | cand_0005 | burn_as_fuel | 0 | 2 | 2 | stable | 0 |
| exp_cand_0005_craft_plank | test_stone_block_010 | cand_0005 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0005_eat | test_stone_block_010 | cand_0005 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0005_mine_by_hand | test_stone_block_010 | cand_0005 | mine_by_hand | 0 | 2 | 2 | stable | 0 |
| exp_cand_0005_mine_with_pickaxe | test_stone_block_010 | cand_0005 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0005_use_as_tool | test_stone_block_010 | cand_0005 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0006_burn_as_fuel | test_wood_log_000 | cand_0006 | burn_as_fuel | 2 | 0 | 2 | stable | 0 |
| exp_cand_0006_craft_plank | test_wood_log_000 | cand_0006 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0006_eat | test_wood_log_000 | cand_0006 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0006_mine_by_hand | test_wood_log_000 | cand_0006 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0006_mine_with_pickaxe | test_wood_log_000 | cand_0006 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0006_use_as_tool | test_wood_log_000 | cand_0006 | use_as_tool | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0007_burn_as_fuel | test_wood_log_005 | cand_0007 | burn_as_fuel | 3 | 0 | 3 | stable | 0 |
| exp_cand_0007_craft_plank | test_wood_log_005 | cand_0007 | craft_plank | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0007_eat | test_wood_log_005 | cand_0007 | eat | 0 | 3 | 3 | stable | 0 |
| exp_cand_0007_mine_by_hand | test_wood_log_005 | cand_0007 | mine_by_hand | 3 | 0 | 3 | stable | 0 |
| exp_cand_0007_mine_with_pickaxe | test_wood_log_005 | cand_0007 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0007_use_as_tool | test_wood_log_005 | cand_0007 | use_as_tool | 0 | 3 | 3 | stable | 0 |
| exp_cand_0008_burn_as_fuel | test_wood_log_010 | cand_0008 | burn_as_fuel | 2 | 0 | 2 | stable | 0 |
| exp_cand_0008_craft_plank | test_wood_log_010 | cand_0008 | craft_plank | 2 | 0 | 2 | stable | 0 |
| exp_cand_0008_eat | test_wood_log_010 | cand_0008 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0008_mine_by_hand | test_wood_log_010 | cand_0008 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0008_mine_with_pickaxe | test_wood_log_010 | cand_0008 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0008_use_as_tool | test_wood_log_010 | cand_0008 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0009_burn_as_fuel | test_wooden_pickaxe_000 | cand_0009 | burn_as_fuel | 2 | 0 | 2 | stable | 0 |
| exp_cand_0009_craft_plank | test_wooden_pickaxe_000 | cand_0009 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0009_eat | test_wooden_pickaxe_000 | cand_0009 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0009_mine_by_hand | test_wooden_pickaxe_000 | cand_0009 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0009_mine_with_pickaxe | test_wooden_pickaxe_000 | cand_0009 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0009_use_as_tool | test_wooden_pickaxe_000 | cand_0009 | use_as_tool | 2 | 0 | 2 | stable | 0 |
| exp_cand_0010_burn_as_fuel | test_apple_001 | cand_0010 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0010_craft_plank | test_apple_001 | cand_0010 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0010_eat | test_apple_001 | cand_0010 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0010_mine_by_hand | test_apple_001 | cand_0010 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0010_mine_with_pickaxe | test_apple_001 | cand_0010 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0010_use_as_tool | test_apple_001 | cand_0010 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0011_burn_as_fuel | test_apple_006 | cand_0011 | burn_as_fuel | 1 | 3 | 4 | contradicted | 1 |
| exp_cand_0011_craft_plank | test_apple_006 | cand_0011 | craft_plank | 1 | 3 | 4 | contradicted | 1 |
| exp_cand_0011_eat | test_apple_006 | cand_0011 | eat | 3 | 1 | 4 | contradicted | 1 |
| exp_cand_0011_mine_by_hand | test_apple_006 | cand_0011 | mine_by_hand | 4 | 0 | 4 | stable | 0 |
| exp_cand_0011_mine_with_pickaxe | test_apple_006 | cand_0011 | mine_with_pickaxe | 4 | 0 | 4 | stable | 0 |
| exp_cand_0011_use_as_tool | test_apple_006 | cand_0011 | use_as_tool | 0 | 4 | 4 | stable | 0 |
| exp_cand_0012_burn_as_fuel | test_apple_011 | cand_0012 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0012_craft_plank | test_apple_011 | cand_0012 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0012_eat | test_apple_011 | cand_0012 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0012_mine_by_hand | test_apple_011 | cand_0012 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0012_mine_with_pickaxe | test_apple_011 | cand_0012 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0012_use_as_tool | test_apple_011 | cand_0012 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0013_burn_as_fuel | test_stone_block_001 | cand_0013 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0013_craft_plank | test_stone_block_001 | cand_0013 | craft_plank | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0013_eat | test_stone_block_001 | cand_0013 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0013_mine_by_hand | test_stone_block_001 | cand_0013 | mine_by_hand | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0013_mine_with_pickaxe | test_stone_block_001 | cand_0013 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0013_use_as_tool | test_stone_block_001 | cand_0013 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0014_burn_as_fuel | test_stone_block_006 | cand_0014 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0014_craft_plank | test_stone_block_006 | cand_0014 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0014_eat | test_stone_block_006 | cand_0014 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0014_mine_by_hand | test_stone_block_006 | cand_0014 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0014_mine_with_pickaxe | test_stone_block_006 | cand_0014 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0014_use_as_tool | test_stone_block_006 | cand_0014 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0015_burn_as_fuel | test_wooden_pickaxe_011 | cand_0015 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0015_craft_plank | test_wooden_pickaxe_011 | cand_0015 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0015_eat | test_wooden_pickaxe_011 | cand_0015 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0015_mine_by_hand | test_wooden_pickaxe_011 | cand_0015 | mine_by_hand | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0015_mine_with_pickaxe | test_wooden_pickaxe_011 | cand_0015 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0015_use_as_tool | test_wooden_pickaxe_011 | cand_0015 | use_as_tool | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0016_burn_as_fuel | test_apple_002 | cand_0016 | burn_as_fuel | 0 | 3 | 3 | stable | 0 |
| exp_cand_0016_craft_plank | test_apple_002 | cand_0016 | craft_plank | 0 | 3 | 3 | stable | 0 |
| exp_cand_0016_eat | test_apple_002 | cand_0016 | eat | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0016_mine_by_hand | test_apple_002 | cand_0016 | mine_by_hand | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0016_mine_with_pickaxe | test_apple_002 | cand_0016 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0016_use_as_tool | test_apple_002 | cand_0016 | use_as_tool | 0 | 3 | 3 | stable | 0 |
| exp_cand_0017_burn_as_fuel | test_stone_block_007 | cand_0017 | burn_as_fuel | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0017_craft_plank | test_stone_block_007 | cand_0017 | craft_plank | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0017_eat | test_stone_block_007 | cand_0017 | eat | 0 | 3 | 3 | stable | 0 |
| exp_cand_0017_mine_by_hand | test_stone_block_007 | cand_0017 | mine_by_hand | 2 | 1 | 3 | contradicted | 1 |
| exp_cand_0017_mine_with_pickaxe | test_stone_block_007 | cand_0017 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0017_use_as_tool | test_stone_block_007 | cand_0017 | use_as_tool | 0 | 3 | 3 | stable | 0 |
| exp_cand_0018_burn_as_fuel | test_wooden_pickaxe_007 | cand_0018 | burn_as_fuel | 2 | 0 | 2 | stable | 0 |
| exp_cand_0018_craft_plank | test_wooden_pickaxe_007 | cand_0018 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0018_eat | test_wooden_pickaxe_007 | cand_0018 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0018_mine_by_hand | test_wooden_pickaxe_007 | cand_0018 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0018_mine_with_pickaxe | test_wooden_pickaxe_007 | cand_0018 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0018_use_as_tool | test_wooden_pickaxe_007 | cand_0018 | use_as_tool | 2 | 0 | 2 | stable | 0 |
| exp_cand_0019_burn_as_fuel | test_wooden_pickaxe_012 | cand_0019 | burn_as_fuel | 2 | 2 | 4 | rejected | 2 |
| exp_cand_0019_craft_plank | test_wooden_pickaxe_012 | cand_0019 | craft_plank | 0 | 4 | 4 | stable | 0 |
| exp_cand_0019_eat | test_wooden_pickaxe_012 | cand_0019 | eat | 2 | 2 | 4 | rejected | 2 |
| exp_cand_0019_mine_by_hand | test_wooden_pickaxe_012 | cand_0019 | mine_by_hand | 4 | 0 | 4 | stable | 0 |
| exp_cand_0019_mine_with_pickaxe | test_wooden_pickaxe_012 | cand_0019 | mine_with_pickaxe | 4 | 0 | 4 | stable | 0 |
| exp_cand_0019_use_as_tool | test_wooden_pickaxe_012 | cand_0019 | use_as_tool | 2 | 2 | 4 | rejected | 2 |
| exp_cand_0020_burn_as_fuel | test_stone_block_003 | cand_0020 | burn_as_fuel | 0 | 2 | 2 | stable | 0 |
| exp_cand_0020_craft_plank | test_stone_block_003 | cand_0020 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0020_eat | test_stone_block_003 | cand_0020 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0020_mine_by_hand | test_stone_block_003 | cand_0020 | mine_by_hand | 0 | 2 | 2 | stable | 0 |
| exp_cand_0020_mine_with_pickaxe | test_stone_block_003 | cand_0020 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0020_use_as_tool | test_stone_block_003 | cand_0020 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0021_burn_as_fuel | test_stone_block_008 | cand_0021 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0021_craft_plank | test_stone_block_008 | cand_0021 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0021_eat | test_stone_block_008 | cand_0021 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0021_mine_by_hand | test_stone_block_008 | cand_0021 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0021_mine_with_pickaxe | test_stone_block_008 | cand_0021 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0021_use_as_tool | test_stone_block_008 | cand_0021 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0022_burn_as_fuel | test_stone_block_013 | cand_0022 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0022_craft_plank | test_stone_block_013 | cand_0022 | craft_plank | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0022_eat | test_stone_block_013 | cand_0022 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0022_mine_by_hand | test_stone_block_013 | cand_0022 | mine_by_hand | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0022_mine_with_pickaxe | test_stone_block_013 | cand_0022 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0022_use_as_tool | test_stone_block_013 | cand_0022 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0023_burn_as_fuel | test_wood_log_013 | cand_0023 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0023_craft_plank | test_wood_log_013 | cand_0023 | craft_plank | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0023_eat | test_wood_log_013 | cand_0023 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0023_mine_by_hand | test_wood_log_013 | cand_0023 | mine_by_hand | 2 | 0 | 2 | stable | 0 |
| exp_cand_0023_mine_with_pickaxe | test_wood_log_013 | cand_0023 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0023_use_as_tool | test_wood_log_013 | cand_0023 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0024_burn_as_fuel | test_wooden_pickaxe_008 | cand_0024 | burn_as_fuel | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0024_craft_plank | test_wooden_pickaxe_008 | cand_0024 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0024_eat | test_wooden_pickaxe_008 | cand_0024 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0024_mine_by_hand | test_wooden_pickaxe_008 | cand_0024 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0024_mine_with_pickaxe | test_wooden_pickaxe_008 | cand_0024 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0024_use_as_tool | test_wooden_pickaxe_008 | cand_0024 | use_as_tool | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0025_burn_as_fuel | test_wooden_pickaxe_013 | cand_0025 | burn_as_fuel | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0025_craft_plank | test_wooden_pickaxe_013 | cand_0025 | craft_plank | 0 | 3 | 3 | stable | 0 |
| exp_cand_0025_eat | test_wooden_pickaxe_013 | cand_0025 | eat | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0025_mine_by_hand | test_wooden_pickaxe_013 | cand_0025 | mine_by_hand | 3 | 0 | 3 | stable | 0 |
| exp_cand_0025_mine_with_pickaxe | test_wooden_pickaxe_013 | cand_0025 | mine_with_pickaxe | 3 | 0 | 3 | stable | 0 |
| exp_cand_0025_use_as_tool | test_wooden_pickaxe_013 | cand_0025 | use_as_tool | 1 | 2 | 3 | contradicted | 1 |
| exp_cand_0026_burn_as_fuel | test_apple_014 | cand_0026 | burn_as_fuel | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0026_craft_plank | test_apple_014 | cand_0026 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0026_eat | test_apple_014 | cand_0026 | eat | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0026_mine_by_hand | test_apple_014 | cand_0026 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0026_mine_with_pickaxe | test_apple_014 | cand_0026 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0026_use_as_tool | test_apple_014 | cand_0026 | use_as_tool | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0027_burn_as_fuel | test_stone_block_014 | cand_0027 | burn_as_fuel | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0027_craft_plank | test_stone_block_014 | cand_0027 | craft_plank | 0 | 2 | 2 | stable | 0 |
| exp_cand_0027_eat | test_stone_block_014 | cand_0027 | eat | 0 | 2 | 2 | stable | 0 |
| exp_cand_0027_mine_by_hand | test_stone_block_014 | cand_0027 | mine_by_hand | 1 | 1 | 2 | contradicted | 1 |
| exp_cand_0027_mine_with_pickaxe | test_stone_block_014 | cand_0027 | mine_with_pickaxe | 2 | 0 | 2 | stable | 0 |
| exp_cand_0027_use_as_tool | test_stone_block_014 | cand_0027 | use_as_tool | 0 | 2 | 2 | stable | 0 |
| exp_cand_0028_burn_as_fuel | test_wooden_pickaxe_009 | cand_0028 | burn_as_fuel | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0028_craft_plank | test_wooden_pickaxe_009 | cand_0028 | craft_plank | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0028_eat | test_wooden_pickaxe_009 | cand_0028 | eat | 0 | 1 | 1 | provisional | 0 |
| exp_cand_0028_mine_by_hand | test_wooden_pickaxe_009 | cand_0028 | mine_by_hand | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0028_mine_with_pickaxe | test_wooden_pickaxe_009 | cand_0028 | mine_with_pickaxe | 1 | 0 | 1 | provisional | 0 |
| exp_cand_0028_use_as_tool | test_wooden_pickaxe_009 | cand_0028 | use_as_tool | 1 | 0 | 1 | provisional | 0 |

## Acceptance Checks

| Check | Result |
|-------|--------|
| no_hidden_leakage | PASS |
| no_oracle_leakage | PASS |
| every_experience_cites_source_event_ids | PASS |
| every_experience_has_anchor | PASS |
| no_feature_only_rules | PASS |
| policy_unchanged | PASS |
| contradictions_reported | PASS |
| phase2_candidates_treated_as_provisional | PASS |
| reproducible | PASS |

**All checks passed: True**


```
[phase_done]
phase=3
doc=protocols/phase3_experience_memory_v0_1.md
shadow_json=present
shadow_summary=present
total_experience_records=174
stable_experience_records=88
feature_only_rule_detected=false
hidden_feature_leakage_detected=false
oracle_leakage_detected=false
policy_decisions_changed=false
implementation_status=pass
failure_reason=none
```
