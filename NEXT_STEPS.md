# FlowAtlas WFC Development Roadmap

**Current Status**: Planning Phase
**Last Updated**: 2026-03-28

---

## Overview

Master TODO list for next major development phases of the WFC/FlowAtlas generation system. Organized by logical phases with clear dependencies.

Quick links to specific phases:
- [Phase 1: Debug & Organic Growth](#phase-1-debug--organic-growth-foundation)
- [Phase 2: Dynamic Weighting & Local Rules](#phase-2-dynamic-weighting--local-rules-algorithm-refinement)
- [Phase 3: Flag System Expansion](#phase-3-flag-system-expansion-national-foundation)
- [Phase 4: National Integration](#phase-4-national-integration-national-bias-system)
- [Phase 5: Multi-Layer Support](#phase-5-multi-layer-support-dimensional-expansion)
- [Phase 6: Travel & Navigation Rules](#phase-6-travel--navigation-rules-complex-balancing)

---

## PHASE 1: Debug & Organic Growth (Foundation)

**Goal**: Understand and fix the issue where WFC sets all borders before provinces

### 1.1 Investigate Border-Before-Province Issue
- **Status**: Not started
- **Description**: Current WFC implementation appears to place all borders before other provinces. Likely causes: length/weight distribution imbalance in options, or actual algorithmic error.
- **Action items**:
  - [ ] Analyze entropy distribution across iterations
  - [ ] Check option availability at different sequence points
  - [ ] Compare lengths and weight distributions of border vs province options
- **Outcome**: Root cause identified and documented
- **Blocked by**: None

### 1.2 Add Entropy/State Tracking to Debug Framework
- **Status**: Not started
- **Description**: Build tools to track element entropy over time, create visualizations/checkpoint maps
- **Action items**:
  - [ ] Extend debug framework with entropy metrics per element type
  - [ ] Add checkpoint/visualization system for intermediate states
- **Outcome**: Visual debugging tools available for future WFC analysis
- **Blocked by**: 1.1

### 1.3 Implement Organic Growth Fix
- **Status**: Not started
- **Description**: Modify WFC to balance borders and provinces more naturally
- **Outcome**: Borders and provinces grow together more organically
- **Blocked by**: 1.2

---

## PHASE 2: Dynamic Weighting & Local Rules (Algorithm Refinement)

**Goal**: Add sophisticated weight adjustment mechanisms and local rule support, with optional spatial adaptation

### 2.1 Dynamic Distance/Weight Adjustment Over Time
- **Status**: Not started
- **Description**: Implement optional mechanism to adjust dist/weight dynamically based on remaining slots and target distribution progress
- **Action items**:
  - [ ] Design time-based weight curve (light touch early, harder near end)
  - [ ] Implement as optional mode
  - [ ] Test for spatial bias when used with spatial seeding (e.g., nation capitals)
  - [ ] Document tradeoffs
- **Note**: May introduce spatial bias with spatial seeding - needs careful balancing
- **Outcome**: Optional dynamic weighting available, documented pros/cons
- **Blocked by**: None (but benefits from 1.3)

### 2.2 Local Distribution Contribution Rules (BanRule-like)
- **Status**: Not started
- **Description**: Create new rule type like `BanRule` but for dist contributions. Built from neighborhood, not fixed target distribution.
- **Action items**:
  - [ ] Design rule class structure (inherit from Rule?)
  - [ ] Define how neighborhood affects dist contribution
  - [ ] Support only specific terrain types (not all)
  - [ ] Create place to house these rules (similar to BanRule pattern)
- **Outcome**: LocalDistRule (or similar) class available and tested
- **Blocked by**: None (possibly 2.1)

### 2.3 Add Natural Rules
- **Status**: Not started
- **Description**: Use new LocalDistRule type to implement natural rules
- **Examples**: Natural gradient rules, neighborhood clustering, etc.
- **Outcome**: Set of natural rules available and documented
- **Blocked by**: 2.2

---

## PHASE 3: Flag System Expansion (National Foundation)

**Goal**: Build infrastructure for non-uniform flag weights and national assignment

### 3.1 Support Non-1.0 Flag Weights
- **Status**: Not started
- **Description**: Handle non-uniform flag weights (currently probably assumes 1.0)
- **Action items**:
  - [ ] Audit current flag system for hardcoded 1.0 assumptions
  - [ ] Implement variable weight support
  - [ ] Test flag interaction with weighting scheme
- **Outcome**: Flag system supports arbitrary non-negative weights
- **Blocked by**: None

### 3.2 Nation Capital Flag Assignment
- **Status**: Not started
- **Description**: Create method to assign flags in neighborhoods around nation capitals
- **Action items**:
  - [ ] Define neighborhood shapes/sizes (radius, custom patterns, etc.)
  - [ ] Implement flag assignment algorithm
  - [ ] Connect to nation data structure
- **Outcome**: Can seed nations with capital flags before WFC generation
- **Blocked by**: 3.1

### 3.3 DreamAtlas Data Import Helpers
- **Status**: Not started
- **Description**: Helper functions to import terrain distributions and constraint data from DreamAtlas
- **Action items**:
  - [ ] Create import functions from `dreamatlas_data.py`
  - [ ] Add support for `dominions_data.py` terrain data
  - [ ] Consider `peripheries` and other db resources
  - [ ] Design API for setting up nation-specific dists
  - [ ] (Future) Helper for determining good border terrains between nations
- **Outcome**: Easy-to-use helpers for populating FlowAtlas with DreamAtlas data
- **Blocked by**: None

---

## PHASE 4: National Integration (National Bias System)

**Goal**: Fully integrate national identity into WFC generation

### 4.1 Full National Info Integration via Flags & Dists
- **Status**: Not started
- **Description**: Complete system to pass national terrain preferences and constraints into WFC generation
- **Action items**:
  - [ ] Use dists from 3.3 for nation-specific terrain preferences
  - [ ] Use flags from 3.2 with variable weights (3.1) for spatial bias
  - [ ] Test stability with multiple overlapping national biases
  - [ ] Validate generated maps reflect national preferences
- **Outcome**: Nations can have distinct terrain preferences and spatial biases in generation
- **Blocked by**: 3.1, 3.2, 3.3

---

## PHASE 5: Multi-Layer Support (Dimensional Expansion)

**Goal**: Support multiple coordinated layers (e.g., surface + cave)

### 5.1 Multi-Layer Coordination
- **Status**: Not started
- **Description**: Design and implement cave layer + normal layer generation in single graph
- **Action items**:
  - [ ] Design flag/marking system to distinguish layer membership
  - [ ] Determine if nodes/edges same or separate per layer
  - [ ] Define terrain sets available per layer
  - [ ] Implement joint population algorithm
  - [ ] Handle transitions between layers
- **Outcome**: Can generate surface and cave layers with coupling constraints
- **Blocked by**: None (but 2.0 would help understand layer constraints)

---

## PHASE 6: Travel & Navigation Rules (Complex Balancing)

**Goal**: Sophisticated rules for inter-national connectivity and pathfinding balance

### 6.1 Travel Difficulty Tracking Framework
- **Status**: Not started
- **Description**: Rules to track difficulty of travel between key points (nation-to-nation, nation-to-throne)
- **Action items**:
  - [ ] Design metric system for "travel difficulty"
  - [ ] Create rule class to monitor and score connectivity
  - [ ] Integrate with map generation feedback loop
  - [ ] Document balance/tuning parameters
- **Outcome**: Can measure and balance map connectivity
- **Blocked by**: None

### 6.2 Multi-Mode Travel Rules Suite
- **Status**: Not started
- **Description**: Implement variety of travel rule types, potentially run in parallel with different weights
- **Action items**:
  - [ ] Design modular travel rule architecture
  - [ ] **Link Counting**: Simple path existence tracking
  - [ ] **Terrain-Specific Link Checking**: Exclude human-hindering terrain in neutral temps (configurable by climate/unit-type)
  - [ ] **Step Cost Pathing**: Route cost-based connectivity, with replaceable link checker for viable/unviable paths
  - [ ] **National "Easier" Pathing**: Nation-local shortcuts / preferred routes
  - [ ] **National "Easier" Step Cost**: Mix of national + step cost
  - [ ] **Sailing Paths**: Water-based routes
  - **Transport modes to support**:
    - [ ] Human (normal/hot/cold climate variants)
    - [ ] Amphibious
    - [ ] Swimming
    - [ ] Mountain walking (both links and cost)
    - [ ] Terrain-specific walking (various terrains)
    - [ ] Flying
    - [ ] Sailing/Water
    - [ ] Magic casting
  - [ ] Make step-cost pathfinding mixable with different link checkers
- **Outcome**: Flexible travel rule system allows fine-grained connectivity balancing
- **Blocked by**: None (but 6.1 provides framework)

### 6.3 Travel Rule Orchestration & Planning
- **Status**: Not started
- **Description**: System to manage multiple travel rules, their ordering, weights, and iteration plans
- **Action items**:
  - [ ] Design config/planning format for combining rules
  - [ ] Make it easy to add/remove rules at runtime
  - [ ] Support detailed sub-planning for individual rules
  - [ ] Create UI/API for tuning rule weights
- **Outcome**: Can experiment with different travel rule compositions easily
- **Blocked by**: 6.2

---

## Development Strategy

### Recommended Iteration Orders

**Path A: Build National System First** (Good for initial completeness)
```
1 → 3 → 4 → 2 → 6 → 5
```
Build the debug framework and basic national system, then add optimizations and complexity.

**Path B: Fix Algorithm First** (Good for immediate stability)
```
1 → 2 → 5 → 3 → 4 → 6
```
Resolve immediate WFC issues and algorithmic complexity, then add national features.

**Path C: Parallel Development** (Good for larger teams)
```
1 (parallel with)
├─ 2 (when 1.2 done)
├─ 3 (independent, start anytime)
├─ 5 (independent, start anytime)
└─ 6 (start immediately, uses other phases as feedback)
```

Choose based on team size and priority.

---

## How to Use This Document

1. **Finding this file**: Check the root of the repository as `NEXT_STEPS.md`
2. **Editing**: Open in any text editor; markdown format is human-readable and git-friendly
3. **Tracking work**:
   - Update status as you progress (Not started → In progress → Completed)
   - Check off `[ ]` items as you complete them
   - Add notes about decisions in the description
4. **Committing**: Git tracks all changes; you can see edit history and collaborate easily
5. **Sharing**: Push to git and share with collaborators; they see the same version

---

## Dependencies & Blocking

```
Phase 1 (Debug) → Phase 2 (Weighting) → Phase 3 (Flags) → Phase 4 (National)
       ↓
Phase 5 (Layers) - mostly independent
Phase 6 (Travel) - can start anytime, feeds on info from other phases
```

- Items marked "Blocked by" cannot start until those items complete
- Items marked "but benefits from" can start independently but will be easier/better after dependencies
- Some parallel work is possible; see recommended iteration paths above

---

## Notes for Collaborators

- If making significant changes to the roadmap structure, update the "Last Updated" date
- When starting work on a phase, create a feature branch and note it here
- For questions about phase dependencies or design, open an issue or discussion
- This document is a living plan; adapt it as you learn more
