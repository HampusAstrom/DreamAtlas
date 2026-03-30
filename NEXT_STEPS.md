# FlowAtlas Development Roadmap

**Current Status**: Planning Phase
**Last Updated**: 2026-03-30

## Maintenance Notes

- 2026-03-28: Hardened safe git hook launcher for cross-platform use by adding OS-specific hook commands and launcher scripts under `.github/hooks/`.
- 2026-03-29: Simplified safe git hooks to a single direct Python entrypoint in `.github/hooks/00-safe-tool-permissions.json`; removed wrapper scripts to reduce VS Code hang risk and set hook infrastructure failures to fail-open in `.github/hooks/apply_safe_git_rules.py`.
- 2026-03-29: Refactored `.github/hooks/safe-git-commands.json` into readable per-category allow rules; added safe auto-allow for `rg`, conda-activated `pytest`, and `git -c ... status`; tightened git safety by restricting `branch`/`tag`/`config` to read-only forms and explicitly blocking `ln` and other write-like shell commands.
- 2026-03-30: Upgraded to hardened-balanced pytest policy in `.github/hooks/safe-git-commands.json`: auto-allow only conda-activated test runs with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, workspace test targets, and no `-p`/`--pyargs`; non-compliant pytest commands are now blocked with an explicit compliant-command suggestion.
- 2026-03-30: Reordered hook policy intent to deny-first evaluation (including non-compliant pytest) before allows; added `nl` to read-only auto-allow and explicitly blocked common terminal virtual-environment creation commands (`python -m venv`, `virtualenv`, `uv venv`, `conda create`, `pipenv`, `poetry env use/remove`).

---

## Overview

Master TODO list for next major development phases of FlowAtlas. Organized by component/work stream with clear dependencies and agent/skill recommendations.

**Quick Navigation**:
- [Foundation Infrastructure (A-B)](#foundation-infrastructure)
- [Algorithm Development (C-E)](#algorithm-development)
- [Systems Integration (F)](#systems-integration)
- [Output & Export (G)](#output--export)
- [Development Strategy](#development-strategy--sequencing)

**Current High Prio**
1. B.1
2. C.1

**Next features**
- C.2, C.4
- C.3 and D
- A

---

# FOUNDATION INFRASTRUCTURE

## COMPONENT A: TerrainGraph Core Improvements

**Goal**: Enhance TerrainGraph to track geographic relationships for natural generation rules
**Recommended Agent/Skill**: Developer + Researcher (investigate coordinate systems)

### A.1 Province Coordinate Tracking
- **Status**: Not started
- **Skill**: Developer
- **Description**: TerrainGraph must track actual coordinates of each province (center point or boundary)
- **Why**: Needed for distance calculations, visualization, organic border rendering, geographic rules
- **Action items**:
  - [ ] Audit current TerrainGraph coordinate storage
  - [ ] Design coordinate tracking for nodes (province centers)
  - [ ] Design coordinate tracking for edges (province boundaries)
  - [ ] Add coordinate getters/accessors to TerrainGraph API
- **Deliverables**:
  - Coordinates accessible for all nodes and edges
  - Clear documentation on coordinate reference systems
- **Blocked by**: None

### A.2 Voronoi Vertex & Topology Tracking
- **Status**: Not started
- **Skill**: Developer + Mathematician (Voronoi-Delaunay duality)
- **Description**: Track where Voronoi cells meet—vertices where 3+ provinces meet
- **Why**: Enables river/strait rules, geographic constraints, vertex neighborhood analysis
- **Action items**:
  - [ ] Design Voronoi vertex data structure (location, adjacent provinces, borders)
  - [ ] Implement vertex detection from Voronoi cells
  - [ ] Create vertex → provinces/borders mapping
  - [ ] Design "geography" connectivity type for get_connected_elements
- **Deliverables**:
  - Voronoi vertices tracked with neighbors
  - Can query: "what borders/provinces meet at vertex?"
  - Can query: "geographic pattern of these elements?"
- **Blocked by**: A.1

### A.3 Distance Metrics for Border Vertices
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker (verify semantics)
- **Description**: Define distance calculations for borders at Voronoi vertices
- **Why**: Needed for river fork/merge rules, strait island chains, natural features
- **Action items**:
  - [ ] Analyze BanRule distance semantics (baseline)
  - [ ] Design "geography" connectivity distance metrics
  - [ ] Test distance calculations against real Voronoi
  - [ ] Document distance semantics clearly
- **Deliverables**:
  - Clear specification of vertex distances
  - Distance metrics usable by Rules
- **Blocked by**: A.2

### A.4 Make TerrainGraph Topology Available to Rules
- **Status**: Not started
- **Skill**: Developer
- **Description**: Expose Voronoi geometry, vertex topology, distances to Rule system
- **Feeds into**: B.2, F.2 (Rules can query topology)
- **Action items**:
  - [ ] Design Rule API for topology queries
  - [ ] Add methods to Rule: "get adjacent borders at vertex", etc.
  - [ ] Create helper functions for common geographic queries
  - [ ] Document examples of using topology in Rules
- **Deliverables**:
  - Rules query Voronoi vertex topology
  - Example geographic Rules (rivers, straits)
- **Blocked by**: A.3

---

## COMPONENT B: WFC Foundation & Debug Framework

**Goal**: Establish stable WFC foundation with visibility into generation process
**Recommended Agent/Skill**: Developer + Code Smell Checker (diagnosis) + Planner

### B.1 Investigate & Fix Border-Before-Province Issue
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker (diagnosis)
- **Description**: Current WFC places all borders before other provinces. Find root cause.
- **Action items**:
  - [ ] Analyze entropy distribution across iterations
  - [ ] Check option availability at sequence points
  - [ ] Compare border vs province option distributions
  - [ ] Use debug framework (B.3) to visualize problem
  - [ ] Implement fix based on findings
- **Outcome**: Organic border/province growth
- **Blocked by**: B.3 (debugging helps diagnosis)
- **Feeds into**: B.2

### B.2 Geographic Rules/Bans and Constraints
- **Status**: Not started
- **Skill**: Developer + Planner
- **Description**: Rules for geographic features (rivers, straits, mountains) to generate naturally

#### B.2.1 Geographic Rules Framework
- **Status**: Not started
- **Description**: Create rule types for geographic constraints
- **Action items**:
  - [ ] Design rule types: river direction/flow, strait formation, mountain ranges, coastlines
  - [ ] Determine if standard Rules/BanRules sufficient or need new types
  - [ ] Integrate with TerrainGraph topology (A.1-A.4)
- **Outcome**: Can express geographic constraints in Rules
- **Blocked by**: A.4

#### B.2.2 River Generation Rules
- **Status**: Not started
- **Description**: Rivers merge high-to-low, branch only near water/swamps
- **Action items**:
  - [ ] Design river-specific rules using geographic topology
  - [ ] Determine: track direction during generation? (game doesn't care, but helps visualization)
  - [ ] How to resolve direction ambiguity until endpoint selected?
  - [ ] Create rules: merged rivers, high-to-low flow, natural deltas
  - [ ] Test river generation produces natural patterns
- **Deliverables**:
  - River direction tracking (optional)
  - Rules for natural river pathways
  - Test cases showing realistic rivers
- **Blocked by**: B.2.1

#### B.2.3 Straits & Water Borders
- **Status**: Not started
- **Description**: Treat straits/rivers differently at generation time for better visuals
- **Action items**:
  - [ ] Analyze mechanical difference between river and strait
  - [ ] Decide: track strait "direction" or "width"?
  - [ ] Consider strait-with-bridge variant
  - [ ] Create strait-specific generation rules
  - [ ] Design visual distinctions (see G.1)
- **Outcome**: Straits and rivers generate with appropriate visuals
- **Blocked by**: B.2.2
- **Feeds into**: G.1

#### B.2.4 Long Linear Geographic Features
- **Status**: Not started
- **Description**: Efficient handling of river/mountain "snakes" spanning many provinces
- **Action items**:
  - [ ] Design representation for "river backbones" or "mountain chains"
  - [ ] Track direction for mountain ranges? (less critical than rivers)
  - [ ] How do long features interact with Voronoi vertices? (endpoints, forks)
  - [ ] Create rules constraining entire chains at once
  - [ ] Test performance impact
- **Outcome**: Efficient constraints on long geographic features
- **Blocked by**: B.2.2

### B.3 WFC Debug & Entropy Tracking Framework
- **Status**: Completed
- **Skill**: Developer
- **Description**: Tools to visualize WFC state evolution and diagnose generation problems
- **Action items**:
  - [x] Extend WFC with state snapshots at key iterations
  - [x] Create entropy metrics tracker (per element, per terrain type)
  - [x] Build visualization for intermediate maps (images/checkpoints)
  - [x] Create comparison tools (iteration N vs N+K)
  - [x] Add logging: option distribution, weight changes, rule firings
    Implemented: selected-element option-distribution logging, full per-step entropy surface logging, and per-step rule-firing/adjusting-weight deltas behind debug flags.
  - Iteration comparison tool available via `WaveFunctionCollapse.compare_debug_iteration_snapshots(step_a, step_b)`.
- Checkpoint visualization implemented in `try_wave_function_collapse.py` via `render_checkpoint_maps(...)` using captured per-step checkpoint states, with unset provinces and borders rendered by entropy and support for separate, grid, or animation playback.
- **Deliverables**:
  - Entropy time-series data during/after generation
  - Checkpoint images for visual debugging
  - Detailed generation logs
- **Outcome**: Can diagnose generation issues by inspecting state over time
- **Blocked by**: None
- **Feeds into**: B.1, B.2, F.3

---

# ALGORITHM DEVELOPMENT

## COMPONENT C: WFC Algorithm Refinement

**Goal**: Improve WFC with dynamic weighting and local rules
**Recommended Agent/Skill**: Developer + Code Smell Checker + QA

### C.1 Dynamic Distance/Weight Adjustment Over Time
- **Status**: Not started
- **Skill**: Developer
- **Description**: Optionally adjust dist/weight dynamically based on progress
- **Strategy**: Light touch early (many slots), harder near end (few options)
- **Caution**: May introduce spatial bias with spatial seeding—test carefully
- **Action items**:
  - [ ] Design time-based weight curve
  - [ ] Implement as optional mode (controlled by flag)
  - [ ] Test for spatial bias with spatial seeding
  - [ ] Measure generation time and quality impact
  - [ ] Document pros, cons, use cases
- **Outcome**: Optional dynamic weighting with documented behavior
- **Blocked by**: None
- **Feeds into**: F (if useful for output quality)

### C.2 Local Distribution Contribution Rules
- **Status**: Not started
- **Skill**: Developer
- **Description**: Rule type for dist contributions built from neighborhood (not fixed targets)
- **Use case**: "Swamps attract swamps", "forests cluster" based on local observation
- **Action items**:
  - [ ] Design LocalDistRule class structure
  - [ ] Define how neighborhood affects dist contribution
  - [ ] Support only specific terrain types (not all)
  - [ ] Create rule placement (similar to BanRule)
- **Deliverables**:
  - LocalDistRule class and API
  - 3-5 example natural rules
  - Documentation on rule creation
- **Outcome**: Can express natural clustering in Rules
- **Blocked by**: None
- **Feeds into**: C.4

### C.3 Flag System: Non-1.0 Weights
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker
- **Description**: Support variable (non-1.0) flag weights for spatial biasing
- **Why**: Needed for national integration and gradual influence
- **Action items**:
  - [ ] Audit flag system for hardcoded 1.0 assumptions
  - [ ] Implement variable weight support
  - [ ] Test flag weights with diverse values (0.1 to 10.0)
  - [ ] Ensure weights interact correctly with dynamic weighting (C.1)
- **Deliverables**:
  - Flags support arbitrary non-negative weights
  - Weight specification API
- **Outcome**: Flags enable fine-grained spatial control
- **Blocked by**: None
- **Feeds into**: D.2

### C.4 Natural Clustering Rules
- **Status**: Not started
- **Skill**: Developer + QA
- **Description**: Use LocalDistRule to implement natural rules
- **Examples**: Swamps attract swamps, mountains cluster, forest edges, water gradients
- **Action items**:
  - [ ] Design 5-10 natural clustering rules
  - [ ] Test individually and in combination
  - [ ] Measure map naturalness (subjective + metrics)
  - [ ] Document rule semantics and effects
- **Outcome**: Set of natural rules available and tested
- **Blocked by**: C.2
- **Feeds into**: F (improved map quality)

---

## COMPONENT D: National & Flag Integration

**Goal**: Enable nation-specific terrain preferences and spatial biasing
**Recommended Agent/Skill**: Developer + Researcher (DreamAtlas data) + Planner

### D.1 Nation-Specific Terrain Distributions
- **Status**: Not started
- **Skill**: Developer + Researcher
- **Description**: Helpers to import terrain preferences from DreamAtlas
- **Action items**:
  - [ ] Design import API from dreamatlas_data.py
  - [ ] Create nation-to-dist mapping from DreamAtlas data
  - [ ] Support per-nation terrain type preferences
  - [ ] (Future) Helper for good border terrains between nations
  - [ ] Document API and provide examples
- **Deliverables**:
  - Import helpers working
  - Nation-specific dists available
- **Outcome**: Can easily populate FlowAtlas with DreamAtlas nation data
- **Blocked by**: None
- **Feeds into**: D.2

### D.2 Nation Capitals & Spatial Seeding
- **Status**: Not started
- **Skill**: Developer
- **Description**: Assign flags in neighborhoods around nation capitals
- **Action items**:
  - [ ] Define neighborhood patterns (radius, custom shapes, gradients)
  - [ ] Implement flag assignment for capital neighborhoods
  - [ ] Support variable weights (from C.3) for distance-based falloff
  - [ ] Connect to nation coordinate system
  - [ ] Test multiple overlapping nations
- **Deliverables**:
  - Can assign capital neighborhoods with flags
  - Supports gradient influence (not hard borders)
- **Outcome**: Nations seed generation with spatial bias
- **Blocked by**: C.3, D.1
- **Feeds into**: Full National Integration

### D.3 Full National Integration
- **Status**: Not started
- **Skill**: Developer + QA + Planner
- **Description**: Combine all national features into full system
- **Action items**:
  - [ ] Integrate D.1 (dists) + D.2 (flags) + C.3 (variable weights)
  - [ ] Test stable generation with multiple nations
  - [ ] Validate maps reflect national terrain preferences
  - [ ] Measure generation quality and time
- **Outcome**: Nations have distinct terrain preferences and spatial influence
- **Blocked by**: D.1, D.2
- **Feeds into**: UI integration

---

## COMPONENT E: Advanced Rules & Balancing

**Goal**: Complex rules for map balance and connectivity
**Recommended Agent/Skill**: Developer + QA + Planner (complex rules)

### E.1 Multi-Layer Coordination
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker
- **Description**: Support multiple coordinated layers (surface + cave) in single graph
- **Action items**:
  - [ ] Design layer membership system (flags, marking, index)
  - [ ] Determine node/edge sharing strategy
  - [ ] Define terrain availability per layer
  - [ ] Implement joint generation algorithm
  - [ ] Handle layer transitions
- **Deliverables**:
  - Multi-layer support in WFC
  - Layer membership queries
  - Example: surface + cave generation
- **Outcome**: Can generate coordinated multi-layer maps
- **Blocked by**: None (benefits from B.3)

### E.2 Travel Difficulty Tracking Framework
- **Status**: Not started
- **Skill**: Developer + Planner
- **Description**: Rules to measure travel difficulty between key points
- **Action items**:
  - [ ] Design travel difficulty metric system
  - [ ] Create rule class for monitoring connectivity
  - [ ] Define balance parameters (target path lengths)
  - [ ] Integrate with generation feedback loop
  - [ ] Document tuning parameters
- **Deliverables**:
  - Travel difficulty API
  - Balance scoring system
  - Constraint configuration
- **Outcome**: Can measure and balance inter-national connectivity
- **Blocked by**: D.3 (needs nations to test against)
- **Feeds into**: E.3

### E.3 Multi-Mode Travel Rules Suite
- **Status**: Not started
- **Skill**: Developer + QA + Researcher (pathfinding algorithms)
- **Description**: Variety of travel rule types, run in parallel with different weights
- **Rule variants**:
  - [ ] Link Counting (simple path existence)
  - [ ] Terrain-Specific Link Checking (exclude certain terrain for unit types)
  - [ ] Step Cost Pathing (route cost sum, with replaceable link checker)
  - [ ] National "Easier" Pathing (nation-local shortcuts)
  - [ ] National "Easier" Step Cost (national + step cost)
  - [ ] Sailing Paths (water-based routes)
- **Transport modes to support**:
  - [ ] Human (normal/hot/cold climate variants)
  - [ ] Amphibious, Swimming
  - [ ] Mountain walking (both links and cost)
  - [ ] Terrain-specific walkers (swamp, forest, etc.)
  - [ ] Flying, Sailing, Magic casting
- **Action items**:
  - [ ] Design modular travel rule architecture
  - [ ] Implement each rule variant
  - [ ] Create transport mode abstraction
  - [ ] Make step-cost mixable with different link checkers
  - [ ] Test rule combinations and weights
- **Deliverables**:
  - Travel rule classes for each variant
  - Transport mode system (extensible)
  - Combinator for step-cost + link checkers
  - Documentation and examples
- **Outcome**: Fine-grained connectivity balancing
- **Blocked by**: E.2
- **Feeds into**: E.4

### E.4 Travel Rule Orchestration & Planning
- **Status**: Not started
- **Skill**: Developer + Planner
- **Description**: System to manage multiple travel rules, weights, and iteration plans
- **Use case**: Experiment with rule combinations easily, adjust weights between generations
- **Action items**:
  - [ ] Design config/planning format (YAML? Python? Custom?)
  - [ ] Build runtime system to load/apply rule combinations
  - [ ] Support adding/removing rules without recompilation
  - [ ] Create UI for rule weight tuning (see F)
  - [ ] Track rule iteration history
  - [ ] Support detailed sub-planning
- **Deliverables**:
  - Rule orchestration config format
  - Runtime applier for rule combinations
  - Planning/history tracking
- **Outcome**: Can experiment with travel rule compositions easily
- **Blocked by**: E.3
- **Feeds into**: F (UI integration)

---

# SYSTEMS INTEGRATION

## COMPONENT F: UI Integration

**Goal**: Integrate TerrainGraph and WFC generation into DreamAtlas UI
**Recommended Agent/Skill**: Developer + Planner (architecture) + QA

**Note**: Large component. We'll expand detail as work approaches.

### F.1 Core UI Integration & Architecture
- **Status**: Not started
- **Skill**: Developer + Planner
- **Description**: Design and implement basic FlowAtlas generation within DreamAtlas UI
- **Action items**:
  - [ ] Audit DreamAtlas UI architecture
  - [ ] Design flow: Settings → FlowAtlas generation → Results
  - [ ] Identify data model classes needed
  - [ ] Prototype basic integration
  - [ ] Document architecture for future extensions
- **Deliverables**:
  - FlowAtlas generation callable from UI
  - Basic settings form
- **Outcome**: Can run FlowAtlas from DreamAtlas UI with basic controls
- **Blocked by**: None
- **Feeds into**: F.2, F.3

### F.2 Settings & Rules Management UI
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker (API design)
- **Description**: UI for specifying generation settings (rules, flags, nations)
- **Action items**:
  - [ ] Design rules management UI (list, edit, add, remove, order)
  - [ ] Design flags/nation UI (select nations, set capitals, adjust weights)
  - [ ] Design weight tuning UI (sliders, numeric input, curves)
  - [ ] Create presets/profiles system
  - [ ] User testing for intuitiveness
- **Deliverables**:
  - Rules management UI working
  - Flags/nation configuration working
  - Weight tuning working
- **Outcome**: Users can configure complex generation without code
- **Blocked by**: F.1 (partially), C, D, E (features to expose)
- **Feeds into**: F.3

### F.3 Visualization & Debug Display
- **Status**: Not started
- **Skill**: Developer
- **Description**: Show intermediate generation steps, debug data, metrics
- **Action items**:
  - [ ] Design visualization architecture
  - [ ] Implement basic iteration viewer (animate through generations)
  - [ ] Add entropy/metric plots (from B.3)
  - [ ] Add rule firing logs
  - [ ] Add geographic overlays (Voronoi vertices, mountains, rivers, etc.)
  - [ ] Add nation influence heatmaps (from D.2)
  - [ ] Add travel difficulty scoring display
  - [ ] User testing for usefulness
- **Deliverables**:
  - Interactive iteration viewer
  - Metric dashboard
  - Debug overlays
- **Outcome**: Generation process fully visible and debuggable from UI
- **Blocked by**: F.1 (basic UI), B.3 (debug framework)
- **Feeds into**: Validation and iteration

---

# OUTPUT & EXPORT

## COMPONENT G: Organic Border Rendering & Output

**Goal**: Make borders natural using geographic features; export playable files
**Recommended Agent/Skill**: Developer + Researcher (game format, rendering)

### G.1 Parametric Border Rendering
- **Status**: Not started
- **Skill**: Developer + Code Smell Checker
- **Description**: Convert straight province edges into organic, feature-following paths
- **Strategy**:
  - Use terrain types and Voronoi geometry to inform border shape
  - Rivers meander; mountains follow ridges; coastlines are complex
  - Different border types render differently
- **Action items**:
  - [ ] Design border rendering API
  - [ ] Analyze existing DreamAtlas border rendering
  - [ ] Parametric river paths (meander, width variation)
  - [ ] Parametric mountain paths (ridge-following)
  - [ ] Parametric coastline (fractal/noise-based)
  - [ ] Different aesthetics per border type
  - [ ] Test against game graphics
- **Deliverables**:
  - Border rendering working for river, mountain, coast types
  - Paths follow geographic features naturally
- **Outcome**: Maps look organic and natural
- **Blocked by**: A (coordinates and topology), B.2.3 (knowing border types)
- **Feeds into**: G.2

### G.2 Export to .d6m Format
- **Status**: Not started
- **Skill**: Developer + Researcher (game format)
- **Description**: Convert fully generated TerrainGraph to Dominions 6 .d6m format
- **Action items**:
  - [ ] Study .d6m format requirements
  - [ ] Analyze DreamAtlas .d6m export code
  - [ ] Design TerrainGraph → .d6m mapping
  - [ ] Implement conversion
  - [ ] Test in-game loading
  - [ ] Handle edge cases
- **Deliverables**:
  - Working .d6m exporter
  - Generated maps playable in Dominions 6
- **Outcome**: Maps can be played in-game
- **Blocked by**: A (complete TerrainGraph), G.1 (borders)
- **Feeds into**: Playtesting and iteration

### G.3 Other Output Formats
- **Status**: Not started
- **Priority**: Low
- **Skill**: Developer + Researcher
- **Description**: Support other outputs (images, intermediate data formats, etc.)
- **Examples**:
  - PNG/SVG visualization of generated maps
  - Detailed map data (JSON export for analysis)
  - Links to other game formats (if applicable)
- **Action items**:
  - [ ] Identify desired formats
  - [ ] Design export API
  - [ ] Implement highest-priority formats
- **Outcome**: Flexible export options
- **Blocked by**: A (complete graph), G.2 (export system)

---

# CROSS-CUTTING CONCERNS

## Testing & Validation
- **Skill**: QA + Test Maintainer
- Regression tests for coordinate tracking, Voronoi vertices, distances
- WFC tests for entropy, border distribution, rule firing, national integration
- Integration tests for full pipeline, export, gameplay
- Performance benchmarks

## Documentation
- **Skill**: Developer + Technical Writer
- API documentation for new classes and methods
- Geographic constraints guide (how to write geographic rules)
- National integration guide
- Travel rules guide (composition and tuning)
- UI user guide

---

# Development Strategy & Sequencing

## Quick Decision Guide

**Order matters?** Check if all predecessors are done. If no predecessors, can start immediately.

## Recommended Paths

### Path A: Foundation First (Recommended)
```
A (Terrain) → B.3 (Debug) → B.1 (Fix WFC) → B.2 (Geo Rules)
                                              ↓
                                          C (Refinement)
                                              ↓
                                          D (National)
                                              ↓
                                          E (Advanced)
                                              ↓
                                          F (UI)
                                              ↓
                                          G (Output)
```
Best for: **Small team, stable foundation, iterative improvement**

### Path B: Parallel Development (for larger teams)
```
A  ─┬─→ B.1 (WFC Fix) ─→ B.2 (Geo) ─→ C/D ──┐
    │                                       │
    ├─→ B.3 (Debug) ──────────────────────→ F (UI)
    │                                       │
    └─→ E (Advanced) ───────────────────────┤
                                            ↓
                                        G (Output)
```
Best for: **Larger team, parallel work streams**

---

## How to Use This Document

1. **Finding this file**: Root of repository as `NEXT_STEPS.md`
2. **Reading**: Start with "Quick Navigation" at top to find relevant section
3. **Tracking work**:
   - Update status (Not started → In progress → Completed)
   - Check off `[ ]` action items as you complete them
   - Update "Last Updated" when making changes
   - Link to Issues/PRs where relevant (e.g., "Blocked by: #123")
4. **Committing**: Commit regularly as work progresses
5. **Sharing**: Push to git; collaborators see latest plan

## Notes for Collaborators

- Before starting a component, look at its "Recommended Agent/Skill" section
- Check "Blocked by" to ensure prerequisites are done
- When starting, create feature branch and update this document
- For questions about dependencies or design, open an Issue (reference as "Q: A.1 coordinate system")
- This is a living document; adapt as understanding evolves

---

# Tooling Issues & Session Notes

**To be filled in as we discover tooling problems to address:**

- [ ] Workspace instructions set up (`.github/copilot-instructions.md`)
- [ ] Agent role specialization documented
- [ ] Cross-machine path compatibility verified
- [ ] Conda environment activation documented
