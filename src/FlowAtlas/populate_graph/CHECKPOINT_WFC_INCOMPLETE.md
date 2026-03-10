# CHECKPOINT: Wave Function Collapse Implementation Status

**Date**: March 5, 2026
**Status**: INCOMPLETE - MVP CORE FUNCTIONALITY MISSING
**Delete this file once you clear this checkpoint**

## ADDED IMPORTANT TODOS
- Set an random processes once, and make sure that instance is used consistently,
  and determine if default should be a fixed seed or a random one
- Make sure that all tests use fixed seeds, and if test for random behavior,
  they should maybe (have an option to) run with a larger fixed set of seeds
  (always if, they are few and short enough).

## Current State Summary

The WFC system has a **basic structure in place** but is **deceptively half-complete**:
- ✅ Graph infrastructure (Element, TerrainGraph) working correctly
- ✅ Main loop and entropy-based selection logic functional
- ❌ **CRITICAL: `update_statistics_and_probabilities()` is a stub** — only increments counters
- ❌ **CRITICAL: No neighbor probability updating or constraint propagation**
- ❌ **CRITICAL: No contradiction detection or backtracking**
- ⚠️ Partial: Some starred tasks in comment plan appear done but lack full implementation

---

## Glaring Issues Before Completion

### 1. **`update_statistics_and_probabilities()` is Non-Functional (BLOCKER)**
**File**: [wave_function_collapse.py](wave_function_collapse.py#L256-L266)
**Status**: Stub only

**Current implementation**:
```python
def update_statistics_and_probabilities(element: Element, graph: TerrainGraph, global_metrics: dict):
    # Update global counters
    if element.is_node:
        global_metrics['set_provinces'] = global_metrics.get('set_provinces', 0) + 1
    else:
        global_metrics['set_borders'] = global_metrics.get('set_borders', 0) + 1

    # TODO we need to do a lot more here
    # For now, just a placeholder that compiles
    joint_prob_dist = calculate_joint_probability_distribution(element, global_metrics)
    # TODO: update neighbor probabilities, constraint propagation, etc.
```

**Missing**:
- ❌ Update `global_metrics['current_dist']` to track actual terrain distribution
- ❌ Calculate how far current_dist is from target_dist
- ❌ Update `global_metrics['global_adjusting_dist']` based on distance to target
- ❌ Get neighbors of the set element and update their probability distributions
- ❌ Apply constraint propagation (if a border must be river, adjacent provinces must accept water)
- ❌ Check for contradictions (any element reaching 0 probability)
- ❌ Update cached joint probability distributions for affected neighbors

**Impact**: The algorithm **will generate maps with NO adherence to target terrain distributions**, because:
- No one is tracking how many plains/forests/deserts are actually assigned
- No one is adjusting future selections to steer back toward target distribution
- Neighbors aren't influenced by already-set neighbors
- Terrain compatibility constraints are completely ignored

---

### 2. **No Constraint Setting Methods Created (BLOCKER)**
**Status**: Missing entirely

The code assumes `element['constraints']` exists and has been populated through some mechanism, but nowhere in the code sets these up beyond minimal support in `preprocess_graph()` via `graph.setup_element_dists()`.

**Missing functions**:
- ❌ `setup_terrain_realism_constraints()` — define which terrains can co-exist (e.g., sea ↔ land provinces must have water/normal borders)
- ❌ `setup_national_constraints()` — enforce nation-specific rules (e.g., kingdom of undead wants swamp/waste, empire of arcane wants highlands)
- ❌ `setup_special_constraints()` — handle edges like "capital must not have impassable borders" or "provinces touching sea must have sea terrain"
- ❌ Generic constraint updater — when one element is set, propagate constraints to neighbors

**Impact**: All constraint-based terrain generation is impossible; maps cannot enforce basic realism (land provinces with water borders, underwater terrain adjacencies, etc.)

---

### 3. **Conditional Probability Updating from Neighbors Not Implemented (CRITICAL)**
**Starred Task**: 4.4 from comment plan
**Status**: Completely missing

The plan specifies:
> "update conditional probabilities for all unset terrains in range of set node/border"
> "based on distance (closer has more influence, most might only care about immediate neighbors) and terrains realism"

**Missing**:
- ❌ When an element is set to a terrain, iterate through neighbors at various distances
- ❌ Adjust their probability distributions to favor compatible terrains (e.g., if a province is sea, neighbors should have higher weight for water/coastal borders)
- ❌ Distance-weighted probability influence (immediate neighbors > 2-hop neighbors > further)
- ❌ Realism/compatibility weighting (e.g., plains + forest = reasonable, but mountains + sea border = needs mountain_pass)

**Current state**: Every unset element's probability is recalculated from scratch each iteration, with zero feedback from neighbors' assigned terrains.

**Impact**: WFC becomes purely distribution-based with zero spatial coherence. Maps will have:
- Sea provinces surrounded by plains (geographically nonsensical)
- Swamps next to deep deserts
- Random scattered rivers disconnected from water provinces
- No natural-looking terrain clustering

---

### 4. **No Contradiction Detection / Backtracking (CRITICAL)**
**Starred Task**: 4.5 from comment plan
**Status**: Completely missing

The plan specifies multiple strategies:
- *(one solution beyond warn needed)*
- 4.5.1 *warn/log problem*
- 4.5.2 *raise error*
- 4.5.3 *backtracking to previous state and trying different option (with bailout after some tries)*
- 4.5.4 *if no options left, backtrack further, or restart with new seed (with bailout after some tries)*
- 4.5.5 *override and set by some base probability*

**Current state**: None of the above exist.

**Missing**:
- ❌ Contradiction detection (check if any unset element has 0 probability left)
- ❌ State snapshots for backtracking
- ❌ Decision stack to undo previous choices
- ❌ Bailout limits on backtrack depth and restart attempts
- ❌ Fallback logic if backtracking exhausted

**What happens now**: If selection logic paints the map into a corner, the algorithm **will hang or crash** with no recovery mechanism.

---

### 5. **No Terrain Realism / Compatibility Logic (MAJOR FEATURE MISSING)**
**Status**: Stub with TODOs

Referenced in multiple places:
- "conditional probabilities from nearby set nodes/borders, base on... *terrains realism*"
- Nowhere in code does anything check "is this terrain combination reasonable?"

**Missing**:
- ❌ Terrain compatibility matrix (which terrains can be adjacent to which)
- ❌ Terrain-border mapping (e.g., sea provinces need water/normal borders, mountains need passes)
- ❌ Dynamic realism weighting (adjust probabilities based on neighbor terrains)

**Impact**: No realism enforcement; algorithm doesn't know or care if combinations make sense.

---

### 6. **National Target Distributions Not Implemented (PARTIAL FEATURE)**
**Starred Task**: 2.2, 2.6 from comment plan
**Status**: Commented as TODO, structure absent

Line in `determine_target_distributions()`:
```python
# TODO setup national target_dist and adjusting_dist
# set global dists as a weighted average of national dists and
# a base global dist
```

**Missing**:
- ❌ Parse settings for per-nation terrain preferences
- ❌ Create per-nation adjusting_dist dicts
- ❌ Compute national "nearby" preferences (e.g., "fire nation wants lava/waste in periphery")
- ❌ Weight global_adjusting_dist as combination of global base + national weighted average

**Impact**: No per-nation terrain influence; all nations get identical terrain distribution regardless of class/preference.

---

### 7. **Adjusting Distribution Weighting Not Implemented (PARTIAL FEATURE)**
**Status**: Set up but never updated based on constraint violations

The adjusting_dist is computed once in `determine_target_distributions()`, but then:
- ❌ Never updated as current_dist diverges from target_dist
- ❌ Never weight up toward forced terrains (e.g., if we've assigned 50 plains and need 100, don't increase plains weight)
- ❌ Never applies the "proportional distance error" weighting described in plan section 2.4

Current behavior:
```python
global_metrics['global_adjusting_dist'] = global_metrics.get('global_target_dist', {}).copy()
```
This just copies target_dist once and never changes it.

**Impact**: No dynamic steering toward target distributions; if early random choices skew toward one terrain 2, late selections can't correct course.

---

### 8. **Distance/Range Calculation Not Integrated (INFRASTRUCTURE MISSING)**
**Status**: Discussed in docs, not implemented in WFC logic

The system doesn't have a built-in "get all neighbors at distance N" query, which is needed for:
- Conditional probability updates
- National region definitions
- Local constraint propagation

**Missing**:
- ❌ Integration with `TerrainGraph.iter_elements_in_range()` (if it exists, not checked yet)
- ❌ Distance weighting in probability updates
- ❌ Setup of neighbor caches for efficiency

**Impact**: Neighbor probability updates (section 3) can't work effectively without this.

---

### 9. **No Caching of Joint Probability Distributions (PERFORMANCE ISSUE)**
**Status**: Known TODO but not addressed

In `select_element_to_set()` and `select_element_terrain()`:
```python
# TODO, instead of calculating the joint probability distribution for each element here,
# we should store it in the element and only update it when needed, which should be much more efficient
```

**Current state**: Every iteration, every unset element's joint distribution is recalculated from scratch.

For large maps (e.g., 5000 provinces × 1000 iterations), this is thousands of distributions computed per iteration.

**Impact**: Slow algorithm; will be unusable for large maps without this optimization.

---

### 10. **Preprocess Graph Implementation is Minimal (PARTIAL FEATURE)**
**Status**: Calls `setup_element_dists()` but doesn't do much else

The plan specifies:
```
3.1 if any found, calculate any initial macro statistics (compared to target_dist)
3.2 determine conditional probabilities for all unset terrains in range of set nodes/borders
*(if not rest implemented, just clean of any values)*
```

**Current state** (line 293-295):
```python
print("- add a 'pointer' to global_adjusting_dist in dict 'dists' for element (node/province and edge/border), making sure to use the respective dicts for each only")

# Use TerrainGraph's built-in setup method
graph.setup_element_dists(global_metrics)
```

This delegates everything to `setup_element_dists()`, which may or may not fully initialize element dists/weights/constraints.

---

## Starred Tasks Status from Comment Plan

| Task | Status | Issues |
|------|--------|--------|
| 1. *Collect global metrics | ✅ DONE | Basic; works |
| 1.2 *# provinces, # borders | ✅ DONE | Returned in dict |
| 2. *Determine target distributions | ⚠️ PARTIAL | Only global, no national; see issue #6 |
| 2.1 *global target_dist | ⚠️ PARTIAL | Read from settings; no computation |
| 2.5 *setup initial global adjusting_dist | ⚠️ PARTIAL | Set once, never updated; see issue #7 |
| 3. *Preprocess incoming graph | ⚠️ PARTIAL | Delegates to setup_element_dists; see issue #10 |
| 4. *Start wave function collapse | ✅ STRUCTURE DONE | Main loop valid, but sub-steps broken |
| 4.1 *Select node by entropy | ✅ DONE | Works; selects lowest-entropy unset element |
| 4.2 *Select and set terrain using probs | ✅ DONE | Uses joint dist; works |
| 4.3 Update global/local dist stats | ❌ NOT DONE | See issue #1 (update_statistics stub) |
| 4.4 *Update conditional probs for neighbors | ❌ NOT DONE | See issue #3 |
| 4.5 *Handle contradictions | ❌ NOT DONE | See issue #4 |
| 4.6 *Repeat until all nodes set | ✅ STRUCTURE DONE | Loop works; no iterations work right |
| Contributions: *base adjusting_dist | ⚠️ PARTIAL | Set up but never updated |
| Contributions: *conditional probs from neighbors | ❌ NOT DONE | See issue #3 |
| Contributions: *distance weighting | ❌ NOT DONE | See issue #3, #8 |
| Contributions: *terrains realism | ❌ NOT DONE | See issue #5 |

---

## What This Means in Practice

If you run the current code on a graph:

1. ✅ It will **enter the main loop** and start selecting elements.
2. ✅ It will **pick an element** (lowest entropy selection works).
3. ✅ It will **assign a terrain** (from global target distribution, ignoring neighbors).
4. ⚠️ It will **pretend to update** — counter increments but nothing else happens.
5. ⚠️ It will **repeat** with zero feedback from previously set elements.
6. ✅ Eventually it will **complete** and return a map.
7. ❌ **But the map will be terrible**:
   - No terrain clustering or spatial coherence
   - Random distribution bearing no relation to requests or realism
   - No guarantee that target distributions are met
   - Potentially geographically nonsensical (sea surrounded by deserts, etc.)
   - No nation-specific rules enforced
   - If an impossible state is reached, **algorithm hangs or crashes**

---

## Minimum Required to Make MVP Work

To move past "deceptively half-done" to "actually working MVP":

### Tier 1 (Correctness)
1. Implement `update_statistics_and_probabilities()` to:
   - ✓ Track `current_dist` (histogram of assigned terrains)
   - ✓ Update `global_adjusting_dist` based on (target_dist - current_dist)
   - ✓ Call neighbor probability updater
   - ✓ Detect contradictions

2. Implement basic contradiction detection:
   - ✓ After setting an element, check if any unset element has 0 probability left
   - ✓ If found, raise error with context (don't silently hang)

3. Implement basic neighbor probability updating:
   - ✓ When element is set, iterate through its neighbors
   - ✓ Adjust their joint distributions (e.g., if sea → increase water-border probability)
   - ✓ Use at least 1-hop neighbors; distance weighting optional for MVP

4. Implement basic terrain realism:
   - ✓ Hard-code at least one realism matrix (e.g., sea provinces must have water/normal borders)
   - ✓ Apply as constraints in `calculate_joint_probability_distribution()`

### Tier 2 (Robustness)
5. Add state snapshots and basic backtracking (single-level):
   - ✓ Save state before selecting/setting
   - ✓ If contradiction found, revert and try next best option
   - ✓ Bailout after N failed attempts at same decision point

6. Implement `setup_terrain_realism_constraints()`:
   - ✓ Define terrain compatibility rules
   - ✓ Populate element['constraints'] appropriately

### Tier 3 (Features)
7. Implement national distributions (per-nation adjusting_dist)
8. Implement distance-weighted neighbor influence
9. Add caching + invalidation of joint distributions
10. Add traversal-cost / movement-balance logic

---

## Commitment Assessment

**CAN THIS CODE BE COMMITTED?** Yes, but **with caveats**.

### Recommended Commit Strategy

```
Commit as: "WFC: MVP structure with placeholder probability updates (non-functional)"

Files:
- wave_function_collapse.py
- try_wave_function_collapse.py (if included)

Tags/Branch: Feature branch, not main

Commit message format:
  WFC: Implement MVP structure with TODO placeholders

  - Graph structure (Element, TerrainGraph) fully functional
  - Main WFC loop, entropy selection, terrain assignment: working
  - update_statistics_and_probabilities: STUB ONLY (no prob updates)
  - Neighbor probability propagation: NOT IMPLEMENTED
  - Contradiction detection/backtracking: NOT IMPLEMENTED
  - Terrain realism constraints: NOT IMPLEMENTED
  - National distributions: NOT IMPLEMENTED

  The code will run and complete a map, but will produce poor results
  because probability updates are non-functional. All starred TODO items
  in wave_function_collapse.py comment plan (section 4.3-4.5) are marked
  but not implemented. See CHECKPOINT_WFC_INCOMPLETE.md for details.

  Do not use output maps for actual content yet; internal testing only.
```

### Mark this in Code

Add a prominent banner at TOP of `wave_function_collapse.py`:

```python
"""
⚠️ WARNING: Wave Function Collapse Implementation INCOMPLETE

This module provides WFC structure and main loop, but CRITICAL probability
update and constraint logic is stubbed. Maps generated will be non-sensical.

For full feature checklist and known issues, see:
  CHECKPOINT_WFC_INCOMPLETE.md

DO NOT use output for production until Tier 1 items completed.
"""
```

### In `setup.py` or package version

If versioning, mark as:
- `__version__ = "0.0.1a1"` (alpha, pre-release)
- Not suitable for stable release

---

## How to Clear This Checkpoint

1. Implement Tier 1 items above (correctness)
2. Delete `CHECKPOINT_WFC_INCOMPLETE.md`
3. Update `wave_function_collapse.py` warning banner
4. Verify with test maps that output makes sense
5. Merge to main with full feature set

---

## References

- **Plan**: Lines 7–108 in wave_function_collapse.py
- **Starred tasks**: Marked with `*` throughout plan comments
- **Main loop**: `wave_function_collapse()` method around line 320
- **Critical stub**: `update_statistics_and_probabilities()` lines 256–266
