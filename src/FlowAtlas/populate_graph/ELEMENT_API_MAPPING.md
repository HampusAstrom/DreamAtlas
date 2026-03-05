# Element API Mapping for Future NetworkX Integration

This document outlines NetworkX Graph/Node/Edge APIs that may warrant Element wrappers in the future.

**Design Principle:** Element wrappers are added only when:
- They enable uniform node/edge handling (currently not changing graph structure during WFC)
- They provide meaningful abstraction over nx API (e.g., distance queries, neighbor filtering)
- They reduce boilerplate in WFC algorithm (e.g., "get all unset elements within N hops")

Other nx APIs remain accessible directly via `element.graph.nodes[...]`, `element.graph.edges[...]`, etc.

---

## Current (Implemented)

✅ **Dict-like attribute access** (Element class)
- `element['terrain']` → `element.attrs['terrain']`
- `element.get('terrain', default)` → `element.attrs.get(...)`
- `element['key'] = value` → in-place mutation

✅ **Iteration & Filtering** (TerrainGraph methods)
- `graph.get_all_elements()` → yield all Element wrappers (nodes then edges)
- `graph.get_unset_elements()` → yield Elements where terrain==None
- `graph.iter_elements(data=True, is_node=None)` → flexible iteration with optional data payload
- `graph.filter_elements(predicate, is_node=None)` → filter by custom predicate
- `graph.count_elements(predicate, is_node=None)` → count matching elements

✅ **Connected-element query** (TerrainGraph method)
- `graph.get_connected_elements(element, mode='topology', topology_edge_scope='distance_leq_1')`
  - implemented mode: `'topology'`
  - not implemented mode: `'geography'` (raises `NotImplementedError` intentionally)
- `graph.get_connected_elements_topology(element, edge_scope='distance_leq_1')`
  - node element: neighboring nodes + incident edges
  - edge element:
    - `edge_scope='incidence_only'`: incident nodes only (distance `0.5`)
    - `edge_scope='distance_leq_1'`: incident nodes + neighboring edges, including `edges_by_node` partition (distance `<=1.0`)

See also [MAP_DISTANCE_AND_ADJACENCY_NOTES.md](MAP_DISTANCE_AND_ADJACENCY_NOTES.md) for map-specific distance semantics, half-step traversal, and future Voronoi-aware neighborhood design.

---

## Future Candidates (Not Yet Implemented)

### Neighbor/Connectivity Queries (High Priority)

These are essential for WFC constraint propagation and distance-weighted preferences.

**For Node Elements:**
- `element.neighbors()` → yield adjacent node Elements
  - Would wrap `graph.neighbors(node_id)` but return Element wrappers
- `element.neighbors_at_distance(n: int)` → yield all nodes exactly N steps away with their distance
  - Useful for "nearby terrain influences probability"
  - Current: requires manual BFS outside WFC
  - Example: `for neighbor, dist in element.neighbors_at_distance(2): ...`

**For Edge Elements:**
- `element.incident_nodes()` → yield the two node Elements this edge connects
  - Would wrap edge tuple unpacking + graph.nodes[...] lookups

**Cross-element:**
- `element.is_adjacent_to(other_element: Element) -> bool`
  - Works for both node-node (via edge existence) and edge-node (via incidence)

### Attribute Batch Operations (Medium Priority)

- `graph.set_elements_attr(predicate, key, value)`
  - Batch set attribute on all matching elements (optimizes common pattern)
  - Example: `graph.set_elements_attr(lambda e: e.is_node, 'visited', False)`

### Constraint/State Queries (Medium Priority)

- `element.is_fully_constrained() -> bool`
  - True if `element['constraints']` bans all remaining terrains
  - Helps detect impossible states early

- `element.possible_terrains() -> set`
  - Returns set of terrains not banned by constraints
  - Current: extracted inside `calculate_joint_probability_distribution`

### Housekeeping (Lower Priority)

- `element.clear_attr(*keys)` → safely delete attributes
- `element.validate_dists()` → check that all required dist pointers exist

---

## Non-Candidates (Direct nx API Sufficient)

These remain on `element.graph` or accessed directly because they don't abstract away complexity:

- Graph topology modification (`add_node`, `add_edge`, `remove_node`, etc.) — restricted by TerrainGraph; use graph directly
- Shortest path queries (`shortest_path`, `all_shortest_paths`) — use nx directly; not element semantic
- Centrality/layout algorithms — use nx directly; out of scope for WFC
- Serialization/persistence — use nx directly

---

## Implementation Notes

1. **Distance queries should be cached** if called frequently (e.g., per WFC iteration). Consider storing as lazy property on Element.

2. **Predicate-based filtering** is general; define commonly-needed predicates as static methods on Element or TerrainGraph for clarity:
   ```python
   @staticmethod
   def is_unset(element: Element) -> bool:
       return element.get('terrain', None) is None
   ```

3. **Preserve backward compatibility** — always allow direct access via `element.graph.nodes[element.element_id]` for nx compatibility.
