# Map Distance and Adjacency Notes for Element-Based Graph Queries

This note captures planned map-centric adjacency and distance semantics for `TerrainGraph`/`Element`.

Goal: keep NetworkX behavior as baseline while adding explicit domain methods for provinces (nodes) and borders (edges).

Priority note: this document supports, but does not replace, the MVP roadmap in the large TODO block at the top of `wave_function_collapse.py` (lines 7-108 in that file at time of writing). MVP topology behavior comes first.

---

## 1) Adjacency Semantics We Need

Status:
- Implemented now:
  - `TerrainGraph.get_connected_elements(element, mode='topology')`
  - `TerrainGraph.get_connected_elements_topology(element)`
- Not implemented:
  - `mode='geography'` in `get_connected_elements(...)`
  - Voronoi-vertex/triangle augmented adjacency

### A. Node-centered adjacency
For a node/province `P`, we need:
- Neighbor nodes/provinces (standard graph adjacency)
- Incident edges/borders connected to `P`
- Optionally both at once in one method call

### B. Edge-centered adjacency
For an edge/border `B=(u,v)`, we need:
- Incident nodes/provinces (`u`, `v`)
- Nearby edges/borders that share an endpoint with `B`
- Endpoint-separated edge neighbors:
  - edges sharing `u`
  - edges sharing `v`

This endpoint-separated representation is useful in map logic where each border has two "sides".

Three explicit edge neighborhood cases:
- Case E1 (incidence-only, distance `0.5` only):
  - include just connected nodes `u` and `v`
- Case E2 (topology, distance `<=1.0`):
  - include `u`, `v` and endpoint-sharing edges
  - this is equivalent to collecting elements in topology half-step distance band `0.5` to `1.0`
- Case E3 (geography/Voronoi vertex-sharing, future):
  - for edge `[u, v]`, include:
    - primary nodes: `u`, `v` (expected 2)
    - triangular nodes: nodes sharing an edge with both `u` and `v` (expected 2 in simple triangulations)
    - triangular edges: edges from triangular nodes to `u`/`v` (expected 4 in simple triangulations)

Status mapping to methods:
- Implemented now:
  - E1/E2 via `get_connected_elements(..., mode='topology', topology_edge_scope=...)`
    - E1: `topology_edge_scope='incidence_only'`
    - E2: `topology_edge_scope='distance_leq_1'`
- Not implemented:
  - E3 via `mode='geography'` (currently raises `NotImplementedError`)

### C. Voronoi-specific edge relationships (future)
Graph-only structure does **not** encode all Voronoi geometric relations.

Future map geometry integration should support:
- Borders that share a Voronoi vertex but may not be graph-adjacent by endpoint semantics
- Triangle-based neighborhood groups (node/edge triples around geometric triangles)

These need an auxiliary geometric index (Voronoi vertices, ridge incidence), not only `nx.Graph` topology.

Design guidance for 1.C (clean/robust/efficient):
- Preferred source of truth: feed geometry metadata from earlier generation stages (Delaunay/Voronoi builders) into `TerrainGraph`.
- Avoid reconstructing Voronoi relations from topology alone; it is ambiguous and error-prone.
- Use explicit metadata containers (examples):
  - graph-level: `graph.graph['geometry_meta']`
  - edge-level: `graph.edges[u, v]['shared_voronoi_vertices']`, `graph.edges[u, v]['triangle_neighbors']`
  - node-level: `graph.nodes[n]['point']` (or wrapped point refs)
- For toroidal / multi-plane maps: store canonical IDs and transformation/context info rather than raw duplicated points only.
- Keep geometry metadata optional; topology APIs must still work when absent.

---

## 2) Distance Metrics: Two Modes

Status:
- Implemented now: none for range-distance APIs.
- Not implemented:
  - `iter_elements_in_range(..., mode='topology')`
  - `iter_elements_in_range(..., mode='geography')`

### Mode 1: Topology-walk distance (graph-native)
- Traverse graph structure only
- Proposed convention: node↔edge transition costs `0.5`
- Consequences:
  - node→adjacent edge = 0.5
  - node→adjacent node through one edge = 1.0
  - edge→adjacent edge sharing one node = 1.0

This mode is easy to compute and stable with NetworkX primitives.

### Mode 2: Geometry-augmented distance (Voronoi-aware)
Includes triangle/vertex relations from map geometry in addition to graph topology.

Open design questions (intentionally undecided for now):
- Distance for node/edge pairs that share only a Voronoi vertex
- Distance for edge/edge pairs in same geometric triangle but not sharing an endpoint

Recommendation: expose these as configurable policy constants, not hard-coded.

---

## 3) Suggested API Shape

Status:
- Implemented now:
  - `get_connected_elements(..., mode='topology')`
  - `get_connected_elements_topology(...)`
- Not implemented:
  - `iter_elements_in_range(...)`
  - any geometry-augmented range queries

### Immediate API (topology-only, implementable now)
- `get_connected_elements(element)`
  - for node: neighbor nodes + incident edges
  - for edge: incident nodes + neighboring edges (with endpoint partition)
- `iter_elements_in_range(start, max_distance, step_mode='half_step')`
  - returns elements with computed distance
  - initial implementation can be topology-only

### Future API (geometry-aware)
- `iter_elements_in_range(..., mode='topology'|'topology_plus_geometry')`
- Internal geometry provider/index for Voronoi vertex and triangle neighborhoods

---

## 4) Caching Guidance

Status:
- Implemented now: no cache layer.
- Not implemented: cache + invalidation hooks.

Default preference: no caching unless profiling shows clear bottleneck.

If needed, cache only derived neighborhood/range lookups and invalidate on:
- any topology mutation (`add/remove node/edge`)
- any geometry index update

Avoid eager caching to reduce drift risk.

Topology-change policy when geometry metadata exists (for future implementation):
- Option A (recommended MVP): forbid topology mutation on geometry-annotated TerrainGraphs.
- Option B: allow mutation but mark geometry metadata invalid and warn loudly.
- Option C: fully recompute metadata on mutation (likely too expensive/complex for MVP).

---

## 5) Path Traversal Cost (Future)

Status:
- Implemented now: none.
- Not implemented: all traversal-cost models.

Later path metrics should support concurrent cost models:
- Node entry/exit terrain costs by nation capability
- Edge traversability/cost constraints by border terrain
- Multiple simultaneous cost functions for different balancing objectives

Recommendation:
- Keep distance and traversal-cost APIs separate
- Distance = structural proximity
- Cost = gameplay movement model

---
