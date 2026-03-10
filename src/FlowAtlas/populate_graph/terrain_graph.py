"""
TerrainGraph and Element classes for wave function collapse-based terrain generation.

These classes wrap NetworkX Graph functionality to provide:
- Element: A lightweight wrapper for node/edge attribute access
- TerrainGraph: A Graph subclass that manages terrain-specific queries and domain constraints

Design principles:
- Maintain compatibility with nx.Graph's dict-like attribute access (.nodes, .edges, graph dict)
- Element provides transparent mutation: element['key'] = value updates the graph in place
- TerrainGraph owns structural queries and domain constraints (no self-loops, global_metrics)
- Both stay lean; complex logic belongs in WaveFunctionCollapse
"""

from typing import Any, Callable

import networkx as nx


class Element:
    """
    Lightweight wrapper for a graph node or edge, supporting in-place attribute mutation.

    Provides dict-like access to attributes while tracking whether the element is a node or edge.
    Mutations are transparent: element['terrain'] = x updates the underlying graph in place.

    Attributes:
        element_id: Node identifier or edge tuple (u, v) or (u, v, key) for multigraphs
        is_node: Boolean; True if this wraps a node, False if it wraps an edge
        graph: Reference to the TerrainGraph (or nx.Graph) containing this element
    """

    def __init__(self, element_id, is_node: bool, graph):
        """
        Initialize an Element wrapper.

        Args:
            element_id: Node ID or edge tuple
            is_node: True for nodes, False for edges
            graph: The graph containing this element
        """
        self.element_id = element_id
        self.is_node = is_node
        self.graph = graph

    @property
    def attrs(self) -> dict:
        """Return the mutable attribute dict for this element."""
        if self.is_node:
            return self.graph.nodes[self.element_id]
        else:
            return self.graph.edges[self.element_id]

    def __getitem__(self, key):
        """Get an attribute value."""
        return self.attrs[key]

    def __setitem__(self, key, value):
        """Set an attribute value (updates graph in place)."""
        self.attrs[key] = value

    def __contains__(self, key):
        """Check if an attribute exists."""
        return key in self.attrs

    def get(self, key, default=None):
        """Get an attribute with a default fallback."""
        return self.attrs.get(key, default)

    def update(self, *args, **kwargs):
        """Update multiple attributes at once (like dict.update)."""
        self.attrs.update(*args, **kwargs)

    def __repr__(self):
        elem_type = "node" if self.is_node else "edge"
        return f"Element({self.element_id}, {elem_type})"

    @staticmethod
    def from_node(node_id, graph):
        """Create an Element wrapper for a node."""
        return Element(node_id, True, graph)

    @staticmethod
    def from_edge(edge_id, graph):
        """Create an Element wrapper for an edge."""
        return Element(edge_id, False, graph)


class TerrainGraph(nx.Graph):
    """
    A NetworkX Graph subclass for terrain generation with built-in global metrics.

    Extends nx.Graph to:
    - Forbid self-loops (not valid for province/border topology)
    - Hold global_metrics (province/border counts, target distributions, etc.)
    - Provide Element-based iteration and queries (get_all_elements, get_unset_elements, is_all_set)
    - Maintain full compatibility with nx.Graph API

    Attributes:
        global_metrics (dict): Global statistics and targets for the generation process
    """

    def __init__(self, settings: dict, *args, **kwargs):
        """Initialize TerrainGraph; forbid self-loops by default."""
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.global_metrics = {}

    @classmethod
    def from_graph(cls, graph: nx.Graph, settings: dict):
        """
        Create a TerrainGraph from an existing nx.Graph.

        Copies graph attributes, node attributes, and edge attributes.
        Self-loops are rejected by TerrainGraph.add_edge.
        """
        terrain_graph = cls(settings)
        terrain_graph.graph.update(graph.graph)
        terrain_graph.add_nodes_from(graph.nodes(data=True))
        terrain_graph.add_edges_from(graph.edges(data=True))
        return terrain_graph

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edge, raising error if it would be a self-loop."""
        if u_of_edge == v_of_edge:
            raise ValueError(
                f"Self-loops are not allowed in TerrainGraph (attempted edge {u_of_edge}-{u_of_edge}). "
                "A province cannot border itself."
            )
        super().add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        """Add multiple edges, checking for self-loops."""
        for edge_data in ebunch_to_add:
            u, v = edge_data[0], edge_data[1]
            if u == v:
                raise ValueError(
                    f"Self-loops are not allowed in TerrainGraph (attempted edge {u}-{u}). "
                    "A province cannot border itself."
                )
        super().add_edges_from(ebunch_to_add, **attr)

    def get_all_elements(self):
        """
        Yield all Element wrappers (nodes first, then edges).

        Yields:
            Element: Wrapper for each node and edge in the graph
        """
        for node_id in self.nodes:
            yield Element.from_node(node_id, self)
        for edge_id in self.edges:
            yield Element.from_edge(edge_id, self)

    def iter_elements(self, data: bool | str = False, is_node: bool | None = None):
        """
        Iterate over elements with optional filtering and data payload.

        Args:
            data:
                - False: yield Element
                - True: yield (Element, attrs_dict)
                - str: yield (Element, element.get(data, None))
            is_node:
                - None: include both nodes and edges
                - True: nodes only
                - False: edges only
        """
        for element in self.get_all_elements():
            if is_node is not None and element.is_node != is_node:
                continue

            if data is True:
                yield element, element.attrs
            elif isinstance(data, str):
                yield element, element.get(data, None)
            else:
                yield element

    def get_connected_elements(
        self,
        element: Element,
        mode: str = 'topology',
        topology_edge_scope: str = 'distance_leq_1',
    ) -> dict[str, Any]:
        """
        Return connected elements for a node or edge element.

        Args:
            element: Element to query neighborhood for.
            mode:
                - 'topology': graph-native adjacency only (implemented)
                - 'geography': Voronoi/triangle-augmented adjacency (not implemented yet)
            topology_edge_scope:
                Applies only when mode='topology' and element is an edge:
                - 'incidence_only': return only the two incident nodes (distance 0.5)
                - 'distance_leq_1': return incident nodes plus adjacent edges (distance <= 1.0)

        Returns:
            dict[str, Any]: Neighborhood payload.

        Notes:
            Use mode='topology' for MVP behavior.
            mode='geography' is intentionally not implemented yet to avoid
            silently returning incorrect semantics.
        """
        if mode == 'topology':
            return self.get_connected_elements_topology(element, edge_scope=topology_edge_scope)

        if mode == 'geography':
            raise NotImplementedError(
                "Geography/Voronoi-augmented adjacency is not implemented yet. "
                "For now, use mode='topology'."
            )

        raise ValueError(f"Unknown mode '{mode}'. Expected 'topology' or 'geography'.")

    def get_connected_elements_topology(self, element: Element, edge_scope: str = 'distance_leq_1') -> dict[str, Any]:
        """
        Return topology-connected elements for a node or edge element.

        Node element:
            - nodes: neighboring node Elements
            - edges: incident edge Elements

        Edge element:
            - edge_scope='incidence_only':
                - nodes: the two incident node Elements
                - edges: []
                - edges_by_node: {}
            - edge_scope='distance_leq_1':
                - nodes: the two incident node Elements
                - edges: neighboring edge Elements that share either endpoint
                - edges_by_node: neighboring edge Elements split by shared endpoint
        """
        if element.is_node:
            node_id = element.element_id
            neighbor_nodes = [Element.from_node(nbr, self) for nbr in self.neighbors(node_id)]
            incident_edges = [Element.from_edge((node_id, nbr), self) for nbr in self.neighbors(node_id)]
            return {
                'nodes': neighbor_nodes,
                'edges': incident_edges,
            }

        u, v = element.element_id
        node_elements = [Element.from_node(u, self), Element.from_node(v, self)]

        if edge_scope == 'incidence_only':
            return {
                'nodes': node_elements,
                'edges': [],
                'edges_by_node': {},
            }

        if edge_scope != 'distance_leq_1':
            raise ValueError(
                f"Unknown edge_scope '{edge_scope}'. Expected 'incidence_only' or 'distance_leq_1'."
            )

        edges_by_u = []
        for nbr in self.neighbors(u):
            edge_id = (u, nbr)
            if edge_id != element.element_id and edge_id[::-1] != element.element_id:
                edges_by_u.append(Element.from_edge(edge_id, self))

        edges_by_v = []
        for nbr in self.neighbors(v):
            edge_id = (v, nbr)
            if edge_id != element.element_id and edge_id[::-1] != element.element_id:
                edges_by_v.append(Element.from_edge(edge_id, self))

        unique_edges = {}
        for edge_elem in edges_by_u + edges_by_v:
            key = tuple(sorted(edge_elem.element_id))
            unique_edges[key] = edge_elem

        return {
            'nodes': node_elements,
            'edges': list(unique_edges.values()),
            'edges_by_node': {
                u: edges_by_u,
                v: edges_by_v,
            },
        }

    def filter_elements(self, predicate: Callable[[Element], bool], is_node=None):
        """Yield elements where predicate(element) is True."""
        for element in self.get_all_elements():
            if is_node is not None and element.is_node != is_node:
                continue
            if predicate(element):
                yield element

    def count_elements(self, predicate: Callable[[Element], bool], is_node=None) -> int:
        """Count elements where predicate(element) is True."""
        return sum(1 for _ in self.filter_elements(predicate, is_node=is_node))

    def get_unset_elements(self):
        """
        Yield all Element wrappers that have terrain == None.

        Useful for WFC: filters to only unsolved elements.

        Yields:
            Element: Wrapper for each unset node and edge
        """
        yield from self.filter_elements(lambda element: element.get('terrain', None) is None)

    def is_all_set(self):
        """
        Check if all nodes and edges have their terrain set.

        Uses global_metrics counters for fast check, falls back to full scan if needed.

        Returns:
            bool: True if all elements have terrain != None
        """
        if (self.global_metrics.get('set_provinces', 0) <
            self.global_metrics.get('provinces', 0)):
            return False
        if (self.global_metrics.get('set_borders', 0) <
            self.global_metrics.get('borders', 0)):
            return False

        # Counters say we're done; verify with a full scan
        for element in self.get_all_elements():
            if element.get('terrain', None) is None:
                return False
        return True

    def setup_element_dists(self, reset_existing_terrain: bool = True):
        """
        Initialize element attributes needed for wave function collapse.

        Sets up 'dists' pointers for each element so they stay in sync with
        global_adjusting_dist when it's updated.

        Args:
            reset_existing_terrain: If False, preserves any pre-set terrain values.
                WARNING: When False, you MUST ensure that global_metrics counters
                (set_provinces, set_borders), element dists, weights, and constraints
                are all properly initialized for the pre-set elements. Currently,
                this function does NOT handle such initialization.
                TODO: Implement full initialization logic for pre-seeded terrains
                (update counters, compute initial dists/weights/constraints).
        """
        if not reset_existing_terrain:
            print("Warning: setup_element_dists called with reset_existing_terrain=False. "
                  "Preserving existing terrain values, but global_metrics counters"
                  "and element dists/weights/constraints may be inconsistent. "
                  "Make sure to initialize those properly for pre-set elements.")

        if 'global_adjusting_dist' not in self.global_metrics:
            return

        global_dist = self.global_metrics['global_adjusting_dist']
        province_terrains = global_dist.get('province_terrains', {})
        border_terrains = global_dist.get('border_terrains', {})

        for element in self.get_all_elements():
            if 'dists' not in element:
                element['dists'] = {}

            # Store pointer to appropriate dist type
            if element.is_node:
                element['dists']['global_adjusting_dist'] = province_terrains
            else:
                element['dists']['global_adjusting_dist'] = border_terrains

            # Optional cleanup of any stale terrain values.
            # Callers can disable this when pre-seeded terrain should be preserved.
            if reset_existing_terrain and element.get('terrain', None) is not None:
                print(f"Warning: element {element} already has terrain {element.get('terrain')}, resetting to None")
                element['terrain'] = None

            # TODO determine if we need to clean other values
