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

import math
from collections.abc import Iterator
from typing import Any, Callable, Generator, Literal

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

    @property
    def is_province(self) -> bool:
        """True if this element represents a province (node)."""
        return self.is_node

    @property
    def is_border(self) -> bool:
        """True if this element represents a border (edge)."""
        return not self.is_node

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

    # Note: We intentionally do not implement __iter__ to yield Elements,
    # as that would break compatibility with nx.Graph.
    # Instead, we provide explicit methods like get_all_elements() and iter_elements() for Element
    # def __iter__(self) -> Iterator[Element]:
    #     """Iterate over all elements (nodes first, then edges)."""
    #     yield from self.get_all_elements()

    def __contains__(self, n: object) -> bool:
        """Check if an element (node or edge) is in the graph."""
        try:
            assert isinstance(n, Element)
            if n.is_node:
                return n.element_id in self.nodes
            else:
                return n.element_id in self.edges
        except TypeError:
            return False

    def __len__(self) -> int:
        """Return total number of elements (nodes + edges)."""
        return self.number_of_nodes() + self.number_of_edges()

    # TODO consider if we want our own version of __getitem__ or not

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

    def get_all_elements(self) -> Generator[Element, Any, None]:
        """
        Yield all Element wrappers (nodes first, then edges).

        Yields:
            Element: Wrapper for each node and edge in the graph
        """
        for node_id in self.nodes:
            yield Element.from_node(node_id, self)
        for edge_id in self.edges:
            yield Element.from_edge(edge_id, self)

    def iter_elements(self, data: bool | str = False, is_node: bool | None = None) -> Generator[Any, None, None]:
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
        range: float = 1.0,
        range_check: Literal['eq', 'leq'] = 'leq',
        element_kind: Literal['nodes', 'edges', 'both'] = 'both',
        include_distance: bool = False,
    ) -> list[Any]:
        """
        Return connected elements for a node or edge element.

        Args:
            element: Element to query neighborhood for.
            mode:
                - 'topology': graph-native adjacency only (implemented)
                - 'geography': Voronoi/triangle-augmented adjacency (not implemented yet)
            range:
                Topological range measured on the element-incidence graph.
                Each node<->incident-edge hop has length 0.5.
            range_check:
                - 'leq': include elements with distance <= range
                - 'eq': include elements with distance approximately equal to range
            element_kind:
                - 'nodes': include province elements only
                - 'edges': include border elements only
                - 'both': include both
            include_distance:
                If True, return (Element, distance) pairs.
                If False, return Element objects.

        Returns:
            list[Any]: Neighbor elements (or (element, distance) pairs).

        Notes:
            Use mode='topology' for MVP behavior.
            mode='geography' is intentionally not implemented yet to avoid
            silently returning incorrect semantics.
            Future design needs to decide whether this API should also support
            filtering by distance / element type directly, and whether returned
            Elements should optionally include an explicit range/distance value.
        """
        if mode == 'topology':
            return self.get_connected_elements_topology(
                element,
                range=range,
                range_check=range_check,
                element_kind=element_kind,
                include_distance=include_distance,
            )

        if mode == 'geography':
            raise NotImplementedError(
                "Geography/Voronoi-augmented adjacency is not implemented yet. "
                "For now, use mode='topology'."
            )

        raise ValueError(f"Unknown mode '{mode}'. Expected 'topology' or 'geography'.")

    def _iter_incidence_neighbors(self, element: Element) -> Generator[tuple[Element, float], None, None]:
        """Yield immediate incidence neighbors and edge weight (always 0.5)."""
        if element.is_node:
            node_id = element.element_id
            for nbr in self.neighbors(node_id):
                yield Element.from_edge((node_id, nbr), self), 0.5
            return

        u, v = element.element_id
        yield Element.from_node(u, self), 0.5
        yield Element.from_node(v, self), 0.5

    def get_connected_elements_topology(
        self,
        element: Element,
        range: float = 1.0,
        range_check: Literal['eq', 'leq'] = 'leq',
        element_kind: Literal['nodes', 'edges', 'both'] = 'both',
        include_distance: bool = False,
    ) -> list[Any]:
        """
        Return topology-connected elements for a node or edge element.

        Contract:
            - Topology neighborhoods are computed on the incidence graph where
              node<->edge adjacency has distance 0.5.
            - This method owns topology-aware distance semantics and optional
              node/edge filtering.
            - iter_elements/filter_elements remain graph-wide generic iterators
              and do not encode topological range semantics.
        """
        if range < 0:
            raise ValueError(f"range must be >= 0, got {range}")
        if range_check not in {'eq', 'leq'}:
            raise ValueError(f"Unknown range_check '{range_check}'. Expected 'eq' or 'leq'.")
        if element_kind not in {'nodes', 'edges', 'both'}:
            raise ValueError(f"Unknown element_kind '{element_kind}'. Expected 'nodes', 'edges', or 'both'.")

        tolerance = 1e-9
        frontier = [(0.0, element)]
        visited: dict[tuple[bool, Any], float] = {(element.is_node, element.element_id): 0.0}

        while frontier:
            current_dist, current = frontier.pop(0)

            for neighbor, step_cost in self._iter_incidence_neighbors(current):
                new_dist = current_dist + step_cost
                if new_dist > range + tolerance:
                    continue

                key = (neighbor.is_node, neighbor.element_id)
                old_dist = visited.get(key, math.inf)
                if new_dist + tolerance < old_dist:
                    visited[key] = new_dist
                    frontier.append((new_dist, neighbor))

        # Remove origin and build filtered result payload.
        origin_key = (element.is_node, element.element_id)
        visited.pop(origin_key, None)

        items: list[tuple[Element, float]] = []
        for (is_node, element_id), dist in visited.items():
            if range_check == 'eq' and not math.isclose(dist, range, rel_tol=0.0, abs_tol=tolerance):
                continue
            if element_kind == 'nodes' and not is_node:
                continue
            if element_kind == 'edges' and is_node:
                continue

            neighbor = Element.from_node(element_id, self) if is_node else Element.from_edge(element_id, self)
            items.append((neighbor, dist))

        items.sort(key=lambda pair: pair[1])
        if include_distance:
            return items
        return [elem for elem, _ in items]

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
