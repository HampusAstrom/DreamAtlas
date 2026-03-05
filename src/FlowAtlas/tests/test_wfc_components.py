"""
Unit tests for TerrainGraph and WaveFunctionCollapse components.

Test coverage:
- Element attribute access and mutation
- TerrainGraph creation and conversion
- TerrainGraph element querying and filtering
- Self-loop prevention
- WaveFunctionCollapse initialization and setup
- Global metrics management
"""

import unittest
import networkx as nx
from FlowAtlas.populate_graph.terrain_graph import Element, TerrainGraph
from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse, collect_global_metrics


class TestElement(unittest.TestCase):
    """Test Element wrapper for node/edge attributes."""

    def setUp(self):
        """Create a simple TerrainGraph for testing."""
        self.graph = TerrainGraph()
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")

    def test_element_from_node(self):
        """Test creating Element wrapper for a node."""
        element = Element.from_node("A", self.graph)
        self.assertTrue(element.is_node)
        self.assertEqual(element.element_id, "A")

    def test_element_from_edge(self):
        """Test creating Element wrapper for an edge."""
        element = Element.from_edge(("A", "B"), self.graph)
        self.assertFalse(element.is_node)
        self.assertEqual(element.element_id, ("A", "B"))

    def test_element_dict_like_access(self):
        """Test dict-like attribute access on Element."""
        element = Element.from_node("A", self.graph)

        # Set and get
        element['terrain'] = 'forest'
        self.assertEqual(element['terrain'], 'forest')
        self.assertEqual(self.graph.nodes["A"]['terrain'], 'forest')

        # Get with default
        self.assertIsNone(element.get('missing_key', None))
        self.assertEqual(element.get('missing_key', 'default'), 'default')

    def test_element_in_place_mutation(self):
        """Test that Element mutations update the graph in place."""
        element = Element.from_node("A", self.graph)
        element['data'] = {'value': 42}

        # Verify graph was updated in place
        self.assertEqual(self.graph.nodes["A"]['data']['value'], 42)

    def test_element_contains(self):
        """Test __contains__ operator."""
        element = Element.from_node("A", self.graph)
        element['key'] = 'value'

        self.assertIn('key', element)
        self.assertNotIn('missing', element)

    def test_element_update(self):
        """Test dict-like update() method."""
        element = Element.from_node("A", self.graph)
        element.update({'foo': 1, 'bar': 2})

        self.assertEqual(element['foo'], 1)
        self.assertEqual(element['bar'], 2)
        self.assertEqual(self.graph.nodes["A"]['foo'], 1)


class TestTerrainGraphConversion(unittest.TestCase):
    """Test TerrainGraph creation and conversion from nx.Graph."""

    def test_from_graph_creates_copy(self):
        """Test that from_graph creates a new TerrainGraph without modifying original."""
        original = nx.Graph()
        original.add_node("X", data="value")
        original.add_edge("X", "Y", weight=5)

        terrain_graph = TerrainGraph.from_graph(original)

        # Check that new graph exists
        self.assertIsInstance(terrain_graph, TerrainGraph)
        self.assertIsNot(terrain_graph, original)

        # Check that data was copied
        self.assertEqual(terrain_graph.nodes["X"]['data'], "value")
        self.assertEqual(terrain_graph.edges["X", "Y"]['weight'], 5)

        # Check that original is unchanged
        self.assertEqual(len(original.nodes), 2)

    def test_from_graph_copies_graph_attrs(self):
        """Test that graph-level attributes are copied."""
        original = nx.Graph()
        original.graph['name'] = 'test_graph'
        original.add_node("A")

        terrain_graph = TerrainGraph.from_graph(original)
        self.assertEqual(terrain_graph.graph['name'], 'test_graph')

    def test_from_graph_rejects_self_loops(self):
        """Test that from_graph raises error if input has self-loops."""
        original = nx.Graph()
        original.add_edge("A", "A")  # self-loop

        with self.assertRaises(ValueError):
            TerrainGraph.from_graph(original)

    def test_terrain_graph_forbids_self_loops(self):
        """Test that TerrainGraph.add_edge rejects self-loops."""
        graph = TerrainGraph()
        graph.add_node("A")

        with self.assertRaises(ValueError):
            graph.add_edge("A", "A")


class TestTerrainGraphElementIteration(unittest.TestCase):
    """Test Element querying and filtering on TerrainGraph."""

    def setUp(self):
        """Create a populated TerrainGraph."""
        self.graph = TerrainGraph()
        self.graph.add_node("A", terrain="forest")
        self.graph.add_node("B", terrain=None)
        self.graph.add_node("C", terrain="water")
        self.graph.add_edge("A", "B", terrain=None)
        self.graph.add_edge("B", "C", terrain="bridge")

    def test_get_all_elements(self):
        """Test get_all_elements returns all nodes and edges."""
        elements = list(self.graph.get_all_elements())

        # 3 nodes + 2 edges = 5 elements
        self.assertEqual(len(elements), 5)

        # Check that nodes come first
        node_elements = [e for e in elements if e.is_node]
        edge_elements = [e for e in elements if not e.is_node]
        self.assertEqual(len(node_elements), 3)
        self.assertEqual(len(edge_elements), 2)

    def test_get_unset_elements(self):
        """Test get_unset_elements filters to terrain==None."""
        unset = list(self.graph.get_unset_elements())

        # B node and A-B edge have terrain=None
        self.assertEqual(len(unset), 2)

        element_ids = [e.element_id for e in unset]
        self.assertIn("B", element_ids)
        self.assertIn(("A", "B"), element_ids)

    def test_count_elements(self):
        """Test count_elements with custom predicate."""
        # Count elements with any terrain set
        count = self.graph.count_elements(
            lambda e: e.get('terrain', None) is not None
        )
        self.assertEqual(count, 3)  # A node, C node, B-C edge

    def test_filter_elements_by_type(self):
        """Test filtering by is_node parameter."""
        nodes = list(self.graph.filter_elements(lambda e: True, is_node=True))
        edges = list(self.graph.filter_elements(lambda e: True, is_node=False))

        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(edges), 2)

    def test_iter_elements_with_data(self):
        """Test iter_elements with data payload."""
        # Iterate with full attributes
        items = list(self.graph.iter_elements(data=True))

        self.assertEqual(len(items), 5)
        for item in items:
            self.assertIsInstance(item, tuple)
            element, attrs = item
            self.assertIsInstance(element, Element)
            self.assertIsInstance(attrs, dict)

    def test_iter_elements_with_data_single_key(self):
        """Test iter_elements extracting single attribute."""
        # Get just the 'terrain' value for each element
        items = list(self.graph.iter_elements(data='terrain'))

        self.assertEqual(len(items), 5)
        terrains = [t for _, t in items]
        self.assertIn('forest', terrains)
        self.assertIn(None, terrains)


class TestTerrainGraphGlobalMetrics(unittest.TestCase):
    """Test global_metrics initialization and storage."""

    def test_terrain_graph_has_empty_metrics(self):
        """Test that new TerrainGraph has empty global_metrics."""
        graph = TerrainGraph()
        self.assertIsInstance(graph.global_metrics, dict)
        self.assertEqual(len(graph.global_metrics), 0)


class TestWaveFunctionCollapseInitialization(unittest.TestCase):
    """Test WaveFunctionCollapse initialization and setup."""

    def setUp(self):
        """Create test settings."""
        self.settings = {
            'base_global_target_dist': {
                'province_terrains': {'forest': 0.5, 'water': 0.5},
                'border_terrains': {'bridge': 0.7, 'mountain_pass': 0.3},
            },
            'base_global_weight': {'global': 1.0},
        }

    def test_wfc_accepts_terrain_graph(self):
        """Test WFC accepts TerrainGraph input."""
        graph = TerrainGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        wfc = WaveFunctionCollapse(self.settings, graph)

        # Should use same graph object
        self.assertIs(wfc.graph, graph)

    def test_wfc_converts_nx_graph(self):
        """Test WFC converts plain nx.Graph to TerrainGraph."""
        original = nx.Graph()
        original.add_node("A")
        original.add_node("B")
        original.add_edge("A", "B")

        wfc = WaveFunctionCollapse(self.settings, original)

        # Should create new TerrainGraph
        self.assertIsInstance(wfc.graph, TerrainGraph)
        self.assertIsNot(wfc.graph, original)

    def test_wfc_initializes_global_metrics(self):
        """Test that WFC initializes graph.global_metrics."""
        graph = TerrainGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        wfc = WaveFunctionCollapse(self.settings, graph)

        # Check metrics were collected
        self.assertIsNotNone(wfc.graph.global_metrics)
        self.assertEqual(wfc.graph.global_metrics['provinces'], 2)
        self.assertEqual(wfc.graph.global_metrics['borders'], 1)

    def test_wfc_global_metrics_alias(self):
        """Test that self.global_metrics is an alias to self.graph.global_metrics."""
        graph = TerrainGraph()
        graph.add_node("A")
        graph.add_edge("A", "A" if False else "B")  # avoid self-loop for test
        graph.add_node("B")
        graph.add_edge("A", "B")

        wfc = WaveFunctionCollapse(self.settings, graph)

        # Both should reference same dict
        self.assertIs(wfc.global_metrics, wfc.graph.global_metrics)

        # Mutations to one affect the other
        wfc.global_metrics['test_key'] = 'test_value'
        self.assertEqual(wfc.graph.global_metrics['test_key'], 'test_value')


class TestSetupElementDists(unittest.TestCase):
    """Test setup_element_dists behavior and warnings."""

    def setUp(self):
        """Create test graph and metrics."""
        self.graph = TerrainGraph()
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")

        self.metrics = {
            'global_adjusting_dist': {
                'province_terrains': {'forest': 0.5, 'water': 0.5},
                'border_terrains': {'bridge': 1.0},
            }
        }

    def test_setup_element_dists_resets_terrain(self):
        """Test that setup_element_dists resets existing terrain by default."""
        self.graph.nodes["A"]['terrain'] = 'existing'

        self.graph.setup_element_dists(self.metrics, reset_existing_terrain=True)

        # Should be reset
        self.assertIsNone(self.graph.nodes["A"].get('terrain'))

    def test_setup_element_dists_preserves_terrain(self):
        """Test that setup_element_dists preserves terrain when flag=False."""
        self.graph.nodes["A"]['terrain'] = 'existing'

        self.graph.setup_element_dists(self.metrics, reset_existing_terrain=False)

        # Should be preserved
        self.assertEqual(self.graph.nodes["A"]['terrain'], 'existing')

    def test_setup_element_dists_creates_pointers(self):
        """Test that setup_element_dists creates dist pointers."""
        self.graph.setup_element_dists(self.metrics)

        # Check that dists pointers were set
        for element in self.graph.get_all_elements():
            self.assertIn('dists', element)
            self.assertIn('global_adjusting_dist', element['dists'])


if __name__ == '__main__':
    unittest.main()
