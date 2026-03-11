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
from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse, collect_global_metrics, update_joint_probability_distribution
from FlowAtlas.populate_graph.rule_management import DistRule, RuleManager


def build_domain_from_rules(rule_managers: list[RuleManager]) -> dict:
    """
    Extract terrain domain from DistRule distributions.

    Collects all terrain names from adjusting_province_dist and adjusting_border_dist across all rules,
    returning a valid base_terrain_domain dict.
    """
    province_terrains = set()
    border_terrains = set()

    for manager in rule_managers:
        for rule in manager.rules:
            if isinstance(rule, DistRule):
                province_terrains.update(rule.adjusting_province_dist.keys())
                border_terrains.update(rule.adjusting_border_dist.keys())

    return {
        'province_terrains': sorted(province_terrains),
        'border_terrains': sorted(border_terrains),
    }


class TestElement(unittest.TestCase):
    """Test Element wrapper for node/edge attributes."""

    def setUp(self):
        """Create a simple TerrainGraph for testing."""
        settings = {}
        self.graph = TerrainGraph(settings)
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

        settings = {}
        terrain_graph = TerrainGraph.from_graph(original, settings)

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

        settings = {}
        terrain_graph = TerrainGraph.from_graph(original, settings)
        self.assertEqual(terrain_graph.graph['name'], 'test_graph')

    def test_from_graph_rejects_self_loops(self):
        """Test that from_graph raises error if input has self-loops."""
        original = nx.Graph()
        original.add_edge("A", "A")  # self-loop

        with self.assertRaises(ValueError):
            TerrainGraph.from_graph(original, settings={})

    def test_terrain_graph_forbids_self_loops(self):
        """Test that TerrainGraph.add_edge rejects self-loops."""
        graph = TerrainGraph(settings={})
        graph.add_node("A")

        with self.assertRaises(ValueError):
            graph.add_edge("A", "A")


class TestTerrainGraphElementIteration(unittest.TestCase):
    """Test Element querying and filtering on TerrainGraph."""

    def setUp(self):
        """Create a populated TerrainGraph."""
        settings = {}
        self.graph = TerrainGraph(settings=settings)
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
        graph = TerrainGraph(settings={})
        self.assertIsInstance(graph.global_metrics, dict)
        self.assertEqual(len(graph.global_metrics), 0)


class TestWaveFunctionCollapseInitialization(unittest.TestCase):
    """Test WaveFunctionCollapse initialization and setup."""

    def setUp(self):
        """Create test settings."""
        self.settings = {
            'base_terrain_domain': {
                'province_terrains': ['forest', 'water'],
                'border_terrains': ['bridge', 'mountain_pass'],
            }
        }

    def test_wfc_accepts_terrain_graph(self):
        """Test WFC accepts TerrainGraph input."""
        graph = TerrainGraph(settings={})
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
        graph = TerrainGraph(settings={})
        graph.add_node("A")
        graph.add_node("B")
        graph.add_edge("A", "B")

        wfc = WaveFunctionCollapse(self.settings, graph)

        # Check metrics were collected
        self.assertIsNotNone(wfc.graph.global_metrics)
        self.assertEqual(wfc.graph.global_metrics['provinces'], 2)
        self.assertEqual(wfc.graph.global_metrics['borders'], 1)
        self.assertIn('terrain_domain', wfc.graph.global_metrics)
        self.assertEqual(wfc.graph.global_metrics['terrain_domain']['province_terrains'], ('forest', 'water'))
        self.assertEqual(wfc.graph.global_metrics['terrain_domain']['border_terrains'], ('bridge', 'mountain_pass'))

class TestWFCInitializationFlow(unittest.TestCase):
    """Test WFC-owned solver initialization and preset replay behavior."""

    def setUp(self):
        """Create test graph and minimal settings for WFC init."""
        self.graph = TerrainGraph(settings={})
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")

        self.settings = {
            'base_terrain_domain': {
                'province_terrains': ['forest', 'water'],
                'border_terrains': ['normal', 'river'],
            }
        }

    def test_wfc_initializes_element_containers(self):
        """Unset elements should have solver containers after WFC initialization."""
        wfc = WaveFunctionCollapse(self.settings, self.graph)

        for element in wfc.graph.get_all_elements():
            self.assertIn('dists', element)
            self.assertIn('dist_weights', element)
            self.assertIn('constraints', element)
            self.assertIn('flags', element)
            self.assertIn('joint_prob_dist', element)

    def test_wfc_replays_preset_assignments(self):
        """Pre-set terrain values should be restored via replay during init."""
        self.graph.nodes["A"]['terrain'] = 'forest'
        self.graph.edges["A", "B"]['terrain'] = 'water'

        wfc = WaveFunctionCollapse(self.settings, self.graph)

        self.assertEqual(wfc.graph.nodes["A"]['terrain'], 'forest')
        self.assertEqual(wfc.graph.edges["A", "B"]['terrain'], 'water')

    def test_wfc_rule_managers_seed_element_distributions(self):
        """DistRules should populate per-element dists and weights during WFC init."""
        global_rule = DistRule(
            adjusting_province_dist={'forest': 0.75, 'water': 0.25},
            adjusting_border_dist={'forest': 0.75, 'water': 0.25},
            adjusting_factor=1.0,
            flag='all',
            name='global_rule',
        )
        global_manager = RuleManager(name='global_manager', rules=[global_rule])

        settings = {
            'base_terrain_domain': build_domain_from_rules([global_manager]),
            'rule_managers': [global_manager],
        }

        wfc = WaveFunctionCollapse(settings, self.graph)

        for element in wfc.graph.get_all_elements():
            self.assertIn('global_rule', element['dists'])
            self.assertEqual(element['dists']['global_rule'], {'forest': 0.75, 'water': 0.25})
            self.assertIn('global_rule', element['dist_weights'])
            self.assertEqual(element['dist_weights']['global_rule'], 1.0)

    def test_wfc_rule_managers_seed_split_dist_by_element_type(self):
        """DistRule split schemas should seed province and border elements with separate terrain domains."""
        global_rule = DistRule(
            adjusting_province_dist={'forest': 0.75, 'water': 0.25},
            adjusting_border_dist={'normal': 0.9, 'river': 0.1},
            adjusting_factor=1.0,
            flag='all',
            name='global_rule',
        )
        global_manager = RuleManager(name='global_manager', rules=[global_rule])

        settings = {
            'base_terrain_domain': build_domain_from_rules([global_manager]),
            'rule_managers': [global_manager],
        }

        wfc = WaveFunctionCollapse(settings, self.graph)

        node_element = Element.from_node("A", wfc.graph)
        edge_element = Element.from_edge(("A", "B"), wfc.graph)

        self.assertEqual(node_element['dists']['global_rule'], {'forest': 0.75, 'water': 0.25})
        self.assertEqual(edge_element['dists']['global_rule'], {'normal': 0.9, 'river': 0.1})

    def test_rule_manager_unsupported_attribute_raises(self):
        """RuleManager should fail fast for unsupported attributes during initialization."""
        global_rule = DistRule(
            adjusting_province_dist={'forest': 1.0},
            adjusting_border_dist={'normal': 1.0},
            adjusting_factor=1.0,
            flag='all',
            name='global_rule',
        )
        bad_manager = RuleManager(name='bad_manager', rules=[global_rule], attribute='region')
        settings = {**self.settings, 'rule_managers': [bad_manager]}

        with self.assertRaises(AssertionError):
            WaveFunctionCollapse(settings, self.graph)


class TestDistRuleBehavior(unittest.TestCase):
    """Behavior-level tests for DistRule adjusting updates and multiplicative composition."""

    def setUp(self):
        self.graph = TerrainGraph(settings={})
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")

    def test_dist_rule_updates_shared_adjusting_dist_on_origin_assignment(self):
        """Setting one origin should recompute shared adjusting_province_dist in place."""
        rule = DistRule(
            adjusting_province_dist={'forest': 0.5, 'water': 0.5},
            adjusting_border_dist={'normal': 1.0},
            adjusting_factor=1.0,
            flag='all',
            name='global_rule',
        )

        # Setup establishes totals used by recompute.
        rule.setup(self.graph)

        origin = Element.from_node("A", self.graph)
        origin['terrain'] = 'forest'
        rule.update_statistics_for_origin(self.graph, origin)

        # For two provinces with target 0.5/0.5 and one forest already assigned:
        # forest gap -> 0, water gap -> +1 -> normalized to forest:0.0, water:1.0
        self.assertAlmostEqual(rule.adjusting_province_dist['forest'], 0.0, places=8)
        self.assertAlmostEqual(rule.adjusting_province_dist['water'], 1.0, places=8)

    def test_dist_rule_setup_raises_for_non_unit_all_flag_weight(self):
        """MVP should fail fast when existing 'all' flag weight is not 1.0."""
        self.graph.nodes["A"]['flags'] = {'all': 0.5}

        rule = DistRule(
            adjusting_province_dist={'forest': 1.0},
            adjusting_border_dist={'normal': 1.0},
            adjusting_factor=1.0,
            flag='all',
            name='global_rule',
        )

        with self.assertRaises(NotImplementedError):
            rule.setup(self.graph)

    def test_joint_probability_uses_multiplicative_factors(self):
        """Joint probabilities should be proportional to product of per-rule factors."""
        graph = TerrainGraph(settings={})
        graph.add_node("A", terrain=None)
        graph.global_metrics = {
            'terrain_domain': {
                'province_terrains': ('forest', 'water'),
                'border_terrains': ('normal',),
            }
        }

        element = Element.from_node("A", graph)
        element['dists'] = {
            'r1': {'forest': 0.8, 'water': 0.2},
            'r2': {'forest': 0.25, 'water': 0.75},
        }
        element['dist_weights'] = {'r1': 1.0, 'r2': 1.0}
        element['constraints'] = {}

        update_joint_probability_distribution(element, graph)

        # Multiplicative factors: forest=0.8*0.25=0.2, water=0.2*0.75=0.15
        # Normalized => forest=0.2/0.35, water=0.15/0.35
        self.assertAlmostEqual(element['joint_prob_dist']['forest'], 0.2 / 0.35, places=8)
        self.assertAlmostEqual(element['joint_prob_dist']['water'], 0.15 / 0.35, places=8)


if __name__ == '__main__':
    unittest.main()
