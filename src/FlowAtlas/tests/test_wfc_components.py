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
from FlowAtlas.populate_graph.rule_management import BanRule, DistRule, RuleManager
from FlowAtlas.populate_graph.wave_function_collapse import make_wfc_settings_from_global_dist
from FlowAtlas.populate_graph.rules_library import make_default_wfc_settings


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


class TestTerrainGraphTopologyNeighborhood(unittest.TestCase):
    """Test topology range queries for connected elements."""

    def setUp(self):
        self.graph = TerrainGraph(settings={})
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_node("C")
        self.graph.add_edge("A", "B")
        self.graph.add_edge("A", "C")
        self.graph.add_edge("B", "C")

    def test_node_range_eq_half_returns_incident_edges(self):
        node_a = Element.from_node("A", self.graph)

        neighbors = self.graph.get_connected_elements_topology(
            node_a,
            range=0.5,
            range_check='eq',
            element_kind='edges',
            include_distance=False,
        )

        edge_ids = {tuple(sorted(edge.element_id)) for edge in neighbors}
        self.assertEqual(edge_ids, {("A", "B"), ("A", "C")})

    def test_node_range_leq_one_returns_incident_edges_and_neighbor_nodes(self):
        node_a = Element.from_node("A", self.graph)

        neighbors = self.graph.get_connected_elements_topology(
            node_a,
            range=1.0,
            range_check='leq',
            element_kind='both',
            include_distance=False,
        )

        node_ids = {elem.element_id for elem in neighbors if elem.is_node}
        edge_ids = {tuple(sorted(elem.element_id)) for elem in neighbors if elem.is_border}
        self.assertEqual(node_ids, {"B", "C"})
        self.assertEqual(edge_ids, {("A", "B"), ("A", "C")})

    def test_include_distance_returns_distance_metadata(self):
        edge_ab = Element.from_edge(("A", "B"), self.graph)

        neighbors = self.graph.get_connected_elements_topology(
            edge_ab,
            range=1.0,
            range_check='leq',
            element_kind='both',
            include_distance=True,
        )

        payload = {
            (elem.is_node, tuple(sorted(elem.element_id)) if elem.is_border else elem.element_id): dist
            for elem, dist in neighbors
        }
        self.assertAlmostEqual(payload[(True, "A")], 0.5, places=8)
        self.assertAlmostEqual(payload[(True, "B")], 0.5, places=8)
        self.assertAlmostEqual(payload[(False, ("A", "C"))], 1.0, places=8)
        self.assertAlmostEqual(payload[(False, ("B", "C"))], 1.0, places=8)


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

        # For two provinces with target 0.5/0.5 and one forest already assigned
        # (adjusting_factor=1.0, NO normalization per updated design):
        # forest: gap=0,  adjustment_part=0,   target_part=0 -> 0.0
        # water:  gap=+1, adjustment_part=+0.5, target_part=0 -> 0.5
        self.assertAlmostEqual(rule.adjusting_province_dist['forest'], 0.0, places=8)
        self.assertAlmostEqual(rule.adjusting_province_dist['water'], 0.5, places=8)

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

    def test_dist_rule_adjusting_dist_stays_unnormalized_after_update(self):
        """Adjusted factors should remain raw factors (not internally normalized)."""
        rule = DistRule(
            adjusting_province_dist={'forest': 0.5, 'water': 0.5},
            adjusting_border_dist={'normal': 1.0},
            adjusting_factor=0.5,
            flag='all',
            name='global_rule',
        )

        rule.setup(self.graph)

        origin = Element.from_node("A", self.graph)
        origin['terrain'] = 'forest'
        rule.update_statistics_for_origin(self.graph, origin)

        # For two provinces and one assigned forest:
        # forest -> 0.25, water -> 0.50 (sum=0.75, intentionally not normalized)
        self.assertAlmostEqual(rule.adjusting_province_dist['forest'], 0.25, places=8)
        self.assertAlmostEqual(rule.adjusting_province_dist['water'], 0.5, places=8)
        self.assertAlmostEqual(sum(rule.adjusting_province_dist.values()), 0.75, places=8)

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

    def test_joint_probability_applies_dist_weights_as_linear_influence(self):
        """Dist weights should linearly damp/boost each rule contribution before multiplication."""
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
            'r1': {'forest': 0.2, 'water': 0.8},
            'r2': {'forest': 0.9, 'water': 0.1},
        }
        element['dist_weights'] = {'r1': 0.0, 'r2': 0.5}
        element['constraints'] = {}

        update_joint_probability_distribution(element, graph)

        # r1 weight 0.0 -> neutral (factor 1.0 for both).
        # r2 weight 0.5 -> forest factor=0.95, water factor=0.55.
        # Normalized: forest=0.95/(0.95+0.55), water=0.55/(0.95+0.55).
        self.assertAlmostEqual(element['joint_prob_dist']['forest'], 0.95 / 1.5, places=8)
        self.assertAlmostEqual(element['joint_prob_dist']['water'], 0.55 / 1.5, places=8)


class TestBanRuleBehavior(unittest.TestCase):
    """Behavior-level tests for BanRule neighborhood evaluation and symmetric bans."""

    def setUp(self):
        self.graph = TerrainGraph(settings={})
        self.graph.add_node("A", terrain=None)
        self.graph.add_node("B", terrain=None)
        self.graph.add_node("C", terrain=None)
        self.graph.add_edge("A", "B", terrain=None)
        self.graph.add_edge("B", "C", terrain=None)

        self.graph.global_metrics = {
            'terrain_domain': {
                'province_terrains': ('forest', 'sea', 'waste'),
                'border_terrains': ('normal', 'river'),
            }
        }

    def test_ban_rule_bans_opposite_set_for_neighboring_match(self):
        rule = BanRule(
            set1={'sea'},
            set2={'forest'},
            range=1.0,
            range_check='leq',
            evaluation=lambda neighbors: len(neighbors) > 0,
            name='sea_forest_ban',
        )

        for element in self.graph.get_all_elements():
            rule.initialize_element(element, self.graph, manager_weight=1.0)

        self.graph.nodes["B"]['terrain'] = 'sea'
        affected = Element.from_node("A", self.graph)
        origin = Element.from_node("B", self.graph)

        rule.update_affected(affected, self.graph, origin)

        self.assertEqual(affected['constraints']['sea_forest_ban'], {'forest'})

    def test_ban_rule_is_symmetric_set1_set2(self):
        rule = BanRule(
            set1={'sea'},
            set2={'forest'},
            range=1.0,
            range_check='leq',
            evaluation=lambda neighbors: len(neighbors) > 0,
            name='sea_forest_ban',
        )

        for element in self.graph.get_all_elements():
            rule.initialize_element(element, self.graph, manager_weight=1.0)

        affected = Element.from_node("A", self.graph)

        self.graph.nodes["B"]['terrain'] = 'sea'
        rule.update_affected(affected, self.graph, Element.from_node("B", self.graph))
        self.assertEqual(affected['constraints']['sea_forest_ban'], {'forest'})

        self.graph.nodes["B"]['terrain'] = 'forest'
        origin = Element.from_node("B", self.graph)
        rule.update_affected(affected, self.graph, origin)
        self.assertEqual(affected['constraints']['sea_forest_ban'], {'sea'})

    def test_ban_rule_does_not_apply_outside_range(self):
        rule = BanRule(
            set1={'sea'},
            set2={'forest'},
            range=1.0,
            range_check='leq',
            evaluation=lambda neighbors: len(neighbors) > 0,
            name='sea_forest_ban',
        )

        for element in self.graph.get_all_elements():
            rule.initialize_element(element, self.graph, manager_weight=1.0)

        self.graph.nodes["C"]['terrain'] = 'sea'
        affected = Element.from_node("A", self.graph)
        origin = Element.from_node("C", self.graph)

        rule.update_affected(affected, self.graph, origin)

        self.assertEqual(affected['constraints']['sea_forest_ban'], set())

    def test_ban_rule_distance_aware_evaluation_payload(self):
        rule = BanRule(
            set1={'river'},
            set2={'forest'},
            range=0.5,
            range_check='eq',
            include_distance=True,
            evaluation=lambda neighbors: any(distance <= 0.5 for _, distance in neighbors),
            name='river_forest_ban',
        )

        for element in self.graph.get_all_elements():
            rule.initialize_element(element, self.graph, manager_weight=1.0)

        self.graph.edges["A", "B"]['terrain'] = 'river'
        affected = Element.from_node("A", self.graph)
        origin = Element.from_edge(("A", "B"), self.graph)

        rule.update_affected(affected, self.graph, origin)

        self.assertEqual(affected['constraints']['river_forest_ban'], {'forest'})


class TestWaveFunctionCollapseIntegration(unittest.TestCase):
    """Integration tests for full WFC loop with rules and constraints."""

    def test_banrule_respected_in_full_collapse(self):
        """
        Integration test: BanRule constraints are respected throughout full WFC collapse.

        Setup: Small 4-node graph with sea/forest BanRule.
        Expected: No forest province ever has a 'river' border (banned) adjacent,
        regardless of which node was set first or the entropy-driven solver order.
        """
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C", "D"])
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")])

        # Setup: BanRule forbids 'river' borders next to 'sea' provinces.
        ban_rule = BanRule(
            set1={'sea'},
            set2={'river'},
            range=1.0,
            range_check='leq',
            evaluation=lambda neighbors: len(neighbors) > 0,
            name='sea_river_ban',
        )
        ban_manager = RuleManager(name='sea_river', rules=[ban_rule])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'sea': 0.5, 'forest': 0.5},
                'border_terrains': {'river': 0.5, 'normal': 0.5},
            },
            'rule_managers': [ban_manager],
        }

        # Process settings through the standard WFC setup function
        wfc_settings = make_wfc_settings_from_global_dist(settings)

        # Pre-seed one sea province so BanRule constraints propagate to unset borders
        # before border terrains are selected by the collapse loop.
        graph.nodes["A"]['terrain'] = 'sea'

        wfc = WaveFunctionCollapse(wfc_settings, graph)
        result = wfc.wave_function_collapse()

        # Verify: No 'sea' province has 'river' border adjacent
        for edge_id in result.edges():
            u, v = edge_id
            u_terrain = result.nodes[u]['terrain']
            v_terrain = result.nodes[v]['terrain']
            edge_terrain = result.edges[edge_id]['terrain']

            if u_terrain == 'sea' or v_terrain == 'sea':
                self.assertNotEqual(
                    edge_terrain, 'river',
                    f"Sea province adjacent to river border: {edge_id} "
                    f"({u}={u_terrain}, {v}={v_terrain}, border={edge_terrain})"
                )

    def test_wfc_all_elements_set_on_completion(self):
        """
        Integration test: WFC.wave_function_collapse() terminates with all elements set.
        """
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        result = wfc.wave_function_collapse()

        # Verify all nodes and edges have terrain set
        for node in result.nodes():
            self.assertIsNotNone(result.nodes[node]['terrain'], f"Node {node} terrain not set")

        for edge in result.edges():
            self.assertIsNotNone(result.edges[edge]['terrain'], f"Edge {edge} terrain not set")

        # Verify is_all_set returns True
        self.assertTrue(result.is_all_set(), "Graph should report all elements set after collapse")

    def test_wfc_debug_statistics_tracks_progress_timing_and_final_distribution(self):
        """Debug mode should expose per-step progress/timing and post-run distribution metrics."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()

        self.assertTrue(debug_stats['enabled'])
        self.assertEqual(debug_stats['steps'], 5)
        self.assertEqual(len(debug_stats['progress']), 5)
        self.assertGreaterEqual(debug_stats['timing']['total_seconds'], 0.0)
        self.assertGreaterEqual(debug_stats['timing']['province_assignment_seconds'], 0.0)
        self.assertGreaterEqual(debug_stats['timing']['border_assignment_seconds'], 0.0)
        self.assertGreaterEqual(debug_stats['timing']['mean_step_seconds'], 0.0)

        final_report = debug_stats['final_report']
        self.assertIn('observed', final_report)
        self.assertIn('rule_targets', final_report)
        self.assertGreaterEqual(len(final_report['rule_targets']), 1)

        first_rule_report = final_report['rule_targets'][0]
        self.assertIn('province_metrics', first_rule_report)
        self.assertIn('border_metrics', first_rule_report)
        self.assertIn('l1_distance', first_rule_report['province_metrics'])
        self.assertIn('l1_distance', first_rule_report['border_metrics'])
        self.assertIn('per_terrain', first_rule_report['province_metrics'])
        self.assertIn('per_terrain', first_rule_report['border_metrics'])
        self.assertIn('equal_weight_match', first_rule_report['province_metrics'])
        self.assertIn('target_weighted_match', first_rule_report['province_metrics'])
        self.assertIn('tv_distance', first_rule_report['province_metrics'])

        self.assertIn('progress_windows', debug_stats)
        self.assertGreaterEqual(len(debug_stats['progress_windows']), 1)

    def test_wfc_debug_report_formatters_return_text(self):
        """Debug formatter methods should provide printable progress/final report text."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        progress_text = wfc.format_debug_progress_report()
        final_text = wfc.format_debug_final_report()

        self.assertIn('WFC Progress Report', progress_text)
        self.assertIn('snapshots=', progress_text)
        self.assertIn('iteration_snapshots=', progress_text)
        self.assertIn('WFC Final Distribution Report', final_text)
        self.assertIn('rule=', final_text)

    def test_wfc_debug_entropy_history_and_selected_metadata(self):
        """Debug stats should include entropy time-series and selected element metadata."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.6, 'forest': 0.4},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_track_entropy_metrics': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()
        self.assertEqual(len(debug_stats['entropy_history']), debug_stats['steps'])
        self.assertGreater(debug_stats['latest_entropy']['all']['count'], 0)

        first_progress_entry = debug_stats['progress'][0]
        self.assertIn('selected_element_id', first_progress_entry)
        self.assertIn('selected_entropy', first_progress_entry)
        self.assertIn('entropy_summary', first_progress_entry)

    def test_wfc_debug_option_distribution_storage(self):
        """Step snapshots should optionally include selected option distributions."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.7, 'forest': 0.3},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_store_option_distributions': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        first_progress_entry = wfc.get_debug_statistics()['progress'][0]
        self.assertIn('selected_option_distribution', first_progress_entry)
        self.assertGreaterEqual(len(first_progress_entry['selected_option_distribution']), 1)

    def test_wfc_debug_iteration_snapshots_capture_checkpoints(self):
        """Collector should store periodic checkpoint snapshots during the run."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C", "D"])
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': False,
            'debug_wfc_store_iteration_snapshots': True,
            'debug_wfc_snapshot_every': 2,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()
        checkpoints = debug_stats['iteration_snapshots']
        self.assertGreaterEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[0]['step'], 1)
        self.assertEqual(checkpoints[-1]['step'], debug_stats['steps'])
        self.assertIn('province_terrain_counts', checkpoints[-1])
        self.assertIn('border_terrain_counts', checkpoints[-1])

    def test_wfc_debug_can_store_full_entropy_maps_for_all_unset_elements(self):
        """Optional entropy surface logging should persist per-step entropy for all unset elements."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_track_entropy_metrics': True,
            'debug_wfc_store_full_entropy_maps': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()
        entropy_surfaces = debug_stats['entropy_surfaces']

        self.assertEqual(len(entropy_surfaces), debug_stats['steps'])
        self.assertIn('all', entropy_surfaces[0])
        self.assertIn('province', entropy_surfaces[0])
        self.assertIn('border', entropy_surfaces[0])
        self.assertGreater(len(entropy_surfaces[0]['all']), 0)

        progress_text = wfc.format_debug_progress_report()
        self.assertIn('entropy_surfaces=', progress_text)

    def test_wfc_debug_can_compare_iteration_snapshots(self):
        """Debug API should compare two iteration snapshots and return count deltas."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C", "D"])
        graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_iteration_snapshots': True,
            'debug_wfc_snapshot_every': 2,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        checkpoints = wfc.get_debug_statistics()['iteration_snapshots']
        first_step = checkpoints[0]['step']
        last_step = checkpoints[-1]['step']
        comparison = wfc.compare_debug_iteration_snapshots(first_step, last_step)

        self.assertEqual(comparison['from_step'], first_step)
        self.assertEqual(comparison['to_step'], last_step)
        self.assertGreaterEqual(comparison['completion_ratio_delta'], 0.0)
        self.assertIn('province_terrain_count_delta', comparison)
        self.assertIn('border_terrain_count_delta', comparison)

    def test_wfc_debug_can_store_checkpoint_states_for_partial_maps(self):
        """Checkpoint states should include set terrains plus entropy values for unset provinces and borders."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_checkpoint_states': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        checkpoint_states = wfc.get_debug_checkpoint_states()
        debug_stats = wfc.get_debug_statistics()
        self.assertEqual(len(checkpoint_states), debug_stats['steps'])
        self.assertIn('province_terrain', checkpoint_states[0])
        self.assertIn('province_entropy', checkpoint_states[0])
        self.assertIn('border_terrain', checkpoint_states[0])
        self.assertIn('border_entropy', checkpoint_states[0])

        progress_text = wfc.format_debug_progress_report()
        self.assertIn('checkpoint_state_samples=', progress_text)

    def test_wfc_debug_can_track_rule_firings_and_weight_changes(self):
        """Optional diagnostics should track rule firings and adjusting-dist changes by step."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': True,
            'debug_wfc_store_step_snapshots': True,
            'debug_wfc_track_rule_firings': True,
            'debug_wfc_track_weight_changes': True,
            'debug_wfc_print_progress': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()
        self.assertEqual(len(debug_stats['rule_firing_history']), debug_stats['steps'])
        self.assertEqual(len(debug_stats['weight_change_history']), debug_stats['steps'])
        self.assertGreaterEqual(len(debug_stats['rule_firing_counts']), 1)

        progress_text = wfc.format_debug_progress_report()
        self.assertIn('rule_firing_samples=', progress_text)
        self.assertIn('weight_change_samples=', progress_text)

    def test_wfc_debug_disabled_skips_debug_bookkeeping(self):
        """When debug is off, collapse should not collect step snapshots or final reports."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
            'debug_wfc_statistics': False,
        }

        wfc_settings = make_wfc_settings_from_global_dist(settings)
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        wfc.wave_function_collapse()

        debug_stats = wfc.get_debug_statistics()
        self.assertFalse(debug_stats['enabled'])
        self.assertEqual(debug_stats['steps'], 0)
        self.assertEqual(debug_stats['progress'], [])
        self.assertEqual(debug_stats['final_report'], {})

    def test_wfc_debug_level_presets_apply_defaults(self):
        """Verify debug_wfc_level presets apply correct flag defaults."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B", "C"])
        graph.add_edges_from([("A", "B"), ("B", "C")])

        base_settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 0.5, 'forest': 0.5},
                'border_terrains': {'normal': 1.0},
            },
        }

        # Test 0 (off) preset: all flags False
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 0
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        self.assertFalse(wfc.debug_enabled)
        self.assertFalse(wfc.debug_print_progress)
        self.assertFalse(wfc.debug_print_progress_report)
        self.assertFalse(wfc.debug_print_final_report)
        self.assertFalse(wfc.debug_wfc_timing)
        self.assertFalse(wfc.debug_store_step_snapshots)
        self.assertFalse(wfc.debug_store_iteration_snapshots)
        self.assertFalse(wfc.debug_track_entropy_metrics)
        self.assertFalse(wfc.debug_store_option_distributions)
        self.assertFalse(wfc.debug_store_full_entropy_maps)
        self.assertFalse(wfc.debug_track_rule_firings)
        self.assertFalse(wfc.debug_track_weight_changes)
        self.assertFalse(wfc.debug_store_checkpoint_states)

        # Test 1 (basic) preset: lightweight runtime debug, no heavier diagnostics
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 1
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        self.assertTrue(wfc.debug_enabled)
        self.assertTrue(wfc.debug_print_progress)
        self.assertFalse(wfc.debug_print_progress_report)
        self.assertFalse(wfc.debug_print_final_report)
        self.assertFalse(wfc.debug_wfc_timing)
        self.assertTrue(wfc.debug_store_step_snapshots)
        self.assertFalse(wfc.debug_store_iteration_snapshots)
        self.assertFalse(wfc.debug_track_entropy_metrics)
        self.assertFalse(wfc.debug_store_option_distributions)
        self.assertFalse(wfc.debug_store_full_entropy_maps)
        self.assertFalse(wfc.debug_track_rule_firings)
        self.assertFalse(wfc.debug_track_weight_changes)
        self.assertFalse(wfc.debug_store_checkpoint_states)

        # Test 2 (verbose) preset: rich summaries, timing, and entropy snapshots
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 2
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        self.assertTrue(wfc.debug_enabled)
        self.assertTrue(wfc.debug_print_progress)
        self.assertTrue(wfc.debug_print_progress_report)
        self.assertFalse(wfc.debug_print_final_report)
        self.assertTrue(wfc.debug_wfc_timing)
        self.assertTrue(wfc.debug_store_step_snapshots)
        self.assertTrue(wfc.debug_store_iteration_snapshots)
        self.assertTrue(wfc.debug_track_entropy_metrics)
        self.assertFalse(wfc.debug_store_option_distributions)
        self.assertFalse(wfc.debug_store_full_entropy_maps)
        self.assertFalse(wfc.debug_track_rule_firings)
        self.assertFalse(wfc.debug_track_weight_changes)
        self.assertFalse(wfc.debug_store_checkpoint_states)

        # Test 3 (max diagnostics) preset: highest level should turn everything on
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 3
        wfc = WaveFunctionCollapse(wfc_settings, graph)
        self.assertTrue(wfc.debug_enabled)
        self.assertTrue(wfc.debug_print_progress)
        self.assertTrue(wfc.debug_print_progress_report)
        self.assertTrue(wfc.debug_print_final_report)
        self.assertTrue(wfc.debug_wfc_timing)
        self.assertTrue(wfc.debug_store_step_snapshots)
        self.assertTrue(wfc.debug_store_iteration_snapshots)
        self.assertTrue(wfc.debug_track_entropy_metrics)
        self.assertTrue(wfc.debug_store_option_distributions)
        self.assertTrue(wfc.debug_store_full_entropy_maps)
        self.assertTrue(wfc.debug_track_rule_firings)
        self.assertTrue(wfc.debug_track_weight_changes)
        self.assertTrue(wfc.debug_store_checkpoint_states)

    def test_wfc_debug_level_explicit_flags_override_presets(self):
        """Verify explicit flags override preset defaults."""
        graph = TerrainGraph(settings={})
        graph.add_nodes_from(["A", "B"])
        graph.add_edge("A", "B")

        base_settings = {
            'base_global_target_dist': {
                'province_terrains': {'plains': 1.0},
                'border_terrains': {'normal': 1.0},
            },
        }

        # Set preset to 0 (off) but override one flag to True
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 0
        wfc_settings['debug_wfc_timing'] = True  # Explicit override
        wfc = WaveFunctionCollapse(wfc_settings, graph)

        self.assertFalse(wfc.debug_enabled)  # Still off (from preset)
        self.assertFalse(wfc.debug_print_progress)  # Still off (from preset)
        self.assertTrue(wfc.debug_wfc_timing)  # Overridden to True

        # Set preset to 3 (max diagnostics) but override one flag to False
        wfc_settings = make_wfc_settings_from_global_dist(base_settings.copy())
        wfc_settings['debug_wfc_level'] = 3
        wfc_settings['debug_wfc_timing'] = False  # Explicit override
        wfc = WaveFunctionCollapse(wfc_settings, graph)

        self.assertTrue(wfc.debug_enabled)  # Still on (from preset)
        self.assertTrue(wfc.debug_print_progress)  # Still on (from preset)
        self.assertFalse(wfc.debug_wfc_timing)  # Overridden to False


class TestRulesLibrary(unittest.TestCase):
    """Test default WFC rule bundle wiring."""

    def test_default_wfc_settings_include_global_and_sea_border_rules(self):
        settings = make_default_wfc_settings()

        manager_names = [manager.name for manager in settings['rule_managers']]
        self.assertIn('global_dist', manager_names)
        self.assertIn('sea_borders', manager_names)

        self.assertIn('base_terrain_domain', settings)
        self.assertIn('province_terrains', settings['base_terrain_domain'])
        self.assertIn('border_terrains', settings['base_terrain_domain'])


if __name__ == '__main__':
    unittest.main()
