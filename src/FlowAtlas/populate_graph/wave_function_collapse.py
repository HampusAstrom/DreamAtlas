import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, Delaunay

from .terrain_graph import Element, TerrainGraph
from .rule_management import RuleManager, DistRule

"""
⚠️  WARNING: Wave Function Collapse Implementation INCOMPLETE ⚠️

This module provides WFC structure and main loop, but CRITICAL probability
update and constraint logic is STUBBED ONLY. Maps generated will be non-sensical.

KNOWN ISSUES:
- update_statistics_and_probabilities() does NOT update neighbor probabilities
- No constraint propagation or contradiction detection
- No terrain realism enforcement (terrains can be geographically invalid)
- No national distribution preferences implemented
- No dynamic adjusting_dist steering

For full checklist of missing features and implementation plan, see:
  CHECKPOINT_WFC_INCOMPLETE.md (DELETE THIS FILE once checkpoint is cleared)

DO NOT use output maps for production content until Tier 1 items completed.
See CHECKPOINT_WFC_INCOMPLETE.md for detailed issue breakdown and roadmap.
"""

""" TODO structure
takes a graph of connected nodes, each node representing a province and
each edge representing a neighbour connection to an adjacent province

sets the terrain type and border terrain type for each province(prov)/border(bor)

Process sequence structure:
* marking required parts for minimal prototype
non-marked are planned expansions

1. *Collect any global metrics for graph
    1.1 user settings
    1.2 *# provinces, # borders
    1.3 nations and their preferences
    1.4 other?
2. *determine target base terrain distributions
    implied probabilities in the form of relative fractions
    1.0 means no relative change, do any preprocessing of dists not added in this format (might require flag)
    2.1 *global target_dist (implied prob) from:
        2.1.1 *base default
        2.1.2 settings
        2.1.3 nation preferences
    2.2 national "nearby" target_dist (implied prob) preferences from:
        2.2.1 nation preferences
    2.3 setup global counter for weighing enforcement need for global current_dist
    2.4 setup local counters for weighing enforcement need for national/local current_dist
        explanation: as prov/bor in a set of (weighted) provinces/borders are set
        their current_dist might vary from their target_dist, the larger the (relative) dist
        difference is, and the fewer the province left to correct it, the higher
        that adjusting_dist's weight get's to contribute to the selection probability
        the adjusting_dist should be formed to most likely move current_dist towards target_dist
        adjusting_dist starts as target_dist if no provinces/borders are set, and is updated as they get set
    2.5 *setup inital global adjusting_dist
    2.6 setup inital local adjusting_dist
3. *preprocess incoming graph, checking for any predefined values/flags beyond graph structure
    *(if not rest implemented, just clean of any values)
    3.1 if any found, calculate any initial macro statistics (compated to target_distt)
    3.2 determine conditional probabilities for all unset terrains in range of set nodes/borders
4. *start wave function collapse procedure
    4.1 *select node to set by some combination of:
        4.1.1 *lowest entropy in node probs (or random fallback, but will have issues if we have constraints)
        4.1.2 fewest provinces assigned in nation "region"
        4.1.3 other?
    4.2 *select and set terrain using probs in node (see prop calc later)
    4.3 update global and local dist statistics
    4.4 *update conditional probabilities for all unset terrains in range of set node/border
        (this could lead to indirect limits beyond range, consider how to handle if/when need arises)
    4.5 *if any contradictions found (0 total probability for any unset node), handle by some combination of:
        *(one solution beyond warn needed)
        4.5.1 *warn/log problem
        4.5.2 raise error
        4.5.3 backtracking to previous state and trying different option (with bailout after some tries)
        4.5.4 if no options left, backtrack further, or restart with new seed (with bailout after some tries)
        4.5.5 override and set by some base probability
    4.6 *repeat until all nodes are set



The intention is that most components of this process should be modular
so we can make a basic version and then add to it and replace and expand option
for how to do the generation over time, with option to toggle parts in settings.
"""

# OBS: in the graph, nodes correspond to provinces,
# and edges correspond to borders between provinces

# placeholder functions for the various steps in the process, to be implemented later
def collect_global_metrics(graph: TerrainGraph, settings: dict):
    # collect any global metrics needed for the process, such as number of provinces, nations, etc.

    provinces = graph.number_of_nodes()
    borders = graph.number_of_edges()
    ret = {
        "provinces": provinces,
        "borders": borders,
        # "settings": settings, # do we need to store settings here?
        "set_provinces": 0,
        "set_borders": 0,
    }
    return ret

def determine_target_distributions(graph: TerrainGraph, settings: dict):
    # Option B: global metrics hold only valid terrain domains.
    # Probability distributions are owned by rules.
    assert 'base_terrain_domain' in settings, (
        "settings must include 'base_terrain_domain' with keys "
        "'province_terrains' and 'border_terrains'"
    )

    domain = settings['base_terrain_domain']
    assert isinstance(domain, dict), "base_terrain_domain must be a dict"
    assert 'province_terrains' in domain, "base_terrain_domain must include 'province_terrains'"
    assert 'border_terrains' in domain, "base_terrain_domain must include 'border_terrains'"

    province_terrains = domain['province_terrains']
    border_terrains = domain['border_terrains']
    assert isinstance(province_terrains, (list, tuple)), "province_terrains must be a list/tuple"
    assert isinstance(border_terrains, (list, tuple)), "border_terrains must be a list/tuple"
    assert len(province_terrains) > 0, "province_terrains must not be empty"
    assert len(border_terrains) > 0, "border_terrains must not be empty"

    graph.global_metrics['terrain_domain'] = {
        'province_terrains': tuple(province_terrains),
        'border_terrains': tuple(border_terrains),
    }

def preprocess_graph(graph: TerrainGraph, settings: dict):
    # preprocess hook kept for settings override compatibility.
    # solver state initialization now lives in WaveFunctionCollapse.
    return


def get_element_terrain_domain(element: Element, graph: TerrainGraph) -> tuple:
    domain = graph.global_metrics['terrain_domain']
    if element.is_province:
        return domain['province_terrains']
    return domain['border_terrains']

def update_joint_probability_distribution(element: Element, graph: TerrainGraph):
    # calculate the joint probability distribution for the given element
    # (province or border) using multiplicative relative-factor composition
    # across all rule contributions and constraints.

    # Multiplicative relative-factor model:
    # Each rule provides a blended distribution (result of dist_weight * adjusting_dist + (1-dist_weight)*uniform)
    # Join these by multiplying the relative factors for each terrain across all rules
    # This way, terrains favored by multiple rules get exponentially higher weight

    weights = element['dist_weights']
    terrain_domain = get_element_terrain_domain(element, graph)

    # Start with relative factor 1.0 for each terrain (neutral)
    joint_factors = {terrain: 1.0 for terrain in terrain_domain}

    # Multiply in contributions from each rule
    for name, dist in element['dists'].items():  # note, a pointer to the rule's shared dist object
        if len(dist) > len(joint_factors):
            extra_terrains = set(dist.keys()) - set(joint_factors.keys())
            raise ValueError(f"Local dist {name} has extra terrains not found in terrain domain:\n{extra_terrains}")
        assert name in weights, f"Dist {name} for element {element} does not have a corresponding weight in dist_weights"

        # dist is already the blended result from update_affected (or from initialize_element)
        # Treat it as a relative-factor contribution: normalize it and multiply into joint_factors
        for terrain, factor in dist.items():
            assert terrain in joint_factors, f"Terrain {terrain} from local dist {name} not found in terrain domain"
            joint_factors[terrain] *= factor

    # apply constraints (zero out banned terrains)
    for name, constraint in element['constraints'].items():
        for terrain, ban in constraint.items():
            if ban:
                joint_factors[terrain] = 0.0

    # normalize the joint probability distribution to sum to 1
    sum_factors = sum(joint_factors.values())
    if sum_factors > 0:
        joint_prob_dist = {terrain: f / sum_factors for terrain, f in joint_factors.items()}
    else:
        print(f"warning: joint probability distribution for element {element} has sum of zero or less, using uniform distribution")
        num_terrains = len(joint_factors)
        joint_prob_dist = {terrain: 1.0 / num_terrains for terrain in joint_factors}

    element['joint_prob_dist'] = joint_prob_dist

def shannon_entropy(prob_dist: dict):
    # calculate the Shannon entropy of the given probability distribution
    # we assume prob_dist is a dict of terrain: probability, and that probabilities are normalized to sum to 1
    entropy = 0.0
    prob_sum = 0
    for terrain, prob in prob_dist.items():
        prob_sum += prob
        if prob > 0:
            entropy -= prob * np.log(prob)
    # if probs seem to not be normalized, we loop a second time, normalizing as we go
    if prob_sum > 1.001 or prob_sum < 0.999:
        entropy = 0.0
        for terrain, prob in prob_dist.items():
            if prob > 0:
                prob /= prob_sum
                entropy -= prob * np.log(prob)
    return entropy

def select_element_to_set(graph: TerrainGraph) -> Element:
    # select the next element (province or border) to set based on entropy and other factors

    # check entropy of unset elements and select the one with lowest entropy,
    # as determined by its joint probability distribution,
    # with a random fallback (that should warn when triggered)
    element_to_set = []
    lowest_entropy = float('inf')

    for element in graph.get_unset_elements():
        entropy = shannon_entropy(element['joint_prob_dist'])

        if entropy < lowest_entropy:
            lowest_entropy = entropy
            element_to_set = [element]
        elif entropy == lowest_entropy:
            element_to_set.append(element)

    if not element_to_set:
        raise RuntimeError("No unset elements found, but should still be setting elements. Check logic.")

    # np.random.choice needs indices/integers, so pick by index
    chosen_index = np.random.choice(len(element_to_set))
    return element_to_set[chosen_index]

def select_element_terrain(element: Element, graph: TerrainGraph):
    # select a terrain for the given element (province or border) based on probabilities and constraints

    return np.random.choice(list(element['joint_prob_dist'].keys()), p=list(element['joint_prob_dist'].values()))


class WaveFunctionCollapse:
    """
    Wave Function Collapse algorithm for terrain generation on a TerrainGraph.

    Behavior:
    - If input is already a TerrainGraph, it is used directly (in-place modification)
    - If input is a plain nx.Graph, a new TerrainGraph is created via from_graph()
      (original graph is NOT modified)

    Always return self.graph from wave_function_collapse() — caller should use the result.
    """
    def __init__(self, settings: dict, graph: nx.Graph):
        self.settings = settings
        if isinstance(graph, TerrainGraph):
            print("Using provided TerrainGraph directly (in-place modification)")
        elif isinstance(graph, nx.Graph):
            print("Creating new TerrainGraph from provided nx.Graph (original graph will NOT be modified)")

        self.graph = graph if isinstance(graph, TerrainGraph) else TerrainGraph.from_graph(graph, settings)

        # 0. setup any function overrides from settings
        self.collect_global_metrics = self.set_func(collect_global_metrics)
        self.determine_target_distributions = self.set_func(determine_target_distributions)
        self.preprocess_graph = self.set_func(preprocess_graph)

        # 1. capture and scrub pre-set terrains so replay uses the same runtime path
        preset_assignments = self._snapshot_preset_terrains()
        self._scrub_all_terrains()

        # TODO in some cases we might want multiple modules as one function,
        # we should then check for a list of a yet to be named class instances in settings
        # these need to have:
        # - some way to determine at what range elements are affected by it
        # - a method that takes in the affected element, the graph, and the origin element and:
        #    - updates it's dist contribution (if it has any)§
        #    - updates it's constraints contribution (if it has any)
        self.rule_managers = self.settings.get('rule_managers', []) # type: list[RuleManager]
        # For each
        for rule_manager in self.rule_managers:
            # TODO replace check with checking that it is an instance of the new class, when implemented
            if not isinstance(rule_manager, RuleManager):
                raise ValueError(f"Class {rule_manager} in rule_managers is not an instance of RuleManager")
        # 2. collect global metrics for graph
        collected_metrics = self.collect_global_metrics(self.graph, self.settings)
        assert isinstance(collected_metrics, dict), "collect_global_metrics should return a dict of global metrics"
        self.graph.global_metrics = collected_metrics

        # 3. determine target base terrain distributions
        self.determine_target_distributions(self.graph, self.settings)

        # 4. preprocess incoming graph (optional hook)
        self.preprocess_graph(self.graph, self.settings)

        # 5. initialize per-element solver containers before rule setup
        self._initialize_element_solver_state()

        # 6. run static rule setup
        for rule_manager in self.rule_managers:
            rule_manager.setup(self.graph)

        # 7. let rules seed their own per-element contributions
        for rule_manager in self.rule_managers:
            rule_manager.initialize_element_state(self.graph)

        # 8. replay pre-set values via normal assignment update path
        self._replay_preset_terrains(preset_assignments)

        # 9. ensure unset elements have fresh joint distributions
        self.refresh_joint_probabilities()

        # 10. wave function collapse procedure can now be started by
        # calling wave_function_collapse()

    # check if settnigs has override for defaiult functions, if not use default functions
    def set_func(self, default_func):
        func_name = default_func.__name__
        if func_name in self.settings:
            func = self.settings[func_name]
            if not callable(func):
                raise ValueError(f"Function {func_name} provided in settings is not callable")
            # TODO check the signature of the function here as well, to make sure it matches what we expect
            return func
        else:
            return default_func

    def _snapshot_preset_terrains(self) -> list[tuple[Element, object]]:
        preset_assignments = []
        for element in self.graph.get_all_elements():
            terrain = element.get('terrain', None)
            if terrain is not None:
                preset_assignments.append((element, terrain))
        return preset_assignments

    def _scrub_all_terrains(self):
        for element in self.graph.get_all_elements():
            if element.get('terrain', None) is not None:
                element['terrain'] = None

    def _initialize_element_solver_state(self):
        for element in self.graph.get_all_elements():
            if 'dists' not in element or not isinstance(element['dists'], dict):
                element['dists'] = {}
            if 'dist_weights' not in element or not isinstance(element['dist_weights'], dict):
                element['dist_weights'] = {}
            if 'constraints' not in element or not isinstance(element['constraints'], dict):
                element['constraints'] = {}
            if 'flags' not in element or not isinstance(element['flags'], dict):
                element['flags'] = {}

            element['joint_prob_dist'] = {}

    def _replay_preset_terrains(self, preset_assignments: list[tuple[Element, object]]):
        for element, terrain in preset_assignments:
            self.set_element_terrain(element, terrain)

    def set_element_terrain(self, element: Element, terrain):
        # set the terrain for the given element (province or border) in the graph
        element['terrain'] = terrain
        self.apply_assignment_updates(element)

    def apply_assignment_updates(self, origin_element: Element):
        # update any statistics and probabilities after setting an element's terrain

        # update information about origin_element
        # Update global counters
        # TODO consider removing in favour of using update_statistics for it too
        if origin_element.is_node:
            self.graph.global_metrics['set_provinces'] = self.graph.global_metrics.get('set_provinces', 0) + 1
        else:
            self.graph.global_metrics['set_borders'] = self.graph.global_metrics.get('set_borders', 0) + 1

        # update local counters that include the origin_element
        for rule_manager in self.rule_managers:
            rule_manager.update_statistics_for_origin(self.graph, origin_element)

        # determine all nearby elements that are affected by the setting of this element,
        # and update their probabilities based on the new information
        # this should probably call some method on the graph that returns the affected elements
        # and then (for each) call a method with that takes the origin_element and
        # the affected_element and updates them accordingly

        # TODO TODO TODO later get only affected elements, for now we rely on the rules checking if they are relevant
        # affected_elements = self.graph.get_elements_affected_by(origin_element)
        unset_elements = self.graph.get_unset_elements() # TODO replace with get_elements_affected_by when implemented
        for unset_element in unset_elements:
            for rule_manager in self.rule_managers:
                rule_manager.update_affected(unset_element, self.graph, origin_element)

        self.refresh_joint_probabilities()
        # TODO nice-to-have: optional debug-only contradiction checks.

    def refresh_joint_probabilities(self):
        for element in self.graph.get_unset_elements():
            update_joint_probability_distribution(element, self.graph)


    def step_wave_function_collapse(self):
        # perform a single step of the wave function collapse process
        element_to_set = select_element_to_set(self.graph)
        terrain = select_element_terrain(element_to_set, self.graph)
        self.set_element_terrain(element_to_set, terrain)

    # the main control flow function for the wave function collapse process
    # taking in a graph and any settings and returning a graph with terrains set
    def wave_function_collapse(self) -> TerrainGraph:
        while not self.graph.is_all_set():
            self.step_wave_function_collapse()

        return self.graph


def make_wfc_settings_from_global_dist(settings: dict) -> dict:
    """
    Parse a 'base_global_target_dist' settings entry into WFC-ready settings.

    Converts the user-friendly global distribution format into the internal
    representation needed by WaveFunctionCollapse: a terrain domain and a
    RuleManager with a DistRule seeded from the provided distributions.

    Expected input key in settings:
        'base_global_target_dist': {
            'province_terrains': {terrain_name: weight, ...},
            'border_terrains':   {terrain_name: weight, ...},
        }

    Produces (merged into the returned settings dict):
        'base_terrain_domain': {'province_terrains': [...], 'border_terrains': [...]},
        'rule_managers': existing_managers + [RuleManager wrapping a global DistRule],
    """
    assert 'base_global_target_dist' in settings, (
        "make_wfc_settings_from_global_dist expects 'base_global_target_dist' in settings"
    )
    base_dist = settings['base_global_target_dist']
    assert 'province_terrains' in base_dist, "base_global_target_dist must include 'province_terrains'"
    assert 'border_terrains' in base_dist, "base_global_target_dist must include 'border_terrains'"

    adjusting_province_dist = base_dist['province_terrains']
    adjusting_border_dist = base_dist['border_terrains']

    terrain_domain = {
        'province_terrains': list(adjusting_province_dist.keys()),
        'border_terrains': list(adjusting_border_dist.keys()),
    }

    global_rule = DistRule(
        adjusting_province_dist=adjusting_province_dist,
        adjusting_border_dist=adjusting_border_dist,
        name='global_dist_rule',
    )
    global_manager = RuleManager(name='global_dist', rules=[global_rule])

    existing_managers = settings.get('rule_managers', [])
    return {
        **settings,
        'base_terrain_domain': terrain_domain,
        'rule_managers': existing_managers + [global_manager],
    }
