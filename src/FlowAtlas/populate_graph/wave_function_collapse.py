import numpy as np
import networkx as nx
from scipy.spatial import Voronoi, Delaunay

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
2. *determine target base terrain distributions (implied probabilities)
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

Contributions to selection distribution, stored separately for easier update:
* marking required parts for minimal prototype
non-marked are planned expansions
- *base adjusting_dist
- local adjusting_dist (often distance weighted)
    - might consider implied terrain impact on income balance
      (gold (population), resources, gems) but those could be adjusted later
    - some nations want some terrains in their perifery, but not in cap circle
      some want at least 1 water next to cap, but not too many
      these things should also be handled
- *conditional probabilities from nearby set nodes/borders, base on:
    - *distance (closer has more influence, most might only care about immediate neighbors)
    - *terrains realism
- pathing balance (intertwined with nearby nodes concept above, but might be separatable), looks at:
    - province terrain impact on movement via terrain cost and special movement possibility like sailing
    - border terrain impact on movement via pass/river crossing
    keeping in mind:
    - rivers can have bridges when needed to make sense for nature and still get pathing balance
    - passes can be blocking, or big enough to be open
      (thus not actually being a pass, but looking like one if d6m supprots it art wise)
    - paths capital2capital
        - each nation must have several options for a first war, at least three
          ideally 4 or more reachable neighbour nations, though ease of reach
          may vary and must not be symmetrical between the neighbour nations
    - paths capital2thone (and compared to others for same throne)
        - thrones should track reachability by it's nearby nations, and be
          somewhat balance in how hard they are to reach of each such nation
        - nations should have simular access to thrones, both in numbers and
          reachability, but we shouldn't go overboard with balance (it makes
          maps less varied and fun, and thrones are diffrently good anyway,
          unless we can specify what exact throne each should be)
    - base movement using human default
    - national movement preferences and capabilities
        - nations should get some benefit from their abilities,
            but should it should not be too much, and they must be reachable
            "normally" by other nations in most cases
            (uw nations and island start nations being and exception)
        - TODO we need to consider if this should look at nations
            wider set of neighbour nation connections?
        - maybe give less movement strong nations slightly higher chance of road,
          both to compensate for problems and for the realism of them needing
          it more, roads work both ways so it's not only beneficial
    - TODO we could consider if we can balance movement benefits with province
      "value" benefits (gold, resources, gems)

The intention is that most components of this process should be modular
so we can make a basic version and then add to it and replace and expand option
for how to do the generation over time, with option to toggle parts in settings.
"""

# OBS: in the graph, nodes correspond to provinces,
# and edges correspond to borders between provinces

# TODO determine if we interact with edges correctly (nodes seems to be correct)

# TODO this method everyhere to avoid more if/else checks and repeated logic
def get_element(graph: nx.Graph, element):
    if element in graph.nodes:
        return graph.nodes[element]
    elif element in graph.edges:
        return graph.edges[element]
    else:
        raise ValueError(f"Element {element} not found in graph")

# TODO this method everyhere to avoid more if/else checks and repeated logic
# TODO determine if we should make our own graph version that has this instead
# so we can use it more consistently thoughout the code
def get_all_elements(graph: nx.Graph):
    for node in graph.nodes:
        yield graph.nodes[node]
    for edge in graph.edges:
        yield graph.edges[edge]

# placeholder functions for the various steps in the process, to be implemented later
def collect_global_metrics(graph: nx.Graph, settings: dict):
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

def determine_target_distributions(global_metrics: dict, settings: dict):
    # determine the target terrain distribution based on global metrics and settings
    assert 'base_global_target_dist' in settings
    assert 'base_global_weight' in settings

    print("determining target distributions, currently just using base global target dist from settings")
    # TODO determine if base_global_target_dist  and base_global_weight should be a copy or pointer
    global_metrics['global_target_dist'] = settings['base_global_target_dist']
    global_metrics['global_target_weight'] = settings['base_global_weight']
    global_metrics['global_adjusting_dist'] = global_metrics.get('global_target_dist', {}).copy()


    # TODO setup national target_dist and adjusting_dist
    # set global dists as a weighted average of national dists and
    # a base global dist

def preprocess_graph(graph: nx.Graph, global_metrics: dict, settings: dict):
    # preprocess the graph, checking for any predefined values and calculating initial statistics

    print("preprocessing graph, currently we:")
    print("- cleaning any existing terrain values from graph, and setting to None and warning")
    print("- add a 'pointer' to global_adjusting_dist in dict 'dists' for element (node/province and edge/border), making sure to use the respective dicts for each only")
    # TODO add parts for print above, and make a clearner print (and maybe move some to a debug flag)
    for element in get_all_elements(graph):
        # setup pointer to global adjusting dist for each element
        # so it stays updated when we update the global adjusting dist
        if 'dists' not in element:
            element['dists'] = {}
        if element in graph.nodes:
            dist = global_metrics['global_adjusting_dist']['province_terrains']
        else: # edge
            dist = global_metrics['global_adjusting_dist']['border_terrains']
        element['dists']['global_adjusting_dist'] = dist

        # clean out any existing terrain values, and warn if we find any
        if 'terrain' in element and element['terrain'] is not None:
            print(f"warning: element {element} already has terrain {element['terrain']}, resetting to None")
            element['terrain'] = None

def all_provNbor_set(graph: nx.Graph, global_metrics: dict):
    # check if all provinces and borders in the graph have their terrain set
    # we use a running counter and check it primarily, and only do a full check
    # if it indicates we might be done
    if global_metrics.get('set_provinces', 0) < global_metrics.get('provinces', 0):
        return False
    if global_metrics.get('set_borders', 0) < global_metrics.get('borders', 0):
        return False

    # we could do a full check here to be sure, if counters indicate that we should be done
    print("count indicates that all provinces and borders are set, doing a full check to confirm")
    for node in graph.nodes:
        if graph.nodes[node].get('terrain', None) is None:
            return False
    for edge in graph.edges:
        if graph.edges[edge].get('terrain', None) is None:
            return False
    return True

def calculate_joint_probability_distribution(element, global_metrics: dict):
    # calculate the joint probability distribution for the given element
    # (province or border) based on global and local factors

    # each contributing factor to the distribution is assumed to provide:
    # a weight and a distribution
    # they may also provide a list of constraints (banned terrains)

    # combine the various factors into a single probability distribution
    # as a weighted average
    weights = element['dist_weights']
    # TODO replace with pointer to correct dict for provinces or borders TODO TODO TODO
    joint_prob_dist = {terrain: 0.0 for terrain in global_metrics.get('global_target_dist', {})}
    for name, dist in element['dists'].items(): # note, a pointer to the global dist should be included here as well
        assert len(dist) == len(joint_prob_dist), f"Local dist {name} has different length than global adjusting dist"
        assert name in weights, f"Dist {name} for element {element} does not have a corresponding weight in dist_weights"
        weight = weights[name]
        for terrain, prob in dist.items():
            assert terrain in joint_prob_dist, f"Terrain {terrain} from local dist {name} not found in global adjusting dist"
            joint_prob_dist[terrain] += prob * weight

    # apply constraints
    for name, constraint in element['constraints'].items():
        for terrain, ban in constraint.items():
            if ban:
                joint_prob_dist[terrain] = 0.0

    # normalize the joint probability distribution to sum to 1
    sum_probs = sum(joint_prob_dist.values())
    if sum_probs > 0:
        for terrain in joint_prob_dist:
            joint_prob_dist[terrain] /= sum_probs
    else:
        print(f"warning: joint probability distribution for element {element} has sum of zero, using uniform distribution")
        num_terrains = len(joint_prob_dist)
        for terrain in joint_prob_dist:
            joint_prob_dist[terrain] = 1.0 / num_terrains

    return joint_prob_dist

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

def select_element_to_set(graph: nx.Graph, global_metrics: dict):
    # select the next element (province or border) to set based on entropy and other factors

    # check entropy of unset provinces and select the one with lowest entropy,
    # as determined by it's joint probability distribution,
    # with a random fallback (that should warn when trigged)
    element_to_set = []
    lowest_entropy = float('inf')
    for node in graph.nodes:
        if graph.nodes[node].get('terrain', None) is None:
            # TODO, instead of calculating the joint probability distribution for each node here,
            # we should store it in the node and only update it when needed, which should be much more efficient
            joint_prob_dist = calculate_joint_probability_distribution(graph.nodes[node], global_metrics)
            entropy = shannon_entropy(joint_prob_dist)
            if entropy == lowest_entropy: # if many with same, list all and then select randomly
                element_to_set.append(node)
            if entropy < lowest_entropy:
                lowest_entropy = entropy
                element_to_set = [node]

    for edge in graph.edges:
        if graph.edges[edge].get('terrain', None) is None:
            # TODO, instead of calculating the joint probability distribution for each edge here,
            # we should store it in the edge and only update it when needed, which should be much more efficient
            joint_prob_dist = calculate_joint_probability_distribution(graph.edges[edge], global_metrics)
            entropy = shannon_entropy(joint_prob_dist)
            if entropy == lowest_entropy: # if many with same, list all and then select randomly
                element_to_set.append(edge)
            if entropy < lowest_entropy:
                lowest_entropy = entropy
                element_to_set = [edge]

    return np.random.choice(element_to_set)

def select_element_terrain(element, graph: nx.Graph, global_metrics: dict):
    # select a terrain for the given element (province or border) based on probabilities and constraints

    # TODO, instead of calculating the joint probability distribution for each element here,
    # we should store it in the element and only update it when needed, which should be much more efficient
    joint_prob_dist = calculate_joint_probability_distribution(element, global_metrics)
    return np.random.choice(list(joint_prob_dist.keys()), p=list(joint_prob_dist.values()))

def set_element_terrain(element, terrain, graph: nx.Graph, global_metrics: dict):
    # set the terrain for the given element (province or border) in the graph
    if element in graph.nodes:
        graph.nodes[element]['terrain'] = terrain
    elif element in graph.edges:
        graph.edges[element]['terrain'] = terrain
    else:
        raise ValueError(f"Element {element} not found in graph")
    update_statistics_and_probabilities(element, graph, global_metrics)

def update_statistics_and_probabilities(element, graph: nx.Graph, global_metrics: dict):
    # update any statistics and probabilities after setting a element's terrain

    # TODO we need to do a lot more here
    joint_prob_dist = calculate_joint_probability_distribution(graph.nodes[element], global_metrics)
    # TODO more


class WaveFunctionCollapse:
    def __init__(self, settings: dict, graph: nx.Graph):
        self.settings = settings
        self.graph = graph
        # 0. setup any function overrides from settings
        self.collect_global_metrics = self.set_func(collect_global_metrics)
        self.determine_target_distributions = self.set_func(determine_target_distributions)
        self.preprocess_graph = self.set_func(preprocess_graph)

        # TODO in some cases we might want multiple modules as one function,
        # we should then check for a list of function in settings if when it's a list
        # it should wrap it and call all of them in order (before possibly merging results somehow)

        # 1. collect global metrics for graph
        self.global_metrics = self.collect_global_metrics(self.graph, self.settings)
        assert isinstance(self.global_metrics, dict), "collect_global_metrics should return a dict of global metrics"

        # 2. determine target base terrain distributions
        self.determine_target_distributions(self.global_metrics, self.settings)

        # 3. preprocess incoming graph
        self.preprocess_graph(self.graph, self.global_metrics, self.settings)

        # 4. start wave function collapse procedure
        # TODO determine if wave_function_collapse() should be called in init or later
        # TODO determine if we want to be able to step single steps in wave_function_collapse
        # so it can be inspected more easily partway
        # self.wave_function_collapse()

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

    # the main control flow function for the wave function collapse process
    # taking in a graph and any settings and returning a graph with terrains set
    def wave_function_collapse(self) -> nx.Graph:
        assert isinstance(self.global_metrics, dict), "collect_global_metrics should return a dict of global metrics"
        while not all_provNbor_set(self.graph, self.global_metrics):
            element_to_set = select_element_to_set(self.graph, self.global_metrics)
            terrain = select_element_terrain(element_to_set, self.graph, self.global_metrics)
            set_element_terrain(element_to_set, terrain, self.graph, self.global_metrics)

        return self.graph
