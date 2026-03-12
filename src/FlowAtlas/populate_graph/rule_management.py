from typing import Literal, Optional, Callable, TypeAlias
from abc import ABC, abstractmethod
from collections import Counter

from .terrain_graph import Element, TerrainGraph

RangeType: TypeAlias = Literal['topology', 'geography']
RangeCheck: TypeAlias = Literal['eq', 'leq']

"""
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


All relationship goals must (and are assumed to) work the same in both directions.
For instance, sea provinces (all types) cannot have river borders (including river with bridge),
that means that both:
"river"->no "sea" adjacent
"sea"->no "river" adjacent
We should make it so that writing one of the rules, will enforce both it and its complement


"""
class Rule(ABC):
    # base class for rules, mostly for type checking and organization
    pass

    @abstractmethod
    def setup(self, graph: TerrainGraph):
        raise NotImplementedError("setup method must be implemented by subclasses of Rule")

    @abstractmethod
    def initialize_element(self,
                           element: Element,
                           graph: TerrainGraph,
                           manager_weight: float):
        raise NotImplementedError("initialize_element method must be implemented by subclasses of Rule")

    @abstractmethod
    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        raise NotImplementedError("update_affected method must be implemented by subclasses of Rule")

    @abstractmethod
    def update_statistics_for_origin(self,
                                     graph: TerrainGraph,
                                     origin_element: Element):
        raise NotImplementedError("update_statistics_for_origin method must be implemented by subclasses of Rule")

# TODO make new rule, or add support for the following behaviour in existing rules:
# dists determined by local neighborhood without flags, like BanRule
# but instead of banning terrains, it adjusts a local dist_contrbution
# it should take an evaluation function, range_type and range_check
# passing all neighboring elements that match the range_type and range_check
# to the evaluation function, and then use the result to determine
# how to adjust the local dist contribution for this rule

class BanRule(Rule):
    # count terrains of all set elements at/within range of an unset element,
    # counts for each terrain in set1 are given to evaluation function, if any returns true ->
    # all terrains in set2 are added to that elements ban list (0 prob.), and vice versa

    def __init__(self,
                 set1: set,
                 set2: set,
                 range: float = 0.5,
                 evaluation: Callable[[list], bool] = lambda set_elements: len(set_elements) > 0,
                 range_type: RangeType = 'topology',
                 range_check: RangeCheck = 'eq', # just these for now, expand if needed
                 include_distance: bool = False,
                 name: Optional[str] = None):
        self.set1 = set1
        self.set2 = set2
        self.range = range
        self.evaluation = evaluation
        self.range_type: RangeType = range_type
        self.range_check: RangeCheck = range_check
        self.include_distance: bool = include_distance
        self.name = name
        self.rule_key = name or f"{self.__class__.__name__}_{id(self)}"

    def setup(self, graph: TerrainGraph):
        # for now we assume that Ban rules don't need any setup, but maybe they will later, so we keep the method here
        pass

    def initialize_element(self,
                           element: Element,
                           graph: TerrainGraph,
                           manager_weight: float):
        domain = graph.global_metrics.get('terrain_domain', None)
        assert isinstance(domain, dict), "graph.global_metrics['terrain_domain'] must be set before BanRule initialization"

        if element.is_province:
            allowed_terrains = domain['province_terrains']
        else:
            allowed_terrains = domain['border_terrains']

        if 'constraints' not in element or not isinstance(element['constraints'], dict):
            element['constraints'] = {}

        # Keep per-rule bans as sets; membership is the only operation the solver needs.
        element['constraints'][self.rule_key] = set()

    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        neighborhood = graph.get_connected_elements(
            affected_element,
            mode=self.range_type,
            range=self.range,
            range_check=self.range_check,
            element_kind='both',
            include_distance=self.include_distance,
        )

        # Reset this rule's constraint entry and recompute from current assignments.
        domain = graph.global_metrics.get('terrain_domain', None)
        assert isinstance(domain, dict), "graph.global_metrics['terrain_domain'] must be set before BanRule updates"
        if affected_element.is_province:
            allowed_terrains = domain['province_terrains']
        else:
            allowed_terrains = domain['border_terrains']

        allowed_terrain_set = set(allowed_terrains)
        rule_constraints = set()

        if self.include_distance:
            set1_neighbors = [
                neighbor_info
                for neighbor_info in neighborhood
                if neighbor_info[0].get('terrain', None) in self.set1
            ]
            set2_neighbors = [
                neighbor_info
                for neighbor_info in neighborhood
                if neighbor_info[0].get('terrain', None) in self.set2
            ]
        else:
            set1_neighbors = [neighbor for neighbor in neighborhood if neighbor.get('terrain', None) in self.set1]
            set2_neighbors = [neighbor for neighbor in neighborhood if neighbor.get('terrain', None) in self.set2]

        if self.evaluation(set1_neighbors):
            for terrain in self.set2:
                if terrain in allowed_terrain_set:
                    rule_constraints.add(terrain)

        if self.evaluation(set2_neighbors):
            for terrain in self.set1:
                if terrain in allowed_terrain_set:
                    rule_constraints.add(terrain)

        if 'constraints' not in affected_element or not isinstance(affected_element['constraints'], dict):
            affected_element['constraints'] = {}
        affected_element['constraints'][self.rule_key] = rule_constraints

    def update_statistics_for_origin(self,
                                     graph: TerrainGraph,
                                     origin_element: Element):
        # for now we assume that Ban rules don't update based only on origin
        pass

class DistRule(Rule):
    # tracks dists (distributions) for flags

    def __init__(self,
                 adjusting_province_dist: dict,
                 adjusting_border_dist: dict,
                 adjusting_factor: float = 1.0,
                 flag: str = 'all',
                 # TODO consider if flags should be dict, so we can store distance weights there,
                 # or if we should just have a separate dict for distance weights in that case
                 name: Optional[str] = None):
        # Shared references — mutate in-place to steer all province/border elements simultaneously.
        self.adjusting_province_dist = adjusting_province_dist
        self.adjusting_border_dist = adjusting_border_dist
        # Immutable targets used to recompute adjusting dists as assignments accumulate.
        self.target_province_dist = adjusting_province_dist.copy()
        self.target_border_dist = adjusting_border_dist.copy()
        self.adjusting_factor = adjusting_factor # 1.0 for guarranteed dist match, 0.0 for fixed dist selection
        self.flag = flag # defaults to all if not specified, meaning it applies to all elements
        self.name = name
        self.rule_key = name or f"{self.__class__.__name__}_{id(self)}"

        # metrics to keep track of
        # TODO later we might want these to be weighted when local dists have uneven impact
        self.total_rule_provinces = 0.0
        self.total_rule_borders = 0.0
        self.assigned_province_attributes = Counter()
        self.assigned_border_attributes = Counter()

        # TODO handle weight by distance for for cap "region" type stuff
        # weight: float = 1.0 # hmm, maybe just rely on RuleManager weight?
        # TODO determine if this should also have range, or if all affected are just marked with flag?
        # range: float = 0.0
        # range_type: Literal['topology', 'geography'] = 'topology'
        # range_check: Literal['eq','leq'] = 'eq' # just these for now, expand if needed

    def setup(self, graph: TerrainGraph):
        if self.flag == 'all':
            for element in graph.get_all_elements():
                # set flags for elements, value is a weight only used when flags
                # can vary in strength, like with distance from a capital, 1.0 is default
                if 'flags' not in element:
                    element['flags'] = {'all': 1.0}
                elif isinstance(element['flags'], dict):
                    if 'all' in element['flags'] and element['flags']['all'] != 1.0:
                        raise NotImplementedError(
                            f"DistRule '{self.rule_key}' currently requires flag weight 1.0 for flag 'all', "
                            f"got {element['flags']['all']} on {element}"
                        )
                    element['flags']['all'] = 1.0
                else:
                    raise ValueError(f"Element {element} has non-dict flags, cannot add 'all' flag for DistRule setup")

                # gather starting metrics
                is_node = element.is_node
                if is_node:
                    self.total_rule_provinces += 1.0
                else:
                    self.total_rule_borders += 1.0

                    # TODO we might want to handle all (already set metrics by just
                    # collecting all assigned elements) in RuleManager and then let it
                    # call update_statistics_for_origin and update_affected for each rule
                    # if 'terrain' in element:
                    #     self.assigned_province_attributes[element['terrain']] += 1
        else:
            raise NotImplementedError("Currently only 'all' flag is supported for DistRule setup, other cases are not implemented yet")
        # TODO handle all other cases!!!

    def initialize_element(self,
                           element: Element,
                           graph: TerrainGraph,
                           manager_weight: float):
        if self.flag != 'all':
            raise NotImplementedError("Currently only 'all' flag is supported for DistRule initialization")

        # Direct reference (not a copy): mutating shared adjusting dists in-place
        # automatically steers all elements pointing to it without re-running initialize_element.
        if element.is_province:
            element['dists'][self.rule_key] = self.adjusting_province_dist
        else:
            element['dists'][self.rule_key] = self.adjusting_border_dist

        element_flag_weight = element['flags'][self.flag]
        if element_flag_weight != 1.0:
            raise NotImplementedError(
                f"DistRule '{self.rule_key}' currently requires flag weight 1.0 for all elements, got {element_flag_weight}"
            )
        element['dist_weights'][self.rule_key] = manager_weight * element_flag_weight

    def _recompute_adjusting_dist(self,
                                  element_kind: Literal['province', 'border']):
        if element_kind == 'province':
            target_dist = self.target_province_dist
            assigned_counts = self.assigned_province_attributes
            total_elements = self.total_rule_provinces
            live_dist = self.adjusting_province_dist
        else:
            target_dist = self.target_border_dist
            assigned_counts = self.assigned_border_attributes
            total_elements = self.total_rule_borders
            live_dist = self.adjusting_border_dist

        if total_elements <= 0:
            return

        adjusting_dist = {}
        for terrain, target_factor in target_dist.items():
            current_count = assigned_counts.get(terrain, 0)
            target_count = target_factor * total_elements
            gap = target_count - current_count
            adjustment_part = self.adjusting_factor * (gap / total_elements)
            target_part = (1.0 - self.adjusting_factor) * target_factor
            adjusted_factor = max(0.0, adjustment_part + target_part)
            adjusting_dist[terrain] = adjusted_factor

        total_adjusted = sum(adjusting_dist.values())
        if total_adjusted > 0:
            normalized = {terrain: weight / total_adjusted for terrain, weight in adjusting_dist.items()}
        else:
            n = len(adjusting_dist)
            normalized = {terrain: 1.0 / n for terrain in adjusting_dist}

        # Mutate in place so all elements holding this dict reference are updated immediately.
        live_dist.clear()
        live_dist.update(normalized)

    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        # DistRule updates shared adjusting distributions in update_statistics_for_origin.
        # Per-element updates are intentionally no-op for MVP while all rule weights are fixed at 1.0.
        return

    def update_statistics_for_origin(self,
                                     graph: TerrainGraph,
                                     origin_element: Element):
        # Track assignments and update shared adjusting distributions once per origin assignment.
        if 'terrain' not in origin_element or origin_element['terrain'] is None:
            return
        terrain = origin_element['terrain']

        # Update assignment counters based on element type
        if origin_element.is_province:
            self.assigned_province_attributes[terrain] += 1
        else:
            self.assigned_border_attributes[terrain] += 1

        self._recompute_adjusting_dist('province')
        self._recompute_adjusting_dist('border')

# tracks a set of rules, and handles applying them to the graph when needed
# mosly a middle layer to structure execution and make weighting easier
class RuleManager:
    def __init__(self,
                 name: str,
                 rules: list[Rule],
                 attribute: str ='terrain'):

        self.name = name
        self.manager_weight = 1.0
        self.attribute = attribute # assumed to be 'terrain' for now
        self.rules = rules

    # input params to be determined
    # should mark elements with applicable flags (if 'all', add to all elements)
    # should prep target_dist and adjusting_dist (for each rule) when used
    def setup(self, graph: TerrainGraph):
        for rule in self.rules:
            rule.setup(graph)

    def initialize_element_state(self, graph: TerrainGraph):
        assert self.attribute == 'terrain', (
            f"RuleManager '{self.name}' has unsupported attribute '{self.attribute}'. "
            "Only attribute='terrain' is currently supported."
        )

        for element in graph.get_all_elements():
            for rule in self.rules:
                rule.initialize_element(element, graph, self.manager_weight)

    # update the dist and constraint entry for this rule in affected_element
    # using a set of rules from this manager
    # should do:
    # 1. check if it's rules applies to the affected element and attribute is None, if not return immediately
    # 2. if it does, evaluate and update any dist contributions and bans for these rules in the affected element
    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        for rule in self.rules:
            rule.update_affected(affected_element, graph, origin_element)

    # for statistics, check if the origin_element has flags relevant for any rules of this manager
    # returns a list of tuples of the form (rule, flags)
    # def get_rules_by_flags(self, element: Element):
    #     # get element flags and bail if none
    #     element_flags = getattr(element, 'flags', None)
    #     if element_flags is None:
    #         return []
    #     relevant_rules = list()
    #     # gather each flagged rule, and return with relevant flags
    #     for rule in self.rules:
    #         flags = getattr(rule, 'flags', None)
    #         if flags is None:
    #             continue
    #         per_rule_flags = []
    #         for flag in flags:
    #             if flag in element_flags:
    #                 per_rule_flags.append(flag)
    #         if per_rule_flags:
    #             relevant_rules.append((rule, per_rule_flags))
    #     return relevant_rules

    # update any statistics that this rule manager uses for the origin element
    # should return immediately if the origin element is not relevant for this rule manager
    def update_statistics_for_origin(self, graph: TerrainGraph, origin_element: Element):
        for rule in self.rules:
            rule.update_statistics_for_origin(graph, origin_element)

        # applicable_rules = self.get_rules_by_flags(origin_element)
        # if not applicable_rules:
        #     return
        # # update the dist for each flag found


        # get old dist from element, for now mostly for debugging
        # old_dists = origin_element.get('dists', None)
        # if old_dists:
        #     old_dist = old_dists.get(self.name, None)
        # else:
        #     old_dist = None
        # new_dist_components = []
        # new_bans = set() # we don't expect bans from origin but maybe there are
        # for rule, flags in applicable_rules:
        #     dist_contrib = dict()
        #     ban_contrib = set()
        #     # TODO collect rule contributions

        #     new_dist_components.append((rule.name, dist_contrib)) # names mostly tracked for debugging
        #     new_bans.update(ban_contrib)
