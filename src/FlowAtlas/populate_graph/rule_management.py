from dataclasses import dataclass
from typing import Literal, Optional, Callable, Union
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

from .terrain_graph import Element, TerrainGraph

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


class BanRule(Rule):
    # count terrains of all set elements at/within range of an unset element,
    # counts for each terrain in set1 are given to evaluation function, if any returns true ->
    # all terrains in set2 are added to that elements ban list (0 prob.), and vice versa

    def __init__(self,
                 set1: set,
                 set2: set,
                 range: float = 0.5,
                 evaluation: Callable[[int, int], bool] = lambda count, total: count>0,
                 range_type: Literal['topology', 'geography'] = 'topology',
                 range_check: Literal['eq','leq'] = 'eq', # just these for now, expand if needed
                 name: Optional[str] = None):
        self.set1 = set1
        self.set2 = set2
        self.range = range
        self.evaluation = evaluation
        self.range_type = range_type
        self.range_check = range_check
        self.name = name

    def setup(self, graph: TerrainGraph):
        # for now we assume that Ban rules don't need any setup, but maybe they will later, so we keep the method here
        pass

    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        # TODO this is where the main work of BanRules happens TODO
        pass

    def update_statistics_for_origin(self,
                                     graph: TerrainGraph,
                                     origin_element: Element):
        # for now we assume that Ban rules don't update based only on origin
        pass

class DistRule(Rule):
    # tracks dists (distributions) for flags

    def __init__(self,
                 dist: dict,
                 adjusting_factor: float = 1.0,
                 flag: str = 'all',
                 # TODO consider if flags should be dict, so we can store distance weights there,
                 # or if we should just have a separate dict for distance weights in that case
                 name: Optional[str] = None):
        self.dist = dist
        self.adjusting_factor = adjusting_factor # 1.0 for guarranteed dist match, 0.0 for fixed dist selection
        self.flag = flag # defaults to all if not specified, meaning it applies to all elements
        self.name = name

        # metrics to keep track of
        # TODO later we might want these to be weighted when local dists have uneven impact
        self.target_province_sum = 0.0
        self.target_border_sum = 0.0
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
                    element['flags']['all'] = 1.0
                else:
                    raise ValueError(f"Element {element} has non-dict flags, cannot add 'all' flag for DistRule setup")

                # gather starting metrics
                is_node = element.is_node
                if is_node:
                    self.target_province_sum += 1.0
                else:
                    self.target_border_sum += 1.0

                    # TODO we might want to handle all (already set metrics by just
                    # collecting all assigned elements) in RuleManager and then let it
                    # call update_statistics_for_origin and update_affected for each rule
                    # if 'terrain' in element:
                    #     self.assigned_province_attributes[element['terrain']] += 1
        else:
            raise NotImplementedError("Currently only 'all' flag is supported for DistRule setup, other cases are not implemented yet")
        # TODO handle all other cases!!!

    def update_affected(self,
                        affected_element: Element,
                        graph: TerrainGraph,
                        origin_element: Element):
        # for now we assume that DistRules don't to explicit updates for each affected element
        pass

    def update_statistics_for_origin(self,
                                     graph: TerrainGraph,
                                     origin_element: Element):
        # TODO this is where the main work of DistRules happens TODO
        pass

# tracks a set of rules, and handles applying them to the graph when needed
# mosly a middle layer to structure execution and make weighting easier
class RuleManager:
    def __init__(self,
                 name: str,
                 rules: list[Rule],
                 attribute: str ='terrain'):

        self.name = name
        self.base_weight = 1.0
        self.attribute = attribute # assumed to be 'terrain' for now
        self.rules = rules

    # input params to be determined
    # should mark elements with applicable flags (if 'all', add to all elements)
    # should prep target_dist and adjusting_dist (for each rule) when used
    def setup(self, graph: TerrainGraph):
        for rule in self.rules:
            rule.setup(graph)

        # get all assigned elements, and update as if we just assigned them
        already_assigned_elements = set()
        for element in graph.get_all_elements():
            if self.attribute in element:
                already_assigned_elements.add(element)

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
