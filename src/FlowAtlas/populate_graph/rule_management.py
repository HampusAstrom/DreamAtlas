from dataclasses import dataclass
from typing import Literal, Optional, Callable

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

@dataclass
class BanRule:
    # count terrains of all set elements at/within range of an unset element,
    # counts for each terrain in set1 are given to evaluation function, if any returns true ->
    # all terrains in set2 are added to that elements ban list (0 prob.), and vice versa
    set1: set
    set2: set
    range: float = 0.5
    evaluation: Callable[[int, int], bool] = lambda count, total: count>0
    range_type: Literal['topology', 'geography'] = 'topology'
    range_check: Literal['eq','leq'] = 'eq' # just these for now, expand if needed
    name: Optional[str] = None


@dataclass
class WeightRule:
    # adds dists as constant dist for any it applies to
    # if flags are used, dists is expected to have one subdict for each flag
    # TODO handle weight by distance for for cap "region" type stuff
    dists: dict
    adjusting_factor: float # 1.0 for guarranteed dist match, 0.0 for fixed dist selection
    weight: float = 1.0
    # TODO determine if this should also have range, or if all affected are just marked with flag?
    # range: float = 0.0
    # range_type: Literal['topology', 'geography'] = 'topology'
    # range_check: Literal['eq','leq'] = 'eq' # just these for now, expand if needed
    flags: Optional[set] = None # if no flag(s), applies everywhere
    name: Optional[str] = None


class RuleManager:
    def __init__(self, name, attribute='terrain'):

        self.name = name
        self.base_weight = 1.0
        self.attribute = attribute


    # update the dist and constraint entry for this rule in affected_element
    # using a set of rules from this manager
    def update(self, affected_element, graph, origin_element):
        pass

    def update_statistics(self, graph, origin_element):
        pass