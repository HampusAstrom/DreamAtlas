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
2. *determine target base terrain distribution (implied probabilities)
    2.1 *global target_dist (implied prob) from:
        2.1.1 *base default
        2.1.2 settings
        2.1.3 nation preferences
    2.2 national "nearby" dist (implied prob) preferences from:
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
"""