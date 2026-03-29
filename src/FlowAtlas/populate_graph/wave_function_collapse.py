import numpy as np
import networkx as nx
from time import perf_counter
from scipy.spatial import Voronoi, Delaunay

from .terrain_graph import Element, TerrainGraph
from .rule_management import RuleManager, DistRule
from ..debug_utils import (
    DebugStatisticsCollector, DebugReportFormatter
)

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
        for terrain, factor in dist.items():
            assert terrain in joint_factors, f"Terrain {terrain} from local dist {name} not found in terrain domain"
            # add factor, but emphasized by dist_weight (higher weight means more influence on final distribution)
            # joint_factors[terrain] *= factor ** weights[name]  # using exponentiation to apply weight as influence level
            joint_factors[terrain] *= factor * weights[name] + (1.0 - weights[name])  # using linear blend to apply weight as influence level

    # apply constraints (zero out banned terrains)
    for name, constraint in element['constraints'].items():
        for terrain in constraint:
            if terrain in joint_factors:
                joint_factors[terrain] = 0.0

    # normalize the joint probability distribution to sum to 1
    sum_factors = sum(joint_factors.values())
    if sum_factors > 0:
        joint_prob_dist = {terrain: f / sum_factors for terrain, f in joint_factors.items()}
    else:
        # TODO consider if this should raise an error instead
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

        self.rule_managers = self.settings.get('rule_managers', []) # type: list[RuleManager]
        # For each
        for rule_manager in self.rule_managers:
            # TODO replace check with checking that it is an instance of the new class, when implemented
            if not isinstance(rule_manager, RuleManager):
                raise ValueError(f"Class {rule_manager} in rule_managers is not an instance of RuleManager")

        # Initialize debug configuration (needs graph and rule_managers to be set)
        self._setup_debug_configuration()

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

        if self.debug_enabled:
            self.debug_stats['initial_state'] = self.debug_collector.get_progress_state()

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


    def step_wave_function_collapse(self, capture_debug_step: bool = False):
        # perform a single step of the wave function collapse process
        element_to_set = select_element_to_set(self.graph)
        if not capture_debug_step:
            terrain = select_element_terrain(element_to_set, self.graph)
            self.set_element_terrain(element_to_set, terrain)
            return None

        # debug only path
        entropy_diagnostics = self._collect_entropy_metrics_from_unset_elements() if self.debug_track_entropy_metrics else None
        option_distribution = dict(element_to_set['joint_prob_dist']) if self.debug_store_option_distributions else None
        terrain = select_element_terrain(element_to_set, self.graph)

        selected_probability = element_to_set['joint_prob_dist'][terrain]
        element_kind = 'province' if element_to_set.is_province else 'border'
        selected_element_id = self._element_debug_id(element_to_set)
        flags_raw = element_to_set.get('flags', {})
        element_flags = dict(flags_raw) if isinstance(flags_raw, dict) else {}
        unset_before_step = len(list(self.graph.get_unset_elements()))

        if entropy_diagnostics is not None:
            entropy_diagnostics['selected_element_id'] = selected_element_id
            entropy_diagnostics['selected_entropy'] = shannon_entropy(element_to_set['joint_prob_dist'])

        rule_diag_before = None
        if self.debug_track_rule_firings or self.debug_track_weight_changes:
            rule_diag_before = self._collect_rule_state_for_debug()

        self.set_element_terrain(element_to_set, terrain)

        rule_diagnostics = None
        if rule_diag_before is not None:
            rule_diag_after = self._collect_rule_state_for_debug()
            rule_diagnostics = self._summarize_rule_state_deltas(rule_diag_before, rule_diag_after)

        step_info = {
            'element_kind': element_kind,
            'selected_terrain': terrain,
            'selected_probability': selected_probability,
            'element_flags': element_flags,
            'selected_element_id': selected_element_id,
            'unset_elements_before_step': unset_before_step,
        }

        if entropy_diagnostics is not None:
            step_info['entropy_diagnostics'] = entropy_diagnostics

        if option_distribution is not None:
            step_info['selected_option_distribution'] = option_distribution

        if rule_diagnostics is not None:
            step_info['rule_diagnostics'] = rule_diagnostics

        return step_info

    # the main control flow function for the wave function collapse process
    # taking in a graph and any settings and returning a graph with terrains set
    def wave_function_collapse(self) -> TerrainGraph:
        if not self.debug_enabled:
            while not self.graph.is_all_set():
                self.step_wave_function_collapse(capture_debug_step=False)
            return self.graph

        total_start = perf_counter() if self.debug_wfc_timing else None

        while not self.graph.is_all_set():
            step_start = perf_counter() if self.debug_wfc_timing else None
            step_info = self.step_wave_function_collapse(capture_debug_step=True)
            step_seconds = (perf_counter() - step_start) if step_start is not None else None
            assert isinstance(step_info, dict)
            self._record_debug_step(step_info, step_seconds)

        total_seconds = (perf_counter() - total_start) if total_start is not None else 0.0

        if self.debug_enabled:
            # Finalize statistics in collector
            self.debug_collector.finalize(total_seconds)

            self._flush_progress_window(force=True)

            if self.debug_print_progress:
                line = (
                    "[WFC Debug] complete "
                    f"steps={self.debug_stats['steps']} "
                    f"completion={1.0:.2%}"
                )
                if self.debug_wfc_timing:
                    line += f" total_seconds={total_seconds:.3f}"
                print(line)

            if self.debug_print_progress_report:
                print(self.format_debug_progress_report())
            if self.debug_print_final_report:
                print(self.format_debug_final_report())

        return self.graph

    # ===== DEBUG STATISTICS & REPORTING (Optional instrumentation) =====
    # Keep the main WFC control flow above and the debug plumbing here.

    def _element_debug_id(self, element: Element) -> str:
        """Create a stable debug id string for an element."""
        if element.is_node:
            return f"province:{element.element_id}"

        edge_id = element.element_id
        if isinstance(edge_id, tuple) and len(edge_id) >= 2:
            return f"border:{edge_id[0]}-{edge_id[1]}"
        return f"border:{edge_id}"

    def _collect_entropy_metrics_from_unset_elements(self) -> dict:
        """Collect per-step entropy diagnostics for all currently unset elements."""
        all_entropies = {}
        province_entropies = {}
        border_entropies = {}

        for element in self.graph.get_unset_elements():
            element_id = self._element_debug_id(element)
            entropy = shannon_entropy(element['joint_prob_dist'])
            all_entropies[element_id] = entropy
            if element.is_province:
                province_entropies[element_id] = entropy
            else:
                border_entropies[element_id] = entropy

        def _summary(values: dict) -> dict:
            if not values:
                return {
                    'count': 0,
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                }

            value_list = list(values.values())
            return {
                'count': len(value_list),
                'min': min(value_list),
                'max': max(value_list),
                'mean': float(np.mean(value_list)),
            }

        return {
            'all': all_entropies,
            'province': province_entropies,
            'border': border_entropies,
            'summary': {
                'all': _summary(all_entropies),
                'province': _summary(province_entropies),
                'border': _summary(border_entropies),
            },
        }

    def _collect_rule_state_for_debug(self) -> dict:
        """Collect a lightweight snapshot of mutable rule state for delta analysis."""
        snapshot = {}
        for manager in self.rule_managers:
            manager_state = {}
            for rule in manager.rules:
                rule_key = getattr(rule, 'rule_key', f"{rule.__class__.__name__}:{id(rule)}")
                rule_state: dict[str, object] = {
                    'rule_type': rule.__class__.__name__,
                }

                # Dist-like rules expose mutable adjusting distributions used in selection weighting.
                if hasattr(rule, 'adjusting_province_dist') and hasattr(rule, 'adjusting_border_dist'):
                    province_dist = getattr(rule, 'adjusting_province_dist')
                    border_dist = getattr(rule, 'adjusting_border_dist')
                    if isinstance(province_dist, dict):
                        rule_state['adjusting_province_dist'] = dict(province_dist)
                    if isinstance(border_dist, dict):
                        rule_state['adjusting_border_dist'] = dict(border_dist)

                manager_state[rule_key] = rule_state

            snapshot[manager.name] = manager_state

        return snapshot

    def _summarize_rule_state_deltas(self, before: dict, after: dict) -> dict:
        """Build step-level rule diagnostics from state before/after assignment."""
        fired_rules = []
        weight_change_summary = []

        for manager_name, manager_after in after.items():
            manager_before = before.get(manager_name, {})
            for rule_key, rule_after in manager_after.items():
                rule_before = manager_before.get(rule_key, {})
                province_before = rule_before.get('adjusting_province_dist', {})
                province_after = rule_after.get('adjusting_province_dist', {})
                border_before = rule_before.get('adjusting_border_dist', {})
                border_after = rule_after.get('adjusting_border_dist', {})

                province_changed = province_before != province_after
                border_changed = border_before != border_after
                changed = province_changed or border_changed

                if not changed:
                    continue

                fired_rules.append({
                    'manager': manager_name,
                    'rule_key': rule_key,
                    'rule_type': rule_after.get('rule_type', 'unknown'),
                })

                if self.debug_track_weight_changes:
                    for terrain, after_val in province_after.items():
                        before_val = province_before.get(terrain)
                        if before_val is None or before_val == after_val:
                            continue
                        weight_change_summary.append({
                            'manager': manager_name,
                            'rule_key': rule_key,
                            'element_kind': 'province',
                            'terrain': terrain,
                            'before': before_val,
                            'after': after_val,
                            'delta': after_val - before_val,
                        })

                    for terrain, after_val in border_after.items():
                        before_val = border_before.get(terrain)
                        if before_val is None or before_val == after_val:
                            continue
                        weight_change_summary.append({
                            'manager': manager_name,
                            'rule_key': rule_key,
                            'element_kind': 'border',
                            'terrain': terrain,
                            'before': before_val,
                            'after': after_val,
                            'delta': after_val - before_val,
                        })

        return {
            'fired_rules': fired_rules,
            'weight_changes': weight_change_summary,
        }

    def _setup_debug_configuration(self):
        """Apply debug level presets and initialize debug flags and collector."""
        debug_level = int(self.settings.get('debug_wfc_level', 0))
        level_additions = {
            0: {
                'debug_wfc_statistics': False,
                'debug_wfc_print_progress': False,
                'debug_wfc_print_progress_report': False,
                'debug_wfc_print_final_report': False,
                'debug_wfc_timing': False,
                'debug_wfc_store_step_snapshots': False,
                'debug_wfc_store_iteration_snapshots': False,
                'debug_wfc_track_entropy_metrics': False,
                'debug_wfc_store_option_distributions': False,
                'debug_wfc_store_full_entropy_maps': False,
                'debug_wfc_track_rule_firings': False,
                'debug_wfc_track_weight_changes': False,
            },
            1: {
                'debug_wfc_statistics': True,
                'debug_wfc_print_progress': True,
                'debug_wfc_store_step_snapshots': True,
            },
            2: {
                'debug_wfc_print_progress_report': True,
                'debug_wfc_timing': True,
                'debug_wfc_store_iteration_snapshots': True,
                'debug_wfc_track_entropy_metrics': True,
            },
            3: {
                'debug_wfc_print_final_report': True,
                'debug_wfc_store_option_distributions': True,
                'debug_wfc_store_full_entropy_maps': True,
                'debug_wfc_track_rule_firings': True,
                'debug_wfc_track_weight_changes': True,
            },
        }

        level_defaults = {}
        cumulative_defaults = {}
        for level in sorted(level_additions):
            cumulative_defaults = {
                **cumulative_defaults,
                **level_additions[level],
            }
            level_defaults[level] = dict(cumulative_defaults)

        defaults = level_defaults.get(debug_level, level_defaults[0])
        for key, default_val in defaults.items():
            if key not in self.settings:
                self.settings[key] = default_val

        self.debug_enabled = bool(self.settings.get('debug_wfc_statistics', False))
        self.debug_print_progress = bool(self.settings.get('debug_wfc_print_progress', False))
        self.debug_print_every = max(1, int(self.settings.get('debug_wfc_print_every', 25)))
        self.debug_store_step_snapshots = bool(self.settings.get('debug_wfc_store_step_snapshots', True))
        self.debug_store_iteration_snapshots = bool(self.settings.get('debug_wfc_store_iteration_snapshots', False))
        self.debug_snapshot_every = max(1, int(self.settings.get('debug_wfc_snapshot_every', 25)))
        self.debug_track_entropy_metrics = bool(self.settings.get('debug_wfc_track_entropy_metrics', True))
        self.debug_store_option_distributions = bool(self.settings.get('debug_wfc_store_option_distributions', False))
        self.debug_store_full_entropy_maps = bool(self.settings.get('debug_wfc_store_full_entropy_maps', False))
        self.debug_track_rule_firings = bool(self.settings.get('debug_wfc_track_rule_firings', False))
        self.debug_track_weight_changes = bool(self.settings.get('debug_wfc_track_weight_changes', False))
        self.debug_print_progress_report = bool(self.settings.get('debug_wfc_print_progress_report', False))
        self.debug_print_final_report = bool(self.settings.get('debug_wfc_print_final_report', False))
        self.debug_wfc_timing = bool(self.settings.get('debug_wfc_timing', False))
        self.debug_wfc_print_distribution_metrics = bool(self.settings.get('debug_wfc_print_distribution_metrics', True))

        self.debug_collector = DebugStatisticsCollector(
            self.graph,
            self.rule_managers,
            store_step_snapshots=self.debug_store_step_snapshots,
            store_iteration_snapshots=self.debug_store_iteration_snapshots,
            snapshot_every=self.debug_snapshot_every,
            track_entropy_metrics=self.debug_track_entropy_metrics,
            store_option_distributions=self.debug_store_option_distributions,
            store_full_entropy_maps=self.debug_store_full_entropy_maps,
            track_rule_firings=self.debug_track_rule_firings,
            track_weight_changes=self.debug_track_weight_changes,
        )
        self.debug_stats = self.debug_collector.stats
        self.debug_stats['enabled'] = self.debug_enabled

    def _flush_progress_window(self, force: bool = False):
        summary = self.debug_collector.flush_window(force=force)
        if summary and self.debug_print_progress:
            print(DebugReportFormatter.format_progress_window_summary(
                summary,
                include_timing=self.debug_wfc_timing,
                include_distribution_metrics=self.debug_wfc_print_distribution_metrics,
            ))
            print()

    def _record_debug_step(self, step_info: dict, step_seconds: float | None):
        if not self.debug_enabled:
            return

        self.debug_collector.record_step(step_info, step_seconds)

        if self.debug_print_progress and self.debug_stats['steps'] % self.debug_print_every == 0:
            self._flush_progress_window(force=False)

    def get_debug_statistics(self) -> dict:
        return self.debug_stats

    def format_debug_progress_report(self) -> str:
        if not self.debug_enabled:
            return "WFC debug statistics are disabled (set debug_wfc_statistics=True)."

        return DebugReportFormatter.format_progress_report(self.debug_stats)

    def format_debug_final_report(self) -> str:
        if not self.debug_enabled:
            return "WFC debug statistics are disabled (set debug_wfc_statistics=True)."

        return DebugReportFormatter.format_final_report(
            self.debug_stats,
            include_distribution_metrics=self.debug_wfc_print_distribution_metrics,
        )

    def compare_debug_iteration_snapshots(self, step_a: int, step_b: int) -> dict:
        """Compare two stored iteration snapshots by step number."""
        if not self.debug_enabled:
            raise RuntimeError("WFC debug statistics are disabled (set debug_wfc_statistics=True).")

        return self.debug_collector.compare_iteration_snapshots(step_a, step_b)



def convert_dist_to_relative_factors(dist: dict) -> dict:
    """
    Convert frequency weights to multiplicative relative factors.

    Args:
        dist: {terrain_name: frequency_weight, ...}

    Returns:
        {terrain_name: relative_factor, ...}

    Notes:
        WFC combines rule distributions multiplicatively, so a uniform
        distribution should contribute neutral factors (1.0 for each terrain).
        We therefore normalize by the mean weight, not by the total sum.
    """
    if len(dist) == 0:
        raise ValueError("Distribution must contain at least one terrain")

    total = sum(dist.values())
    mean_weight = total / len(dist)
    if mean_weight <= 0:
        raise ValueError(f"Distribution mean weight must be > 0, got {mean_weight}")

    return {terrain: weight / mean_weight for terrain, weight in dist.items()}


def make_wfc_settings_from_global_dist(settings: dict, include_global_dist_rule: bool = True) -> dict:
    """
    Parse a 'base_global_target_dist' settings entry into WFC-ready settings.

    Converts the user-friendly global distribution format into the internal
    representation needed by WaveFunctionCollapse: a terrain domain and a
    RuleManager with a DistRule seeded from the provided distributions.

    Distributions are automatically converted to multiplicative relative factors.

    Expected input key in settings:
        'base_global_target_dist': {
            'province_terrains': {terrain_name: weight, ...},
            'border_terrains':   {terrain_name: weight, ...},
        }

    Produces (merged into the returned settings dict):
        'base_terrain_domain': {'province_terrains': [...], 'border_terrains': [...]},
        'rule_managers': existing_managers (+ global DistRule manager by default),
    """
    assert 'base_global_target_dist' in settings, (
        "make_wfc_settings_from_global_dist expects 'base_global_target_dist' in settings"
    )
    base_dist = settings['base_global_target_dist']
    assert 'province_terrains' in base_dist, "base_global_target_dist must include 'province_terrains'"
    assert 'border_terrains' in base_dist, "base_global_target_dist must include 'border_terrains'"

    # Convert input distributions to multiplicative relative factors.
    adjusting_province_dist = convert_dist_to_relative_factors(base_dist['province_terrains'])
    adjusting_border_dist = convert_dist_to_relative_factors(base_dist['border_terrains'])

    terrain_domain = {
        'province_terrains': list(adjusting_province_dist.keys()),
        'border_terrains': list(adjusting_border_dist.keys()),
    }

    existing_managers = settings.get('rule_managers', [])
    if include_global_dist_rule:
        global_rule = DistRule(
            adjusting_province_dist=adjusting_province_dist,
            adjusting_border_dist=adjusting_border_dist,
            name='global_dist_rule',
        )
        global_manager = RuleManager(name='global_dist', rules=[global_rule])
        final_managers = existing_managers + [global_manager]
    else:
        final_managers = existing_managers

    return {
        **settings,
        'base_terrain_domain': terrain_domain,
        'rule_managers': final_managers,
    }
