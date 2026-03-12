from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from FlowAtlas.voronoi_utils import color_voronoi_faces
from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse, make_wfc_settings_from_global_dist
from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph

settings = {
    # this test config includes all normal single terrain types on the surface layer
    # but no cave terrain or special stuff we might want to add later
    # and it assume no combied terrains
    'base_global_target_dist': {
        # missing for later, all cave combinations (inc cave wall),
        # montains and freshwater (from border terrains, and maybe lake neighbour?)
        # small/large province
        # no_start, good_start, bad_start, good_throne_location, bad_throne_location
        # warmer, colder
        # rare terrain masks (one per magic path)
        # also check for others seen in dominions_data.py:
        # unknown, invisible, vast, infernal waste, void, has gate, flooded,
        # attackers rout once, Cave wall effect/draw as cave, Draw as UW, and some ???
        'province_terrains': {
            'plains': 0.6,
            'highlands': 0.2,
            'swamp': 0.1,
            'waste': 0.1,
            'forest': 0.25,
            'farm': 0.15,
            'sea': 0.15,
            'gorge': 0.02, # a combination terrain in d6m map files, but we list it separately
            'kelp_forest': 0.05, # a combination terrain in d6m map files, but we list it separately
            'deep_sea': 0.03, # a combination terrain in d6m map files, but we list it separately
        },
        'border_terrains': {
            'normal': 0.7,
            'mountain_pass': 0.05,
            'river': 0.05,
            # 'impassable': 0.0,
            'road': 0.02,
            'river_with_bridge': 0.02,
            'impassable_mountain_pass': 0.05,
        }
    }
}

# convert from proabilities to factors
# TODO we should move this to wave_function_collapse.py
summed = sum(settings['base_global_target_dist']['province_terrains'].values())
temp = {terrain: prob / summed for terrain, prob in settings['base_global_target_dist']['province_terrains'].items()}
settings['base_global_target_dist']['province_terrains'] = temp
summed = sum(settings['base_global_target_dist']['border_terrains'].values())
temp = {terrain: prob / summed for terrain, prob in settings['base_global_target_dist']['border_terrains'].items()}
settings['base_global_target_dist']['border_terrains'] = temp

# Parse the global dist into WFC-ready settings: extracts terrain domains and creates a DistRule/RuleManager.
wfc_settings = make_wfc_settings_from_global_dist(settings)

# Example usage:
# import networkx as nx
# graph = nx.Graph()
# graph.add_nodes_from([...])
# graph.add_edges_from([...])
# wfc = WaveFunctionCollapse(wfc_settings, graph)
# result = wfc.wave_function_collapse()

def test_wfc():
    num_points = 100
    # generate (empty) node graph
    points = gen_grid(num_points)
    points = spread_points(points)
    graph_plain, voronoi = voronoi_and_graph(points)

    wfc = WaveFunctionCollapse(wfc_settings, graph_plain)
    # overwriting graph with new (TerrainGraph) with assignments
    graph = wfc.wave_function_collapse()

    #print("WFC result:", graph.nodes(data='terrain'))

    # visualize the graph with assigned terrains
    terrain_colors = {
        'plains': '#c2fc4c',
        'forest': 'darkgreen',
        'highlands': "#867D5F",
        'swamp': 'olive',
        'waste': "#D38C6B",
        'farm': 'yellow',
        'sea': 'deepskyblue',
        'kelp_forest': 'lightseagreen',
        'gorge': 'midnightblue',
        'deep_sea': 'blue',
    }

    node_colors = [terrain_colors[graph.nodes[n]['terrain']] for n in graph.nodes]
    pos = {n: n for n in graph_plain.nodes} # use the node coordinates as positions for visualization
    print(pos.popitem())
    print("Sample node:", list(graph.nodes)[0])
    print("Node type:", type(list(graph.nodes)[0]))

    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)

    fig, ax = plt.subplots(figsize=(11, 10))
    point2color = {n: terrain_colors[graph.nodes[n]['terrain']] for n in graph.nodes}
    color_voronoi_faces(point2color)
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8)
    #nx.draw(graph_plain, pos=pos, ax=ax, node_color='red', with_labels=False, node_size=20, edge_color='white', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height]) # type: ignore
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=terrain, markerfacecolor=color, markersize=10) for terrain, color in terrain_colors.items()] # type: ignore
    ax.legend(handles=legend_elements, title="Terrain Types", bbox_to_anchor=(1.02, 0.5), loc='center left')
    plt.title('WFC Terrain Assignment on Voronoi Graph')
    plt.show()

if __name__ == "__main__":
    test_wfc()