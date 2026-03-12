from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from FlowAtlas.voronoi_utils import color_voronoi_faces
from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse
from FlowAtlas.populate_graph.rules_library import make_default_wfc_settings
from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph

# TODO consider what the actual default usage of WFC should look like, how should
# the user or other code setup/define rules and target distributions for it?

# Current default rules and distributions live in rules_library.py.
wfc_settings = make_default_wfc_settings()

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
    # TODO get nx.Draw to work for TerrainGraph or make a workaround for visualising TerrainGraphs connections
    # TODO the goal is to show all borders color/linestyle codes as well
    # as either links between node centers, or as Voronoi edge colors
    # (ideally both options should be supported and usable at the same time)
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