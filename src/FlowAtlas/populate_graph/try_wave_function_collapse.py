from scipy.spatial import voronoi_plot_2d
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from FlowAtlas.voronoi_utils import color_voronoi_faces
from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse
from FlowAtlas.populate_graph.rules_library import make_default_wfc_settings
from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph

# TODO consider what the actual default usage of WFC should look like, how should
# the user or other code setup/define rules and target distributions for it?

# Current default rules and distributions live in rules_library.py.
wfc_settings = make_default_wfc_settings()

# Border render mode options:
# - 'center': draw graph borders as center-to-center connections
# - 'ridge': draw graph borders on finite Voronoi ridges when available
# - 'both': draw both overlays at the same time
BORDER_RENDER_MODE = 'both'

PROVINCE_COLORS = {
    'plains': '#c2fc4c',
    'forest': 'darkgreen',
    'highlands': '#867D5F',
    'swamp': 'olive',
    'waste': '#D38C6B',
    'farm': 'yellow',
    'sea': 'deepskyblue',
    'kelp_forest': 'lightseagreen',
    'gorge': 'midnightblue',
    'deep_sea': 'blue',
}

BORDER_STYLES = {
    'normal': {'color': 'white', 'style': 'solid', 'width': 1.6, 'alpha': 0.8},
    'mountain_pass': {'color': 'saddlebrown', 'style': 'solid', 'width': 2.0, 'alpha': 0.95},
    'river': {'color': 'dodgerblue', 'style': 'solid', 'width': 2.0, 'alpha': 0.95},
    'impassable': {'color': 'black', 'style': 'dashed', 'width': 2.1, 'alpha': 0.95},
    'road': {'color': 'gold', 'style': 'dashed', 'width': 1.8, 'alpha': 0.95},
    'river_with_bridge': {'color': 'cyan', 'style': 'dashdot', 'width': 2.0, 'alpha': 0.95},
    'impassable_mountain_pass': {'color': 'maroon', 'style': 'dashed', 'width': 2.0, 'alpha': 0.95},
}

RIDGE_LINESTYLE = 'solid'
CONNECTION_LINESTYLES = {
    'normal': 'dashed',
    'mountain_pass': 'dashed',
    'river': 'dashed',
    'impassable': 'dashdot',
    'road': 'dotted',
    'river_with_bridge': 'dashdot',
    'impassable_mountain_pass': 'dotted',
}


def draw_center_connections(ax, graph, pos):
    """Overlay node-to-node border connections colored and styled by border terrain."""
    drawn_any = False
    for border_terrain, style in BORDER_STYLES.items():
        edgelist = [edge for edge in graph.edges if graph.edges[edge].get('terrain') == border_terrain]
        if not edgelist:
            continue

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=edgelist,
            edge_color=style['color'],
            style=CONNECTION_LINESTYLES.get(border_terrain, 'dashed'),
            width=style['width'] * 0.5,
            alpha=style['alpha'],
        )
        drawn_any = True

    if drawn_any:
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_size=18,
            node_color='black',
            edgecolors='white',
            linewidths=0.4,
        )


def draw_voronoi_ridge_connections(ax, graph, voronoi):
    """Draw graph borders on their corresponding finite Voronoi ridges."""
    point_to_index = {tuple(point): index for index, point in enumerate(voronoi.points)}
    edge_by_point_pair = {}
    for edge in graph.edges:
        u, v = edge
        try:
            edge_key = frozenset((point_to_index[u], point_to_index[v]))
        except KeyError:
            continue
        edge_by_point_pair[edge_key] = edge

    for (point_index_a, point_index_b), ridge_vertices in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        edge = edge_by_point_pair.get(frozenset((point_index_a, point_index_b)))
        if edge is None:
            continue
        if -1 in ridge_vertices or len(ridge_vertices) != 2:
            continue

        border_terrain = graph.edges[edge].get('terrain')
        style = BORDER_STYLES.get(border_terrain)
        if style is None:
            continue

        start_vertex = voronoi.vertices[ridge_vertices[0]]
        end_vertex = voronoi.vertices[ridge_vertices[1]]
        ax.plot(
            [start_vertex[0], end_vertex[0]],
            [start_vertex[1], end_vertex[1]],
            color=style['color'],
            linestyle=RIDGE_LINESTYLE,
            linewidth=style['width'],
            alpha=style['alpha'],
            solid_capstyle='round',
        )


def draw_terrain_connections(ax, graph, pos, voronoi, render_mode='both'):
    """Draw borders as center connections, Voronoi ridges, or both."""
    if render_mode not in {'center', 'ridge', 'both'}:
        raise ValueError(f"Unknown render_mode '{render_mode}'. Expected 'center', 'ridge', or 'both'.")

    if render_mode in {'center', 'both'}:
        draw_center_connections(ax, graph, pos)
    if render_mode in {'ridge', 'both'}:
        draw_voronoi_ridge_connections(ax, graph, voronoi)


def build_plot_legend_elements(render_mode='both'):
    province_elements = [
        Line2D([0], [0], marker='o', color='w', label=terrain, markerfacecolor=color, markersize=10)
        for terrain, color in PROVINCE_COLORS.items()
    ]
    border_elements = []
    border_labels = []
    border_handler_map = {}

    for terrain, style in BORDER_STYLES.items():
        ridge_handle = Line2D(
            [0], [0],
            color=style['color'],
            linestyle=RIDGE_LINESTYLE,
            linewidth=style['width'],
            alpha=style['alpha'],
        )
        connection_handle = Line2D(
            [0], [0],
            color=style['color'],
            linestyle=CONNECTION_LINESTYLES.get(terrain, 'dashed'),
            linewidth=style['width'] * 0.5,
            alpha=style['alpha'],
        )

        if render_mode == 'ridge':
            ridge_handle.set_label(terrain)
            border_elements.append(ridge_handle)
            border_labels.append(terrain)
        elif render_mode == 'center':
            connection_handle.set_label(terrain)
            border_elements.append(connection_handle)
            border_labels.append(terrain)
        else:
            combined_handle = (ridge_handle, connection_handle)
            border_elements.append(combined_handle)
            border_labels.append(terrain)
            border_handler_map[combined_handle] = HandlerTuple(ndivide=None)

    return province_elements, border_elements, border_labels, border_handler_map

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

    wfc_settings['debug_wfc_level'] = 2

    wfc = WaveFunctionCollapse(wfc_settings, graph_plain)
    # overwriting graph with new (TerrainGraph) with assignments
    graph = wfc.wave_function_collapse()

    pos = {node: node for node in graph.nodes}

    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)
    draw_terrain_connections(ax, graph, pos, voronoi, render_mode=BORDER_RENDER_MODE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'WFC Border Connections ({BORDER_RENDER_MODE})')

    fig, ax = plt.subplots(figsize=(11, 10))
    point2color = {node: PROVINCE_COLORS[graph.nodes[node]['terrain']] for node in graph.nodes}
    color_voronoi_faces(point2color)
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8)
    draw_terrain_connections(ax, graph, pos, voronoi, render_mode=BORDER_RENDER_MODE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.82, box.height]) # type: ignore
    province_legend, border_legend, border_labels, border_handler_map = build_plot_legend_elements(BORDER_RENDER_MODE)
    province_legend_artist = ax.legend(
        handles=province_legend,
        title='Province Terrains',
        bbox_to_anchor=(1.02, 0.98),
        loc='upper left',
    )
    ax.add_artist(province_legend_artist)
    ax.legend(
        handles=border_legend,
        labels=border_labels,
        title='Border Terrains',
        bbox_to_anchor=(1.02, 0.32),
        loc='upper left',
        handler_map=border_handler_map,
    )
    plt.title(f'WFC Terrain Assignment with Border Connections ({BORDER_RENDER_MODE})')
    plt.show()

if __name__ == "__main__":
    test_wfc()