import math

from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import blended_transform_factory

from FlowAtlas.populate_graph.wave_function_collapse import WaveFunctionCollapse
from FlowAtlas.populate_graph.rules_library import make_default_wfc_settings
from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph

# TODO consider what the actual default usage of WFC should look like, how should
# the user or other code setup/define rules and target distributions for it?

# Set to None to keep the previous behavior.
# Example:
# GLOBAL_DIST_DYNAMIC_ADJUSTMENT_SCHEDULE = {
#     'curve': 'linear',
#     'start_multiplier': 0.7,
#     'end_multiplier': 1.4,
# }
GLOBAL_DIST_DYNAMIC_ADJUSTMENT_SCHEDULE = None

# Current default rules and distributions live in rules_library.py.
wfc_settings = make_default_wfc_settings(
    global_dist_dynamic_adjustment_schedule=GLOBAL_DIST_DYNAMIC_ADJUSTMENT_SCHEDULE,
)
# Use initial per-class entropy baseline normalization when comparing provinces and borders.
wfc_settings['entropy_selection_mode'] = 'normalized_initial_mean'

# Border render mode options:
# - 'center': draw graph borders as center-to-center connections
# - 'ridge': draw graph borders on finite Voronoi ridges when available
# - 'both': draw both overlays at the same time
BORDER_RENDER_MODE = 'both'
# Checkpoint render mode options:
# - 'separate': one figure per sampled checkpoint
# - 'grid': subplot grid of sampled checkpoints
# - 'animation': replay sampled checkpoints as an animation, optionally saved to gif
# Can be a single mode string or multiple modes (list/tuple), e.g. ('grid', 'animation')
CHECKPOINT_RENDER_MODE = ('animation',)  # ('grid', 'animation')
CHECKPOINT_MAX_PLOTS = 9
CHECKPOINT_ANIMATION_INTERVAL_MS = 350
CHECKPOINT_ANIMATION_SAVE_PATH = None
CHECKPOINT_ANIMATION_REPEAT = True

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
    # Light steel blue: cool tone contrasts thermally against warm entropy scale and
    # gives chromatic contrast against warm terrains (yellow farm, plains, waste).
    'normal': {'color': "#B4C4DB", 'style': 'solid', 'width': 1.6, 'alpha': 0.95},
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

UNSET_PROVINCE_COLOR = '#d9d9d9'
# Warm grey scale: low entropy -> light warm grey, high entropy -> dark warm brown-grey.
# Warm tone separates the scale from the cool-steel normal borders and from neutral grey
# in general; absent from all terrain and all other border colors in the palette.
UNSET_ENTROPY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'unset_entropy', ["#E6DAD2", "#2B2019"]
)
# Marker colors for province / border min-max lines drawn on the entropy colorbar.
# Both are non-grey hues absent from all terrain and border palettes.
ENTROPY_PROVINCE_MARKER_COLOR = '#CC4400'  # burnt orange
ENTROPY_BORDER_MARKER_COLOR = '#6622CC'    # purple
UNSET_BORDER_STYLE = {
    'style': (0, (7, 1)),
    'width': 2.8,
    'alpha': 0.95,
    'zorder': 3,
}


def _lookup_edge_value(mapping, edge):
    if edge in mapping:
        return mapping[edge]
    reverse_edge = (edge[1], edge[0])
    if reverse_edge in mapping:
        return mapping[reverse_edge]
    return None


def _get_edge_visual(edge_visuals, graph, edge):
    if edge_visuals is not None:
        return _lookup_edge_value(edge_visuals, edge)

    border_terrain = graph.edges[edge].get('terrain')
    if border_terrain is None:
        return None
    style = BORDER_STYLES.get(border_terrain)
    if style is None:
        return None
    return {
        **style,
        'terrain': border_terrain,
    }


def draw_center_connections(ax, graph, pos, edge_visuals=None):
    """Overlay node-to-node border connections colored and styled by border terrain."""
    drawn_any = False
    for edge in graph.edges:
        style = _get_edge_visual(edge_visuals, graph, edge)
        if style is None:
            continue

        u, v = edge
        terrain_name = style.get('terrain')
        line_style = style.get('style', 'dashed')
        if isinstance(terrain_name, str):
            line_style = CONNECTION_LINESTYLES.get(terrain_name, line_style)
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=style['color'],
            linestyle=line_style,
            linewidth=style['width'] * 0.5,
            alpha=style['alpha'],
            zorder=style.get('zorder', 2),
        )
        drawn_any = True

    if drawn_any:
        xs = [pos[node][0] for node in graph.nodes]
        ys = [pos[node][1] for node in graph.nodes]
        ax.scatter(xs, ys, s=18, c='black', edgecolors='white', linewidths=0.4)


def draw_voronoi_ridge_connections(ax, graph, voronoi, edge_visuals=None):
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

        style = _get_edge_visual(edge_visuals, graph, edge)
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
            zorder=style.get('zorder', 2),
        )


def draw_terrain_connections(ax, graph, pos, voronoi, render_mode='both', edge_visuals=None):
    """Draw borders as center connections, Voronoi ridges, or both."""
    if render_mode not in {'center', 'ridge', 'both'}:
        raise ValueError(f"Unknown render_mode '{render_mode}'. Expected 'center', 'ridge', or 'both'.")

    if render_mode in {'center', 'both'}:
        draw_center_connections(ax, graph, pos, edge_visuals=edge_visuals)
    if render_mode in {'ridge', 'both'}:
        draw_voronoi_ridge_connections(ax, graph, voronoi, edge_visuals=edge_visuals)


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


def _max_entropy(domain):
    if len(domain) <= 1:
        return 1.0
    return math.log(len(domain))


def _build_entropy_normalizers(graph):
    terrain_domain = graph.global_metrics.get('terrain_domain', {})
    province_max = _max_entropy(terrain_domain.get('province_terrains', []))
    border_max = _max_entropy(terrain_domain.get('border_terrains', []))
    shared_max = max(province_max, border_max)
    return mcolors.Normalize(vmin=0.0, vmax=shared_max, clip=True)


def _entropy_to_grayscale(entropy, entropy_norm):
    # Direct mapping so the rendered color exactly matches the colorbar.
    normalized = entropy_norm(entropy)
    return UNSET_ENTROPY_CMAP(normalized)


def _entropy_minmax(values_dict):
    values = list(values_dict.values())
    if not values:
        return None, None
    return min(values), max(values)


def _colorbar_title_lines(cb, lines):
    """Write multi-line title above a colorbar with per-line colors.

    `lines` is a list of (text, color) ordered top-to-bottom.
    """
    line_step = 0.03
    base_y = 1.02 + (len(lines) - 1) * line_step
    for i, (text, color) in enumerate(lines):
        cb.ax.text(
            0.5,
            base_y - i * line_step,
            text,
            ha='center',
            va='bottom',
            fontsize=5.5,
            color=color,
            transform=cb.ax.transAxes,
            clip_on=False,
        )


def _add_entropy_colorbars(ax, entropy_norm, checkpoint_state):
    """Shared entropy colorbar with axhline markers for province/border min and max."""
    province_entropy = checkpoint_state.get('province_entropy', {})
    border_entropy = checkpoint_state.get('border_entropy', {})
    province_min, province_max = _entropy_minmax(province_entropy)
    border_min, border_max = _entropy_minmax(border_entropy)

    entropy_cax = ax.inset_axes([1.05, 0.12, 0.028, 0.78])

    entropy_map = ScalarMappable(norm=entropy_norm, cmap=UNSET_ENTROPY_CMAP)
    entropy_map.set_array([])

    entropy_cb = ax.figure.colorbar(entropy_map, cax=entropy_cax)
    entropy_cb.set_label('Unset entropy\nlight=low  dark=high', fontsize=7)
    entropy_cb.ax.tick_params(labelsize=7)
    _colorbar_title_lines(entropy_cb, [
        ('step range', '#777777'),
        (f' \u2014 p=provinces', ENTROPY_PROVINCE_MARKER_COLOR),
        (f'\u2014 b=borders  ', ENTROPY_BORDER_MARKER_COLOR),
    ])

    # Blended transform: x in axes fraction [0,1], y in entropy data coordinates.
    blend = blended_transform_factory(entropy_cb.ax.transAxes, entropy_cb.ax.transData)
    lf = 5.5

    if province_min is not None:
        entropy_cb.ax.axhline(province_min, color=ENTROPY_PROVINCE_MARKER_COLOR, linewidth=1.2)
        entropy_cb.ax.text(
            -0.15, province_min, f'p {province_min:.2f}',
            ha='right', va='center', fontsize=lf,
            color=ENTROPY_PROVINCE_MARKER_COLOR, transform=blend,
        )
    if province_max is not None:
        entropy_cb.ax.axhline(province_max, color=ENTROPY_PROVINCE_MARKER_COLOR, linewidth=1.2, linestyle='--')
        entropy_cb.ax.text(
            1.15, province_max, f'p {province_max:.2f}',
            ha='left', va='center', fontsize=lf,
            color=ENTROPY_PROVINCE_MARKER_COLOR, transform=blend,
        )
    if border_min is not None:
        entropy_cb.ax.axhline(border_min, color=ENTROPY_BORDER_MARKER_COLOR, linewidth=1.2)
        entropy_cb.ax.text(
            -0.15, border_min, f'b {border_min:.2f}',
            ha='right', va='center', fontsize=lf,
            color=ENTROPY_BORDER_MARKER_COLOR, transform=blend,
        )
    if border_max is not None:
        entropy_cb.ax.axhline(border_max, color=ENTROPY_BORDER_MARKER_COLOR, linewidth=1.2, linestyle='--')
        entropy_cb.ax.text(
            1.15, border_max, f'b {border_max:.2f}',
            ha='left', va='center', fontsize=lf,
            color=ENTROPY_BORDER_MARKER_COLOR, transform=blend,
        )


def _draw_colored_voronoi_faces(ax, point2color):
    original_points = list(point2color.keys())
    point_array = [list(point) for point in original_points]

    vor_points = np.array(point_array)
    min_x, min_y = np.min(vor_points, axis=0)
    max_x, max_y = np.max(vor_points, axis=0)
    x_range = max_x - min_x
    y_range = max_y - min_y
    extra_points = np.array([
        [min_x - x_range, min_y - y_range],
        [min_x - x_range, max_y + y_range],
        [max_x + x_range, min_y - y_range],
        [max_x + x_range, max_y + y_range],
    ])
    bounded_voronoi = Voronoi(np.vstack([vor_points, extra_points]))

    for index, point_coord in enumerate(original_points):
        region_id = bounded_voronoi.point_region[index]
        region = bounded_voronoi.regions[region_id]
        if -1 in region:
            continue
        polygon = [bounded_voronoi.vertices[i] for i in region]
        ax.fill(*zip(*polygon), color=point2color[point_coord])


def _make_checkpoint_province_colors(graph, checkpoint_state, entropy_norm):
    province_terrain = checkpoint_state.get('province_terrain', {})
    province_entropy = checkpoint_state.get('province_entropy', {})

    point2color = {}
    for node in graph.nodes:
        if node in province_terrain:
            terrain = province_terrain[node]
            point2color[node] = PROVINCE_COLORS.get(terrain, UNSET_PROVINCE_COLOR)
            continue

        node_entropy = province_entropy.get(node)
        if node_entropy is None:
            point2color[node] = UNSET_PROVINCE_COLOR
            continue

        point2color[node] = _entropy_to_grayscale(node_entropy, entropy_norm)

    return point2color


def _make_checkpoint_edge_visuals(graph, checkpoint_state, entropy_norm):
    border_terrain = checkpoint_state.get('border_terrain', {})
    border_entropy = checkpoint_state.get('border_entropy', {})
    edge_visuals = {}

    for edge in graph.edges:
        terrain = _lookup_edge_value(border_terrain, edge)
        if terrain is not None:
            style = BORDER_STYLES.get(terrain)
            if style is not None:
                edge_visuals[edge] = {
                    **style,
                    'terrain': terrain,
                }
            continue

        entropy = _lookup_edge_value(border_entropy, edge)
        if entropy is None:
            continue

        edge_visuals[edge] = {
            'color': _entropy_to_grayscale(entropy, entropy_norm),
            'style': UNSET_BORDER_STYLE['style'],
            'width': UNSET_BORDER_STYLE['width'],
            'alpha': UNSET_BORDER_STYLE['alpha'],
            'zorder': UNSET_BORDER_STYLE['zorder'],
        }

    return edge_visuals


def _sample_checkpoint_states(checkpoint_states, max_plots):
    if len(checkpoint_states) <= max_plots:
        return list(checkpoint_states)

    if max_plots <= 1:
        return [checkpoint_states[-1]]

    last_index = len(checkpoint_states) - 1
    indices = sorted({round(i * last_index / (max_plots - 1)) for i in range(max_plots)})
    return [checkpoint_states[index] for index in indices]


def _add_checkpoint_figure_note(fig):
    fig.text(
        0.5,
        0.015,
        "Set: terrain colors | Unset: light grey = low entropy, dark grey = high entropy (shared scale). Selection: lowest entropy first (ties random).",
        ha='center',
        va='bottom',
        fontsize=8,
    )


def _render_checkpoint_axis(ax, graph, voronoi, checkpoint_state, entropy_norm):
    pos = {node: node for node in graph.nodes}
    point2color = _make_checkpoint_province_colors(graph, checkpoint_state, entropy_norm)
    edge_visuals = _make_checkpoint_edge_visuals(graph, checkpoint_state, entropy_norm)

    _draw_colored_voronoi_faces(ax, point2color)
    voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False, line_colors='orange', line_width=1.0, line_alpha=0.65)
    draw_terrain_connections(ax, graph, pos, voronoi, render_mode=BORDER_RENDER_MODE, edge_visuals=edge_visuals)

    step = checkpoint_state.get('step', '?')
    set_provinces = len(checkpoint_state.get('province_terrain', {}))
    total_provinces = graph.number_of_nodes()
    set_borders = len(checkpoint_state.get('border_terrain', {}))
    total_borders = graph.number_of_edges()
    ax.set_title(
        f'WFC step={step} | provinces={set_provinces}/{total_provinces} | borders={set_borders}/{total_borders}'
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    _add_entropy_colorbars(ax, entropy_norm, checkpoint_state)


def _normalize_render_modes(render_mode):
    if isinstance(render_mode, str):
        return [render_mode]
    return [mode for mode in render_mode]


def render_checkpoint_maps(
    graph,
    voronoi,
    checkpoint_states,
    max_plots=6,
    render_mode: str | tuple[str, ...] | list[str] = 'grid',
    animation_interval_ms=350,
    animation_save_path=None,
):
    """Render checkpoint states as separate figures, a grid, or an animation."""
    if not checkpoint_states:
        return None

    sampled = _sample_checkpoint_states(checkpoint_states, max_plots)
    entropy_norm = _build_entropy_normalizers(graph)
    animations = []
    render_modes = _normalize_render_modes(render_mode)

    for mode in render_modes:
        if mode == 'separate':
            for checkpoint in sampled:
                fig, ax = plt.subplots(figsize=(10, 9))
                fig.subplots_adjust(bottom=0.08)
                _add_checkpoint_figure_note(fig)
                _render_checkpoint_axis(ax, graph, voronoi, checkpoint, entropy_norm)
            continue

        if mode == 'grid':
            num_plots = len(sampled)
            num_cols = min(3, num_plots)
            num_rows = math.ceil(num_plots / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
            fig.subplots_adjust(bottom=0.08)
            _add_checkpoint_figure_note(fig)
            axes_list = list(axes.flat) if hasattr(axes, 'flat') else [axes]

            for ax, checkpoint in zip(axes_list, sampled):
                _render_checkpoint_axis(ax, graph, voronoi, checkpoint, entropy_norm)

            for ax in axes_list[len(sampled):]:
                ax.axis('off')

            fig.suptitle('WFC Checkpoint Sequence', fontsize=16)
            fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
            continue

        if mode == 'animation':
            fig, ax = plt.subplots(figsize=(10, 9))
            fig.subplots_adjust(bottom=0.08)
            _add_checkpoint_figure_note(fig)

            def _update(frame_index):
                ax.clear()
                checkpoint = sampled[frame_index]
                _render_checkpoint_axis(ax, graph, voronoi, checkpoint, entropy_norm)
                return []

            animation = FuncAnimation(
                fig,
                _update,
                frames=len(sampled),
                interval=animation_interval_ms,
                repeat=CHECKPOINT_ANIMATION_REPEAT,
            )
            animations.append(animation)
            if animation_save_path:
                animation.save(animation_save_path, writer=PillowWriter(fps=max(1, round(1000 / animation_interval_ms))))
            continue

        raise ValueError(f"Unknown render_mode '{mode}'. Expected 'separate', 'grid', or 'animation'.")

    return animations if animations else None

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

    wfc_settings['debug_wfc_level'] = 3

    wfc = WaveFunctionCollapse(wfc_settings, graph_plain)
    # overwriting graph with new (TerrainGraph) with assignments
    graph = wfc.wave_function_collapse()
    checkpoint_states = wfc.get_debug_checkpoint_states()

    pos = {node: node for node in graph.nodes}

    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)
    draw_terrain_connections(ax, graph, pos, voronoi, render_mode=BORDER_RENDER_MODE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'WFC Border Connections ({BORDER_RENDER_MODE})')

    fig, ax = plt.subplots(figsize=(11, 10))
    point2color = {node: PROVINCE_COLORS[graph.nodes[node]['terrain']] for node in graph.nodes}
    _draw_colored_voronoi_faces(ax, point2color)
    voronoi_plot_2d(voronoi, ax=ax, show_points=False, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8)
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

    checkpoint_animation = render_checkpoint_maps(
        graph,
        voronoi,
        checkpoint_states,
        max_plots=CHECKPOINT_MAX_PLOTS,
        render_mode=CHECKPOINT_RENDER_MODE,
        animation_interval_ms=CHECKPOINT_ANIMATION_INTERVAL_MS,
        animation_save_path=CHECKPOINT_ANIMATION_SAVE_PATH,
    )
    plt.show()

if __name__ == "__main__":
    test_wfc()