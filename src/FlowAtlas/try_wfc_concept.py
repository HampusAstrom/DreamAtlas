from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from FlowAtlas.voronoi_utils import color_voronoi_faces

# terrain shorthands:
O = 'ocean/sea'
P = 'plain'
H = 'highland'
F = 'forest'
W = 'waste'
S = 'swamp'
R = 'farm/rich'
TERRAINS = [O, P, H, F, W, S, R]

BASE_PROB = [1.0/len(TERRAINS)] * len(TERRAINS) # the base probability for each terrain type

# a list of all forbidden triangles (independent of order)
tri_bans = []

# forbid all triangles with 3 of the same terrain type
for t in TERRAINS:
    tri_bans.append(tuple([t, t, t]))

# for t in TERRAINS: # forbid all triangles with 2 wastes and any other terrain type
#     tri_bans.append(tuple([W, W, t]))
# this doesn't work, if we add an "unassigned" placeholder later, we might be able to make it work

# make sure all bans are sorted tuples for easy comparison
tri_bans = [tuple(sorted(tri)) for tri in tri_bans]

def is_triangle_banned(tri):
    tri = tuple(sorted(tri))
    return tri in tri_bans

# takes list of triangels of nodes
def check_triangle_ban(graph, triangle):
    terrains = [graph.nodes[n].get('terrain', None) for n in triangle]
    if None not in terrains and is_triangle_banned(terrains):
        return True
    return False

def assign_bans_to_neighbors(graph, node):
    # get the terrain type of the node
    terrain = graph.nodes[node].get('terrain', None)
    if terrain is None:
        return

    # get all triangles with node
    triangles = nx.all_triangles(graph, node)
    # filter to get only triangles with exactly 1 unassigned node (the one we're assigning bans to)
    triangles = [tri for tri in triangles if sum(1 for n in tri if graph.nodes[n].get('terrain', None) is None) == 1]

    # for each triangle, check if assigning a certain terrain type to the unassigned
    # node would violate any bans, and if so add those terrain types to that node's ban list
    for tri in triangles:
        for terrain_type in TERRAINS:
            # get terrains of the triangle, with the unassigned node assigned to the current terrain type
            tri_terrains = [graph.nodes[n].get('terrain', None) for n in tri]
            for i in range(len(tri_terrains)):
                if tri_terrains[i] is None: # only try to assign to unassigned nodes
                    tri_terrains[i] = terrain_type
            if is_triangle_banned(tri_terrains):
                # if the triangle is banned, add the terrain type to the ban list of the unassigned node
                for i in range(len(tri_terrains)):
                    if tri_terrains[i] is None:
                        graph.nodes[tri[i]]['ban'] = graph.nodes[tri[i]].get('ban', set())
                        graph.nodes[tri[i]]['ban'].add(terrain_type)

def assign_terrains(graph: nx.Graph):
    # assign terrains to the graph using a simple constraint satisfaction approach

    # the current probability for each terrain type, which we will update as we assign terrains
    running_prob = BASE_PROB.copy()

    # check if any nodes are assigned yet, if they are:
    # 1. assign bans to their neighbors
    # 2. check that no triangles already violate the bans
    for node in graph.nodes:
        if 'terrain' in graph.nodes[node]:
            assign_bans_to_neighbors(graph, node)
    triangles = nx.all_triangles(graph)
    for tri in triangles:
        if check_triangle_ban(graph, tri):
            raise ValueError(f"Initial terrain assignment violates bans: {tri}")

    # check if any nodes are assigned yet, if not, assign one at random with the base probabilities
    if not any('terrain' in graph.nodes[n] for n in graph.nodes):
        nodes = list(graph.nodes)
        ind = np.random.choice(len(nodes))
        node = nodes[ind]

        terrain = np.random.choice(TERRAINS, p=running_prob)
        graph.nodes[node]['terrain'] = terrain

    # while there are still unassigned nodes, assign terrains to them
    unassigned_nodes = [n for n in graph.nodes if 'terrain' not in graph.nodes[n]]
    # sort unassigned nodes by the number of bans they have, so we assign the most constrained nodes first
    unassigned_nodes.sort(key=lambda n: len(graph.nodes[n].get('ban', set())), reverse=True)
    while unassigned_nodes:
        node = unassigned_nodes.pop(0)
        ban_list = graph.nodes[node].get('ban', set())
        # calculate the probability for each terrain type based on the ban list
        prob = [0 if t in ban_list else running_prob[i] for i, t in enumerate(TERRAINS)]
        # normalize the probabilities
        total_prob = sum(prob)
        if total_prob == 0:
            # if no valid, warn and assign a random terrain type (this shouldn't happen if the bans are consistent, but just in case)
            print(f"Warning: No valid terrain types for node {node} with ban list {ban_list}, assigning random terrain type")
            prob = running_prob.copy()
            total_prob = sum(prob)
        prob = [p / total_prob for p in prob]
        # assign a terrain type to the node based on the probabilities
        terrain = np.random.choice(TERRAINS, p=prob)
        graph.nodes[node]['terrain'] = terrain
        # assign bans to neighbors of the newly assigned node
        assign_bans_to_neighbors(graph, node)
        # update the running probabilities (for simplicity, we won't update them in this implementation, but we could if we wanted to)

        # re-sort the unassigned nodes by the number of bans they have, since assigning this node may have added new bans to its neighbors
        unassigned_nodes.sort(key=lambda n: len(graph.nodes[n].get('ban', set())), reverse=True)

    return graph

def main():
    num_points = 100
    # generate (empty) node graph
    points = gen_grid(num_points)
    points = spread_points(points)
    graph, voronoi = voronoi_and_graph(points)

    # assign terrains to the graph
    graph = assign_terrains(graph)

    # visualize the graph with assigned terrains
    terrain_colors = {O: 'blue', P: 'lightgreen', H: 'sienna', F: 'darkgreen', W: 'gray', S: 'olive', R: 'yellow'}
    node_colors = [terrain_colors[graph.nodes[n]['terrain']] for n in graph.nodes]
    pos = {n: n for n in graph.nodes} # use the node coordinates as positions for visualization

    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    point2color = {n: terrain_colors[graph.nodes[n]['terrain']] for n in graph.nodes}
    color_voronoi_faces(point2color)
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8)
    nx.draw(graph, pos=pos, ax=ax, node_color='red', with_labels=False, node_size=20, edge_color='white', alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('WFC Terrain Assignment on Voronoi Graph')
    plt.show()



if __name__ == "__main__":
    main()