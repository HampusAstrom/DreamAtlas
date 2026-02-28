"""
Voronoi diagram generation script.
Can be run as: python try_voronoi_creation.py
Or as module: python -m FlowAtlas.try_voronoi_creation
"""

from libpysal import weights
from libpysal.cg import voronoi_frames
import libpysal
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas
from DreamAtlas.functions.functions_lloyd import LloydRelaxation
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d

def gen_grid(num_points):
    points = np.random.rand(num_points, 2)
    return points

def p2p_dist(point1, point2):
    # Calculate the Euclidean distance between two points
    # under assumtions of looping the map is a torus (i.e. looping around the edges)
    dx = abs(point1[0] - point2[0])
    dy = abs(point1[1] - point2[1])
    if dx > 0.5:
        dx = 1 - dx
    if dy > 0.5:
        dy = 1 - dy
    return np.sqrt(dx**2 + dy**2)

def spread_points(points):
    lloyd = LloydRelaxation(points)
    for _ in range(5):
        lloyd.relax()
    points = lloyd.get_points()
    return np.array(points)

def voronoi_and_graph(points: np.ndarray) -> tuple[nx.Graph, Voronoi]:
    """
    Convert a point set into a Voronoi diagram and its dual graph (Delaunay triangulation).

    The resulting graph has:
    - Nodes: the original input points
    - Edges: connecting adjacent Voronoi regions

    Args:
        points: np.ndarray of shape (n, 2) - the input points

    Returns:
        G: networkx.Graph where nodes are the original points
        vor: scipy.spatial.Voronoi object
    """

    # Create both Voronoi and Delaunay objects
    vor = Voronoi(points)
    tri = Delaunay(points)

    # Create graph with original points as nodes
    G = nx.Graph()

    # Add nodes with their coordinates
    for i, point in enumerate(points):
        G.add_node(tuple(point))

    # Add edges from Delaunay triangulation (dual of Voronoi)
    # Each simplex (triangle) in Delaunay gives us the neighbors
    for simplex in tri.simplices:
        # simplex contains indices of 3 points that form a triangle
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                p1 = tuple(points[simplex[i]])
                p2 = tuple(points[simplex[j]])
                if G.has_edge(p1, p2):
                    continue  # Edge already exists
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                G.add_edge(p1, p2, weight=distance)

    return G, vor

def main():
    num_points = 100
    min_distance = 0.1
    points = gen_grid(num_points)
    dpi = 100
    fig, axs = plt.subplots(1, 2, figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    axs[0].scatter(points[:, 0], points[:, 1], c='b', marker='.')

    # TODO consider if we can alter the relaxation to account for the looping
    # nature of the map, or if we can just add "ghost" points around the edges to simulate it
    # TODO see if we can change Voronoi/Lloyd to use full map field, ie
    # handle possibly adding vertices on the outside of the outermost points,
    # and handle distance calculations with looping in mind
    points = spread_points(points)
    axs[1].scatter(points[:, 0], points[:, 1], c='b', marker='.')

    fig, ax = plt.subplots(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    # Plot Voronoi diagram using scipy.spatial.Voronoi
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)
    # plt.scatter(points[:, 0], points[:, 1], c='red', marker='x', label='Points')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Voronoi Diagram')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()