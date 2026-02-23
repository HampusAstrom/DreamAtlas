from libpysal import weights
from libpysal.cg import voronoi_frames
import libpysal
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas
from DreamAtlas.functions.functions_lloyd import LloydRelaxation
from scipy.spatial import Voronoi, voronoi_plot_2d

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

def main():
    num_points = 100
    min_distance = 0.1
    points = gen_grid(num_points)
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
    lloyd = LloydRelaxation(points)
    for _ in range(2):
        lloyd.relax()
    points = lloyd.get_points()
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')

    # Plot Voronoi diagram using scipy.spatial.Voronoi
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1.2, line_alpha=0.8, point_size=2)
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='x', label='Points')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Voronoi Diagram')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()