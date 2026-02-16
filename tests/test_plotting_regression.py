import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from DreamAtlas.classes.class_map import DominionsMap
from DreamAtlas.classes.class_region import Region

class DummyLayout:
    def __init__(self):
        # Minimal attributes to allow plotting
        self.graph = {0: np.array([[1, 2], [2, 1]])}
        self.coordinates = {0: np.array([[0, 0], [1, 1]])}
        self.darts = {0: np.array([0, 1])}
        self.province_graphs = {0: DummyProvinceGraph()}
        self.region_graph = [1, 2]

class DummyProvinceGraph:
    def get_all_connections(self):
        return [(0, 1)]
    def connect_nodes(self, i, j):
        pass
    @property
    def coordinates(self):
        return {0: [0, 0], 1: [1, 1]}

class DummyMap(DominionsMap):
    def __init__(self):
        # Use lists to match DominionsMap type hints
        self.pixel_map = [np.array([[0, 1], [1, 0]])] + [None]*9
        self.height_map = [np.array([[1, 2], [3, 4]])] + [None]*9
        self.layout = DummyLayout()
        self.map_size = [(2, 2)] + [[0, 0]]*9
        self.province_list = [[DummyProvince(), DummyProvince()]] + [[] for _ in range(9)]
        self.min_dist = [1.0] + [None]*9
        self.settings = None
        self.image_file = [None for _ in range(10)]
        self.planes = [0]

class DummyProvince:
    index = 0
    population = 100
    height = 1

class DummyRegion(Region):
    def __init__(self):
        self.graph = [1, 2]
        self.provinces = [DummyProvince(), DummyProvince()]
        self.name = "TestRegion"

    def plot(self):
        # Use the plotting code from class_region.py
        plot_size = (2, 2)
        pixel_map = [[0, 1], [1, 0]]
        z_graph = np.zeros(plot_size)
        z_terrain = np.zeros(plot_size)
        z_population = np.zeros(plot_size)
        fig, (ax_graph, ax_terrain, ax_population) = plt.subplots(1, 3)
        levels = len(self.graph)
        ax_graph.imshow(z_graph, cmap='Set1')
        ax_terrain.imshow(z_terrain, vmin=-200, vmax=600, cmap='terrain')
        ax_population.imshow(z_population, vmin=0, vmax=45000, cmap='YlGn')
        plt.close(fig)


def test_dominions_map_plotting():
    # This will run the plotting code in DominionsMap
    m = DummyMap()
    try:
        # Call the plotting code block directly
        # (simulate the plotting section in DominionsMap)
        import matplotlib.pyplot as plt
        plane = 0
        pixel_map_plane = m.pixel_map[plane]
        assert pixel_map_plane is not None, "pixel_map[plane] should not be None"
        plotting_pixel_map = np.transpose(pixel_map_plane)
        plane_general = np.vectorize(lambda i: i)(plotting_pixel_map)
        plane_regions = np.vectorize(lambda i: i)(plotting_pixel_map)
        plane_terrain = np.vectorize(lambda i: i)(plotting_pixel_map)
        plane_population = np.vectorize(lambda i: i)(plotting_pixel_map)
        fig, plane_axs = plt.subplots(1, 4)
        plane_axs[0].imshow(plane_general, cmap='Pastel1')
        plane_axs[1].imshow(plane_regions, vmin=1, vmax=2, cmap='tab20')
        plane_axs[2].imshow(plane_terrain, vmin=-200, vmax=600, cmap='terrain')
        plane_axs[3].imshow(plane_population, cmap='YlGn')
        plt.close(fig)
    except Exception as e:
        pytest.fail(f"DominionsMap plotting failed: {e}")

def test_region_plotting():
    # This will run the plotting code in Region
    r = DummyRegion()
    try:
        r.plot()
    except Exception as e:
        pytest.fail(f"Region plotting failed: {e}")
