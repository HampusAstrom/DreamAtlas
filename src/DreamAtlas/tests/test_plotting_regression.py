import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from DreamAtlas.classes.class_map import DominionsMap
from DreamAtlas.classes.class_region import Region

# Pixel map access test for .tga path
def test_pixel_map_none_access_tga():
    from typing import Optional
    m = DominionsMap()
    m.planes = [1]
    m.terrain_list[1] = [(1, 0)]
    m.map_size[1] = [2, 2]
    m.pixel_owner_list[1] = [[0, 0, 1, 1]]
    m.population_list[1] = [(1, 100)]
    m.neighbour_list[1] = []
    m.province_capital_locations[1] = [(0, 0, 0)]
    m.height_map[1] = np.zeros((2, 2), dtype=int)
    m.pixel_map[1] = np.ones((2, 2), dtype=int)
    # Mock image_pil[1] to return an object with __getitem__ for pixel values
    class FakePixelAccess:
        def __getitem__(self, xy):
            x, y = xy
            # Return (255,255,255) for (0,0) and (1,1), else (0,0,0)
            if (x, y) in [(0, 0), (1, 1)]:
                return (255, 255, 255)
            else:
                return (0, 0, 0)
    class FakeImage:
        def load(self):
            return FakePixelAccess()
    m.image_pil[1] = FakeImage()
    plane_image_types: list[Optional[str]] = [None for _ in range(10)]
    plane_image_types[1] = '.tga'
    m.fill_dreamatlas(plane_image_types)
    assert m.pixel_map[1] is not None
    assert isinstance(m.pixel_map[1], np.ndarray)

class DummyLayout:
    def __init__(self):
        # Use a real DreamAtlasGraph for province_graphs to match DominionsMap.plot expectations
        from DreamAtlas.classes.graph import DreamAtlasGraph
        size = 3
        map_size = (3, 3)
        wraparound = [(0, 0)]
        dag = DreamAtlasGraph(size=size, map_size=map_size, wraparound=wraparound)
        # Connect nodes in a triangle for simplicity
        dag.graph[0, 1] = dag.graph[1, 0] = True
        dag.graph[1, 2] = dag.graph[2, 1] = True
        dag.graph[2, 0] = dag.graph[0, 2] = True
        dag.coordinates[0] = [0, 0]
        dag.coordinates[1] = [1, 0]
        dag.coordinates[2] = [0, 1]
        dag.darts[0, 1] = [1, 0]
        dag.darts[1, 2] = [0, 1]
        dag.darts[2, 0] = [-1, -1]
        self.province_graphs = [dag] + [None]*9
        # Use a simple region_graph with a .size attribute
        class DummyRegionGraph:
            size = 2
        self.region_graph = DummyRegionGraph()

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
        self.pixel_map = [np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])] + [None]*9
        self.height_map = [np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])] + [None]*9
        self.layout = DummyLayout()
        self.map_size = [(3, 3)] + [[0, 0]]*9
        self.province_list = [[DummyProvince(i) for i in range(3)]] + [[] for _ in range(9)]
        self.min_dist = [1.0] + [None]*9
        self.settings = None
        self.image_file = [None for _ in range(10)]
        self.planes = [0]

class DummyParentRegion:
    def __init__(self, index=0):
        self.index = index

class DummyProvince:
    def __init__(self, index=0, coordinates=None):
        self.index = index
        self.population = 100
        self.height = 1
        self.parent_region = DummyParentRegion(0)
        self.terrain_int = 0
        self.coordinates = coordinates if coordinates is not None else (index, index)

class DummyRegion(Region):
    def __init__(self):
        self.graph = {0: [1], 1: [0]}
        self.provinces = [
            DummyProvince(0, coordinates=(100, 100)),
            DummyProvince(1, coordinates=(400, 400))
        ]
        self.name = "TestRegion"

def test_dominions_map_plotting():
    # This will run the actual DominionsMap.plot function
    m = DummyMap()
    try:
        m.plot()  # Directly call the method
    except Exception as e:
        pytest.fail(f"DominionsMap.plot failed: {e}")

def test_region_plotting():
    # This will run the plotting code in Region
    r = DummyRegion()
    try:
        r.plot()
    except Exception as e:
        pytest.fail(f"Region plotting failed: {e}")