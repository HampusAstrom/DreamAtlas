import numpy as np
from DreamAtlas.functions.functions_virtual_graph import make_virtual_graph

def test_make_virtual_graph_basic():
    # Minimal graph: 2 nodes, 1 edge
    graph = {0: [1], 1: [0]}
    coordinates = {0: np.array([0, 0]), 1: np.array([1, 0])}
    darts = {0: [np.array([0, 0])], 1: [np.array([0, 0])]}
    mapsize = [10, 10]
    vg, vc = make_virtual_graph(graph, coordinates, darts, mapsize)
    assert isinstance(vg, dict)
    assert isinstance(vc, dict)
    assert 0 in vg and 1 in vg
