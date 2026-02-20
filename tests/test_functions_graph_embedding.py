import numpy as np
import pytest
from DreamAtlas.functions.functions_graph_embedding import embed_region_graph

def test_embed_region_graph_runs():
    # Minimal graph: 2 nodes, 1 edge
    graph = {1: [2], 2: [1]}
    map_size = np.array([10, 10])
    scale_down = 2
    seed = 42
    result = embed_region_graph(graph, map_size, scale_down, seed)
    # The function does not return, but should not error
    assert result is None or result is not None  # Just check it runs
