import pytest
import networkx as nx
from DreamAtlas.flow_atlas.flow_types import MapNode

def graph_generation():
    test_graph = nx.Graph()
    assert(test_graph)

    node1 = MapNode()
    test_graph.add_node(node1)

    node_list = [MapNode() for _ in range(10)]
    test_graph.add_nodes_from(node_list)

    test_graph.add_edge(node1, node_list[1])

    edge_list = [ ( node1, node_list[i]) for i in range(10)]

    test_graph.add_edges_from(edge_list)

    assert(test_graph.degree[node1] is 10)
    for node in node_list:
        assert(test_graph.degree[node] is 1)
        assert(test_graph.adj[node] is node1)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
