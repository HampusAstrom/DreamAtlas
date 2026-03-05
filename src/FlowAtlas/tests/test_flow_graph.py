import unittest
import networkx as nx
from FlowAtlas.flow_types import ProvinceNode

class TestGraphGeneration(unittest.TestCase):

    def test_graph_generation(self):
        test_graph = nx.Graph()
        self.assertIsNotNone(test_graph)

        node1 = ProvinceNode()
        test_graph.add_node(node1)

        node_list = [ProvinceNode() for _ in range(10)]
        test_graph.add_nodes_from(node_list)

        test_graph.add_edge(node1, node_list[1])

        edge_list = [ ( node1, node_list[i]) for i in range(10)]

        test_graph.add_edges_from(edge_list)

        graph_degrees = test_graph.degree
        assert isinstance(graph_degrees, nx.classes.reportviews.DegreeView)
        node1_degree = graph_degrees[node1]
        self.assertIsInstance(node1_degree, int)
        self.assertEqual(node1_degree, 10)
        for node in node_list:
            node_degree = graph_degrees[node]
            self.assertEqual(node_degree, 1)
            self.assertEqual(test_graph.adj[node], {node1 : {}})

if __name__ == '__main__':
    unittest.main()
