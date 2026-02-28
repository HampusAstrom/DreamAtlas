import matplotlib.pyplot as plt
import networkx as nx

from DreamAtlas.flow_atlas.flow_types import FlowSettings
from DreamAtlas.flow_atlas.map import FlowMap
from DreamAtlas.flow_atlas.graph_generation import generate_nodes

def plot_graph(graph: nx.Graph):
    pos = {node: (node.x, node.y) for node in graph.nodes}
    nx.draw(graph, pos, node_size=30, node_color='lightblue', edge_color='gray')
    plt.show()

if __name__ == "__main__":
    mapObject = FlowMap(FlowSettings())
    generate_nodes(mapObject)
    G = mapObject.region_graphs[0]
    plot_graph(G)