import numpy as np
import networkx as nx

class FlowGraph(nx.Graph):
    '''The graph structure for the map. Nodes are province capitals, edges are connections between provinces.'''
    def __init__(self):
        super().__init__()

    def add_node(self, node: 'MapNode'):
        super().add_node(node)
        self.update_edges(node)

    def update_edges(self, node: 'MapNode'):
        # TODO: implement this function
        super().add_edge(node, list(super().nodes)[0]) # Temporary, just connect to the first node for testing


class Connection:
    '''Used as edges in the mapgraph. Not sure if this is useful beyond what is already implemented for edges in the graph.
    If we only need getters/setters, then it's def not useful'''
    def set_clockwise_neighbor(self, other: 'Connection'):
        self.clockwise = other

    def set_counterclockwise_neighbor(self, other: 'Connection'):
        self.counterclockwise = other

class MapNode:
    '''Used as province capitals in the map graph'''
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y


    def create_neighbor(self, ave_radius, rel_width):
        # Generate a random point within distance of (radius * (1 \pm rel_width/2))
        assert(0 < rel_width < 1), "rel_width must be between 0 and 1"
        angle = np.random.uniform(0, 2 * np.pi)

        radius = (np.sqrt(np.random.uniform(0, 1))*rel_width - rel_width/2) * ave_radius

        x = self.x + radius * np.cos(angle)
        y = self.y + radius * np.sin(angle)
        return MapNode(x, y)

class FlowSettings:
    num_players = 6
    num_prov_per_player = 15
    map_size_x = 30000
    map_size_y = 30000
    cap_connections = 6