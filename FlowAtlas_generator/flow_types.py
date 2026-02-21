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

class FlowSettings:
    num_players = 6
    num_prov_per_player = 15
    map_size_x = 30000
    map_size_y = 30000
    cap_connections = 6