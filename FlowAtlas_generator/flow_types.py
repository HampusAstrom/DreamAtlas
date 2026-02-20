class Connection:
    '''Used as edges in the mapgraph. Not sure if this is useful beyond what is already implemented for edges in the graph.
    If we only need getters/setters, then it's def not useful'''
    def set_clockwise_neighbor(self, other: 'Connection'):
        self.clockwise = other
    
    def set_counterclockwise_neighbor(self, other: 'Connection'):
        self.counterclockwise = other

class MapNode:
    '''Used as province capitals in the map graph'''
    def __init__(self):
        self.x = 0
        self.y = 0