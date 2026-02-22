import networkx as nx
import numpy as np

from flow_atlas.flow_types import MapNode
from map import FlowMap

def generate_nodes(mapObject : FlowMap):
    # Calculate derived values
    ave_area_per_prov = (mapObject.map_size[0] * mapObject.map_size[1]) / mapObject.num_prov
    ave_distance_between_prov = (ave_area_per_prov ** 0.5)

    region_list = list()
    for i in range(mapObject.num_regions):
        region_graph = nx.Graph()
        region_graph.add_node(MapNode())
        for j in range(mapObject.settings.cap_connections):
            # Generate a random point within on a distance of 0.7 to 1.3 times the average distance between provinces, with a random angle
            radius = (np.sqrt(np.random.uniform(0, 1))*0.6 + 0.7) * ave_distance_between_prov
            angle = np.random.uniform(0, 2 * np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            region_graph.add_node(MapNode(x, y))
            region_graph.add_edge(0, j)
        for k in range(mapObject.settings.cap_connections, mapObject.num_prov_per_region):
            # all existing nodes are within distance 1.3 of the capital)
            radius = (np.sqrt(np.random.uniform(0, 1))*0.6 + 1.4) * ave_distance_between_prov
            angle = np.random.uniform(0, 2 * np.pi)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            region_graph.add_node(MapNode(x, y))
        region_list.append(region_graph)

    # TODO: Generate the connections between the non-cap regions

    # some notes that I'm too tired to look at again:
    #   minimum distance for connection
    #   minimum number of layers to border from capital
    # Generate region graph
    # Connect regions
    # Place regions
    # Smooth the province map algorithmically
    # 6 players * 15 provinces per player = 90 provinces
    # 15 regions -> about six provinces per region
    # makes a total of 108 provinces,