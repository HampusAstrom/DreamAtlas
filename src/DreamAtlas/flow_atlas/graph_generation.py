import numpy as np

from DreamAtlas.flow_atlas.flow_types import MapNode, FlowGraph
from DreamAtlas.flow_atlas.map import FlowMap

def generate_nodes(mapObject : FlowMap):
    # Calculate derived values
    ave_area_per_prov = (mapObject.map_size[0] * mapObject.map_size[1]) / mapObject.num_prov
    ave_distance_between_prov = (ave_area_per_prov ** 0.5)
    # (Relative) Width of the disc around old nodes where new nodes may be generated
    rel_radius_width = 0.6

    region_list = list()
    for i in range(mapObject.num_regions):
        region_graph = FlowGraph()
        region_center = MapNode()
        region_graph.add_node(region_center)

        max_iterations = 100

        def is_valid_node(node, graph):
            for existing_node in graph.nodes:
                distance = ((node.x - existing_node.x) ** 2 + (node.y - existing_node.y) ** 2) ** 0.5
                if distance < ave_distance_between_prov * (1 - rel_radius_width/2):
                    return False
            return True

        for j in range(mapObject.settings.cap_connections):
            candidate_node = None
            for n in range(max_iterations):
                candidate_node = region_center.create_neighbor(ave_distance_between_prov, rel_radius_width)
                if is_valid_node(candidate_node, region_graph):
                    break
            assert(candidate_node is not None), "Failed to generate a valid node after {} iterations".format(max_iterations)
            region_graph.add_node(candidate_node)

        for k in range(mapObject.settings.cap_connections, mapObject.num_prov_per_region):
            random_border_node = np.random.choice(list(region_graph.nodes))
            node_candidate = None
            for n in range(max_iterations):
                node_candidate = random_border_node.create_neighbor(ave_distance_between_prov, rel_radius_width)
                if is_valid_node(node_candidate, region_graph):
                    break
            assert(node_candidate is not None), "Failed to generate a valid node after {} iterations".format(max_iterations)
            region_graph.add_node(node_candidate)
        region_list.append(region_graph)
    mapObject.region_graphs = region_list


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
