from flow_atlas.map import FlowMap, setup
from flow_atlas.flow_types import FlowSettings
from flow_atlas.graph_generation import generate_nodes

def run():
    settings = FlowSettings()
    mapObject = FlowMap(settings)
    setup(mapObject) #Set up objects and constants, load any necessary resources
    generate_nodes(mapObject) #Place province capitals, generate edges
    # generate_connections(mapObject)
    # generate_provinces(mapObject) #Generate province borders, determine possible start locations/throne sites ?
    # generate_border_terrain(mapObject)
    # generate_province_terrain(mapObject) #use wavefunction collapse to determine terrain for all provinces
    # finalise_graph(mapObject) #I suspect this will be needed, but not really sure what should happen here yet - probably balancing
    # place_special_sites(mapObject)
    # generate_graphics(mapObject)
    # export_map(mapObject)
#10, 15 or 20 provinces per player ish
