def run():
    mapObject
    setup(mapObject) #Set up objects and constants, load any necessary resources
    generate_nodes(mapObject) #Place province capitals
    generate_connections(mapObject)
    generate_provinces(mapObject) #Generate province borders, determine possible start locations/throne sites ?
    generate_province_terrain(mapObject) #use wavefunction collapse to determine terrain for all provinces
    generate_border_terrain(mapObject)
    finalise_graph(mapObject) #I suspect this will be needed, but not really sure what should happen here yet - probably balancing
    place_special_sites(mapObject)
    generate_graphics(mapObject)
    export_map(mapObject)

