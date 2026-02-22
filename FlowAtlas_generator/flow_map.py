from flow_types import FlowSettings

# TODO add inheritance from abstract parent of DominionsMap
class FlowMap:
    def __init__(self, settings : FlowSettings):
        self.settings = settings
        self.num_regions = settings.num_players + int(0.5 * settings.cap_connections * settings.num_players)
        self.num_prov = settings.num_players * settings.num_prov_per_player
        self.num_prov_per_region = int(self.num_prov / self.num_regions) if self.num_regions > 0 else 0
        self.map_size = (settings.map_size_x, settings.map_size_y)

def setup(mapObject : FlowMap):
    #Load any necessary resources, set up objects and constants
    pass