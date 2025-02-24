import numpy as np

from . import *


def less_first(a, b):
    return [a, b] if a < b else [b, a]


def make_delaunay_graph(province_list: list[Province],
                        map_size: np.array,
                        neighbours: list = NEIGHBOURS_FULL):

    graph, coordinates, darts = dict(), dict(), dict()
    for province in province_list:  # Set up the dicts and assign coordinates
        index = province.index
        graph[index], coordinates[index], darts[index] = list(), np.asarray(province.coordinates, dtype=np.float32), list()

    points, key_list, counter = list(), dict(), 0
    for province in province_list:  # Set up the virtual points on the toroidal plane
        for n in neighbours:
            x, y = np.asarray(province.coordinates + n * map_size)
            points.append([x, y])
            key_list[counter] = province.index
            counter += 1

    tri = sc.spatial.Delaunay(np.array(points), qhull_options='QJ')

    list_of_edges = list()
    for triangle in tri.simplices:
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:  # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1], triangle[e2]))  # always lesser index first

    for p1, p2 in np.unique(list_of_edges, axis=0):  # remove duplicates
        index1, index2 = key_list[p1], key_list[p2]
        i_c = tri.points[p1]
        if (0 <= i_c[0] < map_size[0]) and (0 <= i_c[1] < map_size[1]):  # only do this for nodes in the map
            graph[index1].append(index2)
            graph[index2].append(index1)

            j_c = tri.points[p2]

            dart = [0, 0]
            for axis in range(2):
                if j_c[axis] < 0:
                    dart[axis] = -1
                elif j_c[axis] >= map_size[axis]:
                    dart[axis] = 1

            darts[index1].append(dart)
            darts[index2].append([-dart[0], -dart[1]])

    return graph, coordinates, darts


class DominionsLayout:

    def __init__(self, map_class):  # This class handles layout

        self.map = map_class
        self.seed = map_class.seed
        self.map_size = map_class.map_size

        # Region level layout - supersedes planes
        self.region_planes = None
        self.region_types = None
        self.region_graph = None

        # Province level layout - list per plane
        self.graph = [dict() for _ in range(10)]
        self.coordinates = [dict() for _ in range(10)]
        self.darts = [dict() for _ in range(10)]
        self.edge_types = [list() for _ in range(10)]
        self.neighbours = [list() for _ in range(10)]
        self.special_neighbours = [list() for _ in range(10)]
        self.gates = [list() for _ in range(10)]
        self.min_dist = [np.inf for _ in range(10)]

    def generate_region_layout(self,
                               settings: DreamAtlasSettings,
                               map_size: np.array,
                               seed: int = None):
        dibber(self, seed)  # Setting random seed

        player_team_plane = list()
        for nation_data in settings.vanilla_nations:
            nation = Nation(nation_data)
            player_team_plane.append([nation.index, nation.team, nation.home_plane])
        for custom_nation_data in settings.custom_nations:
            nation = CustomNation(custom_nation_data)
            player_team_plane.append([nation.index, nation.team, nation.home_plane])
        for generic_nation_data in settings.generic_nations:
            nation = GenericNation(generic_nation_data)
            player_team_plane.append([0, nation.team, nation.home_plane])

        teams = dict()
        for player, team, plane in player_team_plane:  # Analyse player teams
            if team not in teams:
                teams[team] = [player]
            else:
                teams[team].append(player)

        homeland_region_num = len(player_team_plane)
        periphery_region_num = int(0.5 * settings.player_neighbours * homeland_region_num)
        num_regions = homeland_region_num + periphery_region_num + settings.throne_region_num + settings.water_region_num + settings.cave_region_num + settings.vast_region_num

        region_graph = DreamAtlasGraph(size=num_regions, map_size=map_size)
        initial_graph = copy(rd.choice(DATASET_GRAPHS[len(player_team_plane)][settings.player_neighbours]))  # Select an initial layout

        nation_dict = dict()
        planes_dict = dict()
        types_dict = dict()
        r = 0

        # Add Homeland regions
        for i in range(homeland_region_num):
            for j in initial_graph[i+1]:
                region_graph.connect_nodes(i, j-1)

            nation_dict[r] = player_team_plane[i][0]
            planes_dict[r] = player_team_plane[i][2]
            types_dict[r] = 0
            r += 1

        weights = dict()
        for i in range(len(player_team_plane)):
            weights[i] = 1

        region_graph.embed_graph(initial_graph, seed)
        region_graph.plot()
        region_graph.spring_adjustment()
        region_graph.plot()

        # Add Peripheral regions
        done_edges = set()
        for i, j in region_graph.get_all_connections():
            if (j, i) not in done_edges:
                done_edges.add((i, j))
                region_graph.insert_node(i, j, r)

                nation_dict[r] = [nation_dict[i], nation_dict[j]]
                planes_dict[r] = 1
                types_dict[r] = 1
                r += 1

        # Add Throne regions
        faces, centroids = region_graph.get_faces_centroids()
        region_graph.plot()
        codebook, distortion = sc.cluster.vq.kmeans(obs=np.array(centroids, dtype=np.float32), k_or_guess=settings.throne_region_num)

        for coordinate in codebook:
            closest_distance = np.inf
            for i, centroid in enumerate(centroids):
                distance = np.linalg.norm(np.subtract(centroid, coordinate))
                if distance < closest_distance:
                    best_centroid = centroid
                    closest_distance = distance
                    face = faces[i]
            centroids.remove(best_centroid)

            for j in face:
                region_graph.connect_nodes(r, j)
            region_graph.coordinates[r] = best_centroid
            planes_dict[r] = 1
            types_dict[r] = 2
            r += 1

        region_types = [
            (settings.water_region_num, 3, 'water'),
            (settings.cave_region_num, 4, 'cave'),
            (settings.vast_region_num, 5, 'vast')
        ]

        for region_num, region_type, region_name in region_types:
            region_locations = rd.sample(centroids, region_num)
            for i in range(region_num):
                for j in face:
                    region_graph.connect_nodes(r, j)
                region_graph.coordinates[r] = region_locations[i]
                planes_dict[r] = 1 if region_name != 'cave' else 2
                types_dict[r] = region_type
                r += 1

        # Add Blocker regions - blocker regions go between cave regions, then into non-triangular faces and then fill out

        # Final Adjustment
        coordinates, darts = spring_electron_adjustment(region_graph.graph, coordinates, darts, weights, map_size, ratios=np.asarray((0.1, 0.2, 30), dtype=np.float32), iterations=0)

        self.region_planes = planes_dict
        self.region_types = types_dict
        self.region_graph = region_graph

    def generate_province_layout(self,
                                 plane: int,
                                 seed: int = None):
        dibber(self, seed)  # Setting random seed

        province_list = self.map.province_list[plane]
        map_size = self.map.map_size[plane]
        base_length = 0.15 * np.sqrt(map_size[0] * map_size[1] / len(province_list))

        weights = dict()
        for i, province in enumerate(province_list):
            weights[province.index] = province.size

        graph, coordinates, darts = make_delaunay_graph(province_list, map_size)
        coordinates, darts = spring_electron_adjustment(graph, coordinates, darts, weights, map_size, ratios=np.asarray((0.25, 0.9, base_length), dtype=np.float32), iterations=1000)

        self.graph[plane] = graph
        self.coordinates[plane] = coordinates
        self.darts[plane] = darts

    def generate_neighbours(self, plane: int):

        done_provinces = set()
        self.neighbours[plane] = list()
        self.min_dist[plane] = np.inf

        for i in self.graph[plane]:  # Assigning all graph edges as neighbours
            done_provinces.add(i)
            for ii, j in enumerate(self.graph[plane][i]):
                if j not in done_provinces:
                    self.neighbours[plane].append([i, j])
                    dist = np.linalg.norm(self.coordinates[plane][j] + np.multiply(self.darts[plane][i][ii], self.map.map_size[plane]) - self.coordinates[plane][i])
                    if dist < self.min_dist[plane]:
                        self.min_dist[plane] = dist

    def generate_special_neighbours(self,
                                    plane: int,
                                    seed: int = None):
        dibber(self, seed)  # Setting random seed

        if not self.neighbours[plane]:
            raise Exception('No neighbours')
        self.special_neighbours[plane] = list()
        index_2_prov = dict()
        for province in self.map.province_list[plane]:
            index_2_prov[province.index] = province

        for i, j in self.neighbours[plane]:  # Randomly assigns special connections (will be improved in v1.1)
            i_j_provs = [index_2_prov[i], index_2_prov[j]]
            choice = int(rd.choices(SPECIAL_NEIGHBOUR, NEIGHBOUR_SPECIAL_WEIGHTS)[0][0])
            if choice != 0:
                fail = False
                for index in range(2):
                    ti = i_j_provs[index].terrain_int
                    if i_j_provs[index].capital_location:  # Ignore caps
                        fail = True
                    elif has_terrain(ti, 4):
                        fail = True
                    elif has_terrain(ti, 4096):
                        fail = True
                    elif has_terrain(ti, 68719476736):  # if cave wall
                        self.special_neighbours[plane].append([i, j, 4])
                        fail = True
                    elif (choice == 33 or choice == 36) and not has_terrain(ti, 8388608):
                        i_j_provs[index].terrain_int += 8388608
                if not fail:
                    self.special_neighbours[plane].append([i, j, choice])

    def generate_gates(self, all_regions, seed: int = None):
        dibber(self, seed)  # Setting random seed
        self.gates = [[] for _ in range(10)]
        gate = 1
        for ii, i in enumerate(self.region_graph):
            i_region = all_regions[ii]
            if self.region_planes[i] == 2 and self.region_types[i] != 6:
                gate_connections = list()
                for j in self.region_graph[i]:
                    if self.region_planes[j] == 1:
                        gate_connections.append(j)

                if self.region_types[i] == 0:
                    gate_connections = gate_connections[0:3]

                for iii, j in enumerate(gate_connections):
                    i_province = i_region.provinces[iii]
                    j_province = rd.choice(all_regions[j].provinces)

                    self.gates[1].append([j_province.index, gate])
                    self.gates[2].append([i_province.index, gate])
                    gate += 1

    def plot(self):

        fig, axs = plt.subplots(3, 1)
        ax_regions = axs[0]
        ax_provinces = axs[1:]

        # Plot regions
        virtual_graph, virtual_coordinates = self.region_graph.get_virtual_graph()
        done_edges = set()
        for i in virtual_graph:  # region connections
            x0, y0 = virtual_coordinates[i]
            for j in virtual_graph[i]:
                if (i, j) not in done_edges:
                    done_edges.add((j, i))
                    x1, y1 = virtual_coordinates[j]
                    colour = 'k-'
                    if i in self.region_types:
                        if self.region_types[i] == 4:
                            colour = 'k--'
                    ax_regions.plot([x0, x1], [y0, y1], colour)

        region_colours = ['g*', 'rD', 'y^', 'bo', 'rv', 'ms', 'kX']
        for i, (x0, y0) in enumerate(self.region_graph.coordinates):
            ax_regions.plot(x0, y0, region_colours[self.region_types[i]])
            ax_regions.text(x0, y0, str(i))
        ax_regions.set(xlim=(0, self.map_size[1][0]), ylim=(0, self.map_size[1][1]))

        # # Plot provinces
        # for i, plane in enumerate(self.map.planes):
        #     ax = ax_provinces[i]
        #     virtual_graph, virtual_coordinates = make_virtual_graph(self.graph[plane], self.coordinates[plane], self.darts[plane], self.map.map_size[plane])
        #     for j in virtual_graph:  # region connections
        #         x0, y0 = virtual_coordinates[j]
        #         for k in virtual_graph[j]:
        #             x1, y1 = virtual_coordinates[k]
        #             ax.plot([x0, x1], [y0, y1], 'k-')
        #
        #     for j in self.graph[plane]:
        #         x0, y0 = self.coordinates[plane][j]
        #         ax.plot(x0, y0, 'ro')
        #         ax.text(x0, y0, str(j))
        #     ax.set(xlim=(0, self.map_size[plane][0]), ylim=(0, self.map_size[plane][1]))

    def __str__(self):  # Printing the class returns this

        string = f'\nType - {type(self)}\n\n'
        for key in self.__dict__:
            string += f'{key} : {self.__dict__[key]}\n'

        return string
