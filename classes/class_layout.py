import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.cluster.vq as sccvq
from copy import copy

from .graph import DreamAtlasGraph
from .class_connection import Connection
from .class_settings import DreamAtlasSettings
from DreamAtlas.functions import dibber, has_terrain
from DreamAtlas.databases.dominions_data import SPECIAL_NEIGHBOUR
from DreamAtlas.databases.dreamatlas_data import (
    DATASET_GRAPHS, NEIGHBOUR_SPECIAL_WEIGHTS, REGION_CAVE_INFO
)


class DominionsLayout:

    def __init__(self, map_class):  # This class handles layout

        self.map = map_class
        self.seed = map_class.seed
        self.map_size = map_class.map_size
        self.wraparound = self.map.wraparound

        # Region level layout - supersedes planes
        self.region_planes: dict[int, int] | None = None
        self.region_types: dict[int, int] | None = None
        self.region_graph: DreamAtlasGraph | None = None

        # Province level layout - list per plane
        self.province_graphs: list[DreamAtlasGraph | None] = [None for _ in range(10)]
        self.edge_types = [list() for _ in range(10)]
        self.connections = [list() for _ in range(10)]
        self.gates = [list() for _ in range(10)]
        self.min_dist = [np.inf for _ in range(10)]

    def generate_region_layout(self,
                               settings: DreamAtlasSettings,
                               map_size: np.ndarray,
                               nation_list: list,
                               seed: int | None = None):
        dibber(self, seed)  # Setting random seed

        teams = dict()
        ug_starts = 0
        for nation in nation_list:  # Analyse player teams
            iid, team = nation.iid, nation.team
            if team not in teams:
                teams[team] = [iid]
            else:
                teams[team].append(iid)
            if nation.home_plane == 2:
                ug_starts += 1

        homeland_region_num = len(nation_list)
        periphery_region_num = int(0.5 * settings.player_neighbours * homeland_region_num)
        blockers = 5 * (settings.cave_region_num + ug_starts) + 50
        num_regions = homeland_region_num + periphery_region_num + settings.throne_region_num + settings.water_region_num + settings.cave_region_num + settings.vast_region_num + blockers

        self.region_graph = DreamAtlasGraph(size=num_regions, map_size=map_size, wraparound=self.wraparound)
        initial_graph = copy(rd.choice(DATASET_GRAPHS[len(nation_list)][settings.player_neighbours]))  # Select an initial layout

        self.region_planes = dict()
        self.region_types = dict()
        region_idx = 0

        # Add Homeland regions
        for i, nation in enumerate(nation_list):
            for j in initial_graph[i+1]:
                self.region_graph.connect_nodes(i, j-1)
            self.region_graph.planes[i] = nation.home_plane
            self.region_planes[region_idx] = nation.home_plane
            self.region_types[region_idx] = 0
            region_idx += 1

        weights = dict()
        for i in range(len(nation_list)):
            weights[i] = 1

        attempt = 0
        allowed_attempts = 30
        while attempt < allowed_attempts:
            self.region_graph.embed_graph(initial_graph, seed)
            if settings.disciples:
                self.region_graph.embed_disciples(teams, seed)  # Embed disciples into the graph

            failed = False
            for axis in range(2):
                nonzero_darts = np.transpose(np.nonzero(self.region_graph.darts[:, :, axis]))
                if len(nonzero_darts) == 0:
                    failed = True

            if not failed:
                break
            print(f"Embedding Error: Concave embedding with seed {seed} on attempt {attempt}")
            attempt += 1
            seed = rd.randint(0, 1000)

        if attempt == allowed_attempts:
            raise Exception("DreamAtlas Error: Failed to embed region graph after 30 attempts")
        elif attempt > 1:
            print(f"Successful embedding with seed {seed} on attempt {attempt}")

        self.region_graph.spring_adjustment(ratios=np.array((0.01, 0.8, 200), dtype=np.float32), iterations=3000)
        # self.plot()
        # plt.show()

        # Add Peripheral regions
        done_edges = set()
        for i, j in self.region_graph.get_all_connections():
            if (j, i) not in done_edges:
                done_edges.add((i, j))
                self.region_graph.insert_connection(i, j, region_idx)
                self.region_planes[region_idx] = 1
                self.region_types[region_idx] = 1
                region_idx += 1

        faces, centroids = self.region_graph.get_faces_centroids()

        region_types = [(settings.throne_region_num, 2, 'throne'),
                        (settings.water_region_num, 3, 'water'),
                        (settings.cave_region_num, 4, 'cave'),
                        (settings.vast_region_num, 5, 'vast')]

        extras = False
        for region_num, region_type, region_name in region_types:
            if region_num == 0:
                continue
            if region_num <= len(centroids):
                codebook, distortion = sccvq.kmeans(obs=np.array(centroids, dtype=np.float32), k_or_guess=region_num)
            else:
                codebook = centroids
                extras = True

            for coordinate in codebook:
                closest_distance = np.inf
                for j, centroid in enumerate(centroids):
                    distance = np.linalg.norm(np.subtract(centroid, coordinate))
                    if distance < closest_distance:
                        best_centroid = centroid
                        closest_distance = distance
                        face = faces[j]
                centroids.remove(best_centroid)
                faces.remove(face)

                self.region_graph.insert_face(face, region_idx, best_centroid)
                self.region_planes[region_idx] = 1 if region_name != 'cave' else 2
                self.region_graph.planes[region_idx] = self.region_planes[region_idx]
                self.region_types[region_idx] = region_type
                region_idx += 1

            if extras:
                extras = False
                faces, centroids = self.region_graph.get_faces_centroids()
                codebook, distortion = sccvq.kmeans(obs=np.array(centroids, dtype=np.float32), k_or_guess=region_num-len(codebook))

                for coordinate in codebook:
                    closest_distance = np.inf
                    for j, centroid in enumerate(centroids):
                        distance = np.linalg.norm(np.subtract(centroid, coordinate))
                        if distance < closest_distance:
                            best_centroid = centroid
                            closest_distance = distance
                            face = faces[j]
                    centroids.remove(best_centroid)
                    faces.remove(face)

                    self.region_graph.insert_face(face, region_idx, best_centroid)
                    self.region_planes[region_idx] = 1 if region_name != 'cave' else 2
                    self.region_graph.planes[region_idx] = self.region_planes[region_idx]
                    self.region_types[region_idx] = region_type
                    region_idx += 1

        # Add Blocker regions - mountain blocker regions go into non-triangular surface faces then cave walls between cave regions
        # faces, centroids = self.region_graph.get_faces_centroids(planes=[1])
        faces = set()
        for i, face in enumerate(faces):
            if len(face) > 3:
                self.region_graph.insert_face(face, region_idx, centroids[i])
                self.region_planes[region_idx] = 1
                self.region_graph.planes[region_idx] = self.region_planes[region_idx]
                self.region_types[region_idx] = 6
                region_idx += 1

        edges, coordinates, darts = self.region_graph.get_small_delaunay(planes=[2])
        for i, edge in enumerate(edges):
            self.region_graph.graph[edge[0], edge[1]] = 0
            self.region_graph.graph[edge[1], edge[0]] = 0
            self.region_graph.graph[edge[0], region_idx] = 1
            self.region_graph.graph[edge[1], region_idx] = 1
            self.region_graph.graph[region_idx, edge[1]] = 1
            self.region_graph.graph[region_idx, edge[0]] = 1
            self.region_graph.coordinates[region_idx] = coordinates[i]
            self.region_graph.darts[edge[0], region_idx] = -darts[i][0]
            self.region_graph.darts[edge[1], region_idx] = -darts[i][1]
            self.region_graph.darts[region_idx, edge[1]] = darts[i][1]
            self.region_graph.darts[region_idx, edge[0]] = darts[i][0]

            self.region_planes[region_idx] = 2
            self.region_graph.planes[region_idx] = self.region_planes[region_idx]
            self.region_types[region_idx] = 8
            region_idx += 1

        self.region_graph.lloyd_relaxation(iterations=3)
        self.region_graph.spring_adjustment(ratios=np.array((0.01, 0.8, 100), dtype=np.float32), iterations=3000)
        faces, centroids = self.region_graph.get_faces_centroids(planes=[2])
        for i, face in enumerate(faces):
            self.region_graph.insert_face(face, region_idx, centroids[i])
            self.region_planes[region_idx] = 2
            self.region_graph.planes[region_idx] = self.region_planes[region_idx]
            self.region_types[region_idx] = 8
            region_idx += 1
        self.region_graph.spring_adjustment(ratios=np.array((0.01, 0.8, 100), dtype=np.float32), iterations=3000)

    def generate_province_layout(self,
                                 province_list,
                                 plane: int,
                                 seed: int | None = None):
        """Initialize province graph for a specific plane. Must be called before generate_connections."""
        dibber(self, seed)  # Setting random seed

        graph = DreamAtlasGraph(len(province_list), map_size=self.map_size[plane], wraparound=self.wraparound)
        for i, province in enumerate(province_list):
            graph.coordinates[i] = province.coordinates

            if province.capital_location or province.capital_circle:
                graph.types[i] = 1
            # graph.weights[i] = province.size

        graph.make_delaunay_graph()
        graph.lloyd_relaxation(iterations=2)
        graph.spring_adjustment()
        graph.clean_delaunay_graph()  # Clean the graph with swaps for non-cap provinces
        # print(len(graph.get_all_connections()))

        self.province_graphs[plane] = graph

    def generate_connections(self,
                                    plane: int,
                                    seed: int | None = None):
        """Generate connections between provinces. Requires generate_province_layout() to be called first."""
        dibber(self, seed)  # Setting random seed

        graph = self.province_graphs[plane]
        assert graph is not None, f"generate_province_layout() must be called for plane {plane} first"

        region_types = self.region_types
        assert region_types is not None, "generate_region_layout() must be called first"

        index_2_prov = dict()
        for province in self.map.province_list[plane]:
            index_2_prov[province.index] = province

        self.connections[plane] = list()
        done_edges = set()
        for i, j in graph.get_all_connections():
            if (j, i) not in done_edges:
                done_edges.add((i, j))

                for province in [index_2_prov[i+1], index_2_prov[j+1]]:
                    choice = int(rd.choices(SPECIAL_NEIGHBOUR, NEIGHBOUR_SPECIAL_WEIGHTS)[0][0])
                    terrain = province.terrain_int
                    if province.capital_location:  # Ignore caps
                        choice = 0
                        break
                    elif has_terrain(terrain, 68719476736):  # if cave wall
                        choice = 4
                        break
                    elif region_types[province.parent_region.index] == 6:  # if blocker
                        choice = 36
                        break
                    elif has_terrain(terrain, 4):
                        choice = 0
                        break
                    elif has_terrain(terrain, 4096):
                        choice = 0
                        break

                for dart in graph.darts[i, j]:
                    if dart != 0 and choice != 4:
                        choice = 0
                        break

                for province in [index_2_prov[i+1], index_2_prov[j+1]]:
                    if (choice == 33 or choice == 36) and not has_terrain(province.terrain_int, 8388608):
                        province.terrain_int += 8388608

                self.connections[plane].append(Connection(connected_provinces={i+1, j+1}, connection_int=choice))

        self.min_dist[plane] = float(graph.get_min_dist())

    def generate_gates(self, region_list, seed: int | None = None):
        """Generate gates between regions. Requires generate_region_layout() to be called first."""
        region_types = self.region_types
        assert region_types is not None, "generate_region_layout() must be called first"

        region_graph = self.region_graph
        assert region_graph is not None, "generate_region_layout() must be called first"

        region_planes = self.region_planes
        assert region_planes is not None, "generate_region_layout() must be called first"

        dibber(self, seed)  # Setting random seed
        self.gates = [[] for _ in range(10)]
        gate = 1

        for region_type in [0, 4]:
            for i, i_region in enumerate(region_list[region_type]):

                if region_planes[i_region.index] == 2:
                    possible_j_regions = list()
                    for j, j_region in enumerate(region_list[1]):
                        if region_graph.graph[i_region.index, j_region.index]:
                            possible_j_regions.append(j_region)
                    rd.shuffle(possible_j_regions)

                    gates_num = REGION_CAVE_INFO[self.map.settings.cave_region_type][4]  # How many gates per region
                    start = 0
                    if region_type == 0:
                        gates_num = 3
                        start = 1

                    i_provinces = rd.sample(i_region.provinces[start:], k=gates_num)
                    for j, j_region in enumerate(possible_j_regions[0:gates_num]):
                        for province in j_region.provinces:
                            if not province.has_gate:
                                j_province = province
                        j_province.has_gate = True
                        self.gates[1].append([j_province.index, gate])
                        self.gates[2].append([i_provinces[j].index, gate])
                        gate += 1

    def plot(self):
        """Visualize the layout. Requires generate_region_layout() to be called first."""
        region_graph = self.region_graph
        assert region_graph is not None, "generate_region_layout() must be called first"

        region_types = self.region_types
        assert region_types is not None, "generate_region_layout() must be called first"

        region_planes = self.region_planes
        assert region_planes is not None, "generate_region_layout() must be called first"

        fix_reg, ax_reg = plt.subplots()

        # Plot regions
        virtual_graph, virtual_coordinates = region_graph.get_virtual_graph()

        for i, (x, y) in enumerate(virtual_coordinates):  # region connections
            if i in region_types:
                for j in np.argwhere(virtual_graph[i, :] == 1):
                    j = j[0]
                    x2, y2 = virtual_coordinates[j]
                    ax_reg.plot([x, x2], [y, y2], 'k-')

        region_colours = ['g*', 'rD', 'y^', 'bo', 'rv', 'ms', 'kX', 'kX', 'kX']
        for i, (x, y) in enumerate(virtual_coordinates):  # region connections
            if i in region_types:
                ax_reg.plot(x, y, region_colours[region_types[i]])
                ax_reg.text(x, y, str(i))

        ax_reg.set(xlim=(0, self.map_size[1][0]), ylim=(0, self.map_size[1][1]))

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