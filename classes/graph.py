import numpy as np
import minorminer as mnm
import networkx as ntx
import random as rd
from numba import njit, prange
from numba.experimental import jitclass
from scipy.special import y1_zeros


@njit(parallel=True)
def _numba_attractor_adjustment(graph: np.array,
                                coordinates: np.array,
                                darts: np.array,
                                attractor_array: np.array,
                                damping_ratio: float,
                                map_size: np.array,
                                iterations: int):

    dict_size = len(coordinates)
    net_velocity = np.zeros((dict_size, 2), dtype=np.float64)
    for _ in range(iterations):
        attractor_force = np.zeros((dict_size, 2), dtype=np.float64)
        for i in prange(dict_size):
            for j in range(dict_size):
                if attractor_array[i, j]:
                    attractor_force[i] += coordinates[j] + darts[i, j] * map_size - coordinates[i]

        net_velocity = damping_ratio * (net_velocity + attractor_force)

        equilibrium = 1
        for c in range(dict_size):  # Check if particles are within tolerance
            if np.linalg.norm(net_velocity[c]) > 0.001:
                equilibrium = 0
                break

        coordinates += net_velocity
        for a in range(dict_size):  # Update the position
            for axis in range(2):
                if not (0 <= coordinates[a, axis] < map_size[axis]):
                    dart_change = -np.sign(coordinates[a, axis])
                    # print(dart_change, coordinates[a, axis], coordinates[a, axis] % map_size[axis])
                    coordinates[a, axis] = coordinates[a, axis] % map_size[axis]

                    for b in range(dict_size):  # Iterating over all of this vertex's connections
                        if graph[a, b]:
                            new_value = darts[a, b, axis] + dart_change
                            if new_value < -1:
                                new_value = 1
                            if new_value > 1:
                                new_value = -1

                            darts[a, b, axis] = new_value  # Setting the dart for this vertex
                            darts[b, a, axis] = -new_value  # Setting the dart for other vertex
        if equilibrium:
            break

    return coordinates, darts


@njit(parallel=True)
def _numba_spring_adjustment(graph: np.array,
                             coordinates: np.array,
                             darts: np.array,
                             weight_array: np.array,
                             map_size: np.array,
                             ratios: np.array,
                             connections: np.array,
                             iterations: int):

    dict_size = len(coordinates)
    damping_ratio, spring_coefficient, base_length = ratios

    net_velocity = np.zeros((dict_size, 2), dtype=np.float32)
    for _ in range(iterations):
        net_spring_force = np.zeros((dict_size, 2), dtype=np.float32)
        for i, j in connections:
            length = base_length * (weight_array[i] + weight_array[j])
            vector = coordinates[j] + darts[i, j] * map_size - coordinates[i]
            unit_vector = vector / (0.0000001 + np.linalg.norm(vector))

            spring_force = vector - unit_vector * length
            net_spring_force[i] += spring_force

        net_velocity = damping_ratio * (net_velocity + spring_coefficient * net_spring_force)

        equilibrium = 1
        for c in range(dict_size):  # Check if particles are within tolerance
            if np.linalg.norm(net_velocity[c]) > 0.0001:
                equilibrium = 0
                break

        coordinates += net_velocity
        for a in range(dict_size):  # Update the position
            for axis in range(2):
                if not (0 <= coordinates[a, axis] < map_size[axis]):
                    dart_change = -np.sign(coordinates[a, axis])
                    coordinates[a, axis] = coordinates[a, axis] % map_size[axis]

                    for b in range(dict_size):  # Iterating over all of this vertex's connections
                        if graph[a, b]:
                            new_value = darts[a, b, axis] + dart_change
                            if new_value < -1:
                                new_value = 1
                            if new_value > 1:
                                new_value = -1

                            darts[a, b, axis] = int(new_value)  # Setting the dart for this vertex
                            darts[b, a, axis] = int(-new_value)  # Setting the dart for other vertex
        if equilibrium:
            break

    return coordinates, darts


# @jitclass
class DreamAtlasGraph:

    def __init__(self, size, map_size):

        self.size = size
        self.graph = np.zeros((size, size), dtype=np.bool_)
        self.coordinates = np.zeros((size, 2), dtype=np.int32)
        self.darts = np.zeros((size, size, 2), dtype=np.int8)
        self.weights = np.ones(size, dtype=np.float32)
        self.map_size = map_size

    def disconnect_nodes(self, i, j):

        self.graph[i, j] = 0
        self.graph[j, i] = 0
        self.darts[i, j] = np.zeros(2, dtype=np.int8)
        self.darts[i, j] = np.zeros(2, dtype=np.int8)

    def connect_nodes(self, i, j):  # Connects nodes i, j with the closest dart

        self.graph[i, j] = 1
        self.graph[j, i] = 1

    def insert_node(self, i, j, k):  # Inserts a new node k between two existing nodes i, j

        self.graph[i, j] = 0
        self.graph[j, i] = 0
        self.graph[i, k] = 1
        self.graph[k, i] = 1
        self.graph[j, k] = 1
        self.graph[k, j] = 1

        self.coordinates[k] = 0.5 * (self.coordinates[j] + self.darts[i, j] * self.map_size + self.coordinates[i])  # Find the coordinate between i, j

        new_dart = np.zeros(2, dtype=np.int8)
        for axis in [0, 1]:
            if self.map_size[axis] < self.coordinates[k][axis] or self.coordinates[k][axis] < 0:
                new_dart[axis] = np.sign(self.coordinates[k][axis])

        self.coordinates[k] = np.mod(self.coordinates[k], self.map_size)

        self.darts[i, j] = np.zeros(2, dtype=np.int8)
        self.darts[j, i] = np.zeros(2, dtype=np.int8)
        self.darts[i, k] = new_dart
        self.darts[k, j] = -new_dart

    def get_node_connections(self, i):
        return np.argwhere(self.graph[i, :] == 1)

    def get_all_connections(self):
        return np.argwhere(self.graph == 1)

    def get_length(self, i, j):
        return np.linalg.norm(self.coordinates[j] + self.darts[i, j] * self.map_size - self.coordinates[i])

    def get_faces_centroids(self):

        edges_set, embedding = set(), dict()
        for i in range(self.size):  # edges_set is an undirected graph as a set of undirected edges
            j_angles = list()
            connections = self.get_node_connections(i)
            if len(connections) == 0:
                continue
            for j in connections:
                j = j[0]
                edges_set |= {(i, j), (j, i)}
                vector = self.coordinates[j] + self.darts[i, j] * self.map_size - self.coordinates[i]
                angle = 90 - np.angle(vector[0] + vector[1] * 1j, deg=True)
                j_angles.append([j, angle])

            j_angles.sort(key=lambda x: x[1])
            embedding[i] = [x[0] for x in j_angles]  # Format: v1:[v2,v3], v2:[v1], v3:[v1] clockwise ordering of neighbors at each vertex

        faces, path = list(), list()  # Storage for face paths
        first_path = (0, embedding[0][0])
        path.append(first_path)
        edges_set -= {first_path}

        while len(edges_set) > 0:  # Trace faces
            neighbors = embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2]) + 1) % (len(neighbors))]
            tup = (path[-1][-1], next_node)
            if tup == path[0]:
                faces.append(path)
                path = list()
                for edge in edges_set:  # Starts next path
                    path.append(edge)
                    edges_set -= {edge}
                    break  # Only one iteration
            else:
                path.append(tup)
                edges_set -= {tup}
        if len(path) != 0:
            faces.append(path)

        centroids = list()
        for face in faces:
            shift = np.subtract(np.divide(self.map_size, 2), self.coordinates[face[0][0]])
            total = np.zeros(2)
            for edge in face:
                i = edge[0]
                total += (np.add(self.coordinates[i], shift)) % self.map_size
            coordinate = (-shift + total / len(face)) % self.map_size
            centroids.append((int(coordinate[0]), int(coordinate[1])))

        return faces, centroids

    def embed_graph(self, s_graph, seed: int):

        # Set the graph size to embed (smaller is faster)
        scale_down = 100
        size = np.array(self.map_size / scale_down, dtype=np.uint32)
        connections = [[1, 0], [0, 1], [0, -1], [-1, 0]]

        # Make the H graph
        h_dict = dict()
        for x in range(size[0]):
            for y in range(size[1]):
                h_dict[(int(x * scale_down), int(y * scale_down))] = list()
                for connection in connections:
                    x_coord = int(((x + connection[0]) % size[0]) * scale_down)
                    y_coord = int(((y + connection[1]) % size[1]) * scale_down)
                    h_dict[(int(x * scale_down), int(y * scale_down))].append((x_coord, y_coord))
        h_graph = ntx.Graph(incoming_graph_data=h_dict)   # H graph
        s_graph = ntx.Graph(incoming_graph_data=s_graph)  # S graph

        for i in range(10):
            initial_embedding, worked = mnm.find_embedding(s_graph, h_graph, return_overlap=True, random_seed=seed)
            if worked:
                break
            else:
                seed = rd.randint(0, 100000)
                print('\033[31mEmbedding failed: Trying with new seed (%i)\x1b[0m' % seed)
        if not worked:
            raise Exception('EmbeddingError: Failed 10 times')

        # Form the subgraph of the target graph
        subgraph_nodes, node_2_r, node_2_a, counter = list(), dict(), dict(), 0
        a_2_r = dict()
        for i in range(1, 1+len(s_graph)):
            for node in initial_embedding[i]:
                node_2_r[node] = i          # Node to real index
                node_2_a[node] = counter    # Node to attractor index
                a_2_r[counter] = i          # Attractor index to real index
                counter += 1
                subgraph_nodes.append(node)
        subgraph = h_graph.subgraph(subgraph_nodes)

        # Build the graph for attractor adjustment from the subgraph
        attractor_graph = np.zeros((len(subgraph), len(subgraph)), dtype=np.bool_)
        attractor_coordinates =  np.zeros((len(subgraph), 2), dtype=np.float64)
        attractor_darts = np.zeros((len(subgraph), len(subgraph), 2), dtype=np.int8)

        for i in range(1, 1+len(s_graph)):
            for node in initial_embedding[i]:  # Loop over all the embedded nodes in the subgraph
                attractor_coordinates[node_2_a[node]] = [node[0], node[1]]

                for edge in ntx.edges(subgraph, node):  # Add the connected subgraph nodes and darts
                    j = node_2_r[edge[1]]
                    if j == i or j in s_graph[i]:
                        dart = np.zeros(2, dtype=np.int8)
                        for axis in range(2):
                            if abs(edge[0][axis] - edge[1][axis]) > scale_down:
                                dart[axis] = np.sign(edge[0][axis] - edge[1][axis])
                        attractor_graph[node_2_a[node], node_2_a[edge[1]]] = 1
                        attractor_darts[node_2_a[node], node_2_a[edge[1]]] = dart

        attractor_array = np.zeros((len(subgraph), len(subgraph)), dtype=np.bool_)
        for i, j in np.argwhere(attractor_graph == 1):  # Determine which nodes are the same node and should be merged
            if a_2_r[i] == a_2_r[j]:
                attractor_array[i, j] = 1

        attractor_coordinates, attractor_darts = _numba_attractor_adjustment(attractor_graph, attractor_coordinates, attractor_darts, attractor_array, damping_ratio=0.5, map_size=self.map_size, iterations=1000)

        for i in range(1, 1+len(s_graph)):  # Merge the alike vertices of the graph
            coordinate_sum = 0
            coordinates_offset = np.subtract(np.divide(self.map_size, 2), attractor_coordinates[node_2_a[initial_embedding[i][0]]])
            for node in initial_embedding[i]:  # Loop over all the nodes for this vertex
                coordinate_sum += np.mod(np.add(coordinates_offset, attractor_coordinates[node_2_a[node]]), self.map_size)
            self.coordinates[i-1] = np.mod(np.subtract(np.divide(coordinate_sum, len(initial_embedding[i])), coordinates_offset), self.map_size)

        dart_dict, dist_dict = dict(), dict()
        for i, j in np.argwhere(attractor_graph == 1):
            dist_dict[a_2_r[i]-1, a_2_r[j]-1] = np.inf

        for i, j in np.argwhere(attractor_graph == 1):
            if a_2_r[i] != a_2_r[j]:
                distance = np.linalg.norm(self.coordinates[a_2_r[j]-1] - attractor_darts[i, j] * self.map_size - self.coordinates[a_2_r[i]-1])
                if distance < dist_dict[a_2_r[i]-1, a_2_r[j]-1]:
                    dist_dict[a_2_r[i] - 1, a_2_r[j] - 1] = distance
                    dart_dict[(a_2_r[i]-1, a_2_r[j]-1)] = attractor_darts[i, j]
                    print(a_2_r[i] - 1, a_2_r[j] - 1, distance, attractor_darts[i, j])

        done_edges = set()
        for i, j in self.get_all_connections():  # Set the darts for the graph
            if (j, i) not in done_edges:
                done_edges.add((i, j))
                self.darts[i, j] = dart_dict[(i, j)]
                self.darts[j, i] = -dart_dict[(i, j)]

    def spring_adjustment(self):
        ratios = np.array((0.1, 0.5, 20), dtype=np.float32)
        iterations = 50
        _coordinates, _darts = _numba_spring_adjustment(self.graph, self.coordinates.astype(dtype=np.float64), self.darts, self.weights, self.map_size, ratios, self.get_all_connections(), iterations)
        self.coordinates, self.darts = _coordinates.astype(dtype=np.int32), _darts.astype(dtype=np.int8)

    def plot(self):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for i, j in self.get_all_connections():
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[j] + self.darts[i, j] * self.map_size
            ax.plot((x1, x2), (y1, y2), 'k-')

        ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='r')
        ax.set(xlim=(0, self.map_size[0]), ylim=(0, self.map_size[1]))
        plt.show()

    # def get_virtual_graph(self):
    #
    #     virtual_size = self.size + len(np.argwhere(self.darts != np.zeros(2)))
    #     virtual_graph = np.zeros((virtual_size, virtual_size), dtype=np.bool_)
    #     virtual_coordinates = np.zeros((virtual_size, 2), dtype=np.int32)
    #
    #     k = self.size
    #     for i in range(self.size):
    #         connections = self.get_node_connections(i)
    #         if len(connections) == 0:
    #             continue
    #         for j in connections:
    #             j = j[0]
    #
    #             if self.darts[i, j] == np.zeros(2):  # If the connection does not cross the torus then just add the edge
    #                 virtual_graph[i, j] = 1
    #             else:                                # Otherwise we need to find the virtual coordinates
    #                 vector = self.coordinates[j] + self.darts[i, j] * self.map_size - self.coordinates[i]
    #                 unit_vector = vector / np.linalg.norm(vector)
    #
    #                 infinite_coordinates = list()  # Find the infinite edge points
    #                 for axis in range(2):
    #                     if self.darts[i, j][axis] == -1:
    #                         ic = 0
    #                     elif self.darts[i, j][axis] == 1:
    #                         ic = self.map_size[axis]
    #                     else:
    #                         continue
    #                     if axis == 0:
    #                         infinite_coordinates.append([ic, self.coordinates[i][1] + (ic - self.coordinates[i][0]) * unit_vector[1] / unit_vector[0]])
    #                     else:
    #                         infinite_coordinates.append([self.coordinates[i][0] + (ic - self.coordinates[i][1]) * unit_vector[0] / unit_vector[1], ic])
    #
    #                 if len(infinite_coordinates) == 2:  # Find which is closer and build virtual graph
    #                     virtual_graph[i, k] = 1
    #                     virtual_coordinates[k] = infinite_coordinates[0]
    #
    #                     virtual_graph[k + 1] = [k + 2]  # 1st edge to 2nd edge
    #                     virtual_graph[k + 2] = [k + 1]
    #                     if np.linalg.norm(np.subtract(self.coordinates[i], infinite_coordinates[0])) > np.linalg.norm(np.subtract(self.coordinates[i], infinite_coordinates[1])):  # seeing which is closer
    #                         infinite_coordinates = [infinite_coordinates[1], infinite_coordinates[0]]
    #                         virtual_coordinates[k + 1] = [infinite_coordinates[0][0], infinite_coordinates[0][1] - dart_y * self.mapsize[1]]
    #                         virtual_coordinates[k + 2] = [infinite_coordinates[1][0], infinite_coordinates[1][1] - dart_y * self.mapsize[1]]
    #                     else:
    #                         infinite_coordinates = [infinite_coordinates[0], infinite_coordinates[1]]
    #                         virtual_coordinates[new_index + 1] = [infinite_coordinates[0][0] - dart_x * self.mapsize[0], infinite_coordinates[0][1]]
    #                         virtual_coordinates[new_index + 2] = [infinite_coordinates[1][0] - dart_x * self.mapsize[0], infinite_coordinates[1][1]]
    #
    #                     virtual_graph[new_index + 3] = [j]  # 2nd edge to 2nd node
    #                     virtual_graph[j].append(new_index + 3)
    #                     virtual_coordinates[new_index + 3] = [infinite_coordinates[1][0] - dart_x * self.mapsize[0], infinite_coordinates[1][1] - dart_y * self.map_size[1]]
    #                 else:
    #                     virtual_graph[i].append(new_index)  # Vertex to edge
    #                     virtual_graph[new_index] = [i]
    #                     virtual_coordinates[new_index] = infinite_coordinates[0]
    #                     virtual_graph[new_index + 1] = [j]  # Edge to connection
    #                     virtual_graph[j].append(new_index + 1)
    #                     virtual_coordinates[new_index + 1] = [infinite_coordinates[0][0] - dart_x * self.mapsize[0], infinite_coordinates[0][1] - dart_y * self.map_size[1]]
    #
    #                 new_index += 2 * len(infinite_coordinates)
    #
    #     return virtual_graph, virtual_coordinates
