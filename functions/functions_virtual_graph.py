import numpy as np


def make_virtual_graph(graph, coordinates, darts, mapsize):

    # graph: adjacency matrix (numpy array), coordinates: numpy array (N,2), darts: numpy array (N,N,2), mapsize: tuple/list
    N = graph.shape[0]
    virtual_graph = {i: [] for i in range(N)}
    virtual_coordinates = {i: coordinates[i].tolist() for i in range(N)}
    new_index = N
    done_edges = set()
    for i in range(N):
        for j in range(N):
            if graph[i, j]:
                if (i, j) in done_edges or (j, i) in done_edges:
                    continue
                done_edges.add((i, j))
                dart_x, dart_y = darts[i, j]
                if dart_x == 0 and dart_y == 0:
                    virtual_graph[i].append(j)
                    virtual_graph[j].append(i)
                else:
                    vector = coordinates[j] + darts[i, j] * np.asarray(mapsize) - coordinates[i]
                    unit_vector = vector / np.linalg.norm(vector)
                    infinite_coordinates = []
                    for axis in range(2):
                        if darts[i, j][axis] == -1:
                            ic = 0
                        elif darts[i, j][axis] == 1:
                            ic = mapsize[axis]
                        else:
                            continue
                        if axis == 0:
                            infinite_coordinates.append([ic, coordinates[i][1] + (ic - coordinates[i][0]) * unit_vector[1] / unit_vector[0]])
                        else:
                            infinite_coordinates.append([coordinates[i][0] + (ic - coordinates[i][1]) * unit_vector[0] / unit_vector[1], ic])
                    if len(infinite_coordinates) == 2:
                        virtual_graph[i].append(new_index)
                        virtual_graph[new_index] = [i]
                        virtual_coordinates[new_index] = infinite_coordinates[0]
                        virtual_graph[new_index + 1] = [new_index + 2]
                        virtual_graph[new_index + 2] = [new_index + 1]
                        if np.linalg.norm(coordinates[i] - infinite_coordinates[0]) > np.linalg.norm(coordinates[i] - infinite_coordinates[1]):
                            infinite_coordinates = [infinite_coordinates[1], infinite_coordinates[0]]
                            virtual_coordinates[new_index + 1] = [infinite_coordinates[0][0], infinite_coordinates[0][1] - dart_y * mapsize[1]]
                            virtual_coordinates[new_index + 2] = [infinite_coordinates[1][0], infinite_coordinates[1][1] - dart_y * mapsize[1]]
                        else:
                            infinite_coordinates = [infinite_coordinates[0], infinite_coordinates[1]]
                            virtual_coordinates[new_index + 1] = [infinite_coordinates[0][0] - dart_x * mapsize[0], infinite_coordinates[0][1]]
                            virtual_coordinates[new_index + 2] = [infinite_coordinates[1][0] - dart_x * mapsize[0], infinite_coordinates[1][1]]
                        virtual_graph[new_index + 3] = [j]
                        virtual_graph[j].append(new_index + 3)
                        virtual_coordinates[new_index + 3] = [infinite_coordinates[1][0] - dart_x * mapsize[0], infinite_coordinates[1][1] - dart_y * mapsize[1]]
                        new_index += 4
                    else:
                        virtual_graph[i].append(new_index)
                        virtual_graph[new_index] = [i]
                        virtual_coordinates[new_index] = infinite_coordinates[0]
                        virtual_graph[new_index + 1] = [j]
                        virtual_graph[j].append(new_index + 1)
                        virtual_coordinates[new_index + 1] = [infinite_coordinates[0][0] - dart_x * mapsize[0], infinite_coordinates[0][1] - dart_y * mapsize[1]]
                        new_index += 2
    return virtual_graph, virtual_coordinates


def ui_find_virtual_graph(graph, coordinates, map_size, wraparound):

    darts = dict()
    for i in graph:
        darts[i] = list()
        ix, iy = coordinates[i]
        for j in graph[i]:
            min_dist = np.inf
            best_dart = [0, 0]
            for n in wraparound:
                jx, jy = np.add(coordinates[j], np.multiply(map_size, n))
                new_dist = np.linalg.norm([ix-jx, iy-jy])
                if new_dist < min_dist:
                    best_dart = n
                    min_dist = new_dist
            darts[i].append(best_dart)

    return make_virtual_graph(graph, coordinates, darts, map_size)
