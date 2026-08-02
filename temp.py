import numpy as np
import matplotlib.pyplot as plt
from src.DreamAtlas.functions.functions_lloyd import LloydRelaxation
from FlowAtlas.try_voronoi_creation import gen_grid, spread_points, voronoi_and_graph

points = [[0.1, 0.6],
          [0.3, 0.8],
          [0.4, 0.4],
          [0.7, 0.9]]
points = np.array(points)

#points = gen_grid(num_points)
#points = spread_points(points)
vor, all_points = LloydRelaxation.get_unit_periodic_voronoi(points)

for point1 in points:
    for point2 in points:
        distance = np.linalg.norm(LloydRelaxation.unit_toroidal_delta(point1, point2))
        non_loop_distance = np.linalg.norm(point1 - point2)
        if distance != non_loop_distance:
            print(f"norm({point1} - {point2}) = {distance} when looping, {non_loop_distance} otherwise")

# for point in all_points:
#     print(LloydRelaxation.get_origin_point(point, all_points))

# for point1 in all_points:
#     for point2 in all_points:
#         # print(point1)
#         # print(point2)
#         print(LloydRelaxation.unit_toroidal_delta(point1, point2))
#         # print()
#     print()

#graph_plain, voronoi = voronoi_and_graph(points)
fig = plt.figure()
plt.scatter(all_points[:, 0,], all_points[:, 1])
plt.xlim([-1,2])
plt.ylim([-1,2])
fig = plt.figure()
plt.scatter(points[:, 0,], points[:, 1])
plt.xlim([0,1])
plt.ylim([0,1])
wrapped = LloydRelaxation.wrap_unit_coordinates(all_points)
fig = plt.figure()
plt.scatter(wrapped[:, 0,], wrapped[:, 1])
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()