# Source - https://stackoverflow.com/a/63359049
# Posted by Alex
# Retrieved 2026-02-28, License - CC BY-SA 4.0

import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt


def verify_voronoi_mapping(vor, points, point_indices=None):
    """
    Verify that vor.point_region correctly maps points to regions.

    For each point, checks that:
    1. The point index has a valid region ID
    2. The region exists in vor.regions
    3. The region's vertices exist in vor.vertices

    Args:
        vor: scipy.spatial.Voronoi object
        points: array of point coordinates
        point_indices: list of point indices to check (default: all points)

    Returns:
        dict with results of verification
    """
    if point_indices is None:
        point_indices = range(len(points))

    results = {
        'valid': True,
        'checked_points': len(point_indices),
        'issues': []
    }

    for j in point_indices:
        if j >= len(vor.point_region):
            results['issues'].append(f"Point {j}: index out of range for point_region")
            results['valid'] = False
            continue

        region_id = vor.point_region[j]

        if region_id >= len(vor.regions):
            results['issues'].append(f"Point {j}: region_id {region_id} out of range for regions")
            results['valid'] = False
            continue

        region = vor.regions[region_id]

        # Check if region contains -1 (unbounded region)
        has_unbounded = -1 in region

        # Check if all vertex indices are valid
        invalid_vertices = [v for v in region if v != -1 and v >= len(vor.vertices)]
        if invalid_vertices:
            results['issues'].append(f"Point {j}: region {region_id} has invalid vertex indices {invalid_vertices}")
            results['valid'] = False

        results[f'point_{j}'] = {
            'region_id': region_id,
            'num_vertices': len(region),
            'is_unbounded': has_unbounded,
            'point_coord': tuple(points[j])
        }

    return results


def color_voronoi_faces(point2color):
    # get all points, and add 4 distante points around the edges to ensure
    # we get faces for all points, distant being determined by max and min of the points
    original_points = list(point2color.keys())
    points = np.array(original_points)
    num_original_points = len(points)

    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    x_range = max_x - min_x
    y_range = max_y - min_y
    extra_points = np.array([[min_x - x_range, min_y - y_range],
                             [min_x - x_range, max_y + y_range],
                             [max_x + x_range, min_y - y_range],
                             [max_x + x_range, max_y + y_range]])
    points = np.vstack([points, extra_points])
    vor = Voronoi(points)

    results = verify_voronoi_mapping(vor, original_points, point_indices=range(num_original_points))
    if results['valid']:
        print("✓ All mappings verified successfully")
    else:
        print("✗ Issues found:")
        for issue in results['issues']:
            print(f"  {issue}")

    # Only color the regions for the original points (indices 0 to num_original_points-1)
    # The extra boundary points (indices num_original_points to num_original_points+3) are only
    # used to ensure boundary cells are closed
    for j in range(num_original_points):
        point_coord = original_points[j]
        color = point2color[point_coord]
        region_id = vor.point_region[j]
        region = vor.regions[region_id]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=color)
