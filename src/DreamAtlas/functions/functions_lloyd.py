import numpy as np
import scipy as sc
import scipy.cluster.vq as sccvq


# Credit to Douglas Duhaime
# https://github.com/duhaime/lloyd/blob/master/lloyd/lloyd.py


class LloydRelaxation:
    """
    Create a Voronoi map that can be used to run Lloyd
    relaxation on an array of 2D points. For background,
    see: https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
    """

    def __init__(self, *args, **kwargs):
        """
        Store the points and bounding box of the points to which
        Lloyd relaxation will be applied.
        @param np.array `arr`: a numpy array with shape n, 2, where n
        is the number of 2D points to be moved
        @param float `epsilon`: the delta between the input point
        domain and the pseudo-points used to constrain the points
        """
        arr = args[0]
        if not isinstance(arr, np.ndarray) or arr.shape[1] != 2:
            raise Exception('Please provide a numpy array with shape n,2')
        self.toroidal = kwargs.get('toroidal', False)
        if self.toroidal:
            self.constrain = False
        else:
            self.constrain = kwargs.get('constrain', True)
        self.points = arr
        too_large = np.any(self.points >= 1) or np.any(self.points < 0)
        if self.toroidal and too_large:
            self.normalize_to_unit()
        # find the bounding box of the input data
        self.domains = self.get_domains(arr)
        # ensure no two points have the exact same coords
        self.jitter_points()
        self.bb_points = self.get_bb_points(arr)
        self.build_voronoi()

        # Placeholders for toroidal bondary conditions
        self.expanded_voronoi, _ = self.get_unit_periodic_voronoi(self.points)
        self.min_bounds = None
        self.scale_factors = None

    def jitter_points(self, scalar=.000000001):
        """
        Ensure no two points have the same coords or else the number
        of regions will be less than the number of input points
        """
        while self.points_contain_duplicates():
            positive = np.random.rand(len(self.points), 2) * scalar
            negative = np.random.rand(len(self.points), 2) * scalar
            self.points = self.points + positive - negative
            self.constrain_points()

    def constrain_points(self):
        """
        Update any points that have drifted beyond the boundaries of this space
        """
        if not self.constrain:
            return

        for point in self.points:
            if point[0] < self.domains['x']['min']: point[0] = self.domains['x']['min']
            if point[0] > self.domains['x']['max']: point[0] = self.domains['x']['max']
            if point[1] < self.domains['y']['min']: point[1] = self.domains['y']['min']
            if point[1] > self.domains['y']['max']: point[1] = self.domains['y']['max']

    def get_domains(self, arr):
        """
        Return an object with the x, y domains of `arr`
        """
        x = arr[:, 0]
        y = arr[:, 1]
        return {
            'x': {
                'min': min(x),
                'max': max(x),
            },
            'y': {
                'min': min(y),
                'max': max(y),
            }
        }

    def get_bb_points(self, arr):
        """
        Given an array of 2D points, return the four vertex bounding box
        """
        return np.array([
            [self.domains['x']['min'], self.domains['y']['min']],
            [self.domains['x']['max'], self.domains['y']['min']],
            [self.domains['x']['min'], self.domains['y']['max']],
            [self.domains['x']['max'], self.domains['y']['max']], ])

    def build_voronoi(self):
        """
        Build a voronoi map from self.points. For background on
        self.voronoi attributes, see: https://docs.scipy.org/doc/scipy/
          reference/generated/scipy.spatial.Voronoi.html
        """
        # build the voronoi tessellation map
        self.voronoi = sc.spatial.Voronoi(self.points, qhull_options='Qbb Qc Qx Qz')

        # constrain voronoi vertices within bounding box
        if self.constrain:
            for idx, vertex in enumerate(self.voronoi.vertices):
                x, y = vertex
                if x < self.domains['x']['min']:
                    self.voronoi.vertices[idx][0] = self.domains['x']['min']
                if x > self.domains['x']['max']:
                    self.voronoi.vertices[idx][0] = self.domains['x']['max']
                if y < self.domains['y']['min']:
                    self.voronoi.vertices[idx][1] = self.domains['y']['min']
                if y > self.domains['y']['max']:
                    self.voronoi.vertices[idx][1] = self.domains['y']['max']

        if self.toroidal:
            self.expanded_voronoi, _ = self.get_unit_periodic_voronoi(self.points)

    def points_contain_duplicates(self):
        """
        Return a boolean indicating whether self.points contains duplicates
        """
        vals, count = np.unique(self.points, return_counts=True)
        return np.any(vals[count > 1])

    def find_centroid(self, vertices):
        """
        Find the centroid of a Voroni region described by `vertices`,
        and return a np array with the x and y coords of that centroid.
        The equation for the method used here to find the centroid of a
        2D polygon is given here: https://en.wikipedia.org/wiki/
          Centroid#Of_a_polygon
        @params: np.array `vertices` a numpy array with shape n,2
        @returns np.array a numpy array that defines the x, y coords
          of the centroid described by `vertices`
        """
        area = 0
        centroid_x = 0
        centroid_y = 0
        for i in range(len(vertices) - 1):
            step = (vertices[i, 0] * vertices[i + 1, 1]) - \
                   (vertices[i + 1, 0] * vertices[i, 1])
            area += step
            centroid_x += (vertices[i, 0] + vertices[i + 1, 0]) * step
            centroid_y += (vertices[i, 1] + vertices[i + 1, 1]) * step
        area /= 2
        # prevent division by zero - equation linked above
        if area == 0: area += 0.0000001
        centroid_x = (1.0 / (6.0 * area)) * centroid_x
        centroid_y = (1.0 / (6.0 * area)) * centroid_y
        # prevent centroids from escaping bounding box
        if self.constrain:
            if centroid_x < self.domains['x']['min']: centroid_x = self.domains['x']['min']
            if centroid_x > self.domains['x']['max']: centroid_x = self.domains['x']['max']
            if centroid_y < self.domains['y']['min']: centroid_y = self.domains['y']['min']
            if centroid_y > self.domains['y']['max']: centroid_y = self.domains['y']['max']

        return np.array([centroid_x, centroid_y])

    def relax(self):
        """
          Moves each point to the centroid of its cell in the voronoi
          map to "relax" the points (i.e. jitter the points so as
          to spread them out within the space).
        """
        if self.toroidal:
            voronoi_graph = self.expanded_voronoi
        else:
            voronoi_graph = self.voronoi
        centroids = []
        for idx in voronoi_graph.point_region:  # the region is a series of indices into voronoi_graph.vertices
            region = [i for i in voronoi_graph.regions[idx] if
                      i != -1]  # remove point at infinity, designated by index -1
            region = region + [region[0]]  # enclose the polygon
            verts = voronoi_graph.vertices[region]  # get the vertices for this region
            centroids.append(self.find_centroid(verts))  # find the centroid of those vertices

        if self.toroidal:
            last_idx = int(len(centroids)/9)
            new_points = self.wrap_unit_coordinates(np.array(centroids)[:last_idx])
            print(len(new_points))
            self.points = new_points
        else:
            self.points = np.array(centroids)
        self.constrain_points()
        self.jitter_points()
        self.build_voronoi()

    def get_points(self):
        """
        Return the input points in the new projected positions
        @returns np.array a numpy array that contains the same number
        of observations in the input points, in identical order
        """
        if self.min_bounds != None and self.scale_factors != None:
            return self.scale_up_from_unit()
        return self.points


    # Extra helper code for periodic boundary conditions

    def normalize_to_unit(self):
        """
        Finds the bounding box of self.points, shifts the minimum to 0,
        and scales all coordinates down to the [0, 1) unit domain.
        """
        # 1. Determine the bounding box (handles negative numbers automatically)
        self.min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)

        # 2. Compute the span (width and height) of the point cloud
        span = max_bounds - self.min_bounds

        # Avoid division by zero if all points share a coordinate line
        self.scale_factors = np.where(span == 0, 1.0, span)

        # 3. Shift to zero, divide by scale, and safely wrap into [0, 1)
        shifted_points = self.points - self.min_bounds
        normalized_points = shifted_points / self.scale_factors

        # Use np.mod to handle edge-case precision rounding at exactly 1.0
        self.points = np.mod(normalized_points, 1.0)

    def scale_up_from_unit(self, unit_points=None):
        """
        Reverses the normalization process to project unit square points
        back into the original coordinate space.
        """
        if self.min_bounds is None or self.scale_factors is None:
            raise ValueError("Cannot scale up. Points have not been normalized yet.")

        # If no custom points array is passed, default to recovering self.points
        target_points = self.points if unit_points is None else np.asarray(unit_points)

        # Multiply by original spans, then shift back to original location
        return (target_points * self.scale_factors) + self.min_bounds


    @staticmethod
    def get_unit_periodic_voronoi(points):
        """
        Computes periodic Voronoi by tiling 8 ghost blocks around a unit square.
        Returns the Voronoi object where the first 1/9 indices map to your seeds.
        """
        # 8 static offset shifts for a unit box
        offsets = np.array([
            [-1,  1], [0,  1], [1,  1],
            [-1,  0],          [1,  0],
            [-1, -1], [0, -1], [1, -1],
        ])

        # Replicate seeds into ghost zones
        ghost_seeds = [points + offset for offset in offsets]
        all_points = np.vstack([points] + ghost_seeds)

        return sc.spatial.Voronoi(all_points, qhull_options='Qbb Qc Qx Qz'), all_points

    @staticmethod
    def wrap_unit_coordinates(coords):
        """Wraps any coordinate outside [0, 1) back into the unit square."""
        return np.mod(coords, 1.0)

    @staticmethod
    def get_origin_point(point, all_points):
        """
        Find the original point for a starting point,
        assuming all_points includes ghost points, avoiding precision errors
        """
        modlength = int(len(all_points)/9)
        idx, = np.where(np.all(all_points == point, axis=1))
        if len(idx) != 1:
            raise ValueError("Found multiple matching points, get_origin_point must have gotten a bad all_points argument")
        return all_points[np.mod(idx[0], modlength)]

    @staticmethod
    def unit_toroidal_delta(coords1, coords2):
        """Shortest signed distance vector from coords2 to coords1 in a unit torus."""
        delta = coords1 - coords2
        return delta - np.round(delta)