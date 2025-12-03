from scipy.spatial import ConvexHull
import numpy as np

def get_minimum_rectangle(features_coord):
        """
        This method computes the minimum bounding rectangle defined by vertex obtained by applying a convexHull algorithm to features coordinates.
        Save in class variables the vertex's coordinates of the hull, the rotation matrix and the rectangle vertex coordinates .
        This method is a modified version of the algorithm described in: https://stackoverflow.com/a/33619018
        Args:
            features_coord:

        Returns:

        """
        # Calculate vertices using Convex Hull
        hull=ConvexHull(features_coord)
        limits_points = features_coord[hull.vertices]
        # calculate edge angles
        pi2 = np.pi / 2
        edges = limits_points[1:] - limits_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            -np.sin(angles),
            np.sin(angles),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))
        # apply rotations to the hull
        rot_points = np.dot(rotations, limits_points.T)
        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)
        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        rotmat = rotations[best_idx]
        rect_coords = np.zeros((4, 2))
        rect_coords[0] = np.dot([x1, y2], rotmat)
        rect_coords[1] = np.dot([x2, y2], rotmat)
        rect_coords[2] = np.dot([x2, y1], rotmat)
        rect_coords[3] = np.dot([x1, y1], rotmat)

        return rotmat,rect_coords,limits_points