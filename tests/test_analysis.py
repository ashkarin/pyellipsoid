import unittest
import numpy as np
from pyellipsoid import drawing
from pyellipsoid import analysis


class TestAnalysisInertia(unittest.TestCase):
    def test_inertia_no_rotation(self):
        V = self.compute_inertia([0, 0, 0])

        err = np.linalg.norm(np.abs(V[0]) - [0, 1, 0])
        self.assertLess(err, 1e-1)

        err = np.linalg.norm(np.abs(V[1]) - [0, 0, 1])
        self.assertLess(err, 1e-1)

        err = np.linalg.norm(np.abs(V[2]) - [1, 0, 0])
        self.assertLess(err, 1e-1)

    def compute_inertia(self, angles):
        shape = (128, 128, 128)
        center_xyz = (64, 64, 64)
        radii_xyz = (5, 50, 30)
        angles_xyz = np.deg2rad(angles)

        image = drawing.make_ellipsoid_image(shape, center_xyz, radii_xyz, angles_xyz)
        points = analysis.sample_random_points(image)
        inertial_ellipsoid = analysis.compute_inertia_ellipsoid(points)
        return inertial_ellipsoid.axes

    def test_inertia_rotation_around_z(self):
        V = self.compute_inertia([0, 0, 45])
        # After rotation around Z (RHS) we expect the following major axis vectors

        # 1st major axis ([0, 1, 0] or [0, -1, 0])
        err1 = np.linalg.norm(V[0] - [-0.70710678, 0.70710678, 0.])
        err2 = np.linalg.norm(V[0] - [0.70710678, -0.70710678, 0.])
        self.assertLess(min(err1, err2), 1e-1)

        # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
        err1 = np.linalg.norm(V[1] - [0, 0, 1])
        err2 = np.linalg.norm(V[1] - [0, 0, -1])
        self.assertLess(min(err1, err2), 1e-1)

        # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
        err1 = np.linalg.norm(V[2] - [0.70710678, 0.70710678, 0.])
        err2 = np.linalg.norm(V[2] - [-0.70710678, -0.70710678, 0.])
        self.assertLess(min(err1, err2), 1e-1)

    def test_inertia_rotation_around_y(self):
        V = self.compute_inertia([0, 45, 0])
        # After rotation around Y (RHS) we expect the following major axis vectors

        # 1st major axis ([0, 1, 0] or [0, -1, 0])
        err1 = np.linalg.norm(V[0] - [0, 1, 0])
        err2 = np.linalg.norm(V[0] - [0, -1, 0])
        self.assertLess(min(err1, err2), 1e-1)

        # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
        err1 = np.linalg.norm(V[1] - [0.70710678, 0, 0.70710678])
        err2 = np.linalg.norm(V[1] - [-0.70710678, 0, -0.70710678])
        self.assertLess(min(err1, err2), 1e-1)

        # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
        err1 = np.linalg.norm(V[2] - [0.70710678, 0, -0.70710678])
        err2 = np.linalg.norm(V[2] - [-0.70710678, 0, 0.70710678])

        self.assertLess(min(err1, err2), 1e-1)

    def test_inertia_rotation_around_x(self):
        V = self.compute_inertia([45, 0, 0])
        # After rotation around X (RHS) we expect the following major axis vectors

        # 1st major axis ([0, 1, 0] or [0, -1, 0])
        err1 = np.linalg.norm(V[0] - [0, 0.70710678, 0.70710678])
        err2 = np.linalg.norm(V[0] - [0, -0.70710678, -0.70710678])
        self.assertLess(min(err1, err2), 1e-1)

        # 2nd Major axis ([0, 0, 1] or [0, 0, -1])
        err1 = np.linalg.norm(V[1] - [0, 0.70710678, -0.70710678])
        err2 = np.linalg.norm(V[1] - [0, -0.70710678, 0.70710678])
        self.assertLess(min(err1, err2), 1e-1)

        # 3rd Major axis ([1, 0, 0] or [-1, 0, 0])
        err1 = np.linalg.norm(V[2] - [1, 0, 0])
        err2 = np.linalg.norm(V[2] - [-1, 0, 0])
        self.assertLess(min(err1, err2), 1e-1)
