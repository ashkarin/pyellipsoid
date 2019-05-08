import math
import numpy as np


def build_rotation_matrix(ax, ay, az, inverse=False):
    """Build a Euler rotation matrix.
    Rotation order is X, Y, Z (right-hand coordinate system).
    Expected vector is [x, y, z].

    Arguments:
        ax {float} -- rotation angle around X (radians)
        ay {float} -- rotation angle around Y (radians)
        az {float} -- rotation angle around Z (radians)

    Keyword Arguments:
        inverse {bool} -- Do inverse rotation (default: {False})

    Returns:
        [numpy.array] -- rotation matrix
    """

    if inverse:
        ax, ay, az = -ax, -ay, -az

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])

    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])

    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def rotation_matrix_to_angles(R):
    """Compute Euler angles from the given Euler rotation matrix.

    Arguments:
        R {array} -- Euler rotation matrix

    Returns:
        [array] -- Euler angles
    """
    if abs(R[2, 0]) != 1:
        ay = -math.asin(R[2, 0])
        c = math.cos(ay)
        ax = math.atan2(R[2, 1] / c, R[2, 2] / c)
        az = math.atan2(R[1, 0] / c, R[0, 0] / c)

    else:
        az = 0
        if R[2, 0] == -1:
            ay = math.pi / 2.0
            ax = az + math.atan2(R[0, 1], R[0, 2])
        else:
            ay = -math.pi / 2.0
            ax = -az + math.atan2(-R[0, 1], -R[0, 2])

    return np.array([ax, ay, az])


def find_relative_vector_rotation(a, b):
    """Find a Euler rotation matrix from the vector `a` to the vector `b` using Rodrigues' rotation formula.

    Arguments:
        a {array} -- source vector [x, y, z]
        b {array} -- target vector [x, y, z]

    Returns:
        [numpy.array] -- rotation matrix
    """

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    if np.linalg.norm(a - b) < 0.001:
        return np.eye(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    Im = np.identity(3)
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    k = np.array(np.matrix(vXStr))
    R = Im + k + np.matmul(k, k) * ((1 - c)/(s**2))
    return R


def find_relative_axes_rotation(source_axes, target_axes, validate=True):
    """Find the rotation from `source_axes` to `target_axes`.

    Arguments:
        source_axes {numpy.array} -- a list of vectors (x, y, z)
        target_axes {numpy.array} -- a list of vectors (x, y, z)

    Keyword Arguments:
        validate {bool} -- validation (default: {True})

    Returns:
        [numpy.array] -- Euler rotation matrix, Euler angles
    """

    # Convert to numpy
    source_axes = np.array(source_axes)
    target_axes = np.array(target_axes)

    # Find rotation between coordinate systems formed by the axes
    R = np.dot(target_axes.T, np.linalg.inv(source_axes.T))
    angles = rotation_matrix_to_angles(R)

    if validate:
        R_rebuild = build_rotation_matrix(*angles)
        axes = np.dot(R_rebuild, source_axes.T).T

        diff = np.linalg.norm(target_axes - axes)
        if diff > 1e-3:
            raise RuntimeError("Found rotation angles are incorrect!"
                               " norm(expected_axes - computed_axes) = {}".format(diff))

    return R


def scalar_projection(source, target):
    """Compute a scalar pojection of source vector to target vector

    Arguments:
        source {array} -- source vector
        target {array} -- target vector

    Returns:
        [float] -- projection
    """
    return np.dot(source, target) / np.linalg.norm(target)


def find_vectors_mapping(source, target):
    """Find indices mapping source vectors to the target one based on the projections

    Arguments:
        source {array} -- source array of vectors
        target {array} -- target array of vectors

    Returns:
        [array] -- indices mapping source vectors to target vectors
    """
    return [np.argmax([abs(scalar_projection(s, t)) for t in target]) for s in source]
