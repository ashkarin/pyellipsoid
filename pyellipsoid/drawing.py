import numpy as np
from pyellipsoid.geometry import build_rotation_matrix


def make_ellipsoid_image(shape, center, radii, angle):
    """Draw a 3D binary image containing a 3D ellipsoid.

    Arguments:
        shape {list} -- image shape [z, y, x]
        center {list} -- center of the ellipsoid [x, y, z]
        radii {list} -- radii [x, y, z]
        angle {list} -- rotation angles [x, y, z]

    Raises:
        ValueError -- arguments are wrong

    Returns:
        [numpy.array] -- image with ellipsoid
    """

    if len(shape) != 3:
        raise ValueError('Only 3D ellipsoids are supported.')

    if not (len(center) == len(radii) == len(shape)):
        raise ValueError('Center, radii of ellipsoid and image shape have different dimensionality.')

    # Do opposite rotation since it is an axes rotation.
    angle = -1 * np.array(angle)
    R = build_rotation_matrix(*angle)

    # Convert to numpy
    radii = np.array(radii)

    # Build a grid and get its points as a list
    xi = tuple(np.linspace(0, s-1, s) - np.floor(0.5 * s) for s in shape)

    # Build a list of points forming the grid
    xi = np.meshgrid(*xi, indexing='ij')
    points = np.array(xi).reshape(3, -1)[::-1]

    # Reorder coordinates to match XYZ order and rotate
    points = np.dot(R, points).T

    # Find grid center and rotate
    grid_center = np.array(center) - 0.5*np.array(shape[::-1])
    grid_center = np.dot(R, grid_center)

    # Reorder coordinates back to ZYX to match the order of numpy array axis
    points = points[:, ::-1]
    grid_center = grid_center[::-1]
    radii = radii[::-1]

    # Draw the ellipsoid
    # dx**2 + dy**2 + dz**2 = r**2
    # dx**2 / r**2 + dy**2 / r**2 + dz**2 / r**2 = 1
    dR = (points - grid_center)**2
    dR = dR / radii**2
    # Sum dx, dy, dz / r**2
    nR = np.sum(dR, axis=1).reshape(shape)

    ell = (nR <= 1).astype(np.uint8)

    return ell
