import numpy as np
from pyellipsoid.geometry import build_rotation_matrix


def make_ellipsoid_image(shape, center, radii, angles):
    """Draw a 3D binary image containing a 3D ellipsoid.

    Arguments:
        shape {list} -- image shape [z, y, x]
        center {list} -- center of the ellipsoid [x, y, z]
        radii {list} -- radii [x, y, z]
        angles {list} -- rotation angles [x, y, z]

    Raises:
        ValueError -- arguments are wrong

    Returns:
        [numpy.array] -- image with ellipsoid
    """
    if len(shape) != 3:
        raise ValueError('Only 3D ellipsoids are supported.')

    if not (len(center) == len(radii) == len(shape)):
        raise ValueError('Center, radii of ellipsoid and image shape have different dimensionality.')

    # Angles specify ellipsoid rotation in the global coordinate system,
    # but for a better quality the global coordinate system should be rotated in opposite direction.
    # For this reason, the rotation matrix is Transposed.
    R = np.transpose(build_rotation_matrix(*angles))
    
    # Convert radii to a numpy array
    radii = np.array(radii)
    
    # Ellipsoid region
    max_radii = np.ceil(max(radii)).astype(np.int32)
    ell_region_shape = np.array([max_radii * 2 for _ in range(3)])

    # Build a grid for the ellipsoid area
    xi = tuple(np.linspace(0, s-1, s) - np.floor(0.5 * s) for s in ell_region_shape)

    # Build a list of points forming the grid
    xi = np.meshgrid(*xi, indexing='ij')
    points = np.array(list(zip(*np.vstack(map(np.ravel, xi)))))

    # Reorder coordinates from ZYX to XYZ format and rotate
    points = points[:, ::-1]
    points = np.dot(R, points.T).T
    
    # Reorder coordinates back to ZYX to match the order of numpy array axis
    points = points[:, ::-1]
    radii = radii[::-1]
    center = center[::-1]

    # Draw the ellipsoid
    # dx**2 + dy**2 + dz**2 = r**2
    # dx**2 / r**2 + dy**2 / r**2 + dz**2 / r**2 = 1
    dR = points**2 / radii**2
    print (ell_region_shape)
    dR = np.sum(dR, axis=1).reshape(ell_region_shape)
    
    ell_region = (dR <= 1).astype(np.uint8)
    
    # Create the image and insert ellipsoid
    begin = np.round(center - 0.5 * ell_region_shape).astype(np.int32)
    end = (begin + ell_region_shape).astype(np.int32)

    image = np.zeros(shape)
    image_region = [slice(b, e) for b, e in zip(begin, end)]
    image[image_region] = ell_region

    return image
