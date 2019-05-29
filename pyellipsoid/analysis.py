import json
import numpy as np
from collections import namedtuple, defaultdict
from pyellipsoid import geometry


Ellipsoid = namedtuple('Ellipsoid', ['center', 'radii', 'axes'])


def ellipsoid_to_dict(ellipsoid):
    """ Convert `ellipsoid` to serializable dictionary.
    Arguments:
        ellipsoid {Ellipsoid} -- ellipsoid instance

    Returns:
        [dict] -- serializable dictionary
    """
    D = {}
    if isinstance(ellipsoid, tuple):
        datas = ellipsoid._asdict()
        for data in datas:
            if isinstance(datas[data], np.ndarray):
                D[data] = datas[data].tolist()
            else:
                D[data] = (datas[data])
    return D


def ellipsoid_from_dict(data):
    """ Create Ellipsoid instance from the `data`.
    Arguments:
        data {dict} -- dictionary

    Returns:
        [analysis.Ellipsoid] -- instance of the Ellipsoid
    """
    if not isinstance(data, dict):
        raise ValueError("`data` should be a dictionary")

    if any(prop not in data for prop in Ellipsoid._fields):
        raise RuntimeError("`data` should contain {}".format(Ellipsoid._fields))

    return Ellipsoid(*[np.array(data[k]) for k in Ellipsoid._fields])


def ellipsoid_to_json(ellipsoid):
    D = ellipsoid_to_dict(ellipsoid)
    return json.dumps(D)


def ellipsoid_from_json(json_data):
    D = json.loads(json_data)
    return ellipsoid_from_dict(D)


def sample_random_points(image, n=2000):
    """Sample `n` random points from non-zero values in the `image`.

    Arguments:
        image {numpy.array} -- image (axes order: Z, Y, X)

    Keyword Arguments:
        n {int} -- number of points (default: {2000})

    Returns:
        [numpy.array] -- array of points (x, y, z)
    """
    if image is None:
        return None

    if not isinstance(n, int) or n <= 0:
        raise ValueError("`n` should be a positive integer")

    # Extract and transform points to XYZ
    points = np.array(np.nonzero(image)).T
    points = points[:, ::-1]

    # Choose randomly subset of points
    indices = np.random.randint(0, points.shape[0], n)
    points = points[indices]

    return points


def sample_all_points(image):
    """Sample all points from non-zero values in the `image`.

    Arguments:
        image {numpy.array} -- image (axes order: Z, Y, X)
    
    Returns:
        [numpy.array] -- array of points (x, y, z)
    """
    points = np.array(np.nonzero(image)).T
    points = points[:, ::-1]
    return points


def compute_inertia_ellipsoid(points):
    """Compute inertia ellipsoid for `points`.

    Please note, that order of Ellipsoid.axes might not correspond to the order of image axes!

    Arguments:
        points {numpy.array} -- an array of points (x, y, z)

    Returns:
        [analysis.Ellipsoid] -- inertia ellipsoid
    """
    npoints = points.shape[0]
    center = np.mean(points, axis=0)
    points = points - center

    covariance = np.cov(points.T) / npoints
    _, s, V = np.linalg.svd(covariance)
    radii = 2 * np.sqrt(s * npoints).T

    return Ellipsoid(center, radii, V)


def map_ellipsoid_to_axes(ellipsoid, axes):
    """Analyze a sequence of inertial ellipsoids `ellipsoids`.

    Arguments:
        ellipsoid {Ellipsoid} -- an `Ellipsoid` instance
        axes {list} -- a list of vectors defining the axes

    Returns:
        [Ellipsoid] -- an `Ellipsoid` instance
    """
    # Find mapping
    mapping = geometry.find_axes_mapping(ellipsoid.axes, axes)

    # Apply mapping
    axes = ellipsoid.axes[mapping]
    radii = ellipsoid.radii[mapping]
    
    # Mirror those ellipsoid axes, which are in the opposite direction from the target
    axes = np.array([sv * np.sign(np.dot(sv, st)) for sv, st in zip(axes, axes)])

    return Ellipsoid(ellipsoid.center, radii, axes)


def analyze_sequence(ellipsoids, validate=True):
    """Analyze a sequence of inertial ellipsoids `ellipsoids`.

    Arguments:
        ellipsoids {list} -- a list of `Ellipsoid` instances

    Keyword Arguments:
        validate {bool} -- validation (default: {True})

    Returns:
        [dict] -- dictionary of stats
    """
    if not all(isinstance(ell, Ellipsoid) for ell in ellipsoids):
        raise ValueError("The entries of `ellipsoids` must be of the `Ellipsoid` type")

    ndims = len(ellipsoids[0].axes)
    if not all(len(ell.axes) == ndims for ell in ellipsoids):
        raise ValueError("The entries of `ellipsoids` must have the same dimensionality")

    # Define the global axes as the source ones
    source_axes = [np.roll([1, 0, 0], i) for i in range(ndims)]

    stats = defaultdict(list)
    for ell in ellipsoids:
        ell = map_ellipsoid_to_axes(ell, source_axes)

        R = geometry.find_relative_axes_rotation(source_axes, ell.axes)
        angles = geometry.rotation_matrix_to_angles(R)
        #angles = [a if abs(a) < np.pi / 2.0 else a - np.sign(a) * np.pi for a in angles]
        stats['rotation'].append(angles)
        source_axes = ell.axes

    return stats