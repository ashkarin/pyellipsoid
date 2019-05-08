================
PyEllipsoid
================

.. image:: https://travis-ci.org/ashkarin/pyellipsoid.svg?branch=master
    :target: https://travis-ci.org/ashkarin/pyellipsoid


**PyEllipsoid** is the package for drawing and analysis of ellipsoids in 3D volumetric images (3D arrays).


Installation
------------

The easiest way to install the latest version is by using pip::

    $ pip install pyellipsoid

You may also use Git to clone the repository and install it manually::

    $ git clone https://github.com/ashkarin/pyellipsoid.git
    $ cd pyellipsoid
    $ python setup.py install

How to use
-----
To draw an ellipsoid:

.. code-block:: python

  import numpy as np
  from pyellipsoid import drawing

  # Define an image shape, axis order is: Z, Y, X
  image_shape = (128, 128, 128)

  # Define an ellipsoid, axis order is: X, Y, Z
  ell_center = (64, 64, 64)
  ell_radii = (5, 50, 30)
  ell_angles = np.deg2rad([80, 30, 20]) # Order of rotations is X, Y, Z

  # Draw a 3D binary image containing the ellipsoid
  image = drawing.make_ellipsoid_image(image_shape, ell_center, ell_radii, ell_angles)


To compute inertia ellipsoid for given ellipsoid image:

.. code-block:: python

  import numpy as np
  from pyellipsoid import drawing, analysis

  # Draw a 3D binary image containing an ellipsoid
  image_shape = (128, 128, 128)
  ell_center = (64, 64, 64)
  ell_radii = (5, 50, 30)
  ell_angles = np.deg2rad([80, 30, 20])
  image = drawing.make_ellipsoid_image(image_shape, ell_center, ell_radii, ell_angles)

  # Compute inertia ellipsoid
  points = analysis.sample_random_points(image)
  inertia_ellipsoid = analysis.compute_inertia_ellipsoid(points)


To rotate axes and find relative rotation between the original and rotated axes:

.. code-block:: python

  import numpy as np
  from pyellipsoid import geometry

  original_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  rot_angles = np.deg2rad([0, 0, 45]) # Rotate around Z-axis by 45 deg

  # Build rotation matrix and rotate the axes
  rotm = geometry.build_rotation_matrix(*rot_angles)
  rotated_axes = np.dot(rotm, np.transpose(original_axes)).T

  # Find relative rotation
  rel_rotm = geometry.find_relative_axes_rotation(original_axes, rotated_axes)

  # Validate relative rotation matrix
  rel_rotated_axes = np.dot(rel_rotm, np.transpose(original_axes)).T
  assert(np.all(rotated_axes == rel_rotated_axes))

  # Compute rotation angles
  # Please note, that these angles might differ from rot_angles!
  rel_rot_angles = geometry.rotation_matrix_to_angles(rel_rotm)
  print(np.rad2deg(rel_rot_angles))
