__doc__ = r"""
This library offers several tools for manipulation of calibrated cameras, projective geometry and computations using homogenous coordinates. 

Camera calibration allows to determine the relation between the camera's pixels (2D coordinates) and points in the real world
(3D coordinates). It implies computation using homogenous coordinates. This python library aims at simplifying implementations
of projective geometry computations, building on top of `numpy` and `cv2`.

The different modules are document here:

- [Computations with homogenous coordinates](./points.html)
- [Projective geometry and calibrated cameras](./calib.html)

"""

from .points import Point3D, Point2D
from .calib import Calib, parameters_to_affine_transform, compute_rotation_matrix, line_plane_intersection
