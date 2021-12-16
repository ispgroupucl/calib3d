# Python camera calibration and projective geometry library

This library offers several tools to ease manipulation of camera calibration, projective geometry and computations using homogenous coordinates. 

1. [2D and 3D points implementation](#2D-and-3D-points-implementation)
2. [Camera calibration](#Camera-calibration)

Full API documentation is available in [here](https://ispgroupucl.github.io/calib3d).

## 2D and 3D points implementation

The vector used to represent 2D and 3D points are _vertical_ vectors, which are stored as 2D matrices in `numpy`. Furthemore, in _homogenous_ coordinates: a 3D point _(x,y,z)_ in the world is represented by a 4 element vector _(𝜆x,𝜆y,𝜆z,𝜆)_ where _𝜆 ∈ ℝ₀_.

To simplify access to _x_ and _y_ (and _z_) coordinates of those points as well as computations in homogenous coordinates, we defined the objects `Point2D` (and `Point3D`) extending `numpy.ndarray`. Therefore, access to y coordinate of `point` is `point.y` instead of `point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`, while it is still possible to use `point` with any `numpy` operators.


### Construction

The construction of such point is made convenient with multiple ways of building them. With 𝜆 represented as `l` a 2D point can be created provided `x` and `y` as scalar for single points, or as `numpy.ndarray`, `list` or `tuple` for array of points.
 - `Point2D(x, y)`
 - `Point2D(l*x, l*y, l)`

The construction can also be made from `numpy` arrays of dimensions _(D,N)_ or _(D+1,N)_ in homogenous coordinates where _D ∈ {2,3}_ is the space dimension and N is the number of points (which can be 0 for an empty set of points). Example:
```
>>> array = np.array([[0, 0, 0, 0, 0],  # x coordinates
                      [1, 2, 3, 4, 5]]) # y coordinates
>>> points = Point2D(array)
>>> points.x
array([0., 0., 0., 0., 0.])

>>> points.H
array([[0., 0., 0., 0., 0.],
       [1., 2., 3., 4., 5.],
       [1., 1., 1., 1., 1.]])
```

### Usage with numpy operators

The implementation extends `numpy.ndarray`. It means that all operators work out of the box.
```
>>> point = Point2D(1,2)
>>> point*2
Point2D([[2.],
         [4.]])

>>> point+5
Point2D([[6.],
         [7.]])

>>> np.linalg.norm(point)
2.23606797749979
```

## Camera calibration


This library implements a `Calib` object that represent a calibrated camera given its intrinsic and extrinsic parameters.
The object has a serie of methods to handle 3D to 2D projections, 2D to 3D liftings, image transformations and more.

### Construction

The `Calib` object requires the camera matirx `K`, the rotation matrix `R`, the translation vector `T`, the image `width` and `height` and optionally the lens distortion coefficients `kc`.

`R` is expressed using Euler angles and represents the rotation applied to the world coordinates system to obtain the camera coordinates system orientation.

`T` is the position of the origin of the world coordinates system expressed in the camera coordinates system
The _camera_ coordinates system is therefore a transformation of the _world_ coordinates systems with:
- A **rotation** defined by a rotation matrix $R$ using euler angles in a right-hand orthogonal system. The rotation is applied to the world coordinates system to obtain the camera orientation.
- A **translation** defined by a translation vector $T$ representing the position of the center of the world in the camera coordinates system !


## Reference

This library is developed and maintained by [Gabriel Van Zandycke](https://github.com/gabriel-vanzandycke). If you use this repository, please consider citing my work.

