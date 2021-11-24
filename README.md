# Python camera calibration and projective geometry library

This library offers several tools to ease manipulation of camera calibration, projective geometry and computations using homogenous coordinates.


## 2D and 3D points implementation

The vector used to represent 2D and 3D points are _vertical_ vectors, which are stored as 2D matrices in `numpy`. Furthemore, in _homogenous_ coordinates: a 3D point ![`(x,y,z)`](https://latex.codecogs.com/svg.latex?x,y,z) in the world is represented by a 4 element vector ![`(lambda*x,lambda*y,lambda*z,lambda)`](https://latex.codecogs.com/svg.latex?\left[\lambda&space;x,\lambda&space;y,\lambda&space;z,\lambda&space;\right]^T) where ![`lambda in R_0`](https://latex.codecogs.com/svg.latex?\lambda\in\mathbb{R}_0).

To simplify access to ![`x`](https://latex.codecogs.com/svg.latex?x) and ![`y`](https://latex.codecogs.com/svg.latex?y) (and ![`z`](https://latex.codecogs.com/svg.latex?z)) coordinates of those points as well as computations in homogenous coordinates, we defined the objects `Point2D` (and `Point3D`) extending `numpy.ndarray`. Therefore, access to ![`y`](https://latex.codecogs.com/svg.latex?y) coordinate of `point` is `point.y` instead of `point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`, while it is still possible to use `point` with any `numpy` operators.

### Construction

The construction of such point is made convenient with multiple ways of building them. Given ![`x`](https://latex.codecogs.com/svg.latex?x)=`x`, ![`y`](https://latex.codecogs.com/svg.latex?y)=`y` and ![`lambda`](https://latex.codecogs.com/svg.latex?\lambda)=`l`, a 2D point can be created the following ways:
 - `Point2D(x, y)`
 - `Point2D(l*x, l*y, l)`

This construction handles array of points where the components are `numpy.ndarray`, `list` or `tuple`.

The construction can also be made from `numpy` arrays of dimensions ![`(D,N)`](https://latex.codecogs.com/svg.latex?(D,N)) or ![`(D+1,N)`](https://latex.codecogs.com/svg.latex?(D+1,N)) in homogenous coordinates where ![`D in {2,3}`](https://latex.codecogs.com/svg.latex?D\in\left{2,3\right}) is the space dimension and ![`N in N`](https://latex.codecogs.com/svg.latex?N\in\mathbb{N}) is the number of points (which can be 0 for an empty set of points). Example:
```
>>> array = np.array([[0, 0, 0, 0, 0],  # x coordinates
                      [1, 2, 3, 4, 5]]) # y coordinates
>>> points = Point2D(array)
>>> points
Point2D([[0., 0., 0., 0., 0.],
         [1., 2., 3., 4., 5.]])

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
