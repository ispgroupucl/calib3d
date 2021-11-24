# Python camera calibration and projective geometry library

This library offers several tools to ease manipulation of camera calibration, projective geometry and computations using homogenous coordinates.


## 2D and 3D points implementation

The vector used to represent 2D and 3D points are _vertical_ vectors, which are stored as 2D matrices in `numpy`. Furthemore, in _homogenous_ coordinates: a 3D point ![x,y,z](https://latex.codecogs.com/svg.latex?x,y,z) in the world is represented by a 4 element vector ![auie](https://latex.codecogs.com/svg.latex?\left[\lambda&spacex,\lambda&spacey,\lambda&spacez,\lambda&space\right]^T) where ![auie](https://latex.codecogs.com/svg.latex?$\lambda\in\mathbb{R}_0).

To simplify access to $x$ and $y$ (and $z$) coordinates of those points as well as computations in homogenous coordinates, we defined the objects `Point2D` (and `Point3D`) extending `numpy.ndarray`. Therefore, access to $y$ coordinate of `point` is `point.y` instead of `point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`, while it is still possible to use `point` with any `numpy` operators.

The construction of such point is made convenient with multiple ways of building them. Given $x$=`x`, $y$=`y` and $\lambda$=`l`, a 2D point can be created the following ways:
 - `Point2D(x, y)`
 - `Point2D(l*x, l*y, l)`

This construction handles array of points where the components are `numpy.ndarray`, `list` or `tuple`.

### Construction

The construction can also be made from `numpy` arrays of dimensions $(D,N)$ or $(D+1,N)$ in homogenous coordinates where $D$ is the space dimension ($D\in\{2,3\}$) and $N$ is the number of points (with $N\in\mathbb{N}$, including $0$ for an empty set of points). Example:
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

>>> point.T @ Point2D(2,1)
array([[0., 0., 0., 0., 0.],
       [1., 2., 3., 4., 5.],
       [1., 1., 1., 1., 1.]])
```
