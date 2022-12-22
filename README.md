# Python camera calibration and projective geometry library

This library offers several tools for manipulation of calibrated cameras, projective geometry and computations using homogenous coordinates.

Camera calibration allows to determine the relation between the camera's pixels (2D coordinates) and points in the real world
(3D coordinates). It implies computation using homogenous coordinates. This python library aims at simplifying implementations
of projective geometry computations, building on top of `numpy` and `cv2`.


## Installation

Installation using the package manager pip

```shell
pip install calib3d
```

## Usage

Full API documentation is available in [here](https://ispgroupucl.github.io/calib3d).


### 2D and 3D points implementation

The `Point2D` (and `Point3D`) class represent 2D (and 3D) points extending `numpy.ndarray`. Access to y coordinate of `point` is `point.y`, and access to homogenous coordinates is made easy with `point.H`, while it is still possible to use `point` with any `numpy` operators.

```python
>>> Point2D(1,2) == Point2D(2,4,2)
True

>>> points = Point2D(np.array([[0, 0, 1, 2, 3],   # x coordinates
                               [1, 2, 3, 4, 5]])) # y coordinates
>>> points.x
array([0., 0., 1., 2., 3.])

>>> points.H
array([[0., 0., 1., 2., 3.],
       [1., 2., 3., 4., 5.],
       [1., 1., 1., 1., 1.]])
```


### Camera calibration

The `Calib` class represents a calibrated camera. It has a serie of methods to handle 3D to 2D projections, 2D to 3D liftings, image transformations, and more.

```python
>>> import numpy as np
>>> from calib3d import Calib, Point3D, compute_rotation_matrix
>>> f = 0.035                                      # lens focal length [m]      35 mm lens
>>> w, h = np.array([4000, 3000])                  # sensor size       [px.px]  12 Mpx sensor
>>> d = w/0.01                                     # pixel density     [px.m⁻¹] with a 1 cm sensor width
>>> K = np.array([[ d*f,  0 , w/2 ],               # Camera matrix (intrinsic parameters)
...               [  0 , d*f, h/2 ],
...               [  0 ,  0 ,  1  ]])
>>> C = Point3D(10,10,10)                          # Camera position in the 3D space
>>> R = compute_rotation_matrix(Point3D(0,0,0), C) # Camera pointing towards origin
>>> calib = Calib(K=K, T=-R@C, R=R, width=w, height=h)
>>> calib.project_3D_to_2D(Point3D(0,0,0))
Point2D([[2000.],
         [1500.]])
```

Cropping or scaling a calib is made easy with the following operations (for more operations, check the documentation)
```python
>>> new_calib = calib.crop(x_slice=slice(10, 110, None), y_slice=slice(500, 600, None))
>>> new_calib = calib.scale(output_width=2000, output_height=1500)
```

Other useful methods
```python
>>> calib.projects_in(Point3D(0, 20, 20))
False

>>> calib.compute_length2D(Point3D(0, 0, 0), .42)  # Number of pixels that represent a length of .42 in the 3D space
array([339.48195828])
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors

This library is developed and maintained by [Gabriel Van Zandycke](https://github.com/gabriel-vanzandycke). If you use this repository, please consider citing my work.

