Module calib3d.points
=====================
# Working with homogenous coordinates

The vector used to represent 2D and 3D points are vertical vectors, which are stored as 2D matrices in `numpy`.
Furthemore, in homogenous coordinates: a 3D point (x,y,z) in the world is represented by a 4 element vector
(𝜆x,𝜆y,𝜆z,𝜆) where 𝜆 ∈ ℝ₀.

To simplify access to x and y (and z) coordinates of those points as well as computations in homogenous coordinates,
we defined the types [`Point2D`](#Point2D) (and [`Point3D`](#Point3D))
extending `numpy.ndarray`. Therefore, access to y coordinate of point is `point.y` instead of
`point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`,
while it is still possible to use point with any numpy operators.

Classes
-------

`HomogeneousCoordinatesPoint(*coords)`
:   Extension of Numpy `np.ndarray` that implements generic homogenous coordinates points
    for `Point2D` and `Point3D` objects. The constructor supports multiple formats for creation
    of a single point or array of multiple points.
    
    Example with creation of `Point2D` objects, all formulations are equivalent:
    ```
    >>> x, y = 1, 2
    >>> Point2D(x, y)
    >>> Point2D(4*x, 4*y, 4)
    >>> Point2D(np.array([[x], [y]]))
    >>> Point2D(4*np.array([[x], [y], [1]]))
    Point2D([[1.],
             [2.]])
    ```

    ### Ancestors (in MRO)

    * numpy.ndarray

    ### Descendants

    * calib3d.points.Point2D
    * calib3d.points.Point3D

    ### Instance variables

    `H`
    :   Point expressed in homogenous coordinates with an homogenous component equal to `1`.
        
        Example:
        ```
        >>> p = Point3D(1,2,3)
        >>> p.H
        array([[1.],
               [2.],
               [3.],
               [1.]])
        ```

    `x`
    :   Point's x component

    `y`
    :   Point's y component

    `z`
    :   Point's z component (only valid for `Point3D` objects)

    ### Methods

    `close(self)`
    :   Copy the first point in an array of points and place it at the end of that array,
        hence "closing" the polygon defined by the initial points.
        
        TODO: add Points2D and Points3D classes

    `flatten(self)`
    :   Flatten the points.
        
        .. todo:: integrate this in the __array_ufunc__ to prevent type forwarding

    `linspace(self, num)`
    :   Linearly interpolate points in `num-1` intervals.
        
        Example:
        ```
        >>> Point2D([0,4,4],[0,0,4]).linspace(5)
        [[0. 1. 2. 3. 4. 4. 4. 4. 4. 4.]
        [0. 0. 0. 0. 0. 0. 1. 2. 3. 4.]]
        ```

    `to_int_tuple(self)`
    :   Transforms a single point to a python tuple with integer coordinates

    `to_list(self)`
    :   Transforms a single point to a python list.
        
        Raises:
            AssertionError if the object is an array of multiple points

`Point2D(*coords)`
:   Numpy representation of a single 2D point or a list of 2D points

    ### Ancestors (in MRO)

    * calib3d.points.HomogeneousCoordinatesPoint
    * numpy.ndarray

    ### Class variables

    `D`
    :

`Point3D(*coords)`
:   Numpy representation of a single 3D point or a list of 3D points

    ### Ancestors (in MRO)

    * calib3d.points.HomogeneousCoordinatesPoint
    * numpy.ndarray

    ### Class variables

    `D`
    :