from abc import ABCMeta, abstractproperty
import warnings
import numpy as np

__doc__ = r"""

# Working with homogenous coordinates

The vector used to represent 2D and 3D points are vertical vectors, which are stored as 2D matrices in `numpy`.
Furthemore, in homogenous coordinates: a 3D point (x,y,z) in the world is represented by a 4 element vector
(𝜆x,𝜆y,𝜆z,𝜆) where 𝜆 ∈ ℝ₀.

To simplify access to x and y (and z) coordinates of those points as well as computations in homogenous coordinates,
we defined the types [`Point2D`](#Point2D) (and [`Point3D`](#Point3D))
extending `numpy.ndarray`. Therefore, access to y coordinate of point is `point.y` instead of
`point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`,
while it is still possible to use point with any numpy operators.

"""


class HomogeneousCoordinatesPoint(np.ndarray, metaclass=ABCMeta):
    """ Extension of Numpy `np.ndarray` that implements generic homogenous coordinates points
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
    """
    def __new__(cls, *coords):
        if len(coords) == 1:
            if isinstance(coords, list):
                coords = np.hstack(coords)
            else:
                coords = coords[0]
        array = np.array(coords) if isinstance(coords, (tuple, list)) else coords
        invalid_shape_message = "Invalid input shape:\n" \
            "Expected a 2D np.array of shape ({l1},N) or (N,{l1},1) in non-homogenous coordinates\n" \
            "                       or shape ({l2},N) or (N,{l2},1) in homogenous coordinates\n" \
            "Received a np.array of shape {shape}".format(l1=cls.D, l2=cls.D+1, shape=array.shape)
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        elif len(array.shape) == 2:
            pass
        elif len(array.shape) == 3:
            array = array[..., 0].T
        else:
            raise ValueError(invalid_shape_message)

        if array.shape[0] == cls.D: # point(s) given in non-homogenous coordinates
            pass
        elif array.shape[0] == cls.D+1: # point(s) given in homogenous coordinates
            array = array[0:cls.D,:]/array[cls.D,:]
        elif array.shape[0] == 0: # point given from an empty list should be an empty point
            array = np.empty((cls.D,0))
        else:
            raise ValueError(invalid_shape_message)
        return array.astype(np.float64).view(cls)
    # def __array_ufunc__():
    #    TODO
    # def __array_wrap__(self, out_arr, context=None):
    #     return super().__array_wrap__(self, out_arr, context)
    @property
    @abstractproperty
    def _coord_names(self):
        raise NotImplementedError

    x = property(fget=lambda self: self._get_coord(0), fset=lambda self, value: self._set_coord(0, value), doc="Point's x component")
    y = property(fget=lambda self: self._get_coord(1), fset=lambda self, value: self._set_coord(1, value), doc="Point's y component")
    z = property(fget=lambda self: self._get_coord(2), fset=lambda self, value: self._set_coord(2, value), doc="Point's z component (only valid for `Point3D` objects)")

    @property
    def H(self):
        """ Point expressed in homogenous coordinates with an homogenous component equal to `1`.

        Example:
        ```
        >>> p = Point3D(1,2,3)
        >>> p.H
        array([[1.],
               [2.],
               [3.],
               [1.]])
        ```
        """
        return np.vstack((self, np.ones((1, self.shape[1]))))
    # @property
    # def D(self):
    #     """ Returns the number of spacial dimensions """
    #     return len(self._coord_names)
    # /!\ a different getitem gives too much trouble with all numpy operator
    #def __getitem__(self, i):
    #    if isinstance(i, int):
    #        return self.__class__(super().__getitem__((slice(None), i)))
    #    return super().__getitem__(i)

    # /!\ iter may conflict with numpy array getitem.
    def __iter__(self):
        return (self.__class__(self[:,i:i+1]) for i in range(self.shape[1]))

    def to_list(self):
        """ Transforms a single point to a python list.

        Raises:
            AssertionError if the object is an array of multiple points
        """
        assert self.shape[1] == 1, "to_list() method can only be used on single point {}".format(self.__class__.__name__)
        return self[:,0].flatten().tolist()

    def flatten(self):
        """ Flatten the points.

        .. todo:: integrate this in the __array_ufunc__ to prevent type forwarding
        """
        return np.asarray(super().flatten())

    def to_int_tuple(self):
        """ Transforms a single point to a python tuple with integer coordinates
        """
        return tuple(int(x) for x in self.to_list())

    def linspace(self, num):
        """ Linearly interpolate points in `num-1` intervals.

            Example:
            ```
            >>> Point2D([0,4,4],[0,0,4]).linspace(5)
            [[0. 1. 2. 3. 4. 4. 4. 4. 4. 4.]
            [0. 0. 0. 0. 0. 0. 1. 2. 3. 4.]]
            ```
        """
        return np.transpose(np.linspace(self[:,:-1], self[:,1:], num), axes=(1,2,0)).reshape(len(self._coord_names),-1)

    def close(self):
        """ Copy the first point in an array of points and place it at the end of that array,
            hence "closing" the polygon defined by the initial points.

            TODO: add Points2D and Points3D classes
        """
        assert self.shape[1] > 1, f"Invalid use of 'close' method: points' shape '{self.shape}' expected to be > 1 in the second dimension"
        return self.__class__(np.hstack((self, self[:,0:1])))

    _get_coord = lambda self, i:        np.asarray(super().__getitem__((i,0)))       if self.shape[1] == 1 else np.asarray(super().__getitem__(i))
    _set_coord = lambda self, i, value:            super().__setitem__((i,0), value) if self.shape[1] == 1 else            super().__setitem__((i), value)


class Point2D(HomogeneousCoordinatesPoint):
    """ Numpy representation of a single 2D point or a list of 2D points
    """
    D = 2
    _coord_names = ("x","y")

class Point3D(HomogeneousCoordinatesPoint):
    """ Numpy representation of a single 3D point or a list of 3D points
    """
    D = 3
    _coord_names = ("x","y","z")
    @property
    def V(self):
        array = self.H
        array[-1] = 0
        return VanishingPoint(array)

class VanishingPoint(Point3D):
    """ Object allowing representation of Vanishing point (with null homogenous
        coordinate). Only the `H` attribute should be used. Handle with care.
    """
    def __new__(cls, array):
        warnings.warn("Vanishing Point feature has not yet been fully tested")
        obj = array.astype(np.float64).view(cls)
        obj.array = array
        return obj
    @property
    def H(self):
        return self.array
    def __getattribute__(self, attr_name):
        if attr_name not in ("H", "__array_finalize__", "array", "shape", "size", "ndim", "x", "y", "z", "_get_coord", "close", "__class__", "astype", "view"):
            raise AttributeError(f"VanishingPoint has no `{attr_name}` attribute.")
        return super().__getattribute__(attr_name)
