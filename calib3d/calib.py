from abc import ABCMeta, abstractproperty
import pickle
import numpy as np
import cv2

class HomogeneousCoordinatesPoint(np.ndarray, metaclass=ABCMeta):
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
            "Received a np.array of shape {shape}".format(l1=len(cls.coord_names), l2=len(cls.coord_names)+1, shape=array.shape)
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        elif len(array.shape) == 2:
            pass
        elif len(array.shape) == 3:
            array = array[..., 0].T
        else:
            raise ValueError(invalid_shape_message)

        if array.shape[0] == len(cls.coord_names): # point(s) given in non-homogenous coordinates
            pass
        elif array.shape[0] == len(cls.coord_names)+1: # point(s) given in homogenous coordinates
            array = array[0:len(cls.coord_names),:]/array[len(cls.coord_names),:]
        else:
            raise ValueError(invalid_shape_message)
        return array.astype(np.float64).view(cls)
    @property
    @abstractproperty
    def coord_names(self):
        raise NotImplementedError
    @property
    def H(self):
        return np.vstack((self, np.ones((1, self.shape[1]))))

    # /!\ a different getitem gives too much trouble with all numpy operator
    #def __getitem__(self, i):
    #    if isinstance(i, int):
    #        return self.__class__(super().__getitem__((slice(None), i)))
    #    return super().__getitem__(i)

    # /!\ iter may conflict with numpy array getitem.
    def __iter__(self):
        return (self.__class__(self[:,i]) for i in range(self.shape[1]))

    def to_list(self):
        assert self.shape[1] == 1, "to_list() method can only be used on single point {}".format(self.__class__.__name__)
        return self[:,0].flatten()

    def flatten(self):
        return np.array(super().flatten())

    def to_int_tuple(self):
        return tuple(int(x) for x in self.to_list())

    get_coord = lambda self, i:        super().__getitem__((i,0))        if self.shape[1] == 1 else super().__getitem__(i)
    set_coord = lambda self, i, value: super().__setitem__((i,0), value) if self.shape[1] == 1 else super().__setitem__((i), value)

    x = property(fget=lambda self: self.get_coord(0), fset=lambda self, value: self.set_coord(0, value))
    y = property(fget=lambda self: self.get_coord(1), fset=lambda self, value: self.set_coord(1, value))
    z = property(fget=lambda self: self.get_coord(2), fset=lambda self, value: self.set_coord(2, value))

class Point3D(HomogeneousCoordinatesPoint):
    coord_names = ("x","y","z")

class Point2D(HomogeneousCoordinatesPoint):
    coord_names = ("x","y")

class Calib():
    def __init__(self, *, width: int, height: int, T: np.ndarray, R: np.ndarray, K: np.ndarray, kc=np.zeros((5,1)), **_) -> None:
        self.width = np.int(width)
        self.height = np.int(height)
        self.T = T
        self.K = K
        self.kc = np.array(kc, dtype=np.float64)
        self.R = R
        self.C = Point3D(-R.T@T)
        self.P = self.K @ np.hstack((self.R, self.T))
        self.Pinv = np.linalg.pinv(self.P)
        self.Kinv = np.linalg.pinv(self.K)

    def update(self, **kwargs):
        """ Creates another Calib object with the given keyword arguments updated
            Arguments:
                Any of the arguments of __init__
            Returns:
                A new Calib object
        """
        return self.__class__(**{**self.dict, **kwargs})

    @classmethod
    def from_P(cls, P, width, height):
        K, R, T, Rx, Ry, Rz, angles = cv2.decomposeProjectionMatrix(P) # pylint: disable=unused-variable
        return cls(K=K, R=R, T=Point3D(-R@Point3D(T)), width=width, height=height)

    @classmethod
    def load(cls, filename):
        """ Loads a Calib object from a file (using the pickle library)
            Argument:
                filename   - the file that stores the Calib object
            Returns:
                The Calib object
        """
        with open(filename, "rb") as f:
            return cls(**pickle.load(f))

    @property
    def dict(self):
        """ Gets a dictionnary representing the calib object (allowing easier serialization)
        """
        return {k: getattr(self, k) for k in self.__dict__}

    def dump(self, filename):
        """ Saves the current calib object to a file (using the pickle library)
            Argument:
                filename    - the file that will store the calib object
        """
        with open(filename, "wb") as f:
            pickle.dump(self.dict, f)

    def project_3D_to_2D_cv2(self, point3D: Point3D):
        """ Using the calib object, project a 3D point in the 2D image space.
            Arguments:
                point3D   - the 3D point to be projected
            Returns:
                The point in the 2D image space on which point3D is projected by calib
        """
        raise BaseException("This function gives errors when rotating the calibration...")
        return Point2D(cv2.projectPoints(point3D, cv2.Rodrigues(self.R)[0], self.T, self.K, self.kc.astype(np.float64))[0][:,0,:].T)

    def project_3D_to_2D(self, point3D: Point3D):
        assert isinstance(point3D, Point3D), "Wrong argument type '{}'. Expected {}".format(type(point3D), Point3D)
        point2D_H = self.P @ point3D.H # returns a np.ndarray object
        point2D_H[2] = point2D_H[2] * np.sign(point2D_H[2]) # correct projection of points being projected behind the camera
        point2D = Point2D(point2D_H)
        # avoid distortion of points too far away
        excluded_points = np.logical_or(np.logical_or(point2D.x < -self.width, point2D.x > 2*self.width),
                                        np.logical_or(point2D.y < -self.height, point2D.y > 2*self.height))
        return Point2D(np.where(excluded_points, point2D, self.distort(point2D)))

    def project_2D_to_3D(self, point2D: Point2D, Z: float):
        """ Using the calib object, project a 2D point in the 3D image space.
            Arguments:
                point2D    - the 2D point to be projected
                Z          - the Z coordinate of the 3D point
            Returns:
                The point in the 3D world for which the z=Z and that projects on point2D
        """
        assert isinstance(point2D, Point2D), "Wrong argument type '{}'. Expected {}".format(type(point2D), Point2D)
        point2D = self.rectify(point2D)
        X = Point3D(self.Pinv @ point2D.H)
        d = (X - self.C)
        return find_intersection(self.C, d, Point3D(0, 0, Z), np.array([[0, 0, 1]]).T)

    def distort(self, point2D: Point2D):
        if np.any(self.kc):
            rad1, rad2, tan1, tan2, rad3 = self.kc.flatten()
            # Convert image coordinates to camera coordinates (with z=1 which is the projection plane)
            point2D = Point2D(self.Kinv @ point2D.H)

            r2 = point2D.x*point2D.x + point2D.y*point2D.y
            delta = 1 + rad1*r2 + rad2*r2*r2 + rad3*r2*r2*r2
            dx = np.array([
                2*tan1*point2D.x*point2D.y + tan2*(r2 + 2*point2D.x*point2D.x),
                2*tan2*point2D.x*point2D.y + tan1*(r2 + 2*point2D.y*point2D.y)
            ]).reshape(2, -1)

            point2D = point2D*delta + dx
            # Convert camera coordinates to pixel coordinates
            point2D = Point2D(self.K @ point2D.H)
        return point2D

    def rectify(self, point2D: Point2D):
        if np.any(self.kc):
            rad1, rad2, tan1, tan2, rad3 = self.kc.flatten()
            point2D = Point2D(self.Kinv @ point2D.H) # to camera coordinates

            r2 = point2D.x*point2D.x + point2D.y*point2D.y
            delta = 1 + rad1*r2 + rad2*r2*r2 + rad3*r2*r2*r2
            dx = np.array([
                2*tan1*point2D.x*point2D.y + tan2*(r2 + 2*point2D.x*point2D.x),
                2*tan2*point2D.x*point2D.y + tan1*(r2 + 2*point2D.y*point2D.y)
            ]).reshape(2, -1)

            point2D = (point2D - dx)/delta
            point2D = Point2D(self.K @ point2D.H) # to pixel coordinates
        return point2D

    def crop(self, x_slice, y_slice):
        x0 = x_slice.start
        y0 = y_slice.start
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        T = np.array([[1, 0,-x0], [0, 1,-y0], [0, 0, 1]])
        return self.update(width=width, height=height, K=T@self.K)

    def scale(self, output_width, output_height):
        sx = output_width/self.width
        sy = output_height/self.height
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return self.update(width=output_width, height=output_height, K=S@self.K)

    def flip(self):
        F = np.array([[-1, 0, self.width-1], [0, 1, 0], [0, 0, 1]])
        return self.update(K=F@self.K)

    def rotate(self, angle):
        if angle == 0:
            return self
        A, new_width, new_height = compute_rotate(self.width, self.height, angle)
        return self.update(K=A@self.K, width=new_width, height=new_height)

    def compute_length2D(self, length3D: int, point3D: Point3D):
        point3D_c = Point3D(np.hstack((self.R, self.T)) @ point3D.H)  # Point3D expressed in camera coordinates system
        point3D_c.x += length3D # add the 3D length to one of the componant
        point2D = self.distort(Point2D(self.K @ point3D_c)) # go in the 2D world
        return np.linalg.norm(point2D - self.project_3D_to_2D(point3D))

    def projects_in(self, point3D):
        point2D = self.project_3D_to_2D(point3D)
        if point2D.x > 0 and point2D.y > 0 and point2D.x < self.width and point2D.y < self.height:
            return True
        return False

def find_intersection(C: Point3D, d, P: Point3D, n):
    """ Finds the intersection between a line and a plane.
        Arguments:
            C - a Point3D of a point on the line
            d - the direction-vector of the line
            P - a Point3D on the plane
            n - the normal vector of the plane
        Returns the Point3D at the intersection between the line and the plane.
    """
    d = d/np.linalg.norm(d, axis=0)
    dist = ((P-C).T @ n) / (d.T @ n)  # Distance between plane z=Z and camera
    return Point3D(C + dist.T*d)

def compute_rotate(width, height, angle):
    """ Computes rotation matrix and new width and height for a rotation of angle degrees of a widthxheight image.
    """
    # Convert the OpenCV 3x2 rotation matrix to 3x3
    R = np.eye(3)
    R[0:2,:] = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    R2D = R[0:2,0:2]

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        np.array([-width/2,  height/2]) @ R2D,
        np.array([ width/2,  height/2]) @ R2D,
        np.array([-width/2, -height/2]) @ R2D,
        np.array([ width/2, -height/2]) @ R2D
    ]

    # Find the size of the new image
    right_bound = max([pt[0] for pt in rotated_coords])
    left_bound = min([pt[0] for pt in rotated_coords])
    top_bound = max([pt[1] for pt in rotated_coords])
    bot_bound = min([pt[1] for pt in rotated_coords])

    new_width = int(abs(right_bound - left_bound))
    new_height = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    T = np.array([
        [1, 0, new_width/2 - width/2],
        [0, 1, new_height/2 - height/2],
        [0, 0, 1]
    ])
    return T@R, new_width, new_height


def rotate_image(image, angle):
    """ Rotates an image around its center by the given angle (in degrees).
    The returned image will be large enough to hold the entire new image, with a black background
    """
    height, width = image.shape[0:2]
    A, new_width, new_height = compute_rotate(width, height, angle)
    return cv2.warpAffine(image, A[0:2,:], (new_width, new_height), flags=cv2.INTER_LINEAR)
