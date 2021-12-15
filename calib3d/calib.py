import pickle
import numpy as np
import cv2
from .points import Point2D, Point3D

class Calib():
    def __init__(self, *, width: int, height: int, T: np.ndarray, R: np.ndarray, K: np.ndarray, kc=None, **_) -> None:
        """ Represents a calibrated camera.

        Args:
            width (int): image width
            height (int): image height
            T (np.ndarray): translation vector
            R (np.ndarray): rotation matrix
            K (np.ndarray): camera matrix holding intrinsic parameters
            kc (np.ndarray, optional): lens distortion coefficients. Defaults to None.
        """
        self.width = int(width)
        self.height = int(height)
        self.T = T
        self.K = K
        self.kc = np.array((kc if kc is not None else [0,0,0,0,0]), dtype=np.float64)
        self.R = R
        self.C = Point3D(-R.T@T)
        self.P = self.K @ np.hstack((self.R, self.T))
        self.Pinv = np.linalg.pinv(self.P)
        self.Kinv = np.linalg.pinv(self.K)
    def update(self, **kwargs) -> 'Calib':
        """ Creates another Calib object with the given keyword arguments updated
            Args:
                **kwargs : Any of the arguments of `Calib`. Other arguments remain unchanged.
            Returns:
                A new Calib object
        """
        return self.__class__(**{**self.dict, **kwargs})

    @classmethod
    def from_P(cls, P, width, height) -> 'Calib':
        """ Create a `Calib` object from a given projection matrix `P` and image dimensions `width` and `height`.
            Args:
                P (np.ndarray) : a 3x4 projection matrix
                width (int) : image width
                height (int) : image height
            Returns:
                A Calib object
        """
        K, R, T, Rx, Ry, Rz, angles = cv2.decomposeProjectionMatrix(P) # pylint: disable=unused-variable
        return cls(K=K, R=R, T=Point3D(-R@Point3D(T)), width=width, height=height)

    @classmethod
    def load(cls, filename) -> 'Calib':
        """ Loads a Calib object from a file (using the pickle library)
            Args:
                filename (str) : the file that stores the Calib object
            Returns:
                The `Calib` object previously saved in `filename`.
        """
        with open(filename, "rb") as f:
            return cls(**pickle.load(f))

    @property
    def dict(self) -> dict:
        """ Gets a dictionnary representing the calib object (allowing easier serialization)
        """
        return {k: getattr(self, k) for k in self.__dict__}

    def dump(self, filename) -> None:
        """ Saves the current calib object to a file (using the pickle library)
            Args:
                filename (str) : the file that will store the calib object
        """
        with open(filename, "wb") as f:
            pickle.dump(self.dict, f)

    def _project_3D_to_2D_cv2(self, point3D: Point3D):
        raise BaseException("This function gives errors when rotating the calibration...")
        return Point2D(cv2.projectPoints(point3D, cv2.Rodrigues(self.R)[0], self.T, self.K, self.kc.astype(np.float64))[0][:,0,:].T)

    def project_3D_to_2D(self, point3D: Point3D) -> Point2D:
        """ Using the calib object, project a 3D point in the 2D image space.
            Args:
                point3D (Point3D) : the 3D point to be projected
            Returns:
                The point in the 2D image space on which point3D is projected by calib
        """
        assert isinstance(point3D, Point3D), "Wrong argument type '{}'. Expected {}".format(type(point3D), Point3D)
        point2D_H = self.P @ point3D.H # returns a np.ndarray object
        point2D_H[2] = point2D_H[2] * np.sign(point2D_H[2]) # correct projection of points being projected behind the camera
        point2D = Point2D(point2D_H)
        # avoid distortion of points too far away
        excluded_points = np.logical_or(np.logical_or(point2D.x < -self.width, point2D.x > 2*self.width),
                                        np.logical_or(point2D.y < -self.height, point2D.y > 2*self.height))
        return Point2D(np.where(excluded_points, point2D, self.distort(point2D)))

    def project_2D_to_3D(self, point2D: Point2D, Z: float) -> Point3D:
        """ Using the calib object, project a 2D point in the 3D image space.
            Args:
                point2D (Point2D) : the 2D point to be projected
                Z (float) : the Z coordinate of the 3D point
            Returns:
                The point in the 3D world for which the z=`Z` and that projects on `point2D`.
        """
        assert isinstance(point2D, Point2D), "Wrong argument type '{}'. Expected {}".format(type(point2D), Point2D)
        point2D = self.rectify(point2D)
        X = Point3D(self.Pinv @ point2D.H)
        d = (X - self.C)
        return find_intersection(self.C, d, Point3D(0, 0, Z), np.array([[0, 0, 1]]).T)

    def distort(self, point2D: Point2D) -> Point2D:
        """ Applies lens distortions to the given `point2D`.
        """
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

    def rectify(self, point2D: Point2D) -> Point2D:
        """ Removes lens distortion to the given `Point2D`.
        """
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

    def crop(self, x_slice, y_slice) -> 'Calib':
        x0 = x_slice.start
        y0 = y_slice.start
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        T = np.array([[1, 0,-x0], [0, 1,-y0], [0, 0, 1]])
        return self.update(width=width, height=height, K=T@self.K)

    def scale(self, output_width, output_height) -> 'Calib':
        sx = output_width/self.width
        sy = output_height/self.height
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return self.update(width=output_width, height=output_height, K=S@self.K)

    def flip(self) -> 'Calib':
        F = np.array([[-1, 0, self.width-1], [0, 1, 0], [0, 0, 1]])
        return self.update(K=F@self.K)

    def rotate(self, angle) -> 'Calib':
        if angle == 0:
            return self
        A, new_width, new_height = compute_rotate(self.width, self.height, angle)
        return self.update(K=A@self.K, width=new_width, height=new_height)

    def compute_length2D(self, length3D: float, point3D: Point3D) -> np.ndarray:
        """ Returns the length in pixel of a 3D length at point3D
        """
        assert np.isscalar(length3D), f"This function expects a scalar `length3D` argument. Received {length3D}"
        point3D_c = Point3D(np.hstack((self.R, self.T)) @ point3D.H)  # Point3D expressed in camera coordinates system
        point3D_c.x += length3D # add the 3D length to one of the componant
        point2D = self.distort(Point2D(self.K @ point3D_c)) # go in the 2D world
        length = np.linalg.norm(point2D - self.project_3D_to_2D(point3D), axis=0)
        return length#float(length) if point3D.shape[1] == 1 else length

    def projects_in(self, point3D: Point3D) -> np.ndarray:
        """ Check wether point3D projects into the `Calib` object.
            Returns `True` where for points that projects in the image and `False` otherwise.
        """
        point2D = self.project_3D_to_2D(point3D)
        cond = np.stack((point2D.x >= 0, point2D.y >= 0, point2D.x <= self.width, point2D.y <= self.height))
        return np.all(cond, axis=0)

def find_intersection(C: Point3D, d, P: Point3D, n) -> Point3D:
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

def parameters_to_affine_transform(angle: float, x_slice: slice, y_slice: slice,
    output_shape: tuple, do_flip: bool=False):
    """ Compute the affine transformation matrix that correspond to a
        - horizontal flip if `do_flip` is `True`, followed by a
        - rotation of `angle` degrees around image center, followed by a
        - crop defined by `x_slice` and `y_slice`, followed by a
        - scale to recover `output_shape`.
    """
    assert not do_flip, "There is a bug with random flip"
    R = np.eye(3)
    center = ((y_slice.start + y_slice.stop)/2, (x_slice.start + x_slice.stop)/2)
    R[0:2,:] = cv2.getRotationMatrix2D(center, angle, 1.0)

    x0 = x_slice.start
    y0 = y_slice.start
    width = x_slice.stop - x_slice.start
    height = y_slice.stop - y_slice.start
    T = np.array([[1, 0,-x0], [0, 1,-y0], [0, 0, 1]])

    sx = output_shape[0]/width
    sy = output_shape[1]/height
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    f = np.random.randint(0,2)*2-1 if do_flip else 1 # random sample in {-1,1}
    F = np.array([[f, 0, 0], [0, 1, 0], [0, 0, 1]])

    return S@T@R@F


def compute_rotation_matrix(point3D: Point3D, camera3D: Point3D):
    """ Computes the rotation matrix of a camera in `camera3D` pointing
        towards the point `point3D`. Both are expressed in word coordinates.
        The convention is that Z is pointing down.
        Credits: François Ledent
    """
    point3D = camera3D - point3D
    x, y, z = point3D.x, point3D.y, point3D.z
    d = np.sqrt(x**2 + y**2)
    D = np.sqrt(x**2 + y**2 + z**2)
    h = d / D
    l = z / D

    # camera basis `B` expressed in the world basis `O`
    _x = np.array([y / d, -x / d, 0])
    _y = np.array([- l * x / d, - l * y / d, h])
    _z = np.array([- h * x / d, - h * y / d, -l])
    B = np.stack((_x, _y, _z), axis=-1)

    # `O = R @ B` (where `O` := `np.identity(3)`)
    R = B.T # inv(B) == B.T since R is a rotation matrix
    return R
