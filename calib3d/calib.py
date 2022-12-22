import pickle
import numpy as np
import cv2
from .points import Point2D, Point3D

__doc__ = r"""

# Theoretical introduction

Camera calibration is based on the pinhole camera model that approximate how lights travels between the scene and the camera sensor. There are two sets of parameters:

- The **extrinsic parameters** define the camera position and orientation relative to the world coordinates system.
- The **Intrinsic parameters** define the camera sensor and lens parameters.

There are 3 different coordinates systems:

- The world 3D coordinates system
- The camera 3D coordinates system
- The 2D pixel positions where 3D positions are being projected.


## Extrinsic parameters
The extrinsic parameters defines the transformations from the world 3D coordinates \(\left[x_O,y_O,z_O\right]^T\) to the camera 3D coordinates \(\left[x_C,y_C,z_C\right]^T\). The camera 3D coordinates system has the following **conventions**:

- The point \((0,0,0)\) is the center of projection of the camera and is called the *principal point*.
- The \(z\) axis of the camera points *towards the scene*, the \(x\) axis is along the sensor width pointing towards the right, and the \(y\) axis is along the sensor height pointing towards the bottom.

The camera coordinates system is therefore a *transformation* of the world coordinates systems with:

- A **rotation** defined by a rotation matrix \(R\) using euler angles in a right-hand orthogonal system. The rotation is applied to the world coordinates system to obtain the camera orientation.
- A **translation** defined by a translation vector \(T\) representing the position of the center of the world in the camera coordinates system !

Hence,

$$\lambda\left[\begin{matrix}x_C\\ y_C\\ z_C\\ 1\end{matrix}\right] = \left[\begin{matrix}
R_{3\times 3} & T_{3\times 1}\\{\bf 0}_{1\times 3}&1
\end{matrix}\right] \left[\begin{matrix}x_O\\y_O\\z_O\\1\end{matrix}\right]$$

**Important notes:**

- The rotation matrix represents a passive (or alias) transformation because it's the coordinates system that rotates and not the objects.
- Euler angles define a 3D rotation starting with a rotation around \(x\) followed by a rotation around \(y\) followed by a rotation around \(z\) (the order matters).
- If \(T\) is expressed in the camera coordinate system, the position of the camera expressed in world coordinates system is \(C:=-R^{-1}T = -R^{T}T\) (since \(R\) is a rotation matrix).


## Intrinsic parameters

The intrinsic parameters defines the transformation between the 3D coordinates relative to the camera center \(\left[x_C,y_C,z_C\right]^T\) and the 2D coordinates in the camera sensor \(\left[i,j\right]^T\). This transformation is called a *projection* and includes:

- the scale produced by the focal length, with \(f\) being the distance between the camera center and the plane on which the image is projected.
- the scale factors \((m_x,m_y)\) relating pixels units to distance units (usually \(m_x=m_y\) because pixels are squares).
- the translation from the camera _principal point_ to a top-left origin, with \((u_0,v_0)\) being the position of the *principal point* expressed in the image coordinates system.
- a skew coefficient \(\gamma\) between the \(x\) and \(y\) axis in the sensor (usually \(\gamma=0\) because pixels are squares).

Those transformations can be aggregated in one single matrix called **camera matrix**:

$$K := \left[\begin{matrix}f\cdot m_x & \gamma & u_0 \\ 0 & f\cdot m_y & v_0 \\ 0 & 0 & 1\end{matrix}\right]$$

Therefore,
$$\lambda\left[\begin{matrix}i\\ j\\ 1\end{matrix}\right]= \left[\begin{matrix}K_{3\times 3}&{\bf 0}_{3\times 1}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\\1\end{matrix}\right]$$

**Notes:**

- The width and height of the image are to be added to those parameters and delimits the sensor width and height in pixels.
- When applying the **direct** projection of a given 3D point, different values of \(\lambda\) will always give the **same** 2D point.
- When applying the **inverse** projection on a given 2D point, different values of \(\lambda\) will give **different** 3D points.

This is obvious when simplifying the relation between the two points (The column \({\bf 0}_{3\times 1}\) cancels the homogenous component of the 3D point):

$$\lambda\left[\begin{matrix}i\\j\\1\end{matrix}\right]= \left[\begin{matrix}K_{3\times 3}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\end{matrix}\right]$$

The 2D vector in homogenous coordinates is not affected by the value of \(\lambda\), while the 3D vector is.



## Projection model

Therefore, by combining

- the transformation from the world coordinates system to the camera coordinates system (defined by \(R\) and \(T\))
- with the projection from the camera coordinates system to the image pixels (defined by \(K\)),

We have a projection model allowing to compute the coordinates of a 2D point in the image \(\left(i,j\right)\) from a 3D point in the real world \(\left(x,y,z\right)\) described by the matrix \(P\):
$$P := \left[\begin{matrix}K_{3\times 3}&{\bf 0}_{3\times 1}\end{matrix}\right] \left[\begin{matrix}R_{3\times 3}&T_{3\times 1}\\{\bf 0}_{1\times 3}&1\end{matrix}\right]=K_{3\times 3}\left[\begin{matrix}R_{3\times 3}&T_{3\times 1}\end{matrix}\right]$$

The opposite operation requires to invert \(P\) and is done by pseudo-inverse inversion because \(P\) is rectangular.


## Limitations
We need a model of the distortions brought by the lens:

- **radial** distortion cause lines far from principal point to look distorted.
- **tangential** distortion: occur when lens is not prefectly align with the \(z\) axis of the camera.

Many lens distortion models exist. Here, we will use the model used in [opencv](https://docs.opencv.org/3.1.0/d4/d94/tutorial_camera_calibration.html).


## Distortion models

There are actually 2 different models:

- The direct model that applies distortion: find the 2D image (distorted) coordinates of a 3D point given its 2D projected (undistorted) coordinates.
- The inverse model that rectifies distortion: find the 2D (undistorted) coordinate allowing to project a 2D (distorted) image coordinate into 3D.

.. image:: https://github.com/ispgroupucl/calib3d/blob/main/assets/distortion_steps.png?raw=true

### Direct model: "distort"

The following relationships allows to compute the distorted 2D coordinates \((x_d,y_d)\) on an image from the 2D coordinates provided by the linear projection model \((x_u,y_u)\). Those coordinates are expressed in the camera coordinate system, i.e. they are distances (not pixels) and the point \((0,0)\) is the principal point of the camera.

$$\begin{align}
    x_d &= x_u\overbrace{\left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right)}^{\text{radial component}} + \overbrace{\left[2p_1 x_uy_u + p_2\left({r_u}^2 + 2{x_u}^2\right)\right]}^{\text{tangeantial component}}\\
    y_d &= y_u\left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right) + \left[2p_2 x_uy_u + p_1\left({r_u}^2 + 2{y_u}^2\right)\right]
\end{align}$$

Where:

- \(k_1, k_2, k_3, \cdots\)  are the radial distortion coefficients
- \(t_1, t_2\) are the tangential distortion coefficients
- \({r_u}^2 := {x_u}^2 + {y_u}^2\)

We usually use only 3 radial distortion coefficients, which makes a total of 5 coefficients. Those coefficients are found by running an optimisation algorithm on a set of 2D point - 3D point relations as we did with `cv2.calibrateCamera`. They are stored in the `kc` vector.

### Inverse model: "rectify"

The distortion operation cannot be inverted analitically using the coefficients \(k_1\), \(k_2\), \(k_3\), \(p_1\), \(p_2\) (i.e. have \(x_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)\) and \(y_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)\)). We either need another model with another set of coefficients, or make an approximation.

Here, we will use the following approximation: We will assume that the distortion at point \((x_d,y_d)\) would be the same that distortion at point \((x_u,y_u)\) ! Therefore:

$$\left\{\begin{align}
    2p_1 x_uy_u + p_2\left({r_u}^2 + 2{x_u}^2\right) &\approx 2p_1 x_dy_d + p_2\left({r_d}^2 + 2{x_d}^2\right)\\
    2p_2 x_uy_u + p_1\left({r_u}^2 + 2{y_u}^2\right) &\approx 2p_2 x_dy_d + p_1\left({r_d}^2 + 2{y_d}^2\right) \\
    \left(1 + k_1 {r_u}^2 + k_2 {r_u}^4 + k_3 {r_u}^6+\cdots\right) &\approx \left(1 + k_1 {r_d}^2 + k_2 {r_d}^4 + k_3 {r_d}^6+\cdots\right)
   \end{align}\right.
    $$

If this approximation holds, it's much easier to get an analytical expression of \(x_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)\) and \(y_u=f_{k_{1,2,3},p_{1,2}}(x_d,y_d)\).


# Implementation

This library defines a [Calib](./calib3d.calib#Calib) object to represent a calibrated camera. Its constructor receives in arguments the intrinsic and extrinsic parameters:

- image dimensions `width` and `height`,
- the translation vector `T`,
- the rotation matrix `R`,
- the camera matrix `K`.

The method `project_3D_to_2D` allows to compute the position in the image of a 3D point in the world. The opposite operation `project_2D_to_3D` requires an additional parameter `Z` that tells the \(z\) coordinate of the 3D point.

"""

EPS = 1e-5

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
                P (np.ndarray): a 3x4 projection matrix
                width (int): image width
                height (int): image height
            Returns:
                A Calib object
        """
        K, R, T, Rx, Ry, Rz, angles = cv2.decomposeProjectionMatrix(P) # pylint: disable=unused-variable
        return cls(K=K, R=R, T=Point3D(-R@Point3D(T)), width=width, height=height)

    @classmethod
    def load(cls, filename) -> 'Calib':
        """ Loads a Calib object from a file (using the pickle library)
            Args:
                filename (str): the file that stores the Calib object
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
                filename (str): the file that will store the calib object
        """
        with open(filename, "wb") as f:
            pickle.dump(self.dict, f)

    def _project_3D_to_2D_cv2(self, point3D: Point3D):
        raise BaseException("This function gives errors when rotating the calibration...")
        return Point2D(cv2.projectPoints(point3D, cv2.Rodrigues(self.R)[0], self.T, self.K, self.kc.astype(np.float64))[0][:,0,:].T)

    def project_3D_to_2D(self, point3D: Point3D) -> Point2D:
        """ Using the calib object, project a 3D point in the 2D image space.
            Args:
                point3D (Point3D): the 3D point to be projected
            Returns:
                The point in the 2D image space on which point3D is projected by calib
        """
        #assert isinstance(point3D, Point3D), "Wrong argument type '{}'. Expected {}".format(type(point3D), Point3D)
        point2D_H = self.P @ point3D.H # returns a np.ndarray object
        point2D_H[2] = point2D_H[2] * np.sign(point2D_H[2]) # correct projection of points being projected behind the camera
        point2D = Point2D(point2D_H)
        # avoid distortion of points too far away
        excluded_points = np.logical_or(np.logical_or(point2D.x < -self.width, point2D.x > 2*self.width),
                                        np.logical_or(point2D.y < -self.height, point2D.y > 2*self.height))
        return Point2D(np.where(excluded_points, point2D, self.distort(point2D)))

    def project_2D_to_3D(self, point2D: Point2D, X: float=None, Y: float=None, Z: float=None) -> Point3D:
        """ Using the calib object, project a 2D point in the 3D image space
            given one of it's 3D coordinates (X,Y or Z). One and only one
            coordinate must be given.
            Args:
                point2D (Point2D): the 2D point to be projected
                X (float): the X coordinate of the 3D point
                Y (float): the Y coordinate of the 3D point
                Z (float): the Z coordinate of the 3D point
            Returns:
                The point in the 3D world that projects on `point2D` and for
                which the given coordinates is given.
        """
        v = [X,Y,Z]
        assert sum([1 for x in v if x is None]) == 2
        assert isinstance(point2D, Point2D), "Wrong argument type '{}'. Expected {}".format(type(point2D), Point2D)
        point2D = self.rectify(point2D)
        X = Point3D(self.Pinv @ point2D.H)
        d = (X - self.C)
        P = np.nan_to_num(Point3D(*v), 0)
        v = np.array([[0 if x is None else 1 for x in v]]).T
        return line_plane_intersection(self.C, d, P, v)

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

    def crop(self, x_slice: slice, y_slice: slice) -> 'Calib':
        x0 = x_slice.start
        y0 = y_slice.start
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        T = np.array([[1, 0,-x0], [0, 1,-y0], [0, 0, 1]])
        return self.update(width=width, height=height, K=T@self.K)

    def scale(self, output_width: int=None, output_height: int=None, sx: float=None, sy: float=None) -> 'Calib':
        """ Returns a calib corresponding to a camera that was scaled by horizontal and vertical scale
            factors `sx` and `sy`. If `sx` and `sy` are not given, the scale factors are computed with the current
            width and height and the given `output_width` and `output_height`.
        """
        sx = sx or output_width/self.width
        sy = sy or output_height/self.height
        width = output_width or int(self.width*sx)
        height = output_height or int(self.height*sy)
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return self.update(width=width, height=height, K=S@self.K)

    def flip(self) -> 'Calib':
        """ Returns a calib corresponding to a camera that was flipped horizontally.
        """
        F = np.array([[-1, 0, self.width-1], [0, 1, 0], [0, 0, 1]])
        return self.update(K=F@self.K)

    def rotate(self, angle) -> 'Calib':
        """ Returns a calib corresponding to a camera that was rotated by `angle` degrees.
            angle (float) : Angle of rotation in degrees.
        """
        if angle == 0:
            return self
        A, new_width, new_height = compute_rotate(self.width, self.height, angle, degrees=True)
        return self.update(K=A@self.K, width=new_width, height=new_height)

    def rot90(self, k) -> 'Calib':
        """ Returns a calib corresponding to a camera that was rotated `k` times around it's main axis.
            k (int) : Number of times the array is rotated by 90 degrees.

            .. todo:: This method should be tested.
        """
        raise NotImplementedError("Method not tested")
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])**k
        transpose = k % 2
        return self.update(K=R@self.K, width=self.height if transpose else self.width, height=self.width if transpose else self.height)

    def compute_length2D(self, point3D: Point3D, length3D: float) -> np.ndarray:
        """ Returns the length in pixel of a 3D length at point3D

            .. versionchanged:: 2.8.0: `length3D` and `point3D` were interverted.
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
        cond = np.stack((point2D.x >= 0, point2D.y >= 0, point2D.x <= self.width-1, point2D.y <= self.height-1))
        return np.all(cond, axis=0)

    def dist_to_border(self, point3D: Point3D) -> np.ndarray:
        """ Returns the distance to `Calib` object's borders in both dimensions,
            For each point given in point3D
        """
        point2D = self.project_3D_to_2D(point3D)
        distx = np.min(np.stack((self.width - point2D.x, point2D.x)), axis=0)
        disty = np.min(np.stack((self.height - point2D.y, point2D.y)), axis=0)
        return distx, disty

    def is_behind(self, point3D: Point3D) -> np.ndarray:
        """ Returns `True` where for points that are behind the camera and `False` otherwise.
        """
        n = Point3D(0,0,1) # normal to the camera plane in camera 3D coordinates system
        point3D_c = Point3D(np.hstack((self.R, self.T)) @ point3D.H)  # point3D expressed in camera 3D coordinates system
        return np.asarray((n.T @ point3D_c)[0] < 0)

    def __str__(self):
        return f"Calib object ({self.width}×{self.height})@({self.C.x: 1.6g},{self.C.y: 1.6g},{self.C.z: 1.6g})"



def line_plane_intersection(C: Point3D, d, P: Point3D, n) -> Point3D:
    """ Finds the intersection between a line and a plane.
        Args:
            C (Point3D): a point on the line
            d (np.ndarray): the direction-vector of the line
            P (Point3D): a Point3D on the plane
            n (np.ndarray): the normal vector of the plane
        Returns the Point3D at the intersection between the line and the plane.
    """
    d = d/np.linalg.norm(d, axis=0)
    dot = d.T @ n
    if np.any(np.abs(dot) < EPS):
        return None
    dist = ((P-C).T @ n) / dot  # Distance between plane z=Z and camera
    return Point3D(C + dist.T*d)

def compute_rotate(width, height, angle, degrees=True):
    """ Computes rotation matrix and new width and height for a rotation of angle degrees of a widthxheight image.
    """
    assert degrees, "Angle in gradient is not implemented (yet)"
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
        - rotation of `angle` degrees around output image center, followed by a
        - crop defined by `x_slice` and `y_slice`, followed by a
        - scale to recover `output_shape`.
    """
    # Rotation
    R = np.eye(3)
    center = ((x_slice.start + x_slice.stop)/2, (y_slice.start + y_slice.stop)/2)
    R[0:2,:] = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Crop
    x0 = x_slice.start
    y0 = y_slice.start
    width = x_slice.stop - x_slice.start
    height = y_slice.stop - y_slice.start
    C = np.array([[1, 0,-x0], [0, 1,-y0], [0, 0, 1]])

    # Scale
    sx = output_shape[0]/width
    sy = output_shape[1]/height
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    # Random Flip
    assert not do_flip, "There is a bug with random flip"
    f = np.random.randint(0,2)*2-1 if do_flip else 1 # random sample in {-1,1}
    F = np.array([[f, 0, 0], [0, 1, 0], [0, 0, 1]])

    return S@C@R@F


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


