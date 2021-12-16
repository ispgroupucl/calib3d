r"""
This library offers several tools for manipulation of calibrated cameras, projective geometry and computations using homogenous coordinates. The different modules are document here:

- [Computations with homogenous coordinates](./calib3d/points)
- [Projective geometry with calibrated cameras](./calib3d/calib)

# Introduction

Camera calibration allows to determine the relation between the camera's pixels (2D coordinates) and points in the real world
(3D coordinates). It implies computation using homogenous coordinates. This python library aims at simplifying implementations
of projective geometry computations, building on top of `numpy` and `cv2`.

# Working with homogenous coordinates

The vector used to represent 2D and 3D points are vertical vectors, which are stored as 2D matrices in `numpy`. Furthemore, in homogenous coordinates: a 3D point (x,y,z) in the world is represented by a 4 element vector (𝜆x,𝜆y,𝜆z,𝜆) where 𝜆 ∈ ℝ₀.

To simplify access to x and y (and z) coordinates of those points as well as computations in homogenous coordinates, we defined the types [`Point2D`](./calib3d.points#Point2D) (and [`Point3D`](./calib3d.points#Point3D)) extending `numpy.ndarray`. Therefore, access to y coordinate of point is `point.y` instead of `point[1][0]` (`point[1][:]` for an array of points), and access to homogenous coordinates is made easy with `point.H`, while it is still possible to use point with any numpy operators.

# Camera calibration

It is based on the pinhole camera model that approximate how lights travels between the scene and the camera sensor. Camera calibration is composed of two sets of parameters:

The extrinsic parameters define the position of the camera center and it's heading relative to the world coordinates.
The Intrinsic parameters define sensor and lens parameters. In the simple projective model, no distortion is taken into account.

They are 3 different coordinates systems:

- The 3D coordinates relative to the origin of the world.
- The 3D coordinates relative to the camera center.
- The 2D pixel positions where 3D positions are being projected.


## Extrinsic parameters
The extrinsic parameters defines the transformations from the 3D coordinates relative the origin of the world $\left[x_O,y_O,z_O\right]^T$ to the 3D coordinates relative to the camera center $\left[x_C,y_C,z_C\right]^T$. The camera 3D coordinates system has the following **conventions**:
- The point (0,0,0) is the center of projection of the camera and is called the _principal point_.
- The $z$ axis of the camera points _towards the scene_, the $x$ axis is along the sensor width pointing towards the right of the image, and the $y$ axis is along the sensor height pointing towards the bottom of the image.

The _camera_ coordinates system is therefore a transformation of the _world_ coordinates systems with:
- A **rotation** defined by a rotation matrix $R$ using euler angles in a right-hand orthogonal system. The rotation is applied to the world coordinates system to obtain the camera orientation.
- A **translation** defined by a translation vector $T$ representing the position of the center of the world in the camera coordinates system !

Hence,

$$\lambda\left[\begin{matrix}x_C\\y_C\\z_C\\1\end{matrix}\right] = \left[\begin{matrix}
R_{3x3} & T_{3x1}\\{\bf 0}_{1x3}&1
\end{matrix}\right] \left[\begin{matrix}x_O\\y_O\\z_O\\1\end{matrix}\right]$$

**Important notes:**
- The rotation matrix represents a passive (or alias) transformation because it's the coordinates system that rotates and not the objects.
- Euler angles define a 3D rotation starting with a rotation around $x$ followed by a rotation around $y$ followed by a rotation around $z$ (the order matters).
- If $T$ is expressed in the camera coordinate system, the position of the camera expressed in world coordinates system is $C:=-R^{-1}T = -R^{T}T$ (since $R$ is a rotation matrix).


## Intrinsic parameters
The intrinsic parameters defines the transformation between the 3D coordinates relative to the camera center $\left[x_C,y_C,z_C\right]^T$ and the 2D coordinates in the camera sensor $\left[i,j\right]^T$. This transformation is called a _projection_ and includes:
- the scale produced by the focal length, with $f$ being the distance between the camera center and the plane on which the image is projected.
- the scale factors $(m_x,m_y)$ relating pixels units to distance units (usually $m_x=m_y$ because pixels are squares).
- the translation from the camera _principal point_ to a top-left origin, with $(u_0,v_0)$ being the position of the _principal point_ expressed in the image coordinates system.
- a skew coefficient $\gamma$ between the $x$ and $y$ axis in the sensor (usually $\gamma=0$ because pixels are squares).

Those transformations can be aggregated in the following matrix called "camera matrix":

$$K := \left[\begin{matrix}f\cdot m_x & \gamma & u_0 \\ 0 & f\cdot m_y & v_0 \\ 0 & 0 & 1\end{matrix}\right]$$

Therefore,
$$\lambda\left[\begin{matrix}i\\ j\\ 1\end{matrix}\right]= \left[\begin{matrix}K_{3x3}&{\bf 0}_{3x1}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\\1\end{matrix}\right]$$

**Notes:**
- The width and height of the image are to be added to those parameters and delimits the sensor width and height in pixels.
- When applying the **direct** projection of a given 3D point, different values of $\lambda$ will always give the **same** 2D point.
- When applying the **inverse** projection on a given 2D point, different values of $\lambda$ will give **different** 3D points.

This is obvious when simplifying the relation between the two points (The column ${\bf 0}_{3x1}$ cancels the homogenous component of the 3D point):

$$\lambda\left[\begin{matrix}i\\j\\1\end{matrix}\right]= \left[\begin{matrix}K_{3x3}\end{matrix}\right]\left[\begin{matrix}x_C\\y_C\\z_C\end{matrix}\right]$$

The 2D vector in homogenous coordinates is not affected by the value of $\lambda$, while the 3D vector is.



## Projection model
Therefore, by combining
- the transformation from the world coordinates system to the camera coordinates system (defined by $R$ and $T$)
- with the projection from the camera coordinates system to the image pixels (defined by $K$),

We have a projection model allowing to compute the coordinates of a 2D point in the image $\left(i,j\right)$ from a 3D point in the real world $\left(x,y,z\right)$ described by the matrix $P$:
$$P := \left[\begin{matrix}K_{3x3}&{\bf 0}_{3x1}\end{matrix}\right] \left[\begin{matrix}R_{3x3}&T_{3x1}\\{\bf 0}_{1x3}&1\end{matrix}\right]=K_{3x3}\left[\begin{matrix}R_{3x3}&T_{3x1}\end{matrix}\right]$$

The opposite operation requires to invert $P$ and is done by pseudo-inverse inversion because $P$ is rectangular.

## Projection model implementation

This library defines a [Calib](./calib3d.calib#Calib) object to represent a calibrated camera. Its constructor receives in arguments the intrinsic and extrinsic parameters:
- image dimensions `width` and `height`,
- the translation vector `T`,
- the rotation matrix `R`,
- the camera matrix `K`.

The method `project_3D_to_2D` allows to compute the position in the image of a 3D point in the world. The opposite operation `project_2D_to_3D` requires an additional parameter `Z` that tells the $z$ coordinate of the 3D point.



"""

from .points import Point3D, Point2D
from .calib import Calib, parameters_to_affine_transform, compute_rotation_matrix
