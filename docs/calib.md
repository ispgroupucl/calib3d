Module calib3d.calib
====================
# Calibrated camera

Camera calibration is based on the pinhole camera model that approximate how lights travels between the scene and the camera sensor. There are two sets of parameters:

- The **extrinsic parameters** define the position of the camera and it's heading relative to the world coordinates system.
- The **Intrinsic parameters** define the camera sensor and lens parameters.

They are 3 different coordinates systems:

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

## Projection model implementation

This library defines a [Calib](./calib3d.calib#Calib) object to represent a calibrated camera. Its constructor receives in arguments the intrinsic and extrinsic parameters:

- image dimensions `width` and `height`,
- the translation vector `T`,
- the rotation matrix `R`,
- the camera matrix `K`.

The method `project_3D_to_2D` allows to compute the position in the image of a 3D point in the world. The opposite operation `project_2D_to_3D` requires an additional parameter `Z` that tells the \(z\) coordinate of the 3D point.

Functions
---------

    
`compute_rotate(width, height, angle)`
:   Computes rotation matrix and new width and height for a rotation of angle degrees of a widthxheight image.

    
`compute_rotation_matrix(point3D: calib3d.points.Point3D, camera3D: calib3d.points.Point3D)`
:   Computes the rotation matrix of a camera in `camera3D` pointing
    towards the point `point3D`. Both are expressed in word coordinates.
    The convention is that Z is pointing down.
    Credits: François Ledent

    
`find_intersection(C: calib3d.points.Point3D, d, P: calib3d.points.Point3D, n) ‑> calib3d.points.Point3D`
:   Finds the intersection between a line and a plane.
    Arguments:
        C - a Point3D of a point on the line
        d - the direction-vector of the line
        P - a Point3D on the plane
        n - the normal vector of the plane
    Returns the Point3D at the intersection between the line and the plane.

    
`parameters_to_affine_transform(angle: float, x_slice: slice, y_slice: slice, output_shape: tuple, do_flip: bool = False)`
:   Compute the affine transformation matrix that correspond to a
    - horizontal flip if `do_flip` is `True`, followed by a
    - rotation of `angle` degrees around image center, followed by a
    - crop defined by `x_slice` and `y_slice`, followed by a
    - scale to recover `output_shape`.

    
`rotate_image(image, angle)`
:   Rotates an image around its center by the given angle (in degrees).
    The returned image will be large enough to hold the entire new image, with a black background

Classes
-------

`Calib(*, width: int, height: int, T: numpy.ndarray, R: numpy.ndarray, K: numpy.ndarray, kc=None, **_)`
:   Represents a calibrated camera.
    
    Args:
        width (int): image width
        height (int): image height
        T (np.ndarray): translation vector
        R (np.ndarray): rotation matrix
        K (np.ndarray): camera matrix holding intrinsic parameters
        kc (np.ndarray, optional): lens distortion coefficients. Defaults to None.

    ### Static methods

    `from_P(P, width, height) ‑> calib3d.calib.Calib`
    :   Create a `Calib` object from a given projection matrix `P` and image dimensions `width` and `height`.
        Args:
            P (np.ndarray) : a 3x4 projection matrix
            width (int) : image width
            height (int) : image height
        Returns:
            A Calib object

    `load(filename) ‑> calib3d.calib.Calib`
    :   Loads a Calib object from a file (using the pickle library)
        Args:
            filename (str) : the file that stores the Calib object
        Returns:
            The `Calib` object previously saved in `filename`.

    ### Instance variables

    `dict: dict`
    :   Gets a dictionnary representing the calib object (allowing easier serialization)

    ### Methods

    `compute_length2D(self, length3D: float, point3D: calib3d.points.Point3D) ‑> numpy.ndarray`
    :   Returns the length in pixel of a 3D length at point3D

    `crop(self, x_slice, y_slice) ‑> calib3d.calib.Calib`
    :

    `distort(self, point2D: calib3d.points.Point2D) ‑> calib3d.points.Point2D`
    :   Applies lens distortions to the given `point2D`.

    `dump(self, filename) ‑> None`
    :   Saves the current calib object to a file (using the pickle library)
        Args:
            filename (str) : the file that will store the calib object

    `flip(self) ‑> calib3d.calib.Calib`
    :

    `project_2D_to_3D(self, point2D: calib3d.points.Point2D, Z: float) ‑> calib3d.points.Point3D`
    :   Using the calib object, project a 2D point in the 3D image space.
        Args:
            point2D (Point2D) : the 2D point to be projected
            Z (float) : the Z coordinate of the 3D point
        Returns:
            The point in the 3D world for which the z=`Z` and that projects on `point2D`.

    `project_3D_to_2D(self, point3D: calib3d.points.Point3D) ‑> calib3d.points.Point2D`
    :   Using the calib object, project a 3D point in the 2D image space.
        Args:
            point3D (Point3D) : the 3D point to be projected
        Returns:
            The point in the 2D image space on which point3D is projected by calib

    `projects_in(self, point3D: calib3d.points.Point3D) ‑> numpy.ndarray`
    :   Check wether point3D projects into the `Calib` object.
        Returns `True` where for points that projects in the image and `False` otherwise.

    `rectify(self, point2D: calib3d.points.Point2D) ‑> calib3d.points.Point2D`
    :   Removes lens distortion to the given `Point2D`.

    `rotate(self, angle) ‑> calib3d.calib.Calib`
    :

    `scale(self, output_width, output_height) ‑> calib3d.calib.Calib`
    :

    `update(self, **kwargs) ‑> calib3d.calib.Calib`
    :   Creates another Calib object with the given keyword arguments updated
        Args:
            **kwargs : Any of the arguments of `Calib`. Other arguments remain unchanged.
        Returns:
            A new Calib object