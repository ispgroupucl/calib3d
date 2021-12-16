Module calib3d.tf1.tf1_calib
============================

Functions
---------

    
`batch_expand(input_tensor, batch_tensor)`
:   

    
`find_intersection(C, d, P, n)`
:   

    
`from_homogenous(points)`
:   

    
`pinv(a, rcond=1e-15, dtype=tf.float64)`
:   

    
`rodrigues_batch(rvecs, dtype=tf.float64)`
:   Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    
`to_homogenous(points, dtype=tf.float64)`
:   

Classes
-------

`TensorflowCalib(*, width, height, T=None, K, kc, r=None, R=None, Kinv=None, Pinv=None, P=None, dtype=tf.float32)`
:   

    ### Static methods

    `from_numpy(calib: calib3d.calib.Calib, dtype=tf.float64)`
    :

    ### Methods

    `distort(self, point2D)`
    :

    `project_2D_to_3D(self, point2D, Z)`
    :

    `project_3D_to_2D(self, point3D)`
    :

    `rectify(self, point2D)`
    :