import cv2
import tensorflow as tf
from calib3d import Calib
# pylint: disable=unexpected-keyword-arg, no-value-for-parameter, too-many-function-args

class TensorflowCalib():
    def __init__(self, *, width, height, T=None, K, kc, r=None, R=None, Kinv=None, Pinv=None, P=None, dtype=tf.float32):
        self.width = tf.cast(width, dtype=dtype)
        self.height = tf.cast(height, dtype=dtype)
        self.K = tf.cast(K, dtype=dtype)
        self.T = tf.cast(T, dtype=dtype)
        self.r = tf.cast(r, dtype=dtype)
        self.R = tf.cast(R, dtype=dtype) if R is not None else rodrigues_batch(r, dtype=dtype)

        self.P = tf.cast(P, dtype=dtype) if P is not None else tf.matmul(self.K, tf.concat((self.R, self.T), axis=-1))
        self.Pinv = tf.cast(Pinv, dtype=dtype) if Pinv is not None else pinv(self.P, dtype=dtype)
        self.Kinv = tf.cast(Kinv, dtype=dtype) if Kinv is not None else pinv(self.K, dtype=dtype)

        self.kc = tf.cast(kc, dtype=dtype) if kc is not None else None
        self.C = -tf.matmul(self.R, self.T, transpose_a=True) # pylint: disable=invalid-unary-operand-type

        self.batch_size = K.shape[0]
        self.dtype = dtype
    @classmethod
    def from_numpy(cls, calib: Calib, dtype=tf.float64):
        return cls(
            K=tf.constant(calib.K, dtype=dtype)[tf.newaxis],
            r=tf.constant(cv2.Rodrigues(calib.R)[0], dtype=dtype)[tf.newaxis],
            T=tf.constant(calib.T, dtype=dtype)[tf.newaxis],
            width=tf.constant(calib.width, dtype=dtype)[tf.newaxis],
            height=tf.constant(calib.height, dtype=dtype)[tf.newaxis],
            kc=tf.constant(calib.kc, dtype=dtype)[tf.newaxis],
            dtype=dtype
        )
    def project_3D_to_2D(self, point3D):
        point2D = from_homogenous(tf.matmul(self.P, to_homogenous(point3D, dtype=self.dtype)))
        # TODO: avoid distort points too much outside the image as the distortion model is not perfect
        return self.distort(point2D)
    def project_2D_to_3D(self, point2D, Z):
        point2D = self.rectify(point2D)
        X = from_homogenous(tf.matmul(self.Pinv, to_homogenous(point2D, dtype=self.dtype)))
        d = (X - self.C)
        P = batch_expand(tf.constant([[0],[0],[Z]], dtype=self.dtype), d)
        n = batch_expand(tf.constant([[0],[0],[1]], dtype=self.dtype), d)
        return find_intersection(self.C, d, P, n)
    def distort(self, point2D):
        if self.kc is None:
            return point2D
        point2D = from_homogenous(tf.matmul(self.Kinv, to_homogenous(point2D, dtype=self.dtype)))
        rad1, rad2, tan1, rad3, tan2 = self.kc[:,0], self.kc[:,1], self.kc[:,2], self.kc[:,3], self.kc[:,4]
        r2 = point2D[:,0]*point2D[:,0] + point2D[:,1]*point2D[:,1]
        delta = 1 + rad1[:,tf.newaxis]*r2 + rad2[:,tf.newaxis]*r2*r2 + rad3[:,tf.newaxis]*r2*r2*r2
        dx = tf.stack((
            2*tan1[:,tf.newaxis]*point2D[:,0]*point2D[:,1] + tan2[:,tf.newaxis]*(r2 + 2*point2D[:,0]*point2D[:,0]),
            2*tan2[:,tf.newaxis]*point2D[:,0]*point2D[:,1] + tan1[:,tf.newaxis]*(r2 + 2*point2D[:,1]*point2D[:,1])
        ), axis=1)
        point2D = point2D*delta + dx
        return from_homogenous(tf.matmul(self.K, to_homogenous(point2D, dtype=self.dtype)))
    def rectify(self, point2D):
        if self.kc is None:
            return point2D
        point2D = from_homogenous(tf.matmul(self.Kinv, to_homogenous(point2D, dtype=self.dtype)))
        rad1, rad2, tan1, rad3, tan2 = self.kc[:,0], self.kc[:,1], self.kc[:,2], self.kc[:,3], self.kc[:,4]
        r2 = point2D[:,0]*point2D[:,0] + point2D[:,1]*point2D[:,1]
        delta = 1 + rad1[:,tf.newaxis]*r2 + rad2[:,tf.newaxis]*r2*r2 + rad3[:,tf.newaxis]*r2*r2*r2
        dx = tf.stack((
            2*tan1[:,tf.newaxis]*point2D[:,0]*point2D[:,1] + tan2[:,tf.newaxis]*(r2 + 2*point2D[:,0]*point2D[:,0]),
            2*tan2[:,tf.newaxis]*point2D[:,0]*point2D[:,1] + tan1[:,tf.newaxis]*(r2 + 2*point2D[:,1]*point2D[:,1])
        ), axis=1)
        point2D = (point2D - dx)/delta[:,tf.newaxis]
        return from_homogenous(tf.matmul(self.K, to_homogenous(point2D, dtype=self.dtype)))

def batch_expand(input_tensor, batch_tensor):
    # https://stackoverflow.com/questions/57716363/explicit-broadcasting-of-variable-batch-size-tensor
    length = len(batch_tensor.shape)-1
    input_tensor = input_tensor[tf.newaxis]
    broadcast_shape = tf.where([True, *[False]*length], tf.shape(batch_tensor), tf.shape(input_tensor))
    return tf.broadcast_to(input_tensor, broadcast_shape)

def to_homogenous(points, dtype=tf.float64):
    _,_,N = points.shape
    ones = tf.ones((1,N), dtype=dtype)
    ones = batch_expand(ones, points)
    return tf.concat((points, ones), axis=-2)

def from_homogenous(points):
    return points[:,:-1,:]/points[:,-1:,:]

def pinv(a, rcond=1e-15, dtype=tf.float64):
    s, u, v = tf.svd(a)
    # Ignore singular values close to zero to prevent numerical overflow
    limit = rcond * tf.reduce_max(s)
    non_zero = tf.greater(s, limit)

    reciprocal = tf.where(non_zero, tf.cast(tf.reciprocal(s), dtype=dtype), tf.zeros_like(s))
    lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
    return tf.matmul(lhs, u, transpose_b=True)

def find_intersection(C, d, P, n):
    d = d/tf.norm(d, axis=-2, keepdims=True)
    dist = tf.tensordot(P-C, n, axes=[[1],[1]])[:,:,0] / tf.tensordot(d, n, axes=[[1],[1]])[:,tf.newaxis,:,0,0]  # Distance between plane z=Z and camera
    return C + dist*d

# https://github.com/blzq/tf_rodrigues/blob/master/rodrigues.py
def rodrigues_batch(rvecs, dtype=tf.float64):
    """ Convert a batch of axis-angle rotations in rotation vector form shaped
        (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
        See
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    rvecs = tf.cast(rvecs, dtype=dtype)
    batch_size = tf.shape(rvecs)[0]
    assert rvecs.shape[1] == 3

    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas
    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = tf.zeros([batch_size], dtype=dtype)  # for broadcasting
    Ks_1 = tf.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = tf.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = tf.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    Rs = tf.eye(3, batch_shape=[batch_size], dtype=dtype) + \
         tf.sin(thetas)[..., tf.newaxis] * Ks + \
         (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)

    # Avoid returning NaNs where division by zero happened
    return tf.where(is_zero, tf.eye(3, batch_shape=[batch_size], dtype=dtype), Rs)
