import numpy as np
from calib3d import Point3D, Point2D, Calib

K = np.array([[2.78137e+03, 0.00000e+00, 9.86217e+02],
              [0.00000e+00, 2.78137e+03, 7.49652e+02],
              [0.00000e+00, 0.00000e+00, 1.00000e+00]])
R = np.array([[ 0.947337,   0.319065,  -0.0273947],
              [-0.116923,   0.424256,   0.897962 ],
              [ 0.298131,  -0.847469,   0.439219 ]])
T = np.array([[-2360.16 ],
              [ -123.465],
              [ 2793.1  ]])
kc = np.array([-0.00865297, -0.0148287, -0.00078693, 0.00025515, 0.])
width = 1936
height = 1458

EPS = 1.0e-6

def test_point3D():
    assert np.all(Point3D(1,2,3) == np.array([[1],[2],[3]]))
    assert np.all(Point3D(np.array([[1],[2],[3]])) == np.array([[1],[2],[3]]))
    assert np.all(Point3D(np.array([[1,2],[2,0],[3,5]])) == np.array([[1,2],[2,0],[3,5]]))
    assert np.all(Point3D([1,2],[2,0],[3,5]) == np.array([[1,2],[2,0],[3,5]]))
    assert np.all(Point3D([1,2,4,9,1],[2,0,0,0,3],[3,5,1,2,3]) == np.array([[1,2,4,9,1],[2,0,0,0,3],[3,5,1,2,3]]))
    assert np.all(Point3D([Point3D(1,2,3), Point3D(5,6,6)]) == Point3D([1,5], [2,6], [3,6]))

def test_getitem():
    return
    point3D = Point3D([1400, 2800],[750, 1500],[0,0])
    assert np.all(point3D.x == [1400, 2800])
    assert np.all(point3D[0] - Point3D(1400, 750, 0) == 0)

def test_iterator():
    points3D = Point3D([1400, 2800],[750, 1500],[0,0])
    points3D_list = [Point3D(1400, 750, 0), Point3D(2800, 1500, 0)]
    for point3D, expected_point3D in zip(points3D, points3D_list):
        assert np.all(point3D - expected_point3D == 0)

def test_flatten():
    p = Point3D(1400, 750, 0)
    assert np.all(p.flatten() == [1400, 750, 0])
    points3D = Point3D([1400, 2800],[750, 1500],[0,0])
    assert np.all(points3D.flatten() == [1400, 2800, 750, 1500, 0, 0])
    assert not isinstance(p.flatten(), Point3D)

def test_tolist():
    p = Point3D(1400.2, 750.3, 0.1)
    assert np.all(p.to_list() == [1400.2, 750.3, 0.1])
    assert np.all(p.to_int_tuple() == (1400, 750, 0))
    points3D = Point3D([1400, 2800],[750, 1500],[0,0])
    try:
        points3D.to_int_tuple()
    except BaseException as e:
        assert isinstance(e, AssertionError)
    else:
        assert False

def test_linspace():
    p1 = Point3D(1,2,3)
    p2 = Point3D(5,6,7)
    points = np.linspace(p1, p2, 5)
    assert np.all(points == np.array([Point3D(1,2,3), Point3D(2,3,4), Point3D(3,4,5), Point3D(4,5,6), Point3D(5,6,7)]))

def test_constructorlist():
    p1 = Point3D(1,2,3)
    p2 = Point3D(5,6,7)
    points = Point3D(np.linspace(p1, p2, 5))
    assert np.all(points == Point3D([1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]))

def test_calib():
    calib = Calib(K=K, R=R, kc=kc, T=T, width=width, height=height)
    assert calib

    assert np.all(calib.P - np.array([[ 2.928916572117e+03, 5.164948427700e+01, 3.569704477840e+02, -3.809875516500e+06],
                                      [-1.017116240980e+02, 5.447060799320e+02, 2.826825969728e+03,  1.750451154150e+06],
                                      [ 2.981310000000e-01,-8.474690000000e-01, 4.392190000000e-01,  2.793100000000e+03]]) < EPS)
    assert np.all(calib.Pinv - np.array([[ 2.525740943199e-04,-4.664282555793e-05, 3.737502209537e-01],
                                         [-8.637913734387e-05, 1.420152674116e-04,-2.068256726337e-01],
                                         [ 6.498330127792e-05, 3.267635024389e-04,-1.161452054241e-01],
                                         [-6.338681359899e-08,-3.315860475191e-09, 2.736415838663e-04]]) < EPS)
    assert np.all(calib.Kinv - np.array([[ 3.595350492743e-04,-2.313036973253e-19,-3.545795776901e-01],
                                         [-1.631270768408e-20, 3.595350492743e-04,-2.695261687586e-01],
                                         [-9.543929521751e-21, 4.085894191784e-19, 1.000000000000e+00]]) < EPS)
    assert np.all(calib.C - np.array([[ 1388.72129962],
                                      [ 3172.49088134],
                                      [-1180.57158572]]) < EPS)

def test_projection():
    calib = Calib(K=K, R=R, kc=kc, T=T, width=width, height=height)

    # Single point
    point3D = Point3D(1400,750,0)
    point2D = calib.project_3D_to_2D(point3D)
    assert np.all(point2D - np.array([[128.950], [782.928]]) < 1.0e-2) # 0.01 pixel error projection
    assert np.all(calib.project_2D_to_3D(point2D, Z=0) - point3D < 1.0e-2) # 0.01 cm error reprojection on image border

    # Multiple points
    points3D = Point3D([1400, 2800],[750, 1500],[0, 0])
    points2D = calib.project_3D_to_2D(points3D)
    assert np.all(points2D - np.array([[128.950, 1895.195], [782.928, 968.128]]) < 1.0e-2) # 0.01 pixel error projection
    assert np.all(calib.project_2D_to_3D(points2D, Z=0) - points3D < 1.0e-2) # 0.01 cm error reprojection on image border

def test_compute_length():
    calib = Calib(K=K, R=R, kc=kc, T=T, width=width, height=height)
    point3D = Point3D(1400,750,0)
    margin3D = 100 #cm
    margin2D = calib.compute_length2D(margin3D, point3D)
    assert len(margin2D.shape) == 1
    assert margin2D - 107.677886 < EPS

