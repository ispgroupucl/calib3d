import cv2
import numpy as np

from calib3d import Calib, Point2D, Point3D

__doc__ = r"""

# Projective Drawing

When drawing 3D objets on a 2D canvas, several things need to be considered:
- Projection onto the 2D space using the calibration information
- Handling lens distortion that make straight lines appear curved
- Handling of objects visiblity given the canvas dimensions
"""

class ProjectiveDrawer():
    """ Given the calibration information with `Calib`, and a number of segments
        to decompose straight lines, this objet offer several functions to draw
        on a 2D canvas given 3D coordinates.
    """
    def __init__(self, calib: Calib, color, thickness: int=1, segments: int=10):
        self.color = color
        self.thickness = thickness
        self.calib = calib
        self.segments = segments

    def _polylines(self, canvas, points: Point2D, color=None, thickness: int=None, markersize=None, **kwargs):
        thickness = thickness or self.thickness
        color = color or self.color
        if isinstance(canvas, np.ndarray):
            points = np.array(points.astype(np.int32).T.reshape((-1,1,2)))
            if thickness < 0:
                cv2.fillPoly(canvas, [points], color=color, **kwargs)
            else:
                cv2.polylines(canvas, [points], False, color=color, thickness=thickness, **kwargs)
            if markersize:
                for point in points:
                    cv2.drawMarker(canvas, tuple(point[0]), color=color, markerSize=markersize, thickness=thickness)
        else:
            if thickness < 0:
                try:
                    import matplotlib as mpl
                    p = mpl.patches.Polygon(np.array(list(zip(points.x, points.y))), facecolor=np.array(color)/255, alpha=.35)
                    canvas.add_patch(p)
                except ImportError:
                    raise ImportError("The current implementation requires matplotlib")
            else:
                canvas.plot(points.x, points.y, linewidth=thickness, color=np.array(color)/255, markersize=markersize, **kwargs)

    def polylines(self, canvas, points3D: Point3D, color=None, thickness: int=None, **kwargs):
        self._polylines(canvas, self.calib.project_3D_to_2D(points3D), color=color, thickness=thickness, **kwargs)
        #for point3D1, point3D2 in zip(points3D, points3D.close()[:,1:]):
        #    self.draw_line(canvas, point3D1, point3D2, *args, **kwargs)

    def draw_line(self, canvas, point3D1: Point3D, point3D2: Point3D, color=None, thickness: int=None, only_visible=True, **kwargs):
        if only_visible:
            try:
                point3D1, point3D2 = visible_segment(self.calib, point3D1, point3D2)
            except ValueError:
                return
        points3D = Point3D(np.linspace(point3D1, point3D2, self.segments+1))
        self._polylines(canvas, self.calib.project_3D_to_2D(points3D), color=color, thickness=thickness, **kwargs)

    def draw_arc(self, canvas, center, radius, start_angle=0.0, stop_angle=2*np.pi, color=None, thickness=None):
        thickness = thickness or self.thickness
        angles = np.linspace(start_angle, stop_angle, self.segments*4+1)
        xs = np.cos(angles)*radius + center.x
        ys = np.sin(angles)*radius + center.y
        zs = np.ones_like(angles)*center.z
        points3D = Point3D(np.vstack((xs,ys,zs)))
        self._polylines(canvas, self.calib.project_3D_to_2D(points3D), color=color, thickness=thickness)

    def draw_rectangle(self, canvas, point3D1, point3D2):
        c1 = point3D1
        c3 = point3D2
        if point3D1.z == point3D2.z:
            c2 = Point3D(c1.x, c3.y, c1.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        elif point3D1.x == point3D2.x:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c1.x, c3.y, c1.z)
        elif point3D1.y == point3D2.y:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        corners = [c1, c2, c3, c4, c1]
        for p1, p2 in zip(corners, corners[1:]):
            self.draw_line(canvas, p1, p2)

    def fill_polygon(self, canvas, points3D):
        points3D = points3D.close().linspace(self.segments)
        self._polylines(canvas, self.calib.project_3D_to_2D(points3D), thickness=-1)

def visible_segment(calib: Calib, point3D1: Point3D, point3D2: Point3D):
    """ From a segment defined by the given two 3D points, compute the two 3D
        points delimiting the visible segment in the given calib.
    """
    def dichotomy(inside, outside, max_it=10):
        middle = Point3D((inside+outside)/2)
        if max_it == 0:
            return middle
        max_it = max_it - 1
        return dichotomy(middle, outside, max_it) if calib.projects_in(middle) else dichotomy(inside, middle, max_it)
    def find_point_inside(p1, p2, max_it=4):
        assert not calib.projects_in(p1) and not calib.projects_in(p2)
        middle = Point3D((p1+p2)/2)
        if calib.projects_in(middle):
            return middle
        if max_it == 0:
            return None
        point_inside = find_point_inside(middle, p2, max_it-1)
        if point_inside is not None:
            return point_inside
        return find_point_inside(middle, p1, max_it-1)

    p1, p2 = point3D1, point3D2
    if calib.projects_in(p1) and calib.projects_in(p2):
        return p1, p2
    elif calib.projects_in(p1):
        return p1, dichotomy(p1, p2)
    elif calib.projects_in(p2):
        return dichotomy(p2, p1), p2
    else:
        point_inside = find_point_inside(p1, p2)
        if point_inside is None:
            raise ValueError
        return dichotomy(point_inside, p1), dichotomy(point_inside, p2)
