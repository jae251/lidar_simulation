import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate

try:
    from .geometry_calculations import rotate_point_cloud
except ImportError:
    from utilities.geometry_calculations import rotate_point_cloud


class BoundingBox2D:
    def __init__(self, point_array, label=None, id=None, angle=0, position=(0, 0, 0), height=None):
        self.p = Polygon(point_array)
        self.points = point_array
        self.angle = angle
        self.position = np.array(position)
        self.label = label
        self.id = id
        self.height = height

    @classmethod
    def from_point_cloud(cls, point_cloud, angle=0, label=None, id=None):
        try:
            height = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])
        except IndexError:
            height = None
        pcloud_2d = point_cloud[:, :2]
        rotated_point_cloud = rotate_point_cloud(pcloud_2d, -angle)
        upper_boundary = np.max(rotated_point_cloud, axis=0)
        lower_boundary = np.min(rotated_point_cloud, axis=0)
        point_array = np.array(((upper_boundary[0], upper_boundary[1]),
                                (upper_boundary[0], lower_boundary[1]),
                                (lower_boundary[0], lower_boundary[1]),
                                (lower_boundary[0], upper_boundary[1])))
        point_array = rotate_point_cloud(point_array, angle)
        position = np.zeros(3)
        return cls(point_array, label=label, id=id, angle=angle, position=position, height=height)

    def get_size(self):
        dim = np.linalg.norm(self.points - np.roll(self.points, 1, axis=0), axis=1)[:2]
        dim[::-1].sort()  # put higher value first
        return dim

    def get_area(self):
        size = self.get_size()
        return size[0] * size[1]

    def intersection_over_union(self, other_bounding_box):
        intersection = self.p.intersection(other_bounding_box.p).area
        union = self.get_area() + other_bounding_box.get_area() - intersection
        iou = intersection / union
        return iou

    def overlaps(self, other_bounding_box):
        return self.p.intersects(other_bounding_box.p)

    def affine_transform(self, translation, angle):
        self.p = translate(rotate(self.p, angle * 180 / np.pi), translation[0], translation[1])
        self.points = np.dstack(self.p.exterior.coords.xy)[0, :4]
        self.angle += angle
        self.position += translation

    def copy(self):
        return BoundingBox2D(self.points, label=self.label, id=self.id, angle=self.angle, position=self.position)

    def __str__(self):
        return "2D Bounding Box:\nid: {}\nlabel: {}\npoints:\n{}".format(self.id, self.label, self.points)

    def to_numpy(self):
        return np.array((  # self.id,
            self.label,
            self.points,
            self.position,
            self.angle), dtype=np.dtype([  # ("id", np.uint8),
            ("label", "S20"),
            ("points", (np.float64, (4, 2))),
            ("position", (np.float64, 3)),
            ("angle", np.float64)]))


########################################################################################################################

if __name__ == "__main__":
    points = np.array(((1, 1), (1, 0), (0, 0), (0, 1)))
    bb = BoundingBox2D(points)
    print(bb.get_size())
    print(bb.get_area())
    print(bb.intersection_over_union(bb))
    print(bb.overlaps(bb))
    print(bb.affine_transform((1, 1, 0), 90))
    print(bb.copy())
