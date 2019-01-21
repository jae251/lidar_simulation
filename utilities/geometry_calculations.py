import numpy as np


def support_vector_representation(vertices, polygons):
    pos = vertices[polygons[:, 0]]
    v1 = vertices[polygons[:, 1]] - pos
    v2 = vertices[polygons[:, 2]] - pos
    return pos, v1, v2


def point_inside_triangle(c1, c2):
    return (0 <= c1) * (c1 <= 1) * (0 <= c2) * (c2 <= 1) * (c1 + c2 <= 1)


def check_inside_polygon(point, edge_base, edge, normal):
    edge_normal = np.cross(edge, normal)
    inside = np.sum(edge_normal * (point - edge_base), axis=2) <= 0
    return inside


def rotate_point_cloud(point_cloud, angle):
    rotated_pcloud = np.zeros(point_cloud.shape)
    s = np.sin(angle)
    c = np.cos(angle)
    rotated_pcloud[:, 0] = point_cloud[:, 0] * c - point_cloud[:, 1] * s
    rotated_pcloud[:, 1] = point_cloud[:, 0] * s + point_cloud[:, 1] * c
    try:
        rotated_pcloud[:, 2] = point_cloud[:, 2]
    except IndexError:
        pass
    return rotated_pcloud


if __name__ == "__main__":
    from data_loaders.load_3d_models import load_Porsche911

    vertices, _, polygons = load_Porsche911()
    polygons = polygons[:, :, 0]
    support_vectors = support_vector_representation(vertices, polygons)
