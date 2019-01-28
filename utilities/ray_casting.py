import numpy as np
from lidar_simulation.utilities.geometry_calculations import support_vector_representation


def ray_intersection(ray_origin, ray_direction, vertices, polygons):
    np.warnings.filterwarnings('ignore')
    pos, v1, v2 = support_vector_representation(vertices, polygons)
    normals = np.cross(v1, v2)
    plane_ray_origin_distance = np.sum((pos - ray_origin) * normals, axis=1)
    ray_hit_distance = plane_ray_origin_distance / np.sum(ray_direction[:, np.newaxis] * normals, axis=2)

    ray_hit = ray_direction[:, np.newaxis] * ray_hit_distance[:, :, np.newaxis] + ray_origin

    inside_polygon = check_inside_polygon(ray_hit, pos, v1, normals)
    inside_polygon *= check_inside_polygon(ray_hit, pos, -v2, normals)
    inside_polygon *= check_inside_polygon(ray_hit, pos + v1, v2 - v1, normals)

    closest_hit = np.argmin(np.abs(np.ma.masked_array(ray_hit_distance, np.logical_not(inside_polygon))), axis=1)
    valid_ray = np.any(inside_polygon, axis=1)
    point_cloud = ray_hit[np.where(valid_ray), closest_hit[valid_ray]][0]
    return point_cloud, valid_ray


def check_inside_polygon(point, edge_base, edge, normal):
    edge_normal = np.cross(edge, normal)
    inside = np.sum(edge_normal * (point - edge_base), axis=2) <= 0
    return inside


########################################################################################################################
def sample_usage():
    from data_loaders.load_3d_models import load_Porsche911
    from lidar import Lidar
    vertices, polygons = load_Porsche911()
    ray_origin, ray_direction = Lidar(delta_azimuth=2 * np.pi / 2000,
                                      delta_elevation=np.pi / 500,
                                      position=(10, 0, 0)).create_rays(vertices)
    print("Created {} rays.".format(len(ray_direction)))
    point_cloud = ray_intersection(ray_origin, ray_direction, vertices, polygons)
    print(len(point_cloud))

    from utilities.visualization import visualize_2d, visualize_3d
    visualize_3d(point_cloud)
    visualize_3d(vertices)

    visualize_2d(point_cloud)
    visualize_2d(vertices)
    visualize_2d(point_cloud, plane=(0, 2))
    visualize_2d(vertices, plane=(0, 2))
    visualize_2d(point_cloud, plane=(1, 2))
    visualize_2d(vertices, plane=(1, 2))


def sample_usage2():
    vertices = np.array([[0, -1, -1], [0, 1, -1], [0, 0, 1]])
    polygons = np.array((0, 1, 2)).reshape(1, 3)
    from lidar import Lidar
    ray_origin, ray_direction = Lidar(position=(10, 0, 0)).create_rays(vertices)
    point_cloud = ray_intersection(ray_origin, ray_direction, vertices, polygons)
    print(len(point_cloud))

    from utilities.visualization import visualize_2d
    visualize_2d(point_cloud)
    visualize_2d(point_cloud, plane=(0, 2))
    visualize_2d(point_cloud, plane=(1, 2))


def sample_usage3():
    from data_loaders.create_test_data import create_box

    vertices, polygons = create_box()
    ray_origin = (10, 0, 0)
    ray_direction = np.array(((-1, 0.001, 0.001),
                              (-1, 0, 0))).reshape(-1, 3)

    point_cloud = ray_intersection(ray_origin, ray_direction, vertices, polygons)
    print(len(point_cloud))

    from utilities.visualization import visualize_3d
    visualize_3d(point_cloud)
    # visualize_3d(vertices)


if __name__ == "__main__":
    sample_usage()
