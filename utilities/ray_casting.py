import numpy as np
from lidar_simulation.utilities.geometry_calculations import support_vector_representation
from numba import cuda, float64
from math import sqrt


@cuda.jit
def ray_intersection_gpu(ray_origin, ray_direction, vertices, polygons, point_cloud):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ray_hit = cuda.local.array(3, float64)
    closest_hit_distance = 1000
    for n in range(len(polygons)):
        pos = vertices[polygons[n, 0]]
        v1 = vertices[polygons[n, 1]]
        v2 = vertices[polygons[n, 2]]
        v1[0] -= pos[0]
        v1[1] -= pos[1]
        v1[2] -= pos[2]
        v2[0] -= pos[0]
        v2[1] -= pos[1]
        v2[2] -= pos[2]

        normal = cuda.local.array(3, float64)
        normal[0] = v1[1] * v2[2] - v1[2] * v2[1]
        normal[1] = v1[2] * v2[0] - v1[0] * v2[2]
        normal[2] = v1[0] * v2[1] - v1[1] * v2[0]

        plane_ray_origin_distance = (pos[0] - ray_origin[0]) * normal[0] + \
                                    (pos[1] - ray_origin[1]) * normal[1] + \
                                    (pos[2] - ray_origin[2]) * normal[2]

        ray_hit_distance = plane_ray_origin_distance / (ray_direction[i, 0] * normal[0] +
                                                        ray_direction[i, 1] * normal[1] +
                                                        ray_direction[i, 2] * normal[2])
        ray_hit[0] = ray_direction[i, 0] * ray_hit_distance + ray_origin[0]
        ray_hit[1] = ray_direction[i, 1] * ray_hit_distance + ray_origin[1]
        ray_hit[2] = ray_direction[i, 2] * ray_hit_distance + ray_origin[2]

        inside_polygon = (v1[1] * normal[2] - v1[2] * normal[1]) * (ray_hit[0] - pos[0]) + \
                         (v1[2] * normal[0] - v1[0] * normal[2]) * (ray_hit[1] - pos[1]) + \
                         (v1[0] * normal[1] - v1[1] * normal[0]) * (ray_hit[2] - pos[2]) <= 0
        inside_polygon *= (-v2[1] * normal[2] - v1[2] * normal[1]) * (ray_hit[0] - pos[0]) + \
                          (-v2[2] * normal[0] - v1[0] * normal[2]) * (ray_hit[1] - pos[1]) + \
                          (-v2[0] * normal[1] - v1[1] * normal[0]) * (ray_hit[2] - pos[2]) <= 0
        inside_polygon *= ((v2[1] - v1[1]) * normal[2] - v1[2] * normal[1]) * (ray_hit[0] - pos[0] - v1[0]) + \
                          ((v2[2] - v1[2]) * normal[0] - v1[0] * normal[2]) * (ray_hit[1] - pos[1] - v1[1]) + \
                          ((v2[0] - v1[0]) * normal[1] - v1[1] * normal[0]) * (ray_hit[2] - pos[2] - v1[2]) <= 0

        if inside_polygon:
            if ray_hit_distance < closest_hit_distance:
                closest_hit_distance = ray_hit_distance
                point_cloud[i, 0] = ray_hit[0]
                point_cloud[i, 1] = ray_hit[1]
                point_cloud[i, 2] = ray_hit[2]

        # from lidar_simulation.data_loaders.load_3d_models import load_Porsche911
        #
        # ray_origin = np.zeros(3)
        # ray_direction = np.random.uniform(0, 1, (10, 3))
        # vertices, polygons = load_Porsche911()
        # point_cloud = np.zeros((len(ray_direction), 3))
        # ray_intersection(ray_origin, ray_direction, vertices, polygons, point_cloud)
        # quit()


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
    from lidar_simulation.data_loaders.load_3d_models import load_Porsche911
    from lidar_simulation.lidar import Lidar
    vertices, polygons = load_Porsche911()
    ray_origin, ray_direction = Lidar(delta_azimuth=2 * np.pi / 1000,
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
    from lidar_simulation.lidar import Lidar
    ray_origin, ray_direction = Lidar(position=(10, 0, 0)).create_rays(vertices)
    point_cloud = ray_intersection(ray_origin, ray_direction, vertices, polygons)
    print(len(point_cloud))

    from lidar_simulation.utilities.visualization import visualize_2d
    visualize_2d(point_cloud)
    visualize_2d(point_cloud, plane=(0, 2))
    visualize_2d(point_cloud, plane=(1, 2))


def sample_usage3():
    from lidar_simulation.data_loaders.create_test_data import create_box

    vertices, polygons = create_box()
    ray_origin = (10, 0, 0)
    ray_direction = np.array(((-1, 0.001, 0.001),
                              (-1, 0, 0))).reshape(-1, 3)

    point_cloud = ray_intersection(ray_origin, ray_direction, vertices, polygons)
    print(len(point_cloud))

    from lidar_simulation.utilities.visualization import visualize_3d
    visualize_3d(point_cloud)
    # visualize_3d(vertices)


if __name__ == "__main__":
    sample_usage()
