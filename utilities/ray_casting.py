import numpy as np
from numba import cuda, float64
from math import inf, sqrt

try:
    from .geometry_calculations import support_vector_representation
except ImportError:
    from utilities.geometry_calculations import support_vector_representation


@cuda.jit
def ray_intersection_uv_gpu(ray_origin, ray_direction, vertices, polygons, uv_coordinates, uv_coordinate_indices,
                            point_cloud, ray_hit_uv):
    '''
    Ray tracing GPU kernel
    :param ray_origin: coordinate of ray origin, e.g. (0,0,0)
    :param ray_direction: an array of 3d unit vectors indicating the ray direction, shape=(r,3)
    :param vertices: array of vertices of 3d object, shape=(v,3)
    :param polygons: array of vertex indices which form a triangular polygon, shape=(p,3)
    :param uv_coordinates: array of uv coordinates of 3d object vertices, shape=(t,2)
    :param uv_coordinate_indices: array of indices pointing to uv coordinates for each polygon, shape=(p,3)
    :param point_cloud: result array in which the ray hits will be stored, shape=(r,3). Entries (0,0,0) indicate a non-hit
    :param ray_hit_uv: result array of uv coordinates of ray hits, shape=(r,2)
    :return: GPU kernel cannot return a result, point_cloud and barycentric_coordinates contain the computed results
    '''
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i < len(ray_direction):
        # allocate memory for intermediate results
        ray_hit = cuda.local.array(3, float64)
        closest_ray_hit = cuda.local.array(3, float64)
        closest_ray_hit[0] = 0
        closest_ray_hit[1] = 0
        closest_ray_hit[2] = 0
        barycentric_coordinates = cuda.local.array(4, float64)
        barycentric_coordinates[0] = 0
        barycentric_coordinates[1] = 0
        barycentric_coordinates[2] = 0
        barycentric_coordinates[3] = 0
        closest_hit_distance = inf
        hit = False

        # check intersection of one ray with every polygon
        for n in range(len(polygons)):
            pos = vertices[polygons[n, 0]]

            # calculate vectors p1->p2 and p1->p3
            v1 = subtract(vertices[polygons[n, 1]], pos)
            v2 = subtract(vertices[polygons[n, 2]], pos)

            normal = cross(v1, v2)
            ray_plane_distance = dot(ray_direction, normal)

            if ray_plane_distance == 0:  # protection against zero division, otherwise kernels fail silently
                continue

            ray_hit_distance = dot(subtract(pos, ray_origin), normal) / ray_plane_distance

            ray_hit[0] = ray_direction[i, 0] * ray_hit_distance + ray_origin[0]
            ray_hit[1] = ray_direction[i, 1] * ray_hit_distance + ray_origin[1]
            ray_hit[2] = ray_direction[i, 2] * ray_hit_distance + ray_origin[2]

            # calculate barycentric coordinates of ray hit with respect to current poylgon
            # barycentric coordinates are the relation of the area of the sub triangle of polygon edge with a point
            # to the total polygon area
            area_polygon = .5 * norm(normal)
            p = subtract(ray_hit, pos)
            area1 = .5 * norm(cross(v1, p))
            area2 = .5 * norm(cross(v2, p))
            area3 = .5 * norm(cross(subtract(v2, v1), subtract(p, v1)))

            # a point is inside the polygon if all 3 areas of ABP, ACP and BCP add up to area of ABC
            # check for equality considering floating point error
            if abs(area1 + area2 + area3 - area_polygon) < .0001 * area_polygon:
                abs_ray_hit_distance = abs(ray_hit_distance)
                # the hit closest to ray source is the only unoccluded hit
                if abs_ray_hit_distance < closest_hit_distance:
                    # storing temporary results in local thread memory
                    barycentric_coordinates[0] = n
                    barycentric_coordinates[1] = area1 / area_polygon
                    barycentric_coordinates[2] = area2 / area_polygon
                    barycentric_coordinates[3] = area3 / area_polygon
                    closest_hit_distance = abs_ray_hit_distance
                    closest_ray_hit[0] = ray_hit[0]
                    closest_ray_hit[1] = ray_hit[1]
                    closest_ray_hit[2] = ray_hit[2]
                    hit = True
        if hit:
            # calculate uv coordinates of ray hit
            # first find uv coordinates of polygon vertex points
            hit_uv_coordinate_indices = uv_coordinate_indices[barycentric_coordinates[0]]
            p1 = uv_coordinates[hit_uv_coordinate_indices[0]]
            p2 = uv_coordinates[hit_uv_coordinate_indices[1]]
            p3 = uv_coordinates[hit_uv_coordinate_indices[2]]

            # use barycentric coordinates of ray hit to calculate uv
            ray_hit_uv[i, 0] = barycentric_coordinates[1] * p1[0] + \
                               barycentric_coordinates[2] * p2[0] + \
                               barycentric_coordinates[3] * p3[0]
            ray_hit_uv[i, 1] = barycentric_coordinates[1] * p1[1] + \
                               barycentric_coordinates[2] * p2[1] + \
                               barycentric_coordinates[3] * p3[1]
        # transfer results from thread memory to main GPU memory
        # if no hit was found this remains 0 on all entries
        point_cloud[i] = closest_ray_hit


@cuda.jit(device=True)
def cross(v1, v2):
    cv = cuda.local.array(3, float64)
    cv[0] = v1[1] * v2[2] - v1[2] * v2[1]
    cv[1] = v1[2] * v2[0] - v1[0] * v2[2]
    cv[2] = v1[0] * v2[1] - v1[1] * v2[0]
    return cv


@cuda.jit(device=True)
def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@cuda.jit(device=True)
def norm(v):
    return sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@cuda.jit(device=True)
def subtract(v1, v2):
    s = cuda.local.array(3, float64)
    s[0] = v1[0] - v2[0]
    s[0] = v1[1] - v2[1]
    s[0] = v1[2] - v2[2]
    return s


@cuda.jit
def ray_intersection_gpu(ray_origin, ray_direction, vertices, polygons, point_cloud):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ray_hit = cuda.local.array(3, float64)
    closest_ray_hit = cuda.local.array(3, float64)
    closest_ray_hit[0] = 0
    closest_ray_hit[1] = 0
    closest_ray_hit[2] = 0
    closest_hit_distance = inf
    for n in range(len(polygons)):
        pos = vertices[polygons[n, 0]]
        v1_ = vertices[polygons[n, 1]]
        v2_ = vertices[polygons[n, 2]]
        v1 = cuda.local.array(3, float64)
        v2 = cuda.local.array(3, float64)
        v1[0] = v1_[0] - pos[0]
        v1[1] = v1_[1] - pos[1]
        v1[2] = v1_[2] - pos[2]
        v2[0] = v2_[0] - pos[0]
        v2[1] = v2_[1] - pos[1]
        v2[2] = v2_[2] - pos[2]

        normal = cuda.local.array(3, float64)
        normal[0] = v1[1] * v2[2] - v1[2] * v2[1]
        normal[1] = v1[2] * v2[0] - v1[0] * v2[2]
        normal[2] = v1[0] * v2[1] - v1[1] * v2[0]

        ray_plane_distance = ray_direction[i, 0] * normal[0] + \
                             ray_direction[i, 1] * normal[1] + \
                             ray_direction[i, 2] * normal[2]

        if ray_plane_distance == 0:  # protection against zero division, otherwise kernels fail silently
            continue

        ray_hit_distance = ((pos[0] - ray_origin[0]) * normal[0] +
                            (pos[1] - ray_origin[1]) * normal[1] +
                            (pos[2] - ray_origin[2]) * normal[2]) / ray_plane_distance

        ray_hit[0] = ray_direction[i, 0] * ray_hit_distance + ray_origin[0]
        ray_hit[1] = ray_direction[i, 1] * ray_hit_distance + ray_origin[1]
        ray_hit[2] = ray_direction[i, 2] * ray_hit_distance + ray_origin[2]

        inside_polygon = (v1[1] * normal[2] - v1[2] * normal[1]) * (ray_hit[0] - pos[0]) + \
                         (v1[2] * normal[0] - v1[0] * normal[2]) * (ray_hit[1] - pos[1]) + \
                         (v1[0] * normal[1] - v1[1] * normal[0]) * (ray_hit[2] - pos[2]) <= 0
        inside_polygon *= (-v2[1] * normal[2] + v2[2] * normal[1]) * (ray_hit[0] - pos[0]) + \
                          (-v2[2] * normal[0] + v2[0] * normal[2]) * (ray_hit[1] - pos[1]) + \
                          (-v2[0] * normal[1] + v2[1] * normal[0]) * (ray_hit[2] - pos[2]) <= 0
        inside_polygon *= ((v2[1] - v1[1]) * normal[2] - (v2[2] - v1[2]) * normal[1]) * (ray_hit[0] - pos[0] - v1[0]) + \
                          ((v2[2] - v1[2]) * normal[0] - (v2[0] - v1[0]) * normal[2]) * (ray_hit[1] - pos[1] - v1[1]) + \
                          ((v2[0] - v1[0]) * normal[1] - (v2[1] - v1[1]) * normal[0]) * (
                                  ray_hit[2] - pos[2] - v1[2]) <= 0

        if inside_polygon:
            abs_ray_hit_distance = abs(ray_hit_distance)
            if abs_ray_hit_distance < closest_hit_distance:
                closest_hit_distance = abs_ray_hit_distance
                closest_ray_hit[0] = ray_hit[0]
                closest_ray_hit[1] = ray_hit[1]
                closest_ray_hit[2] = ray_hit[2]
    point_cloud[i, 0] = closest_ray_hit[0]
    point_cloud[i, 1] = closest_ray_hit[1]
    point_cloud[i, 2] = closest_ray_hit[2]


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
