import numpy as np
from utilities.ray_casting import ray_intersection


class Lidar:
    def __init__(self, delta_azimuth=2 * np.pi / 4000, delta_elevation=np.pi / 128, position=(0, 0, 0)):
        self.delta_azimuth = delta_azimuth
        self.delta_elevation = delta_elevation
        self.position = position

    def _tf_into_spherical_sensor_coordinates(self, points):
        pc_tmp = points - self.position
        distance = np.sqrt(np.sum(pc_tmp ** 2, axis=1))
        azimuth = np.arctan2(pc_tmp[:, 1], pc_tmp[:, 0])
        elevation = np.arcsin(pc_tmp[:, 2] / distance)
        points_spherical = np.dstack((distance, azimuth, elevation))[0]
        return points_spherical

    def _tf_into_cartesian_coordinates(self, points_spherical):
        r = points_spherical[:, 0]
        az = points_spherical[:, 1]
        el = points_spherical[:, 2]
        cos_el = np.cos(el)
        x = r * cos_el * np.cos(az)
        y = r * cos_el * np.sin(az)
        z = r * np.sin(el)
        points = np.dstack((x, y, z))[0]
        return points

    def _model_view_dimensions(self, pcloud_spherical):
        azimuth_min, elevation_min = np.min(pcloud_spherical[:, 1:], axis=0)
        azimuth_max, elevation_max = np.max(pcloud_spherical[:, 1:], axis=0)
        return azimuth_min, azimuth_max, elevation_min, elevation_max

    ####################################################################################################################
    def create_rays(self, vertices):
        vertices_spherical = self._tf_into_spherical_sensor_coordinates(vertices)
        azimuth_min, azimuth_max, elevation_min, elevation_max = self._model_view_dimensions(vertices_spherical)
        n1 = int((azimuth_max - azimuth_min) / self.delta_azimuth)
        n2 = int((elevation_max - elevation_min) / self.delta_elevation)
        az = np.linspace(azimuth_min, azimuth_max, n1)
        el = np.linspace(elevation_min, elevation_max, n2)
        rays_spherical = np.dstack((np.ones(n1 * n2), *[m.ravel() for m in np.meshgrid(az, el)]))[0]
        ray_directions = self._tf_into_cartesian_coordinates(rays_spherical)
        return self.position, ray_directions

    def sample_3d_model(self, vertices, polygons, rays_per_cycle=None):
        '''
        Simulate lidar sensor measurement on a 3d model
        :param  vertices: np.array with x,y,z as columns (shape= n x 3)
                polygons: np.array with vertex indices for each polygon (shape= p x 3),
                          assumes 3-point-polygons
        :return: Measured vertices (shape= m x 3)
        '''
        ray_origin, ray_directions = self.create_rays(vertices)
        if rays_per_cycle is None:
            sampled_points = ray_intersection(ray_origin, ray_directions, vertices, polygons)
        else:
            cycles = int(np.ceil(len(ray_directions) / rays_per_cycle))
            sampled_points = []
            for c in range(cycles):
                idx = c * rays_per_cycle
                sampled_points.append(
                    ray_intersection(ray_origin, ray_directions[idx:idx + rays_per_cycle], vertices, polygons))
            sampled_points = np.vstack(sampled_points)
        return sampled_points


########################################################################################################################

def sample_usage():
    from data_loaders.load_3d_models import load_Porsche911
    from utilities.visualization import visualize_3d

    point_cloud = Lidar(delta_azimuth=2 * np.pi / 2000,
                        delta_elevation=np.pi / 500,
                        position=(0, -10, 0)).sample_3d_model(*load_Porsche911(), rays_per_cycle=400)
    print(point_cloud)
    visualize_3d(point_cloud)


if __name__ == "__main__":
    sample_usage()