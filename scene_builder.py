import numpy as np
from utilities.bounding_boxes import BoundingBox2D
from utilities.geometry_calculations import rotate_point_cloud
import matplotlib.pyplot as plt


class Scene:
    '''
    Initialize with area size, where objects should be distributed in.
    Use <add_model_to_shelf> to give the scene-object placeable models.
    Use <place_object_randomly> to place a selected object somewhere in the scene.
        A box representation is used to find an unoccupied place in the scene.
        Use repeatedly, until the desired number of objects is reached.
    Lastly, use <build_scene> to create the new model.
    '''

    def __init__(self, spawn_area_size=(20, 20)):
        self.spawn_area_size = np.array(spawn_area_size)
        self.model_boundaries = {}
        self.models = {}
        self.scene = []

    def add_model_to_shelf(self, vertices, polygons, label):
        self.model_boundaries[label] = BoundingBox2D.from_point_cloud(vertices, label=label)
        self.models[label] = vertices, polygons

    def place_object_randomly(self, label):  # if there is no space for an additional object, this runs forever
        obj_blueprint = self.model_boundaries[label]
        while True:
            new_obj = obj_blueprint.copy()
            random_nr = np.random.uniform(0, 1, 3)
            position = (random_nr[:2] - .5) * self.spawn_area_size
            angle = 2 * np.pi * random_nr[2]
            new_obj.affine_transform(position, angle)
            if any((new_obj.overlaps(obj) for obj in self.scene)):
                continue
            break
        self.scene.append(new_obj)

    def place_object(self, label, position, angle):  # angle in radians
        position = np.array(position)
        obj_blueprint = self.model_boundaries[label]
        new_obj = obj_blueprint.copy()
        new_obj.affine_transform(position, angle)
        self.scene.append(new_obj)

    def build_scene(self):
        scene_vertices, scene_polygons = [], []
        polygon_count = 0
        for obj in self.scene:
            vertices, polygons = self.models[obj.label]
            vertices = rotate_point_cloud(vertices, obj.angle)
            vertices[:, :2] += obj.position
            scene_vertices.append(vertices)
            scene_polygons.append(polygons + polygon_count)
            polygon_count += len(vertices)
        scene_vertices = np.vstack(scene_vertices)
        scene_polygons = np.vstack(scene_polygons)
        return scene_vertices, scene_polygons

    def get_bounding_boxes(self):
        return np.array([obj.to_numpy() for obj in self.scene])

    def clear(self):
        self.scene = []

    def visualize(self, point_cloud=None, path=None):
        for obj in self.scene:
            p = np.vstack((obj.points, obj.points[0]))
            plt.plot(p[:, 0], p[:, 1], "r--", lw=2)
        if point_cloud is not None:
            plt.plot(point_cloud[:, 0], point_cloud[:, 1], "o", markersize=1)
        plt.gca().set_aspect('equal', adjustable='box')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.gcf().clear()


########################################################################################################################

def sample_usage():
    from lidar import Lidar
    from data_loaders.load_3d_models import load_Porsche911
    from data_loaders.create_test_data import create_box, create_rectangle
    import pptk
    import os

    scene = Scene(spawn_area_size=(20, 20))
    scene.add_model_to_shelf(*load_Porsche911(), "car")
    scene.add_model_to_shelf(*create_box(position=(0, 0, 2), size=(4, 6, 4)), "box")
    scene.add_model_to_shelf(*create_rectangle(position=(0, 0, 0), size=(25, 25)), "ground")

    scene.place_object_randomly("car")
    scene.place_object_randomly("car")
    scene.place_object_randomly("car")
    scene.place_object_randomly("box")
    scene.place_object_randomly("box")
    scene.place_object("ground", position=(0, 0), angle=0)

    scene_vertices, scene_polygons = scene.build_scene()
    path = os.path.join(os.path.dirname(__file__),
                        "scene_sample.png")
    scene.visualize(scene_vertices, path=path)

    point_cloud = Lidar(delta_azimuth=2 * np.pi / 4000,
                        delta_elevation=np.pi / 500,
                        position=(0, -40, 1)).sample_3d_model(scene_vertices,
                                                              scene_polygons,
                                                              rays_per_cycle=400)

    v = pptk.viewer(point_cloud)
    v.set(point_size=.01)


if __name__ == "__main__":
    sample_usage()