from scene_builder import Scene
import numpy as np
from lidar import Lidar
from data_loaders.load_3d_models import load_Porsche911
from data_loaders.create_test_data import create_box, create_rectangle
from h5py import File
import os
from time import time

t1 = time()

scene = Scene(spawn_area_size=(20, 20))
scene.add_model_to_shelf(*load_Porsche911(), "car")
scene.add_model_to_shelf(*create_box(position=(0, 0, 2), size=(4, 6, 4)), "box")
scene.add_model_to_shelf(*create_rectangle(position=(0, 0, 0), size=(25, 25)), "ground")

with File("dataset_01.hdf5", "w") as hdf:
    for n in range(100):
        print(n)
        scene.place_object_randomly("car")
        scene.place_object_randomly("car")
        scene.place_object_randomly("car")
        scene.place_object_randomly("box")
        scene.place_object_randomly("box")
        scene.place_object("ground", position=(0, 0), angle=0)

        scene_vertices, scene_polygons = scene.build_scene()

        path = os.path.join(os.path.dirname(__file__),
                            "{:>02}.png".format(n))
        scene.visualize(scene_vertices, path)
        point_cloud = Lidar(delta_azimuth=2 * np.pi / 4000,
                            delta_elevation=np.pi / 500,
                            position=(0, -40, 1)).sample_3d_model(scene_vertices,
                                                                  scene_polygons,
                                                                  rays_per_cycle=400)
        group = hdf.create_group("{:>02}".format(n))
        group.create_dataset("point_cloud", data=point_cloud)
        group.create_dataset("bounding_boxes", data=scene.get_bounding_boxes())
        scene.clear()

t2 = time()
print("Elapsed time: {}".format(t2 - t1))