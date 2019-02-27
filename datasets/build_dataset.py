from scene_builder import Scene
import numpy as np
from lidar import Lidar
from data_loaders.load_3d_models import load_Porsche911
from data_loaders.create_test_data import create_box, create_rectangle
from h5py import File
import os
from time import time
from multiprocessing import Pool

t1 = time()

scene = Scene()
scene.add_model_to_shelf(*load_Porsche911(), "car")
scene.add_model_to_shelf(*create_box(position=(0, 0, 2), size=(4, 6, 4)), "box")
scene.add_model_to_shelf(*create_rectangle(position=(0, 0, 0), size=(25, 25)), "ground")


def render_scene(n):
    with File("dataset_{:>02}.hdf5".format(n), "w") as hdf:
        print(n)
        scene.place_object_randomly("car")
        scene.place_object_randomly("car")
        scene.place_object_randomly("car")
        scene.place_object_randomly("box")
        scene.place_object_randomly("box")
        scene.place_object("ground", position=(0, 0, 0), angle=0)

        scene_vertices, scene_polygons = scene.build_scene()

        path = os.path.join(os.path.dirname(__file__),
                            "{:>02}.png".format(n))
        scene.visualize(scene_vertices, path)
        point_cloud = Lidar(delta_azimuth=2 * np.pi / 4000,
                            delta_elevation=np.pi / 500,
                            position=(0, -40, 1)).sample_3d_model_gpu(scene_vertices,
                                                                      scene_polygons)
        hdf.create_dataset("point_cloud", data=point_cloud)
        hdf.create_dataset("bounding_boxes", data=scene.get_bounding_boxes())
        scene.clear()


pool = Pool()
pool.map(render_scene, range(100))

t2 = time()
print("Elapsed time: {}".format(t2 - t1))
