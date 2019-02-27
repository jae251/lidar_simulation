lidar_simulation
================

Create a point cloud from a scene built from .obj 3D model files.
Just add a list of models and the scene builder script easily creates random distributions of objects.
The lidar script creates point clouds from these scenes by ray tracing. 
Best used for fast and simple test data creation for object recognition from point clouds.

![Sample scene](https://github.com/jae251/lidar_simulation/blob/master/sample_scene.png "Sample scene")

![Sample point cloud](https://github.com/jae251/lidar_simulation/blob/master/sample_point_cloud.png "Sample point cloud")

## Usage

### The "Lidar" class (in "lidar.py")

The Lidar class internally creates a set of rays, whose intersections with a given 3d object are then calculated.
The difference in the angle between the rays can be specified during initialization of the Lidar class,
as well as the position of the sensor.

The 3d object which is to be sampled needs to be specified as two numpy arrays: A list of vertex positions and a list of indices
that indicate which vertices make up a polygon of the model. Only triangle polygons are allowed.

The sampling method then returns a numpy array in the shape of (n, 3) with n simulated lidar points.

A data loader for .obj Wavefront format is provided under data_loaders/load_3d_models.py 

```python
from data_loaders.load_3d_models import load_obj_file
from lidar import Lidar

vertices, polygon = load_obj_file("sample.obj")
point_cloud = Lidar(delta_azimuth=2 * np.pi / 2000,
                    delta_elevation=np.pi / 800,
                    position=(0, -10, 0)).sample_3d_model_gpu(vertices, polygons)
```

### The "Scene" class (in "scene_builder.py")

This class provides a way to easily compose random scenes from a set of 3d objects.
The following code produced the scene from above:
```python
from scene_builder import Scene
from data_loaders.load_3d_models import load_Porsche911
from data_loaders.create_test_data import create_box, create_rectangle
from shapely.geometry import Polygon

scene = Scene(spawn_area=Polygon(((-10, -10), (-10, 10), (10, 10), (10, -10))))

# Give the scene object the models it can work with and register them with a name string
scene.add_model_to_shelf(*load_Porsche911(), "car")
scene.add_model_to_shelf(*create_box(position=(0, 0, 2), size=(4, 6, 4)), "box")
scene.add_model_to_shelf(*create_rectangle(position=(0, 0, 0), size=(25, 25)), "ground")

# Each call places an object randomly into the scene without intersecting existing objects.
# For this purpose, placed objects are represented by a rectangle in 2d space.
scene.place_object_randomly("car")
scene.place_object_randomly("car")
scene.place_object_randomly("car")
scene.place_object_randomly("box")
scene.place_object_randomly("box")

# place the basic ground object at nonrandom location 
scene.place_object("ground", position=(0, 0, 0), angle=0)

# Compose the scene. Only now are all vertices and polygons of the scene generated.
scene_vertices, scene_polygons = scene.build_scene()

# Get label data from scene
bounding_box_data = scene.get_bounding_boxes()
```

### Building training data sets

```bash
python datasets/build_datset.py
```
This script is an exemplary way of producing training data and should be adapted to your needs.
At the moment scenes with three identical cars and two generic boxes are generated, which are placed randomly
inside a square with the size of 25 meters.