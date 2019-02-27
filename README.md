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

