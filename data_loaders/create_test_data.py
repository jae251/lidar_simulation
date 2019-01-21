'''
Creating random point clouds from primitives for fun and profit
'''
import numpy as np


def create_sphere(n=1000):
    distance = 1
    azimuth = np.random.uniform(0, 2 * np.pi, n)
    elevation = np.random.uniform(-np.pi / 2, np.pi / 2, n)

    xyz = np.dstack((distance * np.cos(azimuth) * np.cos(elevation),
                     distance * np.sin(azimuth) * np.cos(elevation),
                     distance * np.sin(elevation)))[0]
    return xyz


def create_L_shape(n=1000):
    length, width, height = np.random.uniform(2, 10, 3)
    angle = np.random.uniform(0, 2 * np.pi, 1)
    l = np.array((length * np.cos(angle), length * np.sin(angle), 0)).reshape(-1, 1)
    b = np.array((-width * np.sin(angle), width * np.cos(angle), 0)).reshape(-1, 1)

    # p1 = l + b
    # p2 = l - b
    # p3 = -l - b

    slim = np.random.uniform(.9, 1.1, n)
    wide = np.random.uniform(-1.1, 1.1, n)
    cut = int(round(n * width / (length + width)))
    xyz = np.zeros((n, 3))
    xyz1 = l * slim[:cut] + b * wide[:cut]
    xyz2 = l * wide[cut:] + b * slim[cut:]
    xyz[:cut] = xyz1.transpose()
    xyz[cut:] = xyz2.transpose()
    xyz[:, 2] = np.random.uniform(0, .2, n)
    return xyz


def create_box(position=(0, 0, 0), size=(1, 1, 1)):
    position = np.array(position)
    l = np.array((size[0] / 2, 0, 0))
    b = np.array((0, size[1] / 2, 0))
    h = np.array((0, 0, size[2] / 2))
    p1 = position + l + b + h
    p2 = position + l - b + h
    p3 = position + l + b - h
    p4 = position + l - b - h
    p5 = position - l + b + h
    p6 = position - l - b + h
    p7 = position - l + b - h
    p8 = position - l - b - h
    vertices = np.vstack((p1, p2, p3, p4, p5, p6, p7, p8))
    polygons = np.array(((0, 1, 2),
                         (1, 2, 3),
                         (0, 2, 4),
                         (2, 4, 6),
                         (4, 5, 6),
                         (5, 6, 7),
                         (1, 3, 5),
                         (3, 5, 7),
                         (0, 1, 4),
                         (1, 4, 5),
                         (2, 3, 6),
                         (3, 6, 7)))
    return vertices, polygons


def create_rectangle(position=(0, 0, 0), size=(10, 10)):
    position = np.array(position)
    l = np.array((size[0] / 2, 0, 0))
    b = np.array((0, size[1] / 2, 0))
    p1 = position + l + b
    p2 = position + l - b
    p3 = position - l - b
    p4 = position - l + b
    vertices = np.vstack((p1, p2, p3, p4))
    polygons = np.array(((0, 1, 2),
                         (0, 2, 3)))
    return vertices, polygons


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # xyz = create_sphere()
    xyz = create_L_shape()

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    plt.show()
