import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_2d(pcloud, outline=None, markersize=1, plane=(0, 1)):
    plt.plot(pcloud[:, plane[0]], pcloud[:, plane[1]], "o", markersize=markersize)
    if outline is not None:
        outline = np.vstack((outline, outline[0]))
        plt.plot(outline[:, 0], outline[:, 1], "r--", lw=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def visualize_3d(pcloud, markersize=1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pcloud[:, 0], pcloud[:, 1], pcloud[:, 2], s=markersize)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
