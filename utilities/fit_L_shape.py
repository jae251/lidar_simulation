from scipy.spatial import ConvexHull
import numpy as np


def fit_L_shape(pcloud, threshold=1):
    hull = ConvexHull(pcloud)
    vertices = pcloud[hull.vertices]
    l = len(vertices)
    a_list, p_list, lincs, point_count = [], [], [], []
    vertex_pair_indices = list(zip(np.roll(np.arange(l), 1), np.arange(l)))

    for n1, n2 in vertex_pair_indices:
        p0 = vertices[n1]
        edge = vertices[n2] - p0
        edge /= np.linalg.norm(edge)
        normal = np.array((edge[1], -edge[0]))
        normal *= threshold / np.linalg.norm(normal)
        a = np.dstack((edge[:2], normal))[0]
        b = pcloud - p0[:2]
        linc = np.linalg.solve(a, b.transpose()).transpose()
        points_in_threshold = np.sum(linc[:, 1] > -1)
        lincs.append(linc)
        a_list.append(a)
        p_list.append(p0[:2])
        point_count.append(points_in_threshold)

    max_point_count = max(point_count)
    idx = point_count.index(max_point_count)
    # print(idx, max_point_count)

    upper_bound = np.max(lincs[idx], axis=0)
    lower_bound = np.min(lincs[idx], axis=0)

    up = upper_bound[0]
    low = up - threshold
    sum1 = np.sum((low <= lincs[idx][:, 0]) * (lincs[idx][:, 0] <= up))
    # print(sum1)
    low = lower_bound[0]
    up = low + threshold
    sum2 = np.sum((low <= lincs[idx][:, 0]) * (lincs[idx][:, 0] <= up))
    # print(sum2)

    p0 = p_list[idx]
    edge = a_list[idx][:, 0]
    normal = a_list[idx][:, 1]

    p1 = p0 + upper_bound[0] * edge
    p2 = p0 + lower_bound[0] * edge
    if sum1 >= sum2:
        p3 = p1 + lower_bound[1] * normal
        l_shape = np.vstack((p3, p1, p2))
    else:
        p3 = p2 + lower_bound[1] * normal
        l_shape = np.vstack((p1, p2, p3))

    visualize(pcloud[:, :2], vertices, l_shape)
    return l_shape


def visualize(pcloud, outline=None, l_shape=None):
    import matplotlib.pyplot as plt
    plt.plot(pcloud[:, 0], pcloud[:, 1], "o")
    if outline is not None:
        outline = np.vstack((outline, outline[0]))
        plt.plot(outline[:, 0], outline[:, 1], "r--", lw=2)
    if l_shape is not None:
        plt.plot(l_shape[:, 0], l_shape[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    from data_loaders.create_test_data import create_L_shape

    test_data = create_L_shape((200))
    p1, p2, p3 = fit_L_shape(test_data[:, :2])
