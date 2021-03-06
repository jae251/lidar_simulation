import numpy as np
import os


def load_obj_file(filename, texture=False):
    filename = os.path.expanduser(filename)
    with open(filename) as f:
        v, p, uv = [], [], []
        for line in f:
            if line[:2] == "v ":
                q = line.split()
                r = (-float(q[3]), float(q[1]), float(q[2]))
                v.append(r)
            elif line[0] == "f":
                q = [[int(i) for i in l.split("/")] for l in line.split()[1:]]
                p.append(q)
            elif line[:2] == "vt":
                q = line.split()
                uv.append((float(q[1]), float(q[2])))
        vertices = np.array(v)
        z_min = np.min(vertices[:, 2])
        vertices[:, 2] -= z_min
        p = np.array(p) - 1
        polygons = p[:, :, 0].copy()
    if texture:
        uv_coordinates = np.array(uv)
        uv_coordinate_indices = p[:, :, 1].copy()
        return vertices, polygons, uv_coordinates, uv_coordinate_indices
    else:
        return vertices, polygons


def load_Porsche911():
    file = os.path.expanduser("~/Downloads/3d_models/Porsche_911_GT2.obj")
    with open(file) as f:
        v, n, p = [], [], []
        for line in f:
            if line[:2] == "v ":
                q = line.split()
                r = (-float(q[3]), float(q[1]), float(q[2]))
                v.append(r)
            elif line[0] == "f":
                q = [[int(i) for i in l.split("/")] for l in line.split()[1:]]
                p.append(q)
        vertices = np.array(v)
        z_min = np.min(vertices[:, 2])
        vertices[:, 2] -= z_min
        polygons = np.array(p)[:, :, 0] - 1
    return vertices, polygons


if __name__ == "__main__":
    vertices, polygons = load_Porsche911()
    from utilities.visualization import visualize_2d

    visualize_2d(vertices)
    visualize_2d(vertices, plane=(0, 2))
    visualize_2d(vertices, plane=(1, 2))
