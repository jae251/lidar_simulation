import numpy as np
import os


def load_obj_file(filename):
    filename = os.path.expanduser(filename)
    with open(filename) as f:
        v, p = [], []
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


def load_Porsche911():
    file = os.path.expanduser("C:/Users/usr/Downloads/3d models/Porsche_911_GT2.obj")
    with open(file) as f:
        v, n, p = [], [], []
        for line in f:
            if line[:2] == "v ":
                q = line.split()
                r = (-float(q[3]), float(q[1]), float(q[2]))
                v.append(r)
            # elif line[:2] == "vn":
            #     q = line.split()
            #     r = (-float(q[3]), float(q[1]), float(q[2]))
            #     n.append(r)
            elif line[0] == "f":
                q = [[int(i) for i in l.split("/")] for l in line.split()[1:]]
                p.append(q)
        vertices = np.array(v)
        z_min = np.min(vertices[:, 2])
        vertices[:, 2] -= z_min
        # normals = np.array(n)
        polygons = np.array(p)[:, :, 0] - 1
        # unused: vertex-normals <normals>, index info <polygons[:,:,1:]
    return vertices, polygons


if __name__ == "__main__":
    vertices, polygons = load_Porsche911()
    from utilities.visualization import visualize_2d

    visualize_2d(vertices)
    visualize_2d(vertices, plane=(0, 2))
    visualize_2d(vertices, plane=(1, 2))
