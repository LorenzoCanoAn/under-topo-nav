import pyvista as pv
import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        "environment_checker",
        description="Give a dataset folder, and the program will sequentially open the meshes to allow for visual inspection",
    )
    parser.add_argument("-F", "--folder", required=True, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    dataset_folder = args.folder
    envs = os.listdir(dataset_folder)
    envs.sort()
    for env_folder in envs:
        mesh_file = os.path.join(dataset_folder, env_folder, "mesh.obj")
        aps_file = os.path.join(dataset_folder, env_folder, "axis.txt")
        axis_info = np.loadtxt(aps_file)
        aps = axis_info[:, :3]
        rs = axis_info[:, -1]
        plotter = pv.Plotter(title=env_folder)
        plotter.add_mesh(pv.read(mesh_file), style="wireframe")
        plotter.add_mesh(pv.PolyData(aps), scalars=rs, point_size=10)
        plotter.show()
    pass


if __name__ == "__main__":
    main()
