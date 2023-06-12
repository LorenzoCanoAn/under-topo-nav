import pyvista as pv
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--dataset_folder", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    dataset_folder = args.dataset_folder
    env_folders = os.listdir(dataset_folder)
    env_folders.sort()
    for env_folder in env_folders:
        print(env_folder)
        abs_env_folder = os.path.join(dataset_folder, env_folder)
        mesh_file = os.path.join(abs_env_folder, "mesh.obj")
        mesh = pv.read(mesh_file)
        mesh.flip_normals()
        pv.save_meshio(mesh_file, mesh)


if __name__ == "__main__":
    main()
