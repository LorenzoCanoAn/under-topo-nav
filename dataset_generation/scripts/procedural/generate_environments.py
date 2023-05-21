import argparse
import os
import shutil
import pyvista as pv
from subt_proc_gen.tunnel import TunnelNetwork, TunnelNetworkParams
from subt_proc_gen.mesh_generation import (
    TunnelNewtorkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    IntersectionPtClType,
)
import numpy as np
import distinctipy
import logging as log

colors = distinctipy.get_colors(30)

log.basicConfig(level=log.DEBUG)

MODEL_SDF_TEXT = """<?xml version="1.0"?>
<sdf version="1.6">
    <model name="tunnel_network">
        <static>true</static>
        <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def gen_axis_points_file(mesh_generator: TunnelNewtorkMeshGenerator):
    axis_points = np.zeros((0, 3 + 3 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.params_of_tunnel(tunnel).radius
        aps = mesh_generator._aps_of_tunnels[tunnel]
        avs = mesh_generator._avs_of_tunnels[tunnel]
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        axis_points = np.concatenate(
            (axis_points, np.concatenate((aps, avs, rds), axis=1)), axis=0
        )
    for intersection in mesh_generator._tunnel_network.intersections:
        params = mesh_generator.params_of_intersection(intersection)
        if params.ptcl_type == IntersectionPtClType.spherical_cavity:
            radius = params.radius
        elif params.ptcl_type == IntersectionPtClType.no_cavity:
            radiuses = []
            for tunnel in mesh_generator._tunnel_network._tunnels_of_node[intersection]:
                radiuses.append(mesh_generator.params_of_tunnel(tunnel).radius)
            radius = max(radiuses)
        else:
            raise NotImplementedError()
        aps = np.zeros((0, 3))
        avs = np.zeros((0, 3))
        for t in mesh_generator._aps_of_intersections[intersection]:
            aps = np.concatenate(
                (aps, mesh_generator._aps_of_intersections[intersection][t]), axis=0
            )
            avs = np.concatenate(
                (avs, mesh_generator._avs_of_intersections[intersection][t]), axis=0
            )
        assert len(aps) == len(avs) != 0
        rds = np.ones((len(aps), 1)) * radius
        axis_points_of_inter = np.concatenate((aps, avs, rds), axis=1)
        axis_points = np.concatenate((axis_points, axis_points_of_inter), axis=0)
    return axis_points


def args():
    parser = argparse.ArgumentParser(
        prog="DatasetEnvironmentsGenerator",
        description="This script generates N environments and stores them in a folder with the dataset subfolder structure",
    )
    parser.add_argument("-F", "--folder", type=str, required=True)
    parser.add_argument("-N", "--number_of_environments", type=int, required=True)
    parser.add_argument("-NGT", "--number_of_grown_tunnels", default=5, required=False)
    parser.add_argument(
        "-NCT", "--number_of_connector_tunnels", default=2, required=False
    )
    parser.add_argument("-O", "--overwrite", required=False, default=False, type=bool)
    return parser.parse_args()


def main():
    arguments = args()
    base_folder = arguments.folder
    n_envs = arguments.number_of_environments
    n_grown = arguments.number_of_grown_tunnels
    n_connector = arguments.number_of_connector_tunnels
    overwrite = arguments.overwrite
    if os.path.isdir(base_folder):
        if not overwrite:
            raise Exception(
                "Folder already exists, if you want to overwrite it set the argument '-O' to 'true'"
            )
        else:
            shutil.rmtree(base_folder)
    os.mkdir(base_folder)
    for n in range(n_envs):
        base_env_folder = os.path.join(base_folder, f"env_{n+1:03d}")
        os.mkdir(base_env_folder)
        tunnel_network_params = TunnelNetworkParams.from_defaults()
        tunnel_network_params.flat = True
        tunnel_network_params.min_distance_between_intersections = 50
        tunnel_network = TunnelNetwork(params=tunnel_network_params)
        for _ in range(n_grown):
            tunnel_network.add_random_grown_tunnel(n_trials=100)
        for _ in range(n_connector):
            tunnel_network.add_random_connector_tunnel(n_trials=100)
        ptcl_gen_params = TunnelNetworkPtClGenParams.random()
        ptcl_gen_params.general_fta_distance = 1
        mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
        mesh_generator = TunnelNewtorkMeshGenerator(
            tunnel_network,
            ptcl_gen_params=ptcl_gen_params,
            meshing_params=mesh_gen_params,
        )
        mesh_generator.compute_all()
        # plotter = pv.Plotter()
        # for tunnel in mesh_generator._tunnel_network.tunnels:
        #     ptcl = mesh_generator.ptcl_of_tunnel(tunnel)
        #     if len(ptcl) > 0:
        #         plotter.add_mesh(pv.PolyData(ptcl), color="b")
        # for i, intersection in enumerate(mesh_generator._tunnel_network.intersections):
        #     ptcl = mesh_generator.ptcl_of_intersection(intersection)
        #     if len(ptcl) > 0:
        #         plotter.add_mesh(
        #             pv.PolyData(ptcl),
        #             color=colors[i],
        #         )
        axis_points = gen_axis_points_file(mesh_generator)
        # plotter.add_mesh(pv.PolyData(axis_points))
        fta_dist = np.random.uniform(0, 2)
        vertices = mesh_generator.mesh.points
        vertices[np.where(vertices[:, 2] < -fta_dist), 2] = -fta_dist
        path_to_mesh = os.path.join(base_env_folder, "mesh.obj")
        mesh_generator.save_mesh(path_to_mesh)
        np.savetxt(os.path.join(base_env_folder, "axis.txt"), axis_points)
        np.savetxt(os.path.join(base_env_folder, "fta_dist.txt"), np.array((fta_dist,)))
        sdf = MODEL_SDF_TEXT.format(path_to_mesh, path_to_mesh)
        path_to_model_sdf = os.path.join(base_env_folder, "model.sdf")
        with open(path_to_model_sdf, "w") as f:
            f.write(sdf)
        # plotter.add_mesh(mesh_generator.mesh)
        # plotter.show()


if __name__ == "__main__":
    main()
