import argparse
import os
import shutil
import pyvista as pv
from subt_proc_gen.tunnel import (
    TunnelNetwork,
    TunnelNetworkParams,
    Node,
    Tunnel,
    GrownTunnelGenerationParams,
)
from subt_proc_gen.mesh_generation import (
    TunnelNewtorkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
    IntersectionPtClType,
    IntersectionPtClGenParams,
    TunnelPtClGenParams,
)
from subt_proc_gen.display_functions import *
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


def gen_axis_data_file(mesh_generator: TunnelNewtorkMeshGenerator):
    axis_points = np.zeros((0, 3 + 3 + 1))
    for tunnel in mesh_generator._tunnel_network.tunnels:
        radius = mesh_generator.params_of_tunnel(tunnel).radius
        aps = mesh_generator._aps_of_tunnels[tunnel]
        avs = mesh_generator._avs_of_tunnels[tunnel]
        try:
            assert len(aps) == len(avs) != 0
        except:
            continue
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
    return parser.parse_args()


def main():
    arguments = args()
    base_folder = arguments.folder
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    n = len(os.listdir(base_folder))
    base_env_folder = os.path.join(base_folder, f"env_{n+1:03d}")
    os.mkdir(base_env_folder)
    tunnel_network_params = TunnelNetworkParams.from_defaults()
    tunnel_network_params.flat = True
    tunnel_network_params.min_distance_between_intersections = 50
    tunnel_network = TunnelNetwork(params=tunnel_network_params)
    # Generate the network
    node_0_0 = list(tunnel_network.nodes)[0]
    tunnel_0 = Tunnel.grown(
        node_0_0,
        i_direction=(1, 0, 0),
        params=GrownTunnelGenerationParams(200, 0, 0, 0, 0, 20, 30),
    )
    ptcl_gen_params = TunnelNetworkPtClGenParams.random()
    ptcl_gen_params.general_fta_distance = 1
    slim_tunnel_params = TunnelPtClGenParams.from_defaults()
    slim_tunnel_params.radius = 5
    fat_tunnel_params = TunnelPtClGenParams.from_defaults()
    fat_tunnel_params.radius = 7
    fat_tunnel_params.noise_relative_magnitude = 0
    ptcl_gen_params.pre_set_tunnel_params[tunnel_0_0] = slim_tunnel_params
    for intersection in tunnel_network.intersections:
        params = IntersectionPtClGenParams.from_defaults()
        params.radius = 5
        params.ptcl_type = IntersectionPtClType.no_cavity
        ptcl_gen_params.pre_set_intersection_params[intersection] = params
    mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
    mesh_generator = TunnelNewtorkMeshGenerator(
        tunnel_network,
        ptcl_gen_params=ptcl_gen_params,
        meshing_params=mesh_gen_params,
    )
    mesh_generator.compute_all()
    plotter = pv.Plotter()
    for tunnel in mesh_generator._tunnel_network.tunnels:
        ptcl = mesh_generator.ptcl_of_tunnel(tunnel)
        if len(ptcl) > 0:
            plotter.add_mesh(pv.PolyData(ptcl), color="b")
    for i, intersection in enumerate(mesh_generator._tunnel_network.intersections):
        ptcl = mesh_generator.ptcl_of_intersection(intersection)
        if len(ptcl) > 0:
            plotter.add_mesh(
                pv.PolyData(ptcl),
                color=colors[i],
            )

    axis_data = gen_axis_data_file(mesh_generator)
    plotter.add_mesh(pv.PolyData(axis_data[:, :3]))
    fta_dist = mesh_generator._ptcl_gen_params.general_fta_distance
    vertices = mesh_generator.mesh.points
    path_to_mesh = os.path.join(base_env_folder, "mesh.obj")
    mesh_generator.save_mesh(path_to_mesh)
    np.savetxt(os.path.join(base_env_folder, "axis.txt"), axis_data)
    np.savetxt(os.path.join(base_env_folder, "fta_dist.txt"), np.array((fta_dist,)))
    sdf = MODEL_SDF_TEXT.format(path_to_mesh, path_to_mesh)
    path_to_model_sdf = os.path.join(base_env_folder, "model.sdf")
    with open(path_to_model_sdf, "w") as f:
        f.write(sdf)
    plotter.add_mesh(mesh_generator.mesh)
    plotter.show()


if __name__ == "__main__":
    main()
