import numpy as np
from subt_world_generation.tile import Tile


def get_fusioned_axis(tile: Tile):
    fused_axis = dict()
    for i in range(tile.n_connections):
        own_axis = tile.connection_points[i].associated_axis
        neigh = tile.connections[i]
        neigh_conn_idx = neigh.connections.index(tile)
        neigh_axis = neigh.connection_points[neigh_conn_idx].associated_axis
        new_axs = []
        for axs in tile.axis:
            if axs is not own_axis or neigh_axis is None:
                new_axs.append(axs.points)
            else:
                new_axs.append(np.vstack((neigh_axis.points, axs.points)))

        fused_axis[tile.connection_points[i]] = new_axs
    new_axs = []
    for axs in tile.axis:
        new_axs.append(axs.points)
    fused_axis[None] = new_axs
    return fused_axis
