from subt_world_generation.tile import Tile

def get_fusioned_axis(tile:Tile):
    for i in range(tile.n_connections):
        p = tile.connection_points[i]
        neigh = tile.connections[i]

    return normal_axis, fusioned_axis