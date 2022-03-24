from subt_world_generation.tile_tree import TileTree, plot_tree, tree_tiles_to_text
from subt_world_generation.tile import Tile, ALIAS
import matplotlib.pyplot as plt


def main():
    tile_tree = TileTree()
    tile_tree.set_scale(2)
    tile_tree.add_tile(Tile(ALIAS["tunnel_4_way_intersection"]))

    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[0],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[1],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[2],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     1,  tile_tree[3],  1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[4],  2)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[5],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[5],  1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_4_way_intersection"]),    2,  tile_tree[7],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[8],  3)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[8],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[8],  1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[11], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[12], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[0],  1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     0,  tile_tree[14], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[15], 2)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[16], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[15], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[18], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[19], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_4_way_intersection"]),    1,  tile_tree[0],  3)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[21], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[22], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[23], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[24], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[25], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[22], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[27], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[21], 3)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[29], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[29], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[31], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[31], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[33], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[21], 2)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[35], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[36], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[37], 0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[38], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[39], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[40], 1)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[4],  0)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[0],  2)
    tile_tree.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[43], 1)



    plt.figure(figsize=(10,10))
    plot_tree(tile_tree)
    plt.show()

if __name__ == "__main__":
    main()