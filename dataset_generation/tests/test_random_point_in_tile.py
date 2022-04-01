from matplotlib import pyplot as plt
from subt_dataset_generation.training_points_2d import random_points_in_tile_tree
from subt_world_generation.tile_tree import TileTree, plot_tree
from subt_world_generation.tile import ALIAS, Tile
from time import time_ns as ns
from subt_dataset_generation.training_points_2d import random_points_in_tile
import threading

def main():
    tree = gen_tree()
    p, o, l = random_points_in_tile_tree(tree,5)
    t = threading.Thread(target=fuck,args=[tree, p,l])
    t.start()
    plt.show()
    t.join()

def fuck(tree , p, l):
    i=0
    while i < 100:
        input()
        i+=5
        f1 = plt.figure(figsize=(10,10))
        plot_tree(tree,tunnel_axis=False)
        plt.scatter(p[0,i],p[1,i],c="r")
        f2 = plt.figure()
        plt.plot(l[:,i])
        plt.show()


def gen_tree():
    tile_tree = TileTree()
    tile_tree.set_scale(1)
    tile_tree.add_tile(Tile(ALIAS["tunnel_4_way_intersection"]))

    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[0],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[1],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[2],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     1,  tile_tree[3],  1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[4],  2)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[5],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[5],  1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_4_way_intersection"]),    2,  tile_tree[7],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[8],  3)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[8],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[8],  1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[11], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[12], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[0],  1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     0,  tile_tree[14], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[15], 2)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[16], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[15], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[18], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[19], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_4_way_intersection"]),    1,  tile_tree[0],  3)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[21], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[22], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[23], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[24], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[25], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  1,  tile_tree[22], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[27], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[21], 3)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[29], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_t"]),                     2,  tile_tree[29], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[31], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[31], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[33], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[21], 2)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[35], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 1,  tile_tree[36], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[37], 0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[38], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_rect"]),                  0,  tile_tree[39], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[40], 1)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[4],  0)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_curve"]),                 0,  tile_tree[0],  2)
    tile_tree.move_add_and_connect_tile(
        Tile(ALIAS["tunnel_block"]),                 0,  tile_tree[43], 1)
    return tile_tree


if __name__ == "__main__":
    main()
