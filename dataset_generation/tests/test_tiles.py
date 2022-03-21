from world_file_generation.tile import plot_seq_of_tiles
from world_file_generation.tile_tree import TileTree, plot_tree
import matplotlib.pyplot as plt

ALIAS = {
    "blocker":"tunnel_tile_blocker",
    "rect": "tunnel_tile_5",
    "t":"tunnel_intersection_t",
    "intersection":"tunnel_tile_1",
    "curve":"tunnel_tile_2"
}
def main():
    tile_tree = TileTree()
    tile_tree.set_scale(2)
    tile_tree.add_tile(ALIAS["t"], None, None, None)
    tile_tree.add_tile(ALIAS["intersection"],0, 0,0)
    tile_tree.add_tile(ALIAS["curve"],-1,1,1)
    tile_tree.add_tile(ALIAS["curve"],-1,0,1)
    tile_tree.add_tile(ALIAS["curve"],-1,0,1)
    tile_tree.add_tile(ALIAS["curve"],-1,0,1)



    plt.figure(figsize=(5,5))
    plot_tree(tile_tree)
    plt.show()

if __name__ == "__main__":
    main()