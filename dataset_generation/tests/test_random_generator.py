from threading import Thread

from matplotlib import pyplot as plt 
from subt_world_generation.random_tree_generator import RandomTreeGenerator, plot_random_tree
from subt_world_generation.tile_tree import save_tree_as_world_file
import numpy as np    
from time import time_ns as ns
import math



def main():
    N = 10
    step = 50
    MAX = 500
    n_tiles = 105
    tree = RandomTreeGenerator(max_tiles = n_tiles)
    tree.gen_tree()
    tree.full_save(f"/home/lorenzo/Documents/tfm/trees/{n_tiles}_2")
    plot_random_tree(tree,tunnel_axis=False)
    plt.show()

if __name__ == "__main__":
    main()