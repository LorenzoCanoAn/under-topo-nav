from subt_world_generation.random_tree_generator import RandomTreeGenerator, plot_random_tree
import matplotlib.pyplot as plt
import os
from time import time_ns as ns

SCALE = 1.0
BASE_PATH = "/home/lorenzo/catkin_data/worlds"
def main():
    world_name = f"r_{ns()}_s{SCALE}"

    tree = RandomTreeGenerator(max_tiles = 50)
    tree.set_scale(SCALE)
    tree.gen_tree()
    
    plt.figure(figsize=(10,10))
    plot_random_tree(tree)
    plt.show()

    print("Do you want to save [y/n]")
    a = input()
    if "y" in a.lower():
        path = os.path.join(BASE_PATH, world_name)
        tree.full_save(path)
    elif "n" in a.lower():
        return
    else:
        main()


if __name__ == "__main__":
    main()