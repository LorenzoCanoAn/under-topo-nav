from threading import Thread 
from subt_world_generation.random_tree_generator import RandomTreeGenerator, plot_random_tree
import matplotlib.pyplot as plt
from subt_world_generation.tile_tree import save_tree_as_world_file
import cProfile
def generator_thread():
    tree = RandomTreeGenerator(max_tiles = 50)
    tree.gen_tree()
    plot_random_tree(tree,tunnel_axis=True)

def main():
    plt.figure(figsize=(10,10))
    th = Thread(target=generator_thread)
    th.start()
    plt.show()
    th.join()
    

if __name__ == "__main__":
    main()