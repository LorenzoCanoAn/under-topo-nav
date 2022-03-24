from threading import Thread 
import random
from subt_world_generation.random_tree_generator import RandomTreeGenerator, plot_random_tree
import matplotlib.pyplot as plt
random.seed(0)

def generator_thread():
    tree = RandomTreeGenerator()
    tree.gen_tree()
    plot_random_tree(tree)
    plot_random_tree(tree)

def main():
    plt.figure(figsize=(10,10))
    th = Thread(target=generator_thread)
    th.start()
    plt.show()
    th.join()
    

if __name__ == "__main__":
    main()