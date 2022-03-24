from matplotlib import pyplot as plt
import subt_world_generation.tile as tl
import pickle

import geometry_msgs.msg as ros_geom_msgs

WORLD_NAME = "w3"
WORLD_SAVE_FOLDER = "/home/lorenzo/catkin_ws/src/subt_gazebo/worlds"
TREE_SAVE_FOLDER = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/saved_trees"
BOILERPLATE_FILE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/text files/boilerplate.txt"

save_world = True
        
def main():
    my_tree = tl.TileTree(WORLD_NAME, scale=0.5)
    my_tree.add_tile("inter",   None, None, None, True) 
    my_tree.add_tile("rect",    0,  0,  1)
    my_tree.add_tile("rect",    1,  0,  1)
    my_tree.add_tile("curve",   2,  0,  0)
    my_tree.add_tile("t",       3,  1,  1)
    my_tree.add_tile("t",       4,  2,  2)
    my_tree.add_tile("block",   5,  0,  0)
    my_tree.add_tile("curve",   5,  1,  1)
    my_tree.add_tile("inter",   7,  0,  2)
    my_tree.add_tile("block",   8,  3,  0)
    my_tree.add_tile("block",   8,  0,  0)
    my_tree.add_tile("curve",   8,  1,  0)
    my_tree.add_tile("rect",    11, 1,  1)
    my_tree.add_tile("block",   12, 0,  0)
    my_tree.add_tile("rect",    0,  1,  1)
    my_tree.add_tile("t",       14, 0,  0)
    my_tree.add_tile("rect",    15, 2,  1)
    my_tree.add_tile("block",   16, 0,  0)
    my_tree.add_tile("rect",    15, 1,  1)
    my_tree.add_tile("curve",   18, 0,  0)
    my_tree.add_tile("block",   19, 1,  0)
    my_tree.add_tile("inter",   0,  3,  1)
    my_tree.add_tile("t",       21, 0,  2)
    my_tree.add_tile("curve",   22, 1,  0)
    my_tree.add_tile("rect",    23, 1,  1)
    my_tree.add_tile("curve",   24, 0,  1)
    my_tree.add_tile("block",   25, 0,  0)
    my_tree.add_tile("rect",    22, 0,  1)
    my_tree.add_tile("block",   27, 0,  0)
    my_tree.add_tile("t",       21, 3,  2)
    my_tree.add_tile("block",   29, 0,  0)
    my_tree.add_tile("t",       29, 1,  2)
    my_tree.add_tile("block",   31, 1,  0)
    my_tree.add_tile("curve",   31, 0,  1)
    my_tree.add_tile("block",   33, 0,  0)
    my_tree.add_tile("rect",    21, 2,  0)
    my_tree.add_tile("curve",   35, 1,  0)
    my_tree.add_tile("curve",   36, 1,  1)
    my_tree.add_tile("curve",   37, 0,  0)
    my_tree.add_tile("rect",    38, 1,  0)
    my_tree.add_tile("rect",    39, 1,  0)
    my_tree.add_tile("block",   40, 1,  0)
    my_tree.add_tile("block",   4,  0,  0)
    my_tree.add_tile("curve",   0,  2,  0)
    my_tree.add_tile("block",   43, 1,  0)
    #my_tree.save(TREE_SAVE_FOLDER)

    my_tree.plot()

 
    my_tree.save("/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/saved_trees/w33")


    

if __name__ == "__main__":
    main()