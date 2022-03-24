from cgi import test
import pickle
import time
import threading
import os
import sys

import rospy
from kill_functions import *
from topographic_map_from_world_tree import topological_graph_from_world_tree
from mine_topological_map.graph import Graph
from mine_topological_map.drawing import GraphDrawing
from subt_world_generation.tile import Tile, TileTree

from subt_world_generation.automatic_world_file_generator import AutoMapGenerator

class Tester:
    def __init__(self, n_instructions, testing_number, n_obstacles_per_tile):
        self.n_instructions = n_instructions
        self.testing_number = testing_number
        self.n_obstacles_per_tile = n_obstacles_per_tile
        self.BASE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/tests_data/test_{}".format(self.testing_number)
        if not os.path.isdir(self.BASE_PATH):
            os.mkdir(self.BASE_PATH)
        self.WORLD_BASE_PATH = os.path.join(self.BASE_PATH,"worlds")
        self.TREE_BASE_PATH = os.path.join(self.BASE_PATH,"trees")
        if not os.path.isdir(self.WORLD_BASE_PATH):
            os.mkdir(self.WORLD_BASE_PATH)
        if not os.path.isdir(self.TREE_BASE_PATH):
            os.mkdir(self.TREE_BASE_PATH)

        if os.listdir(self.WORLD_BASE_PATH).__len__() < 1:
            self.world_generator = AutoMapGenerator(n_instructions=self.n_instructions)
            self.generate_world()
        

    def start_test(self, world_name = None):
        if world_name == None:
            tree_filename, world_filename = self.get_tree_and_world_filenames()
        else:
            tree_filename = str(world_name) + ".pickle"
            world_filename = str(world_name) + ".world"

        with open(os.path.join(self.TREE_BASE_PATH,tree_filename), "rb") as f:
            tree = pickle.load(f)
            assert(isinstance(tree, TileTree))
        graph = topological_graph_from_world_tree(tree)
        topo_instructions, final_coords, path = graph.auto_gen_instructions()
        graph_drawing = GraphDrawing()
        graph_drawing.set_graph(graph)
        graph_drawing.set_path(path)
        # Start thread that generates the world for the next test
        if world_name == None:
            self.world_generator = AutoMapGenerator(n_instructions=self.n_instructions)
            my_thread = threading.Thread(target=self.generate_world)
            my_thread.start()
        else:
            self.world_name = world_name

        # Launch the test test environment
        process = subprocess.Popen(['roslaunch', 
                                    'tunnel_navigation_launch', 
                                    'testing_env.launch', 
                                    "world_file_path:='{}'".format(os.path.join(self.WORLD_BASE_PATH,world_filename)),
                                    'topological_instructions:={}'.format(topo_instructions)])

        # wait for ros to start up and publish the objective coordinates

        time.sleep(2)
        rospy.set_param("/x_obj",final_coords[0])
        rospy.set_param("/y_obj",final_coords[1])
        rospy.set_param("/env_name", self.world_name)
        graph_drawing.run_gui()

        if world_name == None:
            my_thread.join()
        
        return
        
        

    def generate_world(self):
        self.world_generator.gen_map()

        name = str(time.time_ns())
        world_path = os.path.join(self.WORLD_BASE_PATH, name + ".world")
        tree_path = os.path.join(self.TREE_BASE_PATH, name + ".pickle")

        self.world_generator.save(tree_path)
        self.world_generator.to_world_file_text(world_path,n_obstacles_per_tile = self.n_obstacles_per_tile)

    def get_tree_and_world_filenames(self):
        tree_files = sorted(os.listdir(self.TREE_BASE_PATH))
        tree_filename = tree_files[-1]

        world_files = sorted(os.listdir(self.WORLD_BASE_PATH))
        world_filename = world_files[-1]
        self.world_name = world_filename.replace(".world","")
        return tree_filename, world_filename


if __name__ == "__main__":
    print(sys.argv)
    n_instructions = sys.argv[1]
    testing_number = sys.argv[2]
    n_obstacles_per_tile = int(sys.argv[3])
    print("N instructions {} tesing number {} n_obstacles_per_tile {}".format(n_instructions, testing_number, n_obstacles_per_tile))
    hola = Tester(n_instructions=n_instructions, testing_number=testing_number, n_obstacles_per_tile = n_obstacles_per_tile)
    hola.start_test()


