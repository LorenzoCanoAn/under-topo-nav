import pickle
import time
import threading
import os
import sys

import rospy
import subprocess
from topographic_map_from_world_tree import topological_graph_from_world_tree
from mine_topological_map.graph import Graph
from mine_topological_map.drawing import GraphDrawing
from subt_world_generation.tile import Tile, TileTree

from subt_world_generation.automatic_world_file_generator import AutoMapGenerator

class Tester:
    def __init__(self, testing_run):
        self.WORLD_BASE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/tests_data/test_{}/worlds".format(testing_run)
        self.TREE_BASE_PATH =  "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/tests_data/test_{}/trees".format(testing_run)

        

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
        print(topo_instructions)
        time.sleep(5)
        graph_drawing = GraphDrawing()
        graph_drawing.set_graph(graph)
        graph_drawing.set_path(path)
        # Start thread that generates the world for the next test
        if world_name == None:
            self.world_generator = AutoMapGenerator(n_instructions=int(sys.argv[-1]))
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
        rospy.set_param("/env_name", str(self.world_name))
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
        self.world_generator.to_world_file_text(world_path)

    def get_tree_and_world_filenames(self):
        tree_files = sorted(os.listdir(self.TREE_BASE_PATH))
        tree_filename = tree_files[-1]

        world_files = sorted(os.listdir(self.WORLD_BASE_PATH))
        world_filename = world_files[-1]
        self.world_name = world_filename.replace(".world","")
        return tree_filename, world_filename


if __name__ == "__main__":
    testing_run = 1
    map_to_test = 1645539527945761587
    hola = Tester(testing_run)
    hola.start_test(map_to_test)


