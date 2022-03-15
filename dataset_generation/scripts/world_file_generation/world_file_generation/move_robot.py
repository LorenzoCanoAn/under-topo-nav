#!/bin/python3

# ------------------------------------------------------------
# Descripción:
# El funcionamiento de este script debería ser:
# -1: se lee un documento que contiene las areas en las que se quiere tener el robot, y lo que debería de detectar en cada una
# -2: comienza a mover el robot dentro de cada area
#       - para cada posición, coje un numero de lecturas del lidar y las almacena junto con su etiqueta
# -3: itera en todas las areas
import sys
import matplotlib.pyplot as plt
import tile
import rospy
import time
from datetime import datetime
import random
import gazebo_msgs.msg
import gazebo_msgs.srv
import geometry_msgs.msg
import sensor_msgs.msg
from image_tools import ImageTools
import xml.etree.ElementTree as ET
import os
import pickle

from scipy.spatial.transform import Rotation

from shapely.geometry import Polygon, Point

import numpy as np
import torch


class RobotMover:
    def __init__(self):
        self.get_parameters()
        self.publisher = rospy.Publisher(
            "/gazebo/set_model_state",
            gazebo_msgs.msg._ModelState.ModelState,
            queue_size=5,
        )
        self.message = gazebo_msgs.msg._ModelState.ModelState()
        self.message.model_name = self.model_name

    def get_parameters(self):
        self.model_name = rospy.get_param("model_name", "/")
        self.time_interval = rospy.get_param("time_interval", 0.5)

    def send_position(self, pose=None):
        if type(pose) == geometry_msgs.msg.Pose:
            self.message.pose = pose

        self.publisher.publish(self.message)

class message_storage:
    # Class used to ensure that a laser_scan message is not stored twice.
    def __init__(self):
        self.data = None
        self.fresh = False

    def store(self, msg):
        self.data = msg
        self.fresh = True

    def block(self):
        while not self.fresh:
            pass
        self.fresh = False

TREE_NAME = "w3"

if __name__ == "__main__":
    LASER_SCAN_DATASETS_FOLDER = "/home/lorenzo/Datasets/environment_classification"
    IMAGE_DATASETS_FOLDER = "/home/lorenzo/Documents/PAPERS/IROS2022/figures"
    TILE_TREE_FILE = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/saved_trees/" + TREE_NAME
    SAMPLES_PER_TILE = 300

    
    tile_tree = tile.TileTree("temp")
    tile_tree.load(TILE_TREE_FILE)

    t = datetime.now()
    time_string = "{}_{}_{}_{}_{}".format(
        t.year, t.month, t.day, t.hour, t.minute)


    dataset_name = "{}_{}".format(SAMPLES_PER_TILE, time_string)
    dataset_name = "laserscan_image" + TREE_NAME


    laser_dataset_path = LASER_SCAN_DATASETS_FOLDER + "/" + dataset_name
    image_dataset_path = IMAGE_DATASETS_FOLDER + "/" + dataset_name
    if not os.path.isdir(laser_dataset_path):
        os.mkdir(laser_dataset_path)
    if not os.path.isdir(image_dataset_path):
        os.mkdir(image_dataset_path)

    random.seed(time.localtime().tm_sec)

    rospy.init_node("hola")
    service_name = "/gazebo/get_model_state"
    rospy.wait_for_service(service_name)
    get_model_state = rospy.ServiceProxy(
        service_name, gazebo_msgs.srv.GetModelState)
    store_laser = message_storage()
    store_image = message_storage()
    scan_subscriber = rospy.Subscriber(
        "/scan", sensor_msgs.msg.LaserScan, store_laser.store
    )
    image_subscriber = rospy.Subscriber(
        "/laserscan_image", sensor_msgs.msg.Image, store_image.store
    )
    image_tools = ImageTools()
    robot_mover = RobotMover()
    time.sleep(1)  # Wait for the robot_mover to work
    a = True
    numpy_laser_dataset = None
    numpy_image_dataset = None

    id_laser = 0
    id_image = 0
    
    for j, t in enumerate(tile_tree.tiles):
        
        print("\r\r", end="")
        print("{} of {}".format(j+1, tile_tree.tiles.__len__()))

        
        poses, vectors = t.gen_rand_pose_msgs_and_vector(n=SAMPLES_PER_TILE)
        for n , (pose, vector) in enumerate(zip(poses, vectors)):
            
            robot_mover.send_position(pose)
            time.sleep(0.2)
            trials = 0
            while not rospy.is_shutdown():
                recieved_pose = get_model_state(
                    robot_mover.model_name, "").pose

                # Check that the robot has moved
                if (
                    recieved_pose.position.x - pose.position.x < 0.1
                    and recieved_pose.position.y - pose.position.y < 0.1
                ):
                    store_laser.fresh = False
                    store_image.fresh = False
                    # --------------------
                    # LASER
                    # --------------------

                #     store_laser.block()  # wait for new laser_scan
                #     l_data = np.array(store_laser.data.ranges)
                #     l_data = torch.tensor(
                #         np.where(l_data > 20.0, 20.0, l_data)).float()
                #     l_label = torch.tensor(
                #         np.array(t.get_numeric_label(), dtype=int)).int()
                #     laser_file = laser_dataset_path + "/" + str(id_laser) + ".pt"
                #     torch.save((l_data, l_label), laser_file)
                #     id_laser +=1


                    # --------------------
                    # IMAGE
                    # --------------------
                    store_image.block()  # wait for new image
                    depth_image_msg = store_image.data
                    i_data = torch.tensor(image_tools.convert_ros_msg_to_cv2(
                        depth_image_msg, image_encoding="mono8"
                    )).int()
                    i_label = torch.tensor(vector).float()

                    recieved_pose = get_model_state(
                    robot_mover.model_name, "").pose

                # Check that the robot has moved
                    if not(recieved_pose.position.x - pose.position.x < 0.1
                        and recieved_pose.position.y - pose.position.y < 0.1):
                        print("OUT")
                        break
                    image_file = image_dataset_path + "/" + str(id_image) + ".pt"
                    torch.save((i_data, i_label), image_file)
                    id_image += 1
                    # fig,axs = plt.subplots(2)
                    # axs[0].plot(vector)
                    # axs[1].imshow(i_data.numpy())
                    # for ax in axs: ax.set_xlim((0,359))
                    # plt.show() 
                    break
                else:
                    trials += 1
                    if trials > 1:
                        print("OUT")
                        break
            input()
                