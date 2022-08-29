#!/bin/python3
import time
import torch
import pickle
import os
import gazebo_msgs.msg
import gazebo_msgs.srv
import sensor_msgs.msg
import geometry_msgs.msg
from subt_dataset_generation.training_points_2d import random_label_in_tile
import rospy
import numpy as np
from cv_bridge import CvBridge
from subt_world_generation.tile_tree import TileTree


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return qx, qy, qz, qw


class RobotMover:
    def __init__(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        self.service_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", gazebo_msgs.srv.SetModelState)

    def send_position(self, p, o):
        position = geometry_msgs.msg.Point(p[0], p[1], 0.15)
        qx, qy, qz, qw = get_quaternion_from_euler(0, 0, o)
        orientation = geometry_msgs.msg.Quaternion(qx, qy, qz, qw)
        pose = geometry_msgs.msg.Pose(position, orientation)
        twist = geometry_msgs.msg.Twist(geometry_msgs.msg.Vector3(
            0, 0, 0), geometry_msgs.msg.Vector3(0, 0, 0))
        request = gazebo_msgs.msg.ModelState("/", pose, twist, "")
        response = self.service_proxy(request)


class ImageStorage:
    def __init__(self):
        self._sub = rospy.Subscriber(
            "/lidar_image", sensor_msgs.msg.Image, callback=self.callback)
        self._switch = True
        self._brdg = CvBridge()

    def callback(self, msg):
        self.image = self._brdg.imgmsg_to_cv2(msg,"32FC1")
        self._switch = False
    
    def block(self):
        self._switch = True
        while self._switch:
            time.sleep(0.2)

def save_sample(image, label, path):
    image = torch.tensor(image)
    label = torch.tensor(label)
    with open(path, "wb+") as f:
        pickle.dump((image,label),f)


def main():
    rospy.init_node("dataset_collecter")

    world_name = rospy.get_param("world_name")
    samples_per_tile = int(rospy.get_param("n_samples_per_tile"))
    tree_path = os.path.join("/home/lorenzo/catkin_data/worlds",
                        world_name, "tree.pickle")
    base_dataset_path = "/home/lorenzo/catkin_data/datasets/2d_gallery_detection"
    dataset_name = world_name
    dataset_path = os.path.join(base_dataset_path, dataset_name)
    if os.path.isdir(dataset_path):
        pass
    else:
        os.mkdir(dataset_path)
    with open(tree_path, "rb") as f:
        tree = pickle.load(f)
    assert isinstance(tree, TileTree)
    mover = RobotMover()
    storage = ImageStorage()
    i = 0
    with open(os.path.join(dataset_path,"info.txt"),"w") as info_file:
        for tile in tree.non_blocking_tiles:
            for n in range(samples_per_tile):
                p,o,l = random_label_in_tile(tile,2)
                info_file.write(f"{p}-{o}\n")
                mover.send_position(p, o)
                storage.block()
                storage.block()
                save_path = os.path.join(dataset_path,str(i)+".pickle")
                save_sample(storage.image, l, save_path)
                i+=1
                print(i)
        


if __name__ == "__main__":
    main()
