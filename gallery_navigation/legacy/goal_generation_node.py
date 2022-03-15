#!/bin/python3
from math import sqrt
import time

import rospy
import std_msgs.msg as ros_std_msg
import geometry_msgs.msg as ros_geom_msg
from threading import Thread
import numpy as np
from datetime import datetime
import tf
import actionlib
import move_base_msgs.msg as ros_mb_msg
import gazebo_msgs.srv as ros_gz_srv
import copy
# --------------------
# GENERAL NOTES
# - In this script, a semantic navigation system is implemented for a robot in
#   an underground environment. The expected inputs for this node are:
#       - /gallery_angles: These are obtained by a different node. List of
#         angles wrt the robot in which direction a gallery is found. This also
#         includes the current gallery aka, the front and the back.
#       - /tile_type: This topic should continually publish whether the robot is
#         in an intersection, a rect, a curve etc...
#
#
# --------------------


def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]


class Nodo:
    """La clase nodo está pensada para almacenar
    los distintos nodos del mapa, así como las
    relaciones entre las diferentes galerías que
    a las que está conectado."""

    def __init__(self, map, n_galleries):
        self.n_galleries = n_galleries


class Galeria:
    def __init__(self, map):
        # A gallery can only connect two nodes
        self.map = map
        self.nodes = [None, None]


class Mapa:
    """La clase mapa está pensada para almacenar
    una serie de nodos conectados por galerías,
    y conservar las relaciones entre estos nodos"""

    def __init__(self) -> None:
        self.nodes = list()
        self.galleries = list()


class GoalGenerationNode:
    def __init__(self, goal_time_interval=1, goal_distance=3):
        self.n_instruction = 0
        self.goal_time_interval = goal_time_interval
        self.goal_distance = goal_distance
        self.reset_quadrants()
        self.time = datetime.now()
        self.seq = 0
        self.can_send_goal = True

        rospy.init_node(self.__class__.__name__)

        self.instructions = rospy.get_param("/instructions")
        self.final_blocker_position_x = rospy.get_param(
            "/final_blocker_position_x")
        self.final_blocker_position_y = rospy.get_param(
            "/final_blocker_position_y")

        self.listener = tf.TransformListener()
        self.tf_transformer = tf.TransformerROS()

        self.tile_type_subscriber = rospy.Subscriber(
            "/environment_label", ros_std_msg.String, callback=self.tile_type_callback)
        self.gallery_subscriber = rospy.Subscriber(
            "/gallery_detection_vector", ros_std_msg.Float32MultiArray, self.gallery_detection_callback)
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", ros_mb_msg.MoveBaseAction)

        self.get_robot_pose_client = rospy.ServiceProxy(
            "/gazebo/get_model_state", ros_gz_srv.GetModelState)

        if not self.move_base_client.wait_for_server(rospy.Duration(5)):
            rospy.logerr("THERE IS NO MOVE BASE NODE")

        self.first_callback = False
        while not self.first_callback:
            rospy.sleep(0.5)

        self.run_thread = Thread(target=self.run)

        self.already_chosen_exit = False

        self.run_thread.start()

        self.run_thread.join()

    def reset_quadrants(self):
        self.quadrants = {"front": [],
                          "left": [],
                          "right": [],
                          "back": []
                          }

    def tile_type_callback(self, msg: ros_std_msg.String):
        self.tile_type = msg.data

    def array_position_to_angle(self, array_position):
        return 180 - array_position

    def get_galleries_from_vector(self, vector):
        self.vector = vector
        self.filtered = np.zeros(360)
        for i in range(360):
            to_check = vector[i]
            self.filtered[i] = to_check
            for j in range(31):
                subsection_index = ((-15 + j) + i) % 356
                if vector[subsection_index] > to_check:
                    self.filtered[i] = 0

        max_peak = np.max(self.filtered)
        galleries_indices = np.nonzero(self.filtered > max_peak * 0.6)
        galleries_angles = []
        for index in galleries_indices:
            galleries_angles.append(
                self.array_position_to_angle(index)/180.0 * np.math.pi)

        return np.array(galleries_angles)[0]

    def gallery_detection_callback(self, msg: ros_std_msg.Float32MultiArray):
        """ This function should take the input from the neural network, and 
            translate it to quadrants"""

        data = np.array(msg.data)

        angles_of_galleries = self.get_galleries_from_vector(data)

        quadrants = {"front": [],
                     "left": [],
                     "right": [],
                     "back": []
                     }
        for angle in angles_of_galleries:
            if angle > -np.math.pi/4 and angle < np.math.pi/4:
                quadrants["front"].append(angle)
            elif angle > -np.math.pi*3/4 and angle < -np.math.pi/4:
                quadrants["right"].append(angle)
            elif angle > np.math.pi/4 and angle < np.math.pi*3/4:
                quadrants["left"].append(angle)
            elif angle > np.math.pi*3/4 or angle < -np.math.pi*3/4:
                quadrants["back"].append(angle)

        if self.quadrants["back"].__len__() > 0 and self.tile_type == "block":
            if self.n_instruction == self.instructions.__len__():
                robot_pose = self.get_robot_pose_client.call(
                    ros_gz_srv.GetModelStateRequest("/", ""))
                assert(isinstance(robot_pose, ros_gz_srv.GetModelStateResponse))
                rx = robot_pose.pose.position.x
                ry = robot_pose.pose.position.y
                if ((self.final_blocker_position_x-rx)**2+(self.final_blocker_position_y-ry)**2)**0.5 < 10:
                    rospy.set_param("/test_status", "success")
                    return
                else:
                    self.send_fail_message()
                    return

        if not self.can_send_goal:
            return

        self.quadrants = copy.deepcopy(quadrants)

        n = 0
        for k in self.quadrants.keys():
            if self.quadrants[k].__len__() > 0:
                n += 1

        self.in_intersection = n > 2
        self.first_callback = True
        self.generate_and_send_goal()

    def is_there_exit(self, quadrant: str):
        return len(self.quadrants[quadrant]) > 0

    def goal_from_angle(self, angle, distance=0):
        goal = ros_geom_msg.PoseStamped()
        goal.header.frame_id = "base_link"
        goal.header.seq = self.seq
        goal.header.stamp = rospy.Time.now()
        self.seq += 1
        quaternion = euler_to_quaternion(angle, 0, 0)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        if distance == 0:
            D = self.goal_distance
        else:
            D = distance
        goal.pose.position.x = D * np.math.cos(angle)
        goal.pose.position.y = D * np.math.sin(angle)
        goal.pose.position.z = 0

        # Transform the goal to the map frame
        t = self.listener.getLatestCommonTime("odom", "base_link")
        self.tf_transformer._buffer = self.listener._buffer
        goal.header.stamp = t
        goal = self.tf_transformer.transformPose("odom", goal)

        goal_msg = ros_mb_msg.MoveBaseGoal()
        goal_msg.target_pose = goal

        return goal_msg

    def send_fail_message(self):
        rospy.set_param("/test_status", "fail")
        return

    def generate_and_send_goal(self):

        self.can_send_goal = False
        goal_msg = self.goal_from_angle(0, 0.0)
        if self.in_intersection:
            print("IN INTERSECTION")
            if self.already_chosen_exit:
                print("ALREADY CHOSEN EXIT, GOING FRONT")
                if self.quadrants["front"].__len__() == 0:
                    self.send_fail_message()
                    return
                else:
                    goal_msg = self.goal_from_angle(self.quadrants["front"][0])
            else:
                if self.n_instruction == self.instructions.__len__():
                    self.send_fail_message()
                    return
                inst = self.instructions[self.n_instruction]
                print("NOT CHOSEN EXIT, GOING {}".format(inst))
                if self.quadrants[inst].__len__() == 0:
                    self.send_fail_message()
                    return
                goal_msg = self.goal_from_angle(
                    self.quadrants[inst][0])
                self.already_chosen_exit = True
                self.n_instruction += 1
        else:
            self.already_chosen_exit = False
            if self.quadrants["front"].__len__() > 0:
                goal_msg = self.goal_from_angle(self.quadrants["front"][0])
            elif self.quadrants["right"].__len__() > 0:
                goal_msg = self.goal_from_angle(self.quadrants["right"][0])
            elif self.quadrants["left"].__len__() > 0:
                goal_msg = self.goal_from_angle(self.quadrants["left"][0])
            elif self.quadrants["back"].__len__() > 0 and self.tile_type == "block":
                if self.n_instruction == self.instructions.__len__():
                    robot_pose = self.get_robot_pose_client.call(
                        ros_gz_srv.GetModelStateRequest("/", ""))
                    assert(isinstance(robot_pose, ros_gz_srv.GetModelStateResponse))
                    rx = robot_pose.pose.position.x
                    ry = robot_pose.pose.position.y
                    if ((self.final_blocker_position_x-rx)**2+(self.final_blocker_position_y-ry)**2)**0.5 < 10:
                        rospy.set_param("/test_status", "success")
                        return
                    else:
                        self.send_fail_message()
                        return
                else:
                    self.send_fail_message()
                    return

            else:
                goal_msg = self.goal_from_angle(0, 0.5)

        self.move_base_client.send_goal(
            goal_msg, done_cb=self.done_cb, active_cb=self.active_cb, feedback_cb=self.feedback_cb)

    def done_cb(self, msg, msg2):
        self.can_send_goal = True

    def active_cb(self):
        pass

    def feedback_cb(self, msg):
        pass

    def run(self):
        # while not rospy.is_shutdown():
        #     self.generate_and_send_goal()
        #     rospy.sleep(self.goal_time_interval)

        self.generate_and_send_goal()


if __name__ == "__main__":
    goal_generation_node = GoalGenerationNode()
    rospy.spin()
