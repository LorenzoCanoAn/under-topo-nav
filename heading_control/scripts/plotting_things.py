#!/usr/bin/python3

import time
from typing import final
import rospy
from rospy.names import valid_name
from rospy.topics import Publisher
import std_msgs.msg as ros_std_msg
import geometry_msgs.msg as ros_gmt_msg
import sensor_msgs.msg as ros_sns_msg
import numpy as np
import matplotlib.pyplot as plt
from functions import min_distance, filter_and_inflate_ranges, filter_vector_with_intermediates
from gallery_tracker import GalleryTracker
from params import *
# speed msg type geometry_msgs/Twist


cmd_vel_publisher = Publisher("/cmd_vel", ros_gmt_msg.Twist)
fig = plt.figure(figsize=(5,5))
axis = plt.subplot(111, polar=True)
NN_ANGLES = np.zeros(360)
class SaveData:
    def __init__(self):
        self.data = 0

N_PLOT = SaveData()
for i in range(360):
    NN_ANGLES[i] = (180 - i)/180.0 * np.math.pi


class LaserData:
    ranges = None
    angles = None
    angle_increment = None
    filtered = None


class ObjectiveAngle:
    data = 0


laser_data = LaserData()
gallery_tracker = GalleryTracker()


def gallery_vector_callback(msg: ros_std_msg.Float32MultiArray):
    if isinstance(laser_data.angles, type(None)):
        print("not updeated")
        return

    filtered_vector, gallery_angles = filter_vector_with_intermediates(msg.data)
    gallery_angles = np.array(gallery_angles)
    gallery_tracker.new_angles(gallery_angles)

    # objective_angle = gallery_tracker.get_angle()
    objective_angle = gallery_angles[1]
    if objective_angle == None:          
        rospy.set_param("/test_status","finished")
        time.sleep(0.1)
        plt.close()
        vel_msg = ros_gmt_msg.Twist()
        cmd_vel_publisher.publish(vel_msg)
        return
    else:
        angular_distance_weight = np.zeros(laser_data.angles.__len__())
        for n, i in enumerate(laser_data.angles):
            angular_distance_weight[n] = np.math.pi - \
                min_distance(i, objective_angle)
        angular_distance_weight /= np.max(angular_distance_weight)
        total_value_vector = np.multiply(
            angular_distance_weight, laser_data.filtered)
        max_idx = np.argmax(total_value_vector)
        final_angle = laser_data.angles[max_idx]

        j = 0
        suma = laser_data.ranges[0]
        min_dist = 10
        for _ in np.arange(0, 10/180*np.math.pi, laser_data.angle_increment):
            j += 1
            min_dist = min(min_dist, laser_data.ranges[j])
            min_dist = min(min_dist,laser_data.ranges[-j])

        if N_PLOT.data % 1 == 0:
            plt.cla()
            ax = plt.gca()

            #plt.scatter(laser_data.angles, angular_distance_weight,color="#000D6B",s=5,marker="x")
            plt.plot(NN_ANGLES, np.array(msg.data), color="#FF0000")
            #filtered_2_vector = filtered_vector * (filtered_vector > 0.3 * np.max(filtered_vector))
            #plt.plot(NN_ANGLES, filtered_vector, color="#000D6B")
            #plt.plot(NN_ANGLES, filtered_2_vector, color="#9C19E0")
            plt.gca().set_rlim(0, 1)
            plt.yticks([0.5,1],fontsize=15)

            if 0:
                plt.scatter(laser_data.angles, laser_data.ranges,color="#9C19E0",s=20,marker="o")
                plt.scatter(laser_data.angles, laser_data.filtered,color="#49FF00",s=5,marker="x")
                plt.gca().set_rlim(0, 12)
                plt.yticks([3,6,9,12],fontsize=15)

            if 0:
                plt.scatter(laser_data.angles, total_value_vector,color="#1DB9C3",s=5,marker="x")
                plt.scatter(final_angle, total_value_vector[max_idx],color="#FF0000",s=100,marker="o")
                plt.gca().set_rlim(0, 6)
                plt.yticks([3,6],fontsize=15)


            ax.set_rlabel_position(110)
            plt.xticks(fontsize=15)
            plt.gca().set_theta_zero_location("N")
            label_position=ax.get_rlabel_position()
            ax.text(np.radians(label_position-18),ax.get_rmax()/4*3.,"",rotation=110+90+180,ha='center',va='center',fontsize=15)
            plt.draw()

        N_PLOT.data +=1
        vel_msg = ros_gmt_msg.Twist()

        vel_msg.angular.z = final_angle / abs(final_angle) * min([abs(final_angle), MAX_ANG_VEL])


        vel_msg.linear.x = (MAX_VEL * (min(min_dist, 2)/2)) - abs(vel_msg.angular.z)/MAX_ANG_VEL*MAX_VEL

        vel_msg.linear.x = max((vel_msg.linear.x, 0))
        print("v:{:.2f}\tw:{:.2f}".format(vel_msg.linear.x, vel_msg.angular.z))
        #cmd_vel_publisher.publish(vel_msg)


def laser_scan_callback(msg: ros_sns_msg.LaserScan):
    scan_angles = np.arange(start=msg.angle_min,
                            stop=msg.angle_max, step=msg.angle_increment)
    scan_ranges = np.array(msg.ranges).flatten()

    laser_data.angles = scan_angles
    laser_data.ranges = scan_ranges
    laser_data.angle_increment = msg.angle_increment
    laser_data.filtered = filter_and_inflate_ranges(
        scan_ranges, laser_data.angle_increment)


def main():
    rospy.init_node("heading_control")
    # instructions = rospy.get_param("/topological_instructions",default=None)
    # while instructions == None:
    #     instructions = rospy.get_param("/topological_instructions",default=None)
    #     pass

    #gallery_tracker.set_instructions(instructions)

    rospy.Subscriber("/gallery_detection_vector",
                     ros_std_msg.Float32MultiArray, callback=gallery_vector_callback)
    rospy.Subscriber("/scan", ros_sns_msg.LaserScan,
                     callback=laser_scan_callback)
    plt.show()
    # rospy.spin()


if __name__ == "__main__":
    main()
