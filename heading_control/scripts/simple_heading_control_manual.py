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
from functions import min_distance, filter_and_inflate_ranges, filter_vector
from gallery_tracker import GalleryTracker
from params import *
# speed msg type geometry_msgs/Twist


cmd_vel_publisher = Publisher("/cmd_vel", ros_gmt_msg.Twist)
fig = plt.figure(figsize=(6,6))
axis1 = plt.subplot(111, polar=True)
# axis2 = plt.subplot(132, polar=True)
# axis3 = plt.subplot(133, polar=True)


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
class Data:
    def __init__(self):
        self.data = None

first = Data()
first.data = True
n_errors = Data()
n_errors.data = 0

def gallery_vector_callback(msg: ros_std_msg.Float32MultiArray):
    if isinstance(laser_data.angles, type(None)):
        print("not updeated")
        return
    

    gallery_angles = np.array(filter_vector(msg.data))
    gallery_tracker.new_angles(gallery_angles)
    objective_angle = gallery_tracker.get_angle()

    if objective_angle == None: 
        if n_errors.data == 3:
            rospy.set_param("/test_status","finished")
            time.sleep(0.1)
            plt.close()
            vel_msg = ros_gmt_msg.Twist()
            cmd_vel_publisher.publish(vel_msg)
            return
        else:
            n_errors.data +=1
            return
    else:
        n_errors.data = 0
        angle_value_vector = np.zeros(laser_data.angles.__len__())
        for n, i in enumerate(laser_data.angles):
            angle_value_vector[n] = np.math.pi - \
                min_distance(i, objective_angle)
        angle_value_vector /= np.max(angle_value_vector)
        total_value_vector = np.multiply(
            angle_value_vector, laser_data.filtered)
        max_idx = np.argmax(total_value_vector)
        final_angle = laser_data.angles[max_idx]

        j = 0
        suma = laser_data.ranges[0]
        min_dist = 10
        for _ in np.arange(0, 10/180*np.math.pi, laser_data.angle_increment):
            j += 1
            min_dist = min(min_dist, laser_data.ranges[j])
            min_dist = min(min_dist,laser_data.ranges[-j])
        fontsize = 14
        if N_PLOT.data % 3 == 0:
            # AXIS 1
            plt.sca(axis1)
            plt.cla()
            plt.plot(NN_ANGLES, np.array(msg.data), color="#FF0000")
            plt.scatter(laser_data.angles, angle_value_vector, color="#000D6B")
            plt.scatter(laser_data.angles, laser_data.ranges,color="#9C19E0")
            plt.scatter(laser_data.angles, laser_data.filtered,color="#49FF00")
            plt.scatter(laser_data.angles, total_value_vector,color="#1DB9C3")
            plt.scatter(final_angle, total_value_vector[max_idx],color="#FF0000",s=100)
            plt.gca().set_theta_zero_location("N")
            plt.xticks(size=fontsize)
            plt.gca().set_rlim([0,5])
            plt.yticks([2.5,5],size=fontsize)
            try:
                n = gallery_tracker.n_objective - 1
                ins = gallery_tracker.topological_instructions[n]
                plt.gca().set_title(f"Instruction number {n} is {ins}",pad=20, size=fontsize)
            except:
                
                plt.gca().set_title(f"No instructions yet",pad=20, size=fontsize)

            # AXIS 2    
            #plt.sca(axis2)
            #plt.cla()
            #plt.scatter(laser_data.angles, laser_data.ranges,color="#9C19E0")
            #plt.scatter(laser_data.angles, laser_data.filtered,color="#49FF00")
            #plt.gca().set_theta_zero_location("N")
            #plt.yticks([5,10],size=fontsize)
            #plt.xticks(size=fontsize)
            #plt.gca().set_rlim([0,10])
            #plt.gca().set_title("Laser-scan and obstacle weight",pad=20, size=fontsize)

            # AXIS 3
            #plt.sca(axis3)
            #plt.cla()
            #plt.scatter(laser_data.angles, total_value_vector,color="#1DB9C3")
            #plt.scatter(laser_data.angles, angle_value_vector, color="#000D6B")
            #plt.scatter(laser_data.angles, laser_data.filtered,color="#49FF00")
            #plt.scatter(final_angle, total_value_vector[max_idx],color="#FF0000",s=100)
            #plt.gca().set_theta_zero_location("N")
            #plt.yticks([5,10],size=fontsize)
            #plt.xticks(size=fontsize)
            #plt.gca().set_rlim([0,10])
            #plt.gca().set_title("Direction weight, obstacle weight, \n total value and selected angle",pad=20, size=fontsize)

            

            plt.draw()
        
            

            vel_msg = ros_gmt_msg.Twist()

            vel_msg.angular.z = final_angle / \
                abs(final_angle) * min([abs(final_angle), MAX_ANG_VEL])
            vel_msg.linear.x = (MAX_VEL * (min(min_dist, 2)/2)) - \
                abs(vel_msg.angular.z)/MAX_ANG_VEL*MAX_VEL

            vel_msg.linear.x = max((vel_msg.linear.x, 0))

            if first.data < 100:
                first.data += 1
                print(first.data)
            else:
                cmd_vel_publisher.publish(vel_msg)
                
        N_PLOT.data +=1


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
    instructions = [1, 1, 1, 3, 1, 1, -1, -1, 1, 2, 1, 2, 1, 1, 1]

    gallery_tracker.set_instructions(instructions)

    rospy.Subscriber("/gallery_detection_vector",
                     ros_std_msg.Float32MultiArray, callback=gallery_vector_callback)
    rospy.Subscriber("/scan", ros_sns_msg.LaserScan,
                     callback=laser_scan_callback)
    time.sleep(1)
    plt.show()
    # rospy.spin()


if __name__ == "__main__":
    main()
