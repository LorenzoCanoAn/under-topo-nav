#!/bin/python3
import rospy

import matplotlib.pyplot as plt
import std_msgs.msg as ros_std_msg
import sensor_msgs.msg as ros_sns_msg
from image_tools import ImageTools
from sensor_msgs.msg import Image
import scipy.ndimage
import numpy as np

RAD = 30
DISTANCE = 3


def filter_ranges(i_ranges, i_angles):
    i_ranges = scipy.ndimage.median_filter(i_ranges,size=10, mode="wrap")
    i_ranges[np.where(i_ranges > 10)] = 10
    o_ranges = scipy.ndimage.uniform_filter(i_ranges,size = 50 , mode="wrap")
    o_ranges_f = o_ranges


    return o_ranges, o_ranges_f, i_angles

class every_other_plot:

    def __init__(self, axis_image, axis_output, axis_scan):
        self.axis_image = axis_image
        self.axis_output = axis_output
        self.axis_scan = axis_scan
        self.switch_nn_out = 0
        self.switch_image = 0
        self.image_tools = ImageTools()
        self.nn_out_theta = np.arange(180, -180, -1) / 180 * np.math.pi
        
        self.lock = True

    def filter_vector(self, vector):
        
        filtered = np.zeros(360)
        for i in range(360):
            to_check = vector[i]
            filtered[i] = to_check
            a = 40
            for j in range(a):
                index_inside_subsection = ((-int(a/2) + j) + i) % 356
                if vector[index_inside_subsection] > to_check:
                    filtered[i] = 0
        return filtered

    def gallery_detection_callback(self, msg: ros_std_msg.Float32MultiArray):
        """ This function should take the input from the neural network, and 
            translate it to quadrants"""
        if self.lock:
            self.lock = False
            self.switch_nn_out = (self.switch_nn_out + 1) % 4
            if not self.switch_nn_out:
                data = np.array(msg.data)
                filtered = self.filter_vector(data)
                plt.sca(self.axis_output)
                plt.cla()
                plt.gca().set_rlim(0, 1)
                plt.plot(self.nn_out_theta, data)
                plt.plot(self.nn_out_theta, filtered, "r")
                plt.gca().set_theta_zero_location("N")

                plt.sca(self.axis_scan)
                plt.cla()
                plt.plot(self.scan_angles, self.scan_ranges)
                plt.plot(self.scan_angles, self.scan_ranges_f)
                plt.gca().set_theta_zero_location("N")
                


                plt.draw()
            self.lock = True

    def laserscan_callback(self, msg: ros_sns_msg.LaserScan):
        scan_angles = np.arange(start=msg.angle_min, stop=msg.angle_max, step=msg.angle_increment)
        scan_ranges = np.array(msg.ranges).flatten()
        self.scan_ranges, self.scan_ranges_f, self.scan_angles = filter_ranges(np.array(msg.ranges).flatten(), scan_angles)


    def image_callback(self, msg: Image):
        """ This function should take the input from the neural network, and 
            translate it to quadrants"""
        if self.lock:
            self.lock = False
            self.switch_image = (self.switch_image + 1) % 4
            if not self.switch_image:
                
                plt.sca(self.axis_image)
                plt.cla()
                image = self.image_tools.convert_ros_msg_to_cv2(msg,image_encoding="mono8")
                plt.imshow(image)
                plt.draw()
            self.lock = True

def main():
    rospy.init_node("ahhh")
    fig = plt.figure()
    axis_output = plt.subplot(2,2,2,polar=True)
    axis_scan = plt.subplot(2,2,1,polar=True)
    axis_image = plt.subplot(2,1,2)


    plot_handler = every_other_plot(axis_image, axis_output, axis_scan)

    gallery_subscriber = rospy.Subscriber(
        "/gallery_detection_vector", ros_std_msg.Float32MultiArray, plot_handler.gallery_detection_callback)
    image_subscriber = rospy.Subscriber(
        "/laserscan_image", Image, plot_handler.image_callback)
    laserscan_subscriber = rospy.Subscriber("/scan", ros_sns_msg.LaserScan,callback=plot_handler.laserscan_callback)
    
    plt.show()
    rospy.spin()


if __name__ == "__main__":
    main()
