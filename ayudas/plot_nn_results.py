#!/bin/python3
import rospy

import move_base_msgs.msg as ros_mb_msg
import actionlib
import matplotlib.pyplot as plt
import std_msgs.msg as ros_std_msg
import numpy as np


class every_other_plot:

    def __init__(self):
        self.switch = 0

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
        self.switch = (self.switch + 1) % 2
        if not self.switch:
            data = np.array(msg.data)
            filtered = self.filter_vector(data)

            plt.cla()
            plt.gca().set_rlim(0, 1)
            theta = np.arange(180, -180, -1) / 180 * np.math.pi
            plt.plot(theta, data)
            plt.plot(theta, filtered, "r")
            plt.gca().set_theta_zero_location("N")

            plt.draw()


def main():
    rospy.init_node("ahhh")
    fig = plt.figure()
    plt.subplot(1,1,1,polar=True)


    plot_handler = every_other_plot()

    gallery_subscriber = rospy.Subscriber(
        "/gallery_detection_vector", ros_std_msg.Float32MultiArray, plot_handler.gallery_detection_callback)
    plt.show()
    rospy.spin()


if __name__ == "__main__":
    main()
